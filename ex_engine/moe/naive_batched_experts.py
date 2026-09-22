"""
naive_batched_experts.py — MoE expert computation for BI-V100

Ported from:
  upstream_ref/ds_vllm/vllm/model_executor/layers/fused_moe/experts/fused_batched_moe.py
  class NaiveBatchedExperts.apply()

Key design from upstream:
  - w1[expert].transpose(0, 1) is a VIEW (zero copy)
  - @ operator lets cublas pass transB=CUBLAS_OP_T internally
  - No physical transpose, no gather of full weight matrices
  - Per-expert loop with early exit on num_tokens == 0

Adaptations for BI-V100:
  - Removed modular_kernel / FusedMoEExpertsModular base class
  - Removed triton kernels (BatchedTritonExperts)
  - Removed quantization (FP8, INT8, INT4)
  - Removed workspace_shapes / MoEActivation enum dependency
  - activation uses F.silu directly (torch.ops._C.silu_and_mul not available)
  - Standalone function, not a class — called from qwen3_5.py
"""

import torch
import torch.nn.functional as F
from typing import Optional


def _resize_cache(x: torch.Tensor, v: tuple) -> torch.Tensor:
    """Shrink tensor and reshape. From ds_vllm utils.py."""
    from math import prod
    assert prod(v) <= x.numel(), f"{v} ({prod(v)}) <= {x.shape} ({x.numel()})"
    return x.flatten()[:prod(v)].view(*v)


def naive_batched_moe_forward(
    hidden_states: torch.Tensor,     # (T, H) or (1, H) for decode
    w13: torch.Tensor,               # (E, 2*I, H) — gate+up fused weights
    w2: torch.Tensor,                # (E, H, I)   — down weights
    topk_ids: torch.Tensor,          # (T, top_k) — selected expert ids
    topk_weights: torch.Tensor,      # (T, top_k) — routing weights
    act_fn: Optional[object] = None, # SiluAndMul instance or None
) -> torch.Tensor:
    """
    MoE expert forward — ported from NaiveBatchedExperts.apply().

    For each selected expert:
      1. FC1: input @ w1[expert].transpose(0, 1)  — view transpose, cublas transB
      2. Activation: silu_and_mul (gated)
      3. FC2: act @ w2[expert].transpose(0, 1)

    Source: upstream_ref/ds_vllm/.../experts/fused_batched_moe.py lines 611-647
    """
    T = hidden_states.shape[0]
    H = hidden_states.shape[1]
    I = w2.shape[2]          # intermediate size (per partition)
    top_k = topk_ids.shape[1]

    # Output accumulator
    out = torch.zeros(T, H, dtype=hidden_states.dtype, device=hidden_states.device)

    if T == 1:
        # === Decode path (single token) ===
        # From NaiveBatchedExperts.apply():
        #   input = hidden_states[expert, :num, :] @ w1[expert].transpose(0, 1)
        #
        # For decode, each expert sees exactly 1 token.
        # expert ids are in topk_ids[0] (shape: top_k,)
        eids = topk_ids[0].tolist()   # (top_k,) → CPU list, ONE sync
        ws = topk_weights[0]          # (top_k,) stays on GPU

        for i in range(top_k):
            eid = eids[i]

            # FC1: (1, H) @ (H, 2*I) → (1, 2*I)
            # w13[eid] is (2*I, H), .transpose(0, 1) is (H, 2*I) — VIEW, zero copy
            # @ lets cublas use transB=CUBLAS_OP_T
            gate_up = hidden_states @ w13[eid].transpose(0, 1)  # (1, 2*I)

            # Activation: silu_and_mul
            # From upstream apply_moe_activation():
            #   gate = input[..., :d], up = input[..., d:]
            #   output = F.silu(gate) * up
            if act_fn is not None:
                act = act_fn(gate_up)     # SiluAndMul: (1, 2*I) → (1, I)
            else:
                gate = gate_up[..., :I]
                up = gate_up[..., I:]
                act = F.silu(gate) * up   # (1, I)

            # FC2: (1, I) @ (I, H) → (1, H)
            # w2[eid] is (H, I), .transpose(0, 1) is (I, H) — VIEW, zero copy
            expert_out = act @ w2[eid].transpose(0, 1)  # (1, H)

            # Weighted accumulate
            out += ws[i] * expert_out

    else:
        # === Prefill path (multiple tokens) ===
        # Group tokens by expert, then batch-process each expert.
        # From NaiveBatchedExperts.apply() — the for-expert loop.
        flat_eids = topk_ids.reshape(-1)                 # (T * top_k,)
        flat_weights = topk_weights.reshape(-1)           # (T * top_k,)
        flat_token_ids = torch.arange(
            T, device=hidden_states.device
        ).repeat_interleave(top_k)                        # (T * top_k,)

        num_experts = w13.shape[0]
        for expert in range(num_experts):
            mask = (flat_eids == expert)
            if not mask.any():
                continue

            token_ids = flat_token_ids[mask]               # tokens assigned to this expert
            weights = flat_weights[mask]                   # their routing weights
            expert_input = hidden_states[token_ids]        # (num, H)

            # FC1: (num, H) @ (H, 2*I) → (num, 2*I)
            gate_up = expert_input @ w13[expert].transpose(0, 1)

            # Activation
            if act_fn is not None:
                act = act_fn(gate_up)
            else:
                gate = gate_up[..., :I]
                up = gate_up[..., I:]
                act = F.silu(gate) * up

            # FC2: (num, I) @ (I, H) → (num, H)
            expert_out = act @ w2[expert].transpose(0, 1)

            # Weighted scatter-add back
            out.index_add_(0, token_ids, expert_out * weights.unsqueeze(1))

    return out
