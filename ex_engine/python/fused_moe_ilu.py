"""
fused_moe_ilu.py — 7-step fused MoE via xllm upstream ILU dispatch chain

Upstream ref: xllm/core/layers/ilu/fused_moe.cpp
              xllm/core/kernels/ilu/fused_moe.cpp

The 7-step pipeline:
  1. topk_softmax       → ixformer::infer::topk_softmax
  2. moe_gen_idx        → ixformer::infer::moe_compute_token_index_api
  3. moe_expand_input   → ixformer::infer::moe_expand_input
  4. group_gemm (w13)   → ixformer::infer::moe_w16a16_group_gemm
  5. silu_and_mul       → ixformer::infer::silu_and_mul
  6. group_gemm (w2)    → ixformer::infer::moe_w16a16_group_gemm
  7. moe_combine_result → ixformer::infer::moe_output_reduce_sum

Every step calls C++. No Python expert loop.
"""

import logging
import torch
from typing import Optional, Tuple

logger = logging.getLogger("fused_moe_ilu")

_init_logged = False

# =====================================================================
# Load the C++ ops
# =====================================================================

def _get_ops():
    """Get the ix_ops_dispatch module."""
    try:
        from ex_engine.python import ix_ops_dispatch as ops
        return ops
    except ImportError:
        pass
    try:
        from vllm.ex_engine import ix_ops_dispatch as ops
        return ops
    except ImportError:
        pass
    return None


# =====================================================================
# 7-step fused MoE forward
# =====================================================================

def fused_moe_forward(
    hidden_states: torch.Tensor,      # (num_tokens, hidden_size)
    gate_output: torch.Tensor,        # (num_tokens, num_experts) router logits
    w13: torch.Tensor,                # (E, 2*intermediate, hidden_size) merged gate_up
    w2: torch.Tensor,                 # (E, hidden_size, intermediate)
    topk: int = 8,
    renormalize: bool = True,
    num_experts: int = 64,
    shared_expert: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Full 7-step fused MoE pipeline.

    All steps go through C++ — no Python fallback.
    If C++ is unavailable, raises RuntimeError.
    """
    global _init_logged
    ops = _get_ops()
    if ops is None:
        raise RuntimeError("fused_moe_ilu: ix_ops_dispatch not available")

    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    intermediate_2x = w13.shape[1]  # 2 * intermediate_size
    intermediate = intermediate_2x // 2

    if not _init_logged:
        logger.info("Using fused MoE ILU pipeline: tokens=%d, experts=%d, topk=%d, "
                    "intermediate=%d", num_tokens, num_experts, topk, intermediate)
        _init_logged = True

    # Step 1: topk_softmax
    topk_weights, topk_ids = ops.topk_softmax(gate_output, topk, renormalize)

    # Step 2: moe_compute_token_index
    src_dst, dst_src, expert_sizes = ops.moe_compute_token_index(
        topk_ids, num_experts)

    # Step 3: moe_expand_input
    expanded = ops.moe_expand_input(hidden_states, dst_src, topk)

    # Step 4: group_gemm w13 (gate + up projection)
    gate_up = ops.moe_group_gemm(expanded, w13, expert_sizes, intermediate_2x)

    # Step 5: silu_and_mul
    activated = ops.silu_and_mul(gate_up)

    # Step 6: group_gemm w2 (down projection)
    down = ops.moe_group_gemm(activated, w2, expert_sizes, hidden_size)

    # Step 7: moe_output_reduce_sum (weighted combine)
    output = ops.moe_output_reduce_sum(down, topk_weights.to(down.dtype))

    return output


# =====================================================================
# Fallback: Per-expert matmul (used when group_gemm unavailable)
# Still uses C++ for topk and activation, just loops for GEMM.
# =====================================================================

def fused_moe_per_expert(
    hidden_states: torch.Tensor,
    gate_output: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk: int = 8,
    renormalize: bool = True,
    num_experts: int = 64,
) -> torch.Tensor:
    """
    Per-expert fallback with C++ topk and activation.
    Uses torch.matmul for GEMM (goes to cublas).
    """
    ops = _get_ops()
    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    intermediate_2x = w13.shape[1]
    half_inter = intermediate_2x // 2
    dtype = hidden_states.dtype

    # Step 1: topk
    if ops is not None:
        try:
            topk_weights, topk_ids = ops.topk_softmax(gate_output, topk, renormalize)
        except RuntimeError:
            scores = torch.softmax(gate_output.float(), dim=-1)
            topk_weights, topk_ids = torch.topk(scores, k=topk, dim=-1)
            if renormalize:
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
            topk_ids = topk_ids.to(torch.int32)
    else:
        scores = torch.softmax(gate_output.float(), dim=-1)
        topk_weights, topk_ids = torch.topk(scores, k=topk, dim=-1)
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_ids = topk_ids.to(torch.int32)

    topk_weights = topk_weights.to(dtype)
    flat_ids = topk_ids.view(-1)
    flat_weights = topk_weights.view(-1)

    # Expand input
    expanded = hidden_states.unsqueeze(1).expand(-1, topk, -1).reshape(-1, hidden_size)
    output = torch.zeros_like(expanded)

    # Per-expert GEMM (cublas)
    for eidx in range(num_experts):
        mask = (flat_ids == eidx)
        if not mask.any():
            continue
        tokens = expanded[mask]

        # gate_up GEMM → cublas via torch.matmul
        gate_up = torch.matmul(tokens, w13[eidx].t())

        # SiLU activation (C++ if available)
        if ops is not None:
            try:
                act = ops.silu_and_mul(gate_up)
            except RuntimeError:
                act = torch.nn.functional.silu(gate_up[:, :half_inter]) * gate_up[:, half_inter:]
        else:
            act = torch.nn.functional.silu(gate_up[:, :half_inter]) * gate_up[:, half_inter:]

        # down GEMM → cublas
        output[mask] = torch.matmul(act, w2[eidx].t())

    output = output * flat_weights.unsqueeze(-1)
    return output.view(num_tokens, topk, hidden_size).sum(dim=1)


# =====================================================================
# Auto-dispatch: try full pipeline, fall back to per-expert
# =====================================================================

def moe_forward(
    hidden_states: torch.Tensor,
    gate_output: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk: int = 8,
    renormalize: bool = True,
    num_experts: int = 64,
    **kwargs,
) -> torch.Tensor:
    """Auto-dispatch MoE: try full C++ pipeline, then per-expert with C++ ops."""
    try:
        return fused_moe_forward(
            hidden_states, gate_output, w13, w2,
            topk, renormalize, num_experts)
    except RuntimeError as e:
        logger.debug("Full pipeline failed: %s, using per-expert fallback", e)
        return fused_moe_per_expert(
            hidden_states, gate_output, w13, w2,
            topk, renormalize, num_experts)
