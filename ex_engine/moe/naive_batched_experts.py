"""naive_batched_experts.py -- MoE expert computation for BI-V100."""

import torch
import torch.nn.functional as F
from typing import Optional


def _resize_cache(x: torch.Tensor, v: tuple) -> torch.Tensor:
    from math import prod
    assert prod(v) <= x.numel(), f"{v} ({prod(v)}) <= {x.shape} ({x.numel()})"
    return x.flatten()[:prod(v)].view(*v)


def naive_batched_moe_forward(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    act_fn: Optional[object] = None,
) -> torch.Tensor:
    T = hidden_states.shape[0]
    H = hidden_states.shape[1]
    I = w2.shape[2]
    top_k = topk_ids.shape[1]

    out = torch.zeros(T, H, dtype=hidden_states.dtype, device=hidden_states.device)

    if T == 1:
        eids = topk_ids[0]
        ws = topk_weights[0]

        w13_sel = torch.index_select(w13, 0, eids)
        w2_sel = torch.index_select(w2, 0, eids)

        gate_up = hidden_states @ w13_sel.reshape(-1, H).t()
        gate_up = gate_up.view(top_k, -1)

        if act_fn is not None:
            act = act_fn(gate_up)
        else:
            gate = gate_up[..., :I]
            up = gate_up[..., I:]
            act = F.silu(gate) * up

        expert_out = torch.bmm(w2_sel, act.unsqueeze(-1)).squeeze(-1)
        out = torch.einsum('k,kh->h', ws, expert_out).unsqueeze(0)

    else:
        flat_eids = topk_ids.reshape(-1)
        flat_weights = topk_weights.reshape(-1)
        flat_token_ids = torch.arange(
            T, device=hidden_states.device
        ).repeat_interleave(top_k)

        num_experts = w13.shape[0]
        for expert in range(num_experts):
            mask = (flat_eids == expert)
            if not mask.any():
                continue

            token_ids = flat_token_ids[mask]
            weights = flat_weights[mask]
            expert_input = hidden_states[token_ids]

            gate_up = expert_input @ w13[expert].transpose(0, 1)

            if act_fn is not None:
                act = act_fn(gate_up)
            else:
                gate = gate_up[..., :I]
                up = gate_up[..., I:]
                act = F.silu(gate) * up

            expert_out = act @ w2[expert].transpose(0, 1)
            out.index_add_(0, token_ids, expert_out * weights.unsqueeze(1))

    return out
