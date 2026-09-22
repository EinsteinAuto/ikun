"""DP-aware fused MoE layer for Qwen3.5 Python model executor.

Ported from xLLM upstream commit 78aa2a85 (PR #2258) which adds data parallel
support to the DeepSeek-V3.2 Python model executor.  Adapted here for Qwen3.5's
MoE architecture (256 routed experts + shared expert, top-8 routing).

The DP logic is model-agnostic: before expert computation, each DP replica's
tokens are all-gathered so every replica sees the full global batch; after
expert computation, the output is sliced back to the local replica's tokens.
This ensures each replica routes experts independently while producing correct
outputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DPAwareMoEMixin:
    """Mixin that adds DP all-gather / scatter logic to any MoE forward pass.

    Requires the host class to set ``self.dp_size`` and ``self.dp_rank``.
    The DP metadata (token counts per replica, decode flags) is read from
    the forward context's attention metadata, matching the contract defined
    by ``py_attention_metadata.cpp`` in xLLM's C++ runtime.
    """

    dp_size: int
    dp_rank: int

    def _dp_gather_inputs(
        self,
        hidden_states: torch.Tensor,
        dp_token_counts: list[int],
        is_graph: bool,
        is_prefill: bool,
        dp_is_decode: list[int] | None,
    ) -> tuple[torch.Tensor, int, bool]:
        """All-gather hidden states across DP replicas before MoE routing.

        Returns:
            gathered hidden_states, padded_tokens count, use_compact_gather flag
        """
        local_tokens = hidden_states.shape[0]
        padded_tokens = 0
        use_compact_gather = False

        all_decode = dp_is_decode is not None and all(dp_is_decode)

        if is_graph or is_prefill or not all_decode:
            # Padded all-gather: pad each replica to max token count, then
            # concatenate.  Required for graph capture (fixed shapes) and
            # prefill (variable lengths).
            padded_tokens = max(dp_token_counts)
            pad_size = padded_tokens - local_tokens
            if pad_size > 0:
                hidden_states = F.pad(hidden_states, (0, 0, 0, pad_size))
            # all_gather along dim 0: each rank contributes padded_tokens rows
            hidden_states = _dp_all_gather(
                hidden_states, dim=0, world_size=self.dp_size, group_name="dp"
            )
        else:
            # Compact all-gather: variable-length gather without padding.
            # More efficient for decode when all replicas are decoding.
            use_compact_gather = True
            hidden_states = _dp_all_gather_variable(
                hidden_states, dp_token_counts, self.dp_rank, "dp"
            )

        return hidden_states, padded_tokens, use_compact_gather

    def _dp_scatter_output(
        self,
        output: torch.Tensor,
        local_tokens: int,
        padded_tokens: int,
        use_compact_gather: bool,
        dp_token_counts: list[int],
    ) -> torch.Tensor:
        """Slice the globally-computed MoE output back to this DP replica."""
        if use_compact_gather:
            offset = sum(dp_token_counts[: self.dp_rank])
            output = output.narrow(0, offset, local_tokens)
        elif padded_tokens > 0:
            start = self.dp_rank * padded_tokens
            output = output.narrow(0, start, local_tokens)
        return output


# ---------------------------------------------------------------------------
# Distributed helpers — thin wrappers that can be mocked in unit tests.
# In production these delegate to torch.distributed / xLLM's NCCL groups.
# ---------------------------------------------------------------------------


def _dp_all_gather(
    tensor: torch.Tensor,
    dim: int = 0,
    world_size: int = 1,
    group_name: str = "dp",
) -> torch.Tensor:
    """All-gather ``tensor`` along ``dim`` across the DP process group."""
    if world_size <= 1:
        return tensor
    from vllm.distributed import get_dp_group
    group = get_dp_group()
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, tensor, group=group)
    return torch.cat(gathered, dim=dim)


def _dp_all_gather_variable(
    tensor: torch.Tensor,
    token_counts: list[int],
    dp_rank: int,
    group_name: str = "dp",
) -> torch.Tensor:
    """Variable-length all-gather: each rank contributes a different number
    of tokens.  Returns a compact concatenation without padding."""
    from vllm.distributed import get_dp_group
    group = get_dp_group()
    world_size = len(token_counts)
    hidden_dim = tensor.shape[1] if tensor.dim() > 1 else 1
    recv_tensors = []
    for i, count in enumerate(token_counts):
        if i == dp_rank:
            recv_tensors.append(tensor[:count])
        else:
            recv_tensors.append(
                torch.empty(count, hidden_dim, dtype=tensor.dtype, device=tensor.device)
            )
    torch.distributed.all_gather(recv_tensors, tensor[:token_counts[dp_rank]], group=group)
    return torch.cat(recv_tensors, dim=0)
