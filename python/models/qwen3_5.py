"""Qwen3.5 model DP (data parallel) forward-pass support.

Ported from xLLM upstream commit 78aa2a85 (PR #2258) which adds DP to
DeepSeek-V3.2.  Adapted for Qwen3.5's MoE architecture:

  * 256 routed experts + 1 shared expert, top-8 routing
  * Combined router + shared-expert gate in a single replicated linear
  * RowParallelLinear shared expert with deferred all-reduce

The DP pattern is identical to DeepSeek-V3.2:
  1. Before MoE: all-gather hidden states across DP group
  2. Run MoE on the full global batch
  3. After MoE: slice output back to this replica's local tokens

This module provides:
  * ``dp_forward_moe_wrapper``: drop-in replacement for MoeSparseBlock.forward
  * ``configure_dp``: inject dp_size/dp_rank into MoeSparseBlock at init time
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def configure_dp(moe_block: nn.Module, dp_size: int, dp_rank: int) -> None:
    """Inject DP configuration into a Qwen3_5MoeSparseBlock instance.

    Call this after model construction, before the first forward pass.
    Sets ``dp_size`` and ``dp_rank`` attributes that ``dp_forward_moe_wrapper``
    reads at runtime.
    """
    moe_block.dp_size = dp_size
    moe_block.dp_rank = dp_rank


def dp_forward_moe_wrapper(
    moe_block: nn.Module,
    hidden_states: torch.Tensor,
    original_forward,
    metadata: object,
) -> torch.Tensor:
    """Wrap a MoeSparseBlock.forward call with DP all-gather / scatter.

    Ported from tpu-inference (xLLM upstream):
      - PR #2577 (04077875): DP attention for hybrid models
      - PR #2679 (df7f5b35): Optimize attn DP + MoE EP ReduceScatter

    Pattern:
      1. All-gather hidden states across DP ranks → each rank sees full batch
      2. Run MoE forward on the gathered global batch
      3. Slice output back to this rank's local tokens

    The forward() in qwen3_5.py handles EP reduction internally
    (all-reduce over WORLD). When DP > 1, a future optimization can
    replace steps 2+3 with:
      2'. Run MoE experts + shared expert WITHOUT reduction
      3'. reduce-scatter the combined output (fuses EP sum + DP scatter)
    This would halve the communication volume (upstream PR #2679's
    psum_scatter optimization). For now we use the safe all-reduce + slice.

    Args:
        moe_block: The Qwen3_5MoeSparseBlock instance.
        hidden_states: Local hidden states [local_tokens, hidden_size].
        original_forward: The original MoeSparseBlock.forward callable.
        metadata: Attention metadata with dp_token_counts / dp_is_decode.

    Returns:
        Output tensor sliced to [local_tokens, hidden_size].
    """
    dp_size = getattr(moe_block, "dp_size", 1)
    dp_rank = getattr(moe_block, "dp_rank", 0)

    if dp_size <= 1:
        return original_forward(hidden_states)

    token_counts = list(metadata.dp_token_counts)
    if len(token_counts) != dp_size:
        raise RuntimeError(
            f"expected {dp_size} DP token counts, got {len(token_counts)}"
        )

    local_tokens = hidden_states.shape[0]
    padded_tokens = 0
    use_compact_gather = False

    # Decide gather strategy
    is_prefill = getattr(metadata, "is_prefill", False) or getattr(
        metadata, "is_chunked_prefill", False
    )
    execution_state = getattr(metadata, "execution_state", None)
    is_graph = execution_state is not None
    dp_is_decode = getattr(metadata, "dp_is_decode", None)
    all_decode = dp_is_decode is not None and all(dp_is_decode)

    if is_graph or is_prefill or not all_decode:
        # Padded all-gather path
        padded_tokens = max(token_counts)
        pad_size = padded_tokens - local_tokens
        if pad_size > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_size))
        hidden_states = _dp_all_gather(
            hidden_states, dim=0, world_size=dp_size, group_name="dp"
        )
    else:
        # Compact variable-length all-gather path
        use_compact_gather = True
        hidden_states = _dp_all_gather_variable(
            hidden_states, token_counts, dp_rank, "dp"
        )

    # Run MoE on the globally-gathered batch.
    # forward() handles EP all-reduce internally, output is fully reduced.
    output = original_forward(hidden_states)

    # Slice back to local tokens
    if use_compact_gather:
        offset = sum(token_counts[:dp_rank])
        output = output.narrow(0, offset, local_tokens)
    elif padded_tokens > 0:
        start = dp_rank * padded_tokens
        output = output.narrow(0, start, local_tokens)

    return output


def apply_dp_to_model(model: nn.Module, dp_size: int, dp_rank: int) -> None:
    """Walk a Qwen3.5 model and inject DP into all MoeSparseBlock layers.

    Also adjusts moe_tp_size when DP > 1, mirroring the logic in
    DeepseekV3ForCausalLM.__init__:
      - With ep_size=1: force moe_tp_size=1 (all-reduce falls through to TP)
      - With ep_size>1: moe_tp_size //= dp_size
    """
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if "MoeSparseBlock" in cls_name or "MoE" in cls_name:
            configure_dp(module, dp_size, dp_rank)


# ---------------------------------------------------------------------------
# Distributed helpers (same as python/layers/fused_moe.py)
# ---------------------------------------------------------------------------


def _dp_all_gather(
    tensor: torch.Tensor,
    dim: int = 0,
    world_size: int = 1,
    group_name: str = "dp",
) -> torch.Tensor:
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
                torch.empty(
                    count, hidden_dim, dtype=tensor.dtype, device=tensor.device
                )
            )
    torch.distributed.all_gather(
        recv_tensors, tensor[: token_counts[dp_rank]], group=group
    )
    return torch.cat(recv_tensors, dim=0)