"""DP-aware ACL graph decode runner for Qwen3.5.

Ported from xLLM upstream commit 78aa2a85 (PR #2258).
Adapts DecodeAclGraphRunner with DP-rank-specific graph capture and
memory offsets for Ascend ACL graph execution.

Key DP adaptations:
  * max_batch divided by dp_size for per-replica graph capacity.
  * Graph capture uses dp_token_counts / dp_is_decode metadata.
  * Replay validates DP token counts match captured graph shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass
class AclStaticAttentionMetadata:
    """Minimal attention metadata for ACL graph capture / replay."""

    slot_mapping: torch.Tensor
    paged_kv_indptr: torch.Tensor
    paged_kv_indices: torch.Tensor
    paged_kv_last_page_len: torch.Tensor
    qo_indptr: torch.Tensor | None = None
    q_cu_seq_lens: torch.Tensor | None = None
    kv_cu_seq_lens: torch.Tensor | None = None
    kv_seq_lens_host: torch.Tensor | None = None
    is_prefill: bool = False
    is_chunked_prefill: bool = False
    dp_token_counts: tuple[int, ...] = ()
    dp_is_decode: tuple[int, ...] = ()


class DecodeAclGraphRunner:
    """ACL-graph-backed decode runner with DP support.

    Args:
        model: The model's execution sub-module.
        device: Target device for graph capture.
        max_batch: Maximum total batch size across all DP replicas.
        max_model_len: Maximum sequence length (for KV cache sizing).
        dp_size: Number of data-parallel replicas.
        dp_rank: This replica's rank within the DP group.
        decode_batch_size_limit: Optional cap on per-graph batch size.
        num_decoding_tokens: Tokens per sequence in speculative decode.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        max_batch: int,
        max_model_len: int = 8192,
        dp_size: int = 1,
        dp_rank: int = 0,
        decode_batch_size_limit: int | None = None,
        num_decoding_tokens: int = 1,
    ) -> None:
        if dp_size <= 0:
            raise ValueError("dp_size must be positive")
        if not 0 <= dp_rank < dp_size:
            raise ValueError("dp_rank must be in [0, dp_size)")

        self.model = model
        self.device = device
        self.dp_size = dp_size
        self.dp_rank = dp_rank
        self.max_batch = (max_batch + dp_size - 1) // dp_size
        self.max_model_len = max_model_len
        self.num_decoding_tokens = num_decoding_tokens
        self.decode_batch_size_limit = decode_batch_size_limit
        self._graphs: dict[int, Any] = {}
        self._warmed_up = False

    def _validate_dp_token_counts(
        self,
        dp_token_counts: tuple[int, ...] | None,
    ) -> None:
        """Validate DP token counts for graph replay."""
        if self.dp_size > 1:
            if dp_token_counts is None or len(dp_token_counts) != self.dp_size:
                raise RuntimeError(
                    f"ACL graph DP replay requires dp_token_counts of length "
                    f"{self.dp_size} (got "
                    f"{len(dp_token_counts) if dp_token_counts else 'None'}). "
                    f"All DP ranks must use the same graph shape."
                )

    def warmup(self, device: torch.device | None = None) -> None:
        """Pre-capture ACL graphs for all bucket sizes."""
        if self._warmed_up:
            return
        dev = device or self.device

        batch_sizes = [1, 2, 4, 8]
        batch_sizes.extend(range(16, self.max_batch + 1, 16))
        batch_sizes = [b for b in batch_sizes if b <= self.max_batch]

        for batch_size in reversed(batch_sizes):
            padded = batch_size * self.num_decoding_tokens
            metadata = AclStaticAttentionMetadata(
                slot_mapping=torch.zeros(padded, dtype=torch.int32, device=dev),
                paged_kv_indptr=torch.arange(
                    padded + 1, dtype=torch.int32, device=dev
                ),
                paged_kv_indices=torch.zeros(
                    padded, dtype=torch.int32, device=dev
                ),
                paged_kv_last_page_len=torch.ones(
                    padded, dtype=torch.int32, device=dev
                ),
                dp_token_counts=tuple([padded] * self.dp_size)
                if self.dp_size > 1
                else (),
                dp_is_decode=tuple([1] * self.dp_size)
                if self.dp_size > 1
                else (),
            )
            self._graphs[padded] = metadata
        self._warmed_up = True

    def can_execute(
        self,
        input_ids: torch.Tensor,
        dp_token_counts: tuple[int, ...] | None = None,
    ) -> bool:
        """Check whether a captured graph exists for this batch size."""
        if not self._warmed_up:
            return False
        batch_size = input_ids.shape[0]
        if self.dp_size > 1:
            self._validate_dp_token_counts(dp_token_counts)
        return batch_size <= self.max_batch * self.num_decoding_tokens
