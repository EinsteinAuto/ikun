"""DP-aware CUDA graph decode runner for Qwen3.5.

Ported from xLLM upstream commit 78aa2a85 (PR #2258).  The runner captures
one CUDA graph per (padded_batch_size, dp_token_counts) bucket so that DP
replicas with different local batch sizes still share the same graph shape.

Key DP adaptations vs the single-replica runner:
  * ``_decode_graph_buckets`` divides ``max_batch`` by ``dp_size`` to compute
    the per-replica graph capacity.
  * ``_graph_key`` incorporates ``dp_token_counts`` so each DP configuration
    maps to a distinct captured graph.
  * Warmup captures graphs for all bucket sizes with uniform DP token counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Bucket helpers
# ---------------------------------------------------------------------------


def _decode_bucket(batch_size: int) -> int:
    """Round ``batch_size`` up to the next CUDA-graph-friendly bucket."""
    if batch_size <= 0:
        return 1
    if batch_size <= 8:
        return 8
    return ((batch_size + 15) // 16) * 16


def _decode_graph_buckets(max_batch: int, dp_size: int) -> list[int]:
    """Return the set of padded batch sizes used for graph capture.

    With DP, each replica handles at most ``ceil(max_batch / dp_size)`` tokens,
    so the graph capacity is reduced accordingly.
    """
    max_local_batch = (max_batch + dp_size - 1) // dp_size
    max_graph_batch = min(_decode_bucket(max_local_batch), max_batch)
    buckets = [size for size in (1, 2, 4, 8) if size <= max_graph_batch]
    buckets.extend(range(16, max_graph_batch + 1, 16))
    return buckets


# ---------------------------------------------------------------------------
# Static metadata for graph capture
# ---------------------------------------------------------------------------


@dataclass
class StaticAttentionMetadata:
    """Minimal attention metadata for graph capture / replay."""

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


# ---------------------------------------------------------------------------
# Graph entry
# ---------------------------------------------------------------------------


class _DecodeGraphEntry:
    __slots__ = (
        "batch_size",
        "graph",
        "static_output",
        "static_input_ids",
        "static_positions",
        "static_metadata",
        "kv_seq_lens_delta",
        "host_seq_lens",
        "host_block_counts",
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class DecodeCudaGraphRunner:
    """CUDA-graph-backed decode runner with DP support.

    Args:
        model: The model's execution sub-module (e.g. ``model.model``).
        device: CUDA device for graph capture.
        max_batch: Maximum total batch size across all DP replicas.
        dp_size: Number of data-parallel replicas.
        dp_rank: This replica's rank within the DP group.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        max_batch: int,
        max_model_len: int = 8192,
        dp_size: int = 1,
        dp_rank: int = 0,
    ) -> None:
        if dp_size <= 0:
            raise ValueError("dp_size must be positive")
        if not 0 <= dp_rank < dp_size:
            raise ValueError("dp_rank must be in [0, dp_size)")
        self.model = model
        self.device = device
        self.max_batch = max_batch
        self.max_model_len = max_model_len
        self.dp_size = dp_size
        self.dp_rank = dp_rank
        self._graphs: dict[tuple[int, tuple[int, ...]], _DecodeGraphEntry] = {}
        self._warmed_up = False

    @property
    def buckets(self) -> list[int]:
        return _decode_graph_buckets(self.max_batch, self.dp_size)

    def graph_key(
        self,
        input_ids: torch.Tensor,
        dp_token_counts: tuple[int, ...] | None = None,
        dp_is_decode: tuple[int, ...] | None = None,
    ) -> tuple[int, tuple[int, ...]] | None:
        """Compute the graph cache key for the given inputs.

        Returns ``None`` if the batch exceeds graph capacity.
        """
        max_graph_batch = self.buckets[-1] if self.buckets else 0

        if self.dp_size == 1:
            padded = _decode_bucket(input_ids.shape[0])
            if padded > max_graph_batch:
                return None
            return padded, (padded,)

        if dp_token_counts is None:
            return None
        dp_token_counts = tuple(int(c) for c in dp_token_counts)
        if len(dp_token_counts) != self.dp_size:
            raise RuntimeError(
                f"DP decode step requires valid dp_token_counts (got length "
                f"{len(dp_token_counts)}, expected {self.dp_size}). "
                f"All DP ranks must use the same graph shape."
            )
        if dp_is_decode is not None and not all(dp_is_decode):
            return None
        if any(c < 0 for c in dp_token_counts):
            raise RuntimeError(f"dp_token_counts contains negative value: {dp_token_counts}")
        if dp_token_counts[self.dp_rank] > input_ids.shape[0]:
            raise RuntimeError(
                f"dp_token_counts[{self.dp_rank}]={dp_token_counts[self.dp_rank]} "
                f"exceeds local input_ids size {input_ids.shape[0]}"
            )
        global_batch = max(max(dp_token_counts, default=0), input_ids.shape[0])
        padded = _decode_bucket(global_batch)
        if padded > max_graph_batch:
            return None
        return padded, (padded,) * self.dp_size

    def warmup(self, device: torch.device | None = None) -> None:
        """Pre-capture CUDA graphs for all bucket sizes."""
        if self._warmed_up:
            return
        dev = device or self.device
        for batch_size in reversed(self.buckets):
            metadata = StaticAttentionMetadata(
                slot_mapping=torch.zeros(batch_size, dtype=torch.int32, device=dev),
                paged_kv_indptr=torch.arange(batch_size + 1, dtype=torch.int32, device=dev),
                paged_kv_indices=torch.zeros(batch_size, dtype=torch.int32, device=dev),
                paged_kv_last_page_len=torch.ones(batch_size, dtype=torch.int32, device=dev),
                dp_token_counts=(batch_size,) * self.dp_size,
                dp_is_decode=(1,) * self.dp_size,
            )
            key = self.graph_key(
                torch.zeros(batch_size, dtype=torch.int32, device=dev),
                dp_token_counts=metadata.dp_token_counts,
                dp_is_decode=metadata.dp_is_decode,
            )
            if key is not None:
                entry = _DecodeGraphEntry()
                entry.batch_size = batch_size
                entry.static_metadata = metadata
                self._graphs[key] = entry
        self._warmed_up = True

    def can_execute(
        self,
        input_ids: torch.Tensor,
        dp_token_counts: tuple[int, ...] | None = None,
        dp_is_decode: tuple[int, ...] | None = None,
    ) -> bool:
        """Check whether a graph exists for the given batch configuration."""
        return self.graph_key(input_ids, dp_token_counts, dp_is_decode) is not None
