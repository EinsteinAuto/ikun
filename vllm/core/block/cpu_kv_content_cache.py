"""Scheduler-owned content index for an inclusive CPU KV cache tier."""

from __future__ import annotations

import heapq
import os
from collections import OrderedDict
from typing import Dict, List, Mapping, Optional, Set, Tuple


ContentHash = bytes
SwapMapping = List[Tuple[int, int]]


def cpu_kv_offload_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Read the experimental selector without accepting ambiguous values."""
    source = os.environ if environ is None else environ
    value = source.get("BI100_CPU_KV_OFFLOAD", "0")
    if value == "0":
        return False
    if value == "1":
        return True
    raise RuntimeError(
        "BI100_CPU_KV_OFFLOAD must be exactly '0' or '1', "
        f"got {value!r}")


class CpuKvContentCache:
    """Track immutable KV blocks held in the worker's pinned CPU cache.

    The scheduler owns this metadata and sends identical physical block maps
    to every tensor-parallel worker. CPU copies are inclusive: loading a block
    back to GPU does not remove its CPU entry. Slots touched by either transfer
    direction are pinned for the whole scheduling step so a D2H destination
    can never overwrite an H2D source before workers execute the maps.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("CPU KV content cache capacity must be positive")
        self.capacity = capacity
        self._hash_to_slot: Dict[ContentHash, int] = {}
        self._slot_to_hash: Dict[int, ContentHash] = {}
        self._ready_slots: Set[int] = set()
        self._lru: OrderedDict[int, None] = OrderedDict()
        self._free_slots = list(range(capacity))
        heapq.heapify(self._free_slots)

        self._step_slots_in_use: Set[int] = set()
        self._step_load_slots: Set[int] = set()
        self._step_h2d: Dict[int, int] = {}
        self._step_d2h: Dict[int, int] = {}
        self._deferred_d2h: Dict[int, ContentHash] = {}
        self._deferred_hashes: Set[ContentHash] = set()
        self._pending_ready_slots: Set[int] = set()

        self.hits = 0
        self.misses = 0
        self.stores = 0
        self.deduplicated_stores = 0
        self.evictions = 0
        self.skipped_stores = 0

    @staticmethod
    def _validate_hash(content_hash: ContentHash) -> None:
        if not isinstance(content_hash, bytes) or len(content_hash) != 32:
            raise ValueError("CPU KV cache key must be a 32-byte content hash")

    @staticmethod
    def _validate_block_id(name: str, block_id: int) -> None:
        if not isinstance(block_id, int) or isinstance(block_id, bool):
            raise TypeError(f"{name} must be an integer")
        if block_id < 0:
            raise ValueError(f"{name} must be non-negative")

    def _touch(self, slot: int) -> None:
        self._lru.pop(slot, None)
        self._lru[slot] = None

    def _select_store_slot(self) -> Optional[int]:
        if self._free_slots:
            return heapq.heappop(self._free_slots)
        for slot in self._lru:
            if slot not in self._step_slots_in_use:
                return slot
        return None

    def _commit_store(self, content_hash: ContentHash,
                      gpu_block: int, slot: int) -> None:
        old_hash = self._slot_to_hash.get(slot)
        if old_hash is not None:
            if slot in self._step_slots_in_use:
                raise RuntimeError("selected an in-use CPU KV slot for eviction")
            del self._hash_to_slot[old_hash]
            self._ready_slots.discard(slot)
            self.evictions += 1

        if slot in self._step_h2d:
            raise RuntimeError(
                "a CPU KV slot cannot be an H2D source and D2H destination "
                "in one scheduler step")
        if slot in self._step_d2h.values():
            raise RuntimeError(f"duplicate D2H destination CPU slot {slot}")

        self._hash_to_slot[content_hash] = slot
        self._slot_to_hash[slot] = content_hash
        self._ready_slots.discard(slot)
        self._step_slots_in_use.add(slot)
        self._step_d2h[gpu_block] = slot
        self._touch(slot)
        self.stores += 1

    def begin_step(self) -> None:
        """Publish D2H stores returned by the preceding synchronous step."""
        if (self._step_slots_in_use or self._step_h2d or self._step_d2h
                or self._deferred_d2h or self._deferred_hashes):
            raise RuntimeError("cannot begin a CPU KV step before draining it")
        self._ready_slots.update(self._pending_ready_slots)
        self._pending_ready_slots.clear()

    def _require_step_started(self) -> None:
        if self._pending_ready_slots:
            raise RuntimeError(
                "CPU KV step must begin before content lookup or eviction")

    def claim_load(self, content_hash: ContentHash) -> Optional[int]:
        """Pin and return a ready CPU source for this scheduling step."""
        self._validate_hash(content_hash)
        self._require_step_started()
        slot = self._hash_to_slot.get(content_hash)
        if slot is None or slot not in self._ready_slots:
            self.misses += 1
            return None
        if slot in self._step_slots_in_use:
            raise RuntimeError(
                f"CPU KV slot {slot} was claimed twice in one scheduler step")
        self._step_slots_in_use.add(slot)
        self._step_load_slots.add(slot)
        self._touch(slot)
        self.hits += 1
        return slot

    def cancel_load(self, content_hash: ContentHash, cpu_slot: int) -> None:
        """Release a claim when GPU allocation fails before H2D is staged."""
        self._validate_hash(content_hash)
        self._validate_block_id("cpu_slot", cpu_slot)
        if self._hash_to_slot.get(content_hash) != cpu_slot:
            raise RuntimeError("CPU KV load cancellation key/slot mismatch")
        if cpu_slot in self._step_h2d:
            raise RuntimeError("cannot cancel a CPU KV load after H2D staging")
        if cpu_slot not in self._step_slots_in_use:
            raise RuntimeError("cannot cancel an unclaimed CPU KV load")
        self._step_slots_in_use.remove(cpu_slot)
        self._step_load_slots.remove(cpu_slot)

    def stage_load(self, content_hash: ContentHash, cpu_slot: int,
                   gpu_block: int) -> None:
        """Stage one CPU-to-GPU promotion after the GPU slot is reserved."""
        self._validate_hash(content_hash)
        self._validate_block_id("cpu_slot", cpu_slot)
        self._validate_block_id("gpu_block", gpu_block)
        if self._hash_to_slot.get(content_hash) != cpu_slot:
            raise RuntimeError("CPU KV load key/slot mismatch")
        if cpu_slot not in self._ready_slots:
            raise RuntimeError("CPU KV load source is not ready")
        if cpu_slot not in self._step_slots_in_use:
            raise RuntimeError("CPU KV load source was not claimed")
        if cpu_slot in self._step_h2d:
            raise RuntimeError(f"duplicate H2D source CPU slot {cpu_slot}")
        if gpu_block in self._step_h2d.values():
            raise RuntimeError(f"duplicate H2D destination GPU block {gpu_block}")
        if cpu_slot in self._step_d2h.values():
            raise RuntimeError(
                "a CPU KV slot cannot be an H2D source and D2H destination "
                "in one scheduler step")
        self._step_h2d[cpu_slot] = gpu_block

    def stage_store(self, content_hash: ContentHash,
                    gpu_block: int) -> bool:
        """Stage a lazy GPU-to-CPU copy for an evicted immutable block."""
        self._validate_hash(content_hash)
        self._validate_block_id("gpu_block", gpu_block)
        self._require_step_started()
        if (content_hash in self._hash_to_slot
                or content_hash in self._deferred_hashes):
            slot = self._hash_to_slot.get(content_hash)
            if slot is not None:
                self._touch(slot)
            self.deduplicated_stores += 1
            return False
        if gpu_block in self._step_d2h or gpu_block in self._deferred_d2h:
            raise RuntimeError(f"duplicate D2H source GPU block {gpu_block}")

        if self._free_slots:
            self._commit_store(
                content_hash, gpu_block, heapq.heappop(self._free_slots))
            return True

        # Do not replace resident content until every lookup in this scheduler
        # step is known. A later H2D claim can refer to any current LRU entry.
        self._deferred_d2h[gpu_block] = content_hash
        self._deferred_hashes.add(content_hash)
        return True

    def _resolve_deferred_stores(self) -> None:
        if self._step_load_slots:
            self.skipped_stores += len(self._deferred_d2h)
        else:
            for gpu_block, content_hash in self._deferred_d2h.items():
                slot = self._select_store_slot()
                if slot is None:
                    self.skipped_stores += 1
                    continue
                self._commit_store(content_hash, gpu_block, slot)
        self._deferred_d2h.clear()
        self._deferred_hashes.clear()

    def drain_step(self) -> Tuple[SwapMapping, SwapMapping]:
        """Finalize this synchronous step and return (H2D, D2H) maps."""
        self._resolve_deferred_stores()
        transfer_slots = (
            set(self._step_h2d) | set(self._step_d2h.values()))
        if transfer_slots != self._step_slots_in_use:
            raise RuntimeError(
                "CPU KV scheduler step contains an uncommitted slot claim")
        if set(self._step_h2d) & set(self._step_d2h.values()):
            raise RuntimeError(
                "CPU KV scheduler step reuses a CPU slot across directions")
        if set(self._step_h2d) != self._step_load_slots:
            raise RuntimeError(
                "CPU KV scheduler step contains an unstaged load claim")

        swap_in = sorted(self._step_h2d.items())
        swap_out = sorted(self._step_d2h.items())
        self._pending_ready_slots.update(self._step_d2h.values())
        self._step_h2d.clear()
        self._step_d2h.clear()
        self._step_slots_in_use.clear()
        self._step_load_slots.clear()
        return swap_in, swap_out

    def resident_slot(self, content_hash: ContentHash) -> Optional[int]:
        self._validate_hash(content_hash)
        return self._hash_to_slot.get(content_hash)

    def is_ready(self, content_hash: ContentHash) -> bool:
        self._validate_hash(content_hash)
        slot = self._hash_to_slot.get(content_hash)
        return slot is not None and slot in self._ready_slots

    @property
    def resident_count(self) -> int:
        return len(self._hash_to_slot)
