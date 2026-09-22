import enum
import heapq
import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Dict, List, OrderedDict, Tuple


ContentHash = bytes


class EvictionPolicy(enum.Enum):
    """Enum for eviction policy used by make_evictor to instantiate the correct
       Evictor subclass.
    """
    LRU = enum.auto()
    FREQUENCY_AWARE = enum.auto()


class Evictor(ABC):
    """The Evictor subclasses should be used by the BlockAllocator class to
    handle eviction of freed PhysicalTokenBlocks.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __contains__(self, block_id: int) -> bool:
        pass

    @abstractmethod
    def evict(self) -> Tuple[int, ContentHash]:
        """Runs the eviction algorithm and returns the evicted block's
        content hash along with physical block id along with physical block id
        """
        pass

    @abstractmethod
    def add(self, block_id: int, content_hash: ContentHash,
            num_hashed_tokens: int,
            last_accessed: float):
        """Adds block to the evictor, making it a candidate for eviction"""
        pass

    @abstractmethod
    def update(self, block_id: int, last_accessed: float):
        """Update corresponding block's access time in metadata"""
        pass

    @abstractmethod
    def remove(self, block_id: int):
        """Remove a given block id from the cache."""
        pass

    @property
    @abstractmethod
    def num_blocks(self) -> int:
        pass


class BlockMetaData():
    """Data structure for storing key data describe cached block, so that
    evitor could use to make its decision which one to choose for eviction

    Here we use physical block id as the dict key, as there maybe several
    blocks with the same content hash, but their physical id is unique.
    """

    def __init__(self, content_hash: ContentHash, num_hashed_tokens: int,
                 last_accessed: float):
        self.content_hash = content_hash
        self.num_hashed_tokens = num_hashed_tokens
        self.last_accessed = last_accessed


class LRUEvictor(Evictor):
    """Evicts in a least-recently-used order using the last_accessed timestamp
    that's recorded in the PhysicalTokenBlock. If there are multiple blocks with
    the same last_accessed time, then the one with the largest num_hashed_tokens
    will be evicted. If two blocks each have the lowest last_accessed time and
    highest num_hashed_tokens value, then one will be chose arbitrarily
    """

    def __init__(self):
        self.free_table: OrderedDict[int, BlockMetaData] = OrderedDict()

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def evict(self) -> Tuple[int, ContentHash]:
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        evicted_block, evicted_block_id = None, None
        # The blocks with the lowest timestamps should be placed consecutively
        # at the start of OrderedDict. Loop through all these blocks to
        # find the one with maximum number of hashed tokens.
        for _id, block in self.free_table.items():
            if evicted_block is None:
                evicted_block, evicted_block_id = block, _id
                continue
            if evicted_block.last_accessed < block.last_accessed:
                break
            if evicted_block.num_hashed_tokens < block.num_hashed_tokens:
                evicted_block, evicted_block_id = block, _id

        assert evicted_block is not None
        assert evicted_block_id is not None
        self.free_table.pop(evicted_block_id)

        return evicted_block_id, evicted_block.content_hash

    def add(self, block_id: int, content_hash: ContentHash,
            num_hashed_tokens: int,
            last_accessed: float):
        self.free_table[block_id] = BlockMetaData(content_hash,
                                                  num_hashed_tokens,
                                                  last_accessed)

    def update(self, block_id: int, last_accessed: float):
        self.free_table[block_id].last_accessed = last_accessed

    def remove(self, block_id: int):
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        self.free_table.pop(block_id)

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)


class FrequencyAwareEvictor(Evictor):
    """Evict the least frequently reused logical prefix content first.

    Content frequency survives physical block reuse. Heap entries carry a
    generation and are lazily invalidated so eviction remains O(log N) without
    allowing stale entries to grow without bound.
    """

    _COMPACTION_FACTOR = 2
    _COMPACTION_SLACK = 1

    def __init__(self):
        self.free_table: Dict[int, BlockMetaData] = {}
        self.frequency_by_hash: Dict[ContentHash, int] = {}
        self._heap: List[Tuple[int, float, int, int, int]] = []
        self._generations: Dict[int, int] = {}
        self._next_generation = 0

    @staticmethod
    def _validate_content_hash(content_hash: ContentHash) -> None:
        if not isinstance(content_hash, bytes) or len(content_hash) != 32:
            raise ValueError(
                "frequency-aware eviction requires a 32-byte content hash")

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def _heap_key(self, block_id: int, block: BlockMetaData,
                  generation: int) -> Tuple[int, float, int, int, int]:
        return (
            self.frequency_by_hash[block.content_hash],
            block.last_accessed,
            -block.num_hashed_tokens,
            block_id,
            generation,
        )

    def _push(self, block_id: int) -> None:
        self._next_generation += 1
        generation = self._next_generation
        self._generations[block_id] = generation
        heapq.heappush(
            self._heap,
            self._heap_key(
                block_id, self.free_table[block_id], generation),
        )

    def _compact_if_needed(self) -> None:
        limit = (
            self._COMPACTION_FACTOR * len(self.free_table)
            + self._COMPACTION_SLACK
        )
        if len(self._heap) <= limit:
            return
        self._heap = [
            self._heap_key(block_id, block, self._generations[block_id])
            for block_id, block in self.free_table.items()
        ]
        heapq.heapify(self._heap)

    def evict(self) -> Tuple[int, ContentHash]:
        if not self.free_table:
            raise ValueError("No usable cache memory left")

        while self._heap:
            entry = heapq.heappop(self._heap)
            frequency, _, _, block_id, generation = entry
            block = self.free_table.get(block_id)
            if (
                block is None
                or self._generations.get(block_id) != generation
            ):
                continue
            if self.frequency_by_hash[block.content_hash] != frequency:
                heapq.heappush(
                    self._heap,
                    self._heap_key(block_id, block, generation),
                )
                continue

            block = self.free_table.pop(block_id)
            self._generations.pop(block_id)
            self._compact_if_needed()
            return block_id, block.content_hash

        raise RuntimeError("Evictor heap has no usable entry")

    def add(self, block_id: int, content_hash: ContentHash,
            num_hashed_tokens: int, last_accessed: float):
        self._validate_content_hash(content_hash)
        self.frequency_by_hash[content_hash] = (
            self.frequency_by_hash.get(content_hash, 0) + 1)
        self.free_table[block_id] = BlockMetaData(
            content_hash, num_hashed_tokens, last_accessed)
        self._push(block_id)
        self._compact_if_needed()

    def update(self, block_id: int, last_accessed: float):
        self.free_table[block_id].last_accessed = last_accessed
        self._push(block_id)
        self._compact_if_needed()

    def remove(self, block_id: int):
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        self.free_table.pop(block_id)
        self._generations.pop(block_id)
        self._compact_if_needed()

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)


def eviction_policy_from_env(
    environ: Mapping[str, str] | None = None,
) -> EvictionPolicy:
    source = os.environ if environ is None else environ
    value = source.get("BI100_KV_EVICTION_POLICY", "lru").strip().lower()
    policies = {
        "lru": EvictionPolicy.LRU,
        "frequency": EvictionPolicy.FREQUENCY_AWARE,
    }
    if value not in policies:
        raise ValueError(
            "BI100_KV_EVICTION_POLICY must be one of: frequency, lru")
    return policies[value]


def make_evictor(eviction_policy: EvictionPolicy) -> Evictor:
    if eviction_policy == EvictionPolicy.LRU:
        return LRUEvictor()
    elif eviction_policy == EvictionPolicy.FREQUENCY_AWARE:
        return FrequencyAwareEvictor()
    else:
        raise ValueError(f"Unknown cache eviction policy: {eviction_policy}")
