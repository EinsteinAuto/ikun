"""A block manager that manages token blocks."""
import hashlib
import os
import struct
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence as GenericSequence, Tuple

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency in some envs
    Image = None  # type: ignore
try:
    import torch
except Exception:  # pragma: no cover - optional dependency in some envs
    torch = None  # type: ignore

from vllm.core.block.block_table import BlockTable
from vllm.core.block.cpu_gpu_block_allocator import CpuGpuBlockAllocator
from vllm.core.block.interfaces import Block
from vllm.core.block.prefix_caching_block import (ComputedBlocksTracker,
                                                  LastAccessBlocksTracker)
from vllm.core.block.utils import check_no_caching_or_swa_for_blockmgr_encdec
from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.logger import init_logger
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device

SeqId = int
EncoderSeqId = str

logger = init_logger(__name__)


class BlockSpaceManagerV2(BlockSpaceManager):
    """BlockSpaceManager which manages the allocation of KV cache.

    It owns responsibility for allocation, swapping, allocating memory for
    autoregressively-generated tokens, and other advanced features such as
    prefix caching, forking/copy-on-write, and sliding-window memory allocation.

    This class implements the design described in
    https://github.com/vllm-project/vllm/pull/3492.

    Lookahead slots
        The block manager has the notion of a "lookahead slot". These are slots
        in the KV cache that are allocated for a sequence. Unlike the other
        allocated slots, the content of these slots is undefined -- the worker
        may use the memory allocations in any way.

        In practice, a worker could use these lookahead slots to run multiple
        forward passes for a single scheduler invocation. Each successive
        forward pass would write KV activations to the corresponding lookahead
        slot. This allows low inter-token latency use-cases, where the overhead
        of continuous batching scheduling is amortized over >1 generated tokens.

        Speculative decoding uses lookahead slots to store KV activations of
        proposal tokens.

        See https://github.com/vllm-project/vllm/pull/3250 for more information
        on lookahead scheduling.

    Args:
        block_size (int): The size of each memory block.
        num_gpu_blocks (int): The number of memory blocks allocated on GPU.
        num_cpu_blocks (int): The number of memory blocks allocated on CPU.
        watermark (float, optional): The threshold used for memory swapping.
            Defaults to 0.01.
        sliding_window (Optional[int], optional): The size of the sliding
            window. Defaults to None.
        enable_caching (bool, optional): Flag indicating whether caching is
            enabled. Defaults to False.
    """

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        watermark: float = 0.01,
        sliding_window: Optional[int] = None,
        enable_caching: bool = False,
    ) -> None:
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks

        self.sliding_window = sliding_window
        # max_block_sliding_window is the max number of blocks that need to be
        # allocated
        self.max_block_sliding_window = None
        if sliding_window is not None:
            # +1 here because // rounds down
            num_blocks = sliding_window // block_size + 1
            # +1 here because the last block may not be full,
            # and so the sequence stretches one more block at the beginning
            # For example, if sliding_window is 3 and block_size is 4,
            # we may need 2 blocks when the second block only holds 1 token.
            self.max_block_sliding_window = num_blocks + 1

        self.watermark = watermark
        assert watermark >= 0.0

        self.enable_caching = enable_caching

        self.watermark_blocks = int(watermark * num_gpu_blocks)

        self.block_allocator = CpuGpuBlockAllocator.create(
            allocator_type="prefix_caching" if enable_caching else "naive",
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            block_size=block_size,
        )

        self.block_tables: Dict[SeqId, BlockTable] = {}
        self.cross_block_tables: Dict[EncoderSeqId, BlockTable] = {}
        self._warned_mm_namespace_requests = set[str]()
        self._request_local_namespace: Dict[str, bytes] = {}
        self._runtime_cache_namespace = self._build_runtime_cache_namespace()

        self._computed_blocks_tracker = ComputedBlocksTracker(
            self.block_allocator)
        self._last_access_blocks_tracker = LastAccessBlocksTracker(
            self.block_allocator)

    def can_allocate(self,
                     seq_group: SequenceGroup,
                     num_lookahead_slots: int = 0) -> AllocStatus:
        # FIXME(woosuk): Here we assume that all sequences in the group share
        # the same prompt. This may not be true for preempted sequences.

        check_no_caching_or_swa_for_blockmgr_encdec(self, seq_group)

        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        num_required_blocks = BlockTable.get_num_required_blocks(
            seq.get_token_ids(),
            block_size=self.block_size,
            num_lookahead_slots=num_lookahead_slots,
        )

        if seq_group.is_encoder_decoder():
            encoder_seq = seq_group.get_encoder_seq()
            assert encoder_seq is not None
            num_required_blocks += BlockTable.get_num_required_blocks(
                encoder_seq.get_token_ids(),
                block_size=self.block_size,
            )

        if self.max_block_sliding_window is not None:
            num_required_blocks = min(num_required_blocks,
                                      self.max_block_sliding_window)

        num_free_gpu_blocks = self.block_allocator.get_num_free_blocks(
            device=Device.GPU)

        # Use watermark to avoid frequent cache eviction.
        if (self.num_total_gpu_blocks - num_required_blocks <
                self.watermark_blocks):
            return AllocStatus.NEVER
        if num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks:
            return AllocStatus.OK
        else:
            return AllocStatus.LATER

    def _allocate_sequence(
        self,
        seq: Sequence,
        cache_namespace: Optional[bytes] = None,
    ) -> BlockTable:
        block_table = BlockTable(
            block_size=self.block_size,
            block_allocator=self.block_allocator,
            max_block_sliding_window=self.max_block_sliding_window,
            cache_namespace=cache_namespace,
        )
        if seq.get_token_ids():
            # Add blocks to the block table only if the sequence is non empty.
            block_table.allocate(seq.get_token_ids())

        return block_table

    def allocate(self, seq_group: SequenceGroup) -> None:

        # Allocate self-attention block tables for decoder sequences
        waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
        assert not (set(seq.seq_id for seq in waiting_seqs)
                    & self.block_tables.keys()), "block table already exists"

        # NOTE: Here we assume that all sequences in the group have the same
        # prompt.
        seq = waiting_seqs[0]
        request_id = seq_group.request_id
        cache_namespace = self._get_cache_namespace(
            seq,
            request_id=request_id,
            seq_group=seq_group,
        )
        block_table: BlockTable = self._allocate_sequence(
            seq,
            cache_namespace=cache_namespace,
        )
        self.block_tables[seq.seq_id] = block_table

        # Track seq
        self._computed_blocks_tracker.add_seq(seq.seq_id)
        self._last_access_blocks_tracker.add_seq(seq.seq_id)

        # Assign the block table for each sequence.
        for seq in waiting_seqs[1:]:
            self.block_tables[seq.seq_id] = block_table.fork()

            # Track seq
            self._computed_blocks_tracker.add_seq(seq.seq_id)
            self._last_access_blocks_tracker.add_seq(seq.seq_id)

        # Allocate cross-attention block table for encoder sequence
        #
        # NOTE: Here we assume that all sequences in the group have the same
        # encoder prompt.
        request_id = seq_group.request_id

        assert (request_id
                not in self.cross_block_tables), \
            "block table already exists"

        check_no_caching_or_swa_for_blockmgr_encdec(self, seq_group)

        if seq_group.is_encoder_decoder():
            encoder_seq = seq_group.get_encoder_seq()
            assert encoder_seq is not None
            encoder_cache_namespace = self._get_cache_namespace(
                encoder_seq,
                request_id=request_id,
                seq_group=seq_group)
            block_table = self._allocate_sequence(
                encoder_seq, cache_namespace=encoder_cache_namespace)
            self.cross_block_tables[request_id] = block_table

    @staticmethod
    def _has_multi_modal_payload(multi_modal_data: Any) -> bool:
        if multi_modal_data is None:
            return False
        if isinstance(multi_modal_data, Mapping):
            try:
                return len(multi_modal_data) > 0
            except (TypeError, ValueError, RuntimeError, OSError,
                    OverflowError, AttributeError, LookupError, struct.error):
                # Treat an unusual mapping as payload and let normalization
                # either identify it or select request-local isolation.
                return True
        return True

    def _get_cache_namespace(self, seq: Sequence, request_id: str,
                             seq_group: SequenceGroup) -> bytes:
        digest = hashlib.sha256()
        digest.update(b"bi100-request-prefix-namespace-v1|")
        digest.update(self._runtime_cache_namespace)
        digest.update(self._adapter_cache_namespace(seq_group))

        multi_modal_data = seq.multi_modal_data
        if self._has_multi_modal_payload(multi_modal_data):
            try:
                mm_namespace = self._hash_multi_modal_namespace(
                    multi_modal_data)
            except (TypeError, ValueError, RuntimeError, OSError,
                    OverflowError, AttributeError, LookupError, struct.error):
                if request_id not in self._warned_mm_namespace_requests:
                    logger.warning(
                        "Request %s has multimodal input that cannot be "
                        "normalized for cache namespace hashing. Falling "
                        "back to "
                        "request-local namespace isolation.",
                        request_id,
                    )
                    self._warned_mm_namespace_requests.add(request_id)
                mm_namespace = self._request_local_fallback_cache_namespace(
                    request_id=request_id)
            digest.update(b"mm|")
            digest.update(mm_namespace)
        else:
            digest.update(b"text|")
        return digest.digest()

    def can_append_slots(self, seq_group: SequenceGroup,
                         num_lookahead_slots: int) -> bool:
        """Determine if there is enough space in the GPU KV cache to continue
        generation of the specified sequence group.

        We use a worst-case heuristic: assume each touched block will require a
        new allocation (either via CoW or new block). We can append slots if the
        number of touched blocks is less than the number of free blocks.

        "Lookahead slots" are slots that are allocated in addition to the slots
        for known tokens. The contents of the lookahead slots are not defined.
        This is used by speculative decoding when speculating future tokens.
        """

        num_touched_blocks = 0
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            block_table = self.block_tables[seq.seq_id]

            num_touched_blocks += (
                block_table.get_num_blocks_touched_by_append_slots(
                    token_ids=block_table.get_unseen_token_ids(
                        seq.get_token_ids()),
                    num_lookahead_slots=num_lookahead_slots,
                ))

        num_free_gpu_blocks = self.block_allocator.get_num_free_blocks(
            Device.GPU)
        return num_touched_blocks <= num_free_gpu_blocks

    def append_slots(
        self,
        seq: Sequence,
        num_lookahead_slots: int,
    ) -> List[Tuple[int, int]]:

        block_table = self.block_tables[seq.seq_id]

        block_table.append_token_ids(
            token_ids=block_table.get_unseen_token_ids(seq.get_token_ids()),
            num_lookahead_slots=num_lookahead_slots,
            num_computed_slots=seq.data.get_num_computed_tokens(),
        )
        # Return any new copy-on-writes.
        new_cows = self.block_allocator.clear_copy_on_writes()
        return new_cows

    def free(self, seq: Sequence) -> None:
        seq_id = seq.seq_id

        if seq_id not in self.block_tables:
            # Already freed or haven't been scheduled yet.
            return

        # Update seq block ids with the latest access time
        self._last_access_blocks_tracker.update_seq_blocks_last_access(
            seq_id, self.block_tables[seq.seq_id].physical_block_ids)

        # Untrack seq
        self._last_access_blocks_tracker.remove_seq(seq_id)
        self._computed_blocks_tracker.remove_seq(seq_id)

        # Free table/blocks
        self.block_tables[seq_id].free()
        del self.block_tables[seq_id]

    def free_cross(self, seq_group: SequenceGroup) -> None:
        request_id = seq_group.request_id
        if request_id not in self.cross_block_tables:
            # Already freed or hasn't been scheduled yet.
            return
        self.cross_block_tables[request_id].free()
        del self.cross_block_tables[request_id]

    def get_block_table(self, seq: Sequence) -> List[int]:
        block_ids = self.block_tables[seq.seq_id].physical_block_ids
        return block_ids  # type: ignore

    def get_cross_block_table(self, seq_group: SequenceGroup) -> List[int]:
        request_id = seq_group.request_id
        assert request_id in self.cross_block_tables
        block_ids = self.cross_block_tables[request_id].physical_block_ids
        assert all(b is not None for b in block_ids)
        return block_ids  # type: ignore

    def access_all_blocks_in_seq(self, seq: Sequence, now: float):
        if self.enable_caching:
            # Record the latest access time for the sequence. The actual update
            # of the block ids is deferred to the sequence free(..) call, since
            # only during freeing of block ids, the blocks are actually added to
            # the evictor (which is when the most updated time is required)
            # (This avoids expensive calls to mark_blocks_as_accessed(..))
            self._last_access_blocks_tracker.update_last_access(
                seq.seq_id, now)

    def mark_blocks_as_computed(self, seq_group: SequenceGroup,
                                token_chunk_size: int):
        # If prefix caching is enabled, mark immutable blocks as computed
        # right after they have been scheduled (for prefill). This assumes
        # the scheduler is synchronous so blocks are actually computed when
        # scheduling the next batch.
        self.block_allocator.mark_blocks_as_computed([])

    def get_common_computed_block_ids(
            self, seqs: List[Sequence]) -> GenericSequence[int]:
        """Determine which blocks for which we skip prefill.

        With prefix caching we can skip prefill for previously-generated blocks.
        Currently, the attention implementation only supports skipping cached
        blocks if they are a contiguous prefix of cached blocks.

        This method determines which blocks can be safely skipped for all
        sequences in the sequence group.
        """
        computed_seq_block_ids = []
        for seq in seqs:
            computed_seq_block_ids.append(
                self._computed_blocks_tracker.
                get_cached_computed_blocks_and_update(
                    seq.seq_id,
                    self.block_tables[seq.seq_id].physical_block_ids))

        # NOTE(sang): This assumes seq_block_ids doesn't contain any None.
        return self.block_allocator.get_common_computed_block_ids(
            computed_seq_block_ids)  # type: ignore

    def get_content_hashes(self, seq: Sequence) -> List[bytes]:
        return self.block_tables[seq.seq_id].get_content_hashes()

    def get_and_reset_prefix_swaps(
            self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Return scheduler-owned (CPU->GPU, GPU->CPU) content transfers."""
        return self.block_allocator.get_and_reset_prefix_swaps()

    def begin_prefix_cache_step(self) -> None:
        self.block_allocator.begin_prefix_cache_step()

    def _build_runtime_cache_namespace(self) -> bytes:
        """Bind first-block hashes to the fixed model runtime identity."""
        model = os.getenv("BI100_PREFIX_MODEL_FINGERPRINT",
                          "Qwen3.6-35B-A3B")
        dtype = os.getenv("BI100_PREFIX_DTYPE", "float16")
        tp_raw = os.getenv("BI100_PREFIX_TP_SIZE", "4")
        try:
            tp_size = int(tp_raw)
        except ValueError as exc:
            raise RuntimeError(
                "BI100_PREFIX_TP_SIZE must be a positive integer") from exc
        if tp_size <= 0:
            raise RuntimeError(
                "BI100_PREFIX_TP_SIZE must be a positive integer")

        digest = hashlib.sha256()
        digest.update(b"bi100-runtime-prefix-identity-v1|")
        for label, value in (
                (b"model", model),
                (b"dtype", dtype),
                (b"tp", str(tp_size)),
                (b"block_size", str(self.block_size))):
            encoded = value.encode("utf-8")
            digest.update(label)
            digest.update(struct.pack("!Q", len(encoded)))
            digest.update(encoded)
        return digest.digest()

    @staticmethod
    def _adapter_cache_namespace(seq_group: SequenceGroup) -> bytes:
        digest = hashlib.sha256()
        digest.update(b"bi100-adapter-prefix-identity-v1|")
        lora = getattr(seq_group, "lora_request", None)
        prompt_adapter = getattr(seq_group, "prompt_adapter_request", None)
        identities = (
            ("lora", lora, ("lora_name", "lora_int_id", "lora_path",
                             "base_model_name")),
            ("prompt", prompt_adapter,
             ("prompt_adapter_name", "prompt_adapter_id",
              "prompt_adapter_local_path",
              "prompt_adapter_num_virtual_tokens")),
        )
        for kind, adapter, fields in identities:
            digest.update(kind.encode("ascii"))
            if adapter is None:
                digest.update(b"none|")
                continue
            for field in fields:
                value = str(getattr(adapter, field, ""))
                encoded = value.encode("utf-8")
                digest.update(field.encode("ascii"))
                digest.update(struct.pack("!Q", len(encoded)))
                digest.update(encoded)
        return digest.digest()

    def _request_local_fallback_cache_namespace(self,
                                               request_id: str) -> bytes:
        namespace = self._request_local_namespace.get(request_id)
        if namespace is None:
            digest = hashlib.sha256()
            digest.update(b"multimodal-unsupported-request-local-v1|")
            digest.update(self._runtime_cache_namespace)
            digest.update(os.urandom(32))
            digest.update(request_id.encode("utf-8"))
            namespace = digest.digest()
            self._request_local_namespace[request_id] = namespace
        return namespace

    def release_request_cache_namespace(self, request_id: str) -> None:
        """Release request-local isolation state after request completion."""
        self._request_local_namespace.pop(request_id, None)
        self._warned_mm_namespace_requests.discard(request_id)

    def _hash_multi_modal_namespace(self, mm_data: Any) -> bytes:
        digest = hashlib.sha256()
        self._hash_multi_modal_obj(digest, mm_data)
        return digest.digest()

    @staticmethod
    def _sort_map_keys(mm_map: Mapping[Any, Any]) -> List[Any]:
        return sorted(mm_map.keys(), key=lambda key: repr(key))

    @classmethod
    def _hash_multi_modal_obj(cls, digest: Any, value: Any) -> None:
        if value is None:
            digest.update(b"none|")
            return
        if isinstance(value, Mapping):
            digest.update(b"map|")
            digest.update(struct.pack("!Q", len(value)))
            for key in cls._sort_map_keys(value):
                digest.update(b"k|")
                cls._hash_multi_modal_obj(digest, key)
                digest.update(b"v|")
                cls._hash_multi_modal_obj(digest, value[key])
            return
        if isinstance(value, list):
            digest.update(b"list|")
            digest.update(struct.pack("!Q", len(value)))
            for item in value:
                cls._hash_multi_modal_obj(digest, item)
            return
        if isinstance(value, tuple):
            digest.update(b"tuple|")
            digest.update(struct.pack("!Q", len(value)))
            for item in value:
                cls._hash_multi_modal_obj(digest, item)
            return
        if isinstance(value, str):
            encoded = value.encode()
            digest.update(b"str|")
            digest.update(struct.pack("!Q", len(encoded)))
            digest.update(encoded)
            return
        if isinstance(value, bytes):
            digest.update(b"bytes|")
            digest.update(struct.pack("!Q", len(value)))
            digest.update(value)
            return
        if isinstance(value, bytearray):
            cls._hash_multi_modal_obj(digest, bytes(value))
            return
        if isinstance(value, bool):
            digest.update(b"bool|")
            digest.update(b"1" if value else b"0")
            return
        if isinstance(value, int):
            digest.update(b"int|")
            digest.update(str(value).encode())
            return
        if isinstance(value, float):
            digest.update(b"float|")
            digest.update(struct.pack("!d", value))
            return
        if torch is not None and isinstance(value, torch.Tensor):
            digest.update(b"tensor|")
            tensor = value.detach().cpu().contiguous()
            digest.update(struct.pack("!Q", len(tensor.shape)))
            for dim in tensor.shape:
                digest.update(struct.pack("!Q", int(dim)))
            digest.update(str(tensor.dtype).encode())
            # Byte views work for bfloat16 and other dtypes that NumPy cannot
            # materialize directly.
            tensor_bytes = tensor.view(torch.uint8).numpy().tobytes()
            digest.update(struct.pack("!Q", len(tensor_bytes)))
            digest.update(tensor_bytes)
            return
        if Image is not None and isinstance(value, Image.Image):
            digest.update(b"image|")
            digest.update(value.mode.encode())
            digest.update(struct.pack("!II", value.width, value.height))
            image_bytes = value.tobytes()
            digest.update(struct.pack("!Q", len(image_bytes)))
            digest.update(image_bytes)
            palette = value.getpalette()
            digest.update(b"palette-mode|")
            cls._hash_multi_modal_obj(
                digest, getattr(getattr(value, "palette", None), "mode", None))
            digest.update(b"palette|")
            cls._hash_multi_modal_obj(digest, palette)
            digest.update(b"transparency|")
            cls._hash_multi_modal_obj(
                digest, value.info.get("transparency"))
            return

        raise TypeError(f"Unsupported multimodal namespace value type {type(value)}")


    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        if parent_seq.seq_id not in self.block_tables:
            # Parent sequence has either been freed or never existed.
            return
        src_block_table = self.block_tables[parent_seq.seq_id]
        self.block_tables[child_seq.seq_id] = src_block_table.fork()

        # Track child seq
        self._computed_blocks_tracker.add_seq(child_seq.seq_id)
        self._last_access_blocks_tracker.add_seq(child_seq.seq_id)

    def can_swap_in(self, seq_group: SequenceGroup,
                    num_lookahead_slots: int) -> AllocStatus:
        """Returns the AllocStatus for the given sequence_group 
        with num_lookahead_slots.

        Args:
            sequence_group (SequenceGroup): The sequence group to swap in.
            num_lookahead_slots (int): Number of lookahead slots used in 
                speculative decoding, default to 0.

        Returns:
            AllocStatus: The AllocStatus for the given sequence group.
        """
        if self.block_allocator.content_offload_enabled:
            return AllocStatus.NEVER
        return self._can_swap(seq_group, Device.GPU, SequenceStatus.SWAPPED,
                              num_lookahead_slots)

    def swap_in(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
        """Returns the block id mapping (from CPU to GPU) generated by
        swapping in the given seq_group with num_lookahead_slots.

        Args:
            seq_group (SequenceGroup): The sequence group to swap in.

        Returns:
            List[Tuple[int, int]]: The mapping of swapping block from CPU 
                to GPU.
        """
        physical_block_id_mapping = []
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            blocks = self.block_tables[seq.seq_id].blocks
            if len(blocks) == 0:
                continue

            seq_swap_mapping = self.block_allocator.swap(blocks=blocks,
                                                         src_device=Device.CPU,
                                                         dst_device=Device.GPU)

            # Refresh the block ids of the table (post-swap)
            self.block_tables[seq.seq_id].update(blocks)

            seq_physical_block_id_mapping = {
                self.block_allocator.get_physical_block_id(
                    Device.CPU, cpu_block_id):
                self.block_allocator.get_physical_block_id(
                    Device.GPU, gpu_block_id)
                for cpu_block_id, gpu_block_id in seq_swap_mapping.items()
            }

            physical_block_id_mapping.extend(
                list(seq_physical_block_id_mapping.items()))

        return physical_block_id_mapping

    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        """Returns whether we can swap out the given sequence_group 
        with num_lookahead_slots.

        Args:
            seq_group (SequenceGroup): The sequence group to swap in.
            num_lookahead_slots (int): Number of lookahead slots used in 
                speculative decoding, default to 0.

        Returns:
            bool: Whether it's possible to swap out current sequence group.
        """
        if self.block_allocator.content_offload_enabled:
            return False
        alloc_status = self._can_swap(seq_group, Device.CPU,
                                      SequenceStatus.RUNNING)
        return alloc_status == AllocStatus.OK

    def swap_out(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
        """Returns the block id mapping (from GPU to CPU) generated by
        swapping out the given sequence_group with num_lookahead_slots.

        Args:
            sequence_group (SequenceGroup): The sequence group to swap in.

        Returns:
            List[Tuple[int, int]]: The mapping of swapping block from 
                GPU to CPU.
        """
        physical_block_id_mapping = []
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            blocks = self.block_tables[seq.seq_id].blocks
            if len(blocks) == 0:
                continue

            seq_swap_mapping = self.block_allocator.swap(blocks=blocks,
                                                         src_device=Device.GPU,
                                                         dst_device=Device.CPU)

            # Refresh the block ids of the table (post-swap)
            self.block_tables[seq.seq_id].update(blocks)

            seq_physical_block_id_mapping = {
                self.block_allocator.get_physical_block_id(
                    Device.GPU, gpu_block_id):
                self.block_allocator.get_physical_block_id(
                    Device.CPU, cpu_block_id)
                for gpu_block_id, cpu_block_id in seq_swap_mapping.items()
            }

            physical_block_id_mapping.extend(
                list(seq_physical_block_id_mapping.items()))

        return physical_block_id_mapping

    def get_num_free_gpu_blocks(self) -> int:
        return self.block_allocator.get_num_free_blocks(Device.GPU)

    def get_num_free_cpu_blocks(self) -> int:
        return self.block_allocator.get_num_free_blocks(Device.CPU)

    def get_prefix_cache_hit_rate(self, device: Device) -> float:
        return self.block_allocator.get_prefix_cache_hit_rate(device)

    def _can_swap(self,
                  seq_group: SequenceGroup,
                  device: Device,
                  status: SequenceStatus,
                  num_lookahead_slots: int = 0) -> AllocStatus:
        """Returns the AllocStatus for swapping in/out the given sequence_group 
        on to the 'device'.

        Args:
            sequence_group (SequenceGroup): The sequence group to swap in.
            device (Device): device to swap the 'seq_group' on.
            status (SequenceStatus): The status of sequence which is needed
                for action. RUNNING for swap out and SWAPPED for swap in
            num_lookahead_slots (int): Number of lookahead slots used in 
                speculative decoding, default to 0.

        Returns:
            AllocStatus: The AllocStatus for swapping in/out the given 
                sequence_group on to the 'device'.
        """
        # First determine the number of blocks that will be touched by this
        # swap. Then verify if there are available blocks in the device
        # to perform the swap.
        num_blocks_touched = 0
        blocks: List[Block] = []
        for seq in seq_group.get_seqs(status=status):
            block_table = self.block_tables[seq.seq_id]
            if block_table.blocks is not None:
                # Compute the number blocks to touch for the tokens to be
                # appended. This does NOT include the full blocks that need
                # to be touched for the swap.
                num_blocks_touched += \
                    block_table.get_num_blocks_touched_by_append_slots(
                        block_table.get_unseen_token_ids(seq.get_token_ids()),
                        num_lookahead_slots=num_lookahead_slots)
                blocks.extend(block_table.blocks)
        # Compute the number of full blocks to touch and add it to the
        # existing count of blocks to touch.
        num_blocks_touched += self.block_allocator.get_num_full_blocks_touched(
            blocks, device=device)

        watermark_blocks = 0
        if device == Device.GPU:
            watermark_blocks = self.watermark_blocks

        if self.block_allocator.get_num_total_blocks(
                device) < num_blocks_touched:
            return AllocStatus.NEVER
        elif self.block_allocator.get_num_free_blocks(
                device) - num_blocks_touched >= watermark_blocks:
            return AllocStatus.OK
        else:
            return AllocStatus.LATER
