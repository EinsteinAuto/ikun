from __future__ import annotations

import os
import time
from collections.abc import Mapping

import torch

from vllm.logger import init_logger


logger = init_logger(__name__)

ENABLE_ENV = "BI100_BLOCK_MAJOR_CPU_KV"
TRACE_ENV = "BI100_BLOCK_MAJOR_CPU_KV_TRACE"
CPU_OFFLOAD_ENV = "BI100_CPU_KV_OFFLOAD"
HYBRID_ACCOUNTING_ENV = "BI100_HYBRID_KV_ACCOUNTING"
NUM_ATTENTION_LAYERS = 10
KV_PLANES = 2
ELEMENTS_PER_PLANE_BLOCK = 4096
STAGING_BLOCKS = 512
STAGING_BUFFER_COUNT = 2
BYTES_PER_BLOCK = (
    NUM_ATTENTION_LAYERS * KV_PLANES * ELEMENTS_PER_PLANE_BLOCK * 2
)
GPU_STAGING_BYTES = STAGING_BLOCKS * STAGING_BUFFER_COUNT * BYTES_PER_BLOCK


def _strict_binary_selector(
    name: str,
    environ: Mapping[str, str] | None = None,
) -> bool:
    source = os.environ if environ is None else environ
    raw = source.get(name, "0")
    if raw == "0":
        return False
    if raw == "1":
        return True
    raise RuntimeError(f"{name} must be exactly '0' or '1', got {raw!r}")


def block_major_cpu_kv_enabled(
    environ: Mapping[str, str] | None = None,
) -> bool:
    return _strict_binary_selector(ENABLE_ENV, environ)


def block_major_cpu_kv_trace_enabled(
    environ: Mapping[str, str] | None = None,
) -> bool:
    return _strict_binary_selector(TRACE_ENV, environ)


def _require_block_major_runtime(
    environ: Mapping[str, str] | None = None,
) -> None:
    source = os.environ if environ is None else environ
    if source.get(CPU_OFFLOAD_ENV, "0") != "1":
        raise RuntimeError(
            f"{ENABLE_ENV}=1 requires {CPU_OFFLOAD_ENV}=1")
    if source.get(HYBRID_ACCOUNTING_ENV, "legacy40") != "full_attention":
        raise RuntimeError(
            f"{ENABLE_ENV}=1 requires "
            f"{HYBRID_ACCOUNTING_ENV}=full_attention")


def reserve_block_major_gpu_blocks(
    num_gpu_blocks: int,
    cache_block_size: int,
    environ: Mapping[str, str] | None = None,
) -> int:
    if (not isinstance(num_gpu_blocks, int)
            or isinstance(num_gpu_blocks, bool)
            or num_gpu_blocks < 0):
        raise ValueError("num_gpu_blocks must be a non-negative integer")
    if not block_major_cpu_kv_enabled(environ):
        return num_gpu_blocks

    _require_block_major_runtime(environ)
    if cache_block_size != BYTES_PER_BLOCK:
        raise RuntimeError(
            f"{ENABLE_ENV}=1 requires cache block size "
            f"{BYTES_PER_BLOCK}, got {cache_block_size}")
    reserved_blocks = (
        GPU_STAGING_BYTES + cache_block_size - 1
    ) // cache_block_size
    remaining_blocks = num_gpu_blocks - reserved_blocks
    if remaining_blocks <= 0:
        raise RuntimeError(
            "block-major GPU staging leaves no usable GPU KV blocks")
    logger.info(
        "[BI100 BLOCK KV] capacity reserve blocks=%d bytes=%d "
        "profiled_blocks=%d usable_blocks=%d",
        reserved_blocks,
        GPU_STAGING_BYTES,
        num_gpu_blocks,
        remaining_blocks,
    )
    return remaining_blocks


def validate_block_mapping(
    mapping: torch.Tensor,
    source_limit: int,
    destination_limit: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(mapping, torch.Tensor):
        raise TypeError("block mapping must be a torch.Tensor")
    if mapping.device.type != "cpu":
        raise ValueError("block mapping must be on CPU")
    if mapping.dtype != torch.int64:
        raise ValueError("block mapping must use torch.int64")
    if not mapping.is_contiguous():
        raise ValueError("block mapping must be contiguous")
    if mapping.dim() != 2 or mapping.shape[1] != 2:
        raise ValueError("block mapping must have shape [N, 2]")
    if source_limit <= 0 or destination_limit <= 0:
        raise ValueError("block mapping limits must be positive")

    sources: set[int] = set()
    destinations: set[int] = set()
    for row, pair in enumerate(mapping.tolist()):
        source, destination = pair
        if not 0 <= source < source_limit:
            raise ValueError(
                f"source block out of range at row {row}: {source}")
        if not 0 <= destination < destination_limit:
            raise ValueError(
                f"destination block out of range at row {row}: "
                f"{destination}")
        if source in sources:
            raise ValueError(f"duplicate source block: {source}")
        if destination in destinations:
            raise ValueError(f"duplicate destination block: {destination}")
        sources.add(source)
        destinations.add(destination)

    return mapping[:, 0].contiguous(), mapping[:, 1].contiguous()


class BlockMajorCpuKVCache:

    def __init__(
        self,
        gpu_cache: list[torch.Tensor],
        num_cpu_blocks: int,
        pin_memory: bool,
    ) -> None:
        self._validate_gpu_cache(gpu_cache)
        if block_major_cpu_kv_enabled():
            _require_block_major_runtime()
        if num_cpu_blocks <= 0:
            raise RuntimeError(
                f"{ENABLE_ENV}=1 requires a positive CPU block count")
        if not pin_memory:
            raise RuntimeError(
                f"{ENABLE_ENV}=1 requires pinned CPU memory")

        try:
            from vllm import corex_block_major_kv_transfer as extension
        except ImportError as exc:
            raise RuntimeError(
                "block-major CoreX extension is unavailable") from exc

        self.extension = extension
        self.gpu_cache = gpu_cache
        self.device = gpu_cache[0].device
        self.dtype = gpu_cache[0].dtype
        self.num_gpu_blocks = gpu_cache[0].shape[1]
        self.num_cpu_blocks = num_cpu_blocks
        self.trace_enabled = block_major_cpu_kv_trace_enabled()

        self.cpu_pool = torch.zeros(
            (
                num_cpu_blocks,
                NUM_ATTENTION_LAYERS,
                KV_PLANES,
                ELEMENTS_PER_PLANE_BLOCK,
            ),
            dtype=self.dtype,
            device="cpu",
            pin_memory=True,
        )
        if not self.cpu_pool.is_pinned():
            raise RuntimeError("block-major CPU pool is not pinned")

        # Preserve the public CacheEngine shape without allocating a second
        # layer-major CPU cache. Transfer methods use cpu_pool directly.
        self.layer_views = [
            self.cpu_pool[:, layer, :, :].permute(1, 0, 2)
            for layer in range(NUM_ATTENTION_LAYERS)
        ]
        self.cpu_staging = [
            torch.empty(
                (
                    STAGING_BLOCKS,
                    NUM_ATTENTION_LAYERS,
                    KV_PLANES,
                    ELEMENTS_PER_PLANE_BLOCK,
                ),
                dtype=self.dtype,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(STAGING_BUFFER_COUNT)
        ]
        if not all(staging.is_pinned() for staging in self.cpu_staging):
            raise RuntimeError("block-major CPU staging is not pinned")

        with torch.cuda.device(self.device):
            self.gpu_staging = [
                torch.empty_like(staging, device=self.device)
                for staging in self.cpu_staging
            ]
            self.events = [
                torch.cuda.Event(enable_timing=False)
                for _ in range(STAGING_BUFFER_COUNT)
            ]
            self.error_flag = torch.zeros(
                1, dtype=torch.int32, device=self.device)

        logger.info(
            "[BI100 BLOCK KV] enabled device=%s gpu_blocks=%d cpu_blocks=%d "
            "layers=%d block_bytes=%d staging_blocks=%d staging_buffers=%d",
            self.device,
            self.num_gpu_blocks,
            self.num_cpu_blocks,
            NUM_ATTENTION_LAYERS,
            BYTES_PER_BLOCK,
            STAGING_BLOCKS,
            STAGING_BUFFER_COUNT,
        )

    @staticmethod
    def _validate_gpu_cache(gpu_cache: list[torch.Tensor]) -> None:
        if len(gpu_cache) != NUM_ATTENTION_LAYERS:
            raise RuntimeError(
                f"{ENABLE_ENV}=1 requires exactly "
                f"{NUM_ATTENTION_LAYERS} GPU attention caches, got "
                f"{len(gpu_cache)}")
        first = gpu_cache[0]
        if first.device.type != "cuda":
            raise RuntimeError("block-major GPU cache must be on CUDA")
        if first.dtype != torch.float16:
            raise RuntimeError("block-major GPU cache must use float16")
        if (first.dim() != 3 or first.shape[0] != KV_PLANES
                or first.shape[2] != ELEMENTS_PER_PLANE_BLOCK):
            raise RuntimeError(
                "block-major GPU cache must have shape [2, blocks, 4096]")
        if not first.is_contiguous():
            raise RuntimeError("block-major GPU cache must be contiguous")

        for layer, tensor in enumerate(gpu_cache):
            if tensor.device != first.device:
                raise RuntimeError(
                    f"GPU cache layer {layer} is on a different device")
            if tensor.dtype != first.dtype or tensor.shape != first.shape:
                raise RuntimeError(
                    f"GPU cache layer {layer} has inconsistent geometry")
            if not tensor.is_contiguous():
                raise RuntimeError(
                    f"GPU cache layer {layer} is not contiguous")

    def _to_gpu_ids(self, block_ids: torch.Tensor) -> torch.Tensor:
        return block_ids.to(
            device=self.device,
            dtype=torch.int32,
            non_blocking=False,
        )

    @staticmethod
    def _chunks(
        source: torch.Tensor,
        destination: torch.Tensor,
        gpu_ids: torch.Tensor,
    ):
        for start in range(0, source.numel(), STAGING_BLOCKS):
            end = min(start + STAGING_BLOCKS, source.numel())
            yield (
                source[start:end],
                destination[start:end],
                gpu_ids[start:end],
                end - start,
            )

    def _begin(self) -> None:
        self.error_flag.zero_()

    def _finish(
        self,
        direction: str,
        block_count: int,
        started: float | None,
    ) -> None:
        # check_error performs the final stream synchronization. This also
        # makes every staging slot safe to reuse in the next CacheEngine call.
        self.extension.check_error(self.error_flag)
        if started is not None:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            logger.info(
                "[BI100 BLOCK KV TRACE] direction=%s blocks=%d bytes=%d "
                "elapsed_ms=%.3f",
                direction,
                block_count,
                block_count * BYTES_PER_BLOCK,
                elapsed_ms,
            )

    def swap_out(self, mapping: torch.Tensor) -> None:
        started = time.perf_counter() if self.trace_enabled else None
        source_gpu, destination_cpu = validate_block_mapping(
            mapping,
            source_limit=self.num_gpu_blocks,
            destination_limit=self.num_cpu_blocks,
        )
        block_count = source_gpu.numel()
        if block_count == 0:
            return
        source_gpu_ids = self._to_gpu_ids(source_gpu)

        self._begin()
        pending: tuple[int, torch.Tensor, int] | None = None
        for index, (_, destination, gpu_ids, count) in enumerate(
                self._chunks(
                    source_gpu, destination_cpu, source_gpu_ids)):
            slot = index % STAGING_BUFFER_COUNT
            self.extension.pack(
                self.gpu_cache,
                gpu_ids,
                self.gpu_staging[slot],
                self.error_flag,
                count,
            )
            self.cpu_staging[slot][:count].copy_(
                self.gpu_staging[slot][:count],
                non_blocking=True,
            )
            self.events[slot].record()
            if pending is not None:
                pending_slot, pending_destination, pending_count = pending
                self.events[pending_slot].synchronize()
                self.extension.cpu_scatter(
                    self.cpu_staging[pending_slot],
                    self.cpu_pool,
                    pending_destination,
                    pending_count,
                )
            pending = (slot, destination, count)

        if pending is not None:
            pending_slot, pending_destination, pending_count = pending
            self.events[pending_slot].synchronize()
            self.extension.cpu_scatter(
                self.cpu_staging[pending_slot],
                self.cpu_pool,
                pending_destination,
                pending_count,
            )
        self._finish("d2h", block_count, started)

    def swap_in(self, mapping: torch.Tensor) -> None:
        started = time.perf_counter() if self.trace_enabled else None
        source_cpu, destination_gpu = validate_block_mapping(
            mapping,
            source_limit=self.num_cpu_blocks,
            destination_limit=self.num_gpu_blocks,
        )
        block_count = source_cpu.numel()
        if block_count == 0:
            return
        destination_gpu_ids = self._to_gpu_ids(destination_gpu)

        self._begin()
        for index, (source, _, gpu_ids, count) in enumerate(
                self._chunks(
                    source_cpu, destination_gpu, destination_gpu_ids)):
            slot = index % STAGING_BUFFER_COUNT
            if index >= STAGING_BUFFER_COUNT:
                self.events[slot].synchronize()
            self.extension.cpu_gather(
                self.cpu_pool,
                source,
                self.cpu_staging[slot],
                count,
            )
            self.gpu_staging[slot][:count].copy_(
                self.cpu_staging[slot][:count],
                non_blocking=True,
            )
            self.extension.scatter(
                self.gpu_staging[slot],
                gpu_ids,
                self.gpu_cache,
                self.error_flag,
                count,
            )
            self.events[slot].record()
        self._finish("h2d", block_count, started)
