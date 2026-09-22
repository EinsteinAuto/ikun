"""CacheEngine class for managing the KV cache."""
from typing import List

import torch

from vllm.attention import get_attn_backend
from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig
from vllm.block_major_kv_cache import (
    BlockMajorCpuKVCache,
    block_major_cpu_kv_enabled,
)
from vllm.logger import init_logger
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, LayerBlockType,
                        get_dtype_size, is_pin_memory_available)

logger = init_logger(__name__)

print("=================patch_ok: cache_engine.py loaded===========")


class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.device_config = device_config

        self.head_size = model_config.get_head_size()
        # Models like Jamba, have mixed typed layers, E.g Mamba
        self.num_attention_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        if self.num_gpu_blocks:
            self.num_gpu_blocks //= parallel_config.pipeline_parallel_size
        self.num_cpu_blocks = cache_config.num_cpu_blocks
        if self.num_cpu_blocks:
            self.num_cpu_blocks //= parallel_config.pipeline_parallel_size

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Get attention backend.
        self.attn_backend = get_attn_backend(self.head_size,
                                             model_config.dtype,
                                             cache_config.cache_dtype,
                                             self.block_size,
                                             model_config.is_attention_free,
                                             use_mla=getattr(
                                                 model_config, 'use_mla',
                                                 False))

        # Initialize the cache.
        self.gpu_cache = self._allocate_kv_cache(
            self.num_gpu_blocks, self.device_config.device_type)
        self._bi100_block_major_cpu_kv = None
        if block_major_cpu_kv_enabled():
            self._bi100_block_major_cpu_kv = BlockMajorCpuKVCache(
                self.gpu_cache,
                self.num_cpu_blocks,
                pin_memory=is_pin_memory_available(),
            )
            self.cpu_cache = self._bi100_block_major_cpu_kv.layer_views
        else:
            self.cpu_cache = self._allocate_kv_cache(
                self.num_cpu_blocks, "cpu")

    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device.

        CCCL temporary_storage.cuh layout<SlotsCount> system design:
          Phase 1 (get_size): compute total bytes for all slots
          Phase 2 (map_to_buffer): allocate one blob, alias into slots

        Applied: instead of N separate torch.zeros (one per layer),
        compute total size → allocate one contiguous tensor → slice
        into per-layer views. Reduces cudaMalloc calls from
        num_attention_layers to 1, and guarantees cross-layer memory
        contiguity (better L2 locality for multi-layer KV access).

        The slot/alias pattern maps directly:
          layout slot[i] = layer i's KV cache
          alias<T> = the typed view into that layer's region
        """
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        pin_memory = is_pin_memory_available() if device == "cpu" else False
        kv_cache: List[torch.Tensor] = []

        if self.num_attention_layers == 0 or num_blocks == 0:
            return kv_cache

        # Phase 1: get_size — compute per-layer element count
        import math
        layer_numel = math.prod(kv_cache_shape)

        # Phase 2: map_to_buffer — single contiguous allocation
        total_numel = self.num_attention_layers * layer_numel
        contiguous_buffer = torch.zeros(
            total_numel,
            dtype=self.dtype,
            pin_memory=pin_memory,
            device=device)

        # Alias into per-layer views (CCCL slot.create_alias pattern)
        for i in range(self.num_attention_layers):
            start = i * layer_numel
            layer_flat = contiguous_buffer[start:start + layer_numel]
            kv_cache.append(layer_flat.view(kv_cache_shape))

        return kv_cache

    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        if self._bi100_block_major_cpu_kv is not None:
            self._bi100_block_major_cpu_kv.swap_in(src_to_dst)
            return
        for i in range(self.num_attention_layers):
            self.attn_backend.swap_blocks(self.cpu_cache[i], self.gpu_cache[i],
                                          src_to_dst)

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        if self._bi100_block_major_cpu_kv is not None:
            self._bi100_block_major_cpu_kv.swap_out(src_to_dst)
            return
        for i in range(self.num_attention_layers):
            self.attn_backend.swap_blocks(self.gpu_cache[i], self.cpu_cache[i],
                                          src_to_dst)

    def copy(self, src_to_dsts: torch.Tensor) -> None:
        self.attn_backend.copy_blocks(self.gpu_cache, src_to_dsts)

    @staticmethod
    def get_cache_block_size(
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_attention_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)

        key_cache_block = cache_config.block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_attention_layers * (key_cache_block + value_cache_block)
        if cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        dtype_size = get_dtype_size(dtype)
        return dtype_size * total
