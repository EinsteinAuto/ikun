from patch_utils import package_root, replace_once


CACHE_ENGINE = package_root("vllm") / "worker" / "cache_engine.py"

IMPORT_ANCHOR = """\
from vllm.logger import init_logger
"""

IMPORT_REPLACEMENT = """\
from vllm.block_major_kv_cache import (
    BlockMajorCpuKVCache,
    block_major_cpu_kv_enabled,
)
from vllm.logger import init_logger
"""

ALLOCATION_ANCHOR = """\
        self.gpu_cache = self._allocate_kv_cache(
            self.num_gpu_blocks, self.device_config.device_type)
        self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks, "cpu")
"""

ALLOCATION_REPLACEMENT = """\
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
"""

SWAP_ANCHOR = """\
    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        for i in range(self.num_attention_layers):
            self.attn_backend.swap_blocks(self.cpu_cache[i], self.gpu_cache[i],
                                          src_to_dst)

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        for i in range(self.num_attention_layers):
            self.attn_backend.swap_blocks(self.gpu_cache[i], self.cpu_cache[i],
                                          src_to_dst)
"""

SWAP_REPLACEMENT = """\
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
"""


replace_once(
    CACHE_ENGINE,
    IMPORT_ANCHOR,
    IMPORT_REPLACEMENT,
    required=True,
    already_contains="from vllm.block_major_kv_cache import",
)
replace_once(
    CACHE_ENGINE,
    ALLOCATION_ANCHOR,
    ALLOCATION_REPLACEMENT,
    required=True,
    already_contains="self._bi100_block_major_cpu_kv = None",
)
replace_once(
    CACHE_ENGINE,
    SWAP_ANCHOR,
    SWAP_REPLACEMENT,
    required=True,
    already_contains="self._bi100_block_major_cpu_kv.swap_in",
)
