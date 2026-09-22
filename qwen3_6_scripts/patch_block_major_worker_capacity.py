from patch_utils import package_root, replace_once


WORKER = package_root("vllm") / "worker" / "worker.py"

IMPORT_ANCHOR = """\
from vllm.logger import init_logger
"""

IMPORT_REPLACEMENT = """\
from vllm.block_major_kv_cache import reserve_block_major_gpu_blocks
from vllm.logger import init_logger
"""

CAPACITY_ANCHOR = """\
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)
"""

CAPACITY_REPLACEMENT = """\
        num_gpu_blocks = reserve_block_major_gpu_blocks(
            num_gpu_blocks, cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)
"""


replace_once(
    WORKER,
    IMPORT_ANCHOR,
    IMPORT_REPLACEMENT,
    required=True,
    already_contains=(
        "from vllm.block_major_kv_cache import "
        "reserve_block_major_gpu_blocks"
    ),
)
replace_once(
    WORKER,
    CAPACITY_ANCHOR,
    CAPACITY_REPLACEMENT,
    required=True,
    already_contains=(
        "num_gpu_blocks = reserve_block_major_gpu_blocks("
    ),
)
