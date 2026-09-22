#!/bin/bash
# build_xllm_kernels.sh — Compile xllm CUDA kernels into .so on BI-V100
#
# Uses corex CUB (/usr/local/corex/include/cub/) NOT cccl_upstream
# Each .so = kernel .cu + pybind11 binding .cpp
#
# Run: bash qwen3_6_scripts/build_xllm_kernels.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CUDA_DIR="${PROJECT_DIR}/ex_engine/xllm_kernels/cuda"
HEADER_DIR="${CUDA_DIR}/headers"
BIND_DIR="${CUDA_DIR}/bindings"
PREBUILT_DIR="${SCRIPT_DIR}/prebuilt/corex-3.2.3-ivcore10"
mkdir -p "$PREBUILT_DIR"

build_kernel() {
    local name="$1"
    shift
    local sources="$@"
    echo "=== Building ${name}.so ==="
    python3 -c "
import os, glob, shutil
from torch.utils.cpp_extension import load

sources = '${sources}'.split()
mod = load(
    name='${name}',
    sources=sources,
    extra_cflags=['-std=c++17'],
    extra_include_paths=['${HEADER_DIR}', '/usr/local/corex/include'],
    verbose=True,
)

import torch.utils.cpp_extension as ext
build_dir = ext._get_build_directory('${name}', verbose=False)
for f in glob.glob(os.path.join(build_dir, '*.so')):
    dst = '${PREBUILT_DIR}/${name}.so'
    shutil.copy2(f, dst)
    sz = os.path.getsize(dst)
    print(f'✓ ${name}.so ({sz} bytes) → {dst}')
    break

fns = [x for x in dir(mod) if not x.startswith('_')]
print(f'Functions: {fns}')
"
    echo ""
}

echo "Building xllm CUDA kernels for BI-V100 (ivcore10)"
echo "Using corex CUB: /usr/local/corex/include/cub/"
echo ""

build_kernel "xllm_norm" \
    "${CUDA_DIR}/norm.cu" "${BIND_DIR}/xllm_norm_bind.cpp"

build_kernel "xllm_activation" \
    "${CUDA_DIR}/activation.cu" "${BIND_DIR}/xllm_activation_bind.cpp"

build_kernel "xllm_rope" \
    "${CUDA_DIR}/rope.cu" "${BIND_DIR}/xllm_rope_bind.cpp"

build_kernel "xllm_cache" \
    "${CUDA_DIR}/reshape_paged_cache.cu" "${CUDA_DIR}/block_copy.cu" "${BIND_DIR}/xllm_cache_bind.cpp"

build_kernel "xllm_moe" \
    "${CUDA_DIR}/moe/moe_fused_topk.cu" "${CUDA_DIR}/moe/moe_compute_index.cu" "${CUDA_DIR}/moe/moe_combine.cu" "${BIND_DIR}/xllm_moe_bind.cpp"

echo "=== All kernels built ==="
ls -lh "${PREBUILT_DIR}"/xllm_*.so 2>/dev/null || echo "No .so files found"
