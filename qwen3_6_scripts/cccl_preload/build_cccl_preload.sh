#!/usr/bin/env bash
# Build libcccl_allocator.so
#
# Full CCCL dependency chain (288 headers) in ./include/
# Source: cccl_upstream/cub/cub/util_allocator.cuh + transitive deps
#
# Usage:
#   bash build_cccl_preload.sh [output_dir]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:-${SCRIPT_DIR}}"
SRC="${SCRIPT_DIR}/cccl_allocator_preload.cu"
INC="${SCRIPT_DIR}/include"
OUT="${OUTPUT_DIR}/libcccl_allocator.so"

[[ -d "${INC}/cub" ]] || { echo "CCCL include tree missing: ${INC}/cub"; exit 2; }
[[ -d "${INC}/cuda" ]] || { echo "CCCL include tree missing: ${INC}/cuda"; exit 2; }

# Find compiler
CXX=""
for candidate in \
    /usr/local/corex-3.2.3/bin/clang++ \
    /usr/local/corex/bin/clang++ \
    /usr/local/corex/lib64/clang/16/bin/clang++ \
    ; do
    if [[ -x "${candidate}" ]]; then
        CXX="${candidate}"
        break
    fi
done
[[ -n "${CXX}" ]] || { CXX=g++; echo "[build] no CoreX clang++, falling back to g++"; }
echo "[build] CXX=${CXX}"

# Find CUDA headers (for cuda_runtime_api.h)
CUDA_INC=""
for candidate in \
    /usr/local/corex/include \
    /usr/local/cuda/include \
    ; do
    if [[ -f "${candidate}/cuda_runtime_api.h" ]]; then
        CUDA_INC="${candidate}"
        break
    fi
done

# Find CUDA libs
CUDA_LIB=""
for candidate in \
    /usr/local/corex/lib64 \
    /usr/local/cuda/lib64 \
    ; do
    if [[ -f "${candidate}/libcudart.so" ]]; then
        CUDA_LIB="${candidate}"
        break
    fi
done

echo "[build] CUDA include: ${CUDA_INC:-system}"
echo "[build] CUDA lib:     ${CUDA_LIB:-system}"
echo "[build] CCCL include: ${INC} ($(find "${INC}" -type f | wc -l) files)"
echo "[build] Source:        ${SRC}"
echo "[build] Output:        ${OUT}"

COMMON_FLAGS=(
    -shared -fPIC -O2 -std=c++17
    -I"${INC}"
    ${CUDA_INC:+-I"${CUDA_INC}"}
    ${CUDA_LIB:+-L"${CUDA_LIB}"}
    -lcudart -ldl
    # Suppress CCCL warnings that don't affect correctness
    -Wno-unused-function
    -Wno-unknown-pragmas
    # CUB needs this for non-NVCC compilers
    -D__CUDA_ARCH_LIST__=700
    -DCUB_DISABLE_NAMESPACE_MAGIC
    -DCUB_WRAPPED_NAMESPACE=cccl_preload
)

if [[ "${CXX}" == *clang++* ]]; then
    "${CXX}" "${COMMON_FLAGS[@]}" -x c++ -o "${OUT}" "${SRC}" 2>&1
else
    "${CXX}" "${COMMON_FLAGS[@]}" -x c++ -o "${OUT}" "${SRC}" 2>&1
fi

if [[ -f "${OUT}" ]]; then
    SIZE=$(stat -c%s "${OUT}" 2>/dev/null || echo "?")
    echo ""
    echo "[build] SUCCESS: ${OUT} (${SIZE} bytes)"
    echo ""
    echo "Test:"
    echo "  LD_PRELOAD=${OUT} CCCL_ALLOC_DEBUG=1 \\"
    echo "  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\"
    echo "  python3 verify_preload.py"
else
    echo "[build] FAILED"
    exit 1
fi
