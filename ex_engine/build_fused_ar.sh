#!/usr/bin/env bash
# build_fused_ar.sh — Compile ix_fused_linear_allreduce.so on BI-V100
#
# Produces ix_full_bridge_fused_ar.so — pybind11 module with:
#   linear_allreduce(input, weight, bias) = allreduce(input @ weight.T + bias)
#
# Links against:
#   - _ixformer_torch.so (for ixformer_linear / ixformer_linear_ex)
#   - libnccl.so (for NCCL all-reduce)
#   - libtorch / libc10 (PyTorch C++ API)
#
# Usage:
#   bash ex_engine/build_fused_ar.sh [VLLM_ROOT]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSRC_DIR="${SCRIPT_DIR}/csrc"
VLLM_ROOT="${1:-}"

# --- Compiler ---
COREX_ROOT="${COREX_ROOT:-/usr/local/corex}"
CLANGXX="${COREX_ROOT}/bin/clang++"
if [[ ! -x "$CLANGXX" ]]; then
    CLANGXX=$(command -v clang++ 2>/dev/null || true)
fi
if [[ -z "$CLANGXX" ]]; then
    echo "[fused_ar] ERROR: clang++ not found" >&2
    exit 1
fi

# --- Python / Torch ---
PYTHON="${PYTHON:-python3}"
TORCH_INC=$($PYTHON -c "from torch.utils.cpp_extension import include_paths; print(' '.join(['-I'+p for p in include_paths()]))")
TORCH_LIB=$($PYTHON -c "from torch.utils.cpp_extension import library_paths; print(' '.join(['-L'+p for p in library_paths()]))")
PYTHON_INC=$($PYTHON -c "from sysconfig import get_paths; print('-I' + get_paths()['include'])")

# --- NCCL ---
NCCL_INC=""
NCCL_LIB=""
for d in "${COREX_ROOT}/include" "/usr/include" "/usr/local/include"; do
    if [[ -f "${d}/nccl.h" ]]; then
        NCCL_INC="-I${d}"
        break
    fi
done
for d in "${COREX_ROOT}/lib64" "${COREX_ROOT}/lib" "/usr/lib/x86_64-linux-gnu" "/usr/local/lib"; do
    if [[ -f "${d}/libnccl.so" ]] || [[ -f "${d}/libnccl.so.2" ]]; then
        NCCL_LIB="-L${d} -lnccl"
        break
    fi
done

# --- ixformer .so for linking ---
IX_LIBS=""
for sopath in \
    "${COREX_ROOT}/lib/python3/dist-packages/ixformer"/_ixformer_torch*.so \
    "${COREX_ROOT}/lib64/python3/dist-packages/ixformer"/_ixformer_torch*.so; do
    if [[ -f "$sopath" ]]; then
        IX_LIBS="${IX_LIBS} ${sopath}"
    fi
done

# --- rpath ---
RPATH_DIRS=""
for d in \
    "${COREX_ROOT}/lib64" \
    "${COREX_ROOT}/lib/python3/dist-packages/ixformer" \
    "${COREX_ROOT}/lib64/python3/dist-packages/ixformer"; do
    if [[ -d "$d" ]]; then
        RPATH_DIRS="${RPATH_DIRS} -Wl,-rpath,${d}"
    fi
done

# --- Compile ---
SRC="${CSRC_DIR}/ix_fused_linear_allreduce.cpp"
OUTPUT_DIR="${SCRIPT_DIR}/prebuilt"
mkdir -p "$OUTPUT_DIR"
OUTPUT="${OUTPUT_DIR}/ix_full_bridge_fused_ar.so"

echo "[fused_ar] Compiler:  ${CLANGXX}"
echo "[fused_ar] Source:    ${SRC}"
echo "[fused_ar] NCCL inc:  ${NCCL_INC:-not found}"
echo "[fused_ar] NCCL lib:  ${NCCL_LIB:-not found}"
echo "[fused_ar] IX libs:   ${IX_LIBS:-not found}"

$CLANGXX \
    -shared -fPIC -O2 -std=c++17 \
    -DTORCH_EXTENSION_NAME=ix_full_bridge_fused_ar \
    $PYTHON_INC \
    $TORCH_INC \
    $TORCH_LIB \
    ${NCCL_INC} \
    -ltorch -ltorch_cpu -ltorch_python -lc10 \
    ${NCCL_LIB} \
    ${IX_LIBS} \
    ${RPATH_DIRS} \
    -o "$OUTPUT" \
    "$SRC"

echo "[fused_ar] Built: ${OUTPUT}"
ls -lh "$OUTPUT"

# --- Verify ---
echo "[fused_ar] Symbols:"
nm -D "$OUTPUT" 2>/dev/null | grep -i "linear_allreduce\|PyInit" | head -5

# --- Deploy ---
if [[ -n "$VLLM_ROOT" ]] && [[ -d "$VLLM_ROOT" ]]; then
    mkdir -p "${VLLM_ROOT}/ex_engine"
    cp "$OUTPUT" "${VLLM_ROOT}/ex_engine/ix_full_bridge_fused_ar.so"
    echo "[fused_ar] Deployed to ${VLLM_ROOT}/ex_engine/"
fi

echo "[fused_ar] Done"
