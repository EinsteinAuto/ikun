#!/usr/bin/env bash
# build_ix_bridge.sh — Compile ix_full_bridge_v2.cpp on BI-V100
#
# Upstream ref: xllm/core/kernels/ilu/ixformer.h (all 14 C++ functions)
# Bridge ref:   ex_engine/csrc/ix_full_bridge_v2.cpp
#
# This produces ix_full_bridge_v2.so — a pybind11 module that exposes
# ALL ixformer::infer functions to Python without any Python fallbacks.
#
# Usage:
#   bash build_ix_bridge.sh [VLLM_ROOT]
#
# The .so is deployed to $VLLM_ROOT/ex_engine/ and also to
# ex_engine/prebuilt/ for the prebuilt pipeline.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSRC_DIR="${SCRIPT_DIR}/csrc"
VLLM_ROOT="${1:-}"

# --- Locate tools ---
COREX_ROOT="${COREX_ROOT:-/usr/local/corex}"
CLANGXX="${COREX_ROOT}/bin/clang++"
if [[ ! -x "$CLANGXX" ]]; then
    CLANGXX=$(command -v clang++ 2>/dev/null || true)
fi
if [[ -z "$CLANGXX" ]]; then
    echo "[ix_bridge] ERROR: clang++ not found" >&2
    exit 1
fi

# --- Locate torch and python ---
PYTHON="${PYTHON:-python3}"
TORCH_DIR=$($PYTHON -c "import torch; print(torch.utils.cmake_prefix_path)" 2>/dev/null || \
            $PYTHON -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'share', 'cmake'))" 2>/dev/null || true)
TORCH_INC=$($PYTHON -c "from torch.utils.cpp_extension import include_paths; print(' '.join(['-I'+p for p in include_paths()]))")
TORCH_LIB=$($PYTHON -c "from torch.utils.cpp_extension import library_paths; print(' '.join(['-L'+p for p in library_paths()]))")
PYTHON_INC=$($PYTHON -c "from sysconfig import get_paths; print('-I' + get_paths()['include'])")

# --- Locate ixformer .so files for linking ---
IX_LIBS=""
for sopath in \
    "${COREX_ROOT}/lib/python3/dist-packages/ixformer"/*.so \
    "${COREX_ROOT}/lib64/python3/dist-packages/ixformer"/*.so \
    /usr/local/lib/python3.10/dist-packages/ixformer/*.so; do
    if [[ -f "$sopath" ]]; then
        IX_LIBS="${IX_LIBS} ${sopath}"
    fi
done

# Also link against libixformer*.so in corex lib dirs
for sopath in \
    "${COREX_ROOT}/lib64"/libixformer*.so \
    "${COREX_ROOT}/lib64"/lib*ixformer*.so; do
    if [[ -f "$sopath" ]]; then
        IX_LIBS="${IX_LIBS} ${sopath}"
    fi
done

# Add ixformer_torch_ext if present
for sopath in \
    "${COREX_ROOT}/lib/python3/dist-packages/ixformer"/_ixformer_torch*.so \
    "${COREX_ROOT}/lib64/python3/dist-packages/ixformer"/_ixformer_torch*.so; do
    if [[ -f "$sopath" ]]; then
        IX_LIBS="${IX_LIBS} ${sopath}"
    fi
done

if [[ -z "$IX_LIBS" ]]; then
    echo "[ix_bridge] WARNING: No ixformer .so files found — bridge will compile but may not link all symbols" >&2
fi

# --- Locate rpath dirs ---
RPATH_DIRS=""
for d in \
    "${COREX_ROOT}/lib64" \
    "${COREX_ROOT}/lib/python3/dist-packages/ixformer" \
    "${COREX_ROOT}/lib64/python3/dist-packages/ixformer"; do
    if [[ -d "$d" ]]; then
        RPATH_DIRS="${RPATH_DIRS} -Wl,-rpath,${d}"
    fi
done

# --- Source file ---
SRC="${CSRC_DIR}/ix_full_bridge_v2.cpp"
if [[ ! -f "$SRC" ]]; then
    echo "[ix_bridge] ERROR: source not found: ${SRC}" >&2
    exit 1
fi

OUTPUT_DIR="${SCRIPT_DIR}/prebuilt"
mkdir -p "$OUTPUT_DIR"
OUTPUT="${OUTPUT_DIR}/ix_full_bridge_v2.so"

echo "[ix_bridge] Compiling: ${SRC}"
echo "[ix_bridge] Compiler: ${CLANGXX}"
echo "[ix_bridge] ixformer libs: ${IX_LIBS}"

$CLANGXX \
    -shared -fPIC -O2 -std=c++17 \
    $PYTHON_INC \
    $TORCH_INC \
    $TORCH_LIB \
    -ltorch -ltorch_cpu -ltorch_python -lc10 \
    ${IX_LIBS} \
    ${RPATH_DIRS} \
    -o "$OUTPUT" \
    "$SRC"

echo "[ix_bridge] ✓ Built: ${OUTPUT}"
ls -lh "$OUTPUT"

# --- Deploy if VLLM_ROOT specified ---
if [[ -n "$VLLM_ROOT" ]] && [[ -d "$VLLM_ROOT" ]]; then
    mkdir -p "${VLLM_ROOT}/ex_engine"
    cp "$OUTPUT" "${VLLM_ROOT}/ex_engine/ix_full_bridge_v2.so"
    echo "[ix_bridge] ✓ Deployed to ${VLLM_ROOT}/ex_engine/"
fi

echo "[ix_bridge] Done"
