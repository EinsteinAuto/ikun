#!/usr/bin/env bash
# Build corex_batched_gemm.so — CUTLASS batched GEMM pybind for MoE decode
#
# Verified: 2.462ms for 8-expert decode (issue #68)
#
# Usage: bash build_corex_batched_gemm.sh VLLM_ROOT
#   or:  bash build_corex_batched_gemm.sh  (outputs to prebuilt/)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COREX_ROOT=${COREX_ROOT:-/usr/local/corex-3.2.3}
if [ ! -d "$COREX_ROOT" ]; then
    COREX_ROOT=/usr/local/corex
fi
TORCH_ROOT=${TORCH_ROOT:-${COREX_ROOT}/lib64/python3/dist-packages/torch}
if [ ! -d "$TORCH_ROOT" ]; then
    TORCH_ROOT=$(python3 -c "import torch; import os; print(os.path.dirname(torch.__file__))" 2>/dev/null || echo "/usr/local/corex/lib/python3/dist-packages/torch")
fi

CUTLASS_INCLUDE="$COREX_ROOT/lib64/python3/dist-packages/tensorflow/include/third_party/gpus/cuda/include"
if [ ! -f "$CUTLASS_INCLUDE/cutlass/cutlass.h" ]; then
    # Fallback: search
    CUTLASS_INCLUDE=$(find "$COREX_ROOT" -path "*/cutlass/cutlass.h" -printf '%h\n' 2>/dev/null | head -1 | sed 's|/cutlass$||')
    if [ -z "$CUTLASS_INCLUDE" ]; then
        echo "[build] ERROR: cannot find cutlass/cutlass.h under $COREX_ROOT"
        exit 1
    fi
fi

# Output path
if [ -n "${1:-}" ]; then
    OUTPUT="${1}/corex_batched_gemm.so"
else
    OUTPUT="$SCRIPT_DIR/prebuilt/corex-3.2.3-ivcore10/corex_batched_gemm.so"
fi

# Source files
BIND_CPP="$PROJ_ROOT/ex_engine/xllm_kernels/cuda/bindings/corex_batched_gemm_bind.cpp"
KERNEL_CU="$PROJ_ROOT/ex_engine/xllm_kernels/cuda/corex_batched_gemm_kernel.cu"

echo "[build] COREX_ROOT=$COREX_ROOT"
echo "[build] TORCH_ROOT=$TORCH_ROOT"
echo "[build] CUTLASS_INCLUDE=$CUTLASS_INCLUDE"
echo "[build] OUTPUT=$OUTPUT"

"${COREX_ROOT}/bin/clang++" \
    -std=c++17 -O3 -shared -fPIC \
    --cuda-path="${COREX_ROOT}" \
    --cuda-gpu-arch=ivcore10 \
    --no-cuda-version-check \
    -D_GLIBCXX_USE_CXX11_ABI=0 \
    -DTORCH_EXTENSION_NAME=corex_batched_gemm \
    -DTORCH_API_INCLUDE_EXTENSION_H \
    -I"${TORCH_ROOT}/include" \
    -I"${TORCH_ROOT}/include/torch/csrc/api/include" \
    -I"${TORCH_ROOT}/include/TH" \
    -I"${TORCH_ROOT}/include/THC" \
    -I"${CUTLASS_INCLUDE}" \
    -I/usr/local/include/python3.10 \
    "${KERNEL_CU}" "${BIND_CPP}" \
    -L"${TORCH_ROOT}/lib" \
    -L"${COREX_ROOT}/lib64" \
    -Wl,-rpath,"${TORCH_ROOT}/lib" \
    -Wl,-rpath,"${COREX_ROOT}/lib64" \
    -ltorch_python -ltorch_cuda -ltorch_cpu -ltorch \
    -lc10_cuda -lc10 -lcudart \
    -o "${OUTPUT}"

echo "[build] ✓ built ${OUTPUT}"
echo "[build] size: $(du -h "${OUTPUT}" | cut -f1)"

python3 -c "
import importlib.util
spec = importlib.util.spec_from_file_location('corex_batched_gemm', '${OUTPUT}')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('[build] ✓ import OK:', [x for x in dir(mod) if not x.startswith('_')])
" 2>&1 || echo "[build] import test skipped"

echo "[build] done"
