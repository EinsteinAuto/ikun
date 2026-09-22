#!/usr/bin/env bash
set -euo pipefail

VLLM_ROOT=${1:?usage: build_corex_fused_paged_prefill_split4.sh VLLM_ROOT}
COREX_ROOT=${COREX_ROOT:-/usr/local/corex-3.2.3}
TORCH_ROOT=${TORCH_ROOT:-${COREX_ROOT}/lib64/python3/dist-packages/torch}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT=${VLLM_ROOT}/corex_fused_paged_prefill_split4.so

"${COREX_ROOT}/bin/clang++" \
    -std=c++17 -O3 -shared -fPIC \
    --cuda-path="${COREX_ROOT}" --cuda-gpu-arch=ivcore10 \
    --no-cuda-version-check -D_GLIBCXX_USE_CXX11_ABI=0 \
    -DTORCH_EXTENSION_NAME=corex_fused_paged_prefill \
    -DTORCH_API_INCLUDE_EXTENSION_H \
    -I"${TORCH_ROOT}/include" \
    -I"${TORCH_ROOT}/include/torch/csrc/api/include" \
    -I"${TORCH_ROOT}/include/TH" -I"${TORCH_ROOT}/include/THC" \
    -I/usr/local/include/python3.10 \
    "${SCRIPT_DIR}/corex_fused_paged_prefill_split4.cu" \
    -L"${TORCH_ROOT}/lib" -L"${COREX_ROOT}/lib64" \
    -Wl,-rpath,"${TORCH_ROOT}/lib" -Wl,-rpath,"${COREX_ROOT}/lib64" \
    -ltorch_python -ltorch_cuda -ltorch_cpu -ltorch \
    -lc10_cuda -lc10 -lcublas -lcudart -o "${OUTPUT}"

test -s "${OUTPUT}"
printf '[ok] CoreX split4 fused paged-prefill extension %s\n' "${OUTPUT}"
