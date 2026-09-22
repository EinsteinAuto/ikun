#!/bin/bash
# build_test_cutlass_batched.sh — Compile and test Cu10 TensorOp batched GEMM
set -eo pipefail

SAMPLES="/usr/local/corex-samples-3.2.3_x86_64/samples/cutlass"
SRC="ex_engine/xllm_kernels/cuda/moe_cutlass_batched.cu"

echo "=== Compile Cu10 TensorOp batched HGEMM ==="
/usr/local/corex/bin/clang++ \
    --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/corex \
    -I"${SAMPLES}/include" \
    -I/usr/local/corex/include \
    -L/usr/local/corex/lib64 -lcudart -lcutlass \
    -DBUILD_STANDALONE_TEST \
    -O2 -std=c++17 \
    "$SRC" -o /tmp/test_cutlass_batched 2>&1

if [ -f /tmp/test_cutlass_batched ]; then
    echo "Compile: SUCCESS"
    echo ""
    echo "=== Run ==="
    /tmp/test_cutlass_batched
else
    echo "Compile: FAILED"
fi
