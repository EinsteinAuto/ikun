#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p build && cd build
cmake .. -DWITH_NCCL=ON \
    -DNCCL_LIB=/usr/local/corex/lib64/libnccl.so \
    -DNCCL_INC=/usr/local/corex/include \
    -DCMAKE_CUDA_COMPILER=/usr/local/corex/bin/nvcc 2>&1 | tail -5
make -j$(nproc) 2>&1 | tail -5
echo "=== built ==="
ls -lh src/libinfiniccl.so
