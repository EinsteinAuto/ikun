#!/usr/bin/env bash
# Build cccl_moe_sort_scatter — split compilation
#
# Step 1: Compile .cu with CCCL headers (no torch) → .o
# Step 2: Compile _pybind.cpp with torch headers (no CCCL) → .o
# Step 3: Link both → .so
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INC="${SCRIPT_DIR}/cccl_preload/include"
CU_SRC="${SCRIPT_DIR}/cccl_moe_sort_scatter.cu"
PY_SRC="${SCRIPT_DIR}/cccl_moe_sort_scatter_pybind.cpp"
OUT="${1:-${SCRIPT_DIR}/prebuilt/corex-3.2.3-ivcore10/cccl_moe_sort_scatter.so}"

# Find corex clang++
CXX=""
for c in /usr/local/corex-3.2.3/bin/clang++ /usr/local/corex/bin/clang++; do
    [[ -x "$c" ]] && CXX="$c" && break
done
[[ -n "${CXX}" ]] || { echo "no corex clang++"; exit 2; }

# Find torch paths
TORCH_INC=$(python3 -c "from torch.utils.cpp_extension import include_paths; print(include_paths()[0])")
TORCH_LIB=$(python3 -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
PYTHON_INC=$(python3 -c "from sysconfig import get_paths; print(get_paths()['include'])")
CUDA_INC="/usr/local/corex/include"

echo "[build] CXX=${CXX}"
echo "[build] CCCL=${INC}"
echo "[build] torch=${TORCH_INC}"

# Step 1: Compile CUDA kernels (CCCL headers, no torch)
echo "[build] Step 1: compile CUDA kernels..."
"${CXX}" \
    -fPIC -O3 -std=c++17 \
    -I"${INC}" \
    -I"${CUDA_INC}" \
    -DCCCL_IGNORE_DEPRECATED_CUDA_BELOW_12 \
    -DCUB_WRAPPED_NAMESPACE=cccl_moe \
    --cuda-gpu-arch=ivcore10 \
    --cuda-path=/usr/local/corex \
    -c "${CU_SRC}" -o /tmp/cccl_moe_kernels.o \
    2>&1

# Step 2: Compile pybind wrapper (torch headers, no CCCL)
echo "[build] Step 2: compile pybind wrapper..."
"${CXX}" \
    -fPIC -O2 -std=c++17 \
    -I"${TORCH_INC}" \
    -I"${TORCH_INC}/torch/csrc/api/include" \
    -I"${PYTHON_INC}" \
    -I"${CUDA_INC}" \
    -D_GLIBCXX_USE_CXX11_ABI=0 \
    -DTORCH_EXTENSION_NAME=cccl_moe_sort_scatter \
    -x c++ \
    -c "${PY_SRC}" -o /tmp/cccl_moe_pybind.o \
    2>&1

# Step 3: Link
echo "[build] Step 3: link..."
mkdir -p "$(dirname "${OUT}")"
"${CXX}" \
    -shared -fPIC \
    /tmp/cccl_moe_kernels.o \
    /tmp/cccl_moe_pybind.o \
    -L"${TORCH_LIB}" \
    -ltorch -lc10 -ltorch_cpu -ltorch_cuda \
    -L/usr/local/corex/lib64 -lcudart \
    -Wl,-rpath,"${TORCH_LIB}" \
    -o "${OUT}" \
    2>&1

SIZE=$(stat -c%s "${OUT}" 2>/dev/null || echo "?")
echo "[build] SUCCESS: ${OUT} (${SIZE} bytes)"
