#!/usr/bin/env python3
"""Check BI-V100 warp size."""
import torch
print(f"torch.cuda.get_device_properties(0).warp_size: "
      f"{getattr(torch.cuda.get_device_properties(0), 'warp_size', 'N/A')}")

# Also check via CUDA kernel
from torch.utils.cpp_extension import load
import tempfile, os
cu_code = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
__global__ void check_warp(int* out) {
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        out[0] = warpSize;
    }
}
torch::Tensor get_warp_size() {
    auto out = torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    check_warp<<<1, 32>>>(out.data_ptr<int>());
    return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_warp_size", &get_warp_size);
}
'''
with tempfile.NamedTemporaryFile(suffix='.cu', mode='w', delete=False) as f:
    f.write(cu_code)
    cu_path = f.name
ext = load(name="warpcheck", sources=[cu_path], verbose=False)
ws = ext.get_warp_size().item()
print(f"CUDA kernel warpSize: {ws}")
os.unlink(cu_path)
