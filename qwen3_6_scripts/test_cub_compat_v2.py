#!/usr/bin/env python3
"""
test_cub_compat_v2.py — Fixed CUB compatibility test for BI-V100

Changes from v1:
  - Write .cu to file + torch.utils.cpp_extension.load (not load_inline)
  - Test 3 uses corex's own CUB (not our CCCL 3.6 upstream)
"""
import torch
import os
import tempfile
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def compile_and_run(name, cu_code, fn_name, extra_include=None):
    """Write .cu to temp dir, compile with torch cpp_extension, return module."""
    from torch.utils.cpp_extension import load

    tmpdir = os.path.join(tempfile.gettempdir(), f"test_{name}")
    os.makedirs(tmpdir, exist_ok=True)
    cu_path = os.path.join(tmpdir, f"{name}.cu")
    with open(cu_path, "w") as f:
        f.write(cu_code)

    extra_cflags = ["-std=c++17"]
    extra_include_paths = extra_include or []

    mod = load(
        name=name,
        sources=[cu_path],
        extra_cflags=extra_cflags,
        extra_include_paths=extra_include_paths,
        verbose=True,
    )
    return mod


def test_shfl():
    """Test __shfl_down_sync on ivcore10."""
    code = r"""
#include <torch/extension.h>

__global__ void shfl_kernel(float* out, int n) {
    int tid = threadIdx.x;
    float val = (float)tid;
    float shuffled = __shfl_down_sync(0xffffffff, val, 1);
    if (tid < n) out[tid] = shuffled;
}

torch::Tensor test_shfl(int n) {
    auto out = torch::zeros({n}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    shfl_kernel<<<1, n>>>(out.data_ptr<float>(), n);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("test_shfl", &test_shfl);
}
"""
    try:
        mod = compile_and_run("shfl_test", code, "test_shfl")
        result = mod.test_shfl(32)
        val = result[0].item()
        if abs(val - 1.0) < 0.01:
            print(f"  ✓ __shfl_down_sync works: thread0 got {val}")
            return True
        else:
            print(f"  ✗ __shfl_down_sync wrong: thread0={val} expected=1.0")
            return False
    except Exception as e:
        print(f"  ✗ __shfl_down_sync failed: {str(e)[:300]}")
        return False


def test_block_reduce_manual():
    """Test SMEM + shuffle block reduce."""
    code = r"""
#include <torch/extension.h>

__global__ void block_sum_kernel(const float* input, float* output, int n) {
    __shared__ float smem[32];
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    float val = (tid < n) ? input[tid] : 0.0f;

    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);

    if (lane_id == 0) smem[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        val = (tid < blockDim.x / 32) ? smem[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (tid == 0) output[0] = val;
    }
}

torch::Tensor block_sum(torch::Tensor input) {
    int n = input.size(0);
    auto output = torch::zeros({1}, input.options());
    int threads = ((n + 31) / 32) * 32;
    if (threads > 1024) threads = 1024;
    block_sum_kernel<<<1, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), n);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("block_sum", &block_sum);
}
"""
    try:
        mod = compile_and_run("block_sum_test", code, "block_sum")
        x = torch.ones(256, dtype=torch.float32, device="cuda")
        result = mod.block_sum(x)
        val = result[0].item()
        if abs(val - 256.0) < 0.01:
            print(f"  ✓ block reduce works: sum(ones[256]) = {val}")
            return True
        else:
            print(f"  ✗ block reduce wrong: {val} expected=256.0")
            return False
    except Exception as e:
        print(f"  ✗ block reduce failed: {str(e)[:300]}")
        return False


def test_corex_cub():
    """Test cub::BlockReduce using corex's OWN CUB headers (not our CCCL 3.6)."""
    code = r"""
#include <torch/extension.h>
#include <cub/block/block_reduce.cuh>

__global__ void cub_reduce_kernel(const float* input, float* output, int n) {
    using BlockReduce = cub::BlockReduce<float, 256>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    int tid = threadIdx.x;
    float val = (tid < n) ? input[tid] : 0.0f;
    float aggregate = BlockReduce(temp_storage).Sum(val);
    if (tid == 0) output[0] = aggregate;
}

torch::Tensor cub_reduce(torch::Tensor input) {
    int n = input.size(0);
    auto output = torch::zeros({1}, input.options());
    cub_reduce_kernel<<<1, 256>>>(input.data_ptr<float>(), output.data_ptr<float>(), n);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cub_reduce", &cub_reduce);
}
"""
    # Use corex's own CUB, NOT our cccl_upstream
    corex_include = "/usr/local/corex/include"
    extra = [corex_include] if os.path.isdir(corex_include) else []

    try:
        mod = compile_and_run("corex_cub_test", code, "cub_reduce", extra_include=extra)
        x = torch.arange(256, dtype=torch.float32, device="cuda")
        result = mod.cub_reduce(x)
        val = result[0].item()
        expected = 256 * 255 / 2  # 32640
        if abs(val - expected) < 1.0:
            print(f"  ✓ cub::BlockReduce (corex) works: sum(0..255) = {val}")
            return True
        else:
            print(f"  ✗ cub::BlockReduce wrong: {val} expected={expected}")
            return False
    except Exception as e:
        print(f"  ✗ cub::BlockReduce (corex) failed: {str(e)[:300]}")
        return False


def test_cccl36_define_bypass():
    """Test CCCL 3.6 with CCCL_IGNORE_DEPRECATED_CUDA_BELOW_12 define."""
    cccl_cub = os.path.join(os.path.dirname(SCRIPT_DIR), "cccl_upstream", "cub")
    cccl_libcudacxx = os.path.join(os.path.dirname(SCRIPT_DIR), "cccl_upstream", "libcudacxx", "include")

    if not os.path.isdir(cccl_cub):
        print("  ⊘ SKIP: cccl_upstream not found")
        return None

    code = r"""
#define CCCL_IGNORE_DEPRECATED_CUDA_BELOW_12
#include <torch/extension.h>
#include <cub/block/block_reduce.cuh>

__global__ void cub36_kernel(const float* input, float* output, int n) {
    using BlockReduce = cub::BlockReduce<float, 256>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    int tid = threadIdx.x;
    float val = (tid < n) ? input[tid] : 0.0f;
    float aggregate = BlockReduce(temp_storage).Sum(val);
    if (tid == 0) output[0] = aggregate;
}

torch::Tensor cub36_reduce(torch::Tensor input) {
    int n = input.size(0);
    auto output = torch::zeros({1}, input.options());
    cub36_kernel<<<1, 256>>>(input.data_ptr<float>(), output.data_ptr<float>(), n);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cub36_reduce", &cub36_reduce);
}
"""
    try:
        mod = compile_and_run("cccl36_cub_test", code, "cub36_reduce",
                            extra_include=[cccl_cub, cccl_libcudacxx])
        x = torch.arange(256, dtype=torch.float32, device="cuda")
        result = mod.cub36_reduce(x)
        val = result[0].item()
        expected = 32640
        if abs(val - expected) < 1.0:
            print(f"  ✓ CCCL 3.6 BlockReduce (with bypass) works: {val}")
            return True
        else:
            print(f"  ✗ CCCL 3.6 BlockReduce wrong: {val}")
            return False
    except Exception as e:
        print(f"  ✗ CCCL 3.6 (with bypass) failed: {str(e)[:300]}")
        return False


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("No CUDA"); exit(1)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    print("[1/4] __shfl_down_sync (warp shuffle)")
    r1 = test_shfl()

    print("[2/4] manual block reduce (SMEM + shuffle)")
    r2 = test_block_reduce_manual()

    print("[3/4] cub::BlockReduce (corex's own CUB)")
    r3 = test_corex_cub()

    print("[4/4] cub::BlockReduce (CCCL 3.6 with CUDA<12 bypass)")
    r4 = test_cccl36_define_bypass()

    print()
    results = [r for r in [r1, r2, r3, r4] if r is not None]
    passed = sum(results)
    total = len(results)
    print(f"{'='*50}")
    print(f"  {passed}/{total} passed")
    if r1: print("  → warp shuffle works on ivcore10")
    if r3: print("  → corex CUB is usable for block primitives")
    if r4: print("  → CCCL 3.6 CAN compile with bypass define")
    elif r4 is False: print("  → CCCL 3.6 variadic function issue remains — use corex CUB instead")
    print(f"{'='*50}")
