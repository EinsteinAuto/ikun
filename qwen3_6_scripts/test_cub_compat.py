#!/usr/bin/env python3
"""
test_cub_compat.py — Test CUB block-level primitives on BI-V100

Tests:
  1. __shfl_sync (warp shuffle) — CUB's warp_reduce depends on this
  2. __syncthreads (block sync) — CUB's block_reduce depends on this
  3. atomicAdd (shared mem) — CUB's nondeterministic reduce uses this
  4. cub::BlockReduce<float, 256> — the actual primitive

Run: python3 qwen3_6_scripts/test_cub_compat.py
"""
import torch
import os

def test_shfl():
    """Test warp shuffle via PyTorch inline CUDA."""
    src = r"""
    #include <torch/extension.h>
    #include <cuda_runtime.h>

    __global__ void test_shfl_kernel(float* out, int n) {
        int tid = threadIdx.x;
        float val = (float)tid;
        // warp shuffle down by 1
        float shuffled = __shfl_down_sync(0xffffffff, val, 1);
        if (tid < n) out[tid] = shuffled;
    }

    torch::Tensor test_shfl(int n) {
        auto out = torch::zeros({n}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        test_shfl_kernel<<<1, n>>>(out.data_ptr<float>(), n);
        return out;
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("test_shfl", &test_shfl);
    }
    """
    try:
        from torch.utils.cpp_extension import load_inline
        mod = load_inline("test_shfl", cpp_sources="", cuda_sources=src,
                         functions=["test_shfl"], verbose=False)
        result = mod.test_shfl(32)
        # thread 0 should get thread 1's value (1.0), thread 1 gets 2.0, etc
        expected_0 = 1.0  # shfl_down(0, 1) = value of thread 1
        actual_0 = result[0].item()
        if abs(actual_0 - expected_0) < 0.01:
            print(f"  ✓ __shfl_down_sync works: thread0 got {actual_0} (expected {expected_0})")
            return True
        else:
            print(f"  ✗ __shfl_down_sync wrong: thread0 got {actual_0} (expected {expected_0})")
            return False
    except Exception as e:
        print(f"  ✗ __shfl_down_sync failed: {str(e)[:200]}")
        return False


def test_block_reduce_manual():
    """Test block-wide reduction using shared memory + warp shuffle."""
    src = r"""
    #include <torch/extension.h>
    #include <cuda_runtime.h>

    __global__ void block_sum_kernel(const float* input, float* output, int n) {
        __shared__ float smem[32];  // one per warp
        int tid = threadIdx.x;
        int warp_id = tid / 32;
        int lane_id = tid % 32;

        float val = (tid < n) ? input[tid] : 0.0f;

        // Warp reduce using shuffle
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }

        // Lane 0 of each warp writes to SMEM
        if (lane_id == 0) smem[warp_id] = val;
        __syncthreads();

        // First warp reduces across warps
        if (warp_id == 0) {
            val = (tid < blockDim.x / 32) ? smem[tid] : 0.0f;
            for (int offset = 16; offset > 0; offset >>= 1) {
                val += __shfl_down_sync(0xffffffff, val, offset);
            }
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
        from torch.utils.cpp_extension import load_inline
        mod = load_inline("test_block_sum", cpp_sources="", cuda_sources=src,
                         functions=["block_sum"], verbose=False)
        x = torch.ones(256, dtype=torch.float32, device="cuda")
        result = mod.block_sum(x)
        expected = 256.0
        actual = result[0].item()
        err = abs(actual - expected)
        if err < 0.01:
            print(f"  ✓ block reduce works: sum(ones[256]) = {actual} (expected {expected})")
            return True
        else:
            print(f"  ✗ block reduce wrong: sum(ones[256]) = {actual} (expected {expected})")
            return False
    except Exception as e:
        print(f"  ✗ block reduce failed: {str(e)[:200]}")
        return False


def test_cub_block_reduce():
    """Test actual cub::BlockReduce — needs CUB headers accessible."""
    # Try with CCCL headers from our upstream
    cccl_include = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                "cccl_upstream")
    cub_include = os.path.join(cccl_include, "cub")
    libcudacxx_include = os.path.join(cccl_include, "libcudacxx", "include")

    src = r"""
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

    extra_include = []
    if os.path.isdir(cub_include):
        extra_include.append(cub_include)
    if os.path.isdir(libcudacxx_include):
        extra_include.append(libcudacxx_include)

    # Also check corex bundled CUB
    corex_cub = "/usr/local/corex/include"
    if os.path.isdir(corex_cub):
        extra_include.append(corex_cub)

    try:
        from torch.utils.cpp_extension import load_inline
        mod = load_inline("test_cub_reduce", cpp_sources="", cuda_sources=src,
                         functions=["cub_reduce"],
                         extra_include_paths=extra_include,
                         extra_cflags=["-std=c++17"],
                         verbose=True)
        x = torch.arange(256, dtype=torch.float32, device="cuda")
        result = mod.cub_reduce(x)
        expected = 256 * 255 / 2  # sum(0..255) = 32640
        actual = result[0].item()
        err = abs(actual - expected)
        if err < 1.0:
            print(f"  ✓ cub::BlockReduce works: sum(0..255) = {actual} (expected {expected})")
            return True
        else:
            print(f"  ✗ cub::BlockReduce wrong: {actual} (expected {expected})")
            return False
    except Exception as e:
        print(f"  ✗ cub::BlockReduce compile failed: {str(e)[:300]}")
        return False


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("No CUDA device")
        exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    print("[1/3] warp shuffle (__shfl_down_sync)")
    r1 = test_shfl()

    print("[2/3] manual block reduce (SMEM + shuffle)")
    r2 = test_block_reduce_manual()

    print("[3/3] cub::BlockReduce<float, 256>")
    r3 = test_cub_block_reduce()

    print()
    passed = sum([r1, r2, r3])
    print(f"{'='*50}")
    print(f"  {passed}/3 passed")
    if r1 and r2:
        print("  warp shuffle + SMEM works → CUB block primitives SHOULD compile")
    if r3:
        print("  cub::BlockReduce confirmed working on this GPU")
    print(f"{'='*50}")
