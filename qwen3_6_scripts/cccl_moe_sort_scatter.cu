// cccl_moe_sort_scatter.cu — CCCL CUB device-level MoE token dispatch
//
// Split compilation: this file uses CCCL headers only (no torch).
// Pybind wrapper in cccl_moe_sort_scatter_pybind.cpp links against this.
//
// Build pattern (same as cccl_allocator_preload.cu):
//   clang++ -I cccl_preload/include -DCCCL_IGNORE_DEPRECATED_CUDA_BELOW_12
//           -DCUB_WRAPPED_NAMESPACE=cccl_moe ...

// Suppress CUDA <12 check — corex 10.2 works for block-level CUB
#define CCCL_IGNORE_DEPRECATED_CUDA_BELOW_12

// Isolate from corex CUB
#define CUB_WRAPPED_NAMESPACE cccl_moe

#include <cub/block/block_scan.cuh>
#include <cuda_runtime.h>
#include <cstdint>

// ========================================================================
// Kernels
// ========================================================================

static constexpr int32_t kBlock = 256;

__global__ void moe_histogram_kernel(
    const int32_t* __restrict__ expert_id,
    int32_t* __restrict__ expert_sizes,
    int64_t num_elements,
    int32_t num_experts) {
  int64_t tid = int64_t(blockIdx.x) * kBlock + threadIdx.x;
  if (tid < num_elements) {
    int32_t eid = expert_id[tid];
    if (eid >= 0 && eid < num_experts) {
      atomicAdd(&expert_sizes[eid], 1);
    }
  }
}

__global__ void moe_prefix_sum_kernel(
    const int32_t* __restrict__ expert_sizes,
    int32_t* __restrict__ expert_offsets,
    int32_t num_experts) {
  using BlockScan = cccl_moe::cub::BlockScan<int32_t, 256>;
  __shared__ typename BlockScan::TempStorage s_scan;

  int32_t val = (threadIdx.x < num_experts) ? expert_sizes[threadIdx.x] : 0;
  int32_t offset;
  BlockScan(s_scan).ExclusiveSum(val, offset);
  __syncthreads();

  if (threadIdx.x < num_experts) {
    expert_offsets[threadIdx.x] = offset;
  }
}

__global__ void moe_place_kernel(
    const int32_t* __restrict__ expert_id,
    int32_t* __restrict__ expert_offsets,
    int32_t* __restrict__ dst_src,
    int32_t* __restrict__ src_dst,
    int64_t num_elements,
    int32_t num_experts) {
  int64_t flat_idx = int64_t(blockIdx.x) * kBlock + threadIdx.x;
  if (flat_idx >= num_elements) return;

  int32_t eid = expert_id[flat_idx];
  if (eid < 0 || eid >= num_experts) return;

  int32_t pos = atomicAdd(&expert_offsets[eid], 1);
  dst_src[pos] = static_cast<int32_t>(flat_idx);
  src_dst[flat_idx] = pos;
}

// ========================================================================
// C API — called from pybind wrapper
// ========================================================================

extern "C" {

void cccl_moe_launch_histogram(
    const int32_t* expert_id, int32_t* expert_sizes,
    int64_t N, int32_t E, cudaStream_t stream) {
  int64_t grid = (N + kBlock - 1) / kBlock;
  moe_histogram_kernel<<<grid, kBlock, 0, stream>>>(expert_id, expert_sizes, N, E);
}

void cccl_moe_launch_prefix_sum(
    const int32_t* expert_sizes, int32_t* expert_offsets,
    int32_t E, cudaStream_t stream) {
  moe_prefix_sum_kernel<<<1, kBlock, 0, stream>>>(expert_sizes, expert_offsets, E);
}

void cccl_moe_launch_place(
    const int32_t* expert_id, int32_t* expert_offsets,
    int32_t* dst_src, int32_t* src_dst,
    int64_t N, int32_t E, cudaStream_t stream) {
  int64_t grid = (N + kBlock - 1) / kBlock;
  moe_place_kernel<<<grid, kBlock, 0, stream>>>(
      expert_id, expert_offsets, dst_src, src_dst, N, E);
}

} // extern "C"
