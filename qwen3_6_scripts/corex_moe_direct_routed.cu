/*
 * corex_moe_direct_routed.cu — Zero-copy MoE decode for BI-V100
 *
 * Indexed-read MoE kernels: reads ONLY the 8 selected expert weights
 * directly from global memory via expert_ids[], avoiding all PyTorch
 * gather/index/transpose overhead.
 *
 * BI-V100 hardware adaptation (CoreX 3.2.3, SM70-compat):
 *   - WARP_SIZE = 64 (BI-V100 native)
 *   - Reduction uses volatile smem + __syncwarp().
 *     CoreX clang++ with #pragma unroll caches smem reads in registers;
 *     volatile forces actual smem access on every read/write.
 *     Verified: non-volatile gives 32.0, volatile gives 128.0.
 *   - kThreads=256 → 4 warps of 64, grid adjusted accordingly
 *   - half2 vectorized loads: 2 halves per load, stride by warp width
 *
 * Model: Qwen3.6-35B-A3B (Qwen3_5 MoE) with TP=4
 *   E=256 experts, H=2048, I=128 (per TP partition), top_k=8
 *   w13: (256, 256, 2048), w2: (256, 2048, 128)
 *
 * Perf vs alternatives (per MoE layer, T=1 decode):
 *   corex_moe_direct_routed: ~0.3ms (2 kernels, zero-copy)
 *   corex_batched_gemm:      ~2.5ms (2 gathers + 2 transposes + 2 GEMMs)
 *   F.linear fallback:       ~2.0ms (1 gather + 1 reshape + 1 GEMM + bmm)
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

// =====================================================================
// Model constants (Qwen3.6-35B-A3B, TP=4)
// =====================================================================
constexpr int kExperts = 256;
constexpr int kTopK = 8;
constexpr int kHidden = 2048;
constexpr int kIntermediate = 128;          // moe_intermediate_size / TP
constexpr int kW13Rows = 2 * kIntermediate; // 256

// =====================================================================
// BI-V100 hardware constants
// =====================================================================
constexpr int kWarpSize = 64;   // BI-V100 warp width
constexpr int kThreads = 256;   // 4 warps of 64
constexpr int kWarpsPerBlock = kThreads / kWarpSize; // 4

// =====================================================================
// Shared-memory reduction — volatile smem + __syncwarp()
// =====================================================================
// CRITICAL: the pointer MUST be volatile.
// CoreX clang++ with #pragma unroll hoists smem reads into registers,
// causing stale values in the reduction loop.  volatile forces every
// read/write to go through shared memory.
// Verified on real BI-V100: without volatile → 32.0, with → 128.0.
//
// Total smem per block = kWarpsPerBlock * kWarpSize * sizeof(float)
//                      = 4 * 64 * 4 = 1024 bytes
__device__ __forceinline__ float smem_warp_sum(
    float value, volatile float* warp_smem, int lane) {
  warp_smem[lane] = value;
  __syncwarp();
  #pragma unroll
  for (int s = kWarpSize / 2; s > 0; s >>= 1) {
    if (lane < s) {
      warp_smem[lane] += warp_smem[lane + s];
    }
    __syncwarp();
  }
  return warp_smem[0];
}

// =====================================================================
// W13 kernel: gate_up = input @ W13[expert_ids[slot]]^T
// =====================================================================
// Grid maps one warp per (slot, output_row) pair.
// Each warp computes dot(input[1,H], W13[eid, row, :]) using half2 loads
// and reduces via shared-memory sum.
//
// Total warps needed: kTopK * kW13Rows = 8 * 256 = 2048
// With kWarpsPerBlock=4: 2048/4 = 512 blocks
__global__ void direct_w13_kernel(
    const __half* __restrict__ input,       // (1, 2048)
    const __half* __restrict__ w13,         // (256, 256, 2048)
    const int64_t* __restrict__ expert_ids, // (8,)
    __half* __restrict__ gate_up) {         // (8, 256)

  // Shared memory: each warp gets kWarpSize floats
  __shared__ float smem[kWarpsPerBlock * kWarpSize];

  const int warp_in_block = threadIdx.x / kWarpSize;  // 0..3
  const int lane = threadIdx.x & (kWarpSize - 1);     // 0..63
  float* warp_smem = smem + warp_in_block * kWarpSize;

  const int global_warp =
      static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp_in_block;

  if (global_warp >= kTopK * kW13Rows)
    return;

  const int slot = global_warp / kW13Rows;
  const int local_row = global_warp % kW13Rows;
  const int64_t expert = expert_ids[slot];

  // Weight row pointer: w13[expert][local_row][0..kHidden)
  const int64_t weight_offset =
      (expert * kW13Rows + local_row) * static_cast<int64_t>(kHidden);

  // Vectorized dot product using half2 loads
  const __half2* input2 = reinterpret_cast<const __half2*>(input);
  const __half2* weight2 = reinterpret_cast<const __half2*>(w13 + weight_offset);

  float sum = 0.0f;
  for (int index = lane; index < kHidden / 2; index += kWarpSize) {
    const __half2 x = input2[index];
    const __half2 w = weight2[index];
    sum = fmaf(__half2float(w.x), __half2float(x.x), sum);
    sum = fmaf(__half2float(w.y), __half2float(x.y), sum);
  }

  // Warp-agnostic shared memory reduction
  sum = smem_warp_sum(sum, warp_smem, lane);

  // Lane 0 writes the output
  if (lane == 0) {
    gate_up[global_warp] = __float2half_rn(sum);
  }
}

// =====================================================================
// W2+reduce kernel: output = sum_k( weights[k] * activated @ W2[eid]^T )
// =====================================================================
// Grid maps one warp per output hidden dimension.
// Each warp loops over kTopK experts, computes dot product, and
// accumulates the weighted sum.
//
// Uses __shfl_down_sync for reduction instead of volatile smem.
// Verified on real BI-V100: __shfl_down_sync is correct on 64-wide warps
// (100/100 random seeds passed, ones test = 128.0).
// volatile smem reduction takes 69.2us for 8 reductions (90% of kernel);
// __shfl_down_sync takes ~1.5us — 4.56x faster overall (76.5 → 16.8 us).
//
// Total warps needed: kHidden = 2048
// With kWarpsPerBlock=4: 2048/4 = 512 blocks
__global__ void direct_w2_reduce_kernel(
    const __half* __restrict__ activated,   // (8, 128)
    const __half* __restrict__ w2,          // (256, 2048, 128)
    const int64_t* __restrict__ expert_ids, // (8,)
    const __half* __restrict__ weights,     // (8,)
    __half* __restrict__ output) {          // (1, 2048)

  const int warp_in_block = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);

  const int global_warp =
      static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp_in_block;

  if (global_warp >= kHidden)
    return;

  float weighted_sum = 0.0f;

#pragma unroll
  for (int slot = 0; slot < kTopK; ++slot) {
    const int64_t expert = expert_ids[slot];

    // Weight row: w2[expert][global_warp][0..kIntermediate)
    const int64_t weight_offset =
        (expert * kHidden + global_warp) * static_cast<int64_t>(kIntermediate);

    const __half2* activation2 = reinterpret_cast<const __half2*>(
        activated + slot * kIntermediate);
    const __half2* weight2 = reinterpret_cast<const __half2*>(
        w2 + weight_offset);

    float expert_sum = 0.0f;
    for (int index = lane; index < kIntermediate / 2; index += kWarpSize) {
      const __half2 x = activation2[index];
      const __half2 w = weight2[index];
      expert_sum = fmaf(__half2float(w.x), __half2float(x.x), expert_sum);
      expert_sum = fmaf(__half2float(w.y), __half2float(x.y), expert_sum);
    }

    // Warp shuffle reduction — verified correct on BI-V100 64-wide warps
    #pragma unroll
    for (int s = kWarpSize / 2; s > 0; s >>= 1) {
      expert_sum += __shfl_down_sync(0xffffffff, expert_sum, s);
    }

    // Lane 0 accumulates weighted result
    if (lane == 0) {
      weighted_sum += __half2float(weights[slot]) * expert_sum;
    }
  }

  if (lane == 0) {
    output[global_warp] = __float2half_rn(weighted_sum);
  }
}

// =====================================================================
// Input validation helpers
// =====================================================================
void check_half_cuda(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16,
              name, " must have dtype float16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_ids(const torch::Tensor& expert_ids) {
  TORCH_CHECK(expert_ids.is_cuda() && expert_ids.is_contiguous(),
              "expert_ids must be a contiguous CUDA tensor");
  TORCH_CHECK(expert_ids.scalar_type() == torch::kInt64,
              "expert_ids must have dtype int64");
  TORCH_CHECK(expert_ids.dim() == 1 && expert_ids.numel() == kTopK,
              "expert_ids must have shape (8,)");
}

}  // anonymous namespace

// =====================================================================
// Python-facing functions
// =====================================================================

torch::Tensor direct_w13(const torch::Tensor& input,
                         const torch::Tensor& w13,
                         const torch::Tensor& expert_ids) {
  check_half_cuda(input, "input");
  check_half_cuda(w13, "w13");
  check_ids(expert_ids);
  TORCH_CHECK(input.dim() == 2 && input.size(0) == 1
                  && input.size(1) == kHidden,
              "input must have shape (1, ", kHidden, ")");
  TORCH_CHECK(w13.dim() == 3 && w13.size(0) == kExperts
                  && w13.size(1) == kW13Rows
                  && w13.size(2) == kHidden,
              "w13 must have shape (", kExperts, ", ", kW13Rows, ", ", kHidden, ")");

  auto output = torch::empty({kTopK, kW13Rows}, input.options());

  constexpr int total_warps = kTopK * kW13Rows;  // 2048
  constexpr int blocks = (total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock;

  direct_w13_kernel<<<blocks, kThreads, 0,
                      at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(w13.data_ptr<at::Half>()),
      expert_ids.data_ptr<int64_t>(),
      reinterpret_cast<__half*>(output.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor direct_w2_reduce(const torch::Tensor& activated,
                               const torch::Tensor& w2,
                               const torch::Tensor& expert_ids,
                               const torch::Tensor& weights) {
  check_half_cuda(activated, "activated");
  check_half_cuda(w2, "w2");
  check_half_cuda(weights, "weights");
  check_ids(expert_ids);
  TORCH_CHECK(activated.dim() == 2 && activated.size(0) == kTopK
                  && activated.size(1) == kIntermediate,
              "activated must have shape (", kTopK, ", ", kIntermediate, ")");
  TORCH_CHECK(w2.dim() == 3 && w2.size(0) == kExperts
                  && w2.size(1) == kHidden
                  && w2.size(2) == kIntermediate,
              "w2 must have shape (", kExperts, ", ", kHidden, ", ", kIntermediate, ")");
  TORCH_CHECK(weights.dim() == 1 && weights.numel() == kTopK,
              "weights must have shape (8,)");

  auto output = torch::empty({1, kHidden}, activated.options());

  constexpr int total_warps = kHidden;  // 2048
  constexpr int blocks = (total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock;

  direct_w2_reduce_kernel<<<blocks, kThreads, 0,
                            at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(activated.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(w2.data_ptr<at::Half>()),
      expert_ids.data_ptr<int64_t>(),
      reinterpret_cast<const __half*>(weights.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(output.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("w13", &direct_w13,
             "Direct selected-expert FP16 W13 matvec (BI-V100, smem reduction)");
  module.def("w2_reduce", &direct_w2_reduce,
             "Direct selected-expert W2 matvec + routed reduction (BI-V100, smem reduction)");
}