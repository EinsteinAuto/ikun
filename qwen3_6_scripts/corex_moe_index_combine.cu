// corex_moe_index_combine.cu — Fused MoE index computation + combine
//
// Two kernels from xllm/core/kernels/cuda/moe/:
//   1. moe_compute_index: histogram + prefix_sum + place → {src_dst, dst_src, expert_sizes}
//   2. moe_combine_result: weighted sum of expert outputs → final output
//
// These replace Python argsort+bincount+loop in qwen3_5.py _pure_pytorch_experts prefill path.

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/block/block_scan.cuh>

// ========== moe_compute_index ==========

constexpr int32_t kMoeIndexBlock = 256;

__global__ void moe_histogram_kernel(
    const int32_t* __restrict__ expert_id,
    int32_t* __restrict__ expert_sizes,
    int64_t num_elements,
    int32_t num_experts) {
  int64_t tid = int64_t(blockIdx.x) * kMoeIndexBlock + threadIdx.x;
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
    int32_t num_experts,
    int64_t* __restrict__ total_out) {
  using BlockScan = cub::BlockScan<int32_t, kMoeIndexBlock>;
  __shared__ typename BlockScan::TempStorage s_scan;

  int32_t val = (threadIdx.x < num_experts) ? expert_sizes[threadIdx.x] : 0;
  int32_t offset;
  BlockScan(s_scan).ExclusiveSum(val, offset);
  __syncthreads();

  int32_t total = offset + val;

  if (threadIdx.x < num_experts) {
    expert_offsets[threadIdx.x] = offset;
  }
  if (threadIdx.x == 0 && total_out != nullptr) {
    *total_out = total;
  }
}

__global__ void moe_place_indices_kernel(
    const int32_t* __restrict__ expert_id,
    int32_t* __restrict__ expert_offsets,
    int32_t* __restrict__ dst_src,
    int32_t* __restrict__ src_dst,
    int64_t num_elements,
    int32_t num_experts) {
  int64_t flat_idx = int64_t(blockIdx.x) * kMoeIndexBlock + threadIdx.x;
  if (flat_idx >= num_elements) return;

  int32_t eid = expert_id[flat_idx];
  if (eid < 0 || eid >= num_experts) return;

  int32_t pos = atomicAdd(&expert_offsets[eid], 1);
  dst_src[pos] = static_cast<int32_t>(flat_idx);
  src_dst[flat_idx] = pos;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> moe_compute_index(
    const torch::Tensor& expert_id,
    int64_t num_experts) {
  auto device = expert_id.device();
  auto stream = at::cuda::getCurrentCUDAStream();
  int64_t N = expert_id.numel();
  int32_t E = static_cast<int32_t>(num_experts);
  auto expert_id_i32 = expert_id.to(torch::kInt32).contiguous();
  auto opt_i32 = expert_id_i32.options();

  auto expert_sizes = torch::zeros({num_experts}, opt_i32);
  auto expert_offsets = torch::empty({num_experts}, opt_i32);
  auto dst_src = torch::empty({N}, opt_i32);
  auto src_dst = torch::empty({N}, opt_i32);

  int64_t grid = (N + kMoeIndexBlock - 1) / kMoeIndexBlock;

  moe_histogram_kernel<<<grid, kMoeIndexBlock, 0, stream>>>(
      expert_id_i32.data_ptr<int32_t>(),
      expert_sizes.data_ptr<int32_t>(),
      N, E);

  moe_prefix_sum_kernel<<<1, kMoeIndexBlock, 0, stream>>>(
      expert_sizes.data_ptr<int32_t>(),
      expert_offsets.data_ptr<int32_t>(),
      E, nullptr);

  moe_place_indices_kernel<<<grid, kMoeIndexBlock, 0, stream>>>(
      expert_id_i32.data_ptr<int32_t>(),
      expert_offsets.data_ptr<int32_t>(),
      dst_src.data_ptr<int32_t>(),
      src_dst.data_ptr<int32_t>(),
      N, E);

  return std::make_tuple(src_dst, dst_src, expert_sizes);
}

// ========== moe_combine_result ==========

constexpr int32_t kCombineBlockSize = 256;

template <typename scalar_t>
__global__ void moe_combine_kernel(
    const scalar_t* __restrict__ gemm2,
    const float* __restrict__ reduce_weight,
    scalar_t* __restrict__ output,
    int64_t N,
    int32_t topk,
    int64_t H) {
  int64_t token_id = blockIdx.x;
  if (token_id >= N) return;

  int32_t tid = threadIdx.x;
  int32_t stride = kCombineBlockSize;

  for (int64_t h = tid; h < H; h += stride) {
    float acc = 0.0f;
    for (int32_t k = 0; k < topk; ++k) {
      int64_t flat_idx = token_id * topk + k;
      float w = reduce_weight[flat_idx];
      acc += w * static_cast<float>(gemm2[flat_idx * H + h]);
    }
    output[token_id * H + h] = static_cast<scalar_t>(acc);
  }
}

torch::Tensor moe_combine_result(
    const torch::Tensor& gemm2,
    const torch::Tensor& reduce_weight,
    int64_t N,
    int64_t topk) {
  auto stream = at::cuda::getCurrentCUDAStream();
  int64_t H = gemm2.size(1);
  auto dtype = gemm2.scalar_type();

  auto output = torch::empty({N, H}, gemm2.options());
  auto rw = reduce_weight.to(gemm2.device(), torch::kFloat32).contiguous();

  if (dtype == torch::kFloat16) {
    moe_combine_kernel<c10::Half>
        <<<N, kCombineBlockSize, 0, stream>>>(
            gemm2.data_ptr<c10::Half>(),
            rw.data_ptr<float>(),
            output.data_ptr<c10::Half>(),
            N, static_cast<int32_t>(topk), H);
  } else {
    moe_combine_kernel<float>
        <<<N, kCombineBlockSize, 0, stream>>>(
            gemm2.data_ptr<float>(),
            rw.data_ptr<float>(),
            output.data_ptr<float>(),
            N, static_cast<int32_t>(topk), H);
  }

  return output;
}

// ========== pybind ==========

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_compute_index", &moe_compute_index,
        "Fused MoE token-expert index computation (histogram+prefix_sum+place)");
  m.def("moe_combine_result", &moe_combine_result,
        "Fused MoE expert output weighted combination");
}
