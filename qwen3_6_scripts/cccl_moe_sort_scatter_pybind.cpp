// cccl_moe_sort_scatter_pybind.cpp — Torch pybind wrapper
//
// Links against cccl_moe_sort_scatter.so (built separately with CCCL headers).
// This file only includes torch headers — no CCCL, no namespace conflict.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

// C API from cccl_moe_sort_scatter.so
extern "C" {
void cccl_moe_launch_histogram(
    const int32_t* expert_id, int32_t* expert_sizes,
    int64_t N, int32_t E, cudaStream_t stream);
void cccl_moe_launch_prefix_sum(
    const int32_t* expert_sizes, int32_t* expert_offsets,
    int32_t E, cudaStream_t stream);
void cccl_moe_launch_place(
    const int32_t* expert_id, int32_t* expert_offsets,
    int32_t* dst_src, int32_t* src_dst,
    int64_t N, int32_t E, cudaStream_t stream);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
moe_sort_scatter(const torch::Tensor& expert_id, int64_t num_experts) {
  TORCH_CHECK(expert_id.is_cuda(), "expert_id must be on CUDA");
  auto stream = at::cuda::getCurrentCUDAStream();
  int64_t N = expert_id.numel();
  int32_t E = static_cast<int32_t>(num_experts);

  auto expert_id_i32 = expert_id.to(torch::kInt32).contiguous();
  auto opt_i32 = expert_id_i32.options();

  auto expert_sizes = torch::zeros({num_experts}, opt_i32);
  auto expert_offsets = torch::empty({num_experts}, opt_i32);
  auto dst_src = torch::empty({N}, opt_i32);
  auto src_dst = torch::empty({N}, opt_i32);

  cccl_moe_launch_histogram(
      expert_id_i32.data_ptr<int32_t>(),
      expert_sizes.data_ptr<int32_t>(),
      N, E, stream);

  cccl_moe_launch_prefix_sum(
      expert_sizes.data_ptr<int32_t>(),
      expert_offsets.data_ptr<int32_t>(),
      E, stream);

  cccl_moe_launch_place(
      expert_id_i32.data_ptr<int32_t>(),
      expert_offsets.data_ptr<int32_t>(),
      dst_src.data_ptr<int32_t>(),
      src_dst.data_ptr<int32_t>(),
      N, E, stream);

  return std::make_tuple(src_dst, dst_src, expert_sizes);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_sort_scatter", &moe_sort_scatter,
        "CCCL CUB-based MoE token dispatch (histogram+prefix_sum+scatter)");
}
