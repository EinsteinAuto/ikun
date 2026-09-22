#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int kTopK = 8;
constexpr int kThreads = 256;
constexpr int kGridX = 8;

__global__ void selected_weight_gather_vec16_kernel(
    const uint4* w13, const uint4* w2, const int64_t* expert_ids,
    uint4* selected_w13, uint4* selected_w2,
    int64_t w13_vecs_per_expert, int64_t w2_vecs_per_expert) {
  const int segment = blockIdx.y;
  const int slot = segment & (kTopK - 1);
  const bool copy_w2 = segment >= kTopK;
  const int64_t count =
      copy_w2 ? w2_vecs_per_expert : w13_vecs_per_expert;
  const uint4* source = copy_w2 ? w2 : w13;
  uint4* output = copy_w2 ? selected_w2 : selected_w13;
  const int64_t source_offset = expert_ids[slot] * count;
  const int64_t output_offset = static_cast<int64_t>(slot) * count;
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    output[output_offset + index] = source[source_offset + index];
  }
}

void check_weight(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16,
              name, " must have dtype float16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == 3, name, " must be rank three");
  TORCH_CHECK(tensor.size(1) * tensor.size(2) % 8 == 0,
              name, " expert slices must be divisible by 16 bytes");
}

}  // namespace

std::vector<torch::Tensor> gather_selected_weights(
    const torch::Tensor& w13, const torch::Tensor& w2,
    const torch::Tensor& expert_ids) {
  check_weight(w13, "w13");
  check_weight(w2, "w2");
  TORCH_CHECK(w13.device() == w2.device(),
              "W13/W2 must be on the same device");
  TORCH_CHECK(w13.size(0) == w2.size(0),
              "W13/W2 expert counts differ");
  TORCH_CHECK(w13.size(2) == w2.size(1),
              "W13/W2 hidden dimensions differ");
  TORCH_CHECK(w13.size(1) == 2 * w2.size(2),
              "W13/W2 intermediate dimensions differ");
  TORCH_CHECK(expert_ids.is_cuda() && expert_ids.is_contiguous(),
              "expert_ids must be a contiguous CUDA tensor");
  TORCH_CHECK(expert_ids.device() == w13.device(),
              "weights and expert_ids must be on the same device");
  TORCH_CHECK(expert_ids.scalar_type() == torch::kInt64,
              "expert_ids must have dtype int64");
  TORCH_CHECK(expert_ids.dim() == 1 && expert_ids.numel() == kTopK,
              "expert_ids must have shape (8,)");

  auto selected_w13 = torch::empty(
      {kTopK, w13.size(1), w13.size(2)}, w13.options());
  auto selected_w2 = torch::empty(
      {kTopK, w2.size(1), w2.size(2)}, w2.options());
  const int64_t w13_vecs_per_expert = w13.size(1) * w13.size(2) / 8;
  const int64_t w2_vecs_per_expert = w2.size(1) * w2.size(2) / 8;
  const dim3 grid(kGridX, 2 * kTopK);
  selected_weight_gather_vec16_kernel<<<
      grid, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint4*>(w13.data_ptr<at::Half>()),
      reinterpret_cast<const uint4*>(w2.data_ptr<at::Half>()),
      expert_ids.data_ptr<int64_t>(),
      reinterpret_cast<uint4*>(selected_w13.data_ptr<at::Half>()),
      reinterpret_cast<uint4*>(selected_w2.data_ptr<at::Half>()),
      w13_vecs_per_expert, w2_vecs_per_expert);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {selected_w13, selected_w2};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("gather", &gather_selected_weights,
             "Gather selected FP16 top-8 MoE weights with 16-byte loads");
}
