#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <vector>

namespace {

constexpr int kHeadDim = 256;
constexpr int kThreads = 256;

void check_half_matrix(const torch::Tensor& input, const char* name) {
  TORCH_CHECK(input.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(input.scalar_type() == torch::kFloat16,
              name, " must have dtype float16");
  TORCH_CHECK(input.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(input.dim() == 2 && input.size(1) == kHeadDim,
              name, " must have shape (rows, 256)");
}

__global__ void prepare_kernel(const __half* input, float* converted,
                               float* squares, int rows) {
  const int row = blockIdx.x;
  const int column = threadIdx.x;
  if (row >= rows || column >= kHeadDim) {
    return;
  }
  const int offset = row * kHeadDim + column;
  const float value = __half2float(input[offset]);
  converted[offset] = value;
  squares[offset] = __fmul_rn(value, value);
}

__global__ void apply_inverse_kernel(
    const float* input, const __half* weight, const float* inverse,
    __half* output, int rows) {
  const int row = blockIdx.x;
  const int column = threadIdx.x;
  if (row >= rows || column >= kHeadDim) {
    return;
  }
  const int offset = row * kHeadDim + column;
  const float scaled = __fmul_rn(input[offset], inverse[row]);
  const float factor = __fadd_rn(1.0f, __half2float(weight[column]));
  output[offset] = __float2half_rn(__fmul_rn(scaled, factor));
}

}  // namespace

std::vector<torch::Tensor> prepare(const torch::Tensor& input) {
  check_half_matrix(input, "input");
  auto float_options = input.options().dtype(torch::kFloat32);
  auto converted = torch::empty(input.sizes(), float_options);
  auto squares = torch::empty(input.sizes(), float_options);
  const int rows = static_cast<int>(input.size(0));
  prepare_kernel<<<rows, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
      converted.data_ptr<float>(), squares.data_ptr<float>(), rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {converted, squares};
}

torch::Tensor apply_inverse(const torch::Tensor& input,
                            const torch::Tensor& weight,
                            const torch::Tensor& inverse) {
  TORCH_CHECK(input.is_cuda() && weight.is_cuda() && inverse.is_cuda(),
              "all tensors must be CUDA tensors");
  TORCH_CHECK(input.scalar_type() == torch::kFloat32,
              "input must have dtype float32");
  TORCH_CHECK(weight.scalar_type() == torch::kFloat16,
              "weight must have dtype float16");
  TORCH_CHECK(inverse.scalar_type() == torch::kFloat32,
              "inverse must have dtype float32");
  TORCH_CHECK(input.is_contiguous() && weight.is_contiguous()
                  && inverse.is_contiguous(),
              "all tensors must be contiguous");
  TORCH_CHECK(input.dim() == 2 && input.size(1) == kHeadDim,
              "input must have shape (rows, 256)");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == kHeadDim,
              "weight must have shape (256,)");
  TORCH_CHECK(inverse.numel() == input.size(0),
              "inverse must contain one value per row");
  auto output = torch::empty(
      input.sizes(), input.options().dtype(torch::kFloat16));
  const int rows = static_cast<int>(input.size(0));
  apply_inverse_kernel<<<rows, kThreads, 0,
                         at::cuda::getCurrentCUDAStream()>>>(
      input.data_ptr<float>(),
      reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
      inverse.data_ptr<float>(),
      reinterpret_cast<__half*>(output.data_ptr<at::Half>()), rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("prepare", &prepare,
             "Convert FP16 attention heads and compute exact squares");
  module.def("apply_inverse", &apply_inverse,
             "Apply PyTorch-computed attention head RMSNorm inverse");
}
