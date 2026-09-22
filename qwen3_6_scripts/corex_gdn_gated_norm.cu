#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

constexpr int kHeadDim = 128;

__device__ __forceinline__ float silu(float value) {
  return value / (1.0f + expf(-value));
}

__global__ void gated_rms_norm_inverse_kernel(
    const float* input, const __half* gate, const __half* weight,
    const float* inverse, __half* output, int rows) {
  const int row = blockIdx.x;
  const int column = threadIdx.x;
  if (row >= rows || column >= kHeadDim) {
    return;
  }
  const int offset = row * kHeadDim + column;
  const float scaled = __fmul_rn(input[offset], inverse[row]);
  const float normalized = __fmul_rn(
      __half2float(weight[column]), scaled);
  const float activated = silu(__half2float(gate[offset]));
  output[offset] = __float2half_rn(__fmul_rn(normalized, activated));
}

void check_input(const torch::Tensor& input, const torch::Tensor& gate,
                 const torch::Tensor& weight,
                 const torch::Tensor& inverse) {
  TORCH_CHECK(input.is_cuda() && gate.is_cuda() && weight.is_cuda()
                  && inverse.is_cuda(),
              "all tensors must be CUDA tensors");
  TORCH_CHECK(input.scalar_type() == torch::kFloat32,
              "input must have dtype float32");
  TORCH_CHECK(gate.scalar_type() == torch::kFloat16,
              "gate must have dtype float16");
  TORCH_CHECK(weight.scalar_type() == torch::kFloat16,
              "weight must have dtype float16");
  TORCH_CHECK(inverse.scalar_type() == torch::kFloat32,
              "inverse must have dtype float32");
  TORCH_CHECK(input.is_contiguous() && gate.is_contiguous()
                  && weight.is_contiguous() && inverse.is_contiguous(),
              "all tensors must be contiguous");
  TORCH_CHECK(input.dim() == 2 && input.size(1) == kHeadDim,
              "input must have shape (rows, 128)");
  TORCH_CHECK(gate.sizes() == input.sizes(),
              "gate must match input shape");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == kHeadDim,
              "weight must have shape (128,)");
  TORCH_CHECK(inverse.numel() == input.size(0),
              "inverse must contain one value per row");
}

}  // namespace

torch::Tensor apply_inverse(const torch::Tensor& input,
                            const torch::Tensor& gate,
                            const torch::Tensor& weight,
                            const torch::Tensor& inverse) {
  check_input(input, gate, weight, inverse);
  auto output = torch::empty_like(gate);
  const int rows = static_cast<int>(input.size(0));
  gated_rms_norm_inverse_kernel<<<
      rows, kHeadDim, 0, at::cuda::getCurrentCUDAStream()>>>(
      input.data_ptr<float>(),
      reinterpret_cast<const __half*>(gate.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
      inverse.data_ptr<float>(),
      reinterpret_cast<__half*>(output.data_ptr<at::Half>()), rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("apply_inverse", &apply_inverse,
             "CoreX gated RMSNorm using a PyTorch-computed inverse");
}
