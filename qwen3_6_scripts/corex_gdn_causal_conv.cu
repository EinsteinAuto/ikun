#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

constexpr int kStateLen = 3;
constexpr int kKernelSize = kStateLen + 1;
constexpr int kThreads = 256;

__global__ void causal_conv_update_kernel(
    float* state, const __half* hidden, const __half* weight,
    __half* output, int channels) {
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  const int batch = blockIdx.y;
  if (channel >= channels) {
    return;
  }

  const int state_offset = (batch * channels + channel) * kStateLen;
  const int vector_offset = batch * channels + channel;
  const int weight_offset = channel * kKernelSize;
  const __half current = hidden[vector_offset];
  const __half state0 = __float2half_rn(state[state_offset]);
  const __half state1 = __float2half_rn(state[state_offset + 1]);
  const __half state2 = __float2half_rn(state[state_offset + 2]);

  float value = __half2float(state0) * __half2float(weight[weight_offset]);
  value += __half2float(state1) * __half2float(weight[weight_offset + 1]);
  value += __half2float(state2) * __half2float(weight[weight_offset + 2]);
  value += __half2float(current) * __half2float(weight[weight_offset + 3]);

  state[state_offset] = __half2float(state1);
  state[state_offset + 1] = __half2float(state2);
  state[state_offset + 2] = __half2float(current);
  const __half convolved = __float2half_rn(value);
  const float activation_input = __half2float(convolved);
  output[vector_offset] = __float2half_rn(
      activation_input / (1.0f + expf(-activation_input)));
}

void check_half_cuda_contiguous(const torch::Tensor& tensor,
                                const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16,
              name, " must have dtype float16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

torch::Tensor causal_conv_update(torch::Tensor state,
                                 const torch::Tensor& hidden,
                                 const torch::Tensor& weight) {
  TORCH_CHECK(state.is_cuda(), "state must be a CUDA tensor");
  TORCH_CHECK(state.scalar_type() == torch::kFloat32,
              "state must have dtype float32");
  TORCH_CHECK(state.is_contiguous(), "state must be contiguous");
  check_half_cuda_contiguous(hidden, "hidden");
  check_half_cuda_contiguous(weight, "weight");
  TORCH_CHECK(state.dim() == 3 && state.size(2) == kStateLen,
              "state must have shape (batch, channels, 3)");
  TORCH_CHECK(hidden.dim() == 3 && hidden.size(2) == 1 &&
                  hidden.size(0) == state.size(0) &&
                  hidden.size(1) == state.size(1),
              "hidden must have shape (batch, channels, 1)");
  TORCH_CHECK(weight.dim() == 2 && weight.size(0) == state.size(1) &&
                  weight.size(1) == kKernelSize,
              "weight must have shape (channels, 4)");

  auto output = torch::empty_like(hidden);
  const int channels = static_cast<int>(state.size(1));
  const dim3 blocks((channels + kThreads - 1) / kThreads,
                    static_cast<unsigned int>(state.size(0)));
  causal_conv_update_kernel<<<blocks, kThreads, 0,
                              at::cuda::getCurrentCUDAStream()>>>(
      state.data_ptr<float>(),
      reinterpret_cast<const __half*>(hidden.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(output.data_ptr<at::Half>()), channels);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("causal_conv_update", &causal_conv_update,
             "Fused CoreX Gated DeltaNet causal convolution update");
}
