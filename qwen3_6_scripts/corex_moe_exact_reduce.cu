#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

constexpr int kTopK = 8;
constexpr int kThreads = 256;

enum class Mode { kSerialFloat, kTreeFloat, kSerialHalf };

__global__ void exact_reduce_kernel(const __half* expert_output,
                                    const __half* weights,
                                    __half* output, int hidden,
                                    Mode mode) {
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  if (column >= hidden) {
    return;
  }
  __half products[kTopK];
#pragma unroll
  for (int expert = 0; expert < kTopK; ++expert) {
    products[expert] = __hmul(
        expert_output[expert * hidden + column], weights[expert]);
  }
  if (mode == Mode::kSerialHalf) {
    __half sum = products[0];
#pragma unroll
    for (int expert = 1; expert < kTopK; ++expert) {
      sum = __hadd(sum, products[expert]);
    }
    output[column] = sum;
    return;
  }

  float sum;
  if (mode == Mode::kSerialFloat) {
    sum = __half2float(products[0]);
#pragma unroll
    for (int expert = 1; expert < kTopK; ++expert) {
      sum += __half2float(products[expert]);
    }
  } else {
    const float sum01 = __half2float(products[0]) + __half2float(products[1]);
    const float sum23 = __half2float(products[2]) + __half2float(products[3]);
    const float sum45 = __half2float(products[4]) + __half2float(products[5]);
    const float sum67 = __half2float(products[6]) + __half2float(products[7]);
    sum = (sum01 + sum23) + (sum45 + sum67);
  }
  output[column] = __float2half_rn(sum);
}

void check_input(const torch::Tensor& expert_output,
                 const torch::Tensor& weights) {
  TORCH_CHECK(expert_output.is_cuda() && weights.is_cuda(),
              "inputs must be CUDA tensors");
  TORCH_CHECK(expert_output.scalar_type() == torch::kFloat16
                  && weights.scalar_type() == torch::kFloat16,
              "inputs must have dtype float16");
  TORCH_CHECK(expert_output.is_contiguous() && weights.is_contiguous(),
              "inputs must be contiguous");
  TORCH_CHECK(expert_output.dim() == 2
                  && expert_output.size(0) == kTopK,
              "expert_output must have shape (8, hidden)");
  TORCH_CHECK(weights.dim() == 1 && weights.size(0) == kTopK,
              "weights must have shape (8,)");
}

torch::Tensor launch(const torch::Tensor& expert_output,
                     const torch::Tensor& weights, Mode mode) {
  check_input(expert_output, weights);
  auto output = torch::empty(
      {1, expert_output.size(1)}, expert_output.options());
  const int hidden = static_cast<int>(expert_output.size(1));
  const int blocks = (hidden + kThreads - 1) / kThreads;
  exact_reduce_kernel<<<blocks, kThreads, 0,
                        at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(expert_output.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(weights.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(output.data_ptr<at::Half>()), hidden, mode);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace

torch::Tensor serial_float(const torch::Tensor& expert_output,
                           const torch::Tensor& weights) {
  return launch(expert_output, weights, Mode::kSerialFloat);
}

torch::Tensor tree_float(const torch::Tensor& expert_output,
                         const torch::Tensor& weights) {
  return launch(expert_output, weights, Mode::kTreeFloat);
}

torch::Tensor serial_half(const torch::Tensor& expert_output,
                          const torch::Tensor& weights) {
  return launch(expert_output, weights, Mode::kSerialHalf);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("serial_float", &serial_float);
  module.def("tree_float", &tree_float);
  module.def("serial_half", &serial_half);
}
