#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

__global__ void beta_decay_kernel(const half* beta_input,
                                  const half* decay_input,
                                  const half* a_log,
                                  const half* dt_bias,
                                  float* output, int elements,
                                  int heads) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  const int head = index % heads;

  const float beta_value = __half2float(beta_input[index]);
  const float beta_fp32 = 1.0f / (1.0f + expf(-beta_value));
  output[index] = __half2float(__float2half(beta_fp32));

  const float x = (__half2float(decay_input[index])
                   + __half2float(dt_bias[head]));
  const float softplus = x > 20.0f ? x : log1pf(expf(x));
  output[elements + index] = expf(
      -expf(__half2float(a_log[head])) * softplus);
}

void check_half(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16,
              name, " must have dtype float16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == 2, name, " must have shape (batch, heads)");
}

void check_half_vector(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16,
              name, " must have dtype float16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == 1, name, " must have shape (heads)");
}

}  // namespace

torch::Tensor beta_decay(const torch::Tensor& beta_input,
                         const torch::Tensor& decay_input,
                         const torch::Tensor& a_log,
                         const torch::Tensor& dt_bias) {
  check_half(beta_input, "beta_input");
  check_half(decay_input, "decay_input");
  check_half_vector(a_log, "a_log");
  check_half_vector(dt_bias, "dt_bias");
  TORCH_CHECK(beta_input.sizes() == decay_input.sizes(),
              "beta_input and decay_input shapes must match");
  TORCH_CHECK(beta_input.size(1) == a_log.size(0) &&
                  a_log.sizes() == dt_bias.sizes(),
              "parameter heads must match input heads");

  const int elements = static_cast<int>(beta_input.numel());
  const int heads = static_cast<int>(beta_input.size(1));
  torch::Tensor output = torch::empty(
      {2, beta_input.size(0), beta_input.size(1)},
      beta_input.options().dtype(torch::kFloat32));
  constexpr int threads = 128;
  const int blocks = (elements + threads - 1) / threads;
  beta_decay_kernel<<<blocks, threads, 0,
                      at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(beta_input.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(decay_input.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(a_log.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(dt_bias.data_ptr<at::Half>()),
      output.data_ptr<float>(), elements, heads);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("beta_decay", &beta_decay,
             "Fused GDN beta sigmoid and decay factor");
}
