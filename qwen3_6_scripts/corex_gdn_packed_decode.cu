#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

constexpr int kKeyHeads = 4;
constexpr int kValueHeads = 8;
constexpr int kHeadDim = 128;
constexpr int kMixedDim =
    (2 * kKeyHeads + kValueHeads) * kHeadDim;
constexpr float kQueryScale = 0.08838834764831845f;

__global__ void gdn_packed_decode_kernel(
    float* state, const half* mixed_qkv, const half* beta_input,
    const half* decay_input, const half* a_log, const half* dt_bias,
    float* output) {
  const int batch_head = blockIdx.x;
  const int column = threadIdx.x;
  const int batch = batch_head / kValueHeads;
  const int value_head = batch_head % kValueHeads;
  const int key_head = value_head / (kValueHeads / kKeyHeads);
  const int mixed_offset = batch * kMixedDim;
  const int query_offset = mixed_offset + key_head * kHeadDim;
  const int key_offset =
      mixed_offset + kKeyHeads * kHeadDim + key_head * kHeadDim;
  const int value_offset = mixed_offset + 2 * kKeyHeads * kHeadDim
                           + value_head * kHeadDim;
  const int vector_offset = batch_head * kHeadDim;
  const int state_offset = batch_head * kHeadDim * kHeadDim;

  __shared__ half norm_squares[kHeadDim * 2];
  __shared__ float normalized_query[kHeadDim];
  __shared__ float normalized_key[kHeadDim];
  const half raw_query = mixed_qkv[query_offset + column];
  const half raw_key = mixed_qkv[key_offset + column];
  norm_squares[column] = __hmul(raw_query, raw_query);
  norm_squares[kHeadDim + column] = __hmul(raw_key, raw_key);
  __syncthreads();

  for (int stride = kHeadDim / 2; stride > 0; stride >>= 1) {
    if (column < stride) {
      norm_squares[column] = __hadd(
          norm_squares[column], norm_squares[column + stride]);
      norm_squares[kHeadDim + column] = __hadd(
          norm_squares[kHeadDim + column],
          norm_squares[kHeadDim + column + stride]);
    }
    __syncthreads();
  }

  const half epsilon = __float2half(1e-6f);
  const half query_inverse = __float2half(rsqrtf(__half2float(
      __hadd(norm_squares[0], epsilon))));
  const half key_inverse = __float2half(rsqrtf(__half2float(
      __hadd(norm_squares[kHeadDim], epsilon))));
  normalized_query[column] = __half2float(
      __hmul(raw_query, query_inverse)) * kQueryScale;
  normalized_key[column] = __half2float(__hmul(raw_key, key_inverse));
  __syncthreads();

  const int coefficient_offset = batch * kValueHeads + value_head;
  const float beta_value = __half2float(beta_input[coefficient_offset]);
  const float beta = __half2float(__float2half(
      1.0f / (1.0f + expf(-beta_value))));
  const float decay_x = __half2float(decay_input[coefficient_offset])
                        + __half2float(dt_bias[value_head]);
  const float softplus =
      decay_x > 20.0f ? decay_x : log1pf(expf(decay_x));
  const float decay = expf(
      -expf(__half2float(a_log[value_head])) * softplus);

  float memory = 0.0f;
#pragma unroll
  for (int row = 0; row < kHeadDim; ++row) {
    const int index = state_offset + row * kHeadDim + column;
    const float decayed = state[index] * decay;
    memory += normalized_key[row] * decayed;
  }

  const float value = __half2float(mixed_qkv[value_offset + column]);
  const float delta = (value - memory) * beta;
  float result = 0.0f;
#pragma unroll
  for (int row = 0; row < kHeadDim; ++row) {
    const int index = state_offset + row * kHeadDim + column;
    const float decayed = state[index] * decay;
    const float updated = decayed + normalized_key[row] * delta;
    state[index] = updated;
    result += normalized_query[row] * updated;
  }
  output[vector_offset + column] = result;
}

void check_half_matrix(const torch::Tensor& tensor, const char* name,
                       int64_t width) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16,
              name, " must have dtype float16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == 2 && tensor.size(1) == width,
              name, " must have shape (batch, ", width, ")");
}

void check_half_vector(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16,
              name, " must have dtype float16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == kValueHeads,
              name, " must have shape (", kValueHeads, ")");
}

}  // namespace

torch::Tensor packed_decode(torch::Tensor state,
                            const torch::Tensor& mixed_qkv,
                            const torch::Tensor& beta_input,
                            const torch::Tensor& decay_input,
                            const torch::Tensor& a_log,
                            const torch::Tensor& dt_bias) {
  TORCH_CHECK(state.is_cuda(), "state must be a CUDA tensor");
  TORCH_CHECK(state.scalar_type() == torch::kFloat32,
              "state must have dtype float32");
  TORCH_CHECK(state.is_contiguous(), "state must be contiguous");
  TORCH_CHECK(state.dim() == 4 && state.size(0) == 1
                  && state.size(1) == kValueHeads
                  && state.size(2) == kHeadDim
                  && state.size(3) == kHeadDim,
              "state must have shape (1, 8, 128, 128)");
  check_half_matrix(mixed_qkv, "mixed_qkv", kMixedDim);
  check_half_matrix(beta_input, "beta_input", kValueHeads);
  check_half_matrix(decay_input, "decay_input", kValueHeads);
  check_half_vector(a_log, "a_log");
  check_half_vector(dt_bias, "dt_bias");
  TORCH_CHECK(mixed_qkv.size(0) == 1 && beta_input.size(0) == 1
                  && decay_input.size(0) == 1,
              "packed decode only supports one sequence");
  TORCH_CHECK(state.device() == mixed_qkv.device()
                  && state.device() == beta_input.device()
                  && state.device() == decay_input.device()
                  && state.device() == a_log.device()
                  && state.device() == dt_bias.device(),
              "all inputs must be on the same device");

  torch::Tensor output = torch::empty(
      {1, kValueHeads, kHeadDim}, state.options());
  gdn_packed_decode_kernel<<<kValueHeads, kHeadDim, 0,
                             at::cuda::getCurrentCUDAStream()>>>(
      state.data_ptr<float>(),
      reinterpret_cast<const half*>(mixed_qkv.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(beta_input.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(decay_input.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(a_log.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(dt_bias.data_ptr<at::Half>()),
      output.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("packed_decode", &packed_decode,
             "Packed Qwen3.6 GDN single-token decode");
}
