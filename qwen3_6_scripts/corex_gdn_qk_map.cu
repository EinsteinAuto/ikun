#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

constexpr int kHeadDim = 128;
constexpr float kQueryScale = 0.08838834764831845f;

__global__ void qk_map_kernel(const half* query, const half* key,
                              float* output, int batch, int key_heads,
                              int value_heads, int expand_ratio) {
  const int elements = batch * value_heads * kHeadDim;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  const int dim = index % kHeadDim;
  const int value_head_index = index / kHeadDim;
  const int value_head = value_head_index % value_heads;
  const int batch_index = value_head_index / value_heads;
  const int key_head = value_head / expand_ratio;
  const int source = ((batch_index * key_heads + key_head) * kHeadDim + dim);
  output[index] = __half2float(query[source]) * kQueryScale;
  output[elements + index] = __half2float(key[source]);
}

void check_input(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16,
              name, " must have dtype float16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == 3 && tensor.size(2) == kHeadDim,
              name, " must have shape (batch, key_heads, 128)");
}

}  // namespace

torch::Tensor qk_map(const torch::Tensor& query,
                     const torch::Tensor& key,
                     int64_t value_heads_arg) {
  check_input(query, "query");
  check_input(key, "key");
  TORCH_CHECK(query.sizes() == key.sizes(),
              "query and key shapes must match");
  const int batch = static_cast<int>(query.size(0));
  const int key_heads = static_cast<int>(query.size(1));
  const int value_heads = static_cast<int>(value_heads_arg);
  TORCH_CHECK(value_heads > 0 && value_heads % key_heads == 0,
              "value_heads must be divisible by key_heads");

  torch::Tensor output = torch::empty(
      {2, batch, value_heads, kHeadDim},
      query.options().dtype(torch::kFloat32));
  const int elements = batch * value_heads * kHeadDim;
  constexpr int threads = 256;
  const int blocks = (elements + threads - 1) / threads;
  qk_map_kernel<<<blocks, threads, 0,
                  at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(key.data_ptr<at::Half>()),
      output.data_ptr<float>(), batch, key_heads, value_heads,
      value_heads / key_heads);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("qk_map", &qk_map,
             "Map normalized FP16 key heads to FP32 value heads");
}
