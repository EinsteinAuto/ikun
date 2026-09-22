#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kSmallGridBlocks = 256;
constexpr int kSmallGridMaxSeqLen = 96 * 1024;

__global__ void paged_kv_gather_kernel(
    const __half* key_cache, const __half* value_cache,
    const int* block_table, float* key_output, float* value_output,
    int seq_len, int num_kv_heads, int head_size, int block_size,
    int key_pack) {
  const int64_t total =
      static_cast<int64_t>(seq_len) * num_kv_heads * head_size;
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int dim = index % head_size;
    const int token = (index / head_size) % seq_len;
    const int kv_head = index / (static_cast<int64_t>(head_size) * seq_len);
    const int logical_block = token / block_size;
    const int block_offset = token % block_size;
    const int physical_block = block_table[logical_block];

    const int64_t key_index =
        (((static_cast<int64_t>(physical_block) * num_kv_heads + kv_head)
           * (head_size / key_pack) + dim / key_pack)
          * block_size + block_offset) * key_pack + dim % key_pack;
    const int64_t value_index =
        ((static_cast<int64_t>(physical_block) * num_kv_heads + kv_head)
         * head_size + dim) * block_size + block_offset;
    const int64_t key_output_index =
        (static_cast<int64_t>(kv_head) * head_size + dim) * seq_len + token;
    const int64_t value_output_index =
        (static_cast<int64_t>(kv_head) * seq_len + token) * head_size + dim;

    key_output[key_output_index] = __half2float(key_cache[key_index]);
    value_output[value_output_index] = __half2float(value_cache[value_index]);
  }
}

void check_half_cuda_contiguous(const torch::Tensor& tensor,
                                const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16,
              name, " must have dtype float16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

std::vector<torch::Tensor> gather_paged_kv(
    const torch::Tensor& key_cache, const torch::Tensor& value_cache,
    const torch::Tensor& block_table, int64_t seq_len) {
  check_half_cuda_contiguous(key_cache, "key_cache");
  check_half_cuda_contiguous(value_cache, "value_cache");
  TORCH_CHECK(block_table.is_cuda(), "block_table must be a CUDA tensor");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt32,
              "block_table must have dtype int32");
  TORCH_CHECK(block_table.is_contiguous(), "block_table must be contiguous");
  TORCH_CHECK(key_cache.dim() == 5,
              "key_cache must have shape (blocks, kv_heads, d/x, block, x)");
  TORCH_CHECK(value_cache.dim() == 4,
              "value_cache must have shape (blocks, kv_heads, d, block)");
  TORCH_CHECK(block_table.dim() == 1,
              "block_table must be a one-dimensional row");
  TORCH_CHECK(key_cache.size(0) == value_cache.size(0),
              "key/value block counts differ");
  TORCH_CHECK(key_cache.size(1) == value_cache.size(1),
              "key/value KV-head counts differ");
  TORCH_CHECK(key_cache.size(3) == value_cache.size(3),
              "key/value block sizes differ");
  TORCH_CHECK(key_cache.size(2) * key_cache.size(4) == value_cache.size(2),
              "key/value head sizes differ");
  TORCH_CHECK(seq_len > 0, "seq_len must be positive");

  const int block_size = static_cast<int>(value_cache.size(3));
  const int64_t required_blocks = (seq_len + block_size - 1) / block_size;
  TORCH_CHECK(required_blocks <= block_table.numel(),
              "block_table is too short for seq_len");
  const int num_kv_heads = static_cast<int>(value_cache.size(1));
  const int head_size = static_cast<int>(value_cache.size(2));
  const int key_pack = static_cast<int>(key_cache.size(4));

  auto output_options = key_cache.options().dtype(torch::kFloat32);
  auto key_output = torch::empty(
      {num_kv_heads, head_size, seq_len}, output_options);
  auto value_output = torch::empty(
      {num_kv_heads, seq_len, head_size}, output_options);
  const int64_t total = seq_len * num_kv_heads * head_size;
  const int grid_cap =
      seq_len <= kSmallGridMaxSeqLen ? kSmallGridBlocks : 65535;
  const int blocks = static_cast<int>(std::min<int64_t>(
      (total + kThreads - 1) / kThreads, grid_cap));
  paged_kv_gather_kernel<<<blocks, kThreads, 0,
                           at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(key_cache.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(value_cache.data_ptr<at::Half>()),
      block_table.data_ptr<int>(), key_output.data_ptr<float>(),
      value_output.data_ptr<float>(), static_cast<int>(seq_len),
      num_kv_heads, head_size, block_size, key_pack);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {key_output, value_output};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("gather", &gather_paged_kv,
             "Gather paged FP16 K/V directly into FP32 attention layouts");
}
