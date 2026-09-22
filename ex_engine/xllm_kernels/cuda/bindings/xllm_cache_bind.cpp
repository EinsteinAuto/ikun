// xllm_cache_bind.cpp
#include <torch/extension.h>

namespace xllm::kernel::cuda {
void reshape_paged_cache(torch::Tensor slot_ids, torch::Tensor keys,
                         torch::Tensor values, torch::Tensor key_cache,
                         torch::Tensor value_cache);
void block_copy(torch::Tensor key_cache_ptrs, torch::Tensor value_cache_ptrs,
                torch::Tensor src_block_indices, torch::Tensor dst_block_indices,
                torch::Tensor cum_sum, int64_t numel_per_block,
                torch::ScalarType cache_dtype);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reshape_paged_cache", &xllm::kernel::cuda::reshape_paged_cache,
          "Reshape Paged KV Cache");
    m.def("block_copy", &xllm::kernel::cuda::block_copy,
          "Block Copy for KV Cache");
}
