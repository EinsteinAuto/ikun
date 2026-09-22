// xllm_rope_bind.cpp
#include <torch/extension.h>
#include <optional>

namespace xllm::kernel::cuda {
void rotary_embedding(torch::Tensor& positions, torch::Tensor& query,
                      std::optional<torch::Tensor> key,
                      torch::Tensor& cos_sin_cache, bool is_neox);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rotary_embedding", &xllm::kernel::cuda::rotary_embedding,
          "Rotary Position Embedding (xllm CUDA kernel)",
          py::arg("positions"), py::arg("query"),
          py::arg("key"), py::arg("cos_sin_cache"),
          py::arg("is_neox") = true);
}
