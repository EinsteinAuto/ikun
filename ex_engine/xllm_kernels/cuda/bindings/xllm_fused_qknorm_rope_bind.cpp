// xllm_fused_qknorm_rope_bind.cpp — pybind11 for fused QK-Norm + RoPE kernel
// Source: upstream_ref/xllm/xllm/core/kernels/cuda/fused_qknorm_rope.cu
// Saves 4 kernel launches per layer (separate q_norm, k_norm, q_rope, k_rope)
// Qwen3.5 has 32 full-attention layers → saves 128 kernel launches per forward

#include <torch/extension.h>

namespace xllm::kernel::cuda {
void fused_qk_norm_rope(
    torch::Tensor& qkv,
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t num_heads_v,
    int64_t head_dim,
    double eps,
    const torch::Tensor& q_weight,
    const torch::Tensor& k_weight,
    const torch::Tensor& cos_sin_cache,
    bool interleaved,
    const torch::Tensor& position_ids);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_qk_norm_rope",
          &xllm::kernel::cuda::fused_qk_norm_rope,
          "Fused QK-Norm + RoPE (xllm CUDA kernel)",
          py::arg("qkv"),
          py::arg("num_heads_q"),
          py::arg("num_heads_k"),
          py::arg("num_heads_v"),
          py::arg("head_dim"),
          py::arg("eps") = 1e-6,
          py::arg("q_weight"),
          py::arg("k_weight"),
          py::arg("cos_sin_cache"),
          py::arg("interleaved") = false,
          py::arg("position_ids"));
}
