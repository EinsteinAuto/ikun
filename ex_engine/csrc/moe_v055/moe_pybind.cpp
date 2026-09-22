/*
 * moe_pybind.cpp — pybind11 entry for vllm MoE CUDA kernels
 * 
 * Compiled via torch.utils.cpp_extension.load() on BI-V100 (CoreX)
 * Exposes:
 *   - topk_softmax(topk_weights, topk_indices, token_expert_indices, gating_output)
 *   - moe_align_block_size(topk_ids, num_experts, block_size, sorted_token_ids, experts_ids, num_tokens_post_pad)
 *
 * Source: vllm v0.5.5 csrc/moe/ (torch::Tensor API, pre-libtorch_stable)
 */

#include <torch/extension.h>

// Forward declarations matching vllm v0.5.5 signatures
void topk_softmax(torch::Tensor& topk_weights,
                  torch::Tensor& topk_indices,
                  torch::Tensor& token_expert_indices,
                  torch::Tensor& gating_output);

void moe_align_block_size(torch::Tensor topk_ids,
                           int64_t num_experts,
                           int64_t block_size,
                           torch::Tensor sorted_token_ids,
                           torch::Tensor experts_ids,
                           torch::Tensor num_tokens_post_pad);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("topk_softmax", &topk_softmax,
          "MoE topk softmax (vllm v0.5.5 CUDA kernel)",
          py::arg("topk_weights"),
          py::arg("topk_indices"),
          py::arg("token_expert_indices"),
          py::arg("gating_output"));
    m.def("moe_align_block_size", &moe_align_block_size,
          "MoE align block size (vllm v0.5.5 CUDA kernel)",
          py::arg("topk_ids"),
          py::arg("num_experts"),
          py::arg("block_size"),
          py::arg("sorted_token_ids"),
          py::arg("experts_ids"),
          py::arg("num_tokens_post_pad"));
}
