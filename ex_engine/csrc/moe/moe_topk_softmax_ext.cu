// ex_engine/csrc/moe/moe_topk_softmax_ext.cu
//
// Torch extension wrapper for xllm's topk_gating_softmax kernel.
// Compiles via torch.utils.cpp_extension.load() on BI-V100.
//
// Interface matches vllm's _custom_ops.topk_softmax():
//   topk_softmax(topk_weights, topk_ids, token_expert_indices, gating_output)

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// Include the kernel (adapted from xllm, CHECK→TORCH_CHECK)
#include "moe_topk_softmax_kernels.cuh"

// ---------------------------------------------------------------------------
// Python-facing wrapper: matches _custom_ops.topk_softmax signature exactly
// ---------------------------------------------------------------------------
void topk_softmax_ext(
    torch::Tensor& topk_weights,           // [num_tokens, topk] float32 output
    torch::Tensor& topk_ids,               // [num_tokens, topk] int32 output
    torch::Tensor& token_expert_indices,    // [num_tokens, topk] int32 output
    torch::Tensor& gating_output,           // [num_tokens, num_experts] input
    bool renormalize = false
) {
    // Call the xllm kernel
    xllm::kernel::cuda::topk_softmax(
        topk_weights,
        topk_ids,
        gating_output,
        renormalize,
        0.0,        // moe_softcapping (unused for Qwen3.5)
        std::nullopt // correction_bias
    );

    // Fill token_expert_indices: flatten assignment
    // token_expert_indices[i][j] = i * topk + j
    const int num_tokens = topk_weights.size(0);
    const int topk = topk_weights.size(1);
    auto arange_tokens = torch::arange(num_tokens, topk_ids.options().dtype(torch::kInt32));
    auto arange_topk = torch::arange(topk, topk_ids.options().dtype(torch::kInt32));
    token_expert_indices.copy_(
        arange_tokens.unsqueeze(1) * topk + arange_topk.unsqueeze(0)
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("topk_softmax", &topk_softmax_ext,
          "Fused softmax + topk for MoE routing (xllm CUB kernel)",
          py::arg("topk_weights"),
          py::arg("topk_ids"),
          py::arg("token_expert_indices"),
          py::arg("gating_output"),
          py::arg("renormalize") = false);
}
