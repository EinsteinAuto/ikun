// corex_moe_topk_softmax.cu — CUB-based fused topk+softmax for MoE routing
//
// Source: upstream_ref/xllm/xllm/core/kernels/cuda/moe/moe_topk_softmax_kernels.cuh
// Adapted from vllm v0.7.3 / TensorRT-LLM v0.7.1 topk_softmax_kernels.cu
//
// Builds as corex_moe_topk_softmax.so via corex clang++ (ivcore10)
// Loaded at runtime: from vllm import corex_moe_topk_softmax
//
// Replaces: torch.softmax + torch.topk in qwen3_5.py MoE routing
// Performance: single fused kernel vs 2 separate PyTorch ops

#include "moe_topk_softmax_kernels.cuh"
#include <torch/extension.h>

using namespace xllm::kernel::cuda;

// Python-facing wrapper matching wudixzy corex_*.so convention
std::tuple<torch::Tensor, torch::Tensor> moe_topk_softmax(
    torch::Tensor gating_output,  // (T, num_experts)
    int64_t topk,
    bool renormalize) {

  int64_t num_tokens = gating_output.size(0);

  auto topk_weights = torch::empty(
      {num_tokens, topk},
      torch::dtype(torch::kFloat32).device(gating_output.device()));
  auto topk_indices = torch::empty(
      {num_tokens, topk},
      torch::dtype(torch::kInt32).device(gating_output.device()));

  topk_softmax(
      topk_weights, topk_indices, gating_output,
      renormalize,
      /*moe_softcapping=*/0.0,
      /*correction_bias=*/std::nullopt);

  return std::make_tuple(topk_weights, topk_indices);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_topk_softmax", &moe_topk_softmax,
        "CUB-based fused topk+softmax for MoE routing (xllm upstream)");
}
