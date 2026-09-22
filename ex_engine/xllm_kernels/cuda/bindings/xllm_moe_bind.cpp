// xllm_moe_bind.cpp — pybind11 for MoE CUDA kernels
#include <torch/extension.h>
#include <optional>
#include <tuple>

namespace xllm::kernel::cuda {
std::tuple<torch::Tensor, torch::Tensor> moe_fused_topk(
    torch::Tensor& gating_output, int64_t topk, bool renormalize,
    const std::optional<torch::Tensor>& correction_bias,
    const std::string& scoring_func);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> moe_compute_index(
    const torch::Tensor& expert_id, int64_t num_experts);

torch::Tensor moe_combine_result(
    const torch::Tensor& gemm2, const torch::Tensor& reduce_weight,
    int64_t N, int32_t topk);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("moe_fused_topk", &xllm::kernel::cuda::moe_fused_topk,
          "MoE fused topk (softmax or sigmoid routing)",
          py::arg("gating_output"), py::arg("topk"),
          py::arg("renormalize") = true,
          py::arg("correction_bias") = py::none(),
          py::arg("scoring_func") = "softmax");
    m.def("moe_compute_index", &xllm::kernel::cuda::moe_compute_index,
          "MoE compute permutation index (histogram + prefix_sum + place)",
          py::arg("expert_id"), py::arg("num_experts"));
    m.def("moe_combine_result", &xllm::kernel::cuda::moe_combine_result,
          "MoE combine (reorder + weighted sum)",
          py::arg("gemm2"), py::arg("reduce_weight"),
          py::arg("N"), py::arg("topk"));
}
