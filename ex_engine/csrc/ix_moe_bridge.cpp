// ix_moe_bridge.cpp — Full MoE pipeline bridge to ixformer C++ API
//
// Exposes ALL 6 MoE functions from ixformer::infer (ixformer.h):
//   1. topk_softmax          — fused routing
//   2. moe_compute_token_index_api — permutation maps (src_dst, dst_src)
//   3. moe_expand_input      — gather tokens by expert
//   4. moe_w16a16_group_gemm — batched expert GEMM
//   5. silu_and_mul           — fused activation
//   6. moe_output_reduce_sum — weighted scatter-add
//
// Source: upstream_ref/xllm/xllm/core/kernels/ilu/ixformer.h
// Usage:  upstream_ref/xllm/xllm/core/kernels/ilu/fused_moe.cpp
//         upstream_ref/xllm/xllm/core/layers/ilu/fused_moe.cpp

#include <torch/extension.h>
#include <tuple>
#include <vector>
#include <optional>

static const std::optional<torch::Tensor> kNoneTensor = {};

// Forward-declare ixformer C++ API (from base image SDK)
namespace ixformer {
namespace infer {

void topk_softmax(torch::Tensor& topk_weights,
                  torch::Tensor& topk_indices,
                  torch::Tensor& token_expert_indices,
                  torch::Tensor& gating_output,
                  bool renormalize);

void moe_compute_token_index_api(
    torch::Tensor& topk_ids,
    torch::Tensor& src_dst,
    torch::Tensor& dst_src,
    torch::Tensor& expert_sizes_gpu,
    const std::optional<torch::Tensor>& expert_mask,
    const std::optional<torch::Tensor>& expert_sizes_cpu,
    const std::optional<torch::Tensor>& expand_tokens_gpu,
    int64_t start_expert_id,
    int64_t end_expert_id,
    int64_t num_experts);

void moe_expand_input(torch::Tensor outputs,
                      torch::Tensor inputs,
                      torch::Tensor dst_to_src,
                      const std::optional<torch::Tensor>& src_to_dst,
                      int64_t dst_tokens,
                      int64_t expand_factor);

void moe_w16a16_group_gemm(torch::Tensor output,
                           torch::Tensor inputs,
                           torch::Tensor weights,
                           torch::Tensor tokens_per_experts,
                           const std::optional<torch::Tensor>& dst_to_src,
                           const std::optional<torch::Tensor>& bias,
                           std::string format,
                           int64_t persistent,
                           int64_t output_n);

void moe_output_reduce_sum(torch::Tensor outputs,
                           torch::Tensor inputs,
                           const std::optional<torch::Tensor>& mul_weight,
                           const std::optional<torch::Tensor>& mask,
                           const std::optional<torch::Tensor>& extra_residual,
                           double scaling_factor);

void silu_and_mul(torch::Tensor& input, torch::Tensor& output);

}  // namespace infer
}  // namespace ixformer

// ============================================================================
// Python-callable wrappers
// ============================================================================

// 1. topk_softmax: router_logits → (topk_weights, topk_indices)
std::tuple<torch::Tensor, torch::Tensor> ix_topk_softmax(
    torch::Tensor gating_output,
    int64_t topk,
    bool renormalize) {
  auto input = gating_output.to(torch::kFloat32).contiguous();
  int64_t num_tokens = input.size(0);

  auto topk_weights = torch::empty({num_tokens, topk},
      torch::dtype(torch::kFloat32).device(input.device()));
  auto topk_indices = torch::empty({num_tokens, topk},
      torch::dtype(torch::kInt32).device(input.device()));
  auto token_expert_indices = torch::empty({num_tokens, topk},
      torch::dtype(torch::kInt32).device(input.device()));

  ixformer::infer::topk_softmax(
      topk_weights, topk_indices, token_expert_indices, input, false);

  // Renormalize (match xllm/kernels/ilu/fused_moe.cpp line 55)
  if (renormalize) {
    auto row_sum = topk_weights.sum(-1, /*keepdim=*/true);
    topk_weights = topk_weights / row_sum;
  }

  return std::make_tuple(topk_weights, topk_indices);
}

// 2. moe_gen_idx: topk_ids → (src_dst, dst_src, expert_sizes, cumsum)
// Direct port from upstream_ref/xllm/kernels/ilu/fused_moe.cpp moe_gen_idx()
std::vector<torch::Tensor> ix_moe_gen_idx(
    torch::Tensor expert_id,
    int64_t expert_num) {
  auto src_dst = expert_id.new_empty({expert_id.numel()});
  auto dst_src = torch::empty_like(src_dst);
  auto expert_sizes_gpu = expert_id.new_empty({expert_num});
  auto expert_sizes_gpu_cumsum = expert_id.new_zeros({expert_id.numel() + 1});

  ixformer::infer::moe_compute_token_index_api(
      expert_id, src_dst, dst_src, expert_sizes_gpu,
      /*expert_mask=*/kNoneTensor,
      /*expert_sizes_cpu=*/kNoneTensor,
      /*expand_tokens_gpu=*/kNoneTensor,
      0, expert_num, expert_num);

  expert_sizes_gpu_cumsum = expert_sizes_gpu.cumsum(-1);
  return {src_dst, dst_src, expert_sizes_gpu, expert_sizes_gpu_cumsum};
}

// 3. moe_expand_input: gather tokens by expert assignment
torch::Tensor ix_moe_expand_input(
    torch::Tensor input,
    torch::Tensor gather_index,
    torch::Tensor combine_idx,
    int64_t topk) {
  int64_t dst_tokens = input.size(0) * topk;
  auto output = input.new_empty({dst_tokens, input.size(1)});

  ixformer::infer::moe_expand_input(
      output, input, combine_idx, gather_index, dst_tokens, topk);
  return output;
}

// 4. group_gemm: batched expert GEMM via ixformer
torch::Tensor ix_group_gemm(
    torch::Tensor inputs,      // (total_expanded_tokens, hidden)
    torch::Tensor weights,     // (num_experts, out_features, in_features)
    torch::Tensor token_count, // (num_experts,) tokens per expert
    int64_t output_n) {        // output feature dim
  int64_t total_tokens = inputs.size(0);
  auto output = inputs.new_empty({total_tokens, output_n});

  ixformer::infer::moe_w16a16_group_gemm(
      output, inputs, weights, token_count,
      /*dst_to_src=*/kNoneTensor,
      /*bias=*/kNoneTensor,
      /*format=*/"TN",
      /*persistent=*/0,
      /*output_n=*/output_n);
  return output;
}

// 5. silu_and_mul: fused activation (gated SiLU for MoE)
torch::Tensor ix_silu_and_mul(torch::Tensor input) {
  int64_t half_dim = input.size(-1) / 2;
  auto output = input.new_empty({input.size(0), half_dim});
  ixformer::infer::silu_and_mul(input, output);
  return output;
}

// 6. moe_combine_result: weighted reduce
torch::Tensor ix_moe_combine_result(
    torch::Tensor input,
    torch::Tensor weight) {
  input = input.view({-1, weight.size(1), input.size(1)});
  auto output = input.new_empty({input.size(0), input.size(2)});

  ixformer::infer::moe_output_reduce_sum(
      output, input, weight,
      /*mask=*/kNoneTensor,
      /*extra_residual=*/kNoneTensor,
      /*scaling_factor=*/1.0);
  return output;
}

// ============================================================================
// FULL fused MoE forward — complete pipeline matching xllm
// ============================================================================
// This replaces the entire _pure_pytorch_experts() in qwen3_5.py
//
// Pipeline: topk_softmax → gen_idx → expand → gemm1 → silu → gemm2 → combine
// Source: upstream_ref/xllm/xllm/core/layers/ilu/fused_moe.cpp forward_experts()

torch::Tensor ix_fused_moe_forward(
    torch::Tensor hidden_states,   // (T, H)
    torch::Tensor router_logits,   // (T, E)
    torch::Tensor w13,             // (E, 2*I, H)  gate_up weight
    torch::Tensor w2,              // (E, H, I)    down weight
    int64_t topk,
    int64_t num_experts,
    bool renormalize) {

  // Step 1: routing
  auto [topk_weights, topk_ids] = ix_topk_softmax(router_logits, topk, renormalize);

  // Step 2: build permutation
  auto idx = ix_moe_gen_idx(topk_ids.view({-1}), num_experts);
  auto gather_idx = idx[0];   // src_dst
  auto combine_idx = idx[1];  // dst_src
  auto expert_sizes = idx[2]; // (E,)

  // Step 3: expand hidden states by expert assignment
  auto expanded = ix_moe_expand_input(
      hidden_states, gather_idx, combine_idx, topk);

  // Step 4: group GEMM 1 — gate_up projection
  int64_t gate_up_dim = w13.size(1);  // 2*I
  auto gemm1_out = ix_group_gemm(expanded, w13, expert_sizes, gate_up_dim);

  // Step 5: activation — SiLU(gate) * up
  auto act_out = ix_silu_and_mul(gemm1_out);

  // Step 6: group GEMM 2 — down projection
  int64_t hidden_dim = w2.size(1);  // H
  auto gemm2_out = ix_group_gemm(act_out, w2, expert_sizes, hidden_dim);

  // Step 7: combine — weighted scatter back
  auto output = ix_moe_combine_result(gemm2_out, topk_weights);

  return output;
}

// ============================================================================
// Module registration
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("topk_softmax", &ix_topk_softmax,
        "Fused topk+softmax via ixformer C++ API",
        py::arg("gating_output"), py::arg("topk"), py::arg("renormalize") = true);

  m.def("moe_gen_idx", &ix_moe_gen_idx,
        "Build expert permutation maps (src_dst, dst_src, sizes, cumsum)",
        py::arg("expert_id"), py::arg("expert_num"));

  m.def("moe_expand_input", &ix_moe_expand_input,
        "Gather tokens by expert assignment",
        py::arg("input"), py::arg("gather_index"), py::arg("combine_idx"), py::arg("topk"));

  m.def("group_gemm", &ix_group_gemm,
        "Batched expert GEMM via ixformer group_gemm",
        py::arg("inputs"), py::arg("weights"), py::arg("token_count"), py::arg("output_n"));

  m.def("silu_and_mul", &ix_silu_and_mul,
        "Fused SiLU gate activation",
        py::arg("input"));

  m.def("moe_combine_result", &ix_moe_combine_result,
        "Weighted reduce for MoE output",
        py::arg("input"), py::arg("weight"));

  m.def("fused_moe_forward", &ix_fused_moe_forward,
        "Full fused MoE forward pipeline (topk → expand → gemm → act → gemm → combine)",
        py::arg("hidden_states"), py::arg("router_logits"),
        py::arg("w13"), py::arg("w2"),
        py::arg("topk"), py::arg("num_experts"), py::arg("renormalize") = true);
}
