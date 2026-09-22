// ix_full_bridge_v2.cpp — Bridge to ixformer C++ functions + MoE pipeline
//
// Forward declarations use REAL symbols from nm -D symbol dumps:
//   _ixformer_torch.so → namespace ixformer_torch_ext (7 functions)
//   moe_ops_impl.cu    → namespace ixformer::infer    (5 MoE functions, self-compiled)
//
// Symbol dump verified:
//   ixformer_torch_ext::silu_and_mul_forward(at::Tensor&, at::Tensor&)
//   ixformer_torch_ext::rms_norm_forward(at::Tensor&, at::Tensor&, at::Tensor&, double)
//   ixformer_torch_ext::fused_add_rms_norm_forward(at::Tensor&, at::Tensor&, at::Tensor&, double, double)
//   ixformer_torch_ext::ixformer_linear(at::Tensor&, at::Tensor&, c10::optional<at::Tensor>, c10::optional<at::Tensor>)
//   ixformer_torch_ext::ixformer_linear_ex(at::Tensor&, at::Tensor&, c10::optional<at::Tensor>)
//   ixformer_torch_ext::vllm_rotary_embedding_neox(at::Tensor&, at::Tensor&, at::Tensor&, long, at::Tensor&, long, bool)
//   ixformer_torch_ext::vllm_cache_ops_reshape_and_cache(at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, long, long)
//   ixformer_torch_ext::vllm_single_query_cached_kv_attention(13 params — see below)
//
// NOT available in any .so (confirmed by nm -D on all 4 .so files):
//   ixinfer_flash_attn_unpad_with_block_tables — DOES NOT EXIST
//   xllm_paged_attention — DOES NOT EXIST
//   topk_softmax, moe_w16a16_group_gemm, etc — NOT in libixformer.so
//     (provided by moe_ops_impl.cu instead)

#include <torch/extension.h>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

// ============================================================================
// Forward declarations — ixformer_torch_ext namespace from _ixformer_torch.so
// Signatures EXACTLY match nm -D | c++filt output
// ============================================================================
namespace ixformer_torch_ext {

// silu_and_mul_forward(at::Tensor&, at::Tensor&)
void silu_and_mul_forward(at::Tensor& input, at::Tensor& output);

// rms_norm_forward(at::Tensor&, at::Tensor&, at::Tensor&, double)
// Real ixformer signature order: (input, weight, output, eps)
void rms_norm_forward(at::Tensor& input, at::Tensor& weight,
                      at::Tensor& output, double eps);

// fused_add_rms_norm_forward(at::Tensor&, at::Tensor&, at::Tensor&, double, double)
void fused_add_rms_norm_forward(at::Tensor& input, at::Tensor& residual,
                                at::Tensor& weight, double eps, double alpha);

// ixformer_linear(at::Tensor&, at::Tensor&, c10::optional<at::Tensor> const&, c10::optional<at::Tensor> const&)
at::Tensor ixformer_linear(at::Tensor& input, at::Tensor& weight,
                           c10::optional<at::Tensor> const& bias,
                           c10::optional<at::Tensor> const& out);

// ixformer_linear_ex(at::Tensor&, at::Tensor&, c10::optional<at::Tensor> const&)
at::Tensor ixformer_linear_ex(at::Tensor& input, at::Tensor& weight,
                              c10::optional<at::Tensor> const& bias);

// vllm_rotary_embedding_neox(at::Tensor&, at::Tensor&, at::Tensor&, long, at::Tensor&, long, bool)
void vllm_rotary_embedding_neox(at::Tensor& positions, at::Tensor& query,
                                at::Tensor& key, int64_t head_size,
                                at::Tensor& cos_sin_cache,
                                int64_t max_position, bool is_neox);

// vllm_cache_ops_reshape_and_cache(at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, long, long)
void vllm_cache_ops_reshape_and_cache(at::Tensor& key, at::Tensor& value,
                                      at::Tensor& key_cache,
                                      at::Tensor& value_cache,
                                      at::Tensor& slot_mapping,
                                      int64_t key_token_stride,
                                      int64_t value_token_stride);

// vllm_single_query_cached_kv_attention(at::Tensor& x13)
// Full signature from nm -D:
//   (at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&,
//    double, at::Tensor&, at::Tensor&, long, long, long, bool,
//    c10::optional<at::Tensor> const&)
void vllm_single_query_cached_kv_attention(
    at::Tensor& output, at::Tensor& query,
    at::Tensor& key_cache, at::Tensor& value_cache,
    at::Tensor& head_mapping, double scale,
    at::Tensor& block_tables, at::Tensor& context_lens,
    int64_t block_size, int64_t max_context_len, int64_t num_kv_heads,
    bool is_neox,
    c10::optional<at::Tensor> const& alibi_slopes);

}  // namespace ixformer_torch_ext

// ============================================================================
// Forward declarations — ixformer::infer namespace from moe_ops_impl.cu
// These 5 MoE functions are compiled from our own CUDA code, NOT from .so
// ============================================================================
namespace ixformer { namespace infer {

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

}}  // namespace ixformer::infer


// ============================================================================
// Python wrappers — thin wrappers matching ix_bridge.py's expected API
// ============================================================================

// --- silu_and_mul ---
torch::Tensor ix_silu_and_mul(torch::Tensor input) {
    int64_t half_dim = input.size(-1) / 2;
    auto output = input.new_empty({input.size(0), half_dim});
    ixformer_torch_ext::silu_and_mul_forward(input, output);
    return output;
}

// --- rms_norm ---
void ix_rms_norm(torch::Tensor output, torch::Tensor input,
                 torch::Tensor weight, double eps) {
    // pybind receives (output, input, weight, eps)
    // ixformer expects (input, weight, output, eps)
    ixformer_torch_ext::rms_norm_forward(input, weight, output, eps);
}

// --- fused_add_rms_norm ---
void ix_fused_add_rms_norm(torch::Tensor input, torch::Tensor residual,
                           torch::Tensor weight, double eps) {
    ixformer_torch_ext::fused_add_rms_norm_forward(
        input, residual, weight, eps, /*alpha=*/1.0);
}

// --- linear ---
torch::Tensor ix_linear(torch::Tensor input, torch::Tensor weight,
                        const c10::optional<torch::Tensor>& bias) {
    auto input_2d = input.view({-1, input.size(-1)});
    int64_t m = input_2d.size(0);
    if (m <= 1 && !bias.has_value()) {
        return ixformer_torch_ext::ixformer_linear_ex(input, weight, bias);
    }
    return ixformer_torch_ext::ixformer_linear(
        input, weight, bias, /*out=*/c10::optional<at::Tensor>());
}

// --- rotary_embedding ---
void ix_rotary_embedding(torch::Tensor positions, torch::Tensor query,
                         torch::Tensor key, int64_t head_size,
                         torch::Tensor cos_sin_cache, bool is_neox) {
    int64_t max_position = cos_sin_cache.size(0);
    ixformer_torch_ext::vllm_rotary_embedding_neox(
        positions, query, key, head_size, cos_sin_cache, max_position, is_neox);
}

// --- reshape_and_cache ---
void ix_reshape_and_cache(torch::Tensor key, torch::Tensor value,
                          torch::Tensor key_cache, torch::Tensor value_cache,
                          torch::Tensor slot_mapping) {
    int64_t key_token_stride = 1;
    for (int i = 1; i < key.dim(); i++) key_token_stride *= key.size(i);
    int64_t value_token_stride = 1;
    for (int i = 1; i < value.dim(); i++) value_token_stride *= value.size(i);

    ixformer_torch_ext::vllm_cache_ops_reshape_and_cache(
        key, value, key_cache, value_cache, slot_mapping,
        key_token_stride, value_token_stride);
}

// --- paged_attention (decode only — no prefill available in .so) ---
void ix_paged_attention(
    torch::Tensor output, torch::Tensor query,
    torch::Tensor key_cache, torch::Tensor value_cache,
    torch::Tensor head_mapping, double scale,
    torch::Tensor block_tables, torch::Tensor context_lens,
    int64_t block_size, int64_t max_context_len, int64_t num_kv_heads,
    const c10::optional<torch::Tensor>& alibi_slopes) {
    ixformer_torch_ext::vllm_single_query_cached_kv_attention(
        output, query, key_cache, value_cache,
        head_mapping, scale, block_tables, context_lens,
        block_size, max_context_len, num_kv_heads,
        /*is_neox=*/true, alibi_slopes);
}


// ============================================================================
// MoE wrappers — call moe_ops_impl.cu implementations
// ============================================================================

// --- topk_softmax ---
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
ix_topk_softmax(torch::Tensor gating_output, int64_t topk, bool renormalize) {
    int64_t num_tokens = gating_output.size(0);
    auto topk_weights = torch::empty({num_tokens, topk},
        torch::dtype(torch::kFloat32).device(gating_output.device()));
    auto topk_ids = torch::empty({num_tokens, topk},
        torch::dtype(torch::kInt32).device(gating_output.device()));
    auto token_expert_indices = torch::empty({num_tokens, topk},
        torch::dtype(torch::kInt32).device(gating_output.device()));

    auto gating_f32 = gating_output.to(torch::kFloat32);
    ixformer::infer::topk_softmax(
        topk_weights, topk_ids, token_expert_indices, gating_f32, renormalize);

    return std::make_tuple(topk_weights, topk_ids, token_expert_indices);
}

// --- moe_gen_idx ---
std::vector<torch::Tensor>
ix_moe_gen_idx(torch::Tensor expert_id, int64_t expert_num) {
    auto src_dst = expert_id.new_empty({expert_id.numel()});
    auto dst_src = torch::empty_like(src_dst);
    auto expert_sizes_gpu = expert_id.new_empty({expert_num});

    ixformer::infer::moe_compute_token_index_api(
        expert_id, src_dst, dst_src, expert_sizes_gpu,
        /*expert_mask=*/std::nullopt,
        /*expert_sizes_cpu=*/std::nullopt,
        /*expand_tokens_gpu=*/std::nullopt,
        /*start_expert_id=*/0,
        /*end_expert_id=*/expert_num,
        /*num_experts=*/expert_num);

    auto expert_sizes_cumsum = expert_sizes_gpu.cumsum(-1);
    return {src_dst, dst_src, expert_sizes_gpu, expert_sizes_cumsum};
}

// --- moe_expand_input ---
torch::Tensor ix_moe_expand_input(torch::Tensor input,
                                  torch::Tensor gather_index,
                                  torch::Tensor combine_idx,
                                  int64_t topk) {
    int64_t dst_tokens = input.size(0) * topk;
    auto output = input.new_empty({dst_tokens, input.size(1)});
    ixformer::infer::moe_expand_input(
        output, input, combine_idx, gather_index, dst_tokens, topk);
    return output;
}

// --- group_gemm ---
torch::Tensor ix_group_gemm(torch::Tensor inputs, torch::Tensor weights,
                            torch::Tensor tokens_per_experts,
                            int64_t output_n) {
    int64_t total_tokens = inputs.size(0);
    auto output = inputs.new_empty({total_tokens, output_n});
    int64_t gemm_output_n = tokens_per_experts.sum().item<int64_t>();
    ixformer::infer::moe_w16a16_group_gemm(
        output, inputs, weights, tokens_per_experts,
        /*dst_to_src=*/std::nullopt,
        /*bias=*/std::nullopt,
        /*format=*/"TN",
        /*persistent=*/0,
        gemm_output_n);
    return output;
}

// --- moe_combine_result ---
torch::Tensor ix_moe_combine_result(torch::Tensor input, torch::Tensor weight) {
    auto input_3d = input.view({-1, weight.size(1), input.size(1)});
    auto output = input.new_empty({input_3d.size(0), input_3d.size(2)});
    ixformer::infer::moe_output_reduce_sum(
        output, input_3d, weight,
        /*mask=*/std::nullopt,
        /*extra_residual=*/std::nullopt,
        /*scaling_factor=*/1.0);
    return output;
}

// --- fused_moe_forward (7-step pipeline) ---
torch::Tensor ix_fused_moe_forward(
    torch::Tensor hidden_states,
    torch::Tensor router_logits,
    torch::Tensor w13,
    torch::Tensor w2,
    int64_t topk,
    int64_t num_experts,
    bool renormalize) {

    // Step 1: topk_softmax
    auto [topk_weights, topk_ids, token_expert_indices] =
        ix_topk_softmax(router_logits, topk, renormalize);

    if (renormalize) {
        auto sum = topk_weights.sum(-1, /*keepdim=*/true);
        topk_weights = topk_weights / sum;
    }

    // Step 2: moe_gen_idx
    auto idx_results = ix_moe_gen_idx(topk_ids.view({-1}), num_experts);
    auto& src_dst = idx_results[0];
    auto& dst_src = idx_results[1];
    auto& expert_sizes_gpu = idx_results[2];

    // Step 3: moe_expand_input
    auto expanded = ix_moe_expand_input(hidden_states, src_dst, dst_src, topk);

    // Step 4: group_gemm (w13: gate_up projection)
    int64_t intermediate_2x = w13.size(1);
    auto gate_up = ix_group_gemm(expanded, w13,
                                 expert_sizes_gpu, intermediate_2x);

    // Step 5: silu_and_mul
    auto activated = ix_silu_and_mul(gate_up);

    // Step 6: group_gemm (w2: down projection)
    int64_t hidden_size = w2.size(1);
    auto down = ix_group_gemm(activated, w2,
                              expert_sizes_gpu, hidden_size);

    // Step 7: moe_combine_result
    auto output = ix_moe_combine_result(down, topk_weights);

    return output;
}


// ============================================================================
// Module registration
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Activation
    m.def("silu_and_mul", &ix_silu_and_mul,
          "Fused SiLU+mul via ixformer_torch_ext");

    // Norm
    m.def("rms_norm", &ix_rms_norm,
          "RMSNorm via ixformer_torch_ext");
    m.def("fused_add_rms_norm", &ix_fused_add_rms_norm,
          "Residual + RMSNorm via ixformer_torch_ext");

    // Linear
    m.def("linear", &ix_linear,
          "GEMM via ixformer_torch_ext");

    // RoPE
    m.def("rotary_embedding", &ix_rotary_embedding,
          "Rotary embedding via ixformer_torch_ext");

    // Cache
    m.def("reshape_and_cache", &ix_reshape_and_cache,
          "KV cache reshape+store via ixformer_torch_ext");

    // Attention (decode only)
    m.def("paged_attention", &ix_paged_attention,
          "Paged attention decode via ixformer_torch_ext");

    // MoE (individual steps — from moe_ops_impl.cu)
    m.def("topk_softmax", &ix_topk_softmax,
          "MoE topk+softmax routing");
    m.def("moe_gen_idx", &ix_moe_gen_idx,
          "MoE compute token index");
    m.def("moe_expand_input", &ix_moe_expand_input,
          "MoE expand input for expert dispatch");
    m.def("group_gemm", &ix_group_gemm,
          "MoE grouped GEMM via cuinferCustomGemm");
    m.def("moe_combine_result", &ix_moe_combine_result,
          "MoE output reduce sum");

    // MoE (fused 7-step pipeline)
    m.def("fused_moe_forward", &ix_fused_moe_forward,
          "Complete fused MoE forward (7-step pipeline)");
}