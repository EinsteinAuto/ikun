// ix_attn_bridge.cpp — Bridge to ixformer::infer attention + linear functions
//
// Exposes functions from ixformer.h that are NOT available via ixformer.functions:
//   1. ixinfer_flash_attn_unpad_with_block_tables — fused prefill attention
//   2. xllm_paged_attention — fused paged decode attention
//   3. ixformer_linear — fused linear (matmul + optional activation)
//   4. ixformer_linear_ex — simple fused linear
//   5. residual_rms_norm — fused residual + RMS norm (NOT in ixformer_torch_ext)
//
// Source: xllm/xllm/core/kernels/ilu/ixformer.h
// Usage: xllm/xllm/core/kernels/ilu/attention.cpp
//        xllm/xllm/core/layers/ilu/attention.cpp

#include <torch/extension.h>
#include <optional>

namespace ixformer {
namespace infer {

// Prefill: flash attention with block tables (variable-length batched)
torch::Tensor ixinfer_flash_attn_unpad_with_block_tables(
    torch::Tensor& query,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& out,
    torch::Tensor& block_tables,
    torch::Tensor& cu_seq_q,
    torch::Tensor& cu_seq_k,
    int64_t max_seq_q,
    int64_t max_seq_k,
    bool is_causal,
    int64_t window_left,
    int64_t window_right,
    double scale,
    double softcap,
    bool sqrt_alibi,
    const std::optional<torch::Tensor>& alibi_slopes,
    const std::optional<torch::Tensor>& sinks,
    std::optional<torch::Tensor>& lse);

// Decode: paged attention (single-step cached KV)
torch::Tensor xllm_paged_attention(
    torch::Tensor& out,
    torch::Tensor& query,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    int64_t num_kv_heads,
    double scale,
    torch::Tensor& block_tables,
    torch::Tensor& context_lens,
    int64_t block_size,
    int64_t max_context_len,
    const std::optional<torch::Tensor>& alibi_slopes,
    bool causal,
    int32_t window_left,
    int32_t window_right,
    double softcap,
    bool enable_cuda_graph,
    bool use_sqrt_alibi,
    const std::optional<torch::Tensor>& sinks);

// Fused linear: matmul + optional activation
torch::Tensor ixformer_linear(
    torch::Tensor& input,
    torch::Tensor& weight,
    int64_t act_type,
    const std::optional<torch::Tensor>& bias,
    const std::optional<torch::Tensor>& out,
    const std::optional<bool> persistent);

// Simple linear
torch::Tensor ixformer_linear_ex(
    torch::Tensor& input,
    torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::Tensor>& out);

// Fused residual + RMS norm (not in ixformer_torch_ext, only in ixformer::infer)
void residual_rms_norm(
    torch::Tensor& input,
    torch::Tensor& residual,
    torch::Tensor& weight,
    torch::Tensor& output,
    torch::Tensor& residual_output,
    const std::optional<torch::Tensor>& fused_bias,
    double alpha,
    double eps,
    bool is_post);

}  // namespace infer
}  // namespace ixformer


// ============================================================================
// Python-facing wrappers
// Port from: xllm/xllm/core/kernels/ilu/attention.cpp
// ============================================================================

// Prefill attention via flash_attn_unpad_with_block_tables
torch::Tensor ix_prefill_attention(
    torch::Tensor query,           // (total_q_tokens, num_heads, head_dim)
    torch::Tensor key_cache,       // (num_blocks, num_heads, block_size, head_dim)
    torch::Tensor value_cache,     // (num_blocks, num_heads, block_size, head_dim)
    torch::Tensor output,          // (total_q_tokens, num_heads, head_dim)
    torch::Tensor block_tables,    // (batch, max_blocks)
    torch::Tensor cu_seq_q,        // (batch+1,)
    torch::Tensor cu_seq_k,        // (batch+1,)
    int64_t max_query_len,
    int64_t max_seq_len,
    double scale,
    bool is_causal,
    int64_t window_left,
    int64_t window_right) {

    std::optional<torch::Tensor> lse;

    return ixformer::infer::ixinfer_flash_attn_unpad_with_block_tables(
        query, key_cache, value_cache, output, block_tables,
        cu_seq_q, cu_seq_k,
        max_query_len, max_seq_len,
        is_causal,
        window_left, window_right,
        scale,
        /*softcap=*/0.0,
        /*sqrt_alibi=*/false,
        /*alibi_slopes=*/std::nullopt,
        /*sinks=*/std::nullopt,
        lse);
}

// Decode attention via xllm_paged_attention
torch::Tensor ix_decode_attention(
    torch::Tensor output,          // (num_seqs, num_heads, head_dim)
    torch::Tensor query,           // (num_seqs, num_heads, head_dim)
    torch::Tensor key_cache,       // (num_blocks, num_kv_heads, block_size, head_dim)
    torch::Tensor value_cache,     // (num_blocks, num_kv_heads, block_size, head_dim)
    int64_t num_kv_heads,
    double scale,
    torch::Tensor block_tables,    // (num_seqs, max_blocks)
    torch::Tensor seq_lens,        // (num_seqs,)
    int64_t block_size,
    int64_t max_context_len) {

    return ixformer::infer::xllm_paged_attention(
        output, query, key_cache, value_cache,
        num_kv_heads, scale,
        block_tables, seq_lens,
        block_size, max_context_len,
        /*alibi_slopes=*/std::nullopt,
        /*causal=*/true,
        /*window_left=*/-1,
        /*window_right=*/-1,
        /*softcap=*/0.0,
        /*enable_cuda_graph=*/false,
        /*use_sqrt_alibi=*/false,
        /*sinks=*/std::nullopt);
}

// Fused linear (matmul + optional activation)
// act_type: 0=none, 1=silu, 2=gelu, 3=gelu_tanh
torch::Tensor ix_linear(
    torch::Tensor input,
    torch::Tensor weight,
    int64_t act_type) {
    return ixformer::infer::ixformer_linear(
        input, weight, act_type,
        /*bias=*/std::nullopt,
        /*out=*/std::nullopt,
        /*persistent=*/std::nullopt);
}

// Fused residual + RMS norm
// Port from: xllm/xllm/core/kernels/ilu/norm.cpp residual_layer_norm()
std::tuple<torch::Tensor, torch::Tensor> ix_residual_rms_norm(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight,
    double eps) {
    auto output = torch::zeros_like(input);
    auto residual_output = torch::zeros_like(input);

    ixformer::infer::residual_rms_norm(
        input, residual, weight, output, residual_output,
        /*fused_bias=*/std::nullopt,
        /*alpha=*/1.0,
        eps,
        /*is_post=*/false);

    return std::make_tuple(output, residual_output);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prefill_attention", &ix_prefill_attention,
          "Fused prefill attention via ixformer flash_attn_unpad_with_block_tables",
          py::arg("query"), py::arg("key_cache"), py::arg("value_cache"),
          py::arg("output"), py::arg("block_tables"),
          py::arg("cu_seq_q"), py::arg("cu_seq_k"),
          py::arg("max_query_len"), py::arg("max_seq_len"),
          py::arg("scale"),
          py::arg("is_causal") = true,
          py::arg("window_left") = -1,
          py::arg("window_right") = -1);

    m.def("decode_attention", &ix_decode_attention,
          "Paged decode attention via ixformer xllm_paged_attention",
          py::arg("output"), py::arg("query"),
          py::arg("key_cache"), py::arg("value_cache"),
          py::arg("num_kv_heads"), py::arg("scale"),
          py::arg("block_tables"), py::arg("seq_lens"),
          py::arg("block_size"), py::arg("max_context_len"));

    m.def("linear", &ix_linear,
          "Fused linear via ixformer (matmul + optional activation)",
          py::arg("input"), py::arg("weight"), py::arg("act_type") = 0);

    m.def("residual_rms_norm", &ix_residual_rms_norm,
          "Fused residual + RMS norm via ixformer",
          py::arg("input"), py::arg("residual"),
          py::arg("weight"), py::arg("eps") = 1e-6);
}
