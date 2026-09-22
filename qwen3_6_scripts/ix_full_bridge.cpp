// ix_full_bridge.cpp — Bridge to ixformer_torch_ext C++ functions
//
// Real namespace: ixformer_torch_ext (from nm -D _ixformer_torch.cpython-310.so)
// NOT ixformer::infer (that namespace doesn't exist in BI-V100 ixformer 3.2.3)
//
// Symbols confirmed on real machine:
//   ixformer_torch_ext::silu_and_mul_forward(Tensor&, Tensor&)
//   ixformer_torch_ext::rms_norm_forward(Tensor&, Tensor&, Tensor&, double)
//   ixformer_torch_ext::fused_add_rms_norm_forward(Tensor&, Tensor&, Tensor&, double, double)
//   ixformer_torch_ext::vllm_rotary_embedding_neox(Tensor&, Tensor&, Tensor&, long, Tensor&, long, bool)
//   ixformer_torch_ext::vllm_cache_ops_reshape_and_cache(Tensor&, Tensor&, Tensor&, Tensor&, Tensor&, long, long)

#include <torch/extension.h>

// Forward declarations — exact signatures from nm -D on real machine
namespace ixformer_torch_ext {

void silu_and_mul_forward(at::Tensor& input, at::Tensor& output);

void rms_norm_forward(at::Tensor& output, at::Tensor& input,
                      at::Tensor& weight, double eps);

void fused_add_rms_norm_forward(at::Tensor& input, at::Tensor& residual,
                                at::Tensor& weight, double eps, double dropout);

void vllm_rotary_embedding_neox(at::Tensor& positions, at::Tensor& query,
                                at::Tensor& key, long head_size,
                                at::Tensor& cos_sin_cache, long is_neox, bool interleaved);

void vllm_cache_ops_reshape_and_cache(at::Tensor& key, at::Tensor& value,
                                      at::Tensor& key_cache, at::Tensor& value_cache,
                                      at::Tensor& slot_mapping, long block_size,
                                      long x);

void rms_norm_quant(at::Tensor& output, at::Tensor& input,
                    at::Tensor& weight, double eps);

} // namespace ixformer_torch_ext

// ============================================================================
// Python-facing wrappers
// ============================================================================

void ix_silu_and_mul(torch::Tensor& input, torch::Tensor& output) {
    ixformer_torch_ext::silu_and_mul_forward(input, output);
}

void ix_rms_norm(torch::Tensor& output, torch::Tensor& input,
                 torch::Tensor& weight, double eps) {
    ixformer_torch_ext::rms_norm_forward(output, input, weight, eps);
}

void ix_fused_add_rms_norm(torch::Tensor& input, torch::Tensor& residual,
                           torch::Tensor& weight, double eps) {
    ixformer_torch_ext::fused_add_rms_norm_forward(input, residual, weight, eps, 0.0);
}

void ix_rotary_embedding(torch::Tensor& positions, torch::Tensor& query,
                         torch::Tensor& key, long head_size,
                         torch::Tensor& cos_sin_cache, bool is_neox) {
    ixformer_torch_ext::vllm_rotary_embedding_neox(
        positions, query, key, head_size, cos_sin_cache, is_neox ? 1 : 0, false);
}

void ix_reshape_and_cache(torch::Tensor& key, torch::Tensor& value,
                          torch::Tensor& key_cache, torch::Tensor& value_cache,
                          torch::Tensor& slot_mapping) {
    long block_size = key_cache.size(1);
    long x = key_cache.size(3);
    ixformer_torch_ext::vllm_cache_ops_reshape_and_cache(
        key, value, key_cache, value_cache, slot_mapping, block_size, x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("silu_and_mul", &ix_silu_and_mul,
          "ixformer_torch_ext::silu_and_mul_forward bridge");
    m.def("rms_norm", &ix_rms_norm,
          "ixformer_torch_ext::rms_norm_forward bridge");
    m.def("fused_add_rms_norm", &ix_fused_add_rms_norm,
          "ixformer_torch_ext::fused_add_rms_norm_forward bridge");
    m.def("rotary_embedding", &ix_rotary_embedding,
          "ixformer_torch_ext::vllm_rotary_embedding_neox bridge");
    m.def("reshape_and_cache", &ix_reshape_and_cache,
          "ixformer_torch_ext::vllm_cache_ops_reshape_and_cache bridge");
}
