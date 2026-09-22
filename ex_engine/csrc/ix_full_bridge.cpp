// ix_full_bridge.cpp — Bridge to ixformer C++ functions available in base image
//
// Based on symbol probe of the actual BI-V100 base image:
//   _ixformer_torch.so has: silu_and_mul_forward, rms_norm_forward,
//     fused_add_rms_norm_forward, ixformer_linear, ixformer_linear_ex
//   libixformer.so has: ixinfer_flash_attn_unpad_fwd
//
// MoE functions (topk_softmax, group_gemm, etc.) are NOT in base image.
// They exist only in xllm's compiled library. MoE must use Python fallback.

#include <torch/extension.h>
#include <optional>
#include <tuple>
#include <vector>

// ============================================================================
// Forward declarations — ACTUAL symbols from base image .so files
// Namespace: ixformer_torch_ext (in _ixformer_torch.cpython-310.so)
// ============================================================================
namespace ixformer_torch_ext {

// silu_and_mul: _ZN18ixformer_torch_ext20silu_and_mul_forwardERN2at6TensorES2_
void silu_and_mul_forward(at::Tensor& input, at::Tensor& output);

// rms_norm: _ZN18ixformer_torch_ext16rms_norm_forwardERN2at6TensorES2_S2_d
void rms_norm_forward(at::Tensor& input, at::Tensor& weight, at::Tensor& output, double eps);

// fused_add_rms_norm: _ZN18ixformer_torch_ext26fused_add_rms_norm_forwardERN2at6TensorES2_S2_dd
void fused_add_rms_norm_forward(at::Tensor& input, at::Tensor& residual,
                                 at::Tensor& weight, double eps, double alpha);

// ixformer_linear: _ZN18ixformer_torch_ext15ixformer_linearERN2at6TensorES2_RKN3c108optionalIS1_EES7_
at::Tensor ixformer_linear(at::Tensor& input, at::Tensor& weight,
                           const c10::optional<at::Tensor>& bias,
                           const c10::optional<at::Tensor>& out);

// ixformer_linear_ex: _ZN18ixformer_torch_ext18ixformer_linear_exERN2at6TensorES2_RKN3c108optionalIS1_EE
at::Tensor ixformer_linear_ex(at::Tensor& input, at::Tensor& weight,
                              const c10::optional<at::Tensor>& bias);

}  // namespace ixformer_torch_ext


// ============================================================================
// Python wrappers
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
  ixformer_torch_ext::rms_norm_forward(input, weight, output, eps);
}

// --- fused_add_rms_norm ---
void ix_fused_add_rms_norm(torch::Tensor input, torch::Tensor residual,
                            torch::Tensor weight, double eps) {
  ixformer_torch_ext::fused_add_rms_norm_forward(input, residual, weight, eps, 1.0);
}

// --- linear ---
torch::Tensor ix_linear(torch::Tensor input, torch::Tensor weight,
                         const c10::optional<torch::Tensor>& bias) {
  // Use linear_ex for decode (m<=1), linear for prefill
  auto input_2d = input.view({-1, input.size(-1)});
  int64_t m = input_2d.size(0);
  if (m <= 1 && !bias.has_value()) {
    return ixformer_torch_ext::ixformer_linear_ex(input, weight, bias);
  }
  return ixformer_torch_ext::ixformer_linear(input, weight, bias,
                                              c10::optional<at::Tensor>());
}


// ============================================================================
// Module registration
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("silu_and_mul", &ix_silu_and_mul, "Fused SiLU+mul activation");
  m.def("rms_norm", &ix_rms_norm, "RMSNorm");
  m.def("fused_add_rms_norm", &ix_fused_add_rms_norm, "Fused residual + RMSNorm");
  m.def("linear", &ix_linear, "ixformer GEMM (linear/linear_ex)");
}
