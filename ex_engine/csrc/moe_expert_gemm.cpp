// moe_expert_gemm.cpp — MoE expert GEMM dispatch
//
// Replaces the Python for-loop over experts with a C++ loop calling
// ixformer_linear (via base image's _ixformer_torch.so).
//
// Why this works:
//   1. Eliminates Python interpreter overhead per expert (~0.5ms × 64 experts)
//   2. Eliminates PyTorch dispatcher overhead per F.linear call
//   3. Uses the same ixformer GEMM kernel that the base image uses
//   4. No new dependencies — links against the same .so as ix_full_bridge
//
// For decode (single token, top_k=8 experts):
//   Python: 8 × F.linear → 8 × Python dispatch → 8 × CUDA kernel
//   This:   1 × Python call → 8 × C++ ixformer_linear → 8 × CUDA kernel
//   Savings: ~4ms → ~0.5ms (eliminate 7 Python round-trips)
//
// For prefill (many tokens, up to 64 experts):
//   Python: for eid in 64: F.linear(tokens[eid], w[eid])
//   This:   1 × Python call → C++ loop: 64 × ixformer_linear
//   Savings: ~32ms → ~4ms
//
// Future: replace C++ loop with cublasGemmBatchedEx for true batched GEMM

#include <torch/extension.h>
#include <optional>
#include <vector>

// ============================================================================
// Forward declarations — from base image _ixformer_torch.cpython-310.so
// ============================================================================
namespace ixformer_torch_ext {

at::Tensor ixformer_linear(at::Tensor& input, at::Tensor& weight,
                           const c10::optional<at::Tensor>& bias,
                           const c10::optional<at::Tensor>& out);

at::Tensor ixformer_linear_ex(at::Tensor& input, at::Tensor& weight,
                              const c10::optional<at::Tensor>& bias);

void silu_and_mul_forward(at::Tensor& input, at::Tensor& output);

}  // namespace ixformer_torch_ext


// ============================================================================
// Decode path: single token, top_k experts
// ============================================================================
// Input:  hidden (1, H), w13 (E, 2*I, H), w2 (E, H, I), expert_ids (K,), weights (K,)
// Output: (1, H)
//
// Steps per expert:
//   1. gate_up = ixformer_linear(hidden, w13[eid])   → (1, 2*I)
//   2. act = silu_and_mul(gate_up)                    → (1, I)
//   3. expert_out = ixformer_linear(act, w2[eid])     → (1, H)
//   4. accumulate: out += weight[k] * expert_out

torch::Tensor moe_decode_experts(
    torch::Tensor hidden,       // (1, H)
    torch::Tensor w13,          // (num_experts, 2*inter, H)
    torch::Tensor w2,           // (num_experts, H, inter)
    torch::Tensor expert_ids,   // (top_k,)  int64
    torch::Tensor expert_weights  // (top_k,) fp16/fp32
) {
    int64_t top_k = expert_ids.size(0);
    int64_t H = hidden.size(-1);
    int64_t inter2 = w13.size(1);  // 2 * intermediate
    int64_t inter = inter2 / 2;

    auto out = torch::zeros({1, H}, hidden.options());
    c10::optional<at::Tensor> no_bias;

    for (int64_t k = 0; k < top_k; ++k) {
        int64_t eid = expert_ids[k].item<int64_t>();
        float w = expert_weights[k].item<float>();

        // w13[eid] shape: (2*I, H) — use as weight for linear
        auto w13_e = w13[eid];  // (2*I, H)
        auto w2_e = w2[eid];    // (H, I)

        // gate_up = hidden @ w13_e^T → (1, 2*I)
        auto gate_up = ixformer_torch_ext::ixformer_linear(
            hidden, w13_e, no_bias, c10::optional<at::Tensor>());

        // silu_and_mul: (1, 2*I) → (1, I)
        auto act = torch::empty({1, inter}, hidden.options());
        ixformer_torch_ext::silu_and_mul_forward(gate_up, act);

        // expert_out = act @ w2_e^T → (1, H)
        auto expert_out = ixformer_torch_ext::ixformer_linear(
            act, w2_e, no_bias, c10::optional<at::Tensor>());

        // accumulate
        out.add_(expert_out, w);
    }

    return out;
}


// ============================================================================
// Prefill path: multiple tokens, grouped by expert
// ============================================================================
// Input:  hidden (T, H), w13 (E, 2*I, H), w2 (E, H, I),
//         sorted_token_ids (T*K,), sorted_weights (T*K,), expert_counts list
// Output: (T, H)
//
// For each expert with count > 0:
//   tokens = hidden[sorted_token_ids[start:end]]
//   gate_up = ixformer_linear(tokens, w13[eid])
//   act = silu_and_mul(gate_up)
//   expert_out = ixformer_linear(act, w2[eid])
//   out[token_ids] += expert_out * weights

torch::Tensor moe_prefill_experts(
    torch::Tensor hidden,           // (T, H)
    torch::Tensor w13,              // (E, 2*I, H)
    torch::Tensor w2,               // (E, H, I)
    torch::Tensor sorted_token_ids, // (T*K,) int64
    torch::Tensor sorted_weights,   // (T*K,) fp16/fp32
    torch::Tensor expert_counts     // (E,) int64
) {
    int64_t T = hidden.size(0);
    int64_t H = hidden.size(-1);
    int64_t inter2 = w13.size(1);
    int64_t inter = inter2 / 2;
    int64_t E = expert_counts.size(0);

    auto out = torch::zeros({T, H}, hidden.options());
    c10::optional<at::Tensor> no_bias;

    int64_t start = 0;
    for (int64_t eid = 0; eid < E; ++eid) {
        int64_t count = expert_counts[eid].item<int64_t>();
        if (count == 0) continue;
        int64_t end = start + count;

        auto tok_ids = sorted_token_ids.slice(0, start, end);   // (count,)
        auto tokens = hidden.index_select(0, tok_ids);           // (count, H)
        auto weights = sorted_weights.slice(0, start, end);      // (count,)

        auto w13_e = w13[eid];  // (2*I, H)
        auto w2_e = w2[eid];    // (H, I)

        // FC1: gate_up = tokens @ w13_e^T → (count, 2*I)
        auto gate_up = ixformer_torch_ext::ixformer_linear(
            tokens, w13_e, no_bias, c10::optional<at::Tensor>());

        // SiLU and mul: (count, 2*I) → (count, I)
        auto act = torch::empty({count, inter}, hidden.options());
        ixformer_torch_ext::silu_and_mul_forward(gate_up, act);

        // FC2: expert_out = act @ w2_e^T → (count, H)
        auto expert_out = ixformer_torch_ext::ixformer_linear(
            act, w2_e, no_bias, c10::optional<at::Tensor>());

        // Weighted accumulate: out[tok_ids] += expert_out * weights
        auto weighted = expert_out * weights.unsqueeze(-1);
        out.index_add_(0, tok_ids, weighted.to(out.dtype()));

        start = end;
    }

    return out;
}


// ============================================================================
// Module registration
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("moe_decode_experts", &moe_decode_experts,
          "MoE decode: C++ loop over top_k experts using ixformer_linear",
          py::arg("hidden"), py::arg("w13"), py::arg("w2"),
          py::arg("expert_ids"), py::arg("expert_weights"));
    m.def("moe_prefill_experts", &moe_prefill_experts,
          "MoE prefill: C++ loop over experts using ixformer_linear",
          py::arg("hidden"), py::arg("w13"), py::arg("w2"),
          py::arg("sorted_token_ids"), py::arg("sorted_weights"),
          py::arg("expert_counts"));
}
