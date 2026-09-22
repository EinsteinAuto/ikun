// moe_tcu_dispatch.cpp — MoE expert GEMM via torch::mm (walks Gemm_tcu_bi_kernel)
//
// Replaces Python for-loop over experts with C++ loop.
// torch::mm on corex launches Gemm_tcu_bi_kernel::gemm_h_h_tcu_25 (TCU hardware).
// Probe confirmed: Python loop overhead = 0.892 ms/expert = 7.1 ms for 8 experts.
// This C++ dispatch eliminates that overhead.
//
// No custom GEMM kernel. No ixformer API dependency. Just torch::mm in C++.

#include <torch/extension.h>
#include <vector>

// ============================================================================
// Decode path: single token, top_k experts
// ============================================================================
// hidden:         (1, K)
// gate_up_weights: (num_experts, 2*intermediate, K) — pre-loaded expert weights
// down_weights:    (num_experts, K, intermediate)
// expert_ids:     (top_k,) int64 — selected expert indices
// expert_weights: (top_k,) float — gating weights
//
// For each expert:
//   gate_up = hidden @ gate_up_weights[eid].t()   → (1, 2*I)
//   gate = silu(gate_up[:, :I])
//   up   = gate_up[:, I:]
//   act  = gate * up                               → (1, I)
//   out  = act @ down_weights[eid].t()             → (1, K)
//   result += weight * out

torch::Tensor moe_decode(
    torch::Tensor hidden,          // (1, K)
    torch::Tensor gate_up_weights, // (E, 2*I, K)
    torch::Tensor down_weights,    // (E, K, I)
    torch::Tensor expert_ids,      // (top_k,) int64
    torch::Tensor expert_weights   // (top_k,) float/half
) {
    auto top_k = expert_ids.size(0);
    auto K = hidden.size(1);
    auto inter2 = gate_up_weights.size(1);
    auto inter = inter2 / 2;

    auto result = torch::zeros_like(hidden);  // (1, K)

    for (int64_t k = 0; k < top_k; ++k) {
        auto eid = expert_ids[k].item<int64_t>();
        auto w = expert_weights[k].item<float>();

        // FC1: gate_up = hidden @ w13[eid]^T → (1, 2*I)
        auto gate_up = torch::mm(hidden, gate_up_weights[eid].t());

        // SiLU and mul
        auto gate_slice = gate_up.slice(1, 0, inter); auto gate = gate_slice * torch::sigmoid(gate_slice);
        auto up = gate_up.slice(1, inter, inter2);
        auto act = gate * up;  // (1, I)

        // FC2: expert_out = act @ w2[eid]^T → (1, K)
        auto expert_out = torch::mm(act, down_weights[eid].t());

        // Weighted accumulate
        result.add_(expert_out, w);
    }

    return result;
}


// ============================================================================
// Prefill path: multiple tokens, grouped by expert
// ============================================================================
// hidden:          (T, K)
// gate_up_weights: (E, 2*I, K)
// down_weights:    (E, K, I)
// topk_ids:        (T, top_k) int64 — expert indices per token
// topk_weights:    (T, top_k) float — gating weights per token
//
// Strategy: group tokens by expert, batch the GEMM per expert.

torch::Tensor moe_prefill(
    torch::Tensor hidden,          // (T, K)
    torch::Tensor gate_up_weights, // (E, 2*I, K)
    torch::Tensor down_weights,    // (E, K, I)
    torch::Tensor topk_ids,        // (T, top_k) int64
    torch::Tensor topk_weights     // (T, top_k) float/half
) {
    auto T = hidden.size(0);
    auto K = hidden.size(1);
    auto num_experts = gate_up_weights.size(0);
    auto inter2 = gate_up_weights.size(1);
    auto inter = inter2 / 2;
    auto top_k = topk_ids.size(1);

    auto result = torch::zeros({T, K}, hidden.options());

    // Flatten topk_ids to find tokens per expert
    auto flat_ids = topk_ids.reshape(-1);         // (T*top_k,)
    auto flat_weights = topk_weights.reshape(-1);  // (T*top_k,)

    // Token index for each (token, k) pair
    auto token_idx = torch::arange(T, topk_ids.options())
                         .unsqueeze(1).expand({T, top_k}).reshape(-1);  // (T*top_k,)

    for (int64_t eid = 0; eid < num_experts; ++eid) {
        // Find which entries in flat_ids match this expert
        auto mask = flat_ids.eq(eid);
        auto count = mask.sum().item<int64_t>();
        if (count == 0) continue;

        // Gather token indices and weights for this expert
        auto indices = mask.nonzero().squeeze(1);        // (count,)
        auto tok_indices = token_idx.index_select(0, indices);  // (count,)
        auto weights = flat_weights.index_select(0, indices);   // (count,)

        // Gather hidden states
        auto tokens = hidden.index_select(0, tok_indices);  // (count, K)

        // FC1: gate_up = tokens @ w13[eid]^T → (count, 2*I)
        auto gate_up = torch::mm(tokens, gate_up_weights[eid].t());

        // SiLU and mul
        auto gate_slice = gate_up.slice(1, 0, inter); auto gate = gate_slice * torch::sigmoid(gate_slice);
        auto up = gate_up.slice(1, inter, inter2);
        auto act = gate * up;  // (count, I)

        // FC2: expert_out = act @ w2[eid]^T → (count, K)
        auto expert_out = torch::mm(act, down_weights[eid].t());

        // Weighted scatter-add
        auto weighted = expert_out * weights.unsqueeze(1);
        result.index_add_(0, tok_indices, weighted.to(result.dtype()));
    }

    return result;
}


// ============================================================================
// Simple expert GEMM only (no activation, for benchmarking)
// ============================================================================
// input:         (total_tokens, K)
// weights:       (num_experts, N, K)
// expert_counts: (num_experts,) int64
// Returns:       (total_tokens, N)

torch::Tensor moe_expert_gemm_tcu(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor expert_counts
) {
    auto total_tokens = input.size(0);
    auto K = input.size(1);
    auto num_experts = weights.size(0);
    auto N = weights.size(1);

    auto output = torch::zeros({total_tokens, N}, input.options());

    int64_t offset = 0;
    for (int64_t e = 0; e < num_experts; ++e) {
        auto count = expert_counts[e].item<int64_t>();
        if (count == 0) continue;

        auto tokens = input.slice(0, offset, offset + count);  // (count, K)
        auto w = weights[e];  // (N, K)

        // torch::mm → Gemm_tcu_bi_kernel on BI-V100
        auto out_e = torch::mm(tokens, w.t());  // (count, N)
        output.slice(0, offset, offset + count).copy_(out_e);

        offset += count;
    }

    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("moe_decode", &moe_decode,
          "MoE decode: C++ loop over experts via torch::mm (TCU kernel)",
          py::arg("hidden"), py::arg("gate_up_weights"),
          py::arg("down_weights"), py::arg("expert_ids"),
          py::arg("expert_weights"));

    m.def("moe_prefill", &moe_prefill,
          "MoE prefill: group-by-expert via torch::mm (TCU kernel)",
          py::arg("hidden"), py::arg("gate_up_weights"),
          py::arg("down_weights"), py::arg("topk_ids"),
          py::arg("topk_weights"));

    m.def("moe_expert_gemm_tcu", &moe_expert_gemm_tcu,
          "MoE expert GEMM only via torch::mm (TCU kernel, for benchmarking)",
          py::arg("input"), py::arg("weights"), py::arg("expert_counts"));
}
