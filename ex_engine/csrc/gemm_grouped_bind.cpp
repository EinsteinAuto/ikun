// gemm_grouped_bind.cpp — Python bindings for grouped GEMM
//
// Source lineage:
//   ex_engine/xllm_kernels/cuda/bindings/hgemm_bind.cpp — moe_expert_gemm pattern
//   ex_engine/xllm_kernels/cuda/bindings/corex_batched_gemm_bind.cpp — batched pattern
//
// Exports:
//   moe_group_gemm(input, weights, expert_counts) → output
//   moe_group_gemm_cutlass(input, weights, expert_counts) → output
//   moe_decode_cutlass(hidden, w13, w2, topk_weights) → output

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>

// From gemm_grouped.cu
int cutlass_expert_gemm(
    int num_experts,
    const int* expert_counts, const int* expert_offsets,
    int N, int K,
    const __half* input, const __half* weights, __half* output,
    cudaStream_t stream);


// ============================================================================
// moe_group_gemm: per-expert GEMM using CUTLASS Cu10 TensorOp
//
// input:         (total_tokens, K) fp16
// weights:       (num_experts, N, K) fp16, TN layout
// expert_counts: (num_experts,) int32
// Returns:       (total_tokens, N) fp16
// ============================================================================
torch::Tensor moe_group_gemm(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor expert_counts)
{
    TORCH_CHECK(input.is_cuda() && weights.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(input.scalar_type() == torch::kHalf, "input must be fp16");
    TORCH_CHECK(weights.scalar_type() == torch::kHalf, "weights must be fp16");

    int total_tokens = input.size(0);
    int K = input.size(1);
    int num_experts = weights.size(0);
    int N = weights.size(1);
    TORCH_CHECK(weights.size(2) == K, "weights K dim must match input K");

    auto output = torch::zeros({total_tokens, N}, input.options());

    // Build host arrays
    auto counts_cpu = expert_counts.to(torch::kCPU).to(torch::kInt32).contiguous();
    int32_t* c = counts_cpu.data_ptr<int32_t>();
    std::vector<int> counts(num_experts), offsets(num_experts);
    int cumsum = 0;
    for (int i = 0; i < num_experts; i++) {
        counts[i] = c[i];
        offsets[i] = cumsum;
        cumsum += c[i];
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    int fails = cutlass_expert_gemm(
        num_experts, counts.data(), offsets.data(),
        N, K,
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(weights.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        stream);

    if (fails > 0) {
        // Fallback to PyTorch F.linear per expert
        auto input_a = input.to(torch::kFloat32);
        auto output_f = torch::zeros({total_tokens, N},
            input.options().dtype(torch::kFloat32));
        for (int e = 0; e < num_experts; e++) {
            if (counts[e] <= 0) continue;
            int off = offsets[e];
            auto x = input_a.narrow(0, off, counts[e]);
            auto w = weights[e].to(torch::kFloat32);  // (N, K)
            output_f.narrow(0, off, counts[e]) = torch::mm(x, w.t());
        }
        output = output_f.to(torch::kHalf);
    }

    return output;
}


// ============================================================================
// moe_decode_cutlass: fused MoE decode for single-token (batch=1)
//
// Uses CUTLASS batched GEMM for the topk experts simultaneously.
//
// hidden:       (1, H) fp16
// w13_sel:      (topk, 2*I, H) fp16 — already-gathered expert weights
// w2_sel:       (topk, H, I) fp16
// topk_weights: (topk,) float32
// Returns:      (1, H) fp16
// ============================================================================

// From corex_batched_gemm_kernel.cu
cudaError_t cutlass_batched_hgemm(
    int m, int n, int k,
    __half const *A, int lda, long long int batch_stride_A,
    __half const *B, int ldb, long long int batch_stride_B,
    __half *C, int ldc, long long int batch_stride_C,
    int batch_count);


torch::Tensor moe_decode_cutlass(
    torch::Tensor hidden,         // (1, H)
    torch::Tensor w13_sel,        // (topk, 2*I, H)
    torch::Tensor w2_sel,         // (topk, H, I)
    torch::Tensor topk_weights)   // (topk,)
{
    int topk = w13_sel.size(0);
    int two_I = w13_sel.size(1);
    int H = w13_sel.size(2);
    int I = two_I / 2;

    // x: (1,H) → expand to (topk, 1, H)
    auto x = hidden.expand({topk, 1, H}).contiguous();

    // w13^T: (topk, 2I, H) → transpose → (topk, H, 2I)
    auto w13_t = w13_sel.transpose(1, 2).contiguous();

    // Step 1: gate_up = x @ w13^T → (topk, 1, 2I)
    auto gate_up_3d = torch::empty({topk, 1, two_I}, x.options());
    auto status1 = cutlass_batched_hgemm(
        1, two_I, H,
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        H, H,
        reinterpret_cast<const __half*>(w13_t.data_ptr<at::Half>()),
        two_I, H * two_I,
        reinterpret_cast<__half*>(gate_up_3d.data_ptr<at::Half>()),
        two_I, two_I,
        topk);
    TORCH_CHECK(status1 == cudaSuccess, "batched GEMM 1 failed");

    auto gate_up = gate_up_3d.squeeze(1);  // (topk, 2I)

    // Step 2: SiLU activation
    auto chunks = gate_up.chunk(2, 1);
    auto act = torch::silu(chunks[0]) * chunks[1];  // (topk, I)
    act = act.unsqueeze(1).contiguous();  // (topk, 1, I)

    // w2^T: (topk, H, I) → transpose → (topk, I, H)
    auto w2_t = w2_sel.transpose(1, 2).contiguous();

    // Step 3: down = act @ w2^T → (topk, 1, H)
    auto down_3d = torch::empty({topk, 1, H}, x.options());
    auto status2 = cutlass_batched_hgemm(
        1, H, I,
        reinterpret_cast<const __half*>(act.data_ptr<at::Half>()),
        I, I,
        reinterpret_cast<const __half*>(w2_t.data_ptr<at::Half>()),
        H, I * H,
        reinterpret_cast<__half*>(down_3d.data_ptr<at::Half>()),
        H, H,
        topk);
    TORCH_CHECK(status2 == cudaSuccess, "batched GEMM 2 failed");

    auto down = down_3d.squeeze(1);  // (topk, H)

    // Step 4: weighted sum
    auto out = (down * topk_weights.unsqueeze(1).to(down.dtype())).sum(0, true);
    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("moe_group_gemm", &moe_group_gemm,
          "Per-expert GEMM via CUTLASS Cu10 TensorOp",
          py::arg("input"), py::arg("weights"), py::arg("expert_counts"));
    m.def("moe_decode_cutlass", &moe_decode_cutlass,
          "Fused MoE decode via CUTLASS batched GEMM",
          py::arg("hidden"), py::arg("w13_sel"),
          py::arg("w2_sel"), py::arg("topk_weights"));
}
