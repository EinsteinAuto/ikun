/*
 * corex_batched_gemm_bind.cpp — pybind11 wrapper for CUTLASS batched GEMM
 *
 * Kernel uses RowMajor + OpClassTensorOp + Cu10 (verified 2.462ms).
 * Source: ex_engine/xllm_kernels/cuda/moe_cutlass_batched.cu
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Implemented in corex_batched_gemm_kernel.cu
// RowMajor, FP16 data, FP32 accumulation, TCU, Cu10
cudaError_t cutlass_batched_hgemm(
    int m, int n, int k,
    __half const *A, int lda, long long int batch_stride_A,
    __half const *B, int ldb, long long int batch_stride_B,
    __half *C, int ldc, long long int batch_stride_C,
    int batch_count);

/*
 * batched_gemm_fp16: C[i] = A[i] @ B[i]
 *   A: (batch, M, K) row-major
 *   B: (batch, K, N) row-major
 *   C: (batch, M, N) row-major
 *
 * Both A and B must be contiguous fp16 CUDA tensors.
 */
torch::Tensor batched_gemm_fp16(
    torch::Tensor A,   // (batch, M, K)
    torch::Tensor B)   // (batch, K, N)
{
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(A.scalar_type() == torch::kFloat16 &&
                B.scalar_type() == torch::kFloat16,
                "inputs must be float16");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(),
                "inputs must be contiguous");
    TORCH_CHECK(A.dim() == 3 && B.dim() == 3,
                "inputs must be 3D (batch, rows, cols)");

    int batch = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);
    TORCH_CHECK(B.size(0) == batch, "batch size mismatch");
    TORCH_CHECK(B.size(1) == K, "K dimension mismatch");

    auto C = torch::zeros({batch, M, N}, A.options());

    // RowMajor: A is (M,K) with lda=K, B is (K,N) with ldb=N, C is (M,N) with ldc=N
    auto status = cutlass_batched_hgemm(
        M, N, K,
        reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
        K, (long long)M * K,    // lda, strideA
        reinterpret_cast<const __half*>(B.data_ptr<at::Half>()),
        N, (long long)K * N,    // ldb, strideB
        reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
        N, (long long)M * N,    // ldc, strideC
        batch);

    TORCH_CHECK(status == cudaSuccess,
                "CUTLASS batched HGEMM failed: ", cudaGetErrorString(status));
    return C;
}

/*
 * moe_decode_fused: Full MoE decode using TCU batched GEMM.
 *
 * hidden_states: (1, H)
 * w13_sel: (K, 2*I, H)  — already gathered expert weights
 * w2_sel:  (K, H, I)    — already gathered expert weights
 * topk_weights: (K,)
 *
 * Pipeline:
 *   1. gate_up = x @ w13^T  via batched GEMM  (K, 1, 2I)
 *   2. act = silu(gate) * up
 *   3. down = act @ w2^T    via batched GEMM  (K, 1, H)
 *   4. out = weighted sum
 */
torch::Tensor moe_decode_fused(
    torch::Tensor hidden_states,   // (1, H)
    torch::Tensor w13_sel,         // (K, 2*I, H)
    torch::Tensor w2_sel,          // (K, H, I)
    torch::Tensor topk_weights)    // (K,)
{
    int K_experts = w13_sel.size(0);
    int two_I = w13_sel.size(1);
    int H = w13_sel.size(2);
    int I = two_I / 2;

    // x: (1, H) → expand to (K, 1, H)
    auto x = hidden_states.expand({K_experts, 1, H}).contiguous();

    // w13^T: (K, 2I, H) → transpose last two dims → (K, H, 2I)
    auto w13_t = w13_sel.transpose(1, 2).contiguous();  // (K, H, 2I)

    // Step 1: gate_up = x @ w13^T → (K, 1, 2I)
    auto gate_up = batched_gemm_fp16(x, w13_t);
    gate_up = gate_up.squeeze(1);  // (K, 2I)

    // Step 2: silu activation
    auto chunks = gate_up.chunk(2, /*dim=*/1);
    auto act = torch::sigmoid(chunks[0]) * chunks[0] * chunks[1];  // silu(gate) * up
    act = act.unsqueeze(1);  // (K, 1, I)

    // w2^T: (K, H, I) → transpose → (K, I, H)
    auto w2_t = w2_sel.transpose(1, 2).contiguous();  // (K, I, H)

    // Step 3: down = act @ w2^T → (K, 1, H)
    auto down = batched_gemm_fp16(act, w2_t);
    down = down.squeeze(1);  // (K, H)

    // Step 4: weighted sum
    auto out = (down * topk_weights.unsqueeze(1)).sum(0, true);
    return out.to(hidden_states.dtype());
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUTLASS batched GEMM for MoE decode (BI-V100 TCU, Cu10 TensorOp)";
    m.def("batched_gemm_fp16", &batched_gemm_fp16,
          "Batched GEMM: (B,M,K) x (B,K,N) -> (B,M,N) in fp16 via TCU",
          py::arg("A"), py::arg("B"));
    m.def("moe_decode_fused", &moe_decode_fused,
          "Full MoE decode via TCU batched GEMM",
          py::arg("hidden_states"), py::arg("w13_sel"),
          py::arg("w2_sel"), py::arg("topk_weights"));
}
