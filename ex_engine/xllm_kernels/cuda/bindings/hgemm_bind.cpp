// hgemm_bind.cpp — pybind11 bindings for hgemm_blocktiling.cu
//
// Exports:
//   hgemm(A, B, M, N, K) → C
//   moe_expert_gemm(input, weights, expert_counts) → output

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>

// Forward declarations from hgemm_blocktiling.cu
void launch_hgemm_blocktiling(
    int M, int N, int K,
    const __half* alpha, const __half* A, int lda,
    const __half* B, int ldb,
    const __half* beta, __half* C, int ldc,
    cudaStream_t stream);

void launch_moe_expert_hgemm(
    int num_experts,
    const int* expert_counts,
    const int* expert_offsets,
    int N, int K,
    const __half* input,
    const __half* weights,
    __half* output,
    cudaStream_t stream);


// ============================================================================
// Python-facing wrappers
// ============================================================================

// Simple GEMM: C = A @ B
// A: (M, K) fp16, B: (K, N) fp16 → C: (M, N) fp16
torch::Tensor hgemm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.scalar_type() == torch::kHalf, "A must be fp16");
    TORCH_CHECK(B.scalar_type() == torch::kHalf, "B must be fp16");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    __half alpha = __float2half(1.0f);
    __half beta  = __float2half(0.0f);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    launch_hgemm_blocktiling(
        M, N, K, &alpha,
        reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
        A.size(1),
        reinterpret_cast<const __half*>(B.data_ptr<at::Half>()),
        B.size(1),
        &beta,
        reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
        C.size(1),
        stream);

    return C;
}


// MoE expert GEMM: for each expert e, compute
//   output[offset_e : offset_e + count_e] = input[offset_e : offset_e + count_e] @ weights[e].T
//
// input:          (total_tokens, K) fp16
// weights:        (num_experts, N, K) fp16 — weight layout matches vllm w13/w2 convention
// expert_counts:  (num_experts,) int32 — number of tokens per expert
//
// Returns: output (total_tokens, N) fp16
torch::Tensor moe_expert_gemm(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor expert_counts
) {
    TORCH_CHECK(input.is_cuda() && weights.is_cuda(), "Inputs must be CUDA");
    TORCH_CHECK(input.scalar_type() == torch::kHalf, "input must be fp16");
    TORCH_CHECK(weights.scalar_type() == torch::kHalf, "weights must be fp16");
    TORCH_CHECK(expert_counts.scalar_type() == torch::kInt32 ||
                expert_counts.scalar_type() == torch::kInt64,
                "expert_counts must be int32 or int64");

    int total_tokens = input.size(0);
    int K = input.size(1);
    int num_experts = weights.size(0);
    int N = weights.size(1);  // output dim

    TORCH_CHECK(weights.size(2) == K, "weights K dim must match input");

    auto output = torch::zeros({total_tokens, N}, input.options());

    // Convert expert_counts to host int array
    auto counts_cpu = expert_counts.to(torch::kCPU).to(torch::kInt32).contiguous();
    std::vector<int> counts(num_experts);
    std::vector<int> offsets(num_experts);
    int cumsum = 0;
    for (int i = 0; i < num_experts; i++) {
        counts[i] = counts_cpu.data_ptr<int32_t>()[i];
        offsets[i] = cumsum;
        cumsum += counts[i];
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    launch_moe_expert_hgemm(
        num_experts,
        counts.data(),
        offsets.data(),
        N, K,
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(weights.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        stream);

    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hgemm", &hgemm,
          "FP16 GEMM: C = A @ B (adapted from siboehm kernel 6 for BI-V100)",
          py::arg("A"), py::arg("B"));
    m.def("moe_expert_gemm", &moe_expert_gemm,
          "MoE expert GEMM: per-expert matmul with variable token counts",
          py::arg("input"), py::arg("weights"), py::arg("expert_counts"));
}
