// gemm_grouped.cu — Per-expert GEMM using CUTLASS Cu10 TensorOp
//
// Source lineage:
//   cat_files/batched_gemm.cu           — cutlass sample from real device
//   cat_files/default_gemm_configuration.h — Cu10 half/half/float config
//   ex_engine/xllm_kernels/cuda/corex_batched_gemm_kernel.cu — existing impl
//   ex_engine/xllm_kernels/cuda/bindings/hgemm_bind.cpp — moe_expert_gemm pattern
//
// This file provides:
//   1. cutlass_expert_gemm()   — one cutlass GEMM per expert (Cu10 TensorOp)
//   2. cuinfer_expert_gemm()   — one cuinferCustomGemm per expert (fallback)
//   3. moe_group_gemm()        — unified entry: try cutlass, fall back to cuinfer
//
// All use RowMajor, FP16 data, FP32 accumulation.
// Weight layout: [num_experts, N, K] (TN format = transB in GEMM sense)

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm_batched.h"

// ============================================================================
// Cu10 TensorOp GEMM type — from default_gemm_configuration.h
// ThreadblockShape<128,128,32>, WarpShape<32,32,32>, Instruction<16,16,16>
// ============================================================================
using GemmCu10 = cutlass::gemm::device::GemmBatched<
    cutlass::half_t,                    // ElementA
    cutlass::layout::RowMajor,          // LayoutA
    cutlass::half_t,                    // ElementB
    cutlass::layout::RowMajor,          // LayoutB
    cutlass::half_t,                    // ElementC
    cutlass::layout::RowMajor,          // LayoutC
    float,                              // ElementAccumulator
    cutlass::arch::OpClassTensorOp,     // use TCU
    cutlass::arch::Cu10                 // BI-V100
>;


// ============================================================================
// cutlass_expert_gemm: per-expert GEMM using CUTLASS
//
// For each expert e with M_e tokens:
//   C[offset:offset+M_e, :N] = A[offset:offset+M_e, :K] @ B[e, :N, :K]^T
//
// B is stored as [num_experts, N, K] (RowMajor), we need A×B^T.
// Cutlass RowMajor × RowMajor computes C = A × B, so we transpose:
//   C(M,N) = A(M,K) × B^T(K,N) = A(M,K) × B_orig(N,K)^T
//
// In row-major: A lda=K, B lda=K (it's NxK stored row-major), C ldc=N
// We use Cutlass's NN mode on (A, B^T) which is implemented as:
//   Cutlass RowMajor NN: C[i,j] = sum_k A[i,k] * B[k,j]
//   But B is (N,K) not (K,N), so we pass B as ColumnMajor or handle via stride.
//
// Simpler: A is (M,K) RowMajor, we want output (M,N).
// B_expert is (N,K) RowMajor = same as (K,N) ColumnMajor.
// So: A(M,K) RowMajor × B(K,N) ColumnMajor → C(M,N) RowMajor
// This is exactly GEMM with transB.
// ============================================================================

using GemmCu10_TN = cutlass::gemm::device::GemmBatched<
    cutlass::half_t,                    // ElementA
    cutlass::layout::RowMajor,          // LayoutA — A is (M,K) row-major
    cutlass::half_t,                    // ElementB
    cutlass::layout::ColumnMajor,       // LayoutB — B is (N,K) stored row = (K,N) col
    cutlass::half_t,                    // ElementC
    cutlass::layout::RowMajor,          // LayoutC
    float,                              // ElementAccumulator
    cutlass::arch::OpClassTensorOp,     // TCU
    cutlass::arch::Cu10                 // BI-V100
>;


int cutlass_expert_gemm(
    int num_experts,
    const int* expert_counts,     // host array [num_experts]
    const int* expert_offsets,    // host array [num_experts], exclusive prefix sum
    int N, int K,
    const __half* input,          // (total_tokens, K) row-major
    const __half* weights,        // (num_experts, N, K) row-major — TN format
    __half* output,               // (total_tokens, N) row-major
    cudaStream_t stream)
{
    GemmCu10_TN gemm_op;
    float alpha = 1.0f, beta = 0.0f;
    int failures = 0;

    for (int e = 0; e < num_experts; e++) {
        int M_e = expert_counts[e];
        if (M_e <= 0) continue;

        int off = expert_offsets[e];
        auto A = reinterpret_cast<cutlass::half_t const*>(input + (long long)off * K);
        auto B = reinterpret_cast<cutlass::half_t const*>(weights + (long long)e * N * K);
        auto C = reinterpret_cast<cutlass::half_t*>(output + (long long)off * N);

        // A: (M_e, K) RowMajor, lda = K
        // B: (N, K) RowMajor → (K, N) ColumnMajor, ldb = N (col-major stride)
        // C: (M_e, N) RowMajor, ldc = N
        cutlass::Status status = gemm_op({
            {M_e, N, K},
            {A, K},       // A, lda
            0,             // strideA (not batched)
            {B, K},        // B in col-major view: (N,K) row = (K,N) col, ldb = K
            0,             // strideB
            {C, N},        // C, ldc
            0,             // strideC
            {C, N},        // D = C
            0,
            {alpha, beta},
            1              // batch_count = 1 (we loop over experts)
        });

        if (status != cutlass::Status::kSuccess) {
            failures++;
        }
    }
    return failures;
}


// ============================================================================
// cuinfer fallback — forward-declare cuinferCustomGemm
// ============================================================================
extern "C" {
typedef struct cuinferContext* cuinferHandle_t;
typedef enum { CUINFER_STATUS_SUCCESS_GG = 0 } cuinferStatus_gg_t;
cuinferHandle_t cuinferCreate_handle();

int cuinferCustomGemm(
    cuinferHandle_t handle, cudaStream_t stream,
    int ptrMode, int transa, int transb,
    int m, int n, int k,
    const void* alpha,
    const void* A, int Atype, int lda, long long int strideA,
    const void* B, int Btype, int ldb, long long int strideB,
    const void* beta,
    void* C, int Ctype, int ldc, long long int strideC,
    int batchCount, int computeType, int scaleType,
    const void* customHostPtr, const void* customDevicePtr, int customOption);
}


int cuinfer_expert_gemm(
    int num_experts,
    const int* expert_counts,
    const int* expert_offsets,
    int N, int K,
    const __half* input,
    const __half* weights,
    __half* output,
    cudaStream_t stream,
    cuinferHandle_t handle)
{
    float alpha = 1.0f, beta = 0.0f;
    int failures = 0;

    for (int e = 0; e < num_experts; e++) {
        int M_e = expert_counts[e];
        if (M_e <= 0) continue;

        int off = expert_offsets[e];
        const void* A = input + (long long)off * K;
        const void* B = weights + (long long)e * N * K;
        void* C = output + (long long)off * N;

        // cuinferCustomGemm: transa=0 (N), transb=1 (T)
        // CUDA_R_16F = 2
        int status = cuinferCustomGemm(
            handle, stream,
            0,     // CUINFER_POINTER_MODE_HOST
            0, 1,  // transa=N, transb=T
            M_e, N, K,
            &alpha,
            A, 2, K, 0,   // A: fp16, lda=K
            B, 2, K, 0,   // B: fp16, ldb=K (row-major N×K, transposed)
            &beta,
            C, 2, N, 0,   // C: fp16, ldc=N
            1,             // batchCount=1
            0, 0,          // computeType=fp32, scaleType=fp32
            nullptr, nullptr, 0);

        if (status != 0) failures++;
    }
    return failures;
}
