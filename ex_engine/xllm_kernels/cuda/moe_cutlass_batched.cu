// moe_cutlass_batched.cu — FP16 Cu10 TensorOp batched GEMM for MoE on BI-V100
//
// Adapted from corex-samples cutlass/examples/05_batched_gemm/batched_gemm.cu
// Changes from original:
//   1. float → cutlass::half_t (FP16 data)
//   2. arch::OpClassSimt → arch::OpClassTensorOp (use TCU)
//   3. arch::Sm61 → arch::Cu10 (BI-V100 arch)
//   4. ElementAccumulator = float (FP32 accumulation)
//   5. Row-major layout (PyTorch convention) instead of column-major
//
// Default Cu10 FP16 TensorOp config from default_gemm_configuration.h:
//   ThreadblockShape = GemmShape<128, 128, 32>
//   WarpShape = GemmShape<32, 32, 32>
//   InstructionShape = GemmShape<16, 16, 16>
//   kStages = 2
//
// This uses __ivcorex_matrix_mad_f32x4_f16x4 under the hood (via mma_cu10.h).

#include <iostream>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm_batched.h"

// FP16 batched GEMM using Cu10 TensorOp
// C[i] = alpha * A[i] @ B[i] + beta * C[i]
// All matrices row-major, FP16 in/out, FP32 accumulation.
cudaError_t cutlass_batched_hgemm_tensorop(
    int m, int n, int k,
    float alpha,
    cutlass::half_t const *A, int lda, long long int batch_stride_A,
    cutlass::half_t const *B, int ldb, long long int batch_stride_B,
    cutlass::half_t *C, int ldc, long long int batch_stride_C,
    float beta,
    int batch_count)
{
    using Gemm = cutlass::gemm::device::GemmBatched<
        cutlass::half_t,                    // ElementA
        cutlass::layout::RowMajor,          // LayoutA
        cutlass::half_t,                    // ElementB
        cutlass::layout::RowMajor,          // LayoutB
        cutlass::half_t,                    // ElementC
        cutlass::layout::RowMajor,          // LayoutC
        float,                              // ElementAccumulator
        cutlass::arch::OpClassTensorOp,     // OperatorClass — use TCU
        cutlass::arch::Cu10                 // ArchTag — BI-V100
        // Remaining params use defaults from DefaultGemmConfiguration:
        //   ThreadblockShape = <128, 128, 32>
        //   WarpShape = <32, 32, 32>
        //   InstructionShape = <16, 16, 16>
        //   Stages = 2
    >;

    Gemm gemm_op;

    cutlass::Status status = gemm_op({
        {m, n, k},
        {A, lda},
        batch_stride_A,
        {B, ldb},
        batch_stride_B,
        {C, ldc},
        batch_stride_C,
        {C, ldc},
        batch_stride_C,
        {alpha, beta},
        batch_count
    });

    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}

// ============================================================================
// Standalone test
// ============================================================================
#ifdef BUILD_STANDALONE_TEST

#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

int main() {
    // Test: 8 batches of (1, 256) @ (256, 128) — simulates decode MoE
    int m = 1, n = 128, k = 256;
    int batch_count = 8;
    float alpha = 1.0f, beta = 0.0f;

    int lda = k;    // row-major: (m, k), stride = k
    int ldb = n;    // row-major: (k, n), stride = n
    int ldc = n;    // row-major: (m, n), stride = n

    long long int stride_A = (long long)m * k;
    long long int stride_B = (long long)k * n;
    long long int stride_C = (long long)m * n;

    size_t size_A = batch_count * stride_A * sizeof(cutlass::half_t);
    size_t size_B = batch_count * stride_B * sizeof(cutlass::half_t);
    size_t size_C = batch_count * stride_C * sizeof(cutlass::half_t);

    // Allocate host
    std::vector<cutlass::half_t> h_A(batch_count * stride_A);
    std::vector<cutlass::half_t> h_B(batch_count * stride_B);
    std::vector<cutlass::half_t> h_C(batch_count * stride_C, cutlass::half_t(0.0f));

    // Fill with small values
    for (auto &v : h_A) v = cutlass::half_t(0.01f * (rand() % 100 - 50));
    for (auto &v : h_B) v = cutlass::half_t(0.01f * (rand() % 100 - 50));

    // Allocate device
    cutlass::half_t *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C.data(), size_C, cudaMemcpyHostToDevice);

    // Run CUTLASS batched GEMM
    cudaError_t result = cutlass_batched_hgemm_tensorop(
        m, n, k, alpha,
        d_A, lda, stride_A,
        d_B, ldb, stride_B,
        d_C, ldc, stride_C,
        beta, batch_count);

    cudaDeviceSynchronize();

    if (result != cudaSuccess) {
        printf("CUTLASS batched GEMM FAILED: %s\n", cudaGetErrorString(result));
        cudaError_t last = cudaGetLastError();
        if (last != cudaSuccess)
            printf("Last CUDA error: %s\n", cudaGetErrorString(last));
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        return -1;
    }

    // Copy back
    cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost);

    // Verify against CPU reference
    bool pass = true;
    for (int b = 0; b < batch_count; b++) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float ref = 0.0f;
                for (int p = 0; p < k; p++) {
                    float a_val = float(h_A[b * stride_A + i * k + p]);
                    float b_val = float(h_B[b * stride_B + p * n + j]);
                    ref += a_val * b_val;
                }
                float got = float(h_C[b * stride_C + i * n + j]);
                if (fabs(ref - got) > 1.0f) {
                    printf("MISMATCH batch=%d [%d,%d]: ref=%.4f got=%.4f\n",
                           b, i, j, ref, got);
                    pass = false;
                }
            }
        }
    }

    if (pass) {
        printf("CUTLASS Cu10 TensorOp batched HGEMM: PASSED (%d batches of %dx%d@%dx%d)\n",
               batch_count, m, k, k, n);
    }

    // Benchmark
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    // Warmup
    for (int i = 0; i < 5; i++)
        cutlass_batched_hgemm_tensorop(m, n, k, alpha,
            d_A, lda, stride_A, d_B, ldb, stride_B,
            d_C, ldc, stride_C, beta, batch_count);
    cudaDeviceSynchronize();

    cudaEventRecord(t0);
    for (int i = 0; i < 100; i++)
        cutlass_batched_hgemm_tensorop(m, n, k, alpha,
            d_A, lda, stride_A, d_B, ldb, stride_B,
            d_C, ldc, stride_C, beta, batch_count);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("Perf: %.3f ms/iter (8 batches of 1x256 @ 256x128)\n", ms / 100.0f);

    // Also test MoE-sized: 8 batches of (1, 4096) @ (4096, 11008)
    int m2 = 1, n2 = 11008, k2 = 4096;
    long long stride_A2 = (long long)m2 * k2;
    long long stride_B2 = (long long)k2 * n2;
    long long stride_C2 = (long long)m2 * n2;

    cutlass::half_t *d_A2, *d_B2, *d_C2;
    cudaMalloc(&d_A2, batch_count * stride_A2 * sizeof(cutlass::half_t));
    cudaMalloc(&d_B2, batch_count * stride_B2 * sizeof(cutlass::half_t));
    cudaMalloc(&d_C2, batch_count * stride_C2 * sizeof(cutlass::half_t));

    for (int i = 0; i < 5; i++)
        cutlass_batched_hgemm_tensorop(m2, n2, k2, alpha,
            d_A2, k2, stride_A2, d_B2, n2, stride_B2,
            d_C2, n2, stride_C2, beta, batch_count);
    cudaDeviceSynchronize();

    cudaEventRecord(t0);
    for (int i = 0; i < 20; i++)
        cutlass_batched_hgemm_tensorop(m2, n2, k2, alpha,
            d_A2, k2, stride_A2, d_B2, n2, stride_B2,
            d_C2, n2, stride_C2, beta, batch_count);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms, t0, t1);
    printf("Perf: %.3f ms/iter (8 batches of 1x4096 @ 4096x11008 — MoE decode)\n", ms / 20.0f);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_A2); cudaFree(d_B2); cudaFree(d_C2);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    return pass ? 0 : -1;
}

#endif // BUILD_STANDALONE_TEST
