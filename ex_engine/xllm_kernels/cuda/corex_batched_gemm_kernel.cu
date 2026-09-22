/*
 * corex_batched_gemm_kernel.cu — FP16 Cu10 TensorOp batched GEMM
 *
 * Uses cutlass::gemm::device::GemmBatched with:
 *   - OpClassTensorOp (TCU, not SIMT)
 *   - arch::Cu10 (BI-V100)
 *   - float accumulation (FP32, not FP16)
 *
 * Source: ex_engine/xllm_kernels/cuda/moe_cutlass_batched.cu (verified 2.462ms)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm_batched.h"

cudaError_t cutlass_batched_hgemm(
    int m, int n, int k,
    __half const *A, int lda, long long int batch_stride_A,
    __half const *B, int ldb, long long int batch_stride_B,
    __half *C, int ldc, long long int batch_stride_C,
    int batch_count)
{
    using Gemm = cutlass::gemm::device::GemmBatched<
        cutlass::half_t,                    // ElementA
        cutlass::layout::RowMajor,          // LayoutA
        cutlass::half_t,                    // ElementB
        cutlass::layout::RowMajor,          // LayoutB
        cutlass::half_t,                    // ElementC
        cutlass::layout::RowMajor,          // LayoutC
        float,                              // ElementAccumulator — FP32!
        cutlass::arch::OpClassTensorOp,     // OperatorClass — TCU!
        cutlass::arch::Cu10                 // ArchTag — BI-V100!
        // Defaults from DefaultGemmConfiguration<OpClassTensorOp, Cu10, half, half, half, float>:
        //   ThreadblockShape = <128, 128, 32>
        //   WarpShape = <32, 32, 32>
        //   InstructionShape = <16, 16, 16>
        //   Stages = 2
    >;

    float alpha = 1.0f;
    float beta = 0.0f;

    Gemm gemm_op;

    cutlass::Status status = gemm_op({
        {m, n, k},
        {reinterpret_cast<cutlass::half_t const *>(A), lda},
        batch_stride_A,
        {reinterpret_cast<cutlass::half_t const *>(B), ldb},
        batch_stride_B,
        {reinterpret_cast<cutlass::half_t const *>(C), ldc},
        batch_stride_C,
        {reinterpret_cast<cutlass::half_t *>(C), ldc},
        batch_stride_C,
        {alpha, beta},
        batch_count
    });

    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}
