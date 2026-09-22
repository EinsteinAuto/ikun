// cuinfer_gemm_wrapper.cu — Wrapper around cuinferCustomGemm
//
// ixformer::functions::cuinfer_gemm exists in libixformer.so but
// takes ixformer::Tensor (not torch::Tensor). We need a torch-compatible
// wrapper that calls the C API directly.
//
// Symbol dump shows cuinferCustomGemm in libcuinfer.so with signature:
//   cuinferCustomGemm(handle, stream, ptrMode, transa, transb,
//                     m, n, k, alpha, A, Atype, lda, strideA,
//                     B, Btype, ldb, strideB, beta,
//                     C, Ctype, ldc, strideC, batchCount,
//                     computeType, scaleType, customHostPtr, customDevicePtr, customOption)
//
// Reference:
//   cat_files/ixinfer.h — cuinferCustomGemm signature
//   libixformer.so — ixformer::functions::cuinfer_gemm (confirmed in symbol dump)

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include "cuinfer_handle.h"

// cuinferCustomGemm is already declared in cuinfer_handle.h extern "C" block
// We add the full signature here
extern "C" {
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

// CUDA_R_16F = 2, CUDA_R_32F = 0 (from cudaDataType_t)
static constexpr int kFP16 = 2;
static constexpr int kFP32 = 0;


// ============================================================================
// cuinfer_gemm: C = alpha * A @ B + beta * C
//
// A: (M, K) row-major fp16
// B: (K, N) row-major fp16  (or (N, K) if transb)
// C: (M, N) row-major fp16
// ============================================================================
torch::Tensor cuinfer_gemm(
    torch::Tensor A,       // (M, K)
    torch::Tensor B,       // (K, N) or (N, K) if trans_b
    bool trans_b)
{
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(A.scalar_type() == torch::kHalf, "A must be fp16");
    TORCH_CHECK(B.scalar_type() == torch::kHalf, "B must be fp16");

    int M = A.size(0);
    int K = A.size(1);
    int N = trans_b ? B.size(0) : B.size(1);

    if (!trans_b) {
        TORCH_CHECK(B.size(0) == K, "B rows must equal K");
    } else {
        TORCH_CHECK(B.size(1) == K, "B cols must equal K when transposed");
    }

    auto C = torch::zeros({M, N}, A.options());
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    auto handle = CuinferHandle::get(stream);

    if (!handle) {
        // Fallback to torch::mm
        if (trans_b) {
            return torch::mm(A.to(torch::kFloat32), B.t().to(torch::kFloat32)).to(torch::kHalf);
        }
        return torch::mm(A.to(torch::kFloat32), B.to(torch::kFloat32)).to(torch::kHalf);
    }

    float alpha = 1.0f, beta = 0.0f;
    int transa = 0;  // N = no transpose
    int transb_flag = trans_b ? 1 : 0;

    int lda = K;
    int ldb = trans_b ? K : N;
    int ldc = N;

    int status = cuinferCustomGemm(
        handle, stream,
        0,                           // CUINFER_POINTER_MODE_HOST
        transa, transb_flag,
        M, N, K,
        &alpha,
        A.data_ptr(), kFP16, lda, 0,
        B.data_ptr(), kFP16, ldb, 0,
        &beta,
        C.data_ptr(), kFP16, ldc, 0,
        1,                           // batchCount
        kFP32, kFP32,                // computeType, scaleType
        nullptr, nullptr, 0);

    TORCH_CHECK(status == 0, "cuinferCustomGemm failed with status ", status);
    return C;
}


// ============================================================================
// cuinfer_gemm_batched: batched version
// A: (batch, M, K), B: (batch, K, N) or (batch, N, K)
// ============================================================================
torch::Tensor cuinfer_gemm_batched(
    torch::Tensor A,
    torch::Tensor B,
    bool trans_b)
{
    TORCH_CHECK(A.dim() == 3 && B.dim() == 3, "inputs must be 3D");

    int batch = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = trans_b ? B.size(1) : B.size(2);

    auto C = torch::zeros({batch, M, N}, A.options());
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    auto handle = CuinferHandle::get(stream);

    float alpha = 1.0f, beta = 0.0f;
    int lda = K, ldb = trans_b ? K : N, ldc = N;
    long long strideA = (long long)M * K;
    long long strideB = trans_b ? (long long)N * K : (long long)K * N;
    long long strideC = (long long)M * N;

    int status = cuinferCustomGemm(
        handle, stream,
        0,
        0, trans_b ? 1 : 0,
        M, N, K,
        &alpha,
        A.data_ptr(), kFP16, lda, strideA,
        B.data_ptr(), kFP16, ldb, strideB,
        &beta,
        C.data_ptr(), kFP16, ldc, strideC,
        batch,
        kFP32, kFP32,
        nullptr, nullptr, 0);

    TORCH_CHECK(status == 0, "cuinferCustomGemm batched failed: ", status);
    return C;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cuinfer_gemm", &cuinfer_gemm,
          "GEMM via cuinferCustomGemm (fp16, Cu10)",
          py::arg("A"), py::arg("B"), py::arg("trans_b") = false);
    m.def("cuinfer_gemm_batched", &cuinfer_gemm_batched,
          "Batched GEMM via cuinferCustomGemm",
          py::arg("A"), py::arg("B"), py::arg("trans_b") = false);
}
