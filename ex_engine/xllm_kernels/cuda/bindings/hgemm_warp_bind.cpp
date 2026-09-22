// hgemm_warp_bind.cpp — pybind11 for hgemm_warptiling (kernel 10, warp64)

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

void launch_hgemm_warptiling(
    int M, int N, int K, float alpha,
    const __half* A, const __half* B,
    float beta, __half* C, cudaStream_t stream);

torch::Tensor hgemm_warp(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.scalar_type() == torch::kHalf, "A must be fp16");
    TORCH_CHECK(B.scalar_type() == torch::kHalf, "B must be fp16");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dims must match");

    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    launch_hgemm_warptiling(M, N, K, 1.0f,
        reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(B.data_ptr<at::Half>()),
        0.0f,
        reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
        stream);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hgemm_warp", &hgemm_warp,
          "FP16 GEMM warp-tiling (siboehm K10, WARPSIZE=64 for BI-V100)");
}
