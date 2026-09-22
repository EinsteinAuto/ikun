// xllm_norm_bind.cpp — pybind11 entry point for xllm norm kernels
// Compiled together with norm.cu to produce xllm_norm.so
//
// Exports: rms_norm, fused_add_rms_norm

#include <torch/extension.h>

namespace xllm::kernel::cuda {
void rms_norm(torch::Tensor output, torch::Tensor input,
              torch::Tensor weight, double eps);
void fused_add_rms_norm(torch::Tensor& input, torch::Tensor& residual,
                        torch::Tensor& weight, double epsilon);
}  // namespace xllm::kernel::cuda

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm", &xllm::kernel::cuda::rms_norm,
          "RMS Norm (xllm CUDA kernel)",
          py::arg("output"), py::arg("input"),
          py::arg("weight"), py::arg("eps") = 1e-6);
    m.def("fused_add_rms_norm", &xllm::kernel::cuda::fused_add_rms_norm,
          "Fused Add + RMS Norm (xllm CUDA kernel)",
          py::arg("input"), py::arg("residual"),
          py::arg("weight"), py::arg("epsilon") = 1e-6);
}
