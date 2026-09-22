// xllm_activation_bind.cpp
#include <torch/extension.h>

namespace xllm::kernel::cuda {
void act_and_mul(torch::Tensor out, torch::Tensor input,
                 const std::string& act_mode);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("silu_and_mul", [](torch::Tensor out, torch::Tensor input) {
        xllm::kernel::cuda::act_and_mul(out, input, "silu");
    }, "SiLU and Mul", py::arg("out"), py::arg("input"));
    m.def("gelu_and_mul", [](torch::Tensor out, torch::Tensor input) {
        xllm::kernel::cuda::act_and_mul(out, input, "gelu");
    }, "GELU and Mul", py::arg("out"), py::arg("input"));
    m.def("act_and_mul", &xllm::kernel::cuda::act_and_mul,
          "Activation and Mul", py::arg("out"), py::arg("input"), py::arg("act_mode"));
}
