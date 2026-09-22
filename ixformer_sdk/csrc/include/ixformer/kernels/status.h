#pragma once

#include <string>

namespace ixformer::kernels {

enum KernelStatus {
    kernelSuccess,
    kernelFail,
    kernelCudaError,
    kernelInvalidArgument,
    kernelCuinferError,
    kernelUnsupported,
};


std::string to_string(KernelStatus status);


}// namespace ixformer::kernels
