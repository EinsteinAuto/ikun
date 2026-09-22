#pragma once

#include <stdexcept>

#include "status.h"

namespace ixformer::kernels {

class KernelError : public std::runtime_error {
public:
    template<class ERROR_STR>
    KernelError(KernelStatus error, const ERROR_STR str) : error_{error}, std::runtime_error(str) {}

    KernelStatus status() {
        return error_;
    }

private:
    KernelStatus error_;
};

}// namespace ixformer::kernels
