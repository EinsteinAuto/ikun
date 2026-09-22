#pragma once

#include <stdexcept>
#include "status.h"

namespace ixformer::comm {

class CommError : public std::runtime_error {
public:
    template<class ERROR_STR>
    CommError(CommStatus error, const ERROR_STR str) : error_{error}, std::runtime_error(str) {}

    CommStatus status() {
        return error_;
    }

private:
    CommStatus error_;
};

}// namespace ixformer::comm
