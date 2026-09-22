#pragma once
#include "comm/core/common.h"


namespace ixformer::comm {

enum CommStatus {
    commSuccess,
    commFail,
    commCudaError,
    commNcclError,
    commInvalidArgument,
    commUnsupported,
    commInternalError,
    commInvalidComm// maybe comm is nullptr
};


std::string to_string(CommStatus status);


}// namespace ixformer::comm
