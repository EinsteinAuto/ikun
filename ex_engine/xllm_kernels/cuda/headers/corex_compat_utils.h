// corex_compat_utils.h — Lightweight replacement for xllm's utils.h
// Removes glog/tvm dependencies for BI-V100 corex compilation
// Provides CHECK macro via TORCH_CHECK and DISPATCH macros from device_utils.cuh

#pragma once

#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>

// Replace glog CHECK with TORCH_CHECK
#ifndef CHECK
#define CHECK(cond) TORCH_CHECK(cond)
#endif

#ifndef CHECK_EQ
#define CHECK_EQ(a, b) TORCH_CHECK((a) == (b))
#endif

#ifndef CHECK_GE
#define CHECK_GE(a, b) TORCH_CHECK((a) >= (b))
#endif

// Include device_utils for DISPATCH_HALF_TYPES etc
#include "device_utils.cuh"

// ffi namespace stub (some headers reference it)
namespace ffi {
template <typename T>
using Array = std::vector<T>;
}

// HOST_DEVICE_INLINE
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
#define HOST_DEVICE_INLINE __host__ __device__ __forceinline__
#else
#define HOST_DEVICE_INLINE inline
#endif
