/* Copyright 2025-2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#if defined(USE_DCU)
#include <hip/amd_detail/amd_hip_bf16.h>

#include <hipcub/hipcub.hpp>

namespace cub = hipcub;
#else
#include <cub/cub.cuh>
#if CUB_VERSION >= 200800
#include <cuda/functional>
#endif
#endif

namespace xllm::kernel::cuda {
#if defined(USE_DCU)
using BFloat16Type = hip_bfloat16;

#define WARP_SIZE 64
#define XLLM_KERNEL_ATTR(MAX_THREADS) __launch_bounds__(MAX_THREADS, 1)
#else
using BFloat16Type = __nv_bfloat16;

// Iluvatar BI-V100 (CoreX / ivcore) uses warp_size=64, same as DCU.
// Standard NVIDIA GPUs use 32.  Detect via compiler-defined macro.
#if defined(__ILUVATAR__) || defined(__COREX__) || defined(USE_ILUVATAR)
#define WARP_SIZE 64
#else
#define WARP_SIZE 32
#endif
#define XLLM_KERNEL_ATTR(MAX_THREADS)
#endif
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// Aligned array type
template <typename T,
          // Number of elements in the array
          int N,
          // Alignment requirement in bytes
          int Alignment = sizeof(T) * N>
class alignas(Alignment) AlignedArray {
  T data[N];
};

// Full-warp mask: 64-bit for warp64 (BI-V100 / DCU), 32-bit otherwise
#if WARP_SIZE == 64
#define XLLM_FULL_MASK 0xffffffffffffffffULL
#else
#define XLLM_FULL_MASK 0xffffffffU
#endif

#define XLLM_SHFL_XOR_SYNC(mask, var, lane_mask) \
  __shfl_xor_sync((mask), (var), (lane_mask))
#define XLLM_SHFL_XOR_SYNC_WIDTH(mask, var, lane_mask, width) \
  __shfl_xor_sync((mask), (var), (lane_mask), (width))

template <typename T>
__device__ __forceinline__ T xllm_ldg(const T* ptr) {
#if defined(USE_DCU)
  return *ptr;
#else
  return __ldg(ptr);
#endif
}

// Define reduction operators based on CUB version.
#if defined(USE_DCU)
using MaxReduceOp = hipcub::Max;
using MinReduceOp = hipcub::Min;
#elif CUB_VERSION >= 200800
using MaxReduceOp = ::cuda::maximum<>;
using MinReduceOp = ::cuda::minimum<>;
#else
using MaxReduceOp = cub::Max;
using MinReduceOp = cub::Min;
#endif

template <typename T>
__device__ float convert_to_float(T x) {
  if constexpr (std::is_same_v<T, __half>) {
    return __half2float(x);
#if defined(USE_DCU)
  } else if constexpr (std::is_same_v<T, hip_bfloat16>) {
    return __bfloat162float(reinterpret_cast<const __hip_bfloat16&>(x));
#else
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return __bfloat162float(x);
#endif

  } else if constexpr (std::is_same_v<T, float>) {
    return x;
  } else {
    return static_cast<float>(x);
  }
}

// Constructs some constants needed to partition the work across threads at
// compile time.
template <typename T, int EXPERTS, int BYTES_PER_LDG>
struct TopkConstants {
  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
  static_assert(EXPERTS / (ELTS_PER_LDG * WARP_SIZE) == 0 ||
                    EXPERTS % (ELTS_PER_LDG * WARP_SIZE) == 0,
                "");
  static constexpr int VECs_PER_THREAD =
      MAX(1, EXPERTS / (ELTS_PER_LDG * WARP_SIZE));
  static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
  static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
  static constexpr int ROWS_PER_WARP = WARP_SIZE / THREADS_PER_ROW;
};

}  // namespace xllm::kernel::cuda

// ============================================================================
// Portable macros and utilities (from xllm/core/kernels/cuda/utils.h)
// ============================================================================
#ifndef DEVICE_INLINE
#define DEVICE_INLINE __device__ __forceinline__
#define HOST_DEVICE_INLINE __host__ __device__ __forceinline__
#endif

template <typename T>
HOST_DEVICE_INLINE constexpr std::enable_if_t<std::is_integral_v<T>, T>
ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

// ============================================================================
// Dispatch macros (from xllm/core/kernels/cuda/utils.h)
// These wrap AT_DISPATCH_SWITCH for float16/bfloat16/float32 dispatch.
// Placed here because cuda_ops_api.h → utils.h is not available on corex
// (glog/logging.h dependency).
// ============================================================================
#ifndef DISPATCH_FLOATING_TYPES
#define DISPATCH_CASE_FLOATING_TYPES(...)              \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)
#define DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))
#define DISPATCH_CASE_HALF_TYPES(...)                 \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)
#define DISPATCH_HALF_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_HALF_TYPES(__VA_ARGS__))
#endif
