// cuinfer_handle.h — Singleton handle manager for libcuinfer.so
//
// cuinferCreate/Destroy is expensive. This provides a thread-safe
// singleton that creates once and reuses.
//
// Usage:
//   #include "cuinfer_handle.h"
//   cuinferHandle_t h = CuinferHandle::get(stream);
//
// Reference: ixformer::Context::default_cuinfer_handle (in libixformer.so)

#pragma once

#include <cuda_runtime.h>
#include <mutex>
#include <cstdio>

// Forward-declare cuinfer C API
extern "C" {

typedef struct cuinferContext* cuinferHandle_t;

typedef enum {
    CUINFER_STATUS_SUCCESS_H = 0,
} cuinferStatus_h_t;

int cuinferCreate(cuinferHandle_t* handle);
int cuinferDestroy(cuinferHandle_t handle);
int cuinferSetStream(cuinferHandle_t handle, cudaStream_t stream);

}  // extern "C"


class CuinferHandle {
public:
    static cuinferHandle_t get(cudaStream_t stream = nullptr) {
        static CuinferHandle instance;
        if (stream && stream != instance.last_stream_) {
            cuinferSetStream(instance.handle_, stream);
            instance.last_stream_ = stream;
        }
        return instance.handle_;
    }

private:
    cuinferHandle_t handle_ = nullptr;
    cudaStream_t last_stream_ = nullptr;

    CuinferHandle() {
        int status = cuinferCreate(&handle_);
        if (status != 0) {
            fprintf(stderr, "[cuinfer_handle] WARNING: cuinferCreate failed (%d)\n", status);
            handle_ = nullptr;
        }
    }

    ~CuinferHandle() {
        if (handle_) {
            cuinferDestroy(handle_);
        }
    }

    CuinferHandle(const CuinferHandle&) = delete;
    CuinferHandle& operator=(const CuinferHandle&) = delete;
};
