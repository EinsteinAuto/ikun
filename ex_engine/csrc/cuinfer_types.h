// cuinfer_types.h — C API types from libcuinfer.so
//
// Extracted from: cat_files/ixinfer.h (165952 bytes, from real device)
// Only the types/enums needed by our GEMM and MoE code.
//
// This header replaces the scattered extern "C" blocks across
// moe_ops_impl.cu, cuinfer_gemm_wrapper.cu, gemm_grouped.cu.

#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// --- Handle ---
struct cuinferContext;
typedef struct cuinferContext* cuinferHandle_t;

// --- Status ---
typedef enum {
    CUINFER_STATUS_SUCCESS = 0,
    CUINFER_STATUS_NOT_INITIALIZED = 1,
    CUINFER_STATUS_ALLOC_FAILED = 2,
    CUINFER_STATUS_BAD_PARAM = 3,
    CUINFER_STATUS_INTERNAL_ERROR = 4,
    CUINFER_STATUS_INVALID_VALUE = 5,
    CUINFER_STATUS_ARCH_MISMATCH = 6,
    CUINFER_STATUS_EXECUTION_FAILED = 8,
    CUINFER_STATUS_NOT_SUPPORTED = 9,
} cuinferStatus_t;

// --- Data types ---
typedef enum {
    CUINFER_DATA_FLOAT = 0,
    CUINFER_DATA_DOUBLE = 1,
    CUINFER_DATA_HALF = 2,
    CUINFER_DATA_INT8 = 3,
    CUINFER_DATA_INT32 = 4,
    CUINFER_DATA_INT8x4 = 5,
    CUINFER_DATA_UINT8 = 6,
    CUINFER_DATA_UINT8x4 = 7,
    CUINFER_DATA_INT16 = 8,
    CUINFER_DATA_BFLOAT16 = 9,
} cuinferDataType_t;

// --- Operations ---
typedef enum {
    CUINFER_OP_N = 0,   // no transpose
    CUINFER_OP_T = 1,   // transpose
    CUINFER_OP_C = 2,   // conjugate transpose
} cuinferOperation_t;

// --- Pointer mode ---
typedef enum {
    CUINFER_POINTER_MODE_HOST = 0,
    CUINFER_POINTER_MODE_DEVICE = 1,
} cuinferPointerMode_t;

// --- GEMM custom option ---
typedef enum {
    CUINFER_GEMM_DEFAULT = 0,
} cuinferGEMMCustomOption_t;

// --- Reduce ops ---
typedef enum {
    CUINFER_REDUCE_TENSOR_ADD = 0,
    CUINFER_REDUCE_TENSOR_MUL = 1,
    CUINFER_REDUCE_TENSOR_MIN = 2,
    CUINFER_REDUCE_TENSOR_MAX = 3,
} cuinferReduceTensorOp_t;

// --- Softmax ---
typedef enum {
    CUINFER_SOFTMAX_FAST = 0,
    CUINFER_SOFTMAX_ACCURATE = 1,
    CUINFER_SOFTMAX_LOG = 2,
} cuinferSoftmaxAlgorithm_t;

typedef enum {
    CUINFER_SOFTMAX_MODE_INSTANCE = 0,
    CUINFER_SOFTMAX_MODE_CHANNEL = 1,
} cuinferSoftmaxMode_t;


// ============================================================================
// Function declarations (confirmed in libcuinfer.so symbol dump)
// ============================================================================

cuinferStatus_t cuinferCreate(cuinferHandle_t* handle);
cuinferStatus_t cuinferDestroy(cuinferHandle_t handle);
cuinferStatus_t cuinferSetStream(cuinferHandle_t handle, cudaStream_t stream);
cuinferStatus_t cuinferGetStream(cuinferHandle_t handle, cudaStream_t* stream);
size_t cuinferGetVersion(void);
const char* cuinferGetErrorString(cuinferStatus_t status);

// GEMM
cuinferStatus_t cuinferCustomGemm(
    cuinferHandle_t handle, cudaStream_t stream,
    cuinferPointerMode_t ptrMode,
    cuinferOperation_t transa, cuinferOperation_t transb,
    int m, int n, int k,
    const void* alpha,
    const void* A, cudaDataType_t Atype, int lda, long long int strideA,
    const void* B, cudaDataType_t Btype, int ldb, long long int strideB,
    const void* beta,
    void* C, cudaDataType_t Ctype, int ldc, long long int strideC,
    int batchCount,
    cudaDataType_t computeType, cudaDataType_t scaleType,
    const void* customHostPtr, const void* customDevicePtr,
    cuinferGEMMCustomOption_t customOption);

cuinferStatus_t cuinferCustomGemmEx(
    cuinferHandle_t handle, cudaStream_t stream,
    cuinferPointerMode_t ptrMode,
    cuinferOperation_t transa, cuinferOperation_t transb,
    int m, int n, int k,
    const void* alpha,
    const void* A, cudaDataType_t Atype, int lda, long long int strideA,
    const void* B, cudaDataType_t Btype, int ldb, long long int strideB,
    const void* beta,
    void* C, cudaDataType_t Ctype, int ldc, long long int strideC,
    int batchCount,
    cudaDataType_t computeType, cudaDataType_t scaleType,
    const void* customHostPtr, const void* customDevicePtr,
    cuinferGEMMCustomOption_t customOption,
    const void* workspace);

// TopK
cuinferStatus_t cuinferTopK(
    cuinferHandle_t handle,
    const void* input, int n, int m, int top_k,
    int sort_dim, bool largest, bool sorted,
    void* out_value, int* out_indice,
    cuinferDataType_t datatype, void* workspace);

cuinferStatus_t cuinferGetTopKWorkspace(
    cuinferHandle_t handle,
    int n, int m, int top_k,
    cuinferDataType_t datatype, size_t* workspace_size);

cuinferStatus_t cuinferTopKBatch(
    cuinferHandle_t handle,
    const void* input, int top_k, int batch, int n, int m, int k,
    bool largest, bool sorted, int sort_dim,
    void* output, int* indice,
    cuinferDataType_t datatype, void* workspace);

// Softmax
cuinferStatus_t cuinferSoftmaxForward(
    cuinferHandle_t handle,
    cuinferSoftmaxAlgorithm_t algo,
    cuinferSoftmaxMode_t mode,
    const void* alpha,
    const void* xDesc, const void* x,
    const void* beta,
    const void* yDesc, void* y);

// Reduce
cuinferStatus_t cuinferReduce(
    cuinferHandle_t handle,
    const void* in, void* out,
    cuinferDataType_t in_type,
    cuinferDataType_t acc_type,
    cuinferDataType_t out_type,
    cuinferReduceTensorOp_t reduce_op,
    int n_dims, const int* dims,
    int n_reduce_dims, const int* reduce_dim_index,
    void* workspace);

#ifdef __cplusplus
}  // extern "C"
#endif
