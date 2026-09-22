// moe_ops_impl.cu — Implement the 5 missing MoE functions
//
// These functions are declared in ixformer.h (from xllm upstream)
// but NOT present in the base image's libixformer.so.
//
// We implement them using available primitives:
//   - cuinferCustomGemm (from libcuinfer.so) for group_gemm
//   - Pure CUDA kernels for topk_softmax, moe_compute_index, expand, combine
//   - ixformer::functions::cuinfer_gemm (from libixformer.so) as fallback
//
// Reference AST chain:
//   xllm/core/kernels/ilu/fused_moe.cpp  → calls these 5 functions
//   xllm/core/kernels/ilu/group_gemm.cpp → calls moe_w16a16_group_gemm
//   xllm/core/kernels/ilu/ixformer.h     → declares them in ixformer::infer
//
// We provide them in the SAME namespace so ix_full_bridge_v2.cpp links cleanly.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <optional>
#include <vector>
#include <numeric>

// ============================================================================
// Forward-declare cuinfer C API (from libcuinfer.so, confirmed in symbol dump)
// ============================================================================
extern "C" {

typedef struct cuinferContext* cuinferHandle_t;
typedef enum { CUINFER_STATUS_SUCCESS = 0 } cuinferStatus_t;
typedef enum {
    CUINFER_OP_TENSOR_OP_N = 0,
    CUINFER_OP_TENSOR_OP_T = 1,
} cuinferOperation_t;
typedef enum {
    CUINFER_GEMM_DEFAULT = 0,
} cuinferGEMMCustomOption_t;
typedef enum {
    CUINFER_POINTER_MODE_HOST = 0,
} cuinferPointerMode_t;

cuinferStatus_t cuinferCreate(cuinferHandle_t* handle);
cuinferStatus_t cuinferDestroy(cuinferHandle_t handle);
cuinferStatus_t cuinferSetStream(cuinferHandle_t handle, cudaStream_t stream);

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

}  // extern "C"


// ============================================================================
// Kernel 1: topk_softmax
// Adapted from moe_topk_softmax_v3.cu (already working, 64-expert specialized)
// ============================================================================

// Qwen3.5-27B: 128 routed experts
// Block size = 128 threads (1 thread per expert for ≤128 experts)
static constexpr int MOE_MAX_EXPERTS = 128;
static constexpr int MOE_BLOCK = 128;

// All reductions use blockDim.x (dynamic block size, power-of-2)
__device__ float smem_reduce_max(float val, float* smem) {
    int tid = threadIdx.x;
    smem[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    return smem[0];
}

__device__ float smem_reduce_sum(float val, float* smem) {
    int tid = threadIdx.x;
    smem[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    return smem[0];
}

__device__ void smem_argmax(float val, int idx, float* s_val, int* s_idx) {
    int tid = threadIdx.x;
    s_val[tid] = val;
    s_idx[tid] = idx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && s_val[tid + s] > s_val[tid]) {
            s_val[tid] = s_val[tid + s];
            s_idx[tid] = s_idx[tid + s];
        }
        __syncthreads();
    }
}

__global__ void topk_softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ topk_weights,
    int32_t* __restrict__ topk_indices,
    int32_t* __restrict__ token_expert_indices,
    int num_tokens, int num_experts, int topk, bool renormalize
) {
    int row = blockIdx.x;
    if (row >= num_tokens) return;
    int tid = threadIdx.x;

    extern __shared__ char shared_buf[];
    float* smem = (float*)shared_buf;
    int* smem_idx = (int*)(smem + blockDim.x);

    // num_experts passed via gridDim.y (encoded), or read from shared
    // We use a separate parameter for clarity
    float val = (tid < num_experts) ? input[row * num_experts + tid] : -1e30f;

    // Softmax
    float row_max = smem_reduce_max(val, smem);
    val = (tid < num_experts) ? expf(val - row_max) : 0.0f;
    float row_sum = smem_reduce_sum(val, smem);
    val *= (1.0f / row_sum);

    float* out_w = topk_weights + row * topk;
    int32_t* out_idx = topk_indices + row * topk;
    int32_t* out_src = token_expert_indices + row * topk;

    float my_val = val;
    float topk_sum = 0.0f;

    for (int ki = 0; ki < topk; ki++) {
        smem_argmax(my_val, tid, smem, smem_idx);
        float winner_val = smem[0];
        int winner_idx = smem_idx[0];
        __syncthreads();

        if (tid == 0) {
            out_w[ki] = winner_val;
            out_idx[ki] = winner_idx;
            out_src[ki] = row;
        }
        topk_sum += winner_val;
        if (tid == winner_idx) my_val = -1.0f;
        __syncthreads();
    }

    if (renormalize && tid == 0) {
        float inv = 1.0f / (topk_sum + 1e-8f);
        for (int ki = 0; ki < topk; ki++)
            out_w[ki] *= inv;
    }
}


// ============================================================================
// Kernel 2: moe_compute_token_index
// Histogram + prefix sum + scatter — from xllm_kernels/cuda/moe_compute_index.cu
// ============================================================================

__global__ void histogram_kernel(
    const int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ expert_sizes,
    int num_elements, int num_experts
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        int eid = expert_ids[idx];
        if (eid >= 0 && eid < num_experts) {
            atomicAdd(&expert_sizes[eid], 1);
        }
    }
}

__global__ void place_indices_kernel(
    const int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ expert_offsets,  // will be atomicAdd'd
    int32_t* __restrict__ src_dst,
    int32_t* __restrict__ dst_src,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        int eid = expert_ids[idx];
        int pos = atomicAdd(&expert_offsets[eid], 1);
        src_dst[idx] = pos;  // where token idx goes in sorted order
        dst_src[pos] = idx;  // reverse mapping
    }
}


// ============================================================================
// Kernel 3: moe_expand_input
// Gather-based expand: output[i] = input[gather_index[i]]
// ============================================================================

template <typename scalar_t>
__global__ void expand_input_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int32_t* __restrict__ dst_to_src,
    int num_output_tokens, int hidden_size
) {
    int token = blockIdx.x;
    if (token >= num_output_tokens) return;

    int src_token = dst_to_src[token];
    const scalar_t* src = input + (int64_t)src_token * hidden_size;
    scalar_t* dst = output + (int64_t)token * hidden_size;

    for (int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        dst[h] = src[h];
    }
}


// ============================================================================
// Kernel 4: moe_combine_result (weighted sum of expert outputs)
// output[t] = sum_k( weight[t][k] * gemm2_output[flat_index(t,k)] )
// ============================================================================

template <typename scalar_t>
__global__ void combine_result_kernel(
    scalar_t* __restrict__ output,      // [N, H]
    const scalar_t* __restrict__ input,  // [N*topk, H]
    const float* __restrict__ weights,   // [N, topk]
    int num_tokens, int topk, int hidden_size
) {
    int token = blockIdx.x;
    if (token >= num_tokens) return;

    for (int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < topk; k++) {
            int flat = token * topk + k;
            float w = weights[token * topk + k];
            acc += w * __half2float(input[flat * hidden_size + h]);
        }
        output[token * hidden_size + h] = __float2half(acc);
    }
}

// Float specialization
template <>
__global__ void combine_result_kernel<float>(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weights,
    int num_tokens, int topk, int hidden_size
) {
    int token = blockIdx.x;
    if (token >= num_tokens) return;

    for (int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < topk; k++) {
            int flat = token * topk + k;
            float w = weights[token * topk + k];
            acc += w * input[flat * hidden_size + h];
        }
        output[token * hidden_size + h] = acc;
    }
}


// ============================================================================
// C++ wrapper functions — ixformer::infer namespace
// These provide the MISSING symbols that ix_full_bridge_v2.cpp needs.
// ============================================================================

namespace ixformer { namespace infer {

void topk_softmax(
    torch::Tensor& topk_weights,
    torch::Tensor& topk_indices,
    torch::Tensor& token_expert_indices,
    torch::Tensor& gating_output,
    bool renormalize
) {
    int num_tokens = gating_output.size(0);
    int num_experts = gating_output.size(1);
    int topk = topk_weights.size(1);
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto input_f32 = gating_output.to(torch::kFloat32).contiguous();

    // Block size must be >= num_experts, round up to next power of 2
    int block_size = 1;
    while (block_size < num_experts) block_size <<= 1;
    TORCH_CHECK(block_size <= 1024, "Too many experts for topk kernel: ", num_experts);

    size_t smem_bytes = block_size * (sizeof(float) + sizeof(int));
    topk_softmax_kernel<<<num_tokens, block_size, smem_bytes, stream>>>(
        input_f32.data_ptr<float>(),
        topk_weights.data_ptr<float>(),
        topk_indices.data_ptr<int32_t>(),
        token_expert_indices.data_ptr<int32_t>(),
        num_tokens, num_experts, topk, renormalize);
}

void moe_compute_token_index_api(
    torch::Tensor& topk_ids,
    torch::Tensor& src_dst,
    torch::Tensor& dst_src,
    torch::Tensor& expert_sizes_gpu,
    const std::optional<torch::Tensor>& expert_mask,
    const std::optional<torch::Tensor>& expert_sizes_cpu,
    const std::optional<torch::Tensor>& expand_tokens_gpu,
    int64_t start_expert_id,
    int64_t end_expert_id,
    int64_t num_experts
) {
    auto stream = c10::cuda::getCurrentCUDAStream();
    int num_elements = topk_ids.numel();

    // Zero expert_sizes
    cudaMemsetAsync(expert_sizes_gpu.data_ptr<int32_t>(), 0,
                    num_experts * sizeof(int32_t), stream);

    // Phase 1: histogram
    int blocks1 = (num_elements + 255) / 256;
    histogram_kernel<<<blocks1, 256, 0, stream>>>(
        topk_ids.data_ptr<int32_t>(),
        expert_sizes_gpu.data_ptr<int32_t>(),
        num_elements, num_experts);

    // Phase 2: prefix sum for offsets (exclusive scan on GPU)
    // Use a separate buffer for offsets, then reset for place_indices
    auto expert_offsets = torch::zeros({num_experts}, topk_ids.options().dtype(torch::kInt32));
    // Copy sizes → do exclusive scan on CPU (small: 64 experts)
    auto sizes_cpu = expert_sizes_gpu.to(torch::kCPU);
    auto offsets_cpu = torch::zeros({num_experts}, torch::dtype(torch::kInt32));
    int32_t* s = sizes_cpu.data_ptr<int32_t>();
    int32_t* o = offsets_cpu.data_ptr<int32_t>();
    int32_t running = 0;
    for (int i = 0; i < num_experts; i++) {
        o[i] = running;
        running += s[i];
    }
    expert_offsets = offsets_cpu.to(topk_ids.device());

    // Phase 3: place indices
    int blocks3 = (num_elements + 255) / 256;
    place_indices_kernel<<<blocks3, 256, 0, stream>>>(
        topk_ids.data_ptr<int32_t>(),
        expert_offsets.data_ptr<int32_t>(),
        src_dst.data_ptr<int32_t>(),
        dst_src.data_ptr<int32_t>(),
        num_elements);
}

void moe_expand_input(
    torch::Tensor outputs,
    torch::Tensor inputs,
    torch::Tensor dst_to_src,
    const std::optional<torch::Tensor>& src_to_dst,
    int64_t dst_tokens,
    int64_t expand_factor
) {
    auto stream = c10::cuda::getCurrentCUDAStream();
    int hidden_size = inputs.size(1);
    int block = std::min(hidden_size, 256);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(inputs.scalar_type(), "expand_input", [&] {
        expand_input_kernel<scalar_t><<<dst_tokens, block, 0, stream>>>(
            outputs.data_ptr<scalar_t>(),
            inputs.data_ptr<scalar_t>(),
            dst_to_src.data_ptr<int32_t>(),
            dst_tokens, hidden_size);
    });
}

void moe_w16a16_group_gemm(
    torch::Tensor output,
    torch::Tensor inputs,
    torch::Tensor weights,
    torch::Tensor tokens_per_experts,
    const std::optional<torch::Tensor>& dst_to_src,
    const std::optional<torch::Tensor>& bias,
    std::string format,
    int64_t persistent,
    int64_t output_n
) {
    // Implementation: loop over experts, call cuinferCustomGemm for each
    // weights: [num_experts, N, K] with format "TN" means transB
    // For each expert e with count tokens:
    //   A = inputs[offset:offset+count, :]        (count × K, row-major)
    //   B = weights[e, :, :]                      (N × K, needs transB)
    //   C = output[offset:offset+count, :]        (count × N, row-major)
    //   GEMM: C = A × B^T  → (count, K) × (K, N) = (count, N)

    auto stream = c10::cuda::getCurrentCUDAStream();
    int num_experts = weights.size(0);
    int N = weights.size(1);  // output dim
    int K = weights.size(2);  // input dim

    // Get token counts on CPU
    auto counts_cpu = tokens_per_experts.to(torch::kCPU).to(torch::kInt32);
    int32_t* counts = counts_cpu.data_ptr<int32_t>();

    // Create cuinfer handle
    cuinferHandle_t handle;
    cuinferCreate(&handle);
    cuinferSetStream(handle, stream);

    float alpha = 1.0f, beta = 0.0f;

    int offset = 0;
    for (int e = 0; e < num_experts; e++) {
        int M = counts[e];
        if (M <= 0) continue;

        // A: inputs[offset : offset+M, :]  → M × K
        // B: weights[e, :, :]               → N × K (transposed: compute A × B^T)
        // C: output[offset : offset+M, :]   → M × N
        const void* A_ptr = (const char*)inputs.data_ptr() +
            (int64_t)offset * K * inputs.element_size();
        const void* B_ptr = (const char*)weights.data_ptr() +
            (int64_t)e * N * K * weights.element_size();
        void* C_ptr = (char*)output.data_ptr() +
            (int64_t)offset * N * output.element_size();

        cudaDataType_t dtype = (inputs.scalar_type() == torch::kFloat16)
            ? CUDA_R_16F : CUDA_R_32F;

        // cuinferCustomGemm: row-major convention
        // We want C = A × B^T
        // In cuinfer (column-major internally): transa=N, transb=T
        // M_gemm = M (rows of C), N_gemm = N (cols of C), K_gemm = K
        cuinferCustomGemm(
            handle, stream,
            CUINFER_POINTER_MODE_HOST,
            CUINFER_OP_TENSOR_OP_N,  // transa = no transpose
            CUINFER_OP_TENSOR_OP_T,  // transb = transpose (TN format)
            M, N, K,
            &alpha,
            A_ptr, dtype, K, 0,       // lda=K for row-major A
            B_ptr, dtype, K, 0,       // ldb=K for row-major B (will be transposed)
            &beta,
            C_ptr, dtype, N, 0,       // ldc=N for row-major C
            1,                         // batchCount=1
            CUDA_R_32F,               // computeType
            CUDA_R_32F,               // scaleType
            nullptr, nullptr,          // custom pointers
            CUINFER_GEMM_DEFAULT);

        offset += M;
    }

    cuinferDestroy(handle);
}

void moe_output_reduce_sum(
    torch::Tensor outputs,
    torch::Tensor inputs,
    const std::optional<torch::Tensor>& mul_weight,
    const std::optional<torch::Tensor>& mask,
    const std::optional<torch::Tensor>& extra_residual,
    double scaling_factor
) {
    // inputs: [N, topk, H] — expert outputs per token
    // mul_weight: [N, topk]  — router weights
    // outputs: [N, H]       — weighted sum
    auto stream = c10::cuda::getCurrentCUDAStream();
    int num_tokens = inputs.size(0);
    int topk = inputs.size(1);
    int hidden_size = inputs.size(2);
    int block = std::min(hidden_size, 256);

    // Reshape inputs to [N*topk, H] for the kernel
    auto input_flat = inputs.reshape({num_tokens * topk, hidden_size});

    if (inputs.scalar_type() == torch::kFloat16) {
        combine_result_kernel<__half><<<num_tokens, block, 0, stream>>>(
            reinterpret_cast<__half*>(outputs.data_ptr()),
            reinterpret_cast<const __half*>(input_flat.data_ptr()),
            mul_weight.value().data_ptr<float>(),
            num_tokens, topk, hidden_size);
    } else {
        combine_result_kernel<float><<<num_tokens, block, 0, stream>>>(
            outputs.data_ptr<float>(),
            input_flat.data_ptr<float>(),
            mul_weight.value().data_ptr<float>(),
            num_tokens, topk, hidden_size);
    }
}

}}  // namespace ixformer::infer
