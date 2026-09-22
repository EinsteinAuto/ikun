// ex_engine/csrc/factor_moe_topk_softmax.cu
//
// Factor 0: MOE_TOPK_SOFTMAX — fused softmax + top-k for MoE routing
//
// Based on: ds_vllm/csrc/moe/topk_softmax_kernels.cu (TensorRT-LLM derived)
// and: xllm/kernels/cuda/moe/moe_topk_softmax_kernels.cuh
//
// Key insight from upstream: 64 experts is a power-of-2, so we use the
// specialized topkGating kernel that packs multiple rows per warp and
// eliminates shared memory entirely.
//
// For NUM_EXPERTS=64, VPT=2, THREADS_PER_ROW=32:
//   - Each warp handles 1 row (64 experts / 2 per thread = 32 threads)
//   - Softmax via warp shuffle butterfly reduce
//   - TopK via iterative warp argmax with winner suppression
//   - No shared memory needed, no CTA sync needed
//
// BI-V100 (SM70): 32-wide warps, 16 SMs, 49152 SMEM (not used here)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <stdint.h>

extern "C" {
#include "ex_engine.h"
}

// ---------------------------------------------------------------------------
// Compile-time config for Qwen3.5: 64 experts, top_k=8
// ---------------------------------------------------------------------------
static constexpr int NUM_EXPERTS = 64;
static constexpr int VPT = 2;  // Values Per Thread (64 experts / 32 threads)
static constexpr int THREADS_PER_ROW = NUM_EXPERTS / VPT;  // 32 = 1 warp
static constexpr int WARPS_PER_CTA = 4;
static constexpr int ROWS_PER_CTA = WARPS_PER_CTA;  // 1 row per warp

// ---------------------------------------------------------------------------
// topkGatingSoftmax kernel — directly from ds_vllm/TRT-LLM pattern
//
// Each warp processes one token's row of 64 experts.
// Thread i in warp holds experts [2i, 2i+1] (VPT=2).
// All reduces via warp shuffle (__shfl_xor_sync) — zero shared memory.
// ---------------------------------------------------------------------------

__global__ void topk_gating_softmax_kernel(
    const float* __restrict__ input,    // (num_tokens, num_experts)
    float* __restrict__ output,         // (num_tokens, k)
    int32_t* __restrict__ indices,      // (num_tokens, k)
    int32_t* __restrict__ source_rows,  // (num_tokens, k) — token_expert_indices
    int num_tokens,
    int k,
    bool renormalize
) {
    // CTA and warp row assignment
    const int cta_base_row = blockIdx.x * ROWS_PER_CTA;
    const int warp_id = threadIdx.y;
    const int thread_row = cta_base_row + warp_id;

    if (thread_row >= num_tokens) return;

    const int lane = threadIdx.x;

    // ===== Load this thread's VPT=2 experts =====
    const float* row_ptr = input + thread_row * NUM_EXPERTS;
    float row_chunk[VPT];
    #pragma unroll
    for (int i = 0; i < VPT; i++) {
        row_chunk[i] = row_ptr[lane * VPT + i];
    }

    // ===== Softmax: max reduction via butterfly =====
    float thread_max = row_chunk[0];
    #pragma unroll
    for (int i = 1; i < VPT; i++) {
        thread_max = fmaxf(thread_max, row_chunk[i]);
    }
    // Butterfly reduce for max across warp (32 threads = 64 experts)
    #pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask >>= 1) {
        thread_max = fmaxf(thread_max,
            __shfl_xor_sync(0xFFFFFFFF, thread_max, mask, THREADS_PER_ROW));
    }

    // ===== Softmax: exp and sum =====
    float row_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < VPT; i++) {
        row_chunk[i] = expf(row_chunk[i] - thread_max);
        row_sum += row_chunk[i];
    }
    // Butterfly reduce for sum
    #pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask >>= 1) {
        row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, mask, THREADS_PER_ROW);
    }

    // ===== Normalize =====
    float inv_sum = 1.0f / row_sum;
    #pragma unroll
    for (int i = 0; i < VPT; i++) {
        row_chunk[i] *= inv_sum;
        // Clamp NaN/Inf to 0 — prevents duplicate expert IDs downstream
        if (isnan(row_chunk[i]) || isinf(row_chunk[i])) {
            row_chunk[i] = 0.0f;
        }
    }

    // ===== TopK via iterative warp argmax with winner suppression =====
    int start_col = lane * VPT;
    float selected_sum = 0.0f;

    for (int k_idx = 0; k_idx < k; k_idx++) {
        // Thread-local argmax
        float max_val = row_chunk[0];
        int expert = start_col;
        #pragma unroll
        for (int i = 1; i < VPT; i++) {
            if (row_chunk[i] > max_val) {
                max_val = row_chunk[i];
                expert = start_col + i;
            }
        }

        // Warp butterfly argmax — all threads agree on winner
        #pragma unroll
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask >>= 1) {
            float other_val = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, THREADS_PER_ROW);
            int other_expert = __shfl_xor_sync(0xFFFFFFFF, expert, mask, THREADS_PER_ROW);
            // Lower index wins ties (stable selection)
            if (other_val > max_val ||
                (other_val == max_val && other_expert < expert)) {
                max_val = other_val;
                expert = other_expert;
            }
        }

        // Lane 0 writes result
        if (lane == 0) {
            int idx = k * thread_row + k_idx;
            output[idx] = max_val;
            indices[idx] = expert;
            source_rows[idx] = k_idx * num_tokens + thread_row;
            selected_sum += max_val;
        }

        // Suppress winner: the thread that owns the winning expert zeroes it
        int winner_ldg = expert / VPT;  // which thread owns this expert
        int winner_offset = expert % VPT;  // which slot in that thread
        if (lane == winner_ldg) {
            row_chunk[winner_offset] = -1.0f;  // suppress for next iteration
        }
    }

    // ===== Renormalize =====
    if (renormalize && lane == 0) {
        float denom = (selected_sum > 0.0f) ? selected_sum : 1.0f;
        for (int k_idx = 0; k_idx < k; k_idx++) {
            int idx = k * thread_row + k_idx;
            output[idx] /= denom;
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatch function matching EX Engine interface
// ---------------------------------------------------------------------------

static int moe_topk_softmax_dispatch(
    void*          output_v,
    const void*    input_v,
    const void*    aux_inputs[],
    int            n_aux,
    const int64_t  dims[],
    int            n_dims,
    void*          stream
) {
    // dims[0] = T (tokens), dims[1] = num_experts, dims[2] = top_k
    // output  = topk_weights (T, K) float32
    // aux[0]  = topk_ids (T, K) int32
    // aux[1]  = token_expert_indices (T, K) int32  [needed by vllm]
    if (n_dims < 3 || !output_v || !input_v) return -1;

    int T = (int)dims[0];
    int num_experts = (int)dims[1];
    int top_k = (int)dims[2];

    // Currently only optimized for 64 experts (Qwen3.5-MoE)
    if (num_experts != NUM_EXPERTS) return -1;

    float* topk_weights = (float*)output_v;
    int32_t* topk_ids = (n_aux >= 1 && aux_inputs) ? (int32_t*)aux_inputs[0] : NULL;
    int32_t* token_expert_indices = (n_aux >= 2 && aux_inputs) ? (int32_t*)aux_inputs[1] : NULL;
    const float* logits = (const float*)input_v;

    if (!topk_ids) return -1;

    cudaStream_t cu_stream = (cudaStream_t)stream;

    int num_blocks = (T + ROWS_PER_CTA - 1) / ROWS_PER_CTA;
    dim3 grid(num_blocks);
    dim3 block(THREADS_PER_ROW, WARPS_PER_CTA);  // (32, 4) = 128 threads

    topk_gating_softmax_kernel<<<grid, block, 0, cu_stream>>>(
        logits, topk_weights, topk_ids, token_expert_indices,
        T, top_k, true /* renormalize */
    );

    return 0;
}

// ---------------------------------------------------------------------------
// Also provide a direct C call for the Python ctypes loader
// ---------------------------------------------------------------------------
extern "C" int ex_dispatch_moe_topk_softmax(
    float* topk_weights,
    int32_t* topk_ids,
    const float* logits,
    int T, int E, int top_k,
    void* stream
) {
    if (E != NUM_EXPERTS) return -1;

    cudaStream_t cu_stream = (cudaStream_t)stream;
    int num_blocks = (T + ROWS_PER_CTA - 1) / ROWS_PER_CTA;
    dim3 grid(num_blocks);
    dim3 block(THREADS_PER_ROW, WARPS_PER_CTA);

    // Allocate token_expert_indices alongside (vllm needs it)
    // For EX dispatch, caller is responsible for this buffer
    // Here we skip it and only write topk_weights + topk_ids
    topk_gating_softmax_kernel<<<grid, block, 0, cu_stream>>>(
        logits, topk_weights, topk_ids, NULL,
        T, top_k, true
    );

    return 0;
}

// ---------------------------------------------------------------------------
// .so export
// ---------------------------------------------------------------------------
static ex_factor_t s_factor;

extern "C" ex_factor_t* ex_get_factor(const ex_hardware_t* hw) {
    s_factor.factor_id   = EX_FACTOR_MOE_TOPK_SOFTMAX;
    s_factor.name        = "moe_topk_softmax";
    s_factor.version     = "2.0.0";
    s_factor.tuning      = (ex_tuning_t){
        .threads_per_block = THREADS_PER_ROW * WARPS_PER_CTA,  // 128
        .items_per_thread  = VPT,        // 2 experts per thread
        .vec_size          = 1,          // scalar loads (64 < 128B threshold)
        .shared_mem_bytes  = 0,          // zero — all warp shuffle
        .num_warps         = WARPS_PER_CTA,  // 4 rows per CTA
        .num_stages        = 1
    };
    s_factor.kernel          = moe_topk_softmax_dispatch;
    s_factor.kernel_fallback = NULL;
    return &s_factor;
}
