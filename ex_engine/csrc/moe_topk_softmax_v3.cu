// moe_topk_softmax_v3.cu — Fused softmax+topk for Qwen3.5 MoE routing
//
// 64 experts, topk=8, one block per row, warp shuffle reduction.
// BI-V100 safe: no warp-size assumption (works with warpSize=32 or 64).
//
// Each block = 64 threads, each thread owns 1 expert value.
// Softmax: parallel exp + warp reduce. TopK: iterative argmax + mask.
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <cuda_runtime.h>

static constexpr int NUM_EXPERTS = 64;
static constexpr int BLOCK_SIZE = 64;  // 1 thread per expert, 1 block per row

// Reduce over all 64 threads using shared memory (warp-size agnostic)
__device__ float block_reduce_max(float val, float* smem) {
    int tid = threadIdx.x;
    smem[tid] = val;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    return smem[0];
}

__device__ float block_reduce_sum(float val, float* smem) {
    int tid = threadIdx.x;
    smem[tid] = val;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    return smem[0];
}

// Find global argmax: returns (max_val, max_idx) via shared memory
__device__ void block_argmax(float val, int idx, float* s_val, int* s_idx) {
    int tid = threadIdx.x;
    s_val[tid] = val;
    s_idx[tid] = idx;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_val[tid + s] > s_val[tid]) {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        __syncthreads();
    }
}

__global__ void topk_gating_softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output_weights,
    int32_t* __restrict__ output_indices,
    int32_t* __restrict__ output_source_rows,
    int num_tokens, int k, bool renormalize
) {
    int row = blockIdx.x;
    if (row >= num_tokens) return;
    int tid = threadIdx.x;  // 0..63, one per expert

    __shared__ float smem[BLOCK_SIZE];
    __shared__ int smem_idx[BLOCK_SIZE];

    // Load gating logit for this expert
    float val = input[row * NUM_EXPERTS + tid];

    // Softmax: max-subtract, exp, normalize
    float row_max = block_reduce_max(val, smem);
    val = expf(val - row_max);
    float row_sum = block_reduce_sum(val, smem);
    val *= (1.0f / row_sum);

    // Output pointers for this row
    float* out_w = output_weights + row * k;
    int32_t* out_idx = output_indices + row * k;
    int32_t* out_src = output_source_rows + row * k;

    // Iterative top-k: find max, write, mask, repeat
    float topk_sum = 0.0f;
    float my_val = val;  // will be set to -1 when selected

    for (int ki = 0; ki < k; ki++) {
        block_argmax(my_val, tid, smem, smem_idx);
        // Thread 0 has the winner
        float winner_val = smem[0];
        int winner_idx = smem_idx[0];
        // Broadcast via shared memory (already in smem[0])
        __syncthreads();

        if (tid == 0) {
            out_w[ki] = winner_val;
            out_idx[ki] = winner_idx;
            out_src[ki] = row;
        }
        topk_sum += winner_val;

        // Mask out the selected expert
        if (tid == winner_idx) my_val = -1.0f;
        __syncthreads();
    }

    if (renormalize && tid == 0) {
        float inv = 1.0f / (topk_sum + 1e-8f);
        for (int ki = 0; ki < k; ki++)
            out_w[ki] *= inv;
    }
}

std::vector<torch::Tensor> moe_topk_softmax(
    torch::Tensor gating_output, int64_t topk, bool renormalize
) {
    int num_tokens = gating_output.size(0);
    int num_experts = gating_output.size(1);
    TORCH_CHECK(num_experts == 64, "Specialized for 64 experts, got ", num_experts);

    auto opts_f = torch::dtype(torch::kFloat32).device(gating_output.device());
    auto opts_i = torch::dtype(torch::kInt32).device(gating_output.device());
    auto topk_weights = torch::empty({num_tokens, topk}, opts_f);
    auto topk_ids = torch::empty({num_tokens, topk}, opts_i);
    auto token_expert_ids = torch::empty({num_tokens, topk}, opts_i);

    auto input_f32 = gating_output.to(torch::kFloat32).contiguous();

    topk_gating_softmax_kernel<<<num_tokens, BLOCK_SIZE, 0,
        c10::cuda::getCurrentCUDAStream()>>>(
        input_f32.data_ptr<float>(),
        topk_weights.data_ptr<float>(),
        topk_ids.data_ptr<int32_t>(),
        token_expert_ids.data_ptr<int32_t>(),
        num_tokens, topk, renormalize);

    return {topk_weights, topk_ids, token_expert_ids};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("moe_topk_softmax", &moe_topk_softmax,
          "Fused softmax+topk for MoE routing (64 experts, shared mem, warp-agnostic)");
}
