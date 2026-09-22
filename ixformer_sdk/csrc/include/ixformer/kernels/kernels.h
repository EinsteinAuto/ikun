#pragma once

#include "error.h"
#include "status.h"
#include "tensor.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <ixinfer.h>


namespace ixformer::kernels::infer {


/// ========================================================
// Paged attention
// ========================================================

typedef enum {
    KV_CACHE_FORMAT_STD,
    KV_CACHE_FORMAT_NHD,
    KV_CACHE_FORMAT_HND
} kvCacheFormat;

/**
 * @brief
 *
 * @tparam DType: input data type: half or bfloat16
 * @tparam IndexDType: index data type, int32
 * @tparam Format: KV_CACHE_FORMAT_STD or  KV_CACHE_FORMAT_NHD or  KV_CACHE_FORMAT_HND
 * @param key: key, shape: [num_tokens,num_heads, head_size]
 * @param value: value, shape: [num_tokens, num_heads, head_size]
 * @param key_cache: key cache, half or bfloat16
 *        1. kv_cache_format == "STD", shape: [num_blocks, num_kv_heads, head_size // x, block_size, x]
 *        2. kv_cache_format == "NHD", shape: [num_blocks, block_size, num_kv_heads, head_size]
 *        3. kv_cache_format == "HND", shape: [num_blocks, num_kv_heads, block_size, head_size]
 * @param value_cache: value cache, half or bfloat16
 *        1. kv_cache_format == "STD", shape: [num_blocks, num_kv_heads, head_size, block_size]
 *        2. kv_cache_format == "NHD", shape: [num_blocks, block_size, num_kv_heads, head_size]
 *        3. kv_cache_format == "HND", shape: [num_blocks, num_kv_heads, block_size, head_size]
 * @param slot_mapping: The mapping position of the token in blocks.
 * @param key_stride: key.stride(0)
 * @param value_stride: value.stride(0)
 * @param key_cache_stride: key_cache.stride(0)
 * @param value_cache_stride: value_cache.stride(0)
 * @param num_tokens: the number of tokens
 * @param num_heads: the number of heads
 * @param head_size: head size
 * @param block_size: tokens of each page
 * @param x: usually x = 16 / sizeof(DType)
 * @param stream: CUDA Stream
 */
template<typename DType, typename IndexDType, kvCacheFormat Format>
void paged_attention_cache_appended_f16_kernel(
        const DType *key,
        const DType *value,
        DType *key_cache,
        DType *value_cache,
        const IndexDType *slot_mapping,
        unsigned key_stride,
        unsigned value_stride,
        unsigned key_cache_stride,
        unsigned value_cache_stride,
        unsigned num_tokens,
        unsigned num_heads,
        unsigned head_size,
        unsigned block_size,
        unsigned x,
        cudaStream_t stream);

/**
 * @brief
 *
 * @tparam DType: input data type: half or bfloat16
 * @tparam IndexDType: index data type, int32
 * @tparam Format: only support  KV_CACHE_FORMAT_NHD or  KV_CACHE_FORMAT_HND
 * @param key: key, shape: [num_tokens, num_heads, head_size]
 * @param value: value, shape: [num_tokens, num_heads, head_size]
 * @param key_cache: key cache, int8,
 *        1. kv_cache_format == "NHD", shape: [num_blocks, block_size, num_kv_heads, head_size]
 *        2. kv_cache_format == "HND", shape: [num_blocks, num_kv_heads, block_size, head_size]
 * @param value_cache: value cache, int8,
 *        1. kv_cache_format == "NHD", shape: [num_blocks, block_size, num_kv_heads, head_size]
 *        2. kv_cache_format == "HND", shape: [num_blocks, num_kv_heads, block_size, head_size]
 * @param key_cache_scales: key cache scales, shape: [num_blocks, block_size]
 * @param value_cache_scales: value cache scales, shape: [num_blocks, block_size]
 * @param slot_mapping: The mapping position of the token in blocks.
 * @param key_stride: key.stride(0)
 * @param value_stride: value.stride(0)
 * @param key_cache_stride: key_cache.stride(0)
 * @param value_cache_stride: value_cache.stride(0)
 * @param num_tokens: the number of tokens
 * @param num_heads: the number of heads
 * @param head_size: head size
 * @param block_size: tokens of each page
 * @param x: usually x = 16 / sizeof(DType)
 * @param stream: CUDA Stream
 */
template<typename DType, typename IndexDType, kvCacheFormat Format>
void paged_attention_cache_appended_i8_kernel(
        const DType *key,
        const DType *value,
        int8_t *key_cache,
        int8_t *value_cache,
        DType *key_cache_scales,
        DType *value_cache_scales,
        const IndexDType *slot_mapping,
        unsigned key_stride,
        unsigned value_stride,
        unsigned key_cache_stride,
        unsigned value_cache_stride,
        unsigned num_tokens,
        unsigned num_heads,
        unsigned head_size,
        unsigned block_size,
        unsigned x,
        cudaStream_t stream);

/**
 * @brief
 *
 * @tparam DType: input data type: half or bfloat16
 * @tparam Format: KV_CACHE_FORMAT_STD or KV_CACHE_FORMAT_NHD or  KV_CACHE_FORMAT_HND
 * @param query: query, shape: [num_tokens, num_heads, head_size]
 * @param key_cache: key cache, half or bfloat16
 *        1. kv_cache_format == "STD", shape: [num_blocks, num_kv_heads, head_size // 8, block_size, 8]
 *        2. kv_cache_format == "NHD", shape: [num_blocks, block_size, num_kv_heads, head_size]
 *        3. kv_cache_format == "HND", shape: [num_blocks, num_kv_heads, block_size, head_size]
 * @param value_cache: value cache, half or bfloat16
 *        1. kv_cache_format == "STD", shape: [num_blocks, num_kv_heads, head_size // 8, block_size, 8]
 *        2. kv_cache_format == "NHD", shape: [num_blocks, block_size, num_kv_heads, head_size]
 *        3. kv_cache_format == "HND", shape: [num_blocks, num_kv_heads, block_size, head_size]
 * @param block_tables: bloack tables, is used to store block
 * @param seq_lens: shape: [num_tokens]
 * @param alibi_slopes: alibi slopes
 * @param out: output tensor, shape: [num_tokens, num_heads, head_size]
 * @param aux_md: aux_max_expsum, is used to store intermediate values
 * @param aux_o: aux_output, is used to store intermediate values
 * @param max_seq_len: max seq len in a batch
 * @param num_kv_heads: the number of kv heads
 * @param num_heads: the number of query heads
 * @param num_seqs: the number of seqs, num_seqs = query.size(0)
 * @param head_size: head size
 * @param block_size: tokens of each page
 * @param max_num_blocks_per_seq: (MAX_SEQ_LEN + block_size - 1) // block_size
 * @param q_stride: query.stride(0)
 * @param kv_block_stride: key_cache.stride(0)
 * @param scale: attention scale value
 * @param use_sqrt_alibi: whether to use sqrt alibi
 * @param stream: CUDA Stream
 */
template<typename DType, kvCacheFormat Format>
void paged_attention_f16_algo0_kernel(
        const DType *query,
        const DType *key_cache,
        const DType *value_cache,
        const int *block_tables,
        const int *seq_lens,
        const float *alibi_slopes,
        DType *out,
        float *aux_md,
        float *aux_o,
        unsigned max_seq_len,
        unsigned num_kv_heads,
        unsigned num_heads,
        unsigned num_seqs,
        unsigned head_size,
        unsigned block_size,
        unsigned max_num_blocks_per_seq,
        unsigned q_stride,
        unsigned kv_block_stride,
        float scale,
        bool use_sqrt_alibi,
        cudaStream_t stream);

/**
 * @brief
 *
 * @tparam DType: input data type: half or bfloat16
 * @tparam Format: KV_CACHE_FORMAT_STD or KV_CACHE_FORMAT_NHD or KV_CACHE_FORMAT_HND
 * @param query: query, shape: [num_tokens, num_heads, head_size]
 * @param key_cache: key cache, half or bfloat16
 *        1. kv_cache_format == "STD", shape: [num_blocks, num_kv_heads, head_size // 8, block_size, 8]
 *        2. kv_cache_format == "NHD", shape: [num_blocks, block_size, num_kv_heads, head_size]
 *        3. kv_cache_format == "HND", shape: [num_blocks, num_kv_heads, block_size, head_size]
 * @param value_cache: value cache, half or bfloat16
 *        1. kv_cache_format == "STD", shape: [num_blocks, num_kv_heads, head_size // 8, block_size, 8]
 *        2. kv_cache_format == "NHD", shape: [num_blocks, block_size, num_kv_heads, head_size]
 *        3. kv_cache_format == "HND", shape: [num_blocks, num_kv_heads, block_size, head_size]
 * @param block_tables: bloack tables, is used to store block
 * @param seq_lens: shape: [num_tokens]
 * @param alibi_slopes: alibi slopes
 * @param out: output tensor, shape: [num_tokens, num_heads, head_size]
 * @param aux_md: aux_max_expsum, is used to store intermediate values
 * @param aux_o: aux_output, is used to store intermediate values
 * @param max_seq_len: max seq len in a batch
 * @param num_kv_heads: the number of kv heads
 * @param num_heads: the number of query heads
 * @param num_seqs: the number of seqs, num_seqs = query.size(0)
 * @param head_size: head size
 * @param block_size: block size
 * @param max_num_blocks_per_seq: (MAX_SEQ_LEN + block_size - 1) // block_size
 * @param q_stride: query.stride(0)
 * @param kv_block_stride: key_cache.stride(0)
 * @param scale: attention scale value
 * @param use_sqrt_alibi: whether to use sqrt alibi
 * @param stream: CUDA Stream
 */
template<typename DType, kvCacheFormat Format>
void paged_attention_f16_algo1_kernel(
        const DType *query,
        const DType *key_cache,
        const DType *value_cache,
        const int *block_tables,
        const int *seq_lens,
        const float *alibi_slopes,
        DType *out,
        float *aux_md,
        float *aux_o,
        unsigned max_seq_len,
        unsigned num_kv_heads,
        unsigned num_heads,
        unsigned num_seqs,
        unsigned head_size,
        unsigned block_size,
        unsigned max_num_blocks_per_seq,
        unsigned q_stride,
        unsigned kv_block_stride,
        float scale,
        bool use_sqrt_alibi,
        cudaStream_t stream);

/**
 * @brief
 *
 * @tparam DType: input data type: half or bfloat16
 * @tparam Format: KV_CACHE_FORMAT_NHD or  KV_CACHE_FORMAT_HND
 * @param query: query, shape: [num_tokens, num_heads, head_size]
 * @param key_cache: key cache, int8,
 *        1. kv_cache_format == "NHD", shape: [num_blocks, block_size, num_kv_heads, head_size]
 *        2. kv_cache_format == "HND", shape: [num_blocks, num_kv_heads, block_size, head_size]
 * @param value_cache: value cache, int8,
 *        1. kv_cache_format == "NHD", shape: [num_blocks, block_size, num_kv_heads, head_size]
 *        2. kv_cache_format == "HND", shape: [num_blocks, num_kv_heads, block_size, head_size]
 * @param key_cache_scales: key cache scales, shape: [num_blocks, block_size]
 * @param value_cache_scales: value cache scales, shape: [num_blocks, block_size]
 * @param block_tables: bloack tables, is used to store block, shape:[num_seqs, max_num_blocks_per_seq]
 * @param seq_lens: shape: [num_tokens]
 * @param alibi_slopes: alibi slopes, shape:[num_heads]
 * @param out: output tensor, shape: [num_tokens, num_heads, head_size]
 * @param aux_md: aux_max_expsum, is used to store intermediate values
 * @param aux_o: aux_output, is used to store intermediate values
 * @param max_seq_len: max seq len in a batch
 * @param num_kv_heads: the number of kv heads
 * @param num_heads: the number of query heads
 * @param num_seqs: the number of seqs, num_seqs = query.size(0)
 * @param head_size: head size
 * @param block_size: tokens of each page
 * @param max_num_blocks_per_seq: (MAX_SEQ_LEN + block_size - 1) // block_size
 * @param q_stride: query.stride(0)
 * @param kv_block_stride: key_cache.stride(0)
 * @param scale: attention scale value
 * @param use_sqrt_alibi: whether to use sqrt alibi
 * @param stream: CUDA Stream
 */
template<typename DType, kvCacheFormat Format>
void paged_attention_i8_algo1_kernel(
        const DType *query,
        const int8_t *key_cache,
        const int8_t *value_cache,
        const DType *key_cache_scales,
        const DType *value_cache_scales,
        const int *block_tables,
        const int *seq_lens,
        const float *alibi_slopes,
        DType *out,
        float *aux_md,
        float *aux_o,
        unsigned max_seq_len,
        unsigned num_kv_heads,
        unsigned num_heads,
        unsigned num_seqs,
        unsigned head_size,
        unsigned block_size,
        unsigned max_num_blocks_per_seq,
        unsigned q_stride,
        unsigned kv_block_stride,
        float scale,
        bool use_sqrt_alibi,
        cudaStream_t stream);

/**
 * @brief
 *
 * @tparam DType: input data type: half or bfloat16
 * @tparam Format: only support KV_CACHE_FORMAT_NHD or  KV_CACHE_FORMAT_HND
 * @param query: query, shape: [num_tokens, num_heads, head_size]
 * @param paged_k_data: key_cache,
 *        1. kv_cache_format == "NHD", shape: [num_blocks, block_size, num_kv_heads, head_size]
 *        2. kv_cache_format == "HND", shape: [num_blocks, num_kv_heads, block_size, head_size]
* @param paged_v_data: value_cache,
 *        1. kv_cache_format == "NHD", shape: [num_blocks, block_size, num_kv_heads, head_size]
 *        2. kv_cache_format == "HND", shape: [num_blocks, num_kv_heads, block_size, head_size]
 * @param paged_kv_indptr: the number of kv blocks in per token. shape: [num_tokens]
 * @param paged_kv_indices: kv block index. shape: [num_blocks]
 * @param paged_kv_last_page_len: shape: [num_tokens]
 * @param alibi_slopes: alibi slopes, shape:[num_heads]
 * @param out: output tensor, shape: [num_tokens, num_heads, head_size]
 * @param aux_md: aux_max_expsum, is used to store intermediate values
 * @param aux_o: aux_output, is used to store intermediate values
 * @param max_seq_len: max seq len
 * @param num_kv_heads: the number of kv heads
 * @param num_qo_heads: the number of query heads
 * @param num_seqs: the number of seqs
 * @param head_size: head size
 * @param page_size: block size
 * @param q_stride: query.stride(0)
 * @param kv_block_stride: kv_block.stride(0)
 * @param scale: attention scale value
 * @param use_sqrt_alibi: whether to use sqrt alibi
 * @param stream: CUDA Stream
 */
template<typename DType, kvCacheFormat Format>
void paged_attention_flashinfer_f16_kernel(const DType *query,
                                           const DType *paged_k_data,
                                           const DType *paged_v_data,
                                           const int32_t *paged_kv_indptr,
                                           const int32_t *paged_kv_indices,
                                           const int32_t *paged_kv_last_page_len,
                                           const float *alibi_slopes,
                                           DType *out,
                                           float *aux_md,
                                           float *aux_o,
                                           int32_t max_seq_len,
                                           unsigned num_kv_heads,
                                           unsigned num_qo_heads,
                                           unsigned num_seqs,
                                           unsigned head_size,
                                           unsigned page_size,
                                           unsigned q_stride,
                                           unsigned kv_block_stride,
                                           float scale,
                                           bool use_sqrt_alibi,
                                           cudaStream_t stream);




// ========================================================
// MOE
// ========================================================

/**
 * @brief
 *
 * @tparam T1: input type, half or bfloat16
 * @tparam T2: output type,  half or bfloat16
 * @param A: The input tensor representing tokens  with shape (num_tokens, K),
 *        where K is the feature dimension of each token. shape: [num_tokens, K]
 * @param align_A: The input tensor representing tokens post padding with shape (pad_m, K),
 *        where pad_m is the total number of tokens post padding and K is the feature dimension of each token.
 * @param B: The stacked MOE weight tensor with shape (E, N, K),
 *        where E is the number of experts, K is the input feature dimension, and N is the output feature dimension.
 * @param C: The output cache tensor with shape (M, topk, N), where M is the total number of tokens post padding,
 *        topk is the number of times each token is repeated, and N is the output feature dimension.
 * @param topk_weight: topk weight, shape: [num_tokens, topk]
 * @param topk_ids: topk index, shape: [num_tokens, topk]
 * @param sorted_token_ids: The tensor containing the sorted indices of tokens,
 *        repeated topk times and arranged by the expert index they are assigned to.
 *        shape: [topk_ids.numel() + num_experts * (block_size - 1)]
 * @param expert_ids:The tensor containing the indices of the expert for each block.
 *        It determines which expert matrix from B should be used for each block in A.
 *        shape: [topk_ids.numel() + num_experts]
 * @param m: the total number of tokens
 * @param pad_m: the total number of tokens post padding
 * @param n: B.size(1), the output feature dimension
 * @param k: B.size(2), the feature dimension of each token
 * @param top_k: topk
 * @param block_size_m: BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix multiplication
 *        across different blocks processed by the same expert.
 * @param mul_routed_weight: Whether to apply route weight
 * @param stream: CUDA Stream
 */
template<typename T1, typename T2>
void fused_moe(const T1 *A, T1 *align_A, const T1 *B, T1 *C,
               const float *topk_weight, const int32_t *topk_ids, const int32_t *sorted_token_ids,
               const int32_t *expert_ids, unsigned m, unsigned pad_m, unsigned n, unsigned k,
               unsigned top_k, unsigned block_size_m,
               bool mul_routed_weight, cudaStream_t stream);

/**
 * @brief
 *
 * @tparam T: input type, int8
 * @param A: The input tensor representing tokens  with shape (num_tokens, K),
 *        where K is the feature dimension of each token
 * @param align_A: The input tensor representing tokens post padding with shape (pad_m, K),
 *        where pad_m is the total number of tokens post padding and K is the feature dimension of each token.
 * @param B: The stacked MOE weight tensor with shape (E, N, K), where E is the number of experts,
 *        K is the input feature dimension, and N is the output feature dimension. shape: [E, N, K]
 * @param C: The output cache tensor with shape (M, topk, N), where M is the total number of tokens post padding,
 *        topk is the number of times each token is repeated, and N is the output feature dimension.
 * @param topk_weight: topk weight, shape: [num_tokens, topk]
 * @param topk_ids: topk index, shape: [num_tokens, topk]
 * @param sorted_token_ids: The tensor containing the sorted indices of tokens,
 *        repeated topk times and arranged by the expert index they are assigned to.
 *        shape: [topk_ids.numel() + num_experts * (block_size - 1)]
 * @param expert_ids: The tensor containing the indices of the expert for each block.
 *        It determines which expert matrix from B should be used for each block in A.
 *        shape: [topk_ids.numel() + num_experts]
 * @param w_scale: B scale
 * @param a_scale: A scale
 * @param persistent:  persistent
 * @param expert_num: the number of experts
 * @param m: the total number of tokens
 * @param pad_m: the total number of tokens post padding
 * @param n: B.size(1), the output feature dimension
 * @param k: B.size(2), the feature dimension of each token
 * @param top_k: topk
 * @param block_size_m: BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
 *        multiplication across different blocks processed by the same expert.
 * @param mul_routed_weight: Whether to apply route weight
 * @param input_extend: Determine whether the input tensor needs to be extended
 * @param stream: CUDA Stream
 * @param cuinfer_handle: CUINFER HANDLE
 */
template<typename T>
void fused_moe_ixinfer(const int8_t *A, int8_t *align_A, const int8_t *B, T *C,
                       const float *topk_weight, const int32_t *topk_ids,
                       const int32_t *sorted_token_ids, const int32_t *expert_ids,
                       const float *w_scale, const float *a_scale, int64_t persistent, unsigned expert_num,
                       unsigned m, unsigned pad_m, unsigned n, unsigned k, unsigned top_k, unsigned block_size_m,
                       bool mul_routed_weight, bool input_extend, cudaStream_t stream, cuinferHandle_t cuinfer_handle);

/**
 * @brief
 *
 * @tparam T: input type, float
 * @param gating_output: input tensor, shape: [num_tokens, num_experts]
 * @param topk_weights: topk weights, shape: [num_tokens, topk]
 * @param topk_indices: topk indices, shape: [num_tokens, topk]
 * @param token_expert_indices: expert indices, shape: [num_tokens, topk]
 * @param softmax_workspace: softmax workspace
 * @param num_tokens: the number of tokens
 * @param num_experts: the number of experts
 * @param topk: topk
 * @param renormalize: whether renormalize the result
 * @param stream: CUDA Stream
 */
template<typename T>
void moe_topk_softmax(
        const T *gating_output,
        T *topk_weights,
        int *topk_indices,
        int *token_expert_indices,
        T *softmax_workspace,
        int num_tokens,
        int num_experts,
        int topk,
        bool renormalize,
        cudaStream_t stream);

/**
 * @brief
 *
 * @tparam IN_DTYPE: gating_output type, half or bfloat16
 * @tparam INDEX_DTYPE: topk_indices type, int32 or int64
 * @param topk_weights: topk weights, shape: [num_tokens, topk]
 * @param topk_indices: topk indices, shape: [num_tokens, topk]
 * @param gating_output: input tensor, shape: [num_tokens, num_experts]
 * @param bias: bias tensor for grouped topk, shape: [num_experts]
 * @param num_tokens: the number of tokens
 * @param num_experts: the number of experts
 * @param topk: topk
 * @param num_expert_group: num_expert_group
 * @param topk_group: topk_group
 * @param renormalize: whether renormalize the result
 * @param scoring_func: scoring function for grouped topk
 * @param stream: CUDA Stream
 */
template<typename IN_DTYPE, typename INDEX_DTYPE>
void moe_grouped_topk(
        float *topk_weights,
        INDEX_DTYPE *topk_indices,
        const IN_DTYPE *gating_output,
        const IN_DTYPE *bias,
        int num_tokens, int num_experts, int topk,
        int num_expert_group, int topk_group, bool renormalize,
        std::string scoring_func,
        cudaStream_t stream);

/**
 * @brief
 *
 * @tparam T: input type, int32_t
 * @param topk_ids: topk index, shape: [num_tokens, topk]
 * @param sorted_token_ids: The tensor containing the sorted indices of tokens,
 *        repeated topk times and arranged by the expert index they are assigned to.
 *        shape: [topk_ids.numel() + num_experts * (block_size - 1)]
 * @param expert_ids: The tensor containing the indices of the expert for each block.
 *        It determines which expert matrix from B should be used for each block in A.
 *        shape: [topk_ids.numel() + num_experts]
 * @param total_tokens_post_pad: the number of tokens
 * @param aux_tokens_cnts: used for large num_experts
 * @param aux_cumsum: used for large num_experts
 * @param num_experts: the number of experts
 * @param block_size: tokens of each page
 * @param numel: topk_ids.numel()
 * @param stream: CUDA Stream
 */
template<typename T>
void moe_align_block_size(const T *topk_ids,
                          int32_t *sorted_token_ids,
                          int32_t *expert_ids,
                          int32_t *total_tokens_post_pad,
                          int32_t *aux_tokens_cnts,
                          int32_t *aux_cumsum,
                          int32_t num_experts,
                          int32_t block_size,
                          size_t numel,
                          cudaStream_t stream);

/**
 * @brief
 *
 * @tparam T: input type, half or bfloat16
 * @param input: input tensor, shape: [outer_size, reduce_size, inner_size]
 * @param mul_weights: broadcast mul before reduce sum, tensor shape: [outer_size, reduce_size]
 * @param mask: control the validity of each vector, tensor shape: [outer_size, reduce_size]
 * @param extra_residual: add on the final output, tensor shape: [outer_size, inner_size]
 * @param out: output tensor, shape: [outer_size, inner_size]
 * @param outer_size: outer_size
 * @param reduce_size: reduce_size
 * @param inner_size: inner_size
 * @param in_stride: input.stride(1)
 * @param out_stride: out.stride(0)
 * @param scaling_factor: scaling factor for the output before residual
 * @param stream: CUDA Stream
 */
template<typename T>
void moe_output_reduce_sum(
        const T *input,
        const float *mul_weights,
        const bool *mask,
        const T *extra_residual,
        T *out,
        unsigned outer_size,
        unsigned reduce_size,
        unsigned inner_size,
        unsigned in_stride,
        unsigned out_stride,
        float scaling_factor,
        cudaStream_t stream);

/**
 * @brief
 *
 * @tparam T: input type, half or bfloat16
 * @tparam SCALE_T: smooth scales type, float32 or same as input
 * @param input: input tensor, shape: [num_tokens, hidden_size]
 * @param topk_ids: expert id for each tokens, shape: [num_tokens, topk]
 * @param smooth_scales: smooth quant scales tensor for each experts, shape: [num_experts, hidden_size]
 * @param dst_to_src: index of dst to src, shape: [num_tokens * topk]
 * @param src_to_dst: index of src to dst, e.g.  src_tensor[i] = dst_tensor[src_to_dst[i]]. shape: [num_tokens * topk]
 * @param i8_outputs: output tensor shape: [dst_tokens, hidden_size]
 * @param output_scales: scales tensor for output, shape: [dst_tokens]
 * @param num_tokens: number tokens of input
 * @param dst_tokens: the number of tokens after expansion
 * @param hidden_size: hidden_size
 * @param topk: topk for moe
 * @param output_format: setting output format
 * @param stream: CUDA Stream
 */
template<typename T, typename SCALE_T>
void moe_expand_input_dynamic_scaled_int8(const T *input, const int32_t *topk_ids, const SCALE_T *smooth_scales,
                                          const int32_t *dst_to_src, const int32_t *src_to_dst,
                                          int8_t *i8_outputs, float *output_scales, int num_tokens,
                                          int dst_tokens, int hidden_size, int topk, int output_format, cudaStream_t stream);

/**
 * @brief
 *
 * @tparam T: input and output type, half or bfloat16
 * @param input: input tensor, shape: [num_tokens, hidden_size]
 * @param dst_to_src: index of dst to src, shape: [num_tokens * topk]
 * @param src_to_dst: index of src to dst, e.g.  src_tensor[i] = dst_tensor[src_to_dst[i]]. shape: [num_tokens * topk]
 * @param output: output tensor shape: [dst_tokens, hidden_size]
 * @param num_tokens: number tokens of input
 * @param dst_tokens: the number of tokens after expansion
 * @param hidden_size: hidden_size
 * @param topk: topk for moe
 * @param stream: CUDA Stream
 */
template<typename T>
void moe_expand_input(const T *input, const int32_t *dst_to_src, const int32_t *src_to_dst, T *output,
                      int num_tokens, int dst_tokens, int hidden_size, int topk, cudaStream_t stream);

/**
 * @brief
 *
 * @param topk_ids: expert id for each tokens, shape: [num_tokens, topk]
 * @param src_dst: index of src to dst, e.g.  src_tensor[i] = dst_tensor[src_to_dst[i]]. shape: [num_tokens * topk]
 * @param dst_src: index of dst to src, shape: [num_tokens * topk]
 * @param expert_sizes: the number of tokens allocated to each expert, shape: [num_experts]
 * @param expand_tokens: the sum of expert_sizes, shape: [1]
 * @param aux_tokens_cnts: used for large num_experts
 * @param aux_cumsum: used for large num_experts
 * @param num_experts: the numbers of num_experts overall
 * @param start_expert_id: start expert id of the vaild expert interval
 * @param end_expert_id: end expert id of the vaild expert interval [start_expert_id, end_expert_id)
 * @param numel: size of topk_ids, the numbers of tokens
 * @param stream: CUDA Stream
 */
void moe_compute_token_index(
        int32_t *topk_ids,
        int32_t *src_dst,
        int32_t *dst_src,
        int32_t *expert_sizes,
        int32_t *expand_tokens,
        int32_t *aux_tokens_cnts,
        int32_t *aux_cumsum,
        int32_t num_experts,
        int32_t start_expert_id,
        int32_t end_expert_id,
        size_t numel,
        cudaStream_t stream);

/**
 * @brief
 *
 * @tparam T: input type, half or bfloat16
 * @tparam SCALE_T: smooth scales type, float32 or same as input
 * @tparam BIAS_T: bias type, float32 or same as input
 * @param input: input tensor, shape: [num_tokens, hidden_size]
 * @param bias: bias tensor, shape: [num_experts, hidden_size]
 * @param smooth_scales: smooth quant scales tensor for each experts, shape: [num_experts, hidden_size // 2] if act_type==swiglu else [num_experts, hidden_size]
 * @param dst_to_src: index of dst to src, shape: [num_tokens * topk]
 * @param topk_ids: expert id for each tokens, shape: [num_tokens]
 * @param out: output tensor, shape: [num_tokens, hidden_size // 2] if act_type==swiglu else [num_tokens, hidden_size]
 * @param output_scales: scales tensor for output, shape: [num_tokens]
 * @param act_type: str activation type. Options include gelu, silu, and swiglu.
 * @param num_tokens: number tokens of input
 * @param hidden_size: hidden_size
 * @param output_format: setting output format
 * @param stream: CUDA Stream
 */
template<typename T, typename SCALE_T, typename BIAS_T>
void activation_dynamic_scaled_int8(
        const T *input, const BIAS_T *bias,
        const SCALE_T *smooth_scales, const int32_t *dst_to_src,
        const int32_t *topk_ids, int8_t *out, float *output_scales,
        std::string act_type, unsigned num_tokens, unsigned hidden_size, int output_format, cudaStream_t stream);




// ========================================================
// Dynamic INT8
// ========================================================
/**
 * @brief
 *
 * @tparam T: input type, half or bfloat16
 * @tparam ST: smooth_scales type, same as T or be float32
 * @param input: input tensor, shape: [num_token, hidden_size]
 * @param smooth_scales: input smooth scale tensor, shape: [hidden_size]
 * @param out: output tensor, shape: [num_token, hidden_size]
 * @param scale_output: output scale tensor, shape: [num_token]
 * @param num_tokens: the number of tokens
 * @param hidden_size: HiddenSize
 * @param stream: CUDA Stream
 */
template<typename T, typename ST>
void dynamic_scaled_quant_smoothquant(const T *input, const ST *smooth_scales, int8_t *out, float *scale_output,
                                      int num_tokens, int hidden_size, cudaStream_t stream);

/**
 * @brief
 *
 * @tparam T: input type, half or bfloat16
 * @param input: input tensor, shape: [num_tokens, hidden_size]
 * @param smooth_scales: input smooth scale tensor, shape: [hidden_size]
 * @param out: output tensor, shape: [num_tokens, hidden_size]
 * @param scale_output: output scale tensor, shape: [num_tokens]
 * @param num_tokens: the number of tokens
 * @param hidden_size: HiddenSize
 * @param stream: CUDA Stream
 */
template<typename T>
void silu_and_mul_smoothquant(const T *input, const T *smooth_scales, int8_t *out, float *scale_output,
                              int num_tokens, const int hidden_size, cudaStream_t stream);

/**
 * @brief
 *
 * @tparam T: input type, half or bfloat16
 * @param input: input tensor, shape: [num_tokens, hidden_size]
 * @param weight: weight, shape: [hidden_size]
 * @param fused_bias: fused_bias[optional], shape: [hidden_size]
 * @param smooth_scales: input smooth scale tensor, shape: [hidden_size]
 * @param out: output tensor, shape: [num_tokens, hidden_size]
 * @param scale_output: output scale tensor, shape: [num_tokens]
 * @param num_tokens: the number of tokens
 * @param hidden_size: HiddenSize
 * @param in_stride: int, the stride of dim "HiddenSize" to support non-contiguous input
 * @param eps: float, a value added to the denominator for numerical stability
 * @param stream: CUDA Stream
 */
template<typename T, typename ST>
void rmsnorm_smoothquant(const T *input, const T *weight,
                         const T *fused_bias, const ST *smooth_scales,
                         int8_t *out, float *scale_output,
                         int num_tokens, int hidden_size, int in_stride,
                         float eps, cudaStream_t stream);

/**
 * @brief
 *
 * @tparam T: input type, half or bfloat16
 * @param input: input tensor, shape: [num_tokens, hidden_size]
 * @param weight: weight, shape: [hidden_size]
 * @param fused_bias: fused_bias[optional], shape: [hidden_size]
 * @param out: output tensor, shape: [num_tokens, hidden_size]
 * @param scale_output: output scale tensor, shape: [num_tokens]
 * @param num_tokens: the number of tokens
 * @param hidden_size: HiddenSize
 * @param in_stride: int, the stride of dim "HiddenSize" to support non-contiguous input
 * @param eps: float, a value added to the denominator for numerical stability
 * @param stream: CUDA Stream
 */
template<typename T>
void rmsnorm_dynamic_int8(const T *input, const T *weight, const T *fused_bias,
                          int8_t *out, float *scale_output,
                          int num_tokens, int hidden_size, int in_stride,
                          float eps, cudaStream_t stream);

/**
 * @brief
 *
 * @tparam T: input type, half or bfloat16
 * @tparam ST: smooth_scales type, same as T or be float32
 * @tparam IS_POST: norm type, post or pre
 * @param input: input tensor, shape: [num_tokens, hidden_size]
 * @param residual: residual tensor, shape: [num_tokens, hidden_size]
 * @param weight: weight, shape: [hidden_size]
 * @param fused_bias: fused_bias[optional], shape: [hidden_size]
 * @param smooth_scales: input smooth scale tensor, shape: [hidden_size]
 * @param out: output tensor, shape: [num_tokens, hidden_size]
 * @param scale_output: output scale tensor, shape: [num_tokens]
 * @param residual_output: residual_output[optional], shape: [num_tokens, hidden_size]
 * @param num_tokens: the number of tokens
 * @param hidden_size: HiddenSize
 * @param in_stride: int, the stride of dim "HiddenSize" to support non-contiguous input
 * @param resi_stride: int, the stride of dim "HiddenSize" to support non-contiguous residual
 * @param eps: float, a value added to the denominator for numerical stability
 * @param stream: CUDA Stream
 */
template<typename T, typename ST, bool IS_POST>
void residual_rmsnorm_smoothquant(const T *input, T *residual, const T *weight,
                                  const T *fused_bias, const ST *smooth_scales,
                                  int8_t *out, float *scale_output, T *residual_output,
                                  int num_tokens, int hidden_size, int in_stride, int resi_stride,
                                  float eps, cudaStream_t stream);

/**
 * @brief
 *
 * @tparam T: input type, half or bfloat16
 * @tparam IS_POST: norm type, post or pre
 * @param input: input tensor, shape: [num_tokens, hidden_size]
 * @param residual: residual tensor, shape: [num_tokens, hidden_size]
 * @param weight: weight, shape: [hidden_size]
 * @param fused_bias: fused_bias[optional], shape: [hidden_size]
 * @param out: output tensor, shape: [num_tokens, hidden_size]
 * @param scale_output: output scale tensor, shape: [num_tokens]
 * @param residual_output: residual_output[optional], shape: [num_tokens, hidden_size]
 * @param num_tokens: the number of tokens
 * @param hidden_size: HiddenSize
 * @param in_stride: int, the stride of dim "HiddenSize" to support non-contiguous input
 * @param resi_stride: int, the stride of dim "HiddenSize" to support non-contiguous residual
 * @param eps: float, a value added to the denominator for numerical stability
 * @param stream: CUDA Stream
 */
template<typename T, bool IS_POST>
void residual_rmsnorm_dynamic_int8(const T *input, T *residual, const T *weight, const T *fused_bias,
                                   int8_t *out, float *scale_output, T *residual_output,
                                   int num_tokens, int hidden_size, int in_stride, int resi_stride,
                                   float eps, cudaStream_t stream);

/**
 * @brief
 *
 * @tparam T: input type, half or bfloat16
 * @tparam ST: smooth_scales type, same as T or be float32
 * @param input: input tensor, shape: [num_tokens, hidden_size]
 * @param scale: weight, shape: [hidden_size]
 * @param bias: bias, shape: [hidden_size]
 * @param fused_bias: fused_bias[optional], shape: [hidden_size]
 * @param smooth_scales: input smooth scale tensor, shape: [hidden_size]
 * @param out: output tensor, shape: [num_tokens, hidden_size]
 * @param scale_output: output scale tensor, shape: [num_tokens]
 * @param num_tokens: the number of tokens
 * @param hidden_size: HiddenSize
 * @param in_stride: int, the stride of dim "HiddenSize" to support non-contiguous input
 * @param eps: float, a value added to the denominator for numerical stability
 * @param stream: CUDA Stream
 */
template<typename T, typename ST>
void layernorm_smoothquant(const T *input, const T *scale, const T *bias,
                           const T *fused_bias, const ST *smooth_scales,
                           int8_t *out, float *scale_output,
                           int num_tokens, int hidden_size, int in_stride,
                           float eps, cudaStream_t stream);

/**
 * @brief
 *
 * @tparam T: input type, half or bfloat16
 * @param input: input tensor, shape: [num_tokens, hidden_size]
 * @param scale: weight, shape: [hidden_size]
 * @param bias: bias, shape: [hidden_size]
 * @param fused_bias: fused_bias[optional], shape: [hidden_size]
 * @param out: output tensor, shape: [num_tokens, hidden_size]
 * @param scale_output: output scale tensor, shape: [num_tokens]
 * @param num_tokens: the number of tokens
 * @param hidden_size: HiddenSize
 * @param in_stride: int, the stride of dim "HiddenSize" to support non-contiguous input
 * @param eps: float, a value added to the denominator for numerical stability
 * @param stream: CUDA Stream
 */
template<typename T>
void layernorm_dynamic_int8(const T *input, const T *scale, const T *bias, const T *fused_bias,
                            int8_t *out, float *scale_output,
                            int num_tokens, int hidden_size, int in_stride,
                            float eps, cudaStream_t stream);

/**
 * @brief
 *
 * @tparam T: input type, half or bfloat16
 * @tparam ST: smooth_scales type, same as T or be float32
 * @param input: input tensor, shape: [num_tokens, hidden_size]
 * @param residual: residual tensor, shape: [num_tokens, hidden_size]
 * @param scale: weight, shape: [hidden_size]
 * @param bias: bias, shape: [hidden_size]
 * @param fused_bias: fused_bias[optional], shape: [hidden_size]
 * @param smooth_scales: input smooth scale tensor, shape: [hidden_size]
 * @param out: output tensor, shape: [num_tokens, hidden_size]
 * @param scale_output: output scale tensor, shape: [num_tokens]
 * @param residual_output: residual_output[optional], shape: [num_tokens, hidden_size]
 * @param num_tokens: the number of tokens
 * @param hidden_size: HiddenSize
 * @param in_stride: int, the stride of dim "HiddenSize" to support non-contiguous input
 * @param resi_stride: int, the stride of dim "HiddenSize" to support non-contiguous residual
 * @param eps: float, a value added to the denominator for numerical stability
 * @param stream: CUDA Stream
 */
template<typename T, typename ST>
void residual_layernorm_smoothquant(const T *input, T *residual,
                                    const T *scale, const T *bias,
                                    const T *fused_bias, const ST *smooth_scales,
                                    int8_t *out, float *scale_output, T *residual_output,
                                    int num_tokens, int hidden_size,
                                    int in_stride, int resi_stride,
                                    float eps, cudaStream_t stream);

/**
 * @brief
 *
 * @tparam T: input type, half or bfloat16
 * @param input: input tensor, shape: [num_tokens, hidden_size]
 * @param residual: residual tensor, shape: [num_tokens, hidden_size]
 * @param scale: weight, shape: [hidden_size]
 * @param bias: bias, shape: [hidden_size]
 * @param fused_bias: fused_bias[optional], shape: [hidden_size]
 * @param out: output tensor, shape: [num_tokens, hidden_size]
 * @param scale_output: output scale tensor, shape: [num_tokens]
 * @param residual_output: residual_output[optional], shape: [num_tokens, hidden_size]
 * @param num_tokens: the number of tokens
 * @param hidden_size: HiddenSize
 * @param in_stride: int, the stride of dim "HiddenSize" to support non-contiguous input
 * @param resi_stride: int, the stride of dim "HiddenSize" to support non-contiguous residual
 * @param eps: float, a value added to the denominator for numerical stability
 * @param stream: CUDA Stream
 */
template<typename T>
void residual_layernorm_dynamic_int8(const T *input, T *residual,
                                     const T *scale, const T *bias, const T *fused_bias,
                                     int8_t *out, float *scale_output, T *residual_output,
                                     int num_tokens, int hidden_size,
                                     int in_stride, int resi_stride,
                                     float eps, cudaStream_t stream);

/**
 * @brief
 *
 * @tparam T: input type, half or bfloat16
 * @param input: input tensor, shape: [num_tokens, hidden_size]
 * @param scale1: weight, shape: [hidden_size]
 * @param bias1: bias, shape: [hidden_size]
 * @param smooth_scales1: input smooth scale tensor, shape: [hidden_size]
 * @param scale2: weight, shape: [hidden_size]
 * @param bias2: bias, shape: [hidden_size]
 * @param smooth_scales2: input smooth scale tensor, shape: [hidden_size]
 * @param output1: output tensor, shape: [num_tokens, hidden_size]
 * @param scale_output1: output scale tensor, shape: [num_tokens]
 * @param output2: output tensor, shape: [num_tokens, hidden_size]
 * @param scale_output2: output scale tensor, shape: [num_tokens]
 * @param num_tokens: the number of tokens
 * @param hidden_size: HiddenSize
 * @param eps: float, a value added to the denominator for numerical stability
 * @param stream: CUDA Stream
 */
template<typename T>
void layernorm_2sb_smoothquant(const T *input,
                               const T *scale1, const T *bias1, const T *smooth_scales1,
                               const T *scale2, const T *bias2, const T *smooth_scales2,
                               int8_t *output1, float *scale_output1,
                               int8_t *output2, float *scale_output2,
                               int num_tokens, int hidden_size,
                               float eps, cudaStream_t stream);

/**
 * @brief
 *
 * @tparam T
 * @param input: input tensor, shape: [num_tokens, hidden_size]
 * @param residual: residual tensor, shape: [num_tokens, hidden_size]
 * @param scale1: weight, shape: [hidden_size]
 * @param bias1: bias, shape: [hidden_size]
 * @param smooth_scales1: input smooth scale tensor, shape: [hidden_size]
 * @param scale2: weight, shape: [hidden_size]
 * @param bias2: bias, shape: [hidden_size]
 * @param smooth_scales2: input smooth scale tensor, shape: [hidden_size]
 * @param output1: output tensor, shape: [num_tokens, hidden_size]
 * @param scale_output1: output scale tensor, shape: [num_tokens]
 * @param output2: output tensor, shape: [num_tokens, hidden_size]
 * @param scale_output2: output scale tensor, shape: [num_tokens]
 * @param num_tokens: the number of tokens
 * @param hidden_size: HiddenSize
 * @param eps: float, a value added to the denominator for numerical stability
 * @param stream: CUDA Stream
 */
template<typename T>
void layernorm_2sb_residual_smoothquant(const T *input, T *residual,
                                        const T *scale1, const T *bias1, const T *smooth_scales1,
                                        const T *scale2, const T *bias2, const T *smooth_scales2,
                                        int8_t *output1, float *scale_output1,
                                        int8_t *output2, float *scale_output2,
                                        int num_tokens, int hidden_size,
                                        float eps, cudaStream_t stream);




// ========================================================
// Lightllm
// ========================================================

/**
 * @brief
 *
 * @tparam T: input type, float
 * @param logits: apply_penalty input, shape: [batch, vocab_size]
 * @param presence_penalty: Penalty term that controls whether the word exists. shape: [batch]
 * @param freqency_penalty: Used to control the overall frequency of words in the generated text. shape: [batch]
 * @param p_token_ids: The id corresponding to per token in the vocabulary，shape: [num_tokens]
 * @param p_token_counts: The counts corresponding to per token. shape: [num_tokens]
 * @param p_cumsum_seq_len: The cumulative value of seq_len in a batch. shape: [batch+1]
 * @param p_max_len_in_batch: The maximum length of seq in a batch
 * @param batch: Batch Size
 * @param vocab_size: vocabulary size
 * @param stream: CUDA Stream
 */
template<typename T>
void lightllm_apply_penalty(T *logits, const T *presence_penalty, const T *freqency_penalty,
                            const int *p_token_ids, const int *p_token_counts,
                            const int *p_cumsum_seq_len, int p_max_len_in_batch,
                            int batch, int vocab_size, cudaStream_t stream);


/**
 * @brief
 *
 * @tparam T: input type, half or bfloat16
 * @param key_cache: key cache, shape: [num_tokens, num_kv_heads, head_size]
 * @param b_mem_idx: Index of the destination location corresponding to the token. shape: [num_tokens].
 * @param out: output tensor, shape: [max_tokens, num_kv_heads, head_size]
 * @param num_tokens: the number of tokens.
 * @param num_heads: num_kv_heads.
 * @param headdim: head_size
 * @param stream: CUDA Stream
 */
template<typename T>
void lightllm_destindex_copy_kv(
        const T *key_cache,
        const int *b_mem_idx,
        T *out,
        int num_tokens,
        int num_heads,
        int headdim,
        cudaStream_t stream);

/**
 * @brief
 *
 * @tparam T: input type, half
 * @param input: input tensor, shape: [num_tokens, head_num, head_dim]
 * @param cos: shape: [num_tokens, 1, head_dim // 2 // 2]
 * @param sin: shape: [num_tokens, 1, head_dim //2 //2]
 * @param num_tokens: the number of tokens.
 * @param head_num: the number of head.
 * @param head_dim: head_size
 * @param rot_dim: rot_dim = cos.size(-1)
 * @param stream: CUDA Stream
 */

template<typename T>
void lightllm_glm2_rope(T *input, const T *cos, const T *sin, int num_tokens,
                        int head_num, int head_dim, int rot_dim, cudaStream_t stream);
/**
 * @brief
 *
 * @tparam T: input type, half or bfloat16
 * @param out: output tensor, shape: [batch, head_num, head_dim]
 * @param partition_size: partition size
 * @param exp_sums: shape: [batch, num_heads, max_num_partitions]
 * @param max_logits: shape: [batch, num_heads, max_num_partitions]
 * @param tmp_out: shape: [batch, num_heads, max_num_partitions,head_size]
 * @param query: shape: [batch,head_num,head_dim]
 * @param key_cache: key cache. shape: [max_num_tokens, head_num_kv, head_dim]
 * @param value_cache: value cache. shape: [max_num_tokens, head_num_kv, head_dim]
 * @param scale:The scaling of QK^T before applying softmax.
 * @param reg_to_tokens: shape: [max_requset,max_tokens]
 * @param b_req_idx: request index in a batch, shape: [batch]
 * @param b_seq_len: seq len in a batch. shape: [batch]
 * @param q_stride: query.stride(0)
 * @param kv_token_stride: key_cache.stride(0)
 * @param kv_head_stride: key_cache.stride(1)
 * @param max_context_len_cur_batch: b_seq_len.max()
 * @param num_heads: the number of query head.
 * @param num_kv_head: the number of kv head.
 * @param batch: batch size
 * @param stream:: CUDA Stream
 */
template<typename T>
void lightllm_token_attention(
        T *out, int64_t partition_size, float *exp_sums,
        float *max_logits, T *tmp_out,
        const T *query,
        const T *key_cache,
        const T *value_cache,
        float scale,
        const int *reg_to_tokens,
        const int *b_req_idx,
        const int *b_seq_len,
        int q_stride,
        int kv_token_stride,
        int kv_head_stride,
        int max_context_len_cur_batch, int num_heads, int num_kv_head,
        int batch, cudaStream_t stream);




// ========================================================
// Quant
// ========================================================

typedef enum {
    QUANT_AWQ,
    QUANT_GPTQ,
    QUANT_INT8,
    QUANT_NF4,
    QUANT_FP4
} QuantType;

/**
 * @brief
 *
 * @tparam T: input type, half or bfloat16
 * @param input: input tensor, shape: [m, k]
 * @param i8_input: int8 quant output, shape: [m, k]
 * @param input_scales: scale
 * @param is_dynamic: whether to use dynamic scale
 * @param input_channel: input row
 * @param output_channel: i8_input col
 * @param stream:CUDA Stream
 */
template<typename T>
void scaled_int8_quant(const T *input, int8_t *i8_input, float *input_scales, bool is_dynamic, int input_channel, int output_channel,
                       cudaStream_t stream);

/**
 * @brief
 *
 * @tparam T1: output type, half or bfloat16
 * @tparam T2: output type after pack 4B, half2 or bfloat162
 * @tparam TYPE: QUANT_NF4 or QUANT_FP4
 * @param qweights: quant weight, uint8, shape: [output_channel, input_channel // 2]
 * @param scales: scale, float, shape: [output_channel * input_channel // g]
 * @param out: output tensor, shape: [input_channel, output_channel]
 * @param input_channel: row
 * @param output_channel: column
 * @param group_size: group size
 * @param stream: CUDA Stream
 */
template<typename T1, typename T2, QuantType TYPE>
void weight_dequant_float4(const unsigned char *qweights, const float *scales, T1 *out,
                           unsigned input_channel, unsigned output_channel, unsigned group_size, cudaStream_t stream);
/**
 * @brief
 *
 * @tparam T: output type, half or bfloat16
 * @param qweights: quant weight, int32, shape: [input_channel // (32 / bits), output_channel]
 * @param scales: scale, half or bfloat16, shape: [input_channel // g, output_channel]
 * @param zeros: quant zeros, int32, shape: [input_channel // g, output_channel // (32 / bits)]
 * @param g_idx: g_idx
 * @param out: dequant output tensor, shape: [input_channel, output_channel]
 * @param input_channel: output row
 * @param bits: weight bits
 * @param output_channel: output col
 * @param group_size: group size
 * @param deq_mode: dequant mode,
 *        0: don't use g_idx
 *        1: exllama with g_idx (g_idx has been argsort)
 *        2: general with g_idx (g_idx mapping group_index for each input channel)
 * @param stream: CUDA Stream
 */
template<typename T>
void weight_dequant_gptq(const int *qweights, const T *scales, const int *zeros, const int32_t *g_idx, T *out,
                         int bits, unsigned input_channel, unsigned output_channel, unsigned group_size, int deq_mode, cudaStream_t stream);

/**
 * @brief
 *
 * @tparam T: input type, float, half, bfloat16
 * @tparam TYPE: QUANT_INT8 or QUANT_NF4 or QUANT_FP4
 * @param code: quantiztion map
 * @param A: input tensor, shape: [row, col]
 * @param absmax: shape: [row]
 * @param out: output tensor
 * @param rand: only support "None"
 * @param rand_offset: only support 0
 * @param blocksize: block size
 * @param n: total size of A
 */
template<typename T, QuantType TYPE>
void quantize_block_wise(const float *code, const T *A, float *absmax, unsigned char *out, const float *rand,
                         int rand_offset, int blocksize, int n, cudaStream_t stream);

/**
 * @brief
 *
 * @tparam T1: input type, half or bfloat16
 * @tparam T2: Intermediate variable type, half2 or bfloat162
 * @tparam use_ex: whether to use_exllama
 * @param input: input tensor, shape: [bs, input_channel]
 * @param scales: scale value, half or bfloat16, shape: [input_channel // g, output_channel]
 * @param qweights: quant weight, int32, shape: [input_channel // 8, output_channel]
 * @param qzeros: quant zero, int32, shape: [input_channel // g, output_channel // 8]
 * @param bias: shape: [output_channel]
 * @param g_idx:int32, shape: [input_channel]
 * @param out: output tensor, shape: [bs, output_channel]
 * @param bs: input row
 * @param input_channel: input col
 * @param output_channel: output col
 * @param group_size: group size
 * @param bits: quant bits
 * @param stream: CUDA Stream
 */
template<typename T1, typename T2, bool USE_EX>
void quantized_linear_int4_gptq(const T1 *input, const T1 *scales, const unsigned *qweights, const unsigned *qzeros,
                                T1 *bias, const int32_t *g_idx, T1 *out, unsigned bs, unsigned input_channel, unsigned output_channel,
                                unsigned group_size, unsigned bits, cudaStream_t stream);

/**
 * @brief
 *
 * @tparam T1: input type, half or bfloat16
 * @tparam T2: Intermediate variable type, half2 or bfloat162
 * @param input: input tensor, shape: [bs, input_channel]
 * @param scales: scale value, half or bfloat16, shape: [input_channel // g, output_channel]
 * @param qweights: quant weight, int32, shape: [input_channel // (32/BITS), output_channel]
 * @param qzeros: quant zero, int32, shape: [input_channel // g, output_channel // (32/BITS)]
 * @param bias: shape: [output_channel]
 * @param g_idx:int32, shape: [input_channel]
 * @param out: output tensor, shape: [bs, output_channel]
 * @param bs: input row
 * @param input_channel: input col
 * @param output_channel: output col
 * @param group_size: group size
 * @param use_ex: wheather use exllama
 * @param stream: CUDA Stream
 */
template<typename T1, typename T2>
void quantized_linear_int8_gptq(const T1 *input, const T1 *scales, const unsigned *qweights, const unsigned *qzeros,
                                const T1 *bias, const int *g_idx, T1 *out, unsigned bs, unsigned input_channel, unsigned output_channel, unsigned group_size, bool use_ex,
                                cudaStream_t stream);

/**
 * @brief
 *
 * @tparam T1: input type, half or bfloat16
 * @tparam T2: Intermediate variable type, half2 or bfloat162
 * @tparam quant_type: fp4 or nf4
 * @param input: input tensor, shape: [bs, input_channel]
 * @param scales: scale value, float, shape: [output_channel * input_channel // g]
 * @param qweights: quant weight, unint8, shape: [output_channel * input_channel // 2, 1]
 * @param bias: shape: [output_channel]
 * @param out: output tensor, shape: [bs, output_channel]
 * @param bs: input row
 * @param input_channel: input col
 * @param output_channel: output col
 * @param group_size: group size
 * @param stream: CUDA Stream
 */
template<typename T1, typename T2, QuantType TYPE>
void quantized_linear_float4(const T1 *input, const float *scales, const unsigned char *qweights, const T1 *bias, T1 *out,
                             unsigned bs, unsigned input_channel, unsigned output_channel, unsigned group_size, cudaStream_t stream);




// ========================================================
// act and mul
// ========================================================

/**
 * @brief gelu_and_mul
 *
 * @tparam T: input type, half or bfloat16 or float
 * @param input: gelu_and_mul input,  shape: [num_tokens, 2 * hidden_size]
 * @param out: output tensor, shape: [num_tokens, hidden_size]
 * @param num_tokens: the number of tokens
 * @param hidden_size: HiddenSize
 * @param gate_first: bool type, Deciding whether the gelu function should be applied to the first half
 *         or the second half of the input
 * @param stream: CUDA Stream
 */
template<typename T>
void gelu_and_mul(const T *input, T *out,
                  int num_tokens, int hidden_size, bool gate_first, cudaStream_t stream);

/**
 * @brief gelu_tanh_and_mul
 *
 * @tparam T: input type, half or bfloat16 or float
 * @param input: gelu_tanh_and_mul input, shape: [num_tokens, 2 * hidden_size]
 * @param out: output tensor, shape: [num_tokens, hidden_size]
 * @param num_tokens: the number of tokens
 * @param hidden_size: HiddenSize
 * @param stream: CUDA Stream
 */
template<typename T>
void gelu_tanh_and_mul(const T *input, T *out,
                       int num_tokens, int hidden_size, cudaStream_t stream);

/**
 * @brief silu_and_mul
 *
 * @tparam T: input type, half or bfloat16 or float
 * @param input: silu_and_mul input, shape: [num_tokens, 2 * hidden_size]
 * @param out: output tensor, shape: [num_tokens, hidden_size]
 * @param num_tokens: the number of tokens
 * @param hidden_size: HiddenSize
 * @param stream: CUDA Stream
 */
template<typename T>
void silu_and_mul(const T *input, T *out,
                  int num_tokens, int hidden_size, cudaStream_t stream);


// ========================================================
// LayerNorm
// ========================================================

/**
 * @brief layernorm
 *
 * @tparam T: input type, half or bfloat16
 * @param input: layernorm input, shape: [batch_count * seq_len, hidden_size]
 * @param scale: weight, shape: [hidden_size]
 * @param bias: bias, shape: [hidden_size]
 * @param fused_bias: fused_bias[optional], shape: [hidden_size]
 * @param out: output tensor, shape: [batch_count * seq_len, hidden_size]
 * @param batch_tokens: int, Batch * InputTokens
 * @param hidden_size: int, HiddenSize
 * @param in_stride: int, the stride of dim "HiddenSize" to support non-contiguous input
 * @param eps: float, a value added to the denominator for numerical stability
 * @param stream: CUDA Stream
 */
template<typename T>
void layernorm(const T *input, const T *scale, const T *bias, const T *fused_bias,
               T *out, int batch_tokens, int hidden_size, int in_stride,
               float eps, cudaStream_t stream);

/**
 * @brief layernorm_residual
 *
 * @tparam T: input type, half or bfloat16
 * @tparam IS_POST: bool type, post-layernorm(true) or pre-layernorm(false)
 * @param input: layernorm input, shape: [batch_count * seq_len, hidden_size]
 * @param residual: residual tensor, shape: [batch_count * seq_len, hidden_size]
 * @param scale: weight, shape: [hidden_size]
 * @param bias: bias, shape: [hidden_size]
 * @param fused_bias: fused_bias[optional], shape: [hidden_size]
 * @param output: output[optional], shape: [batch_count * seq_len, hidden_size]
 * @param residual_output: residual_output[optional], shape: [batch_count * seq_len, hidden_size]
 * @param alpha: float, residual scale factor
 * @param batch_tokens: int, Batch * InputTokens
 * @param hidden_size: int, HiddenSize
 * @param in_stride: int, the stride of dim "HiddenSize" to support non-contiguous input
 * @param resi_stride: int, the stride of dim "HiddenSize" to support non-contiguous residual
 * @param eps: float, a value added to the denominator for numerical stability
 * @param stream: CUDA Stream
 */
template<typename T, bool IS_POST>
void layernorm_residual(T *input, T *residual,
                        const T *scale, const T *bias,
                        const T *fused_bias,
                        T *output, T *residual_output,
                        float alpha, int batch_tokens, int hidden_size,
                        int in_stride, int resi_stride,
                        float eps, cudaStream_t stream);
/**
 * @brief layernorm_2sb
 *
 * @tparam T: input type, half or bfloat16
 * @param input: layernorm input, shape: [batch_count * seq_len, hidden_size]
 * @param scale1: the first weight, shape: [hidden_size]
 * @param bias1: the first bias, shape: [hidden_size]
 * @param scale2: the second weight, shape: [hidden_size]
 * @param bias2: the second bias, shape: [hidden_size]
 * @param eps: float, a value added to the denominator for numerical stability
 * @param output1: the first output tensor, shape: [batch_count * seq_len, hidden_size]
 * @param output2: the second output tensor, shape: [batch_count * seq_len, hidden_size]
 * @param batch_tokens: int, Batch * InputTokens
 * @param hidden_size: int, HiddenSize
 * @param stream: CUDA Stream
 */
template<typename T>
void layernorm_2sb(const T *input,
                   const T *scale1, const T *bias1,
                   const T *scale2, const T *bias2,
                   float eps,
                   T *output1, T *output2,
                   int batch_tokens, int hidden_size, cudaStream_t stream);

/**
 * @brief layernorm + residual + 2sb
 *
 * @tparam T: input type, half or bfloat16
 * @param input: layernorm input, shape: [batch_count * seq_len, hidden_size]
 * @param residual: residual tensor, shape: [batch_count * seq_len, hidden_size]
 * @param scale1: the first weight, shape: [hidden_size]
 * @param bias1: the first bias, shape: [hidden_size]
 * @param scale2: the second weight, shape: [hidden_size]
 * @param bias2: the second bias, shape: [hidden_size]
 * @param eps: float, a value added to the denominator for numerical stability
 * @param output1: the first output tensor, shape: [batch_count * seq_len, hidden_size]
 * @param output2: the second output tensor, shape: [batch_count * seq_len, hidden_size]
 * @param batch_tokens: int, Batch * InputTokens
 * @param hidden_size: int, HiddenSize
 * @param stream: CUDA Stream
 */
template<typename T>
void layernorm_residual_2sb(const T *input, T *residual,
                            const T *scale1, const T *bias1,
                            const T *scale2, const T *bias2,
                            float eps,
                            T *output1, T *output2,
                            int batch_tokens, int hidden_size, cudaStream_t stream);


// ========================================================
// RMS Norm
// ========================================================

/**
 * @brief RMS Norm
 * @tparam T: input type, half or bfloat16
 * @param input: RMS Norm input, shape: [Batch, InputTokens, HiddenSize] or [Batch * InputTokens, HiddenSize]
 * @param weight: RMS Norm weight tensor, shape: [HiddenSize]
 * @param fused_bias: fused_bias tensor[optional], shape: [HiddenSize]
 * @param out: output tensor, shape: [Batch, InputTokens, HiddenSize] or [Batch * InputTokens, HiddenSize]
 * @param batch_tokens: Batch * InputTokens
 * @param hidden_size: HiddenSize
 * @param in_stride: int, the stride of dim "HiddenSize" to support non-contiguous input
 * @param eps: a value added to the denominator for numerical stability
 * @param stream: CUDA Stream
 */
template<typename T>
void rms_norm(const T *input, const T *weight, const T *fused_bias, T *out,
              int batch_tokens, int hidden_size, int in_stride,
              float eps, cudaStream_t stream);

/**
 * @brief RMS Norm + Residual
 * @tparam T: input type, half or bfloat16
 * @tparam IS_POST: bool type, post-layernorm(true) or pre-layernorm(false)
 * @param input: RMS Norm input, shape: [Batch, InputTokens, HiddenSize] or [Batch * InputTokens, HiddenSize]
 * @param residual: residual tensor, shape: [Batch, InputTokens, HiddenSize] or [Batch * InputTokens, HiddenSize]
 * @param weight: RMS Norm weight tensor, shape: [HiddenSize]
 * @param fused_bias: fused_bias tensor[optional], shape: [HiddenSize]
 * @param output: output tensor, shape: [Batch, InputTokens, HiddenSize] or [Batch * InputTokens, HiddenSize]
 * @param residual_output: residual output tensor[optional], shape: [Batch, InputTokens, HiddenSize] or [Batch * InputTokens, HiddenSize]
 * @param batch_tokens: Batch * InputTokens
 * @param alpha: float, residual scale factor
 * @param hidden_size: HiddenSize
 * @param in_stride: int, the stride of dim "HiddenSize" to support non-contiguous input
 * @param resi_stride: int, the stride of dim "HiddenSize" to support non-contiguous residual
 * @param eps: a value added to the denominator for numerical stability
 * @param stream: CUDA Stream
 */
template<typename T, bool IS_POST>
void rms_norm_residual(T *input, T *residual, const T *weight,
                       const T *fused_bias, T *output, T *residual_output,
                       int batch_tokens, float alpha, int hidden_size, int in_stride, int resi_stride,
                       float eps, cudaStream_t stream);

// ========================================================
// Softmax
// ========================================================

/**
 * @brief softmax_2d
 *
 * @tparam T: input type, half
 * @param input: softmax_2D input, shape: [*], where * means, any number of additional dimensions
 * @param out:output tensor, same shape as input
 * @param outer_dim: the product of all dimensions of input except the last dim
 * @param inner_dim: the value is input.size(input.dim()-1)
 * @param stream: CUDA Stream
 */
template<typename T>
void softmax_2d(const T *input, T *out, int outer_dim, int inner_dim,
                cudaStream_t stream);

/**
 * @brief fast_softmax_forwardimp
 *
 * @tparam T: input type, half,shape: [*], where * means, any number of additional dimensions
 * @param stream: CUDA Stream
 * @param input: fast_softmax input
 * @param out: output tensor, same shape as input
 * @param outer_dim: The product of all dimensions of input except the last dim
 * @param inner_dim: the value is input.size(input.dim()-1)
 */
template<typename T>
void fast_softmax_forwardimp(const T *input, T *out, int outer_dim, int inner_dim, cudaStream_t stream);

// ========================================================
// Add
// ========================================================

/**
 * @brief element wise add
 *
 * @tparam T input type, half or bfloat16 or float
 * @param A: input tensor, shape: (...)
 * @param B: other tensor, shape: (...) same as A
 * @param C: out tensor, shape: (...) same as A
 * @param m: default = 1
 * @param n: A.numel()
 * @param stream: CUDA Stream
 */
template<typename T>
void add(const T *A, const T *B, T *C, int m, int n, cudaStream_t stream);

// ========================================================
// GroupNorm
// ========================================================

/**
 * @brief groupnorm_ixinfer
 * @tparam T: input type, half
 * @param input: groupnorm_ixinfer input, shape: [N, C, H, W] or [N, H, W, C] or [N, C, HW] where C = num_channels
 * @param scale: weight, shape: [C]
 * @param bias: bias, shape: [C]
 * @param out: output tensor, shape: [N, C, H, W] or [N, H, W, C] or [N, C, HW] where C = num_channels
 * @param batch: Batch Size = N
 * @param hw: Product of H and W
 * @param num_channel: the number of channel, the value is C
 * @param num_group: number of groups to separate the channels into
 * @param eps: a value added to the denominator for numerical stability
 * @param is_nhwc: bool type. NHWC or NCHW
 * @param act_type: 0 or 1, if act_type=1, use silu; if act_type=0, no activate
 * @param stream: CUDA Stream
 */
template<typename T>
void groupnorm_ixinfer(const T *input, const T *scale, const T *bias, T *out, int batch, int hw,
                       int num_channel, int num_group, float eps, bool is_nhwc, int act_type, cudaStream_t stream);




// ========================================================
// TGI
// ========================================================
/**
 * @brief
 *
 * @tparam T: input type, half or bfloat16
 * @param logits: input tensor, shape: [num_tokens, vocab_size]
 * @param index : the indices of elements to gather
 * @param out: output tensor
 * @param n: The number of elements in index tensor
 * @param vocab_size: vocabulary size
 * @param stream: CUDA Stream
 */
template<typename T>
void tgi_gather_prefill_logprobs(const T *logits, const int32_t *index, T *out,
                                 int n, int vocab_size, cudaStream_t stream);

/**
 * @brief
 *
 * @tparam T1: input type, half or bfloat16 or float
 * @tparam IS_NEOX: Determine whether to use Neox, that is, whether to use interleaved
 * @param query1: The first half of the query tensor in last dimension, shape: [num_tokens, num_heads, head_size //2]
 * @param query2: The second half of the query tensor in last dimension, shape: [num_tokens, num_heads, head_size //2]
 * @param cos: applied in query1, shape: [max_position, 1, head_size //2]
 * @param sin: applied in query2, shape: [max_position, 1, head_size //2]
 * @param out1: The first half of output tensor in last dimension, shape: [num_tokens, num_heads, head_size //2]
 * @param out2: The second half of output tensor in last dimension, shape: [num_tokens, num_heads, head_size //2]
 * @param rot_dim: cos.size(2)
 * @param query1_stride: query1.stride(0)
 * @param num_tokens: the number of tokens
 * @param num_heads: the number of heads
 * @param head_size: head size
 * @param stream: CUDA Stream
 */
template<typename T1>
void tgi_rotary_embedding_neox(const T1 *query1,
                               const T1 *query2,
                               const T1 *cos,
                               const T1 *sin,
                               T1 *out1,
                               T1 *out2,
                               int rot_dim, int query1_stride,
                               int num_tokens, int num_heads, int head_size, bool is_neox,
                               cudaStream_t stream);




// ========================================================
// VLLM
// ========================================================
/**
 * @brief
 *
 * @tparam T: input type, half or bfloat16 or float
 * @param key_cache: key cache, shape:[[num_blocks, num_kv_heads, block_size, head_size],...]
 * @param value_cache: value cache, shape:[[num_blocks, num_kv_heads, block_size, head_size],...]
 * @param block_mapping: shape: [num_tokens, 2]
 * @param num_layers: the number of layers in a model.
 * @param num_pairs: The number of tokens to be mapped. num_pairs = block_mapping.size(0)
 * @param numel_per_block: the number of elements in per block, num_kv_heads* block_size*head_size
 * @param stream: CUDA Stream
 */

template<typename T>
void vllm_copy_blocks(
        int64_t *key_cache,
        int64_t *value_cache,
        const int64_t *block_mapping,
        int num_layers, int num_pairs,
        int numel_per_block, cudaStream_t stream);

/**
 * @brief
 *
 * @tparam T: input type, half or bfloat16 or float
 * @param key: key. shape: [num_tokens, num_heads, head_size]
 * @param value: value. shape: [num_tokens, num_heads, head_size]
 * @param key_cache: key cache, shape: [num_blocks, num_heads, head_size//8, block_size, 8]
 * @param value_cache: value cache. shape: [num_blocks, num_heads, head_size//8, block_size, 8]
 * @param slot_mapping: The mapping position of the token in blocks. shape: [num_tokens]
 * @param key_stride: key.stride(0)
 * @param value_stride: value.stride(0)
 * @param num_heads: the number of heads
 * @param head_size: head size
 * @param block_size: block size
 * @param x: key_cache.size(4)
 * @param num_tokens: the number of tokens
 * @param stream: CUDA Stream
 */
template<typename T>
void vllm_reshape_and_cache_v4(
        const T *key,
        const T *value,
        T *key_cache,
        T *value_cache,
        const int64_t *slot_mapping,
        int key_stride,
        int value_stride,
        int num_heads,
        int head_size,
        int block_size,
        int x,
        int num_tokens,
        cudaStream_t stream);

/**
 * @brief
 *
 * @tparam T: input type, half or bfloat16
 * @param key: key. shape: [num_tokens, num_heads, head_size]
 * @param value: value. shape: [num_tokens, num_heads, head_size]
 * @param key_cache: key cache, shape: [num_blocks, num_heads, block_size, head_size]
 * @param value_cache: value cache. shape: [num_blocks, num_heads, block_size, head_size]
 * @param slot_mapping:The mapping position of the token in blocks. shape: [num_tokens]
 * @param key_token_stride: key.stride(0)
 * @param value_token_stride: value.stride(0)
 * @param value_head_stride: value.stride(1)
 * @param num_heads: the number of heads
 * @param head_size: head size
 * @param value_head_size: value head size, could be different from head size
 * @param block_size: block size
 * @param num_tokens: the number of tokens
 * @param stream: CUDA Stream
 */
template<typename T>
void vllm_reshape_and_cache(
        const T *key,
        const T *value,
        T *key_cache,
        T *value_cache,
        const int64_t *slot_mapping,
        int key_token_stride,
        int value_token_stride,
        int value_head_stride,
        int num_heads,
        int head_size,
        int value_head_size,
        int block_size,
        int num_tokens,
        cudaStream_t stream);

/**
 * @brief
 *
 * @param q_weight: quant weight
 * @param aux_workspace: auxiliary workspace
 * @param q_perm: g_idx
 * @param height: q_weight.size(0) * 32 / bit
 * @param width: q_weight.size(1)
 * @param bit: quant weight bits
 * @param stream: CUDA Stream
 */
template<typename T>
void vllm_shuffle_exllama_weight(
        T *q_weight,
        T *aux_workspace,
        const int *q_perm,
        int height,
        int width,
        int bit,
        cudaStream_t stream);

/**
 * @brief
 *
 * @tparam T: input type, half or bfloat16 or float
 * @tparam IS_NEOX: Determine whether to use Neox, that is, whether to use interleaved
 * @param positions: token positions. shape: [num_tokens]
 * @param query: query,shape: [num_tokens, num_heads * head_size]
 * @param key: key, shape: [num_tokens, num_heads * head_size]
 * @param cos_sin_cache: cos and sin value. shape: [max_position, head_size]
 * @param rot_dim: cos_sin_cache.size(1)
 * @param query_head_stride: stride on dim "head"
 * @param query_token_stride: stride on dim "token"
 * @param key_head_stride: stride on dim "head"
 * @param key_token_stride: stride on dim "token"
 * @param num_heads: the number of query heads
 * @param num_kv_heads: the number of kv heads
 * @param head_size: head size
 * @param num_tokens: the number of tokens
 * @param stream: CUDA Stream
 */

template<typename T, bool IS_NEOX>
void vllm_rotary_embedding(const int64_t *positions,
                           T *query,
                           T *key,
                           const T *cos_sin_cache,
                           int rot_dim, int query_head_stride, int query_token_stride,
                           int key_head_stride, int key_token_stride,
                           int num_heads, int num_kv_heads, int head_size,
                           int num_tokens, cudaStream_t stream);


/**
 * @brief
 *
 * @tparam T: input type, half or bfloat16 or float
 * @tparam IS_NEOX: Determine whether to use Neox, that is, whether to use interleaved
 * @param positions: token positions. shape: [num_tokens]
 * @param query: query,shape: [num_tokens, num_heads * head_size] or [num_tokens, num_heads, head_size]
 * @param key: key, shape: [num_tokens, num_kv_heads * head_size] or [num_tokens, num_kv_heads, head_size]
 * @param cos_sin_cache: cos and sin value. shape: [max_position, head_size]
 * @param scales: scales for key layer norm. shape: [head_size]
 * @param bias: bias for key layer norm. shape: [head_size]
 * @param key_out: result for saving key[nullptr will use inplace operation]
 * @param rot_dim: cos_sin_cache.size(1)
 * @param num_heads: the number of query heads
 * @param num_kv_heads: the number of kv heads
 * @param head_size: head size
 * @param query_head_stride: stride of "num_heads" dim to support non contiguous query
 * @param query_token_stride:  stride of "num_tokens" dim to support non contiguous query
 * @param key_head_stride: stride of "num_kv_heads" dim to support non contiguous key
 * @param key_token_stride: stride of "num_tokens" dim to support non contiguous key
 * @param eps: a value added to the denominator for numerical stability
 * @param num_tokens: the number of tokens
 * @param stream: CUDA Stream
 */
template<typename scalar_t, bool IS_NEOX>
void vllm_rotary_embedding_with_key_layer_norm(const int64_t *positions,
                                               scalar_t *query,
                                               scalar_t *key,
                                               const scalar_t *cos_sin_cache,
                                               const scalar_t *scales,
                                               const scalar_t *bias,
                                               scalar_t *key_out,
                                               int rot_dim,
                                               int num_heads,
                                               int num_kv_heads,
                                               int head_size,
                                               int64_t query_head_stride,
                                               int64_t query_token_stride,
                                               int64_t key_head_stride,
                                               int64_t key_token_stride,
                                               float eps,
                                               int num_tokens,
                                               cudaStream_t stream);

/**
 * @brief
 *
 * @tparam T: input type, half or bfloat16 or float
 * @tparam IS_NEOX: Determine whether to use Neox, that is, whether to use interleaved
 * @param positions: token positions. shape: [num_tokens]
 * @param query: query,shape: [num_tokens, num_heads * head_size]
 * @param key: key, shape: [num_tokens, num_heads * head_size]
 * @param cos_sin_cache: cos and sin value. shape: [max_position, head_size]
 * @param cos_sin_cache_offsets: position offsets. shape: [num_tokens]
 * @param rot_dim: cos_sin_cache.size(1)
 * @param query_stride: query.stride(-2)
 * @param key_stride: key.stride(-2)
 * @param num_heads: the number of query heads
 * @param num_kv_heads: the number of kv heads
 * @param head_size: head size
 * @param num_tokens: the number of tokens
 * @param stream: CUDA Stream
 */
template<typename T, bool IS_NEOX>
void vllm_batched_rotary_embedding(const int64_t *positions,
                                   T *query,
                                   T *key,
                                   const T *cos_sin_cache,
                                   const int64_t *cos_sin_cache_offsets,
                                   int rot_dim, int query_stride, int key_stride,
                                   int num_heads, int num_kv_heads, int head_size,
                                   int num_tokens, cudaStream_t stream);

/**
 * @brief
 *
 * @param num_seqs: the number of sequences
 * @param num_queries: NUM_QUERIES decode request numbers
 * @param block_size: block size
 * @param input_tokens: input token tensor
 * @param sampled_token_ids: sampled token ids tensor
 * @param input_positions: input positions tensor
 * @param seq_lens: seq lens tensor
 * @param slot_mapping: slot mapping tensor
 * @param block_tables: block tables tensor
 * @param block_tables_stride: block_tables.stride(0)
 * @param stream: CUDA Stream
 */
void vllm_advance_step_flashattn(int num_seqs, int num_queries, int block_size,
                                 long *input_tokens,
                                 const long *sampled_token_ids,
                                 long *input_positions,
                                 int *seq_lens,
                                 long *slot_mapping,
                                 const int *block_tables,
                                 long block_tables_stride,
                                 cudaStream_t stream);


/**
 * @brief
 *
 * @param positions: [num_tokens]
 * @param long_prompt_offset: [num_tokens]
 * @param long_short_cos_sin_cache: [num_tokens, head_dim]
 * @param query: shape=[num_tokens, num_q_heads, head_dim] stride=[query_stride_0, query_stride_1, 1]
 * @param key: shape=[num_tokens, num_kv_heads, head_dim] stride=[key_stride_0, key_stride_1, 1]
 * @param out_query: shape=[num_tokens, num_q_heads, head_dim] stride=[out_query_stride_0, out_query_stride_1, 1]
 * @param out_key: shape=[num_tokens, num_kv_heads, head_dim] stride=[out_key_stride_0, out_key_stride_1, 1]
 */

template<typename scalar_t>
void minicpm3_fused_rope(
        const int64_t *positions,
        const int64_t *long_prompt_offset,
        const scalar_t *long_short_cos_sin_cache,
        const scalar_t *query,
        const scalar_t *key,
        scalar_t *out_query,
        scalar_t *out_key,
        int64_t num_tokens,
        int64_t num_q_heads,
        int64_t num_kv_heads,
        int64_t head_dim,
        int64_t query_stride_0,
        int64_t query_stride_1,
        int64_t key_stride_0,
        int64_t key_stride_1,
        int64_t out_query_stride_0,
        int64_t out_query_stride_1,
        int64_t out_key_stride_0,
        int64_t out_key_stride_1,
        cudaStream_t stream);

/**
 * @brief
 *
 * @param k_nope: shape=(num_tokens, num_kv_heads, k_head_dim) stride=(k_nope_stride_0, k_nope_stride_1, 1)
 * @param k_pe: shape=(num_tokens, 1, head_dim - k_head_dim) stride=(k_pe_stride_0, -1, 1)
 * @param v: shape=(num_tokens, num_kv_heads, v_head_dim) stride=(v_stride_0, v_stride_1, 1)
 * @param new_k: shape=(num_tokens, num_kv_heads, head_dim) contiguous
 * @param new_v: shape=(num_tokens, num_kv_heads, head_dim) contiguous
 */
template<typename scalar_t>
void minicpm3_fused_copy_kv(
        const scalar_t *k_nope,
        const scalar_t *k_pe,
        const scalar_t *v,
        scalar_t *new_k,
        scalar_t *new_v,
        int64_t num_tokens,
        int64_t num_kv_heads,
        int64_t head_dim,
        int64_t k_head_dim,
        int64_t v_head_dim,
        int64_t k_nope_stride_0,
        int64_t k_nope_stride_1,
        int64_t k_pe_stride_0,
        int64_t v_stride_0,
        int64_t v_stride_1,
        cudaStream_t stream);


/**
 * @brief
 *
 * @tparam T: input type, half or bfloat16 or float
 * @param positions: token positions. shape: [num_tokens]
 * @param query: query,shape: [num_tokens, num_heads * head_size]
 * @param key: key, shape: [num_tokens, num_heads * head_size]
 * @param cos_sin_cache: cos and sin value. shape: [max_position, head_size]
 * @param offset: offset for lora, could be nullptr. shape: [max_position,]
 * @param long_offset: add k or not. shape: [1, ]
 * @param k: offset for long inputs
 * @param rot_dim: cos_sin_cache.size(1)
 * @param query_head_stride: stride on dim "head"
 * @param query_token_stride: stride on dim "token"
 * @param key_head_stride: stride on dim "head"
 * @param key_token_stride: stride on dim "token"
 * @param num_heads: the number of query heads
 * @param num_kv_heads: the number of kv heads
 * @param head_size: head size
 * @param num_tokens: the number of tokens
 * @param stream: CUDA Stream
 */
template<typename scalar_t>
void vllm_rotary_embedding_phi(const int64_t *positions,
                               scalar_t *query,
                               scalar_t *key,
                               const scalar_t *cos_sin_cache,
                               const int64_t *offset,
                               const bool *long_offset,
                               const int64_t k,
                               int rot_dim,
                               int query_head_stride,
                               int query_token_stride,
                               int key_head_stride,
                               int key_token_stride,
                               int num_heads,
                               int num_kv_heads,
                               int head_size,
                               int num_tokens,
                               cudaStream_t stream);


/**
 * @brief
 *
 * @tparam T: input type, half or bfloat16 or float
 * @param positions: token positions. shape: [num_tokens]
 * @param query: query,shape: [num_tokens, num_heads * head_size]
 * @param key: key, shape: [num_tokens, head_size]
 * @param key_out: key_out, shape: [num_tokens, num_heads * head_size]
 * @param cos_sin_cache: cos and sin value. shape: [max_position, head_size]
 * @param offset: offset for lora, could be nullptr. shape: [max_position,]
 * @param long_offset: add k or not. shape: [1, ]
 * @param k: offset for long inputs
 * @param rot_dim: cos_sin_cache.size(1)
 * @param query_head_stride: stride on dim "head"
 * @param query_token_stride: stride on dim "token"
 * @param key_head_stride: stride on dim "head"
 * @param key_token_stride: stride on dim "token"
 * @param key_out_head_stride: stride on dim "head"
 * @param key_out_token_stride: stride on dim "token"
 * @param num_heads: the number of query heads
 * @param head_size: head size
 * @param num_tokens: the number of tokens
 * @param stream: CUDA Stream
 */
template<typename scalar_t>
void rotary_embedding_mla_phi(const int64_t *positions,
                              scalar_t *query,
                              scalar_t *key,
                              scalar_t *key_out,
                              const scalar_t *cos_sin_cache,
                              const int64_t *offset,
                              bool *long_offset,
                              int64_t k,
                              int rot_dim,
                              int query_head_stride,
                              int query_token_stride,
                              int key_head_stride,
                              int key_token_stride,
                              int key_out_head_stride,
                              int key_out_token_stride,
                              int num_heads,
                              int head_size,
                              int num_tokens,
                              cudaStream_t stream);


/**
 * @brief
 *
 * @tparam T: input type, half or bfloat16 or float
 * @tparam IS_NEOX: Determine whether to use Neox, that is, whether to use interleaved
 * @param positions: token positions. shape: [num_tokens]
 * @param query: query,shape: [num_tokens, num_heads * head_size]
 * @param key: key, shape: [num_tokens, num_heads * head_size]
 * @param key_out: key_out, shape: [num_tokens, num_heads * head_size]
 * @param cos_sin_cache: cos and sin value. shape: [max_position, head_size]
 * @param offset: offset for lora, could be nullptr. shape: [max_position,]
 * @param rot_dim: cos_sin_cache.size(1)
 * @param query_head_stride: stride on dim "head"
 * @param query_token_stride: stride on dim "token"
 * @param key_head_stride: stride on dim "head"
 * @param key_token_stride: stride on dim "token"
 * @param key_out_head_stride: stride on dim "head"
 * @param key_out_token_stride: stride on dim "token"
 * @param num_heads: the number of query heads
 * @param head_size: head size
 * @param num_tokens: the number of tokens
 * @param stream: CUDA Stream
 */
template<typename scalar_t, bool IS_NEOX>
void rotary_embedding_mla(const int64_t *positions,
                          scalar_t *query,
                          scalar_t *key,
                          scalar_t *key_out,
                          const scalar_t *cos_sin_cache,
                          const int64_t *offset,
                          int rot_dim,
                          int query_head_stride,
                          int query_token_stride,
                          int key_head_stride,
                          int key_token_stride,
                          int key_out_head_stride,
                          int key_out_token_stride,
                          int num_heads,
                          int head_size,
                          int num_tokens,
                          cudaStream_t stream);


/**
 * @brief
 *
 * @param key_nope: key_nope,shape: [num_tokens, num_heads, k_nope_dim]
 * @param value_nope: value_nope,shape: [num_tokens, num_heads, v_head_dim]
 * @param key: key, shape: [num_tokens, num_heads, head_size]
 * @param value: value, shape: [num_tokens, num_heads, head_size]
 * @param num_tokens: num_tokens
 * @param num_heads: num_heads
 * @param head_dim: head_size
 * @param k_nope_dim: k_nope_dim
 * @param v_head_dim: v_head_dim
 * @param key_head_stride: stride on dim "head"
 * @param key_token_stride: stride on dim "token"
 * @param key_out_head_stride: stride on dim "head"
 * @param key_out_token_stride: stride on dim "token"
 * @param stream: CUDA Stream
 */
template<typename scalar_t>
void copy_kv_mla(
        const scalar_t *key_nope,
        const scalar_t *value_nope,
        scalar_t *key,
        scalar_t *value,
        int64_t num_tokens,
        int64_t num_heads,
        int64_t head_dim,
        int64_t k_nope_dim,
        int64_t v_head_dim,
        int64_t k_nope_head_stride,
        int64_t k_nope_token_stride,
        int64_t v_nope_head_stride,
        int64_t v_nope_token_stride,
        cudaStream_t stream);

/**
 * @brief
 *
 * @param src_cache: src_cache,shape: [NUM_BLOCKS, BLOCK_SIZE,ENTRIES...]
 * @param dst: workspace,shape: [TOT_TOKENS, ENTRIES...]
 * @param block_table: block_table, shape: [BATCH, BLOCK_INDICES]
 * @param cu_seq_lens: cu_seq_lens, shape: [BATCH+1]
 * @param seq_starts: Optional: starting offsets per batch, shape: [BATCH]
 * @param batch_size: batch size
 * @param block_size: block size
 * @param entry_size: entry size
 * @param block_table_stride: stride on dim "BATCH"
 * @param cache_block_stride: stride on dim "NUM_BLOCKS"
 * @param cache_entry_stride: stride on dim "BLOCK_SIZE"
 * @param dst_entry_stride: stride on dim "TOT_TOKENS"
 * @param stream: CUDA Stream
 */
template<typename scalar_t>
void vllm_gather_cache(
        const scalar_t *src_cache,
        scalar_t *dst,
        const int32_t *block_table,
        const int32_t *cu_seq_lens,
        const int32_t *seq_starts,
        const int64_t batch_size,
        const int32_t block_size,
        const int32_t entry_size,
        const int64_t block_table_stride,
        const int64_t cache_block_stride,
        const int64_t cache_entry_stride,
        const int64_t dst_entry_stride,
        cudaStream_t stream);

/**
 * @brief
 *
 * @param src_cache: src_cache,shape: [NUM_BLOCKS, BLOCK_SIZE,ENTRIES...]
 * @param src_cache_scale: src_cache,shape: [NUM_BLOCKS, BLOCK_SIZE,2]
 * @param dst: workspace,shape: [TOT_TOKENS, ENTRIES...]
 * @param block_table: block_table, shape: [BATCH, BLOCK_INDICES]
 * @param cu_seq_lens: cu_seq_lens, shape: [BATCH+1]
 * @param seq_starts: Optional: starting offsets per batch, shape: [BATCH]
 * @param kv_lora_rank: kv_lora_rank
 * @param batch_size: batch size
 * @param block_size: block size
 * @param entry_size: entry size
 * @param block_table_stride: stride on dim "BATCH"
 * @param cache_block_stride: stride on dim "NUM_BLOCKS" of src_cache
 * @param scale_cache_block_stride: stride on dim "NUM_BLOCKS" of src_cache_scale
 * @param cache_entry_stride: stride on dim "BLOCK_SIZE" of src_cache
 * @param scale_cache_entry_stride: stride on dim "BLOCK_SIZE" of src_cache_scale
 * @param dst_entry_stride: stride on dim "TOT_TOKENS"
 * @param stream: CUDA Stream
 */
template<typename scalar_t>
void vllm_gather_cache_int8(
        const int8_t *src_cache,
        const float *src_cache_scale,
        scalar_t *dst,
        const int32_t *block_table,
        const int32_t *cu_seq_lens,
        const int32_t *seq_starts,
        const int64_t kv_lora_rank,
        const int64_t batch_size,
        const int32_t block_size,
        const int32_t entry_size,
        const int64_t block_table_stride,
        const int64_t cache_block_stride,
        const int64_t scale_cache_block_stride,
        const int64_t cache_entry_stride,
        const int64_t scale_cache_entry_stride,
        const int64_t dst_entry_stride,
        cudaStream_t stream);

/**
 * @brief
 *
 * @param kv_c: kv_c, shape: [num_tokens, kv_lora_rank]
 * @param k_pe: query, shape: [num_tokens, 1(n), pe_dim]
 * @param key_cache: key, shape: [num_tokens, block_size, (kv_lora_rank + pe_dim)]
 * @param slot_mapping: slot_mapping, shape: [num_tokens]
 * @param kv_lora_rank: kv_lora_rank
 * @param pe_dim: pe_dim
 * @param block_size: block_size
 * @param kv_c_stride: stride on dim "num_tokens"
 * @param k_pe_stride: stride on dim "num_tokens"
 * @param block_stride: stride on dim "num_tokens" of key_cache
 * @param dim_stride: stride on dim "block_size" of key_cache
 * @param num_tokens: num_tokens
 * @param stream: CUDA Stream
 */
template<typename scalar_t>
void vllm_concat_and_cache_mla(
        const scalar_t *kv_c,
        const scalar_t *k_pe,
        scalar_t *key_cache,
        const int64_t *slot_mapping,
        int kv_lora_rank,
        int pe_dim,
        int block_size,
        int kv_c_stride,
        int k_pe_stride,
        int block_stride,
        int dim_stride,
        int num_tokens,
        cudaStream_t stream);
/**
 * @brief
 *
 * @param kv_c: kv_c, shape: [num_tokens, kv_lora_rank]
 * @param kv_c_scale: kv_c_scale, shape: [num_tokens]
 * @param k_pe: query, shape: [num_tokens, 1(n), pe_dim]
 * @param k_pe_scale: query, shape: [num_tokens, 1(n)]
 * @param key_cache: key, shape: [num_tokens, block_size, (kv_lora_rank + pe_dim)]
 * @param key_cache_scale: key, shape: [num_tokens, block_size, 2]
 * @param slot_mapping: slot_mapping, shape: [num_tokens]
 * @param kv_lora_rank: kv_lora_rank
 * @param pe_dim: pe_dim
 * @param block_size: block_size
 * @param kv_c_stride: stride on dim "num_tokens"
 * @param kv_c_scale_stride: stride on dim "num_tokens"
 * @param k_pe_stride: stride on dim "num_tokens"
 * @param k_pe_scale_stride: stride on dim "num_tokens"
 * @param block_stride: stride on dim "num_tokens" of key_cache
 * @param scale_block_stride: stride on dim "num_tokens" of key_cache_scale
 * @param dim_stride: stride on dim "block_size" of key_cache
 * @param scale_dim_stride: stride on dim "block_size" of key_cache_scale
 * @param num_tokens: num_tokens
 * @param stream: CUDA Stream
*/
template<typename scalar_t>
void vllm_concat_and_cache_mla_int8(
        const scalar_t *kv_c,
        const float *kv_c_scale,
        const scalar_t *k_pe,
        const float *k_pe_scale,
        scalar_t *key_cache,
        float *key_cache_scale,
        const int64_t *slot_mapping,
        int kv_lora_rank,
        int pe_dim,
        int block_size,
        int kv_c_stride,
        int kv_c_scale_stride,
        int k_pe_stride,
        int k_pe_scale_stride,
        int block_stride,
        int scale_block_stride,
        int dim_stride,
        int scale_dim_stride,
        int num_tokens,
        cudaStream_t stream);

/**
 * @brief
 *
 * @param output, shape: [seq_len, num_heads, head_dim]
 * @param output_lse, shape: [num_heads, seq_len]
 * @param prefix_output, shape: [seq_len, num_heads, head_dim]
 * @param prefix_lse, shape: [num_heads, seq_len]
 * @param suffix_output, shape: [seq_len, num_heads, head_dim]
 * @param suffix_lse, shape: [num_heads, seq_len]
 */
template<typename scalar_t>
void merge_attn_states(
        scalar_t *output,
        float *output_lse,
        const scalar_t *prefix_output,
        const float *prefix_lse,
        const scalar_t *suffix_output,
        const float *suffix_lse,
        int num_heads,
        int seq_len,
        int head_dim,
        cudaStream_t stream);



/*
    MARLIN_FORMAT_K16N32
        w:(batch, k/16, n/32, 64)     int32     pack order:[0 2 4 6 1 3 5 7]
        s:(batch, k_groups, n/32, 32) float16   32 data order:[0 16 1 17 ... 15 31]
        z:(batch, k_groups, n/32, 32) int4      32 data order:[0 16 1 17 ... 15 31]
    MARLIN_FORMAT_K16N32_GROUPED_ON_N
        w:(batch, k/16, n/32, 64)     int32     pack order:[0 2 4 6 1 3 5 7]
        s:(batch, n_groups, k)        float16
        z:(batch, n_groups, k/8)      int4      pack order:[0 1 2 3 4 5 6 7]
    MARLIN_FORMAT_K16N16
        w:(batch, k/16, n/16, 64)     int32     pack order:[0 1 2 3]
        s:(batch, k_groups, n)        float32
    MARLIN_FORMAT_K16N16_GROUPED_ON_N
        w:(batch, k/16, n/16, 64)     int32     pack order:[0 1 2 3]
        s:(batch, n_groups, k)        float32
*/
typedef enum {
    MARLIN_FORMAT_K16N32,
    MARLIN_FORMAT_K16N32_GROUPED_ON_N,
    MARLIN_FORMAT_K16N16,
    MARLIN_FORMAT_K16N16_GROUPED_ON_N,
} MarlinFormat;

/*
    ORIGIN_FORMAT_AWQ,
        pack_order:[0 2 4 6 1 3 5 7]
        w:(batch, k, n/8)           int32
        s:(batch, k_groups, n)      float16
        z:(batch, k_groups, n/8)    int32
    ORIGIN_FORMAT_GPTQ,
        pack_order:[0 1 2 3 4 5 6 7]
        w:(batch, k/8, n)           int32
        s:(batch, k_groups, n)      float16
        z:(batch, k_groups, n/8)    int32
    ORIGIN_FORMAT_GPTQ_GROUPED_N,
        pack_order:[0 2 4 6 1 3 5 7]
        w:(batch, k/8, n)           int32
        s:(batch, n_groups, k)      float16
        z:(batch, n_groups, k/8)    int32
    ORIGIN_FORMAT_INT8
        w:(batch, k, n)             int8
*/
typedef enum {
    ORIGIN_FORMAT_AWQ,
    ORIGIN_FORMAT_GPTQ,
    ORIGIN_FORMAT_GPTQ_GROUPED_N,
    ORIGIN_FORMAT_INT8,
} WeightFormat;

typedef enum {
    PACK_ORDER_01234567,
    PACK_ORDER_02461357,
} PackOrder;

/**
 * @brief
 *
 * @tparam DType: input type, half or bfloat16
 * @param input: input tensor, shape: batch_first ? [batch_count, m, k] : [m, batch_count, k]
 * @param weight: marlin repack weights, shape: [batch_count, k/16, n/32, 64]
 * @param scale: marlin repack scale, shape: weight_format == "k16n32" ? [batch, k_groups, n] : [batch, n_groups, k]
 * @param zero: marlin repack zero, shape: weight_format == "k16n32" ? [batch, k_groups, n/8] : [batch, n_groups, k/8]
 * @param bias: bias for result, TODO
 * @param out: output tensor, shape: batch_first ? [batch_count, m, n] : [m, batch_count, n]
 * @param aux: workspace for kernel
 * @param batch_count: batched gemm paraments
 * @param m: gemm paraments
 * @param k: gemm paraments
 * @param n: gemm paraments
 * @param group_size: group size of quant
 * @param pad_k: stride for k dimension of input
 * @param batch_first: describe format of input and output
 * @param weight_format: describe format of weight
 * @param stream: CUDA Stream
 */
template<typename DType>
void marlin_w4a16(const DType *input, const int32_t *weight, const DType *scale, const int32_t *zero, const DType *bias,
                  DType *out, float *aux, int batch_count, int m, int k, int n, int group_size, int pad_k, bool batch_first, MarlinFormat weight_format, cudaStream_t stream);


/**
 * @brief
 *
 * @param weight: origin weight tensor
 * @param repack_weight: marlin repack weight tensor
 * @param scale: origin scale tensor
 * @param repack_scale: marlin repack scale tensor
 * @param zero: origin zero tensor
 * @param repack_zero: marlin repack zero tensor
 * @param batch_count: batched gemm paraments
 * @param n: gemm paraments
 * @param k: gemm paraments
 * @param groups: groups of quant
 * @param origin_format: describe format of origin weight
 * @param origin_pack_order: describe pack order of origin weight
 * @param marlin_format: describe format of repack weight
 * @param stream: CUDA Stream
 */
void marlin_w4_weight_repack(const void *weight, void *repack_weight,
                             const void *scale, void *repack_scale,
                             const void *zero, void *repack_zero,
                             int batch_count, int n, int k, int groups,
                             WeightFormat origin_format, PackOrder origin_pack_order, MarlinFormat marlin_format, cudaStream_t stream);

/**
 * @brief
 *
 * @tparam DType: input type, half or bfloat16
 * @param input: input tensor, shape: batch_first ? [batch_count, m, k] : [m, batch_count, k]
 * @param weight: marlin repack weights, shape: [batch_count, k/16, n/16, 64]
 * @param scale: marlin repack scale, shape: weight_format == "k16n16" ? [batch, k_groups, n] : [batch, n_groups, k]
 * @param bias: bias for result, TODO
 * @param out: output tensor, shape: batch_first ? [batch_count, m, n] : [m, batch_count, n]
 * @param aux: workspace for kernel
 * @param batch_count: batched gemm paraments
 * @param m: gemm paraments
 * @param k: gemm paraments
 * @param n: gemm paraments
 * @param group_size: group size of quant
 * @param pad_k: stride for k dimension of input
 * @param batch_first: describe format of input and output
 * @param weight_format: describe format of weight
 * @param stream: CUDA Stream
 */
template<typename DType>
void marlin_w8a16(const DType *input, const int32_t *weight, const float *scale, const DType *bias,
                  DType *out, float *aux, int batch_count, int m, int k, int n, int group_size, int pad_k, bool batch_first, MarlinFormat weight_format, cudaStream_t stream);

/**
 * @brief
 *
 * @param weight: origin weight tensor
 * @param repack_weight: marlin repack weight tensor
 * @param scale: origin scale tensor
 * @param repack_scale: marlin repack scale tensor
 * @param batch_count: batched gemm paraments
 * @param n: gemm paraments
 * @param k: gemm paraments
 * @param groups: groups of quant
 * @param origin_format: describe format of origin weight
 * @param marlin_format: describe format of repack weight
 * @param stream: CUDA Stream
 */
void marlin_w8_weight_repack(const void *weight, void *repack_weight,
                             const void *scale, void *repack_scale,
                             int batch_count, int n, int k, int groups,
                             WeightFormat origin_format, MarlinFormat marlin_format, cudaStream_t stream);

// ========================================================
// bert unpad
// ========================================================
/**
 * @brief bert layernorm fused add residual
 *
 * @tparam T
 * @param input: shape:[num_tokens, hidden_size] half bf16
 * @param residual: shape:[num_tokens, hidden_size] same as input
 * @param ln_weight：  layernorm weight,shape:[hidden_size] same as input
 * @param ln_bias：layernorm bias,shape:[hidden_size]  same as input
 * @param output: shape:[num_tokens, hidden_size] same as input
 * @param num_tokens: int ,total tokens in a batch
 * @param hidden_size: HiddenSize
 * @param epsilon:  float
 * @param stream: CUDA Stream
 */
template<typename T>
void bert_add_norm(const T *input, const T *residual,
                   const T *ln_weight, const T *ln_bias,
                   T *output, int num_tokens, int hidden_size,
                   float epsilon, cudaStream_t stream);
/**
 * @brief bert embeding same as transformers
 *
 * @tparam T
 * @tparam TYPE_INT: type for token_ids pos_ids type_ids
 * @param token_weight:  shape: [vocab_size, hidden_size] half bf16
 * @param pos_weight: shape: [pos_size, hidden_size] same as token_weight
 * @param type_weight:  shape:[type_size, hidden_size] same as token_weight
 * @param ln_weight:  layernorm weight,shape:[hidden_size] same as token_weight
 * @param ln_bias: layernorm bias,shape:[hidden_size] same as token_weight
 * @param token_ids: shape: [num_tokens]
 * @param pos_ids:  shape: [num_tokens]
 * @param type_ids:  shape: [num_tokens]
 * @param output:  shape: [num_tokens, hidden_size]
 * @param num_tokens: int ,total tokens in a batch
 * @param hidden_size: HiddenSize
 * @param epsilon:  float
 * @param stream: CUDA Stream
 */
template<typename T, typename TYPE_INT>
void bert_embedding(const T *token_weight, const T *pos_weight,
                    const T *type_weight, const T *ln_weight,
                    const T *ln_bias,
                    const TYPE_INT *token_ids, const TYPE_INT *,
                    const TYPE_INT *type_ids,
                    T *output, int num_tokens, int hidden_size,
                    float epsilon, cudaStream_t stream);

/**
 * @brief  bert output numtokens unpack to batch,tokens
 *
 * @tparam T
 * @tparam TYPE_INT
 * @param logits: shape:[num_tokens, 2] half bf16
 * @param cu_seq_len: shape:[batch+1],same as in flash atten,accumlate seq_len in a batch,first is 0
 * @param start_logits: shape:[ batch, max_seq_len] half bf16
 * @param end_logits: shape:[ batch, max_seq_len] half bf16
 * @param batch： Batch Size
 * @param max_seq_len
 * @param stream
 */
template<typename T, typename TYPE_INT>
void bert_unpack_start_end_logits(const T *logits, const TYPE_INT *cu_seq_len,
                                  T *start_logits, T *end_logits,
                                  int batch, int max_seq_len,
                                  cudaStream_t stream);

// ========================================================
// Linalg.solve
// ========================================================
/**
 * @brief
 *
 * @tparam T: input type, float
 * @param A: tensor of shape [*, n, n] where * is zero or more batch dimensions.
 * @param B: right-hand side tensor of shape [*, n] or [*, n, k] or or [*, k, n],
 *        where * is zero or more batch dimensions.
 * @param X: output tensor, shape: [*, n] or [*, n, k] or [*, k,n]
 * @param batch: batch size, the value is A.numel() / (n * n)
 * @param n: One of the dimensions of the param B tensor
 * @param k: One of the dimensions of the param B tensor
 * @param stream: CUDA Stream
 */
template<typename T>
void gauss_small(const T *A, const T *B, T *X, int batch, int n, int k, cudaStream_t stream);

// ========================================================
// store_kv_cache
// ========================================================

/**
 * @brief
 *
 * @tparam T: input type, half, bfloat16
 * @param k: key. shape: [batch_size, seqlen_new, head_num, head_dim]
 * @param v: value. shape: [batch_size, seqlen_new, head_num, head_dim]
 * @param k_cache: key cache. shape: [batch_size_cache, seqlen_cache, head_num, head_dim]
 * @param v_cache: value cache. shape: [batch_size_cache, seqlen_cache, head_num, head_dim]
 * @param cache_batch_idx: The indices used to index into the KV cache. shape: [batch_size,]
 * @param cache_seqlens: The sequence lengths of the KV cache. shape: [batch_size,]
 * @param k_stride_1: k.stride(0)
 * @param k_stride_2: k.stride(1)
 * @param k_stride_3: k.stride(1)
 * @param v_stride_1: v.stride(0)
 * @param v_stride_2: v.stride(1)
 * @param v_stride_3: v.stride(1)
 * @param batch_size: k.size(0)
 * @param seq_len_new: k.size(1)
 * @param seqlen_cache: k_cache.size(1)
 * @param head_num: k.size(2)
 * @param head_dim: k.size(3)
 * @param stream: CUDA Stream
 */
template<typename T>
void store_kv_cache(const T *k, const T *v,
                    T *k_cache, T *v_cache,
                    const int32_t *cache_batch_idx, const int32_t *cache_seqlens,
                    int64_t k_stride_1, int64_t k_stride_2, int64_t k_stride_3,
                    int64_t v_stride_1, int64_t v_stride_2, int64_t v_stride_3,
                    int batch_size, int seq_len_new,
                    int seqlen_cache,
                    int head_num, int head_dim,
                    cudaStream_t stream);

// ========================================================
// T5 model
// ========================================================

/**
 * @brief t5_split_qkv
 *
 * @tparam T: input type, half or bfloat16
 * @param qkv: input tensor, shape: [batch_size, seq_len, hidden_size*3]
 * @param q: query tensor, shape: [batch_size, head_num, seq_len, head_dim]
 * @param k: key tensor, shape: [batch_size, head_num, seq_len, head_dim]
 * @param v: value tensor, shape: [batch_size, head_num, seq_len, head_dim]
 * @param batch: int, batch size
 * @param seq_len: int, seq_len
 * @param head_num: int, the number of head
 * @param head_dim: int, head dim
 * @param stream: CUDA Stream
 */
template<typename T>
void t5_split_qkv(const T *qkv, T *q, T *k, T *v, int batch,
                  int seq_len, int head_num, int head_dim, cudaStream_t stream);

/**
 * @brief
 *
 * @tparam T: input type, half or bfloat16
 * @param qkv: input tensor, shape: [batch_size,1,hidden_size*3],hidden_size = head_num*head_dim
 * @param past_key: past key tensor, shape: [batch_size, head_num, seq_len-1, head_dim]
 * @param past_value: past value tensor, shape: [batch_size, head_num, seq_len-1, head_dim]
 * @param q: query tensor, shape: [batch_size, head_num, 1, head_dim]
 * @param k: key tensor, shape: [batch_size, head_num, 1, head_dim]
 * @param v: value tensor, shape: [batch_size, head_num, 1, head_dim]
 * @param batch: int, batch size
 * @param seq_len: int, seq_len
 * @param head_num: int, the number of head
 * @param head_dim: int, head dim
 * @param stream: CUDA Stream
 */
template<typename T>
void t5_split_qkv_update_kv_cache(const T *qkv, const T *past_key, const T *past_value,
                                  T *q, T *k, T *v, int batch,
                                  int seq_len, int head_num, int head_dim,
                                  cudaStream_t stream);

}// namespace ixformer::kernels::infer
