/* Migrated from xllm/core/framework/kv_cache/kv_cache_estimation.h
   [BI-V100 20%] Replaced torch::ScalarType with int32_t dtype_enum
   to avoid libtorch dependency in scheduling layer. */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "framework/kv_cache/kv_cache_capacity.h"
#include "framework/kv_cache/layerwise_split_layout.h"

namespace xllm {

class ModelArgs;

struct KVCacheEstimateOptions {
  int32_t dtype_enum = 15;  // bfloat16; upstream uses torch::ScalarType
  std::string kv_cache_dtype = "auto";
  std::string indexer_cache_dtype = "auto";
  int64_t cache_size_in_bytes = 0;
  int64_t block_size = 0;
  int64_t world_size = 1;
  int64_t n_local_kv_heads = 0;
  int64_t n_local_linear_k_heads = 0;
  int64_t n_local_linear_v_heads = 0;
  int64_t max_seqs_per_batch = 0;
  int64_t num_speculative_tokens = 0;
  int64_t max_tokens_per_batch = 0;
  int64_t max_linear_state_cache_slots = 0;
  bool is_draft_engine = false;
  bool enable_prefix_cache = false;
  int32_t layerwise_split_size = 1;
  const ModelArgs* draft_model_args = nullptr;
  const KVCacheEstimateOptions* draft_options = nullptr;
};

std::vector<bool> resolve_indexer_cache_enabled_layers(
    const ModelArgs& model_args,
    int64_t num_cache_layers);

// Linear-attention layers stay owned on every rank. Full-attention layers
// follow LayerwiseSplitLayout.
std::vector<bool> build_layer_cache_owned(const ModelArgs& model_args,
                                          const LayerwiseSplitLayout& layout,
                                          int64_t num_layers);

// Common block count across a layerwise split group.
int64_t estimate_layerwise_split_block_count(
    const ModelArgs& model_args,
    int32_t layerwise_split_size,
    const KVCacheCapacity& kv_cache_cap,
    int64_t available_bytes,
    int64_t additional_block_bytes);

KVCacheCapacity estimate_kv_cache_capacity(
    const ModelArgs& model_args,
    const KVCacheEstimateOptions& options);

}  // namespace xllm
