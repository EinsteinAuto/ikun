/* Migrated from xllm/core/framework/kv_cache/kv_cache_estimation.cpp (729 lines)
   [BI-V100 20%] Changes:
   - Removed torch dependency (dtype_size via switch on int32_t enum)
   - Removed Platform::supports_dsa_* calls (no DSA indexer on BI-V100)
   - Removed DeepSeek V4 estimation (not target model)
   - Removed MLU/NPU conditional compilation
   - Kept layerwise_split_block_count, build_layer_cache_owned,
     estimate_layerwise_split_block_count, init_standard_counts,
     estimate_kv_cache_capacity — the scheduling-layer functions */

#include "framework/kv_cache/kv_cache_estimation.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include <glog/logging.h>

#include "framework/kv_cache/layerwise_split_layout.h"
#include "framework/model/model_args.h"

namespace xllm {

namespace {

// [BI-V100 20%] Replaces torch::elementSize(dtype) — no libtorch needed.
int64_t dtype_size_from_enum(int32_t dtype_enum) {
  switch (dtype_enum) {
    case 5:  return 2;   // float16
    case 15: return 2;   // bfloat16
    case 6:  return 4;   // float32
    case 2:  return 1;   // int8
    default: return 2;
  }
}

// Upstream: kv_cache_dtype_size
int64_t kv_cache_dtype_size(const std::string& kv_cache_dtype,
                            int64_t model_dtype_size) {
  if (kv_cache_dtype == "auto") {
    return model_dtype_size;
  }
  if (kv_cache_dtype == "int8") {
    return 1;
  }
  if (kv_cache_dtype == "fp8_e4m3" || kv_cache_dtype == "fp8_e5m2") {
    return 1;
  }
  return model_dtype_size;
}

// Upstream: kv_slot_size
int64_t kv_slot_size(const ModelArgs& model_args,
                     const KVCacheEstimateOptions& options,
                     int64_t cache_dtype_size) {
  if (model_args.enable_mla()) {
    return cache_dtype_size *
           (model_args.kv_lora_rank() + model_args.qk_rope_head_dim());
  }
  return 2 * cache_dtype_size * model_args.head_dim() *
         options.n_local_kv_heads;
}

// Upstream: scale_slot_size
int64_t scale_slot_size(const ModelArgs& model_args,
                        const KVCacheEstimateOptions& options) {
  if (options.kv_cache_dtype == "auto") {
    return 0;
  }
  if (model_args.enable_mla()) {
    return sizeof(float);
  }
  return 2 * sizeof(float) * options.n_local_kv_heads;
}

// Upstream: standard_full_cache_block_size_in_bytes
int64_t standard_full_cache_block_size_in_bytes(
    const KVCacheCapacity& kv_cache_cap) {
  const int64_t full_attention_layers =
      std::max<int64_t>(kv_cache_cap.num_full_attention_layers(), 1);
  const int64_t indexer_layers = kv_cache_cap.num_indexer_layers();
  CHECK_GE(indexer_layers, 0) << "num_indexer_layers must be non-negative";
  CHECK_LE(indexer_layers, full_attention_layers)
      << "num_indexer_layers cannot exceed full-attention layers";
  const int64_t logical_block_bytes =
      kv_cache_cap.block_size() *
      (full_attention_layers *
           (kv_cache_cap.slot_size() + kv_cache_cap.scale_slot_size()) +
       indexer_layers * kv_cache_cap.index_slot_size());
  CHECK_GT(logical_block_bytes, 0) << "logical block bytes must be positive";
  return logical_block_bytes;
}

// Upstream: layerwise_split_block_count (lines 126-186)
int64_t layerwise_split_block_count(const ModelArgs& model_args,
                                    int32_t layerwise_split_size,
                                    const KVCacheCapacity& kv_cache_cap,
                                    int64_t available_bytes,
                                    int64_t additional_block_bytes) {
  const int64_t num_layers = kv_cache_cap.n_layers();
  const std::vector<bool> indexer_layer_mask =
      resolve_indexer_cache_enabled_layers(model_args, num_layers);

  std::vector<int64_t> layer_bytes(static_cast<size_t>(num_layers), 0);
  bool any_indexer_layer = false;
  for (int64_t layer_id = 0; layer_id < num_layers; ++layer_id) {
    if (!is_full_attention_layer(model_args, layer_id)) {
      continue;
    }
    const bool has_indexer =
        kv_cache_cap.index_slot_size() > 0 &&
        (indexer_layer_mask.empty() ||
         indexer_layer_mask[static_cast<size_t>(layer_id)]);
    any_indexer_layer = any_indexer_layer || has_indexer;
    const int64_t bytes =
        kv_cache_cap.block_size() *
        (kv_cache_cap.slot_size() + kv_cache_cap.scale_slot_size() +
         (has_indexer ? kv_cache_cap.index_slot_size() : 0));
    layer_bytes[static_cast<size_t>(layer_id)] = bytes;
  }

  // Scratch matches an owned layer, including indexer when any layer has one.
  const int64_t scratch_bytes_per_block =
      kv_cache_cap.block_size() *
      (kv_cache_cap.slot_size() + kv_cache_cap.scale_slot_size() +
       (any_indexer_layer ? kv_cache_cap.index_slot_size() : 0));
  CHECK_GT(scratch_bytes_per_block, 0);

  int64_t common_block_count = std::numeric_limits<int64_t>::max();
  for (int32_t split_rank = 0; split_rank < layerwise_split_size;
       ++split_rank) {
    const LayerwiseSplitLayout layout(
        /*enabled=*/true, layerwise_split_size, split_rank);
    const std::vector<bool> layer_cache_owned =
        build_layer_cache_owned(model_args, layout, num_layers);
    int64_t owned_bytes = 0;
    for (int64_t layer_id = 0; layer_id < num_layers; ++layer_id) {
      if (layer_cache_owned[static_cast<size_t>(layer_id)]) {
        owned_bytes += layer_bytes[static_cast<size_t>(layer_id)];
      }
    }
    if (owned_bytes == 0) {
      continue;
    }
    const int64_t per_block_bytes =
        owned_bytes + scratch_bytes_per_block + additional_block_bytes;
    common_block_count =
        std::min(common_block_count, available_bytes / per_block_bytes);
  }

  CHECK_NE(common_block_count, std::numeric_limits<int64_t>::max())
      << "No layerwise split rank owns a model layer.";
  CHECK_GT(common_block_count, 0) << "No memory for one layerwise split block.";
  return common_block_count;
}

// Upstream: enable_qwen3_5_spec_verify
bool enable_qwen3_5_spec_verify(const ModelArgs& model_args,
                                const KVCacheEstimateOptions& options) {
  return options.num_speculative_tokens > 0 && !options.is_draft_engine &&
         is_qwen3_5_target_model_type(model_args.model_type());
}

// Upstream: linear_slot_size
int64_t linear_slot_size(const ModelArgs& model_args,
                         const KVCacheEstimateOptions& options,
                         int64_t dtype_size) {
  if (model_args.linear_num_value_heads() <= 0) {
    return 0;
  }
  const int64_t num_speculative_tokens =
      enable_qwen3_5_spec_verify(model_args, options)
          ? options.num_speculative_tokens
          : 0;

  const int64_t head_k_dim = model_args.linear_key_head_dim();
  const int64_t head_v_dim = model_args.linear_value_head_dim();
  // [BI-V100 20%] ssm_dtype defaults to model dtype on BI-V100
  const int64_t ssm_dtype_size = dtype_size;

  const int64_t linear_ssm_slot_size =
      ssm_dtype_size * options.n_local_linear_v_heads * head_k_dim * head_v_dim;
  const int64_t linear_conv_state_len =
      model_args.linear_conv_kernel_dim() - 1 + num_speculative_tokens;
  const int64_t linear_conv_slot_size =
      dtype_size *
      (head_k_dim * options.n_local_linear_k_heads * 2 +
       head_v_dim * options.n_local_linear_v_heads) *
      linear_conv_state_len;
  return linear_conv_slot_size +
         linear_ssm_slot_size * (num_speculative_tokens + 1);
}

constexpr int64_t kPaddingLinearStateBlocks = 2;

// Upstream: calculate_linear_state_blocks (simplified — no prefix cache auto-sizing)
int64_t calculate_linear_state_blocks(int64_t cache_size_in_bytes,
                                      int64_t num_linear_attention_layers,
                                      int64_t linear_slot_sz,
                                      int64_t full_cache_block_size_in_bytes,
                                      int64_t max_seqs_per_batch,
                                      int64_t max_linear_state_cache_slots,
                                      bool enable_prefix_cache) {
  if (num_linear_attention_layers <= 0 || linear_slot_sz <= 0) {
    return kPaddingLinearStateBlocks;
  }
  const int64_t linear_bytes_per_block =
      num_linear_attention_layers * linear_slot_sz;
  CHECK_GT(linear_bytes_per_block, 0);

  int64_t max_blocks =
      (cache_size_in_bytes - 1) / linear_bytes_per_block;
  const int64_t balanced =
      (cache_size_in_bytes +
       kPaddingLinearStateBlocks * full_cache_block_size_in_bytes) /
      (linear_bytes_per_block + full_cache_block_size_in_bytes);
  max_blocks = std::min(max_blocks, balanced);
  max_blocks = std::max(max_blocks, kPaddingLinearStateBlocks);

  if (max_linear_state_cache_slots > 0) {
    const int64_t requested =
        max_linear_state_cache_slots + kPaddingLinearStateBlocks;
    CHECK_LE(requested, max_blocks)
        << "max_linear_state_cache_slots requires " << requested
        << " linear-state blocks, but only " << max_blocks << " fit.";
    return requested;
  }

  if (!enable_prefix_cache) {
    const int64_t live =
        max_seqs_per_batch + kPaddingLinearStateBlocks;
    return std::max<int64_t>(std::min(live, max_blocks),
                             kPaddingLinearStateBlocks);
  }

  // Auto-size with prefix cache (upstream ratio 0.9)
  constexpr double kRatio = 0.9;
  const double frac = kRatio / (1.0 + kRatio);
  int64_t auto_blocks = std::max<int64_t>(
      static_cast<int64_t>(cache_size_in_bytes * frac / linear_bytes_per_block),
      kPaddingLinearStateBlocks);
  return std::min(auto_blocks, max_blocks);
}

// Upstream: init_standard_counts (lines 483-550)
void init_standard_counts(const ModelArgs& model_args,
                          const KVCacheEstimateOptions& options,
                          KVCacheCapacity* kv_cache_cap) {
  for (int64_t layer_id = 0; layer_id < kv_cache_cap->n_layers(); ++layer_id) {
    if (is_full_attention_layer(model_args, layer_id)) {
      ++kv_cache_cap->num_full_attention_layers();
    } else {
      ++kv_cache_cap->num_linear_attention_layers();
    }
  }

  // [BI-V100 20%] No DSA indexer on BI-V100 — indexer_layers stays 0.

  const int64_t full_cache_block_size_in_bytes =
      standard_full_cache_block_size_in_bytes(*kv_cache_cap);
  kv_cache_cap->num_linear_state_blocks(
      calculate_linear_state_blocks(kv_cache_cap->cache_size_in_bytes(),
                                    kv_cache_cap->num_linear_attention_layers(),
                                    kv_cache_cap->linear_slot_size(),
                                    full_cache_block_size_in_bytes,
                                    options.max_seqs_per_batch,
                                    options.max_linear_state_cache_slots,
                                    options.enable_prefix_cache));
  kv_cache_cap->linear_cache_size_in_bytes(
      kv_cache_cap->num_linear_attention_layers() *
      kv_cache_cap->num_linear_state_blocks() *
      kv_cache_cap->linear_slot_size());
  const int64_t available_full_cache_size_in_bytes =
      kv_cache_cap->cache_size_in_bytes() -
      kv_cache_cap->linear_cache_size_in_bytes();
  if (kv_cache_cap->linear_slot_size() > 0) {
    CHECK_GT(kv_cache_cap->cache_size_in_bytes(),
             kv_cache_cap->linear_cache_size_in_bytes())
        << "failed to reserve linear state cache";
  }
  CHECK_GT(available_full_cache_size_in_bytes, 0)
      << "no memory left for full-attention kv cache";
  if (options.layerwise_split_size > 1) {
    kv_cache_cap->n_blocks(
        layerwise_split_block_count(model_args,
                                    options.layerwise_split_size,
                                    *kv_cache_cap,
                                    available_full_cache_size_in_bytes,
                                    /*additional_block_bytes=*/0));
  } else {
    kv_cache_cap->n_blocks(available_full_cache_size_in_bytes /
                           full_cache_block_size_in_bytes);
  }
  CHECK_GT(kv_cache_cap->n_blocks(), 0) << "no n_blocks for kv cache";
}

}  // namespace

// [BI-V100 20%] No DSA indexer on BI-V100 — returns empty mask.
std::vector<bool> resolve_indexer_cache_enabled_layers(
    const ModelArgs& /*model_args*/,
    int64_t /*num_cache_layers*/) {
  return {};
}

// Upstream: build_layer_cache_owned (lines 563-574)
std::vector<bool> build_layer_cache_owned(const ModelArgs& model_args,
                                          const LayerwiseSplitLayout& layout,
                                          int64_t num_layers) {
  std::vector<bool> layer_cache_owned;
  layer_cache_owned.reserve(static_cast<size_t>(num_layers));
  for (int64_t layer_id = 0; layer_id < num_layers; ++layer_id) {
    layer_cache_owned.emplace_back(
        !is_full_attention_layer(model_args, layer_id) ||
        layout.owns(layer_id));
  }
  return layer_cache_owned;
}

// Upstream: estimate_layerwise_split_block_count (lines 576-587)
int64_t estimate_layerwise_split_block_count(
    const ModelArgs& model_args,
    int32_t layerwise_split_size,
    const KVCacheCapacity& kv_cache_cap,
    int64_t available_bytes,
    int64_t additional_block_bytes) {
  return layerwise_split_block_count(model_args,
                                     layerwise_split_size,
                                     kv_cache_cap,
                                     available_bytes,
                                     additional_block_bytes);
}

// Upstream: estimate_kv_cache_capacity (lines 589-641, stripped DSV4 path)
KVCacheCapacity estimate_kv_cache_capacity(
    const ModelArgs& model_args,
    const KVCacheEstimateOptions& options) {
  KVCacheCapacity kv_cache_cap;
  kv_cache_cap
      .cache_size_in_bytes(
          std::max(options.cache_size_in_bytes, static_cast<int64_t>(0)))
      .block_size(options.block_size);
  CHECK_GT(kv_cache_cap.cache_size_in_bytes(), 0)
      << "Available kv cache size must be greater than 0";

  // [BI-V100 20%] dtype_size via enum, not torch
  const int64_t dtype_size = dtype_size_from_enum(options.dtype_enum);
  const int64_t cache_dtype_size =
      kv_cache_dtype_size(options.kv_cache_dtype, dtype_size);

  kv_cache_cap.slot_size(kv_slot_size(model_args, options, cache_dtype_size))
      .scale_slot_size(scale_slot_size(model_args, options))
      .linear_slot_size(linear_slot_size(model_args, options, dtype_size))
      .n_layers(model_args.n_layers())
      .block_size(options.block_size);

  const int64_t num_speculative_tokens =
      enable_qwen3_5_spec_verify(model_args, options)
          ? options.num_speculative_tokens
          : 0;
  kv_cache_cap.linear_conv_state_len(model_args.linear_conv_kernel_dim() - 1 +
                                     num_speculative_tokens);
  kv_cache_cap.linear_ssm_checkpoint_stride(num_speculative_tokens + 1);

  init_standard_counts(model_args, options, &kv_cache_cap);
  return kv_cache_cap;
}

}  // namespace xllm
