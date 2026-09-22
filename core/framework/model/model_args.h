/* Minimal ModelArgs for project_6 layerwise split.
   Migrated from xllm/core/framework/model/model_args.h — only the fields
   and functions used by kv_cache_estimation.cpp's layerwise_split code path.

   Full ModelArgs in upstream is 861 lines; we keep only the interface that
   the layerwise split scheduling layer needs. */

#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "common/macros.h"

namespace xllm {

class ModelArgs {
  PROPERTY(std::string, model_type);
  PROPERTY(int64_t, n_layers) = 0;
  PROPERTY(int64_t, head_dim) = 0;
  PROPERTY(int64_t, n_heads) = 0;
  PROPERTY(std::optional<int64_t>, n_kv_heads);
  PROPERTY(int64_t, hidden_size) = 0;

  // Linear attention (GDN/DeltaNet) fields — Qwen3.5
  PROPERTY(int64_t, full_attention_interval) = 1;
  PROPERTY(std::vector<std::string>, layer_types);
  PROPERTY(int64_t, linear_num_value_heads) = 0;
  PROPERTY(int64_t, linear_key_head_dim) = 0;
  PROPERTY(int64_t, linear_value_head_dim) = 0;
  PROPERTY(int64_t, linear_conv_kernel_dim) = 0;
  PROPERTY(std::string, mamba_ssm_dtype);

  // MLA (DeepSeek)
  PROPERTY(bool, enable_mla) = false;
  PROPERTY(int64_t, kv_lora_rank) = 0;
  PROPERTY(int64_t, qk_rope_head_dim) = 0;

  // DSA indexer
  PROPERTY(int64_t, index_n_heads) = 0;
  PROPERTY(int64_t, index_head_dim) = 0;

  // Sliding window
  PROPERTY(int64_t, window_size) = 0;
  PROPERTY(int64_t, max_seq_len) = 0;

  // DeepSeek V4 compress ratios
  PROPERTY(std::vector<int32_t>, compress_ratios);

  // Speculative
  PROPERTY(int64_t, num_nextn_predict_layers) = 0;
};

// Qwen hybrid models may describe full-attention layers explicitly via
// layer_types or implicitly via full_attention_interval.
inline bool is_full_attention_layer(const ModelArgs& args, int64_t layer_id) {
  const auto& hybrid_layer_types = args.layer_types();
  if (layer_id >= 0 &&
      layer_id < static_cast<int64_t>(hybrid_layer_types.size())) {
    const auto& layer_type = hybrid_layer_types[layer_id];
    return layer_type == "full_attention" || layer_type == "attention";
  }

  int32_t attention_interval = args.full_attention_interval();
  if (attention_interval <= 1) {
    return true;
  }
  return (layer_id + 1) % attention_interval == 0;
}

inline bool has_linear_attention_layers(const ModelArgs& args) {
  const auto& hybrid_layer_types = args.layer_types();
  if (!hybrid_layer_types.empty()) {
    return std::any_of(hybrid_layer_types.begin(),
                       hybrid_layer_types.end(),
                       [](const std::string& layer_type) {
                         return layer_type != "full_attention" &&
                                layer_type != "attention";
                       });
  }
  return args.full_attention_interval() > 1;
}

// Closed set by design: a new target variant must be enumerated here rather
// than matched by a "qwen3_5_" prefix.
inline bool is_qwen3_5_target_model_type(std::string_view model_type) {
  return model_type == "qwen3_5" || model_type == "qwen3_5_moe" ||
         model_type == "qwen3_5_text" || model_type == "qwen3_5_moe_text";
}

}  // namespace xllm
