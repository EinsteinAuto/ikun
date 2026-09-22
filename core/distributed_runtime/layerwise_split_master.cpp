/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

// Commit: 494f293b5629 · feat · PR #2260  (adapted for Iluvatar BI-V100)
// Master-side orchestration: at startup the master reads model config to
// extract per-layer KV head counts, computes the layout, and stores it
// for distribution to workers.
//
// For Qwen3.5 (Qwen3.6-35B-A3B):
//   - layer_types is a mix of "full_attention" and "linear_attention"
//   - Only full_attention layers have KV cache
//   - Each full_attention layer has the same KV head count (4 for the
//     35B-A3B variant)

#include "distributed_runtime/layerwise_split_master.h"

#include <glog/logging.h>

#include <optional>
#include <string>
#include <vector>

#include "distributed_runtime/layerwise_split_engine_ext.h"
#include "framework/kv_cache/kv_cache_estimation_layerwise.h"
#include "framework/kv_cache/layerwise_split_layout.h"

DECLARE_bool(enable_layerwise_split);

namespace xllm {

std::optional<IluLayerwiseLayout> master_compute_layerwise_layout(
    const std::vector<std::string>& layer_types,
    int64_t kv_heads_per_full_attn_layer,
    int32_t world_size,
    int64_t n_blocks,
    int64_t block_size,
    int64_t head_dim,
    int64_t max_tokens,
    int dtype_enum) {
  if (!FLAGS_enable_layerwise_split) {
    LOG(INFO) << "[LayerwiseSplit] Disabled; using uniform KV sharding.";
    return std::nullopt;
  }

  // Count and build per-layer KV head count vector for full-attention
  // layers only.  Linear-attention layers are skipped.
  int64_t full_attn_count = 0;
  for (const auto& lt : layer_types) {
    if (lt == "full_attention") {
      ++full_attn_count;
    }
  }
  CHECK_GT(full_attn_count, 0) << "No full_attention layers found.";

  std::vector<int64_t> per_layer_heads(full_attn_count,
                                        kv_heads_per_full_attn_layer);

  auto layout = maybe_compute_layerwise_layout(
      full_attn_count, per_layer_heads, world_size);

  if (layout.has_value()) {
    // Run estimation for logging / capacity planning.
    auto est = estimate_layerwise_kv_memory(
        *layout, n_blocks, block_size, head_dim, max_tokens,
        dtype_enum, world_size);

    LOG(INFO) << "[LayerwiseSplit] Qwen3.5 layout: "
              << full_attn_count << " full-attn layers, "
              << (layer_types.size() - full_attn_count) << " linear-attn layers"
              << " — Peak per-rank KV: "
              << (est.peak_per_rank_bytes >> 20) << " MiB  (uniform would be "
              << (est.uniform_per_rank_bytes >> 20) << " MiB, saving "
              << est.savings_vs_uniform_pct << "%)";
  }

  return layout;
}

}  // namespace xllm
