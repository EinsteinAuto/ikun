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

#pragma once

#include <cstdint>
#include <vector>

#include "framework/kv_cache/ilu_layerwise_layout.h"

namespace xllm {

struct LayerwiseKVMemoryEstimate {
  int64_t peak_per_rank_bytes    = 0;  // worst-case rank
  int64_t average_per_rank_bytes = 0;
  int64_t uniform_per_rank_bytes = 0;  // baseline (uniform sharding)
  std::vector<int64_t> per_rank_bytes;  // detailed per-rank breakdown
  double savings_vs_uniform_pct  = 0.0;
};

/// Estimate per-rank KV cache memory for a layerwise-split layout.
///
/// This uses the IluLayerwiseLayout (BI-V100-specific detailed layout)
/// to compute exact per-rank memory accounting for the transposed
/// block-major tensor layout:
///   key:   n_blocks * n_heads * block_size * head_dim * elem_bytes
///   value: n_blocks * n_heads * head_dim * block_size * elem_bytes
///   (both are equivalent in total bytes; the axes are just reordered)
///
/// |dtype_enum| matches torch::ScalarType integer values.
LayerwiseKVMemoryEstimate estimate_layerwise_kv_memory(
    const IluLayerwiseLayout& layout,
    int64_t n_blocks,
    int64_t block_size,
    int64_t head_dim,
    int64_t max_tokens,
    int dtype_enum,
    int32_t world_size);

/// Estimate per-rank block count using the upstream-compatible
/// LayerwiseSplitLayout (round-robin ownership).
/// Returns the minimum block count that any rank can afford.
int64_t estimate_layerwise_split_block_count(
    int32_t layerwise_split_size,
    int64_t num_full_attn_layers,
    int64_t per_layer_block_bytes,
    int64_t scratch_block_bytes,
    int64_t available_bytes);

}  // namespace xllm
