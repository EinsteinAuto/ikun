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
// Memory estimation for layerwise-split KV cache.
//
// BI-V100 verified: 32768 MiB HBM per card (ixsmi -q -d MEMORY),
// baseline usage ~257 MiB idle.  Available for KV cache ≈ 32511 MiB
// minus model weights and activations.
//
// Qwen3.5 (Qwen3.6-35B-A3B) KV cache per full-attention layer per rank:
//   key:   n_blocks * 1 * 32 * 16 * 8 * 2 bytes (fp16) = n_blocks * 8192 B
//   value: n_blocks * 1 * 256 * 16 * 2 bytes (fp16)     = n_blocks * 8192 B
//   total per layer per rank: n_blocks * 16384 bytes
//
// With layerwise split (group_size=2), each rank owns ~half the
// full-attention layers, so peak KV memory drops by ~50% minus
// scratch overhead.

#include "framework/kv_cache/kv_cache_estimation_layerwise.h"
#include "framework/kv_cache/layerwise_split_layout.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#include "config/ilu_hw_constants.h"

namespace xllm {

namespace {

/// Bytes per KV element for a given dtype.
int64_t dtype_bytes(int dtype_enum) {
  // torch::kBFloat16 = 15, torch::kHalf = 5, torch::kFloat = 6
  switch (dtype_enum) {
    case 5:  return 2;   // float16
    case 15: return 2;   // bfloat16
    case 6:  return 4;   // float32
    case 2:  return 1;   // int8
    default: return 2;   // conservative
  }
}

/// Round up to next multiple of |align|.
inline int64_t align_up(int64_t val, int64_t align) {
  return ((val + align - 1) / align) * align;
}

}  // namespace

LayerwiseKVMemoryEstimate estimate_layerwise_kv_memory(
    const IluLayerwiseLayout& layout,
    int64_t n_blocks,
    int64_t block_size,
    int64_t head_dim,
    int64_t max_tokens,
    int dtype_enum,
    int32_t world_size) {
  CHECK_GT(layout.num_layers(), 0);
  CHECK_GT(world_size, 0);

  const int64_t elem_bytes = dtype_bytes(dtype_enum);

  // BI-V100 warp = 64: pad head_dim to the next multiple of 64
  // so each warp's contiguous load spans an aligned region.
  // Qwen3.5 head_dim=256 → 256/64=4 warps, already aligned.
#if defined(USE_ILU)
  const int64_t padded_head_dim = align_up(head_dim, ilu_hw::kWarpSize);
#else
  const int64_t padded_head_dim = head_dim;
#endif

  // Per-rank KV bytes: sum over layers of (2 * heads * n_blocks *
  // block_size * padded_head_dim * elem_bytes).  Factor 2 = K + V.
  std::vector<int64_t> per_rank_bytes(world_size, 0);
  for (int64_t lid = 0; lid < layout.num_layers(); ++lid) {
    const auto& spec = layout.layer_spec(lid);
    for (size_t i = 0; i < spec.assigned_ranks.size(); ++i) {
      int32_t rank = spec.assigned_ranks[i];
      int64_t heads = spec.heads_per_rank[i];
      int64_t layer_bytes = 2 * heads * n_blocks * block_size *
                            padded_head_dim * elem_bytes;
      CHECK_GE(rank, 0);
      CHECK_LT(rank, world_size);
      per_rank_bytes[rank] += layer_bytes;
    }
  }

  // Uniform baseline (also with padding for fair comparison).
  int64_t uniform_total = 0;
  for (const auto& s : layout.specs())
    uniform_total += s.total_heads();
  int64_t uniform_per_rank =
      2 * (uniform_total / world_size) * n_blocks * block_size *
      padded_head_dim * elem_bytes;

  int64_t peak = *std::max_element(per_rank_bytes.begin(),
                                    per_rank_bytes.end());
  int64_t sum  = std::accumulate(per_rank_bytes.begin(),
                                  per_rank_bytes.end(), int64_t{0});
  double average = static_cast<double>(sum) / world_size;

  LayerwiseKVMemoryEstimate est;
  est.peak_per_rank_bytes     = peak;
  est.average_per_rank_bytes  = static_cast<int64_t>(average);
  est.uniform_per_rank_bytes  = uniform_per_rank;
  est.per_rank_bytes          = std::move(per_rank_bytes);
  est.savings_vs_uniform_pct  =
      uniform_per_rank > 0
          ? 100.0 * (1.0 - static_cast<double>(peak) / uniform_per_rank)
          : 0.0;

  LOG(INFO) << "[LayerwiseSplit] KV memory estimate: peak="
            << (peak >> 20) << " MiB, avg="
            << (static_cast<int64_t>(average) >> 20) << " MiB, uniform="
            << (uniform_per_rank >> 20) << " MiB, saving="
            << est.savings_vs_uniform_pct << "%";

  return est;
}

int64_t estimate_layerwise_split_block_count(
    int32_t layerwise_split_size,
    int64_t num_full_attn_layers,
    int64_t per_layer_block_bytes,
    int64_t scratch_block_bytes,
    int64_t available_bytes) {
  CHECK_GE(layerwise_split_size, 1);
  CHECK_GT(num_full_attn_layers, 0);
  CHECK_GT(per_layer_block_bytes, 0);
  CHECK_GT(scratch_block_bytes, 0);

  // Each split rank owns ceil(num_full_attn_layers / group_size) layers.
  // The block count is limited by the rank with the most owned layers.
  int64_t common_block_count = std::numeric_limits<int64_t>::max();
  for (int32_t split_rank = 0; split_rank < layerwise_split_size;
       ++split_rank) {
    const LayerwiseSplitLayout layout(
        /*enabled=*/true, layerwise_split_size, split_rank);
    int64_t owned_bytes = 0;
    for (int64_t lid = 0; lid < num_full_attn_layers; ++lid) {
      if (layout.owns(lid)) {
        owned_bytes += per_layer_block_bytes;
      }
    }
    if (owned_bytes == 0) {
      continue;
    }
    const int64_t per_block_bytes = owned_bytes + scratch_block_bytes;
    common_block_count =
        std::min(common_block_count, available_bytes / per_block_bytes);
  }

  if (common_block_count == std::numeric_limits<int64_t>::max()) {
    common_block_count = available_bytes / scratch_block_bytes;
  }

  LOG(INFO) << "[LayerwiseSplit] Block count estimate: "
            << common_block_count << " blocks (split_size="
            << layerwise_split_size << ", attn_layers="
            << num_full_attn_layers << ")";

  return common_block_count;
}

}  // namespace xllm
