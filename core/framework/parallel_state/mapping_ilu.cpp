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
// ILU-specific device-to-layer mapping for layerwise split KV cache.
//
// Verified Iluvatar BI-V100 topology (ixsmi topo -m):
//   - 4 cards: GPU0-GPU3, Bus-Id 4B:00.0 – 4E:00.0, all on NUMA node 1
//   - All pairs connected via PIX (single PCIe bridge) — FLAT topology
//   - No NVLink, no HCCS mesh, no multi-switch hierarchy
//   - 32 GB HBM per card (32768 MiB), baseline ~257 MiB
//   - 1500 MHz SM, 1200 MHz mem
//   - CPU affinity: 16-31,80-95
//   - Warp size: 64 (verified via BI-V150 docs, same ivcore architecture)
//   - IX-ML 3.2.3, Driver 3.2.1, CUDA 10.2 (CoreX)
//   - CoreX SDK at /usr/local/corex/
//
// Model: Qwen3.6-35B-A3B (Qwen3_5 MoE architecture)
//   - Interleaved full_attention + linear_attention layers
//   - Only full_attention layers have KV cache
//   - num_kv_heads=4, with TP=4: local_kv_heads=1 per rank per layer
//   - head_dim=256, GQA ratio=4
//
// Strategy (flat PIX topology):
//   All full-attention layers have kv_heads=4 >= world_size=4,
//   so every layer shards across ALL ranks (each gets 1 head).
//   If kv_heads < world_size, round-robin starting rank to spread
//   HBM load (no grouping benefit since all PIX links are equal).

#include "framework/parallel_state/mapping_ilu.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include "framework/kv_cache/ilu_layerwise_layout.h"
#include "framework/kv_cache/layerwise_split_layout.h"

namespace xllm {

IluLayerwiseLayout compute_ilu_layerwise_layout(
    int64_t num_layers,
    const std::vector<int64_t>& per_layer_kv_heads,
    int32_t world_size,
    IluTopoKind topo_kind) {
  CHECK_EQ(static_cast<int64_t>(per_layer_kv_heads.size()), num_layers);
  CHECK_GT(world_size, 0);

  std::vector<LayerShardSpec> specs;
  specs.reserve(num_layers);

  // For layers with fewer heads than ranks, we round-robin the starting
  // rank so that different layers land on different subsets, balancing HBM
  // pressure across the flat PIX topology.
  int32_t rr_offset = 0;

  for (int64_t lid = 0; lid < num_layers; ++lid) {
    LayerShardSpec spec;
    spec.layer_id = lid;
    const int64_t total_heads = per_layer_kv_heads[lid];

    if (total_heads >= world_size) {
      // Dense case (Qwen3.5 full_attention with TP=4, 4 heads → 1 each):
      // shard across all ranks.
      for (int32_t r = 0; r < world_size; ++r)
        spec.assigned_ranks.push_back(r);
      int64_t base = total_heads / world_size;
      int64_t rem  = total_heads % world_size;
      for (int32_t r = 0; r < world_size; ++r)
        spec.heads_per_rank.push_back(base + (r < rem ? 1 : 0));
    } else {
      // Fewer heads than ranks: round-robin subset.
      // (Not the case for Qwen3.5 with TP=4, but needed for other models.)
      int32_t needed = static_cast<int32_t>(total_heads);
      for (int32_t j = 0; j < needed; ++j) {
        int32_t rank = (rr_offset + j) % world_size;
        spec.assigned_ranks.push_back(rank);
      }
      int64_t base = total_heads / needed;
      int64_t rem  = total_heads % needed;
      for (int32_t j = 0; j < needed; ++j)
        spec.heads_per_rank.push_back(base + (j < rem ? 1 : 0));
      rr_offset = (rr_offset + needed) % world_size;
    }
    specs.push_back(std::move(spec));
  }

  LOG(INFO) << "[LayerwiseSplit] ILU layout computed: " << num_layers
            << " layers, " << world_size << " ranks, topo="
            << (topo_kind == IluTopoKind::kFlatPIX ? "flat_PIX" : "grouped");

  return IluLayerwiseLayout(std::move(specs));
}

}  // namespace xllm
