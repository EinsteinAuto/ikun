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

/// Topology kind for Iluvatar BI-V100 device mapping.
/// Verified via `ixsmi topo -m` on actual hardware.
enum class IluTopoKind : int8_t {
  /// All cards connected via PIX (single PCIe bridge).  All inter-card
  /// bandwidth is equal — no grouping benefit.
  /// Observed on: 4× BI-V100, Bus-Id 4B-4E, NUMA 1.
  kFlatPIX = 0,

  /// Cards grouped by PCIe switch (e.g. PXB/PHB between groups).
  /// Use when `ixsmi topo` shows mixed PIX + PXB/PHB/SYS entries.
  kGrouped = 1,
};

/// Compute a layerwise-split IluLayerwiseLayout for Iluvatar BI-V100.
///
/// |per_layer_kv_heads|: total KV head count for each full-attention layer.
///   For Qwen3.5 with TP=4: each full_attention layer has 4 total KV heads,
///   so local_kv_heads = 4/4 = 1 per rank.
///   Linear-attention layers are NOT included in this vector — they have no
///   KV cache.
///
/// Dense attention layers (heads >= world_size) spread across ALL TP ranks.
/// Layers with fewer heads than ranks are round-robin distributed.
///
/// Default: kFlatPIX — matches the verified 4-card BI-V100 topology
/// where all pairs are PIX-connected with equal bandwidth.
IluLayerwiseLayout compute_ilu_layerwise_layout(
    int64_t num_layers,
    const std::vector<int64_t>& per_layer_kv_heads,
    int32_t world_size,
    IluTopoKind topo_kind = IluTopoKind::kFlatPIX);

}  // namespace xllm
