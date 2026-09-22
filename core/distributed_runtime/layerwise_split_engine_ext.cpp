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
// Engine-level plumbing:  Both llm_engine and speculative_engine propagate
// the layerwise layout to workers during initialisation.
//
// In the upstream xLLM, this would be edits to llm_engine.cpp (+18 lines)
// and speculative_engine.cpp (+12 lines).  Here we isolate them in a
// self-contained compilation unit that the engines call into.
//
// The vendor vLLM on BI-V100 uses prebuilt CoreX .so extensions
// (corex_paged_kv_gather.so, etc.) for the kernel layer.  This module
// operates at the scheduling layer above — deciding which layers each
// rank owns for KV cache purposes.

#include "distributed_runtime/layerwise_split_engine_ext.h"

#include <glog/logging.h>

#include <memory>
#include <optional>
#include <vector>

#include "framework/kv_cache/layerwise_split_layout.h"
#include "framework/parallel_state/mapping_ilu.h"

// The flag is declared in parallel_config_layerwise.h / .cpp (sub-task 7).
DECLARE_bool(enable_layerwise_split);

namespace xllm {

std::optional<IluLayerwiseLayout> maybe_compute_layerwise_layout(
    int64_t num_layers,
    const std::vector<int64_t>& per_layer_kv_heads,
    int32_t world_size) {
  if (!FLAGS_enable_layerwise_split) {
    return std::nullopt;
  }

  LOG(INFO) << "[LayerwiseSplit] Computing layout for " << num_layers
            << " full-attention layers, world_size=" << world_size;

  // Iluvatar BI-V100: verified 4-card flat PIX topology (ixsmi topo -m).
  // All pairs connected via single PCIe bridge, equal bandwidth.
  return compute_ilu_layerwise_layout(
      num_layers, per_layer_kv_heads, world_size,
      IluTopoKind::kFlatPIX);
}

}  // namespace xllm
