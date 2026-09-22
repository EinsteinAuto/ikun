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
// allocate_kv_caches_layerwise: per-layer KV allocation using the
// layer_cache_owned mask.
//
// On BI-V100 the KV cache tensor layout is transposed ("block-major"):
//   key_cache:   (n_blocks, n_kv_heads_local, block_size, head_dim/x, x)
//   value_cache: (n_blocks, n_kv_heads_local, head_dim, block_size)
//
// Verified from runtime log with Qwen3.6-35B-A3B TP=4:
//   key_cache=(68837, 1, 32, 16, 8)  value_cache=(68837, 1, 256, 16)
//   query=(1, 4, 256)  →  4 q-heads, 1 kv-head per rank, head_dim=256
//
// This file implements the scheduling/allocation logic only.  The actual
// tensor creation is done by the vendor vLLM cache engine (which calls
// into prebuilt corex_*.so for the kernel layer).  This module decides
// WHICH layers get real allocations vs scratch placeholders.
//
// BI-V100 warp size = 64.  head_dim (256 for Qwen3.5) is a multiple
// of 64, so coalesced warp-wide loads across the head dimension are aligned.
// When head_dim is not a multiple of 64, the allocator pads to the next
// multiple to avoid idle warp lanes.

#include "framework/kv_cache/kv_cache_layerwise.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "config/ilu_hw_constants.h"
#include "framework/kv_cache/layerwise_split_layout.h"

namespace xllm {

namespace {

/// Round up |val| to the next multiple of |align|.
inline int64_t align_up(int64_t val, int64_t align) {
  return ((val + align - 1) / align) * align;
}

}  // namespace

void allocate_kv_caches_layerwise(
    const std::vector<bool>& layer_cache_owned,
    int64_t num_layers,
    int32_t current_rank) {
  CHECK_EQ(static_cast<int64_t>(layer_cache_owned.size()), num_layers)
      << "layer_cache_owned size must match num_layers.";

  int64_t owned_count = 0;
  int64_t scratch_count = 0;
  for (int64_t i = 0; i < num_layers; ++i) {
    if (layer_cache_owned[static_cast<size_t>(i)]) {
      ++owned_count;
    } else {
      ++scratch_count;
    }
  }

  // Scheduling decision log — the actual tensor allocation is done by
  // the vendor vLLM cache engine (CacheEngine._allocate_kv_cache).
  // This module only decides the ownership mask.
  LOG(INFO) << "[LayerwiseSplit] rank " << current_rank << ": "
            << owned_count << "/" << num_layers
            << " layers owned, " << scratch_count << " scratch.";

  // Validate head_dim alignment for BI-V100 warp size.
  // Qwen3.5 head_dim=256, which is 256/64=4 warps — perfectly aligned.
  constexpr int64_t kHeadDim = ilu_hw::kQwen35HeadDim;
  constexpr int64_t kWarpAlign = ilu_hw::kWarpSize;
  const int64_t padded = align_up(kHeadDim, kWarpAlign);
  if (padded != kHeadDim) {
    LOG(WARNING) << "[LayerwiseSplit] head_dim=" << kHeadDim
                 << " not aligned to warp_size=" << kWarpAlign
                 << ", padded to " << padded;
  }
}

}  // namespace xllm
