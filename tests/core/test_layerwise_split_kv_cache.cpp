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

// Test suite for layerwise split KV cache sharding (PR #2260).
// Adapted for Iluvatar BI-V100 running Qwen3.6-35B-A3B (Qwen3_5 arch).
//
// Verified hardware:
//   4× BI-V100, Bus-Id 4B-4E, NUMA 1, flat PIX topology (all pairs PIX).
//   32768 MiB HBM each, IX-ML 3.2.3, Driver 3.2.1, CUDA 10.2 (CoreX).
//   Warp size: 64 (ivcore architecture).
//
// Model: Qwen3.6-35B-A3B
//   num_attention_heads=16, num_key_value_heads=4, head_dim=256
//   layer_types: interleaved full_attention + linear_attention
//   With TP=4: local_q_heads=4, local_kv_heads=1, GQA=4
//   KV cache shape: key=(n,1,32,16,8) value=(n,1,256,16)

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

#include "config/ilu_hw_constants.h"
#include "config/parallel_config_layerwise.h"
#include "distributed_runtime/layerwise_split_engine_ext.h"
#include "distributed_runtime/layerwise_split_master.h"
#include "framework/kv_cache/kv_cache_estimation_layerwise.h"
#include "framework/kv_cache/layerwise_split_layout.h"
#include "framework/model/model_args.h"
#include "framework/kv_cache/kv_cache_estimation.h"
#include "framework/kv_cache/ilu_layerwise_layout.h"
#include "framework/parallel_state/mapping_ilu.h"
#include "runtime/worker_layerwise_init.h"

namespace xllm {
namespace {

// Matches verified hardware: 4 BI-V100 cards.
constexpr int32_t kBIV100WorldSize = 4;
// Qwen3.5 model constants.
constexpr int64_t kQwen35KVHeads = ilu_hw::kQwen35NumKVHeads;  // 4
constexpr int64_t kQwen35HeadDim = ilu_hw::kQwen35HeadDim;     // 256

// Helper: build a realistic Qwen3.5 layer_types vector.
// The actual pattern alternates full_attention and linear_attention.
std::vector<std::string> make_qwen35_layer_types(int64_t num_layers) {
  std::vector<std::string> types;
  types.reserve(static_cast<size_t>(num_layers));
  for (int64_t i = 0; i < num_layers; ++i) {
    // Qwen3.5 pattern: every other layer is linear_attention
    types.push_back(i % 2 == 0 ? "full_attention" : "linear_attention");
  }
  return types;
}

// ---------------------------------------------------------------------------
// TC-01  Layerwise KV allocation correctness (Qwen3.5 parameters)
// ---------------------------------------------------------------------------
TEST(LayerwiseSplitKV, TC01_AllocationCorrectness) {
  // 16 full-attention layers (Qwen3.5 has interleaved layers, but the
  // IluLayerwiseLayout only tracks full-attention layers).
  const int64_t num_full_attn = 16;

  std::vector<int64_t> per_layer_heads(num_full_attn, kQwen35KVHeads);

  auto layout = compute_ilu_layerwise_layout(
      num_full_attn, per_layer_heads, kBIV100WorldSize);

  ASSERT_EQ(layout.num_layers(), num_full_attn);

  // Qwen3.5 with TP=4: kv_heads=4 = world_size → each rank gets 1 head.
  for (int64_t lid = 0; lid < num_full_attn; ++lid) {
    const auto& spec = layout.layer_spec(lid);
    EXPECT_EQ(static_cast<int32_t>(spec.assigned_ranks.size()),
              kBIV100WorldSize);
    for (int32_t r = 0; r < kBIV100WorldSize; ++r) {
      EXPECT_EQ(layout.heads_for_rank(r, lid), 1)
          << "Each rank should have exactly 1 KV head for Qwen3.5 TP=4";
    }
  }

  // Total heads should match.
  int64_t total = 0;
  for (int64_t lid = 0; lid < num_full_attn; ++lid)
    total += layout.layer_spec(lid).total_heads();
  EXPECT_EQ(total, num_full_attn * kQwen35KVHeads);
}

// ---------------------------------------------------------------------------
// TC-02  Memory estimation accuracy
// ---------------------------------------------------------------------------
TEST(LayerwiseSplitKV, TC02_MemoryEstimation) {
  const int64_t num_full_attn = 16;
  const int64_t n_blocks    = 68837;  // from real runtime log
  const int64_t block_size  = 16;     // from real runtime log
  const int64_t head_dim    = kQwen35HeadDim;  // 256
  const int64_t max_tokens  = 262144;
  const int dtype_enum      = 5;  // float16 (from real runtime)

  std::vector<int64_t> per_layer_heads(num_full_attn, kQwen35KVHeads);

  auto layout = compute_ilu_layerwise_layout(
      num_full_attn, per_layer_heads, kBIV100WorldSize);

  auto est = estimate_layerwise_kv_memory(
      layout, n_blocks, block_size, head_dim, max_tokens, dtype_enum,
      kBIV100WorldSize);

  // With uniform sharding (kv_heads == world_size), layerwise split
  // doesn't save memory — all ranks own all layers with 1 head each.
  // Peak should equal uniform in this case.
  EXPECT_EQ(est.peak_per_rank_bytes, est.uniform_per_rank_bytes);
  EXPECT_GT(est.peak_per_rank_bytes, 0);
  EXPECT_GT(est.average_per_rank_bytes, 0);

  // Per-rank breakdown should have exactly 4 entries.
  EXPECT_EQ(static_cast<int32_t>(est.per_rank_bytes.size()),
            kBIV100WorldSize);

  // All ranks should have equal memory (symmetric distribution).
  for (int32_t r = 1; r < kBIV100WorldSize; ++r) {
    EXPECT_EQ(est.per_rank_bytes[0], est.per_rank_bytes[r]);
  }
}

// ---------------------------------------------------------------------------
// TC-03  ILU topology-aware mapping (flat PIX)
// ---------------------------------------------------------------------------
TEST(LayerwiseSplitKV, TC03_IluTopologyMapping) {
  const int64_t num_full_attn = 16;
  std::vector<int64_t> per_layer_heads(num_full_attn, kQwen35KVHeads);

  auto layout = compute_ilu_layerwise_layout(
      num_full_attn, per_layer_heads, kBIV100WorldSize,
      IluTopoKind::kFlatPIX);

  ASSERT_EQ(layout.num_layers(), num_full_attn);

  // All layers must have at least one assigned rank.
  for (int64_t lid = 0; lid < num_full_attn; ++lid) {
    EXPECT_FALSE(layout.layer_spec(lid).assigned_ranks.empty());
  }

  // Qwen3.5 kv_heads=4 == world_size=4: all layers on all ranks.
  for (int64_t lid = 0; lid < num_full_attn; ++lid) {
    for (int32_t r = 0; r < kBIV100WorldSize; ++r) {
      EXPECT_TRUE(layout.rank_owns_layer(r, lid));
    }
  }
}

// ---------------------------------------------------------------------------
// TC-03b  ILU topology with fewer heads than ranks (future model)
// ---------------------------------------------------------------------------
TEST(LayerwiseSplitKV, TC03b_FewerHeadsThanRanks) {
  const int64_t num_layers = 16;
  // Hypothetical model with 2 KV heads per layer (< world_size=4).
  std::vector<int64_t> per_layer_heads(num_layers, 2);

  auto layout = compute_ilu_layerwise_layout(
      num_layers, per_layer_heads, kBIV100WorldSize,
      IluTopoKind::kFlatPIX);

  ASSERT_EQ(layout.num_layers(), num_layers);

  // Each layer should have exactly 2 assigned ranks.
  for (int64_t lid = 0; lid < num_layers; ++lid) {
    EXPECT_EQ(static_cast<int64_t>(layout.layer_spec(lid).assigned_ranks.size()), 2);
  }

  // Round-robin should distribute evenly: 16 layers × 2 ranks = 32
  // assignments over 4 ranks → ~8 each.
  std::vector<int32_t> rank_count(kBIV100WorldSize, 0);
  for (int64_t lid = 0; lid < num_layers; ++lid) {
    for (auto r : layout.layer_spec(lid).assigned_ranks) {
      rank_count[r]++;
    }
  }
  int32_t min_c = *std::min_element(rank_count.begin(), rank_count.end());
  int32_t max_c = *std::max_element(rank_count.begin(), rank_count.end());
  EXPECT_LE(max_c - min_c, 1)
      << "Round-robin should balance assignments across flat PIX topology";
}

// ---------------------------------------------------------------------------
// TC-04  Distributed engine layout propagation
// ---------------------------------------------------------------------------
TEST(LayerwiseSplitKV, TC04_EnginePropagation) {
  FLAGS_enable_layerwise_split = true;

  std::vector<int64_t> heads(16, kQwen35KVHeads);

  auto layout = maybe_compute_layerwise_layout(
      16, heads, kBIV100WorldSize);
  ASSERT_TRUE(layout.has_value());
  EXPECT_EQ(layout->num_layers(), 16);

  for (int32_t r = 0; r < kBIV100WorldSize; ++r) {
    EXPECT_GT(layout->layers_on_rank(r), 0);
  }

  FLAGS_enable_layerwise_split = false;
}

// ---------------------------------------------------------------------------
// TC-05  Worker layer_cache_owned computation (Qwen3.5 layer types)
// ---------------------------------------------------------------------------
TEST(LayerwiseSplitKV, TC05_WorkerLayerCacheOwned) {
  const int64_t num_layers = 32;
  auto layer_types = make_qwen35_layer_types(num_layers);
  // 16 full_attention + 16 linear_attention

  // rank=0, split_size=2: full-attn layers at even layer_ids (0,2,4,...30)
  // layer_id % 2 == 0 for all of them → all owned by rank 0.
  // Use rank=1 to see the split: layer_id%2==1 needed, but full layers are
  // at even ids → none owned → 16 linear (always owned) + 0 full = 16 owned.
  auto owned_r0 = worker_compute_layer_cache_owned(
      layer_types, /*layerwise_split_size=*/2, /*rank=*/0, num_layers);
  auto owned_r1 = worker_compute_layer_cache_owned(
      layer_types, /*layerwise_split_size=*/2, /*rank=*/1, num_layers);

  ASSERT_EQ(static_cast<int64_t>(owned_r0.size()), num_layers);
  ASSERT_EQ(static_cast<int64_t>(owned_r1.size()), num_layers);

  // Linear-attention layers always owned on both ranks.
  for (int64_t i = 1; i < num_layers; i += 2) {
    EXPECT_TRUE(owned_r0[static_cast<size_t>(i)]);
    EXPECT_TRUE(owned_r1[static_cast<size_t>(i)]);
  }

  // Full-attention layers: rank 0 owns all (even layer_ids % 2 == 0),
  // rank 1 owns none.
  int64_t r0_full = 0, r1_full = 0;
  for (int64_t i = 0; i < num_layers; i += 2) {
    if (owned_r0[static_cast<size_t>(i)]) ++r0_full;
    if (owned_r1[static_cast<size_t>(i)]) ++r1_full;
  }
  EXPECT_EQ(r0_full, 16);  // all full-attn layers owned
  EXPECT_EQ(r1_full, 0);   // none owned
}

// ---------------------------------------------------------------------------
// TC-06  Fallback to uniform when disabled
// ---------------------------------------------------------------------------
TEST(LayerwiseSplitKV, TC06_FallbackUniform) {
  FLAGS_enable_layerwise_split = false;

  std::vector<int64_t> heads(16, kQwen35KVHeads);
  auto layout = maybe_compute_layerwise_layout(
      16, heads, kBIV100WorldSize);
  EXPECT_FALSE(layout.has_value());
}

// ---------------------------------------------------------------------------
// TC-06b  Worker with split_size=1 (all owned)
// ---------------------------------------------------------------------------
TEST(LayerwiseSplitKV, TC06b_WorkerNoSplit) {
  const int64_t num_layers = 32;
  auto layer_types = make_qwen35_layer_types(num_layers);

  auto owned = worker_compute_layer_cache_owned(
      layer_types, /*layerwise_split_size=*/1, /*rank=*/0, num_layers);

  ASSERT_EQ(static_cast<int64_t>(owned.size()), num_layers);

  // All layers should be owned when split_size=1.
  for (int64_t i = 0; i < num_layers; ++i) {
    EXPECT_TRUE(owned[static_cast<size_t>(i)]);
  }
}

// ---------------------------------------------------------------------------
// TC-07  Upstream-compatible LayerwiseSplitLayout API
// ---------------------------------------------------------------------------
TEST(LayerwiseSplitKV, TC07_UpstreamCompatibleLayout) {
  const LayerwiseSplitLayout layout(/*enabled=*/true,
                                     /*group_size=*/2,
                                     /*local_rank=*/0);

  // Round-robin: layer 0 → rank 0 (owns), layer 1 → rank 1 (not owned)
  EXPECT_TRUE(layout.owns(0));
  EXPECT_FALSE(layout.owns(1));
  EXPECT_TRUE(layout.owns(2));
  EXPECT_FALSE(layout.owns(3));

  // Disabled layout → everything owned.
  const LayerwiseSplitLayout disabled(/*enabled=*/false,
                                       /*group_size=*/2,
                                       /*local_rank=*/0);
  EXPECT_TRUE(disabled.owns(0));
  EXPECT_TRUE(disabled.owns(1));
}

// ---------------------------------------------------------------------------
// TC-08  build_layer_cache_owned with mixed layer types
// ---------------------------------------------------------------------------
TEST(LayerwiseSplitKV, TC08_BuildLayerCacheOwned) {
  std::vector<std::string> types = {
    "full_attention", "linear_attention", "full_attention", "linear_attention",
    "full_attention", "linear_attention", "full_attention", "linear_attention",
  };
  const LayerwiseSplitLayout layout(/*enabled=*/true,
                                     /*group_size=*/2,
                                     /*local_rank=*/0);

  ModelArgs args;
  args.n_layers(8);
  args.layer_types(types);
  auto owned = build_layer_cache_owned(args, layout, 8);
  ASSERT_EQ(owned.size(), 8u);

  // Linear-attention layers (indices 1,3,5,7) → always true.
  EXPECT_TRUE(owned[1]);
  EXPECT_TRUE(owned[3]);
  EXPECT_TRUE(owned[5]);
  EXPECT_TRUE(owned[7]);

  // Full-attention layers (indices 0,2,4,6):
  // Upstream uses absolute layer_id % group_size, NOT full_attn_idx.
  //   layer 0: 0%2==0 → owned (local_rank=0)
  //   layer 2: 2%2==0 → owned
  //   layer 4: 4%2==0 → owned
  //   layer 6: 6%2==0 → owned
  // All even layer_ids are owned by rank 0! This is correct upstream behavior.
  EXPECT_TRUE(owned[0]);
  EXPECT_TRUE(owned[2]);
  EXPECT_TRUE(owned[4]);
  EXPECT_TRUE(owned[6]);
}

// ---------------------------------------------------------------------------
// TC-09  Model type validation
// ---------------------------------------------------------------------------
TEST(LayerwiseSplitKV, TC09_ModelTypeValidation) {
  EXPECT_TRUE(is_layerwise_split_supported_model("qwen3_5"));
  EXPECT_TRUE(is_layerwise_split_supported_model("qwen3_5_moe_text"));
  EXPECT_TRUE(is_layerwise_split_supported_model("deepseek_v32"));
  EXPECT_TRUE(is_layerwise_split_supported_model("glm_moe_dsa"));
  EXPECT_FALSE(is_layerwise_split_supported_model("llama"));
  EXPECT_FALSE(is_layerwise_split_supported_model(""));
}

// ---------------------------------------------------------------------------
// TC-10  Block count estimation with layerwise split
// ---------------------------------------------------------------------------
TEST(LayerwiseSplitKV, TC10_BlockCountEstimation) {
  // Qwen3.5 per-layer KV cache per block (TP=4, 1 kv head, head_dim=256):
  //   key:   1 * 16 * 16 * 8 * 2 = 4096 bytes
  //   value: 1 * 256 * 16 * 2    = 8192 bytes
  //   total per layer per block:   ~12288 bytes (but varies by layout)
  //
  // Simplified: use 16384 bytes per layer per block (K+V).
  const int64_t per_layer_block_bytes = 16384;
  const int64_t scratch_block_bytes = 16384;
  const int64_t available_bytes = int64_t{30} * 1024 * 1024 * 1024;  // ~30 GiB

  auto blocks_split1 = estimate_layerwise_split_block_count(
      /*layerwise_split_size=*/1, /*num_full_attn_layers=*/16,
      per_layer_block_bytes, scratch_block_bytes, available_bytes);

  auto blocks_split2 = estimate_layerwise_split_block_count(
      /*layerwise_split_size=*/2, /*num_full_attn_layers=*/16,
      per_layer_block_bytes, scratch_block_bytes, available_bytes);

  // With split_size=2, each rank owns ~8 layers instead of 16,
  // so it should be able to afford more blocks.
  EXPECT_GT(blocks_split2, blocks_split1);
  EXPECT_GT(blocks_split1, 0);
}

}  // namespace
}  // namespace xllm
