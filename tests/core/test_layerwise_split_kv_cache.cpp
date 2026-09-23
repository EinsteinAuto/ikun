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
// Tests are split into two sections:
//   1. Generic tests — always compiled, test upstream-portable code paths.
//   2. ILU-specific tests — only compiled with -DUSE_ILU, test ILU device
//      mapping, engine propagation, and memory estimation on BI-V100.
//
// Verified hardware (ILU tests):
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
#include "framework/kv_cache/kv_cache_estimation_layerwise.h"
#include "framework/kv_cache/layerwise_split_layout.h"
#include "framework/model/model_args.h"
#include "framework/kv_cache/kv_cache_estimation.h"
#include "framework/kv_cache/ilu_layerwise_layout.h"
#include "runtime/worker_layerwise_init.h"

// ILU-specific headers (guarded by USE_ILU in the headers themselves).
#if defined(USE_ILU)
#include "distributed_runtime/layerwise_split_engine_ext.h"
#include "distributed_runtime/layerwise_split_master.h"
#include "framework/parallel_state/mapping_ilu.h"
#endif

namespace xllm {
namespace {

// Qwen3.5 model constants — from ilu_hw_constants.h (always available).
constexpr int32_t kWorldSize4 = 4;
constexpr int64_t kQwen35KVHeads = ilu_hw::kQwen35NumKVHeads;   // 4
constexpr int64_t kQwen35HeadDim = ilu_hw::kQwen35HeadDim;      // 256

// Helper: build a realistic Qwen3.5 layer_types vector.
// The actual pattern alternates full_attention and linear_attention.
std::vector<std::string> make_qwen35_layer_types(int64_t num_layers) {
  std::vector<std::string> types;
  types.reserve(static_cast<size_t>(num_layers));
  for (int64_t i = 0; i < num_layers; ++i) {
    types.push_back(i % 2 == 0 ? "full_attention" : "linear_attention");
  }
  return types;
}

// Helper: build all-full-attention layer types.
std::vector<std::string> make_all_full_attention(int64_t num_layers) {
  return std::vector<std::string>(static_cast<size_t>(num_layers),
                                  "full_attention");
}

// ==========================================================================
//  SECTION 1: Generic tests — always compiled (no USE_ILU dependency)
// ==========================================================================

// ---------------------------------------------------------------------------
// TC-05  Worker layer_cache_owned computation (Qwen3.5 layer types)
// ---------------------------------------------------------------------------
TEST(LayerwiseSplitKV, TC05_WorkerLayerCacheOwned) {
  const int64_t num_layers = 32;
  auto layer_types = make_qwen35_layer_types(num_layers);

  auto owned_r0 = worker_compute_layer_cache_owned(
      layer_types, /*layerwise_split_size=*/2, /*rank=*/0, num_layers);
  auto owned_r1 = worker_compute_layer_cache_owned(
      layer_types, /*layerwise_split_size=*/2, /*rank=*/1, num_layers);

  ASSERT_EQ(static_cast<int64_t>(owned_r0.size()), num_layers);
  ASSERT_EQ(static_cast<int64_t>(owned_r1.size()), num_layers);

  // Linear-attention layers have no KV cache, so they are NOT "owned"
  // in the cache-splitting sense — they always return false.
  // (Previously they returned true due to a negation bug.)
  for (int64_t i = 1; i < num_layers; i += 2) {
    EXPECT_FALSE(owned_r0[static_cast<size_t>(i)]);
    EXPECT_FALSE(owned_r1[static_cast<size_t>(i)]);
  }

  // Full-attention layers: rank 0 owns all (even layer_ids % 2 == 0),
  // rank 1 owns none.
  int64_t r0_full = 0, r1_full = 0;
  for (int64_t i = 0; i < num_layers; i += 2) {
    if (owned_r0[static_cast<size_t>(i)]) ++r0_full;
    if (owned_r1[static_cast<size_t>(i)]) ++r1_full;
  }
  EXPECT_EQ(r0_full, 16);
  EXPECT_EQ(r1_full, 0);
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
  EXPECT_TRUE(layout.owns(0));
  EXPECT_FALSE(layout.owns(1));
  EXPECT_TRUE(layout.owns(2));
  EXPECT_FALSE(layout.owns(3));

  const LayerwiseSplitLayout disabled(/*enabled=*/false,
                                       /*group_size=*/2,
                                       /*local_rank=*/0);
  EXPECT_TRUE(disabled.owns(0));
  EXPECT_TRUE(disabled.owns(1));
}

// ---------------------------------------------------------------------------
// TC-07b  LayerwiseSplitLayout with group_size=4 (4-way split)
// ---------------------------------------------------------------------------
TEST(LayerwiseSplitKV, TC07b_FourWaySplit) {
  for (int32_t rank = 0; rank < 4; ++rank) {
    const LayerwiseSplitLayout layout(true, /*group_size=*/4, rank);
    EXPECT_EQ(layout.group_size(), 4);
    EXPECT_EQ(layout.local_rank(), rank);
    // Each rank owns layer_id where layer_id % 4 == rank.
    for (int64_t lid = 0; lid < 32; ++lid) {
      EXPECT_EQ(layout.owns(lid), (lid % 4 == rank))
          << "rank=" << rank << " lid=" << lid;
    }
  }
}

// ---------------------------------------------------------------------------
// TC-07c  LayerwiseSplitLayout owner_rank correctness
// ---------------------------------------------------------------------------
TEST(LayerwiseSplitKV, TC07c_OwnerRank) {
  const LayerwiseSplitLayout layout(true, /*group_size=*/3, /*local_rank=*/1);
  EXPECT_EQ(layout.owner_rank(0), 0);
  EXPECT_EQ(layout.owner_rank(1), 1);
  EXPECT_EQ(layout.owner_rank(2), 2);
  EXPECT_EQ(layout.owner_rank(3), 0);
  EXPECT_EQ(layout.owner_rank(4), 1);
  EXPECT_EQ(layout.owner_rank(5), 2);
  // Rank 1 owns layers 1, 4, 7, 10, ...
  EXPECT_TRUE(layout.owns(1));
  EXPECT_TRUE(layout.owns(4));
  EXPECT_FALSE(layout.owns(0));
  EXPECT_FALSE(layout.owns(2));
}

// ---------------------------------------------------------------------------
// TC-08  build_layer_cache_owned with mixed layer types
// ---------------------------------------------------------------------------
TEST(LayerwiseSplitKV, TC08_BuildLayerCacheOwned) {
  std::vector<std::string> types = {
    "full_attention", "linear_attention", "full_attention", "linear_attention",
    "full_attention", "linear_attention", "full_attention", "linear_attention",
  };
  const LayerwiseSplitLayout layout(true, /*group_size=*/2, /*local_rank=*/0);

  ModelArgs args;
  args.n_layers(8);
  args.layer_types(types);
  auto owned = build_layer_cache_owned(args, layout, 8);
  ASSERT_EQ(owned.size(), 8u);

  // Linear-attention layers (indices 1,3,5,7) → false (no KV cache to split).
  EXPECT_FALSE(owned[1]);
  EXPECT_FALSE(owned[3]);
  EXPECT_FALSE(owned[5]);
  EXPECT_FALSE(owned[7]);

  // Full-attention layers (indices 0,2,4,6): all even → owned by rank 0.
  EXPECT_TRUE(owned[0]);
  EXPECT_TRUE(owned[2]);
  EXPECT_TRUE(owned[4]);
  EXPECT_TRUE(owned[6]);
}

// ---------------------------------------------------------------------------
// TC-08b  build_layer_cache_owned with rank 1 (odd layers owned)
// ---------------------------------------------------------------------------
TEST(LayerwiseSplitKV, TC08b_BuildLayerCacheOwnedRank1) {
  std::vector<std::string> types = {
    "full_attention", "linear_attention", "full_attention", "linear_attention",
    "full_attention", "linear_attention", "full_attention", "linear_attention",
  };
  const LayerwiseSplitLayout layout(true, /*group_size=*/2, /*local_rank=*/1);

  ModelArgs args;
  args.n_layers(8);
  args.layer_types(types);
  auto owned = build_layer_cache_owned(args, layout, 8);
  ASSERT_EQ(owned.size(), 8u);

  // Linear-attention layers always owned.
  EXPECT_TRUE(owned[1]);
  EXPECT_TRUE(owned[3]);

  // Full-attention layers: lid 0 → 0%2=0 (rank 0), lid 2 → 2%2=0 (rank 0)
  // → rank 1 owns NONE of the full-attention layers (all at even indices).
  EXPECT_FALSE(owned[0]);
  EXPECT_FALSE(owned[2]);
  EXPECT_FALSE(owned[4]);
  EXPECT_FALSE(owned[6]);
}

// ---------------------------------------------------------------------------
// TC-08c  build_layer_cache_owned with all full-attention layers
// ---------------------------------------------------------------------------
TEST(LayerwiseSplitKV, TC08c_AllFullAttention) {
  auto types = make_all_full_attention(8);
  const LayerwiseSplitLayout layout(true, /*group_size=*/2, /*local_rank=*/0);

  ModelArgs args;
  args.n_layers(8);
  args.layer_types(types);
  auto owned = build_layer_cache_owned(args, layout, 8);
  ASSERT_EQ(owned.size(), 8u);

  // Rank 0 owns even-index layers only.
  int64_t count = 0;
  for (size_t i = 0; i < owned.size(); ++i) {
    if (owned[i]) ++count;
    EXPECT_EQ(owned[i], (i % 2 == 0)) << "layer " << i;
  }
  EXPECT_EQ(count, 4);
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
  EXPECT_FALSE(is_layerwise_split_supported_model("qwen3_5_prefix"));
  EXPECT_FALSE(is_layerwise_split_supported_model("Qwen3_5"));  // case
}

// ---------------------------------------------------------------------------
// TC-09b  is_full_attention_layer with explicit layer_types
// ---------------------------------------------------------------------------
TEST(LayerwiseSplitKV, TC09b_IsFullAttentionLayer) {
  ModelArgs args;
  args.n_layers(6);
  args.layer_types({"full_attention", "linear_attention", "attention",
                     "full_attention", "linear_attention", "full_attention"});

  EXPECT_TRUE(is_full_attention_layer(args, 0));   // full_attention
  EXPECT_FALSE(is_full_attention_layer(args, 1));  // linear_attention
  EXPECT_TRUE(is_full_attention_layer(args, 2));   // "attention" is full
  EXPECT_TRUE(is_full_attention_layer(args, 3));   // full_attention
  EXPECT_FALSE(is_full_attention_layer(args, 4));  // linear_attention
  EXPECT_TRUE(is_full_attention_layer(args, 5));   // full_attention
}

// ---------------------------------------------------------------------------
// TC-09c  is_full_attention_layer with full_attention_interval fallback
// ---------------------------------------------------------------------------
TEST(LayerwiseSplitKV, TC09c_FullAttentionInterval) {
  ModelArgs args;
  args.n_layers(8);
  args.full_attention_interval(4);
  // No layer_types set → use interval. full_attention when (lid+1)%4 == 0.
  // Layers 3, 7 are full-attention.

  EXPECT_FALSE(is_full_attention_layer(args, 0));
  EXPECT_FALSE(is_full_attention_layer(args, 1));
  EXPECT_FALSE(is_full_attention_layer(args, 2));
  EXPECT_TRUE(is_full_attention_layer(args, 3));   // (3+1)%4=0
  EXPECT_FALSE(is_full_attention_layer(args, 4));
  EXPECT_FALSE(is_full_attention_layer(args, 5));
  EXPECT_FALSE(is_full_attention_layer(args, 6));
  EXPECT_TRUE(is_full_attention_layer(args, 7));   // (7+1)%4=0
}

// ---------------------------------------------------------------------------
// TC-09d  has_linear_attention_layers detection
// ---------------------------------------------------------------------------
TEST(LayerwiseSplitKV, TC09d_HasLinearAttentionLayers) {
  {
    ModelArgs args;
    args.layer_types({"full_attention", "linear_attention"});
    EXPECT_TRUE(has_linear_attention_layers(args));
  }
  {
    ModelArgs args;
    args.layer_types({"full_attention", "full_attention"});
    EXPECT_FALSE(has_linear_attention_layers(args));
  }
  {
    ModelArgs args;
    args.full_attention_interval(4);
    EXPECT_TRUE(has_linear_attention_layers(args));
  }
  {
    ModelArgs args;
    args.full_attention_interval(1);
    EXPECT_FALSE(has_linear_attention_layers(args));
  }
}

// ---------------------------------------------------------------------------
// TC-09e  is_qwen3_5_target_model_type exhaustive
// ---------------------------------------------------------------------------
TEST(LayerwiseSplitKV, TC09e_Qwen35TargetModelType) {
  EXPECT_TRUE(is_qwen3_5_target_model_type("qwen3_5"));
  EXPECT_TRUE(is_qwen3_5_target_model_type("qwen3_5_moe"));
  EXPECT_TRUE(is_qwen3_5_target_model_type("qwen3_5_text"));
  EXPECT_TRUE(is_qwen3_5_target_model_type("qwen3_5_moe_text"));
  EXPECT_FALSE(is_qwen3_5_target_model_type("qwen3_5_"));
  EXPECT_FALSE(is_qwen3_5_target_model_type("qwen3_6"));
  EXPECT_FALSE(is_qwen3_5_target_model_type(""));
}

// ---------------------------------------------------------------------------
// TC-10  Block count estimation with layerwise split
// ---------------------------------------------------------------------------
TEST(LayerwiseSplitKV, TC10_BlockCountEstimation) {
  const int64_t per_layer_block_bytes = 16384;
  const int64_t scratch_block_bytes = 16384;
  const int64_t available_bytes = int64_t{30} * 1024 * 1024 * 1024;

  auto blocks_split1 = estimate_layerwise_split_block_count(
      /*layerwise_split_size=*/1, /*num_full_attn_layers=*/16,
      per_layer_block_bytes, scratch_block_bytes, available_bytes);

  auto blocks_split2 = estimate_layerwise_split_block_count(
      /*layerwise_split_size=*/2, /*num_full_attn_layers=*/16,
      per_layer_block_bytes, scratch_block_bytes, available_bytes);

  EXPECT_GT(blocks_split2, blocks_split1);
  EXPECT_GT(blocks_split1, 0);
}

// ---------------------------------------------------------------------------
// TC-10b  Block count estimation with split_size=4
// ---------------------------------------------------------------------------
TEST(LayerwiseSplitKV, TC10b_BlockCountSplit4) {
  const int64_t per_layer_block_bytes = 16384;
  const int64_t scratch_block_bytes = 16384;
  const int64_t available_bytes = int64_t{30} * 1024 * 1024 * 1024;

  auto blocks_split2 = estimate_layerwise_split_block_count(
      2, 16, per_layer_block_bytes, scratch_block_bytes, available_bytes);
  auto blocks_split4 = estimate_layerwise_split_block_count(
      4, 16, per_layer_block_bytes, scratch_block_bytes, available_bytes);

  // More splits → fewer owned layers per rank → more blocks.
  EXPECT_GT(blocks_split4, blocks_split2);
}

// ---------------------------------------------------------------------------
// TC-11  LayerwiseSplitLayout validation configuration
// ---------------------------------------------------------------------------
TEST(LayerwiseSplitKV, TC11_ValidateConfig) {
  // Valid configs.
  EXPECT_NO_FATAL_FAILURE(validate_layerwise_split_size_config(1));
  EXPECT_NO_FATAL_FAILURE(validate_layerwise_split_size_config(2));
  EXPECT_NO_FATAL_FAILURE(validate_layerwise_split_size_config(4));

  // Valid enablement.
  EXPECT_NO_FATAL_FAILURE(
      validate_layerwise_split_enablement(1, 4, "llama"));  // disabled
  EXPECT_NO_FATAL_FAILURE(
      validate_layerwise_split_enablement(2, 4, "qwen3_5"));
  EXPECT_NO_FATAL_FAILURE(
      validate_layerwise_split_enablement(2, 4, "deepseek_v32"));
}

// ---------------------------------------------------------------------------
// TC-12  Worker with all-full-attention and split_size=4
// ---------------------------------------------------------------------------
TEST(LayerwiseSplitKV, TC12_WorkerAllFullAttention) {
  auto types = make_all_full_attention(16);

  // split_size=4, rank=2: owns layers where lid%4==2 → 2,6,10,14 = 4 layers
  auto owned = worker_compute_layer_cache_owned(types, 4, 2, 16);
  ASSERT_EQ(static_cast<int64_t>(owned.size()), 16);

  int64_t count = 0;
  for (int64_t i = 0; i < 16; ++i) {
    bool expected = (i % 4 == 2);
    EXPECT_EQ(owned[static_cast<size_t>(i)], expected) << "layer " << i;
    if (owned[static_cast<size_t>(i)]) ++count;
  }
  EXPECT_EQ(count, 4);
}

// ---------------------------------------------------------------------------
// TC-13  Worker split symmetry: union of all ranks covers all layers
// ---------------------------------------------------------------------------
TEST(LayerwiseSplitKV, TC13_WorkerSplitSymmetry) {
  const int64_t num_layers = 32;
  auto types = make_qwen35_layer_types(num_layers);

  for (int32_t group_size : {2, 4}) {
    std::vector<int64_t> full_attn_coverage(num_layers, 0);

    for (int32_t rank = 0; rank < group_size; ++rank) {
      auto owned = worker_compute_layer_cache_owned(
          types, group_size, rank, num_layers);
      for (int64_t i = 0; i < num_layers; ++i) {
        if (owned[static_cast<size_t>(i)]) {
          ++full_attn_coverage[i];
        }
      }
    }

    // Every layer must be owned by at least one rank.
    for (int64_t i = 0; i < num_layers; ++i) {
      EXPECT_GE(full_attn_coverage[i], 1)
          << "layer " << i << " not owned by any rank (group_size="
          << group_size << ")";
    }

    // Linear-attention layers are owned by ALL ranks.
    for (int64_t i = 1; i < num_layers; i += 2) {
      EXPECT_EQ(full_attn_coverage[i], group_size)
          << "linear-attn layer " << i << " should be owned by all ranks";
    }
  }
}

// ---------------------------------------------------------------------------
// TC-14  KVCacheCapacity property accessors (PROPERTY macro)
// ---------------------------------------------------------------------------
TEST(LayerwiseSplitKV, TC14_KVCacheCapacityProperties) {
  KVCacheCapacity cap;
  cap.cache_size_in_bytes(int64_t{32} * 1024 * 1024 * 1024);
  cap.block_size(16);
  cap.slot_size(512);
  cap.linear_slot_size(1024);
  cap.n_layers(64);
  cap.num_full_attention_layers(32);
  cap.num_linear_attention_layers(32);

  EXPECT_EQ(cap.cache_size_in_bytes(), int64_t{32} * 1024 * 1024 * 1024);
  EXPECT_EQ(cap.block_size(), 16);
  EXPECT_EQ(cap.slot_size(), 512);
  EXPECT_EQ(cap.linear_slot_size(), 1024);
  EXPECT_EQ(cap.n_layers(), 64);
  EXPECT_EQ(cap.num_full_attention_layers(), 32);
  EXPECT_EQ(cap.num_linear_attention_layers(), 32);
}

// ---------------------------------------------------------------------------
// TC-15  ModelArgs property accessors
// ---------------------------------------------------------------------------
TEST(LayerwiseSplitKV, TC15_ModelArgsProperties) {
  ModelArgs args;
  args.model_type("qwen3_5");
  args.n_layers(64);
  args.head_dim(256);
  args.n_heads(16);
  args.n_kv_heads(4);
  args.linear_num_value_heads(4);
  args.linear_key_head_dim(128);
  args.linear_value_head_dim(128);
  args.linear_conv_kernel_dim(4);

  EXPECT_EQ(args.model_type(), "qwen3_5");
  EXPECT_EQ(args.n_layers(), 64);
  EXPECT_EQ(args.head_dim(), 256);
  EXPECT_EQ(args.n_kv_heads().value(), 4);
  EXPECT_EQ(args.linear_num_value_heads(), 4);
  EXPECT_EQ(args.linear_conv_kernel_dim(), 4);
}

// ==========================================================================
//  SECTION 2: ILU-specific tests — only compiled with -DUSE_ILU
// ==========================================================================
#if defined(USE_ILU)

// ---------------------------------------------------------------------------
// TC-01  Layerwise KV allocation correctness (Qwen3.5 parameters)
// ---------------------------------------------------------------------------
TEST(LayerwiseSplitKV_ILU, TC01_AllocationCorrectness) {
  const int64_t num_full_attn = 16;
  std::vector<int64_t> per_layer_heads(num_full_attn, kQwen35KVHeads);

  auto layout = compute_ilu_layerwise_layout(
      num_full_attn, per_layer_heads, kWorldSize4);

  ASSERT_EQ(layout.num_layers(), num_full_attn);

  for (int64_t lid = 0; lid < num_full_attn; ++lid) {
    const auto& spec = layout.layer_spec(lid);
    EXPECT_EQ(static_cast<int32_t>(spec.assigned_ranks.size()), kWorldSize4);
    for (int32_t r = 0; r < kWorldSize4; ++r) {
      EXPECT_EQ(layout.heads_for_rank(r, lid), 1)
          << "Each rank should have exactly 1 KV head for Qwen3.5 TP=4";
    }
  }

  int64_t total = 0;
  for (int64_t lid = 0; lid < num_full_attn; ++lid)
    total += layout.layer_spec(lid).total_heads();
  EXPECT_EQ(total, num_full_attn * kQwen35KVHeads);
}

// ---------------------------------------------------------------------------
// TC-02  Memory estimation accuracy
// ---------------------------------------------------------------------------
TEST(LayerwiseSplitKV_ILU, TC02_MemoryEstimation) {
  const int64_t num_full_attn = 16;
  const int64_t n_blocks   = 68837;
  const int64_t block_size = 16;
  const int64_t head_dim   = kQwen35HeadDim;
  const int64_t max_tokens = 262144;
  const int dtype_enum     = 5;  // float16

  std::vector<int64_t> per_layer_heads(num_full_attn, kQwen35KVHeads);

  auto layout = compute_ilu_layerwise_layout(
      num_full_attn, per_layer_heads, kWorldSize4);

  auto est = estimate_layerwise_kv_memory(
      layout, n_blocks, block_size, head_dim, max_tokens, dtype_enum,
      kWorldSize4);

  EXPECT_EQ(est.peak_per_rank_bytes, est.uniform_per_rank_bytes);
  EXPECT_GT(est.peak_per_rank_bytes, 0);
  EXPECT_GT(est.average_per_rank_bytes, 0);
  EXPECT_EQ(static_cast<int32_t>(est.per_rank_bytes.size()), kWorldSize4);

  for (int32_t r = 1; r < kWorldSize4; ++r) {
    EXPECT_EQ(est.per_rank_bytes[0], est.per_rank_bytes[r]);
  }
}

// ---------------------------------------------------------------------------
// TC-03  ILU topology-aware mapping (flat PIX)
// ---------------------------------------------------------------------------
TEST(LayerwiseSplitKV_ILU, TC03_IluTopologyMapping) {
  const int64_t num_full_attn = 16;
  std::vector<int64_t> per_layer_heads(num_full_attn, kQwen35KVHeads);

  auto layout = compute_ilu_layerwise_layout(
      num_full_attn, per_layer_heads, kWorldSize4, IluTopoKind::kFlatPIX);

  ASSERT_EQ(layout.num_layers(), num_full_attn);

  for (int64_t lid = 0; lid < num_full_attn; ++lid) {
    EXPECT_FALSE(layout.layer_spec(lid).assigned_ranks.empty());
    for (int32_t r = 0; r < kWorldSize4; ++r) {
      EXPECT_TRUE(layout.rank_owns_layer(r, lid));
    }
  }
}

// ---------------------------------------------------------------------------
// TC-03b  ILU topology with fewer heads than ranks (future model)
// ---------------------------------------------------------------------------
TEST(LayerwiseSplitKV_ILU, TC03b_FewerHeadsThanRanks) {
  const int64_t num_layers = 16;
  std::vector<int64_t> per_layer_heads(num_layers, 2);

  auto layout = compute_ilu_layerwise_layout(
      num_layers, per_layer_heads, kWorldSize4, IluTopoKind::kFlatPIX);

  ASSERT_EQ(layout.num_layers(), num_layers);

  for (int64_t lid = 0; lid < num_layers; ++lid) {
    EXPECT_EQ(static_cast<int64_t>(
        layout.layer_spec(lid).assigned_ranks.size()), 2);
  }

  std::vector<int32_t> rank_count(kWorldSize4, 0);
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
TEST(LayerwiseSplitKV_ILU, TC04_EnginePropagation) {
  FLAGS_enable_layerwise_split = true;

  std::vector<int64_t> heads(16, kQwen35KVHeads);
  auto layout = maybe_compute_layerwise_layout(16, heads, kWorldSize4);
  ASSERT_TRUE(layout.has_value());
  EXPECT_EQ(layout->num_layers(), 16);

  for (int32_t r = 0; r < kWorldSize4; ++r) {
    EXPECT_GT(layout->layers_on_rank(r), 0);
  }

  FLAGS_enable_layerwise_split = false;
}

// ---------------------------------------------------------------------------
// TC-06  Fallback to uniform when disabled (ILU path)
// ---------------------------------------------------------------------------
TEST(LayerwiseSplitKV_ILU, TC06_FallbackUniform) {
  FLAGS_enable_layerwise_split = false;

  std::vector<int64_t> heads(16, kQwen35KVHeads);
  auto layout = maybe_compute_layerwise_layout(16, heads, kWorldSize4);
  EXPECT_FALSE(layout.has_value());
}

// ---------------------------------------------------------------------------
// TC-16  ILU memory estimation with asymmetric heads
// ---------------------------------------------------------------------------
TEST(LayerwiseSplitKV_ILU, TC16_AsymmetricHeads) {
  // Model with varying head counts per layer: 8, 4, 2, 1, 8, 4, 2, 1
  std::vector<int64_t> per_layer_heads = {8, 4, 2, 1, 8, 4, 2, 1};
  const int64_t num_layers = static_cast<int64_t>(per_layer_heads.size());

  auto layout = compute_ilu_layerwise_layout(
      num_layers, per_layer_heads, kWorldSize4, IluTopoKind::kFlatPIX);

  ASSERT_EQ(layout.num_layers(), num_layers);

  // Layers with 8 heads → 2 per rank. Layers with 1 head → 1 rank.
  EXPECT_EQ(layout.heads_for_rank(0, 0), 2);  // 8 heads / 4 ranks
  // Layer 3 has 1 head → only 1 rank assigned.
  EXPECT_EQ(static_cast<int64_t>(
      layout.layer_spec(3).assigned_ranks.size()), 1);

  auto est = estimate_layerwise_kv_memory(
      layout, /*n_blocks=*/1000, /*block_size=*/16, /*head_dim=*/256,
      /*max_tokens=*/65536, /*dtype_enum=*/5, kWorldSize4);

  // Asymmetric → peak > average.
  EXPECT_GE(est.peak_per_rank_bytes, est.average_per_rank_bytes);
  EXPECT_GT(est.peak_per_rank_bytes, 0);
}

#endif  // defined(USE_ILU)

}  // namespace
}  // namespace xllm
