/* Copyright 2026 The xLLM Authors. All Rights Reserved.
   Migrated from xllm/core/framework/kv_cache/kv_cache_capacity.h */

#pragma once

#include <cstdint>

#include "common/macros.h"

namespace xllm {

class KVCacheCapacity final {
 public:
  PROPERTY(int64_t, n_blocks) = 0;
  PROPERTY(int64_t, cache_size_in_bytes) = 0;
  PROPERTY(int64_t, block_size) = 0;
  PROPERTY(int64_t, slot_size) = 0;

  // for index cache
  PROPERTY(int64_t, index_slot_size) = 0;
  PROPERTY(int64_t, num_indexer_layers) = 0;
  PROPERTY(bool, enable_indexer_cache_quant) = false;

  // for kv cache quantization scale cache
  PROPERTY(int64_t, scale_slot_size) = 0;

  // for linear attention
  PROPERTY(int64_t, linear_slot_size) = 0;
  PROPERTY(int64_t, linear_cache_size_in_bytes) = 0;
  PROPERTY(int64_t, linear_conv_state_len) = 0;
  PROPERTY(int64_t, linear_ssm_checkpoint_stride) = 1;
  PROPERTY(int64_t, n_layers) = 0;
  PROPERTY(int64_t, num_linear_state_blocks) = 0;
  PROPERTY(int64_t, num_full_attention_layers) = 0;
  PROPERTY(int64_t, num_linear_attention_layers) = 0;

  // DeepSeek V4 uses separate block pools for sliding-window and compressed
  // caches. These fields are only meaningful for deepseek_v4.
  PROPERTY(int64_t, swa_count) = 0;
  PROPERTY(int64_t, c4_count) = 0;
  PROPERTY(int64_t, c128_count) = 0;
};

}  // namespace xllm
