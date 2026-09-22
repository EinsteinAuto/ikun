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

#include "framework/kv_cache/layerwise_split_layout.h"

namespace xllm {

/// Allocate KV caches with per-layer ownership determined by |layout|.
/// Layers not owned by |current_rank| receive an empty (placeholder) KVCache.
///
/// |layer_cache_owned|: precomputed ownership mask from build_layer_cache_owned.
///   - true  → this rank allocates real KV cache tensors for this layer.
///   - false → this rank uses a shared scratch (zero-sized) placeholder.
///
/// For Qwen3.5: linear_attention layers always have owned=true but their
/// cache slot is unused (GDN uses conv+temporal state, not KV cache).
/// Only full_attention layers participate in layerwise split.
///
/// BI-V100 KV cache tensor layout (from verified runtime logs):
///   key_cache:   (n_blocks, n_kv_heads_local, block_size, head_dim/x, x)
///   value_cache: (n_blocks, n_kv_heads_local, head_dim, block_size)
///   where x = 128 / sizeof_dtype_bits (=8 for fp16/bf16)
///
/// This is the "block-major" transposed layout used by the vendor vLLM
/// on Iluvatar BI-V100.  head dimension sits at axis 1.
void allocate_kv_caches_layerwise(
    const std::vector<bool>& layer_cache_owned,
    int64_t num_layers,
    int32_t current_rank);

}  // namespace xllm
