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
#include <optional>
#include <string>
#include <vector>

#include "framework/kv_cache/ilu_layerwise_layout.h"

namespace xllm {

/// Master-side entry point: compute and log the layerwise layout.
///
/// For Qwen3.5 (Qwen3.6-35B-A3B):
///   |layer_types|: "full_attention" or "linear_attention" per layer.
///   |kv_heads_per_full_attn_layer|: KV heads for each full-attention layer.
///   Linear-attention layers are skipped (they use GDN conv+temporal state).
std::optional<IluLayerwiseLayout> master_compute_layerwise_layout(
    const std::vector<std::string>& layer_types,
    int64_t kv_heads_per_full_attn_layer,
    int32_t world_size,
    int64_t n_blocks,
    int64_t block_size,
    int64_t head_dim,
    int64_t max_tokens,
    int dtype_enum);

}  // namespace xllm
