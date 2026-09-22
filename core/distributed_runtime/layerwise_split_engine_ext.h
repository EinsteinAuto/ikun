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
#include <vector>

#include "framework/kv_cache/ilu_layerwise_layout.h"

namespace xllm {

/// Called by llm_engine / speculative_engine at startup.
/// Returns a IluLayerwiseLayout if the feature is enabled, otherwise
/// std::nullopt (fallback to uniform allocation).
///
/// |per_layer_kv_heads|: KV head count for each full-attention layer.
///   Linear-attention layers are excluded.
///   For Qwen3.5 TP=4: all entries are 4 (each full-attn layer has 4 KV heads).
std::optional<IluLayerwiseLayout> maybe_compute_layerwise_layout(
    int64_t num_layers,
    const std::vector<int64_t>& per_layer_kv_heads,
    int32_t world_size);

}  // namespace xllm
