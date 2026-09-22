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
#include <string>
#include <vector>

#include "framework/kv_cache/layerwise_split_layout.h"

namespace xllm {

/// Worker-side entry: compute the layer_cache_owned mask and validate.
///
/// For Qwen3.5: the worker receives layerwise_split_size from the master
/// and builds its ownership mask.  Only full_attention layers participate
/// in the split; linear_attention layers are always "owned" (no KV cache).
///
/// Returns the layer_cache_owned mask, which the vendor vLLM cache engine
/// uses to decide which layers get real KV allocations vs scratch.
std::vector<bool> worker_compute_layer_cache_owned(
    const std::vector<std::string>& layer_types,
    int32_t layerwise_split_size,
    int32_t rank,
    int64_t num_layers);

}  // namespace xllm
