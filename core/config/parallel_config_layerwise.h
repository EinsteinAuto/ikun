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

#include <gflags/gflags.h>

// Bool flag for enabling layerwise split (used by engine_ext / master).
DECLARE_bool(enable_layerwise_split);

// Int32 flag matching upstream xLLM's parallel_config.
// 1 = disabled (default), >1 = group size for layerwise split.
DECLARE_int32(layerwise_split_size);
