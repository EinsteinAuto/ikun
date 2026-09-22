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
// gflag definitions for layerwise split KV cache.
//
// Usage:
//   --enable_layerwise_split=true --layerwise_split_size=2
//
// For Qwen3.5 (Qwen3.6-35B-A3B) on 4× BI-V100 with TP=4:
//   layerwise_split_size=2 means each rank owns ~half the full-attention
//   layers' KV cache, reducing peak per-rank KV memory.

#include <gflags/gflags.h>

DEFINE_bool(enable_layerwise_split, false,
            "Enable layerwise-split KV cache sharding.  When true, each "
            "full-attention layer's KV cache ownership is assigned to a "
            "subset of TP ranks via round-robin.  Linear-attention layers "
            "(GDN/DeltaNet) are unaffected — they use conv+temporal state, "
            "not KV cache.  Default: false (uniform sharding).");

DEFINE_int32(layerwise_split_size, 1,
             "Layer-owner KV cache group size inside each attention TP group. "
             "1 disables layerwise split; values > 1 shard persistent KV by "
             "layer owner and enable layerwise-split communication. The value "
             "must divide attention TP size.  Matches upstream xLLM "
             "parallel_config.layerwise_split_size.");
