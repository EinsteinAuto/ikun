/* Migrated from xllm/core/framework/config/parallel_config.cpp
   Stripped json — uses gflags directly. */

#include "framework/config/parallel_config.h"

#include <gflags/gflags.h>

DEFINE_int32(dp_size, 1, "Data parallel size.");
DEFINE_int32(ep_size, 1, "Expert parallel size for MoE model.");
DEFINE_int32(cp_size, 1, "Context parallel size.");
DEFINE_int32(
    layerwise_split_size,
    1,
    "Layer-owner KV cache group size inside each attention TP group. "
    "1 disables layerwise split; values > 1 shard persistent KV by layer owner "
    "and enable layerwise-split communication. The value must divide attention "
    "TP size.");
DEFINE_int32(kv_split_size, 1, "KV-cache split width.");
DEFINE_int64(tp_size, 1, "Tensor parallelism size.");
DEFINE_string(communication_backend, "nccl", "Communication backend.");
DEFINE_bool(enable_multi_stream_parallel, false,
            "Enable multi-stream parallel.");

namespace xllm {

void ParallelConfig::from_flags() {
  dp_size(FLAGS_dp_size);
  ep_size(FLAGS_ep_size);
  cp_size(FLAGS_cp_size);
  layerwise_split_size(FLAGS_layerwise_split_size);
  kv_split_size(FLAGS_kv_split_size);
  tp_size(FLAGS_tp_size);
  communication_backend(FLAGS_communication_backend);
  enable_multi_stream_parallel(FLAGS_enable_multi_stream_parallel);
}

ParallelConfig& ParallelConfig::get_instance() {
  static ParallelConfig config;
  return config;
}

}  // namespace xllm
