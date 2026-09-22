/* Worker-side helper: compute layer_cache_owned mask.
   Migrated pattern from upstream worker_impl.cpp lines 490-498. */

#include "runtime/worker_layerwise_init.h"

#include <glog/logging.h>
#include <string>
#include <vector>

#include "framework/kv_cache/kv_cache_estimation.h"
#include "framework/kv_cache/layerwise_split_layout.h"
#include "framework/model/model_args.h"

namespace xllm {

std::vector<bool> worker_compute_layer_cache_owned(
    const std::vector<std::string>& layer_types,
    int32_t layerwise_split_size,
    int32_t rank,
    int64_t num_layers) {
  CHECK_GE(layerwise_split_size, 1);
  CHECK_GE(rank, 0);
  CHECK_GT(num_layers, 0);

  if (layerwise_split_size <= 1) {
    LOG(INFO) << "[Worker " << rank << "] No layerwise split, all "
              << num_layers << " layers owned.";
    return std::vector<bool>(static_cast<size_t>(num_layers), true);
  }

  // Build a ModelArgs with the layer_types for upstream build_layer_cache_owned
  ModelArgs args;
  args.n_layers(num_layers);
  args.layer_types(layer_types);

  const LayerwiseSplitLayout layout(
      /*enabled=*/true,
      layerwise_split_size,
      rank % layerwise_split_size);

  auto owned = build_layer_cache_owned(args, layout, num_layers);

  int64_t owned_count = 0;
  int64_t linear_count = 0;
  for (int64_t i = 0; i < num_layers; ++i) {
    if (owned[static_cast<size_t>(i)]) ++owned_count;
    if (!is_full_attention_layer(args, i)) ++linear_count;
  }

  LOG(INFO) << "[Worker " << rank << "] Layerwise split (size="
            << layerwise_split_size << "): " << owned_count << "/"
            << num_layers << " layers owned, " << linear_count
            << " linear-attention (always owned).";

  return owned;
}

}  // namespace xllm
