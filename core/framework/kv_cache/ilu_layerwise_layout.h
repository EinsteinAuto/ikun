/* BI-V100-specific detailed per-layer shard descriptors.
   This is the 20% adaptation on top of upstream LayerwiseSplitLayout. */

#pragma once

#include <cstdint>
#include <numeric>
#include <vector>

#include <glog/logging.h>

namespace xllm {

/// Per-layer KV shard descriptor for BI-V100.
struct LayerShardSpec {
  int64_t layer_id = -1;
  std::vector<int32_t> assigned_ranks;
  std::vector<int64_t> heads_per_rank;

  int64_t total_heads() const {
    return std::accumulate(heads_per_rank.begin(), heads_per_rank.end(),
                           int64_t{0});
  }

  bool is_valid() const {
    if (layer_id < 0 || assigned_ranks.empty()) return false;
    if (assigned_ranks.size() != heads_per_rank.size()) return false;
    for (auto h : heads_per_rank) {
      if (h <= 0) return false;
    }
    return true;
  }
};

/// Full BI-V100-specific layout: one LayerShardSpec per model layer.
class IluLayerwiseLayout {
 public:
  IluLayerwiseLayout() = default;
  explicit IluLayerwiseLayout(std::vector<LayerShardSpec> specs)
      : specs_(std::move(specs)) { validate(); }

  int64_t num_layers() const { return static_cast<int64_t>(specs_.size()); }

  const LayerShardSpec& layer_spec(int64_t lid) const {
    CHECK_GE(lid, 0);
    CHECK_LT(lid, num_layers());
    return specs_[lid];
  }

  bool rank_owns_layer(int32_t rank, int64_t lid) const {
    for (auto r : specs_[lid].assigned_ranks)
      if (r == rank) return true;
    return false;
  }

  int64_t heads_for_rank(int32_t rank, int64_t lid) const {
    const auto& s = specs_[lid];
    for (size_t i = 0; i < s.assigned_ranks.size(); ++i)
      if (s.assigned_ranks[i] == rank) return s.heads_per_rank[i];
    return 0;
  }

  int64_t layers_on_rank(int32_t rank) const {
    int64_t n = 0;
    for (const auto& s : specs_)
      for (auto r : s.assigned_ranks)
        if (r == rank) { ++n; break; }
    return n;
  }

  void validate() const {
    for (int64_t i = 0; i < num_layers(); ++i) {
      CHECK(specs_[i].is_valid()) << "Invalid LayerShardSpec at " << i;
      CHECK_EQ(specs_[i].layer_id, i) << "Layer id mismatch at " << i;
    }
  }

  const std::vector<LayerShardSpec>& specs() const { return specs_; }

 private:
  std::vector<LayerShardSpec> specs_;
};

}  // namespace xllm
