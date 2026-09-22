/* Migrated from xllm/core/framework/config/parallel_config.h
   Stripped json / option_category — project_6 uses gflags directly. */

#pragma once

#include <cstdint>
#include <string>

#include "common/macros.h"

namespace xllm {

class ParallelConfig final {
 public:
  ParallelConfig() = default;

  static ParallelConfig& get_instance();
  void from_flags();

  PROPERTY(int32_t, dp_size) = 1;
  PROPERTY(int32_t, ep_size) = 1;
  PROPERTY(int32_t, cp_size) = 1;
  PROPERTY(int32_t, layerwise_split_size) = 1;
  PROPERTY(int32_t, kv_split_size) = 1;
  PROPERTY(int64_t, tp_size) = 1;
  PROPERTY(std::string, communication_backend) = "nccl";
  PROPERTY(bool, enable_multi_stream_parallel) = false;

  [[nodiscard]] int32_t kv_split_size_effective() const noexcept {
    return kv_split_size_ > 0 ? kv_split_size_ : cp_size_;
  }
};

}  // namespace xllm
