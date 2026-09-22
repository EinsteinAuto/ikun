/* Migrated from xllm/core/framework/parallel_state/parallel_args.h (251 lines)
   Stripped ProcessGroup, mapping_data, vendor comm — scheduling layer only. */

#pragma once

#include <cstdint>

#include "common/macros.h"

namespace xllm {

class ParallelArgs {
 public:
  ParallelArgs() = default;
  ParallelArgs(int32_t rank, int32_t world_size)
      : rank_(rank), world_size_(world_size) {}
  ParallelArgs(int32_t rank, int32_t world_size, int32_t dp_size,
               int32_t cp_size, int32_t layerwise_split_size)
      : rank_(rank), world_size_(world_size), dp_size_(dp_size),
        cp_size_(cp_size), layerwise_split_size_(layerwise_split_size) {}

  PROPERTY(int32_t, rank) = 0;
  PROPERTY(int32_t, world_size) = 1;
  PROPERTY(int32_t, dp_size) = 1;
  PROPERTY(int32_t, cp_size) = 1;
  PROPERTY(int32_t, layerwise_split_size) = 1;
};

}  // namespace xllm
