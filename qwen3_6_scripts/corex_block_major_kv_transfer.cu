#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

constexpr int kAttentionLayers = 10;
constexpr int kKvPlanes = 2;
constexpr int kElementsPerPlaneBlock = 4096;
constexpr int kElementsPerVector = 8;
constexpr int kVectorsPerPlaneBlock =
    kElementsPerPlaneBlock / kElementsPerVector;
constexpr int kVectorsPerBlockMajorRow =
    kAttentionLayers * kKvPlanes * kVectorsPerPlaneBlock;
constexpr int kThreads = 256;
constexpr int kMaxGridBlocks = 65535;

using PackedVector = uint4;

__device__ __forceinline__ const PackedVector* select_const_layer(
    int layer, const PackedVector* layer0, const PackedVector* layer1,
    const PackedVector* layer2, const PackedVector* layer3,
    const PackedVector* layer4, const PackedVector* layer5,
    const PackedVector* layer6, const PackedVector* layer7,
    const PackedVector* layer8, const PackedVector* layer9) {
  switch (layer) {
    case 0:
      return layer0;
    case 1:
      return layer1;
    case 2:
      return layer2;
    case 3:
      return layer3;
    case 4:
      return layer4;
    case 5:
      return layer5;
    case 6:
      return layer6;
    case 7:
      return layer7;
    case 8:
      return layer8;
    default:
      return layer9;
  }
}

__device__ __forceinline__ PackedVector* select_mutable_layer(
    int layer, PackedVector* layer0, PackedVector* layer1,
    PackedVector* layer2, PackedVector* layer3, PackedVector* layer4,
    PackedVector* layer5, PackedVector* layer6, PackedVector* layer7,
    PackedVector* layer8, PackedVector* layer9) {
  switch (layer) {
    case 0:
      return layer0;
    case 1:
      return layer1;
    case 2:
      return layer2;
    case 3:
      return layer3;
    case 4:
      return layer4;
    case 5:
      return layer5;
    case 6:
      return layer6;
    case 7:
      return layer7;
    case 8:
      return layer8;
    default:
      return layer9;
  }
}

__global__ void pack_block_major_kernel(
    const PackedVector* layer0, const PackedVector* layer1,
    const PackedVector* layer2, const PackedVector* layer3,
    const PackedVector* layer4, const PackedVector* layer5,
    const PackedVector* layer6, const PackedVector* layer7,
    const PackedVector* layer8, const PackedVector* layer9,
    const int* source_blocks, PackedVector* staging, int* error_flag,
    int count, int gpu_blocks) {
  const int64_t total =
      static_cast<int64_t>(count) * kVectorsPerBlockMajorRow;
  for (int64_t linear =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < total;
       linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int64_t cursor = linear;
    const int feature_vector = cursor % kVectorsPerPlaneBlock;
    cursor /= kVectorsPerPlaneBlock;
    const int kv_plane = cursor % kKvPlanes;
    cursor /= kKvPlanes;
    const int layer = cursor % kAttentionLayers;
    const int row = cursor / kAttentionLayers;
    const int source_block = source_blocks[row];
    if (static_cast<unsigned int>(source_block) >=
        static_cast<unsigned int>(gpu_blocks)) {
      atomicExch(error_flag, 1);
      continue;
    }
    const PackedVector* source = select_const_layer(
        layer, layer0, layer1, layer2, layer3, layer4, layer5, layer6,
        layer7, layer8, layer9);
    const int64_t source_index =
        ((static_cast<int64_t>(kv_plane) * gpu_blocks + source_block)
         * kVectorsPerPlaneBlock) +
        feature_vector;
    staging[linear] = source[source_index];
  }
}

__global__ void scatter_block_major_kernel(
    const PackedVector* staging, const int* destination_blocks,
    PackedVector* layer0, PackedVector* layer1, PackedVector* layer2,
    PackedVector* layer3, PackedVector* layer4, PackedVector* layer5,
    PackedVector* layer6, PackedVector* layer7, PackedVector* layer8,
    PackedVector* layer9, int* error_flag, int count, int gpu_blocks) {
  const int64_t total =
      static_cast<int64_t>(count) * kVectorsPerBlockMajorRow;
  for (int64_t linear =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < total;
       linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int64_t cursor = linear;
    const int feature_vector = cursor % kVectorsPerPlaneBlock;
    cursor /= kVectorsPerPlaneBlock;
    const int kv_plane = cursor % kKvPlanes;
    cursor /= kKvPlanes;
    const int layer = cursor % kAttentionLayers;
    const int row = cursor / kAttentionLayers;
    const int destination_block = destination_blocks[row];
    if (static_cast<unsigned int>(destination_block) >=
        static_cast<unsigned int>(gpu_blocks)) {
      atomicExch(error_flag, 1);
      continue;
    }
    PackedVector* destination = select_mutable_layer(
        layer, layer0, layer1, layer2, layer3, layer4, layer5, layer6,
        layer7, layer8, layer9);
    const int64_t destination_index =
        ((static_cast<int64_t>(kv_plane) * gpu_blocks + destination_block)
         * kVectorsPerPlaneBlock) +
        feature_vector;
    destination[destination_index] = staging[linear];
  }
}

void check_gpu_layers(const std::vector<torch::Tensor>& layers) {
  TORCH_CHECK(layers.size() == kAttentionLayers, "expected exactly ",
              kAttentionLayers, " GPU attention-layer tensors");
  const auto device = layers.front().device();
  const int64_t blocks = layers.front().size(1);
  for (int layer = 0; layer < kAttentionLayers; ++layer) {
    const auto& tensor = layers[layer];
    TORCH_CHECK(tensor.is_cuda(), "GPU layer ", layer,
                " must be a CUDA tensor");
    TORCH_CHECK(tensor.device() == device, "GPU layer ", layer,
                " is on a different device");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat16, "GPU layer ",
                layer, " must use float16");
    TORCH_CHECK(tensor.is_contiguous(), "GPU layer ", layer,
                " must be contiguous");
    TORCH_CHECK(tensor.dim() == 3 && tensor.size(0) == kKvPlanes &&
                    tensor.size(1) == blocks &&
                    tensor.size(2) == kElementsPerPlaneBlock,
                "GPU layer ", layer, " must have shape [2, blocks, 4096]");
    TORCH_CHECK(
        reinterpret_cast<uintptr_t>(tensor.data_ptr<at::Half>()) %
                alignof(PackedVector) ==
            0,
        "GPU layer ", layer, " is not 16-byte aligned");
  }
}

void check_gpu_transfer_args(const std::vector<torch::Tensor>& layers,
                             const torch::Tensor& block_ids,
                             const torch::Tensor& staging,
                             const torch::Tensor& error_flag,
                             int64_t count) {
  check_gpu_layers(layers);
  TORCH_CHECK(block_ids.is_cuda(), "block_ids must be a CUDA tensor");
  TORCH_CHECK(block_ids.device() == layers.front().device(),
              "block_ids must be on the cache device");
  TORCH_CHECK(block_ids.scalar_type() == torch::kInt32,
              "block_ids must use int32");
  TORCH_CHECK(block_ids.dim() == 1 && block_ids.is_contiguous(),
              "block_ids must be a contiguous one-dimensional tensor");
  TORCH_CHECK(count > 0 && count <= block_ids.numel(),
              "count must be in [1, block_ids.numel()]");
  TORCH_CHECK(staging.is_cuda(), "staging must be a CUDA tensor");
  TORCH_CHECK(staging.device() == layers.front().device(),
              "staging must be on the cache device");
  TORCH_CHECK(staging.scalar_type() == torch::kFloat16,
              "staging must use float16");
  TORCH_CHECK(staging.is_contiguous(), "staging must be contiguous");
  TORCH_CHECK(
      staging.dim() == 4 && staging.size(0) >= count &&
          staging.size(1) == kAttentionLayers &&
          staging.size(2) == kKvPlanes &&
          staging.size(3) == kElementsPerPlaneBlock,
      "staging must have shape [capacity>=count, 10, 2, 4096]");
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(staging.data_ptr<at::Half>()) %
              alignof(PackedVector) ==
          0,
      "staging is not 16-byte aligned");
  TORCH_CHECK(error_flag.is_cuda(),
              "error_flag must be a CUDA tensor");
  TORCH_CHECK(error_flag.device() == layers.front().device(),
              "error_flag must be on the cache device");
  TORCH_CHECK(error_flag.scalar_type() == torch::kInt32,
              "error_flag must use int32");
  TORCH_CHECK(error_flag.is_contiguous() && error_flag.numel() == 1,
              "error_flag must be one contiguous int32 value");
}

int launch_blocks(int64_t count) {
  const int64_t total = count * kVectorsPerBlockMajorRow;
  return static_cast<int>(std::min<int64_t>(
      (total + kThreads - 1) / kThreads, kMaxGridBlocks));
}

void pack_block_major(const std::vector<torch::Tensor>& layers,
                      const torch::Tensor& source_blocks,
                      torch::Tensor staging, torch::Tensor error_flag,
                      int64_t count) {
  check_gpu_transfer_args(
      layers, source_blocks, staging, error_flag, count);
  const int blocks = static_cast<int>(layers.front().size(1));
  pack_block_major_kernel<<<launch_blocks(count), kThreads, 0,
                            at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const PackedVector*>(
          layers[0].data_ptr<at::Half>()),
      reinterpret_cast<const PackedVector*>(
          layers[1].data_ptr<at::Half>()),
      reinterpret_cast<const PackedVector*>(
          layers[2].data_ptr<at::Half>()),
      reinterpret_cast<const PackedVector*>(
          layers[3].data_ptr<at::Half>()),
      reinterpret_cast<const PackedVector*>(
          layers[4].data_ptr<at::Half>()),
      reinterpret_cast<const PackedVector*>(
          layers[5].data_ptr<at::Half>()),
      reinterpret_cast<const PackedVector*>(
          layers[6].data_ptr<at::Half>()),
      reinterpret_cast<const PackedVector*>(
          layers[7].data_ptr<at::Half>()),
      reinterpret_cast<const PackedVector*>(
          layers[8].data_ptr<at::Half>()),
      reinterpret_cast<const PackedVector*>(
          layers[9].data_ptr<at::Half>()),
      source_blocks.data_ptr<int>(),
      reinterpret_cast<PackedVector*>(staging.data_ptr<at::Half>()),
      error_flag.data_ptr<int>(), static_cast<int>(count), blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void scatter_block_major(const torch::Tensor& staging,
                         const torch::Tensor& destination_blocks,
                         const std::vector<torch::Tensor>& layers,
                         torch::Tensor error_flag,
                         int64_t count) {
  check_gpu_transfer_args(
      layers, destination_blocks, staging, error_flag, count);
  const int blocks = static_cast<int>(layers.front().size(1));
  scatter_block_major_kernel<<<launch_blocks(count), kThreads, 0,
                               at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const PackedVector*>(
          staging.data_ptr<at::Half>()),
      destination_blocks.data_ptr<int>(),
      reinterpret_cast<PackedVector*>(layers[0].data_ptr<at::Half>()),
      reinterpret_cast<PackedVector*>(layers[1].data_ptr<at::Half>()),
      reinterpret_cast<PackedVector*>(layers[2].data_ptr<at::Half>()),
      reinterpret_cast<PackedVector*>(layers[3].data_ptr<at::Half>()),
      reinterpret_cast<PackedVector*>(layers[4].data_ptr<at::Half>()),
      reinterpret_cast<PackedVector*>(layers[5].data_ptr<at::Half>()),
      reinterpret_cast<PackedVector*>(layers[6].data_ptr<at::Half>()),
      reinterpret_cast<PackedVector*>(layers[7].data_ptr<at::Half>()),
      reinterpret_cast<PackedVector*>(layers[8].data_ptr<at::Half>()),
      reinterpret_cast<PackedVector*>(layers[9].data_ptr<at::Half>()),
      error_flag.data_ptr<int>(), static_cast<int>(count), blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void check_transfer_error(const torch::Tensor& error_flag) {
  TORCH_CHECK(error_flag.is_cuda(),
              "error_flag must be a CUDA tensor");
  TORCH_CHECK(error_flag.scalar_type() == torch::kInt32,
              "error_flag must use int32");
  TORCH_CHECK(error_flag.is_contiguous() && error_flag.numel() == 1,
              "error_flag must be one contiguous int32 value");
  TORCH_CHECK(error_flag.item<int>() == 0,
              "GPU block mapping contains an out-of-range id");
}

void check_cpu_transfer_args(const torch::Tensor& pool,
                             const torch::Tensor& block_ids,
                             const torch::Tensor& staging, int64_t count) {
  TORCH_CHECK(!pool.is_cuda() && !staging.is_cuda() &&
                  !block_ids.is_cuda(),
              "CPU gather/scatter tensors must be on CPU");
  TORCH_CHECK(pool.scalar_type() == torch::kFloat16 &&
                  staging.scalar_type() == torch::kFloat16,
              "CPU pool and staging must use float16");
  TORCH_CHECK(pool.is_contiguous() && staging.is_contiguous(),
              "CPU pool and staging must be contiguous");
  TORCH_CHECK(
      pool.dim() == 4 && pool.size(1) == kAttentionLayers &&
          pool.size(2) == kKvPlanes &&
          pool.size(3) == kElementsPerPlaneBlock,
      "CPU pool must have shape [slots, 10, 2, 4096]");
  TORCH_CHECK(
      staging.dim() == 4 && staging.size(0) >= count &&
          staging.size(1) == kAttentionLayers &&
          staging.size(2) == kKvPlanes &&
          staging.size(3) == kElementsPerPlaneBlock,
      "CPU staging must have shape [capacity>=count, 10, 2, 4096]");
  TORCH_CHECK(block_ids.scalar_type() == torch::kInt64,
              "CPU block_ids must use int64");
  TORCH_CHECK(block_ids.dim() == 1 && block_ids.is_contiguous(),
              "CPU block_ids must be contiguous and one-dimensional");
  TORCH_CHECK(count > 0 && count <= block_ids.numel(),
              "count must be in [1, block_ids.numel()]");

  const int64_t* ids = block_ids.data_ptr<int64_t>();
  for (int64_t row = 0; row < count; ++row) {
    TORCH_CHECK(ids[row] >= 0 && ids[row] < pool.size(0),
                "CPU block id out of range at row ", row, ": ", ids[row]);
  }
}

void cpu_gather_rows(const torch::Tensor& pool,
                     const torch::Tensor& source_blocks,
                     torch::Tensor staging, int64_t count) {
  check_cpu_transfer_args(pool, source_blocks, staging, count);
  const int64_t row_elements =
      kAttentionLayers * kKvPlanes * kElementsPerPlaneBlock;
  const size_t row_bytes =
      static_cast<size_t>(row_elements) * sizeof(at::Half);
  const char* source = reinterpret_cast<const char*>(
      pool.data_ptr<at::Half>());
  char* destination =
      reinterpret_cast<char*>(staging.data_ptr<at::Half>());
  const int64_t* ids = source_blocks.data_ptr<int64_t>();
  at::parallel_for(0, count, 8, [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      std::memcpy(destination + row * row_bytes,
                  source + ids[row] * row_bytes, row_bytes);
    }
  });
}

void cpu_scatter_rows(const torch::Tensor& staging,
                      torch::Tensor pool,
                      const torch::Tensor& destination_blocks,
                      int64_t count) {
  check_cpu_transfer_args(pool, destination_blocks, staging, count);
  const int64_t row_elements =
      kAttentionLayers * kKvPlanes * kElementsPerPlaneBlock;
  const size_t row_bytes =
      static_cast<size_t>(row_elements) * sizeof(at::Half);
  const char* source = reinterpret_cast<const char*>(
      staging.data_ptr<at::Half>());
  char* destination =
      reinterpret_cast<char*>(pool.data_ptr<at::Half>());
  const int64_t* ids = destination_blocks.data_ptr<int64_t>();
  at::parallel_for(0, count, 8, [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      std::memcpy(destination + ids[row] * row_bytes,
                  source + row * row_bytes, row_bytes);
    }
  });
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("pack", &pack_block_major,
             "Pack ten layer-major FP16 KV caches into block-major staging");
  module.def("scatter", &scatter_block_major,
             "Scatter block-major FP16 staging into ten layer-major caches");
  module.def("check_error", &check_transfer_error,
             "Fail fast after a bounds-safe asynchronous transfer");
  module.def("cpu_gather", &cpu_gather_rows,
             "Gather block-major CPU pool rows into bounded staging");
  module.def("cpu_scatter", &cpu_scatter_rows,
             "Scatter bounded staging rows into the block-major CPU pool");
}
