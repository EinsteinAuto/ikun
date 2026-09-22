#pragma once

#include <string>

namespace ixformer::comm {

enum class AllGatherAlgo {
    kNone,
    kAuto,
    kNCCL,
    kNumAlgo
};

enum class AllReduceAlgo {
    kNone,          // None
    kAuto,          // 自动选择算法
    kAllGatherSum,  // 针对小数据量的算法
    kBroadcastSum,  // 针对小数据量的算法
    kRing,          // Ring AllReduce
    kQuant,         // 对通讯算法进行量化，默认使用 kQuantL1
    kQuantL1,       // 对通讯算法进行量化，优先使用量化算法以及最大保留精度，在部分 Size 性能不佳时，退化为 Auto 算法
    kQuantL2,       // 对通讯算法进行量化，优先使用量化算法以及最大化速度，在部分 Size 性能不佳时，退化为 Auto 算法
    kQuantL1AllSize,// 对所有的 Size 都使用量化算法
    kQuantL2AllSize,// 对所有的 Size 都使用量化算法
    kNCCL,          // 使用 NCCL
    kStride,        // 输入或输出的 Tensor 不是连续的
    kNumAlgo
};


enum class BroadcastAlgo {
    kNone,
    kAuto,
    kNCCL,
    kNumAlgo
};

enum class GatherAlgo {
    kNone,
    kAuto,
    kNCCL,
    kNumAlgo
};

enum class SendAlgo {
    kNone,
    kAuto,
    kNCCL,
    kNumAlgo
};

typedef SendAlgo RecvAlgo;

enum class ReduceAlgo {
    kNone,
    kAuto,
    kNCCL,
    kNumAlgo
};

enum class ReduceScatterAlgo {
    kNone,
    kAuto,
    kNCCL,
    kNumAlgo
};


std::string to_string(AllGatherAlgo algo);
std::string to_string(AllReduceAlgo algo);
std::string to_string(BroadcastAlgo algo);
std::string to_string(GatherAlgo algo);
std::string to_string(SendAlgo algo);
std::string to_string(ReduceAlgo algo);
std::string to_string(ReduceScatterAlgo algo);

template<typename Algo>
Algo get_algo_from_str(const std::string &name);

}// namespace ixformer::comm
