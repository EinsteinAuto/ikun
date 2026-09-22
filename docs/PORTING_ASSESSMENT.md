# BI-V100 移植评估：全仓库编译目标清单

## 架构差异

| | NVIDIA V100 | Iluvatar BI-V100 |
|---|---|---|
| 架构标识 | `sm_70` | `ivcore10` |
| 编译器 | `nvcc` / `clang --cuda-gpu-arch=sm_70` | `corex clang/16 --cuda-gpu-arch=ivcore10` |
| 运行时编译 | `nvrtc` + `nvjitlink` | **不支持** |
| Driver API | `cuLibraryLoadData` / `cuLibraryGetKernel` | **不支持** |
| Tensor Core | HMMA (SM70) | **不支持** |
| Warp size | 32 | 32 (确认) |
| SMEM | 96KB (configurable) | 48KB |
| L2 Cache | 6MB | 不同 |
| SMs | 80 | 16 |
| CUB block-level | ✅ header-only | ✅ 可通过 corex clang 编译 |
| CUB device-level | ✅ via nvrtc JIT | ❌ 需要 AOT 替代方案 |

## 1. NVIDIA/CCCL (10,083 files)

### 1.1 c/parallel SHARED LIBRARY — cccl.c.parallel.so

**状态: ❌ 不能直接移植**

12 个算法全部依赖 NVRTC JIT 编译。每个 .cu 通过 `nvrtc_translation_unit` 生成源码，`-arch=sm_XX` 编译，`cuLibraryLoadData` 加载。

| 算法 | 源文件 | 行数 | NVRTC 依赖 | 移植方案 |
|---|---|---|---|---|
| reduce | reduce.cu | 783 | nvrtc × 30 | AOT: 直接调用 cub::DeviceReduce with corex |
| scan | scan.cu | 943 | nvrtc × 25 | AOT: cub::DeviceScan |
| radix_sort | radix_sort.cu | 947 | nvrtc × 24 | AOT: cub::DeviceRadixSort |
| merge_sort | merge_sort.cu | 763 | nvrtc × 25 | AOT: cub::DeviceMergeSort |
| transform | transform.cu | 1014 | nvrtc × 38 | AOT: cub::DeviceTransform |
| select_if | three_way_partition.cu | 697 | nvrtc × 29 | AOT: cub::DeviceSelect |
| histogram | histogram.cu | 858 | nvrtc × 18 | AOT: cub::DeviceHistogram |
| segmented_reduce | segmented_reduce.cu | 655 | nvrtc × 26 | AOT: cub::DeviceSegmentedReduce |
| segmented_sort | segmented_sort.cu | 1306 | nvrtc × 40 | AOT: cub::DeviceSegmentedSort |
| binary_search | binary_search.cu | 547 | nvrtc × 8 | AOT: cub::DeviceBinarySearch |
| unique_by_key | unique_by_key.cu | 768 | nvrtc × 19 | AOT: cub::DeviceUniqueByKey |
| for | for.cu | 426 | nvrtc × 15 | AOT: cub::DeviceFor |

**移植策略**: 不搬 c/parallel，而是直接用 CUB header-only API 写 AOT .cu 文件，用 corex clang 编译成 .so。每个算法 = 一组固定类型特化。

### 1.2 c/parallel.v2 SHARED LIBRARY

**状态: ❌ 不能直接移植 (依赖 hostjit/libnvcc)**

v2 用嵌入式 clang 做 JIT，不用 nvrtc。理论上可以用 corex clang 替换 libnvcc 的 clang，但改造量大。

### 1.3 CUB block/warp/thread 原语 (header-only)

**状态: ✅ 可直接使用**

| 类别 | 文件数 | 说明 |
|---|---|---|
| block primitives | 25 .cuh | BlockReduce, BlockScan, BlockSort, BlockLoad, BlockStore 等 |
| warp primitives | 17 .cuh | WarpReduce, WarpScan, WarpSort 等 |
| thread primitives | 8 .cuh | ThreadReduce, ThreadScan, ThreadSort 等 |
| agent implementations | 26 .cuh | 每个 device algorithm 的 kernel 实现 |
| dispatch kernels | 17 .cuh | kernel launch 模板 |
| tuning policies | 27 .cuh | SM-specific 参数选择 (需适配 ivcore10) |

**移植策略**: `#include <cub/block/block_reduce.cuh>` 直接在 corex .cu 中使用。tuning policy 需要为 ivcore10 写新的参数表。

### 1.4 CUB/Thrust benchmarks + examples

| 类别 | 数量 | 移植状态 |
|---|---|---|
| CUB benchmarks | 82 | 需适配 ivcore10 编译 |
| CUB examples | 18 | 需适配 ivcore10 编译 |
| Thrust examples | 60 | 需适配 ivcore10 编译 |
| Thrust benchmarks | 75 | 需适配 ivcore10 编译 |
| cudax examples | 68 | 依赖 cudax runtime，暂不移植 |
| libcudacxx benchmarks | 62 | 需适配 ivcore10 编译 |

---

## 2. NVIDIA/CUTLASS (7,787 files)

### 2.1 核心 GEMM 库 (header-only)

**状态: ⚠️ 部分可移植**

| SM 架构 | 文件数 | BI-V100 兼容 |
|---|---|---|
| SM70 (Volta SIMT) | ~20 | ✅ 需验证 ivcore10 兼容性 |
| SM75 (Turing) | ~30 | ⚠️ 部分 (SIMT mode) |
| SM80 (Ampere Tensor) | ~200 | ❌ 需要 HMMA |
| SM90 (Hopper) | ~300 | ❌ |
| SM100/120 (Blackwell) | ~200 | ❌ |

### 2.2 Grouped GEMM (MoE 核心)

| Example | 文件 | SM 要求 | 移植状态 |
|---|---|---|---|
| 24_gemm_grouped | gemm_grouped.cu | SM70+ SIMT | ✅ 可移植 |
| 57_hopper_grouped_gemm | — | SM90 | ❌ |
| 64_ada_fp8_gemm_grouped | — | SM89 | ❌ |
| 92_blackwell_moe_gemm | — | SM100 | ❌ |

**移植策略**: example 24 (SIMT grouped GEMM) 是唯一能在 BI-V100 跑的。搬过来，接口适配到 xllm group_gemm。

### 2.3 编译目标汇总

| 类别 | 数量 |
|---|---|
| Example executables | 164 .cu |
| Test executables | 862 .cu |
| Include headers | 785 |
| SM70 兼容子集 | ~20 examples + ~50 tests |

---

## 3. Dao-AILab/flash-attention (606 .cu files)

### 3.1 flash_attn_2_cuda.so

**状态: ❌ 不能直接移植 (SM80+ Tensor Core)**

所有 kernel 使用 `cute::MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>` — 依赖 Ampere Tensor Core。

| Kernel 类别 | .cu 数量 | SM 要求 |
|---|---|---|
| SM80 fwd | 48 | ❌ Tensor Core |
| SM80 bwd | 24 | ❌ Tensor Core |
| SM80 fwd_split | 48 | ❌ Tensor Core |
| SM80 fwd_split_align | 42 | ❌ Tensor Core |
| Hopper (SM90+) | 453 | ❌ |

### 3.2 可用的算法模板

| 文件 | 行数 | 价值 |
|---|---|---|
| flash_fwd_kernel.h | 1301 | attention 算法流程 (Q×K softmax V) |
| softmax.h | 189 | online softmax 实现 |
| kernel_traits.h | 344 | SMEM/register 分配策略 |
| mask.h | 214 | causal mask 实现 |
| rotary.h | 153 | RoPE in-kernel 实现 |

**移植策略**: 不搬 .cu kernel（依赖 Tensor Core），搬算法模板头文件，基于 CUB block primitives 重写 SIMT attention kernel for ivcore10。或者直接用 ixformer base image 的 `ixinfer_flash_attn_unpad_with_block_tables`（已编译好）。

### 3.3 Layer Norm kernels

| 类别 | .cu 数量 | SM 要求 |
|---|---|---|
| ln_fwd | 14 (256~8192 width) | ✅ 纯 SIMT |
| ln_bwd | 14 | ✅ 纯 SIMT |
| ln_parallel_fwd | 14 | ✅ 纯 SIMT |
| ln_parallel_bwd | 14 | ✅ 纯 SIMT |

**移植策略**: Layer norm kernel 是纯 SIMT，不依赖 Tensor Core。可直接用 corex clang 编译。hidden_size=5120 对应 ln_fwd_5120.cu。

---

## 4. jd-opensource/xllm (全平台推理引擎)

### 4.1 ILU (BI-V100) 专用代码

**状态: ✅ 已在项目中 (upstream_ref + ex_engine)**

| 文件 | 行数 | 作用 | 状态 |
|---|---|---|---|
| ilu/activation.cpp | 32 | silu_and_mul → ixformer::infer | ✅ 已搬 |
| ilu/norm.cpp | 50 | rms_norm → ixformer::infer | ✅ 已搬 |
| ilu/rope.cpp | 31 | rotary_embedding → ixformer::infer | ✅ 已搬 |
| ilu/attention.cpp | 162 | prefill + decode → ixformer::infer | ✅ 已搬 |
| ilu/fused_moe.cpp | 99 | topk + expand + combine → ixformer::infer | ✅ 已搬 |
| ilu/group_gemm.cpp | 39 | group_gemm → ixformer::infer | ✅ 已搬 |
| ilu/matmul.cpp | 73 | linear → ixformer::infer | ✅ 已搬 |
| ilu/ixformer.h | 147 | 完整 ixformer::infer API 声明 | ✅ 已搬 |
| ilu/ilu_ops_api.h | 153 | xllm kernel 层 API | ✅ 已搬 |
| ilu/utils.h | 62 | 工具函数 | ✅ 已搬 |
| layers/ilu/fused_moe.cpp | 806 | 完整 MoE 7步 pipeline | ✅ 已搬 |
| layers/ilu/attention.cpp | 189 | attention layer 封装 | ✅ 已搬 |

### 4.2 CUDA kernels (SM-agnostic)

| 文件 | 行数 | SM 限制 | 状态 |
|---|---|---|---|
| activation.cu | 188 | 无 | ✅ 已搬 |
| norm.cu | 600 | 需 cub::BlockReduce | ✅ 已搬 |
| rope.cu | 258 | 无 | ✅ 已搬 |
| block_copy.cu | 209 | 无 | ✅ 已搬 |
| reshape_paged_cache.cu | 101 | 无 | ✅ 已搬 |
| moe/moe_topk_softmax_kernels.cuh | 867 | 无 | ✅ 已搬 |
| moe/moe_compute_index.cu | 155 | 无 | ✅ 已搬 |
| moe/moe_combine.cu | 105 | 无 | ✅ 已搬 |
| moe/moe_fused_topk.cu | 59 | 无 | ✅ 已搬 |

### 4.3 CUDA kernels (SM80+ only)

| 文件 | 行数 | SM 限制 | 移植方案 |
|---|---|---|---|
| fused_qknorm_rope.cu | 473 | SM80 (`__CUDA_ARCH__ >= 800`) | 拆出 SIMT 部分 |
| fp8_quant_utils.cuh | 239 | SM89 (`__CUDA_ARCH__ >= 890`) | 不适用 |
| cutlass_w8a8/*.cu | ~400 | SM90/100/120 | 不适用 |

### 4.4 其他平台代码 (参考用)

| 平台 | kernel 文件数 | layer 文件数 | 说明 |
|---|---|---|---|
| DCU (AMD ROCm) | 14 | 12 | GDN 完整实现可参考 |
| MLU (Cambricon) | 21 | 35 | GDN + MoE 最完整 |
| MUSA (Moore Threads) | 14 | 12 | GDN kernel 最近代 |
| NPU (Ascend) | 30+ | 30+ | tilelang GDN 可参考 |

---

## 5. fla-org/flash-linear-attention (349 Triton kernels)

### 5.1 GatedDeltaNet 专用 kernels

**状态: ⚠️ 需验证 Triton 在 BI-V100 上是否工作**

| 文件 | @triton.jit | 行数 | 说明 |
|---|---|---|---|
| chunk_fwd.py | 2 | 428 | GDN 前向 chunk (核心) |
| fused_recurrent.py | 2 | 478 | GDN decode (单步) |
| wy_fast.py | 4 | 351 | WY representation |
| gate.py | 6 | 344 | gate cumsum |

### 5.2 通用 Triton 算子

| 目录 | kernel 数 | 说明 |
|---|---|---|
| common/ | 36 | chunk_h, chunk_o, fused_recurrent (所有 linear attention 共享) |
| utils/ | 44 | cumsum, softmax, matmul, solve_tril |
| gated_delta_rule/ | 14 | GDN 专用 |
| gdn2/ | 12 | GDN v2 (新版) |
| kda/ | 24 | Key-dependent attention |
| delta_rule/ | 12 | 原始 delta rule |
| gla/ | 18 | Gated Linear Attention |

### 5.3 Backend 分发

| Backend | SM 要求 | 说明 |
|---|---|---|
| FlashQLA | SM90+ | ❌ 不适用 BI-V100 |
| Triton (default) | 任意 GPU | ⚠️ 需验证 corex Triton |
| triton_ascend | Ascend NPU | ❌ 不适用 |

---

## 移植优先级

### P0 — 直接可编译 (corex clang ivcore10)

1. **xllm CUDA kernels** (9 files, 2542 lines) — 已搬，需在真机编译测试
2. **CUB block/warp headers** — 已在 cccl_upstream/，可直接 #include
3. **ix_moe_bridge.so + ix_attn_bridge.so** — pybind11 桥接 ixformer::infer

### P1 — 需适配后可编 (改 SM 架构 + tuning 参数)

4. **FlashAttention layer_norm kernels** (56 .cu) — 纯 SIMT，改编译 flag
5. **CUTLASS SM70 SIMT GEMM** (example 24 grouped_gemm) — MoE group_gemm 替代方案
6. **CUB tuning policies** (27 .cuh) — 为 ivcore10 写参数表 (SMEM=48KB, SM=16)

### P2 — 需要重写 (算法可用，硬件指令不兼容)

7. **FlashAttention fwd kernel** — 基于算法模板用 CUB BlockReduce 重写 SIMT 版
8. **CCCL c/parallel AOT 版** — 绕过 NVRTC，直接用 CUB device API + corex 编译
9. **FLA Triton GDN kernels** — 需验证 Triton on corex 可行性

### P3 — 不移植

10. FlashAttention SM80+ Tensor Core kernels
11. CUTLASS SM80/90/100/120 kernels
12. CCCL nvrtc/nvjitlink 依赖代码
13. xllm fp8/cutlass_w8a8 quantization kernels
