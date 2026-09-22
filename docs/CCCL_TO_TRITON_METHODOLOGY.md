# CCCL → Triton 方法论迁移

> 核心观点: CCCL 的 policy_selector 和 %RANGE% benchmark 框架是 NVIDIA 几十年 GPU 性能优化的结晶。
> 竞赛中几万人都在用同一份 EngineX 代码调参数。我们的差异化来自 CCCL 的方法论——不是复制参数，是复制思维方式。

---

## 一、CCCL 的 tuning 方法论

NVIDIA 在 CCCL 中的参数搜索基础设施:

```
%RANGE% TUNE_ITEMS_PER_THREAD ipt 7:24:1      ← 每线程处理的元素数
%RANGE% TUNE_THREADS_PER_BLOCK tpb 128:1024:32 ← 每 CTA 的线程数
%RANGE% TUNE_ITEMS_PER_VEC_LOAD_POW2 ipv 1:2:1 ← 向量化加载宽度
```

这些 %RANGE% 注释由 CCCL 的 benchmark runner 读取，生成笛卡尔积，每个组合跑 4 个 problem size，输出:
```
ipt_22.tpb_384.ns_1904.dcid_6.l2w_830.trp_1.ld_0 1.148 0.997 1.140 1.463
```

选几何均值最高的组合写入 policy_selector。

**关键**: 不是人类凭经验猜参数，是系统化的笛卡尔积搜索 + 实测数据驱动。

---

## 二、EngineX Triton 的参数对应

### 2.1 prefix_prefill.py (Context Attention — 影响 Input TPS 14%)

CCCL scan 的 %RANGE%:
```
ipt 7:24:1      → Triton: BLOCK_M ∈ {32, 64, 128, 256}
tpb 128:1024:32 → Triton: num_warps ∈ {2, 4, 8, 16} (warps × 32 = threads)
ns 0:2048:4     → BI-V100 不适用 (Triton 没有 delay policy)
trp 0:1:1       → BI-V100 不适用 (Triton 自动选择 memory layout)
ld 0:1:1        → BI-V100 不适用 (Triton 自动选择 cache modifier)
```

| CCCL 参数 | Triton 参数 | 当前值 | 搜索范围 |
|-----------|-----------|--------|---------|
| items_per_thread | BLOCK_M (和 BLOCK_N) | 128 or 64 | {32, 64, 128} |
| threads_per_block | num_warps × 32 | 8×32=256 | {2,4,8}×32 |
| N/A | num_stages | 1 | {1, 2, 3} |

### 2.2 triton_flash_attention.py (Decode Attention — 影响 Output TPS 83%)

CCCL reduce 的 %RANGE%:
```
ipt 7:24:1  → BLOCK_M ∈ {16, 32, 64, 128, 256}
tpb 128:1024:32 → num_warps ∈ {4, 8}
ipv 1:2:1   → PRE_LOAD_V ∈ {True, False}
```

| 当前 Triton config | CCCL 对应 | BI-V100 评估 |
|-------------------|----------|-------------|
| BLOCK_M=256, BLOCK_N=64, warps=8 | ipt=高, tpb=高 | ⚠️ SMEM 可能不够 |
| BLOCK_M=128, BLOCK_N=128, warps=4 | ipt=中, tpb=低 | ✓ 可能最优 |
| BLOCK_M=128, BLOCK_N=64, warps=4 | ipt=中, tpb=低 | ✓ 安全 |
| BLOCK_M=64, BLOCK_N=64, warps=8 | ipt=低, tpb=高 | ✓ 安全 |
| BLOCK_M=32, BLOCK_N=32, warps=8 | ipt=极低, tpb=高 | ✓ 保守 |
| BLOCK_M=16, BLOCK_N=16, warps=4 | ipt=极低, tpb=低 | ✓ 最保守 |

### 2.3 fused_moe.py (MoE GEMM — Qwen3.6 的核心瓶颈)

CCCL 没有直接的 MoE tuning，但 transform 和 reduce 的参数搜索逻辑适用:

| Triton 参数 | 当前值 (batch≤8) | 搜索范围 | CCCL 类比 |
|-----------|----------------|---------|---------|
| BLOCK_SIZE_M | 32 | {16, 32, 64} | threads_per_block 的 M 维度 |
| BLOCK_SIZE_N | 64 | {32, 64, 128} | items 的 N 维度 |
| BLOCK_SIZE_K | 32 | {32, 64, 128} | vec_size 的 K 维度 |
| GROUP_SIZE_M | 8 | {1, 4, 8} | CTA swizzle pattern |

---

## 三、执行计划: 从 CCCL benchmark runner 到 Triton autotune

### Step 1: 在 Phanthy Cloud 确认硬件参数 (阻塞一切)
```python
import torch
props = torch.cuda.get_device_properties(0)
print(f"SMEM: {props.max_shared_memory_per_block}")  # 32KB? 48KB?
print(f"SMs: {props.multi_processor_count}")          # 16? 50?
print(f"Warp size: {props.warp_size}")                 # 32?
```

### Step 2: prefix_prefill BLOCK/NUM_WARPS 网格搜索
```python
# 等价于 CCCL: %RANGE% TUNE_ITEMS ipt 32:128:32 × %RANGE% TUNE_THREADS tpb 64:256:32
for BLOCK in [32, 64, 128]:
    for NUM_WARPS in [2, 4, 8]:
        if BLOCK * 128 * 2 <= SMEM_LIMIT:  # SMEM check (CCCL scale_mem_bound 等价)
            measure_input_tps(BLOCK, NUM_WARPS)
```

### Step 3: fused_moe BLOCK_SIZE 网格搜索
```python
for M in [16, 32, 64]:
    for N in [32, 64, 128]:
        for K in [32, 64, 128]:
            if M * K * 2 + K * N * 2 <= SMEM_LIMIT:  # A_tile + B_tile
                measure_moe_latency(M, N, K)
```

### Step 4: triton_flash_attention 过滤不安全 configs
```python
# 从 CCCL 的 scale_mem_bound 逻辑: tile_bytes = BLOCK_M * head_dim * 2 (fp16)
safe_configs = [c for c in autotune_configs 
                if c.BLOCK_M * 128 * 2 <= SMEM_LIMIT]  # head_dim=128 for Qwen3.6
# 添加 BI-V100 特化 config
safe_configs.append(triton.Config(
    {'BLOCK_M': 64, 'BLOCK_N': 32, 'waves_per_eu': 2, 'PRE_LOAD_V': False},
    num_stages=1, num_warps=4
))
```

---

## 四、为什么这比其他参赛者的方法强

| 方法 | 其他参赛者 | 我们 |
|------|----------|------|
| 参数来源 | 猜 / 从 NVIDIA 博客抄 / 凭经验 | CCCL 27 个 tuning header 的 160+ 条 benchmark 注释 |
| 搜索策略 | 手动试几个值 | CCCL %RANGE% 笛卡尔积系统搜索 |
| SMEM 约束 | 运行时 crash 才发现 | CCCL scale_mem_bound 编译期检查 |
| 硬件适配 | 用 NVIDIA 默认值 | muh 27 个 BI-V100 policy_selector |
| MoE 调优 | 用 EngineX 默认 config | 从 CCCL partition + select_if 逻辑指导 MoE tile 选择 |
