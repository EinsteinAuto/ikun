# BI-V100 实际可调参数面（Honest Assessment）

> 最后更新: 2026-08-03
> 基于 `vllm/_custom_ops.py` 中 ixf_F 调用的逐行分析

---

## 事实 1: ixformer 预编译 kernel 不接受大部分调参

所有 decode 热路径的 CUDA kernel 打包在 `ixformer.functions` 里。Python 侧
只传入 tensor 和少量标量，**不传入 block size / items_per_thread / load_algorithm**。

| ixf_F 调用 | Python 传入的调参 | **不接受的参数** |
|---|---|---|
| `vllm_single_query_cached_kv_attention` | scale, block_size, max_context_len | threads_per_block, items_per_thread, reduce_algorithm |
| `vllm_invoke_fused_moe_kernel` | **仅 BLOCK_SIZE_M** | BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M |
| `silu_and_mul` / `rms_norm` / `rotary_embedding` | 无调参 | 一切 |
| `copy_blocks` | 无调参 | 一切 |

## 事实 2: 实际可调的 5 个参数

| # | 参数 | 文件 | 当前值 | 影响 |
|---|------|------|--------|------|
| 1 | `BLOCK_SIZE_M` | fused_moe.py → _custom_ops.py | 16/64/256 (heuristic) | MoE GEMM 的 M 维 tile，传给 ixformer |
| 2 | `use_v1` / V1-V2 threshold | paged_attn.py:126-128 | True (hardcoded) | decode attention 选路 (V2 is NotImplementedError) |
| 3 | `BLOCK` / `NUM_WARPS` | prefix_prefill.py:726-728 | 64 / 4 | Triton prefill kernel **（真正的 JIT，可调）** |
| 4 | `get_max_shared_memory` | _custom_ops.py:891 | 32 * 1024 | 影响 Triton 编译器的 SMEM 分配上限 |
| 5 | `triton.Config` autotune set | triton_flash_attention.py:212-303 | 8 个 AMD 风格 config | Triton flash attention **（JIT，autotune 自选最优）** |

## 事实 3: V2 是 NotImplementedError

`paged_attention_v2` 直接 `raise NotImplementedError()`。对 paged_attn.py 的
V1/V2 heuristic 修改**对实际性能没有影响**，因为 V2 永远不会执行。`use_v1 = True`
硬编码是正确的防御措施。

我的 patch 移除这个硬编码是**错误的**——如果 V2 被触发会导致运行时 crash。

## 事实 4: bench_bi100.py 的 benchmark 函数全部无效

`bench_reduce(point, ...)` 接收 `point` 参数但**没有注入到 kernel 里**。
`torch.sum(x)` 调用 PyTorch 的内置 reduce，不是 CUB。所有 variant 执行同一个
kernel，speedup 恒等于 1.0。

bench_bi100.py 的空间分析功能（`--prune-only`）是有效的。benchmark 功能需要
重写为针对 **Triton JIT kernel 的实际参数注入 benchmark**。

## 事实 5: 真正有竞争力的调优路径

1. **prefix_prefill.py 的 Triton kernel**：3 个 `@triton.jit` 函数，
   `BLOCK_M/BLOCK_N` 是 `tl.constexpr`，Triton JIT 编译器会为每组
   constexpr 值编译独立的 kernel binary。**这是真正能改 kernel 的地方。**

2. **triton_flash_attention.py 的 autotune**：`@triton.autotune` 会
   实际跑每个 Config 并选最快的。**添加 BI-V100 适配 config 是有效的。**

3. **computility-run.yaml 的 vllm 启动参数**：`max_num_seqs`、
   `max_num_batched_tokens`、`enable_chunked_prefill` 等。
   这些在引擎级别影响 batch 策略和内存分配。

4. **BLOCK_SIZE_M**（fused_moe）：唯一传给 ixformer 的 tile 参数。
   值得 benchmark 不同 M 值（16/32/64/128/256）。

## 需要撤回的修改

| 文件 | 修改 | 状态 |
|------|------|------|
| paged_attn.py | 移除 use_v1=True | **应撤回** — V2 是 NotImplementedError |
| fused_moe.py | BLOCK_SIZE_K 32→64, BLOCK_SIZE_N 32→64 | **无效** — ixformer 不读这两个值 |
| _custom_ops.py | SMEM 32→48KB | 待确认 — 影响 Triton 编译但不影响 ixformer |
| prefix_prefill.py | 注释增强 | 无害，保留 |
| triton_flash_attention.py | 添加 2 个 config | **有效** — autotune 会实际测试 |
