# Layerwise Split KV Cache Sharding

**Commit:** 494f293b5629 · **PR:** #2260 · **Upstream:** xLLM  
**Adaptation:** Iluvatar BI-V100 (PCIe topology)  
**LOC:** +455 −10 across 19 files

## Problem

For models with heterogeneous layer structures (e.g., DeepSeek-V3 with dense
attention layers interleaved with MoE layers), the KV cache is uniformly
sharded across all tensor-parallel ranks.  Each rank stores KV for all layers,
even though different layers may have vastly different head counts.

On Iluvatar BI-V100 (32 GB HBM per card), this wastes memory on ranks that
serve layers with fewer KV heads and prevents optimal utilisation of each
card's HBM.

## Solution

Introduce **layerwise split KV cache sharding**: a new KV cache layout
strategy where each layer's KV cache can be sharded independently across a
configurable subset of TP ranks.

### Key Components

| # | Component | Files | Description |
|---|-----------|-------|-------------|
| 1 | `LayerwiseSplitLayout` | `core/framework/kv_cache/layerwise_split_layout.h` | Per-layer KV shard mappings.  Dense layers spread across all TP ranks; MoE layers concentrate on fewer ranks. |
| 2 | Layerwise allocation | `core/framework/kv_cache/kv_cache_layerwise.{h,cpp}` | `allocate_kv_caches_layerwise()` — allocates per-layer shard sizes from layout. Handles the ILU/MLU transposed cache layout `[n_blocks, n_heads, block_size, head_dim]`. |
| 3 | Memory estimation | `core/framework/kv_cache/kv_cache_estimation_layerwise.{h,cpp}` | Reports peak/average per-rank memory; computes savings vs uniform. |
| 4 | ILU topology mapping | `core/framework/parallel_state/mapping_ilu.{h,cpp}` | PCIe-aware assignment: MoE layers placed on ranks sharing a PCIe switch to maximise intra-group bandwidth. |
| 5 | Engine integration | `core/distributed_runtime/layerwise_split_{engine_ext,master}.{h,cpp}` | Master computes layout at startup; engines propagate to workers. |
| 6 | Worker init | `core/runtime/worker_layerwise_init.{h,cpp}` | Workers receive and apply per-layer KV shard assignments. |
| 7 | Config flag | `core/config/parallel_config_layerwise.{h,cpp}` | `--enable_layerwise_split` gflag (default: false). |

### Iluvatar BI-V100 Hardware Context (verified via ixsmi + debug_warpsize.py)

- **4× BI-V100**, Bus-Id `4B:00.0` – `4E:00.0`, NUMA node 1
- **Warp size: 64** (NOT 32 — verified via CUDA kernel `warpSize` builtin)
- **32768 MiB HBM** per card, 1500 MHz SM clock, 1200 MHz mem clock
- **Flat PIX topology** — all pairs connected via single PCIe bridge (equal BW)
- IX-ML 3.2.3, Driver 3.2.1, CUDA 10.2 (CoreX)
- CoreX SDK at `/usr/local/corex/`
- NCCL for collective communication (same process group as CUDA)
- KV cache tensor layout (ILU): `[n_blocks, n_heads, block_size, head_dim]` (axis 1 = heads)
- All verified constants centralized in `core/config/ilu_hw_constants.h`

### Usage

```bash
# Enable layerwise split KV cache
./xllm_server --model deepseek-v3 --enable_layerwise_split=true

# Disable (default — uniform sharding, no regression)
./xllm_server --model deepseek-v3 --enable_layerwise_split=false
```

## Test Plan

| ID | Level | Description | Criteria |
|----|-------|-------------|----------|
| TC-01 | L1 | Allocation correctness | Per-rank allocation matches layout; unassigned layers get zero KV; total equals sum |
| TC-02 | L1 | Memory estimation accuracy | Layerwise peak ≤ uniform; estimation error ≤ 5% |
| TC-03 | L1 | ILU PCIe topology mapping | All layers assigned; MoE layers on same-switch ranks; no oversubscription |
| TC-04 | L1 | Engine layout propagation | All 8 workers receive consistent layout; full layer coverage |
| TC-05 | L1 | Worker KV shard application | KV populated for assigned layers; zero for unassigned; ASAN clean |
| TC-06 | L2 | Fallback when disabled | Uniform allocation identical to pre-feature behaviour |
| TC-07 | L2 | Speculative engine | No crash; KV correctly partitioned per model |

## File Summary

```
core/
├── CMakeLists.txt
├── config/
│   ├── ilu_hw_constants.h
│   ├── parallel_config_layerwise.cpp
│   └── parallel_config_layerwise.h
├── distributed_runtime/
│   ├── layerwise_split_engine_ext.cpp
│   ├── layerwise_split_engine_ext.h
│   ├── layerwise_split_master.cpp
│   └── layerwise_split_master.h
├── framework/
│   ├── kv_cache/
│   │   ├── kv_cache_estimation_layerwise.cpp
│   │   ├── kv_cache_estimation_layerwise.h
│   │   ├── kv_cache_layerwise.cpp
│   │   ├── kv_cache_layerwise.h
│   │   └── layerwise_split_layout.h
│   └── parallel_state/
│       ├── mapping_ilu.cpp
│       └── mapping_ilu.h
└── runtime/
    ├── worker_layerwise_init.cpp
    └── worker_layerwise_init.h
docs/
└── LAYERWISE_SPLIT_KV_CACHE.md
tests/core/
└── test_layerwise_split_kv_cache.cpp
```
