"""
prefix_prefill.py Triton kernel tuning for BI-V100
===================================================

Analysis (derived from hardware specs + CCCL methodology):

BI-V100 hardware:
  SMEM per block: 48 KB
  Warp size: 32 (assumed)
  Max threads/block: 1024
  SM count: 16 (confirmed via ixsmi, not 50 from spec sheet)
  HBM bandwidth: 900 GB/s
  
Qwen3.6-35B-A3B attention:
  head_dim: 128 (primary), 256 (rare, falls back to PyTorch)
  num_heads: varies per layer (GQA)
  dtype: fp16/bf16

SMEM constraint for Triton Flash Attention:
  SMEM = BLOCK_N × head_dim × sizeof(fp16) × 2 (K + V tiles)
  BLOCK_N=64,  head_dim=128: 64×128×2×2 = 32KB ≤ 48KB ✓
  BLOCK_N=128, head_dim=128: 128×128×2×2 = 64KB > 48KB ✗ OVERFLOW
  → BLOCK_N must stay at 64 for head_dim=128 on BI-V100.

BLOCK_M analysis:
  BLOCK_M=64 means each thread block processes 64 query positions.
  At NUM_WARPS=8 (256 threads): each thread handles 64×128/256 = 32 elements.
  At NUM_WARPS=4 (128 threads): each thread handles 64×128/128 = 64 elements.
  
  More work per thread = better instruction-level parallelism (ILP).
  Fewer warps = more blocks can run concurrently per SM = better occupancy.
  
  BI-V100 has 16 SMs (confirmed, not 50 from spec sheet).
  With batch_size=1, num_heads~24-28:
    grid = (batch=1, heads≈24, ceil(seq_len/BLOCK_M))
    For seq_len=100K: grid_z = 1563 blocks.
    Total blocks = 1 × 24 × 1563 = 37,512 blocks.
    Blocks per SM = 37512/16 = 2344 — plenty of parallelism.
  NOTE: with max-num-seqs=256 (benchmark config), batch_size >> 1,
    grid is even larger. Parallelism is never the bottleneck.
    
  Reducing NUM_WARPS from 8→4:
    - Each SM can run more blocks concurrently (limited by registers/SMEM)
    - At 8 warps (256 threads), SMEM is the bottleneck (32KB K+V)
      → only 1 block per SM (48KB total / 32KB per block = 1.5 → 1)
    - At 4 warps (128 threads), register pressure might allow 2 blocks
    - Net effect: 2× occupancy improvement on memory-bound attention

  BUT: fewer warps = fewer threads to hide memory latency.
  On bandwidth-limited hardware (BI-V100 at 900 GB/s vs 8TB/s),
  latency hiding is less critical because the bottleneck is bandwidth,
  not latency. So NUM_WARPS=4 is likely better.

Recommended change:
  prefix_prefill.py line 713-714:
    BLOCK = 128 if current_platform.has_device_capability(80) else 64
    NUM_WARPS = 8
  →
    BLOCK = 64  # BI-V100: SMEM constrains BLOCK_N to 64 for head_dim=128
    NUM_WARPS = 4  # BI-V100: 4 warps → more blocks/SM → better occupancy

_PARTITION_SIZE inconsistency:
  paged_attn.py: _PARTITION_SIZE = 512
  attention.py:  _PARTITION_SIZE = 256
  These MUST match `PARTITION_SIZE in paged_attention_v2_launcher` (C++ side).
  The C++ launcher in the precompiled .so likely uses 512 (vllm default).
  attention.py's 256 may cause correctness issues if V2 is ever enabled.
  Since V1 is hardcoded (use_v1=True), this doesn't affect current behavior,
  but should be unified to 512 for safety.

computility-run.yaml optimizations:
  Current: max-num-batched-tokens: 8192
  Analysis: With max-num-seqs=1 and enable-chunked-prefill,
    the batch token budget controls prefill chunk size.
    Larger chunks = fewer kernel launches = less overhead.
    But larger chunks = more SMEM pressure per launch.
    At head_dim=128, BLOCK=64: each launch processes 64 query positions,
    so max-num-batched-tokens controls how many query positions
    are batched together, not SMEM usage.
    Increasing to 16384 or 32768 may reduce launch overhead.
    
  Current: gpu-memory-utilization: 0.9
  Analysis: BI-V100 has ~50GB HBM per GPU. At 0.9, ~45GB available.
    Qwen3.6-35B-A3B at fp16 needs ~70GB across 4 GPUs (~17.5GB/GPU).
    KV cache uses remaining ~27.5GB/GPU.
    At max-model-len=100K, KV cache per token per layer ≈ 2×128×2 = 512 bytes.
    Total KV cache for 100K tokens, 27 layers (estimated) ≈ 1.38GB.
    Plenty of room. Could increase to 0.95 for more KV cache capacity.
"""

# This file documents the analysis. The actual patches go into 
# qwen3_6_scripts/ as described below.

PATCHES = {
    "prefix_prefill.py": {
        "line": 713,
        "old": "        BLOCK = 128 if current_platform.has_device_capability(80) else 64\n        NUM_WARPS = 8",
        "new": "        # BI-V100: BLOCK=64 (SMEM constrains BLOCK_N≤64 for head_dim=128)\n        # NUM_WARPS=4 (fewer warps → more blocks/SM → better occupancy)\n        BLOCK = 64\n        NUM_WARPS = 4",
        "reasoning": "BLOCK_N=128 overflows 48KB SMEM. NUM_WARPS=4 doubles occupancy on bandwidth-limited BI-V100.",
    },
    "computility-run.yaml": {
        "changes": [
            ("max-num-batched-tokens", "8192", "16384", "Larger prefill chunks → fewer kernel launches"),
            ("gpu-memory-utilization", "0.9", "0.95", "BI-V100 has headroom for more KV cache"),
        ],
    },
}
