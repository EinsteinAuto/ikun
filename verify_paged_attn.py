#!/usr/bin/env python3
"""Test ixformer paged attention v1/v2 with head_dim=256 on BI-V100.

Now that we know the correct signature (needs head_mapping for GQA),
test if paged attention works for Qwen3.5 decode path.

Qwen3.5 TP=4: num_heads=4, num_kv_heads=1, head_dim=256, block_size=16
"""
import sys
import time
import torch
import ixformer


def main():
    print("=" * 60)
    print("BI-V100 paged attention v1/v2 test (head_dim=256)")
    print("=" * 60)

    num_heads = 4
    num_kv_heads = 1
    head_dim = 256
    block_size = 16

    # head_mapping: maps each query head to its KV head
    # For GQA with 4 q heads and 1 kv head: [0, 0, 0, 0]
    head_mapping = torch.zeros(num_heads, dtype=torch.int32, device="cuda")

    scale = head_dim ** -0.5

    # --- Test V1: Basic decode ---
    print("\n--- V1: vllm_single_query_cached_kv_attention ---")
    for num_blocks in [4, 16, 64, 256]:
        context_len = num_blocks * block_size
        num_seqs = 1

        query = torch.randn(num_seqs, num_heads, head_dim,
                             device="cuda", dtype=torch.float16)
        # KV cache: (num_blocks_total, num_kv_heads, head_dim, block_size)
        # This is the standard vllm KV cache layout
        key_cache = torch.randn(num_blocks, num_kv_heads, head_dim, block_size,
                                 device="cuda", dtype=torch.float16)
        value_cache = torch.randn(num_blocks, num_kv_heads, head_dim, block_size,
                                   device="cuda", dtype=torch.float16)
        block_tables = torch.arange(num_blocks, device="cuda",
                                     dtype=torch.int32).unsqueeze(0)
        context_lens = torch.tensor([context_len], device="cuda",
                                     dtype=torch.int32)
        output = torch.empty(num_seqs, num_heads, head_dim,
                              device="cuda", dtype=torch.float16)

        try:
            ixformer.vllm_single_query_cached_kv_attention(
                output, query, key_cache, value_cache,
                head_mapping, scale, block_tables, context_lens,
                block_size, context_len)
            has_nan = output.isnan().any().item()
            print(f"  ctx={context_len:5d}: OK nan={has_nan}")
        except Exception as e:
            print(f"  ctx={context_len:5d}: EXCEPTION: {e}")

    # --- Test V2: Partitioned decode (for long contexts) ---
    print("\n--- V2: vllm_single_query_cached_kv_attention_v2 ---")
    for num_blocks in [64, 256, 512]:
        context_len = num_blocks * block_size
        num_seqs = 1
        partition_size = 512  # standard vllm partition size

        query = torch.randn(num_seqs, num_heads, head_dim,
                             device="cuda", dtype=torch.float16)
        key_cache = torch.randn(num_blocks, num_kv_heads, head_dim, block_size,
                                 device="cuda", dtype=torch.float16)
        value_cache = torch.randn(num_blocks, num_kv_heads, head_dim, block_size,
                                   device="cuda", dtype=torch.float16)
        block_tables = torch.arange(num_blocks, device="cuda",
                                     dtype=torch.int32).unsqueeze(0)
        context_lens_t = torch.tensor([context_len], device="cuda",
                                       dtype=torch.int32)
        output = torch.empty(num_seqs, num_heads, head_dim,
                              device="cuda", dtype=torch.float16)

        max_num_partitions = (context_len + partition_size - 1) // partition_size
        exp_sums = torch.empty(num_seqs, num_heads, max_num_partitions,
                                device="cuda", dtype=torch.float32)
        max_logits = torch.empty(num_seqs, num_heads, max_num_partitions,
                                  device="cuda", dtype=torch.float32)
        temp_output = torch.empty(num_seqs, num_heads, max_num_partitions, head_dim,
                                   device="cuda", dtype=torch.float32)

        try:
            ixformer.vllm_single_query_cached_kv_attention_v2(
                output, partition_size, exp_sums, max_logits, temp_output,
                query, key_cache, value_cache,
                head_mapping, scale, block_tables, context_lens_t,
                block_size, context_len)
            has_nan = output.isnan().any().item()
            print(f"  ctx={context_len:5d}: OK nan={has_nan}")
        except Exception as e:
            print(f"  ctx={context_len:5d}: EXCEPTION: {e}")

    # --- Performance: V1 vs Python decode ---
    print("\n--- Performance: V1 paged decode vs Python ---")
    num_blocks = 64
    context_len = num_blocks * block_size  # 1024
    query = torch.randn(1, num_heads, head_dim, device="cuda", dtype=torch.float16)
    key_cache = torch.randn(num_blocks, num_kv_heads, head_dim, block_size,
                             device="cuda", dtype=torch.float16)
    value_cache = torch.randn(num_blocks, num_kv_heads, head_dim, block_size,
                               device="cuda", dtype=torch.float16)
    block_tables = torch.arange(num_blocks, device="cuda", dtype=torch.int32).unsqueeze(0)
    context_lens_t = torch.tensor([context_len], device="cuda", dtype=torch.int32)
    output = torch.empty(1, num_heads, head_dim, device="cuda", dtype=torch.float16)

    # Warmup
    for _ in range(10):
        ixformer.vllm_single_query_cached_kv_attention(
            output, query, key_cache, value_cache,
            head_mapping, scale, block_tables, context_lens_t,
            block_size, context_len)
    torch.cuda.synchronize()

    N = 100
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        ixformer.vllm_single_query_cached_kv_attention(
            output, query, key_cache, value_cache,
            head_mapping, scale, block_tables, context_lens_t,
            block_size, context_len)
    torch.cuda.synchronize()
    ix_ms = (time.perf_counter() - t0) / N * 1000

    # Python reference: gather KV from cache + matmul
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        # Gather all KV blocks
        k_all = key_cache[block_tables[0]].permute(0, 3, 1, 2).reshape(
            1, context_len, num_kv_heads, head_dim)
        v_all = value_cache[block_tables[0]].permute(0, 3, 1, 2).reshape(
            1, context_len, num_kv_heads, head_dim)
        # Expand for GQA
        k_all = k_all.expand(-1, -1, num_heads, -1)
        v_all = v_all.expand(-1, -1, num_heads, -1)
        q_4d = query.unsqueeze(1)  # (1, 1, H, D)
        attn = torch.matmul(
            q_4d.transpose(1, 2).float(),
            k_all.transpose(1, 2).transpose(-2, -1).float()) * scale
        attn = torch.softmax(attn, dim=-1)
        _ = torch.matmul(attn, v_all.transpose(1, 2).float()).to(torch.float16)
    torch.cuda.synchronize()
    pt_ms = (time.perf_counter() - t0) / N * 1000

    print(f"  ixformer paged: {ix_ms:.3f} ms")
    print(f"  Python gather+matmul: {pt_ms:.3f} ms")
    print(f"  Speedup: {pt_ms/ix_ms:.1f}x")

    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
