#!/usr/bin/env python3
"""Verify ixformer.flash_attn_func for Qwen3.5 prefill attention.

flash_attn_func works with head_dim=256 on BI-V100!
Now test correctness vs PyTorch ref and benchmark on real prefill lengths.

Also test flash_attn_varlen_func (used by vllm for variable-length batching)
and vllm_single_query_cached_kv_attention (used for decode with KV cache).
"""
import sys
import time
import torch
import torch.nn.functional as F


def pytorch_attention_ref(q, k, v, causal=True):
    """(batch, seqlen, nheads, headdim) format."""
    q = q.transpose(1, 2)  # (B, H, S, D)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    scale = q.shape[-1] ** -0.5
    attn = torch.matmul(q.float() * scale, k.float().transpose(-2, -1))
    if causal and q.shape[-2] > 1:
        L, S = q.shape[-2], k.shape[-2]
        mask = torch.triu(torch.ones(L, S, device=q.device, dtype=torch.bool),
                          diagonal=S - L + 1)
        attn = attn.masked_fill(mask, float('-inf'))
    attn = torch.softmax(attn, dim=-1)
    out = torch.matmul(attn, v.float())
    return out.transpose(1, 2).to(q.dtype)  # back to (B, S, H, D)


def main():
    print("=" * 60)
    print("BI-V100 flash_attn_func verification for Qwen3.5")
    print("=" * 60)

    import ixformer

    # Qwen3.5 full attention dims (TP=4):
    # num_heads=4, num_kv_heads=1, head_dim=256
    num_heads = 4
    num_kv_heads = 1
    head_dim = 256

    # --- Test 1: Correctness with GQA (different q/kv heads) ---
    print("\n--- Test 1: Correctness (GQA: q_heads=4, kv_heads=1) ---")
    for seq_len in [1, 4, 16, 64, 128, 256]:
        torch.manual_seed(42)
        q = torch.randn(1, seq_len, num_heads, head_dim,
                         device="cuda", dtype=torch.float16)
        k = torch.randn(1, seq_len, num_kv_heads, head_dim,
                         device="cuda", dtype=torch.float16)
        v = torch.randn(1, seq_len, num_kv_heads, head_dim,
                         device="cuda", dtype=torch.float16)

        try:
            out = ixformer.flash_attn_func(q, k, v, causal=(seq_len > 1))
            # For ref, expand kv heads to match q
            k_exp = k.expand(-1, -1, num_heads, -1)
            v_exp = v.expand(-1, -1, num_heads, -1)
            ref = pytorch_attention_ref(q, k_exp, v_exp, causal=(seq_len > 1))
            diff = (out.float() - ref.float()).abs().max().item()
            has_nan = out.isnan().any().item()
            status = "PASS" if diff < 0.05 and not has_nan else "FAIL"
            print(f"  seq={seq_len:4d}: diff={diff:.6f} nan={has_nan} {status}")
        except Exception as e:
            print(f"  seq={seq_len:4d}: EXCEPTION: {e}")

    # --- Test 2: Longer sequences (actual prefill lengths) ---
    print("\n--- Test 2: Long sequence prefill ---")
    for seq_len in [512, 1024, 2048, 4096]:
        q = torch.randn(1, seq_len, num_heads, head_dim,
                         device="cuda", dtype=torch.float16)
        k = torch.randn(1, seq_len, num_kv_heads, head_dim,
                         device="cuda", dtype=torch.float16)
        v = torch.randn(1, seq_len, num_kv_heads, head_dim,
                         device="cuda", dtype=torch.float16)
        try:
            out = ixformer.flash_attn_func(q, k, v, causal=True)
            has_nan = out.isnan().any().item()
            print(f"  seq={seq_len:5d}: shape={out.shape} nan={has_nan}")
        except Exception as e:
            print(f"  seq={seq_len:5d}: EXCEPTION: {e}")

    # --- Test 3: flash_attn_varlen_func (variable length, used by vllm) ---
    print("\n--- Test 3: flash_attn_varlen_func ---")
    if hasattr(ixformer, 'flash_attn_varlen_func'):
        for seq_len in [64, 256, 1024]:
            q = torch.randn(seq_len, num_heads, head_dim,
                             device="cuda", dtype=torch.float16)
            k = torch.randn(seq_len, num_kv_heads, head_dim,
                             device="cuda", dtype=torch.float16)
            v = torch.randn(seq_len, num_kv_heads, head_dim,
                             device="cuda", dtype=torch.float16)
            cu_seqlens = torch.tensor([0, seq_len], device="cuda", dtype=torch.int32)
            try:
                out = ixformer.flash_attn_varlen_func(
                    q, k, v, cu_seqlens, cu_seqlens,
                    seq_len, seq_len, causal=True)
                has_nan = out.isnan().any().item()
                print(f"  varlen seq={seq_len:5d}: shape={out.shape} nan={has_nan}")
            except Exception as e:
                print(f"  varlen seq={seq_len:5d}: EXCEPTION: {e}")

    # --- Test 4: Performance ---
    print("\n--- Test 4: Performance flash_attn_func vs PyTorch ---")
    for seq_len in [64, 256, 1024]:
        q = torch.randn(1, seq_len, num_heads, head_dim,
                         device="cuda", dtype=torch.float16)
        k = torch.randn(1, seq_len, num_kv_heads, head_dim,
                         device="cuda", dtype=torch.float16)
        v = torch.randn(1, seq_len, num_kv_heads, head_dim,
                         device="cuda", dtype=torch.float16)
        k_exp = k.expand(-1, -1, num_heads, -1).contiguous()
        v_exp = v.expand(-1, -1, num_heads, -1).contiguous()

        # Warmup
        for _ in range(5):
            ixformer.flash_attn_func(q, k, v, causal=True)
            pytorch_attention_ref(q, k_exp, v_exp, causal=True)
        torch.cuda.synchronize()

        N = 20
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N):
            ixformer.flash_attn_func(q, k, v, causal=True)
        torch.cuda.synchronize()
        ix_ms = (time.perf_counter() - t0) / N * 1000

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N):
            pytorch_attention_ref(q, k_exp, v_exp, causal=True)
        torch.cuda.synchronize()
        pt_ms = (time.perf_counter() - t0) / N * 1000

        print(f"  seq={seq_len:5d}: flash={ix_ms:.2f}ms pytorch={pt_ms:.2f}ms "
              f"speedup={pt_ms/ix_ms:.1f}x")

    # --- Test 5: vllm paged attention (decode) ---
    print("\n--- Test 5: vllm_single_query_cached_kv_attention ---")
    if hasattr(ixformer, 'vllm_single_query_cached_kv_attention'):
        # Simulate decode with KV cache
        # This is the function vllm uses for decode path
        num_seqs = 1
        num_kv_heads_total = num_kv_heads
        block_size = 16
        num_blocks = 64  # 64*16 = 1024 context tokens
        max_context_len = num_blocks * block_size

        q = torch.randn(num_seqs, num_heads, head_dim,
                         device="cuda", dtype=torch.float16)
        k_cache = torch.randn(num_blocks * num_seqs, num_kv_heads_total,
                               head_dim, block_size,
                               device="cuda", dtype=torch.float16)
        v_cache = torch.randn(num_blocks * num_seqs, num_kv_heads_total,
                               head_dim, block_size,
                               device="cuda", dtype=torch.float16)
        block_tables = torch.arange(num_blocks, device="cuda",
                                     dtype=torch.int32).unsqueeze(0)
        context_lens = torch.tensor([max_context_len], device="cuda",
                                     dtype=torch.int32)
        scale = head_dim ** -0.5
        out = torch.empty(num_seqs, num_heads, head_dim,
                          device="cuda", dtype=torch.float16)

        try:
            ixformer.vllm_single_query_cached_kv_attention(
                out, q, k_cache, v_cache, scale,
                block_tables, context_lens, block_size, max_context_len)
            has_nan = out.isnan().any().item()
            print(f"  paged attn: shape={out.shape} nan={has_nan}")
        except Exception as e:
            print(f"  paged attn: EXCEPTION: {e}")

    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
