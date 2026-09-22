#!/usr/bin/env python3
"""Test ixformer native attention with head_dim=256 on BI-V100.

Qwen3.5 uses head_dim=256 for full attention layers.
We bypassed ixformer because of "head_dim > 128 limit".
Test if that's actually true on this hardware.

Tests:
1. ixformer.scaled_dot_product_attention with head_dim=256
2. ixformer.flash_attn_func with head_dim=256
3. ixformer.vllm_single_query_cached_kv_attention with head_dim=256
4. Compare outputs vs PyTorch reference
"""
import sys
import torch
import torch.nn.functional as F


def pytorch_sdpa_ref(q, k, v, is_causal=True):
    """Reference: standard scaled dot-product attention."""
    scale = q.shape[-1] ** -0.5
    attn = torch.matmul(q * scale, k.transpose(-2, -1))
    if is_causal:
        L = q.shape[-2]
        S = k.shape[-2]
        mask = torch.triu(torch.ones(L, S, device=q.device, dtype=torch.bool), diagonal=S-L+1)
        attn = attn.masked_fill(mask, float('-inf'))
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, v)


def main():
    print("=" * 60)
    print("BI-V100 ixformer attention head_dim=256 test")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("FATAL: No CUDA")
        return 1

    try:
        import ixformer
        print(f"ixformer available: True")
    except ImportError:
        print("ixformer not available")
        return 1

    # Check what's available
    has_sdpa = hasattr(ixformer, 'scaled_dot_product_attention')
    has_flash = hasattr(ixformer, 'flash_attn_func')
    has_varlen = hasattr(ixformer, 'flash_attn_varlen_func')
    has_paged = hasattr(ixformer, 'vllm_single_query_cached_kv_attention')
    print(f"scaled_dot_product_attention: {has_sdpa}")
    print(f"flash_attn_func: {has_flash}")
    print(f"flash_attn_varlen_func: {has_varlen}")
    print(f"vllm_single_query_cached_kv_attention: {has_paged}")

    # Qwen3.5 full attention dims (TP=4):
    # num_heads=4, num_kv_heads=1, head_dim=256
    batch = 1
    num_heads = 4
    num_kv_heads = 1
    head_dim = 256
    torch.manual_seed(42)

    # --- Test 1: scaled_dot_product_attention ---
    if has_sdpa:
        for seq_len in [1, 4, 16, 64, 128]:
            q = torch.randn(batch, num_heads, seq_len, head_dim,
                            device="cuda", dtype=torch.float16)
            # GQA: kv has fewer heads
            k = torch.randn(batch, num_kv_heads, seq_len, head_dim,
                            device="cuda", dtype=torch.float16)
            v = torch.randn(batch, num_kv_heads, seq_len, head_dim,
                            device="cuda", dtype=torch.float16)
            # Expand kv to match q heads for reference
            k_exp = k.expand(-1, num_heads, -1, -1)
            v_exp = v.expand(-1, num_heads, -1, -1)
            ref = pytorch_sdpa_ref(q, k_exp, v_exp, is_causal=(seq_len > 1))

            try:
                # Try without causal first
                out = ixformer.scaled_dot_product_attention(
                    q, k_exp, v_exp, is_causal=(seq_len > 1))
                diff = (out.float() - ref.float()).abs().max().item()
                has_nan = out.isnan().any().item()
                print(f"\n  SDPA seq={seq_len}: diff={diff:.6f} nan={has_nan} "
                      f"{'PASS' if diff < 0.01 and not has_nan else 'FAIL'}")
            except Exception as e:
                print(f"\n  SDPA seq={seq_len}: EXCEPTION: {e}")

    # --- Test 2: flash_attn_func ---
    if has_flash:
        for seq_len in [1, 4, 16, 64]:
            # flash_attn expects (batch, seqlen, nheads, headdim)
            q = torch.randn(batch, seq_len, num_heads, head_dim,
                            device="cuda", dtype=torch.float16)
            k = torch.randn(batch, seq_len, num_kv_heads, head_dim,
                            device="cuda", dtype=torch.float16)
            v = torch.randn(batch, seq_len, num_kv_heads, head_dim,
                            device="cuda", dtype=torch.float16)
            try:
                out = ixformer.flash_attn_func(q, k, v, causal=True)
                has_nan = out.isnan().any().item()
                print(f"  flash_attn seq={seq_len}: shape={out.shape} nan={has_nan}")
            except Exception as e:
                print(f"  flash_attn seq={seq_len}: EXCEPTION: {e}")

    # --- Test 3: head_dim=128 (known to work) vs head_dim=256 ---
    if has_sdpa:
        print("\n--- Comparison: head_dim=128 vs head_dim=256 ---")
        for hd in [128, 256]:
            q = torch.randn(1, 4, 16, hd, device="cuda", dtype=torch.float16)
            k = torch.randn(1, 4, 16, hd, device="cuda", dtype=torch.float16)
            v = torch.randn(1, 4, 16, hd, device="cuda", dtype=torch.float16)
            try:
                out = ixformer.scaled_dot_product_attention(q, k, v, is_causal=True)
                print(f"  head_dim={hd}: OK shape={out.shape} nan={out.isnan().any().item()}")
            except Exception as e:
                print(f"  head_dim={hd}: EXCEPTION: {e}")

    # --- Test 4: Performance if it works ---
    if has_sdpa:
        print("\n--- Performance: SDPA head_dim=256 seq=64 ---")
        q = torch.randn(1, 4, 64, 256, device="cuda", dtype=torch.float16)
        k = torch.randn(1, 4, 64, 256, device="cuda", dtype=torch.float16)
        v = torch.randn(1, 4, 64, 256, device="cuda", dtype=torch.float16)

        import time
        # Warmup
        try:
            for _ in range(5):
                ixformer.scaled_dot_product_attention(q, k, v, is_causal=True)
            torch.cuda.synchronize()

            N = 50
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(N):
                ixformer.scaled_dot_product_attention(q, k, v, is_causal=True)
            torch.cuda.synchronize()
            ix_ms = (time.perf_counter() - t0) / N * 1000

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(N):
                pytorch_sdpa_ref(q, k, v, is_causal=True)
            torch.cuda.synchronize()
            pt_ms = (time.perf_counter() - t0) / N * 1000

            print(f"  ixformer: {ix_ms:.3f} ms")
            print(f"  PyTorch:  {pt_ms:.3f} ms")
            print(f"  Speedup:  {pt_ms/ix_ms:.2f}x")
        except Exception as e:
            print(f"  Performance test failed: {e}")

    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
