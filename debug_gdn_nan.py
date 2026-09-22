#!/usr/bin/env python3
"""Debug NaN in C++ torch_chunk_gated_delta_rule.

Tests with smaller dimensions to isolate the issue.
"""
import sys
import os
import importlib.util
import torch

def load_mod():
    so = "/tmp/gdn_test/corex_gdn_chunk_recurrent.so"
    if not os.path.exists(so):
        print("Run verify_gdn_cpp.py first to compile")
        return None
    spec = importlib.util.spec_from_file_location("corex_gdn_chunk_recurrent", so)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main():
    mod = load_mod()
    if mod is None:
        return 1

    # Test with tiny dimensions to isolate
    for T in [1, 2, 4, 8, 16, 32, 64, 128]:
        torch.manual_seed(42)
        B = 1
        Hk, Hv, D = 4, 8, 128
        chunk = min(64, T)

        q = torch.randn(B, T, Hk, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, T, Hk, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, T, Hv, D, device="cuda", dtype=torch.float16)
        g = torch.randn(B, T, Hv, device="cuda", dtype=torch.float16)
        beta = torch.randn(B, T, Hv, device="cuda", dtype=torch.float16)

        out, state = mod.torch_chunk_gated_delta_rule(
            q, k, v, g, beta, chunk, None, True, True)

        has_nan = out.isnan().any().item()
        nan_count = out.isnan().sum().item() if has_nan else 0
        print(f"T={T:4d} chunk={chunk:3d}: NaN={has_nan} (count={nan_count}/{out.numel()})")

        if has_nan and T <= 16:
            # Print where NaN is
            nan_mask = out.isnan()
            print(f"  NaN positions: {nan_mask.nonzero()[:5].tolist()}")

    # Test: does chunk_size=T (no actual chunking) work?
    print("\n--- Single chunk (chunk_size == T) ---")
    for T in [32, 64]:
        torch.manual_seed(42)
        q = torch.randn(1, T, 4, 128, device="cuda", dtype=torch.float16)
        k = torch.randn(1, T, 4, 128, device="cuda", dtype=torch.float16)
        v = torch.randn(1, T, 8, 128, device="cuda", dtype=torch.float16)
        g = torch.randn(1, T, 8, device="cuda", dtype=torch.float16)
        beta = torch.randn(1, T, 8, device="cuda", dtype=torch.float16)

        out, state = mod.torch_chunk_gated_delta_rule(
            q, k, v, g, beta, T, None, True, True)
        print(f"T={T} chunk={T}: NaN={out.isnan().any().item()}")

    # Test: float32 input instead of float16
    print("\n--- Float32 input ---")
    for T in [64, 128]:
        torch.manual_seed(42)
        q = torch.randn(1, T, 4, 128, device="cuda", dtype=torch.float32)
        k = torch.randn(1, T, 4, 128, device="cuda", dtype=torch.float32)
        v = torch.randn(1, T, 8, 128, device="cuda", dtype=torch.float32)
        g = torch.randn(1, T, 8, device="cuda", dtype=torch.float32)
        beta = torch.randn(1, T, 8, device="cuda", dtype=torch.float32)

        out, state = mod.torch_chunk_gated_delta_rule(
            q, k, v, g, beta, 64, None, True, True)
        print(f"T={T} chunk=64 f32: NaN={out.isnan().any().item()}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
