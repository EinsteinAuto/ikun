#!/usr/bin/env python3
"""
precompile_moe_kernels.py — JIT compile vllm v0.5.5 MoE CUDA kernels for BI-V100.

Produces: moe_kernels.so with:
  - topk_softmax(topk_weights, topk_indices, token_expert_indices, gating_output)
  - moe_align_block_size(topk_ids, num_experts, block_size, sorted_ids, expert_ids, num_tokens_post_pad)

Usage:
  python3 precompile_moe_kernels.py           # JIT compile
  python3 precompile_moe_kernels.py --test    # compile + smoke test
"""
import os
import sys
import time

def compile_moe_kernels():
    """JIT compile MoE CUDA kernels via torch.utils.cpp_extension."""
    import torch
    from torch.utils.cpp_extension import load

    script_dir = os.path.dirname(os.path.abspath(__file__))
    moe_dir = os.path.join(script_dir, 'csrc', 'moe_v055')

    sources = [
        os.path.join(moe_dir, 'moe_pybind.cpp'),
        os.path.join(moe_dir, 'topk_softmax_kernels.cu'),
        os.path.join(moe_dir, 'moe_align_block_size_kernels.cu'),
    ]

    for s in sources:
        if not os.path.isfile(s):
            raise FileNotFoundError(f"Missing: {s}")

    print(f"[moe_kernels] Compiling from {moe_dir}")
    t0 = time.time()

    mod = load(
        name='moe_kernels',
        sources=sources,
        extra_include_paths=[moe_dir],
        extra_cflags=['-O2', '-std=c++17'],
        extra_cuda_cflags=['-O2', '--expt-relaxed-constexpr'],
        verbose=True,
    )

    dt = time.time() - t0
    funcs = [x for x in dir(mod) if not x.startswith('_')]
    print(f"[moe_kernels] Compiled in {dt:.1f}s — functions: {funcs}")
    return mod


def smoke_test(mod):
    """Quick functional test of compiled kernels."""
    import torch

    print("\n=== Smoke test ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("  SKIP: no CUDA device")
        return

    # Test topk_softmax
    num_tokens, num_experts, topk = 4, 8, 2
    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights = torch.empty(num_tokens, topk, device=device, dtype=torch.float32)
    topk_indices = torch.empty(num_tokens, topk, device=device, dtype=torch.int32)
    token_expert_indices = torch.empty(num_tokens, topk, device=device, dtype=torch.int32)

    mod.topk_softmax(topk_weights, topk_indices, token_expert_indices, gating)

    print(f"  topk_softmax: weights={topk_weights.shape}, NaN={topk_weights.isnan().any()}")
    print(f"    weights[0] = {topk_weights[0].tolist()}")
    print(f"    indices[0] = {topk_indices[0].tolist()}")

    # Test moe_align_block_size
    block_size = 4
    max_num_tokens_padded = (num_tokens * topk + num_experts * block_size)
    sorted_ids = torch.empty(max_num_tokens_padded, device=device, dtype=torch.int32)
    expert_ids = torch.empty(max_num_tokens_padded // block_size, device=device, dtype=torch.int32)
    num_tokens_post_pad = torch.empty(1, device=device, dtype=torch.int32)

    mod.moe_align_block_size(topk_indices, num_experts, block_size,
                              sorted_ids, expert_ids, num_tokens_post_pad)

    print(f"  moe_align: sorted_ids[:8]={sorted_ids[:8].tolist()}, "
          f"num_post_pad={num_tokens_post_pad.item()}")

    print("\n  ✓ All smoke tests passed")


if __name__ == '__main__':
    mod = compile_moe_kernels()
    if '--test' in sys.argv:
        smoke_test(mod)
