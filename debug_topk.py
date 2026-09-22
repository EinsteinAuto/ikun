#!/usr/bin/env python3
"""Debug topk_softmax CUDA kernel mismatch."""
import torch
import os
from torch.utils.cpp_extension import load

ext = load(name="moe_topk_softmax_v3",
           sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "ex_engine/csrc/moe_topk_softmax_v3.cu")],
           extra_cuda_cflags=["-O3"], verbose=False)

torch.manual_seed(123)
gating = torch.randn(8, 64, device='cuda', dtype=torch.float32)

# CUDA kernel
results = ext.moe_topk_softmax(gating, 8, False)
tw_cuda, ti_cuda = results[0], results[1]

# PyTorch reference
probs = torch.softmax(gating, dim=-1)
tw_ref, ti_ref = torch.topk(probs, 8, dim=-1)

print("=== Per-row comparison ===")
for r in range(8):
    ids_match = set(ti_cuda[r].tolist()) == set(ti_ref[r].tolist())
    w_diff = (tw_cuda[r].sort()[0] - tw_ref[r].sort()[0]).abs().max().item()
    print(f"Row {r}: CUDA ids={ti_cuda[r].tolist()[:4]}...  "
          f"Ref ids={ti_ref[r].tolist()[:4]}...  "
          f"ids_match={ids_match}  w_diff={w_diff:.6e}  "
          f"cuda_sum={tw_cuda[r].sum():.4f}  ref_sum={tw_ref[r].sum():.4f}")

# Check if consecutive rows are identical
print("\n=== Row duplication check ===")
for r in range(0, 8, 2):
    same = (ti_cuda[r] == ti_cuda[r+1]).all().item()
    print(f"Row {r} == Row {r+1}: {same}")

# Minimal 2-row test
print("\n=== Minimal 2-row test ===")
g2 = torch.tensor([[1.0, 2.0, 3.0] + [0.0]*61,
                    [3.0, 2.0, 1.0] + [0.0]*61], device='cuda', dtype=torch.float32)
r2 = ext.moe_topk_softmax(g2, 3, False)
p2 = torch.softmax(g2, dim=-1)
t2w, t2i = torch.topk(p2, 3, dim=-1)
print(f"CUDA row0 ids: {r2[1][0].tolist()[:3]}  weights: {r2[0][0].tolist()[:3]}")
print(f"CUDA row1 ids: {r2[1][1].tolist()[:3]}  weights: {r2[0][1].tolist()[:3]}")
print(f"Ref  row0 ids: {t2i[0].tolist()[:3]}  weights: {t2w[0].tolist()[:3]}")
print(f"Ref  row1 ids: {t2i[1].tolist()[:3]}  weights: {t2w[1].tolist()[:3]}")
