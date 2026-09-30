"""
Verify corex_moe_topk_softmax is a drop-in replacement for PyTorch
softmax+topk in the MoE routing path.

Tests:
  1. Output shape matches PyTorch path
  2. Output dtypes match (ids: int32 from corex, needs .to(int64))
  3. renormalize=True produces weights summing to 1.0 (like softmax)
  4. Numerical agreement: same top-K expert selection as PyTorch
  5. Batch-size sweep: bs=1,2,4,8 all produce correct shapes
  6. Edge cases: all-zero logits, extreme values, negative logits
"""
import sys
import os

def main():
    try:
        import torch
    except ImportError:
        print("SKIP: torch not available")
        return 0

    try:
        from vllm import corex_moe_topk_softmax as _corex
    except ImportError:
        print("SKIP: corex_moe_topk_softmax not available")
        return 0

    E, K = 256, 8
    failures = 0
    total = 0

    def check(name, cond, detail=""):
        nonlocal failures, total
        total += 1
        if cond:
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name}: {detail}")
            failures += 1

    # --- Test 1: Shape consistency ---
    print("\n=== 1. Shape consistency ===")
    for bs in [1, 2, 4, 8]:
        logits = torch.randn(bs, E, device="cuda", dtype=torch.float16)
        ws_c, ids_c = _corex.moe_topk_softmax(logits.float(), K, True)
        check(f"bs={bs} weights shape", ws_c.shape == (bs, K),
              f"got {ws_c.shape}")
        check(f"bs={bs} ids shape", ids_c.shape == (bs, K),
              f"got {ids_c.shape}")

    # --- Test 2: dtype ---
    print("\n=== 2. dtype checks ===")
    logits = torch.randn(1, E, device="cuda", dtype=torch.float16)
    ws_c, ids_c = _corex.moe_topk_softmax(logits.float(), K, True)
    check("weights dtype is float32", ws_c.dtype == torch.float32,
          f"got {ws_c.dtype}")
    check("ids dtype is int32 or int64",
          ids_c.dtype in (torch.int32, torch.int64),
          f"got {ids_c.dtype}")
    # Verify .to(int64) works (this is what qwen3_5.py does)
    ids_i64 = ids_c[0].to(torch.int64)
    check("ids .to(int64) succeeds", ids_i64.dtype == torch.int64)

    # --- Test 3: renormalize=True → weights sum to 1.0 ---
    print("\n=== 3. Weight renormalization ===")
    for trial in range(5):
        logits = torch.randn(1, E, device="cuda", dtype=torch.float16)
        ws_c, _ = _corex.moe_topk_softmax(logits.float(), K, True)
        wsum = ws_c[0].sum().item()
        check(f"trial {trial} weights sum ≈ 1.0",
              abs(wsum - 1.0) < 0.01, f"sum={wsum:.6f}")

    # --- Test 4: Numerical agreement with PyTorch ---
    print("\n=== 4. Numerical agreement with PyTorch topk ===")
    torch.manual_seed(42)
    agree_count = 0
    n_trials = 20
    for trial in range(n_trials):
        logits = torch.randn(1, E, device="cuda", dtype=torch.float16)

        # PyTorch path
        probs_pt = torch.softmax(logits.float(), dim=-1)
        ws_pt, ids_pt = torch.topk(probs_pt, K, dim=-1)
        # Re-normalize
        ws_pt = ws_pt / ws_pt.sum(dim=-1, keepdim=True)
        ids_pt_sorted = ids_pt[0].sort()[0]

        # Corex path
        ws_c, ids_c = _corex.moe_topk_softmax(logits.float(), K, True)
        ids_c_sorted = ids_c[0].to(torch.int64).sort()[0]

        same_experts = torch.equal(ids_pt_sorted, ids_c_sorted)
        if same_experts:
            agree_count += 1

        # Check weight closeness (only when same experts selected)
        if same_experts:
            # Sort both by expert id for comparison
            _, pt_order = ids_pt[0].sort()
            _, c_order = ids_c[0].to(torch.int64).sort()
            ws_pt_s = ws_pt[0][pt_order]
            ws_c_s = ws_c[0][c_order]
            max_diff = (ws_pt_s - ws_c_s).abs().max().item()
            check(f"trial {trial} weight max_diff",
                  max_diff < 0.005, f"max_diff={max_diff:.6f}")

    check(f"expert selection agreement",
          agree_count >= n_trials - 2,
          f"{agree_count}/{n_trials} trials agree")

    # --- Test 5: Edge cases ---
    print("\n=== 5. Edge cases ===")
    # All zeros
    logits_zero = torch.zeros(1, E, device="cuda", dtype=torch.float16)
    ws_z, ids_z = _corex.moe_topk_softmax(logits_zero.float(), K, True)
    check("all-zero logits: no NaN in weights",
          not torch.isnan(ws_z).any().item())
    check("all-zero logits: weights sum ≈ 1.0",
          abs(ws_z[0].sum().item() - 1.0) < 0.01,
          f"sum={ws_z[0].sum().item()}")

    # Large positive
    logits_big = torch.full((1, E), 100.0, device="cuda", dtype=torch.float16)
    ws_b, ids_b = _corex.moe_topk_softmax(logits_big.float(), K, True)
    check("large logits: no NaN/Inf",
          not (torch.isnan(ws_b).any() or torch.isinf(ws_b).any()).item())

    # Large negative
    logits_neg = torch.full((1, E), -100.0, device="cuda", dtype=torch.float16)
    ws_n, ids_n = _corex.moe_topk_softmax(logits_neg.float(), K, True)
    check("large negative logits: no NaN/Inf",
          not (torch.isnan(ws_n).any() or torch.isinf(ws_n).any()).item())

    # --- Summary ---
    print(f"\n{'='*50}")
    print(f"  {total - failures}/{total} passed, {failures} failed")
    if failures == 0:
        print("  ✅ corex_moe_topk_softmax is a safe drop-in replacement")
    else:
        print("  ⚠️  FIX BEFORE ENABLING IN PRODUCTION")
    return failures


if __name__ == "__main__":
    sys.exit(main())
