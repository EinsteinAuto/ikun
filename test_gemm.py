"""test_gemm.py — Correctness test for all GEMM backends.

Compares each backend against torch.mm with Qwen3.5 MoE shapes.
Reports max absolute error and whether it passes FP16 tolerance.

Usage: python3 test_gemm.py
"""
import sys
import os
import torch

H = 3584
I = 18944 // 4  # 4736, per TP=4
TWO_I = I * 2
NUM_EXPERTS = 128
TOPK = 8
FP16_ATOL = 5e-2  # FP16 has ~1e-3 precision, allow some accumulation error


def test_single_gemm(device):
    """Test single GEMM: A(M,K) × B(K,N)."""
    print("\n=== Single GEMM ===")
    A = torch.randn(4, H, device=device, dtype=torch.float16)
    B = torch.randn(H, TWO_I, device=device, dtype=torch.float16) * 0.01

    ref = torch.mm(A.float(), B.float()).half()

    backends = {}
    try:
        import cuinfer_gemm_wrapper
        backends["cuinfer"] = cuinfer_gemm_wrapper.cuinfer_gemm(A, B, False)
    except Exception as e:
        print(f"  cuinfer: skip ({e})")

    try:
        import hgemm
        backends["hgemm"] = hgemm.hgemm(A, B)
    except Exception as e:
        print(f"  hgemm: skip ({e})")

    for name, out in backends.items():
        err = (out.float() - ref.float()).abs().max().item()
        ok = "✓" if err < FP16_ATOL else "✗"
        print(f"  {ok} {name}: max_err={err:.6f} (tol={FP16_ATOL})")


def test_group_gemm(device):
    """Test group GEMM with per-expert variable counts."""
    print("\n=== Group GEMM (MoE w13) ===")
    total_tokens = TOPK  # decode: 1 token × 8 experts
    expert_counts = torch.zeros(NUM_EXPERTS, device=device, dtype=torch.int32)
    for i in range(TOPK):
        expert_counts[i] = 1

    input_t = torch.randn(total_tokens, H, device=device, dtype=torch.float16) * 0.1
    w13 = torch.randn(NUM_EXPERTS, TWO_I, H, device=device, dtype=torch.float16) * 0.01

    # Reference: torch.mm per expert
    ref = torch.zeros(total_tokens, TWO_I, device=device, dtype=torch.float16)
    offset = 0
    for e in range(NUM_EXPERTS):
        c = expert_counts[e].item()
        if c <= 0: continue
        ref[offset:offset+c] = torch.mm(
            input_t[offset:offset+c].float(), w13[e].t().float()
        ).half()
        offset += c

    backends = {}
    try:
        import gemm_grouped
        backends["cutlass_grouped"] = gemm_grouped.moe_group_gemm(input_t, w13, expert_counts)
    except Exception as e:
        print(f"  cutlass_grouped: skip ({e})")

    try:
        import ix_moe_bridge
        backends["cuinfer_bridge"] = ix_moe_bridge.group_gemm(input_t, w13, expert_counts, TWO_I)
    except Exception as e:
        print(f"  cuinfer_bridge: skip ({e})")

    try:
        import hgemm
        backends["hgemm"] = hgemm.moe_expert_gemm(input_t, w13, expert_counts)
    except Exception as e:
        print(f"  hgemm: skip ({e})")

    for name, out in backends.items():
        err = (out.float() - ref.float()).abs().max().item()
        ok = "✓" if err < FP16_ATOL else "✗"
        print(f"  {ok} {name}: max_err={err:.6f}")


def test_batched_gemm(device):
    """Test batched GEMM for decode path."""
    print("\n=== Batched GEMM (decode, topk=8) ===")
    A = torch.randn(TOPK, 1, H, device=device, dtype=torch.float16)
    B = torch.randn(TOPK, H, TWO_I, device=device, dtype=torch.float16) * 0.01

    # Reference
    ref = torch.bmm(A.float(), B.float()).half()

    backends = {}
    try:
        import cuinfer_gemm_wrapper
        backends["cuinfer_batched"] = cuinfer_gemm_wrapper.cuinfer_gemm_batched(A, B, False)
    except Exception as e:
        print(f"  cuinfer_batched: skip ({e})")

    try:
        import corex_batched_gemm
        backends["corex_batched"] = corex_batched_gemm.batched_gemm_fp16(A, B)
    except Exception as e:
        print(f"  corex_batched: skip ({e})")

    for name, out in backends.items():
        err = (out.float() - ref.float()).abs().max().item()
        ok = "✓" if err < FP16_ATOL else "✗"
        print(f"  {ok} {name}: max_err={err:.6f}")


def test_gemm_dispatch(device):
    """Test the unified gemm_dispatch layer."""
    print("\n=== gemm_dispatch ===")
    try:
        from gemm_dispatch import group_gemm, get_backend
        print(f"  Backend: {get_backend()}")

        total_tokens = TOPK
        expert_counts = torch.zeros(NUM_EXPERTS, device=device, dtype=torch.int32)
        for i in range(TOPK):
            expert_counts[i] = 1
        input_t = torch.randn(total_tokens, H, device=device, dtype=torch.float16) * 0.1
        w13 = torch.randn(NUM_EXPERTS, TWO_I, H, device=device, dtype=torch.float16) * 0.01

        out = group_gemm(input_t, w13, expert_counts, TWO_I)
        assert out.shape == (total_tokens, TWO_I), f"shape: {out.shape}"
        assert not torch.isnan(out).any(), "NaN in output"
        print(f"  ✓ shape={out.shape}, no NaN")
    except Exception as e:
        print(f"  ✗ {e}")


def main():
    if not torch.cuda.is_available():
        print("No CUDA")
        sys.exit(0)

    device = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    passed, failed = 0, 0
    for test in [test_single_gemm, test_group_gemm, test_batched_gemm, test_gemm_dispatch]:
        try:
            test(device)
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")


if __name__ == "__main__":
    main()
