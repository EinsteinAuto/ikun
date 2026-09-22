"""bench_gemm.py — Benchmark all GEMM backends on real device.

Tests with Qwen3.5-27B MoE shapes:
  - Decode: M=1, K=3584, N=18944*2 (gate_up) / N=3584 (down)
  - Prefill: M=variable, same K/N

Usage:
    python3 bench_gemm.py
"""
import sys
import os
import time
import torch

# Qwen3.5-27B params (per TP=4 partition)
H = 3584               # hidden_size
I = 18944 // 4          # intermediate per partition (4736)
TWO_I = I * 2           # gate + up
NUM_EXPERTS = 128
TOPK = 8

WARMUP = 5
REPEATS = 20


def bench_fn(fn, *args, name=""):
    """Benchmark a function, return ms per call."""
    for _ in range(WARMUP):
        fn(*args)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(REPEATS):
        fn(*args)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / REPEATS * 1000
    print(f"  {name}: {elapsed:.3f} ms")
    return elapsed


def bench_single_gemm(device):
    """Benchmark single GEMM: (M,K) × (K,N) for various M."""
    print("\n=== Single GEMM (M,K)×(K,N) ===")
    for M in [1, 4, 8, 32]:
        A = torch.randn(M, H, device=device, dtype=torch.float16)
        B = torch.randn(H, TWO_I, device=device, dtype=torch.float16)

        bench_fn(torch.mm, A, B, name=f"torch.mm M={M} K={H} N={TWO_I}")

        # Try hgemm
        try:
            import hgemm
            bench_fn(hgemm.hgemm, A, B, name=f"hgemm M={M}")
        except Exception:
            pass

        # Try ixformer linear
        try:
            import ix_moe_bridge as bridge
            bench_fn(bridge.linear, A, B.t().contiguous(), name=f"ixformer_linear M={M}")
        except Exception:
            pass


def bench_group_gemm(device):
    """Benchmark group GEMM with MoE shapes."""
    print("\n=== Group GEMM (MoE w13 projection) ===")

    # Simulate decode: 1 token → topk=8 experts, each gets ~1 token
    total_tokens = TOPK
    expert_counts = torch.zeros(NUM_EXPERTS, device=device, dtype=torch.int32)
    # Distribute tokens to first TOPK experts
    for i in range(TOPK):
        expert_counts[i] = 1

    input_t = torch.randn(total_tokens, H, device=device, dtype=torch.float16)
    w13 = torch.randn(NUM_EXPERTS, TWO_I, H, device=device, dtype=torch.float16) * 0.01

    # PyTorch baseline
    def torch_group_gemm():
        offset = 0
        out = torch.zeros(total_tokens, TWO_I, device=device, dtype=torch.float16)
        for e in range(NUM_EXPERTS):
            c = expert_counts[e].item()
            if c <= 0: continue
            out[offset:offset+c] = torch.mm(input_t[offset:offset+c], w13[e].t())
            offset += c
        return out

    bench_fn(torch_group_gemm, name=f"torch.mm loop (decode, {TOPK} experts)")

    # Try gemm_grouped
    try:
        import gemm_grouped
        bench_fn(gemm_grouped.moe_group_gemm, input_t, w13, expert_counts,
                 name=f"cutlass_grouped (decode, {TOPK} experts)")
    except Exception as e:
        print(f"  cutlass_grouped: {e}")

    # Try ix_moe_bridge
    try:
        import ix_moe_bridge as bridge
        bench_fn(bridge.group_gemm, input_t, w13, expert_counts, TWO_I,
                 name=f"cuinfer_group_gemm (decode, {TOPK} experts)")
    except Exception as e:
        print(f"  cuinfer_group_gemm: {e}")

    # Try hgemm
    try:
        import hgemm
        bench_fn(hgemm.moe_expert_gemm, input_t, w13, expert_counts,
                 name=f"hgemm_expert (decode, {TOPK} experts)")
    except Exception as e:
        print(f"  hgemm_expert: {e}")

    # Prefill shape: 32 tokens
    print("\n=== Group GEMM (MoE w13, prefill M=32) ===")
    total_pf = 32 * TOPK  # 256
    expert_counts_pf = torch.zeros(NUM_EXPERTS, device=device, dtype=torch.int32)
    for i in range(total_pf):
        expert_counts_pf[i % NUM_EXPERTS] += 1
    input_pf = torch.randn(total_pf, H, device=device, dtype=torch.float16)

    def torch_group_gemm_pf():
        offset = 0
        out = torch.zeros(total_pf, TWO_I, device=device, dtype=torch.float16)
        for e in range(NUM_EXPERTS):
            c = expert_counts_pf[e].item()
            if c <= 0: continue
            out[offset:offset+c] = torch.mm(input_pf[offset:offset+c], w13[e].t())
            offset += c
        return out

    bench_fn(torch_group_gemm_pf, name=f"torch.mm loop (prefill, 256 tokens)")

    try:
        import gemm_grouped
        bench_fn(gemm_grouped.moe_group_gemm, input_pf, w13, expert_counts_pf,
                 name=f"cutlass_grouped (prefill, 256 tokens)")
    except Exception as e:
        print(f"  cutlass_grouped: {e}")


def bench_decode_fused(device):
    """Benchmark full MoE decode pipeline."""
    print("\n=== Full MoE Decode (1 token, topk=8) ===")
    hidden = torch.randn(1, H, device=device, dtype=torch.float16)
    w13_sel = torch.randn(TOPK, TWO_I, H, device=device, dtype=torch.float16) * 0.01
    w2_sel = torch.randn(TOPK, H, I, device=device, dtype=torch.float16) * 0.01
    topk_w = torch.softmax(torch.randn(TOPK), dim=0).to(device)

    # PyTorch baseline
    def torch_decode():
        results = []
        for k in range(TOPK):
            gu = torch.mm(hidden, w13_sel[k].t())
            act = torch.silu(gu[:, :I]) * gu[:, I:]
            down = torch.mm(act, w2_sel[k].t())
            results.append(down * topk_w[k])
        return sum(results)

    bench_fn(torch_decode, name="torch.mm loop")

    try:
        import gemm_grouped
        bench_fn(gemm_grouped.moe_decode_cutlass,
                 hidden, w13_sel, w2_sel, topk_w,
                 name="cutlass_batched")
    except Exception as e:
        print(f"  cutlass_batched: {e}")

    try:
        import corex_batched_gemm
        bench_fn(corex_batched_gemm.moe_decode_fused,
                 hidden, w13_sel, w2_sel, topk_w,
                 name="corex_batched")
    except Exception as e:
        print(f"  corex_batched: {e}")


def main():
    if not torch.cuda.is_available():
        print("No CUDA, skipping")
        sys.exit(0)

    device = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Shapes: H={H}, I={I}, 2I={TWO_I}, experts={NUM_EXPERTS}, topk={TOPK}")

    bench_single_gemm(device)
    bench_group_gemm(device)
    bench_decode_fused(device)

    print("\n=== Active backend ===")
    try:
        from gemm_dispatch import get_backend
        print(f"  gemm_dispatch: {get_backend()}")
    except Exception:
        print("  gemm_dispatch not loaded")


if __name__ == "__main__":
    main()
