"""test_moe_bridge.py — Integration test for ix_moe_bridge on real device.

Run after build_moe_bridge.sh. No model weights needed — uses random tensors.
Tests each of the 5 MoE functions + the fused pipeline.

Usage:
    python3 test_moe_bridge.py
"""
import sys
import os
import torch
import time

# Qwen3.5-27B MoE params
NUM_EXPERTS = 128
TOPK = 8
HIDDEN_SIZE = 3584
INTERMEDIATE_SIZE = 18944  # per-partition (full=18944*2 for gate+up, /TP if sharded)
NUM_TOKENS = 4

def load_bridge():
    """Try to load ix_moe_bridge."""
    # Try prebuilt
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for p in [
        os.path.join(script_dir, "prebuilt", "ix_moe_bridge.so"),
        os.path.join(script_dir, "ix_moe_bridge.so"),
    ]:
        if os.path.isfile(p):
            import importlib.util
            spec = importlib.util.spec_from_file_location("ix_moe_bridge", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

    # Try import
    import ix_moe_bridge
    return ix_moe_bridge


def test_topk_softmax(bridge, device):
    print("\n--- topk_softmax ---")
    gating = torch.randn(NUM_TOKENS, NUM_EXPERTS, device=device, dtype=torch.float32)
    topk_w, topk_ids, token_expert_ids = bridge.topk_softmax(gating, TOPK, True)

    assert topk_w.shape == (NUM_TOKENS, TOPK), f"weights shape: {topk_w.shape}"
    assert topk_ids.shape == (NUM_TOKENS, TOPK), f"ids shape: {topk_ids.shape}"
    assert topk_w.dtype == torch.float32
    assert topk_ids.dtype == torch.int32
    assert (topk_ids >= 0).all() and (topk_ids < NUM_EXPERTS).all(), "ids out of range"
    assert torch.allclose(topk_w.sum(-1), torch.ones(NUM_TOKENS, device=device), atol=1e-5), \
        f"weights don't sum to 1: {topk_w.sum(-1)}"
    print(f"  ✓ shape={topk_w.shape}, sum={topk_w.sum(-1).tolist()}")
    print(f"  ✓ top expert ids (row 0): {topk_ids[0].tolist()}")


def test_moe_gen_idx(bridge, device):
    print("\n--- moe_gen_idx ---")
    expert_ids = torch.randint(0, NUM_EXPERTS, (NUM_TOKENS * TOPK,),
                                device=device, dtype=torch.int32)
    results = bridge.moe_gen_idx(expert_ids, NUM_EXPERTS)
    src_dst, dst_src, expert_sizes, expert_cumsum = results

    assert src_dst.shape == (NUM_TOKENS * TOPK,), f"src_dst shape: {src_dst.shape}"
    assert dst_src.shape == (NUM_TOKENS * TOPK,), f"dst_src shape: {dst_src.shape}"
    assert expert_sizes.shape[0] == NUM_EXPERTS, f"expert_sizes shape: {expert_sizes.shape}"
    assert expert_sizes.sum().item() == NUM_TOKENS * TOPK, \
        f"expert_sizes sum: {expert_sizes.sum().item()} != {NUM_TOKENS * TOPK}"
    print(f"  ✓ src_dst={src_dst.shape}, expert_sizes sum={expert_sizes.sum().item()}")


def test_moe_expand_input(bridge, device):
    print("\n--- moe_expand_input ---")
    hidden = torch.randn(NUM_TOKENS, HIDDEN_SIZE, device=device, dtype=torch.float16)
    # Create simple gather index: [0,1,2,...,NUM_TOKENS*TOPK-1] mod NUM_TOKENS
    gather_idx = torch.arange(NUM_TOKENS * TOPK, device=device, dtype=torch.int32) % NUM_TOKENS
    combine_idx = torch.arange(NUM_TOKENS * TOPK, device=device, dtype=torch.int32)

    expanded = bridge.moe_expand_input(hidden, gather_idx, combine_idx, TOPK)
    assert expanded.shape == (NUM_TOKENS * TOPK, HIDDEN_SIZE), f"shape: {expanded.shape}"
    print(f"  ✓ shape={expanded.shape}, dtype={expanded.dtype}")


def test_group_gemm(bridge, device):
    print("\n--- group_gemm ---")
    # Simulate: expanded tokens × expert weights
    total_tokens = NUM_TOKENS * TOPK  # 32
    inputs = torch.randn(total_tokens, HIDDEN_SIZE, device=device, dtype=torch.float16)
    # weights: [NUM_EXPERTS, 2*INTERMEDIATE, HIDDEN] — 3D
    weights = torch.randn(NUM_EXPERTS, INTERMEDIATE_SIZE * 2, HIDDEN_SIZE,
                           device=device, dtype=torch.float16) * 0.01
    # tokens_per_expert: distribute evenly
    tpe = torch.zeros(NUM_EXPERTS, device=device, dtype=torch.int32)
    for i in range(total_tokens):
        tpe[i % NUM_EXPERTS] += 1

    output_n = INTERMEDIATE_SIZE * 2
    result = bridge.group_gemm(inputs, weights, tpe, output_n)
    assert result.shape == (total_tokens, output_n), f"shape: {result.shape}"
    assert not torch.isnan(result).any(), "NaN in group_gemm output"
    print(f"  ✓ shape={result.shape}, max={result.abs().max().item():.4f}")


def test_silu_and_mul(bridge, device):
    print("\n--- silu_and_mul ---")
    gate_up = torch.randn(NUM_TOKENS, INTERMEDIATE_SIZE * 2,
                           device=device, dtype=torch.float16)
    activated = bridge.silu_and_mul(gate_up)
    assert activated.shape == (NUM_TOKENS, INTERMEDIATE_SIZE), f"shape: {activated.shape}"
    print(f"  ✓ shape={activated.shape}")


def test_moe_combine_result(bridge, device):
    print("\n--- moe_combine_result ---")
    expert_out = torch.randn(NUM_TOKENS * TOPK, HIDDEN_SIZE,
                              device=device, dtype=torch.float16)
    weights = torch.randn(NUM_TOKENS, TOPK, device=device, dtype=torch.float32)
    weights = torch.softmax(weights, dim=-1)

    combined = bridge.moe_combine_result(expert_out, weights)
    assert combined.shape == (NUM_TOKENS, HIDDEN_SIZE), f"shape: {combined.shape}"
    assert not torch.isnan(combined).any(), "NaN in combine output"
    print(f"  ✓ shape={combined.shape}")


def test_fused_pipeline(bridge, device):
    print("\n--- fused_moe_forward (7-step pipeline) ---")
    hidden = torch.randn(NUM_TOKENS, HIDDEN_SIZE, device=device, dtype=torch.float16)
    router = torch.randn(NUM_TOKENS, NUM_EXPERTS, device=device, dtype=torch.float16)
    w13 = torch.randn(NUM_EXPERTS, INTERMEDIATE_SIZE * 2, HIDDEN_SIZE,
                       device=device, dtype=torch.float16) * 0.01
    w2 = torch.randn(NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE,
                      device=device, dtype=torch.float16) * 0.01

    t0 = time.time()
    output = bridge.fused_moe_forward(hidden, router, w13, w2, TOPK, NUM_EXPERTS, True)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    assert output.shape == (NUM_TOKENS, HIDDEN_SIZE), f"shape: {output.shape}"
    assert not torch.isnan(output).any(), "NaN in fused output"
    print(f"  ✓ shape={output.shape}, time={elapsed*1000:.1f}ms")


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU tests")
        sys.exit(0)

    device = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Params: {NUM_EXPERTS} experts, topk={TOPK}, hidden={HIDDEN_SIZE}, "
          f"inter={INTERMEDIATE_SIZE}, tokens={NUM_TOKENS}")

    bridge = load_bridge()
    funcs = [f for f in dir(bridge) if not f.startswith('_')]
    print(f"Bridge loaded: {len(funcs)} functions: {funcs}")

    passed = 0
    failed = 0

    for test_fn in [
        test_topk_softmax,
        test_moe_gen_idx,
        test_moe_expand_input,
        test_group_gemm,
        test_silu_and_mul,
        test_moe_combine_result,
        test_fused_pipeline,
    ]:
        try:
            test_fn(bridge, device)
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()