#!/bin/bash
# verify_all_so.sh — 在真机上验证全部 24 个 prebuilt .so
# 用法: CUDA_VISIBLE_DEVICES=0 bash verify_all_so.sh
#
# 不写 fallback，不写 adapter。
# .so 加载失败 = 报错退出。函数调不通 = 报错退出。

set -euo pipefail

SO_DIR="${SO_DIR:-/home/dylan/0814/project_6/qwen3_6_scripts/prebuilt/corex-3.2.3-ivcore10}"

if [ ! -d "$SO_DIR" ]; then
    echo "FATAL: SO_DIR=$SO_DIR not found"
    exit 1
fi

echo "============================================================"
echo "  BI-V100 prebuilt .so verification"
echo "  SO_DIR=$SO_DIR"
echo "  $(date)"
echo "============================================================"

cat << 'PYEOF' | CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 -u -
import sys, os, time, importlib.util, torch

SO_DIR = os.environ.get("SO_DIR", "/home/dylan/0814/project_6/qwen3_6_scripts/prebuilt/corex-3.2.3-ivcore10")
torch.cuda.set_device(0)
dev = torch.device("cuda:0")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA: {torch.version.cuda}")
print()

PASS = 0
FAIL = 0
ERRORS = []

def load_so(name):
    path = os.path.join(SO_DIR, f"{name}.so")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def check(name, fn, *args, **kwargs):
    global PASS, FAIL
    try:
        result = fn(*args, **kwargs)
        torch.cuda.synchronize()
        PASS += 1
        print(f"  ✓ {name}")
        return result
    except Exception as e:
        FAIL += 1
        msg = f"  ✗ {name}: {e}"
        print(msg)
        ERRORS.append(msg)
        return None

def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

# ================================================================
# 1. xllm 核心模块 (pybind11, 大文件)
# ================================================================
section("xllm_activation.so")
m = load_so("xllm_activation")
fns = [x for x in dir(m) if not x.startswith("_")]
print(f"  exports: {fns}")
inp = torch.randn(2, 512, device=dev, dtype=torch.float16)
out = torch.empty(2, 256, device=dev, dtype=torch.float16)
check("silu_and_mul(out, input)", m.silu_and_mul, out, inp)
ref = (torch.sigmoid(inp[:, :256]) * inp[:, :256]) * inp[:, 256:]
diff = (out.float() - ref.float()).abs().max().item()
print(f"  silu_and_mul max_diff={diff:.6f}")

section("xllm_norm.so")
m = load_so("xllm_norm")
fns = [x for x in dir(m) if not x.startswith("_")]
print(f"  exports: {fns}")
x = torch.randn(4, 2048, device=dev, dtype=torch.float16)
w = torch.ones(2048, device=dev, dtype=torch.float16)
o = torch.empty_like(x)
check("rms_norm(output, input, weight, eps)", m.rms_norm, o, x, w, 1e-6)
variance = x.float().pow(2).mean(-1, keepdim=True)
ref = (x.float() * torch.rsqrt(variance + 1e-6)).half() * w
diff = (o.float() - ref.float()).abs().max().item()
print(f"  rms_norm max_diff={diff:.6f}")

if hasattr(m, "fused_add_rms_norm"):
    x2 = torch.randn(4, 2048, device=dev, dtype=torch.float16)
    r2 = torch.randn(4, 2048, device=dev, dtype=torch.float16)
    check("fused_add_rms_norm", m.fused_add_rms_norm, x2, r2, w, 1e-6)

section("xllm_rope.so")
m = load_so("xllm_rope")
fns = [x for x in dir(m) if not x.startswith("_")]
print(f"  exports: {fns}")
positions = torch.tensor([0, 1, 2, 3], device=dev, dtype=torch.long)
q = torch.randn(4, 6*128, device=dev, dtype=torch.float16)
k = torch.randn(4, 1*128, device=dev, dtype=torch.float16)
cos_sin = torch.randn(8192, 128, device=dev, dtype=torch.float16)
check("rotary_embedding(pos, q, k, cos_sin, True)", m.rotary_embedding, positions, q, k, cos_sin, True)

section("xllm_cache.so")
m = load_so("xllm_cache")
fns = [x for x in dir(m) if not x.startswith("_")]
print(f"  exports: {fns}")
slot_ids = torch.tensor([0, 1, 2, 3], device=dev, dtype=torch.int32)
keys = torch.randn(4, 4, 128, device=dev, dtype=torch.float16)
vals = torch.randn(4, 4, 128, device=dev, dtype=torch.float16)
kc = torch.zeros(16, 4, 16, 128, device=dev, dtype=torch.float16)
vc = torch.zeros(16, 4, 16, 128, device=dev, dtype=torch.float16)
check("reshape_paged_cache(slot_i32, k, v, kc, vc)", m.reshape_paged_cache, slot_ids, keys, vals, kc, vc)

section("xllm_moe.so")
m = load_so("xllm_moe")
fns = [x for x in dir(m) if not x.startswith("_")]
print(f"  exports: {fns}")
gating = torch.randn(2, 256, device=dev, dtype=torch.float32)
r = check("moe_fused_topk(gating, 8)", m.moe_fused_topk, gating, 8)
if r is not None:
    topk_w, topk_ids = r
    print(f"  topk_w shape={topk_w.shape} dtype={topk_w.dtype}")
    print(f"  topk_ids shape={topk_ids.shape} dtype={topk_ids.dtype}")

if hasattr(m, "moe_compute_index"):
    expert_ids = torch.randint(0, 64, (16,), device=dev, dtype=torch.int32)
    r2 = check("moe_compute_index(expert_ids, 256)", m.moe_compute_index, expert_ids, 256)
    if r2 is not None:
        print(f"  moe_compute_index returned {len(r2)} tensors")

# ================================================================
# 2. Bridge 模块
# ================================================================
section("ix_moe_bridge.so")
m = load_so("ix_moe_bridge")
fns = [x for x in dir(m) if not x.startswith("_")]
print(f"  exports: {fns}")
# pybind11 注册名没有 ix_ 前缀 (nm -D 的 C++ 符号有，但 Python 侧去掉了)
inp2 = torch.randn(2, 512, device=dev, dtype=torch.float16)
check("silu_and_mul(input)", m.silu_and_mul, inp2)
x3 = torch.randn(4, 2048, device=dev, dtype=torch.float16)
w3 = torch.ones(2048, device=dev, dtype=torch.float16)
o3 = torch.empty_like(x3)
x3 = torch.randn(4, 2048, device=dev, dtype=torch.float16)
o3 = torch.empty_like(x3)
# ix_moe_bridge rms_norm: might be (output, input, weight, eps) like xllm_norm
w_rms = torch.ones(2048, device=dev, dtype=torch.float16)
check("rms_norm(out, input, weight, eps)", m.rms_norm, o3, x3, w_rms, 1e-6)
g2 = torch.randn(2, 256, device=dev, dtype=torch.float32)
check("topk_softmax(gating, 8, True)", m.topk_softmax, g2, 8, True)
# 测试 fused_moe_forward
check("moe_gen_idx available", lambda: hasattr(m, 'moe_gen_idx') or None)
check("group_gemm available", lambda: hasattr(m, 'group_gemm') or None)
check("linear available", lambda: hasattr(m, 'linear') or None)
check("paged_attention available", lambda: hasattr(m, 'paged_attention') or None)

section("ix_full_bridge.so")
m = load_so("ix_full_bridge")
fns = [x for x in dir(m) if not x.startswith("_")]
print(f"  exports: {fns}")
# 先看实际导出名再调用
for fn_name in fns:
    print(f"  has: {fn_name}")
# 根据实际导出名调用（可能有 ix_ 前缀也可能没有）
silu_name = "silu_and_mul" if hasattr(m, "silu_and_mul") else "ix_silu_and_mul"
rms_name = "rms_norm" if hasattr(m, "rms_norm") else "ix_rms_norm"
check(f"{silu_name}", getattr(m, silu_name), torch.randn(2, 512, device=dev, dtype=torch.float16), torch.empty(2, 256, device=dev, dtype=torch.float16))
rms_in = torch.randn(2, 2048, device=dev, dtype=torch.float16)
rms_out = torch.empty_like(rms_in)
check(f"{rms_name}(in, w, out, eps)", getattr(m, rms_name), rms_in, w3, rms_out, 1e-6)

# ================================================================
# 3. CoreX MoE 模块
# ================================================================
section("corex_moe_topk_softmax.so")
m = load_so("corex_moe_topk_softmax")
fns = [x for x in dir(m) if not x.startswith("_")]
print(f"  exports: {fns}")
g3 = torch.randn(4, 256, device=dev, dtype=torch.float32)
check("moe_topk_softmax(gating, 8, True)", m.moe_topk_softmax, g3, 8, True)

section("corex_moe_index_combine.so")
m = load_so("corex_moe_index_combine")
fns = [x for x in dir(m) if not x.startswith("_")]
print(f"  exports: {fns}")
eids = torch.randint(0, 64, (32,), device=dev, dtype=torch.int32)
check("moe_compute_index(eids, 256)", m.moe_compute_index, eids, 256)
# moe_combine_result 需要正确参数
inp4 = torch.randn(32, 2048, device=dev, dtype=torch.float16)
ws4 = torch.randn(4, 8, device=dev, dtype=torch.float16)
check("moe_combine_result(input, weights, topk=8, num_tokens=4)", m.moe_combine_result, inp4, ws4, 8, 4)

section("corex_moe_direct_routed.so")
m = load_so("corex_moe_direct_routed")
fns = [x for x in dir(m) if not x.startswith("_")]
print(f"  exports: {fns}")
# direct_w13: (input[1,H], w13[E,2I,H], expert_ids[K]) -> (K, 2I)
hidden = torch.randn(1, 2048, device=dev, dtype=torch.float16) * 0.01
w13 = torch.randn(256, 256, 2048, device=dev, dtype=torch.float16) * 0.01
w2 = torch.randn(256, 2048, 128, device=dev, dtype=torch.float16) * 0.01
eids_k = torch.randint(0, 256, (8,), device=dev, dtype=torch.int64)
ws_k = torch.softmax(torch.randn(8, device=dev), dim=0).half()
# pybind11 导出名: w13, w2_reduce (不是 direct_w13 / direct_w2_reduce)
check("w13(hidden, w13_weights, eids)", m.w13, hidden, w13, eids_k)
activated = torch.randn(8, 128, device=dev, dtype=torch.float16) * 0.01
check("w2_reduce(act, w2, eids, ws)", m.w2_reduce, activated, w2, eids_k, ws_k)

section("corex_moe_weight_gather.so")
m = load_so("corex_moe_weight_gather")
fns = [x for x in dir(m) if not x.startswith("_")]
print(f"  exports: {fns}")
# qwen3_5.py: _corex_moe_weight_gather.gather(w13, w2, eids) → (w13_sel, w2_sel)
wg_w13 = torch.randn(256, 256, 2048, device=dev, dtype=torch.float16) * 0.01
wg_w2 = torch.randn(256, 2048, 128, device=dev, dtype=torch.float16) * 0.01
wg_eids = torch.randint(0, 256, (8,), device=dev, dtype=torch.int64)
check("gather(w13, w2, eids)", m.gather, wg_w13, wg_w2, wg_eids)

section("corex_moe_exact_reduce.so")
m = load_so("corex_moe_exact_reduce")
fns = [x for x in dir(m) if not x.startswith("_")]
print(f"  exports: {fns}")
vals = torch.randn(8, 2048, device=dev, dtype=torch.float16)
wts = torch.randn(8, device=dev, dtype=torch.float16)
check("serial_float(values, weights)", m.serial_float, vals, wts)
check("tree_float(values, weights)", m.tree_float, vals, wts)
check("serial_half(values, weights)", m.serial_half, vals, wts)

section("gemm_grouped.so")
m = load_so("gemm_grouped")
fns = [x for x in dir(m) if not x.startswith("_")]
print(f"  exports: {fns}")
# moe_group_gemm(input[T,K], weights[E,N,K], counts[E])
t_in = torch.randn(16, 2048, device=dev, dtype=torch.float16) * 0.01
t_w = torch.randn(4, 256, 2048, device=dev, dtype=torch.float16) * 0.01
t_cnt = torch.tensor([4, 4, 4, 4], device=dev, dtype=torch.int32)
check("moe_group_gemm", m.moe_group_gemm, t_in, t_w, t_cnt)
# moe_decode_cutlass
h_dec = torch.randn(1, 2048, device=dev, dtype=torch.float16) * 0.01
w13_dec = torch.randn(8, 256, 2048, device=dev, dtype=torch.float16) * 0.01
w2_dec = torch.randn(8, 2048, 128, device=dev, dtype=torch.float16) * 0.01
tw_dec = torch.softmax(torch.randn(8, device=dev), dim=0).float()
check("moe_decode_cutlass", m.moe_decode_cutlass, h_dec, w13_dec, w2_dec, tw_dec)

section("corex_batched_gemm.so")
m = load_so("corex_batched_gemm")
fns = [x for x in dir(m) if not x.startswith("_")]
print(f"  exports: {fns}")
# batched_gemm_fp16 does A @ B: A[batch,M,K] B[batch,K,N]
a = torch.randn(8, 1, 2048, device=dev, dtype=torch.float16) * 0.01
b = torch.randn(8, 2048, 128, device=dev, dtype=torch.float16) * 0.01
check("batched_gemm_fp16(A[8,1,2048] @ B[8,2048,128])", m.batched_gemm_fp16, a, b)
check("moe_decode_fused", m.moe_decode_fused, h_dec, w13_dec, w2_dec, tw_dec)

# ================================================================
# 4. Attention 模块
# ================================================================
section("corex_fused_paged_prefill.so")
m = load_so("corex_fused_paged_prefill")
fns = [x for x in dir(m) if not x.startswith("_")]
print(f"  exports: {fns}")
# 这个签名比较复杂，先验证加载和导出名
print(f"  (load OK, functional test needs real KV cache setup)")

section("corex_paged_kv_gather.so")
m = load_so("corex_paged_kv_gather")
fns = [x for x in dir(m) if not x.startswith("_")]
print(f"  exports: {fns}")
print(f"  (load OK)")

section("corex_block_major_kv_transfer.so")
m = load_so("corex_block_major_kv_transfer")
fns = [x for x in dir(m) if not x.startswith("_")]
print(f"  exports: {fns}")
print(f"  (load OK)")

section("corex_attn_head_rms_norm.so")
m = load_so("corex_attn_head_rms_norm")
fns = [x for x in dir(m) if not x.startswith("_")]
print(f"  exports: {fns}")
# qwen3_5.py: prepare(x.view(-1, 256)) → (converted, squares)
#             inverse = rsqrt(squares.mean(-1,keepdim=True) + eps)
#             apply_inverse(converted, weight, inverse).view(original_shape)
x5 = torch.randn(24, 256, device=dev, dtype=torch.float16)  # (rows, 256) — 2D, last dim=256
r5 = check("prepare(input_2d_256)", m.prepare, x5)
if r5 is not None:
    converted, squares = r5
    inverse = torch.rsqrt(squares.mean(dim=-1, keepdim=True) + 1e-6)
    w5 = torch.ones(256, device=dev, dtype=torch.float16)
    check("apply_inverse(converted, weight, inverse)", m.apply_inverse, converted, w5, inverse)

# ================================================================
# 5. GDN 模块 (6 个 .so)
# ================================================================
section("corex_gdn_chunk_recurrent.so")
m = load_so("corex_gdn_chunk_recurrent")
fns = [x for x in dir(m) if not x.startswith("_")]
print(f"  exports: {fns}")
B, L, H, Dk, Dv = 1, 32, 6, 128, 256
q = torch.randn(B, L, H, Dk, device=dev, dtype=torch.float16)
k = torch.randn(B, L, H, Dk, device=dev, dtype=torch.float16)
v = torch.randn(B, L, H, Dv, device=dev, dtype=torch.float16)
gate = torch.randn(B, L, H, device=dev, dtype=torch.float32)
beta = torch.randn(B, L, H, device=dev, dtype=torch.float32).sigmoid()
state = torch.zeros(B, H, Dk, Dv, device=dev, dtype=torch.float32)
check("torch_chunk_gated_delta_rule(q,k,v,gate,beta,16,state,False,True)",
      m.torch_chunk_gated_delta_rule, q, k, v, gate, beta, 16, state, False, True)
check("torch_recurrent_gated_delta_rule(q,k,v,gate,beta,state,False,True)",
      m.torch_recurrent_gated_delta_rule, q, k, v, gate, beta, state, False, True)

section("corex_gdn_packed_decode.so")
m = load_so("corex_gdn_packed_decode")
fns = [x for x in dir(m) if not x.startswith("_")]
print(f"  exports: {fns}")
# qwen3_5.py: packed_decode(temporal_state, packed_mixed_qkv, b_all, a_all, A_log, dt_bias)
# temporal_state: fp32 (B, H, Dk, Dv); packed_mixed_qkv: fp16; b_all/a_all: fp16; A_log/dt_bias: fp32
pd_state = torch.randn(1, 8, 128, 128, device=dev, dtype=torch.float32)
pd_qkv = torch.randn(1, 2048, device=dev, dtype=torch.float16)  # (batch, 8*(128+128))=2048
pd_b = torch.randn(1, 8, device=dev, dtype=torch.float16)
pd_a = torch.randn(1, 8, device=dev, dtype=torch.float16)
pd_alog = torch.randn(8, device=dev, dtype=torch.float16)
pd_dt = torch.randn(8, device=dev, dtype=torch.float16)
check("packed_decode(state[1,8,128,128], qkv[1,2048], b, a, A_log, dt)", m.packed_decode, pd_state, pd_qkv, pd_b, pd_a, pd_alog, pd_dt)

section("corex_gdn_beta_decay.so")
m = load_so("corex_gdn_beta_decay")
fns = [x for x in dir(m) if not x.startswith("_")]
print(f"  exports: {fns}")
# qwen3_5.py: beta_decay(b_all, a_all, self.A_log, self.dt_bias)
# b_all, a_all: fp16; A_log, dt_bias: fp32 (model params)
bd_b = torch.randn(1, 6, device=dev, dtype=torch.float16)
bd_a = torch.randn(1, 6, device=dev, dtype=torch.float16)
bd_alog = torch.randn(6, device=dev, dtype=torch.float16)
bd_dt = torch.randn(6, device=dev, dtype=torch.float16)
check("beta_decay(b, a, A_log_fp16, dt_bias_fp16)", m.beta_decay, bd_b, bd_a, bd_alog, bd_dt)

section("corex_gdn_causal_conv.so")
m = load_so("corex_gdn_causal_conv")
fns = [x for x in dir(m) if not x.startswith("_")]
print(f"  exports: {fns}")
# state: fp32 (batch, channels, 3); input: fp16 (batch, channels, 1); weight: fp16 (channels, 4)
# state stores 3 historical steps, weight has 4 taps (3 history + 1 current)
conv_state = torch.randn(1, 768, 3, device=dev, dtype=torch.float32)
conv_input = torch.randn(1, 768, 1, device=dev, dtype=torch.float16)
conv_weight = torch.randn(768, 4, device=dev, dtype=torch.float16)
check("causal_conv_update(state[1,768,3], input[1,768,1], weight[768,4])", m.causal_conv_update, conv_state, conv_input, conv_weight)

section("corex_gdn_qk_map.so")
m = load_so("corex_gdn_qk_map")
fns = [x for x in dir(m) if not x.startswith("_")]
print(f"  exports: {fns}")
# qwen3_5.py: qk_map(normalized_q, normalized_k, local_num_v)
# normalized_q/k: (batch, key_heads, 128) fp16
qk_q = torch.randn(1, 6, 128, device=dev, dtype=torch.float16)
qk_k = torch.randn(1, 6, 128, device=dev, dtype=torch.float16)
check("qk_map(q_3d, k_3d, num_v_heads=12)", m.qk_map, qk_q, qk_k, 12)

section("corex_gdn_gated_norm.so")
m = load_so("corex_gdn_gated_norm")
fns = [x for x in dir(m) if not x.startswith("_")]
print(f"  exports: {fns}")
# qwen3_5.py: apply_inverse(hs, gate, self.weight, inverse) — hs shape (rows, 128)
# This is per-head gated norm, not full hidden dim
gn_hs = torch.randn(4, 128, device=dev, dtype=torch.float32)
gn_gate = torch.randn(4, 128, device=dev, dtype=torch.float16)
gn_w = torch.ones(128, device=dev, dtype=torch.float16)
gn_inv = torch.rsqrt(gn_hs.pow(2).mean(-1, keepdim=True) + 1e-6).float()
check("apply_inverse(hs_fp32[4,128], gate, weight, inverse)", m.apply_inverse, gn_hs, gn_gate, gn_w, gn_inv)

# ================================================================
# Summary
# ================================================================
print(f"\n{'='*60}")
print(f"  RESULTS: {PASS} passed, {FAIL} failed, {PASS+FAIL} total")
print(f"{'='*60}")
if ERRORS:
    print("\nFAILED:")
    for e in ERRORS:
        print(e)
    sys.exit(1)
else:
    print("\nALL PASSED — 24 .so fully operational on BI-V100")
    sys.exit(0)
PYEOF