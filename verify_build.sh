#!/usr/bin/env bash
# verify_build.sh — 在BI-V100真机上验证完整build链
#
# Exit codes:
#   0 — all checks passed
#   1 — one or more checks failed (see summary at end)
#
# Usage:
#   ./verify_build.sh           # run all checks
#   ./verify_build.sh --quick   # syntax checks only (no CUDA)
set -euo pipefail
cd "$(dirname "$0")"

FAIL_COUNT=0
PASS_COUNT=0

run_check() {
  local name="$1"; shift
  echo ""
  echo "=== ${name} ==="
  if "$@"; then
    echo "✓ PASS: ${name}"
    ((PASS_COUNT++)) || true
  else
    echo "✗ FAIL: ${name}"
    ((FAIL_COUNT++)) || true
  fi
}

run_check "1. patch_ops.sh syntax" bash -n qwen3_6_scripts/patch_ops.sh

echo ""
echo "=== 2. build ix_unified_bridge.so ==="
bash ex_engine/build_unified_bridge.sh 2>&1 | tail -10

echo ""
echo "=== 3. verify bridge load ==="
python3 << 'PY'
import importlib.util, glob
so = glob.glob("ex_engine/build/ix_unified_bridge*.so")
if not so:
    print("✗ .so not built"); exit(1)
spec = importlib.util.spec_from_file_location("ix_unified_bridge", so[0])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
funcs = [x for x in dir(mod) if not x.startswith('_')]
print(f"✓ {len(funcs)} functions: {funcs}")
PY

echo ""
echo "=== 4. ix_unified dispatch smoke test ==="
python3 << 'PY'
import sys, os
sys.path.insert(0, "ex_engine/build")
sys.path.insert(0, "ex_engine/python")
os.environ["IX_BRIDGE_PATH"] = "ex_engine/build"
from ix_unified import ix
print(f"bridge={ix._bridge is not None}, ixformer={ix._ixf is not None}")

import torch
# silu_and_mul
x = torch.randn(2, 512, device="cuda", dtype=torch.float16)
out = ix.silu_and_mul(x)
print(f"✓ silu_and_mul: {x.shape} → {out.shape}")

# rms_norm
inp = torch.randn(4, 2048, device="cuda", dtype=torch.float16)
outp = torch.empty_like(inp)
w = torch.ones(2048, device="cuda", dtype=torch.float16)
ix.rms_norm(outp, inp, w, 1e-6)
print(f"✓ rms_norm: {inp.shape}")

# moe_topk_softmax
gate = torch.randn(8, 256, device="cuda", dtype=torch.float16)
weights, indices = ix.moe_topk_softmax(gate, 8, True)
print(f"✓ moe_topk_softmax: {gate.shape} → weights={weights.shape} indices={indices.shape}")

print("ALL SMOKE TESTS PASSED")
PY

echo ""
echo "=== 5. prebuilt .so still load ==="
python3 << 'PY'
import importlib.util, os
so_dir = "qwen3_6_scripts/prebuilt/corex-3.2.3-ivcore10"
for name in ["corex_moe_direct_routed", "corex_gdn_packed_decode", "corex_gdn_causal_conv"]:
    so = os.path.join(so_dir, f"{name}.so")
    spec = importlib.util.spec_from_file_location(name, so)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    funcs = [x for x in dir(mod) if not x.startswith('_')]
    print(f"✓ {name}: {funcs}")
PY

run_check "6. py_compile all" bash -c '
  cd qwen3_6_scripts &&
  find . -path "./wheels" -prune -o -name "*.py" -print0 |
    xargs -0 python3 -m py_compile 2>&1
'

# ---------- C++ core library build (no CUDA required) ----------
if command -v cmake >/dev/null 2>&1; then
  run_check "7. core/ CMake configure" bash -c '
    cd core && mkdir -p build_verify && cd build_verify &&
    cmake .. -DUSE_ILU=ON -DBUILD_TESTING=ON 2>&1 | tail -5
  '
  run_check "8. core/ compile + link" bash -c '
    cd core/build_verify && make -j$(nproc) 2>&1 | tail -5
  '
  run_check "9. core/ unit tests" bash -c '
    cd core/build_verify && ./test_layerwise_split 2>&1 | tail -5
  '
fi

echo ""
echo "==========================================="
echo "  SUMMARY: ${PASS_COUNT} passed, ${FAIL_COUNT} failed"
echo "==========================================="
if [ "${FAIL_COUNT}" -gt 0 ]; then
  exit 1
fi
