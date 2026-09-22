#!/bin/bash
# verify_deployment.sh — Run on real BI-V100 machine to verify deployment correctness
#
# USAGE: bash verify_deployment.sh
#
# This script verifies:
# 1. Base image qwen3_5.py is intact (NOT overwritten)
# 2. Base corex_gdn.py / corex_moe.py / corex_fa2.py exist
# 3. libcorex_gdn.so / libixformer.so exist
# 4. ix_moe_bridge.so compiles successfully
# 5. Serving layer patches are deployed
# 6. computility-run.yaml has correct max_model_len

set -euo pipefail

PASS=0
FAIL=0
WARN=0

check_pass() { echo "  ✅ PASS: $1"; PASS=$((PASS+1)); }
check_fail() { echo "  ❌ FAIL: $1"; FAIL=$((FAIL+1)); }
check_warn() { echo "  ⚠️  WARN: $1"; WARN=$((WARN+1)); }

echo "============================================"
echo "  DEPLOYMENT VERIFICATION — $(date)"
echo "============================================"

# --- Find vllm ---
VLLM=""
for P in /usr/local/corex/lib/python3/dist-packages/vllm \
         /usr/local/corex/lib64/python3/dist-packages/vllm; do
    [ -d "$P" ] && VLLM="$P" && break
done
if [ -z "$VLLM" ]; then
    check_fail "vllm not found in expected paths"
    exit 1
fi
echo "  vllm path: $VLLM"

echo ""
echo "=== 1. BASE IMAGE MODEL LAYER ==="
_QW="$VLLM/model_executor/models/qwen3_5.py"
if [ -f "$_QW" ]; then
    _SIZE=$(wc -c < "$_QW")
    _LINES=$(wc -l < "$_QW")
    if [ "$_SIZE" -gt 1000 ]; then
        check_pass "qwen3_5.py exists (${_SIZE} bytes, ${_LINES} lines) — base image intact"
    else
        check_fail "qwen3_5.py exists but tiny (${_SIZE} bytes) — may be overwritten"
    fi
else
    check_fail "qwen3_5.py NOT FOUND — model can't load"
fi

echo ""
echo "=== 2. COREX MODULES (base image) ==="
for m in corex_gdn.py corex_moe.py corex_fa2.py; do
    _F="$VLLM/model_executor/models/$m"
    if [ -f "$_F" ]; then
        check_pass "$m exists ($(wc -c < "$_F") bytes)"
    else
        check_warn "$m NOT in base — our version will be deployed"
    fi
done

echo ""
echo "=== 3. NATIVE .so LIBRARIES ==="
for so in /usr/local/corex/lib64/libcorex_gdn.so \
          /usr/local/corex/lib64/python3/dist-packages/ixformer/libixformer.so; do
    if [ -f "$so" ]; then
        check_pass "$(basename $so) exists at $so"
    else
        check_fail "$(basename $so) NOT FOUND at $so"
    fi
done

echo ""
echo "=== 4. IXFORMER PYTHON BINDING ==="
python3 -c "
import ixformer.functions as ixf_F
# Check critical functions
critical = ['silu_and_mul', 'rms_norm', 'fused_add_rms_norm', 'flash_attn',
            'vllm_single_query_cached_kv_attention_v2']
missing_critical = [f for f in critical if not hasattr(ixf_F, f)]
if missing_critical:
    print(f'FAIL: missing critical ixformer functions: {missing_critical}')
    exit(1)
else:
    print(f'PASS: all critical ixformer functions present')

# Check topk_softmax (expected to be missing from Python binding)
if hasattr(ixf_F, 'vllm_moe_topk_softmax'):
    print('NOTE: vllm_moe_topk_softmax IS in ixf_F (unexpected but good)')
else:
    print('NOTE: vllm_moe_topk_softmax NOT in ixf_F (expected — corex_moe.py bypasses this)')
" 2>&1 && check_pass "ixformer Python binding OK" || check_fail "ixformer Python binding broken"

echo ""
echo "=== 5. IX_MOE_BRIDGE COMPILE TEST ==="
if [ -f /workspace/ex_engine/csrc/ix_moe_bridge.cpp ]; then
    python3 /workspace/ex_engine/precompile_ix_bridge.py 2>&1
    if ls /workspace/ex_engine/build/ix_moe_bridge*.so 1>/dev/null 2>&1; then
        check_pass "ix_moe_bridge.so compiled: $(ls /workspace/ex_engine/build/ix_moe_bridge*.so)"
        # Verify symbols
        python3 -c "
import importlib.util, glob
so = glob.glob('/workspace/ex_engine/build/ix_moe_bridge*.so')[0]
spec = importlib.util.spec_from_file_location('ix', so)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
fns = [x for x in dir(m) if not x.startswith('_')]
print(f'Functions: {fns}')
required = ['topk_softmax', 'moe_gen_idx', 'moe_group_gemm', 'silu_and_mul', 'moe_combine_result']
missing = [f for f in required if f not in fns]
if missing:
    print(f'FAIL: missing functions: {missing}')
    exit(1)
print('PASS: all MoE pipeline functions present')
" 2>&1 && check_pass "ix_moe_bridge symbols verified" || check_warn "ix_moe_bridge symbol check failed"
    else
        check_warn "ix_moe_bridge.so compile failed (OK if base has corex_moe.py)"
    fi
else
    check_warn "ix_moe_bridge.cpp not found at /workspace/ex_engine/csrc/"
fi

echo ""
echo "=== 6. SERVING LAYER ==="
for f in protocol.py serving_chat.py api_server.py; do
    _F="$VLLM/entrypoints/openai/$f"
    if [ -f "$_F" ]; then
        check_pass "$f deployed"
    else
        check_fail "$f NOT deployed — serving broken"
    fi
done

# Check tool parser
_TP="$VLLM/entrypoints/openai/tool_parsers/qwen3coder_tool_parser.py"
if [ -f "$_TP" ]; then
    check_pass "qwen3_coder tool parser deployed"
else
    check_warn "qwen3_coder tool parser missing"
fi

echo ""
echo "=== 7. COMPUTILITY-RUN.YAML ==="
_YAML="/workspace/computility-run.yaml"
if [ -f "$_YAML" ]; then
    _MAX_LEN=$(grep -oP "max-model-len.*?'(\d+)'" "$_YAML" | grep -oP "\d+" | head -1)
    if [ "$_MAX_LEN" = "80000" ]; then
        check_pass "max_model_len=80000 (correct)"
    else
        check_fail "max_model_len=$_MAX_LEN (should be 80000)"
    fi
else
    check_fail "computility-run.yaml not found"
fi

echo ""
echo "=== 8. FUNCTIONAL SMOKE TEST ==="
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}, {torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB')
print(f'torch version: {torch.__version__}')
import vllm
print(f'vllm version: {vllm.__version__}')
" 2>&1 && check_pass "CUDA + vllm import OK" || check_fail "CUDA or vllm import failed"

echo ""
echo "============================================"
echo "  RESULTS: $PASS passed, $FAIL failed, $WARN warnings"
echo "============================================"

if [ $FAIL -gt 0 ]; then
    echo "  ⛔ DEPLOYMENT HAS FAILURES — DO NOT SUBMIT"
    exit 1
else
    echo "  ✅ DEPLOYMENT READY FOR SUBMISSION"
    exit 0
fi
