#!/bin/bash
# verify_docker_build.sh — 在真机上模拟Dockerfile build的每一步
# 用法: cd /home/dylan/project_6 && bash verify_docker_build.sh
#
# 真机已经是docker容器（和竞赛平台镜像一样），所以直接在/workspace执行

set -x  # 打印每条命令
WORKSPACE="/tmp/build_test_$$"
mkdir -p "$WORKSPACE"

echo "========================================="
echo "  模拟 Docker Build — $(date)"
echo "========================================="

# --- 模拟 COPY ---
echo "[STEP 0] Simulating COPY..."
SRCDIR="$(cd "$(dirname "$0")" && pwd)"
cp -r "$SRCDIR/qwen3_6_scripts" "$WORKSPACE/qwen3_6_scripts"
cp "$SRCDIR/computility-run.yaml" "$WORKSPACE/computility-run.yaml"
cp -r "$SRCDIR/ex_engine" "$WORKSPACE/ex_engine"
cp -r "$SRCDIR/vllm_overrides" "$WORKSPACE/vllm_overrides" 2>/dev/null || echo "[WARN] vllm_overrides not found"

cd "$WORKSPACE"

# --- Step 1: Build EX Engine ---
echo ""
echo "[STEP 1] Build EX Engine .so libraries..."
chmod +x ./ex_engine/build.sh
bash ./ex_engine/build.sh --corex 2>&1 | tail -20
echo "[STEP 1] exit code: $?"

# --- Step 2: Precompile MoE CUDA kernels ---
echo ""
echo "[STEP 2] Precompile MoE topk..."
python3 ./ex_engine/precompile_moe_topk.py 2>&1 | tail -10
echo "[STEP 2] exit code: $?"

# --- Step 3: Precompile vllm v0.5.5 MoE kernels ---
echo ""
echo "[STEP 3] Precompile MoE v055..."
python3 ./ex_engine/precompile_moe_kernels.py 2>&1 | tail -10
echo "[STEP 3] exit code: $?"

# --- Step 4: Stage vendor_overrides ---
echo ""
echo "[STEP 4] Stage vendor_overrides..."
mkdir -p ./qwen3_6_scripts/vendor_overrides/vllm/core/block
mkdir -p ./qwen3_6_scripts/vendor_overrides/vllm/model_executor/layers
for src in \
    "vllm_overrides/core/evictor_v2.py:vendor_overrides/vllm/core/evictor_v2.py" \
    "vllm_overrides/core/block/cpu_kv_content_cache.py:vendor_overrides/vllm/core/block/cpu_kv_content_cache.py" \
    "vllm_overrides/core/block/cpu_gpu_block_allocator.py:vendor_overrides/vllm/core/block/cpu_gpu_block_allocator.py" \
    "vllm_overrides/core/block/prefix_caching_block.py:vendor_overrides/vllm/core/block/prefix_caching_block.py" \
    "vllm_overrides/core/block/block_table.py:vendor_overrides/vllm/core/block/block_table.py" \
    "vllm_overrides/core/block_manager_v2.py:vendor_overrides/vllm/core/block_manager_v2.py" \
    "vllm_overrides/sampling_params.py:vendor_overrides/vllm/sampling_params.py" \
    "vllm_overrides/model_executor/sampling_metadata.py:vendor_overrides/vllm/model_executor/sampling_metadata.py" \
    "vllm_overrides/model_executor/layers/sampler.py:vendor_overrides/vllm/model_executor/layers/sampler.py"; do
    SRC="${src%%:*}"
    DST="./qwen3_6_scripts/${src##*:}"
    cp "./$SRC" "$DST" 2>/dev/null && echo "  ✓ $SRC" || echo "  ✗ $SRC (missing)"
done
echo "[STEP 4] exit code: $?"

# --- Step 5: Deploy patches ---
echo ""
echo "[STEP 5] patch_ops.sh..."
chmod +x ./qwen3_6_scripts/patch_ops.sh
cd ./qwen3_6_scripts
bash ./patch_ops.sh 2>&1 | tail -40
PATCH_EXIT=$?
echo "[STEP 5] exit code: $PATCH_EXIT"
cd "$WORKSPACE"

# --- Step 6: Build ix_unified_bridge ---
echo ""
echo "[STEP 6] Build ix_unified_bridge..."
if [ -f ./ex_engine/build_unified_bridge.sh ]; then
    chmod +x ./ex_engine/build_unified_bridge.sh
    bash ./ex_engine/build_unified_bridge.sh 2>&1 | tail -10
    echo "[STEP 6] exit code: $?"
else
    echo "[STEP 6] SKIP: build_unified_bridge.sh not found"
fi

# --- Step 7: Deploy ex_engine Python modules ---
echo ""
echo "[STEP 7] Deploy ex_engine Python modules..."
VLLM_ROOT=$(python3 -c "import vllm; print(vllm.__path__[0])" 2>/dev/null | tail -1 || echo "/usr/local/corex/lib64/python3/dist-packages/vllm")
for f in ix_unified.py corex_so_loader.py moe_fused_dispatch.py; do
    cp "./ex_engine/python/$f" "${VLLM_ROOT}/$f" 2>/dev/null && echo "  ✓ $f" || echo "  ✗ $f"
done
ls ./ex_engine/build/ix_unified_bridge*.so 2>/dev/null && \
    cp ./ex_engine/build/ix_unified_bridge*.so "${VLLM_ROOT}/" 2>/dev/null || true
echo "[STEP 7] exit code: $?"

# --- Step 8: Precompile GDN kernel ---
echo ""
echo "[STEP 8] Precompile GDN kernel..."
python3 ./qwen3_6_scripts/precompile_gdn.py \
    ./qwen3_6_scripts/flash_qla_sm70 2>&1 | tail -10
echo "[STEP 8] exit code: $?"

# --- 验证结果 ---
echo ""
echo "========================================="
echo "  Build Verification Summary"
echo "========================================="
echo "VLLM_ROOT: $VLLM_ROOT"

echo ""
echo "=== Deployed .so files ==="
ls -la "${VLLM_ROOT}"/corex_*.so 2>/dev/null | wc -l
ls "${VLLM_ROOT}"/corex_*.so 2>/dev/null

echo ""
echo "=== Key deployed .py files ==="
for f in qwen3_5.py _custom_ops.py bi100_env.py paged_attn.py serving_chat.py protocol.py; do
    p=$(find "$VLLM_ROOT" -name "$f" -type f 2>/dev/null | head -1)
    [ -n "$p" ] && echo "  ✓ $f ($(wc -c < "$p") bytes)" || echo "  ✗ $f NOT FOUND"
done

echo ""
echo "=== Quick import test ==="
python3 << 'PY'
import sys
errors = []
try:
    import vllm
    print(f"✓ vllm {vllm.__version__}")
except Exception as e:
    errors.append(f"✗ vllm: {e}")

try:
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM
    print("✓ Qwen3_5ForCausalLM importable")
except Exception as e:
    errors.append(f"✗ Qwen3_5ForCausalLM: {e}")

# Check corex .so imports
for mod in ['corex_gdn_causal_conv', 'corex_gdn_packed_decode', 'corex_moe_direct_routed',
            'corex_paged_kv_gather', 'corex_fused_paged_prefill']:
    try:
        m = __import__(f'vllm.{mod}', fromlist=[mod])
        funcs = [x for x in dir(m) if not x.startswith('_')]
        print(f"  ✓ {mod}: {funcs}")
    except Exception as e:
        errors.append(f"  ✗ {mod}: {e}")

for e in errors:
    print(e)
if not errors:
    print("\n✓ ALL IMPORTS OK")
else:
    print(f"\n✗ {len(errors)} FAILURES")
PY

# Cleanup
echo ""
echo "Cleaning up $WORKSPACE..."
rm -rf "$WORKSPACE"
echo "DONE"
