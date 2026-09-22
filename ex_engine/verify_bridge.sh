#!/usr/bin/env bash
# verify_bridge.sh — 验证 prebuilt ix_full_bridge.so 并决定是否重编
#
# 在真机上跑: bash ex_engine/verify_bridge.sh
#
# 验证步骤:
#   1. nm -D 检查 prebuilt ix_full_bridge.so 的导出符号
#   2. 对比 v1 (5函数) vs v2 (13函数) 的期望
#   3. 检查 MoE 符号是否缺失
#   4. 如果缺失，用 build_moe_bridge.sh 重编
#   5. 验证新编译的 .so 符号是否完整

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# =========================================================================
# Step 1: 找到 prebuilt .so
# =========================================================================
echo "========================================="
echo "[verify] Step 1: 定位 prebuilt ix_full_bridge.so"
echo "========================================="

PREBUILT=""
for p in \
    "${REPO_ROOT}/qwen3_6_scripts/prebuilt/corex-3.2.3-ivcore10/ix_full_bridge.so" \
    "${SCRIPT_DIR}/prebuilt/ix_full_bridge.so" \
    "${SCRIPT_DIR}/prebuilt/ix_full_bridge_v2.so" \
    "${SCRIPT_DIR}/prebuilt/ix_moe_bridge.so"; do
    if [[ -f "$p" ]]; then
        PREBUILT="$p"
        echo "[verify] 找到: $p ($(stat -c%s "$p" 2>/dev/null || stat -f%z "$p") bytes)"
        break
    fi
done

if [[ -z "$PREBUILT" ]]; then
    echo "[verify] ⚠ 没找到任何 prebuilt .so"
    echo "[verify] 直接跳到 Step 4 重编"
    NEED_REBUILD=1
else
    NEED_REBUILD=0
fi

# =========================================================================
# Step 2: nm -D 检查导出符号
# =========================================================================
if [[ "$NEED_REBUILD" -eq 0 ]]; then
    echo ""
    echo "========================================="
    echo "[verify] Step 2: nm -D 检查导出符号"
    echo "========================================="

    echo "[verify] 所有 T (text) 符号:"
    nm -D "$PREBUILT" 2>/dev/null | grep " T " | while read -r line; do
        # c++filt demangle
        sym=$(echo "$line" | awk '{print $3}')
        demangled=$(echo "$sym" | c++filt 2>/dev/null || echo "$sym")
        echo "  $demangled"
    done

    echo ""
    echo "[verify] 检查 v1 函数 (5个 base ops):"
    V1_FUNCS=("silu_and_mul" "rms_norm" "fused_add_rms_norm" "rotary_embedding" "reshape_and_cache")
    V1_COUNT=0
    for func in "${V1_FUNCS[@]}"; do
        if nm -D "$PREBUILT" 2>/dev/null | grep -q "$func"; then
            echo "  ✓ $func"
            ((V1_COUNT++)) || true
        else
            echo "  ✗ $func MISSING"
        fi
    done

    echo ""
    echo "[verify] 检查 v2 新增函数 (8个 MoE ops):"
    V2_FUNCS=("paged_attention" "topk_softmax" "moe_gen_idx" "moe_expand_input" "group_gemm" "moe_combine_result" "fused_moe_forward" "ix_linear")
    V2_COUNT=0
    for func in "${V2_FUNCS[@]}"; do
        if nm -D "$PREBUILT" 2>/dev/null | grep -q "$func"; then
            echo "  ✓ $func"
            ((V2_COUNT++)) || true
        else
            echo "  ✗ $func MISSING"
        fi
    done

    echo ""
    echo "[verify] 结果: v1=${V1_COUNT}/5, v2_new=${V2_COUNT}/8"

    if [[ "$V2_COUNT" -ge 6 ]]; then
        echo "[verify] ✓ 这个 .so 是 v2 编的，MoE 函数完整"
        NEED_REBUILD=0
    elif [[ "$V1_COUNT" -ge 3 ]]; then
        echo "[verify] ⚠ 这个 .so 是 v1 编的（或中间版本），缺少 MoE 函数"
        NEED_REBUILD=1
    else
        echo "[verify] ✗ 这个 .so 符号异常，需要重编"
        NEED_REBUILD=1
    fi
fi

# =========================================================================
# Step 3: 检查源文件是否就绪
# =========================================================================
echo ""
echo "========================================="
echo "[verify] Step 3: 检查编译源文件"
echo "========================================="

MOE_CU=""
BRIDGE_CPP=""
for base in "${SCRIPT_DIR}" "${SCRIPT_DIR}/ex_engine"; do
    [[ -f "${base}/csrc/moe_ops_impl.cu" ]] && MOE_CU="${base}/csrc/moe_ops_impl.cu"
    [[ -f "${base}/csrc/ix_full_bridge_v2.cpp" ]] && BRIDGE_CPP="${base}/csrc/ix_full_bridge_v2.cpp"
done

echo "[verify] moe_ops_impl.cu: ${MOE_CU:-NOT FOUND} $([ -n "$MOE_CU" ] && wc -l < "$MOE_CU" || echo 0) lines"
echo "[verify] ix_full_bridge_v2.cpp: ${BRIDGE_CPP:-NOT FOUND} $([ -n "$BRIDGE_CPP" ] && wc -l < "$BRIDGE_CPP" || echo 0) lines"

# 检查v2里的pybind导出数量
if [[ -n "$BRIDGE_CPP" ]]; then
    MDEF_COUNT=$(grep -c 'm.def(' "$BRIDGE_CPP" || true)
    echo "[verify] v2 m.def() 数量: ${MDEF_COUNT} (期望13)"
fi

# 检查moe_ops_impl里的5个函数
if [[ -n "$MOE_CU" ]]; then
    echo "[verify] moe_ops_impl.cu 实现的函数:"
    grep -E "^void |^torch::Tensor " "$MOE_CU" | while read -r line; do
        echo "  → $line"
    done
fi

# 检查编译工具链
echo ""
echo "[verify] 编译环境:"
COREX_ROOT="${COREX_ROOT:-/usr/local/corex}"
echo "  COREX_ROOT: ${COREX_ROOT}"
echo "  clang++: $(command -v clang++ 2>/dev/null || echo 'NOT FOUND') $(${COREX_ROOT}/bin/clang++ --version 2>/dev/null | head -1 || echo '')"
echo "  python3: $(python3 --version 2>/dev/null || echo 'NOT FOUND')"
echo "  torch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'NOT FOUND')"
echo "  ixformer: $(python3 -c 'import ixformer; print(ixformer.__version__)' 2>/dev/null || echo 'NOT FOUND')"

# libcuinfer.so
CUINFER=""
for d in "${COREX_ROOT}/lib64" "${COREX_ROOT}/lib" "/usr/lib64" "/usr/lib"; do
    if [[ -f "${d}/libcuinfer.so" ]]; then
        CUINFER="${d}/libcuinfer.so"
        break
    fi
done
echo "  libcuinfer.so: ${CUINFER:-NOT FOUND}"

# ixformer .so
IX_DIR=""
IX_SO_COUNT=0
for d in \
    "${COREX_ROOT}/lib/python3/dist-packages/ixformer" \
    "${COREX_ROOT}/lib64/python3/dist-packages/ixformer" \
    "$(python3 -c 'import ixformer, os; print(os.path.dirname(ixformer.__file__))' 2>/dev/null || echo '')"; do
    if [[ -d "$d" ]]; then
        IX_DIR="$d"
        IX_SO_COUNT=$(find "$d" -name "*.so" -type f 2>/dev/null | wc -l)
        break
    fi
done
echo "  ixformer dir: ${IX_DIR:-NOT FOUND} (${IX_SO_COUNT} .so files)"

# _ixformer_torch.so — 关键: v2 bridge链接的对象
IX_TORCH=""
if [[ -n "$IX_DIR" ]]; then
    IX_TORCH=$(find "$IX_DIR" -name "_ixformer_torch*" -type f 2>/dev/null | head -1)
fi
echo "  _ixformer_torch.so: ${IX_TORCH:-NOT FOUND}"
if [[ -n "$IX_TORCH" ]]; then
    echo "  _ixformer_torch.so 导出 (v2需要的7个):"
    for sym in silu_and_mul_forward rms_norm_forward fused_add_rms_norm_forward \
               ixformer_linear vllm_rotary_embedding_neox \
               vllm_cache_ops_reshape_and_cache vllm_single_query_cached_kv; do
        if nm -D "$IX_TORCH" 2>/dev/null | grep -q "$sym"; then
            echo "    ✓ $sym"
        else
            echo "    ✗ $sym MISSING"
        fi
    done
fi

# =========================================================================
# Step 4: 重编（如果需要）
# =========================================================================
if [[ "$NEED_REBUILD" -eq 1 ]]; then
    echo ""
    echo "========================================="
    echo "[verify] Step 4: 需要重编 — 调用 build_moe_bridge.sh"
    echo "========================================="

    if [[ -z "$MOE_CU" ]] || [[ -z "$BRIDGE_CPP" ]]; then
        echo "[verify] ✗ 源文件缺失，无法编译"
        exit 1
    fi

    BUILD_SCRIPT="${SCRIPT_DIR}/build_moe_bridge.sh"
    if [[ -f "$BUILD_SCRIPT" ]]; then
        echo "[verify] 执行: bash ${BUILD_SCRIPT}"
        bash "$BUILD_SCRIPT"
        echo ""
    else
        echo "[verify] build_moe_bridge.sh 不存在，尝试用 build_ix_bridge.sh"
        ALT_SCRIPT="${SCRIPT_DIR}/build_ix_bridge.sh"
        if [[ -f "$ALT_SCRIPT" ]]; then
            echo "[verify] 执行: bash ${ALT_SCRIPT}"
            bash "$ALT_SCRIPT"
        else
            echo "[verify] ✗ 没有可用的编译脚本"
            exit 1
        fi
    fi
else
    echo ""
    echo "========================================="
    echo "[verify] Step 4: 跳过 — .so 已经是 v2"
    echo "========================================="
fi

# =========================================================================
# Step 5: 验证编译结果
# =========================================================================
echo ""
echo "========================================="
echo "[verify] Step 5: 验证最终 .so"
echo "========================================="

# 找新编译的 .so
FINAL_SO=""
for p in \
    "${SCRIPT_DIR}/prebuilt/ix_moe_bridge.so" \
    "${SCRIPT_DIR}/prebuilt/ix_full_bridge_v2.so" \
    "$PREBUILT"; do
    if [[ -f "$p" ]]; then
        FINAL_SO="$p"
        break
    fi
done

if [[ -z "$FINAL_SO" ]]; then
    echo "[verify] ✗ 找不到最终 .so"
    exit 1
fi

echo "[verify] 验证: $FINAL_SO"

# Python import 测试
python3 << PYTEST
import sys, os, ctypes, importlib

so_path = "${FINAL_SO}"
print(f"[verify] Loading: {so_path}")

# 方法1: ctypes 检查符号
try:
    lib = ctypes.CDLL(so_path)
    print("[verify] ✓ ctypes.CDLL 加载成功")
except Exception as e:
    print(f"[verify] ✗ ctypes.CDLL 失败: {e}")

# 方法2: importlib (pybind11 module)
try:
    so_dir = os.path.dirname(so_path)
    so_name = os.path.splitext(os.path.basename(so_path))[0]
    sys.path.insert(0, so_dir)
    mod = importlib.import_module(so_name)
    funcs = [f for f in dir(mod) if not f.startswith('_')]
    print(f"[verify] ✓ import {so_name} 成功，导出 {len(funcs)} 个函数:")
    for f in funcs:
        print(f"    → {f}")

    # 验证关键函数
    expected = ['silu_and_mul', 'rms_norm', 'topk_softmax',
                'group_gemm', 'moe_combine_result', 'fused_moe_forward']
    missing = [f for f in expected if f not in funcs]
    if missing:
        print(f"[verify] ⚠ 缺少: {missing}")
    else:
        print(f"[verify] ✓ 所有关键函数都在")
except Exception as e:
    print(f"[verify] ✗ import 失败: {e}")
PYTEST

echo ""
echo "========================================="
echo "[verify] 完成"
echo "========================================="
