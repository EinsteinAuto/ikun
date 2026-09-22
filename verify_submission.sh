#!/usr/bin/env bash
# verify_submission.sh — 竞赛提交前的完整验证
#
# 在真机上运行:
#   cd /home/dylan/project_6
#   bash verify_submission.sh
#
# 检查项:
#   1. Dockerfile语法
#   2. patch_ops.sh可执行 + 引用的文件全部存在
#   3. CCCL preload编译链完整(头文件+源文件+build脚本)
#   4. computility-run.yaml路径与build输出匹配
#   5. prebuilt .so文件完整性(SHA256)
#   6. qwen3_5.py imports的corex模块全部有对应.so或build脚本
#   7. Docker context大小(不要超过平台限制)
#   8. 单卡冒烟测试(如果有GPU)

PASS=0
FAIL=0
WARN=0

check() {
    local name=$1; shift
    if "$@" >/dev/null 2>&1; then
        echo "  ✓ $name"
        PASS=$((PASS+1))
    else
        echo "  ✗ $name"
        FAIL=$((FAIL+1))
    fi
}

warn() {
    echo "  △ $1"
    WARN=$((WARN+1))
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "=== 1. 文件结构 ==="
check "Dockerfile存在" test -f Dockerfile
check "computility-run.yaml存在" test -f computility-run.yaml
check "patch_ops.sh存在" test -f qwen3_6_scripts/patch_ops.sh
check "patch_ops.sh可执行" test -x qwen3_6_scripts/patch_ops.sh
check ".dockerignore存在" test -f .dockerignore

echo ""
echo "=== 2. CCCL Preload编译链 ==="
check "build脚本存在" test -f qwen3_6_scripts/cccl_preload/build_cccl_preload.sh
check "源文件存在" test -f qwen3_6_scripts/cccl_preload/cccl_allocator_preload.cu
check "CCCL头文件目录存在" test -d qwen3_6_scripts/cccl_preload/include/cub
check "libcudacxx头文件存在" test -d qwen3_6_scripts/cccl_preload/include/cuda

CCCL_HEADERS=$(find qwen3_6_scripts/cccl_preload/include -type f 2>/dev/null | wc -l)
if [[ "$CCCL_HEADERS" -ge 280 ]]; then
    echo "  ✓ CCCL头文件数量: $CCCL_HEADERS (≥280)"
    PASS=$((PASS+1))
else
    echo "  ✗ CCCL头文件数量: $CCCL_HEADERS (期望≥280)"
    FAIL=$((FAIL+1))
fi

# 检查关键头文件
check "cub/util_allocator.cuh" test -f qwen3_6_scripts/cccl_preload/include/cub/util_allocator.cuh
check "cub/config.cuh" test -f qwen3_6_scripts/cccl_preload/include/cub/config.cuh
check "cuda/__cccl_config" test -f qwen3_6_scripts/cccl_preload/include/cuda/__cccl_config
check "nv/target" test -f qwen3_6_scripts/cccl_preload/include/nv/target

echo ""
echo "=== 3. patch_ops.sh → CCCL preload 调用链 ==="
if grep -q "cccl_preload/build_cccl_preload.sh" qwen3_6_scripts/patch_ops.sh; then
    echo "  ✓ patch_ops.sh调用新版build脚本"
    PASS=$((PASS+1))
else
    echo "  ✗ patch_ops.sh未调用cccl_preload/build_cccl_preload.sh"
    FAIL=$((FAIL+1))
fi

# 检查旧文件是否残留
if [[ -f qwen3_6_scripts/cccl_preload_allocator.cu ]]; then
    echo "  ✗ 旧mock文件残留: cccl_preload_allocator.cu"
    FAIL=$((FAIL+1))
else
    echo "  ✓ 旧mock文件已清理"
    PASS=$((PASS+1))
fi

echo ""
echo "=== 4. computility-run.yaml 路径匹配 ==="
YAML_PRELOAD=$(grep -A1 "LD_PRELOAD" computility-run.yaml | grep "value:" | awk '{print $2}')
if [[ -n "$YAML_PRELOAD" ]]; then
    echo "  ✓ LD_PRELOAD已配置: $YAML_PRELOAD"
    PASS=$((PASS+1))
    # 检查路径与build输出一致
    BUILD_OUTPUT_DIR=$(grep "build_cccl_preload.sh" qwen3_6_scripts/patch_ops.sh | grep -oP '/workspace/\S+' | head -1 | tr -d ';')
    EXPECTED_SO="${BUILD_OUTPUT_DIR}/libcccl_allocator.so"
    if [[ "$YAML_PRELOAD" == "$EXPECTED_SO" || "$YAML_PRELOAD" == "/workspace/qwen3_6_scripts/libcccl_allocator.so" ]]; then
        echo "  ✓ 路径匹配build输出"
        PASS=$((PASS+1))
    else
        echo "  ✗ 路径不匹配: yaml=$YAML_PRELOAD expected=$EXPECTED_SO"
        FAIL=$((FAIL+1))
    fi
else
    echo "  ✗ computility-run.yaml缺少LD_PRELOAD"
    FAIL=$((FAIL+1))
fi

# 检查CCCL_ALLOC_DISABLE
if grep -q "CCCL_ALLOC_DISABLE" computility-run.yaml; then
    echo "  ✓ CCCL_ALLOC_DISABLE可控"
    PASS=$((PASS+1))
fi

echo ""
echo "=== 5. Prebuilt .so 完整性 ==="
PREBUILT_DIR="qwen3_6_scripts/prebuilt/corex-3.2.3-ivcore10"
if [[ -f "${PREBUILT_DIR}/SHA256SUMS" ]]; then
    EXPECTED=$(wc -l < "${PREBUILT_DIR}/SHA256SUMS")
    ACTUAL=$(ls "${PREBUILT_DIR}"/*.so 2>/dev/null | wc -l)
    if [[ "$ACTUAL" -ge "$EXPECTED" ]]; then
        echo "  ✓ prebuilt .so数量: $ACTUAL (manifest expects $EXPECTED)"
        PASS=$((PASS+1))
    else
        echo "  ✗ prebuilt .so数量不足: $ACTUAL < $EXPECTED"
        FAIL=$((FAIL+1))
    fi
    # 如果有sha256sum工具, 验证checksum
    if command -v sha256sum &>/dev/null; then
        if (cd "$PREBUILT_DIR" && sha256sum --status --check SHA256SUMS 2>/dev/null); then
            echo "  ✓ SHA256校验通过"
            PASS=$((PASS+1))
        else
            echo "  ✗ SHA256校验失败"
            FAIL=$((FAIL+1))
        fi
    fi
else
    echo "  △ SHA256SUMS不存在, 跳过校验"
    WARN=$((WARN+1))
fi

echo ""
echo "=== 6. qwen3_5.py corex imports vs prebuilt ==="
IMPORTS=$(grep "from vllm import corex_" qwen3_6_scripts/qwen3_5.py 2>/dev/null | sed 's/.*import //' | sed 's/ as.*//' | sort -u)
for mod in $IMPORTS; do
    SO_FILE="${PREBUILT_DIR}/${mod}.so"
    BUILD_SCRIPT="qwen3_6_scripts/build_${mod}.sh"
    if [[ -f "$SO_FILE" ]]; then
        echo "  ✓ $mod → prebuilt .so"
        PASS=$((PASS+1))
    elif [[ -f "$BUILD_SCRIPT" ]]; then
        echo "  △ $mod → build脚本存在 (docker内编译)"
        WARN=$((WARN+1))
    else
        echo "  ✗ $mod → 无.so也无build脚本"
        FAIL=$((FAIL+1))
    fi
done

echo ""
echo "=== 7. Docker context大小 ==="
# 排除.git和大文件
if [[ -f .dockerignore ]]; then
    # 粗略估算
    CONTEXT_SIZE=$(du -sh --exclude='.git' --exclude='cccl_upstream' --exclude='upstream_ref' --exclude='*.zip' --exclude='vllm' --exclude='ixformer_sdk' qwen3_6_scripts/ computility-run.yaml Dockerfile 2>/dev/null | tail -1 | awk '{print $1}')
    echo "  核心文件大小: ~$CONTEXT_SIZE"
    echo "  (.dockerignore应排除cccl_upstream/, upstream_ref/, vllm/, *.zip等)"
fi

# 检查.dockerignore是否排除大目录
if [[ -f .dockerignore ]]; then
    for dir in cccl_upstream upstream_ref vllm ixformer_sdk "*.zip"; do
        if grep -q "$dir" .dockerignore 2>/dev/null; then
            echo "  ✓ .dockerignore排除: $dir"
        else
            warn ".dockerignore未排除: $dir (可能导致docker context过大)"
        fi
    done
fi

echo ""
echo "=== 8. 单卡冒烟测试 ==="
if command -v ixsmi &>/dev/null || command -v nvidia-smi &>/dev/null; then
    echo "  检测到GPU, 运行基础验证..."
    if python3 -c "import torch; assert torch.cuda.is_available(); print(f'  ✓ PyTorch CUDA: {torch.cuda.get_device_name(0)}')" 2>/dev/null; then
        PASS=$((PASS+1))
    else
        echo "  ✗ PyTorch CUDA不可用"
        FAIL=$((FAIL+1))
    fi

    # 测试CCCL preload编译
    if [[ -f qwen3_6_scripts/cccl_preload/build_cccl_preload.sh ]]; then
        echo "  尝试编译CCCL preload .so..."
        if bash qwen3_6_scripts/cccl_preload/build_cccl_preload.sh /tmp 2>/dev/null; then
            if [[ -s /tmp/libcccl_allocator.so ]]; then
                echo "  ✓ CCCL preload编译成功"
                PASS=$((PASS+1))
                # 测试加载
                if LD_PRELOAD=/tmp/libcccl_allocator.so python3 -c "import torch; x=torch.zeros(1024,device='cuda'); del x; print('  ✓ LD_PRELOAD加载正常')" 2>/dev/null; then
                    PASS=$((PASS+1))
                else
                    echo "  ✗ LD_PRELOAD加载失败"
                    FAIL=$((FAIL+1))
                fi
                rm -f /tmp/libcccl_allocator.so
            else
                echo "  ✗ 编译产物为空"
                FAIL=$((FAIL+1))
            fi
        else
            echo "  ✗ CCCL preload编译失败"
            FAIL=$((FAIL+1))
        fi
    fi
else
    echo "  (无GPU, 跳过)"
fi

echo ""
echo "==========================================="
echo "  PASS: $PASS   FAIL: $FAIL   WARN: $WARN"
echo "==========================================="

if [[ "$FAIL" -gt 0 ]]; then
    echo ""
    echo "⚠ 有 $FAIL 个检查失败, 提交前请修复!"
    exit 1
else
    echo ""
    echo "✓ 全部检查通过, 可以提交"
    exit 0
fi
