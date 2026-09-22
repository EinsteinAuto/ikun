#!/usr/bin/env bash
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_SRC="/root/public-storage/models/Qwen/Qwen3.6-35B-A3B"
MODEL_DST="/tmp/model"

# All verify steps that import vllm must NOT run from the project
# directory, because the local vllm/ folder shadows site-packages.
# We cd to /tmp for those steps and use absolute paths back to PROJ_DIR.

echo "============================================"
echo "  STEP 0: Prepare model weights"
echo "============================================"
if [ ! -d "$MODEL_DST" ]; then
    echo "  Copying model from $MODEL_SRC to $MODEL_DST ..."
    mkdir -p "$MODEL_DST"
    cp -r "$MODEL_SRC"/* "$MODEL_DST"/
    echo "  Done. $(du -sh $MODEL_DST | cut -f1)"
else
    echo "  Model already at $MODEL_DST ($(du -sh $MODEL_DST | cut -f1))"
fi
echo ""

echo "============================================"
echo "  STEP 1: Deploy via patch_ops.sh"
echo "============================================"
cd "$PROJ_DIR/qwen3_6_scripts"
chmod +x patch_ops.sh
bash patch_ops.sh 2>&1 | tee /tmp/patch_ops_pr1.log
echo "[deploy] exit code: $?"
echo ""

echo "============================================"
echo "  STEP 2: Verify deployment integrity"
echo "============================================"
# verify_deployment.sh imports vllm in its smoke test, run from /tmp
cd /tmp
bash "$PROJ_DIR/verify_deployment.sh"
echo ""

echo "============================================"
echo "  STEP 3: Verify .so loading + forward pass"
echo "============================================"
# verify_forward.py imports vllm modules, run from /tmp
cd /tmp
python3 "$PROJ_DIR/verify_forward.py"
echo ""

echo "============================================"
echo "  STEP 4: Verify paged_attn.py (our main change)"
echo "============================================"
cd /tmp
VLLM_ROOT=$(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))")
echo "  vllm root: $VLLM_ROOT"

python3 << PYEOF
import ast
path = "${VLLM_ROOT}/attention/ops/paged_attn.py"
with open(path) as f:
    source = f.read()
ast.parse(source)
lines = source.split("\n")
print(f"  Parsed {len(lines)} lines OK")

assert "[decode_pytorch ERROR]" not in source, "Site 1 FAIL"
print("  Site 1 OK: _forward_decode_pytorch catch-log-raise removed")

assert "cache_write_sync_failed" not in source, "Site 2 FAIL"
print("  Site 2 OK: write_kv_cache try/except removed")

assert 'error_stage="candidate-execution"' not in source, "Site 3 FAIL"
print("  Site 3 OK: fused prefill shadow try/except removed")

assert 'error_stage="reference-execution"' not in source, "Site 4 FAIL"
print("  Site 4 OK: reference shadow try/except removed")

assert "shadow record stays pending" in source, "Missing orphan comment"
print("  Shadow orphan comment present")
print("  ALL 4 AST SITES VERIFIED")
PYEOF
echo ""

echo "[4b] Run paged_attn GPU decode test..."
cd /tmp
python3 "$PROJ_DIR/verify_paged_attn.py"
echo ""

echo "============================================"
echo "  STEP 5: Verify executor.py (our second change)"
echo "============================================"
cd /tmp
python3 << PYEOF
import sys, os
sys.path.insert(0, "${PROJ_DIR}")
import torch, torch.nn as nn
from python.model_executor.executor import ModelExecutor

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Linear(16, 16).cuda()
    def forward(self, ids, pos):
        return self.p(torch.randn(ids.shape[0], 16, device="cuda"))

model = DummyModel()
meta = type("M", (), {"dp_token_counts": None, "dp_is_decode": None})()

ex_eager = ModelExecutor(model, {"python_graph_backend": "off"}, 64)
ex_eager.bind_kv_caches([])
out = ex_eager.execute(
    torch.randint(0, 100, (8,), device="cuda"),
    torch.arange(8, device="cuda"), meta)
print(f"  [A] Eager: shape={out.shape} OK")

ex_graph = ModelExecutor(model, {
    "python_graph_backend": "cudagraphs", "dp_size": 1,
    "max_position_embeddings": 4096
}, 64)
ex_graph.bind_kv_caches([])
out = ex_graph.execute(
    torch.randint(0, 100, (8,), device="cuda"),
    torch.arange(8, device="cuda"), meta)
print(f"  [B] Graph dispatch: shape={out.shape} OK")

try:
    ex_graph.execute(
        torch.randint(0, 100, (256,), device="cuda"),
        torch.arange(256, device="cuda"), meta)
    print("  [C] BUG: should have raised RuntimeError!")
    sys.exit(1)
except RuntimeError as e:
    msg = str(e)
    assert "DecodeCudaGraphRunner" in msg, f"Missing backend name: {msg}"
    assert "Implicit eager fallback" in msg
    print(f"  [C] RuntimeError correctly raised")

print("  ALL EXECUTOR TESTS PASSED")
PYEOF
echo ""

echo "============================================"
echo "  STEP 6: Full serving smoke test"
echo "============================================"
cd /tmp
echo "  Starting vllm server..."
CUDA_VISIBLE_DEVICES=0 VLLM_ENGINE_ITERATION_TIMEOUT_S=600 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DST" \
    --port 18888 \
    --served-model-name llm \
    --max-model-len 4096 \
    --trust-remote-code \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 1 \
    --enforce-eager \
    --disable-log-requests &
SERVER_PID=$!
echo "  Server PID: $SERVER_PID"

READY=0
for i in $(seq 1 180); do
    if curl -s http://localhost:18888/health >/dev/null 2>&1; then
        echo "  Server ready after ${i}s"
        READY=1
        break
    fi
    sleep 1
done

if [ $READY -eq 0 ]; then
    echo "  TIMEOUT: server did not start in 180s"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

echo "  Sending test request..."
RESPONSE=$(curl -s http://localhost:18888/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "llm",
        "messages": [{"role": "user", "content": "Say hello in one sentence."}],
        "max_tokens": 32,
        "temperature": 0.1
    }')

echo "$RESPONSE" | python3 -c "
import json, sys
d = json.load(sys.stdin)
if 'choices' in d and len(d['choices']) > 0:
    text = d['choices'][0]['message']['content']
    print(f'  Generated: {text}')
    print(f'  Usage: {d.get(\"usage\", {})}')
    print('  SERVING TEST PASSED')
else:
    print(f'  ERROR: {d}')
    sys.exit(1)
"

echo "  Stopping server..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null || true
echo ""
echo "============================================"
echo "  ALL 6 STEPS COMPLETED"
echo "============================================"
