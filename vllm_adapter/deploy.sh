#!/bin/bash
set -e

echo "=== Deploying Qwen3.5 vllm adapter ==="

VLLM_MODELS_DIR="/usr/local/corex/lib64/python3/dist-packages/vllm/model_executor/models"

# 1. Copy qwen3_5.py to vllm models directory
cp -v /root/project_6/vllm_adapter/qwen3_5.py "${VLLM_MODELS_DIR}/qwen3_5.py"
echo "✓ qwen3_5.py installed"

# 2. Verify registry already has the entry (it does from our earlier discovery)
python3 -c "
from vllm.model_executor.models.registry import _TEXT_GENERATION_MODELS
assert 'Qwen3_5MoeForCausalLM' in _TEXT_GENERATION_MODELS, 'Registry entry missing!'
mod, cls = _TEXT_GENERATION_MODELS['Qwen3_5MoeForCausalLM']
print(f'✓ Registry: Qwen3_5MoeForCausalLM -> ({mod}, {cls})')
"

# 3. Quick import test
python3 -c "
from vllm.model_executor.models.qwen3_5 import Qwen3_5MoeForCausalLM
print(f'✓ Import OK: {Qwen3_5MoeForCausalLM}')
"

# 4. Test config loading
python3 -c "
from transformers import AutoConfig
c = AutoConfig.from_pretrained('/root/public-storage/models/Qwen/Qwen3.6-35B-A3B', trust_remote_code=True)
print(f'✓ Config OK: {c.model_type}, experts={c.text_config.num_experts}')
"

echo ""
echo "=== Deployment complete. Starting vllm server... ==="
echo ""

# 5. Launch vllm server
export NCCL_FORCESYNC_DISABLE=1
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m vllm.entrypoints.openai.api_server \
    --model /root/public-storage/models/Qwen/Qwen3.6-35B-A3B \
    --gpu-memory-utilization 0.90 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 64 \
    --host 127.0.0.1 \
    --port 12345 \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --max-model-len 2048 \
    --dtype float16 \
    --disable-log-requests
