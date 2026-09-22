#!/bin/bash
# Run after deploy.sh has started the server and you see "Application startup complete"
# Execute in a SECOND terminal

echo "=== Waiting for server... ==="
for i in $(seq 1 60); do
    if curl -s --max-time 2 http://127.0.0.1:12345/v1/models > /dev/null 2>&1; then
        echo "✓ Server ready"
        break
    fi
    echo "  waiting... ($i/60)"
    sleep 5
done

echo ""
echo "=== Quick sanity check ==="
curl -s http://127.0.0.1:12345/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/root/public-storage/models/Qwen/Qwen3.6-35B-A3B",
        "prompt": "Hello, how are you?",
        "max_tokens": 32,
        "temperature": 0.0
    }' | python3 -m json.tool

echo ""
echo "=== Running benchmark ==="
cd ~/apps/llm-modelzoo/benchmark/vllm

python3 benchmark_serving_tokens.py \
    --model /root/public-storage/models/Qwen/Qwen3.6-35B-A3B \
    --host 127.0.0.1 --port 12345 \
    --num-prompts 32 \
    --input-tokens 128 \
    --output-tokens 128

echo ""
echo "=== Benchmark complete ==="
