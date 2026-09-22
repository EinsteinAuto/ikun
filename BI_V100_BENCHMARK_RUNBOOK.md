# BI-V100 Benchmark Runbook

在 Phanthy Cloud 实机上执行。目标：拿到实测数据，替换所有 `ns*0.5, l2w*0.6` 猜测值。

## 环境确认

```bash
# 已确认：CUDA 10.2, 4×BI-V100, corex 运行时
# Python: /usr/local/corex/lib64/python3/dist-packages 里有 torch + vllm

# 先确认 torch 可用
python3 -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"

# 确认 SMEM 到底是 48KB 还是 32KB（hardware.cuh 和 _custom_ops.py 有矛盾）
python3 -c "
import torch
props = torch.cuda.get_device_properties(0)
print(f'sharedMemPerBlock:       {props.total_memory}')  # 总显存
# PyTorch 不直接暴露 SMEM，用 CUDA runtime 查
"

# 用这个方法精确测 SMEM
python3 -c "
import torch, torch.utils.cpp_extension
# 如果 cpp_extension 可用，编译一个查 SMEM 的 kernel
# 否则用下面的方法推断
import ctypes
try:
    cuda = ctypes.CDLL('libcuda.so')
    # cudaDeviceGetAttribute
    val = ctypes.c_int(0)
    # attribute 48 = CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
    cuda.cuDeviceGetAttribute(ctypes.byref(val), 48, 0)
    print(f'SMEM per block: {val.value} bytes ({val.value/1024:.0f} KB)')
except:
    print('libcuda not accessible, try ixsmi or corex API')
"
```

## Phase 0: 硬件探测（5 分钟）

这是最关键的一步——确认 SMEM 到底是多少。

```bash
cd ~/project_6

# 探测脚本
python3 << 'PROBE'
import torch
import time

device = torch.device('cuda:0')
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"SM count: {torch.cuda.get_device_properties(0).multi_processor_count}")

# SMEM 探测：分配越来越大的 shared memory 直到失败
# 用一个简单的 kernel 测试实际可用 SMEM
print("\n--- SMEM probe via allocation ---")
for smem_kb in [32, 48, 64, 96]:
    smem_bytes = smem_kb * 1024
    try:
        # torch.zeros 不直接测 SMEM，用 tensor 大小间接推断
        # 真正的 SMEM 测试需要自定义 kernel
        pass
    except:
        pass

# 更直接的方法：torch.cuda.get_device_properties
props = torch.cuda.get_device_properties(0)
print(f"\ntorch.cuda properties:")
for attr in dir(props):
    if not attr.startswith('_'):
        try:
            val = getattr(props, attr)
            if isinstance(val, (int, float, str)):
                print(f"  {attr}: {val}")
        except:
            pass

# 测带宽
print("\n--- Memory bandwidth probe ---")
sizes = [2**20, 2**24, 2**28]  # 1MB, 16MB, 256MB
for n in sizes:
    x = torch.randn(n, device=device)
    y = torch.empty_like(x)
    
    torch.cuda.synchronize()
    warmup = 5
    repeats = 20
    for _ in range(warmup):
        y.copy_(x)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(repeats):
        y.copy_(x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    bytes_moved = n * 4 * 2 * repeats  # read + write, float32
    bw = bytes_moved / elapsed / 1e9
    print(f"  {n*4/1024/1024:>6.0f} MB: {bw:.0f} GB/s")

print("\nDone. Use SM count and BW to validate hardware.cuh values.")
PROBE
```

## Phase 1: Quick benchmark（~40 分钟总计）

按竞赛权重优先级跑：reduce (83% weight) → topk → scan → transform

```bash
cd ~/project_6

# 确保用 GPU 0（最空闲的）
export CUDA_VISIBLE_DEVICES=2

# --- reduce: 最高优先级，~2 分钟 ---
python3 muh/bench_bi100.py --algo reduce --dtype float32 --quick -o results/
python3 muh/bench_bi100.py --algo reduce --dtype float16 --quick -o results/
python3 muh/bench_bi100.py --algo reduce --dtype bfloat16 --quick -o results/

# --- topk: 采样热路径，<1 分钟 ---
python3 muh/bench_bi100.py --algo topk --dtype float32 --quick -o results/
python3 muh/bench_bi100.py --algo topk --dtype float16 --quick -o results/

# --- scan: prefix scan，~29 分钟 ---
# scan 的搜索空间最大，先跑 quick
python3 muh/bench_bi100.py --algo scan --dtype float32 --quick -o results/

# --- transform: 元素级操作，~8 分钟 ---
python3 muh/bench_bi100.py --algo transform --dtype float16 --quick -o results/
python3 muh/bench_bi100.py --algo transform --dtype bfloat16 --quick -o results/

echo "=== Quick benchmark complete ==="
ls -la results/
```

## Phase 2: 如果 SMEM 是 32KB（检查 Phase 0 结果后决定）

```bash
# 如果 Phase 0 确认 SMEM=32KB，重跑所有 benchmark
python3 muh/bench_bi100.py --algo reduce --dtype float32 --quick --smem-limit 32768 -o results_32k/
python3 muh/bench_bi100.py --algo topk --dtype float32 --quick --smem-limit 32768 -o results_32k/
python3 muh/bench_bi100.py --algo scan --dtype float32 --quick --smem-limit 32768 -o results_32k/
```

## Phase 3: 端到端验证

benchmark top-5 候选值在真实 vllm 推理中的效果。

```bash
cd ~/project_6

# 启动 vllm 服务（用竞赛配置）
python3 -m vllm.entrypoints.openai.api_server \
  --model /model \
  --served-model-name llm \
  --max-model-len 100000 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  -tp 4 \
  --max-num-seqs 8 \
  --disable-log-requests \
  --disable-frontend-multiprocessing \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --max-seq-len-to-capture 32768 \
  --num-scheduler-steps 8 \
  --preemption-mode recompute \
  --enable-prefix-caching &

# 等服务启动
sleep 120

# 测 output TPS（83% 权重）
python3 << 'E2E'
import requests, time, json

url = "http://localhost:80/v1/chat/completions"
headers = {"Content-Type": "application/json"}

# 短输入长输出 = 测 decode（Output TPS）
payload = {
    "model": "llm",
    "messages": [{"role": "user", "content": "请详细解释量子计算的基本原理，包括量子比特、量子门、量子纠缠和量子退相干。请尽可能详细。"}],
    "max_tokens": 2048,
    "temperature": 0.7,
    "stream": False
}

# Warmup
for _ in range(3):
    r = requests.post(url, headers=headers, json=payload, timeout=300)

# Timed
times = []
tokens = []
for i in range(5):
    start = time.perf_counter()
    r = requests.post(url, headers=headers, json=payload, timeout=300)
    elapsed = time.perf_counter() - start
    
    data = r.json()
    output_tokens = data["usage"]["completion_tokens"]
    tps = output_tokens / elapsed
    times.append(elapsed)
    tokens.append(output_tokens)
    print(f"  Run {i+1}: {output_tokens} tokens in {elapsed:.2f}s = {tps:.1f} tok/s")

avg_tps = sum(t/e for t,e in zip(tokens, times)) / len(times)
print(f"\nAvg Output TPS: {avg_tps:.1f}")
print(f"Weighted score contribution: {avg_tps * 16.796:.0f} (83% of total)")
E2E

# 停止 vllm
kill %1
```

## 结果回填

拿到 results/ 里的 JSON 后，回到 Claude 对话：

```
把 results/reduce_float32.json 的内容贴给我，
我会用实测 top-5 替换 tuning_reduce.cuh 的 bi100_* 值。
```

每个 JSON 里的 best point 直接映射到 C++ header 的 bi100_* struct：
- `ipt` → `items_per_thread` / `items`
- `tpb` → `threads_per_block` / `threads`
- `ipv` → `vec_size` / `items_per_vec_load`
