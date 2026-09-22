# Competition Server Profile
**Captured**: 2026-08-01

## Hardware
- **GPU**: 4× Iluvatar BI-V100 32GB HBM each (128GB total)
  - Clock: SM 1500MHz / Mem 1200MHz
  - Driver: 3.2.1, COREX 10.2
  - Power: 250W TDP per card
- **CPU**: Intel Xeon Gold 6530
- **RAM**: 503GB DDR
- **Disk**: 3.5TB overlay, 100GB JuiceFS (public-storage)

## Software
- **OS**: Ubuntu 20.04.6 LTS, kernel 5.15.0-119
- **COREX**: 3.2.3 at `/usr/local/corex`
- **torch**: 2.1.0+corex.3.2.3
- **vllm**: 0.6.3+corex.3.2.3
- **transformers**: 4.51.3

## Model
- **Path**: `/root/public-storage/models/Qwen/Qwen3.6-35B-A3B/`
- **Name**: Qwen3.6-35B-A3B (MoE, 35B total, 3B active)
- **Note**: 4 cards × 32GB = 128GB total, model fits

## Key Paths
- `/root/llm-infer/` — benchmark scripts, README
- `/root/public-storage/models/Qwen/` — model weights
- `/root/apps/llm-modelzoo/benchmark/vllm/` — benchmark tools
- `/share/fshare/common/models/` — shared model storage (NFS)

## Benchmark Tools
- `benchmark_server_v0.5.0.py` — automated server benchmark
  - Sweeps: max-num-seqs=[128,256] × num-prompts=[1,128] × input=[128,1024] × output=[128,1024]
- `benchmark_server_v0.5.0.sh` — launches vllm server + benchmark client
  - Sets `NCCL_FORCESYNC_DISABLE=1`
  - Auto-cleanup of vllm processes
- `benchmark_serving_tokens.py` — online serving benchmark client

## Scoring Formula
`Output TPS × 16.796 + Input TPS × 2.799 + Cache TPS × 0.56`
- Threshold: ≥ 8000 weighted score
- Output TPS weight: 83% of total score
