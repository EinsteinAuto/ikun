# ixformer overlap comm integration — target 30 tok/s

## current state

4x BI-V100 TP=4 Qwen3 35B decode: **p50=12.5 tok/s** (80ms per token)

trace breakdown per decode step:
```
fence (NCCL sync)     18.6ms   32.5%   ← target
GEMM                  10.2ms   17.7%
memcpy DtoD            7.3ms   12.8%
dtype cast             5.9ms   10.3%
GDN decode             2.9ms    5.1%
MoE compute            2.6ms    4.5%
other                  9.7ms   17.1%
total CUDA            57.2ms
CPU dispatch overhead 22.8ms
wall time             80.0ms
```

fence cannot be eliminated — InfiniCCL on BI-V100 use NCCL backend internally (session 5 confirmed). but fence can be **hidden** by overlapping compute with communication

## what ixformer overlap do

ixformer SDK split the GEMM output into chunks, launch allreduce on chunk[0] while computing chunk[1]. the fence wait for chunk[0] happen while GPU is busy with chunk[1] compute. net effect: fence latency hidden behind compute

available modules in `ixformer_sdk/inference/overlap/`:

| module | what it fuse | where it apply |
|--------|-------------|----------------|
| `linear_mlp_overlap_comm` | GEMM + allreduce, split into N chunks | attention o_proj, GDN out_proj |
| `group_gemm_moe_reduce_sum_allreduce_overlap` | MoE FC2 group_gemm + reduce_sum + allreduce | MoE combine (40 layers, biggest target) |
| `fmha_oproj_allreduce_ln_gating_overlap` | flash_attn + o_proj + allreduce + layernorm + gating | full attention layer (10 layers) |
| `moe_reduce_allreduce_ln_linear_overlap` | MoE reduce + allreduce + layernorm + next linear | MoE-to-next-layer transition |
| `llama_decoder_layer_overlap` | full decoder layer overlap protocol | reference implementation for Llama |

## target: 30 tok/s (33.3ms per token)

need to cut 46.7ms from 80ms. three things stack:

1. **overlap fence behind compute**: hide 18.6ms fence. realistic saving ~12ms (not 100% overlap, some fence still exposed at chunk boundaries)
2. **remove --enforce-eager**: cut 22.8ms CPU dispatch to ~3ms with CUDA graph. net saving ~20ms
3. **dtype cast elimination**: cut 5.9ms to ~1ms by keeping fp16 through allreduce path

projected: 80 - 12 - 20 - 5 = 43ms → ~23 tok/s. with batch_size increase from 2 to 4: amortize remaining fence → ~30 tok/s

## key finding: decode vs prefill

`GemmAllReduceSplitOverlapComm.is_supported` requires `m >= 512` for chunked overlap. during decode, m = batch_size (1-2 tokens) so chunked overlap never fires. overlap is only effective during prefill

for decode (the throughput bottleneck), three remaining paths to 30 tok/s:

1. `ixfd.all_reduce(async_op=True)` — does not chunk, but puts allreduce on comm stream so it can overlap with next op on compute stream
2. remove `--enforce-eager` — CUDA graph eliminates 22.8ms CPU dispatch overhead
3. increase batch_size from 2 to 4-8 — amortize 18.6ms fence across more tokens

## integration plan

### phase 1: ixformer native_forward for GEMM+AR (implemented)

for attention o_proj and GDN out_proj, replace bridge.linear + infiniccl_allreduce with `GemmAllReduceSplitOverlapComm.native_forward` which use `ixff.linear` (ixformer optimized GEMM) + `ixfd.all_reduce(async_op=True)` in one call. saves Python dispatch overhead and use optimized GEMM kernel

gated by: `BI100_OVERLAP_COMM=1` (off by default)
call sites: `_fused_linear_ar` in vendor_overrides/qwen3_5.py

when m >= 512 (prefill): uses chunked overlap (4 chunks, compute overlaps with comm)
when m < 512 (decode): uses native_forward (fused GEMM+AR, async comm stream)

### phase 2: MoE async allreduce (implemented)

replace `tensor_model_parallel_all_reduce(out)` with `ixfd.all_reduce(out, async_op=True, use_comm_stream=True)` so MoE allreduce runs on comm stream and overlaps with start of next layer residual add / layernorm

gated by: same `BI100_OVERLAP_COMM=1`

### phase 3: CUDA graph (remove --enforce-eager)

SYSTEM_DESIGN.md line 173 set enforce_eager=true for BI-V100 compatibility. test which custom kernels break graph capture. 22.8ms CPU overhead → ~3ms is the single biggest win available

### phase 4: batch_size tuning

increase --max-num-seqs from 2 to 4-8. GEMM and comm overhead amortized across more tokens. combined with phase 3 this is the path to 30

### phase 5: full layer overlap (prefill)

use `fmha_oproj_allreduce_ln_gating_overlap` for end-to-end layer fusion during prefill. decode already handled by phase 1-2

## risk

- `group_gemm_moe_reduce_sum_allreduce` non-overlap fallback use `F.moe_w8a8_group_gemm` (INT8 only). cannot use this function directly for fp16 MoE. but the overlap mechanism (`SplitOverlapComm` / `GemmAllReduceSplitOverlapComm`) is dtype-agnostic — split chunks, overlap compute stream with comm stream. plan: use `GemmAllReduceSplitOverlapComm` directly for fp16 GEMM+allreduce overlap, skip the W8A8 group_gemm wrapper
- `SplitOverlapComm` use `ixfd._distributed` which may conflict with vllm distributed init
- overlap chunk count tuning: too few chunks = no overlap benefit, too many = launch overhead. need to benchmark N=2,4,8
- multi-card hang risk same as before, need infiniccl safety net

## dependency

stacked on PR #10 (infiniccl safety + fused kernel baseline)
