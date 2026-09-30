"""
analyze_trace.py - Automated chrome trace analyzer for BI-V100 decode profiling.

Reads a torch.profiler chrome trace JSON and produces a structured breakdown
of GPU kernel time, NCCL communication overhead, dtype cast waste, and
aten::item synchronization points.

Usage:
    python3 tools/analyze_trace.py <trace.json> [--format md|json|text]
"""
import argparse, json, sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

_CATEGORY_RULES: List[Tuple[str, str]] = [
    ("fenceflagwait", "NCCL:fenceWait"), ("fenceops", "NCCL:fenceOps"),
    ("nccl", "NCCL:other"), ("general_gemm", "GEMM"),
    ("direct_w13", "MoE:w13"), ("direct_w2_reduce", "MoE:w2_reduce"),
    ("topk_gating", "MoE:topk_softmax"),
    ("gdn_packed_decode", "GDN:packed_decode"),
    ("causal_conv", "GDN:causal_conv"), ("gated_rms_norm", "GDN:gated_rms_norm"),
    ("fused_add_rms_norm", "Norm:fused_add_rms"),
    ("fused_qknorm_rope", "Norm:qknorm_rope"),
    ("rms_norm_kernel", "Norm:rms_norm"),
    ("kernelcopy", "Copy:kernelCopy"),
    ("act_and_mul", "Activation:silu_and_mul"),
    ("silu", "Activation:silu"), ("sigmoid", "Activation:sigmoid"),
    ("cached_kv_attention", "PagedAttention"),
    ("reshape_and_cache", "KVCache:reshape"),
    ("reduce_kernel", "Reduce"), ("embedding", "Embedding"),
    ("arange", "Utility:arange"), ("cat", "Utility:cat"),
]

def categorize_kernel(name: str) -> str:
    nl = name.lower()
    for pat, cat in _CATEGORY_RULES:
        if pat in nl:
            return cat
    if "elementwise" in nl and "sum" in nl:
        return "Elementwise:sum(NCCL)"
    if "elementwise" in nl:
        return "Elementwise:other"
    return f"Other:{nl[:40]}"

def load_trace(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)

def analyze_metadata(data):
    dp = data.get("deviceProperties", [])
    di = data.get("distributedInfo", {})
    events = data.get("traceEvents", [])
    return {"devices": [{"id": d["id"], "name": d["name"], "sms": d["numSms"],
             "hbm_gb": d["totalGlobalMem"] // (1024**3)} for d in dp],
            "distributed": di, "total_events": len(events)}

def analyze_kernels(events):
    kernels = [e for e in events if e.get("cat") == "kernel"]
    by_cat = defaultdict(lambda: {"dur": 0, "count": 0, "kernels": defaultdict(lambda: {"dur": 0, "count": 0})})
    for e in kernels:
        name, dur = e.get("name", "?"), e.get("dur", 0)
        cat = categorize_kernel(name)
        top = cat.split(":")[0]
        by_cat[top]["dur"] += dur
        by_cat[top]["count"] += 1
        by_cat[top]["kernels"][name]["dur"] += dur
        by_cat[top]["kernels"][name]["count"] += 1
    total = sum(v["dur"] for v in by_cat.values())
    result = {"total_kernel_us": total, "categories": {}}
    for top, v in sorted(by_cat.items(), key=lambda x: -x[1]["dur"]):
        pct = v["dur"] / max(total, 1) * 100
        result["categories"][top] = {
            "time_us": v["dur"], "pct": round(pct, 1), "count": v["count"],
            "top_kernels": [{"name": k[:80], "time_us": kv["dur"], "count": kv["count"]}
                for k, kv in sorted(v["kernels"].items(), key=lambda x: -x[1]["dur"])[:5]]}
    return result

def analyze_nccl(events):
    kernels = [e for e in events if e.get("cat") == "kernel"]
    fw = [e for e in kernels if "fenceFlagWaitE" in e.get("name", "")]
    fo = [e for e in kernels if "fenceOps" in e.get("name", "")]
    fw_durs = sorted([e["dur"] for e in fw])
    n_fw = len(fw_durs)
    logical_ar = len(fo) // 6 if len(fo) >= 6 else len(fo)
    total_nccl = sum(e["dur"] for e in fw) + sum(e["dur"] for e in fo)
    per_ar_us = total_nccl / max(logical_ar, 1)
    return {"fence_wait": {"count": n_fw, "total_us": sum(e["dur"] for e in fw),
            "p50": fw_durs[n_fw//2] if n_fw else 0,
            "p99": fw_durs[int(n_fw*0.99)] if n_fw else 0,
            "max": max(fw_durs) if fw_durs else 0},
            "fence_ops": {"count": len(fo), "total_us": sum(e["dur"] for e in fo)},
            "logical_allreduce": logical_ar, "per_ar_cost_us": round(per_ar_us)}

def analyze_item_sync(events):
    items = sorted([e for e in events if e.get("cat")=="cpu_op" and e.get("name")=="aten::item"], key=lambda e: e["ts"])
    durs = [e["dur"] for e in items]
    intervals = [(items[i+1]["ts"]-items[i]["ts"])/1000 for i in range(len(items)-1)] if len(items)>1 else []
    return {"count": len(items), "total_us": sum(durs),
            "mean_us": round(sum(durs)/max(len(durs),1)),
            "mean_interval_ms": round(sum(intervals)/max(len(intervals),1),1) if intervals else 0,
            "source": "block_table.max().item() -> PagedAttention dispatch"}

def analyze_dtype_casts(events):
    ops = [e for e in events if e.get("cat")=="cpu_op" and e.get("name") in ("aten::to","aten::_to_copy")]
    return {"count": len(ops), "total_us": sum(e["dur"] for e in ops)}

def analyze_memcpy(events):
    mc = [e for e in events if e.get("cat") == "gpu_memcpy"]
    by_type = defaultdict(lambda: {"count": 0, "dur": 0, "bytes": 0})
    for e in mc:
        t = e.get("name", "unknown")
        by_type[t]["count"] += 1
        by_type[t]["dur"] += e.get("dur", 0)
        by_type[t]["bytes"] += e.get("args", {}).get("bytes", 0)
    return dict(by_type)

def analyze_gpu_utilization(events):
    kernels = sorted([e for e in events if e.get("cat")=="kernel"], key=lambda e: e["ts"])
    mc = [e for e in events if e.get("cat") == "gpu_memcpy"]
    if not kernels:
        return {"utilization_pct": 0}
    gpu_busy = sum(e["dur"] for e in kernels) + sum(e.get("dur",0) for e in mc)
    span = (kernels[-1]["ts"]+kernels[-1]["dur"]) - kernels[0]["ts"]
    gaps = []
    for i in range(len(kernels)-1):
        end = kernels[i]["ts"]+kernels[i]["dur"]
        start_next = kernels[i+1]["ts"]
        if start_next > end:
            gaps.append(start_next-end)
    return {"busy_us": gpu_busy, "span_us": span,
            "utilization_pct": round(gpu_busy/max(span,1)*100, 1),
            "total_gap_us": sum(gaps), "gap_count": len(gaps),
            "large_gaps_gt50us": len([g for g in gaps if g > 50])}

def compute_roadmap(nccl, item, dtype, kernel):
    total = kernel["total_kernel_us"]
    nccl_total = nccl["fence_wait"]["total_us"] + nccl["fence_ops"]["total_us"]
    return [
        {"priority": "P0", "target": "Fused GEMM+all-reduce",
         "current_us": nccl_total, "current_pct": round(nccl_total/max(total,1)*100,1),
         "estimated_saving_us": int(nccl["logical_allreduce"]*0.75*nccl["per_ar_cost_us"]),
         "approach": "Wire ix_full_bridge_fused_ar.so into RowParallelLinear"},
        {"priority": "P1", "target": "Eliminate dtype round-trips",
         "current_us": dtype["total_us"], "current_pct": round(dtype["total_us"]/max(total,1)*100,1),
         "estimated_saving_us": int(dtype["total_us"]*0.7),
         "approach": "Keep fp16 through NCCL; remove .float()/.to(dtype)"},
        {"priority": "P2", "target": "Eliminate aten::item sync",
         "current_us": item["total_us"], "current_pct": round(item["total_us"]/max(total,1)*100,1),
         "estimated_saving_us": int(item["total_us"]*0.9),
         "approach": "Cache max_seq_len across layers in decode step"},
    ]

def format_text(results):
    lines = []
    meta = results["metadata"]
    lines.append(f"Trace: {meta['total_events']} events, "
                 f"{len(meta['devices'])} GPUs ({meta['devices'][0]['name'] if meta['devices'] else '?'}), "
                 f"TP={meta['distributed'].get('world_size', '?')}")
    k = results["kernels"]
    lines.append(f"\nKernel time: {k['total_kernel_us']/1000:.1f} ms")
    for cat, v in k["categories"].items():
        lines.append(f"  {v['pct']:5.1f}%  {v['time_us']/1000:7.1f}ms  {v['count']:5d}x  {cat}")
    n = results["nccl"]
    lines.append(f"\nNCCL: {n['logical_allreduce']} logical ARs, {n['per_ar_cost_us']}us each")
    i = results["item_sync"]
    lines.append(f"aten::item: {i['count']}x, {i['total_us']/1000:.1f}ms, interval={i['mean_interval_ms']}ms")
    u = results["gpu_util"]
    lines.append(f"GPU util: {u['utilization_pct']}%")
    lines.append("\nOptimization Roadmap:")
    for r in results["roadmap"]:
        lines.append(f"  {r['priority']}: {r['target']} — save ~{r['estimated_saving_us']/1000:.1f}ms ({r['current_pct']}%)")
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Analyze BI-V100 decode trace")
    parser.add_argument("trace", help="Path to chrome trace JSON")
    parser.add_argument("--format", choices=["text","json","md"], default="text")
    args = parser.parse_args()
    data = load_trace(args.trace)
    events = data.get("traceEvents", [])
    results = {"metadata": analyze_metadata(data), "kernels": analyze_kernels(events),
               "nccl": analyze_nccl(events), "item_sync": analyze_item_sync(events),
               "dtype_casts": analyze_dtype_casts(events), "memcpy": analyze_memcpy(events),
               "gpu_util": analyze_gpu_utilization(events)}
    results["roadmap"] = compute_roadmap(results["nccl"], results["item_sync"], results["dtype_casts"], results["kernels"])
    if args.format == "json":
        print(json.dumps(results, indent=2))
    else:
        print(format_text(results))
    return 0

if __name__ == "__main__":
    sys.exit(main())
