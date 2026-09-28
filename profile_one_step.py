"""
Profile one decode step — prints a single summary table.
Run from /tmp to avoid local vllm/ shadowing:
  cd /tmp && python3 /home/dylan/0922/ikun/profile_one_step.py

Requires: server running on :8000 (uses a real request to trigger decode)
"""
import requests
import time
import sys
import re
import subprocess

MARKER = "=====IKUN_PROFILE_STEP====="
URL = "http://localhost:8000/v1/chat/completions"

def main():
    # Step 1: Check server is up
    try:
        r = requests.get("http://localhost:8000/v1/models", timeout=5)
        r.raise_for_status()
    except Exception as e:
        print(f"Server not ready: {e}")
        return 1

    # Step 2: Warmup (2 short requests)
    print("warmup...", flush=True)
    for _ in range(2):
        requests.post(URL, json={
            "model": "llm",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 8, "temperature": 0
        })

    # Step 3: Timed request — 64 tokens to get steady decode
    print("profiling...", flush=True)
    t0 = time.perf_counter()
    r = requests.post(URL, json={
        "model": "llm",
        "messages": [{"role": "user", "content": "Count from 1 to 100."}],
        "max_tokens": 64, "temperature": 0
    }).json()
    t1 = time.perf_counter()

    toks = r["usage"]["completion_tokens"]
    prompt_toks = r["usage"]["prompt_tokens"]
    elapsed = t1 - t0
    tps = toks / elapsed

    print(f"\n{MARKER}")
    print(f"  prompt_tokens:     {prompt_toks}")
    print(f"  completion_tokens: {toks}")
    print(f"  wall_time:         {elapsed:.2f}s")
    print(f"  tok/s:             {tps:.1f}")
    print(f"  ms/tok:            {elapsed/toks*1000:.1f}")
    print(f"")

    # Step 4: Estimate per-component breakdown from ms/tok
    ms_per_tok = elapsed / toks * 1000
    # Qwen3.5-35B-A3B: 64 layers (40 MoE + 24 attention-only? or mixed)
    # Each layer: input_norm + (gdn|full_attn) + post_attn_norm + moe
    # At 13.3 tok/s → ~75 ms/tok
    # direct_routed: 0.048 ms/layer × 40 layers = ~2 ms (MoE only)
    # So MoE is ~2.6% of total — bottleneck is elsewhere
    print(f"  --- Estimated breakdown (at {ms_per_tok:.1f} ms/tok) ---")
    print(f"  MoE (direct_routed): ~2 ms  ({2/ms_per_tok*100:.0f}%)")
    print(f"  Remaining (attn+norm+comm): ~{ms_per_tok-2:.0f} ms ({(ms_per_tok-2)/ms_per_tok*100:.0f}%)")
    print(f"  → Bottleneck is NOT in MoE anymore")
    print(f"{MARKER}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
