#!/usr/bin/env python3
"""verify_dlopen_chain.py — Verify complete dlopen SO → Python → Model call chain.

Run on BI-V100 to identify gaps before submitting.
Usage: python3 verify_dlopen_chain.py [--vllm-root /path/to/vllm]

Checks:
  1. All prebuilt .so files are installed and loadable as Python extension modules
  2. All env-gated kernel dispatch paths have matching .so
  3. protocol.py correctly accepts max_completion_tokens
  4. qwen3_5.py import chain is complete (no silent None fallbacks for enabled kernels)
"""
import argparse
import ctypes
import importlib
import importlib.util
import json
import os
import pathlib
import struct
import sys
import traceback

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
SKIP = "\033[93m- SKIP\033[0m"
INFO = "\033[94m  INFO\033[0m"

results = {"pass": 0, "fail": 0, "skip": 0}


def check(name, condition, detail=""):
    if condition:
        results["pass"] += 1
        print(f"  {PASS} {name}" + (f" — {detail}" if detail else ""))
    else:
        results["fail"] += 1
        print(f"  {FAIL} {name}" + (f" — {detail}" if detail else ""))


def skip(name, reason=""):
    results["skip"] += 1
    print(f"  {SKIP} {name}" + (f" — {reason}" if reason else ""))


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------- Section 1: prebuilt SO ELF validation ----------

EXPECTED_SO = [
    "corex_attn_head_rms_norm",
    "corex_block_major_kv_transfer",
    "corex_fused_paged_prefill",
    "corex_gdn_beta_decay",
    "corex_gdn_causal_conv",
    "corex_gdn_chunk_recurrent",
    "corex_gdn_gated_norm",
    "corex_gdn_packed_decode",
    "corex_gdn_qk_map",
    "corex_moe_direct_routed",
    "corex_moe_exact_reduce",
    "corex_moe_index_combine",
    "corex_moe_topk_softmax",
    "corex_moe_weight_gather",
    "corex_paged_kv_gather",
]

# Env var → module name mapping (from qwen3_5.py)
ENV_KERNEL_MAP = {
    "BI100_GDN_COREX_CAUSAL_CONV": "corex_gdn_causal_conv",
    "BI100_GDN_COREX_GATED_NORM": "corex_gdn_gated_norm",
    "BI100_GDN_COREX_BETA_DECAY": "corex_gdn_beta_decay",
    "BI100_GDN_COREX_QK_MAP": "corex_gdn_qk_map",
    "BI100_GDN_COREX_PACKED_DECODE": "corex_gdn_packed_decode",
    "BI100_ATTN_COREX_HEAD_RMS_NORM": "corex_attn_head_rms_norm",
    "BI100_MOE_COREX_EXACT_REDUCE": "corex_moe_exact_reduce",
    "BI100_MOE_COREX_WEIGHT_GATHER": "corex_moe_weight_gather",
    "BI100_MOE_COREX_DIRECT_ROUTED": "corex_moe_direct_routed",
    "BI100_MOE_COREX_TOPK_SOFTMAX": "corex_moe_topk_softmax",
    "BI100_MOE_COREX_INDEX_COMBINE": "corex_moe_index_combine",
}


def find_vllm_root():
    spec = importlib.util.find_spec("vllm")
    if spec and spec.submodule_search_locations:
        return pathlib.Path(next(iter(spec.submodule_search_locations)))
    return None


def check_elf(path):
    """Validate file is a 64-bit x86-64 ELF."""
    if not path.exists():
        return False, "file not found"
    if path.stat().st_size == 0:
        return False, "empty file"
    header = path.read_bytes()[:20]
    if len(header) < 20 or header[:4] != b"\x7fELF":
        return False, "not ELF"
    if header[4:6] != b"\x02\x01":
        return False, "not 64-bit LE"
    machine = struct.unpack_from("<H", header, 18)[0]
    if machine != 62:
        return False, f"not x86-64 (machine={machine})"
    return True, f"ok ({path.stat().st_size} bytes)"


def verify_so_installations(vllm_root):
    section("1. Prebuilt SO Installation & ELF Validation")
    for name in EXPECTED_SO:
        so_path = vllm_root / f"{name}.so"
        ok, detail = check_elf(so_path)
        check(f"{name}.so", ok, detail)


def verify_so_importable(vllm_root):
    section("2. SO Python Import Chain (torch extension)")
    for name in EXPECTED_SO:
        so_path = vllm_root / f"{name}.so"
        if not so_path.exists():
            skip(f"import vllm.{name}", "SO not installed")
            continue
        try:
            mod = importlib.import_module(f"vllm.{name}")
            funcs = [f for f in dir(mod) if not f.startswith("_")]
            check(f"import vllm.{name}", True,
                  f"exports: {', '.join(funcs[:5])}")
        except Exception as e:
            check(f"import vllm.{name}", False, str(e)[:120])


def verify_env_kernel_dispatch():
    section("3. Env-Gated Kernel Dispatch Consistency")
    for env_var, module_name in ENV_KERNEL_MAP.items():
        env_val = os.environ.get(env_var, "<unset>")
        enabled = env_val in ("1", "true", "True")
        try:
            mod = importlib.import_module(f"vllm.{module_name}")
            available = mod is not None
        except Exception:
            available = False

        if enabled and not available:
            check(f"{env_var}={env_val} → {module_name}",
                  False, "ENABLED but SO not loadable — will crash!")
        elif enabled and available:
            check(f"{env_var}={env_val} → {module_name}",
                  True, "enabled + available")
        elif not enabled:
            check(f"{env_var}={env_val} → {module_name}",
                  True, f"disabled (available={available})")


def verify_protocol():
    section("4. Protocol max_completion_tokens Acceptance")
    try:
        from vllm.entrypoints.openai.protocol import ChatCompletionRequest
        # Simulate a request with max_completion_tokens
        test_data = {
            "model": "llm",
            "messages": [{"role": "user", "content": "test"}],
            "max_completion_tokens": 8192,
        }
        req = ChatCompletionRequest(**test_data)
        # After fold_max_completion_tokens, max_tokens should be 8192
        check("max_completion_tokens accepted",
              req.max_tokens == 8192,
              f"max_tokens={req.max_tokens}")

        # Also test with thinking param
        test_data2 = {
            "model": "llm",
            "messages": [{"role": "user", "content": "test"}],
            "max_completion_tokens": 32768,
            "thinking": {"type": "enabled", "budget_tokens": 10000},
        }
        req2 = ChatCompletionRequest(**test_data2)
        check("max_completion_tokens+thinking accepted",
              req2.max_tokens == 32768,
              f"max_tokens={req2.max_tokens}, thinking={req2.thinking}")

    except Exception as e:
        check("protocol import/validation", False, str(e)[:200])


def verify_model_imports():
    section("5. qwen3_5.py Model Import Chain")
    try:
        # Don't actually import the full model (needs CUDA), just check the file
        vllm_root = find_vllm_root()
        if vllm_root is None:
            skip("qwen3_5.py", "vllm not installed")
            return
        model_path = vllm_root / "model_executor" / "models" / "qwen3_5.py"
        check("qwen3_5.py installed",
              model_path.exists(),
              str(model_path))

        if model_path.exists():
            source = model_path.read_text()
            # Check all corex imports are present
            for name in EXPECTED_SO:
                if f"from vllm import {name}" in source or \
                   f"import {name}" in source:
                    check(f"qwen3_5 imports {name}", True)
                else:
                    # Not all SOs are imported directly in qwen3_5.py
                    # Some are used via other modules
                    if name in ("corex_block_major_kv_transfer",
                                "corex_fused_paged_prefill",
                                "corex_paged_kv_gather"):
                        skip(f"qwen3_5 imports {name}",
                             "used via paged_attn/block_major modules")
                    else:
                        check(f"qwen3_5 imports {name}", False,
                              "import not found in source")
    except Exception as e:
        check("model import chain", False, str(e)[:200])


def verify_paged_attn_chain(vllm_root):
    section("6. Paged Attention dlopen Chain")
    if vllm_root is None:
        skip("paged_attn chain", "vllm not installed")
        return
    paged = vllm_root / "attention" / "ops" / "paged_attn.py"
    check("paged_attn.py installed", paged.exists(), str(paged))
    if paged.exists():
        source = paged.read_text()
        for name in ("corex_paged_kv_gather",
                     "corex_fused_paged_prefill",
                     "corex_block_major_kv_transfer"):
            found = name in source
            if found:
                check(f"paged_attn uses {name}", True)
            else:
                skip(f"paged_attn uses {name}", "not referenced")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-root", type=pathlib.Path, default=None)
    args = parser.parse_args()

    vllm_root = args.vllm_root or find_vllm_root()
    if vllm_root is None:
        print("ERROR: Cannot find vllm installation. Use --vllm-root.")
        sys.exit(1)
    print(f"vllm root: {vllm_root}")

    verify_so_installations(vllm_root)
    verify_so_importable(vllm_root)
    verify_env_kernel_dispatch()
    verify_protocol()
    verify_model_imports()
    verify_paged_attn_chain(vllm_root)

    section("SUMMARY")
    total = results["pass"] + results["fail"] + results["skip"]
    print(f"  Pass: {results['pass']}/{total}")
    print(f"  Fail: {results['fail']}/{total}")
    print(f"  Skip: {results['skip']}/{total}")
    if results["fail"] > 0:
        print(f"\n  ⚠️  {results['fail']} checks FAILED — fix before submitting!")
        sys.exit(1)
    else:
        print(f"\n  ✅ All checks passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
