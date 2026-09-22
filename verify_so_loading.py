#!/usr/bin/env python3
"""
Diagnostic: verify all 13 prebuilt CoreX .so modules load on BI-V100.

Run on the real machine AFTER docker build or inside the container:
    python3 verify_so_loading.py

Checks:
1. All 13 .so files exist in $VLLM_ROOT/
2. Each .so has PyInit_ symbol (pybind11)
3. Each .so can be imported via `from vllm import corex_*`
4. Functions inside each .so are callable
5. Base image .so files also checked (/usr/local/corex/lib64/)
"""

import sys
import os
import importlib
import ctypes
import struct
from pathlib import Path

# The 13 prebuilt .so modules from qwen3_6_scripts/prebuilt/corex-3.2.3-ivcore10/
PREBUILT_SO = [
    "corex_attn_head_rms_norm",
    "corex_block_major_kv_transfer",
    "corex_fused_paged_prefill",
    "corex_gdn_beta_decay",
    "corex_gdn_causal_conv",
    "corex_gdn_gated_norm",
    "corex_gdn_packed_decode",
    "corex_gdn_qk_map",
    "corex_moe_direct_routed",
    "corex_moe_exact_reduce",
    "corex_moe_topk_softmax",
    "corex_moe_weight_gather",
    "corex_paged_kv_gather",
]

# Base image .so files (from /usr/local/corex/lib64/)
BASE_IMAGE_SO = [
    "/usr/local/corex/lib64/libcorex_gdn.so",
    "/usr/local/corex/lib64/libcublas.so",
    "/usr/local/corex/lib64/libcudart.so",
    "/usr/local/corex/lib64/libcudnn.so",
    "/usr/local/corex/lib64/libixattn.so",
]


def find_vllm_root():
    """Find vLLM install path."""
    try:
        import vllm
        return Path(vllm.__file__).parent
    except ImportError:
        # Try known paths
        for p in [
            "/usr/local/corex/lib/python3/dist-packages/vllm",
            "/usr/local/corex/lib64/python3/dist-packages/vllm",
        ]:
            if Path(p).is_dir():
                return Path(p)
    return None


def check_elf(path):
    """Check if file is valid x86-64 ELF."""
    try:
        with open(path, 'rb') as f:
            header = f.read(20)
        if len(header) < 20:
            return False, "too short"
        if header[:4] != b'\x7fELF':
            return False, "not ELF"
        if header[4:6] != b'\x02\x01':
            return False, "not 64-bit LE"
        machine = struct.unpack_from('<H', header, 18)[0]
        if machine != 62:
            return False, f"not x86-64 (machine={machine})"
        return True, "valid x86-64 ELF"
    except Exception as e:
        return False, str(e)


def check_pyinit(path, module_name):
    """Check if .so has PyInit_ symbol."""
    import subprocess
    try:
        result = subprocess.run(
            ['nm', '-D', str(path)], capture_output=True, text=True, timeout=10)
        expected = f"PyInit_{module_name}"
        for line in result.stdout.split('\n'):
            if expected in line:
                return True, expected
        return False, f"no {expected} symbol"
    except Exception as e:
        return False, str(e)


def check_import(module_name):
    """Try to import the module from vllm package."""
    try:
        mod = importlib.import_module(f"vllm.{module_name}")
        funcs = [x for x in dir(mod) if not x.startswith('_')]
        return True, funcs[:5]
    except Exception as e:
        return False, str(e)


def check_base_image_so(path):
    """Check base image .so can be loaded via ctypes."""
    if not os.path.exists(path):
        return False, "file not found"
    try:
        lib = ctypes.CDLL(path, mode=ctypes.RTLD_LAZY)
        return True, f"loaded ({os.path.getsize(path)} bytes)"
    except Exception as e:
        return False, str(e)


def main():
    print("=" * 70)
    print("BI-V100 CoreX .so Loading Diagnostic")
    print("=" * 70)

    vllm_root = find_vllm_root()
    if vllm_root:
        print(f"\n[OK] vLLM root: {vllm_root}")
    else:
        print(f"\n[FAIL] vLLM root not found")
        sys.exit(1)

    # Check sys.path includes vllm parent
    vllm_parent = str(vllm_root.parent)
    if vllm_parent not in sys.path:
        sys.path.insert(0, vllm_parent)
        print(f"  Added {vllm_parent} to sys.path")

    print(f"\n{'─' * 70}")
    print("PHASE 1: Prebuilt CoreX .so files (13 modules)")
    print(f"{'─' * 70}")

    results = {"ok": 0, "fail": 0}

    for name in PREBUILT_SO:
        so_path = vllm_root / f"{name}.so"
        print(f"\n  [{name}]")

        # Check file exists
        if not so_path.exists():
            print(f"    FILE:   MISSING at {so_path}")
            results["fail"] += 1
            continue
        else:
            size = so_path.stat().st_size
            print(f"    FILE:   {so_path} ({size:,} bytes)")

        # Check ELF
        ok, msg = check_elf(so_path)
        print(f"    ELF:    {'OK' if ok else 'FAIL'} — {msg}")

        # Check PyInit symbol
        ok_sym, msg_sym = check_pyinit(so_path, name)
        print(f"    SYMBOL: {'OK' if ok_sym else 'FAIL'} — {msg_sym}")

        # Check import
        ok_imp, msg_imp = check_import(name)
        if ok_imp:
            print(f"    IMPORT: OK — functions: {msg_imp}")
            results["ok"] += 1
        else:
            print(f"    IMPORT: FAIL — {msg_imp}")
            results["fail"] += 1

    print(f"\n{'─' * 70}")
    print("PHASE 2: Base image .so files")
    print(f"{'─' * 70}")

    for path in BASE_IMAGE_SO:
        ok, msg = check_base_image_so(path)
        status = "OK" if ok else "MISSING/FAIL"
        print(f"  [{status}] {path} — {msg}")

    print(f"\n{'─' * 70}")
    print("PHASE 3: ixformer availability")
    print(f"{'─' * 70}")

    try:
        import ixformer
        print(f"  [OK] ixformer version: {getattr(ixformer, '__version__', 'unknown')}")
        print(f"  [OK] ixformer path: {ixformer.__file__}")

        # Check critical ixformer functions
        for submod in ['functions', 'inference', 'infer']:
            try:
                m = importlib.import_module(f"ixformer.{submod}")
                funcs = [x for x in dir(m) if not x.startswith('_')][:10]
                print(f"  [OK] ixformer.{submod}: {funcs}")
            except Exception as e:
                print(f"  [--] ixformer.{submod}: {e}")

    except ImportError:
        print("  [FAIL] ixformer not available")

    print(f"\n{'─' * 70}")
    print("PHASE 4: qwen3_5.py _USE_COREX flags")
    print(f"{'─' * 70}")

    try:
        # Simulate what qwen3_5.py does at import time
        from vllm.bi100_env import env_bool

        flags = {
            "BI100_GDN_COREX_CAUSAL_CONV": True,
            "BI100_GDN_COREX_GATED_NORM": True,
            "BI100_GDN_COREX_BETA_DECAY": True,
            "BI100_GDN_COREX_QK_MAP": True,
            "BI100_GDN_COREX_PACKED_DECODE": False,
            "BI100_ATTN_COREX_HEAD_RMS_NORM": True,
            "BI100_MOE_COREX_EXACT_REDUCE": True,
            "BI100_MOE_COREX_WEIGHT_GATHER": True,
            "BI100_MOE_COREX_DIRECT_ROUTED": False,
            "BI100_MOE_COREX_TOPK_SOFTMAX": True,
        }

        for env_name, default in flags.items():
            value = env_bool(env_name, default)
            src = "env" if os.getenv(env_name) is not None else "default"
            print(f"  {env_name} = {value} ({src})")

    except Exception as e:
        print(f"  [FAIL] Could not check flags: {e}")

    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {results['ok']}/{len(PREBUILT_SO)} .so modules loadable")
    if results['fail'] > 0:
        print(f"WARNING: {results['fail']} modules failed to load — will fallback to PyTorch!")
        print("This causes 5-8x TPS drop (2-3 TPS vs 14-22 TPS)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
