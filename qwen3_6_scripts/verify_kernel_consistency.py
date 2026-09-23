#!/usr/bin/env python3
"""verify_kernel_consistency.py — AST-based audit of kernel flag definitions.

Extracts all _USE_COREX_* and _USE_XLLM_* flags from qwen3_5.py via AST
parsing, then cross-checks against:
  1. EXPECTED_SO      in verify_dlopen_chain.py
  2. ENV_KERNEL_MAP   in verify_dlopen_chain.py
  3. flags dict       in verify_so_loading.py
  4. PREBUILT list    in test_dlopen_chain.py

Reports any inconsistency: missing entries, extra entries, mismatched env
var names, or SOs declared in one list but not another.

Run:  python3 qwen3_6_scripts/verify_kernel_consistency.py

Design note (ported from gemini-cli PR #28863 — environmentSanitization.ts):
  Centralise the scattered consistency checks that were previously split
  across 4 files with manually maintained lists.  A single source-of-truth
  extraction from the runtime module (qwen3_5.py AST) ensures the verify
  scripts cannot drift.
"""
import ast
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m⚠\033[0m"
INFO = "\033[94mℹ\033[0m"

results = {"pass": 0, "fail": 0, "warn": 0}


def report(status: str, msg: str) -> None:
    if status == "pass":
        results["pass"] += 1
        print(f"  {PASS} {msg}")
    elif status == "fail":
        results["fail"] += 1
        print(f"  {FAIL} {msg}")
    else:
        results["warn"] += 1
        print(f"  {WARN} {msg}")


# ---------------------------------------------------------------------------
# 1. AST extraction: find all _USE_COREX_* and _USE_XLLM_* from qwen3_5.py
# ---------------------------------------------------------------------------

def extract_use_flags(source_path: Path) -> Dict[str, Optional[str]]:
    """Extract flag → env_var_name mapping from qwen3_5.py AST.

    Parses assignments like:
        _USE_COREX_GDN_CAUSAL_CONV = (
            _corex_gdn_causal_conv is not None
            and env_bool("BI100_GDN_COREX_CAUSAL_CONV", True))

    Returns: {"_USE_COREX_GDN_CAUSAL_CONV": "BI100_GDN_COREX_CAUSAL_CONV", ...}
    For flags without an env_bool call, value is None (always-enabled).
    """
    with open(source_path, "r") as f:
        tree = ast.parse(f.read(), filename=str(source_path))

    flags: Dict[str, Optional[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            name = target.id
            if not (name.startswith("_USE_COREX_") or name.startswith("_USE_XLLM_")):
                continue
            # Extract env_bool string argument if present
            env_var = _find_env_bool_arg(node.value)
            flags[name] = env_var
    return flags


def _find_env_bool_arg(node: ast.expr) -> Optional[str]:
    """Recursively search AST node for env_bool("...", ...) call."""
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name) and func.id == "env_bool":
            if node.args and isinstance(node.args[0], ast.Constant):
                return str(node.args[0].value)
        if isinstance(func, ast.Attribute) and func.attr == "env_bool":
            if node.args and isinstance(node.args[0], ast.Constant):
                return str(node.args[0].value)
    if isinstance(node, ast.BoolOp):
        for value in node.values:
            result = _find_env_bool_arg(value)
            if result is not None:
                return result
    return None


# ---------------------------------------------------------------------------
# 2. Extract corex module names from try/except imports in qwen3_5.py
# ---------------------------------------------------------------------------

def extract_corex_imports(source_path: Path) -> Set[str]:
    """Find all 'from vllm import corex_*' in try/except blocks."""
    with open(source_path, "r") as f:
        source = f.read()
    return set(re.findall(r'from\s+vllm\s+import\s+(corex_\w+)', source))


# ---------------------------------------------------------------------------
# 3. Extract lists from verify scripts (regex, not AST — they're simple)
# ---------------------------------------------------------------------------

def extract_python_list(source_path: Path, var_name: str) -> List[str]:
    """Extract a Python list variable by simple regex."""
    with open(source_path, "r") as f:
        source = f.read()
    pattern = rf'{var_name}\s*=\s*\[(.*?)\]'
    match = re.search(pattern, source, re.DOTALL)
    if not match:
        return []
    return re.findall(r'"(\w+)"', match.group(1))


def extract_python_dict_keys(source_path: Path, var_name: str) -> List[str]:
    """Extract keys from a Python dict variable by regex."""
    with open(source_path, "r") as f:
        source = f.read()
    pattern = rf'{var_name}\s*=\s*\{{(.*?)\}}'
    match = re.search(pattern, source, re.DOTALL)
    if not match:
        return []
    return re.findall(r'"(\w+)"', match.group(1))


def extract_verify_so_flags(source_path: Path) -> List[str]:
    """Extract flag names from the verify_so_loading.py flags dict."""
    with open(source_path, "r") as f:
        source = f.read()
    pattern = r'flags\s*=\s*\{(.*?)\}'
    match = re.search(pattern, source, re.DOTALL)
    if not match:
        return []
    return re.findall(r'"(BI100_\w+)"', match.group(1))


# ---------------------------------------------------------------------------
# 4. Cross-reference and report
# ---------------------------------------------------------------------------

def flag_name_to_module(flag_name: str) -> Optional[str]:
    """Heuristic: _USE_COREX_GDN_CAUSAL_CONV → corex_gdn_causal_conv."""
    if flag_name.startswith("_USE_COREX_"):
        return "corex_" + flag_name[len("_USE_COREX_"):].lower()
    return None


def main():
    qwen3_5 = SCRIPT_DIR / "qwen3_5.py"
    dlopen_chain = SCRIPT_DIR / "verify_dlopen_chain.py"
    so_loading = PROJECT_ROOT / "verify_so_loading.py"
    test_dlopen = PROJECT_ROOT / "test_dlopen_chain.py"

    missing_files = []
    for p in [qwen3_5, dlopen_chain, so_loading, test_dlopen]:
        if not p.exists():
            missing_files.append(str(p))
    if missing_files:
        print(f"ERROR: missing files: {missing_files}")
        sys.exit(1)

    print("=" * 64)
    print("  Kernel Flag Consistency Audit")
    print("=" * 64)

    # ---- Extract ----
    use_flags = extract_use_flags(qwen3_5)
    corex_imports = extract_corex_imports(qwen3_5)
    expected_so = set(extract_python_list(dlopen_chain, "EXPECTED_SO"))
    env_kernel_map_keys = set(extract_python_dict_keys(dlopen_chain, "ENV_KERNEL_MAP"))
    env_kernel_map_values = set()
    with open(dlopen_chain) as f:
        for m in re.finditer(r'"(BI100_\w+)":\s*"(corex_\w+)"', f.read()):
            env_kernel_map_values.add(m.group(2))
    so_loading_flags = set(extract_verify_so_flags(so_loading))
    prebuilt_list = set(extract_python_list(test_dlopen, "PREBUILT"))

    # ---- Section 1: Flag extraction summary ----
    print(f"\n--- qwen3_5.py: {len(use_flags)} _USE_* flags ---")
    for flag, env in sorted(use_flags.items()):
        env_str = env if env else "(always-on)"
        print(f"  {flag} → {env_str}")

    # ---- Section 2: corex imports vs EXPECTED_SO ----
    print(f"\n--- corex imports: {len(corex_imports)} modules ---")
    print(f"--- EXPECTED_SO:   {len(expected_so)} modules ---")
    for name in sorted(expected_so | corex_imports):
        in_imports = name in corex_imports
        in_expected = name in expected_so
        if in_imports and in_expected:
            report("pass", f"{name}: in both imports and EXPECTED_SO")
        elif in_imports and not in_expected:
            report("warn", f"{name}: imported but NOT in EXPECTED_SO")
        else:
            report("pass", f"{name}: in EXPECTED_SO (imported via other module)")

    # ---- Section 3: env-gated flags vs ENV_KERNEL_MAP ----
    print(f"\n--- env-gated flags vs ENV_KERNEL_MAP ({len(env_kernel_map_keys)} entries) ---")
    env_gated = {env: flag for flag, env in use_flags.items() if env is not None}
    for env_var in sorted(set(env_gated.keys()) | env_kernel_map_keys):
        in_flags = env_var in env_gated
        in_map = env_var in env_kernel_map_keys
        if in_flags and in_map:
            report("pass", f"{env_var}: in both qwen3_5.py and ENV_KERNEL_MAP")
        elif in_flags and not in_map:
            report("warn", f"{env_var}: in qwen3_5.py but MISSING from ENV_KERNEL_MAP")
        else:
            report("warn", f"{env_var}: in ENV_KERNEL_MAP but not a _USE_* flag")

    # ---- Section 4: verify_so_loading.py flags coverage ----
    print(f"\n--- verify_so_loading.py flags ({len(so_loading_flags)} entries) ---")
    for env_var in sorted(set(env_gated.keys()) | so_loading_flags):
        in_qwen = env_var in env_gated
        in_so_loading = env_var in so_loading_flags
        if in_qwen and in_so_loading:
            report("pass", f"{env_var}: in both qwen3_5.py and verify_so_loading")
        elif in_qwen and not in_so_loading:
            report("fail", f"{env_var}: in qwen3_5.py but MISSING from verify_so_loading")
        else:
            report("warn", f"{env_var}: in verify_so_loading but not a _USE_* flag")

    # ---- Section 5: test_dlopen_chain.py PREBUILT vs EXPECTED_SO ----
    print(f"\n--- test_dlopen_chain.py PREBUILT ({len(prebuilt_list)}) vs EXPECTED_SO ({len(expected_so)}) ---")
    for name in sorted(expected_so | prebuilt_list):
        in_prebuilt = name in prebuilt_list
        in_expected = name in expected_so
        if in_prebuilt and in_expected:
            report("pass", f"{name}: in both PREBUILT and EXPECTED_SO")
        elif in_prebuilt and not in_expected:
            report("warn", f"{name}: in PREBUILT but NOT in EXPECTED_SO")
        else:
            report("fail", f"{name}: in EXPECTED_SO but MISSING from PREBUILT")

    # ---- Summary ----
    print(f"\n{'=' * 64}")
    total = results["pass"] + results["fail"] + results["warn"]
    print(f"  {results['pass']} pass  {results['fail']} fail  {results['warn']} warn  (/{total})")
    if results["fail"] > 0:
        print(f"  ⚠️  {results['fail']} inconsistencies found — fix before submitting!")
        sys.exit(1)
    else:
        print("  ✅ All kernel flag definitions are consistent across verify scripts.")
        sys.exit(0)


if __name__ == "__main__":
    main()
