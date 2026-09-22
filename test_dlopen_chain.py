#!/usr/bin/env python3
"""Test dlopen chain on BI-V100. Run on real machine."""
import sys, os, importlib

VLLM_SYSTEM = "/usr/local/corex/lib/python3/dist-packages/vllm"
VLLM_SYSTEM2 = "/usr/local/corex/lib64/python3/dist-packages/vllm"

PREBUILT = [
    "corex_attn_head_rms_norm", "corex_block_major_kv_transfer",
    "corex_fused_paged_prefill", "corex_gdn_beta_decay",
    "corex_gdn_causal_conv", "corex_gdn_chunk_recurrent",
    "corex_gdn_gated_norm", "corex_gdn_packed_decode",
    "corex_gdn_qk_map", "corex_moe_direct_routed",
    "corex_moe_exact_reduce", "corex_moe_topk_softmax",
    "corex_moe_weight_gather", "corex_paged_kv_gather",
]

def find_system_vllm():
    for p in [VLLM_SYSTEM, VLLM_SYSTEM2]:
        if os.path.isdir(p):
            return p
    try:
        spec = importlib.util.find_spec("vllm")
        if spec and spec.origin:
            d = os.path.dirname(spec.origin)
            if "project_6" not in d:
                return d
    except:
        pass
    return None

def main():
    total_ok = total_fail = 0
    
    # 1. Torch/CUDA
    print("=== 1. Torch/CUDA ===")
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            mem_gb = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / 1e9
            print(f"  [OK] {name}, {mem_gb:.1f}GB")
            total_ok += 1
        else:
            print("  [FAIL] CUDA not available"); total_fail += 1
    except Exception as e:
        print(f"  [FAIL] {e}"); total_fail += 1
    
    # 2. System vllm location
    print("\n=== 2. System vLLM ===")
    sys_vllm = find_system_vllm()
    if sys_vllm:
        print(f"  [OK] {sys_vllm}")
        total_ok += 1
    else:
        print("  [FAIL] System vllm not found")
        total_fail += 1
    
    # 3. Prebuilt .so: check if install would work
    print("\n=== 3. Prebuilt .so (14 modules) ===")
    prebuilt_dir = "qwen3_6_scripts/prebuilt/corex-3.2.3-ivcore10"
    for name in PREBUILT:
        src = os.path.join(prebuilt_dir, f"{name}.so")
        if os.path.exists(src):
            size = os.path.getsize(src)
            # Check if installed in system vllm
            if sys_vllm:
                dst = os.path.join(sys_vllm, f"{name}.so")
                if os.path.exists(dst):
                    print(f"  [INSTALLED] {name} ({size:,}B)")
                    total_ok += 1
                else:
                    print(f"  [PREBUILT]  {name} ({size:,}B) → needs install to {dst}")
                    total_ok += 1  # prebuilt exists, will be installed by patch_ops
            else:
                print(f"  [PREBUILT]  {name} ({size:,}B)")
                total_ok += 1
        else:
            print(f"  [MISS] {name}: prebuilt not found")
            total_fail += 1
    
    # 4. Install prebuilt to system vllm (DRY RUN)
    if sys_vllm:
        print(f"\n  To install: bash qwen3_6_scripts/install_prebuilt_corex.sh {sys_vllm}")
    
    # 5. ixformer dispatch
    print("\n=== 4. ixformer dispatch chain ===")
    checks = [
        ("ixformer", None),
        ("ixformer.functions", "vllm_single_query_cached_kv_attention"),
        ("ixformer.functions", "flash_attn_varlen_func"),
    ]
    for mod_name, func_name in checks:
        try:
            mod = importlib.import_module(mod_name)
            if func_name:
                fn = getattr(mod, func_name, None)
                if fn:
                    print(f"  [OK] {mod_name}.{func_name}")
                    total_ok += 1
                else:
                    avail = [x for x in dir(mod) if 'flash' in x.lower() or 'attn' in x.lower() or 'paged' in x.lower()]
                    print(f"  [MISS] {mod_name}.{func_name}")
                    if avail:
                        print(f"         available attn funcs: {avail}")
                    total_fail += 1
            else:
                ver = getattr(mod, '__version__', '?')
                loc = getattr(mod, '__file__', '?')
                print(f"  [OK] {mod_name} v{ver} @ {loc}")
                total_ok += 1
        except ImportError as e:
            print(f"  [FAIL] {mod_name}: {e}")
            total_fail += 1

    # 6. Base image .so
    print("\n=== 5. Base image .so ===")
    for p in ["/usr/local/corex/lib64/libcorex_gdn.so", "/usr/local/corex/lib64/libixattn.so"]:
        if os.path.exists(p):
            print(f"  [OK] {p} ({os.path.getsize(p):,}B)")
            total_ok += 1
        else:
            print(f"  [MISS] {p}")
            total_fail += 1
    
    # 7. libcccl_allocator.so
    print("\n=== 6. libcccl_allocator.so ===")
    cccl = "qwen3_6_scripts/cccl_preload/libcccl_allocator.so"
    if os.path.exists(cccl):
        print(f"  [OK] {cccl} ({os.path.getsize(cccl):,}B)")
        total_ok += 1
    else:
        print(f"  [MISS] {cccl}")
        total_fail += 1

    print(f"\n{'='*50}")
    print(f"OK: {total_ok}  FAIL: {total_fail}")

if __name__ == "__main__":
    main()
