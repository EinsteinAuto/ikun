"""
Precompile moe_topk_softmax_v3.cu → .so during Docker build.
Build-only — does NOT require GPU. Verification deferred to runtime.

The .so will be cached by torch and loaded at runtime via:
  import moe_topk_softmax_v3
"""
import os, sys

def main():
    cu_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "moe_topk_softmax_v3.cu")
    if not os.path.isfile(cu_path):
        print(f"[MOE] ERROR: {cu_path} not found")
        sys.exit(1)

    print(f"[MOE] Compiling {cu_path} ...")

    # Detect corex compiler (BI-V100 Docker image)
    corex_clang = "/usr/local/corex/bin/clang++"
    use_corex = os.path.isfile(corex_clang)

    from torch.utils.cpp_extension import load

    extra_cuda_cflags = ["-O3"]
    extra_ldflags = []

    if use_corex:
        print(f"[MOE] Using corex clang at {corex_clang}")
        # corex torch extension picks up CUDA_HOME automatically
        # No special flags needed — torch.utils.cpp_extension handles ivcore10

    ext = load(
        name="moe_topk_softmax_v3",
        sources=[cu_path],
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=extra_ldflags,
        verbose=True,
    )
    print("[MOE] ✓ moe_topk_softmax_v3.so compiled")

    # Optional GPU verification — skip if no GPU (Docker build)
    import torch
    if torch.cuda.is_available():
        gating = torch.randn(4, 64, device='cuda', dtype=torch.float16)
        w, ids, _ = ext.moe_topk_softmax(gating, 8, True)
        assert not w.isnan().any(), "NaN in topk weights!"
        print("[MOE] ✓ GPU verification passed")
    else:
        print("[MOE] No GPU — skipping runtime verification (will verify at first inference)")

if __name__ == "__main__":
    main()
