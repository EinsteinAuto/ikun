#!/usr/bin/env python3
"""Test Triton availability on BI-V100."""
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")

try:
    import triton
    import triton.language as tl
    print(f"Triton version: {triton.__version__}")

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    n = 1024
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = torch.empty(n, device="cuda")
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    add_kernel[grid](x, y, out, n, BLOCK=256)
    torch.cuda.synchronize()
    ref = x + y
    diff = (out - ref).abs().max().item()
    print(f"Triton kernel test: diff={diff:.8f} {'PASS' if diff < 1e-6 else 'FAIL'}")
except ImportError as e:
    print(f"Triton not available: {e}")
except Exception as e:
    print(f"Triton error: {e}")
