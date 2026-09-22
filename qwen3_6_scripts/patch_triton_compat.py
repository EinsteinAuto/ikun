"""
BI-V100 Triton/torch._inductor compatibility patch.

Problem:
    torch._inductor.triton_heuristics (corex torch build) line 43:
        from triton.runtime.jit import get_cuda_stream, KernelInterface
    But Triton 2.3.1 (standard, shipped in BI-V100 image) does not export
    get_cuda_stream from triton.runtime.jit.

Root cause:
    PyTorch's inductor was written against an older Triton that re-exported
    get_cuda_stream.  Newer Triton moved it out; upstream PyTorch >=2.5
    imports from torch._C._cuda_getCurrentRawStream instead, and wraps
    Triton symbols through torch._inductor.runtime.triton_compat.

Fix:
    1. Inject get_cuda_stream into triton.runtime.jit at the Python level
       so the old torch._inductor import succeeds.
    2. Optionally patch the .py file on disk (deploy-time) so the fix
       survives across process forks and reimports.

This module should be imported BEFORE torch._inductor is loaded.
Safe to import multiple times (idempotent).
"""

import importlib
import sys


def _get_cuda_stream_from_torch(device_index: int = 0) -> int:
    """Get the current CUDA stream pointer, matching the old Triton API."""
    import torch
    if hasattr(torch._C, '_cuda_getCurrentRawStream'):
        return torch._C._cuda_getCurrentRawStream(device_index)
    # Fallback: return default stream (0)
    return 0


def patch_triton_runtime_jit():
    """
    Ensure triton.runtime.jit.get_cuda_stream exists.

    If triton.runtime.jit is already imported and lacks get_cuda_stream,
    inject it.  If not yet imported, inject after import.  This makes
    torch._inductor.triton_heuristics importable.
    """
    try:
        import triton.runtime.jit as jit_module
    except ImportError:
        # Triton not installed at all; nothing to patch
        return False

    if hasattr(jit_module, 'get_cuda_stream'):
        # Already present (compatible Triton version), nothing to do
        return False

    # Inject the compat shim
    jit_module.get_cuda_stream = _get_cuda_stream_from_torch
    return True


def patch_triton_heuristics_on_disk(torch_inductor_path: str = None):
    """
    Patch the torch._inductor triton_heuristics.py file on disk to use
    torch._C._cuda_getCurrentRawStream instead of triton.runtime.jit.get_cuda_stream.

    This is the deploy-time fix applied by patch_ops.sh.

    Args:
        torch_inductor_path: Path to torch/_inductor directory.
            If None, auto-detected from torch installation.

    Returns:
        True if file was patched, False if already patched or not found.
    """
    import os

    if torch_inductor_path is None:
        import torch
        torch_root = os.path.dirname(torch.__file__)
        torch_inductor_path = os.path.join(torch_root, '_inductor')

    # Check both old location and new location
    candidates = [
        os.path.join(torch_inductor_path, 'triton_heuristics.py'),
        os.path.join(torch_inductor_path, 'runtime', 'triton_heuristics.py'),
    ]

    patched = False
    for filepath in candidates:
        if not os.path.exists(filepath):
            continue

        with open(filepath, 'r') as f:
            content = f.read()

        # Idempotency: skip if already patched (try/except block present)
        if '_cuda_getCurrentRawStream as get_cuda_stream' in content:
            print(f'[skip] already patched: {filepath}')
            continue

        old_import = 'from triton.runtime.jit import get_cuda_stream'
        if old_import not in content:
            continue

        # Replace the problematic import
        # Handle the case where KernelInterface is on the same line
        if 'from triton.runtime.jit import get_cuda_stream, KernelInterface' in content:
            old_line = 'from triton.runtime.jit import get_cuda_stream, KernelInterface'
            new_block = """try:
    from triton.runtime.jit import get_cuda_stream, KernelInterface
except ImportError:
    from torch._C import _cuda_getCurrentRawStream as get_cuda_stream
    try:
        from triton.runtime.jit import KernelInterface
    except ImportError:
        from triton.compiler import KernelInterface"""
            content = content.replace(old_line, new_block)
        else:
            new_import = """try:
    from triton.runtime.jit import get_cuda_stream
except ImportError:
    from torch._C import _cuda_getCurrentRawStream as get_cuda_stream"""
            content = content.replace(old_import, new_import)

        with open(filepath, 'w') as f:
            f.write(content)

        # Clear __pycache__ for the patched file
        cache_dir = os.path.join(os.path.dirname(filepath), '__pycache__')
        if os.path.isdir(cache_dir):
            import shutil
            shutil.rmtree(cache_dir, ignore_errors=True)

        patched = True
        print(f'[ok] patched triton_heuristics: {filepath}')

    return patched


# Auto-apply runtime patch on import
_applied = patch_triton_runtime_jit()
if _applied:
    print('[triton_compat] injected get_cuda_stream into triton.runtime.jit')
