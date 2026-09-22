#!/usr/bin/env python3
"""
Verify CCCL allocator preload on BI-V100.

Run WITHOUT preload (should crash with expandable_segments:True):
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 verify_preload.py

Run WITH preload (should succeed):
    LD_PRELOAD=./libcccl_allocator.so CCCL_ALLOC_DEBUG=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 verify_preload.py
"""
import os
import sys
import time

print(f"PYTORCH_CUDA_ALLOC_CONF = {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '(not set)')}")
print(f"LD_PRELOAD = {os.environ.get('LD_PRELOAD', '(not set)')}")
print()

import torch

print(f"torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("CUDA not available, exiting")
    sys.exit(1)

device = torch.device("cuda:0")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print()

# Test 1: Basic allocation
print("=== Test 1: Basic allocation ===")
t1 = torch.zeros(1024, 1024, dtype=torch.float16, device=device)
print(f"  Allocated 1024x1024 fp16 tensor: {t1.shape}, {t1.element_size() * t1.nelement() / 1024**2:.1f} MB")

# Test 2: Large allocation (simulates model weight loading)
print("=== Test 2: Large allocation (512MB) ===")
t2 = torch.zeros(256 * 1024 * 1024, dtype=torch.float16, device=device)
print(f"  Allocated 512MB tensor: {t2.nelement() * t2.element_size() / 1024**2:.0f} MB")

# Test 3: Alloc-free-realloc cycle (tests caching)
print("=== Test 3: Alloc-free-realloc cycle ===")
t3 = torch.zeros(64 * 1024 * 1024, dtype=torch.float16, device=device)
ptr_first = t3.data_ptr()
del t3
torch.cuda.empty_cache()
t3b = torch.zeros(64 * 1024 * 1024, dtype=torch.float16, device=device)
ptr_second = t3b.data_ptr()
reused = "YES (cached)" if ptr_first == ptr_second else "NO (new alloc)"
print(f"  First ptr:  0x{ptr_first:x}")
print(f"  Second ptr: 0x{ptr_second:x}")
print(f"  Block reused: {reused}")

# Test 4: Multiple sizes (tests bin routing)
print("=== Test 4: Multiple bin sizes ===")
sizes = [512, 4096, 32768, 262144, 2*1024*1024, 32*1024*1024]
tensors = []
for sz in sizes:
    t = torch.zeros(sz // 2, dtype=torch.float16, device=device)
    tensors.append(t)
    print(f"  {sz:>12} bytes -> allocated at 0x{t.data_ptr():x}")
del tensors

# Test 5: OOM recovery
print("=== Test 5: Memory pressure ===")
mem_free = torch.cuda.mem_get_info()[0]
print(f"  Free memory: {mem_free / 1024**3:.2f} GB")

# Cleanup
del t1, t2, t3b
torch.cuda.empty_cache()

mem_after = torch.cuda.mem_get_info()[0]
print(f"  After cleanup: {mem_after / 1024**3:.2f} GB")
print(f"  Recovered: {(mem_after - mem_free) / 1024**2:.0f} MB")

print()
print("ALL TESTS PASSED")
