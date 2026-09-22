# SPDX-License-Identifier: Apache-2.0
import os

import torch

# set some common config/environment variables that should be set
# for all processes created by vllm and all processes
# that interact with vllm workers.
# they are executed whenever `import vllm` is called.

# see https://github.com/NVIDIA/nccl/issues/1234
os.environ['NCCL_CUMEM_ENABLE'] = '0'

# see https://github.com/vllm-project/vllm/pull/15951
# it avoids unintentional cuda initialization from torch.cuda.is_available()
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'

# see https://github.com/vllm-project/vllm/issues/10480
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
# see https://github.com/vllm-project/vllm/issues/10619
# torch._inductor.config.compile_threads = 1

# --- BI-V100 Triton compat: inject get_cuda_stream into triton.runtime.jit ---
# torch._inductor.triton_heuristics expects triton.runtime.jit.get_cuda_stream
# but Triton 2.3.1 (standard, BI-V100 image) does not export it.
# Upstream PyTorch >=2.5 switched to torch._C._cuda_getCurrentRawStream.
# We inject a compat shim so the import succeeds in worker processes too.
# This must run before any torch._inductor import.
def _patch_triton_get_cuda_stream():
    try:
        import triton.runtime.jit as _jit
    except ImportError:
        return
    if hasattr(_jit, 'get_cuda_stream'):
        return
    def _get_cuda_stream(device_index: int = 0) -> int:
        if hasattr(torch._C, '_cuda_getCurrentRawStream'):
            return torch._C._cuda_getCurrentRawStream(device_index)
        return 0
    _jit.get_cuda_stream = _get_cuda_stream

_patch_triton_get_cuda_stream()
del _patch_triton_get_cuda_stream

# --- BI-V100 ixformer compat: bridge SDK 0.6.0 infer API to CoreX 3.2.3 ---
# SDK 0.6.0 inference/functions/*.py calls ops.infer.xxx but CoreX 3.2.3
# _C.so (v0.3.0) only has _C._functions.xxx_forward. We bridge them.
# Must run before any ixformer.inference.functions import.
def _patch_ixformer_infer():
    try:
        import ixformer._C as _C
    except ImportError:
        return
    if hasattr(_C, 'infer'):
        return
    try:
        # Use the full bridge module if deployed
        import importlib.util, os
        shim_path = os.path.join(os.path.dirname(__file__),
                                 '..', '..', 'patch_ixformer_infer.py')
        if not os.path.exists(shim_path):
            # Try qwen3_6_scripts location
            for candidate in [
                '/usr/local/corex/lib64/python3/dist-packages/vllm/patch_ixformer_infer.py',
                os.path.join(os.path.dirname(__file__), 'patch_ixformer_infer.py'),
            ]:
                if os.path.exists(candidate):
                    shim_path = candidate
                    break
        if os.path.exists(shim_path):
            spec = importlib.util.spec_from_file_location(
                'patch_ixformer_infer', shim_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return
    except Exception:
        pass

    # Minimal inline fallback: just map linear
    if not hasattr(_C, '_functions'):
        return
    import types
    infer = types.ModuleType('ixformer._C.infer')
    _fn = _C._functions

    def _linear(input, weight, act_type=-1, bias=None, output=None,
                persistent=False):
        import torch.nn.functional as F
        result = F.linear(input, weight, bias)
        if act_type == 3: result = F.gelu(result)
        elif act_type == 4: result = F.relu(result)
        elif act_type == 12: result = F.silu(result)
        if output is not None:
            output.copy_(result)
            return output
        return result

    infer.linear = _linear
    infer.linear_ex = lambda inp, w, b=None, o=None: _linear(inp, w, -1, b, o)
    infer.mixed_type_linear = lambda inp, w, b=None, o=None: _linear(
        inp.to(w.dtype), w, -1, b, o)
    _C.infer = infer

_patch_ixformer_infer()
del _patch_ixformer_infer

# --- BI-V100 ixformer.distributed: inject missing gather/send/recv ---
# SDK 0.6.0 distributed has gather() but CoreX 3.2.3 system install doesn't.
# Inject it using all_gather_into_tensor (which does exist).
def _patch_ixformer_distributed():
    try:
        import ixformer.distributed as ixfd
    except ImportError:
        return
    if hasattr(ixfd, 'gather'):
        return

    def gather(tensor, gather_list=None, dst=0, group=None,
               async_op=False, use_comm_stream=False):
        """gather via all_gather_into_tensor + slice on non-dst ranks."""
        import torch
        import torch.distributed as dist
        world_size = ixfd.get_group_world_size(group) if group else 1
        if world_size <= 1:
            if gather_list is not None:
                gather_list[0].copy_(tensor)
            return
        # Use all_gather_into_tensor which exists in this version
        flat_output = torch.empty(
            [world_size] + list(tensor.shape),
            dtype=tensor.dtype, device=tensor.device)
        ixfd.all_gather_into_tensor(
            flat_output.view(-1), tensor.contiguous(),
            group=group, async_op=async_op)
        # Get current rank via torch.distributed (reliable across versions)
        rank = dist.get_rank()
        if rank == dst and gather_list is not None:
            for i in range(world_size):
                gather_list[i].copy_(flat_output[i])

    ixfd.gather = gather
    print('[ixformer_compat] injected gather into ixformer.distributed')

_patch_ixformer_distributed()
del _patch_ixformer_distributed
