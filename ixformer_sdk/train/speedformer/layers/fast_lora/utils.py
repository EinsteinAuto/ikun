# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ctypes
import bitsandbytes as bnb
from packaging.version import Version
import torch
import triton
MAX_FUSED_SIZE = 65536
next_power_of_2 = triton.next_power_of_2

# torch.cuda.amp.custom_fwd is deprecated >= 2.4
if Version(torch.__version__) < Version("2.4.0"):
    torch_amp_custom_fwd = torch.cuda.amp.custom_fwd
    torch_amp_custom_bwd = torch.cuda.amp.custom_bwd
else:
    torch_amp_custom_fwd = torch.amp.custom_fwd(device_type="cuda")
    torch_amp_custom_bwd = torch.amp.custom_bwd(device_type="cuda")
pass


# tl.math.tanh now is libdevice.tanh
if Version(triton.__version__) >= Version("3.0.0"):
    from triton.language.extra import libdevice
    triton_tanh = libdevice.tanh
else:
    import triton.language as tl
    triton_tanh = tl.math.tanh
pass


def calculate_settings(n):
    BLOCK_SIZE = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"Cannot launch Triton kernel since n = {n} exceeds "
                           f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}.")
    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


pass


get_ptr = bnb.functional.get_ptr
cdequantize_blockwise_fp32 = bnb.functional.lib.cdequantize_blockwise_fp32
cdequantize_blockwise_fp16_nf4 = bnb.functional.lib.cdequantize_blockwise_fp16_nf4
cdequantize_blockwise_bf16_nf4 = bnb.functional.lib.cdequantize_blockwise_bf16_nf4


def QUANT_STATE(W):
    return getattr(W, "quant_state", None)


pass


def get_lora_parameters(proj):
    # For DPO or disabled adapters
    base_layer = (proj.base_layer if hasattr(proj, "base_layer") else proj)
    W = base_layer.weight

    if not hasattr(proj, "disable_adapters") or proj.disable_adapters or proj.merged:
        return W, QUANT_STATE(W), None, None, None
    pass

    active_adapter = proj.active_adapters[0] if \
        hasattr(proj, "active_adapters") else proj.active_adapter
    A = proj.lora_A[active_adapter].weight
    B = proj.lora_B[active_adapter].weight
    s = proj.scaling[active_adapter]
    return W, QUANT_STATE(W), A, B, s


pass


def get_lora_parameters_bias(proj):
    # For DPO or disabled adapters
    base_layer = (proj.base_layer if hasattr(proj, "base_layer") else proj)
    W = base_layer.weight
    bias = base_layer.bias

    if not hasattr(proj, "disable_adapters") or proj.disable_adapters or proj.merged:
        return W, QUANT_STATE(W), None, None, None, bias
    pass

    active_adapter = proj.active_adapters[0] if \
        hasattr(proj, "active_adapters") else proj.active_adapter
    A = proj.lora_A[active_adapter].weight
    B = proj.lora_B[active_adapter].weight
    s = proj.scaling[active_adapter]
    return W, QUANT_STATE(W), A, B, s, bias


pass


def fast_dequantize(W, quant_state=None, out=None):
    if quant_state is None:
        return W
    if type(quant_state) is not list:
        # New quant_state as a class
        # https://github.com/TimDettmers/bitsandbytes/pull/763/files
        absmax = quant_state.absmax
        shape = quant_state.shape
        dtype = quant_state.dtype
        blocksize = quant_state.blocksize
        offset = quant_state.offset
        state2 = quant_state.state2
        absmax2 = state2.absmax
        code2 = state2.code
        blocksize2 = state2.blocksize
    else:
        # Old quant_state as a list of lists
        absmax, shape, dtype, blocksize, compressed_stats, _, _ = quant_state
        offset, state2 = compressed_stats
        absmax2, code2, blocksize2, _, _, _, _ = state2
    pass

    # Create weight matrix
    if out is None:
        out = torch.empty(shape, dtype=dtype, device="cuda:0")
    else:
        assert (out.shape == shape)
        assert (out.dtype == dtype)

    # NF4 dequantization of statistics
    n_elements_absmax = absmax.numel()
    out_absmax = torch.empty(
        n_elements_absmax, dtype=torch.float32, device="cuda:0")

    # Do dequantization
    ptr_out_absmax = get_ptr(out_absmax)
    cdequantize_blockwise_fp32(
        get_ptr(code2), get_ptr(absmax), get_ptr(absmax2), ptr_out_absmax,
        ctypes.c_int(blocksize2), ctypes.c_int(n_elements_absmax)
    )
    out_absmax += offset

    fx = cdequantize_blockwise_fp16_nf4 if dtype == torch.float16 else \
        cdequantize_blockwise_bf16_nf4
    fx(get_ptr(None), get_ptr(W), ptr_out_absmax, get_ptr(out),
       ctypes.c_int(blocksize), ctypes.c_int(out.numel()))

    # Careful returning transposed data
    is_transposed = (True if W.shape[0] == 1 else False)
    return out.t() if is_transposed else out


pass


def matmul_lora(X, W, W_quant, A, B, s, out=None):
    dtype = X.dtype
    W = fast_dequantize(W.t(), W_quant)

    if X.dim() == 3:
        batch, seq_len, d = X.shape
        X = X.view(-1, X.shape[-1])
        reshape = True
    else:
        reshape = False
    pass

    out = torch.matmul(X, W, out=out)
    if W_quant is not None:
        del W

    if A is not None:
        # LoRA is enabled
        A, B = A.t(), B.t()
        out += (X @ A.to(dtype)) @ (s * B.to(dtype))
    pass

    return out.view(batch, seq_len, -1) if reshape else out


pass
