"""
Bridge ixformer SDK 0.6.0 ops.infer.* API to CoreX 3.2.3 _C._functions.* API.

CoreX 3.2.3 ships ixformer._C with _functions submodule (v0.3.0 API).
SDK 0.6.0 inference/functions/*.py expects _C.infer.* namespace.
This shim creates _C.infer and maps each call to _C._functions or
a torch-native fallback where _functions lacks the operation.

Must be imported BEFORE ixformer.inference.functions.
Idempotent.
"""
import types
import functools


def patch():
    try:
        import ixformer._C as _C
    except ImportError:
        return False

    if hasattr(_C, 'infer'):
        return False  # already patched or native

    _fn = _C._functions
    infer = types.ModuleType('ixformer._C.infer')

    # --- Direct mappings (same semantics, slightly different signatures) ---

    def _linear(input, weight, act_type=-1, bias=None, output=None,
                persistent=False):
        """Map infer.linear -> _functions.linear_forward + activation."""
        import torch, torch.nn.functional as F
        if output is None:
            out_shape = list(input.shape)
            out_shape[-1] = weight.shape[0]
            output = input.new_empty(out_shape)
        # _functions.linear_forward expects (input, weight) or (input, weight, bias)
        # and writes into a pre-allocated output via linear_forward_
        try:
            if bias is not None:
                _fn.linear_forward_(input, weight, bias, output)
            else:
                _fn.linear_forward_(input, weight, output)
        except Exception:
            # Signature mismatch fallback to torch
            output = F.linear(input, weight, bias)

        # Apply activation
        if act_type == 3:
            output = F.gelu(output)
        elif act_type == 4:
            output = F.relu(output)
        elif act_type == 12:
            output = F.silu(output)
        return output

    def _linear_ex(input, weight, bias=None, output=None):
        """Map infer.linear_ex -> _functions.linear_forward (gemv path)."""
        import torch, torch.nn.functional as F
        if output is None:
            out_shape = list(input.shape)
            out_shape[-1] = weight.shape[0]
            output = input.new_empty(out_shape)
        try:
            if bias is not None:
                _fn.linear_forward_(input, weight, bias, output)
            else:
                _fn.linear_forward_(input, weight, output)
        except Exception:
            output = F.linear(input, weight, bias)
        return output

    def _mixed_type_linear(input, weight, bias=None, output=None):
        import torch.nn.functional as F
        result = F.linear(input.to(weight.dtype), weight, bias)
        if output is not None:
            output.copy_(result)
            return output
        return result

    # --- Functions with direct name mapping ---
    def _map(infer_name, fn_name):
        """Create a passthrough if _functions has the function."""
        if hasattr(_fn, fn_name):
            setattr(infer, infer_name, getattr(_fn, fn_name))
            return True
        return False

    # Register mappings
    infer.linear = _linear
    infer.linear_ex = _linear_ex
    infer.mixed_type_linear = _mixed_type_linear

    # Direct name mappings (infer.X -> _functions.X_forward or _functions.X)
    direct_maps = {
        'rms_norm': 'rms_norm_forward',
        'silu_and_mul': 'silu_and_mul_forward',
        'softmax': 'softmax_forward',
        'layernorm': 'layernorm_forward',
        'layer_norm': 'layernorm_forward',
        'groupnorm': 'groupnorm_forward',
        'conv': 'conv2d_forward',
        'matmul': 'matmul',
        'gelu_forward': 'gelu_forward',
        'act_bias_mm': 'act_bias_mm_forward',
        'bnb_dequant': 'bnb_dequant_forward',
        'bnb_qgemm': 'bnb_qgemm_forward',
        'bnb_quant': 'bnb_quant_forward',
        'bnb_rowcol_absmax': 'bnb_rowcol_absmax_forward',
        'bnb_doubleRowColQuant': 'bnb_doubleRowColQuant',
        'bnb_getColRowStats': 'bnb_getColRowStats',
        'bnb_mm_dequant': 'bnb_mm_dequant',
        'ixinfer_flash_attn_pad_fwd': 'ixinfer_flash_attn_pad_fwd',
        'ixinfer_flash_attn_pad_fwd_nomask': 'ixinfer_flash_attn_pad_fwd_nomask',
        'ixinfer_flash_attn_unpad': 'ixinfer_flash_attn_unpad_fwd',
        'vllm_cache_ops_reshape_and_cache': 'vllm_cache_ops_reshape_and_cache',
        'vllm_rotary_embedding': 'vllm_rotary_embedding_neox',
    }
    for infer_name, fn_name in direct_maps.items():
        _map(infer_name, fn_name)

    # For any remaining ops.infer.xxx call that we haven't mapped,
    # provide a __getattr__ that gives a clear error
    class _InferModule(types.ModuleType):
        def __getattr__(self, name):
            # Check _functions first
            fn_name = name + '_forward'
            if hasattr(_fn, fn_name):
                return getattr(_fn, fn_name)
            if hasattr(_fn, name):
                return getattr(_fn, name)
            raise AttributeError(
                f"ixformer._C.infer has no attribute '{name}' "
                f"(CoreX 3.2.3 _functions does not provide it)")

    # Transfer all explicit mappings to the smart module
    smart_infer = _InferModule('ixformer._C.infer')
    for attr in dir(infer):
        if not attr.startswith('_'):
            setattr(smart_infer, attr, getattr(infer, attr))

    _C.infer = smart_infer
    import sys
    sys.modules['ixformer._C.infer'] = smart_infer
    print('[ixformer_infer] bridged ops.infer -> _C._functions '
          f'({len(direct_maps)+3} mappings)')
    return True


_applied = patch()
