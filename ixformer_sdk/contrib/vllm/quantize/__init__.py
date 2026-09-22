from .smoothquant import smoothquant_prepare_quantize,smoothquant_export_quantized_weights
from .w8a16 import w8a16_prepare_quantize,w8a16_export_quantized_weights

SUPPORT_METHOD = {
    "smoothquant": [smoothquant_prepare_quantize,smoothquant_export_quantized_weights],
    "w8a16": [w8a16_prepare_quantize,w8a16_export_quantized_weights],
}

def get_quantize_method(method_name:str):
    try:
        method = SUPPORT_METHOD[method_name]
    except:
        raise ValueError(f"Only support quantization methods: {SUPPORT_METHOD.keys()}, but got {method_name}")
    return method