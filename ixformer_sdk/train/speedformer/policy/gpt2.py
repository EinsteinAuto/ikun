import torch
import torch.nn as nn
from torch.nn import LayerNorm
from types import ModuleType, MethodType
from abc import ABC

from ixformer.train.speedformer.models.gpt2.modeling_gpt2 import GPT2FlashAttention2

from ixformer.train.speedformer.layers.normalization import replace_layernorm_forward
from ixformer.train.speedformer.layers.gpt2.attention import replace_flash_attn_forward


class GPT2Replacer(ABC):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def accelerate(model):
        # layer/kernel replace
        for name, module in model.named_modules():
            if isinstance(module, LayerNorm):
                module.forward = MethodType(replace_layernorm_forward, module)
            if isinstance(module, GPT2FlashAttention2):
                module._flash_attention_forward = MethodType(
                    replace_flash_attn_forward, module)

        return model
