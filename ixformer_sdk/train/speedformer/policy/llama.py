import warnings
import types
from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Dict, List, Union

import torch.nn as nn
from torch import Tensor
from torch.nn import Module

from ixformer.train.speedformer.policy.utils import SubModuleReplacementDescription
from ixformer.train.speedformer.policy.replacer import Replacer

from ixformer.train.speedformer.layers.normalization import APEXFusedRMSNorm, IXFFusedRMSNorm
from ixformer.train.speedformer.layers.llama.attention import LlamaAttention as IXF_LlamaAttention
from ixformer.train.speedformer.layers.llama.mlp import IXFLlamaMLP
from ixformer.train.speedformer.layers.llama.llama_method import LlamaModel_forward, LlamaForCausalLM_forward
from ixformer.train.speedformer.layers.fast_lora.fast_lora import apply_lora_mlp_swiglu

from peft import PeftType


class LlamaReplacer(Replacer):
    def __init__(self):
        self.policy = {}

    def module_policy(self) -> Dict[Union[str, nn.Module], List[SubModuleReplacementDescription]]:
        self.append_or_create_submodule_replacement(
            description=[
                SubModuleReplacementDescription(
                    suffix="input_layernorm",
                    target_module=APEXFusedRMSNorm,
                    kwargs={}
                ),
                SubModuleReplacementDescription(
                    suffix="post_attention_layernorm",
                    target_module=APEXFusedRMSNorm,
                    kwargs={},
                ),
                SubModuleReplacementDescription(
                    suffix="self_attn",
                    target_module=IXF_LlamaAttention,
                    kwargs={}
                ),
                # SubModuleReplacementDescription(
                #     suffix="mlp",
                #     target_module=IXFLlamaMLP,
                #     kwargs={}
                # ),
            ],
            target_key="LlamaDecoderLayer"
        )

        self.append_or_create_submodule_replacement(
            description=[
                SubModuleReplacementDescription(
                    suffix="norm",
                    target_module=APEXFusedRMSNorm,
                    kwargs={}
                ),
            ],
            target_key="LlamaModel"
        )

        self.append_or_create_method_replacement(
            description=[
                {"forward": LlamaModel_forward()}
            ],
            target_key="LlamaModel"
        )
        self.append_or_create_method_replacement(
            description=[
                {"forward": LlamaForCausalLM_forward()}
            ],
            target_key="LlamaForCausalLM"
        )

    def post_process(self, model: nn.Module):
        if model.peft_type != PeftType.LORA:
            return
        peft_config = model.peft_config
        active_adapter = model.active_adapters[0] if \
            hasattr(model, "active_adapters") else model.active_adapter
        target_modules = peft_config[active_adapter].target_modules

        # for now, fast_lora only support lora_dropout=0 and bias=None
        lora_dropout = model.peft_config[active_adapter].lora_dropout
        bias = model.peft_config[active_adapter].bias

        # 首先判断是否可以使用fast_lora
        check = lora_dropout == 0 and bias == "none"

        # 其次确定mlp的3个线性层是否在target_modules
        mlp_use_fastlora = "gate_proj" in target_modules and "up_proj" in target_modules and "up_proj" in target_modules

        n_mlp = 0
        if check:
            if mlp_use_fastlora:
                for layer in model.model.model.layers:
                    layer.mlp.forward = types.MethodType(
                        apply_lora_mlp_swiglu, layer.mlp)
                    n_mlp += 1

        print(f"{len(model.model.model.layers)} layers replace mlp with fast_lora mlp")
