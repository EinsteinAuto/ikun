import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Dict, List, Union

import torch.nn as nn
from torch import Tensor
from torch.nn import Module

from ixformer.train.speedformer.policy.utils import SubModuleReplacementDescription
from ixformer.train.speedformer.policy.replacer import Replacer
import os
import sys
from ixformer.train.speedformer.layers.normalization import APEXFusedRMSNorm, IXFFusedRMSNorm
from ixformer.train.speedformer.layers.qwen2.attention import QwenAttention as IXF_QwenAttention


class Qwen2Replacer(Replacer):
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
                    target_module=IXF_QwenAttention,
                    kwargs={}
                ),
            ],
            target_key="Qwen2DecoderLayer"
        )
        
        self.append_or_create_submodule_replacement(
            description=[
                SubModuleReplacementDescription(
                    suffix="norm",
                    target_module=APEXFusedRMSNorm,
                    kwargs={}
                ),
            ],
            target_key="Qwen2Model"
        )
        

        
    