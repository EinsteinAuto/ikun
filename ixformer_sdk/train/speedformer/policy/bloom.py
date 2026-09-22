import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Dict, List, Union

import torch.nn as nn
from torch import Tensor
from torch.nn import Module

from ixformer.train.speedformer.policy.utils import SubModuleReplacementDescription
from ixformer.train.speedformer.policy.replacer import Replacer

from ixformer.train.speedformer.models.bloom.modeling_bloom import BloomModel, BloomBlock
from ixformer.train.speedformer.layers.normalization import APEXFusedRMSNorm, IXFFusedRMSNorm
from ixformer.train.speedformer.layers.bloom.attention import BloomFlashAttention


class BloomReplacer(Replacer):
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
                    suffix="self_attention",
                    target_module=BloomFlashAttention,
                    kwargs={}
                ),
            ],
            target_key="BloomBlock"
        )

        self.append_or_create_submodule_replacement(
            description=[
                SubModuleReplacementDescription(
                    suffix="ln_f",
                    target_module=APEXFusedRMSNorm,
                    kwargs={}
                ),
            ],
            target_key=BloomModel
        )
