import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

import ixformer.train.functions as F
from ixformer.train.speedformer.models.baichuan.configuration_baichuan import BaichuanConfig
from ixformer.train.speedformer.models.baichuan.modeling_baichuan import MLP
from transformers.utils import logging

from ixformer.train.speedformer.layers.lazy import LazyInitContext


class BaseMLP(MLP):
    """
    这个层主要的优化点是：将linear1(act(cat(linear2(x), linear3(x))))的结构变成 linear1(act(linear23(x)))
    """

    def __init__(self, hidden_size, intermediate_size, hidden_act):
        super().__init__(hidden_size, intermediate_size, hidden_act)
        self.gate_up = nn.Linear(
            hidden_size, intermediate_size * 2, bias=False)
        del self.gate_proj, self.up_proj
        del self.act_fn

    def forward(self, x):
        res = self.gate_up(x)
        down_proj = self.down_proj(F.swiglu(res))
        return down_proj


class IXFBaichuanMLP(BaseMLP):
    def __init__(self) -> None:
        raise NotImplementedError(
            "IXFLlamaMLP is not implemented as a physical class. "
            "It is meant to be used only with the from_native_module interface to Convert a native LlamaAttention module to IXFLlamaMLP module provided above."
        )

    @staticmethod
    def from_native_module(module: nn.Module, *args, **kwargs) -> nn.Module:
        hidden_size, intermediate_size = module.gate_proj.in_features, module.gate_proj.out_features
        hidden_act = "silu"

        mlp = BaseMLP(hidden_size=hidden_size,
                      intermediate_size=intermediate_size, hidden_act=hidden_act)

        mlp.gate_up.weight.data = torch.concat(
            (module.gate_proj.weight.data, module.up_proj.weight.data), dim=0)
        mlp.down_proj.weight.data = module.down_proj.weight.data

        return mlp
