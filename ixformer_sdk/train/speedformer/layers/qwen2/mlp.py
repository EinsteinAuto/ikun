import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

import ixformer.train.functions as F
from ixformer.train.speedformer.models.qwen2.configuration_qwen2 import Qwen2Config
from ixformer.train.speedformer.models.qwen2.modeling_qwen2 import Qwen2MLP
from transformers import Cache
from transformers.utils import logging

from ixformer.train.speedformer.layers.lazy import LazyInitContext


class BaseQwen2MLP(Qwen2MLP):
    """
    这个层主要的优化点是：将linear1(act(cat(linear2(x), linear3(x))))的结构变成 linear1(act(linear23(x)))
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate_up = nn.Linear(
            self.hidden_size, self.intermediate_size * 2, bias=False)
        del self.gate_proj, self.up_proj
        del self.act_fn

    def forward(self, x):
        res = self.gate_up(x)
        down_proj = self.down_proj(F.swiglu(res))
        return down_proj


class IXFQwen2MLP(BaseQwen2MLP):
    def __init__(self) -> None:
        raise NotImplementedError(
            "IXFQwen2MLP is not implemented as a physical class. "
            "It is meant to be used only with the from_native_module interface to Convert a native Qwen2MLP module to BaseQwen2MLP module provided above."
        )

    @staticmethod
    def from_native_module(module: nn.Module, *args, **kwargs) -> nn.Module:

        LazyInitContext.materialize(module)

        config = getattr(module, "config")

        mlp = BaseQwen2MLP(config=config)

        mlp.gate_up.weight.data = torch.concat(
            (module.gate_proj.weight.data, module.up_proj.weight.data), dim=0)
        mlp.down_proj.weight.data = module.down_proj.weight.data

        return mlp
