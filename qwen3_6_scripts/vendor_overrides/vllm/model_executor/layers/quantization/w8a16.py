from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.parameter import (GroupQuantScaleParameter,
                                           PackedvLLMParameter)
from vllm.model_executor.utils import set_weight_attrs


class W8a16Config(QuantizationConfig):
    """Config class for W8a16.
    
    """

    def __init__(
        self,
    ) -> None:
        pass

    def __repr__(self) -> str:
        return ("W8a16Config")

    def get_name(self) -> str:
        return "w8a16"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    def get_min_capability(self) -> int:
        return 75

    @staticmethod
    def get_config_filenames():
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "W8a16Config":
        return cls()
    
    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["W8a16LinearMethod"]:
        if isinstance(layer, LinearBase):
            return W8a16LinearMethod(self)
        return None


    def get_scaled_act_names(self) -> List[str]:
        return []


class W8a16LinearMethod(LinearMethodBase):
    """Linear method for w8a16.

    """

    def __init__(self, quant_config: W8a16Config):
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: List[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        output_size_per_partition = sum(output_partition_sizes)
        weight = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            weight, {
                "input_dim": 1,
                "output_dim": 0,
            })
        
        scales = Parameter(
            torch.empty(
                1,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(scales, {
            "input_dim": None,
            "output_dim": 1,
        })
        
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)
        layer.register_parameter("scales", scales)
        set_weight_attrs(scales, extra_weight_attrs)
        
        
    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        qweight = layer.weight
        scales = layer.scales
        out_shape = (x.shape[:-1] + (qweight.shape[-2],))
        reshaped_x = x.reshape(-1, x.shape[-1])
        out = ops.linear_w8a16(reshaped_x, qweight, scales, format="TN")
        if bias is not None:
            out = out + bias
        return out.reshape(out_shape)