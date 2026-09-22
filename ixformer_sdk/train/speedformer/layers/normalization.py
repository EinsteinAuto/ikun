#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import warnings
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import ixformer.functions as ixff
from ixformer.train.functions import FusedRMSNorm as ixf_FusedRMSNorm
from apex.normalization.fused_layer_norm import FusedRMSNorm as apex_FusedRMSNorm
from ixformer.train.speedformer.layers.lazy import LazyInitContext


class BaseLayerNorm(ABC):
    @abstractmethod
    def from_native_module(module: nn.Module, sp_partial_derived: bool = False):
        """
        Convert a native PyTorch layer normalization module to a specific layer normalization module,
        and optionally mark parameters for gradient aggregation.

        Args:
            module (nn.Module): The native PyTorch layer normalization module to be converted.
            sp_partial_derived (bool): Whether this module's gradients are partially derived in sequence parallelism.

        Returns:
            nn.Module: The specific layer normalization module.

        Raises:
            AssertionError: If the provided module is not an instance of the supported layer normalization type.
        """


class IXFFusedRMSNorm(BaseLayerNorm):
    """
    This is a wrapper around the apex fused rms norm implementation. It is meant to be used only with the from_native_module interface.
    """

    def __init__(self) -> None:
        raise NotImplementedError(
            "FusedRMSNorm is not implemented as a physical class. "
            "It is meant to be used only with the from_native_module interface to Convert a native RMSNorm module to FusedRMSNorm module provided by apex."
        )

    @staticmethod
    def from_native_module(module: nn.Module, *args, **kwargs) -> nn.Module:
        r"""
        Convert a native RMSNorm module module to FusedRMSNorm module provided by ixformer,
        and optionally marking parameters for gradient aggregation.

        Args:
            module (nn.LayerNorm): The native PyTorch LayerNorm module to be converted.
            sp_partial_derived (bool): Whether this module's gradients are partially derived in sequence parallelism.

        Returns:
            nn.Module: FusedRMSNorm module.
        """

        LazyInitContext.materialize(module)

        # try to get normalized_shape, eps, elementwise_affine from the module
        normalized_shape = getattr(
            module, "normalized_shape", module.weight.shape[0])
        eps = module.variance_epsilon if hasattr(
            module, "variance_epsilon") else module.eps
        elementwise_affine = getattr(module, "elementwise_affine", True)

        rmsnorm = ixf_FusedRMSNorm(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
        )

        rmsnorm.weight = module.weight

        return rmsnorm


class APEXFusedRMSNorm(BaseLayerNorm):
    """
    This is a wrapper around the apex fused rms norm implementation. It is meant to be used only with the from_native_module interface.
    """

    def __init__(self) -> None:
        raise NotImplementedError(
            "FusedRMSNorm is not implemented as a physical class. "
            "It is meant to be used only with the from_native_module interface to Convert a native RMSNorm module to FusedRMSNorm module provided by apex."
        )

    @staticmethod
    def from_native_module(module: nn.Module, *args, **kwargs) -> nn.Module:
        r"""
        Convert a native RMSNorm module module to FusedRMSNorm module provided by ixformer,
        and optionally marking parameters for gradient aggregation.

        Args:
            module (nn.LayerNorm): The native PyTorch LayerNorm module to be converted.
            sp_partial_derived (bool): Whether this module's gradients are partially derived in sequence parallelism.

        Returns:
            nn.Module: FusedRMSNorm module.
        """

        LazyInitContext.materialize(module)

        # try to get normalized_shape, eps, elementwise_affine from the module
        normalized_shape = getattr(
            module, "normalized_shape", module.weight.shape[0])
        eps = module.variance_epsilon if hasattr(
            module, "variance_epsilon") else module.eps
        elementwise_affine = getattr(module, "elementwise_affine", True)

        rmsnorm = apex_FusedRMSNorm(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
        )

        rmsnorm.weight = module.weight

        return rmsnorm


# 替换torch LayerNorm 的forward
@staticmethod
def replace_layernorm_forward(self, input: torch.Tensor) -> torch.Tensor:

    output = torch.empty_like(input)

    return ixff.layernorm_train(input, self.weight, self.bias, self.normalized_shape, output, True)
