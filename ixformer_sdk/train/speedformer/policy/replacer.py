import warnings
from types import MethodType
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set, Union
import tabulate

import torch.nn as nn

from ixformer.train.speedformer.policy.utils import SubModuleReplacementDescription, ModulePolicyDescription, getattr_, setattr_, print_rank_0


class Replacer(ABC):
    def __init__(self):
        self.policy = {}

    def module_policy(self) -> Dict[Union[str, nn.Module], List[SubModuleReplacementDescription]]:
        r"""
        This method returns the module policy, which is a dictionary. The key is the module name or the module object,
        and the value is the ModulePolicyDescription object. The ModulePolicyDescription object describes how the module
        will be transformed.
        """

    def append_or_create_submodule_replacement(
        self,
        description: Union[SubModuleReplacementDescription, List[SubModuleReplacementDescription]],
        target_key: Union[str, nn.Module],
    ) -> Dict[Union[str, nn.Module], List]:
        r"""
        Append or create a new submodule replacement description to the policy for the given key.

        Args:
            submodule_replace_desc (Union[SubModuleReplacementDescription, List[SubModuleReplacementDescription]]): the submodule replacement description to be appended
            policy (Dict[Union[str, nn.Module], ModulePolicyDescription]): the policy to be updated
            target_key (Union[str, nn.Module]): the key of the policy to be updated
        """
        # convert to list
        if isinstance(description, SubModuleReplacementDescription):
            description = [description]

        # append or create a new description
        if target_key in self.policy:
            if self.policy[target_key].sub_module_replacement is None:
                self.policy[target_key].sub_module_replacement = description
            else:
                self.policy[target_key].sub_module_replacement.extend(
                    description)
        else:
            self.policy[target_key] = ModulePolicyDescription(
                sub_module_replacement=description)

    def append_or_create_method_replacement(
        self,
        description: Dict[str, Callable],
        target_key: Union[str, nn.Module],
    ) -> Dict[Union[str, nn.Module], ModulePolicyDescription]:
        r"""
        Append or create a new method replacement description to the policy for the given key.

        Args:
            description (Union[SubModuleReplacementDescription, List[SubModuleReplacementDescription]]): the submodule replacement description to be appended
            policy (Dict[Union[str, nn.Module], ModulePolicyDescription]): the policy to be updated
            target_key (Union[str, nn.Module]): the key of the policy to be updated
        """
        if target_key in self.policy:
            if self.policy[target_key].method_replacement is None:
                self.policy[target_key].method_replacement = description
            else:
                self.policy[target_key].method_replacement.extend(description)
        else:
            self.policy[target_key] = ModulePolicyDescription(
                method_replacement=description)

    def append_or_create_attribute_replacement(
        self,
        description: Dict[str, Callable],
        target_key: Union[str, nn.Module],
    ) -> Dict[Union[str, nn.Module], ModulePolicyDescription]:
        r"""
        Append or create a new method replacement description to the policy for the given key.

        Args:
            description (Union[SubModuleReplacementDescription, List[SubModuleReplacementDescription]]): the submodule replacement description to be appended
            policy (Dict[Union[str, nn.Module], ModulePolicyDescription]): the policy to be updated
            target_key (Union[str, nn.Module]): the key of the policy to be updated
        """
        if target_key in self.policy:
            if self.policy[target_key].attribute_replacement is None:
                self.policy[target_key].attribute_replacement = description
            else:
                self.policy[target_key].attribute_replacement.extend(
                    description)
        else:
            self.policy[target_key] = ModulePolicyDescription(
                attribute_replacement=description)

    def accelerate(self, model) -> None:
        r"""
        Replace the module according to the policy, and replace the module one by one

        Args:
            model (:class:`torch.nn.Module`): The model to shard
        """
        self.module_policy()
        self.module_replace = []
        for layer_cls, module_description in self.policy.items():
            self.replace_sub_module(
                model, layer_cls, module_description.sub_module_replacement)
            self._replace_method(
                model, layer_cls, module_description.method_replacement)
        print_rank_0(tabulate.tabulate(self.module_replace, headers=[
                     "old_layer", "new_layer"], tablefmt="psql"))
        return model

    def replace_sub_module(
        self,
        module: nn.Module,
        origin_cls: Union[str, nn.Module],
        sub_module_replacement: List[SubModuleReplacementDescription],
    ) -> None:
        r"""
        Reverse the replace layer operation
        """
        if not sub_module_replacement:
            return

        if (isinstance(origin_cls, str) and origin_cls == module.__class__.__name__) or (
            module.__class__ == origin_cls
        ):
            for description in sub_module_replacement:
                suffix = description.suffix
                target_module = description.target_module
                kwargs = {} if description.kwargs is None else description.kwargs

                assert target_module is not None, "target_module should not be None"

                native_sub_module = getattr_(module, suffix, ignore=True)

                assert not isinstance(
                    native_sub_module, target_module
                ), f"The module with suffix {suffix} has been replaced, please check the policy"

                # if it is None and we are allowed to ignore this module
                # just skip
                if description.ignore_if_not_exist and native_sub_module is None:
                    continue
                try:
                    replace_layer = target_module.from_native_module(
                        native_sub_module, **kwargs)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to replace {suffix} of type {native_sub_module.__class__.__qualname__}"
                        f" with {target_module.__qualname__} with the exception: {e}. "
                        "Please check your model configuration or sharding policy, you can set up an issue for us to help you as well."
                    )

                setattr_(module, suffix, replace_layer)
                self.module_replace.append(
                    [native_sub_module.__class__.__qualname__, target_module.__qualname__])

        for name, child in module.named_children():
            self.replace_sub_module(
                child,
                origin_cls,
                sub_module_replacement,
            )

    def _replace_method(self, module: nn.Module, origin_cls: Union[str, nn.Module], method_replacement: List[Dict[str, Callable]]):
        if not method_replacement:
            return

        if (isinstance(origin_cls, str) and origin_cls == module.__class__.__name__) or (
            module.__class__ == origin_cls
        ):
            for method in method_replacement:
                for method_name, new_method in method.items():
                    # bind the new method to the module
                    bound_method = MethodType(new_method, module)
                    setattr(module, method_name, bound_method)

        for name, child in module.named_children():
            self._replace_method(
                child,
                origin_cls,
                method_replacement,
            )

    def _replace_attr(
        self,
        module: nn.Module,
        origin_cls: Union[str, nn.Module],
        attr_replacement: List[Dict[str, Any]],
    ) -> None:
        r"""
        Replace the attribute of the layer

        Args:
            module (:class:`torch.nn.Module`): The object of layer to shard
            attr_replacement (Dict): The attribute dict to modify
        """
        if not attr_replacement:
            return

        if (isinstance(origin_cls, str) and origin_cls == module.__class__.__name__) or (
            module.__class__ == origin_cls
        ):
            for attr in attr_replacement:
                for module_attr, target_attr in attr.items():
                    native_attr = getattr_(module, module_attr, ignore=False)
                    if isinstance(native_attr, type):
                        replace_attr = target_attr.from_native_attr(
                            native_attr)
                        setattr_(module, module_attr,
                                 replace_attr, ignore=False)
                    else:
                        setattr_(module, module_attr,
                                 target_attr, ignore=False)

        for name, child in module.named_children():
            self._replace_attr(
                child,
                origin_cls,
                attr_replacement,
            )
