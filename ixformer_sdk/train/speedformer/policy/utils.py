from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union
import re
import torch
import torch.nn as nn


@dataclass
class SubModuleReplacementDescription:
    r"""
    Describe how a submodule will be replaced

    Args:
        suffix (str): used to get the submodule object
        target_module (ParallelModule): specifies the module class used to replace to submodule
        kwargs (Dict[str, Any]): the dictionary used to pass extra arguments to the `ParallelModule.from_native_module` method.
        ignore_if_not_exist (bool): if the submodule does not exist, ignore it or raise an exception
    """

    suffix: str
    target_module: nn.Module
    kwargs: Dict[str, Any] = None
    ignore_if_not_exist: bool = False


@dataclass
class ModulePolicyDescription:
    "copy from colossalai, for now sub_module_replacement and method_replacement is used"
    r"""
    Describe how the attributes and parameters will be transformed in a policy.

    Args:
        attribute_replacement (Dict[str, Any]): key is the attribute name, value is the attribute value after sharding
        param_replacement (List[Callable]): a list of functions to perform in-place param replacement. The function
                    must receive only one arguments: module. One example is

                    ```python
                    def example_replace_weight(module: torch.nn.Module):
                        weight = module.weight
                        new_weight = shard_rowwise(weight, process_group)
                        module.weight = torch.nn.Parameter(new_weight)
                    ```
        sub_module_replacement (List[SubModuleReplacementDescription]): each element in the list is a SubModuleReplacementDescription
                    object which specifies the module to be replaced and the target module used to replacement.
        method_replace (Dict[str, Callable]): key is the method name, value is the method for replacement
    """

    attribute_replacement: List[Dict[str, Any]] = None
    param_replacement: List[Callable] = None
    sub_module_replacement: List[SubModuleReplacementDescription] = None
    method_replacement: List[Dict[str, Callable]] = None


def getattr_(obj, attr: str, ignore: bool = False):
    r"""
    Get the object's multi sublevel attr

    Args:
        obj (object): The object to set
        attr (str): The multi level attr to set
        ignore (bool): Whether to ignore when the attr doesn't exist
    """

    attrs = attr.split(".")
    for a in attrs:
        try:
            obj = get_obj_list_element(obj, a)
        except AttributeError:
            if ignore:
                return None
            raise AttributeError(
                f"Object {obj.__class__.__name__} has no attribute {attr}")
    return obj


def get_obj_list_element(obj, attr: str):
    r"""
    Get the element of the list in the object

    If the attr is a normal attribute, return the attribute of the object.
    If the attr is a index type, return the element of the index in the list, like `layers[0]`.

    Args:
        obj (Object): The object to get
        attr (str): The suffix of the attribute to get

    """
    re_pattern = r"\[\d+\]"
    prog = re.compile(re_pattern)
    result = prog.search(attr)
    if result:
        matched_brackets = result.group()
        matched_index = matched_brackets.replace("[", "")
        matched_index = matched_index.replace("]", "")
        attr_ = attr.replace(matched_brackets, "")
        container_obj = getattr(obj, attr_)
        obj = container_obj[int(matched_index)]
    else:
        obj = getattr(obj, attr)
    return obj


def setattr_(obj, attr: str, value, ignore: bool = False):
    r"""
    Set the object's multi sublevel attr to value, if ignore, ignore when it doesn't exist

    Args:
        obj (object): The object to set
        attr (str): The multi level attr to set
        value (Any): The value to set
        ignore (bool): Whether to ignore when the attr doesn't exist
    """

    attrs = attr.split(".")
    for a in attrs[:-1]:
        try:
            obj = get_obj_list_element(obj, a)
        except AttributeError:
            if ignore:
                return
            raise AttributeError(
                f"Object {obj.__class__.__name__} has no attribute {attr}")
    set_obj_list_element(obj, attrs[-1], value)


def set_obj_list_element(obj, attr: str, value):
    r"""
    Set the element to value of a list object

    It used like set_obj_list_element(obj, 'layers[0]', new_layer), it will set obj.layers[0] to value

    Args:
        obj (object): The object to set
        attr (str): the string including a list index like `layers[0]`
    """
    re_pattern = r"\[\d+\]"
    prog = re.compile(re_pattern)
    result = prog.search(attr)
    if result:
        matched_brackets = result.group()
        matched_index = matched_brackets.replace("[", "")
        matched_index = matched_index.replace("]", "")
        attr_ = attr.replace(matched_brackets, "")
        container_obj = getattr(obj, attr_)
        container_obj[int(matched_index)] = value
    else:
        setattr(obj, attr, value)


def print_rank_0(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)
