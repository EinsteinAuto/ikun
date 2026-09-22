import abc
import bisect
import functools
import itertools
import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

import ixformer.distributed as ixfd
from ixformer.utils.benchmark.cuda_benchmark import Functor, cuda_benchmark


def sync_ranks_metric(value, group=None):
    if not isinstance(value, (torch.Tensor, int, float)):
        raise RuntimeError(
            f"Invalid metric value, expect `Tensor`, `int`, or `float` type, but got {value}."
        )

    if torch.is_tensor(value):
        value = value.to("cuda")
    else:
        value = torch.tensor([value], dtype=torch.float, device="cuda")

    dist.broadcast(value, src=0, group=group)
    return value.cpu().item()


class AutotuningFinder(object):
    def freeze(self):
        pass

    @abc.abstractmethod
    def get(self, key) -> Callable:
        pass

    @abc.abstractmethod
    def set(self, *args, **kwargs):
        pass


class BasedKeyFinder(AutotuningFinder):
    def __init__(self):
        self._key_to_value: Dict[Any, Callable] = dict()

    def get(self, key, **kwargs) -> Callable:
        if "default" in kwargs:
            return self._key_to_value.get(key, kwargs["default"])
        return self._key_to_value[key]

    def set(self, key, value):
        self._key_to_value[key] = value

    def containe(self, key):
        return key in self._key_to_value


class TreeNode:
    def __init__(self):
        self.nodes: List[Union[Any, TreeNode]] = list()
        self.key_to_nodes: Dict[Any, TreeNode] = dict()

    def add(self, key, value):
        if isinstance(key, (tuple, list)):
            if len(key) == 1:
                self.insert_value(key[0], value)
            else:
                self.recurse_add_node(key, value)
        else:
            self.insert_value(key, value)

    def insert_value(self, key, value):
        self.nodes.append((key, value))
        self.key_to_nodes[key] = value

    def recurse_add_node(self, key, value):
        if key[0] in self.key_to_nodes:
            node = self.key_to_nodes[key[0]]
        else:
            node = TreeNode()
            self.key_to_nodes[key[0]] = node
            self.insert_value(key[0], node)

        node.add(key[1:], value)

    def sort(self):
        self.nodes.sort(key=lambda x: x[0])
        for _, node in self.nodes:
            if isinstance(node, TreeNode):
                node.sort()

    def find(self, key):
        is_list_key = isinstance(key, (tuple, list))
        if not is_list_key:
            key = (key,)

        num_querys = len(key)
        node = self
        for key_idx in range(num_querys):
            query_key = key[key_idx]
            idx = bisect.bisect_left(node.nodes, (query_key,)) - 1
            if idx <= 0:
                node = node.nodes[0][1]
            elif idx >= len(node.nodes):
                node = node.nodes[-1][1]
            else:
                node = node.nodes[idx][1]

        return node

    def show(self, indent=0):
        for k, node in self.nodes:
            print(" " * indent, end="")
            if isinstance(node, TreeNode):
                print(f"key: {k}")
                node.show(indent=indent + 4)
            else:
                print(f"key: {k}, node: {node}")


class BasedRangeFinder(AutotuningFinder):
    def __init__(self):
        super().__init__()

        self.tree = TreeNode()
        self._found_cache: Dict[Any, Callable] = dict()

    def freeze(self):
        self.tree.sort()

    def get(self, key) -> Callable:
        value = self._found_cache.get(key, None)
        if value is not None:
            return value

        value = self.tree.find(key)
        self._found_cache[key] = value
        return value

    def set(self, key, value):
        self.tree.add(key, value)


class OperatorAutotuning(object):
    def __init__(self, num_repeated=5, num_warmup=3, dist_barrier=False):
        self.num_repeated = num_repeated
        self.num_warmup = num_warmup
        self.dist_barrier = dist_barrier

    @abc.abstractmethod
    def operators(self):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.exec_best_operator(args, kwargs)

    @abc.abstractmethod
    def exec_best_operator(self, args, kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def autotuning(self, *args, **kwargs):
        raise NotImplementedError()

    def perf_best_operator(self, *args, **kwargs) -> Callable:
        best_operator = None
        best_operator_time = float("inf")

        for idx, operator in enumerate(self.operators()):
            op_time = self.perf_operator_time(operator, *args, **kwargs)
            if op_time < best_operator_time:
                best_operator = operator
                best_operator_time = op_time

            # print(operator, op_time)

        return best_operator

    def perf_operator_time(self, op: Callable, *args, **kwargs) -> float:
        fn = Functor(op, *args, **kwargs)
        time = cuda_benchmark(fn, self.num_repeated, self.num_warmup, self.dist_barrier)
        return time.gpu


class OperatorRuntimeAutotuning(OperatorAutotuning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.operator_finder = BasedKeyFinder()

    @abc.abstractmethod
    def get_operator_key(self, *args, **kwargs):
        raise NotImplementedError()

    def exec_best_operator(self, args, kwargs):
        key = self.get_operator_key(*args, **kwargs)
        operator = self.operator_finder.get(key, default=None)
        if operator is None:
            operator = self.perf_best_operator(*args, **kwargs)
            self.operator_finder.set(key, operator)

        return operator(*args, **kwargs)


class OperatorPreBaseRangeAutotuning(OperatorAutotuning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.operator_finder = BasedRangeFinder()
        self._finished_autotuning = False

    @abc.abstractmethod
    def get_operator_key(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def generate_operator_inputs(self) -> Iterable[Tuple[Tuple, Dict]]:
        raise NotImplementedError()

    def exec_best_operator(self, args, kwargs):
        if not self._finished_autotuning:
            self.autotuning()

        best_op = self.operator_finder.get(self.get_operator_key(*args, **kwargs))
        return best_op(*args, **kwargs)

    def autotuning(self):
        for op_args, op_kwargs in self.generate_operator_inputs():
            best_op = self.perf_best_operator(*op_args, **op_kwargs)
            self.operator_finder.set(
                self.get_operator_key(*op_args, **op_kwargs), best_op
            )

        self.operator_finder.freeze()
        self._finished_autotuning = True
        # self.operator_finder.tree.show()
