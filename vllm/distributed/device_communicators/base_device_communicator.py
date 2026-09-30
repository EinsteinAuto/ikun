# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
import ixformer.distributed as ixfd
from ixformer.contrib.torch.extension.ixformer_torch.distributed import create_ixformer_group_from_pg
import os
import sys


class DeviceCommunicatorBase:
    """
    Base class for device-specific communicator.
    It can use the `cpu_group` to initialize the communicator.
    If the device has PyTorch integration (PyTorch can recognize its
    communication backend), the `device_group` will also be given.
    """

    def __init__(self,
                 cpu_group: ProcessGroup,
                 device: Optional[torch.device] = None,
                 device_group: Optional[ProcessGroup] = None,
                 unique_name: str = ""):
        self.device = device or torch.device("cpu")
        self.cpu_group = cpu_group
        self.device_group = device_group
        self.unique_name = unique_name
        self.rank = dist.get_rank(cpu_group)
        self.world_size = dist.get_world_size(cpu_group)
        self.ranks = dist.get_process_group_ranks(cpu_group)
        self.global_rank = dist.get_rank()
        self.global_world_size = dist.get_world_size()
        self.rank_in_group = dist.get_group_rank(self.cpu_group,
                                                 self.global_rank)
        self.use_vllm_comm = os.environ.get("VLLM_FORCE_NCCL_COMM",None) not in ["1", "Y", "y"]

        # Convert torch ProcessGroup to ixformer NcclGroup for ixfd calls.
        if self.use_vllm_comm and device_group is not None:
            self.ixformer_group = create_ixformer_group_from_pg(device_group)
        else:
            self.ixformer_group = None

        if "pp" in unique_name:
            # pipeline parallel does not need custom allreduce
            use_custom_allreduce = False
        else:
            from vllm.distributed.parallel_state import (
                _ENABLE_CUSTOM_ALL_REDUCE)
            use_custom_allreduce = _ENABLE_CUSTOM_ALL_REDUCE
        self.use_custom_allreduce = use_custom_allreduce
         # lazy import to avoid documentation build error
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce)
        self.ca_comm: Optional[CustomAllreduce] = None
        if use_custom_allreduce and self.world_size > 1:
            # Initialize a custom fast all-reduce implementation.
            self.ca_comm = CustomAllreduce(
                group=self.cpu_group,
                device=self.device,
            )   

        self._use_infiniccl = (
            os.environ.get("BI100_FUSED_LINEAR_ALLREDUCE", "0") == "1"
            and "tp" in unique_name
            and self.world_size > 1
        )
        self._infiniccl_ready = False

    def _init_infiniccl(self):
        if self._infiniccl_ready:
            return True
        try:
            import ctypes
            for p in ['/home/ikun', '/workspace']:
                if p not in sys.path:
                    sys.path.insert(0, p)
            from ex_engine.python.infiniccl_bridge import (
                _find_and_load, _lib, _comm, InfiniCclUniqueId,
            )
            import ex_engine.python.infiniccl_bridge as bridge

            if bridge._comm is not None:
                self._infiniccl_ready = True
                return True

            if bridge._lib is None:
                _find_and_load()

            uid = InfiniCclUniqueId()
            if self.rank_in_group == 0:
                ret = bridge._lib.infinicclGetUniqueId(ctypes.byref(uid))
                assert ret == 0, f"infinicclGetUniqueId failed: {ret}"

            with torch.inference_mode(False):
                uid_tensor = torch.tensor(list(bytes(uid)), dtype=torch.uint8)
            dist.broadcast(uid_tensor, src=self.ranks[0], group=self.cpu_group)
            ctypes.memmove(ctypes.byref(uid), bytes(uid_tensor.tolist()), 128)

            comm_ptr = ctypes.c_void_p()
            ret = bridge._lib.infinicclCommInitRank(
                ctypes.byref(comm_ptr), self.world_size, uid, self.rank_in_group)
            assert ret == 0, f"infinicclCommInitRank failed: {ret}"

            bridge._comm = comm_ptr
            bridge._rank = self.rank_in_group
            bridge._world_size = self.world_size
            self._infiniccl_ready = True
            print(f"[infiniccl] comm ready rank={self.rank_in_group}/{self.world_size}",
                  file=sys.stderr, flush=True)
            return True
        except Exception as e:
            print(f"[infiniccl] FATAL: init failed: {e}",
                  file=sys.stderr, flush=True)
            print(f"[infiniccl] refusing to fallback to nccl (driver corruption risk)",
                  file=sys.stderr, flush=True)
            os._exit(1)

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        
        if self.world_size == 1:
            return input_

        if self._use_infiniccl:
            if self._infiniccl_ready or self._init_infiniccl():
                from ex_engine.python.infiniccl_bridge import infiniccl_allreduce
                return infiniccl_allreduce(input_)
        
        if self.use_vllm_comm:
            ixfd.all_reduce(input_, group=self.ixformer_group, async_op=True)
        else:
            torch.distributed.all_reduce(input_, group=self.device_group)   
        return input_

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()
        input_size = input_.size()
        # NOTE: we have to use concat-style all-gather here,
        # stack-style all-gather has compatibility issues with
        # torch.compile . see https://github.com/pytorch/pytorch/issues/138795
        output_size = (input_size[0] * self.world_size, ) + input_size[1:]
        # Allocate output tensor.
        output_tensor = torch.empty(output_size,
                                    dtype=input_.dtype,
                                    device=input_.device)
        # All-gather.
        if self.use_vllm_comm:
            ixfd.all_gather_into_tensor(output_tensor,
                                        input_,
                                        group=self.ixformer_group,
                                        async_op=True)
        else:
            torch.distributed.all_gather_into_tensor(output_tensor,
                                                 input_,
                                                 group=self.device_group)
        # Reshape
        output_tensor = output_tensor.reshape((self.world_size, ) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(input_size[:dim] +
                                              (self.world_size *
                                               input_size[dim], ) +
                                              input_size[dim + 1:])
        return output_tensor

    def gather(self,
               input_: torch.Tensor,
               dst: int = 0,
               dim: int = -1) -> Optional[torch.Tensor]:
        """
        NOTE: We assume that the input tensor is on the same device across
        all the ranks.
        NOTE: `dst` is the local rank of the destination rank.
        """
        world_size = self.world_size
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Allocate output tensor.
        if self.rank_in_group == dst:
            gather_list = [torch.empty_like(input_) for _ in range(world_size)]
        else:
            gather_list = None
        # Gather.
        if self.use_vllm_comm:
            ixfd.gather(input_,
                        gather_list,
                        dst=self.ranks[dst],
                        group=self.ixformer_group,
                        async_op=True)
        else:
            torch.distributed.gather(input_,
                                    gather_list,
                                    dst=self.ranks[dst],
                                    group=self.device_group)
        if self.rank_in_group == dst:
            output_tensor = torch.cat(gather_list, dim=dim)
        else:
            output_tensor = None
        return output_tensor

    def send(self, tensor: torch.Tensor, dst: Optional[int] = None) -> None:
        """Sends a tensor to the destination rank in a non-blocking way"""
        """NOTE: `dst` is the local rank of the destination rank."""
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        if self.use_vllm_comm:
                ixfd.send(tensor, self.ranks[dst], self.ixformer_group)
        else:
            torch.distributed.send(tensor, self.ranks[dst], self.device_group)

    def recv(self,
             size: torch.Size,
             dtype: torch.dtype,
             src: Optional[int] = None) -> torch.Tensor:
        """Receives a tensor from the source rank."""
        """NOTE: `src` is the local rank of the source rank."""
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size

        tensor = torch.empty(size, dtype=dtype, device=self.device)
        if self.use_vllm_comm:
            ixfd.recv(tensor, self.ranks[src], self.ixformer_group)
        else:
            torch.distributed.recv(tensor, self.ranks[src], self.device_group)
        return tensor

    def destroy(self):
        pass
