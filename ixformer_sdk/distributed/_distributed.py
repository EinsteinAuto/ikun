import warnings
from collections import defaultdict
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from ixformer._C import _distributed as cdist
from ixformer._C._distributed import comm
from ixformer._C._distributed.comm import (
    AllGatherAlgo,
    AllReduceAlgo,
    BroadcastAlgo,
    ReduceAlgo,
    ReduceOp,
    ReduceScatterAlgo,
    SendAlgo,
)
from ixformer.core.multi_level_cache import MultiLevelCache
from torch import Tensor
from torch.distributed import ProcessGroup

from ixformer.core import config

IxformerCommType = int
RecvAlgo = SendAlgo

_GROUP_TO_IXFC_COMM_CACHE = MultiLevelCache()
_IXFC_COMM_TO_GROUP_CACHE = MultiLevelCache()


def get_store(group: dist.ProcessGroup = None) -> dist.Store:
    if group is None:
        group = c10d._get_default_group()

    return c10d._pg_map[group][1]


class StoreWrapper(cdist.comm.C10dStoreWrapper):
    _GROUP_COUNT = defaultdict(dict)

    def __init__(self, group: ProcessGroup):
        super().__init__()
        self.store = get_store()

        ranks = dist.get_process_group_ranks(group)

        group_key = "_".join([str(r) for r in ranks])
        if group not in self._GROUP_COUNT[group_key]:
            self._GROUP_COUNT[group_key][group] = len(self._GROUP_COUNT[group_key])
        group_count = self._GROUP_COUNT[group_key][group]

        self.prefix = f"gid_{group_count}_" + group_key

    def _gen_unique_key(self, key):
        return f"{self.prefix}_{key}"

    def set(self, key: str, value: str):
        key = self._gen_unique_key(key)
        self.store.set(key, value)

    def get(self, key: str) -> str:
        key = self._gen_unique_key(key)
        self.store.wait([key])
        return self.store.get(key).decode("utf8")


def init_comm_with_store(group=None, shmsize: int = None):
    if group is None:
        group = c10d._get_default_group()

    world_size = dist.get_world_size(group=group)
    rank = dist.get_group_rank(group=group, global_rank=dist.get_rank())

    if shmsize is None:
        shmsize = config.IXFORMER_COMM_SHM_SIZE

    store_wrapper = StoreWrapper(group=group)
    ixfc_comm = cdist.comm.init_communicator_by_store(
        store=store_wrapper, world_size=world_size, rank=rank, max_shm_mem_size=shmsize
    )

    _GROUP_TO_IXFC_COMM_CACHE.set(group, ixfc_comm)
    _IXFC_COMM_TO_GROUP_CACHE.set(ixfc_comm, group)
    return ixfc_comm


_sub_store = None


def create_nccl_unique_id(addr: str, port: str, world_size: int, rank: int):
    global _sub_store
    _sub_store = dist.TCPStore(
        host_name=addr, port=int(port), world_size=world_size, is_master=rank == 0
    )
    store_key = "ncclUniqueId"
    if rank == 0:
        commid = cdist.comm.create_nccl_unique_id()
        _sub_store.set(store_key, commid)
    else:
        _sub_store.wait([store_key])
        commid = _sub_store.get(store_key).decode("utf8")

    return commid


def init_comm_with_eth(
    addr: str, port: str, world_size: int, rank: int, shmsize: int = None
):
    commid = create_nccl_unique_id(addr, port, world_size=world_size, rank=rank)
    return cdist.comm.init_communicator_by_nccl_id(commid, world_size, rank, shmsize)


def _check_group(group: Optional[ProcessGroup] = None):
    if group is None:
        group = c10d._get_default_group()

    if isinstance(group, ProcessGroup):
        ixfc_comm = _GROUP_TO_IXFC_COMM_CACHE.get(group, None)
        if ixfc_comm is None:
            return init_comm_with_store(group)
        return ixfc_comm

    return group


def get_comm_group_stream(group: Optional[ProcessGroup] = None):
    group = _check_group(group)
    return comm.get_comm_group_stream(group)


def set_comm_group_stream(stream: int, group: Optional[ProcessGroup] = None):
    group = _check_group(group)
    return comm.set_comm_group_stream(group, stream)


def get_group_rank(group: Optional[ProcessGroup], global_rank) -> int:
    """将 global rank 映射到 group 中的相对 rank"""
    if isinstance(group, IxformerCommType):
        _pg = _IXFC_COMM_TO_GROUP_CACHE.get(group, None)
        if _pg is None:
            return global_rank
        else:
            group = _IXFC_COMM_TO_GROUP_CACHE.get(group)

    if group is None:
        group = c10d._get_default_group()

    return dist.get_group_rank(group, global_rank)


def get_global_rank(group: Optional[ProcessGroup], group_rank: int) -> int:
    """将一个 group rank 映射到 global rank"""
    if group is None:
        group = c10d._get_default_group()
    return c10d.get_global_rank(group, group_rank)


def get_process_group_ranks(group: Optional[ProcessGroup] = None) -> List[int]:
    """获取 Group 的 global ranks"""
    if group is None:
        group = c10d._get_default_group()
    return c10d.get_process_group_ranks(group)


def new_group(ranks: List[int] = None, shmsize=None, *args, **kwargs):
    """通过 global ranks 去创建一个通讯组"""
    group = c10d.new_group(ranks, *args, **kwargs)

    if ranks is None:
        ranks = dist.get_process_group_ranks(group)

    if get_rank() in ranks:
        init_comm_with_store(group=group, shmsize=shmsize)
    return group


def new_subgroups_by_enumeration(
    ranks_per_subgroup_list, shmsize=None, *args, **kwargs
) -> Tuple[ProcessGroup, List[ProcessGroup]]:
    """
    通过一组 global ranks 去创建通讯组

    :param ranks_per_subgroup_list: global ranks
    :return: 返回当前 rank 所在的通讯组 和 新的 subgroups
    """
    self_group, other_group = c10d.new_subgroups_by_enumeration(
        ranks_per_subgroup_list, *args, **kwargs
    )
    init_comm_with_store(self_group, shmsize=shmsize)
    return self_group, other_group


def destroy_process_group(group: Optional[ProcessGroup] = None):
    """销毁 Group"""
    if group is None:
        group = c10d._get_default_group()
    ixfc_comm = _GROUP_TO_IXFC_COMM_CACHE.get(group, None)

    if ixfc_comm is None:
        dist.destroy_process_group(group)
    else:
        comm.destroy(ixfc_comm)
        dist.destroy_process_group(group)


def get_rank(group: Optional[ProcessGroup] = None) -> int:
    """获取当前进程的 Rank，如果 group 是 null，那么返回的是 Global Rank, 否则返回的相对的 Rank，即在当前组中的 rank"""
    return c10d.get_rank(group)


def get_world_size(group: Optional[ProcessGroup] = None) -> int:
    """获取 Group 中的成员大小"""
    return c10d.get_world_size(group)


def barrier(group: Optional[ProcessGroup] = None, use_comm_stream: bool = False):
    """同步 Group 中的 rank"""
    group = _check_group(group)
    comm.barrier(group, use_comm_stream)


def isend(
    tensor: Tensor,
    dst: int,
    group: Optional[ProcessGroup] = None,
    use_comm_stream: bool = False,
):
    dst = get_group_rank(group, dst)
    group = _check_group(group)
    return comm.send(group, tensor, dst, use_comm_stream, SendAlgo.kNone)


def send(*args, **kwargs):
    warnings.warn("not support sync mode, as async to call.")
    return isend(*args, **kwargs)


def irecv(
    tensor: torch.Tensor,
    src: int,
    group: Optional[ProcessGroup] = None,
    use_comm_stream: bool = False,
):
    src = get_group_rank(group, src)
    group = _check_group(group)
    return comm.recv(group, tensor, src, use_comm_stream, SendAlgo.kNone)


def recv(*args, **kwargs):
    warnings.warn("not support sync mode, as async to call.")
    return irecv(*args, **kwargs)


def point_to_point(
    tensor: Tensor,
    src: int,
    dst: int,
    group: Optional[ProcessGroup] = None,
    use_comm_stream: bool = False,
):
    """在 src rank 发送 tensor，在 dst_rank 上接收数据到 tensor 中"""
    src = get_group_rank(group, src)
    dst = get_group_rank(group, dst)
    group = _check_group(group)
    return comm.p2p(group, tensor, src, dst, use_comm_stream)


def reduce(
    tensor,
    root: int,
    op=ReduceOp.SUM,
    group: Optional[ProcessGroup] = None,
    async_op=False,
    out: Tensor = None,
    use_comm_stream: bool = False,
):
    """
    Example:
        ixf_tensor = torch.tensor([1], device="cuda")
        ixfd.reduce(ixf_tensor, 1, async_op=True)
        print("rank {rank}:", ixf_tensor)

        # output
        rank 0:  tensor([1], device='cuda:0')
        rank 1:  tensor([4], device='cuda:1')
        rank 2:  tensor([1], device='cuda:2')
        rank 3:  tensor([1], device='cuda:3')
    """

    if not async_op:
        raise RuntimeError("Not support sync operation now.")

    if out is None:
        out = tensor

    root = get_group_rank(group, root)
    group = _check_group(group)
    return comm.reduce(group, tensor, out, op, root, use_comm_stream, ReduceAlgo.kNone)


def broadcast(
    tensor: Tensor,
    src: int,
    group: Optional[ProcessGroup] = None,
    async_op=False,
    out: Tensor = None,
    use_comm_stream: bool = False,
):
    """
    Example:
        ixf_tensor = torch.tensor([rank], device="cuda")
        ixfd.broadcast(ixf_tensor, 1, async_op=True)
        print("rank {rank}: ", ixf_tensor)

        # output
        rank 0:  tensor([1], device='cuda:0')
        rank 1:  tensor([1], device='cuda:1')
        rank 2:  tensor([1], device='cuda:2')
        rank 3:  tensor([1], device='cuda:3')
    """
    if not async_op:
        raise RuntimeError("Not support sync operation now.")

    if out is None:
        out = tensor

    src = get_group_rank(group, src)
    group = _check_group(group)
    return comm.broadcast(group, tensor, out, src, use_comm_stream, BroadcastAlgo.kNone)


def reduce_scatter_tensor(
    output: Tensor,
    input: Tensor,
    op=ReduceOp.SUM,
    group: Optional[ProcessGroup] = None,
    async_op=False,
    use_comm_stream: bool = False,
):
    """
    Example:
        ixf_tensor_out = torch.zeros(2, dtype=torch.int64, device="cuda")
        tensor_in = torch.arange(world_size * 2, dtype=torch.int64, device="cuda")
        # tensor_in: tensor([0, 1, 2, 3, 4, 5, 6, 7], device='cuda:0')

        ixfd.reduce_scatter_tensor(ixf_tensor_out, tensor_in, async_op=True)
        print("rank {rank}:", ixf_tensor_out)

        # output
        rank 0:  tensor([0, 4], device='cuda:0')
        rank 1:  tensor([ 8, 12], device='cuda:1')
        rank 2:  tensor([16, 20], device='cuda:2')
        rank 3:  tensor([24, 28], device='cuda:3')
    """
    if not async_op:
        raise RuntimeError("Not support sync operation now.")

    group = _check_group(group)
    return comm.reduce_scatter(
        group, input, output, op, use_comm_stream, ReduceScatterAlgo.kNone
    )


def all_reduce(
    tensor: Tensor,
    op=ReduceOp.SUM,
    group: Optional[ProcessGroup] = None,
    async_op=False,
    out: Tensor = None,
    algo: AllReduceAlgo = AllReduceAlgo.kNone,
    use_comm_stream: bool = False,
):
    """
    Args:
        tensor: inpute tensor
        op: ReduceOp: SUM, MIN or MAX
        group: communicator group
        async_op: ixformer support async mode
        out: output tensor
        algo: AllReduce Algo: Auto, Quant, QuantL1, QuantL2, NCCL, Ring, AllGatherSum, BroadcastSum
        use_comm_stream: ixformer support set communication stream by ixformer.distributed.set_comm_group_stream,
          if true, submit the kernels of communication to communication stream,
          if false, use current stream by torch.cuda.current_stream
    Returns: out

    Example:
        >>> # All tensors below are of torch.int64 type.
        >>> # We have 2 process groups, 2 ranks.
        >>> tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
        >>> tensor
        tensor([1, 2]) # Rank 0
        tensor([3, 4]) # Rank 1
        >>> ixfd.all_reduce(tensor, op=ReduceOp.SUM, async_op=True)
        >>> tensor
        tensor([4, 6]) # Rank 0
        tensor([4, 6]) # Rank 1
    """
    if not async_op:
        raise RuntimeError("Not support sync operation now.")

    group = _check_group(group)

    if out is None:
        out = tensor

    comm.all_reduce(
        group,
        tensor,
        out,
        op,
        use_comm_stream=use_comm_stream,
        algo=algo,
    )


def all_gather_into_tensor(
    output: Tensor,
    input: Tensor,
    group: Optional[ProcessGroup] = None,
    async_op=False,
    use_comm_stream: bool = False,
):
    """
    Example:
        tensor_in = torch.arange(2, dtype=torch.int64, device="cuda") + 1 + 2 * rank
            rank 0:  tensor in:  tensor([1, 2], device='cuda:0')
            rank 1:  tensor in:  tensor([3, 4], device='cuda:1')
            rank 2:  tensor in:  tensor([5, 6], device='cuda:2')
            rank 3:  tensor in:  tensor([7, 8], device='cuda:3')

        ixf_tensor_out = torch.zeros(world_size * 2, dtype=torch.int64, device="cuda")
        ixfd.all_gather_into_tensor(ixf_tensor_out, tensor_in, async_op=True)
        print("rank {rank}:", ixf_tensor_out)

        # output:
        rank 0:  tensor([1, 2, 3, 4, 5, 6, 7, 8], device='cuda:0')
        rank 1:  tensor([1, 2, 3, 4, 5, 6, 7, 8], device='cuda:1')
        rank 2:  tensor([1, 2, 3, 4, 5, 6, 7, 8], device='cuda:2')
        rank 3:  tensor([1, 2, 3, 4, 5, 6, 7, 8], device='cuda:3')
    """
    if not async_op:
        raise RuntimeError("Not support sync operation now.")

    group = _check_group(group)
    return comm.all_gather(
        group, input, output, use_comm_stream, algo=AllGatherAlgo.kNone
    )


def gather(
    tensor,
    gather_list=None,
    dst=0,
    group: Optional[ProcessGroup] = None,
    async_op=False,
    use_comm_stream: bool = False,
):
    """
    Example:
        >>> # We have 2 process groups, 2 ranks.
        >>> tensor = torch.tensor(rank+1,dtype=torch.float32).cuda()
        >>> tensor
        tensor(1.) # Rank 0
        tensor(2.) # Rank 1
        >>> gather_list = [torch.zeros(1).cuda() for _ in range(rank)] if rank == dst else None
        >>> gather_list
        [tensor([0,]),tensor([1,])] # Rank 0
        None                        # Rank 1
        ixfd.gather(tensor,gather_list,0,async_op=True)
        >>> gather_list
        [tensor([1.]),tensor([2.])] # Rank 0
        None                        # Rank 1
    """
    gather_list = gather_list if gather_list is not None else []
    if not async_op:
        raise RuntimeError("Not support sync operation now.")

    dst = get_group_rank(group, dst)
    group = _check_group(group)
    return comm.gather(group, tensor, gather_list, dst, use_comm_stream)
