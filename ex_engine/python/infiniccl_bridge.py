"""
infiniccl_bridge — dlopen libinfiniccl.so and expose allreduce for BI-V100 TP
"""

import ctypes
import os
import sys

import torch
import torch.distributed as dist


class InfiniCclUniqueId(ctypes.Structure):
    """128-byte struct, passed BY VALUE to infinicclCommInitRank"""
    _fields_ = [('internal', ctypes.c_char * 128)]


_lib = None
_comm = None
_rank = -1
_world_size = -1


def _find_and_load():
    global _lib
    import vllm
    vllm_root = os.path.dirname(vllm.__file__)
    search = [
        os.path.join(vllm_root, "..", "ex_engine", "prebuilt", "libinfiniccl.so"),
        os.path.join(vllm_root, "libinfiniccl.so"),
        os.path.join(os.path.dirname(__file__), "..", "prebuilt", "libinfiniccl.so"),
        "/usr/local/corex/lib64/libinfiniccl.so",
    ]

    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    ctypes.CDLL(os.path.join(torch_lib, "libtorch_cuda.so"), mode=ctypes.RTLD_GLOBAL)

    for path in search:
        path = os.path.abspath(path)
        if os.path.isfile(path):
            lib = ctypes.CDLL(path)
            assert hasattr(lib, "infinicclAllReduce"), f"{path} missing infinicclAllReduce"

            # set argtypes — infinicclUniqueId is 128-byte struct BY VALUE
            lib.infinicclGetUniqueId.argtypes = [ctypes.POINTER(InfiniCclUniqueId)]
            lib.infinicclGetUniqueId.restype = ctypes.c_int
            lib.infinicclCommInitRank.argtypes = [
                ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, InfiniCclUniqueId, ctypes.c_int
            ]
            lib.infinicclCommInitRank.restype = ctypes.c_int
            lib.infinicclAllReduce.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p
            ]
            lib.infinicclAllReduce.restype = ctypes.c_int

            _lib = lib
            print(f"[infiniccl] loaded from {path}", file=sys.stderr, flush=True)
            return lib

    raise FileNotFoundError(f"libinfiniccl.so not found in {search}")


def init_comm(rank, world_size):
    global _comm, _rank, _world_size
    if _lib is None:
        _find_and_load()

    _rank = rank
    _world_size = world_size
    torch.cuda.set_device(rank)

    uid = InfiniCclUniqueId()
    if rank == 0:
        ret = _lib.infinicclGetUniqueId(ctypes.byref(uid))
        assert ret == 0, f"infinicclGetUniqueId failed: {ret}"

    # broadcast uid via TP group
    from vllm.distributed.parallel_state import get_tp_group
    tp = get_tp_group()
    # bytes(uid) gives full 128 bytes; uid.internal truncates at first \x00
    uid_tensor = torch.tensor(list(bytes(uid)), dtype=torch.uint8).cuda()
    dist.broadcast(uid_tensor, src=tp.ranks[0], group=tp.device_group)
    ctypes.memmove(ctypes.byref(uid), bytes(uid_tensor.cpu().tolist()), 128)

    print(f"[infiniccl] CommInitRank rank={rank}/{world_size} device={torch.cuda.current_device()}",
          file=sys.stderr, flush=True)

    comm_ptr = ctypes.c_void_p()
    ret = _lib.infinicclCommInitRank(ctypes.byref(comm_ptr), world_size, uid, rank)
    assert ret == 0, f"infinicclCommInitRank failed: {ret} rank={rank}"

    _comm = comm_ptr
    print(f"[infiniccl] comm initialized rank={rank}/{world_size}",
          file=sys.stderr, flush=True)

    # register for cleanup on signal — prevents GPU driver corruption
    from ex_engine.python.nccl_cleanup import install, register_comm
    install()
    register_comm(_comm)


def infiniccl_allreduce(tensor):
    assert _comm is not None, "call init_comm first"

    if tensor.dtype == torch.float16:
        dtype_enum = 8
    elif tensor.dtype == torch.float32:
        dtype_enum = 10
    elif tensor.dtype == torch.bfloat16:
        dtype_enum = 9
    else:
        raise TypeError(f"unsupported dtype: {tensor.dtype}")

    stream = torch.cuda.current_stream().cuda_stream

    ret = _lib.infinicclAllReduce(
        ctypes.c_void_p(tensor.data_ptr()),
        ctypes.c_void_p(tensor.data_ptr()),
        tensor.numel(),
        dtype_enum,
        0,
        _comm,
        ctypes.c_void_p(stream),
    )
    assert ret == 0, f"infinicclAllReduce failed: {ret}"
    return tensor
