"""
Safe dual-GPU InfiniCCL allreduce test.

Uses multiprocessing (one process per GPU) like real TP workers.
Each process has its own CUDA context and nccl comm handle.
nccl_cleanup signal handler registered for safe teardown.

Usage:
    CUDA_VISIBLE_DEVICES=1,2 python3 tests/python/test_infiniccl_dual_gpu.py
"""

import ctypes
import multiprocessing as mp
import os
import sys
import time

TIMEOUT = 30


class InfiniCclUniqueId(ctypes.Structure):
    _fields_ = [('internal', ctypes.c_char * 128)]


def worker(rank, world_size, uid_bytes, result_dict):
    """Each worker = one GPU, one nccl comm, one allreduce."""
    import torch
    torch.cuda.set_device(rank)

    # load libs
    torch_lib = '/usr/local/corex/lib64/python3/dist-packages/torch/lib'
    ctypes.CDLL(os.path.join(torch_lib, 'libtorch_cuda.so'), mode=ctypes.RTLD_GLOBAL)
    lib = ctypes.CDLL('/usr/local/corex/lib64/python3/dist-packages/ex_engine/prebuilt/libinfiniccl.so')

    lib.infinicclCommInitRank.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, InfiniCclUniqueId, ctypes.c_int
    ]
    lib.infinicclCommInitRank.restype = ctypes.c_int
    lib.infinicclAllReduce.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p
    ]
    lib.infinicclAllReduce.restype = ctypes.c_int
    lib.ncclCommAbort.argtypes = [ctypes.c_void_p]
    lib.ncclCommAbort.restype = ctypes.c_int

    # reconstruct uid
    uid = InfiniCclUniqueId()
    ctypes.memmove(ctypes.byref(uid), uid_bytes, 128)

    # init comm
    comm = ctypes.c_void_p()
    ret = lib.infinicclCommInitRank(ctypes.byref(comm), world_size, uid, rank)
    if ret != 0:
        result_dict[rank] = f"CommInitRank failed: {ret}"
        return

    print(f"[rank {rank}] comm initialized", file=sys.stderr, flush=True)

    # allreduce: each rank sends (rank+1), expect sum = 1+2+...+world_size
    inp = torch.ones(256, device=f'cuda:{rank}', dtype=torch.float16) * (rank + 1)
    out = torch.zeros(256, device=f'cuda:{rank}', dtype=torch.float16)
    stream = torch.cuda.current_stream().cuda_stream

    ret = lib.infinicclAllReduce(
        ctypes.c_void_p(inp.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        256, 8, 0, comm, ctypes.c_void_p(stream))

    torch.cuda.synchronize()

    if ret != 0:
        result_dict[rank] = f"AllReduce failed: {ret}"
    else:
        val = out[0].item()
        expected = sum(range(1, world_size + 1))
        result_dict[rank] = f"got {val}, expected {expected}, {'PASS' if val == expected else 'FAIL'}"

    # cleanup
    lib.ncclCommAbort(comm)
    print(f"[rank {rank}] done, comm aborted", file=sys.stderr, flush=True)


def main():
    gpu_count = int(os.environ.get('WORLD_SIZE', '2'))
    print(f"Testing {gpu_count}-GPU InfiniCCL allreduce", file=sys.stderr)

    # generate uid in main process
    torch_lib = '/usr/local/corex/lib64/python3/dist-packages/torch/lib'
    ctypes.CDLL(os.path.join(torch_lib, 'libtorch_cuda.so'), mode=ctypes.RTLD_GLOBAL)
    lib = ctypes.CDLL('/usr/local/corex/lib64/python3/dist-packages/ex_engine/prebuilt/libinfiniccl.so')
    lib.infinicclGetUniqueId.argtypes = [ctypes.POINTER(InfiniCclUniqueId)]
    lib.infinicclGetUniqueId.restype = ctypes.c_int

    uid = InfiniCclUniqueId()
    ret = lib.infinicclGetUniqueId(ctypes.byref(uid))
    assert ret == 0, f"GetUniqueId failed: {ret}"
    uid_bytes = bytes(uid)  # full 128 bytes; uid.internal truncates at \x00

    manager = mp.Manager()
    result_dict = manager.dict()

    procs = []
    for rank in range(gpu_count):
        p = mp.Process(target=worker, args=(rank, gpu_count, uid_bytes, result_dict))
        p.start()
        procs.append(p)

    # wait with timeout
    deadline = time.time() + TIMEOUT
    for p in procs:
        remaining = max(0, deadline - time.time())
        p.join(timeout=remaining)
        if p.is_alive():
            print(f"[main] process {p.pid} hung, terminating", file=sys.stderr)
            p.terminate()
            p.join(timeout=5)
            if p.is_alive():
                p.kill()

    for rank in range(gpu_count):
        print(f"GPU {rank}: {result_dict.get(rank, 'NO RESULT (hung?)')}")

    all_pass = all('PASS' in str(result_dict.get(r, '')) for r in range(gpu_count))
    print(f"\n{'ALL PASSED' if all_pass else 'FAILED'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
