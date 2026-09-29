import ctypes
import multiprocessing as mp
import os
import sys
import time

TIMEOUT = 60


class InfiniCclUniqueId(ctypes.Structure):
    _fields_ = [('internal', ctypes.c_char * 128)]


def worker(rank, world_size, result_dict):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['BI100_FUSED_LINEAR_ALLREDUCE'] = '1'

    import torch
    torch.cuda.set_device(rank)

    torch.distributed.init_process_group('gloo', rank=rank, world_size=world_size)
    print(f'[rank {rank}] gloo init ok', file=sys.stderr, flush=True)

    sys.path.insert(0, '/home/ikun')
    from vllm.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase

    comm = DeviceCommunicatorBase(
        cpu_group=torch.distributed.group.WORLD,
        device=torch.device(f'cuda:{rank}'),
        device_group=None,
        unique_name='tp_test',
    )
    print(f'[rank {rank}] communicator created, _use_infiniccl={comm._use_infiniccl}', file=sys.stderr, flush=True)

    t = torch.ones(256, device=f'cuda:{rank}', dtype=torch.float16) * (rank + 1)
    result = comm.all_reduce(t)
    torch.cuda.synchronize()

    val = result[0].item()
    expected = float(sum(range(1, world_size + 1)))
    status = 'PASS' if val == expected else 'FAIL'
    print(f'[rank {rank}] allreduce val={val} expected={expected} {status}', file=sys.stderr, flush=True)
    result_dict[rank] = (val, expected, status)

    torch.distributed.destroy_process_group()


def main():
    world_size = 2
    print(f"Testing vllm communicator with infiniccl, {world_size} GPUs", flush=True)

    manager = mp.Manager()
    result_dict = manager.dict()

    procs = []
    for rank in range(world_size):
        p = mp.Process(target=worker, args=(rank, world_size, result_dict))
        p.start()
        procs.append(p)

    for p in procs:
        p.join(timeout=TIMEOUT)
        if p.is_alive():
            print(f"HUNG pid={p.pid}", file=sys.stderr, flush=True)
            p.kill()
            p.join(timeout=5)

    all_pass = True
    for rank in range(world_size):
        if rank in result_dict:
            val, expected, status = result_dict[rank]
            print(f"GPU {rank}: got {val}, expected {expected}, {status}")
            if status != 'PASS':
                all_pass = False
        else:
            print(f"GPU {rank}: NO RESULT (hung or crashed)")
            all_pass = False

    print(f"\n{'ALL PASSED' if all_pass else 'FAILED'}")
    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()
