import time
from collections import namedtuple, OrderedDict
from typing import List, Dict, Any

import tabulate
import torch
import torch.distributed as dist

DeviceTime = namedtuple("DeviceTime", ["cpu", "gpu"])


class Functor:

    def __init__(self, fn, *args, **kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return self.fn(*self.args, **self.kwargs)


def cuda_timeit(fn: Functor, dist_barrier=False) -> DeviceTime:
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record(torch.cuda.current_stream())

    t0 = time.time()
    fn()
    t1 = time.time()

    stop.record(torch.cuda.current_stream())

    torch.cuda.synchronize()
    if dist_barrier:
        dist.barrier()

    gpu_time = start.elapsed_time(stop)
    cpu_time = t1 - t0
    return DeviceTime(cpu_time, gpu_time)


def cuda_benchmark(fn: Functor, num_repeated=10, num_warmup=1, dist_barrier=False) -> DeviceTime:
    [fn() for _ in range(num_warmup)]
    times = [cuda_timeit(fn, dist_barrier=dist_barrier) for _ in range(num_repeated)]
    times.sort(key=lambda t: t.gpu)

    if num_repeated >= 10:
        times = times[3:-3]

    avg_gpu_time = sum([t.gpu for t in times]) / len(times)
    avg_cpu_time = sum([t.cpu for t in times]) / len(times)

    return DeviceTime(avg_cpu_time * 1000, avg_gpu_time)


def show_benchmark_results(times: List[DeviceTime], extra_info: Dict[Any, List]=None):
    data = extra_info or OrderedDict()

    if len(times) != 0:
        cpu_times = [round(t.cpu, 6) for t in times]
        gpu_times = [round(t.gpu, 6) for t in times]

        data["CPU Time(ms)"] = cpu_times
        data["GPU Time(ms)"] = gpu_times

    print(tabulate.tabulate(extra_info, headers=data.keys()))
