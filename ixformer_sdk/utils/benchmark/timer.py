import time
from collections import OrderedDict
from typing import Callable

from tabulate import tabulate
from tqdm import tqdm
import torch


class BenchmarkTimer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = None
        self.end_time = None
        self.running_times = []

    def __enter__(self):
        self.start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.running_times.append(self.end_time - self.start_time)


class Benchmark:
    def __init__(
        self,
        warmup: int = None,
        number: int = 100,
        timer=None,
        description: str = None,
        show_progress: bool = False,
        fn_desc_key: str = "fn_desc",
        sync: bool = True,
    ):
        if warmup is None:
            warmup = int(number // 100) + 10
        self.warmup = warmup
        self.number = number
        self.description = description
        self.show_progress = show_progress
        self.fn_desc_key = fn_desc_key
        self.sync = sync

        if timer is None:
            timer = BenchmarkTimer()
        self.timer = timer

        self.reset()

    def reset(self):
        self.results = OrderedDict()
        self._run_index = 0
        self._fn_name = None

    def run(self, fn, *args, **kwargs):
        self._run_index += 1

        if self.fn_desc_key in kwargs:
            self.set_fn_name(kwargs[self.fn_desc_key])
            kwargs.pop(self.fn_desc_key)
        key = self._get_fn_key(fn)

        # warmup
        self._run_fn(False, fn, *args, **kwargs)

        # get running times
        results = self._run_fn(True, fn, *args, **kwargs)
        self.results[key] = results

        return results

    def set_fn_name(self, name):
        self._fn_name = name

    def _run_fn(self, benchmark: bool, fn: Callable, *args, **kwargs):
        self.timer.reset()
        n = self.number if benchmark else self.warmup
        if self.show_progress and benchmark:
            progress = tqdm(range(n), desc=self._get_fn_key(fn))
        else:
            progress = range(n)

        torch.cuda.synchronize()

        for _ in progress:
            with self.timer:
                fn(*args, **kwargs)

                if self.sync:
                    torch.cuda.synchronize()

        return self.timer.running_times

    def _get_fn_key(self, fn: Callable):
        if self._fn_name is not None:
            return self._fn_name

        if hasattr(fn, "__name__"):
            fn_name = fn.__name__
        else:
            fn_name = str(fn)

        return f"{fn_name}_{self._run_index}"

    def render(self) -> str:
        head = [""] + list(self.results.keys())
        total = ["Total (s)"] + [sum(times) for times in self.results.values()]
        mean = ["Mean  (s)"] + [_t / self.number for _t in total[1:]]
        min_ = ["Min   (s)"] + [min(times) for times in self.results.values()]
        max_ = ["Max   (s)"] + [max(times) for times in self.results.values()]
        count = ["Count"] + [len(list(times)) for times in self.results.values()]

        return tabulate(
            headers=head,
            tabular_data=[total, mean, min_, max_, count],
            numalign="right",
        )

    def print_caption(self):
        if self.description is not None:
            caption = "\n" + "=" * 60 + "\n"
            caption += f"= {self.description}" + "\n"
            caption += "=" * 60 + "\n"
            print(caption)

    def print(self):
        print(self.render())
