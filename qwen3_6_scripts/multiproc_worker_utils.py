# SPDX-License-Identifier: Apache-2.0
# BI100 adapted multiproc_worker_utils.py
# Based on new vllm API + BI100 startup debug

import asyncio
import os
import sys
import threading
import uuid
from dataclasses import dataclass
from multiprocessing import Queue
from multiprocessing.process import BaseProcess
from typing import (Any, Callable, Dict, Generic, List, Optional, TextIO,
                    TypeVar, Union)

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils import _maybe_force_spawn, get_mp_context, run_method

logger = init_logger(__name__)

print("=================patch_ok: multiproc_worker_utils.py loaded===========")

T = TypeVar("T")

_TERMINATION_SIGNAL = "TERMINATION_SIGNAL"

# ANSI color codes for worker log prefixes
CYAN = "\033[1;36m"
RESET = "\033[0;0m"


@dataclass
class Result(Generic[T]):
    """Result of a task execution."""
    task_id: uuid.UUID
    value: Optional[T] = None
    exception: Optional[BaseException] = None


class ResultFuture(threading.Event, Generic[T]):
    """Synchronous future for worker results."""

    def __init__(self):
        super().__init__()
        self.result: Optional[Result[T]] = None

    def set_result(self, result: Result[T]):
        self.result = result
        self.set()

    def get(self) -> T:
        self.wait()
        assert self.result is not None
        if self.result.exception is not None:
            raise self.result.exception
        return self.result.value  # type: ignore[return-value]


class ResultHandler(threading.Thread):
    """Handle results from all workers (in background thread)."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.result_queue = get_mp_context().Queue()
        self.tasks: Dict[uuid.UUID, Union[ResultFuture,
                                          asyncio.Future]] = {}

    def run(self):
        for result in iter(self.result_queue.get, _TERMINATION_SIGNAL):
            future = self.tasks.pop(result.task_id)
            if isinstance(future, ResultFuture):
                future.set_result(result)
            else:
                loop = future.get_loop()
                if result.exception is not None:
                    loop.call_soon_threadsafe(future.set_exception,
                                             result.exception)
                else:
                    loop.call_soon_threadsafe(future.set_result,
                                             result.value)

    def close(self):
        self.result_queue.put(_TERMINATION_SIGNAL)


class WorkerMonitor(threading.Thread):
    """Monitor worker processes and detect failures."""

    def __init__(self, workers: List["ProcessWorkerWrapper"],
                 result_handler: ResultHandler):
        super().__init__(daemon=True)
        self.workers = workers
        self.result_handler = result_handler
        self._close = False

    def run(self) -> None:
        # Blocks until any worker exits
        dead_sentinels = self._wait_any_worker()
        if not self._close:
            self._on_worker_exit(dead_sentinels)

    def _wait_any_worker(self):
        from multiprocessing.connection import wait
        sentinels = {w.process.sentinel: w for w in self.workers}
        alive = set(sentinels.keys())
        while alive and not self._close:
            ready = wait(alive, timeout=5.0)
            if ready:
                return ready
        return []

    def _on_worker_exit(self, dead_sentinels):
        logger.error("Worker process died unexpectedly")
        # Kill remaining workers
        for worker in self.workers:
            if worker.process.is_alive():
                worker.process.kill()

    def close(self):
        self._close = True
        # Terminate all workers
        for worker in self.workers:
            if worker.process.is_alive():
                worker.process.terminate()
        for worker in self.workers:
            worker.process.join(timeout=5)
        self.result_handler.close()

    def is_alive(self) -> bool:
        return all(w.process.is_alive() for w in self.workers)


class ProcessWorkerWrapper:
    """Local process wrapper for vllm.worker.worker_base.WorkerBase,
    for handling single-node multi-GPU tensor parallel."""

    def __init__(self, result_handler: ResultHandler,
                 worker_factory: Callable[[VllmConfig, int], Any],
                 vllm_config: VllmConfig, rank: int) -> None:
        self.mp = get_mp_context()
        self._task_queue = self.mp.Queue()
        self.result_queue = result_handler.result_queue
        self.tasks = result_handler.tasks
        self.process: BaseProcess = self.mp.Process(
            target=_run_worker_process,
            kwargs=dict(
                worker_factory=worker_factory,
                task_queue=self._task_queue,
                result_queue=self.result_queue,
                vllm_config=vllm_config,
                rank=rank,
            ),
            daemon=True)

        self.process.start()

    def _enqueue_task(self, future: Union[ResultFuture, asyncio.Future],
                      method: Union[str, bytes], args, kwargs):
        task_id = uuid.uuid4()
        self.tasks[task_id] = future
        try:
            self._task_queue.put((task_id, method, args, kwargs))
        except BaseException as e:
            del self.tasks[task_id]
            raise ChildProcessError("worker died") from e

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        future: ResultFuture = ResultFuture()
        self._enqueue_task(future, method, args, kwargs)
        return future

    async def execute_method_async(self, method: Union[str, bytes], *args,
                                   **kwargs):
        future = asyncio.get_running_loop().create_future()
        self._enqueue_task(future, method, args, kwargs)
        return await future

    def kill(self):
        self.process.kill()

    def terminate(self):
        self.process.terminate()


def _run_worker_process(
    worker_factory: Callable[[VllmConfig, int], Any],
    task_queue: Queue,
    result_queue: Queue,
    vllm_config: VllmConfig,
    rank: int,
) -> None:
    """Worker process event loop"""

    # Add process-specific prefix to stdout and stderr
    process_name = get_mp_context().current_process().name
    pid = os.getpid()
    _add_prefix(sys.stdout, process_name, pid)
    _add_prefix(sys.stderr, process_name, pid)

    # Initialize worker
    worker = worker_factory(vllm_config, rank)
    del worker_factory

    # Accept tasks from the engine in task_queue
    # and return task results in result_queue
    logger.info("Worker ready; entering event loop")
    try:
        for items in iter(task_queue.get, _TERMINATION_SIGNAL):
            output = None
            exception = None
            task_id, method, args, kwargs = items
            try:
                if os.getenv("BI100_EXECUTOR_STARTUP_DEBUG") == "1":
                    logger.info("[BI100 worker] start method=%s", method)
                output = run_method(worker, method, args, kwargs)
                if os.getenv("BI100_EXECUTOR_STARTUP_DEBUG") == "1":
                    logger.info("[BI100 worker] done method=%s", method)
            except SystemExit:
                raise
            except KeyboardInterrupt:
                break
            except BaseException as e:
                logger.exception(
                    "Exception in worker %s while processing method %s.",
                    process_name, method)
                exception = e
            result_queue.put(
                Result(task_id=task_id, value=output, exception=exception))
    except KeyboardInterrupt:
        pass
    except Exception:
        logger.exception("Worker failed")

    # Flush TunableOp results when TunableOp is enabled and
    # online (in situ) tuning is enabled.
    if torch.cuda.is_available():
        import torch.cuda.tunable as tunable
        if (tunable.is_enabled() and tunable.tuning_is_enabled()
                and not tunable.record_untuned_is_enabled()):
            tunable.write_file()

    logger.info("Worker exiting")


def _add_prefix(file: TextIO, worker_name: str, pid: int) -> None:
    prefix = f"{CYAN}({worker_name} pid={pid}){RESET} "
    file_write = file.write

    def write_with_prefix(s):
        if not s:
            return
        if '\n' in s[:-1]:
            s = s.replace('\n', '\n' + prefix)
        file_write(prefix + s)

    file.write = write_with_prefix  # type: ignore[method-assign]


def set_multiprocessing_worker_envs(parallel_config):
    """ Set up environment variables that should be used when there are workers
    in a multiprocessing environment. This should be called by the parent
    process before worker processes are created"""

    _maybe_force_spawn()

    # Configure thread parallelism if OMP_NUM_THREADS isn't set
    #
    # Helps to avoid CPU contention. The default of spawning a thread per
    # core combined with multiprocessing for each GPU can have a negative
    # impact on performance. The contention is amplified when running in a
    # container where CPU limits can cause throttling.
    default_omp_num_threads = 1
    if "OMP_NUM_THREADS" not in os.environ and (
            current_parallelism :=
            torch.get_num_threads()) > default_omp_num_threads:
        logger.warning(
            "Reducing Torch parallelism from %d threads to %d to avoid "
            "unnecessary CPU contention. Set OMP_NUM_THREADS in the "
            "external environment to tune this value as needed.",
            current_parallelism, default_omp_num_threads)
        os.environ["OMP_NUM_THREADS"] = str(default_omp_num_threads)
        torch.set_num_threads(default_omp_num_threads)
