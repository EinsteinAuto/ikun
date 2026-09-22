from patch_utils import package_root, replace_once

VLLM_ROOT = package_root("vllm")

MULTIPROC_GPU_EXECUTOR = VLLM_ROOT / "executor" / "multiproc_gpu_executor.py"
MULTIPROC_WORKER_UTILS = VLLM_ROOT / "executor" / "multiproc_worker_utils.py"


def ensure_import_os(path):
    text = path.read_text()
    if "import os\n" in text:
        print(f"[skip] import os already present: {path}")
        return
    for anchor in ("import time\n", "import signal\n", "import sys\n"):
        if anchor in text:
            replace_once(
                path,
                anchor,
                anchor + "import os\n",
                required=True,
                already_contains="import os\n",
            )
            return
    raise RuntimeError(f"no import anchor found for os in {path}")


ensure_import_os(MULTIPROC_GPU_EXECUTOR)
ensure_import_os(MULTIPROC_WORKER_UTILS)


replace_once(
    MULTIPROC_GPU_EXECUTOR,
    """logger = init_logger(__name__)\n""",
    """logger = init_logger(__name__)\n\n\ndef _bi100_startup_debug(message: str, *args) -> None:\n    if os.getenv(\"BI100_EXECUTOR_STARTUP_DEBUG\") == \"1\":\n        logger.info(\"[BI100 startup] \" + message, *args)\n""",
    required=True,
    already_contains="def _bi100_startup_debug(",
)

replace_once(
    MULTIPROC_GPU_EXECUTOR,
    """        self.driver_worker = self._create_worker(\n            distributed_init_method=distributed_init_method)\n        self._run_workers(\"init_device\")\n        self._run_workers(\"load_model\",\n                          max_concurrent_workers=self.parallel_config.\n                          max_parallel_loading_workers)\n""",
    """        _bi100_startup_debug(\"creating driver worker\")\n        self.driver_worker = self._create_worker(\n            distributed_init_method=distributed_init_method)\n        _bi100_startup_debug(\"created driver worker\")\n        _bi100_startup_debug(\"starting init_device\")\n        self._run_workers(\"init_device\")\n        _bi100_startup_debug(\"finished init_device\")\n        _bi100_startup_debug(\"starting load_model\")\n        self._run_workers(\"load_model\",\n                          max_concurrent_workers=self.parallel_config.\n                          max_parallel_loading_workers)\n        _bi100_startup_debug(\"finished load_model\")\n""",
    required=True,
    already_contains='_bi100_startup_debug("starting init_device")',
)

replace_once(
    MULTIPROC_GPU_EXECUTOR,
    """        # Start all remote workers first.\n        worker_outputs = [\n            worker.execute_method(method, *args, **kwargs)\n            for worker in self.workers\n        ]\n\n        driver_worker_method = getattr(self.driver_worker, method)\n        driver_worker_output = driver_worker_method(*args, **kwargs)\n\n        # Get the results of the workers.\n        return [driver_worker_output\n                ] + [output.get() for output in worker_outputs]\n""",
    """        _bi100_startup_debug(\"enqueue remote method=%s workers=%d\", method,\n                              len(self.workers))\n        # Start all remote workers first.\n        worker_outputs = [\n            worker.execute_method(method, *args, **kwargs)\n            for worker in self.workers\n        ]\n        _bi100_startup_debug(\"remote enqueued method=%s\", method)\n\n        driver_worker_method = getattr(self.driver_worker, method)\n        _bi100_startup_debug(\"driver start method=%s\", method)\n        driver_worker_output = driver_worker_method(*args, **kwargs)\n        _bi100_startup_debug(\"driver done method=%s\", method)\n\n        # Get the results of the workers.\n        _bi100_startup_debug(\"waiting remote results method=%s\", method)\n        remote_outputs = [output.get() for output in worker_outputs]\n        _bi100_startup_debug(\"remote done method=%s\", method)\n        return [driver_worker_output] + remote_outputs\n""",
    required=True,
    already_contains='_bi100_startup_debug("enqueue remote method=%s workers=%d"',
)

replace_once(
    MULTIPROC_WORKER_UTILS,
    """            task_id, method, args, kwargs = items\n            try:\n                executor = getattr(worker, method)\n                output = executor(*args, **kwargs)\n            except SystemExit:\n""",
    """            task_id, method, args, kwargs = items\n            if os.getenv(\"BI100_EXECUTOR_STARTUP_DEBUG\") == \"1\":\n                logger.info(\"[BI100 worker] start method=%s\", method)\n            try:\n                executor = getattr(worker, method)\n                output = executor(*args, **kwargs)\n                if os.getenv(\"BI100_EXECUTOR_STARTUP_DEBUG\") == \"1\":\n                    logger.info(\"[BI100 worker] done method=%s\", method)\n            except SystemExit:\n""",
    required=True,
    already_contains='logger.info("[BI100 worker] start method=%s", method)',
)
