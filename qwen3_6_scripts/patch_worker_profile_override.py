from patch_utils import package_root, replace_one_of

WORKER = package_root("vllm") / "worker" / "worker.py"

CLEAN_BLOCK = """\
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        self.model_runner.profile_run()
"""

GUARDED_BLOCK = """\
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model. Mark this synthetic pass so BI100_PROFILE can skip
        # timing it by default; profiling real requests is the useful signal.
        _bi100_prev_startup_profile = os.environ.get("BI100_IN_STARTUP_PROFILE")
        os.environ["BI100_IN_STARTUP_PROFILE"] = "1"
        try:
            self.model_runner.profile_run()
        finally:
            if _bi100_prev_startup_profile is None:
                os.environ.pop("BI100_IN_STARTUP_PROFILE", None)
            else:
                os.environ["BI100_IN_STARTUP_PROFILE"] = _bi100_prev_startup_profile
"""

NEW_BLOCK = """\
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()

        # BI100: Qwen3.6 batched dummy profile_run can trip GDN non-finite
        # checks before the server starts. If the operator explicitly provides
        # --num-gpu-blocks-override, trust that conservative capacity value and
        # skip only the synthetic profile pass. Real inference still uses the
        # normal GDN fail-fast path.
        if self.cache_config.num_gpu_blocks_override is not None:
            cache_block_size = self.get_cache_block_size_bytes()
            if cache_block_size == 0:
                num_cpu_blocks = 0
            else:
                num_cpu_blocks = int(self.cache_config.swap_space_bytes //
                                     cache_block_size)
            logger.warning(
                "[BI100] skipping worker.profile_run because "
                "num_gpu_blocks_override=%d was explicitly set",
                self.cache_config.num_gpu_blocks_override)
            gc.collect()
            torch.cuda.empty_cache()
            return self.cache_config.num_gpu_blocks_override, max(num_cpu_blocks, 0)

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model. Mark this synthetic pass so BI100_PROFILE can skip
        # timing it by default; profiling real requests is the useful signal.
        _bi100_prev_startup_profile = os.environ.get("BI100_IN_STARTUP_PROFILE")
        os.environ["BI100_IN_STARTUP_PROFILE"] = "1"
        try:
            self.model_runner.profile_run()
        finally:
            if _bi100_prev_startup_profile is None:
                os.environ.pop("BI100_IN_STARTUP_PROFILE", None)
            else:
                os.environ["BI100_IN_STARTUP_PROFILE"] = _bi100_prev_startup_profile
"""

replace_one_of(
    WORKER,
    [
        (GUARDED_BLOCK, NEW_BLOCK),
        (CLEAN_BLOCK, NEW_BLOCK),
    ],
    required=True,
    already_contains="[BI100] skipping worker.profile_run",
)
