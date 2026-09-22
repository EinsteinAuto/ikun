from patch_utils import package_root, replace_one_of


WORKER = package_root("vllm") / "worker" / "worker.py"

CLEAN_BLOCK = """\
        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        self.model_runner.profile_run()
"""

GUARDED_BLOCK = """\
        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model. Mark this synthetic pass so BI100_PROFILE can exclude
        # it without changing vLLM's normal capacity calculation.
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
    [(CLEAN_BLOCK, GUARDED_BLOCK)],
    required=True,
    already_contains=(
        "Mark this synthetic pass so BI100_PROFILE can exclude"),
)
