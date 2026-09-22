import os
from typing import Callable, Optional

# =========================================================
# Utils
# =========================================================


def number_type(scalar_type):
    def wrap(val: Optional[str]):
        if val is None:
            return None

        return scalar_type(val)

    return wrap


def bool_type(val: Optional[str]):
    if val is None:
        return False

    if isinstance(val, str):
        return val.lower() in ["1", "t", "true"]

    if isinstance(val, int):
        return val != 0

    raise RuntimeError(f"Invalid bool type, got {type(val), val}")


def list_type(scalar_type=str):
    def wrap(val: Optional[str]):
        if val is None:
            return []

        if not isinstance(val, str):
            raise RuntimeError(
                f"list_type: Got invalid type, expect str, but got {val}."
            )

        return [scalar_type(v) for v in val.split(",")]

    return wrap


def Field(
    name: str,
    static: bool = True,
    type: Callable = str,
    choices: Optional[list] = None,
    help: Optional[str] = None,
    **kwargs,
):
    """
    Define environment variable field

    Example:
    Static mode:
        # define
        ENABLE_XX = Field("ENABLE_XX", type=bool, help="ENABLE_XX")

        # use
        config.ENABLE_XX

    Dynamic mode:
        # Please use lowercase naming to differentiate it with static mode.

        # define
        enable_cc = Field("ENABLE_CC", type=bool, static=False, help="enable_cc")

        # use
        config.enable_cc()

    Set default value:
        # define
        ENABLE_TT = Field("ENABLE_TT", type=bool, default=False, help="ENABLE_TT")

        # use
        config.ENABLE_TT

    Use list:
        # define
        CUDA_VISIBLE_DEVICES = Field("CUDA_VISIBLE_DEVICES", type=list_type(int), help="CUDA_VISIBLE_DEVICES")

        # use
        # the CUDA_VISIBLE_DEVICES is parsed to list, and it's value is int type.
        for device_id in CUDA_VISIBLE_DEVICES:
            ...

    """

    if type == bool:
        type = bool_type

    elif type in [list, tuple]:
        type = list_type(scalar_type=str)

    elif type in [int, float]:
        type = number_type(type)

    if static:
        env_val = type(os.environ.get(name, **kwargs))
        if choices is not None and env_val is not None and env_val not in choices:
            raise RuntimeError(
                f"Got invalid value, expect {choices}, but got {env_val}."
            )
        return env_val

    def _get():
        env_val = type(os.environ.get(name, **kwargs))
        if choices is not None and env_val is not None and env_val not in choices:
            raise RuntimeError(
                f"Got invalid value, expect {choices}, but got {env_val}."
            )
        return env_val

    return _get


# =========================================================
# Functions Config
# =========================================================

IXFORMER_GEMV_THRESHOLD = Field(
    "IXFORMER_GEMV_THRESHOLD",
    type=int,
    default=1,
    help="Set the threshold for using gemv.",
)


# =========================================================
# Distributed Config
# =========================================================

IXFORMER_COMM_SHM_SIZE = Field(
    "IXFORMER_COMM_SHM_SIZE",
    type=int,
    default=None,
    help="set shared memory size of ipc comm.",
)

IXFORMER_ENABLE_OVERLAP_COMM = Field(
    "IXFORMER_ENABLE_OVERLAP_COMM",
    type=bool,
    default=False,
    help="enable overlap communcation and compute.",
)

IXFORMER_OVERLAP_GEMM_METHOD = Field(
    "IXFORMER_OVERLAP_GEMM_METHOD",
    type=int,
    default=None,
    choices=[0, 1],
    help="set gemm backend, 0: ixinfer, 1: cublas.",
)

IXFORMER_OVERLAP_CHUNKS = Field(
    "IXFORMER_OVERLAP_CHUNKS", type=int, default=2, help="set split chunks."
)

IXFORMER_OVERLAP_SPLIT_RATIO = Field(
    "IXFORMER_OVERLAP_SPLIT_RATIO",
    type=float,
    default=None,
    help="set split chunks ratio.",
)

IXFORMER_PAGED_ATTENTION_ALGO = Field(
    "IXFORMER_PAGED_ATTENTION_ALGO",
    type=str,
    default="ixinfer",
    choices=["ixinfer", "ixformer"],
    help="set paged attention algo.",
)

IXFORMER_UNPAD_ATTENTION_ALGO = Field(
    "IXFORMER_UNPAD_ATTENTION_ALGO",
    type=str,
    default="ixinfer",
    choices=["ixinfer", "ixinfer-ex"],
    help="set enpad attention algo.",
)
