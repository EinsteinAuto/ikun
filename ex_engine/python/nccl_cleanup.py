"""
nccl_cleanup — register signal handlers that call ncclCommAbort before exit

import this module early in any process that creates nccl communicators.
on SIGINT/SIGTERM it will:
1. call ncclCommAbort on all registered comms
2. call cudaDeviceReset on the current device
3. then let the default handler run (exit/raise)

this prevents GPU driver state corruption on BI-V100 corex 3.2.3
where interrupted nccl ops leave P2P mappings in a broken state.
"""

import ctypes
import signal
import sys
import os

_registered_comms = []
_nccl_lib = None
_cudart = None
_original_sigint = None
_original_sigterm = None


def _load_libs():
    global _nccl_lib, _cudart
    if _nccl_lib is not None:
        return
    torch_lib = "/usr/local/corex/lib64/python3/dist-packages/torch/lib"
    _nccl_lib = ctypes.CDLL(
        os.path.join(torch_lib, "libtorch_cuda.so"), mode=ctypes.RTLD_GLOBAL
    )
    _nccl_lib.ncclCommAbort.argtypes = [ctypes.c_void_p]
    _nccl_lib.ncclCommAbort.restype = ctypes.c_int
    _cudart = ctypes.CDLL("/usr/local/corex/lib64/libcudart.so")


def register_comm(comm_handle):
    """Register an nccl comm handle for cleanup on signal."""
    _registered_comms.append(comm_handle)


def _cleanup_handler(signum, frame):
    """Abort all nccl comms and reset GPU before exiting."""
    _load_libs()
    print(f"\n[nccl_cleanup] signal {signum}, aborting {len(_registered_comms)} comms",
          file=sys.stderr, flush=True)

    for comm in _registered_comms:
        if comm is not None:
            try:
                ret = _nccl_lib.ncclCommAbort(comm)
                print(f"[nccl_cleanup] ncclCommAbort: {ret}", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[nccl_cleanup] ncclCommAbort failed: {e}", file=sys.stderr, flush=True)
    _registered_comms.clear()

    # reset current device
    try:
        _cudart.cudaDeviceReset()
        print("[nccl_cleanup] cudaDeviceReset done", file=sys.stderr, flush=True)
    except Exception:
        pass

    # restore and re-raise
    if signum == signal.SIGINT:
        signal.signal(signal.SIGINT, _original_sigint or signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGINT)
    elif signum == signal.SIGTERM:
        signal.signal(signal.SIGTERM, _original_sigterm or signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGTERM)


def install():
    """Install signal handlers. Call once at process start."""
    global _original_sigint, _original_sigterm
    _load_libs()
    _original_sigint = signal.getsignal(signal.SIGINT)
    _original_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _cleanup_handler)
    signal.signal(signal.SIGTERM, _cleanup_handler)
    print("[nccl_cleanup] signal handlers installed", file=sys.stderr, flush=True)
