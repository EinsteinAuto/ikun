"""
gpu_watchdog — OS-level recovery for hung NCCL operations on BI-V100

problem: nccl operations that get interrupted leave GPU in unrecoverable
state. cudaDeviceReset and ncclCommAbort don't help because the driver-
level P2P mapping is corrupted.

solution: run as a wrapper around the target process. if the process hangs
(no output for N seconds) or receives SIGINT, the watchdog:
1. sends SIGTERM to the process
2. waits briefly for graceful shutdown
3. if still alive, sends SIGKILL
4. calls cudaDeviceReset on all GPUs via a FRESH process
   (fresh process = fresh driver context, avoids inheriting corruption)
5. verifies each GPU responds to basic operations

usage:
    python3 tools/gpu_watchdog.py -- python3 -c "your_nccl_test.py"
    python3 tools/gpu_watchdog.py --timeout 30 -- python3 -m vllm.entrypoints...
"""

import subprocess
import signal
import sys
import os
import time


def reset_gpus():
    """Reset GPUs from a fresh process (no inherited CUDA context)."""
    reset_script = '''
import ctypes, sys
cudart = ctypes.CDLL("/usr/local/corex/lib64/libcudart.so")
for i in range(4):
    cudart.cudaSetDevice(i)
    r = cudart.cudaDeviceReset()
    print(f"GPU {i} reset: {r}", file=sys.stderr)
'''
    subprocess.run(
        [sys.executable, "-c", reset_script],
        timeout=10,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1,2,3"},
    )


def verify_gpu(gpu_id):
    """Verify a single GPU works from a fresh process."""
    verify_script = f'''
import torch
a = torch.ones(100, device="cuda:0")
print(a.sum().item())
'''
    try:
        r = subprocess.run(
            [sys.executable, "-c", verify_script],
            timeout=10,
            capture_output=True, text=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)},
        )
        return r.returncode == 0 and "100" in r.stdout
    except subprocess.TimeoutExpired:
        return False


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--timeout", type=int, default=60,
                   help="kill process if no output for this many seconds")
    p.add_argument("--gpus", default="0,1,2,3",
                   help="GPU ids to manage")
    p.add_argument("cmd", nargs="+", help="command to run")
    args = p.parse_args()

    gpu_ids = [int(x) for x in args.gpus.split(",")]

    print(f"[watchdog] starting: {' '.join(args.cmd)}", file=sys.stderr)
    print(f"[watchdog] timeout={args.timeout}s gpus={gpu_ids}", file=sys.stderr)

    proc = subprocess.Popen(
        args.cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=os.environ,
    )

    def handle_signal(sig, frame):
        print(f"\n[watchdog] caught signal {sig}, shutting down", file=sys.stderr)
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("[watchdog] SIGKILL", file=sys.stderr)
            proc.kill()
            proc.wait(timeout=5)

        print("[watchdog] resetting GPUs from fresh process", file=sys.stderr)
        reset_gpus()

        for gid in gpu_ids:
            ok = verify_gpu(gid)
            print(f"[watchdog] GPU {gid}: {'ok' if ok else 'DEAD'}",
                  file=sys.stderr)

        sys.exit(1)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    ret = proc.wait()
    print(f"[watchdog] process exited with code {ret}", file=sys.stderr)
    if ret != 0:
        print("[watchdog] non-zero exit, resetting GPUs", file=sys.stderr)
        reset_gpus()
    sys.exit(ret)


if __name__ == "__main__":
    main()
