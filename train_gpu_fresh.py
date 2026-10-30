#!/usr/bin/env python3
"""GPU-only training entry with dependency setup in a separate interpreter.

Use this as the startup file in the existing OpenI training task template.
All normal train.py hyperparameters and C2Net output handling are preserved.
"""
from pathlib import Path
import os
import shutil
import subprocess
import sys


def gpu_preflight():
    """Read driver state and verify allocation before expensive setup."""
    utilities = (("ix-smi", "ixsmi") if os.getenv("TAVERN_ACCELERATOR") == "iluvatar"
                 else ("ix-smi", "ixsmi", "nvidia-smi"))
    for executable in utilities:
        command = shutil.which(executable)
        if command:
            subprocess.run([command], timeout=15, check=False)
    # CoreX's native allocator was used by the verified prior BI-V100 run.
    # Its uncached allocation timed out in task 918043; keep NVIDIA's existing
    # workaround scoped to that backend instead of imposing it on CoreX.
    if os.getenv("TAVERN_ACCELERATOR") == "iluvatar":
        os.environ.pop("PYTORCH_NO_CUDA_MEMORY_CACHING", None)
    else:
        os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    probe = """
import faulthandler, json, os, socket, torch
faulthandler.enable()
faulthandler.dump_traceback_later(15, repeat=True)
print(json.dumps({"event":"gpu_preflight","host":socket.gethostname(),
 "torch":torch.__version__,"torch_path":torch.__file__,
 "cuda_visible":os.getenv("CUDA_VISIBLE_DEVICES"),
 "no_memory_cache":os.getenv("PYTORCH_NO_CUDA_MEMORY_CACHING")}),flush=True)
if not torch.cuda.is_available():
    raise RuntimeError("GPU unavailable; CPU training prohibited")
torch.cuda.set_device(0)
print("GPU allocation device: "+torch.cuda.get_device_name(0),flush=True)
print("GPU allocation: empty tensor",flush=True)
x=torch.empty(1,device="cuda")
print("GPU allocation: fill tensor",flush=True)
x.fill_(1)
print("GPU allocation: synchronize",flush=True)
torch.cuda.synchronize()
assert x.item()==1
print("GPU allocation passed",flush=True)
"""
    child = subprocess.Popen([sys.executable, "-u", "-c", probe],
                             start_new_session=True)
    try:
        # Allow CoreX cold context initialization to finish before declaring
        # a driver hang. This is bounded and precedes dependency installation.
        timeout = 180 if os.getenv("TAVERN_ACCELERATOR") == "iluvatar" else 60
        result = child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Do not synchronously wait again for an unresponsive driver process.
        child.kill()
        raise RuntimeError("GPU allocation timed out; no training started")
    if result:
        raise RuntimeError(f"GPU preflight failed ({result}); no CPU fallback")


def main():
    arguments = sys.argv[1:]
    if any(arg == "--cpu" or arg.startswith("--cpu=") for arg in arguments):
        raise ValueError("This OpenI entry is GPU-only; --cpu is prohibited")
    # The template passes hyperparameters; this entry owns the two phases.
    arguments = [arg for arg in arguments if arg not in {"--install", "--train"}]
    gpu_preflight()
    trainer = str(Path(__file__).resolve().with_name("train.py"))
    subprocess.run([sys.executable, "-u", trainer, "--install", *arguments],
                   check=True)
    # No Torch import/context survives setup. A fresh train.py parent creates
    # the existing CUDA child probe; failed probes still raise before training.
    os.execv(sys.executable,
             [sys.executable, "-u", trainer, "--train", *arguments])


if __name__ == "__main__":
    main()
