#!/usr/bin/env python3
"""GPU-only training entry with dependency setup in a separate interpreter.

Use this as the startup file in the existing OpenI training task template.
All normal train.py hyperparameters and C2Net output handling are preserved.
"""
from pathlib import Path
import os
import subprocess
import sys


def main():
    arguments = sys.argv[1:]
    if any(arg == "--cpu" or arg.startswith("--cpu=") for arg in arguments):
        raise ValueError("This OpenI entry is GPU-only; --cpu is prohibited")
    # The template passes hyperparameters; this entry owns the two phases.
    arguments = [arg for arg in arguments if arg not in {"--install", "--train"}]
    trainer = str(Path(__file__).resolve().with_name("train.py"))
    subprocess.run([sys.executable, "-u", trainer, "--install", *arguments],
                   check=True)
    # No Torch import/context survives setup. A fresh train.py parent creates
    # the existing CUDA child probe; failed probes still raise before training.
    os.execv(sys.executable,
             [sys.executable, "-u", trainer, "--train", *arguments])


if __name__ == "__main__":
    main()
