#!/usr/bin/env python3
"""Guide-informed GPU warm-start entry for the existing OpenI A100 template."""
import sys

from train_gpu_fresh import main


if __name__ == "__main__":
    defaults = {
        "teacher": "guide",
        "resume": "auto",
        "resume-sha256": "5e152a82b861b99c3c7c5a83ed34188ffe6cf30eb3b671d591ae60939df49c53",
        "samples": "120000",
        "epochs": "4",
        "lr": "1e-5",
        "rl-rounds": "160",
        "rl-games": "128",
        "ppo-epochs": "2",
        "rl-lr": "1e-5",
        "bc-anchor": "0.03",
        "top4-bonus": "0.5",
        "selection-interval": "10",
        "eval-games": "128",
        "final-eval-games": "256",
        "final-eval-seed": "20401029",
        "seed": "20261030",
    }
    supplied = {arg.split("=", 1)[0] for arg in sys.argv[1:] if arg.startswith("--")}
    for key, value in defaults.items():
        if "--" + key not in supplied:
            sys.argv.extend(["--" + key, value])
    main()
