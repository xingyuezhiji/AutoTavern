#!/usr/bin/env python3
"""Guide warm-start with action-aligned identities; GPU updates only."""
import sys

from train_gpu_fresh import main


if __name__ == "__main__":
    defaults = {
        "teacher": "guide",
        "resume": "auto",
        "resume-sha256": "5e152a82b861b99c3c7c5a83ed34188ffe6cf30eb3b671d591ae60939df49c53",
        "samples": "180000",
        "epochs": "12",
        "lr": "1e-4",
        "rl-rounds": "200",
        "rl-games": "128",
        "ppo-epochs": "2",
        "rl-lr": "2e-5",
        "bc-anchor": "0.1",
        "top4-bonus": "0.5",
        "selection-interval": "10",
        "eval-games": "128",
        "final-eval-games": "256",
        "final-eval-seed": "20501029",
        "seed": "20261031",
    }
    supplied = {arg.split("=", 1)[0] for arg in sys.argv[1:] if arg.startswith("--")}
    sys.argv.append("--action-identity")
    for key, value in defaults.items():
        if "--" + key not in supplied:
            sys.argv.extend(["--" + key, value])
    main()
