#!/usr/bin/env python3
"""Frozen-opponent continuation and matched entity capacity experiment."""
from pathlib import Path
import sys
from train_gpu_fresh import main

if __name__ == "__main__":
    defaults = {
        "teacher": "guide", "resume": "auto", "league-pool": "auto",
        "samples": "16000", "epochs": "0", "batch-size": "512",
        "rl-rounds": "120", "rl-games": "64", "ppo-epochs": "2",
        "rl-lr": "1e-5", "entropy-coef": "0.002", "bc-anchor": "0.02",
        "top4-bonus": "0.5", "selection-interval": "10", "eval-games": "64",
        "final-eval-games": "256", "league-final-games": "128",
        "final-eval-seed": "20901030", "seed": "20261102",
        "capacity-epochs": "8", "capacity-eval-games": "64",
        "recommendation-snapshot": str(Path(__file__).parent / "data/recommendation-20261030.json"),
    }
    supplied = {arg.split("=", 1)[0] for arg in sys.argv[1:] if arg.startswith("--")}
    sys.argv.append("--capacity-ablation")
    for key, value in defaults.items():
        if "--" + key not in supplied:
            sys.argv.extend(["--" + key, value])
    main()
