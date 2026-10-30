#!/usr/bin/env python3
"""BI-V100 CoreX-only entry for the authorized league/capacity run."""
import os
from pathlib import Path
import runpy
import shutil

if __name__ == "__main__":
    for root in Path("/usr/local").glob("corex*"):
        corex_bin = root / "bin"
        if corex_bin.is_dir():
            os.environ["PATH"] = str(corex_bin) + os.pathsep + os.environ["PATH"]
    if not any(shutil.which(name) for name in ("ix-smi", "ixsmi")):
        raise RuntimeError("CoreX device utility unavailable; refuse a different accelerator")
    os.environ["TAVERN_ACCELERATOR"] = "iluvatar"
    runpy.run_path(str(Path(__file__).with_name("train_league_gpu.py")),
                   run_name="__main__")
