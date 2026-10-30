#!/usr/bin/env python3
"""MetaX C500 MACA-only entry for the authorized league/capacity run."""
import os
from pathlib import Path
import runpy
import shutil

if __name__ == "__main__":
    maca = Path(os.environ.get("MACA_PATH", "/opt/maca"))
    if not maca.is_dir():
        raise RuntimeError("MACA runtime directory unavailable")
    os.environ.setdefault("MACA_PATH", str(maca))
    os.environ["PATH"] = str(maca / "bin") + os.pathsep + os.environ["PATH"]
    if not shutil.which("mx-smi"):
        raise RuntimeError("MetaX device utility unavailable; refuse a different accelerator")
    os.environ["TAVERN_ACCELERATOR"] = "metax"
    runpy.run_path(str(Path(__file__).with_name("train_league_gpu.py")),
                   run_name="__main__")
