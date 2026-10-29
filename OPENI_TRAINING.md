# OpenI CloudBrains Training

This repository is pinned to AutoTavern commit `fb61201a557ddd36a411cf44aac51ce4d9b606ce`
with the TavernArena public observe2 patch applied or applied by `train.py`.

## CloudBrains Command

Use the Python training entry:

```bash
cd AutoTavern
python train.py --install --train --samples 50000 --epochs 8 --batch-size 512 --eval-games 16
```

For a quick smoke run:

```bash
cd AutoTavern
python train.py --install --train --samples 2000 --epochs 1 --batch-size 256 --eval-games 2
```

The script uses CUDA automatically when PyTorch sees the BI-V100 / ILUVATAR GPGPU as a CUDA
device. Many ILUVATAR images ship a platform-adapted PyTorch build, so `train.py --install`
keeps an existing `torch` package and only installs `torch` from pip when it is missing. If the
image's PyTorch build cannot see the card, the script falls back to CPU unless `--cpu` is passed
explicitly.

Outputs:

```text
outputs/tavern_bc_policy.pt
```

On OpenI, `train.py` calls `c2net.context.prepare()` and writes the checkpoint to
`c2net_context.output_path` (typically `/tmp/output`). After training and evaluation
finish, it calls `upload_output()` to return the result to the task's **结果下载**
tab. A custom `--output-dir` must stay inside that output directory.

Outside OpenI, the default is `OUTPUT_PATH` when set, otherwise `outputs/`.

The installer downloads Rust with Python (the BI-V100 image has no `curl`),
uses the rsproxy mirror and Rust 1.98.1, then builds TavernGym with
`--no-build-isolation` so maturin does not bootstrap another toolchain.

## Local Use After Training

Download `tavern_bc_policy.pt` from the CloudBrains output.

From the local arena connector directory:

```bash
cd /Users/bytedance/project/codeai/tavernarena-openi
cp /Users/bytedance/project/codeai/AutoTavern/tavern_bc_runtime.py .
export TAVERN_BC_CHECKPOINT="/absolute/path/to/tavern_bc_policy.pt"
export TAVERN_POLICY_RUNTIME="tavern_bc_runtime:predict"
./run_local.sh practice-policy
```

Only after practice succeeds should we consider arena matching.
