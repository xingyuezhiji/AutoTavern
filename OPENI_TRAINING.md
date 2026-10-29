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

The script uses CUDA after an isolated policy-transfer / forward / backward / optimizer
check passes. The check has a 45-second timeout: GPU detection alone is insufficient on
CoreX, where a card can be visible while `model.to("cuda")` hangs. If the check fails
or times out, the script reports the reason and trains on CPU. `--cpu` skips the GPU check.

Many ILUVATAR images ship a platform-adapted PyTorch build, so `train.py --install`
keeps the existing `torch` and NumPy 1.x packages. It only installs torch if missing.

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

## Verified OpenI Smoke Run

[Task 917110](https://openi.pcl.ac.cn/cloudbrains/detail/917110) succeeded on
2026-10-29 using code commit `0c45b03`, 2000 samples, 1 epoch, batch size 256
and 2 evaluation games. Total task duration was 3 minutes 50 seconds.

The CoreX 3.2 / Torch 2.1 image detected the BI-V100 but hung while moving the
policy to CUDA. The isolated GPU check timed out after 45 seconds, so this
verified run used **CPU**. GPU training is not yet verified on this image.

Validation loss: 2.8379; validation accuracy: 0.1450.
Evaluation placements: 8 and 7. This is a pipeline check, not a competitive model.
The remote **结果下载** tab contains `tavern_bc_policy.pt` (10.93 MiB),
confirming output persistence beyond the upload function returning.

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
