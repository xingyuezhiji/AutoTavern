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

For expanded BC followed by outcome-reward PPO:

```bash
python train.py --install --train --samples 100000 --epochs 16 --batch-size 512 --eval-games 32 --rl-rounds 40 --rl-games 32 --ppo-epochs 4
```

BC validation uses separate complete game IDs. PPO plays 1280 fresh full games
against seven scripted opponents with rotating seats, samples only legal actions,
and trains with terminal reward `(4.5-placement)/3.5 + 0.5*(placement==1)`.
There is no per-decision discount. PPO uses clipped likelihood ratios, an
independent value baseline, entropy exploration, a small BC anchor and KL early
stopping. CPU replicas batch rollout inference; actor/value updates use the
verified GPU path. Dropout is disabled during PPO for consistent action ratios.

`tavern_bc_only.pt` preserves the best BC checkpoint, `tavern_ppo_final.pt` saves
the last PPO actor, and `tavern_bc_policy.pt` contains the best policy selected on
fixed development games (which can remain BC if PPO does not improve it).
`ppo_metrics.json` records rewards, placements, action counts and selection.
These development games are separate from final local tests. Opponent strength
and the small validation sample limit how much a better rank demonstrates.

The script uses CUDA after an isolated policy-transfer / forward / backward / optimizer
check passes. The check has a 180-second timeout: GPU detection alone is insufficient on
CoreX, where a card can be visible while `model.to("cuda")` hangs. If the check fails
or times out, the script raises an error and exits. Automatic CPU fallback is
disabled. `--cpu` is reserved for an explicitly requested local smoke run.

For CoreX CPU-to-CUDA transfers, the script follows the workaround from successful
[BenchCAD task 898380](https://openi.pcl.ac.cn/cloudbrains/detail/898380)
(commit `78ccc8f61427b578223458975986cd5923f25290`): allocate a flat GPU tensor,
clone CPU chunks of at most 8 MiB, copy each chunk synchronously, and restore
the tensor shape. This applies to policy parameters, buffers, training batches
and evaluation inputs. Other devices retain the native transfer path.
The allocator uses `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` unless the
environment already specifies it. The probe logs its first allocation separately.
CUDA detection and explicit device selection run inside the probe before its
first allocation. The parent performs no CUDA detection or seeding until the
probe exits. Staged transfer alone still hung in the earlier probe sequence.

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

## Verified OpenI GPU Smoke Run

[Task 917143](https://openi.pcl.ac.cn/cloudbrains/detail/917143) succeeded on
2026-10-29 using code commit `70b0736`, 2000 samples, 1 epoch, batch size 256
and 2 evaluation games. Total task duration was 3 minutes 10 seconds.

The CoreX 3.2 / Torch 2.1 image retained Torch 2.1.0 and NumPy 1.23.5.
Logs confirm `CUDA probe: allocator ready`, `CUDA probe passed`,
`device=cuda`, `gpu=Iluvatar BI-V100` and `model transfer=staged_copy device=cuda`.
Actual training, validation, checkpoint saving and evaluation all completed
on the CUDA path without CPU fallback.

Training loss: 2.8386; validation loss: 2.8500; validation accuracy: 0.1450.
Evaluation placements: 7 and 8. This is a pipeline check, not a competitive model.
The remote **结果下载** tab independently lists `tavern_bc_policy.pt` (10.93 MiB),
updated at 15:06:52, confirming persistence beyond the upload function returning.

Earlier task 917110 (`0c45b03`) verified CPU fallback only.
Task 917130 (`5a4bf5a`) applied staged transfer but still blocked at the first
GPU allocation. Task 917134 (`b7671bb`) matched the allocator configuration and
waited 180 seconds, but still fell back to CPU. The successful `70b0736` change
then moved device detection/selection into the probe and kept parent CUDA
untouched until it exited. These runs verify the final combination; they do
not independently isolate parent context interference from child initialization.

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
