# [FIX VERIFIED IN TRAINING] GPU allocation hang

User requires GPU-only BC plus outcome PPO; continue autonomous diagnosis and
resubmission without CPU fallback.

Hypotheses:
- A: node/driver/device state hangs initial allocation independently of policy.
- B: parent Torch import before a fresh child affects CoreX initialization.
- C: explicit device selection / CoreX allocator path hangs.

Evidence: 917143 passed allocation, full GPU probe and actual GPU BC on 70b0736.
917175 and 917183 hang at first one-element GPU allocation using the same probe.
The diff adds dataset game IDs, BC/PPO and removes fallback; no probe sequencing
change. 917183 stopped before dataset generation or parameter updates.

Next: inspect available GPU resources and image choices; prefer a working GPU
runtime. If same hardware is required, collect node/driver information and test
fresh-process allocation without prior Torch import. Preserve debugging artifacts.

917209 on commit 4e132a1 used the existing training template and startup
train_gpu_fresh.py. Setup finished, the new train interpreter detected C2Net's
prepared caches and started CUDA probe, but torch.empty still hung. Installation
Torch context surviving in the training parent is not supported as the cause.
Next experiment records ix-smi and hostname/version/visible-device fields and
sets PYTORCH_NO_CUDA_MEMORY_CACHING=1 before all Torch imports. A 60s minimal
allocation+fill+synchronize preflight precedes any simulator compilation.
All training remains CUDA-only; a failed preflight exits before training.

917216, commit 5af1eba, submitted 16:17:32 and started 16:17:46, verifies
GPU uncached allocation passed: torch.empty + fill_ + synchronize + item all
completed on Iluvatar BI-V100 with PYTORCH_NO_CUDA_MEMORY_CACHING=1.
Torch remains vendor 2.1.0 under /usr/local/corex-3.2.0; CUDA_VISIBLE_DEVICES
is absent. No ix-smi/nvidia-smi executable was discovered in PATH.
At 16:19 the simulator is compiling. Full policy backward/Adam and actual
BC/PPO are still pending; do not claim the full training fixed yet.

At 16:21, refreshing task 917216 logs confirms the full GPU probe passed,
100000 samples generated from 94 games (90212 train / 9788 validation),
policy moved by staged_copy to CUDA, and epoch 1 completed with
train_loss=1.3241, train_acc=0.5988, val_loss=1.0126, val_acc=0.6774.
A checkpoint was written to /tmp/output/tavern_bc_policy.pt and epoch 2
started. This verifies actual backward/Adam training on the GPU after the
uncached allocator change. BC completion, PPO and remote download remain
pending. This run supports the allocator workaround; it does not establish
the vendor driver's underlying defect.

Final task 917216 state: FAILED, ended 16:24:46 after 7 minutes.
All 16 CUDA BC epochs completed (best val_loss=0.5864, val_acc=0.8207).
24 PPO rounds completed on GPU. Round 25 development assessment raised
"PPO game exceeded action guard". This is a deterministic-policy action
loop, distinct from the repaired GPU allocation issue. The global 50000
guard correctly exposed it; increasing that limit does not solve it.
Apply the existing bot.rs per-turn policy (>60 actions => EndTurn) using
observe2 feature 3227, only when EndTurn is legal. Share this mask across
PPO collection, checkpoint evaluation and local runtime. Pending choices
retain their original legal mask.
