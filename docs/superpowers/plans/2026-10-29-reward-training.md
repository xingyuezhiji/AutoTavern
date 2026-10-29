# Tavern reward training implementation plan

Goal: Train a compatible local policy using 100000 bot decisions, 16 BC epochs,
then 40 PPO rounds of 32 complete games against seven scripted bots.

Architecture: Keep the existing actor/checkpoint interface and GPU transfer
workaround. A separate value network supplies a baseline. Rollouts use a CPU
replica, legal masks and stochastic actions; PPO updates run on the selected
training device with dropout disabled. Terminal placement rewards are propagated
without discount so that long decision sequences do not erase the outcome.

- [x] Extend bot collection with game IDs and split validation by whole games.
- [x] Add independent PPO module with batched rollouts, clipped ratios, value
  regression, entropy exploration, finite loss checks and gradient clipping.
- [x] Retain best held-out deterministic policy, export final PPO policy too,
  and save JSON metrics, rewards, action counts and checkpoint selection.
- [x] Run a small CPU BC/PPO smoke and reload the checkpoint for complete games.
- [x] Push only source/docs to openi-train, create the authorized GPU task with
  `--samples 100000 --epochs 16 --rl-rounds 40 --rl-games 32 --ppo-epochs 4`.
- [ ] Verify OpenI device/training/remote output and compare downloaded weights
  using the existing ten fixed seeds and rotated seats.

Reward: `(4.5-placement)/3.5 + 0.5*(placement==1)`. Seat rotates, training,
validation and final fixed test seeds are disjoint. Report real improvement or
regression; more training does not imply winning.

Submitted task: https://openi.pcl.ac.cn/cloudbrains/detail/917175,
code cac056fe3a, created 15:31:44 and started 15:32:06 on 2026-10-29.
CPU smoke: 3000 decisions, 2 BC epochs, 2 PPO rounds of 4 games; finite losses,
checkpoint runtime reload and ten complete independent games verified.

User correction: no CPU training or automatic fallback. Task 917175 stalled
at the first CUDA allocation and was stopped (confirmed STOPPED).
Commit 12961eb makes a failed GPU probe raise before any dataset/training work;
the failure branch was verified by mocking the probe (no training executed).
New GPU-only task: https://openi.pcl.ac.cn/cloudbrains/detail/917183,
created 15:45:05, RUNNING at 15:46 on 2026-10-29. Configuration remains
100000/16/512/32 eval/40 RL rounds/32 RL games/4 PPO epochs.

Latest verification at 15:51: task 917183 is STOPPED, ended 15:50:17
after 4m51s. Simulator compilation succeeded; repeated child stack dumps
show the first `torch.empty(1, device="cuda")` allocation never completed.
No dataset or BC/PPO update ran, and no new policy is available.
The platform stop reason/actor is not established. GPU-only constraint remains.
Browser interactions with the alternative-image form failed to change state;
no replacement job was submitted. Next step is restore reliable UI interaction
and select/diagnose a working GPU resource before resubmitting full training.

917202 reused the training template and also stalled at its first CUDA
allocation before any dataset/training work. Selecting a recommendation image
in the template form did not produce a different image in the task details.
Do not treat it as a controlled alternate-image experiment.

Next controlled experiment retains the user's training task template, BI-V100
and all hyperparameters. Startup file `train_gpu_fresh.py` installs dependencies
in a separate process, then execs a fresh `train.py --train` interpreter. This
tests whether the install-stage Torch import/context affects the later probe.
It is a training task, not an interactive/debug task, and rejects `--cpu`.
