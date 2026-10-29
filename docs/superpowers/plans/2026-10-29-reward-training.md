# Tavern reward training implementation plan

Goal: Train a compatible local policy using 100000 bot decisions, 16 BC epochs,
then 40 PPO rounds of 32 complete games against seven scripted bots.

Architecture: Keep the existing actor/checkpoint interface and GPU transfer
workaround. A separate value network supplies a baseline. Rollouts use a CPU
replica, legal masks and stochastic actions; PPO updates run on the selected
training device with dropout disabled. Terminal placement rewards are propagated
without discount so that long decision sequences do not erase the outcome.

- [ ] Extend bot collection with game IDs and split validation by whole games.
- [ ] Add independent PPO module with batched rollouts, clipped ratios, value
  regression, entropy exploration, finite loss checks and gradient clipping.
- [ ] Retain best held-out deterministic policy, export final PPO policy too,
  and save JSON metrics, rewards, action counts and checkpoint selection.
- [ ] Run a small CPU BC/PPO smoke and reload the checkpoint for complete games.
- [ ] Push only source/docs to openi-train, create the authorized GPU task with
  `--samples 100000 --epochs 16 --rl-rounds 40 --rl-games 32 --ppo-epochs 4`.
- [ ] Verify OpenI device/training/remote output and compare downloaded weights
  using the existing ten fixed seeds and rotated seats.

Reward: `(4.5-placement)/3.5 + 0.5*(placement==1)`. Seat rotates, training,
validation and final fixed test seeds are disjoint. Report real improvement or
regression; more training does not imply winning.
