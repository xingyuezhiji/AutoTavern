# Historical-opponent continuation implementation plan

**Goal:** Complete a GPU continuation and fresh evaluation against stronger,
documented opponents using the user's new checkpoint and research references.

**Architecture:** Isolated worktree based on user's OpenI commit `e5125e0`.
Keep original training modes intact. Add batched league rollout/assessment and
optional pool-aware PPO selection with an initial-policy KL anchor.

**Tech Stack:** TavernGym observe2, PyTorch, OpenI A100/C2Net, native OpenI API.

- [x] Inspect ten exact uploaded images, paper sections 6/8/C/D, and reference
  repository; download the Bilibili video to Downloads.
- [x] Retrieve task 917730's six artifacts and configuration; preserve user
  upstream changes in isolated `AutoTavern-league-1030`.
- [ ] Implement `league_rollout.py`: validated frozen policy loading, own-seat
  inputs, batched masked inference, complete episodes, rotating seats and
  deterministic opponent assignment for bot/mixed/hard modes.
- [ ] Verify full-game progress, learner legal actions, episode log probabilities,
  matching old bot rollout placements and historical opponent behavior.
- [ ] Run matched fresh baseline comparisons, record hashes and uncertainty.
- [ ] Add optional league flags to `train.py` and extend `tavern_ppo.py` only
  for league mode; select on mixed development mean with bot regression guard.
- [ ] Add `train_league_gpu.py` entrypoint using zero BC epochs, guide anchor
  samples, 120 PPO rounds, 64 games per round, two PPO epochs and low learning
  rate; freeze stage-start model for KL anchoring.
- [ ] Verify initial-only checkpoint preservation, selection guard, finite
  loss paths and inference parity. CPU must never update parameters.
- [ ] Commit source/docs, push the authorized OpenI training branch, upload
  hash-verified private warm-start/opponent bundle and submit one A100 task.
- [ ] Follow GPU training through errors; repair concrete failures and continue.
- [ ] Download/check artifacts, reproducibly compare all final placements and
  report independent results and trinket/core/timing behavior honestly.

The initial discovery commands and source claims are recorded in the design.
The target remains improved actual policy behavior; against equally strong
seven-model opponents, fair placement is 4.5/top4 50%, so the bot target
70%/3.5 is not imposed on every opponent setting.
