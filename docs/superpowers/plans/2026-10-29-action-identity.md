# Action Identity Repair Implementation Plan

**Goal:** Recover card-to-action binding and verify guide-informed GPU training.

**Architecture:** Add a zero-initialized contextual identity score to the existing
policy. Preserve legacy logits during strictly bounded migration and reuse the
same head implementation in training and local inference.

**Tech Stack:** PyTorch, TavernGym observe2, OpenI A100, C2Net.

- [x] Inspect observe2 and action ABI; reproduce slot-permutation invariance.
- [x] Design seven contextual query groups using shop, board, hand and choice
  slices; document strict migration and evaluation conditions.
- [ ] Implement `action_identity.py`; update `train.py`,
  `checkpoint_resume.py` and `tavern_bc_runtime.py`.
- [ ] Run CPU inference checks for exact initial parity, indexed permutation
  response, output dimensions and strict migration rejection.
- [ ] Finish task 917362; verify all artifacts and independent evaluation.
- [ ] Add GPU entrypoint with fresh train/development/final seeds, larger guide
  data and stronger supervised adaptation; retain the strongest initial fallback.
- [ ] Commit only task-related files, push `openi-train`, submit using the existing
  user-provided A100 template, monitor through failures and retrieve artifacts.
- [ ] Verify completed GPU epochs/rounds, finite metrics, independent placements
  and economy-trinket/core-lineup behavior. Report measured results honestly.
