# Guide-informed warm-start implementation plan

**Goal:** Resume 917216 on GPU with demonstrator trajectories following the user's economy/trinket, upgrade and core-lineup instructions, then run more PPO rounds.

**Architecture:** A local observable-state demonstrator selects only masked actions using own-player information. Its trajectories supply an auxiliary supervised objective. PPO continues to use newly sampled on-policy actions and placement rewards; terminal public boards never become fabricated expert action labels.

**Tech Stack:** TavernGym observe2, Python, PyTorch, OpenI A100 template, C2Net.

- [x] Read the NetEase guide and prior public lineup study. User corrected the economic preference to trinkets, not minions.
- [x] Add `guide_teacher.py`: own-player view only; turns 1/2/5/6-7 pacing; Dark Gift 3-5; economic offer preference; spells/scaling cores; safe legal-mask fallback.
- [x] Integrate `--teacher guide` into sampling; keep rulebot mode available. Log teacher placements separately and filter training by complete episodes with useful outcomes.
- [x] Verify legal actions, full-game progress, tier timing and teacher results locally without CPU parameter updates.
- [x] Preserve checkpoint SHA and ruleset validation; reserve fresh rotated-seat final evaluation seeds.
- [ ] Upload strongest checkpoint as a private OpenI model, push source changes only, attach model to A100 template.
- [ ] Run GPU warm-start with 120000 guide decisions, 4 supervised epochs at low LR, 160 PPO rounds of 128 games, 2 PPO epochs, 128 development games and 256 fresh final games.
- [ ] Verify GPU update logs and exported artifacts; compare selected and initial checkpoint independently, continue or repair when failures provide a concrete diagnosis.

Source: https://ds.163.com/article/6ad1d321ef679856d2134001/
The article's rankings are reference material, not measurements under TavernArena rules.
Default tier targets are conditional on legality and survival; tier 5/6 requires adequate health/economy.

Verification: seed 20261031 produced 4 complete teacher games, mean placement
3.25, top-four rate 75%; the 3 successful episodes yielded 631 decisions,
all legal. This small demonstrator check is not trained-model acceptance.
Checkpoint SHA, ruleset, catalog and simulator source compatibility passed.
C2Net 0.2.0 wheel confirms `ctx.pretrain_model_path` is the supported field.

Warm-start preservation now saves `tavern_initial.pt`; the original also
participates in development-set selection. Final evaluation uses 256 new
games at seed 20401029, rotating all eight seats, and writes initial/selected
results to `independent_evaluation.json` without selecting on that holdout.

Online old-policy sampling completed: 517 accepted decisions across 5 complete
games. Local temporary authentication file removed after completion.

OpenI private model `autotavern-guide-warmstart-917216` exists but is empty:
hidden Dropzone input cannot yet be populated using the browser uploader.
The A100 template form is open at
https://openi.pcl.ac.cn/cloudbrains/create?tmpl=33eb24b6-18ca-4619-8fe1-874ebc42f860 .
Native page search successfully brings offscreen controls into view; ordinary
scroll calls did not. No new GPU training task has been submitted yet.
