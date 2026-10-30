# Live Tavern Script Implementation Plan

> **For agentic workers:** implement the steps in sequence in the isolated live-script-1030 worktree. User authorization covers implementation and offline verification.

**Goal:** Build a runnable Windows client script that selects UI actions using the trained local policy.

**Architecture:** Incremental Power.log parser → visible-state adapter → training-compatible policy → calibrated action plan → guarded Windows executor. Exact simulator replay provides a separate reference path.

**Tech Stack:** Python 3.10+, PyTorch CPU inference, NumPy, Windows user32, PyAutoGUI, Pillow.

---

- [x] Create `live_tavern/power.py`: preserve entity identities, tags, zones, game generation, options and choices. Reject missing ownership and noncontiguous positions.
- [x] Create `live_tavern/observation.py`: read training vocabulary in original order, encode named blocks from visible log state, report unobserved context explicitly; reject unknown cards.
- [x] Create `live_tavern/policy.py`: instantiate `train.ModelConfig` and `train.build_model_class`, validate input and finite legal logits; load on CPU with weights-only deserialization.
- [x] Create `live_tavern/actions.py` and `windows.py`: decode action IDs, bind entity IDs, use calibrated points, verify foreground and unchanged revision, release mouse on stop.
- [x] Create `live_tavern/__main__.py`: support `watch`, `replay`, `sim-replay`, `calibrate`, `doctor`, stop/timeout and structured decision records.
- [x] Add Windows startup commands, example configuration and `docs/LIVE_SCRIPT.md`. Run `python -m live_tavern --help` to check documented entrypoints.
- [x] Add focused offline verification in `tools/verify_live_script.py`; run with the existing 917730 checkpoint and native simulator. Verified model inference, native catalog identity, 24 exact legal simulator steps, entity targets, move insertion and stale revision.
- [x] Prepare implementation for commit and fast-forward integration; the main checkout remains at the recorded base and unrelated user files are untouched.

Portable delivery: `exports/tavern-live-pilot-20261030.zip` outside the repository,
including the 917730 checkpoint. Isolated zip extraction and sample model replay passed.

Live client validation remains a separate Windows session; do not claim it from offline results.
