# Debug Session: metax-native-segfault

- Status: [OPEN]
- Baseline: MetaX C500 task 918069, MACA 3.3, Torch 2.8, Python 3.12.
- Symptom: 45 completed PPO rounds followed by SIGSEGV during a repeated
  faulthandler stack dump. No independent holdout or capacity result completed.

## Hypotheses

| ID | Hypothesis | Evidence needed |
|---|---|---|
| A | CPU BLAS/OpenMP multithreading is unstable in this vendor image | Runtime thread/library info; one-thread comparison |
| B | MACA native allocation/update crashes after repeated iterations | GPU allocator statistics; successful sustained updates under mitigation |
| C | Periodic Python traceback dumping interacts with native execution | Compare sustained run without recurring dump; fatal handler remains |

## Existing evidence

The last complete metrics line is round 45 at 490.4 seconds. All logged losses
were finite. The fatal output is `File Segmentation fault (core dumped)` within
a periodic dump whose main-thread stack contains `torch/nn/modules/linear.py`.
The dump location does not establish the original faulting native function.
Task 918069 failed after 10 minutes 20 seconds and returned no downloadable artifacts.

## Next experiment

Record vendor/runtime/thread configuration before model execution.
Use one CPU math thread and retain fatal fault reporting while disabling recurring
stack dumps after startup checks. All optimizer updates remain on C500.
This is an environment mitigation experiment, not a confirmed root-cause fix.

## Implemented experiment

CPU math and inter-op threads are fixed at 1 on the MetaX entry before Torch
imports. Training cancels the periodic traceback timer after startup; fatal
fault reporting remains enabled. Each PPO round records allocated/reserved/
peak GPU memory. The local watcher posts sanitized runtime evidence to the
debug server. Task 918069 was rechecked as FAILED before a new submission.

The stale historical-observation fix remains in place. Recommendation rewards
only read `player_view` for the learner and leave evaluation policy selection
based on unshaped placements.

## Queue restart

Task 918083 stayed WAITING with zero runtime. The user requested a restart.
The stop API accepted the request, and STOPPED was confirmed before creating
task 918095. The replacement uses C500 and commit 01ad9fd with unchanged
training code, initial weights, reward snapshot, and runtime mitigation.
Its first status was WAITING; this supplies no new native-crash evidence.
Detached monitoring and postprocessing were started for task 918095.
