# Debug Session: corex-allocation-hang

- Status: [OPEN]
- Issue: BI-V100 is detected, but the first CUDA tensor allocation does not return.
- Debug Server: local collector; endpoint stored in .dbg/corex-allocation-hang.env

## Hypotheses & Verification

| ID | Hypothesis | Evidence |
|---|---|---|
| A | CoreX needs more than 60 seconds for cold initialization | Task 918048 also timed out after 180 seconds |
| B | PyTorch caching configuration causes the hang | Both cached and uncached allocation timed out in 918045 and 918043 |
| C | CoreX PyTorch runtime and the installed driver are incompatible | Both report 3.2.0; further library/driver evidence needed |
| D | The issue originates below PyTorch in device allocation | A minimal driver allocation probe can distinguish this |

## Log Evidence

918043 and 918045 detected Iluvatar BI-V100, then stopped at `torch.empty(1, device="cuda")`.
No PPO or capacity optimizer updates occurred.
918047 could not start because the listed CoreX 3.1.1 image was unavailable.
918048 timed out at the same allocation after 180 seconds; no GPU updates occurred.
User authorized an additional MetaX C500 trial, submitted as 918049.
918049 passed allocation but stalled during Rust download; stopped before training.
918059 passed allocation but the mounted pretrained model did not provide a verified wheel.
918065 bundles the verified Linux wheel. MLP GPU backward/optimizer passed;
both entity architectures verified actual GPU parameter updates on MetaX C500.
918065 completed three PPO rounds before a stale historical-opponent batch action
failed with "a choice is pending". Execution now refreshes changed observations
and masks before deciding. Three inference-only seeds completed 192 games.

## Verification Conclusion

CoreX root cause remains unresolved. MetaX GPU compatibility is verified;
the complete training and independent evaluation remain pending.
Training remains GPU-only and existing checkpoints are preserved.
