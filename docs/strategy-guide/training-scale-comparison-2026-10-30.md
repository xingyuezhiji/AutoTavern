# Model and training scale comparison

Sources: original paper https://arxiv.org/html/2610.33845v1 sections 6, 8,
Appendix C tables 7/8/9 and Appendix D; the user-uploaded
`AutoTavern/docs/superpowers/plans/分析论文模型训练细节.md`; downloaded tasks
917395 and 917730, counted model Parameters and PPO decision metrics.

Task 917730: 2,992,951 policy parameters, FP32 artifact 11,978,376 bytes.
Components: card embedding 131,072; feature MLP 2,327,906; policy head
404,725; contextual action identity 129,248. This is not an entity Transformer.
Its 40 PPO rounds collected 1,280 learner episodes and 226,309 decisions,
mean 5,657.725 decisions/round. Its supervised continuation used an attached
self-play dataset; 16 epochs configured, early stopping permitted.

Task 917395: same model size, 200 PPO rounds, 25,600 learner episodes and
4,297,342 decisions, mean 21,486.71 decisions/round; 180,082 guide BC examples
and 12 supervised epochs.

The paper's final architecture: about 18M parameters, entity tokens with card
ID/static features/rule-text embeddings, 8 layers, 8 heads, d=384; masked pointer
policy, shared value head and auxiliary placement head. Symmetric-log inputs.
Final stage: 300 iterations, about 1.05M decisions each, about 310M decisions
and 570k games/stage; minibatch 8192, 1 epoch, LR 6e-6 to 5e-6, clip .2,
target KL .05, gamma 1, GAE lambda .995, entropy .002, value .5,
auxiliary placement .1. Frozen stage-start KL anchor .1 for first 100 iterations.
Opponents are reality-calibrated libraries; exam libraries are held out.

Ratios: paper parameter count approximately six times ours. One paper stage
uses about 1,370 times 917730's PPO decisions, or 72 times 917395's.
These are counts, not wall-clock or efficiency estimates.

At matched training budget the paper's early model-size experiment measured
main-exam mean place 3.84/3.77/3.67 for 3.5M/8M/18M, top4
73.9%/75.6%/78.5%. These results predate the final environment and opponent
libraries and cannot be compared numerically to our bot reports.
The paper's opponent ablation raises first-place rate 20.9% to 27.2%
under matched initialization and budget.

Conclusion: larger model capacity is plausible but not established as the
current primary bottleneck. First measure strong-opponent performance and
encoder parity, then compare an entity Transformer under matched data/budget.
An architectural change cannot directly load the MLP checkpoint; distillation
or fresh training is required, and must preserve the existing baseline.
Do not equate widening the MLP to the paper's entity relational representation.

Runtime correction: Game starts at turn 0 for hero selection and becomes turn 1
after all hero choices. The reference selfplay shaper's turn-zero recruit
assumption is offset. Verified old-versus-league bot rollout placement parity
passed. Corrected complete baseline uses seed 20601030:

| Policy | Opponents | Games | Mean place | Top4 |
|---|---|---:|---:|---:|
| 917395 | 7 bots | 128 | 3.609375 | 67.96875% |
| 917730 | 7 bots | 128 | 3.453125 | 71.875% |
| 917395 | 3 historical + 4 bots | 64 | 3.078125 | 78.125% |
| 917730 | 3 historical + 4 bots | 64 | 3.203125 | 79.6875% |
| 917395 | 7 historical | 64 | 3.875 | 57.8125% |
| 917730 | 7 historical | 64 | 4.0 | 57.8125% |

These small samples do not establish which warm-start is better overall.
Continue the user's latest 917730 with a frozen pool and regression guard.
An asynchronous opponent batch can become obsolete when another seat ends the
game; terminal and needs-action checks now discard those pending actions.
After the fix, hard/mixed games and stored masked log-probability parity pass.

Capacity prototypes: 57 tokens, symmetric-log numeric inputs, contextual card
identity and action pointers. Small d168/4-layer/6-head: 3,035,303 parameters.
Large d384/8-layer/8-head: 18,669,911 parameters. They do not contain pretrained
rule-text embeddings. Both use identical fresh guide samples, game split,
distillation teacher, shuffled batch order, 8 epochs, optimizer and holdouts.
This checks initialization and capacity at a modest sample budget; it does not
establish the optimum parameter count after hundreds of millions of PPO steps.
