# Action-aligned identity repair

User authorized autonomous guide-informed debugging and retraining through failures.
The guide refers to economy-producing trinkets. Targets remain top4 >= 60–70%
and mean placement <= 3.5; parameter updates must use GPU.

Confirmed defect: permuting card identities among candidate slots while retaining
float features changes legacy logits by exactly zero. Mean pooling loses the
binding between card identity and the indexed action.

Options: flatten all card embeddings (large dense input); attention over all cards
(higher inference and rollout cost); contextual per-action candidate scoring
(chosen, small additive head retaining the existing model).

Seven zero-initialized queries score shop, board, choice and hand identities.
Queries use existing feature hidden state, pooled cards and owned-board/hand
context. Buy, Sell, Activate, Choose, Play and Magnetize scores receive additive
identity scores. Other actions retain the original head. Observation/action ABI
is unchanged. No hidden opponent information or arena credentials are used.

Legacy checkpoint migration may add only the two zero query parameters. Strict
state loading, simulator source, catalog, ruleset and SHA validation remain.
Zero initialization must exactly preserve initial logits and runtime parity.
Old config defaults remain valid. The initial checkpoint saved during migration
uses the new config so PPO development selection can load it strictly.

Verify inference invariance before migration, exact warm-start parity, candidate
permutation sensitivity after nonzero query assignment, action mapping, strict
rejection of unexpected keys and matching training/runtime outputs. These are
CPU inference checks only. Train the repaired model on GPU with fresh guide data,
select on development games and evaluate once on fresh independent games.
Current task 917362 must reach terminal state before submitting the next task.
