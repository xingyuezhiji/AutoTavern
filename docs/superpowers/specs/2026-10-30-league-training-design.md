# Historical-opponent GPU continuation

Continue the user's authorized training work from task 917730, retaining task
917395 as a benchmark. Use the existing action-identity MLP and ruleset
`tg36-bc45cfc55a0e5cd6c14c`; all parameter updates remain GPU only.

The paper at https://arxiv.org/html/2610.33845v1 uses reality-calibrated board
libraries, not pure self-play. Its Appendix C reports an entity Transformer,
GAE and frozen stage-start KL anchoring. Its live loop uses a human pressing
the keys named by the policy. These are source claims, not measured results
for our model. The present scope implements historical-policy opponents
because no validated human turn-by-turn board library has been provided.

Compare 917730 and 917395 with matched fresh seeds under three opponents:
seven scripted bots; three historical models plus four bots; seven historical
models. Rotate learner seats. Freeze the historical pool throughout a run.
Do not interpret the combined placements of eight identical self-play seats
as performance: their mean is 4.5 and top-four share 50% by construction.

Train one stochastic learner per complete game against a mixture of these
opponents. Record only learner decisions, the actual masked action distribution,
episode placement and opponent identities. Batch inference by policy.
Only observe2/player_view at that policy's own seat may reach its network.
No game copying, future RNG inspection or hidden-opponent state is allowed.

Warm-start the stronger reproducibly evaluated compatible checkpoint.
Preserve both initial and selected checkpoints. Use zero supervised epochs
to avoid overwriting the starting policy, with a small fresh guide dataset
only as a weak imitation anchor. Add frozen stage-start KL regularization.
PPO uses the current stochastic policy's actual log probabilities.

Select on separate fixed development seeds: favor lower mean placement against
mixed historical opponents, with a bot-performance regression guard. Never
select on final holdouts. Fresh final reports compare initial and selected
under all three opponents and report uncertainty, ruleset, hashes, pool,
GPU updates and behavior diagnostics. If no improvement survives holdout,
report that outcome and retain the initial model.

The reference repository https://github.com/xingyuezhiji/tavern-arena was pinned
at its retrieved commit. Its guide shaper assumes turn zero is the first recruit
turn. Runtime evidence shows turn zero is hero selection and turn one is the
first recruit turn; it also caches economy/board state by turn,
while purchases, gifts and upgrades change it inside that turn. Its all-seat
self-play summary is not a skill metric. Reuse only verified concepts.

The ten uploaded images were inspected directly. They support: same-build
comparisons, unseen seeds, stronger opponents, action/encoding parity,
economy/trinket and choice importance, and real-client error accounting.
They do not supply expert intermediate action labels.

Official-client autonomous play is outside this iteration. TavernArena
practice/arena can verify our connector. An official-game feedback loop
requires a compatible state reader and user-operated client, and must
separately validate patch, input encoding, legal actions and client failures.
No arena credential may enter OpenI artifacts or source.

Capacity extension requested during execution: implement 57-token entity
Transformers at 3,035,303 and 18,669,911 parameters. Preserve the public
observe2 ABI and all numeric extensions in a global token. Use identical GPU
distillation data and game-level validation, 8 epochs, teacher-KL plus guide
labels, and three matched final opponent holdouts. Do not claim paper parity:
pretrained rule-text embeddings and reality-calibrated boards are absent.
Save students separately, without promotion over the mature MLP.
