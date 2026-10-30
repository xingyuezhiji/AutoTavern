"""Complete-game historical-policy opponents using only each seat's observe2."""
import hashlib
import json
from pathlib import Path

from train import ARENA_CATALOG_SHA256, ARENA_RULESET_ID, ModelConfig, build_model_class, decode_obs, move_tensor


def load_pool(np, torch, nn, tg, manifest_path):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    entries = manifest["policies"]
    if not entries:
        raise ValueError("historical policy pool is empty")
    policies, provenance = [], []
    for entry in entries:
        path = (manifest_path.parent / entry["file"]).resolve()
        if not path.is_file():
            raise ValueError("historical checkpoint is missing")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError("historical checkpoint SHA256 mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["ruleset_id"] != ARENA_RULESET_ID or payload["catalog_sha256"] != ARENA_CATALOG_SHA256:
            raise ValueError("historical policy ruleset/catalog mismatch")
        if payload["taverngym_build_info"]["source_hash"] != tg.build_info()["source_hash"]:
            raise ValueError("historical policy simulator mismatch")
        cfg = ModelConfig(**payload["model_config"])
        model = build_model_class(nn, torch)(cfg)
        model.load_state_dict(payload["state_dict"], strict=True)
        if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
            raise ValueError("non-finite historical policy")
        model.eval()
        policies.append(model)
        provenance.append({"name": entry["name"], "sha256": digest,
                           "model_config": payload["model_config"]})
    return policies, provenance


def assignment(mode, game_index, learner, pool_size):
    """Stable assignment independent of RNG draws and learner decision count."""
    if mode not in {"bot", "mixed", "hard", "curriculum"}:
        raise ValueError("unknown league mode")
    if mode == "curriculum":
        mode = ("bot", "mixed", "mixed", "hard")[game_index % 4]
    count = {"bot": 0, "mixed": 3, "hard": 7}[mode]
    if count and pool_size < 1:
        raise ValueError("model opponents require a historical pool")
    seats = {}
    for offset in range(1, 8):
        seat = (learner + offset) % 8
        seats[seat] = (game_index + offset - 1) % pool_size if offset <= count else None
    return seats


def collect(np, torch, tg, actor, critic, games, seed, *, stochastic,
            opponents=(), mode="bot", record=True, recommendation=None):
    """All updates happen elsewhere; returns learner-only complete episodes."""
    active = [tg.Game(seed + i) for i in range(games)]
    seats = [i % 8 for i in range(games)]
    assigned = [assignment(mode, i, seats[i], len(opponents)) for i in range(games)]
    rows = [[] for _ in active]
    guards = [0] * games
    counts = [0] * len(opponents)
    refreshed_opponent_decisions = 0
    rewards = [recommendation.episode() for _ in active] if recommendation is not None else None
    actor_device = next(actor.parameters()).device
    critic_device = next(critic.parameters()).device if critic is not None else actor_device
    actor.eval()
    if critic is not None:
        critic.eval()
    for policy in opponents:
        policy.eval()
    while True:
        pending, observations = [], []
        batches = [[] for _ in opponents]
        for i, game in enumerate(active):
            if game.is_over():
                continue
            guards[i] += 1
            if guards[i] > 50000:
                raise RuntimeError("league exceeded complete-game guard")
            for seat in range(game.num_players):
                if seat == seats[i] or not game.needs_action(seat):
                    continue
                policy_index = assigned[i][seat]
                if policy_index is None:
                    game.step(seat, int(game.bot_action(seat)))
                else:
                    raw_obs, raw_mask = game.observe2(seat), game.legal_mask(seat)
                    obs = decode_obs(np, raw_obs, raw_mask)
                    batches[policy_index].append((i, seat, obs, raw_obs, raw_mask))
        for policy_index, batch in enumerate(batches):
            if not batch:
                continue
            features = torch.from_numpy(np.stack([x[2][0] for x in batch]))
            cards = torch.from_numpy(np.stack([x[2][1] for x in batch]).astype("int64"))
            mask = torch.from_numpy(np.stack([x[2][2] for x in batch]))
            if not mask.any(dim=1).all():
                raise RuntimeError("historical policy empty legal mask")
            with torch.no_grad():
                actions = opponents[policy_index](features, cards).masked_fill(~mask, -1e9).argmax(-1)
            for row_index, (i, seat, obs, raw_obs, raw_mask) in enumerate(batch):
                # Another seat can end the final combat while this batch waits.
                if active[i].is_over() or not active[i].needs_action(seat):
                    continue
                action = int(actions[row_index])
                # Other seats can give this seat a card/triple/choice while its
                # batch waits. Recompute from its current view before applying.
                current_obs = active[i].observe2(seat)
                current_mask = active[i].legal_mask(seat)
                if current_obs != raw_obs or current_mask != raw_mask:
                    obs = decode_obs(np, current_obs, current_mask)
                    if not obs[2].any():
                        raise RuntimeError("historical policy empty refreshed legal mask")
                    with torch.no_grad():
                        logits = opponents[policy_index](
                            torch.from_numpy(obs[0][None]),
                            torch.from_numpy(obs[1][None].astype("int64")))
                        action = int(logits.masked_fill(
                            ~torch.from_numpy(obs[2][None]), -1e9).argmax(-1)[0])
                    refreshed_opponent_decisions += 1
                if not obs[2][action]:
                    raise RuntimeError("historical policy illegal action")
                active[i].step(seat, action)
                counts[policy_index] += 1
        for i, game in enumerate(active):
            if not game.is_over() and game.needs_action(seats[i]):
                observations.append(decode_obs(np, game.observe2(seats[i]), game.legal_mask(seats[i])))
                pending.append(i)
        if not pending:
            if all(g.is_over() for g in active):
                break
            continue
        feat = torch.from_numpy(np.stack([o[0] for o in observations]))
        card = torch.from_numpy(np.stack([o[1] for o in observations]).astype("int64"))
        mask = torch.from_numpy(np.stack([o[2] for o in observations]))
        if not mask.any(dim=1).all():
            raise RuntimeError("learner empty legal mask")
        with torch.no_grad():
            logits = actor(move_tensor(torch, feat, actor_device),
                           move_tensor(torch, card, actor_device)).masked_fill(
                ~move_tensor(torch, mask, actor_device), -1e9)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample() if stochastic else logits.argmax(-1)
            logp = dist.log_prob(action)
            values = critic(move_tensor(torch, feat, critic_device)).squeeze(-1) if critic is not None else torch.zeros(len(pending))
        for j, i in enumerate(pending):
            a = int(action[j])
            if not observations[j][2][a]:
                raise RuntimeError("learner illegal action")
            bonus = rewards[i].step(json.loads(active[i].player_view(seats[i])),
                                    a, observations[j][2]) if rewards is not None else 0.0
            if record:
                row = (*observations[j], a, float(logp[j]), float(values[j]))
                rows[i].append((*row, bonus) if rewards is not None else row)
            active[i].step(seats[i], a)
    placements = [int(g.placements()[seats[i]]) for i, g in enumerate(active)]
    if any(not 1 <= place <= 8 for place in placements):
        raise RuntimeError("invalid completed placement")
    audit = {"mode": mode, "pool_decision_counts": counts,
             "refreshed_opponent_decisions": refreshed_opponent_decisions,
             "opponent_assignments": assigned, "learner_seats": seats,
             "own_seat_observations_only": True, "complete_games": len(placements)}
    if rewards is not None:
        audit["recommendation"] = {"provenance": recommendation.metadata,
                                   "episodes": [reward.audit() for reward in rewards]}
    return rows, placements, audit


def evaluate(np, torch, tg, actor, games, seed, *, opponents=(), mode="bot"):
    _, places, audit = collect(np, torch, tg, actor, None, games, seed,
                              stochastic=False, opponents=opponents, mode=mode, record=False)
    arr = np.asarray(places)
    return {"games": games, "seed": seed, "mode": mode, "rotated_seats": True,
            "avg_place": float(arr.mean()), "top4": float((arr <= 4).mean()),
            "first": float((arr == 1).mean()), "placements": places, "audit": audit}
