"""Masked PPO with complete-game placement rewards; no live arena access."""
from collections import Counter
from pathlib import Path
import json
import time

from train import build_model_class, decode_obs, move_policy, move_tensor, save_checkpoint


def placement_reward(place, top4_bonus=0.0):
    if not 1 <= place <= 8:
        raise ValueError(f"invalid placement {place}")
    return (4.5 - place) / 3.5 + 0.5 * (place == 1) + top4_bonus * (place <= 4)


def selection_stats(np, places, args):
    mean = float(np.mean(places))
    top4 = float(np.mean(np.asarray(places) <= 4))
    # Both requested targets must hold; otherwise favor progress in top4.
    key = (top4 >= args.target_top4 and mean <= args.target_avg_place, top4, -mean)
    return key, mean, top4


def collect(np, torch, taverngym, actor, critic, games, seed, *, stochastic):
    """Batch one pending decision per game, rotating the learner's seat."""
    active = [taverngym.Game(seed + i) for i in range(games)]
    seats = [i % 8 for i in range(games)]
    rows = [[] for _ in active]
    guards = [0] * games
    actor.eval()
    critic.eval()
    while True:
        pending, observations = [], []
        for i, game in enumerate(active):
            if game.is_over():
                continue
            guards[i] += 1
            if guards[i] > 50000:
                raise RuntimeError("PPO game exceeded action guard")
            progressed = False
            for seat in range(game.num_players):
                if seat != seats[i] and game.needs_action(seat):
                    game.step(seat, int(game.bot_action(seat)))
                    progressed = True
            if not game.is_over() and game.needs_action(seats[i]):
                observations.append(decode_obs(np, game.observe2(seats[i]),
                                               game.legal_mask(seats[i])))
                pending.append(i)
            elif not progressed and not game.is_over():
                raise RuntimeError("PPO game stopped progressing")
        if not pending:
            if all(g.is_over() for g in active):
                break
            continue
        feat = torch.from_numpy(np.stack([o[0] for o in observations]))
        card = torch.from_numpy(np.stack([o[1] for o in observations]).astype("int64"))
        mask = torch.from_numpy(np.stack([o[2] for o in observations]))
        if not mask.any(dim=1).all():
            raise RuntimeError("empty legal mask")
        with torch.no_grad():
            logits = actor(feat, card).masked_fill(~mask, -1e9)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample() if stochastic else logits.argmax(-1)
            logp = dist.log_prob(action)
            values = critic(feat).squeeze(-1)
        for j, i in enumerate(pending):
            a = int(action[j])
            if not observations[j][2][a]:
                raise RuntimeError("illegal rollout action")
            rows[i].append((*observations[j], a, float(logp[j]), float(values[j])))
            active[i].step(seats[i], a)
    placements = [int(g.placements()[seats[i]]) for i, g in enumerate(active)]
    return rows, placements


def train_ppo(args, np, torch, nn, F, taverngym, model, cfg, device, bc):
    output = Path(args.output_dir)
    actor = build_model_class(nn, torch)(cfg)
    critic = nn.Sequential(nn.LayerNorm(cfg.feature_dim),
                           nn.Linear(cfg.feature_dim, 256), nn.GELU(), nn.Linear(256, 1))
    value_model = move_policy(torch, critic, device)
    cpu_critic = nn.Sequential(nn.LayerNorm(cfg.feature_dim),
                              nn.Linear(cfg.feature_dim, 256), nn.GELU(), nn.Linear(256, 1))
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(value_model.parameters()),
                                 lr=args.rl_lr)
    history = []
    selection_seed = args.seed + 200000
    league = bool(args.league_pool)
    opponents, provenance = (), []
    if league:
        from league_rollout import load_pool, evaluate as league_eval, collect as league_collect
        opponents, provenance = load_pool(np, torch, nn, taverngym, args.league_pool)
    stage_anchor = None
    if league and args.stage_kl:
        stage_anchor = move_policy(torch, build_model_class(nn, torch)(cfg), device)
        stage_anchor.load_state_dict(model.state_dict(), strict=True)
        stage_anchor.eval()
        for parameter in stage_anchor.parameters():
            parameter.requires_grad_(False)
    assessment = {}
    bot_baseline = None
    def sync():
        actor.load_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items()})
        cpu_critic.load_state_dict({k: v.detach().cpu() for k, v in value_model.state_dict().items()})
    def assess():
        sync()
        if league:
            assessment["mixed"] = league_eval(np, torch, taverngym, actor, args.eval_games,
                                             selection_seed, opponents=opponents, mode="mixed")
            assessment["bot"] = league_eval(np, torch, taverngym, actor, args.eval_games,
                                           selection_seed, mode="bot")
            return assessment["mixed"]["placements"]
        _, places = collect(np, torch, taverngym, actor, cpu_critic,
                            args.eval_games, selection_seed, stochastic=False)
        return places
    baseline = assess()
    if league:
        bot_baseline = dict(assessment["bot"])
    def selection(places):
        key, mean, top4 = selection_stats(np, places, args)
        if league:
            guard = (assessment["bot"]["avg_place"] <= bot_baseline["avg_place"] + 0.4
                     and assessment["bot"]["top4"] >= bot_baseline["top4"] - 0.125)
            key = (guard, -mean, top4)
        return key, mean, top4
    best_key, best_mean, best_top4 = selection(baseline)
    metadata = {"reward": "(4.5-placement)/3.5 + 0.5*(placement==1) + top4_bonus*(placement<=4)",
                "top4_bonus": args.top4_bonus,
                "initialization": "resumed_policy" if args.resume else "behavior_cloning",
                "bc_placements": baseline, "selection_seed": selection_seed,
                "selection_games": args.eval_games,
                "target_top4": args.target_top4, "target_avg_place": args.target_avg_place,
                "best_round": 0, "rounds": history}
    metadata.update({"league": league, "pool": provenance, "update_device": str(device),
                     "baseline_assessment": dict(assessment),
                     "gae_lambda": args.gae_lambda if league else None,
                     "stage_kl": args.stage_kl if league else 0})
    bc_payload = torch.load(output / "tavern_bc_policy.pt", map_location="cpu")
    torch.save(bc_payload, output / "tavern_bc_only.pt")
    # The warm-start policy is also eligible on the development set, so
    # supervised imitation cannot silently replace a stronger original.
    initial_path = output / "tavern_initial.pt"
    if initial_path.exists() and not league:
        initial_payload = torch.load(initial_path, map_location="cpu", weights_only=True)
        actor.load_state_dict(initial_payload["state_dict"], strict=True)
        _, initial_places = collect(np, torch, taverngym, actor, cpu_critic,
                                    args.eval_games, selection_seed, stochastic=False)
        metadata["initial_placements"] = initial_places
        key, mean, top4 = selection_stats(np, initial_places, args)
        if key > best_key:
            best_key, best_mean, best_top4 = key, mean, top4
            metadata["best_round"] = "initial"
            torch.save(initial_payload, output / "tavern_bc_policy.pt")
    print(f"PPO baseline placements={baseline} avg={best_mean:.3f}", flush=True)
    started = time.time()
    xf, xc, xm, y, train_idx = bc
    # npz 里存 float16/int16/uint8，统一转成训练 dtype。
    xf = xf.astype("float32")
    xc = xc.astype("int64")
    xm = xm.astype("bool")
    y = y.astype("int64")
    for iteration in range(1, args.rl_rounds + 1):
        sync()
        rollout_audit = None
        if league:
            rows, places, rollout_audit = league_collect(
                np, torch, taverngym, actor, cpu_critic, args.rl_games,
                args.seed + 1000000 + iteration * args.rl_games, stochastic=True,
                opponents=opponents, mode=args.league_mode)
        else:
            rows, places = collect(np, torch, taverngym, actor, cpu_critic, args.rl_games,
                                  args.seed + 1000000 + iteration * args.rl_games,
                                  stochastic=True)
        flat, returns = [], []
        for episode, place in zip(rows, places):
            flat.extend(episode)
            reward = placement_reward(place, args.top4_bonus)
            if league:
                targets = [0.0] * len(episode)
                gae, next_value = 0.0, 0.0
                for t in range(len(episode) - 1, -1, -1):
                    value = episode[t][5]
                    delta = (reward if t == len(episode) - 1 else 0.0) + next_value - value
                    gae = delta + args.gae_lambda * gae
                    targets[t] = gae + value
                    next_value = value
                returns.extend(targets)
            else:
                returns.extend([reward] * len(episode))
        if not flat:
            raise RuntimeError("no PPO decisions")
        feat = np.stack([r[0] for r in flat])
        cards = np.stack([r[1] for r in flat]).astype("int64")
        masks = np.stack([r[2] for r in flat])
        actions = np.asarray([r[3] for r in flat], dtype="int64")
        old_logp = np.asarray([r[4] for r in flat], dtype="float32")
        returns = np.asarray(returns, dtype="float32")
        advantage = returns - np.asarray([r[5] for r in flat], dtype="float32")
        advantage = (advantage - advantage.mean()) / max(float(advantage.std()), 1e-6)
        # Each game has equal aggregate weight despite variable decision counts.
        weights = np.concatenate([np.full(len(ep), len(flat) / (len(rows) * max(1, len(ep))),
                                          dtype="float32") for ep in rows])
        model.eval()  # Dropout would invalidate the stored likelihood ratios.
        value_model.eval()
        losses, kls = [], []
        optimizer_steps = 0
        stopped = False
        for _ in range(args.ppo_epochs):
            order = np.random.permutation(len(flat))
            for start in range(0, len(flat), args.batch_size):
                idx = order[start:start + args.batch_size]
                def tensor(a):
                    return move_tensor(torch, torch.from_numpy(a), device)
                f, c, m = tensor(feat[idx]), tensor(cards[idx]), tensor(masks[idx])
                dist = torch.distributions.Categorical(logits=model(f, c).masked_fill(~m, -1e9))
                lp = dist.log_prob(tensor(actions[idx]))
                delta = lp - tensor(old_logp[idx])
                ratio = delta.exp()
                adv, w = tensor(advantage[idx]), tensor(weights[idx])
                policy_loss = -(torch.minimum(ratio * adv, ratio.clamp(0.8, 1.2) * adv) * w).mean()
                value_loss = ((value_model(f).squeeze(-1) - tensor(returns[idx])).square() * w).mean()
                anchors = np.random.choice(train_idx, size=len(idx), replace=True)
                anchor_loss = F.cross_entropy(model(tensor(xf[anchors]), tensor(xc[anchors]))
                                              .masked_fill(~tensor(xm[anchors]), -1e9),
                                              tensor(y[anchors]))
                loss = policy_loss + 0.5 * value_loss - args.entropy_coef * (dist.entropy() * w).mean()
                loss = loss + args.bc_anchor * anchor_loss
                if stage_anchor is not None and iteration <= args.stage_kl_rounds:
                    with torch.no_grad():
                        anchor_dist = torch.distributions.Categorical(
                            logits=stage_anchor(f, c).masked_fill(~m, -1e9))
                    loss = loss + args.stage_kl * (
                        torch.distributions.kl_divergence(anchor_dist, dist) * w).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite PPO loss")
                kl = float(((ratio - 1) - delta).mean().detach().cpu())
                if kl > 0.03:
                    stopped = True
                    break
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(list(model.parameters()) + list(value_model.parameters()),
                                         args.grad_clip)
                optimizer.step()
                optimizer_steps += 1
                losses.append(float(loss.detach().cpu()))
                kls.append(kl)
            if stopped:
                break
        record = {"round": iteration, "games": len(places), "decisions": len(flat),
                  "placements": places, "avg_place": float(np.mean(places)),
                  "mean_reward": float(np.mean([placement_reward(p, args.top4_bonus) for p in places])),
                  "top4": float(np.mean(np.asarray(places) <= 4)),
                  "first": float(np.mean(np.asarray(places) == 1)),
                  "loss": float(np.mean(losses)) if losses else None,
                  "kl": float(np.mean(kls)) if kls else None,
                  "actions": dict(Counter(map(int, actions))), "kl_stopped": stopped,
                  "optimizer_steps": optimizer_steps}
        if league:
            record["rollout_audit"] = rollout_audit
        # A fixed development set selects the checkpoint; external fixed games
        # remain untouched for the final independent comparison.
        if iteration % args.selection_interval == 0 or iteration == args.rl_rounds:
            selected = assess()
            record["selection_placements"] = selected
            key, mean, top4 = selection(selected)
            if league:
                record["assessment"] = dict(assessment)
            record["selection_avg_place"] = mean
            record["selection_top4"] = top4
            if key > best_key:
                best_key = key
                best_mean = mean
                best_top4 = top4
                metadata["best_round"] = iteration
                save_checkpoint(torch, model, cfg, output / "tavern_bc_policy.pt", args,
                                bc_payload["metrics"]["val_loss"], bc_payload["metrics"]["val_acc"],
                                taverngym)
        history.append(record)
        metadata["best_avg_place"] = best_mean
        metadata["best_top4"] = best_top4
        (output / "ppo_metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"PPO round={iteration}/{args.rl_rounds} decisions={len(flat)} "
              f"avg_place={record['avg_place']:.3f} reward={record['mean_reward']:.3f} "
              f"top4={record['top4']:.3f} loss={record['loss']} "
              f"elapsed={time.time()-started:.1f}s", flush=True)
    save_checkpoint(torch, model, cfg, output / "tavern_ppo_final.pt", args,
                    bc_payload["metrics"]["val_loss"], bc_payload["metrics"]["val_acc"], taverngym)
    model.load_state_dict(torch.load(output / "tavern_bc_policy.pt", map_location="cpu")["state_dict"])
    print(f"PPO selected round={metadata['best_round']} avg_place={best_mean:.3f}", flush=True)
