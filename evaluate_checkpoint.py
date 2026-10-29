"""Independent offline games with the same greedy legal action selection as local runtime."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from train import ModelConfig, build_model_class, choose_model_action, import_runtime


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20262029)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    np, torch, nn, _, taverngym = import_runtime()
    torch.set_num_threads(4)
    payload = torch.load(args.checkpoint, map_location="cpu")
    model = build_model_class(nn, torch)(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    rows = []
    for i in range(args.games):
        game = taverngym.Game(args.seed + i)
        seat = i % 8
        counts = Counter()
        guard = 0
        while not game.is_over():
            progressed = False
            for j in range(game.num_players):
                if not game.needs_action(j):
                    continue
                a = (choose_model_action(np, torch, game, j, model, "cpu")
                     if j == seat else int(game.bot_action(j)))
                if j == seat:
                    counts[a] += 1
                game.step(j, a)
                progressed = True
            guard += 1
            if not progressed or guard > 50000:
                raise RuntimeError("evaluation did not progress")
        state = json.loads(game.state_json())["players"][seat]
        rows.append({"seed": args.seed + i, "seat": seat,
                     "placement": int(game.placements()[seat]),
                     "tier": state["tier"], "board_count": len(state["board"]),
                     "board": state["board"], "upgrades": counts[3],
                     "actions": dict(counts)})
    places = np.asarray([r["placement"] for r in rows])
    report = {"checkpoint": str(Path(args.checkpoint).resolve()),
              "sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
              "games": args.games, "avg_place": float(places.mean()),
              "top4": float((places <= 4).mean()), "first": float((places == 1).mean()),
              "placements": places.tolist(), "rows": rows}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}), flush=True)
    print("tiers", [r["tier"] for r in rows], "boards", [r["board_count"] for r in rows],
          "upgrades", [r["upgrades"] for r in rows], flush=True)


if __name__ == "__main__":
    main()
