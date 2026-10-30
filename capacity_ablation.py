"""Matched GPU-only distillation ablation; does not promote untested models."""
from pathlib import Path
import hashlib
import json
import time

from train import ModelConfig, build_model_class, move_policy, save_checkpoint
from league_rollout import evaluate, load_pool


def run(args, np, torch, nn, F, tg, teacher, data, device):
    if str(device) != "cuda":
        raise ValueError("capacity parameter updates require GPU")
    output = Path(args.output_dir)
    xf, xc, xm, y, train_idx, val_idx = data
    opponents, provenance = load_pool(np, torch, nn, tg, args.league_pool)
    teacher.eval()
    targets = np.empty((len(y), 245), dtype="float32")
    with torch.no_grad():
        for start in range(0, len(y), 256):
            end = min(len(y), start + 256)
            f = torch.from_numpy(xf[start:end].astype("float32")).to(device)
            c = torch.from_numpy(xc[start:end].astype("int64")).to(device)
            m = torch.from_numpy(xm[start:end].astype("bool")).to(device)
            targets[start:end] = teacher(f, c).masked_fill(~m, -1e9).softmax(-1).cpu().numpy()
    report = {"experiment": "matched distillation initialization, not full PPO scaling",
              "teacher_sha256": hashlib.sha256((output / "tavern_bc_policy.pt").read_bytes()).hexdigest(),
              "data_samples": len(y), "train_samples": len(train_idx),
              "validation_samples": len(val_idx), "epochs": args.capacity_epochs,
              "common_seed": args.seed + 700000, "batch_size": 128,
              "loss": "0.8*teacher_KL + 0.2*guide_cross_entropy",
              "rule_text_embeddings": False, "entity_tokens": 57,
              "all_updates_gpu": True, "pool": provenance, "models": {}}
    # Same samples, splits, batch order, epochs, loss and optimizer for both.
    for name, dimension, layers, heads in (("small", 168, 4, 6), ("18m", 384, 8, 8)):
        torch.manual_seed(args.seed + 700000)
        rng = np.random.default_rng(args.seed + 700000)
        cfg = ModelConfig(architecture="entity_transformer", dropout=0.0,
                          transformer_dim=dimension, transformer_layers=layers,
                          transformer_heads=heads)
        model = move_policy(torch, build_model_class(nn, torch)(cfg), device)
        model.eval()
        parameters = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
        path = output / f"tavern_entity_{name}.pt"
        best_loss = float("inf")
        history = []
        started = time.time()
        def loss_for(index):
            f = torch.from_numpy(xf[index].astype("float32")).to(device)
            c = torch.from_numpy(xc[index].astype("int64")).to(device)
            m = torch.from_numpy(xm[index].astype("bool")).to(device)
            label = torch.from_numpy(y[index].astype("int64")).to(device)
            target = torch.from_numpy(targets[index]).to(device)
            logits = model(f, c).masked_fill(~m, -1e9)
            kd = F.kl_div(F.log_softmax(logits, -1), target, reduction="batchmean")
            return 0.8 * kd + 0.2 * F.cross_entropy(logits, label)
        for epoch in range(1, args.capacity_epochs + 1):
            order = rng.permutation(train_idx)
            total, count = 0.0, 0
            for start in range(0, len(order), 128):
                index = order[start:start + 128]
                optimizer.zero_grad(set_to_none=True)
                loss = loss_for(index)
                if not torch.isfinite(loss):
                    raise RuntimeError("capacity non-finite loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += float(loss.detach().cpu()) * len(index)
                count += len(index)
            val_total = 0.0
            with torch.no_grad():
                for start in range(0, len(val_idx), 128):
                    index = val_idx[start:start + 128]
                    val_total += float(loss_for(index).cpu()) * len(index)
            val_loss = val_total / len(val_idx)
            history.append({"epoch": epoch, "train_loss": total / count, "val_loss": val_loss})
            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(torch, model, cfg, path, args, val_loss, 0, tg)
            print(f"CAPACITY model={name} parameters={parameters} epoch={epoch} "
                  f"val_loss={val_loss:.5f} elapsed={time.time()-started:.1f}s", flush=True)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(payload["state_dict"], strict=True)
        scores = {}
        for mode in ("bot", "mixed", "hard"):
            scores[mode] = evaluate(np, torch, tg, model, args.capacity_eval_games,
                                    args.final_eval_seed + 500000,
                                    opponents=opponents, mode=mode)
            print(f"CAPACITY eval model={name} mode={mode} "
                  f"avg_place={scores[mode]['avg_place']:.3f}", flush=True)
        report["models"][name] = {"parameters": parameters, "gpu_updates": args.capacity_epochs *
                                 ((len(train_idx) + 127) // 128), "history": history,
                                 "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                 "evaluation": scores, "elapsed_seconds": time.time() - started}
        (output / "capacity_report.json").write_text(json.dumps(report, indent=2))
        del model, optimizer
        torch.cuda.empty_cache()
    # Compare the mature teacher on identical holdouts; tiny student budgets may
    # not yet preserve its playing ability even if distillation loss falls.
    report["teacher_evaluation"] = {}
    for mode in ("bot", "mixed", "hard"):
        report["teacher_evaluation"][mode] = evaluate(
            np, torch, tg, teacher, args.capacity_eval_games,
            args.final_eval_seed + 500000, opponents=opponents, mode=mode)
    (output / "capacity_report.json").write_text(json.dumps(report, indent=2))
