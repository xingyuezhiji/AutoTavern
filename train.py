#!/usr/bin/env python3
"""Train a small TavernArena-compatible policy on OpenI CloudBrains.

This script is designed for environments where the job can clone this repository
but cannot clone GitHub directly. It can:

1. verify/apply the public observe2 patch;
2. install TavernGym and PyTorch dependencies;
3. train a behavior-cloning policy from TavernGym's scripted bot;
4. export a checkpoint that can be loaded locally by `tavern_policy.py`.

No arena key, live match state, model registry credential, or game save is read
or written by this script.
"""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent
# Match the allocator configuration used by the successful CoreX reference.
# This must be set before importing Torch in the parent or CUDA probe.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
PATCH_URL = "https://rizling.top/downloads/autotavern-observe2.patch"
PATCH_SHA256 = "1e48a071cad63c284debbf917cb012595a405a4ef5ab66180eb2918e205aaac0"
PINNED_COMMIT = "fb61201a557ddd36a411cf44aac51ce4d9b606ce"
ARENA_RULESET_ID = "tg36-bc45cfc55a0e5cd6c14c"
ARENA_CATALOG_SHA256 = "c4c42e6ce5060e1b79fbcd5f4edbf6d17495f0f52690072481d690610f7ef6d7"
FEATURE_DIM = 4017
CARD_DIM = 447
ACTION_DIM = 245


def run(cmd: list[str], *, cwd: Path = REPO_ROOT, check: bool = True) -> subprocess.CompletedProcess:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(cwd), check=check)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_observe2_patch() -> None:
    """Apply the arena public patch if this checkout is still the base release."""
    patched_marker = REPO_ROOT / "crates" / "taverngym" / "src" / "obs2.rs"
    if patched_marker.exists():
        print("observe2 patch already present", flush=True)
        return

    current = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    ).stdout.strip()
    if current and current != PINNED_COMMIT:
        print(f"warning: current git commit is {current}, expected base {PINNED_COMMIT}", flush=True)

    patch_path = Path("/tmp/autotavern-observe2.patch")
    print(f"downloading observe2 patch: {PATCH_URL}", flush=True)
    urllib.request.urlretrieve(PATCH_URL, patch_path)
    got = sha256_file(patch_path)
    if got != PATCH_SHA256:
        raise RuntimeError(f"patch sha256 mismatch: got {got}, expected {PATCH_SHA256}")

    run(["git", "apply", str(patch_path)])
    print("observe2 patch applied", flush=True)


def install_environment() -> None:
    """Install dependencies inside the OpenI job container."""
    ensure_observe2_patch()
    # Avoid maturin's implicit Rust bootstrap over a slow overseas connection.
    os.environ.setdefault("RUSTUP_DIST_SERVER", "https://rsproxy.cn")
    os.environ.setdefault("RUSTUP_UPDATE_ROOT", "https://rsproxy.cn/rustup")
    os.environ.setdefault("RUSTUP_TOOLCHAIN", "1.98.1")
    cargo_home = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo"))
    os.environ["PATH"] = str(cargo_home / "bin") + os.pathsep + os.environ["PATH"]
    if shutil.which("rustup") is None:
        # Minimal BI-V100 images have neither curl nor wget. Download the
        # standalone installer with Python rather than the shell bootstrap.
        architecture = {"x86_64": "x86_64", "aarch64": "aarch64"}.get(platform.machine())
        if platform.system() != "Linux" or architecture is None:
            raise RuntimeError("automatic Rust installation requires Linux x86_64/aarch64")
        # rustup selects installer/proxy mode from its executable filename.
        installer = Path("/tmp/rustup-init")
        url = f"https://rsproxy.cn/rustup/dist/{architecture}-unknown-linux-gnu/rustup-init"
        for attempt in range(1, 4):
            try:
                print(f"downloading Rust installer (attempt {attempt}): {url}", flush=True)
                with urllib.request.urlopen(url, timeout=120) as response, installer.open("wb") as target:
                    shutil.copyfileobj(response, target)
                break
            except (OSError, urllib.error.URLError):
                installer.unlink(missing_ok=True)
                if attempt == 3:
                    raise
        installer.chmod(0o755)
        run([str(installer), "-y", "--profile", "minimal", "--default-toolchain", "none"])
    run(["rustup", "toolchain", "install", os.environ["RUSTUP_TOOLCHAIN"],
         "--profile", "minimal"])
    cargo_home.mkdir(parents=True, exist_ok=True)
    config = cargo_home / "config.toml"
    if not config.exists():
        config.write_text(
            '[source.crates-io]\nreplace-with = "rsproxy-sparse"\n'
            '[source.rsproxy-sparse]\nregistry = "sparse+https://rsproxy.cn/index/"\n'
            '[net]\nretry = 3\n[http]\ntimeout = 120\n', encoding="utf-8")
    run(["rustc", "--version"])
    run([sys.executable, "-m", "pip", "install", "maturin>=1.15,<2"])
    # Preserve the image's NumPy 1.x: its CoreX TensorFlow requires <1.24
    # and CuPy requires <1.26. Torch 2.1 also needs the NumPy 1.x ABI.
    if find_spec("numpy") is None or int(version("numpy").split(".")[0]) >= 2:
        run([sys.executable, "-m", "pip", "install", "numpy>=1.23,<1.24"])
    else:
        print(f"keeping existing numpy={version('numpy')}", flush=True)
    # ILUVATAR / BI-V100 images often carry a platform-adapted torch build.
    # Do not overwrite it unless torch is genuinely absent.
    if find_spec("torch") is None:
        run([sys.executable, "-m", "pip", "install", "torch"])
    else:
        import torch

        print(f"keeping existing torch={torch.__version__}", flush=True)
    run([sys.executable, "-m", "pip", "install", "--no-build-isolation", "."])


def import_runtime():
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import taverngym

    return np, torch, nn, F, taverngym


@dataclass
class ModelConfig:
    feature_dim: int = FEATURE_DIM
    card_dim: int = CARD_DIM
    action_dim: int = ACTION_DIM
    card_vocab: int = 4096
    card_emb: int = 32
    hidden: int = 512
    dropout: float = 0.05


def build_model_class(nn, torch):
    class TavernBCPolicy(nn.Module):
        def __init__(self, cfg: ModelConfig):
            super().__init__()
            self.cfg = cfg
            self.card_emb = nn.Embedding(cfg.card_vocab, cfg.card_emb, padding_idx=0)
            self.feature_net = nn.Sequential(
                nn.LayerNorm(cfg.feature_dim),
                nn.Linear(cfg.feature_dim, cfg.hidden),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden, cfg.hidden),
                nn.GELU(),
            )
            self.head = nn.Sequential(
                nn.Linear(cfg.hidden + cfg.card_emb, cfg.hidden),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden, cfg.action_dim),
            )

        def forward(self, features, cards):
            cards = cards.clamp(min=0, max=self.cfg.card_vocab - 1)
            emb = self.card_emb(cards)
            present = (cards != 0).float().unsqueeze(-1)
            pooled = (emb * present).sum(dim=1) / present.sum(dim=1).clamp_min(1.0)
            x = self.feature_net(features.float())
            return self.head(torch.cat([x, pooled], dim=-1))

    return TavernBCPolicy


def decode_obs(np, obs_bytes: tuple[bytes, bytes], mask_bytes: bytes):
    features = np.frombuffer(obs_bytes[0], dtype="<f4").copy()
    cards = np.frombuffer(obs_bytes[1], dtype="<i4").copy()
    mask = np.frombuffer(mask_bytes, dtype=np.uint8).astype(bool)
    if features.shape != (FEATURE_DIM,) or cards.shape != (CARD_DIM,) or mask.shape != (ACTION_DIM,):
        raise ValueError(f"bad obs shapes: {features.shape}, {cards.shape}, {mask.shape}")
    return features, cards, mask


def generate_bot_dataset(np, taverngym, *, samples: int, seed: int, max_games: int):
    features, cards, masks, actions = [], [], [], []
    game_count = 0
    started = time.time()
    print(f"generating bot dataset: samples={samples} seed={seed}", flush=True)
    while len(actions) < samples and game_count < max_games:
        game = taverngym.Game(seed + game_count)
        guard = 0
        while not game.is_over() and len(actions) < samples:
            progressed = False
            for seat in range(game.num_players):
                if not game.needs_action(seat):
                    continue
                action = int(game.bot_action(seat))
                obs = game.observe2(seat)
                mask_bytes = game.legal_mask(seat)
                f, c, m = decode_obs(np, obs, mask_bytes)
                if not m[action]:
                    raise RuntimeError(f"bot produced illegal action {action}")
                features.append(f)
                cards.append(c)
                masks.append(m)
                actions.append(action)
                if len(actions) % 500 == 0:
                    print(f"dataset samples={len(actions)}/{samples} game={game_count} "
                          f"turn={game.turn} elapsed={time.time() - started:.1f}s", flush=True)
                game.step(seat, action)
                progressed = True
                if len(actions) >= samples:
                    break
            guard += 1
            if not progressed or guard > 50000:
                raise RuntimeError("game did not progress while generating data")
        game_count += 1
        if game_count % 10 == 0:
            elapsed = time.time() - started
            print(f"generated {len(actions)}/{samples} samples from {game_count} games in {elapsed:.1f}s", flush=True)

    if len(actions) < samples:
        raise RuntimeError(f"only generated {len(actions)} samples; increase --max-games")

    return (
        np.stack(features).astype("float32"),
        np.stack(cards).astype("int64"),
        np.stack(masks).astype(bool),
        np.asarray(actions, dtype="int64"),
    )


def iter_minibatches(np, n: int, batch_size: int, *, shuffle: bool = True) -> Iterable:
    order = np.arange(n)
    if shuffle:
        np.random.shuffle(order)
    for start in range(0, n, batch_size):
        yield order[start : start + batch_size]


def cuda_training_ready(timeout: float = 180) -> bool:
    """Exercise the actual policy in a killable process before touching GPU here."""
    probe = """
import faulthandler
import torch
from train import ModelConfig, build_model_class, move_policy, move_tensor
faulthandler.enable()
faulthandler.dump_traceback_later(20, repeat=True)
torch.set_num_threads(4)
print("CUDA probe: checking device", flush=True)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
torch.cuda.set_device(0)
print(f"CUDA probe: gpu={torch.cuda.get_device_name(0)}", flush=True)
print("CUDA probe: initializing allocator", flush=True)
torch.empty(1, device="cuda")
torch.cuda.synchronize()
print("CUDA probe: allocator ready", flush=True)
print("CUDA probe: copying policy", flush=True)
cfg = ModelConfig()
model = move_policy(torch, build_model_class(torch.nn, torch)(cfg), "cuda")
print("CUDA probe: forward/backward/optimizer", flush=True)
opt = torch.optim.AdamW(model.parameters())
features = move_tensor(torch, torch.zeros(2, cfg.feature_dim), "cuda")
cards = move_tensor(torch, torch.zeros(2, cfg.card_dim, dtype=torch.long), "cuda")
mask = move_tensor(torch, torch.ones(2, cfg.action_dim, dtype=torch.bool), "cuda")
target = move_tensor(torch, torch.zeros(2, dtype=torch.long), "cuda")
loss = torch.nn.functional.cross_entropy(model(features, cards).masked_fill(~mask, -1e9), target)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
opt.step()
torch.cuda.synchronize()
assert torch.isfinite(loss).item()
print("CUDA probe passed", flush=True)
"""
    print(f"checking CUDA training in isolated process (timeout={timeout}s)", flush=True)
    try:
        result = subprocess.run(
            [sys.executable, "-u", "-c", probe], cwd=str(REPO_ROOT), timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print("CUDA training probe timed out; using CPU for this run", flush=True)
        return False
    if result.returncode != 0:
        print(f"CUDA training probe failed (exit={result.returncode}); using CPU for this run", flush=True)
        return False
    return True


def move_tensor(torch, tensor, device, *, chunk_bytes: int = 8 * 1024 * 1024):
    """Use the staged dense transfer verified by OpenI task 898380 on CoreX."""
    target = torch.device(device)
    corex = "corex" in str(Path(torch.__file__).resolve()).lower()
    if not (corex and tensor.device.type == "cpu" and target.type == "cuda"):
        return tensor.to(target)
    if tensor.layout != torch.strided or tensor.is_quantized:
        raise TypeError("CoreX staged transfer requires dense, non-quantized tensors")
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    source = tensor.reshape(-1)
    destination = torch.empty(source.numel(), dtype=tensor.dtype, device=target)
    elements = max(1, chunk_bytes // tensor.element_size())
    for offset in range(0, source.numel(), elements):
        length = min(elements, source.numel() - offset)
        # Clone even contiguous inputs to avoid CPU storage-backed Tensor.to.
        staging = source.narrow(0, offset, length).clone()
        destination.narrow(0, offset, length).copy_(staging, non_blocking=False)
    return destination.view(tensor.shape)


def move_policy(torch, model, device):
    corex = "corex" in str(Path(torch.__file__).resolve()).lower()
    staged = corex and torch.device(device).type == "cuda"
    print(f"model transfer={'staged_copy' if staged else 'native'} device={device}", flush=True)
    # _apply preserves Parameters, gradients, buffers and child modules.
    return model._apply(lambda tensor: move_tensor(torch, tensor, device))


def train_bc(args) -> None:
    # Keep the parent CUDA context untouched until the child has exited.
    # Vendor runtimes may block allocation in another process after detection
    # initializes a context in the parent.
    use_cuda = not args.cpu and cuda_training_ready()
    np, torch, nn, F, taverngym = import_runtime()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"device={device}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    print(f"taverngym build={taverngym.build_info()}", flush=True)
    if taverngym.OBS2_FLOATS != FEATURE_DIM or taverngym.OBS2_CARDS != CARD_DIM:
        raise RuntimeError(f"unexpected observe2 dims: {taverngym.OBS2_FLOATS}, {taverngym.OBS2_CARDS}")

    x_feat, x_card, x_mask, y = generate_bot_dataset(
        np, taverngym, samples=args.samples, seed=args.seed, max_games=args.max_games
    )
    print(f"dataset ready: samples={len(y)}", flush=True)
    split = int(len(y) * (1.0 - args.val_ratio))
    train_idx = np.arange(split)
    val_idx = np.arange(split, len(y))

    cfg = ModelConfig(card_vocab=max(int(x_card.max()) + 1, args.card_vocab))
    TavernBCPolicy = build_model_class(nn, torch)
    print(f"moving policy to {device}", flush=True)
    model = move_policy(torch, TavernBCPolicy(cfg), device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "tavern_bc_policy.pt"

    def batch_loss(indexes, train: bool):
        feat = move_tensor(torch, torch.from_numpy(x_feat[indexes]), device)
        card = move_tensor(torch, torch.from_numpy(x_card[indexes]), device)
        mask = move_tensor(torch, torch.from_numpy(x_mask[indexes]), device)
        target = move_tensor(torch, torch.from_numpy(y[indexes]), device)
        logits = model(feat, card).masked_fill(~mask, -1e9)
        loss = F.cross_entropy(logits, target)
        pred = logits.argmax(dim=-1)
        acc = (pred == target).float().mean()
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
        return float(loss.detach().cpu()), float(acc.detach().cpu())

    for epoch in range(1, args.epochs + 1):
        print(f"starting epoch={epoch}", flush=True)
        model.train()
        train_losses, train_accs = [], []
        for idx in iter_minibatches(np, len(train_idx), args.batch_size, shuffle=True):
            loss, acc = batch_loss(train_idx[idx], True)
            train_losses.append(loss)
            train_accs.append(acc)

        model.eval()
        val_losses, val_accs = [], []
        with torch.no_grad():
            for idx in iter_minibatches(np, len(val_idx), args.batch_size, shuffle=False):
                loss, acc = batch_loss(val_idx[idx], False)
                val_losses.append(loss)
                val_accs.append(acc)

        train_loss, train_acc = float(np.mean(train_losses)), float(np.mean(train_accs))
        val_loss, val_acc = float(np.mean(val_losses)), float(np.mean(val_accs))
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}",
            flush=True,
        )
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(torch, model, cfg, best_path, args, val_loss, val_acc, taverngym)
            print(f"saved {best_path}", flush=True)

    evaluate_policy(np, torch, taverngym, model, device, args.eval_games, args.seed + 100000)
    print(f"final checkpoint: {best_path}", flush=True)


def save_checkpoint(torch, model, cfg: ModelConfig, path: Path, args, val_loss, val_acc, taverngym) -> None:
    payload = {
        "model_type": "TavernBCPolicy",
        "model_config": asdict(cfg),
        "state_dict": model.state_dict(),
        "ruleset_id": ARENA_RULESET_ID,
        "catalog_sha256": ARENA_CATALOG_SHA256,
        "taverngym_build_info": taverngym.build_info(),
        "train_args": vars(args),
        "metrics": {"val_loss": float(val_loss), "val_acc": float(val_acc)},
    }
    torch.save(payload, path)


def choose_model_action(np, torch, game, seat: int, model, device) -> int:
    f, c, m = decode_obs(np, game.observe2(seat), game.legal_mask(seat))
    with torch.no_grad():
        logits = model(
            move_tensor(torch, torch.from_numpy(f).unsqueeze(0), device),
            move_tensor(torch, torch.from_numpy(c.astype("int64")).unsqueeze(0), device),
        )[0].detach().cpu().numpy()
    logits = np.where(m, logits, -np.inf)
    action = int(np.argmax(logits))
    if not m[action]:
        raise RuntimeError("model selected illegal action after masking")
    return action


def evaluate_policy(np, torch, taverngym, model, device, games: int, seed: int) -> None:
    placements = []
    model.eval()
    for i in range(games):
        game = taverngym.Game(seed + i)
        guard = 0
        while not game.is_over():
            progressed = False
            for seat in range(game.num_players):
                if not game.needs_action(seat):
                    continue
                action = choose_model_action(np, torch, game, seat, model, device) if seat == 0 else int(game.bot_action(seat))
                game.step(seat, action)
                progressed = True
            guard += 1
            if not progressed or guard > 50000:
                raise RuntimeError("evaluation game did not progress")
        placements.append(int(game.placements()[0]))
    arr = np.asarray(placements)
    print(
        "eval "
        f"games={games} avg_place={arr.mean():.3f} top4={(arr <= 4).mean():.3f} "
        f"first={(arr == 1).mean():.3f} placements={placements}",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true", help="apply observe2 patch if needed and pip install dependencies")
    parser.add_argument("--train", action="store_true", help="run behavior-cloning training")
    parser.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--max-games", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--eval-games", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--card-vocab", type=int, default=4096)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    # #region debug-point H1:H2-stack
    faulthandler.enable()
    faulthandler.dump_traceback_later(60, repeat=True)
    # #endregion
    args = parse_args()
    upload = None
    if find_spec("c2net") is not None:
        from c2net.context import prepare, upload_output

        ctx = prepare()
        output_root = Path(ctx.output_path).resolve()
        destination = Path(args.output_dir).resolve() if args.output_dir else output_root
        if not destination.is_relative_to(output_root):
            raise ValueError("--output-dir must be within c2net output_path")
        args.output_dir = str(destination)
        upload = upload_output
        print(f"C2Net output directory: {args.output_dir}", flush=True)
    else:
        args.output_dir = args.output_dir or os.getenv("OUTPUT_PATH", "outputs")
    if not args.install and not args.train:
        args.install = True
        args.train = True
    if args.install:
        install_environment()
    if args.train:
        train_bc(args)
        if upload is not None:
            upload()
            print("C2Net output upload returned", flush=True)


if __name__ == "__main__":
    main()
