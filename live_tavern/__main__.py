from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from .actions import Geometry, decode, executable
from .observation import Encoder, exact_observation
from .policy import Policy
from .power import LogTail, NotReady, PowerState
from .platform import backend

ROOT = Path(__file__).resolve().parent.parent


def load_config(path):
    return json.loads(path.read_text(encoding="utf-8"))


def emit(data, output=None):
    text = json.dumps(data, ensure_ascii=False)
    print(text, flush=True)
    if output:
        with output.open("a", encoding="utf-8") as f:
            f.write(text + "\n")


def decide(state, encoder, policy, geometry):
    encoder.validate_policy(policy)
    view = state.view()
    fingerprint = state.fingerprint()
    plans = executable(view, geometry, fingerprint)
    obs = encoder.encode(view, plans)
    action, top = policy.predict(obs, obs["action_mask"])
    return plans[action], dict(
        type="decision", action=action, name=decode(action), turn=view["turn"],
        gold=obs["gold"], top=top, coverage=obs["coverage"], unobserved=obs["unobserved"],
        catalog_sha256=encoder.sha256, model_sha256=policy.sha256,
        build=view["build"], plan=plans[action].json())


def replay(args):
    config = load_config(args.config)
    state, policy = PowerState(config["player_name"]), Policy(args.checkpoint)
    encoder, geometry = Encoder(args.catalog), Geometry(config)
    # Decide only at historical client SendOption/SendChoices boundaries. Reading
    # every tag as a snapshot would generate decisions during animations.
    count, waiting = 0, {}
    with args.log.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if "GameState.SendOption()" in line or "GameState.SendChoices()" in line:
                state.settle()
                try:
                    _, record = decide(state, encoder, policy, geometry)
                    emit(record, args.output)
                    count += 1
                except (NotReady, ValueError) as error:
                    waiting[str(error)] = waiting.get(str(error), 0) + 1
            state.ingest(line)
    emit(dict(type="replay_finished", decisions=count, skipped=waiting), args.output)
    return 0 if count else 2


def watch(args):
    config = load_config(args.config)
    geometry, encoder, policy = Geometry(config), Encoder(args.catalog), Policy(args.checkpoint)
    encoder.validate_policy(policy)
    if args.execute and not args.allow_estimated_context:
        raise ValueError("Power.log context is incomplete; pilot execution requires --allow-estimated-context")
    if args.execute and not config.get("calibrated"):
        raise ValueError("run calibrate against your actual client first")
    state = PowerState(config["player_name"])
    tail = LogTail(Path(os.path.expanduser(os.path.expandvars(config["log_root"]))), state)
    windows = None
    if args.execute:
        windows = backend(config, args.stop_file)
    last_data = time.monotonic()
    last_decision = None
    pending = None
    waiting = None
    ended_turn = None
    action_turn, action_count = None, 0
    try:
        while True:
            if args.stop_file.exists():
                break
            if windows:
                windows.check_stop()
            try:
                changed = tail.poll()
                now = time.monotonic()
                if changed:
                    last_data = now
                    continue
                if now - last_data < args.settle_seconds:
                    time.sleep(0.05)
                    continue
                if tail.pending:
                    raise NotReady("waiting for the incomplete log line")
                state.settle()
                fingerprint = state.fingerprint()
                if pending:
                    old_fingerprint, old_sent, started = pending
                    # An acknowledged client action PLUS a changed settled state is
                    # required. Unrelated tag changes alone are not an acknowledgement.
                    if state.sent > old_sent and fingerprint != old_fingerprint:
                        pending = None
                    elif now - started > args.action_timeout:
                        emit(dict(type="stopped", reason="action not acknowledged before timeout"), args.output)
                        return 3
                    else:
                        time.sleep(0.05)
                        continue
                if fingerprint == last_decision:
                    time.sleep(0.05)
                    continue
                view = state.view()
                if ended_turn == (state.generation, view["turn"]):
                    time.sleep(0.05)
                    continue
                if windows:
                    if tail.path is None or time.time() - tail.path.stat().st_mtime > 15:
                        raise NotReady("live log is stale")
                    if state.build not in config.get("accepted_builds", []):
                        raise NotReady("client build needs explicit verification in accepted_builds")
                    key = (state.generation, view["turn"])
                    if action_turn != key:
                        action_turn, action_count = key, 0
                    if action_count >= args.max_actions_per_turn:
                        emit(dict(type="stopped", reason="pilot per-turn action limit reached"), args.output)
                        return 3
                plan, record = decide(state, encoder, policy, geometry)
                record["mode"] = "execute" if windows else "recommend"
                emit(record, args.output)
                if windows:
                    # Foreground or stale-state failures stop execution; there is no
                    # automatic retry of a possibly partly completed gesture.
                    before_sent = state.sent
                    try:
                        windows.execute(plan, state, tail)
                    except NotReady as error:
                        emit(dict(type="stopped", reason=str(error)), args.output)
                        return 3
                    pending = (fingerprint, before_sent, time.monotonic())
                    action_count += 1
                    if plan.action == 0:
                        ended_turn = (state.generation, view["turn"])
                last_decision = fingerprint
                waiting = None
            except NotReady as error:
                if waiting != str(error):
                    emit(dict(type="waiting", reason=str(error)), args.output)
                    waiting = str(error)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        if windows:
            windows.stop.set()
    emit(dict(type="stopped", reason="user stop"), args.output)
    return 0


def calibrate(args):
    from .actions import Geometry
    config = load_config(args.config)
    if (sys.platform == "darwin" and config.get("calibrated")
            and config.get("coordinate_space") != "quartz_window_points"):
        raise ValueError("cannot reuse Windows calibration on Mac; start with an empty Mac config")
    windows = backend(config, args.stop_file)
    points, size = [], None
    try:
        for i in range(args.count if args.row else 1):
            print(f"Point {i + 1}: focus Hearthstone, place pointer, press F6 (F8 stops).", flush=True)
            point, this_size = windows.capture_point()
            if size is not None and size != this_size:
                raise ValueError("client size changed during calibration")
            Geometry.check(point)
            points.append(point)
            size = this_size
        old_size = config.get("client_size")
        if config.get("calibrated") and old_size != size:
            raise ValueError("size differs; start a new empty calibration config")
        if args.row:
            config.setdefault("rows", {}).setdefault(args.row, {})[str(args.count)] = points
        else:
            config.setdefault("points", {})[args.point] = points[0]
        config["calibrated"], config["client_size"] = True, size
        if sys.platform == "darwin":
            config["coordinate_space"] = "quartz_window_points"
        Geometry(config)
        temp = args.config.with_suffix(".json.tmp")
        temp.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(args.config)
        emit(dict(type="calibrated", name=args.row or args.point, points=points, size=size))
    finally:
        windows.stop.set()
    return 0


def main():
    parser = argparse.ArgumentParser(description="Trained-model Battlegrounds macOS/Windows UI pilot")
    parser.add_argument("--config", type=Path, default=ROOT / "live_tavern" / (
        "config.mac.local.json" if sys.platform == "darwin" else "config.local.json"))
    parser.add_argument("--catalog", type=Path, default=ROOT / "crates/taverngym/data/cards.json")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stop-file", type=Path, default=ROOT / "live_tavern/STOP")
    commands = parser.add_subparsers(dest="command", required=True)
    w = commands.add_parser("watch", help="follow Power.log; recommendation mode by default")
    w.add_argument("--execute", action="store_true")
    w.add_argument("--allow-estimated-context", action="store_true")
    w.add_argument("--settle-seconds", type=float, default=0.7)
    w.add_argument("--action-timeout", type=float, default=8.)
    w.add_argument("--max-actions-per-turn", type=int, default=60)
    r = commands.add_parser("replay", help="replay historical Power.log without UI input")
    r.add_argument("log", type=Path)
    s = commands.add_parser("sim-replay", help="infer from exact saved simulator JSON")
    s.add_argument("state", type=Path)
    s.add_argument("--seat", type=int, default=0)
    c = commands.add_parser("calibrate", help="record client-relative coordinates with F6")
    group = c.add_mutually_exclusive_group(required=True)
    group.add_argument("--point", choices=("end", "refresh", "freeze", "upgrade", "power0",
                                         "power1", "dark_gift", "neutral", "hand_drop",
                                         "sell_drop", "spell_drop"))
    group.add_argument("--row", choices=("board", "board_gaps", "hand", "shop", "choice"))
    c.add_argument("--count", type=int, default=1)
    commands.add_parser("doctor", help="validate checkpoint, vocabulary and calibration")
    commands.add_parser("ui-doctor", help="read macOS permissions, windows and log path; no input")
    args = parser.parse_args()
    if args.command in ("watch", "replay", "sim-replay", "doctor") and not args.checkpoint:
        parser.error("--checkpoint is required before the subcommand")
    if args.command == "watch" and (args.settle_seconds < 0.2 or args.action_timeout <= 0
                                   or args.max_actions_per_turn < 1):
        parser.error("settle-seconds must be >=0.2 and action-timeout >0")
    if args.command == "calibrate" and not 1 <= args.count <= 16:
        parser.error("--count must be in [1,16]")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "watch":
        return watch(args)
    if args.command == "replay":
        return replay(args)
    if args.command == "calibrate":
        return calibrate(args)
    if args.command == "ui-doctor":
        from .macos import diagnose
        config = load_config(args.config)
        result = diagnose(config)
        log = Path(os.path.expanduser(os.path.expandvars(config["log_root"])))
        result.update(log_root=str(log), log_root_exists=log.exists(),
                      log_config=str(Path.home() / "Library/Preferences/Blizzard/Hearthstone/log.config"))
        emit(dict(type="ui-doctor", **result), args.output)
        return 0
    if args.command == "sim-replay":
        policy = Policy(args.checkpoint)
        obs = exact_observation(args.state.read_text(), args.seat)
        action, top = policy.predict(obs, obs["action_mask"])
        emit(dict(action=action, name=decode(action), top=top,
                  coverage=obs["coverage"], model_sha256=policy.sha256), args.output)
        return 0
    config = load_config(args.config)
    Geometry(config)
    encoder, policy = Encoder(args.catalog), Policy(args.checkpoint)
    encoder.validate_policy(policy)
    emit(dict(type="doctor", model_config=asdict(policy.config), parameters=policy.parameters,
              catalog_sha256=encoder.sha256, model_sha256=policy.sha256,
              catalog_client=encoder.client_version, calibrated=config.get("calibrated", False),
              rows={k: sorted(v) for k, v in config.get("rows", {}).items()},
              points=sorted(config.get("points", {})), ui_platform=sys.platform))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, OSError) as error:
        print(f"live_tavern: {error}", file=sys.stderr)
        sys.exit(2)
