from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import time

from live_tavern.actions import Geometry
from live_tavern.observation import Encoder
from live_tavern.policy import Policy
from live_tavern.power import LogTail, NotReady
from live_tavern.platform import backend
from .controller import Controller
from .heroes import HeroStats
from .input import execute
from .session import Recorder, SessionLock
from .state import DesktopState, Phase
from .vision import SCENES, Vision, capture, crop

ROOT = Path(__file__).resolve().parent.parent
PLATFORM = "mac" if sys.platform == "darwin" else "windows"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, config):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def resources(args):
    config = load(args.config)
    policy, encoder = Policy(args.checkpoint), Encoder(args.catalog)
    encoder.validate_policy(policy)
    stats_path = config.get("hero_stats")
    heroes = HeroStats(args.config.parent / stats_path if stats_path else None)
    return config, encoder, policy, Geometry(config), heroes


def controller(args, recorder):
    config, encoder, policy, geometry, heroes = resources(args)
    return config, Controller(config, encoder, policy, geometry, heroes, recorder.emit)


class Tail(LogTail):
    def poll(self):
        try:
            return super().poll()
        except NotReady:
            # A fresh installation can calibrate/queue before Power.log exists.
            # Once a game is known, disappearing logs must not allow more input.
            if self.state.game_id:
                raise
            return 0


def replay(args):
    recorder = Recorder(args.runs)
    config, control = controller(args, recorder)
    state = DesktopState(config["player_name"])
    decisions, skipped = 0, {}
    for line in args.log.read_text(encoding="utf-8").splitlines():
        if "GameState.SendOption()" in line or "GameState.SendChoices()" in line:
            state.settle()
            try:
                if control.prepare(state):
                    decisions += 1
            except (NotReady, ValueError) as error:
                skipped[str(error)] = skipped.get(str(error), 0) + 1
        state.ingest(line)
    recorder.emit(dict(type="replay_finished", decisions=decisions, skipped=skipped,
                       synthetic_or_user_supplied=True))
    return 0 if decisions else 2


def run(args):
    recorder = Recorder(args.runs)
    config, control = controller(args, recorder)
    vision = Vision(config, args.config.parent)
    if args.execute and not args.allow_estimated_context:
        raise ValueError("pilot execution requires --allow-estimated-context; Power.log omits model context")
    if args.execute and not config.get("calibrated"):
        raise ValueError("calibrate this platform/window before execution")
    if args.execute:
        check_platform_calibration(config)
    state = DesktopState(config["player_name"])
    tail = Tail(Path(os.path.expanduser(os.path.expandvars(config["log_root"]))), state)
    driver = backend(config, args.stop_file) if args.execute or args.observe_scenes else None
    last_data, waiting = time.monotonic(), None
    recorder.emit(dict(type="session", mode="execute" if args.execute else "recommend",
                       model_sha256=control.policy.sha256, parameters=control.policy.parameters,
                       estimated_context=True))
    try:
        while not args.stop_file.exists() and not control.finished:
            if driver:
                driver.check_stop()
            try:
                changed = tail.poll()
                now = time.monotonic()
                if changed:
                    last_data = now
                # Pending timeouts remain active even under a constantly changing log.
                if control.pending and now - control.pending.started > config.get("action_timeout", 10):
                    raise RuntimeError(f"{control.pending.kind}: acknowledgement timed out")
                if changed or now - last_data < args.settle_seconds or tail.pending:
                    time.sleep(.05)
                    continue
                state.settle()
                evidence = vision.read(driver) if driver and vision.templates else None
                plan = control.prepare(state, evidence)
                if plan and args.execute:
                    def guard():
                        if plan.kind not in ("queue", "settle"):
                            if tail.path is None or time.time() - tail.path.stat().st_mtime > 15:
                                raise NotReady("live Power.log is stale")
                        control.guard(plan, state, evidence, vision, driver)
                    execute(driver, plan, state, tail, guard)
                    control.sent(plan, state)
                    recorder.emit(dict(type="executed", plan=plan.json()), state)
                waiting = None
            except NotReady as error:
                # An interrupted gesture must never be automatically retried.
                if 'plan' in locals() and plan and args.execute:
                    raise RuntimeError(str(error)) from error
                if waiting != str(error):
                    recorder.emit(dict(type="waiting", reason=str(error)), state)
                    waiting = str(error)
            finally:
                plan = None
            time.sleep(.1)
    except KeyboardInterrupt:
        recorder.emit(dict(type="stopped", reason="F8 / Ctrl+C / STOP"), state)
    except (RuntimeError, ValueError, OSError) as error:
        recorder.emit(dict(type="error", reason=str(error)), state)
        return 3
    finally:
        if driver:
            driver.stop.set()
            driver.mouse.mouseUp()
    recorder.emit(dict(type="stopped", reason="session finished or STOP"), state)
    return 0


def calibrate(args):
    config = load(args.config)
    check_platform_calibration(config)
    driver = backend(config, args.stop_file)
    try:
        points, size = [], None
        count = 2 if args.scene else args.count if args.row else 1
        for index in range(count):
            print(f"Point {index + 1}/{count}: focus Hearthstone, point mouse, press F6 (Fn+F6 on Mac).",
                  flush=True)
            point, new_size = driver.capture_point()
            if size and size != new_size:
                raise ValueError("window changed during calibration")
            points.append(Geometry.check(point))
            size = new_size
        if config.get("calibrated") and config.get("client_size") != size:
            raise ValueError("window size differs; use a new config and calibrate again")
        config.update(calibrated=True, client_size=size,
                      coordinate_space="quartz_window_points" if sys.platform == "darwin"
                      else "windows_client_pixels")
        if args.scene:
            # Choose upper-left and lower-right of a distinct, stable UI element.
            roi = [*points[0], *points[1]]
            frame, _ = capture(driver)
            template = crop(frame, roi)
            folder = args.config.parent / "templates"
            folder.mkdir(exist_ok=True)
            path = folder / f"{PLATFORM}-{args.scene}.png"
            template.save(path)
            config.setdefault("scenes", {})[args.scene] = dict(
                roi=roi, template=str(path.relative_to(args.config.parent)), max_error=.035)
            Vision(config, args.config.parent)
        elif args.row:
            config.setdefault("rows", {}).setdefault(args.row, {})[str(count)] = points
        else:
            config.setdefault("points", {})[args.point] = points[0]
        Geometry(config)
        save(args.config, config)
        print(json.dumps(dict(type="calibrated", size=size,
                              name=args.scene or args.row or args.point)))
    finally:
        driver.stop.set()
    return 0


def check_platform_calibration(config):
    expected = "quartz_window_points" if sys.platform == "darwin" else "windows_client_pixels"
    if config.get("calibrated") and config.get("coordinate_space") != expected:
        raise ValueError("cannot reuse another platform's calibration; use a new empty config")


def doctor(args):
    config, encoder, policy, _, heroes = resources(args)
    vision = Vision(config, args.config.parent)
    result = dict(type="doctor", model_sha256=policy.sha256, parameters=policy.parameters,
                  model_config=asdict(policy.config), catalog_sha256=encoder.sha256,
                  calibrated=config.get("calibrated"), accepted_builds=config.get("accepted_builds"),
                  scenes=sorted(vision.templates), hero_stats_rows=len(heroes.rows),
                  live_client_tested=False, stop_marker=args.stop_file.exists())
    if sys.platform == "darwin":
        from live_tavern.macos import diagnose
        result["ui"] = diagnose(config)
    log = Path(os.path.expanduser(os.path.expandvars(config["log_root"])))
    result.update(log_root=str(log), log_root_exists=log.exists())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description="AutoTavern desktop V2 (HsBgAI lifecycle reference)")
    parser.add_argument("--config", type=Path,
                        default=ROOT / f"desktop_tavern/config.{PLATFORM}.local.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/tavern_bc_policy.pt")
    parser.add_argument("--catalog", type=Path, default=ROOT / "crates/taverngym/data/cards.json")
    parser.add_argument("--runs", type=Path, default=ROOT / "runs/desktop-v2")
    parser.add_argument("--stop-file", type=Path, default=ROOT / "desktop_tavern/STOP")
    commands = parser.add_subparsers(dest="command", required=True)
    watch = commands.add_parser("run")
    watch.add_argument("--execute", action="store_true")
    watch.add_argument("--allow-estimated-context", action="store_true")
    watch.add_argument("--observe-scenes", action="store_true")
    watch.add_argument("--settle-seconds", type=float, default=.7)
    replay_parser = commands.add_parser("replay")
    replay_parser.add_argument("log", type=Path)
    commands.add_parser("doctor")
    commands.add_parser("stop")
    commands.add_parser("reset")
    calibration = commands.add_parser("calibrate")
    group = calibration.add_mutually_exclusive_group(required=True)
    group.add_argument("--scene", choices=SCENES)
    group.add_argument("--point", choices=("queue", "settle", "hero_confirm", "end", "refresh",
                                        "freeze", "upgrade", "power0", "power1", "neutral",
                                        "hand_drop", "sell_drop", "spell_drop", "dark_gift"))
    group.add_argument("--row", choices=("heroes", "board", "board_gaps", "shop", "hand", "choice"))
    calibration.add_argument("--count", type=int, default=1)
    args = parser.parse_args()
    if args.command == "run" and (not math.isfinite(args.settle_seconds) or args.settle_seconds < .2):
        parser.error("--settle-seconds must be >=0.2")
    if args.command == "calibrate" and not 1 <= args.count <= 16:
        parser.error("--count must be in [1,16]")
    args.config = args.config.resolve()
    if args.command == "stop":
        args.stop_file.parent.mkdir(parents=True, exist_ok=True)
        args.stop_file.touch()
        print("STOP recorded. Active input will release its mouse button.")
        return 0
    lock = args.stop_file.parent / "session.lock"
    if args.command == "reset":
        with SessionLock(lock):
            args.stop_file.unlink(missing_ok=True)
        print("STOP cleared.")
        return 0
    if args.command == "doctor":
        return doctor(args)
    if args.command == "replay":
        return replay(args)
    if args.stop_file.exists():
        raise RuntimeError("persistent STOP is set; use reset before restarting")
    with SessionLock(lock):
        return calibrate(args) if args.command == "calibrate" else run(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, OSError) as error:
        print(f"desktop_tavern: {error}", file=sys.stderr)
        sys.exit(2)
