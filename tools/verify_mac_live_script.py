#!/usr/bin/env python3
"""macOS adapter acceptance with mocked input; native diagnostics are read-only."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from live_tavern.actions import executable, Geometry
from live_tavern.macos import MacOS, QuartzMouse, diagnose, permissions, select_rectangle
from live_tavern.power import PowerState, NotReady
from verify_live_script import fixture, config


class Quartz:
    kCGMouseButtonLeft, kCGMouseButtonRight = 0, 1
    kCGEventLeftMouseDown, kCGEventLeftMouseUp = 1, 2
    kCGEventRightMouseDown, kCGEventRightMouseUp = 3, 4
    kCGEventMouseMoved, kCGEventLeftMouseDragged, kCGEventRightMouseDragged = 5, 6, 7
    kCGHIDEventTap, kCGEventSourceStateCombinedSessionState = 0, 0
    kCGWindowListOptionOnScreenOnly, kCGWindowListExcludeDesktopElements = 1, 16
    kCGNullWindowID = 0

    def __init__(self):
        self.events, self.pos = [], (420, 450)
        self.keys = set()
        self.on_post = None
        self.windows = [dict(kCGWindowOwnerPID=42, kCGWindowOwnerName="Hearthstone",
                             kCGWindowName="Hearthstone", kCGWindowLayer=0,
                             kCGWindowBounds=dict(X=-1600, Y=120, Width=1200, Height=800))]

    def CGWindowListCopyWindowInfo(self, *args):
        return self.windows

    def CGEventSourceKeyState(self, _, key):
        return key in self.keys

    def CGEventCreate(self, _):
        return self.pos

    def CGEventGetLocation(self, _):
        return SimpleNamespace(x=self.pos[0], y=self.pos[1])

    def CGEventCreateMouseEvent(self, _, kind, point, button):
        return kind, point, button

    def CGEventPost(self, _, event):
        self.events.append(event)
        self.pos = event[1]
        if self.on_post:
            self.on_post(event)

    def AXIsProcessTrusted(self):
        return True

    def CGPreflightListenEventAccess(self):
        return True

    def CGPreflightScreenCaptureAccess(self):
        return True


def reject(function, text):
    try:
        function()
    except (NotReady, ValueError, RuntimeError) as error:
        assert text in str(error), str(error)
        return
    raise AssertionError(f"expected rejection: {text}")


def adapter(q, cfg, stop_file):
    app = SimpleNamespace(processIdentifier=lambda: 42)
    kit = SimpleNamespace(NSWorkspace=SimpleNamespace(
        sharedWorkspace=lambda: SimpleNamespace(frontmostApplication=lambda: app)))
    with patch("live_tavern.macos.frameworks", return_value=(q, kit, q)):
        return MacOS(cfg, stop_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    checks = []
    cfg = config()
    cfg.update(calibrated=True, client_size=[1200, 800],
               coordinate_space="quartz_window_points", window_title="Hearthstone")
    q = Quartz()
    assert select_rectangle(q.windows, 42, cfg) == (-1600, 120, 1200, 800)
    reject(lambda: select_rectangle(q.windows, 99, cfg), "foreground")
    reject(lambda: select_rectangle(q.windows, 42, dict(cfg, client_size=[2400, 1600])),
           "calibration")
    reject(lambda: select_rectangle(q.windows, 42, dict(cfg, coordinate_space="windows")),
           "calibration")
    dialog = dict(q.windows[0], kCGWindowName="Dialog")
    reject(lambda: select_rectangle([dialog] + q.windows, 42, cfg), "owner/title")
    checks.append("logical Retina coordinates and negative multi-display origins; reject wrong front window, size and Windows calibration")
    with tempfile.TemporaryDirectory(prefix="tavern-mac-verify-") as tmp:
        stop_file = Path(tmp) / "STOP"
        state = PowerState("Pilot#1234")
        for line in fixture().splitlines():
            state.ingest(line)
        state.settle()
        chosen = executable(state.view(), Geometry(cfg), state.fingerprint())[8]
        tail = SimpleNamespace(poll=lambda: None)
        ui = adapter(q, cfg, stop_file)
        try:
            q.pos = (-1000, 520)
            presses = iter((False, True))
            with patch.object(ui, "key", side_effect=lambda key: next(presses) if key == 97 else False):
                assert ui.capture_point() == ([.5, .5], [1200, 800])
            assert not q.events
            checks.append("F6 captures normalized window coordinates without mouse input")
            ui.execute(chosen, state, tail)
            assert any(event[0] == q.kCGEventLeftMouseDragged for event in q.events)
            assert q.events[-1] == (q.kCGEventLeftMouseUp, (-1000, 840), 0)
            assert ui.mouse.held is None
            checks.append("model buy drag uses window-relative logical coordinates and releases button")
            q.events.clear()
            click_plan = SimpleNamespace(
                revision=(state.generation, state.revision), fingerprint=state.fingerprint(),
                operations=[dict(type="click", point=[.8, .5], button="right")])
            ui.execute(click_plan, state, tail)
            assert q.events[-2:] == [(q.kCGEventRightMouseDown, (-640, 520), 1),
                                    (q.kCGEventRightMouseUp, (-640, 520), 1)]
            checks.append("right click dispatches right down/up without extra left release")
            q.events.clear()
            def lose_focus(event):
                if event[0] == q.kCGEventLeftMouseDragged:
                    q.windows.insert(0, dict(q.windows[0], kCGWindowOwnerPID=99))
            q.on_post = lose_focus
            reject(lambda: ui.execute(chosen, state, tail), "foreground")
            assert q.events[-1][0] == q.kCGEventLeftMouseUp and ui.mouse.held is None
            checks.append("focus loss during drag aborts and releases held button")
            q.on_post = None
            q.windows.pop(0)
            q.events.clear()
            def move_window(event):
                if event[0] == q.kCGEventLeftMouseDragged:
                    q.windows[0]["kCGWindowBounds"]["X"] += 10
            q.on_post = move_window
            reject(lambda: ui.execute(chosen, state, tail), "moved")
            assert q.events[-1][0] == q.kCGEventLeftMouseUp and ui.mouse.held is None
            checks.append("window movement during drag aborts and releases held button")
            q.on_post = None
            q.events.clear()
            state.revision += 1
            reject(lambda: ui.execute(chosen, state, tail), "changed")
            assert not q.events
            checks.append("stale log decision rejected with zero injected events")
            for stop in ("key", "file", "corner"):
                if stop == "key":
                    q.keys.add(100)
                elif stop == "file":
                    stop_file.touch()
                else:
                    bounds = q.windows[0]["kCGWindowBounds"]
                    q.pos = bounds["X"], bounds["Y"]
                try:
                    ui.mouse.moveTo(10, 10)
                except KeyboardInterrupt:
                    pass
                else:
                    raise AssertionError("stop was ignored")
                assert not q.events
                q.keys.clear()
                stop_file.unlink(missing_ok=True)
                ui.stop.clear()
            checks.append("F8, STOP and window-corner failsafe prevent mouse injection")
        finally:
            ui.stop.set()
        blocked = Quartz()
        blocked.AXIsProcessTrusted = lambda: False
        reject(lambda: adapter(blocked, cfg, stop_file), "accessibility")
        assert not blocked.events
        checks.append("missing permissions reject UI setup without events or prompts")
    for script in sorted((ROOT / "live_tavern").glob("*mac*")):
        if script.suffix in (".sh", ".command"):
            subprocess.run(["/bin/bash", "-n", str(script)], check=True)
    checks.append("all Mac shell entrypoints pass bash syntax validation")
    if sys.platform == "darwin":
        with tempfile.TemporaryDirectory(prefix="tavern mac shell ") as tmp:
            folder = Path(tmp)
            scripts = folder / "live_tavern"
            scripts.mkdir()
            for name in ("run-mac.sh", "doctor-mac.command", "start-mac.command",
                         "stop-mac.command", "reset-mac.command"):
                shutil.copyfile(ROOT / "live_tavern" / name, scripts / name)
            (scripts / "config.mac.local.json").write_text("{}")
            python = folder / ".venv-live/bin/python"
            python.parent.mkdir(parents=True)
            python.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
            python.chmod(0o755)
            model = folder / "models/tavern_bc_policy.pt"
            model.parent.mkdir()
            model.touch()
            def run(name, *options):
                return subprocess.run(["/bin/bash", str(scripts / name), *options],
                                      capture_output=True, text=True)
            doctor = run("doctor-mac.command")
            assert doctor.returncode == 0 and doctor.stdout.splitlines()[-1] == "ui-doctor", doctor.stderr
            start = run("start-mac.command", "--execute", "--allow-estimated-context")
            assert start.returncode == 0 and start.stdout.splitlines()[-3:] == [
                "watch", "--execute", "--allow-estimated-context"], start.stderr
            assert run("stop-mac.command").returncode == 0
            assert run("start-mac.command").returncode == 2
            assert run("reset-mac.command").returncode == 0
            assert run("start-mac.command").returncode == 0
        checks.append("Mac Bash 3.2 wrappers handle empty model args, paths with spaces, execute flags and STOP/reset")
    native = {}
    if sys.platform == "darwin":
        native = diagnose(cfg)
        checks.append("native Quartz/AppKit permission and window query completed read-only")
    report = dict(passed=checks, native=native, desktop_input_sent=False,
                  live_client_tested=False, coordinate_space="quartz_window_points")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
