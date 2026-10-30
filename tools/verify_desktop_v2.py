#!/usr/bin/env python3
"""Synthetic lifecycle, real checkpoint and mocked desktop acceptance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from desktop_tavern.controller import Controller
from desktop_tavern.heroes import HeroStats
from desktop_tavern.input import execute
from desktop_tavern.session import SessionLock
from desktop_tavern.state import DesktopState, Phase
from desktop_tavern.vision import Evidence, Vision, capture
from live_tavern.actions import Geometry
from live_tavern.observation import Encoder
from live_tavern.policy import Policy
from live_tavern.power import NotReady
from verify_live_script import config, fixture
from verify_mac_live_script import Quartz, adapter


def hero_fixture():
    prefix = "D 12:00:00.0000000 GameState.DebugPrintPower() - "
    lines = [prefix + "CREATE_GAME", prefix + "GameEntity EntityID=1",
             prefix + "tag=STATE value=RUNNING",
             prefix + "Player EntityID=2 PlayerID=2"]
    lines += ["D 12:00:00 GameState.DebugPrintGame() - GameType=GT_BATTLEGROUNDS",
              "D 12:00:00 GameState.DebugPrintGame() - BuildNumber=253932",
              "D 12:00:00 GameState.DebugPrintGame() - PlayerID=2, PlayerName=Pilot#1234"]
    for index in range(4):
        lines += [prefix + f"FULL_ENTITY - Creating ID={100 + index} CardID=TEST_HERO_{index}"]
        tags = dict(CARDTYPE="HERO", CONTROLLER=2, ZONE="HAND", ZONE_POSITION=index + 1,
                    ARMOR=10 - index, BACON_LOCKED_MULLIGAN_HERO=int(index == 0))
        if index == 2:
            tags["BACON_SKIN_PARENT_ID"] = 12345
        lines += [prefix + f"tag={name} value={value}" for name, value in tags.items()]
    lines += ["D 12:00:00 GameState.DebugPrintEntityChoices() - id=1 ChoiceType=MULLIGAN"]
    # Log list order is deliberately different from physical UI order.
    lines += [f"D 12:00:00 GameState.DebugPrintEntityChoices() - Entities[{i}]={eid}"
              for i, eid in enumerate((102, 100, 103, 101))]
    return "\n".join(lines) + "\n"


def fixture_config():
    cfg = config()
    cfg.update(client_size=[1200, 800], accepted_builds=["253932"],
               auto_queue=False, auto_settle=False, max_games=1, action_timeout=10)
    cfg["rows"]["heroes"] = {"4": [[.2, .5], [.4, .5], [.6, .5], [.8, .5]]}
    cfg["points"].update(queue=[.8, .8], settle=[.5, .8], hero_confirm=[.5, .8])
    return cfg


def state_from(text):
    state = DesktopState("Pilot#1234")
    for line in text.splitlines():
        state.ingest(line)
    state.settle()
    return state


def reject(function, text):
    try:
        function()
    except (NotReady, RuntimeError, ValueError) as error:
        assert text in str(error), str(error)
        return
    raise AssertionError(f"expected rejection: {text}")


class FastStop:
    def __init__(self):
        self.stopped = False

    def wait(self, _):
        return self.stopped

    def set(self):
        self.stopped = True

    def is_set(self):
        return self.stopped


class FakeMouse:
    FAILSAFE = True

    def __init__(self):
        self.pos, self.events, self.held = (400, 300), [], False
        self.on_move = None

    def position(self):
        return self.pos

    def moveTo(self, x, y):
        self.pos = (x, y)
        self.events.append(("move", x, y))
        if self.on_move:
            self.on_move()

    def mouseDown(self):
        self.held = True
        self.events.append(("down",))

    def mouseUp(self):
        if self.held:
            self.events.append(("up",))
            self.held = False

    def click(self, button):
        self.events.append(("click", button))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    checks = []
    policy, encoder = Policy(args.checkpoint), Encoder(ROOT / "crates/taverngym/data/cards.json")
    encoder.validate_policy(policy)
    cfg = fixture_config()
    records, clock = [], [0.]
    emit = lambda row, state=None: records.append(row)
    make = lambda settings=cfg, stats=None: Controller(
        settings, encoder, policy, Geometry(settings), stats or HeroStats(), emit,
        clock=lambda: clock[0])
    hero = state_from(hero_fixture())
    assert hero.phase() == Phase.HERO
    assert [e.id for e in hero.heroes()] == [100, 101, 102, 103]
    index, entity, reason = HeroStats().choose(hero.heroes())
    assert index == 1 and entity.id == 101 and reason["basis"].startswith("visible")
    checks.append("MULLIGAN before TURN/gold, physical UI ordering and locked hero filtering")
    hero.entities[101].tags["CONTROLLER"] = "8"
    reject(hero.heroes, "another player")
    hero.entities[101].tags["CONTROLLER"] = "2"
    hero.entities[102].tags["ZONE_POSITION"] = "0"
    reject(hero.heroes, "physical slots")
    hero.entities[102].tags["ZONE_POSITION"] = "3"
    checks.append("unknown ownership and incomplete hero slots rejected")
    with tempfile.TemporaryDirectory(prefix="desktop-v2-acceptance-") as tmp:
        root = Path(tmp)
        stats_path = root / "stats.json"
        stats_path.write_text(json.dumps({"heroes": [{"dbf_id": 12345, "avg_placement": 3.8}]}))
        stats = HeroStats(stats_path)
        assert stats.choose(hero.heroes())[0] == 2
        stats_path.write_text('{"heroes":[{"card_id":"A","avg_placement":0}]}')
        reject(lambda: HeroStats(stats_path), "[1,8]")
        checks.append("skin parent DBF statistics and invalid statistic rejection")
        control = make(stats=stats)
        plan = control.prepare(hero)
        assert plan.entity == 102 and plan.operations[0]["point"] == [.6, .5]
        control.sent(plan, hero)
        assert control.prepare(hero) is None
        ev = Evidence("hero_confirm", 0, 0, (0, 0, 1200, 800), "synthetic")
        confirm = control.prepare(hero, ev)
        assert confirm.kind == "hero_confirm"
        control.sent(confirm, hero)
        assert control.prepare(hero, ev) is None
        hero.ingest("D 12:00:01 GameState.SendChoices() - id=1 ChoiceType=MULLIGAN")
        assert control.prepare(hero) is None and control.pending is None
        checks.append("hero select -> confirm -> SendChoices, no repeated hero clicks")
        hero = state_from(hero_fixture())
        control = make()
        control.sent(control.prepare(hero), hero)
        clock[0] = 11
        reject(lambda: control.prepare(hero), "timed out")
        clock[0] = 0
        checks.append("hero acknowledgement timeout stops without retry")
        recruit = state_from(fixture())
        control = make()
        buy = control.prepare(recruit)
        assert buy.action == 8 and buy.kind == "buy"
        assert policy.parameters == 2992951
        control.sent(buy, recruit)
        recruit.ingest("D 12:00:01 GameState.DebugPrintPower() - TAG_CHANGE Entity=2 tag=RESOURCES value=2")
        assert control.prepare(recruit) is None  # unrelated state is not an ack
        recruit.ingest("D 12:00:01 GameState.SendOption() - selectedOption=0")
        assert control.prepare(recruit) is None and control.pending is None
        checks.append("917730 checkpoint -> legal buy; tag-only changes cannot acknowledge a gesture")
        recruit.entities[1].tags["TURN"] = "2"
        assert recruit.phase() == Phase.COMBAT
        recruit.entities[1].tags["STATE"] = "COMPLETE"
        assert recruit.phase() == Phase.RESULT
        control.prepare(recruit)
        assert control.finished
        checks.append("recruit/combat/result lifecycle and one-game cap")
        blank = DesktopState("Pilot#1234")
        settings = dict(cfg, auto_queue=True)
        control = make(settings)
        lobby = Evidence("lobby", 0, 0, (0, 0, 1200, 800), "synthetic")
        queue = control.prepare(blank, lobby)
        assert queue.kind == "queue"
        control.sent(queue, blank)
        assert control.prepare(blank, lobby) is None
        queued = Evidence("queue", 0, 0, lobby.rectangle, "synthetic")
        assert control.prepare(blank, queued) is None and control.pending is None
        assert control.prepare(blank, lobby) is None
        blank.revision += 1
        blank.sent += 1
        assert control.prepare(blank, lobby) is None  # fingerprint jitter cannot requeue
        checks.append("queue needs explicit enable/evidence; duplicate queue guarded")
        settings = dict(cfg, auto_settle=True)
        control = make(settings)
        active = state_from(fixture())
        control.prepare(active)
        active.entities[1].tags["STATE"] = "COMPLETE"
        result = Evidence("result", 0, 0, lobby.rectangle, "synthetic")
        settle = control.prepare(active, result)
        assert settle.kind == "settle"
        control.sent(settle, active)
        control.prepare(active, lobby)
        assert control.finished and control.pending is None
        checks.append("result dismissal acknowledged by lobby, stops at configured game cap")
        frame = Image.fromarray(np.random.default_rng(17).integers(0, 256, (800, 1200, 3), dtype=np.uint8))
        roi = [0, 0, .2, .2]
        frame.crop((0, 0, 240, 160)).save(root / "template.png")
        settings = dict(cfg, scenes={"lobby": dict(roi=roi, template="template.png")})
        vision = Vision(settings, root, clock=lambda: clock[0])
        hit = vision.match(frame, (10, 20, 1200, 800))
        assert hit.scene == "lobby" and hit.score == 0
        reject(lambda: vision.match(frame, (10, 20, 600, 400)), "size")
        ambiguous = dict(settings, scenes=dict(
            lobby=settings["scenes"]["lobby"], result=settings["scenes"]["lobby"]))
        reject(lambda: Vision(ambiguous, root).match(frame, hit.rectangle), "multiple")
        Image.new("RGB", (240, 160), "white").save(root / "uniform.png")
        reject(lambda: Vision(dict(cfg, scenes={"lobby": dict(roi=roi, template="uniform.png")}),
                              root), "uniform")
        clock[0] = 2
        reject(lambda: vision.require(hit, None), "stale")
        clock[0] = 0
        with patch.object(vision, "read", return_value=None):
            reject(lambda: vision.require(hit, None), "changed")
        checks.append("fresh unique screenshot template, stale/ambiguous/changed/uniform/size rejection")
        lock = SessionLock(root / "session.lock")
        reject(lambda: SessionLock(root / "session.lock"), "another")
        lock.close()
        with SessionLock(root / "session.lock"):
            pass
        checks.append("OS session lock contention and release")
        # Actual Windows rectangle code, mocked Win32 and mouse only.
        from live_tavern.windows import Windows
        driver = Windows.__new__(Windows)
        driver.stop, driver.stop_file, driver.mouse = FastStop(), root / "STOP", FakeMouse()
        driver.config = dict(cfg, calibrated=True, window_title="Hearthstone",
                             coordinate_space="windows_client_pixels")
        def client_rect(_, ptr):
            ptr._obj.right, ptr._obj.bottom = 1200, 800
            return 1
        def screen(_, ptr):
            ptr._obj.x, ptr._obj.y = -1600, 120
            return 1
        driver.user32 = SimpleNamespace(FindWindowW=lambda *_: 42, IsIconic=lambda _: 0,
                                       GetForegroundWindow=lambda: 42, GetClientRect=client_rect,
                                       ClientToScreen=screen)
        assert driver.rectangle() == (-1600, 120, 1200, 800)
        clean_tail = SimpleNamespace(poll=lambda: 0, pending=b"")
        recruit = state_from(fixture())
        control = make()
        buy = control.prepare(recruit)
        execute(driver, buy, recruit, clean_tail, lambda: None)
        assert ("down",) in driver.mouse.events and ("up",) in driver.mouse.events
        assert not driver.mouse.held
        driver.mouse = FakeMouse()
        driver.mouse.on_move = lambda: setattr(recruit, "revision", recruit.revision + 1)
        reject(lambda: execute(driver, buy, recruit, clean_tail, lambda: None), "changed")
        assert ("down",) not in driver.mouse.events
        checks.append("Windows client rectangle/negative monitor gesture and pre-click state invalidation")
        driver.stop_file.touch()
        try:
            driver.check_stop()
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError("STOP was ignored")
        checks.append("persistent STOP rejects input")
        q = Quartz()
        mac = adapter(q, dict(cfg, calibrated=True, coordinate_space="quartz_window_points",
                              window_title="Hearthstone"), root / "mac-STOP")
        mac.stop = FastStop()
        try:
            recruit = state_from(fixture())
            buy = make().prepare(recruit)
            execute(mac, buy, recruit, clean_tail, lambda: None)
            assert any(e[0] == q.kCGEventLeftMouseDown for e in q.events)
            assert any(e[0] == q.kCGEventLeftMouseUp for e in q.events)
            assert mac.mouse.held is None
            q.events.clear()
            hero = state_from(hero_fixture())
            control = make()
            select = control.prepare(hero)
            panel = Evidence("heroes", 0, 0, mac.rectangle(), "synthetic")
            with patch.object(vision, "read", return_value=panel):
                execute(mac, select, hero, clean_tail,
                        lambda: control.guard(select, hero, panel, vision, mac))
            assert any(e[0] == q.kCGEventLeftMouseDown for e in q.events)
            assert not hero.options_ready and hero.choice_ready
            q.events.clear()
            q.on_post = lambda e: q.keys.add(100) if e[0] == q.kCGEventLeftMouseDown else None
            try:
                execute(mac, buy, recruit, clean_tail, lambda: None)
            except KeyboardInterrupt:
                pass
            else:
                raise AssertionError("F8 during drag was ignored")
            assert mac.mouse.held is None
            assert any(e[0] == q.kCGEventLeftMouseUp for e in q.events)
        finally:
            mac.stop.set()
        checks.append("Mac Quartz logical coordinates, F8 drag release and choice-only hero input (mocked)")
        # Screenshot dimensions normalize into the same coordinate space.
        simple = SimpleNamespace(rectangle=lambda: (0, 0, 1200, 800))
        with patch("desktop_tavern.vision.ImageGrab.grab", return_value=frame.resize((2400, 1600))):
            image, rect = capture(simple)
            assert image.size == (1200, 800)
        checks.append("screenshot coordinate normalization")
        hero = state_from(hero_fixture())
        control = make()
        plan = control.prepare(hero)
        reject(lambda: control.guard(plan, hero, None, vision, None), "template")
        recruit = state_from(fixture())
        buy = make().prepare(recruit)
        reject(lambda: control.guard(buy, recruit, lobby, vision, None), "non-recruit")
        checks.append("hero execution needs screenshot panel confirmation")
        settings = dict(cfg, max_actions_per_turn=1)
        control = make(settings)
        recruit = state_from(fixture())
        buy = control.prepare(recruit)
        control.sent(buy, recruit)
        recruit.ingest("D 12:00:01 GameState.SendOption() - selectedOption=0")
        # New options after an acknowledged gesture, still in the same turn.
        for line in fixture().splitlines()[-2:]:
            recruit.ingest(line)
        recruit.settle()
        reject(lambda: control.prepare(recruit), "per-turn")
        reject(lambda: make(dict(cfg, auto_queue="false")), "boolean")
        checks.append("per-turn action cap and strict automation switch validation")
        log, config_path = root / "Power.log", root / "geometry.json"
        log.write_text(hero_fixture() + "D 12:00:01 GameState.SendChoices() - id=1 ChoiceType=MULLIGAN\n"
                       + fixture() + "D 12:00:02 GameState.SendOption() - selectedOption=0\n")
        config_path.write_text(json.dumps(cfg))
        replay = subprocess.run([sys.executable, "-m", "desktop_tavern", "--config", str(config_path),
                                 "--checkpoint", str(args.checkpoint), "--runs", str(root / "runs"),
                                 "replay", str(log)], cwd=ROOT, capture_output=True, text=True, check=True)
        rows = [json.loads(line) for line in replay.stdout.splitlines()]
        assert rows[-1]["decisions"] == 2
        checks.append("standalone desktop V2 CLI synthetic hero+recruit replay and session snapshots")
    report = dict(status="passed", checks=checks, parameters=policy.parameters,
                  model_sha256=policy.sha256, catalog_sha256=encoder.sha256,
                  platforms=dict(mac="mocked Quartz input; native diagnostics separate",
                                 windows="mocked Win32 rectangle and mouse"),
                  live_client_tested=False, real_matches_started=0,
                  limitations=["Power.log leaves model counters/opponent context unobserved",
                               "actual Mac and Windows game clients not tested",
                               "Intel Mac dependencies not validated",
                               "hero/general target-sensitive workflows require calibration"])
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(status="passed", checks=len(checks), report=str(args.report))))


if __name__ == "__main__":
    main()
