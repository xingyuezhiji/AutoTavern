#!/usr/bin/env python3
"""Offline integration acceptance. No desktop input, matches, or model updates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from live_tavern.actions import Geometry, decode, executable, plan, validate_revision
from live_tavern.observation import Encoder, exact_observation
from live_tavern.policy import Policy
from live_tavern.power import Entity, LogTail, NotReady, PowerState


def rejected(function, text):
    try:
        function()
    except (NotReady, ValueError) as e:
        assert text in str(e), str(e)
        return
    raise AssertionError(f"expected rejection: {text}")


def fixture():
    """Synthetic actual-format log; never represented as a real recorded match."""
    prefix = "D 12:00:00.0000000 GameState.DebugPrintPower() - "
    lines = [prefix + "CREATE_GAME", prefix + "GameEntity EntityID=1",
             prefix + "tag=TURN value=1", prefix + "tag=STATE value=RUNNING"]
    def entity(eid, card, tags, pid=None):
        if pid is None:
            lines.append(prefix + f"FULL_ENTITY - Creating ID={eid} CardID={card}")
        else:
            lines.append(prefix + f"Player EntityID={eid} PlayerID={pid}")
        for tag, value in tags.items():
            lines.append(prefix + f"tag={tag} value={value}")
    entity(2, "", dict(CARDTYPE="PLAYER", CONTROLLER=2, HERO_ENTITY=10, RESOURCES=3), 2)
    entity(3, "", dict(CARDTYPE="PLAYER", CONTROLLER=10, BACON_DUMMY_PLAYER=1), 10)
    lines.extend(["D 12:00:00.0000000 GameState.DebugPrintGame() - GameType=GT_BATTLEGROUNDS",
                  "D 12:00:00.0000000 GameState.DebugPrintGame() - BuildNumber=253932",
                  "D 12:00:00.0000000 GameState.DebugPrintGame() - PlayerID=2, PlayerName=Pilot#1234"])
    entity(10, "TB_BaconShop_HERO_37",
           dict(CARDTYPE="HERO", CONTROLLER=2, ZONE="PLAY", HEALTH=30, PLAYER_TECH_LEVEL=1))
    entity(30, "BG20_100", dict(CARDTYPE="MINION", CONTROLLER=10, ZONE="PLAY",
                                ZONE_POSITION=1, ATK=2, HEALTH=1))
    lines.extend(["D 12:00:00.0000000 GameState.DebugPrintOptions() - id=1",
                  "D 12:00:00.0000000 GameState.DebugPrintOptions() - "
                  "option 0 type=PLAY mainEntity=[id=30 zone=PLAY zonePos=1 cardId=BG20_100 player=10] error=NONE errorParam="])
    return "\n".join(lines) + "\n"


def config():
    return dict(player_name="Pilot#1234", calibrated=False,
                points=dict(hand_drop=[.5, .9], sell_drop=[.5, .25], end=[.88, .47]),
                rows=dict(shop={"1": [[.5, .35]]},
                          board={"2": [[.44, .6], [.56, .6]]},
                          board_gaps={"1": [[.5, .6]], "2": [[.4, .6], [.6, .6]]},
                          hand={"1": [[.5, .85]]},
                          choice={"2": [[.4, .45], [.6, .45]]}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    encoder = Encoder(ROOT / "crates/taverngym/data/cards.json")
    policy = Policy(args.checkpoint)
    encoder.validate_policy(policy)
    checks = []
    state = PowerState("Pilot#1234")
    for line in fixture().splitlines():
        state.ingest(line)
    state.settle()
    geometry = Geometry(config())
    view = state.view()
    plans = executable(view, geometry, state.fingerprint())
    obs = encoder.encode(view, plans)
    assert obs["features"].shape == (4017,) and obs["cards"].shape == (447,)
    assert obs["features"][7] == np.float32(.3)
    assert obs["cards"][36] == encoder.catalog["BG20_100"][0]
    assert np.flatnonzero(obs["action_mask"]).tolist() == [8]
    action, top = policy.predict(obs, obs["action_mask"])
    assert action == 8 and top[0]["probability"] == 1.
    checks.append("real-format entity/options parsing and checkpoint -> buy gesture")
    state.entities[2].tags["RESOURCES"] = "0"
    assert encoder.encode(state.view(), plans)["gold"] == 0
    checks.append("zero gold preserved")
    state.entities[30].tags["ZONE_POSITION"] = "0"
    rejected(state.view, "positions")
    state.entities[30].tags["ZONE_POSITION"] = "1"
    original = state.entities[30].card
    state.entities[30].card = "UNKNOWN_NEW_PATCH_CARD"
    rejected(lambda: encoder.encode(state.view(), plans), "unknown")
    state.entities[30].card = original
    checks.append("unknown card and missing physical position rejected")
    chosen = plans[8]
    state.revision += 1
    rejected(lambda: validate_revision(chosen, state), "changed")
    checks.append("stale decision revision rejected before input")
    # Target identity must resolve to a physical region, not the discover overlay.
    target = Entity(41, original, dict(CARDTYPE="MINION", ATK="2", HEALTH="1"))
    view = state.view()
    view["board"] = [target, Entity(42, original, target.tags.copy())]
    view["revision"] = (state.generation, state.revision)
    target_plan = plan(30, view, geometry, "target", dict(entity=41, region="target"))
    assert target_plan.operations[0]["point"] == [.44, .6]
    move = plan(197, view, geometry, "move", dict(entity=41))  # from 0 to 1
    assert move.operations[0]["end"] == [.6, .6]
    rejected(lambda: plan(198, view, geometry, "bad", dict(entity=41)), "invalid move")
    checks.append("target entities and move insertion geometry")
    assert decode(46) == ("play", 0, 0) and decode(125) == ("play", 9, 7)
    assert decode(126) == ("magnetize", 0, 0) and decode(244) == ("move", 6, 6)
    assert len({decode(i) for i in range(245)}) == 245
    rejected(lambda: decode(-1), "integer")
    checks.append("all 245 action IDs are distinct and range checked")
    with tempfile.TemporaryDirectory(prefix="tavern-live-verify-") as tmp:
        folder = Path(tmp)
        log = folder / "Power.log"
        log.write_text(fixture(), encoding="utf-8")
        tail_state = PowerState("Pilot#1234")
        tail = LogTail(log, tail_state)
        tail.poll()
        before = tail_state.generation
        with log.open("ab") as f:
            f.write(b"D 12:00:00.0000000 GameState.DebugPrintPower() - TAG_CHANGE Entity=2 tag=RESOURCES value=")
        tail.poll()
        assert tail_state.entities[2].number("RESOURCES") == 3
        with log.open("ab") as f:
            f.write(b"0\n")
        tail.poll()
        assert tail_state.entities[2].number("RESOURCES") == 0
        log.write_text("new session\n")
        tail.poll()
        assert tail_state.generation > before and not tail_state.entities
        checks.append("partial lines and log truncation reset")
        log.write_text(fixture() + "D 12:00:01.0000000 GameState.SendOption() - selectedOption=0\n")
        cfg = folder / "config.json"
        cfg.write_text(json.dumps(config()))
        result = subprocess.run([sys.executable, "-m", "live_tavern", "--config", str(cfg),
                                 "--checkpoint", str(args.checkpoint), "replay", str(log)],
                                cwd=ROOT, capture_output=True, text=True, check=True)
        records = [json.loads(s) for s in result.stdout.splitlines()]
        assert records[0]["action"] == 8 and records[-1]["decisions"] == 1
        checks.append("replay CLI emits model decision without UI dependencies")
    native = {}
    try:
        import taverngym
        layout = dict(taverngym.layout())
        assert (layout["OBS2_FLOATS"], layout["OBS2_CARDS"], layout["N_ACTIONS"]) == (4017, 447, 245)
        vocabulary = [dict(index=i, id=x[0], name=x[1], name_zh=x[2], tier=x[3])
                      for i in range(1, taverngym.N_CARDS) for x in (taverngym.card_info(i),)]
        catalog = dict(empty_index=0, cards=vocabulary, details=json.loads(taverngym.catalog_json()))
        digest = hashlib.sha256(json.dumps(catalog, sort_keys=True, separators=(",", ":"),
                                           ensure_ascii=False).encode()).hexdigest()
        assert digest == policy.catalog_sha256
        game = taverngym.Game(seed=7)
        decisions = 0
        while decisions < 24 and not game.is_over():
            for seat in range(game.num_players):
                if game.needs_action(seat):
                    exact = exact_observation(game.state_json(), seat)
                    f, c = game.observe2(seat)
                    assert np.array_equal(exact["features"], np.frombuffer(f, "<f4"))
                    assert np.array_equal(exact["cards"], np.frombuffer(c, "<i4"))
                    a, _ = policy.predict(exact, exact["action_mask"])
                    assert exact["action_mask"][a]
                    game.step(seat, a)
                    decisions += 1
                    if decisions >= 24:
                        break
        native = dict(decisions=decisions, source_hash=taverngym.build_info()["source_hash"],
                      catalog_sha256=digest)
        checks.append("native v2 observation, training vocabulary and 24 legal model steps")
    except ImportError:
        native = dict(skipped="matching native taverngym unavailable")
    report = dict(passed=checks, checkpoint_sha256=policy.sha256,
                  parameters=policy.parameters, architecture=policy.config.architecture,
                  native=native, live_client_tested=False, log_context="estimated",
                  synthetic_fixture=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
