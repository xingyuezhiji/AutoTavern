from __future__ import annotations

from dataclasses import dataclass
import math
import time

from live_tavern.__main__ import decide
from live_tavern.actions import Plan
from live_tavern.power import NotReady
from .state import Phase


@dataclass
class Pending:
    kind: str
    fingerprint: str
    generation: int
    sent: int
    sent_choices: int
    started: float
    confirmed: bool = False


class Controller:
    def __init__(self, config, encoder, policy, geometry, heroes, emit, clock=time.monotonic):
        self.config, self.encoder, self.policy = config, encoder, policy
        self.geometry, self.heroes, self.emit, self.clock = geometry, heroes, emit, clock
        self.pending = None
        self.last_phase = None
        self.last_decision = None
        self.games = set()
        self.completed = set()
        self.turn_key, self.actions = None, 0
        self.ended_turn = None
        self.finished = False
        self.queue_cycle = None
        self.settled_generations = set()
        for name in ("auto_queue", "auto_settle"):
            if not isinstance(config.get(name, False), bool):
                raise ValueError(f"{name} must be a JSON boolean")
        for name, default in (("action_timeout", 10), ("max_games", 1),
                              ("max_actions_per_turn", 60)):
            value = config.get(name, default)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive")
            if name != "action_timeout" and int(value) != value:
                raise ValueError(f"{name} must be an integer")

    def point_plan(self, state, kind, point, entity=0):
        return Plan(-1, kind, (state.generation, state.revision), state.fingerprint(),
                    entity, [dict(type="click", point=point, button="left")])

    def prepare(self, state, evidence=None):
        now = self.clock()
        scene = evidence.scene if evidence else None
        phase = state.phase(scene)
        if phase == Phase.ERROR:
            raise RuntimeError(state.fault)
        if phase != self.last_phase:
            self.emit(dict(type="phase", phase=phase.value), state)
            self.last_phase = phase
        if self.finished:
            return None
        if state.game_id and state.game_type == "GT_BATTLEGROUNDS" and state.local_pid in state.players:
            if phase not in (Phase.RESULT, Phase.LOBBY, Phase.QUEUE):
                self.games.add(state.generation)
            elif state.generation in self.games:
                self.completed.add(state.generation)
        if self.pending:
            p = self.pending
            changed = state.fingerprint() != p.fingerprint
            if p.kind == "queue":
                ack = scene == "queue" or (state.generation != p.generation
                                          and phase in (Phase.HERO, Phase.RECRUIT, Phase.COMBAT))
            elif p.kind == "settle":
                ack = scene == "lobby"
            elif p.kind in ("hero_select", "hero_confirm"):
                ack = (state.generation == p.generation and state.sent_choices > p.sent_choices
                       and changed)
            else:
                ack = state.generation == p.generation and state.sent > p.sent and changed
            if ack:
                self.emit(dict(type="ack", kind=p.kind), state)
                self.pending = None
            elif now - p.started > self.config.get("action_timeout", 10):
                raise RuntimeError(f"{p.kind}: acknowledgement timed out; stopped without retry")
            elif (p.kind == "hero_select" and not p.confirmed and scene == "hero_confirm"
                  and phase == Phase.HERO):
                return self.point_plan(state, "hero_confirm", self.geometry.point("hero_confirm"))
            else:
                return None
        if len(self.completed) >= self.config.get("max_games", 1):
            # Allow a configured result dismissal, then stop in the lobby.
            if not (phase == Phase.RESULT and scene == "result"
                    and self.config.get("auto_settle")):
                self.finished = True
                self.emit(dict(type="stopped", reason="session game limit reached"), state)
                return None
        fp = state.fingerprint()
        decision_key = (fp, scene)
        if decision_key == self.last_decision:
            return None
        record = None
        if phase == Phase.HERO:
            offers = state.heroes()
            index, entity, reason = self.heroes.choose(offers)
            plan = self.point_plan(state, "hero_select",
                                   self.geometry.row("heroes", len(offers), index), entity.id)
            record = dict(type="decision", kind=plan.kind, card_id=entity.card,
                          physical_slot=index + 1, **reason)
        elif phase in (Phase.RECRUIT, Phase.CHOICE):
            if not (state.options_ready or state.choice_ready):
                return None
            key = (state.generation, state.entities[state.game_id].number("TURN"))
            if key == self.ended_turn:
                return None
            if key != self.turn_key:
                self.turn_key, self.actions = key, 0
            if self.actions >= self.config.get("max_actions_per_turn", 60):
                raise RuntimeError("per-turn action limit reached")
            plan, record = decide(state, self.encoder, self.policy, self.geometry)
        elif phase == Phase.LOBBY and self.config.get("auto_queue") and scene == "lobby":
            if self.queue_cycle == len(self.completed):
                return None
            if len(self.games) >= self.config.get("max_games", 1):
                self.finished = True
                return None
            plan = self.point_plan(state, "queue", self.geometry.point("queue"))
        elif phase == Phase.RESULT and self.config.get("auto_settle") and scene == "result":
            if state.generation in self.settled_generations:
                return None
            plan = self.point_plan(state, "settle", self.geometry.point("settle"))
        else:
            return None
        if record is None:
            record = dict(type="decision", kind=plan.kind)
        record.update(plan=plan.json())
        self.emit(record, state)
        self.last_decision = decision_key
        return plan

    def sent(self, plan, state):
        if plan.kind == "hero_confirm" and self.pending:
            self.pending.confirmed = True
            self.pending.kind = "hero_confirm"
            return
        if plan.kind == "queue":
            self.queue_cycle = len(self.completed)
        elif plan.kind == "settle":
            self.settled_generations.add(state.generation)
        self.pending = Pending(plan.kind, state.fingerprint(), state.generation,
                               state.sent, state.sent_choices, self.clock())
        if plan.action >= 0:
            self.actions += 1
            if plan.action == 0:
                self.ended_turn = (state.generation, state.entities[state.game_id].number("TURN"))

    def guard(self, plan, state, evidence, vision, driver):
        if state.fault:
            raise NotReady(state.fault)
        if plan.kind in ("queue", "settle", "hero_select", "hero_confirm"):
            expected = dict(queue="lobby", settle="result", hero_select="heroes",
                            hero_confirm="hero_confirm")[plan.kind]
            if evidence is None or evidence.scene != expected:
                raise NotReady(f"{plan.kind} needs a fresh calibrated {expected} template")
            vision.require(evidence, driver)
            if plan.kind in ("hero_select", "hero_confirm"):
                state.heroes()
                if state.build not in self.config.get("accepted_builds", []):
                    raise NotReady("client build is not verified in accepted_builds")
        else:
            if evidence is not None:
                raise NotReady("a non-recruit scene is visible; discard the game gesture")
            if state.phase() not in (Phase.RECRUIT, Phase.CHOICE):
                raise NotReady("game phase changed before input")
            if state.build not in self.config.get("accepted_builds", []):
                raise NotReady("client build is not verified in accepted_builds")
            if state.depth or not (state.options_ready or state.choice_ready):
                raise NotReady("client options/choices are not ready")
