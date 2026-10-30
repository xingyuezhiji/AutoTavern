from __future__ import annotations

from enum import Enum
import hashlib
import re

from live_tavern.power import NotReady, PowerState


class Phase(str, Enum):
    LOBBY = "lobby"
    QUEUE = "queue"
    HERO = "heroes"
    RECRUIT = "recruit"
    CHOICE = "choice"
    COMBAT = "combat"
    RESULT = "result"
    WAIT = "waiting"
    STOP = "stopped"
    ERROR = "error"


class DesktopState(PowerState):
    def reset(self):
        super().reset()
        self.choice_type = ""
        self.sent_choices = 0

    def ingest(self, line):
        if "GameState.DebugPrintEntityChoices()" in line:
            match = re.search(r"\bChoiceType=(\w+)", line)
            if match:
                self.choice_type = match[1]
        if "GameState.SendChoices()" in line:
            self.sent_choices += 1
        super().ingest(line)

    def choice_entities(self):
        if self.choice_id is None or not self.choices or self.depth:
            raise NotReady("choice entities are not settled")
        keys = sorted(self.choices)
        if keys != list(range(len(keys))):
            raise NotReady("choice indices are incomplete")
        cards = [self.entities.get(self.choices[k]) for k in keys]
        if any(e is None or not e.card for e in cards):
            raise NotReady("choice card identity missing")
        owners = {e.number("CONTROLLER") for e in cards}
        if self.local_pid is None or owners != {self.local_pid}:
            raise NotReady("choice belongs to another player or ownership is unknown")
        if self.choice_pid not in (None, self.local_pid):
            raise NotReady("choice header belongs to another player")
        return cards

    def heroes(self):
        if self.choice_type != "MULLIGAN":
            raise NotReady("not a hero choice")
        cards = sorted(self.choice_entities(), key=lambda e: e.number("ZONE_POSITION"))
        if any(e.tags.get("CARDTYPE") != "HERO" or e.tags.get("ZONE") != "HAND"
               for e in cards):
            raise NotReady("hero offers must be visible in local HAND")
        if [e.number("ZONE_POSITION") for e in cards] != list(range(1, len(cards) + 1)):
            raise NotReady("hero physical slots missing or noncontiguous")
        if not any(not e.number("BACON_LOCKED_MULLIGAN_HERO") for e in cards):
            raise NotReady("all hero offers are locked")
        return cards

    def settle(self):
        super().settle()
        # MULLIGAN can lack ChoiceCardMgr messages. Structural evidence is enough
        # for a recommendation; execution additionally needs a calibrated panel.
        if self.choice_type == "MULLIGAN" and self.choices:
            try:
                self.heroes()
                self.choice_ready = True
                self.choice_pid = self.local_pid
            except NotReady:
                self.choice_ready = False
        elif self.choices:
            try:
                self.choice_entities()
                self.choice_pid = self.local_pid
            except NotReady:
                self.choice_ready = False

    def phase(self, scene=None):
        if self.fault:
            return Phase.ERROR
        game = self.entities.get(self.game_id)
        if game and self.game_type == "GT_BATTLEGROUNDS" and self.local_pid in self.players:
            player = self.entities[self.players[self.local_pid]]
            if (game.tags.get("STATE") == "COMPLETE"
                    or player.tags.get("PLAYSTATE") in ("WON", "LOST", "TIED", "CONCEDED")):
                return Phase.RESULT if scene not in ("lobby", "queue") else Phase(scene)
            if self.depth:
                return Phase.WAIT
            if self.choices:
                if not self.choice_ready:
                    return Phase.WAIT
                return Phase.HERO if self.choice_type == "MULLIGAN" else Phase.CHOICE
            turn = game.number("TURN")
            if turn > 0:
                return Phase.RECRUIT if turn % 2 else Phase.COMBAT
            return Phase.WAIT
        return Phase(scene) if scene in ("lobby", "queue", "result") else Phase.WAIT

    def fingerprint(self):
        raw = f"{super().fingerprint()}:{self.choice_type}:{self.choice_pid}:{self.local_pid}:{self.build}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def snapshot(self):
        return dict(generation=self.generation, build=self.build, local_pid=self.local_pid,
                    choice_type=self.choice_type, choices=self.choices,
                    entities={i: dict(card=e.card, tags=e.tags) for i, e in self.entities.items()},
                    fingerprint=self.fingerprint())
