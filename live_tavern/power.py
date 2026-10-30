"""Incremental public Power.log parsing. No process memory or hidden-state reads."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re


class NotReady(ValueError):
    """A visible state cannot currently support a decision."""


def entity_id(text: str) -> int | None:
    text = text.strip()
    if text.isdigit():
        return int(text)
    match = re.search(r"\bid=(\d+)\b", text)
    return int(match[1]) if match else None


@dataclass
class Entity:
    id: int
    card: str = ""
    tags: dict[str, str] = field(default_factory=dict)

    def number(self, tag: str, default: int = 0) -> int:
        try:
            return int(self.tags.get(tag, default))
        except (TypeError, ValueError):
            return default


@dataclass
class Option:
    entity: int
    kind: str
    legal: bool
    targets: list[int] = field(default_factory=list)
    index: int = 0
    has_suboptions: bool = False


class PowerState:
    def __init__(self, player_name: str):
        if not player_name.strip():
            raise ValueError("player_name must identify the local account")
        self.player_name = player_name.strip()
        self.generation = 0
        self.reset()

    def reset(self):
        self.generation += 1
        self.entities: dict[int, Entity] = {}
        self.players: dict[int, int] = {}
        self.names: dict[str, int] = {}
        self.local_pid: int | None = None
        self.game_id: int | None = None
        self.current: int | None = None
        self.options: list[Option] = []
        self.options_ready = False
        self.options_revision = 0
        self.choice_id: int | None = None
        self.choice_pid: int | None = None
        self.choices: dict[int, int] = {}
        self.choice_ready = False
        self.revision = 0
        self.depth = 0
        self.game_type = ""
        self.build = ""
        self.sent = 0
        self.fault: str | None = None

    def resolve(self, value: str) -> int | None:
        if value.strip() == "GameEntity":
            return self.game_id
        return entity_id(value) or self.names.get(value.strip())

    def ingest(self, line: str):
        line = line.rstrip("\r\n")
        if len(line) > 1_048_576:
            self.fault = "oversized log line"
            return
        # A duplicated DebugPrintPower/PowerTaskList CREATE_GAME must not clear state.
        if "GameState.DebugPrintPower() - CREATE_GAME" in line:
            self.reset()
            return
        if "GameType=" in line:
            self.game_type = line.split("GameType=", 1)[1].strip()
        if "BuildNumber=" in line:
            self.build = line.split("BuildNumber=", 1)[1].strip()
        match = re.search(r"PlayerID=(\d+), PlayerName=(.*)", line)
        if match:
            pid, name = int(match[1]), match[2].strip()
            if name == self.player_name:
                self.local_pid = pid
            if pid in self.players:
                self.names[name] = self.players[pid]
        if "GameState.SendOption()" in line or "GameState.SendChoices()" in line:
            self.sent += 1
            self.options_ready = False
            self.options.clear()
            if "GameState.SendChoices()" in line:
                self.choices.clear()
                self.choice_ready = False
            self.revision += 1
        if "GameState.DebugPrintEntityChoices()" in line:
            body = line.split(" - ", 1)[-1]
            match = re.match(r"id=(\d+).*ChoiceType=(\w+)", body)
            if match:
                self.choice_id = int(match[1])
                self.choice_ready = False
                self.choices.clear()
                pid = re.search(r"\bPlayer=(\d+)", body)
                self.choice_pid = int(pid[1]) if pid else None
            match = re.search(r"Entities\[(\d+)\]=(.*)", body)
            if match and self.choice_id is not None:
                eid = entity_id(match[2])
                pid = re.search(r"\bplayer=(\d+)", match[2])
                if eid is not None:
                    self.choices[int(match[1])] = eid
                if pid:
                    self.choice_pid = int(pid[1])
            self.revision += 1
        if "ChoiceCardMgr.WaitThenShowChoices()" in line:
            match = re.search(r"id=(\d+)\s+BEGIN", line)
            if match and int(match[1]) == self.choice_id:
                self.choice_ready = True
                self.revision += 1
        if "GameState.DebugPrintOptions()" in line:
            body = line.split(" - ", 1)[-1]
            if re.fullmatch(r"id=\d+", body):
                self.options.clear()
                self.options_ready = False
                self.options_revision += 1
            match = re.search(
                r"option (\d+) type=(\w+) mainEntity=(.*?) error=(\S+)", body)
            if match:
                eid = self.resolve(match[3])
                # END_TURN can have a non-entity mainEntity.
                self.options.append(Option(eid or 0, match[2], match[4] == "NONE",
                                           index=int(match[1])))
            match = re.search(r"target \d+ entity=(.*?) error=(\S+)", body)
            if match and self.options and match[2] == "NONE":
                eid = self.resolve(match[1])
                if eid is not None:
                    self.options[-1].targets.append(eid)
            if "subOption " in body and self.options:
                self.options[-1].has_suboptions = True
            self.revision += 1
            return
        # Only the authoritative stream; PowerTaskList duplicates entities and blocks.
        if "GameState.DebugPrintPower()" not in line:
            return
        body = line.split(" - ", 1)[-1].strip()
        if body.startswith("BLOCK_START"):
            self.depth += 1
            self.options_ready = False
            self.options.clear()
        elif body.startswith("BLOCK_END"):
            self.depth = max(0, self.depth - 1)
        match = re.match(r"GameEntity EntityID=(\d+)", body)
        if match:
            self.game_id = self.current = int(match[1])
            self.entities[self.current] = Entity(self.current)
        match = re.match(r"Player EntityID=(\d+) PlayerID=(\d+)", body)
        if match:
            self.current = int(match[1])
            self.players[int(match[2])] = self.current
            self.entities[self.current] = Entity(self.current)
        match = re.match(r"FULL_ENTITY - (?:Creating ID=|Updating (?:Entity=)?)(.*?) CardID=(\S*)",
                         body)
        if match:
            eid = self.resolve(match[1])
            if eid is not None:
                self.current = eid
                self.entities[eid] = Entity(eid, match[2])
        match = re.match(r"(?:SHOW_ENTITY|CHANGE_ENTITY) - Updating Entity=(.*?) CardID=(\S*)",
                         body)
        if match:
            eid = self.resolve(match[1])
            if eid is not None:
                self.current = eid
                self.entities.setdefault(eid, Entity(eid)).card = match[2]
        match = re.match(r"TAG_CHANGE Entity=(.*?) tag=(\S+) value=(\S+)", body)
        if match:
            eid = self.resolve(match[1])
            if eid is not None:
                self.entities.setdefault(eid, Entity(eid)).tags[match[2]] = match[3]
            else:
                self.fault = "unresolved entity in TAG_CHANGE"
        match = re.fullmatch(r"tag=(\S+) value=(\S+)", body)
        if match and self.current is not None:
            self.entities[self.current].tags[match[1]] = match[2]
        self.revision += 1

    def settle(self):
        """Called only after a quiet interval, not at every physical line."""
        if self.depth == 0 and self.options:
            self.options_ready = True

    def slots(self, controller: int, zone: str, kinds: tuple[str, ...]) -> list[Entity]:
        out = [e for e in self.entities.values()
               if e.number("CONTROLLER") == controller and e.tags.get("ZONE") == zone
               and e.tags.get("CARDTYPE") in kinds]
        out.sort(key=lambda e: e.number("ZONE_POSITION"))
        positions = [e.number("ZONE_POSITION") for e in out]
        if positions != list(range(1, len(out) + 1)):
            raise NotReady(f"{zone} positions missing or noncontiguous: {positions}")
        return out

    def view(self) -> dict:
        if self.fault:
            raise NotReady(self.fault)
        if self.local_pid is None or self.local_pid not in self.players:
            raise NotReady("local PlayerName must match exactly, including # suffix")
        if self.game_type != "GT_BATTLEGROUNDS":
            raise NotReady("not a solo Battlegrounds game")
        game = self.entities.get(self.game_id)
        if game is None or game.tags.get("STATE") == "COMPLETE":
            raise NotReady("no active game")
        turn = game.number("TURN")
        if turn < 1 or turn % 2 == 0 or self.depth:
            raise NotReady("not a settled recruit phase")
        player = self.entities[self.players[self.local_pid]]
        if "RESOURCES" not in player.tags:
            raise NotReady("gold is not visible")
        bob = [pid for pid, eid in self.players.items()
               if self.entities[eid].number("BACON_DUMMY_PLAYER") == 1]
        if len(bob) != 1:
            raise NotReady("cannot identify the shop controller")
        hero = self.entities.get(player.number("HERO_ENTITY"))
        if hero is None or "HEALTH" not in hero.tags or "PLAYER_TECH_LEVEL" not in hero.tags:
            raise NotReady("local hero health/tier missing")
        board = self.slots(self.local_pid, "PLAY", ("MINION",))
        hand = self.slots(self.local_pid, "HAND", ("MINION", "SPELL"))
        # Bob also owns leaderboard heroes; only minion/spell PLAY slots are offers.
        shop = self.slots(bob[0], "PLAY", ("MINION", "SPELL"))
        if len(board) > 7 or len(hand) > 10 or len(shop) > 8:
            raise NotReady("visible slots exceed the trained action layout")
        if not self.options_ready and not (self.choices and self.choice_ready):
            raise NotReady("client legal options not settled")
        choices = []
        if self.choices:
            if not self.choice_ready or self.choice_pid != self.local_pid:
                raise NotReady("choice panel is not ready for the local player")
            keys = sorted(self.choices)
            if keys != list(range(keys[0], keys[0] + len(keys))):
                raise NotReady("choice indices are incomplete")
            for k in keys:
                if self.choices[k] not in self.entities:
                    raise NotReady("choice entity is not visible")
                choices.append(self.entities[self.choices[k]])
        return dict(game=game, player=player, hero=hero, board=board, hand=hand, shop=shop,
                    choices=choices, options=self.options, turn=(turn + 1) // 2,
                    revision=(self.generation, self.revision), build=self.build,
                    entities=self.entities, local_pid=self.local_pid,
                    powers=sorted([e for e in self.entities.values()
                                   if e.number("CONTROLLER") == self.local_pid
                                   and e.tags.get("ZONE") == "PLAY"
                                   and e.tags.get("CARDTYPE") == "HERO_POWER"],
                                  key=lambda e: e.id))

    def fingerprint(self) -> str:
        """Meaningful state, excluding log timestamps and duplicate tag updates."""
        data = dict(generation=self.generation, sent=self.sent,
                    entities={i: (e.card, e.tags) for i, e in self.entities.items()},
                    options=[(o.entity, o.kind, o.legal, o.targets, o.has_suboptions)
                             for o in self.options],
                    choices=self.choices, choice_ready=self.choice_ready)
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def latest_log(root: Path) -> Path:
    if root.is_file():
        return root
    sessions = sorted(p for p in root.glob("Hearthstone_*") if p.is_dir())
    if not sessions or not (sessions[-1] / "Power.log").is_file():
        raise NotReady("Power.log is missing from the latest Hearthstone session")
    return sessions[-1] / "Power.log"


class LogTail:
    """Byte cursor preserves partial UTF-8 lines; truncation resets parser state."""
    def __init__(self, root: Path, state: PowerState):
        self.root, self.state = root, state
        self.path = None
        self.identity = None
        self.offset = 0
        self.pending = b""

    def poll(self) -> int:
        path = latest_log(self.root)
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if path != self.path or identity != self.identity or stat.st_size < self.offset:
            self.state.reset()
            self.path, self.identity, self.offset, self.pending = path, identity, 0, b""
        with path.open("rb") as f:
            f.seek(self.offset)
            data = f.read(4 * 1024 * 1024)
            self.offset += len(data)
        chunks = (self.pending + data).split(b"\n")
        self.pending = chunks.pop()
        if len(self.pending) > 1_048_576:
            self.state.fault = "oversized unterminated log line"
            self.pending = b""
        for chunk in chunks:
            self.state.ingest(chunk.decode("utf-8", errors="replace"))
        return len(data)
