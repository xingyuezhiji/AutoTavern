"""Training action IDs -> stable entities -> calibrated UI primitives."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np

from .power import NotReady


def decode(action: int) -> tuple:
    if isinstance(action, bool) or not isinstance(action, (int, np.integer)) or not 0 <= action < 245:
        raise ValueError("action must be an integer in [0,245)")
    if action < 8:
        return (("end", "refresh", "freeze", "upgrade", "power0", "power1",
                 "dark_gift", "cancel")[action],)
    for start, stop, name in ((8, 16, "buy"), (16, 23, "sell"), (23, 30, "activate"),
                             (30, 46, "choose")):
        if start <= action < stop:
            return name, action - start
    if action < 126:
        return "play", (action - 46) // 8, (action - 46) % 8
    if action < 196:
        return "magnetize", (action - 126) // 7, (action - 126) % 7
    return "move", (action - 196) // 7, (action - 196) % 7


def candidates(view: dict) -> dict[int, dict]:
    """Only actions supported by current client legal options, with target identity."""
    result: dict[int, dict] = {}
    board, hand, shop = (view[k] for k in ("board", "hand", "shop"))
    options = [o for o in view["options"] if o.legal and not o.has_suboptions]
    # Suboptions (Choose One) cannot silently become ordinary cards.
    legal_ids = {o.entity for o in options}
    targets = {o.entity: o.targets for o in options}
    def add(a: int, eid: int = 0, **extra):
        result[a] = dict(entity=eid, **extra)
    if view["choices"]:
        if len(view["choices"]) > 16:
            raise NotReady("choice count exceeds model layout")
        for i, e in enumerate(view["choices"]):
            add(30 + i, e.id, region="choice", index=i)
        return result
    # Explicit TARGET options are already modal. They use Choose IDs in model space.
    modal = [o for o in options if o.kind in ("TARGET", "CHOOSE")]
    if modal:
        for i, o in enumerate(modal[:16]):
            add(30 + i, o.entity, region="target", index=i)
        return result
    if any(o.kind == "END_TURN" for o in options):
        add(0)
    for o in options:
        e = view["entities"].get(o.entity)
        if e is None:
            continue
        if e.card == "TB_BaconShop_8p_Reroll_Button":
            add(1, e.id)
        elif e.card == "TB_BaconShopLockAll_Button":
            add(2, e.id)
        elif e.card.startswith("TB_BaconShopTechUp"):
            add(3, e.id)
        elif e.card in ("TB_BaconShop_DragBuy", "TB_BaconShop_DragBuy_Spell"):
            for i, item in enumerate(shop):
                if item.id in o.targets:
                    add(8 + i, item.id)
        elif e.card == "TB_BaconShop_DragSell":
            for i, item in enumerate(board):
                if item.id in o.targets:
                    add(16 + i, item.id)
        elif e.tags.get("CARDTYPE") == "HERO_POWER" and not o.targets:
            powers = view.get("powers", [])
            if e in powers and powers.index(e) < 2:
                add(4 + powers.index(e), e.id)
        # Targeted powers and cards require a verified modal transition; currently
        # omitted instead of pretending a drag completed a pending target.
    for i, e in enumerate(shop):
        if e.id in legal_ids and not targets.get(e.id) and len(hand) < 10:
            add(8 + i, e.id)
    for i, e in enumerate(hand):
        if e.id not in legal_ids or targets.get(e.id):
            continue
        if e.tags.get("CARDTYPE") == "MINION" and len(board) < 7:
            for pos in range(len(board) + 1):
                add(46 + i * 8 + pos, e.id)
        elif e.tags.get("CARDTYPE") == "SPELL":
            add(46 + i * 8, e.id)
    for i, e in enumerate(board):
        moves = [o for o in options if o.entity == e.id and o.kind == "MOVE_MINION"]
        if moves:
            for j in range(len(board)):
                if i != j:
                    add(196 + i * 7 + j, e.id, target_entity=board[j].id)
    return result


@dataclass
class Plan:
    action: int
    kind: str
    revision: tuple[int, int]
    fingerprint: str
    entity: int
    operations: list[dict]

    def json(self):
        return asdict(self)


def validate_revision(plan: Plan, state):
    if (plan.revision != (state.generation, state.revision)
            or plan.fingerprint != state.fingerprint()):
        raise NotReady("state changed after inference; action discarded")


class Geometry:
    """All coordinates are fractions of the client rectangle, not screen pixels."""
    def __init__(self, config: dict):
        self.config = config
        for name, point in config.get("points", {}).items():
            self.check(point)
        for row in config.get("rows", {}).values():
            for points in row.values():
                for point in points:
                    self.check(point)

    @staticmethod
    def check(point):
        if len(point) != 2 or not all(np.isfinite(v) and 0 < v < 1 for v in point):
            raise ValueError("coordinates must lie strictly inside the client rectangle")
        return list(point)

    def point(self, name: str):
        try:
            return self.check(self.config["points"][name])
        except KeyError:
            raise NotReady(f"point is not calibrated: {name}") from None

    def row(self, name: str, count: int, index: int):
        try:
            positions = self.config["rows"][name][str(count)]
        except KeyError:
            raise NotReady(f"row needs calibration: {name}/{count}") from None
        if len(positions) != count or not 0 <= index < count:
            raise NotReady(f"invalid calibrated slot: {name}/{count}/{index}")
        return self.check(positions[index])

    def insertion(self, count: int, index: int):
        # Calibrate board gaps independently: card centers are not insertion gaps.
        return self.row("board_gaps", count + 1, index)


def plan(action: int, view: dict, geometry: Geometry, fingerprint: str,
         candidate: dict) -> Plan:
    spec = decode(action)
    kind = spec[0]
    board, hand, shop = (view[k] for k in ("board", "hand", "shop"))
    ops = []
    def click(p, button="left"):
        ops.append(dict(type="click", point=p, button=button))
    def drag(a, b):
        ops.append(dict(type="drag", start=a, end=b))
    def row(name, xs, i):
        if not 0 <= i < len(xs):
            raise NotReady("action refers to an empty slot")
        return geometry.row(name, len(xs), i)
    if kind in ("end", "refresh", "freeze", "upgrade", "power0", "power1", "dark_gift"):
        click(geometry.point(kind))
    elif kind == "cancel":
        click(geometry.point("neutral"), "right")
    elif kind == "buy":
        drag(row("shop", shop, spec[1]), geometry.point("hand_drop"))
    elif kind == "sell":
        drag(row("board", board, spec[1]), geometry.point("sell_drop"))
    elif kind == "activate":
        click(row("board", board, spec[1]), "right")
    elif kind == "choose":
        eid = candidate["entity"]
        if candidate.get("region") == "target":
            found = False
            for name, xs in (("board", board), ("shop", shop), ("hand", hand)):
                for i, e in enumerate(xs):
                    if e.id == eid:
                        click(row(name, xs, i))
                        found = True
            if not found:
                raise NotReady("target entity is not visible in a supported region")
        else:
            click(geometry.row("choice", len(view["choices"]), spec[1]))
    elif kind == "play":
        h, pos = spec[1:]
        if not 0 <= h < len(hand):
            raise NotReady("hand slot is empty")
        if hand[h].tags.get("CARDTYPE") == "SPELL":
            if pos != 0:
                raise NotReady("spell position must be zero")
            dest = geometry.point("spell_drop")
        else:
            if len(board) >= 7 or pos > len(board):
                raise NotReady("board insertion is unavailable")
            dest = geometry.insertion(len(board), pos)
        drag(row("hand", hand, h), dest)
    elif kind == "magnetize":
        drag(row("hand", hand, spec[1]), row("board", board, spec[2]))
    elif kind == "move":
        src, dst = spec[1:]
        if src == dst or src >= len(board) or dst >= len(board):
            raise NotReady("invalid move")
        # Remove the source first; final destination is an insertion in n-1 cards.
        drag(row("board", board, src), geometry.insertion(len(board) - 1, dst))
    return Plan(int(action), kind, view["revision"], fingerprint,
                candidate.get("entity", 0), ops)


def executable(view: dict, geometry: Geometry, fingerprint: str):
    plans = {}
    for a, candidate in candidates(view).items():
        try:
            plans[a] = plan(a, view, geometry, fingerprint, candidate)
        except NotReady:
            pass
    return plans
