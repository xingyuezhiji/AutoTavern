"""v2 tensor layout with explicit uncertainty for the experimental log adapter.

The simulator replay path is exact. Power.log does not by itself populate all
engine script counters: unobserved blocks are marked and never called complete.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

from .power import Entity, NotReady

RACES = ("BEAST", "DEMON", "DRAGON", "ELEMENTAL", "MECH", "MURLOC",
         "PIRATE", "QUILBOAR", "UNDEAD", "ABERRATION", "NAGA")
KEYWORDS = ("TAUNT", "DIVINE_SHIELD", "WINDFURY", "REBORN", "VENOMOUS",
            "POISONOUS", "STEALTH", "MAGNETIC", "MEGA_WINDFURY", "CLEAVE",
            "IMMUNE", "IMMUNE_ATTACKING")
FLOAT_BLOCKS = (
    ("g2", 74), ("board_x", 63), ("board_att_n", 42), ("hand_att_n", 20),
    ("shop_att_n", 16), ("choice_att_n", 32), ("oppb_att_n", 147),
    ("effects_n", 8), ("counters_n", 6), ("levels_n", 4), ("sched_f", 12),
    ("quest_data", 3), ("last_combat", 2), ("opp_sum", 245), ("phase", 2),
    ("item_key", 82), ("cond", 32))
CARD_BLOCKS = (
    ("board_att", 42), ("hand_att", 20), ("shop_att", 16), ("choice_att", 32),
    ("oppb_att", 147), ("opp_trinkets", 28), ("opp_power", 7), ("effects", 8),
    ("counters", 6), ("levels", 4), ("sched", 4), ("last_spell", 1), ("goal_cards", 8))


class Encoder:
    def __init__(self, catalog: Path):
        raw = catalog.read_bytes()
        payload = json.loads(raw)
        # cards.rs enumerates this order; never sort or filter reachable cards.
        self.catalog = {c["id"]: (i + 1, c) for i, c in enumerate(payload["cards"])}
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.client_version = payload["meta"]["client_version"]
        contract = json.loads(Path(__file__).with_name("compatibility.json").read_text())
        if self.sha256 != contract["cards_file_sha256"]:
            raise ValueError("cards.json differs from the verified training vocabulary")
        self.training_sha256 = contract["training_catalog_sha256"]

    def validate_policy(self, policy):
        if self.training_sha256 != policy.catalog_sha256:
            raise ValueError("checkpoint and visible-state vocabulary differ")

    def card(self, e: Entity):
        if e.card not in self.catalog:
            raise NotReady(f"unknown visible card: {e.card or '<empty>'}")
        return self.catalog[e.card]

    def minion(self, e: Entity | None):
        if e is None:
            return [0.] * 31
        _, meta = self.card(e)
        if "ATK" not in e.tags or "HEALTH" not in e.tags:
            raise NotReady(f"live attack/health missing for entity {e.id}")
        hp = e.number("HEALTH")
        kw = [float(e.number(k)) for k in KEYWORDS]
        for i, k in enumerate(KEYWORDS):
            if k in ("MAGNETIC", "CLEAVE", "IMMUNE_ATTACKING") and k not in e.tags:
                kw[i] = float(k in meta["keywords"])
        if e.number("MODULAR") or e.number("MAGNETIC"):
            kw[7] = 1.
        return [1., e.number("ATK") / 10., (hp - e.number("DAMAGE")) / 10.,
                hp / 10., e.number("TECH_LEVEL", meta["tier"]) / 6.,
                float(bool(meta.get("normal"))), 0.,
                float(bool(e.number("DIVINE_SHIELD"))) / 3.,
                *kw, *[float(r in meta["races"] or "ALL" in meta["races"]) for r in RACES]]

    def item(self, e: Entity | None, cost=0, frozen=False):
        if e is None:
            return [0.] * 40
        _, meta = self.card(e)
        is_minion = e.tags.get("CARDTYPE") == "MINION" or meta["kind"] == "minion"
        if is_minion:
            tail = self.minion(e)
        else:
            tail = [1., 0., 0., 0., meta["tier"] / 6., *([0.] * 26)]
        return [1., float(is_minion), cost / 5., 0., 0., 0., float(frozen), 0., 0., *tail]

    def encode(self, view: dict, plans: dict) -> dict:
        from .actions import candidates
        player, hero, game = (view[k] for k in ("player", "hero", "game"))
        board, hand, shop = (view[k] for k in ("board", "hand", "shop"))
        powers = view["powers"]
        for e in [hero, *board, *hand, *shop, *powers, *view["choices"]]:
            self.card(e)
        gold = player.number("RESOURCES") - player.number("RESOURCES_USED") + player.number("TEMP_RESOURCES")
        if gold < 0:
            raise NotReady("negative visible gold indicates an unsettled state")
        options = candidates(view)
        buttons = [view["entities"].get(o.entity) for o in view["options"]]
        buttons = [e for e in buttons if e is not None]
        upgrade = next((e for e in buttons if e.card.startswith("TB_BaconShopTechUp")), None)
        refresh = next((e for e in buttons if e.card == "TB_BaconShop_8p_Reroll_Button"), None)
        upgrade_cost = upgrade.number("COST") if upgrade and "COST" in upgrade.tags else None
        refresh_cost = refresh.number("COST") if refresh and "COST" in refresh.tags else None
        # No spendable action may depend on a guessed price.
        if 3 in plans and upgrade_cost is None:
            plans.pop(3)
        if 1 in plans and refresh_cost is None:
            plans.pop(1)
        unknown = [
            "opponent last-seen warbands and combat summaries",
            "player script/effect counters and schedules",
            "attached effects, Dark Gift and trinket timing",
            "lobby races, quests, secrets and next opponent",
            "per-card prices/discounts not explicitly logged",
        ]
        if upgrade_cost is None:
            unknown.append("upgrade cost")
        if refresh_cost is None:
            unknown.append("refresh cost")
        # Minions use the standard 3 only as an annotated context estimate. Legal
        # buying comes from client options; it never uses this estimate for legality.
        public = [e for e in view["entities"].values()
                  if e.tags.get("CARDTYPE") == "HERO" and e.number("PLAYER_ID")
                  and e.id != hero.id and e.tags.get("PLAYER_LEADERBOARD_PLACE")]
        public.sort(key=lambda e: e.number("PLAYER_ID"))
        higher = [e for e in public if e.number("PLAYER_ID") > view["local_pid"]]
        lower = [e for e in public if e.number("PLAYER_ID") < view["local_pid"]]
        public = (higher + lower)[:7]
        alive = 1 + sum(e.number("HEALTH") > e.number("DAMAGE") for e in public)
        if len(public) != 7:
            unknown.append("complete public opponent roster")
        f = np.zeros(4017, dtype=np.float32)
        c = np.zeros(447, dtype=np.int64)
        f[:42] = [
            view["turn"] / 20., 0., 0., alive / 8., 1.,
            (hero.number("HEALTH") - hero.number("DAMAGE")) / 40.,
            hero.number("ARMOR") / 20., gold / 10., 0., 0.,
            hero.number("PLAYER_TECH_LEVEL") / 6., (upgrade_cost or 0) / 10.,
            (refresh_cost or 0) / 2., 1., (refresh.number("BACON_FREE_REFRESH_COUNT") if refresh else 0) / 3.,
            float(any(e.number("FROZEN") for e in shop)),
            float(4 in options), (powers[0].number("COST") if powers else 0) / 5., 0.,
            float(5 in options), (powers[1].number("COST") if len(powers) > 1 else 0) / 5., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            len(board) / 7., len(hand) / 10., 0., 0., 0., 0., 0., 0., 0.]
        kind = 2 if view["choices"] else (4 if any(30 <= a < 46 for a in plans) else 0)
        f[59 + kind] = 1.
        c[0] = self.card(hero)[0]
        for i, power in enumerate(powers[:2]):
            c[1 + i] = self.card(power)[0]
        cursor, card_cursor = 68, 19
        for xs, limit, area in ((board, 7, "board"), (hand, 10, "hand"),
                                (shop, 8, "shop"), (view["choices"], 16, "choice")):
            width = 31 if area == "board" else 40
            for i in range(limit):
                e = xs[i] if i < len(xs) else None
                if area == "board":
                    values = self.minion(e)
                else:
                    price = 0
                    if e and area == "shop":
                        _, meta = self.card(e)
                        price = e.number("BACON_OVERRIDE_BG_COST", 3 if meta["kind"] == "minion" else meta["cost"])
                    values = self.item(e, price, bool(e and area == "shop" and e.number("FROZEN")))
                f[cursor:cursor + width] = values
                c[card_cursor] = self.card(e)[0] if e else 0
                cursor += width
                card_cursor += 1
        assert cursor == 1645 and card_cursor == 60
        for i, e in enumerate(public):
            f[1645 + i * 9:1645 + (i + 1) * 9] = [
                1., float(e.number("HEALTH") > e.number("DAMAGE")),
                (e.number("HEALTH") - e.number("DAMAGE")) / 40.,
                e.number("ARMOR") / 20., e.number("PLAYER_TECH_LEVEL") / 6.,
                0., 0., 0., 0.]
            c[68 + i] = self.card(e)[0]
        # v2 extension blocks retain their exact offsets. Reserved fields are known
        # zero; unobserved context is zero-valued only with the provenance above.
        f[3227 + 15] = float(3 in options)  # g2.can_upgrade
        mask = np.zeros(245, dtype=bool)
        mask[list(plans)] = True
        if not mask.any():
            raise NotReady("no calibrated, supported client-legal action")
        return dict(features=f, cards=c, action_mask=mask,
                    coverage="estimated_log_context", unobserved=unknown,
                    catalog_sha256=self.sha256, gold=gold)


def exact_observation(game_json: str, seat: int):
    import taverngym
    g = taverngym.Game.from_json(game_json)
    f, c = g.observe2(seat)
    mask = g.legal_mask(seat)
    return dict(features=np.frombuffer(f, dtype="<f4").copy(),
                cards=np.frombuffer(c, dtype="<i4").astype(np.int64),
                action_mask=np.frombuffer(mask, dtype=np.uint8).astype(bool),
                coverage="exact_simulator", unobserved=[])
