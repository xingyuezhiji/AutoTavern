"""Small, bounded public-guide priors; never reads hidden game state."""
import hashlib
import json
import math
from pathlib import Path


class Recommendation:
    def __init__(self, path, composition_weight=0.1, trinket_weight=0.05):
        if not (0 <= composition_weight <= 0.1 and 0 <= trinket_weight <= 0.05):
            raise ValueError("recommendation weights exceed validated bounds")
        raw = Path(path).read_bytes()
        self.data = json.loads(raw)
        catalog = Path(__file__).parent / "crates/taverngym/data/cards.json"
        if hashlib.sha256(catalog.read_bytes()).hexdigest() != self.data["catalog_sha256"]:
            raise ValueError("recommendation catalog fingerprint mismatch")
        cards = {c["id"]: c for c in json.loads(catalog.read_text())["cards"]}
        self.normal = {key: card.get("normal") or key for key, card in cards.items()}
        self.cores = []
        for guide in self.data["compositions"]:
            ids = frozenset(guide["core_ids"])
            if len(ids) < 2 or any(i not in cards or not cards[i]["reachable"] for i in ids):
                raise ValueError("invalid recommendation core identities")
            self.cores.append(ids)
        self.trinkets = {}
        for entry in self.data["lesser_trinkets"]:
            key, average = entry["id"], entry["average_placement"]
            if (key in self.trinkets or key not in cards or not cards[key]["reachable"]
                    or cards[key]["kind"] != "trinket_lesser"
                    or not math.isfinite(average) or not 1 <= average <= 8):
                raise ValueError("invalid recommendation trinket")
            self.trinkets[key] = average
        self.composition_weight = composition_weight
        self.trinket_weight = trinket_weight
        self.metadata = {"snapshot_sha256": hashlib.sha256(raw).hexdigest(),
                         "sources": self.data["sources"],
                         "snapshot_date": self.data["snapshot_date"],
                         "source_urls": sorted({g["url"] for g in self.data["compositions"]}
                                               | {t["source_url"] for t in self.data["lesser_trinkets"]}),
                         "compositions": len(self.cores), "lesser_trinkets": len(self.trinkets),
                         "composition_weight": composition_weight,
                         "trinket_weight": trinket_weight,
                         "net_bounds": [-trinket_weight, composition_weight + trinket_weight],
                         "rule_equivalence_verified": False}

    def episode(self):
        return EpisodeReward(self)

    def identity(self, item):
        card = item.get("card", item)
        return self.normal.get(card.get("id"), card.get("id"))


class EpisodeReward:
    def __init__(self, recommendation):
        self.source = recommendation
        self.potential = 0.0
        self.composition = self.trinket = 0.0
        self.turns = set()
        self.choices = set()
        self.end_events = self.choice_events = self.skipped_choices = 0

    def step(self, view, action, legal):
        """Call on the current own-seat view immediately before a legal action."""
        if not 0 <= action < len(legal) or not legal[action]:
            raise ValueError("recommendation requires a legal action")
        me, turn = view["me"], view["turn"]
        source = self.source
        if action == 0 and turn not in self.turns:
            ids = {source.identity(card) for card in me["board"]}
            phi = max((max(0, (len(ids & core) - 1) / (len(core) - 1))
                       for core in source.cores), default=0.0)
            delta = source.composition_weight * (phi - self.potential)
            self.potential = phi
            self.composition += delta
            self.turns.add(turn)
            self.end_events += 1
            return delta
        pending = me.get("pending") or {}
        if pending.get("kind") != "trinket" or not 30 <= action < 46:
            return 0.0
        options = pending.get("options") or []
        index = action - 30
        if index >= len(options) or options[index].get("kind") != "trinket_lesser":
            return 0.0
        key = (turn, tuple(source.identity(x) for x in options),
               tuple(source.identity(x) for x in me.get("trinkets", [])))
        if key in self.choices:
            return 0.0
        self.choices.add(key)
        chosen = source.identity(options[index])
        averages = [source.trinkets[source.identity(x)] for n, x in enumerate(options[:16])
                    if legal[30 + n] and source.identity(x) in source.trinkets]
        if chosen not in source.trinkets or len(averages) < 2:
            self.skipped_choices += 1
            return 0.0
        raw = source.trinket_weight * max(-1.0, min(
            1.0, sum(averages) / len(averages) - source.trinkets[chosen]))
        total = max(-source.trinket_weight, min(source.trinket_weight, self.trinket + raw))
        delta = total - self.trinket
        self.trinket = total
        self.choice_events += 1
        return delta

    def audit(self):
        return {"composition": self.composition, "trinket": self.trinket,
                "total": self.composition + self.trinket,
                "end_events": self.end_events, "choice_events": self.choice_events,
                "skipped_choices": self.skipped_choices}
