"""User-supplied hero statistics; never download or invent performance figures."""
from __future__ import annotations

import json
import math
from pathlib import Path


class HeroStats:
    def __init__(self, path: Path | None = None):
        self.rows = []
        if path:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not isinstance(payload.get("heroes"), list):
                raise ValueError("hero statistics require a {heroes: [...]} file")
            self.rows = payload["heroes"]
            for row in self.rows:
                score = row.get("avg_placement")
                if (isinstance(score, bool) or not isinstance(score, (float, int))
                        or not math.isfinite(score) or not 1 <= score <= 8):
                    raise ValueError("hero avg_placement must be finite and in [1,8]")
                if not row.get("card_id") and not row.get("dbf_id"):
                    raise ValueError("hero statistics require card_id or dbf_id")

    def choose(self, offers):
        unlocked = [(i, e) for i, e in enumerate(offers)
                    if not e.number("BACON_LOCKED_MULLIGAN_HERO")]
        if not unlocked:
            raise ValueError("no unlocked heroes")
        ranked = []
        for i, entity in unlocked:
            dbfs = {str(entity.number(t)) for t in ("DBF_ID", "BACON_SKIN_PARENT_ID")
                    if entity.number(t) > 0}
            hits = [r["avg_placement"] for r in self.rows
                    if r.get("card_id") == entity.card or str(r.get("dbf_id")) in dbfs]
            if hits:
                ranked.append((min(hits), i, entity))
        if ranked:
            score, index, entity = min(ranked, key=lambda x: (x[0], x[1]))
            return index, entity, dict(basis="local_avg_placement", avg_placement=score)
        index, entity = max(unlocked, key=lambda x: (x[1].number("ARMOR"), -x[0]))
        return index, entity, dict(basis="visible_armor_then_physical_slot",
                                   armor=entity.number("ARMOR"))
