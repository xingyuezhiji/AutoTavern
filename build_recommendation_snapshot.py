"""Build a reproducible identity-only prior from explicitly supplied public exports."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def build(source, output):
    catalog_path = Path(__file__).parent / "crates/taverngym/data/cards.json"
    catalog = {c["id"]: c for c in json.loads(catalog_path.read_text())["cards"]}
    compositions = []
    for guide in json.loads((source / "composition-guides.json").read_text()):
        ids = set()
        for section in guide["sections"]:
            if "核心卡牌" not in section["heading"]:
                continue
            for card in section["cards"]:
                key = card.get("local_card_id")
                if (card.get("mapping_method") == "exact_dbf_id"
                        and card.get("local_reachable") and key in catalog
                        and catalog[key]["reachable"] and catalog[key]["dbf"] == card["dbf_id"]):
                    ids.add(catalog[key].get("normal") or key)
        if len(ids) >= 2:
            compositions.append({"title": guide["title"], "url": guide["url"],
                                 "core_ids": sorted(ids)})
    trinkets = {}
    for row in json.loads((source / "hsreplay-entity-data.json").read_text())["entities"]:
        key = row["card_id"]
        average = row.get("statistics", {}).get("average_placement")
        if (row["category"] != "lesser_trinket" or row["quality_warnings"]
                or row["local_mapping_method"] != "card_id_exact"
                or key not in catalog or not catalog[key]["reachable"]
                or catalog[key]["id"] != row["local_card"]["id"]
                or catalog[key]["dbf"] != row["local_card"]["dbf"]
                or catalog[key]["kind"] != "trinket_lesser"
                or average is None or not math.isfinite(average) or not 1 <= average <= 8):
            continue
        if key in trinkets:
            raise ValueError("duplicate eligible trinket requires explicit aggregation")
        trinkets[key] = {"id": key, "average_placement": average,
                        "source_url": row["source_url"],
                        "statistical_scope": row["statistical_scope"]}
    data = {"schema_version": 1, "snapshot_date": "2026-10-30",
            "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
            "sources": [{"file": name, "sha256": hashlib.sha256((source / name).read_bytes()).hexdigest()}
                        for name in ("COLLECTION.md", "composition-guides.json", "hsreplay-entity-data.json")],
            "rule_equivalence_verified": False, "interpretation": "weak observational prior",
            "compositions": compositions, "lesser_trinkets": list(trinkets.values())}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"compositions": len(compositions), "lesser_trinkets": len(trinkets)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.source, args.output)
