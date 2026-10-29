"""Guide-informed demonstrator using only the learner's public player view."""
from collections import Counter
import json
import re

from action_limits import limit_action_mask


# Normal card IDs also match their golden variants.
CORES = {
    "BG_LOE_077": 23, "BG35_883": 27, "BG36_640": 30,
    "BG32_111": 17, "BG35_881": 17, "BG28_707": 22,
    "BG28_633": 23, "BG36_762": 18, "BG36_701": 16,
    "BG32_873": 14,
}
# Match actual resources/discounts, not "Golden", spending Gold for stats,
# or buffs triggered by paid Refreshes.
ECONOMY = re.compile(
    r"\b(?:gain|get|earn)\b[^.!]{0,55}\bgold\b"
    r"|\b(?:increase|maximum)\b[^.!]{0,35}\b(?:maximum\s+)?gold\b"
    r"|\bget\b[^.!]{0,30}\bcoins?\b"
    r"|\bfree\s+refresh(?:es)?\b"
    r"|\brefresh(?:es)?\b[^.!]{0,35}\bfree\b"
    r"|\b(?:buy|cost)[^.!]{0,35}\bfree\b"
    r"|\bfree\b[^.!]{0,20}\b(?:minion|card|spell)s?\b"
    r"|\bcosts?\s+health\s+instead\s+of\s+gold\b"
    r"|\breduce\b[^.!]{0,35}\bcost\b"
    r"|\bcosts?\s*\([012]\)"
    r"|\bcost[^.!]{0,25}\bless\b|\bdiscount\b",
    re.I,
)
LINEUP_UTILITY = re.compile(
    r"\bgolden(?:izer)?\b|\brefresh(?:ed|es)?\b"
    r"|\bspend\b[^.!]{0,20}\bgold\b"
    r"|\btrinkets?\b[^.!]{0,40}\bfree\b",
    re.I,
)


def card_of(item):
    card = item.get("card", item)
    return card if isinstance(card, dict) else {}


def normal_id(card):
    return card.get("normal_id") or card.get("id", "").replace("_G", "")


def score_card(item, owned, *, offer=False):
    card = card_of(item)
    cid = normal_id(card)
    text = card.get("text", "")
    name = card.get("name", "")
    score = CORES.get(cid, 0)
    if owned[cid] >= 1 and cid in {"BG_LOE_077", "BG35_883"}:
        score -= 15  # Duplicate ordinary multipliers do not multiply each other.
    if owned[cid] == 2:
        score += 12
    if card.get("kind") == "spell":
        score += 5 + (8 if owned["BG36_640"] or owned["BG28_633"] else 0)
    if "Battlecry" in text and owned["BG_LOE_077"]:
        score += 7
    if cid == "BG35_881" and owned["BG28_707"]:
        score += 10
    if cid == "BG28_633" and (owned["BG28_707"] or owned["BG36_762"]):
        score += 12
    if cid == "BG36_640" and owned["BG35_883"]:
        score += 12
    # Economy preference applies especially to trinket and Dark Gift offers.
    economy = ECONOMY.search(text + " " + name)
    if economy:
        score += 22 if offer else 4
    # Powerful lineup effects remain useful without being mislabeled as money.
    # The guide prefers economy when suitable; it still needs scaling/core cards.
    if not economy and LINEUP_UTILITY.search(text):
        score += 20 if offer else 4
    if offer and card.get("kind", "").startswith("trinket"):
        if "Brann Bronzebeard" in text:
            score += 18 + 8 * bool(owned["BG36_640"] or owned["BG28_633"])
        if "Titus Rivendare" in text:
            score += 14
    atk = item.get("atk", card.get("atk", 0))
    hp = item.get("hp", card.get("hp", 0))
    score += min(25, (max(0, atk) + max(0, hp)) ** 0.5)
    if "Divine Shield" in text:
        score += 4
    return score


def act(game, seat, np):
    view = json.loads(game.player_view(seat))
    me = view["me"]
    features = np.frombuffer(game.observe2(seat)[0], dtype="<f4")
    mask = limit_action_mask(features, np.frombuffer(game.legal_mask(seat),
                                                    dtype=np.uint8).astype(bool))
    legal = np.flatnonzero(mask)
    if not len(legal):
        raise RuntimeError("guide teacher has no legal action")
    board, hand, shop = me["board"], me["hand"], me["shop"]
    owned = Counter(normal_id(card_of(x)) for x in board + hand)
    pending = me.get("pending") or {}
    kind = pending.get("kind")
    if any(30 <= a < 46 for a in legal):
        options = pending.get("options") or []
        targets = {t["action"]: t["location"] for t in pending.get("targets") or []}
        def option_score(a):
            i = int(a) - 30
            if kind == "target":
                location = targets.get(int(a), {})
                zone = {"board": board, "shop": shop, "hand": hand}.get(location.get("zone"), [])
                slot = location.get("slot", -1)
                item = zone[slot] if 0 <= slot < len(zone) else {}
            else:
                item = options[i] if i < len(options) else {}
            if not isinstance(item, dict):
                return 0
            score = score_card(item, owned, offer=kind in {"trinket", "dark_gift", "discover"})
            if kind == "target":
                card = card_of(item)
                score = score_card(item, owned)
                if normal_id(card) == "BG36_640":
                    score += 60
                # Targeted buffs should use friendly eligible minions; the mask
                # remains authoritative for spells with different target rules.
                if targets.get(int(a), {}).get("zone") == "board":
                    score += 10
            if kind == "hero":
                power = card_of(item).get("hero_power") or {}
                score += 15 * bool(ECONOMY.search(power.get("text", "")))
                score += card_of(item).get("armor", 0) / 2
            return score
        return int(max((a for a in legal if 30 <= a < 46), key=option_score))

    turn, tier, gold = view["turn"], me["tier"], me["gold"]
    health = me["health"] + me["armor"]
    # Resolve playable cards before spending the remaining gold.
    playable = [int(a) for a in legal if 46 <= a < 126]
    if playable:
        return max(playable, key=lambda a: score_card(hand[(a - 46) // 8], owned))
    target_tier = 1 if turn == 1 else 2 if turn < 5 else 3 if turn < 6 else 4
    if turn >= 9 and health >= 22 and len(board) >= 5:
        target_tier = 5
    if turn >= 11 and health >= 25 and len(board) >= 6:
        target_tier = 6
    if mask[3] and tier < target_tier:
        return 3
    if mask[6] and 3 <= turn <= 5:
        return 6
    buys = [int(a) for a in legal if 8 <= a < 16]
    if buys:
        best = max(buys, key=lambda a: score_card(shop[a - 8], owned))
        value = score_card(shop[best - 8], owned)
        if len(board) < 7 or value >= 12:
            if len(hand) < 8:
                return best
    # Free a slot only for a useful card already in hand.
    if len(board) >= 7 and hand:
        candidates = [int(a) for a in legal if 16 <= a < 23]
        if candidates:
            weakest = min(candidates, key=lambda a: score_card(board[a - 16], owned))
            weakest_value = score_card(board[weakest - 16], owned)
            if max(score_card(x, owned) for x in hand) > weakest_value + 2:
                return weakest
    activations = [int(a) for a in legal if 23 <= a < 30]
    if activations and gold >= 1 and (owned["BG32_111"] or owned["BG35_881"]):
        for a in activations:
            if normal_id(card_of(board[a - 23])) == "BG36_701":
                return a
    if mask[1] and (gold >= 3 or me.get("free_refreshes", 0) > 0):
        return 1
    # The simulator's bot handles hero powers and unusual pending mechanics.
    fallback = int(game.bot_action(seat))
    return fallback if mask[fallback] else int(0 if mask[0] else legal[0])


def generate_dataset(np, taverngym, *, samples, seed, max_games):
    """Keep successful complete learner episodes, with independent game IDs."""
    from train import decode_obs
    rows, ids, places = [], [], []
    for i in range(max_games):
        game = taverngym.Game(seed + i)
        seat = i % game.num_players
        episode = []
        guard = 0
        while not game.is_over():
            progressed = False
            for j in range(game.num_players):
                if not game.needs_action(j):
                    continue
                action = act(game, j, np) if j == seat else int(game.bot_action(j))
                if j == seat:
                    f, c, m = decode_obs(np, game.observe2(j), game.legal_mask(j))
                    if not m[action]:
                        raise RuntimeError("guide demonstrator selected illegal action")
                    episode.append((f, c, m, action))
                game.step(j, action)
                progressed = True
            guard += 1
            if not progressed or guard > 50000:
                raise RuntimeError("guide episode stopped progressing")
        place = int(game.placements()[seat])
        places.append(place)
        if place <= 4:
            rows.extend(episode)
            ids.extend([i] * len(episode))
        if (i + 1) % 10 == 0 or len(rows) >= samples:
            print(f"guide games={i+1} samples={len(rows)}/{samples} "
                  f"top4={np.mean(np.asarray(places)<=4):.3f} "
                  f"avg_place={np.mean(places):.3f}", flush=True)
        if len(rows) >= samples:
            # Keep the final complete episode; samples is a minimum budget.
            return (np.stack([r[0] for r in rows]).astype("float32"),
                    np.stack([r[1] for r in rows]).astype("int64"),
                    np.stack([r[2] for r in rows]).astype(bool),
                    np.asarray([r[3] for r in rows], dtype="int64"),
                    np.asarray(ids, dtype="int64"))
    raise RuntimeError("not enough successful guide samples; increase --max-games")
