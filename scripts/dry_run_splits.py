"""Dry-run of the new parsing split logic.

Walks data/t20s_json/*.json and applies the same date-split + gender filter that
parsing_v2.py will apply on the next destructive rebuild, but writes nothing.
Reports per-split counts, women's matches filtered, betting_eval subset size,
top test-split events, and a rough date histogram.

Run:  uv run python scripts/dry_run_splits.py
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
JSON_DIR = REPO_ROOT / "data" / "t20s_json"
ODDS_FILE = REPO_ROOT / "betting_odds_polymarket.json"

TRAIN_END = datetime(2024, 12, 31)
TEST_START = datetime(2025, 7, 1)
GOLDEN_START = datetime(2026, 4, 17)

ACCEPT_TRAIN_MIN = 9000
ACCEPT_BETTING_MIN = 400
ACCEPT_BETTING_BAIL = 300


def classify(match_date: datetime) -> str:
    if match_date < TRAIN_END:
        return "train"
    if match_date < TEST_START:
        return "validation"
    if match_date < GOLDEN_START:
        return "test"
    return "golden_test"


def main() -> int:
    json_paths = sorted(JSON_DIR.glob("*.json"))
    print(f"Scanning {len(json_paths)} files from {JSON_DIR}")

    split_counts: Counter[str] = Counter()
    split_date_hist: dict[str, Counter[str]] = defaultdict(Counter)
    test_events: Counter[str] = Counter()
    test_match_ids: set[str] = set()
    skipped_non_male = 0
    skipped_bad = 0

    for path in json_paths:
        try:
            with path.open() as f:
                data = json.load(f)
            info = data["info"]
        except (json.JSONDecodeError, KeyError, OSError):
            skipped_bad += 1
            continue

        gender = info.get("gender", "male")
        if gender != "male":
            skipped_non_male += 1
            continue

        try:
            match_date = datetime.strptime(info["dates"][0], "%Y-%m-%d")
        except (KeyError, ValueError, IndexError):
            skipped_bad += 1
            continue

        split = classify(match_date)
        split_counts[split] += 1
        split_date_hist[split][match_date.strftime("%Y-%m")] += 1

        if split == "test":
            event_info = info.get("event", {})
            event_name = (
                event_info.get("name", "") if isinstance(event_info, dict) else ""
            )
            test_events[event_name or "(no event)"] += 1

            teams = info.get("teams", [])
            venue = info.get("venue", "")
            if len(teams) == 2:
                match_id = (
                    f"{info['dates'][0]}_{teams[0]}_{teams[1]}_{venue}".replace(" ", "_")
                )
                test_match_ids.add(match_id)

    print("\n=== Split counts ===")
    for split in ("train", "validation", "test", "golden_test"):
        print(f"  {split:12s} {split_counts[split]:>6d}")
    print(f"  women's     {skipped_non_male:>6d} (filtered)")
    if skipped_bad:
        print(f"  unparseable {skipped_bad:>6d}")

    betting_eval_count = 0
    betting_ids_total = 0
    betting_missing: list[str] = []
    if ODDS_FILE.exists():
        with ODDS_FILE.open() as f:
            odds = json.load(f)
        odds_matches = odds.get("matches", [])
        betting_ids_total = len(odds_matches)
        odds_ids = {m["match_id"] for m in odds_matches}
        intersect = odds_ids & test_match_ids
        betting_eval_count = len(intersect)
        betting_missing = sorted(odds_ids - test_match_ids)
    else:
        print(f"\n(missing {ODDS_FILE.name} — run build_polymarket_odds.py first)")

    print("\n=== Betting eval subset ===")
    print(f"  odds file entries       : {betting_ids_total}")
    print(f"  intersection with test  : {betting_eval_count}")
    if betting_missing:
        print(
            f"  odds match_ids not found in test split: {len(betting_missing)}"
        )
        for mid in betting_missing[:10]:
            print(f"    {mid}")

    print("\n=== Top 10 events in test split ===")
    for event, count in test_events.most_common(10):
        print(f"  {count:>4d}  {event}")

    print("\n=== Test-split date histogram ===")
    for ym in sorted(split_date_hist["test"]):
        print(f"  {ym}  {split_date_hist['test'][ym]}")

    print("\n=== Acceptance thresholds ===")
    ok = True
    train_n = split_counts["train"]
    print(
        f"  train ≥ {ACCEPT_TRAIN_MIN:>5d}   : "
        f"{train_n:>5d}  {'OK' if train_n >= ACCEPT_TRAIN_MIN else 'FAIL'}"
    )
    if train_n < ACCEPT_TRAIN_MIN:
        ok = False

    print(
        f"  betting_eval ≥ {ACCEPT_BETTING_MIN:>4d}: "
        f"{betting_eval_count:>5d}  "
        f"{'OK' if betting_eval_count >= ACCEPT_BETTING_MIN else 'WARN'}"
    )
    if betting_eval_count < ACCEPT_BETTING_BAIL:
        print(
            f"  !!  below hard bail threshold ({ACCEPT_BETTING_BAIL}) — "
            f"team mapping may be broken"
        )
        ok = False

    print(
        f"  women's ≈ 0          : {skipped_non_male:>5d}  "
        f"{'OK' if skipped_non_male < 50 else 'REVIEW'}"
    )

    # golden_test is expected to be empty (golden_start = 2026-04-17, corpus
    # cuts off at 2026-04-16 — it's a hook for future golden-set data).
    for split in ("train", "validation", "test"):
        if split_counts[split] == 0:
            print(f"  empty split detected: {split}  FAIL")
            ok = False
    if split_counts["golden_test"] == 0:
        print("  golden_test empty     : expected (future-hook, not failure)")

    print("\n" + ("ALL CHECKS PASSED" if ok else "CHECKS FAILED — DO NOT REBUILD"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
