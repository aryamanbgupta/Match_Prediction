"""Stage the 2025-season Major League Cricket (USA) matches into a dedicated
eval pool: `data/mlc_2025/`.

Unlike `extract_blast_golden.py`, MLC is NOT new data — all 75 MLC matches
(2023/2024/2025) are already in `data/t20s_json/` and the production SQLite
cache. This script just *isolates* the 33 2025-season MLC matches into a
test-dir the sim/prop pipeline can consume, and emits a manifest recording
each match's train/val/test split membership.

Why split membership matters: the production match-level model trains on
train+val (date <= val_end 2025-06-30). So:
  * MLC June 2025 matches (<= 2025-06-30) land in the *validation* split —
    in-sample (seen during training/early-stopping). NOT a clean OOS test.
  * MLC July 2025 matches (> 2025-06-30, <= test_end 2026-04-16) land in the
    *test* split — held out from fitting; the clean accuracy subset.

This is staging-only: it does NOT touch data/t20s_json/, the production
SQLite cache, or any model artifact.

Usage:
    uv run python scripts/extract_mlc.py
    uv run python scripts/extract_mlc.py --dry-run
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MLC_ZIP = REPO_ROOT / "data" / ".cricsheet_zips" / "mlc_json.zip"
OUT_POOL = REPO_ROOT / "mlc" / "data" / "mlc_2025"
MANIFEST = OUT_POOL / "_manifest.json"

SEASON = "2025"
# Mirror materialize_features.DEFAULT_SPLITS.
TRAIN_END = "2024-12-31"
VAL_END = "2025-06-30"
TEST_END = "2026-04-16"


def classify(date: str) -> str:
    if date <= TRAIN_END:
        return "train"
    if date <= VAL_END:
        return "validation"
    if date <= TEST_END:
        return "test"
    return "golden_test"


def winner(info: dict) -> str | None:
    outcome = info.get("outcome") or {}
    return outcome.get("winner")  # None for no-result/tie


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not MLC_ZIP.exists():
        print(f"ERROR: {MLC_ZIP} not found")
        return 1

    OUT_POOL.mkdir(parents=True, exist_ok=True)
    rows = []
    copied = 0

    with zipfile.ZipFile(MLC_ZIP) as zf:
        for name in (n for n in zf.namelist() if n.endswith(".json")):
            try:
                with zf.open(name) as f:
                    data = json.load(io.TextIOWrapper(f))
            except Exception:
                continue
            info = data.get("info") or {}
            if str(info.get("season")) != SEASON:
                continue
            if info.get("match_type") != "T20" or info.get("gender") != "male":
                continue
            dates = info.get("dates") or []
            if not dates:
                continue
            date = dates[0]
            mid = name.split("/")[-1].replace(".json", "")
            teams = info.get("teams", ["?", "?"])
            rows.append({
                "match_id": mid,
                "date": date,
                "team1": teams[0],
                "team2": teams[1],
                "winner": winner(info),
                "split": classify(date),
            })
            if not args.dry_run:
                out = OUT_POOL / f"{mid}.json"
                tmp = out.with_suffix(".json.tmp")
                with open(tmp, "w") as g:
                    json.dump(data, g, separators=(",", ":"))
                tmp.replace(out)
            copied += 1

    rows.sort(key=lambda r: (r["date"], r["match_id"]))
    by_split: dict[str, int] = {}
    for r in rows:
        by_split[r["split"]] = by_split.get(r["split"], 0) + 1

    print(f"MLC {SEASON}: {copied} matches")
    print(f"by split: {by_split}")
    print(f"{'date':12} {'split':12} {'match':40} winner")
    for r in rows:
        m = f"{r['team1']} v {r['team2']}"
        print(f"  {r['date']:10} {r['split']:12} {m:40} {r['winner']}")

    if not args.dry_run:
        with open(MANIFEST, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nWrote {copied} files + manifest → {OUT_POOL}")
    else:
        print(f"\nDry run; would write {copied} files → {OUT_POOL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
