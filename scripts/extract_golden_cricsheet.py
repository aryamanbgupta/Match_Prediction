"""Extract new T20 cricsheet JSONs from stat-generator zips into the golden
test directory. Strictly date >= 2026-04-17, match_type=T20, gender=male.

This is staging-only: it does NOT touch data/t20s_json/, the SQLite cache,
or any model artifact. The output dir data/golden/t20s_json/ is consumed
only by build_polymarket_odds_golden.py.

Usage:
    uv run python scripts/extract_golden_cricsheet.py
    uv run python scripts/extract_golden_cricsheet.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ZIP_DIR = Path("/Users/aryamangupta/Projects/stat-generator/data/cricsheet")
LOCAL_POOL = REPO_ROOT / "data" / "t20s_json"
GOLDEN_POOL = REPO_ROOT / "data" / "golden" / "t20s_json"

# User-confirmed scope: strictly post-test-set window, T20-format leagues only.
# 2026-07-30 expansion (user-approved golden refresh): MLC + LPL added for
# May-July coverage. ntb (Blast) stays OUT — it has its own golden pool
# under data/golden_blast/ and must not be double-pooled here.
CUTOFF_DATE = "2026-04-17"
ZIPS = ["t20s_json.zip", "ipl_json.zip", "psl_json.zip", "sat_json.zip",
        "mlc_json.zip", "lpl_json.zip"]

# Pools whose matches must NEVER enter golden: the 137 consumed forward
# EVALUATED fixtures (zero-overlap invariant; verify_forward_holdout checks
# it) and the separately-managed Blast golden pool. The forward CONTEXT pool
# is shared state, not consumed bets — golden already legitimately overlaps
# it (155 files as of 2026-07-30) and may keep absorbing context-era
# fixtures that were never part of the sealed evaluation.
EXCLUDED_POOL_DIRS = [
    REPO_ROOT / "data" / "forward_holdout" / "2026-06-01_2026-07-13"
    / "polymarket_test",
    REPO_ROOT / "data" / "golden_blast" / "t20s_json",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be copied; don't write.")
    args = ap.parse_args()

    if not ZIP_DIR.exists():
        print(f"ERROR: source zip dir not found: {ZIP_DIR}")
        return 1

    GOLDEN_POOL.mkdir(parents=True, exist_ok=True)
    local_existing = {p.name for p in LOCAL_POOL.glob("*.json")}
    golden_existing = {p.name for p in GOLDEN_POOL.glob("*.json")}
    excluded_names: set[str] = set()
    for pool in EXCLUDED_POOL_DIRS:
        if pool.is_dir():
            excluded_names |= {p.name for p in pool.glob("*.json")}

    print(f"Local pool size:  {len(local_existing):,}")
    print(f"Golden pool size: {len(golden_existing):,}")
    print(f"Excluded (forward-holdout/blast pools): {len(excluded_names):,}")
    print(f"Cutoff:           date >= {CUTOFF_DATE}, match_type=T20, gender=male")
    print()

    total_copied = 0
    total_skipped_already = 0
    total_skipped_in_local = 0
    total_skipped_format = 0
    total_skipped_gender = 0
    total_skipped_date = 0
    total_skipped_error = 0

    per_zip_stats: dict[str, dict[str, int]] = {}

    for zname in ZIPS:
        zpath = ZIP_DIR / zname
        if not zpath.exists():
            print(f"  WARN: missing {zpath}; skipping")
            continue

        copied = skipped_local = skipped_already = skipped_format = 0
        skipped_gender = skipped_date = skipped_error = 0
        skipped_excluded = 0

        with zipfile.ZipFile(zpath) as zf:
            json_names = [n for n in zf.namelist() if n.endswith(".json")]
            for name in json_names:
                # Never absorb matches owned by the consumed forward holdout
                # or the separate Blast golden pool.
                if name in excluded_names:
                    skipped_excluded += 1
                    continue
                # Skip if file already in our local production pool.
                if name in local_existing:
                    skipped_local += 1
                    continue
                # Skip if already in golden pool (idempotent re-run).
                if name in golden_existing:
                    skipped_already += 1
                    continue
                try:
                    with zf.open(name) as f:
                        data = json.load(f)
                except Exception:
                    skipped_error += 1
                    continue

                info = data.get("info") or {}
                if info.get("match_type") != "T20":
                    skipped_format += 1
                    continue
                if info.get("gender") != "male":
                    skipped_gender += 1
                    continue
                dates = info.get("dates") or []
                if not dates or dates[0] < CUTOFF_DATE:
                    skipped_date += 1
                    continue

                if args.dry_run:
                    copied += 1
                    continue

                # Extract via temp path to avoid partial files on crash.
                out_path = GOLDEN_POOL / name
                tmp = out_path.with_suffix(".json.tmp")
                with open(tmp, "w") as f:
                    json.dump(data, f, separators=(",", ":"))
                tmp.replace(out_path)
                copied += 1

        per_zip_stats[zname] = {
            "copied": copied,
            "skipped_excluded_pool": skipped_excluded,
            "skipped_in_local_pool": skipped_local,
            "skipped_already_in_golden": skipped_already,
            "skipped_non_t20": skipped_format,
            "skipped_non_male": skipped_gender,
            "skipped_pre_cutoff": skipped_date,
            "skipped_error": skipped_error,
        }
        total_copied += copied
        total_skipped_in_local += skipped_local
        total_skipped_already += skipped_already
        total_skipped_format += skipped_format
        total_skipped_gender += skipped_gender
        total_skipped_date += skipped_date
        total_skipped_error += skipped_error

    print(f"{'zip':<18} {'copied':>7} {'in_local':>9} {'already':>8} "
          f"{'non_t20':>8} {'non_male':>9} {'pre_cut':>8} {'error':>6}")
    for z, s in per_zip_stats.items():
        print(f"{z:<18} {s['copied']:>7} {s['skipped_in_local_pool']:>9} "
              f"{s['skipped_already_in_golden']:>8} {s['skipped_non_t20']:>8} "
              f"{s['skipped_non_male']:>9} {s['skipped_pre_cutoff']:>8} "
              f"{s['skipped_error']:>6}")
    print(f"{'TOTAL':<18} {total_copied:>7} {total_skipped_in_local:>9} "
          f"{total_skipped_already:>8} {total_skipped_format:>8} "
          f"{total_skipped_gender:>9} {total_skipped_date:>8} "
          f"{total_skipped_error:>6}")
    print()
    if args.dry_run:
        print(f"Dry run; would copy {total_copied} files into {GOLDEN_POOL}")
    else:
        print(f"Copied {total_copied} files into {GOLDEN_POOL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
