"""Extract the 2026-season Vitality Blast (T20 Blast) matches from the
stat-generator cricsheet zip into a dedicated *Blast-golden* pool.

Directly analogous to `extract_golden_cricsheet.py`, but scoped to the
English domestic T20 Blast (`ntb_json.zip`) — which is NOT in the IPL
golden extractor's zip list, so the 2026 Blast season was never ingested
locally. Strictly date >= 2026-04-17, match_type=T20, gender=male.

This is staging-only: it does NOT touch data/t20s_json/, the production
SQLite cache, or any model artifact. The output dir
data/golden_blast/t20s_json/ is a separate out-of-sample pool, mirroring
how data/golden/t20s_json/ holds the IPL-2026 golden set.

Usage:
    uv run python scripts/extract_blast_golden.py
    uv run python scripts/extract_blast_golden.py --dry-run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ZIP_DIR = Path("/Users/aryamangupta/Projects/stat-generator/data/cricsheet")
LOCAL_POOL = REPO_ROOT / "data" / "t20s_json"
GOLDEN_IPL_POOL = REPO_ROOT / "data" / "golden" / "t20s_json"
BLAST_POOL = REPO_ROOT / "data" / "golden_blast" / "t20s_json"

# Strictly post-test-set window; only the T20 Blast zip.
CUTOFF_DATE = "2026-04-17"
ZIPS = ["ntb_json.zip"]


def main() -> int:
    import zipfile

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be copied; don't write.")
    args = ap.parse_args()

    if not ZIP_DIR.exists():
        print(f"ERROR: source zip dir not found: {ZIP_DIR}")
        return 1

    BLAST_POOL.mkdir(parents=True, exist_ok=True)
    # Avoid any file already in the production pool or the IPL golden pool.
    local_existing = {p.name for p in LOCAL_POOL.glob("*.json")}
    ipl_golden_existing = {p.name for p in GOLDEN_IPL_POOL.glob("*.json")}
    blast_existing = {p.name for p in BLAST_POOL.glob("*.json")}

    print(f"Production pool size: {len(local_existing):,}")
    print(f"IPL-golden pool size: {len(ipl_golden_existing):,}")
    print(f"Blast-golden pool:    {len(blast_existing):,}")
    print(f"Cutoff:               date >= {CUTOFF_DATE}, match_type=T20, gender=male")
    print()

    copied = skipped_local = skipped_ipl = skipped_already = 0
    skipped_format = skipped_gender = skipped_date = skipped_error = 0
    copied_rows = []

    for zname in ZIPS:
        zpath = ZIP_DIR / zname
        if not zpath.exists():
            print(f"  WARN: missing {zpath}; skipping")
            continue
        with zipfile.ZipFile(zpath) as zf:
            for name in (n for n in zf.namelist() if n.endswith(".json")):
                if name in local_existing:
                    skipped_local += 1
                    continue
                if name in ipl_golden_existing:
                    skipped_ipl += 1
                    continue
                if name in blast_existing:
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

                teams = info.get("teams", ["?", "?"])
                copied_rows.append((dates[0], name.replace(".json", ""),
                                    teams[0], teams[1]))
                if args.dry_run:
                    copied += 1
                    continue

                out_path = BLAST_POOL / name
                tmp = out_path.with_suffix(".json.tmp")
                with open(tmp, "w") as f:
                    json.dump(data, f, separators=(",", ":"))
                tmp.replace(out_path)
                copied += 1

    print(f"{'copied':>7} {'in_prod':>8} {'in_ipl':>7} {'already':>8} "
          f"{'non_t20':>8} {'non_male':>9} {'pre_cut':>8} {'error':>6}")
    print(f"{copied:>7} {skipped_local:>8} {skipped_ipl:>7} {skipped_already:>8} "
          f"{skipped_format:>8} {skipped_gender:>9} {skipped_date:>8} "
          f"{skipped_error:>6}")
    print()
    for dt, mid, t1, t2 in sorted(copied_rows):
        print(f"  {dt}  {mid:>8}  {t1} vs {t2}")
    print()
    if args.dry_run:
        print(f"Dry run; would copy {copied} files into {BLAST_POOL}")
    else:
        print(f"Copied {copied} files into {BLAST_POOL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
