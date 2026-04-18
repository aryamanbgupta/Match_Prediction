"""Enrich Cricsheet player IDs with metadata via the R cricketdata package.

No website scraping. Uses only:
  - data/cricsheet_people.csv             (Cricsheet register: identifier → name + key_cricinfo)
  - R cricketdata::fetch_player_meta(playerid=cricinfo_id)
  - R cricketdata::find_player_id(name)   (fallback when key_cricinfo is missing)

Pipeline:
  1. Scan data/t20s_json/*.json for info.registry.people cricsheet IDs.
  2. Diff against data/all_players_enriched.csv → set of IDs needing enrichment.
  3. For each missing ID, look up in cricsheet_people.csv to get (name, key_cricinfo).
  4. If key_cricinfo is present, fetch metadata directly. Otherwise, search by name.
  5. Append new rows to data/all_players_enriched.csv atomically.

Usage:
    uv run python scripts/enrich_players_cricketdata.py --limit 1     # smoke test
    uv run python scripts/enrich_players_cricketdata.py --dry-run     # count only
    uv run python scripts/enrich_players_cricketdata.py               # full run
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
MATCH_DIR = REPO_ROOT / "data" / "t20s_json"
PEOPLE_PATH = REPO_ROOT / "data" / "cricsheet_people.csv"
ENRICHED_PATH = REPO_ROOT / "data" / "all_players_enriched.csv"

ENRICHED_COLUMNS = [
    "cricsheet_id",
    "name",
    "cricinfo_id",
    "unique_name",
    "full_name",
    "country",
    "dob",
    "batting_style",
    "bowling_style",
]


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def collect_referenced_cricsheet_ids(match_dir: Path) -> set[str]:
    """Scan every match JSON and return the union of info.registry.people IDs."""
    ids: set[str] = set()
    files = sorted(glob.glob(str(match_dir / "*.json")))
    for path in tqdm(files, desc="Scanning match registry IDs"):
        try:
            with open(path) as f:
                blob = json.load(f)
        except Exception:
            continue
        registry = (blob.get("info", {}) or {}).get("registry", {}) or {}
        people = registry.get("people", {}) or {}
        ids.update(str(v) for v in people.values() if v)
    return ids


def load_enriched_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            val = (row.get("cricsheet_id") or "").strip()
            if val:
                ids.add(val)
    return ids


def load_cricsheet_register(path: Path) -> dict[str, dict]:
    """Return {identifier: {'name', 'unique_name', 'key_cricinfo'}}."""
    by_id: dict[str, dict] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            ident = (row.get("identifier") or "").strip()
            if not ident:
                continue
            by_id[ident] = {
                "name": (row.get("name") or "").strip(),
                "unique_name": (row.get("unique_name") or "").strip(),
                "key_cricinfo": (row.get("key_cricinfo") or "").strip(),
            }
    return by_id


# ---------------------------------------------------------------------------
# R bridge (cricketdata)
# ---------------------------------------------------------------------------

class CricketData:
    """Thin wrapper around rpy2 + cricketdata, lazy-imported."""

    def __init__(self) -> None:
        import rpy2.robjects as robjects
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects.packages import importr

        self._robjects = robjects
        self._pandas2ri = pandas2ri
        self._localconverter = localconverter
        self.pkg = importr("cricketdata")

    def _r_to_df(self, r_obj) -> pd.DataFrame:
        with self._localconverter(self._robjects.default_converter + self._pandas2ri.converter):
            return self._robjects.conversion.rpy2py(r_obj)

    def find_player_id(self, name: str) -> str | None:
        """Name search → first matching cricinfo ID, or None."""
        try:
            df = self._r_to_df(self.pkg.find_player_id(name))
            if len(df) == 0 or "ID" not in df.columns:
                return None
            raw = df.iloc[0]["ID"]
            if pd.isna(raw):
                return None
            return str(int(raw))
        except Exception:
            return None

    def fetch_player_meta(self, cricinfo_id: str) -> dict | None:
        """Direct metadata fetch for a known cricinfo ID."""
        try:
            r_meta = self.pkg.fetch_player_meta(playerid=cricinfo_id)
            df = self._r_to_df(r_meta)
            if len(df) == 0:
                return None
            return df.iloc[0].to_dict()
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

def convert_r_date(raw) -> str | None:
    """cricketdata returns DOB as days-since-1970 float; convert to YYYY-MM-DD."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    try:
        return (pd.Timestamp("1970-01-01") + pd.to_timedelta(float(raw), unit="d")).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def clean_str(raw) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, float) and pd.isna(raw):
        return None
    s = str(raw).strip()
    if s in {"", "NA", "NA_character_", "nan", "None"}:
        return None
    return s


def build_row(cricsheet_id: str, register_entry: dict, meta: dict | None, cricinfo_id: str | None) -> dict:
    name = register_entry.get("name") or ""
    unique_name = register_entry.get("unique_name") or ""
    if meta is None:
        return {
            "cricsheet_id": cricsheet_id,
            "name": name,
            "cricinfo_id": cricinfo_id or "",
            "unique_name": unique_name,
            "full_name": name,
            "country": "",
            "dob": "",
            "batting_style": "",
            "bowling_style": "",
        }
    return {
        "cricsheet_id": cricsheet_id,
        "name": name,
        "cricinfo_id": cricinfo_id or "",
        "unique_name": unique_name,
        "full_name": clean_str(meta.get("name")) or name,
        "country": clean_str(meta.get("country")) or "",
        "dob": convert_r_date(meta.get("dob")) or "",
        "batting_style": clean_str(meta.get("batting_style")) or "",
        "bowling_style": clean_str(meta.get("bowling_style")) or "",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich cricsheet player IDs via R cricketdata (no scraping)")
    parser.add_argument("--limit", type=int, default=None, help="Only enrich the first N missing IDs")
    parser.add_argument("--dry-run", action="store_true", help="Print counts and exit; no R calls, no writes")
    parser.add_argument("--sleep-every", type=int, default=10, help="Throttle: sleep 0.3s every N calls")
    args = parser.parse_args()

    # --- Input sanity ---
    if not PEOPLE_PATH.exists():
        print(f"ERROR: {PEOPLE_PATH} not found. Run scripts/fetch_cricsheet.py first.", file=sys.stderr)
        sys.exit(1)
    if not ENRICHED_PATH.exists():
        print(f"ERROR: {ENRICHED_PATH} not found.", file=sys.stderr)
        sys.exit(1)

    print("Scanning match registry IDs ...")
    referenced = collect_referenced_cricsheet_ids(MATCH_DIR)
    enriched = load_enriched_ids(ENRICHED_PATH)
    register = load_cricsheet_register(PEOPLE_PATH)

    missing = sorted(referenced - enriched)
    not_in_register = [cid for cid in missing if cid not in register]
    in_register = [cid for cid in missing if cid in register]

    print(f"Referenced in matches:         {len(referenced):>7,}")
    print(f"Already enriched:              {len(enriched):>7,}")
    print(f"Missing enrichment:            {len(missing):>7,}")
    print(f"  with register entry:         {len(in_register):>7,}")
    print(f"  not in cricsheet_people.csv: {len(not_in_register):>7,}  (cannot enrich)")

    with_cricinfo = sum(1 for cid in in_register if register[cid]["key_cricinfo"])
    print(f"  with key_cricinfo (fast):    {with_cricinfo:>7,}")
    print(f"  name-search fallback:        {len(in_register) - with_cricinfo:>7,}")

    if args.dry_run:
        print("\nDry run — no R calls, no writes.")
        return

    if not in_register:
        print("\nNothing to do.")
        return

    todo = in_register if args.limit is None else in_register[: args.limit]
    print(f"\nEnriching {len(todo)} player(s) via R cricketdata ...")

    cd = CricketData()

    new_rows: list[dict] = []
    succeeded = 0
    failed = 0

    for i, cid in enumerate(tqdm(todo, desc="Enriching")):
        entry = register[cid]
        cricinfo_id = entry.get("key_cricinfo") or None
        if not cricinfo_id and entry.get("name"):
            cricinfo_id = cd.find_player_id(entry["name"])

        meta = cd.fetch_player_meta(cricinfo_id) if cricinfo_id else None
        row = build_row(cid, entry, meta, cricinfo_id)
        new_rows.append(row)

        if meta is not None:
            succeeded += 1
        else:
            failed += 1

        if args.sleep_every > 0 and (i + 1) % args.sleep_every == 0:
            time.sleep(0.3)

    print(f"\nFetched metadata for {succeeded}; {failed} unresolved (row written with blank fields).")

    # --- Append atomically ---
    existing = pd.read_csv(ENRICHED_PATH, dtype=str, keep_default_na=False)
    # Ensure schema compat
    for col in ENRICHED_COLUMNS:
        if col not in existing.columns:
            existing[col] = ""
    new_df = pd.DataFrame(new_rows, columns=ENRICHED_COLUMNS)
    combined = pd.concat([existing[ENRICHED_COLUMNS], new_df], ignore_index=True)

    tmp = ENRICHED_PATH.with_suffix(".csv.tmp")
    combined.to_csv(tmp, index=False)
    tmp.replace(ENRICHED_PATH)

    print(f"Appended {len(new_rows)} row(s) → {ENRICHED_PATH}  (now {len(combined):,} total)")

    # Show samples
    if new_rows:
        print("\nSample of newly-enriched players:")
        preview_cols = ["cricsheet_id", "name", "cricinfo_id", "country", "dob", "batting_style", "bowling_style"]
        print(new_df[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
