#!/usr/bin/env python
"""I12-L: assemble the women's *league* T20 corpus (the w2 family).

`extract_womens_t20s.py` builds the T20I-only w1 pool from the stat-generator
cricsheet mirror.  That mirror carries no women's franchise cricket except The
Hundred and the Super Smash, so the I12 scoping memo recorded women's leagues
as unavailable.  They are available — they just live under cricsheet
competition codes the mirror does not sync:

    wtb  Vitality Blast Women      (the 2026 "T20 Blast Women" markets)
    wpl  Women's Premier League
    hnd  The Hundred               (mixed zip; women's competition filtered)
    wbb  Women's Big Bash League
    cec  Charlotte Edwards Cup     (the pre-2025 English women's T20 comp)
    ssm  Super Smash               (mixed zip; women's competition filtered)

This matters because the league markets are where the *liquid* women's odds
are: the Hundred Women fixtures price at $300k+, an order of magnitude above a
typical women's T20I.

Isolation: zips land in ``data/cricsheet_womens_leagues/`` and matches in
``data/w_league_json/``.  The stat-generator mirror is never written to, and
the w1 pool (``data/w_t20s_json/``) is never touched — the two corpora are
disjoint by construction (T20I vs franchise) and are modelled separately
because their team namespaces share no ELO history.

Only ``info.gender == "female"`` and ``info.match_type == "T20"`` matches are
kept.  Cricsheet types Hundred matches as T20, which is why they qualify.

Usage:
    uv run python scripts/extract_womens_leagues.py --download
    uv run python scripts/extract_womens_leagues.py            # extract only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CRICSHEET_DOWNLOADS = "https://cricsheet.org/downloads"

# code -> human label.  Mixed-gender zips are filtered by info.gender.
COMPETITIONS = {
    "wtb": "Vitality Blast Women",
    "wpl": "Women's Premier League",
    "hnd": "The Hundred (women's competition)",
    "wbb": "Women's Big Bash League",
    "cec": "Charlotte Edwards Cup",
    "ssm": "Super Smash (women's competition)",
}

DEFAULT_ZIP_DIR = REPO_ROOT / "data" / "cricsheet_womens_leagues"
DEFAULT_OUT = REPO_ROOT / "data" / "w_league_json"


def download(zip_dir: Path) -> None:
    zip_dir.mkdir(parents=True, exist_ok=True)
    for code in COMPETITIONS:
        url = f"{CRICSHEET_DOWNLOADS}/{code}_json.zip"
        target = zip_dir / f"{code}_json.zip"
        print(f"  fetching {url}")
        with urllib.request.urlopen(url, timeout=120) as response:
            payload = response.read()
        if not payload.startswith(b"PK"):
            raise SystemExit(f"{url} did not return a zip archive")
        target.write_bytes(payload)
        print(f"    -> {target} ({len(payload) / 1e6:.1f} MB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-dir", type=Path, default=DEFAULT_ZIP_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Refresh the competition zips from cricsheet.org first.",
    )
    args = parser.parse_args()

    if args.download:
        print("downloading women's league zips from cricsheet...")
        download(args.zip_dir)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    seen_ids: dict[str, str] = {}
    total_written = 0

    for code, label in COMPETITIONS.items():
        path = args.zip_dir / f"{code}_json.zip"
        if not path.exists():
            print(f"  SKIP {code}: {path} missing (run with --download)")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        written = 0
        skipped_gender = 0
        skipped_format = 0
        events: dict[str, int] = defaultdict(int)
        dates: list[str] = []

        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if not name.endswith(".json"):
                    continue
                raw = archive.read(name)
                try:
                    info = json.loads(raw).get("info", {})
                except json.JSONDecodeError:
                    continue
                if info.get("gender") != "female":
                    skipped_gender += 1
                    continue
                if info.get("match_type") != "T20":
                    skipped_format += 1
                    continue
                stem = Path(name).name
                if stem in seen_ids and seen_ids[stem] != code:
                    # Same cricsheet id in two competition zips would double a
                    # match in the tracker walk; keep the first and report it.
                    print(f"  WARN duplicate match {stem}: {seen_ids[stem]} / {code}")
                    continue
                seen_ids[stem] = code
                (args.out / stem).write_bytes(raw)
                written += 1
                event = info.get("event") or {}
                events[
                    event.get("name") if isinstance(event, dict) else str(event)
                ] += 1
                if info.get("dates"):
                    dates.append(str(info["dates"][0]))

        total_written += written
        manifest[code] = {
            "label": label,
            "zip_sha256": digest,
            "matches": written,
            "skipped_non_female": skipped_gender,
            "skipped_non_t20": skipped_format,
            "events": dict(sorted(events.items(), key=lambda kv: -kv[1])),
            "date_range": [min(dates), max(dates)] if dates else None,
        }
        span = (
            f"{manifest[code]['date_range'][0]}..{manifest[code]['date_range'][1]}"
            if dates
            else "n/a"
        )
        print(f"  {code}: {written:4} female T20 matches  {span}  ({label})")

    all_dates = [
        d
        for entry in manifest.values()
        if entry["date_range"]
        for d in entry["date_range"]
    ]
    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": CRICSHEET_DOWNLOADS,
        "filter": "info.gender == female AND info.match_type == T20",
        "out_dir": str(args.out),
        "total_matches": total_written,
        "date_range": [min(all_dates), max(all_dates)] if all_dates else None,
        "competitions": manifest,
    }
    (args.out.parent / "w_league_manifest.json").write_text(
        json.dumps(payload, indent=2)
    )
    print(
        f"\nwrote {total_written} women's league T20s to {args.out}"
        + (f"  ({payload['date_range'][0]}..{payload['date_range'][1]})"
           if all_dates else "")
    )
    print(f"manifest -> {args.out.parent / 'w_league_manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
