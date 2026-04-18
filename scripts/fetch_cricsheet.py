"""Download Cricsheet men's T20 JSON archives and merge into data/t20s_json/.

Safe by design:
  - Downloads go through a .staging/ directory first
  - Zip cache is only replaced after a successful SHA-256 compare
  - Match JSON merge is append-only (Cricsheet JSONs are immutable once published)
  - If the script crashes mid-run, live data (data/t20s_json/) is untouched

Usage:
    # Full refresh (all 14 men's T20 leagues + player register)
    uv run python scripts/fetch_cricsheet.py

    # Preview (no files written)
    uv run python scripts/fetch_cricsheet.py --dry-run

    # Single league(s)
    uv run python scripts/fetch_cricsheet.py --only ipl,bbl

    # List leagues (including what's deliberately excluded)
    uv run python scripts/fetch_cricsheet.py --list

    # Force re-download even when the hash matches the manifest
    uv run python scripts/fetch_cricsheet.py --force

    # Verbose logging
    uv run python scripts/fetch_cricsheet.py --verbose

This script does NOT trigger parsing_v2.py. After a successful run it prints the
command to run manually — see `docs/OPERATIONS.md` for details.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL = "https://cricsheet.org/downloads"
REGISTER_URL = "https://cricsheet.org/register/people.csv"

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
ZIP_DIR = DATA_DIR / ".cricsheet_zips"
STAGING_DIR = ZIP_DIR / ".staging"
EXTRACT_DIR = ZIP_DIR / ".extract"
MANIFEST_PATH = ZIP_DIR / "manifest.json"
LOG_FILE = ZIP_DIR / ".refresh.log"

MATCH_DIR = DATA_DIR / "t20s_json"
PEOPLE_PATH = DATA_DIR / "cricsheet_people.csv"
ENRICHED_PATH = DATA_DIR / "all_players_enriched.csv"

# key → download slug + description. 14 men's T20 leagues.
# NOTE: `t20s_male_json.zip` is the men-only T20I archive. `t20s_json.zip`
# (no gender suffix) is T20Is of both genders — we don't use that one because
# the pipeline is men's only.
LEAGUES: dict[str, dict] = {
    "t20i":      {"slug": "t20s_male_json.zip", "desc": "Men's T20 Internationals"},
    "ipl":       {"slug": "ipl_json.zip",       "desc": "Indian Premier League"},
    "bbl":       {"slug": "bbl_json.zip",       "desc": "Big Bash League"},
    "cpl":       {"slug": "cpl_json.zip",       "desc": "Caribbean Premier League"},
    "t20_blast": {"slug": "ntb_json.zip",       "desc": "T20 Blast (England)"},
    "ctc":       {"slug": "ctc_json.zip",       "desc": "CSA T20 Challenge"},
    "ilt20":     {"slug": "ilt_json.zip",       "desc": "International League T20"},
    "sa20":      {"slug": "sat_json.zip",       "desc": "SA20"},
    "mlc":       {"slug": "mlc_json.zip",       "desc": "Major League Cricket"},
    "bpl":       {"slug": "bpl_json.zip",       "desc": "Bangladesh Premier League"},
    "ssm":       {"slug": "ssm_json.zip",       "desc": "Super Smash (NZ)"},
    "psl":       {"slug": "psl_json.zip",       "desc": "Pakistan Super League"},
    "smat":      {"slug": "sma_json.zip",       "desc": "Syed Mushtaq Ali Trophy"},
    "lpl":       {"slug": "lpl_json.zip",       "desc": "Lanka Premier League"},
}

# Printed by --list for transparency about deliberate exclusions.
EXCLUSIONS = [
    ("hnd_json.zip",          "The Hundred — 100-ball format; pipeline hardcodes 120 balls/innings. See TODO."),
    ("t20s_female_json.zip",  "Women's T20Is — excluded; pipeline is men's only."),
    ("wbb_json.zip / wpl",    "Women's leagues — excluded; pipeline is men's only."),
    ("it20s_*_json.zip",      "Non-official associate T20Is — data quality concerns."),
    ("tests_*, odis_*, all_json.zip", "Out of scope — we fetch men's T20 only."),
]

log = logging.getLogger("fetch_cricsheet")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def file_hash(path: Path) -> str:
    """SHA-256 hash of a file, streamed in 128KB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(128 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path) -> bool:
    """Download `url` to `dest` via a .tmp intermediate. True on success."""
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 cricml-fetcher"})
        with urlopen(req, timeout=180) as resp, open(tmp, "wb") as f:
            while True:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        tmp.rename(dest)
        return True
    except Exception as e:
        log.error("Download failed: %s — %s", url, e)
        tmp.unlink(missing_ok=True)
        return False


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    try:
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    except Exception as e:
        log.warning("Manifest unreadable (%s); starting fresh", e)
        return {}


def save_manifest(manifest: dict) -> None:
    tmp = MANIFEST_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    tmp.rename(MANIFEST_PATH)


def extract_zip(zip_path: Path, out_dir: Path) -> list[Path]:
    """Extract JSON files from zip_path into out_dir. Returns list of extracted paths."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".json"):
                continue
            # Flatten: take just the basename so we land directly in out_dir.
            basename = Path(name).name
            if not basename:
                continue
            target = out_dir / basename
            with zf.open(name) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(target)
    return extracted


def merge_new_matches(extracted: list[Path], live_dir: Path) -> list[tuple[str, str | None, str | None]]:
    """Copy any extracted JSONs whose filenames are not already present in live_dir.

    Returns a list of (match_id, date, event_name) for the newly-added files.
    """
    added: list[tuple[str, str | None, str | None]] = []
    for src in extracted:
        target = live_dir / src.name
        if target.exists():
            continue
        # Read minimal metadata before we commit to copying, so the summary is informative.
        date_str: str | None = None
        event_name: str | None = None
        try:
            with open(src) as f:
                blob = json.load(f)
            info = blob.get("info", {}) or {}
            dates = info.get("dates") or []
            date_str = dates[0] if dates else None
            ev = info.get("event") or {}
            event_name = ev.get("name") if isinstance(ev, dict) else (ev if isinstance(ev, str) else None)
        except Exception as e:
            log.debug("Could not parse %s for metadata: %s", src.name, e)
        shutil.copy2(src, target)
        added.append((src.stem, date_str, event_name))
    return added


def registry_ids_in_matches(paths: list[Path]) -> set[str]:
    """Collect cricsheet registry IDs (info.registry.people values) across match JSONs."""
    ids: set[str] = set()
    for p in paths:
        try:
            with open(p) as f:
                blob = json.load(f)
        except Exception:
            continue
        registry = (blob.get("info", {}) or {}).get("registry", {}) or {}
        people = registry.get("people", {}) or {}
        ids.update(str(v) for v in people.values() if v)
    return ids


def enriched_cricsheet_ids() -> set[str]:
    if not ENRICHED_PATH.exists():
        return set()
    ids: set[str] = set()
    with open(ENRICHED_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = (row.get("cricsheet_id") or "").strip()
            if val:
                ids.add(val)
    return ids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Cricsheet men's T20 JSON archives and merge into data/t20s_json/",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded")
    parser.add_argument("--only", type=str, help="Comma-separated league keys (e.g. ipl,bbl,t20i)")
    parser.add_argument("--force", action="store_true", help="Re-download even if hash matches")
    parser.add_argument("--list", action="store_true", help="List leagues + exclusions and exit")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    if args.list:
        print("Leagues included (14 men's T20 archives):")
        for key, info in sorted(LEAGUES.items()):
            print(f"  {key:10s}  {info['desc']:36s}  {info['slug']}")
        print("\nDeliberately excluded:")
        for slug, reason in EXCLUSIONS:
            print(f"  {slug:32s}  {reason}")
        return

    # Setup dirs + logging
    ZIP_DIR.mkdir(parents=True, exist_ok=True)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    MATCH_DIR.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE, mode="a")],
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    log.info("=" * 60)
    log.info("Cricsheet refresh started at %s", timestamp)

    # Filter leagues
    leagues = dict(LEAGUES)
    if args.only:
        requested = {s.strip().lower() for s in args.only.split(",") if s.strip()}
        unknown = requested - set(leagues)
        if unknown:
            log.error("Unknown league keys: %s (available: %s)", sorted(unknown), sorted(leagues))
            sys.exit(1)
        leagues = {k: v for k, v in leagues.items() if k in requested}
    log.info("Leagues in scope: %d", len(leagues))

    manifest = load_manifest()

    results: dict[str, dict] = {}  # key → {"status": str, "added": [...]}
    all_added_paths: list[Path] = []
    all_added_meta: list[tuple[str, str | None, str | None]] = []

    for key in sorted(leagues):
        info = leagues[key]
        slug = info["slug"]
        url = f"{BASE_URL}/{slug}"
        live_zip = ZIP_DIR / slug
        staged_zip = STAGING_DIR / slug

        log.info("[%s] %s — %s", key, info["desc"], slug)

        if args.dry_run:
            prior = manifest.get(slug, {})
            prior_hash = (prior.get("sha256") or "")[:10] or "none"
            log.info("[%s] DRY RUN — would download (prior hash: %s)", key, prior_hash)
            results[key] = {"status": "dry_run", "added": []}
            continue

        if not download_file(url, staged_zip):
            results[key] = {"status": "download_failed", "added": []}
            continue

        new_hash = file_hash(staged_zip)
        prior_hash = manifest.get(slug, {}).get("sha256")

        if prior_hash == new_hash and live_zip.exists() and not args.force:
            log.info("[%s] Zip unchanged (hash match) — no extract needed", key)
            staged_zip.unlink(missing_ok=True)
            results[key] = {"status": "unchanged", "added": []}
            continue

        # Commit new zip to cache
        staged_zip.replace(live_zip)

        # Extract + diff-merge
        try:
            league_extract = EXTRACT_DIR / key
            extracted = extract_zip(live_zip, league_extract)
            added = merge_new_matches(extracted, MATCH_DIR)
            # Keep paths of newly-added match files (in MATCH_DIR) for downstream registry diff
            added_paths = [MATCH_DIR / f"{mid}.json" for (mid, _, _) in added]
            all_added_paths.extend(added_paths)
            all_added_meta.extend(added)
            log.info(
                "[%s] Zip updated (%d KB) — %d JSONs extracted, %d new (added to %s)",
                key,
                live_zip.stat().st_size // 1024,
                len(extracted),
                len(added),
                MATCH_DIR,
            )
        except Exception as e:
            log.error("[%s] Extract/merge failed: %s — leaving manifest unchanged", key, e)
            results[key] = {"status": "extract_failed", "added": []}
            continue
        finally:
            # Clean extract dir regardless of outcome
            shutil.rmtree(EXTRACT_DIR / key, ignore_errors=True)

        manifest[slug] = {
            "sha256": new_hash,
            "downloaded_at": timestamp,
            "n_matches_in_zip": len(extracted),
            "n_new_this_run": len(added),
        }
        save_manifest(manifest)
        results[key] = {"status": "updated", "added": added}

    # ----- Player register -----
    register_status = "skipped"
    if not args.dry_run:
        log.info("Downloading player register (people.csv)...")
        people_staged = STAGING_DIR / "people.csv"
        if download_file(REGISTER_URL, people_staged):
            if PEOPLE_PATH.exists() and not args.force and file_hash(people_staged) == file_hash(PEOPLE_PATH):
                log.info("people.csv unchanged")
                people_staged.unlink(missing_ok=True)
                register_status = "unchanged"
            else:
                people_staged.replace(PEOPLE_PATH)
                log.info("people.csv → %s", PEOPLE_PATH)
                register_status = "updated"
        else:
            log.warning("Failed to download people.csv — player register may be stale")
            register_status = "failed"

    # Cleanup staging
    for leftover in STAGING_DIR.glob("*"):
        leftover.unlink(missing_ok=True)
    shutil.rmtree(EXTRACT_DIR, ignore_errors=True)

    # ----- Summary -----
    log.info("-" * 60)
    updated = [k for k, r in results.items() if r["status"] == "updated"]
    unchanged = [k for k, r in results.items() if r["status"] == "unchanged"]
    failed = [k for k, r in results.items() if r["status"].endswith("failed")]
    dry = [k for k, r in results.items() if r["status"] == "dry_run"]

    log.info("Per-league summary:")
    for key in sorted(results):
        r = results[key]
        log.info("  %-10s  %-16s  +%d new", key, r["status"], len(r["added"]))

    total_new = len(all_added_meta)
    log.info("Totals: %d new match JSONs across %d updated leagues", total_new, len(updated))
    if all_added_meta:
        dates = sorted(d for (_, d, _) in all_added_meta if d)
        if dates:
            log.info("New-match date range: %s → %s", dates[0], dates[-1])

    log.info("Player register: %s", register_status)
    if unchanged:
        log.info("Unchanged: %s", unchanged)
    if failed:
        log.info("Failed: %s", failed)
    if dry:
        log.info("Dry-run leagues: %s", dry)

    # Registry diff vs enriched player metadata
    if all_added_paths:
        new_registry_ids = registry_ids_in_matches(all_added_paths)
        enriched = enriched_cricsheet_ids()
        unenriched = new_registry_ids - enriched
        log.info(
            "Player registry: %d unique IDs referenced by new matches; %d not in all_players_enriched.csv",
            len(new_registry_ids),
            len(unenriched),
        )
        if unenriched:
            sample = sorted(unenriched)[:10]
            log.info("  Sample unenriched IDs: %s%s", sample, " ..." if len(unenriched) > 10 else "")
            log.info("  (Enrichment is a manual follow-up — not invoked here.)")

    if args.dry_run:
        log.info("Dry run complete — no files were changed")
        return

    if updated or total_new:
        log.info("")
        log.info("NEXT STEP (manual):")
        log.info("  uv run python scripts/parsing_v2.py")
        log.info("  (regenerates data/xgb_data_v3/ and models/cache_chunks_v3/ — destructive, 10-15 min)")

    if failed:
        sys.exit(1)

    log.info("Refresh complete")


if __name__ == "__main__":
    main()
