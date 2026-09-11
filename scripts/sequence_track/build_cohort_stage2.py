#!/usr/bin/env python3
"""Build the stage 2 untouched ball-level cohort (D5 of
``docs/sequence_track/stage2_acceptance.md``).

The cohort is every male six-ball T20 match dated 2026-04-17 .. 2026-08-05
that no pipeline on this branch has trained, tuned or inspected: the raw
JSONs come from the stat-generator zips (never the closed golden tree), and
the sealed forward fixtures plus the match-level golden fixtures are excluded
from *scoring* while remaining chronological *context* for tracker state.

Access rule
-----------
The only permitted reads under ``data/golden/`` and ``data/forward_holdout/``
are the two id-only reads the user permitted in writing on 2026-09-11 ("you
can read the files"), implemented by :func:`read_forward_fixture_ids` and
:func:`read_golden_fixture_ids`. Both return a sorted list of match ids and
nothing else. This module deliberately does not import
``scripts/build_forward_state.py`` (that script verifies and reads the
sealed holdout).

Steps (``--step``, default ``all``)::

    extract      5.1  zips -> cohort/raw_json/ + raw_manifest.json
    audit        5.2  eligibility_audit.csv (+ 5.3 consumer notes)
    state        5.4  sidecar stats cache under cohort/state/
    materialize  5.6  ball rows for the eligible matches
    parity       5.6  20 i7 test matches rebuilt bit-for-bit
    context      5.5  scored ids == eligible; one hand-verified context case
    freeze       5.7  cohort/FROZEN.json

Usage::

    uv run --no-sync python scripts/sequence_track/build_cohort_stage2.py \
        --step all
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

ZIP_DIR = Path(
    "/Users/aryamangupta/Projects/stat-generator/data/cricsheet"
)
SKIP_ZIPS = {"odis_json.zip", "tests_json.zip"}

WINDOW_START = "2026-04-17"
WINDOW_END = "2026-08-05"
EXPECTED_RAW_COUNT = 471
EXPECTED_FORWARD_HITS = 137
EXPECTED_GOLDEN_HITS = 124
EXPECTED_IN_REPO_HITS = 0

OUT_DIR = ROOT / "models" / "embeddings" / "seq_stage2" / "cohort"
RAW_JSON_DIR = OUT_DIR / "raw_json"
RAW_MANIFEST = OUT_DIR / "raw_manifest.json"
AUDIT_CSV = OUT_DIR / "eligibility_audit.csv"
# Written by the audit step: the counts of the two permitted id-only
# reads, so no later step has to re-open a closed file to restate them.
AUDIT_SUMMARY_JSON = OUT_DIR / "eligibility_summary.json"
NOTES_CONSUMERS = OUT_DIR / "notes_consumers.md"
NOTES_CONTEXT = OUT_DIR / "notes_context_case.md"
STATE_DIR = OUT_DIR / "state"
CORPUS_DIR = STATE_DIR / "corpus_json"
SIDECAR_SQLITE = STATE_DIR / "player_stats_cache_i7.sqlite"
FULL_OUT_DIR = OUT_DIR / "materialized_full"
COHORT_PARQUET = OUT_DIR / "cricket_data_i7_cohort.parquet"
COHORT_FEATURE_HASH = OUT_DIR / ".feature_hash"
FROZEN_JSON = OUT_DIR / "FROZEN.json"
PARITY_JSON = OUT_DIR / "parity_test_split.json"

REPO_CORPUS = ROOT / "data" / "t20s_json"
FRAME_DIR = ROOT / "data" / "xgb_data_i7"
FRAME_FEATURE_HASH = FRAME_DIR / ".feature_hash"
FRAME_TEST_PARQUET = FRAME_DIR / "cricket_data_i7_test.parquet"
PROD_I7_SQLITE = ROOT / "models" / "player_stats_cache_i7.sqlite"
METADATA_CSV = ROOT / "data" / "all_players_enriched.csv"

FORWARD_MANIFEST = (
    ROOT / "data" / "forward_holdout" / "2026-06-01_2026-07-13"
    / "manifest.json"
)
GOLDEN_ODDS = ROOT / "data" / "golden" / "betting_odds_golden_v2.json"

# Directory names the D5.3 scan must not list, derived from the two
# permitted-read constants so no closed path is spelled out twice.
CLOSED_SKIP_DIRS = frozenset({
    GOLDEN_ODDS.parent.name,
    FORWARD_MANIFEST.parent.parent.name,
    "forward_state",
})

USER_PERMISSION_QUOTE = "you can read the files"
USER_PERMISSION_DATE = "2026-09-11"

# i7 contract (experiments/configs/xgb_i7_venue_identity.yaml).
I7_VERSION = "i7"
I7_GENDER = "male"
I7_DELIVERY_SEMANTICS = "inclusive_total_runs_v1"
I7_SPLITS = {
    "train_end": "2024-12-31",
    "val_end": "2025-06-30",
    "test_end": "2026-04-16",
    "golden_start": "2026-04-17",
}
I7_K_PLAYER = 30.0
I7_K_VENUE = 200.0

PARITY_N_MATCHES = 20
PARITY_SEED = 29

# Precedence order for the eligibility reason. `hundred` cannot occur after
# the 5.1 balls_per_over == 6 filter but is kept so the reason vocabulary
# matches the acceptance file and the filter stays auditable.
REASON_ORDER = (
    "hundred",
    "in_repo_corpus",
    "forward_fixture",
    "golden_fixture",
)


# --------------------------------------------------------------------------
# id-only readers (the two permitted reads)
# --------------------------------------------------------------------------

def _match_ids_only(path: Path) -> list[str]:
    """Return the sorted match ids in ``path`` and nothing else.

    Opens the file, extracts only the ``match_id`` / ``cricsheet_id`` fields
    of ``matches[*]``, closes it, and returns the sorted unique id list. No
    other field is read, logged, stored or returned.
    """
    with path.open() as handle:
        payload = json.load(handle)
    ids: set[str] = set()
    for entry in payload.get("matches", []):
        if not isinstance(entry, dict):
            continue
        for key in ("match_id", "cricsheet_id"):
            value = entry.get(key)
            if isinstance(value, str) and value:
                ids.add(value)
    del payload
    return sorted(ids)


def read_forward_fixture_ids(
    path: Path = FORWARD_MANIFEST,
) -> list[str]:
    """Sorted ids of the sealed forward holdout (ids only; see module doc)."""
    return _match_ids_only(path)


def read_golden_fixture_ids(path: Path = GOLDEN_ODDS) -> list[str]:
    """Sorted ids of the match-level golden set (ids only; see module doc)."""
    return _match_ids_only(path)


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path) -> str:
    """Repo-relative string where possible, absolute otherwise."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _in_window(date_str: str) -> bool:
    return WINDOW_START <= date_str <= WINDOW_END


def match_id_from_innings_id(innings_id: str) -> str:
    """``"1_1482013"`` -> ``"1482013"`` (the Cricsheet file stem)."""
    return innings_id.split("_", 1)[1]


# --------------------------------------------------------------------------
# 5.1 extraction
# --------------------------------------------------------------------------

def keep_match(info: dict) -> bool:
    if info.get("gender") != "male":
        return False
    if info.get("match_type") != "T20":
        return False
    if int(info.get("balls_per_over", 6)) != 6:
        return False
    dates = info.get("dates") or []
    if not dates:
        return False
    return _in_window(str(dates[0]))


def step_extract(strict: bool = True) -> dict:
    _guard_frozen("extract")
    t0 = time.time()
    zips = sorted(
        p for p in ZIP_DIR.glob("*.zip") if p.name not in SKIP_ZIPS
    )
    if not zips:
        raise RuntimeError(f"no candidate zips under {ZIP_DIR}")

    RAW_JSON_DIR.mkdir(parents=True, exist_ok=True)
    for stale in RAW_JSON_DIR.glob("*.json"):
        stale.unlink()

    matches: dict[str, dict] = {}
    duplicates: list[dict] = []
    scanned = 0
    for zip_path in zips:
        with zipfile.ZipFile(zip_path) as archive:
            for member in sorted(archive.namelist()):
                if not member.endswith(".json"):
                    continue
                scanned += 1
                payload = archive.read(member)
                try:
                    data = json.loads(payload)
                    info = data["info"]
                except (json.JSONDecodeError, KeyError):
                    continue
                if not keep_match(info):
                    continue
                stem = Path(member).stem
                digest = sha256_bytes(payload)
                event = info.get("event") or {}
                record = {
                    "sha256": digest,
                    "source_zip": zip_path.name,
                    "source_member": member,
                    "date": str(info["dates"][0]),
                    "event": (
                        event.get("name", "")
                        if isinstance(event, dict)
                        else str(event)
                    ),
                    "teams": list(info.get("teams", [])),
                }
                previous = matches.get(stem)
                if previous is not None:
                    if previous["sha256"] != digest:
                        raise RuntimeError(
                            "differing duplicate cricsheet stem "
                            f"{stem!r}: {previous['source_zip']}"
                            f":{previous['source_member']} sha "
                            f"{previous['sha256'][:12]} vs "
                            f"{zip_path.name}:{member} sha {digest[:12]}"
                        )
                    duplicates.append(
                        {
                            "stem": stem,
                            "first": (
                                f"{previous['source_zip']}"
                                f":{previous['source_member']}"
                            ),
                            "duplicate": f"{zip_path.name}:{member}",
                            "sha256": digest,
                        }
                    )
                    continue
                matches[stem] = record
                (RAW_JSON_DIR / f"{stem}.json").write_bytes(payload)

    manifest = {
        "generated_at": utc_now(),
        "window": {"start": WINDOW_START, "end": WINDOW_END},
        "filters": {
            "gender": "male",
            "match_type": "T20",
            "balls_per_over": 6,
            "balls_per_over_default_when_absent": 6,
        },
        "source_zip_dir": str(ZIP_DIR),
        "source_zips": [
            {"name": p.name, "sha256": sha256_file(p)} for p in zips
        ],
        "skipped_zips": sorted(SKIP_ZIPS),
        "members_scanned": scanned,
        "byte_identical_duplicates": duplicates,
        "match_count": len(matches),
        "matches": {stem: matches[stem] for stem in sorted(matches)},
    }
    RAW_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    print(
        f"[5.1] scanned {scanned:,} members in {len(zips)} zips -> "
        f"{len(matches)} matches "
        f"({len(duplicates)} byte-identical duplicates) in "
        f"{time.time() - t0:.0f}s"
    )
    if len(matches) != EXPECTED_RAW_COUNT:
        message = (
            f"expected {EXPECTED_RAW_COUNT} matches, got {len(matches)}"
        )
        if strict:
            raise RuntimeError(message)
        print(f"  WARNING: {message}")
    return manifest


# --------------------------------------------------------------------------
# 5.2 / 5.3 eligibility audit
# --------------------------------------------------------------------------

def classify(
    stem: str,
    balls_per_over: int,
    forward_ids: set[str],
    golden_ids: set[str],
    repo_ids: set[str],
) -> str:
    flags = {
        "hundred": balls_per_over != 6,
        "in_repo_corpus": stem in repo_ids,
        "forward_fixture": stem in forward_ids,
        "golden_fixture": stem in golden_ids,
    }
    for reason in REASON_ORDER:
        if flags[reason]:
            return reason
    return "eligible"


def step_audit() -> dict:
    _guard_frozen("audit")
    manifest = json.loads(RAW_MANIFEST.read_text())
    forward_ids = set(read_forward_fixture_ids())
    golden_ids = set(read_golden_fixture_ids())
    repo_ids = {p.stem for p in REPO_CORPUS.glob("*.json")}

    rows = []
    counts: dict[str, int] = {}
    for stem in sorted(manifest["matches"]):
        record = manifest["matches"][stem]
        reason = classify(stem, 6, forward_ids, golden_ids, repo_ids)
        counts[reason] = counts.get(reason, 0) + 1
        rows.append(
            {
                "stem": stem,
                "date": record["date"],
                "event": record["event"],
                "eligible": reason == "eligible",
                "reason": reason,
            }
        )

    AUDIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["stem", "date", "event", "eligible", "reason"]
        )
        writer.writeheader()
        writer.writerows(rows)

    stems = set(manifest["matches"])
    summary = {
        "total": len(rows),
        "counts_by_reason": dict(sorted(counts.items())),
        "eligible": counts.get("eligible", 0),
        "forward_ids_read": len(forward_ids),
        "golden_ids_read": len(golden_ids),
        "forward_hits_in_cohort_window": len(forward_ids & stems),
        "golden_hits_in_cohort_window": len(golden_ids & stems),
        "forward_golden_id_overlap": len(forward_ids & golden_ids),
        "in_repo_hits": len(repo_ids & stems),
    }
    # The two permitted closed-file reads happen HERE and only here, so their
    # counts are written down for the freeze to quote (Astra SHOULD 2).
    AUDIT_SUMMARY_JSON.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(f"[5.2] {json.dumps(summary, indent=2)}")
    for key, expected in (
        ("forward_hits_in_cohort_window", EXPECTED_FORWARD_HITS),
        ("golden_hits_in_cohort_window", EXPECTED_GOLDEN_HITS),
        ("in_repo_hits", EXPECTED_IN_REPO_HITS),
    ):
        if summary[key] != expected:
            print(
                f"  NOTE: {key} = {summary[key]}, expected {expected}"
            )
    write_consumer_notes(summary)
    return summary


def write_consumer_notes(summary: dict) -> None:
    """5.3 — record the grep for other consumers of post-2026-04-16 data."""
    lines: list[str] = []

    def run(label: str, command: list[str]) -> None:
        lines.append(f"### {label}")
        lines.append("")
        lines.append("```")
        lines.append("$ " + " ".join(command))
        proc = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True
        )
        out = (proc.stdout + proc.stderr).strip()
        lines.append(out if out else "(no output)")
        lines.append("```")
        lines.append("")

    lines.append("# D5.3 — consumers of post-2026-04-16 Cricsheet")
    lines.append("")
    lines.append(f"Recorded {utc_now()} by `build_cohort_stage2.py`.")
    lines.append("")
    lines.append(
        "Known consumers on the branch: `data/t20s_json` (ends 2026-04-16), "
        "the sealed forward fixture ids, the match-level golden fixture ids, "
        "and the Hundred (`balls_per_over` 5, excluded by the 5.1 filter). "
        "The greps below are the evidence that nothing else on the branch "
        "consumes this window."
    )
    lines.append("")
    lines.append("### top-level `data/` listing")
    lines.append("")
    lines.append("```")
    for entry in sorted(p.name for p in (ROOT / "data").iterdir()):
        lines.append(entry)
    lines.append("```")
    lines.append("")
    run(
        "grep scripts/daily/ and daily/ for 2026-04/05/06/07/08 dates",
        [
            "grep", "-rn", "-E",
            r"2026-0[45678]",
            "scripts/daily", "daily",
        ],
    )
    run(
        "grep scripts/daily/ and daily/ for cricsheet sources",
        ["grep", "-rn", "-E", r"cricsheet|t20s_json|stat-generator",
         "scripts/daily", "daily"],
    )
    lines.append("### cohort-window raw JSONs elsewhere under `data/`")
    lines.append("")
    lines.append(
        "Every top-level directory under `data/` is scanned for Cricsheet "
        "stems belonging to the 471-match window. The closed golden and "
        "forward-holdout trees, and the forward state sidecar derived from "
        "the latter, are skipped by "
        "the access rule (their listings would themselves reveal ids beyond "
        "the two permitted id-only reads); their fixture ids are already "
        "excluded through those reads."
    )
    lines.append("")
    lines.append("```")
    try:
        window_ids = set(json.loads(RAW_MANIFEST.read_text())["matches"])
    except (OSError, KeyError, json.JSONDecodeError):
        window_ids = set()
    skipped = CLOSED_SKIP_DIRS
    for entry in sorted((ROOT / "data").iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in skipped:
            lines.append(f"{entry.name}: skipped by the access rule")
            continue
        if entry.resolve() == CORPUS_DIR.resolve():
            continue
        stems = {q.stem for q in entry.rglob("*.json")}
        overlap = len(stems & window_ids)
        lines.append(
            f"{entry.name}: {len(stems)} json files, "
            f"{overlap} in the cohort window"
        )
    lines.append("```")
    lines.append("")
    lines.append(
        "Finding (2026-09-11): `data/t20s_json_fresh_export_20260805/` holds "
        "all 471 window matches. It is the parked local Cricsheet 1.2.0 "
        "export renamed aside under "
        "`docs/remediation/item5_acceptance.md` (update 2026-09-10); no "
        "script, config or report references it, so it is raw data at rest "
        "rather than a consumer. The corpus of record remains "
        "`data/t20s_json` (11,264 files, ends 2026-04-16)."
    )
    lines.append("")
    lines.append("### audit summary")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(summary, indent=2))
    lines.append("```")
    NOTES_CONSUMERS.write_text("\n".join(lines) + "\n")
    print(f"[5.3] wrote {_rel(NOTES_CONSUMERS)}")


def eligible_stems() -> list[str]:
    with AUDIT_CSV.open() as handle:
        return sorted(
            row["stem"]
            for row in csv.DictReader(handle)
            if row["reason"] == "eligible"
        )


def stems_by_reason(reason: str) -> list[str]:
    """Ids the RECORDED audit assigned to ``reason`` (no closed-file read).

    Every exclusion the audit made is already written down in
    ``eligibility_audit.csv``, so the later steps read it from there instead of
    re-opening the sealed forward manifest or the golden odds file (Astra
    SHOULD 2). ``forward_fixture`` and ``golden_fixture`` here are already
    intersected with the cohort window, because only window matches are rows.
    """
    with AUDIT_CSV.open() as handle:
        return sorted(
            row["stem"]
            for row in csv.DictReader(handle)
            if row["reason"] == reason
        )


def recorded_id_read_counts() -> dict:
    """How many ids each permitted helper returned, from what is written down.

    Preference order: the audit's own summary sidecar (written by the step
    that made the two reads), then an existing ``FROZEN.json``. Never a fresh
    read of a closed file — the freeze records the counts of the reads the
    AUDIT made, and re-opening the sealed trees to restate them is exactly
    what SHOULD 2 removes.
    """
    if AUDIT_SUMMARY_JSON.is_file():
        summary = json.loads(AUDIT_SUMMARY_JSON.read_text())
        return {"forward": int(summary["forward_ids_read"]),
                "golden": int(summary["golden_ids_read"]),
                "source": _rel(AUDIT_SUMMARY_JSON)}
    if FROZEN_JSON.is_file():
        reads = json.loads(FROZEN_JSON.read_text())[
            "permitted_id_only_reads"]["reads"]
        by_helper = {row["helper"]: int(row["ids_returned"]) for row in reads}
        return {"forward": by_helper["read_forward_fixture_ids"],
                "golden": by_helper["read_golden_fixture_ids"],
                "source": _rel(FROZEN_JSON)}
    raise RuntimeError(
        f"neither {_rel(AUDIT_SUMMARY_JSON)} nor {_rel(FROZEN_JSON)} exists, "
        "so the permitted id-only read counts are not recorded anywhere: run "
        "--step audit first (it makes the only two closed-file reads)"
    )


# Set by --allow-refreeze. The cohort is opened exactly once (D5.9), so once
# FROZEN.json exists every mutating step refuses by default rather than
# rebuilding the corpus, the audit, the state, the parquet or the freeze
# underneath evidence that already cites their hashes (Astra SHOULD 2).
ALLOW_REFREEZE = False

MUTATING_STEPS = ("extract", "audit", "state", "materialize", "parity",
                  "context", "freeze")


def _guard_frozen(step: str) -> None:
    if not FROZEN_JSON.is_file() or ALLOW_REFREEZE:
        return
    raise RuntimeError(
        f"step {step!r} would write into a FROZEN cohort: "
        f"{_rel(FROZEN_JSON)} exists. The cohort is frozen with hashes and is "
        "opened exactly once (D5.7, D5.9), so rebuilding it silently would "
        "replace evidence other files already cite. Pass --allow-refreeze if "
        "a deliberate, recorded rebuild is really intended."
    )


# --------------------------------------------------------------------------
# 5.4 sidecar state
# --------------------------------------------------------------------------

def build_corpus_symlinks() -> int:
    """Merged corpus dir: repo corpus + all 471 raw matches as context.

    The materializer's corpus-membership guard compares the cache's
    ``source_dirs_json`` with the single directory being walked, so state and
    materialisation must both see one directory. Symlinks keep the repo
    corpus and `raw_json/` byte-identical and untouched.
    """
    if CORPUS_DIR.exists():
        for stale in CORPUS_DIR.iterdir():
            stale.unlink()
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for source in (REPO_CORPUS, RAW_JSON_DIR):
        for path in sorted(source.glob("*.json")):
            target = CORPUS_DIR / path.name
            if target.exists() or target.is_symlink():
                raise RuntimeError(
                    f"stem collision between corpora: {path.name}"
                )
            target.symlink_to(path.resolve())
            n += 1
    print(f"[5.4] merged corpus: {n} json symlinks -> "
          f"{_rel(CORPUS_DIR)}")
    return n


def step_state() -> dict:
    _guard_frozen("state")
    from build_stats_cache import build as build_cache, freeze_priors_from_sqlite
    from build_stats_cache import PRIOR_META_KEYS

    n_corpus = build_corpus_symlinks()
    t0 = time.time()
    build_cache(
        [CORPUS_DIR],
        SIDECAR_SQLITE,
        gender=I7_GENDER,
        metadata_csv=METADATA_CSV,
        delivery_semantics=I7_DELIVERY_SEMANTICS,
        schema_version=4,
    )
    build_seconds = time.time() - t0
    provenance = freeze_priors_from_sqlite(SIDECAR_SQLITE, PROD_I7_SQLITE)

    prod = _meta_of(PROD_I7_SQLITE)
    side = _meta_of(SIDECAR_SQLITE)
    mismatched = [
        key for key in PRIOR_META_KEYS if prod.get(key) != side.get(key)
    ]
    if mismatched:
        raise RuntimeError(f"frozen priors differ from production: {mismatched}")
    for key in ("same_day_order_version", "venue_alias_version",
                "venue_alias_sha256", "delivery_semantics",
                "gender_filter", "schema_version", "features"):
        if prod.get(key) != side.get(key):
            raise RuntimeError(
                f"sidecar/production _meta mismatch on {key}: "
                f"{side.get(key)!r} vs {prod.get(key)!r}"
            )
    report = {
        "corpus_json_count": n_corpus,
        "build_seconds": round(build_seconds, 1),
        "sidecar_sqlite": _rel(SIDECAR_SQLITE),
        "sidecar_md5": md5_file(SIDECAR_SQLITE),
        "prior_contract": provenance["prior_contract"],
        "prior_source_sqlite": provenance["prior_source_sqlite"],
        "prior_source_sha256": provenance["prior_source_sha256"],
        "frozen_prior_keys": len(PRIOR_META_KEYS),
        "source_match_count": side.get("source_match_count"),
        "num_dates": side.get("num_dates"),
        "num_players": side.get("num_players"),
        "num_venues": side.get("num_venues"),
    }
    print(f"[5.4] {json.dumps(report, indent=2)}")
    (STATE_DIR / "state_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    return report


def _meta_of(path: Path) -> dict[str, str]:
    conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        return dict(conn.execute("SELECT key, value FROM _meta"))
    finally:
        conn.close()


# --------------------------------------------------------------------------
# 5.6 materialisation
# --------------------------------------------------------------------------

def step_materialize() -> dict:
    _guard_frozen("materialize")
    import pandas as pd

    from materialize_features import materialize

    frame_hash = json.loads(FRAME_FEATURE_HASH.read_text())
    t0 = time.time()
    n_matches, counts = materialize(
        source_dir=CORPUS_DIR,
        sqlite_dir=STATE_DIR,
        out_dir=FULL_OUT_DIR,
        version=I7_VERSION,
        splits=I7_SPLITS,
        gender=I7_GENDER,
        metadata_csv=METADATA_CSV,
        feature_hash_info=dict(frame_hash),
        k_player=I7_K_PLAYER,
        k_venue=I7_K_VENUE,
        delivery_semantics=I7_DELIVERY_SEMANTICS,
    )
    seconds = time.time() - t0

    golden = FULL_OUT_DIR / f"cricket_data_{I7_VERSION}_golden_test.parquet"
    df = pd.read_parquet(golden)
    keep = set(eligible_stems())
    ids = df["innings_id"].map(match_id_from_innings_id)
    cohort = df.loc[ids.isin(keep)].reset_index(drop=True)
    cohort.to_parquet(COHORT_PARQUET, index=False)
    COHORT_FEATURE_HASH.write_text(json.dumps(frame_hash))

    written_hash = json.loads(
        (FULL_OUT_DIR / ".feature_hash").read_text()
    )
    if written_hash != frame_hash:
        raise RuntimeError("materialized .feature_hash != frame .feature_hash")

    report = {
        "materialize_seconds": round(seconds, 1),
        "matches_walked": n_matches,
        "split_row_counts": counts,
        "golden_rows_all_471": int(len(df)),
        "cohort_rows": int(len(cohort)),
        "cohort_matches": int(ids[ids.isin(keep)].nunique()),
        "cohort_columns": int(cohort.shape[1]),
        "feature_hash": frame_hash,
        "cohort_parquet_sha256": sha256_file(COHORT_PARQUET),
    }
    print(f"[5.6] {json.dumps(report, indent=2)}")
    (OUT_DIR / "materialize_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    return report


def _canonical(value):
    """Hashable, container-agnostic form of a parquet cell value."""
    if isinstance(value, (list, tuple)):
        return tuple(_canonical(v) for v in value)
    if hasattr(value, "tolist") and getattr(value, "ndim", 0) > 0:
        return tuple(_canonical(v) for v in value.tolist())
    return value


def step_parity(full_split: bool = False) -> dict:
    """5.6 — 20 i7 test matches rebuilt through the sidecar path, bit-for-bit.

    ``full_split`` widens the comparison from the registered 20-match sample
    to every match in the i7 test split (stronger, same contract).
    """
    _guard_frozen("parity")
    import numpy as np
    import pandas as pd

    frame = pd.read_parquet(FRAME_TEST_PARQUET)
    rebuilt = pd.read_parquet(
        FULL_OUT_DIR / f"cricket_data_{I7_VERSION}_test.parquet"
    )
    frame_ids = frame["innings_id"].map(match_id_from_innings_id)
    rebuilt_ids = rebuilt["innings_id"].map(match_id_from_innings_id)

    rng = np.random.default_rng(PARITY_SEED)
    candidates = sorted(frame_ids.unique())
    picked = sorted(
        rng.choice(candidates, size=PARITY_N_MATCHES, replace=False).tolist()
    )

    if full_split:
        picked = candidates
    left = frame.loc[frame_ids.isin(picked)].reset_index(drop=True)
    right = rebuilt.loc[rebuilt_ids.isin(picked)].reset_index(drop=True)

    from feature_registry import V3_GROUPS, resolve_feature_list

    registry_features = set(resolve_feature_list(V3_GROUPS))
    frame_only = sorted(set(left.columns) - set(right.columns))
    rebuilt_only = sorted(set(right.columns) - set(left.columns))
    # The rebuilt frame may carry evaluation-only metadata columns added to
    # parsing_v2.py after the i7 frame was materialised (I9 exposure
    # counts). They are deliberately outside the feature registry, so they
    # cannot change the 114-feature model matrix; anything else is a defect.
    unexpected_extras = sorted(set(rebuilt_only) & registry_features)

    mismatches: dict[str, dict] = {}
    if frame_only or unexpected_extras:
        mismatches["__columns__"] = {
            "frame_only": frame_only,
            "rebuilt_only_in_feature_registry": unexpected_extras,
        }
    if len(left) != len(right):
        mismatches["__rows__"] = {"frame": len(left), "rebuilt": len(right)}
    if not mismatches:
        for col in left.columns:
            a, b = left[col], right[col]
            if a.dtype != b.dtype:
                mismatches[col] = {"dtype": [str(a.dtype), str(b.dtype)]}
                continue
            if a.dtype == object:
                # Some columns hold per-ball lists (extras_types,
                # wicket_kinds, dismissed_batter_ids); compare canonicalised
                # values so a list/ndarray container difference is not read
                # as a content difference.
                left_vals = [_canonical(v) for v in a.tolist()]
                right_vals = [_canonical(v) for v in b.tolist()]
                equal = np.array(
                    [x == y for x, y in zip(left_vals, right_vals)],
                    dtype=bool,
                )
            else:
                equal = (a.values == b.values) | (
                    a.isna().values & b.isna().values
                )
            n_bad = int((~equal).sum())
            if n_bad:
                idx = int(np.flatnonzero(~equal)[0])
                mismatches[col] = {
                    "n_differing_rows": n_bad,
                    "first_row": idx,
                    "frame": str(a.iloc[idx]),
                    "rebuilt": str(b.iloc[idx]),
                }

    report = {
        "scope": "full_test_split" if full_split else "sample",
        "n_matches": len(picked),
        "seed": PARITY_SEED,
        "match_ids": list(picked) if not full_split else "all",
        "rows_compared": int(len(left)),
        "columns_compared": int(len(left.columns)),
        "frame_columns": int(len(left.columns)),
        "rebuilt_columns": int(len(right.columns)),
        "non_feature_metadata_only_in_rebuild": rebuilt_only,
        "feature_columns_in_registry": len(registry_features),
        "full_test_split_rows": {
            "frame": int(len(frame)), "rebuilt": int(len(rebuilt))
        },
        "mismatched_columns": mismatches,
        "pass": not mismatches,
    }
    out = (
        PARITY_JSON.with_name("parity_test_split_full.json")
        if full_split else PARITY_JSON
    )
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(
        f"[5.6 parity] {len(picked)} matches, {report['rows_compared']:,} rows, "
        f"{report['columns_compared']} columns, "
        f"pass={report['pass']} "
        f"({len(mismatches)} mismatched columns)"
    )
    if not report["pass"]:
        raise RuntimeError(f"parity failed: {sorted(mismatches)}")
    return report


# --------------------------------------------------------------------------
# 5.5 context vs scoring
# --------------------------------------------------------------------------

def step_context() -> dict:
    _guard_frozen("context")
    import pandas as pd

    cohort = pd.read_parquet(COHORT_PARQUET, columns=[
        "innings_id", "match_date", "batter_id", "batsman_avg",
        "batter_balls_faced",
    ])
    scored = set(cohort["innings_id"].map(match_id_from_innings_id))
    eligible = set(eligible_stems())
    if scored != eligible:
        raise RuntimeError(
            "scored ids != eligible ids: "
            f"+{sorted(scored - eligible)[:10]} "
            f"-{sorted(eligible - scored)[:10]}"
        )

    manifest = json.loads(RAW_MANIFEST.read_text())
    # The forward exclusions are already recorded in eligibility_audit.csv by
    # the audit step, intersected with the cohort window by construction, so
    # this step does NOT re-open the sealed forward manifest (Astra SHOULD 2).
    forward_ids = set(stems_by_reason("forward_fixture"))

    case = _find_context_case(forward_ids, eligible, manifest)
    report = {
        "scored_matches": len(scored),
        "scored_equals_eligible": True,
        "hand_verified_case": case,
    }
    _write_context_notes(report)
    print(f"[5.5] scored matches == eligible set ({len(scored)})")
    print(f"[5.5] {json.dumps(case, indent=2)}")
    return report


def _player_batting_asof(
    path: Path, player: str, date: str
) -> tuple[int, int] | None:
    """Cumulative (runs, balls) for ``player`` strictly before ``date``."""
    conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        row = conn.execute(
            """
            SELECT b.runs, b.balls
            FROM batting b
            JOIN players p ON p.id = b.player_id
            JOIN dates d ON d.id = b.date_id
            WHERE p.player_id = ? AND d.date < ?
            ORDER BY d.date DESC LIMIT 1
            """,
            (player, date),
        ).fetchone()
    finally:
        conn.close()
    return (int(row[0]), int(row[1])) if row else None


def _batting_in_match(stem: str, player: str) -> dict:
    """Raw batting contribution of ``player`` (a Cricsheet registry id).

    Both run conventions are reported because the tracker's own convention
    is a versioned contract (``inclusive_total_runs_v1``); ``balls`` counts
    legal deliveries only, matching ``PlayerStatsTracker``.
    """
    data = json.loads((RAW_JSON_DIR / f"{stem}.json").read_text())
    registry = (data["info"].get("registry") or {}).get("people", {}) or {}
    runs_batter = runs_total = balls = 0
    for innings in data.get("innings", []):
        for over in innings.get("overs", []):
            for delivery in over.get("deliveries", []):
                if registry.get(delivery.get("batter")) != player:
                    continue
                extras = delivery.get("extras", {}) or {}
                runs_batter += int(delivery["runs"].get("batter", 0))
                runs_total += int(delivery["runs"].get("total", 0))
                if "wides" not in extras and "noballs" not in extras:
                    balls += 1
    return {
        "runs_batter_only": runs_batter,
        "runs_total_inclusive": runs_total,
        "balls_legal": balls,
    }


def _match_players(stem: str) -> set[str]:
    data = json.loads((RAW_JSON_DIR / f"{stem}.json").read_text())
    registry = (data["info"].get("registry") or {}).get("people", {}) or {}
    out: set[str] = set()
    for names in (data["info"].get("players") or {}).values():
        for name in names:
            pid = registry.get(name)
            if pid:
                out.add(pid)
    return out


def _find_context_case(
    forward_ids: set, eligible: set, manifest: dict
) -> dict:
    """Hand-verifiable case: an eligible match whose as-of state contains an
    excluded (sealed-forward) match.

    Picks a player whose appearances in the cohort window contain a
    consecutive pair (F, E) where F is an excluded forward fixture, E is an
    eligible fixture, F's date is the player's only appearance that day, and
    E is strictly later. Then the sidecar's cumulative batting delta between
    "strictly before F" and "strictly before E" is F's contribution alone.
    """
    dates = {s: manifest["matches"][s]["date"] for s in manifest["matches"]}
    players = {s: _match_players(s) for s in sorted(manifest["matches"])}
    by_player = {}
    for stem in sorted(manifest["matches"], key=lambda s: (dates[s], s)):
        for pid in players[stem]:
            by_player.setdefault(pid, []).append(stem)

    fallback = None
    exact_cases = []
    for pid, appearances in sorted(by_player.items()):
        for first, second in zip(appearances, appearances[1:]):
            if first not in forward_ids or second not in eligible:
                continue
            if dates[second] <= dates[first]:
                continue
            same_day = [s for s in appearances if dates[s] == dates[first]]
            if len(same_day) != 1:
                continue
            contribution = _batting_in_match(first, pid)
            if contribution["balls_legal"] == 0:
                continue
            before = _player_batting_asof(SIDECAR_SQLITE, pid, dates[first])
            after = _player_batting_asof(SIDECAR_SQLITE, pid, dates[second])
            if before is None or after is None:
                continue
            delta = {
                "runs": after[0] - before[0],
                "balls": after[1] - before[1],
            }
            prod = _player_batting_asof(PROD_I7_SQLITE, pid, dates[second])
            exact = (
                delta["balls"] == contribution["balls_legal"]
                and delta["runs"] in (
                    contribution["runs_batter_only"],
                    contribution["runs_total_inclusive"],
                )
            )
            case = {
                "player_id": pid,
                "excluded_forward_match": first,
                "excluded_match_date": dates[first],
                "eligible_match": second,
                "eligible_match_date": dates[second],
                "player_appearances_in_cohort_window": len(appearances),
                "player_batting_in_excluded_match": contribution,
                "sidecar_asof_before_excluded_match": {
                    "runs": before[0], "balls": before[1]
                },
                "sidecar_asof_before_eligible_match": {
                    "runs": after[0], "balls": after[1]
                },
                "delta_attributable_to_excluded_match": delta,
                "delta_matches_excluded_match_contribution": exact,
                "production_i7_cache_asof_before_eligible_match": (
                    None if prod is None
                    else {"runs": prod[0], "balls": prod[1],
                          "note": "production cache ends 2026-04-16"}
                ),
                "note": (
                    "The excluded (sealed-forward) fixture is this player's "
                    "immediately preceding appearance, and it is his only "
                    "appearance on that date, so the entire cumulative "
                    "batting delta the sidecar shows between the two "
                    "fixtures is that excluded match. It is therefore "
                    "present in the state as chronological context while "
                    "being absent from the 210 scored matches."
                ),
            }
            if exact:
                exact_cases.append(case)
            elif fallback is None:
                fallback = case
    if exact_cases:
        # Report the most substantial case: the largest batting contribution
        # in the excluded match, so the delta is unmistakable.
        return max(
            exact_cases,
            key=lambda c: (
                c["player_batting_in_excluded_match"]["balls_legal"],
                c["player_batting_in_excluded_match"]["runs_batter_only"],
                c["player_id"],
            ),
        )
    if fallback is not None:
        fallback["note"] += (
            " NOTE: the run delta does not equal either raw run convention "
            "exactly (the tracker's run accounting is the versioned "
            "inclusive_total_runs_v1 contract); the balls delta is the "
            "unambiguous evidence."
        )
        return fallback
    raise RuntimeError("no context case found")


def _write_context_notes(report: dict) -> None:
    case = report["hand_verified_case"]
    lines = [
        "# D5.5 — context vs scoring",
        "",
        f"Recorded {utc_now()} by `build_cohort_stage2.py`.",
        "",
        f"- scored matches in `cricket_data_i7_cohort.parquet`: "
        f"{report['scored_matches']}",
        "- scored match ids equal the eligible set exactly (asserted).",
        "- every one of the 471 raw matches, eligible or excluded, is in the "
        "sidecar state as chronological context.",
        "",
        "## Hand-verified case",
        "",
        "```json",
        json.dumps(case, indent=2),
        "```",
    ]
    NOTES_CONTEXT.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# 5.7 freeze
# --------------------------------------------------------------------------

def step_freeze() -> dict:
    _guard_frozen("freeze")
    import pandas as pd

    ids = eligible_stems()
    recorded_counts = recorded_id_read_counts()
    rows = int(len(pd.read_parquet(COHORT_PARQUET, columns=["innings_id"])))
    frozen = {
        "frozen_at_utc": utc_now(),
        "spec": "docs/sequence_track/stage2_acceptance.md D5",
        "window": {"start": WINDOW_START, "end": WINDOW_END},
        "raw_match_count": EXPECTED_RAW_COUNT,
        "eligible_match_count": len(ids),
        "eligible_match_ids": ids,
        "row_count": rows,
        "parquet": _rel(COHORT_PARQUET),
        "parquet_sha256": sha256_file(COHORT_PARQUET),
        "feature_hash": json.loads(COHORT_FEATURE_HASH.read_text()),
        "sidecar_cache": _rel(SIDECAR_SQLITE),
        "sidecar_cache_md5": md5_file(SIDECAR_SQLITE),
        "raw_manifest_sha256": sha256_file(RAW_MANIFEST),
        "builder_script": _rel(Path(__file__)),
        "builder_script_sha256": sha256_file(Path(__file__).resolve()),
        "eligibility_audit_sha256": sha256_file(AUDIT_CSV),
        "permitted_id_only_reads": {
            "permission": {
                "quote": USER_PERMISSION_QUOTE,
                "date": USER_PERMISSION_DATE,
                "recorded_in": (
                    "docs/sequence_track/stage2_acceptance.md D2.3"
                ),
            },
            # Quoted from what the AUDIT recorded, not re-read here: the two
            # closed files are opened once each, by step_audit, and every
            # later step consumes the written-down result (Astra SHOULD 2).
            "reads_recorded_from": recorded_counts["source"],
            "reads": [
                {
                    "path": _rel(FORWARD_MANIFEST),
                    "helper": "read_forward_fixture_ids",
                    "fields_read": ["matches[].match_id",
                                    "matches[].cricsheet_id"],
                    "ids_returned": recorded_counts["forward"],
                },
                {
                    "path": _rel(GOLDEN_ODDS),
                    "helper": "read_golden_fixture_ids",
                    "fields_read": ["matches[].match_id",
                                    "matches[].cricsheet_id"],
                    "ids_returned": recorded_counts["golden"],
                },
            ],
        },
    }
    FROZEN_JSON.write_text(json.dumps(frozen, indent=2, sort_keys=True))
    print(
        f"[5.7] froze {len(ids)} eligible matches / {rows:,} rows -> "
        f"{_rel(FROZEN_JSON)}"
    )
    return frozen


# --------------------------------------------------------------------------

STEPS = ("extract", "audit", "state", "materialize", "parity", "context",
         "freeze")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--step", choices=STEPS + ("all",), default=None, action="append",
        dest="steps",
    )
    ap.add_argument(
        "--parity-full-split", action="store_true",
        help="Also compare every match in the i7 test split, not just 20.",
    )
    ap.add_argument(
        "--allow-count-mismatch", action="store_true",
        help="Do not refuse when the 5.1 extraction count is not 471.",
    )
    ap.add_argument(
        "--allow-refreeze", action="store_true",
        help="Permit a mutating step even though FROZEN.json exists. Every "
             "step refuses by default once the cohort is frozen: the freeze "
             "is cited by hash elsewhere and the cohort is opened exactly "
             "once (D5.7, D5.9).",
    )
    args = ap.parse_args()
    global ALLOW_REFREEZE
    ALLOW_REFREEZE = bool(args.allow_refreeze)
    if ALLOW_REFREEZE and FROZEN_JSON.is_file():
        print(f"WARNING: --allow-refreeze given and {_rel(FROZEN_JSON)} "
              "exists; this rebuild REPLACES a frozen cohort whose hashes "
              "other files cite.")
    steps = args.steps or ["all"]
    if "all" in steps:
        steps = list(STEPS)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for step in STEPS:
        if step not in steps:
            continue
        ts = time.time()
        if step == "extract":
            step_extract(strict=not args.allow_count_mismatch)
        elif step == "audit":
            step_audit()
        elif step == "state":
            step_state()
        elif step == "materialize":
            step_materialize()
        elif step == "parity":
            step_parity()
            if args.parity_full_split:
                step_parity(full_split=True)
        elif step == "context":
            step_context()
        elif step == "freeze":
            step_freeze()
        print(f"--- step {step}: {time.time() - ts:.0f}s", flush=True)
    print(f"TOTAL {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
