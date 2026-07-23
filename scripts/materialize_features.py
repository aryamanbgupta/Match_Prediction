"""Materialize per-ball feature parquet from SQLite + JSON.

Phase B deliverable §3. Replaces the parquet-writing half of
`parsing_v2.py:process_folder_v2_with_splits` (lines 1328-1429).

Key architectural contract (proven by Phase A harness):

  * **Cross-date stateless**: every date's batch starts by rehydrating
    temp trackers from SQLite — no carryover from prior dates needed.
  * **Within-date stateful**: same-day matches share the temp trackers,
    which accumulate via `parse_match_data_v2`'s ball-by-ball updates
    + post-match venue updates. This reproduces monolith drift on
    same-day secondaries.

Splits come from the YAML `data.splits` block. If the block is absent,
we fall back to the hardcoded cutoffs at the original parsing_v2.py:
1210-1215, so existing configs keep working without edits.

Parallelism: ships serial. Per-date parallelism via ProcessPoolExecutor
is a follow-up — `temp_*` trackers are dates-wide, and SQLite mmap
supports N readers, so it's cheap to enable later.

Usage:
    uv run python scripts/materialize_features.py                      # default config
    uv run python scripts/materialize_features.py --config experiments/configs/xgb_v3_baseline.yaml
    uv run python scripts/materialize_features.py --out-dir /tmp/mat   # alternate output
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

try:
    import yaml
except ImportError:
    yaml = None

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from loaders_common import (
    DEFAULT_SPLITS,
    effective_splits,
    extract_match_metadata,
    iter_matches_chronological,
)
from parsing_v2 import parse_match_data_v2
from player_metadata import PlayerMetadataProvider
from stats_provider import StatsProvider
from tracker_rehydration import (
    extract_match_player_ids,
    rehydrate_elo_tracker,
    rehydrate_stats_tracker,
    rehydrate_venue_tracker,
)


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def classify_split(match_date: datetime, splits: dict) -> str:
    """Port of parsing_v2.py:1277-1284 — bucket a match_date into one
    of four named splits based on YAML cutoff dates."""
    train_end = _parse_date(splits.get("train_end", DEFAULT_SPLITS["train_end"]))
    val_end = _parse_date(splits.get("val_end", DEFAULT_SPLITS["val_end"]))
    test_end = _parse_date(splits.get("test_end", DEFAULT_SPLITS["test_end"]))
    golden_start = _parse_date(
        splits.get("golden_start", DEFAULT_SPLITS["golden_start"]))

    if match_date < train_end:
        return "train"
    if match_date < val_end:
        return "validation"
    if match_date < golden_start:
        return "test"
    return "golden_test"


def group_by_date(it):
    """Yield (match_date, list_of_entries) batches from an
    iter_matches_chronological-shaped iterator.

    Each entry is (match_id, json_text, data_dict, venue, k_factor).
    Consecutive same-date matches accumulate into one batch.
    """
    buf_date: Optional[datetime] = None
    buf: list = []

    for match_id, json_text, match_date in it:
        data = json.loads(json_text)
        meta = extract_match_metadata(data)
        entry = (match_id, json_text, data, meta["venue"], meta["k_factor"])

        if buf_date is None:
            buf_date, buf = match_date, [entry]
        elif match_date == buf_date:
            buf.append(entry)
        else:
            yield buf_date, buf
            buf_date, buf = match_date, [entry]

    if buf_date is not None:
        yield buf_date, buf


def materialize(
    source_dir: Path,
    sqlite_dir: Path,
    out_dir: Path,
    version: str,
    splits: dict,
    gender: str,
    metadata_csv: Path,
    feature_hash_info: Optional[dict] = None,
    k_player: float = 30.0,
    k_venue: float = 200.0,
) -> Tuple[int, dict]:
    """Walk the corpus per-date; for each date, rehydrate temp trackers
    from SQLite and replay same-day matches in deterministic match-ID order.

    Returns (n_matches, split_counts).
    """
    provider = StatsProvider(
        str(sqlite_dir),
        version=version,
        require_order_contract=True,
    )
    if provider.backend_name != "sqlite":
        raise RuntimeError(
            f"materialize_features requires SQLite backend; got "
            f"{provider.backend_name!r}"
        )
    metadata = PlayerMetadataProvider(str(metadata_csv))

    # Schema v4: empirical outcome prior π loaded from SQLite _meta.
    # parse_match_data_v2 receives it as `prior=` and emits the 42
    # distribution features; without it, those columns are zeros.
    # Trigger lookup-load before reading _prior so the attribute exists.
    provider._backend._ensure_conn()
    prior = provider._backend._prior
    # Phase 3: per-phase priors loaded from SQLite _meta. On pre-Phase-3
    # caches, _phase_priors collapses to {phase: π} for every phase, so
    # phase_p* features still emit (just with degenerate values).
    phase_priors = provider._backend._phase_priors

    split_rows: dict[str, List[dict]] = {
        "train": [], "validation": [], "test": [], "golden_test": [],
    }
    n_matches = 0
    t_start = time.time()

    for match_date, batch in group_by_date(
        iter_matches_chronological(source_dir, gender=gender)
    ):
        # Rehydrate once per date using the union of all same-day players
        # + venues; Phase A proved this is sufficient for bit-exact parity.
        union_pids: set = set()
        union_venues: set = set()
        for _, _, data, venue, _ in batch:
            union_pids.update(extract_match_player_ids(data))
            union_venues.add(venue)

        temp_stats = rehydrate_stats_tracker(provider, match_date, union_pids)
        temp_elo = rehydrate_elo_tracker(provider, match_date, union_pids)
        temp_venue = rehydrate_venue_tracker(
            provider, match_date, union_venues)

        for match_id, json_text, data, venue, k_factor in batch:
            rows, _it, vname, innings_details, chase_won = (
                parse_match_data_v2(
                    json_text, temp_stats, temp_venue, metadata,
                    elo_tracker=temp_elo, match_k_factor=k_factor,
                    prior=prior, phase_priors=phase_priors,
                    k_player=k_player, k_venue=k_venue,
                    match_ref=match_id,
                )
            )
            # Advance temp_venue so the NEXT same-day match sees it.
            # Monolith does this on live_venue at parsing_v2.py:1321-1325.
            for det in innings_details:
                temp_venue.update_venue_stats_detailed(vname, det)
            if chase_won is not None:
                temp_venue.update_venue_match_result(vname, chase_won)

            split = classify_split(match_date, splits)
            split_rows[split].extend(rows)
            n_matches += 1

        if n_matches % 500 == 0 and n_matches > 0:
            dt = time.time() - t_start
            print(f"  [{n_matches}] matches in {dt:.0f}s "
                  f"({n_matches / dt:.1f} match/s)", flush=True)

    # Write parquet per split, plus .feature_hash marker.
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split_name, rows in split_rows.items():
        counts[split_name] = len(rows)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        out_path = out_dir / f"cricket_data_{version}_{split_name}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  wrote {out_path} ({len(rows):,} rows, "
              f"{out_path.stat().st_size / 1e6:.1f} MB)")

    # feature_hash marker — format matches parsing_v2.py:1420-1429 exactly.
    if feature_hash_info is not None:
        with open(out_dir / ".feature_hash", "w") as fh:
            json.dump(feature_hash_info, fh)

    return n_matches, counts


def load_config(path: Optional[Path]) -> dict:
    """Load YAML config if provided; return {} otherwise. Missing PyYAML
    is a soft error — we fall back to CLI-driven defaults."""
    if path is None:
        return {}
    if yaml is None:
        print(f"warning: pyyaml not installed; ignoring --config {path}",
              file=sys.stderr)
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None,
                    help="Optional YAML (reads data.version / data.splits / "
                    "data.gender_filter).")
    ap.add_argument("--source-dir", type=Path,
                    default=Path("data/t20s_json"))
    ap.add_argument("--sqlite-dir", type=Path, default=Path("models"),
                    help="Directory containing player_stats_cache_v3.sqlite.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output parquet dir. Default: data/xgb_data_<version>.")
    ap.add_argument("--version", type=str, default=None,
                    help="Cache/parquet version (e.g. v3). Reads from "
                    "config.data.version if omitted.")
    ap.add_argument("--gender-filter", type=str, default=None,
                    help="Match gender filter. Reads from config if omitted; "
                    "defaults to 'male'.")
    ap.add_argument("--metadata-csv", type=Path,
                    default=Path("data/all_players_enriched.csv"))
    args = ap.parse_args()

    config = load_config(args.config)
    data_cfg = config.get("data", {})
    version = args.version or data_cfg.get("version", "v3")
    gender = (
        args.gender_filter
        if args.gender_filter is not None
        else data_cfg.get("gender_filter", "male")
    ) or None
    splits = data_cfg.get("splits", {})
    out_dir = args.out_dir or Path(
        "data/xgb_data" if version == "v2" else f"data/xgb_data_{version}"
    )
    # Phase 6: outcome-dist k overrides from YAML.
    od_cfg = config.get("outcome_dist", {}) if config else {}
    k_player = float(od_cfg.get("k_player", 30.0))
    k_venue  = float(od_cfg.get("k_venue", 200.0))
    if k_player != 30.0 or k_venue != 200.0:
        print(f"  outcome_dist overrides: k_player={k_player}, "
              f"k_venue={k_venue}")

    # Compute feature hash via feature_registry so the .feature_hash
    # marker matches parsing_v2.py's output for cache-hit detection.
    feature_hash_info = None
    try:
        from feature_registry import (
            get_feature_hash, resolve_feature_list, V3_GROUPS,
        )
        if config and "features" in config:
            feats = resolve_feature_list(
                config["features"].get("groups", V3_GROUPS),
                config["features"].get("exclude", []),
                config["features"].get("include_extra", []),
            )
        else:
            feats = resolve_feature_list(V3_GROUPS)
        feature_hash_info = {
            "hash": get_feature_hash(feats),
            "version": version,
            "n_features": len(feats),
            # Store the *effective* splits (YAML merged over DEFAULT_SPLITS)
            # so the cache identity matches the actual parquet content.
            # run_experiment.py's _check_parquet_cache must use the same
            # merge (via loaders_common.effective_splits) to compare.
            "splits": effective_splits(splits),
            # None → 'all' canonicalization: splits with no gender filter
            # still pick up all men's matches today because of corpus
            # composition, but the cache key must reflect intent.
            "gender_filter": gender if gender is not None else "all",
        }
    except ImportError:
        pass  # feature_registry unavailable — skip the marker

    t0 = time.time()
    # Bake k_player into the .feature_hash payload so the smart cache in
    # run_experiment.py invalidates the parquet on a k change. Without
    # this, a k-sweep config would inherit the previous run's parquet.
    if feature_hash_info is not None:
        feature_hash_info = dict(feature_hash_info)
        feature_hash_info["k_player"] = k_player
        feature_hash_info["k_venue"] = k_venue

    n_matches, counts = materialize(
        source_dir=args.source_dir,
        sqlite_dir=args.sqlite_dir,
        out_dir=out_dir,
        version=version,
        splits=splits,
        gender=gender,
        metadata_csv=args.metadata_csv,
        feature_hash_info=feature_hash_info,
        k_player=k_player,
        k_venue=k_venue,
    )
    dt = time.time() - t0

    print(f"\nDONE: {n_matches:,} matches → {out_dir} in {dt:.0f}s")
    for name, n in counts.items():
        print(f"  {name}: {n:,} balls")
    if feature_hash_info:
        print(f"  feature hash: {feature_hash_info['hash']} "
              f"({feature_hash_info['n_features']} features)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
