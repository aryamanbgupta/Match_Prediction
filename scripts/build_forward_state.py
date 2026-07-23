#!/usr/bin/env python3
"""Build deterministic, non-production feature state for a sealed holdout.

The output is a sidecar under ``data/forward_state``. It never overwrites the
production SQLite cache, tracker snapshot, model artifacts, or sealed holdout.
No model is imported or scored.

Example:

    uv run python scripts/build_forward_state.py \
      --holdout-dir data/forward_holdout/2026-06-01_2026-07-13
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

from build_stats_cache import (  # noqa: E402
    build as build_stats_cache,
    freeze_priors_from_sqlite,
)
from loaders_common import (  # noqa: E402
    DEFAULT_SPLITS,
    SAME_DAY_ORDER_VERSION,
)
from materialize_match_features import materialize  # noqa: E402
from verify_forward_holdout import verify as verify_holdout  # noqa: E402

DEFAULT_HOLDOUT = (
    ROOT / "data" / "forward_holdout" / "2026-06-01_2026-07-13"
)
DEFAULT_BASE = ROOT / "data" / "t20s_json"
DEFAULT_METADATA = ROOT / "data" / "all_players_enriched.csv"
DEFAULT_PRIOR_SOURCE = ROOT / "models" / "player_stats_cache_v3.sqlite"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(128 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_match_index(directory: Path) -> tuple[dict[str, str], dict]:
    records: dict[str, str] = {}
    by_date: dict[str, list[str]] = defaultdict(list)
    legacy_by_date: dict[str, list[str]] = defaultdict(list)
    rejected = Counter()
    for path in directory.glob("*.json"):
        try:
            data = json.loads(path.read_text())
            info = data["info"]
            date = info["dates"][0]
        except (OSError, json.JSONDecodeError, KeyError, IndexError):
            rejected["invalid_json_or_metadata"] += 1
            continue
        if info.get("match_type") != "T20":
            rejected["non_t20"] += 1
            continue
        if info.get("gender", "male") != "male":
            rejected["non_male"] += 1
            continue
        if path.stem in records:
            raise RuntimeError(f"duplicate match ID within {directory}: {path.stem}")
        records[path.stem] = date
        by_date[date].append(path.stem)
        legacy_by_date[date].append(path.stem)

    same_day_groups = {
        date: sorted(ids) for date, ids in by_date.items() if len(ids) > 1
    }
    legacy_order_diff_groups = sum(
        legacy_by_date[date] != sorted(ids)
        for date, ids in same_day_groups.items()
    )
    return records, {
        "directory": str(directory.resolve()),
        "accepted_matches": len(records),
        "date_min": min(records.values()) if records else None,
        "date_max": max(records.values()) if records else None,
        "same_day_groups": len(same_day_groups),
        "matches_in_same_day_groups": sum(
            len(ids) for ids in same_day_groups.values()
        ),
        "legacy_filesystem_order_diff_groups": legacy_order_diff_groups,
        "rejections": dict(sorted(rejected.items())),
    }


def inspect_sources(
    holdout_dir: Path,
    base_dir: Path,
) -> tuple[dict, dict, dict, dict]:
    holdout_report = verify_holdout(holdout_dir)
    context_dir = holdout_dir / "context_t20s_json"
    base_index, base_report = load_match_index(base_dir)
    context_index, context_report = load_match_index(context_dir)
    if not base_index or not context_index:
        raise RuntimeError("base and context sources must both be non-empty")

    overlap = sorted(set(base_index) & set(context_index))
    if overlap:
        raise RuntimeError(
            f"base/context Cricsheet ID overlap: {overlap[:20]}"
        )
    base_max = max(base_index.values())
    context_min = min(context_index.values())
    if not base_max < context_min:
        raise RuntimeError(
            "context is not a strict chronological continuation: "
            f"base_max={base_max}, context_min={context_min}"
        )

    holdout_manifest = json.loads((holdout_dir / "manifest.json").read_text())
    selected_ids = {
        str(row["cricsheet_id"]) for row in holdout_manifest["matches"]
    }
    missing_selected = sorted(selected_ids - set(context_index))
    if missing_selected:
        raise RuntimeError(
            f"selected holdout IDs missing from context: {missing_selected[:20]}"
        )
    return holdout_report, base_report, context_report, {
        "base_context_overlap": 0,
        "base_date_max": base_max,
        "context_date_min": context_min,
        "selected_ids": selected_ids,
        "context_dir": context_dir,
    }


def build(args: argparse.Namespace) -> dict:
    holdout_dir = args.holdout_dir.resolve()
    base_dir = args.base_dir.resolve()
    metadata_csv = args.metadata_csv.resolve()
    prior_source_sqlite = args.prior_source_sqlite.resolve()
    if not metadata_csv.is_file():
        raise FileNotFoundError(metadata_csv)
    if not prior_source_sqlite.is_file():
        raise FileNotFoundError(prior_source_sqlite)

    holdout_report, base_report, context_report, source_guard = inspect_sources(
        holdout_dir,
        base_dir,
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (
            ROOT
            / "data"
            / "forward_state"
            / holdout_dir.name
        ).resolve()
    )
    plan = {
        "status": "READY",
        "model_scoring_performed": False,
        "same_day_order_version": SAME_DAY_ORDER_VERSION,
        "prior_source_sqlite": str(prior_source_sqlite),
        "prior_source_sha256": sha256_file(prior_source_sqlite),
        "holdout": holdout_report,
        "base_source": base_report,
        "context_source": context_report,
        "base_context_overlap": source_guard["base_context_overlap"],
        "output_dir": str(output_dir),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return plan

    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} already exists; forward state builds are immutable"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir()

    source_dirs = [base_dir, source_guard["context_dir"]]
    sqlite_path = staging / "player_stats_cache_v3.sqlite"
    feature_dir = staging / "match_features"
    try:
        if args.reuse_cache:
            reusable = args.reuse_cache.resolve()
            if not reusable.is_file():
                raise FileNotFoundError(reusable)
            shutil.copy2(reusable, sqlite_path)
        else:
            build_stats_cache(
                source_dirs,
                sqlite_path,
                gender="male",
                metadata_csv=metadata_csv,
            )

        prior_provenance = freeze_priors_from_sqlite(
            sqlite_path,
            prior_source_sqlite,
        )
        conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
        meta = dict(conn.execute("SELECT key, value FROM _meta"))
        conn.close()
        expected_source_dirs = json.dumps(
            [str(path.resolve()) for path in source_dirs],
            separators=(",", ":"),
        )
        if meta.get("same_day_order_version") != SAME_DAY_ORDER_VERSION:
            raise RuntimeError("sidecar SQLite ordering metadata mismatch")
        if meta.get("prior_contract") != "frozen_external_sqlite_v1":
            raise RuntimeError("sidecar SQLite prior contract mismatch")
        if (
            meta.get("prior_source_sha256")
            != prior_provenance["prior_source_sha256"]
        ):
            raise RuntimeError("sidecar SQLite prior source hash mismatch")
        if meta.get("source_dirs_json") != expected_source_dirs:
            raise RuntimeError("sidecar SQLite source directories mismatch")
        expected_source_count = (
            base_report["accepted_matches"]
            + context_report["accepted_matches"]
        )
        if int(meta.get("source_match_count", -1)) != expected_source_count:
            raise RuntimeError(
                "sidecar SQLite match count differs from accepted sources"
            )
        source_files = [
            path
            for source_dir in source_dirs
            for path in source_dir.glob("*.json")
        ]
        if int(meta.get("source_json_file_count", -1)) != len(source_files):
            raise RuntimeError(
                "sidecar SQLite JSON file count differs from live sources"
            )
        live_mtime_max = max(
            (path.stat().st_mtime for path in source_files),
            default=0.0,
        )
        if float(meta.get("source_json_mtime_max", 0)) + 1 < live_mtime_max:
            raise RuntimeError("sidecar SQLite is stale versus live sources")

        n_materialized, split_counts = materialize(
            source_dir=source_dirs,
            sqlite_dir=staging,
            out_dir=feature_dir,
            version="v3",
            splits=dict(DEFAULT_SPLITS),
            gender="male",
            metadata_csv=metadata_csv,
        )

        golden_path = feature_dir / "golden_test.parquet"
        if not golden_path.is_file():
            raise RuntimeError("match materializer did not write golden_test.parquet")
        golden = pd.read_parquet(
            golden_path,
            columns=["cricsheet_id", "match_date"],
        )
        materialized_ids = set(golden["cricsheet_id"].astype(str))
        missing_features = sorted(
            source_guard["selected_ids"] - materialized_ids
        )
        if missing_features:
            raise RuntimeError(
                "selected holdout IDs missing materialized features: "
                f"{missing_features[:20]}"
            )

        artifacts = []
        for path in sorted(
            item for item in staging.rglob("*") if item.is_file()
        ):
            artifacts.append(
                {
                    "path": path.relative_to(staging).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        manifest = {
            "schema_version": 1,
            "purpose": (
                "deterministic chronological feature state for sealed "
                "forward evaluation; never model fitting"
            ),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_scoring_performed": False,
            "same_day_order_version": SAME_DAY_ORDER_VERSION,
            "prior_contract": prior_provenance,
            "holdout_directory": str(holdout_dir),
            "holdout_dataset_fingerprint_sha256": holdout_report[
                "dataset_fingerprint_sha256"
            ],
            "source_reports": {
                "base": base_report,
                "context": context_report,
            },
            "sqlite_meta": meta,
            "materialized_matches": n_materialized,
            "split_counts": split_counts,
            "selected_holdout_feature_rows_verified": len(
                source_guard["selected_ids"]
            ),
            "artifacts": artifacts,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )
        (staging / "NO_MODEL_SCORING").write_text(
            "This sidecar contains deterministic cache and feature state only.\n"
            "No model was loaded or scored during construction.\n"
        )
        staging.replace(output_dir)
    except Exception:
        # Retain staging for forensic inspection, never publish partial state.
        raise

    print(json.dumps(manifest, indent=2))
    print(f"Forward state written to {output_dir}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--holdout-dir",
        type=Path,
        default=DEFAULT_HOLDOUT,
    )
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=DEFAULT_METADATA,
    )
    parser.add_argument(
        "--prior-source-sqlite",
        type=Path,
        default=DEFAULT_PRIOR_SOURCE,
        help=(
            "Pre-holdout cache whose global/phase priors are frozen into "
            "the forward sidecar. Defaults to the production April-16 cache."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--reuse-cache",
        type=Path,
        default=None,
        help=(
            "Copy and verify an already-built deterministic SQLite cache "
            "before rematerializing features. Intended only for recovery "
            "from a downstream feature-build failure."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
