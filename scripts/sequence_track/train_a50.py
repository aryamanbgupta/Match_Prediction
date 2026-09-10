"""A50 — the 50-feature ball model (sequence track, stage 0, deliverable D4).

A50 is the production i7 ball model's training recipe run on the 50 columns
the T1 transformer sees, and nothing else: EB batter (18) + EB bowler (18) +
venue outcome distribution (6) + context (4) + state (4). Every other choice
is copied from `xgboost_v2.py` so the only difference against the 114-feature
production stack is the feature set:

  * frame        — the i7 ball frame (`cricket_data_i7_{train,validation,test}`)
  * label        — `ball_outcome`, -1 -> 7, then {0:0, 1:1, 2:2, 4:3, 6:4, 7:5}
  * class weights— none at all (the `--no-class-weights` semantics: the
                   `sample_weight` kwarg is omitted, not set to ones)
  * hyperparams  — read from the experiment YAML's `model.hyperparameters`
  * fit          — random_state 29, eval_metric mlogloss,
                   early_stopping_rounds 100, eval_set [(train), (validation)]
  * metrics      — sklearn `log_loss` / `accuracy_score` over `predict_proba`

The 50 column names are imported (never re-typed) from `embeddings_e1` and
`transformer_t1`, which is what makes the arm comparable to the T1 ablations.

Outputs live in one directory (default under the stage-1 embeddings
namespace) and nothing outside it is written. Serving sidecars are copied
byte-for-byte from the production ball model under BOTH the `_i7` and `_a50`
suffixes, so `sim_v1_2.XGBoostModelV2` — which derives its sidecar suffix
from the model filename — resolves them without any code change.

Usage:
    OMP_NUM_THREADS=8 uv run --no-sync python \\
        scripts/sequence_track/train_a50.py --n-jobs 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sklearn
import xgboost
import yaml
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from elo_update import (  # noqa: E402
    BASELINE_ELO_UPDATE_VERSION,
    assert_elo_update_version,
)
from embeddings_e1 import (  # noqa: E402
    CTX_COLS,
    EB_BAT_COLS,
    EB_BOWL_COLS,
    VENUE_COLS,
)
from identity_maps import (  # noqa: E402
    assert_venue_alias_contract,
    venue_alias_contract,
)
from transformer_t1 import STATE_COLS  # noqa: E402

ARM = "a50"
DATA_VERSION = "i7"
SPLITS = ("train", "validation", "test")
EXPECTED_ROWS = {"train": 1876971, "validation": 124292, "test": 186667}

LABEL_COL = "ball_outcome"
WICKET_RAW = -1
WICKET_INTERMEDIATE = 7
# xgboost_v2.py: ball_outcome -> target (-1 becomes 7), then remapped to the
# six consecutive classes the sim wrapper's class_to_outcome dict expects.
CLASS_MAPPING = {0: 0, 1: 1, 2: 2, 4: 3, 6: 4, 7: 5}
CLASS_LABELS = [0, 1, 2, 4, 6, "wicket"]
N_CLASSES = 6

# Bookkeeping columns copied through when the source frame carries them.
BOOKKEEPING_CANDIDATES = ("match_id", "innings_id", "ball_idx", "match_date")

# Serving sidecars copied verbatim from the production ball model.
SIDECAR_PKLS = (
    "batter_encoder",
    "bowler_encoder",
    "venue_encoder",
    "matchup_encoder",
)
SIDECAR_JSONS = ("outcome_dist_config",)

FORBIDDEN_READ_ROOTS = ("data/golden", "data/forward_holdout")

FEATURE_HASH_DEFINITION = (
    "sha256 of the ordered feature column names joined by '\\n' with a "
    "trailing '\\n', encoded utf-8"
)


# ---------------------------------------------------------------- helpers


def a50_feature_columns() -> list[str]:
    """The 50 A50 columns, in the order T1 concatenates them.

    Imported from the modules that define them so the arm can never drift
    from the transformer's feature contract.
    """
    columns = (
        list(EB_BAT_COLS)
        + list(EB_BOWL_COLS)
        + list(VENUE_COLS)
        + list(CTX_COLS)
        + list(STATE_COLS)
    )
    if len(columns) != 50:
        raise AssertionError(
            f"A50 must have exactly 50 columns, got {len(columns)}"
        )
    duplicates = sorted(
        {name for name in columns if columns.count(name) > 1}
    )
    if duplicates:
        raise AssertionError(f"duplicate A50 columns: {duplicates}")
    return columns


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def md5_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()  # noqa: S324 - artifact identity, not security
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def feature_columns_sha256(columns: list[str]) -> str:
    return sha256_text("".join(f"{name}\n" for name in columns))


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def assert_readable(path: Path) -> None:
    """Fail closed on the two corpora stage 0 must never read."""
    posix = rel(path)
    for root in FORBIDDEN_READ_ROOTS:
        if posix == root or posix.startswith(root + "/"):
            raise RuntimeError(f"stage 0 must not read {posix}")


def assert_writable(out_dir: Path) -> None:
    """Stage-0 output is confined to the stage-1 embeddings namespace."""
    allowed = (REPO / "models" / "embeddings" / "seq_stage1").resolve()
    resolved = out_dir.resolve()
    if resolved != allowed and allowed not in resolved.parents:
        raise RuntimeError(
            f"refusing to write outside {rel(allowed)}: {rel(resolved)}"
        )


def encode_target(frame: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Replicate xgboost_v2.py's label pipeline exactly, and prove it did."""
    raw = frame[LABEL_COL]
    observed = sorted(int(v) for v in pd.unique(raw))
    allowed = set(CLASS_MAPPING) | {WICKET_RAW}
    allowed.discard(WICKET_INTERMEDIATE)
    unexpected = [v for v in observed if v not in allowed]
    if unexpected:
        raise AssertionError(
            f"{LABEL_COL} carries values outside the 6-class contract: "
            f"{unexpected}"
        )
    target = raw.copy()
    target[target == WICKET_RAW] = WICKET_INTERMEDIATE
    kept = target <= WICKET_INTERMEDIATE
    dropped = int((~kept).sum())
    if dropped:
        raise AssertionError(
            f"{dropped} rows fall outside target <= 7; the A50 frame must "
            "preserve row count and order"
        )
    mapped = target.map(CLASS_MAPPING)
    if mapped.isna().any():
        raise AssertionError("unmapped target values after class remapping")
    mapped = mapped.astype(np.int64)
    present = sorted(int(v) for v in pd.unique(mapped))
    if present != list(range(N_CLASSES)):
        raise AssertionError(
            f"split does not carry all six classes: {present}"
        )
    return mapped, {
        "raw_values_observed": observed,
        "rows_dropped_by_target_filter": dropped,
        "classes_present": present,
    }


# ------------------------------------------------------------------ frame


def build_frame(
    source_dir: Path,
    out_dir: Path,
    columns: list[str],
) -> tuple[dict, dict]:
    """Write the A50 column selection and its provenance sidecar.

    Returns (record, written_paths); training reads the returned path
    objects, so the frame it trains on is the frame this function wrote.
    """
    assert_readable(source_dir)
    source_meta_path = source_dir / ".feature_hash"
    with source_meta_path.open() as handle:
        source_meta = json.load(handle)
    assert_venue_alias_contract(source_meta, context="A50 source frame")
    assert_elo_update_version(
        source_meta,
        expected=BASELINE_ELO_UPDATE_VERSION,
        context="A50 source frame",
    )
    if str(source_meta.get("version")) != DATA_VERSION:
        raise RuntimeError(
            f"source frame is version {source_meta.get('version')!r}, "
            f"expected {DATA_VERSION!r}"
        )

    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    record: dict = {
        "arm": ARM,
        "data_version": DATA_VERSION,
        "source_dir": rel(source_dir),
        "source_feature_hash": source_meta,
        "feature_columns": list(columns),
        "feature_columns_sha256": feature_columns_sha256(columns),
        "feature_columns_sha256_definition": FEATURE_HASH_DEFINITION,
        "label_column": LABEL_COL,
        "class_mapping": {
            "raw_to_intermediate": {str(WICKET_RAW): WICKET_INTERMEDIATE},
            "intermediate_to_class": {
                str(k): v for k, v in CLASS_MAPPING.items()
            },
            "classes": CLASS_LABELS,
        },
        "splits": {},
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    bookkeeping_used: list[str] | None = None
    written_paths: dict[str, Path] = {}
    for split in SPLITS:
        source_path = source_dir / f"cricket_data_{DATA_VERSION}_{split}.parquet"
        assert_readable(source_path)
        available = set(pq.ParquetFile(source_path).schema_arrow.names)
        missing = [name for name in columns if name not in available]
        if missing:
            raise RuntimeError(
                f"{rel(source_path)} is missing A50 columns: {missing}"
            )
        if LABEL_COL not in available:
            raise RuntimeError(f"{rel(source_path)} has no {LABEL_COL}")
        bookkeeping = [
            name for name in BOOKKEEPING_CANDIDATES if name in available
        ]
        if bookkeeping_used is None:
            bookkeeping_used = bookkeeping
        elif bookkeeping != bookkeeping_used:
            raise RuntimeError(
                "bookkeeping columns differ across splits: "
                f"{bookkeeping_used} vs {bookkeeping}"
            )
        written = list(columns) + [LABEL_COL] + bookkeeping

        print(f"[frame] {split}: reading {rel(source_path)}")
        frame = pd.read_parquet(source_path, columns=written)
        rows = int(len(frame))
        expected = EXPECTED_ROWS[split]
        if rows != expected:
            raise RuntimeError(
                f"{split}: {rows} rows, expected {expected}"
            )
        out_path = data_dir / f"{ARM}_{split}.parquet"
        frame.to_parquet(out_path, index=False)
        written_paths[split] = out_path
        del frame

        record["splits"][split] = {
            "source_parquet": rel(source_path),
            "source_parquet_md5": md5_file(source_path),
            "a50_parquet": rel(out_path),
            "a50_parquet_md5": md5_file(out_path),
            "rows": rows,
            "expected_rows": expected,
        }
        print(f"[frame] {split}: wrote {rel(out_path)} ({rows} rows)")

    record["bookkeeping_columns"] = bookkeeping_used or []
    record["written_columns"] = (
        list(columns) + [LABEL_COL] + (bookkeeping_used or [])
    )
    record["row_counts"] = {
        split: record["splits"][split]["rows"] for split in SPLITS
    }

    hash_path = data_dir / "feature_hash.json"
    with hash_path.open("w") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    print(f"[frame] wrote {rel(hash_path)}")
    return record, written_paths


def resolve_recorded(path_str: str) -> Path:
    """Recorded paths are repo-relative when they live under the repo."""
    path = Path(path_str)
    return path if path.is_absolute() else REPO / path


def verify_frame(
    out_dir: Path,
    columns: list[str],
    source_dir: Path,
    *,
    expected_rows: dict | None = None,
) -> tuple[dict, dict, dict]:
    """Re-derive every hash a reused A50 frame claims, before training on it.

    Reusing `data/feature_hash.json` on trust would let the contract describe
    data the booster never saw: a rewritten A50 parquet, or an i7 source frame
    that moved underneath it.

    The hashes are taken over exactly the files THIS invocation will read —
    `out_dir/data/a50_<split>.parquet` and
    `<source_dir>/cricket_data_i7_<split>.parquet` — and never over whatever
    paths the record happens to name. The record's own paths are then required
    to resolve (symlinks and all) to those same files, so a frame directory
    copied elsewhere with its original manifest is refused instead of
    verifying the originals and training on the copies.

    Returns (record, verification, verified_paths); training must read the
    returned path objects, not freshly constructed ones.
    """
    if expected_rows is None:
        expected_rows = EXPECTED_ROWS
    record_path = out_dir / "data" / "feature_hash.json"
    if not record_path.exists():
        raise RuntimeError(
            f"no A50 frame to reuse: {rel(record_path)} does not exist"
        )
    with record_path.open() as handle:
        record = json.load(handle)

    problems: list[str] = []
    if record.get("feature_columns") != columns:
        problems.append(
            f"{rel(record_path)}: recorded feature_columns differ from the "
            "imported A50 column list"
        )
    recorded_hash = record.get("feature_columns_sha256")
    actual_hash = feature_columns_sha256(columns)
    if recorded_hash != actual_hash:
        problems.append(
            f"{rel(record_path)}: feature_columns_sha256 {recorded_hash!r} "
            f"!= recomputed {actual_hash!r}"
        )
    recorded_source_dir = record.get("source_dir")
    if recorded_source_dir is None:
        problems.append(f"{rel(record_path)}: no recorded source_dir")
    elif (
        resolve_recorded(recorded_source_dir).resolve()
        != source_dir.resolve()
    ):
        problems.append(
            f"{rel(record_path)}: recorded source_dir "
            f"{resolve_recorded(recorded_source_dir).resolve().as_posix()} "
            f"does not resolve to the requested "
            f"{source_dir.resolve().as_posix()}"
        )

    verification: dict = {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "feature_hash_record": rel(record_path),
        "feature_columns_sha256": actual_hash,
        "splits": {},
    }
    verified_paths: dict[str, Path] = {}
    written = record.get("written_columns")
    for split in SPLITS:
        entry = record.get("splits", {}).get(split)
        if entry is None:
            problems.append(
                f"{rel(record_path)}: no record for split {split!r}"
            )
            continue
        # The files this invocation will actually read.
        a50_path = out_dir / "data" / f"{ARM}_{split}.parquet"
        source_path = (
            source_dir / f"cricket_data_{DATA_VERSION}_{split}.parquet"
        )
        split_result: dict = {}
        for role, md5_key, path in (
            ("a50_parquet", "a50_parquet_md5", a50_path),
            ("source_parquet", "source_parquet_md5", source_path),
        ):
            assert_readable(path)
            split_result[role] = rel(path)
            recorded_path = entry.get(role)
            if recorded_path is None:
                problems.append(
                    f"{rel(record_path)}: split {split!r} records no {role}"
                )
            elif (
                resolve_recorded(recorded_path).resolve() != path.resolve()
            ):
                problems.append(
                    f"{rel(path)}: recorded {role} "
                    f"{resolve_recorded(recorded_path).resolve().as_posix()} "
                    f"does not resolve to the file this run reads, "
                    f"{path.resolve().as_posix()}"
                )
            if not path.exists():
                problems.append(f"{rel(path)}: {role} is missing")
                continue
            digest = md5_file(path)
            split_result[md5_key] = digest
            if digest != entry.get(md5_key):
                problems.append(
                    f"{rel(path)}: md5 {digest} != recorded "
                    f"{entry.get(md5_key)}"
                )
        if a50_path.exists():
            handle = pq.ParquetFile(a50_path)
            rows = int(handle.metadata.num_rows)
            split_result["rows"] = rows
            verified_paths[split] = a50_path
            if rows != entry.get("rows"):
                problems.append(
                    f"{rel(a50_path)}: {rows} rows != recorded "
                    f"{entry.get('rows')}"
                )
            wanted = expected_rows.get(split)
            if wanted is not None and rows != wanted:
                problems.append(
                    f"{rel(a50_path)}: {rows} rows != expected {wanted}"
                )
            if written is not None:
                actual_columns = list(handle.schema_arrow.names)
                if actual_columns != written:
                    problems.append(
                        f"{rel(a50_path)}: columns differ from the recorded "
                        "written_columns"
                    )
        verification["splits"][split] = split_result

    if problems:
        raise RuntimeError(
            "A50 frame verification failed; refusing to train:\n  "
            + "\n  ".join(problems)
        )
    return record, verification, verified_paths


# --------------------------------------------------------------- training


def load_split(path: Path, columns: list[str]):
    """Read one verified A50 split. `path` is the object the frame step
    wrote or the verifier hashed — never a freshly constructed one."""
    frame = pd.read_parquet(path, columns=list(columns) + [LABEL_COL])
    target, label_audit = encode_target(frame)
    return frame[columns], target, label_audit


def copy_sidecars(production_dir: Path, out_dir: Path) -> dict:
    """Copy the production serving sidecars under both suffix conventions."""
    copied: dict[str, dict] = {}
    for stem, suffix in (
        *[(name, "pkl") for name in SIDECAR_PKLS],
        *[(name, "json") for name in SIDECAR_JSONS],
    ):
        source = production_dir / f"{stem}_{DATA_VERSION}.{suffix}"
        if not source.exists():
            raise FileNotFoundError(f"missing production sidecar {rel(source)}")
        digest = md5_file(source)
        targets = [
            out_dir / f"{stem}_{DATA_VERSION}.{suffix}",
            out_dir / f"{stem}_{ARM}.{suffix}",
        ]
        for target in targets:
            shutil.copyfile(source, target)
            if md5_file(target) != digest:
                raise RuntimeError(f"sidecar copy mismatch for {rel(target)}")
        copied[source.name] = {
            "source": rel(source),
            "md5": digest,
            "copies": [target.name for target in targets],
        }
        print(f"[sidecar] {source.name} -> {[t.name for t in targets]}")
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the A50 50-feature ball model (stage 0 D4)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "experiments" / "configs"
        / "xgb_v6_hierarchical_shrink.yaml",
        help="experiment YAML supplying model.hyperparameters",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=REPO / "data" / f"xgb_data_{DATA_VERSION}",
        help="i7 ball frame directory",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "models" / "embeddings" / "seq_stage1" / ARM,
        help="A50 artifact directory (must sit under the stage-1 namespace)",
    )
    parser.add_argument(
        "--production-dir",
        type=Path,
        default=REPO / "models" / f"xgb_{DATA_VERSION}_noweights_production",
        help="production ball model whose sidecars are copied verbatim",
    )
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument(
        "--skip-frame",
        action="store_true",
        help="reuse an already-written A50 frame; every recorded A50 and "
             "source parquet md5 is recomputed and must match before "
             "training starts",
    )
    parser.add_argument(
        "--verify-frames-only",
        action="store_true",
        help="recompute and check the existing A50 frame hashes, print the "
             "result, and exit without training or writing anything",
    )
    args = parser.parse_args()

    started = time.time()
    out_dir = args.out_dir
    assert_writable(out_dir)
    if not args.verify_frames_only:
        out_dir.mkdir(parents=True, exist_ok=True)

    columns = a50_feature_columns()
    print(f"[a50] {len(columns)} features (imported, deduplicated)")

    config_path = args.config
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    hyperparameters = dict(config["model"]["hyperparameters"])
    if not hyperparameters:
        raise RuntimeError(f"{rel(config_path)} has no model.hyperparameters")
    print(f"[a50] hyperparameters from {rel(config_path)}: {hyperparameters}")

    frame_verification: dict | None = None
    if args.skip_frame or args.verify_frames_only:
        frame_record, frame_verification, split_paths = verify_frame(
            out_dir, columns, args.source_dir
        )
        print("[frame] reuse verified — every md5 recomputed on the files "
              "this run reads:")
        for split in SPLITS:
            entry = frame_verification["splits"][split]
            print(
                f"  {split:11s} a50 {entry['a50_parquet_md5']}  "
                f"source {entry['source_parquet_md5']}  "
                f"rows {entry['rows']}"
            )
        if args.verify_frames_only:
            print(json.dumps(frame_verification, indent=2))
            print("[verify] frames OK; nothing trained, nothing written")
            return 0
    else:
        frame_record, split_paths = build_frame(
            args.source_dir, out_dir, columns
        )

    splits = {}
    label_audit = {}
    for split in SPLITS:
        features, target, audit = load_split(split_paths[split], columns)
        print(f"[data] {split}: reading {rel(split_paths[split])}")
        splits[split] = (features, target)
        label_audit[split] = audit
        counts = target.value_counts().sort_index().to_dict()
        print(f"[data] {split}: {features.shape} classes {counts}")

    params = dict(hyperparameters)
    params.update(
        {
            "random_state": 29,
            "eval_metric": "mlogloss",
            "early_stopping_rounds": 100,
            "scale_pos_weight": None,
            "n_jobs": args.n_jobs,
        }
    )
    model = XGBClassifier(**params)

    x_train, y_train = splits["train"]
    x_val, y_val = splits["validation"]
    x_test, y_test = splits["test"]

    print("[train] fitting (no sample_weight: --no-class-weights semantics)")
    fit_started = time.time()
    # D6 `--no-class-weights` semantics: the sample_weight kwarg is omitted
    # entirely, so the booster estimates P(outcome|state) directly.
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_train, y_train), (x_val, y_val)],
        verbose=50,
    )
    fit_seconds = time.time() - fit_started
    best_iteration = int(model.best_iteration)
    print(f"[train] done in {fit_seconds:.1f}s, best_iteration={best_iteration}")

    metrics = {
        "arm": ARM,
        "data_version": DATA_VERSION,
        "n_features": len(columns),
        "best_iteration": best_iteration,
        "best_score_validation_mlogloss": float(model.best_score),
        "splits": {},
        "fit_seconds": round(fit_seconds, 3),
    }
    labels = list(range(N_CLASSES))
    for split, (features, target) in splits.items():
        proba = model.predict_proba(features)
        predicted = proba.argmax(axis=1)
        metrics["splits"][split] = {
            "rows": int(len(target)),
            "log_loss": float(log_loss(target, proba, labels=labels)),
            "accuracy": float(accuracy_score(target, predicted)),
        }
        print(
            f"[eval] {split}: LL {metrics['splits'][split]['log_loss']:.6f} "
            f"acc {metrics['splits'][split]['accuracy']:.6f}"
        )

    model_path = out_dir / f"xgboost_model_{ARM}.pkl"
    joblib.dump(model, model_path)
    booster_md5 = md5_file(model_path)
    print(f"[save] {rel(model_path)} md5 {booster_md5}")

    columns_path = out_dir / f"feature_columns_{ARM}.txt"
    with columns_path.open("w") as handle:
        for name in columns:
            handle.write(f"{name}\n")

    sidecars = copy_sidecars(args.production_dir, out_dir)

    metrics["wall_seconds"] = round(time.time() - started, 3)
    metrics["model_pkl_md5"] = booster_md5
    metrics["feature_columns_sha256"] = feature_columns_sha256(columns)
    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")

    contract = {
        "arm": ARM,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_version": DATA_VERSION,
        "cache_schema_version": 4,
        "delivery_semantics": "inclusive_total_runs_v1",
        "elo_update_version": BASELINE_ELO_UPDATE_VERSION,
        "extras_contract": None,
        "ball_calibrator": None,
        "class_weights": None,
        "class_weights_note": (
            "no sample_weight kwarg is passed to fit at all — the "
            "--no-class-weights semantics of xgboost_v2.py"
        ),
        "hyperparameters": hyperparameters,
        "fit_params": {
            key: value for key, value in params.items()
            if key not in hyperparameters
        },
        "xgb_classifier_params": params,
        "random_state": 29,
        "eval_metric": "mlogloss",
        "early_stopping": {
            "rounds": 100,
            "eval_set": ["train", "validation"],
            "monitored": "validation (the last eval_set entry)",
            "best_iteration": best_iteration,
        },
        "label_mapping": {
            "source_column": LABEL_COL,
            "raw_to_intermediate": {str(WICKET_RAW): WICKET_INTERMEDIATE},
            "intermediate_to_class": {
                str(k): v for k, v in CLASS_MAPPING.items()
            },
            "classes": CLASS_LABELS,
            "per_split_audit": label_audit,
        },
        "feature_columns": columns,
        "feature_columns_file": columns_path.name,
        "feature_hash": {
            "sha256": feature_columns_sha256(columns),
            "definition": FEATURE_HASH_DEFINITION,
            "record": rel(out_dir / "data" / "feature_hash.json"),
            "source_frame": frame_record.get("source_feature_hash"),
        },
        "frame": {
            "source_dir": frame_record["source_dir"],
            "written_columns": frame_record["written_columns"],
            "bookkeeping_columns": frame_record["bookkeeping_columns"],
            "splits": frame_record["splits"],
            "built_in_this_run": frame_verification is None,
        },
        # True only when this run reused an existing frame AND recomputed
        # every A50 / source parquet md5 against the record before training.
        "frame_reuse_verified": frame_verification is not None,
        "frame_verification": frame_verification,
        "row_counts": frame_record["row_counts"],
        "model_artifact": model_path.name,
        "booster_md5": booster_md5,
        "sidecars": sidecars,
        "source_config": {
            "path": rel(config_path),
            "sha256": sha256_file(config_path),
        },
        "venue_identity": venue_alias_contract(),
        "environment": {
            "python": platform.python_version(),
            "xgboost": xgboost.__version__,
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "n_jobs": args.n_jobs,
        },
        "metrics": metrics["splits"],
        "wall_seconds": metrics["wall_seconds"],
    }
    contract_path = out_dir / "training_contract.json"
    with contract_path.open("w") as handle:
        json.dump(contract, handle, indent=2)
        handle.write("\n")

    # The sim wrapper derives its sidecar suffix from the model filename, so
    # it looks for `training_contract_a50.json`. Write the same serving
    # fields the production `training_contract_i7.json` carries, so the
    # delivery-semantics / ELO checks resolve identically instead of by
    # default.
    serving_contract = {
        "data_version": DATA_VERSION,
        "cache_schema_version": 4,
        "delivery_semantics": "inclusive_total_runs_v1",
        "elo_update_version": BASELINE_ELO_UPDATE_VERSION,
        "extras_contract": None,
        "ball_calibrator": None,
        "train_rows": frame_record["row_counts"]["train"],
        "validation_rows": frame_record["row_counts"]["validation"],
        "test_rows": frame_record["row_counts"]["test"],
        "classes": CLASS_LABELS,
        "venue_identity": venue_alias_contract(),
    }
    serving_path = out_dir / f"training_contract_{ARM}.json"
    with serving_path.open("w") as handle:
        json.dump(serving_contract, handle, indent=2)
        handle.write("\n")

    print(f"[done] {rel(out_dir)} in {metrics['wall_seconds']:.1f}s")
    for split in SPLITS:
        entry = metrics["splits"][split]
        print(f"  {split:11s} LL {entry['log_loss']:.6f}  "
              f"acc {entry['accuracy']:.6f}  rows {entry['rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
