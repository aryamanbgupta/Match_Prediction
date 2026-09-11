#!/usr/bin/env python3
"""Base log-probabilities for the stage 2 residual arms (checks D4.1-D4.4).

The residual arms (15 `residual_mlp`, 16 `residual_t1`) model
``p = softmax(log p_base + r_theta)``, with ``p_base`` from the production ball
model (`models/xgb_i7_noweights_production`, manifest role `ball_model_prod`,
the D16 no-class-weights stack, served RAW). This script writes those
``log p_base`` matrices, row-aligned with the parquets of the i7 ball frame
(`data/xgb_data_i7`, role `ball_frame_i7`), so a trainer only ever reads a
frozen `npz`.

Splits
------
* ``validation``, ``test`` and (later) ``cohort``: scored by the production
  booster itself. Those rows are out of the booster's training sample, so its
  own predictions are legitimate base logits.
* ``train``: the production booster was fit on these rows, so its in-sample
  predictions would hand the residual arms a leaked base. Instead the split is
  cut into **5 contiguous `match_date` blocks** of roughly equal row count and
  each block is scored by a **refit on the other four blocks only**
  (leave-one-date-block-out, out-of-fold).

Load path and feature preparation
---------------------------------
The repository's own path is used, exactly as `scripts/evaluate_i8_ball_model.py`
does it: `feature_columns_i7.txt` gives the 114 columns,
`calibration._apply_encoders_to_df` applies the **production** label encoders
(`batter/bowler/venue/matchup_encoder_i7.pkl`) to produce the four `*_encoded`
columns, and `XGBClassifier.predict_proba` scores `df[feature_columns]`. The
encoders are fit artifacts of the production training run and are reused
verbatim for every refit — they are **never refit**, so no refit can differ
from production in its identity encoding, and a refit's encoding of its
held-out block is the same mapping production used.

Class order
-----------
The booster's classes are the remap ``{0:0, 1:1, 2:2, 4:3, 6:4, wicket:5}``
from `scripts/xgboost_v2.py`. T1's order is `embeddings_e1.CLASS_MAPPING`
``{0:0, 1:1, 2:2, 4:3, 6:4, -1:5}``. The two agree, but the permutation is
derived programmatically from both maps and applied (and asserted to be a
permutation of all six classes) rather than assumed.

Hyperparameters for the refits
------------------------------
Read from the production artifact and the registered config, and recorded in
the sidecar under ``hyperparameters`` with a ``sources`` block:

* ``max_depth``, ``learning_rate`` (0.24036372383981375), ``subsample``,
  ``colsample_bytree``, ``reg_alpha``, ``reg_lambda``, ``random_state`` (29),
  ``objective``, ``eval_metric`` — read from the pickled production
  `XGBClassifier.get_params()`, and cross-checked against
  `experiments/configs/xgb_i7_venue_identity.yaml`
  (`model.hyperparameters`), which is the config D16 trained from. A mismatch
  refuses.
* ``n_estimators = best_iteration + 1 = 25`` — the production fit ran
  ``n_estimators=444`` with ``early_stopping_rounds=100`` and stopped at
  ``best_iteration=24`` (booster attribute; CLAUDE.md records the same 24), so
  its `predict_proba` uses ``iteration_range=(0, 25)``. The refits therefore
  run a fixed 25 rounds with **no early stopping and no eval set**: the
  effective round count is production's, each refit is deterministic, and the
  number of rounds is not re-selected per fold against the validation split
  (which the residual arms also consume). Recorded as a deviation.
* **No class weights.** D16 is the ``--no-class-weights`` arm: the trainer
  omits the ``sample_weight`` kwarg entirely, so every refit calls ``fit``
  with no weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import artifacts  # noqa: E402
from calibration import _apply_encoders_to_df  # noqa: E402
from embeddings_e1 import CLASS_MAPPING  # noqa: E402

CLASS_NAMES = ("dot", "one", "two", "four", "six", "wicket")
# scripts/xgboost_v2.py, "Remapping target classes to be consecutive":
PRODUCTION_CLASS_MAPPING = {0: 0, 1: 1, 2: 2, 4: 3, 6: 4, 7: 5}
# ball_outcome -> booster target, including the parquet's -1 spelling of a
# wicket (scripts/evaluate_i8_ball_model.TARGET_MAP).
TARGET_MAP = {0: 0, 1: 1, 2: 2, 4: 3, 6: 4, 7: 5, -1: 5}
FLOOR = 1e-4
N_BLOCKS = 5
DEFAULT_OUT = REPO / "models" / "embeddings" / "seq_stage2" / "base_logits"
CONFIG_YAML = REPO / "experiments" / "configs" / "xgb_i7_venue_identity.yaml"
DATE_COL = "match_date"
ENCODED_SOURCES = {
    "batter_encoded": "batter_id",
    "bowler_encoded": "bowler_id",
    "venue_encoded": "venue",
    "matchup_type_encoded": "matchup_type",
}


# --------------------------------------------------------------------------
# identity helpers
# --------------------------------------------------------------------------

def md5_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    """md5 of a file's bytes (same contract as `transformer_t1.md5_file`)."""
    digest = hashlib.md5()  # noqa: S324 - artifact identity, not security
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def booster_md5(model) -> str:
    """md5 of a fitted booster's raw UBJSON bytes."""
    raw = model.get_booster().save_raw(raw_format="ubj")
    return hashlib.md5(bytes(raw)).hexdigest()  # noqa: S324


# --------------------------------------------------------------------------
# class order
# --------------------------------------------------------------------------

def class_permutation() -> list[int]:
    """Booster class index -> T1 class index, derived from both maps.

    ``out[:, j] = booster_proba[:, perm.index(j)]`` is expressed here as
    ``perm[booster_index] = t1_index``.
    """
    t1 = dict(CLASS_MAPPING)  # {0,1,2,4,6,-1} -> 0..5
    t1[7] = t1.pop(-1)  # the booster spells a wicket 7, the frame spells it -1
    perm = [None] * len(PRODUCTION_CLASS_MAPPING)
    for outcome, booster_index in PRODUCTION_CLASS_MAPPING.items():
        if outcome not in t1:
            raise RuntimeError(f"outcome {outcome} absent from CLASS_MAPPING")
        perm[booster_index] = t1[outcome]
    if sorted(perm) != list(range(len(CLASS_NAMES))):
        raise RuntimeError(f"class map is not a permutation: {perm}")
    return [int(x) for x in perm]


def to_t1_order(proba: np.ndarray, perm: list[int]) -> np.ndarray:
    out = np.empty_like(proba)
    for booster_index, t1_index in enumerate(perm):
        out[:, t1_index] = proba[:, booster_index]
    return out


def floor_log(proba: np.ndarray, floor: float = FLOOR) -> np.ndarray:
    """Floor at ``floor``, renormalise rows to 1, take the log."""
    p = np.clip(np.asarray(proba, dtype=np.float64), floor, None)
    p /= p.sum(axis=1, keepdims=True)
    return np.log(p).astype(np.float32)


def log_loss_from_logp(logp: np.ndarray, labels: np.ndarray) -> float:
    return float(-logp[np.arange(len(labels)), labels].astype(np.float64).mean())


# --------------------------------------------------------------------------
# production artifact
# --------------------------------------------------------------------------

def read_feature_columns(model_dir: Path, version: str) -> list[str]:
    path = model_dir / f"feature_columns_{version}.txt"
    return [line.strip() for line in path.read_text().splitlines()
            if line.strip()]


def _yaml_hyperparameters(path: Path) -> dict:
    """The `model.hyperparameters` block of the i7 config, read with pyyaml."""
    import yaml

    with path.open() as handle:
        config = yaml.safe_load(handle)
    return dict(config["model"]["hyperparameters"])


def production_hyperparameters(model, config_yaml: Path | None) -> dict:
    """Refit hyperparameters plus a provenance block naming every source."""
    params = model.get_params()
    best_iteration = int(model.get_booster().attributes()["best_iteration"])
    keys = ("max_depth", "learning_rate", "subsample", "colsample_bytree",
            "reg_alpha", "reg_lambda", "random_state", "objective",
            "eval_metric")
    refit = {k: params[k] for k in keys}
    refit["n_estimators"] = best_iteration + 1
    refit["nthread"] = int(os.environ.get("OMP_NUM_THREADS", "8"))

    sources = {k: f"{model.__class__.__name__}.get_params() of the production "
                  "xgboost_model_i7.pkl" for k in keys}
    sources["n_estimators"] = (
        f"best_iteration {best_iteration} + 1 = {best_iteration + 1}; booster "
        "attribute 'best_iteration' of the production pickle (production fit "
        f"was n_estimators={params['n_estimators']} with "
        f"early_stopping_rounds={params['early_stopping_rounds']}, so its "
        "predict_proba uses iteration_range=(0, "
        f"{best_iteration + 1})); CLAUDE.md records best_iteration = 24")
    sources["nthread"] = "OMP_NUM_THREADS environment variable (default 8)"
    sources["sample_weight"] = (
        "omitted entirely: D16 is the --no-class-weights arm of "
        "scripts/xgboost_v2.py (research/reports/auto/D16.md)")
    sources["early_stopping_rounds"] = (
        "not used in the refits: rounds are fixed at the production "
        "best_iteration + 1 so no fold re-selects rounds against the "
        "validation split (recorded deviation)")

    cross_check = None
    if config_yaml is not None and config_yaml.exists():
        yaml_hp = _yaml_hyperparameters(config_yaml)
        mismatch = {
            k: {"config": yaml_hp[k], "artifact": params[k]}
            for k in yaml_hp
            if k in params and yaml_hp[k] != params[k]
        }
        if mismatch:
            raise RuntimeError(
                "production artifact hyperparameters disagree with "
                f"{config_yaml}: {mismatch}")
        cross_check = {
            "config": config_yaml.relative_to(REPO).as_posix(),
            "checked_keys": sorted(k for k in yaml_hp if k in params),
            "agrees": True,
        }
    return {"refit": refit, "sources": sources, "config_cross_check":
            cross_check}


def load_production_model(model_dir: Path, version: str):
    import joblib

    path = model_dir / f"xgboost_model_{version}.pkl"
    model = joblib.load(path)
    return model, path


# --------------------------------------------------------------------------
# frames
# --------------------------------------------------------------------------

def columns_to_read(feature_columns: list[str], with_date: bool) -> list[str]:
    """Only what the model and the block cut need, to keep memory sane."""
    cols = [c for c in feature_columns if c not in ENCODED_SOURCES]
    for encoded, source in ENCODED_SOURCES.items():
        if encoded in feature_columns:
            cols.append(source)
    cols.append("ball_outcome")
    if with_date:
        cols.append(DATE_COL)
    seen: dict[str, None] = {}
    for c in cols:
        seen.setdefault(c, None)
    return list(seen)


def prepare_frame(parquet: Path, feature_columns: list[str], model_dir: Path,
                  with_date: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    """Read a split, apply the production encoders, return (X-ready df, labels)."""
    df = pd.read_parquet(parquet,
                         columns=columns_to_read(feature_columns, with_date))
    labels = df["ball_outcome"].map(TARGET_MAP)
    if labels.isna().any():
        unknown = sorted(df.loc[labels.isna(), "ball_outcome"].unique())
        raise RuntimeError(f"{parquet} has unknown ball outcomes: {unknown}")
    _apply_encoders_to_df(df, feature_columns, encoder_dir=str(model_dir))
    missing = sorted(set(feature_columns) - set(df.columns))
    if missing:
        raise RuntimeError(f"{parquet} is missing model features: {missing}")
    perm = class_permutation()
    labels = labels.to_numpy(dtype=np.int64)
    return df, np.asarray([perm[i] for i in range(len(CLASS_NAMES))],
                          dtype=np.int64)[labels]


# --------------------------------------------------------------------------
# date blocks
# --------------------------------------------------------------------------

def date_blocks(dates: pd.Series, n_blocks: int = N_BLOCKS) -> list[dict]:
    """Cut a chronologically sorted split into contiguous whole-date blocks.

    Boundaries are chosen greedily at date granularity so the blocks have
    roughly equal row counts: block ``b`` closes at the first date whose
    cumulative row count reaches ``(b + 1) * n_rows / n_blocks``.
    """
    dates = pd.Series(dates).astype(str)
    counts = dates.value_counts().sort_index()
    order = list(counts.index)
    cumulative = counts.cumsum().to_numpy()
    n_rows = int(cumulative[-1])
    if len(order) < n_blocks:
        raise RuntimeError(
            f"{len(order)} distinct dates cannot make {n_blocks} blocks")
    cuts: list[int] = []
    for b in range(1, n_blocks):
        target = b * n_rows / n_blocks
        idx = int(np.searchsorted(cumulative, target, side="left"))
        idx = max(idx, (cuts[-1] + 1) if cuts else 0)
        idx = min(idx, len(order) - (n_blocks - b))
        cuts.append(idx)
    edges = [-1, *cuts, len(order) - 1]
    blocks = []
    for b in range(n_blocks):
        lo, hi = edges[b] + 1, edges[b + 1]
        block_dates = order[lo:hi + 1]
        blocks.append({
            "block": b,
            "date_min": block_dates[0],
            "date_max": block_dates[-1],
            "n_dates": len(block_dates),
            "n_rows": int(counts.loc[block_dates].sum()),
        })
    if sum(b["n_rows"] for b in blocks) != n_rows:
        raise RuntimeError("date blocks do not partition the split")
    return blocks


def block_assignment(dates: pd.Series, blocks: list[dict]) -> np.ndarray:
    """Row -> block index, from the recorded date boundaries."""
    dates = pd.Series(dates).astype(str).to_numpy()
    assign = np.full(len(dates), -1, dtype=np.int64)
    for block in blocks:
        mask = (dates >= block["date_min"]) & (dates <= block["date_max"])
        assign[mask] = block["block"]
    if (assign < 0).any():
        raise RuntimeError("some rows fall outside every date block")
    return assign


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def score_with_production(model, df: pd.DataFrame,
                          feature_columns: list[str]) -> np.ndarray:
    proba = model.predict_proba(df[feature_columns])
    if proba.shape != (len(df), len(CLASS_NAMES)):
        raise RuntimeError(f"unexpected probability shape {proba.shape}")
    return to_t1_order(proba, class_permutation())


def _make_refit(hyperparameters: dict):
    from xgboost import XGBClassifier

    return XGBClassifier(**hyperparameters)


def oof_train_logits(df: pd.DataFrame, labels: np.ndarray,
                     feature_columns: list[str], blocks: list[dict],
                     hyperparameters: dict, *, fit_hook=None,
                     verbose: bool = True) -> tuple[np.ndarray, list[dict]]:
    """Leave-one-date-block-out base probabilities for the train split.

    Block ``b``'s rows are scored only by a booster fit on the rows of the
    other four blocks. ``fit_hook(block, fit_row_ids, score_row_ids)`` is
    called for every fold (the no-leak test uses it).
    """
    assign = block_assignment(df[DATE_COL], blocks)
    x_all = df[feature_columns]
    proba = np.zeros((len(df), len(CLASS_NAMES)), dtype=np.float64)
    perm = class_permutation()
    records = []
    for block in blocks:
        b = block["block"]
        score_rows = np.flatnonzero(assign == b)
        fit_rows = np.flatnonzero(assign != b)
        if fit_hook is not None:
            fit_hook(b, fit_rows, score_rows)
        if verbose:
            print(f"[oof] block {b} ({block['date_min']}..{block['date_max']}): "
                  f"fit {len(fit_rows)} rows, score {len(score_rows)} rows",
                  flush=True)
        model = _make_refit(hyperparameters)
        started = time.time()
        # No sample_weight: D16 is the --no-class-weights arm.
        model.fit(x_all.iloc[fit_rows], labels[fit_rows])
        wall = time.time() - started
        proba[score_rows] = to_t1_order(
            model.predict_proba(x_all.iloc[score_rows]), perm)
        records.append({
            **block,
            "n_fit_rows": int(len(fit_rows)),
            "n_scored_rows": int(len(score_rows)),
            "fit_wall_seconds": round(wall, 3),
            "booster_md5": booster_md5(model),
        })
        if verbose:
            print(f"[oof] block {b} refit in {wall:.1f} s", flush=True)
    return proba, records


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

def write_split(out_dir: Path, split: str, logp: np.ndarray,
                sidecar: dict) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{split}.npz"
    json_path = out_dir / f"{split}.json"
    np.savez(
        npz_path,
        logp=logp.astype(np.float32),
        n_rows=np.int64(logp.shape[0]),
        parquet_md5=np.array(sidecar["parquet_md5"]),
    )
    sidecar = {**sidecar, "npz_md5": md5_file(npz_path)}
    json_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    return npz_path, json_path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--split", required=True,
                   choices=["validation", "test", "train", "cohort"])
    p.add_argument("--parquet", default=None,
                   help="required for --split cohort; overrides the frame path")
    p.add_argument("--frame-dir", default=None,
                   help="default: manifest role ball_frame_i7")
    p.add_argument("--model-dir", default=None,
                   help="default: manifest role ball_model_prod")
    p.add_argument("--version", default="i7")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--n-blocks", type=int, default=N_BLOCKS)
    p.add_argument("--floor", type=float, default=FLOOR)
    p.add_argument("--threads", type=int, default=8,
                   help="OMP_NUM_THREADS / booster nthread for the refits")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ["OMP_NUM_THREADS"] = str(args.threads)

    frame_dir = Path(artifacts.artifact_path("ball_frame_i7", args.frame_dir))
    if not frame_dir.is_absolute():
        frame_dir = REPO / frame_dir
    model_dir = Path(artifacts.artifact_path("ball_model_prod", args.model_dir))
    if not model_dir.is_absolute():
        model_dir = REPO / model_dir

    if args.split == "cohort":
        if not args.parquet:
            raise SystemExit("--split cohort requires --parquet PATH")
        parquet = Path(args.parquet)
    elif args.parquet:
        parquet = Path(args.parquet)
    else:
        parquet = (frame_dir
                   / f"cricket_data_{args.version}_{args.split}.parquet")
    if not parquet.exists():
        raise SystemExit(f"missing parquet: {parquet}")

    feature_columns = read_feature_columns(model_dir, args.version)
    model, model_path = load_production_model(model_dir, args.version)
    hyper = production_hyperparameters(model, CONFIG_YAML)

    is_train = args.split == "train"
    started = time.time()
    df, labels = prepare_frame(parquet, feature_columns, model_dir,
                               with_date=is_train)
    print(f"[{args.split}] {len(df)} rows, {len(feature_columns)} features, "
          f"prepared in {time.time() - started:.1f} s", flush=True)

    sidecar: dict = {
        "script": Path(__file__).relative_to(REPO).as_posix(),
        "script_sha256": sha256_file(Path(__file__)),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "parquet": parquet.as_posix(),
        "parquet_md5": md5_file(parquet),
        "n_rows": int(len(df)),
        "frame_dir": frame_dir.relative_to(REPO).as_posix()
        if frame_dir.is_relative_to(REPO) else frame_dir.as_posix(),
        "model_dir": model_dir.relative_to(REPO).as_posix()
        if model_dir.is_relative_to(REPO) else model_dir.as_posix(),
        "model_role": "ball_model_prod",
        "booster_md5": md5_file(model_path),
        "booster_raw_md5": booster_md5(model),
        "n_features": len(feature_columns),
        "class_order": list(CLASS_NAMES),
        "class_permutation_booster_to_t1": class_permutation(),
        "floor": args.floor,
        "renormalised": True,
        "threads": args.threads,
    }

    if is_train:
        blocks = date_blocks(df[DATE_COL], args.n_blocks)
        sidecar["date_col"] = DATE_COL
        sidecar["date_min"] = str(df[DATE_COL].astype(str).min())
        sidecar["date_max"] = str(df[DATE_COL].astype(str).max())
        sidecar["n_blocks"] = args.n_blocks
        sidecar["base_source"] = (
            "leave-one-date-block-out refits of the production configuration")
        wall_started = time.time()
        proba, records = oof_train_logits(df, labels, feature_columns, blocks,
                                          hyper["refit"])
        sidecar["blocks"] = records
        sidecar["oof_total_wall_seconds"] = round(time.time() - wall_started, 3)
        sidecar["refit_wall_seconds"] = [r["fit_wall_seconds"]
                                         for r in records]
    else:
        sidecar["base_source"] = "production booster predict_proba"
        proba = score_with_production(model, df, feature_columns)

    logp = floor_log(proba, args.floor)
    ll = log_loss_from_logp(logp, labels)
    if is_train:
        sidecar["oof_train_log_loss"] = ll
    elif args.split == "validation":
        sidecar["production_model_validation_ll_not_a_stage2_number"] = ll
    else:
        sidecar[f"production_model_{args.split}_log_loss"] = ll
    sidecar["hyperparameters"] = hyper["refit"]
    sidecar["hyperparameter_sources"] = hyper["sources"]
    sidecar["hyperparameter_config_cross_check"] = hyper["config_cross_check"]
    sidecar["deviations"] = [
        "refits run a fixed n_estimators = production best_iteration + 1 = "
        f"{hyper['refit']['n_estimators']} with no early stopping and no eval "
        "set, instead of re-running production's early stopping per fold",
        "label encoders are the production fit artifacts, reused verbatim and "
        "never refit",
    ]

    npz_path, json_path = write_split(Path(args.out_dir), args.split, logp,
                                      sidecar)
    print(f"[{args.split}] wrote {npz_path} and {json_path}", flush=True)
    print(json.dumps({k: v for k, v in sidecar.items()
                      if k not in {"hyperparameter_sources"}},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
