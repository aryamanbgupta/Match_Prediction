# manifest-exempt: embeddings-ladder experimental artifact namespace
"""Stage 4 fixed reference models on the i7 ball frame — VALIDATION ONLY.

Builds the three reference points every Stage 4 rung is measured against:

  eb_ctx        multinomial logistic on the 42 EB-shrunk outcome-dist
                features + the 4 E2/E15 match-state context columns.
                This is the E15 "fair control" (B0b+ctx) ported from the
                v3 frame to the i7 frame of record.
  raw_rate_ctx  the same 4 context columns and the same 6 venue EB
                columns, but the BATTER and BOWLER rates are UNSHRUNK
                as-of empirical rates n_c/N read out of the i7 stats
                cache. It isolates how much of eb_ctx's skill is the
                empirical-Bayes shrinkage rather than the raw history.
  ref_lin_50    multinomial logistic on the full 50-feature transformer
                contract (`transformer_t1.build_features`), i.e. the
                linear floor under T1 on exactly T1's inputs.

Why: 4b/4d need frozen, reproducible linear references on the exact
frame the ladder trains on, so an embedding rung cannot be credited for
signal a plain rate table or a linear model already carries. The
row-aligned validation probability arrays are saved so 4b can use them
as frozen base logits.

TEST IS SEALED. This script fits on train and scores on validation only;
it raises if asked for the test split or any golden/forward_holdout path.

The as-of read reproduces `_SQLiteBackend._resolve_date_id` semantics
exactly (largest snapshot date <= match_date; snapshots for date D
already exclude D's own matches), but vectorised via merge_asof so
1.9M rows finish in seconds instead of per-row SQL.

Artifacts: models/embeddings/stage4/refs/ (gitignored namespace).

Usage:
    uv run --no-sync python scripts/sequence_track/stage4_references.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from feature_registry import FEATURE_GROUPS  # noqa: E402

from stage4_guard import assert_permitted_inputs  # noqa: E402

DATA_DIR = ROOT / "data/xgb_data_i7"
CACHE = ROOT / "models/player_stats_cache_i7.sqlite"
OUT_DIR = ROOT / "models/embeddings/stage4/refs"

# Sealed-split guard. Nothing in Stage 4 may read these.
ALLOWED_SPLITS = ("train", "validation")
FORBIDDEN_PATH_PARTS = ("golden", "forward_holdout")

# embeddings_eval_kit.py:48 — ball_outcome -> 6 contiguous classes.
CLASS_MAPPING = {0: 0, 1: 1, 2: 2, 4: 3, 6: 4, -1: 5}
CLASS_NAMES = ["dot", "one", "two", "four", "six", "wicket"]

C_REG = 1.0
MAX_ITER = 2000

EB_COLS = (
    FEATURE_GROUPS["batter_outcome_dist"]
    + FEATURE_GROUPS["bowler_outcome_dist"]
    + FEATURE_GROUPS["batter_vs_type_dist"]
    + FEATURE_GROUPS["bowler_vs_hand_dist"]
    + FEATURE_GROUPS["venue_outcome_dist"]
)
VENUE_COLS = FEATURE_GROUPS["venue_outcome_dist"]

# embeddings_e1.py:60 — the 4 state features the E2/E15 fair control used.
CTX_COLS = ["is_middle_overs", "is_death_overs", "wickets_in_hand",
            "chase_target"]

COUNT_SUFFIXES = ["p0", "p1", "p2", "p4", "p6", "pw"]
RAW_BAT_COLS = [f"batter_raw_{s}" for s in COUNT_SUFFIXES]
RAW_BOWL_COLS = [f"bowler_raw_{s}" for s in COUNT_SUFFIXES]


def check_split(name: str) -> str:
    """Fail closed on the sealed test split and on sealed corpora."""
    if name not in ALLOWED_SPLITS:
        raise ValueError(
            f"split {name!r} is not permitted: Stage 4 references are "
            f"validation-only (allowed: {', '.join(ALLOWED_SPLITS)}). "
            "The test split is sealed for this stage.")
    return name


def split_path(name: str, data_dir: Path = DATA_DIR) -> Path:
    """The parquet for `name`, validated by the Stage 4 guard first.

    check_split rejects the sealed split by name; assert_permitted_inputs
    then resolves symlinks and checks size + head sha256 against the pins,
    so a permitted NAME pointing at a sealed file is refused before any
    read. FORBIDDEN_PATH_PARTS is a redundant early tripwire, not the
    defence — the allowlist is.
    """
    check_split(name)
    path = data_dir / f"cricket_data_i7_{name}.parquet"
    if set(Path(os.path.realpath(path)).parts) & set(FORBIDDEN_PATH_PARTS):
        raise ValueError(f"refusing to read sealed corpus path: {path}")
    assert_permitted_inputs(path)
    return path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log_loss_from_probs(y: np.ndarray, probs: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(y)), y], 1e-15, 1.0)
    return float(-np.mean(np.log(p)))


# --- as-of machinery (shared with stage4_exposure_sidecar) -----------------

def load_cache_tables(cache_path: Path = CACHE) -> dict:
    """Read the i7 cache's date index, player map, and the cumulative
    batting/bowling count tables into memory (150k + 115k rows)."""
    assert_permitted_inputs(cache_path)
    conn = sqlite3.connect(str(cache_path))
    dates = [r[0] for r in conn.execute("SELECT date FROM dates ORDER BY id ASC")]
    players = pd.read_sql_query("SELECT id, player_id FROM players", conn)
    id_to_key = dict(zip(players["id"], players["player_id"]))
    meta = dict(conn.execute("SELECT key, value FROM _meta"))
    prior = np.array([float(meta[f"prior_{s}"]) for s in COUNT_SUFFIXES])

    bat = pd.read_sql_query(
        "SELECT player_id, date_id, c0, c1, c2, c4, c6, cw, recent_balls "
        "FROM batting", conn)
    bowl = pd.read_sql_query(
        "SELECT player_id, date_id, c0, c1, c2, c4, c6, cw, "
        "recent_balls_bowled AS recent_balls FROM bowling", conn)
    conn.close()
    for df in (bat, bowl):
        df["player_key"] = df["player_id"].map(id_to_key)
        df.drop(columns=["player_id"], inplace=True)
        df.sort_values("date_id", kind="mergesort", inplace=True)
        df.reset_index(drop=True, inplace=True)
    return {"dates": np.array(dates), "prior": prior, "meta": meta,
            "batting": bat, "bowling": bowl}


def resolve_date_ids(match_date: pd.Series, dates: np.ndarray) -> np.ndarray:
    """Vectorised _resolve_date_id: largest date_id whose date <= as-of."""
    return np.searchsorted(dates, match_date.to_numpy().astype(str),
                           side="right") - 1


def asof_counts(frame: pd.DataFrame, side: str, tables: dict) -> pd.DataFrame:
    """As-of cumulative counts for `side` in {'batter','bowler'}.

    Returns a frame in the SAME row order as `frame` with columns
    c0,c1,c2,c4,c6,cw,recent_balls (NaN -> 0 for unseen players/dates).
    """
    table = tables["batting" if side == "batter" else "bowling"]
    left = pd.DataFrame({
        "player_key": frame[f"{side}_id"].to_numpy(),
        "did": frame["_did"].to_numpy(),
        "_pos": np.arange(len(frame)),
    })
    left.sort_values("did", kind="mergesort", inplace=True)
    merged = pd.merge_asof(
        left, table, left_on="did", right_on="date_id",
        by="player_key", direction="backward", allow_exact_matches=True,
    )
    merged.sort_values("_pos", kind="mergesort", inplace=True)
    cols = ["c0", "c1", "c2", "c4", "c6", "cw", "recent_balls"]
    # NaN = player unseen, or as-of precedes every snapshot (did < 0, which
    # matches no date_id >= 0) -> no history -> zero counts.
    return pd.DataFrame(merged[cols].fillna(0.0).to_numpy(dtype=np.float64),
                        columns=cols)


def raw_rates(counts: pd.DataFrame, prior: np.ndarray) -> np.ndarray:
    """Unshrunk n_c/N, falling back to the global prior when N == 0."""
    c = counts[["c0", "c1", "c2", "c4", "c6", "cw"]].to_numpy(dtype=np.float64)
    n = c.sum(axis=1)
    rates = np.empty_like(c)
    nz = n > 0
    rates[nz] = c[nz] / n[nz, None]
    rates[~nz] = prior
    return rates


# --- frames ---------------------------------------------------------------

def build_frames(tables: dict, splits: list[str]) -> dict:
    import transformer_t1 as t1

    extra = (t1.EB_BAT_COLS + t1.EB_BOWL_COLS + t1.VENUE_COLS
             + t1.CTX_COLS + t1.STATE_COLS)
    cols = sorted(set(EB_COLS + CTX_COLS + extra
                      + ["ball_outcome", "match_date", "batter_id",
                         "bowler_id"]))
    out = {}
    for name in splits:
        df = pd.read_parquet(split_path(name), columns=cols)
        df["y"] = df["ball_outcome"].map(CLASS_MAPPING).astype(np.int8)
        df["_did"] = resolve_date_ids(df["match_date"], tables["dates"])
        bat = asof_counts(df, "batter", tables)
        bowl = asof_counts(df, "bowler", tables)
        df[RAW_BAT_COLS] = raw_rates(bat, tables["prior"])
        df[RAW_BOWL_COLS] = raw_rates(bowl, tables["prior"])
        zb = (bat[["c0", "c1", "c2", "c4", "c6", "cw"]].sum(1) == 0).mean()
        zw = (bowl[["c0", "c1", "c2", "c4", "c6", "cw"]].sum(1) == 0).mean()
        print(f"  {name}: {len(df):,} balls (batter N=0 on {zb:.2%}, "
              f"bowler N=0 on {zw:.2%})", flush=True)
        out[name] = df
    return out


def fit_reference(name: str, frames: dict, manifest: dict,
                  feat_cols: list[str] | None = None,
                  matrix_fn=None, feat_names: list[str] | None = None,
                  emit_train_probs: bool = False) -> None:
    """Fit one reference on train, score on validation only.

    Either `feat_cols` (plain parquet columns) or `matrix_fn` (a callable
    df -> ndarray, with `feat_names` labelling its columns) must be given.

    `emit_train_probs` additionally writes `<name>_train_probs.npz`. Those are
    IN-SAMPLE predictions of the frozen linear reference on the rows it was
    fitted on, needed because rung 4b's residual has to be trained on top of
    the base somewhere. They are recorded as the deviation
    `ref_train_probs_in_sample` and are NOT a performance read of any kind:
    the reference's own reported log loss stays the validation one.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import joblib

    if matrix_fn is None:
        names = list(feat_cols)
        def matrix_fn(df, _c=names):  # noqa: E306
            return df[_c].to_numpy(dtype=np.float32)
    else:
        names = list(feat_names)

    print(f"\n{name}: logistic on {len(names)} features...", flush=True)
    tr = frames["train"]
    # Scaler is fitted on TRAIN rows only, inside the pipeline, so the
    # frozen joblib carries its means/scales and validation is never used
    # to set them.
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(C=C_REG, max_iter=MAX_ITER,
                                      solver="lbfgs", n_jobs=-1)),
    ])
    t0 = time.time()
    X_tr = matrix_fn(tr)
    if X_tr.shape[1] != len(names):
        raise AssertionError(f"{name}: matrix has {X_tr.shape[1]} columns, "
                             f"feature-name list has {len(names)}")
    clf.fit(X_tr, tr["y"].to_numpy())
    logreg = clf.named_steps["logreg"]
    scaler = clf.named_steps["scaler"]
    n_iter = int(np.max(logreg.n_iter_))
    converged = n_iter < MAX_ITER
    print(f"  fit in {time.time()-t0:.0f}s (n_iter={n_iter}, "
          f"converged={converged})", flush=True)
    if not converged:
        raise RuntimeError(
            f"{name}: lbfgs hit max_iter={MAX_ITER} without converging; "
            "refusing to write a non-converged frozen reference")

    val = frames["validation"]
    probs = clf.predict_proba(matrix_fn(val))
    ll = log_loss_from_probs(val["y"].to_numpy(), probs)
    print(f"  validation LL = {ll:.4f}", flush=True)

    train_probs_entry = None
    if emit_train_probs:
        train_probs = clf.predict_proba(X_tr)
        train_ll = log_loss_from_probs(tr["y"].to_numpy(), train_probs)
        train_probs_path = OUT_DIR / f"{name}_train_probs.npz"
        np.savez_compressed(train_probs_path,
                            probs=train_probs.astype(np.float32),
                            y=tr["y"].to_numpy(np.int8))
        print(f"  train (IN-SAMPLE) LL = {train_ll:.4f}", flush=True)
        train_probs_entry = {
            "path": str(train_probs_path.relative_to(ROOT)),
            "sha256": sha256(train_probs_path),
            "shape": list(train_probs.shape),
            "in_sample": True,
            "log_loss_in_sample": round(train_ll, 4),
            "note": "row-aligned to the train parquet's row order; IN-SAMPLE "
                    "predictions of the frozen reference on its own fit "
                    "rows, emitted only so rung 4b's identity residual has a "
                    "base to train on. Deviation: "
                    "ref_train_probs_in_sample. Never a performance read.",
        }

    model_path = OUT_DIR / f"{name}.joblib"
    probs_path = OUT_DIR / f"{name}_validation_probs.npz"
    joblib.dump(clf, model_path)
    np.savez_compressed(probs_path, probs=probs.astype(np.float32),
                        y=val["y"].to_numpy(np.int8))

    manifest[name] = {
        "feature_names": names,
        "n_features": len(names),
        "C": C_REG,
        "solver": "lbfgs",
        "max_iter": MAX_ITER,
        "n_iter": n_iter,
        "converged": converged,
        "scaler": {
            "kind": "StandardScaler (fitted on train rows only)",
            "mean": [float(v) for v in scaler.mean_],
            "scale": [float(v) for v in scaler.scale_],
        },
        "class_mapping": {str(k): v for k, v in CLASS_MAPPING.items()},
        "n_rows": {k: int(len(v)) for k, v in frames.items()},
        "log_loss": {"validation": round(ll, 4)},
        "artifacts": {
            "model": {"path": str(model_path.relative_to(ROOT)),
                      "sha256": sha256(model_path)},
            "validation_probs": {
                "path": str(probs_path.relative_to(ROOT)),
                "sha256": sha256(probs_path),
                "shape": list(probs.shape),
                "note": "row-aligned to the validation parquet's row order; "
                        "frozen base probabilities for rung 4b",
            },
        },
    }
    if train_probs_entry is not None:
        manifest[name]["artifacts"]["train_probs"] = train_probs_entry
        manifest[name].setdefault("deviations", []).append(
            "ref_train_probs_in_sample: the train-row base probabilities are "
            "in-sample predictions of this frozen reference. Rung 4b's "
            "residual therefore learns on top of a base that is optimistic "
            "on the train rows; the base cannot move, and every reported "
            "number is on validation.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train,validation",
                    help="comma-separated; only train,validation are allowed")
    ap.add_argument("--emit-train-probs", default="",
                    help="comma-separated reference names that should also "
                         "write <name>_train_probs.npz (IN-SAMPLE train-row "
                         "probabilities, needed to train rung 4b's identity "
                         "residual); off by default")
    args = ap.parse_args()
    emit = {n.strip() for n in args.emit_train_probs.split(",") if n.strip()}
    known = {"eb_ctx", "raw_rate_ctx", "ref_lin_50"}
    if not emit <= known:
        raise ValueError(f"--emit-train-probs names {sorted(emit - known)}, "
                         f"which are not references; known: {sorted(known)}")
    splits = [check_split(s.strip()) for s in args.splits.split(",") if s.strip()]
    if "train" not in splits or "validation" not in splits:
        raise ValueError("both train and validation are required")

    # Every input validated before the first read (resolved paths + pins).
    assert_permitted_inputs(
        [DATA_DIR / f"cricket_data_i7_{s}.parquet" for s in splits] + [CACHE])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading i7 stats cache tables...", flush=True)
    tables = load_cache_tables()
    print(f"  {len(tables['dates'])} snapshot dates; "
          f"prior={np.round(tables['prior'], 4).tolist()}", flush=True)

    print("Building split frames (as-of raw rates via merge_asof)...", flush=True)
    frames = build_frames(tables, splits)

    import transformer_t1 as t1

    manifest = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "frame": str(DATA_DIR.relative_to(ROOT)),
        "splits": splits,
        "sealed": "test split, data/golden and data/forward_holdout are "
                  "never read by this script (fail-closed in check_split / "
                  "split_path)",
        "stats_cache": str(CACHE.relative_to(ROOT)),
        "stats_cache_build": tables["meta"].get("build_timestamp"),
        "class_mapping": {str(k): v for k, v in CLASS_MAPPING.items()},
        "class_names": CLASS_NAMES,
        "ctx_cols": CTX_COLS,
        "ctx_cols_source": "embeddings_e1.py:60 (CTX_COLS), the E2/E15 fair "
                           "control's 4 match-state features",
        "global_prior": tables["prior"].round(10).tolist(),
        "as_of_rule": "largest cache snapshot date <= match_date; cache "
                      "snapshots for date D exclude D's own matches "
                      "(_SQLiteBackend._resolve_date_id semantics)",
    }

    manifest["emit_train_probs"] = sorted(emit)
    fit_reference("eb_ctx", frames, manifest, feat_cols=EB_COLS + CTX_COLS,
                  emit_train_probs="eb_ctx" in emit)
    fit_reference("raw_rate_ctx", frames, manifest,
                  feat_cols=RAW_BAT_COLS + RAW_BOWL_COLS + VENUE_COLS
                  + CTX_COLS,
                  emit_train_probs="raw_rate_ctx" in emit)
    fit_reference("ref_lin_50", frames, manifest,
                  matrix_fn=t1.build_features,
                  feat_names=t1.feature_names(),
                  emit_train_probs="ref_lin_50" in emit)

    (OUT_DIR / "references.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {OUT_DIR}/references.json", flush=True)
    for k in ("eb_ctx", "raw_rate_ctx", "ref_lin_50"):
        print(f"  {k}: validation LL "
              f"{manifest[k]['log_loss']['validation']:.4f}", flush=True)


if __name__ == "__main__":
    main()
