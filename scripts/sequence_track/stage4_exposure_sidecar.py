# manifest-exempt: embeddings-ladder experimental artifact namespace
"""Stage 4 exposure sidecar for rung 4d (uncertainty-aware pooling).

For every ball in the i7 frame, writes how much evidence the EB-shrunk
player features were actually built from:

  {batter,bowler}_N_asof        total as-of legal-ball count
  {batter,bowler}_asof_c*       the six per-class as-of counts
  {batter,bowler}_var_p*        Dirichlet posterior spread of the shrunk
                                rate at k=30: p_c(1-p_c)/(N+k+1), with
                                p_c = (n_c + k*pi_c)/(N+k)
  {batter,bowler}_recent_N      the cache's recent_* ball count

Why: the frame carries only the shrunk point estimates p_c, so a model
cannot tell a 20-ball rookie from a 20,000-ball veteran with the same
mean. 4d needs the exposure and the posterior spread as explicit inputs.

The sidecar is ROW-ALIGNED to the frame (same order, asserted equal
length) and carries innings_id/ball_idx so alignment is verifiable.
As-of semantics, the merge_asof machinery and the sealed-split guard are
imported from stage4_references.py: this script builds train and
validation only and fails closed on the test split and on any
data/golden or data/forward_holdout path.

Artifacts: models/embeddings/stage4/exposure/{split}.parquet + manifest.

Usage:
    uv run --no-sync python scripts/sequence_track/stage4_exposure_sidecar.py
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from stage4_guard import assert_permitted_inputs  # noqa: E402
from stage4_references import (  # noqa: E402
    ALLOWED_SPLITS, CACHE, COUNT_SUFFIXES, ROOT, asof_counts, check_split,
    load_cache_tables, resolve_date_ids, sha256, split_path,
)

OUT_DIR = ROOT / "models/embeddings/stage4/exposure"
K_SHRINK = 30  # k_player from the production hierarchical-shrinkage config.
COUNT_COLS = ["c0", "c1", "c2", "c4", "c6", "cw"]


def side_block(counts: pd.DataFrame, side: str, prior: np.ndarray) -> dict:
    c = counts[COUNT_COLS].to_numpy(dtype=np.float64)
    n = c.sum(axis=1)
    # Shrunk posterior mean and its Dirichlet(n + k*pi) marginal variance.
    p = (c + K_SHRINK * prior) / (n + K_SHRINK)[:, None]
    var = p * (1.0 - p) / (n + K_SHRINK + 1.0)[:, None]
    block = {f"{side}_N_asof": n.astype(np.int64),
             f"{side}_recent_N": counts["recent_balls"].to_numpy(np.int64)}
    for i, s in enumerate(COUNT_SUFFIXES):
        block[f"{side}_asof_c{s[1:]}"] = c[:, i].astype(np.int64)
        block[f"{side}_var_{s}"] = var[:, i].astype(np.float32)
    return block


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=",".join(ALLOWED_SPLITS),
                    help="comma-separated; only train,validation are allowed")
    args = ap.parse_args()
    splits = [check_split(s.strip())
              for s in args.splits.split(",") if s.strip()]
    # Every input validated before the first read (resolved paths + pins).
    assert_permitted_inputs([split_path(s) for s in splits] + [CACHE])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables = load_cache_tables()
    prior = tables["prior"]
    manifest = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "frame": "data/xgb_data_i7",
        "stats_cache": "models/player_stats_cache_i7.sqlite",
        "k": K_SHRINK,
        "prior": prior.round(10).tolist(),
        "variance_formula": "var_c = p_c(1-p_c)/(N+k+1), "
                            "p_c = (n_c + k*pi_c)/(N+k)",
        "as_of_rule": "largest cache snapshot date <= match_date; snapshots "
                      "for date D exclude D's own matches",
        "recent_source": "batting.recent_balls / bowling.recent_balls_bowled "
                         "(ball counts over the cache's recency window)",
        "sealed": "test split, data/golden and data/forward_holdout are never "
                  "read by this script; inputs are allowlisted by resolved "
                  "path + content pin in stage4_guard.assert_permitted_inputs "
                  "before any read",
        "splits": {},
    }

    for name in splits:
        src = split_path(name)
        frame = pd.read_parquet(
            src, columns=["innings_id", "ball_idx", "match_date",
                          "batter_id", "bowler_id"])
        frame["_did"] = resolve_date_ids(frame["match_date"], tables["dates"])
        out = {"innings_id": frame["innings_id"].to_numpy(),
               "ball_idx": frame["ball_idx"].to_numpy()}
        for side in ("batter", "bowler"):
            out.update(side_block(asof_counts(frame, side, tables),
                                  side, prior))
        sidecar = pd.DataFrame(out)
        assert len(sidecar) == len(frame), "sidecar/frame row mismatch"

        path = OUT_DIR / f"{name}.parquet"
        sidecar.to_parquet(path, index=False)
        manifest["splits"][name] = {
            "rows": int(len(sidecar)),
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            "batter_N_asof_median": float(sidecar["batter_N_asof"].median()),
            "bowler_N_asof_median": float(sidecar["bowler_N_asof"].median()),
            "batter_N_zero_frac": round(
                float((sidecar["batter_N_asof"] == 0).mean()), 4),
            "bowler_N_zero_frac": round(
                float((sidecar["bowler_N_asof"] == 0).mean()), 4),
        }
        print(f"{name}: {len(sidecar):,} rows -> {path}", flush=True)
        if name == "validation":
            print(sidecar.head(5).to_string(), flush=True)

    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {OUT_DIR}/manifest.json", flush=True)


if __name__ == "__main__":
    main()
