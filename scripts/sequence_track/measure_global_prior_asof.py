#!/usr/bin/env python
"""Measure how far the stats cache's whole-corpus global outcome prior sits
from a prior computed only from pre-cutoff balls.

Context (sequence track stage 1, acceptance D2.4; TODO.md "Global outcome
prior is not as-of-date"): `scripts/build_stats_cache.py` writes one global
prior pi (`_meta.prior_p0 .. prior_pw`) by summing the tracker's FINAL-STATE
outcome counts over the whole corpus, so every empirical-Bayes feature at
every date shrinks toward a base rate that includes post-cutoff balls. This
script quantifies the exposure on a ball frame: pi from the training split
alone versus pi from all three splits, and the implied feature shift
diff * k / (n + k) at a few history sizes. It reads only the `ball_outcome`
and `match_date` columns of the frame's parquets. It never opens data/golden
or data/forward_holdout.

It measures feature displacement only. It says nothing about how any model
responds to that displacement; that is unmeasured.

Usage:
    uv run --no-sync python scripts/sequence_track/measure_global_prior_asof.py \
        [--data-dir <frame dir>] [--cache <stats cache>]
(defaults: manifest role ball_frame_i7; no cache)
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ on the path

OUTCOME_ORDER = [-1, 0, 1, 2, 4, 6]  # wicket, dot, single, two, four, six
OUTCOME_NAME = {-1: "wicket", 0: "dot", 1: "single", 2: "two", 4: "four", 6: "six"}
PRIOR_KEY = {-1: "prior_pw", 0: "prior_p0", 1: "prior_p1", 2: "prior_p2",
             4: "prior_p4", 6: "prior_p6"}
SHIFT_POINTS = [("venue", 200.0, 0), ("venue", 200.0, 200), ("venue", 200.0, 2000),
                ("venue", 200.0, 10000), ("player", 30.0, 0), ("player", 30.0, 30),
                ("player", 30.0, 300)]


def frame_version(data_dir: Path) -> str:
    meta = json.loads((data_dir / ".feature_hash").read_text())
    return str(meta["version"])


def load(data_dir: Path, version: str, split: str) -> pd.DataFrame:
    return pd.read_parquet(data_dir / f"cricket_data_{version}_{split}.parquet",
                           columns=["match_date", "ball_outcome"])


def pi(df: pd.DataFrame) -> dict:
    counts = df["ball_outcome"].value_counts()
    total = int(counts.sum())
    return {o: float(counts.get(o, 0)) / total for o in OUTCOME_ORDER}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="ball frame directory; default: manifest role "
                         "ball_frame_i7 resolved through scripts/artifacts.py")
    ap.add_argument("--cache", type=Path, default=None,
                    help="optional stats cache; its _meta prior is printed "
                         "beside the frame-derived whole-corpus prior")
    ap.add_argument("--json", type=Path, default=None,
                    help="also write the result as JSON")
    args = ap.parse_args(argv)
    if args.data_dir is None:
        from artifacts import artifact_path  # manifest role, not a literal
        args.data_dir = Path(artifact_path("ball_frame_i7"))
    for forbidden in ("golden", "forward_holdout"):
        if forbidden in str(args.data_dir):
            raise SystemExit(f"refusing sealed path: {args.data_dir}")

    version = frame_version(args.data_dir)
    splits = {s: load(args.data_dir, version, s)
              for s in ("train", "validation", "test")}
    for name, df in splits.items():
        print(f"{name:>10}: {len(df):>9,} balls  "
              f"{df['match_date'].min()} -> {df['match_date'].max()}")
    p_train = pi(splits["train"])
    p_all = pi(pd.concat(splits.values(), ignore_index=True))

    cache_prior = None
    if args.cache is not None:
        conn = sqlite3.connect(f"file:{args.cache}?mode=ro", uri=True)
        meta = dict(conn.execute("SELECT key, value FROM _meta"))
        conn.close()
        cache_prior = {o: float(meta[PRIOR_KEY[o]]) for o in OUTCOME_ORDER}

    print("\nglobal prior pi: train-only vs whole corpus"
          + (" vs cache _meta" if cache_prior else ""))
    rows = []
    for o in OUTCOME_ORDER:
        row = {"outcome": OUTCOME_NAME[o], "train_only": p_train[o],
               "whole_corpus": p_all[o], "diff": p_all[o] - p_train[o]}
        if cache_prior:
            row["cache_meta"] = cache_prior[o]
        rows.append(row)
        line = (f"{OUTCOME_NAME[o]:>8}: train {p_train[o]:.6f}  "
                f"all {p_all[o]:.6f}  diff {p_all[o] - p_train[o]:+.6f}")
        if cache_prior:
            line += f"  cache {cache_prior[o]:.10f}"
        print(line)
    max_abs = max(abs(r["diff"]) for r in rows)
    print(f"\nmax |diff| = {max_abs:.6f}")

    print("\nimplied feature shift = max|diff| * k / (n + k)")
    shifts = []
    for kind, k, n in SHIFT_POINTS:
        shift = max_abs * k / (n + k)
        shifts.append({"kind": kind, "k": k, "n": n, "shift": shift})
        print(f"  {kind:>6} k={k:>5.0f} n={n:>6d}: {shift:.6f}")

    if args.json:
        args.json.write_text(json.dumps({
            "data_dir": str(args.data_dir), "frame_version": version,
            "split_ranges": {s: [str(df['match_date'].min()),
                                 str(df['match_date'].max()), int(len(df))]
                             for s, df in splits.items()},
            "prior": rows, "max_abs_diff": max_abs, "shifts": shifts,
            "cache": None if args.cache is None else str(args.cache),
        }, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
