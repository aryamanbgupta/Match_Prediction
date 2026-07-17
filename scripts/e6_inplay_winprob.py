"""E6 — Direct in-play win-probability model (match-level supervision on ball states).

The MLC 2025 in-play test (mlc/reports/mlc_inplay_winprob.md) showed the
ball-by-ball *simulator* loses to "chase math + pre-match team rating"
(LL 0.4936 vs sim 0.6042 on 88 chase states). Same lesson as the
winner-market: direct supervision beats rolling a per-ball generative
model forward. This experiment trains P(batting team wins | ball state)
directly on every delivery of the corpus — the in-play analogue of the
match-level direct model.

Nested model ladder (each adds information; all XGBoost-hist, early stop
on the val split, identical hyperparams — so deltas are feature value,
not tuning):
  B0 prior_only    pre-match team strength only (static through match)
  B1 resource_only chase math only (score/wickets/balls/target)
  B2 fair_blend    B0 + B1 — the baseline that beat the sim at MLC
  E6 full          B2 + crease/momentum/venue state (who is in, form,
                   partnership, recent scoring, pressure)

Labels: y = 1 iff the batting team of that innings won the match.
Excluded: no-result / tie / D-L-decided matches (resource math breaks).

Eval on the test split (2025-07-01 → 2026-04-16, 1,533 matches with
winners): overall per-ball LL with cluster bootstrap by match, plus
per-checkpoint readout (balls 30/60/90 of each innings).

Decision gate (a priori): keep iff full beats fair_blend on overall test
LL with the match-clustered 95% CI of ΔLL excluding 0.

Usage:
    uv run python scripts/e6_inplay_winprob.py            # full run
    uv run python scripts/e6_inplay_winprob.py --quick    # 200k-row train subsample
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from xgboost import XGBClassifier

REPO = Path(__file__).resolve().parent.parent
BALL_DIR = REPO / "data" / "xgb_data_v3"
CORPUS = REPO / "data" / "t20s_json"
WINNER_CACHE = REPO / "models" / "match_winner_map.json"
MODEL_OUT = REPO / "models" / "inplay_winprob_v1"

PRIOR = ["batting_team_elo", "bowling_team_elo", "elo_diff",
         "team_batting_avg", "team_batting_sr", "team_bowling_avg",
         "team_bowling_econ", "is_batting_first", "venue_chase_win_pct"]
RESOURCE = ["inning_idx", "score", "wickets", "balls_bowled",
            "balls_remaining", "wickets_in_hand", "run_rate",
            "chase_target", "run_rate_required", "lead_gap"]
EXTRAS = ["batsman_avg", "batsman_sr", "batsman_recent_avg",
          "batsman_recent_sr", "non_striker_sr", "partnership_runs",
          "last_10_balls_runs", "last_30_balls_runs",
          "balls_since_boundary", "dot_percentage_recent",
          "boundary_percentage_recent", "striker_elo", "bowler_elo_rating",
          "venue_avg_score", "is_powerplay", "is_death_overs",
          "pressure_cooker_index"]
META = ["innings_id", "inning_idx", "match_date", "balls_bowled"]

LADDER = {
    "prior_only": PRIOR,
    "resource_only": RESOURCE,
    "fair_blend": PRIOR + RESOURCE,
    "full": PRIOR + RESOURCE + EXTRAS,
}

PARAMS = dict(objective="binary:logistic", eval_metric="logloss",
              tree_method="hist", n_estimators=600, max_depth=6,
              learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
              early_stopping_rounds=30, random_state=29)


def build_winner_map() -> dict:
    """(date, venue) -> [candidate matches] for clean male T20s with an
    outright winner (no tie / no-result / D-L). Each candidate carries
    the winner, per-innings batting teams, and the innings-1 legal-ball
    count + total used to disambiguate same-day same-venue doubleheaders.

    NOTE: historically the parquet's innings_id suffix was
    `hash(json) % 100000` — salted per process and collision-prone — so
    the only reliable join was (match_date, venue) + innings-1 shape.
    Fixed 2026-07-16 (B2): parquets built since then carry the cricsheet
    filename stem as the suffix, so `innings_id.split("_", 1)[1]` now
    joins directly to `data/t20s_json/<stem>.json`. This workaround is
    kept for compatibility with pre-B2 parquets.
    """
    if WINNER_CACHE.exists():
        return json.load(open(WINNER_CACHE))
    out = defaultdict(list)
    for f in sorted(CORPUS.glob("*.json")):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        info = j.get("info", {})
        if info.get("gender") != "male":
            continue
        outcome = info.get("outcome", {})
        winner = outcome.get("winner")
        if not winner or outcome.get("method"):
            continue  # tie / no result / D-L
        inns = j.get("innings", [])
        if len(inns) < 2:
            continue
        i1_balls, i1_runs = 0, 0
        for ov in inns[0].get("overs", []):
            for d in ov.get("deliveries", []):
                ex = d.get("extras", {}) or {}
                if "wides" not in ex and "noballs" not in ex:
                    i1_balls += 1
                i1_runs += d["runs"]["total"]
        key = f"{info['dates'][0]}|{info.get('venue', '?')}"
        out[key].append({"winner": winner,
                         "inn1": inns[0].get("team"),
                         "inn2": inns[1].get("team"),
                         "i1_balls": i1_balls, "i1_runs": i1_runs})
    out = dict(out)
    json.dump(out, open(WINNER_CACHE, "w"))
    print(f"winner map: {sum(len(v) for v in out.values())} clean matches "
          f"({len(out)} date×venue keys) -> {WINNER_CACHE.name}")
    return out


def load_split(split: str, winners: dict) -> pd.DataFrame:
    cols = sorted(set(PRIOR + RESOURCE + EXTRAS + META + ["venue", "is_wide",
                                                          "is_noball"]))
    df = pd.read_parquet(BALL_DIR / f"cricket_data_v3_{split}.parquet",
                         columns=cols)
    # hash-suffix is only unique *within* a (date, venue); combine.
    df["match_key"] = (df["innings_id"].str.split("_", n=1).str[1] + "|" +
                       df["match_date"].astype(str) + "|" + df["venue"])
    df["dv"] = df["match_date"].astype(str) + "|" + df["venue"]

    # innings-1 legal-ball count per match_key for doubleheader disambig
    inn1 = df[df["inning_idx"].astype(int) == 1]
    legal = (~inn1["is_wide"].astype(bool)) & (~inn1["is_noball"].astype(bool))
    i1_balls = inn1[legal].groupby("match_key").size()

    lab = {}
    n_multi, n_drop = 0, 0
    for mk, dv in df[["match_key", "dv"]].drop_duplicates().itertuples(index=False):
        cands = winners.get(dv)
        if not cands:
            n_drop += 1
            continue
        if len(cands) == 1:
            lab[mk] = cands[0]
            continue
        n_multi += 1
        nb = int(i1_balls.get(mk, -1))
        best = min(cands, key=lambda c: abs(c["i1_balls"] - nb))
        if abs(best["i1_balls"] - nb) <= 2:
            lab[mk] = best
        else:
            n_drop += 1
    df = df[df["match_key"].isin(lab)].copy()
    inn = df["inning_idx"].astype(int)
    w = df["match_key"].map(lambda k: lab[k]["winner"])
    t1 = df["match_key"].map(lambda k: lab[k]["inn1"])
    t2 = df["match_key"].map(lambda k: lab[k]["inn2"])
    batting = np.where(inn == 1, t1, t2)
    df["y"] = (batting == w).astype(int)
    print(f"  {split}: joined {df.match_key.nunique()} matches "
          f"({n_multi} doubleheader keys, {n_drop} dropped)")
    return df


def cluster_ci_delta(ll_a: np.ndarray, ll_b: np.ndarray,
                     match_keys: np.ndarray, n_boot=1000, seed=29):
    """95% CI of mean(ll_a - ll_b), bootstrap clustered by match."""
    rng = np.random.default_rng(seed)
    delta = ll_a - ll_b
    by_match = defaultdict(list)
    for d, mk in zip(delta, match_keys):
        by_match[mk].append(d)
    sums = np.array([np.sum(v) for v in by_match.values()])
    cnts = np.array([len(v) for v in by_match.values()])
    n = len(sums)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stats.append(sums[idx].sum() / cnts[idx].sum())
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def per_ball_ll(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    winners = build_winner_map()
    train = load_split("train", winners)
    val = load_split("validation", winners)
    test = load_split("test", winners)
    if args.quick:
        train = train.sample(200_000, random_state=29)
    print(f"rows: train {len(train):,}  val {len(val):,}  test {len(test):,}")
    print(f"matches: train {train.match_key.nunique():,}  "
          f"val {val.match_key.nunique():,}  test {test.match_key.nunique():,}")
    print(f"base rate (batting side wins): train {train.y.mean():.3f}  "
          f"test {test.y.mean():.3f}")

    results, test_probs = {}, {}
    for name, feats in LADDER.items():
        m = XGBClassifier(**PARAMS)
        m.fit(train[feats], train["y"],
              eval_set=[(val[feats], val["y"])], verbose=False)
        p_te = m.predict_proba(test[feats])[:, 1]
        test_probs[name] = p_te
        results[name] = {
            "best_iter": int(m.best_iteration),
            "val_ll": float(log_loss(val["y"], m.predict_proba(val[feats])[:, 1])),
            "test_ll": float(log_loss(test["y"], p_te)),
            "test_auc": float(roc_auc_score(test["y"], p_te)),
        }
        print(f"{name:14s} best_iter {m.best_iteration:>3d}  "
              f"val LL {results[name]['val_ll']:.4f}  "
              f"test LL {results[name]['test_ll']:.4f}  "
              f"AUC {results[name]['test_auc']:.4f}")
        # fair_blend is the production artifact (the E6 gate showed the
        # crease/momentum extras add nothing — keep the simpler model);
        # full is saved alongside for reference.
        if name in ("fair_blend", "full") and not args.quick:
            out = MODEL_OUT if name == "fair_blend" else \
                MODEL_OUT.parent / f"{MODEL_OUT.name}_full"
            out.mkdir(parents=True, exist_ok=True)
            import joblib
            joblib.dump(m, out / "model.pkl")
            (out / "feature_columns.txt").write_text("\n".join(feats))

    # gate: full vs fair_blend, match-clustered CI
    y = test["y"].values
    mk = test["match_key"].values
    ll_full = per_ball_ll(y, test_probs["full"])
    ll_fair = per_ball_ll(y, test_probs["fair_blend"])
    lo, hi = cluster_ci_delta(ll_full, ll_fair, mk)
    print(f"\nGATE  Δ test LL (full − fair_blend) = "
          f"{ll_full.mean() - ll_fair.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
          f"  -> {'PASS' if hi < 0 else 'FAIL'}")

    # checkpoint table
    print("\n-- per-checkpoint test LL (balls into innings) --")
    print(f"{'model':14s}" + "".join(
        f" inn{i}@{b:>3d}" for i in (1, 2) for b in (30, 60, 90)))
    for name in LADDER:
        cells = []
        for i in (1, 2):
            for b in (30, 60, 90):
                sel = (test["inning_idx"].astype(int) == i) & \
                      (test["balls_bowled"] == b)
                cells.append(f" {log_loss(y[sel], np.clip(test_probs[name][sel], 1e-9, 1-1e-9), labels=[0,1]):.4f}"
                             if sel.sum() > 50 else "      —")
        print(f"{name:14s}" + "".join(cells))

    summary = {"results": results,
               "gate_delta_ci": [lo, hi],
               "n_test_matches": int(test.match_key.nunique())}
    json.dump(summary, open(REPO / "eval_out_e6_summary.json", "w"), indent=2)
    print(f"\nsummary -> eval_out_e6_summary.json")


if __name__ == "__main__":
    main()
