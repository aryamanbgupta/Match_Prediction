"""B3 — Continuous-forecast shrinkage blend (productize E2's finding).

E2's standing result: the sim's only validated prop skill is continuous
score forecasts. This idea fits a per-family convex blend

    forecast = alpha * sim_mean + (1 - alpha) * fair_baseline

on the ball model's VALIDATION split (2024-12-31..2025-06-29, 545
matches — unseen by ball-model training, strictly before the test
period) and evaluates it on the existing canonical D15 test detail
(`models/auto/d15/detail_d15_s43_n261.json`). No sim-engine change, no
re-baseline; `prop_fair_baselines.py` is imported READ-ONLY (loop rule 1).

Stages (run in this order):
  build-val  copy the exact val-split cricsheet JSONs (stems from
             data/xgb_data_v3/cricket_data_v3_validation.parquet
             innings_id, post-B2 joinable ids) -> data/auto/b3/val_matches/
  (external) one prop_backtest run on that dir, same engine config as the
             canonical D15 run (venue-ON default path, stale v1 vector
             calibrator): --n-matches all --n-sims 100 --seed 42
             --ball-calibrator vector
  fit        val detail -> E2 fair baselines -> alpha* per family by grid
             search (0..1 step 0.01) minimizing val row-mean MAE; ties
             broken toward 0.5 then toward lower alpha. VAL-ONLY — no
             test data touched. -> models/auto/b3/alpha.json
  gate       apply frozen alpha.json to the canonical D15 detail; paired
             per-row |err| deltas, cluster bootstrap by match
             (2000 resamples, seed 29 — the E2/B4/B9 statistic).

PRE-COMMITTED GATE (fixed before any val or test result is seen):
  Families = the 5 MAE families with defined E2 fair baselines:
    batter_runs_mae, highest_individual_mae, team_total_fours_mae,
    team_total_sixes_mae, team_first_over_mae
    (batter_fours_mae has no baseline in prop_fair_baselines.py -> excluded).
  GATE 1 (win): >=1 family where BOTH paired deltas are CI-clean < 0:
    d_bs = |blend - y| - |sim - y|   (blend beats sim)
    d_bb = |blend - y| - |base - y|  (blend beats baseline)
  GATE 2 (no-regress): NO family where d_bp = |blend - y| - |best - y|
    is CI-clean > 0, best = parent with the lower test MAE point
    estimate for that family.
  Both -> LANDED; exactly one -> TABLED; none -> FAILED.
  Degenerate alphas (0.0 / 1.0) make blend == a parent; zero deltas are
  by construction neither CI-clean wins nor regressions.

Known caveat, stated pre-run: the v1 vector calibrator was FIT on these
same val balls (E5), so sim forecasts on val carry a small in-sample
optimism vs test -> alpha may lean slightly toward the sim relative to
the test optimum. A 6-parameter marginal correction fit on 124k balls;
expected to be negligible, reported as context (test-optimal alpha is
printed as a diagnostic AFTER the verdict, never used for selection).

Usage:
  uv run python scripts/auto/b3_shrinkage_blend.py build-val
  uv run python scripts/auto/b3_shrinkage_blend.py fit \
      --val-detail models/auto/b3/detail_val_s42_n545.json
  uv run python scripts/auto/b3_shrinkage_blend.py gate \
      --test-detail models/auto/d15/detail_d15_s43_n261.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "models" / "auto" / "b3"
VAL_DIR = REPO / "data" / "auto" / "b3" / "val_matches"
VAL_PARQUET = REPO / "data" / "xgb_data_v3" / "cricket_data_v3_validation.parquet"
ALPHA_JSON = OUT_DIR / "alpha.json"

FAMILIES = [
    "batter_runs_mae",
    "highest_individual_mae",
    "team_total_fours_mae",
    "team_total_sixes_mae",
    "team_first_over_mae",
]

N_BOOT = 2000
BOOT_SEED = 29
ALPHA_GRID = np.round(np.arange(0.0, 1.0001, 0.01), 2)


# ---------------------------------------------------------- pfb (read-only)
def _load_pfb():
    spec = importlib.util.spec_from_file_location(
        "prop_fair_baselines",
        REPO / "scripts" / "sim_eval" / "prop_fair_baselines.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _paired_rows(detail_path: Path) -> dict:
    """family -> [{p_sim, p_base, y, mid}] via the E2 construction."""
    pfb = _load_pfb()
    cache = REPO / "models" / "prop_fair_baseline_corpus.pkl"
    logs = pickle.load(open(cache, "rb"))
    asof = pfb.AsOf(logs)
    detail = json.load(open(detail_path))
    print(f"detail: {detail_path.name} (n={len(detail)} matches)")
    return pfb.baseline_rows(detail, asof)


# --------------------------------------------------------------- bootstrap
def _cluster_boot(rows, fn, n_boot=N_BOOT, seed=BOOT_SEED):
    """Mean of fn(row) + 95% CI, cluster bootstrap by match (mid)."""
    by_match = defaultdict(list)
    for r in rows:
        by_match[r["mid"]].append(fn(r))
    mids = list(by_match)
    per = [np.array(by_match[m]) for m in mids]
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        idx = rng.choice(len(mids), size=len(mids), replace=True)
        means.append(np.mean(np.concatenate([per[i] for i in idx])))
    means = np.array(means)
    point = float(np.mean(np.concatenate(per)))
    return point, float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------------------------------------------------------- build-val
def cmd_build_val(_args):
    import pandas as pd
    df = pd.read_parquet(VAL_PARQUET, columns=["innings_id", "match_date"])
    stems = sorted(df["innings_id"].astype(str).str.split("_").str[-1].unique())
    print(f"val split: {len(stems)} matches, "
          f"{df['match_date'].min()}..{df['match_date'].max()}")
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for s in stems:
        src = REPO / "data" / "t20s_json" / f"{s}.json"
        if not src.exists():
            raise SystemExit(f"missing source file {src}")
        shutil.copy2(src, VAL_DIR / src.name)
        n += 1
    print(f"copied {n} val JSONs -> {VAL_DIR}")


# ---------------------------------------------------------------------- fit
def cmd_fit(args):
    paired = _paired_rows(Path(args.val_detail))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {"val_detail": str(args.val_detail), "families": {}}
    for fam in FAMILIES:
        rows = paired.get(fam, [])
        if not rows:
            raise SystemExit(f"no val rows for family {fam}")
        ps = np.array([r["p_sim"] for r in rows], dtype=float)
        pb = np.array([r["p_base"] for r in rows], dtype=float)
        y = np.array([r["y"] for r in rows], dtype=float)
        maes = np.array([np.mean(np.abs(a * ps + (1 - a) * pb - y))
                         for a in ALPHA_GRID])
        best = maes.min()
        cand = [a for a, m in zip(ALPHA_GRID, maes) if m == best]
        # ties: toward 0.5, then lower alpha (deterministic)
        alpha = sorted(cand, key=lambda a: (abs(a - 0.5), a))[0]
        i = int(np.where(ALPHA_GRID == alpha)[0][0])
        rec = {
            "alpha": float(alpha),
            "n_rows": len(rows),
            "n_matches": len({r["mid"] for r in rows}),
            "val_mae_sim": float(np.mean(np.abs(ps - y))),
            "val_mae_base": float(np.mean(np.abs(pb - y))),
            "val_mae_blend": float(maes[i]),
            "val_mae_pm0p1": [float(maes[max(i - 10, 0)]),
                              float(maes[min(i + 10, len(maes) - 1)])],
        }
        out["families"][fam] = rec
        print(f"{fam:28s} alpha*={alpha:.2f}  n={rec['n_rows']:5d}  "
              f"val MAE sim {rec['val_mae_sim']:.3f} / base "
              f"{rec['val_mae_base']:.3f} / blend {rec['val_mae_blend']:.3f}  "
              f"(±0.1: {rec['val_mae_pm0p1'][0]:.3f}/{rec['val_mae_pm0p1'][1]:.3f})")
    json.dump(out, open(ALPHA_JSON, "w"), indent=2)
    print(f"alpha -> {ALPHA_JSON}")


# --------------------------------------------------------------------- gate
def cmd_gate(args):
    alpha_cfg = json.load(open(ALPHA_JSON))
    paired = _paired_rows(Path(args.test_detail))
    print("\n=== B3 GATE (pre-committed; see module docstring) ===")
    gate1_wins, gate2_regressions = [], []
    report = {}
    for fam in FAMILIES:
        rows = paired.get(fam, [])
        a = alpha_cfg["families"][fam]["alpha"]
        for r in rows:
            r["p_blend"] = a * r["p_sim"] + (1 - a) * r["p_base"]
        mae_sim, _, _ = _cluster_boot(rows, lambda r: abs(r["p_sim"] - r["y"]))
        mae_base, _, _ = _cluster_boot(rows, lambda r: abs(r["p_base"] - r["y"]))
        mae_blend, _, _ = _cluster_boot(rows, lambda r: abs(r["p_blend"] - r["y"]))
        d_bs = _cluster_boot(rows, lambda r: abs(r["p_blend"] - r["y"])
                             - abs(r["p_sim"] - r["y"]))
        d_bb = _cluster_boot(rows, lambda r: abs(r["p_blend"] - r["y"])
                             - abs(r["p_base"] - r["y"]))
        best_parent = "sim" if mae_sim <= mae_base else "base"
        d_bp = d_bs if best_parent == "sim" else d_bb
        win = d_bs[2] < 0 and d_bb[2] < 0
        regress = d_bp[1] > 0
        if win:
            gate1_wins.append(fam)
        if regress:
            gate2_regressions.append(fam)
        report[fam] = {
            "alpha": a, "n_rows": len(rows),
            "test_mae_sim": mae_sim, "test_mae_base": mae_base,
            "test_mae_blend": mae_blend, "best_parent": best_parent,
            "d_blend_minus_sim": d_bs, "d_blend_minus_base": d_bb,
            "d_blend_minus_best": d_bp,
            "win": win, "regress": regress,
        }
        print(f"\n{fam} (alpha={a:.2f}, n={len(rows)})")
        print(f"  test MAE: sim {mae_sim:.3f} | base {mae_base:.3f} | "
              f"blend {mae_blend:.3f}  (best parent: {best_parent})")
        print(f"  d(blend-sim)  {d_bs[0]:+.4f} [{d_bs[1]:+.4f}, {d_bs[2]:+.4f}]")
        print(f"  d(blend-base) {d_bb[0]:+.4f} [{d_bb[1]:+.4f}, {d_bb[2]:+.4f}]")
        print(f"  win={win}  regress={regress}")

    gate1 = len(gate1_wins) >= 1
    gate2 = len(gate2_regressions) == 0
    verdict = ("LANDED" if gate1 and gate2
               else "TABLED" if gate1 or gate2 else "FAILED")
    print("\n--- verdict ---")
    print(f"GATE 1 (blend beats BOTH parents CI-clean on >=1 family): "
          f"{'MET' if gate1 else 'NOT MET'}  {gate1_wins}")
    print(f"GATE 2 (no family regresses CI-clean vs best parent): "
          f"{'MET' if gate2 else 'NOT MET'}  {gate2_regressions}")
    print(f"VERDICT: {verdict}")

    # context only, computed after the verdict: test-optimal alpha
    print("\ncontext (post-verdict diagnostics, not used for selection):")
    for fam in FAMILIES:
        rows = paired[fam]
        ps = np.array([r["p_sim"] for r in rows], dtype=float)
        pb = np.array([r["p_base"] for r in rows], dtype=float)
        y = np.array([r["y"] for r in rows], dtype=float)
        maes = np.array([np.mean(np.abs(a * ps + (1 - a) * pb - y))
                         for a in ALPHA_GRID])
        a_test = float(ALPHA_GRID[int(np.argmin(maes))])
        report[fam]["test_optimal_alpha"] = a_test
        print(f"  {fam:28s} val alpha {report[fam]['alpha']:.2f} -> "
              f"test-optimal {a_test:.2f}")

    report["_verdict"] = {"gate1": gate1, "gate1_wins": gate1_wins,
                          "gate2": gate2,
                          "gate2_regressions": gate2_regressions,
                          "verdict": verdict}
    out = OUT_DIR / "gate_numbers.json"
    json.dump(report, open(out, "w"), indent=2)
    print(f"\nnumbers -> {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-val")
    p_fit = sub.add_parser("fit")
    p_fit.add_argument("--val-detail",
                       default=str(OUT_DIR / "detail_val_s42_n545.json"))
    p_gate = sub.add_parser("gate")
    p_gate.add_argument("--test-detail",
                        default=str(REPO / "models/auto/d15/detail_d15_s43_n261.json"))
    args = ap.parse_args()
    {"build-val": cmd_build_val, "fit": cmd_fit, "gate": cmd_gate}[args.cmd](args)


if __name__ == "__main__":
    main()
