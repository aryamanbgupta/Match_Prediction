"""D9 — decayed margin-aware team-results ELO (replacement test for the
win_rate features), paired 5-seed eval.

`win_rate_diff` (crude last-10) is the model's highest-gain feature; a
time-decayed, margin-aware team ELO is a strictly richer estimator of the
same construct and is match-level BY NATURE (immune to the lineup-aggregation
collapse that killed M3–M5). The materializer (opt-in `--team-elo`) emits a
6-variant grid (K ∈ {16,32} × margin {off,on} × decay {off, HL=365d}) in one
pass; this harness runs the whole pre-committed decision pipeline.

=== PRE-COMMITTED DECISION RULES (written before any result exists) ===

Stage 0 — unit check (hard-fail): TeamEloTracker math on scripted fixtures
  (expected-score symmetry, zero-sum update, margin multiplier caps and the
  narrowest-win fallback, decay half-life arithmetic, pre-match query purity).

Stage 1 — parity: the d9 parquet's 51 shared columns (v2_clean schema) must
  be EXACTLY equal to data/xgb_match_data_v2_clean on train/validation/test.
  If parity holds, A1's logged per-seed base numbers are the paired control
  (D7 verified same-session reproduction of all 5 seeds). If parity fails,
  a fresh 5-seed base arm is trained on the 51-col subset and becomes the
  control (program.md same-session-baseline rule).

Stage 2 — dual correlation check (train split, before any training):
  For each variant v with diff feature d_v:
    R_v  = max over existing 43 numeric features f of |corr(d_v, f)|
    T_v  = |corr(d_v, team1_wins)|;  T_wr = |corr(win_rate_diff, team1_wins)|
  ADD framing viable(v):     (R_v <= 0.5 AND T_v >= 0.03)  OR
                             (R_v > 0.5  AND T_v >= 1.2 * |corr(argmax_f, y)|)
  REPLACE framing viable(v): T_v >= max(0.03, 1.1 * T_wr)
  Cheap exit (verdict FAILED, à la M5) iff NO variant is viable under EITHER
  framing. Otherwise proceed to both arms (A5/A6 precedent: report the check
  verbatim either way).

Stage 3 — variant selection (VAL ONLY, D8 discipline — no iteration-set
  shopping): train seed-29 models for all 6 variants on BOTH arms; for each
  arm the chosen variant is argmin val_log_loss. Iteration-set numbers are
  not read at this stage.

Stage 4 — paired 5-seed (A1 seeds {29,7,13,42,101}), recipe A, both arms at
  their chosen variants. Gate metrics from the >=$50k slice; >=$100k context.

Verdict (program.md): paired arm-minus-base MEAN on >=$50k vs the fresh
  baseline mean (A1: LL 0.6318 / ROI +20.56), retrain floors 0.007 LL /
  2.3pp ROI. Per-arm verdicts: BOTH clear -> LANDED; exactly one -> TABLED;
  none -> FAILED. The idea's verdict = best over the two arms
  (LANDED > TABLED > FAILED); the driving arm is reported.

Arm schemas (column order is load-bearing under colsample):
  add_<v>:  v2_clean 51 cols + [team1_elo_v, team2_elo_v, team_elo_diff_v]
  repl_<v>: v2_clean 51 cols - [team1_win_rate_last_10, team2_win_rate_last_10,
            win_rate_diff] + the same 3 elo cols appended at the end.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from materialize_match_features import TEAM_ELO_GRID, TeamEloTracker  # noqa: E402

SEEDS = [29, 7, 13, 42, 101]
V2_CLEAN = ROOT / "data/xgb_match_data_v2_clean"
D9_DATA = ROOT / "data/auto/d9"
ARMS_DIR = D9_DATA / "arms"
MODELS = ROOT / "models/auto/d9"
SIM_ENVELOPE = ROOT / "eval_out/phase5_hier/hier_all_20260425_165622.json"
ODDS = ROOT / "betting_odds_polymarket.json"
SPLITS = ["train", "validation", "test"]

VARIANTS = [name for name, _ in TEAM_ELO_GRID]
REPL_DROP = ["team1_win_rate_last_10", "team2_win_rate_last_10",
             "win_rate_diff"]

# A1 logged per-seed >=$50k results (research/reports/auto/A1.md) — the
# paired control when Stage-1 parity holds. Mean LL 0.6318 / ROI +20.56.
A1_LOGGED = {29: (0.6231, 23.96), 7: (0.6373, 18.22), 13: (0.6293, 21.30),
             42: (0.6292, 18.73), 101: (0.6399, 20.61)}
A1_MEAN_LL = 0.6318
A1_MEAN_ROI = 20.56
FLOOR_LL = 0.007
FLOOR_ROI = 2.3


# ---------------------------------------------------------------- stage 0
def unit_check() -> None:
    d0 = datetime(2020, 1, 1)
    # Expected-score symmetry + zero-sum at equal ratings, margin off.
    t = TeamEloTracker(k=32.0, margin=False)
    t.update("A", "B", d0, True)
    assert abs(t.ratings["A"][0] - 1516.0) < 1e-9, t.ratings["A"]
    assert abs(t.ratings["B"][0] - 1484.0) < 1e-9
    # Zero-sum holds after asymmetric ratings too.
    t.update("A", "B", datetime(2020, 1, 8), True)
    tot = t.ratings["A"][0] + t.ratings["B"][0]
    assert abs(tot - 3000.0) < 1e-9
    # Winner gains less when heavily favored.
    gain2 = t.ratings["A"][0] - 1516.0
    assert 0 < gain2 < 16.0
    # Margin multiplier: caps + narrowest-win fallback.
    mm = TeamEloTracker._margin_mult
    assert mm(0, None) == 0.5
    assert mm(60, None) == 2.0
    assert mm(200, None) == 2.0          # cap
    assert mm(30, None) == 1.25          # 30-run win = mid scale
    assert mm(None, 8) == 2.0
    assert mm(None, 10) == 2.0           # cap
    assert mm(None, 4) == 1.25           # 4-wkt win = mid scale
    assert mm(None, None) == 0.5         # super-over: narrowest
    # Margin-aware update scales delta by the multiplier exactly.
    tm = TeamEloTracker(k=32.0, margin=True)
    tm.update("A", "B", d0, True, margin_runs=60)
    assert abs(tm.ratings["A"][0] - 1532.0) < 1e-9   # 16 * 2.0
    # Decay: half-life arithmetic, pull toward base, query purity.
    td = TeamEloTracker(k=32.0, margin=False, half_life_days=365.0)
    td.update("A", "B", d0, True)                     # A -> 1516
    r365 = td.get("A", datetime(2021, 1, 1, 0, 0))    # 366 days later
    expect = 1500.0 + 16.0 * 0.5 ** (366 / 365.0)
    assert abs(r365 - expect) < 1e-9, (r365, expect)
    assert td.get("A", d0) == td.ratings["A"][0]      # as_of <= last: no decay
    before = td.ratings["A"]
    td.get("A", datetime(2022, 1, 1))
    assert td.ratings["A"] == before                   # get() never mutates
    # Unknown team = base rating.
    assert td.get("ZZZ", d0) == 1500.0
    print("  [stage0] TeamEloTracker unit check: ALL PASS")


# ---------------------------------------------------------------- stage 1
def verify_parity() -> bool:
    ok = True
    shared_cols = None
    for split in SPLITS:
        a = pd.read_parquet(V2_CLEAN / f"{split}.parquet")
        b = pd.read_parquet(D9_DATA / f"{split}.parquet")
        shared_cols = list(a.columns)
        if len(a) != len(b):
            print(f"  [stage1] {split}: ROW COUNT differs "
                  f"({len(a)} vs {len(b)}) -> parity FAIL")
            ok = False
            continue
        missing = [c for c in shared_cols if c not in b.columns]
        if missing:
            print(f"  [stage1] {split}: missing cols {missing} -> parity FAIL")
            ok = False
            continue
        bad = []
        for c in shared_cols:
            av, bv = a[c], b[c]
            if pd.api.types.is_numeric_dtype(av):
                eq = np.array_equal(av.to_numpy(), bv.to_numpy())
            else:
                eq = av.equals(bv)
            if not eq:
                bad.append(c)
        if bad:
            print(f"  [stage1] {split}: {len(bad)} cols differ: {bad[:8]}")
            ok = False
        else:
            print(f"  [stage1] {split}: all {len(shared_cols)} shared cols "
                  f"EXACTLY equal ({len(a):,} rows)")
    return ok


# ---------------------------------------------------------------- stage 2
def corr_check() -> dict:
    train = pd.read_parquet(D9_DATA / "train.parquet")
    base_cols = [c for c in pd.read_parquet(
        V2_CLEAN / "train.parquet").columns]
    meta = {"match_id", "cricsheet_id", "match_date", "team1", "team2",
            "venue", "competition_tier", "team1_wins"}
    existing = [c for c in base_cols if c not in meta]
    y = train["team1_wins"].astype(float)
    t_wr = abs(np.corrcoef(train["win_rate_diff"], y)[0, 1])
    print(f"  [stage2] T_wr = |corr(win_rate_diff, y)| = {t_wr:.4f}")
    out = {"T_wr": t_wr, "variants": {}}
    any_viable = False
    for v in VARIANTS:
        d = train[f"team_elo_diff_{v}"].astype(float)
        rs = {f: abs(np.corrcoef(d, train[f].astype(float))[0, 1])
              for f in existing}
        f_max = max(rs, key=rs.get)
        R = rs[f_max]
        T = abs(np.corrcoef(d, y)[0, 1])
        T_fmax = abs(np.corrcoef(train[f_max].astype(float), y)[0, 1])
        add_ok = (R <= 0.5 and T >= 0.03) or (R > 0.5 and T >= 1.2 * T_fmax)
        repl_ok = T >= max(0.03, 1.1 * t_wr)
        any_viable = any_viable or add_ok or repl_ok
        out["variants"][v] = {
            "R_max": R, "R_argmax": f_max, "T": T, "T_argmax_feat": T_fmax,
            "add_viable": bool(add_ok), "replace_viable": bool(repl_ok),
        }
        print(f"  [stage2] {v:>9}: T={T:.4f}  R_max={R:.3f} ({f_max}, "
              f"its T={T_fmax:.4f})  ADD={'OK' if add_ok else 'no'}  "
              f"REPL={'OK' if repl_ok else 'no'}")
    out["any_viable"] = bool(any_viable)
    print(f"  [stage2] any framing viable: {any_viable}"
          + ("" if any_viable else "  -> CHEAP EXIT (FAILED)"))
    return out


# ---------------------------------------------------------------- arms
def build_arm_parquets() -> None:
    base_cols = list(pd.read_parquet(V2_CLEAN / "train.parquet").columns)
    for split in SPLITS:
        full = pd.read_parquet(D9_DATA / f"{split}.parquet")
        for v in VARIANTS:
            elo_cols = [f"team1_elo_{v}", f"team2_elo_{v}",
                        f"team_elo_diff_{v}"]
            add_dir = ARMS_DIR / f"add_{v}"
            repl_dir = ARMS_DIR / f"repl_{v}"
            add_dir.mkdir(parents=True, exist_ok=True)
            repl_dir.mkdir(parents=True, exist_ok=True)
            full[base_cols + elo_cols].to_parquet(
                add_dir / f"{split}.parquet", index=False)
            repl_cols = [c for c in base_cols if c not in REPL_DROP]
            full[repl_cols + elo_cols].to_parquet(
                repl_dir / f"{split}.parquet", index=False)
    print(f"  [arms] wrote {2 * len(VARIANTS)} arm datasets under {ARMS_DIR}")


# ---------------------------------------------------------------- training
def run(cmd):
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def train_one(arm: str, seed: int) -> dict:
    """Train (trainer defaults = A1/M7 config) on ARMS_DIR/<arm>; return
    train_metrics.json contents."""
    mdir = MODELS / f"{arm}_seed{seed}"
    run(["uv", "run", "python", "scripts/xgboost_match_v1.py",
         "--cmd", "both", "--data-dir", str(ARMS_DIR / arm),
         "--model-dir", str(mdir), "--seed", str(seed)])
    return json.loads((mdir / "train_metrics.json").read_text())


def eval_one(arm: str, seed: int) -> dict:
    mdir = MODELS / f"{arm}_seed{seed}"
    evd = mdir / "eval"
    run(["uv", "run", "python", "scripts/sim_eval/blend_eval_json.py",
         "--sim-json", str(SIM_ENVELOPE),
         "--direct-json", str(mdir / "test_predictions.json"),
         "--w", "0.0", "--out-dir", str(evd)])
    blended = evd / "hier_all_20260425_165622_w0p00.json"
    run(["uv", "run", "python", "scripts/sim_eval/reslice_eval_json.py",
         "--in", str(blended), "--odds", str(ODDS),
         "--out-dir", str(evd / "sliced"),
         "--min-volume", "50000", "--min-volume", "100000"])
    out = {}
    for tag in ("50000", "100000"):
        sliced = (evd / "sliced"
                  / f"hier_all_20260425_165622_w0p00_min_volume_{tag}.json")
        s = json.loads(sliced.read_text())["summary"]
        out[tag] = {
            "ll": s["avg_log_loss"], "market_ll": s.get("market_avg_log_loss"),
            "roi": s["flat_betting_roi_pct"],
            "roi_lo": s["flat_betting_roi_ci_low"],
            "roi_hi": s["flat_betting_roi_ci_high"],
            "n_bets": s["flat_betting_bets_placed"],
            "win": s["flat_betting_win_rate"],
        }
    return out


# ---------------------------------------------------------------- stages 3+4
def select_variants() -> dict:
    """Seed-29 val-LL sweep over all variants, both arms. VAL ONLY."""
    sel = {}
    for arm_kind in ("add", "repl"):
        best_v, best_ll = None, float("inf")
        for v in VARIANTS:
            m = train_one(f"{arm_kind}_{v}", 29)
            vll = m["val_log_loss"]
            print(f"  [stage3] {arm_kind}_{v}: val LL {vll:.4f}")
            if vll < best_ll:
                best_v, best_ll = v, vll
        sel[arm_kind] = {"variant": best_v, "val_ll": best_ll}
        print(f"  [stage3] {arm_kind} CHOSEN: {best_v} (val LL {best_ll:.4f})")
    return sel


def paired_run(sel: dict, base_control: dict) -> dict:
    results = {}
    for arm_kind in ("add", "repl"):
        arm = f"{arm_kind}_{sel[arm_kind]['variant']}"
        results[arm] = {}
        for seed in SEEDS:
            train_one(arm, seed)
            results[arm][seed] = eval_one(arm, seed)
            r = results[arm][seed]["50000"]
            print(f"  [stage4] {arm} seed {seed}: LL {r['ll']:.4f}  "
                  f"ROI {r['roi']:+.2f}%  (n={r['n_bets']})")
    summary = {"selection": sel, "base_control": base_control,
               "arms": results}
    return summary


def verdict_block(summary: dict) -> None:
    base = summary["base_control"]
    b_ll = base["mean_ll"]
    b_roi = base["mean_roi"]
    print("\n============ D9 PAIRED SUMMARY (>=$50k) ============")
    print(f"base control ({base['source']}): mean LL {b_ll:.4f}  "
          f"ROI {b_roi:+.2f}%")
    for arm, res in summary["arms"].items():
        lls = [res[s]["50000"]["ll"] for s in SEEDS]
        rois = [res[s]["50000"]["roi"] for s in SEEDS]
        mll, mroi = float(np.mean(lls)), float(np.mean(rois))
        dll, droi = mll - b_ll, mroi - b_roi
        ll_up = dll <= -FLOOR_LL
        roi_up = droi >= FLOOR_ROI
        v = ("LANDED" if ll_up and roi_up
             else "TABLED" if ll_up or roi_up else "FAILED")
        n_ll_better = sum(1 for s in SEEDS
                          if res[s]["50000"]["ll"] < A1_LOGGED[s][0])
        n_roi_better = sum(1 for s in SEEDS
                           if res[s]["50000"]["roi"] > A1_LOGGED[s][1])
        ll100 = float(np.mean([res[s]["100000"]["ll"] for s in SEEDS]))
        roi100 = float(np.mean([res[s]["100000"]["roi"] for s in SEEDS]))
        print(f"\n  {arm}: mean LL {mll:.4f} (dLL {dll:+.4f}, floor -{FLOOR_LL}"
              f"; per-seed better {n_ll_better}/5)")
        print(f"  {arm}: mean ROI {mroi:+.2f}% (dROI {droi:+.2f}pp, floor "
              f"+{FLOOR_ROI}; per-seed better {n_roi_better}/5)")
        print(f"  {arm}: >=$100k context LL {ll100:.4f} ROI {roi100:+.2f}%")
        print(f"  {arm}: ARM VERDICT (pre-committed floors): {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pre", "full"], default="full",
                    help="pre = stages 0-2 only (unit, parity, corr check); "
                    "full = everything.")
    args = ap.parse_args()

    print("=== D9 stage 0: TeamEloTracker unit check ===")
    unit_check()

    print("\n=== D9 stage 1: parity vs v2_clean ===")
    parity = verify_parity()

    print("\n=== D9 stage 2: dual correlation check ===")
    cc = corr_check()
    MODELS.mkdir(parents=True, exist_ok=True)
    (MODELS / "corr_check.json").write_text(json.dumps(cc, indent=2))
    if not cc["any_viable"]:
        print("CHEAP EXIT: no variant viable under either framing -> FAILED")
        return
    if args.stage == "pre":
        return

    print("\n=== D9 arms: build subset parquets ===")
    build_arm_parquets()

    if parity:
        base_control = {"source": "A1 logged per-seed (parity held)",
                        "mean_ll": A1_MEAN_LL, "mean_roi": A1_MEAN_ROI}
    else:
        # Fresh same-session base on the 51-col subset (v2_clean schema).
        base_dir = ARMS_DIR / "base"
        base_dir.mkdir(parents=True, exist_ok=True)
        base_cols = list(pd.read_parquet(V2_CLEAN / "train.parquet").columns)
        for split in SPLITS:
            full = pd.read_parquet(D9_DATA / f"{split}.parquet")
            full[base_cols].to_parquet(base_dir / f"{split}.parquet",
                                       index=False)
        res = {}
        for seed in SEEDS:
            train_one("base", seed)
            res[seed] = eval_one("base", seed)
        base_control = {
            "source": "same-session base (parity FAILED)",
            "mean_ll": float(np.mean([res[s]["50000"]["ll"] for s in SEEDS])),
            "mean_roi": float(np.mean([res[s]["50000"]["roi"] for s in SEEDS])),
            "per_seed": {s: res[s]["50000"] for s in SEEDS},
        }

    print("\n=== D9 stage 3: variant selection (val LL only) ===")
    sel = select_variants()

    print("\n=== D9 stage 4: paired 5-seed, both arms ===")
    summary = paired_run(sel, base_control)
    (MODELS / "d9_results.json").write_text(json.dumps(summary, indent=2))

    verdict_block(summary)


if __name__ == "__main__":
    main()
