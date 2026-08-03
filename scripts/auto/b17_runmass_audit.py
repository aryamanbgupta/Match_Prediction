"""B17 Task 2 — teacher-forced run-mass audit of BOTH serving stacks.

Diagnostic only. Measures how much run mass per LEGAL BALL each serving
stack carries under the sim's flat extras graft, versus actual, and
decomposes the gap into channels.

Stacks (serving configs mirrored exactly):
  i7     models/xgb_i7_noweights_production  scored RAW  (D17: no calibrator)
         on data/xgb_data_i7/cricket_data_i7_validation.parquet
  legacy models/xgb_v3 + vector_scaling_calibrator_v1.pkl (calibrator runs
         BEFORE the extras graft in sim_v1_2.XGBoostModelV2.predict_next_ball)
         on data/xgb_data_v3/cricket_data_v3_validation.parquet

Both stacks autoload a `venue_encoder_<suffix>.pkl` sidecar next to the
model in `XGBoostModelV2.__init__`, so `venue_on` is the serving arm for
both; `venue_zero` is reported as context only.

ENGINE COMPOSITION (derived from scripts/sim_v1_2.py, verified 2026-08-03):
  predict_next_ball (:1638-1643 for XGBoostModelV2) sets
      outcome_probs['wide'] = 0.01 ; outcome_probs['no_ball'] = 0.01
  then renormalises over the 8 keys, so per DELIVERY:
      p'_c    = p_c / 1.02      (6 model classes, sum p_c = 1)
      p'_wide = p'_nb = 0.01 / 1.02
  T20Rules.process_ball credits WIDE/NO_BALL exactly 1 run (:860-862, the
  `team_runs is None` branch) and MatchState.update marks
  `is_legal = outcome not in [WIDE, NO_BALL]` (:345) so the ball counter
  does NOT advance -> the delivery is re-bowled. A legacy-path no-ball
  produces NO off-bat runs (the off-bat composition at :934 is the I5
  `legal_off_bat_v1` branch only).

  Renewal identity per LEGAL ball: exactly one 6-class draw terminates the
  legal ball, and the expected number of wide/no-ball events before it is
  (0.02/1.02) / (1/1.02) = 0.02, each crediting 1 run. Hence

      E_delivery = (sum_c p_c * runs_c)/1.02 + (0.01/1.02)*1 + (0.01/1.02)*1
      M          = E_delivery / (1 - 0.02/1.02) = 1.02 * E_delivery
                 = R_model + 0.02          [runs per legal ball]

  with STRICT class run values runs_c = {dot:0, one:1, two:2, four:4,
  six:6, wicket:0} (threes are folded into class two by the label scheme
  and are reported as a separate channel, never inside M).

  Second-order engine details NOT modelled analytically (both stacks
  equally): `is_legal_outcome` converts a drawn wide/no-ball to DOT on
  ball 119, and a drawn wicket to DOT at 10 down.

ACTUAL DECOMPOSITION (exact identity on the scored rows, per legal ball):
      A            = total runs (all channels) / legal deliveries
      L_legal      = sum runs_class(label) over LEGAL rows / N_legal
      Residual     = (total runs on LEGAL rows)/N_legal - L_legal
                     (threes folded to 2, fives folded to 4, byes/leg-byes
                      that the label rounds down)
      Extras_act   = total runs on NON-LEGAL rows / N_legal
      A            = L_legal + Residual + Extras_act
  hence
      g = M - A = (R_model - L_legal) + (0.02 - Extras_act) + (-Residual)
                =   C_class          +   C_extras           + C_fold
  C_extras and C_fold are model-independent, so
      g_i7 - g_legacy == R_model_i7 - R_model_legacy  exactly.

Run:
    uv run python scripts/auto/b17_runmass_audit.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

CLASS_NAMES = ["dot", "one", "two", "four", "six", "wicket"]
RUNS_BY_CLASS = np.array([0, 1, 2, 4, 6, 0], dtype=float)
LABEL_TO_CLASS = {0: 0, 1: 1, 2: 2, 4: 3, 6: 4, -1: 5}

GRAFT_WIDE = 0.01
GRAFT_NB = 0.01
GRAFT_TOTAL = GRAFT_WIDE + GRAFT_NB          # 0.02 runs per legal ball

PHASES = [("pp_0_5", 0, 5), ("mid_6_14", 6, 14), ("death_15_19", 15, 19)]

# Pre-committed attribution thresholds (research/handoff/B17/plan.md).
COND_A_THRESHOLD = -0.0285
COND_B_THRESHOLD = -0.0538
CP6_BIAS_PER_BALL = {"i7": -4.781 / 84, "legacy": 4.259 / 84}

STACKS = {
    "i7": {
        "model_dir": REPO / "models/xgb_i7_noweights_production",
        "suffix": "i7",
        "parquet": REPO / "data/xgb_data_i7/cricket_data_i7_validation.parquet",
        "calibrator": None,
        "note": "promoted D16 no-weights, served RAW (D17)",
    },
    "legacy": {
        "model_dir": REPO / "models/xgb_v3",
        "suffix": "v3",
        "parquet": REPO / "data/xgb_data_v3/cricket_data_v3_validation.parquet",
        "calibrator": REPO / "models/xgb_v3/vector_scaling_calibrator_v1.pkl",
        "note": "legacy v7 booster + val-fit vector scaling (pre-graft)",
    },
}


# ---------------------------------------------------------------------------
# frame helpers
# ---------------------------------------------------------------------------

def feature_list(model_dir: Path, suffix: str) -> list:
    return [ln.strip() for ln
            in (model_dir / f"feature_columns_{suffix}.txt").read_text().splitlines()
            if ln.strip()]


def encode_frame(df: pd.DataFrame, model_dir: Path, suffix: str,
                 feat: list, log: list) -> pd.DataFrame:
    """Training-time encoded columns using `model_dir`'s OWN encoders.

    Mirrors scripts/auto/d16_marginal_audit.py::encode_frame, which in turn
    mirrors the per-ball lookup caches built in
    sim_v1_2.XGBoostModelV2.__init__ ({str(c): int(i)}, no offset).
    """
    out = df.copy()
    specs = [
        ("batter_id", "batter_encoded", f"batter_encoder_{suffix}.pkl"),
        ("bowler_id", "bowler_encoded", f"bowler_encoder_{suffix}.pkl"),
        ("matchup_type", "matchup_type_encoded", f"matchup_encoder_{suffix}.pkl"),
        ("venue", "venue_encoded", f"venue_encoder_{suffix}.pkl"),
    ]
    for raw, enc_name, enc_file in specs:
        if enc_name not in feat:
            continue
        enc_path = model_dir / enc_file
        if not enc_path.exists():
            raise FileNotFoundError(f"missing encoder {enc_path}")
        le = joblib.load(enc_path)
        lut = {str(c): i for i, c in enumerate(le.classes_)}
        out[enc_name] = out[raw].astype(str).map(lut).fillna(-1).astype(int)
        n_unk = int((out[enc_name] == -1).sum())
        log.append(f"    {enc_name}: {len(le.classes_)} classes, "
                   f"{n_unk:,} rows unseen -> -1")
    return out


def cricsheet_innings_totals(path: Path) -> dict:
    m = json.loads(path.read_text())
    agg = dict(deliveries=0, legal=0, total_runs=0, batter_runs=0,
               wides=0, noballs=0, byes=0, legbyes=0, penalty=0,
               n_wide_deliveries=0, n_noball_deliveries=0)
    for inn in m.get("innings", []):
        for ov in inn.get("overs", []):
            for d in ov.get("deliveries", []):
                ex = d.get("extras") or {}
                agg["deliveries"] += 1
                if "wides" not in ex and "noballs" not in ex:
                    agg["legal"] += 1
                if "wides" in ex:
                    agg["n_wide_deliveries"] += 1
                if "noballs" in ex:
                    agg["n_noball_deliveries"] += 1
                agg["total_runs"] += d["runs"]["total"]
                agg["batter_runs"] += d["runs"]["batter"]
                for k in ("wides", "noballs", "byes", "legbyes", "penalty"):
                    agg[k] += int(ex.get(k, 0) or 0)
    agg["gender"] = m.get("info", {}).get("gender")
    agg["date"] = str((m.get("info", {}).get("dates") or [""])[0])
    return agg


# ---------------------------------------------------------------------------
# core audit
# ---------------------------------------------------------------------------

def actual_channels(df: pd.DataFrame, mask=None) -> dict:
    """Exact per-legal-ball actual decomposition on the scored rows."""
    sub = df if mask is None else df[mask]
    legal = ~(sub["is_wide"].astype(bool) | sub["is_noball"].astype(bool))
    n_l = int(legal.sum())
    n_d = int(len(sub))
    y = sub["ball_outcome"].map(LABEL_TO_CLASS).to_numpy()
    label_runs = RUNS_BY_CLASS[y]
    runs = sub["runs"].to_numpy(dtype=float)
    legal_np = legal.to_numpy()

    s_total = float(runs.sum())
    s_legal = float(runs[legal_np].sum())
    s_nonlegal = float(runs[~legal_np].sum())
    l_legal_mass = float(label_runs[legal_np].sum())

    A = s_total / n_l
    L_legal = l_legal_mass / n_l
    residual = s_legal / n_l - L_legal
    extras_act = s_nonlegal / n_l
    return {
        "n_deliveries": n_d,
        "n_legal": n_l,
        "n_nonlegal": n_d - n_l,
        "deliveries_per_legal_ball": n_d / n_l,
        "total_runs": s_total,
        "A_runs_per_legal_ball": A,
        "L_legal_label_mass_per_legal_ball": L_legal,
        "residual_fold_per_legal_ball": residual,
        "extras_actual_per_legal_ball": extras_act,
        "identity_check": A - (L_legal + residual + extras_act),
        # label-mass over ALL deliveries (the population the booster is
        # trained/scored on) — comparator for R_model measured all-rows
        "label_mass_per_delivery_all_rows": float(label_runs.mean()),
        "label_mass_per_delivery_legal_rows": float(label_runs[legal_np].mean()),
    }


def score_stack(name: str, cfg: dict, log: list):
    """Returns (audit_dict, encoded_dataframe)."""
    model_dir = cfg["model_dir"]
    suffix = cfg["suffix"]
    log.append("")
    log.append("=" * 78)
    log.append(f"STACK {name}  ({cfg['note']})")
    log.append("=" * 78)
    log.append(f"  model dir : {model_dir}")
    log.append(f"  parquet   : {cfg['parquet']}")
    log.append(f"  calibrator: {cfg['calibrator']}")

    df = pd.read_parquet(cfg["parquet"])
    feat = feature_list(model_dir, suffix)
    log.append(f"  rows {len(df):,}  features {len(feat)}")
    mids = df["innings_id"].astype(str).str.split("_").str[1]
    log.append(f"  matches {mids.nunique()}  innings {df['innings_id'].nunique()}  "
               f"dates {df['match_date'].min()} .. {df['match_date'].max()}")

    df = encode_frame(df, model_dir, suffix, feat, log)
    model = joblib.load(model_dir / f"xgboost_model_{suffix}.pkl")

    X_on = df[feat]
    p_on = model.predict_proba(X_on)
    df0 = df.copy()
    df0["venue_encoded"] = 0
    p_zero = model.predict_proba(df0[feat])
    del df0

    calib = None
    if cfg["calibrator"]:
        calib = joblib.load(cfg["calibrator"])
        _v = getattr(calib, "_v", None)
        log.append("  calibrator vector v = "
                   + (str(np.asarray(_v, dtype=float).tolist())
                      if _v is not None else "<not exposed>"))
        p_on = calib.calibrate_probs(p_on)
        p_zero = calib.calibrate_probs(p_zero)
        log.append("  -> calibrated probabilities used for the serving arm "
                   "(calibrator runs BEFORE the graft in the engine)")
    else:
        log.append("  -> RAW probabilities (no calibrator)")

    legal_mask = ~(df["is_wide"].astype(bool) | df["is_noball"].astype(bool))
    y = df["ball_outcome"].map(LABEL_TO_CLASS).to_numpy()

    out = {"name": name, "model_dir": str(model_dir),
           "parquet": str(cfg["parquet"]),
           "calibrator": (str(cfg["calibrator"]) if cfg["calibrator"] else None),
           "n_rows": int(len(df)), "n_features": len(feat),
           "n_matches": int(mids.nunique()),
           "date_min": str(df["match_date"].min()),
           "date_max": str(df["match_date"].max())}

    for arm, probs in (("venue_on", p_on), ("venue_zero", p_zero)):
        pred_marg = probs.mean(axis=0)
        actual_marg = np.bincount(y, minlength=6) / len(y)
        r_all = float((probs @ RUNS_BY_CLASS).mean())
        r_legal = float((probs[legal_mask.to_numpy()] @ RUNS_BY_CLASS).mean())
        arm_out = {
            "per_class": [
                {"class": CLASS_NAMES[c], "pred": float(pred_marg[c]),
                 "actual_label_freq": float(actual_marg[c]),
                 "delta": float(pred_marg[c] - actual_marg[c])}
                for c in range(6)],
            "R_model_all_rows": r_all,
            "R_model_legal_rows_only": r_legal,
            "phases": {},
        }
        for pname, lo, hi in PHASES:
            m = ((df["over_idx"] >= lo) & (df["over_idx"] <= hi)).to_numpy()
            arm_out["phases"][pname] = {
                "n_deliveries": int(m.sum()),
                "R_model_all_rows": float((probs[m] @ RUNS_BY_CLASS).mean()),
                "pred_wicket": float(probs[m][:, 5].mean()),
            }
        out[arm] = arm_out

    out["actual"] = actual_channels(df)
    out["actual_phases"] = {}
    for pname, lo, hi in PHASES:
        m = (df["over_idx"] >= lo) & (df["over_idx"] <= hi)
        out["actual_phases"][pname] = actual_channels(df, m)

    # finer actual breakdown (i7 frame only carries the extras columns)
    extra_cols = ["batter_runs", "wide_runs", "noball_runs", "bye_runs",
                  "legbye_runs", "penalty_runs"]
    if all(c in df.columns for c in extra_cols):
        n_l = out["actual"]["n_legal"]
        out["actual_channel_detail"] = {
            c: float(df[c].sum()) / n_l for c in extra_cols}
        out["actual_channel_detail"]["sum_check_per_legal_ball"] = float(
            sum(df[c].sum() for c in extra_cols) / n_l)
    # threes fold: what class-2-labeled legal balls really scored
    lm = legal_mask.to_numpy()
    cls2 = lm & (df["ball_outcome"].to_numpy() == 2)
    out["class2_fold"] = {
        "n_legal_class2": int(cls2.sum()),
        "mean_actual_runs_on_class2_legal": float(df["runs"].to_numpy()[cls2].mean()),
        "strict_class_value": 2.0,
    }
    cls4 = lm & (df["ball_outcome"].to_numpy() == 4)
    out["class4_fold"] = {
        "n_legal_class4": int(cls4.sum()),
        "mean_actual_runs_on_class4_legal": float(df["runs"].to_numpy()[cls4].mean()),
        "strict_class_value": 4.0,
    }
    out["_mids"] = sorted(set(mids))
    out["_df_cols"] = list(df.columns)
    return out, df


def compose(stack: dict, arm: str = "venue_on", pop: str = "all_rows") -> dict:
    key = "R_model_all_rows" if pop == "all_rows" else "R_model_legal_rows_only"
    r = stack[arm][key]
    a = stack["actual"]
    M = r + GRAFT_TOTAL
    A = a["A_runs_per_legal_ball"]
    c_class = r - a["L_legal_label_mass_per_legal_ball"]
    c_extras = GRAFT_TOTAL - a["extras_actual_per_legal_ball"]
    c_fold = -a["residual_fold_per_legal_ball"]
    return {"arm": arm, "population": pop, "R_model": r, "M": M, "A": A,
            "g": M - A, "C_class": c_class, "C_extras": c_extras,
            "C_fold": c_fold,
            "channel_sum_check": (c_class + c_extras + c_fold) - (M - A)}


def compose_phase(stack: dict, pname: str, arm: str = "venue_on") -> dict:
    r = stack[arm]["phases"][pname]["R_model_all_rows"]
    a = stack["actual_phases"][pname]
    M = r + GRAFT_TOTAL
    A = a["A_runs_per_legal_ball"]
    return {"phase": pname, "R_model": r, "M": M, "A": A, "g": M - A,
            "C_class": r - a["L_legal_label_mass_per_legal_ball"],
            "C_extras": GRAFT_TOTAL - a["extras_actual_per_legal_ball"],
            "C_fold": -a["residual_fold_per_legal_ball"],
            "n_legal": a["n_legal"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", default=str(REPO / "data/t20s_json"))
    ap.add_argument("--n-semantics-matches", type=int, default=10)
    ap.add_argument("--out-json",
                    default=str(REPO / "models/auto/b17/runmass_audit.json"))
    ap.add_argument("--out-txt",
                    default=str(REPO / "research/handoff/B17/raw/runmass_audit.txt"))
    args = ap.parse_args()

    log: list = []
    log.append("B17 TASK 2 — teacher-forced run-mass audit, both serving stacks")
    log.append("(diagnostic only; no engine change, no sim run)")
    log.append("")
    log.append(f"graft: wide {GRAFT_WIDE} + no_ball {GRAFT_NB} pre-normalisation "
               f"-> effective {GRAFT_WIDE / 1.02:.6f} each per delivery; "
               f"{GRAFT_TOTAL:.4f} runs per legal ball")
    log.append("composition: M = R_model + 0.02   (renewal identity, derived "
               "in the module docstring from sim_v1_2.py)")

    payload = {"graft": {"wide": GRAFT_WIDE, "no_ball": GRAFT_NB,
                         "effective_per_delivery": GRAFT_WIDE / 1.02,
                         "runs_per_legal_ball": GRAFT_TOTAL},
               "stacks": {}}

    # ---- D16 marginal_audit.json cross-check ------------------------------
    ma_path = REPO / "models/xgb_i7_noweights_production/marginal_audit.json"
    log.append("")
    log.append("-" * 78)
    log.append("D16 marginal_audit.json (sidecar) — cross-check reference")
    log.append("-" * 78)
    if ma_path.exists():
        ma = json.loads(ma_path.read_text())
        payload["d16_marginal_audit"] = {
            "test_parquet": ma.get("test_parquet"),
            "n_balls": ma.get("n_balls"),
            "venue_on": ma["arms"]["venue_on"],
        }
        log.append(f"  split scored there: {ma.get('test_parquet')}")
        log.append(f"  n_balls {ma.get('n_balls'):,}   "
                   f"(NOTE: that is the TEST split; B17 scores VALIDATION — "
                   f"the two are different populations, so exact agreement is "
                   f"NOT expected)")
        for row in ma["arms"]["venue_on"]["per_class"]:
            log.append(f"    {row['class']:<7} pred {row['pred']:.6f}  "
                       f"actual {row['actual']:.6f}  delta {row['delta']:+.6f}")
        log.append(f"    pred_runs_per_ball {ma['arms']['venue_on']['pred_runs_per_ball']:.6f}"
                   f"  actual {ma['arms']['venue_on']['actual_runs_per_ball']:.6f}"
                   f"  delta {ma['arms']['venue_on']['delta_runs_per_ball']:+.6f}")
    else:
        log.append(f"  ABSENT: {ma_path}")

    # ---- score both stacks -------------------------------------------------
    stacks = {}
    frames = {}
    for name, cfg in STACKS.items():
        stacks[name], frames[name] = score_stack(name, cfg, log)

    # ---- STEP 1: frame agreement + row semantics --------------------------
    log.append("")
    log.append("=" * 78)
    log.append("STEP 1 — frame agreement and ROW SEMANTICS (row =? delivery =? legal ball)")
    log.append("=" * 78)
    mids_i7 = set(stacks["i7"]["_mids"])
    mids_lg = set(stacks["legacy"]["_mids"])
    inter = mids_i7 & mids_lg
    log.append(f"  i7 matches {len(mids_i7)}   legacy matches {len(mids_lg)}   "
               f"intersection {len(inter)}   "
               f"i7-only {len(mids_i7 - mids_lg)}   legacy-only {len(mids_lg - mids_i7)}")
    log.append(f"  i7 rows {stacks['i7']['n_rows']:,}   "
               f"legacy rows {stacks['legacy']['n_rows']:,}")
    log.append(f"  window i7 {stacks['i7']['date_min']}..{stacks['i7']['date_max']}   "
               f"legacy {stacks['legacy']['date_min']}..{stacks['legacy']['date_max']}")
    payload["frame_agreement"] = {
        "n_matches_i7": len(mids_i7), "n_matches_legacy": len(mids_lg),
        "n_intersection": len(inter),
        "i7_only": sorted(mids_i7 - mids_lg), "legacy_only": sorted(mids_lg - mids_i7),
        "rows_i7": stacks["i7"]["n_rows"], "rows_legacy": stacks["legacy"]["n_rows"]}

    src = Path(args.source_dir)
    semantics = {}
    for name in STACKS:
        df = frames[name]
        mid_col = df["innings_id"].astype(str).str.split("_").str[1]
        per_match_rows = mid_col.value_counts()
        legal_mask = ~(df["is_wide"].astype(bool) | df["is_noball"].astype(bool))
        per_match_legal = mid_col[legal_mask.to_numpy()].value_counts()
        sample = sorted(per_match_rows.index)[: args.n_semantics_matches]
        log.append("")
        log.append(f"  --- {name}: per-match row counts vs cricsheet "
                   f"(first {len(sample)} matches) ---")
        log.append(f"  | {'match_id':>10} | {'pq rows':>8} | {'cs deliv':>8} | "
                   f"{'pq legal':>8} | {'cs legal':>8} | {'rows==deliv':>11} | "
                   f"{'legal match':>11} |")
        rows_eq = legal_eq = 0
        detail = []
        for mid in sample:
            cs = cricsheet_innings_totals(src / f"{mid}.json")
            pr = int(per_match_rows[mid])
            pl = int(per_match_legal.get(mid, 0))
            e1 = pr == cs["deliveries"]
            e2 = pl == cs["legal"]
            rows_eq += e1
            legal_eq += e2
            log.append(f"  | {mid:>10} | {pr:>8} | {cs['deliveries']:>8} | "
                       f"{pl:>8} | {cs['legal']:>8} | {str(e1):>11} | "
                       f"{str(e2):>11} |")
            detail.append({"match_id": mid, "parquet_rows": pr,
                           "cricsheet_deliveries": cs["deliveries"],
                           "parquet_legal": pl, "cricsheet_legal": cs["legal"],
                           "rows_eq_deliveries": bool(e1),
                           "legal_eq_legal": bool(e2)})
        log.append(f"  sample: rows==cricsheet deliveries on {rows_eq}/{len(sample)}; "
                   f"parquet legal==cricsheet legal on {legal_eq}/{len(sample)}")
        semantics[name] = {"sample": detail,
                           "sample_rows_eq": rows_eq,
                           "sample_legal_eq": legal_eq,
                           "sample_n": len(sample)}
    payload["row_semantics"] = semantics

    # full-corpus cricsheet aggregate on the scored match set
    log.append("")
    log.append("  --- full cricsheet aggregate over the 545 scored matches ---")
    tot = dict(deliveries=0, legal=0, total_runs=0, batter_runs=0, wides=0,
               noballs=0, byes=0, legbyes=0, penalty=0,
               n_wide_deliveries=0, n_noball_deliveries=0)
    for mid in sorted(mids_i7):
        cs = cricsheet_innings_totals(src / f"{mid}.json")
        for k in tot:
            tot[k] += cs[k]
    cs_A = tot["total_runs"] / tot["legal"]
    log.append(f"  cricsheet: {tot['deliveries']:,} deliveries, {tot['legal']:,} legal, "
               f"{tot['total_runs']:,} runs")
    log.append(f"  cricsheet A (all runs / legal balls) = {cs_A:.6f}")
    log.append(f"  cricsheet channels per legal ball: off-bat "
               f"{tot['batter_runs'] / tot['legal']:.6f}  wides "
               f"{tot['wides'] / tot['legal']:.6f}  no-balls "
               f"{tot['noballs'] / tot['legal']:.6f}  byes "
               f"{tot['byes'] / tot['legal']:.6f}  leg-byes "
               f"{tot['legbyes'] / tot['legal']:.6f}  penalty "
               f"{tot['penalty'] / tot['legal']:.6f}")
    log.append(f"  cricsheet p_wide {tot['n_wide_deliveries'] / tot['deliveries']:.6f} "
               f"(D3 anchor 0.037702)   p_no_ball "
               f"{tot['n_noball_deliveries'] / tot['deliveries']:.6f} "
               f"(D3 anchor 0.004409)")
    pq = stacks["i7"]["actual"]
    log.append(f"  parquet  : {pq['n_deliveries']:,} deliveries, "
               f"{pq['n_legal']:,} legal, {pq['total_runs']:,.0f} runs   "
               f"A = {pq['A_runs_per_legal_ball']:.6f}")
    log.append(f"  parquet-vs-cricsheet delta: deliveries "
               f"{pq['n_deliveries'] - tot['deliveries']:+d}  legal "
               f"{pq['n_legal'] - tot['legal']:+d}  runs "
               f"{pq['total_runs'] - tot['total_runs']:+.0f}  A "
               f"{pq['A_runs_per_legal_ball'] - cs_A:+.6f}")
    log.append("  ROW SEMANTICS VERDICT: parquet rows are ALL DELIVERIES "
               "(wides and no-balls included as rows, labels folding their runs); "
               "legal balls = rows with is_wide==0 and is_noball==0.")
    payload["cricsheet_actuals"] = {
        **{k: int(v) for k, v in tot.items()},
        "A_runs_per_legal_ball": cs_A,
        "offbat_per_legal_ball": tot["batter_runs"] / tot["legal"],
        "wides_per_legal_ball": tot["wides"] / tot["legal"],
        "noballs_per_legal_ball": tot["noballs"] / tot["legal"],
        "byes_per_legal_ball": tot["byes"] / tot["legal"],
        "legbyes_per_legal_ball": tot["legbyes"] / tot["legal"],
        "penalty_per_legal_ball": tot["penalty"] / tot["legal"],
        "p_wide_per_delivery": tot["n_wide_deliveries"] / tot["deliveries"],
        "p_no_ball_per_delivery": tot["n_noball_deliveries"] / tot["deliveries"],
    }

    # ---- STEP 2: marginals -------------------------------------------------
    log.append("")
    log.append("=" * 78)
    log.append("STEP 2 — per-class marginals (venue_on serving arm, all delivery rows)")
    log.append("=" * 78)
    log.append(f"| {'class':<7}| {'i7 pred':>9} | {'legacy pred':>11} | "
               f"{'actual':>9} | {'d i7':>9} | {'d legacy':>9} |")
    log.append("|" + "-" * 8 + "|" + "-" * 11 + "|" + "-" * 13 + "|" + "-" * 11
               + "|" + "-" * 11 + "|" + "-" * 11 + "|")
    for c in range(6):
        ri = stacks["i7"]["venue_on"]["per_class"][c]
        rl = stacks["legacy"]["venue_on"]["per_class"][c]
        log.append(f"| {CLASS_NAMES[c]:<7}| {ri['pred']:>9.5f} | {rl['pred']:>11.5f} | "
                   f"{ri['actual_label_freq']:>9.5f} | {ri['delta']:>+9.5f} | "
                   f"{rl['delta']:>+9.5f} |")
    log.append("")
    log.append("  per-phase R_model (expected 6-class runs per delivery, strict values):")
    log.append(f"  | {'phase':<12}| {'n deliv':>8} | {'i7':>9} | {'legacy':>9} | "
               f"{'i7-legacy':>10} |")
    for pname, _, _ in PHASES:
        a = stacks["i7"]["venue_on"]["phases"][pname]
        b = stacks["legacy"]["venue_on"]["phases"][pname]
        log.append(f"  | {pname:<12}| {a['n_deliveries']:>8,} | "
                   f"{a['R_model_all_rows']:>9.5f} | {b['R_model_all_rows']:>9.5f} | "
                   f"{a['R_model_all_rows'] - b['R_model_all_rows']:>+10.5f} |")
    log.append("")
    log.append("  venue_zero context arm R_model (all rows): "
               f"i7 {stacks['i7']['venue_zero']['R_model_all_rows']:.6f}  "
               f"legacy {stacks['legacy']['venue_zero']['R_model_all_rows']:.6f}")

    # ---- STEP 3-5: composition + channels ---------------------------------
    log.append("")
    log.append("=" * 78)
    log.append("STEP 3-5 — composed run mass M, actual A, gap g, channel decomposition")
    log.append("=" * 78)
    a_i7 = stacks["i7"]["actual"]
    log.append(f"  scored population: {a_i7['n_deliveries']:,} deliveries, "
               f"{a_i7['n_legal']:,} legal, {a_i7['n_nonlegal']:,} non-legal; "
               f"deliveries per legal ball {a_i7['deliveries_per_legal_ball']:.6f} "
               f"(engine assumes 1.02)")
    log.append(f"  actual identity check A - (L_legal + residual + extras) = "
               f"{a_i7['identity_check']:.3e}")
    log.append(f"  A                          = {a_i7['A_runs_per_legal_ball']:.6f}")
    log.append(f"    L_legal   (6-class label mass on legal balls) = "
               f"{a_i7['L_legal_label_mass_per_legal_ball']:.6f}")
    log.append(f"    residual  (threes/fives folded down, label rounding) = "
               f"{a_i7['residual_fold_per_legal_ball']:.6f}")
    log.append(f"    extras_act(runs on non-legal deliveries) = "
               f"{a_i7['extras_actual_per_legal_ball']:.6f}   "
               f"vs graft {GRAFT_TOTAL:.4f}")
    if "actual_channel_detail" in stacks["i7"]:
        d = stacks["i7"]["actual_channel_detail"]
        log.append("  parquet actual channel detail (runs per legal ball): "
                   + "  ".join(f"{k} {v:.6f}" for k, v in d.items()))
    c2 = stacks["i7"]["class2_fold"]
    c4 = stacks["i7"]["class4_fold"]
    log.append(f"  class-2 legal balls: n {c2['n_legal_class2']:,}  mean actual runs "
               f"{c2['mean_actual_runs_on_class2_legal']:.6f} vs strict 2.0 "
               f"(fold shortfall {c2['mean_actual_runs_on_class2_legal'] - 2.0:+.6f} per class-2 ball)")
    log.append(f"  class-4 legal balls: n {c4['n_legal_class4']:,}  mean actual runs "
               f"{c4['mean_actual_runs_on_class4_legal']:.6f} vs strict 4.0 "
               f"(fold shortfall {c4['mean_actual_runs_on_class4_legal'] - 4.0:+.6f} per class-4 ball)")
    log.append("")

    comp = {}
    for pop in ("all_rows", "legal_rows_only"):
        log.append(f"  --- composition, R_model measured on {pop} "
                   f"(primary = all_rows: the sim calls predict_next_ball "
                   f"once per DELIVERY) ---")
        log.append(f"  | {'stack':<8}| {'R_model':>9} | {'M':>9} | {'A':>9} | "
                   f"{'g = M-A':>9} | {'C_class':>9} | {'C_extras':>9} | "
                   f"{'C_fold':>9} |")
        log.append("  |" + "-" * 9 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 11
                   + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 11 + "|")
        comp[pop] = {}
        for name in ("i7", "legacy"):
            c = compose(stacks[name], "venue_on", pop)
            comp[pop][name] = c
            log.append(f"  | {name:<8}| {c['R_model']:>9.6f} | {c['M']:>9.6f} | "
                       f"{c['A']:>9.6f} | {c['g']:>+9.6f} | {c['C_class']:>+9.6f} | "
                       f"{c['C_extras']:>+9.6f} | {c['C_fold']:>+9.6f} |")
        d = comp[pop]["i7"]["g"] - comp[pop]["legacy"]["g"]
        dr = comp[pop]["i7"]["R_model"] - comp[pop]["legacy"]["R_model"]
        comp[pop]["paired"] = {"g_i7_minus_g_legacy": d,
                               "R_model_i7_minus_legacy": dr,
                               "cancellation_check": d - dr}
        log.append(f"  paired: g_i7 - g_legacy = {d:+.6f}   "
                   f"(== R_model_i7 - R_model_legacy = {dr:+.6f}; "
                   f"C_extras and C_fold are model-independent and cancel, "
                   f"residual {d - dr:.3e})")
        log.append("")
    payload["composition"] = comp

    # per-phase g
    log.append("  --- per-phase g (venue_on, all_rows) ---")
    log.append(f"  | {'phase':<12}| {'n legal':>8} | {'i7 M':>8} | {'legacy M':>9} | "
               f"{'A':>8} | {'g i7':>9} | {'g legacy':>9} |")
    payload["per_phase"] = {}
    for pname, _, _ in PHASES:
        pi = compose_phase(stacks["i7"], pname)
        pl = compose_phase(stacks["legacy"], pname)
        payload["per_phase"][pname] = {"i7": pi, "legacy": pl}
        log.append(f"  | {pname:<12}| {pi['n_legal']:>8,} | {pi['M']:>8.5f} | "
                   f"{pl['M']:>9.5f} | {pi['A']:>8.5f} | {pi['g']:>+9.5f} | "
                   f"{pl['g']:>+9.5f} |")
    log.append("")

    # ---- STEP 6: pre-committed conditions ---------------------------------
    g_i7 = comp["all_rows"]["i7"]["g"]
    g_lg = comp["all_rows"]["legacy"]["g"]
    cond_a = g_i7 <= COND_A_THRESHOLD
    cond_b = (g_i7 - g_lg) <= COND_B_THRESHOLD
    log.append("=" * 78)
    log.append("STEP 6 — PRE-COMMITTED ATTRIBUTION CONDITIONS (arithmetic only; "
               "NO VERDICT — orchestrator decides)")
    log.append("=" * 78)
    log.append(f"  observed cp6 quote bias per legal ball: "
               f"i7 -4.781/84 = {CP6_BIAS_PER_BALL['i7']:+.6f}   "
               f"legacy +4.259/84 = {CP6_BIAS_PER_BALL['legacy']:+.6f}   "
               f"sign-flip delta = "
               f"{CP6_BIAS_PER_BALL['i7'] - CP6_BIAS_PER_BALL['legacy']:+.6f}")
    log.append("")
    log.append(f"  CONDITION (a): g_i7 <= {COND_A_THRESHOLD}")
    log.append(f"      g_i7            = {g_i7:+.6f}")
    log.append(f"      threshold       = {COND_A_THRESHOLD:+.6f}")
    log.append(f"      margin (g - th) = {g_i7 - COND_A_THRESHOLD:+.6f}")
    log.append(f"      (a) MET: {cond_a}")
    log.append("")
    log.append(f"  CONDITION (b): g_i7 - g_legacy <= {COND_B_THRESHOLD}")
    log.append(f"      g_i7            = {g_i7:+.6f}")
    log.append(f"      g_legacy        = {g_lg:+.6f}")
    log.append(f"      g_i7 - g_legacy = {g_i7 - g_lg:+.6f}")
    log.append(f"      threshold       = {COND_B_THRESHOLD:+.6f}")
    log.append(f"      margin          = {(g_i7 - g_lg) - COND_B_THRESHOLD:+.6f}")
    log.append(f"      (b) MET: {cond_b}")
    log.append("")
    log.append(f"  BOTH (a) AND (b) MET: {bool(cond_a and cond_b)}")
    log.append("  (executor states the arithmetic only; the verdict mapping is "
               "the orchestrator's)")
    log.append("")
    log.append("  sensitivity — same conditions with R_model on legal rows only:")
    g_i7_l = comp["legal_rows_only"]["i7"]["g"]
    g_lg_l = comp["legal_rows_only"]["legacy"]["g"]
    log.append(f"      g_i7 = {g_i7_l:+.6f}  -> (a) MET: {g_i7_l <= COND_A_THRESHOLD}")
    log.append(f"      g_i7 - g_legacy = {g_i7_l - g_lg_l:+.6f}  -> (b) MET: "
               f"{(g_i7_l - g_lg_l) <= COND_B_THRESHOLD}")
    log.append("  sensitivity — A taken from cricsheet instead of the parquet:")
    for name in ("i7", "legacy"):
        gg = comp["all_rows"][name]["M"] - cs_A
        log.append(f"      {name}: M {comp['all_rows'][name]['M']:.6f} - cricsheet A "
                   f"{cs_A:.6f} = g {gg:+.6f}")

    payload["conditions"] = {
        "cp6_bias_per_legal_ball": CP6_BIAS_PER_BALL,
        "sign_flip_delta_per_legal_ball":
            CP6_BIAS_PER_BALL["i7"] - CP6_BIAS_PER_BALL["legacy"],
        "threshold_a": COND_A_THRESHOLD, "threshold_b": COND_B_THRESHOLD,
        "g_i7": g_i7, "g_legacy": g_lg, "g_i7_minus_g_legacy": g_i7 - g_lg,
        "condition_a_met": bool(cond_a), "condition_b_met": bool(cond_b),
        "both_met": bool(cond_a and cond_b),
        "sensitivity_legal_rows_only": {
            "g_i7": g_i7_l, "g_i7_minus_g_legacy": g_i7_l - g_lg_l,
            "condition_a_met": bool(g_i7_l <= COND_A_THRESHOLD),
            "condition_b_met": bool((g_i7_l - g_lg_l) <= COND_B_THRESHOLD)},
        "sensitivity_cricsheet_A": {
            name: comp["all_rows"][name]["M"] - cs_A for name in ("i7", "legacy")},
    }

    for name in stacks:
        stacks[name].pop("_mids", None)
        stacks[name].pop("_df_cols", None)
    payload["stacks"] = stacks

    txt = "\n".join(log)
    print(txt)
    out_txt = Path(args.out_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(txt + "\n")
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nwrote {out_txt}")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
