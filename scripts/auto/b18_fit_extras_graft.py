"""B18 — fit the empirical extras graft sidecar for the promoted i7 stack.

Recovers D3's marginal-preserving extras composition (rates half) AND adds
the piece D3 never had: per-event extras-RUN crediting. B17's teacher-forced
run-mass audit measured the promoted i7 serving path carrying

    g_i7 = -0.052785 runs per legal ball

of which -0.039559 is the flat 1%+1% graft under-carrying explicit extras
(0.0200 grafted vs 0.059559 actual runs on non-legal deliveries per legal
ball). B18 grafts the empirical rates AND credits an empirical integer per
extras event, so the carried extras mass matches the measured channel.

POPULATION (identical to B17 Task 2, whose loading/verification code this
script imports verbatim): `data/xgb_data_i7/cricket_data_i7_validation.parquet`
— the i7 ball model's validation split, 2024-12-31 <= date < 2025-06-30,
ALL DELIVERIES as rows (wides / no-balls included as rows). B17 verified
row-for-row against cricsheet: 124,292 deliveries, 119,058 legal, 169,099
runs, delta +0 on every count.

RATES (per DELIVERY — `predict_next_ball` is drawn once per delivery and
wides / no-balls are re-bowled):
    p_wide    = deliveries with is_wide   / all deliveries   (D3 anchor 0.037702)
    p_no_ball = deliveries with is_noball / all deliveries   (D3 anchor 0.004409)
Both must reproduce D3's pre-committed val-split anchors to 6 dp or this
script STOPS.

PER-EVENT RUN LAWS (the same extras-channel accounting b17_runmass_audit
uses, so the fix targets the measured deficit):
    wide    -> total runs on the wide delivery. A batter cannot score off
               the bat on a wide, so every run on the delivery is extras.
    no_ball -> the `noball_runs` CHANNEL ONLY (the no-ball penalty). Off-bat
               runs on a no-ball are folded into the 6-class labels and are
               NOT credited here — crediting delivery totals would
               double-count exactly the way D3 died on the legacy stack.

ORCHESTRATOR RULING 2026-08-03 (the operative definition; see
research/handoff/B18/result.md for the verbatim text). The first run of this
script used the plan's prose gloss "extras-portion only (penalty + byes)"
== `runs - batter_runs`, which measures 1.175182 and tripped the plan's own
>0.05 STOP against its ~1.071 anchor. The orchestrator ruled that the
NARROW `noball_runs` channel (1.071168) is operative: the prose gloss was
the error, the pre-committed anchor is the B17-committed design arithmetic,
and the narrow reading is the one consistent with the plan's Scope guard and
Expected residual sections, which leave byes/leg-byes (on legal AND no-ball
deliveries, ~ -0.0072) UNMODELED. Both candidates are still measured and
reported below; only the operative one is written to the sidecar.

Sanity anchors from B17's channel table: wide mean ~1.204, no-ball
`noball_runs` mean ~1.071. A deviation > 0.05 on either STOPS the script.

ANALYTIC PRE-CHECK (before any sim runs). Under the graft, the renewal
identity per LEGAL ball becomes

    carried_extras = (p_w * r_w + p_nb * r_nb) / (1 - p_w - p_nb)
    M_new          = R_model + carried_extras
    g_new          = M_new - A

with R_model the i7 booster's expected 6-class run mass per delivery
(venue_on serving arm, all rows) and A the actual runs per legal ball.
PRE-COMMITTED TOLERANCE (research/handoff/B18/plan.md): g_new must land in
[-0.030, -0.012]. Outside -> STOP, record, run nothing.

Artifact: models/auto/b18/extras_graft_v1.json
Run: uv run python scripts/auto/b18_fit_extras_graft.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "auto"))

# B17's own loading / verification code — reused verbatim, not re-derived.
from b17_runmass_audit import (  # noqa: E402
    GRAFT_TOTAL,
    RUNS_BY_CLASS,
    actual_channels,
    cricsheet_innings_totals,
    encode_frame,
    feature_list,
)

MODEL_DIR = REPO / "models/xgb_i7_noweights_production"
SUFFIX = "i7"
PARQUET = REPO / "data/xgb_data_i7/cricket_data_i7_validation.parquet"
OUT_DIR = REPO / "models/auto/b18"

# D3's pre-committed val-split anchors (models/auto/d3/extras_rates.json,
# re-confirmed by B17 against cricsheet on this exact match set).
D3_P_WIDE = 0.037702
D3_P_NO_BALL = 0.004409

# B17 channel-table anchors for the per-event means.
ANCHOR_WIDE_MEAN = 1.204
ANCHOR_NB_MEAN = 1.071
MEAN_TOL = 0.05

# B17's measured constants on this population (reproduction cross-check).
B17_G_I7 = -0.052785
B17_A = 1.420308
B17_EXTRAS_ACTUAL = 0.059559

# Pre-committed analytic tolerance for the predicted post-fix gap.
G_NEW_LO = -0.030
G_NEW_HI = -0.012


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def law_from_counts(counts: Counter) -> dict:
    support = sorted(counts)
    n = sum(counts.values())
    probs = [counts[s] / n for s in support]
    # renormalise the last cell so the stored law sums to exactly 1.0
    probs[-1] = 1.0 - sum(probs[:-1])
    mean = float(sum(s * p for s, p in zip(support, probs)))
    return {"support": [int(s) for s in support],
            "probs": [float(p) for p in probs],
            "mean": mean,
            "n_events": int(n),
            "counts": {str(int(s)): int(counts[s]) for s in support}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", default=str(REPO / "data/t20s_json"))
    ap.add_argument("--out-json", default=str(OUT_DIR / "extras_graft_v1.json"))
    ap.add_argument("--out-txt",
                    default=str(REPO / "research/handoff/B18/raw/fit.txt"))
    args = ap.parse_args()

    log: list = []

    def emit(line: str = "") -> None:
        print(line)
        log.append(line)

    emit("B18 — empirical extras graft fit (promoted i7 stack, val split)")
    emit("")
    emit(f"  model dir : {MODEL_DIR}")
    emit(f"  parquet   : {PARQUET}")
    emit(f"  booster md5: {md5_of(MODEL_DIR / f'xgboost_model_{SUFFIX}.pkl')}")
    emit(f"  parquet md5: {md5_of(PARQUET)}")

    # ------------------------------------------------------------ population
    df = pd.read_parquet(PARQUET)
    feat = feature_list(MODEL_DIR, SUFFIX)
    mids = df["innings_id"].astype(str).str.split("_").str[1]
    emit(f"  rows {len(df):,}  features {len(feat)}  matches {mids.nunique()}  "
         f"innings {df['innings_id'].nunique()}")
    emit(f"  window {df['match_date'].min()} .. {df['match_date'].max()}")
    pop_hash = hashlib.md5(
        ("|".join(sorted(set(mids))) + f"#{len(df)}").encode()).hexdigest()
    emit(f"  population hash (sorted match ids + row count): {pop_hash}")

    is_wide = df["is_wide"].astype(bool).to_numpy()
    is_nb = df["is_noball"].astype(bool).to_numpy()
    legal = ~(is_wide | is_nb)
    n_deliveries = int(len(df))
    n_legal = int(legal.sum())

    # ------------------------------------------------- STEP 1: rate anchors
    emit("")
    emit("=" * 78)
    emit("STEP 1 — per-delivery rates vs D3's pre-committed val anchors")
    emit("=" * 78)
    p_wide = float(is_wide.sum()) / n_deliveries
    p_no_ball = float(is_nb.sum()) / n_deliveries
    emit(f"  deliveries {n_deliveries:,}   legal {n_legal:,}   "
         f"wide rows {int(is_wide.sum()):,}   no-ball rows {int(is_nb.sum()):,}")
    emit(f"  p_wide    = {p_wide:.9f}  (D3 anchor {D3_P_WIDE:.6f})  "
         f"rounded {round(p_wide, 6):.6f}")
    emit(f"  p_no_ball = {p_no_ball:.9f}  (D3 anchor {D3_P_NO_BALL:.6f})  "
         f"rounded {round(p_no_ball, 6):.6f}")
    rate_ok = (round(p_wide, 6) == D3_P_WIDE
               and round(p_no_ball, 6) == D3_P_NO_BALL)
    emit(f"  anchors reproduce to 6 dp: {rate_ok}")
    if not rate_ok:
        emit("  STOP — D3 rate anchors did NOT reproduce on this population.")
        Path(args.out_txt).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_txt).write_text("\n".join(log) + "\n")
        raise SystemExit(2)

    # cricsheet cross-check on the same match set (B17's own aggregator)
    src = Path(args.source_dir)
    tot = dict(deliveries=0, legal=0, total_runs=0, batter_runs=0, wides=0,
               noballs=0, byes=0, legbyes=0, penalty=0,
               n_wide_deliveries=0, n_noball_deliveries=0)
    for mid in sorted(set(mids)):
        cs = cricsheet_innings_totals(src / f"{mid}.json")
        for k in tot:
            tot[k] += cs[k]
    emit(f"  cricsheet cross-check: {tot['deliveries']:,} deliveries, "
         f"{tot['legal']:,} legal, {tot['total_runs']:,} runs   "
         f"p_wide {tot['n_wide_deliveries'] / tot['deliveries']:.6f}  "
         f"p_no_ball {tot['n_noball_deliveries'] / tot['deliveries']:.6f}")
    emit(f"  parquet-vs-cricsheet delta: deliveries "
         f"{n_deliveries - tot['deliveries']:+d}  legal "
         f"{n_legal - tot['legal']:+d}  runs "
         f"{float(df['runs'].sum()) - tot['total_runs']:+.0f}")

    # ------------------------------------------- STEP 2: per-event run laws
    emit("")
    emit("=" * 78)
    emit("STEP 2 — per-event extras-run laws")
    emit("=" * 78)
    runs = df["runs"].to_numpy(dtype=float)
    batter_runs = df["batter_runs"].to_numpy(dtype=float)

    wide_runs_ev = runs[is_wide]

    # Candidate no-ball run definitions. ORCHESTRATOR RULING 2026-08-03:
    # `noball_runs_channel_only` is OPERATIVE. The other two stay measured
    # and printed as the permanent discrepancy record.
    NB_OPERATIVE = "noball_runs_channel_only"
    nb_candidates = {
        "noball_runs_channel_only": df["noball_runs"].to_numpy(
            dtype=float)[is_nb],
        "extras_portion_runs_minus_batter": (runs - batter_runs)[is_nb],
        "total_delivery_runs": runs[is_nb],
    }
    if not np.all(np.equal(np.mod(wide_runs_ev, 1), 0)):
        raise SystemExit("wide event runs are not integral")
    for k, arr in nb_candidates.items():
        if not np.all(np.equal(np.mod(arr, 1), 0)):
            raise SystemExit(f"no-ball runs are not integral under {k}")

    wide_law = law_from_counts(Counter(int(v) for v in wide_runs_ev))
    nb_candidate_laws = {
        k: law_from_counts(Counter(int(v) for v in v_arr))
        for k, v_arr in nb_candidates.items()
    }
    nb_law = nb_candidate_laws[NB_OPERATIVE]

    emit(f"  WIDE (total runs on the delivery; all runs on a wide are extras)")
    emit(f"    n_events {wide_law['n_events']:,}   mean {wide_law['mean']:.6f}   "
         f"anchor {ANCHOR_WIDE_MEAN} (tol {MEAN_TOL})")
    emit("    " + "  ".join(
        f"{s}:{p:.6f}" for s, p in zip(wide_law["support"], wide_law["probs"])))
    emit("    counts " + "  ".join(
        f"{k}x{v}" for k, v in wide_law["counts"].items()))
    emit(f"  NO_BALL (OPERATIVE per the orchestrator ruling: "
         f"`noball_runs` channel ONLY)")
    emit(f"    n_events {nb_law['n_events']:,}   mean {nb_law['mean']:.6f}   "
         f"anchor {ANCHOR_NB_MEAN} (tol {MEAN_TOL})")
    emit("    " + "  ".join(
        f"{s}:{p:.6f}" for s, p in zip(nb_law["support"], nb_law["probs"])))
    emit("    counts " + "  ".join(
        f"{k}x{v}" for k, v in nb_law["counts"].items()))
    nb_offbat = float(batter_runs[is_nb].mean())
    emit(f"  context: mean OFF-BAT runs on no-balls = {nb_offbat:.6f} "
         f"(deliberately NOT credited — the 6-class labels carry it)")
    emit(f"  context: mean TOTAL runs on no-balls   = "
         f"{float(runs[is_nb].mean()):.6f}")

    emit("")
    emit("  candidate no-ball run definitions (permanent discrepancy record; "
         "OPERATIVE marked *):")
    for k, law in nb_candidate_laws.items():
        emit(f"   {'*' if k == NB_OPERATIVE else ' '}{k:<36} "
             f"mean {law['mean']:.6f}  "
             f"|mean - {ANCHOR_NB_MEAN}| = "
             f"{abs(law['mean'] - ANCHOR_NB_MEAN):.6f}")

    d_w = abs(wide_law["mean"] - ANCHOR_WIDE_MEAN)
    d_nb = abs(nb_law["mean"] - ANCHOR_NB_MEAN)
    emit("")
    emit(f"  |wide mean - anchor|    = {d_w:.6f}  -> "
         f"{'OK' if d_w <= MEAN_TOL else 'OUT OF TOLERANCE'}")
    emit(f"  |no-ball mean - anchor| = {d_nb:.6f}  -> "
         f"{'OK' if d_nb <= MEAN_TOL else 'OUT OF TOLERANCE'}")
    mean_stop = d_w > MEAN_TOL or d_nb > MEAN_TOL

    # --------------------------------------- STEP 3: analytic g pre-check
    emit("")
    emit("=" * 78)
    emit("STEP 3 — analytic run-mass pre-check (B17 arithmetic, re-derived)")
    emit("=" * 78)
    enc_log: list = []
    enc = encode_frame(df, MODEL_DIR, SUFFIX, feat, enc_log)
    for line in enc_log:
        emit(line)
    model = joblib.load(MODEL_DIR / f"xgboost_model_{SUFFIX}.pkl")
    probs = model.predict_proba(enc[feat])
    r_model = float((probs @ RUNS_BY_CLASS).mean())
    act = actual_channels(df)
    A = act["A_runs_per_legal_ball"]

    m_old = r_model + GRAFT_TOTAL
    g_old = m_old - A
    carried_new = (p_wide * wide_law["mean"] + p_no_ball * nb_law["mean"]) \
        / (1.0 - p_wide - p_no_ball)
    m_new = r_model + carried_new
    g_new = m_new - A

    emit(f"  R_model (venue_on, all delivery rows) = {r_model:.6f}")
    emit(f"  A (actual runs per legal ball)        = {A:.6f}   "
         f"(B17 {B17_A:.6f}, delta {A - B17_A:+.3e})")
    emit(f"  actual extras channel per legal ball  = "
         f"{act['extras_actual_per_legal_ball']:.6f}   "
         f"(B17 {B17_EXTRAS_ACTUAL:.6f})")
    emit("")
    emit(f"  OLD graft: carried extras {GRAFT_TOTAL:.6f}  M {m_old:.6f}  "
         f"g {g_old:+.6f}   (B17 logged {B17_G_I7:+.6f}, delta "
         f"{g_old - B17_G_I7:+.3e})")
    emit(f"  NEW graft: carried extras = (p_w*r_w + p_nb*r_nb)/(1-p_w-p_nb)")
    emit(f"             = ({p_wide:.6f}*{wide_law['mean']:.6f} + "
         f"{p_no_ball:.6f}*{nb_law['mean']:.6f}) / "
         f"{1.0 - p_wide - p_no_ball:.6f}")
    emit(f"             = {carried_new:.6f}")
    emit(f"             M {m_new:.6f}  g {g_new:+.6f}")
    emit(f"  predicted improvement in carried mass: "
         f"{carried_new - GRAFT_TOTAL:+.6f} runs per legal ball")
    emit(f"  predicted residual channels (byes/leg-byes on legal balls, "
         f"threes fold, 6-class head): {g_new:+.6f}")
    emit("")
    emit(f"  PRE-COMMITTED TOLERANCE: g_new in [{G_NEW_LO}, {G_NEW_HI}]")
    tol_ok = G_NEW_LO <= g_new <= G_NEW_HI
    emit(f"    g_new = {g_new:+.6f}  ->  {'IN TOLERANCE' if tol_ok else 'OUT OF TOLERANCE'}")
    emit(f"  expected cp6 quote-bias shrink ratio |g_new/g_old| = "
         f"{abs(g_new / g_old):.4f}  "
         f"(plan projection: -4.78 -> {-4.781 * abs(g_new / g_old):.2f} runs)")

    emit("")
    emit("  sensitivity — analytic g under EVERY candidate no-ball run law:")
    cand_g = {}
    for k, law in nb_candidate_laws.items():
        c = (p_wide * wide_law["mean"] + p_no_ball * law["mean"]) \
            / (1.0 - p_wide - p_no_ball)
        gk = r_model + c - A
        cand_g[k] = {"nb_mean": law["mean"], "carried_extras": c, "g_new": gk,
                     "in_tolerance": bool(G_NEW_LO <= gk <= G_NEW_HI)}
        emit(f"    {k:<36} r_nb {law['mean']:.6f}  carried {c:.6f}  "
             f"g_new {gk:+.6f}  "
             f"{'IN' if G_NEW_LO <= gk <= G_NEW_HI else 'OUT OF'} tolerance")

    if mean_stop:
        emit("")
        emit("=" * 78)
        emit("STOP — PRE-COMMITTED SANITY CHECK TRIPPED")
        emit("=" * 78)
        emit("  The plan's operative clause defines the no-ball event runs as")
        emit("  'extras-portion only (exclude off-bat runs)' == "
             "runs - batter_runs,")
        emit("  and its 'Easy to get wrong' note glosses that as "
             "'penalty + byes'.")
        emit(f"  That quantity measures {nb_law['mean']:.6f} on this "
             f"population, not the")
        emit(f"  plan's sanity anchor {ANCHOR_NB_MEAN} "
             f"(|delta| {d_nb:.6f} > tol {MEAN_TOL}).")
        emit("  The anchor corresponds to the NARROWER `noball_runs` channel "
             "alone")
        emit("  (no-ball penalty, excluding byes/leg-byes on the same "
             "delivery):")
        for k, law in nb_candidate_laws.items():
            emit(f"    {k:<36} mean {law['mean']:.6f}")
        emit("  Per the plan: STOP and record; do NOT proceed to evals.")
        emit("  NOTE (arithmetic, not a decision): the pre-committed ANALYTIC")
        emit(f"  tolerance g_new in [{G_NEW_LO}, {G_NEW_HI}] is met under "
             "EVERY candidate")
        emit("  definition above, so the B17 attribution arithmetic is not "
             "broken;")
        emit("  the mismatch is between the plan's prose definition and its "
             "own")
        emit("  sanity anchor. Disambiguation is the orchestrator's call.")
        emit("  NO SIDECAR WRITTEN.")
        Path(args.out_txt).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_txt).write_text("\n".join(log) + "\n")
        raise SystemExit(3)

    if not tol_ok:
        emit("  STOP — predicted g outside the pre-committed tolerance; "
             "no sidecar written, no eval launched.")
        Path(args.out_txt).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_txt).write_text("\n".join(log) + "\n")
        raise SystemExit(4)

    # ------------------------------------------------------ STEP 4: sidecar
    payload = {
        "version": "extras_graft_v1",
        "idea": "B18",
        "p_wide": round(p_wide, 6),
        "p_no_ball": round(p_no_ball, 6),
        "wide_runs": {k: wide_law[k] for k in ("support", "probs", "mean")},
        "no_ball_runs": {k: nb_law[k] for k in ("support", "probs", "mean")},
        "fit": {
            "population": str(PARQUET.relative_to(REPO)),
            "population_hash": pop_hash,
            "parquet_md5": md5_of(PARQUET),
            "n_deliveries": n_deliveries,
            "n_legal": n_legal,
            "n_matches": int(mids.nunique()),
            "date_min": str(df["match_date"].min()),
            "date_max": str(df["match_date"].max()),
            "n_wide_events": wide_law["n_events"],
            "n_no_ball_events": nb_law["n_events"],
            "wide_counts": wide_law["counts"],
            "no_ball_counts": nb_law["counts"],
            "p_wide_full_precision": p_wide,
            "p_no_ball_full_precision": p_no_ball,
            "d3_anchor_p_wide": D3_P_WIDE,
            "d3_anchor_p_no_ball": D3_P_NO_BALL,
            "no_ball_run_definition":
                "noball_runs channel only (orchestrator ruling 2026-08-03)",
            "no_ball_run_definition_candidates": {
                k: law["mean"] for k, law in nb_candidate_laws.items()},
            "wide_run_definition": "runs (all runs on a wide are extras)",
            "no_ball_mean_offbat_context": nb_offbat,
        },
        "analytic_precheck": {
            "R_model_venue_on_all_rows": r_model,
            "A_runs_per_legal_ball": A,
            "extras_actual_per_legal_ball": act["extras_actual_per_legal_ball"],
            "carried_extras_old": GRAFT_TOTAL,
            "carried_extras_new": carried_new,
            "g_old": g_old,
            "g_new": g_new,
            "tolerance": [G_NEW_LO, G_NEW_HI],
            "in_tolerance": bool(tol_ok),
            "b17_g_i7": B17_G_I7,
        },
    }
    # The stored rates are ROUNDED to 6 dp (D3's pre-committed precision);
    # the analytic pre-check above used full precision. Confirm the rounding
    # does not move the predicted g materially.
    carried_rounded = (payload["p_wide"] * wide_law["mean"]
                       + payload["p_no_ball"] * nb_law["mean"]) \
        / (1.0 - payload["p_wide"] - payload["p_no_ball"])
    payload["analytic_precheck"]["carried_extras_new_at_stored_rates"] = \
        carried_rounded
    payload["analytic_precheck"]["g_new_at_stored_rates"] = \
        r_model + carried_rounded - A
    emit(f"  g_new at the STORED (6 dp) rates = "
         f"{r_model + carried_rounded - A:+.6f}")

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, default=float))
    emit("")
    emit(f"  wrote {out_json}")

    # Round-trip through the engine's own loader.
    from sim_v1_2 import ExtrasGraftConfig  # noqa: E402
    cfg = ExtrasGraftConfig.from_path(out_json)
    emit(f"  engine loader round-trip OK: {cfg.banner()}")

    out_txt = Path(args.out_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(log) + "\n")
    print(f"wrote {out_txt}")


if __name__ == "__main__":
    main()
