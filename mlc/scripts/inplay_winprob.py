"""In-play chase win-probability: does the ball-by-ball sim beat a coarse resource
baseline? This is the one place a ball sim *should* win — it knows who is at the
crease and the live state; a resource model (runs needed / balls left / wickets)
does not.

Method, on MLC 2025 chases (2nd innings):
  * Baseline: HistGradientBoosting trained on (runs_needed, balls_remaining,
    wickets_in_hand) from corpus 2nd innings (excluding MLC 2025) → P(chase wins).
    A strong, standard resource model.
  * Sim: reconstruct the real mid-chase MatchState at overs 5/10/15 (replay actual
    deliveries through the engine, then set the two crease batters from cricsheet),
    roll forward N times → fraction of sims the chase succeeds.
  * Score both against the actual result (chase won?) with log-loss / Brier / AUC.

If the sim ties the baseline, ball-resolution buys nothing for in-play either.

Usage:
  uv run python mlc/scripts/inplay_winprob.py --limit 5 --n-sims 120   # quick sanity
  uv run python mlc/scripts/inplay_winprob.py --n-sims 200             # full
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from sim_v1_2 import (  # noqa: E402
    SimulationEngine, T20Rules, XGBoostModelV2, EmpiricalBowlerSelector, Outcome)
from sim_eval.loaders import TestMatchLoader  # noqa: E402
from stats_provider import StatsProvider  # noqa: E402
from player_metadata import PlayerMetadataProvider  # noqa: E402

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

MLC_DIR = REPO / "mlc" / "data" / "mlc_2025"
CORPUS = REPO / "data" / "t20s_json"
OUT = REPO / "mlc" / "reports" / "mlc_inplay_winprob.md"
CHECKPOINTS = [30, 60, 90]  # legal balls = overs 5, 10, 15


# ---------- cricsheet parsing ----------
def deliveries(inn):
    """Yield (batter, non_striker, runs_total, is_wkt, is_wide, is_noball) in order."""
    for ov in inn.get("overs", []):
        for d in ov.get("deliveries", []):
            ex = d.get("extras", {}) or {}
            yield (d["batter"], d.get("non_striker"), d["runs"]["total"],
                   bool(d.get("wickets")), "wides" in ex, "noballs" in ex)


def innings_total(inn):
    return sum(d["runs"]["total"] for ov in inn.get("overs", []) for d in ov["deliveries"])


# ---------- baseline ----------
def build_baseline(limit_files=2600):
    """Train a resource model on corpus 2nd innings. Returns fitted clf."""
    import zipfile  # not needed; corpus is loose json
    X, y = [], []
    files = sorted(CORPUS.glob("*.json"))
    rng = random.Random(0)
    rng.shuffle(files)
    mlc_ids = {p.stem for p in MLC_DIR.glob("[0-9]*.json")}
    used = 0
    for fp in files:
        if used >= limit_files:
            break
        if fp.stem in mlc_ids:
            continue
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        info = d.get("info", {})
        if info.get("match_type") != "T20" or len(d.get("innings", [])) < 2:
            continue
        if (info.get("outcome", {}) or {}).get("result") in ("no result", "tie"):
            continue
        winner = (info.get("outcome", {}) or {}).get("winner")
        if not winner:
            continue
        inn1, inn2 = d["innings"][0], d["innings"][1]
        target = innings_total(inn1) + 1
        chasing_team = inn2.get("team")
        chase_won = int(winner == chasing_team)
        runs = wkts = balls = 0
        cps = set(CHECKPOINTS)
        for (_, _, rt, wk, wide, nb) in deliveries(inn2):
            runs += rt
            if not (wide or nb):
                balls += 1
            if wk:
                wkts += 1
            if balls in cps and not (wide or nb):
                if wkts < 10 and runs < target:
                    X.append([target - runs, 120 - balls, 10 - wkts])
                    y.append(chase_won)
                cps.discard(balls)
        used += 1
    clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08,
                                         max_iter=300, min_samples_leaf=40)
    clf.fit(np.array(X, float), np.array(y))
    print(f"baseline trained on {len(y)} chase states from {used} corpus matches "
          f"(base win rate {np.mean(y):.2f})")
    return clf


# ---------- sim engine ----------
def build_engine():
    sp = StatsProvider("models", version="v3")
    pm = PlayerMetadataProvider("data/all_players_enriched.csv")
    model = XGBoostModelV2(
        model_path="models/xgb_v3/xgboost_model_v3.pkl",
        batter_encoder_path="models/xgb_v3/batter_encoder_v3.pkl",
        bowler_encoder_path="models/xgb_v3/bowler_encoder_v3.pkl",
        feature_columns_path="models/xgb_v3/feature_columns_v3.txt",
        matchup_encoder_path="models/xgb_v3/matchup_encoder_v3.pkl",
        stats_provider=sp, player_metadata=pm, ball_calibrator=None)
    return SimulationEngine(model, T20Rules(EmpiricalBowlerSelector(
        usage_path="models/bowler_phase_usage.json")))


def outcome_for(rt, wk, wide, nb):
    if wk:
        return Outcome.WICKET
    if wide:
        return Outcome.WIDE
    if nb:
        return Outcome.NO_BALL
    return {0: Outcome.DOT, 1: Outcome.ONE, 2: Outcome.TWO,
            4: Outcome.FOUR, 6: Outcome.SIX}.get(rt, Outcome.ONE if rt % 2 else Outcome.DOT)


def reconstruct(state, data, checkpoint):
    """Replay actual 2nd innings to `checkpoint` legal balls; set crease batters."""
    info = data["info"]
    reg = (info.get("registry", {}) or {}).get("people", {})
    inn1, inn2 = data["innings"][0], data["innings"][1]
    first_total = innings_total(inn1)

    # put the engine in a fresh 2nd innings
    st = state
    st.innings = 2
    cti = st.current_team_idx                # chasing index
    st.runs = np.zeros(2); st.runs[1 - cti] = first_total
    st.wickets = np.zeros(2, dtype=int)
    st.balls = 0; st.current_over = []; st.partnership_runs = 0
    st.history = np.zeros((400, 9)); st.history_idx = 0
    st.bowler_balls = {}; st.batsman_stats = {}
    st.batsmen_out = {0: [], 1: []}
    st.striker_idx, st.non_striker_idx = 0, 1

    bat_lineup = st.batting_lineup
    id_to_idx = {p.player_id: i for i, p in enumerate(bat_lineup.players)}
    last_batter = last_non = None
    for (batter, non_striker, rt, wk, wide, nb) in deliveries(inn2):
        if st.balls >= checkpoint:
            break
        st.update(outcome_for(rt, wk, wide, nb), runs=int(rt))
        last_batter, last_non = batter, non_striker
    # override crease batters from the actual data (the key in-play signal)
    if last_batter in reg and reg[last_batter] in id_to_idx:
        st.striker_idx = id_to_idx[reg[last_batter]]
    if last_non in reg and reg[last_non] in id_to_idx:
        st.non_striker_idx = id_to_idx[reg[last_non]]
    return st, first_total


def sim_winprob(engine, base_state, first_total, n, seed):
    random.seed(seed); np.random.seed(seed)
    cti = base_state.current_team_idx
    target = first_total + 1
    wins = 0
    for _ in range(n):
        s = base_state.copy()
        guard = 0
        while not s.is_match_over() and guard < 200:
            engine.rules.simulate_ball(s, engine.model)
            guard += 1
        wins += int(s.runs[cti] >= target)
    return wins / n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap MLC matches (0=all)")
    ap.add_argument("--n-sims", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    clf = build_baseline()
    engine = build_engine()
    loader = TestMatchLoader()

    files = sorted(MLC_DIR.glob("[0-9]*.json"))
    if args.limit:
        files = files[:args.limit]

    rows = []
    for k, fp in enumerate(files):
        data = json.load(open(fp))
        if len(data.get("innings", [])) < 2:
            continue
        info = data["info"]
        winner = (info.get("outcome", {}) or {}).get("winner")
        if not winner or (info.get("outcome", {}) or {}).get("method"):  # skip D/L
            continue
        inn2 = data["innings"][1]
        chasing = inn2.get("team")
        chase_won = int(winner == chasing)
        first_total = innings_total(data["innings"][0])

        # actual checkpoint states (to know which are valid / not already decided)
        runs = wkts = balls = 0
        valid_cp = {}
        for (_, _, rt, wk, wide, nb) in deliveries(inn2):
            runs += rt
            if not (wide or nb):
                balls += 1
            if wk:
                wkts += 1
            if balls in CHECKPOINTS and not (wide or nb) and wkts < 10 and runs < first_total + 1:
                valid_cp[balls] = (first_total + 1 - runs, 120 - balls, 10 - wkts)

        for cp, (need, blleft, wih) in valid_cp.items():
            _, st = loader._create_match_state(data)
            st, ft = reconstruct(st, data, cp)
            sw = sim_winprob(engine, st, ft, args.n_sims, args.seed + cp)
            bw = float(clf.predict_proba([[need, blleft, wih]])[0, 1])
            rows.append({"match": fp.stem, "cp_over": cp // 6, "need": need,
                         "balls_left": blleft, "wkts_in_hand": wih,
                         "sim": sw, "base": bw, "won": chase_won})
        cps_str = {r['cp_over']: (round(r['sim'], 2), round(r['base'], 2))
                   for r in rows if r['match'] == fp.stem}
        print(f"  [{k+1}/{len(files)}] {chasing[:22]:22} won={chase_won} "
              f"sim/base by over: {cps_str}")

    if not rows:
        print("no valid chase states"); return 1
    y = np.array([r["won"] for r in rows])
    sim = np.clip(np.array([r["sim"] for r in rows]), 1e-4, 1 - 1e-4)
    base = np.clip(np.array([r["base"] for r in rows]), 1e-4, 1 - 1e-4)

    def metrics(p):
        auc = roc_auc_score(y, p) if len(set(y)) > 1 else float("nan")
        return (log_loss(y, p, labels=[0, 1]), brier_score_loss(y, p), auc,
                float(np.mean((p >= 0.5).astype(int) == y)))

    sm = metrics(sim); bm = metrics(base)
    L = ["# MLC 2025 — in-play chase win-probability: sim vs resource baseline\n",
         f"{len(rows)} mid-chase states (overs {CHECKPOINTS[0]//6}/{CHECKPOINTS[1]//6}/"
         f"{CHECKPOINTS[2]//6}) across the MLC 2025 chases, {args.n_sims} sims/state. "
         "Baseline = gradient boosting on (runs needed, balls left, wickets in hand) "
         "trained on the corpus. Lower log-loss / Brier and higher AUC = better.\n",
         "| predictor | log loss | Brier | AUC | accuracy |",
         "|---|---:|---:|---:|---:|",
         f"| resource baseline | {bm[0]:.4f} | {bm[1]:.4f} | {bm[2]:.3f} | {bm[3]*100:.0f}% |",
         f"| **ball-by-ball sim** | {sm[0]:.4f} | {sm[1]:.4f} | {sm[2]:.3f} | {sm[3]*100:.0f}% |",
         f"\n**Verdict:** sim − baseline log-loss = {sm[0]-bm[0]:+.4f} "
         f"({'sim better' if sm[0]<bm[0] else 'baseline better/equal'}); "
         f"AUC {sm[2]:.3f} vs {bm[2]:.3f}.\n",
         "Per-state detail (first 30):\n",
         "| match | over | need | balls | wkts | sim P | base P | won |",
         "|---|---:|---:|---:|---:|---:|---:|:--:|"]
    for r in rows[:30]:
        L.append(f"| {r['match']} | {r['cp_over']} | {r['need']} | {r['balls_left']} | "
                 f"{r['wkts_in_hand']} | {r['sim']:.2f} | {r['base']:.2f} | {r['won']} |")
    OUT.write_text("\n".join(L) + "\n")
    print(f"\nSIM  LL {sm[0]:.4f} Brier {sm[1]:.4f} AUC {sm[2]:.3f}")
    print(f"BASE LL {bm[0]:.4f} Brier {bm[1]:.4f} AUC {bm[2]:.3f}")
    print(f"report → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
