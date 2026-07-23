"""B5 — in-play over/under quote prototype (analytics-engine seed).

Teacher-forced replay of ACTUAL innings-1 balls to checkpoint states (end
of overs 6/10/15 = after exactly 36/60/90 legal deliveries), then n-sim
continuations from each state -> P10/P50/P90 remaining-innings-runs quotes,
scored vs realized remaining runs and vs a naive run-rate-extrapolation
baseline (current run rate x remaining overs).

Non-engine harness: sim_v1_2.py untouched; runs the CURRENT default sim
path (venue-ON sidecar autoload, run_rate-aligned features, D15
attribution unit) with the stale v1 vector calibrator and the
EmpiricalBowlerSelector — identical model config to the canonical D15 run.

Pre-declared scope (see also b5_gate_analysis.py):
  - Innings 1 only.
  - Match filter: info.overs == 20 AND innings 1 completed (>=120 legal
    balls or >=10 dismissals) — excludes rain-curtailed innings where
    "remaining runs" is not a well-defined target. Loader must build a
    state and its toss-derived batting_first must match innings[0].team.
  - Checkpoint filter: innings 1 continued beyond the checkpoint
    (legal balls > 6*cp) AND the over structure yields exactly 6*cp legal
    balls at the snapshot.
  - Replay fidelity: striker/non-striker/bowler indices FORCED from the
    actual delivery each ball; state.update() fed the actual total runs.
    Replay parity (runs / legal balls / wickets vs an independent
    aggregation of the raw JSON) is hard-asserted per match.
  - Quote: P50 is the point quote scored by MAE; P10-P90 is the coverage
    band (inclusive). Naive baseline is a point forecast only.

Continuation semantics: per-sim seed = seed + i (mirrors
SimulationEngine._simulate_sequential); the next-over bowler is selected
per-sim via T20Rules.select_next_bowler on the state copy — exactly the
engine's own over-boundary flow (simulate_ball selects the next bowler
after an over-final ball; state.update() alone does not).

Run:
  uv run python scripts/auto/b5_inplay_quotes.py \
      --test-dir data/polymarket_test --n-sims 100 --seed 43 \
      --out models/auto/b5/quotes_s43_n261.json
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from sim_v1_2 import (  # noqa: E402
    EmpiricalBowlerSelector,
    Outcome,
    SimulationEngine,
    T20Rules,
    XGBoostModelV2,
)

CHECKPOINTS = (6, 10, 15)
# Cricsheet wickets[] kinds that are NOT dismissals for the innings-
# completed rain filter (the replay itself applies every wickets[] entry,
# matching the training convention: parsing_v2.py labels any wickets[]
# delivery as WICKET).
NON_DISMISSAL_KINDS = {"retired hurt", "retired not out", "retired out"}


class ReplayError(Exception):
    pass


# ---------------------------------------------------------------------------
# Independent innings aggregation (ground truth for parity asserts).
# ---------------------------------------------------------------------------

def innings_summary(inn):
    """(total_runs, legal_balls, dismissals, wicket_entries) from raw JSON."""
    total = legal = dismissals = wicket_entries = 0
    for ov in inn.get("overs", []):
        for d in ov.get("deliveries", []):
            total += d["runs"]["total"]
            ex = d.get("extras", {})
            if "wides" not in ex and "noballs" not in ex:
                legal += 1
            for w in d.get("wickets", []):
                wicket_entries += 1
                if w.get("kind") not in NON_DISMISSAL_KINDS:
                    dismissals += 1
    return total, legal, dismissals, wicket_entries


# ---------------------------------------------------------------------------
# Teacher-forced replay.
# ---------------------------------------------------------------------------

def classify(d):
    """Map a cricsheet delivery to (Outcome, runs_for_update, wickets_list).

    The runs param carries the exact total (state.update() adds it to the
    team score verbatim); the Outcome tag only drives legal-ball counting,
    wicket processing and strike rotation parity. Wicket on a wide/no-ball
    is returned as WIDE/NO_BALL with the wicket left in the list for
    manual application (state.update() processes wickets only on WICKET).
    """
    ex = d.get("extras", {})
    wk = list(d.get("wickets", []))
    total = d["runs"]["total"]
    batter = d["runs"]["batter"]
    if "wides" in ex:
        return Outcome.WIDE, total, wk
    if "noballs" in ex:
        return Outcome.NO_BALL, total, wk
    if wk:
        return Outcome.WICKET, total, wk
    if batter == 4:
        return Outcome.FOUR, total, []
    if batter == 6:
        return Outcome.SIX, total, []
    if total == 0:
        return Outcome.DOT, total, []
    if total == 1:
        return Outcome.ONE, total, []
    return Outcome.TWO, total, []  # 2s/3s/byes/legbyes/penalty: runs=truth


def apply_manual_dismissal(state, out_name, bat_idx):
    """Apply a dismissal that state.update() did not process (wicket on a
    wide/no-ball, or a second wicket entry on one delivery)."""
    ti = state.current_team_idx
    out_idx = bat_idx.get(out_name)
    state.wickets[ti] += 1
    if out_idx is not None and out_idx == state.non_striker_idx:
        state.batsmen_out[ti].append(out_idx)
        state.non_striker_idx = state.get_next_batsman_idx()
    else:
        state.batsmen_out[ti].append(
            out_idx if out_idx is not None else state.striker_idx)
        if out_idx is None or out_idx == state.striker_idx:
            state.striker_idx = state.get_next_batsman_idx()
    state.partnership_runs = 0


def replay_delivery(state, d, bat_idx, bowl_idx):
    try:
        s_i = bat_idx[d["batter"]]
        ns_i = bat_idx[d["non_striker"]]
        b_i = bowl_idx[d["bowler"]]
    except KeyError as e:
        raise ReplayError(f"unmapped player {e}")
    # Force the actual participants — replay never trusts the sim's own
    # rotation/next-batter bookkeeping between balls.
    state.striker_idx = s_i
    state.non_striker_idx = ns_i
    state.bowler_idx = b_i

    outcome, runs, wickets = classify(d)
    manual = list(wickets)
    dismissal = None
    if outcome == Outcome.WICKET:
        w0 = manual.pop(0)
        kind = w0.get("kind", "")
        po = w0.get("player_out", "")
        if po == d["non_striker"]:
            dismissal = "runout_nonstriker"
        elif kind == "run out":
            dismissal = "runout_striker"
        else:
            dismissal = "bowler"
    state.update(outcome, runs, dismissal=dismissal)
    for w in manual:
        apply_manual_dismissal(state, w.get("player_out", ""), bat_idx)


def build_checkpoint_states(state, inn1, checkpoints=CHECKPOINTS):
    """Replay innings 1 on `state` (mutates it to the innings-1 end state).

    Returns {cp: snapshot} for every checkpoint where the replay reached
    exactly 6*cp legal balls at the end of over cp-1.
    """
    bat_idx = {p.name: i for i, p in enumerate(state.batting_lineup.players)}
    bowl_idx = {p.name: i for i, p in enumerate(state.bowling_lineup.players)}
    snaps = {}
    for ov in inn1.get("overs", []):
        for d in ov.get("deliveries", []):
            replay_delivery(state, d, bat_idx, bowl_idx)
        cp = ov["over"] + 1
        if cp in checkpoints and state.balls == cp * 6:
            snaps[cp] = state.copy()
    return snaps


# ---------------------------------------------------------------------------
# Per-match processing.
# ---------------------------------------------------------------------------

def prepare_match(fp, loader, checkpoints=CHECKPOINTS):
    """Scope-filter + replay one match.

    Returns (match_id, snaps, meta) or raises ReplayError with the skip
    reason. meta = dict(total, legal, ti).
    """
    with open(fp) as f:
        data = json.load(f)
    info = data["info"]
    if info.get("overs") != 20:
        raise ReplayError(f"not a 20-over match (overs={info.get('overs')})")
    if not data.get("innings"):
        raise ReplayError("no innings data")
    match_id, state = loader._create_match_state(data)
    if state is None:
        raise ReplayError("loader could not build state")
    inn1 = data["innings"][0]
    if inn1.get("team") != state.batting_first:
        raise ReplayError(
            f"batting_first mismatch (toss-derived {state.batting_first!r} "
            f"vs innings[0] {inn1.get('team')!r})")
    total, legal, dismissals, wicket_entries = innings_summary(inn1)
    if legal < 120 and dismissals < 10:
        raise ReplayError(
            f"innings 1 curtailed ({legal} legal balls, "
            f"{dismissals} dismissals)")

    snaps = build_checkpoint_states(state, inn1, checkpoints)

    # Replay parity — hard asserts vs the independent aggregation.
    ti = 0 if state.batting_first == state.team1 else 1
    got_runs = int(state.runs[ti])
    got_balls = int(state.balls)
    got_wkts = int(state.wickets[ti])
    if got_runs != total:
        raise ReplayError(f"parity: runs {got_runs} != {total}")
    if got_balls != legal:
        raise ReplayError(f"parity: balls {got_balls} != {legal}")
    if got_wkts != wicket_entries:
        raise ReplayError(f"parity: wickets {got_wkts} != {wicket_entries}")
    for cp, snap in snaps.items():
        if int(snap.balls) != cp * 6:
            raise ReplayError(f"parity: cp{cp} balls {int(snap.balls)}")

    return match_id, snaps, {"total": total, "legal": legal, "ti": ti}


def quote_checkpoint(cp_state, engine, rules, n_sims, seed):
    """n_sims seeded continuations; returns array of remaining runs."""
    rem = np.empty(n_sims, dtype=np.float64)
    for i in range(n_sims):
        random.seed(seed + i)
        np.random.seed(seed + i)
        st = cp_state.copy()
        st.bowler_idx = rules.select_next_bowler(st)
        res = engine._simulate_innings(st)
        rem[i] = float(res.total_runs)
    return rem


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="data/polymarket_test")
    ap.add_argument("--n-matches", default="all")
    ap.add_argument("--n-sims", type=int, default=100)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--out", default="models/auto/b5/quotes_s43_n261.json")
    args = ap.parse_args()

    from sim_eval.loaders import TestMatchLoader  # noqa: E402
    from stats_provider import StatsProvider  # noqa: E402
    from player_metadata import PlayerMetadataProvider  # noqa: E402
    import joblib

    print("Loading stats provider + player metadata + model ...")
    stats_provider = StatsProvider("models", version="v3")
    player_metadata = PlayerMetadataProvider("data/all_players_enriched.csv")
    ball_calibrator = joblib.load(
        "models/xgb_v3/vector_scaling_calibrator_v1.pkl")
    print("Ball calibrator: vector scaling "
          "(models/xgb_v3/vector_scaling_calibrator_v1.pkl)")
    model = XGBoostModelV2(
        model_path="models/xgb_v3/xgboost_model_v3.pkl",
        batter_encoder_path="models/xgb_v3/batter_encoder_v3.pkl",
        bowler_encoder_path="models/xgb_v3/bowler_encoder_v3.pkl",
        feature_columns_path="models/xgb_v3/feature_columns_v3.txt",
        stats_provider=stats_provider,
        player_metadata=player_metadata,
        ball_calibrator=ball_calibrator,
    )
    rules = T20Rules(EmpiricalBowlerSelector())
    engine = SimulationEngine(model, rules)
    loader = TestMatchLoader()

    files = sorted(Path(args.test_dir).glob("*.json"))
    if args.n_matches != "all":
        files = files[: int(args.n_matches)]
    print(f"B5 in-play quotes: {len(files)} matches x {args.n_sims} sims "
          f"x checkpoints {list(CHECKPOINTS)}, seed {args.seed}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_path.with_suffix(".partial.jsonl")
    jsonl = open(jsonl_path, "w")

    rows, skips = [], []
    t_start = time.time()
    for k, fp in enumerate(files):
        try:
            match_id, snaps, meta = prepare_match(fp, loader)
        except ReplayError as e:
            skips.append({"file": fp.name, "reason": str(e)})
            print(f"  [{k+1}/{len(files)}] SKIP {fp.name}: {e}")
            continue

        t0 = time.time()
        made = []
        for cp in sorted(snaps):
            if meta["legal"] <= cp * 6:
                continue  # innings ended exactly at the checkpoint
            cp_state = snaps[cp]
            runs_at_cp = int(cp_state.runs[meta["ti"]])
            wkts_at_cp = int(cp_state.wickets[meta["ti"]])
            actual_remaining = meta["total"] - runs_at_cp
            naive = (runs_at_cp / cp) * (20 - cp)
            rem = quote_checkpoint(cp_state, engine, rules,
                                   args.n_sims, args.seed)
            row = {
                "match_id": match_id,
                "file": fp.name,
                "checkpoint": cp,
                "runs_at_cp": runs_at_cp,
                "wkts_at_cp": wkts_at_cp,
                "actual_final": meta["total"],
                "actual_remaining": actual_remaining,
                "naive_remaining": float(naive),
                "sim_p10": float(np.percentile(rem, 10)),
                "sim_p50": float(np.percentile(rem, 50)),
                "sim_p90": float(np.percentile(rem, 90)),
                "sim_mean": float(rem.mean()),
                "sim_std": float(rem.std()),
                "n_sims": args.n_sims,
            }
            rows.append(row)
            made.append(cp)
            jsonl.write(json.dumps(row) + "\n")
        jsonl.flush()
        print(f"  [{k+1}/{len(files)}] {match_id[:55]:55s} cps={made} "
              f"({time.time() - t0:.1f}s)")

    elapsed = time.time() - t_start
    jsonl.close()
    payload = {
        "config": {
            "test_dir": args.test_dir,
            "n_sims": args.n_sims,
            "seed": args.seed,
            "checkpoints": list(CHECKPOINTS),
            "model": "models/xgb_v3/xgboost_model_v3.pkl",
            "ball_calibrator": "models/xgb_v3/vector_scaling_calibrator_v1.pkl",
            "bowler_selector": "empirical",
            "quote_center": "sim_p50",
            "elapsed_s": elapsed,
        },
        "rows": rows,
        "skips": skips,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=1)
    n_m = len({r["match_id"] for r in rows})
    print(f"\nDone in {elapsed:.1f}s — {len(rows)} quote rows from {n_m} "
          f"matches ({len(skips)} matches skipped) -> {out_path}")


if __name__ == "__main__":
    main()
