#!/usr/bin/env python3
"""Replay held-out rows through T1's live simulator feature path.

This is a feature-contract audit, not a model evaluation. It reconstructs
each pre-ball MatchState from the ordinary v3 test parquet, causally feeds
completed deliveries into the online EB overlay, and compares the resulting
50-vector with `transformer_t1.build_features` row by row. Sealed golden and
forward-holdout paths are rejected.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from embeddings_e1 import EB_BAT_COLS, EB_BOWL_COLS, VENUE_COLS  # noqa: E402
from player_metadata import PlayerMetadataProvider  # noqa: E402
from registered_experiment import reject_sealed, sha256  # noqa: E402
from sim_eval.loaders import TestMatchLoader  # noqa: E402
from sim_eval.same_day_stats import SameDayReplayStatsProvider  # noqa: E402
from sim_t1 import TransformerT1SimModel  # noqa: E402
from stats_provider import StatsProvider  # noqa: E402
from t1_ppc_common import build_context_by_date, replay_selected  # noqa: E402
from transformer_t1 import build_features  # noqa: E402


FEATURE_NAMES = (
    EB_BAT_COLS + EB_BOWL_COLS + VENUE_COLS
    + ["is_middle_overs", "is_death_overs", "wickets_in_hand_scaled",
       "chasing", "balls_remaining_scaled", "score_scaled",
       "run_rate_scaled", "run_rate_required_scaled"]
)
ROW_COLS = [
    "innings_id", "inning_idx", "over_idx", "ball_idx", "batter_id",
    "bowler_id", "score", "wickets", "balls_bowled", "chase_target",
    "run_rate", "run_rate_required", "balls_remaining", "wickets_in_hand",
    "is_middle_overs", "is_death_overs", "team_runs", "is_wicket",
] + EB_BAT_COLS + EB_BOWL_COLS + VENUE_COLS


def _player_index(lineup, player_id) -> int:
    target = str(player_id)
    for i, player in enumerate(lineup.players):
        if str(player.player_id) == target:
            return i
    raise RuntimeError(f"player {target} is absent from {lineup.team_name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--parquet", type=Path,
        default=Path("data/xgb_data_v3/cricket_data_v3_test.parquet"))
    ap.add_argument("--match-dir", type=Path, default=Path("data/t1_eval"))
    ap.add_argument("--context-dir", type=Path,
                    default=Path("data/t20s_json"))
    ap.add_argument("--stats-version", default="v3")
    ap.add_argument("--metadata", type=Path,
                    default=Path("data/all_players_enriched.csv"))
    ap.add_argument("--max-matches", type=int, default=None)
    ap.add_argument("--tolerance", type=float, default=1e-6)
    ap.add_argument(
        "--out", type=Path,
        default=Path("research/reports/embeddings/artifacts/"
                     "t1_sim_feature_parity.json"))
    args = ap.parse_args()
    for path in (args.parquet, args.match_dir, args.context_dir):
        reject_sealed(path)

    match_files = sorted(args.match_dir.glob("*.json"))
    if args.max_matches is not None:
        match_files = match_files[:args.max_matches]
    match_ids = {path.stem for path in match_files}
    if not match_ids:
        raise SystemExit(f"no match JSON files found in {args.match_dir}")

    frame = pd.read_parquet(args.parquet, columns=ROW_COLS)
    frame["match_id"] = frame["innings_id"].str.split("_", n=1).str[1]
    frame = frame[frame["match_id"].isin(match_ids)].copy()
    n_nonregulation = int((frame["inning_idx"] > 2).sum())
    # SimulationEngine currently models the two regulation innings only.
    # Super-over rows are valid T1 training examples but have no serving
    # MatchState contract to audit here.
    frame = frame[frame["inning_idx"] <= 2].copy()
    frame["audit_pos"] = np.arange(len(frame), dtype=np.int64)
    expected = build_features(frame)

    metadata = PlayerMetadataProvider(str(args.metadata))
    provider = SameDayReplayStatsProvider(
        StatsProvider("models", version=args.stats_version), metadata)
    # Feature-path-only wrapper: identical `_ball_features` path to serving,
    # no torch checkpoint load (the wrap-identity assert lives inside).
    wrapper = TransformerT1SimModel.for_feature_audit(provider, metadata)
    loader = TestMatchLoader()

    max_by_feature = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    row_maxima = []
    n_over_tolerance = 0

    selected_dates = set()
    for path in match_files:
        with path.open() as fh:
            document = json.load(fh)
        selected_dates.add(str(document["info"]["dates"][0]))

    context_by_date = build_context_by_date(args.context_dir, selected_dates)
    groups = {str(mid): group for mid, group in
              frame.groupby("match_id", sort=False)}

    def on_match(match_id: str, document: dict, date_str: str) -> None:
        nonlocal n_over_tolerance
        group = groups[match_id]
        _, state = loader.create_match_state(document, cricsheet_id=match_id)
        if state is None:
            raise RuntimeError(f"could not build MatchState for {match_id}")

        for _, row in group.iterrows():
            state.innings = int(row["inning_idx"])
            state.balls = int(row["balls_bowled"])
            batting_team = state.current_team_idx
            state.runs[batting_team] = float(row["score"])
            state.wickets[batting_team] = int(row["wickets"])
            if state.innings == 2:
                state.runs[1 - batting_team] = (
                    float(row["chase_target"]) - 1)
            state.striker_idx = _player_index(
                state.batting_lineup, row["batter_id"])
            state.bowler_idx = _player_index(
                state.bowling_lineup, row["bowler_id"])

            actual = wrapper._ball_features(state)
            delta = np.abs(actual - expected[int(row["audit_pos"])])
            np.maximum(max_by_feature, delta, out=max_by_feature)
            row_max = float(delta.max())
            row_maxima.append(row_max)
            n_over_tolerance += int(row_max > args.tolerance)

            h = state.history_idx
            state.history[h] = [
                state.innings, int(row["over_idx"]),
                int(row["ball_idx"]), int(row["team_runs"]),
                int(row["is_wicket"]), batting_team,
                state.striker_idx, 1 - batting_team, state.bowler_idx,
            ]
            state.history_idx += 1

    audited_matches, context_matches = replay_selected(
        context_by_date, provider, set(groups), on_match)

    row_maxima = np.asarray(row_maxima, dtype=np.float64)
    result = {
        "contract": "t1_sim_feature_parity_v1",
        "sealed_data_used": False,
        "parquet": str(args.parquet),
        "match_dir": str(args.match_dir),
        "context_dir": str(args.context_dir),
        "stats_version": args.stats_version,
        "n_matches": audited_matches,
        "n_context_matches_replayed": context_matches,
        "n_rows": int(len(frame)),
        "rows_excluded_super_over": n_nonregulation,
        "n_features": len(FEATURE_NAMES),
        "tolerance": args.tolerance,
        "rows_over_tolerance": n_over_tolerance,
        "max_abs_delta": float(row_maxima.max()),
        "row_max_abs_delta_quantiles": {
            str(q): float(np.quantile(row_maxima, q))
            for q in (0.5, 0.9, 0.99, 1.0)
        },
        "max_abs_delta_by_feature": {
            name: float(value)
            for name, value in zip(FEATURE_NAMES, max_by_feature)
        },
        "status": "PASS" if n_over_tolerance == 0 else "FAIL",
        "sha256": {
            "parquet": sha256(args.parquet),
            "stats_cache": sha256(
                Path("models")
                / f"player_stats_cache_{args.stats_version}.sqlite"),
            "sim_t1_source": sha256(Path("scripts/sim_t1.py")),
            "transformer_t1_source": sha256(
                Path("scripts/transformer_t1.py")),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
