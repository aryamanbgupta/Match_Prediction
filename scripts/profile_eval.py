"""
Profile script for sim_eval hot paths.

Runs a small reproducible evaluation (few matches, few sims) under cProfile
and prints the top cumulative-time functions. Also separately times:
  - model load
  - stats_provider load
  - per-ball `extract_features` + `predict_next_ball`
  - a full simulate_match call

so we can distinguish per-ball overhead from fixed-cost startup.

Usage:
    uv run python scripts/profile_eval.py --matches 2 --n-sims 20
"""

import argparse
import cProfile
import pstats
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sim_v1_2 import (
    SimulationEngine, SimulationConfig, XGBoostModelV2, T20Rules, ResultAggregator,
)
from stats_provider import StatsProvider
from player_metadata import PlayerMetadataProvider
from sim_eval.loaders import TestMatchLoader, BettingOddsLoader
from sim_eval.match_evaluator import MatchLevelEvaluator


def time_block(label):
    class _Timer:
        def __enter__(self):
            self.t0 = time.perf_counter()
            return self
        def __exit__(self, *exc):
            dt = time.perf_counter() - self.t0
            print(f"  [timing] {label}: {dt*1000:.1f} ms")
    return _Timer()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matches", type=int, default=2)
    p.add_argument("--n-sims", type=int, default=20)
    p.add_argument("--test-dir", default="data/betting_test")
    p.add_argument("--odds", default="betting_odds_v3.json")
    p.add_argument("--profile-out", default="eval_profile.prof")
    args = p.parse_args()

    print(f"=== Profiling eval with {args.matches} matches x {args.n_sims} sims ===\n")

    # --- Setup ---
    with time_block("StatsProvider load"):
        stats_provider = StatsProvider("models", version="v3")

    with time_block("PlayerMetadataProvider load"):
        player_metadata = PlayerMetadataProvider("data/all_players_enriched.csv")

    with time_block("XGBoost model load"):
        model = XGBoostModelV2(
            model_path="models/xgb_v3/xgboost_model_v3.pkl",
            batter_encoder_path="models/xgb_v3/batter_encoder_v3.pkl",
            bowler_encoder_path="models/xgb_v3/bowler_encoder_v3.pkl",
            feature_columns_path="models/xgb_v3/feature_columns_v3.txt",
            stats_provider=stats_provider,
            player_metadata=player_metadata,
        )

    engine = SimulationEngine(model, T20Rules())

    with time_block("TestMatchLoader.load_matches"):
        match_loader = TestMatchLoader()
        matches = match_loader.load_matches(args.test_dir)

    odds_lookup = BettingOddsLoader.load_odds(args.odds)

    # Only keep matches that have odds (same behavior as eval script) and clip
    matches_with_odds = [(mid, st) for mid, st in matches if mid in odds_lookup]
    matches_subset = matches_with_odds[: args.matches]
    print(f"\nRunning on {len(matches_subset)} matches (of {len(matches_with_odds)} with odds)")

    # --- Micro-benchmark one ball ---
    sample_match_id, sample_state = matches_subset[0]
    st = sample_state.copy()
    # make sure at least one ball can be simulated: innings 1 start is fine
    print("\n--- Per-ball micro-benchmark (initial state, 200 calls) ---")
    N = 200
    t0 = time.perf_counter()
    for _ in range(N):
        feats = model.extract_features(st)
    dt_extract = (time.perf_counter() - t0) / N * 1000
    print(f"  extract_features: {dt_extract:.2f} ms/ball")

    t0 = time.perf_counter()
    for _ in range(N):
        probs = model.predict_next_ball(feats)
    dt_predict = (time.perf_counter() - t0) / N * 1000
    print(f"  predict_next_ball: {dt_predict:.2f} ms/ball")

    print(f"  combined: {dt_extract + dt_predict:.2f} ms/ball")
    balls_per_match_est = 240  # ~120 balls/innings x 2 innings
    per_match_ms = (dt_extract + dt_predict) * balls_per_match_est
    print(f"  -> ~{per_match_ms:.0f} ms per simulation ({balls_per_match_est} balls est.)")
    print(f"  -> ~{per_match_ms * args.n_sims / 1000:.1f} s per match ({args.n_sims} sims)")
    print(f"  -> ~{per_match_ms * args.n_sims * 44 / 60000:.1f} min extrapolated for 44 matches")

    # --- Full-match wall time ---
    print("\n--- Full simulate_match timing ---")
    t0 = time.perf_counter()
    _ = engine.simulate_match(sample_state, "bench")
    dt_sim = time.perf_counter() - t0
    print(f"  1 full match sim: {dt_sim*1000:.0f} ms")

    # --- cProfile the real eval path on a tiny subset ---
    print(f"\n--- cProfile: evaluate_all on {len(matches_subset)} matches x {args.n_sims} sims ---")
    evaluator = MatchLevelEvaluator(
        model=model, simulation_engine=engine,
        n_simulations=args.n_sims, parallel=False,
    )

    profiler = cProfile.Profile()
    t_start = time.perf_counter()
    profiler.enable()
    results = evaluator.evaluate_all(matches_subset, odds_lookup)
    profiler.disable()
    elapsed = time.perf_counter() - t_start
    print(f"\n  total wall time: {elapsed:.1f} s for "
          f"{results.n_matches} matches x {args.n_sims} sims")

    profiler.dump_stats(args.profile_out)
    print(f"  profile dumped to {args.profile_out}")

    # Top cumulative time functions
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(30)
    print("\n--- Top 30 by cumulative time ---")
    print(s.getvalue())

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("tottime")
    ps.print_stats(25)
    print("\n--- Top 25 by tottime (self time, excluding callees) ---")
    print(s.getvalue())


if __name__ == "__main__":
    main()
