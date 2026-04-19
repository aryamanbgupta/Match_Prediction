"""Validate that XGBoost predict_proba on numpy matches DataFrame bit-for-bit,
and measure the speedup. Must run green before touching sim_v1_2.py.

Exit 0 on all-pass, 1 on any parity or perf failure.

Usage:
    uv run python scripts/validate_numpy_predict.py
"""

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Make imports mirror run_sim_eval.py
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "scripts"))

from sim_v1_2 import XGBoostModelV2, T20Rules, SimulationEngine  # noqa: E402
from stats_provider import StatsProvider  # noqa: E402
from player_metadata import PlayerMetadataProvider  # noqa: E402
from sim_eval.loaders import TestMatchLoader  # noqa: E402

MODEL_PATH = "models/xgb_v3/xgboost_model_v3.pkl"
BATTER_ENCODER_PATH = "models/xgb_v3/batter_encoder_v3.pkl"
BOWLER_ENCODER_PATH = "models/xgb_v3/bowler_encoder_v3.pkl"
MATCHUP_ENCODER_PATH = "models/xgb_v3/matchup_encoder_v3.pkl"
FEATURE_COLUMNS_PATH = "models/xgb_v3/feature_columns_v3.txt"
TEST_DIR = "data/betting_test"
METADATA_PATH = "data/all_players_enriched.csv"

N_PARITY_STATES = 100      # how many (state) feature-rows to compare
N_BENCH_CALLS = 1000
N_WARMUP_CALLS = 100
PARITY_ATOL = 1e-12
SPEEDUP_MIN = 4.0


def _build_model():
    stats_provider = StatsProvider("models", version="v3")
    player_metadata = PlayerMetadataProvider(METADATA_PATH)
    model = XGBoostModelV2(
        model_path=MODEL_PATH,
        batter_encoder_path=BATTER_ENCODER_PATH,
        bowler_encoder_path=BOWLER_ENCODER_PATH,
        feature_columns_path=FEATURE_COLUMNS_PATH,
        stats_provider=stats_provider,
        player_metadata=player_metadata,
        matchup_encoder_path=MATCHUP_ENCODER_PATH,
    )
    return model


def _collect_feature_rows(model, n_rows):
    """Walk a handful of real matches ball-by-ball, returning a list of
    (feature_dataframe, feature_numpy) pairs built from the SAME dict."""
    loader = TestMatchLoader()
    matches = loader.load_matches(TEST_DIR)
    if not matches:
        raise RuntimeError(f"No matches loaded from {TEST_DIR}")

    engine = SimulationEngine(model, T20Rules())
    rules = engine.rules
    cols = model.feature_columns

    rows = []
    np.random.seed(1)
    import random as _random
    _random.seed(1)

    for match_id, init_state in matches:
        if len(rows) >= n_rows:
            break
        state = init_state.copy()
        # Step the match forward ball-by-ball; capture a DataFrame snapshot before each ball
        guard = 0
        while not state.is_match_over() and len(rows) < n_rows and guard < 400:
            guard += 1
            if state.is_innings_over():
                state.start_new_innings()
                if state.is_match_over():
                    break
            df_row = model.extract_features(state)   # today's path: DataFrame
            np_row = df_row[cols].iloc[0].to_numpy(dtype=np.float64)
            rows.append((df_row, np_row))
            outcome, runs = rules.simulate_ball(state, model)
            # simulate_ball already updates state
        print(f"  collected {len(rows)} rows after match {match_id}")
    if len(rows) < n_rows:
        raise RuntimeError(f"only collected {len(rows)} rows (wanted {n_rows})")
    return rows[:n_rows]


def _probs_df(model, df_row):
    return model.model.predict_proba(df_row)[0]


def _probs_np(model, np_row):
    return model.model.predict_proba(np_row.reshape(1, -1))[0]


def check_parity(model, rows):
    print(f"\n[parity] Comparing {len(rows)} rows; atol={PARITY_ATOL:g}")
    max_diff = 0.0
    mismatches = 0
    for i, (df_row, np_row) in enumerate(rows):
        p_df = _probs_df(model, df_row)
        p_np = _probs_np(model, np_row)
        diff = float(np.max(np.abs(p_df - p_np)))
        if diff > max_diff:
            max_diff = diff
        if not np.allclose(p_df, p_np, atol=PARITY_ATOL, rtol=0):
            mismatches += 1
            if mismatches <= 3:
                print(f"  MISMATCH row {i}: max |diff|={diff:.3e}")
                print(f"    df: {p_df}")
                print(f"    np: {p_np}")
    print(f"  max abs diff: {max_diff:.3e}")
    print(f"  mismatches: {mismatches}/{len(rows)}")
    return mismatches == 0


def check_edge_cases(model, rows):
    print("\n[edge cases]")
    cols = model.feature_columns
    df_row_0, np_row_0 = rows[0]
    ok = True

    # (1) Unseen player IDs -> -1 codes
    try:
        bat_i = cols.index("batter_encoded")
        bow_i = cols.index("bowler_encoded")
        patched = np_row_0.copy()
        patched[bat_i] = -1
        patched[bow_i] = -1
        df_patched = df_row_0.copy()
        df_patched.loc[df_patched.index[0], "batter_encoded"] = -1
        df_patched.loc[df_patched.index[0], "bowler_encoded"] = -1
        p_df = _probs_df(model, df_patched)
        p_np = _probs_np(model, patched)
        passed = np.allclose(p_df, p_np, atol=PARITY_ATOL, rtol=0)
        print(f"  unseen_player_ids: {'OK' if passed else 'FAIL'} "
              f"(max |diff|={float(np.max(np.abs(p_df - p_np))):.3e})")
        ok = ok and passed
    except ValueError:
        print("  skip unseen_player_ids (columns not present)")

    # (2) All-zero row
    zeros_np = np.zeros(len(cols), dtype=np.float64)
    zeros_df = pd.DataFrame([dict(zip(cols, zeros_np))])[cols]
    p_df = _probs_df(model, zeros_df)
    p_np = _probs_np(model, zeros_np)
    passed = np.allclose(p_df, p_np, atol=PARITY_ATOL, rtol=0)
    print(f"  all_zero: {'OK' if passed else 'FAIL'} "
          f"(max |diff|={float(np.max(np.abs(p_df - p_np))):.3e})")
    ok = ok and passed

    # (3) Extremes
    extreme = np_row_0.copy()
    for name, val in (
        ("striker_elo", 3000.0),
        ("batting_team_elo", 3000.0),
        ("run_rate", 30.0),
        ("run_rate_required", 30.0),
    ):
        if name in cols:
            extreme[cols.index(name)] = val
    extreme_df = pd.DataFrame([dict(zip(cols, extreme))])[cols]
    p_df = _probs_df(model, extreme_df)
    p_np = _probs_np(model, extreme)
    passed = np.allclose(p_df, p_np, atol=PARITY_ATOL, rtol=0)
    print(f"  extremes: {'OK' if passed else 'FAIL'} "
          f"(max |diff|={float(np.max(np.abs(p_df - p_np))):.3e})")
    ok = ok and passed

    # (4) Buffer reuse
    buf = np.zeros(len(cols), dtype=np.float64)
    buf[:] = np_row_0
    p1 = _probs_np(model, buf)
    buf.fill(0.0)
    buf[:] = np_row_0
    p2 = _probs_np(model, buf)
    buf_ok = np.allclose(p1, p2, atol=0, rtol=0)
    print(f"  buffer_reuse: {'OK' if buf_ok else 'FAIL'}")
    ok = ok and buf_ok

    return ok


def benchmark(model, rows):
    print(f"\n[benchmark] {N_BENCH_CALLS} calls each, after {N_WARMUP_CALLS} warmup")
    df_row, np_row = rows[0]

    for _ in range(N_WARMUP_CALLS):
        _probs_df(model, df_row)
        _probs_np(model, np_row)

    t0 = time.perf_counter()
    for _ in range(N_BENCH_CALLS):
        _probs_df(model, df_row)
    dt_df = (time.perf_counter() - t0) / N_BENCH_CALLS * 1000

    t0 = time.perf_counter()
    for _ in range(N_BENCH_CALLS):
        _probs_np(model, np_row)
    dt_np = (time.perf_counter() - t0) / N_BENCH_CALLS * 1000

    speedup = dt_df / dt_np if dt_np > 0 else float("inf")
    print(f"  DataFrame path: {dt_df:.3f} ms/call")
    print(f"  numpy     path: {dt_np:.3f} ms/call")
    print(f"  speedup:        {speedup:.2f}x (need ≥ {SPEEDUP_MIN}x)")
    return speedup >= SPEEDUP_MIN, dt_df, dt_np


def main():
    print(f"cwd: {ROOT}")

    print("\nBuilding XGBoostModelV2...")
    model = _build_model()
    n_features_in = getattr(model.model, "n_features_in_", None)
    print(f"  model.n_features_in_: {n_features_in}  "
          f"feature_columns: {len(model.feature_columns)}")
    if n_features_in is not None and n_features_in != len(model.feature_columns):
        print(f"  FAIL: feature count mismatch")
        sys.exit(1)

    print(f"\nCollecting {N_PARITY_STATES} feature rows from real match states...")
    rows = _collect_feature_rows(model, N_PARITY_STATES)

    parity_ok = check_parity(model, rows)
    edge_ok = check_edge_cases(model, rows)
    bench_ok, dt_df, dt_np = benchmark(model, rows)

    print("\n=== summary ===")
    print(f"  parity:     {'PASS' if parity_ok else 'FAIL'}")
    print(f"  edge cases: {'PASS' if edge_ok else 'FAIL'}")
    print(f"  speedup:    {'PASS' if bench_ok else 'FAIL'} "
          f"(df={dt_df:.3f} ms, np={dt_np:.3f} ms)")

    if parity_ok and edge_ok and bench_ok:
        print("ALL PASS — safe to apply Fix B to sim_v1_2.py")
        sys.exit(0)
    else:
        print("FAILURE — do NOT modify sim_v1_2.py yet")
        sys.exit(1)


if __name__ == "__main__":
    main()
