"""End-to-end bit-exactness check for the Phase-0 StatsProviderCache wrapper.

Runs one match through `SimulationEngine.simulate_multiple` twice against the
same seeded RNG: once with the cache wrapper in place (current default) and
once with the cache wrapper peeled off so the underlying StatsProvider is
called directly. Any divergence in per-sim winners or innings scores means
the wrapper has introduced a correctness regression; bit-equal results prove
the memoization is a pure-speedup change.

Run standalone:
    uv run python scripts/tests/check_team_memo_e2e.py

Not in the pytest default collection — it loads the real XGBoost model
and the v3 stats cache (~5 sec + a few GB RSS).
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))

from stats_provider import StatsProvider, StatsProviderCache
from sim_v1_2 import SimulationEngine, SimulationConfig, XGBoostModelV2
from sim_eval.loaders import TestMatchLoader


def _summarize(results):
    """Compact, order-preserving summary of per-sim outcomes for equality."""
    return [
        (r.winner, r.team1_score, r.team1_wickets, r.team2_score, r.team2_wickets)
        for r in results
    ]


def main():
    matches_dir = ROOT / 'data' / 'betting_test'
    model_dir = ROOT / 'models' / 'xgb_v3'

    print("loading stats provider + model...", flush=True)
    t0 = time.time()
    provider = StatsProvider(str(ROOT / 'models'), version='v3')
    model = XGBoostModelV2(
        str(model_dir / 'xgboost_model_v3.pkl'),
        str(model_dir / 'batter_encoder_v3.pkl'),
        str(model_dir / 'bowler_encoder_v3.pkl'),
        str(model_dir / 'feature_columns_v3.txt'),
        stats_provider=provider,
        matchup_encoder_path=str(model_dir / 'matchup_encoder_v3.pkl'),
    )
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    assert isinstance(model.stats_provider, StatsProviderCache), \
        "XGBoostModelV2 should have wrapped stats_provider by default"

    loader = TestMatchLoader()
    matches = loader.load_matches(str(matches_dir))
    assert matches, f"no test matches found in {matches_dir}"
    match_id, state = matches[0]
    print(f"picked match: {match_id}", flush=True)

    engine = SimulationEngine(model)
    config = SimulationConfig(n_simulations=20, parallel=False, random_seed=42)

    print("run A: with StatsProviderCache wrapper...", flush=True)
    t0 = time.time()
    results_a = engine.simulate_multiple(state, config)
    dt_a = time.time() - t0
    summary_a = _summarize(results_a)

    # Peel the wrapper off and rerun with the raw provider.
    model.stats_provider = model.stats_provider._provider
    assert not isinstance(model.stats_provider, StatsProviderCache)

    print("run B: with raw StatsProvider (cache bypassed)...", flush=True)
    t0 = time.time()
    results_b = engine.simulate_multiple(state, config)
    dt_b = time.time() - t0
    summary_b = _summarize(results_b)

    if summary_a == summary_b:
        print(f"PASS — {len(summary_a)} sims bit-identical "
              f"(wrapped {dt_a:.2f}s vs raw {dt_b:.2f}s; "
              f"speedup {dt_b/dt_a:.2f}x)")
        return 0

    print("FAIL — per-sim outputs diverge. First 3 mismatches:")
    shown = 0
    for i, (a, b) in enumerate(zip(summary_a, summary_b)):
        if a != b:
            print(f"  sim[{i}]: wrapped={a}  raw={b}")
            shown += 1
            if shown >= 3:
                break
    return 1


if __name__ == '__main__':
    sys.exit(main())
