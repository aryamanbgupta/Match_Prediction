"""Warm-chunk benchmark: StatsProviderCache wrapper vs raw StatsProvider.

The first simulation of a match is always chunk-load-bound (disk → RAM
pickle expansion). That cost must NOT be attributed to either path, or
the benchmark measures chunk I/O instead of the cache win we care about.

Protocol:
  1. Load model + provider.
  2. Warmup: one full simulate_multiple to populate LRU chunks.
  3. Alternate raw / wrapped timings for `TRIALS` runs each; report median.
"""
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))

from stats_provider import StatsProvider, StatsProviderCache
from sim_v1_2 import SimulationEngine, SimulationConfig, XGBoostModelV2
from sim_eval.loaders import TestMatchLoader

N_SIMS = 100
TRIALS = 3


def _set_raw(model, raw):
    model.stats_provider = raw


def _set_wrapped(model, raw):
    wrapped = StatsProviderCache(raw)
    model.stats_provider = wrapped


def main():
    matches_dir = ROOT / 'data' / 'betting_test'
    model_dir = ROOT / 'models' / 'xgb_v3'

    provider = StatsProvider(str(ROOT / 'models'), version='v3')
    model = XGBoostModelV2(
        str(model_dir / 'xgboost_model_v3.pkl'),
        str(model_dir / 'batter_encoder_v3.pkl'),
        str(model_dir / 'bowler_encoder_v3.pkl'),
        str(model_dir / 'feature_columns_v3.txt'),
        stats_provider=provider,
        matchup_encoder_path=str(model_dir / 'matchup_encoder_v3.pkl'),
    )
    # The model's wrap_with_cache(provider) wraps it; grab the raw underneath.
    raw_provider = model.stats_provider._provider \
        if isinstance(model.stats_provider, StatsProviderCache) \
        else model.stats_provider

    loader = TestMatchLoader()
    matches = loader.load_matches(str(matches_dir))
    _, state = matches[0]
    print(f"benchmarking on match: {matches[0][0]}", flush=True)

    engine = SimulationEngine(model)
    config = SimulationConfig(n_simulations=N_SIMS, parallel=False, random_seed=42)

    print(f"warmup ({N_SIMS} sims) to populate chunk LRU...", flush=True)
    _set_raw(model, raw_provider)
    t0 = time.time()
    _ = engine.simulate_multiple(state, config)
    print(f"  warmup wall time: {time.time()-t0:.2f}s", flush=True)

    print(f"\nmeasuring {TRIALS} trials of {N_SIMS} sims each (alternating)...",
          flush=True)
    raw_times, wrapped_times = [], []
    for t in range(TRIALS):
        _set_raw(model, raw_provider)
        t0 = time.time()
        engine.simulate_multiple(state, config)
        dt = time.time() - t0
        raw_times.append(dt)
        print(f"  trial {t+1} raw:     {dt:.3f}s", flush=True)

        _set_wrapped(model, raw_provider)
        t0 = time.time()
        engine.simulate_multiple(state, config)
        dt = time.time() - t0
        wrapped_times.append(dt)
        print(f"  trial {t+1} wrapped: {dt:.3f}s", flush=True)

    raw_med = statistics.median(raw_times)
    wrap_med = statistics.median(wrapped_times)
    speedup = raw_med / wrap_med
    saved_pct = (1 - wrap_med / raw_med) * 100

    print("\n=== results (median of 3 trials, 100 sims / trial) ===")
    print(f"  raw StatsProvider:       {raw_med:.3f}s")
    print(f"  StatsProviderCache:      {wrap_med:.3f}s")
    print(f"  speedup:                 {speedup:.2f}x  (saved {saved_pct:.1f}%)")


if __name__ == '__main__':
    main()
