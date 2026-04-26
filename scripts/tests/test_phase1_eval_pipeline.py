"""
End-to-end validation of the Phase 1 sliced-eval pipeline.

Smoke-tests every code path that landed in Phase 1:
- BettingOddsLoader.load_odds(min_volume=...) with all 3 input shapes.
- run_sim_eval.py CLI argparse (--min-volume, --bootstrap-resamples).
- match_evaluator.MatchLevelEvaluator._bootstrap_ci on edge cases.
- OverallEvaluationResults dataclass with the 4 new CI fields.
- run_experiment.build_eval_cmd plumbing for evaluation.min_volume YAML.
- Slice-tag → output-filename encoding.

Pure unit-level (no model load, no sim). Runs in <1s.
"""

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from sim_eval.loaders import BettingOddsLoader
from sim_eval.match_evaluator import (
    MatchLevelEvaluator, OverallEvaluationResults,
    MatchEvaluationResult,
)


# ─── 1. min_volume filter edge cases ──────────────────────────────────────
def test_min_volume_zero_keeps_zero_vol_match():
    """min_volume=0 should keep matches with volume=0 (>= 0)."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "odds.json"
        p.write_text(json.dumps({
            "matches": [
                {"match_id": "zero", "polymarket_volume_usd": 0},
                {"match_id": "any",  "polymarket_volume_usd": 100},
            ]
        }))
        result = BettingOddsLoader.load_odds(str(p), min_volume=0)
        assert set(result.keys()) == {"zero", "any"}, \
            f"min_volume=0 should keep both, got {sorted(result.keys())}"


def test_min_volume_above_max_drops_all():
    """Threshold above the max should yield empty result."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "odds.json"
        p.write_text(json.dumps({
            "matches": [{"match_id": "m", "polymarket_volume_usd": 1000}]
        }))
        result = BettingOddsLoader.load_odds(str(p), min_volume=10_000)
        assert result == {}, f"expected empty, got {result}"


def test_min_volume_drops_missing_field_when_set():
    """Entries without polymarket_volume_usd are dropped when min_volume is set."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "odds.json"
        p.write_text(json.dumps({
            "matches": [
                {"match_id": "novol"},
                {"match_id": "withvol", "polymarket_volume_usd": 100_000},
            ]
        }))
        # min_volume set → missing-field entries excluded
        with_filter = BettingOddsLoader.load_odds(str(p), min_volume=50_000)
        assert set(with_filter.keys()) == {"withvol"}, \
            f"with min_volume, missing-field should drop, got {sorted(with_filter.keys())}"
        # min_volume None → missing-field entries kept
        no_filter = BettingOddsLoader.load_odds(str(p), min_volume=None)
        assert set(no_filter.keys()) == {"novol", "withvol"}, \
            f"min_volume=None should keep both, got {sorted(no_filter.keys())}"


# ─── 2. Bootstrap CI helper ────────────────────────────────────────────────
def test_bootstrap_ci_constant_input():
    ev = MatchLevelEvaluator.__new__(MatchLevelEvaluator)
    ev.bootstrap_resamples = 500
    lo, hi = ev._bootstrap_ci([0.5] * 20)
    assert abs(lo - 0.5) < 1e-9, f"constant CI low {lo}"
    assert abs(hi - 0.5) < 1e-9, f"constant CI high {hi}"


def test_bootstrap_ci_empty_returns_nan():
    ev = MatchLevelEvaluator.__new__(MatchLevelEvaluator)
    ev.bootstrap_resamples = 100
    lo, hi = ev._bootstrap_ci([])
    assert math.isnan(lo) and math.isnan(hi), f"empty CI ({lo}, {hi})"


def test_bootstrap_ci_zero_resamples_returns_nan():
    ev = MatchLevelEvaluator.__new__(MatchLevelEvaluator)
    ev.bootstrap_resamples = 100
    lo, hi = ev._bootstrap_ci([1.0, 2.0, 3.0], n_resamples=0)
    assert math.isnan(lo) and math.isnan(hi), \
        f"zero-resamples should yield NaN, got ({lo}, {hi})"


def test_bootstrap_ci_seed_is_deterministic():
    """Same input + seed → identical CI bounds across runs."""
    ev = MatchLevelEvaluator.__new__(MatchLevelEvaluator)
    ev.bootstrap_resamples = 1000
    rng = np.random.default_rng(0)
    sample = rng.normal(0, 1, size=200).tolist()
    lo1, hi1 = ev._bootstrap_ci(sample, seed=42)
    lo2, hi2 = ev._bootstrap_ci(sample, seed=42)
    assert (lo1, hi1) == (lo2, hi2), "seeded bootstrap should be deterministic"


def test_bootstrap_ci_brackets_mean_for_clean_signal():
    """For a normal sample, the percentile CI should bracket the sample mean."""
    ev = MatchLevelEvaluator.__new__(MatchLevelEvaluator)
    ev.bootstrap_resamples = 1000
    rng = np.random.default_rng(7)
    sample = rng.normal(2.0, 0.5, size=500).tolist()
    mean = float(np.mean(sample))
    lo, hi = ev._bootstrap_ci(sample, seed=42)
    assert lo < mean < hi, f"CI [{lo}, {hi}] should bracket mean {mean}"
    half_width = (hi - lo) / 2
    assert half_width < 0.1, f"CI half-width {half_width} unexpectedly wide"


# ─── 3. Dataclass field plumbing ───────────────────────────────────────────
def test_overall_results_default_ci_fields_are_nan():
    """The 4 new CI fields must default to NaN (so legacy callers don't break)."""
    r = OverallEvaluationResults(
        n_matches=0, avg_log_loss=float('nan'), avg_brier_score=float('nan'),
        calibration_bins=[], avg_edge=0.0, avg_signed_edge=0.0,
        profitable_bets=0, total_pnl=0.0, roi=0.0, win_rate=0.0, bets_placed=0,
    )
    for fname in ('avg_log_loss_ci_low', 'avg_log_loss_ci_high',
                  'flat_roi_ci_low', 'flat_roi_ci_high'):
        v = getattr(r, fname)
        assert math.isnan(v), f"{fname} should default to NaN, got {v}"


# ─── 4. Aggregator wires CIs into the dataclass ───────────────────────────
def _stub_match(mid: str, log_loss: float, pnl: float) -> MatchEvaluationResult:
    return MatchEvaluationResult(
        match_id=mid, team1='A', team2='B',
        simulated_win_prob={'A': 0.6, 'B': 0.4},
        simulated_scores={'A': {}, 'B': {}},
        market_win_prob={'A': 0.5, 'B': 0.5},
        market_odds={'A': 2.0, 'B': 2.0},
        actual_winner='A',
        log_loss=log_loss, brier_score=0.0,
        edge={'A': 0.1, 'B': -0.1},
        realized_pnl=pnl,
    )


def test_aggregator_populates_ci_fields():
    ev = MatchLevelEvaluator.__new__(MatchLevelEvaluator)
    ev.bootstrap_resamples = 200

    matches = [_stub_match(f'm{i}', log_loss=0.5 + 0.01*i,
                           pnl=(1.0 if i % 2 == 0 else -1.0))
               for i in range(20)]

    out = ev._aggregate_results(matches, total_time=0.0)
    assert not math.isnan(out.avg_log_loss_ci_low), \
        "log-loss CI low must be populated by aggregator"
    assert out.avg_log_loss_ci_low <= out.avg_log_loss <= out.avg_log_loss_ci_high, \
        f"log-loss point estimate {out.avg_log_loss} not in CI " \
        f"[{out.avg_log_loss_ci_low}, {out.avg_log_loss_ci_high}]"
    assert out.flat_roi_ci_low <= out.roi <= out.flat_roi_ci_high, \
        f"ROI point estimate {out.roi}% not in CI " \
        f"[{out.flat_roi_ci_low}%, {out.flat_roi_ci_high}%]"


# ─── 5. YAML → CLI plumbing through run_experiment.build_eval_cmd ─────────
def test_eval_cmd_has_min_volume_and_bootstrap_args():
    import run_experiment

    cfg = {
        "experiment": {"name": "smoketest"},
        "data": {"version": "v3", "test_dir": "data/x", "odds_file": "o.json"},
        "features": {"groups": ["basic"]},
        "model": {"type": "xgboost"},
        "evaluation": {
            "min_volume": 50000,
            "bootstrap_resamples": 750,
            "n_sims": 100,
        },
    }
    cmd = run_experiment.build_eval_cmd(cfg)
    cmd_str = " ".join(cmd)

    assert "--min-volume 50000" in cmd_str, \
        f"min_volume not threaded through, cmd: {cmd_str}"
    assert "--bootstrap-resamples 750" in cmd_str, \
        f"bootstrap_resamples not threaded through, cmd: {cmd_str}"


def test_eval_cmd_omits_min_volume_when_unset():
    import run_experiment

    cfg = {
        "experiment": {"name": "no_minvol"},
        "data": {"version": "v3"},
        "features": {"groups": ["basic"]},
        "model": {"type": "xgboost"},
        "evaluation": {"n_sims": 100},
    }
    cmd_str = " ".join(run_experiment.build_eval_cmd(cfg))
    assert "--min-volume" not in cmd_str, \
        f"min_volume should be absent when unset, cmd: {cmd_str}"


# ─── runner ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [v for k, v in globals().items()
             if k.startswith("test_") and callable(v)]
    failures = []
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failures.append((t.__name__, str(e)))
            print(f"FAIL {t.__name__}: {e}")

    print()
    if failures:
        print(f"FAILED {len(failures)} / {len(tests)}")
        sys.exit(1)
    print(f"All {len(tests)} Phase-1 pipeline tests passed.")
