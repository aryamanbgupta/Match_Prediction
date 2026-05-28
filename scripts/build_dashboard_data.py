"""Builds docs/dashboard_data.json — the single source of truth consumed by
docs/model_dashboard_terminal.html, docs/model_dashboard_trader.html, and
docs/predictions_tracker.html.

Most match-level v3 numbers are curated from the M1–M8 phase reports
under reports/m{1..8}_*.md (those reports ARE the canonical source — they
report bootstrap CIs that match_evaluator emits but in pre-formatted form).
Sizing-sweep numbers are read from the M7 production CSVs verbatim.
Train-time metrics + feature importances are read from train_metrics.json.
Fixture inputs are read from fixtures/*.json.

Run:    uv run python scripts/build_dashboard_data.py
Output: docs/dashboard_data.json
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / 'models'
REPORTS = REPO / 'reports'
FIXTURES = REPO / 'fixtures'
OUT = REPO / 'docs' / 'dashboard_data.json'


# ---------------------------------------------------------------------------
# Reference baselines (reported in CLAUDE.md / m*_eval.md)
# ---------------------------------------------------------------------------
BENCHMARKS = {
    'market_ll_50k': 0.6267,   # polymarket close-line iter ≥$50k
    'coinflip_ll': 0.6931,
    'always_favorite_roi_pct': 4.15,
    'iter_50k_n': 170,
    'iter_100k_n': 110,
    'golden_50k_n': 50,
    'golden_100k_n': 45,
}


# ---------------------------------------------------------------------------
# Match-level direct: full lineage. Numbers transcribed from the m*_eval.md
# reports + leakage_fix_comparison.md + no_leakage_diagnostic_clean.md.
# Each model entry carries: identity, parent, what changed vs parent, eval
# metrics on iter ≥$50k / ≥$100k / golden / adversarial slices, and gates.
# ---------------------------------------------------------------------------
MATCH_LEVEL_MODELS = [
    {
        'id': 'xgb_match_v2_clean',
        'label': 'v2 clean (post-leakage-fix)',
        'phase': 'pre-v3',
        'date': '2026-05-09',
        'status': 'ARCHIVED',
        'parent': 'xgb_match_v2_frozen',
        'feature_count': 47,
        'mode': 'frozen',
        'what_changed': 'Snapshotted temp_elo before parse to fix ELO leakage. Same A2 features as v2_frozen (top-6 batting / bottom-5 bowling ELO splits, TeamFormTracker, H2H, HomeVenue).',
        'why': 'First clean (no-leakage) match-level direct model. Replaced inflated v2_frozen numbers (ROI +50% retracted as leakage-driven).',
        'metrics': {
            'val_ll': 0.6424, 'test_ll': 0.6021,
            'iter_50k_raw':   {'ll': 0.6339, 'roi': 22.63, 'roi_ci': [2.42, 45.04], 'win_pct': 52.4, 'n': 168},
            'iter_50k_platt': None,
            'iter_100k_raw':  None,
            'golden_50k_raw': {'ll': 0.6747, 'roi': 32.61, 'roi_ci': [-0.20, 63.6], 'win_pct': 59.2, 'n': 50},
            'golden_100k_raw':{'ll': 0.6698, 'roi': 34.75, 'roi_ci': [3.79, 65.5],  'win_pct': 61.4, 'n': 45},
        },
        'gates': {'iter_50k_ll_beats_market': False, 'iter_50k_roi_ci_pos': True,
                  'iter_100k_roi_ci_pos': None, 'golden_50k_roi_ci_pos': False, 'golden_100k_roi_ci_pos': True},
    },
    {
        'id': 'xgb_match_v3_baseline',
        'label': 'M1 baseline (monotone)',
        'phase': 'M1',
        'date': '2026-05-10',
        'status': 'REF',
        'parent': 'xgb_match_v2_clean',
        'feature_count': 45,
        'mode': 'frozen',
        'what_changed': 'Added monotone constraints on 10 directional features (top6_batting_elo_diff, bottom5_bowling_elo_diff, elo_diff_*, win_rate_diff, batting_avg_diff, bowling_econ_diff, h2h, is_team{1,2}_home). Eval infra: stratified bootstrap, adversarial slices (close/mismatch/IPL/intl), walk-forward, Platt calibration as sizing layer.',
        'why': 'Generalization guard, not a one-shot LL win. M1+Platt was the FIRST variant to clear iter ≥$50k LL gate (0.6235 < market). Golden ~0.025 LL worse than v2_clean — within bootstrap noise on n=50, not promoted.',
        'metrics': {
            'val_ll': 0.6497, 'test_ll': 0.6146,
            'iter_50k_raw':   {'ll': 0.6302, 'roi': 17.40, 'roi_ci': [-2.85, 41.14], 'win_pct': 50.6, 'n': 168},
            'iter_50k_platt': {'ll': 0.6235, 'roi': 23.00, 'roi_ci': [1.94, 44.16],  'win_pct': 54.2, 'n': 168},
            'iter_100k_raw':  {'ll': 0.5931, 'roi': 20.34, 'roi_ci': [-5.70, 49.97], 'win_pct': None, 'n': 110},
            'iter_100k_platt':{'ll': 0.5827, 'roi': 25.81, 'roi_ci': [-1.22, 55.82], 'win_pct': None, 'n': 110},
            'golden_50k_raw': {'ll': 0.7006, 'roi': 25.37, 'roi_ci': [-7.89, 56.78], 'win_pct': 55.1, 'n': 49},
            'golden_50k_platt':{'ll':0.6926, 'roi': 24.28, 'roi_ci': [-8.86, 55.02], 'win_pct': 55.1, 'n': 49},
            'close':    {'ll': 0.7186, 'roi': -5.73,  'roi_ci': [-34.96, 29.15],  'win_pct': 37.9, 'n': 60},
            'mismatch': {'ll': 0.4737, 'roi': 60.82,  'roi_ci': [3.59, 134.06],   'win_pct': 66.7, 'n': 33},
            'ipl':      {'ll': 0.6758, 'roi': 16.40,  'roi_ci': [-25.53, 61.52],  'win_pct': 54.5, 'n': 22},
            'international':{'ll':0.5661,'roi':36.28, 'roi_ci': [1.59, 76.92],    'win_pct': 54.7, 'n': 75},
        },
        'walk_forward': [
            {'month': '2025-10', 'n': 1,  'll': 0.6825, 'roi': -100.0, 'win_pct': 0.0},
            {'month': '2025-11', 'n': 5,  'll': 0.7843, 'roi': 57.26, 'win_pct': 40.0},
            {'month': '2025-12', 'n': 27, 'll': 0.6625, 'roi': -3.34, 'win_pct': 44.4},
            {'month': '2026-01', 'n': 45, 'll': 0.6722, 'roi': -5.18, 'win_pct': 41.9},
            {'month': '2026-02', 'n': 49, 'll': 0.5322, 'roi': 52.63, 'win_pct': 63.3},
            {'month': '2026-03', 'n': 8,  'll': 0.5839, 'roi': -13.99,'win_pct': 37.5},
            {'month': '2026-04', 'n': 35, 'll': 0.6778, 'roi': 16.68, 'win_pct': 54.3},
        ],
        'gates': {'iter_50k_ll_beats_market': True, 'iter_50k_roi_ci_pos': True,
                  'iter_100k_roi_ci_pos': False, 'golden_50k_roi_ci_pos': False},
    },
    {
        'id': 'xgb_match_v3_m2_venue_only',
        'label': 'M2 venue-only (frozen)',
        'phase': 'M2',
        'date': '2026-05-10',
        'status': 'LANDED',
        'parent': 'xgb_match_v3_baseline',
        'feature_count': 48,
        'mode': 'frozen',
        'what_changed': '+3 venue outcome-dist features (venue_p4, venue_p6, venue_pw — empirical-Bayes shrunk to corpus prior π, k=200). Drop-one ablation killed the batter (9 feats, neutral) and bowler (9 feats, hurts) outcome-dist groups.',
        'why': 'First variant with positive iter ≥$50k ROI CI lower bound (+2.73). Strict LL gate not cleared (Δ +0.005 vs M1 raw). Bowler features killed because bottom-5 squad-order is not a clean bowling unit.',
        'metrics': {
            'val_ll': 0.6521, 'test_ll': 0.6227,
            'iter_50k_raw':   {'ll': 0.6347, 'roi': 22.77, 'roi_ci': [2.73, 43.46],  'win_pct': 53.0, 'n': 170},
            'iter_50k_platt': {'ll': 0.6279, 'roi': 20.33, 'roi_ci': [-0.23, 41.27], 'win_pct': 52.4, 'n': 170},
            'iter_100k_raw':  {'ll': 0.5836, 'roi': 24.61, 'roi_ci': [-2.37, 55.80], 'win_pct': 54.5, 'n': 110},
            'golden_50k_raw': {'ll': 0.6885, 'roi': 20.85, 'roi_ci': [-12.31, 52.54],'win_pct': 53.1, 'n': 49},
            'golden_50k_platt':{'ll':0.6775, 'roi': 23.18, 'roi_ci': [-10.49, 54.24],'win_pct': 55.1, 'n': 49},
            'close':    {'ll': 0.7177, 'roi': 1.74,  'roi_ci': [-29.21, 35.52],  'win_pct': 41.4, 'n': 60},
            'mismatch': {'ll': 0.4868, 'roi': 60.82, 'roi_ci': [3.59, 134.06],   'win_pct': 66.7, 'n': 33},
        },
        'gates': {'iter_50k_ll_beats_market': False, 'iter_50k_roi_ci_pos': True,
                  'iter_100k_roi_ci_pos': False, 'golden_50k_roi_ci_pos': False},
    },
    {
        'id': 'xgb_match_v3_m2_venue_only_unfrozen',
        'label': 'M2 v.o. UNFROZEN',
        'phase': 'M3',
        'date': '2026-05-10',
        'status': 'LANDED',
        'parent': 'xgb_match_v3_m2_venue_only',
        'feature_count': 48,
        'mode': 'unfrozen',
        'what_changed': 'Re-materialized parquet WITHOUT --freeze-trackers-after, so each match sees the chronological tracker state through pre-match date. Same 48 features as M2 v.o. M3 rolling-form features themselves were DROPPED (career aggs already capture variance), but the materialization-mode finding is the M3 take-away.',
        'why': 'First variant clearing BOTH iter ≥$50k gates simultaneously (LL 0.6279 < market, ROI CI lower +6.26). Also clears ≥$100k. Golden ROI CI cleanly excludes 0 for the first time (+1.92). Trade-off: golden LL +0.004 worse vs frozen.',
        'metrics': {
            'val_ll': 0.6521, 'test_ll': 0.6015,
            'iter_50k_raw':   {'ll': 0.6348, 'roi': 25.40, 'roi_ci': [4.75, 48.11],  'win_pct': 53.6, 'n': 170},
            'iter_50k_platt': {'ll': 0.6279, 'roi': 26.69, 'roi_ci': [6.26, 48.65],  'win_pct': 55.4, 'n': 170},
            'iter_100k_raw':  {'ll': 0.6006, 'roi': 26.21, 'roi_ci': [-0.93, 57.10], 'win_pct': 53.6, 'n': 110},
            'iter_100k_platt':{'ll': 0.5908, 'roi': 26.61, 'roi_ci': [0.63, 57.42],  'win_pct': 55.5, 'n': 110},
            'golden_50k_raw': {'ll': 0.6925, 'roi': 14.39, 'roi_ci': [-16.98, 44.38],'win_pct': 51.0, 'n': 49},
            'golden_50k_platt':{'ll':0.6849, 'roi': 31.29, 'roi_ci': [1.92, 59.11],  'win_pct': 61.2, 'n': 49},
            'golden_100k_raw':{'ll': 0.6944, 'roi': 14.46, 'roi_ci': [-17.08, 47.16],'win_pct': 52.3, 'n': 44},
            'golden_100k_platt':{'ll':0.6850,'roi': 33.28, 'roi_ci': [0.08, 64.37],  'win_pct': 63.6, 'n': 44},
            'close':    {'ll': 0.6982, 'roi': 26.12, 'roi_ci': [-2.34, 52.30],   'win_pct': 52.7, 'n': 74},
            'mismatch': {'ll': 0.3731, 'roi': 22.54, 'roi_ci': [-13.48, 58.37],  'win_pct': 62.5, 'n': 24},
        },
        'walk_forward': [
            {'month': '2025-12', 'n': 27, 'll': 0.7187, 'roi': -13.23, 'win_pct': None},
            {'month': '2026-01', 'n': 45, 'll': 0.6690, 'roi': 11.36, 'win_pct': None},
            {'month': '2026-02', 'n': 49, 'll': 0.5218, 'roi': 52.63, 'win_pct': None},
            {'month': '2026-04', 'n': 35, 'll': 0.6583, 'roi': 43.67, 'win_pct': 68.6, 'roi_ci': [10.36, 76.65]},
        ],
        'gates': {'iter_50k_ll_beats_market': True, 'iter_50k_roi_ci_pos': True,
                  'iter_100k_roi_ci_pos': True, 'golden_50k_roi_ci_pos': True, 'golden_100k_roi_ci_pos': True},
    },
    {
        'id': 'xgb_match_v3_m3_full',
        'label': 'M3 rolling form',
        'phase': 'M3',
        'date': '2026-05-10',
        'status': 'DROPPED',
        'parent': 'xgb_match_v3_m2_venue_only_unfrozen',
        'feature_count': 67,
        'mode': 'unfrozen',
        'what_changed': '+18 player-level rolling-form features: top-6 batting avg/SR_recent + diffs, all-11 bowlers avg/econ_recent + diffs, in-form/out-of-form indicators + diffs.',
        'why': 'Dropped — career-aggregate features already capture predictive variance. Bowling-recent actively hurts (same bottom-5 squad-order issue). LL/ROI both worse than baseline. The diagnosis surfaced UNFROZEN materialization as the real win.',
        'metrics': {
            'iter_50k_raw': {'ll': 0.6517, 'roi': 16.81, 'roi_ci': [-3.14, 39.19], 'n': 170},
            'golden_50k_raw': {'ll': 0.6946, 'roi': 10.22, 'n': 49},
        },
        'gates': {'dropped': True, 'reason': 'redundancy with M1 career aggregates'},
    },
    {
        'id': 'xgb_match_v3_m4_unfrozen',
        'label': 'M4 within-tournament',
        'phase': 'M4',
        'date': '2026-05-10',
        'status': 'DROPPED',
        'parent': 'xgb_match_v3_m2_venue_only_unfrozen',
        'feature_count': 63,
        'mode': 'unfrozen',
        'what_changed': '+15 features: date-windowed form (60d), competition-filtered form (365d, same tier), scheduling proxies (days_since, back_to_back).',
        'why': 'Dropped — form features 0.7-0.8 correlated with M1 win_rate_diff (redundant signal). Scheduling features have ~zero target r but XGBoost picks them up via interactions, fitting noise. Result: ~10% more confident predictions, LESS accurate at the tail (89.6% → 86.0% acc at |p−0.5|>0.20).',
        'metrics': {
            'val_ll': 0.6463, 'test_ll': 0.5947,
            'iter_50k_raw':   {'ll': 0.6319, 'roi': 20.19, 'roi_ci': [0.84, 42.22],  'n': 170},
            'iter_50k_platt': {'ll': 0.6272, 'roi': 12.84, 'roi_ci': [-6.38, 34.95], 'n': 170},
        },
        'gates': {'dropped': True, 'reason': 'over-confidence from correlated features'},
    },
    {
        'id': 'xgb_match_v3_m5',
        'label': 'M5 player × opp',
        'phase': 'M5',
        'date': '2026-05-10',
        'status': 'DROPPED_AT_CORR',
        'parent': 'xgb_match_v3_m2_venue_only_unfrozen',
        'feature_count': 56,
        'mode': 'unfrozen',
        'what_changed': '+8 player × opposition features (h2h-aggregated avg/SR/balls per top-6 batter, shrunk to career).',
        'why': 'DROPPED at pre-training correlation check. All 8 features failed: 7 have |r|>0.5 with an M1 baseline feature AND target r ≤ baseline\'s; the 1 borderline is target r essentially identical. Lineup aggregation collapses per-player matchup signal to team career means.',
        'metrics': {},
        'gates': {'dropped': True, 'reason': 'redundancy: lineup aggregation collapses signal'},
    },
    {
        'id': 'xgb_match_v3_m6_unfrozen',
        'label': 'M6 conditions',
        'phase': 'M6',
        'date': '2026-05-10',
        'status': 'DROPPED',
        'parent': 'xgb_match_v3_m2_venue_only_unfrozen',
        'feature_count': 51,
        'mode': 'unfrozen',
        'what_changed': '+3 date-derived condition features: month_of_year, day_of_week, is_dew_prone_month.',
        'why': 'Dropped — passes redundancy check but FAILS new target-correlation-floor check (|target r| ≥ 0.03). All 3 features have |target r| ≤ 0.011. Same M4 over-confidence pattern: 50+ more high-confidence preds, each less accurate. Discipline upgrade → dual-condition correlation check.',
        'metrics': {
            'val_ll': 0.6498, 'test_ll': 0.5994,
            'iter_50k_raw':   {'ll': 0.6412, 'roi': 23.53, 'roi_ci': [2.28, 46.45],  'n': 170},
            'iter_50k_platt': {'ll': 0.6341, 'roi': 24.47, 'roi_ci': [3.81, 46.55],  'n': 170},
        },
        'gates': {'dropped': True, 'reason': 'target-correlation floor failed'},
    },
    {
        'id': 'xgb_match_v3_m7_production',
        'label': 'M7 production (current)',
        'phase': 'M7',
        'date': '2026-05-10',
        'status': 'PROD',
        'parent': 'xgb_match_v3_m2_venue_only_unfrozen',
        'feature_count': 49,
        'mode': 'unfrozen',
        'what_changed': 'Same 49 features as M2 v.o. unfrozen. ONLY hyperparameters changed: lr 0.10→0.05, colsample 0.8→0.9. From 81-config grid (max_depth × lr × subsample × colsample); winner picked by val LL, then validated on iter ≥$50k. Old config was over-aggressive.',
        'why': 'Closed M2 v.o.\'s residual fit-gap. Iter ≥$100k ROI CI cleanly excludes 0 for the first time (+0.57). Close-match slice ROI CI also clears (+4.36) — historically weakest slice. 2026-04 IPL: ROI +34.87% [+2.04, +68.06], win 65.7%. Production uses RAW probabilities (Platt over-corrects on this config and kills iteration ROI).',
        'metrics': {
            'val_ll': 0.6459, 'test_ll': 0.5924,
            'iter_50k_raw':   {'ll': 0.6299, 'roi': 21.90, 'roi_ci': [2.28, 43.83],  'win_pct': None, 'n': 170},
            'iter_50k_platt': {'ll': 0.6223, 'roi': 13.54, 'roi_ci': [-7.29, 34.94], 'n': 170},
            'iter_100k_raw':  {'ll': 0.5929, 'roi': 26.39, 'roi_ci': [0.57, 58.78],  'n': 110},
            'close':    {'ll': 0.6880, 'roi': 33.27, 'roi_ci': [4.36, 61.53],   'win_pct': 56.8, 'n': 74},
            'mismatch': {'ll': 0.3565, 'roi': None, 'n': 24},
            'ipl':      {'ll': 0.6709, 'roi': None, 'n': 22},
        },
        'walk_forward': [
            {'month': '2025-10', 'n': 1,  'll': 0.6468, 'roi': -100.0, 'win_pct': 0.0},
            {'month': '2025-11', 'n': 5,  'll': 0.8872, 'roi': 57.26, 'win_pct': 40.0,  'roi_ci': [-100, 228.94]},
            {'month': '2025-12', 'n': 27, 'll': 0.7080, 'roi': -34.70,'win_pct': 25.9,  'roi_ci': [-74.91, 9.58]},
            {'month': '2026-01', 'n': 45, 'll': 0.6676, 'roi': 13.14, 'win_pct': 48.8,  'roi_ci': [-24.13, 49.19]},
            {'month': '2026-02', 'n': 49, 'll': 0.5159, 'roi': 58.20, 'win_pct': 63.3,  'roi_ci': [9.88, 118.31]},
            {'month': '2026-03', 'n': 8,  'll': 0.5874, 'roi': -25.95,'win_pct': 37.5,  'roi_ci': [-78.63, 45.23]},
            {'month': '2026-04', 'n': 35, 'll': 0.6554, 'roi': 34.87, 'win_pct': 65.7,  'roi_ci': [2.04, 68.06]},
        ],
        'gates': {'iter_50k_ll_beats_market': True, 'iter_50k_roi_ci_pos': True,
                  'iter_100k_roi_ci_pos': True, 'close_roi_ci_pos': True},
    },
]


# ---------------------------------------------------------------------------
# Ball-level sim: shorter lineage, only headline numbers since these models
# lost the winner-market race and v7 is now reserved for prop bets.
# ---------------------------------------------------------------------------
BALL_LEVEL_MODELS = [
    {
        'id': 'xgb_v4_team_strength',
        'label': 'v4 team-strength + ELO',
        'phase': 'v4',
        'date': '2026-03-21',
        'status': 'ARCHIVED',
        'parent': 'xgb_v3',
        'what_changed': '+ELO + aggregated team stats (9 features). 19% log-loss improvement on ball-level.',
        'metrics': {
            'iter_50k_raw': {'ll': 0.7838, 'roi': 12.10, 'roi_ci': None, 'n': 170},
        },
    },
    {
        'id': 'xgb_v6_outcome_dist',
        'label': 'v6 outcome-dist',
        'phase': 'v6',
        'date': '2026-04-23',
        'status': 'ARCHIVED',
        'parent': 'xgb_v5b_venue_pruned',
        'what_changed': 'Empirical-Bayes outcome-dist features (schema v4, +42 cols). Five hierarchies: batter, bowler, batter-vs-pace/spin, bowler-vs-LHB/RHB, venue.',
        'why': 'log loss 0.7518→0.7122 (-5.3%) but flat ROI +6.51% → -7.1% (calibration win, ROI loss).',
        'metrics': {
            'iter_50k_raw': {'ll': 0.7370, 'roi': -9.81, 'n': 170},
        },
    },
    {
        'id': 'xgb_v7_hierarchical_shrink',
        'label': 'v7 (props production)',
        'phase': 'v7',
        'date': '2026-04-25',
        'status': 'PROD_PROPS',
        'parent': 'xgb_v6_outcome_dist',
        'feature_count': 114,
        'what_changed': 'Hierarchical shrinkage: vs-type/vs-hand cells shrink toward player overall (not π). k_player=30 (Phase 6 sweep), k_venue=200.',
        'why': 'Lost winner-market race to clean direct (LL ~0.7402 vs 0.6747 on golden ≥$50k). REPURPOSED for prop bets via EmpiricalBowlerSelector — now scoring +0.091 Brier skill on batter_fours_2plus, +0.075 on batter_fours_1plus.',
        'metrics': {
            'iter_50k_raw': {'ll': 0.7402, 'roi': 6.11, 'roi_ci': [-10.7, 23.9], 'n': 170},
        },
        'prop_skill': [
            {'family': 'batter_fours_2plus', 'n': 4254, 'brier': 0.2095, 'base': 0.2306, 'skill': 0.091, 'verdict': 'ship'},
            {'family': 'batter_fours_1plus', 'n': 4254, 'brier': 0.2250, 'base': 0.2433, 'skill': 0.075, 'verdict': 'ship'},
            {'family': 'batter_fours_3plus', 'n': 4254, 'brier': 0.1642, 'base': 0.1744, 'skill': 0.058, 'verdict': 'ship'},
            {'family': 'batter_6plus_six',   'n': 4254, 'brier': 0.2288, 'base': 0.2397, 'skill': 0.046, 'verdict': 'ship'},
            {'family': 'top_batter',         'n': 5835, 'brier': 0.0775, 'base': 0.0810, 'skill': 0.043, 'verdict': 'ship'},
            {'family': 'innings_runs_ou_160_5','n': 522,'brier': 0.2416, 'base': 0.2496, 'skill': 0.032, 'verdict': 'ship'},
            {'family': 'top_bowler',         'n': 5835, 'brier': 0.0793, 'base': 0.0806, 'skill': 0.016, 'verdict': 'modest'},
            {'family': 'pp_total_ou_55_5',   'n': 522,  'brier': 0.2460, 'base': 0.2080, 'skill': -0.183, 'verdict': 'fade'},
            {'family': 'bowler_wkts_2plus',  'n': 3106, 'brier': 0.2365, 'base': 0.2037, 'skill': -0.161, 'verdict': 'fade'},
            {'family': 'bowler_wkts_3plus',  'n': 3106, 'brier': 0.1056, 'base': 0.0922, 'skill': -0.146, 'verdict': 'fade'},
            {'family': 'pp_total_ou_50_5',   'n': 522,  'brier': 0.2812, 'base': 0.2455, 'skill': -0.145, 'verdict': 'fade'},
        ],
    },
]


# ---------------------------------------------------------------------------
# Sizing sweep — read M7 production CSVs verbatim
# ---------------------------------------------------------------------------
def load_sizing_sweep(slice_name: str) -> list[dict]:
    path = MODELS / 'xgb_match_v3_m7_production' / f'm8_sizing_sweep_{slice_name}.csv'
    rows: list[dict] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                'threshold': float(r['threshold']),
                'sizing': r['sizing'],
                'kelly_mult': float(r['kelly_mult']) if r['kelly_mult'] else None,
                'cap': float(r['cap']) if r['cap'] else None,
                'n_eligible': int(r['n_eligible']),
                'n_bets': int(r['n_bets']),
                'pnl': float(r['total_pnl']),
                'roi': float(r['roi_pct']),
                'roi_ci': [float(r['roi_ci_lo']), float(r['roi_ci_hi'])],
                'win_rate': float(r['win_rate']),
                'max_dd': float(r['max_drawdown']),
            })
    return rows


# ---------------------------------------------------------------------------
# Feature importance — production model
# ---------------------------------------------------------------------------
def load_feature_importance() -> list[dict]:
    train_metrics = json.loads((MODELS / 'xgb_match_v3_m7_production' / 'train_metrics.json').read_text())
    fi = train_metrics['feature_importances']
    sorted_fi = sorted(fi.items(), key=lambda kv: -kv[1])
    return [{'feature': k, 'importance': v} for k, v in sorted_fi]


# ---------------------------------------------------------------------------
# Hyperparameter sweep summary (top 6 from M7.A grid)
# ---------------------------------------------------------------------------
HPARAM_SWEEP_TOP = [
    {'config': 'baseline (md=4 lr=0.10 ss=0.8 cs=0.8)', 'val_ll': 0.6521, 'test_ll': 0.6015, 'iter_50k_ll': 0.6348, 'iter_50k_roi': 25.40, 'iter_50k_roi_ci': [4.75, 48.11], 'iter_100k_roi': 26.21, 'iter_100k_roi_ci': [-0.93, 57.10]},
    {'config': 'best_val (md=4 lr=0.05 ss=0.8 cs=0.9) ← LANDED', 'val_ll': 0.6459, 'test_ll': 0.5924, 'iter_50k_ll': 0.6299, 'iter_50k_roi': 21.90, 'iter_50k_roi_ci': [2.28, 43.83], 'iter_100k_roi': 26.39, 'iter_100k_roi_ci': [0.57, 58.78]},
    {'config': 'best_test (md=4 lr=0.05 ss=0.7 cs=0.7)', 'val_ll': 0.6530, 'test_ll': 0.5879, 'iter_50k_ll': 0.6372, 'iter_50k_roi': 14.24, 'iter_50k_roi_ci': [-6.77, 36.68], 'iter_100k_roi': 17.18},
    {'config': 'alt_md5 (md=5 lr=0.10 ss=0.7 cs=0.7)', 'val_ll': 0.6465, 'test_ll': 0.5988, 'iter_50k_ll': 0.6459, 'iter_50k_roi': 21.96, 'iter_50k_roi_ci': [2.43, 43.71], 'iter_100k_roi': 27.32},
    {'config': 'alt_md3 (md=3 lr=0.05 ss=0.8 cs=0.7)', 'val_ll': 0.6466, 'test_ll': 0.5978, 'iter_50k_ll': 0.6382, 'iter_50k_roi': 16.48, 'iter_50k_roi_ci': [-4.04, 39.13], 'iter_100k_roi': 18.75},
    {'config': 'alt_balanced (md=4 lr=0.05 ss=0.8 cs=0.8)', 'val_ll': 0.6473, 'test_ll': 0.5919, 'iter_50k_ll': 0.6356, 'iter_50k_roi': 11.97, 'iter_50k_roi_ci': [-9.05, 34.30], 'iter_100k_roi': 17.18},
]


# ---------------------------------------------------------------------------
# IPL 2026 per-match equity curve — extracted from existing dashboard's
# Chart.js dataset (xgb_match_v2_frozen, NOT M7 production — kept as a
# historical record. Re-running with the M7 model is a TODO.)
# ---------------------------------------------------------------------------
IPL_2026_EQUITY = {
    'model': 'xgb_match_v2_frozen',
    'caveat': 'This equity curve uses xgb_match_v2_frozen (the previous production), NOT M7. Iteration-set IPL matches were part of test split that drove model selection — not strictly out-of-sample. Golden subset (post 2026-04-17) is.',
    'dates': ["2026-03-28","2026-03-29","2026-03-30","2026-03-31","2026-04-01","2026-04-02","2026-04-03","2026-04-04","2026-04-04","2026-04-05","2026-04-05","2026-04-06","2026-04-07","2026-04-08","2026-04-09","2026-04-10","2026-04-11","2026-04-11","2026-04-12","2026-04-12","2026-04-13","2026-04-14","2026-04-15","2026-04-16","2026-04-17","2026-04-18","2026-04-18","2026-04-19","2026-04-19","2026-04-20","2026-04-21","2026-04-22","2026-04-23","2026-04-24","2026-04-25","2026-04-25","2026-04-26","2026-04-26","2026-04-27","2026-04-28","2026-04-29","2026-04-30","2026-05-01","2026-05-02","2026-05-03","2026-05-03","2026-05-04","2026-05-05","2026-05-06","2026-05-07"],
    'cum_pnl': [0.0,-1.0,-0.058,-1.058,-2.058,-0.953,-0.118,1.622,2.642,1.642,2.308,2.308,3.607,4.713,5.654,6.635,5.635,6.785,5.785,7.084,8.331,7.331,6.331,7.529,8.268,7.268,8.137,9.606,8.606,9.476,8.476,7.476,9.142,8.142,8.944,10.049,11.111,11.111,10.111,11.41,12.515,13.925,15.278,16.525,17.994,19.293,19.974,21.221,22.326,21.326],
    'cum_roi': [0.0,-100.0,-2.92,-35.28,-51.46,-19.06,-1.97,23.17,33.02,18.24,23.09,23.09,32.79,39.27,43.5,47.39,37.56,42.41,34.03,39.36,43.85,36.66,30.15,34.22,35.95,30.28,32.55,36.95,31.88,33.84,29.23,24.92,29.49,25.44,27.1,29.56,31.75,31.75,28.09,30.84,32.94,35.71,38.19,40.31,42.84,44.87,45.4,47.16,48.54,45.38],
    'final_pnl': 21.33,
    'final_roi': 45.4,
    'wins': 32,
    'losses': 15,
    'bets': 47,
}


def parse_ipl_dashboard_rows() -> list[dict]:
    """Extract per-match rows from reports/ipl_2026_dashboard_clean.html so the
    predictions tracker can render the same per-match table from JSON without
    re-running the build_ipl_dashboard pipeline.
    """
    html = (REPORTS / 'ipl_2026_dashboard_clean.html').read_text()
    # Each <tr class='row-win|row-loss|'> ... </tr> block
    row_re = re.compile(r"<tr class='([^']*)'>(.*?)</tr>", re.S)
    out = []
    for m in row_re.finditer(html):
        cls, body = m.group(1), m.group(2)
        # First td is the row index — used to skip the header row
        idx_m = re.search(r'<td>(\d+)</td>', body)
        if not idx_m:
            continue
        idx = int(idx_m.group(1))
        # Date / venue / tag (live/golden)
        date_m = re.search(r"<td>(\d{4}-\d{2}-\d{2})<br><span class='dim'>([^<]+)</span><br><span class='tag tag-(live|golden)'>", body)
        if not date_m:
            continue
        date, venue, set_tag = date_m.group(1), date_m.group(2), date_m.group(3)
        # Matchup
        teams_m = re.search(r"<div class='matchup'><b>([^<]+)</b><br>vs<br><b>([^<]+)</b></div>", body)
        team1, team2 = (teams_m.group(1), teams_m.group(2)) if teams_m else ('', '')
        toss_m = re.search(r"<div class='dim toss'>([^<]+)</div>", body)
        toss = toss_m.group(1) if toss_m else ''
        # Result line + scores
        result_m = re.search(r"<div class='result-line'>([^<]+)</div>", body)
        result = result_m.group(1) if result_m else ''
        scores = re.findall(r"<div class='score'>([^<]+)</div>", body)
        # Polymarket odds + volume
        odds_pairs = re.findall(r"<div><b>([^<]+)</b>: ([\d.]+) \(([\d.]+)%\)</div>", body)
        market = [{'team': t, 'odds': float(o), 'pct': float(p)} for t, o, p in odds_pairs[:2]]
        vol_m = re.search(r"<div class='vol'>vol \$([\d,]+)</div>", body)
        volume = int(vol_m.group(1).replace(',', '')) if vol_m else None
        # Prediction
        pred_pairs = re.findall(r"<div><b>([^<]+)</b>: ([\d.]+)%</div>", body)
        prediction = [{'team': t, 'pct': float(p)} for t, p in pred_pairs[:2]]
        pred_src_m = re.search(r"<div class='pred-source'>(\w+)</div>", body)
        pred_src = pred_src_m.group(1) if pred_src_m else None
        # Bet placed
        bet_m = re.search(r"<td><div><b>([^<]+)</b></div><div class='edge'>edge ([+\-][\d.]+)pp @ ([\d.]+)</div></td>", body)
        bet = None
        if bet_m:
            bet = {'team': bet_m.group(1), 'edge_pp': float(bet_m.group(2)), 'odds': float(bet_m.group(3))}
        # PnL
        pnl_m = re.search(r"<td class='pnl-cell'><span class='pnl-(win|loss)'>([+\-][\d.]+)</span></td>", body)
        pnl = None
        if pnl_m:
            pnl = float(pnl_m.group(2))
        # Cumulative
        cum_m = re.search(r"<td class='cum-cell'><div>([+\-][\d.]+)</div><div class='dim'>ROI ([+\-][\d.]+)%</div><div class='dim'>(\d+)/(\d+)</div></td>", body)
        if cum_m:
            cum = {'pnl': float(cum_m.group(1)), 'roi_pct': float(cum_m.group(2)),
                   'wins': int(cum_m.group(3)), 'bets': int(cum_m.group(4))}
        else:
            cum = {'pnl': None, 'roi_pct': None, 'wins': None, 'bets': None}
        out.append({
            'idx': idx, 'date': date, 'venue': venue, 'set': set_tag,
            'team1': team1, 'team2': team2, 'toss': toss,
            'result': result, 'scores': scores,
            'market': market, 'volume_usd': volume,
            'prediction': prediction, 'pred_source': pred_src,
            'bet': bet, 'pnl': pnl, 'cum': cum,
            'row_class': cls,
        })
    return out


def load_fixtures() -> list[dict]:
    """Pending / recent fixtures from fixtures/*.json (skip _template, _validation)."""
    out = []
    for p in sorted(FIXTURES.glob('*.json')):
        if p.name.startswith('_'):
            continue
        d = json.loads(p.read_text())
        out.append({
            'file': p.name,
            'date': d.get('date'),
            'team1': d.get('team1'),
            'team2': d.get('team2'),
            'venue': d.get('venue'),
            'competition': d.get('competition_tier'),
            'team1_lineup_names': d.get('_team1_lineup_names', []),
            'team2_lineup_names': d.get('_team2_lineup_names', []),
            'toss_winner': d.get('toss_winner'),
            'toss_decision': d.get('toss_decision'),
            'polymarket_odds': d.get('polymarket_odds'),
        })
    return out


# ---------------------------------------------------------------------------
# Phase narrative — for the "step through M1 → M8" view
# ---------------------------------------------------------------------------
PHASE_NARRATIVE = [
    {
        'phase': 'M1', 'title': 'Eval infrastructure + monotone constraints',
        'outcome': 'LANDED', 'outcome_color': 'green',
        'date': '2026-05-10',
        'goal': 'Build the per-slice eval lens and add monotone constraints as a generalization guard for M2+ feature work.',
        'what_we_did': 'Stratified bootstrap CIs, adversarial slices (close/mismatch/IPL/intl), walk-forward by month, Platt-as-sizing layer, monotone constraints on 10 directional features.',
        'result': 'M1+Platt was the FIRST variant to clear iter ≥$50k LL gate (0.6235 < market 0.6267) AND ROI CI > 0 simultaneously. Golden ~0.025 LL worse than v2_clean — within bootstrap noise on n=50, not promoted.',
        'lesson': 'Monotone constraints + Platt as sizing layer give resolution-preserving calibration. Use Platt for sizing, RAW for the LL gate metric.',
    },
    {
        'phase': 'M2', 'title': 'Outcome-dist transfer (venue-only landed)',
        'outcome': 'LANDED (subset)', 'outcome_color': 'green',
        'date': '2026-05-10',
        'goal': 'Transfer v7 ball-level outcome-dist features (P(four), P(six), P(wicket)) up to match level via lineup aggregation.',
        'what_we_did': 'Added 21 features (top-6 batter pX, bottom-5 bowler pX, venue pX). Drop-one ablation across batter / bowler / venue.',
        'result': 'Bowler features actively hurt LL (bottom-5 squad-order is not a clean bowling unit). Batter features neutral. Venue (3 features) is the only group that helps. Landed as M2 venue-only — first variant with positive iter ≥$50k ROI CI lower bound (+2.73).',
        'lesson': 'Drop-one is essential — full feature groups can mask which subset carries the signal. Bottom-5-by-squad-order is not a bowling unit.',
    },
    {
        'phase': 'M3', 'title': 'Rolling form DROPPED + UNFROZEN materialization adopted',
        'outcome': 'UNFROZEN MODE LANDED', 'outcome_color': 'green',
        'date': '2026-05-10',
        'goal': 'Add player-level rolling-form features (last-5-match avg/SR per player, in-form / out-of-form indicators).',
        'what_we_did': 'Added 18 rolling-form features. Drop-one showed bowling-recent hurts (same bottom-5 problem). Stale-tracker hypothesis test: re-materialized in UNFROZEN mode.',
        'result': 'M3 features themselves do not add value — career aggregates already capture variance. But the UNFROZEN materialization fix delivers the M3-phase improvement: M2 v.o. unfrozen clears BOTH ≥$50k gates simultaneously (LL 0.6279 < market, ROI +26.69% [+6.26, +48.65]).',
        'lesson': 'Frozen at val/test boundary lets test features drift past the model\'s training scope. Unfrozen (chronological tracker walk per fixture) matches deployment semantics and clears the gates.',
    },
    {
        'phase': 'M4', 'title': 'Within-tournament features DROPPED',
        'outcome': 'DROPPED', 'outcome_color': 'red',
        'date': '2026-05-10',
        'goal': 'Add date-windowed form (60d), competition-filtered form, scheduling proxies (days_since, back_to_back).',
        'what_we_did': 'Added 15 features. Trained M4 full + 4 drop-one variants.',
        'result': 'Form features 0.7-0.8 correlated with M1 win_rate_diff (redundant). Scheduling features have ~zero target r but XGBoost picks them up via interactions, fitting noise. Net: ~10% more confident predictions, LESS accurate at the tail (89.6%→86.0% at |p−0.5|>0.20).',
        'lesson': 'Discipline added: pre-training correlation check vs M1+M2 baseline. |r|>0.5 against existing requires demonstrating orthogonal target signal.',
    },
    {
        'phase': 'M5', 'title': 'Player × opp affinity DROPPED at correlation check',
        'outcome': 'DROPPED (pre-train)', 'outcome_color': 'red',
        'date': '2026-05-10',
        'goal': 'Add 8 player × opposition features via h2h-matrix aggregation, shrunk to career.',
        'what_we_did': 'Implemented; ran the correlation check before training.',
        'result': '7 of 8 features fail redundancy (|r|>0.5 with M1 baseline AND no higher target r); 8th borderline. Lineup aggregation collapses per-player matchup signal toward team career means. SKIPPED training entirely.',
        'lesson': 'First successful pre-training rejection. Aggregated-player features at the match level structurally collapse to team-level career features.',
    },
    {
        'phase': 'M6', 'title': 'Conditions DROPPED + target-floor discipline added',
        'outcome': 'DROPPED', 'outcome_color': 'red',
        'date': '2026-05-10',
        'goal': 'Add 3 date-derived condition features (month_of_year, day_of_week, is_dew_prone_month).',
        'what_we_did': 'Passed redundancy check. Trained anyway to confirm M4 over-confidence pattern.',
        'result': 'Same M4 pattern repeats: standalone test LL improves -0.007 BUT iteration ROI drops, tail accuracy regresses -3.2pp at |p−0.5|>0.15. All 3 features have |target r| ≤ 0.011.',
        'lesson': 'Discipline upgrade: dual correlation check (redundancy AND target-floor |target r| ≥ 0.03). 5 of 5 v3 feature phases now drop — frontier exhausted, M7 must be ARCHITECTURE work.',
    },
    {
        'phase': 'M7', 'title': 'Architecture sweep (hyperparameters retuned)',
        'outcome': 'LANDED', 'outcome_color': 'green',
        'date': '2026-05-10',
        'goal': 'After 5/5 feature phases dropped, attack hyperparameters: 81-config grid (max_depth × lr × subsample × colsample).',
        'what_we_did': 'Trained all 81 configs in 13s. Selected top 6 by val LL, validated each on iter ≥$50k. Picked best_val: lr 0.10→0.05, cs 0.8→0.9.',
        'result': 'Iter ≥$100k ROI CI cleanly excludes 0 for the first time (+0.57). Close-match slice ROI CI also clears (+4.36) — historically weakest slice. 2026-04 IPL: ROI +34.87% [+2.04, +68.06], win 65.7%. Old config was over-aggressive (early-stopped at ~60 rounds; new config trains ~80 rounds with smaller steps).',
        'lesson': 'Picked best_val NOT best_test. Standalone test LL alone is misleading; iter ≥$50k ROI/LL is the gate. Production uses RAW probabilities — Platt over-corrects on this lower-lr config and kills iteration ROI.',
    },
    {
        'phase': 'M8', 'title': 'Sizing rules: simpler is better',
        'outcome': 'LANDED', 'outcome_color': 'green',
        'date': '2026-05-10',
        'goal': 'Test edge thresholds + Kelly variants. Conventional wisdom: bet only when edge > 3%, fractional Kelly stake.',
        'what_we_did': 'Sweep over thresholds {0, 1%, 2%, 3%, 5%, 7%, 10%, 15%}. Tested flat, quarter-Kelly, full-Kelly with/without 2% per-bet cap.',
        'result': 'Counter-intuitive: flat 1-unit at threshold 0 is the only config where iter ≥$50k ROI CI cleanly excludes 0 (+0.91 lower bound). Higher thresholds increase point ROI (cherry-picking) but widen CIs faster. Slice-conditional finding: on mismatch fixtures, threshold=10% wins (+44.06% [+1.15, +78.30], win 72%) — documented for future, not landed.',
        'lesson': 'M7 production probabilities have enough resolution that even 1% edges carry signal. The "edge > threshold" heuristic from betting folklore doesn\'t apply to a calibrated low-lr model.',
    },
]


def main() -> None:
    data = {
        'generated_at': '2026-05-14',
        'production': {
            'match_level': {
                'model_id': 'xgb_match_v3_m7_production',
                'feature_count': 49,
                'mode': 'unfrozen, raw probabilities',
                'sizing': 'flat 1 unit, edge threshold 0',
                'headline_iter_50k': {'ll': 0.6299, 'market_ll': 0.6267, 'roi': 21.90, 'roi_ci': [2.28, 43.83]},
                'headline_iter_100k': {'ll': 0.5929, 'roi': 26.39, 'roi_ci': [0.57, 58.78]},
                'headline_close': {'ll': 0.6880, 'roi': 33.27, 'roi_ci': [4.36, 61.53], 'win_pct': 56.8},
                'headline_2026_04_ipl': {'roi': 34.87, 'roi_ci': [2.04, 68.06], 'win_pct': 65.7},
            },
            'ball_level': {
                'model_id': 'xgb_v7_hierarchical_shrink',
                'role': 'props / scores / in-play (lost winner-market race)',
                'feature_count': 114,
                'top_props_skill': [
                    ('batter_fours_2plus', 0.091),
                    ('batter_fours_1plus', 0.075),
                    ('batter_fours_3plus', 0.058),
                ],
                'fade_props': [
                    ('pp_total_ou_55_5', -0.183),
                    ('bowler_wkts_2plus', -0.161),
                ],
            },
        },
        'benchmarks': BENCHMARKS,
        'match_level_models': MATCH_LEVEL_MODELS,
        'ball_level_models': BALL_LEVEL_MODELS,
        'phase_narrative': PHASE_NARRATIVE,
        'sizing_sweep_50k': load_sizing_sweep('50k'),
        'sizing_sweep_100k': load_sizing_sweep('100k'),
        'hparam_sweep_top': HPARAM_SWEEP_TOP,
        'feature_importance': load_feature_importance(),
        'ipl_2026_equity': IPL_2026_EQUITY,
        'ipl_2026_matches': parse_ipl_dashboard_rows(),
        'fixtures': load_fixtures(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2))
    print(f'Wrote {OUT.relative_to(REPO)} ({OUT.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
