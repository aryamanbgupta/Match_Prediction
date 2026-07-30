"""
Feature Registry — Central source of truth for all features used across training scripts.

Usage:
    from feature_registry import FEATURE_GROUPS, resolve_feature_list, get_feature_hash

    # Get all v3 features
    features = resolve_feature_list(['basic', 'player_stats', 'h2h', 'momentum',
                                      'pressure', 'chase', 'medium',
                                      'player_metadata', 'matchup', 'type_based'])

    # Ablation: remove metadata features
    features = resolve_feature_list(['basic', 'player_stats', 'h2h', 'momentum',
                                      'pressure', 'chase', 'medium',
                                      'matchup', 'type_based'])

    # Get deterministic hash for smart caching
    hash_val = get_feature_hash(features)
"""

import hashlib
from typing import Dict, List, Optional, Set


# ── Feature Groups ──────────────────────────────────────────────────────────

FEATURE_GROUPS: Dict[str, List[str]] = {
    'basic': [
        'inning_idx', 'score', 'wickets', 'balls_bowled', 'run_rate',
        'wickets_ratio', 'balls_ratio', 'wickets_in_hand', 'balls_remaining',
        'is_powerplay', 'is_middle_overs', 'is_death_overs', 'balls_in_over',
        'venue_encoded', 'is_toss_winner', 'is_batting_first',
    ],

    'player_stats': [
        'batter_encoded', 'bowler_encoded',
        'batsman_avg', 'batsman_sr', 'bowler_avg', 'bowler_econ',
        'batsman_recent_avg', 'batsman_recent_sr',
        'bowler_recent_avg', 'bowler_recent_econ',
        'batter_balls_faced', 'batter_runs_scored',
        'bowler_balls_in_innings', 'bowler_overs_in_innings',
    ],

    'h2h': [
        'h2h_avg', 'h2h_sr',
    ],

    'momentum': [
        'last_5_balls_runs', 'last_10_balls_runs', 'last_30_balls_runs',
        'balls_since_boundary', 'last_10_dots',
        'partnership_runs',
    ],

    'pressure': [
        'dot_percentage_recent', 'boundary_percentage_recent',
        'pressure_cooker_index',
    ],

    'chase': [
        'chase_target', 'run_rate_required', 'lead_gap',
    ],

    'medium': [
        'venue_avg_score', 'non_striker_sr',
    ],

    'player_metadata': [
        'batter_hand', 'bowler_arm', 'is_pace', 'bowling_type',
        'batter_age', 'bowler_age',
    ],

    'matchup': [
        'spin_matchup_advantage', 'same_arm_matchup', 'matchup_type_encoded',
    ],

    'type_based': [
        'batter_avg_vs_pace', 'batter_sr_vs_pace',
        'batter_avg_vs_spin', 'batter_sr_vs_spin',
        'bowler_avg_vs_lhb', 'bowler_econ_vs_lhb',
        'bowler_avg_vs_rhb', 'bowler_econ_vs_rhb',
    ],

    'team_strength': [
        'striker_elo', 'bowler_elo_rating',
        'batting_team_elo', 'bowling_team_elo', 'elo_diff',
        'team_batting_avg', 'team_batting_sr',
        'team_bowling_avg', 'team_bowling_econ',
    ],

    'venue_profile': [
        'venue_boundary_pct',
        'venue_dot_pct',
        'venue_wicket_rate',
        'venue_powerplay_avg',
        'venue_death_avg',
        'venue_first_innings_avg',
        'venue_chase_win_pct',
    ],

    'match_context': [
        'chose_to_bat',
        'match_importance',
        'is_international',
        'competition_tier',
    ],

    # Schema v4: empirical-Bayes-shrunk outcome distributions.
    # Each bucket emits P(0,1,2,4,6,W) shrunk toward the global corpus
    # prior π via Dirichlet-posterior-mean shrinkage:
    #     p̂_c = (n_c + k · π_c) / (N + k)
    # k=30 for per-player cells, k=200 for venue. See
    # IMPROVEMENTS.md §"Empirical Outcome Distributions".
    'batter_outcome_dist': [
        'batter_p0', 'batter_p1', 'batter_p2',
        'batter_p4', 'batter_p6', 'batter_pw',
    ],
    'bowler_outcome_dist': [
        'bowler_p0', 'bowler_p1', 'bowler_p2',
        'bowler_p4', 'bowler_p6', 'bowler_pw',
    ],
    'batter_vs_type_dist': [
        'batter_p0_vs_pace', 'batter_p1_vs_pace', 'batter_p2_vs_pace',
        'batter_p4_vs_pace', 'batter_p6_vs_pace', 'batter_pw_vs_pace',
        'batter_p0_vs_spin', 'batter_p1_vs_spin', 'batter_p2_vs_spin',
        'batter_p4_vs_spin', 'batter_p6_vs_spin', 'batter_pw_vs_spin',
    ],
    'bowler_vs_hand_dist': [
        'bowler_p0_vs_lhb', 'bowler_p1_vs_lhb', 'bowler_p2_vs_lhb',
        'bowler_p4_vs_lhb', 'bowler_p6_vs_lhb', 'bowler_pw_vs_lhb',
        'bowler_p0_vs_rhb', 'bowler_p1_vs_rhb', 'bowler_p2_vs_rhb',
        'bowler_p4_vs_rhb', 'bowler_p6_vs_rhb', 'bowler_pw_vs_rhb',
    ],
    'venue_outcome_dist': [
        'venue_p0', 'venue_p1', 'venue_p2',
        'venue_p4', 'venue_p6', 'venue_pw',
    ],

    # Phase 3 of outcome-dist follow-ups: phase prior. Per ball, emit 6
    # features sourced from the phase prior matching the ball's phase
    # (powerplay / middle / death). The phase boundaries match
    # parsing_v2.calculate_basic_features (PP: balls<36, mid: 36..<96,
    # death: balls>=96). Phase priors are global constants over millions
    # of balls — no per-cell shrinkage needed; loaded from _meta at
    # runtime by stats_sqlite_backend.
    'phase_outcome_dist': [
        'phase_p0', 'phase_p1', 'phase_p2',
        'phase_p4', 'phase_p6', 'phase_pw',
    ],

    # I8 / schema v5: the current-phase player cells and the exact
    # batter-bowler matchup. These are 6-class distributions, not raw
    # sparse rates; their hierarchy and fixed shrinkage strengths are
    # defined in docs/I8_FEATURE_CONTRACT.md.
    'batter_phase_dist': [
        'batter_phase_p0', 'batter_phase_p1', 'batter_phase_p2',
        'batter_phase_p4', 'batter_phase_p6', 'batter_phase_pw',
    ],
    'bowler_phase_dist': [
        'bowler_phase_p0', 'bowler_phase_p1', 'bowler_phase_p2',
        'bowler_phase_p4', 'bowler_phase_p6', 'bowler_phase_pw',
    ],
    'h2h_outcome_dist': [
        'h2h_p0', 'h2h_p1', 'h2h_p2',
        'h2h_p4', 'h2h_p6', 'h2h_pw',
    ],
}


# Categorical features: encoded column name → source column
CATEGORICAL_FEATURES: Dict[str, str] = {
    'batter_encoded': 'batter_id',
    'bowler_encoded': 'bowler_id',
    'venue_encoded': 'venue',
    'matchup_type_encoded': 'matchup_type',
}


# All features across all groups
ALL_FEATURES: Set[str] = set()
for _group in FEATURE_GROUPS.values():
    ALL_FEATURES.update(_group)


def resolve_feature_list(
    groups: List[str],
    exclude: Optional[List[str]] = None,
    include_extra: Optional[List[str]] = None,
) -> List[str]:
    """Resolve a feature list from group names with optional exclusions/additions.

    Args:
        groups: List of group names from FEATURE_GROUPS.
        exclude: Individual feature names to drop.
        include_extra: Extra feature names to add beyond groups.

    Returns:
        Ordered list of feature names (preserves group ordering, no duplicates).
    """
    seen = set()
    features = []

    for group in groups:
        if group not in FEATURE_GROUPS:
            raise ValueError(f"Unknown feature group: '{group}'. "
                             f"Available: {sorted(FEATURE_GROUPS.keys())}")
        for feat in FEATURE_GROUPS[group]:
            if feat not in seen:
                features.append(feat)
                seen.add(feat)

    if include_extra:
        for feat in include_extra:
            if feat not in seen:
                features.append(feat)
                seen.add(feat)

    if exclude:
        exclude_set = set(exclude)
        features = [f for f in features if f not in exclude_set]

    return features


def get_feature_hash(feature_list: List[str]) -> str:
    """Deterministic hash of a feature list for smart caching."""
    canonical = sorted(set(feature_list))
    content = ','.join(canonical)
    return hashlib.sha256(content.encode()).hexdigest()[:12]


# Convenience: default v3/v4 groups (excludes venue_profile + match_context)
V3_GROUPS = ['basic', 'player_stats', 'h2h', 'momentum', 'pressure', 'chase', 'medium',
             'player_metadata', 'matchup', 'type_based', 'team_strength']

# Convenience: v2 groups (no metadata/matchup/type_based)
V2_GROUPS = ['basic', 'player_stats', 'h2h', 'momentum', 'pressure', 'chase', 'medium']

# Convenience: v5 groups (all groups including venue_profile + match_context)
V5_GROUPS = list(FEATURE_GROUPS.keys())

# Convenience: v6 groups — V3 baseline + 5 new empirical-outcome-distribution
# groups (42 new features). Does NOT include venue_profile / match_context
# (IMPROVEMENTS.md §"Venue Profile + Match Context Features" — those hurt
# all metrics; kept disabled since the March 2026 experiments).
V6_GROUPS = V3_GROUPS + [
    'batter_outcome_dist',
    'bowler_outcome_dist',
    'batter_vs_type_dist',
    'bowler_vs_hand_dist',
    'venue_outcome_dist',
]

# Convenience: v7 groups — V6 + Phase 3 phase prior (6 features dispatched
# by ball-phase). Strictly additive over V6.
V7_GROUPS = V6_GROUPS + ['phase_outcome_dist']

# I8 is isolated from the optional global phase-prior experiment: it adds
# exactly 18 candidate features to the canonical I7/V6 recipe.
I8_GROUPS = V6_GROUPS + [
    'batter_phase_dist',
    'bowler_phase_dist',
    'h2h_outcome_dist',
]
