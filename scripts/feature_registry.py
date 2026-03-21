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


# Convenience: default v3 groups (all groups)
V3_GROUPS = list(FEATURE_GROUPS.keys())

# Convenience: v2 groups (no metadata/matchup/type_based)
V2_GROUPS = ['basic', 'player_stats', 'h2h', 'momentum', 'pressure', 'chase', 'medium']
