# parse for xgboost model.py
import json
import numpy as np
from pathlib import Path
import pandas as pd
from collections import defaultdict, deque
from datetime import datetime
import pickle

from identity_maps import canonicalize_venue

# Import player metadata provider for Tier 1/2/3 features
from player_metadata import (
    PlayerMetadataProvider,
    encode_batter_hand,
    encode_bowler_arm,
    encode_is_pace,
    encode_bowling_type
)

ICC_FULL_MEMBERS = {
    'India', 'Australia', 'England', 'South Africa', 'New Zealand',
    'Pakistan', 'Sri Lanka', 'West Indies', 'Bangladesh',
    'Zimbabwe', 'Afghanistan', 'Ireland'
}

PREMIUM_LEAGUES = {
    'Indian Premier League', 'Big Bash League', 'Caribbean Premier League',
    'SA20', 'International League T20', 'Major League Cricket',
    'Pakistan Super League'
}

STANDARD_LEAGUES = {
    'Vitality Blast', 'NatWest T20 Blast', 'CSA T20 Challenge',
    'Ram Slam T20 Challenge', 'Super Smash', 'Bangladesh Premier League'
}

def classify_match_k_factor(event_name, team_type, teams):
    """Return K-factor based on match importance."""
    event_lower = event_name.lower()

    # ICC events (World Cup, World Twenty20)
    if 'world cup' in event_lower or 'world twenty20' in event_lower:
        return 4.0

    # Premium franchise leagues
    if event_name in PREMIUM_LEAGUES:
        return 4.0

    # Standard leagues
    if event_name in STANDARD_LEAGUES:
        return 2.0

    # International matches
    if team_type == 'international':
        full_member_count = sum(1 for t in teams if t in ICC_FULL_MEMBERS)
        if full_member_count >= 2:
            return 4.0  # Both full members
        elif full_member_count == 1:
            return 2.0  # One full member vs associate
        else:
            return 1.0  # Both associates

    # Default for unknown leagues/domestic
    return 1.0


def classify_match_context(event_name, team_type, teams):
    """Return match context features for venue/match-level modeling.

    Returns:
        dict with match_importance (1-4), is_international (0/1), competition_tier (1-4)
    """
    event_lower = event_name.lower()
    is_international = 1 if team_type == 'international' else 0

    # match_importance: how much pressure/stakes
    if 'world cup' in event_lower or 'world twenty20' in event_lower:
        match_importance = 4
    elif event_name in PREMIUM_LEAGUES:
        match_importance = 3
    elif is_international and any(t in ICC_FULL_MEMBERS for t in teams):
        match_importance = 3
    elif event_name in STANDARD_LEAGUES:
        match_importance = 2
    else:
        match_importance = 1

    # competition_tier: quality of competition
    if 'world cup' in event_lower or 'world twenty20' in event_lower:
        competition_tier = 4
    elif event_name in PREMIUM_LEAGUES:
        competition_tier = 3
    elif event_name in STANDARD_LEAGUES or (is_international and any(t in ICC_FULL_MEMBERS for t in teams)):
        competition_tier = 2
    else:
        competition_tier = 1

    return {
        'match_importance': match_importance,
        'is_international': is_international,
        'competition_tier': competition_tier,
    }


class PlayerEloTracker:
    """Ball-by-ball ELO for batters and bowlers.

    Each delivery is a mini-match: batter vs bowler.
    Scoring above expected = batter "wins", below = bowler "wins".
    ELO follows the player across all leagues (implicit cross-league calibration).
    """
    DEFAULT_ELO = 1500.0
    DEFAULT_K_FACTOR = 4.0

    def __init__(self):
        self.batting_elo = {}   # player_id -> float
        self.bowling_elo = {}   # player_id -> float

    def get_batting_elo(self, player_id):
        return self.batting_elo.get(player_id, self.DEFAULT_ELO)

    def get_bowling_elo(self, player_id):
        return self.bowling_elo.get(player_id, self.DEFAULT_ELO)

    def update(self, batter_id, bowler_id, runs, is_wicket, k_factor=None):
        """Update ELO after a ball."""
        k = k_factor if k_factor is not None else self.DEFAULT_K_FACTOR
        bat_elo = self.get_batting_elo(batter_id)
        bowl_elo = self.get_bowling_elo(bowler_id)

        # Expected outcome (from batter's perspective)
        expected_batter = 1.0 / (1.0 + 10 ** ((bowl_elo - bat_elo) / 400.0))

        # Actual outcome: linear mapping to [0, 1] scale
        # wicket=0.0, dot=0.4, 1=0.5, 2=0.6, 3=0.7, 4=0.8, 6=1.0
        # E[actual] ≈ 0.50 across typical T20 ball distribution
        if is_wicket:
            actual_batter = 0.0
        else:
            actual_batter = min(0.4 + runs * 0.1, 1.0)

        # ELO update
        self.batting_elo[batter_id] = bat_elo + k * (actual_batter - expected_batter)
        self.bowling_elo[bowler_id] = bowl_elo + k * ((1 - actual_batter) - (1 - expected_batter))

    def get_team_batting_elo(self, player_ids):
        """Sum of batting ELOs for a list of player IDs."""
        return sum(self.get_batting_elo(pid) for pid in player_ids)

    def get_team_bowling_elo(self, player_ids):
        """Sum of bowling ELOs for a list of player IDs."""
        return sum(self.get_bowling_elo(pid) for pid in player_ids)


_OUTCOME_COUNT_KEYS = ('c0', 'c1', 'c2', 'c4', 'c6', 'cw')
_BOWLER_WICKET_KINDS = {
    'bowled', 'caught', 'caught and bowled', 'lbw', 'stumped', 'hit wicket',
}
LEGACY_DELIVERY_SEMANTICS = 'inclusive_total_runs_v1'
I5_DELIVERY_SEMANTICS = 'legal_off_bat_v1'
DELIVERY_SEMANTICS = {
    LEGACY_DELIVERY_SEMANTICS,
    I5_DELIVERY_SEMANTICS,
}

# Zero-valued fallbacks for outcome distribution features. Used when
# parse_match_data_v2 is invoked without a prior (legacy pre-v4 path).
# Names MUST match the feature_registry dist-group columns exactly — the
# parquet schema is built from these keys.
_ZERO_BATTER_DIST = {f'batter_p{c}': 0.0 for c in ('0', '1', '2', '4', '6', 'w')}
_ZERO_BOWLER_DIST = {f'bowler_p{c}': 0.0 for c in ('0', '1', '2', '4', '6', 'w')}
_ZERO_BATTER_VS_TYPE_DIST = {
    **{f'batter_p{c}_vs_pace': 0.0 for c in ('0', '1', '2', '4', '6', 'w')},
    **{f'batter_p{c}_vs_spin': 0.0 for c in ('0', '1', '2', '4', '6', 'w')},
}
_ZERO_BOWLER_VS_HAND_DIST = {
    **{f'bowler_p{c}_vs_lhb': 0.0 for c in ('0', '1', '2', '4', '6', 'w')},
    **{f'bowler_p{c}_vs_rhb': 0.0 for c in ('0', '1', '2', '4', '6', 'w')},
}
_ZERO_VENUE_DIST = {f'venue_p{c}': 0.0 for c in ('0', '1', '2', '4', '6', 'w')}
# Phase 3: phase prior — 6 features per ball, dispatched by phase.
# `phase_priors` is a dict {'powerplay'|'middle'|'death': (p0,p1,p2,p4,p6,pw)}.
# When absent, all 6 phase features fall back to zero.
_ZERO_PHASE_DIST = {f'phase_p{c}': 0.0 for c in ('0', '1', '2', '4', '6', 'w')}
_PHASE_OUTCOME_KEYS = ('c0', 'c1', 'c2', 'c4', 'c6', 'cw')


def _classify_phase_pre_ball(balls_bowled: int) -> str:
    """Return 'powerplay' / 'middle' / 'death' for the phase the *next*
    ball is in, given pre-ball legal-ball count. Boundaries match
    calculate_basic_features (`is_powerplay = balls_bowled < 36`,
    `is_middle_overs = 36 <= balls_bowled < 96`,
    `is_death_overs = balls_bowled >= 96`). Used by both the live-emit
    path (parse_match_data_v2 ball loop) and the SQLite hot path in
    sim_v1_2._fill_outcome_dists, so the model sees the same phase tag
    at training and inference time."""
    if balls_bowled < 36:
        return 'powerplay'
    if balls_bowled < 96:
        return 'middle'
    return 'death'


def _phase_dist_from_priors(phase_priors, balls_bowled):
    """Look up the 6-tuple for this ball's phase and return a
    feature-named dict. Defensive: returns zeros if priors are missing
    or malformed."""
    if not phase_priors:
        return _ZERO_PHASE_DIST
    phase = _classify_phase_pre_ball(int(balls_bowled))
    p = phase_priors.get(phase)
    if p is None or len(p) != 6:
        return _ZERO_PHASE_DIST
    return {
        'phase_p0': p[0], 'phase_p1': p[1], 'phase_p2': p[2],
        'phase_p4': p[3], 'phase_p6': p[4], 'phase_pw': p[5],
    }


def _outcome_bucket_key(runs, is_wicket):
    """Collapse a delivery to its normalized-outcome count-bucket key.

    Aligns with `normalize_ball_outcome` (3→2, 5→4, 7+→6) and the XGBoost
    training target, so Σ cX always equals total balls in the same tracker
    dict.
    """
    if is_wicket:
        return 'cw'
    oc = normalize_ball_outcome(runs, False)
    # oc ∈ {0, 1, 2, 4, 6} after normalization.
    return f'c{oc}'


def _shrink_counts(counts, prior, k):
    """Dirichlet-posterior-mean shrinkage — mirrors
    stats_sqlite_backend._SQLiteBackend._shrink. Kept here so the live-
    tracker path (materializer, parity harness) doesn't depend on the
    backend module."""
    n = sum(counts)
    denom = n + k
    return tuple(
        (counts[i] + k * prior[i]) / denom for i in range(6)
    )


def _empty_batting_counts():
    return {'runs': 0, 'balls': 0, 'dismissals': 0,
            'c0': 0, 'c1': 0, 'c2': 0, 'c4': 0, 'c6': 0, 'cw': 0}


def _empty_bowling_counts():
    return {'runs_given': 0, 'balls_bowled': 0, 'wickets': 0,
            'c0': 0, 'c1': 0, 'c2': 0, 'c4': 0, 'c6': 0, 'cw': 0}


def _empty_outcome_counts():
    return {'c0': 0, 'c1': 0, 'c2': 0, 'c4': 0, 'c6': 0, 'cw': 0}


def _empty_h2h_counts():
    return {
        'runs': 0, 'balls': 0, 'dismissals': 0,
        **_empty_outcome_counts(),
    }


def _empty_phase_counts():
    return {
        phase: _empty_outcome_counts()
        for phase in ('powerplay', 'middle', 'death')
    }


class PlayerStatsTracker:
    """
    DESIGN DECISION: Separate class for player stats to maintain state across matches.
    REASONING: Encapsulation - keeps complex state management isolated from parsing logic.
    This makes it easy to add new stats without cluttering the main parsing code.
    """
    def __init__(self, enable_i8: bool = False):
        self.enable_i8 = bool(enable_i8)
        # Career stats - accumulate over time. Schema v4: each row also
        # carries c0/c1/c2/c4/c6/cw outcome counts so the model can read
        # empirical P(0,1,2,4,6,W | context) directly (no reconstruction
        # from avg/SR scalars).
        self.batting_stats = defaultdict(_empty_batting_counts)
        self.bowling_stats = defaultdict(_empty_bowling_counts)

        # NEW: Type-based batting stats (batter_id -> {pace/spin -> stats})
        # Tracks batter performance against pace vs spin bowlers
        self.batting_vs_type = defaultdict(lambda: {
            'pace': _empty_batting_counts(),
            'spin': _empty_batting_counts(),
        })

        # NEW: Type-based bowling stats (bowler_id -> {lhb/rhb -> stats})
        # Tracks bowler performance against left vs right hand batters
        self.bowling_vs_hand = defaultdict(lambda: {
            'left': _empty_bowling_counts(),
            'right': _empty_bowling_counts(),
        })

        # Head-to-head records
        # DESIGN DECISION: Use tuple (batter, bowler) as key for h2h
        # REASONING: Direct lookup, memory efficient, naturally handles bidirectional relationships
        self.h2h_stats = defaultdict(
            _empty_h2h_counts
            if self.enable_i8
            else lambda: {'runs': 0, 'balls': 0, 'dismissals': 0}
        )

        # I8 / schema v5. Kept disabled for v4 trackers so frozen parsing,
        # cache builds, and simulations pay no extra sparse-cell memory cost.
        self.batting_phase = (
            defaultdict(_empty_phase_counts) if self.enable_i8 else None
        )
        self.bowling_phase = (
            defaultdict(_empty_phase_counts) if self.enable_i8 else None
        )

        # Recent form tracking (last 5 matches)
        # DESIGN DECISION: Track last 5 match performances
        # REASONING: Recent form often more predictive than career average
        self.recent_batting = defaultdict(lambda: deque(maxlen=5))
        self.recent_bowling = defaultdict(lambda: deque(maxlen=5))

        # Track current match performance
        self.current_match_batting = defaultdict(lambda: {'runs': 0, 'balls': 0, 'dismissals': 0})
        self.current_match_bowling = defaultdict(lambda: {'runs_given': 0, 'balls_bowled': 0, 'wickets': 0})

    def start_match(self):
        """Reset current match stats"""
        self.current_match_batting.clear()
        self.current_match_bowling.clear()

    def end_match(self):
        """Push current match stats to recent history"""
        for player_id, stats in self.current_match_batting.items():
            # I5 can touch a striker on a wide without adding batter runs,
            # a legal ball, or a dismissal. That is not a batting
            # performance and must not consume one of the five recent-form
            # slots (nor create a match-log row).
            if any(int(value) != 0 for value in stats.values()):
                self.recent_batting[player_id].append(stats.copy())

        for player_id, stats in self.current_match_bowling.items():
            if any(int(value) != 0 for value in stats.values()):
                self.recent_bowling[player_id].append(stats.copy())

    def get_batting_features(self, batter_id):
        """Get batting stats features at current point in time"""
        stats = self.batting_stats[batter_id]
        # DESIGN DECISION: Return 0 for unknown players rather than None
        # REASONING: Models handle 0s better than missing values, represents "no history"
        if stats['balls'] == 0:
            return {'batsman_avg': 0, 'batsman_sr': 0, 'batsman_recent_avg': 0, 'batsman_recent_sr': 0}

        avg = stats['runs'] / max(stats['dismissals'], 1)  # Avoid division by zero
        sr = (stats['runs'] / stats['balls']) * 100 if stats['balls'] > 0 else 0

        # Calculate recent form
        recent_runs = sum(m['runs'] for m in self.recent_batting[batter_id])
        recent_balls = sum(m['balls'] for m in self.recent_batting[batter_id])
        recent_dismissals = sum(m['dismissals'] for m in self.recent_batting[batter_id])

        recent_avg = recent_runs / max(recent_dismissals, 1)
        recent_sr = (recent_runs / recent_balls) * 100 if recent_balls > 0 else 0

        return {
            'batsman_avg': avg,
            'batsman_sr': sr,
            'batsman_recent_avg': recent_avg,
            'batsman_recent_sr': recent_sr
        }

    def get_bowling_features(self, bowler_id):
        """Get bowling stats features at current point in time"""
        stats = self.bowling_stats[bowler_id]
        if stats['balls_bowled'] == 0:
            return {'bowler_avg': 0, 'bowler_econ': 0, 'bowler_recent_avg': 0, 'bowler_recent_econ': 0}

        avg = stats['runs_given'] / max(stats['wickets'], 1)
        econ = (stats['runs_given'] / stats['balls_bowled']) * 6 if stats['balls_bowled'] > 0 else 0

        # Calculate recent form
        recent_runs = sum(m['runs_given'] for m in self.recent_bowling[bowler_id])
        recent_balls = sum(m['balls_bowled'] for m in self.recent_bowling[bowler_id])
        recent_wickets = sum(m['wickets'] for m in self.recent_bowling[bowler_id])

        recent_avg = recent_runs / max(recent_wickets, 1)
        recent_econ = (recent_runs / recent_balls) * 6 if recent_balls > 0 else 0

        return {
            'bowler_avg': avg,
            'bowler_econ': econ,
            'bowler_recent_avg': recent_avg,
            'bowler_recent_econ': recent_econ
        }

    def get_h2h_features(self, batter_id, bowler_id):
        """Get head-to-head matchup features"""
        stats = self.h2h_stats[(batter_id, bowler_id)]
        if stats['balls'] == 0:
            return {'h2h_avg': 0, 'h2h_sr': 0}

        avg = stats['runs'] / max(stats['dismissals'], 1)
        sr = (stats['runs'] / stats['balls']) * 100 if stats['balls'] > 0 else 0
        return {'h2h_avg': avg, 'h2h_sr': sr}

    def get_batting_vs_type_features(self, batter_id):
        """
        Get batter's stats against pace and spin bowlers.
        NEW: Type-based batting features.
        """
        stats = self.batting_vs_type[batter_id]

        # vs Pace
        pace_stats = stats['pace']
        if pace_stats['balls'] == 0:
            pace_avg, pace_sr = 0.0, 0.0
        else:
            pace_avg = pace_stats['runs'] / max(pace_stats['dismissals'], 1)
            pace_sr = (pace_stats['runs'] / pace_stats['balls']) * 100

        # vs Spin
        spin_stats = stats['spin']
        if spin_stats['balls'] == 0:
            spin_avg, spin_sr = 0.0, 0.0
        else:
            spin_avg = spin_stats['runs'] / max(spin_stats['dismissals'], 1)
            spin_sr = (spin_stats['runs'] / spin_stats['balls']) * 100

        return {
            'batter_avg_vs_pace': pace_avg,
            'batter_sr_vs_pace': pace_sr,
            'batter_avg_vs_spin': spin_avg,
            'batter_sr_vs_spin': spin_sr,
        }

    def get_bowling_vs_hand_features(self, bowler_id):
        """
        Get bowler's stats against left and right hand batters.
        NEW: Hand-based bowling features.
        """
        stats = self.bowling_vs_hand[bowler_id]

        # vs LHB
        lhb_stats = stats['left']
        if lhb_stats['balls_bowled'] == 0:
            lhb_avg, lhb_econ = 0.0, 0.0
        else:
            lhb_avg = lhb_stats['runs_given'] / max(lhb_stats['wickets'], 1)
            lhb_econ = (lhb_stats['runs_given'] / lhb_stats['balls_bowled']) * 6

        # vs RHB
        rhb_stats = stats['right']
        if rhb_stats['balls_bowled'] == 0:
            rhb_avg, rhb_econ = 0.0, 0.0
        else:
            rhb_avg = rhb_stats['runs_given'] / max(rhb_stats['wickets'], 1)
            rhb_econ = (rhb_stats['runs_given'] / rhb_stats['balls_bowled']) * 6

        return {
            'bowler_avg_vs_lhb': lhb_avg,
            'bowler_econ_vs_lhb': lhb_econ,
            'bowler_avg_vs_rhb': rhb_avg,
            'bowler_econ_vs_rhb': rhb_econ,
        }

    def update_stats(self, batter_id, bowler_id, runs, is_wicket,
                     batter_hand=None, is_pace=None, *, bowler_runs=None,
                     is_legal=True, dismissed_batter_id=None,
                     is_bowler_wicket=None, phase=None):
        """
        Update all statistics after a delivery.

        Args:
            batter_id: Batter's cricsheet ID
            bowler_id: Bowler's cricsheet ID
            runs: Runs credited to the batter (off the bat)
            is_wicket: Whether any team wicket fell (the model target)
            batter_hand: 'left', 'right', or None (for type-based bowling stats)
            is_pace: True/False/None (for type-based batting stats)
            bowler_runs: Runs charged to the bowler. Defaults to ``runs`` for
                backwards compatibility with callers that represent a plain
                legal delivery.
            is_legal: False for wides and no-balls. Runs still accrue, but
                balls and legal-ball outcome counts do not.
            dismissed_batter_id: Player actually dismissed. This can be the
                non-striker on a run-out; defaults to the striker for legacy
                wicket calls.
            is_bowler_wicket: Whether the dismissal is credited to the
                bowler. Defaults to ``is_wicket`` for legacy calls.
            phase: I8 pre-ball phase name. Required for legal deliveries when
                this tracker was constructed with ``enable_i8=True``.
        """
        if bowler_runs is None:
            bowler_runs = runs
        if is_bowler_wicket is None:
            is_bowler_wicket = is_wicket
        if is_wicket and dismissed_batter_id is None:
            dismissed_batter_id = batter_id
        if (
            self.enable_i8
            and is_legal
            and phase not in ('powerplay', 'middle', 'death')
        ):
            raise ValueError(
                f"I8 tracker requires a valid pre-ball phase; got {phase!r}"
            )
        ck = _outcome_bucket_key(runs, is_wicket) if is_legal else None

        # Update batting stats
        bs = self.batting_stats[batter_id]
        bs['runs'] += runs
        if is_legal:
            bs['balls'] += 1
            bs[ck] += 1
        if dismissed_batter_id is not None:
            self.batting_stats[dismissed_batter_id]['dismissals'] += 1

        # Update bowling stats
        bw = self.bowling_stats[bowler_id]
        bw['runs_given'] += bowler_runs
        if is_legal:
            bw['balls_bowled'] += 1
            bw[ck] += 1
        if is_bowler_wicket:
            bw['wickets'] += 1

        # Update scalar H2H state and, under I8, its normalized outcome cell.
        h2h = self.h2h_stats[(batter_id, bowler_id)]
        h2h['runs'] += runs
        if is_legal:
            h2h['balls'] += 1
            if self.enable_i8:
                h2h[ck] += 1
        if is_bowler_wicket:
            h2h['dismissals'] += 1

        if self.enable_i8 and is_legal:
            self.batting_phase[batter_id][phase][ck] += 1
            self.bowling_phase[bowler_id][phase][ck] += 1

        # NEW: Update type-based batting stats (batter vs pace/spin)
        if is_pace is not None:
            type_key = 'pace' if is_pace else 'spin'
            bvt = self.batting_vs_type[batter_id][type_key]
            bvt['runs'] += runs
            if is_legal:
                bvt['balls'] += 1
                bvt[ck] += 1
            if dismissed_batter_id == batter_id:
                bvt['dismissals'] += 1

        # NEW: Update hand-based bowling stats (bowler vs LHB/RHB)
        if batter_hand in ('left', 'right'):
            bvh = self.bowling_vs_hand[bowler_id][batter_hand]
            bvh['runs_given'] += bowler_runs
            if is_legal:
                bvh['balls_bowled'] += 1
                bvh[ck] += 1
            if is_bowler_wicket:
                bvh['wickets'] += 1

        # Update current match stats
        self.current_match_batting[batter_id]['runs'] += runs
        if is_legal:
            self.current_match_batting[batter_id]['balls'] += 1
        if dismissed_batter_id is not None:
            self.current_match_batting[dismissed_batter_id]['dismissals'] += 1

        self.current_match_bowling[bowler_id]['runs_given'] += bowler_runs
        if is_legal:
            self.current_match_bowling[bowler_id]['balls_bowled'] += 1
        if is_bowler_wicket:
            self.current_match_bowling[bowler_id]['wickets'] += 1

    # --- Schema v4: empirical outcome distribution getters ------------------

    def _batting_counts_vec(self, batter_id):
        s = self.batting_stats[batter_id]
        return (s['c0'], s['c1'], s['c2'], s['c4'], s['c6'], s['cw'])

    def _bowling_counts_vec(self, bowler_id):
        s = self.bowling_stats[bowler_id]
        return (s['c0'], s['c1'], s['c2'], s['c4'], s['c6'], s['cw'])

    def get_batter_outcome_dist(self, batter_id, prior, k=30.0):
        p = _shrink_counts(self._batting_counts_vec(batter_id), prior, k)
        return {
            'batter_p0': p[0], 'batter_p1': p[1], 'batter_p2': p[2],
            'batter_p4': p[3], 'batter_p6': p[4], 'batter_pw': p[5],
        }

    def get_bowler_outcome_dist(self, bowler_id, prior, k=30.0):
        p = _shrink_counts(self._bowling_counts_vec(bowler_id), prior, k)
        return {
            'bowler_p0': p[0], 'bowler_p1': p[1], 'bowler_p2': p[2],
            'bowler_p4': p[3], 'bowler_p6': p[4], 'bowler_pw': p[5],
        }

    def get_batter_vs_type_outcome_dist(self, batter_id, prior, k=30.0,
                                        hierarchical=True):
        """Phase 5 hierarchical shrinkage: vs-pace and vs-spin cells
        shrink toward the batter's overall distribution (already shrunk
        toward π) instead of directly toward π. Mirrors
        `_SQLiteBackend.get_batter_vs_type_outcome_dist`. Set
        `hierarchical=False` for the legacy flat-shrink behavior."""
        def _vec(stats):
            return (stats['c0'], stats['c1'], stats['c2'],
                    stats['c4'], stats['c6'], stats['cw'])
        if hierarchical:
            parent = _shrink_counts(self._batting_counts_vec(batter_id),
                                    prior, k)
        else:
            parent = prior
        entry = self.batting_vs_type[batter_id]
        pp = _shrink_counts(_vec(entry['pace']), parent, k)
        ps = _shrink_counts(_vec(entry['spin']), parent, k)
        return {
            'batter_p0_vs_pace': pp[0], 'batter_p1_vs_pace': pp[1],
            'batter_p2_vs_pace': pp[2], 'batter_p4_vs_pace': pp[3],
            'batter_p6_vs_pace': pp[4], 'batter_pw_vs_pace': pp[5],
            'batter_p0_vs_spin': ps[0], 'batter_p1_vs_spin': ps[1],
            'batter_p2_vs_spin': ps[2], 'batter_p4_vs_spin': ps[3],
            'batter_p6_vs_spin': ps[4], 'batter_pw_vs_spin': ps[5],
        }

    def get_bowler_vs_hand_outcome_dist(self, bowler_id, prior, k=30.0,
                                        hierarchical=True):
        """Phase 5 hierarchical shrinkage: vs-LHB and vs-RHB cells shrink
        toward the bowler's overall distribution. Mirrors
        `_SQLiteBackend.get_bowler_vs_hand_outcome_dist`."""
        def _vec(stats):
            return (stats['c0'], stats['c1'], stats['c2'],
                    stats['c4'], stats['c6'], stats['cw'])
        if hierarchical:
            parent = _shrink_counts(self._bowling_counts_vec(bowler_id),
                                    prior, k)
        else:
            parent = prior
        entry = self.bowling_vs_hand[bowler_id]
        pl = _shrink_counts(_vec(entry['left']), parent, k)
        pr = _shrink_counts(_vec(entry['right']), parent, k)
        return {
            'bowler_p0_vs_lhb': pl[0], 'bowler_p1_vs_lhb': pl[1],
            'bowler_p2_vs_lhb': pl[2], 'bowler_p4_vs_lhb': pl[3],
            'bowler_p6_vs_lhb': pl[4], 'bowler_pw_vs_lhb': pl[5],
            'bowler_p0_vs_rhb': pr[0], 'bowler_p1_vs_rhb': pr[1],
            'bowler_p2_vs_rhb': pr[2], 'bowler_p4_vs_rhb': pr[3],
            'bowler_p6_vs_rhb': pr[4], 'bowler_pw_vs_rhb': pr[5],
        }

    def _require_i8(self):
        if not self.enable_i8:
            raise RuntimeError(
                "I8 outcome distributions require PlayerStatsTracker("
                "enable_i8=True)"
            )

    def get_batter_phase_outcome_dist(
        self,
        batter_id,
        prior,
        balls_bowled,
        k_player=30.0,
        k_phase=30.0,
    ):
        self._require_i8()
        phase = _classify_phase_pre_ball(int(balls_bowled))
        parent = _shrink_counts(
            self._batting_counts_vec(batter_id), prior, k_player)
        cell = self.batting_phase[batter_id][phase]
        counts = tuple(cell[key] for key in _OUTCOME_COUNT_KEYS)
        p = _shrink_counts(counts, parent, k_phase)
        return {
            'batter_phase_p0': p[0], 'batter_phase_p1': p[1],
            'batter_phase_p2': p[2], 'batter_phase_p4': p[3],
            'batter_phase_p6': p[4], 'batter_phase_pw': p[5],
        }

    def get_bowler_phase_outcome_dist(
        self,
        bowler_id,
        prior,
        balls_bowled,
        k_player=30.0,
        k_phase=30.0,
    ):
        self._require_i8()
        phase = _classify_phase_pre_ball(int(balls_bowled))
        parent = _shrink_counts(
            self._bowling_counts_vec(bowler_id), prior, k_player)
        cell = self.bowling_phase[bowler_id][phase]
        counts = tuple(cell[key] for key in _OUTCOME_COUNT_KEYS)
        p = _shrink_counts(counts, parent, k_phase)
        return {
            'bowler_phase_p0': p[0], 'bowler_phase_p1': p[1],
            'bowler_phase_p2': p[2], 'bowler_phase_p4': p[3],
            'bowler_phase_p6': p[4], 'bowler_phase_pw': p[5],
        }

    def get_h2h_outcome_dist(
        self,
        batter_id,
        bowler_id,
        prior,
        k_player=30.0,
        k_h2h=60.0,
    ):
        self._require_i8()
        batter_parent = _shrink_counts(
            self._batting_counts_vec(batter_id), prior, k_player)
        bowler_parent = _shrink_counts(
            self._bowling_counts_vec(bowler_id), prior, k_player)
        parent = tuple(
            (bat + bowl) / 2.0
            for bat, bowl in zip(batter_parent, bowler_parent)
        )
        cell = self.h2h_stats[(batter_id, bowler_id)]
        counts = tuple(cell[key] for key in _OUTCOME_COUNT_KEYS)
        p = _shrink_counts(counts, parent, k_h2h)
        return {
            'h2h_p0': p[0], 'h2h_p1': p[1], 'h2h_p2': p[2],
            'h2h_p4': p[3], 'h2h_p6': p[4], 'h2h_pw': p[5],
        }


class VenueStatsTracker:
    """
    DESIGN DECISION: Track venue statistics with temporal integrity.
    REASONING: Venue averages should only use historical data to avoid lookahead bias.
    Similar pattern to PlayerStatsTracker - accumulate stats and snapshot before each match.
    """
    def __init__(self):
        self.venue_stats = defaultdict(lambda: {
            'total_runs': 0,
            'innings_count': 0,
            'total_balls': 0,
            'total_boundaries': 0,
            'total_dots': 0,
            'total_wickets': 0,
            'powerplay_runs': 0,
            'powerplay_balls': 0,
            'death_runs': 0,
            'death_balls': 0,
            'first_innings_totals': [],
            'matches_total': 0,
            'chase_wins': 0,
            # Schema v4: 6-class outcome count bucket.
            'c0': 0, 'c1': 0, 'c2': 0, 'c4': 0, 'c6': 0, 'cw': 0,
        })

    def get_venue_avg_score(self, venue: str) -> float:
        """
        Get historical average score at venue (before current match).
        Returns 0 if no historical data exists.
        """
        stats = self.venue_stats[venue]
        if stats['innings_count'] == 0:
            return 0.0
        return stats['total_runs'] / stats['innings_count']

    def get_venue_profile(self, venue: str) -> dict:
        """Return computed venue profile features from accumulated stats."""
        stats = self.venue_stats[venue]
        total_balls = stats['total_balls']
        if total_balls == 0:
            return {
                'venue_boundary_pct': 0.0,
                'venue_dot_pct': 0.0,
                'venue_wicket_rate': 0.0,
                'venue_powerplay_avg': 0.0,
                'venue_death_avg': 0.0,
                'venue_first_innings_avg': 0.0,
                'venue_chase_win_pct': 0.5,
            }

        pp_balls = stats['powerplay_balls']
        death_balls = stats['death_balls']
        fi_totals = stats['first_innings_totals']

        return {
            'venue_boundary_pct': stats['total_boundaries'] / total_balls,
            'venue_dot_pct': stats['total_dots'] / total_balls,
            'venue_wicket_rate': stats['total_wickets'] / total_balls,
            'venue_powerplay_avg': (stats['powerplay_runs'] / pp_balls * 36) if pp_balls > 0 else 0.0,
            'venue_death_avg': (stats['death_runs'] / death_balls * 30) if death_balls > 0 else 0.0,
            'venue_first_innings_avg': sum(fi_totals) / len(fi_totals) if fi_totals else 0.0,
            'venue_chase_win_pct': stats['chase_wins'] / stats['matches_total'] if stats['matches_total'] > 0 else 0.5,
        }

    def update_venue_stats(self, venue: str, innings_total: int):
        """Legacy method — update venue stats with just innings total."""
        self.venue_stats[venue]['total_runs'] += innings_total
        self.venue_stats[venue]['innings_count'] += 1

    def update_venue_stats_detailed(self, venue: str, innings_data: dict):
        """
        Update venue stats with rich per-innings data.
        Called AFTER match innings is processed (not during).

        Args:
            innings_data: dict with keys: total_runs, total_balls, boundaries, dots,
                          wickets, powerplay_runs, powerplay_balls, death_runs, death_balls,
                          is_first_innings, is_chase_win (bool or None),
                          c0, c1, c2, c4, c6, cw  (schema v4 — optional; default to 0)
        """
        stats = self.venue_stats[venue]
        stats['total_runs'] += innings_data['total_runs']
        stats['innings_count'] += 1
        stats['total_balls'] += innings_data['total_balls']
        stats['total_boundaries'] += innings_data['boundaries']
        stats['total_dots'] += innings_data['dots']
        stats['total_wickets'] += innings_data['wickets']
        stats['powerplay_runs'] += innings_data['powerplay_runs']
        stats['powerplay_balls'] += innings_data['powerplay_balls']
        stats['death_runs'] += innings_data['death_runs']
        stats['death_balls'] += innings_data['death_balls']
        for ck in _OUTCOME_COUNT_KEYS:
            stats[ck] += innings_data.get(ck, 0)

        if innings_data['is_first_innings']:
            stats['first_innings_totals'].append(innings_data['total_runs'])

    def update_venue_match_result(self, venue: str, chase_won: bool):
        """Update match-level venue stats (called once per match, after both innings)."""
        stats = self.venue_stats[venue]
        stats['matches_total'] += 1
        if chase_won:
            stats['chase_wins'] += 1

    def get_all_venue_stats(self) -> dict:
        """Return copy of all venue stats for caching."""
        result = {}
        for venue, stats in self.venue_stats.items():
            s = dict(stats)
            s['first_innings_totals'] = list(stats['first_innings_totals'])
            result[venue] = s
        return result

    def get_venue_outcome_dist(self, venue: str, prior, k=200.0):
        """Return empirical-Bayes-shrunk P(0,1,2,4,6,W | venue)."""
        s = self.venue_stats[venue]
        counts = (s['c0'], s['c1'], s['c2'], s['c4'], s['c6'], s['cw'])
        p = _shrink_counts(counts, prior, k)
        return {
            'venue_p0': p[0], 'venue_p1': p[1], 'venue_p2': p[2],
            'venue_p4': p[3], 'venue_p6': p[4], 'venue_pw': p[5],
        }


def deep_copy_stats(tracker, venue_tracker=None, elo_tracker=None):
    """
    Create a deep copy of tracker stats at current state.
    This represents what we knew at this point in time for simulations.

    DESIGN DECISION: Deep copy to avoid reference issues
    REASONING: Snapshots must be immutable - changes to tracker shouldn't affect past snapshots
    """
    # Pre-sum last-5-match recent totals once per player so consumers
    # (SQLite cache, StatsProvider) don't need to carry the raw deques.
    def _batting_row(pid, stats):
        row = dict(stats)  # Copies c0..cw keys too (schema v4).
        # Conservation guard: Σ outcome-bucket == balls. An off-by-one
        # in update_stats or a schema migration that missed a bucket
        # would silently poison every distribution feature downstream.
        cs = sum(row[k] for k in _OUTCOME_COUNT_KEYS)
        if cs != row['balls']:
            raise AssertionError(
                f"batting outcome-count drift: player={pid} "
                f"balls={row['balls']} sum_c={cs} counts="
                f"{{c0:{row['c0']},c1:{row['c1']},c2:{row['c2']},"
                f"c4:{row['c4']},c6:{row['c6']},cw:{row['cw']}}}"
            )
        dq = tracker.recent_batting.get(pid)
        if dq:
            row['recent_runs'] = sum(m['runs'] for m in dq)
            row['recent_balls'] = sum(m['balls'] for m in dq)
            row['recent_dismissals'] = sum(m['dismissals'] for m in dq)
        else:
            row['recent_runs'] = 0
            row['recent_balls'] = 0
            row['recent_dismissals'] = 0
        return row

    def _bowling_row(pid, stats):
        row = dict(stats)
        cs = sum(row[k] for k in _OUTCOME_COUNT_KEYS)
        if cs != row['balls_bowled']:
            raise AssertionError(
                f"bowling outcome-count drift: player={pid} "
                f"balls_bowled={row['balls_bowled']} sum_c={cs}"
            )
        dq = tracker.recent_bowling.get(pid)
        if dq:
            row['recent_runs_given'] = sum(m['runs_given'] for m in dq)
            row['recent_balls_bowled'] = sum(m['balls_bowled'] for m in dq)
            row['recent_wickets'] = sum(m['wickets'] for m in dq)
        else:
            row['recent_runs_given'] = 0
            row['recent_balls_bowled'] = 0
            row['recent_wickets'] = 0
        return row

    snapshot = {
        'batting': {
            player_id: _batting_row(player_id, stats)
            for player_id, stats in tracker.batting_stats.items()
        },
        'bowling': {
            player_id: _bowling_row(player_id, stats)
            for player_id, stats in tracker.bowling_stats.items()
        },
        'h2h': {
            matchup: dict(stats)
            for matchup, stats in tracker.h2h_stats.items()
        },
        # NEW: Type-based stats
        'batting_vs_type': {
            player_id: {k: dict(v) for k, v in stats.items()}
            for player_id, stats in tracker.batting_vs_type.items()
        },
        'bowling_vs_hand': {
            player_id: {k: dict(v) for k, v in stats.items()}
            for player_id, stats in tracker.bowling_vs_hand.items()
        },
    }
    if tracker.enable_i8:
        snapshot['batting_phase'] = {
            player_id: {
                phase: dict(counts)
                for phase, counts in by_phase.items()
            }
            for player_id, by_phase in tracker.batting_phase.items()
        }
        snapshot['bowling_phase'] = {
            player_id: {
                phase: dict(counts)
                for phase, counts in by_phase.items()
            }
            for player_id, by_phase in tracker.bowling_phase.items()
        }

    # Include venue stats if tracker provided
    if venue_tracker is not None:
        snapshot['venue'] = venue_tracker.get_all_venue_stats()

    # Include ELO ratings if tracker provided
    if elo_tracker is not None:
        snapshot['batting_elo'] = dict(elo_tracker.batting_elo)
        snapshot['bowling_elo'] = dict(elo_tracker.bowling_elo)

    return snapshot


class InningsFeatureCalculator:
    """
    DESIGN DECISION: Separate class for innings-level features that reset each innings.
    REASONING: Clear separation between match-level and innings-level state.
    Makes it obvious which features reset and which persist.
    """
    def __init__(self):
        # Recent ball tracking for momentum features
        self.last_5_balls = deque(maxlen=5)
        self.last_10_balls = deque(maxlen=10)
        self.last_30_balls = deque(maxlen=30)
        self.balls_since_boundary = 0

        # NEW: Per-batter tracking (resets each innings)
        self.batter_balls_faced = defaultdict(int)
        self.batter_runs_scored = defaultdict(int)

        # NEW: Partnership tracking (resets on wicket)
        self.partnership_runs = 0

        # NEW: Per-bowler tracking (resets each innings)
        self.bowler_balls_bowled = defaultdict(int)

    def update_ball_history(self, runs, is_boundary, is_wicket=False,
                            batter_id=None, bowler_id=None, *,
                            batter_runs=None, is_legal=True):
        """Update innings state after a delivery.

        ``runs`` is the team-score change and feeds momentum/partnership
        features. ``batter_runs`` is the off-the-bat value and feeds the
        batter card. Wides/no-balls remain in delivery-history momentum but
        do not increment balls faced or bowler balls.
        """
        if batter_runs is None:
            batter_runs = runs
        self.last_5_balls.append(runs)
        self.last_10_balls.append(runs)
        self.last_30_balls.append(runs)

        if is_boundary:
            self.balls_since_boundary = 0
        else:
            self.balls_since_boundary += 1

        # NEW: Update per-batter stats
        if batter_id is not None:
            if is_legal:
                self.batter_balls_faced[batter_id] += 1
            self.batter_runs_scored[batter_id] += batter_runs

        # NEW: Update per-bowler stats
        if bowler_id is not None and is_legal:
            self.bowler_balls_bowled[bowler_id] += 1

        # NEW: Update partnership
        self.partnership_runs += runs
        if is_wicket:
            self.partnership_runs = 0  # Reset on wicket

    def get_batter_innings_stats(self, batter_id):
        """Get current batter's innings stats (balls faced, runs scored)"""
        return {
            'batter_balls_faced': self.batter_balls_faced.get(batter_id, 0),
            'batter_runs_scored': self.batter_runs_scored.get(batter_id, 0),
        }

    def get_non_striker_sr(self, non_striker_id):
        """
        Get non-striker's strike rate in this innings.
        Returns 0 if non-striker hasn't faced any balls yet.
        """
        balls = self.batter_balls_faced.get(non_striker_id, 0)
        runs = self.batter_runs_scored.get(non_striker_id, 0)
        if balls == 0:
            return 0.0
        return (runs / balls) * 100

    def get_bowler_innings_stats(self, bowler_id):
        """Get current bowler's innings stats (balls/overs bowled)"""
        balls = self.bowler_balls_bowled.get(bowler_id, 0)
        return {
            'bowler_balls_in_innings': balls,
            'bowler_overs_in_innings': balls / 6,  # Fractional overs
        }

    def get_partnership_features(self):
        """Get partnership-related features"""
        return {
            'partnership_runs': self.partnership_runs,
        }

    def get_momentum_features(self):
        """Calculate all momentum-based features"""
        # DESIGN DECISION: Return 0 for insufficient history rather than None
        # REASONING: Allows model to learn from early-ball situations too
        return {
            'last_5_balls_runs': sum(self.last_5_balls),
            'last_10_balls_runs': sum(self.last_10_balls),
            'last_30_balls_runs': sum(self.last_30_balls),
            'balls_since_boundary': self.balls_since_boundary,
            # Dot ball pressure
            'last_10_dots': sum(1 for r in self.last_10_balls if r == 0),
        }


def normalize_ball_outcome(runs, is_wicket):
    """
    DESIGN DECISION: Normalize rare run outcomes to reduce class imbalance.
    REASONING: 3,5,7+ runs are very rare and hurt model performance.
    """
    if is_wicket:
        return -1  # Keep wickets as -1
    
    # Normalize rare run values
    if runs == 3:
        return 2
    elif runs == 5:
        return 4
    elif runs >= 7:
        return 6
    else:
        return runs  # 0,1,2,4,6 stay the same


def extract_delivery_semantics(delivery, player_registry):
    """Return cricket-correct run, legality, boundary, and wicket semantics.

    Cricsheet exposes team, batter, and extras runs separately. Keeping those
    channels separate is essential: byes/leg-byes affect the score but not
    the batter or bowler figures, while wides/no-balls are not legal balls.
    """
    run_data = delivery.get('runs', {})
    team_runs = int(run_data.get('total', 0))
    batter_runs = int(run_data.get('batter', 0))
    extras_runs = int(run_data.get('extras', team_runs - batter_runs))
    if team_runs != batter_runs + extras_runs:
        raise ValueError(
            "delivery run conservation failed: "
            f"total={team_runs} batter={batter_runs} extras={extras_runs}"
        )

    extras = {
        str(kind): int(value)
        for kind, value in delivery.get('extras', {}).items()
    }
    is_wide = extras.get('wides', 0) > 0
    is_noball = extras.get('noballs', 0) > 0
    is_legal = not (is_wide or is_noball)
    non_bowler_extras = (
        extras.get('byes', 0)
        + extras.get('legbyes', 0)
        + extras.get('penalty', 0)
    )
    bowler_runs = team_runs - non_bowler_extras

    wickets = delivery.get('wickets', [])
    dismissed_ids = tuple(
        player_registry.get(wicket.get('player_out'), wicket.get('player_out'))
        for wicket in wickets
        if wicket.get('player_out') is not None
    )
    wicket_kinds = tuple(str(wicket.get('kind', 'unknown')) for wicket in wickets)
    is_bowler_wicket = any(kind in _BOWLER_WICKET_KINDS for kind in wicket_kinds)

    return {
        'team_runs': team_runs,
        'batter_runs': batter_runs,
        'extras_runs': extras_runs,
        'bowler_runs': bowler_runs,
        'extras_types': tuple(sorted(extras)),
        'wide_runs': extras.get('wides', 0),
        'noball_runs': extras.get('noballs', 0),
        'bye_runs': extras.get('byes', 0),
        'legbye_runs': extras.get('legbyes', 0),
        'penalty_runs': extras.get('penalty', 0),
        'is_wide': is_wide,
        'is_noball': is_noball,
        'is_legal': is_legal,
        'is_boundary': (
            batter_runs in (4, 6) and not bool(run_data.get('non_boundary', False))
        ),
        'is_wicket': bool(wickets),
        'wicket_kinds': wicket_kinds,
        'dismissed_batter_ids': dismissed_ids,
        'is_bowler_wicket': is_bowler_wicket,
    }


def extract_raw_state(delivery, player_registry, score, wickets, balls):
    """
    DESIGN DECISION: Pure function that extracts raw state from delivery.
    REASONING: Separation of concerns - parsing vs feature engineering.
    This function only extracts what's directly in the data.
    """
    batter = delivery['batter']
    non_striker = delivery['non_striker']
    bowler = delivery['bowler']
    delivery_state = extract_delivery_semantics(delivery, player_registry)
    
    # Player IDs from registry
    batter_id = player_registry[batter]
    non_striker_id = player_registry[non_striker]
    bowler_id = player_registry[bowler]
    
    return {
        'batter_id': batter_id,
        'non_striker_id': non_striker_id,
        'bowler_id': bowler_id,
        # Backwards-compatible alias: ``runs`` remains the team-score change.
        'runs': delivery_state['team_runs'],
        **delivery_state,
        'score': score,
        'wickets': wickets,
        'balls_bowled': balls,
    }


def calculate_basic_features(state):
    """
    DESIGN DECISION: Separate function for stateless features.
    REASONING: These can be calculated independently without any history.
    Easy to test and reason about.
    """
    features = {}

    # Run rate and required rate
    overs = state['balls_bowled'] / 6
    features['run_rate'] = state['score'] / max(overs, 0.1)  # Avoid division by zero

    # Resource percentages
    features['wickets_ratio'] = state['wickets'] / 10
    features['balls_ratio'] = state['balls_bowled'] / 120  # T20 format
    features['wickets_in_hand'] = 10 - state['wickets']
    features['balls_remaining'] = 120 - state['balls_bowled']  # NEW: Explicit balls remaining

    # Match phase indicators
    # DESIGN DECISION: Use simple binary flags for phases
    # REASONING: Easier for tree-based models than continuous over count
    features['is_powerplay'] = state['balls_bowled'] < 36
    features['is_middle_overs'] = 36 <= state['balls_bowled'] < 96
    features['is_death_overs'] = state['balls_bowled'] >= 96

    # Current over progress
    features['balls_in_over'] = state['balls_bowled'] % 6

    return features


def calculate_pressure_features(state, innings_calc):
    """
    DESIGN DECISION: Separate pressure indicators as they're conceptually related.
    REASONING: Groups related features, makes it easy to experiment with adding/removing
    the entire pressure feature set.
    """
    features = {}
    
    momentum = innings_calc.get_momentum_features()
    
    # Dot ball percentage in recent balls
    if len(innings_calc.last_30_balls) > 0:
        features['dot_percentage_recent'] = momentum['last_10_dots'] / min(len(innings_calc.last_10_balls), 10)
    else:
        features['dot_percentage_recent'] = 0
    
    # Boundary percentage
    boundaries_recent = sum(1 for r in innings_calc.last_30_balls if r >= 4)
    if len(innings_calc.last_30_balls) > 0:
        features['boundary_percentage_recent'] = boundaries_recent / len(innings_calc.last_30_balls)
    else:
        features['boundary_percentage_recent'] = 0
    
    return features


def parse_match_data_v2(json_data, player_stats_tracker, venue_tracker=None,
                        player_metadata=None, elo_tracker=None, match_k_factor=None,
                        prior=None, phase_priors=None,
                        k_player=30.0, k_venue=200.0, match_ref=None,
                        delivery_semantics=LEGACY_DELIVERY_SEMANTICS):
    """
    DESIGN DECISION: Pass tracker as parameter rather than global.
    REASONING: Makes dependencies explicit, easier to test, allows multiple trackers
    for different experiments (e.g., one with h2h, one without).

    Args:
        json_data: Raw JSON string of match data
        player_stats_tracker: PlayerStatsTracker for player statistics
        venue_tracker: Optional VenueStatsTracker for venue statistics
        player_metadata: Optional PlayerMetadataProvider for player attributes
        prior: Optional 6-tuple (p0,p1,p2,p4,p6,pw) — global empirical prior
               for outcome-distribution features (schema v4). If None, all
               42 dist features are emitted as 0.0 (legacy pre-v4 behavior).
        phase_priors: Optional dict
               {'powerplay'|'middle'|'death': (p0,p1,p2,p4,p6,pw)}. When
               provided, 6 phase_p{0,1,2,4,6,w} features are emitted per
               ball, dispatched by pre-ball phase. Phase 3 of the
               outcome-dist follow-up plan; loaded from SQLite _meta.
        match_ref: Optional stable match identifier (cricsheet filename
               stem) used as the innings_id suffix. B2 fix (2026-07-16):
               the legacy `hash(json_data) % 100000` suffix was salted per
               process (irreproducible across runs) and collision-prone,
               blocking parquet↔cricsheet joins. Callers that don't emit
               ball rows for joining may omit it (legacy hash fallback).
        delivery_semantics: Versioned label/state contract. The legacy mode
               preserves the deployed model's inclusive-total-run behavior;
               ``legal_off_bat_v1`` enables the isolated I5 rebuild.

    Returns:
        Tuple of (all_balls list, innings_totals list for venue update)
    """
    if delivery_semantics not in DELIVERY_SEMANTICS:
        raise ValueError(
            f"unsupported delivery semantics {delivery_semantics!r}; "
            f"expected one of {sorted(DELIVERY_SEMANTICS)}"
        )
    use_i5_semantics = delivery_semantics == I5_DELIVERY_SEMANTICS

    data = json.loads(json_data)
    match_key = match_ref if match_ref is not None else hash(json_data) % 100000
    player_registry = data['info']['registry']['people']

    # DESIGN DECISION: Store match metadata for potential venue/team features
    # REASONING: Might want venue-specific features later
    match_info = {
        'venue': canonicalize_venue(data['info'].get('venue')),
        'date': data['info']['dates'][0] if 'dates' in data['info'] else None,
        'teams': data['info'].get('teams', []),
        'toss_winner': data['info'].get('toss', {}).get('winner', 'unknown'),
        'toss_decision': data['info'].get('toss', {}).get('decision', 'unknown')
    }

    # Parse match date for age calculation
    match_date = None
    if match_info['date']:
        try:
            match_date = datetime.strptime(match_info['date'], '%Y-%m-%d')
        except:
            pass

    # NEW: Get venue average BEFORE this match (temporal integrity)
    # This ensures we only use historical data, no lookahead bias
    venue_avg_score = 0.0
    venue_profile = {
        'venue_boundary_pct': 0.0, 'venue_dot_pct': 0.0, 'venue_wicket_rate': 0.0,
        'venue_powerplay_avg': 0.0, 'venue_death_avg': 0.0,
        'venue_first_innings_avg': 0.0, 'venue_chase_win_pct': 0.5,
    }
    if venue_tracker is not None:
        venue_avg_score = venue_tracker.get_venue_avg_score(match_info['venue'])
        venue_profile = venue_tracker.get_venue_profile(match_info['venue'])

    # Schema v4: venue outcome distribution (constant across the match).
    if prior is not None and venue_tracker is not None:
        venue_dist = venue_tracker.get_venue_outcome_dist(
            match_info['venue'], prior, k=k_venue
        )
    else:
        venue_dist = _ZERO_VENUE_DIST

    # Match context features (constant for all balls in this match)
    event_info = data['info'].get('event', {})
    event_name = event_info.get('name', '') if isinstance(event_info, dict) else ''
    team_type = data['info'].get('team_type', 'unknown')
    match_context = classify_match_context(event_name, team_type, match_info['teams'])
    chose_to_bat = 1 if match_info['toss_decision'] == 'bat' else 0

    player_stats_tracker.start_match()

    # Compute team-level features BEFORE the match (constant for all balls)
    # Resolve lineup player IDs for both teams
    team_features_by_team = {}  # team_name -> {features}
    teams = match_info['teams']
    for team_name in teams:
        lineup_names = data['info'].get('players', {}).get(team_name, [])
        lineup_ids = [player_registry.get(name, name) for name in lineup_names]

        # Team ELO (sum of individual ELOs)
        if elo_tracker is not None:
            t_bat_elo = elo_tracker.get_team_batting_elo(lineup_ids)
            t_bowl_elo = elo_tracker.get_team_bowling_elo(lineup_ids)
        else:
            t_bat_elo = PlayerEloTracker.DEFAULT_ELO * len(lineup_ids)
            t_bowl_elo = PlayerEloTracker.DEFAULT_ELO * len(lineup_ids)

        # Aggregated player stats (from historical cache)
        bat_avgs, bat_srs = [], []
        bowl_avgs, bowl_econs = [], []
        for pid in lineup_ids:
            bstats = player_stats_tracker.get_batting_features(pid)
            if bstats['batsman_avg'] > 0:
                bat_avgs.append(bstats['batsman_avg'])
                bat_srs.append(bstats['batsman_sr'])
            bwstats = player_stats_tracker.get_bowling_features(pid)
            if bwstats['bowler_avg'] > 0:
                bowl_avgs.append(bwstats['bowler_avg'])
                bowl_econs.append(bwstats['bowler_econ'])

        team_features_by_team[team_name] = {
            'team_batting_elo': t_bat_elo,
            'team_bowling_elo': t_bowl_elo,
            'lineup_ids': lineup_ids,
            'team_batting_avg': sum(bat_avgs) / len(bat_avgs) if bat_avgs else 0.0,
            'team_batting_sr': sum(bat_srs) / len(bat_srs) if bat_srs else 0.0,
            'team_bowling_avg': sum(bowl_avgs) / len(bowl_avgs) if bowl_avgs else 0.0,
            'team_bowling_econ': sum(bowl_econs) / len(bowl_econs) if bowl_econs else 0.0,
        }

    all_balls = []
    innings_totals = []  # Track innings totals for venue stats update
    innings_details = []  # Rich per-innings data for venue profile

    # NEW: Track first innings score for chase features
    first_innings_score = 0

    for inning_idx, inning in enumerate(data['innings'], 1):
        score = 0
        wickets = 0
        balls = 0

        # Per-innings accumulator for venue profile stats
        inn_agg = {
            'boundaries': 0, 'dots': 0, 'wickets': 0,
            'total_balls': 0, 'total_runs': 0,
            'powerplay_runs': 0, 'powerplay_balls': 0,
            'death_runs': 0, 'death_balls': 0,
            # Schema v4: 6-class outcome counts for venue aggregation.
            'c0': 0, 'c1': 0, 'c2': 0, 'c4': 0, 'c6': 0, 'cw': 0,
            # Phase 3: phase-split outcome counts for per-phase prior
            # accumulation. Boundaries: pre-ball balls_bowled < 36 → PP,
            # 36..<96 → middle, ≥96 → death (matches
            # calculate_basic_features). Σ over all phases ≡ Σ cX.
            'c0_powerplay': 0, 'c1_powerplay': 0, 'c2_powerplay': 0,
            'c4_powerplay': 0, 'c6_powerplay': 0, 'cw_powerplay': 0,
            'c0_middle': 0, 'c1_middle': 0, 'c2_middle': 0,
            'c4_middle': 0, 'c6_middle': 0, 'cw_middle': 0,
            'c0_death': 0, 'c1_death': 0, 'c2_death': 0,
            'c4_death': 0, 'c6_death': 0, 'cw_death': 0,
        }

        # NEW: Calculate target for 2nd innings
        target = first_innings_score + 1 if inning_idx == 2 else 0

        # Resolve batting/bowling team for this innings
        batting_team_name = inning.get('team', teams[inning_idx - 1] if inning_idx <= len(teams) else 'unknown')
        bowling_team_name = [t for t in teams if t != batting_team_name][0] if len(teams) == 2 else 'unknown'

        # Get team-level features for this innings
        bat_team_feats = team_features_by_team.get(batting_team_name, {})
        bowl_team_feats = team_features_by_team.get(bowling_team_name, {})
        batting_team_elo = bat_team_feats.get('team_batting_elo', PlayerEloTracker.DEFAULT_ELO * 11)
        bowling_team_elo = bowl_team_feats.get('team_bowling_elo', PlayerEloTracker.DEFAULT_ELO * 11)
        elo_diff = batting_team_elo - bowling_team_elo
        team_batting_avg = bat_team_feats.get('team_batting_avg', 0.0)
        team_batting_sr = bat_team_feats.get('team_batting_sr', 0.0)
        team_bowling_avg = bowl_team_feats.get('team_bowling_avg', 0.0)
        team_bowling_econ = bowl_team_feats.get('team_bowling_econ', 0.0)

        # DESIGN DECISION: Reset innings calculator per innings
        # REASONING: Momentum features should not carry over between innings
        innings_calc = InningsFeatureCalculator()

        for over_idx, over in enumerate(inning['overs']):
            for delivery in over['deliveries']:
                # Extract raw state
                state = extract_raw_state(delivery, player_registry, score, wickets, balls)

                # Get player statistics BEFORE this ball
                # DESIGN DECISION: Features reflect state BEFORE the ball
                # REASONING: This is what the model would know when predicting
                batting_features = player_stats_tracker.get_batting_features(state['batter_id'])
                bowling_features = player_stats_tracker.get_bowling_features(state['bowler_id'])
                h2h_features = player_stats_tracker.get_h2h_features(state['batter_id'], state['bowler_id'])

                # NEW: Get type-based stats (BEFORE this ball)
                batting_vs_type_features = player_stats_tracker.get_batting_vs_type_features(state['batter_id'])
                bowling_vs_hand_features = player_stats_tracker.get_bowling_vs_hand_features(state['bowler_id'])

                # Schema v4: empirical outcome distributions, pre-ball state.
                if prior is not None:
                    batter_dist = player_stats_tracker.get_batter_outcome_dist(
                        state['batter_id'], prior, k=k_player)
                    bowler_dist = player_stats_tracker.get_bowler_outcome_dist(
                        state['bowler_id'], prior, k=k_player)
                    batter_vs_type_dist = player_stats_tracker.get_batter_vs_type_outcome_dist(
                        state['batter_id'], prior, k=k_player)
                    bowler_vs_hand_dist = player_stats_tracker.get_bowler_vs_hand_outcome_dist(
                        state['bowler_id'], prior, k=k_player)
                else:
                    batter_dist = _ZERO_BATTER_DIST
                    bowler_dist = _ZERO_BOWLER_DIST
                    batter_vs_type_dist = _ZERO_BATTER_VS_TYPE_DIST
                    bowler_vs_hand_dist = _ZERO_BOWLER_VS_HAND_DIST

                # Phase 3: phase prior (6 features, dispatched by ball phase).
                phase_dist = _phase_dist_from_priors(
                    phase_priors, state['balls_bowled'])

                # Calculate all features
                basic_features = calculate_basic_features(state)
                momentum_features = innings_calc.get_momentum_features()
                pressure_features = calculate_pressure_features(state, innings_calc)

                # NEW: Get per-batter and per-bowler innings stats (BEFORE this ball)
                batter_innings_stats = innings_calc.get_batter_innings_stats(state['batter_id'])
                bowler_innings_stats = innings_calc.get_bowler_innings_stats(state['bowler_id'])
                partnership_features = innings_calc.get_partnership_features()

                # NEW: Get non-striker's strike rate (BEFORE this ball)
                non_striker_sr = innings_calc.get_non_striker_sr(state['non_striker_id'])

                # NEW: Get player metadata features (hand, arm, type, age)
                if player_metadata is not None:
                    batter_meta = player_metadata.get_player_metadata(state['batter_id'])
                    bowler_meta = player_metadata.get_player_metadata(state['bowler_id'])

                    # Tier 1: Direct features
                    batter_hand = batter_meta['batter_hand']
                    bowler_arm = bowler_meta['bowler_arm']
                    is_pace = bowler_meta['is_pace']
                    bowling_type = bowler_meta['bowling_type']

                    # Ages
                    batter_age = player_metadata.get_player_age(state['batter_id'], match_date) if match_date else None
                    bowler_age = player_metadata.get_player_age(state['bowler_id'], match_date) if match_date else None

                    # Tier 2: Matchup features
                    matchup_type = player_metadata.get_matchup_type(state['batter_id'], state['bowler_id'])
                    spin_matchup_advantage = player_metadata.get_spin_matchup_advantage(state['batter_id'], state['bowler_id'])
                    same_arm_matchup = player_metadata.get_same_arm_matchup(state['batter_id'], state['bowler_id'])
                else:
                    # Defaults if no metadata provider
                    batter_hand = 'unknown'
                    bowler_arm = 'unknown'
                    is_pace = None
                    bowling_type = 'unknown'
                    batter_age = None
                    bowler_age = None
                    matchup_type = 'UNK_vs_unknown'
                    spin_matchup_advantage = 0
                    same_arm_matchup = None

                # NEW: Calculate chase features (2nd innings only)
                balls_remaining = 120 - state['balls_bowled']
                if inning_idx == 2 and target > 0:
                    runs_needed = target - score
                    run_rate_required = (runs_needed * 6 / balls_remaining) if balls_remaining > 0 else 0
                    lead_gap = -runs_needed  # Negative means chasing team is behind
                else:
                    run_rate_required = 0
                    lead_gap = score  # First innings: just the current score

                # NEW: Calculate pressure_cooker_index (RRR / wickets_remaining)
                wickets_remaining = 10 - state['wickets']
                if inning_idx == 2 and wickets_remaining > 0 and run_rate_required > 0:
                    pressure_cooker_index = run_rate_required / wickets_remaining
                else:
                    pressure_cooker_index = 0

                # Combine all features
                # DESIGN DECISION: Flatten all features into single dict
                # REASONING: Simpler for DataFrame creation and model training
                ball_record = {
                    'innings_id': f"{inning_idx}_{match_key}",
                    'inning_idx': inning_idx,
                    'over_idx': over_idx,
                    'ball_idx': balls,
                    'match_date': match_date.strftime('%Y-%m-%d') if match_date else None,
                    # Raw state
                    **state,
                    # Computed features
                    **basic_features,
                    **batting_features,
                    **bowling_features,
                    **h2h_features,
                    **momentum_features,
                    **pressure_features,
                    # NEW: Per-batter/bowler innings stats
                    **batter_innings_stats,
                    **bowler_innings_stats,
                    **partnership_features,
                    # NEW: Type-based stats (Tier 3)
                    **batting_vs_type_features,
                    **bowling_vs_hand_features,
                    # Schema v4: empirical outcome distributions.
                    **batter_dist,
                    **bowler_dist,
                    **batter_vs_type_dist,
                    **bowler_vs_hand_dist,
                    **venue_dist,
                    # Phase 3: phase prior (6 features for this ball's phase).
                    **phase_dist,
                    # NEW: Chase features
                    'chase_target': target,  # Renamed from 'target' to avoid collision with prediction target
                    'run_rate_required': run_rate_required,
                    'lead_gap': lead_gap,
                    'pressure_cooker_index': pressure_cooker_index,
                    # NEW: Medium features
                    'venue_avg_score': venue_avg_score,  # Historical venue average (no lookahead)
                    'non_striker_sr': non_striker_sr,  # Non-striker's strike rate this innings
                    # NEW: Player metadata features (Tier 1)
                    'batter_hand': encode_batter_hand(batter_hand),
                    'bowler_arm': encode_bowler_arm(bowler_arm),
                    'is_pace': encode_is_pace(is_pace),
                    'bowling_type': encode_bowling_type(bowling_type),
                    'batter_age': batter_age if batter_age is not None else 0,
                    'bowler_age': bowler_age if bowler_age is not None else 0,
                    # NEW: Matchup features (Tier 2)
                    'matchup_type': matchup_type,  # Will be encoded later
                    'spin_matchup_advantage': spin_matchup_advantage,
                    'same_arm_matchup': 1 if same_arm_matchup else (0 if same_arm_matchup is False else -1),
                    # Team strength features (ELO + aggregated stats)
                    'striker_elo': elo_tracker.get_batting_elo(state['batter_id']) if elo_tracker else PlayerEloTracker.DEFAULT_ELO,
                    'bowler_elo_rating': elo_tracker.get_bowling_elo(state['bowler_id']) if elo_tracker else PlayerEloTracker.DEFAULT_ELO,
                    'batting_team_elo': batting_team_elo,
                    'bowling_team_elo': bowling_team_elo,
                    'elo_diff': elo_diff,
                    'team_batting_avg': team_batting_avg,
                    'team_batting_sr': team_batting_sr,
                    'team_bowling_avg': team_bowling_avg,
                    'team_bowling_econ': team_bowling_econ,
                    # I5 uses the legal-delivery, off-the-bat outcome. Raw
                    # threes remain available as batter_runs even though the
                    # current six-class target groups 3→2. Legacy mode keeps
                    # the deployed inclusive-total-run target.
                    'ball_outcome': normalize_ball_outcome(
                        (state['batter_runs'] if use_i5_semantics
                         else state['team_runs']),
                        state['is_wicket']),

                    # Match Context Features
                    'venue': match_info['venue'],
                    'is_toss_winner': 1 if match_info['toss_winner'] == data['innings'][inning_idx-1]['team'] else 0,
                    'is_batting_first': 1 if inning_idx == 1 else 0,
                    # Venue profile features (historical, no lookahead)
                    **venue_profile,
                    # Match context features
                    'chose_to_bat': chose_to_bat,
                    **match_context,
                }

                # I5: the model represents legal-delivery outcomes. Wides and
                # no-balls update match/player state below but are modeled by
                # the simulator's separate extras process.
                if state['is_legal'] or not use_i5_semantics:
                    all_balls.append(ball_record)

                # Update states AFTER recording the ball
                # DESIGN DECISION: Update after recording
                # REASONING: Features should reflect pre-ball state
                if use_i5_semantics:
                    player_stats_tracker.update_stats(
                        state['batter_id'],
                        state['bowler_id'],
                        state['batter_runs'],
                        state['is_wicket'],
                        batter_hand=(
                            batter_hand if batter_hand != 'unknown' else None
                        ),
                        is_pace=is_pace,
                        bowler_runs=state['bowler_runs'],
                        is_legal=state['is_legal'],
                        dismissed_batter_id=(
                            state['dismissed_batter_ids'][0]
                            if state['dismissed_batter_ids'] else None
                        ),
                        is_bowler_wicket=state['is_bowler_wicket'],
                        phase=_classify_phase_pre_ball(
                            state['balls_bowled']),
                    )
                else:
                    player_stats_tracker.update_stats(
                        state['batter_id'],
                        state['bowler_id'],
                        state['team_runs'],
                        state['is_wicket'],
                        batter_hand=(
                            batter_hand if batter_hand != 'unknown' else None
                        ),
                        is_pace=is_pace,
                        phase=_classify_phase_pre_ball(
                            state['balls_bowled']),
                    )

                # ELO describes the legal batter-vs-bowler outcome channel.
                if elo_tracker is not None and (
                    state['is_legal'] or not use_i5_semantics
                ):
                    elo_tracker.update(
                        state['batter_id'],
                        state['bowler_id'],
                        (state['batter_runs'] if use_i5_semantics
                         else state['team_runs']),
                        state['is_wicket'],
                        k_factor=match_k_factor
                    )

                # NEW: Pass batter/bowler IDs and wicket status for tracking
                if use_i5_semantics:
                    innings_calc.update_ball_history(
                        state['team_runs'],
                        is_boundary=state['is_boundary'],
                        is_wicket=state['is_wicket'],
                        batter_id=state['batter_id'],
                        bowler_id=state['bowler_id'],
                        batter_runs=state['batter_runs'],
                        is_legal=state['is_legal'],
                    )
                else:
                    innings_calc.update_ball_history(
                        state['team_runs'],
                        is_boundary=(state['team_runs'] >= 4),
                        is_wicket=state['is_wicket'],
                        batter_id=state['batter_id'],
                        bowler_id=state['bowler_id'],
                    )

                # Update match state
                if state['is_wicket']:
                    wickets += 1
                score += state['team_runs']
                if state['is_legal']:
                    balls += 1

                # Accumulate per-innings stats for venue profile.
                if use_i5_semantics:
                    # All score changes count toward venue/phase run rates;
                    # only legal balls count in denominators/outcome dists.
                    inn_agg['total_runs'] += state['team_runs']
                    if state['balls_bowled'] < 36:
                        inn_agg['powerplay_runs'] += state['team_runs']
                    if state['balls_bowled'] >= 90:
                        inn_agg['death_runs'] += state['team_runs']

                    if state['is_legal']:
                        inn_agg['total_balls'] += 1
                        if (state['team_runs'] == 0
                                and not state['is_wicket']):
                            inn_agg['dots'] += 1
                        if state['is_boundary'] and not state['is_wicket']:
                            inn_agg['boundaries'] += 1
                        if state['is_wicket']:
                            inn_agg['wickets'] += 1
                        bucket = _outcome_bucket_key(
                            state['batter_runs'], state['is_wicket'])
                        inn_agg[bucket] += 1
                        pre_phase = _classify_phase_pre_ball(
                            state['balls_bowled'])
                        inn_agg[f"{bucket}_{pre_phase}"] += 1
                        if state['balls_bowled'] < 36:
                            inn_agg['powerplay_balls'] += 1
                        if state['balls_bowled'] >= 90:
                            inn_agg['death_balls'] += 1
                elif state['is_legal']:
                    # Exact deployed legacy accounting.
                    inn_agg['total_balls'] += 1
                    inn_agg['total_runs'] += state['team_runs']
                    if state['team_runs'] == 0 and not state['is_wicket']:
                        inn_agg['dots'] += 1
                    if state['team_runs'] >= 4 and not state['is_wicket']:
                        inn_agg['boundaries'] += 1
                    if state['is_wicket']:
                        inn_agg['wickets'] += 1
                    bucket = _outcome_bucket_key(
                        state['team_runs'], state['is_wicket'])
                    inn_agg[bucket] += 1
                    pre_phase = _classify_phase_pre_ball(
                        state['balls_bowled'])
                    inn_agg[f"{bucket}_{pre_phase}"] += 1
                    if balls <= 36:
                        inn_agg['powerplay_runs'] += state['team_runs']
                        inn_agg['powerplay_balls'] += 1
                    if balls > 90:
                        inn_agg['death_runs'] += state['team_runs']
                        inn_agg['death_balls'] += 1

        # NEW: Store first innings score for chase calculation
        if inning_idx == 1:
            first_innings_score = score

        # Track innings total for venue stats update
        innings_totals.append(score)
        innings_details.append({
            **inn_agg,
            'is_first_innings': inning_idx == 1,
        })

    # Determine chase outcome for venue stats
    chase_won = None
    if len(innings_totals) == 2:
        outcome = data['info'].get('outcome', {})
        winner = outcome.get('winner', None)
        if winner and len(data['innings']) == 2:
            batting_second_team = data['innings'][1].get('team', '')
            chase_won = (winner == batting_second_team)

    player_stats_tracker.end_match()

    return all_balls, innings_totals, match_info['venue'], innings_details, chase_won
