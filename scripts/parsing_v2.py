# parse for xgboost model.py
import json
import numpy as np
from pathlib import Path
import pandas as pd
from collections import defaultdict, deque
from datetime import datetime
import pickle

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


class PlayerStatsTracker:
    """
    DESIGN DECISION: Separate class for player stats to maintain state across matches.
    REASONING: Encapsulation - keeps complex state management isolated from parsing logic.
    This makes it easy to add new stats without cluttering the main parsing code.
    """
    def __init__(self):
        # Career stats - accumulate over time
        self.batting_stats = defaultdict(lambda: {'runs': 0, 'balls': 0, 'dismissals': 0})
        self.bowling_stats = defaultdict(lambda: {'runs_given': 0, 'balls_bowled': 0, 'wickets': 0})

        # NEW: Type-based batting stats (batter_id -> {pace/spin -> stats})
        # Tracks batter performance against pace vs spin bowlers
        self.batting_vs_type = defaultdict(lambda: {
            'pace': {'runs': 0, 'balls': 0, 'dismissals': 0},
            'spin': {'runs': 0, 'balls': 0, 'dismissals': 0},
        })

        # NEW: Type-based bowling stats (bowler_id -> {lhb/rhb -> stats})
        # Tracks bowler performance against left vs right hand batters
        self.bowling_vs_hand = defaultdict(lambda: {
            'left': {'runs_given': 0, 'balls_bowled': 0, 'wickets': 0},
            'right': {'runs_given': 0, 'balls_bowled': 0, 'wickets': 0},
        })

        # Head-to-head records
        # DESIGN DECISION: Use tuple (batter, bowler) as key for h2h
        # REASONING: Direct lookup, memory efficient, naturally handles bidirectional relationships
        self.h2h_stats = defaultdict(lambda: {'runs': 0, 'balls': 0, 'dismissals': 0})

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
            self.recent_batting[player_id].append(stats.copy())

        for player_id, stats in self.current_match_bowling.items():
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
                     batter_hand=None, is_pace=None):
        """
        Update all statistics after a ball.

        Args:
            batter_id: Batter's cricsheet ID
            bowler_id: Bowler's cricsheet ID
            runs: Runs scored off the ball
            is_wicket: Whether a wicket fell
            batter_hand: 'left', 'right', or None (for type-based bowling stats)
            is_pace: True/False/None (for type-based batting stats)
        """
        # Update batting stats
        self.batting_stats[batter_id]['runs'] += runs
        self.batting_stats[batter_id]['balls'] += 1
        if is_wicket:
            self.batting_stats[batter_id]['dismissals'] += 1

        # Update bowling stats
        self.bowling_stats[bowler_id]['runs_given'] += runs
        self.bowling_stats[bowler_id]['balls_bowled'] += 1
        if is_wicket:
            self.bowling_stats[bowler_id]['wickets'] += 1

        # Update h2h
        self.h2h_stats[(batter_id, bowler_id)]['runs'] += runs
        self.h2h_stats[(batter_id, bowler_id)]['balls'] += 1
        if is_wicket:
            self.h2h_stats[(batter_id, bowler_id)]['dismissals'] += 1

        # NEW: Update type-based batting stats (batter vs pace/spin)
        if is_pace is not None:
            type_key = 'pace' if is_pace else 'spin'
            self.batting_vs_type[batter_id][type_key]['runs'] += runs
            self.batting_vs_type[batter_id][type_key]['balls'] += 1
            if is_wicket:
                self.batting_vs_type[batter_id][type_key]['dismissals'] += 1

        # NEW: Update hand-based bowling stats (bowler vs LHB/RHB)
        if batter_hand in ('left', 'right'):
            self.bowling_vs_hand[bowler_id][batter_hand]['runs_given'] += runs
            self.bowling_vs_hand[bowler_id][batter_hand]['balls_bowled'] += 1
            if is_wicket:
                self.bowling_vs_hand[bowler_id][batter_hand]['wickets'] += 1

        # Update current match stats
        self.current_match_batting[batter_id]['runs'] += runs
        self.current_match_batting[batter_id]['balls'] += 1
        if is_wicket:
            self.current_match_batting[batter_id]['dismissals'] += 1

        self.current_match_bowling[bowler_id]['runs_given'] += runs
        self.current_match_bowling[bowler_id]['balls_bowled'] += 1
        if is_wicket:
            self.current_match_bowling[bowler_id]['wickets'] += 1


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
                          is_first_innings, is_chase_win (bool or None)
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
        row = dict(stats)
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

    def update_ball_history(self, runs, is_boundary, is_wicket=False, batter_id=None, bowler_id=None):
        """Update rolling windows after each ball"""
        self.last_5_balls.append(runs)
        self.last_10_balls.append(runs)
        self.last_30_balls.append(runs)

        if is_boundary:
            self.balls_since_boundary = 0
        else:
            self.balls_since_boundary += 1

        # NEW: Update per-batter stats
        if batter_id is not None:
            self.batter_balls_faced[batter_id] += 1
            self.batter_runs_scored[batter_id] += runs

        # NEW: Update per-bowler stats
        if bowler_id is not None:
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
    

def extract_raw_state(delivery, player_registry, score, wickets, balls):
    """
    DESIGN DECISION: Pure function that extracts raw state from delivery.
    REASONING: Separation of concerns - parsing vs feature engineering.
    This function only extracts what's directly in the data.
    """
    batter = delivery['batter']
    non_striker = delivery['non_striker']
    bowler = delivery['bowler']
    runs = delivery['runs']['total']
    
    # Player IDs from registry
    batter_id = player_registry[batter]
    non_striker_id = player_registry[non_striker]
    bowler_id = player_registry[bowler]
    
    # Check for events
    is_wicket = 'wickets' in delivery
    extra_type = list(delivery.get('extras', {}).keys()) if 'extras' in delivery else []
    is_wide = 'wides' in extra_type
    is_noball = 'noballs' in extra_type
    
    return {
        'batter_id': batter_id,
        'non_striker_id': non_striker_id,
        'bowler_id': bowler_id,
        'runs': runs,
        'is_wicket': is_wicket,
        'is_wide': is_wide,
        'is_noball': is_noball,
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


def parse_match_data_v2(json_data, player_stats_tracker, venue_tracker=None, player_metadata=None, elo_tracker=None, match_k_factor=None):
    """
    DESIGN DECISION: Pass tracker as parameter rather than global.
    REASONING: Makes dependencies explicit, easier to test, allows multiple trackers
    for different experiments (e.g., one with h2h, one without).

    Args:
        json_data: Raw JSON string of match data
        player_stats_tracker: PlayerStatsTracker for player statistics
        venue_tracker: Optional VenueStatsTracker for venue statistics
        player_metadata: Optional PlayerMetadataProvider for player attributes

    Returns:
        Tuple of (all_balls list, innings_totals list for venue update)
    """
    data = json.loads(json_data)
    player_registry = data['info']['registry']['people']

    # DESIGN DECISION: Store match metadata for potential venue/team features
    # REASONING: Might want venue-specific features later
    match_info = {
        'venue': data['info'].get('venue', 'unknown'),
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
                    'innings_id': f"{inning_idx}_{hash(json_data) % 100000}",
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
                    # Target
                    'ball_outcome': normalize_ball_outcome(state['runs'], state['is_wicket']),

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

                all_balls.append(ball_record)

                # Update states AFTER recording the ball
                # DESIGN DECISION: Update after recording
                # REASONING: Features should reflect pre-ball state
                player_stats_tracker.update_stats(
                    state['batter_id'],
                    state['bowler_id'],
                    state['runs'],
                    state['is_wicket'],
                    batter_hand=batter_hand if batter_hand != 'unknown' else None,
                    is_pace=is_pace
                )

                # Update ELO ratings AFTER recording the ball
                if elo_tracker is not None:
                    elo_tracker.update(
                        state['batter_id'],
                        state['bowler_id'],
                        state['runs'],
                        state['is_wicket'],
                        k_factor=match_k_factor
                    )

                # NEW: Pass batter/bowler IDs and wicket status for tracking
                innings_calc.update_ball_history(
                    state['runs'],
                    is_boundary=(state['runs'] >= 4),
                    is_wicket=state['is_wicket'],
                    batter_id=state['batter_id'],
                    bowler_id=state['bowler_id']
                )

                # Update match state
                if state['is_wicket']:
                    wickets += 1
                score += state['runs']
                is_legal = not (state['is_wide'] or state['is_noball'])
                if is_legal:
                    balls += 1

                # Accumulate per-innings stats for venue profile
                if is_legal:
                    inn_agg['total_balls'] += 1
                    inn_agg['total_runs'] += state['runs']
                    if state['runs'] == 0 and not state['is_wicket']:
                        inn_agg['dots'] += 1
                    if state['runs'] >= 4 and not state['is_wicket']:
                        inn_agg['boundaries'] += 1
                    if state['is_wicket']:
                        inn_agg['wickets'] += 1
                    # Phase stats (balls is already incremented)
                    if balls <= 36:  # powerplay: first 6 overs
                        inn_agg['powerplay_runs'] += state['runs']
                        inn_agg['powerplay_balls'] += 1
                    if balls > 90:  # death: overs 16-20 (balls 91-120)
                        inn_agg['death_runs'] += state['runs']
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

'''
def process_folder_v2(folder_path):
    """
    DESIGN DECISION: Single pass through chronologically sorted matches.
    REASONING: Ensures player stats are accumulated in correct temporal order,
    preventing data leakage.
    """
    # Initialize tracker that will accumulate across all matches
    player_stats_tracker = PlayerStatsTracker()
    
    all_balls = []
    processed_files = 0
    
    # CRITICAL: Sort by date to ensure correct temporal ordering
    json_files = sorted(
        Path(folder_path).glob('*.json'),
        key=lambda x: json.loads(x.read_text())['info']['dates'][0]
    )
    
    print(f"Processing {len(json_files)} files in chronological order...")
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as file:
                json_data = file.read()
            
            match_balls = parse_match_data_v2(json_data, player_stats_tracker)
            all_balls.extend(match_balls)
            processed_files += 1
            
            if processed_files % 100 == 0:
                print(f"Processed {processed_files} matches, {len(all_balls)} balls")
                
        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")
    
    return all_balls, processed_files


# Main execution
if __name__ == "__main__":
    all_balls, total_files = process_folder_v2('data/t20s_json')
    
    print(f"\nTotal files processed: {total_files}")
    print(f"Total balls collected: {len(all_balls)}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_balls)
    
    # DESIGN DECISION: Save with version suffix
    # REASONING: Can compare different feature sets without losing previous work
    output_file = 'cricket_data_v2_with_features.parquet'
    df.to_parquet(output_file, index=False)
    
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Number of features: {len(df.columns)}")
    print(f"Saved to {output_file}")
    
    # Display feature categories for verification
    print("\nFeature categories:")
    print("- Basic features:", [c for c in df.columns if c.startswith('is_') or '_ratio' in c])
    print("- Player stats:", [c for c in df.columns if 'batsman_' in c or 'bowler_' in c])
    print("- H2H features:", [c for c in df.columns if 'h2h_' in c])
    print("- Momentum features:", [c for c in df.columns if 'last_' in c or 'since_boundary' in c])
'''

def process_folder_v2_with_splits(folder_path):
    """
    Process matches chronologically and save separate parquet files for each temporal split
    """
    # Split boundaries (Option A — betting_eval ⊂ test, driven by the
    # polymarket odds file at eval time rather than by an event-name filter).
    train_end = datetime(2024, 12, 31)
    val_start = datetime(2024, 12, 31)
    val_end = datetime(2025, 6, 30)
    test_start = datetime(2025, 7, 1)
    test_end = datetime(2026, 4, 16)
    golden_start = datetime(2026, 4, 17)
    
    # Initialize trackers that will accumulate across all matches
    player_stats_tracker = PlayerStatsTracker()
    venue_tracker = VenueStatsTracker()  # NEW: Track venue statistics
    elo_tracker = PlayerEloTracker()  # Track player-level ELO ratings

    # NEW: Initialize player metadata provider for hand/arm/type/age features
    player_metadata = PlayerMetadataProvider('data/all_players_enriched.csv')

    # Data containers for each split. The old `betting_test` subset (T20 WC 2024)
    # is gone — betting eval now lives in `data/polymarket_test/` + betting_odds_polymarket.json
    # and is applied at eval time, not parse time.
    split_data = {
        'train': [],
        'validation': [],
        'test': [],
        'golden_test': []
    }

    # Stats cache: snapshots at each match date
    # To avoid memory issues, we'll save incrementally
    stats_snapshots = {}
    # Dates we have ever snapshotted (across all chunks) — persists after
    # stats_snapshots is cleared on chunk save. Without this, a date that
    # straddles a chunk boundary (i.e. its first match triggers a chunk
    # save) would get a second, wrong snapshot when its later same-day
    # matches are processed.
    snapshotted_dates: set = set()
    cache_chunks = []  # List of saved chunk files
    save_interval = 50  # Save every 50 snapshots (smaller to prevent huge chunks)

    processed_files = 0
    
    # Sort files chronologically
    print("Sorting files chronologically...")
    json_files = sorted(
        Path(folder_path).glob('*.json'),
        key=lambda x: json.loads(x.read_text())['info']['dates'][0]
    )

    print(f"Processing {len(json_files)} files in chronological order...")
    
    skipped_non_male = 0
    for file_path in json_files:
        try:
            with open(file_path, 'r') as file:
                json_data = file.read()

            data = json.loads(json_data)

            # Parse-time gender filter: women's ball data was previously still
            # contributing to ball-level training even though ELO was gender-segregated.
            # Skip entirely so training sees only men's T20 cricket.
            gender = data['info'].get('gender', 'male')
            if gender != 'male':
                skipped_non_male += 1
                continue

            match_date = datetime.strptime(data['info']['dates'][0], '%Y-%m-%d')

            # Determine which split this match belongs to
            if match_date < train_end:
                current_split = 'train'
            elif match_date < test_start:
                current_split = 'validation'
            elif match_date < golden_start:
                current_split = 'test'
            else:
                current_split = 'golden_test'

            event_info = data['info'].get('event', {})
            event_name = event_info.get('name', '') if isinstance(event_info, dict) else ''
            team_type = data['info'].get('team_type', 'unknown')
            teams = data['info'].get('teams', [])

            match_k_factor = classify_match_k_factor(event_name, team_type, teams)
            use_elo = elo_tracker  # gender filter upstream means this is always men's T20

            # CRITICAL: Take snapshot BEFORE processing this match
            # This represents what we knew at the START of this match (for simulations)
            # Only save first snapshot per date to avoid overwriting when multiple matches on same day
            match_date_str = match_date.strftime('%Y-%m-%d')
            if match_date_str not in snapshotted_dates:
                snapshotted_dates.add(match_date_str)
                # Include venue stats and ELO in snapshot for temporal integrity
                stats_snapshots[match_date_str] = deep_copy_stats(player_stats_tracker, venue_tracker, elo_tracker)

                # Periodically save snapshots to avoid memory issues
                if len(stats_snapshots) >= save_interval:
                    chunk_dir = Path('models/cache_chunks_v3')
                    chunk_dir.mkdir(parents=True, exist_ok=True)
                    chunk_file = chunk_dir / f'cache_chunk_{len(cache_chunks)}.pkl'
                    with open(chunk_file, 'wb') as f:
                        pickle.dump(stats_snapshots, f, protocol=pickle.HIGHEST_PROTOCOL)
                    cache_chunks.append(chunk_file)
                    print(f"  💾 Saved snapshot chunk {len(cache_chunks)} ({len(stats_snapshots)} dates)")
                    stats_snapshots = {}  # Clear memory

            # Process the match (pass gender-filtered ELO tracker and K-factor)
            match_balls, innings_totals, venue, innings_details, chase_won = parse_match_data_v2(
                json_data, player_stats_tracker, venue_tracker, player_metadata,
                elo_tracker=use_elo, match_k_factor=match_k_factor
            )
            print(f"  Processed match on {match_date_str}: {len(match_balls)} balls")

            # Update venue stats AFTER processing (temporal integrity)
            for inn_detail in innings_details:
                venue_tracker.update_venue_stats_detailed(venue, inn_detail)
            if chase_won is not None:
                venue_tracker.update_venue_match_result(venue, chase_won)

            # Add to appropriate split(s)
            split_data[current_split].extend(match_balls)

            processed_files += 1

            if processed_files % 100 == 0:
                total_balls = sum(len(balls) for balls in split_data.values())
                print(f"✓ Processed {processed_files} matches, {total_balls} total balls, {len(stats_snapshots)} snapshots")

        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")
    
    # Save separate parquet files for each split
    # v3: New data with player metadata features (Tier 1/2/3)
    output_dir = Path('data/xgb_data_v3')
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, balls in split_data.items():
        if balls:  # Only save if there's data
            df = pd.DataFrame(balls)
            output_file = output_dir / f'cricket_data_v3_{split_name}.parquet'
            df.to_parquet(output_file, index=False)
            print(f"Saved {split_name}: {len(balls)} balls to {output_file}")

    # Save player stats cache for simulations (chunked format - no merge!)
    print(f"\nSaving player stats cache (chunked format)...")

    # Save any remaining snapshots as final chunk
    if stats_snapshots:
        chunk_dir = Path('models/cache_chunks_v3')
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_file = chunk_dir / f'cache_chunk_{len(cache_chunks)}.pkl'
        with open(chunk_file, 'wb') as f:
            pickle.dump(stats_snapshots, f, protocol=pickle.HIGHEST_PROTOCOL)
        cache_chunks.append(chunk_file)
        print(f"  💾 Saved final snapshot chunk {len(cache_chunks)} ({len(stats_snapshots)} dates)")

    # Build metadata with date lists for lazy loading
    print(f"\nBuilding metadata with date indices for lazy loading...")
    chunks_with_dates = []

    for i, chunk_file in enumerate(cache_chunks):
        print(f"  Indexing chunk {i+1}/{len(cache_chunks)}...", end=' ')

        # Load chunk to get its dates
        with open(chunk_file, 'rb') as f:
            chunk_data = pickle.load(f)

        dates = sorted(chunk_data.keys())
        print(f"✓ ({len(dates)} dates)")

        chunks_with_dates.append({
            'file': str(chunk_file.relative_to('models')),
            'dates': dates,
            'num_dates': len(dates)
        })

        del chunk_data  # Free memory

    # Save metadata file with date indices
    # v3: Includes type-based stats (batting_vs_type, bowling_vs_hand)
    total_dates = sum(c['num_dates'] for c in chunks_with_dates)
    metadata = {
        'version': 'v3',  # Version identifier
        'num_chunks': len(cache_chunks),
        'num_matches': processed_files,
        'num_dates': total_dates,
        'num_players_batting': len(player_stats_tracker.batting_stats),
        'num_players_bowling': len(player_stats_tracker.bowling_stats),
        'num_h2h_matchups': len(player_stats_tracker.h2h_stats),
        'build_timestamp': datetime.now().isoformat(),
        'chunk_files': [str(f.relative_to('models')) for f in cache_chunks],  # Kept for backwards compat
        'chunks': chunks_with_dates,  # New format with date indices
        'features': ['batting_vs_type', 'bowling_vs_hand', 'venue', 'batting_elo', 'bowling_elo'],
    }

    metadata_path = Path('models/player_stats_cache_v3_metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Calculate total size
    total_size_mb = sum(f.stat().st_size for f in cache_chunks) / (1024 * 1024)

    print(f"\n✓ Saved player stats cache (chunked format)")
    print(f"  Total chunks: {len(cache_chunks)}")
    print(f"  Total size: {total_size_mb:.1f} MB")
    print(f"  Date snapshots: ~{total_dates:,}")
    print(f"  Unique batters: {metadata['num_players_batting']:,}")
    print(f"  Unique bowlers: {metadata['num_players_bowling']:,}")
    print(f"  H2H matchups: {metadata['num_h2h_matchups']:,}")
    print(f"  Metadata saved to: {metadata_path}")

    # Write feature hash for smart caching by experiment runner
    try:
        from feature_registry import get_feature_hash, resolve_feature_list, V3_GROUPS
        _feature_list = resolve_feature_list(V3_GROUPS)
        _hash_data = {"hash": get_feature_hash(_feature_list), "version": "v3", "n_features": len(_feature_list)}
        import json as _json
        with open(output_dir / '.feature_hash', 'w') as f:
            _json.dump(_hash_data, f)
        print(f"  Feature hash saved: {_hash_data['hash']}")
    except ImportError:
        pass  # feature_registry not available, skip

    return split_data, processed_files


# Update the main execution
if __name__ == "__main__":
    split_data, total_files = process_folder_v2_with_splits('data/t20s_json')
    
    print(f"\nTotal files processed: {total_files}")
    
    # Summary statistics
    for split_name, balls in split_data.items():
        if balls:
            df = pd.DataFrame(balls)
            print(f"\n{split_name.upper()}:")
            print(f"  Balls: {len(balls)}")
            print(f"  Features: {len(df.columns)}")