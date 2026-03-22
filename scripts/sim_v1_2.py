import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from abc import ABC, abstractmethod
import random
from multiprocessing import Pool, cpu_count
import time
from datetime import datetime
import pandas as pd

# Import player metadata provider for Tier 1/2/3 features
from player_metadata import (
    PlayerMetadataProvider,
    encode_batter_hand,
    encode_bowler_arm,
    encode_is_pace,
    encode_bowling_type
)


class Outcome(Enum):
    DOT = 0
    ONE = 1
    TWO = 2
    FOUR = 4
    SIX = 6
    WICKET = 7
    WIDE = 8
    NO_BALL = 9

@dataclass
class Player:
    """Represents a cricket player"""
    player_id: str  # Unique ID matching your training data
    name: str
    team: str
    role: str = "allrounder" 


@dataclass
class TeamLineup:
    """Team lineup with batting order"""
    team_name: str
    players: List[Player]  # In batting order (0-10)
    
    def get_player_by_index(self, idx: int) -> Optional[Player]:
        if 0 <= idx < len(self.players):
            return self.players[idx]
        return None

@dataclass
class MatchState:
    # Match setup
    team1_lineup: TeamLineup
    team2_lineup: TeamLineup
    batting_first: str
    venue: str
    match_date: datetime  # Added for temporal features

    # Current state
    innings: int = 1  # 1 or 2
    balls: int = 0  # balls bowled in current innings (0-119 for T20)
    runs: np.ndarray = field(default_factory=lambda: np.zeros(2))  # [team1_runs, team2_runs]
    wickets: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=int))  # [team1_wickets, team2_wickets]
    
    # Current over state
    current_over: List[Outcome] = field(default_factory=list)  # outcomes in current over
    
    # Player tracking
    striker_idx: int = 0  # index in batting order (0-10)
    non_striker_idx: int = 1
    bowler_idx: int = 0  # index in bowling team
    last_bowler_idx: int = -1 

    # Track who has batted (for getting next batsman)
    batsmen_out: Dict[int, List[int]] = field(default_factory=lambda: {0: [], 1: []})  # team_idx -> list of out batsman indices
    
    # Ball-by-ball history (for features)
    # Each row: [innings, over, ball, runs, wicket, batting_team_idx, striker_idx, bowling_team_idx, bowler_idx]
    history: np.ndarray = field(default_factory=lambda: np.zeros((300, 9)))  # Increased size for extras
    history_idx: int = 0

    # Tracking: (team_idx, player_idx) -> value
    bowler_balls: Dict[Tuple[int, int], int] = field(default_factory=dict)
    batsman_stats: Dict[Tuple[int, int], Tuple[int, int]] = field(default_factory=dict)

    # NEW: Partnership tracking (runs since last wicket)
    partnership_runs: int = 0

    @property
    def current_team_idx(self) -> int:
        """0 for team1, 1 for team2 (batting)"""
        return 0 if self.batting_team == self.team1 else 1

    @property 
    def bowling_team_idx(self) -> int:
        """0 for team1, 1 for team2 (bowling)"""
        return 1 - self.current_team_idx
    
    @property
    def batting_team(self) -> str:
        if self.innings == 1:
            return self.batting_first
        return self.team2 if self.batting_first == self.team1 else self.team1
    
    @property
    def bowling_team(self) -> str:
        return self.team1 if self.batting_team == self.team2 else self.team2
    
    @property
    def overs_completed(self) -> float:
        return self.balls // 6 + (self.balls % 6) / 10
    
    @property
    def balls_remaining(self) -> int:
        return 120 - self.balls
    
    @property
    def target(self) -> Optional[int]:
        """Target for team batting second"""
        if self.innings == 2:
            return int(self.runs[1 - self.current_team_idx]) + 1
        return None
    
    @property
    def required_run_rate(self) -> Optional[float]:
        if self.target and self.balls_remaining > 0:
            return (self.target - self.runs[self.current_team_idx]) * 6 / self.balls_remaining
        return None
    
    @property
    def team1(self) -> str:
        return self.team1_lineup.team_name
    
    @property
    def team2(self) -> str:
        return self.team2_lineup.team_name
    
    @property
    def batting_lineup(self) -> TeamLineup:
        """Get current batting team's lineup"""
        if self.batting_team == self.team1:
            return self.team1_lineup
        return self.team2_lineup
    
    @property
    def bowling_lineup(self) -> TeamLineup:
        """Get current bowling team's lineup"""
        if self.bowling_team == self.team1:
            return self.team1_lineup
        return self.team2_lineup
    
    @property
    def current_striker(self) -> Optional[Player]:
        """Get current striker Player object"""
        return self.batting_lineup.get_player_by_index(self.striker_idx)
    
    @property
    def current_bowler(self) -> Optional[Player]:
        """Get current bowler Player object"""
        return self.bowling_lineup.get_player_by_index(self.bowler_idx)
    
    def get_next_batsman_idx(self) -> int:
        """Get the next batsman in order who hasn't batted yet"""
        team_idx = self.current_team_idx
        batsmen_out = self.batsmen_out[team_idx]
        
        # Find the lowest index player who hasn't batted yet
        # and isn't currently batting
        for idx in range(11):
            if (idx not in batsmen_out and 
                idx != self.striker_idx and 
                idx != self.non_striker_idx):
                return idx
        
        # This shouldn't happen in a valid game
        return 10  # Last player
    
    def is_innings_over(self) -> bool:
        """Check if current innings is complete"""
        team_idx = self.current_team_idx
        
        # All out
        if self.wickets[team_idx] >= 10:
            return True
        
        # Overs complete
        if self.balls >= 120:
            return True
        
        # Target achieved (2nd innings only)
        if self.innings == 2 and self.target:
            if self.runs[team_idx] >= self.target:
                return True
        
        return False
    
    def is_match_over(self) -> bool:
        """Check if match is complete"""
        if self.innings == 1:
            return False
        return self.is_innings_over()
    
    def get_available_bowlers(self) -> List[int]:
        """Get indices of bowlers who can bowl next over"""
        available = []
        bowling_team = self.bowling_team_idx
        
        for idx in range(11):  # All 11 players
            # Can't bowl consecutive overs
            if self.last_bowler_idx >= 0 and idx == int(self.last_bowler_idx):
                continue
            
            # Check over limit (max 24 balls = 4 overs in T20)
            if self.bowler_balls.get((bowling_team, idx), 0) >= 24:
                continue
            
            available.append(idx)
        
        return available
    
    def update(self, outcome: Outcome, runs: int = 0):
        """Update state after a ball"""
        # Check if we have space in history (defensive programming)
        if self.history_idx >= len(self.history):
            # Extend history array if needed
            new_history = np.zeros((self.history.shape[0] + 100, self.history.shape[1]))
            new_history[:self.history.shape[0]] = self.history
            self.history = new_history
        
        # Record in history
        over_num = self.balls // 6
        ball_in_over = self.balls % 6
        
        self.history[self.history_idx] = [
            self.innings, over_num, ball_in_over, runs, 
            int(outcome == Outcome.WICKET), 
            self.current_team_idx, self.striker_idx,
            self.bowling_team_idx, self.bowler_idx
        ]
        self.history_idx += 1

        # Update balls and bowler tracking
        if outcome not in [Outcome.WIDE, Outcome.NO_BALL]:
            self.balls += 1
            bowler_key = (self.bowling_team_idx, self.bowler_idx)
            self.bowler_balls[bowler_key] = self.bowler_balls.get(bowler_key, 0) + 1

        # Update team score
        self.runs[self.current_team_idx] += runs

        # NEW: Update partnership runs
        self.partnership_runs += runs

        # Update batsman stats (fixed: removed non-existent BYE, LEG_BYE)
        if outcome != Outcome.WIDE:
            batsman_key = (self.current_team_idx, self.striker_idx)
            stats = self.batsman_stats.get(batsman_key, (0, 0))
            self.batsman_stats[batsman_key] = (stats[0] + runs, stats[1] + 1)

        # Handle wicket
        if outcome == Outcome.WICKET:
            self.wickets[self.current_team_idx] += 1
            # Track who got out
            self.batsmen_out[self.current_team_idx].append(self.striker_idx)
            # Get next batsman
            self.striker_idx = self.get_next_batsman_idx()
            # NEW: Reset partnership on wicket
            self.partnership_runs = 0
        
        # Rotate strike
        if runs % 2 == 1:
            self.striker_idx, self.non_striker_idx = self.non_striker_idx, self.striker_idx
        
        # Add to current over
        self.current_over.append(outcome)
        
        # End of over
        if self.balls % 6 == 0 and outcome not in [Outcome.WIDE, Outcome.NO_BALL]:
            self.end_over()

    def end_over(self):
        """Handle end of over"""
        # Rotate strike
        self.striker_idx, self.non_striker_idx = self.non_striker_idx, self.striker_idx
        
        # Track last bowler
        self.last_bowler_idx = self.bowler_idx
        
        # Clear current over
        self.current_over = []

    def start_new_innings(self):
        """Setup for second innings"""
        self.innings = 2
        self.balls = 0
        self.striker_idx = 0
        self.non_striker_idx = 1
        self.bowler_idx = 0  # Will be selected by strategy
        self.last_bowler_idx = -1
        self.current_over = []
        self.partnership_runs = 0  # NEW: Reset partnership for new innings
        # Note: We keep bowler_balls and batsman_stats as they track both teams
    
    def copy(self):
        """Efficient copy for parallel simulations"""
        new_state = MatchState(
            team1_lineup=self.team1_lineup,  # Fixed
            team2_lineup=self.team2_lineup,  # Fixed
            batting_first=self.batting_first,
            venue=self.venue,
            match_date=self.match_date       # Added
        )
        
        # Copy all attributes
        new_state.innings = self.innings
        new_state.balls = self.balls
        new_state.runs = self.runs.copy()
        new_state.wickets = self.wickets.copy()
        new_state.current_over = self.current_over.copy()
        new_state.striker_idx = self.striker_idx
        new_state.non_striker_idx = self.non_striker_idx
        new_state.bowler_idx = self.bowler_idx
        new_state.last_bowler_idx = self.last_bowler_idx
        new_state.batsmen_out = {k: v.copy() for k, v in self.batsmen_out.items()}  # Added
        new_state.history = self.history.copy()
        new_state.history_idx = self.history_idx
        new_state.bowler_balls = self.bowler_balls.copy()
        new_state.batsman_stats = self.batsman_stats.copy()
        new_state.partnership_runs = self.partnership_runs  # NEW

        return new_state

# Bowler Selection
class BowlerSelector(ABC):
    """Interface for bowler selection strategies"""
    @abstractmethod
    def select_bowler(self, state: MatchState, available: List[int]) -> int:
        pass

class RandomBowlerSelector(BowlerSelector):
    """Simple random selection for now"""
    def select_bowler(self, state: MatchState, available: List[int]) -> int:
        return random.choice(available)

# T20 Rules
class T20Rules:
    """Enforces T20 cricket rules and match flow"""
    
    def __init__(self, bowler_selector: Optional[BowlerSelector] = None):
        self.bowler_selector = bowler_selector or RandomBowlerSelector()  # Fixed: add default
    
    def select_next_bowler(self, state: MatchState) -> int:
        """Select bowler for next over"""
        available = state.get_available_bowlers()
        
        if not available:
            raise ValueError("No available bowlers! This shouldn't happen in T20.")
        
        return self.bowler_selector.select_bowler(state, available)
    
    def is_legal_outcome(self, state: MatchState, outcome: Outcome) -> bool:
        """Check if outcome is legal in current state"""
        # Can't get wicket if already 10 down
        if outcome == Outcome.WICKET and state.wickets[state.current_team_idx] >= 10:
            return False
        
        # Last ball of innings can't be wide/no-ball (simplified)
        if state.balls == 119 and outcome in [Outcome.WIDE, Outcome.NO_BALL]:
            return False
        
        return True
    
    def process_ball(self, state: MatchState, outcome: Outcome) -> int:
        """Process a ball and return runs scored"""
        runs = 0
        
        # Direct run outcomes (fixed: removed non-existent THREE, FIVE)
        if outcome in [Outcome.ONE, Outcome.TWO, Outcome.FOUR, Outcome.SIX]:
            runs = outcome.value
        
        # Extras
        elif outcome == Outcome.WIDE:
            runs = 1  # Simplified: 1 run for wide
        elif outcome == Outcome.NO_BALL:
            runs = 1  # Simplified: 1 run for no-ball
        
        # Wicket - no runs (simplified, ignoring run outs with runs)
        elif outcome == Outcome.WICKET:
            runs = 0
        
        # Update state
        state.update(outcome, runs)
        
        return runs
    
    def simulate_ball(self, state: MatchState, model: 'PredictionModel') -> Tuple[Outcome, int]:
        """Simulate next ball using prediction model"""
        # Extract features
        features = model.extract_features(state)
        
        # Get outcome probabilities
        probs = model.predict_next_ball(features)
        
        # Map string outcomes to Enum
        outcome_map = {
            'dot': Outcome.DOT,
            'one': Outcome.ONE,
            'two': Outcome.TWO,
            'four': Outcome.FOUR,
            'six': Outcome.SIX,
            'wicket': Outcome.WICKET,
            'wide': Outcome.WIDE,
            'no_ball': Outcome.NO_BALL
        }
        
        # Get outcomes and their probabilities
        outcomes = []
        weights = []
        for name, prob in probs.items():
            if name in outcome_map:
                outcomes.append(outcome_map[name])
                weights.append(prob)
        
        # Sample outcome
        outcome = random.choices(outcomes, weights=weights)[0]
        
        # Ensure legal outcome
        if not self.is_legal_outcome(state, outcome):
            outcome = Outcome.DOT
        
        # Process the ball
        runs = self.process_ball(state, outcome)
        
        # Select new bowler if over just ended (and not end of innings)
        if state.balls % 6 == 0 and state.balls > 0 and not state.is_innings_over():
            state.bowler_idx = self.select_next_bowler(state)
        
        return outcome, runs

# Prediction Models
class PredictionModel(ABC):
    @abstractmethod
    def predict_next_ball(self, features: np.ndarray) -> Dict[str, float]:
        """Returns probability distribution over outcomes"""
        pass
    
    @abstractmethod
    def extract_features(self, state: MatchState) -> np.ndarray:
        """Extract features from match state"""
        pass

class XGBoostModelV2(PredictionModel):
    def __init__(self, model_path: str, batter_encoder_path: str, bowler_encoder_path: str,
                 feature_columns_path: str, stats_provider=None, player_metadata=None,
                 matchup_encoder_path: str = None, ball_calibrator=None):
        import joblib
        self.model = joblib.load(model_path)
        self.batter_encoder = joblib.load(batter_encoder_path)
        self.bowler_encoder = joblib.load(bowler_encoder_path)
        self.stats_provider = stats_provider  # Optional stats provider for simulations
        self.player_metadata = player_metadata  # NEW: Optional player metadata for Tier 1/2/3 features
        self.ball_calibrator = ball_calibrator  # Optional ball-level calibrator

        # NEW: Load matchup encoder if provided
        self.matchup_encoder = None
        if matchup_encoder_path:
            try:
                self.matchup_encoder = joblib.load(matchup_encoder_path)
            except:
                print(f"  Warning: Could not load matchup encoder from {matchup_encoder_path}")

        # Load feature columns to ensure consistency
        with open(feature_columns_path, 'r') as f:
            self.feature_columns = [line.strip() for line in f.readlines()]

        # Map model output classes to our Outcome enum
        # XGBoost trains with 6-class remapping: {0:dot, 1:one, 2:two, 4:four, 6:six, 7:wicket}
        # → remapped to classes {0,1,2,3,4,5}
        self.class_to_outcome = {
            0: 'dot', 1: 'one', 2: 'two', 3: 'four', 4: 'six', 5: 'wicket'
        }

        stats_mode = "with real player stats" if stats_provider else "with zero stats (fallback)"
        metadata_mode = "with player metadata" if player_metadata else "without player metadata"
        print(f"Loaded XGBoost v2 model with {len(self.feature_columns)} features {stats_mode} {metadata_mode}")

    def extract_features(self, state: MatchState) -> pd.DataFrame:
        """Extract comprehensive feature set matching v2 training"""
        import pandas as pd

        team_idx = state.current_team_idx
        striker = state.current_striker
        bowler = state.current_bowler
        wickets_in_hand = 10 - int(state.wickets[team_idx])

        # Basic state features
        features = {
            'inning_idx': state.innings,
            'score': int(state.runs[team_idx]),
            'wickets': int(state.wickets[team_idx]),
            'balls_bowled': state.balls,
            'run_rate': float(state.runs[team_idx]) / float(state.balls + 1),
            'wickets_ratio': float(state.wickets[team_idx]) / 10.0,
            'balls_ratio': float(state.balls) / 120.0,
            'wickets_in_hand': wickets_in_hand,
            'balls_remaining': state.balls_remaining,  # NEW

            # Match phase indicators
            'is_powerplay': state.balls < 36,
            'is_middle_overs': 36 <= state.balls < 96,
            'is_death_overs': state.balls >= 96,
            'balls_in_over': state.balls % 6,
        }

        # Player encoding
        try:
            features['batter_encoded'] = self.batter_encoder.transform([str(striker.player_id)])[0]
        except:
            features['batter_encoded'] = -1

        try:
            features['bowler_encoded'] = self.bowler_encoder.transform([str(bowler.player_id)])[0]
        except:
            features['bowler_encoded'] = -1

        # NEW: Per-innings batter stats (from batsman_stats tracking)
        batsman_key = (team_idx, state.striker_idx)
        batter_innings_stats = state.batsman_stats.get(batsman_key, (0, 0))
        features['batter_runs_scored'] = batter_innings_stats[0]
        features['batter_balls_faced'] = batter_innings_stats[1]

        # NEW: Per-innings bowler stats (from bowler_balls tracking)
        bowler_key = (state.bowling_team_idx, state.bowler_idx)
        bowler_balls_in_innings = state.bowler_balls.get(bowler_key, 0)
        features['bowler_balls_in_innings'] = bowler_balls_in_innings
        features['bowler_overs_in_innings'] = bowler_balls_in_innings / 6

        # NEW: Partnership runs
        features['partnership_runs'] = state.partnership_runs

        # NEW: Chase features (2nd innings only)
        target = state.target or 0
        features['chase_target'] = target  # Renamed from 'target' to avoid collision with prediction target
        if state.innings == 2 and target > 0 and state.balls_remaining > 0:
            runs_needed = target - int(state.runs[team_idx])
            run_rate_required = (runs_needed * 6 / state.balls_remaining)
            lead_gap = -runs_needed  # Negative means chasing team is behind
        else:
            run_rate_required = 0
            lead_gap = int(state.runs[team_idx])  # First innings: just the score
        features['run_rate_required'] = run_rate_required
        features['lead_gap'] = lead_gap

        # NEW: Pressure cooker index (RRR / wickets_remaining)
        if state.innings == 2 and wickets_in_hand > 0 and run_rate_required > 0:
            features['pressure_cooker_index'] = run_rate_required / wickets_in_hand
        else:
            features['pressure_cooker_index'] = 0

        # NEW: Non-striker's strike rate (from batsman_stats tracking)
        non_striker_key = (team_idx, state.non_striker_idx)
        non_striker_stats = state.batsman_stats.get(non_striker_key, (0, 0))
        if non_striker_stats[1] > 0:  # balls faced > 0
            features['non_striker_sr'] = (non_striker_stats[0] / non_striker_stats[1]) * 100
        else:
            features['non_striker_sr'] = 0.0

        # Player stats features - use real stats if provider available
        if self.stats_provider:
            # Get real player stats from historical cache
            batting_stats = self.stats_provider.get_batting_stats(striker.player_id, state.match_date)
            bowling_stats = self.stats_provider.get_bowling_stats(bowler.player_id, state.match_date)
            h2h_stats = self.stats_provider.get_h2h_stats(striker.player_id, bowler.player_id, state.match_date)

            # NEW: Get venue average score (temporal integrity)
            venue_avg = self.stats_provider.get_venue_avg_score(state.venue, state.match_date)

            features.update({
                'batsman_avg': batting_stats['avg'],
                'batsman_sr': batting_stats['sr'],
                'bowler_avg': bowling_stats['avg'],
                'bowler_econ': bowling_stats['econ'],
                'h2h_avg': h2h_stats['avg'],
                'h2h_sr': h2h_stats['sr'],
                'venue_avg_score': venue_avg,  # Historical venue average
            })

            # NEW: Get type-based stats (Tier 3) if available
            if hasattr(self.stats_provider, 'get_batting_vs_type_stats'):
                bat_vs_type = self.stats_provider.get_batting_vs_type_stats(striker.player_id, state.match_date)
                features.update({
                    'batter_avg_vs_pace': bat_vs_type.get('avg_vs_pace', 0.0),
                    'batter_sr_vs_pace': bat_vs_type.get('sr_vs_pace', 0.0),
                    'batter_avg_vs_spin': bat_vs_type.get('avg_vs_spin', 0.0),
                    'batter_sr_vs_spin': bat_vs_type.get('sr_vs_spin', 0.0),
                })
            else:
                features.update({
                    'batter_avg_vs_pace': 0.0, 'batter_sr_vs_pace': 0.0,
                    'batter_avg_vs_spin': 0.0, 'batter_sr_vs_spin': 0.0,
                })

            if hasattr(self.stats_provider, 'get_bowling_vs_hand_stats'):
                bowl_vs_hand = self.stats_provider.get_bowling_vs_hand_stats(bowler.player_id, state.match_date)
                features.update({
                    'bowler_avg_vs_lhb': bowl_vs_hand.get('avg_vs_lhb', 0.0),
                    'bowler_econ_vs_lhb': bowl_vs_hand.get('econ_vs_lhb', 0.0),
                    'bowler_avg_vs_rhb': bowl_vs_hand.get('avg_vs_rhb', 0.0),
                    'bowler_econ_vs_rhb': bowl_vs_hand.get('econ_vs_rhb', 0.0),
                })
            else:
                features.update({
                    'bowler_avg_vs_lhb': 0.0, 'bowler_econ_vs_lhb': 0.0,
                    'bowler_avg_vs_rhb': 0.0, 'bowler_econ_vs_rhb': 0.0,
                })

            # Team strength features (ELO + aggregated stats)
            bat_player_ids = [p.player_id for p in state.batting_lineup.players]
            bowl_player_ids = [p.player_id for p in state.bowling_lineup.players]

            features['striker_elo'] = self.stats_provider.get_batting_elo(striker.player_id, state.match_date)
            features['bowler_elo_rating'] = self.stats_provider.get_bowling_elo(bowler.player_id, state.match_date)

            batting_team_elo = self.stats_provider.get_team_batting_elo(bat_player_ids, state.match_date)
            bowling_team_elo = self.stats_provider.get_team_bowling_elo(bowl_player_ids, state.match_date)
            features['batting_team_elo'] = batting_team_elo
            features['bowling_team_elo'] = bowling_team_elo
            features['elo_diff'] = batting_team_elo - bowling_team_elo

            bat_strength = self.stats_provider.get_team_batting_strength(bat_player_ids, state.match_date)
            bowl_strength = self.stats_provider.get_team_bowling_strength(bowl_player_ids, state.match_date)
            features.update(bat_strength)
            features.update(bowl_strength)
        else:
            # Fallback to zeros if no stats provider
            features.update({
                'batsman_avg': 0.0,
                'batsman_sr': 0.0,
                'bowler_avg': 0.0,
                'bowler_econ': 0.0,
                'h2h_avg': 0.0,
                'h2h_sr': 0.0,
                'venue_avg_score': 0.0,
                # Tier 3 fallbacks
                'batter_avg_vs_pace': 0.0, 'batter_sr_vs_pace': 0.0,
                'batter_avg_vs_spin': 0.0, 'batter_sr_vs_spin': 0.0,
                'bowler_avg_vs_lhb': 0.0, 'bowler_econ_vs_lhb': 0.0,
                'bowler_avg_vs_rhb': 0.0, 'bowler_econ_vs_rhb': 0.0,
                # Team strength fallbacks
                'striker_elo': 1500.0, 'bowler_elo_rating': 1500.0,
                'batting_team_elo': 16500.0, 'bowling_team_elo': 16500.0, 'elo_diff': 0.0,
                'team_batting_avg': 0.0, 'team_batting_sr': 0.0,
                'team_bowling_avg': 0.0, 'team_bowling_econ': 0.0,
            })

        # NEW: Player metadata features (Tier 1 and 2)
        if self.player_metadata:
            batter_meta = self.player_metadata.get_player_metadata(striker.player_id)
            bowler_meta = self.player_metadata.get_player_metadata(bowler.player_id)

            # Tier 1: Direct features (encoded)
            batter_hand = batter_meta['batter_hand']
            bowler_arm = bowler_meta['bowler_arm']
            is_pace = bowler_meta['is_pace']
            bowling_type = bowler_meta['bowling_type']

            features.update({
                'batter_hand': encode_batter_hand(batter_hand),
                'bowler_arm': encode_bowler_arm(bowler_arm),
                'is_pace': encode_is_pace(is_pace),
                'bowling_type': encode_bowling_type(bowling_type),
            })

            # Ages
            batter_age = self.player_metadata.get_player_age(striker.player_id, state.match_date)
            bowler_age = self.player_metadata.get_player_age(bowler.player_id, state.match_date)
            features['batter_age'] = batter_age if batter_age is not None else 0
            features['bowler_age'] = bowler_age if bowler_age is not None else 0

            # Tier 2: Matchup features
            matchup_type = self.player_metadata.get_matchup_type(striker.player_id, bowler.player_id)
            spin_matchup_advantage = self.player_metadata.get_spin_matchup_advantage(striker.player_id, bowler.player_id)
            same_arm_matchup = self.player_metadata.get_same_arm_matchup(striker.player_id, bowler.player_id)

            features['spin_matchup_advantage'] = spin_matchup_advantage
            features['same_arm_matchup'] = 1 if same_arm_matchup else (0 if same_arm_matchup is False else -1)

            # Encode matchup_type if encoder available
            if self.matchup_encoder:
                try:
                    features['matchup_type_encoded'] = self.matchup_encoder.transform([matchup_type])[0]
                except:
                    features['matchup_type_encoded'] = -1  # Unknown matchup
            else:
                features['matchup_type_encoded'] = 0  # Default if no encoder
        else:
            # Fallback if no player metadata
            features.update({
                'batter_hand': 2,  # unknown
                'bowler_arm': 2,   # unknown
                'is_pace': 2,      # unknown
                'bowling_type': 8, # unknown
                'batter_age': 0,
                'bowler_age': 0,
                'spin_matchup_advantage': 0,
                'same_arm_matchup': -1,
                'matchup_type_encoded': 0,
            })

        # Momentum features from match history
        features.update(self._extract_momentum_features(state))

        # Pressure indicators
        features.update(self._extract_pressure_features(state))
        
        # Create DataFrame with only features that exist in training
        df_features = {}
        for col in self.feature_columns:
            df_features[col] = features.get(col, 0.0)
        
        return pd.DataFrame([df_features])
    
    def _extract_momentum_features(self, state: MatchState) -> dict:
        """Extract momentum features from match history"""
        # Handle case where no balls have been bowled yet
        if state.history_idx == 0:
            return {
                'last_5_balls_runs': 0,
                'last_10_balls_runs': 0,
                'last_30_balls_runs': 0,
                'balls_since_boundary': 0,
                'last_10_dots': 0,
            }
        
        # Get last N balls from history for current innings
        current_innings_history = state.history[:state.history_idx]
        current_innings_mask = current_innings_history[:, 0] == state.innings
        
        if not np.any(current_innings_mask):
            return {
                'last_5_balls_runs': 0,
                'last_10_balls_runs': 0,
                'last_30_balls_runs': 0,
                'balls_since_boundary': 0,
                'last_10_dots': 0,
            }
        
        innings_history = current_innings_history[current_innings_mask]
        runs_history = innings_history[:, 3]  # runs column
        
        # Calculate features
        last_5 = runs_history[-5:] if len(runs_history) >= 5 else runs_history
        last_10 = runs_history[-10:] if len(runs_history) >= 10 else runs_history
        last_30 = runs_history[-30:] if len(runs_history) >= 30 else runs_history
        
        # Balls since boundary
        balls_since_boundary = 0
        for i in range(len(runs_history) - 1, -1, -1):
            if runs_history[i] >= 4:
                break
            balls_since_boundary += 1
        
        # Dot balls in last 10
        last_10_dots = np.sum(last_10 == 0) if len(last_10) > 0 else 0
        
        return {
            'last_5_balls_runs': int(np.sum(last_5)),
            'last_10_balls_runs': int(np.sum(last_10)),
            'last_30_balls_runs': int(np.sum(last_30)),
            'balls_since_boundary': balls_since_boundary,
            'last_10_dots': int(last_10_dots),
        }
    
    def _extract_pressure_features(self, state: MatchState) -> dict:
        """Extract pressure indicator features"""
        # Handle case where no balls have been bowled yet
        if state.history_idx == 0:
            return {
                'dot_percentage_recent': 0.0,
                'boundary_percentage_recent': 0.0,
            }
        
        # Get recent balls for current innings
        current_innings_history = state.history[:state.history_idx]
        current_innings_mask = current_innings_history[:, 0] == state.innings
        
        if not np.any(current_innings_mask):
            return {
                'dot_percentage_recent': 0.0,
                'boundary_percentage_recent': 0.0,
            }
        
        innings_history = current_innings_history[current_innings_mask]
        runs_history = innings_history[:, 3]
        
        # Recent 10 balls for dot percentage
        recent_10 = runs_history[-10:] if len(runs_history) >= 10 else runs_history
        dot_pct = np.sum(recent_10 == 0) / len(recent_10) if len(recent_10) > 0 else 0.0
        
        # Recent 30 balls for boundary percentage
        recent_30 = runs_history[-30:] if len(runs_history) >= 30 else runs_history
        boundary_pct = np.sum(recent_30 >= 4) / len(recent_30) if len(recent_30) > 0 else 0.0
        
        return {
            'dot_percentage_recent': dot_pct,
            'boundary_percentage_recent': boundary_pct,
        }
    
    def predict_next_ball(self, features: pd.DataFrame) -> Dict[str, float]:
        """Get probabilities from model"""
        probs = self.model.predict_proba(features)[0]

        # Apply ball-level calibration if available
        if self.ball_calibrator:
            probs = self.ball_calibrator.calibrate_probs(probs)

        # Initialize all outcomes with 0 probability
        outcome_probs = {
            'dot': 0.0, 'one': 0.0, 'two': 0.0, 'four': 0.0,
            'six': 0.0, 'wicket': 0.0, 'wide': 0.0, 'no_ball': 0.0
        }

        # Map model predictions to our outcomes
        for class_idx, prob in enumerate(probs):
            if class_idx in self.class_to_outcome:
                outcome_name = self.class_to_outcome[class_idx]
                outcome_probs[outcome_name] = prob

        # Add small probabilities for extras (not in your model)
        outcome_probs['wide'] = 0.01
        outcome_probs['no_ball'] = 0.01

        # Normalize
        total = sum(outcome_probs.values())
        if total > 0:
            outcome_probs = {k: v/total for k, v in outcome_probs.items()}

        return outcome_probs

class XGBoostModel(PredictionModel):
    def __init__(self, model_path: str, batter_encoder_path: str, bowler_encoder_path: str):
        import joblib
        self.model = joblib.load(model_path)
        self.batter_encoder = joblib.load(batter_encoder_path)
        self.bowler_encoder = joblib.load(bowler_encoder_path)
        
        # Map model output classes to our Outcome enum
        # XGBoost trains with 6-class remapping: {0:dot, 1:one, 2:two, 4:four, 6:six, 7:wicket}
        # → remapped to classes {0,1,2,3,4,5}
        self.class_to_outcome = {
            0: 'dot', 1: 'one', 2: 'two', 3: 'four', 4: 'six', 5: 'wicket'
        }

    def extract_features(self, state: MatchState) -> np.ndarray:
        team_idx = state.current_team_idx
        
        # Get actual player IDs
        striker = state.current_striker
        bowler = state.current_bowler
        
        # Calculate features matching your GBM training
        inning_idx = state.innings
        score = int(state.runs[team_idx])
        wickets = int(state.wickets[team_idx])  # Fixed: was state.runs[team_idx]
        balls_bowled = state.balls
        
        # For batter/bowler encoding, use placeholders for now
        try:
            # These encoders expect player IDs from training data
            batter_encoded = self.batter_encoder.transform([striker.player_id])[0]
            bowler_encoded = self.bowler_encoder.transform([bowler.player_id])[0]
        except:
            # Fallback if player not in training data
            batter_encoded = -1  # Unknown player encoding
            bowler_encoded = -1

        
        # Derived features
        run_rate = score / (balls_bowled + 1)
        wickets_ratio = wickets / 10.0
        balls_ratio = balls_bowled / 120.0
        
        return np.array([
            inning_idx, score, wickets, balls_bowled,
            batter_encoded, bowler_encoded, run_rate,
            wickets_ratio, balls_ratio
        ]).reshape(1, -1)
    
    def predict_next_ball(self, features: np.ndarray) -> Dict[str, float]:
        """Get probabilities from model"""
        probs = self.model.predict_proba(features)[0]
        
        # Initialize all outcomes with 0 probability
        outcome_probs = {
            'dot': 0.0, 'one': 0.0, 'two': 0.0, 'four': 0.0,
            'six': 0.0, 'wicket': 0.0, 'wide': 0.0, 'no_ball': 0.0
        }
        
        # Map model predictions to our outcomes
        for class_idx, prob in enumerate(probs):
            if class_idx in self.class_to_outcome:
                outcome_name = self.class_to_outcome[class_idx]
                outcome_probs[outcome_name] = prob
        
        # Add small probabilities for extras (not in your model)
        outcome_probs['wide'] = 0.01
        outcome_probs['no_ball'] = 0.01
        
        # Normalize
        total = sum(outcome_probs.values())
        if total > 0:
            outcome_probs = {k: v/total for k, v in outcome_probs.items()}
        
        return outcome_probs

class DummyModel(PredictionModel):
    """Simple probability-based model for testing"""
    def extract_features(self, state: MatchState) -> np.ndarray:
        return np.array([0])  # Dummy

    def predict_next_ball(self, features: np.ndarray) -> Dict[str, float]:
        # Simple phase-based probabilities
        return {
            'dot': 0.32,
            'one': 0.39,
            'two': 0.08,
            'four': 0.10,
            'six': 0.04,
            'wicket': 0.05,
            'wide': 0.01,
            'no_ball': 0.01
        }


class LSTMModelV1(PredictionModel):
    """LSTM-based ball outcome predictor with sequence context"""

    def __init__(self,
                 model_path: str,
                 batter_encoder_path: str,
                 bowler_encoder_path: str,
                 feature_columns_path: str,
                 scaler_path: str,
                 config_path: str,
                 stats_provider=None,
                 player_metadata=None,
                 matchup_encoder_path: str = None,
                 venue_encoder_path: str = None,
                 window_size: int = 10,
                 device: str = 'cpu'):
        import torch
        import torch.nn as nn
        import joblib
        import json

        self.device = torch.device(device)
        self.window_size = window_size
        self.stats_provider = stats_provider
        self.player_metadata = player_metadata

        # Load model config
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Import and create model architecture
        # Handle import from different locations
        try:
            from lstm_v1 import LSTMBallPredictor
        except ImportError:
            from scripts.lstm_v1 import LSTMBallPredictor
        self.model = LSTMBallPredictor(
            n_continuous=config['n_continuous'],
            n_batters=config['n_batters'],
            n_bowlers=config['n_bowlers'],
            n_venues=config['n_venues'],
            n_matchups=config['n_matchups'],
            embed_dim_player=config['embed_dim_player'],
            embed_dim_venue=config['embed_dim_venue'],
            embed_dim_matchup=config['embed_dim_matchup'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            n_classes=config['n_classes']
        )

        # Load trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()

        # Load encoders
        self.batter_encoder = joblib.load(batter_encoder_path)
        self.bowler_encoder = joblib.load(bowler_encoder_path)
        self.scaler = joblib.load(scaler_path)

        if venue_encoder_path:
            self.venue_encoder = joblib.load(venue_encoder_path)
        else:
            self.venue_encoder = None

        if matchup_encoder_path:
            self.matchup_encoder = joblib.load(matchup_encoder_path)
        else:
            self.matchup_encoder = None

        # Load feature columns
        with open(feature_columns_path, 'r') as f:
            self.feature_columns = [line.strip() for line in f.readlines()]

        # Continuous columns (all except categorical)
        self.categorical_cols = {'batter_encoded', 'bowler_encoded', 'venue_encoded', 'matchup_type_encoded'}
        self.continuous_cols = [c for c in self.feature_columns if c not in self.categorical_cols]

        # Class mapping (6 classes after remapping)
        self.class_to_outcome = {
            0: 'dot', 1: 'one', 2: 'two', 3: 'four', 4: 'six', 5: 'wicket'
        }

        # Sequence buffer for sliding window
        self.history_buffer = []

        print(f"Loaded LSTM v1 model with {len(self.feature_columns)} features, window_size={window_size}")

    def reset_sequence(self):
        """Reset sequence buffer (call at start of new innings)"""
        self.history_buffer = []

    def extract_features(self, state: MatchState) -> Dict:
        """
        Extract features from match state.
        Returns dict with continuous features and categorical IDs.

        NOTE: This must match XGBoostModelV2.extract_features() to ensure
        the LSTM model receives the same features it was trained on.
        """
        team_idx = state.current_team_idx
        striker = state.current_striker
        bowler = state.current_bowler
        wickets_in_hand = 10 - int(state.wickets[team_idx])

        # Basic state features
        features = {
            'inning_idx': state.innings,
            'score': int(state.runs[team_idx]),
            'wickets': int(state.wickets[team_idx]),
            'balls_bowled': state.balls,
            'run_rate': float(state.runs[team_idx]) / max(float(state.balls), 1) * 6,  # runs per over
            'wickets_ratio': float(state.wickets[team_idx]) / 10.0,
            'balls_ratio': float(state.balls) / 120.0,
            'wickets_in_hand': wickets_in_hand,
            'balls_remaining': state.balls_remaining,
            'is_powerplay': 1 if state.balls < 36 else 0,
            'is_middle_overs': 1 if 36 <= state.balls < 96 else 0,
            'is_death_overs': 1 if state.balls >= 96 else 0,
            'balls_in_over': state.balls % 6,
            'is_toss_winner': 0,  # Not tracked in simulation
            'is_batting_first': 1 if state.innings == 1 else 0,
        }

        # Player encoding (+1 to leave 0 for padding)
        try:
            features['batter_encoded'] = self.batter_encoder.transform([str(striker.player_id)])[0] + 1
        except:
            features['batter_encoded'] = 0  # Unknown player

        try:
            features['bowler_encoded'] = self.bowler_encoder.transform([str(bowler.player_id)])[0] + 1
        except:
            features['bowler_encoded'] = 0

        # Venue encoding
        if self.venue_encoder and hasattr(state, 'venue') and state.venue:
            try:
                features['venue_encoded'] = self.venue_encoder.transform([str(state.venue)])[0] + 1
            except:
                features['venue_encoded'] = 0
        else:
            features['venue_encoded'] = 0

        # Per-innings stats
        batsman_key = (team_idx, state.striker_idx)
        batter_innings_stats = state.batsman_stats.get(batsman_key, (0, 0))
        features['batter_runs_scored'] = batter_innings_stats[0]
        features['batter_balls_faced'] = batter_innings_stats[1]

        bowler_key = (state.bowling_team_idx, state.bowler_idx)
        bowler_balls_in_innings = state.bowler_balls.get(bowler_key, 0)
        features['bowler_balls_in_innings'] = bowler_balls_in_innings
        features['bowler_overs_in_innings'] = bowler_balls_in_innings / 6

        # Non-striker strike rate
        non_striker_key = (team_idx, state.non_striker_idx)
        non_striker_stats = state.batsman_stats.get(non_striker_key, (0, 0))
        if non_striker_stats[1] > 0:
            features['non_striker_sr'] = (non_striker_stats[0] / non_striker_stats[1]) * 100
        else:
            features['non_striker_sr'] = 0.0

        # Partnership
        features['partnership_runs'] = state.partnership_runs

        # Player stats from stats provider
        if self.stats_provider and hasattr(state, 'match_date'):
            match_date = state.match_date
            bat_stats = self.stats_provider.get_batting_stats(str(striker.player_id), match_date)
            bowl_stats = self.stats_provider.get_bowling_stats(str(bowler.player_id), match_date)
            h2h_stats = self.stats_provider.get_h2h_stats(str(striker.player_id), str(bowler.player_id), match_date)

            features['batsman_avg'] = bat_stats.get('avg', 25.0)
            features['batsman_sr'] = bat_stats.get('sr', 125.0)
            features['bowler_avg'] = bowl_stats.get('avg', 30.0)
            features['bowler_econ'] = bowl_stats.get('econ', 8.0)
            features['h2h_avg'] = h2h_stats.get('avg', 25.0)
            features['h2h_sr'] = h2h_stats.get('sr', 125.0)

            # Recent form stats (use same as career stats if not available separately)
            features['batsman_recent_avg'] = bat_stats.get('recent_avg', bat_stats.get('avg', 25.0))
            features['batsman_recent_sr'] = bat_stats.get('recent_sr', bat_stats.get('sr', 125.0))
            features['bowler_recent_avg'] = bowl_stats.get('recent_avg', bowl_stats.get('avg', 30.0))
            features['bowler_recent_econ'] = bowl_stats.get('recent_econ', bowl_stats.get('econ', 8.0))

            # Venue average score
            venue_avg = self.stats_provider.get_venue_avg_score(state.venue, match_date)
            features['venue_avg_score'] = venue_avg

            # Type-based stats (Tier 3)
            if hasattr(self.stats_provider, 'get_batting_vs_type_stats'):
                bat_vs_type = self.stats_provider.get_batting_vs_type_stats(str(striker.player_id), match_date)
                features['batter_avg_vs_pace'] = bat_vs_type.get('avg_vs_pace', 0.0)
                features['batter_sr_vs_pace'] = bat_vs_type.get('sr_vs_pace', 0.0)
                features['batter_avg_vs_spin'] = bat_vs_type.get('avg_vs_spin', 0.0)
                features['batter_sr_vs_spin'] = bat_vs_type.get('sr_vs_spin', 0.0)
            else:
                features['batter_avg_vs_pace'] = 0.0
                features['batter_sr_vs_pace'] = 0.0
                features['batter_avg_vs_spin'] = 0.0
                features['batter_sr_vs_spin'] = 0.0

            if hasattr(self.stats_provider, 'get_bowling_vs_hand_stats'):
                bowl_vs_hand = self.stats_provider.get_bowling_vs_hand_stats(str(bowler.player_id), match_date)
                features['bowler_avg_vs_lhb'] = bowl_vs_hand.get('avg_vs_lhb', 0.0)
                features['bowler_econ_vs_lhb'] = bowl_vs_hand.get('econ_vs_lhb', 0.0)
                features['bowler_avg_vs_rhb'] = bowl_vs_hand.get('avg_vs_rhb', 0.0)
                features['bowler_econ_vs_rhb'] = bowl_vs_hand.get('econ_vs_rhb', 0.0)
            else:
                features['bowler_avg_vs_lhb'] = 0.0
                features['bowler_econ_vs_lhb'] = 0.0
                features['bowler_avg_vs_rhb'] = 0.0
                features['bowler_econ_vs_rhb'] = 0.0
        else:
            features['batsman_avg'] = 25.0
            features['batsman_sr'] = 125.0
            features['bowler_avg'] = 30.0
            features['bowler_econ'] = 8.0
            features['h2h_avg'] = 25.0
            features['h2h_sr'] = 125.0
            features['batsman_recent_avg'] = 25.0
            features['batsman_recent_sr'] = 125.0
            features['bowler_recent_avg'] = 30.0
            features['bowler_recent_econ'] = 8.0
            features['venue_avg_score'] = 160.0
            features['batter_avg_vs_pace'] = 0.0
            features['batter_sr_vs_pace'] = 0.0
            features['batter_avg_vs_spin'] = 0.0
            features['batter_sr_vs_spin'] = 0.0
            features['bowler_avg_vs_lhb'] = 0.0
            features['bowler_econ_vs_lhb'] = 0.0
            features['bowler_avg_vs_rhb'] = 0.0
            features['bowler_econ_vs_rhb'] = 0.0

        # Player metadata features (Tier 1 and 2)
        if self.player_metadata:
            batter_meta = self.player_metadata.get_player_metadata(str(striker.player_id))
            bowler_meta = self.player_metadata.get_player_metadata(str(bowler.player_id))

            # Tier 1: Direct features (encoded)
            features['batter_hand'] = encode_batter_hand(batter_meta['batter_hand'])
            features['bowler_arm'] = encode_bowler_arm(bowler_meta['bowler_arm'])
            features['is_pace'] = encode_is_pace(bowler_meta['is_pace'])
            features['bowling_type'] = encode_bowling_type(bowler_meta['bowling_type'])

            # Ages
            batter_age = self.player_metadata.get_player_age(str(striker.player_id), state.match_date)
            bowler_age = self.player_metadata.get_player_age(str(bowler.player_id), state.match_date)
            features['batter_age'] = batter_age if batter_age is not None else 0
            features['bowler_age'] = bowler_age if bowler_age is not None else 0

            # Tier 2: Matchup features
            spin_matchup_advantage = self.player_metadata.get_spin_matchup_advantage(str(striker.player_id), str(bowler.player_id))
            same_arm_matchup = self.player_metadata.get_same_arm_matchup(str(striker.player_id), str(bowler.player_id))

            features['spin_matchup_advantage'] = spin_matchup_advantage
            features['same_arm_matchup'] = 1 if same_arm_matchup else (0 if same_arm_matchup is False else -1)
        else:
            features['batter_hand'] = 2  # unknown
            features['bowler_arm'] = 2   # unknown
            features['is_pace'] = 2      # unknown
            features['bowling_type'] = 8 # unknown
            features['batter_age'] = 0
            features['bowler_age'] = 0
            features['spin_matchup_advantage'] = 0
            features['same_arm_matchup'] = -1

        # Momentum features from history
        momentum = self._extract_momentum_features(state)
        features.update(momentum)

        # Pressure features
        features['dot_percentage_recent'] = momentum.get('dot_percentage_recent', 0.5)
        features['boundary_percentage_recent'] = momentum.get('boundary_percentage_recent', 0.15)

        # Chase features (2nd innings)
        if state.innings == 2 and state.target:
            runs_needed = state.target - int(state.runs[team_idx])
            features['chase_target'] = state.target
            features['run_rate_required'] = (runs_needed * 6 / max(state.balls_remaining, 1)) if state.balls_remaining > 0 else 0
            features['lead_gap'] = -runs_needed
            features['pressure_cooker_index'] = features['run_rate_required'] / max(wickets_in_hand, 1)
        else:
            features['chase_target'] = 0
            features['run_rate_required'] = 0
            features['lead_gap'] = int(state.runs[team_idx])
            features['pressure_cooker_index'] = 0

        # Matchup encoding
        if self.matchup_encoder and self.player_metadata:
            try:
                matchup_type = self.player_metadata.get_matchup_type(str(striker.player_id), str(bowler.player_id))
                features['matchup_type_encoded'] = self.matchup_encoder.transform([matchup_type])[0] + 1
            except:
                features['matchup_type_encoded'] = 0
        else:
            features['matchup_type_encoded'] = 0

        return features

    def _extract_momentum_features(self, state: MatchState) -> dict:
        """Extract momentum features from match history"""
        # Get current innings history
        current_innings_history = state.history[:state.history_idx]
        if len(current_innings_history) == 0:
            return {
                'last_5_balls_runs': 0,
                'last_10_balls_runs': 0,
                'last_30_balls_runs': 0,
                'balls_since_boundary': 0,
                'last_10_dots': 0,
                'dot_percentage_recent': 0.5,
                'boundary_percentage_recent': 0.15,
            }

        current_innings_mask = current_innings_history[:, 0] == state.innings
        innings_history = current_innings_history[current_innings_mask]

        if len(innings_history) == 0:
            return {
                'last_5_balls_runs': 0,
                'last_10_balls_runs': 0,
                'last_30_balls_runs': 0,
                'balls_since_boundary': 0,
                'last_10_dots': 0,
                'dot_percentage_recent': 0.5,
                'boundary_percentage_recent': 0.15,
            }

        runs_history = innings_history[:, 3]

        last_5 = runs_history[-5:] if len(runs_history) >= 5 else runs_history
        last_10 = runs_history[-10:] if len(runs_history) >= 10 else runs_history
        last_30 = runs_history[-30:] if len(runs_history) >= 30 else runs_history

        # Balls since boundary
        balls_since_boundary = 0
        for i in range(len(runs_history) - 1, -1, -1):
            if runs_history[i] >= 4:
                break
            balls_since_boundary += 1

        # Dots in last 10
        last_10_dots = int(np.sum(last_10 == 0)) if len(last_10) > 0 else 0

        # Percentages
        dot_pct = np.sum(last_10 == 0) / max(len(last_10), 1) if len(last_10) > 0 else 0.5
        boundary_pct = np.sum(last_30 >= 4) / max(len(last_30), 1) if len(last_30) > 0 else 0.15

        return {
            'last_5_balls_runs': int(np.sum(last_5)),
            'last_10_balls_runs': int(np.sum(last_10)),
            'last_30_balls_runs': int(np.sum(last_30)),
            'balls_since_boundary': balls_since_boundary,
            'last_10_dots': last_10_dots,
            'dot_percentage_recent': float(dot_pct),
            'boundary_percentage_recent': float(boundary_pct),
        }

    def predict_next_ball(self, features: Dict) -> Dict[str, float]:
        """Predict outcome probabilities using LSTM with sequence context"""
        import torch

        # Add current features to history buffer
        self.history_buffer.append(features)

        # Keep only last window_size entries
        if len(self.history_buffer) > self.window_size:
            self.history_buffer = self.history_buffer[-self.window_size:]

        # Prepare sequence tensors
        continuous_seq = []
        batter_seq = []
        bowler_seq = []
        venue_seq = []
        matchup_seq = []

        # Pad if needed
        pad_length = self.window_size - len(self.history_buffer)
        for _ in range(pad_length):
            continuous_seq.append(np.zeros(len(self.continuous_cols)))
            batter_seq.append(0)
            bowler_seq.append(0)
            venue_seq.append(0)
            matchup_seq.append(0)

        # Add features from buffer
        for feat in self.history_buffer:
            cont_features = [feat.get(col, 0.0) for col in self.continuous_cols]
            continuous_seq.append(cont_features)
            batter_seq.append(feat.get('batter_encoded', 0))
            bowler_seq.append(feat.get('bowler_encoded', 0))
            venue_seq.append(feat.get('venue_encoded', 0))
            matchup_seq.append(feat.get('matchup_type_encoded', 0))

        # Convert to tensors
        continuous = np.array(continuous_seq, dtype=np.float32)
        # Scale continuous features
        continuous = self.scaler.transform(continuous)

        continuous_tensor = torch.FloatTensor(continuous).unsqueeze(0).to(self.device)
        batter_tensor = torch.LongTensor(batter_seq).unsqueeze(0).to(self.device)
        bowler_tensor = torch.LongTensor(bowler_seq).unsqueeze(0).to(self.device)
        venue_tensor = torch.LongTensor(venue_seq).unsqueeze(0).to(self.device)
        matchup_tensor = torch.LongTensor(matchup_seq).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            logits = self.model(continuous_tensor, batter_tensor, bowler_tensor, venue_tensor, matchup_tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        # Convert to outcome dict
        outcome_probs = {
            'dot': 0.0, 'one': 0.0, 'two': 0.0, 'four': 0.0,
            'six': 0.0, 'wicket': 0.0, 'wide': 0.0, 'no_ball': 0.0
        }

        for class_idx, prob in enumerate(probs):
            outcome_name = self.class_to_outcome.get(class_idx)
            if outcome_name:
                outcome_probs[outcome_name] = float(prob)

        # Add extras
        outcome_probs['wide'] = 0.01
        outcome_probs['no_ball'] = 0.01

        # Normalize
        total = sum(outcome_probs.values())
        if total > 0:
            outcome_probs = {k: v/total for k, v in outcome_probs.items()}

        return outcome_probs


class MLPModelV1(PredictionModel):
    """Simple MLP-based ball outcome predictor.
    
    Uses same features as XGBoost v3 but with a neural network architecture.
    Faster inference than LSTM (no sequence state), good baseline NN model.
    """

    def __init__(self,
                 model_path: str,
                 batter_encoder_path: str,
                 bowler_encoder_path: str,
                 feature_columns_path: str,
                 scaler_path: str,
                 config_path: str,
                 stats_provider=None,
                 player_metadata=None,
                 matchup_encoder_path: str = None,
                 device: str = 'cpu'):
        import torch
        import torch.nn as nn
        import joblib
        import json

        self.device = torch.device(device)
        self.stats_provider = stats_provider
        self.player_metadata = player_metadata

        # Load model config
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Create model architecture
        self.model = self._create_model(
            n_features=config['n_features'],
            hidden_sizes=config['hidden_sizes'],
            dropout=config['dropout'],
            n_classes=config['n_classes']
        )

        # Load trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()

        # Load encoders and scaler
        self.batter_encoder = joblib.load(batter_encoder_path)
        self.bowler_encoder = joblib.load(bowler_encoder_path)
        self.scaler = joblib.load(scaler_path)

        if matchup_encoder_path:
            try:
                self.matchup_encoder = joblib.load(matchup_encoder_path)
            except:
                self.matchup_encoder = None
        else:
            self.matchup_encoder = None

        # Load feature columns
        with open(feature_columns_path, 'r') as f:
            self.feature_columns = [line.strip() for line in f.readlines()]

        # Class mapping (6 classes)
        self.class_to_outcome = {
            0: 'dot', 1: 'one', 2: 'two', 3: 'four', 4: 'six', 5: 'wicket'
        }

        print(f"Loaded MLP v1 model with {len(self.feature_columns)} features")

    def _create_model(self, n_features: int, hidden_sizes: list, dropout: float, n_classes: int):
        """Create MLP architecture matching training."""
        import torch.nn as nn

        layers = []
        in_size = n_features

        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            drop_rate = dropout if i < len(hidden_sizes) - 1 else dropout * 0.67
            layers.append(nn.Dropout(drop_rate))
            in_size = hidden_size

        feature_layers = nn.Sequential(*layers)
        classifier = nn.Linear(hidden_sizes[-1], n_classes)

        class MLPModel(nn.Module):
            def __init__(self, feature_layers, classifier):
                super().__init__()
                self.feature_layers = feature_layers
                self.classifier = classifier

            def forward(self, x):
                features = self.feature_layers(x)
                return self.classifier(features)

        return MLPModel(feature_layers, classifier)

    def extract_features(self, state: MatchState) -> Dict:
        """
        Extract features from match state.
        Same feature extraction as XGBoostModelV2 for consistency.
        """
        team_idx = state.current_team_idx
        striker = state.current_striker
        bowler = state.current_bowler
        wickets_in_hand = 10 - int(state.wickets[team_idx])

        # Basic state features
        features = {
            'inning_idx': state.innings,
            'score': int(state.runs[team_idx]),
            'wickets': int(state.wickets[team_idx]),
            'balls_bowled': state.balls,
            'run_rate': float(state.runs[team_idx]) / max(float(state.balls), 1) * 6,
            'wickets_ratio': float(state.wickets[team_idx]) / 10.0,
            'balls_ratio': float(state.balls) / 120.0,
            'wickets_in_hand': wickets_in_hand,
            'balls_remaining': state.balls_remaining,
            'is_powerplay': 1 if state.balls < 36 else 0,
            'is_middle_overs': 1 if 36 <= state.balls < 96 else 0,
            'is_death_overs': 1 if state.balls >= 96 else 0,
            'balls_in_over': state.balls % 6,
        }

        # Player encoding
        try:
            features['batter_encoded'] = self.batter_encoder.transform([str(striker.player_id)])[0]
        except:
            features['batter_encoded'] = -1

        try:
            features['bowler_encoded'] = self.bowler_encoder.transform([str(bowler.player_id)])[0]
        except:
            features['bowler_encoded'] = -1

        # Per-innings stats
        batsman_key = (team_idx, state.striker_idx)
        batter_innings_stats = state.batsman_stats.get(batsman_key, (0, 0))
        features['batter_runs_scored'] = batter_innings_stats[0]
        features['batter_balls_faced'] = batter_innings_stats[1]

        bowler_key = (state.bowling_team_idx, state.bowler_idx)
        bowler_balls_in_innings = state.bowler_balls.get(bowler_key, 0)
        features['bowler_balls_in_innings'] = bowler_balls_in_innings
        features['bowler_overs_in_innings'] = bowler_balls_in_innings / 6

        # Non-striker strike rate
        non_striker_key = (team_idx, state.non_striker_idx)
        non_striker_stats = state.batsman_stats.get(non_striker_key, (0, 0))
        if non_striker_stats[1] > 0:
            features['non_striker_sr'] = (non_striker_stats[0] / non_striker_stats[1]) * 100
        else:
            features['non_striker_sr'] = 0.0

        # Partnership
        features['partnership_runs'] = state.partnership_runs

        # Player stats from stats provider
        if self.stats_provider and hasattr(state, 'match_date'):
            match_date = state.match_date
            bat_stats = self.stats_provider.get_batting_stats(str(striker.player_id), match_date)
            bowl_stats = self.stats_provider.get_bowling_stats(str(bowler.player_id), match_date)
            h2h_stats = self.stats_provider.get_h2h_stats(str(striker.player_id), str(bowler.player_id), match_date)

            features['batsman_avg'] = bat_stats.get('avg', 25.0)
            features['batsman_sr'] = bat_stats.get('sr', 125.0)
            features['bowler_avg'] = bowl_stats.get('avg', 30.0)
            features['bowler_econ'] = bowl_stats.get('econ', 8.0)
            features['h2h_avg'] = h2h_stats.get('avg', 25.0)
            features['h2h_sr'] = h2h_stats.get('sr', 125.0)
            features['venue_avg_score'] = self.stats_provider.get_venue_avg_score(state.venue, match_date)

            # Type-based stats
            if hasattr(self.stats_provider, 'get_batting_vs_type_stats'):
                bat_vs_type = self.stats_provider.get_batting_vs_type_stats(str(striker.player_id), match_date)
                features['batter_avg_vs_pace'] = bat_vs_type.get('avg_vs_pace', 0.0)
                features['batter_sr_vs_pace'] = bat_vs_type.get('sr_vs_pace', 0.0)
                features['batter_avg_vs_spin'] = bat_vs_type.get('avg_vs_spin', 0.0)
                features['batter_sr_vs_spin'] = bat_vs_type.get('sr_vs_spin', 0.0)
            else:
                features['batter_avg_vs_pace'] = features['batter_sr_vs_pace'] = 0.0
                features['batter_avg_vs_spin'] = features['batter_sr_vs_spin'] = 0.0

            if hasattr(self.stats_provider, 'get_bowling_vs_hand_stats'):
                bowl_vs_hand = self.stats_provider.get_bowling_vs_hand_stats(str(bowler.player_id), match_date)
                features['bowler_avg_vs_lhb'] = bowl_vs_hand.get('avg_vs_lhb', 0.0)
                features['bowler_econ_vs_lhb'] = bowl_vs_hand.get('econ_vs_lhb', 0.0)
                features['bowler_avg_vs_rhb'] = bowl_vs_hand.get('avg_vs_rhb', 0.0)
                features['bowler_econ_vs_rhb'] = bowl_vs_hand.get('econ_vs_rhb', 0.0)
            else:
                features['bowler_avg_vs_lhb'] = features['bowler_econ_vs_lhb'] = 0.0
                features['bowler_avg_vs_rhb'] = features['bowler_econ_vs_rhb'] = 0.0

            # Team strength features (ELO + aggregated stats)
            bat_player_ids = [p.player_id for p in state.batting_lineup.players]
            bowl_player_ids = [p.player_id for p in state.bowling_lineup.players]
            features['striker_elo'] = self.stats_provider.get_batting_elo(str(striker.player_id), match_date)
            features['bowler_elo_rating'] = self.stats_provider.get_bowling_elo(str(bowler.player_id), match_date)
            batting_team_elo = self.stats_provider.get_team_batting_elo(bat_player_ids, match_date)
            bowling_team_elo = self.stats_provider.get_team_bowling_elo(bowl_player_ids, match_date)
            features['batting_team_elo'] = batting_team_elo
            features['bowling_team_elo'] = bowling_team_elo
            features['elo_diff'] = batting_team_elo - bowling_team_elo
            features.update(self.stats_provider.get_team_batting_strength(bat_player_ids, match_date))
            features.update(self.stats_provider.get_team_bowling_strength(bowl_player_ids, match_date))
        else:
            features['batsman_avg'] = 25.0
            features['batsman_sr'] = 125.0
            features['bowler_avg'] = 30.0
            features['bowler_econ'] = 8.0
            features['h2h_avg'] = 25.0
            features['h2h_sr'] = 125.0
            features['venue_avg_score'] = 160.0
            features['batter_avg_vs_pace'] = features['batter_sr_vs_pace'] = 0.0
            features['batter_avg_vs_spin'] = features['batter_sr_vs_spin'] = 0.0
            features['bowler_avg_vs_lhb'] = features['bowler_econ_vs_lhb'] = 0.0
            features['bowler_avg_vs_rhb'] = features['bowler_econ_vs_rhb'] = 0.0
            features['striker_elo'] = 1500.0
            features['bowler_elo_rating'] = 1500.0
            features['batting_team_elo'] = 16500.0
            features['bowling_team_elo'] = 16500.0
            features['elo_diff'] = 0.0
            features['team_batting_avg'] = 0.0
            features['team_batting_sr'] = 0.0
            features['team_bowling_avg'] = 0.0
            features['team_bowling_econ'] = 0.0

        # Player metadata features
        if self.player_metadata:
            batter_meta = self.player_metadata.get_player_metadata(str(striker.player_id))
            bowler_meta = self.player_metadata.get_player_metadata(str(bowler.player_id))

            features['batter_hand'] = encode_batter_hand(batter_meta['batter_hand'])
            features['bowler_arm'] = encode_bowler_arm(bowler_meta['bowler_arm'])
            features['is_pace'] = encode_is_pace(bowler_meta['is_pace'])
            features['bowling_type'] = encode_bowling_type(bowler_meta['bowling_type'])

            batter_age = self.player_metadata.get_player_age(str(striker.player_id), state.match_date)
            bowler_age = self.player_metadata.get_player_age(str(bowler.player_id), state.match_date)
            features['batter_age'] = batter_age if batter_age is not None else 0
            features['bowler_age'] = bowler_age if bowler_age is not None else 0

            spin_matchup = self.player_metadata.get_spin_matchup_advantage(str(striker.player_id), str(bowler.player_id))
            same_arm = self.player_metadata.get_same_arm_matchup(str(striker.player_id), str(bowler.player_id))
            features['spin_matchup_advantage'] = spin_matchup
            features['same_arm_matchup'] = 1 if same_arm else (0 if same_arm is False else -1)

            if self.matchup_encoder:
                try:
                    matchup_type = self.player_metadata.get_matchup_type(str(striker.player_id), str(bowler.player_id))
                    features['matchup_type_encoded'] = self.matchup_encoder.transform([matchup_type])[0]
                except:
                    features['matchup_type_encoded'] = 0
            else:
                features['matchup_type_encoded'] = 0
        else:
            features['batter_hand'] = 2
            features['bowler_arm'] = 2
            features['is_pace'] = 2
            features['bowling_type'] = 8
            features['batter_age'] = 0
            features['bowler_age'] = 0
            features['spin_matchup_advantage'] = 0
            features['same_arm_matchup'] = -1
            features['matchup_type_encoded'] = 0

        # Momentum features
        features.update(self._extract_momentum_features(state))

        # Chase features
        if state.innings == 2 and state.target:
            runs_needed = state.target - int(state.runs[team_idx])
            features['chase_target'] = state.target
            features['run_rate_required'] = (runs_needed * 6 / max(state.balls_remaining, 1))
            features['lead_gap'] = -runs_needed
            features['pressure_cooker_index'] = features['run_rate_required'] / max(wickets_in_hand, 1)
        else:
            features['chase_target'] = 0
            features['run_rate_required'] = 0
            features['lead_gap'] = int(state.runs[team_idx])
            features['pressure_cooker_index'] = 0

        return features

    def _extract_momentum_features(self, state: MatchState) -> dict:
        """Extract momentum features from match history."""
        if state.history_idx == 0:
            return {
                'last_5_balls_runs': 0, 'last_10_balls_runs': 0, 'last_30_balls_runs': 0,
                'balls_since_boundary': 0, 'last_10_dots': 0,
                'dot_percentage_recent': 0.5, 'boundary_percentage_recent': 0.15,
            }

        current_innings_history = state.history[:state.history_idx]
        current_innings_mask = current_innings_history[:, 0] == state.innings
        innings_history = current_innings_history[current_innings_mask]

        if len(innings_history) == 0:
            return {
                'last_5_balls_runs': 0, 'last_10_balls_runs': 0, 'last_30_balls_runs': 0,
                'balls_since_boundary': 0, 'last_10_dots': 0,
                'dot_percentage_recent': 0.5, 'boundary_percentage_recent': 0.15,
            }

        runs_history = innings_history[:, 3]
        last_5 = runs_history[-5:] if len(runs_history) >= 5 else runs_history
        last_10 = runs_history[-10:] if len(runs_history) >= 10 else runs_history
        last_30 = runs_history[-30:] if len(runs_history) >= 30 else runs_history

        balls_since_boundary = 0
        for i in range(len(runs_history) - 1, -1, -1):
            if runs_history[i] >= 4:
                break
            balls_since_boundary += 1

        last_10_dots = int(np.sum(last_10 == 0)) if len(last_10) > 0 else 0
        dot_pct = np.sum(last_10 == 0) / max(len(last_10), 1)
        boundary_pct = np.sum(last_30 >= 4) / max(len(last_30), 1)

        return {
            'last_5_balls_runs': int(np.sum(last_5)),
            'last_10_balls_runs': int(np.sum(last_10)),
            'last_30_balls_runs': int(np.sum(last_30)),
            'balls_since_boundary': balls_since_boundary,
            'last_10_dots': last_10_dots,
            'dot_percentage_recent': float(dot_pct),
            'boundary_percentage_recent': float(boundary_pct),
        }

    def predict_next_ball(self, features: Dict) -> Dict[str, float]:
        """Predict outcome probabilities using MLP."""
        import torch

        # Build feature vector in correct order
        feature_vector = []
        for col in self.feature_columns:
            feature_vector.append(features.get(col, 0.0))

        # Convert to tensor and scale
        X = np.array([feature_vector], dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X).to(self.device)

        # Forward pass
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        # Convert to outcome dict
        outcome_probs = {
            'dot': 0.0, 'one': 0.0, 'two': 0.0, 'four': 0.0,
            'six': 0.0, 'wicket': 0.0, 'wide': 0.01, 'no_ball': 0.01
        }

        for class_idx, prob in enumerate(probs):
            outcome_name = self.class_to_outcome.get(class_idx)
            if outcome_name:
                outcome_probs[outcome_name] = float(prob)

        # Normalize
        total = sum(outcome_probs.values())
        if total > 0:
            outcome_probs = {k: v/total for k, v in outcome_probs.items()}

        return outcome_probs


class MLPModelV2(PredictionModel):
    """MLP with embedding layers for player/venue/matchup.
    
    Uses same features as XGBoost v3 plus learnable embeddings for:
    - Batter ID
    - Bowler ID
    - Venue
    - Matchup type
    """

    def __init__(self,
                 model_path: str,
                 batter_encoder_path: str,
                 bowler_encoder_path: str,
                 continuous_columns_path: str,
                 categorical_columns_path: str,
                 scaler_path: str,
                 config_path: str,
                 stats_provider=None,
                 player_metadata=None,
                 matchup_encoder_path: str = None,
                 venue_encoder_path: str = None,
                 device: str = 'cpu'):
        import torch
        import torch.nn as nn
        import joblib
        import json

        self.device = torch.device(device)
        self.stats_provider = stats_provider
        self.player_metadata = player_metadata

        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Create model architecture
        self.model = self._create_model()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()

        # Load encoders and scaler
        self.batter_encoder = joblib.load(batter_encoder_path)
        self.bowler_encoder = joblib.load(bowler_encoder_path)
        self.scaler = joblib.load(scaler_path)

        if venue_encoder_path:
            try:
                self.venue_encoder = joblib.load(venue_encoder_path)
            except:
                self.venue_encoder = None
        else:
            self.venue_encoder = None

        if matchup_encoder_path:
            try:
                self.matchup_encoder = joblib.load(matchup_encoder_path)
            except:
                self.matchup_encoder = None
        else:
            self.matchup_encoder = None

        # Load feature columns
        with open(continuous_columns_path, 'r') as f:
            self.continuous_columns = [line.strip() for line in f.readlines()]

        with open(categorical_columns_path, 'r') as f:
            self.categorical_columns = json.load(f)

        self.class_to_outcome = {
            0: 'dot', 1: 'one', 2: 'two', 3: 'four', 4: 'six', 5: 'wicket'
        }

        print(f"Loaded MLP v2 model with {len(self.continuous_columns)} continuous features + embeddings")

    def _create_model(self):
        """Create MLP architecture matching training."""
        import torch
        import torch.nn as nn

        n_continuous = self.config['n_continuous']
        n_batters = self.config['n_batters']
        n_bowlers = self.config['n_bowlers']
        n_venues = self.config['n_venues']
        n_matchups = self.config['n_matchups']
        embed_dim_player = self.config['embed_dim_player']
        embed_dim_venue = self.config['embed_dim_venue']
        embed_dim_matchup = self.config['embed_dim_matchup']
        hidden_sizes = self.config['hidden_sizes']
        dropout = self.config['dropout']
        n_classes = self.config['n_classes']

        class MLPModelV2Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.batter_embed = nn.Embedding(n_batters, embed_dim_player)
                self.bowler_embed = nn.Embedding(n_bowlers, embed_dim_player)
                self.venue_embed = nn.Embedding(n_venues, embed_dim_venue)
                self.matchup_embed = nn.Embedding(n_matchups, embed_dim_matchup)

                total_embed_dim = 2 * embed_dim_player + embed_dim_venue + embed_dim_matchup
                input_size = n_continuous + total_embed_dim

                layers = []
                in_size = input_size

                for i, hidden_size in enumerate(hidden_sizes):
                    layers.append(nn.Linear(in_size, hidden_size))
                    layers.append(nn.LayerNorm(hidden_size))
                    layers.append(nn.GELU())
                    drop_rate = dropout if i < len(hidden_sizes) - 1 else dropout * 0.5
                    layers.append(nn.Dropout(drop_rate))
                    in_size = hidden_size

                self.feature_layers = nn.Sequential(*layers)
                self.classifier = nn.Linear(hidden_sizes[-1], n_classes)

            def forward(self, continuous, categorical):
                batter_emb = self.batter_embed(categorical['batter_encoded'])
                bowler_emb = self.bowler_embed(categorical['bowler_encoded'])
                venue_emb = self.venue_embed(categorical['venue_encoded'])
                matchup_emb = self.matchup_embed(categorical['matchup_type_encoded'])

                x = torch.cat([continuous, batter_emb, bowler_emb, venue_emb, matchup_emb], dim=-1)
                features = self.feature_layers(x)
                return self.classifier(features)

        return MLPModelV2Inner()

    def extract_features(self, state: MatchState) -> Dict:
        """Extract features matching XGBoost v3 feature set."""
        team_idx = state.current_team_idx
        striker = state.current_striker
        bowler = state.current_bowler
        wickets_in_hand = 10 - int(state.wickets[team_idx])

        features = {
            'inning_idx': state.innings,
            'score': int(state.runs[team_idx]),
            'wickets': int(state.wickets[team_idx]),
            'balls_bowled': state.balls,
            'run_rate': float(state.runs[team_idx]) / max(float(state.balls), 1) * 6,
            'wickets_ratio': float(state.wickets[team_idx]) / 10.0,
            'balls_ratio': float(state.balls) / 120.0,
            'wickets_in_hand': wickets_in_hand,
            'balls_remaining': state.balls_remaining,
            'over_idx': state.balls // 6,
            'ball_idx': state.balls,
            'balls_in_over': state.balls % 6,
            'is_powerplay': 1 if state.balls < 36 else 0,
            'is_middle_overs': 1 if 36 <= state.balls < 96 else 0,
            'is_death_overs': 1 if state.balls >= 96 else 0,
            # Batting order and toss
            'is_batting_first': 1 if state.innings == 1 else 0,
            'is_toss_winner': getattr(state, 'is_toss_winner', 0),
        }

        # Player encoding for embeddings
        try:
            features['batter_encoded'] = self.batter_encoder.transform([str(striker.player_id)])[0]
        except:
            features['batter_encoded'] = 0

        try:
            features['bowler_encoded'] = self.bowler_encoder.transform([str(bowler.player_id)])[0]
        except:
            features['bowler_encoded'] = 0

        # Venue encoding
        if self.venue_encoder and hasattr(state, 'venue'):
            try:
                features['venue_encoded'] = self.venue_encoder.transform([str(state.venue)])[0]
            except:
                features['venue_encoded'] = 0
        else:
            features['venue_encoded'] = 0

        # In-innings stats
        batsman_key = (team_idx, state.striker_idx)
        batter_innings_stats = state.batsman_stats.get(batsman_key, (0, 0))
        features['batter_runs_scored'] = batter_innings_stats[0]
        features['batter_balls_faced'] = batter_innings_stats[1]

        bowler_key = (state.bowling_team_idx, state.bowler_idx)
        bowler_balls_in_innings = state.bowler_balls.get(bowler_key, 0)
        features['bowler_balls_in_innings'] = bowler_balls_in_innings
        features['bowler_overs_in_innings'] = bowler_balls_in_innings / 6

        non_striker_key = (team_idx, state.non_striker_idx)
        non_striker_stats = state.batsman_stats.get(non_striker_key, (0, 0))
        features['non_striker_sr'] = (non_striker_stats[0] / non_striker_stats[1]) * 100 if non_striker_stats[1] > 0 else 0.0
        features['partnership_runs'] = state.partnership_runs

        # Player stats from StatsProvider (same as XGBoost)
        if self.stats_provider and hasattr(state, 'match_date'):
            match_date = state.match_date
            bat_stats = self.stats_provider.get_batting_stats(str(striker.player_id), match_date)
            bowl_stats = self.stats_provider.get_bowling_stats(str(bowler.player_id), match_date)
            h2h_stats = self.stats_provider.get_h2h_stats(str(striker.player_id), str(bowler.player_id), match_date)

            # Historical stats
            features['batsman_avg'] = bat_stats.get('avg', 25.0)
            features['batsman_sr'] = bat_stats.get('sr', 125.0)
            features['bowler_avg'] = bowl_stats.get('avg', 30.0)
            features['bowler_econ'] = bowl_stats.get('econ', 8.0)
            
            # Recent form stats (CRITICAL for player differentiation)
            features['batsman_recent_avg'] = bat_stats.get('recent_avg', bat_stats.get('avg', 25.0))
            features['batsman_recent_sr'] = bat_stats.get('recent_sr', bat_stats.get('sr', 125.0))
            features['bowler_recent_avg'] = bowl_stats.get('recent_avg', bowl_stats.get('avg', 30.0))
            features['bowler_recent_econ'] = bowl_stats.get('recent_econ', bowl_stats.get('econ', 8.0))
            
            # H2H stats
            features['h2h_avg'] = h2h_stats.get('avg', 25.0)
            features['h2h_sr'] = h2h_stats.get('sr', 125.0)
            features['venue_avg_score'] = self.stats_provider.get_venue_avg_score(state.venue, match_date)

            if hasattr(self.stats_provider, 'get_batting_vs_type_stats'):
                bat_vs_type = self.stats_provider.get_batting_vs_type_stats(str(striker.player_id), match_date)
                features['batter_avg_vs_pace'] = bat_vs_type.get('avg_vs_pace', 0.0)
                features['batter_sr_vs_pace'] = bat_vs_type.get('sr_vs_pace', 0.0)
                features['batter_avg_vs_spin'] = bat_vs_type.get('avg_vs_spin', 0.0)
                features['batter_sr_vs_spin'] = bat_vs_type.get('sr_vs_spin', 0.0)
            else:
                features['batter_avg_vs_pace'] = features['batter_sr_vs_pace'] = 0.0
                features['batter_avg_vs_spin'] = features['batter_sr_vs_spin'] = 0.0

            if hasattr(self.stats_provider, 'get_bowling_vs_hand_stats'):
                bowl_vs_hand = self.stats_provider.get_bowling_vs_hand_stats(str(bowler.player_id), match_date)
                features['bowler_avg_vs_lhb'] = bowl_vs_hand.get('avg_vs_lhb', 0.0)
                features['bowler_econ_vs_lhb'] = bowl_vs_hand.get('econ_vs_lhb', 0.0)
                features['bowler_avg_vs_rhb'] = bowl_vs_hand.get('avg_vs_rhb', 0.0)
                features['bowler_econ_vs_rhb'] = bowl_vs_hand.get('econ_vs_rhb', 0.0)
            else:
                features['bowler_avg_vs_lhb'] = features['bowler_econ_vs_lhb'] = 0.0
                features['bowler_avg_vs_rhb'] = features['bowler_econ_vs_rhb'] = 0.0

            # Team strength features (ELO + aggregated stats)
            bat_player_ids = [p.player_id for p in state.batting_lineup.players]
            bowl_player_ids = [p.player_id for p in state.bowling_lineup.players]
            features['striker_elo'] = self.stats_provider.get_batting_elo(str(striker.player_id), match_date)
            features['bowler_elo_rating'] = self.stats_provider.get_bowling_elo(str(bowler.player_id), match_date)
            batting_team_elo = self.stats_provider.get_team_batting_elo(bat_player_ids, match_date)
            bowling_team_elo = self.stats_provider.get_team_bowling_elo(bowl_player_ids, match_date)
            features['batting_team_elo'] = batting_team_elo
            features['bowling_team_elo'] = bowling_team_elo
            features['elo_diff'] = batting_team_elo - bowling_team_elo
            features.update(self.stats_provider.get_team_batting_strength(bat_player_ids, match_date))
            features.update(self.stats_provider.get_team_bowling_strength(bowl_player_ids, match_date))
        else:
            features['batsman_avg'] = 25.0
            features['batsman_sr'] = 125.0
            features['batsman_recent_avg'] = 25.0
            features['batsman_recent_sr'] = 125.0
            features['bowler_avg'] = 30.0
            features['bowler_econ'] = 8.0
            features['bowler_recent_avg'] = 30.0
            features['bowler_recent_econ'] = 8.0
            features['h2h_avg'] = 25.0
            features['h2h_sr'] = 125.0
            features['venue_avg_score'] = 160.0
            features['batter_avg_vs_pace'] = features['batter_sr_vs_pace'] = 0.0
            features['batter_avg_vs_spin'] = features['batter_sr_vs_spin'] = 0.0
            features['bowler_avg_vs_lhb'] = features['bowler_econ_vs_lhb'] = 0.0
            features['bowler_avg_vs_rhb'] = features['bowler_econ_vs_rhb'] = 0.0
            features['striker_elo'] = 1500.0
            features['bowler_elo_rating'] = 1500.0
            features['batting_team_elo'] = 16500.0
            features['bowling_team_elo'] = 16500.0
            features['elo_diff'] = 0.0
            features['team_batting_avg'] = 0.0
            features['team_batting_sr'] = 0.0
            features['team_bowling_avg'] = 0.0
            features['team_bowling_econ'] = 0.0

        # Player metadata
        if self.player_metadata:
            batter_meta = self.player_metadata.get_player_metadata(str(striker.player_id))
            bowler_meta = self.player_metadata.get_player_metadata(str(bowler.player_id))

            features['batter_hand'] = encode_batter_hand(batter_meta['batter_hand'])
            features['bowler_arm'] = encode_bowler_arm(bowler_meta['bowler_arm'])
            features['is_pace'] = encode_is_pace(bowler_meta['is_pace'])
            features['bowling_type'] = encode_bowling_type(bowler_meta['bowling_type'])

            batter_age = self.player_metadata.get_player_age(str(striker.player_id), state.match_date)
            bowler_age = self.player_metadata.get_player_age(str(bowler.player_id), state.match_date)
            features['batter_age'] = batter_age if batter_age is not None else 0
            features['bowler_age'] = bowler_age if bowler_age is not None else 0

            spin_matchup = self.player_metadata.get_spin_matchup_advantage(str(striker.player_id), str(bowler.player_id))
            same_arm = self.player_metadata.get_same_arm_matchup(str(striker.player_id), str(bowler.player_id))
            features['spin_matchup_advantage'] = spin_matchup
            features['same_arm_matchup'] = 1 if same_arm else (0 if same_arm is False else -1)

            if self.matchup_encoder:
                try:
                    matchup_type = self.player_metadata.get_matchup_type(str(striker.player_id), str(bowler.player_id))
                    features['matchup_type_encoded'] = self.matchup_encoder.transform([matchup_type])[0]
                except:
                    features['matchup_type_encoded'] = 0
            else:
                features['matchup_type_encoded'] = 0
        else:
            features['batter_hand'] = 2
            features['bowler_arm'] = 2
            features['is_pace'] = 2
            features['bowling_type'] = 8
            features['batter_age'] = 0
            features['bowler_age'] = 0
            features['spin_matchup_advantage'] = 0
            features['same_arm_matchup'] = -1
            features['matchup_type_encoded'] = 0

        # Momentum features
        features.update(self._extract_momentum_features(state))

        # Chase features
        if state.innings == 2 and state.target:
            runs_needed = state.target - int(state.runs[team_idx])
            features['chase_target'] = state.target
            features['run_rate_required'] = (runs_needed * 6 / max(state.balls_remaining, 1))
            features['lead_gap'] = -runs_needed
            features['pressure_cooker_index'] = features['run_rate_required'] / max(wickets_in_hand, 1)
        else:
            features['chase_target'] = 0
            features['run_rate_required'] = 0
            features['lead_gap'] = int(state.runs[team_idx])
            features['pressure_cooker_index'] = 0

        return features

    def _extract_momentum_features(self, state: MatchState) -> dict:
        if state.history_idx == 0:
            return {
                'last_5_balls_runs': 0, 'last_10_balls_runs': 0, 'last_30_balls_runs': 0,
                'balls_since_boundary': 0, 'last_10_dots': 0,
                'dot_percentage_recent': 0.5, 'boundary_percentage_recent': 0.15,
            }

        current_innings_history = state.history[:state.history_idx]
        current_innings_mask = current_innings_history[:, 0] == state.innings
        innings_history = current_innings_history[current_innings_mask]

        if len(innings_history) == 0:
            return {
                'last_5_balls_runs': 0, 'last_10_balls_runs': 0, 'last_30_balls_runs': 0,
                'balls_since_boundary': 0, 'last_10_dots': 0,
                'dot_percentage_recent': 0.5, 'boundary_percentage_recent': 0.15,
            }

        runs_history = innings_history[:, 3]
        last_5 = runs_history[-5:] if len(runs_history) >= 5 else runs_history
        last_10 = runs_history[-10:] if len(runs_history) >= 10 else runs_history
        last_30 = runs_history[-30:] if len(runs_history) >= 30 else runs_history

        balls_since_boundary = 0
        for i in range(len(runs_history) - 1, -1, -1):
            if runs_history[i] >= 4:
                break
            balls_since_boundary += 1

        last_10_dots = int(np.sum(last_10 == 0)) if len(last_10) > 0 else 0
        dot_pct = np.sum(last_10 == 0) / max(len(last_10), 1)
        boundary_pct = np.sum(last_30 >= 4) / max(len(last_30), 1)

        return {
            'last_5_balls_runs': int(np.sum(last_5)),
            'last_10_balls_runs': int(np.sum(last_10)),
            'last_30_balls_runs': int(np.sum(last_30)),
            'balls_since_boundary': balls_since_boundary,
            'last_10_dots': last_10_dots,
            'dot_percentage_recent': float(dot_pct),
            'boundary_percentage_recent': float(boundary_pct),
        }

    def predict_next_ball(self, features: Dict) -> Dict[str, float]:
        import torch

        # Build continuous feature vector
        continuous_vector = []
        for col in self.continuous_columns:
            continuous_vector.append(features.get(col, 0.0))

        X_cont = np.array([continuous_vector], dtype=np.float32)
        X_cont = np.nan_to_num(X_cont, nan=0.0, posinf=0.0, neginf=0.0)
        X_cont = self.scaler.transform(X_cont)
        X_cont_tensor = torch.FloatTensor(X_cont).to(self.device)

        # Build categorical features
        categorical = {
            'batter_encoded': torch.LongTensor([min(features.get('batter_encoded', 0), self.config['n_batters']-1)]).to(self.device),
            'bowler_encoded': torch.LongTensor([min(features.get('bowler_encoded', 0), self.config['n_bowlers']-1)]).to(self.device),
            'venue_encoded': torch.LongTensor([min(features.get('venue_encoded', 0), self.config['n_venues']-1)]).to(self.device),
            'matchup_type_encoded': torch.LongTensor([min(features.get('matchup_type_encoded', 0), self.config['n_matchups']-1)]).to(self.device),
        }

        # Forward pass with temperature scaling
        with torch.no_grad():
            logits = self.model(X_cont_tensor, categorical)
            # Temperature scaling: <1 sharpens predictions, >1 smooths
            temperature = self.config.get('temperature', 1.0)
            probs = torch.softmax(logits / temperature, dim=-1).cpu().numpy()[0]

        outcome_probs = {
            'dot': 0.0, 'one': 0.0, 'two': 0.0, 'four': 0.0,
            'six': 0.0, 'wicket': 0.0, 'wide': 0.01, 'no_ball': 0.01
        }

        for class_idx, prob in enumerate(probs):
            outcome_name = self.class_to_outcome.get(class_idx)
            if outcome_name:
                outcome_probs[outcome_name] = float(prob)

        total = sum(outcome_probs.values())
        if total > 0:
            outcome_probs = {k: v/total for k, v in outcome_probs.items()}

        return outcome_probs


class TransformerModelV1(PredictionModel):
    """Transformer-based ball outcome predictor with FULL INNINGS context (up to 120 balls).

    Key difference from LSTM: Uses full innings history instead of 10-ball sliding window.
    This allows the model to attend to patterns from anywhere in the innings.

    Supports both PyTorch and MLX backends:
    - PyTorch (default): Universal compatibility (CUDA, MPS, CPU)
    - MLX (--mlx flag): Optimized for Apple Silicon (unified memory, Metal GPU)
    """

    def __init__(self,
                 model_path: str,
                 batter_encoder_path: str,
                 bowler_encoder_path: str,
                 feature_columns_path: str,
                 scaler_path: str,
                 config_path: str,
                 stats_provider=None,
                 player_metadata=None,
                 matchup_encoder_path: str = None,
                 venue_encoder_path: str = None,
                 max_seq_len: int = 120,
                 device: str = 'cpu',
                 use_mlx: bool = False):
        import joblib
        import json

        self.max_seq_len = max_seq_len
        self.stats_provider = stats_provider
        self.player_metadata = player_metadata
        self.use_mlx = use_mlx

        # Load model config
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Load encoders (shared between backends)
        self.batter_encoder = joblib.load(batter_encoder_path)
        self.bowler_encoder = joblib.load(bowler_encoder_path)
        self.scaler = joblib.load(scaler_path)

        if venue_encoder_path:
            self.venue_encoder = joblib.load(venue_encoder_path)
        else:
            self.venue_encoder = None

        if matchup_encoder_path:
            self.matchup_encoder = joblib.load(matchup_encoder_path)
        else:
            self.matchup_encoder = None

        # Load feature columns
        with open(feature_columns_path, 'r') as f:
            self.feature_columns = [line.strip() for line in f.readlines()]

        # Continuous columns (all except categorical)
        self.categorical_cols = {'batter_encoded', 'bowler_encoded', 'venue_encoded', 'matchup_type_encoded'}
        self.continuous_cols = [c for c in self.feature_columns if c not in self.categorical_cols]

        # Class mapping (6 classes after remapping)
        self.class_to_outcome = {
            0: 'dot', 1: 'one', 2: 'two', 3: 'four', 4: 'six', 5: 'wicket'
        }

        # Full innings history buffer (not sliding window)
        self.history_buffer = []

        # Initialize backend
        if use_mlx:
            self._init_mlx(model_path)
        else:
            self._init_pytorch(model_path, device)

    def _init_pytorch(self, model_path: str, device: str):
        """Initialize PyTorch backend (universal compatibility)."""
        import torch

        self.device = torch.device(device)

        # Import and create model architecture
        try:
            from transformer_v1 import TransformerBallPredictor
        except ImportError:
            from scripts.transformer_v1 import TransformerBallPredictor

        self.model = TransformerBallPredictor(
            n_continuous=self.config['n_continuous'],
            n_batters=self.config['n_batters'],
            n_bowlers=self.config['n_bowlers'],
            n_venues=self.config['n_venues'],
            n_matchups=self.config['n_matchups'],
            embed_dim_player=self.config['embed_dim_player'],
            embed_dim_venue=self.config['embed_dim_venue'],
            embed_dim_matchup=self.config['embed_dim_matchup'],
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            nhead=self.config['nhead'],
            dim_feedforward=self.config['dim_feedforward'],
            dropout=self.config['dropout'],
            max_seq_len=self.config.get('max_seq_len', 120),
            n_classes=self.config['n_classes']
        )

        # Load trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()

        print(f"Loaded Transformer v1 (PyTorch) with {len(self.feature_columns)} features, max_seq_len={self.max_seq_len}")

    def _init_mlx(self, model_path: str):
        """Initialize MLX backend (Apple Silicon optimized)."""
        import platform
        from pathlib import Path

        # Verify Apple Silicon
        if platform.system() != 'Darwin' or platform.machine() != 'arm64':
            raise RuntimeError("MLX backend requires Apple Silicon Mac (M1/M2/M3/M4)")

        try:
            import mlx.core as mx
            from transformer_mlx import (
                TransformerBallPredictorMLX,
                load_mlx_weights,
                convert_pytorch_to_mlx
            )
        except ImportError as e:
            raise ImportError(f"MLX not available. Install with: pip install mlx safetensors\nError: {e}")

        # Create MLX model
        self.model = TransformerBallPredictorMLX(
            n_continuous=self.config['n_continuous'],
            n_batters=self.config['n_batters'],
            n_bowlers=self.config['n_bowlers'],
            n_venues=self.config['n_venues'],
            n_matchups=self.config['n_matchups'],
            embed_dim_player=self.config['embed_dim_player'],
            embed_dim_venue=self.config['embed_dim_venue'],
            embed_dim_matchup=self.config['embed_dim_matchup'],
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            nhead=self.config['nhead'],
            dim_feedforward=self.config['dim_feedforward'],
            dropout=0.0,  # No dropout at inference
            max_seq_len=self.config.get('max_seq_len', 120),
            n_classes=self.config['n_classes']
        )

        # Try to load MLX weights first, fall back to converting PyTorch weights
        mlx_path = model_path.replace('.pt', '_mlx.safetensors')
        mlx_npz_path = model_path.replace('.pt', '_mlx.npz')

        if Path(mlx_path).exists():
            load_mlx_weights(self.model, mlx_path)
        elif Path(mlx_npz_path).exists():
            load_mlx_weights(self.model, mlx_npz_path)
        elif Path(model_path).exists():
            # Convert PyTorch weights on-the-fly
            print("Converting PyTorch weights to MLX format...")
            import torch
            pt_state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
            convert_pytorch_to_mlx(pt_state_dict, self.model)
        else:
            raise FileNotFoundError(f"No model weights found at {model_path} or MLX variants")

        print(f"Loaded Transformer v1 (MLX, unified memory) with {len(self.feature_columns)} features")

    def reset_sequence(self):
        """Reset sequence buffer (call at start of new innings)"""
        self.history_buffer = []

    def extract_features(self, state: MatchState) -> Dict:
        """
        Extract features from match state.
        Returns dict with continuous features and categorical IDs.

        NOTE: This is identical to LSTMModelV1.extract_features() to ensure
        the Transformer model receives the same features it was trained on.
        """
        team_idx = state.current_team_idx
        striker = state.current_striker
        bowler = state.current_bowler
        wickets_in_hand = 10 - int(state.wickets[team_idx])

        # Basic state features
        features = {
            'inning_idx': state.innings,
            'score': int(state.runs[team_idx]),
            'wickets': int(state.wickets[team_idx]),
            'balls_bowled': state.balls,
            'run_rate': float(state.runs[team_idx]) / max(float(state.balls), 1) * 6,
            'wickets_ratio': float(state.wickets[team_idx]) / 10.0,
            'balls_ratio': float(state.balls) / 120.0,
            'wickets_in_hand': wickets_in_hand,
            'balls_remaining': state.balls_remaining,
            'is_powerplay': 1 if state.balls < 36 else 0,
            'is_middle_overs': 1 if 36 <= state.balls < 96 else 0,
            'is_death_overs': 1 if state.balls >= 96 else 0,
            'balls_in_over': state.balls % 6,
            'is_toss_winner': 0,
            'is_batting_first': 1 if state.innings == 1 else 0,
        }

        # Player encoding (+1 to leave 0 for padding)
        try:
            features['batter_encoded'] = self.batter_encoder.transform([str(striker.player_id)])[0] + 1
        except:
            features['batter_encoded'] = 0

        try:
            features['bowler_encoded'] = self.bowler_encoder.transform([str(bowler.player_id)])[0] + 1
        except:
            features['bowler_encoded'] = 0

        # Venue encoding
        if self.venue_encoder and hasattr(state, 'venue') and state.venue:
            try:
                features['venue_encoded'] = self.venue_encoder.transform([str(state.venue)])[0] + 1
            except:
                features['venue_encoded'] = 0
        else:
            features['venue_encoded'] = 0

        # Per-innings stats
        batsman_key = (team_idx, state.striker_idx)
        batter_innings_stats = state.batsman_stats.get(batsman_key, (0, 0))
        features['batter_runs_scored'] = batter_innings_stats[0]
        features['batter_balls_faced'] = batter_innings_stats[1]

        bowler_key = (state.bowling_team_idx, state.bowler_idx)
        bowler_balls_in_innings = state.bowler_balls.get(bowler_key, 0)
        features['bowler_balls_in_innings'] = bowler_balls_in_innings
        features['bowler_overs_in_innings'] = bowler_balls_in_innings / 6

        # Non-striker strike rate
        non_striker_key = (team_idx, state.non_striker_idx)
        non_striker_stats = state.batsman_stats.get(non_striker_key, (0, 0))
        if non_striker_stats[1] > 0:
            features['non_striker_sr'] = (non_striker_stats[0] / non_striker_stats[1]) * 100
        else:
            features['non_striker_sr'] = 0.0

        # Partnership
        features['partnership_runs'] = state.partnership_runs

        # Player stats from stats provider
        if self.stats_provider and hasattr(state, 'match_date'):
            match_date = state.match_date
            bat_stats = self.stats_provider.get_batting_stats(str(striker.player_id), match_date)
            bowl_stats = self.stats_provider.get_bowling_stats(str(bowler.player_id), match_date)
            h2h_stats = self.stats_provider.get_h2h_stats(str(striker.player_id), str(bowler.player_id), match_date)

            features['batsman_avg'] = bat_stats.get('avg', 25.0)
            features['batsman_sr'] = bat_stats.get('sr', 125.0)
            features['bowler_avg'] = bowl_stats.get('avg', 30.0)
            features['bowler_econ'] = bowl_stats.get('econ', 8.0)
            features['h2h_avg'] = h2h_stats.get('avg', 25.0)
            features['h2h_sr'] = h2h_stats.get('sr', 125.0)

            features['batsman_recent_avg'] = bat_stats.get('recent_avg', bat_stats.get('avg', 25.0))
            features['batsman_recent_sr'] = bat_stats.get('recent_sr', bat_stats.get('sr', 125.0))
            features['bowler_recent_avg'] = bowl_stats.get('recent_avg', bowl_stats.get('avg', 30.0))
            features['bowler_recent_econ'] = bowl_stats.get('recent_econ', bowl_stats.get('econ', 8.0))

            venue_avg = self.stats_provider.get_venue_avg_score(state.venue, match_date)
            features['venue_avg_score'] = venue_avg

            # Type-based stats
            if hasattr(self.stats_provider, 'get_batting_vs_type_stats'):
                bat_vs_type = self.stats_provider.get_batting_vs_type_stats(str(striker.player_id), match_date)
                features['batter_avg_vs_pace'] = bat_vs_type.get('avg_vs_pace', 0.0)
                features['batter_sr_vs_pace'] = bat_vs_type.get('sr_vs_pace', 0.0)
                features['batter_avg_vs_spin'] = bat_vs_type.get('avg_vs_spin', 0.0)
                features['batter_sr_vs_spin'] = bat_vs_type.get('sr_vs_spin', 0.0)
            else:
                features['batter_avg_vs_pace'] = 0.0
                features['batter_sr_vs_pace'] = 0.0
                features['batter_avg_vs_spin'] = 0.0
                features['batter_sr_vs_spin'] = 0.0

            if hasattr(self.stats_provider, 'get_bowling_vs_hand_stats'):
                bowl_vs_hand = self.stats_provider.get_bowling_vs_hand_stats(str(bowler.player_id), match_date)
                features['bowler_avg_vs_lhb'] = bowl_vs_hand.get('avg_vs_lhb', 0.0)
                features['bowler_econ_vs_lhb'] = bowl_vs_hand.get('econ_vs_lhb', 0.0)
                features['bowler_avg_vs_rhb'] = bowl_vs_hand.get('avg_vs_rhb', 0.0)
                features['bowler_econ_vs_rhb'] = bowl_vs_hand.get('econ_vs_rhb', 0.0)
            else:
                features['bowler_avg_vs_lhb'] = 0.0
                features['bowler_econ_vs_lhb'] = 0.0
                features['bowler_avg_vs_rhb'] = 0.0
                features['bowler_econ_vs_rhb'] = 0.0
        else:
            features['batsman_avg'] = 25.0
            features['batsman_sr'] = 125.0
            features['bowler_avg'] = 30.0
            features['bowler_econ'] = 8.0
            features['h2h_avg'] = 25.0
            features['h2h_sr'] = 125.0
            features['batsman_recent_avg'] = 25.0
            features['batsman_recent_sr'] = 125.0
            features['bowler_recent_avg'] = 30.0
            features['bowler_recent_econ'] = 8.0
            features['venue_avg_score'] = 160.0
            features['batter_avg_vs_pace'] = 0.0
            features['batter_sr_vs_pace'] = 0.0
            features['batter_avg_vs_spin'] = 0.0
            features['batter_sr_vs_spin'] = 0.0
            features['bowler_avg_vs_lhb'] = 0.0
            features['bowler_econ_vs_lhb'] = 0.0
            features['bowler_avg_vs_rhb'] = 0.0
            features['bowler_econ_vs_rhb'] = 0.0

        # Player metadata features
        if self.player_metadata:
            batter_meta = self.player_metadata.get_player_metadata(str(striker.player_id))
            bowler_meta = self.player_metadata.get_player_metadata(str(bowler.player_id))

            features['batter_hand'] = encode_batter_hand(batter_meta['batter_hand'])
            features['bowler_arm'] = encode_bowler_arm(bowler_meta['bowler_arm'])
            features['is_pace'] = encode_is_pace(bowler_meta['is_pace'])
            features['bowling_type'] = encode_bowling_type(bowler_meta['bowling_type'])

            batter_age = self.player_metadata.get_player_age(str(striker.player_id), state.match_date)
            bowler_age = self.player_metadata.get_player_age(str(bowler.player_id), state.match_date)
            features['batter_age'] = batter_age if batter_age is not None else 0
            features['bowler_age'] = bowler_age if bowler_age is not None else 0

            spin_matchup_advantage = self.player_metadata.get_spin_matchup_advantage(str(striker.player_id), str(bowler.player_id))
            same_arm_matchup = self.player_metadata.get_same_arm_matchup(str(striker.player_id), str(bowler.player_id))

            features['spin_matchup_advantage'] = spin_matchup_advantage
            features['same_arm_matchup'] = 1 if same_arm_matchup else (0 if same_arm_matchup is False else -1)
        else:
            features['batter_hand'] = 2
            features['bowler_arm'] = 2
            features['is_pace'] = 2
            features['bowling_type'] = 8
            features['batter_age'] = 0
            features['bowler_age'] = 0
            features['spin_matchup_advantage'] = 0
            features['same_arm_matchup'] = -1

        # Momentum features from history
        momentum = self._extract_momentum_features(state)
        features.update(momentum)

        # Pressure features
        features['dot_percentage_recent'] = momentum.get('dot_percentage_recent', 0.5)
        features['boundary_percentage_recent'] = momentum.get('boundary_percentage_recent', 0.15)

        # Chase features (2nd innings)
        if state.innings == 2 and state.target:
            runs_needed = state.target - int(state.runs[team_idx])
            features['chase_target'] = state.target
            features['run_rate_required'] = (runs_needed * 6 / max(state.balls_remaining, 1)) if state.balls_remaining > 0 else 0
            features['lead_gap'] = -runs_needed
            features['pressure_cooker_index'] = features['run_rate_required'] / max(wickets_in_hand, 1)
        else:
            features['chase_target'] = 0
            features['run_rate_required'] = 0
            features['lead_gap'] = int(state.runs[team_idx])
            features['pressure_cooker_index'] = 0

        # Matchup encoding
        if self.matchup_encoder and self.player_metadata:
            try:
                matchup_type = self.player_metadata.get_matchup_type(str(striker.player_id), str(bowler.player_id))
                features['matchup_type_encoded'] = self.matchup_encoder.transform([matchup_type])[0] + 1
            except:
                features['matchup_type_encoded'] = 0
        else:
            features['matchup_type_encoded'] = 0

        return features

    def _extract_momentum_features(self, state: MatchState) -> dict:
        """Extract momentum features from match history"""
        current_innings_history = state.history[:state.history_idx]
        if len(current_innings_history) == 0:
            return {
                'last_5_balls_runs': 0,
                'last_10_balls_runs': 0,
                'last_30_balls_runs': 0,
                'balls_since_boundary': 0,
                'last_10_dots': 0,
                'dot_percentage_recent': 0.5,
                'boundary_percentage_recent': 0.15,
            }

        current_innings_mask = current_innings_history[:, 0] == state.innings
        innings_history = current_innings_history[current_innings_mask]

        if len(innings_history) == 0:
            return {
                'last_5_balls_runs': 0,
                'last_10_balls_runs': 0,
                'last_30_balls_runs': 0,
                'balls_since_boundary': 0,
                'last_10_dots': 0,
                'dot_percentage_recent': 0.5,
                'boundary_percentage_recent': 0.15,
            }

        runs_history = innings_history[:, 3]

        last_5 = runs_history[-5:] if len(runs_history) >= 5 else runs_history
        last_10 = runs_history[-10:] if len(runs_history) >= 10 else runs_history
        last_30 = runs_history[-30:] if len(runs_history) >= 30 else runs_history

        # Balls since boundary
        balls_since_boundary = 0
        for i in range(len(runs_history) - 1, -1, -1):
            if runs_history[i] >= 4:
                break
            balls_since_boundary += 1

        last_10_dots = int(np.sum(last_10 == 0)) if len(last_10) > 0 else 0
        dot_pct = np.sum(last_10 == 0) / max(len(last_10), 1) if len(last_10) > 0 else 0.5
        boundary_pct = np.sum(last_30 >= 4) / max(len(last_30), 1) if len(last_30) > 0 else 0.15

        return {
            'last_5_balls_runs': int(np.sum(last_5)),
            'last_10_balls_runs': int(np.sum(last_10)),
            'last_30_balls_runs': int(np.sum(last_30)),
            'balls_since_boundary': balls_since_boundary,
            'last_10_dots': last_10_dots,
            'dot_percentage_recent': float(dot_pct),
            'boundary_percentage_recent': float(boundary_pct),
        }

    def predict_next_ball(self, features: Dict) -> Dict[str, float]:
        """
        Predict outcome probabilities using Transformer with FULL INNINGS context.

        Key difference from LSTM: No sliding window - keeps ALL previous balls
        in the innings (up to max_seq_len=120).

        Dispatches to PyTorch or MLX backend based on use_mlx flag.
        """
        # Add current features to history buffer - NO TRUNCATION to sliding window
        self.history_buffer.append(features)

        # Only cap at max_seq_len if innings exceeds this (shouldn't happen in T20)
        if len(self.history_buffer) > self.max_seq_len:
            self.history_buffer = self.history_buffer[-self.max_seq_len:]

        # Dispatch to appropriate backend
        if self.use_mlx:
            return self._predict_mlx()
        else:
            return self._predict_pytorch()

    def _predict_pytorch(self) -> Dict[str, float]:
        """PyTorch prediction backend."""
        import torch

        # Prepare sequence tensors from FULL history
        continuous_seq = []
        batter_seq = []
        bowler_seq = []
        venue_seq = []
        matchup_seq = []

        for feat in self.history_buffer:
            cont_features = [feat.get(col, 0.0) for col in self.continuous_cols]
            continuous_seq.append(cont_features)
            batter_seq.append(feat.get('batter_encoded', 0))
            bowler_seq.append(feat.get('bowler_encoded', 0))
            venue_seq.append(feat.get('venue_encoded', 0))
            matchup_seq.append(feat.get('matchup_type_encoded', 0))

        # Convert to tensors (variable length: 1 to 120 balls)
        continuous = np.array(continuous_seq, dtype=np.float32)
        continuous = np.nan_to_num(continuous, nan=0.0, posinf=0.0, neginf=0.0)
        continuous = self.scaler.transform(continuous)

        continuous_tensor = torch.FloatTensor(continuous).unsqueeze(0).to(self.device)
        batter_tensor = torch.LongTensor(batter_seq).unsqueeze(0).to(self.device)
        bowler_tensor = torch.LongTensor(bowler_seq).unsqueeze(0).to(self.device)
        venue_tensor = torch.LongTensor(venue_seq).unsqueeze(0).to(self.device)
        matchup_tensor = torch.LongTensor(matchup_seq).unsqueeze(0).to(self.device)

        # Forward pass with full innings context
        with torch.no_grad():
            logits = self.model(continuous_tensor, batter_tensor, bowler_tensor, venue_tensor, matchup_tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        return self._build_outcome_dict(probs)

    def _predict_mlx(self) -> Dict[str, float]:
        """MLX prediction backend (Apple Silicon optimized)."""
        import mlx.core as mx

        # Prepare sequence arrays from FULL history
        continuous_seq = []
        batter_seq = []
        bowler_seq = []
        venue_seq = []
        matchup_seq = []

        for feat in self.history_buffer:
            cont_features = [feat.get(col, 0.0) for col in self.continuous_cols]
            continuous_seq.append(cont_features)
            batter_seq.append(int(feat.get('batter_encoded', 0)))
            bowler_seq.append(int(feat.get('bowler_encoded', 0)))
            venue_seq.append(int(feat.get('venue_encoded', 0)))
            matchup_seq.append(int(feat.get('matchup_type_encoded', 0)))

        # Convert to numpy then MLX arrays (unified memory - instant)
        continuous = np.array(continuous_seq, dtype=np.float32)
        continuous = np.nan_to_num(continuous, nan=0.0, posinf=0.0, neginf=0.0)
        continuous = self.scaler.transform(continuous)

        # Convert to MLX arrays (add batch dimension)
        continuous_mx = mx.array(continuous[np.newaxis, :, :])
        batter_mx = mx.array(np.array(batter_seq, dtype=np.int32)[np.newaxis, :])
        bowler_mx = mx.array(np.array(bowler_seq, dtype=np.int32)[np.newaxis, :])
        venue_mx = mx.array(np.array(venue_seq, dtype=np.int32)[np.newaxis, :])
        matchup_mx = mx.array(np.array(matchup_seq, dtype=np.int32)[np.newaxis, :])

        # Forward pass
        logits = self.model(continuous_mx, batter_mx, bowler_mx, venue_mx, matchup_mx)
        probs = mx.softmax(logits, axis=-1)

        # Force computation and convert to numpy
        mx.eval(probs)
        probs_np = np.array(probs)[0]

        return self._build_outcome_dict(probs_np)

    def _build_outcome_dict(self, probs: np.ndarray) -> Dict[str, float]:
        """Convert model probabilities to outcome dictionary."""
        outcome_probs = {
            'dot': 0.0, 'one': 0.0, 'two': 0.0, 'four': 0.0,
            'six': 0.0, 'wicket': 0.0, 'wide': 0.0, 'no_ball': 0.0
        }

        for class_idx, prob in enumerate(probs):
            outcome_name = self.class_to_outcome.get(class_idx)
            if outcome_name:
                outcome_probs[outcome_name] = float(prob)

        # Add extras
        outcome_probs['wide'] = 0.01
        outcome_probs['no_ball'] = 0.01

        # Normalize
        total = sum(outcome_probs.values())
        if total > 0:
            outcome_probs = {k: v/total for k, v in outcome_probs.items()}

        return outcome_probs


class LLMModelV1(PredictionModel):
    """Fine-tuned Qwen 1.5-1.8B LLM for cricket outcome prediction.

    Uses structured text prompts to predict ball-by-ball outcomes.
    Requires GPU for reasonable inference speed.

    The model was trained on ~1.9M ball examples from CricSheet data
    using LoRA fine-tuning on the Qwen 1.5-1.8B base model.
    """

    # Token mappings (matching llm-finetune training)
    OUTCOMES = ["0", "1", "2", "3", "4", "6", "WICKET"]
    EXTRA_BASE = 40
    # Explicit dict to avoid Python scoping issues with class-level comprehensions
    OUTCOME2TOK = {
        "0": "<|extra_40|>",
        "1": "<|extra_41|>",
        "2": "<|extra_42|>",
        "3": "<|extra_43|>",
        "4": "<|extra_44|>",
        "6": "<|extra_45|>",
        "WICKET": "<|extra_46|>"
    }

    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        """Initialize the LLM model.

        Args:
            checkpoint_path: Path to the LoRA checkpoint directory
            device: 'cuda' for GPU (required for reasonable speed) or 'cpu'
        """
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        # Check GPU availability
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("LLM model requires GPU but CUDA not available. "
                             "Please run on a machine with CUDA GPU.")

        self.device = device
        self.torch = torch

        print(f"Loading LLM base model (Qwen/Qwen1.5-1.8B)...")

        # Load base model with appropriate settings
        if device == 'cuda':
            self.base_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen1.5-1.8B",
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen1.5-1.8B",
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
            self.base_model = self.base_model.to(device)

        # Load tokenizer - try checkpoint first, fall back to base model
        print(f"Loading tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
            print(f"  Loaded tokenizer from checkpoint")
        except Exception:
            # Tokenizer not saved in checkpoint, load from base model
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B", trust_remote_code=True)
            print(f"  Loaded tokenizer from base model")

        # Ensure special tokens are set up (outcome tokens for prediction)
        special_tokens = list(self.OUTCOME2TOK.values())
        if not all(tok in self.tokenizer.get_vocab() for tok in special_tokens):
            self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
            print(f"  Added {len(special_tokens)} special outcome tokens")

        self.base_model.resize_token_embeddings(len(self.tokenizer))

        # Load LoRA adapter
        print(f"Loading LoRA adapter from {checkpoint_path}...")
        self.model = PeftModel.from_pretrained(self.base_model, checkpoint_path)
        self.model.eval()

        # Get outcome token IDs
        self.outcome_token_ids = self.tokenizer.convert_tokens_to_ids(
            list(self.OUTCOME2TOK.values())
        )

        # State for prompt generation (tracks ball history within a match)
        self.recent_balls = []  # Track last 5 ball outcomes
        self.partnership_runs = 0
        self.partnership_balls = 0
        self.bowler_runs_conceded = {}  # Track runs conceded by each bowler

        print(f"✓ LLM model loaded successfully on {device}")

    def reset_match_state(self):
        """Reset tracking state for a new match simulation."""
        self.recent_balls = []
        self.partnership_runs = 0
        self.partnership_balls = 0
        self.bowler_runs_conceded = {}

    def update_after_ball(self, outcome: str, runs: int, bowler_idx: int):
        """Update internal tracking after a ball is bowled.

        Args:
            outcome: The outcome string ('dot', 'one', 'wicket', etc.)
            runs: Runs scored on this ball
            bowler_idx: Index of the current bowler
        """
        # Update recent balls
        if outcome == 'wicket':
            self.recent_balls.append("W")
            self.partnership_runs = 0
            self.partnership_balls = 0
        else:
            self.recent_balls.append(str(runs))
            self.partnership_runs += runs

        self.partnership_balls += 1

        # Keep only last 5 balls
        if len(self.recent_balls) > 5:
            self.recent_balls.pop(0)

        # Track bowler runs
        if bowler_idx not in self.bowler_runs_conceded:
            self.bowler_runs_conceded[bowler_idx] = 0
        self.bowler_runs_conceded[bowler_idx] += runs

    def extract_features(self, state: MatchState) -> str:
        """Generate structured prompt from MatchState.

        Format matches the training data:
        "Over.Ball: Score/Wickets [Need X@RRR] | Recent: B1-B2-B3-B4-B5 | P:Runs@Ballsb(RR) | Bowler(EconX.X) vs Batter(status) | Phase | Teams, Venue"
        """
        team_idx = state.current_team_idx

        # 1. SITUATION - Over.Ball: Score/Wickets [Need X@RRR for 2nd innings]
        next_ball = (state.balls % 6) + 1
        current_over = state.balls // 6
        over_ball = f"{current_over}.{next_ball}"

        if state.innings == 2:
            # Second innings - include target info
            target = int(state.runs[1 - team_idx]) + 1
            runs_needed = target - int(state.runs[team_idx])
            balls_remaining = 120 - state.balls
            if balls_remaining > 0:
                rrr = round(runs_needed / (balls_remaining / 6), 1)
            else:
                rrr = 0.0
            situation = f"{over_ball}: {int(state.runs[team_idx])}/{int(state.wickets[team_idx])} Need{runs_needed}@{rrr}"
        else:
            situation = f"{over_ball}: {int(state.runs[team_idx])}/{int(state.wickets[team_idx])}"

        # 2. RECENT - Last 5 balls (padded with "-" at start)
        if len(self.recent_balls) >= 5:
            recent = self.recent_balls[-5:]
        else:
            recent = ["-"] * (5 - len(self.recent_balls)) + self.recent_balls
        recent_str = f"Recent: {'-'.join(recent)}"

        # 3. PARTNERSHIP
        if self.partnership_balls > 0:
            p_rate = round((self.partnership_runs / self.partnership_balls) * 6, 1)
            partnership_str = f"P:{self.partnership_runs}@{self.partnership_balls}b({p_rate}rr)"
        else:
            partnership_str = "P:0@0b(0.0rr)"

        # 4. PLAYERS
        bowler = state.current_bowler
        striker = state.current_striker

        # Get last names
        bowler_name = bowler.name.split()[-1] if " " in bowler.name else bowler.name
        striker_name = striker.name.split()[-1] if " " in striker.name else striker.name

        # Get batter stats from state.batsman_stats
        batsman_key = (team_idx, state.striker_idx)
        batter_stats = state.batsman_stats.get(batsman_key, (0, 0))
        batter_runs, batter_balls = batter_stats[0], batter_stats[1]
        sr = int((batter_runs / batter_balls * 100)) if batter_balls > 0 else 0

        if batter_balls == 0:
            batsman_str = f"{striker_name}(new,0 Runs @ 0 SR)"
        elif batter_balls < 10:
            batsman_str = f"{striker_name}(new,{batter_runs} Runs @ {sr} SR)"
        else:
            batsman_str = f"{striker_name}(set,{batter_runs} Runs @ {sr} SR)"

        # Bowler economy
        bowler_key = (state.bowling_team_idx, state.bowler_idx)
        bowler_balls = state.bowler_balls.get(bowler_key, 0)
        bowler_runs = self.bowler_runs_conceded.get(state.bowler_idx, 0)
        if bowler_balls > 0:
            economy = round((bowler_runs / bowler_balls) * 6, 1)
            bowler_str = f"{bowler_name}(Econ{economy})"
        else:
            bowler_str = f"{bowler_name}(Econ0.0)"

        players_str = f"{bowler_str} vs {batsman_str}"

        # 5. CONTEXT - Phase + Teams + Venue
        if current_over < 6:
            phase_str = f"PP {over_ball}"
        elif current_over >= 16:
            phase_str = f"Death {over_ball}"
        else:
            phase_str = None

        # Venue abbreviation
        venue = state.venue if hasattr(state, 'venue') else "Unknown"
        venue_words = venue.replace(" Stadium", "").replace(" Cricket Ground", "").replace(" International", "").split()
        if len(venue_words) >= 3:
            venue_short = ''.join([w[0].upper() for w in venue_words[:3]])
        elif len(venue_words) == 2:
            venue_short = venue_words[0][:2].upper() + venue_words[1][0].upper()
        else:
            venue_short = venue[:3].upper()

        teams = f"{state.batting_team} vs {state.bowling_team}"

        if phase_str:
            context = f"{phase_str} | {teams}, {venue_short}"
        else:
            context = f"{teams}, {venue_short}"

        return f"{situation} | {recent_str} | {partnership_str} | {players_str} | {context}"

    def predict_next_ball(self, features) -> Dict[str, float]:
        """Predict outcome distribution from prompt.

        Args:
            features: The prompt string from extract_features()

        Returns:
            Dict mapping outcome names to probabilities
        """
        prompt = features  # features is the prompt string

        # Tokenize with trailing space (matches training format)
        prompt_with_space = prompt.rstrip() + " "
        inputs = self.tokenizer(prompt_with_space, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with self.torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]
            probs = self.torch.softmax(logits, dim=-1)
            outcome_probs = probs[self.outcome_token_ids].cpu().tolist()

        # Map 7 LLM outcomes → 8 match_prediction outcomes
        llm_probs = dict(zip(self.OUTCOMES, outcome_probs))

        result = {
            'dot': llm_probs["0"],
            'one': llm_probs["1"],
            'two': llm_probs["2"] + llm_probs["3"],  # Merge "3" into "two"
            'four': llm_probs["4"],
            'six': llm_probs["6"],
            'wicket': llm_probs["WICKET"],
            'wide': 0.015,      # Fixed small probability
            'no_ball': 0.005,   # Fixed small probability
        }

        # Renormalize to sum to 1.0
        total = sum(result.values())
        return {k: v/total for k, v in result.items()}


# Simulation Engine
@dataclass
class SimulationConfig:
    """Configuration for match simulation"""
    n_simulations: int = 100
    parallel: bool = True
    n_workers: Optional[int] = None  # None = use all CPUs
    random_seed: Optional[int] = None
    verbose: bool = False

@dataclass 
class BallResult:
    """Result of a single ball"""
    innings: int
    over: int
    ball: int
    outcome: Outcome
    runs: int
    striker_idx: int
    bowler_idx: int
    team_runs: int
    team_wickets: int

@dataclass
class InningsResult:
    """Result of a single innings"""
    batting_team: str
    bowling_team: str
    total_runs: int
    total_wickets: int
    total_balls: int
    run_rate: float
    
    # Batsman performances: {player_idx: (runs, balls, fours, sixes)}
    batting_card: Dict[int, Tuple[int, int, int, int]]
    
    # Bowler performances: {player_idx: (balls, runs, wickets)}
    bowling_card: Dict[int, Tuple[int, int, int]]
    
    # Ball-by-ball data
    balls: List[BallResult]

@dataclass
class MatchResult:
    """Result of a single match simulation"""
    match_id: str
    team1: str
    team2: str
    winner: str
    margin: str
    
    # Innings results
    innings: List[InningsResult]
    
    # Quick access
    team1_score: int
    team1_wickets: int
    team2_score: int
    team2_wickets: int

class SimulationEngine:
    """Orchestrates cricket match simulations"""
    
    def __init__(self, model: PredictionModel, rules: Optional[T20Rules] = None):
        self.model = model
        self.rules = rules or T20Rules(RandomBowlerSelector())

    def simulate_match(self, initial_state: MatchState, match_id: str = "sim") -> MatchResult:
        """Simulate a complete match"""
        state = initial_state.copy()
        innings_results = []
        
        # Simulate both innings
        for innings_num in [1, 2]:
            innings_result = self._simulate_innings(state)
            innings_results.append(innings_result)
            
            if state.is_match_over():
                break
            
            # Start second innings
            if innings_num == 1:
                state.start_new_innings()
        
        # Determine result
        team1_score = int(state.runs[0])
        team1_wickets = int(state.wickets[0])
        team2_score = int(state.runs[1]) 
        team2_wickets = int(state.wickets[1])
        
        if team1_score > team2_score:
            winner = state.team1
            margin = f"{team1_score - team2_score} runs"
        elif team2_score > team1_score:
            winner = state.team2
            margin = f"{10 - team2_wickets} wickets"
        else:
            winner = "Tie"
            margin = "Tied"
        
        # In sim_v1_2.py, inside the simulate_match method

        # --- DEBUGGING TRIPWIRE START ---
        # This block will crash the program if a non-integer is found,
        # telling us exactly when the data corruption happens.
        if not all(isinstance(val, int) for val in [team1_score, team1_wickets, team2_score, team2_wickets]):
            
            t1s_type = type(team1_score).__name__
            t1w_type = type(team1_wickets).__name__
            t2s_type = type(team2_score).__name__
            t2w_type = type(team2_wickets).__name__
            
            error_message = (
                f"\n\nFATAL: Data type corruption detected in simulation result!\n"
                f"----------------------------------------------------------\n"
                f"Match ID: {match_id}\n"
                f"  Team 1 Score:   {team1_score} (Type: {t1s_type})\n"
                f"  Team 1 Wickets: {team1_wickets} (Type: {t1w_type})\n"
                f"  Team 2 Score:   {team2_score} (Type: {t2s_type})\n"
                f"  Team 2 Wickets: {team2_wickets} (Type: {t2w_type})\n"
                f"----------------------------------------------------------\n"
                f"This error was triggered intentionally to pinpoint the source of the bug.\n"
            )
            raise TypeError(error_message)
        # --- DEBUGGING TRIPWIRE END ---
        
        return MatchResult(
            match_id=match_id,
            team1=state.team1,
            team2=state.team2,
            winner=winner,
            margin=margin,
            innings=innings_results,
            team1_score=team1_score,
            team1_wickets=team1_wickets,
            team2_score=team2_score,
            team2_wickets=team2_wickets
        )
    
    def _simulate_innings(self, state: MatchState) -> InningsResult:
        """Simulate a single innings"""
        balls = []
        batting_card = {}
        bowling_card = {}
        
        start_runs = int(state.runs[state.current_team_idx])
        start_wickets = int(state.wickets[state.current_team_idx])
        
        while not state.is_innings_over():
            # Simulate next ball
            outcome, runs = self.rules.simulate_ball(state, self.model)
            
            # Record ball result
            ball_result = BallResult(
                innings=state.innings,
                over=state.balls // 6,
                ball=state.balls % 6,
                outcome=outcome,
                runs=runs,
                striker_idx=state.striker_idx,
                bowler_idx=state.bowler_idx,
                team_runs=int(state.runs[state.current_team_idx]),
                team_wickets=int(state.wickets[state.current_team_idx])
            )
            balls.append(ball_result)
            
            # Update batting card
            if outcome != Outcome.WIDE:
                key = state.striker_idx
                runs_scored = runs if outcome not in [Outcome.WIDE, Outcome.NO_BALL] else 0
                balls_faced = 1 if outcome not in [Outcome.WIDE, Outcome.NO_BALL] else 0
                fours = 1 if outcome == Outcome.FOUR else 0
                sixes = 1 if outcome == Outcome.SIX else 0
                
                if key in batting_card:
                    prev = batting_card[key]
                    batting_card[key] = (
                        prev[0] + runs_scored,
                        prev[1] + balls_faced,
                        prev[2] + fours,
                        prev[3] + sixes
                    )
                else:
                    batting_card[key] = (runs_scored, balls_faced, fours, sixes)
            
            # Update bowling card
            if outcome not in [Outcome.WIDE, Outcome.NO_BALL]:
                key = state.bowler_idx
                wickets = 1 if outcome == Outcome.WICKET else 0
                
                if key in bowling_card:
                    prev = bowling_card[key]
                    bowling_card[key] = (prev[0] + 1, prev[1] + runs, prev[2] + wickets)
                else:
                    bowling_card[key] = (1, runs, wickets)
        
        # Calculate innings summary
        total_runs = int(state.runs[state.current_team_idx]) - start_runs
        total_wickets = int(state.wickets[state.current_team_idx]) - start_wickets
        total_balls = state.balls
        run_rate = (total_runs / total_balls * 6) if total_balls > 0 else 0.0
        
        return InningsResult(
            batting_team=state.batting_team,
            bowling_team=state.bowling_team,
            total_runs=total_runs,
            total_wickets=total_wickets,
            total_balls=total_balls,
            run_rate=run_rate,
            batting_card=batting_card,
            bowling_card=bowling_card,
            balls=balls
        )
    
    def simulate_multiple(self, initial_state: MatchState, 
                         config: SimulationConfig) -> List[MatchResult]:
        """Run multiple simulations"""
        if config.verbose:
            print(f"Running {config.n_simulations} simulations...")
            start_time = time.time()
        
        if config.parallel and config.n_simulations > 1:
            results = self._simulate_parallel(initial_state, config)
        else:
            results = self._simulate_sequential(initial_state, config)
        
        if config.verbose:
            elapsed = time.time() - start_time
            print(f"Completed in {elapsed:.2f}s ({config.n_simulations/elapsed:.1f} sims/sec)")
        
        return results
    
    def _simulate_sequential(self, initial_state: MatchState, 
                           config: SimulationConfig) -> List[MatchResult]:
        """Sequential simulation"""
        results = []
        for i in range(config.n_simulations):
            if config.random_seed:
                random.seed(config.random_seed + i)
                np.random.seed(config.random_seed + i)
            
            result = self.simulate_match(initial_state, f"sim_{i}")
            results.append(result)
        
        return results
    
    def _simulate_parallel(self, initial_state: MatchState,
                          config: SimulationConfig) -> List[MatchResult]:
        """Parallel simulation using multiprocessing"""
        n_workers = config.n_workers or cpu_count()
        
        # Create tasks
        tasks = []
        for i in range(config.n_simulations):
            seed = (config.random_seed + i) if config.random_seed else None
            tasks.append((initial_state, f"sim_{i}", seed))
        
        # Run in parallel
        with Pool(n_workers) as pool:
            results = pool.starmap(self._simulate_match_with_seed, tasks)
        
        return results
    
    def _simulate_match_with_seed(self, state: MatchState, match_id: str, 
                                 seed: Optional[int]) -> MatchResult:
        """Simulate match with specific seed (for parallel execution)"""
        if seed:
            random.seed(seed)
            np.random.seed(seed)
        
        return self.simulate_match(state, match_id)

class ResultAggregator:
    """Aggregates results from multiple simulations"""

    @staticmethod
    def aggregate(results: List[MatchResult]) -> Dict[str, Any]:
        """Create summary statistics from simulation results"""
        n_sims = len(results)
        
        if n_sims == 0:
            return {}
        
        # Extract data
        team1 = results[0].team1
        team2 = results[0].team2
        
        # Win probabilities
        team1_wins = sum(1 for r in results if r.winner == team1)
        team2_wins = sum(1 for r in results if r.winner == team2)
        ties = sum(1 for r in results if r.winner == "Tie")
        
        # Score distributions
        team1_scores = [r.team1_score for r in results]
        team2_scores = [r.team2_score for r in results]
        
        # Wickets distributions
        team1_wickets = [r.team1_wickets for r in results]
        team2_wickets = [r.team2_wickets for r in results]
        
        return {
            'n_simulations': n_sims,
            'team1': team1,
            'team2': team2,
            
            # Win probabilities
            'win_probability': {
                team1: team1_wins / n_sims,
                team2: team2_wins / n_sims,
                'tie': ties / n_sims
            },
            
            # Score statistics
            'score_stats': {
                team1: {
                    'mean': np.mean([int(s) for s in team1_scores]),
                    'std': np.std([int(s) for s in team1_scores]),
                    'min': np.min([int(s) for s in team1_scores]),
                    'max': np.max([int(s) for s in team1_scores]),
                    'percentiles': {
                        '25': np.percentile([int(s) for s in team1_scores], 25),
                        '50': np.percentile([int(s) for s in team1_scores], 50),
                        '75': np.percentile([int(s) for s in team1_scores], 75)
                    }
                },
                team2: {
                    'mean': np.mean([int(s) for s in team2_scores]),
                    'std': np.std([int(s) for s in team2_scores]),
                    'min': np.min([int(s) for s in team2_scores]),
                    'max': np.max([int(s) for s in team2_scores]),
                    'percentiles': {
                        '25': np.percentile([int(s) for s in team2_scores], 25),
                        '50': np.percentile([int(s) for s in team2_scores], 50),
                        '75': np.percentile([int(s) for s in team2_scores], 75)
                    }
                }
            },
            
            # Wickets statistics
            'wicket_stats': {
                team1: {
                    'mean': np.mean(team1_wickets),
                    'distribution': dict(zip(*np.unique(team1_wickets, return_counts=True)))
                },
                team2: {
                    'mean': np.mean(team2_wickets),
                    'distribution': dict(zip(*np.unique(team2_wickets, return_counts=True)))
                }
            },
            
            # Raw results for further analysis
            'raw_results': results
        }

# Example usage
if __name__ == "__main__":
    # Initialize components
    model = DummyModel()  # Use this for testing without XGBoost model
    
    # To use XGBoost model (if you have the trained model files):
    # model = XGBoostModel(
    #     model_path='models/gradient_boosting_model.pkl',
    #     batter_encoder_path='models/batter_encoder.pkl', 
    #     bowler_encoder_path='models/bowler_encoder.pkl'
    # )
    
    rules = T20Rules(RandomBowlerSelector())
    engine = SimulationEngine(model, rules)

    # Create player objects
    india_players = [
        Player("rohit_sharma", "Rohit Sharma", "India", "batsman"),
        Player("shubman_gill", "Shubman Gill", "India", "batsman"),
        Player("virat_kohli", "Virat Kohli", "India", "batsman"),
        Player("suryakumar_yadav", "Suryakumar Yadav", "India", "batsman"),
        Player("hardik_pandya", "Hardik Pandya", "India", "allrounder"),
        Player("ravindra_jadeja", "Ravindra Jadeja", "India", "allrounder"),
        Player("ms_dhoni", "MS Dhoni", "India", "wicketkeeper"),
        Player("ravichandran_ashwin", "R Ashwin", "India", "bowler"),
        Player("mohammed_shami", "Mohammed Shami", "India", "bowler"),
        Player("jasprit_bumrah", "Jasprit Bumrah", "India", "bowler"),
        Player("yuzvendra_chahal", "Yuzvendra Chahal", "India", "bowler"),
    ]

    australia_players = [
        Player("david_warner", "David Warner", "Australia", "batsman"),
        Player("travis_head", "Travis Head", "Australia", "batsman"),
        Player("steve_smith", "Steve Smith", "Australia", "batsman"),
        Player("glenn_maxwell", "Glenn Maxwell", "Australia", "allrounder"),
        Player("marcus_stoinis", "Marcus Stoinis", "Australia", "allrounder"),
        Player("tim_david", "Tim David", "Australia", "batsman"),
        Player("matthew_wade", "Matthew Wade", "Australia", "wicketkeeper"),
        Player("pat_cummins", "Pat Cummins", "Australia", "bowler"),
        Player("mitchell_starc", "Mitchell Starc", "Australia", "bowler"),
        Player("adam_zampa", "Adam Zampa", "Australia", "bowler"),
        Player("josh_hazlewood", "Josh Hazlewood", "Australia", "bowler"),
    ]

    # Create lineups
    india_lineup = TeamLineup("India", india_players)
    australia_lineup = TeamLineup("Australia", australia_players)

    # Create match state with full details
    state = MatchState(
        team1_lineup=india_lineup,
        team2_lineup=australia_lineup,
        batting_first="India",
        venue="MCG",
        match_date=datetime(2024, 12, 25)
    )


    # Run simulations
    config = SimulationConfig(
        n_simulations=1000,
        parallel=True,
        verbose=True,
        random_seed=42
    )

    results = engine.simulate_multiple(state, config)

    # Aggregate results
    summary = ResultAggregator.aggregate(results)

    print(f"\nWin Probabilities:")
    for team, prob in summary['win_probability'].items():
        print(f"  {team}: {prob:.2%}")

    print(f"\nScore Predictions:")
    for team, stats in summary['score_stats'].items():
        print(f"  {team}: {stats['mean']:.1f} ± {stats['std']:.1f}")
        print(f"    Range: {stats['min']}-{stats['max']}")
        print(f"    Median: {stats['percentiles']['50']:.0f}")
    
    print(f"\nWickets Analysis:")
    for team, stats in summary['wicket_stats'].items():
        print(f"  {team}: Average {stats['mean']:.1f} wickets")
        
    # Example: Analyze a single match in detail
    print(f"\n--- Single Match Example ---")
    single_match = engine.simulate_match(state, "example_match")
    
    print(f"Result: {single_match.winner} by {single_match.margin}")
    print(f"Scores: {single_match.team1} {single_match.team1_score}/{single_match.team1_wickets}")
    print(f"        {single_match.team2} {single_match.team2_score}/{single_match.team2_wickets}")
    
    # Show batting performances from first innings
    if single_match.innings:
        first_innings = single_match.innings[0]
        print(f"\n{first_innings.batting_team} Batting:")
        for player_idx, (runs, balls, fours, sixes) in first_innings.batting_card.items():
            if balls > 0:  # Only show players who batted
                sr = (runs / balls * 100) if balls > 0 else 0
                print(f"  Player {player_idx}: {runs} runs ({balls} balls, {fours}x4, {sixes}x6) SR: {sr:.1f}")