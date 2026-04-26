from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from abc import ABC, abstractmethod
import random
from multiprocessing import Pool, cpu_count
import time

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
class MatchState:
    # Match setup
    team1: str
    team2: str
    batting_first: str
    venue: str
    
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

    # Ball-by-ball history (for features)
    # Each row: [innings, over, ball, runs, wicket, batting_team_idx, striker_idx, bowling_team_idx, bowler_idx]
    history: np.ndarray = field(default_factory=lambda: np.zeros((240, 9)))
    history_idx: int = 0

    # Tracking: (team_idx, player_idx) -> value
    bowler_balls: Dict[Tuple[int, int], int] = field(default_factory=dict)
    batsman_stats: Dict[Tuple[int, int], Tuple[int, int]] = field(default_factory=dict)

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
            if idx == self.last_bowler_idx:
                continue
            
            # Check over limit (max 24 balls = 4 overs in T20)
            if self.bowler_balls.get((bowling_team, idx), 0) >= 24:
                continue
            
            available.append(idx)
        
        return available
    
    def update(self, outcome: Outcome, runs: int = 0):
        """Update state after a ball"""
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
        
        # Update batsman stats (fixed: removed non-existent BYE, LEG_BYE)
        if outcome != Outcome.WIDE:
            batsman_key = (self.current_team_idx, self.striker_idx)
            stats = self.batsman_stats.get(batsman_key, (0, 0))
            self.batsman_stats[batsman_key] = (stats[0] + runs, stats[1] + 1)
        
        # Handle wicket
        if outcome == Outcome.WICKET:
            self.wickets[self.current_team_idx] += 1
            # New batsman comes in (next in order)
            self.striker_idx = max(self.striker_idx, self.non_striker_idx) + 1
        
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
        # Note: We keep bowler_balls and batsman_stats as they track both teams
    
    def copy(self):
        """Efficient copy for parallel simulations"""
        new_state = MatchState(
            team1=self.team1,
            team2=self.team2,
            batting_first=self.batting_first,
            venue=self.venue
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
        new_state.history = self.history.copy()
        new_state.history_idx = self.history_idx
        new_state.bowler_balls = self.bowler_balls.copy()
        new_state.batsman_stats = self.batsman_stats.copy()
        
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

class XGBoostModel(PredictionModel):
    def __init__(self, model_path: str, batter_encoder_path: str, bowler_encoder_path: str):
        import joblib
        self.model = joblib.load(model_path)
        self.batter_encoder = joblib.load(batter_encoder_path)
        self.bowler_encoder = joblib.load(bowler_encoder_path)
        
        # Map model output classes to our Outcome enum
        self.class_to_outcome = {
            0: 'dot', 1: 'one', 2: 'two', 3: 'four',
            4: 'four', 5: 'six', 6: 'six', 7: 'wicket'
        }

    def extract_features(self, state: MatchState) -> np.ndarray:
        team_idx = state.current_team_idx
        
        # Calculate features matching your GBM training
        inning_idx = state.innings
        score = int(state.runs[team_idx])
        wickets = int(state.wickets[team_idx])  # Fixed: was state.runs[team_idx]
        balls_bowled = state.balls
        
        # For batter/bowler encoding, use placeholders for now
        batter_encoded = state.striker_idx  # Simplified
        bowler_encoded = state.bowler_idx   # Simplified
        
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
            'wide': 0.0,
            'no_ball': 0.0
        }

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
                    'mean': np.mean(team1_scores),
                    'std': np.std(team1_scores),
                    'min': np.min(team1_scores),
                    'max': np.max(team1_scores),
                    'percentiles': {
                        '25': np.percentile(team1_scores, 25),
                        '50': np.percentile(team1_scores, 50),
                        '75': np.percentile(team1_scores, 75)
                    }
                },
                team2: {
                    'mean': np.mean(team2_scores),
                    'std': np.std(team2_scores),
                    'min': np.min(team2_scores),
                    'max': np.max(team2_scores),
                    'percentiles': {
                        '25': np.percentile(team2_scores, 25),
                        '50': np.percentile(team2_scores, 50),
                        '75': np.percentile(team2_scores, 75)
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

    # Setup match
    state = MatchState(
        team1="India",
        team2="Australia", 
        batting_first="India",
        venue="MCG"
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