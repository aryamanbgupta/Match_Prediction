# Cricket Simulation System - Complete Documentation

## System Overview

A modular Monte Carlo simulation engine for cricket match prediction that operates on ball-by-ball data. The system predicts individual ball outcomes using ML models and aggregates them through simulation to generate match-level predictions.

### Architecture Flow
```
[Raw Data] â†’ [Clean & Feature Engineering] â†’ [Model Training] â†’ [Ball Predictions] 
                                                      â†“
[Evaluation] â† [Match Simulation] â† [Monte Carlo Engine]
      â†“
[Experiment Logging]
```

## Core Architecture

### Module Structure
```
cricket_sim.py
â”œâ”€â”€ Data Classes
â”‚   â”œâ”€â”€ Outcome (Enum)           # Ball outcome types
â”‚   â”œâ”€â”€ Player                   # Player representation
â”‚   â”œâ”€â”€ TeamLineup              # Team with batting order
â”‚   â””â”€â”€ MatchState              # Complete match state
â”œâ”€â”€ Rules Engine
â”‚   â”œâ”€â”€ BowlerSelector (ABC)    # Bowler selection strategy
â”‚   â”œâ”€â”€ RandomBowlerSelector    # Random bowler selection
â”‚   â””â”€â”€ T20Rules                # T20 cricket rules engine
â”œâ”€â”€ Prediction Models
â”‚   â”œâ”€â”€ PredictionModel (ABC)   # Model interface
â”‚   â”œâ”€â”€ XGBoostModel           # XGBoost implementation
â”‚   â””â”€â”€ DummyModel             # Testing model
â”œâ”€â”€ Simulation Engine
â”‚   â”œâ”€â”€ SimulationConfig       # Simulation parameters
â”‚   â”œâ”€â”€ BallResult            # Single ball result
â”‚   â”œâ”€â”€ InningsResult         # Innings summary
â”‚   â”œâ”€â”€ MatchResult           # Match summary
â”‚   â””â”€â”€ SimulationEngine      # Main simulation orchestrator
â””â”€â”€ Analysis
    â””â”€â”€ ResultAggregator      # Result aggregation and stats
```

## Data Structures

### Match State Management

The `MatchState` class is the core data structure that maintains:

**Immutable Setup**:
- Team lineups with player batting orders
- Match metadata (venue, date, batting first)

**Dynamic Game State**:
- Current innings (1 or 2)
- Balls bowled (0-119 for T20)
- Team scores and wickets
- Current players (striker, non-striker, bowler indices)

**Historical Tracking**:
- Ball-by-ball history (300x9 numpy array)
- Player statistics (runs, balls faced, overs bowled)
- Batsmen dismissal tracking

### Key Properties

```python
# Match context
state.current_team_idx        # 0 or 1 (batting team)
state.bowling_team_idx        # 1 or 0 (bowling team)
state.overs_completed         # Overs.balls format
state.balls_remaining         # Balls left in innings

# Chase scenarios
state.target                  # Target for team 2 (if innings == 2)
state.required_run_rate       # RRR for chasing team

# Current players
state.current_striker         # Player object
state.current_bowler         # Player object
state.batting_lineup         # TeamLineup object
state.bowling_lineup         # TeamLineup object
```

## Ball Simulation Pipeline

### 1. Feature Extraction
```python
features = model.extract_features(state)
# Returns: numpy array shaped (1, n_features)
```

### 2. Outcome Prediction
```python
probs = model.predict_next_ball(features)
# Returns: {'dot': 0.3, 'one': 0.4, 'wicket': 0.05, ...}
```

### 3. Outcome Sampling & Validation
```python
outcome = random.choices(outcomes, weights=probabilities)[0]
if not rules.is_legal_outcome(state, outcome):
    outcome = Outcome.DOT  # Fallback
```

### 4. State Update
```python
runs = rules.process_ball(state, outcome)
state.update(outcome, runs)  # Updates scores, rotates strike, handles wickets
```

## Prediction Model Interface

### Required Implementation

```python
class CustomModel(PredictionModel):
    def extract_features(self, state: MatchState) -> np.ndarray:
        """Extract features from current match state"""
        # Must return shape (1, n_features)
        pass
    
    def predict_next_ball(self, features: np.ndarray) -> Dict[str, float]:
        """Return probability distribution over outcomes"""
        # Must return dict with keys:
        # 'dot', 'one', 'two', 'four', 'six', 'wicket', 'wide', 'no_ball'
        # Values should sum to ~1.0
        pass
```

### Current XGBoost Implementation

**Features Extracted** (9 total):
- `inning_idx`: Current innings number
- `score`: Current team total runs
- `wickets`: Current team wickets fallen
- `balls_bowled`: Balls bowled in current innings
- `batter_encoded`: Striker player ID (LabelEncoded)
- `bowler_encoded`: Bowler player ID (LabelEncoded)
- `run_rate`: Current run rate
- `wickets_ratio`: Wickets fallen / 10
- `balls_ratio`: Balls bowled / 120

**Output Mapping**:
```python
class_to_outcome = {
    0: 'dot', 1: 'one', 2: 'two', 3: 'four',
    4: 'four', 5: 'six', 6: 'six', 7: 'wicket'
}
```

## API Reference

### Creating a Match

```python
# Define players with IDs matching training data
players_team1 = [
    Player("rohit_sharma", "Rohit Sharma", "India", "batsman"),
    Player("shubman_gill", "Shubman Gill", "India", "batsman"),
    # ... 11 players total
]

players_team2 = [
    Player("david_warner", "David Warner", "Australia", "batsman"),
    # ... 11 players total
]

# Create lineups
lineup1 = TeamLineup("India", players_team1)
lineup2 = TeamLineup("Australia", players_team2)

# Initialize match state
state = MatchState(
    team1_lineup=lineup1,
    team2_lineup=lineup2,
    batting_first="India",  # Must match team name
    venue="MCG",
    match_date=datetime(2024, 12, 25)
)
```

### Running Simulations

```python
# Initialize engine
model = XGBoostModel("model.pkl", "batter_encoder.pkl", "bowler_encoder.pkl")
engine = SimulationEngine(model, T20Rules())

# Single simulation
result = engine.simulate_match(state, match_id="test_001")

# Multiple simulations
config = SimulationConfig(
    n_simulations=1000,
    parallel=True,
    n_workers=4,
    random_seed=42,
    verbose=True
)
results = engine.simulate_multiple(state, config)
```

### Accessing Results

```python
# Single match results
result.winner              # "India", "Australia", or "Tie"
result.margin             # "23 runs" or "5 wickets"
result.team1_score        # Final score
result.innings[0].balls   # List of BallResult objects

# Aggregated results
summary = ResultAggregator.aggregate(results)
summary['win_probability']['India']           # 0.653
summary['score_stats']['India']['mean']       # 167.4
summary['score_stats']['India']['percentiles']['50']  # Median score
```

## State Management Details

### Critical State Transitions

**Strike Rotation**:
- After odd runs (1, 3, 5)
- At end of over (automatic)

**Wicket Handling**:
- Striker gets out â†’ `get_next_batsman_idx()` finds replacement
- Updates `batsmen_out` tracking dict
- Strike doesn't rotate

**Over Completion**:
- Automatic strike rotation
- New bowler selection via `BowlerSelector`
- `last_bowler_idx` updated (prevents consecutive overs)

**Innings Transition**:
- Reset balls to 0
- Reset player indices (striker=0, non_striker=1, bowler=0)
- Maintain statistical tracking across innings

### Player Tracking

**Bowler Constraints**:
- Maximum 24 balls (4 overs) per bowler in T20
- Cannot bowl consecutive overs
- Tracked via `bowler_balls` dict

**Batsman Management**:
- Batting order strictly followed (indices 0-10)
- `batsmen_out` tracks dismissed players
- Next batsman = lowest unused index

## Implementation Features

### Performance Optimizations

**Parallel Processing**:
- Uses `multiprocessing.Pool` for multiple simulations
- Each worker gets independent random seed
- Models must be pickle-serializable

**Memory Management**:
- Dynamic history array expansion (starts at 300 balls)
- Efficient state copying for parallel execution
- Numpy arrays for numerical data

**Simulation Speed**:
- ~1000 simulations/second reported in examples
- Vectorized operations where possible

### Error Handling

**Player Encoding Fallbacks**:
```python
try:
    batter_encoded = self.batter_encoder.transform([striker.player_id])[0]
except:
    batter_encoded = -1  # Unknown player
```

**Boundary Checking**:
- History array auto-expansion
- Bowler availability validation
- Legal outcome verification

## Known Issues & Uncertainties

### 1. Feature Engineering Gap
The current XGBoost model only uses 9 basic features, while the workflow document specifies comprehensive feature sets including:
- Player career statistics
- Recent form (sliding windows)
- Match phase indicators
- Head-to-head records
- Pressure indicators

### 2. Player ID Dependencies
The system requires player IDs to exactly match training data encoders. Unknown players get encoded as `-1`, which may significantly impact prediction quality.

### 3. Model Output Mapping Ambiguity
```python
# Current mapping has duplicates - is this intentional?
class_to_outcome = {
    0: 'dot', 1: 'one', 2: 'two', 3: 'four',
    4: 'four', 5: 'six', 6: 'six', 7: 'wicket'
}
```

### 4. Bowler Selection Simplification
Current implementation uses random bowler selection. The workflow suggests more sophisticated strategies based on matchups and game situation.

### 5. Extras Handling
Simplified extras logic:
- Wide = 1 run always
- No-ball = 1 run always
- No handling for wide+runs or no-ball+runs scenarios

## Usage Examples

### Basic Simulation
```python
# Quick setup with dummy model for testing
model = DummyModel()
engine = SimulationEngine(model)

# Create teams and state
state = create_match_state()

# Run single simulation
result = engine.simulate_match(state)
print(f"Winner: {result.winner} by {result.margin}")
```

### Batch Analysis
```python
# Multiple simulations for statistical analysis
config = SimulationConfig(n_simulations=1000, parallel=True)
results = engine.simulate_multiple(state, config)

# Get aggregated statistics
summary = ResultAggregator.aggregate(results)

# Analyze win probabilities
for team, prob in summary['win_probability'].items():
    print(f"{team}: {prob:.1%}")

# Score distributions
for team, stats in summary['score_stats'].items():
    print(f"{team}: {stats['mean']:.1