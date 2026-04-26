# Data Formats Reference

Complete specifications for all data formats used in the CricML Match Prediction system.

---

## 1. Raw Match JSON (Cricsheet Format)

**Location**: `data/t20s_json/*.json`, `data/betting_test/*.json`

**Source**: [Cricsheet](https://cricsheet.org/) - Ball-by-ball cricket data

### Structure

```json
{
  "info": {
    "teams": ["India", "Australia"],
    "dates": ["2024-06-15"],
    "venue": "Melbourne Cricket Ground",
    "city": "Melbourne",
    "toss": {
      "winner": "India",
      "decision": "bat"
    },
    "gender": "male",
    "match_type": "T20",
    "outcome": {
      "winner": "India",
      "by": {
        "runs": 23
      }
    },
    "registry": {
      "people": {
        "Rohit Sharma": "253802",
        "Virat Kohli": "277906",
        "Mitchell Starc": "290630"
      }
    }
  },
  "innings": [
    {
      "team": "India",
      "overs": [
        {
          "over": 0,
          "deliveries": [
            {
              "batter": "Rohit Sharma",
              "bowler": "Mitchell Starc",
              "non_striker": "Shubman Gill",
              "runs": {
                "batter": 4,
                "extras": 0,
                "total": 4
              },
              "wickets": null
            },
            {
              "batter": "Rohit Sharma",
              "bowler": "Mitchell Starc",
              "non_striker": "Shubman Gill",
              "runs": {
                "batter": 0,
                "extras": 1,
                "total": 1
              },
              "extras": {
                "wides": 1
              }
            },
            {
              "batter": "Rohit Sharma",
              "bowler": "Mitchell Starc",
              "non_striker": "Shubman Gill",
              "runs": {
                "batter": 0,
                "extras": 0,
                "total": 0
              },
              "wickets": [
                {
                  "player_out": "Rohit Sharma",
                  "kind": "caught",
                  "fielders": ["David Warner"]
                }
              ]
            }
          ]
        }
      ]
    },
    {
      "team": "Australia",
      "overs": [...]
    }
  ]
}
```

### Key Fields

**info**:
- `teams`: List of two team names
- `dates`: List with single date string (YYYY-MM-DD)
- `venue`: Full venue name
- `toss.winner`: Team that won toss
- `toss.decision`: "bat" or "field"
- `outcome.winner`: Match winner
- `outcome.by`: Margin (runs or wickets)
- `registry.people`: Map player names → IDs

**innings**:
- `team`: Batting team name
- `overs`: List of over objects

**deliveries** (balls):
- `batter`: Batsman facing (name, not ID)
- `bowler`: Bowler (name, not ID)
- `non_striker`: Non-striker batsman
- `runs.batter`: Runs scored by batter
- `runs.extras`: Extra runs (wides, no-balls, byes, leg-byes)
- `runs.total`: Total runs for ball
- `extras`: Type of extras (optional)
- `wickets`: Wicket information (optional)

---

## 2. Processed Parquet Files

**Location**: `data/xgb_data/cricket_data_v2_*.parquet`

**Files**:
- `cricket_data_v2_train.parquet` (~3.5M rows)
- `cricket_data_v2_validation.parquet` (~400K rows)
- `cricket_data_v2_test.parquet` (~300K rows)
- `cricket_data_v2_betting_test.parquet` (~50K rows)

### Schema (29 features + metadata + target)

**Metadata Columns**:
```
innings_id           str      Unique innings identifier
inning_idx           int      1 or 2
over_idx             int      0-19 (T20)
ball_idx             int      0-119 (ball number in innings)
batter_id            str      Player ID
non_striker_id       str      Player ID
bowler_id            str      Player ID
```

**Basic State Features** (12):
```
score                int      Current team total runs
wickets              int      Current team wickets fallen
balls_bowled         int      Balls bowled so far (0-119)
run_rate             float    Current run rate
wickets_ratio        float    wickets / 10
balls_ratio          float    balls_bowled / 120
wickets_in_hand      int      10 - wickets
is_powerplay         bool     First 6 overs
is_middle_overs      bool     Overs 7-15
is_death_overs       bool     Overs 16-20
balls_in_over        int      Balls bowled in current over (0-5)
```

**Player Features** (6):
```
batter_encoded       int      LabelEncoded batter ID
bowler_encoded       int      LabelEncoded bowler ID
batsman_avg          float    Career batting average
batsman_sr           float    Career batting strike rate
bowler_avg           float    Career bowling average
bowler_econ          float    Career bowling economy
```

**Head-to-Head Features** (2):
```
h2h_avg              float    Batter vs bowler average
h2h_sr               float    Batter vs bowler strike rate
```

**Momentum Features** (5):
```
last_5_balls_runs    int      Runs in last 5 balls
last_10_balls_runs   int      Runs in last 10 balls
last_30_balls_runs   int      Runs in last 30 balls
balls_since_boundary int      Balls since last 4 or 6
last_10_dots         int      Dot balls in last 10 balls
```

**Pressure Features** (4):
```
dot_percentage_recent        float    % dots in recent balls
boundary_percentage_recent   float    % boundaries in recent balls
```

**Target Column**:
```
ball_outcome         int      0, 1, 2, 4, 6, -1 (wicket)
```

### Data Types
```python
import pandas as pd

df = pd.read_parquet('data/xgb_data/cricket_data_v2_train.parquet')

print(df.dtypes)
# innings_id                     object
# inning_idx                      int64
# batter_id                      object
# batter_encoded                  int64
# score                           int64
# run_rate                      float64
# batsman_avg                   float64
# ...
# ball_outcome                    int64
```

---

## 3. Player Stats Cache

**Location**: `models/cache_chunks/` + `models/player_stats_cache_metadata.pkl`

### Chunked Structure

**Files**:
- `cache_chunk_0.pkl` through `cache_chunk_68.pkl` (69 total)
- Each chunk: ~110MB
- Total size: ~7.6GB

### Metadata File (`player_stats_cache_metadata.pkl`)

```python
{
    'num_chunks': 69,
    'num_matches': 8341,
    'num_dates': 3442,
    'num_players_batting': 7240,
    'num_players_bowling': 5319,
    'num_h2h_matchups': 50000,
    'build_timestamp': '2024-10-14T18:56:00',

    'chunk_files': [
        'cache_chunks/cache_chunk_0.pkl',
        'cache_chunks/cache_chunk_1.pkl',
        ...
    ],

    'chunks': [
        {
            'file': 'cache_chunks/cache_chunk_0.pkl',
            'dates': ['2005-02-17', '2005-02-19', ..., '2007-04-20'],
            'num_dates': 50
        },
        {
            'file': 'cache_chunks/cache_chunk_1.pkl',
            'dates': ['2007-04-22', ..., '2009-06-15'],
            'num_dates': 50
        },
        ...
    ]
}
```

### Chunk File Structure

Each `.pkl` file contains:

```python
{
    '2020-01-15': {
        'batting': {
            'player_id_1': {
                'runs': 450,
                'balls': 320,
                'dismissals': 12
            },
            'player_id_2': {
                'runs': 890,
                'balls': 650,
                'dismissals': 22
            },
            # ... ~7,000 players
        },
        'bowling': {
            'player_id_1': {
                'runs_given': 1250,
                'balls_bowled': 540,
                'wickets': 18
            },
            # ... ~5,000 players
        },
        'h2h': {
            ('batter_id_1', 'bowler_id_1'): {
                'runs': 45,
                'balls': 32,
                'dismissals': 1
            },
            ('batter_id_1', 'bowler_id_2'): {
                'runs': 89,
                'balls': 67,
                'dismissals': 3
            },
            # ... ~50,000 matchups
        }
    },
    '2020-01-16': {
        # Same structure, updated stats
    },
    # ... ~50 dates per chunk
}
```

### Derived Stats

**Batting Average**:
```python
avg = runs / dismissals if dismissals > 0 else 0.0
```

**Batting Strike Rate**:
```python
sr = (runs / balls * 100) if balls > 0 else 0.0
```

**Bowling Average**:
```python
avg = runs_given / wickets if wickets > 0 else 0.0
```

**Bowling Economy**:
```python
econ = (runs_given / balls_bowled * 6) if balls_bowled > 0 else 0.0
```

---

## 4. Trained Model Artifacts

**Location**: `models/xgb/`

### Files

**xgboost_model_v2.pkl** (~50MB):
- XGBoost classifier
- 6-class output (dot, 1, 2, 4, 6, wicket)
- 29 input features

**batter_encoder_v2.pkl** (~200KB):
- LabelEncoder for batter IDs
- Vocabulary: ~3,000 players

**bowler_encoder_v2.pkl** (~150KB):
- LabelEncoder for bowler IDs
- Vocabulary: ~2,500 players

**feature_columns_v2.txt** (~1KB):
```
inning_idx
score
wickets
balls_bowled
run_rate
wickets_ratio
balls_ratio
wickets_in_hand
is_powerplay
is_middle_overs
is_death_overs
balls_in_over
batter_encoded
bowler_encoded
batsman_avg
batsman_sr
bowler_avg
bowler_econ
h2h_avg
h2h_sr
last_5_balls_runs
last_10_balls_runs
last_30_balls_runs
balls_since_boundary
last_10_dots
dot_percentage_recent
boundary_percentage_recent
```

**optuna_study_v2.pkl** (~10MB):
- Optuna study object
- 50 trials
- Best hyperparameters

### Model Predictions

**Input**: Single row DataFrame with 29 features

**Output**: Probability array (6 classes)
```python
# Example
[0.32, 0.39, 0.08, 0.10, 0.04, 0.05]
#  dot   1    2     4     6    wicket
```

---

## 5. Betting Odds JSON

**Location**: `betting_odds_v3.json`

### Structure

```json
{
  "matches": [
    {
      "match_id": "2024-06-15_India_Australia_MCG",
      "date": "2024-06-15",
      "team1": "India",
      "team2": "Australia",
      "venue": "MCG",
      "odds": {
        "winner": {
          "India": 2.10,
          "Australia": 1.75,
          "timestamp": "2024-06-14T10:00:00Z"
        }
      },
      "actual_winner": "India",
      "actual_margin": {
        "runs": 23
      }
    },
    {
      "match_id": "2024-06-17_England_SouthAfrica_Oval",
      "date": "2024-06-17",
      "team1": "England",
      "team2": "South Africa",
      "venue": "The Oval",
      "odds": {
        "winner": {
          "England": 1.55,
          "South Africa": 2.40,
          "timestamp": "2024-06-16T10:00:00Z"
        }
      },
      "actual_winner": "England",
      "actual_margin": {
        "wickets": 5
      }
    }
  ]
}
```

### Fields

**Odds Format**: Decimal odds
- `2.10` means bet $1 to win $2.10 total ($1.10 profit)
- Lower odds = higher probability (favorite)
- Higher odds = lower probability (underdog)

**Converting to Probability**:
```python
implied_prob = 1 / decimal_odds

# Example
india_prob = 1 / 2.10 = 0.476 (47.6%)
australia_prob = 1 / 1.75 = 0.571 (57.1%)

# Note: Sum > 1.0 due to bookmaker margin (overround)
total = 0.476 + 0.571 = 1.047 (4.7% margin)

# Normalize to remove margin
fair_india = 0.476 / 1.047 = 0.455 (45.5%)
fair_australia = 0.571 / 1.047 = 0.545 (54.5%)
```

---

## 6. Evaluation Results JSON

**Location**: `match_evaluation_results.json`

### Structure

```json
{
  "summary": {
    "n_matches": 45,
    "avg_log_loss": 0.6234,
    "avg_brier_score": 0.1845,
    "avg_edge": 0.083,
    "avg_signed_edge": 0.021,
    "profitable_bets": 28,
    "total_pnl": 12.45,
    "roi": 0.445,
    "win_rate": 0.571,
    "bets_placed": 28,
    "total_simulation_time": 450.2,
    "calibration_bins": [
      {
        "predicted_prob": 0.452,
        "actual_win_rate": 0.421,
        "num_matches": 18
      },
      {
        "predicted_prob": 0.558,
        "actual_win_rate": 0.583,
        "num_matches": 24
      }
    ]
  },
  "matches": [
    {
      "match_id": "2024-06-15_India_Australia_MCG",
      "team1": "India",
      "team2": "Australia",
      "simulated_win_prob": {
        "India": 0.653,
        "Australia": 0.347
      },
      "simulated_scores": {
        "India": {
          "mean": 167.4,
          "std": 15.2,
          "percentiles": {
            "25": 157,
            "50": 167,
            "75": 178
          }
        },
        "Australia": {
          "mean": 145.8,
          "std": 18.3,
          "percentiles": {
            "25": 133,
            "50": 146,
            "75": 159
          }
        }
      },
      "market_win_prob": {
        "India": 0.455,
        "Australia": 0.545
      },
      "market_odds": {
        "India": 2.10,
        "Australia": 1.75
      },
      "edge": {
        "India": 0.198,
        "Australia": -0.198
      },
      "actual_winner": "India",
      "log_loss": 0.426,
      "brier_score": 0.121,
      "realized_pnl": 1.10,
      "n_simulations": 1000,
      "simulation_time": 10.4
    }
  ]
}
```

### Metrics

**Log Loss**:
```python
log_loss = -log(P(actual_winner))
# Lower is better
# 0 = perfect (100% confidence in winner)
# ∞ = worst (0% confidence in winner)
```

**Brier Score**:
```python
brier = (P(team1) - actual)²
# where actual = 1 if team1 won, else 0
# Lower is better
# 0 = perfect
# 1 = worst
```

**Edge**:
```python
edge = model_prob - market_prob
# Positive = model favors team more than market
# Negative = model favors team less than market
```

**Signed Edge**:
```python
if predicted_winner == actual_winner:
    signed_edge = +abs(edge)  # Correct prediction
else:
    signed_edge = -abs(edge)  # Incorrect prediction
```

**ROI**:
```python
roi = total_pnl / bets_placed
# Decimal format (0.445 = 44.5% return)
```

---

## 7. Simulation Output Structures

### MatchResult (Python)

```python
@dataclass
class MatchResult:
    match_id: str
    team1: str
    team2: str
    winner: str                        # "Team1", "Team2", or "Tie"
    margin: str                        # "23 runs" or "5 wickets"
    innings: List[InningsResult]       # 1 or 2 innings
    team1_score: int
    team1_wickets: int
    team2_score: int
    team2_wickets: int
```

### InningsResult (Python)

```python
@dataclass
class InningsResult:
    batting_team: str
    bowling_team: str
    total_runs: int
    total_wickets: int
    total_balls: int
    run_rate: float

    # Performance cards
    batting_card: Dict[int, Tuple[int, int, int, int]]
    # player_idx -> (runs, balls, 4s, 6s)

    bowling_card: Dict[int, Tuple[int, int, int]]
    # player_idx -> (balls, runs, wickets)

    balls: List[BallResult]            # Ball-by-ball records
```

### BallResult (Python)

```python
@dataclass
class BallResult:
    innings: int                       # 1 or 2
    over: int                          # 0-19
    ball: int                          # 0-5
    outcome: Outcome                   # DOT, ONE, TWO, FOUR, SIX, WICKET, etc.
    runs: int                          # Runs scored
    striker_idx: int                   # Batsman facing (0-10)
    bowler_idx: int                    # Bowler (0-10)
    team_runs: int                     # Running total
    team_wickets: int                  # Running wickets
```

### AggregatedResults (Python)

```python
{
    'win_probability': {
        'India': 0.653,
        'Australia': 0.347
    },
    'score_stats': {
        'India': {
            'mean': 167.4,
            'std': 15.2,
            'min': 124,
            'max': 198,
            'percentiles': {
                5: 142,
                25: 157,
                50: 167,
                75: 178,
                95: 192
            }
        },
        'Australia': { ... }
    },
    'wicket_stats': {
        'India': {
            'mean': 6.2,
            'distribution': {
                0: 2,    # 2 matches all out for 0 wickets
                1: 15,
                ...
                10: 85   # 85 matches all out
            }
        },
        'Australia': { ... }
    }
}
```

---

## File Size Reference

```
data/t20s_json/                  ~5 GB    (15,000+ JSON files)
data/xgb_data/*.parquet          ~800 MB  (4 parquet files)
models/cache_chunks/             ~7.6 GB  (69 chunk files)
models/xgb/*                     ~50 MB   (model + encoders)
betting_odds_v3.json             ~100 KB  (45 matches)
match_evaluation_results.json    ~200 KB  (45 match results)
```

---

**For implementation details on how to work with these formats, see [OPERATIONS.md](./OPERATIONS.md).**

**For design rationale, see [DESIGN_DECISIONS.md](./DESIGN_DECISIONS.md).**
