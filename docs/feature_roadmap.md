# Feature Roadmap & Guide

This document outlines the feature sets used in the model and tracks implementation status.

## Model Versions

| Version | Features | Data Path | Model Path | Status |
|---------|----------|-----------|------------|--------|
| **v3** | 46+ | `data/xgb_data_v3/` | `models/xgb_v3/` | Current (with player metadata) |
| v2 | 29 | `data/xgb_data/` | `models/xgb/` | Legacy |

---

## 1. Existing Features
These features are currently implemented in `scripts/parsing_v2.py` and used in `scripts/xgboost_v2.py`.

### Match State
- **`innings_id`**: Unique identifier for the innings.
- **`inning_idx`**: 1st or 2nd innings.
- **`score`**: Current runs on the board.
- **`wickets`**: Number of wickets lost.
- **`balls_bowled`**: Number of legal deliveries bowled.
- **`run_rate`**: Current runs per over.
- **`wickets_ratio`**: Wickets lost / 10.
- **`balls_ratio`**: Balls bowled / 120.
- **`wickets_in_hand`**: 10 - wickets lost.
- **`balls_in_over`**: Legal balls bowled in the current over (0-5).
- **`is_powerplay`**: Boolean (Overs 0-6).
- **`is_middle_overs`**: Boolean (Overs 6-16).
- **`is_death_overs`**: Boolean (Overs 16-20).
- **`is_batting_first`**: Boolean.
- **`is_toss_winner`**: Boolean.

### Player Stats (Historical)
- **`batsman_avg`**: Career batting average.
- **`batsman_sr`**: Career batting strike rate.
- **`bowler_avg`**: Career bowling average.
- **`bowler_econ`**: Career bowling economy rate.
- **`batsman_recent_avg`**: Batting average in last 5 matches.
- **`batsman_recent_sr`**: Strike rate in last 5 matches.
- **`bowler_recent_avg`**: Bowling average in last 5 matches.
- **`bowler_recent_econ`**: Economy rate in last 5 matches.
- **`h2h_avg`**: Head-to-head average (Batter vs Bowler).
- **`h2h_sr`**: Head-to-head strike rate (Batter vs Bowler).

### Momentum & Pressure
- **`last_5_balls_runs`**: Runs scored in the last 5 balls.
- **`last_10_balls_runs`**: Runs scored in the last 10 balls.
- **`last_30_balls_runs`**: Runs scored in the last 30 balls.
- **`balls_since_boundary`**: Deliveries since the last 4 or 6.
- **`last_10_dots`**: Number of dot balls in the last 10 deliveries.
- **`dot_percentage_recent`**: % of dots in recent history.
- **`boundary_percentage_recent`**: % of boundaries in recent history.

### Identifiers (Encoded)
- **`batter_encoded`**: Label encoded Batter ID.
- **`bowler_encoded`**: Label encoded Bowler ID.
- **`venue_encoded`**: Label encoded Venue ID.

---

## 2. Missing / To-Be-Implemented Features
These features are requested to enhance the model's understanding of context, matchups, and game dynamics.

### Match Context (The "Pressure" Features)
- [x] **`run_rate_required`**: Required Run Rate (RRR) for 2nd innings. *Critical for chasing logic.*
- [x] **`lead_gap`**: Difference between current score and opponent's score (or target).
- [x] **`balls_remaining`**: Explicit count of balls left (currently using `balls_ratio`).

### The Micro-Battle (Batter vs. Bowler)
*Implemented using player metadata from `data/all_players_enriched.csv`.*
- [x] **`batter_hand`**: Right/Left hand batting. (Encoded: 0=right, 1=left, 2=unknown)
- [x] **`bowler_arm`**: Right/Left arm bowling. (Encoded: 0=right, 1=left, 2=unknown)
- [x] **`is_pace`**: Pace vs Spin bowler. (Encoded: 0=spin, 1=pace, 2=unknown)
- [x] **`bowling_type`**: Granular bowling type (fast, medium-fast, offspin, legspin, etc.)
- [x] **`batter_age`**: Player age as of match date.
- [x] **`bowler_age`**: Player age as of match date.
- [x] **`matchup_type_encoded`**: Derived interaction (e.g., "RHB_vs_offspin").
- [x] **`spin_matchup_advantage`**: Known spin matchup advantages (-1, 0, or 1).
- [x] **`same_arm_matchup`**: Whether batter and bowler have same dominant side.
- [x] **`batter_avg_vs_pace`**: Batter's average against pace bowlers.
- [x] **`batter_sr_vs_pace`**: Batter's strike rate against pace bowlers.
- [x] **`batter_avg_vs_spin`**: Batter's average against spin bowlers.
- [x] **`batter_sr_vs_spin`**: Batter's strike rate against spin bowlers.
- [x] **`bowler_avg_vs_lhb`**: Bowler's average against left-hand batters.
- [x] **`bowler_econ_vs_lhb`**: Bowler's economy against left-hand batters.
- [x] **`bowler_avg_vs_rhb`**: Bowler's average against right-hand batters.
- [x] **`bowler_econ_vs_rhb`**: Bowler's economy against right-hand batters.
- [ ] **`batter_style`**: Accumulator vs Power Hitter (derived from SR).

### "In-the-Moment" Features (Dynamic State)
- [x] **`batter_balls_faced`**: Number of balls the current batter has faced in this innings (Set vs Fresh).
- [x] **`batter_runs_scored`**: Current score of the batter in this innings.
- [x] **`bowler_balls_in_innings`**: Balls bowled by the bowler in this innings.
- [x] **`bowler_overs_in_innings`**: Fractional overs bowled by the bowler in this innings.
- [ ] **`bowler_over_in_spell`**: Is this the bowler's 1st, 2nd, or 4th over in the current spell?
- [ ] **`total_wickets_match`**: Wickets taken by the bowler in this match so far.

### Environmental & Physical Features
- [x] **`venue_avg_score`**: Historical average score at this venue. *Temporal integrity enforced.*
- [ ] **`boundary_dimensions`**: Distance to boundaries (if available).
- [ ] **`dew_factor`**: Boolean/Scale for dew presence.

### Advanced / Creative Features
- [x] **`pressure_cooker_index`**: RRR / Wickets Remaining.
- [x] **`partnership_runs`**: Current partnership runs.
- [x] **`non_striker_sr`**: Strike rate of the partner (pressure release).
- [x] **`chase_target`**: First innings score + 1 (for 2nd innings chase). *Renamed from `target` to avoid collision with prediction target.*
- [ ] **`h2h_dismissals`**: Count of times this bowler has dismissed this batter.
- [ ] **`projected_score_differential`**: Difference vs Par Score (DLS).
