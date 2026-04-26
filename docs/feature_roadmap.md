# Feature Roadmap & Guide

This document outlines the feature sets used in the model and tracks implementation status.

## Model Versions

| Version | Features | Data Path | Model Path | Status |
|---------|----------|-----------|------------|--------|
| **v7** | **114** (V6 + hierarchical shrinkage on vs-type/vs-hand) | `data/xgb_data_v3/` | `models/xgb_v3/` | **Current (Phase 5 hier shrink + Phase 6 k=30, 2026-04-25)** |
| v6 | 114 (V3 + 42 outcome-dist, flat shrinkage) | `data/xgb_data_v3/` | `models/xgb_v3_v6_backup/` | Prior baseline (2026-04-23) |
| v4 (team strength) | 72 | `data/xgb_data_v3/` | (retrain to recover) | Prior baseline (March 2026) |
| v3 | 46+ | `data/xgb_data_v3/` | (retrain to recover) | With player metadata, no team strength |
| v2 | 29 | `data/xgb_data/` | `models/xgb/` | Legacy |

> Filename note: `models/xgb_v3/` and `data/xgb_data_v3/` paths are retained across versions; the SQLite cache is `models/player_stats_cache_v3.sqlite`. The actual schema/feature contract is governed by `_meta.schema_version` (currently 4) and `feature_registry.V6_GROUPS`. v7 has the same feature names as v6 but the *computed values* differ (shrinkage composition).
>
> **v7 vs v6**: same SQLite schema, same parquet column count (141 incl. metadata + emit-but-ignored phase_p*), same 114-feature feature_columns_v3.txt. v7 adds two-stage shrinkage on the 4 narrow cells (`batter_p*_vs_pace`, `batter_p*_vs_spin`, `bowler_p*_vs_lhb`, `bowler_p*_vs_rhb`) — these now shrink toward the player's overall distribution instead of toward the global prior π. k_player=30 is the Phase 6 sweep optimum. The hierarchical/flat split is controlled by the `hierarchical=True` default kwarg on `_SQLiteBackend.get_{batter_vs_type,bowler_vs_hand}_outcome_dist` and the equivalent tracker getters; pass `hierarchical=False` to recover v6 flat-shrunk values.
>
> **v7 backups**: `models/xgb_v3_v6_backup/` (v6), `models/xgb_v3_phase5_k30/` (Phase 5 origin), `models/xgb_v3_phase6_k{10,100,300}/` (k-sweep variants). The active `models/xgb_v3/` carries the Phase 5 model + sidecar `outcome_dist_config_v3.json` declaring k_player=30.0, k_venue=200.0; sim wrappers read this at __init__ time.
>
> **Deferred / negative results**: `phase_outcome_dist` (Phase 3) is implemented end-to-end (cache rebuild populates `prior_{pp,mid,death}_p*` in `_meta`; `parsing_v2._classify_phase_pre_ball` + `stats_sqlite_backend.get_phase_outcome_dist` + `sim_v1_2._fill_outcome_dists` all wired) but the corresponding feature group is NOT in V7_GROUPS — Phase 3 ablation showed it regresses LL (collinear with is_powerplay/middle/death indicators). Code stays inert; resurrect via `experiments/configs/xgb_v7_phase_prior.yaml` if you want to re-test.

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

### Empirical Outcome Distributions (LANDED 2026-04-23, schema v4)
Direct multi-class target encoding — for each context, emit `P(0,1,2,4,6,W)` shrunk toward the global corpus prior π via Dirichlet-posterior-mean shrinkage `(n_c + k·π_c)/(N + k)`. π is computed during `build_stats_cache.py` and stored in `_meta.prior_p*`. 42 features total.
- `batter_p{0,1,2,4,6,w}` — overall batter outcome distribution. **k=30**.
- `bowler_p{0,1,2,4,6,w}` — overall bowler outcome distribution. **k=30**.
- `batter_p{c}_vs_{pace,spin}` — 12 features, batter's distribution against bowler type. **k=30**.
- `bowler_p{c}_vs_{lhb,rhb}` — 12 features, bowler's distribution against batter hand. **k=30**.
- `venue_p{0,1,2,4,6,w}` — venue outcome distribution. **k=200** (more data per venue).

See IMPROVEMENTS.md §"Empirical Outcome Distributions" for the implementation, eval results, and 5 deferred follow-ups (phase prior, k-sweep, encoding ablation, hierarchical shrinkage, per-slice eval).

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
