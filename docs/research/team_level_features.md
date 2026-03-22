# Team-Level Features: Research & Options Analysis

**Date**: March 2026
**Status**: Phase 1 implemented and evaluated. Log Loss improved 19% (0.8754 → 0.7100), Brier Score improved 19% (0.3168 → 0.2554). ROI still negative (-44.8%). Next: calibration layer or Phase 2 (bridge-player normalization).
**Priority**: P1 — root cause of all models losing money against betting markets

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [The Cross-League Challenge](#2-the-cross-league-challenge)
3. [Approaches Evaluated](#3-approaches-evaluated)
   - [Option 1: Aggregated Player Stats](#option-1-aggregated-player-stats-simplest)
   - [Option 2: Per-Team ELO](#option-2-per-team-elo-within-league)
   - [Option 3: Bridge Player League Difficulty Factors](#option-3-bridge-player-league-difficulty-factors)
   - [Option 4: Opposition-Adjusted Player Ratings](#option-4-opposition-adjusted-player-ratings)
   - [Option 5: Hierarchical Bayesian Model](#option-5-hierarchical-bayesian-model)
   - [Option 6: Player-Level ELO (Ball-by-Ball)](#option-6-player-level-elo-ball-by-ball)
   - [Option 7: League-Tiered Team ELO with Bridge Calibration](#option-7-league-tiered-team-elo-with-bridge-calibration)
4. [Comparison Matrix](#4-comparison-matrix)
5. [Industry Approaches](#5-what-the-industry-does)
6. [Recommendation](#6-recommendation)
7. [Sources](#7-sources)

---

## 1. Problem Statement

All 4 models (XGBoost, LSTM, Transformer, MLP) currently **lose money** against betting markets:

| Model | Ball Accuracy | Flat ROI | Win Rate |
|-------|--------------|----------|----------|
| XGBoost v3 | ~55-60% | -43.9% | 26.8% |
| LSTM v1 | ~44% | -63.4% | 12.2% |
| Transformer v1 | ~44% | -63.4% | 12.2% |
| MLP v1 | ~44% | -63.8% | 12.2% |

**Root cause**: Models have **no team-level signal**. They see individual batter/bowler stats but cannot distinguish India from Scotland at the team level. Models predict ~50% for nearly all matches regardless of team quality.

**What's needed**: Features that encode team strength, allowing the model to differentiate between strong and weak teams. This signal must propagate from ball-level features through Monte Carlo simulation to produce differentiated match-level win probabilities.

---

## 2. The Cross-League Challenge

Our dataset spans **multiple disconnected T20 leagues**:

| League | Type | Teams | Connectivity |
|--------|------|-------|-------------|
| T20I (Internationals) | National | ~20+ | ICC events only |
| IPL (India) | Franchise | 10 | Insular |
| BBL (Australia) | Franchise | 8 | Insular |
| PSL (Pakistan) | Franchise | 6 | Insular |
| CPL (Caribbean) | Franchise | 6 | Insular |
| SA20 (South Africa) | Franchise | 6 | Insular |
| T20 Blast (England) | Franchise | 18 | Insular |
| BPL (Bangladesh) | Franchise | 7 | Insular |

**Key challenges**:

1. **Teams never play across leagues** — Mumbai Indians never faces Sydney Sixers. There are no "Champions League" inter-league matches to calibrate relative strength.

2. **Statistics are league-relative** — A batting average of 35 in IPL (against world-class bowling) is fundamentally different from 35 in BPL (against weaker bowling). Strike rates and bowling stats have the same problem.

3. **International matches mix league contexts** — When India plays Australia in a T20I, the players come from IPL, BBL, domestic cricket, etc. Their historical stats are a blend of performances across different quality levels.

4. **Franchise rosters change frequently** — Mumbai Indians 2020 had a completely different squad than 2024. Team-level ELO for franchises is less meaningful than for national teams.

5. **Rarely-meeting teams** — When Sri Lanka plays Scotland, there's almost no head-to-head history and no shared league context.

---

## 3. Approaches Evaluated

### Option 1: Aggregated Player Stats (Simplest)

**How it works**: For each match, look up the 11 players' individual batting/bowling stats from the existing `StatsProvider` cache. Compute team-level aggregates:

```
team_batting_avg = mean(batting averages of 11 players)
team_batting_sr  = mean(strike rates of 11 players)
team_bowling_avg = mean(bowling averages of bowling XI)
team_bowling_econ = mean(economy rates of bowling XI)
relative_strength = team_batting_sr - opp_bowling_econ (or similar)
batting_depth     = mean(bottom 5 batters' averages)
```

**Pros**:
- Trivially implementable — all data already exists in `StatsProvider`
- Immediately adds team signal (models can distinguish India from Scotland)
- Temporally sound — uses pre-match stat snapshots that already exist
- Works for simulation too — lineup is known at match start
- No new data infrastructure needed

**Cons**:
- Stats are NOT league-adjusted — a player with avg 35 from BPL looks identical to avg 35 from IPL
- Players who only play in weaker leagues appear artificially stronger
- Doesn't account for opposition quality in the underlying stats
- Averages can be misleading for all-rounders or players with few matches
- Doesn't capture team synergy, strategy, or depth nuances

**Cross-league verdict**: Partially solves the problem. Players who play across leagues have blended global stats, which is an implicit (but noisy) form of cross-league normalization. But pure domestic players carry league-biased stats.

**Implementation effort**: ~1 day
**Expected impact**: Medium — adds clear team signal but with league bias

---

### Option 2: Per-Team ELO (Within League)

**How it works**: Maintain a running ELO rating per team (or per franchise), updated after each match result. Standard ELO formula:

```
E(A) = 1 / (1 + 10^((R_B - R_A) / 400))
R_A_new = R_A + K * (S_A - E(A))
```

Where K is the update factor (typically 20-40 for cricket), S is actual outcome (1=win, 0=loss), E is expected outcome.

Features: `batting_team_elo`, `bowling_team_elo`, `elo_differential`

**Pros**:
- Simple, well-understood algorithm
- Captures form and momentum — a team on a winning streak has higher ELO
- Naturally temporal (ELO is historical by construction)
- Good within a single league (IPL ELO works well for IPL predictions)
- K-factor can be tuned per league

**Cons**:
- **Completely fails cross-league** — IPL ELO scale ≠ BBL ELO scale ≠ T20I ELO scale
- Disconnected leagues produce incomparable ELO values
- Franchise roster changes every season (different players, same team name)
- International teams play infrequently (T20I teams may play only 10-15 matches/year)
- New teams/franchises start with no history
- A team's ELO doesn't reflect its current squad composition

**Cross-league verdict**: Does NOT solve it. Disconnected leagues produce incomparable ELO scales. You'd need a separate ELO system per league, making them useless as a unified feature for cross-league prediction.

**Implementation effort**: ~1-2 days
**Expected impact**: Low-Medium — useful within league, useless across leagues

---

### Option 3: Bridge Player League Difficulty Factors

**How it works**: This is the CricViz approach, validated at scale with 4,500+ players.

1. **Identify bridge players**: Find all players who have played in 2+ leagues
2. **Compute performance differentials**: For each bridge player, compare their stats across leagues
   ```
   Player X: avg 42.3 in BBL, avg 31.1 in IPL
   → IPL is approximately 26% harder than BBL for batting
   ```
3. **Aggregate into league difficulty factors**: Average the differentials across all bridge players between each pair of leagues
   ```
   IPL difficulty factor = 1.00 (reference)
   BBL difficulty factor = 0.85
   PSL difficulty factor = 0.82
   T20I difficulty factor = 0.95
   BPL difficulty factor = 0.65
   ```
4. **Normalize all player stats**: Multiply raw stats by league difficulty factors before computing team strength
   ```
   BPL avg 35 × 0.65 = adjusted avg 22.75
   IPL avg 35 × 1.00 = adjusted avg 35.00
   ```

**CricViz's league quality ranking** (from their published research):
1. IPL (highest quality)
2. T20 Internationals
3. PSL, CPL, BBL (next tier)
4. T20 Blast, BPL (lower tier)

**Pros**:
- Solves the cross-league problem empirically — grounded in actual performance data
- Validated at production scale by CricViz (4,500+ bridge players)
- Intuitive and explainable
- Can be combined with Option 1 (normalize stats first, then aggregate into team strength)
- League difficulty can be computed per time window (rolling) to capture changes over time

**Cons**:
- Assumes bridge players try equally hard in all leagues (motivation/fatigue bias — a player might underperform in BBL right after an exhausting IPL season)
- Some league pairs have few bridge players (e.g., SA20↔BPL may have very few shared players)
- League difficulty changes over time (IPL 2010 quality ≠ IPL 2024 quality) — need rolling windows
- Requires tracking **which league each ball was played in** — our current stats cache doesn't separate stats by league
- Players may play different roles in different leagues (opener in IPL, middle-order in national team)
- The normalization assumes a linear scaling which may not hold (difficulty might affect dots vs boundaries differently)

**Cross-league verdict**: The best practical solution for explicit league calibration. CricViz's entire cross-league comparison system is built on this approach.

**Implementation effort**: ~3-5 days (need to add league tracking to parsing pipeline and stats cache, compute difficulty factors, normalize stats)
**Expected impact**: High — directly addresses league bias in stats

---

### Option 4: Opposition-Adjusted Player Ratings (Iterative)

**How it works**: An iterative self-consistent approach (similar to PageRank):

1. Start with raw player averages as initial ratings
2. For each player, recompute their average weighted by the quality of opposition faced:
   ```
   adjusted_avg(batter) = weighted_mean(
       innings_runs,
       weights = opposition_bowler_quality
   )
   ```
3. But opposition quality depends on *their* opposition → iterate:
   ```
   Round 0: raw averages
   Round 1: adjusted by Round 0 opposition quality
   Round 2: adjusted by Round 1 opposition quality
   ...
   Round N: converged ratings
   ```
4. Use converged ratings instead of raw averages

**Related approaches**:
- **ESPNcricinfo Adjusted Averages**: For each innings, calculate the average quality of the bowling attack (weighted by overs bowled). Compare to a benchmark (geometric mean = 31.44). If attack is weaker, runs are discounted.
  ```
  Adjusted_Runs = Actual_Runs * (Benchmark / Opposition_Quality)
  ```
- **cricWAR (Runs Above Replacement)**: Assigns run values per ball based on game state, then measures each player's contribution above a replacement-level player.

**Pros**:
- Naturally handles league difficulty — a player who scores against better bowlers gets higher adjusted ratings
- No need for explicit league difficulty factors
- Self-calibrating through iteration
- Captures nuances that bridge-player averaging misses
- Mathematically principled (fixed-point iteration)

**Cons**:
- Computationally expensive (iterative convergence over 4M+ balls, multiple rounds)
- Can be unstable with sparse data (rare matchups may oscillate)
- Harder to interpret than raw stats
- Still partially disconnected — if all opponents are within one league, adjusted ratings only reflect within-league quality
- Needs careful implementation to maintain temporal integrity (can't use future opposition ratings to adjust past performances)
- Convergence not guaranteed without careful damping

**Cross-league verdict**: Partially solves it. Bridge players create indirect connections across leagues, but convergence is slower and noisier across league boundaries. Better than raw stats, comparable to but less transparent than explicit bridge-player calibration.

**Implementation effort**: ~1-2 weeks
**Expected impact**: High, but high complexity

---

### Option 5: Hierarchical Bayesian Model

**How it works**: This is the Opta/Stats Perform approach — the most sophisticated published system in cricket analytics.

```
P(outcome | batter_ability, bowler_ability, venue, league_difficulty, match_context)

# Hierarchical priors
batter_ability[i] ~ Normal(mu_league[l], sigma_league[l])
mu_league[l] ~ Normal(mu_global, sigma_global)
league_difficulty[l] ~ Normal(0, sigma_ld)
```

Fit jointly across ALL leagues using MCMC (PyMC, Stan, or NumPyro). League difficulty emerges as a latent variable. Player abilities are automatically comparable across leagues through partial pooling.

**Opta's specific implementation**:
- Takes all available ball-by-ball data from international AND domestic leagues
- Creates 6 standardized Bayesian metrics per player per format (0=worst to 1=best):
  - 3 runs-per-ball ratings (batter, bowler, venue)
  - 3 balls-per-out ratings (batter, bowler, venue)
- Key insight: "A maiden in the final over against a quality batter is more valuable than a maiden in the middle overs" — the system naturally devalues performances against weak opposition
- Venue ratings are standardized for historic player quality

**Partial pooling properties**:
- **No pooling** (treat leagues independently): Noisy, especially for leagues with few matches
- **Complete pooling** (treat all leagues as identical): Ignores real quality differences
- **Partial pooling** (hierarchical): Each league's player abilities are pulled toward the global mean, with shrinkage proportional to uncertainty. Data-poor leagues shrink more.

**Pros**:
- Theoretically the most principled approach
- Automatic uncertainty quantification
- League difficulty emerges naturally as a latent parameter
- Partial pooling handles data-sparse leagues gracefully
- Handles new players well (shrink toward prior)
- Can incorporate venue effects, home advantage, toss impact simultaneously

**Cons**:
- Very complex to implement (requires probabilistic programming framework)
- MCMC sampling is slow — hours to days on 4M balls
- Hard to integrate into existing XGBoost + cache pipeline architecture
- Requires careful model specification, prior selection, and convergence diagnostics
- The fitted player ratings essentially become the model — risks replacing XGBoost rather than supplementing it
- Overly academic for the expected marginal benefit vs simpler approaches
- Maintaining temporal integrity in a Bayesian setting requires online/sequential fitting

**Cross-league verdict**: Theoretically optimal. In practice, the amount of shared players still determines how well leagues are connected. With few bridges, the prior dominates and cross-league estimates are uncertain.

**Implementation effort**: ~3-4 weeks, plus significant refactoring
**Expected impact**: Highest potential, but diminishing returns vs simpler approaches

---

### Option 6: Player-Level ELO (Ball-by-Ball)

**How it works**: Treat each delivery as a mini-match between batter and bowler. Update both players' ELO after every ball based on outcome vs expectation.

```
# Per-ball update
expected_runs = DLS_par_score(over, wickets)  # or global average for this phase
actual_runs = ball_outcome

# Batter perspective: scoring above expectation = "winning"
batter_score = sigmoid(actual_runs - expected_runs)  # 0 to 1
bowler_score = 1 - batter_score

# ELO updates
E_batter = 1 / (1 + 10^((bowler_elo - batter_elo) / 400))
batter_elo += K * (batter_score - E_batter)
bowler_elo += K * (bowler_score - (1 - E_batter))
```

Team strength = sum (or weighted average) of player ELOs in the starting XI.

Features:
```
batting_team_elo  = sum(batter_elo for each player in batting XI)
bowling_team_elo  = sum(bowler_elo for each active bowler)
elo_differential  = batting_team_elo - bowling_team_elo
striker_elo       = current batter's ELO
bowler_elo        = current bowler's ELO
```

**Key design decisions**:
- **K-factor**: Very small (~0.5-2.0) since each ball is a tiny signal. Higher K = more responsive but noisier.
- **Baseline rating**: All players start at a default (e.g., 1500). New players with uncertain ability will regress as they face rated opponents.
- **Expected outcome baseline**: Can use DLS par scores, phase-specific averages (powerplay/middle/death), or global T20 scoring rates.
- **Separate batting/bowling ELO**: Each player has two ratings — one as a batter, one as a bowler.

**Pros**:
- **Naturally cross-league** — a player's ELO follows them between leagues. Virat Kohli's batting ELO reflects his IPL, T20I, and any other T20 cricket performance.
- **Implicit league calibration** — a BPL-only bowler who faces IPL-calibrated batters gets their ELO adjusted through the quality of opponents (who themselves carry ELO from facing other calibrated players). The connectivity flows through shared players, exactly like chess ratings in an open tournament.
- **Handles roster changes perfectly** — team strength = sum of *current* players' ELOs, not historical team identity. Mumbai Indians' strength automatically changes when they acquire/release players.
- **Granular updates** — 4M+ ball updates vs ~15K match updates. Much richer signal.
- **Fits existing pipeline architecture** — just add ELO tracking alongside the existing stats tracking in `parsing_v2.py`, then aggregate per team.
- **Temporally sound by construction** — ELO at any point reflects only past performance.
- No explicit league labeling needed — the cross-league adjustment is implicit.

**Cons**:
- K-factor tuning is important and tricky (too high = noisy, too low = slow adaptation, especially for new players)
- A player who faces weak bowling frequently still gets inflated ELO, though the "expected outcome" component partially addresses this
- Computationally moderate (one update per ball × 4M balls, but it's O(n) single-pass)
- New/rare players start at baseline, which may misrepresent their actual ability (cold start)
- ELO can drift — long-inactive players retain stale ratings (could add decay)
- Ball outcomes are noisy (a dot ball vs Bumrah isn't the same as a dot ball vs a part-timer, but ELO only sees the outcome)

**Cross-league verdict**: The most elegant solution for this project's architecture. ELO follows the *player*, not the *team*. Players who move between leagues act as natural bridges — their ELO is calibrated against opponents from all leagues they've participated in, and those opponents' ELOs are similarly calibrated through their own opponents, creating a connected rating graph.

**Implementation effort**: ~3-5 days
**Expected impact**: High — clean cross-league team signal with minimal new infrastructure

---

### Option 7: League-Tiered Team ELO with Bridge Calibration (Hybrid)

**How it works**: Combines Options 2 and 3:

1. Maintain per-team ELO within each league (standard team ELO)
2. Use bridge players to compute league difficulty offsets
3. Apply offsets to make ELOs comparable across leagues:
   ```
   adjusted_elo = raw_elo + league_offset
   ```
4. For international matches, map franchise ELO to national teams via player composition

Inspired by FiveThirtyEight's club soccer model, which uses Champions League results and player market values to calibrate across La Liga, Premier League, Bundesliga, etc.

**Pros**:
- Gets team-level signal (ELO) AND cross-league comparability (bridge calibration)
- Simpler than hierarchical Bayesian
- Captures team-level momentum and form within a league
- Can incorporate home advantage and margin of victory

**Cons**:
- Franchise roster changes make team ELO less meaningful (team identity ≠ team composition)
- Double complexity compared to either approach alone
- League offsets are noisy if few bridge players connect certain league pairs
- Requires maintaining parallel systems (team ELO + bridge player tracking + offset computation)
- FiveThirtyEight's soccer model benefits from Champions League as direct inter-league competition — cricket has no equivalent for franchise leagues

**Cross-league verdict**: Decent but convoluted. Player-level ELO (Option 6) achieves the same goal more cleanly by operating at the player level from the start.

**Implementation effort**: ~1 week
**Expected impact**: Medium-High, but more complexity than warranted

---

## 4. Comparison Matrix

| Approach | Cross-League | Complexity | Temporal Integrity | Signal Strength | Fit with Architecture |
|----------|-------------|-----------|-------------------|----------------|----------------------|
| 1. Aggregated stats | Weak | Very Low | Yes | Medium | Excellent |
| 2. Team ELO | None | Low | Yes | Medium (within league) | Good |
| 3. Bridge player factors | Strong | Medium | Yes | High | Good |
| 4. Opposition-adjusted | Moderate | High | Tricky | High | Moderate |
| 5. Hierarchical Bayesian | Strong | Very High | Yes | Highest | Poor (replaces pipeline) |
| 6. Player-level ELO | Strong | Medium | Yes | High | Excellent |
| 7. Tiered hybrid | Moderate | High | Yes | Medium-High | Moderate |

### Suitability for our specific architecture

Our system processes balls chronologically in `parsing_v2.py`, tracks per-player stats in a temporal cache, and uses ball-level features in XGBoost/LSTM/Transformer models. The ideal team-level feature approach should:

- Integrate into the existing chronological processing loop
- Be storable alongside existing player stats
- Be computable at simulation time from known lineups
- Not require a separate modeling framework

**Options 1 and 6 score highest** on architectural fit. Option 3 is also strong but requires new infrastructure (league tracking in stats cache).

---

## 5. What the Industry Does

### CricViz
- **Bridge player method** for league quality estimation (4,500+ bridge players analyzed)
- **Match Impact model** adjusted for conditions, venues, and match situations
- **xR/xW models** (Expected Runs / Expected Wickets) using ball-tracking + k-nearest neighbors
- Published league quality ranking: IPL > T20I > PSL/CPL/BBL > T20 Blast/BPL

### Opta / Stats Perform
- **Hierarchical Bayesian rating system** across ALL international and domestic leagues
- Six standardized ability metrics (0-1) per player per format
- Next Ball Predictor: Multilayer feedforward neural network
- Adjusts for venue, opposition quality, and game state simultaneously

### ESPNcricinfo
- **Adjusted Averages**: Weights innings by opposition bowling quality relative to a benchmark (geometric mean = 31.44)
- **Smart Stats**: Per-ball valuation multiplied by pressure-index incorporating batter quality, bowler quality, required rate, wickets in hand

### Betting Companies
- Typically use **Monte Carlo simulation** (10,000+ iterations)
- Combine historical data, player form, venue conditions, weather
- Sophisticated market-making models that self-correct through market forces
- Likely use opposition-adjusted player ratings, but methodologies are proprietary
- Achieve 2-5% margins implying very well-calibrated probabilities

### Academic Research
- **cricWAR** (Sloan Sports Conference): Runs Above Replacement per player. ~10 runs = ~1 win.
- **Bradley-Terry models**: Fitted on T20I fixtures with home advantage coefficient = 0.28. Used for tournament simulation (HeavyBail Statistics predicted Pakistan's 2022 T20WC performance)
- **Glicko-2 adaptation** (arXiv 2025): Recalibrated for cricket with d=85 scaling factor, margin of victory incorporation. Achieved 78.6% winner prediction on Test cricket.
- **DPPI** (Deep Player Performance Index): K-Means clustering + Random Forest + PCA composite index. "Aggregated DPPI values give approximate team strength."

### Key Takeaway from Industry

Every major analytics platform uses some form of **player-level quality assessment with opposition/context adjustment**. The specific method varies (Bayesian for Opta, bridge-player for CricViz, regression-based for ESPNcricinfo), but the principle is universal: raw stats are insufficient, and cross-league comparison requires either shared players or a hierarchical model.

---

## 6. Recommendation

### Proposed: Two-Phase Implementation

#### Phase 1 — Player-Level ELO + Aggregated Team Strength

**Why Player-Level ELO (Option 6) is the best fit**:

1. **ELO follows players, not teams** — naturally handles roster changes and cross-league movement without any explicit league tracking
2. **Ball-by-ball updates** — we already process 4M balls chronologically in `parsing_v2.py`; adding ELO tracking is a natural extension
3. **Sum-of-player-ELOs = team strength** — simple, interpretable, and accounts for actual squad composition
4. **Implicit league calibration** — the connected graph of batter↔bowler matchups naturally calibrates across leagues through shared players
5. **Excellent architectural fit** — integrates into existing chronological processing, storable in stats cache, computable at simulation time

**Combined with Aggregated Stats (Option 1)** for complementary signal:
- ELO captures relative quality (who beats whom)
- Aggregated stats capture absolute performance levels (run rates, economy)
- Together: richer team representation than either alone

**New features (6-8 total)**:
```
batting_team_elo       = sum of 11 players' batting ELO
bowling_team_elo       = sum of active bowlers' bowling ELO
elo_differential       = batting_team_elo - bowling_team_elo
team_batting_avg       = mean of 11 players' batting averages
team_bowling_avg       = mean of bowlers' bowling averages
relative_strength      = team_batting_sr - opp_bowling_econ (or similar composite)
```

**Estimated effort**: ~3-5 days
**Expected impact**: First meaningful team-level signal in the model

#### Phase 2 — Bridge Player League Normalization (if needed)

If Phase 1's implicit ELO calibration proves insufficient for cross-league prediction:

1. Add league tracking to the parsing pipeline and stats cache
2. Compute explicit league difficulty factors from bridge player performance differentials
3. Normalize raw stats by league difficulty before aggregating into team strength features
4. This gives both ELO-based team strength AND league-adjusted statistical team strength

**Estimated effort**: ~3-5 additional days
**Decision point**: Evaluate Phase 1 results against betting odds before committing to Phase 2

---

## 7. Sources

### Rating Systems
- [ICC Rankings Methodology](https://www.icc-cricket.com/rankings/about)
- [ICC T20I Team Rankings — Wikipedia](https://en.wikipedia.org/wiki/ICC_Men%27s_T20I_Team_Rankings)
- [Cricket ELO with DLS Par Score — SwingDoctor/Substack](https://swingdoctor.substack.com/p/cricket-data-science-101-elo-ratings)
- [Augmented Glicko Rating for Test Cricket — arXiv:2603.02574](https://arxiv.org/html/2603.02574)
- [chinmay-choudhary/cricket-elo-rating — GitHub](https://github.com/chinmay-choudhary/cricket-elo-rating)
- [AusSportsTipping IPL ELO Ratings](https://www.aussportstipping.com/sports/ipl/elo_ratings/)
- [TrueSkill 2 — Microsoft Research](https://www.microsoft.com/en-us/research/publication/trueskill-2-improved-bayesian-skill-rating-system/)
- [ELO-type Rating for Variable Team Composition — Royal Society](https://royalsocietypublishing.org/doi/10.1098/rsta.2021.0155)

### Cross-League Comparison
- [CricViz: Evaluating Different Standards of T20 Cricket](https://cricviz.com/evaluating-different-standards-of-t20-cricket/)
- [CricViz/Wisden: Why IPL is Higher Quality than T20Is](https://wisden.com/stories/opinion/cricviz-why-the-ipl-is-higher-quality-than-t20is-and-the-t20-blast-isnt)
- [FiveThirtyEight Club Soccer Predictions Methodology](https://fivethirtyeight.com/methodology/how-our-club-soccer-predictions-work/)
- [FootballDatabase Ranking Methodology](https://www.footballdatabase.com/methodology)
- [Massey Ratings Theory](https://masseyratings.com/theory/massey.htm)

### Opposition-Adjusted Statistics
- [ESPNcricinfo: Adjusting Averages for Bowling Strengths](https://www.espncricinfo.com/story/adjusting-averages-to-account-for-bowling-strengths-612664)
- [ESPNcricinfo Smart Stats](https://www.espn.com/cricket/story/_/id/23046591/espncricinfo-smart-stats-new-way-understand-t20-cricket)
- [xR with Machine Learning — Cricket Savant](https://cricketsavant.wordpress.com/2017/01/24/xr-with-machine-learning/)
- [cricWAR: Player Evaluation — Sloan Sports Conference](https://www.sloansportsconference.com/research-papers/cricwar-a-reproducible-system-for-evaluating-player-performance-in-limited-overs-cricket)
- [RAAR Metric — ESPNcricinfo](https://www.espncricinfo.com/story/using-the-runs-above-average-replacement-metric-to-assess-the-quality-of-test-batsmen-1226970)

### Bayesian & Hierarchical Models
- [Opta: Introducing Cricket Simulation Models](https://theanalyst.com/articles/introducing-cricket-simulation-models)
- [Opta: Next Ball Predictor](https://theanalyst.com/articles/opta-next-ball-predictor)
- [PyMC Hierarchical Rugby Model](https://www.pymc.io/projects/examples/en/latest/case_studies/rugby_analytics.html)
- [Bayesian Partial Pooling — Bayesian Notes](https://jrnold.github.io/bayesian_notes/shrinkage-and-hierarchical-models.html)
- [Bayesian Statistics Meets Sports — de Gruyter](https://www.degruyterbrill.com/document/doi/10.1515/jqas-2018-0106/html?lang=en)

### Bradley-Terry & Paired Comparison
- [Bradley-Terry Model for ODI Cricket — Journal of Data Science](https://jds-online.org/journal/JDS/article/292/file/pdf)
- [HeavyBail Statistics: T20 World Cup Simulation](https://www.heavybailstatistics.com/post/simulating-the-2022-icc-men-s-t20-world-cup)

### Player Evaluation & Team Strength
- [DPPI for T20 Cricket — ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2772662222000029)
- [Enhanced Cricket Match Prediction — Nature (2026)](https://www.nature.com/articles/s41598-026-36555-6)
- [Player Evaluation in T20 Cricket — Swartz/SFU](https://www.sfu.ca/~tswartz/papers/moneyball.pdf)
- [White Ball Analytics](https://www.whiteballanalytics.com/)

### Industry & Betting
- [ML in Sports Betting: Systematic Review — arXiv](https://arxiv.org/html/2410.21484v1)
- [Stats Insider Cricket Model](https://www.statsinsider.com.au/about-us)
- [T20 Ball Simulation Model — dr00bot](https://dr00bot.com/blog/t20-cricket-simulation-engine)
