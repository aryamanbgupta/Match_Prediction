# M5 — player × opposition affinity (2026-05-10)

Phase 5 of match-level v3. Investigated player × opposition affinity features via h2h-matrix aggregation. **Outcome: DROPPED before training**, caught by the pre-training correlation-check discipline (established post-M4).

**Production baseline unchanged**: `models/xgb_match_v3_m2_venue_only_unfrozen/`.

## Hypothesis

For each top-6 batter in team1, aggregate (runs, balls, dismissals) across the h2h matrix against every player in team2's lineup. Compute the batter's expected avg/SR against this specific attack; shrink to career avg/SR via convex combination with k_balls weight. Aggregate to lineup level by mean over top-6. Symmetric for team2. The intent: capture "Pollard vs Mumbai Indians" or "Warner vs CSK" effects that team-level career stats hide.

## Features designed (8)

```
team1_top6_avg_vs_opp_shrunk, team1_top6_sr_vs_opp_shrunk
team2_top6_avg_vs_opp_shrunk, team2_top6_sr_vs_opp_shrunk
avg_vs_opp_diff, sr_vs_opp_diff
team1_h2h_balls_total, team2_h2h_balls_total
```

Implementation: `_player_vs_opposition_features` helper in `materialize_match_features.py`. Pre-parse via live `temp_stats.h2h_stats` dict (chronologically correct as of match date).

## Correlation check (the discipline)

**k_balls = 60** (one innings worth of evidence required for ~50% cell weight):

| M5 feature | top baseline correlation | r | M5 target r | baseline target r | M5/baseline target ratio |
|---|---|---|---|---|---|
| `team1_top6_avg_vs_opp_shrunk` | `team1_batting_avg` | +0.709 | +0.095 | +0.108 | 0.89× |
| `team1_top6_sr_vs_opp_shrunk` | `team1_batting_sr` | +0.771 | +0.107 | +0.107 | 1.00× |
| `team2_top6_avg_vs_opp_shrunk` | `team2_batting_avg` | +0.630 | -0.038 | -0.075 | 0.51× |
| `team2_top6_sr_vs_opp_shrunk` | `team2_batting_sr` | +0.728 | -0.044 | -0.074 | 0.60× |
| `avg_vs_opp_diff` | `batting_avg_diff` | +0.592 | +0.116 | +0.149 | 0.78× |
| `sr_vs_opp_diff` | `batting_avg_diff` | +0.447 | +0.148 | +0.149 | 0.99× |
| `team1_h2h_balls_total` | `h2h_n_meetings` | +0.536 | +0.003 | +0.020 | 0.16× |
| `team2_h2h_balls_total` | `h2h_n_meetings` | +0.562 | +0.005 | +0.020 | 0.24× |

**Every M5 feature** either:
- Has |r| > 0.5 with an existing baseline feature AND no higher target correlation (rows 1–5, 7, 8), OR
- Is borderline (`sr_vs_opp_diff` at |r|=0.447) but target r is essentially identical to baseline (0.99×).

**Per the discipline** (`feedback_correlation_check_before_features.md`): all 8 features SKIP.

## Sanity check at low shrinkage (k_balls = 10)

Re-materialized with k_balls=10 to test whether heavier reliance on raw h2h cells (less career-shrinkage) would surface orthogonal signal:

| Feature | k=60 target r | k=10 target r | Δ |
|---|---|---|---|
| `team1_top6_avg_vs_opp_shrunk` | +0.0954 | +0.0892 | -0.006 |
| `team1_top6_sr_vs_opp_shrunk` | +0.1066 | +0.1095 | +0.003 |
| `team2_top6_avg_vs_opp_shrunk` | -0.0381 | -0.0437 | -0.006 |
| `team2_top6_sr_vs_opp_shrunk` | -0.0443 | -0.0525 | -0.008 |
| `avg_vs_opp_diff` | +0.1160 | +0.1115 | -0.005 |
| `sr_vs_opp_diff` | +0.1480 | +0.1523 | +0.004 |

Target correlations are essentially unchanged. Baseline correlations decrease modestly (e.g., `team1_top6_avg_vs_opp_shrunk` ↔ `team1_batting_avg` goes 0.71 → 0.59) but the per-team features are still r > 0.5 with the equivalent career feature.

## Why lineup aggregation kills the matchup signal

At the per-(batter, bowler) cell level, h2h stats may carry real signal: Pollard's record against Bumrah is genuinely different from Pollard's overall record. But:

- Most (batter, bowler) cells have 0 balls (players haven't faced each other)
- Cells with non-zero data are dominated by 2-4 specific matchups; the remaining 7-9 opp players don't contribute
- Summing across 11-player opposition lineup AVERAGES OUT player-specific signal — over 11 opp players, the aggregate h2h profile looks ~ identical to the batter's overall profile
- Career-aggregate `team1_batting_avg` already captures "this batter's mean across all opposition encountered" — h2h-aggregate gives essentially the same number

**Structural result**: matchup affinity at team level is redundant with team-level career aggregates. The v7 ball-level sim already exploits per-ball matchup features (`bowler_pX_vs_lhb/rhb`, `batter_pX_vs_pace/spin`); collapsing those to match level via lineup-aggregate doesn't preserve the signal.

## Sample-size features fail for a different reason

`team1_h2h_balls_total` and `team2_h2h_balls_total` are essentially redundant with `h2h_n_meetings` (M1 A2), which is itself a near-zero-target-correlation feature (+0.020). Adding two more variants doesn't help — they all encode "have these teams played each other often?" which the model already knows from `h2h_n_meetings`.

## Player × venue: deferred, not attempted

The plan listed player × venue as the M5 second half (needs SCHEMA_VERSION=5 schema bump). Given player × opposition failed for structural reasons (lineup aggregation collapses signal), the same failure mode likely applies to player × venue: aggregating "Pollard's record at Wankhede" across the top-6 batting lineup collapses toward the team's overall record at Wankhede, which `is_team1_home` + `venue_p4/p6/pw` already encode. Defer until M7+ unless we find a way to surface per-player affinity that doesn't collapse under aggregation (e.g., variance-weighted aggregation, or per-player features as additional inputs not just means).

## Status of M5 verification criteria

1. ❌ **iteration ≥$50k Δ LL ≤ -0.003**: not tested — correlation check failed first
2. ❌ **Clear lift on opposition-specific slice**: not tested
3. ✅ **Pre-training correlation check**: caught all 8 features as redundant with M1+M2 baseline. Saved a training run and a full ablation.
4. ✅ **Lower-shrinkage sanity check**: confirmed the redundancy isn't an artifact of the shrinkage choice. The hypothesis itself is structurally flawed at the match level.

## What this means for M6

- **M6 baseline still `models/xgb_match_v3_m2_venue_only_unfrozen/` (raw)**.
- **Numbers to beat unchanged**: ≥$50k raw LL 0.6348 / ROI +25.40% [+4.75, +48.11].
- **M6 features (conditions, captain) must pass the same correlation check**: month_of_year × venue interaction must not be |r|>0.5 with existing venue features; captain win-rate features must not be |r|>0.5 with `team1_win_rate_last_10`.
- **Lesson**: features that collapse to existing aggregates under lineup-mean don't help. M6 should focus on features that are *match-level by nature* (not player-aggregated) — captain identity is per-match (not aggregated), pitch conditions are per-match (not aggregated), so these are cleaner candidates.
- **Player × venue deferred indefinitely**: any feature pattern requiring "aggregate per-player stat at team level" should be treated with high prior skepticism after M3/M4/M5 all failed this way.

## Artifacts preserved

- `data/xgb_match_data_v3_m5_unfrozen/` — full 107-column parquet with M5 features (k_balls=60)
- `data/xgb_match_data_v3_m5_unfrozen_k10/` — same with k_balls=10 (sanity check parquet)
- `scripts/materialize_match_features.py:_player_vs_opposition_features` — helper kept for future re-evaluation
- `/tmp/claude/m5_corr_check.py` — correlation-check pattern reusable for future M-phases

## Headline (one-line)

Player × opposition affinity at team level collapses to team-level career aggregates under lineup-mean aggregation. The correlation check caught this before training; saved compute and confirmed the M3→M4→M5 pattern: **aggregated-player features don't beat team-level career features**.
