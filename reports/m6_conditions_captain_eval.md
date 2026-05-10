# M6 — conditions / captain (2026-05-10)

Phase 6 of match-level v3. Designed match-level conditions features (date-derived). **Outcome: DROPPED on iteration evidence**, with a discipline upgrade.

**Production baseline unchanged**: `models/xgb_match_v3_m2_venue_only_unfrozen/`.

## Scope adjustments

- **Captain features SKIPPED**: cricsheet doesn't tag captains in `info`. The "first player in lineup" heuristic is unreliable, and noisy captain identification would re-create the M4 scheduling failure (near-zero target signal picked up via interactions). Defer until cricsheet adds captain tagging or an external captain-identity source lands.
- **Start time / `is_day_match` SKIPPED**: cricsheet has no start-time field. Could infer from venue + month, but heuristic.

## Features designed (3) and correlation check

Match-level scalars derived from `match_date` alone — no lineup aggregation, so they avoid the M3/M4/M5 collapse pattern:

| Feature | target r | Top baseline correlation | Pass redundancy? | Pass target-floor (≥ 0.03)? |
|---|---|---|---|---|
| `month_of_year` (1-12) | +0.0052 | `is_international` r=+0.198 | YES (no |r|>0.5) | **NO** (|r|=0.005 < 0.03) |
| `day_of_week` (0-6) | +0.0067 | `team2_batting_sr` r=+0.061 | YES | **NO** (|r|=0.007) |
| `is_dew_prone_month` | -0.0102 | `venue_pw` r=-0.116 | YES | **NO** (|r|=0.010) |

**Redundancy check passes (clean orthogonality vs baseline) but target-correlation floor fails for all 3.** Target r is essentially zero — these features carry no marginal-rank signal about team1_wins. The hope was XGBoost would find interactions (e.g., month × venue → seasonal dew effects). It does, but those interactions fit noise.

I trained anyway to confirm the M4 over-confidence pattern repeats. It did.

## Iteration evidence (the gate)

| Variant | iter ≥$50k LL raw | iter ≥$50k ROI raw [CI] | iter ≥$50k LL Platt | iter ≥$50k ROI Platt [CI] |
|---|---|---|---|---|
| **M2 v.o. unfrozen (baseline)** | **0.6348** | **+25.40% [+4.75, +48.11]** | **0.6279** | **+26.69% [+6.26, +48.65]** |
| M6 full (3 features) | 0.6412 | +23.53% [+2.28, +46.45] | 0.6341 | +24.47% [+3.81, +46.55] |
| M6 month-only | 0.6372 | +21.82% [+1.89, +44.55] | 0.6350 | +15.14% [-4.19, +37.22] |
| M6 − month_of_year | (drop-one only ran standalone) | — | — | — |
| M6 − day_of_week | (drop-one only ran standalone) | — | — | — |

All M6 variants regress on iteration ≥$50k LL by Δ +0.005-0.010, and Platt ROI degrades. M6 month-only Platt CI now includes 0 (worst calibrated ROI of any v3 variant).

## Drop-one ablation (standalone test LL)

| Variant | val LL | test LL | Δ test LL vs M2 v.o. |
|---|---|---|---|
| M2 v.o. unfrozen | 0.6521 | 0.6015 | — |
| M6 full | 0.6498 | 0.5994 | -0.0021 |
| M6 − month_of_year | 0.6504 | 0.5985 | -0.0030 |
| M6 − day_of_week | 0.6513 | 0.5983 | -0.0032 |
| M6 − is_dew_prone_month | 0.6505 | 0.5983 | -0.0032 |
| **M6 month-only** | 0.6517 | **0.5944** | **-0.0071** |

Misleading: M6 month-only has the BEST standalone test LL (Δ -0.007) but the WORST iteration ROI. The standalone improvement is from over-confidence, not real signal.

## Tail-accuracy diagnosis (M6 month-only vs M2 v.o.)

| Confidence band | M2 v.o. n / acc | M6 month-only n / acc | Δacc |
|---|---|---|---|
| \|p−0.5\| > 0.05 | 510 / 73.5% | 546 / 74.2% | +0.6pp |
| \|p−0.5\| > 0.10 | 324 / **80.6%** | 374 / 78.9% | -1.7pp |
| \|p−0.5\| > 0.15 | 225 / **85.3%** | 263 / 82.1% | **-3.2pp** |
| \|p−0.5\| > 0.20 | 154 / **89.6%** | 185 / 87.6% | -2.0pp |

Mean |p-0.5|: 0.114 → 0.128 (+13% confidence). M6 month-only makes 50+ MORE high-confidence predictions but each is materially less accurate. Same mechanism as M4 scheduling: near-zero-target-correlation features picked up via tree interactions fit train noise.

## Discipline upgrade — target-correlation floor

Added to `feedback_correlation_check_before_features.md`:

> 2. **Target-correlation floor** (added 2026-05-10 after M6) — |target r| ≥ 0.03. Features with target |r| < 0.03 are essentially noise from the model's perspective and XGBoost will misuse them via tree interactions, fitting noise that doesn't transfer to test (the M4 scheduling and M6 month_of_year failure mode).

The original M5 correlation check would have passed M6 features (clean redundancy check). The new dual-condition rule (redundancy AND target floor) catches both failure modes:
- M3, M4 form, M5 player×opp: redundancy failure
- M4 scheduling, M6 conditions: target-floor failure

## Why the v3 phased plan keeps failing — root cause analysis

**5 of 5 named feature phases (M3–M7)** have now landed as DROP. The pattern:

| Phase | Failure mode | Root cause |
|---|---|---|
| M3 rolling form | redundancy | last-5-match aggregates collapse to career stats under lineup-mean |
| M4 form | redundancy | 60d/365d windows correlate 0.7-0.8 with M1's `win_rate_diff` |
| M4 scheduling | target floor | `days_since_*` near-zero target r, picked up via interactions, fits noise |
| M5 player×opp | redundancy | h2h-aggregated stats collapse to career stats under lineup-mean |
| M6 conditions | target floor | month/day-of-week near-zero target r |

**Common thread**: The match-level model with M1+M2-venue-only features (49 features) appears to be at a local LL optimum given the available pre-match information signal. Adding more features that operate on the same predictive variance (form, momentum, conditions) over-fits or over-confidently re-uses the existing signal.

**What might break this pattern (for M7)**:
- Features that capture *new orthogonal signal types* — not derivative of team strength, recent form, or h2h:
  - **Market-deviation residual** (deferred per project policy: no market data on training corpus)
  - **Per-player matchup at ball level** (already exploited by v7 sim, but match-level supervision can't reconstruct it from aggregates)
  - **Hyperparameter / architecture tuning** (B1 in catalog) — likely the path with most remaining gain
  - **Stacking with disjoint feature subsets** (B2)
  - **Per-tier model specialization** (B5) — IPL-specific model trained only on tier-3 matches
- Features that capture single-match-instance signals (squad changes mid-tournament, injury news, weather) — but these need external data sources

## Status of M6 verification criteria

1. ❌ **iteration ≥$50k LL improvement OR clear ROI lift on IPL slice**: not cleared — LL regressed +0.006, IPL slice (not run on M6 due to dominance of broader regression) wouldn't justify keeping M6 alone.
2. ✅ **Pre-training correlation check (redundancy)**: passed for all 3 M6 features.
3. ✅ **Discipline upgrade**: target-correlation floor identified and added to the rule.
4. ✅ **Drop-one + tail-accuracy**: confirmed M4-style over-confidence pattern.

## What this means for M7

- **M7 baseline still `models/xgb_match_v3_m2_venue_only_unfrozen/` (raw)**.
- **Numbers to beat unchanged** (this baseline has held since M2): iter ≥$50k raw LL 0.6348 / ROI +25.40%; Platt LL 0.6279 / ROI +26.69% [+6.26, +48.65].
- **M7 should be primarily ARCHITECTURE work, not feature work** — the feature-engineering search space has been thoroughly explored and the marginal frontier is exhausted. Move to:
  - Hyperparameter resweep (B1)
  - Stacking with disjoint feature subsets (B2)
  - Per-tier specialization (B5) — IPL-only model trained on tier-3 only
  - LightGBM/CatBoost (B6) low priority
- **Captain features remain deferred** until reliable captain identification is available (post M7, possibly via cricsheet update or external source).

## Artifacts preserved

- `models/xgb_match_v3_m6_unfrozen/` — full M6 model
- `models/xgb_match_v3_m6_{no_month,no_dow,no_dew,month_only}/` — drop-one variants
- `data/xgb_match_data_v3_m6_unfrozen/` — 110-feature parquet
- `_match_conditions_features` helper kept in materializer (behind `--drop-features` filter)
- `/tmp/claude/m6_corr_check.py`, `/tmp/claude/m6_tail_check.py` — reusable patterns

## Headline (one-line)

M6's date-derived features pass the redundancy check but fail a newly-identified target-correlation-floor check (|target r| ≥ 0.03). Same M4 over-confidence failure mode at smaller magnitude. Discipline upgraded for M7+: dual-condition correlation check.
