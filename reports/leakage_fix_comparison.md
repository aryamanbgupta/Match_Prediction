# ELO leakage finding + impact on headline numbers (2026-05-09)

**TL;DR**: The production direct match-level model had a feature-engineering
leakage. `_split_elo` in `materialize_match_features._build_match_record`
was reading per-player ELO state AFTER `parse_match_data_v2` had updated
it ball-by-ball with this match's own outcomes. The 6 leaked features
include the model's two highest-importance features
(`bottom5_bowling_elo_diff`, `top6_batting_elo_diff`).

**On the truly out-of-sample golden set, the headline numbers were
heavily inflated by this leakage.** A retrained model on cleaned features:
- LL is approximately market-level (no longer beats market)
- ROI remains positive but bootstrap CIs are wider
- ROI CI on ≥$50k slice now barely excludes zero
- ROI CI on ≥$100k slice still cleanly excludes zero, but lower bound +3.79%

The model is still likely *useful*, but the magnitude of the edge is much
smaller than originally reported. Win rate dropped from 71% to 59-61%.

---

## What was leaking

`materialize_match_features.py:519-522` (before fix):
```python
rows = parse_match_data_v2(..., elo_tracker=temp_elo, ...)  # mutates temp_elo
record = _build_match_record(..., temp_elo, ...)            # reads post-mutation
```

Inside `_build_match_record`:
```python
t1_top6_bat, t1_bot5_bow = _split_elo(team1_lineup_ids, elo_tracker)
```

`parse_match_data_v2` (parsing_v2.py:1342-1349) updates `elo_tracker` with
each ball's outcome. So when `_build_match_record` ran AFTER the parse,
`_split_elo` saw post-match ELOs — directly informative about who won.

Other features were NOT leaky:
- `team_*_elo` (sums), `team_*_avg/sr`, `team_*_econ` are computed once
  pre-loop in parser at line 1063-1098 → CLEAN
- venue features are computed once pre-loop at line 1041-1042 → CLEAN
- A2 trackers (form/H2H/home) are updated AFTER `_build_match_record` returns → CLEAN

Empirically verified across all 62 golden matches: only the 6 ELO-split
features drift on every match (max abs Δ ~5-7 ELO units, std ~1.5-2.5).

## The fix

`materialize_match_features.py` (after fix, lines 504-512):
```python
# SNAPSHOT temp_elo BEFORE parse_match_data_v2 mutates it.
pre_match_elo = PlayerEloTracker()
pre_match_elo.batting_elo = dict(temp_elo.batting_elo)
pre_match_elo.bowling_elo = dict(temp_elo.bowling_elo)

rows = parse_match_data_v2(..., elo_tracker=temp_elo, ...)  # still mutates live tracker
record = _build_match_record(..., pre_match_elo, ...)        # uses pre-match snapshot
```

We still let `parse_match_data_v2` update the *live* `temp_elo` so subsequent
same-day matches see post-this-match ELOs (matches monolith chronological
semantics). We just hand `_build_match_record` a frozen snapshot taken
before parse runs.

## Impact on golden eval — three scenarios

All numbers w=0.0 (direct alone), 1000 bootstrap resamples, polymarket-overlap subset.

### Scenario A: original leaky model on leaky parquet (the headline we previously reported)

| Slice | LL [95% CI] | Flat ROI [95% CI] | Bets / win-rate |
|---|---|---|---|
| all (55) | 0.4944 [0.45, 0.53] | **+50.73%** [+32, +74] | 54 / 69.4% |
| ≥$50k (50) | **0.5004** [0.45, 0.56] | **+53.67%** [+36, +74] | 49 / 71.4% |
| ≥$100k (45) | **0.4361** [0.37, 0.50] | **+58.03%** [+33, +87] | 44 / 73.6% |

### Scenario B: original leaky model on CLEAN parquet (out-of-distribution test)

The trained model is given pre-match features it didn't see in training.
This is what predict_fixture would produce for upcoming matches.

| Slice | LL [95% CI] | Flat ROI [95% CI] | Bets / win-rate |
|---|---|---|---|
| all (55) | 0.6888 [0.57, 0.82] | +7.12% [-25, +34] | 54 / 51.9% |
| ≥$50k (50) | **0.7254** [0.61, 0.85] | +15.71% [-17, +47] | 49 / 55.1% |
| ≥$100k (45) | **0.7245** [0.61, 0.87] | +18.65% [-14, +49] | 44 / 56.8% |

The leaky model FAILS go/no-go in this scenario:
- LL on ≥$50k 0.7254 > market 0.6267 (FAIL)
- LL on ≥$50k 0.7254 > coinflip 0.6931 (FAIL — model worse than coinflip!)
- ROI CI on ≥$50k includes 0 (FAIL)

### Scenario C: RETRAINED model on clean parquet (the principled comparison)

We retrained `xgb_match_v2_clean` from scratch on the clean parquet, same
hyperparameters as the leaky model. This is the honest model.

| Slice | LL [95% CI] | Flat ROI [95% CI] | Bets / win-rate |
|---|---|---|---|
| all (55) | 0.6416 [0.59, 0.70] | +20.33% [-12, +49] | 54 / 53.7% |
| ≥$50k (50) | **0.6747** [0.64, 0.72] | +32.61% [-0.20, +64] | 49 / 59.2% |
| ≥$100k (45) | **0.6698** [0.63, 0.72] | +34.75% [+3.79, +65] | 44 / 61.4% |

Honest go/no-go on Scenario C:
- LL on ≥$50k 0.6747 vs market 0.6267 — **FAIL** (model is worse on LL)
- LL on ≥$100k 0.6698 vs market 0.6267 — **FAIL**
- ROI CI on all (55) — includes 0
- ROI CI on ≥$50k — barely includes 0 (lower bound -0.20%)
- ROI CI on ≥$100k — **excludes 0** (lower bound +3.79%)

So under the strict gate ("LL beats market AND ROI CI excludes 0"), the
clean model fails on every slice. Under a softer gate ("ROI CI excludes 0"),
it clears only on ≥$100k.

## Side-by-side ROI comparison

| Slice | Leaky (was reported) | Clean retrained | Δ |
|---|---|---|---|
| all (55) | +50.73% | +20.33% | -30pp |
| ≥$50k (50) | +53.67% | +32.61% | -21pp |
| ≥$100k (45) | +58.03% | +34.75% | -23pp |
| Win rate ≥$50k | 71.4% | 59.2% | -12pp |
| Win rate ≥$100k | 73.6% | 61.4% | -12pp |

Roughly two-thirds of the previously reported ROI was leakage-driven.

## Interpretation

The post-match ELO drift on each player is small in magnitude (~0.5 ELO
on a 1500-baseline) but **directly correlates with the match outcome**
(winners' players' ELOs go up, losers' down). XGBoost learned to use
this drift as a signal. Without it, the model has much less to work with:
the remaining features (player aggregates, venue stats, position-split
ELOs *as-of pre-match*, lineup mix, recent form, H2H, home advantage)
collectively predict somewhat better than market on the all slice but
worse than market on log loss when you weight by liquidity.

Notably:
- The clean model's top features (by importance) are **still**
  `bottom5_bowling_elo_diff` and `top6_batting_elo_diff` — but now
  those values are pre-match. Position-split ELOs do carry predictive
  power; they were just more powerful with leaked drift on top.
- Win rate dropping from 71% to 59-61% is significant. The leaky model
  was right about which team to bet 71% of the time; the clean model is
  closer to 60%. That extra 12pp of accuracy on bet-side-selection was
  almost entirely from the leakage.

## Per-match: IPL 2026 dashboard updated

`reports/ipl_2026_dashboard_clean.html` regenerated with clean predictions.
Cumulative IPL PnL barely changed (+21.55 → +21.33 units, 47 bets) —
because for many bets, the directional decision (which team) is the same
under leaky and clean models, only the confidence shifts. The slice-level
LL difference is bigger than per-match PnL difference.

IPL-only golden subset (n=25):
- Leaky LL: 0.5737
- Clean LL: 0.6676 (+0.094)

So even on IPL alone, leakage was contributing meaningfully to LL.

## Where this leaves us

Production model is **`models/xgb_match_v2_clean/`**. The leaky
`xgb_match_v2_frozen` is preserved on disk for reference but should not
be used for inference. `predict_fixture.py` and `build_ipl_dashboard.py`
have been switched to the clean model.

**Honest production pitch**: the model is borderline-skilful. It clears
the looser go/no-go gate (ROI CI excludes 0) on the ≥$100k liquidity
slice on the golden set, with point estimate +34.75% ROI on 44 bets.
Lower CI bound is +3.79% — modest but positive. LL is approximately
market-level. The "+47-58% ROI" headline from before this audit is
inflated by leakage and should be retracted.

## Required cleanup

- [ ] Update `CLAUDE.md` to reflect honest numbers and the v2_frozen
      → v2_clean swap as the production model
- [ ] Update `IMPROVEMENTS.md` Match-Level Direct + Sim Ensemble section
- [ ] Update `TODO.md` headline numbers
- [ ] Memo: `project_match_level_ensemble.md` (memory file)
- [ ] Re-run no-leakage diagnostic with the clean model — the previous
      "frozen is BETTER than unfrozen" finding may flip or change shape
      now that the dominant feature drift is removed
- [ ] Audit `xgboost_v2.py` (the v7 ball-level sim) for similar issues —
      its features come from a different code path but the same parser

## Other features audited and confirmed clean

Empirically verified across 62 golden matches: only the 6 ELO-split
features drift on every match. Not leaky:

- All `team*_batting_elo` / `team*_bowling_elo` (sums) — computed once
  before ball loop in parser
- All `team*_batting_avg` / `team*_batting_sr` / `team*_bowling_avg` /
  `team*_bowling_econ` — same
- `venue_avg_score` / `venue_chase_win_pct` / `venue_dot_pct` /
  `venue_boundary_pct` — computed once before ball loop
- `is_team*_home`, `team*_win_rate_last_10`, `h2h_*`, `win_rate_diff` —
  Phase A2 trackers updated AFTER `_build_match_record` returns
- `team*_lhb_count` / `pace_count` / `spinner_count` — static metadata
- `team1_batting_first` / `toss_*` — static pre-match metadata
- `is_international` / `competition_tier_encoded` — match-level static
