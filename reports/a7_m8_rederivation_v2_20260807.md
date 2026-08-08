# A7 + M8 re-derivation on v2 prices (2026-08-07)

Closes the two P0 items from the 2026-08-05 market-benchmark correction
(`reports/market_benchmark_toss_defect_20260805.md`): the M8 edge-threshold
choice and the A7 slice-conditional rule were both selected on the corrupt
(toss-market) ROI surface and are retracted/withdrawn. This is a **fresh
derivation on `betting_odds_polymarket_v2.json`** under the I3
`tournament_time_block_v1` bootstrap — not a re-run of the retracted
analyses.

**This section (context, inputs, sweeps, decision rules) was written and
committed BEFORE any sweep was executed.** The Results section is filled in
afterward. No decision rule may be altered after the first sweep runs.

## Inputs

- **Primary (production arm)**: `models/xgb_match_i7_swap_production`
  predictions, resliced against v2 odds by the 2026-08-05 audit:
  `eval_out/toss_defect_20260805/pairsl_i17_swap_corrected/envN_i7_corrected_w0p00_{min_volume_50000,min_volume_100000}.json`
  (eval_out is regenerable: `reslice_eval_json.py --odds
  betting_odds_polymarket_v2.json --cluster-source-dir
  data/polymarket_test_v2`). Feature parquet for slice membership:
  `data/xgb_match_data_i7_v2/test.parquet` (frame of record).
  **Input-integrity check**: the script's baseline row must reproduce the
  restated headline (≥$50k: ROI +3.38% [−14.63, +37.06], 18 blocks;
  ≥$100k: −5.19% [−28.73, +27.50], 11 blocks) before any variant is read.
- **Sensitivity (legacy arm)**: `pairsl_d12_swap_corrected/` twins
  (`models/xgb_match_v3_m7_swap_production`; restated ≥$50k +7.40%
  [−7.65, +34.82], ≥$100k −0.54%). Feature parquet:
  `data/xgb_match_data_v3_m6_unfrozen/test.parquet` (the exact parquet the
  original A7/A11 harness used).
- Tooling: `scripts/auto/a7_conditional_threshold.py` unchanged (I3
  whole-competition bootstrap, seed 42, 10,000 resamples; framework
  `realized_pnl` / `flat_bet_*` helpers). LL is untouched by construction —
  the model-vs-market LL verdict (0.6249 vs 0.5940 at ≥$50k, model loses)
  stands regardless of this report.

## Sweeps

1. **M8 global edge-threshold sweep** (flat 1-unit staking): thresholds
   {0.01, 0.02, 0.03, 0.05, 0.10} applied to every fixture, implemented as
   `--boundary 0` (every fixture with |top6_batting_elo_diff| > 0 requires
   edge > thr). Fixtures with a missing feature row default to diff 0.0 and
   are always kept — their count is reported. Baseline = flat threshold 0
   (all placed bets). Kelly variants are out of scope: the retraction kept
   flat staking and only the threshold choice needs re-derivation.
2. **A7 conditional grid**: boundary {3, 5, 8, 12} × mismatch edge
   threshold {0.05, 0.10, 0.15}; close side always flat at threshold 0.
   Both slices, both arms.

## Pre-committed decision rules

- **R1 (M8 threshold)**: flat 1-unit at threshold 0 remains the default
  reporting rule. A global threshold replaces it only if, on the primary
  arm, it improves ROI over baseline on BOTH slices by more than 2.3pp
  (the repo ROI noise floor) AND its ≥$50k block CI excludes 0.
- **R2 (A7 re-validation)**: the conditional rule is re-validated only if
  some grid cell, on the primary arm, (a) improves ROI over baseline on
  BOTH slices by more than 2.3pp, (b) has a ≥$50k block CI excluding 0,
  and (c) degrades neither win rate nor max drawdown vs baseline (the
  original A7 "nothing degrades" bar). Cell precedence to prevent
  post-hoc shopping: if any cell clears, the originally-landed (5, 0.10)
  is preferred when it is among the clearing cells; otherwise the clearing
  cell closest to it (boundary first, then threshold). Point-estimate wins
  by non-clearing cells count for nothing.
- **R3 (outcome mapping)**: if no cell clears R2 → **A7 is RETIRED** (not
  merely withdrawn): `predict_fixture.py`'s A7 shadow output is disabled,
  CLAUDE.md / TODO.md are updated, and any future betting-layer rule must
  be derived fresh on v2-or-later prices under I3 blocks. If no threshold
  clears R1 → threshold 0 stays, and the M8 report's retraction banner
  gains a pointer here as the re-derivation of record.
- Blocks discipline: any slice with <10 blocks is descriptive only
  (invariant 7). Legacy-arm results are sensitivity context and cannot
  re-validate anything on their own.

## Results

**Input-integrity check: PASSED.** Baselines reproduce the restated
headlines exactly — i7 arm ≥$50k: n=167 bets, ROI **+3.38%**
[−14.63, +37.06], 18 blocks; ≥$100k: n=110, **−5.19%** [−28.73, +27.50],
11 blocks. Legacy arm: +7.40% [−7.65, +34.82] / −0.54% [−23.24, +24.03].
All artifacts in `eval_out/a7_m8_rederivation_v2/` (regenerable).

### M8 global edge-threshold sweep — primary (i7) arm, flat 1-unit

| thr | ≥$50k n | ≥$50k ROI [CI] | ≥$100k n | ≥$100k ROI [CI] |
|---|---|---|---|---|
| 0 (baseline) | 167 | **+3.38** [−14.63, +37.06] | 110 | **−5.19** [−28.73, +27.50] |
| 0.01 | 151 | +7.29 [−11.73, +41.10] | 100 | −2.97 [−29.34, +28.14] |
| 0.02 | 142 | +4.91 [−14.84, +40.71] | 99 | −4.42 [−30.10, +24.72] |
| 0.03 | 126 | +5.67 [−16.10, +43.92] | 89 | −2.26 [−27.58, +27.63] |
| 0.05 | 104 | +1.36 [−29.23, +52.42] | 75 | −3.69 [−37.66, +38.61] |
| 0.10 | 58 | −11.30 [−46.31, +32.91] | 49 | −21.39 [−79.16, +9.55] |

**R1 verdict: NOT cleared — flat 1-unit at threshold 0 stands.** No
threshold's ≥$50k CI excludes 0, and no threshold improves both slices by
>2.3pp (thr 0.01 gains +3.91pp at ≥$50k but its CI straddles 0). The shape
is the *inverse* of the retracted M8 finding: on honest prices ROI degrades
as the required edge rises (monotonically beyond 3%), because a large
"edge" against a sharp market is predominantly model error — the model
loses to the market on LL, so its biggest disagreements are its worst bets.
Win rate falls from 42.5% to 25.9% (≥$50k) across the sweep. Legacy arm
identical in shape (+7.40 → −10.68 at thr 0.10). Missing-feature fallback:
0 fixtures (every bet had a real `top6_batting_elo_diff`; n=167 vs the
168-fixture slice is one zero-edge fixture where the framework placed no
flat bet).

### A7 conditional grid — primary (i7) arm

ROI% [CI] vs baseline (+3.38 / −5.19). Cells are (mismatch edge thr) rows ×
boundary columns; close side always flat.

**≥$50k:**

| thr \ boundary | 3 | 5 | 8 | 12 |
|---|---|---|---|---|
| 0.05 | +4.71 [−19.5,+44.9] | +1.58 [−24.2,+41.7] | +2.20 [−21.3,+41.0] | +5.05 [−15.3,+39.2] |
| 0.10 | +7.37 [−12.2,+39.5] | **+0.19** [−23.5,+32.4] | +3.85 [−18.9,+38.1] | +4.25 [−17.5,+36.3] |
| 0.15 | +5.38 [−17.1,+37.9] | +4.46 [−26.4,+36.5] | +4.31 [−21.0,+38.3] | +5.85 [−16.5,+36.3] |

**≥$100k:**

| thr \ boundary | 3 | 5 | 8 | 12 |
|---|---|---|---|---|
| 0.05 | −1.54 | −3.32 | −4.33 | −2.57 |
| 0.10 | −10.27 | **−12.05** | −6.16 | −5.22 |
| 0.15 | −18.62 | −8.96 | −4.68 | −3.37 |

(≥$100k CIs all straddle 0 except none favorable; b3/0.15 lower bound
+0.54 is the only one that even approaches an exclusion — in the *wrong*
direction on ROI −18.62.)

**R2 verdict: NOT cleared — no cell satisfies any of (a)/(b)/(c).** No
≥$50k CI excludes 0 anywhere on the grid; no cell improves both slices by
>2.3pp; win rate degrades vs baseline in every cell. **The originally
landed cell (5, 0.10) — bolded — now *hurts* on both slices**: ≥$50k
+3.38 → +0.19 (−3.19pp), ≥$100k −5.19 → −12.05 (−6.86pp). On the legacy
arm the same cell reads +7.40 → +1.00 and −0.54 → −14.37, and the
adjacent (3, 0.15) cell at ≥$100k is CI-clean **negative** (−42.49
[−72.60, −22.99]): the mismatch-high-edge subset that the corrupt surface
scored as the honest bettable slice is, on real prices, the single most
toxic subset in the sweep. A7's retracted +15.03pp improvement was the
toss artifact end to end — the rule filtered *into* the corrupted
fixtures' fake edges.

### R3 outcome — executed

**A7 is RETIRED.** `predict_fixture.py`: `A7_POLICY_RETIRED = True` —
`shadow_bet_placed` forced False with suppression reason
`policy_retired_20260807` and `policy_status: "retired"` in the bet block;
edge/market diagnostics still emitted. `research/reports/auto/A7.md`
carries a retirement addendum; the M8 report's retraction banner points
here as the re-derivation of record. **Flat 1-unit at threshold 0 remains
the reporting default** (R1), which changes nothing operationally: no
production betting edge is established anywhere (every CI above straddles
zero), and the model-vs-market LL verdict (model loses on every slice) is
untouched by construction.

Any future betting-layer rule starts from scratch on v2-or-later prices
under I3 blocks, and must contend with the structural finding above: this
model's large disagreements with a liquid market are evidence against the
model, not against the market.
