# M8 — sizing rules (2026-05-10)

> # ⚠️ RETRACTION — 2026-08-05: this sizing rule was selected on a corrupt ROI surface
>
> **Every ROI number and CI in this report was computed against
> `betting_odds_polymarket.json`, which scored the event's *"Who wins the
> toss?"* market as the winner market on 23 of 261 fixtures** (the builder
> ranked candidates by a `max price ≤ 0.92 = "plausible"` flag first, which
> rejects lopsided — i.e. informative — winner markets and keeps the ~0.50
> coin flip). See `reports/market_benchmark_toss_defect_20260805.md`.
>
> **What this retracts.** The landing argument here is verbatim *"Threshold 0
> is the only config where the ROI CI cleanly excludes 0"*. That statement is
> a property of the corrupt ROI surface — 17 lopsided fixtures were recorded
> at ~0.50 when their true prices were 0.72–0.99, which manufactures large
> fake edges and pays them at ~2.0 decimal odds instead of ~1.05. The whole
> threshold sweep, both slices, and the mismatch-slice table (which A7 was
> then built on) are therefore **not evidence for anything** and must be
> **re-derived on `betting_odds_polymarket_v2.json`** (255 fixtures,
> `--cluster-source-dir data/polymarket_test_v2`) before being quoted again.
> Note also that these CIs are per-match i.i.d. bootstraps, which invariant 7
> superseded in I3 — a second, independent reason not to quote them.
>
> **What is NOT retracted.** *Flat 1-unit staking* is a conservative default
> that carries no ROI-surface dependence and is not itself at risk; keep it.
> `scripts/sim_eval/sizing_rules.py` is fine as tooling. What is gone is the
> *empirical justification* for the specific edge threshold and for the
> "low-edge bets are predictive" claim below.
>
> **Downstream:** `research/reports/auto/A7.md` (the slice-conditional rule
> derived from the mismatch table) is retracted with it, and A7 is withdrawn
> from forward use. Corrected production ROI on the ≥$50k slice is **+3.38%**
> (i7 arm, block CI [−14.63%, +37.06%]) / **+7.40%** (legacy arm), not the
> +20–25% these tables assume as the baseline.
>
> **RE-DERIVED 2026-08-07 — `reports/a7_m8_rederivation_v2_20260807.md` is
> the re-derivation of record.** Under a pre-registered protocol on v2
> prices with I3 blocks: no global edge threshold clears the bar (ROI
> *degrades* as the threshold rises on honest prices — the inverse of this
> report's retracted finding), so **flat 1-unit at threshold 0 stands as
> the default**; and no A7 grid cell clears, so **A7 is retired** (the
> originally-landed cell hurts on both slices). No betting edge of any kind
> is established — every interval straddles zero.

Phase 8 of match-level v3. Final phase. **Outcome: flat 1-unit betting at edge threshold 0 is the production sizing rule.** Higher edge thresholds and Kelly variants explored; none improve aggregate ROI/CI on iteration evidence. A slice-conditional threshold finding is documented for future enhancement but not landed (complexity + small-n on mismatch slice).

**Production sizing rule (LANDED)**:
- **Edge threshold**: 0 (bet on every match where model's preferred team has positive edge vs market)
- **Stake**: flat 1 unit per bet
- **Kelly variants**: tooling preserved (`scripts/sim_eval/sizing_rules.py`) for future use when bank-sizing matters more than per-bet ROI clarity.

## Hypothesis tested

E1 in the catalog said: "Bet only when calibrated edge > 3%, quarter-Kelly stake." We tested whether either (a) a positive edge threshold or (b) Kelly sizing with a per-bet outlier cap improves iteration metrics on the M7 production model.

Tool built: `scripts/sim_eval/sizing_rules.py` — post-hoc layer on existing eval JSONs. Loads model edges + market odds, applies an edge filter + Kelly-cap, bootstraps ROI CI.

## Sweep results — iteration ≥$50k (n=170 eligible)

### Flat 1-unit, edge threshold sweep

| Threshold | n_bets | ROI | ROI 95% CI | Win % | Max DD |
|---|---|---|---|---|---|
| **0** (production) | **170** | **+20.47%** | **[+0.91, +41.77]** | 51.2 | 12.52 |
| 1% | 154 | +17.52% | [-3.99, +40.65] | 48.7 | 15.34 |
| 2% | 148 | +16.27% | [-4.66, +42.96] | 48.6 | 16.65 |
| 3% | 129 | +14.18% | [-9.85, +42.00] | 47.3 | 20.04 |
| 5% | 107 | +19.72% | [-6.66, +54.60] | 48.6 | 18.89 |
| 7% | 89 | +30.67% | [-1.33, +66.16] | 52.8 | 10.01 |
| 10% | 59 | +38.09% | [-5.35, +91.50] | 50.8 | 5.04 |
| 15% | 30 | +27.42% | [-21.01, +73.17] | 53.3 | 3.00 |

**Threshold 0 is the only config where the ROI CI cleanly excludes 0**. Higher thresholds increase point ROI (cherry-picking high-edge bets) but widen CIs faster (due to dropping bet count) — and at 1-3% thresholds, ROI actually drops AND CIs go negative. Maximum drawdown is also WORST at 3% threshold (20.04 units).

### Counter-intuitive finding

The model's low-edge bets (1-3% edge) ARE predictive. Filtering them out hurts on every metric simultaneously. The "edge > threshold" rule from betting folklore doesn't apply to this model — small disagreements with market carry signal because the M7 production model is calibrated well at the low-edge end (post-hyperparameter-sweep with lr=0.05).

The model's low-edge bets are NOT the model "guessing where it doesn't know". They're calibrated small-disagreement bets that aggregate to positive expectation.

### Iteration ≥$100k same pattern

| Threshold | n_bets | ROI | ROI 95% CI |
|---|---|---|---|
| **0** | **110** | **+26.39%** | **[+0.57, +58.78]** |
| 1% | 101 | +25.30% | [-2.90, +58.83] |
| 5% | 74 | +31.05% | [-5.74, +72.46] |
| 10% | 46 | +38.66% | [-11.69, +105.78] |

Threshold 0 again the cleanest CI clearance.

### Slice-conditional finding (NOT landed, documented for future)

The pattern reverses on the **mismatch slice** (|top6 ELO diff| ≥ 15, n=24):

| Threshold | n_bets | ROI | ROI 95% CI | Win % |
|---|---|---|---|---|
| 0 | 24 | +8.05% | [-33.32, +49.64] | 54.2 |
| 5% | 19 | +36.48% | [-5.03, +78.06] | 68.4 |
| **10%** | **18** | **+44.06%** | **[+1.15, +78.30]** | **72.2** |
| 20% | 9 | +77.64% | [+32.01, +102.19] | 88.9 |

On lopsided matches: model is over-confident at small edges; filtering to threshold ≥ 5% materially helps. At threshold=10%, ROI +44% with CI excluding 0 and win rate 72%.

On the **close slice** (|top6 ELO diff| ≤ 5, n=76), threshold 0 is best:

| Threshold | n_bets | ROI | ROI 95% CI |
|---|---|---|---|
| **0** | **76** | **+29.76%** | **[+3.99, +58.36]** |
| 1% | 67 | +23.03% | [-8.36, +52.81] |
| 5% | 47 | +21.26% | [-16.26, +58.40] |

**Why the slice-dependent behavior**: in close matches, both model and market are essentially guessing, and the model's marginal signal vs market carries information at every edge level. In mismatch matches, the market is well-calibrated and only large model disagreements (rare 10%+ edges) carry edge.

**Why NOT landing slice-conditional threshold**:
- Mismatch slice n=18-24 is small. ROI CIs are wide.
- Adds complexity to the production prediction path (need to compute slice membership at inference time and apply different thresholds).
- The simple flat-at-threshold-0 already clears the gate cleanly on aggregate.
- Documented in `TODO.md` as a potential post-deployment refinement once forward-capture provides more mismatch samples.

## Kelly variants explored

Quarter Kelly with 2% per-bet cap (the M8 catalog default):
- ROI per bet: +0.10% (vs flat +20.47%)
- This is the per-fractional-bet ROI; cumulative bank return = ~0.17 of 1.0 unit bank
- Max DD: 0.06 of bank

Full Kelly with 2% cap: per-bet ROI +0.38%, max DD 0.25
Full Kelly with no cap (cap=1.0): per-bet ROI +4.06% [+0.59, +7.99], max DD 2.52 of bank

The capped Kelly produces meaningfully smaller cumulative returns than uncapped, AND the per-bet ROI metric is not directly comparable to flat ROI (different stake sizes). **Practical recommendation**:
- For a 1-unit flat bank exercise: use flat. Easier to interpret; clear CI; +20.47% ROI [+0.91, +41.77].
- For a real-money Kelly approach: quarter Kelly with 2% per-bet cap is the conservative default. Tooling supports both via `--sizing` flag.
- Kelly multiplier optimization is data-poor on 170 iteration bets; defer until forward-capture provides 500+ bets.

## Status of M8 verification criteria

1. ✅ **Edge threshold + fractional Kelly tested (E1)**: data says no positive threshold improves aggregate. Production uses threshold=0.
2. ✅ **Outlier per-bet cap explored (E2)**: 2% cap with quarter Kelly preserved as tooling default. No data-driven optimization (small n).
3. ⏸ **Final golden re-evaluation deferred**: per the iteration-only-decisions discipline, golden is held for production-launch confirmation, NOT for sizing-rule selection. The single golden re-eval will happen at deployment time only — after M8 lands and we're committing the model.
4. ✅ **CLAUDE.md honest headline block updated**: already done at M7 with the iteration numbers (golden numbers not used).

## Production sizing rule (final)

```
edge_threshold = 0.0       # bet on every match with positive model edge
stake = 1                  # flat 1 unit
```

For Kelly users:
```
edge_threshold = 0.0
kelly_multiplier = 0.25    # quarter Kelly
per_bet_cap = 0.02         # max 2% of bank on any single bet
```

The tooling at `scripts/sim_eval/sizing_rules.py` supports both. Sweep CSVs preserved at:
- `models/xgb_match_v3_m7_production/m8_sizing_sweep_50k.csv`
- `models/xgb_match_v3_m7_production/m8_sizing_sweep_100k.csv`

## What the v3 pipeline produced — final state

After 8 M-phases:

| Phase | Outcome | Headline |
|---|---|---|
| M1 | Landed | eval infra + monotone + Platt sizing layer |
| M2 | Landed (venue subset) | venue outcome-dist features |
| M3 | Drop + unfrozen materialization adopted | rolling form failed; mode change won |
| M4 | Drop | within-tournament form / scheduling failed |
| M5 | Drop at corr check | player×opp affinity redundant |
| M6 | Drop + target-floor discipline added | conditions too low-signal |
| M7 | Landed | hyperparameter sweep (lr 0.10→0.05, cs 0.8→0.9) |
| M8 | Landed (flat at threshold 0) | sizing rules: simpler is better |

**Production model**: `models/xgb_match_v3_m7_production/`
**Production sizing**: flat 1 unit, edge threshold 0
**Production fixture predictor**: `predict_fixture.py` (uses raw probabilities; no Platt)

**Iteration headline** (the gate we tuned to):
- ≥$50k: LL 0.6299 vs market 0.6267, ROI +20.47% [+0.91, +41.77]
- ≥$100k: LL 0.5929, ROI +26.39% [+0.57, +58.78]
- Close-slice: LL 0.6880, ROI +33.27% [+4.36, +61.53]
- 2026-04 IPL: ROI +34.87% [+2.04, +68.06], win 65.7%

**Next**: forward polymarket capture continues (C2). Once 30-60 fresh matches accumulate, run the production model + sizing rule on them to confirm out-of-sample performance. Then golden re-eval to lock in the production headline.

## Artifacts

- `scripts/sim_eval/sizing_rules.py` — new harness
- `models/xgb_match_v3_m7_production/m8_sizing_sweep_{50k,100k}.csv` — full sweep data
- `reports/m8_sizing_rules_eval.md` (this file)

## Headline (one-line)

Edge thresholds don't help on this model; the M7 production probability output has enough resolution that even 1% edges carry signal. Production: flat 1-unit at edge threshold 0. Slice-conditional threshold (10% on mismatch fixtures) deferred until forward-capture provides more samples.
