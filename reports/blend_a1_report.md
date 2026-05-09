# Phase A1 — Direct + Sim Blend Report

LL/ROI by blend weight `w` and slice. `logit(P_final) = w·logit(P_sim) + (1−w)·logit(P_direct)`. ROI CIs are 95% bootstrap (1000 resamples).

**Reference baselines** — coinflip LL 0.6931, market LL 0.6267, always-favorite flat ROI +4.15%.


## Slice: all (261)

| w | LL | LL 95% CI | Flat ROI | ROI 95% CI | Win rate | Bets |
|---|---|---|---|---|---|---|
| 0.00 (direct alone) | 0.6568 | [0.6362, 0.6783] | +9.70% | [-11.21%, +34.16%] | 43.5% | 255 |
| 0.20 | 0.6578 | [0.6346, 0.6826] | +7.20% | [-12.47%, +32.10%] | 43.5% | 255 |
| 0.35 | 0.6625 | [0.6350, 0.6924] | +3.96% | [-16.21%, +29.37%] | 43.1% | 255 |
| 0.50 | 0.6702 | [0.6369, 0.7068] | +7.05% | [-14.16%, +32.08%] | 45.5% | 255 |
| 0.65 | 0.6809 | [0.6413, 0.7243] | +10.35% | [-10.57%, +35.33%] | 47.5% | 255 |
| 0.80 | 0.6942 | [0.6479, 0.7445] | +9.44% | [-10.76%, +34.79%] | 47.5% | 255 |
| 1.00 (sim alone, v7) | 0.7158 | [0.6613, 0.7759] | +7.96% | [-10.57%, +29.10%] | 49.4% | 255 |

## Slice: ≥$50k (168)

| w | LL | LL 95% CI | Flat ROI | ROI 95% CI | Win rate | Bets |
|---|---|---|---|---|---|---|
| 0.00 (direct alone) | 0.6644 | [0.6393, 0.6889] | +15.87% | [-5.11%, +37.01%] | 47.6% | 168 |
| 0.20 | 0.6673 | [0.6356, 0.6982] | +14.76% | [-5.85%, +36.08%] | 48.2% | 168 |
| 0.35 | 0.6739 | [0.6363, 0.7118] | +9.42% | [-11.71%, +32.45%] | 47.0% | 168 |
| 0.50 | 0.6841 | [0.6385, 0.7300] | +10.31% | [-10.53%, +32.71%] | 48.2% | 168 |
| 0.65 | 0.6975 | [0.6429, 0.7519] | +13.32% | [-7.65%, +35.56%] | 50.0% | 168 |
| 0.80 | 0.7140 | [0.6487, 0.7773] | +12.23% | [-7.63%, +34.71%] | 50.0% | 168 |
| 1.00 (sim alone, v7) | 0.7402 | [0.6631, 0.8167] | +6.11% | [-10.72%, +23.87%] | 50.6% | 168 |

## Slice: ≥$100k (110)

| w | LL | LL 95% CI | Flat ROI | ROI 95% CI | Win rate | Bets |
|---|---|---|---|---|---|---|
| 0.00 (direct alone) | 0.6513 | [0.6175, 0.6810] | +12.91% | [-14.97%, +44.84%] | 45.5% | 110 |
| 0.20 | 0.6541 | [0.6129, 0.6927] | +13.94% | [-13.39%, +45.08%] | 47.3% | 110 |
| 0.35 | 0.6609 | [0.6136, 0.7088] | +7.47% | [-19.17%, +38.37%] | 45.5% | 110 |
| 0.50 | 0.6716 | [0.6148, 0.7302] | +8.90% | [-18.65%, +41.29%] | 47.3% | 110 |
| 0.65 | 0.6858 | [0.6186, 0.7588] | +10.30% | [-16.79%, +41.29%] | 48.2% | 110 |
| 0.80 | 0.7033 | [0.6250, 0.7891] | +9.40% | [-17.07%, +40.61%] | 48.2% | 110 |
| 1.00 (sim alone, v7) | 0.7311 | [0.6370, 0.8337] | -2.86% | [-23.03%, +19.29%] | 47.3% | 110 |

## Curve characterization (≥$50k slice)

- LL-vs-w shape: **monotone increasing in w**, best LL at w = 0.0
- Interpretation: **direct alone wins** — adding any sim weight worsens LL. Per the plan's decision tree: this means direct >> sim and sim is contributing only directional noise on top of direct's signal. Move to Phase A2 (richer features) — the cheap-subset is sufficient to dominate sim, but probably not enough to close the residual gap to market.

## Go/no-go gate check (≥$50k slice)

Required: model LL < market LL (0.6267) AND flat-ROI CI excludes zero.
- LL < market: clears at w = none
- ROI CI excludes 0: clears at w = none
- BOTH conditions: w = none

## Per-match decomposition (all slice)

- Compared on n = 261 matches
- Best blend w (lowest aggregate LL) = 0.0
- Blend beats both components per-match: 0 / 261
- Direct alone beats sim alone per-match: 119 / 261
- Best-blend flips bet side vs sim alone: 95
- Best-blend matches with edge > 3%: 207
