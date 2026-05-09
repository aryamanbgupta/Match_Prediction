# Phase A1 — Direct + Sim Blend Report

LL/ROI by blend weight `w` and slice. `logit(P_final) = w·logit(P_sim) + (1−w)·logit(P_direct)`. ROI CIs are 95% bootstrap (1000 resamples).

**Reference baselines** — coinflip LL 0.6931, market LL 0.6267, always-favorite flat ROI +4.15%.


## Slice: all (261)

| w | LL | LL 95% CI | Flat ROI | ROI 95% CI | Win rate | Bets |
|---|---|---|---|---|---|---|
| 0.00 (direct alone) | 0.4944 | [0.4541, 0.5344] | +50.73% | [+32.40%, +74.43%] | 69.4% | 255 |
| 0.20 | 0.5140 | [0.4773, 0.5512] | +43.28% | [+24.70%, +66.33%] | 65.9% | 255 |
| 0.35 | 0.5363 | [0.5016, 0.5727] | +38.21% | [+19.12%, +61.42%] | 62.4% | 255 |
| 0.50 | 0.5657 | [0.5301, 0.6049] | +30.33% | [+11.18%, +53.60%] | 58.0% | 255 |
| 0.65 | 0.6024 | [0.5645, 0.6451] | +26.01% | [+6.33%, +48.95%] | 55.3% | 255 |
| 0.80 | 0.6464 | [0.6029, 0.6944] | +20.88% | [+1.07%, +44.52%] | 53.3% | 255 |
| 1.00 (sim alone, v7) | 0.7158 | [0.6613, 0.7759] | +7.96% | [-10.57%, +29.10%] | 49.4% | 255 |

## Slice: ≥$50k (168)

| w | LL | LL 95% CI | Flat ROI | ROI 95% CI | Win rate | Bets |
|---|---|---|---|---|---|---|
| 0.00 (direct alone) | 0.5004 | [0.4455, 0.5552] | +53.67% | [+36.01%, +73.80%] | 71.4% | 168 |
| 0.20 | 0.5204 | [0.4715, 0.5696] | +45.67% | [+27.62%, +66.38%] | 67.9% | 168 |
| 0.35 | 0.5439 | [0.4964, 0.5903] | +37.71% | [+18.42%, +57.86%] | 62.5% | 168 |
| 0.50 | 0.5755 | [0.5284, 0.6244] | +33.23% | [+13.37%, +53.75%] | 60.1% | 168 |
| 0.65 | 0.6155 | [0.5631, 0.6690] | +28.59% | [+7.99%, +51.44%] | 57.1% | 168 |
| 0.80 | 0.6636 | [0.6043, 0.7238] | +22.97% | [+3.02%, +43.84%] | 55.4% | 168 |
| 1.00 (sim alone, v7) | 0.7402 | [0.6631, 0.8167] | +6.11% | [-10.72%, +23.87%] | 50.6% | 168 |

## Slice: ≥$100k (110)

| w | LL | LL 95% CI | Flat ROI | ROI 95% CI | Win rate | Bets |
|---|---|---|---|---|---|---|
| 0.00 (direct alone) | 0.4361 | [0.3729, 0.5000] | +58.03% | [+33.35%, +86.63%] | 73.6% | 110 |
| 0.20 | 0.4630 | [0.4087, 0.5216] | +52.99% | [+28.49%, +82.78%] | 71.8% | 110 |
| 0.35 | 0.4929 | [0.4409, 0.5479] | +43.83% | [+18.97%, +73.83%] | 65.5% | 110 |
| 0.50 | 0.5320 | [0.4788, 0.5918] | +45.15% | [+19.48%, +74.61%] | 65.5% | 110 |
| 0.65 | 0.5807 | [0.5185, 0.6479] | +32.76% | [+6.24%, +63.30%] | 59.1% | 110 |
| 0.80 | 0.6390 | [0.5635, 0.7207] | +26.95% | [+1.77%, +59.17%] | 57.3% | 110 |
| 1.00 (sim alone, v7) | 0.7311 | [0.6370, 0.8337] | -2.86% | [-23.03%, +19.29%] | 47.3% | 110 |

## Curve characterization (≥$50k slice)

- LL-vs-w shape: **monotone increasing in w**, best LL at w = 0.0
- Interpretation: **direct alone wins** — adding any sim weight worsens LL. Per the plan's decision tree: this means direct >> sim and sim is contributing only directional noise on top of direct's signal. Move to Phase A2 (richer features) — the cheap-subset is sufficient to dominate sim, but probably not enough to close the residual gap to market.

## Go/no-go gate check (≥$50k slice)

Required: model LL < market LL (0.6267) AND flat-ROI CI excludes zero.
- LL < market: clears at w = [0.0, 0.5, 0.2, 0.65, 0.35]
- ROI CI excludes 0: clears at w = [0.0, 0.5, 0.8, 0.2, 0.65, 0.35]
- BOTH conditions: w = [0.0, 0.2, 0.35, 0.5, 0.65]

## Per-match decomposition (all slice)

- Compared on n = 261 matches
- Best blend w (lowest aggregate LL) = 0.0
- Blend beats both components per-match: 0 / 261
- Direct alone beats sim alone per-match: 174 / 261
- Best-blend flips bet side vs sim alone: 97
- Best-blend matches with edge > 3%: 230
