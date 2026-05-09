# Phase A1 — Direct + Sim Blend Report

LL/ROI by blend weight `w` and slice. `logit(P_final) = w·logit(P_sim) + (1−w)·logit(P_direct)`. ROI CIs are 95% bootstrap (1000 resamples).

**Reference baselines** — coinflip LL 0.6931, market LL 0.6267, always-favorite flat ROI +4.15%.


## Slice: all (261)

| w | LL | LL 95% CI | Flat ROI | ROI 95% CI | Win rate | Bets |
|---|---|---|---|---|---|---|
| 0.00 (direct alone) | 0.5226 | [0.4771, 0.5667] | +43.12% | [+23.93%, +68.98%] | 65.9% | 255 |
| 0.20 | 0.5362 | [0.4949, 0.5766] | +39.36% | [+20.95%, +66.24%] | 63.1% | 255 |
| 0.35 | 0.5542 | [0.5170, 0.5924] | +33.88% | [+14.80%, +58.60%] | 60.0% | 255 |
| 0.50 | 0.5794 | [0.5416, 0.6195] | +23.87% | [+4.06%, +48.28%] | 54.9% | 255 |
| 0.65 | 0.6119 | [0.5731, 0.6538] | +21.78% | [+1.95%, +46.08%] | 53.7% | 255 |
| 0.80 | 0.6518 | [0.6092, 0.7000] | +16.91% | [-3.17%, +41.98%] | 51.8% | 255 |
| 1.00 (sim alone, v7) | 0.7158 | [0.6613, 0.7759] | +7.96% | [-10.57%, +29.10%] | 49.4% | 255 |

## Slice: ≥$50k (168)

| w | LL | LL 95% CI | Flat ROI | ROI 95% CI | Win rate | Bets |
|---|---|---|---|---|---|---|
| 0.00 (direct alone) | 0.5135 | [0.4585, 0.5697] | +47.35% | [+29.14%, +68.08%] | 69.6% | 168 |
| 0.20 | 0.5295 | [0.4820, 0.5781] | +43.97% | [+24.93%, +65.07%] | 66.7% | 168 |
| 0.35 | 0.5507 | [0.5051, 0.5958] | +36.61% | [+17.62%, +57.60%] | 62.5% | 168 |
| 0.50 | 0.5803 | [0.5328, 0.6271] | +27.65% | [+9.00%, +50.20%] | 57.7% | 168 |
| 0.65 | 0.6185 | [0.5651, 0.6714] | +26.43% | [+6.57%, +48.69%] | 57.1% | 168 |
| 0.80 | 0.6652 | [0.6037, 0.7261] | +19.66% | [-0.24%, +41.86%] | 54.2% | 168 |
| 1.00 (sim alone, v7) | 0.7402 | [0.6631, 0.8167] | +6.11% | [-10.72%, +23.87%] | 50.6% | 168 |

## Slice: ≥$100k (110)

| w | LL | LL 95% CI | Flat ROI | ROI 95% CI | Win rate | Bets |
|---|---|---|---|---|---|---|
| 0.00 (direct alone) | 0.4554 | [0.3851, 0.5295] | +51.04% | [+27.09%, +80.16%] | 71.8% | 110 |
| 0.20 | 0.4774 | [0.4147, 0.5388] | +46.37% | [+21.92%, +76.72%] | 67.3% | 110 |
| 0.35 | 0.5040 | [0.4456, 0.5590] | +38.36% | [+12.86%, +68.44%] | 62.7% | 110 |
| 0.50 | 0.5401 | [0.4825, 0.6006] | +30.02% | [+4.81%, +59.82%] | 58.2% | 110 |
| 0.65 | 0.5861 | [0.5227, 0.6546] | +27.28% | [+0.80%, +57.67%] | 57.3% | 110 |
| 0.80 | 0.6420 | [0.5663, 0.7241] | +21.26% | [-4.57%, +52.23%] | 54.5% | 110 |
| 1.00 (sim alone, v7) | 0.7311 | [0.6370, 0.8337] | -2.86% | [-23.03%, +19.29%] | 47.3% | 110 |

## Curve characterization (≥$50k slice)

- LL-vs-w shape: **monotone increasing in w**, best LL at w = 0.0
- Interpretation: **direct alone wins** — adding any sim weight worsens LL. Per the plan's decision tree: this means direct >> sim and sim is contributing only directional noise on top of direct's signal. Move to Phase A2 (richer features) — the cheap-subset is sufficient to dominate sim, but probably not enough to close the residual gap to market.

## Go/no-go gate check (≥$50k slice)

Required: model LL < market LL (0.6267) AND flat-ROI CI excludes zero.
- LL < market: clears at w = [0.0, 0.5, 0.2, 0.65, 0.35]
- ROI CI excludes 0: clears at w = [0.0, 0.5, 0.2, 0.65, 0.35]
- BOTH conditions: w = [0.0, 0.2, 0.35, 0.5, 0.65]

## Per-match decomposition (all slice)

- Compared on n = 261 matches
- Best blend w (lowest aggregate LL) = 0.0
- Blend beats both components per-match: 0 / 261
- Direct alone beats sim alone per-match: 163 / 261
- Best-blend flips bet side vs sim alone: 104
- Best-blend matches with edge > 3%: 243
