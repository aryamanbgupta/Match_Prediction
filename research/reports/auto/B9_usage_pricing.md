# B9 — top_bowler margin vs a usage-share fair baseline

Detail: `detail_d15_s43_n261.json` (n=261 matches, 522 team-markets, 5835 rows). Headline shrinkage K_USAGE=5 appearances / K_RATE=120 balls (pre-committed); cluster bootstrap by match, 2000 resamples, seed 29.

## Baseline strength (standalone Brier; lower = better)

- sim (calibrated): 0.0785
- E2 career-wickets share: 0.0802
- usage-share baseline: 0.0747

## Stronger-bar check (paired dBrier usage − career; negative = usage stronger)

- headline: -0.0055 CI [-0.0070, -0.0040] → **usage IS the stronger bar** (CI-clean)

## The margin (paired dBrier sim − baseline; negative = sim adds skill)

- sim − career (B4 reproduction): -0.0017 CI [-0.0030, -0.0003]
- **sim − usage (headline)**: +0.0038 CI [+0.0026, +0.0051]
- sim − usage (ku2_kw60): +0.0038 CI [+0.0026, +0.0051]; usage − career: -0.0055 CI [-0.0070, -0.0039]
- sim − usage (ku10_kw240): +0.0038 CI [+0.0026, +0.0050]; usage − career: -0.0055 CI [-0.0069, -0.0039]

## Flat 1u YES ROI vs the usage-priced market (vig × threshold)

| vig | thr | bets | ROI % | ROI 95% CI | win % | avg odds | avg EV % |
|---|---|---|---|---|---|---|---|
| 0% | 0% | 2589 | -29.08 | [-38.45, -19.42] | 8.9 | 95.11 | +243.1 |
| 0% | 5% | 2461 | -31.50 | [-40.41, -22.52] | 8.7 | 99.37 | +255.6 |
| 0% | 10% | 2341 | -33.01 | [-42.25, -23.63] | 8.3 | 103.77 | +268.3 |
| 0% | 20% | 2120 | -36.44 | [-46.17, -26.25] | 7.6 | 112.57 | +294.7 |
| 2% | 0% | 2539 | -30.37 | [-39.54, -20.61] | 8.9 | 94.87 | +241.0 |
| 2% | 5% | 2410 | -32.76 | [-41.65, -23.86] | 8.7 | 99.18 | +253.8 |
| 2% | 10% | 2289 | -35.17 | [-44.12, -25.78] | 8.3 | 103.54 | +266.8 |
| 2% | 20% | 2070 | -39.74 | [-49.43, -29.53] | 7.2 | 112.84 | +293.5 |
| 5% | 0% | 2461 | -34.76 | [-43.25, -26.21] | 8.7 | 94.63 | +238.7 |
| 5% | 5% | 2339 | -36.14 | [-44.98, -27.20] | 8.3 | 98.90 | +251.0 |
| 5% | 10% | 2216 | -38.28 | [-47.35, -28.83] | 7.9 | 103.08 | +264.5 |
| 5% | 20% | 1998 | -42.94 | [-52.74, -32.79] | 7.0 | 113.11 | +291.7 |
| 10% | 0% | 2341 | -39.10 | [-47.50, -30.57] | 8.3 | 94.34 | +234.8 |
| 10% | 5% | 2216 | -41.08 | [-49.74, -32.06] | 7.9 | 98.40 | +247.9 |
| 10% | 10% | 2097 | -43.12 | [-52.02, -33.73] | 7.4 | 103.37 | +261.6 |
| 10% | 20% | 1875 | -45.13 | [-55.15, -34.53] | 6.9 | 114.26 | +290.8 |

- break-even vig (thr 0, 0.5% grid, in-sample): last positive —; first non-positive 0.0% (ROI -29.08%)

## Baseline-price bands (vig 5%, thr 0)

| p_usage band | bets | ROI % | ROI 95% CI | win % | total PnL u |
|---|---|---|---|---|---|
| >=20% | 119 | -5.67 | [-37.74, +29.57] | 21.0 | -6.7 |
| 10-20% | 896 | +12.63 | [-1.37, +26.20] | 17.9 | +113.2 |
| 5-10% | 379 | -27.81 | [-56.62, +2.71] | 5.5 | -105.4 |
| 2-5% | 288 | -26.93 | [-71.76, +30.92] | 2.8 | -77.6 |
| <2% | 779 | -100.00 | [-100.00, -100.00] | 0.0 | -779.0 |

## Zero-career-wicket players — how each baseline prices them

| group | rows | actual top rate | mean p_sim | mean p_career | mean p_usage |
|---|---|---|---|---|---|
| all zero-career-wkt | 1661 | 0.90% (15) | 1.30% | 0.26% | 1.58% |
| true debutant (0 appearances) | 59 | 8.47% (5) | 1.63% | 0.74% | 9.06% |
| seen, never took a wkt | 1602 | 0.62% (10) | 1.29% | 0.25% | 1.30% |

- YES bets the sim would place on them vs usage prices (vig 5%, thr 0): 576 bets, ROI -96.38% CI [-100.00, -88.86], total PnL -555.2u
