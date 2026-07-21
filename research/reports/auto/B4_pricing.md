# B4 — top_bowler pricing margin vs the E2 fair baseline

Primary detail: `detail_d15_s43_n261.json` (n=261 matches, 522 team-markets, 5835 player rows; 6 markets with no y=1 — bowlers took zero wickets, every YES bet loses).

Synthetic market = E2 as-of fair baseline (career-wickets share within XI), multiplicative overround q = p_base*(1+vig), YES-side flat 1u, settle on cricsheet y. CIs: cluster bootstrap by match (2000 resamples, seed 29).

## Margin re-verification (paired dBrier sim - base; negative = sim beats baseline)

- **primary** `detail_d15_s43_n261.json`: -0.0017 CI [-0.0030, -0.0003] (n=5835)
- **context** `detail_d1_s43_n261.json`: -0.0023 CI [-0.0035, -0.0010] (n=5835)

## Edge distribution (p_sim - p_base, per player row)

- quantiles (pp): P5 -13.6, P25 -1.1, P50 +0.2, P75 +3.1, P95 +10.2
- mean |edge| 4.39pp; share |edge|>2pp 52.4%; >5pp 29.9%

## Flat 1u YES ROI vs vig x edge-threshold

| vig | thr | bets | ROI % | ROI 95% CI | win % | avg odds | avg EV % | Kelly-staked ROI % | mean/med/p90 Kelly |
|---|---|---|---|---|---|---|---|---|---|
| 0% | 0% | 3015 | +154.75 | [+83.50, +234.85] | 8.4 | 197.39 | +451.1 | +77.53 | 0.047/0.031/0.113 |
| 0% | 5% | 2915 | +156.40 | [+83.43, +238.16] | 8.1 | 203.66 | +466.5 | +77.77 | 0.048/0.034/0.114 |
| 0% | 10% | 2802 | +161.44 | [+85.61, +246.80] | 8.0 | 211.18 | +485.0 | +78.71 | 0.050/0.036/0.116 |
| 0% | 20% | 2601 | +165.66 | [+82.48, +257.81] | 7.5 | 225.91 | +521.3 | +80.96 | 0.052/0.038/0.119 |
| 2% | 0% | 2981 | +148.58 | [+77.96, +227.71] | 8.3 | 195.56 | +445.3 | +75.52 | 0.046/0.030/0.111 |
| 2% | 5% | 2864 | +153.81 | [+80.84, +235.73] | 8.0 | 202.91 | +463.4 | +75.86 | 0.048/0.033/0.113 |
| 2% | 10% | 2763 | +157.82 | [+82.42, +242.55] | 7.8 | 209.77 | +480.1 | +76.52 | 0.049/0.035/0.114 |
| 2% | 20% | 2552 | +163.92 | [+81.15, +255.79] | 7.4 | 225.40 | +518.6 | +79.29 | 0.051/0.037/0.116 |
| 5% | 0% | 2914 | +144.27 | [+74.71, +222.17] | 8.1 | 194.02 | +439.7 | +72.75 | 0.045/0.029/0.108 |
| 5% | 5% | 2798 | +149.34 | [+77.06, +230.65] | 8.0 | 201.40 | +457.8 | +73.18 | 0.047/0.032/0.110 |
| 5% | 10% | 2672 | +150.10 | [+73.76, +235.48] | 7.6 | 210.05 | +479.0 | +73.76 | 0.048/0.034/0.112 |
| 5% | 20% | 2471 | +162.49 | [+78.98, +255.25] | 7.4 | 225.37 | +516.7 | +77.33 | 0.050/0.036/0.115 |
| 10% | 0% | 2802 | +137.67 | [+68.74, +215.27] | 8.0 | 191.98 | +431.8 | +68.48 | 0.043/0.029/0.104 |
| 10% | 5% | 2674 | +138.56 | [+65.86, +220.00] | 7.6 | 200.37 | +452.3 | +68.74 | 0.045/0.030/0.105 |
| 10% | 10% | 2585 | +142.62 | [+66.68, +226.79] | 7.5 | 206.52 | +467.7 | +69.53 | 0.046/0.032/0.107 |
| 10% | 20% | 2343 | +152.30 | [+69.64, +245.06] | 7.1 | 225.80 | +514.4 | +73.52 | 0.049/0.035/0.112 |

## Break-even vig (thr=0, flat 1u, 0.5% grid; in-sample)

- last positive-ROI vig: 50.0% (ROI +93.48%); first non-positive: none <= 50%

## Where the units come from — baseline-price bands (vig 5%, thr 0)

| p_base band | bets | ROI % | ROI 95% CI | win % | total PnL u |
|---|---|---|---|---|---|
| >=20% | 142 | -20.79 | [-47.34, +9.03] | 19.0 | -29.5 |
| 10-20% | 568 | +0.82 | [-18.02, +19.54] | 14.8 | +4.6 |
| 5-10% | 559 | +54.14 | [+23.43, +84.58] | 12.2 | +302.7 |
| 2-5% | 405 | +108.30 | [+30.99, +195.51] | 7.2 | +438.6 |
| <2% | 1240 | +281.26 | [+123.89, +459.26] | 2.3 | +3487.7 |

## D5 diagnostic — zero-career-wicket players (as-of)

- rows 1661, actual top-bowler outcomes 15 (base rate 0.90%); mean p_sim 1.30% vs mean p_base 0.26%
- rows with p_sim >= 2%: 477 (mean p_sim 3.84%)
- YES bets the sim would place on them (vig 5%, thr 0): 795 bets, ROI +309.22% CI [+70.55, +602.54], total PnL +2458.3u
