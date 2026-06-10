# MLC 2026 — weighted factor report (best XIs)

*Exact XGBoost TreeSHAP attribution of `xgb_match_v3_m7_production`. Each factor's push is in log-odds and the pushes + base rate sum to the model's margin. "Favors" = which side the factor pushes toward; "weight" = share of total absolute movement.*


## LAKR v SFU — LAKR 42.2% / SFU 57.8%  (Grand Prairie Stadium, Dallas, 2026-06-19)


| Factor | LAKR vs SFU difference | push (logit) | favors | weight |
|---|---|---|---|---|
| Recent form (last 10) | 0.20 vs 0.60 | -0.251 | SFU | 35.2% |
| Home advantage | home: T1 1, T2 1 | -0.078 | SFU | 10.9% |
| Career batting (avg/SR) | avg 21.1 vs 17.1; SR 130 vs 120 | +0.071 | LAKR |  9.9% |
| Head-to-head | rate 0.33, n=4 | -0.058 | SFU |  8.1% |
| Whole-XI bowling (ELO) | 16530 vs 16506 (+23) | +0.056 | LAKR |  7.9% |
| Toss / bat-first | pre-toss default (shared) | +0.049 | LAKR |  6.8% |
| Whole-XI batting (ELO) | 16625 vs 16613 (+11) | -0.040 | SFU |  5.6% |
| Top-6 batting (ELO) | 1517 vs 1520 (-3) | -0.034 | SFU |  4.8% |
| Venue profile | avg 154, chase 0.58 (shared) | -0.023 | SFU |  3.2% |
| Bottom-5 bowling (ELO) | 1504 vs 1504 (+0) | -0.019 | SFU |  2.6% |
| Career bowling (avg/econ) | econ 8.30 vs 8.95 | +0.018 | LAKR |  2.5% |
| Competition / international | MLC / club (shared) | +0.011 | LAKR |  1.6% |
| Lineup matchup (hand/pace/spin) | LHB 3/1, pace 6/6, spin 4/2 | +0.007 | LAKR |  0.9% |


## TSK v SFU — TSK 55.4% / SFU 44.6%  (Grand Prairie Stadium, Dallas, 2026-06-20)


| Factor | TSK vs SFU difference | push (logit) | favors | weight |
|---|---|---|---|---|
| Top-6 batting (ELO) | 1526 vs 1520 (+6) | +0.171 | TSK | 28.2% |
| Whole-XI bowling (ELO) | 16574 vs 16506 (+67) | +0.087 | TSK | 14.3% |
| Bottom-5 bowling (ELO) | 1510 vs 1504 (+6) | +0.084 | TSK | 13.9% |
| Home advantage | home: T1 1, T2 1 | -0.068 | SFU | 11.2% |
| Career bowling (avg/econ) | econ 7.46 vs 8.95 | +0.046 | TSK |  7.5% |
| Career batting (avg/SR) | avg 24.2 vs 17.1; SR 133 vs 120 | +0.044 | TSK |  7.3% |
| Toss / bat-first | pre-toss default (shared) | +0.044 | TSK |  7.3% |
| Whole-XI batting (ELO) | 16658 vs 16613 (+44) | -0.021 | SFU |  3.5% |
| Venue profile | avg 154, chase 0.58 (shared) | -0.015 | SFU |  2.5% |
| Competition / international | MLC / club (shared) | +0.008 | TSK |  1.3% |
| Head-to-head | rate 0.43, n=5 | -0.008 | SFU |  1.3% |
| Lineup matchup (hand/pace/spin) | LHB 3/1, pace 4/6, spin 7/2 | +0.006 | TSK |  1.0% |
| Recent form (last 10) | 0.60 vs 0.60 | -0.005 | SFU |  0.8% |

