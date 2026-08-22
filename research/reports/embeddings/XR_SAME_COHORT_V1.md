# Same-cohort staged xR/xW v1

All arms use identical rows, input width, residual-head capacity, training recipe, and seeds. Delivery, shot, and contact are realized post-ball information, not pre-ball forecasts.

> **Estimator note (2026-08-21).** The "seed+match" CIs in this report are
> the **seed-draw** estimator: each bootstrap replicate draws one fitted seed,
> so the intervals include single-seed variance and are systematically wider
> than the T1 ablation report's seed-mean estimator. Do not compare CI widths
> across the two reports.

## Coverage

| split | selected rows | all rows | coverage | matches | dates |
|---|---:|---:|---:|---:|---|
| train | 279,779 | 1,876,971 | 14.9% | 1,250 | 2021-01-01 to 2024-12-30 |
| validation | 49,709 | 124,292 | 40.0% | 221 | 2024-12-31 to 2025-06-29 |
| test | 72,801 | 186,912 | 38.9% | 328 | 2025-07-01 to 2026-04-16 |

Train-only run mapping for classes [0, 1, 2, 4, 6, wicket]: 0.0000, 1.0000, 2.0611, 4.0108, 6.0074, 0.0275.

## Validation

| arm | seed mean LL ± SD | ensemble xR MAE | xR bias | xW Brier |
|---|---:|---:|---:|---:|
| context | 1.454923 ± 0.000812 | 1.267277 | -0.024214 | 0.049784 |
| delivery | 1.427739 ± 0.001163 | 1.241114 | -0.015970 | 0.049652 |
| shot | 1.279329 ± 0.000663 | 1.171249 | +0.023885 | 0.048262 |
| contact | 1.143395 ± 0.000333 | 1.020972 | +0.026461 | 0.041453 |

### Calibration

| arm | xR intercept | xR slope | xW intercept | xW slope | xW log loss |
|---|---:|---:|---:|---:|---:|
| context | -0.023151 | 1.033644 | +0.003111 | 0.938750 | 0.204036 |
| delivery | +0.009656 | 1.004459 | +0.005141 | 0.874514 | 0.202982 |
| shot | +0.009053 | 0.977377 | -0.002526 | 1.000469 | 0.192633 |
| contact | -0.017826 | 0.994080 | +0.000342 | 1.013953 | 0.140249 |

| comparison | ΔLL | seed+match 95% CI | better seeds | ΔxR MAE |
|---|---:|---:|---:|---:|
| delivery_vs_context | -0.027184 | [-0.030517, -0.024023] | 5/5 | -0.025894 |
| shot_vs_delivery | -0.148411 | [-0.154258, -0.142720] | 5/5 | -0.069597 |
| contact_vs_shot | -0.135933 | [-0.140786, -0.131123] | 5/5 | -0.150397 |

### Process metrics (runs per delivery)

| metric | mean | match-bootstrap 95% CI | SD |
|---|---:|---:|---:|
| delivery_value | +0.008244 | [+0.001930, +0.014394] | 0.314729 |
| shot_selection_value | +0.039854 | [+0.029705, +0.049901] | 0.555018 |
| contact_value | +0.002576 | [-0.006810, +0.011638] | 0.731609 |
| execution_residual | -0.023885 | [-0.039349, -0.009520] | 1.580061 |

The mean staged increment can be near zero even when the information is valuable: positive and negative delivery/shot/contact effects cancel across the cohort. Use paired LL/MAE for information value and row/player aggregates for process value; do not interpret the cohort mean alone.

## Test

| arm | seed mean LL ± SD | ensemble xR MAE | xR bias | xW Brier |
|---|---:|---:|---:|---:|
| context | 1.454070 ± 0.000702 | 1.220468 | -0.001123 | 0.051931 |
| delivery | 1.430987 ± 0.001144 | 1.197846 | -0.000637 | 0.051789 |
| shot | 1.276667 ± 0.001227 | 1.141499 | +0.061521 | 0.050113 |
| contact | 1.139156 ± 0.000815 | 0.994778 | +0.059922 | 0.042779 |

### Calibration

| arm | xR intercept | xR slope | xW intercept | xW slope | xW log loss |
|---|---:|---:|---:|---:|---:|
| context | +0.080936 | 0.941533 | -0.004334 | 1.125618 | 0.210307 |
| delivery | +0.106038 | 0.922816 | +0.000900 | 0.999911 | 0.209225 |
| shot | +0.014733 | 0.946591 | -0.005198 | 1.084247 | 0.197261 |
| contact | -0.038514 | 0.984989 | -0.000920 | 1.073820 | 0.142462 |

| comparison | ΔLL | seed+match 95% CI | better seeds | ΔxR MAE |
|---|---:|---:|---:|---:|
| delivery_vs_context | -0.023083 | [-0.026214, -0.020125] | 5/5 | -0.022268 |
| shot_vs_delivery | -0.154319 | [-0.159775, -0.148796] | 5/5 | -0.056094 |
| contact_vs_shot | -0.137511 | [-0.141574, -0.133228] | 5/5 | -0.146970 |

### Process metrics (runs per delivery)

| metric | mean | match-bootstrap 95% CI | SD |
|---|---:|---:|---:|
| delivery_value | +0.000486 | [-0.004420, +0.005668] | 0.304280 |
| shot_selection_value | +0.062157 | [+0.053521, +0.070965] | 0.552412 |
| contact_value | -0.001598 | [-0.008642, +0.005216] | 0.721901 |
| execution_residual | -0.061521 | [-0.073867, -0.047941] | 1.544921 |

The mean staged increment can be near zero even when the information is valuable: positive and negative delivery/shot/contact effects cancel across the cohort. Use paired LL/MAE for information value and row/player aggregates for process value; do not interpret the cohort mean alone.

## Claim gate

**Realized delivery-information gate: PASS.** This gate does not establish future predictive value. The contact arm uses DeepCrease control as the available clean-contact proxy; no independent contact field exists.
