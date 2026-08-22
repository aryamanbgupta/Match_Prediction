# Future player xR validation v1

Prior windows use the first N eligible deliveries for a batter; targets use the next 100 deliveries from strictly later matches. Shrinkage constants were selected on validation and frozen for test. No golden or forward holdout was read.

**Promotion gate: FAIL.** The process metric does not clear the registered future-value gate; retain it as an internal descriptive measurement and do not optimize auction decisions against it yet.

## Test results

| N prior | players | raw MAE | EB MAE | RAA MAE | Delta MAE | shot xR MAE | Delta−EB [95% CI] |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 144 | 0.28097 | 0.18331 | 0.17798 | 0.20564 | 0.20637 | +0.02233 [+0.00350, +0.04193] |
| 100 | 114 | 0.23219 | 0.19207 | 0.18761 | 0.20269 | 0.20412 | +0.01062 [-0.00816, +0.02978] |
| 250 | 48 | 0.23592 | 0.20977 | 0.20909 | 0.22787 | 0.22826 | +0.01809 [-0.00369, +0.04091] |
| 500 | 11 | 0.20764 | 0.18780 | 0.17865 | 0.17321 | 0.17881 | -0.01458 [-0.04748, +0.02074] |
| 1000 | 0 | — | — | — | — | — | — |

## Decision payload

The locked payload is an internal luck-adjusted auction shortlist: Delta expected runs per 100 deliveries above the test-pool mean, restricted to batters with at least 250 prior and 100 strictly future labeled deliveries. It remains internal because the process labels derive from DeepCrease.

> **Erratum (2026-08-14 review, BR5):** the payload CSV shipped with this
> report computed `auction_runs_above_mean_per_100` against the
> **validation**-pool mean, not the test-pool mean the registered unit
> states — a constant offset of 100×(test mean − validation mean) runs on
> every row; rankings and the gate verdict are unaffected. The script now
> uses the test-pool mean (and records both means in the summary JSON);
> regenerate the payload before quoting absolute values.

## Interpretation

The process metric does not clear the registered future-value gate; retain it as an internal descriptive measurement and do not optimize auction decisions against it yet.
