# T1 exact-information ablation v1 — interim result

> **SUPERSEDED (2026-08-11, same day) by
> [`T1_ABLATION_V1_MPS.md`](T1_ABLATION_V1_MPS.md)** — the full five-seed ×
> five-arm matrix and the authoritative claim status. Kept as registration
> evidence of the partial run; do not cite numbers from this file.

2026-08-11 · registered config:
`experiments/configs/t1_ablation_v1.yaml` · sealed golden/forward holdouts were
not read · raw local artifacts: `models/embeddings/t1_ablation_v1/`

## Status: PARTIAL — logistic, all five MLP seeds, no-attention seed 7

| split | scaled logistic | token MLP mean ± seed SD | ΔLL (MLP − logistic) | paired match-clustered 95% CI | better seeds |
|---|---:|---:|---:|---:|---:|
| validation | 1.443294 | **1.438392 ± 0.000663** | **−0.004901** | [−0.005648, −0.004183] | 5/5 |
| test | 1.434040 | **1.429620 ± 0.000854** | **−0.004420** | [−0.005050, −0.003826] | 5/5 |

The scaled multinomial logistic control converged in 271 iterations. The
capacity-matched token MLP has 272,006 parameters. Across its five seeds, mean
confidence ECE is 0.003849 validation / 0.006455 test and mean multiclass Brier
is 0.702571 / 0.700839. The LL gain is therefore clearer than the calibration
gain; confidence ECE is slightly worse than logistic in aggregate.

This establishes a stable nonlinear current-token gain over logistic. It does
**not** establish a sequence gain. Compared descriptively with the old full-T1
seed-42 headline (1.4372 / 1.4288, rounded), only about 0.0012 / 0.0008 LL
remains. That comparison is neither seed-paired nor uncertainty-qualified.

## Where the nonlinear gain appears

| slice | validation ΔLL | test ΔLL |
|---|---:|---:|
| thin pair (<200 train balls) | −0.005400 | −0.004215 |
| powerplay | −0.004052 | −0.003142 |
| middle overs | −0.003177 | −0.002970 |
| death overs | **−0.011984** | **−0.011504** |
| innings 1 | −0.004387 | −0.003514 |
| innings 2 | **−0.005612** | **−0.005463** |

The concentration in death overs and the larger second-innings gain point to
nonlinear pressure/current-state interactions as the main explanation. The
remaining arms must determine whether previous outcomes or longer history add
anything beyond that.

## No-attention seed-7 diagnostic

| split | no-attention LL | ΔLL vs logistic (95% CI) | ΔLL vs same-seed MLP (95% CI) |
|---|---:|---:|---:|
| validation | 1.439925 | −0.003369 [−0.004159, −0.002592] | **+0.001747** [+0.000949, +0.002556] |
| test | 1.432048 | −0.001992 [−0.002634, −0.001359] | **+0.002632** [+0.001973, +0.003353] |

This arm beat logistic but lost to the MLP despite also receiving the shifted
previous-outcome token. It reached the 30-epoch ceiling and was still improving,
so the result is optimization-limited and must not be read as evidence that the
previous outcome hurts. The registered five-seed run remains necessary; a
separate longer-schedule sensitivity may be added after the registered matrix.

The initial implementation also exposed a padding-mask NaN: diagonal attention
plus key-padding left padded queries with no legal key. The invalid run was
discarded; the arm now omits the key-padding mask because diagonal attention
cannot mix padding into real tokens, and a finite-padding regression test was
added before the reported rerun.

## Required before a T1 verdict

- Finish all five registered seeds for MLP, no-attention, no-history and full.
- Compare full T1 directly against every arm using row-paired, match-clustered
  intervals—not only against logistic.
- Report seed dispersion and the registered thin-player, phase and innings
  slices.
- Retain “promising pilot” unless the full arm clears every preregistered claim
  gate in at least four of five seeds.

## Reproducibility record

| input | SHA-256 |
|---|---|
| registered config | `f68cb59e67ab6e805ae28a5462b150777d94819f44c672665c60877b8a56422a` |
| train parquet | `cb90cd00eb5d84474fa7c4cab05997fcb3c1d437387ae6c6edb18181fee4b378` |
| validation parquet | `04a943d6b6369559b021485d1782caf1562c4c64b5b5687863272a13f5da00f4` |
| test parquet | `47d1cd18758193812fb2fbf89223329b26ffde3cdcaf086e487b969f329c6725` |
| unseen-pair masks | `6c75172cf538b32254a6d6cdf56096ec2512cab6d4d44180ab26871c9267fe96` |
| probe labels | `d5c6d3a16157c89f9589dcb58fab1c805d2e191069ccc3c756811bf55aa1d292` |

The runner writes pickle-free row-aligned probability archives, checkpoints,
per-run metrics, an environment manifest and the final aggregate summary. The
causal contracts, padding behavior and archive format are covered by nine
focused tests.
