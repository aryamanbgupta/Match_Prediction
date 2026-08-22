# T1 exact-information ablation v1 — final MPS matrix

2026-08-11 · config `experiments/configs/t1_ablation_v1_mps.yaml` · five MPS
seeds `{7,13,29,42,101}` · validation early stopping · paired two-level
bootstrap over training seeds and complete matches (2,000 repetitions, seed
29) · no sealed golden or forward holdout read.

> **Estimator note (2026-08-21; resolved 2026-08-22).** Every "seed+match"
> CI in this report is the **seed-mean** estimator: seeds resampled with
> replacement and averaged, targeting the across-seed mean. It is narrower
> than the xR reports' seed-draw estimator (one seed per replicate), which
> additionally carries single-seed variance — the two are not comparable
> across reports. **Gate-estimator decision (2026-08-22):** the registered
> gate reads the seed-mean+match CI — a comparison must clear zero net of
> both match resampling and seed noise; the YAML's shorthand
> "match-clustered" is read as this joint interval, and the runner now
> stamps `claim_gate.estimator` accordingly. The v1 verdict is
> **estimator-robust**: full_vs_mlp straddles zero on the plain
> match-clustered CI too (validation [−0.001598, +0.000044]; test
> [−0.001061, +0.000327]), so the sequence-gate FAIL does not depend on the
> estimator choice.

## Claim status: NONLINEAR CURRENT-STATE GAIN SUPPORTED; SEQUENCE GATE FAILED

All models receive the identical 50 pre-ball features. Negative ΔLL is better.

| split | logistic | token MLP | no attention | no outcome history | full T1 |
|---|---:|---:|---:|---:|---:|
| validation LL, mean ± seed SD | 1.443294 | **1.438116 ± 0.000306** | 1.440668 ± 0.000919 | 1.440416 ± 0.001501 | **1.437331 ± 0.000648** |
| test LL, mean ± seed SD | 1.434040 | **1.429320 ± 0.000288** | 1.432332 ± 0.000586 | 1.431773 ± 0.001310 | **1.428934 ± 0.000573** |

The token MLP beats converged logistic in 5/5 seeds: validation ΔLL −0.005177
[−0.005964, −0.004414], test −0.004720 [−0.005471, −0.004046]. Nonlinear
current-state interactions therefore explain most of the original T1 headline.

## The decisive comparison

| full T1 minus reference | validation ΔLL [seed+match 95% CI] | test ΔLL [seed+match 95% CI] | direction |
|---|---:|---:|---:|
| token MLP | **−0.000785 [−0.001736, +0.000229]** | **−0.000387 [−0.001215, +0.000489]** | 4/5 seeds |
| no attention | −0.003337 [−0.004558, −0.002213] | −0.003398 [−0.004231, −0.002575] | 5/5 |
| no outcome history | −0.003085 [−0.004349, −0.001939] | −0.002839 [−0.004005, −0.001803] | 5/5 |

Full T1 does **not** beat the strongest non-sequential control with an interval
below zero. The preregistered gate therefore fails on both validation and test.
The broad claim that within-innings sequence context adds predictive value is
not supported at the registered resolution.

Within the transformer family, the full model does beat the masked variants.
That supports an architecture-specific role for the outcome-history/attention
pathway, but the masked arms are optimization-sensitive and both lose to the
simpler MLP. This is not evidence that a sequence model is the better predictor.

Per-seed full-minus-MLP LL:

| seed | validation | test |
|---:|---:|---:|
| 7 | −0.001162 | −0.000955 |
| 13 | +0.000314 | +0.000514 |
| 29 | −0.000602 | −0.000443 |
| 42 | −0.001355 | −0.000766 |
| 101 | −0.001121 | −0.000283 |

## Where the residual appears

Full T1 minus token MLP:

| slice | validation ΔLL [95% CI] | test ΔLL [95% CI] | interpretation |
|---|---:|---:|---|
| thin pair (<200 train balls) | −0.001602 [−0.003179, −0.000024] | −0.000267 [−0.001450, +0.000980] | validation-only hint |
| powerplay | −0.002245 [−0.003778, −0.000702] | −0.000637 [−0.001904, +0.000590] | validation-only hint |
| middle overs | **−0.001580 [−0.002945, −0.000130]** | **−0.002062 [−0.003181, −0.000828]** | consistent benefit |
| death overs | **+0.004615 [+0.001989, +0.007315]** | **+0.005342 [+0.003222, +0.007818]** | consistent harm, 0/5 seeds |
| innings 1 | −0.001872 [−0.003066, −0.000683] | −0.000952 [−0.001936, +0.000050] | mainly validation |
| innings 2 | +0.000455 [−0.001011, +0.002041] | +0.000245 [−0.001052, +0.001572] | no benefit |

The stable death-over harm and absence of a second-innings benefit matter for
T4. They are compatible with—though do not prove—a model whose learned history
path is less useful during chases and late-innings pressure, exactly where the
rollout showed first-innings orientation bias. Simulator parity and posterior-
predictive checks should test that connection before any calibration.

## Raw calibration diagnostics

No post-hoc calibrator was fit or applied.

| split | metric | logistic | MLP | full T1 |
|---|---|---:|---:|---:|
| validation | multiclass Brier | 0.704471 | 0.702531 | **0.701970** |
| validation | confidence ECE-15 | **0.003098** | 0.003318 | 0.004706 |
| test | multiclass Brier | 0.702632 | 0.700813 | **0.700449** |
| test | confidence ECE-15 | **0.005570** | 0.006001 | 0.007082 |

Full T1 has the best Brier score but the worst confidence ECE of these three.
The small LL/Brier edge must not be described as a calibration improvement.

## Decision and next model step

1. Retire the broad T1 “sequence hypothesis is real” claim.
2. Keep the MLP as the strongest supported 50-feature control.
3. Treat full T1 as an architecture-specific research arm, not a promoted
   production model.
4. After simulator parity, test a residual sequence model over MLP or production
   logits, targeted at middle-over residuals and explicitly gated against
   death-over/chase degradation.

## Reproducibility record

Raw checkpoints, row-aligned probabilities, metrics, aggregate YAML and an
84-file SHA-256 artifact manifest are under
`models/embeddings/t1_ablation_v1_mps/`. These are local ignored artifacts; this
tracked report and registered config are the durable record.

| input/source | SHA-256 |
|---|---|
| registered config | `29bea640bcbea6a74cacb16c9fe745c2066ee36f76900816b436e127bde93dc1` |
| `transformer_t1.py` used for training | `e748ab2bdb1affb5c7a26ade71b5bd4388358917c553fb1ab31fe5485f8e0231` |
| aggregation runner | `e07f9b6fcfc5ad203c7b8885dbc7e6cc1544fcebd71807c127286ce61080e235` |
| feature builder (`embeddings_e1.py`) | `9d348751b8008c1776769bd8c4021056ba3d71a0cbdde53a169f50a2eb77a3e4` |
| train parquet | `cb90cd00eb5d84474fa7c4cab05997fcb3c1d437387ae6c6edb18181fee4b378` |
| validation parquet | `04a943d6b6369559b021485d1782caf1562c4c64b5b5687863272a13f5da00f4` |
| test parquet | `47d1cd18758193812fb2fbf89223329b26ffde3cdcaf086e487b969f329c6725` |
| unseen-pair masks | `6c75172cf538b32254a6d6cdf56096ec2512cab6d4d44180ab26871c9267fe96` |
| probe labels | `d5c6d3a16157c89f9589dcb58fab1c805d2e191069ccc3c756811bf55aa1d292` |

Environment: Python 3.9.25, NumPy 1.24.3, pandas 2.3.2, scikit-learn
1.6.1, PyTorch 2.5.1, Apple MPS. Ten focused causal, padding, archive and
two-level-bootstrap tests pass.

