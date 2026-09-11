# Sequence track — stage 1 report (three ball models through the fixed simulator)

Generated 2026-09-11T19:16+05:30 by `scripts/sequence_track/render_stage1_report.py`
from the files of record: every number in a table is read from a file;
the narrative passages are authored text in the generator. Config `experiments/configs/seq_stage1_sim_v1.yaml`
(sha256 `b41ceb34b8c1…`, `pin_stage1.py --verify` OK).
Acceptance checks and every result block: `docs/sequence_track/stage1_acceptance.md`.
Plan of record: `docs/SEQUENCE_TRACK_PLAN.md` § Stage 1.

**Status: single-checkpoint screen. No arm advances. No market claim.
Every gate is provisional (one training checkpoint per neural arm), so per
invariant 9 nothing here can be LANDED.**

## 1. Question

(i) System comparison: does the 50-feature token MLP (B) or full T1 (C),
on the certified replay path, simulate the winner market as well as the
production i7 ball model (A, 114 features)? (ii) Incremental evidence: does
full T1 improve on the token MLP in rollout? A50 (production XGBoost on
T1's 50 features) is the equal-information control. Pre-registered
expectation, written before any run: **parity everywhere, no arm advancing.**

## 2. Arms and settings

| arm | role | model dir | checkpoint | cache |
|---|---|---|---|---|
| A | production system (reference for B-A, C-A and A50-A) | `models/xgb_i7_noweights_production` | `7ee1e1809917` | i7 |
| A50 | production family at equal information (exploratory system control) | `models/embeddings/seq_stage1/a50` | `7705b330bac5` | i7 |
| B | nonlinear control (reference for C-B) | `models/embeddings/seq_stage1/retrain_i7/mlp/seed_101` | `dd240ccab040` | i7 |
| C | sequence candidate | `models/embeddings/seq_stage1/retrain_i7/full/seed_101` | `aa79292c2c53` | i7 |

One stats cache (i7, `venue_aliases_v1`, same-day ordering contract) for all
four arms (user decision 2026-09-11); B and C retrained on `data/xgb_data_i7`
(five seeds each, validation-LL selection, both select seed 101). Iteration
set `data/polymarket_test_v2` (255 fixtures, 2025-09-10 → 2026-04-16), odds
role `odds_iteration_v2`, roster-aware empirical bowler selector, B18 extras
graft on every arm, run-out constant 0.075077, no calibration, prefix cache
off, cpu, **threads 1**, clip [0.01, 0.99], per-fixture seed = low 31 bits of
sha256(`<id>:20260910`), **n_sims = 1600**. One runner, one
chronology, one replay lifecycle; the cross-arm audit asserted identical
fixture sets, as-of stamps, eligibility, odds-row identity, selector and
sidecar hashes, config/odds/context/engine/runner hashes on every shard.

## 3. Decision rule (registered before the run)

Primary slice ≥$50k. Contrasts are candidate − reference in winner log
loss (negative favours the candidate). Confirmatory family C−B, B−A, C−A,
Holm-adjusted; A50−A exploratory. Equivalence margin 0.007: interval
inside ±margin → parity; interval excludes zero and point beyond the margin
→ favourable / adverse; else inconclusive. B advances only if B−A is
favourable, C only if C−A is favourable; a favourable C−B alone is evidence
about that pair for stage 2, not an advancement.

## 4. Result — primary slice, Holm step-down

Intervals in the Holm tables are rank-local percentile intervals at level
1−α/(m−k+1) from the gate's own resamples; they are not simultaneous Holm
confidence intervals. Favourable/adverse requires the Holm-adjusted p ≤ 0.05
(step-down) and a point beyond ±0.007; parity is decided on the gate 95%
interval.

| contrast | family | point | rank-local interval (not simultaneous) | raw p | Holm p | level | label |
|---|---|---|---|---|---|---|---|
| C-B | confirmatory | -0.0257 | [-0.0346, -0.0102] | 0.0006 | 0.0018 | 0.9833 | **favourable** |
| B-A | confirmatory | +0.0247 | [-0.0027, +0.0554] | 0.0820 | 0.1640 | 0.9750 | **inconclusive** |
| C-A | confirmatory | -0.0009 | [-0.0277, +0.0288] | 0.8936 | 0.8936 | 0.9500 | **inconclusive** |
| A50-A | exploratory | +0.0276 | [+0.0095, +0.0419] | 0.0002 | — | 0.9500 | **adverse** |

**No arm advances.** C−B is favourable (departs from the parity expectation
in the favourable direction); B−A and C−A are inconclusive (intervals wider
than the ±0.007 band; B−A leans adverse); A50−A is adverse.

Secondary slices (the same Holm step applied for reporting; they are outside the confirmatory decision, which is the primary slice only):

≥$100k:

| contrast | family | point | interval | raw p | Holm p | level | label |
|---|---|---|---|---|---|---|---|
| C-B | confirmatory | -0.0267 | [-0.0402, -0.0102] | 0.0048 | 0.0096 | 0.9750 | **favourable** |
| B-A | confirmatory | +0.0401 | [+0.0059, +0.0662] | 0.0012 | 0.0036 | 0.9833 | **adverse** |
| C-A | confirmatory | +0.0133 | [-0.0230, +0.0380] | 0.6498 | 0.6498 | 0.9500 | **inconclusive** |
| A50-A | exploratory | +0.0423 | [+0.0260, +0.0496] | <0.0002 | — | 0.9500 | **adverse** |

all:

| contrast | family | point | interval | raw p | Holm p | level | label |
|---|---|---|---|---|---|---|---|
| C-B | confirmatory | -0.0215 | [-0.0323, -0.0074] | 0.0002 | 0.0006 | 0.9833 | **favourable** |
| B-A | confirmatory | +0.0270 | [+0.0054, +0.0527] | 0.0010 | 0.0020 | 0.9750 | **adverse** |
| C-A | confirmatory | +0.0055 | [-0.0158, +0.0294] | 0.6800 | 0.6800 | 0.9500 | **inconclusive** |
| A50-A | exploratory | +0.0258 | [+0.0130, +0.0375] | <0.0002 | — | 0.9500 | **adverse** |

Full gate table (Δprofit is units per flat 1-unit bet at cost scenario
0 bps / 0 bps on winnings; descriptive only — no betting claim):

| contrast | slice | n | blocks | ΔLL point | ΔLL ci95 | Δprofit point | Δprofit ci95 | gate verdict | provisional | gate sha256 |
|---|---|---|---|---|---|---|---|---|---|---|
| C-B | >=$50k (primary) | 167 | 18 | -0.02569 | [-0.03307, -0.01346] | +0.1859 | [+0.0493, +0.3255] | PROMISING | yes | `2f43c19b88c2` |
| B-A | >=$50k (primary) | 167 | 18 | +0.02475 | [-0.00118, +0.05257] | -0.0687 | [-0.1489, -0.0070] | FAILED | yes | `aa13ad29f269` |
| C-A | >=$50k (primary) | 167 | 18 | -0.00095 | [-0.02772, +0.02877] | +0.1173 | [-0.0690, +0.2887] | FAILED | yes | `038fdfac04f2` |
| A50-A | >=$50k (primary) | 167 | 18 | +0.02762 | [+0.00945, +0.04194] | -0.0918 | [-0.1741, -0.0081] | FAILED | yes | `f62d1c197151` |
| C-B | >=$100k | 110 | 11 | -0.02672 | [-0.03780, -0.01382] | +0.1696 | [+0.0297, +0.3646] | PROMISING | yes | `27dab32a4691` |
| B-A | >=$100k | 110 | 11 | +0.04005 | [+0.00879, +0.06267] | -0.1288 | [-0.3103, +0.0000] | FAILED | yes | `3e5177e628b4` |
| C-A | >=$100k | 110 | 11 | +0.01334 | [-0.02302, +0.03801] | +0.0408 | [-0.1516, +0.3087] | FAILED | yes | `28c63b055010` |
| A50-A | >=$100k | 110 | 11 | +0.04235 | [+0.02602, +0.04957] | -0.1955 | [-0.3176, -0.0880] | FAILED | yes | `8117301ae2a8` |
| C-B | all | 252 | 25 | -0.02149 | [-0.03053, -0.01004] | +0.1266 | [+0.0233, +0.2251] | PROMISING | yes | `39f277ce7100` |
| B-A | all | 252 | 25 | +0.02695 | [+0.00707, +0.04907] | -0.0794 | [-0.1418, -0.0250] | FAILED | yes | `013b5bb1d399` |
| C-A | all | 252 | 25 | +0.00546 | [-0.01578, +0.02942] | +0.0472 | [-0.0748, +0.1686] | FAILED | yes | `9e713e7947d7` |
| A50-A | all | 252 | 25 | +0.02584 | [+0.01296, +0.03751] | -0.1081 | [-0.1707, -0.0417] | FAILED | yes | `77d698ee9662` |

The gate's `PROMISING` / `FAILED` labels are its own LANDED-rule readout;
the stage-1 classification is the label column of the Holm tables.

### Reading

- **Full T1 beats the token MLP in rollout on every slice** (primary
  −0.0257 [−0.0346, −0.0102] rank-local interval; Holm p 0.0018). This is
  one checkpoint per arm. Historical motivation only: the 2026-08
  teacher-forced ablation (`T1_ABLATION_V1_MPS.md`) found a T1−MLP gap of
  −0.0004 with an interval crossing zero on *different* checkpoints and a
  different frame; that is not a matched teacher-forced evaluation of these
  retrained checkpoints, and a significant result in one experiment beside a
  non-significant one in another does not establish a difference between
  them. A matched teacher-forced score of these two checkpoints is the
  follow-up that would test whether rollout and teacher forcing disagree.
- **C−A is inconclusive with a point estimate near zero** (−0.0009,
  [−0.0277, +0.0288]); the interval establishes neither equivalence nor
  non-inferiority under ±0.007. **A50−A is adverse** (+0.0276
  [+0.0095, +0.0419], exploratory): within XGBoost, dropping the 64
  production-only features costs about 0.028 on this slice. Together these
  motivate further investigation of architecture and information use; they
  do not establish parity and do not quantify any recovery from sequence
  history (the registered rule forbids reading C−B or C−A as a measurement
  of sequence memory).
- **The token MLP loses to production** on both secondary slices and leans
  that way on the primary (inconclusive there).
- The MLP−logistic and T1−MLP findings of the ablation report describe
  different checkpoints on a different frame and are not restated here.

## 5. Uncertainty and what is not claimed

`tournament_time_block_v1`: 10,000 seed-42 whole-event resamples; 18 blocks
on the primary slice (167 paired records; one of 168 dropped symmetrically
by the gate's pairing rule), 11 at ≥$100k, 25 on all. Monte Carlo noise was
not measured on the primary slice or at 1,600 simulations; § 7 gives the
extrapolated two-batch difference spreads from the ten-fixture timing shard
(at four threads, before the thread-cap change), which are the basis of the
user's count decision and nothing more. Every gate is provisional (single
checkpoint). Confirmation of any contrast needs
five training seeds and two independent simulation batches (plan § Stage 1).
No market claim: the Δprofit intervals are reported, not claimed.

## 6. Exploratory readouts (no advancement, no claim)

Realism, first innings, A…C (247 fixtures scored per arm; 3
no-decided-winner and 5 D/L fixtures excluded):

| arm | n | bias (runs) | abs bias | P10–P90 coverage | sim P10 / P50 / P90 | wickets actual / sim |
|---|---|---|---|---|---|---|
| A | 247 | +2.12 | 28.24 | 0.753 | 130.3 / 170.3 / 208.5 | 7.00 / 6.86 |
| A50 | 247 | +0.50 | 28.59 | 0.745 | 127.6 / 168.7 / 207.9 | 7.00 / 6.82 |
| B | 247 | +3.01 | 28.53 | 0.761 | 129.7 / 170.9 / 211.4 | 7.00 / 6.68 |
| C | 247 | +5.06 | 28.21 | 0.773 | 130.8 / 172.6 / 214.5 | 7.00 / 6.68 |

Props, paired sim − as-of fair baseline per family (Brier; MAE for `_mae`),
250 fixtures per arm, 15 of 33 families scorable; `*` marks an interval
excluding zero; negative favours the simulator:

| family | metric | A | A50 | B | C |
|---|---|---|---|---|---|
| batter_50plus | brier | -0.0013 [-0.0022, -0.0005]* | -0.0015 [-0.0024, -0.0007]* | -0.0015 [-0.0024, -0.0006]* | -0.0017 [-0.0027, -0.0007]* |
| batter_runs_mae | mae | -0.7458 [-0.8832, -0.6110]* | -0.7545 [-0.8886, -0.6230]* | -0.6889 [-0.8380, -0.5474]* | -0.6012 [-0.7521, -0.4569]* |
| bowler_wkts_1plus | brier | +0.0078 [+0.0047, +0.0110]* | +0.0075 [+0.0044, +0.0105]* | +0.0070 [+0.0037, +0.0101]* | +0.0066 [+0.0032, +0.0098]* |
| bowler_wkts_2plus | brier | +0.0056 [+0.0030, +0.0082]* | +0.0062 [+0.0036, +0.0087]* | +0.0067 [+0.0037, +0.0096]* | +0.0061 [+0.0034, +0.0091]* |
| bowler_wkts_3plus | brier | +0.0023 [+0.0013, +0.0033]* | +0.0023 [+0.0013, +0.0033]* | +0.0025 [+0.0015, +0.0036]* | +0.0024 [+0.0014, +0.0034]* |
| innings_runs_ou_160_5 | brier | -0.0305 [-0.0478, -0.0124]* | -0.0283 [-0.0442, -0.0124]* | -0.0267 [-0.0444, -0.0090]* | -0.0276 [-0.0464, -0.0082]* |
| innings_runs_ou_170_5 | brier | -0.0265 [-0.0438, -0.0094]* | -0.0240 [-0.0392, -0.0093]* | -0.0252 [-0.0424, -0.0087]* | -0.0266 [-0.0455, -0.0075]* |
| innings_runs_ou_180_5 | brier | -0.0275 [-0.0426, -0.0122]* | -0.0234 [-0.0364, -0.0104]* | -0.0255 [-0.0405, -0.0108]* | -0.0265 [-0.0436, -0.0093]* |
| pp_total_ou_45_5 | brier | -0.0283 [-0.0405, -0.0162]* | -0.0261 [-0.0377, -0.0152]* | -0.0287 [-0.0436, -0.0146]* | -0.0288 [-0.0423, -0.0159]* |
| pp_total_ou_50_5 | brier | -0.0330 [-0.0467, -0.0189]* | -0.0298 [-0.0426, -0.0169]* | -0.0301 [-0.0468, -0.0138]* | -0.0322 [-0.0473, -0.0171]* |
| pp_total_ou_55_5 | brier | -0.0224 [-0.0353, -0.0097]* | -0.0218 [-0.0332, -0.0104]* | -0.0215 [-0.0365, -0.0065]* | -0.0222 [-0.0362, -0.0085]* |
| team_highest_individual_ou_29_5 | brier | -0.0051 [-0.0089, -0.0018]* | -0.0050 [-0.0090, -0.0016]* | -0.0037 [-0.0066, -0.0010]* | -0.0039 [-0.0066, -0.0014]* |
| team_highest_individual_ou_34_5 | brier | -0.0099 [-0.0156, -0.0049]* | -0.0102 [-0.0157, -0.0053]* | -0.0085 [-0.0129, -0.0043]* | -0.0088 [-0.0133, -0.0046]* |
| team_highest_individual_ou_39_5 | brier | -0.0108 [-0.0176, -0.0044]* | -0.0105 [-0.0170, -0.0046]* | -0.0092 [-0.0149, -0.0036]* | -0.0098 [-0.0157, -0.0041]* |
| top_batter | brier | -0.0003 [-0.0008, +0.0001] | -0.0002 [-0.0007, +0.0003] | -0.0000 [-0.0003, +0.0002] | -0.0001 [-0.0003, +0.0001] |

All four arms beat the fair baselines on the innings-total, powerplay-total,
team-highest-score and batter-50+ lines with intervals excluding zero, and
lose on the bowler-wicket lines. Context: the original E2 v2 audit
(`reports/e2_prop_fair_baselines.md`, retired v7 stack, 100 simulations)
found no binary family beating an as-of fair baseline; BR2's later
restatement (`research/handoff/BR2/e2_restated.md`) already showed
favourable movement on several overlapping families on the repaired
engine, so stage 1 is not the first such reading. These are per-family
intervals with no multiplicity adjustment, on a different engine state and
simulation count from either predecessor; exploratory, and not comparable
line for line.

## 7. 1b convergence and timing (spreads only)

Ten-fixture long-innings shard, three batch seeds, candidates 50/100/200/400/800.
Empirical range (max − min over the three batches; the config calls this
`range_95`) of the paired shard-mean contrast and its full-set equivalent
(× sqrt(10/255)):

| n_sims | contrast | range_95 | scaled range_95 | SD |
|---|---|---|---|---|
| 50 | C-B | 0.1010 | 0.0200 | 0.0522 |
| 100 | C-B | 0.0700 | 0.0139 | 0.0351 |
| 200 | C-B | 0.0443 | 0.0088 | 0.0241 |
| 400 | C-B | 0.0547 | 0.0108 | 0.0285 |
| 800 | C-B | 0.0350 | 0.0069 | 0.0179 |
| 50 | B-A | 0.0438 | 0.0087 | 0.0224 |
| 100 | B-A | 0.1055 | 0.0209 | 0.0556 |
| 200 | B-A | 0.0551 | 0.0109 | 0.0314 |
| 400 | B-A | 0.0811 | 0.0161 | 0.0419 |
| 800 | B-A | 0.0437 | 0.0086 | 0.0221 |

Two-seed variability: the SD of the per-fixture DIFFERENCE between the
contrast under batch seed 20260910 and under 20260911 (so, for equal
independent batch variances, √2 × a single batch's SD), divided by
sqrt(10), then × sqrt(10/255). An extrapolation from a deliberately
long-innings shard on all ten fixtures (not the primary slice), measured at
four threads (the full run used one; floating-point reductions can differ),
and never measured at 1,600:

| n_sims | contrast | scaled SD |
|---|---|---|
| 50 | C-B | 0.0105 |
| 100 | C-B | 0.0084 |
| 200 | C-B | 0.0042 |
| 400 | C-B | 0.0042 |
| 800 | C-B | 0.0025 |
| 50 | B-A | 0.0105 |
| 100 | B-A | 0.0094 |
| 200 | B-A | 0.0074 |
| 400 | B-A | 0.0052 |
| 800 | B-A | 0.0032 |

"range_95" above is the min–max of three batches, an empirical range, not
an established population 95% range. The registered range rule (< 0.002)
was not met at any permitted count (the a/sqrt(n) fit crosses at ≈5,264 for
C−B, above the 3,200 joint seed cap). **User decision 2026-09-11:** the
operational reading is the two-batch-difference scaled SD at or below
≈0.002 at the chosen count; n_sims = 1,600 (≈0.0018 / 0.0023 extrapolated
by 1/√2 from 800). Recorded in `convergence_protocol.stop_rule_reading` as
a change of statistic made with the numbers in hand. 50 is a noise-curve
point only.

Thread cap: `sim_t1.py` hard-coded four torch threads and defeated the
runner's cap; fixed (honours `OMP_NUM_THREADS`), threads registered at 1,
ten one-thread shards give 6.4× the throughput of one process. Full run wall
(ten concurrent shards):

| arm | shards | wall min / mean / max (min) | peak RSS (MB) |
|---|---|---|---|
| A | 10 | 36 / 37 / 38 | 792 |
| A50 | 10 | 40 / 40 / 41 | 888 |
| B | 10 | 40 / 41 / 43 | 629 |
| C | 10 | 139 / 142 / 145 | 512 |

## 8. 1c shard consistency

Serial vs sharded per-fixture outputs identical on all four arms across the
five registered boundary cases (parsed evaluator record, raw simulation
rows and per-fixture provenance compared field by field; the one exclusion
is the run-cumulative `matches_advanced` counter, replaced by the ordered
same-day list and a decomposition check — acceptance D10) after one fix: the frozen runner derived the I3
block id from the run's own fixture directory, so a shard stamped a
different `block_start` than the serial run for one fixture; `run_arm
--cluster-source-dir` (pinned to the registered set, asserted by the audit)
fixed it at the source; re-run 117/117 checks pass, merge re-stamp changed 0
records on every arm.

## 9. Registered deviations, asymmetries, limitations (restated, none dropped)

Deviations (9), each with its registered consequence:

- `single_global_rng_stream`: the sub-seeds are derived and recorded in arm_provenance.json but are not separately consumed. Every arm still shares one identical draw schedule per fixture, which is what the paired comparison needs; what is lost is the ability to hold, say, the extras draws fixed while the outcome draws move
- `extras_graft_attribute_seam`: run_arm refuses to run if a model directory carries its own sidecar that differs from the requested one, so an ambiguous extras law fails closed rather than passing silently
- `run_out_sidecar_absent`: the config pins the constant, its value read by import, and its source line, and the engine md5 in provenance covers any change to it
- `scorers_consume_raw_sims`: realism and prop numbers describe exactly the simulations the winner-LL numbers came from; they are not comparable line for line with historical prop_backtest.py reports
- `no_parallel_within_a_run`: the 1c shard-consistency check exists precisely because sharding is the only way to use more than one core; each shard keeps the full replay context
- `clip_seam_replaces_rather_than_composes`: no stage-1 winner log loss is comparable to any number produced under the [0.05, 0.95] clip, BR2 G1 included. prob_clip = None still reproduces the legacy behaviour exactly, so no existing runner or stored result moves
- `resolved_flag_reads_cricsheet_not_the_odds_row`: a fixture the odds file calls settled but cricsheet records with no winner (a no-result or an abandoned match) counts as unresolved for every arm alike. The cross-arm audit tests a real invariant rather than a copied field
- `as_of_stamp_for_odds_less_fixtures`: the two stamps are equal wherever both exist, and the audit compares them across arms uniformly; an odds-less fixture is still stamped and still audited
- `roster_selector_installed_by_factory_swap`: WITHOUT --bowler-roster-policy no roster policy is applied and the arm silently runs the plain empirical selector, which is NOT the registered configuration. Every command line in this config carries the flag, and the cross-arm audit asserts the selector class and roster hash match across arms

Known asymmetries (3):

- `feature_set_114_vs_50`: B-A and C-A are system contrasts, not model-only contrasts: they compare a 114-feature system with a 50-feature one. A50-A isolates the feature set at equal information, within one model family (same family, same frame, same cache, same hyperparameters, 50 columns instead of 114). C-B is the registered full-sequence T1 versus token-MLP contrast (architectures differ in outcome-history embedding, attention and parameter count), so it is not an isolation of sequence memory: it is a comparison of two specific architectures that happen to differ in several ways at once
- `arm_a_path_is_the_replay_lifecycle`: BR2's G1 winner-LL line is NOT a stage-1 reference number for arm A. The only valid reference for every stage-1 contrast is arm A's own stage-1 run under this config
- `br2_g1_is_not_a_stage_1_number`: NO stage-1 number is comparable to BR2, in either direction. A stage-1 result that differs from BR2's G1 line is not evidence of anything; the only admissible baseline for B-A, C-A and A50-A is arm A's own run under this config

Removed 2026-09-11 by the i7 retrain: stats_cache_i7_vs_v3; training_frame_i7_vs_v3.

Known limitations (1):

- `global_prior_not_as_of`: shared exposure, unknown differential effect. The shift enters every arm's inputs identically, but the four arms are different functions of those inputs — a 114-feature gradient-boosted model, the same family at 50 columns, a token MLP and a transformer — and how much each one's output moves in response has not been measured. This limitation therefore cannot be used to argue that a stage-1 contrast is unaffected, nor that any arm is favoured or disfavoured: the direction and size of the effect on B-A, C-A, C-B and A50-A are simply unknown. It is an inherited limitation of the iteration screen, not a stage-1 defect, and it is on the backlog (TODO.md, 'Global outcome prior is not as-of-date')

Astra review rounds for the retrain/re-pin commit: D8 in the acceptance
file; end-of-stage review: D12.9.

## 10. In plain language: what was tested, what came out, what "no arm advances" means

**What we tested.** Four ball-by-ball models were each asked to play out
every one of 255 real T20 matches 1,600 times, from the same starting
information, using the same simulator, the same bowler-choice rules and the
same extras law. From those replays each model produced a win probability
for every match, and we compared those probabilities with what actually
happened, using log loss (lower is better). The four models: **A**, the
current production XGBoost with 114 features (the model we already use);
**A50**, the same XGBoost family given only the 50 simpler features the
neural models see; **B**, a small neural network (token MLP) on those 50
features with no memory of earlier balls; **C**, the transformer (T1) on the
same 50 features, which can look back over the balls already bowled in the
innings.

**What came out, on the 168 matches with the deepest betting markets.**
The transformer beat the memory-less network clearly (about 0.026 better
log loss, and the interval does not come near zero). The transformer and
production came out indistinguishable: the point difference is about
0.001 in the transformer's favour, but the uncertainty band is ±0.028
wide, so we cannot say either is better. The memory-less network was
somewhat worse than production, and on the wider slices clearly worse.
The XGBoost restricted to 50 features was clearly worse than production by
about 0.028, which says the 64 extra hand-built features matter to XGBoost.

**What "no arm advances" means.** Before the run we wrote down a rule: a
neural model "advances" (goes forward to the next round of testing as a
candidate to replace or join production) only if it beats production by a
clear margin, with the uncertainty band excluding zero and the point beyond
0.007. Neither B nor C met that. C tied production; B lost. So under the
rule nothing moves forward as a production candidate, and production stays
the model of record. "No arm advances" is not a statement that the
transformer is worse; it is a statement that we did not show it to be
better.

**So which is the best model right now?** For predicting match winners in
production: the existing XGBoost (A). It is the model of record, the
transformer only matched it, and the transformer was trained once (one
random seed), so its number could move if retrained. For the research
question "does looking back over the innings help?": the transformer beat
the memory-less network on identical inputs, which is the first clean
rollout evidence in this repo that within-innings sequence carries
something. That is the finding that goes into stage 2.

**What comes next.** (1) Confirm, not assume: retrain the transformer and
the MLP on five seeds and simulate two independent batches, so the C−B gap
is shown to be a property of the architecture and not of one lucky
checkpoint. (2) Score these same two checkpoints under teacher forcing (one
ball at a time, no rollout) to see whether the rollout advantage is real
or an artefact of how the simulator uses the model. (3) Stage 2 as planned:
variants of the transformer that forget stale history or attend only to the
same batter and bowler, to learn *which* part of the history helps and
whether the death-over harm seen in August goes away. (4) A separate,
registered look at why every arm beat the fair prop baselines on totals
lines here when the old audit said none did; that could matter for prop
markets but is exploratory today. Nothing here is a betting claim: every
result is one checkpoint, and no market edge has been established.

## 11. Next (as registered)

Per the plan: no arm advances; C−B goes to stage 2 as evidence about the
full-T1 vs token-MLP pair (which arm variants, forgetting / ownership). A
matched teacher-forced score of these two checkpoints would test whether
rollout and teacher forcing actually disagree. A50−A adverse beside an
inconclusive C−A is a lead on what the 64 production-only features carry,
not a measurement. Any claim about this checkpoint needs the five-seed,
two-batch confirmation.
