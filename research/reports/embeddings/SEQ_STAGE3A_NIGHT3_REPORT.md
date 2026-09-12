# Sequence track — stage 3a night-3 report (negative-transfer screen, six configurations, five seeds, validation only)

Generated (declared timestamp, the only value that moves on a re-render): **2026-09-12T20:32:02.036240+00:00** by `scripts/sequence_track/render_night3_report.py` from the files of record. Every number in every table is read from a file; the narrative passages are authored text in the generator.

* statistics: `eval_out/seq_stage3_night3/stats.json` (sha256 `35c5b301cd0a…`), written 2026-09-12T20:30:32.078867+00:00 by `scripts/sequence_track/stage2_stats.py`
* config: `experiments/configs/seq_stage3_night3_v1.yaml` (sha256 `fc952ba6ab4b…`)
* runs: `models/embeddings/seq_stage3/night3/runs/<config>/seed_<s>/` — `summary.yaml`, `metrics.json`
* frozen commit `90acca8`; this render's launch state `65f2f63`

**Status: validation-only five-seed, outcome-informed screen. `cohort_status: DEFERRED_UNOPENED`, `cohort_scored: false`, `advances: []`, `provisional: true`. No arm advances. No market claim. No LANDED verdict — this evidence is provisional (validation-only, checkpoint-selected on the same split it is scored on, with the untouched cohort unopened), and provisional evidence can never be LANDED (invariant 9). The verdict is the user's decision and none was logged: `research/log_verdict.py` was not called.**

Registered evidence status, verbatim from the statistics file: VALIDATION-ONLY, PROVISIONAL. Every number of this block is screening evidence: checkpoint selection is on the same validation split the contrasts are computed on, the target definitions and the whole screening hypothesis are OUTCOME-INFORMED by the stage 2 five-seed readout (they are frozen here before any night-3 performance read, which makes them pre-registered, not confirmatory), the untouched cohort is NOT opened, and no test split, market price or betting layer is touched anywhere. Nothing here can be LANDED (CLAUDE.md invariant 9), nothing advances, and no artifact of record changes.

Statuses this stage can emit: `SCREEN_PASS`, `SCREEN_NOT_PASS`, `NOT_EVALUABLE`. There is no advancement status.

## 1. The question

Does pooling every tier of T20 cricket into one training set **hurt the tier you actually care about**? That is the negative-transfer question, and it is the only question this block asks. If pooling hurts, a model trained on the target tier alone should beat the pooled model **on the target tier's own validation rows**; if pooling helps, it should not.

Two targets are registered. **P** is the premium-league target, `competition_tier == 3`. **E** is P together with every international in which both teams are ICC full members. `competition_tier` is a pre-match event-name rule, not a market or quality measurement: "premium" and "elite" are the rule's labels, not findings.

Registered purpose, verbatim from the config: Screen for negative transfer from pooled T20 training onto two registered "elite" targets. Six token-MLP arms under one step budget: a pooled unconditioned control, a pooled tier-conditioned arm, and for each target T in {P, E} a tier-conditioned arm trained on that target's matches only plus a row-matched pooled control trained on a deterministic subsample of whole training matches of the same row count. The question is whether restricting training to the target buys anything on the target's own validation rows over (a) the pooled conditioned arm and (b) a pooled arm with the same number of rows; and, separately, whether tier conditioning alone buys anything on all rows.

Sign convention: **candidate minus reference, in validation log loss; negative favourable.** Every arm here is `token_mlp` — no attention layer anywhere in this block — so nothing here bears on any sequence mechanism.

| config id | training rows selected | tier conditioning | role |
|---|---|---|---|
| `mlp_pool` | 1,876,971 | none | control |
| `mlp_pool_tiercond` | 1,876,971 | tier embed dim 8 + per-tier output bias | candidate; reference(s) `mlp_pool` |
| `mlp_P_tiercond` | 591,342 | tier embed dim 8 + per-tier output bias | candidate; reference(s) `mlp_pool_tiercond`, `mlp_rowmatch_P_tiercond`, `mlp_pool` |
| `mlp_rowmatch_P_tiercond` | 591,582 | tier embed dim 8 + per-tier output bias | control; reference(s) `mlp_pool` |
| `mlp_E_tiercond` | 812,169 | tier embed dim 8 + per-tier output bias | candidate; reference(s) `mlp_pool_tiercond`, `mlp_rowmatch_E_tiercond`, `mlp_pool` |
| `mlp_rowmatch_E_tiercond` | 812,250 | tier embed dim 8 + per-tier output bias | control; reference(s) `mlp_pool` |

Training rows are read from each run's `metrics.json` `training_schedule.train_rows_selected` (seed 7; identical across seeds within a config). 5 of the 6 configurations are tier-conditioned; only `mlp_pool` is not.

Every arm trains under one step budget, read from the config's `training` block: 3,840 steps at `eval_every` 128, `d_model` 128, 2 layers, 4 heads, batch 128, lr 0.0003, device `mps`, on frame `data/xgb_data_i7` with cache role `stats_cache_i7`. Validation is **never** filtered: validation log loss = ball-weighted mean negative log-likelihood over ALL validation rows, for EVERY arm including the target-only arms. Checkpoint selection is never re-done per target (see the shared-artifact contract).

## 2. The decision rule, as registered before any result was read

Three families, each registered once, each with its own Holm step-down within the family and no correction across families.

* **`family_3a_P`** and **`family_3a_E`** — candidate `mlp_<T>_tiercond`, four members: `superiority_1` = target-only − **pooled-conditioned** on the target slice; `superiority_2` = target-only − **row-matched pooled** on the target slice; `death_gate` and `chase_gate` = target-only − `mlp_pool` on the death and chase rows **restricted to the target**. `SCREEN_PASS` requires **all four** members to reject favourably under Holm — both superiority members CI-clean favourable, both gates strictly `U95 < +0.002` — plus at least ten tournament blocks on each member's slice, five complete paired seeds, and at least 4 of 5 favourable per-seed directions on **both** superiority members. The inherited all-row condition is **amended away** (`all_row_condition: none`): a target-only model is not asked to be better on rows outside its target, so its `all` reading is exploratory and not part of this screen.
* **`family_3a_cond`** — candidate `mlp_pool_tiercond`, three members: `primary` = pooled-conditioned − pooled on `all`, plus the same two non-inferiority gates on `death` and `chase` against `mlp_pool`. `all_row_condition: member` — here the all-row reading *is* the primary, so the condition is discharged inside the family. The 4-of-5 direction requirement applies to `primary`.

Uncertainty: 2,000 tournament-time-block bootstrap replicates at rng seed 29, α = 0.05, non-inferiority margin +0.002 log loss. p-value convention, verbatim: p_raw = min(1, 2 * min[P(d* - t <= 0), P(d* - t >= 0)]) over the bootstrap delta draws, with t = 0 for superiority and t = +0.002 for non-inferiority.

**Holm adjustment, in its general form:** `p_holm(r) = min(1, max over j <= r of (m − j + 1) · p_(j))` with the step-down stopping rule, where **m is that family's own member count** — m = 4 for `family_3a_P` and `family_3a_E`, m = 3 for `family_3a_cond` — applied **within each family only**, never pooled across families. A slice with fewer than 10 blocks is **descriptive** and its family is `NOT_EVALUABLE`. Seeds [7, 13, 29, 42, 101] — five complete paired seeds are required.

Two estimands are reported. **(i)** is one seed checkpoint with paired block-only uncertainty. **(ii)** is the arithmetic across-seed mean under joint seed-and-block resampling — a descriptive five-seed robustness screen, **not** the log loss of averaged probabilities and **not** uncertainty for a newly trained single checkpoint.

## 3. Slices

Block lookup `data/t20s_json`: 545 validation matches map to 47 tournament blocks, 0 unmapped. Slice identity is compared by a row-membership digest, never by matching row, match and block counts.

| slice | rows | matches | blocks | descriptive (<10 blocks) | predicate |
|---|---|---|---|---|---|
| `all` | 124,292 | 545 | 47 | no | every validation row |
| `death` | 20,301 | 514 | 47 | no | is_death_overs == 1 |
| `chase` | 57,994 | 533 | 47 | no | chase_target > 0 |
| `powerplay` | 40,594 | 545 | 47 | no | is_powerplay == 1 |
| `middle` | 63,397 | 541 | 47 | no | is_middle_overs == 1 |
| `innings_1` | 66,252 | 545 | 47 | no | inning_idx == 1 |
| `innings_2` | 57,994 | 533 | 47 | no | inning_idx == 2 |
| `target_P` | 51,452 | 222 | 6 | yes | competition_tier == 3 |
| `target_E` | 55,981 | 243 | 13 | no | the innings_id match-id suffix is in experiments/stage3a/target_E_validation_matches.json |
| `big3` | 1,895 | 8 | 2 | yes | the innings_id match-id suffix is in experiments/stage3a/big3_validation_matches.json |
| `death@target_P` | 8,653 | 213 | 6 | yes | is_death_overs == 1 and competition_tier == 3 |
| `chase@target_P` | 24,240 | 216 | 6 | yes | chase_target > 0 and competition_tier == 3 |
| `death@target_E` | 9,388 | 232 | 13 | no | is_death_overs == 1 and the innings_id match-id suffix is in experiments/stage3a/target_E_validation_matches.json |
| `chase@target_E` | 26,263 | 235 | 13 | no | chase_target > 0 and the innings_id match-id suffix is in experiments/stage3a/target_E_validation_matches.json |

**The decisive line of this table:** `target_P` carries 51,452 rows over 222 matches but only **6 tournament blocks** — the premium-league validation matches come from very few distinct events — so by the registered ten-block rule every member of `family_3a_P` is **descriptive** and the family is **`NOT_EVALUABLE`**. `target_E` reaches **13 blocks** and is evaluable. `big3` has 2 blocks over 8 matches and 1,895 rows: it is a registered descriptive readout only, reported because it was designed in, not because it can support an inference.

`thin_pair` is reported **unavailable**, not invented: thin_pair needs per-row batter and bowler EXPOSURE columns (career or as-of ball counts) and a threshold. The i7 ball frame carries no exposure column: `batter_balls_faced` and `bowler_balls_in_innings` are within-innings counters, not exposure, and the eval kit that carries `train_balls_batting` / `train_balls_bowling` is off for this stage (`eval_kit: none`, `--no-kit`). D10.7 requires the slice to name its exposure columns and threshold or be reported unavailable, so it is reported unavailable rather than invented. To enable it, register `statistics.slice_predicates.thin_pair.{exposure_columns, threshold}` in the config and re-run.

## 4. Family results

Member tables are the **estimand (ii)** readout, `seed_mean_joint`. The per-seed line under each table gives the same family's status at each of the five registered seeds under estimand (i). `p (Holm)` is the step-down adjusted p within that family only.

### `family_3a_cond` — candidate `mlp_pool_tiercond` — **SCREEN_NOT_PASS**

| member | contrast | slice | point | 95% CI | U95 | p (raw) | p (Holm) | rank | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|
| `primary` | mlp_pool_tiercond - mlp_pool on all | `all` | +0.00033 | [-0.00031, +0.00087] | +0.00087 | 0.3010 | 0.3010 | 3 | false | SCREEN_NOT_PASS |
| `death_gate` | mlp_pool_tiercond - mlp_pool on death | `death` | +0.00108 | [-0.00018, +0.00229] | +0.00229 | 0.1420 | 0.2840 | 2 | false | SCREEN_NOT_PASS |
| `chase_gate` | mlp_pool_tiercond - mlp_pool on chase | `chase` | +0.00034 | [-0.00021, +0.00078] | +0.00078 | < 0.001 | 0.0030 | 1 | true | SCREEN_PASS |

Per-seed family status (estimand (i)): seed 7 `SCREEN_NOT_PASS`, seed 13 `SCREEN_NOT_PASS`, seed 29 `SCREEN_NOT_PASS`, seed 42 `SCREEN_NOT_PASS`, seed 101 `SCREEN_NOT_PASS`.

Screen: required members `primary`, `death_gate`, `chase_gate`; member statuses `primary` SCREEN_NOT_PASS, `death_gate` SCREEN_NOT_PASS, `chase_gate` SCREEN_PASS. All direction requirements met: **false**. 2 of 5 per-seed primary directions are favourable; 4 of 5 are required, so this requirement is NOT met

| direction-required member | contrast | seeds | favourable directions | required | met |
|---|---|---|---|---|---|
| `primary` | `mlp_pool_tiercond-mlp_pool@all` | 5 | 2/5 | 4/5 | false |

### `family_3a_P` — candidate `mlp_P_tiercond` — **NOT_EVALUABLE**

| member | contrast | slice | point | 95% CI | U95 | p (raw) | p (Holm) | rank | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|
| `superiority_1` | mlp_P_tiercond - mlp_pool_tiercond on target_P | `target_P` | +0.00442 | [+0.00146, +0.00742] | +0.00742 | 0.0020 | 1.0000 | 1 | false | NOT_EVALUABLE |
| `superiority_2` | mlp_P_tiercond - mlp_rowmatch_P_tiercond on target_P | `target_P` | +0.00351 | [+0.00049, +0.00667] | +0.00667 | 0.0210 | 1.0000 | 2 | false | NOT_EVALUABLE |
| `death_gate` | mlp_P_tiercond - mlp_pool on death within target_P | `death@target_P` | +0.00846 | [+0.00324, +0.01360] | +0.01360 | 0.0180 | 1.0000 | 3 | false | NOT_EVALUABLE |
| `chase_gate` | mlp_P_tiercond - mlp_pool on chase within target_P | `chase@target_P` | +0.00410 | [+0.00100, +0.00787] | +0.00787 | 0.2280 | 1.0000 | 4 | false | NOT_EVALUABLE |

Per-seed family status (estimand (i)): seed 7 `NOT_EVALUABLE`, seed 13 `NOT_EVALUABLE`, seed 29 `NOT_EVALUABLE`, seed 42 `NOT_EVALUABLE`, seed 101 `NOT_EVALUABLE`.

Screen: required members `superiority_1`, `superiority_2`, `death_gate`, `chase_gate`; member statuses `superiority_1` NOT_EVALUABLE, `superiority_2` NOT_EVALUABLE, `death_gate` NOT_EVALUABLE, `chase_gate` NOT_EVALUABLE. All direction requirements met: **false**. 1 of 5 per-seed primary directions are favourable; 4 of 5 are required, so this requirement is NOT met

| direction-required member | contrast | seeds | favourable directions | required | met |
|---|---|---|---|---|---|
| `superiority_1` | `mlp_P_tiercond-mlp_pool_tiercond@target_P` | 5 | 1/5 | 4/5 | false |
| `superiority_2` | `mlp_P_tiercond-mlp_rowmatch_P_tiercond@target_P` | 5 | 1/5 | 4/5 | false |

### `family_3a_E` — candidate `mlp_E_tiercond` — **SCREEN_NOT_PASS**

| member | contrast | slice | point | 95% CI | U95 | p (raw) | p (Holm) | rank | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|
| `superiority_1` | mlp_E_tiercond - mlp_pool_tiercond on target_E | `target_E` | +0.00190 | [-0.00039, +0.00554] | +0.00554 | 0.2130 | 0.8520 | 1 | false | SCREEN_NOT_PASS |
| `superiority_2` | mlp_E_tiercond - mlp_rowmatch_E_tiercond on target_E | `target_E` | +0.00144 | [-0.00079, +0.00487] | +0.00487 | 0.3960 | 1.0000 | 2 | false | SCREEN_NOT_PASS |
| `death_gate` | mlp_E_tiercond - mlp_pool on death within target_E | `death@target_E` | +0.00430 | [-0.00032, +0.01027] | +0.01027 | 0.4170 | 1.0000 | 3 | false | SCREEN_NOT_PASS |
| `chase_gate` | mlp_E_tiercond - mlp_pool on chase within target_E | `chase@target_E` | +0.00175 | [-0.00032, +0.00516] | +0.00516 | 0.7710 | 1.0000 | 4 | false | SCREEN_NOT_PASS |

Per-seed family status (estimand (i)): seed 7 `SCREEN_NOT_PASS`, seed 13 `SCREEN_NOT_PASS`, seed 29 `SCREEN_NOT_PASS`, seed 42 `SCREEN_NOT_PASS`, seed 101 `SCREEN_NOT_PASS`.

Screen: required members `superiority_1`, `superiority_2`, `death_gate`, `chase_gate`; member statuses `superiority_1` SCREEN_NOT_PASS, `superiority_2` SCREEN_NOT_PASS, `death_gate` SCREEN_NOT_PASS, `chase_gate` SCREEN_NOT_PASS. All direction requirements met: **false**. 2 of 5 per-seed primary directions are favourable; 4 of 5 are required, so this requirement is NOT met

| direction-required member | contrast | seeds | favourable directions | required | met |
|---|---|---|---|---|---|
| `superiority_1` | `mlp_E_tiercond-mlp_pool_tiercond@target_E` | 5 | 2/5 | 4/5 | false |
| `superiority_2` | `mlp_E_tiercond-mlp_rowmatch_E_tiercond@target_E` | 5 | 2/5 | 4/5 | false |

## 5. Seed spread, every family member and the exploratory readouts

| contrast | slice | seed 7 | seed 13 | seed 29 | seed 42 | seed 101 | seed range | favourable directions | mean point (ii) | mean 95% interval (ii) | blocks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `mlp_pool_tiercond − mlp_pool` | `all` | +0.00025 | +0.00066 | -0.00007 | -0.00003 | +0.00083 | +0.00091 | 2/5 | +0.00033 | [-0.00031, +0.00087] | 47 |
| `mlp_pool_tiercond − mlp_pool` | `death` | -0.00029 | +0.00271 | +0.00025 | +0.00110 | +0.00165 | +0.00300 | 1/5 | +0.00108 | [-0.00018, +0.00229] | 47 |
| `mlp_pool_tiercond − mlp_pool` | `chase` | +0.00054 | +0.00059 | +0.00001 | +0.00010 | +0.00047 | +0.00059 | 0/5 | +0.00034 | [-0.00021, +0.00078] | 47 |
| `mlp_P_tiercond − mlp_pool_tiercond` | `target_P` | -0.00056 | +0.00819 | +0.00752 | +0.00469 | +0.00226 | +0.00875 | 1/5 | +0.00442 | [+0.00146, +0.00742] | 6 |
| `mlp_P_tiercond − mlp_rowmatch_P_tiercond` | `target_P` | -0.00132 | +0.00769 | +0.00631 | +0.00380 | +0.00108 | +0.00901 | 1/5 | +0.00351 | [+0.00049, +0.00667] | 6 |
| `mlp_P_tiercond − mlp_pool` | `death@target_P` | -0.00049 | +0.01572 | +0.01302 | +0.00815 | +0.00592 | +0.01621 | 1/5 | +0.00846 | [+0.00324, +0.01360] | 6 |
| `mlp_P_tiercond − mlp_pool` | `chase@target_P` | -0.00068 | +0.00791 | +0.00746 | +0.00408 | +0.00174 | +0.00859 | 1/5 | +0.00410 | [+0.00100, +0.00787] | 6 |
| `mlp_E_tiercond − mlp_pool_tiercond` | `target_E` | -0.00056 | +0.00083 | +0.00864 | +0.00105 | -0.00048 | +0.00920 | 2/5 | +0.00190 | [-0.00039, +0.00554] | 13 |
| `mlp_E_tiercond − mlp_rowmatch_E_tiercond` | `target_E` | -0.00095 | +0.00027 | +0.00782 | +0.00093 | -0.00086 | +0.00877 | 2/5 | +0.00144 | [-0.00079, +0.00487] | 13 |
| `mlp_E_tiercond − mlp_pool` | `death@target_E` | -0.00106 | +0.00397 | +0.01534 | +0.00306 | +0.00020 | +0.01640 | 1/5 | +0.00430 | [-0.00032, +0.01027] | 13 |
| `mlp_E_tiercond − mlp_pool` | `chase@target_E` | -0.00016 | +0.00134 | +0.00746 | +0.00089 | -0.00080 | +0.00825 | 2/5 | +0.00175 | [-0.00032, +0.00516] | 13 |
| *— exploratory rows below: no family, no screen —* | | | | | | | | | | | |
| `mlp_P_tiercond − mlp_pool` | `all` | +0.09099 | +0.10231 | +0.06836 | +0.03573 | +0.04712 | +0.06658 | 0/5 | +0.06890 | [+0.03648, +0.11421] | 47 |
| `mlp_P_tiercond − mlp_pool` | `target_E` | +0.01991 | +0.03232 | +0.01046 | +0.00665 | +0.01039 | +0.02567 | 0/5 | +0.01594 | [+0.00639, +0.04618] | 13 |
| `mlp_P_tiercond − mlp_pool` | `big3` | +0.25830 | +0.31416 | +0.04649 | +0.03707 | +0.10804 | +0.27709 | 0/5 | +0.15281 | [+0.05844, +0.25060] | 2 |
| `mlp_E_tiercond − mlp_pool` | `all` | +0.01249 | +0.02209 | +0.05712 | +0.00849 | +0.01170 | +0.04864 | 0/5 | +0.02238 | [+0.00775, +0.04586] | 47 |
| `mlp_E_tiercond − mlp_pool` | `target_P` | -0.00090 | +0.00156 | +0.00886 | +0.00049 | -0.00010 | +0.00976 | 2/5 | +0.00198 | [-0.00041, +0.00567] | 6 |
| `mlp_E_tiercond − mlp_pool` | `big3` | +0.00476 | +0.00315 | +0.00803 | +0.00776 | +0.00199 | +0.00604 | 0/5 | +0.00514 | [+0.00254, +0.00777] | 2 |
| `mlp_rowmatch_P_tiercond − mlp_pool` | `all` | +0.00164 | +0.00201 | +0.00208 | +0.00134 | +0.00182 | +0.00074 | 0/5 | +0.00178 | [+0.00101, +0.00246] | 47 |
| `mlp_rowmatch_P_tiercond − mlp_pool` | `target_P` | +0.00066 | +0.00130 | +0.00141 | +0.00072 | +0.00155 | +0.00088 | 0/5 | +0.00113 | [+0.00015, +0.00186] | 6 |
| `mlp_rowmatch_E_tiercond − mlp_pool` | `all` | +0.00118 | +0.00150 | +0.00117 | +0.00087 | +0.00107 | +0.00063 | 0/5 | +0.00116 | [+0.00052, +0.00171] | 47 |
| `mlp_rowmatch_E_tiercond − mlp_pool` | `target_E` | +0.00036 | +0.00127 | +0.00087 | +0.00001 | +0.00071 | +0.00126 | 0/5 | +0.00064 | [-0.00018, +0.00128] | 13 |
| `mlp_pool_tiercond − mlp_pool` | `target_P` | -0.00009 | +0.00080 | +0.00019 | -0.00017 | +0.00036 | +0.00098 | 2/5 | +0.00022 | [-0.00028, +0.00068] | 6 |
| `mlp_pool_tiercond − mlp_pool` | `target_E` | -0.00003 | +0.00071 | +0.00005 | -0.00011 | +0.00033 | +0.00082 | 2/5 | +0.00019 | [-0.00028, +0.00067] | 13 |
| `mlp_pool_tiercond − mlp_pool` | `big3` | +0.00027 | -0.00072 | -0.00082 | +0.00144 | +0.00014 | +0.00226 | 2/5 | +0.00006 | [-0.00098, +0.00096] | 2 |

Seed ranges are empirical spreads over five seeds, not confidence intervals. No seed is selected and no CI endpoint is averaged. The favourable-direction count is the **zero-threshold** count (per-seed points below 0), which on a non-inferiority gate row is not the same as the count below that row's registered `+0.002` margin. Every row below the divider is exploratory: no family names it, no screen reads it, and `big3` rows in particular sit on 2 blocks.

## 6. Absolute validation log loss, and what each arm actually saw

Numbers of record: full-precision per-seed validation log loss from `runs/<config>/summary.yaml`; never rounded report values or LL reconstructed from saved probabilities.

| config | seeds | mean val LL | min | max |
|---|---|---|---|---|
| `mlp_pool` | 5 | 1.437423 | 1.437122 (seed 13) | 1.437802 (seed 42) |
| `mlp_pool_tiercond` | 5 | 1.437751 | 1.437259 (seed 29) | 1.438270 (seed 101) |
| `mlp_P_tiercond` | 5 | 1.506323 | 1.473536 (seed 42) | 1.539432 (seed 13) |
| `mlp_rowmatch_P_tiercond` | 5 | 1.439202 | 1.439057 (seed 7) | 1.439415 (seed 29) |
| `mlp_E_tiercond` | 5 | 1.459801 | 1.446288 (seed 42) | 1.494455 (seed 29) |
| `mlp_rowmatch_E_tiercond` | 5 | 1.438581 | 1.438505 (seed 101) | 1.438671 (seed 42) |

Exposure, per run, from each `metrics.json` `training_schedule`. **These are full-training totals — tokens seen over the whole 3,840-step budget, not exposure at the selected checkpoint**, which for most arms is reached earlier (`steps to best` below). `tokens seen in P` / `tokens seen in E` are the tokens drawn from matches on the two frozen report lists. `untrained tiers` are tier codes present in validation but absent from that arm's training rows, recorded verbatim from `training_schedule.untrained_tiers`; their embedding and bias slots stay at initialisation and are still exercised at validation time.

| config | seed | tokens seen | tokens seen in P | tokens seen in E | steps to best / total | best epoch | untrained tiers |
|---|---|---|---|---|---|---|---|
| `mlp_pool` | 7 | 56,309,130 | 17,740,260 | 24,365,070 | 3072 / 3840 | 23 | none |
| `mlp_pool` | 13 | 56,309,130 | 17,740,260 | 24,365,070 | 3200 / 3840 | 24 | none |
| `mlp_pool` | 29 | 56,309,130 | 17,740,260 | 24,365,070 | 3840 / 3840 | 29 | none |
| `mlp_pool` | 42 | 56,309,130 | 17,740,260 | 24,365,070 | 3712 / 3840 | 28 | none |
| `mlp_pool` | 101 | 56,309,130 | 17,740,260 | 24,365,070 | 3712 / 3840 | 28 | none |
| `mlp_pool_tiercond` | 7 | 56,309,130 | 17,740,260 | 24,365,070 | 3456 / 3840 | 26 | none |
| `mlp_pool_tiercond` | 13 | 56,309,130 | 17,740,260 | 24,365,070 | 3712 / 3840 | 28 | none |
| `mlp_pool_tiercond` | 29 | 56,309,130 | 17,740,260 | 24,365,070 | 3840 / 3840 | 29 | none |
| `mlp_pool_tiercond` | 42 | 56,309,130 | 17,740,260 | 24,365,070 | 3072 / 3840 | 23 | none |
| `mlp_pool_tiercond` | 101 | 56,309,130 | 17,740,260 | 24,365,070 | 3712 / 3840 | 28 | none |
| `mlp_P_tiercond` | 7 | 56,768,832 | 56,768,832 | 56,768,832 | 3072 / 3840 | 76 | 1, 2, 4 |
| `mlp_P_tiercond` | 13 | 56,768,832 | 56,768,832 | 56,768,832 | 256 / 3840 | 6 | 1, 2, 4 |
| `mlp_P_tiercond` | 29 | 56,768,832 | 56,768,832 | 56,768,832 | 256 / 3840 | 6 | 1, 2, 4 |
| `mlp_P_tiercond` | 42 | 56,768,832 | 56,768,832 | 56,768,832 | 512 / 3840 | 12 | 1, 2, 4 |
| `mlp_P_tiercond` | 101 | 56,768,832 | 56,768,832 | 56,768,832 | 1024 / 3840 | 25 | 1, 2, 4 |
| `mlp_rowmatch_P_tiercond` | 7 | 55,413,767 | 17,799,426 | 24,623,907 | 2560 / 3840 | 62 | none |
| `mlp_rowmatch_P_tiercond` | 13 | 55,414,694 | 17,799,928 | 24,624,766 | 3072 / 3840 | 74 | none |
| `mlp_rowmatch_P_tiercond` | 29 | 55,414,003 | 17,801,999 | 24,626,377 | 3200 / 3840 | 78 | none |
| `mlp_rowmatch_P_tiercond` | 42 | 55,413,770 | 17,800,304 | 24,625,549 | 3072 / 3840 | 74 | none |
| `mlp_rowmatch_P_tiercond` | 101 | 55,413,877 | 17,800,956 | 24,626,967 | 2560 / 3840 | 62 | none |
| `mlp_E_tiercond` | 7 | 56,709,061 | 41,290,185 | 56,709,061 | 3840 / 3840 | 69 | 1 |
| `mlp_E_tiercond` | 13 | 56,708,364 | 41,286,108 | 56,708,364 | 1664 / 3840 | 30 | 1 |
| `mlp_E_tiercond` | 29 | 56,708,836 | 41,287,249 | 56,708,836 | 256 / 3840 | 4 | 1 |
| `mlp_E_tiercond` | 42 | 56,708,928 | 41,289,399 | 56,708,928 | 2176 / 3840 | 39 | 1 |
| `mlp_E_tiercond` | 101 | 56,709,331 | 41,290,761 | 56,709,331 | 2816 / 3840 | 51 | 1 |
| `mlp_rowmatch_E_tiercond` | 7 | 55,703,297 | 17,616,102 | 24,271,387 | 3840 / 3840 | 68 | none |
| `mlp_rowmatch_E_tiercond` | 13 | 55,703,889 | 17,621,787 | 24,275,792 | 2432 / 3840 | 43 | none |
| `mlp_rowmatch_E_tiercond` | 29 | 55,703,583 | 17,618,003 | 24,271,328 | 3328 / 3840 | 59 | none |
| `mlp_rowmatch_E_tiercond` | 42 | 55,704,326 | 17,621,322 | 24,277,248 | 3584 / 3840 | 63 | none |
| `mlp_rowmatch_E_tiercond` | 101 | 55,702,777 | 17,617,978 | 24,272,398 | 3072 / 3840 | 54 | none |

All 6 arms run exactly 3,840 steps, so the arms trained on fewer rows see **more epochs** over their smaller sets. Registered residual confounding, verbatim: equal steps, equal tokens and equal target-tier exposure cannot all hold at the pool's proportions. EQUAL STEPS is registered; tokens seen and target-tier tokens seen therefore DIFFER across arms, a smaller training set sees more epochs, and every arm reports total tokens seen, target-tier tokens seen and steps-to-best so the difference is visible.

## 7. In plain language

**What was tested.** Six small models were trained on historical T20 deliveries and then asked, one ball at a time, to put a probability on each of the six outcomes of the *next* delivery of 124,292 held-out balls. They differ in **which matches they were allowed to learn from**, and in whether they are told which tier a match belongs to. This is teacher-forced ball prediction: nobody simulated a match, no market price appears anywhere in this block, and a better log loss here implies nothing about a score distribution, a win probability or a betting edge.

**The registered reading, in one paragraph.** Target-only point estimates are **adverse** on their own targets. `family_3a_P` is **descriptive and `NOT_EVALUABLE`** — 6 tournament blocks against the ten the rule requires. `family_3a_E` and `family_3a_cond` **do not pass their registered screens**. **Negative transfer was not detected; neither its absence nor positive transfer is established.**

**The target-side numbers.** Training only on premium leagues moves the point estimate on the premium-league validation rows by +0.0044 log loss against the pooled conditioned model ([+0.0015, +0.0074], descriptive on 6 blocks), and by +0.0035 [+0.0005, +0.0067] against the row-matched pooled control. On the elite target E the same two readings are +0.0019 [-0.0004, +0.0055] on 13 blocks, with 2/5 seeds in the favourable direction against the 4/5 the screen requires, and +0.0014 [-0.0008, +0.0049] against the row-matched control. Every point estimate is adverse, and no superiority member of either family produced a favourable interval.

**Training exposure.** Over the full 3,840-step budget `mlp_P_tiercond` drew 56,768,832 tokens from premium-league matches against the pooled control's 17,740,260 — **3.2×** — and `mlp_E_tiercond` drew 2.3× the pooled arm's E tokens. These are full-training totals, not exposure at the selected checkpoint. More target exposure did not come with a favourable target-side estimate.

**On all validation rows.** +0.0686 [+0.0365, +0.1137] for the P arm and +0.0220 [+0.0076, +0.0455] for the E arm against the same conditioned pooled model. That reading is exploratory — the screens amend the all-row condition away — and the untrained tier slots those arms carry into rows they never trained on are **one possible contributor**, not an established cause.

**Tier conditioning on its own.** `mlp_pool_tiercond − mlp_pool` on all rows is +0.0003 [-0.0003, +0.0009], with 2/5 seeds in the favourable direction against the 4/5 required, so `family_3a_cond` does not pass its screen. The interval straddles zero: this is **unresolved evidence at this resolution**, not a demonstration that conditioning is inert.

**What is not established.** "Not detected" is not proof of absence, and an adverse point estimate on a screen that does not pass is not a demonstration of positive transfer. The P family cannot carry an inference at all on 6 blocks; on target E the superiority intervals straddle or sit near zero and read unresolved. Nothing here establishes that the pooled model is better on the target, only that restricting training to the target was **not shown** to help.

**Two possible explanations for the adverse direction, neither established.** (a) The per-seed points are not uniform: on target P they run from -0.00056 to +0.00819, the two most adverse being seed 13 (+0.00819) and seed 29 (+0.00752); on target E from -0.00056 to +0.00864, the two most adverse being seed 29 (+0.00864) and seed 42 (+0.00105). That pattern **would be consistent with** overfitting under the equal-step budget, since an arm trained on a fraction of the rows takes roughly proportionally more epochs over them — but no overfitting diagnostic was run and this is a hypothesis. (b) Checkpoints for **every** arm, target-only arms included, were selected on **all** validation rows, which can work against target-only performance by construction; that cost is a registered part of the shared-artifact contract. Both caveats stand, and an **equal-epoch or early-stopped replication is the registered follow-up** that would separate them from the transfer question.

## 8. Registered deviations, asymmetries and limitations (restated, none dropped)

### Deviations (config `deviations`)

* **`outcome_informed_registration`** — the two target definitions, the negative-transfer hypothesis and the conditioning arm follow from the stage 2 five-seed readout, which was already read when this config was written. *Reason as registered:* it is a NEW screening hypothesis, frozen in this file and in the design draft before any night-3 performance read, so it is pre-registered rather than confirmatory. It is not a confirmation of stage 2 and must never be reported as one.
* **`target_E_validation_list_repointed`** — the `target_E` VALIDATION slice is a separate frozen list, experiments/stage3a/target_E_validation_matches.json, and NOT the E training list experiments/stage3a/target_E_matches.json that `mlp_E_tiercond` trains on. *Reason as registered:* the two lists are disjoint by construction: the freeze manifest asserts that no match id in any frozen TRAINING list appears in the validation parquet (0 leaked for target_E), so reading the E validation slice off the training list would select ZERO rows and silently make every family_3a_E member NOT_EVALUABLE. The E RULE is one rule applied to two splits; the two files are its two halves. Found while writing this config; the freeze script was extended to emit the validation half, and night3_acceptance N3 checks that both halves exist, that the config pins the validation half for the slice and the training half for training, and that their sha256s match the regenerated manifest.
* **`big3_thin_8_matches`** — the `big3` readout has 8 validation matches, 16 innings and 1,895 rows — far below the ten-block rule. *Reason as registered:* it is registered as a descriptive readout only, with fixed contrasts, row and block counts reported, no family and no screen. It is reported because it was designed in, not because it can support an inference.
* **`all_row_selection_for_target_arms`** — the target-only arms' checkpoints are selected on ALL validation rows, including rows from matches outside their target. *Reason as registered:* one shared artifact per (configuration, seed) is what makes the two families share controls and keeps the count at 30 runs. The cost — the selection can work against target-only performance — is disclosed and is part of the shared-artifact contract, not a finding.
* **`equal_steps_not_equal_tokens`** — all six arms train for exactly 3,840 steps, so the arms trained on fewer rows see MORE epochs, and tokens seen and target-tier tokens seen differ across arms. *Reason as registered:* equal steps, equal tokens and equal target-tier exposure cannot all hold at the pool's proportions. Equal steps is the registered control; the residual confounding is stated and every arm reports total tokens seen, target-tier tokens seen and steps-to-best.
* **`ref_train_probs_in_sample`** — RESERVED for Block E rung 4b, whose frozen reference logits are fitted on the training rows they are later read on. *Reason as registered:* registered here so the night's deviation ids are stable across its blocks. It bears on NO Block B arm: no arm of this config reads a frozen reference, and this entry must not be cited in a Block B reading.

### Known asymmetries (config `known_asymmetries`)

* **`tiercond_arms_see_the_tier_column`** — the three conditioned arms see `competition_tier`, which `mlp_pool` never sees. So any contrast whose reference is `mlp_pool` — both gates of every family, and every exploratory arm-minus-control reading — mixes the conditioning with whatever else the arm changed, and only `mlp_pool_tiercond - mlp_pool` isolates the conditioning. The two superiority members avoid this by taking conditioned references.
* **`untrained_tier_slots_in_the_target_arms`** — `mlp_P_tiercond` sees only tier 3, so its tier embedding rows and output bias rows for tiers 0, 1, 2 and 4 stay at initialisation and are recorded as untrained; `mlp_E_tiercond` sees tier 3 plus whatever tiers its elite internationals carry. Those slots are exercised at validation time on every non-target row, which is one mechanism by which a target-only arm can be worse on `all` for reasons unrelated to transfer.
* **`rowmatch_controls_are_row_matched_not_innings_matched`** — the row-matched controls match the target's ROW count (overshoot 240 and 81 rows, at most one match) by adding whole matches, so innings counts and per-innings length distributions differ slightly from the target's.
* **`target_P_and_target_E_overlap`** — target E contains all of target P, and the `target_P` and `target_E` validation slices overlap heavily. The two families share `mlp_pool` and `mlp_pool_tiercond` as well, so their screens are correlated, not replications.
* **`token_mlp_has_no_attention_layer`** — every arm here is token_mlp, so none has an attention layer and none of this block's readings bears on any sequence mechanism. Parameter counts differ between the conditioned and unconditioned arms by the tier embedding and bias, reported per run in arm_params.n_parameters.

**Correction to config-sourced wording above.** `untrained_tier_slots_in_the_target_arms` says "tiers 0, 1, 2 and 4" for `mlp_P_tiercond`. The value actually recorded by every seed of that run is `training_schedule.untrained_tiers` = [1, 2, 4]; tier 0 is not among them. The same config text says "the three conditioned arms"; five of the six configurations are tier-conditioned (every arm except `mlp_pool`), and the asymmetry it describes applies to all five. The asymmetry itself is unchanged; only the tier list and the arm count are corrected, and the table in § 6 prints the recorded values.

### Known limitations (config `known_limitations`)

* SEED-MACHINE CONFOUNDING, inherited from night 1 and not modelled: seeds 7, 29 and 42 run on the laptop and 13, 101 on the mini, and MPS kernels are not bit-reproducible across machines. The split is BY SEED, so every arm carries the same three-laptop / two-mini mix and no arm is advantaged relative to another; the machine is still a component of each per-seed number and is reported, never adjusted for.
* family_3a_P and family_3a_E are CORRELATED screens over overlapping rows with two shared controls. Claims are FAMILY-LOCAL only: no aggregate, no "either target passes", no comparison of the two families' outcomes to each other, and no inference from one passing where the other does not.
* checkpoint selection uses the same validation split the contrasts are computed on, so every number of this block is screening evidence.
* MPS kernels are not bit-reproducible across runs or machines, so a rerun of any checkpoint may differ in the low decimals; every checkpoint records mps_bit_reproducible false.
* the global outcome prior in the stats cache is not as-of-date, and venue history in the frame is not recency-weighted. Both are inherited frame limitations shared by all six arms, so they are level effects rather than per-arm advantages.
* `competition_tier` is a pre-match EVENT-NAME rule, not a market or quality measurement. No market designation attaches to a tier anywhere in this block, and "premium" and "elite" are the rule's labels, not findings.

### Restated in this report's own terms

* **Seed–machine confounding.** Machine is recorded per run and is confounded with seed. Read from the run provenance:

| seed | machine | chip | host |
|---|---|---|---|
| 7 | laptop | Apple M5 Pro | Aryamans-MacBook-Pro.local |
| 13 | mini | Apple M4 | Aryamans-Mac-mini.local |
| 29 | laptop | Apple M5 Pro | Aryamans-MacBook-Pro.local |
| 42 | laptop | Apple M5 Pro | Aryamans-MacBook-Pro.local |
| 101 | mini | Apple M4 | Aryamans-Mac-mini.local |

  The split is **by seed**, so every arm carries the same mix and no arm is advantaged relative to another; every registered contrast is within-machine at each seed, so an additive machine effect cancels inside it. Arm-by-machine interaction remains inseparable from seed variation and **no machine term is fitted**.
* **The P and E screens are correlated, not replications.** Target E contains all of target P, the two validation slices overlap heavily, and the two families share `mlp_pool` and `mlp_pool_tiercond`. Claims are **family-local only**: no aggregate, no "either target passes", no comparison of the two families' outcomes to each other.
* **`big3` is too thin to read.** 2 blocks, 8 matches, 1,895 rows. It is reported because it was designed in.
* **Selection optimism throughout.** Checkpoint selection used the same validation split every contrast is computed on.

## 9. What happens next

**Nothing advances.** `advances: []`, `cohort_status: DEFERRED_UNOPENED`, `reads_performed: none`. No cohort feature, prediction or base-logit read happened, no artifact of record changed, no test split or market price was touched, and no verdict was logged — that call is the user's.

The next separately frozen batch is **Stage 4** (rungs 4d and 4b) and **C114**. Each gets its own config, its own freeze and its own acceptance file; nothing from tonight is carried into them as an established result.

Stage 3a's own follow-up is **backlog, not queued**: an **equal-epoch or early-stopped replication** of the 6 arms, the design change that would separate the transfer question from the equal-step budget and from all-validation checkpoint selection. Until that runs, the registered reading stands exactly as stated and no further: **target-only point estimates are adverse; `family_3a_P` is descriptive and `NOT_EVALUABLE` on the block rule; `family_3a_E` and `family_3a_cond` do not pass their registered screens; negative transfer was not detected, and neither its absence nor positive transfer is established.**
