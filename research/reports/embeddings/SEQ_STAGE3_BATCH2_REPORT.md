# Sequence track — stage 3 BATCH 2 report (rung 4d, block C114, rung 4b; validation only)

Generated (declared timestamp, the only value that moves on a re-render): **2026-09-12T22:54:47.649135+00:00** by `scripts/sequence_track/render_batch2_report.py` from the files of record. Every number in every table is read from a file; the narrative passages are authored text in the generator.

* statistics, rung 4d: `eval_out/seq_stage3_batch2/4d_stats.json` (sha256 `ea2a4046ca3b…`), written 2026-09-12T22:48:06.357502+00:00 by `scripts/sequence_track/stage2_stats.py`
* statistics, block C114: `eval_out/seq_stage3_batch2/c114_stats.json` (sha256 `26e3b042b256…`), written 2026-09-12T22:48:11.502772+00:00 by `scripts/sequence_track/stage2_stats.py`
* configs: `experiments/configs/seq_stage3_batch2_4d_v1.yaml` (sha256 `ced2bec38270…`), `experiments/configs/seq_stage3_batch2_c114_v1.yaml` (sha256 `fd6ffa790a85…`), `experiments/configs/seq_stage3_batch2_4b_v1.yaml` (sha256 `9def95ee3e04…`)
* runs: `models/embeddings/seq_stage3/batch2/runs/<config>/seed_<s>/` — `summary.yaml`, `metrics.json`, `run_record.json`
* frozen references: `models/embeddings/stage4/refs/references.json` (sha256 `cb8688335193…`) — validation log loss `ref_eb_ctx` 1.4500, `ref_raw_ctx` 1.4534, `ref_lin_50` 1.4433
* freeze commit `86e049e`; launch record commit `a003d5d`

**Status: validation-only, five-seed, outcome-informed screening. Rung 4d — `cohort_status: DEFERRED_UNOPENED`, `cohort_scored: false`, `advances: []`, `provisional: true`. Block C114 — `cohort_status: DEFERRED_UNOPENED`, `cohort_scored: false`, `advances: []`, `provisional: true`. No arm advances. No market claim and no betting layer anywhere. No LANDED verdict — this evidence is provisional (validation-only, checkpoint-selected on the same split it is scored on, with the untouched cohort unopened), and provisional evidence can never be LANDED (CLAUDE.md invariant 9). The verdict is the user's decision and none was logged: `research/log_verdict.py` was not called.**

Registered evidence status, verbatim from the rung-4d statistics file: VALIDATION-ONLY, PROVISIONAL. Checkpoint selection is on the same validation split the contrasts are computed on, the untouched cohort is NOT opened, and no test split, market price or betting layer is touched anywhere. Nothing here can be LANDED (CLAUDE.md invariant 9), nothing advances, and no artifact of record changes.

Registered evidence status, verbatim from the C114 statistics file: VALIDATION-ONLY, PROVISIONAL. Checkpoint selection is on the same validation split the contrasts are computed on, the untouched cohort is NOT opened, and no test split, market price or betting layer is touched anywhere. Nothing here can be LANDED (CLAUDE.md invariant 9), nothing advances, and no artifact of record changes. In particular a favourable `full_114 - full_50` says nothing about the PRODUCTION ball model, which is an XGBoost booster on the same columns and is not an arm here.

Statuses this batch can emit: `SCREEN_PASS`, `SCREEN_NOT_PASS`, `NOT_EVALUABLE`. There is no advancement status. Rung 4b emits **no status at all** — its family is deferred by registration (§ 6).

## 1. The two questions

Two separately registered questions on two separate sets of arms, in one batch. They share the frame, the schedule and the seeds and nothing else; **no arm of one question may be tabled against an arm of the other**, and no batch-2 arm may be tabled against a night-3 Block B arm (both configs register that deviation — the schedules differ).

Sign convention, verbatim from both configs: candidate minus reference, in validation log loss, negative favourable.

### 1.1 Rung 4d — do as-of exposure counts, posterior spread and recency help a token MLP?

Registered purpose, verbatim: Does telling a token model HOW MUCH evidence its EB rate features rest on — and how spread out and how recent that evidence is — buy anything over the 50-feature contract alone? Two token-MLP arms on the same frame and the same epoch schedule: `mlp_counts` adds only the two as-of ball counts (batter and bowler), and `mlp_spread_recency` adds those two plus the twelve Dirichlet spread columns and the two recency counts. The contrast is spread PLUS recency against counts alone; it does NOT isolate spread.

The registered contrast is `mlp_spread_recency − mlp_counts`: **both** arms see the two as-of ball counts, so the contrast isolates the twelve Dirichlet spread columns plus the two recency counts and never confounds them with "the model was told how much data there is".

| config id | arm | role | wiring | history input | inputs | parameters | reference(s) |
|---|---|---|---|---|---|---|---|
| `mlp_counts` | mlp | control | token_mlp | none | default 50-feature contract; + 2 sidecar columns | 272,262 | — (shared control) |
| `mlp_spread_recency` | mlp | candidate | token_mlp | none | default 50-feature contract; + 16 sidecar columns | 274,054 | `mlp_counts` |

Schedule, read from the config's `training` block: `epochs` 30 with `patience` 3 (live early stopping — batch 2 sets no step budget), `d_model` 128, 2 layers, 4 heads, batch 128, lr 0.0003, device `mps`, seeds [7, 13, 29, 42, 101], on frame `data/xgb_data_i7` with cache role `stats_cache_i7`. Selection rule, verbatim: validation log loss = ball-weighted mean negative log-likelihood over ALL validation rows, for both arms.

Both arms are additionally reported **beside the three frozen stage-4 references**, whose validation log losses are read from `models/embeddings/stage4/refs/references.json`:

| reference | `references.json` key | features | validation log loss (as recorded, 4 dp) |
|---|---|---|---|
| `ref_eb_ctx` | `eb_ctx` | 46 | 1.4500 |
| `ref_raw_ctx` | `raw_rate_ctx` | 22 | 1.4534 |
| `ref_lin_50` | `ref_lin_50` | 50 | 1.4433 |

That comparison is a **level readout only**: the references are deterministic fitted npz artifacts, not runs, so they carry no seeds, no interval and no Holm slot, and no family member may take one as its reference (`deviations.fixed_reference_is_not_a_member`).

### 1.2 Block C114 — does sequence access buy anything on the production contract, and do the extra 64 columns add on top?

Registered purpose, verbatim: Two questions on one set of three arms. (1) Does sequence access buy anything ON THE PRODUCTION 114-COLUMN CONTRACT, where every earlier stage asked it on the 50-feature contract — `full_114 - mlp_114`. (2) Does the 114-column contract buy anything over the 50-feature contract for the SAME standard-wiring arm — `full_114 - full_50`. The three arms are fresh; none reuses a stage 1, stage 2 or night-3 checkpoint.

Two questions, one set of three arms. **(1)** `full_114 − mlp_114` — the transformer (standard causal attention) against the token MLP on the same 114 columns: sequence access with the inputs held exactly fixed. **(2)** `full_114 − full_50` — the same standard-wiring arm on the production 114-column contract against the 50-feature contract: the contract at fixed wiring. Both families' non-inferiority gates are taken against the shared control `mlp_114`, so every gate asks the same question on `death` and `chase`.

| config id | arm | role | wiring | history input | inputs | parameters | reference(s) |
|---|---|---|---|---|---|---|---|
| `full_114` | full | candidate | standard | innings_previous | `v7_114` contract | 306,950 | `mlp_114`, `full_50` |
| `mlp_114` | mlp | control | token_mlp | none | `v7_114` contract | 280,198 | — (shared control) |
| `full_50` | full | candidate | standard | innings_previous | default 50-feature contract | 298,758 | `mlp_114` |

Schedule, read from the config's `training` block: `epochs` 30 with `patience` 3 (live early stopping — batch 2 sets no step budget), `d_model` 128, 2 layers, 4 heads, batch 128, lr 0.0003, device `mps`, seeds [7, 13, 29, 42, 101], on frame `data/xgb_data_i7` with cache role `stats_cache_i7`. Selection rule, verbatim: validation log loss = ball-weighted mean negative log-likelihood over ALL validation rows, for all three arms.

The C114 arms are read against the same three frozen references as a level readout, on the same terms.

## 2. The decision rule, as registered before any result was read

Read from each config's `statistics` block: the families, their members, their thresholds, each family's `screen.require` and each family's `all_row_condition`.

### 2.1 Rung 4d

Shared control: `mlp_counts` — forced to `role: control` and a candidate in no family. Multiplicity: holm, scope within a family, across its members.

**`family_4d`** — registered under family key `candidate: mlp_spread_recency`, `all_row_condition: member`, 3 members, Holm **m = 3**.

| member | primary | kind | contrast | slice | threshold |
|---|---|---|---|---|---|
| `primary` | yes | superiority | `mlp_spread_recency − mlp_counts` | `all` | +0.000 |
| `death_gate` | no | non_inferiority | `mlp_spread_recency − mlp_counts` | `death` | +0.002 |
| `chase_gate` | no | non_inferiority | `mlp_spread_recency − mlp_counts` | `chase` | +0.002 |

`screen.require`: `primary`, `death_gate`, `chase_gate`. Registered note, verbatim: decision-bearing and registered once. SCREEN_PASS requires all three members to reject favourably (primary U95 < 0, each gate strictly U95 < +0.002) under Holm, with at least ten tournament blocks on each member's slice, five complete paired seeds, and at least 4 of 5 favourable per-seed directions on `primary`.

Holm adjustment as registered: step-down over the family's sorted raw p-values; with m members the adjusted value at rank r is min(1, max over j <= r of (m + 1 - j) * p_(j)), which makes the adjusted values monotone in rank. **m is that family's own member count**, and Holm is never pooled across families. Missing member rule, verbatim: an unavailable member (a run that did not finish, a slice with no rows, a slice below ten blocks) takes a non-rejecting placeholder (adjusted p = 1), still occupies its slot, and the family CANNOT pass.

Uncertainty: contract `tournament_time_block_v1`, 2,000 replicates at rng seed 29, ball_weighted, blocks from `scripts/sim_eval/eval_statistics.load_competition_clusters` over `data/t20s_json`, `min_blocks` 10 and below that descriptive (CLAUDE.md invariant 7). Non-inferiority margin +0.002 log loss on ['death', 'chase'], against `mlp_counts`; a candidate is non-inferior on a gate slice when U95, the 97.5th percentile of the paired harm draws of (candidate - mlp_counts), is STRICTLY below +0.002 log loss. U95 = 0.002 exactly does not pass.

Raw p convention, verbatim from the statistics file: p_raw = min(1, 2 * min[P(d* - t <= 0), P(d* - t >= 0)]) over the bootstrap delta draws, with t = 0 for superiority and t = +0.002 for non-inferiority. Seeds [7, 13, 29, 42, 101]; 5 complete paired seeds are required and a family below that is `NOT_EVALUABLE`, never reported at a reduced seed count.

### 2.2 Block C114

Shared control: `mlp_114` — forced to `role: control` and a candidate in no family. Multiplicity: holm, scope within a family, across its members.

**`family_c114_seq`** — registered under family key `candidate: full_114`, `all_row_condition: member`, 3 members, Holm **m = 3**.

| member | primary | kind | contrast | slice | threshold |
|---|---|---|---|---|---|
| `primary` | yes | superiority | `full_114 − mlp_114` | `all` | +0.000 |
| `death_gate` | no | non_inferiority | `full_114 − mlp_114` | `death` | +0.002 |
| `chase_gate` | no | non_inferiority | `full_114 − mlp_114` | `chase` | +0.002 |

`screen.require`: `primary`, `death_gate`, `chase_gate`. Registered note, verbatim: SCREEN_PASS requires all three members to reject favourably (primary U95 < 0, each gate strictly U95 < +0.002) under Holm, with at least ten tournament blocks on each member's slice, five complete paired seeds, and at least 4 of 5 favourable per-seed directions on `primary`.

**`family_c114_feats`** — registered under family key `candidate: full_50`, `all_row_condition: member`, 4 members, Holm **m = 4**.

| member | primary | kind | contrast | slice | threshold |
|---|---|---|---|---|---|
| `primary` | yes | superiority | `full_114 − full_50` | `all` | +0.000 |
| `all_row` | no | superiority | `full_114 − mlp_114` | `all` | +0.000 |
| `death_gate` | no | non_inferiority | `full_114 − mlp_114` | `death` | +0.002 |
| `chase_gate` | no | non_inferiority | `full_114 − mlp_114` | `chase` | +0.002 |

`screen.require`: `primary`, `all_row`, `death_gate`, `chase_gate`. Registered note, verbatim: four members, as the design draft registers ("`full_114 - full_50` on `all` (primary) plus the all-row member `full_114 - mlp_114` and gates vs `mlp_114` (four members)"). SCREEN_PASS requires all four to reject favourably under Holm across the four, with at least ten tournament blocks on every member's slice, five complete paired seeds, and at least 4 of 5 favourable per-seed directions on `primary`. The family is keyed on `full_50` for the schema reason in deviations.family_key_is_the_registration_slot; its candidate in every member is `full_114`.

Holm adjustment as registered: step-down over the family's sorted raw p-values; with m members the adjusted value at rank r is min(1, max over j <= r of (m + 1 - j) * p_(j)), which makes the adjusted values monotone in rank. **m is that family's own member count**, and Holm is never pooled across families. Missing member rule, verbatim: an unavailable member takes a non-rejecting placeholder (adjusted p = 1), still occupies its slot, and the family CANNOT pass.

Uncertainty: contract `tournament_time_block_v1`, 2,000 replicates at rng seed 29, ball_weighted, blocks from `scripts/sim_eval/eval_statistics.load_competition_clusters` over `data/t20s_json`, `min_blocks` 10 and below that descriptive (CLAUDE.md invariant 7). Non-inferiority margin +0.002 log loss on ['death', 'chase'], against `mlp_114`; a candidate is non-inferior on a gate slice when U95, the 97.5th percentile of the paired harm draws of (full_114 - mlp_114), is STRICTLY below +0.002 log loss. U95 = 0.002 exactly does not pass.

Raw p convention, verbatim from the statistics file: p_raw = min(1, 2 * min[P(d* - t <= 0), P(d* - t >= 0)]) over the bootstrap delta draws, with t = 0 for superiority and t = +0.002 for non-inferiority. Seeds [7, 13, 29, 42, 101]; 5 complete paired seeds are required and a family below that is `NOT_EVALUABLE`, never reported at a reduced seed count.

Two estimands are reported for every family, as registered (**(i)** per checkpoint (one seed), reported separately; **(ii)** joint resampling of seeds and tournament blocks, reported separately). **(i)** is one seed's checkpoint with paired block-only uncertainty. **(ii)** is the arithmetic across-seed mean under joint seed-and-block resampling — a descriptive five-seed robustness screen, **not** the log loss of averaged probabilities and **not** uncertainty for a newly trained single checkpoint.

**What the intervals printed in this report are.** Every `95% CI` and `U95` column in §§ 3–4 is the member's `ci95` / `u95` field: an **ordinary two-sided 95% percentile interval from that member's own paired bootstrap draws — a marginal interval, one member at a time**. It is *not* a rank-local interval (the statistics file carries those separately as `rank_local_interval` at level `rank_local_level`, and this report does not print them), and it is not a simultaneous interval over the family. **Rejection is governed by the Holm-adjusted p within the family**, shown in the `p (Holm)` and `rejected` columns — never by reading whether a printed marginal interval clears zero or the margin. A marginal interval can exclude zero on a member that Holm does not reject, and that is not a contradiction.

## 3. Family results

The rejection rule is the Holm step-down **within** each family at the registered α, on the registered member set. A family passes only when every required member rejects favourably, every required slice carries at least ten tournament blocks, all five paired seeds are complete, and the registered per-seed direction count is met. The `95% CI` and `U95` columns are **marginal** two-sided 95% percentile intervals, one member at a time, and are reported for context only — see § 2.

### `family_4d` — candidate `mlp_spread_recency`, shared control `mlp_counts` — **SCREEN_NOT_PASS**

Holm within this family only, **m = 3**. Member table is the **estimand (ii)** readout, `seed_mean_joint`.

| member | contrast | slice | kind | point | 95% CI | U95 | p (raw) | p (Holm) | rank | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `primary` | mlp_spread_recency - mlp_counts on all | `all` | superiority | -0.00000 | [-0.00068, +0.00078] | +0.00078 | 0.9880 | 0.9880 | 3 | false | SCREEN_NOT_PASS |
| `death_gate` | mlp_spread_recency - mlp_counts on death | `death` | non_inferiority | -0.00010 | [-0.00199, +0.00175] | +0.00175 | 0.0260 | 0.0520 | 2 | false | SCREEN_NOT_PASS |
| `chase_gate` | mlp_spread_recency - mlp_counts on chase | `chase` | non_inferiority | -0.00034 | [-0.00111, +0.00049] | +0.00049 | < 0.001 | 0.0030 | 1 | true | SCREEN_PASS |

Per-seed family status (estimand (i)): seed 7 `SCREEN_NOT_PASS`, seed 13 `SCREEN_NOT_PASS`, seed 29 `SCREEN_NOT_PASS`, seed 42 `SCREEN_NOT_PASS`, seed 101 `SCREEN_NOT_PASS`.

Screen: required members `primary`, `death_gate`, `chase_gate`; member statuses `primary` SCREEN_NOT_PASS, `death_gate` SCREEN_NOT_PASS, `chase_gate` SCREEN_PASS. All direction requirements met: **false**. Status reason as recorded: *(none recorded)*. 2 of 5 per-seed primary directions are favourable; 4 of 5 are required, so this requirement is NOT met

| direction-required member | contrast | seeds | favourable directions | required | met |
|---|---|---|---|---|---|
| `primary` | `mlp_spread_recency-mlp_counts@all` | 5 | 2/5 | 4/5 | false |

### `family_c114_seq` — candidate `full_114`, shared control `mlp_114` — **SCREEN_NOT_PASS**

Holm within this family only, **m = 3**. Member table is the **estimand (ii)** readout, `seed_mean_joint`.

| member | contrast | slice | kind | point | 95% CI | U95 | p (raw) | p (Holm) | rank | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `primary` | full_114 - mlp_114 on all | `all` | superiority | +0.00285 | [+0.00185, +0.00376] | +0.00376 | < 0.001 | 0.0030 | 1 | false | SCREEN_NOT_PASS |
| `death_gate` | full_114 - mlp_114 on death | `death` | non_inferiority | +0.00485 | [+0.00311, +0.00702] | +0.00702 | 0.0010 | 0.0030 | 2 | false | SCREEN_NOT_PASS |
| `chase_gate` | full_114 - mlp_114 on chase | `chase` | non_inferiority | +0.00327 | [+0.00199, +0.00435] | +0.00435 | 0.0550 | 0.0550 | 3 | false | SCREEN_NOT_PASS |

Per-seed family status (estimand (i)): seed 7 `SCREEN_NOT_PASS`, seed 13 `SCREEN_NOT_PASS`, seed 29 `SCREEN_NOT_PASS`, seed 42 `SCREEN_NOT_PASS`, seed 101 `SCREEN_NOT_PASS`.

Screen: required members `primary`, `death_gate`, `chase_gate`; member statuses `primary` SCREEN_NOT_PASS, `death_gate` SCREEN_NOT_PASS, `chase_gate` SCREEN_NOT_PASS. All direction requirements met: **false**. Status reason as recorded: *(none recorded)*. 0 of 5 per-seed primary directions are favourable; 4 of 5 are required, so this requirement is NOT met

| direction-required member | contrast | seeds | favourable directions | required | met |
|---|---|---|---|---|---|
| `primary` | `full_114-mlp_114@all` | 5 | 0/5 | 4/5 | false |

### `family_c114_feats` — candidate `full_50`, shared control `mlp_114` — **SCREEN_NOT_PASS**

Holm within this family only, **m = 4**. Member table is the **estimand (ii)** readout, `seed_mean_joint`.

| member | contrast | slice | kind | point | 95% CI | U95 | p (raw) | p (Holm) | rank | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `primary` | full_114 - full_50 on all | `all` | superiority | -0.00676 | [-0.00772, -0.00584] | -0.00584 | < 0.001 | 0.0040 | 1 | true | SCREEN_PASS |
| `all_row` | full_114 - mlp_114 on all | `all` | superiority | +0.00285 | [+0.00185, +0.00376] | +0.00376 | < 0.001 | 0.0040 | 2 | false | SCREEN_NOT_PASS |
| `death_gate` | full_114 - mlp_114 on death | `death` | non_inferiority | +0.00485 | [+0.00311, +0.00702] | +0.00702 | 0.0010 | 0.0040 | 3 | false | SCREEN_NOT_PASS |
| `chase_gate` | full_114 - mlp_114 on chase | `chase` | non_inferiority | +0.00327 | [+0.00199, +0.00435] | +0.00435 | 0.0550 | 0.0550 | 4 | false | SCREEN_NOT_PASS |

Per-seed family status (estimand (i)): seed 7 `SCREEN_NOT_PASS`, seed 13 `SCREEN_NOT_PASS`, seed 29 `SCREEN_NOT_PASS`, seed 42 `SCREEN_NOT_PASS`, seed 101 `SCREEN_NOT_PASS`.

Screen: required members `primary`, `all_row`, `death_gate`, `chase_gate`; member statuses `primary` SCREEN_PASS, `all_row` SCREEN_NOT_PASS, `death_gate` SCREEN_NOT_PASS, `chase_gate` SCREEN_NOT_PASS. All direction requirements met: **true**. Status reason as recorded: *(none recorded)*. 5 of 5 per-seed primary directions are favourable; 4 of 5 are required, so this requirement is met

| direction-required member | contrast | seeds | favourable directions | required | met |
|---|---|---|---|---|---|
| `primary` | `full_114-full_50@all` | 5 | 5/5 | 4/5 | true |

## 4. Seed spread, every family member

Per-seed points are the estimand (i) point estimates at each registered seed. **Seed ranges are empirical spreads over the seeds, not confidence intervals.** No seed is selected and no CI endpoint is averaged. The favourable-direction count is the **zero-threshold** count (per-seed points below 0), which on a non-inferiority gate row is not the same as the count below that row's registered `+0.002` margin.

### 4.1 Rung 4d

| contrast | slice | seed 7 | seed 13 | seed 29 | seed 42 | seed 101 | seed range | favourable directions | mean point (ii) | mean 95% interval (ii) | blocks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `mlp_spread_recency − mlp_counts` | `all` | +0.00056 | +0.00051 | +0.00036 | -0.00066 | -0.00077 | +0.00133 | 2/5 | -0.00000 | [-0.00068, +0.00078] | 47 |
| `mlp_spread_recency − mlp_counts` | `death` | +0.00030 | +0.00140 | +0.00184 | -0.00232 | -0.00173 | +0.00417 | 2/5 | -0.00010 | [-0.00199, +0.00175] | 47 |
| `mlp_spread_recency − mlp_counts` | `chase` | +0.00019 | -0.00012 | -0.00015 | -0.00073 | -0.00091 | +0.00109 | 4/5 | -0.00034 | [-0.00111, +0.00049] | 47 |

### 4.2 Block C114

| contrast | slice | seed 7 | seed 13 | seed 29 | seed 42 | seed 101 | seed range | favourable directions | mean point (ii) | mean 95% interval (ii) | blocks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_114 − mlp_114` | `all` | +0.00305 | +0.00273 | +0.00298 | +0.00285 | +0.00261 | +0.00044 | 0/5 | +0.00285 | [+0.00185, +0.00376] | 47 |
| `full_114 − mlp_114` | `death` | +0.00532 | +0.00423 | +0.00588 | +0.00412 | +0.00470 | +0.00176 | 0/5 | +0.00485 | [+0.00311, +0.00702] | 47 |
| `full_114 − mlp_114` | `chase` | +0.00333 | +0.00333 | +0.00364 | +0.00337 | +0.00268 | +0.00096 | 0/5 | +0.00327 | [+0.00199, +0.00435] | 47 |
| `full_114 − full_50` | `all` | -0.00625 | -0.00803 | -0.00678 | -0.00654 | -0.00618 | +0.00185 | 5/5 | -0.00676 | [-0.00772, -0.00584] | 47 |

In C114, `full_114 − mlp_114 on all` appears as a member of **both** families — `family_c114_seq`'s primary and `family_c114_feats`' `all_row` member. It is one estimand on one set of rows occupying a Holm slot in two families, and a pass in both is not two findings (`known_asymmetries.shared_member`).

## 5. Absolute validation log loss per configuration

Numbers of record: full-precision per-seed validation log loss from `runs/<config>/summary.yaml`; never rounded report values and never log loss reconstructed from saved probabilities. Early stopping is live in batch 2, so arms may stop at different epochs — the stopping epoch is printed per run and **equal compute is not claimed**. The three frozen references sit in the same table as a level readout only; they are deterministic artifacts and `references.json` records them to four decimals.

### 5.1 Rung 4d

| config | seeds recorded | mean val LL | min | max | note |
|---|---|---|---|---|---|
| `mlp_counts` | 5 | 1.437878 | 1.437523 (seed 101) | 1.438554 (seed 42) | 5/5 runs, complete true |
| `mlp_spread_recency` | 5 | 1.437878 | 1.436754 (seed 101) | 1.438402 (seed 13) | 5/5 runs, complete true |
| `ref_eb_ctx` (frozen reference, 46 features) | — | 1.4500 | — | — | deterministic npz; `references.json` records 4 dp, no seeds, no interval |
| `ref_raw_ctx` (frozen reference, 22 features) | — | 1.4534 | — | — | deterministic npz; `references.json` records 4 dp, no seeds, no interval |
| `ref_lin_50` (frozen reference, 50 features) | — | 1.4433 | — | — | deterministic npz; `references.json` records 4 dp, no seeds, no interval |

| config | seed | validation LL | best epoch | epochs run | steps to best / total | parameters | wall s | machine |
|---|---|---|---|---|---|---|---|---|
| `mlp_counts` | 7 | 1.437690 | 15 | 19 | 2048 / 2432 | 272,262 | 43.5 | laptop |
| `mlp_counts` | 13 | 1.437895 | 16 | 20 | 2176 / 2560 | 272,262 | 106.3 | mini |
| `mlp_counts` | 29 | 1.437729 | 15 | 19 | 2048 / 2432 | 272,262 | 55.2 | laptop |
| `mlp_counts` | 42 | 1.438554 | 10 | 14 | 1408 / 1792 | 272,262 | 46.6 | laptop |
| `mlp_counts` | 101 | 1.437523 | 16 | 20 | 2176 / 2560 | 272,262 | 107.4 | mini |
| `mlp_spread_recency` | 7 | 1.438252 | 17 | 21 | 2304 / 2688 | 274,054 | 50.2 | laptop |
| `mlp_spread_recency` | 13 | 1.438402 | 16 | 20 | 2176 / 2560 | 274,054 | 107.9 | mini |
| `mlp_spread_recency` | 29 | 1.438091 | 15 | 19 | 2048 / 2432 | 274,054 | 66.8 | laptop |
| `mlp_spread_recency` | 42 | 1.437889 | 18 | 22 | 2432 / 2816 | 274,054 | 63.1 | laptop |
| `mlp_spread_recency` | 101 | 1.436754 | 29 | 30 | 3840 / 3840 | 274,054 | 160.9 | mini |

### 5.2 Block C114

| config | seeds recorded | mean val LL | min | max | note |
|---|---|---|---|---|---|
| `full_114` | 5 | 1.430577 | 1.430314 (seed 13) | 1.430749 (seed 7) | 5/5 runs, complete true |
| `mlp_114` | 5 | 1.427732 | 1.427580 (seed 13) | 1.427864 (seed 42) | 5/5 runs, complete true |
| `full_50` | 5 | 1.437333 | 1.436614 (seed 101) | 1.438344 (seed 13) | 5/5 runs, complete true |
| `ref_eb_ctx` (frozen reference, 46 features) | — | 1.4500 | — | — | deterministic npz; `references.json` records 4 dp, no seeds, no interval |
| `ref_raw_ctx` (frozen reference, 22 features) | — | 1.4534 | — | — | deterministic npz; `references.json` records 4 dp, no seeds, no interval |
| `ref_lin_50` (frozen reference, 50 features) | — | 1.4433 | — | — | deterministic npz; `references.json` records 4 dp, no seeds, no interval |

| config | seed | validation LL | best epoch | epochs run | steps to best / total | parameters | wall s | machine |
|---|---|---|---|---|---|---|---|---|
| `full_114` | 7 | 1.430749 | 16 | 20 | 2176 / 2560 | 306,950 | 113.2 | laptop |
| `full_114` | 13 | 1.430314 | 17 | 21 | 2304 / 2688 | 306,950 | 234.8 | mini |
| `full_114` | 29 | 1.430678 | 15 | 19 | 2048 / 2432 | 306,950 | 91.9 | laptop |
| `full_114` | 42 | 1.430710 | 17 | 21 | 2304 / 2688 | 306,950 | 1234.9 | laptop |
| `full_114` | 101 | 1.430435 | 16 | 20 | 2176 / 2560 | 306,950 | 224.7 | mini |
| `mlp_114` | 7 | 1.427695 | 17 | 21 | 2304 / 2688 | 280,198 | 59.7 | laptop |
| `mlp_114` | 13 | 1.427580 | 16 | 20 | 2176 / 2560 | 280,198 | 114.3 | mini |
| `mlp_114` | 29 | 1.427696 | 16 | 20 | 2176 / 2560 | 280,198 | 257.5 | laptop |
| `mlp_114` | 42 | 1.427864 | 13 | 17 | 1792 / 2176 | 280,198 | 1912.7 | laptop |
| `mlp_114` | 101 | 1.427825 | 15 | 19 | 2048 / 2432 | 280,198 | 109.6 | mini |
| `full_50` | 7 | 1.437001 | 23 | 27 | 3072 / 3456 | 298,758 | 137.3 | laptop |
| `full_50` | 13 | 1.438344 | 16 | 20 | 2176 / 2560 | 298,758 | 216.3 | mini |
| `full_50` | 29 | 1.437461 | 19 | 23 | 2560 / 2944 | 298,758 | 1738.1 | laptop |
| `full_50` | 42 | 1.437246 | 28 | 30 | 3712 / 3840 | 298,758 | 439.0 | laptop |
| `full_50` | 101 | 1.436614 | 24 | 28 | 3200 / 3584 | 298,758 | 302.1 | mini |

**Stage 1 cross-reference for C114, text only and a DIFFERENT measurement.** Stage 1's § 4 Holm table records C−B at **-0.0257** [-0.0346, -0.0102] (rank-local, raw p 0.0006, Holm p 0.0018, labelled favourable), read from `research/reports/embeddings/SEQ_STAGE1_REPORT.md` (sha256 `9da7b63d6cad…`). That quantity is **winner log loss in rollout through the simulator**, on selected seed-101 checkpoints and the **50-feature** contract. Nothing in this section is that quantity: here every number is **teacher-forced ball log loss on the validation split**, on the 114-column contract, under the batch-2 epoch schedule. Different target, different information set, different feature contract — so a batch-2 reading in either direction is **not a contradiction of** the stage-1 rollout gap, no table places the two side by side, and neither confirms the other.

## 6. Rung 4b — no admissible runs, family deferred

**Rung 4b produced no admissible runs.** Every `identity_residual` job — `identity_residual_l3` λ = 0.001, `identity_residual_l2` λ = 0.01, every seed attempted — **exited after training**, when the driver's checkpoint verification refused the checkpoint: `scripts/sequence_track/retrain_stage2.py` requires `feat_proj` / `head` parameters in the saved state dict, and the identity-only arm does not carry them. Training itself ran and was retried once; nothing survived the verification step, so there is no `COMPLETE.json` and no `summary.yaml` for either configuration, no statistics file for this rung, and **no number of this rung enters any family, any contrast table or any status**. Markers actually on disk for this rung: FAILED.

The family was already deferred by registration, before any of this: NO family is registered. `map: []` is deliberate and is what makes `registered_families` return an empty list with an expected count of zero: both configurations declare `role: control`, so there is no candidate-role configuration to require a family for. The deferral reason is a schema limitation, not a result: `stage2_stats.py` refuses a family member whose reference is not a registered configuration with run directories, and 4b's designed reference is a frozen npz artifact.

**Logged OBSERVATIONS from `run.log`, explicitly NOT evidence.** What the logs contain is listed below and nothing is inferred from it. The printed validation log losses are **rounded to four decimals** — they include 1.4500 and also 1.4499 and 1.4501 — so they cannot support any statement about whether the arm's predictions moved: a rounded log loss is not a prediction, equal rounded values are not equal predictions, and the frozen `ref_eb_ctx` level (1.4500 as recorded in `references.json`) is quoted here only as a coincident level, not as a comparison. Read from each log's **last attempt** (the driver retries once, so earlier attempts' epoch lines are not counted again):

| config | seed | attempts | epochs printed (last attempt) | distinct val_ll printed (4 dp) | early stop | recorded `validation_ll` | markers on disk |
|---|---|---|---|---|---|---|---|
| `identity_residual_l3` | 7 | 2 | 8 (0–7) | 1.4500 | true | 1.4499719169346799 | FAILED |
| `identity_residual_l3` | 13 | 2 | 4 (0–3) | 1.4500, 1.4501 | true | 1.4500008391953891 | FAILED |
| `identity_residual_l3` | 29 | 2 | 4 (0–3) | 1.4500 | true | 1.449984347509915 | FAILED |
| `identity_residual_l3` | 42 | 2 | 7 (0–6) | 1.4500, 1.4501 | true | 1.450003977767032 | FAILED |
| `identity_residual_l3` | 101 | 2 | 4 (0–3) | 1.4500, 1.4501 | true | 1.4500122468895664 | FAILED |
| `identity_residual_l2` | 7 | 2 | 8 (0–7) | 1.4499, 1.4500 | true | 1.4499554606431204 | FAILED |
| `identity_residual_l2` | 13 | 2 | 4 (0–3) | 1.4500 | true | 1.4500027987167887 | FAILED |
| `identity_residual_l2` | 29 | 2 | 4 (0–3) | 1.4500 | true | 1.449980081517222 | FAILED |
| `identity_residual_l2` | 42 | 2 | 7 (0–6) | 1.4500, 1.4501 | true | 1.4499804377821999 | FAILED |
| `identity_residual_l2` | 101 | 2 | 4 (0–3) | 1.4500, 1.4501 | true | 1.4500060080772195 | FAILED |

These are log lines, not measurements. **The defect, not the numbers, is the finding of this rung:** the arm trained, the driver's artefact check refused every checkpoint, and the runs are therefore **excluded from evidence entirely** — no bootstrap, no interval, no Holm slot, no direction count and no comparison against any reference was computed, and none may be computed from these directories. Nothing above may be quoted as "the identity residual matches the reference", as "the residual did not move the base", or as any statement about what the arm learned.

* `identity_residual_l3/seed_7/run.log` (sha256 `b2f3c367b4b5…`), 74 lines; the epoch, stop and refusal lines, verbatim:

```
epoch 0: train_ll=1.4189 val_ll=1.4500
epoch 1: train_ll=1.4189 val_ll=1.4500
epoch 2: train_ll=1.4188 val_ll=1.4500
epoch 3: train_ll=1.4188 val_ll=1.4500
epoch 4: train_ll=1.4188 val_ll=1.4500
epoch 5: train_ll=1.4188 val_ll=1.4500
epoch 6: train_ll=1.4187 val_ll=1.4500
epoch 7: train_ll=1.4187 val_ll=1.4500
early stop
"validation_ll": 1.4499719169346799
model.pt carries no ['feat_proj', 'head'] parameter(s), so it is not arm 'identity_residual''s checkpoint
```
* `identity_residual_l2/seed_7/run.log` (sha256 `8c4c4ddf6853…`), 74 lines; the epoch, stop and refusal lines, verbatim:

```
epoch 0: train_ll=1.4189 val_ll=1.4500
epoch 1: train_ll=1.4189 val_ll=1.4500
epoch 2: train_ll=1.4189 val_ll=1.4500
epoch 3: train_ll=1.4188 val_ll=1.4500
epoch 4: train_ll=1.4188 val_ll=1.4500
epoch 5: train_ll=1.4188 val_ll=1.4500
epoch 6: train_ll=1.4188 val_ll=1.4500
epoch 7: train_ll=1.4188 val_ll=1.4499
early stop
"validation_ll": 1.4499554606431204
model.pt carries no ['feat_proj', 'head'] parameter(s), so it is not arm 'identity_residual''s checkpoint
```

**What is required before rung 4b can say anything.** A driver fix (checkpoint verification that accepts an arm with no `feat_proj` / `head`), a full rerun of both λ arms at all five seeds, and — separately and beforehand — a new frozen config registering the family if any interval against the frozen reference is ever to be computed. Registering that family after reading the descriptive table would make the registration outcome-informed, and this report does not read one.

## 7. In plain language

**What was tested.** Small models were trained on historical T20 deliveries and then asked, one ball at a time, to put a probability on each of the six outcomes of the *next* delivery of 124,292 held-out validation balls. Rung 4d's two arms differ only in how many exposure columns they read; C114's three arms differ in whether they can see earlier balls and in which feature contract they read. This is teacher-forced ball prediction: nobody simulated a match, no market price appears anywhere in this batch, and a better log loss here implies nothing about a score distribution, a win probability, a prop or a betting edge.

**The registered reading, rung 4d.**

* **`family_4d`: does not pass its screen** (`SCREEN_NOT_PASS`). Its primary member `primary` (mlp_spread_recency - mlp_counts on all) is -0.00000 [-0.00068, +0.00078] on `all` rows: **unresolved — the interval straddles zero** (marginal interval; Holm `rejected` false). The family fails on its other required members: `death_gate` (mlp_spread_recency - mlp_counts on death) -0.00010 [-0.00199, +0.00175] — unresolved — the interval straddles zero. Not passing is not evidence against the mechanism; where the interval straddles zero the reading is **unresolved at this resolution**.

**The registered reading, block C114.**

* **`family_c114_seq`: does not pass its screen** (`SCREEN_NOT_PASS`). Its primary member `primary` (full_114 - mlp_114 on all) is +0.00285 [+0.00185, +0.00376] on `all` rows: **CI-clean adverse** (marginal interval; Holm `rejected` false). As registered, that is a **CI-clean adverse teacher-forced screening reading on the 114-feature contract** — outcome-informed and validation-only, on the split that also selected every checkpoint. It is not an out-of-sample, rollout, simulation or market statement, and it bears on no production artifact. The family fails on its other required members: `death_gate` (full_114 - mlp_114 on death) +0.00485 [+0.00311, +0.00702] — CI-clean adverse; `chase_gate` (full_114 - mlp_114 on chase) +0.00327 [+0.00199, +0.00435] — CI-clean adverse.
* **`family_c114_feats`: does not pass its screen** (`SCREEN_NOT_PASS`). Its primary member `primary` (full_114 - full_50 on all) is -0.00676 [-0.00772, -0.00584] on `all` rows: **CI-clean favourable** (marginal interval; Holm `rejected` true). As registered, that is a **CI-clean favourable teacher-forced screening reading on the 114-feature contract** — outcome-informed and validation-only, on the split that also selected every checkpoint. It is not an out-of-sample, rollout, simulation or market statement, and it bears on no production artifact. The family fails on its other required members: `all_row` (full_114 - mlp_114 on all) +0.00285 [+0.00185, +0.00376] — CI-clean adverse; `death_gate` (full_114 - mlp_114 on death) +0.00485 [+0.00311, +0.00702] — CI-clean adverse; `chase_gate` (full_114 - mlp_114 on chase) +0.00327 [+0.00199, +0.00435] — CI-clean adverse. So the family's own primary question is answered favourably while the family still does **not** pass: the screen requires every registered member, and the members shared with the sequence question are adverse.

**What none of this establishes.** A screen that does not pass does not show the mechanism is inert, and a CI-clean adverse reading on the validation split whose rows also selected every checkpoint is a **screening** reading: it says the transformer was worse than the token model on these held-out balls under this schedule, not that sequence access hurts out of sample, in rollout, or anywhere a decision is made. The C114 families are **correlated screens** over the same rows with a shared member and a shared gate reference, so there is no "either question passes" reading and no inference from one passing where the other does not. `full_114 − full_50` moves player identity together with the other 64 columns, so a favourable reading there is "the production contract beats the 50-feature contract", never "more state features help". Neither 114-column arm reproduces the production ball model — its categorical encoders are fitted differently — so nothing here bears on the model of record. Rung 4b established nothing at all: its runs were refused (§ 6).

**Rung 4b, in one sentence.** It did not produce admissible runs — what is on disk is training logs and checkpoints the driver refused — so it is excluded from evidence, its family stays deferred, and a driver fix plus a full rerun is required before the rung can be read at all.

## 8. Registered deviations, asymmetries and limitations (restated, none dropped)

### 8.1 Rung 4d — config `deviations`

* **`fixed_reference_is_not_a_member`** — the three frozen stage 4 references (`ref_eb_ctx`, `ref_lin_50`, `ref_raw_ctx`) are read DESCRIPTIVELY only. No family member of this config takes one as its reference. *Reason as registered:* stage2_stats loads every family member's candidate and reference from `<runs_root>/<config_id>/seed_<s>/predictions_validation.npz`, so a member's reference must be a registered CONFIGURATION with runs. A deterministic npz has no run directory and no seeds. The design draft asks for both arms to be "reported against the three references"; that report is a level comparison of validation log losses, carries no interval and no Holm slot, and cannot pass or fail anything. The same limitation is what defers rung 4b's family — see experiments/configs/seq_stage3_batch2_4b_v1.yaml.
* **`spread_and_recency_move_together`** — `mlp_spread_recency` adds the twelve spread columns AND the two recency counts in one step; no arm adds spread alone. *Reason as registered:* registered as the design draft registers it ("the arm adds spread PLUS recency; it does not isolate spread"). A favourable primary therefore licenses no claim about spread specifically, and an unfavourable one licenses no claim about recency specifically.
* **`epoch_schedule_not_step_budget`** — batch 2 runs the stage 2 epoch loop with live early stopping, while night 3's Block B ran a fixed 3,840-step budget. *Reason as registered:* the two are different schedules and their numbers are not comparable. No batch-2 arm may be tabled against a Block B arm, and this file registers no Block B configuration id.

### 8.2 Rung 4d — config `known_asymmetries`

* **`prior_is_not_as_of`** — the Dirichlet spread columns shrink toward the i7 cache's WHOLE-CORPUS global prior, which was computed over a corpus running to 2026-04-16 while the training split ends 2024-12-30. `mlp_spread_recency` reads that prior through fourteen columns and `mlp_counts` through none, so the two arms are not equally exposed to it. The production 50-feature EB columns both arms read already carry the same prior, so this is a difference of degree; the differential effect is unknown and is disclosed rather than adjusted for.
* **`sixteen_inputs_against_two`** — the candidate has fourteen more input columns and therefore more first-layer parameters than the control. Parameter counts are reported per run in `arm_params.n_parameters`; the contrast is "these columns" and never "this capacity".
* **`token_mlp_has_no_attention_layer`** — both arms are token_mlp, so neither has an attention layer and none of this rung's readings bears on any sequence mechanism.
* **`exposure_counts_are_as_of_the_cache_not_the_ball`** — the sidecar's as-of rule is the largest cache snapshot date <= the match date, so within a match every ball of both innings carries the same count; exposure does not advance ball by ball. Both arms are affected identically, but a reading must not be described as "balls faced so far in this innings".

### 8.3 Rung 4d — config `known_limitations`

* SEED-MACHINE CONFOUNDING, inherited from nights 1 and 3 and not modelled: seeds 7, 29 and 42 run on the laptop and 13, 101 on the mini, and MPS kernels are not bit-reproducible across machines. The split is BY SEED, so both arms carry the same three-laptop / two-mini mix and neither is advantaged; the machine is still a component of each per-seed number and is reported, never adjusted for.
* checkpoint selection uses the same validation split the contrasts are computed on, so every number of this rung is screening evidence.
* MPS kernels are not bit-reproducible across runs or machines, so a rerun of any checkpoint may differ in the low decimals; every checkpoint records mps_bit_reproducible false.
* the global outcome prior in the stats cache is not as-of-date and venue history in the frame is not recency-weighted. Both are inherited frame limitations; see known_asymmetries.prior_is_not_as_of for the part of it that is NOT symmetric between these two arms.
* early stopping at patience 3 on the validation split means the two arms may stop at different epochs. Steps-to-best and the stopping epoch are reported per run so the difference is visible; equal-compute is NOT claimed.

### 8.4 Block C114 — config `deviations`

* **`family_key_is_the_registration_slot`** — `family_c114_feats` is registered under the family key `candidate: full_50`, although the candidate of every one of its members is `full_114`. *Reason as registered:* `stage2_stats.registered_families` refuses a repeated family key ("family candidate ... is repeated") and requires every candidate-role configuration to be a family key exactly once. Two families whose members are all `full_114 - X` therefore cannot both be keyed on `full_114`. The family key is a REGISTRATION SLOT, not a claim: each member carries its own explicit `contrast: {candidate, reference}`, and that is what the statistics read. Found while writing this config; the alternative — a `family_key` distinct from the candidate — is a schema change and is out of scope for batch 2. Recorded in batch2_acceptance.md as the one-line change that would remove the awkwardness.
* **`encoders_fitted_on_train_only`** — the `v7_114` contract fits its four LabelEncoders on TRAIN rows only, while the production ball model (scripts/xgboost_v2.py) fits them on the union of train, validation and test unique values. *Reason as registered:* the production fitting would let a validation row's category index depend on the test split, which stage 2's `score_test: false` contract forbids. The consequence is that `full_114` and `mlp_114` see an out-of-vocabulary slot where production sees a fitted index, so neither is a reproduction of the production model and neither may be compared to one. This is the trainer's own registered deviation (transformer_t1 V7_114_CATEGORICALS), restated here because it bounds what a favourable `full_114 - full_50` licenses.
* **`epoch_schedule_not_step_budget`** — batch 2 runs the stage 2 epoch loop with live early stopping, while night 3's Block B ran a fixed 3,840-step budget. *Reason as registered:* the two are different schedules and their numbers are not comparable. No batch-2 arm may be tabled against a Block B arm, and this file registers no Block B configuration id.
* **`full_50_is_a_fresh_run`** — `full_50` is trained fresh under this config and is NOT stage 1's or stage 2's `full`. *Reason as registered:* a different config id, a different output tree, a different training signature and a different machine mix. Reusing an older `full` checkpoint would compare a batch-2 arm to a run selected under another registration, so it is trained again and costs five runs.

### 8.5 Block C114 — config `known_asymmetries`

* **`contract_adds_identity`** — the v7_114 contract carries `batter_encoded` and `bowler_encoded`, which no 50-feature arm sees. `full_114 - full_50` therefore moves PLAYER IDENTITY together with the other 64 columns; it is "the production contract" against "the 50-feature contract" and never "more state features". Any identity-specific reading belongs to rung 4b, not here.
* **`shared_member`** — `full_114 - mlp_114 on all` is family_c114_seq's primary AND family_c114_feats' `all_row` member. It is ONE estimand on ONE set of rows occupying a Holm slot in two families. The two families are correlated by construction; a pass in both is not two findings, and the Holm adjustment inside each family does not account for the other.
* **`capacity_differs_between_contracts`** — a 114-column input gives a larger first projection than a 50-column one, and the two 114-column arms differ from each other only in the attention stack. Parameter counts are reported per run in `arm_params.n_parameters`; no contrast here is equal-parameter.
* **`full_arms_are_slower`** — `full_114` trains roughly 3-4x slower per step than a token arm at the same batch size, and its early stopping may land on a different epoch. Steps-to-best and the stopping epoch are reported per run; equal-compute is NOT claimed anywhere in this block.

### 8.6 Block C114 — config `known_limitations`

* SEED-MACHINE CONFOUNDING, inherited from nights 1 and 3 and not modelled: seeds 7, 29 and 42 run on the laptop and 13, 101 on the mini, and MPS kernels are not bit-reproducible across machines. The split is BY SEED, so every arm carries the same three-laptop / two-mini mix and no arm is advantaged relative to another.
* the two families are CORRELATED screens over the same validation rows with a shared member and a shared gate reference. Claims are FAMILY-LOCAL only: no aggregate, no "either question passes", and no inference from one passing where the other does not.
* checkpoint selection uses the same validation split the contrasts are computed on, so every number of this block is screening evidence.
* MPS kernels are not bit-reproducible across runs or machines, so a rerun of any checkpoint may differ in the low decimals; every checkpoint records mps_bit_reproducible false.
* the global outcome prior in the stats cache is not as-of-date, and venue history in the frame is not recency-weighted. Both are inherited frame limitations shared by all three arms, so they are level effects rather than per-arm advantages.

### 8.7 Restated in this report's own terms

* **Seed–machine confounding.** Machine is recorded per run and is confounded with seed. Read from the run provenance:

| seed | machine | chip | host |
|---|---|---|---|
| 7 | laptop | Apple M5 Pro | Aryamans-MacBook-Pro.local |
| 13 | mini | Apple M4 | Aryamans-Mac-mini.local |
| 29 | laptop | Apple M5 Pro | Aryamans-MacBook-Pro.local |
| 42 | laptop | Apple M5 Pro | Aryamans-MacBook-Pro.local |
| 101 | mini | Apple M4 | Aryamans-Mac-mini.local |

  The split is **by seed** (laptop 7, 29, 42; mini 13, 101), so every arm carries the same mix and no arm is advantaged relative to another; every registered contrast is within-machine at each seed, so an additive machine effect cancels inside it. Arm-by-machine interaction remains inseparable from seed variation and **no machine term is fitted**. MPS kernels are not bit-reproducible across machines or runs, and every checkpoint records `mps_bit_reproducible false`.
* **The mini smoke was not run — a launch-gate deviation.** Acceptance check B6 requires one short token-arm run and one short `full`-arm run to complete **on each machine** before launch, with wall seconds in the runbook's batch-2 capacity table. The mini's token and `full` cells were filled from **laptop** smokes and marked "mini: NOT smoked — deviation, see B6"; only the `identity_residual` arm was smoked on the mini. So the mini's per-job budgets are extrapolations, not measurements, and a mini timeout would be a budgeting artifact rather than a result. The acceptance file's own B6 row, verbatim from `docs/sequence_track/batch2_acceptance.md`:

```
| B6 | Smoke on each machine before launch: one short run of a token arm and one of a `full` arm completes on each machine; wall seconds recorded in the runbook's Batch 2 capacity table. **Launch is refused if any row of that table is blank.** | **UNMET — launch deviation.** No pre-launch smoke of a token arm or a `full` arm was run on the mini, and the laptop's evidence is the gate-2 smokes at 65f2f63 (identity_residual 6 s, mlp+66 inputs 7 s, full+v7_114 16 s per 200 steps; trainer byte-identical at 86e049e) rather than a smoke at the freeze commit. Only a 1-epoch identity_residual run (7 s) preceded the mini launch. Measured in the queue instead: laptop full arms 2–38 min (slowdown after 02:30 IST), mini full arms per its log. Recorded as a deviation; the check is not satisfied. |
```
* **Correlated screens, family-local claims only.** C114's two families share a member and a gate reference; 4d has one family. No aggregate across families, and no comparison of one family's outcome to another's.
* **Selection optimism throughout.** Checkpoint selection used the same validation split every contrast is computed on, so every number in this report is screening evidence.
* **Two questions, one batch, no cross-tabling.** Rung 4d and block C114 are separate registrations with separate shared controls; batch-2 arms are also never tabled against night-3 Block B arms, whose schedule was a fixed step budget.

## 9. What happens next

**Nothing advances.** `advances: []`, `cohort_status: DEFERRED_UNOPENED`, `reads_performed: none`. No cohort feature, prediction or base-logit read happened, no artifact of record changed, no test split, golden set, forward holdout or market price was touched, and no verdict was logged — that call is the user's.

Run completeness as recorded in each configuration's `summary.yaml` on this checkout:

| rung | config | runs recorded / expected | complete | note |
|---|---|---|---|---|
| 4d | `mlp_counts` | 5 / 5 | true | seeds recorded [7, 13, 29, 42, 101] |
| 4d | `mlp_spread_recency` | 5 / 5 | true | seeds recorded [7, 13, 29, 42, 101] |
| C114 | `full_114` | 5 / 5 | true | seeds recorded [7, 13, 29, 42, 101] |
| C114 | `mlp_114` | 5 / 5 | true | seeds recorded [7, 13, 29, 42, 101] |
| C114 | `full_50` | 5 / 5 | true | seeds recorded [7, 13, 29, 42, 101] |
| 4b | `identity_residual_l3` | n/a | n/a | summary.yaml absent |
| 4b | `identity_residual_l2` | n/a | n/a | summary.yaml absent |

A family with fewer than five complete paired seeds is `NOT_EVALUABLE` and is **never** reported at a reduced seed count; rung 4b has no `summary.yaml` at all, which is the refusal described in § 6 and not a seed shortfall.

**The remaining work is rung 4b, and only rung 4b.** Rungs 4d and C114 are done: they ran, they were scored under their frozen registrations, and their screens did not pass — a screen that does not pass is a finished reading, not an unfinished run, so **there is nothing to re-run and nothing to advance for them**. What is outstanding:

1. **Fix the driver defect.** `scripts/sequence_track/retrain_stage2.py` refuses an `identity_residual` checkpoint because it looks for `feat_proj` / `head` parameters the identity-only arm does not carry. The artefact check must accept that arm's parameter set.
2. **Re-run rung 4b** — both λ arms at all five seeds — after the fix, and only then consolidate a `summary.yaml`.
3. **Register 4b's family in a new frozen config *before* any 4b readout is read as evidence.** The family is deferred because a frozen npz reference cannot be a family member under the present schema; registering after reading would make the registration outcome-informed.
4. **Leave the cohort closed.** It is `DEFERRED_UNOPENED`; the one confirmatory read must not be spent on screening evidence, and no batch-2 result changes that.

**Nothing in this batch advances, is promoted, or becomes an artifact of record**, and no verdict is logged by this report.
