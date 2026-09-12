# Batch 2 acceptance — rung 4d, C114, rung 4b

Written 2026-09-13, **before any batch-2 run**. Spec of record:
`docs/sequence_track/night3_design_draft.md` v7 § "Block E" (4d, 4b) and
§ "C114". Runbook: `docs/sequence_track/night3_runbook.md` § "Batch 2 (4d, 4b,
C114)". Configs:

| rung | config | configurations | seeds | runs |
|---|---|---|---|---|
| 4d | `experiments/configs/seq_stage3_batch2_4d_v1.yaml` | `mlp_counts` (control), `mlp_spread_recency` | 7, 13, 29, 42, 101 | 10 |
| C114 | `experiments/configs/seq_stage3_batch2_c114_v1.yaml` | `full_114`, `mlp_114` (control), `full_50` | 7, 13, 29, 42, 101 | 15 |
| 4b | `experiments/configs/seq_stage3_batch2_4b_v1.yaml` | `identity_residual_l3`, `identity_residual_l2` | 7, 13, 29, 42, 101 | 10 |

Queues: `research/sequence_track/queue_laptop_batch2.yaml` (21 jobs, seeds 7,
29, 42) and `research/sequence_track/queue_mini_batch2.yaml` (14 jobs, seeds
13, 101). 35 runs total.

Scope of THIS freeze: 4d, C114 and 4b only. Block B is a separate, already
closed freeze (`docs/sequence_track/night3_acceptance.md`); no batch-2 arm may
be tabled against a Block B arm, because Block B ran a fixed 3,840-step budget
and batch 2 runs the stage 2 epoch loop with live early stopping. 3b and 3c
remain deferred; Block A is backlog; 4a and 4c are audits, not runs; 4e is not
in this freeze.

Every result is validation-only and provisional; cohort `DEFERRED_UNOPENED`;
no advancement; no market claim; no verdict is logged by this batch (the
user's decision).

## Why batch 2 is three config files and not one

`statistics.families.shared_control` is **file-wide**. `stage2_stats.py` forces
that id to `role: control`, requires it to be a candidate in no family, and
uses it for the role derivation and the exploratory pair generator. The three
rungs have three different controls — `mlp_counts`, `mlp_114`, and (for 4b) a
frozen npz that is not a configuration at all — so one file cannot express
them. Three files, one shared control each, three job groups in each queue.
Every queue job names its own `--config` explicitly.

## The 4b decision

**4b's family is DEFERRED. Its two configurations are registered and run; their
readouts are descriptive only.**

The design draft specifies one six-member Holm family whose reference in every
member is `ref_eb_ctx` — a deterministic fitted artifact
(`models/embeddings/stage4/refs/eb_ctx.joblib` plus two row-aligned npz files).
`stage2_stats.py` cannot express that:

* `_general_members` refuses any member whose `contrast.candidate` or
  `contrast.reference` is not a **registered configuration id**;
* `load_run` then loads each side from
  `<runs_root>/<config_id>/seed_<seed>/predictions_validation.npz`, and a frozen
  reference has no run directory, no seeds, no `metrics.json` and no
  `COMPLETE.json`;
* the one code path that reads a non-run reference, `base_only_readout`, is
  hard-wired to stage 2's production base logits
  (`<base_logits_dir>/validation.npz` plus a sidecar carrying `npz_md5`,
  `parquet_md5` and `booster_md5`), is labelled `role: exploratory`, and is not
  a family member type.

**Rejected alternatives, explicitly.** Registering `mlp_counts` or `mlp_114` as
4b's reference was considered and rejected — they answer a different question on
a different input set, and differencing against one would silently redefine the
rung. Registering an `identity_residual` arm with a very large `residual_lambda`
as a stand-in for the frozen reference was also rejected — a heavily penalised
residual approximates the reference but is not it, and its checkpoint is still
seed-dependent and early-stopped. Either would have produced a family that
passes or fails on a question nobody asked.

**What runs anyway.** Both configurations run all five seeds, so the arm, the
trainer path and the reference plumbing are exercised end to end and the family
can be computed **later from the same checkpoints without retraining** — provided
the family is frozen in a new config **before** the descriptive table below is
read as evidence for it. Reading first and registering afterwards would make the
registration outcome-informed, and this file records that.

**What is reported instead:** each configuration's per-seed validation log loss
over all validation rows, tabled beside the frozen references' own validation
log losses (`ref_eb_ctx` 1.4500, `ref_lin_50` 1.4433, `ref_raw_ctx` 1.4534),
plus per-seed `death` and `chase` log losses, plus
`identity_residual_l2 − identity_residual_l3`. No bootstrap against the
reference, no Holm, no PASS, no direction count, and **no statement of the form
"the identity residual beats the reference"**.

**The one-line change that would lift the deferral.** Add a fixed-reference
member type to `scripts/sequence_track/stage2_stats.py`: a member whose
`contrast.reference` may name an entry of a new `statistics.fixed_references`
block (`{npz, sha256, n_rows, label}`) instead of a configuration id, loaded
row-aligned like a run's `predictions_validation.npz` and paired on the
**candidate side only**, so estimand (i) resamples blocks alone and estimand
(ii) resamples seed × block on the candidate. Holm, the gate margin, the
ten-block rule and the missing-member placeholder are already
reference-agnostic.

## One further schema note (C114)

`registered_families` refuses a repeated family key and requires every
candidate-role configuration to be a family key exactly once. Both C114 families
have `full_114` as the candidate of every member, so they cannot both be keyed
on `full_114`. `family_c114_feats` is therefore keyed on `full_50`, and every
one of its members carries an explicit `contrast: {candidate: full_114,
reference: …}`. The family key is a **registration slot**, not a claim. The
one-line change that would remove the awkwardness is an optional `family_key`
distinct from `candidate` in the family entry; it is out of scope for batch 2.

## Checks (written before results)

| # | check | result |
|---|---|---|
| B1 | Each of the three configs loads under `retrain_stage2.load_config` and renders the registered argv under `--dry-run` for at least one configuration and seed: 4d renders `--extra-features models/embeddings/stage4/exposure --extra-cols <16 columns in registered order>`; C114 renders `--feature-contract v7_114`; 4b renders `--base-probs-dir models/embeddings/stage4/refs --residual-lambda 0.001 / 0.01`. No config renders `--max-steps`, `--eval-every`, `--train-match-list` or `--report-match-lists`, and `config["_freeze"]` is `None` for all three (no frozen-input block is read). | **PASS, 2026-09-13.** All three load. Dry-runs: `mlp_spread_recency` s7 rendered the 16 columns in order; `full_114` s7 rendered `--feature-contract v7_114`; `identity_residual_l2` s7 rendered `--base-probs-dir … --residual-lambda 0.01`. `_freeze` `None` for all three. |
| B2 | Each config loads under `stage2_stats`: `shared_control(cfg)` and `registered_families(cfg)` return the registered map with no `RefusalError`, every member's candidate and reference is a registered configuration, exactly one `primary: true` per family, and `unknown_keys` is empty on every family. | **PASS, 2026-09-13.** 4d: control `mlp_counts`, 1 family (`family_4d`, 3 members, `all_row_condition: member`). C114: control `mlp_114`, 2 families (`family_c114_seq` 3 members; `family_c114_feats` 4 members) . 4b: control `identity_residual_l3`, **0 families** (both configurations `role: control`, `map: []`, expected count 0) — the deferral above. `unknown_keys` empty everywhere. |
| B3 | Both queues dry-run with `--queue` explicit and no warnings: laptop 21 jobs (seeds 7, 29, 42), mini 14 jobs (seeds 13, 101); job ids `<config>-s<seed>`; per-seed output dirs under `models/embeddings/seq_stage3/batch2/runs/`; stop files `STOP_laptop_batch2` / `STOP_mini_batch2`; order rung-major (4d, C114, 4b) then seed-major. | **PASS, 2026-09-13.** `run_queue.sh --dry-run`: laptop "jobs listed: 21", mini 14 jobs resolved, every decision `run` / `no COMPLETE marker`, no warnings or errors, "dry run complete; nothing launched" on both. |
| B4 | References and sidecars hash-verified against their own manifests **before launch**: `models/embeddings/stage4/refs/references.json` records `eb_ctx` train probs sha256 `35f28faf…` and validation `86c5a531…`; `models/embeddings/stage4/exposure/manifest.json` records train `167730fb…` and validation `aaeab6ba…`. Recompute all four from the files on disk and compare; the configs pin the same digests. | **PASS, 2026-09-13.** All four recomputed digests equal the manifest values: train probs `35f28faf6ea3d874f25f5cac31d57d9e003518e1dd145d15853d5f57ac7da0d3`, validation probs `86c5a531dcbdcf18067a0aa749cadc356db373d7834aa66970a31d04a2437d62`, exposure train `167730fb55c958c68606aea9cf68dbb29bc2690c01a36a0cdfc8b758f42c0ccc`, exposure validation `aaeab6ba07c225f53d0ba3ac9c6d7efe6770e8bc767d2d14af011da00e6c0316`. Config-pinned `references.json` sha256 `cb868833…` and `manifest.json` sha256 `a1072431…` also match. Re-verify immediately before launch. |
| B5 | One commit on both machines: laptop worktree detached at the batch-2 freeze commit; mini reset to the same commit; both recorded here with times. The three config sha256s the queue dry-run reports must be identical on both machines. | freeze commit `86e049e` (2026-09-13 02:11 IST): mini `git reset --hard 86e049e` before launch; laptop worktree `MP_train_s7` detached at `86e049e`. Queue logs opened 02:13 IST (laptop) and 02:14 IST (mini). The runner does not print config digests; the run-time evidence is in each run's `run_record.json` (`config_sha256`) and `metrics.json` (commit): `seq_stage3_batch2_4b_v1.yaml` 9def95ee3e04… on laptop/mini; `seq_stage3_batch2_4d_v1.yaml` ced2bec38270… on laptop/mini; `seq_stage3_batch2_c114_v1.yaml` fd6ffa790a85… on laptop/mini. Commit recorded in the 25 admitted metrics.json files: ['86e049e'] (key ['git_head_short']). Paths: models/embeddings/seq_stage3/batch2/runs/<config>/seed_<s>/run_record.json. Stage 4 inputs synced file by file; exposure train.parquet md5 `2ca950ac…` on both machines. |
| B6 | Smoke on each machine before launch: one short run of a token arm and one of a `full` arm completes on each machine; wall seconds recorded in the runbook's Batch 2 capacity table. **Launch is refused if any row of that table is blank.** | **UNMET — launch deviation.** No pre-launch smoke of a token arm or a `full` arm was run on the mini, and the laptop's evidence is the gate-2 smokes at 65f2f63 (identity_residual 6 s, mlp+66 inputs 7 s, full+v7_114 16 s per 200 steps; trainer byte-identical at 86e049e) rather than a smoke at the freeze commit. Only a 1-epoch identity_residual run (7 s) preceded the mini launch. Measured in the queue instead: laptop full arms 2–38 min (slowdown after 02:30 IST), mini full arms per its log. Recorded as a deviation; the check is not satisfied. |
| B7 | Run completion: 35 `COMPLETE.json` files (10 + 15 + 10); 0 FAILED / TIMEOUT / KILLED_MEMORY / REFUSED_MEMORY markers on either machine; both queues log `queue complete`. Any missing member is `NOT_EVALUABLE`, never a reduced seed count. | 25/25 non-4b runs COMPLETE (laptop 15, queue complete 04:11 IST; mini 10, queue complete 02:44 IST), 0 failure markers among them. **4b: all 10 `identity_residual` runs FAILED** (mini seeds 13/101 and laptop 7/29/42, both λ): training finished (run.log records validation LL at the frozen `ref_eb_ctx` level with early stop) but `retrain_stage2.py`'s checkpoint verification refused the checkpoint (`model.pt carries no ['feat_proj', 'head'] parameter(s)`) — the driver's expected-parameter check does not know the identity-only arm. Not admitted; recorded as a driver defect; 4b deferred whole. Laptop full-arm jobs after 02:30 IST ran 20–38 min each (load average ~7; the stage 2 seed-42 laptop slowdown recurred), within the 2× alarm, no retries. |
| B8 | Consolidation verify-only per config on the laptop after the mini's seeds are synced; every checkpoint verifies against its signature; `summary.yaml` written with 5/5 per configuration; `arm_params` recorded for each rung carries what it should (`extra_features.{dir,cols,sha256}` for 4d, `feature_contract_n` 114 + `feature_contract_sha256` `d968f9f9…` for the two 114-column arms and **absent** for `full_50`, `base_probs_sha256` + `residual_lambda` for 4b). | seeds 13/101 rsync'd for the five non-4b configs (7 files × 10 dirs, 04:18 IST); `--consolidate` verified 5/5 for `mlp_counts`, `mlp_spread_recency`, `full_114`, `mlp_114`, `full_50` (arm_params cross-checks incl. the batch-2 identity fields passed); the 4b config was not consolidated (no admissible run). |
| B9 | Statistics: `stage2_stats.py stats` once per config against `--runs-root models/embeddings/seq_stage3/batch2/runs`, 2,000 reps, seed 29, seeds 7,13,29,42,101. 4d: one family, 3 members, ≥10 blocks on `all`, `death`, `chase` or the member is `NOT_EVALUABLE`. C114: two families (3 and 4 members), same block rule. 4b: **no family** — the descriptive table only, and the run must not emit any PASS/SCREEN status for this rung. | `eval_out/seq_stage3_batch2/4d_stats.json` `ea2a4046ca3b1704…` (10/10 admitted, 47 blocks on `all`; `family_4d` SCREEN_NOT_PASS, primary −0.00000 [−0.00068, +0.00078], 2/5) and `c114_stats.json` `26e3b042b256a25e…` (15/15 admitted; `family_c114_seq` SCREEN_NOT_PASS — `full_114 − mlp_114` +0.00285 [+0.00185, +0.00376], 0/5, CI-clean ADVERSE; `family_c114_feats` SCREEN_NOT_PASS — `full_114 − full_50` −0.00676 [−0.00772, −0.00584], 5/5, CI-clean favourable, but its all-row member `full_114 − mlp_114` is adverse so the screen fails). Five complete seeds everywhere. |
| B10 | Report and gate: one batch-2 report rendered from the stats JSONs with **every number read from a file**; analysis pin written and verified; Astra results gate with all MUST-FIX closed before the report is committed. The report must state the 4b deferral in its own words and must not present any 4b number as a comparison against the frozen reference. | report `research/reports/embeddings/SEQ_STAGE3_BATCH2_REPORT.md` rendered by `scripts/sequence_track/render_batch2_report.py`; hashes in `docs/sequence_track/batch2_analysis_record.json`; Astra results gate: round 1 NO SIGN-OFF (renderer wording, 4b overreach, interval labelling, hard-coded Stage 1 number, B5/B6), round 2 NO SIGN-OFF (B5 digests), **round 3 SIGN-OFF 2026-09-13 ~05:05 IST**. |

## What a pass would and would not establish

* **4d** — a `family_4d` SCREEN_PASS would say that, on this validation split
  and under this schedule, adding twelve Dirichlet spread columns and two
  recency counts to a token model that already sees the two as-of counts
  improves all-row log loss without costing death overs or chases beyond
  +0.002. It would say nothing about spread alone (spread and recency move
  together by registration), nothing about any production model, and nothing
  out of sample.
* **C114** — `family_c114_seq` asks whether sequence access buys anything on
  the production 114-column contract; `family_c114_feats` asks whether that
  contract buys anything over the 50-feature contract at fixed wiring. The two
  share a member and a gate reference and are correlated screens, not
  replications. Neither bears on the production ball model, whose categorical
  encoders are fitted differently (train-only here, train∪validation∪test in
  `scripts/xgboost_v2.py`).
* **4b** — establishes nothing. It has no registered family.

Nothing in this batch can be LANDED (CLAUDE.md invariant 9): checkpoint
selection is on the same validation split every contrast is computed on, the
cohort stays closed, and no test split, market price or betting layer is
touched anywhere.
