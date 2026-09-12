# Night 3 acceptance — Stage 3a negative-transfer screen (two targets)

Written 2026-09-13 00:40 IST, before any run. Spec of record:
`docs/sequence_track/night3_design_draft.md` v7 § Block B. Runbook:
`docs/sequence_track/night3_runbook.md`. Config:
`experiments/configs/seq_stage3_night3_v1.yaml`. Queues:
`research/sequence_track/queue_{laptop,mini}_night3.yaml`.

Scope of THIS freeze: Block B only (6 configurations × 5 seeds = 30 runs).
4d, 4b and C114 are NOT in this freeze: their trainer code is present in the
commit but unexercised, their configs and queues do not exist, and they run
as a separately frozen batch after their own code gate. 3b and 3c are
deferred (3c: leakage audit found a masked objective not viable on this
contract). Block A is backlog.

Every result is validation-only, provisional, outcome-informed screening;
cohort `DEFERRED_UNOPENED`; no advancement; no market claim; no verdict is
logged by this stage (the user's decision).

## Checks (written before results)

| # | check | result |
|---|---|---|
| N1 | Config validates under `retrain_stage2.py --dry-run` for every configuration and renders the registered argv (max_steps 3840, eval_every 128, tier_embed 8 where registered, the frozen match list where registered, no early stopping). | dry-run of `mlp_pool`, `mlp_E_tiercond`, `mlp_rowmatch_P_tiercond` rendered the expected argv 2026-09-13 00:26; remaining ids rendered by the queue dry-run. |
| N2 | Both queues dry-run with `--queue` explicit; laptop 18 jobs (seeds 7, 29, 42), mini 12 jobs (seeds 13, 101); job ids `<config>-s<seed>`; per-seed output dirs under `models/embeddings/seq_stage3/night3/runs/`. | both dry-runs complete, nothing launched, 00:24. |
| N3 | Freeze lists: `experiments/stage3a/manifest.json` sha256s equal the sha256s pinned in the config for `target_E_validation_matches.json` and `big3_validation_matches.json`; `steps.json` S = 3840, E = 128 from the unfiltered 16,319 training innings; every training list disjoint from validation match ids. | verified 00:45: both slice sha256s pin==live; all four training lists' hashes present in the manifest; S=3840/E=128 from 16,319 innings; 0 listed matches in validation for every list. |
| N4 | Provenance: `pin_stage2.py --verify --config experiments/configs/seq_stage3_night3_v1.yaml` OK at the frozen commit; the stage 2 config pins and both stage 2 analysis pins re-verify after the trainer/stats edits (evidence hashes unchanged). | night config pin OK 00:33; five-seed analysis pin OK 00:39. |
| N5 | Stage 2 reproduction: `stage2_stats.py` on the generalised code reproduces the shipped five-seed `stats.json` and `k_selection.json` byte-identically modulo timestamps; digests committed in `research/reports/embeddings/stage2_five_seed_repro_digests.json`. | digests `19bcb604…` / `f46fa475…`, three-way agreement, test passing. |
| N6 | Shared-artifact contract: `mlp_pool` and `mlp_pool_tiercond` carry no target-specific parameter (`target_tier` is reporting-only and outside the signature), so one checkpoint per (config, seed) serves all three families. | CLOSED from the ten pooled-control run records (`mlp_pool`, `mlp_pool_tiercond` × 5 seeds): `arm_params_expected` keys are arm, base_logits_md5, bias, history_input, k, key_construction, residual_l2, tier_embed, wiring — no train_tier, train_match_list or target_tier; one checkpoint per (config, seed) serves all three families. |
| N7 | One commit on both machines: laptop worktree detached at the frozen commit; mini at the same commit; recorded here. | freeze commit `90acca8` at 00:54:57 IST, docs-only commit `65f2f63` at 00:55:59 IST; mini `git reset --hard` to each in turn (push + reset, 00:55 IST); laptop worktree `MP_train_s7` detached at `65f2f63` before launch. Both queue logs open at the launch commit: laptop 00:56:08 IST, mini 00:56:37 IST (mini clock 15:26:37 −0400). Earlier draft of this row said 01:50; that was a mis-read clock and is corrected here. |
| N8 | Smoke on each machine before launch: a short run completes on each machine; wall seconds recorded in the runbook table. | **Deviation recorded:** the laptop smoke (full 3,840-step `mlp_pool_tiercond` s7, 69 s, metrics written 00:34:53 IST) ran on the working tree BEFORE the freeze commit existed; `scripts/transformer_t1.py` was byte-identical to its committed state at 90acca8 (only `stage2_stats.py` and tests changed between 00:34 and 00:54), so the timing is valid but it is not a smoke of the frozen commit. The mini smoke (384-step `mlp_pool_tiercond` s13, 20 s, metrics 00:55:36 IST) ran at `90acca8` after the reset, before the launch. Earlier draft said 01:05 / 01:55; corrected. |
| N9 | Run completion: 30 `COMPLETE.json` files (6 × 5); 0 FAILED / TIMEOUT / KILLED_MEMORY / REFUSED_MEMORY markers; any missing member is `NOT_EVALUABLE`, never a reduced seed count. | 30/30 COMPLETE (first run started 00:56:10 IST, last run started 01:26:19 IST; laptop `queue complete` 01:21:52 IST, mini `queue complete` 01:29:00 IST), 0 failure markers on either machine; both queues logged `queue complete`. |
| N10 | Consolidation verify-only per seed on the laptop after the mini's seeds are synced; every checkpoint verifies against its signature; `tiers_seen_in_training` / `untrained_tiers` recorded for the target-only arms. | seeds 13/101 rsync'd (7 files each × 12 dirs, 01:45); `--consolidate` verified all 30 runs and wrote 6 summary.yaml (5/5 each); untrained tiers recorded: P-only [1, 2, 4], E-only [1]. |
| N11 | Statistics: `stage2_stats.py stats --config seq_stage3_night3_v1.yaml --runs-root models/embeddings/seq_stage3/night3/runs`, 2,000 reps, seed 29, seeds 7,13,29,42,101; three families (`family_3a_cond` 3 members, `family_3a_P` 4, `family_3a_E` 4), `direction_required` on both superiority members of P and E, `all_row_condition: none` on P and E, ≥10 blocks on every required slice or the member is `NOT_EVALUABLE`; `big3` reported descriptive (8 validation matches). | `eval_out/seq_stage3_night3/stats.json` sha256 `ff8ab0926f9e8142…`, 30/30 runs admitted, 47 blocks on `all`; `family_3a_cond` SCREEN_NOT_PASS (2/5), `family_3a_P` NOT_EVALUABLE (target_P has 6 blocks < 10), `family_3a_E` SCREEN_NOT_PASS (13 blocks, 2/5 on both superiority members). No screen passes; no advancement. |
| N12 | Report and gate: a night-3 report rendered from the stats JSON with every number read from a file; analysis pin written and verified; Astra results gate with all MUST-FIX closed before the report is committed. | report `research/reports/embeddings/SEQ_STAGE3A_NIGHT3_REPORT.md` rendered by `scripts/sequence_track/render_night3_report.py` (every number from stats.json / summary.yaml / metrics.json); hashes of the 16 files of record in `docs/sequence_track/night3_analysis_record.json`; Astra results gate: rounds 1–3 NO SIGN-OFF (overclaims, chronology, Holm display, literal settings, stale summary hashes, margin-based direction counts — all closed), **round 4 SIGN-OFF 2026-09-13 ~03:30 IST**. |

## Result — 2026-09-13

Block B ran to completion (30/30) and was scored under the registered rule. `family_3a_P` is NOT_EVALUABLE (6 tournament blocks on the premium-league validation slice, below the ten the rule requires); `family_3a_E` and `family_3a_cond` are SCREEN_NOT_PASS. Target-only point estimates are adverse on their own targets (P +0.00442 [+0.00146, +0.00742], descriptive; E +0.00190 [−0.00039, +0.00554], 2/5 favourable on both superiority members); tier conditioning on the pooled model reads +0.00033 [−0.00031, +0.00087]. Negative transfer was not detected; neither its absence nor positive transfer is established. No arm advances; cohort unopened; no market claim; no verdict logged (the user's decision). Batch 2 (4d, 4b, C114) is a separate freeze.
