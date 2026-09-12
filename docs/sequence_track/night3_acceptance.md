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
| N6 | Shared-artifact contract: `mlp_pool` and `mlp_pool_tiercond` carry no target-specific parameter (`target_tier` is reporting-only and outside the signature), so one checkpoint per (config, seed) serves all three families. | pending (verified from arm_params in each run_record.json at consolidation). |
| N7 | One commit on both machines: laptop worktree detached at the frozen commit; mini at the same commit; recorded here. | frozen commit `90acca8` (plus this runbook/acceptance commit on top, docs only): mini `git reset --hard` to it 01:50; laptop worktree `MP_train_s7` detached at it 01:50. |
| N8 | Smoke on each machine before launch: a short run completes on each machine; wall seconds recorded in the runbook table. | laptop: full 3,840-step `mlp_pool_tiercond` s7 in 69 s (01:05); mini: 384-step `mlp_pool_tiercond` s13 in 20 s (01:55), metrics/model/predictions written. Both queues dry-run clean at the frozen commit. |
| N9 | Run completion: 30 `COMPLETE.json` files (6 × 5); 0 FAILED / TIMEOUT / KILLED_MEMORY / REFUSED_MEMORY markers; any missing member is `NOT_EVALUABLE`, never a reduced seed count. | pending. |
| N10 | Consolidation verify-only per seed on the laptop after the mini's seeds are synced; every checkpoint verifies against its signature; `tiers_seen_in_training` / `untrained_tiers` recorded for the target-only arms. | pending. |
| N11 | Statistics: `stage2_stats.py stats --config seq_stage3_night3_v1.yaml --runs-root models/embeddings/seq_stage3/night3/runs`, 2,000 reps, seed 29, seeds 7,13,29,42,101; three families (`family_3a_cond` 3 members, `family_3a_P` 4, `family_3a_E` 4), `direction_required` on both superiority members of P and E, `all_row_condition: none` on P and E, ≥10 blocks on every required slice or the member is `NOT_EVALUABLE`; `big3` reported descriptive (8 validation matches). | pending. |
| N12 | Report and gate: a night-3 report rendered from the stats JSON with every number read from a file; analysis pin written and verified; Astra results gate with all MUST-FIX closed before the report is committed. | pending. |

## Result — to be filled in the morning
