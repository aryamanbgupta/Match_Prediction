You are Codex Astra. A short, practical audit request, read-only — not a formal gate. The stage 2 orchestrator has made three launch mistakes in a row on the seed-extension work and wants the remaining setup checked BEFORE more machine time is spent failing. Find what is still wrong.

Read: `docs/sequence_track/stage2_acceptance.md` § D12 (registers the extension) and § D8 (the machine-split rules the extension must preserve); `experiments/configs/seq_stage2_ext_v1.yaml`, `seq_stage2_ext2_v1.yaml`, `seq_stage2_5seed_v1.yaml`; `research/sequence_track/queue_laptop_ext.yaml`, `queue_mini_ext.yaml`, `queue_laptop_ext2.yaml`, `queue_mini_ext2.yaml`; `scripts/sequence_track/retrain_stage2.py` (its config validation, its `--consolidate` path, its signature and reuse logic); `scripts/sequence_track/stage2_stats.py` (how it resolves seeds, families and the k sweep). Do not open `data/golden/`, `data/forward_holdout/` or `models/embeddings/seq_stage2/cohort/`.

## What exists

Night 1 trained all sixteen configurations at seeds 7 and 13 under `seq_stage2_v1.yaml` and is complete and consolidated. Two extensions then followed, both at seeds 29, 42 and 101, both preserving the by-seed machine split (29 and 42 laptop, 101 mini):

- **Extension 1**, `seq_stage2_ext_v1.yaml`: a whole-file copy of the night-1 config with only the seed list changed, so it still declares all sixteen configurations; the queues select eight — `mlp`, `full`, `fixed_decay`, `fox`, `aligned_hist`, `xlstm`, `residual_mlp`, `residual_t1`. Complete on the mini (8/8 at seed 101), in progress on the laptop (seed 29 done, seed 42 running).
- **Extension 2**, `seq_stage2_ext2_v1.yaml`: built by *filtering* the configuration list down to the other eight — the five `same_entity` window arms, `recency_k30`, `aligned_hist_rf`, `lstm` — so that all five window arms exist at five seeds, which the unlock conditions require before any window claim is retained. Just launched on the mini at seed 101.
- **`seq_stage2_5seed_v1.yaml`**: also built by filtering, registering all five seeds for extension 1's eight configurations and seven families. Consolidation and analysis ONLY, never training. It exists because `--consolidate` rewrites each `summary.yaml` from the *config's* registered seeds, so consolidating under either training config would drop the other night's seeds from the summary even though every run directory survives.

## The three mistakes already made, for calibration

1. The morning rsync used `--include='*/seed_13/***'`; macOS openrsync ignores `***`, so it created directory shells, copied nothing, and reported success. Replaced with a subtree copy plus `--exclude='summary.yaml'`.
2. Extension 1's queues named the extension config in each job's `config` field (what the runner hashes) but did not pass `--config` on the command line, so the driver read the night-1 config and correctly refused to widen its seeds. All sixteen jobs failed in about a second.
3. Both *filtered* configs kept their original `queue_order` values (6..13 and 1,2,3,4,5,14,15,16), which the driver validates must be 1..N exactly once. Extension 2 failed all eight mini jobs; the merge config had the same latent defect and would have failed later, at consolidation. Both renumbered contiguously and re-pinned.

## What I want from you

1. **What else is wrong with these seven files?** Specifically: does filtering a config's `configurations` list leave any other field internally inconsistent — the family map, the contrast declarations, `statistics`, the k-sweep registration, anything that names a configuration no longer present, or any count the driver or the statistics tool validates? Extension 2's family map was filtered to its eight candidates and the merge config's to seven; check that the *references* those families name are all still resolvable in each file, and say what happens if they are not.
2. **Will `--consolidate --config seq_stage2_5seed_v1.yaml` actually work** over a tree where the eight extension-1 configurations have five seed directories and the other eight have two? Will it refuse, silently skip, or rewrite something it should not? What exactly should the invocation be, and in what order relative to the rsync?
3. **Will the statistics tool read a five-seed table correctly** at `--seeds 7,13,29,42,101` against the merge config — including the 4/5 favourable-direction count the plan requires at five seeds, the Holm families, and the estimand (ii) joint resampling with five seeds rather than two? Is there any place where two seeds is assumed structurally?
4. **After extension 2 finishes, what is the correct way to get a five-seed reading for the ownership and window families?** They are not in the merge config. Do I need a fourth config, can the merge config be extended to sixteen configurations and fifteen families, or is there a cleaner arrangement? Say which, concretely, because I would rather build the right thing once.
5. **Is there any risk to the already-completed night-1 evidence** from any of this — the consolidated two-seed summaries, the statistics JSON, the analysis pin, the report — given that the extensions are rewriting `summary.yaml` files in the same run directories?
6. Anything about the by-seed split with three seeds across two machines (29 and 42 on the laptop, 101 on the mini) that breaks the reasoning D8 rests on.

Be concrete and terse. Numbered MUST-FIX for anything that will fail or corrupt, then SHOULD, then NOTE. No verdict line needed; I want the defect list.
