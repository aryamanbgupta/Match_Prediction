# Astra gate 2 (results) — prompt template

Written 2026-09-12 by the stage 2 orchestrator, before the training runs
finished, so that the results gate does not have to be composed under time
pressure with numbers already in hand. **Fill the bracketed structural facts
from the files before sending. Do not soften any question because of what the
numbers turn out to be.**

---

You are Codex Astra, reviewing the stage 2 RESULTS GATE of the sequence/embeddings track (read-only). You reviewed the plan, the machine split, the statistics design and the code, and signed off for training and launch with three analysis defects explicitly deferred to this gate. Training is now complete and the statistics have been run. This review stands between the results and both the report and the commit.

## Read first

- `docs/sequence_track/stage2_acceptance.md` IN FULL — the contract, now carrying the D8 launch record, the D9 selection and the D10 results.
- `docs/SEQUENCE_TRACK_PLAN.md` § "Statistical rules for the whole track" and § "Stage 2".
- `research/reports/embeddings/SEQ_STAGE2_REPORT.md` — the generated report under review.
- `scripts/sequence_track/stage2_stats.py`, `render_stage2_report.py` and their tests — **your three deferred items were fixed here and this is where you verify them**.
- The statistics JSON, the D9 selection record, `models/embeddings/seq_stage2/runs/*/summary.yaml`, and `models/embeddings/seq_stage2/dependency/*.json`.
- Do not open `data/golden/`, `data/forward_holdout/` or `models/embeddings/seq_stage2/cohort/`. The cohort is `DEFERRED_UNOPENED` on your own ruling and must stay that way.

## What you deferred, and what was done

1. **Admission failing open** (`verify_run_admission()` accepting an absent or empty artefact manifest and missing size/hash declarations; `assert_comparable()` skipping absent components; a `provenance` block without a validation hash qualifying as a pin). Now: every expected artefact must be declared with size and md5 and must match on disk, absent or incomplete manifests refuse, components must be present and correctly typed, and the training signature is **recomputed from its components** so an edited signature string cannot pass.
2. **Unauthenticated summary values and k-sweep comparability** (an edited `ll` in `summary.yaml` could change the winner; `load_runs()` bypassed the summary verifier; `k_sweep()` never called `assert_comparable()`). Now: each summary LL is checked against the manifest-verified `metrics.json` value while the summary remains the number of record per D9.2; `load_runs()` verifies; `k_sweep()` asserts comparability across all five k configurations.
3. **Your four cross-arm requirements** from the signature ruling: common implementation-source hashes agreeing across arms with the recurrent entry excepted, identical `base_logits` identity between the two residual arms, each configuration's arm-specific components verified against its registered expected identity, and shared frame/cache/training identity anchored to the pin rather than to agreement among runs.
4. **Report fixes**: the dependency heading now says pending rather than recertified, a failed or missing certificate for a masked arm forces that arm's family to `NOT_EVALUABLE` and says so, and the identical `innings_2` / `chase` row sets are disclosed and never presented as independent corroboration.

## Structural facts of this run


## The questions

1. **Verify the three deferred items are genuinely closed**, with file:line evidence, and say whether the fixes introduced any new way for bad evidence to be admitted.
2. **Is every number in the report traceable to a file**, with no hand-entered cell, and does the § 9 coverage check actually fail when a limitation is dropped? Confirm the D10.13 list is complete in the report, naming any entry that is missing.
3. **Is the statistics arithmetic right**: the threshold-centred p convention at both thresholds, the within-family Holm step-down with a non-rejecting placeholder for an unavailable member keeping m = 3, the rank-local interval labelling, the strict `U95 < 0.002` gate, the joint seed-and-block resampling estimator, and the ≥10-block rule per slice.
4. **Does the report's language match the evidence?** Tonight is a two-seed screen with the cohort unopened, so nothing may be called confirmed, advanced or LANDED. Check especially the § 10 plain-language section and the D10.15 falsification wording against what the numbers actually support. Name any sentence that claims more than the evidence carries, and any place where a null result is presented as proof of absence.
5. **Given these results, what does the stage actually establish**, in your judgment, about the stage question: does the death-over harm come from stale history, from history belonging to other players, or from neither? Apply the registered falsification rules rather than the point estimates alone.
6. **What is the correct next step**, and specifically: does anything in these results change the five-seed preconditions in D10.16 before the cohort may be opened, or change which families should be extended to seeds 29, 42 and 101?
7. Anything that must be fixed before the results commit.

## Output

Numbered MUST-FIX first, then SHOULD, then NOTE, each with file:line. Then a short section answering question 5 in plain language, since it goes to the user. End with exactly one line: "VERDICT: SIGN-OFF" or "VERDICT: AGREE WITH CHANGES" or "VERDICT: NO SIGN-OFF".

### Structural facts (filled 2026-09-12 from the files)

- **32 of 32 seed runs admitted.** All sixteen configurations complete at both seeds 7 and 13. Nothing refused, nothing incomplete. `--consolidate` verified all 32 and rewrote every `summary.yaml` at 2/2.
- Machines: seed 7 on the laptop from the training worktree, seed 13 on the mini, both launched 2026-09-12T01:31+0530 from commit `972cdf3`; both queues reported `queue complete`, 0 FAILED / TIMEOUT / KILLED_MEMORY / REFUSED_MEMORY markers. The cross-machine signature identity was verified in production (`full` reads `f30f355f…` on both).
- Blocks: **47 tournament blocks / 545 matches / 124,292 rows**, 0 unmapped, contract `tournament_time_block_v1`, 120-day gap, 2,000 reps, rng seed 29. No slice fell under 10 blocks.
- **k sweep: SELECTED `same_entity_k30`.** Two-seed means: k=0 1.4391049, k=6 1.4381209, k=12 1.4371491, k=30 1.4366354, k=unr 1.4362082. `unr` is nominally best but beats 30 by only 0.000427, inside the registered 0.002 tolerance, so the rule keeps 30. Note the monotone ordering: log loss improves as the window widens.
- **Ownership recertification on TRAINED checkpoints: 8 of 8 PASS at exactly 0.** `same_entity_k0`, `same_entity_k30`, `same_entity_unr`, `recency_k30`, each at both seeds, 2,000 targets, seed 29. A positive control on the trained `full` checkpoint moved 1.839 (larger than the 0.752 it moved on the one-epoch smoke).
- Families: 15 of 15 evaluable, 0 contrasts `NOT_EVALUABLE`, 147 contrasts computed. Screen statuses on the seed-mean joint estimand: **`residual_mlp` SCREEN_PASS; the other fourteen SCREEN_NOT_PASS.** `fixed_decay` and `xlstm` are SCREEN_PASS on seed 7 only.

### The results themselves (candidate − reference, negative favours candidate; seed-mean joint estimand, slice `all`)

Registered primaries: `full − mlp` −0.00042 [−0.00195, +0.00100] 1/2; `fixed_decay − mlp` **−0.00349 [−0.00576, −0.00136] 2/2 CI-clean**; `fox − fixed_decay` −0.00005 [−0.00022, +0.00011] 1/2; `aligned_hist − full` **−0.00154 [−0.00263, −0.00051] 2/2 CI-clean**; `aligned_hist_rf − aligned_hist` +0.00059 1/2; `recency_k30 − mlp` −0.00027 1/2; `same_entity_k30 − recency_k30` −0.00117 [−0.00226, +0.00021] 2/2 not clean; `same_entity_unr − aligned_hist_rf` −0.00051 1/2; `same_entity_k0 − mlp` +0.00102 0/2; `same_entity_k6 − mlp` +0.00004 1/2; `same_entity_k12 − mlp` −0.00093 2/2; `lstm − mlp` +0.00269 0/2; `xlstm − mlp` **−0.00468 [−0.00636, −0.00316] 2/2 CI-clean**; `residual_mlp − mlp` **−0.00552 [−0.00671, −0.00435] 2/2 CI-clean**; `residual_t1 − residual_mlp` −0.00003 [−0.00060, +0.00044] 1/2.

**The death-over harm is reproduced.** `full − mlp` on `death` is **+0.00548 [+0.00168, +0.00900], 0 of 2 seeds favourable** — the interval excludes zero adversely. The 2026-08 ablation reported +0.0053 on 0 of 5 seeds on a different frame, and its all-slice gap was −0.0004 against this stage's −0.00042. By phase, `full − mlp` is favourable early and adverse late: powerplay −0.00193 [−0.00357, −0.00033], middle −0.00134, innings_1 −0.00150, chase +0.00085, death +0.00548.

Death-slice gates (`candidate − mlp`, pass needs U95 strictly < +0.002): `residual_mlp` −0.00357 (U95 −0.00072) **pass**; `residual_t1` −0.00339 (U95 −0.00042) **pass**; `xlstm` −0.00090 (U95 +0.00246) fail, narrowly; `fixed_decay` +0.00172 (U95 +0.00563) fail; `same_entity_unr` +0.00320 fail; `aligned_hist` +0.00399 fail; `same_entity_k30` +0.00440 fail; `lstm` +0.00514 fail; `recency_k30` +0.00562 fail. Chase gates pass for every arm except `lstm` (+0.00306, U95 +0.00952) and `recency_k30` (U95 +0.00215).

### Additional questions for this gate, beyond the template's seven

8. **Does this stage answer its own question?** My reading: the death-over harm reproduces almost exactly; ownership does **not** explain it (every `same_entity` arm still fails the death gate, and `same_entity_k30 − recency_k30` is 2/2 favourable but not CI-clean); learned forgetting adds nothing over fixed decay (`fox − fixed_decay` ≈ 0), while **fixed** decay is itself CI-clean favourable overall and roughly a third of the death harm; the one CI-clean mechanism result is that **participant-aligned history beats innings-previous history** (`aligned_hist − full`); and the only arm clearing the screen, `residual_mlp`, carries **no history at all**, with `residual_t1 − residual_mlp` ≈ 0 showing sequence adds nothing on top of the production prior. Is that reading correct and adequately hedged, and does the monotone k ordering (wider window always better) sit consistently with the ownership contrasts?
9. **`full − mlp` at ball level is −0.00042 and not CI-clean, while stage 1 measured −0.0257 CI-clean for the same pair in rollout through the simulator.** Both cannot be a property of "the architecture" in the same sense. What does the repository now get to say about that disagreement, and what is the minimal experiment that would resolve it?
10. **`xlstm` is the one sequence architecture that does not hurt the death overs** (−0.00090, U95 +0.00246, failing 0.002 only narrowly) and is CI-clean favourable overall. Is it the right single candidate to extend to seeds 29, 42 and 101, and which controls must go with it?
