You are Codex Astra, reviewing the stage 2 FIVE-SEED ADDENDUM (read-only) — the final gate of this stage. You have already: reviewed the plan; recommended the by-seed machine split; written the D9 and D10 acceptance tables; ruled the untouched cohort DEFERRED; signed off training and launch; refused the two-seed results twice and then authorised that commit once its two blockers closed; and audited the extension setup, finding that the statistics readouts were hard-coded to two seeds and the five-seed config could not run `ksweep`. All of that is closed and committed. This is the addendum those fixes existed to make possible.

Read `docs/sequence_track/stage2_acceptance.md` IN FULL — its **D12 "Result … COMPLETE"** block is the record under review, and D11 carries your earlier rounds. Then `research/reports/embeddings/SEQ_STAGE2_FIVE_SEED_ADDENDUM.md` (1,509 lines), the committed two-seed `SEQ_STAGE2_REPORT.md` (987 lines, unchanged), `eval_out/seq_stage2_5seed/{stats.json,k_selection.json,analysis_pin.json}`, `experiments/configs/seq_stage2_5seed_v1.yaml`, and the code you last reviewed. Do not open `data/golden/`, `data/forward_holdout/` or `models/embeddings/seq_stage2/cohort/` — the cohort remains `DEFERRED_UNOPENED` on your ruling.

## What happened

**80 runs, 16 configurations × 5 seeds (7, 13, 29, 42, 101), 0 failures, 0 timeouts, 0 memory refusals** across four queues on two machines. Seeds 7, 29, 42 on the laptop and 13, 101 on the mini, so every contrast stays within-machine at each seed. The mini's 16 seed-101 runs were rsynced home config by config after its writers were confirmed dead. `--consolidate --config seq_stage2_5seed_v1.yaml --seeds 7,13,29,42,101` verified all 80 and rewrote all 16 summaries at 5/5 **with no refusals**, so the cross-registration training-signature identity held on real data.

**Five-seed k sweep: `same_entity_k30` again.** Means k=0 1.4393197, k=6 1.4381917, k=12 1.4373955, k=30 1.4368982, k=unr 1.4365476; `unr` leads by 0.0003506, inside the 0.002 tolerance, so the default is kept — same decision as at two seeds, now on five.

**What five seeds changed** (candidate − reference, seed-mean joint, slice `all`):

- **`aligned_hist − full` DID NOT SURVIVE**: −0.00154 [−0.00263, −0.00051] 2/2 CI-clean at two seeds → **−0.00074 [−0.00174, +0.00027], 3/5** at five. It was the only CI-clean *mechanism* result in the two-seed report and its substantive claim is now withdrawn.
- **`same_entity_unr − aligned_hist_rf` reversed sign**: −0.00051 (1/2) → **+0.00004 [−0.00088, +0.00086], 1/5**.
- **Held and strengthened**: `fixed_decay − mlp` −0.00405 [−0.00559, −0.00263] **5/5 clean**; `xlstm − mlp` −0.00400 [−0.00534, −0.00280] **5/5 clean**; `residual_mlp − mlp` −0.00553 [−0.00674, −0.00427] **5/5 clean**.
- **Still unresolved**: `full − mlp` −0.00078 [−0.00201, +0.00034] 4/5; `fox − fixed_decay` −0.00011 [−0.00024, +0.00004] 4/5; `same_entity_k30 − recency_k30` −0.00084 [−0.00177, +0.00020] 4/5; `residual_t1 − residual_mlp` −0.00013 [−0.00068, +0.00030] 4/5.
- **Newly CI-clean ADVERSE**: `same_entity_k0 − mlp` **+0.00121 [+0.00004, +0.00251], 0/5**. `lstm − mlp` +0.00239, **0/5**.
- **The death-over harm is confirmed**: `full − mlp` on `death` **+0.00463 [+0.00177, +0.00733], 0 of 5 seeds favourable**. The 2026-08 ablation reported +0.0053 on 0 of 5 seeds on a different frame.

**Extension qualification** (your 4/5 rule, enforced only at ≥5 seeds): met by `full`, `fixed_decay`, `fox`, `same_entity_k30`, `same_entity_k12`, `xlstm`, `residual_mlp`, `residual_t1`; not met by `aligned_hist`, `aligned_hist_rf`, `recency_k30`, `same_entity_unr` (1/5), `same_entity_k6`, `same_entity_k0` (0/5), `lstm` (0/5). **Only `residual_mlp` is `extension qualified: yes`** — screen pass on all five seeds and the joint estimand. Every other family is `SCREEN_NOT_PASS` on the joint estimand; `xlstm` passes on seeds 7 and 101 only.

The addendum went to its own paths; the tools now refuse to overwrite night-1 evidence. Its analysis pin verifies. The report's prose is now seed-derived: it says "five seeds", its mechanism, estimand, residual and window tables have one column per seed, its machine-confound paragraph is built from the runs' own recorded provenance, and § 10 drops the seed-count reason while stating explicitly that this is not an advancement and the deferred cohort alone is decisive.

## One defect I am raising myself

The k-rule text the report quotes from the config still reads "the best (lowest) **two-seed** mean validation log loss". At five seeds the tool computes the five-seed mean, so the **quoted registered rule is stale**. The config body is held byte-equal to the two-seed one by test and by your own instruction, so I did not edit it. Rule on the right fix.

## Questions

1. **Verify the numbers and the arithmetic.** Recompute what you can. Does the 4/5 enforcement do what you specified, and is `residual_mlp` correctly the only qualified family?
2. **Is the addendum's language right given that a two-seed finding was withdrawn?** It must be unmistakable that `aligned_hist − full` is no longer claimed, without overcorrecting into "alignment does not help", which the interval does not support either.
3. **What does the stage now establish**, applying your own falsification rules to five seeds: the death harm, ownership, learned forgetting, and the fact that the only qualifying arm carries no history at all while `residual_t1 − residual_mlp` stays unresolved? State it as you would want it written.
4. **`same_entity_k0 − mlp` is now CI-clean adverse at 0/5.** Does that license any positive statement about history, or only a negative one about that particular restriction?
5. **Does anything here change D10.16?** Five seeds are now in hand for every family, so condition (1) is met. What remains before the cohort may be opened, and is `residual_mlp` — a production-prior control with no sequence access — even the kind of candidate a cohort read is for?
6. The stale k-rule wording, and anything else that must change before this is committed.

## Output

Numbered MUST-FIX, then SHOULD, then NOTE, each with file:line. Then a short plain-language section answering question 3, since it goes to the user. End with exactly one line: "VERDICT: SIGN-OFF" or "VERDICT: AGREE WITH CHANGES" or "VERDICT: NO SIGN-OFF".
