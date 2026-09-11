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

## Structural facts of this run (fill before sending)

- Runs admitted: [N] of 32. Configurations complete at both seeds: [list]. Incomplete or refused: [list with reason].
- Machines: seed 7 laptop, seed 13 mini, launched [stamp] from commit [sha]; wall time and peak RSS per run in D8's Result.
- Blocks per slice, and which slices fell under 10 blocks: [table].
- k sweep: [selected k, or BLOCKED_INCOMPLETE], with the D9.6 interpretation flags [raised / not raised].
- Dependency recertification on trained checkpoints: [status per masked arm].
- Families evaluable: [N] of 15. Screen statuses: [counts].

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
