You are Codex Astra, round 2 of the stage 2 FIVE-SEED ADDENDUM gate (read-only). In round 1 you returned `VERDICT: NO SIGN-OFF` with four MUST-FIX and two SHOULD. All are actioned and committed at `d188b8b`. This is a verification round: confirm or reject each closure, name any regression, and decide whether this stage can close.

Read `docs/sequence_track/stage2_acceptance.md` — its **"Gate 3 round 1"** section records every closure verbatim, and D12's COMPLETE result carries the five-seed numbers. Then `research/reports/embeddings/SEQ_STAGE2_FIVE_SEED_ADDENDUM.md` (1,560 lines), the re-rendered `SEQ_STAGE2_REPORT.md` (1,001 lines), `models/embeddings/seq_stage2/dependency/recert/` (20 certificates), `eval_out/seq_stage2_5seed/{stats.json,k_selection.json,analysis_pin.json}`, `eval_out/seq_stage2/analysis_pin.json`, and the changed code and tests. Do not open `data/golden/`, `data/forward_holdout/` or `models/embeddings/seq_stage2/cohort/`.

## Closures

1. **Dependency coverage.** Closed **with evidence, not a relaxation**: the 12 missing recertifications were run, so `dependency/recert/` now holds **20 certificates — 4 masked arms × seeds 7, 13, 29, 42, 101 — every one passing at max |Δ| exactly 0.000e+00 over 2,000 targets at seed 29**. In code, `dependency_required_checkpoint_seeds(stats)` derives the required seeds from the same registration `seeds_of(stats)` reads; the block propagates into the **4/5 qualification table** as well as results, gates, mechanism and falsification; the pin records `dependency_coverage.required_checkpoint_seeds`. A test renders a five-seed analysis over two-seed certificates and asserts all four arms block.
2. **Withdrawal and independence.** A new § 12 (only with `--prior-stats-json`) compares two seeds against five for every registered primary and **explicitly withdraws `aligned_hist − full`** with "unresolved, not evidence of no benefit"; sign reversals read "point estimate reversed; evidence remains unresolved"; the death paragraph says the harm is **reproduced in this five-seed validation extension**, that the extension **is not an independent replication** because seeds 7 and 13 are in both, and that a different frame does not make two analyses independent. The renderer refuses if the prior seed list equals the current one.
3. **The k-rule** is corrected by the same substitution-with-disclosure mechanism as `known_limitations[1]`: § 9 quotes the original and states the operative rule applies to the arithmetic mean over all five registered seeds, tolerance/default/sweep/tie unchanged. The config body is untouched and still byte-equal to the two-seed config; the correction fires only when the render's seed count differs from two.
4. **Both direction counts** are reported and separately labelled with the threshold exposed; gate arithmetic and every primary's zero-threshold count untouched. Your xLSTM row now renders `+0.00200 | 5/5 | 2/5`.
5. Obsolete seed prose removed; the five-seed statistics were **re-run** so the artifact no longer carries the 90 stale `screening: two seeds` strings. Selection unchanged: k=30, unrestricted margin 0.0003506.
6. The original D12 eight-configuration scope stands as the historical registration; the executed scope — all 16 configurations at 5 seeds — is recorded in D12's COMPLETE result.

## The thing I most want you to check

**I re-rendered and re-committed the two-seed report of record.** Two causes: § 2 enumerates every certificate under the recorded `--dependency-dir` and twelve landed after the first commit; and MUST-FIX 4's relabel is in the shared renderer, so it reaches the two-seed rendering — it had to, because **the committed two-seed report carried the same overstatement on 15 gate rows** (`full − mlp@chase` 2/2 below the margin but 1/2 favourable, and fourteen others).

I verified section by section before committing that **no result moved**: § 4's Holm tables (1,217 numbers), § 6, § 7, § 8's gates (850 numbers), § 10 and § 11 are all identical in their number sequences; only § 2 (196 → 220 numbers) and § 5 (270 → 316) changed. 987 → 1,001 lines. The stale test that had permitted unbounded drift here was replaced by `test_the_two_seed_report_of_record_is_byte_reproducible`.

Questions: was re-committing the report of record the right call rather than leaving a known mislabel standing with an erratum? Is the verification I did sufficient to support "no result moved"? And is the new reproducibility test the right guard, given that § 2's content depends on a directory whose contents can grow again?

## Also

- Both analysis pins re-written and verifying; both tracked snapshots refreshed. Suite **1799 passed, 1 skipped, 56 deselected**.
- Carried forward as you ruled: `same_entity_k0 − mlp` is CI-clean adverse with **Holm-adjusted p 0.0880**, so not a multiplicity-adjusted finding and not an isolation of history; and the **sequence cohort stays unopened** because confirming a production-prior control would not confirm the sequence hypothesis.
- `research/log_verdict.py` has **not** been run. The verdict is the user's.

## Output

Confirm or reject each closure with file:line. Answer the three questions about re-committing the report of record. Then state whether **stage 2 can close** on this evidence, and what the single next action should be. List any remaining MUST-FIX. End with exactly one line: "VERDICT: SIGN-OFF" or "VERDICT: AGREE WITH CHANGES" or "VERDICT: NO SIGN-OFF".
