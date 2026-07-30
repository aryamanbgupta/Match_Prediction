# I15 match-identity checkpoint — 2026-07-30

## Outcome

I15 is complete for future artifacts. Cricsheet file stems are the stable
primary key; the old date/team/venue string is display and frozen-artifact
compatibility metadata only.

## Evidence

- I7 test parquet: 798 rows, 788 legacy keys, 798 Cricsheet IDs.
- I7 full test parquet: the same 10-collision result.
- New primary dictionaries reject duplicates.
- Legacy alias joins reject one-to-many mappings.
- A real smoke loaded all 261 frozen Polymarket odds rows and resolved the
  Cricsheet-primary match `1477609` through its unique historical alias.
- Focused identity and evaluation-math suite: 44 passed.

## Compatibility decision

No frozen odds, model, evaluation, or consumed forward artifact was rewritten.
Future forward-holdout output uses schema 2 and persists the identity contract.
Schema-1 holdouts remain readable.

## Disclosed limitations (review, 2026-07-30)

- **Forward-protocol scoring-code drift.** 11 of the 17 artifacts pinned in
  `evaluation/forward_protocol_2026-06-01_2026-07-13.yaml` no longer match
  their SHA-256 (6 had already drifted at HEAD from I6–I8; this change set
  drifted `evaluate_forward_predictions.py`, `score_forward_match_m7.py`,
  `verify_forward_holdout.py` and deepened five others). `preflight` on the
  consumed protocol now fails closed, and
  `test_forward_eval_contract.py` was rewritten to expect that failure. The
  published forward result therefore stands as committed artifacts only; it
  is not re-runnable from the current tree, by design.
- **Post-checkpoint fixes (same day).** The initial checkpoint left three
  holes that were closed after review: the sim-eval odds join could let two
  doubleheader matches silently share one legacy odds row
  (`match_evaluator._resolve_odds_row` now claim-tracks rows per pass);
  `predict_golden.py` still keyed output by the synthetic id and silently
  dropped one match of a real golden doubleheader (62 rows / 61 keys —
  Malta v Gibraltar 2026-05-07; now keyed by Cricsheet ID, duplicates fail
  closed); and the blend/cluster/report joins
  (`blend_eval_json.py`, `eval_statistics.py`, `blend_report.py`,
  `build_ipl_dashboard.py`, `reliability_diagnostic.py`,
  `compare_i8_match_eval.py`) either could not resolve mixed
  new/frozen keying or degraded silently on shared aliases; all now join
  bidirectionally and fail loudly on doubleheader ambiguity.
- **Historical golden numbers.** Because of the `predict_golden` collision,
  historical golden metrics were computed over 61 of 62 matches. The next
  golden refresh will include all 62; expect a small, mechanical delta.

See `docs/I15_MATCH_IDENTITY_CONTRACT.md` for the normative contract.

