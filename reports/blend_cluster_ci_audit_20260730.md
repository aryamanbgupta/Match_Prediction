# Blend cluster-CI audit — 2026-07-30

## Result

No tournament-block ROI confidence interval quoted between the I3 change on
2026-07-23 and the `26ebe56` fix on 2026-07-30 was computed through the buggy
`blend_eval_json.py` → `reslice_eval_json.py` path. There are therefore no
affected intervals to correct, and no report conclusion changes.

| File | Quoted interval | Corrected interval | Conclusion change |
|---|---|---|---|
| None | — | — | No |

## Evidence checked

The audit enumerated every commit that added or modified `reports/` or
`research/reports/auto/` after I3 commit `61e52cd` and before fix commit
`26ebe56`, then searched the current report text for tournament/block ROI
intervals and for references to the blend and reslice commands.

- `reports/i3_eval_statistics_hardening.md`,
  `reports/m7_architecture_eval.md`, and `research/reports/auto/A7.md` carry
  the I3 intervals. I3 read the archived pre-bug M7 evaluation JSON directly;
  the latter two files only repeat that I3 result.
- `reports/forward_evaluation_2026-06-01_2026-07-13.md` was produced by the
  locked forward evaluator and its committed `evaluation_report.json`, not by
  blend/reslice.
- `reports/i5_legal_off_bat_evaluation_20260724.md` reports match-winner CIs
  produced by `scripts/sim_eval/run_sim_eval.py`; it did not blend direct-model
  predictions.
- `reports/i7_rebuild_checkpoint_20260725.md` reports run-sim intervals and
  paired model-delta bootstraps. Its iteration tables retain the expected
  25/19/11 event blocks; the affected blend output instead produced 134
  clusters on the ≥$50k slice. The paired delta intervals are also not an
  output that `reslice_eval_json.py` computes.
- `reports/i8_phase_matchup_checkpoint_20260730.md` uses run-sim sliced
  outputs and `scripts/compare_i8_match_eval.py`, which resamples the explicit
  event clusters already present in those outputs. It does not call
  `blend_eval_json.py`.
- The I9, I15, and Hundred reports created before the fix do not quote a
  tournament-block ROI interval produced by blend/reslice. The other reports
  changed in the window concern prop metrics, identity/state work, or forward
  provenance rather than this path.
- No report under `research/reports/auto/` was created or modified during the
  buggy interval. The B3 report was committed earlier on 2026-07-23, before
  I3 introduced the faulty stamping.

The explicitly exempt reports were also checked: the I3 report uses its
archived input, while `reports/d12_swap_promotion_20260730.md` and
`reports/golden_extension_eval_20260730.md` were generated at or after the
fix.

## Fixed-path control

The reproduce chain from `reports/d12_swap_promotion_20260730.md` was rerun
with the current code for frozen M7. The ≥$50k slice resolved **19** betting
blocks and returned ROI **+21.90% [−10.48%, +49.94%]**, matching the corrected
D12 control. This contrasts with the known buggy 134-cluster interval
**[+0.98%, +44.22%]** and confirms that the audit was run against the repaired
cluster contract.
