# Item 6 acceptance checks (paired multi-seed experiment harness)

Written 2026-09-10 by Claude before implementation. Steps 1–2 (harness,
schema, tests, verify hooks) plus the trainer's fold-local encoder contract
are in scope now; candidates A–D (steps 3–6) each get a check block appended
before they run, and they run only after the harness is reviewed.

| # | Check | Pass condition |
|---|---|---|
| H1 | `scripts/experiment_harness.py run <config.yaml>` | config names frame (manifest role or explicit dir), baseline and candidate trainer args, seed list (default 29, 7, 13, 42, 101; `seeds: [29]` and `seeds: 3` accepted), slices, cost model, `expected_minutes_per_seed`, ceiling; schema validated with named errors |
| H2 | Per-seed artifacts | `<out>/<arm>_seed<N>/` holds the trained model dir, `test_predictions.json`, blended and sliced JSONs whose summary carries `model_seed` (the gate's stamp) and a `config_hash`; seeds whose artifacts exist with a matching hash are skipped (resume test after a killed run); a changed hash forces retrain |
| H3 | Gate call | the harness calls `claim_gate.decide` with the registered odds role and cluster dir, writes `gate.json` and one `harness.json` with per-seed rows and the Verdict; it never passes cluster labels; the result is logged only via `log_verdict --gate-json` (harness prints the command, does not run it) |
| H4 | Budget guard | seeds × expected minutes > ceiling refuses without `--allow-long`; test |
| H5 | Seed rule | provisional stamping and estimator come from the gate, not the harness; a 1-seed and a 3-seed dry run show `provisional: true` |
| H6 | Pairing | both arms' `test_predictions.json` must carry identical match-id sets, else the harness refuses (test) |
| H7 | Ensemble estimand | `candidate.kind: ensemble` averages the five seed models in logit space into ONE prediction set; the baseline is the seed-29 production model; the gate runs under the 1-seed rule (provisional, 0.007 floor); per-seed diagnostics against the other four baseline seeds are reported but not decided; test on synthetic predictions |
| H8 | Fold-local encoders | `xgboost_match_v1.py --fit-encoders-on {train,all}`; harness default `train`, legacy CLI default `all`; unseen venue/tier maps to code −1 at transform time; test trains without one venue and predicts a row at it with code −1 and the degraded flag; leakage test: no test-fold venue in a fold's encoder vocabulary unless in that fold's training rows |
| H9 | Calibration split | `calibrate_match_predictions.py --calib-after <date>`: early stopping uses validation rows before the date, calibration rows on or after; leakage test: no calibration row id in the early-stop set |
| H10 | Rolling-origin folds (candidate C support) | `folds:` config as row masks by `match_date` over the existing frame, encoders refit per fold; fold-mean LL reported; production re-selection is a separate, human decision |
| H11 | Verify hooks | `d12_run.verify` frame checks ported as optional `verify:` entries (diff-negation identity, h2h prior 0.5, swap doubling) that run before training |
| H12 | Dry run | `--dry-run` on the mini corpus + a tiny frame completes end to end in under 2 minutes without a real model (uses a stub trainer) and exercises H2–H7 |
| H13 | Tests | `scripts/tests/test_experiment_harness.py` covers H1–H12 on tmp trees; no real artifacts unless `needs_artifacts` |
| H14 | Suite | 0 failures; artifact-free collection unchanged; frozen evidence untouched; no edits to program.md/research/ |
| H15 | Review | Astra light + Claude |
