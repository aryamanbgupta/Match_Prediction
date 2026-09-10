# Item 6 acceptance checks (paired multi-seed experiment harness)

## Fix pass 2 (2026-09-10)

Astra's three blocking provenance findings, independently verified by Claude,
are locked by tests:

- J1: each arm/seed writes `harness_run.json` only after evaluation completes.
  Resume requires the recorded arm, seed, config hash, effective trainer argv
  SHA-256, model/stub SHA-256, prediction SHA-256, every sliced-output SHA-256,
  and matching arm/seed/config summaries. Swapping complete baseline/candidate
  directories or editing `test_predictions.json` forces retraining.
- J2: a manifest-backed frame is content-verified through
  `artifacts.verify_role`; `MISMATCH`/`MISSING` are named `frame` errors and the
  recomputed content hash is used in config identity. A parquet mutation with
  an unchanged manifest is refused.
- J3: the production baseline role is verified before its predictions are
  copied. `MISMATCH`/`MISSING` are named `baseline.role` errors; the copy
  records the source file SHA-256 and retains its summary under
  `source_summary`. A tampered production prediction file is refused.

## Fix pass (2026-09-10)

Astra/Claude review findings I1–I7 and I9–I11 are locked into the harness:
harness-owned trainer options reject exact, abbreviated, and equals forms;
fresh real trainer output requires `model.pkl`; stub and real resume identities
are distinct; config identity includes the frame and trainer bytes; ensemble
baseline evidence is copied from manifest role `match_model_prod` and never
retrained; serving uses the trainer's shared unknown-category `-1` encoder;
evaluation always uses a synthesized direct-only envelope with exact match-id
coverage; calibration verifies the recorded early-stop boundary and disjoint
IDs; integer seeds mean the first N ladder entries; the registered cluster
directory is explicit at the gate; and blended output carries seed/hash stamps.

Written 2026-09-10 by Claude before implementation. Steps 1–2 (harness,
schema, tests, verify hooks) plus the trainer's fold-local encoder contract
are in scope now; candidates A–D (steps 3–6) each get a check block appended
before they run, and they run only after the harness is reviewed.

| # | Check | Pass condition |
|---|---|---|
| H1 | `scripts/experiment_harness.py run <config.yaml>` | config names frame (manifest role or explicit dir), baseline and candidate trainer args, seed list (default 29, 7, 13, 42, 101; `seeds: [29]` and `seeds: 3` → `[29, 7, 13]` accepted), slices, cost model, `expected_minutes_per_seed`, ceiling; schema validated with named errors |
| H2 | Per-seed artifacts | `<out>/<arm>_seed<N>/` holds the trained model dir, `test_predictions.json`, blended and sliced JSONs whose summary carries arm, `model_seed` (the gate's stamp), and `config_hash`; `harness_run.json` binds arm/seed/config, effective argv, model, predictions, and every slice by SHA-256; only fully reverified seeds resume, while swapped directories, edited artifacts, or a changed hash force retrain |
| H3 | Gate call | the harness calls `claim_gate.decide` with the registered odds role and cluster dir, writes `gate.json` and one `harness.json` with per-seed rows and the Verdict; it never passes cluster labels; the result is logged only via `log_verdict --gate-json` (harness prints the command, does not run it) |
| H4 | Budget guard | seeds × expected minutes > ceiling refuses without `--allow-long`; test |
| H5 | Seed rule | provisional stamping and estimator come from the gate, not the harness; a 1-seed and a 3-seed dry run show `provisional: true` |
| H6 | Pairing | both arms' `test_predictions.json` must carry identical match-id sets, else the harness refuses (test) |
| H7 | Ensemble estimand | `candidate.kind: ensemble` averages the five candidate models in logit space into ONE prediction set; the baseline is the verified existing `match_model_prod/test_predictions.json`, stamped `prod_seed29`, the verified manifest hash, source file SHA-256, and `source_summary`, and is never retrained; role `MISMATCH`/`MISSING` refuses before copying; the gate runs under the 1-seed rule (provisional, 0.007 floor); member diagnostics are reported but not decided; test with a temporary stub manifest role |
| H8 | Fold-local encoders | `xgboost_match_v1.py --fit-encoders-on {train,all}`; harness default `train`, legacy CLI default `all`; unseen venue/tier maps to code −1 at transform time; test trains without one venue and predicts a row at it with code −1 and the degraded flag; leakage test: no test-fold venue in a fold's encoder vocabulary unless in that fold's training rows |
| H9 | Calibration split | `calibrate_match_predictions.py --calib-after <date>` reads `train_metrics.json`, requires the identical recorded `early_stop_before`, and refuses missing early-stop training or any calibration/early-stop match-id overlap |
| H10 | Rolling-origin folds (candidate C support) | `folds:` config as row masks by `match_date` over the existing frame, encoders refit per fold; fold-mean LL reported; production re-selection is a separate, human decision |
| H11 | Verify hooks | `d12_run.verify` frame checks ported as optional `verify:` entries (diff-negation identity, h2h prior 0.5, swap doubling) that run before training |
| H12 | Dry run | `--dry-run` on the mini corpus + a tiny frame completes end to end in under 2 minutes without a real model (uses a stub trainer) and exercises H2–H7 |
| H13 | Tests | `scripts/tests/test_experiment_harness.py` covers H1–H12 on tmp trees; no real artifacts unless `needs_artifacts` |
| H14 | Suite | 0 failures; artifact-free collection unchanged; frozen evidence untouched; no edits to program.md/research/ |
| H15 | Review | Astra light + Claude |

## Review closure, steps 1–2 (Claude, 2026-09-10)

Two Astra rounds. Round 1: eleven findings (trainer-arg bypass, stub resume,
hash ignoring resolved inputs, ensemble baseline retrained, serving unseen
code, mixed-estimand blend, calibration boundary unchecked, seeds:N,
cluster dir, stamps) fixed by Sol with locking tests; the atlas-file finding
was out of scope. Round 2: three provenance findings (resume unbound to
arm/model, role frames trusted by declared hash, production baseline
unverified) fixed by Sol with locking tests. Claude recorded the serving
encoder change as an intentional change in IMPROVEMENTS.md. Suite: 496
passed, 0 failed. Candidates A–D follow with their own check blocks.
