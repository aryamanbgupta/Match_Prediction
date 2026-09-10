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

## Harness validation run and candidate A checks (written before running)

| # | Check | Pass condition |
|---|---|---|
| HA0 | Harness reproduces a known result | `experiments/harness/swap_smoke.yaml` (base vs swap, M7 config, five seeds, zero-cost decision) on this laptop; the gate's paired ≥$50k ΔLL must be favourable on 5/5 seeds with mean within seed noise of the I17 record (−0.0144, `docs/I17_I7_SWAP_SUCCESSOR.md`); verdict LANDED or TABLED (profit is noise); no fallback clusters |
| HA1 | Candidate A config | `seed_ensemble.yaml`: candidate = logit mean of five swap-config seeds; baseline = the verified production artifact's own predictions; decision at zero cost; slices all / ≥$50k / ≥$100k |
| HA2 | Estimand | gate runs under the 1-seed rule, so the best verdict is PROMISING; the 0.007 point floor applies; per-seed diagnostics reported only |
| HA3 | Record | harness.json + gate.json under `experiments/results/`; IDEAS.md entry and results.tsv row written only through `log_verdict --gate-json` with the reassess/queue-confirm semantics; the verdict is whatever the gate says |
| HA4 | Resources | run on this laptop (frame 8.6 MB in memory); seeds sequential; no mini load |

### HA0 result (2026-09-10): harness reproduces the I17 swap transfer

`swap_smoke.yaml` on this laptop (base vs swap, M7 config, seeds 29/7/13/42/101,
zero-cost decision, registered iteration odds v2). Gate: **LANDED**, five
non-provisional seeds, ≥$50k paired ΔLL **−0.0135** [−0.0256, −0.0010],
Δprofit +0.041 [−0.0006, +0.083], 18 tournament blocks, 18 profit blocks,
zero cluster fallbacks, n=167 with 5 dropped symmetrically; per-seed rows in
`experiments/results/item6_swap_smoke/harness.json`. I17 recorded −0.0144 on
the same comparison; the difference is inside the 0.007 seed floor. The run is
a harness validation, not an idea: no IDEAS.md or results.tsv entry.

Harness fix found by the run: the direct-only envelope refused a test frame
with more matches than the registered odds cover (798 vs 255). It now
evaluates the covered set and stamps `coverage`, matching the blend/reslice
chain; the id-set assertion compares against the covered set.

### Candidate A result (2026-09-10): FAILED

Idea R1, the first verdict decided by the gate. Ensemble ≥$50k LL 0.6289 vs
the deployed seed-29 model 0.6249: paired ΔLL **+0.0040 [+0.0023, +0.0076]**,
unfavourable with the interval excluding zero; Δprofit +0.030
[+0.004, +0.052] is noise on 167 bets. Members: 0.6249 / 0.6266 / 0.6284 /
0.6336 / 0.6336; the ensemble beats the member mean (0.6294) by a Jensen
sliver but not the deployed seed. Because the iteration set was used for
selection, part of seed 29's edge is selection luck; any future ensemble
claim should be confirmed on the golden set rather than retried here. Logged
via `log_verdict --gate-json` (results.tsv carries the gate sha256);
evidence copied to `research/handoff/R1/`. HA1–HA4 met.

## Candidate B checks (written before running)

| # | Check | Pass condition |
|---|---|---|
| HB1 | Harness `candidate.kind: calibrated` | trains the candidate arm with `--early-stop-before <date>`, then applies `calibrate_match_predictions.py --calib-after <date> --method platt` to its test predictions; the baseline arm is the same trainer args without calibration (raw, as served); leakage test H9 applies per seed |
| HB2 | Date | `calib_after` chosen from the validation split's date range so both halves are non-empty; recorded in the config; the same date for every seed |
| HB3 | Decision | five seeds, zero cost, registered odds v2; the gate decides; P2's 2026-08-07 re-look (val-fit Platt sharpens, a=1.107, point-favourable, CIs straddling zero) is the prior, not evidence |
| HB4 | Record | R2 in IDEAS.md, verdict only through the gate, evidence copied to `research/handoff/R2/` |

### Candidate B result (2026-09-10): FAILED

Idea R2. Five non-provisional seeds; paired ≥$50k ΔLL **−0.0068
[−0.0230, +0.0125]**, favourable on 5/5 seeds but the interval straddles
zero; Δprofit noise. The effect is consistent and just under the seed floor,
which 18 blocks cannot resolve; serving stays raw. HB1–HB4 met; evidence in
`research/handoff/R2/`.
