# Paired experiment harness

Run a configured match-model comparison from the repository root:

```bash
uv run --no-sync python scripts/experiment_harness.py run experiments/harness/swap_smoke.yaml
```

Use `--dry-run` to replace XGBoost with deterministic predictions and build
temporary odds/cluster evidence under `out_dir`. Use `--allow-long` only after
reviewing a run whose `len(seeds) * expected_minutes_per_seed` exceeds
`ceiling_minutes`.

## Configuration

`name`, `frame`, `baseline`, `candidate`, `expected_minutes_per_seed`, and
`out_dir` are required. `frame` is either a directory or a role from
`models/MANIFEST.yaml`. A manifest role is accepted only after
`artifacts.verify_role` recomputes its directory content hash and returns
`OK`; `MISMATCH` and `MISSING` are named configuration errors. The recomputed
content hash, not an unverified declaration, enters the harness configuration
hash. Each arm has `trainer_args`, expressed as a YAML list
of command-line tokens or as a flag/value mapping. `candidate.kind` is
`trainer` or `ensemble`.

`seeds` defaults to `[29, 7, 13, 42, 101]`. An integer `N` selects the first
`N` entries from that ladder (for example, `seeds: 3` means `[29, 7, 13]`);
an explicit list selects those seed values.
`slices` defaults to `[all, 50000, 100000]` and must include `50000` for the
gate. `cost` accepts `spread_bps`, `fee_bps`, and `fee_basis` (`winnings` or
`stake`). `ceiling_minutes` defaults to 240 and `odds_role` defaults to
`odds_iteration_v2`.

Optional `verify` entries are `diff_negation_identity`, `h2h_prior_half`, and
`swap_doubling`. Optional rolling-origin `folds` entries each require
`train_until`, `select_from`, and `select_until`; the harness rebuilds the
train/select parquet split and refits encoders for every fold.

Trainer arguments may not set or abbreviate harness-owned options: `--cmd`,
`--seed`, `--model-dir`, `--data-dir`, `--fit-encoders-on`, `--config-json`, or
`--early-stop-before`. Exact, `--option=value`, and argparse-style prefix forms
are rejected during configuration loading.

## Contract and outputs

For each arm and seed, the harness trains with fold-local encoders, blends at
direct weight `w=0`, reslices against the odds role's registered prices and
cluster directory, and stamps the arm, seed, and configuration hash on
predictions, blended output, and slices. After all outputs are complete it
writes `<model_dir>/harness_run.json`. That completion record binds the arm,
seed, configuration hash, effective trainer argv and its SHA-256, `model.pkl`
SHA-256 (or the dry-run stub identity), `test_predictions.json` SHA-256, and
every configured sliced JSON SHA-256. Resume recomputes every digest and
requires matching arm/seed/hash summaries in predictions, blended output, and
all slices. Swapped arm directories, modified predictions/models/slices, or
changed effective trainer arguments therefore retrain. The configuration hash
binds canonical YAML (excluding
`out_dir`), the resolved frame path, the frame's manifest hash (or directory
hash), and the trainer script SHA-256. A real run resumes only when `model.pkl`
is present and the output is not a dry-run stub; dry-run and real-run artifacts
never resume each other. Pairing is checked before the shared claim gate is
called.

Every evaluation synthesizes a direct-only envelope from the prediction rows
under test and the registered odds. Recorded simulation envelopes are never
used, and the blended match-id set must equal the prediction set. The registered
cluster source directory is passed explicitly to both reslicing and the gate.

For `candidate.kind: ensemble`, only candidate members are trained. The
baseline is always the immutable `match_model_prod` artifact from
`models/MANIFEST.yaml`, regardless of any baseline role named in the config.
Before its existing `test_predictions.json` is read or copied,
`artifacts.verify_role` must return `OK`; `MISMATCH` and `MISSING` refuse the
run. The evaluation copy is stamped `prod_seed29` with the verified production
hash and the source file's SHA-256, while the source prediction summary is
preserved under `source_summary`; it is never retrained.

`calibrate_match_predictions.py --calib-after DATE` is accepted only when the
model's `train_metrics.json` records the same `early_stop_before` date and a
disjoint `early_stop_match_ids` set. Models trained without the paired early
stopping boundary cannot use `--calib-after`.

`gate.json` is the gate's replayable decision. `harness.json` contains config,
hash, per-seed/slice LL and ROI rows, fold means, ensemble diagnostics, and the
verdict. The harness prints the exact `log_verdict.py verdict ... --gate-json`
command but never runs it.

The harness decides only the pre-registered iteration-set claim. Provisional
status, estimators, confidence intervals, the LL floor, profit safety, and
descriptive demotions belong to `claim_gate.py`. Fold means and ensemble
per-seed comparisons are diagnostics; production re-selection and promotion
remain human decisions.
