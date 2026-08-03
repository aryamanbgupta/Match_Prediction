# I18 — i7-identity golden frame + swap-i7 golden audit (I17 promotion gate)

## Idea, hypothesis, gate

**Idea id:** I18 (P1, claimed `d3e4d64`, 2026-07-31T01:35:54Z).

**What this is:** build the golden (124-fixture) match frame under the I7
identity contract and score the FIXED I17 successor candidate on it. This is
a *gate readout for a human promotion decision*: golden is audit-only, the
candidate is pre-specified (`models/auto/i17/swap_seed29`, control
`models/auto/i17/base_seed29`), and NOTHING about the result feeds model
selection or production. The verdict is **DESCRIPTIVE regardless of the
numbers** (golden slices are ≤11 blocks). You produce the readout; you do
NOT adopt, promote, or revert anything.

**Gate pair (instrumentation-style, like A1/B4/D10):**
- Gate A (integrity): all 124 golden odds rows join by `cricsheet_id`;
  `verify_forward_holdout` passes with zero overlap; no write touches
  production caches, `data/golden/` inputs, or `data/forward_holdout/`;
  re-materialized train/val/test are content-identical to the existing
  `data/xgb_match_data_i7/` splits.
- Gate B (readout): both arms scored on golden at all/≥$50k/≥$100k with I3
  tournament blocks, numbers reported verbatim.

## Golden-access authorization (read this before the DO NOT CHEAT section scares you)

`program.md` rule 2 forbids the loop from touching golden. The supervisor
queued I18 on 2026-07-30 (commit `bdb1131`) with an explicit, scoped
exception — the entry begins "**Night-executable.**" and states "golden is
audit-only, the candidate is pre-specified, and NOTHING about the result
feeds model selection or production". That authorization covers, for this
idea only:

- **READ-ONLY** access to `data/golden/t20s_json/` (materialization input),
  `data/golden/betting_odds_golden.json` (odds), and
  `data/golden/polymarket_test/` (cricsheet-id stamping input).
- **ZERO writes** anywhere under `data/golden/` or `data/forward_holdout/`.
  All outputs go to `data/auto/i18/`, `models/auto/i18/`, and (one file,
  after the parity gate passes) `data/xgb_match_data_i7/golden_test.parquet`.
- No selection: both model artifacts are frozen inputs; you never pick
  among candidates based on golden numbers.

Everything else in program.md's DO NOT CHEAT section applies in full.

## Context you'd otherwise have to re-derive

- The I7 stack: `models/player_stats_cache_i7.sqlite` (cache), full frame
  materialized by `materialize_match_features.py --version i7`, then
  subset to the exact 48-column M7 production feature contract by
  `scripts/build_i7_match_frame.py` (it processes a `golden_test` split
  automatically if present in its `--source-dir`, and fails closed if
  `venue_identity.json` / the feature contract is missing). See
  `docs/I7_VENUE_IDENTITY_CONTRACT.md` § "Required rebuild order".
- Split assignment is by date (`classify_split`; golden_start =
  2026-04-17 in `loaders_common.py`), so merging the golden cricsheet pool
  via `--extra-source-dir` automatically lands those matches in
  `golden_test.parquet`. Golden matches (2026-04-17 → 2026-06-17) all
  post-date the test split (ends 2026-04-16), so train/val/test rows must
  come out IDENTICAL to the existing frame — that identity is Gate A's
  parity check, and it is what makes copying only `golden_test.parquet`
  into `data/xgb_match_data_i7/` safe.
- The join chain (this is the part that silently breaks if done naively):
  golden odds rows carry legacy display `match_id`s built from RAW venue
  strings and have NO `cricsheet_id` field. The i7 frame canonicalizes
  venues, and `predict_golden.py` keys its output by `cricsheet_id` when
  the parquet has that column (it does). So:
  - `synthesize_golden_envelope.py` produces an envelope keyed by legacy
    display ids;
  - `scripts/patch_envelope_cricsheet_ids.py --test-dir
    data/golden/polymarket_test` stamps `cricsheet_id` onto every envelope
    entry (124 files there, named by cricsheet id; the tool FAILS CLOSED
    on unmatched/ambiguous — that failure mode satisfies Gate A, do not
    work around it);
  - `blend_eval_json.py` joins envelope→direct via
    (match_id, cricsheet_id, display_match_id), so the stamped envelope
    joins the cricsheet-keyed predictions exactly;
  - `reslice_eval_json.py` indexes the blended rows by all three id fields,
    so the odds rows re-join via their legacy display ids. Pass
    `--cluster-source-dir data/golden/t20s_json` to get I3
    tournament-block CIs (contract `tournament_time_block_v1`).
  This mirrors `reports/golden_extension_eval_20260730.md` § Reproduce and
  `reports/i17_i7_swap_eval_20260730.md` § Reproduce.
- `predict_golden.py` maps unseen venues to a fallback encoder class and
  prints a WARN listing them — capture that list in result.md (the legacy
  extension audit had 5 unseen associate-nation venues; expect something
  similar).
- Its printed "standalone LL" covers ALL golden-frame rows (every
  golden-window match, not just the 124 odds-matched ones). Report it as
  context but the audit numbers are the resliced ones.

## Steps (implement in this order)

### 0. Preconditions (fail closed on any miss)
```bash
ls models/auto/i17/swap_seed29/model.pkl models/auto/i17/base_seed29/model.pkl
uv run python scripts/verify_forward_holdout.py data/forward_holdout/2026-06-01_2026-07-13
```
- If either i17 model dir is missing/incomplete, regenerate BOTH arms with
  the exact commands in `reports/i17_i7_swap_eval_20260730.md` § Reproduce
  (seed 29 only). They currently exist — regeneration should not be needed.
- Determinism spot-check (cheap, protects against artifact drift since
  I17): compare `models/auto/i17/base_seed29/test_predictions.json` vs
  `models/xgb_match_i7/test_predictions.json` — max |Δp_team1| over shared
  keys must be exactly 0.0. Print the number.
- Record the `verify_forward_holdout` fingerprint + zero-overlap line
  verbatim into `research/handoff/I18/raw/`.

### 1. Materialize the i7 full frame WITH the golden pool (~30–60 min)
```bash
uv run python scripts/materialize_match_features.py \
  --version i7 \
  --extra-source-dir data/golden/t20s_json \
  --out-dir data/auto/i18/i7_full_golden
```
Defaults otherwise (source-dir default corpus, no `--freeze-trackers-after`
— unfrozen matches how `data/xgb_match_data_i7_full` was built, baseline
ELO contract). Do NOT write to `data/xgb_match_data_i7_full`.
Tee output to `research/handoff/I18/raw/materialize.log`.

### 2. Subset to the production 48-column frame
```bash
uv run python scripts/build_i7_match_frame.py \
  --source-dir data/auto/i18/i7_full_golden \
  --out-dir data/auto/i18/frame
```
Must print a `golden_test` line; record the row count.

### 3. Parity gate (Gate A core) — write `scripts/auto/i18_frame_parity.py`
Compare `data/auto/i18/frame/{train,validation,test}.parquet` against
`data/xgb_match_data_i7/{train,validation,test}.parquet`: same shape, same
columns in the same order, and content-equal (pandas `.equals()` after
`reset_index(drop=True)`; if it fails, print which columns differ and the
max abs numeric delta). Also require `venue_identity.json` in
`data/auto/i18/frame` == `data/xgb_match_data_i7/venue_identity.json`, and
that `golden_test.parquet` carries `cricsheet_id` and
`match_identity_version` columns.
- ALL equal → copy exactly ONE file:
  `cp data/auto/i18/frame/golden_test.parquet data/xgb_match_data_i7/golden_test.parquet`
  Touch nothing else in that directory.
- ANY mismatch → STOP. Do not copy. Write result.md describing the
  mismatch; that is a Gate A failure for the orchestrator to judge.

### 4. Envelope + stamp (Gate A join)
```bash
uv run python scripts/synthesize_golden_envelope.py \
  --odds data/golden/betting_odds_golden.json \
  --out models/auto/i18/golden_envelope.json
uv run python scripts/patch_envelope_cricsheet_ids.py \
  --envelope models/auto/i18/golden_envelope.json \
  --test-dir data/golden/polymarket_test \
  --out models/auto/i18/golden_envelope_cricsheet.json
```
Must report 124 entries stamped. If it exits with unmatched/ambiguous
entries, STOP and report (Gate A fail-closed).

### 5. Score both arms (Gate B)
For `arm` in `swap_seed29` (candidate), `base_seed29` (control):
```bash
uv run python scripts/predict_golden.py \
  --model-dir models/auto/i17/<arm> \
  --parquet data/xgb_match_data_i7/golden_test.parquet \
  --out-json models/auto/i18/<arm>_golden_predictions.json
uv run python scripts/sim_eval/blend_eval_json.py \
  --sim-json models/auto/i18/golden_envelope_cricsheet.json \
  --direct-json models/auto/i18/<arm>_golden_predictions.json \
  --w 0.0 --out-dir models/auto/i18/<arm>
# then reslice the w0p00 output 3× (all / >=50k / >=100k):
uv run python scripts/sim_eval/reslice_eval_json.py \
  --in models/auto/i18/<arm>/golden_envelope_cricsheet_w0p00.json \
  --odds data/golden/betting_odds_golden.json \
  --out-dir models/auto/i18/<arm>/sliced_all \
  --cluster-source-dir data/golden/t20s_json
# repeat with --min-volume 50000 --out-dir .../sliced_50k
# repeat with --min-volume 100000 --out-dir .../sliced_100k
```
(If the blend output filename differs, use whatever `--w 0.0` actually
wrote — it is `<sim-json-stem>_w0p00.json`.)
JOIN CHECK: the blend step must consume all 124 envelope rows against
direct predictions (its printed blended/passthrough counts — passthrough
must be 0). If any row passes through unblended, STOP and report which.
Tee ALL raw output to `research/handoff/I18/raw/` (one file per command or
one big log; do not summarize it away).

### 6. result.md
Write `research/handoff/I18/result.md` with, copied VERBATIM from tool
output for each arm × slice {all, ≥50k, ≥100k}: n matches/bets, Avg Log
Loss, market LL, Flat ROI, ROI CI + n blocks (I3), win rate. Plus: the
determinism spot-check number, verify_forward_holdout fingerprint, unseen
venue list, golden_test row count, parity-gate outcome, commit SHAs you
created, `git diff --stat` vs the claim commit `d3e4d64`, wall-clock of the
materialization, and anything that crashed or ran long.

## Commit discipline

- Commit the harness (`scripts/auto/i18_frame_parity.py` and any small
  helper) BEFORE running the materialization/eval:
  `Auto[I18]: i7 golden frame build + parity harness`.
- `data/` and `models/` are gitignored — artifacts stay untracked; the
  copied `golden_test.parquet` is untracked too. That is expected.
- Further commits only for handoff files (`research/handoff/I18/...`) if
  you want checkpoints; prefix everything `Auto[I18]:`.

## Easy to get wrong

1. **Never write under `data/golden/`** — synthesize_golden_envelope's
   DEFAULT `--out` is `data/golden/golden_sim_envelope.json`; you MUST pass
   the explicit `--out` above. Same style of care everywhere: every output
   path must be under `data/auto/i18/`, `models/auto/i18/`,
   `research/handoff/I18/`, or the single sanctioned
   `data/xgb_match_data_i7/golden_test.parquet` copy.
2. Do not touch `data/xgb_match_data_i7/{train,validation,test}.parquet`,
   `data/xgb_match_data_i7_full/`, any production model dir, or
   `models/player_stats_cache_*.sqlite`. The materializer writes ONLY to
   its `--out-dir`; if any step asks to rebuild a cache, something is
   wrong — stop.
3. Do not "fix" join failures by editing odds/envelopes by hand or fuzzy
   matching — the patch tool failing closed IS the designed behavior;
   report it.
4. `--version i7` (not v3) on the materializer, or you get legacy identity
   and the frame build fails (or worse, silently joins wrong venues).
5. No `--freeze-trackers-after` flag here (the i7 frame is unfrozen;
   recipe-A's `--freeze-trackers-after 2025-06-30` guidance is for the
   legacy v2_clean frame and does NOT apply).
6. The eval framework (`scripts/sim_eval/`), `parsing_v2.py`,
   `stats_provider.py`, `stats_sqlite_backend.py` are untouchable, as
   always. `predict_golden.py`, `synthesize_golden_envelope.py`,
   `patch_envelope_cricsheet_ids.py` are also to be used AS-IS — if one
   errors, capture the error and stop rather than patching it.
7. ROI CIs: quote the I3 tournament-block CI printed by reslice, never an
   i.i.d. interval; note blocks count per slice (<10 blocks = descriptive,
   which is expected here).
8. You do NOT decide the verdict, do not revert anything, do not touch
   `research/results.tsv` or `research/IDEAS.md`, do not `git push`, do
   not start any other idea.

## Baseline / comparison rows (for the report, NOT for any decision)

- Legacy-line golden extension (2026-07-30, `xgb_match_v3_m7_swap_production`
  vs frozen base, same 124 fixtures, legacy sidecar features), from
  `reports/golden_extension_eval_20260730.md`:
  - all (n=124): swap LL 0.5831, base LL 0.5916, market 0.5513;
    swap ROI −4.35% [−39.4, +13.0], 11 blocks
  - ≥$50k (n=75): swap LL 0.6685, base 0.6736, market 0.6573;
    swap ROI +9.69% [−46.9, +19.5], 9 blocks
  - ≥$100k (n=66): swap LL 0.6938, base 0.6964, market 0.6843;
    swap ROI +14.38% [−85.1, +20.3], 5 blocks
- I17 iteration-set numbers (i7 frame, 261 fixtures): swap-i7 5-seed mean
  ≥$50k LL 0.6306 / ROI +22.80% vs base-i7 0.6450 / +17.61%; market 0.6482.
  Seed-29 swap ROI CI [−5.56, +43.47] (19 blocks).
These are context lines for the descriptive comparison in result.md.
