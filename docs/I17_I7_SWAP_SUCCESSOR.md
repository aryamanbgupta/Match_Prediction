# I17 — Swap augmentation on the I7 identity frame (production-successor candidate)

**Status: PRECOMMITTED 2026-07-30, before any swap-arm training.** This
document freezes the design, metrics, and decision rules first; results are
appended only after the runs complete (I9 discipline).

## Why this experiment

1. **The legacy production line is operationally dead-ended.** The serving
   state cache for `xgb_match_v3_m7_swap_production` ends 2026-04-16 and
   cannot be regenerated: all current cache/materializer tooling fails
   closed without a venue-identity declaration, and the legacy replay
   contract is deliberately not extendable. `predict_fixture.py` already
   fails its >14-day staleness gate for live fixtures. Only the I7
   identity stack has a working fresh-state build path (proven through
   2026-07-13 by the Hundred operation).
2. **D12 swap augmentation is validated on the legacy frame only** (better
   LL and ROI on 5/5 paired seeds; adopted into production 2026-07-30).
   Whether it transfers to the I7 frame is a hypothesis, not a result.
3. Therefore the next production candidate is: **M7 config + swap
   augmentation trained on the I7 identity frame**, evaluated with the
   same paired-seed discipline as D12.

## Design (frozen)

- **Frame**: `data/xgb_match_data_i7` — 7,972 train / 528 val / 798 test
  rows, `venue_identity.json` = `venue_aliases_v1`
  (sha256 `853b32b0…`, 94 active aliases). 49 model features: the 46
  legacy player/team/venue features plus `venue_id_encoded` and
  `competition_tier_encoded`. `_swap_frame` verified to classify all 54
  frame columns (identity columns are swap-invariant; no code change).
- **Arms** (paired by seed):
  - `base`: `--monotone`, trainer defaults (M7 config: lr 0.05,
    colsample 0.9, depth 4, n=1000, early stop 30).
  - `swap`: base + `--swap-augment`.
- **Seeds**: A1 set {7, 13, 29, 42, 101}.
- **Command**:
  `uv run python scripts/xgboost_match_v1.py --cmd both
  --data-dir data/xgb_match_data_i7 --model-dir models/auto/i17/<arm>_seed<s>
  --monotone [--swap-augment] --seed <s>`
- **Determinism check**: `models/xgb_match_i7` is already base/seed29
  (verified: lr 0.05, cs 0.9, seed 29, monotone on). The freshly trained
  `base_seed29` must reproduce its test predictions (max |Δp| ≈ 0);
  a mismatch invalidates the run until explained.
- **Evaluation chain**: the 2026-04-25 iteration envelope predates I15 and
  carries no `cricsheet_id`, and 3 of its 261 display ids (SCG,
  Kingsmead, Wanderers fixtures) no longer match i7 display ids after
  venue canonicalization. Fix: `scripts/patch_envelope_cricsheet_ids.py`
  derives each envelope entry's cricsheet stem from
  `data/polymarket_test/*.json` (date+teams+venue display-id
  reconstruction; fails closed on ambiguity) and writes
  `eval_out/i17/hier_all_cricsheet.json`. Then per model:
  `blend_eval_json --w 0.0` (pure direct) → `reslice_eval_json` vs
  `betting_odds_polymarket.json` at min-volume 0 / 50k / 100k.
  **All 261 fixtures must join** (the blend consumed-count is checked);
  any drop fails the run.

## Metrics and decision rules (frozen)

- **Metric of record**: iteration ≥$50k log loss and flat ROI, per seed
  (point values). The seed-29 pair additionally gets the full I3
  `tournament_time_block_v1` block-CI readout (10k seed-42 resamples).
- **Precommitted noise floors** (A1): 0.007 LL, 2.3pp ROI.
- **Rule 1 — D12 transfer confirmed** iff swap beats base on ≥$50k LL on
  ≥4/5 seeds AND the mean paired ΔLL improvement ≥ 0.007. ROI is
  reported and directionally supportive but does not gate (its floor is
  wide relative to plausible effects at n=261).
- **Rule 2 — successor-candidacy readout** (descriptive, not a gate): the
  better i7 arm is compared against frozen legacy production
  (`xgb_match_v3_m7_swap_production`, ≥$50k LL 0.6215) on the same
  slice. The i7 line does **not** have to beat the legacy line — its
  rationale is operational (only line with a fresh-state path) — but if
  it trails by > 0.02 LL on ≥$50k that is a named blocking concern for
  any promotion discussion.
- **Rule 3 — no golden, no forward.** The extended-golden sidecar is
  legacy-semantics and invalid for i7-identity models; an i7 golden
  frame is follow-up work, not part of I17. The consumed forward set is
  never scored.
- **Outcome vocabulary**: ADOPT-CANDIDATE (Rule 1 passes → swap-on-i7
  becomes the successor candidate config), NEUTRAL (Rule 1 fails on
  magnitude but ≥3/5 seeds agree → keep base-i7 as candidate, swap
  unproven on this frame), REFUTED (base beats swap on ≥4/5 seeds).
  Promotion to production is a separate future decision requiring the
  operational cutover plan; nothing in I17 authorizes it.

## Outputs

- `models/auto/i17/{base,swap}_seed{7,13,29,42,101}/`
- `eval_out/i17/` (patched envelope, blends, slices)
- `reports/i17_i7_swap_eval_20260730.md` (results + verdict)
- Verdict row appended to `research/results.tsv`

---

## Results (appended 2026-07-30, same day, after the precommitted runs)

- **Determinism gate passed**: fresh `base_seed29` reproduces the archived
  `models/xgb_match_i7` test predictions at max |Δp| = 0.0 on all 778
  uniquely display-matched rows (the archived file's 788 keys lose 10 rows
  to pre-I15 doubleheader collisions; the new writer keys by stem, 798).
- **Join gate passed**: all 10 blends consumed 261/261 envelope fixtures
  via the stamped `cricsheet_id` (including the 3 venue-renamed misses).

Iteration set, per seed (LL = avg log loss, ROI = flat %):

| arm | seed | LL all | LL ≥50k | ROI ≥50k | LL ≥100k | ROI ≥100k |
|---|---:|---:|---:|---:|---:|---:|
| base | 7 | 0.6392 | 0.6497 | +12.15 | 0.6173 | +14.17 |
| base | 13 | 0.6394 | 0.6474 | +18.55 | 0.6141 | +23.64 |
| base | 29 | 0.6356 | 0.6421 | +17.49 | 0.6067 | +24.27 |
| base | 42 | 0.6383 | 0.6431 | +18.98 | 0.6078 | +24.68 |
| base | 101 | 0.6387 | 0.6429 | +20.89 | 0.6102 | +25.63 |
| swap | 7 | 0.6265 | 0.6348 | +19.81 | 0.5949 | +21.98 |
| swap | 13 | 0.6217 | 0.6276 | +23.37 | 0.5872 | +23.98 |
| swap | 29 | 0.6187 | 0.6262 | +20.54 | 0.5886 | +21.95 |
| swap | 42 | 0.6251 | 0.6296 | +23.29 | 0.5902 | +23.75 |
| swap | 101 | 0.6289 | 0.6348 | +26.98 | 0.5956 | +28.09 |

Slice-matched market LL: 0.6267 (all, n=255 priced), 0.6482 (≥50k),
0.6224 (≥100k).

- **Rule 1 — D12 transfer CONFIRMED**: swap beats base on ≥$50k LL on
  **5/5 seeds**, mean paired ΔLL **−0.0144** (floor 0.007; the legacy-frame
  D12 effect was −0.0092, so the transfer is *larger* on the I7 frame).
  Same 5/5 pattern on the all and ≥$100k slices (mean ΔLL −0.0141 /
  −0.0199). ROI supportive: mean ΔROI +5.18pp on ≥$50k, positive on 4/5
  seeds.
- **Seed-29 I3 readout (≥$50k, 19 blocks, reliable)**: swap LL 0.6262,
  ROI +20.54% [−5.56, +43.47], 168 bets, win 51.2%; base LL 0.6421,
  ROI +17.49% [−12.37, +40.59]. CIs straddle zero — no betting-edge claim.
- **Rule 2 — successor readout**: swap-i7 5-seed mean ≥$50k LL 0.6306
  (seed29 0.6262) vs legacy production `xgb_match_v3_m7_swap_production`
  0.6215 — the i7 line trails by ~0.005–0.009, well inside the 0.02
  blocking threshold. Swap-i7 beats the slice-matched market LL (0.6482)
  on 5/5 seeds; base only on 4/5.

**Verdict: ADOPT-CANDIDATE.** Swap + M7 config on the I7 identity frame is
the designated production-successor configuration. Production is
unchanged; promotion requires the operational cutover plan (fresh-state
serving, i7 golden frame audit) as a separate decision.

Report: `reports/i17_i7_swap_eval_20260730.md`.
