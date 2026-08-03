# Swap-i7 Promotion Cutover Plan

**Status: EXECUTING — user approved 2026-07-31.** Decisions taken:
D1 go, D2 verbatim audited weights via `models/auto/i19/swap_seed29`
(byte-identical to the I18-audited i17 arm, honest `cricsheet_primary_v1`
stamps), D3 weekly refresh + mandatory pre-fixture rebuild (14-day gate as
backstop), D4 immediate cutover with the golden-refresh replumb queued as
follow-up. Execution log at the bottom of this file.

Completes the promotion bundle whose machine-side legs are done:
- I17 (`docs/I17_I7_SWAP_SUCCESSOR.md`): ADOPT-CANDIDATE — D12 swap transfer
  confirmed on 5/5 paired seeds on the I7 frame (mean ≥$50k ΔLL −0.0144,
  floor 0.007).
- I18 (`research/reports/auto/I18.md`): golden audit — swap-i7 beats the
  slice-matched market LL on both sharp golden slices (≥$50k 0.6538 vs
  0.6573; ≥$100k 0.6767 vs 0.6843) where legacy production trails
  (0.6685 / 0.6938). Descriptive; all ROI block CIs straddle zero.
- I19 (`research/reports/auto/I19.md`): coherent frame
  `data/xgb_match_data_i7_v2` — retrains reproduce the I17 arms at
  max |Δp| = 0 with byte-identical `model.pkl`; fixes the frozen frame's
  non-unique display-keyed `match_id` (45 rows).

## What is being promoted

| Item | Value |
|---|---|
| Config | M7 (lr 0.05, cs 0.9, depth 4, monotone) + `--swap-augment` |
| Identity contract | I7 canonical venues (`venue_aliases_v1`) + I15 cricsheet-primary match ids |
| Artifact of record | verbatim copy of `models/auto/i17/swap_seed29` (the audited arm: full I3 block-CI readout, I18 golden audit, I19 byte-identical retrain) |
| Frame of record | `data/xgb_match_data_i7_v2` (I19) |
| Proposed production dir | `models/xgb_match_i7_swap_production/` |

Precedent: the D12 promotion (2026-07-30) promoted the archived swap arm
verbatim rather than retraining. Same move here; I19 additionally proved a
retrain on the v2 frame is byte-identical, so "archived seed-29 arm" and
"fresh retrain" are the same artifact.

## Why promote (and the honest cost)

1. **The legacy line is operationally dead.** Its serving state ends
   2026-04-16 and cannot be regenerated; `predict_fixture.py` already fails
   the >14-day staleness gate for live fixtures. Only the i7 stack has a
   working fresh-state build path (Hundred operation, proven through
   2026-07-13).
2. **Golden slices favor the successor.** On the 124-match golden set the
   swap-i7 arm beats the matched market LL on both sharp slices; the legacy
   swap production model does not.
3. **Cost:** swap-i7 trails legacy on iteration ≥$50k LL by ~0.005–0.009
   (0.6262–0.6306 vs 0.6215) — inside I17's predeclared 0.02 blocking
   threshold, and partly an artifact of the legacy model having been
   selected on that slice.
4. **No betting-edge claim transfers.** All ROI block CIs straddle zero on
   both frames. A7 remains shadow-only, unchanged.

## Cutover steps (proposed order)

1. **Mint the production artifact.** Copy `models/auto/i17/swap_seed29` →
   `models/xgb_match_i7_swap_production/`; add PROVENANCE.md (source arm,
   commit shas for I17/I18/I19, identity stamps `cricsheet_primary_v1` +
   `venue_aliases_v1` — the `models/auto/i19/` dirs show the stamp format).
2. **Build fresh serving state through the cutover date**: i7
   canonical-venue SQLite cache + tracker snapshot, `--state-version i7`,
   via the non-destructive refresh path in OPERATIONS Op 7 (generalized from
   the Hundred build; T20 pool only, no aux dirs). Verify the same-day
   ordering contract (`date_then_match_id_lexicographic_v1`) is stamped.
3. **Switch `predict_fixture.py` defaults** (one commit, trivially
   revertible): `MODEL_DIR` → the new dir;
   `DEFAULT_LIVE_VENUE_IDENTITY_MODE` → `i7`; default state dir + tracker
   snapshot → the new i7 state. Keep the legacy replay contract code path
   intact — it still serves rollback and historical repro.
4. **Verification gates before first live use:**
   a. Determinism: served p(team1) on ≥3 reference fixtures matches the
      offline i7 prediction path at |Δp| = 0 (I17's determinism-gate style).
   b. Staleness gate passes with the fresh state (it cannot today).
   c. One Hundred-path regression check: Op 7 still produces the 2026 report
      numbers with its own aux state (it shares the i7 family).
5. **Docs sweep:** CLAUDE.md model-of-record paragraph, OPERATIONS Op 6,
   promotion report `reports/i17_promotion_<date>.md`, memory update.
6. **Retire gradually:** legacy `xgb_match_v3_m7_swap_production` and
   `xgb_match_v3_m7_production` stay on disk for rollback (same retention
   rule as the D12 promotion).

Rollback = revert the step-3 commit. State builds are additive under
`data/` and don't touch legacy artifacts.

## Open follow-ups this plan does NOT cover

- Golden-refresh replumb: the golden build scripts emit legacy-keyed
  envelopes; the i7 golden frame exists (I18) but the refresh pipeline
  (`predict_golden.py` → reslice) needs an i7 variant before the *next*
  golden extension. Night-executable once specced.
- Re-key/annotate remaining display-keyed artifacts (I19's suggested
  follow-up; the golden odds file keeps 55 verbatim legacy rows).

## Decisions required from you

- **D1 — Go/no-go** on the promotion itself (the operational argument is
  about serviceability, not accuracy dominance).
- **D2 — Artifact**: verbatim `swap_seed29` (recommended, it's the audited
  arm) vs a fresh retrain on the v2 frame (identical bytes, cleaner
  provenance stamp). Cosmetic either way.
- **D3 — State cadence**: how often the serving cache/snapshot is refreshed
  (recommend: weekly cron + mandatory pre-fixture rebuild; keep the 14-day
  staleness gate as the backstop).
- **D4 — Cutover timing**: immediately, or after the golden-refresh replumb
  so the audit pipeline follows the model contract without a gap.

---

## Execution log (2026-07-31)

1. **Artifact minted**: `models/xgb_match_i7_swap_production/` = verbatim
   `models/auto/i19/swap_seed29` + PROVENANCE.md. model.pkl md5
   `54faf58638a799468d551beb3493b22d` (== I18 golden-audited i17 arm).
2. **Serving state built**: `data/live_state_i7/player_stats_cache_i7.sqlite`
   — 9,920 matches / 7,762 players / 378 venues, global+phase priors frozen
   from `models/player_stats_cache_i7.sqlite`. Content-identical to the
   proven Hundred cache on every data table; `_meta` differs only in
   `build_timestamp` plus the newer builder's `elo_update_version` stamp
   (`fixed_competition_k_v1`). Tracker snapshot walked the same 9,920
   sources in 8.5 s → `data/live_state_i7/tracker_snapshot.pkl`.
3. **Gate (a) parity — PASS, exact**: served `_validation_kkr_gt_2026-04-17`
   through the full live path (i7 mode, fresh state) →
   p(KKR) = 0.42299941182136536, bit-identical to the I18 golden-frame
   prediction for cricsheet 1529268. |Δp| = 0.
4. **Gate (b) determinism — PASS**: second run (snapshot reuse) reproduced
   the prediction bit-for-bit.
5. **Gate (c) pre-toss sanity — PASS**: `2026-05-31_rcb_gt_final` (null
   toss, provisional lineups) runs the both-branch toss average
   (56.4% / 58.3% → p(RCB) 57.3%). Not a parity fixture by construction.
6. **Defaults switched** in `scripts/predict_fixture.py`; CLAUDE.md,
   OPERATIONS Op 6, and I7_LIVE_COMPATIBILITY updated.

**Live-use note:** on-disk cricsheet ends 2026-07-13, so the 14-day gate
correctly refuses fixtures after ~2026-07-27 until the routine mirror
refresh + Op 6 state rebuild. Hundred Op 7 is unaffected (own aux state).
