# Golden eval — xgb_match_v2_frozen on truly unseen data (2026-05-09)

> ⚠️ **RETRACTED 2026-05-09 same day** — This report's headline numbers are
> inflated by an ELO feature leakage in the materializer (see
> `reports/leakage_fix_comparison.md`). The model was reading post-match
> ELO state for the match it was predicting. The honest replacement is
> `reports/leakage_fix_comparison.md` and the production model has been
> swapped to `xgb_match_v2_clean`. Keeping this file for historical record.
> Honest ≥$50k LL is 0.6747 (not 0.5004) and ROI is +32.61% (not +53.67%);
> strict go/no-go FAILS on all slices, soft gate clears only on ≥$100k.

**TL;DR (RETRACTED)**: First out-of-sample test of the production direct match-level model on
data that was never seen by training, validation, OR the iterated test set. **All
three liquidity slices clear the go/no-go gate** (model LL < market 0.6267, flat
ROI 95% CI excludes zero). Headline numbers are within bootstrap-CI overlap of
the original 261-match polymarket headline — modest LL drift up, modest ROI drift
down, **no evidence of overfitting collapse**.

## Setup

| | Original (iteration set) | New (golden) |
|---|---|---|
| Date window | 2025-07-01 → 2026-04-16 | 2026-04-17 → 2026-05-07 |
| Source polymarket file | `polymarket_prematch_odds.json` (Apr 17 capture) | `polymarket_prematch_odds_2026-05-09.json` (May 9 capture) |
| Cricsheet pool | `data/t20s_json/` (live) | `data/golden/t20s_json/` (64 new T20s extracted from stat-generator zips) |
| Polymarket markets after dedupe | 261 | 55 |
| Model | `models/xgb_match_v2_frozen/` (no retraining) | same |
| Materialized parquet | `data/xgb_match_data_v2_frozen/test.parquet` (791 rows) | `data/xgb_match_data_v2_golden/golden_test.parquet` (62 rows) |
| Frozen-mode SQLite snapshot | 2025-07-01 | 2025-07-01 (same — frozen flag applies to all post-val matches) |
| A2 trackers state | locked at 2025-06-30 | same |

The frozen snapshot is identical between runs by design — both use the
2025-07-01 SQLite snapshot and the 2025-06-30 tracker state. The model and
features are temporally apples-to-apples; only the eval target set changed.

**Sanity check**: `test.parquet` byte-content identical between v2_frozen and
v2_golden materializations (other than parquet metadata). Train/val drifted on
~1-2% of rows due to a same-day-secondary ordering nuance in the multi-source
iterator I added — irrelevant for this eval since we don't retrain, but flagged
as cleanup follow-up below.

## Headline comparison

All numbers direct alone (w=0). Bootstrap 95% CIs on 1000 resamples.

| Slice | Set | LL [95% CI] | Flat ROI [95% CI] | n / bets / win-rate |
|---|---|---|---|---|
| **all** | original 261 | 0.4944 [0.45, 0.53] | **+50.73% [+32, +74]** | 255 / — / 69.4% |
| **all** | **golden 55** | **0.5162 [0.41, 0.62]** | **+38.28% [+11, +65]** | 55 / 54 / 68.5% |
| **≥$50k** | original 168 | 0.5004 [0.45, 0.56] | **+53.67% [+36, +74]** | 168 / — / 71.4% |
| **≥$50k** | **golden 50** | **0.5366 [0.44, 0.65]** | **+47.63% [+20, +76]** | 50 / 49 / 71.4% |
| **≥$100k** | original 110 | 0.4361 [0.37, 0.50] | **+58.03% [+33, +87]** | 110 / — / 73.6% |
| **≥$100k** | **golden 45** | **0.5356 [0.44, 0.65]** | **+54.21% [+26, +80]** | 45 / 44 / 75.0% |
| reference | market | 0.6267 | — | — |
| reference | coinflip | 0.6931 | — | — |
| reference | always-favourite | — | +4.15% | — |

### Go/no-go gate (golden set)

| Slice | LL < 0.6267? | ROI CI excludes 0? | Pass? |
|---|---|---|---|
| all | ✓ (0.5162) | ✓ ([+11, +65]) | ✓ |
| ≥$50k | ✓ (0.5366) | ✓ ([+20, +76]) | ✓ |
| ≥$100k | ✓ (0.5356) | ✓ ([+26, +80]) | ✓ |

### Reading the deltas

- **LL +0.022 to +0.10 worse** vs original on each slice. CI overlap is heavy
  (e.g. ≥$50k: original [0.45, 0.56] vs new [0.44, 0.65]) — well within bootstrap
  noise at n=50, but the central tendency does drift up. Nothing alarming yet.
- **ROI -3pp to -12pp** vs original. Win rate stable at ~70%; the ROI gap is
  mostly the absence of the late-test World Cup mismatches that propped the
  original headline (no Italy / Vanuatu / France long-shot wins in this window).
- **Both metrics still well clear of every reference benchmark.** Coinflip and
  market are not even within the LL CI; flat-bet always-favourite is not within
  the ROI CI.

## Composition shift

The golden window has different tournament mix than the iteration set:

| Tournament | Original 261 share | Golden 55 share |
|---|---|---|
| International (incl. T20 WC qualifying) | majority | 53% (29 / 55) |
| IPL | minor | 47% (26 / 55) |
| (no PSL / SA20 / domestic-league representation) | mixed | none |

The composition is also significantly more *liquid*: **50/55 (91%) golden
fixtures are ≥$50k volume**, vs 168/261 (64%) on the original set. So the
"all" and "≥$50k" slices are nearly the same matches in the golden eval —
which is why their numbers are very close.

### Tournament sub-cuts

| Tournament | n | LL | Flat ROI | Bets | Win-rate |
|---|---|---|---|---|---|

(Sub-cut numbers were not generated by the standard reslicer; see "Open
follow-ups" below to compute.)

## What's reassuring

1. **No catastrophic generalization failure.** The model trained through 2024-12-31
   with frozen 2025-06-30 trackers is still strongly profitable on data 10+
   months past the freeze date.
2. **Win rate held perfectly** (71.4% on ≥$50k, identical to original).
   Indicates the bet-side decision (which team is value) is generalizing; the
   modest ROI drop is mostly average-odds-paid drift.
3. **The earlier "outlier sensitivity" caveat from `no_leakage_diagnostic.md`
   doesn't bite here.** Original headline +50% ROI dropped to ~+32% after
   stripping France @ 20.0 / Zimbabwe @ 11.76; the golden window has no
   comparable tail wins, and we still see +38 to +54% across slices. That
   suggests a meaningful chunk of edge is real, not tail-driven.

## What's concerning

1. **n=55 is small.** CIs are wide (±20pp on ROI). One bad weekend of IPL
   (~7 matches) could swing the number ±5pp.
2. **No PSL / SA20 / domestic-league coverage.** Original eval had some;
   golden has none — Polymarket coverage of those leagues didn't fall in the
   window, and our cricsheet PSL pull had matches but no Polymarket counterpart.
   Domestic leagues are the harder generalization test (tighter strength
   gaps); we still don't have a clean read on that.
3. **9 international markets unmatched** due to Cricsheet upload lag
   (Bangladesh-NZ tour, Bangladesh-SL tour, etc.). They'll resolve on next
   refresh, at which point we should re-run and add ~30-50 matches.
4. **1 super-over-tied IPL match dropped** — KKR vs LSG on 2026-04-26
   resolved by Eliminator, not regulation winner. Both `build_polymarket_odds.py`
   and `materialize_match_features.py` check only `info.outcome.winner`, not
   `eliminator`. Cleanup item below.

## Caveat that doesn't apply (note for clarity)

The **+33pp early-vs-late ROI gap in the original set** (no-leakage diagnostic)
was a *composition effect* from the late-test T20 WC qualifying matches. Those
were 47 of 131 late-test fixtures — fully present in the original headline.
The golden window contains 12 T20 WC 2026 markets per Polymarket's tagging,
but only a handful matched cricsheet (most are in the unmatched-due-to-lag
bucket). So the golden numbers under-represent the WC-mismatch tail and could
be expected to be more conservative than the original headline. Which they are.

## Open follow-ups

- **Cricsheet refresh + re-run** in 1-2 weeks: 9 unmatched international markets
  will likely have JSONs by then; eval would grow to ~64 matches.
- **Wait for IPL 2026 final + post-season re-run**: we have 26 IPL matches in
  the current golden set; final season total will be ~70. n=70 IPL-only would
  give a much tighter sub-cut.
- **Sub-cut report**: write a 5-line script to break the golden numbers down
  by tournament (IPL vs International) — the existing reslice doesn't do this.
- **Cleanup: handle super-over outcomes**. Both Polymarket-odds builder and
  match-feature materializer should fall back to `outcome.eliminator` when
  `outcome.winner` is absent and `outcome.result == 'tie'`. Would have added
  the KKR-LSG match to this eval.
- **Cleanup: stable secondary sort in iter_matches_chronological_multi**.
  My multi-source iterator sorts `(date, match_id)` while the single-source
  version relies on `Path.glob` order; on dates that exist in only one pool
  this can give a different same-day order. Caused the train/val 1-2% row
  drift on this run. Fix by porting the same secondary key into the
  single-source iterator (would change golden run determinism but not
  correctness; should be done before any retrain).
- **Forward test continues**: capture polymarket pre-match snapshots for new
  markets weekly; in another 30-60 days we'll have a non-overlapping second
  golden window for re-validation.

## Where the artifacts live

```
data/golden/
├── betting_odds_golden.json       # 55 matches, polymarket
├── polymarket_test/*.json         # 55 cricsheet match copies
├── t20s_json/*.json               # 64 cricsheet T20s ≥ 2026-04-17
├── golden_sim_envelope.json       # sim-shaped envelope for w=0 blend
├── blended/golden_sim_envelope_w0p00.json
├── sliced/
│   ├── golden_sim_envelope_w0p00_all.json
│   ├── golden_sim_envelope_w0p00_min_volume_50000.json
│   └── golden_sim_envelope_w0p00_min_volume_100000.json
└── build_unmatched.json           # 70 unmatched, dedupe diagnostics

data/xgb_match_data_v2_golden/      # full re-materialization w/ golden pool
└── {train,validation,test,golden_test}.parquet

models/xgb_match_v2_frozen/golden_predictions.json   # 62 matches scored
```

Production model artifact (`models/xgb_match_v2_frozen/model.pkl`) untouched.
Production data (`data/t20s_json/`, `data/polymarket_test/`,
`betting_odds_polymarket.json`, SQLite cache) untouched.
