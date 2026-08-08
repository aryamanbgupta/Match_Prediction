# T20 Blast 2026 — betting eval restated on v2 odds

**Date:** 2026-08-07
**Status:** rebuilt, restated, **numbers unchanged to every decimal place**.
**Audit this closes out:** `reports/market_benchmark_toss_defect_20260805.md`
(TODO.md § "Market-benchmark correction follow-ups", P1 "Blast track is EXPOSED
and its ROI needs a rebuild").
**Artifacts:** `data/golden_blast/betting_odds_blast_v2.json`,
`data/golden_blast/polymarket_test_v2/`,
`data/golden_blast/build_unmatched_v2.json`,
`reports/blast_2026_dashboard.html` (regenerated on v2).

---

## Why this was opened

`scripts/build_blast_odds.py` is a thin wrapper around
`scripts/build_polymarket_odds.py`, which until 2026-08-05 selected between a
fixture's sibling Polymarket markets with a tuple whose first element was
`max(price) <= 0.92 = "plausible"`. That flag rejects exactly the informative
lopsided winner markets and keeps the ~0.50 "Who wins the toss?" coin flip, so
23 of 261 iteration fixtures were scored against the toss market. The Blast
odds file was built by the same code on 2026-06-04, so the eval downstream of
it — flat ROI −12.1%, market LL 0.6894, underdog on 29/34 picks — was flagged
unverified pending a rebuild.

## The rebuild

The fixed builder re-attaches each capture record's Gamma market identity and
keeps only structurally head-to-head markets, outcome-blind. On the Blast
capture:

| check | result |
|---|---|
| capture records | 42 (one per Gamma event; 42 distinct `event_id`s) |
| resolved market identity | 34/34 matched fixtures, all via `market_volume_exact` |
| `sportsMarketType == "moneyline"` **and** `question == event_title` | **34/34, both signals agree, 0 conflicts** |
| non-H2H (toss/side) records selected | **0** |
| fixtures dropped for lack of an H2H market | **0** |
| H2H siblings deduped | **0** |
| rows priced at/after scheduled start (`--timestamp-guard report`) | **0** |

**Why the Blast was immune.** The defect lived in the *dedup* between two
capture records that map to the same fixture (`if mid in best_by_match: ...
score(m) <= score(prev_m)`). The Blast extractor emits **one record per event**
and every event is a distinct fixture, so that branch never executed — there
was never a sibling to rank, and the `plausible` flag was never consulted. The
prices are also nowhere near the cap that caused the harm: the most lopsided
Blast market in the file tops out at **0.665**, and 0 of 34 exceed 0.92. County
T20 is an evenly-matched competition, which is precisely the regime the
defective rule handled correctly.

## Old vs v2 diff

| | pre-fix `betting_odds_blast.json` | v2 `betting_odds_blast_v2.json` |
|---|---|---|
| fixtures | 34 | **34** |
| fixtures dropped | — | **0** |
| fixtures added | — | **0** |
| **rows with a changed price** | — | **0 of 34** |
| changed `actual_winner` / volume / price timestamp | — | **0** |
| unmatched Polymarket markets | 2 (Hampshire v Sussex 06-02, Surrey v Middlesex 06-03 — absent from the local Cricsheet pool) | same 2 |
| `match_id` re-keyed | — | 34/34 (Cricsheet-primary `cricsheet_primary_v1`, the I15 identity migration — unrelated to the toss fix) |
| provenance | none | `selection_rule.version = h2h_identity_outcome_blind_v2`, `winner_used_for_market_selection: false`, `price_magnitude_filter: null` |

The only substantive differences are the identity re-key and the new
provenance block. **No price moved.**

## Restated headline

Recipe identical to the 2026-06-04 eval: `scripts/build_blast_dashboard.py`'s
join and bet rule (1 unit flat on the largest positive edge, model − market
de-vigged 2-way, no bet at edge ≤ 0), 34-match pool
`data/golden_blast/t20s_json/`, predictions
`models/xgb_match_v3_m7_production/blast_golden_predictions.json`.

| metric | original (pre-fix odds) | **restated (v2 odds)** | Δ |
|---|---|---|---|
| matches joined | 34 | **34** | 0 |
| Model LL | 0.7138 | **0.7138** | 0.0000 |
| **Market LL (de-vigged)** | 0.6894 | **0.6894** | 0.0000 |
| Coinflip LL | 0.6931 | 0.6931 | — |
| Bets placed | 34 | **34** | 0 |
| Wins / win rate | 12 / 35.3% | **12 / 35.3%** | 0 |
| Total PnL | −4.13 units | **−4.13 units** | 0 |
| **Flat ROI** | −12.14% | **−12.14%** | 0.00pp |
| ROI CI, match-level i.i.d. (as originally quoted) | [−51.0%, +28.8%] | **[−51.0%, +28.8%]** | 0 |
| ROI CI, `tournament_time_block_v1` | not computed | **degenerate — 1 block** | — |
| Bets on the market underdog | 29 / 34 | **29 / 34** | 0 |

Both columns were computed by the same script over the two odds files; the
pre-v2 file is read here **only** to produce this diff, and is never used to
score anything going forward.

### Two notes on the intervals

- The originally-quoted CI [−50%, +29%] is a **match-level i.i.d.** bootstrap,
  which invariant 7 superseded: match-winner ROI uncertainty must use
  `tournament_time_block_v1`. Reproduced above at 10,000 seed-42 resamples as
  [−51.0%, +28.8%] so the two columns are like-for-like.
- Under the I3 contract the whole Blast pool is **one competition block**
  (a single event, nine match-days), so the block bootstrap is degenerate and
  reports no interval at all. `<10 blocks = descriptive` — the Blast ROI was
  never entitled to an interval, and that is a stronger statement of "no edge
  established" than the wide i.i.d. one it replaces.

## Honest reading

The rebuild changes nothing, and the reason is structural rather than lucky:
the Blast capture carried one market per fixture, so the defective tiebreak
never ran, and no Blast market was priced anywhere near the 0.92 cap that made
the tiebreak harmful. **Both original conclusions survive intact.** The
no-edge conclusion survives — flat ROI is still −12.14% over 34 bets, still
negative, and its uncertainty is if anything worse-characterised than before
(one I3 block, so descriptive only). The LL ordering survives exactly as
published: **market 0.6894 < coinflip 0.6931 < model 0.7138**, i.e. the model
is beaten by the market *and* by a coin flip on this pool, while the model
backs the market underdog on 29 of 34 fixtures — the low-resolution
favourite-fading failure mode the original eval described. Unlike the
iteration set, where the correction moved the market line by 0.037 LL and cut
ROI by 17pp, the Blast track's published numbers were never contaminated. They
were unverified, and they are now verified.

## Commands run

```bash
# 1. Confirm the fixed selection path and inspect identity resolution
uv run python scripts/build_blast_odds.py --dry-run

# 2. Rebuild to the new _v2 defaults (the pre-fix file is never touched)
uv run python scripts/build_blast_odds.py

# 3. Restate the eval on both odds files (same recipe as build_blast_dashboard)
#    — scratch script; the dashboard below reproduces the point estimates.
uv run python scripts/build_blast_dashboard.py
```

## Files changed

- `scripts/build_blast_odds.py` — output defaults moved to
  `betting_odds_blast_v2.json` / `polymarket_test_v2/` / `build_unmatched_v2.json`
  so a plain run cannot clobber the shipped evidence, mirroring
  `build_polymarket_odds{,_golden}.py`; `--out-odds` / `--out-test-dir` /
  `--out-unmatched` / `--timestamp-guard` / `--restrict-to-manifest` wired
  through from the shared builder.
- `scripts/build_blast_dashboard.py` — `ODDS` repointed to the v2 file; the
  odds lookup now indexes `match_id`, `cricsheet_id` and `display_match_id`
  (the v2 rebuild emits Cricsheet-primary ids, which the old single-key lookup
  would not have joined).
- `scripts/build_ipl_dashboard.py` — `ODDS_MAIN` → `betting_odds_polymarket_v2.json`,
  `ODDS_GOLDEN` → `data/golden/betting_odds_golden_v2.json`.
- `reports/blast_2026_dashboard.html` — regenerated on v2 odds (identical
  numbers; only the generated-at stamp moves).

`reports/blast_golden_2026_eval.md` keeps its 2026-08-05 warning banner; this
report is the restatement that banner asked for.
