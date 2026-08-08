# Market benchmark defect: the toss market was scored as the winner market

**Date:** 2026-08-05
**Status:** defect confirmed, fixed, both benchmarks rebuilt to new files, both
production artifacts re-scored.
**Artifacts:** `betting_odds_polymarket_v2.json`,
`data/golden/betting_odds_golden_v2.json`,
`eval_out/toss_defect_20260805/`.

---

## Scope: read this first

This defect corrupts **model-versus-market** comparisons on the iteration set
and, marginally, on the golden set. It does **not** corrupt
**model-versus-model** comparisons, and no model-selection decision needs to be
re-run.

The reason is structural. A model's log loss is computed against the Cricsheet
outcome, not against a market price — the market never enters it. The market
only enters (a) the market-LL reference column, (b) flat ROI, which prices the
bet off `market_odds`, and (c) slice membership via `polymarket_volume_usd`,
which on the iteration set is **event**-level volume and is therefore identical
for a fixture's head-to-head and toss rows. Every arm of every A/B was scored
against one and the same benchmark, so a paired difference between two arms
cancels the benchmark almost exactly.

Measured, not asserted — the D12 and I17 promotion evidence, seed 29, iteration
≥$50k slice, re-scored under both benchmarks:

| pair | benchmark | base LL | swap LL | **ΔLL** | base ROI | swap ROI | **ΔROI** | n |
|---|---|---|---|---|---|---|---|---|
| D12 | shipped   | 0.6299 | 0.6215 | **−0.0084** | +21.90% | +24.53% | **+2.63pp** | 170 |
| D12 | corrected | 0.6283 | 0.6196 | **−0.0086** | +6.20%  | +7.40%  | **+1.21pp** | 168 |
| I17 | shipped   | 0.6421 | 0.6262 | **−0.0159** | +17.49% | +20.54% | **+3.05pp** | 170 |
| I17 | corrected | 0.6399 | 0.6249 | **−0.0150** | +1.76%  | +3.38%  | **+1.62pp** | 168 |

The ΔLL that gated both promotions moves by 0.0002 (D12) and 0.0009 (I17) —
both still clear the 0.007 floor with the same sign. ΔROI keeps its sign and
shrinks, as it must, since ROI is priced off the market. **The D12 and I17
promotions stand.** So does everything else decided on paired model-vs-model
deltas: the M-phase ablations, the seed sweeps, the ball-level D16/D17/D18
work (which never touches this file at all).

What **is** invalidated: every published claim of the form "the model beats the
Polymarket line on log loss" on the iteration set, and every iteration ROI
point estimate. Those are restated below.

**The sealed forward holdout is unaffected and remains valid** — verified
independently in § "Forward holdout" below.

---

## The defect

`scripts/build_polymarket_odds.py` selected the wrong Polymarket market for
23 of 261 iteration fixtures (8.8%).

Each Gamma cricket event carries several binary markets: the head-to-head
winner market (`sportsMarketType == "moneyline"`), `"… - Who wins the toss?"`
(`cricket_toss_winner`), and `"… - Completed match?"`
(`cricket_completed_match`). The upstream prematch capture
(`polymarket_prematch_odds.json`) emits **one bare record per market** and
persists neither the market id nor the question, so two records for the same
fixture arrive at the builder looking nearly identical.

The pre-fix dedup ranked those siblings with this tuple (line numbers are from
the pre-fix file, `git show HEAD:scripts/build_polymarket_odds.py`):

```
383:    PLAUSIBLE_TOP_P = 0.92
392:        matches_cric = 1 if (mapped and cric and mapped == cric) else 0
396:            plausible = 1 if max(p1, p2) <= PLAUSIBLE_TOP_P else 0
400:        return (plausible, matches_cric, vol)
420:            if score(m) <= score(prev_m):
```

A toss is a coin flip, so it prices near 0.50 and is always "plausible". The
real winner market on a lopsided fixture prices above 0.92 and is not. Because
`plausible` is the **first** element of the tuple, it outranks everything —
so on exactly the fixtures where the market is most informative, the informative
market was rejected and the coin flip was kept.

**India vs USA, 2026-02-07, Gamma event 195081:**

| market id | question | prematch top price |
|---|---|---|
| 1311639 | `T20 World Cup: India vs USA (Game 1)` | **0.9665** — rejected |
| 1311640 | `T20 World Cup: India vs USA (Game 1) - Who wins the toss?` | **0.5100** — shipped |

Here the toss also resolved to the *other* team (USA won the toss, India won
the match), so `matches_cric` preferred the correct market — and was overruled
by `plausible`, one rank above it.

**England vs Nepal, 2026-02-08, Gamma event 196274:**

| market id | question | prematch top price |
|---|---|---|
| 1317232 | `T20 World Cup: England vs Nepal (Game 1)` | **0.9560** — rejected |
| 1317238 | `T20 World Cup: England vs Nepal (Game 1) - Who wins the toss?` | **0.5050** — shipped |

England won both the toss and the match, so `matches_cric` was 1 for both rows
and could not discriminate at all; the 0.92 cap decided alone.

### Root cause, counted

Replaying the pre-fix tuple reproduces **all 261 shipped rows with 0 price
mismatches**, so the attribution below is exact:

| | count |
|---|---|
| shipped rows that are a non-H2H (toss) market | **23** of 261 (8.8%) |
| … at ≥$50k | **19** of 170 (11.2%) |
| … at ≥$100k | **16** of 110 (14.5%) |
| correctable (a head-to-head sibling exists) | **17** |
| no head-to-head sibling → must be dropped | **6** |
| caused by the 0.92 plausibility cap | **12** |
| caused by arbitrary capture-order tie-breaking (line 420) | **5** |
| caused by `matches_cric` on its own | 0 |

On the 17 correctable fixtures **the heavy favourite won 16/17**, and the model
was scored against a benchmark that had priced them at a coin flip.

### The cap was a misdiagnosis

The 0.92 cap was introduced to suppress "in-play or post-match" snapshots. Over
the in-window filtered capture records the extremity it was built to suppress
is almost entirely a toss artifact:

| top price | head-to-head rows (n=513) | toss rows (n=296) |
|---|---|---|
| > 0.92 | 7.6% | 13.9% |
| > 0.99 | **1.4%** | **10.5%** |
| in [0.49, 0.53] | 15.0% | **70.9%** |

Once selection is done structurally, the contamination the cap was chasing is
gone: **0 of the 255 corrected rows carry a price stamped at or after the
scheduled start** (timestamp guard, reported below). A price-magnitude filter
was never the right instrument; a timestamp guard is, and it now finds nothing
to drop.

### Second, independent defect

`matches_cric` (rank 2) preferred the sibling whose Polymarket resolution
matched the Cricsheet winner. That is **outcome-dependent benchmark
construction** — the benchmark becomes a function of the result it is used to
score — and is a contract violation irrespective of the toss issue. On this
data it never flipped a row by itself (the cap decided first), but it has been
removed from selection entirely.

---

## The fix

`scripts/build_polymarket_odds.py`. Selection is now **structural and
outcome-blind**, mirroring the proven-clean
`/Users/aryamangupta/Projects/polymarket-cricket/extract_match_prematch_odds_strict.py`.

1. **Market identity is re-attached before anything is decided.** The capture
   drops the market question and id, so they are restored from a Gamma
   `event → markets` snapshot (`data/polymarket_market_catalog.json`) plus the
   ordered raw market list the capture iterated over. Two resolvers:
   - `ordered_capture_index` — the capture output is an order-preserving
     subsequence of the raw market list, so a single forward scan recovers the
     exact market. Resolves **1161/1161** iteration records.
   - `market_volume_exact` — a unique market in the event whose volume equals
     the record's. Used for the golden captures, which persist market-level
     volume. Resolves **262/263** (the one ambiguous record is below the
     $1,000 volume floor and never reaches selection).
2. **Primary rule — head-to-head identity.** `market.question == event.title`,
   with Gamma's explicit `sportsMarketType == "moneyline"` preferred when
   present. Where both signals exist they must **agree**, or the record is
   unresolved rather than guessed. **They never disagree — 0 conflicts in
   2,326 records across all three captures**: iteration 970/970 agree (191
   further records pre-date Gamma's `sportsMarketType` and carry the identity
   signal only), golden 2026-05-09 capture 262/262, golden 2026-07-23 capture
   1094/1094.
3. **Tiebreak** — highest market `volume_usd`, then lowest `market_id`.
   Capture order is never consulted.
4. **`matches_cric` removed.** The winner is now used only for the
   post-selection disagreement diagnostic. `winner_used_for_market_selection:
   false` is written into every output file.
5. **`PLAUSIBLE_TOP_P` removed as a selection criterion.** Replaced by an
   optional `--timestamp-guard {off,report,enforce}` (default `report`)
   comparing `price_timestamp` against the market's `gameStartTime`. The
   low-liquidity and $1,000 volume filters are unchanged.
6. **Fixtures with no head-to-head sibling are DROPPED**, with the rejected
   question recorded in `dropped_fixtures` in the diagnostics file. They are
   never retained at a toss price.
7. **Provenance.** Each row carries a `market_selection` block (market id,
   question, event title, `sports_market_type`, market volume, scheduled start,
   resolver); each file carries a `selection_rule` block
   (`h2h_identity_outcome_blind_v2`) and `winner_used_for_market_selection:
   false`.
8. **New flags** `--out-odds` / `--out-test-dir` / `--out-unmatched` (so the
   shipped files are never overwritten) and `--restrict-to-manifest` (rebuild a
   frozen benchmark's own fixtures under the new rule without re-deriving its
   membership).

`scripts/build_polymarket_odds_golden.py` inherits all of the above through
`base.write_outputs`. Its `_merge_staging_into_golden` also had an incidental
header bug — it appended rows without updating `total_matches`, which is why
the shipped `data/golden/betting_odds_golden.json` says `55` while holding
124 rows. **The writer is fixed**; the shipped file is left as-is on purpose,
since it is the evidence of what was shipped. *Recommendation, not a decision:*
correct the shipped header in place in a separate, clearly-labelled commit —
the field is descriptive metadata that nothing reads, and leaving a wrong 55
in a sealed artifact is a worse trap than a documented one-field fix. Defer to
the owner.

### One deliberate non-change

`polymarket_volume_usd` still carries the **capture-level** volume, because
downstream `--min-volume` slices are defined against it and changing it would
confound the shipped-vs-corrected comparison. Market-level volume is reported
separately under `market_selection.market_volume_usd`. On the iteration set
this is moot — both siblings share the event volume, so **no fixture changes
slice membership**. On golden exactly one does; see below.

---

## Independent confirmation: two blind rules agree

The head-to-head identity rule and a pure **max-market-volume** rule are
structurally independent and both outcome-blind. Applied to the same 398
matched iteration records:

- they select the **same market on 253 of the 255** fixtures both retain;
- market LL agrees to **0.0000** at ≥$100k (0.5377 vs 0.5377) and **0.0012** at
  ≥$50k (0.5940 vs 0.5928).

The two fixtures where they disagree are cases where a *side* market out-traded
the head-to-head market, which is precisely why identity, not volume, is the
primary rule. (Verification's report of agreement to 0.0007 is in the same
place; my numbers are 0.0012 / 0.0000 and supersede it.)

---

## Per-fixture diff — iteration set (audit trail)

`betting_odds_polymarket.json` (261 rows, shipped) →
`betting_odds_polymarket_v2.json` (255 rows, corrected).

Volume is unchanged on every shared fixture, so slice counts move only through
the 6 drops: **n 261→255 (all), 170→168 (≥$50k), 110→110 (≥$100k)**.

### Price corrected (17 fixtures — all 17 are ≥$50k, 16 are ≥$100k)

| date | fixture | shipped p(team1) | corrected p(team1) | shipped top | corrected top | winner | volume | ≥50k | ≥100k | corrected market question |
|---|---|---|---|---|---|---|---|---|---|---|
| 2025-12-19 | Perth Scorchers vs Brisbane Heat | 0.4550 | 0.5850 | 0.5450 | 0.5850 | Brisbane Heat | $86,262 | Y | · | `Big Bash League: Brisbane Heat vs Perth Scorchers (Game 1)` |
| 2026-01-23 | New Zealand vs India | 0.4750 | 0.2500 | 0.5250 | 0.7500 | India | $481,918 | Y | Y | `International T20 Series: India vs New Zealand (Game 2)` |
| 2026-02-07 | India vs United States of America | 0.5100 | 0.9665 | 0.5100 | 0.9665 | India | $917,327 | Y | Y | `T20 World Cup: India vs USA (Game 1)` |
| 2026-02-08 | England vs Nepal | 0.4950 | 0.9560 | 0.5050 | 0.9560 | England | $1,075,694 | Y | Y | `T20 World Cup: England vs Nepal (Game 1)` |
| 2026-02-10 | Namibia vs Netherlands | 0.4850 | 0.2900 | 0.5150 | 0.7100 | Netherlands | $903,065 | Y | Y | `T20 World Cup: Namibia vs Netherlands (Game 1)` |
| 2026-02-10 | United Arab Emirates vs New Zealand | 0.4950 | 0.0360 | 0.5050 | 0.9640 | New Zealand | $543,459 | Y | Y | `T20 World Cup: New Zealand vs UAE (Game 1)` |
| 2026-02-11 | Australia vs Ireland | 0.5050 | 0.9350 | 0.5050 | 0.9350 | Australia | $742,888 | Y | Y | `T20 World Cup: Australia vs Ireland (Game 1)` |
| 2026-02-12 | Sri Lanka vs Oman | 0.5050 | 0.9350 | 0.5050 | 0.9350 | Sri Lanka | $607,695 | Y | Y | `T20 World Cup: Sri Lanka vs Oman (Game 1)` |
| 2026-02-12 | India vs Namibia | 0.4750 | 0.9855 | 0.5250 | 0.9855 | India | $1,383,960 | Y | Y | `T20 World Cup: India vs Namibia (Game 1)` |
| 2026-02-15 | United States of America vs Namibia | 0.4950 | 0.7150 | 0.5050 | 0.7150 | United States of America | $848,124 | Y | Y | `T20 World Cup: Namibia vs USA (Game 1)` |
| 2026-02-16 | England vs Italy | 0.5050 | 0.9450 | 0.5050 | 0.9450 | England | $1,157,022 | Y | Y | `T20 World Cup: England vs Italy (Game 1)` |
| 2026-02-17 | Canada vs New Zealand | 0.5100 | 0.0255 | 0.5100 | 0.9745 | New Zealand | $525,831 | Y | Y | `T20 World Cup: Canada vs New Zealand (Game 1)` |
| 2026-02-18 | Pakistan vs Namibia | 0.5050 | 0.8750 | 0.5050 | 0.8750 | Pakistan | $884,771 | Y | Y | `T20 World Cup: Namibia vs Pakistan (Game 1)` |
| 2026-02-18 | India vs Netherlands | 0.5050 | 0.9745 | 0.5050 | 0.9745 | India | $1,302,155 | Y | Y | `T20 World Cup: India vs Netherlands (Game 1)` |
| 2026-02-18 | United Arab Emirates vs South Africa | 0.4900 | 0.0650 | 0.5100 | 0.9350 | South Africa | $670,971 | Y | Y | `T20 World Cup: South Africa vs UAE (Game 1)` |
| 2026-02-20 | Oman vs Australia | 0.4950 | 0.0450 | 0.5050 | 0.9550 | Australia | $262,760 | Y | Y | `T20 World Cup: Australia vs Oman (Game 1)` |
| 2026-02-26 | India vs Zimbabwe | 0.5050 | 0.9250 | 0.5050 | 0.9250 | India | $858,762 | Y | Y | `T20 World Cup: India vs Zimbabwe` |

Sixteen of the seventeen are 2026 T20 World Cup group-stage mismatches; the
shipped benchmark priced every one of them within 0.055 of a coin flip.

### Dropped — no head-to-head sibling survives the filters (6 fixtures)

| date | fixture | shipped top price | shipped market (rejected) | volume | ≥50k | ≥100k |
|---|---|---|---|---|---|---|
| 2025-12-28 | Durban's Super Giants vs MI Cape Town | 0.9870 | `SA20: … (Game 1) - Who wins the toss?` | $24,873 | · | · |
| 2026-01-08 | Joburg Super Kings vs Paarl Royals | 0.5050 | `SA20: … (Game 1) - Who wins the toss?` | $52,375 | Y | · |
| 2026-01-11 | Sydney Sixers vs Hobart Hurricanes | 0.9975 | `Big Bash League: … (Game 1) - Who wins the toss?` | $13,569 | · | · |
| 2026-04-06 | Portugal vs Norway | 0.5150 | `T20 Portugal Tri-Series: … - Who wins the toss?` | $22,195 | · | · |
| 2026-04-07 | France vs Norway | 0.5200 | `T20 Portugal Tri-Series: … - Who wins the toss?` | $50,603 | Y | · |
| 2026-04-08 | Portugal vs Norway | 0.5050 | `T20 Portugal Tri-Series: … - Who wins the toss?` | $9,164 | · | · |

Note rows 1 and 3: shipped top prices of **0.9870 and 0.9975** on a *toss*
market. Those are exactly the "implausible" prices the 0.92 cap existed to
reject, and it kept them, because in both cases the head-to-head sibling had
already been filtered out and the cap only ever chose *between* survivors.

---

## Per-fixture diff — golden set

`data/golden/betting_odds_golden.json` (124 rows) →
`data/golden/betting_odds_golden_v2.json` (124 rows). Rebuilt through the same
two-stage path (2026-05-09 capture, then the 2026-07-23 capture merged), with
`--restrict-to-manifest` so the sealed membership is reproduced rather than
re-derived. **No fixture dropped.**

| date | fixture | shipped p(team1) | corrected p(team1) | winner | shipped vol | corrected vol | corrected market |
|---|---|---|---|---|---|---|---|
| 2026-05-28 | Botswana vs Ivory Coast | 0.4900 | **0.9840** | Botswana | $2,853 | **$145,464** | `T20 World Cup, Sub Regional Africa, Qualifier A: Ivory Coast vs Botswana` |

Golden is barely touched, for a mechanical reason: the golden captures persist
**market**-level volume, so a fixture's toss row usually carries a tiny volume
and is killed by the $1,000 floor or the low-liquidity flag before selection
ever runs. This single survivor also changes slice membership (its true
head-to-head volume is $145k, not $2.8k), so golden slice counts move
**75→76 (≥$50k)** and **66→67 (≥$100k)**.

Golden market log loss, shipped → corrected:

| slice | shipped | corrected | Δ |
|---|---|---|---|
| all (n=123 priced) | 0.5513 | 0.5457 | −0.0057 |
| ≥$50k (74→75) | 0.6573 | 0.6488 | −0.0085 |
| ≥$100k (65→66) | 0.6843 | 0.6742 | −0.0101 |

**Golden verdict: the golden conclusions are materially unchanged.** Both
production arms already trailed the market on golden LL; the market gets
0.006–0.010 *better*, so they trail by slightly more. Nothing flips.

---

## Restated headline — iteration set

Method: envelope → `blend_eval_json --w 0.0` → `reslice_eval_json`, with each
model's own **shipped** `test_predictions.json` (no re-prediction). The
envelope uses the margin-normalized implied probabilities that
`BettingOddsLoader` produces; this reproduces every documented shipped number
exactly (LL 0.6215 / market 0.6482 / ROI +24.53% / CI [−1.98, +46.37] / 19
blocks for the M7 arm; LL 0.6262 / ROI +20.54% / 19 blocks for the i7 arm), so
shipped-vs-corrected is a like-for-like delta. ROI intervals are
`tournament_time_block_v1`: 10,000 seed-42 whole-event resamples, flat 1 unit
at edge threshold 0.

### `models/xgb_match_i7_swap_production` (production of record)

| slice | n | model LL | **shipped** market LL | **corrected** market LL | shipped ROI (CI, blocks) | **corrected ROI (CI, blocks)** |
|---|---|---|---|---|---|---|
| all | 261→255 | 0.6187→0.6180 | 0.6267 | **0.5901** | +17.60% [−7.59, +46.34] (25) | **+5.55% [−15.39, +37.55] (25)** |
| ≥$50k | 170→168 | 0.6262→0.6249 | 0.6482 | **0.5940** | +20.54% [−5.25, +43.04] (19) | **+3.38% [−14.63, +37.06] (18)** |
| ≥$100k | 110→110 | 0.5886→0.5886 | 0.6224 | **0.5377** | +21.95% [−19.63, +40.00] (11) | **−5.19% [−28.73, +27.50] (11)** |

### `models/xgb_match_v3_m7_swap_production` (legacy line)

| slice | n | model LL | **shipped** market LL | **corrected** market LL | shipped ROI (CI, blocks) | **corrected ROI (CI, blocks)** |
|---|---|---|---|---|---|---|
| all | 261→255 | 0.6178→0.6169 | 0.6267 | **0.5901** | +19.67% [−4.44, +45.80] (25) | **+7.65% [−10.27, +35.08] (25)** |
| ≥$50k | 170→168 | 0.6215→0.6196 | 0.6482 | **0.5940** | +24.53% [−1.98, +46.37] (19) | **+7.40% [−7.65, +34.82] (18)** |
| ≥$100k | 110→110 | 0.5796→0.5796 | 0.6224 | **0.5377** | +26.60% [−17.21, +45.42] (11) | **−0.54% [−23.24, +24.03] (11)** |

Block counts: 25 / 18 / 11 on the corrected set (the ≥$50k slice loses one
block because both of its dropped fixtures — the SA20 and Portugal Tri-Series
toss-only rows — were the last members of theirs). All slices remain at or
above the 10-block reliability floor.

### What this means

1. **The model no longer beats the market on log loss anywhere.** Shipped, the
   i7 arm read as 0.6262 vs 0.6482 at ≥$50k — a 0.022 win. Corrected, it is
   0.6249 vs **0.5940** — a **0.031 loss**. At ≥$100k the gap is 0.051 against
   the model. The legacy M7 arm goes the same way (0.6196 vs 0.5940). The
   apparent probability edge was the coin flip, not the model.
2. **The ROI headline collapses.** ≥$50k: +24.53% → **+7.40%** (M7),
   +20.54% → **+3.38%** (i7). At ≥$100k both go negative
   (−0.54% and −5.19%). Model LL barely moves — 0.5886 is identical at ≥$100k
   before and after — so the entire move is the benchmark repricing 16 heavy
   favourites out of coin-flip territory, which removes the bets that were
   generating the return.
3. **The uncertainty conclusion is unchanged, and was always the right one.**
   Every block CI straddled zero before and every block CI straddles zero now.
   The repo's standing position — "no production betting edge is established" —
   was correct, and is now correct for a second, larger reason.
4. **Deltas vs the verification's point estimates.** Verification expected i7
   ≥$50k corrected market LL 0.5944 and ROI +2.77%, and ≥$100k ROI −5.19%. I
   compute **0.5940**, **+3.38%**, and **−5.19%**. The ≥$100k figure matches
   exactly; the ≥$50k pair differs by 0.0004 LL and 0.61pp, attributable to the
   implied-probability convention (margin-normalized here, matching
   `BettingOddsLoader` and the frozen I17 artifact, which the raw-`1/odds`
   convention does not reproduce — it yields 167 bets and +19.85% for the
   shipped i7 arm instead of the documented 168 and +20.54%). **My numbers
   stand.**

---

## Forward holdout: verified untouched

Re-verified from scratch, three independent ways.

1. **Code.** `scripts/build_forward_holdout.py` enforces
   `market_question == event_title` itself at line 191 (`not_exact_h2h`),
   requires `price_timestamp < scheduled_start_timestamp` at line 211
   (`not_strictly_prematch`), and its `market_selection_key` (line 250) is
   `(volume_usd, price_timestamp, market_id)` under the docstring
   "Outcome-blind duplicate selection key; do not add winner/result." There is
   no price-magnitude cap and no `matches_cric` anywhere in the file. The
   contract asserts `winner_used_for_market_selection: False` at line 62 and is
   re-checked against the source capture at line 462.
2. **Data.** All 137 manifest rows join to the strict capture by `market_id`;
   **0** have `market_question != event_title`, **0** contain "toss", **0**
   violate the timestamp invariant. `duplicate_fixture_groups: 0`,
   `winner_disagreements_after_outcome_blind_selection: 0`,
   `overlap_with_existing_evaluated_pools: 0`, `status: PASS`.
3. **Third-party.** Re-fetching all 137 events live from Gamma and looking up
   each selected `market_id`: **`{'moneyline': 137}`**, 0 violations. No toss
   market, no side market, anywhere in the forward set.

The forward set also demonstrates the cap's harm from the other side: it
contains **7 rows with a top price above 0.92** (one above 0.98), all of them
genuine head-to-head markets that the iteration builder's cap would have
thrown away. Only 29 of 137 rows sit in [0.49, 0.53], against 70.9% of the
iteration capture's toss rows.

**Forward ≥$50k market log loss recomputed from the sealed odds file: 0.7445**
— identical to the documented figure. The documented comparison **M7 0.6823 vs
market 0.7445 vs ball-v7 0.7015 stands unchanged**, as does everything in
`reports/forward_evaluation_2026-06-01_2026-07-13.md`. The forward holdout was
built by a different script from a different, strict capture, and this defect
never reached it.

---

## Explicit scope

**Invalidated (iteration set):**
- every "model beats the Polymarket line on LL" claim on the iteration set;
- every iteration flat-ROI point estimate and its block CI;
- the constants 0.6267 (all) and 0.6482 (≥$50k) as the market LL — they are
  0.5901 and 0.5940;
- `reports/blend_report.py`'s hardcoded `MARKET_LL_ALL = 0.6267`;
- the market columns of the D12 and I17 evaluation tables (their **paired**
  columns are fine).

**Marginally affected (golden set):** market LL improves 0.006–0.010; both arms
already trailed and still do. One fixture changes slice membership.

**Not affected:**
- the **sealed forward holdout**, in full;
- every **model-vs-model** comparison — paired ΔLL moves by ≤0.0009 and keeps
  its sign, so the **D12 and I17 promotions stand** and no model selection
  needs re-running;
- all **model log losses**, which are computed against Cricsheet outcomes and
  never reference a market price;
- **iteration slice membership**, since `polymarket_volume_usd` is event-level
  and identical across a fixture's siblings;
- the entire **ball-level** line (D16/D17/D18, props, in-play), which does not
  consume this file;
- the **women's** track, which has its own builders.

---

## Reproducing

```bash
# 1. Gamma market catalog (event -> markets; 1,643 events, ~2 min).
#    Batched GET /events?id=...; markets are closed so the snapshot is stable.
#    Regenerate only if the capture files gain new event ids.
#    -> data/polymarket_market_catalog.json

# 2. Corrected iteration benchmark (261 -> 255 rows)
uv run python scripts/build_polymarket_odds.py \
    --out-odds betting_odds_polymarket_v2.json \
    --out-test-dir data/polymarket_test_v2 \
    --out-unmatched data/polymarket_build_unmatched_v2.json

# 3. Corrected golden benchmark (124 rows, two stages)
uv run python scripts/build_polymarket_odds_golden.py \
    --restrict-to-manifest data/golden/betting_odds_golden.json \
    --out-odds data/golden/betting_odds_golden_v2.json \
    --out-test-dir data/golden/polymarket_test_v2 \
    --out-unmatched data/golden/build_unmatched_v2.json
uv run python scripts/build_polymarket_odds_golden.py \
    --polymarket-path /Users/aryamangupta/Projects/polymarket-cricket/data/polymarket_prematch_odds_2026-07-23.json \
    --merge-into-existing \
    --restrict-to-manifest data/golden/betting_odds_golden.json \
    --out-odds data/golden/betting_odds_golden_v2.json \
    --out-test-dir data/golden/polymarket_test_v2 \
    --out-unmatched data/golden/build_unmatched_v2.json

# 4. Re-score a production artifact against either benchmark.
#    (Envelope uses margin-normalized implied probs, matching BettingOddsLoader.)
uv run python scripts/sim_eval/blend_eval_json.py \
    --sim-json  eval_out/toss_defect_20260805/envN_i7_corrected.json \
    --direct-json models/xgb_match_i7_swap_production/test_predictions.json \
    --w 0.0 --out-dir eval_out/toss_defect_20260805/blendN_i7_corrected
uv run python scripts/sim_eval/reslice_eval_json.py \
    --in   eval_out/toss_defect_20260805/blendN_i7_corrected/envN_i7_corrected_w0p00.json \
    --odds betting_odds_polymarket_v2.json \
    --cluster-source-dir data/polymarket_test_v2 \
    --out-dir eval_out/toss_defect_20260805/slicedN_i7_corrected
```

The shipped `betting_odds_polymarket.json` and
`data/golden/betting_odds_golden.json` are **unmodified** (mtimes 2026-07-30)
and remain the evidence of what was shipped.

~~**Operational caveat for whoever promotes this:** the builders' *default*
output paths still point at the shipped files, so running either script
without `--out-odds` — including via the golden-refresh recipe in `CLAUDE.md`
— will overwrite the shipped evidence with a corrected build. That is left as
found rather than changed unilaterally. Whoever promotes `_v2` to the file of
record should decide between (a) archiving the shipped files under a
`_pre_h2h_fix` name and letting the defaults write corrected output, or
(b) making the default path refuse to overwrite an existing file.~~

**RESOLVED 2026-08-05, during the documentation correction that promoted `_v2`
to the benchmark of record.** Option (a), minus the rename: the builders' default
output paths now point at the `_v2` paths, so the shipped pre-fix files can no
longer be clobbered by a plain run.

- `scripts/build_polymarket_odds.py` → `betting_odds_polymarket_v2.json`,
  `data/polymarket_test_v2/`, `data/polymarket_build_unmatched_v2.json`.
- `scripts/build_polymarket_odds_golden.py` →
  `data/golden/betting_odds_golden_v2.json`, `data/golden/polymarket_test_v2/`,
  `data/golden/build_unmatched_v2.json`.

The `--out-odds` / `--out-test-dir` / `--out-unmatched` overrides are unchanged,
and `scripts/build_blast_odds.py` sets `base.OUT_*` explicitly so it is
unaffected by the new defaults (it remains exposed to the *selection* defect
until its odds file is rebuilt — tracked in TODO.md). Both shipped files still
carry their 2026-07-30 mtimes.

Verification state, for anyone re-running: the guard is non-vacuous — all 255
corrected iteration rows carry a known scheduled start, and 0 are non-pre-match.
243 of the 255 are explicitly `moneyline` in Gamma; the other 12 pre-date the
`sportsMarketType` field and were selected on the question-equals-title rule.
