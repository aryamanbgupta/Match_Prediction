# The Hundred men's 2026 — predeclared flat-stake betting backtest

**Run date:** 2026-08-03 (executor run; all runs deterministic, no fitting)
**Primary arm:** `models/xgb_match_i7`, toss-blind (`--hide-toss`)
**Secondary arm (DIAGNOSTIC ONLY):** `models/xgb_match_i7_swap_production`, same everything
**Verdict up front:** descriptive only. n = 15 settled backtest bets plus 2 settled
forward bets is one I3 tournament-time block. No confidence interval is computed,
none may be quoted, and nothing here authorises execution.

---

## 1. The predeclared rule (stated verbatim, implemented verbatim)

> For each fixture with a market, let `p_m` = model P(team1), `q` = market P(team1)
> at the **PRE-TOSS** quote (`pretoss_prob_team1`). If `p_m > q`: bet 1 unit on
> team1 at price `q` (win profit = `1/q − 1`, lose −1). If `p_m < q`: bet 1 unit on
> team2 at price `(1−q)` (win profit = `1/(1−q) − 1`, lose −1). Skip only if
> `p_m == q` exactly or a quote is missing. Flat stakes. No edge threshold.

No tuning, no post-hoc rule changes, no sizing rule, no liquidity gate. The rule
was fixed before any number below was produced, and the implementation
(`scripts/hundred_roi_eval.py::settle`) is a direct transcription of it.

**Slices**

| slice | definition | n fixtures | n with a market | n settled |
|---|---|---|---|---|
| BACKTEST | played matches 2026-07-21 → 2026-08-01 | 16 | 15 | 15 |
| FORWARD | 2026-08-02 onward, predicted with tracker/aux state **frozen at 2026-08-01** | 6 | 3 | 2 |
| combined | both | 22 | 18 | 17 |

The 2026-07-21 opener (Sunrisers Leeds v MI London) has no Polymarket event, so it
is scored for probability but carries no bet. Aug 3 has a real pre-toss quote but
was still in progress at report time; Aug 4/5 have no pre-toss quote at all and
therefore produce no bet under the rule.

**Uncertainty contract.** One tournament = one `tournament_time_block_v1` block
(invariant 7). Everything below is DESCRIPTIVE. The only inferential number shown
is an exact Poisson-binomial tail on the bet win count under the market's own
per-bet probabilities; it ignores block structure and is labelled
**i.i.d.-optimistic** everywhere it appears.

---

## 2. Data provenance

| artefact | path | note |
|---|---|---|
| Matches | `data/hundred/season_2026_men_v2/` | 18 real Cricsheet men's 2026 Hundred matches, ids 1521231–1521248, 2026-07-21 → 2026-08-02 |
| Truncated pool | `data/hundred/season_2026_men_v2_cut0801/` | the 16 matches dated ≤ 2026-08-01, built for the forward state base and the leakage check |
| Historical context | `data/hundred/context_hnd_json/` | 167 prior Hundred matches, tracker-aux only |
| Odds | `data/hundred/polymarket_odds_2026_v2.json` | fetched 2026-08-03T19:27:39Z; `pretoss_prob_team1` = last quote before T−60min, `prematch_prob_team1` = last quote strictly before scheduled start (POST-toss) |
| Team aliases | `data/hundred/team_aliases_2026.json` | 3 franchise renames folded into tracker state (copy-fold, per the 2026-07-30 fix) |
| SQLite state | `data/hundred/state/player_stats_cache_i7.sqlite` | T20-only, canonical venues, coverage through **2026-07-13**; contains zero Hundred balls |
| Tracker (full) | `data/hundred/tracker_snapshot_2026-08-02_aux_hundred_v2.pkl` | 9,920 primary + 185 aux; new file, nothing overwritten |
| Tracker (frozen) | `data/hundred/tracker_snapshot_2026-08-01_aux_hundred_cut_v2.pkl` | 9,920 primary + 183 aux (Aug 2 results removed) |
| Predictions | `eval_out/hundred_roi_2026-08-03/preds_{i7,swap}.json` | all 18 matches, both arms |
| Forward preds | `predictions/hundred/<fixture>__{i7,swap}__{cut0801,full0802}.json` | 4 fixtures × 2 arms × 2 state bases |
| Betting eval | `eval_out/hundred_roi_2026-08-03/roi_eval.json` | full ledger + every metric in this report |

Reproduce commands are in `reports/hundred_2026_adaptation.md` § Reproduce; the only
deltas here are the `season_2026_men_v2` source/aux pool and the new snapshot names.

### Leakage check (both passed)

The model can only cheat through the tracker snapshot, because the SQLite state
stops at 2026-07-13 and holds no Hundred data at all. Every tracker query filters
`record_date < as_of_date` strictly (`materialize_match_features.py`), so same-day
siblings never see each other. Two checks confirm it empirically:

1. **`--max-date 2026-08-01`** vs the full run: 16 overlapping match ids,
   **0 probability mismatches** (bit-identical `repr()`).
2. **Aux pool truncated to ≤ 2026-08-01** vs full aux pool, over **all 18**
   match ids including the two Aug-2 fixtures: **0 mismatches**. Removing the
   Aug-2 results does not move the Aug-2 predictions, which is the property the
   FORWARD slice depends on.

Forward-fixture sensitivity (Aug 3/4/5 predicted from both state bases): only
**Manchester Super Giants v Welsh Fire (Aug 5)** moves — MSG's Aug-2 defeat to
MI London drops p(MSG) from 0.5603 (frozen at Aug 1) to 0.5382 (through Aug 2),
2.2pp; swap 0.6300 → 0.6072. The other three are identical to the last bit.

### Sanity vs the 2026-07-30 historical finding

The corrected historical run compressed probabilities into ~0.37–0.62. This run:
**p ∈ [0.3729, 0.6214]** (primary), mean |p − 0.5| = 0.0548. Squarely inside the
band — the compression failure mode is fully intact on the 2026 season, and the
model sits within a mean 4.6pp of the Polymarket line. Nothing anomalous.

---

## 3. Per-match ledger — PRIMARY ARM (`xgb_match_i7`), PRE-TOSS basis

### BACKTEST (2026-07-21 → 2026-08-01)

| date | fixture (team1 v team2) | model p(t1) | market p(t1) | side backed | price | dec. odds | winner | P&L |
|---|---|---|---|---|---|---|---|---|
| 2026-07-21 | Sunrisers Leeds v MI London | 0.5383 | — | — (no market) | — | — | MI London | no bet |
| 2026-07-22 | Southern Brave v Welsh Fire | 0.6214 | 0.555 | Southern Brave | 0.555 | 1.802 | Welsh Fire | −1.000 |
| 2026-07-23 | Manchester Super Giants v London Spirit | 0.5057 | 0.485 | Manchester Super Giants | 0.485 | 2.062 | Manchester Super Giants | +1.062 |
| 2026-07-24 | Birmingham Phoenix v Trent Rockets | 0.4539 | 0.415 | Birmingham Phoenix | 0.415 | 2.410 | Birmingham Phoenix | +1.410 |
| 2026-07-25 | Sunrisers Leeds v Southern Brave | 0.5524 | 0.545 | Sunrisers Leeds | 0.545 | 1.835 | Sunrisers Leeds | +0.835 |
| 2026-07-25 | Welsh Fire v MI London | 0.4026 | 0.425 | MI London | 0.575 | 1.739 | Welsh Fire | −1.000 |
| 2026-07-26 | Manchester Super Giants v Birmingham Phoenix | 0.5296 | 0.575 | Birmingham Phoenix | 0.425 | 2.353 | Manchester Super Giants | −1.000 |
| 2026-07-26 | London Spirit v Trent Rockets | 0.4496 | 0.410 | London Spirit | 0.410 | 2.439 | Trent Rockets | −1.000 |
| 2026-07-27 | Southern Brave v MI London | 0.4688 | 0.455 | Southern Brave | 0.455 | 2.198 | MI London | −1.000 |
| 2026-07-28 | Manchester Super Giants v Sunrisers Leeds | 0.4483 | 0.495 | Sunrisers Leeds | 0.505 | 1.980 | Sunrisers Leeds | +0.980 |
| 2026-07-29 | Welsh Fire v Trent Rockets | 0.3729 | 0.455 | Trent Rockets | 0.545 | 1.835 | Trent Rockets | +0.835 |
| 2026-07-29 | MI London v London Spirit | 0.5385 | 0.575 | London Spirit | 0.425 | 2.353 | London Spirit | +1.353 |
| 2026-07-30 | Southern Brave v Birmingham Phoenix | 0.5028 | 0.565 | Birmingham Phoenix | 0.435 | 2.299 | Southern Brave | −1.000 |
| 2026-07-31 | Manchester Super Giants v Trent Rockets | 0.4510 | 0.525 | Trent Rockets | 0.475 | 2.105 | Trent Rockets | +1.105 |
| 2026-08-01 | Birmingham Phoenix v Welsh Fire | 0.5792 | 0.460 | Birmingham Phoenix | 0.460 | 2.174 | Welsh Fire | −1.000 |
| 2026-08-01 | London Spirit v Southern Brave | 0.5326 | 0.475 | London Spirit | 0.475 | 2.105 | Southern Brave | −1.000 |

**15 bets, 7 wins, −0.420 units, flat ROI −2.80%.**

### FORWARD (2026-08-02 onward; state frozen at 2026-08-01)

| date | fixture (team1 v team2) | model p(t1) | market p(t1) | side backed | price | dec. odds | winner | P&L |
|---|---|---|---|---|---|---|---|---|
| 2026-08-02 | Trent Rockets v Sunrisers Leeds | 0.5504 | 0.515 | Trent Rockets | 0.515 | 1.942 | Trent Rockets | +0.942 |
| 2026-08-02 | MI London v Manchester Super Giants | 0.5662 | 0.545 | MI London | 0.545 | 1.835 | MI London | +0.835 |
| 2026-08-03 | Welsh Fire v Southern Brave | 0.4655 | 0.515 | Southern Brave | 0.485 | 2.062 | **unresolved** | unsettled |
| 2026-08-04 | Sunrisers Leeds v London Spirit | 0.5725 | — (no pre-toss quote) | — | — | — | unresolved | no bet |
| 2026-08-05 | Manchester Super Giants v Welsh Fire | 0.5603 | — (no pre-toss quote) | — | — | — | unresolved | no bet |
| 2026-08-05 | Trent Rockets v Birmingham Phoenix | 0.5750 | — (no pre-toss quote) | — | — | — | unresolved | no bet |

**2 settled bets, 2 wins, +1.777 units, flat ROI +88.83%** — on two fixtures. That
figure is arithmetic, not evidence.

### Secondary arm (swap) — differing rows only, BACKTEST

The swap arm disagrees with the primary arm on the side backed in three fixtures:

| date | fixture | i7 side / P&L | swap p(t1) | swap side / P&L |
|---|---|---|---|---|
| 2026-07-23 | Manchester Super Giants v London Spirit | MSG / +1.062 | 0.4692 | London Spirit / −1.000 |
| 2026-07-25 | Welsh Fire v MI London | MI London / −1.000 | 0.4292 | Welsh Fire / +1.353 |
| 2026-07-26 | London Spirit v Trent Rockets | London Spirit / −1.000 | 0.3995 | Trent Rockets / +0.695 |

Net: swap +1.048 units over i7 across three coin-flips. Every other backtest bet is
the same side at the same price. The swap arm's forward ledger is identical to the
primary arm's on both settled Aug-2 fixtures. The entire arm-level ROI gap is
three fixtures.

---

## 4. Slice and combined summaries — both arms, both price bases

| arm | slice | basis | fixtures | scored | bets | wins | P&L (u) | flat ROI |
|---|---|---|---|---|---|---|---|---|
| **i7** | backtest | **pre-toss** | 16 | 15 | 15 | 7 | −0.420 | **−2.80%** |
| i7 | backtest | post-toss *(diag)* | 16 | 15 | 15 | 7 | +0.242 | +1.62% |
| **i7** | forward | **pre-toss** | 6 | 2 | 2 | 2 | +1.777 | **+88.83%** |
| i7 | forward | post-toss *(diag)* | 6 | 2 | 2 | 2 | +1.960 | +98.02% |
| **i7** | combined | **pre-toss** | 22 | 17 | 17 | 9 | +1.356 | **+7.98%** |
| i7 | combined | post-toss *(diag)* | 22 | 17 | 17 | 9 | +2.203 | +12.96% |
| **swap** | backtest | **pre-toss** | 16 | 15 | 15 | 8 | +1.566 | **+10.44%** |
| swap | backtest | post-toss *(diag)* | 16 | 15 | 15 | 8 | +1.982 | +13.21% |
| **swap** | forward | **pre-toss** | 6 | 2 | 2 | 2 | +1.777 | **+88.83%** |
| swap | forward | post-toss *(diag)* | 6 | 2 | 2 | 2 | +1.960 | +98.02% |
| **swap** | combined | **pre-toss** | 22 | 17 | 17 | 10 | +3.342 | **+19.66%** |
| swap | combined | post-toss *(diag)* | 22 | 17 | 17 | 10 | +3.942 | +23.19% |

---

## 5. Log loss — model vs market vs coinflip, on identical matched fixtures

Market LL is scored at the same price basis as the row. Coinflip LL = 0.6931.

| arm | slice | basis | n | model LL | market LL | coinflip LL | model Brier | market Brier | model acc | market acc |
|---|---|---|---|---|---|---|---|---|---|---|
| i7 | backtest | pre-toss | 15 | **0.7040** | 0.6857 | 0.6931 | 0.2552 | 0.2462 | 60.0% | 60.0% |
| i7 | backtest | post-toss | 15 | 0.7040 | 0.6878 | 0.6931 | 0.2552 | 0.2470 | 60.0% | 66.7% |
| i7 | forward | pre-toss | 2 | 0.5830 | 0.6353 | 0.6931 | 0.1952 | 0.2211 | 100% | 100% |
| i7 | forward | post-toss | 2 | 0.5830 | 0.6832 | 0.6931 | 0.1952 | 0.2450 | 100% | 100% |
| i7 | combined | pre-toss | 17 | **0.6897** | 0.6797 | 0.6931 | 0.2482 | 0.2433 | 64.7% | 64.7% |
| i7 | combined | post-toss | 17 | 0.6897 | 0.6873 | 0.6931 | 0.2482 | 0.2467 | 64.7% | 70.6% |
| swap | backtest | pre-toss | 15 | **0.6956** | 0.6857 | 0.6931 | 0.2510 | 0.2462 | 53.3% | 60.0% |
| swap | backtest | post-toss | 15 | 0.6956 | 0.6878 | 0.6931 | 0.2510 | 0.2470 | 53.3% | 66.7% |
| swap | forward | pre-toss | 2 | 0.5869 | 0.6353 | 0.6931 | 0.1971 | 0.2211 | 100% | 100% |
| swap | forward | post-toss | 2 | 0.5869 | 0.6832 | 0.6931 | 0.1971 | 0.2450 | 100% | 100% |
| swap | combined | pre-toss | 17 | **0.6828** | 0.6797 | 0.6931 | 0.2447 | 0.2433 | 58.8% | 64.7% |
| swap | combined | post-toss | 17 | 0.6828 | 0.6873 | 0.6931 | 0.2447 | 0.2467 | 58.8% | 70.6% |

**This is the load-bearing table.** On the 15-match backtest slice the primary
model's log loss (0.7040) is worse than both the market (0.6857) and a coinflip
(0.6931). It loses to the market on Brier as well. Whatever the ROI column says,
the probabilities carry no demonstrated information advantage over the Polymarket
line, and on the backtest slice they carry none over a coin.

Full-season probability scoring (all 18 matches, `backtest_hundred.py` output):
primary LL 0.6943 / Brier 0.2505 / accuracy 61.1%; swap LL 0.6873 / Brier 0.2469 /
accuracy 55.6%. The swap arm scores better probabilities and picks fewer winners
— exactly the "more confident, no more accurate" pattern seen in earlier phases.

---

## 6. Diagnostics (never the headline)

### 6.1 Favourite / underdog split of bets placed — the known failure mode

The 2026-07-30 report identified probability compression as the core defect: a
model pinned near 0.5 will systematically take the market underdog whenever the
market moves off 0.5 further than the model does.

| arm | slice | bets | underdog bets | underdog share | dog wins | dog P&L | dog ROI | fav bets | fav wins | fav P&L | fav ROI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| i7 | backtest | 15 | 10 | **66.7%** | 4 | −1.070 | −10.70% | 5 | 3 | +0.650 | +13.00% |
| i7 | forward | 2 | 0 | 0.0% | 0 | 0.000 | n/a | 2 | 2 | +1.777 | +88.83% |
| i7 | combined | 17 | 10 | **58.8%** | 4 | −1.070 | −10.70% | 7 | 5 | +2.427 | +34.66% |
| swap | backtest | 15 | 9 | 60.0% | 4 | +0.221 | +2.45% | 6 | 4 | +1.345 | +22.41% |
| swap | combined | 17 | 9 | 52.9% | 4 | +0.221 | +2.45% | 8 | 6 | +3.121 | +39.02% |

**The failure mode is present and it is where the money is lost.** Two thirds of
the primary arm's backtest bets back the market underdog, those bets return
−10.70%, and every unit of positive P&L in this report comes from the minority of
bets that agree with the market's favourite. The pattern is unchanged in the swap
arm (60% underdog bets) even though its dog ROI happens to land slightly positive
on nine bets.

### 6.2 Min-edge threshold ladder

| arm | slice | basis | 0pp (the rule) | ≥2pp | ≥5pp | ≥10pp |
|---|---|---|---|---|---|---|
| i7 | backtest | pre-toss | 15 / −2.80% | 13 / −1.96% | 6 / −34.33% | 1 / −100.00% |
| i7 | backtest | post-toss | 15 / +1.62% | 15 / +1.62% | 7 / −6.02% | 1 / −100.00% |
| i7 | combined | pre-toss | 17 / +7.98% | 15 / +10.14% | 6 / −34.33% | 1 / −100.00% |
| i7 | combined | post-toss | 17 / +12.96% | 17 / +12.96% | 8 / +6.99% | 1 / −100.00% |
| swap | backtest | pre-toss | 15 / +10.44% | 10 / +25.18% | 6 / −1.33% | 1 / −100.00% |
| swap | backtest | post-toss | 15 / +13.21% | 12 / +23.59% | 10 / +30.92% | 2 / −100.00% |
| swap | combined | pre-toss | 17 / +19.66% | 11 / +31.45% | 6 / −1.33% | 1 / −100.00% |
| swap | combined | post-toss | 17 / +23.19% | 14 / +34.22% | 11 / +37.02% | 2 / −100.00% |

ROI gets **worse** as the required edge grows, on both arms and both bases: the
fixtures where the model most disagrees with the market are the fixtures where it
is most wrong. That is the compression story again — a big model-market gap on a
compressed model means the market has moved, not that the model has found
something. Cell counts at ≥5pp and ≥10pp are 6 and 1; treat them as anecdotes.

### 6.3 Post-toss price basis

Settling the identical rule at the post-toss `prematch_prob_*` quote improves ROI
on every arm and slice (i7 backtest −2.80% → +1.62%; swap backtest +10.44% →
+13.21%). This is not a finding in the model's favour. The post-toss market is
strictly better informed than the pre-toss market (it has seen the toss, which at
The Hundred is a real chase/defend signal), so a toss-blind model priced against
it is being handed a stale line. The pre-toss basis is the honest one and is the
headline everywhere in this report.

Note the market's own accuracy rises from 60.0% pre-toss to 66.7% post-toss on the
backtest slice while the model's stays flat at 60.0% — the toss information the
market absorbs is information the toss-blind model structurally cannot have.

### 6.4 Volume note

| arm | slice | bets | n ≥ $50k | total volume | median | min | volume-weighted ROI |
|---|---|---|---|---|---|---|---|
| i7 | backtest | 15 | 15 | $3,777,260 | $197,610 | $58,557 | −6.70% |
| i7 | forward | 2 | 2 | $242,747 | $143,247 | $99,500 | +87.87% |
| i7 | combined | 17 | 17 | $4,020,007 | $195,095 | $58,557 | −0.99% |
| swap | backtest | 15 | 15 | $3,777,260 | $197,610 | $58,557 | −8.83% |
| swap | combined | 17 | 17 | $4,020,007 | $195,095 | $58,557 | −2.99% |

Every settled bet clears $50k of market volume, so no liquidity slicing is
possible or meaningful here — the whole set is the sharp slice. Weighting P&L by
market volume flips both arms' combined ROI negative (−0.99% and −2.99%), i.e. the
positive flat ROI is concentrated in the *thinner* of these thick markets. Not a
robust sign.

### 6.5 Sign test on bet wins (i.i.d.-OPTIMISTIC — orientation only)

Exact Poisson-binomial tail, P(X ≥ observed wins) under the market's own per-bet
probabilities. It assumes independent fixtures, which one tournament's fixtures
are not.

| arm | slice | bets | market-implied expected wins | actual wins | P(X ≥ wins) |
|---|---|---|---|---|---|
| i7 | backtest | 15 | 7.185 | 7 | 0.637 |
| i7 | combined | 17 | 8.245 | 9 | 0.450 |
| swap | backtest | 15 | 7.245 | 8 | 0.446 |
| swap | combined | 17 | 8.305 | 10 | 0.280 |

Nothing approaches significance even under the optimistic assumption.

---

## 7. Forward predictions (sealed)

`predictions/hundred/forward_2026-08-03_sealed.json`
**sha256 = `caecbe114b02dae0e8bc11e0dcbf91c145e446de7c3873000ed6259ac81746fd`**

Four unresolved fixtures, each with both arms under both state bases, the exact
quote the rule would settle at, and XI provenance.

| date | fixture | p(t1) i7 frozen@0801 | p(t1) i7 through-0802 | p(t1) swap frozen@0801 | pre-toss quote p(t1) | rule's bet | market volume |
|---|---|---|---|---|---|---|---|
| 2026-08-03 | Welsh Fire v Southern Brave | 0.4655 | 0.4655 | 0.4189 | 0.515 | **Southern Brave @ 0.485** | $43,137 |
| 2026-08-04 | Sunrisers Leeds v London Spirit | 0.5725 | 0.5725 | 0.5862 | — none | no bet | **$131 (seed)** |
| 2026-08-05 | Manchester Super Giants v Welsh Fire | 0.5603 | 0.5382 | 0.6300 | — none | no bet | **none reported (seed)** |
| 2026-08-05 | Trent Rockets v Birmingham Phoenix | 0.5750 | 0.5750 | 0.6392 | — none | no bet | **$8 (seed)** |

**Aug 4/5 quotes are seed liquidity and are non-actionable.** Their only quote is
a live price read at the 2026-08-03T19:27Z pull with $0–131 of traded volume;
these are placeholder prices, not markets. For orientation only, the seed prices
imply p(team1) of 0.555 / 0.525 / 0.575 — within 2–6pp of the primary arm on all
three, which is the same near-agreement the 2026-07-30 report found.

**Aug 3 status.** Welsh Fire v Southern Brave (Cricinfo id 1521249) was still in
progress at the last check (2026-08-03T19:45Z): Southern Brave 115/8 off their
100 balls, Welsh Fire 60/2 chasing 116. It is **left sealed and unsettled**; it
has a genuine $43k pre-toss quote and will settle at price 0.485 on Southern Brave
when the result lands. It is excluded from every number in sections 3–6.

**XI provenance.** Aug 3 uses the **actual** teamsheets read off the ESPNcricinfo
scorecard for 1521249 at 2026-08-03T19:33Z (Welsh Fire: Salt, Short, Root,
Ravindra, JM Cox, Kohler-Cadmore, Kellaway, Jansen, Woakes, Ferguson, Cook —
Short replaces Tribe from the Aug-1 XI; Southern Brave unchanged from Aug 1).
Aug 4 and Aug 5 use **projected** XIs — each side's most recent XI in
`season_2026_men_v2` (Sunrisers Leeds and Trent Rockets from 1521247 on Aug 2,
Manchester Super Giants from 1521248 on Aug 2, London Spirit / Welsh Fire /
Birmingham Phoenix from Aug 1). Those are not confirmed teamsheets and a late
change moves the probability. Toss is withheld on all four, so `predict_fixture`
averages both bat-first branches, matching the pre-toss information set.

---

## 8. Caveats

1. **Single tournament, descriptive only.** 15 settled backtest bets + 2 settled
   forward bets = one I3 tournament-time block. Invariant 7 forbids a match-level
   i.i.d. CI, and with one block no block CI exists. **No ROI figure here may be
   quoted as an edge.** The +88.83% forward ROI is two coin-flips that both landed;
   the +19.66% swap combined ROI is 17 bets whose sign is decided by three of them.
2. **The probabilities lose to the market and, on the backtest slice, to a coin.**
   Primary arm backtest LL 0.7040 vs market 0.6857 vs coinflip 0.6931. Any positive
   ROI coexisting with worse-than-coinflip log loss is variance, not skill.
3. **The compression failure mode is present and is doing the damage.** 66.7% of
   primary-arm backtest bets back the market underdog, returning −10.70%; ROI
   degrades monotonically as the required edge rises. This is the 2026-07-30
   finding reproduced on live 2026 fixtures, not a new result.
4. **The post-toss basis is strictly better informed** and its higher ROI is an
   artifact of pricing a toss-blind model against a stale line, not evidence.
5. **Aug 4/5 markets are seed liquidity** ($0–131 traded) and non-actionable.
   Aug 4/5 XIs are projections from the previous match, not teamsheets.
6. **Aug 3 is unresolved** and sealed; including it later changes the forward and
   combined numbers, and the seal hash pins the pre-result prediction.
7. **State is stale for the forward fixtures.** The SQLite cache stops at
   2026-07-13, so Aug 3/4/5 predictions run 21–23 days behind and required
   `--allow-stale-state`. `predict_fixture` marks them `stale_override` and
   suppresses any betting recommendation. Player-level form since 2026-07-13 —
   including the entire Hundred season — is absent from the career/ELO features;
   only the team-level form/H2H/home trackers see it.
8. **Prior finding stands:** the 2026-07-30 adaptation report found the model lands
   within ~3pp of the Polymarket line on 2026 fixtures and compresses to 0.37–0.62.
   Both hold here (mean |p − q| = 4.6pp, p ∈ [0.3729, 0.6214]). The model is a
   directional lean on The Hundred. **No edge, no betting.**
9. **One unresolved player id.** `0b189b5e` did not resolve to a Cricsheet name in
   the metadata CSV (one player, within the ≤2-per-XI tolerance); their career
   stats default. No fixture was skipped and no lineup failed closed.
10. **Not a promotion path.** `models/xgb_match_i7` is not the production match
    model (`models/xgb_match_i7_swap_production` is), and the Hundred path
    deliberately uses `xgb_match_i7` per the 2026-07-30 convention. The swap arm
    here is a diagnostic second read, not a candidate selection.

---

## 9. Files written

| path | what |
|---|---|
| `eval_out/hundred_roi_2026-08-03/preds_i7.json` | primary arm, all 18 matches |
| `eval_out/hundred_roi_2026-08-03/preds_swap.json` | swap arm, all 18 matches |
| `eval_out/hundred_roi_2026-08-03/preds_i7_maxdate0801.json` | leakage check 1 |
| `eval_out/hundred_roi_2026-08-03/preds_i7_cutaux0801.json` | leakage check 2 + forward state base |
| `eval_out/hundred_roi_2026-08-03/roi_eval.json` | full ledger + every metric here |
| `scripts/hundred_roi_eval.py` | the predeclared-rule evaluator |
| `scripts/seal_hundred_forward.py` | the forward sealer |
| `predictions/hundred/forward_2026-08-03_sealed.json` | sealed forward artifact |
| `predictions/hundred/<fixture>__{i7,swap}__{cut0801,full0802}.json` | 16 forward predictions |
| `fixtures/hundred/2026-08-0{3,4,5}_*.json` | 4 fixture definitions |
| `data/hundred/season_2026_men_v2_cut0801/` | 16-match truncated pool |
| `data/hundred/tracker_snapshot_2026-08-02_aux_hundred_v2.pkl` | full tracker snapshot (new name) |
| `data/hundred/tracker_snapshot_2026-08-01_aux_hundred_cut_v2.pkl` | frozen tracker snapshot (new name) |

Nothing under `models/`, `data/hundred/state/`, `data/hundred/season_2026_men/`,
`data/hundred/polymarket_odds_2026.json`, or the two pre-existing
`tracker_snapshot_*2026-07-26*.pkl` files was modified.

**Note on the seal.** The repo's `.gitignore` carries a blanket `*.json` rule, so
`predictions/hundred/forward_2026-08-03_sealed.json` and the four fixture JSONs are
untracked — the same treatment the pre-existing
`fixtures/hundred/2026-07-27_southern_brave_mi_london.json` gets. The seal's
integrity therefore rests on the sha256 recorded in this tracked report. If the
artifact itself should be committed, it needs an explicit `git add -f`.

---
---

# Settlement addendum (2026-08-05)

**Everything above this line is preserved verbatim as what was known on
2026-08-03.** No original finding, table, or number has been edited. This
addendum only settles the four fixtures that were sealed and unresolved at the
time, and restates the forward/combined tables with those results folded in.

**Nothing was re-predicted.** The sealed probabilities are used exactly as
sealed; the only new input is a winners-only results file.

## A.1 Seal verification

```
$ shasum -a 256 predictions/hundred/forward_2026-08-03_sealed.json
caecbe114b02dae0e8bc11e0dcbf91c145e446de7c3873000ed6259ac81746fd
```

Matches the sha256 recorded in § 7 **exactly**. Verified before any result was
looked up and re-verified after the settlement run; unchanged both times. The
seal holds — every probability settled below is the pre-result probability.

## A.2 Results, with sources

All four fixtures are **finished**. The last of them (Trent Bridge, 17:30 UTC
start) had completed well before the settlement run at 2026-08-05T23:27Z. No
fixture is left in progress, and nothing below is a projection.

ESPNcricinfo blocks direct fetch (403), so its results are taken from search-result
summaries of its own scorecard/report pages and each is corroborated by at least
one fully independent outlet. **Wikipedia's "2026 The Hundred season" page was
consulted and discarded**: the extraction returned internally contradictory
margins (e.g. a side "winning by 5 wickets" while scoring 144 chasing 159). It is
not used as a source for anything here.

| date | fixture (team1 v team2) | winner | margin | scores | in cricsheet? | finished? | independent sources |
|---|---|---|---|---|---|---|---|
| 2026-08-03 | Welsh Fire v Southern Brave (`1521249`) | **Welsh Fire** | 6 wickets | SB 115/8; WF 116/4 | **yes** | yes | cricsheet `1521249.json` (`outcome.winner`); ESPNcricinfo report *"Short trumps Stoinis as Fire keep pace with Rockets"*; Sky Sports |
| 2026-08-04 | Sunrisers Leeds v London Spirit (`1521250`) | **Sunrisers Leeds** | 37 runs | SRL 241/2; LS 204 | no | yes | ESPNcricinfo scorecard header *"Sunrisers won by 37 runs"* + report; Sky Sports *"…after scoring 241 to set highest total in tournament history"* |
| 2026-08-05 | Man Super Giants v Welsh Fire (`1521251`) | **Man Super Giants** | 9 wickets | WF 155/4; MSG 161/1 (69 balls) | no | yes | cricket.com.au report 4554800; ESPNcricinfo report; cricketworld.com |
| 2026-08-05 | Trent Rockets v Birmingham Phoenix (`1521252`) | **Trent Rockets** | 7 wickets | BP 111/6; TR 116/3 (83 balls) | no | yes | cricket.com.au report 4554800; ESPNcricinfo scorecard (PoM L Gregory 2/20) |

**Cricsheet status.** `hnd_json.zip` was re-downloaded on 2026-08-05. It now
publishes **`1521249` only** (copied into `data/hundred/season_2026_men_v2/`,
which goes 18 → 19 matches). Its recorded outcome — Welsh Fire by 6 wickets —
agrees with the web sources, which is the only fixture where an authoritative
ball-by-ball cross-check was possible. **`1521250`, `1521251` and `1521252` are
not yet published**; those three are settled from web sources alone. Since only
the Aug-3 fixture carries a bet, the single cricsheet-confirmed result is also
the only one that moves a P&L number — the other three affect probability
diagnostics only.

## A.3 The settled bet

Exactly one sealed fixture had a real (non-seed) market, as predeclared. Aug 4
and Aug 5 remain **NO BET**: their only quotes are seed liquidity ($0–131
traded), and they were **not** opportunistically settled at those prices.

| item | value |
|---|---|
| fixture | 2026-08-03 Welsh Fire v Southern Brave, Sophia Gardens (`1521249`) |
| sealed model p(WF) | **0.4655** (i7) / **0.4189** (swap) — both < market |
| pre-toss market p(WF) | 0.515 |
| rule's bet (both arms) | **Southern Brave @ 0.485** (dec. 2.062), 1 unit |
| market volume | $43,137 |
| result | **Welsh Fire won by 6 wickets** |
| **P&L** | **−1.000 unit (LOSS)** |

Both arms backed the same side at the same price, so the settlement is identical
for both: the swap arm's larger edge (9.6pp vs 4.95pp) bought it nothing. This
was a market-underdog bet (price 0.485 < 0.5) — the failure mode § 6.1 named.

## A.4 Updated FORWARD slice — PRE-TOSS (headline basis)

| date | fixture | model p(t1) i7 | market p(t1) | side backed | price | winner | P&L |
|---|---|---|---|---|---|---|---|
| 2026-08-02 | Trent Rockets v Sunrisers Leeds | 0.5504 | 0.515 | Trent Rockets | 0.515 | Trent Rockets | +0.942 |
| 2026-08-02 | MI London v Man Super Giants | 0.5662 | 0.545 | MI London | 0.545 | MI London | +0.835 |
| 2026-08-03 | Welsh Fire v Southern Brave | 0.4655 | 0.515 | Southern Brave | 0.485 | **Welsh Fire** | **−1.000** |
| 2026-08-04 | Sunrisers Leeds v London Spirit | 0.5725 | — (seed only) | — | — | Sunrisers Leeds | no bet |
| 2026-08-05 | Man Super Giants v Welsh Fire | 0.5603 | — (seed only) | — | — | Man Super Giants | no bet |
| 2026-08-05 | Trent Rockets v Birmingham Phoenix | 0.5750 | — (seed only) | — | — | Trent Rockets | no bet |

| arm | slice | | bets | wins | P&L (u) | flat ROI | model LL | market LL | coinflip LL |
|---|---|---|---|---|---|---|---|---|---|
| i7 | forward | **was (2026-08-03)** | 2 | 2 | +1.777 | +88.83% | 0.5830 | 0.6353 | 0.6931 |
| **i7** | **forward** | **now (settled)** | **3** | **2** | **+0.777** | **+25.89%** | **0.6435** | 0.6447 | 0.6931 |
| swap | forward | **was (2026-08-03)** | 2 | 2 | +1.777 | +88.83% | 0.5869 | 0.6353 | 0.6931 |
| **swap** | **forward** | **now (settled)** | **3** | **2** | **+0.777** | **+25.89%** | **0.6813** | 0.6447 | 0.6931 |

Forward log loss is scored on the 3 fixtures that have a market to compare
against; the model-only figures over all 6 forward fixtures are in § A.6.

## A.5 Updated COMBINED totals — PRE-TOSS (headline basis)

| arm | | fixtures | scored | bets | wins | P&L (u) | flat ROI | model LL | market LL | coinflip LL | model acc | market acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| i7 | was (2026-08-03) | 22 | 17 | 17 | 9 | +1.356 | +7.98% | 0.6897 | 0.6797 | 0.6931 | 64.7% | 64.7% |
| **i7** | **now (settled)** | 22 | **18** | **18** | **9** | **+0.356** | **+1.98%** | **0.6939** | 0.6788 | 0.6931 | 61.1% | 66.7% |
| swap | was (2026-08-03) | 22 | 17 | 17 | 10 | +3.342 | +19.66% | 0.6828 | 0.6797 | 0.6931 | 58.8% | 64.7% |
| **swap** | **now (settled)** | 22 | **18** | **18** | **10** | **+2.342** | **+13.01%** | **0.6932** | 0.6788 | 0.6931 | 55.6% | 66.7% |

The backtest slice is untouched (15 bets / 7 wins / −0.420u / −2.80% for i7;
15 / 8 / +1.566u / +10.44% for swap) — no backtest fixture was affected.

Supporting diagnostics on the settled combined pre-toss slice:

| arm | underdog bets | dog P&L | dog ROI | fav bets | fav P&L | fav ROI | vol-weighted ROI | expected wins | P(X ≥ wins), i.i.d.-optimistic |
|---|---|---|---|---|---|---|---|---|---|
| i7 | 11 / 18 (61.1%) | −2.070 | **−18.82%** | 7 | +2.427 | +34.67% | **−2.04%** | 8.730 | 0.542 |
| swap | 10 / 18 (55.6%) | −0.779 | **−7.79%** | 8 | +3.121 | +39.02% | **−4.02%** | 8.790 | 0.368 |

The § 6.1 pattern is now **stronger**, not weaker: the settled bet was an
underdog bet and it lost, pushing i7's underdog ROI from −10.70% to −18.82% and
flipping swap's from +2.45% to −7.79%. Volume-weighted ROI is negative for both
arms. The sign test moves away from significance for i7 (0.450 → 0.542).

> **Do not read the post-toss forward/combined diagnostic this time.** Aug 4 and
> Aug 5 have no pre-toss quote but *do* carry a post-toss seed price, so the
> post-toss basis now settles three bets at $0–131-volume placeholder prices
> (forward post-toss reads 6 bets / +3.406u / +56.77%). Those are not tradeable
> prices and that number is an artifact of settling against seed liquidity —
> precisely what the predeclared rule exists to prevent. It is retained in
> `roi_eval_settled_20260805.json` for completeness and is **void as a betting
> result**. The pre-toss headline above is unaffected: it correctly excludes all
> three.

## A.6 Probability accuracy on the newly-resolved fixtures (incl. NO-BET)

Directional accuracy is informative even where the rule placed no bet, so all
four newly-resolved fixtures are scored here. Probabilities are the sealed
`cut0801` (state frozen 2026-08-01) values — the ones the seal pins.

| date | fixture | winner | i7 p(t1) | i7 pick | i7 LL | swap p(t1) | swap pick | swap LL | bet? |
|---|---|---|---|---|---|---|---|---|---|
| 2026-08-03 | Welsh Fire v Southern Brave | Welsh Fire | 0.4655 | Southern Brave ✗ | 0.7646 | 0.4189 | Southern Brave ✗ | 0.8700 | **yes (lost)** |
| 2026-08-04 | Sunrisers Leeds v London Spirit | Sunrisers Leeds | 0.5725 | Sunrisers Leeds ✓ | 0.5578 | 0.5862 | Sunrisers Leeds ✓ | 0.5340 | no |
| 2026-08-05 | Man Super Giants v Welsh Fire | Man Super Giants | 0.5603 | Man Super Giants ✓ | 0.5792 | 0.6300 | Man Super Giants ✓ | 0.4620 | no |
| 2026-08-05 | Trent Rockets v Birmingham Phoenix | Trent Rockets | 0.5750 | Trent Rockets ✓ | 0.5533 | 0.6392 | Trent Rockets ✓ | 0.4476 | no |
| | **4 newly-resolved** | | | **3/4 (75%)** | **0.6137** | | **3/4 (75%)** | **0.5784** | 1 bet |

**The uncomfortable shape of this result: the model went 3/4 directionally, and
its single miss was the only fixture it was allowed to bet.** The three fixtures
it called correctly were all no-market fixtures. That is a coincidence at n = 4
and must not be read as a pattern, but it is exactly why the ROI line moved down
while the probability line moved up.

Model-only scoring (market quote not required), sealed probabilities:

| arm | slice | n | model LL | model Brier | model accuracy | coinflip LL |
|---|---|---|---|---|---|---|
| i7 | forward (all 6) | 6 | 0.6035 | 0.2055 | 83.3% (5/6) | 0.6931 |
| i7 | whole season (all 22) | 22 | 0.6797 | — | 63.6% (14/22) | 0.6931 |
| swap | forward (all 6) | 6 | 0.5812 | 0.1950 | 83.3% (5/6) | 0.6931 |
| swap | whole season (all 22) | 22 | 0.6675 | — | 59.1% (13/22) | 0.6931 |

**Read this narrowly.** On all 22 fixtures the model-only LL (0.6797 i7 / 0.6675
swap) now edges under coinflip, but that comparison is *not* against the market —
it includes the four fixtures where no market existed to compare with. On the 18
fixtures that have a matched market, i7 remains **worse than the market**
(0.6939 vs 0.6788) and still marginally worse than a coin (0.6931). § 5's
load-bearing conclusion is unchanged.

**State-basis note.** Only Man Super Giants v Welsh Fire moved between the two
sealed state bases (§ 2). Both bases called it correctly: i7 `full0802` p = 0.5382
(LL 0.6196) vs `cut0801` 0.5603 (LL 0.5792); swap `full0802` 0.6069 (LL 0.4994)
vs `cut0801` 0.6300 (LL 0.4620). The frozen basis happened to score better on
this one fixture. n = 1; no conclusion.

## A.7 What this does and does not change

**Changes:**

1. The forward slice is no longer 2-for-2. It is **3 bets, 2 wins, +0.777u,
   +25.89%** — the +88.83% figure in § 3 was, as that section said, arithmetic on
   two fixtures, and the third fixture took most of it back.
2. i7 combined flat ROI falls **+7.98% → +1.98%**; swap combined **+19.66% →
   +13.01%**. Both arms lose exactly 1.000u.
3. The § 6.1 underdog failure mode is reinforced: i7 underdog ROI −10.70% →
   −18.82%, swap +2.45% → −7.79%. Both arms' volume-weighted combined ROI stays
   negative.
4. Directionally the model went 3/4 on the newly-resolved fixtures, and its
   forward-slice model-only LL (0.6035 i7 / 0.5812 swap over 6) is its best
   stretch of the season.

**Does not change:**

1. **n is still tiny and this is still one I3 block.** 18 settled bets in one
   tournament = one `tournament_time_block_v1` block. No confidence interval
   exists or may be quoted. Everything here remains **DESCRIPTIVE**. Adding one
   bet does not move it toward inference.
2. **The probabilities still lose to the market.** Combined matched-market LL:
   i7 0.6939 vs market 0.6788; swap 0.6932 vs 0.6788. The market also out-picks
   both arms (66.7% vs 61.1% / 55.6%).
3. **No edge and no betting.** The 2026-07-30 conclusion — the model is a
   directional lean on The Hundred with no demonstrated market edge — stands
   unaltered. Caveats § 8.1–8.5 and § 8.7–8.10 apply verbatim; § 8.6 ("Aug 3 is
   unresolved") is now discharged by this addendum.
4. **Nothing was refit, retuned, or re-selected.** No model, calibrator,
   threshold, or rule was touched. The rule settled the one fixture it declared
   in advance, at the price it declared in advance, and lost.

## A.8 Files written / modified by the settlement

| path | status | what |
|---|---|---|
| `predictions/hundred/forward_2026-08-03_sealed.json` | **unmodified** (hash re-verified) | the seal |
| `eval_out/hundred_roi_2026-08-03/roi_eval.json` | **unmodified** | pre-settlement ledger, kept as-is |
| `eval_out/hundred_roi_2026-08-03/roi_eval_settled_20260805.json` | new | settled ledger + all metrics in this addendum |
| `data/hundred/forward_results_2026-08-05.json` | new | winners-only results file (no prices, no probabilities) |
| `data/hundred/season_2026_men_v2/1521249.json` | new | the one newly-published cricsheet match (18 → 19) |
| `scripts/hundred_roi_eval.py` | extended | new optional `--forward-results`; new `model_only_DIAGNOSTIC` block |
| `reports/hundred_roi_backtest_2026-08-03.md` | appended | this addendum only; §§ 1–9 untouched |

**Evaluator extension.** `--forward-results PATH` ingests a winners-only JSON
(keyed by cricsheet id and/or date + teams) to fill results for fixtures that
were unresolved when the odds pull was taken. It carries no price and no
probability, so it structurally cannot alter a sealed prediction. A companion
`model_only_DIAGNOSTIC` block scores every fixture with a known winner regardless
of whether a market quote exists, so NO-BET fixtures still contribute directional
accuracy. Both additions are purely additive: **re-running without
`--forward-results` reproduces `roi_eval.json` exactly** — verified, the only
difference being the `inputs.aliases` provenance string.

Reproduce:

```bash
uv run python scripts/hundred_roi_eval.py \
  --odds data/hundred/polymarket_odds_2026_v2.json \
  --aliases data/hundred/team_aliases_2026.json \
  --arm i7=eval_out/hundred_roi_2026-08-03/preds_i7.json \
  --arm swap=eval_out/hundred_roi_2026-08-03/preds_swap.json \
  --forward-arm i7=eval_out/hundred_roi_2026-08-03/preds_i7_cutaux0801.json \
  --fixture-pred i7=predictions/hundred/2026-08-03_welsh_fire_southern_brave__i7__cut0801.json \
  --fixture-pred i7=predictions/hundred/2026-08-04_sunrisers_leeds_london_spirit__i7__cut0801.json \
  --fixture-pred i7=predictions/hundred/2026-08-05_manchester_super_giants_welsh_fire__i7__cut0801.json \
  --fixture-pred i7=predictions/hundred/2026-08-05_trent_rockets_birmingham_phoenix__i7__cut0801.json \
  --fixture-pred swap=predictions/hundred/2026-08-03_welsh_fire_southern_brave__swap__cut0801.json \
  --fixture-pred swap=predictions/hundred/2026-08-04_sunrisers_leeds_london_spirit__swap__cut0801.json \
  --fixture-pred swap=predictions/hundred/2026-08-05_manchester_super_giants_welsh_fire__swap__cut0801.json \
  --fixture-pred swap=predictions/hundred/2026-08-05_trent_rockets_birmingham_phoenix__swap__cut0801.json \
  --forward-results data/hundred/forward_results_2026-08-05.json \
  --out-json eval_out/hundred_roi_2026-08-03/roi_eval_settled_20260805.json
```
