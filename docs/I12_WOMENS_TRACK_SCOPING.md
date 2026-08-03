# I12 — Women's-corpus model: scoping memo + v1 results

**Status: V1 BUILT AND GATED (2026-07-31, interactive track).** Results at
the bottom; scoping preserved below as written.

> **Correction (2026-08-01): the "no odds exist" premise below is wrong.**
> A women's-scoped Polymarket pull finds 266 resolved women's T20 markets and
> joins 175 of them to w1 fixtures. See "Market data (2026-08-01)" at the
> bottom. The scoping text is left unedited as the record of what was believed
> at build time.

## Data reality (verified today)

- `t20s_json.zip` in the stat-generator cricsheet mirror holds **2,086
  women's T20Is** (IDEAS said 1,745 — the mirror has grown), 2009-06-18 →
  2026-07-12, 90 distinct national sides. Zero-download coverage, currently
  filtered out of every extraction.
- Women's *league* cricket (WBBL, WPL, the Hundred women's comp) is **not**
  on disk — the mirror's league zips are men's competitions. Adding leagues
  means new cricsheet downloads (small, but a scope decision).
- Polymarket check (2026-07-23 capture): only 4 women's fixture markets, all
  from a minor state league (Gulbarga Mystics W, Hubli Tigers W, …). **There
  is no odds-based eval gate available today.** Women's markets appear
  seasonally (T20 World Cup, WBBL Oct–Dec, WPL Feb–Mar).

## Pipeline fit

The whole stack is gender-agnostic: cricsheet schema is identical, and the
extraction filter is the only place gender enters. A women's track needs:

1. Extraction pass with `gender == female` into `data/w_t20s_json/`.
2. **Fully isolated artifact family** (cache, parquets, model dirs, e.g.
   `models/player_stats_cache_w1.sqlite`, `data/xgb_match_data_w1/`) — same
   isolation discipline as the i5/i7 families. Zero player/venue overlap
   assumptions; venue canonicalization can reuse `venue_aliases_v1` but
   needs a coverage check on women's-only grounds.
3. Match-level direct model first (M7 config as the starting point), ball
   model later only if props ever matter. Priors, ELO K, and shrinkage k's
   should be re-fit, not inherited — base rates differ (scoring rates,
   boundary/six mix, wicket rates).

Effort: pipeline + first model ≈ one night-loop idea once specced; the eval
design below is the interactive part.

## Eval design (the part IDEAS flags as missing)

No odds → no market-LL or ROI gate. Honest v1 gates:

- Chronological holdout (e.g. train ≤2024, test 2025→): **LL vs coinflip**
  and vs a naive ELO-only baseline. The Blast precedent (LL 0.672, 64.7%
  acc, no market edge) is the right shape of claim.
- Resolution check: mean |p−0.5| vs the men's model's 0.105 on its T20
  control — the Hundred experience says compressed probabilities are the
  failure mode to watch on thin corpora.
- Market comparison deferred until a WBBL/T20WC Polymarket season is
  captured live (Oct 2026 earliest realistic window).

## Decisions required from you

- **W1 — Why build it**: portfolio/coverage (defensible now) vs betting
  (no market to bet until at least WBBL — an eval-only track for months).
  This decides priority more than anything technical.
- **W2 — Scope**: T20I-only (zero download, 2,086 matches) vs T20I+leagues
  (new downloads, adds WBBL/WPL depth where future markets actually are).
  Recommend T20I-only for v1.
- **W3 — Queue placement**: spec it as a night-executable idea (pipeline +
  v1 model + coinflip/ELO gates are all mechanical once designed) or keep
  fully interactive.

---

## v1 results (2026-07-31, T20I-only scope)

Pipeline: `scripts/extract_womens_t20s.py` (2,086 female T20Is →
`data/w_t20s_json/`) → `build_stats_cache.py --gender-filter female`
(`models/player_stats_cache_w1.sqlite`: 2,685 players, 268 venues, priors
from the women's corpus itself — never the men's) →
`materialize_match_features.py --gender female` (flag added; was hardcoded
male) → `xgboost_match_v1.py` M7 config + monotone, seed 29, base and swap
arms (`models/xgb_match_w1_{base,swap}`). The swap-augment map was extended
to the full-superset frame schema — all 15 new diff columns verified
exactly t1−t2 before negation entries were added.

Splits (standard chronological contract): train 1,362 (2009→2024) /
val 192 / test 316 (2025-07→2026-04) / golden 182 (2026-04-17→07-12).

| arm | split | n | LL | acc | resolution |
|---|---|---:|---:|---:|---:|
| coinflip | — | — | 0.6931 | — | 0.000 |
| elo-logistic baseline | test | 316 | 0.6537 | 0.614 | 0.114 |
| **w1_base** | test | 316 | **0.4484** | **0.810** | 0.256 |
| w1_swap | test | 316 | 0.4562 | 0.794 | 0.252 |
| elo-logistic baseline | golden | 182 | 0.6559 | 0.582 | 0.092 |
| w1_base | golden | 182 | 0.5174 | 0.753 | 0.263 |
| **w1_swap** | golden | 182 | **0.5045** | **0.769** | 0.251 |

**Both v1 gates pass on both splits**: the model beats coinflip and the
ELO-only baseline by wide margins, and probabilities are NOT compressed
(resolution ~0.25 vs the Hundred's 0.036 failure mode). **Reference arm:
`xgb_match_w1_base`.**

**Paired 5-seed swap test (2026-08-01, A1 seeds {7,13,29,42,101})**: the
split verdict is confirmed, not resolved — swap is WORSE on test (2/5
seeds better, mean ΔLL +0.0067) and BETTER on golden (4/5, mean −0.0148;
floor 0.007). Because the two windows disagree, the D12 swap effect does
not transfer cleanly to this corpus and **base stays the v1 reference**;
re-test when another season of women's data accrues. Arms in
`models/auto/i12_seeds/` (gitignored).

Honest caveats: (1) the T20I pool is full of full-member-vs-associate
mismatches, which inflates separability relative to the men's league-heavy
eval pools — a members-only slice is the sharper follow-up claim; (2) the
w1 frame is the full feature superset including groups the men's line
dropped (M3–M6) — no ablation done; (3) player-metadata coverage for
women's players in `all_players_enriched.csv` is unaudited (unknowns
default-rate); (4) **no market comparison exists** — this is model-vs-naive
skill only, no edge claim of any kind. *(Caveat 4 is superseded by the
section below; the market comparison now exists and w1 loses it.)*

---

## Market data (2026-08-01) — the odds do exist

The scoping memo concluded there was "no odds-based eval gate available
today" from a 2026-07-23 capture that surfaced only four women's markets.
That capture was **men's-scoped**. The strict extractor's gender guard keys
partly on slugs, and women's internationals are the one case where the slug
gives nothing away: Gamma lists them as plain `crint-nam4-uga2-2026-07-27`
with outcomes `["Namibia", "Uganda"]`, and the only gender signal is the
event title (`"T20 Namibia Quadrangular Series, Women: Namibia vs Uganda"`).
Filtering by slug therefore hides exactly the fixtures this track needs.

A female-scoped pull
(`extract_match_prematch_odds_strict.py --gender female --format t20`,
2025-07-01 → 2026-07-12) returns **266 resolved women's T20 markets**, held
in the sibling repo at
`data/polymarket_match_odds_strict_female_2025-07-01_2026-07-12.json`.

`scripts/build_womens_polymarket_odds.py` joins them to the w1 fixture pool
and writes `data/womens_polymarket/` (isolated; no existing odds set is read
or written). **175 of 498 w1 fixtures joined** — 129/182 golden (71%) and
46/316 test (15%). The test coverage is low for a structural reason, not a
matching failure: Polymarket's first women's cricket market in this window is
dated **2026-01-20**, so most of the test split predates any market at all.
Of the joined rows, 103 clear $10k volume, 51 clear $50k, 40 clear $100k.

Join integrity: all 175 matched at day offset zero, and the market's resolved
winner agreed with Cricsheet on **175/175** rows — an independent check,
since the winner is never used to select or orient a market. The join is
exact-date by contract: women's series repeat the same pairing on consecutive
days (Malawi played Botswana and Mozambique on adjacent dates in April 2026)
while Polymarket's event date can sit up to two days off Cricsheet's, so a
fuzzy window would risk pairing a market with the wrong leg of a series.
Near misses are recorded in `join_report.json` and never joined.

**Descriptive first look (log loss; no ROI — invariant 7 requires the I3
tournament-block contract for any economic claim):**

| slice | n | market LL | w1_base LL | w1_swap LL |
|---|---:|---:|---:|---:|
| all joined | 175 | **0.4674** | 0.5082 | 0.5096 |
| golden, all | 129 | **0.4895** | 0.5159 | 0.5085 |
| golden, ≥$10k | 94 | 0.5689 | 0.5676 | **0.5569** |
| golden, ≥$50k | 48 | **0.5093** | 0.5842 | 0.5619 |
| golden, ≥$100k | 38 | **0.4871** | 0.5676 | 0.5550 |

**The market beats w1 on log loss on every slice that matters.** The single
slice where a model arm edges ahead (golden ≥$10k, swap by 0.012) reverses at
≥$50k and ≥$100k, which is the wrong direction — the men's line's experience
is that a real calibration win *grows* with liquidity. Accuracy is closer
(swap 0.789 vs market 0.737 at ≥$100k), but accuracy is not the betting
metric and the men's Hundred work already showed picking winners without
resolution is worthless. **No edge is established and none should be
claimed.** The v1 skill result stands as written — beating coinflip and an
ELO baseline is real — but it is now bounded above by a market the model does
not beat.

Follow-ups this unblocks, in order: (1) a proper I3 block-resampled ROI
reading on the ≥$50k joined slice, which is the honest gate rather than the
LL table above; (2) the members-only slice, since the associate-heavy T20I
pool inflates separability and the market prices those mismatches too;
(3) re-pull after WBBL (Oct–Dec) and WPL to grow test-side coverage.

Regenerate the table with:

```bash
uv run python scripts/eval_womens_market.py \
    --odds data/womens_polymarket/betting_odds_womens_w1.json \
    --markdown reports/womens_market_eval_internationals.md
```

---

## I12-L (2026-08-01) — the women's league track, and what it costs w1

The memo above scoped v1 as T20I-only on the belief that "women's league
cricket is not on disk — adding leagues means new cricsheet downloads". The
downloads exist and are small; the competition codes just are not in the
stat-generator mirror. `scripts/extract_womens_leagues.py --download` pulls
six of them into an isolated pool:

| code | competition | matches | range |
|---|---|---:|---|
| wbb | Women's Big Bash League | 519 | 2015-12-18 → 2025-12-13 |
| ssm | Super Smash (women's) | 197 | 2020-01-02 → 2026-01-31 |
| hnd | The Hundred (women's) | 166 | 2021-07-21 → 2026-07-29 |
| cec | Charlotte Edwards Cup | 124 | 2021-06-26 → 2024-06-22 |
| wtb | Vitality Blast Women | 112 | 2025-05-30 → 2026-07-17 |
| wpl | Women's Premier League | 88 | 2023-03-04 → 2026-02-05 |

**1,206 matches → `data/w_league_json/`.** This is why it is worth having:
the league books are where the liquid women's money is. A single Hundred
Women fixture prices at $100k–$307k, against a typical women's T20I in the
low tens of thousands.

The w2 family is fully isolated from w1 and built with the same contracts
(schema v4, `inclusive_total_runs_v1`, `fixed_competition_k_v1`, M7 config,
monotone, seed 29). Franchise and international team namespaces share no ELO
history, so they are modelled separately rather than pooled.

```bash
uv run python scripts/extract_womens_leagues.py --download
uv run python scripts/build_stats_cache.py --source-dir data/w_league_json \
    --gender-filter female --out models/player_stats_cache_w2.sqlite
uv run python scripts/materialize_match_features.py \
    --source-dir data/w_league_json --sqlite-dir models --version w2 \
    --out-dir data/xgb_match_data_w2 --gender female
uv run python scripts/xgboost_match_v1.py --cmd both \
    --data-dir data/xgb_match_data_w2 --model-dir models/xgb_match_w2_base \
    --monotone --seed 29
```

Splits: train 878 (2015→2024-12) / val 79 / test 145 (2025-07→2026-02) /
golden 64 (2026-05-22→2026-07-29). Cache: 683 players, 110 venues.

### w2 fails both v1 gates

| arm | split | n | LL | acc | resolution |
|---|---|---:|---:|---:|---:|
| coinflip | — | — | 0.6931 | — | 0.000 |
| elo-logistic baseline | test | 145 | **0.6781** | 0.552 | 0.075 |
| w2_base | test | 145 | 0.6858 | 0.497 | 0.080 |
| w2_swap | test | 145 | 0.6798 | 0.579 | 0.102 |
| elo-logistic baseline | golden | 64 | **0.6671** | 0.641 | 0.098 |
| w2_base | golden | 64 | 0.6931 | 0.562 | 0.107 |
| w2_swap | golden | 64 | 0.7238 | 0.531 | 0.156 |

The ELO-only baseline beats both league arms on both splits, base sits
exactly at coinflip on golden, and swap is worse than coinflip. **The league
model has no demonstrated skill and is not a reference arm for anything.**

**This is the most important result on the women's track, and it is about
w1, not w2.** Caveat (1) above warned that the T20I pool's full-member vs
associate mismatches inflate separability. w2 measures that inflation: hand
the same pipeline a corpus of evenly-matched franchise sides and its
LL goes from 0.4484 to 0.6858 and its accuracy from 81% to 50%. w1's
headline is largely the model learning that Australia beats Indonesia — real,
but nearly all of it is roster mismatch, not cricket modelling. A
members-only w1 slice is now the *required* follow-up, not an optional one.

### League market join

51 league fixtures joined (WPL 10, Blast Women 33, Hundred Women 8), zero
winner conflicts, in `data/womens_polymarket_leagues/`. Blast Women League 2
(19 markets), the Maharani Trophy (11) and CSA T20 Women (3) have no
cricsheet counterpart and stay unjoined; three late-July Hundred markets
post-date the cricsheet pull.

Read the league market table (`reports/womens_market_eval_leagues.md`) with
the coinflip guard in mind. An arm "beats the market" on 7 of 12 slices —
and **0 of those 7 are informative**, because on every one of them either the
market itself scored worse than a coinflip or n < 30:

| slice | n | market LL | w2_base LL |
|---|---:|---:|---:|
| ALL | 51 | **0.6831** | 0.7084 |
| ALL ≥$10k | 27 | 0.6950 (worse than coinflip) | 0.6998 |
| ALL ≥$50k | 12 | 0.7082 (worse than coinflip) | 0.6536 |
| ALL ≥$100k | 8 | 0.7112 (worse than coinflip) | 0.6661 |

The ≥$100k league slice is eight matches; the market priced every one between
0.425 and 0.60 and called four of eight wrong. Beating a line like that is
not evidence of anything, which is why `eval_womens_market.py` flags
market-worse-than-coinflip slices with `!` and reports an explicit
informative-slice count. **No edge on leagues either.**

Follow-ups: (1) the members-only w1 slice, now the headline question;
(2) re-pull Hundred Women odds after the 2026 season completes — it is the
only women's book deep enough to test an edge against, and this season is
still in progress; (3) WBBL Oct–Dec 2026 will be the first league season with
both a market and a completed cricsheet record from the start.
