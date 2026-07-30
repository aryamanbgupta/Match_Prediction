# Adapting the match-winner model to The Hundred — 2026-07-27

## Question

The Hundred 2026 is in progress (2026-07-21 → 2026-08-16, 34 men's matches).
Can the production match-level winner model predict its fixtures, how would it
have done on the matches already played, and what does it say about the next
one?

Short answer: it runs, it picks the winner about as often as it does in T20
(63.5% on 159 historical Hundred matches, binomial p = 0.0004), but its
probabilities are compressed to a 0.37–0.62 band, so log loss barely beats a
coinflip and there is no basis for betting. Its numbers sit within ~3
percentage points of the Polymarket line on every 2026 fixture.

## What The Hundred changes

| | T20 (what the model was trained on) | The Hundred |
|---|---|---|
| Innings length | 120 balls, 6-ball overs | 100 balls, 5-ball overs |
| Franchises in the corpus | yes | **no** — never parsed into training data |
| Venues | — | all 8 grounds already in the corpus via the Blast / internationals |
| Players | — | overwhelmingly the same T20 franchise pool |
| 2026 team names | — | three sides renamed after the 2025 ownership sales |

The match-level model never sees a ball. It consumes squad-aggregate career
stats, ELO, venue profile, form/H2H/home flags and toss. None of that is
arithmetically tied to 120 balls, which is why the model can be pointed at a
100-ball fixture at all. What it cannot know is that a 100-ball match is
higher-variance than a T20, so its probabilities should be pulled toward 0.5
relative to a T20 with the same squads. In practice this turned out to be moot
— they are already pinned near 0.5 for a different reason (below).

## What was built

**Data**

- `data/hundred/context_hnd_json/` — 167 men's Hundred matches (2021–2025)
  extracted from Cricsheet's `hnd_json` pool. Context state only: they are
  never used to build a stats cache or to train anything.
- `data/hundred/season_2026_men_source.json` + `data/hundred/season_2026_men/`
  — the eight completed 2026 matches, hand-transcribed from ESPNcricinfo
  scorecards (Cricsheet has not published the 2026 season; its latest Hundred
  match is 2025-08-31). Written in Cricsheet shape, `info` only, no deliveries.
- `data/hundred/team_aliases_2026.json` — the three franchise renames
  (Oval Invincibles → MI London, Northern Superchargers → Sunrisers Leeds,
  Manchester Originals → Manchester Super Giants), folded into tracker history
  so the renamed sides are not treated as expansion teams.
- `data/hundred/polymarket_odds_2026.json` — pre-match Polymarket prices for
  the completed fixtures, read from CLOB price history at a cutoff strictly
  before the scheduled start. The sibling polymarket-cricket capture stops at
  2026-07-23, so `scripts/fetch_hundred_polymarket.py` goes to the public API
  directly. Match 1 (2026-07-21) has no Polymarket event; the other seven do,
  at $84k–$642k volume.

**State**

- `data/hundred/state/player_stats_cache_i7.sqlite` — T20-only stats cache
  through 2026-07-13, canonical venue identities, priors frozen from the
  pre-holdout cache. 9,920 matches. Built with the documented non-destructive
  refresh recipe; production caches untouched.
- `data/hundred/tracker_snapshot_2026-07-26_aux_hundred.pkl` — form/H2H/home
  trackers over the 9,920-match primary T20 pool **plus** 175 auxiliary
  Hundred matches.

**Code**

- `scripts/build_hundred_matches.py` — extract history, build the 2026 records,
  hard-fail on any unresolved player name.
- `scripts/backtest_hundred.py` — score any directory of Cricsheet-shaped
  matches through the exact `compute_features` / `apply_encoders_and_predict`
  path `predict_fixture.py` uses live, so backtest and live numbers mean the
  same thing. `--hide-toss` reproduces live pre-toss conditions.
- `scripts/fetch_hundred_polymarket.py` — Gamma + CLOB odds pull.
- `scripts/predict_fixture.py` — three changes, described under "Changes to the
  live path".

## Which model, and why not M7

`models/xgb_match_v3_m7_production` is the production model of record. It can
still be replayed through `predict_fixture.py --venue-identity-mode legacy`,
but that mode is deliberately restricted to the frozen pre-I7 artifact family.
It cannot be paired with the canonical-venue state built for this new Hundred
workflow, and legacy mode must not acquire new data or feature capabilities.

So everything here uses `models/xgb_match_i7` — same 48 features, same
hyperparameters, retrained on canonical venue identities — paired with a
canonical-venue cache. I7 was evaluated as *slightly worse* than M7 on the
iteration set (≥$50k LL 0.6421 vs 0.6299, ROI +17.49% vs +21.90%), though no
delta was competition-block CI-clean, and it was deliberately not promoted.
Read every number below as "the contract-clean model", not "the production
model". The gap is small relative to the effect being measured.

## Results

All runs: `models/xgb_match_i7`, cache through 2026-07-13, Hundred-inclusive
trackers with renames folded in, `--hide-toss` (pre-toss, as live).

### The Hundred, 2021–2026 (167 scored, 8 no-result)

| season | n | log loss | Brier | accuracy | mean \|p−0.5\| |
|---|---:|---:|---:|---:|---:|
| 2021 | 31 | 0.6758 | 0.2413 | 67.7% | 0.034 |
| 2022 | 34 | 0.6856 | 0.2463 | 61.8% | 0.030 |
| 2023 | 29 | 0.6686 | 0.2378 | 69.0% | 0.036 |
| 2024 | 32 | 0.6821 | 0.2446 | 56.2% | 0.026 |
| 2025 | 33 | 0.6858 | 0.2463 | 63.6% | 0.036 |
| **2026** | **8** | **0.7426** | **0.2742** | **37.5%** | 0.056 |
| all | 167 | 0.6829 | 0.2449 | 62.3% | 0.034 |

Coinflip log loss is 0.6931. Home sides won 48.5% of these matches, so
"back the host" is not a baseline worth beating.

Five seasons is fewer than the ten blocks the evaluation contract requires
before a block interval means anything, so the per-season rows are descriptive.
On the historical seasons pooled (2021–2025, n = 159):

- **101/159 = 63.5% correct**, one-sided binomial p = 0.0004,
  i.i.d. bootstrap 95% CI [56.0%, 71.1%].
- Log-loss improvement over a coinflip **+0.0132** [+0.0013, +0.0252]. The
  interval excludes zero, but the point estimate is only about twice the
  repo's documented seed-noise floor of 0.007 LL.

### The compression is the finding

Same model, same code path, same state, scored over the 401 T20 matches in the
forward-holdout context pool (June–July 2026, all out of training):

| | The Hundred (167) | T20 control (401) |
|---|---:|---:|
| log loss | 0.6829 | **0.6270** |
| Brier | 0.2449 | **0.2206** |
| accuracy | 62.3% | 61.8% |
| mean \|p − 0.5\| | 0.034 | **0.105** |
| probability range | 0.373 – 0.621 | 0.089 – 0.924 |
| top-6-ELO favourite accuracy | 54.5% | 61.1% |

The model is *as accurate* on the Hundred as on T20 and roughly three times
less confident. Its reliability table says the same thing — 165 of 167 Hundred
predictions land in the 0.50–0.60 confidence bucket, where it actually wins
62.4%; the T20 control spreads across every bucket and tracks the diagonal.

That is what a deliberately balanced draft competition looks like through
squad-aggregate features: eight sides assembled from one player pool, at eight
grounds the model already knows, produce squad differentials too small to
separate. The raw ELO signal degrades the same way — the stronger top-6 wins
54.5% of Hundred matches versus 61.1% of T20s.

### Does Hundred history in the trackers help?

| tracker configuration | log loss | accuracy |
|---|---:|---:|
| T20 pool only (Hundred sides have no form/H2H/home) | 0.6862 | 56.9% |
| + Hundred history, renames folded in | **0.6829** | **62.3%** |

Worth having, and the reason for the auxiliary-pool mechanism. Competition tier
was also swept (tier 1 "unknown league" / tier 2 Blast / tier 3 BBL): it moves
individual probabilities by ~1pp and the aggregates not at all. Tier 2 (Blast,
the closest in-corpus analogue) is used throughout.

### 2026 so far, against the market

| date | fixture | model | market | winner | model | market |
|---|---|---:|---:|---|:--:|:--:|
| 07-21 | MI London v Sunrisers Leeds | 46.2% | — | MI London | ✗ | — |
| 07-22 | Southern Brave v Welsh Fire | 62.1% | 56.5% | Welsh Fire | ✗ | ✗ |
| 07-23 | London Spirit v Man Super Giants | 50.9% | 53.5% | Man Super Giants | ✗ | ✗ |
| 07-24 | Birmingham Phoenix v Trent Rockets | 46.2% | 41.5% | Birmingham Phoenix | ✗ | ✗ |
| 07-25 | Sunrisers Leeds v Southern Brave | 55.2% | 54.5% | Sunrisers Leeds | ✓ | ✓ |
| 07-25 | Welsh Fire v MI London | 40.3% | 37.5% | Welsh Fire | ✗ | ✗ |
| 07-26 | Man Super Giants v Birmingham Phoenix | 53.0% | 57.5% | Man Super Giants | ✓ | ✓ |
| 07-26 | Trent Rockets v London Spirit | 56.4% | 57.5% | Trent Rockets | ✓ | ✓ |

Probabilities are for the first-named (home) side. On the seven fixtures with a
market: **model LL 0.7382, market LL 0.7389**, both 3/7 correct, both worse than
a coinflip. Mean absolute disagreement with the market is **3.1 percentage
points** — the model is tracking the line, not beating it. Week one has simply
been upset-heavy, and eight matches say nothing either way.

### Sharpening: tempting, rejected

Under-confidence invites a temperature fix. Leave-one-season-out on the
historical seasons picks k ≈ 2.3–3.1 for `p' = σ(k·logit p)` and improves
out-of-fold log loss 0.6799 → 0.6734 — still barely past coinflip. Fitting k on
all of 2021–2025 (k = 2.70) and applying it forward to 2026 makes log loss
*worse*: 0.7426 → 0.8650. Combined with the repo's standing result that
recalibration destroys ROI on this model, sharpening is not adopted. Left here
as a documented dead end.

## Tonight's fixture

**2026-07-27, Southern Brave v MI London, The Rose Bowl, Southampton
(9th match)** — `predictions/hundred/2026-07-27_southern_brave_mi_london.json`

- **Southern Brave 48.2% / MI London 51.8%** (toss unknown; the two bat-first
  branches give 48.0% / 48.4%, so the toss is nearly irrelevant here).
- Polymarket at 17:02Z: Southern Brave 41.5% / MI London 58.5%, market volume
  $15.1k, liquidity $147k.
- Model edge on Southern Brave +6.7pp. **No A7 shadow bet**: volume is below
  the $50k gate and the edge is below the 10pp mismatch-regime threshold. A7
  was declared over IPL-style markets and its economic performance is
  unconfirmed; it has no standing in this competition.
- Feature picture: Brave at home, form 0.40 vs 0.70 over the last ten,
  H2H 0.25 across 6 prior meetings (the MI London slot, as Oval Invincibles,
  dominated this fixture), top-6 batting ELO −8.2 to the Brave, bottom-5
  bowling ELO +8.9 to them.

Lineups are each side's most recent XI, not confirmed teamsheets. The
historical backtest had actual XIs, which is an advantage a live prediction
never has.

## Changes to the live path

`predict_fixture.py` gained three things, all needed to make a non-T20
competition predictable without weakening a guard:

1. **`--tracker-aux-dir`** — competition pools that feed the form/H2H/home
   trackers but are not part of the SQLite state pool. Counted separately, so
   the SQLite/tracker source-count agreement check still means "built from the
   same primary pool", and `as_of` still reports primary coverage (a fresh
   Hundred result cannot disguise stale T20 state).
2. **`--team-aliases`** — folds a renamed franchise's tracker history into its
   current name.
3. **`--state-version`** — select `player_stats_cache_<version>.sqlite`, which
   was previously hardcoded to `v3`.

Two bugs surfaced and were fixed:

- **Cricsheet-style names did not resolve.** The fixture name lookup indexed
  only `name` and `full_name`, not `unique_name`, so a lineup pasted from a
  scorecard ("WG Jacks", "N Pooran") silently failed. The first run of tonight's
  fixture lost 9 of MI London's 11 players to default ratings and returned
  Southern Brave 67.2% — a confident, entirely wrong number produced from
  warnings alone.
- **Unresolved players were a soft warning.** Now more than two per XI raises,
  with `--max-unresolved-players` to override. One or two (an uncapped
  debutant) still warns.

Model artifacts are also loaded once per process instead of per call, which is
what makes a 175-match backtest through the live code path practical.

## Known limitations (review, 2026-07-30)

Three caveats qualify the numbers above; none changes the no-edge verdict,
but the exact headline figures should not be quoted without them.

1. **The alias fold handicapped the historical backtest — FIXED and rerun
   (2026-07-30).** The original 2026 team-alias merge `pop()`ed each
   renamed franchise's records out of the form/H2H/home trackers, so 99 of
   the 159 historical matches were scored with neutral form, zero H2H, and
   no home flag on at least one side. `merge_team_aliases` is now a
   **copy-fold** (old-name records retained for pre-rename queries; 2026
   fixtures unchanged), and the backtest was rerun
   (`eval_out/hundred/backtest_final_pre_toss_datefold.json`). Corrected
   2021–2025 numbers: **97/159 = 61.0%** (was 101/159 = 63.5%), LL
   **0.6763** (was 0.6799 on the same recompute), mean |p−0.5| 0.0358,
   i.i.d. binomial p **0.0034** (was 0.0004). Rows with a neutral-form side
   drop 124 → 62; the remainder are genuine no-history cases (season
   openers). So the handicap had been *inflating* directional accuracy
   while slightly worsening log loss; every conclusion — directional lean,
   heavy compression, no edge — survives with the corrected, less
   dramatic headline. The 2026 rows are identical under either fold.
2. **The pooled p-value assumes independence it doesn't have.** 159 matches
   from five seasons of the same eight teams are not i.i.d. Bernoulli
   trials — the same correlation concern I3 codified for ROI. A
   season-block sign test (5/5 seasons above 50%) gives p ≈ 0.03:
   still suggestive, much less dramatic than 0.0004.
3. **Snapshot provenance.** The shipped tracker snapshot was built by an
   intermediate code state and originally carried no
   `venue_identity_mode` key; it has been restamped `i7` (factual — it
   carries the active venue-alias contract hashes) and the live path now
   fails closed on aux-pool snapshots with undeclared mode, on legacy-mode
   serving of aux-pool snapshots, and on `--tracker-aux-dir`/
   `--team-aliases` in legacy mode. A7 shadow decisions now also carry
   `policy_scope_eligible`: Hundred fixtures (aux pools, aliases, or a
   non-T20 `format` field) are suppressed as
   `competition_out_of_policy_scope` rather than relying on generic
   liquidity/edge gates — the predeclared T20 shadow record stays clean.

## Verdict

The pipeline adapts to The Hundred cleanly, and the adaptation is honest about
what it produces: a directional lean that has been right **61%** of the time
across five seasons (corrected rerun, caveat 1; season-block sign test
p ≈ 0.03 per caveat 2), expressed as probabilities so close to 50% that
they carry almost no information beyond that lean. There is no edge to bet, no
disagreement with the market worth acting on, and eight matches of 2026 (3/8,
market also 3/7) neither confirm nor refute anything.

If this is worth pursuing, the useful next step is not calibration — it is
asking whether a competition of drafted, deliberately balanced squads is
separable by squad-aggregate features at all. The T20 control says the features
work; the Hundred result says these eight squads look alike to them.

## Reproduce

```bash
# Data
uv run python scripts/build_hundred_matches.py --extract-history --build-2026
uv run python scripts/fetch_hundred_polymarket.py

# State (T20-only cache through 2026-07-13, canonical venues; ~8 min)
uv run python scripts/build_stats_cache.py \
  --source-dir data/t20s_json \
  --extra-source-dir data/forward_holdout/2026-06-01_2026-07-13/context_t20s_json \
  --out data/hundred/state/player_stats_cache_i7.sqlite \
  --metadata-csv data/all_players_enriched.csv \
  --prior-source-sqlite models/player_stats_cache_i7.sqlite --force-rebuild

# Backtest (rebuilds the tracker snapshot with the Hundred as an aux pool)
uv run python scripts/backtest_hundred.py \
  --state-dir data/hundred/state --state-version i7 \
  --venue-identity-mode i7 \
  --model-dir models/xgb_match_i7 \
  --tracker-snapshot data/hundred/tracker_snapshot_2026-07-26_aux_hundred.pkl \
  --rebuild-snapshot \
  --tracker-source-dir data/t20s_json \
  --tracker-source-dir data/forward_holdout/2026-06-01_2026-07-13/context_t20s_json \
  --tracker-aux-dir data/hundred/context_hnd_json \
  --tracker-aux-dir data/hundred/season_2026_men \
  --team-aliases data/hundred/team_aliases_2026.json \
  --hide-toss --out-json eval_out/hundred/backtest_final_pre_toss.json

# T20 control
uv run python scripts/backtest_hundred.py \
  --source-dir data/forward_holdout/2026-06-01_2026-07-13/context_t20s_json \
  --state-dir data/hundred/state --state-version i7 \
  --venue-identity-mode i7 \
  --model-dir models/xgb_match_i7 \
  --tracker-snapshot data/hundred/tracker_snapshot_2026-07-26_aux_hundred.pkl \
  --hide-toss --out-json eval_out/hundred/control_t20_forward.json

# Predict a fixture (see fixtures/hundred/ for the shape)
uv run python scripts/predict_fixture.py \
  --fixture fixtures/hundred/2026-07-27_southern_brave_mi_london.json \
  --venue-identity-mode i7 \
  --model-dir models/xgb_match_i7 \
  --state-dir data/hundred/state --state-version i7 \
  --tracker-snapshot data/hundred/tracker_snapshot_2026-07-26_aux_hundred.pkl \
  --tracker-source-dir data/t20s_json \
  --tracker-source-dir data/forward_holdout/2026-06-01_2026-07-13/context_t20s_json \
  --tracker-aux-dir data/hundred/context_hnd_json \
  --tracker-aux-dir data/hundred/season_2026_men \
  --team-aliases data/hundred/team_aliases_2026.json \
  --out predictions/hundred/<fixture>.json
```
