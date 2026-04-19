# Polymarket Odds Ingestion & Test-Set Expansion

**Created**: April 2026
**Status**: All phases complete as of 2026-04-18. New XGBoost v3 baseline retrained on the post-split/gender-filter corpus and evaluated against the 261-match Polymarket set — results below.

Expands the betting eval set from the legacy 44 T20 WC 2024 matches to **261 matches** across T20 WC 2026, BBL 2025-26, ILT20 2025-26, SA20 2025-26, and bilateral internationals, using Polymarket pre-match prices as the market baseline.

## Split strategy (Option A)

Betting eval is a subset of the `test` split, driven by what's in `betting_odds_polymarket.json` at eval time (not by a parse-time event-name filter).

| Split       | Window                        |
|-------------|-------------------------------|
| train       | ≤ 2024-12-31                  |
| validation  | 2025-01-01 → 2025-06-30       |
| test        | 2025-07-01 → 2026-04-16       |
| golden_test | ≥ 2026-04-17 (empty; hook)    |

## New / modified files

| Path                                          | Role                                                                                     |
|-----------------------------------------------|------------------------------------------------------------------------------------------|
| `scripts/build_polymarket_odds.py`            | New. Reads Polymarket JSON + Cricsheet, emits matched odds file + copies test JSONs.     |
| `scripts/dry_run_splits.py`                   | New. Non-destructive sanity check for split counts + betting-eval intersection.          |
| `scripts/parsing_v2.py`                       | Modified. New date-split constants; parse-time `info.gender == 'male'` filter; old `betting_test` subset removed. |
| `betting_odds_polymarket.json`                | New artifact. 261 matches in `BettingOddsLoader` format.                                 |
| `data/polymarket_test/`                       | New artifact. 261 Cricsheet match JSONs copied from `data/t20s_json/`.                   |
| `data/polymarket_build_unmatched.json`        | New artifact. Audit log: unmatched markets + winner-disagreement report.                 |
| `data/betting_test/` + `betting_odds_v3.json` | Untouched. Legacy 44-match WC 2024 set remains usable by old configs.                    |

## Build pipeline

`uv run python scripts/build_polymarket_odds.py [--dry-run] [--verify-mapping]`

### Filters (1,161 → 261)

| Stage                                    | Dropped | Remaining |
|------------------------------------------|--------:|----------:|
| volume ≥ $1,000                          |     138 |     1,023 |
| not `low_liquidity`                      |      64 |       959 |
| resolved `winner` present                |      48 |       911 |
| both prematch prices present             |       0 |       911 |
| date in 2025-07-01 → 2026-04-16          |     102 |       809 |
| matched to Cricsheet (T20 male)          |     411 |       398 |
| dedup by `match_id`                      |     137 |       261 |

The 411 unmatched are mostly ODIs, Tests, U19 WC, women's T20s (WPL/WNCL), and the Afghanistan men's T20I coverage gap in our Cricsheet corpus.

### Team-name normalization

`TEAM_NAME_MAP` in `build_polymarket_odds.py` covers abbreviations (USA, UAE, Lanka, Kong), IPL single-city → franchise names, and `Emirates → United Arab Emirates` (Asia Cup context — `MI Emirates` is always spelled out in ILT20 records). Use `--verify-mapping` to dump unmapped Polymarket team names and fuzzy Cricsheet candidates before rerunning the build.

### Dedup: Polymarket YES/NO binary markets + in-play contamination

Polymarket runs each cricket fixture as two binary YES/NO markets ("Will Team1 win?" / "Will Team2 win?"). The upstream extractor (`polymarket-cricket/extract_prematch_odds.py`) emits one record per binary market, each with its own `winner` label and its own `prematch_price_team*` orientation.

A second contamination source: the upstream `prematch_price_team*` field is occasionally populated from an in-play or post-match snapshot with `max(p1, p2)` near 1.0. Raw inspection of siblings shows **61 fixtures** where one sibling has `top_p > 0.92` and another sibling for the same fixture has a plausible `top_p < 0.85`; 59 of those are recoverable by picking the plausible sibling.

Dedup tiebreak, in priority order (see `scripts/build_polymarket_odds.py::score`):
1. **Plausibility**: `max(prematch_price_team1, prematch_price_team2) ≤ 0.92`. Rejects in-play snapshots masquerading as prematch while keeping legitimate lopsided markets.
2. Polymarket `winner` matches Cricsheet `info.outcome.winner`.
3. Highest `volume_usd`.

When both siblings are implausible we still keep the best-scoring one rather than dropping the fixture; `actual_winner` always comes from Cricsheet regardless.

**Impact of plausibility prefilter** (same 261 matches, only the kept sibling changes):

| Metric (full 261)                  | Buggy dedup | Clean dedup | Δ            |
|------------------------------------|------------:|------------:|-------------:|
| shipped prices with `top_p > 0.92` | 23          | 9           | −14 recovered|
| shipped prices with `top_p > 0.99` | 3           | 0           | −3           |
| market log loss                    | 0.5917      | 0.6267      | +0.035       |
| model flat ROI                     | −7.60%      | +0.06%      | +7.7 pp      |
| model beats market (count)         | 100 / 255   | 110 / 255   | +10          |

Model log loss and Brier do not change (predictions are independent of the odds file). The buggy numbers made the market look falsely prescient (its "prematch" price was actually post-match) and fabricated 49%+ "edges" that the model kept betting into and losing. Both effects disappear with the fix.

Residual disagreements post-dedup: **7** (vs 1 before). The rise is an expected side-effect of preferring the plausible sibling over the matches-Cricsheet sibling in the 7 cases where the plausible one has the "wrong" orientation — benign because team1/team2 prices in each market are oriented to the fixture's teams (not the YES/NO orientation), so the kept prices are still correctly aligned.

## Parse-time gender filter

`parsing_v2.py` now skips any match with `info.gender != 'male'` before any feature engineering. This was a pre-existing correctness gap — ELO was gender-segregated but ball outcomes still trained on women's matches (which have different scoring distributions). Effect on the current corpus: **1,745 women's matches filtered** (not "near-zero" as originally assumed; cricsheet's archives bundle women's fixtures).

## Verification

Dry-run sanity check:

```
uv run python scripts/dry_run_splits.py
```

Last result (2026-04-18):

| Metric              | Value                                    |
|---------------------|------------------------------------------|
| train               | 8,152                                    |
| validation          | 545                                      |
| test                | 822                                      |
| golden_test         | 0 (expected — cutoff 2026-04-17)         |
| women's filtered    | 1,745                                    |
| betting_eval subset | 261 (odds ∩ test-window match IDs)       |

Test-split events look sane (Vitality Blast, T20 WC, BBL, ILT20, BPL, CPL, SA20, Super Smash, …) and dates spread evenly across 2025-07 → 2026-04.

## Phase 4 — destructive rebuild (completed 2026-04-18)

Before rebuilding, the old artifacts were renamed rather than overwritten so we had an atomic rollback path:

```
mv data/xgb_data_v3      data/xgb_data_v3_old        (159 MB)
mv models/cache_chunks_v3 models/cache_chunks_v3_old  (8.9 GB)
mv models/xgb_v3          models/xgb_v3_old           (23 MB)
uv run python scripts/parsing_v2.py                   (~10-15 min)
```

Post-rebuild sanity check (`/tmp/claude/post_rebuild_sanity.py`) passed:

| Check                                            | Value                           |
|--------------------------------------------------|---------------------------------|
| train parquet rows                               | 1,876,971                       |
| validation parquet rows                          | 124,292                         |
| test parquet rows                                | 186,667                         |
| cache chunks generated                           | 75                              |
| cache snapshot dates                             | 2005-02-17 → 2026-04-16 (3,709) |
| `data/polymarket_test/*.json` count              | 261                             |
| odds match_ids resolving to test JSONs           | 261 / 261                       |
| StatsProvider temporal lookup (random v3 batter) | returns stats cleanly           |

## Phase 5 — retrain + sim-eval (completed 2026-04-18)

Config: `experiments/configs/xgb_v3_polymarket.yaml` (v2 Optuna hyperparams, `n_sims: 100`, `remove_margin=False` handled in loader).

```
uv run python scripts/run_experiment.py \
    experiments/configs/xgb_v3_polymarket.yaml --skip-parsing
```

Runtime: ~4 min training + ~38 min sim-eval (exit 0). Experiment dir: `experiments/results/xgb_v3_polymarket_20260418_223639_9848fb3/`.

### Training (ball-level, XGBoost v3, 63 features)

| Metric              | Value  |
|---------------------|-------:|
| val log loss        | 1.6447 |
| test log loss       | 1.6379 |
| val accuracy        | 0.3159 |
| test accuracy       | 0.3181 |
| n_estimators used   | 444    |

Top features: `is_middle_overs`, `is_pace`, `is_death_overs`, `score`, `balls_bowled`, `batter_runs_scored`, `balls_ratio`, `batter_balls_faced`, `wickets`, `wickets_in_hand`.

### Sim-eval (match-level, 261 Polymarket matches, 100 sims each)

Two numbers are needed: a **full-set baseline** on all 261 matches (maximum data, keeps thin-market noise) and a **high-liquidity subset** so readers can see what the numbers look like once thin markets are stripped out. No single threshold is right — report both and let downstream decisions pick.

**Headline framing (full 261, clean dedup):**

| Log loss           | value  | n   |
|--------------------|-------:|----:|
| Polymarket         | 0.6267 | 255 |
| Coinflip (50/50)   | 0.6931 | —   |
| Our model          | 0.7319 | 255 |

Our model beats Polymarket on **110 / 255 (43%)** fixtures and is worse than coinflip in log loss. The 0.875 → 0.732 improvement over the prior v3 eval is real, but it's mostly eval-mix change (44 hardest WC 2024 matches → 261 mixed-tournament) plus more training data. **The model has never been closer to the market's own prices than it is now — and the market still beats it by 0.105 log loss units.**

**Liquidity splits (clean dedup, full model predictions):**

| Subset               |   n | Model LL | Market LL | Model flat ROI | Always-fav ROI | CI on model ROI    |
|----------------------|----:|---------:|----------:|---------------:|---------------:|--------------------|
| All matches          | 261 |   0.7319 |    0.6267 |         +0.06% |         +4.15% | [−19.0%, +23.2%]   |
| ≥ $10K volume        | 239 |   0.7338 |    0.6373 |         +4.20% |         +2.94% | [−16.5%, +29.2%]   |
| ≥ $50K volume        | 170 |   0.7664 |    0.6482 |         +0.36% |         +0.87% | [−19.5%, +22.7%]   |
| ≥ $100K volume       | 110 |   0.7151 |    0.6224 |         +7.29% |         +9.28% | [−19.3%, +38.8%]   |
| ≥ $500K volume       |  64 |   0.7437 |    0.6839 |        +21.67% |         −0.75% | [−18.5%, +70.2%]   |

**Key reads from the liquidity table:**
- Market beats model by ~0.10 log loss units **in every slice**. Volume doesn't unlock a hidden edge — it just shrinks the sample.
- Model > always-bet-favorite baseline only at the ≥$500K bucket (n=64, CI [−18.5%, +70.2%] spans zero so not significant).
- Filtering to ≥$50K makes model log loss *worse* (0.77), suggesting positive flat ROI on thin markets is mostly the model betting into dumb counterparties rather than finding real edges. This is the reverse of what we'd want in a skill story.
- The favorite baseline is the honest benchmark: **+4.15% ROI / 64% WR on all 261 matches**. Any future modeling change has to beat that, not just beat our own prior number.

**Calibration (all 261):** well-behaved in the middle band (predicted 45–65% tracks actual within ±1 pp), severely **over-confident at the high end** (predicted 74% → actual 51% on n=51; predicted 84% → actual 52% on n=21), and **under-confident at the low end** (predicted 16% → actual 40% on n=20). Isotonic regression or Platt scaling on the match-level output is the single most actionable next lever.

**Prior vs new (context, not a skill claim):**

| Metric              | Prior v3 (WC 2024, 44) | Clean-dedup Polymarket (261) |
|---------------------|-----------------------:|-----------------------------:|
| matches             |                     44 |                          261 |
| avg log loss        |                  0.875 |                       0.7319 |
| avg Brier           |                  0.317 |                       0.2649 |
| flat ROI            |                 −43.9% |                       +0.06% |
| win rate            |                  26.8% |                        42.1% |

Delta is driven by eval-mix change and larger training corpus, not by the model getting closer to the market.

### How to reproduce both baselines

```bash
# full 261
uv run python scripts/sim_eval/run_sim_eval.py \
    --test-dir data/polymarket_test \
    --odds betting_odds_polymarket.json \
    --n-sims 100

# high-liquidity subset — filter the odds file, point sim-eval at the filtered copy
uv run python - <<'PY'
import json
src = json.load(open('betting_odds_polymarket.json'))
kept = [m for m in src['matches'] if (m.get('polymarket_volume_usd') or 0) >= 50_000]
src['matches'] = kept; src['total_matches'] = len(kept)
json.dump(src, open('betting_odds_polymarket_50k.json', 'w'), indent=2)
PY
uv run python scripts/sim_eval/run_sim_eval.py \
    --test-dir data/polymarket_test \
    --odds betting_odds_polymarket_50k.json \
    --n-sims 100
```

The sim-eval script evaluates every match whose `match_id` appears in the odds file and whose JSON exists under `--test-dir`. `data/polymarket_test/` contains all 261 match JSONs; filtering happens purely via the odds file.

### Known caveats

- **96 / 261 matches (37%) emit "Incomplete team lineups"** during eval. `TestMatchLoader._extract_team_players` only sees players who actually batted/bowled; rain-shortened / large-margin matches leave some players unobserved. Loader pads with dummy `Player` objects, which carry fallback stats into the sim. Pre-existing behavior; not a regression from the rebuild. Worth investigating whether the full `info.players[team]` roster should be used for lineups even when innings coverage is partial.
- **1 residual winner disagreement** (France vs Norway 2026-04-07) remains after YES/NO dedup — single-entry fixture where Polymarket's upstream `winner` field is genuinely wrong; Cricsheet outcome is authoritative for `actual_winner`.
- **Afghanistan men's T20I gap**: ~24 Polymarket markets for Afghanistan bilaterals don't match any Cricsheet JSON. Coverage gap in our corpus, not a mapping bug.

## Rollback

The Polymarket artifacts are additive and ignored by legacy configs:

- Delete `betting_odds_polymarket.json`, `data/polymarket_test/`, `data/polymarket_build_unmatched.json` — no impact on legacy pipeline.
- `parsing_v2.py` split-constant + gender-filter change is the one destructive step. `git revert` and re-run `uv run python scripts/parsing_v2.py` to restore old splits. Cricsheet corpus + `all_players_enriched.csv` are append-only and safe.
- `data/xgb_data_v3_old/`, `models/cache_chunks_v3_old/`, `models/xgb_v3_old/` hold the pre-rebuild copies (~9 GB). Kept until the new baseline is confirmed stable; delete to reclaim disk.
