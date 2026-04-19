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

### Dedup: Polymarket YES/NO binary markets

Polymarket runs each cricket fixture as two binary YES/NO markets ("Will Team1 win?" / "Will Team2 win?"). The upstream extractor (`polymarket-cricket/extract_prematch_odds.py`) emits one record per binary market, each with its own `winner` label and its own `prematch_price_team*` orientation. In the raw export, **201 (date, team-set) groups had contradictory winner labels** across their duplicate entries.

Dedup tiebreak, in order:
1. Polymarket `winner` (normalized) matches the authoritative Cricsheet `info.outcome.winner`.
2. Highest `volume_usd`.

This ensures the prematch prices we keep come from the binary market whose YES side actually won. Without step 1, 62 of 261 eval entries were orientation-flipped (picking a match with implied probability 0.73 for the team that lost because we kept the wrong sibling).

Post-fix: 63 raw disagreements → **1 residual disagreement** (France vs Norway 2026-04-07, single-entry fixture where Polymarket's upstream data is genuinely wrong). Cricsheet remains authoritative for `actual_winner` in the eval file regardless.

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

| Metric              | Prior v3 (WC 2024, 44 matches) | New v3 (Polymarket, 261 matches) | Δ                |
|---------------------|-------------------------------:|---------------------------------:|-----------------:|
| matches             |                             44 |                              261 | 6×               |
| avg log loss        |                          0.875 |                            0.732 | −16.4%           |
| avg Brier           |                          0.317 |                            0.265 | −16.4%           |
| avg edge            |                         +33.4% |                           +19.2% | closer to market |
| avg signed edge     |                         −22.2% |                            −6.8% | closer to 0      |
| flat ROI            |                         −43.9% |                            −5.4% | +38.5 pp         |
| full Kelly ROI      |                         −22.7% |                            +0.3% | +23 pp, breakeven|
| win rate            |                          26.8% |                            39.2% | +12.4 pp         |

**Segment breakdown:**
- Favorites (odds < 2.0, 82 bets): WR 59.0%, flat ROI **+3.9%**, Kelly −0.54 u, edge 10.9%.
- Underdogs (odds ≥ 2.0, 173 bets of 178): WR 28.7%, flat ROI −9.9%, Kelly **+1.25 u**, edge 23.1%.

**Calibration:** model is well-calibrated in the middle band (predicted 45–65% tracks actual within ±1 pp) but over-confident at the high end (predicted 74.1% → actual 51.0%; predicted 83.5% → actual 52.4%). Underconfident at the low end (predicted 15.8% → actual 40.0%). Isotonic/Platt on match-level output is an obvious next lever.

### Known caveats

- **96 / 261 matches (37%) emit "Incomplete team lineups"** during eval. `TestMatchLoader._extract_team_players` only sees players who actually batted/bowled; rain-shortened / large-margin matches leave some players unobserved. Loader pads with dummy `Player` objects, which carry fallback stats into the sim. Pre-existing behavior; not a regression from the rebuild. Worth investigating whether the full `info.players[team]` roster should be used for lineups even when innings coverage is partial.
- **1 residual winner disagreement** (France vs Norway 2026-04-07) remains after YES/NO dedup — single-entry fixture where Polymarket's upstream `winner` field is genuinely wrong; Cricsheet outcome is authoritative for `actual_winner`.
- **Afghanistan men's T20I gap**: ~24 Polymarket markets for Afghanistan bilaterals don't match any Cricsheet JSON. Coverage gap in our corpus, not a mapping bug.

## Rollback

The Polymarket artifacts are additive and ignored by legacy configs:

- Delete `betting_odds_polymarket.json`, `data/polymarket_test/`, `data/polymarket_build_unmatched.json` — no impact on legacy pipeline.
- `parsing_v2.py` split-constant + gender-filter change is the one destructive step. `git revert` and re-run `uv run python scripts/parsing_v2.py` to restore old splits. Cricsheet corpus + `all_players_enriched.csv` are append-only and safe.
- `data/xgb_data_v3_old/`, `models/cache_chunks_v3_old/`, `models/xgb_v3_old/` hold the pre-rebuild copies (~9 GB). Kept until the new baseline is confirmed stable; delete to reclaim disk.
