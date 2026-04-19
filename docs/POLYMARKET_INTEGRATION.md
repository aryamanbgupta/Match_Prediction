# Polymarket Odds Ingestion & Test-Set Expansion

**Created**: April 2026
**Status**: Phases 1-3 complete. Phase 4 (destructive parse rebuild) and Phase 5 (retrain + sim-eval) pending user trigger.

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

## Pending phases

### Phase 4 — destructive rebuild (user-triggered)

```
uv run python scripts/parsing_v2.py
```

Regenerates `data/xgb_data_v3/` + `models/cache_chunks_v3/` (~10-15 min). Only run after the dry-run looks good.

### Phase 5 — retrain + sim-eval

```
uv run python scripts/run_experiment.py \
    experiments/configs/xgb_v3_baseline.yaml --skip-parsing

uv run python scripts/sim_eval/run_sim_eval.py \
    --test-dir data/polymarket_test \
    --odds betting_odds_polymarket.json \
    --n-sims 100
```

`BettingOddsLoader` should be called with `remove_margin=False` — Polymarket is margin-free by construction (prices sum to ~1.0 from the two binary YES tokens, not bookmaker-padded).

## Rollback

The Polymarket artifacts are additive and ignored by legacy configs:

- Delete `betting_odds_polymarket.json`, `data/polymarket_test/`, `data/polymarket_build_unmatched.json` — no impact on legacy pipeline.
- `parsing_v2.py` split-constant + gender-filter change is the one destructive step. `git revert` and re-run `uv run python scripts/parsing_v2.py` to restore old splits. Cricsheet corpus + `all_players_enriched.csv` are append-only and safe.
