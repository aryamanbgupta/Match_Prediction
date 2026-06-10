# MLC — Major League Cricket (USA) workspace

Everything for the MLC work lives here. Two threads:

1. **Backtest** of the production pipeline on **MLC 2025** (the most recent completed
   season) — does the model actually work on this league?
2. **The pitch** — `EDGE_BRIEF.md`: the franchise-facing edge we can take to an MLC
   team, grounded in the backtest numbers.

> Note on the 2026 picks: the MLC 2026 Dallas-leg win probabilities
> (`../fixtures/mlc_2026/`, `../reports/mlc_2026_dallas_leg.md`) are for **unplayed**
> June-2026 fixtures, so they can't be scored yet. MLC 2025 is the scoreable proxy
> for the same pipeline.

## Layout

```
mlc/
├── EDGE_BRIEF.md                  # the pitch: "decision deltas, not predictions"
├── data/
│   ├── mlc_2025/                  # 33 staged cricsheet matches + _manifest.json
│   └── mlc_2025_predictions.json  # match-level winner predictions
├── reports/
│   ├── mlc_2025_eval.md           # consolidated backtest writeup (start here)
│   ├── mlc_2025_prop_report.md    # ball-sim prop calibration (player performances)
│   ├── mlc_2025_prop_per_match/   # per-match top-scorer / top-wkt drilldowns
│   ├── mlc_2025_matchups.md       # batter-vs-bowler replay (matchup ranking)
│   ├── mlc_decision_demo.md       # worked counterfactual (the edge, demonstrated)
│   └── *_detail.json              # machine-readable details
└── scripts/
    ├── extract_mlc.py             # stage MLC 2025 cricsheet → data/mlc_2025/
    ├── eval_mlc_match_level.py    # winner-market accuracy (prod model)
    ├── eval_mlc_matchups.py       # teacher-forced ball replay, matchup ranking
    └── sim_decision_demo.py       # decision-delta counterfactual on one match
```

## Reproduce

All scripts run from the repo root via `uv run` and read shared assets
(`models/`, `data/xgb_data_v3`, the SQLite cache) in place — nothing here mutates
the production cache or training data.

```bash
# 1. stage MLC 2025 (idempotent)
uv run python mlc/scripts/extract_mlc.py

# 2. winner-market accuracy
uv run python mlc/scripts/eval_mlc_match_level.py

# 3. ball-level prop backtest (player performances) — shared prop engine, MLC test-dir
uv run python scripts/sim_eval/prop_backtest.py \
    --test-dir mlc/data/mlc_2025 --n-matches all --n-sims 100 \
    --detail-out mlc/reports/mlc_2025_prop_detail.json \
    --report-out mlc/reports/mlc_2025_prop_report.md
uv run python scripts/sim_eval/render_prop_per_match.py \
    --detail mlc/reports/mlc_2025_prop_detail.json \
    --out-dir mlc/reports/mlc_2025_prop_per_match/

# 4. batter-vs-bowler matchup replay
uv run python mlc/scripts/eval_mlc_matchups.py

# 5. decision-delta demo (the edge)
uv run python mlc/scripts/sim_decision_demo.py --match-date 2025-07-13 --n-sims 300
```

## Headline findings (see `reports/mlc_2025_eval.md` for the full, caveated version)

- **Caveat:** MLC was already in training; June 2025 is in-sample (validation), only
  **July (n=12) is held-out**. Underpowered — directional, not a verdict.
- **Winner market:** held-out July 8/12 = 67%, beats coinflip, low-resolution.
- **Player props:** real skill on **rankings** (top batter +0.069, #1 pick = actual
  top scorer 27% vs ~9% base); **over-states tails** (sixes, powerplay totals,
  specific bowler wickets) — fade those.
- **Matchups:** ranks batter quality (SR rank corr +0.52) and bowler economy (+0.22),
  but **no validated pairwise duel edge** (+0.11 ≈ chance). Runs ~2.3× hot on wickets.
- **The edge:** because the sim ranks relative outcomes correctly, it can score a
  team's *decisions* (order, deployment) as **deltas**, where the bias cancels. See
  `EDGE_BRIEF.md`.
