# External Analysis — `mshashi11/CricketAI` T10 Win-Probability Models

> Reference note. Analysis of a third-party repo for comparison against CricML.
> Not part of the CricML pipeline. Source read from GitHub on 2026-06-08
> (repo last pushed 2026-06-04). Nothing here is load-bearing for our code.

- **Repo**: <https://github.com/mshashi11/CricketAI>
- **Description**: "Repository for exploring the use of AI in Cricket Analytics"
- **License / lang**: MIT, Python. Created 2026-03-20, 1 star, single author.
- **What it models**: live **Win Probability (WP)** + projected innings score
  for the **T10** format (10 overs = 60 balls). A WASP/DLS-style state model.

## Actual repository contents

The git tree (authoritative, via the GitHub API) contains only:

```
README.md, LICENSE, .gitignore
WinProbability/
├── instruction.md         # design spec / TODO
├── wp_model_xgb.py        # XGBoost implementation (4 models + volatility bounds)
├── wp_model_nn.py         # PyTorch NN implementation (same idea, no monotonicity)
├── probs_i1_t10.txt       # precomputed 1st-innings lookup table (output artifact)
└── probs_i2_t10.txt       # precomputed 2nd-innings lookup table (output artifact)
```

> ⚠️ The README prose references a `DataPrep/` folder and a "Cricmetric
> integration script." **Neither exists in the git tree** — stale/aspirational.
> Training data lives as two CSVs attached to GitHub Release `v1.0.0`
> (`t10_inn1_data.csv`, `t10_inn2_data.csv`), auto-downloaded at runtime by
> `ensure_data_exists()`.

## Features (the core)

A **pure game-state model — exactly 3 features per innings.** No team strength,
player identity, venue, toss, or form. Each training row is one ball-state from a
completed match, labeled with that match's eventual outcome.

| | 1st-innings model inputs | 2nd-innings model inputs |
|---|---|---|
| Feature 1 | `cum_balls` — balls bowled so far | `rem_balls` — balls remaining |
| Feature 2 | `cum_wickets` — wickets fallen | `wickets_hand` — wickets in hand |
| Feature 3 | `cum_runs` — runs scored so far | `runs_chase` — runs still needed to win |
| Labels | `won` (binary); `total_score` (final score) | `won` (binary) |

## How it works — XGBoost version (`wp_model_xgb.py`)

**Four models**, all `n_estimators=1000, max_depth=5, lr=0.05`:

| Model | Type | Predicts | Monotone constraints `(balls, wkts, runs)` |
|---|---|---|---|
| `wp_model_inn1` | `XGBClassifier` | P(team batting first wins) | `(-1, -1, +1)` |
| `score_model_mean` | `XGBRegressor` | projected final 1st-inns score | `(-1, -1, +1)` |
| `score_model_vol` | `XGBRegressor` | \|residual\| of mean model (spread) | `(-1, -1, 0)` |
| `wp_model_inn2` | `XGBClassifier` | P(chasing team wins) | `(+1, +1, -1)` |

The **monotone constraints** bake cricket domain logic directly into the trees:
WP falls as balls/wickets are used, rises with runs; the chase WP rises with balls
left and wickets in hand, falls with runs still required.

**Volatility bounds.** `score_model_vol` is trained on the *absolute residuals* of
the mean-score model, giving a state-dependent spread. Bounds are
`mean ± k·vol` with `k = 1.15` (a normal-approx ~75% interval). Cheap stand-in
for quantile regression. Uncertainty shrinks as balls/wickets run out
(constraint `(-1,-1,0)`).

**Innings-break continuity (the clever bit).** `instruction.md` states the
invariant: *"WP at the end of the 1st innings for the team batting first must
match its WP at the start of the 2nd innings."* Enforced by an override when
building the table: at `balls == 60`, instead of trusting `wp_model_inn1`, it
queries `wp_model_inn2` with the chaser needing `runs + 1` off 60 balls with 10
wickets, then sets `WP_inn1 = 1 − P(chaser wins)`. Score projection is likewise
clamped to actual runs at the over-cap (`balls==60`) and at all-out
(`wickets==10`).

**Output = precomputed state tables.** Both scripts brute-force the full grid
(`balls 0–60 × wickets 0–10 × runs 0–200` ≈ 135k rows) and dump TSV lookup
tables — inference is a table lookup, no model load. The inn1 table carries
`balls, wickets, runs, WP, mean_score, score_low, score_high`; inn2 carries
`balls, wickets, runs_chase, WP`.

## Neural-net version (`wp_model_nn.py`)

Same 3 features, same data, same table-generation + break-override logic, but a
thin MLP: `Linear(3→20) → ReLU → Linear(20→1)`, trained full-batch with Adam
(lr 0.01, 2000 iters), `BCEWithLogitsLoss` for WP, `MSELoss` for score.
Differences vs the XGB version:

- **No monotonicity** (hard to enforce in a plain NN) → WP surface can be
  non-monotone/wiggly. A real regression vs XGB.
- **No volatility model** — WP + mean score only, no bounds.
- Prints an explicit innings-break consistency check (WP1 vs 1−WP2) before
  generating tables.
- No input normalization (balls≤60, runs≤200 fed raw into a tiny net).

## Assessment (relative to CricML)

- **Context-free, league-average WP.** With no team/player/venue signal it can't
  tell a strong chaser from a weak one — fine for a broadcast win-bar, not for a
  betting edge. Contrast CricML: 49 match-level / 114 ball-level features, direct
  match-level XGBoost as the winner-market model.
- **No evaluation anywhere in the code.** No train/test split, no match-level
  clustering (ball-states within a match are highly correlated → inflated
  effective *n*), no log-loss / Brier / calibration reported. It trains on
  everything and dumps a table.
- **Sound bits worth noting.** The monotone-constraint XGB design, the inn1↔inn2
  stitching at the break, and the predict-\|residual\| volatility trick are all
  clean, pragmatic choices.

**Net:** a minimal, single-author reference implementation of a WASP/DLS-style T10
win-probability *surface* — good as a template for the monotone state-table
approach, but with none of the strength-adjustment or eval rigor needed for market
work. Closest CricML analogue is the in-play win-prob model
(`models/inplay_winprob_v1`), which is also a direct P(win | ball-state) model but
adds team rating + chase math and is evaluated OOS (LL 0.5418 / AUC 0.80).
