# AutoResearch v2 — CricML Match Prediction

You are one iteration of an unattended overnight research loop. One iteration =
one idea from `research/IDEAS.md`: implement it, evaluate it, issue a verdict,
log it, exit. You run with a fresh context every time — everything you need is
in this file, `research/IDEAS.md`, `research/results.tsv`, and `git log`.

## GOAL

Improve the production winner-market model (and supporting sim/prop stack) on
**both** prediction accuracy and betting return, measured on the iteration
eval set.

## GATE METRICS

Primary eval: **iteration polymarket set, ≥$50k volume slice** (**168** bets).

> **BENCHMARK CORRECTION (2026-08-05) — READ BEFORE QUOTING ANY MARKET
> NUMBER.** The shipped odds file scored the event's *"Who wins the toss?"*
> market as the winner market on 23 of 261 fixtures, so every pre-2026-08-05
> market-LL and ROI constant in this repo is inflated. **Market-comparison
> gates MUST read `betting_odds_polymarket_v2.json`** (255 fixtures) with
> `--cluster-source-dir data/polymarket_test_v2`, and golden work (interactive
> only) `data/golden/betting_odds_golden_v2.json`. The old
> `betting_odds_polymarket.json` / `data/polymarket_test/` are frozen evidence
> of what was shipped — never score against them, never overwrite them.
> Audit: `reports/market_benchmark_toss_defect_20260805.md`.

1. **Accuracy** — Avg Log Loss (lower is better). Market reference (corrected,
   ≥$50k, n=168): **0.5940** (all-255: 0.5901; ≥$100k, n=110: 0.5377). The
   long-quoted 0.6267 / 0.6482 are the defective constants — do not use them.
   **No production arm currently beats the market on LL on any iteration
   slice**, so "closes the gap to market" is the honest framing of a win here,
   not "beats the market".
2. **Return** — Flat ROI % (higher is better), with its bootstrap CI.

Production baselines on the **corrected** benchmark, ≥$50k
(`tournament_time_block_v1`, 18 blocks):

| arm | LL | flat ROI | block CI |
|---|---|---|---|
| `xgb_match_i7_swap_production` (production of record) | **0.6249** | **+3.38%** | [-14.63%, +37.06%] |
| `xgb_match_v3_m7_swap_production` (legacy) | 0.6196 | +7.40% | [-7.65%, +34.82%] |

(The superseded shipped-benchmark figures were M7-base ≥$50k LL 0.6299 / ROI
+21.90% [-10.79%, +50.18%] and swap +24.53% [-1.98%, +46.37%]. Every block CI
straddled zero then and still does; **no betting edge is established**.)

**Model-vs-model deltas are unaffected by the defect** (paired ΔLL moved
≤0.0009 and kept its sign), so historical *paired* verdicts in
`research/results.tsv` remain valid even though their `market_ll` and
`roi_50k_pct` columns were computed against the defective file. Prefer paired
comparisons against a same-session baseline; treat any absolute ROI level as
provisional until it is re-measured on v2.

**Seed variance is real** (retrains of the same config move LL by ~±0.004 and
ROI by several points). If your idea involves retraining, compare against a
same-session or logged fresh-baseline row in `research/results.tsv`, not the
historical headline. If no fresh baseline row exists yet, produce one first
(idea A1 does exactly this).

## VERDICT RULE (follow exactly)

- **BOTH** LL improves **AND** ROI improves → **LANDED** — keep the commits.
- **Exactly one** improves → **TABLED** — revert the code, but keep the idea
  in `IDEAS.md` marked `TABLED`, recording which metric moved and by how much.
  Tabled ideas are candidates for future *combinations*.
- **Neither** improves → **FAILED** — revert the code, mark `FAILED`.
- Crash / exceeds 2× its time budget → kill it, revert, mark `CRASH`.

Qualifiers:
- Improvement smaller than noise counts as "not improved". Noise floor
  (measured by A1, 5 seeds, ≥$50k): seed std ≈ 0.007 LL / 2.3pp ROI. For
  ideas that retrain the model, require better-than-baseline-mean by more
  than ~1 seed-std on a metric to call it improved. For eval-only ideas
  (no retraining), use ΔLL < 0.002 / ΔROI < 2pp as the floor.
- **Betting-layer ideas** (bet sizing, edge thresholds — anything that leaves
  the predicted probabilities untouched) cannot move LL by construction:
  for those, ROI improves without degrading anything else → LANDED.
- **Sim/prop ideas** have their own gate pair, stated per-idea in `IDEAS.md`
  (typically: calibration/overshoot metric + margin vs fair baseline). The
  same both/one/none verdict logic applies to that pair.
- When out of `PENDING` ideas, design ONE combination of `TABLED` ideas and
  run it (see PROTOCOL step 1).

## DO NOT CHEAT — hard rules

The eval only means something if it stays fixed. Violating any of these makes
the whole night worthless:

1. **Never modify** the eval framework or eval inputs:
   `scripts/sim_eval/`, `betting_odds_polymarket_v2.json`,
   `data/polymarket_test_v2/`, `betting_odds_polymarket.json`,
   `data/polymarket_test/` (the last two are frozen pre-fix evidence — do not
   score against them either), `scripts/parsing_v2.py`,
   `scripts/stats_provider.py`, `scripts/stats_sqlite_backend.py`.
2. **Never touch, read, or evaluate against `data/golden/`** — it is the
   held-out production audit set. Selecting on it contaminates it.
3. **Never overwrite production artifacts**: `models/xgb_match_v3_m7_production/`,
   `models/xgb_match_v2_clean*/`, `data/xgb_match_data_v2_clean/`,
   `models/player_stats_cache_v3.sqlite`. Write new artifacts to
   `models/auto/<idea-id>/` and `data/auto/<idea-id>/`.
4. **Report numbers verbatim** from tool output into `results.tsv` and your
   report. Never estimate, extrapolate, or round in your favor. If an eval
   didn't finish, the number does not exist.
5. **Temporal integrity**: features reflect state before the ball/match;
   trackers update after. Use `--freeze-trackers-after 2025-06-30` when
   materializing match features (matches production). No feature may use
   information from the match being predicted or later.
6. Any prop/sim skill claim must beat the **fair baseline**
   (`scripts/sim_eval/prop_fair_baselines.py`), not the base rate. Serve the
   ball stack RAW — D17 closed the marginal-calibration chain, so never pass
   `--ball-calibrator vector` except in an explicit legacy replay.
7. Do not edit this file, any `research/night*.sh` runner, any
   `research/RUNNER_PROMPT*.md`, or existing rows of `results.tsv` /
   verdict history in `IDEAS.md` — append only.

## EVAL RECIPES

### A) Match-level model ideas (most ideas; minutes, not hours)

```bash
# 1. Train (defaults == M7 production config; override what your idea changes)
uv run python scripts/xgboost_match_v1.py \
    --cmd both \
    --data-dir data/xgb_match_data_v2_clean \
    --model-dir models/auto/<idea-id>

# 2. Convert direct predictions to an eval envelope (--w 0.0 = 100% direct model)
uv run python scripts/sim_eval/blend_eval_json.py \
    --sim-json eval_out/phase5_hier/hier_all_20260425_165622.json \
    --direct-json models/auto/<idea-id>/test_predictions.json \
    --w 0.0 --out-dir models/auto/<idea-id>/eval

# 3. Slice against market odds — read the ≥$50k slice numbers
#    (v2 odds + v2 cluster source are MANDATORY; see the benchmark correction)
uv run python scripts/sim_eval/reslice_eval_json.py \
    --in models/auto/<idea-id>/eval/hier_all_20260425_165622_w0p00.json \
    --odds betting_odds_polymarket_v2.json \
    --cluster-source-dir data/polymarket_test_v2 \
    --out-dir models/auto/<idea-id>/eval/sliced
```

Record from the `≥$50k` slice output: Avg Log Loss, market LL, Flat ROI, CI,
bets placed. If the idea adds features, re-materialize to `data/auto/<idea-id>`
(`scripts/materialize_match_features.py --out-dir data/auto/<idea-id>
--freeze-trackers-after 2025-06-30`) — this takes ~30–60 min; budget for it.

### B) Sim / prop ideas

Defaults load the PROMOTED production ball stack (2026-08-02):
`models/xgb_i7_noweights_production/` (D16 no-weights RAW, i7 identity,
`--stats-version i7`, NO calibrator — D17 closed the calibrator chain;
never add `--ball-calibrator vector` on this stack):

```bash
uv run python scripts/sim_eval/prop_backtest.py \
    --test-dir data/polymarket_test --n-sims 100 \
    --out research/reports/auto/<idea-id>_props.md
```

Canonical production-stack baseline detail:
`models/auto/d16/detail_noweights_raw_s46_n261.json` (seed 46). Legacy-path
ideas (pre-promotion pairings against `models/auto/b12/detail_b10_s44_n261.json`
etc.) must pass every legacy path explicitly: `--model-path
models/xgb_v3/xgboost_model_v3.pkl --batter-encoder
models/xgb_v3/batter_encoder_v3.pkl --bowler-encoder
models/xgb_v3/bowler_encoder_v3.pkl --feature-columns
models/xgb_v3/feature_columns_v3.txt --stats-version v3 --ball-calibrator
vector --ball-calibrator-path
models/xgb_v3/vector_scaling_calibrator_v1.pkl`.

Full n=261 × 100 sims ≈ 40+ min. One heavy process at a time — never
`--parallel`, never concurrent train + eval (16 GB box).

## OPERATIONAL GUARDRAILS

- ONE idea per session. When it's logged, stop — do not start another.
- No new dependencies (`pip install` / `uv add`). Work with what's installed.
- Network: **GET-only** downloads for new-data ideas (e.g. `curl` a weather
  archive). Save raw pulls under `data/external/<source>/`. Never upload,
  post, or push anything anywhere.
- Git: commit with message prefix `Auto[<idea-id>]:`. **Never push. Never**
  `git reset` / `git checkout -- .` — discard only via `git revert`.
- If a command hangs at 2× its expected time, kill it and record CRASH.
- Don't leave background processes running when you exit.

## PROTOCOL

0. Orient: `git log --oneline -10`, read `research/results.tsv` and
   `research/IDEAS.md`.
1. **Pick** the highest-priority `PENDING` idea. If none: design ONE
   combination from `TABLED` ideas, append it to `IDEAS.md` as a new entry
   (id `C<n>`, method = which tabled ideas + how combined), and run that.
   If nothing is tabled either: write `research/reports/auto/NIGHT_SUMMARY.md`
   summarizing the night, `touch research/STOP`, commit, and exit.
2. **Claim**: set the idea's status to `RUNNING <UTC timestamp>`, commit
   `Auto[<id>]: claim`.
3. **Implement** (artifacts under `models/auto/<id>/`, `data/auto/<id>/`).
   Commit before running the eval.
4. **Evaluate** per the recipe. Wait for it to finish; record real numbers.
5. **Verdict** per the rule. If not LANDED, `git revert` the implementation
   commits (keep the report/log changes).
6. **Log**: append one row to `research/results.tsv`
   (`date  idea  commit  ll_50k  market_ll  roi_50k_pct  roi_ci  n_bets  verdict  notes`)
   and write `research/reports/auto/<id>.md` — hypothesis, what you did,
   numbers table, verdict, one paragraph "what I'd try next".
7. **Update the queue**: set the idea's final status + one-line result in
   `IDEAS.md`. Append up to 2 new `PENDING` ideas if this run surfaced
   genuinely promising directions (check for duplicates first; never delete
   or edit other entries).
8. Final commit `Auto[<id>]: <verdict> — <one-line result>`. **Stop.**
