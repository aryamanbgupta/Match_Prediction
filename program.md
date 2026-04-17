# AutoResearch: CricML Match Prediction

## GOAL

Improve the XGBoost T20 cricket match prediction model by discovering better features, hyperparameters, or training strategies. Each iteration makes ONE focused change, evaluates it, and keeps or discards.

## METRICS

### Primary metric (used for keep/discard decisions)
**Avg Log Loss** on 44 T20 World Cup 2024 test matches (100 simulations each).
Lower is better.

### Secondary metrics (log but do NOT optimize directly)
- **Avg Brier Score** — confirms calibration isn't just compression toward 50%
- **Flat ROI** — sanity check on betting performance
- **Win Rate** — % of bets that would have won
- **Avg Edge** — average |model_prob - market_prob| across matches. Measures how differentiated predictions are from the market. If this shrinks while log loss improves, the model is becoming conservative, not better.

### Red flag rule
If log loss improves BUT avg edge decreases (predictions converge toward market odds / 50-50), **DISCARD** the change. The model is becoming less decisive, not better calibrated. This is the Platt scaling trap — better log loss through conservative predictions doesn't translate to profitable betting. Also discard if log loss improves but flat ROI gets significantly worse.

### Run evaluation
```bash
uv run python scripts/sim_eval/run_sim_eval.py \
    --test-dir data/betting_test \
    --odds betting_odds_v3.json \
    --n-sims 100
```

Record ALL metrics from the output.

## CURRENT BASELINE

- Model: XGBoost v4 (team strength features)
- Log Loss: ~0.710
- Features: 72 (see `scripts/feature_registry.py`)
- Training: `uv run python scripts/xgboost_v2.py`

## DIRECTION

Before each iteration:
1. Read `results.tsv` to see what's been tried and what patterns emerge (if the file doesn't exist yet, skip this step)
2. Check `git log --oneline -10` to see the line of evolution
3. Think about what single change is most likely to improve the metric

Types of changes to explore (in rough priority order):
- **Feature engineering**: Modify feature groups, ablate or recombine existing features
- **Hyperparameter tuning**: max_depth, learning_rate, min_child_weight, subsample, colsample_bytree
- **Class weights**: Rebalance dot/1/2/4/6/wicket weights
- **Feature selection**: Ablate low-importance features (check feature importance first)
- **Feature scaling/normalization**: StandardScaler, log transforms on skewed features
- **Regularization**: L1/L2 via reg_alpha/reg_lambda

## EDITABLE FILES

You may ONLY modify these files:
- `scripts/xgboost_v2.py` — model architecture, hyperparameters, training loop
- `scripts/feature_registry.py` — feature definitions, group composition

## FIXED FILES (DO NOT MODIFY)

- `scripts/parsing_v2.py` — data pipeline, temporal integrity (CRITICAL)
- `scripts/stats_provider.py` — temporal stats access (CRITICAL)
- `scripts/sim_v1_2.py` — simulation engine
- `scripts/sim_eval/` — evaluation framework (metrics must stay consistent)
- `data/betting_test/` — test data
- `betting_odds_v3.json` — market odds
- `pyproject.toml` / `requirements.txt` — no new dependencies

## PROTOCOL

For each iteration:

### 1. COMMIT BEFORE RUNNING
```bash
git add scripts/xgboost_v2.py scripts/feature_registry.py
git commit -m "Try: [one-line description of change]"
```

### 2. TRAIN THE MODEL
```bash
uv run python scripts/xgboost_v2.py
```
If training crashes or takes > 60 minutes, treat as CRASH.

### 3. RUN EVALUATION
```bash
uv run python scripts/sim_eval/run_sim_eval.py \
    --test-dir data/betting_test \
    --odds betting_odds_v3.json \
    --n-sims 100
```
Record the "Avg Log Loss" from the output.

### 4. DECIDE: KEEP or DISCARD
- If log loss **decreased** AND avg edge did NOT shrink: **KEEP** — the commit stays
- If log loss **increased or stayed same**, OR avg edge shrank significantly: **DISCARD** — revert:
  ```bash
  git revert --no-edit HEAD
  ```

### 5. LOG TO results.tsv
Append a tab-separated row. On the very first iteration, create the file with a header row first.

**Header:**
```
commit_hash	log_loss	brier	flat_roi	win_rate	avg_edge	train_time_sec	eval_time_sec	status	description
```

**Fields** (all from the eval console output):
- `commit_hash`: 7-char git hash (or "none" if discarded)
- `log_loss`: "Average Log Loss" from eval output
- `brier`: "Average Brier Score" from eval output
- `flat_roi`: "ROI" under "Flat Staking" (e.g., -44.8%)
- `win_rate`: "Win Rate" under "Flat Staking" (e.g., 24.4%)
- `avg_edge`: "Average Edge (magnitude)" from eval output — used for the red flag rule
- `train_time_sec`: approximate training duration
- `eval_time_sec`: approximate evaluation duration
- `status`: keep, discard, or crash
- `description`: what you changed (1-2 sentences)

### 6. REPEAT
Go back to step 1. Continue until you've completed the requested number of iterations.

## GUARDRAILS

- **ONE change per iteration** — no compound changes (can't tell what helped)
- **Never modify fixed files** — especially parsing_v2.py and stats_provider.py
- **No new dependencies** — work with existing packages only
- **No test set changes** — the 44 matches and odds file are sacred
- **Do NOT use --parallel on eval** — it spawns N processes each loading ~550MB stats cache, which OOM-kills this machine. The default sequential eval is fine (~5 min total).
- **Memory limit: 8GB** — do not launch concurrent heavy processes (no background training + eval, no parallel simulations, no multi-process anything). One heavy process at a time.
- **Run commands EXACTLY as written** — do not add flags, redirect output, or wrap in background processes unless the protocol says to
- **If you crash twice in a row** — step back, try a more conservative change
- **If 5 consecutive discards** — try a completely different direction
- **Log EVERY attempt** — including crashes and discards. results.tsv is the full history.

## ITERATIONS

Run the number of iterations requested by the user. If not specified, default to 20 rounds, then stop and summarize findings.

## SUMMARY

After completing all iterations, provide:
1. Best log loss achieved vs baseline
2. Brier score and ROI trends — did they move in the same direction as log loss?
3. Avg edge trend — did the model stay decisive or become conservative?
4. Which changes helped most (pattern analysis from results.tsv)
5. Suggested next directions based on what you learned
6. The cumulative diff: `git diff <first-commit>..HEAD`
