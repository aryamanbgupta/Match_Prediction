# Evaluation Speed Profiling

**Date**: 2026-04-18
**Context**: `run_sim_eval.py` with `--n-sims 100` on 44 matches took 107–169 min during autoresearch on 2026-04-17 — far above the ~5–10 min target.

---

## 1. Profiling Method

A dedicated benchmark script `scripts/profile_eval.py` runs a small reproducible slice of the real eval path under `cProfile`.

```bash
uv run python scripts/profile_eval.py --matches 2 --n-sims 20
```

What it measures:

| Phase | How |
|---|---|
| Fixed-cost setup | Wraps `StatsProvider`, `PlayerMetadataProvider`, `XGBoostModelV2`, `TestMatchLoader` loads in a `time_block` context manager. |
| Per-ball cost | 200 calls each to `model.extract_features(state)` and `model.predict_next_ball(features)` on a fixed initial state. |
| Full match sim | One `engine.simulate_match(state)` wall-clock. |
| End-to-end eval | `MatchLevelEvaluator.evaluate_all` on the subset, wrapped in `cProfile`; dumps `eval_profile.prof` and prints top 30 by cumulative time + top 25 by `tottime`. |

### Extrapolation formula

Observed per-ball cost × 240 balls/sim × `n_sims` × 44 matches.
240 is a reasonable upper bound for 2 full innings (120 balls each) before wickets.

### Inspecting the profile later

```bash
uv run python -m pstats eval_profile.prof
# then inside pstats:
# sort cumulative
# stats 30
```

---

## 2. Baseline Numbers (this machine, v3 XGBoost, sequential)

From a 2 × 20 run:

| Metric | Value |
|---|---|
| Wall time | **91.4 s** for 40 match-sims |
| Per match-sim | ~2.3 s |
| Per ball | **23.2 ms** (18.8 `extract_features` + 4.4 `predict_next_ball`) |
| Extrapolated 44 × 100 | **~81 min sequential** |

The extrapolation lines up with the observed 107–169 min — environmental factors (orphan processes, swap) explain the *spread* but not the baseline. **The code is genuinely this slow on this hardware.**

---

## 3. Where the Time Goes

### 3a. XGBoost `predict_proba` on a 1-row DataFrame — 57% of wall time

```
cumtime  function
 51.98s  xgboost/data.py:596(_transform_pandas_df)
 34.21s  xgboost/data.py:473(pandas_transform_data)
 12.30s  xgboost/data.py:414(is_pd_cat_dtype)
 11.46s  xgboost/data.py:430(is_pd_sparse_dtype)
```

XGBoost re-introspects the dtype of every column on every call:

- 125M `abc.__instancecheck__` calls
- 1.18M `is_pd_cat_dtype` / `is_pd_sparse_dtype` calls (≈216 per ball × 5,481 balls)

We rebuild a fresh 1-row `pd.DataFrame` per ball (`sim_v1_2.py:753`) and hand it to `predict_proba`, so this introspection happens on every ball.

### 3b. sklearn `LabelEncoder.transform` — 37% of wall time

```
cumtime  function                               ncalls
 33.79s  sklearn/preprocessing/_label.py:114(transform)   10,962
 30.44s  sklearn/utils/_encode.py:157(__init__)           10,962
 26.49s  sklearn/utils/_missing.py:9(is_scalar_nan)   62,735,526
```

`sim_v1_2.py:533` and `:538` call:

```python
self.batter_encoder.transform([str(striker.player_id)])[0]
self.bowler_encoder.transform([str(bowler.player_id)])[0]
```

Each call rebuilds an internal hash-map and runs nan-checks, despite only ~22 unique player IDs per match.

### 3c. Everything else — ~6%

Feature dict construction, `StatsProvider` lookups (LRU cache is working well), `PlayerMetadataProvider` — none are hot.

---

## 4. Recommended Fixes

Both fixes target `scripts/sim_v1_2.py` inside `XGBoostModelV2` and should be behind small unit tests to prove outputs match the current path.

| Fix | Status | Measured gain |
|---|---|---|
| A — Cache encoder lookups | **Pending** (next lever) | Projected ~30% |
| B — Skip DataFrame round-trip | **Implemented 2026-04-18** | 2.6× on full 44×100 eval (107 min → 41.7 min); bit-exact |

### Suggestion A — Cache encoder lookups per model instance

**Expected gain**: ~30% wall time (removes §3b).

**Where**: `XGBoostModelV2.__init__` + `extract_features` (sim_v1_2.py:469–540).

**Sketch**:

```python
# in __init__ after loading encoders:
self._batter_id_to_code = {
    cls: int(i) for i, cls in enumerate(self.batter_encoder.classes_)
}
self._bowler_id_to_code = {
    cls: int(i) for i, cls in enumerate(self.bowler_encoder.classes_)
}

# in extract_features:
features['batter_encoded'] = self._batter_id_to_code.get(str(striker.player_id), -1)
features['bowler_encoded'] = self._bowler_id_to_code.get(str(bowler.player_id), -1)
```

Matchup encoder (sim_v1_2.py:707) has the same pattern and should be cached the same way.

**Risk**: Encoders persist `classes_` as a sorted numpy array of strings; the dict must be built from `str(c)` keys to match the current `str(player_id)` input. Verify behavior for unseen IDs (current code catches `ValueError` and assigns `-1`).

### Suggestion B — Skip the DataFrame round-trip into XGBoost *(implemented 2026-04-18)*

**Expected gain**: ~45–50% wall time (removes most of §3a).
**Actual gain** (44×100 sequential): 107.7 min → **41.7 min** (2.6×). Output `simulated_prob` is bit-identical to the pre-fix baseline with seed=42 (verified on 5 matches, max abs diff = 0.00e+00).

**Where**: end of `extract_features` (sim_v1_2.py:748–753) and `predict_next_ball` (sim_v1_2.py:841–843).

Two options were considered:

1. **NumPy path** *(chosen)*: `extract_features` returns a preallocated `np.float64` 1-D array in `self.feature_columns` order; `predict_next_ball` calls `self.model.predict_proba(arr.reshape(1, -1))`. Avoids all pandas dtype inspection. `float64` (not `float32`) to match the training-time dtype from parquet exactly and keep probs bit-identical.
2. **DMatrix path**: `self.model.get_booster().inplace_predict(arr)`. Slightly faster but skips the sklearn wrapper entirely — not needed to hit the target.

**Sketch (option 1 — shipped)**:

```python
# in __init__:
self._feat_buf = np.zeros(len(self.feature_columns), dtype=np.float64)
if hasattr(self.model, 'n_features_in_'):
    assert self.model.n_features_in_ == len(self.feature_columns), (
        f"Feature count mismatch: model expects {self.model.n_features_in_}, "
        f"feature_columns file has {len(self.feature_columns)}"
    )

# in extract_features, replace the final DataFrame construction with:
buf = self._feat_buf
buf.fill(0.0)
for i, col in enumerate(self.feature_columns):
    val = features.get(col)
    if val is not None:
        buf[i] = val
return buf  # 1-D np.float64, reused across calls

# in predict_next_ball:
probs = self.model.predict_proba(features.reshape(1, -1))[0]
```

**Risks addressed during rollout**:
- `"X does not have valid feature names"` warning — already filtered in `run_sim_eval.py:15`, so no noise.
- Column-order drift — guarded by the `n_features_in_` assert at init; `feature_columns_v3.txt` is the single source of truth, written at train time and read at inference.
- Float32 vs float64 — chose float64 to match the implicit parquet/pandas training dtype. `predict_proba(df)` and `predict_proba(np_row)` came out bit-identical across 100 real feature rows (max abs diff 0.00e+00).
- Thread safety — `_feat_buf` is reused, so returned arrays must not be retained across calls. Safe under multiprocessing (each worker has its own model); documented in the `extract_features` docstring.

### Combined expected result

~5–8× speedup → 44 × 100 sequential in **~10–15 min**, matching the original target. Parallel mode (`--parallel`) would further multiply by core count, but only after these fixes — parallelism over slow inner loops is what caused the OOM before.

**Actual after Fix B alone**: 2.6× (107 min → 41.7 min). Remaining gap to the 10–15 min target is now dominated by §3b — Fix A is the next lever.

---

## 5. Out of Scope (But Worth Noting)

- **Batching predictions across sims**: would be the fastest option but requires decoupling simulation state from prediction — a larger refactor. Not needed to hit the 10–15 min target.
- **LRU cache tuning**: `StatsProvider` LRU is already fine per the profile (no chunk thrashing visible).
- **Parallel eval orchestration**: the 2026-04-17 OOM came from `--parallel` with the current heavy inner loop; once A+B land, sequential should be fast enough that parallel isn't needed for most runs.

---

## 6. Post-Fix-B Measurements (2026-04-18)

### Profile benchmark (2 × 20)

| Metric | Pre-fix | Post-Fix B |
|---|---|---|
| Wall time | 91.4 s | **35.8 s** |
| Per ball (extract + predict) | 23.2 ms | **~6.5 ms** |
| `predict_next_ball` cumulative | 52 s | 1.56 s |
| `_transform_pandas_df` | 52 s (#1 hot spot) | **gone from top-30** |
| `extract_features` cumulative | 39 s | 34.1 s (now dominated by sklearn `LabelEncoder.transform` — Fix A territory) |

### Full eval (44 matches × 100 sims, sequential, seed=42)

| Metric | Pre-fix baseline (`..._20260417_193516.json`) | Post-fix (`..._20260418_211339.json`) |
|---|---|---|
| Wall time | 107.7 min | **41.7 min** |
| avg_log_loss | 0.71988 | 0.71988 |
| avg_brier_score | 0.26167 | 0.26167 |
| flat P&L / ROI | -20.54 / -50.1% | -20.54 / -50.1% |

Metrics are identical to the baseline — the optimization didn't perturb the Monte Carlo.

### Parity verification

`scripts/validate_numpy_predict.py` drives real `MatchState` snapshots through `XGBoostModelV2.extract_features`, builds both the DataFrame and numpy inputs from the same feature dict, and asserts `predict_proba` outputs match to `atol=1e-12`. It also covers four edge cases (unseen player IDs → -1, all-zero row, extreme values, buffer reuse) and a 1,000-call micro-benchmark (requires ≥4× speedup). Run it any time XGBoost is upgraded:

```bash
uv run python scripts/validate_numpy_predict.py
```

---

## 7. Reproducing

```bash
# Quick smoke (under 2 min)
uv run python scripts/profile_eval.py --matches 2 --n-sims 20

# Larger slice (~5 min)
uv run python scripts/profile_eval.py --matches 5 --n-sims 50

# After implementing a fix, compare:
uv run python -c "import pstats; pstats.Stats('eval_profile.prof').sort_stats('cumulative').print_stats(15)"
```

The benchmark prints a per-ball `ms/ball` number — use that as the primary regression signal; target is **≤ 5 ms/ball** post-fix.
