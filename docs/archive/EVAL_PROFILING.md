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
| A — Cache encoder lookups | **Implemented 2026-04-18** | 16× on 2×20 profile bench (35.8s → 2.25s); 5×100 eval slice runs in 23.7s (extrapolates to ~3.5 min for 44×100). Bit-exact. |
| B — Skip DataFrame round-trip | **Implemented 2026-04-18** | 2.6× on full 44×100 eval (107 min → 41.7 min); bit-exact |

### Suggestion A — Cache encoder lookups per model instance *(implemented 2026-04-18)*

**Expected gain**: ~30% wall time vs pre-B; expected to dominate post-B (was the residual ~80% of post-B time).
**Actual gain** (2×20 profile bench, post-B baseline): 35.8 s → **2.25 s** (~16×). 5×100 real eval slice ran in **23.7 s wall** — extrapolates to ~3.5 min for 44×100, well inside the 10–15 min target.

**Where**: `XGBoostModelV2.__init__` and `extract_features` in `scripts/sim_v1_2.py`.

**Sketch (shipped)**:

```python
# in __init__ after loading encoders:
self._batter_id_to_code = {
    str(c): int(i) for i, c in enumerate(self.batter_encoder.classes_)
}
self._bowler_id_to_code = {
    str(c): int(i) for i, c in enumerate(self.bowler_encoder.classes_)
}
self._matchup_to_code = (
    {str(c): int(i) for i, c in enumerate(self.matchup_encoder.classes_)}
    if self.matchup_encoder is not None else None
)

# in extract_features (replaces try/except: -1 path):
features['batter_encoded'] = self._batter_id_to_code.get(str(striker.player_id), -1)
features['bowler_encoded'] = self._bowler_id_to_code.get(str(bowler.player_id), -1)
if self._matchup_to_code is not None:
    features['matchup_type_encoded'] = self._matchup_to_code.get(matchup_type, -1)
else:
    features['matchup_type_encoded'] = 0
```

**Risks addressed during rollout**:
- `numpy.str_` vs plain `str` — `LabelEncoder.classes_` stores `numpy.str_`. Both ends use `str()` to coerce; verified bit-exact via the round-trip check in `scripts/tests/test_xgboost_model_v2_encoder_cache.py` against the real production encoders (6,590 batter + 4,856 bowler + 27 matchup classes).
- Unknown IDs — old path raised `ValueError` then assigned `-1`. New path uses `dict.get(key, -1)` for the same observable behavior. Tested on a sentinel ID across all three caches.
- Calibration drift — `predict_next_ball` outputs are bit-equal between cached and legacy paths on a real `MatchState` (max |Δ prob| = 0.00e+00 across all outcomes). Test: `scripts/tests/test_xgboost_model_v2_encoder_cache.py`.
- External callers (`sim_v1.py`, debug scripts) — original `self.batter_encoder` / `self.bowler_encoder` / `self.matchup_encoder` attributes are kept untouched; only the internal `extract_features` path changed.

**Out-of-scope follow-ups (same recipe, different files)**: legacy `XGBoostModel` (sim_v1_2.py:902–903 — note: uses raw int IDs, not `str()`); `LSTMModelV1` (1094, 1099, 1106, 1261); `MLPModelV1` (1546, 1551, 1670); `MLPModelV2` (1964, 1969, 1976, 2103); `TransformerModelV1` (2448, 2453, 2460, 2610). Each adds the `+1` offset for embedding padding. Not in the XGBoost eval hot path.

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

**Actual after Fix B alone**: 2.6× (107 min → 41.7 min).
**Actual after Fix A + B**: ~30× cumulative (107 min → ~3.5 min projected from 5×100 in 23.7 s). Inside the 10–15 min target with headroom; the new dominant cost is `xgboost.core.inplace_predict` itself, which is the legitimate floor.

---

## 5. Out of Scope (But Worth Noting)

- **Batching predictions across sims**: would be the fastest option but requires decoupling simulation state from prediction — a larger refactor. Not needed to hit the 10–15 min target.
- **LRU cache tuning**: `StatsProvider` LRU is already fine per the profile (no chunk thrashing visible).
- **Parallel eval orchestration**: the 2026-04-17 OOM came from `--parallel` with the current heavy inner loop; once A+B land, sequential should be fast enough that parallel isn't needed for most runs.

---

## 6. Post-Fix-A Measurements (2026-04-18)

### Profile benchmark (2 × 20)

| Metric | Pre-fix | Post-Fix B | **Post-Fix A + B** |
|---|---|---|---|
| Wall time | 91.4 s | 35.8 s | **2.25 s** |
| `extract_features` cumtime | 39 s | 34.1 s | **1.07 s** |
| `predict_next_ball` cumtime | 52 s | 1.56 s | 1.06 s |
| `LabelEncoder.transform` | 33.8 s (top 3) | 33.8 s (top 1) | **gone from top-30** |
| `is_scalar_nan` | 26.5 s (top 5) | 26.5 s (top 4) | **gone from top-30** |
| New top hot spot (tottime) | `_transform_pandas_df` (52 s) | `LabelEncoder.transform` (33.8 s) | `xgboost.core.inplace_predict` (0.68 s) |

### Real eval slice (5 matches × 100 sims, sequential)

23.7 s wall. Extrapolation to 44 × 100: **~3.5 min** (target was 10–15). Output `simulated_prob` is bit-identical to the post-B path (predict_next_ball test asserts max |Δ prob| < 1e-12 on a real `MatchState`).

### Parity verification

`scripts/tests/test_xgboost_model_v2_encoder_cache.py` — runs four checks against the real `models/xgb_v3/*` artifacts:
1. Round-trip equality for every class in batter/bowler/matchup encoders (cache dict vs `LabelEncoder.transform`).
2. Unknown ID → -1 across all three caches.
3. `extract_features` parity vs the legacy try/except path on a real test match.
4. `predict_next_ball` bit-equality on the same `MatchState` (max |Δ prob| < 1e-12).

```bash
uv run python scripts/tests/test_xgboost_model_v2_encoder_cache.py
```

---

## 7. Post-Fix-B Measurements (2026-04-18)

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

## 8. Post-Expansion Profile (2026-04-18, retrained model)

After the 2026-04-18 data refresh + retrain on 1.88M training balls, the 2×20 profile bench runs in **5.92 s** wall (vs 2.25 s pre-expansion). Per-match cost on the 44-match betting_test eval (seq, 100 sims) is **9.41 s/match → 6.9 min total**. Extrapolations to other test sets:

| Test set | Matches | Sequential wall |
|---|---|---|
| `betting_test` + `betting_odds_v3.json` (current default) | 44 | **6.9 min** (measured) |
| `betting_test` all files (if odds added) | 68 | ~10.7 min |
| `polymarket_test` + `betting_odds_polymarket.json` | 261 | **~41 min** |
| `golden_test` (no odds today) | 835 | ~131 min |

### New dominant hot spots

| Rank | Function | tottime / 5.92s | Notes |
|---|---|---|---|
| 1 | `xgboost.core.inplace_predict` | 2.71 s (46%) | Grew ~4× after retrain (deeper/more trees). Only batching kills it. |
| 2 | `datetime.strftime` | **0.53 s (9%)** | 53 calls/ball, all on the same `match_date`. Easy memoize. |
| 3 | `json.encoder.iterencode` | 0.38 s (6%) | Inside XGBoost's `make_jcargs`. Only reachable via batching. |
| 4 | `_get_snapshot_for_date` (ex-strftime) | ~0.20 s (3%) | Same date across a match — memoize result. |
| 5 | `extract_features` Python body | 0.20 s (3%) | Marginal. |

---

## 9. Candidate Next-Step Improvements

Ordered by payoff vs effort. None are implemented yet.

### Option A — `--n-workers N` for sim-level parallelism *(recommended first)*

`SimulationEngine._simulate_parallel` (`sim_v1_2.py:3352`) already supports multiprocessing within a match (100 sims split across workers), but defaults to `cpu_count()` under `spawn`. On a 16 GB Mac Mini each worker re-instantiates the XGBoost model (~125 MB) + StatsProvider cache (≤550 MB), so 8 workers blew out RAM in the 2026-04-17 attempt.

**Change**: expose `n_workers` through `SimulationConfig` → `MatchLevelEvaluator` → `run_sim_eval.py --n-workers N`. Bundle two tweaks:
- Pass `max_cached_chunks=2` to the workers' `StatsProvider` (cuts per-worker 550 MB → 220 MB).
- Set `OMP_NUM_THREADS=1` inside each worker so XGBoost doesn't over-subscribe CPU.

**Expected gain**: ~3× (Amdahl + spawn overhead). Polymarket 261 × 100: **41 min → ~14 min**.
**Memory budget at 4 workers**: 4 × (~125 MB model + ~220 MB cache + overhead) ≈ 2.5 GB — comfortable on 16 GB.
**Code size**: ~15 lines across `run_sim_eval.py`, `SimulationConfig`, `_simulate_parallel`.
**Risk**: spawn serializes `self`, which includes a warm StatsProvider cache — construct the StatsProvider inside the worker entry point, not at the parent, to avoid shipping 500 MB of chunk bytes per worker.

### Option B — Batch predictions across sims *(biggest lever, real refactor)*

Run all 100 sims lock-step per ball: at each ball, gather all live sims' feature rows into shape `(n_live, 63)` and call `predict_proba` once instead of 100 times. XGBoost amortizes kernel launch + `make_jcargs` over the batch — empirically ~5-10× faster than row-at-a-time on models of this size.

**Expected gain**: 3-5× over Option A (single-process, no RAM multiplication). Polymarket: **41 min → ~10 min**.
**Blockers**:
- Sims diverge (wicket / innings end / match end at different balls) — need to thread a "live sims" mask through `simulate_ball`.
- `MatchState` + `T20Rules.simulate_ball` are built assuming single-state-per-call — needs either a vectorized variant or batched-loop + masked predict.
- Determinism: per-sim seeding still works but requires per-sim RNG instances (not `random.seed()` global).
**Code size**: 100-200 LOC in `sim_v1_2.py`, plus a new test that asserts batched vs sequential produce identical distributions at a known seed.
**Risk**: subtle — the refactor touches the inner loop used by every XGBoost eval. Would need a side-by-side parity run on 5 matches before rollout.

### Option C — Small wins, stackable *(lowest risk, modest total)*

Independent drop-in optimizations. Good filler when bigger levers are blocked.

| Tweak | Where | Est. gain | Notes |
|---|---|---|---|
| Memoize `strftime` in `StatsProvider` | 9 call sites → one `@functools.lru_cache` helper | ~9% | `datetime` is hashable; per-worker cache is per-instance, safe. |
| Memoize `_get_snapshot_for_date` result for the most-recent date | `stats_provider.py:126` | ~5% | Caller must not mutate the returned dict — it's already treated as read-only. |
| Reduce `--n-sims` 100 → 50 | CLI flag | 2× | Sacrifices ~√2 MC noise on win probs; fine for iteration, not for final calibration. |

**Combined (without n-sims drop)**: ~15% → Polymarket ~35 min. With n-sims=50: ~18 min.

### Recommended sequencing

1. Ship Option A + the strftime memoization from C → Polymarket under 15 min, safe on 16 GB.
2. Run a Polymarket baseline at n_sims=100 to lock in calibration metrics.
3. If iteration speed still matters (e.g. for autoresearch), tackle Option B as a dedicated PR with parity tests.

Out of scope here: `--parallel` at the match level (across matches instead of within-match sims). Would require a different orchestration layer and `MatchLevelEvaluator` refactor; only worth it if single-match wall stays above a few seconds after A+B.

---

## 10. Reproducing

```bash
# Quick smoke (under 2 min)
uv run python scripts/profile_eval.py --matches 2 --n-sims 20

# Larger slice (~5 min)
uv run python scripts/profile_eval.py --matches 5 --n-sims 50

# After implementing a fix, compare:
uv run python -c "import pstats; pstats.Stats('eval_profile.prof').sort_stats('cumulative').print_stats(15)"
```

The benchmark prints a per-ball `ms/ball` number — use that as the primary regression signal; target is **≤ 5 ms/ball** post-fix.
