# B14 — raw excerpts

Small committable excerpt of the executor's tool output.

*(Correction 2026-08-01: the original note here claimed the raw files "stay
on disk untracked, as in B10/B12" — that was wrong on both counts: B10 and
B12 both track their `raw/gate_output.txt`. B14's `gate_output.txt` and
`.log.txt` twins of the three logs are now committed, matching the
B10/B12/D16/D17 convention.)*

## 1. Val quote run — startup banner

From `research/handoff/B14/raw/val_run.log`, lines 1–13, verbatim:

```
Loading stats provider + player metadata + model ...
WARNING: SQLite same-day ordering mismatch: models/player_stats_cache_v3.sqlite has None, code expects 'date_then_match_id_lexicographic_v1'. Rebuild the cache before deterministic materialization.
StatsProvider: using SQLite backend player_stats_cache_v3.sqlite (56.8 MB)
Loading player metadata from data/all_players_enriched.csv...
  Loaded 11,256 players
  Batting: 8178 right, 1814 left, 1264 unknown
  Bowling: 6256 right, 1381 left, 3619 unknown
Ball calibrator: vector scaling (models/xgb_v3/vector_scaling_calibrator_v1.pkl)
Loaded XGBoost v2 model with 114 features with real player stats with player metadata; venue encoder ACTIVE (467 venues)
Run-out dismissal channel ACTIVE (p_runout=0.0751, nonstriker_share=0.4685)
B5 in-play quotes: 545 matches x 100 sims x checkpoints [6, 10, 15], seed 47
  [1/545] 1439899                                                 cps=[6, 10, 15] (7.5s)
  [2/545] 1439900                                                 cps=[6, 10, 15] (7.5s)
```

Note what is present and what is absent: `venue encoder ACTIVE (467 venues)`
is present; there is **no `B10 ... ACTIVE` banner** — a grep for `b10|usage`
over all 562 lines of the log returns nothing, confirming the run used the
pinned pre-B12 legacy usage prior
(`models/auto/b12/bowler_phase_usage_pre_b12.json`,
md5 `ea0c73d3ddb48f499b6273f9a397b0e3`) rather than the B10 as-of usage.

## 2. Val quote run — first skip lines and final Done line

```
  [24/545] SKIP 1443093.json: innings 1 curtailed (31 legal balls, 1 dismissals)
  [29/545] SKIP 1443098.json: innings 1 curtailed (114 legal balls, 7 dismissals)
  [34/545] SKIP 1443550.json: innings 1 curtailed (90 legal balls, 9 dismissals)
  [55/545] SKIP 1449649.json: innings 1 curtailed (66 legal balls, 3 dismissals)
```

All 33 skips are structural: 30 `innings 1 curtailed (...)` plus 3
`not a 20-over match (overs=50)`. 33/545 = 6.06%, under the plan's 10% abort
condition.

Final line (log line 562), verbatim:

```
  [545/545] 1490890                                                 cps=[6, 10, 15] (6.0s)

Done in 3215.2s — 1532 quote rows from 512 matches (33 matches skipped) -> models/auto/b14/quotes_val_s47_n545.json
```

## 3. Calibrator fit table

Full contents of `research/handoff/B14/raw/fit.log`, verbatim:

```
B14 quote-calibrator fit on models/auto/b14/quotes_val_s47_n545.json
  config: n_sims=100 seed=47 quote_center=sim_p50 usage_json=models/auto/b12/bowler_phase_usage_pre_b12.json elapsed=3215.2s
  rows: 1532 from 512 matches (33 matches skipped)

 cp     n    shift  scale  cov_raw  cov_corr  mae_raw  mae_corr
---------------------------------------------------------------
  6   512  -1.4482   1.19   0.7344    0.7988  20.2490   20.1953
 10   512  -1.7812   1.09   0.7754    0.8027  16.0430   16.0785
 15   508  -2.9715   1.26   0.7106    0.7992  11.1604   11.0864

Wrote models/auto/b14/quote_calibrator.json
```

## 4. Gate verdict block

Full contents of `research/handoff/B14/raw/gate_output.txt`, verbatim:

```
B14 gate analysis
  calibrator: models/auto/b14/quote_calibrator.json (fit on models/auto/b14/quotes_val_s47_n545.json, val target coverage 0.8)
  TEST quotes: models/auto/b5/quotes_s43_n261.json
  config: n_sims=100 seed=43 quote_center=sim_p50 elapsed=1496s
  rows: 756 from 253 matches (8 matches skipped)
  correction applied (val-fit): cp6: shift -1.4482 scale 1.19  cp10: shift -1.7812 scale 1.09  cp15: shift -2.9715 scale 1.26

checkpoint  6 (n=253):
  MAE  corr(P50)  21.139  naive  25.897  dMAE  -4.759 [-7.790, -1.791]  CORRECTED BETTER
  MAE  raw (P50)  20.860  naive  25.897  dMAE  -5.038 [-7.970, -2.082]   (uncorrected context)
  P10-P90 coverage corr  0.802 [0.755, 0.850]  target [0.7, 0.9]  IN BAND
  P10-P90 coverage raw   0.755 [0.704, 0.810]   (uncorrected context)
  context: bias corr P50 +6.118  bias raw P50 +4.670  band width corr 68.8  raw 57.8  actual sd 29.9
checkpoint 10 (n=253):
  MAE  corr(P50)  17.435  naive  20.000  dMAE  -2.565 [-4.679, -0.392]  CORRECTED BETTER
  MAE  raw (P50)  17.061  naive  20.000  dMAE  -2.939 [-4.947, -0.909]   (uncorrected context)
  P10-P90 coverage corr  0.802 [0.755, 0.854]  target [0.7, 0.9]  IN BAND
  P10-P90 coverage raw   0.794 [0.743, 0.846]   (uncorrected context)
  context: bias corr P50 +4.985  bias raw P50 +3.204  band width corr 52.3  raw 48.0  actual sd 25.3
checkpoint 15 (n=250):
  MAE  corr(P50)  12.597  naive  13.575  dMAE  -0.977 [-2.565, +0.461]  CORRECTED BETTER
  MAE  raw (P50)  12.314  naive  13.575  dMAE  -1.261 [-2.613, -0.004]   (uncorrected context)
  P10-P90 coverage corr  0.756 [0.704, 0.808]  target [0.7, 0.9]  IN BAND
  P10-P90 coverage raw   0.664 [0.608, 0.724]   (uncorrected context)
  context: bias corr P50 +3.485  bias raw P50 +0.514  band width corr 37.2  raw 29.5  actual sd 16.9

pooled paired dMAE (corrected - naive, 756 rows, cluster-boot by match): -2.774 [-4.631, -0.864]

GATE 1' (corrected MAE beats naive at all 3 cps + pooled cluster-boot CI hi < 0): MET (per-cp [True, True, True], pooled hi -0.864)
GATE 2' (corrected coverage in [0.7,0.9] at all 3 cps): MET (per-cp [True, True, True])

(verdict mapping is applied by the orchestrator, not here)
```

## 5. Pre-flight re-verification output

```
$ md5 -q models/auto/b12/bowler_phase_usage_pre_b12.json
ea0c73d3ddb48f499b6273f9a397b0e3          (expected ea0c73d3ddb48f499b6273f9a397b0e3)

top-level keys: ['built_at', 'by_player', 'by_year_league', 'gender', 'global_league', 'n_deliveries', 'n_matches', 'n_unresolved_names', 'schema_version', 'source_dir']
has b10_asof_usage: False

quotes keys: ['config', 'rows', 'skips']
config: {"ball_calibrator": "models/xgb_v3/vector_scaling_calibrator_v1.pkl", "bowler_selector": "empirical", "checkpoints": [6, 10, 15], "elapsed_s": 1495.8358759880066, "model": "models/xgb_v3/xgboost_model_v3.pkl", "n_sims": 100, "quote_center": "sim_p50", "seed": 43, "test_dir": "data/polymarket_test"}
n_rows: 756
n_matches distinct: 253
   skips = list len 8
```

The frozen test quotes file `models/auto/b5/quotes_s43_n261.json` was read
only — never regenerated, never written.
