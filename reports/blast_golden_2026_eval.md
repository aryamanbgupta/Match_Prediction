# T20 Blast 2026 — out-of-sample golden eval (2026-05-28)

A second golden pool, analogous to the IPL-2026 golden set, but for the
**English domestic T20 Blast** (cricsheet `ntb_json.zip`). The 2026 Blast
season was never ingested locally — `extract_golden_cricsheet.py` only
pulls the IPL/PSL/SAT/internationals zips. This eval pulls the 17 matches
played so far (2026-05-22 → 2026-05-25) into a fresh out-of-sample pool
and scores the production match-level model on them.

## Method (no future leakage)

- **Extract**: `scripts/extract_blast_golden.py` → `data/golden_blast/t20s_json/`
  (17 matches; T20/male/date ≥ 2026-04-17; does not touch production data or
  the SQLite cache).
- **Materialize**: `materialize_match_features.py --source-dir data/t20s_json
  --extra-source-dir data/golden_blast/t20s_json --out-dir data/xgb_blast_golden`
  (unfrozen chronological walk). Each match's features reflect only
  pre-match state: player ELO/career stats via SQLite first-write-wins
  (state strictly before the match date, i.e. through the 2026-04-16 frozen
  corpus), and form/H2H/home trackers update *after* each match — so the
  5/24–5/25 matches can see 5/22–5/23 results, never the reverse. The 17
  matches land in the `golden_test` split.
- **Predict**: `predict_golden.py --model-dir models/xgb_match_v3_m7_production
  --parquet data/xgb_blast_golden/golden_test.parquet`. Production model,
  raw probabilities, no retrain.

No Polymarket (or any) market odds exist for the Blast → this is a pure
**prediction-quality** eval (LL / Brier / accuracy / calibration), not a
betting-ROI eval. The model has Blast history 2014–2025 in training, so
the teams/venues/most players are in-distribution; only the 2026 season is
held out.

## Headline

| Metric | Model | Coinflip ref |
|---|---|---|
| Log loss | **0.6724** | 0.6931 |
| Brier | **0.2396** | 0.2500 |
| Accuracy | **11/17 = 64.7%** | 50% |

The model beats coinflip on all three metrics on a brand-new season it has
never seen.

## Per-match

| Date | Match (team1 v team2) | P(team1) | Predicted | Actual | ✓/✗ |
|---|---|---:|---|---|:--:|
| 05-22 | Durham v Derbyshire | 55.6% | Durham | Durham | ✓ |
| 05-22 | Essex v Sussex | 42.6% | Sussex | Sussex | ✓ |
| 05-22 | Gloucestershire v Birmingham Bears | 44.0% | Birmingham Bears | Gloucestershire | ✗ |
| 05-22 | Hampshire v Somerset | 46.8% | Somerset | Somerset | ✓ |
| 05-22 | Kent v Middlesex | 53.4% | Kent | Kent | ✓ |
| 05-22 | Northamptonshire v Glamorgan | 51.3% | Northamptonshire | Northamptonshire | ✓ |
| 05-22 | Nottinghamshire v Yorkshire | 58.6% | Nottinghamshire | Yorkshire | ✗ |
| 05-22 | Surrey v Lancashire | 52.4% | Surrey | Surrey | ✓ |
| 05-22 | Worcestershire v Leicestershire | 46.6% | Leicestershire | Worcestershire | ✗ |
| 05-23 | Glamorgan v Gloucestershire | 50.7% | Glamorgan | Gloucestershire | ✗ |
| 05-24 | Birmingham Bears v Somerset | 46.9% | Somerset | Somerset | ✓ |
| 05-24 | Derbyshire v Yorkshire | 44.7% | Yorkshire | Yorkshire | ✓ |
| 05-24 | Leicestershire v Durham | 49.7% | Durham | Durham | ✓ |
| 05-24 | Middlesex v Surrey | 46.4% | Surrey | Surrey | ✓ |
| 05-24 | Northamptonshire v Worcestershire | 47.5% | Worcestershire | Northamptonshire | ✗ |
| 05-25 | Lancashire v Nottinghamshire | 49.9% | Nottinghamshire | Lancashire | ✗ |
| 05-25 | Sussex v Kent | 43.2% | Kent | Kent | ✓ |

## Calibration

| Confidence bucket | n | avg predicted | favored-team actual win% |
|---|---:|---:|---:|
| 0.50–0.55 | 11 | 52.2% | 63.6% |
| 0.55–0.60 | 6 | 56.6% | 66.7% |

The model is **low-resolution and under-confident** on the Blast: every
pick falls in 42–59% (max confidence 58.6%, no pick ≥ 60%), yet the favored
team won ~64–67% of the time. Pushing probabilities further from 50% would
have *improved* both LL and accuracy here. This is the same resolution
limitation seen on close IPL fixtures — county sides are tightly bunched in
ELO/team-strength, so the model rarely separates them strongly. Despite
that, the directional signal is real (64.7% accuracy on near-coinflip
picks).

## Caveats

- **n=17, single sample.** Wildly underpowered for ROI/CI claims; the
  64.7% accuracy carries a ~±23pp 95% interval. Treat as a directional
  sanity check, not a verdict.
- ELO/career features reflect state through 2026-04-16 only (no intra-2026
  refresh of the SQLite cache) — identical to the IPL golden methodology,
  the conservative no-leakage choice. Form/H2H trackers do see earlier 2026
  Blast results within the walk.
- 9 of 17 matches are on the same day (5/22) with no prior 2026 Blast
  signal available — hardest slice, and the model went 6/9 there.

## Artifacts

- `scripts/extract_blast_golden.py`
- `data/golden_blast/t20s_json/` (17 matches)
- `data/xgb_blast_golden/golden_test.parquet`
- `models/xgb_match_v3_m7_production/blast_golden_predictions.json`
