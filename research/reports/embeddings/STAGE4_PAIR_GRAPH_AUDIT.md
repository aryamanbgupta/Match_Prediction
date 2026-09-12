# Stage 4 — batter x bowler exposure-graph audit (rung 4a)

Built 2026-09-12 · frame `data/xgb_data_i7` · `scripts/sequence_track/stage4_pair_graph_audit.py`

## Train-split bipartite graph

- balls: 1,876,971
- distinct batters: 6,309; distinct bowlers: 4,707
- distinct (batter, bowler) edges: 250,649
- connected components: **1**
- largest component: **11,016 nodes** (100.00% of all batter/bowler nodes)

## Edge locality by competition tier

`internal` = the pair is only ever seen inside this tier.

| tier | distinct edges | internal to tier |
|---|---|---|
| 1 | 64,248 | 91.7% |
| 2 | 96,834 | 88.4% |
| 3 | 66,979 | 84.2% |
| 4 | 40,237 | 82.7% |

## Held-out pair coverage

| split | rows | pair seen | pair unseen, both players seen | batter or bowler unseen |
|---|---|---|---|---|
| validation | 124,292 | 40.7% | 45.9% | 13.4% |

Expected from the ladder's earlier accounting: 40.7 / 46.0 / 13.4 % on validation. The test split is sealed for Stage 4 and is not reported here.
