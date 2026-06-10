# MLC 2025 — in-play chase win-probability: does ball-resolution add an edge?

88 mid-chase states (overs 5/10/15) across 32 MLC 2025 chases, 250 sims/state.
The ball-by-ball sim knows the live state AND who is at the crease; the baselines do
not. Question: does that buy anything over simple models? Lower log-loss / Brier and
higher AUC = better.

| predictor | log loss | Brier | AUC | what it knows |
|---|---:|---:|---:|---|
| resource baseline (GBM) | 0.6055 | 0.2167 | 0.795 | runs needed / balls left / wickets |
| **ball-by-ball sim** | 0.6042 | 0.1890 | 0.810 | live state + players at crease + team |
| pre-match team prior (STATIC) | 0.5954 | 0.2023 | 0.843 | only the pre-toss team rating — **no in-play info** |
| **resource + team prior** (fair baseline) | **0.4936** | 0.1654 | 0.833 | chase math + team rating |

## The verdict: no in-play edge either

1. **Sim vs the handicapped resource-only baseline:** ΔLL = -0.0004, match-clustered
   95% CI [-0.2730, +0.2898], P(sim better) = 0.51. A coin flip — the small
   raw gap is noise at n=32 matches.
2. **A STATIC pre-match team rating — using zero in-play information — already beats the
   sim** (LL 0.5954 / AUC 0.843 vs 0.6042 / 0.810).
3. **Against a fair baseline** (chase math + that team rating), the sim is clearly *worse*:
   ΔLL = +0.1122, CI [-0.0929, +0.3634], P(sim better) = 0.17.

So even in-play — the one place a ball-by-ball sim *should* have an edge — it does not beat
"team rating + chase arithmetic". Its live, delivery-level machinery adds nothing you can't
get from a pre-match rating plus the resource state, and its tail-hot miscalibration
actively costs it. **The edge hunt ends here: no validated edge, pre-match or in-play.**

*Artifacts: `scripts/inplay_winprob.py` (sim + resource baseline), `data/inplay_states.json`
(per-state sim/base/prior/outcome). Controls (prior, blend, bootstrap) computed from that.*
