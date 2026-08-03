# B5 executor result — in-play over/under quote prototype

Claim commit: `404905f` (`Auto[B5]: claim`, 2026-07-31T07:57Z).
Executor scope: PROTOCOL steps 3–4 only (preflight, eval, record). The verdict
line below is **script output copied verbatim**, not an executor judgment.

## 1. Preflight — gate-script integrity (BEFORE any eval result existed)

`git log --oneline -2 -- scripts/auto/b5_gate_analysis.py` returned exactly one
commit, as expected:

```
26a7fd9 Auto[B5]: implement — in-play remaining-runs quote harness (non-engine; sim_v1_2.py + eval framework untouched). ...
```

(full subject preserved in `raw/gate_provenance.log`; it contains the
pre-committed gate definition verbatim.)

`git status --porcelain` at preflight: **empty (tree clean)**.

The gate script was **not modified at any point** in this session.

## 2. Unit check — re-run on CURRENT code (post-I15-compat), PASS

`uv run python scripts/auto/b5_unit_check.py` — full log in `raw/unit_check.log`.
Summary lines verbatim:

```
Part 1: replay parity PASS on 253/261 matches (8 out of scope)
  skip reason x8: innings 1 curtailed
Part 2: crease-pair match 756/756 (100.0%)
Part 3: live-path smoke (first in-scope match, cp=6, 3 sims x2)
  1477609: remaining draws [110.0, 122.0, 120.0] (repeat [110.0, 122.0, 120.0])

B5 unit check: ALL PASS
```

Part 3 environment assertions held on the current default sim path:

```
Loaded XGBoost v2 model with 114 features with real player stats with player metadata; venue encoder ACTIVE (467 venues)
Run-out dismissal channel ACTIVE (p_runout=0.0751, nonstriker_share=0.4685)
```

No compat fix to `b5_inplay_quotes.py` / `b5_unit_check.py` was needed — the
harness passed unmodified on current code. **Zero code changes this session.**

Non-fatal warning present in both unit check and eval (pre-existing, unchanged
by this run):

```
WARNING: SQLite same-day ordering mismatch: models/player_stats_cache_v3.sqlite has None, code expects 'date_then_match_id_lexicographic_v1'. Rebuild the cache before deterministic materialization.
```

## 3. Eval

```
uv run python scripts/auto/b5_inplay_quotes.py \
    --test-dir data/polymarket_test --n-sims 100 --seed 43 \
    --out models/auto/b5/quotes_s43_n261.json
```

Run synchronously in the executor session, exit code 0. Full log in
`raw/eval.log` (274 lines). Header and terminal lines verbatim:

```
B5 in-play quotes: 261 matches x 100 sims x checkpoints [6, 10, 15], seed 43
```

```
Done in 1495.8s — 756 quote rows from 253 matches (8 matches skipped) -> models/auto/b5/quotes_s43_n261.json
```

**Wall time 1495.8s (24.9 min).** Per-match pace ~5.8–6.1 s, one progress line
per match through `[261/261]`. Row count 756 exactly matches the unit check's
756 checkpoint crease-pairs.

All 8 skips were the pre-declared curtailed-innings exclusions, verbatim:

```
  [21/261] SKIP 1493238.json: innings 1 curtailed (66 legal balls, 5 dismissals)
  [51/261] SKIP 1493268.json: innings 1 curtailed (30 legal balls, 0 dismissals)
  [62/261] SKIP 1493279.json: innings 1 curtailed (60 legal balls, 5 dismissals)
  [68/261] SKIP 1494256.json: innings 1 curtailed (24 legal balls, 0 dismissals)
  [77/261] SKIP 1494267.json: innings 1 curtailed (72 legal balls, 7 dismissals)
  [196/261] SKIP 1514482.json: innings 1 curtailed (72 legal balls, 6 dismissals)
  [213/261] SKIP 1527562.json: innings 1 curtailed (78 legal balls, 5 dismissals)
  [238/261] SKIP 1527686.json: innings 1 curtailed (66 legal balls, 3 dismissals)
```

## 4. Gate analysis — output copied VERBATIM

```
uv run python scripts/auto/b5_gate_analysis.py \
    --quotes models/auto/b5/quotes_s43_n261.json
```

```
B5 gate analysis on models/auto/b5/quotes_s43_n261.json
  config: n_sims=100 seed=43 quote_center=sim_p50 elapsed=1496s
  rows: 756 from 253 matches (8 matches skipped)

checkpoint  6 (n=253):
  MAE  sim(P50)  20.860  naive  25.897  dMAE  -5.038 [-7.970, -2.082]  SIM BETTER
  P10-P90 coverage  0.755 [0.704, 0.810]  target [0.7, 0.9]  IN BAND
  context: bias P50 +4.670  mean +3.163  band width P90-P10 57.8  actual sd 29.9
checkpoint 10 (n=253):
  MAE  sim(P50)  17.061  naive  20.000  dMAE  -2.939 [-4.947, -0.909]  SIM BETTER
  P10-P90 coverage  0.794 [0.743, 0.846]  target [0.7, 0.9]  IN BAND
  context: bias P50 +3.204  mean +1.994  band width P90-P10 48.0  actual sd 25.3
checkpoint 15 (n=250):
  MAE  sim(P50)  12.314  naive  13.575  dMAE  -1.261 [-2.613, -0.004]  SIM BETTER
  P10-P90 coverage  0.664 [0.608, 0.724]  target [0.7, 0.9]  OUT OF BAND
  context: bias P50 +0.514  mean +0.199  band width P90-P10 29.5  actual sd 16.9

pooled paired dMAE (sim - naive, 756 rows, cluster-boot by match): -3.086 [-4.869, -1.289]

GATE 1 (MAE beats naive, all 3 cps + pooled CI<0): MET (per-cp [True, True, True], pooled hi -1.289)
GATE 2 (coverage in [0.7,0.9] at all 3 cps): NOT MET (per-cp [True, True, False])

VERDICT: TABLED
```

## 5. Commits created this session

| SHA | Subject |
|---|---|
| `ab06f0c` | `Auto[B5]: preflight — unit check re-run PASS on current code, gate script verified intact at 26a7fd9` |
| (this file) | `Auto[B5]: executor result` |

`a979bbe` (`Auto[B5]: executor plan`) predates this executor session.

`git diff --stat 404905f HEAD` at the point of the result commit:

```
 research/handoff/B5/plan.md                 | 143 ++++++++++++++++++++++++++++
 research/handoff/B5/raw/gate_provenance.log |   1 +
 research/handoff/B5/raw/unit_check.log      |  15 +++
 3 files changed, 159 insertions(+)
```

**No source file changed.** The diff vs the claim commit is the plan plus
preflight evidence only; `scripts/auto/b5_*.py` are byte-identical to `26a7fd9`.

## 6. Crashes / retries / long runs

- **Nothing crashed. Nothing was retried. The eval ran once, to completion.**
- The eval exceeded the executor's single 600s tool-call cap, so the foreground
  call was moved to background by the harness while the `uv run` child kept
  running. It was **not** restarted; the same PID (28304) was polled to
  completion and reported exit code 0. `pgrep` confirmed a single eval process
  throughout — no second copy was ever started.
- Progress during the tool-call gap was tracked via the
  `quotes_s43_n261.partial.jsonl` sidecar (310 rows @ 10 min → 613 @ 20 min →
  756 final), because Python block-buffers stdout through `tee`; `raw/eval.log`
  only flushed at process exit. The sidecar is a progress artifact; the gate
  read only the final JSON.
- Wall time 24.9 min, inside the 30–60 min plan estimate and far inside the 2 h
  kill threshold.
- Raw evidence logs are `*.log`, which the repo `.gitignore` excludes, so they
  were committed with `git add -f`. `models/auto/b5/` was **not** force-added
  and remains gitignored.
