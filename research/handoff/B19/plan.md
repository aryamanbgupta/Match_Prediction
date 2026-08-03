# B19 — orchestrator plan (executor handoff)

Idea: **B19 [P2] Fresh-seed confirmation of the graft's batter-level cost
(B18 flag; B6 pattern)**. Claimed `RUNNING 2026-08-03T13:12:18Z` @ `ab32820`.

## Hypothesis (from IDEAS.md, verbatim in substance)

B18's only adverse signal anywhere was the G-2 positional cross-check on
`batter_runs_mae`: **+0.1174 [+0.0279, +0.2083] CI-clean worse** (n=2913),
while the pre-committed identity-keyed statistic read **+0.0688
[−0.0145, +0.1453] ~noise** (n=4254). Positional (row-order) pairing is
unreliable under engine changes — first-appearance row order permutes, the
documented reason B12 pre-committed keyed pairing — so this is plausibly a
pairing artifact. But B1→B6 established that a post-hoc-surfaced
batter-level signal must be confirmed or killed on fresh Monte Carlo draws
before it can be trusted either way. If the graft genuinely costs
batter-runs accuracy (the E2-validated continuous skill), the promotion
decision needs to know before any adoption of the sidecar.

This is a **diagnostic confirmation run — NO engine edit, NO new model, NO
fitting**. Both arms run committed code exactly as it stands at HEAD.
If at any point you believe a code change to `scripts/sim_v1_2.py` (or any
other engine/eval file) is required, STOP and record it in `result.md` —
that would mean the premise is broken.

## Step 0 — preconditions (verify, record output in raw/)

1. `git status --porcelain -- scripts/sim_v1_2.py` → must be EMPTY (clean
   working tree on the engine; verified by the orchestrator at claim time,
   re-verify).
2. `ls models/xgb_i7_noweights_production/` → must contain NO
   `extras_graft_v1.json` (production arm is sidecar-absent → graft
   inactive; B18 proved the sidecar-absent path float-exact + zero extra
   RNG draws).
3. `ls models/auto/b18/extras_graft_v1.json` → must exist, and its md5 must
   match the committed copy: `git show HEAD:models/auto/b18/extras_graft_v1.json | md5`
   — actually `models/` is gitignored; instead verify the file's content
   matches the B18 fit log values (p_wide 0.037702, p_no_ball 0.004409,
   wide mean 1.204439, no-ball mean 1.071168) by reading the JSON. If the
   sidecar is missing or its rates differ, STOP and record.
4. Usage json is the production default on both arms:
   `md5 -q models/bowler_phase_usage.json` == `2e650423f0c949631fca1f15dd1c8a56`.
   Do NOT add any B13 sidecar key anywhere.
5. Baseline detail JSONs for the self-test exist:
   `models/auto/d16/detail_noweights_raw_s46_n261.json` and
   `models/auto/b18/detail_b18_s46_n261.json`. Do NOT overwrite either.

## Step 1 — gate script FIRST (pre-commit before ANY s47 output exists)

Write `scripts/auto/b19_gate_analysis.py`, reusing the pairing machinery
from `scripts/auto/b18_gate_analysis.py` (which already computes BOTH the
identity-keyed and the positional (row-order) `batter_runs_mae` pairings,
plus per-family dBrier with cluster-boot by match, 2000 resamples, boot
seed 29). Requirements:

- **MANDATORY SELF-TEST** (B16/B18 precedent): from the two frozen s46
  detail JSONs it must reproduce B18's logged G-2 numbers before any fresh
  number exists:
  - keyed `batter_runs_mae` delta **+0.0688 [−0.0145, +0.1453]** (n=4254),
    raw 13.8864 → graft 13.9552;
  - positional delta **+0.1174 [+0.0279, +0.2083]** (n=2913);
  and B18's G-1 line deltas: innings_runs_ou_160_5 −0.0154
  [−0.0278,−0.0035], 170_5 −0.0115 [−0.0223,−0.0012], 180_5 −0.0153
  [−0.0246,−0.0064], pp_total_ou_45_5 −0.0126 [−0.0213,−0.0040], 50_5
  −0.0106 [−0.0196,−0.0012], 55_5 −0.0047 [−0.0121,+0.0027].
  Self-test tolerance: exact to the printed precision (same code, same
  boot seed). If the self-test fails, STOP — do not run any eval.
- Given the two fresh s47 details, it must print: keyed AND positional
  `batter_runs_mae` pairings side by side for BOTH seeds (s46 frozen +
  s47 fresh), the six G-1 line dBriers at s47, and the full 33-family
  context scan at s47 flagging every CI-clean mover in either direction.
- Commit it (`Auto[B19]: pre-commit gate script — self-test PASS, zero
  s47 output exists`) BEFORE launching any eval.

## Step 2 — TWO recipe-B runs at FRESH seed 47, SEQUENTIAL (one heavy
process at a time; 16 GB box; tee ALL raw output to
`research/handoff/B19/raw/`)

Seed choice is pre-decided: **47**. (46 consumed by d16/B18 recipe-B;
42/43/44 legacy history; 45/48/49 quote-path. B14's s47 was a VAL-split
quote run — different corpus and harness, so recipe-B test-set s47 is
fresh. Do not substitute another seed.)

Run A — no-graft arm (production stack):

```
uv run python scripts/sim_eval/prop_backtest.py \
    --test-dir data/polymarket_test --n-matches all --n-sims 100 --seed 47 \
    --model-path models/xgb_i7_noweights_production/xgboost_model_i7.pkl \
    --batter-encoder models/xgb_i7_noweights_production/batter_encoder_i7.pkl \
    --bowler-encoder models/xgb_i7_noweights_production/bowler_encoder_i7.pkl \
    --feature-columns models/xgb_i7_noweights_production/feature_columns_i7.txt \
    --stats-version i7 \
    --detail-out models/auto/b19/detail_raw_s47_n261.json \
    --report-out research/reports/auto/B19_props_raw.md
```

Run B — graft arm (B18 sidecar dir, unchanged):

```
uv run python scripts/sim_eval/prop_backtest.py \
    --test-dir data/polymarket_test --n-matches all --n-sims 100 --seed 47 \
    --model-path models/auto/b18/xgboost_model_i7.pkl \
    --batter-encoder models/auto/b18/batter_encoder_i7.pkl \
    --bowler-encoder models/auto/b18/bowler_encoder_i7.pkl \
    --feature-columns models/auto/b18/feature_columns_i7.txt \
    --stats-version i7 \
    --detail-out models/auto/b19/detail_b18_s47_n261.json \
    --report-out research/reports/auto/B19_props_b18.md
```

(Arg names are exactly B18's logged working invocation. NO
`--ball-calibrator` on either run — the promoted i7 stack runs RAW, D17.)

Banner verification (line-by-line, record in raw/): BOTH arms must show
i7 sqlite stats, venue encoder ACTIVE (373), B10 k_u=5.0, run-out channel
ACTIVE, NO calibrator line. Run B must ADDITIONALLY show
`B18 empirical extras graft ACTIVE (p_wide=0.037702, p_no_ball=0.004409, …)`;
Run A must NOT show the graft banner. If banners mismatch, kill and record.

Expected wall: ~22 min each (B18 s46: 1335.9 s; D17: 1310 s). Kill + record
CRASH at >50 min per run. Do NOT run them concurrently. Do not launch a
process and end your turn while it runs — block until completion (D2
lesson).

Commit evidence after both runs (`Auto[B19]: eval evidence — s47 twin
recipe-B runs`): the teed logs land in `research/handoff/B19/raw/`
(`models/auto/b19/` is gitignored — expected; commit code + logs only).

## Step 3 — gate output

Run `b19_gate_analysis.py` on the two s47 details; tee to
`research/handoff/B19/raw/gate_output.txt`.

## Gate (pre-committed here; the ORCHESTRATOR applies the verdict, not you)

- **P-1 (PRIMARY)**: keyed `batter_runs_mae` paired delta at s47
  (graft − no-graft, cluster-boot by match 2000/s29) shows **NO CI-clean
  regression** (a regression = the full CI above 0).
- **P-2**: the six B18 G-1 lines hold direction at s47 — NO line CI-clean
  worse (dBrier CI fully above 0).

Verdict mapping (recorded for the orchestrator; from IDEAS.md):
- P-1 AND P-2 → **LANDED** (flag resolved as pairing artifact; NOTHING
  ships — no production change, no sidecar promotion; the point of the
  idea is the information).
- P-1 fails (keyed CI-clean regression at s47) → **FAILED-with-finding**:
  the real batter-level cost gets recorded prominently for the promotion
  decision. B18 itself stays LANDED (its gate was met on its pre-committed
  statistic) — you do NOT revert anything of B18's.
- P-1 holds but P-2 fails → **TABLED** (exactly one gate met; a new
  instability flag on the G-1 lines).

Context (report, not gate): positional pairing at s47 side by side with
keyed; full 33-family scan; flag any CI-clean regression anywhere.

## Baselines (verbatim from B18's log — the numbers being re-tested)

- s46 keyed `batter_runs_mae`: 13.8864 → 13.9552, delta +0.0688
  [−0.0145, +0.1453] (n=4254) — the pre-committed G-2 statistic, ~noise.
- s46 positional: **+0.1174 [+0.0279, +0.2083]** (n=2913) — the flag.
- s46 G-1 lines: all six listed in Step 1; 5/6 CI-clean better, none worse.
- D16 context: production no-weights batter_runs_mae 13.891 (s46 baseline
  detail).

## Easy to get wrong

1. **This idea ships nothing and edits no engine code.** Both arms are
   committed artifacts + committed engine. Any temptation to "fix" the
   positional pairing inside the gate script's shared machinery: don't —
   reuse `b18_gate_analysis.py` functions verbatim so the s46 self-test
   proves the statistic is computed identically.
2. **Keyed vs positional**: the PRIMARY is the identity-keyed statistic
   (B12 precedent). The positional number is reported context — it does
   NOT enter the verdict. Don't let the gate script conflate them.
3. **Detail-out paths**: write NEW files under `models/auto/b19/`. Never
   overwrite `models/auto/b18/detail_b18_s46_n261.json`,
   `models/auto/d16/detail_noweights_raw_s46_n261.json`, or anything in
   `models/xgb_i7_noweights_production/` (read-only; record md5 of the
   booster before/after: `7ee1e180…`).
4. **Seed discipline**: `--seed 47` on BOTH runs (same-seed pairing).
5. One heavy process at a time; block synchronously until each finishes.
6. `prop_backtest.py` lives in `scripts/sim_eval/` — RUN it, never edit
   it. Same for every DO-NOT-CHEAT surface in `program.md` (read it
   first; obey it in full).

## Deliverables

- Commits, all prefixed `Auto[B19]:` — (1) gate-script pre-commit,
  (2) eval evidence. Commit before/after evals as specified above.
- `research/handoff/B19/raw/`: step-0 precondition output, self-test
  output, both run logs (teed, raw), banner blocks, gate output.
- `research/handoff/B19/result.md`: numbers VERBATIM from tool output —
  keyed + positional batter_runs_mae deltas with CIs at s47 (and the s46
  frozen values from the self-test side by side), all six G-1 dBriers +
  CIs at s47, 33-family scan summary (count + list of CI-clean movers),
  banner confirmations, wall times, commit SHAs, `git diff --stat` vs the
  claim commit `ab32820`, anything that crashed or ran long.
- NO verdict, NO revert, NO `results.tsv`/`IDEAS.md` edits, NO push, NO
  golden, NO second idea, NO background processes left running.
