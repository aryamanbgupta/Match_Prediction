#!/usr/bin/env python
"""Render the stage 1 report from the files of record (table numbers read from files; narrative authored here).

Reads: the twelve gate JSONs and three Holm outputs under
models/embeddings/seq_stage1/full/gate/, the merged realism.json / props.json
per arm, the 1b convergence JSON, the 1d full_record.json, and the pinned
config. Writes research/reports/embeddings/SEQ_STAGE1_REPORT.md.

Never opens data/golden or data/forward_holdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

CONFIG = REPO / "experiments" / "configs" / "seq_stage1_sim_v1.yaml"
STAGE = Path("models") / "embeddings" / "seq_stage1"
GATE = STAGE / "full" / "gate"
MERGED = STAGE / "full" / "merged"
TIMING = STAGE / "timing"
OUT = REPO / "research" / "reports" / "embeddings" / "SEQ_STAGE1_REPORT.md"

ARMS = ["A", "A50", "B", "C"]
CONTRASTS = ["C-B", "B-A", "C-A", "A50-A"]
SLICES = [("50000", ">=$50k (primary)"), ("100000", ">=$100k"), ("all", "all")]


def load(p: Path):
    return json.loads((REPO / p).read_text())


def sha12(p: Path) -> str:
    return hashlib.sha256((REPO / p).read_bytes()).hexdigest()[:12]


def f5(x):
    return f"{x:+.5f}"


def ci(v, nd=5):
    return f"[{v[0]:+.{nd}f}, {v[1]:+.{nd}f}]"


def gate_rows():
    rows = []
    for key, label in SLICES:
        for c in CONTRASTS:
            g = load(GATE / f"{c}_{key}.json")
            d, pr = g["delta_log_loss"], g["delta_profit"]
            rows.append(
                f"| {c} | {label} | {g['n_records']} | {g['block_count']} | "
                f"{f5(d['point'])} | {ci(d['ci95'])} | {pr['point']:+.4f} | "
                f"{ci(pr['ci95'], 4)} | {g['verdict']} | {'yes' if g.get('provisional') else 'no'} | "
                f"`{sha12(GATE / f'{c}_{key}.json')}` |")
    return rows


def holm_rows(key):
    h = load(GATE / f"holm_{key}.json")
    rows = []
    items = h.get("contrasts") or h.get("results") or h
    if isinstance(items, dict):
        items = [dict(name=k, **v) if isinstance(v, dict) else v for k, v in items.items()]
    for r in items:
        name = r.get("contrast") or r.get("name")
        fam = r.get("family", "")
        pt = r.get("point")
        adj = r.get("rank_local_interval") or r.get("adjusted_interval")
        praw = r.get("p_raw", r.get("raw_p"))
        floor = r.get("p_at_resolution_floor")
        ph = r.get("p_holm")
        lvl = r.get("rank_local_level", r.get("adjusted_level"))
        lab = r.get("label") or r.get("classification")
        rows.append(
            f"| {name} | {fam} | {pt:+.4f} | {ci(adj, 4)} | {('<0.0002' if floor else f'{praw:.4f}')} | "
            f"{'—' if ph is None else f'{ph:.4f}'} | {lvl:.4f} | **{lab}** |")
    return rows, h


def realism_table():
    rows = []
    for a in ARMS:
        r = load(MERGED / a / "realism.json")
        fi = r["aggregate"]["first_innings"]
        w = fi["fields"]["wickets"]
        rows.append(
            f"| {a} | {r['n_fixtures_scored']} | {fi['bias']['mean']:+.2f} | "
            f"{fi['abs_bias']['mean']:.2f} | {fi['coverage_p10_p90']['mean']:.3f} | "
            f"{fi['quantile_means']['p10']:.1f} / {fi['quantile_means']['p50']:.1f} / "
            f"{fi['quantile_means']['p90']:.1f} | {w['actual_mean']:.2f} / {w['sim_posterior_mean']:.2f} |")
    return rows


def props_table():
    fams = None
    per = {}
    for a in ARMS:
        p = load(MERGED / a / "props.json")
        per[a] = p["families"]
        if fams is None:
            fams = list(p["families"].keys())
    rows = []
    for fam in fams:
        cells = []
        for a in ARMS:
            f = per[a][fam]
            d = f["delta_sim_minus_baseline"]
            lo, hi = f["delta_ci95"]
            mark = "*" if (lo > 0 or hi < 0) else ""
            cells.append(f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]{mark}")
        metric = per["A"][fam]["metric"]
        rows.append(f"| {fam} | {metric} | " + " | ".join(cells) + " |")
    return rows


def timing_summary():
    rec = load(STAGE / "full" / "full_record.json")
    jobs = rec["jobs"]
    rows = list(jobs.values()) if isinstance(jobs, dict) else jobs
    out = []
    for a in ARMS:
        xs = [x for x in rows if x["arm"] == a]
        w = [x["wall_seconds"] for x in xs]
        out.append(f"| {a} | {len(xs)} | {min(w)/60:.0f} / {sum(w)/len(w)/60:.0f} / {max(w)/60:.0f} | "
                   f"{max(x.get('peak_rss_mb') or 0 for x in xs):.0f} |")
    return out, rec


def convergence_rows():
    c = load(TIMING / "convergence_1b.json")
    rows = []
    for r in c["spread_table"]:
        if r["contrast"] in ("C-B", "B-A"):
            rows.append(f"| {r['n_sims']} | {r['contrast']} | {r['range_95']:.4f} | "
                        f"{r['scaled_range_95']:.4f} | {r['sd']:.4f} |")
    var = [f"| {r['n_sims']} | {r['contrast']} | {r['scaled_paired_diff_sd']:.4f} |"
           for r in c["variability_rerun"] if r["contrast"] in ("C-B", "B-A")]
    return rows, var


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(CONFIG.read_text())
    cp = cfg["convergence_protocol"]
    dr = cfg["decision_rule"]
    h50_rows, h50 = holm_rows("50000")
    h100_rows, _ = holm_rows("100000")
    hall_rows, _ = holm_rows("all")
    tim_rows, rec = timing_summary()
    conv_rows, var_rows = convergence_rows()

    arms_tbl = []
    for a in ARMS:
        blk = cfg["arms"][a]
        arms_tbl.append(f"| {a} | {blk.get('role','')} | `{blk.get('model_dir','')}` | "
                        f"`{str(blk.get('checkpoint_md5', blk.get('booster_md5','')))[:12]}` | {blk.get('stats_version')} |")

    dev = cfg.get("deviations", [])
    asym = cfg.get("known_asymmetries", [])
    lim = cfg.get("known_limitations", [])
    rem = cfg.get("removed_asymmetries", [])

    md = f"""# Sequence track — stage 1 report (three ball models through the fixed simulator)

Generated {__import__('datetime').datetime.now().astimezone().isoformat(timespec='minutes')} by `scripts/sequence_track/render_stage1_report.py`
from the files of record: every number in a table is read from a file;
the narrative passages are authored text in the generator. Config `experiments/configs/seq_stage1_sim_v1.yaml`
(sha256 `{rec['config_sha256'][:12]}…`, `pin_stage1.py --verify` OK).
Acceptance checks and every result block: `docs/sequence_track/stage1_acceptance.md`.
Plan of record: `docs/SEQUENCE_TRACK_PLAN.md` § Stage 1.

**Status: single-checkpoint screen. No arm advances. No market claim.
Every gate is provisional (one training checkpoint per neural arm), so per
invariant 9 nothing here can be LANDED.**

## 1. Question

(i) System comparison: does the 50-feature token MLP (B) or full T1 (C),
on the certified replay path, simulate the winner market as well as the
production i7 ball model (A, 114 features)? (ii) Incremental evidence: does
full T1 improve on the token MLP in rollout? A50 (production XGBoost on
T1's 50 features) is the equal-information control. Pre-registered
expectation, written before any run: **parity everywhere, no arm advancing.**

## 2. Arms and settings

| arm | role | model dir | checkpoint | cache |
|---|---|---|---|---|
{chr(10).join(arms_tbl)}

One stats cache (i7, `venue_aliases_v1`, same-day ordering contract) for all
four arms (user decision 2026-09-11); B and C retrained on `data/xgb_data_i7`
(five seeds each, validation-LL selection, both select seed 101). Iteration
set `data/polymarket_test_v2` (255 fixtures, 2025-09-10 → 2026-04-16), odds
role `odds_iteration_v2`, roster-aware empirical bowler selector, B18 extras
graft on every arm, run-out constant 0.075077, no calibration, prefix cache
off, cpu, **threads 1**, clip [0.01, 0.99], per-fixture seed = low 31 bits of
sha256(`<id>:20260910`), **n_sims = {cp['chosen_n_sims']}**. One runner, one
chronology, one replay lifecycle; the cross-arm audit asserted identical
fixture sets, as-of stamps, eligibility, odds-row identity, selector and
sidecar hashes, config/odds/context/engine/runner hashes on every shard.

## 3. Decision rule (registered before the run)

Primary slice ≥$50k. Contrasts are candidate − reference in winner log
loss (negative favours the candidate). Confirmatory family C−B, B−A, C−A,
Holm-adjusted; A50−A exploratory. Equivalence margin {dr['equivalence_margin_ll']}: interval
inside ±margin → parity; interval excludes zero and point beyond the margin
→ favourable / adverse; else inconclusive. B advances only if B−A is
favourable, C only if C−A is favourable; a favourable C−B alone is evidence
about that pair for stage 2, not an advancement.

## 4. Result — primary slice, Holm step-down

Intervals in the Holm tables are rank-local percentile intervals at level
1−α/(m−k+1) from the gate's own resamples; they are not simultaneous Holm
confidence intervals. Favourable/adverse requires the Holm-adjusted p ≤ 0.05
(step-down) and a point beyond ±0.007; parity is decided on the gate 95%
interval.

| contrast | family | point | rank-local interval (not simultaneous) | raw p | Holm p | level | label |
|---|---|---|---|---|---|---|---|
{chr(10).join(h50_rows)}

**No arm advances.** C−B is favourable (departs from the parity expectation
in the favourable direction); B−A and C−A are inconclusive (intervals wider
than the ±0.007 band; B−A leans adverse); A50−A is adverse.

Secondary slices (the same Holm step applied for reporting; they are outside the confirmatory decision, which is the primary slice only):

≥$100k:

| contrast | family | point | interval | raw p | Holm p | level | label |
|---|---|---|---|---|---|---|---|
{chr(10).join(h100_rows)}

all:

| contrast | family | point | interval | raw p | Holm p | level | label |
|---|---|---|---|---|---|---|---|
{chr(10).join(hall_rows)}

Full gate table (Δprofit is units per flat 1-unit bet at cost scenario
0 bps / 0 bps on winnings; descriptive only — no betting claim):

| contrast | slice | n | blocks | ΔLL point | ΔLL ci95 | Δprofit point | Δprofit ci95 | gate verdict | provisional | gate sha256 |
|---|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(gate_rows())}

The gate's `PROMISING` / `FAILED` labels are its own LANDED-rule readout;
the stage-1 classification is the label column of the Holm tables.

### Reading

- **Full T1 beats the token MLP in rollout on every slice** (primary
  −0.0257 [−0.0346, −0.0102] rank-local interval; Holm p 0.0018). This is
  one checkpoint per arm. Historical motivation only: the 2026-08
  teacher-forced ablation (`T1_ABLATION_V1_MPS.md`) found a T1−MLP gap of
  −0.0004 with an interval crossing zero on *different* checkpoints and a
  different frame; that is not a matched teacher-forced evaluation of these
  retrained checkpoints, and a significant result in one experiment beside a
  non-significant one in another does not establish a difference between
  them. A matched teacher-forced score of these two checkpoints is the
  follow-up that would test whether rollout and teacher forcing disagree.
- **C−A is inconclusive with a point estimate near zero** (−0.0009,
  [−0.0277, +0.0288]); the interval establishes neither equivalence nor
  non-inferiority under ±0.007. **A50−A is adverse** (+0.0276
  [+0.0095, +0.0419], exploratory): within XGBoost, dropping the 64
  production-only features costs about 0.028 on this slice. Together these
  motivate further investigation of architecture and information use; they
  do not establish parity and do not quantify any recovery from sequence
  history (the registered rule forbids reading C−B or C−A as a measurement
  of sequence memory).
- **The token MLP loses to production** on both secondary slices and leans
  that way on the primary (inconclusive there).
- The MLP−logistic and T1−MLP findings of the ablation report describe
  different checkpoints on a different frame and are not restated here.

## 5. Uncertainty and what is not claimed

`tournament_time_block_v1`: 10,000 seed-42 whole-event resamples; 18 blocks
on the primary slice (167 paired records; one of 168 dropped symmetrically
by the gate's pairing rule), 11 at ≥$100k, 25 on all. Monte Carlo noise was
not measured on the primary slice or at 1,600 simulations; § 7 gives the
extrapolated two-batch difference spreads from the ten-fixture timing shard
(at four threads, before the thread-cap change), which are the basis of the
user's count decision and nothing more. Every gate is provisional (single
checkpoint). Confirmation of any contrast needs
five training seeds and two independent simulation batches (plan § Stage 1).
No market claim: the Δprofit intervals are reported, not claimed.

## 6. Exploratory readouts (no advancement, no claim)

Realism, first innings, {ARMS[0]}…{ARMS[-1]} (247 fixtures scored per arm; 3
no-decided-winner and 5 D/L fixtures excluded):

| arm | n | bias (runs) | abs bias | P10–P90 coverage | sim P10 / P50 / P90 | wickets actual / sim |
|---|---|---|---|---|---|---|
{chr(10).join(realism_table())}

Props, paired sim − as-of fair baseline per family (Brier; MAE for `_mae`),
250 fixtures per arm, 15 of 33 families scorable; `*` marks an interval
excluding zero; negative favours the simulator:

| family | metric | A | A50 | B | C |
|---|---|---|---|---|---|
{chr(10).join(props_table())}

All four arms beat the fair baselines on the innings-total, powerplay-total,
team-highest-score and batter-50+ lines with intervals excluding zero, and
lose on the bowler-wicket lines. Context: the original E2 v2 audit
(`reports/e2_prop_fair_baselines.md`, retired v7 stack, 100 simulations)
found no binary family beating an as-of fair baseline; BR2's later
restatement (`research/handoff/BR2/e2_restated.md`) already showed
favourable movement on several overlapping families on the repaired
engine, so stage 1 is not the first such reading. These are per-family
intervals with no multiplicity adjustment, on a different engine state and
simulation count from either predecessor; exploratory, and not comparable
line for line.

## 7. 1b convergence and timing (spreads only)

Ten-fixture long-innings shard, three batch seeds, candidates 50/100/200/400/800.
Empirical range (max − min over the three batches; the config calls this
`range_95`) of the paired shard-mean contrast and its full-set equivalent
(× sqrt(10/255)):

| n_sims | contrast | range_95 | scaled range_95 | SD |
|---|---|---|---|---|
{chr(10).join(conv_rows)}

Two-seed variability: the SD of the per-fixture DIFFERENCE between the
contrast under batch seed 20260910 and under 20260911 (so, for equal
independent batch variances, √2 × a single batch's SD), divided by
sqrt(10), then × sqrt(10/255). An extrapolation from a deliberately
long-innings shard on all ten fixtures (not the primary slice), measured at
four threads (the full run used one; floating-point reductions can differ),
and never measured at 1,600:

| n_sims | contrast | scaled SD |
|---|---|---|
{chr(10).join(var_rows)}

"range_95" above is the min–max of three batches, an empirical range, not
an established population 95% range. The registered range rule (< 0.002)
was not met at any permitted count (the a/sqrt(n) fit crosses at ≈5,264 for
C−B, above the 3,200 joint seed cap). **User decision 2026-09-11:** the
operational reading is the two-batch-difference scaled SD at or below
≈0.002 at the chosen count; n_sims = 1,600 (≈0.0018 / 0.0023 extrapolated
by 1/√2 from 800). Recorded in `convergence_protocol.stop_rule_reading` as
a change of statistic made with the numbers in hand. 50 is a noise-curve
point only.

Thread cap: `sim_t1.py` hard-coded four torch threads and defeated the
runner's cap; fixed (honours `OMP_NUM_THREADS`), threads registered at 1,
ten one-thread shards give 6.4× the throughput of one process. Full run wall
(ten concurrent shards):

| arm | shards | wall min / mean / max (min) | peak RSS (MB) |
|---|---|---|---|
{chr(10).join(tim_rows)}

## 8. 1c shard consistency

Serial vs sharded per-fixture outputs identical on all four arms across the
five registered boundary cases (parsed evaluator record, raw simulation
rows and per-fixture provenance compared field by field; the one exclusion
is the run-cumulative `matches_advanced` counter, replaced by the ordered
same-day list and a decomposition check — acceptance D10) after one fix: the frozen runner derived the I3
block id from the run's own fixture directory, so a shard stamped a
different `block_start` than the serial run for one fixture; `run_arm
--cluster-source-dir` (pinned to the registered set, asserted by the audit)
fixed it at the source; re-run 117/117 checks pass, merge re-stamp changed 0
records on every arm.

## 9. Registered deviations, asymmetries, limitations (restated, none dropped)

Deviations ({len(dev)}), each with its registered consequence:

{chr(10).join(f"- `{d['id']}`: {d.get('consequence', d.get('what', ''))}" for d in dev)}

Known asymmetries ({len(asym)}):

{chr(10).join(f"- `{a['id']}`: {a.get('consequence', a.get('what', ''))}" for a in asym)}

Removed 2026-09-11 by the i7 retrain: {"; ".join(r["id"] for r in rem)}.

Known limitations ({len(lim)}):

{chr(10).join(f"- `{l['id']}`: {l.get('consequence', l.get('what', ''))}" for l in lim)}

Astra review rounds for the retrain/re-pin commit: D8 in the acceptance
file; end-of-stage review: D12.9.

## 10. In plain language: what was tested, what came out, what "no arm advances" means

**What we tested.** Four ball-by-ball models were each asked to play out
every one of 255 real T20 matches 1,600 times, from the same starting
information, using the same simulator, the same bowler-choice rules and the
same extras law. From those replays each model produced a win probability
for every match, and we compared those probabilities with what actually
happened, using log loss (lower is better). The four models: **A**, the
current production XGBoost with 114 features (the model we already use);
**A50**, the same XGBoost family given only the 50 simpler features the
neural models see; **B**, a small neural network (token MLP) on those 50
features with no memory of earlier balls; **C**, the transformer (T1) on the
same 50 features, which can look back over the balls already bowled in the
innings.

**What came out, on the 168 matches with the deepest betting markets.**
The transformer beat the memory-less network clearly (about 0.026 better
log loss, and the interval does not come near zero). The transformer and
production came out indistinguishable: the point difference is about
0.001 in the transformer's favour, but the uncertainty band is ±0.028
wide, so we cannot say either is better. The memory-less network was
somewhat worse than production, and on the wider slices clearly worse.
The XGBoost restricted to 50 features was clearly worse than production by
about 0.028, which says the 64 extra hand-built features matter to XGBoost.

**What "no arm advances" means.** Before the run we wrote down a rule: a
neural model "advances" (goes forward to the next round of testing as a
candidate to replace or join production) only if it beats production by a
clear margin, with the uncertainty band excluding zero and the point beyond
0.007. Neither B nor C met that. C tied production; B lost. So under the
rule nothing moves forward as a production candidate, and production stays
the model of record. "No arm advances" is not a statement that the
transformer is worse; it is a statement that we did not show it to be
better.

**So which is the best model right now?** For predicting match winners in
production: the existing XGBoost (A). It is the model of record, the
transformer only matched it, and the transformer was trained once (one
random seed), so its number could move if retrained. For the research
question "does looking back over the innings help?": the transformer beat
the memory-less network on identical inputs, which is the first clean
rollout evidence in this repo that within-innings sequence carries
something. That is the finding that goes into stage 2.

**What comes next.** (1) Confirm, not assume: retrain the transformer and
the MLP on five seeds and simulate two independent batches, so the C−B gap
is shown to be a property of the architecture and not of one lucky
checkpoint. (2) Score these same two checkpoints under teacher forcing (one
ball at a time, no rollout) to see whether the rollout advantage is real
or an artefact of how the simulator uses the model. (3) Stage 2 as planned:
variants of the transformer that forget stale history or attend only to the
same batter and bowler, to learn *which* part of the history helps and
whether the death-over harm seen in August goes away. (4) A separate,
registered look at why every arm beat the fair prop baselines on totals
lines here when the old audit said none did; that could matter for prop
markets but is exploratory today. Nothing here is a betting claim: every
result is one checkpoint, and no market edge has been established.

## 11. Next (as registered)

Per the plan: no arm advances; C−B goes to stage 2 as evidence about the
full-T1 vs token-MLP pair (which arm variants, forgetting / ownership). A
matched teacher-forced score of these two checkpoints would test whether
rollout and teacher forcing actually disagree. A50−A adverse beside an
inconclusive C−A is a lead on what the 64 production-only features carry,
not a measurement. Any claim about this checkpoint needs the five-seed,
two-batch confirmation.
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
