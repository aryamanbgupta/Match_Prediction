# manifest-exempt: embeddings-ladder experimental artifact namespace
"""Render the Stage 3a night-3 report from the files of record.

Every number in every table is read from
`eval_out/seq_stage3_night3/stats.json`, from
`experiments/configs/seq_stage3_night3_v1.yaml`, or from the runs'
`summary.yaml` / `metrics.json` under
`models/embeddings/seq_stage3/night3/runs/<config>/seed_<s>/`.

The prose is authored text in this generator. The only value that moves on a
re-render is the declared render timestamp at the top of the report.

Usage:
    uv run --no-sync python scripts/sequence_track/render_night3_report.py
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
STATS_PATH = REPO / "eval_out/seq_stage3_night3/stats.json"
CONFIG_PATH = REPO / "experiments/configs/seq_stage3_night3_v1.yaml"
RUNS_ROOT = REPO / "models/embeddings/seq_stage3/night3/runs"
OUT_PATH = REPO / "research/reports/embeddings/SEQ_STAGE3A_NIGHT3_REPORT.md"

CONFIG_ORDER = [
    "mlp_pool",
    "mlp_pool_tiercond",
    "mlp_P_tiercond",
    "mlp_rowmatch_P_tiercond",
    "mlp_E_tiercond",
    "mlp_rowmatch_E_tiercond",
]
SEEDS = [7, 13, 29, 42, 101]
FAMILY_ORDER = ["family_3a_cond", "family_3a_P", "family_3a_E"]

# Authored: the exploratory (no-family) rows section 6 must carry.
EXPLORATORY_ROWS = [
    "mlp_P_tiercond-mlp_pool@all",
    "mlp_P_tiercond-mlp_pool@target_E",
    "mlp_P_tiercond-mlp_pool@big3",
    "mlp_E_tiercond-mlp_pool@all",
    "mlp_E_tiercond-mlp_pool@target_P",
    "mlp_E_tiercond-mlp_pool@big3",
    "mlp_rowmatch_P_tiercond-mlp_pool@all",
    "mlp_rowmatch_P_tiercond-mlp_pool@target_P",
    "mlp_rowmatch_E_tiercond-mlp_pool@all",
    "mlp_rowmatch_E_tiercond-mlp_pool@target_E",
    "mlp_pool_tiercond-mlp_pool@target_P",
    "mlp_pool_tiercond-mlp_pool@target_E",
    "mlp_pool_tiercond-mlp_pool@big3",
]


# ---------------------------------------------------------------- formatting


def d5(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:+.5f}"


def d4(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:+.4f}"


def ci5(ci) -> str:
    if not ci:
        return "n/a"
    return f"[{ci[0]:+.5f}, {ci[1]:+.5f}]"


def ci4(ci) -> str:
    if not ci:
        return "n/a"
    return f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"


def ll6(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.6f}"


def integer(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{int(x):,}"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bullets(items) -> str:
    out = []
    for it in items:
        if isinstance(it, dict):
            text = " ".join(str(it.get("statement") or it.get("text") or "").split())
            reason = " ".join(str(it.get("reason") or "").split())
            head = f"**`{it['id']}`** — {text}" if "id" in it else text
            if reason:
                head += f" *Reason as registered:* {reason}"
            out.append(f"* {head}")
        else:
            out.append(f"* {' '.join(str(it).split())}")
    return "\n".join(out)


# ------------------------------------------------------------------- loaders


def load_run_metrics():
    """metrics.json per (config, seed), read from the run directories."""
    out = {}
    for cfg in CONFIG_ORDER:
        for seed in SEEDS:
            p = RUNS_ROOT / cfg / f"seed_{seed}" / "metrics.json"
            out[(cfg, seed)] = json.loads(p.read_text()) if p.exists() else None
    return out


def load_summaries():
    out = {}
    for cfg in CONFIG_ORDER:
        p = RUNS_ROOT / cfg / "summary.yaml"
        out[cfg] = yaml.safe_load(p.read_text()) if p.exists() else None
    return out


# ------------------------------------------------------------------ sections


def section_1(stats, render_ts) -> str:
    cohort = stats["cohort"]
    return f"""# Sequence track — stage 3a night-3 report (negative-transfer screen, six configurations, five seeds, validation only)

Generated (declared timestamp, the only value that moves on a re-render): **{render_ts}** by `scripts/sequence_track/render_night3_report.py` from the files of record. Every number in every table is read from a file; the narrative passages are authored text in the generator.

* statistics: `eval_out/seq_stage3_night3/stats.json` (sha256 `{sha256_of(STATS_PATH)[:12]}…`), written {stats['generated_at_utc']} by `{stats['tool']}`
* config: `experiments/configs/seq_stage3_night3_v1.yaml` (sha256 `{sha256_of(CONFIG_PATH)[:12]}…`)
* runs: `models/embeddings/seq_stage3/night3/runs/<config>/seed_<s>/` — `summary.yaml`, `metrics.json`
* frozen commit `90acca8`; this render's launch state `65f2f63`

**Status: validation-only five-seed, outcome-informed screen. `cohort_status: {cohort['cohort_status']}`, `cohort_scored: {str(cohort['cohort_scored']).lower()}`, `advances: {cohort['advances']}`, `provisional: {str(cohort['provisional']).lower()}`. No arm advances. No market claim. No LANDED verdict — this evidence is provisional (validation-only, checkpoint-selected on the same split it is scored on, with the untouched cohort unopened), and provisional evidence can never be LANDED (invariant 9). The verdict is the user's decision and none was logged: `research/log_verdict.py` was not called.**

Registered evidence status, verbatim from the statistics file: {stats['evidence_status']}

Statuses this stage can emit: `SCREEN_PASS`, `SCREEN_NOT_PASS`, `NOT_EVALUABLE`. There is no advancement status."""


def section_2(stats, cfg_yaml, metrics) -> str:
    purpose = " ".join(cfg_yaml["experiment"]["purpose"].split())
    by_id = {c["id"]: c for c in cfg_yaml["configurations"]}
    tr = cfg_yaml["training"]
    sb = tr["step_budget"]
    rows = []
    n_conditioned = 0
    for cid in CONFIG_ORDER:
        c = by_id[cid]
        m = metrics[(cid, 7)]
        train_rows = m["training_schedule"]["train_rows_selected"] if m else None
        params = c.get("params", {})
        tier_dim = params.get("tier_embed")
        if tier_dim:
            n_conditioned += 1
            cond = f"tier embed dim {tier_dim} + per-tier output bias"
        else:
            cond = "none"
        refs = c.get("reference") or []
        role = c.get("role", "")
        role_cell = role if not refs else f"{role}; reference(s) {', '.join('`' + r + '`' for r in refs)}"
        rows.append(
            f"| `{cid}` | {integer(train_rows)} | {cond} | {role_cell} |"
        )
    return """## 1. The question

Does pooling every tier of T20 cricket into one training set **hurt the tier you actually care about**? That is the negative-transfer question, and it is the only question this block asks. If pooling hurts, a model trained on the target tier alone should beat the pooled model **on the target tier's own validation rows**; if pooling helps, it should not.

Two targets are registered. **P** is the premium-league target, `competition_tier == 3`. **E** is P together with every international in which both teams are ICC full members. `competition_tier` is a pre-match event-name rule, not a market or quality measurement: "premium" and "elite" are the rule's labels, not findings.

Registered purpose, verbatim from the config: """ + purpose + """

Sign convention: **candidate minus reference, in validation log loss; negative favourable.** Every arm here is `token_mlp` — no attention layer anywhere in this block — so nothing here bears on any sequence mechanism.

| config id | training rows selected | tier conditioning | role |
|---|---|---|---|
""" + "\n".join(rows) + f"""

Training rows are read from each run's `metrics.json` `training_schedule.train_rows_selected` (seed 7; identical across seeds within a config). {n_conditioned} of the {len(CONFIG_ORDER)} configurations are tier-conditioned; only `mlp_pool` is not.

Every arm trains under one step budget, read from the config's `training` block: {sb['max_steps']:,} steps at `eval_every` {sb['eval_every']}, `d_model` {tr['dmodel']}, {tr['layers']} layers, {tr['heads']} heads, batch {tr['batch']}, lr {tr['learning_rate']}, device `{tr['device']}`, on frame `{cfg_yaml['data']['directory']}` with cache role `{cfg_yaml['data']['stats_cache_role']}`. Validation is **never** filtered: {" ".join(tr['selection_rule'].split())}"""


def section_3(stats) -> str:
    ct = stats["contract"]
    fams = {f["holm_group"]: f for f in stats["families"]}
    m_cond = len(fams["family_3a_cond"]["members"])
    m_target = len(fams["family_3a_P"]["members"])
    assert m_target == len(fams["family_3a_E"]["members"])
    return f"""## 2. The decision rule, as registered before any result was read

Three families, each registered once, each with its own Holm step-down within the family and no correction across families.

* **`family_3a_P`** and **`family_3a_E`** — candidate `mlp_<T>_tiercond`, four members: `superiority_1` = target-only − **pooled-conditioned** on the target slice; `superiority_2` = target-only − **row-matched pooled** on the target slice; `death_gate` and `chase_gate` = target-only − `mlp_pool` on the death and chase rows **restricted to the target**. `SCREEN_PASS` requires **all four** members to reject favourably under Holm — both superiority members CI-clean favourable, both gates strictly `U95 < {ct['margin_ll']:+.3f}` — plus at least ten tournament blocks on each member's slice, five complete paired seeds, and at least 4 of 5 favourable per-seed directions on **both** superiority members. The inherited all-row condition is **amended away** (`all_row_condition: none`): a target-only model is not asked to be better on rows outside its target, so its `all` reading is exploratory and not part of this screen.
* **`family_3a_cond`** — candidate `mlp_pool_tiercond`, three members: `primary` = pooled-conditioned − pooled on `all`, plus the same two non-inferiority gates on `death` and `chase` against `mlp_pool`. `all_row_condition: member` — here the all-row reading *is* the primary, so the condition is discharged inside the family. The 4-of-5 direction requirement applies to `primary`.

Uncertainty: {ct['reps']:,} tournament-time-block bootstrap replicates at rng seed {ct['rng_seed']}, α = {ct['alpha']}, non-inferiority margin {ct['margin_ll']:+.3f} log loss. p-value convention, verbatim: {" ".join(ct['p_convention'].split())}.

**Holm adjustment, in its general form:** `p_holm(r) = min(1, max over j <= r of (m − j + 1) · p_(j))` with the step-down stopping rule, where **m is that family's own member count** — m = {m_target} for `family_3a_P` and `family_3a_E`, m = {m_cond} for `family_3a_cond` — applied **within each family only**, never pooled across families. A slice with fewer than 10 blocks is **descriptive** and its family is `NOT_EVALUABLE`. Seeds {ct['seeds']} — five complete paired seeds are required.

Two estimands are reported. **(i)** is one seed checkpoint with paired block-only uncertainty. **(ii)** is the arithmetic across-seed mean under joint seed-and-block resampling — a descriptive five-seed robustness screen, **not** the log loss of averaged probabilities and **not** uncertainty for a newly trained single checkpoint."""


def section_4(stats) -> str:
    stats_by_slice = stats["slices"]["stats"]
    preds = {p["slice"]: p for p in stats["slices"]["predicates"]}
    order = [
        "all", "death", "chase", "powerplay", "middle", "innings_1", "innings_2",
        "target_P", "target_E", "big3",
        "death@target_P", "chase@target_P", "death@target_E", "chase@target_E",
    ]
    rows = []
    for name in order:
        s = stats_by_slice.get(name)
        if not s or not s.get("available"):
            continue
        pred = " ".join(str(preds.get(name, {}).get("predicate") or "").split())
        rows.append(
            f"| `{name}` | {integer(s['n_rows'])} | {integer(s['n_matches'])} | "
            f"{s['n_blocks']} | {'yes' if s.get('descriptive') else 'no'} | {pred} |"
        )
    tp = stats_by_slice["target_P"]
    te = stats_by_slice["target_E"]
    b3 = stats_by_slice["big3"]
    blocks = stats["blocks"]
    unavailable = [
        p for p in stats["slices"]["predicates"] if not p.get("available")
    ]
    unav = ""
    if unavailable:
        u = unavailable[0]
        unav = (
            f"\n\n`{u['slice']}` is reported **unavailable**, not invented: "
            + " ".join(str(u.get("unavailable_reason", "")).split())
        )
    return f"""## 3. Slices

Block lookup `{blocks['lookup_source']}`: {blocks['n_matches']} validation matches map to {blocks['n_blocks']} tournament blocks, {blocks['unmapped']} unmapped. Slice identity is compared by a row-membership digest, never by matching row, match and block counts.

| slice | rows | matches | blocks | descriptive (<10 blocks) | predicate |
|---|---|---|---|---|---|
{chr(10).join(rows)}

**The decisive line of this table:** `target_P` carries {integer(tp['n_rows'])} rows over {integer(tp['n_matches'])} matches but only **{tp['n_blocks']} tournament blocks** — the premium-league validation matches come from very few distinct events — so by the registered ten-block rule every member of `family_3a_P` is **descriptive** and the family is **`NOT_EVALUABLE`**. `target_E` reaches **{te['n_blocks']} blocks** and is evaluable. `big3` has {b3['n_blocks']} blocks over {integer(b3['n_matches'])} matches and {integer(b3['n_rows'])} rows: it is a registered descriptive readout only, reported because it was designed in, not because it can support an inference.{unav}"""


def section_5(stats) -> str:
    fams = {f["holm_group"]: f for f in stats["families"]}
    out = ["""## 4. Family results

Member tables are the **estimand (ii)** readout, `seed_mean_joint`. The per-seed line under each table gives the same family's status at each of the five registered seeds under estimand (i). `p (Holm)` is the step-down adjusted p within that family only."""]
    for key in FAMILY_ORDER:
        f = fams[key]
        joint = f["holm"]["seed_mean_joint"]
        screen = f["screen"]["seed_mean_joint"]
        rows = []
        for m in joint["members"]:
            ph = m.get("p_holm")
            rows.append(
                f"| `{m['member']}` | {m['contrast']} | `{m['slice']}` | "
                f"{d5(m.get('point'))} | {ci5(m.get('ci95'))} | {d5(m.get('u95'))} | "
                f"{m.get('p_display', 'n/a')} | {f'{ph:.4f}' if ph is not None else 'n/a'} | "
                f"{m.get('rank', 'n/a')} | {str(m.get('rejected')).lower()} | {m.get('status')} |"
            )
        per_seed = ", ".join(
            f"seed {s} `{f['screen'][f'seed_{s}']['status']}`" for s in SEEDS
        )
        dirs = screen.get("direction_requirement_by_member", {})
        dir_rows = [
            f"| `{name}` | `{v['contrast_key']}` | {v['n_seeds']} | "
            f"{v['favourable_direction_count']}/{v['n_seeds']} | "
            f"{screen['required_favourable_directions']}/{screen['n_seeds']} | "
            f"{str(v['favourable_direction_requirement_met']).lower()} |"
            for name, v in dirs.items()
        ]
        out.append(f"""### `{key}` — candidate `{f['candidate']}` — **{screen['status']}**

| member | contrast | slice | point | 95% CI | U95 | p (raw) | p (Holm) | rank | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

Per-seed family status (estimand (i)): {per_seed}.

Screen: required members {', '.join('`' + x + '`' for x in screen['screen_required_members'])}; member statuses {', '.join(f'`{k}` {v}' for k, v in screen['screen_required_member_statuses'].items())}. All direction requirements met: **{str(screen['all_direction_requirements_met']).lower()}**. {" ".join(screen['five_seed_qualification_note'].split())}

| direction-required member | contrast | seeds | favourable directions | required | met |
|---|---|---|---|---|---|
{chr(10).join(dir_rows)}""")
    return "\n\n".join(out)


def section_6(stats) -> str:
    fams = {f["holm_group"]: f for f in stats["families"]}
    keys = []
    for fk in FAMILY_ORDER:
        for m in fams[fk]["holm"]["seed_mean_joint"]["members"]:
            if m.get("contrast_key") and m["contrast_key"] not in keys:
                keys.append(m["contrast_key"])
    family_n = len(keys)
    for k in EXPLORATORY_ROWS:
        if k in stats["contrasts"] and k not in keys:
            keys.append(k)
    rows = []
    for i, k in enumerate(keys):
        c = stats["contrasts"][k]
        pts = c["per_seed_points"]
        ii = c["estimand_ii"]
        cells = " | ".join(d5(pts.get(str(s))) for s in SEEDS)
        if i == family_n:
            rows.append(
                "| *— exploratory rows below: no family, no screen —* | | | | | | | | | | | |"
            )
        rows.append(
            f"| `{c['candidate']} − {c['reference']}` | `{c['slice']}` | {cells} | "
            f"{c['seed_spread']['range']:+.5f} | "
            f"{c.get('seeds_below_zero_count', c['favourable_direction_count'])}/{c['n_seeds']} | {d5(ii['point'])} | "
            f"{ci5(ii['ci95'])} | {c['slice_stats']['n_blocks']} |"
        )
    heads = " | ".join(f"seed {s}" for s in SEEDS)
    return f"""## 5. Seed spread, every family member and the exploratory readouts

| contrast | slice | {heads} | seed range | favourable directions | mean point (ii) | mean 95% interval (ii) | blocks |
|---|---|{'---|' * 5}---|---|---|---|---|
{chr(10).join(rows)}

Seed ranges are empirical spreads over five seeds, not confidence intervals. No seed is selected and no CI endpoint is averaged. The favourable-direction count is the **zero-threshold** count (per-seed points below 0), which on a non-inferiority gate row is not the same as the count below that row's registered `+0.002` margin. Every row below the divider is exploratory: no family names it, no screen reads it, and `big3` rows in particular sit on {stats['slices']['stats']['big3']['n_blocks']} blocks."""


def section_7(stats, summaries, metrics, cfg_yaml) -> str:
    sb = cfg_yaml["training"]["step_budget"]
    total_steps = sb["max_steps"]
    n_cfg = len(CONFIG_ORDER)
    residual = " ".join(sb["residual_confounding"].split())
    ll_rows = []
    for cfg in CONFIG_ORDER:
        per_seed = summaries[cfg]["splits"]["validation"]["per_seed"]
        lls = [r["ll"] for r in per_seed]
        mean = sum(lls) / len(lls)
        best = min(per_seed, key=lambda r: r["ll"])
        worst = max(per_seed, key=lambda r: r["ll"])
        ll_rows.append(
            f"| `{cfg}` | {len(lls)} | {ll6(mean)} | {ll6(best['ll'])} (seed {best['seed']}) | "
            f"{ll6(worst['ll'])} (seed {worst['seed']}) |"
        )
    exp_rows = []
    for cfg in CONFIG_ORDER:
        for seed in SEEDS:
            m = metrics[(cfg, seed)]
            ts = m["training_schedule"]
            by_list = ts.get("tokens_seen_by_list") or {}
            untrained = ts.get("untrained_tiers")
            untrained_s = ", ".join(str(t) for t in untrained) if untrained else "none"
            exp_rows.append(
                f"| `{cfg}` | {seed} | {integer(ts.get('tokens_seen'))} | "
                f"{integer(by_list.get('target_P_matches.json'))} | "
                f"{integer(by_list.get('target_E_matches.json'))} | "
                f"{ts.get('steps_to_best')} / {ts.get('total_steps')} | "
                f"{ts.get('best_epoch')} | {untrained_s} |"
            )
    return f"""## 6. Absolute validation log loss, and what each arm actually saw

Numbers of record: full-precision per-seed validation log loss from `runs/<config>/summary.yaml`; never rounded report values or LL reconstructed from saved probabilities.

| config | seeds | mean val LL | min | max |
|---|---|---|---|---|
{chr(10).join(ll_rows)}

Exposure, per run, from each `metrics.json` `training_schedule`. **These are full-training totals — tokens seen over the whole {total_steps:,}-step budget, not exposure at the selected checkpoint**, which for most arms is reached earlier (`steps to best` below). `tokens seen in P` / `tokens seen in E` are the tokens drawn from matches on the two frozen report lists. `untrained tiers` are tier codes present in validation but absent from that arm's training rows, recorded verbatim from `training_schedule.untrained_tiers`; their embedding and bias slots stay at initialisation and are still exercised at validation time.

| config | seed | tokens seen | tokens seen in P | tokens seen in E | steps to best / total | best epoch | untrained tiers |
|---|---|---|---|---|---|---|---|
{chr(10).join(exp_rows)}

All {n_cfg} arms run exactly {total_steps:,} steps, so the arms trained on fewer rows see **more epochs** over their smaller sets. Registered residual confounding, verbatim: {residual}"""


def section_8(stats, metrics) -> str:
    c = stats["contrasts"]

    def ii(key):
        return c[key]["estimand_ii"]

    p1 = ii("mlp_P_tiercond-mlp_pool_tiercond@target_P")
    p2 = ii("mlp_P_tiercond-mlp_rowmatch_P_tiercond@target_P")
    e1 = ii("mlp_E_tiercond-mlp_pool_tiercond@target_E")
    e2 = ii("mlp_E_tiercond-mlp_rowmatch_E_tiercond@target_E")
    pall = ii("mlp_P_tiercond-mlp_pool_tiercond@all")
    eall = ii("mlp_E_tiercond-mlp_pool_tiercond@all")
    cond = ii("mlp_pool_tiercond-mlp_pool@all")
    fams = {f["holm_group"]: f for f in stats["families"]}
    e_dirs = fams["family_3a_E"]["screen"]["seed_mean_joint"]["direction_requirement_by_member"]
    p_blocks = c["mlp_P_tiercond-mlp_pool_tiercond@target_P"]["slice_stats"]["n_blocks"]
    e_blocks = c["mlp_E_tiercond-mlp_pool_tiercond@target_E"]["slice_stats"]["n_blocks"]

    pool_p_tok = metrics[("mlp_pool", 7)]["training_schedule"]["tokens_seen_by_list"]["target_P_matches.json"]
    tgt_p_tok = metrics[("mlp_P_tiercond", 7)]["training_schedule"]["tokens_seen_by_list"]["target_P_matches.json"]
    pool_e_tok = metrics[("mlp_pool", 7)]["training_schedule"]["tokens_seen_by_list"]["target_E_matches.json"]
    tgt_e_tok = metrics[("mlp_E_tiercond", 7)]["training_schedule"]["tokens_seen_by_list"]["target_E_matches.json"]
    ratio_p = tgt_p_tok / pool_p_tok
    ratio_e = tgt_e_tok / pool_e_tok

    pseed = c["mlp_P_tiercond-mlp_pool_tiercond@target_P"]["per_seed_points"]
    eseed = c["mlp_E_tiercond-mlp_pool_tiercond@target_E"]["per_seed_points"]

    def worst_two(d):
        ranked = sorted(d.items(), key=lambda kv: -float(kv[1]))[:2]
        return " and ".join(f"seed {k} ({d5(float(v))})" for k, v in ranked)

    p_worst = worst_two(pseed)
    e_worst = worst_two(eseed)
    total_steps = metrics[("mlp_pool", 7)]["training_schedule"]["total_steps"]

    return f"""## 7. In plain language

**What was tested.** Six small models were trained on historical T20 deliveries and then asked, one ball at a time, to put a probability on each of the six outcomes of the *next* delivery of {integer(stats['slices']['stats']['all']['n_rows'])} held-out balls. They differ in **which matches they were allowed to learn from**, and in whether they are told which tier a match belongs to. This is teacher-forced ball prediction: nobody simulated a match, no market price appears anywhere in this block, and a better log loss here implies nothing about a score distribution, a win probability or a betting edge.

**The registered reading, in one paragraph.** Target-only point estimates are **adverse** on their own targets. `family_3a_P` is **descriptive and `NOT_EVALUABLE`** — {p_blocks} tournament blocks against the ten the rule requires. `family_3a_E` and `family_3a_cond` **do not pass their registered screens**. **Negative transfer was not detected; neither its absence nor positive transfer is established.**

**The target-side numbers.** Training only on premium leagues moves the point estimate on the premium-league validation rows by {d4(p1['point'])} log loss against the pooled conditioned model ({ci4(p1['ci95'])}, descriptive on {p_blocks} blocks), and by {d4(p2['point'])} {ci4(p2['ci95'])} against the row-matched pooled control. On the elite target E the same two readings are {d4(e1['point'])} {ci4(e1['ci95'])} on {e_blocks} blocks, with {e_dirs['superiority_1']['favourable_direction_count']}/5 seeds in the favourable direction against the 4/5 the screen requires, and {d4(e2['point'])} {ci4(e2['ci95'])} against the row-matched control. Every point estimate is adverse, and no superiority member of either family produced a favourable interval.

**Training exposure.** Over the full {total_steps:,}-step budget `mlp_P_tiercond` drew {integer(tgt_p_tok)} tokens from premium-league matches against the pooled control's {integer(pool_p_tok)} — **{ratio_p:.1f}×** — and `mlp_E_tiercond` drew {ratio_e:.1f}× the pooled arm's E tokens. These are full-training totals, not exposure at the selected checkpoint. More target exposure did not come with a favourable target-side estimate.

**On all validation rows.** {d4(pall['point'])} {ci4(pall['ci95'])} for the P arm and {d4(eall['point'])} {ci4(eall['ci95'])} for the E arm against the same conditioned pooled model. That reading is exploratory — the screens amend the all-row condition away — and the untrained tier slots those arms carry into rows they never trained on are **one possible contributor**, not an established cause.

**Tier conditioning on its own.** `mlp_pool_tiercond − mlp_pool` on all rows is {d4(cond['point'])} {ci4(cond['ci95'])}, with {fams['family_3a_cond']['screen']['seed_mean_joint']['favourable_direction_count']}/5 seeds in the favourable direction against the 4/5 required, so `family_3a_cond` does not pass its screen. The interval straddles zero: this is **unresolved evidence at this resolution**, not a demonstration that conditioning is inert.

**What is not established.** "Not detected" is not proof of absence, and an adverse point estimate on a screen that does not pass is not a demonstration of positive transfer. The P family cannot carry an inference at all on {p_blocks} blocks; on target E the superiority intervals straddle or sit near zero and read unresolved. Nothing here establishes that the pooled model is better on the target, only that restricting training to the target was **not shown** to help.

**Two possible explanations for the adverse direction, neither established.** (a) The per-seed points are not uniform: on target P they run from {d5(min(float(v) for v in pseed.values()))} to {d5(max(float(v) for v in pseed.values()))}, the two most adverse being {p_worst}; on target E from {d5(min(float(v) for v in eseed.values()))} to {d5(max(float(v) for v in eseed.values()))}, the two most adverse being {e_worst}. That pattern **would be consistent with** overfitting under the equal-step budget, since an arm trained on a fraction of the rows takes roughly proportionally more epochs over them — but no overfitting diagnostic was run and this is a hypothesis. (b) Checkpoints for **every** arm, target-only arms included, were selected on **all** validation rows, which can work against target-only performance by construction; that cost is a registered part of the shared-artifact contract. Both caveats stand, and an **equal-epoch or early-stopped replication is the registered follow-up** that would separate them from the transfer question."""


def section_9(stats, cfg_yaml, metrics) -> str:
    p_untrained = metrics[("mlp_P_tiercond", 7)]["training_schedule"]["untrained_tiers"]
    prov_rows = []
    for cfg in CONFIG_ORDER:
        for s in stats["runs"][cfg]["seeds"]:
            p = s.get("provenance") or {}
            prov_rows.append((s["seed"], p.get("machine"), p.get("chip"), p.get("hostname")))
    by_seed = {}
    for seed, machine, chip, host in prov_rows:
        by_seed.setdefault(seed, set()).add((machine, chip, host))
    mach_rows = [
        f"| {seed} | {', '.join(sorted(m or 'unrecorded' for m, _, _ in v))} | "
        f"{', '.join(sorted(ch or 'unrecorded' for _, ch, _ in v))} | "
        f"{', '.join(sorted(h or 'unrecorded' for _, _, h in v))} |"
        for seed, v in sorted(by_seed.items())
    ]
    b3 = stats["slices"]["stats"]["big3"]
    return f"""## 8. Registered deviations, asymmetries and limitations (restated, none dropped)

### Deviations (config `deviations`)

{bullets(cfg_yaml['deviations'])}

### Known asymmetries (config `known_asymmetries`)

{bullets(cfg_yaml['known_asymmetries'])}

**Correction to config-sourced wording above.** `untrained_tier_slots_in_the_target_arms` says "tiers 0, 1, 2 and 4" for `mlp_P_tiercond`. The value actually recorded by every seed of that run is `training_schedule.untrained_tiers` = {p_untrained}; tier 0 is not among them. The same config text says "the three conditioned arms"; five of the six configurations are tier-conditioned (every arm except `mlp_pool`), and the asymmetry it describes applies to all five. The asymmetry itself is unchanged; only the tier list and the arm count are corrected, and the table in § 6 prints the recorded values.

### Known limitations (config `known_limitations`)

{bullets(cfg_yaml['known_limitations'])}

### Restated in this report's own terms

* **Seed–machine confounding.** Machine is recorded per run and is confounded with seed. Read from the run provenance:

| seed | machine | chip | host |
|---|---|---|---|
{chr(10).join(mach_rows)}

  The split is **by seed**, so every arm carries the same mix and no arm is advantaged relative to another; every registered contrast is within-machine at each seed, so an additive machine effect cancels inside it. Arm-by-machine interaction remains inseparable from seed variation and **no machine term is fitted**.
* **The P and E screens are correlated, not replications.** Target E contains all of target P, the two validation slices overlap heavily, and the two families share `mlp_pool` and `mlp_pool_tiercond`. Claims are **family-local only**: no aggregate, no "either target passes", no comparison of the two families' outcomes to each other.
* **`big3` is too thin to read.** {b3['n_blocks']} blocks, {integer(b3['n_matches'])} matches, {integer(b3['n_rows'])} rows. It is reported because it was designed in.
* **Selection optimism throughout.** Checkpoint selection used the same validation split every contrast is computed on."""


def section_10(stats) -> str:
    cohort = stats["cohort"]
    n_cfg = len(CONFIG_ORDER)
    return f"""## 9. What happens next

**Nothing advances.** `advances: {cohort['advances']}`, `cohort_status: {cohort['cohort_status']}`, `reads_performed: {cohort['reads_performed']}`. No cohort feature, prediction or base-logit read happened, no artifact of record changed, no test split or market price was touched, and no verdict was logged — that call is the user's.

The next separately frozen batch is **Stage 4** (rungs 4d and 4b) and **C114**. Each gets its own config, its own freeze and its own acceptance file; nothing from tonight is carried into them as an established result.

Stage 3a's own follow-up is **backlog, not queued**: an **equal-epoch or early-stopped replication** of the {n_cfg} arms, the design change that would separate the transfer question from the equal-step budget and from all-validation checkpoint selection. Until that runs, the registered reading stands exactly as stated and no further: **target-only point estimates are adverse; `family_3a_P` is descriptive and `NOT_EVALUABLE` on the block rule; `family_3a_E` and `family_3a_cond` do not pass their registered screens; negative transfer was not detected, and neither its absence nor positive transfer is established.**"""


def main() -> None:
    stats = json.loads(STATS_PATH.read_text())
    cfg_yaml = yaml.safe_load(CONFIG_PATH.read_text())
    summaries = load_summaries()
    metrics = load_run_metrics()
    render_ts = datetime.now(timezone.utc).isoformat()

    parts = [
        section_1(stats, render_ts),
        section_2(stats, cfg_yaml, metrics),
        section_3(stats),
        section_4(stats),
        section_5(stats),
        section_6(stats),
        section_7(stats, summaries, metrics, cfg_yaml),
        section_8(stats, metrics),
        section_9(stats, cfg_yaml, metrics),
        section_10(stats),
    ]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n\n".join(parts) + "\n")
    print(f"wrote {OUT_PATH.relative_to(REPO)} ({len(OUT_PATH.read_text().splitlines())} lines)")


if __name__ == "__main__":
    main()
