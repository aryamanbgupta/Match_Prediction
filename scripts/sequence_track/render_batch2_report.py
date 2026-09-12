# manifest-exempt: embeddings-ladder experimental artifact namespace
"""Render the stage-3 BATCH 2 report (rungs 4d and 4b, block C114) from the
files of record.

Every number in every table is read from

* `eval_out/seq_stage3_batch2/4d_stats.json` and
  `eval_out/seq_stage3_batch2/c114_stats.json`
  (same schema as `eval_out/seq_stage3_night3/stats.json`),
* `experiments/configs/seq_stage3_batch2_4d_v1.yaml`,
  `experiments/configs/seq_stage3_batch2_c114_v1.yaml` and
  `experiments/configs/seq_stage3_batch2_4b_v1.yaml`,
* the runs' `summary.yaml` / `metrics.json` / `run_record.json` under
  `models/embeddings/seq_stage3/batch2/runs/<config>/seed_<s>/`,
* `models/embeddings/stage4/refs/references.json` for the three frozen
  reference validation log losses, and
* `docs/sequence_track/batch2_acceptance.md` for the B6 smoke row.

The prose is authored text in this generator. The only value that moves on a
re-render is the declared render timestamp at the top of the report.

Usage:
    uv run --no-sync python scripts/sequence_track/render_batch2_report.py
    uv run --no-sync python scripts/sequence_track/render_batch2_report.py \
        --stats-4d eval_out/seq_stage3_night3/stats.json \
        --stats-c114 eval_out/seq_stage3_night3/stats.json \
        --out /tmp/smoke.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]

STATS_4D = REPO / "eval_out/seq_stage3_batch2/4d_stats.json"
STATS_C114 = REPO / "eval_out/seq_stage3_batch2/c114_stats.json"
CONFIG_4D = REPO / "experiments/configs/seq_stage3_batch2_4d_v1.yaml"
CONFIG_C114 = REPO / "experiments/configs/seq_stage3_batch2_c114_v1.yaml"
CONFIG_4B = REPO / "experiments/configs/seq_stage3_batch2_4b_v1.yaml"
RUNS_ROOT = REPO / "models/embeddings/seq_stage3/batch2/runs"
REFS_PATH = REPO / "models/embeddings/stage4/refs/references.json"
ACCEPTANCE = REPO / "docs/sequence_track/batch2_acceptance.md"
OUT_PATH = REPO / "research/reports/embeddings/SEQ_STAGE3_BATCH2_REPORT.md"

SEEDS = [7, 13, 29, 42, 101]
FREEZE_COMMIT = "86e049e"
LAUNCH_RECORD_COMMIT = "a003d5d"

# The three frozen stage 4 references, in the report's display order:
# (display id used by both configs, references.json key).
REFERENCE_KEYS = [
    ("ref_eb_ctx", "eb_ctx"),
    ("ref_raw_ctx", "raw_rate_ctx"),
    ("ref_lin_50", "ref_lin_50"),
]

# The two rung-4b configurations, whose family is deferred (§ 6).
FOURB_CONFIGS = ["identity_residual_l3", "identity_residual_l2"]

# Stage 1's rollout winner-log-loss C-B reading is READ from the stage 1
# report's § 4 Holm table, never hard-coded here, and is labelled in this
# report as a DIFFERENT measurement from anything in this batch.
STAGE1_REPORT = REPO / "research/reports/embeddings/SEQ_STAGE1_REPORT.md"


# ---------------------------------------------------------------- formatting


def d5(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{float(x):+.5f}"


def d4(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{float(x):+.4f}"


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
    return f"{float(x):.6f}"


def ll4(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{float(x):.4f}"


def integer(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{int(x):,}"


def flat(x) -> str:
    return " ".join(str(x).split())


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def short_sha(path: Path) -> str:
    return f"{sha256_of(path)[:12]}…" if path.exists() else "absent"


def bullets(items) -> str:
    out = []
    for it in items or []:
        if isinstance(it, dict):
            text = flat(it.get("statement") or it.get("text") or "")
            reason = flat(it.get("reason") or "")
            head = f"**`{it['id']}`** — {text}" if "id" in it else text
            if reason:
                head += f" *Reason as registered:* {reason}"
            out.append(f"* {head}")
        else:
            out.append(f"* {flat(it)}")
    return "\n".join(out) if out else "* *(none registered)*"


def yesno(x) -> str:
    return str(bool(x)).lower()


# ------------------------------------------------------------------- loaders


def config_ids(cfg_yaml) -> list:
    return [c["id"] for c in cfg_yaml.get("configurations", [])]


def load_summaries(ids) -> dict:
    out = {}
    for cid in ids:
        p = RUNS_ROOT / cid / "summary.yaml"
        out[cid] = yaml.safe_load(p.read_text()) if p.exists() else None
    return out


def load_metrics(ids) -> dict:
    out = {}
    for cid in ids:
        for seed in SEEDS:
            p = RUNS_ROOT / cid / f"seed_{seed}" / "metrics.json"
            out[(cid, seed)] = json.loads(p.read_text()) if p.exists() else None
    return out


def load_run_records(ids) -> dict:
    out = {}
    for cid in ids:
        for seed in SEEDS:
            p = RUNS_ROOT / cid / f"seed_{seed}" / "run_record.json"
            out[(cid, seed)] = json.loads(p.read_text()) if p.exists() else None
    return out


def reference_lls() -> list:
    """(display id, references.json key, validation log loss as recorded)."""
    refs = json.loads(REFS_PATH.read_text())
    out = []
    for display, key in REFERENCE_KEYS:
        block = refs.get(key) or {}
        ll = (block.get("log_loss") or {}).get("validation")
        out.append((display, key, ll, block.get("n_features")))
    return out


def stage1_cb_row() -> dict:
    """The C-B row of the stage 1 report's § 4 Holm table, parsed from the file.

    Returns the point, the rank-local interval, the Holm p and the label
    exactly as that report prints them, plus the file's sha256.
    """
    if not STAGE1_REPORT.exists():
        return {"available": False}
    text = STAGE1_REPORT.read_text()
    section = text.split("## 4.", 1)[-1].split("\n## ", 1)[0]
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 8 and cells[0] == "C-B":
            return {
                "available": True,
                "point": cells[2],
                "interval": cells[3],
                "raw_p": cells[4],
                "holm_p": cells[5],
                "label": cells[7].strip("*"),
                "sha256": sha256_of(STAGE1_REPORT),
            }
    return {"available": False}


def per_seed_rows(summary) -> list:
    if not summary:
        return []
    return ((summary.get("splits") or {}).get("validation") or {}).get("per_seed") or []


def acceptance_b6_row() -> str:
    """The B6 check row, verbatim from the acceptance file's checks table."""
    if not ACCEPTANCE.exists():
        return ""
    for line in ACCEPTANCE.read_text().splitlines():
        if line.startswith("| B6 |"):
            return line.strip()
    return ""


def readout_seed_keys(block) -> list:
    """The per-seed readout keys present in a family's screen/holm block."""
    keys = [k for k in block if k.startswith("seed_") and k != "seed_mean_joint"]

    def order(k):
        try:
            return SEEDS.index(int(k.split("_", 1)[1]))
        except (ValueError, IndexError):
            return 99

    return sorted(keys, key=order)


def families_of(stats) -> list:
    return stats.get("families") or []


def family_members(fam, readout="seed_mean_joint") -> list:
    return ((fam.get("holm") or {}).get(readout) or {}).get("members") or []


def input_cell(cfg_entry, cfg_yaml, metrics) -> str:
    """What this configuration reads, from the config and the run metrics."""
    params = cfg_entry.get("params") or {}
    extra = params.get("extra_features") or {}
    bits = []
    contract = params.get("feature_contract")
    if contract:
        n = None
        for seed in SEEDS:
            m = metrics.get((cfg_entry["id"], seed))
            if m:
                n = (m.get("arm_params") or {}).get("feature_contract_n") or (
                    m.get("training_contract") or {}
                ).get("feature_contract")
                break
        bits.append(f"`{contract}` contract" + (f" ({n} columns)" if isinstance(n, int) else ""))
    else:
        bits.append(f"default {cfg_yaml.get('data', {}).get('n_features')}-feature contract")
    if extra.get("cols"):
        bits.append(f"+ {len(extra['cols'])} sidecar columns")
    if params.get("residual_lambda") is not None:
        bits.append(f"residual_lambda {params['residual_lambda']}")
    return "; ".join(bits)


def arms_table(cfg_yaml, metrics) -> str:
    rows = []
    for c in cfg_yaml.get("configurations", []):
        refs = c.get("reference") or []
        ref_cell = ", ".join(f"`{r}`" for r in refs) if refs else "— (shared control)"
        n_params = None
        for seed in SEEDS:
            m = metrics.get((c["id"], seed))
            if m:
                n_params = m.get("n_params") or (m.get("arm_params") or {}).get("n_parameters")
                break
        rows.append(
            f"| `{c['id']}` | {c.get('arm')} | {c.get('role')} | {c.get('wiring')} | "
            f"{c.get('history_input')} | {input_cell(c, cfg_yaml, metrics)} | "
            f"{integer(n_params)} | {ref_cell} |"
        )
    return (
        "| config id | arm | role | wiring | history input | inputs | parameters | reference(s) |\n"
        "|---|---|---|---|---|---|---|---|\n" + "\n".join(rows)
    )


def schedule_sentence(cfg_yaml) -> str:
    tr = cfg_yaml.get("training", {})
    return (
        f"Schedule, read from the config's `training` block: `epochs` {tr.get('epochs')} "
        f"with `patience` {tr.get('patience')} (live early stopping — batch 2 sets no step "
        f"budget), `d_model` {tr.get('dmodel')}, {tr.get('layers')} layers, "
        f"{tr.get('heads')} heads, batch {tr.get('batch')}, lr {tr.get('learning_rate')}, "
        f"device `{tr.get('device')}`, seeds {tr.get('seeds')}, on frame "
        f"`{cfg_yaml.get('data', {}).get('directory')}` with cache role "
        f"`{cfg_yaml.get('data', {}).get('stats_cache_role')}`. Selection rule, verbatim: "
        f"{flat(tr.get('selection_rule'))}"
    )


def completeness_table(cfg_label_ids) -> str:
    rows = []
    for label, cid, summary in cfg_label_ids:
        if not summary:
            rows.append(f"| {label} | `{cid}` | n/a | n/a | summary.yaml absent |")
            continue
        exp = summary.get("experiment", {})
        seeds_present = [r.get("seed") for r in per_seed_rows(summary)]
        rows.append(
            f"| {label} | `{cid}` | {exp.get('runs_recorded')} / {exp.get('runs_expected')} | "
            f"{yesno(exp.get('complete'))} | seeds recorded {seeds_present} |"
        )
    return (
        "| rung | config | runs recorded / expected | complete | note |\n"
        "|---|---|---|---|---|\n" + "\n".join(rows)
    )


# ------------------------------------------------------------------ sections


def section_header(stats_4d, stats_c114, paths, render_ts) -> str:
    coh_4d = stats_4d.get("cohort", {})
    coh_c114 = stats_c114.get("cohort", {})
    refs = reference_lls()
    ref_line = ", ".join(f"`{d}` {ll4(ll)}" for d, _k, ll, _n in refs)
    return f"""# Sequence track — stage 3 BATCH 2 report (rung 4d, block C114, rung 4b; validation only)

Generated (declared timestamp, the only value that moves on a re-render): **{render_ts}** by `scripts/sequence_track/render_batch2_report.py` from the files of record. Every number in every table is read from a file; the narrative passages are authored text in the generator.

* statistics, rung 4d: `{paths['stats_4d'].relative_to(REPO)}` (sha256 `{short_sha(paths['stats_4d'])}`), written {stats_4d.get('generated_at_utc')} by `{stats_4d.get('tool')}`
* statistics, block C114: `{paths['stats_c114'].relative_to(REPO)}` (sha256 `{short_sha(paths['stats_c114'])}`), written {stats_c114.get('generated_at_utc')} by `{stats_c114.get('tool')}`
* configs: `{CONFIG_4D.relative_to(REPO)}` (sha256 `{short_sha(CONFIG_4D)}`), `{CONFIG_C114.relative_to(REPO)}` (sha256 `{short_sha(CONFIG_C114)}`), `{CONFIG_4B.relative_to(REPO)}` (sha256 `{short_sha(CONFIG_4B)}`)
* runs: `models/embeddings/seq_stage3/batch2/runs/<config>/seed_<s>/` — `summary.yaml`, `metrics.json`, `run_record.json`
* frozen references: `{REFS_PATH.relative_to(REPO)}` (sha256 `{short_sha(REFS_PATH)}`) — validation log loss {ref_line}
* freeze commit `{FREEZE_COMMIT}`; launch record commit `{LAUNCH_RECORD_COMMIT}`

**Status: validation-only, five-seed, outcome-informed screening. Rung 4d — `cohort_status: {coh_4d.get('cohort_status')}`, `cohort_scored: {yesno(coh_4d.get('cohort_scored'))}`, `advances: {coh_4d.get('advances')}`, `provisional: {yesno(coh_4d.get('provisional'))}`. Block C114 — `cohort_status: {coh_c114.get('cohort_status')}`, `cohort_scored: {yesno(coh_c114.get('cohort_scored'))}`, `advances: {coh_c114.get('advances')}`, `provisional: {yesno(coh_c114.get('provisional'))}`. No arm advances. No market claim and no betting layer anywhere. No LANDED verdict — this evidence is provisional (validation-only, checkpoint-selected on the same split it is scored on, with the untouched cohort unopened), and provisional evidence can never be LANDED (CLAUDE.md invariant 9). The verdict is the user's decision and none was logged: `research/log_verdict.py` was not called.**

Registered evidence status, verbatim from the rung-4d statistics file: {flat(stats_4d.get('evidence_status'))}

Registered evidence status, verbatim from the C114 statistics file: {flat(stats_c114.get('evidence_status'))}

Statuses this batch can emit: `SCREEN_PASS`, `SCREEN_NOT_PASS`, `NOT_EVALUABLE`. There is no advancement status. Rung 4b emits **no status at all** — its family is deferred by registration (§ 6)."""


def section_questions(cfg_4d, cfg_c114, metrics_4d, metrics_c114) -> str:
    refs = reference_lls()
    ref_rows = "\n".join(
        f"| `{d}` | `{k}` | {n if n is not None else 'n/a'} | {ll4(ll)} |"
        for d, k, ll, n in refs
    )
    return f"""## 1. The two questions

Two separately registered questions on two separate sets of arms, in one batch. They share the frame, the schedule and the seeds and nothing else; **no arm of one question may be tabled against an arm of the other**, and no batch-2 arm may be tabled against a night-3 Block B arm (both configs register that deviation — the schedules differ).

Sign convention, verbatim from both configs: {flat(cfg_4d['experiment']['sign_convention'])}

### 1.1 Rung 4d — do as-of exposure counts, posterior spread and recency help a token MLP?

Registered purpose, verbatim: {flat(cfg_4d['experiment']['purpose'])}

The registered contrast is `mlp_spread_recency − mlp_counts`: **both** arms see the two as-of ball counts, so the contrast isolates the twelve Dirichlet spread columns plus the two recency counts and never confounds them with "the model was told how much data there is".

{arms_table(cfg_4d, metrics_4d)}

{schedule_sentence(cfg_4d)}

Both arms are additionally reported **beside the three frozen stage-4 references**, whose validation log losses are read from `{REFS_PATH.relative_to(REPO)}`:

| reference | `references.json` key | features | validation log loss (as recorded, 4 dp) |
|---|---|---|---|
{ref_rows}

That comparison is a **level readout only**: the references are deterministic fitted npz artifacts, not runs, so they carry no seeds, no interval and no Holm slot, and no family member may take one as its reference (`deviations.fixed_reference_is_not_a_member`).

### 1.2 Block C114 — does sequence access buy anything on the production contract, and do the extra 64 columns add on top?

Registered purpose, verbatim: {flat(cfg_c114['experiment']['purpose'])}

Two questions, one set of three arms. **(1)** `full_114 − mlp_114` — the transformer (standard causal attention) against the token MLP on the same 114 columns: sequence access with the inputs held exactly fixed. **(2)** `full_114 − full_50` — the same standard-wiring arm on the production 114-column contract against the 50-feature contract: the contract at fixed wiring. Both families' non-inferiority gates are taken against the shared control `mlp_114`, so every gate asks the same question on `death` and `chase`.

{arms_table(cfg_c114, metrics_c114)}

{schedule_sentence(cfg_c114)}

The C114 arms are read against the same three frozen references as a level readout, on the same terms."""


def decision_rule_block(cfg_yaml, stats, title) -> str:
    st = cfg_yaml["statistics"]
    fams_cfg = st["families"]
    boot = st.get("bootstrap", {})
    noninf = st.get("non_inferiority", {})
    ct = stats.get("contract", {})
    blocks = []
    for entry in fams_cfg.get("map", []):
        members = entry.get("members", [])
        m = len(members)
        rows = "\n".join(
            f"| `{mem['name']}` | {'yes' if mem.get('primary') else 'no'} | {mem.get('kind')} | "
            f"`{mem['contrast']['candidate']} − {mem['contrast']['reference']}` | "
            f"`{mem.get('slice')}` | {mem.get('threshold'):+.3f} |"
            for mem in members
        )
        require = (entry.get("screen") or {}).get("require") or []
        blocks.append(
            f"""**`{entry['holm_group']}`** — registered under family key `candidate: {entry['candidate']}`, `all_row_condition: {entry.get('all_row_condition')}`, {m} members, Holm **m = {m}**.

| member | primary | kind | contrast | slice | threshold |
|---|---|---|---|---|---|
{rows}

`screen.require`: {', '.join('`' + x + '`' for x in require)}. Registered note, verbatim: {flat(entry.get('note'))}"""
        )
    return f"""### {title}

Shared control: `{fams_cfg.get('shared_control')}` — forced to `role: control` and a candidate in no family. Multiplicity: {fams_cfg.get('multiplicity')}, scope {flat(fams_cfg.get('multiplicity_scope'))}.

{chr(10).join(chr(10).join([b, '']) for b in blocks)}
Holm adjustment as registered: {flat((fams_cfg.get('holm') or {}).get('adjustment'))} **m is that family's own member count**, and Holm is never pooled across families. Missing member rule, verbatim: {flat((fams_cfg.get('holm') or {}).get('missing_member'))}

Uncertainty: contract `{boot.get('contract')}`, {boot.get('reps'):,} replicates at rng seed {boot.get('seed')}, {boot.get('weighting')}, blocks from `{boot.get('blocks_from')}` over `{boot.get('block_source_dir')}`, `min_blocks` {boot.get('min_blocks')} and below that {boot.get('below_min_blocks')} (CLAUDE.md invariant 7). Non-inferiority margin {noninf.get('margin_ll'):+.3f} log loss on {noninf.get('applies_to')}, against `{noninf.get('reference')}`; {flat(noninf.get('sign'))}

Raw p convention, verbatim from the statistics file: {flat(ct.get('p_convention')).rstrip('.')}. Seeds {ct.get('seeds')}; {ct.get('n_seeds')} complete paired seeds are required and a family below that is `NOT_EVALUABLE`, never reported at a reduced seed count."""


def section_decision(cfg_4d, cfg_c114, stats_4d, stats_c114) -> str:
    est_4d = "; ".join(
        f"**({e.get('id')})** {e.get('definition')}, reported {e.get('reported')}"
        for e in cfg_4d["statistics"].get("estimands") or []
    )
    return f"""## 2. The decision rule, as registered before any result was read

Read from each config's `statistics` block: the families, their members, their thresholds, each family's `screen.require` and each family's `all_row_condition`.

{decision_rule_block(cfg_4d, stats_4d, "2.1 Rung 4d")}

{decision_rule_block(cfg_c114, stats_c114, "2.2 Block C114")}

Two estimands are reported for every family, as registered ({est_4d}). **(i)** is one seed's checkpoint with paired block-only uncertainty. **(ii)** is the arithmetic across-seed mean under joint seed-and-block resampling — a descriptive five-seed robustness screen, **not** the log loss of averaged probabilities and **not** uncertainty for a newly trained single checkpoint.

**What the intervals printed in this report are.** Every `95% CI` and `U95` column in §§ 3–4 is the member's `ci95` / `u95` field: an **ordinary two-sided 95% percentile interval from that member's own paired bootstrap draws — a marginal interval, one member at a time**. It is *not* a rank-local interval (the statistics file carries those separately as `rank_local_interval` at level `rank_local_level`, and this report does not print them), and it is not a simultaneous interval over the family. **Rejection is governed by the Holm-adjusted p within the family**, shown in the `p (Holm)` and `rejected` columns — never by reading whether a printed marginal interval clears zero or the margin. A marginal interval can exclude zero on a member that Holm does not reject, and that is not a contradiction."""


def family_section(stats, label) -> str:
    out = []
    for fam in families_of(stats):
        key = fam.get("holm_group")
        joint_holm = (fam.get("holm") or {}).get("seed_mean_joint") or {}
        joint_screen = (fam.get("screen") or {}).get("seed_mean_joint") or {}
        members = joint_holm.get("members") or []
        m = joint_holm.get("m", len(members))
        rows = []
        for mem in members:
            ph = mem.get("p_holm")
            rows.append(
                f"| `{mem.get('member')}` | {mem.get('contrast')} | `{mem.get('slice')}` | "
                f"{mem.get('kind')} | {d5(mem.get('point'))} | {ci5(mem.get('ci95'))} | "
                f"{d5(mem.get('u95'))} | {mem.get('p_display', 'n/a')} | "
                f"{f'{ph:.4f}' if isinstance(ph, (int, float)) else 'n/a'} | "
                f"{mem.get('rank', 'n/a')} | {yesno(mem.get('rejected'))} | {mem.get('status')} |"
            )
        seed_keys = readout_seed_keys(fam.get("screen") or {})
        per_seed = ", ".join(
            f"{k.replace('seed_', 'seed ')} `{(fam['screen'][k] or {}).get('status')}`"
            for k in seed_keys
        ) or "*(no per-seed readout in the statistics file)*"
        dirs = joint_screen.get("direction_requirement_by_member", {}) or {}
        dir_rows = "\n".join(
            f"| `{name}` | `{v.get('contrast_key')}` | {v.get('n_seeds')} | "
            f"{v.get('favourable_direction_count')}/{v.get('n_seeds')} | "
            f"{joint_screen.get('required_favourable_directions')}/{joint_screen.get('n_seeds')} | "
            f"{yesno(v.get('favourable_direction_requirement_met'))} |"
            for name, v in dirs.items()
        ) or "| *(no direction-required member recorded)* | | | | | |"
        req = joint_screen.get("screen_required_members") or []
        statuses = joint_screen.get("screen_required_member_statuses") or {}
        out.append(
            f"""### `{key}` — candidate `{fam.get('candidate')}`, shared control `{fam.get('shared_control')}` — **{joint_screen.get('status')}**

Holm within this family only, **m = {m}**. Member table is the **estimand (ii)** readout, `seed_mean_joint`.

| member | contrast | slice | kind | point | 95% CI | U95 | p (raw) | p (Holm) | rank | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(rows) if rows else '| *(no members in the statistics file)* | | | | | | | | | | | |'}

Per-seed family status (estimand (i)): {per_seed}.

Screen: required members {', '.join('`' + x + '`' for x in req) or 'n/a'}; member statuses {', '.join(f'`{k}` {v}' for k, v in statuses.items()) or 'n/a'}. All direction requirements met: **{yesno(joint_screen.get('all_direction_requirements_met'))}**. Status reason as recorded: {joint_screen.get('status_reason') if joint_screen.get('status_reason') else '*(none recorded)*'}. {flat(joint_screen.get('five_seed_qualification_note'))}

| direction-required member | contrast | seeds | favourable directions | required | met |
|---|---|---|---|---|---|
{dir_rows}"""
        )
    if not out:
        return f"### {label}\n\n*(no family is registered in this statistics file.)*"
    return "\n\n".join(out)


def section_families(stats_4d, stats_c114) -> str:
    return f"""## 3. Family results

The rejection rule is the Holm step-down **within** each family at the registered α, on the registered member set. A family passes only when every required member rejects favourably, every required slice carries at least ten tournament blocks, all five paired seeds are complete, and the registered per-seed direction count is met. The `95% CI` and `U95` columns are **marginal** two-sided 95% percentile intervals, one member at a time, and are reported for context only — see § 2.

{family_section(stats_4d, 'rung 4d')}

{family_section(stats_c114, 'block C114')}"""


def spread_table(stats) -> str:
    keys = []
    for fam in families_of(stats):
        for mem in family_members(fam):
            k = mem.get("contrast_key")
            if k and k not in keys:
                keys.append(k)
    contrasts = stats.get("contrasts", {})
    seed_keys = []
    for k in keys:
        for s in (contrasts.get(k, {}).get("per_seed_points") or {}):
            if s not in seed_keys:
                seed_keys.append(s)
    seed_keys = sorted(seed_keys, key=lambda s: SEEDS.index(int(s)) if int(s) in SEEDS else 99)
    rows = []
    for k in keys:
        c = contrasts.get(k)
        if not c:
            rows.append(f"| `{k}` | *(absent from the statistics file)* |" + " |" * (len(seed_keys) + 5))
            continue
        pts = c.get("per_seed_points") or {}
        ii = c.get("estimand_ii") or {}
        cells = " | ".join(d5(pts.get(s)) for s in seed_keys)
        spread = (c.get("seed_spread") or {}).get("range")
        rows.append(
            f"| `{c.get('candidate')} − {c.get('reference')}` | `{c.get('slice')}` | {cells} | "
            f"{d5(spread)} | "
            f"{c.get('seeds_below_zero_count', c.get('favourable_direction_count'))}/{c.get('n_seeds')} | "
            f"{d5(ii.get('point'))} | {ci5(ii.get('ci95'))} | "
            f"{(c.get('slice_stats') or {}).get('n_blocks')} |"
        )
    heads = " | ".join(f"seed {s}" for s in seed_keys)
    return (
        f"| contrast | slice | {heads} | seed range | favourable directions | mean point (ii) | mean 95% interval (ii) | blocks |\n"
        f"|---|---|{'---|' * len(seed_keys)}---|---|---|---|---|\n" + "\n".join(rows)
    )


def section_spread(stats_4d, stats_c114) -> str:
    return f"""## 4. Seed spread, every family member

Per-seed points are the estimand (i) point estimates at each registered seed. **Seed ranges are empirical spreads over the seeds, not confidence intervals.** No seed is selected and no CI endpoint is averaged. The favourable-direction count is the **zero-threshold** count (per-seed points below 0), which on a non-inferiority gate row is not the same as the count below that row's registered `+0.002` margin.

### 4.1 Rung 4d

{spread_table(stats_4d)}

### 4.2 Block C114

{spread_table(stats_c114)}

In C114, `full_114 − mlp_114 on all` appears as a member of **both** families — `family_c114_seq`'s primary and `family_c114_feats`' `all_row` member. It is one estimand on one set of rows occupying a Holm slot in two families, and a pass in both is not two findings (`known_asymmetries.shared_member`)."""


def abs_ll_block(cfg_yaml, summaries, metrics, refs, title) -> str:
    ll_rows = []
    for cid in config_ids(cfg_yaml):
        rows = per_seed_rows(summaries.get(cid))
        if not rows:
            ll_rows.append(f"| `{cid}` | 0 | n/a | n/a | n/a | no `summary.yaml` per-seed rows on this checkout |")
            continue
        lls = [r["ll"] for r in rows if r.get("ll") is not None]
        best = min(rows, key=lambda r: r["ll"])
        worst = max(rows, key=lambda r: r["ll"])
        exp = (summaries[cid].get("experiment") or {})
        note = f"{exp.get('runs_recorded')}/{exp.get('runs_expected')} runs, complete {yesno(exp.get('complete'))}"
        ll_rows.append(
            f"| `{cid}` | {len(lls)} | {ll6(sum(lls) / len(lls))} | {ll6(best['ll'])} (seed {best['seed']}) | "
            f"{ll6(worst['ll'])} (seed {worst['seed']}) | {note} |"
        )
    for display, key, ll, n in refs:
        ll_rows.append(
            f"| `{display}` (frozen reference, {n} features) | — | {ll4(ll)} | — | — | "
            f"deterministic npz; `references.json` records 4 dp, no seeds, no interval |"
        )
    seed_rows = []
    for cid in config_ids(cfg_yaml):
        for r in per_seed_rows(summaries.get(cid)):
            seed = r.get("seed")
            m = metrics.get((cid, seed)) or {}
            ts = m.get("training_schedule") or {}
            mp = r.get("machine_provenance") or {}
            seed_rows.append(
                f"| `{cid}` | {seed} | {ll6(r.get('ll'))} | {r.get('best_epoch')} | "
                f"{ts.get('epochs_run', 'n/a')} | {ts.get('steps_to_best', 'n/a')} / {ts.get('total_steps', 'n/a')} | "
                f"{integer(m.get('n_params') or (m.get('arm_params') or {}).get('n_parameters'))} | "
                f"{r.get('wall_seconds'):.1f} | {mp.get('machine', 'unrecorded')} |"
            )
    return f"""### {title}

| config | seeds recorded | mean val LL | min | max | note |
|---|---|---|---|---|---|
{chr(10).join(ll_rows)}

| config | seed | validation LL | best epoch | epochs run | steps to best / total | parameters | wall s | machine |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(seed_rows) if seed_rows else '| *(no run directories on this checkout)* | | | | | | | | |'}"""


def section_absolute(cfg_4d, cfg_c114, summaries_4d, summaries_c114, metrics_4d, metrics_c114) -> str:
    refs = reference_lls()
    return f"""## 5. Absolute validation log loss per configuration

Numbers of record: full-precision per-seed validation log loss from `runs/<config>/summary.yaml`; never rounded report values and never log loss reconstructed from saved probabilities. Early stopping is live in batch 2, so arms may stop at different epochs — the stopping epoch is printed per run and **equal compute is not claimed**. The three frozen references sit in the same table as a level readout only; they are deterministic artifacts and `references.json` records them to four decimals.

{abs_ll_block(cfg_4d, summaries_4d, metrics_4d, refs, "5.1 Rung 4d")}

{abs_ll_block(cfg_c114, summaries_c114, metrics_c114, refs, "5.2 Block C114")}

{stage1_paragraph()}"""


def stage1_paragraph() -> str:
    cb = stage1_cb_row()
    if not cb.get("available"):
        return (
            "**Stage 1 cross-reference for C114 — unavailable.** "
            f"`{STAGE1_REPORT.relative_to(REPO)}` is not present on this checkout, so the stage-1 "
            "rollout C−B reading is not quoted here rather than being restated from memory."
        )
    return (
        "**Stage 1 cross-reference for C114, text only and a DIFFERENT measurement.** Stage 1's § 4 "
        f"Holm table records C−B at **{cb['point']}** {cb['interval']} (rank-local, raw p "
        f"{cb['raw_p']}, Holm p {cb['holm_p']}, labelled {cb['label']}), read from "
        f"`{STAGE1_REPORT.relative_to(REPO)}` (sha256 `{cb['sha256'][:12]}…`). That quantity is "
        "**winner log loss in rollout through the simulator**, on selected seed-101 checkpoints and "
        "the **50-feature** contract. Nothing in this section is that quantity: here every number is "
        "**teacher-forced ball log loss on the validation split**, on the 114-column contract, under "
        "the batch-2 epoch schedule. Different target, different information set, different feature "
        "contract — so a batch-2 reading in either direction is **not a contradiction of** the "
        "stage-1 rollout gap, no table places the two side by side, and neither confirms the other."
    )


def parse_4b_log(path: Path) -> dict:
    """Epoch readings, early stop and the refusal line, read from one run.log."""
    full = path.read_text()
    lines = [ln.rstrip() for ln in full.splitlines()]
    # A run.log can carry several attempts (the driver retries once). Parse the
    # LAST attempt only, so no epoch line is counted twice.
    starts = [m.start() for m in re.finditer(r"^=== job .* attempt \d+ start ", full, re.M)]
    n_attempts = len(starts)
    text = full[starts[-1]:] if starts else full
    epochs = re.findall(r"^epoch (\d+): train_ll=([0-9.]+) val_ll=([0-9.]+)", text, re.M)
    recorded = re.findall(r'"validation_ll":\s*([0-9.]+)', text)
    refusal = [ln.strip() for ln in text.splitlines() if "carries no" in ln and "parameter" in ln]
    return {
        "path": path,
        "n_lines": len(lines),
        "n_attempts": n_attempts,
        "epochs": epochs,
        "epoch_numbers": [int(e[0]) for e in epochs],
        "val_lls": sorted({e[2] for e in epochs}),
        "early_stop": "early stop" in text,
        "recorded_validation_ll": recorded[-1] if recorded else None,
        "refusal": refusal[-1] if refusal else None,
    }


def section_4b(cfg_4b) -> str:
    lambdas = {}
    for c in cfg_4b.get("configurations", []):
        lam = (c.get("params") or {}).get("residual_lambda")
        if lam is not None:
            lambdas[c["id"]] = lam
    lam_text = ", ".join(f"`{k}` λ = {v}" for k, v in lambdas.items()) or "the two registered λ arms"
    refs = dict((d, ll) for d, _k, ll, _n in reference_lls())
    eb = refs.get("ref_eb_ctx")

    rows, quoted, markers = [], [], []
    for cid in FOURB_CONFIGS:
        for seed in SEEDS:
            rd = RUNS_ROOT / cid / f"seed_{seed}"
            log = rd / "run.log"
            if not rd.exists() and not log.exists():
                continue
            marks = [
                n for n in ("COMPLETE.json", "COMPLETE", "FAILED", "summary.yaml")
                if (rd / n).exists() or (RUNS_ROOT / cid / n).exists()
            ]
            markers.extend(marks)
            if not log.exists():
                rows.append(
                    f"| `{cid}` | {seed} | n/a | run.log absent | — | — | — | {', '.join(marks) or 'none'} |"
                )
                continue
            p = parse_4b_log(log)
            rows.append(
                f"| `{cid}` | {seed} | {p['n_attempts']} | {len(p['epochs'])} "
                f"({min(p['epoch_numbers'])}–{max(p['epoch_numbers'])}) | "
                f"{', '.join(p['val_lls'])} | {yesno(p['early_stop'])} | "
                f"{p['recorded_validation_ll'] or 'n/a'} | {', '.join(marks) or 'none'} |"
            )
    for cid in FOURB_CONFIGS:
        log = RUNS_ROOT / cid / "seed_7" / "run.log"
        if not log.exists():
            quoted.append(
                f"* `{cid}/seed_7/run.log` — **absent on this checkout**; no line is quoted from it here."
            )
            continue
        p = parse_4b_log(log)
        body = [f"epoch {n}: train_ll={t} val_ll={v}" for n, t, v in p["epochs"]]
        if p["early_stop"]:
            body.append("early stop")
        if p["recorded_validation_ll"]:
            body.append(f'"validation_ll": {p["recorded_validation_ll"]}')
        if p["refusal"]:
            body.append(p["refusal"])
        quoted.append(
            f"* `{cid}/seed_7/run.log` (sha256 `{short_sha(log)}`), {p['n_lines']} lines; "
            f"the epoch, stop and refusal lines, verbatim:\n\n```\n" + "\n".join(body) + "\n```"
        )
    marker_note = (
        f"Markers actually on disk for this rung: {', '.join(sorted(set(markers))) or 'none'}."
    )
    deferral = flat((cfg_4b.get("statistics", {}).get("families", {}) or {}).get("definition") or "")
    return f"""## 6. Rung 4b — no admissible runs, family deferred

**Rung 4b produced no admissible runs.** Every `identity_residual` job — {lam_text}, every seed attempted — **exited after training**, when the driver's checkpoint verification refused the checkpoint: `scripts/sequence_track/retrain_stage2.py` requires `feat_proj` / `head` parameters in the saved state dict, and the identity-only arm does not carry them. Training itself ran and was retried once; nothing survived the verification step, so there is no `COMPLETE.json` and no `summary.yaml` for either configuration, no statistics file for this rung, and **no number of this rung enters any family, any contrast table or any status**. {marker_note}

The family was already deferred by registration, before any of this: {deferral if deferral else 'the 4b config registers `map: []` — no family at all.'} The deferral reason is a schema limitation, not a result: `stage2_stats.py` refuses a family member whose reference is not a registered configuration with run directories, and 4b's designed reference is a frozen npz artifact.

**Logged OBSERVATIONS from `run.log`, explicitly NOT evidence.** What the logs contain is listed below and nothing is inferred from it. The printed validation log losses are **rounded to four decimals** — they include {ll4(eb)} and also 1.4499 and 1.4501 — so they cannot support any statement about whether the arm's predictions moved: a rounded log loss is not a prediction, equal rounded values are not equal predictions, and the frozen `ref_eb_ctx` level ({ll4(eb)} as recorded in `references.json`) is quoted here only as a coincident level, not as a comparison. Read from each log's **last attempt** (the driver retries once, so earlier attempts' epoch lines are not counted again):

| config | seed | attempts | epochs printed (last attempt) | distinct val_ll printed (4 dp) | early stop | recorded `validation_ll` | markers on disk |
|---|---|---|---|---|---|---|---|
{chr(10).join(rows) if rows else '| *(no rung-4b run directory on this checkout)* | | | | | | | |'}

These are log lines, not measurements. **The defect, not the numbers, is the finding of this rung:** the arm trained, the driver's artefact check refused every checkpoint, and the runs are therefore **excluded from evidence entirely** — no bootstrap, no interval, no Holm slot, no direction count and no comparison against any reference was computed, and none may be computed from these directories. Nothing above may be quoted as "the identity residual matches the reference", as "the residual did not move the base", or as any statement about what the arm learned.

{chr(10).join(quoted)}

**What is required before rung 4b can say anything.** A driver fix (checkpoint verification that accepts an arm with no `feat_proj` / `head`), a full rerun of both λ arms at all five seeds, and — separately and beforehand — a new frozen config registering the family if any interval against the frozen reference is ever to be computed. Registering that family after reading the descriptive table would make the registration outcome-informed, and this report does not read one."""


def member_reading(mem) -> str:
    """The registered reading of one member's marginal interval, as the JSON says."""
    ci = mem.get("ci95")
    if not ci:
        return "no interval available"
    if ci[1] < 0:
        return "CI-clean favourable"
    if ci[0] > 0:
        return "CI-clean adverse"
    return "unresolved — the interval straddles zero"


def plain_family_sentences(stats, block) -> str:
    """`block` is '4d' or 'C114'; C114 carries the 114-feature qualifier."""
    is_c114 = block == "C114"
    lines = []
    for fam in families_of(stats):
        key = fam.get("holm_group")
        screen = (fam.get("screen") or {}).get("seed_mean_joint") or {}
        status = screen.get("status")
        members = family_members(fam)
        by_name = {m.get("member"): m for m in members}
        primary_name = screen.get("primary_member") or "primary"
        primary = by_name.get(primary_name) or (members[0] if members else None)
        detail, primary_reading = "", None
        if primary:
            primary_reading = member_reading(primary)
            detail = (
                f" Its primary member `{primary.get('member')}` ({primary.get('contrast')}) is "
                f"{d5(primary.get('point'))} {ci5(primary.get('ci95'))} on `{primary.get('slice')}` rows: "
                f"**{primary_reading}** (marginal interval; Holm `rejected` {yesno(primary.get('rejected'))})."
            )
        required = screen.get("screen_required_members") or []
        failing = [
            by_name[n] for n in required
            if n in by_name and by_name[n].get("status") != "SCREEN_PASS" and n != primary_name
        ]
        fail_text = ""
        if failing:
            fail_text = " The family fails on its other required members: " + "; ".join(
                f"`{m.get('member')}` ({m.get('contrast')}) {d5(m.get('point'))} {ci5(m.get('ci95'))} "
                f"— {member_reading(m)}"
                for m in failing
            ) + "."
        if status == "SCREEN_PASS":
            lines.append(
                f"* **`{key}`: `SCREEN_PASS`.**{detail} A pass here is a **screen**, not a confirmation: it "
                f"remains provisional and outcome-informed (validation only, checkpoint selected on the same "
                f"split, cohort unopened), nothing advances, and no market or out-of-sample claim follows."
            )
        elif status == "NOT_EVALUABLE":
            lines.append(
                f"* **`{key}`: `NOT_EVALUABLE`.**{detail} That reading is **descriptive only** — the "
                f"registered preconditions were not met (block count or complete paired seeds), so the "
                f"family carries no interval-based claim and no inference in either direction."
            )
        elif is_c114:
            qualifier = (
                f" As registered, that is a **{primary_reading} teacher-forced screening reading on the "
                f"114-feature contract** — outcome-informed and validation-only, on the split that also "
                f"selected every checkpoint. It is not an out-of-sample, rollout, simulation or market "
                f"statement, and it bears on no production artifact."
                if primary_reading
                else ""
            )
            extra = ""
            if primary_reading == "CI-clean favourable" and failing:
                extra = (
                    " So the family's own primary question is answered favourably while the family still "
                    "does **not** pass: the screen requires every registered member, and the members shared "
                    "with the sequence question are adverse."
                )
            lines.append(
                f"* **`{key}`: does not pass its screen** (`{status}`).{detail}{qualifier}{fail_text}{extra}"
            )
        else:
            boiler = (
                " Not passing is not evidence against the mechanism; where the interval straddles zero the "
                "reading is **unresolved at this resolution**."
            )
            lines.append(
                f"* **`{key}`: does not pass its screen** (`{status}`).{detail}{fail_text}{boiler}"
            )
    if not lines:
        return f"* {block}: no family is registered in this statistics file."
    return "\n".join(lines)


def section_plain(stats_4d, stats_c114) -> str:
    all_4d = ((stats_4d.get("slices") or {}).get("stats") or {}).get("all") or {}
    return f"""## 7. In plain language

**What was tested.** Small models were trained on historical T20 deliveries and then asked, one ball at a time, to put a probability on each of the six outcomes of the *next* delivery of {integer(all_4d.get('n_rows'))} held-out validation balls. Rung 4d's two arms differ only in how many exposure columns they read; C114's three arms differ in whether they can see earlier balls and in which feature contract they read. This is teacher-forced ball prediction: nobody simulated a match, no market price appears anywhere in this batch, and a better log loss here implies nothing about a score distribution, a win probability, a prop or a betting edge.

**The registered reading, rung 4d.**

{plain_family_sentences(stats_4d, '4d')}

**The registered reading, block C114.**

{plain_family_sentences(stats_c114, 'C114')}

**What none of this establishes.** A screen that does not pass does not show the mechanism is inert, and a CI-clean adverse reading on the validation split whose rows also selected every checkpoint is a **screening** reading: it says the transformer was worse than the token model on these held-out balls under this schedule, not that sequence access hurts out of sample, in rollout, or anywhere a decision is made. The C114 families are **correlated screens** over the same rows with a shared member and a shared gate reference, so there is no "either question passes" reading and no inference from one passing where the other does not. `full_114 − full_50` moves player identity together with the other 64 columns, so a favourable reading there is "the production contract beats the 50-feature contract", never "more state features help". Neither 114-column arm reproduces the production ball model — its categorical encoders are fitted differently — so nothing here bears on the model of record. Rung 4b established nothing at all: its runs were refused (§ 6).

**Rung 4b, in one sentence.** It did not produce admissible runs — what is on disk is training logs and checkpoints the driver refused — so it is excluded from evidence, its family stays deferred, and a driver fix plus a full rerun is required before the rung can be read at all."""


def section_deviations(cfg_4d, cfg_c114, summaries_all, stats_list) -> str:
    by_seed = {}
    for stats in stats_list:
        for cid, block in (stats.get("runs") or {}).items():
            for s in block.get("seeds") or []:
                p = s.get("provenance") or {}
                by_seed.setdefault(s.get("seed"), set()).add(
                    (p.get("machine"), p.get("chip"), p.get("hostname"))
                )
    for cid, summary in summaries_all.items():
        for r in per_seed_rows(summary):
            mp = r.get("machine_provenance") or {}
            by_seed.setdefault(r.get("seed"), set()).add(
                (mp.get("machine"), mp.get("chip"), mp.get("hostname"))
            )
    mach_rows = "\n".join(
        f"| {seed} | {', '.join(sorted(m or 'unrecorded' for m, _, _ in v))} | "
        f"{', '.join(sorted(c or 'unrecorded' for _, c, _ in v))} | "
        f"{', '.join(sorted(h or 'unrecorded' for _, _, h in v))} |"
        for seed, v in sorted(by_seed.items(), key=lambda kv: SEEDS.index(kv[0]) if kv[0] in SEEDS else 99)
    ) or "| *(no run provenance on this checkout)* | | | |"
    b6 = acceptance_b6_row()
    b6_block = (
        f"The acceptance file's own B6 row, verbatim from `{ACCEPTANCE.relative_to(REPO)}`:\n\n```\n{b6}\n```"
        if b6
        else f"`{ACCEPTANCE.relative_to(REPO)}` carries no `B6` row on this checkout."
    )
    return f"""## 8. Registered deviations, asymmetries and limitations (restated, none dropped)

### 8.1 Rung 4d — config `deviations`

{bullets(cfg_4d.get('deviations'))}

### 8.2 Rung 4d — config `known_asymmetries`

{bullets(cfg_4d.get('known_asymmetries'))}

### 8.3 Rung 4d — config `known_limitations`

{bullets(cfg_4d.get('known_limitations'))}

### 8.4 Block C114 — config `deviations`

{bullets(cfg_c114.get('deviations'))}

### 8.5 Block C114 — config `known_asymmetries`

{bullets(cfg_c114.get('known_asymmetries'))}

### 8.6 Block C114 — config `known_limitations`

{bullets(cfg_c114.get('known_limitations'))}

### 8.7 Restated in this report's own terms

* **Seed–machine confounding.** Machine is recorded per run and is confounded with seed. Read from the run provenance:

| seed | machine | chip | host |
|---|---|---|---|
{mach_rows}

  The split is **by seed** (laptop 7, 29, 42; mini 13, 101), so every arm carries the same mix and no arm is advantaged relative to another; every registered contrast is within-machine at each seed, so an additive machine effect cancels inside it. Arm-by-machine interaction remains inseparable from seed variation and **no machine term is fitted**. MPS kernels are not bit-reproducible across machines or runs, and every checkpoint records `mps_bit_reproducible false`.
* **The mini smoke was not run — a launch-gate deviation.** Acceptance check B6 requires one short token-arm run and one short `full`-arm run to complete **on each machine** before launch, with wall seconds in the runbook's batch-2 capacity table. The mini's token and `full` cells were filled from **laptop** smokes and marked "mini: NOT smoked — deviation, see B6"; only the `identity_residual` arm was smoked on the mini. So the mini's per-job budgets are extrapolations, not measurements, and a mini timeout would be a budgeting artifact rather than a result. {b6_block}
* **Correlated screens, family-local claims only.** C114's two families share a member and a gate reference; 4d has one family. No aggregate across families, and no comparison of one family's outcome to another's.
* **Selection optimism throughout.** Checkpoint selection used the same validation split every contrast is computed on, so every number in this report is screening evidence.
* **Two questions, one batch, no cross-tabling.** Rung 4d and block C114 are separate registrations with separate shared controls; batch-2 arms are also never tabled against night-3 Block B arms, whose schedule was a fixed step budget."""


def section_next(stats_4d, stats_c114, cfg_label_ids) -> str:
    coh = stats_4d.get("cohort", {})
    return f"""## 9. What happens next

**Nothing advances.** `advances: {coh.get('advances')}`, `cohort_status: {coh.get('cohort_status')}`, `reads_performed: {coh.get('reads_performed')}`. No cohort feature, prediction or base-logit read happened, no artifact of record changed, no test split, golden set, forward holdout or market price was touched, and no verdict was logged — that call is the user's.

Run completeness as recorded in each configuration's `summary.yaml` on this checkout:

{completeness_table(cfg_label_ids)}

A family with fewer than five complete paired seeds is `NOT_EVALUABLE` and is **never** reported at a reduced seed count; rung 4b has no `summary.yaml` at all, which is the refusal described in § 6 and not a seed shortfall.

**The remaining work is rung 4b, and only rung 4b.** Rungs 4d and C114 are done: they ran, they were scored under their frozen registrations, and their screens did not pass — a screen that does not pass is a finished reading, not an unfinished run, so **there is nothing to re-run and nothing to advance for them**. What is outstanding:

1. **Fix the driver defect.** `scripts/sequence_track/retrain_stage2.py` refuses an `identity_residual` checkpoint because it looks for `feat_proj` / `head` parameters the identity-only arm does not carry. The artefact check must accept that arm's parameter set.
2. **Re-run rung 4b** — both λ arms at all five seeds — after the fix, and only then consolidate a `summary.yaml`.
3. **Register 4b's family in a new frozen config *before* any 4b readout is read as evidence.** The family is deferred because a frozen npz reference cannot be a family member under the present schema; registering after reading would make the registration outcome-informed.
4. **Leave the cohort closed.** It is `{coh.get('cohort_status')}`; the one confirmatory read must not be spent on screening evidence, and no batch-2 result changes that.

**Nothing in this batch advances, is promoted, or becomes an artifact of record**, and no verdict is logged by this report."""


# ---------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stats-4d", default=str(STATS_4D))
    ap.add_argument("--stats-c114", default=str(STATS_C114))
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    paths = {
        "stats_4d": Path(args.stats_4d).resolve(),
        "stats_c114": Path(args.stats_c114).resolve(),
    }
    for label, p in paths.items():
        if not p.exists():
            raise SystemExit(
                f"{label}: {p} does not exist. Run `stage2_stats.py stats` for that config first, "
                f"or point --stats-4d / --stats-c114 at an existing statistics file."
            )
    stats_4d = json.loads(paths["stats_4d"].read_text())
    stats_c114 = json.loads(paths["stats_c114"].read_text())
    cfg_4d = yaml.safe_load(CONFIG_4D.read_text())
    cfg_c114 = yaml.safe_load(CONFIG_C114.read_text())
    cfg_4b = yaml.safe_load(CONFIG_4B.read_text())

    ids_4d = config_ids(cfg_4d)
    ids_c114 = config_ids(cfg_c114)
    summaries_4d = load_summaries(ids_4d)
    summaries_c114 = load_summaries(ids_c114)
    metrics_4d = load_metrics(ids_4d)
    metrics_c114 = load_metrics(ids_c114)

    cfg_label_ids = [("4d", cid, summaries_4d.get(cid)) for cid in ids_4d] + [
        ("C114", cid, summaries_c114.get(cid)) for cid in ids_c114
    ] + [("4b", cid, load_summaries([cid]).get(cid)) for cid in config_ids(cfg_4b)]

    summaries_all = dict(summaries_4d)
    summaries_all.update(summaries_c114)

    render_ts = datetime.now(timezone.utc).isoformat()
    parts = [
        section_header(stats_4d, stats_c114, paths, render_ts),
        section_questions(cfg_4d, cfg_c114, metrics_4d, metrics_c114),
        section_decision(cfg_4d, cfg_c114, stats_4d, stats_c114),
        section_families(stats_4d, stats_c114),
        section_spread(stats_4d, stats_c114),
        section_absolute(cfg_4d, cfg_c114, summaries_4d, summaries_c114, metrics_4d, metrics_c114),
        section_4b(cfg_4b),
        section_plain(stats_4d, stats_c114),
        section_deviations(cfg_4d, cfg_c114, summaries_all, [stats_4d, stats_c114]),
        section_next(stats_4d, stats_c114, cfg_label_ids),
    ]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n\n".join(parts) + "\n")
    print(f"wrote {out_path} ({len(out_path.read_text().splitlines())} lines)")


if __name__ == "__main__":
    main()
