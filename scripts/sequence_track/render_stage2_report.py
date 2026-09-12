#!/usr/bin/env python
"""Render `research/reports/embeddings/SEQ_STAGE2_REPORT.md` (D10.8, D10.13-D10.15).

Every numerical cell is read from a file — the statistics JSON written by
`stage2_stats.py stats`, the k-selection record written by
`stage2_stats.py ksweep`, the per-configuration `summary.yaml` facts those
carry, the `metrics.json` provenance they carry, the dependency JSONs and the
registered config. The narrative passages are authored text in this
generator, as in `render_stage1_report.py`. No result cell is hand-entered.

§ 9 restates every deviation, asymmetry and limitation in the config plus the
full list D10.13 enumerates; `assert_coverage` fails the render if any one of
them is missing from the rendered markdown. § 10 is the plain-language
section required by D10.14 and § 11 carries the D10.15 falsification wording.

The report states `cohort_status: DEFERRED_UNOPENED`, that no arm advances,
that there is no market claim and no LANDED verdict, and that the user's
verdict is outstanding. It reads no smoke log loss, never opens the cohort,
and never opens `data/golden/` or `data/forward_holdout/`.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from sequence_track.stage2_stats import (  # noqa: E402
    ALLOWED_STATUSES,
    DEFAULT_CONFIG,
    DEFAULT_KSWEEP_OUT,
    DEFAULT_STATS_OUT,
    FORBIDDEN_FRAGMENTS,
    JOINT_READOUT,
    MARGIN_LL,
    READOUTS,
    RefusalError,
    require_distinct_out,
    guard_path,
    md5_file,
    read_json,
    read_yaml,
    rel,
    sha256_file,
)

DEFAULT_DEPENDENCY_DIR = (REPO / "models" / "embeddings" / "seq_stage2"
                          / "dependency")
DEFAULT_OUT = (REPO / "research" / "reports" / "embeddings"
               / "SEQ_STAGE2_REPORT.md")
ACCEPTANCE = REPO / "docs" / "sequence_track" / "stage2_acceptance.md"

READOUT_LABELS = {"seed_7": "seed 7 (estimand i)",
                  "seed_13": "seed 13 (estimand i)",
                  "seed_mean_joint": "seed mean (estimand ii)"}


def readout_label(readout: str) -> str:
    """The column label for one readout, derived from its name.

    Astra gate 2 round 2: the labels were a two-seed dictionary, so a
    five-seed readout would have fallen back to its raw key. Seeds 29, 42 and
    101 now get the same wording seeds 7 and 13 do.
    """
    if readout in READOUT_LABELS:
        return READOUT_LABELS[readout]
    if str(readout).startswith("seed_"):
        return f"seed {str(readout)[len('seed_'):]} (estimand i)"
    return str(readout)


def readouts_of(stats: Mapping[str, Any]) -> tuple[str, ...]:
    """The readout list the statistics actually computed, in its own order.

    Read from `contract.readouts`, which `stage2_stats` derives from the seeds
    it was given, so a five-seed statistics JSON renders five per-seed columns
    instead of silently dropping seeds 29, 42 and 101. A statistics JSON
    predating that field falls back to the two-seed tuple.
    """
    recorded = ((stats.get("contract") or {}).get("readouts")
                if isinstance(stats, Mapping) else None)
    if isinstance(recorded, (list, tuple)) and recorded:
        return tuple(str(r) for r in recorded)
    return tuple(READOUTS)

# ---------------------------------------------------------------------------
# D10.12 — the masked arms the dependency recertification must cover, and the
# configuration each certificate belongs to.
#
# Astra gate 1 round 3 ruling: the deferred recertification is acceptable, but
# the heading must say **pending** rather than "recertified" until trained-
# checkpoint coverage is verified, and a failed certificate must BLOCK the
# affected interpretation and eligibility rather than merely appear in a
# table. A failed or missing certificate for a masked arm therefore forces
# that arm's family to NOT_EVALUABLE here, in the report itself.
DEPENDENCY_REQUIRED: dict[tuple[str, str], str] = {
    ("same_entity", "0"): "same_entity_k0",
    ("same_entity", "30"): "same_entity_k30",
    ("same_entity", "unr"): "same_entity_unr",
    ("recency", "30"): "recency_k30",
}

# Astra gate 2 round 1 MUST-FIX 1. A positive control is a STANDARD-wiring arm
# scored against the MASKED arm's S(i), so it never carries the masked arm's
# own `arm`/`k`: matching on those skips the real `full` and `aligned_hist`
# controls entirely. The control is matched through the dependency set it was
# scored against — `dependency_set_arm` / `dependency_set_k` — which is the
# only field that names the masked arm it exercises.
#
# D7.4 registers exactly two controls: `full` against `recency_k30`'s S(i) and
# `aligned_hist` against `same_entity_k30`'s S(i). `same_entity_k0` and
# `same_entity_unr` have no separately registered control; that gap is
# disclosed in the report rather than silently treated as satisfied.
DEPENDENCY_CONTROLS: dict[tuple[str, str], dict[str, str]] = {
    ("recency", "30"): {"config_id": "recency_k30", "control_arm": "full"},
    ("same_entity", "30"): {"config_id": "same_entity_k30",
                            "control_arm": "aligned_hist"},
}
# The perturbation settings D7.1/D10.12 register. A certificate computed under
# any other setting is not the registered test and does not count as coverage.
DEPENDENCY_REGISTERED_SETTINGS: dict[str, Any] = {
    "n_targets": 2000, "seed": 29, "split": "validation"}
# D10.12 requires the recertification at BOTH registered training seeds.
DEPENDENCY_REQUIRED_CHECKPOINT_SEEDS: tuple[int, ...] = (7, 13)

DEPENDENCY_HEADING_PENDING = (
    "### Ownership dependency certificates (D7; D10.12 recertification "
    "**pending**)")
DEPENDENCY_HEADING_COMPLETE = (
    "### Ownership dependency certificates (D7; D10.12 recertification "
    "**complete on trained checkpoints at both registered seeds, with the "
    "registered positive controls matched**)")
DEPENDENCY_BLOCKED_STATUS = "NOT_EVALUABLE"


def dependency_heading(certificates: Mapping[str, Any]) -> str:
    """SHOULD 8: the heading is derived from verified coverage, never asserted.

    The report may not say "pending" while also saying coverage is complete,
    nor the reverse.
    """
    return (DEPENDENCY_HEADING_COMPLETE
            if certificates.get("trained_checkpoint_coverage_complete")
            else DEPENDENCY_HEADING_PENDING)

NO_SEQUENCE_GAIN_SENTENCE = ("This model family, at this resolution, shows no "
                             "further sequence gain")
NO_SEQUENCE_GAIN_QUALIFIER = (
    "That is screening evidence from two seeds on one validation split, and "
    "it is **not** proof that the scoreboard summary suffices.")

# D10.13 — every item the report must carry, beyond the config's own entries.
D10_13_REQUIRED: tuple[tuple[str, str], ...] = (
    ("blocked_cross_fitting_not_past_only_forecasting",
     "the residual base logits on the train split are BLOCKED cross-fitting, "
     "not past-only forecasting: earlier date blocks are scored by boosters "
     "fitted on later blocks, so no residual arm may be described as fully "
     "as-of or whole-pipeline leakage-free (Astra gate 1 NOTE 3)."),
    ("later_block_training_for_earlier_oof",
     "the same construction means a fold's out-of-fold predictions for early "
     "dates come from a model that saw later cricket; the direction of that "
     "exposure on any residual contrast is unmeasured."),
    ("caches_and_encoders_not_fold_refitted",
     "the stats cache and the production label encoders are not refitted per "
     "fold, so fold independence is partial by construction."),
    ("fixed_round_residual_refits",
     "each leave-one-block-out refit runs a fixed 25 boosting rounds with no "
     "early stopping and no eval set, so no fold re-selects its round count "
     "against the validation split."),
    ("wiring_plus_key_construction",
     "`aligned_hist_rf - aligned_hist` measures the relay-free wiring PLUS "
     "the key construction. It is not the wiring alone and is never reported "
     "as such."),
    ("own_outcome_versus_shifted_history_keys",
     "the relay-free arms read an earlier row as a (state, OWN outcome) pair "
     "while the standard-wiring arms read every row as a (state, SHIFTED "
     "previous outcome) pair; no arm can read its own label (certified in "
     "D7)."),
    ("ownership_plus_alignment",
     "`same_entity_k30 - recency_k30` is ownership PLUS alignment beyond "
     "recency, because the two arms differ in the history input as well as "
     "the mask, so that contrast does not isolate ownership from alignment. "
     "The registered `same_entity_unr - aligned_hist_rf` contrast DOES hold "
     "the aligned history input, the relay-free wiring and the key "
     "construction fixed, so it isolates the attention mask given aligned "
     "inputs; no claim of isolated ownership is drawn from the k30 contrast "
     "(Astra gate 2 round 1 MUST-FIX 4)."),
    ("machine_confounded_with_seed_no_machine_term",
     "seed 7 trained on the laptop and seed 13 on the Mac mini, so MACHINE IS "
     "CONFOUNDED WITH SEED. Every registered contrast is within-machine at "
     "each seed, so an additive machine effect cancels inside it, but "
     "arm-by-machine interaction is inseparable from seed variation and **no "
     "machine term is fitted**."),
    ("xlstm_simplifications",
     "every registered xLSTM simplification stands as recorded by "
     "`scripts/sequence_track/recurrent_arms.simplifications()`; the arm is "
     "plain torch, not the paper's reference implementation."),
    ("a15_not_adopted_and_checkpoint_selection_optimism",
     "A15 is not adopted: early stopping reads the same validation split the "
     "contrasts are computed on, so checkpoint-selection optimism is "
     "uncorrected and every number here is screening evidence."),
    ("two_seed_weakness",
     "two seeds are a directional screen. Estimand (ii) is descriptive; no "
     "interval here is a confirmation, and a two-seed result can never be "
     "LANDED (invariant 9)."),
    ("unmatched_parameter_counts",
     "parameter counts are not matched across arms — a token-MLP arm has no "
     "attention layer at all — so no contrast is a capacity-controlled "
     "comparison; counts are reported per run."),
    ("access_and_position_schemes",
     "access rows and positional-embedding schemes differ by arm: the "
     "recurrent arms carry no positional embedding, fixed_decay and fox carry "
     "none by registration, and the residual arms alone see the production "
     "prior."),
    ("mps_non_bit_reproducibility",
     "MPS kernels are not bit-reproducible across runs or machines, so any "
     "rerun may differ in the low decimals; every checkpoint records "
     "`mps_bit_reproducible false`."),
    ("fresh_stage2_runs_not_stage1_replications",
     "stage 1 checkpoints are not reused: these are fresh stage 2 runs under "
     "the stage 2 config, not replications of stage 1 numbers."),
    ("global_prior_not_as_of",
     "the global outcome prior in the stats cache is not as-of-date; the "
     "direction and size of its effect on any stage 2 contrast are unknown, "
     "so it cannot be used to argue a contrast is unaffected."),
    ("venue_history_not_recency_weighted",
     "venue history in the frame is not recency-weighted; every arm is given "
     "the same feature, so no arm has it while another lacks it. Whether its "
     "effect is the SAME across architectures is unmeasured — a shared "
     "feature can have different effects in different functions of it — so "
     "it is not asserted to be a level effect rather than a per-arm "
     "advantage (Astra gate 2 round 1 MUST-FIX 4)."),
    ("unresolved_historical_consumption",
     "whether the 2026-04-17 -> 2026-08-05 cohort window is untouched as an "
     "EVALUATION set is NOT settled. `docs/sequence_track/"
     "stage2_cohort_consumer_audit.md` establishes clean training frame and "
     "cache ancestry only, and clean training ancestry does not prove "
     "untouched evaluation status; D10.16(0) keeps Astra gate 1 round 2 "
     "MUST-FIX 6 open, so the cohort may not be opened on seed extension "
     "alone, however many seeds are added."),
    ("prior_test_inspection_and_d4_d5_reads",
     "the i7 test split was read by the 2026-08 program and is not a clean "
     "holdout; D4 scored test base logits and D5 ran a whole-test parity "
     "rebuild. Stage 2 does not train or select on test, and the untouched "
     "cohort exists because of this exposure."),
    ("shared_features_do_not_prove_shared_effects",
     "a shared feature set does not prove a shared effect across "
     "architectures: the arms are different functions of the same inputs."),
    ("invalid_historical_rss_unavailable",
     "the historical per-seed peak RSS recorded as a `RUSAGE_CHILDREN` delta "
     "is invalid and is reported as unavailable, never reconstructed."),
)


# ---------------------------------------------------------------------------
# Astra gate 2 round 1 MUST-FIX 4 — registered corrections to config-sourced
# § 9 prose.
#
# `experiments/configs/seq_stage2_v1.yaml` is launch-era training provenance
# and is NOT edited here. Where a config entry states a conclusion the
# evidence does not support, the renderer replaces exactly that clause and
# DISCLOSES the replacement beside the entry, naming the original wording and
# the reason. Nothing else in the entry changes, and the correction is applied
# to the § 9 coverage manifest too, so the corrected text is what must appear.
# ---------------------------------------------------------------------------

CONFIG_TEXT_CORRECTIONS: tuple[dict[str, str], ...] = (
    {"source": "known_limitations[1]",
     "old": ("every arm inherits the same feature, so it is a level effect "
             "on all of them rather than a per-arm advantage"),
     "new": ("every arm is given the same feature, so no arm has it while "
             "another lacks it; whether its effect is the SAME across "
             "architectures is unmeasured, and no level effect is claimed"),
     "reason": ("Astra gate 2 round 1 MUST-FIX 4: a shared venue feature can "
                "have different effects in different architectures, so "
                '"a level effect rather than a per-arm advantage" is not '
                "established by the fact that every arm reads the feature")},
)


def correct_config_text(text: str) -> tuple[str, list[dict[str, str]]]:
    """Apply the registered § 9 prose corrections to one config-sourced entry."""
    applied: list[dict[str, str]] = []
    for correction in CONFIG_TEXT_CORRECTIONS:
        if correction["old"] in text:
            text = text.replace(correction["old"], correction["new"])
            applied.append(correction)
    return text, applied


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------

DASH = "–"


def f5(value: Any) -> str:
    return DASH if value is None else f"{float(value):+.5f}"


def f4(value: Any) -> str:
    return DASH if value is None else f"{float(value):.4f}"


def f6(value: Any) -> str:
    return DASH if value is None else f"{float(value):.6f}"


def thread_caps(caps: Any) -> str:
    if not isinstance(caps, Mapping) or not caps:
        return DASH
    return ", ".join(f"{name.split('_')[0]}={value}"
                     for name, value in sorted(caps.items()))


def sci(value: Any) -> str:
    return DASH if value is None else f"{float(value):.3e}"


def ci(bounds: Any, digits: int = 5) -> str:
    if not bounds:
        return DASH
    return f"[{float(bounds[0]):+.{digits}f}, {float(bounds[1]):+.{digits}f}]"


def flag(value: Any) -> str:
    if value is None:
        return DASH
    return "yes" if value else "no"


def text_or_dash(value: Any) -> str:
    if value in (None, ""):
        return DASH
    return str(value).replace("\n", " ").strip()


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# § 9 coverage (D10.13)
# ---------------------------------------------------------------------------

SECTION_9_HEADING = ("## 9. Registered deviations, asymmetries, limitations "
                     "(restated, none dropped)")


def section_9_body(markdown: str) -> str | None:
    """The § 9 text only, or None if the section is absent.

    Astra gate 2 round 1 MUST-FIX 5: the coverage check looked for short
    prefixes ANYWHERE in the document, so a truncated consequence clause and
    an entry rendered in some other section both passed. Coverage is checked
    inside § 9 itself.
    """
    if SECTION_9_HEADING not in markdown:
        return None
    tail = markdown.split(SECTION_9_HEADING, 1)[1]
    for line in ("\n## ",):
        if line in tail:
            tail = tail.split(line, 1)[0]
    return tail


def coverage_manifest(config: Mapping[str, Any]) -> list[dict]:
    """Every § 9 entry the report must carry, with what proves it is there.

    Every fragment is the COMPLETE normalised entry text, consequence clauses
    included — not a prefix. A truncated entry therefore fails the check.
    """
    manifest: list[dict] = []
    for entry in config.get("deviations") or []:
        key = str(entry.get("id"))
        must = [f"`{key}`", correct_config_text(
            normalise(str(entry.get("statement") or "")))[0]]
        if entry.get("reason"):
            must.append(correct_config_text(
                normalise(str(entry.get("reason"))))[0])
        for line in entry.get("simplifications") or []:
            must.append(correct_config_text(normalise(str(line)))[0])
        manifest.append({"key": f"deviation:{key}", "must_contain": must})
    for entry in config.get("known_asymmetries") or []:
        key = str(entry.get("id"))
        manifest.append({"key": f"asymmetry:{key}",
                         "must_contain": [
                             f"`{key}`",
                             correct_config_text(
                                 normalise(str(entry.get("text") or "")))[0]]})
    for index, entry in enumerate(config.get("known_limitations") or []):
        manifest.append({"key": f"limitation:{index}",
                         "must_contain": [
                             f"`known_limitations[{index}]`",
                             correct_config_text(normalise(str(entry)))[0]]})
    for key, text in D10_13_REQUIRED:
        manifest.append({"key": f"d10.13:{key}",
                         "must_contain": [f"`{key}`", normalise(text)]})
    return manifest


def assert_coverage(markdown: str, manifest: Sequence[Mapping[str, Any]]
                    ) -> None:
    """Refuse the render if any registered § 9 entry is missing (D10.13).

    The complete normalised entry, consequence clause included, must appear
    inside § 9 — not merely as a prefix, and not elsewhere in the document.
    """
    section = section_9_body(markdown)
    if section is None:
        raise RefusalError(
            "report § 9 coverage check failed: the section heading "
            f"{SECTION_9_HEADING!r} is absent, so no registered deviation, "
            "asymmetry or limitation can be shown to be restated")
    body = normalise(section)
    missing = []
    for entry in manifest:
        for fragment in entry["must_contain"]:
            if fragment and fragment not in body:
                missing.append((entry["key"], fragment))
                break
    if missing:
        detail = "; ".join(f"{key} (looked for {fragment!r})"
                           for key, fragment in missing)
        raise RefusalError(
            f"report § 9 coverage check failed: {len(missing)} registered "
            f"entries are missing from § 9 of the rendered report: {detail}")


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------

def header(stats: Mapping[str, Any], config_path: Path, out: Path,
           stats_path: Path, k_path: Path | None) -> list[str]:
    cohort = stats.get("cohort") or {}
    return [
        "# Sequence track — stage 2 report "
        "(sixteen configurations, two seeds, validation only)",
        "",
        f"Generated (declared timestamp, the only value that moves on a "
        f"re-render): **{stats.get('generated_at_utc')}** by "
        f"`{rel(Path(__file__))}` from the files of record. Every number in "
        "every table is read from a file; the narrative passages are "
        "authored text in the generator.",
        "",
        f"* statistics: `{rel(stats_path)}` "
        f"(sha256 `{sha256_file(stats_path)[:12]}…`)",
        f"* k selection: "
        + (f"`{rel(k_path)}` (sha256 `{sha256_file(k_path)[:12]}…`)"
           if k_path and Path(k_path).exists()
           else "**not written** — the D9 record is absent"),
        f"* config: `{rel(config_path)}` "
        f"(sha256 `{stats['config']['sha256'][:12]}…`)",
        f"* acceptance checks and every result block: `{rel(ACCEPTANCE)}`",
        "",
        "**Status: validation-only two-seed directional screen. "
        f"`cohort_status: {cohort.get('cohort_status')}`, "
        f"`cohort_scored: {str(cohort.get('cohort_scored')).lower()}`, "
        f"`advances: {cohort.get('advances')}`, "
        f"`provisional: {str(cohort.get('provisional')).lower()}`. "
        "No arm advances. No market claim. No LANDED verdict — two-seed "
        "evidence is provisional and can never be LANDED (invariant 9). "
        "The user's verdict is outstanding.**",
        "",
        "Statuses this stage can emit: "
        + ", ".join(f"`{s}`" for s in ALLOWED_STATUSES)
        + ". There is no advancement status.",
        "",
    ]


def section_question(config: Mapping[str, Any]) -> list[str]:
    purpose = normalise(str((config.get("experiment") or {}).get("purpose")
                            or ""))
    return [
        "## 1. Question",
        "",
        "Does any registered change to the within-innings sequence model "
        "improve teacher-forced ball-outcome prediction over the "
        "memory-less token MLP on the i7 identity frame — and if so, which "
        "mechanism explains it? Sixteen configurations were trained fresh "
        "under one protocol so that every contrast in the arm register is a "
        "matched pair: fixed decay, learned forgetting (FoX), "
        "participant-aligned history, relay-free wiring, recency and "
        "ownership masks across five window sizes, two recurrent arms, and "
        "two residual arms over the production ball model.",
        "",
        f"Registered purpose, verbatim from the config: {purpose}",
        "",
        "Sign convention: "
        + normalise(str((config.get("experiment") or {}).get(
            "sign_convention") or "")),
        "",
    ]


def section_arms(config: Mapping[str, Any], stats: Mapping[str, Any],
                 dependency: Sequence[Mapping[str, Any]],
                 certificates: Mapping[str, Any] | None = None) -> list[str]:
    if certificates is None:
        certificates = dependency_certificates(dependency)
    training = config.get("training") or {}
    runs = stats.get("runs") or {}
    pin = stats.get("pin") or {}
    admission = stats.get("admission") or {}
    lines = [
        "## 2. Arms and settings",
        "",
        "One training block for every arm, identical to stage 1's so the two "
        "stages' protocols are comparable: "
        f"device `{training.get('device')}`, seeds {training.get('seeds')}, "
        f"d_model {training.get('dmodel')}, {training.get('layers')} layers, "
        f"{training.get('heads')} heads, batch {training.get('batch')}, "
        f"{training.get('epochs')} epochs, lr "
        f"{training.get('learning_rate')}, patience "
        f"{training.get('patience')}, aux {training.get('aux')}, "
        f"no-kit {training.get('no_kit')}, "
        f"score_test {training.get('score_test')}. Frame "
        f"`{(config.get('data') or {}).get('directory')}` "
        f"({(config.get('data') or {}).get('n_features')} features, role "
        f"`{(config.get('data') or {}).get('frame_role')}`), cache role "
        f"`{(config.get('data') or {}).get('stats_cache_role')}`. Stage 1 "
        "checkpoints are **not** reused (D2.2).",
        "",
        "| config id | arm | k | history input | wiring | access "
        "feat/hist/identity/prod | reference | seeds admitted | "
        "parameters |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for entry in config.get("configurations") or []:
        config_id = str(entry["id"])
        block = runs.get(config_id) or {}
        admitted = block.get("admitted_seeds") or []
        params = None
        for row in block.get("seeds") or []:
            value = (row.get("arm_params") or {}).get("n_parameters")
            if value is not None:
                params = value
        access = entry.get("access") or {}
        access_text = "/".join(
            "y" if access.get(key) else "n"
            for key in ("features", "history", "identity", "prod_logits"))
        lines.append(
            f"| `{config_id}` | `{entry.get('arm')}` | "
            f"{text_or_dash((entry.get('params') or {}).get('k'))} | "
            f"{text_or_dash(entry.get('history_input'))} | "
            f"{text_or_dash(entry.get('wiring'))} | {access_text} | "
            f"{', '.join(f'`{r}`' for r in entry.get('reference') or []) or DASH} | "
            f"{admitted if admitted else '**none**'} | "
            f"{text_or_dash(params)} |")

    incomplete = [config_id for config_id, block in runs.items()
                  if not block.get("complete_paired_seeds")]
    lines += [
        "",
        ("Every configuration has both registered seeds."
         if not incomplete else
         "**Incomplete configurations** (a reported contrast requires both "
         "seeds for every member, D8.9): "
         + ", ".join(f"`{c}`" for c in sorted(incomplete)) + "."),
        "",
        "`arm_params` is read verbatim from each run's `metrics.json`; the "
        "recurrent arms carry extra `cell` and `simplifications` keys beyond "
        "the standard block and those are displayed as recorded.",
        "",
        "### Machine provenance per run (D8.8)",
        "",
        "Seed 7 trained on the laptop and seed 13 on the Mac mini, so "
        "**machine is confounded with seed**. Every registered contrast is "
        "within-machine at each seed, so an additive machine effect cancels "
        "inside it; arm-by-machine interaction remains inseparable from seed "
        "variation and **no machine term is fitted**.",
        "",
        "| config id | seed | machine | host | chip | os | torch | device / "
        "MPS | thread caps | wall s | reconstructed val LL | summary.yaml "
        "val LL |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for config_id, block in runs.items():
        for row in block.get("seeds") or []:
            prov = row.get("provenance") or {}
            if not row.get("admitted"):
                lines.append(
                    f"| `{config_id}` | {row.get('seed')} | — | — | — | — | "
                    f"— | — | — | — | — | — |   <!-- "
                    f"{text_or_dash(row.get('reason'))} -->")
                continue
            lines.append(
                f"| `{config_id}` | {row.get('seed')} | "
                f"{text_or_dash(prov.get('machine'))} | "
                f"{text_or_dash(prov.get('hostname'))} | "
                f"{text_or_dash(prov.get('chip'))} | "
                f"{text_or_dash(prov.get('os'))} | "
                f"{text_or_dash(prov.get('torch_version'))} | "
                f"{text_or_dash(prov.get('mps_backend'))} | "
                f"{thread_caps(prov.get('thread_caps'))} | "
                f"{text_or_dash(prov.get('wall_seconds'))} | "
                f"{f6(row.get('reconstructed_validation_ll'))} | "
                f"{f6(row.get('summary_yaml_validation_ll'))} |")
    lines += [
        "",
        "The reconstructed column is "
        + normalise(str((stats.get("contract") or {}).get(
            "reconstructed_ll_label") or ""))
        + " — the `summary.yaml` value is the number of record for D9 and is "
        "never replaced by the reconstruction.",
        "",
    ]
    if any(not (row.get("provenance") or {}).get("recorded")
           for block in runs.values() for row in block.get("seeds") or []
           if row.get("admitted")):
        lines += [
            "Some admitted runs carry no `machine_provenance` block in their "
            "`run_record.json`; those cells read `–` rather than a guess, and "
            "nothing is reconstructed after the fact.",
            "",
        ]

    lines += [
        "### Admission and provenance verification",
        "",
        "No number in this report comes from a run that was admitted on file "
        "existence alone. Every run passed a read-only admission verifier "
        "first: its `metrics.json`, `run_record.json` and `COMPLETE.json` all "
        "name this configuration id and this seed, the arm and `k` match "
        "registration, a `training_signature` is present and identical in the "
        "run record and the completion record, and the completion record's "
        "per-artefact size and md5 manifest still matches the files on disk. "
        "Duplicate seed rows are rejected, never deduplicated.",
        "",
        f"* Pin available: **{flag(pin.get('available'))}**"
        + (f" (config body sha256 `{str(pin.get('config_body_sha256'))[:12]}…`)"
           if pin.get("available") else f" — {pin.get('reason')}"),
        f"* Validation parquet md5 `{stats['frame'].get('parquet_md5')}`, "
        f"pinned `{pin.get('validation_parquet_md5')}`; "
        f"{stats['frame']['n_rows']} rows, pinned "
        f"{pin.get('validation_parquet_rows')}. "
        + normalise(str(pin.get("verified") or "")),
        f"* {normalise(str(admission.get('rule') or ''))}",
        f"* {normalise(str(admission.get('arm_specific_note') or ''))}",
        "",
        "| configuration | seed-independent training signature |",
        "|---|---|",
    ]
    signatures = admission.get("training_signature_by_config") or {}
    if not signatures:
        lines.append("| *no configuration has an admitted run* | — |")
    for config_id, signature in signatures.items():
        lines.append(f"| `{config_id}` | `{str(signature)[:16]}…` |")
    lines += [
        "",
        "Shared signature components, identical across every admitted "
        "configuration: "
        + (", ".join(f"`{name}` = `{str(digest)[:12]}…`"
                     for name, digest
                     in (admission.get("shared_components") or {}).items())
           or "*none recorded*")
        + ".",
        "",
        dependency_heading(certificates),
        "",
        "| arm | k | scored against S(i) of | checkpoint | seed | n targets | "
        "max |Δ| | p99 |Δ| | n > 1e-6 | checkpoint md5 | role | result |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    if not dependency:
        lines.append("| — | — | — | *no dependency JSON found* | — | — | — | "
                     "— | — | — | — | — |")
    auth_by_checkpoint = {
        entry["checkpoint"]: entry
        for block in certificates["by_config"].values()
        for entry in block["authentication"]}
    for record in dependency:
        role = ("positive control" if record.get("positive_control_expected")
                else "masked arm")
        if record.get("positive_control_expected"):
            result = ("SEES EXCLUDED-PAST INFORMATION"
                      if record.get("positive_control_observed")
                      else "**CONTROL DID NOT FIRE**")
        else:
            result = "PASS" if record.get("pass") else "**FAIL**"
        checkpoint = str(record.get("checkpoint") or "")
        note = " (smoke checkpoint)" if "/smoke/" in checkpoint else ""
        auth = auth_by_checkpoint.get(Path(checkpoint).as_posix())
        md5_cell = (f"`{auth['status']}`" if auth
                    else "`NOT_A_REQUIRED_MASKED_ARM`")
        lines.append(
            f"| `{record.get('arm')}` | {text_or_dash(record.get('k'))} | "
            f"`{text_or_dash(record.get('dependency_set_arm'))}` "
            f"k={text_or_dash(record.get('dependency_set_k'))} | "
            f"`{Path(checkpoint).name}`{note} | "
            f"{text_or_dash(_checkpoint_seed(checkpoint))} | "
            f"{text_or_dash(record.get('n_targets'))} | "
            f"{sci(record.get('max_abs_delta'))} | "
            f"{sci(record.get('p99_abs_delta'))} | "
            f"{text_or_dash(record.get('n_nonzero_gt_1e-6'))} | "
            f"{md5_cell} | {role} | {result} |")
    lines += [
        "",
        "The positive controls establish sensitivity to excluded-past "
        "information, not specifically multi-layer relay (Astra gate 1, "
        "D7.4). A relay-free certificate always uses its own S(i); a control "
        "is a standard-wiring arm scored against the **masked arm's** S(i), "
        "which is why it is matched through the `scored against S(i) of` "
        "column and never through its own arm and k (Astra gate 2 round 1 "
        "MUST-FIX 1). Where a row is marked *smoke checkpoint* the "
        "certificate is the structural one from the one-epoch smoke weights; "
        "masking is a property of the architecture, and that row is never "
        "counted as trained-checkpoint coverage. Checkpoints inside the "
        "closed smoke tree are not opened by this report, so their recorded "
        "md5 reads `NOT_AUTHENTICATED_CLOSED_TREE`. A smoke record can never "
        "carry an arm's eligibility either: an arm whose trained "
        "recertifications are absent reads `TRAINED_SEEDS_INCOMPLETE` and is "
        "blocked, even when the smoke record is the only certificate present "
        "(Astra gate 2 round 2).",
        "",
        "Trained-checkpoint coverage is verified, not asserted: it requires, "
        "for each of the four masked configurations, a passing certificate at "
        "**both** registered training seeds "
        + ", ".join(str(s) for s in certificates["required_checkpoint_seeds"])
        + ", each certificate's recorded checkpoint md5 recomputed from "
        "`model.pt` on disk and matching, the registered perturbation "
        "settings ("
        + ", ".join(f"{field} {value!r}" for field, value
                    in certificates["registered_settings"].items())
        + "), and the registered matched positive control present and fired. "
        "Coverage complete: **"
        + flag(certificates["trained_checkpoint_coverage_complete"])
        + "**"
        + (" — " + ", ".join(f"`{name}`" for name
                             in certificates["trained_checkpoint_coverage"])
           if certificates["trained_checkpoint_coverage"]
           else " — **no configuration has it**")
        + ".",
        "",
        "| configuration | certificate | trained-checkpoint coverage | "
        "matched positive control | consequence |",
        "|---|---|---|---|---|",
    ]
    for config_id in certificates["required"]:
        block = certificates["by_config"][config_id]
        consequence = (
            f"family forced to `{DEPENDENCY_BLOCKED_STATUS}`; arm not eligible"
            if config_id in certificates["blocked"]
            else "no block from this check")
        if not block["control_registered"]:
            control = ("none registered (D7.4 registers controls only against "
                       "`recency` k=30 and `same_entity` k=30)")
        elif not block["controls"]:
            control = (f"`{block['registered_control_arm']}` **absent**")
        else:
            control = "; ".join(
                f"`{entry['arm']}` vs S(i) of {entry['dependency_set']} → "
                + ("fired" if entry["fired"] else "**did not fire**")
                + ("" if entry["arm_matches_registration"]
                   else f" (**not the registered `{entry['expected_arm']}`**)")
                for entry in block["controls"])
        lines.append(f"| `{config_id}` | `{block['status']}` | "
                     f"{block['coverage']} | {control} | {consequence} |")
    lines.append("")
    if certificates["configurations_without_a_registered_control"]:
        lines += [
            "**Disclosed gap in the certification design.** "
            + ", ".join(f"`{name}`" for name in certificates[
                "configurations_without_a_registered_control"])
            + " have no separately registered positive control: D7.4 "
            "registers `full` against `recency_k30`'s S(i) and "
            "`aligned_hist` against `same_entity_k30`'s S(i) only. Their "
            "sensitivity evidence is inherited from the same-entity "
            "construction at k = 30 and is not independent of it. That is "
            "recorded here rather than treated as satisfied.",
            "",
        ]
    if certificates["unmatched_controls"]:
        lines += [
            "**Positive-control records that match no registered dependency "
            "set** (they certify nothing here and are not counted): "
            + ", ".join(f"`{entry['arm']}` vs {entry['dependency_set']}"
                        for entry in certificates["unmatched_controls"])
            + ".",
            "",
        ]
    if certificates["blocked"]:
        lines += ["A failed, missing, unauthenticated or seed-incomplete "
                  "masked-arm certificate, or a missing or silent registered "
                  "positive control, **blocks** the affected interpretation "
                  "and eligibility; it does not merely appear in the table "
                  "above (Astra gate 1 round 3; gate 2 round 1 MUST-FIX 1). "
                  "Blocked: "
                  + ", ".join(f"`{name}`"
                              for name in sorted(certificates["blocked"]))
                  + ".",
                  "",
                  "The statistics JSON was written before this check and "
                  "retains its own eligibility statuses for those arms; where "
                  "the two differ, **the block in this report is the "
                  "operative disposition** and the statistics status is "
                  "superseded, not the other way round.",
                  ""]
    return lines


def section_rule(config: Mapping[str, Any],
                 stats: Mapping[str, Any]) -> list[str]:
    statistics = config.get("statistics") or {}
    families = statistics.get("families") or {}
    holm = families.get("holm") or {}
    bootstrap = statistics.get("bootstrap") or {}
    non_inf = statistics.get("non_inferiority") or {}
    contract = stats.get("contract") or {}
    k_rule = statistics.get("k_selection") or {}
    lines = [
        "## 3. Decision rule (as registered, before any result was read)",
        "",
        f"* **Families.** {normalise(str(families.get('definition') or ''))}",
        f"* **Gate reference.** "
        f"{normalise(str(families.get('gate_reference_rule') or ''))}",
        f"* **Multiplicity.** {normalise(str(holm.get('scope') or ''))} "
        f"{normalise(str(holm.get('adjustment') or ''))} Ties: "
        f"{normalise(str(holm.get('ties') or ''))}. Missing member: "
        f"{normalise(str(holm.get('missing_member') or ''))}",
        f"* **Raw p.** {normalise(str(contract.get('p_convention') or ''))}",
        f"* **Non-inferiority.** margin "
        f"{non_inf.get('margin_ll')} log loss on "
        f"{non_inf.get('applies_to')} against `{non_inf.get('reference')}`; "
        f"{normalise(str(non_inf.get('sign') or ''))}",
        f"* **Bootstrap.** {bootstrap.get('reps')} replicates, rng seed "
        f"{bootstrap.get('seed')}, {bootstrap.get('weighting')}, "
        f"{bootstrap.get('resample')}, blocks from "
        f"`{bootstrap.get('blocks_from')}` over "
        f"`{bootstrap.get('block_source_dir')}`, contract "
        f"`{contract.get('bootstrap_contract_version')}` with "
        f"`MAX_EVENT_GAP_DAYS = {contract.get('max_event_gap_days')}`. "
        f"Fewer than {bootstrap.get('min_blocks')} blocks on a slice is "
        f"{bootstrap.get('below_min_blocks')}. "
        f"{contract.get('resampled_unit_passed_to_estimator', 'block ids')} "
        "are what reach the cluster estimators, never match ids.",
        f"* **Estimands.** (i) each registered seed checkpoint separately "
        "with paired block-only uncertainty; (ii) the arithmetic across-seed "
        "mean under joint seed-and-block resampling. Reported separately, "
        "never pooled, never selected between. No best-seed selection and no "
        "averaging of CI endpoints.",
        f"* **k selection.** tolerance {k_rule.get('tolerance_ll')} over the "
        f"registered sweep {k_rule.get('registered_sweep')}: "
        f"{normalise(str(k_rule.get('rule') or ''))}",
        f"* **Intervals.** "
        f"{normalise(str((statistics.get('interval_labelling') or {}).get('rank_local') or ''))}",
        f"* **Not imported.** "
        f"{normalise(str(non_inf.get('not_imported_from_stage_1') or ''))}",
        "",
        "Registered validation totals, asserted before anything was "
        f"differenced: {stats['frame']['n_rows']} rows / "
        f"{stats['blocks']['n_matches']} matches / "
        f"{stats['blocks']['n_blocks']} tournament blocks / "
        f"{stats['blocks']['unmapped']} unmapped, block lookup sha256 "
        f"`{stats['blocks']['lookup_sha256'][:12]}…` over "
        f"`{stats['blocks']['lookup_source']}`. Row alignment rule: "
        + normalise(str((stats.get("alignment") or {}).get("rule") or ""))
        + ".",
        "",
        "### Slice predicates (frozen before computation, D10.7)",
        "",
        "| slice | role | predicate | rows | matches | blocks | "
        "descriptive (<10 blocks) |",
        "|---|---|---|---|---|---|---|",
    ]
    slice_stats = (stats.get("slices") or {}).get("stats") or {}
    for predicate in (stats.get("slices") or {}).get("predicates") or []:
        name = predicate["slice"]
        block = slice_stats.get(name) or {}
        if not predicate.get("available"):
            lines.append(
                f"| `{name}` | {predicate.get('role')} | **unavailable** | — "
                "| — | — | — |")
            continue
        lines.append(
            f"| `{name}` | {predicate.get('role')} | "
            f"`{predicate.get('predicate')}` | {block.get('n_rows')} | "
            f"{block.get('n_matches')} | {block.get('n_blocks')} | "
            f"{flag(block.get('descriptive'))} |")
    for predicate in (stats.get("slices") or {}).get("predicates") or []:
        if not predicate.get("available"):
            lines += ["",
                      f"`{predicate['slice']}` is reported **unavailable**: "
                      + normalise(str(predicate.get("unavailable_reason")
                                      or ""))]
    lines += ["", ]
    return lines


def _holm_table(family: Mapping[str, Any], readout: str,
                blocked: Mapping[str, str] | None = None) -> list[str]:
    table = (family.get("holm") or {}).get(readout) or {}
    candidate = family["candidate"]
    computed = (family.get("screen") or {}).get(readout, {}).get("status")
    is_blocked = candidate in (blocked or {})
    status = DEPENDENCY_BLOCKED_STATUS if is_blocked else computed
    lines = [
        f"**{candidate}** — {readout_label(readout)}, "
        f"Holm group `{family.get('holm_group')}`, screen status "
        f"`{status}`"
        + (f" (forced from `{computed}` by a failed or missing dependency "
           "certificate)" if is_blocked else ""),
        "",
        "| member | contrast | slice | t | point | 95% interval | U95 | "
        "raw p | Holm p | rank | rank-local interval | rejected | status |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in table.get("members") or []:
        lines.append(
            f"| {row['member']} | `{row['contrast']}` | `{row['slice']}` | "
            f"{row['threshold']} | {f5(row.get('point'))} | "
            f"{ci(row.get('ci95'))} | {f5(row.get('u95'))} | "
            f"{text_or_dash(row.get('p_display'))} | {f4(row.get('p_holm'))} | "
            f"{row.get('rank')} | {ci(row.get('rank_local_interval'))} | "
            f"{flag(row.get('rejected'))} | `{row.get('status')}` |")
    placeholders = [row["member"] for row in table.get("members") or []
                    if row.get("placeholder_non_rejecting")]
    if placeholders:
        lines += ["",
                  "Non-rejecting placeholders (unavailable or descriptive "
                  "members, which cannot shrink the family and cannot let it "
                  "pass): " + ", ".join(placeholders) + "."]
    lines.append("")
    return lines


def section_results(stats: Mapping[str, Any],
                    blocked: Mapping[str, str] | None = None) -> list[str]:
    blocked = dict(blocked or {})
    lines = [
        "## 4. Results — the 15 registered families, Holm step-down",
        "",
        "Each family has exactly three members: the registered primary "
        "contrast on `all`, and `candidate − mlp` non-inferiority gates on "
        "`death` and on `chase`. Holm runs **within** a family and "
        "**separately per estimand**; it is never pooled across the 15 "
        "families and never applied across the k search, and no correction "
        "across families is claimed. Rank-local intervals are "
        + normalise(str((stats.get("contract") or {}).get("rank_local_note")
                        or ""))
        + ": classification is never read off one.",
        "",
        "Rejection requires all three of the step-down rule, the favourable "
        "direction, and the relevant strict upper-bound condition (U95 < 0 "
        f"for a primary, U95 < {MARGIN_LL} for a gate). An empty bootstrap "
        "tail is reported as `< 0.001` with a resolution flag, never as "
        "exact zero.",
        "",
    ]
    for family in stats.get("families") or []:
        if family["candidate"] in blocked:
            lines += [blocking_note(family["candidate"], blocked), ""]
        for readout in readouts_of(stats):
            lines += _holm_table(family, readout, blocked)
        note = family.get("note")
        if note:
            lines += [f"Registered note on `{family['candidate']}`: "
                      + normalise(str(note)), ""]
    lines += [
        "Family members are registered **for confirmation**, but their "
        "validation results remain screening evidence; every other contrast "
        "and every other slice is exploratory.",
        "",
    ]
    return lines


def section_estimands(stats: Mapping[str, Any]) -> list[str]:
    contract = stats.get("contract") or {}
    lines = [
        "## 5. The two estimands, seed spread, and what (ii) is not",
        "",
        "Estimand (i) is one seed checkpoint with paired block-only "
        "uncertainty. Estimand (ii) is the arithmetic across-seed mean under "
        "joint resampling: each replicate draws S seed indices with "
        "replacement, then B block indices with replacement, applies the "
        "same sampled seeds and blocks to both arms, and forms summed "
        "sampled losses over (S × summed sampled block row counts). It is "
        "**not** the log loss of averaged probabilities. Tonight's (ii) is "
        + normalise(str(contract.get("estimand_ii_label") or "")) + ".",
        "",
        "| contrast | slice | seed-7 point | seed-13 point | seed range | "
        "favourable seeds | mean point | mean 95% interval |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, record in (stats.get("contrasts") or {}).items():
        if not str(record.get("role", "")).startswith("family_"):
            continue
        spread = record.get("seed_spread") or {}
        points = record.get("per_seed_points") or {}
        joint = record.get("estimand_ii") or {}
        lines.append(
            f"| `{record['candidate']} − {record['reference']}` | "
            f"`{record['slice']}` | {f5(points.get('7'))} | "
            f"{f5(points.get('13'))} | {f5(spread.get('range'))} | "
            f"{text_or_dash(record.get('favourable_direction_count'))}"
            f"/{text_or_dash(record.get('n_seeds'))} | "
            f"{f5(joint.get('point'))} | {ci(joint.get('ci95'))} |")
    if not any(str(r.get("role", "")).startswith("family_")
               and r.get("available")
               for r in (stats.get("contrasts") or {}).values()):
        lines.append("| *no family contrast is evaluable* | — | — | — | — | "
                     "— | — | — |")
    lines += [
        "",
        "Seed ranges are empirical spreads over two seeds, not confidence "
        "intervals. No seed is selected and no CI endpoint is averaged.",
        "",
    ]
    return lines


def section_ksweep(record: Mapping[str, Any] | None) -> list[str]:
    lines = ["## 6. The k sweep (D9)", ""]
    if record is None:
        lines += ["The D9 selection record has not been written, so no k is "
                  "selected and no k is defaulted.", ""]
        return lines
    lines += [
        "Numbers of record: " + normalise(str(record.get(
            "numbers_of_record") or "")) + ". The k-to-configuration mapping "
        "is " + normalise(str(record.get("k_to_config_id_source") or ""))
        + ": "
        + ", ".join(f"k {k} → `{config_id}`" for k, config_id
                    in (record.get("k_to_config_id") or {}).items())
        + ".",
        "",
        f"Rule: {normalise(str(record.get('rule') or ''))} "
        f"Tie rule: {normalise(str(record.get('tie_rule') or ''))}",
        "",
        f"**Selection: `{record.get('selection')}`"
        + (f" → `{record.get('selected_config_id')}` "
           f"(k = {record.get('selected_k')})**"
           if record.get("selected_config_id") else "**"),
        "",
        "| k | config id | seed 7 LL | seed 13 LL | mean | min | max | "
        "range | paired mean vs k30 | paired range | favourable seeds |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    paired = record.get("paired_vs_k30") or {}
    for row in record.get("rows") or []:
        block = paired.get(row["k"]) or {}
        per_seed = row.get("per_seed") or {}
        lines.append(
            f"| {row['k']} | `{row['config_id']}` | "
            f"{f6(per_seed.get('7'))} | {f6(per_seed.get('13'))} | "
            f"{f6(row.get('mean_ll'))} | {f6(row.get('min_ll'))} | "
            f"{f6(row.get('max_ll'))} | {f6(row.get('range_ll'))} | "
            f"{f5(block.get('mean_delta'))} | "
            f"{f5(block.get('range_delta'))} | "
            f"{text_or_dash(block.get('favourable_direction_count'))}"
            f"/{text_or_dash(block.get('n_seeds'))} |")
    rejections = record.get("admission_rejections") or []
    if rejections:
        lines += ["",
                  "Summaries rejected by the admission verifier before their "
                  "log losses were read: "
                  + "; ".join(normalise(str(r)) for r in rejections) + "."]
    if record.get("selection") == "BLOCKED_INCOMPLETE":
        lines += ["",
                  "**BLOCKED_INCOMPLETE**: "
                  + ", ".join(f"`{c}`" for c in record.get("blocked_on") or [])
                  + " lack both registered seeds. "
                  + normalise(str(record.get("note") or ""))]
    else:
        lines += ["",
                  f"Best mean is k = {record.get('best_mean_k')}; margin "
                  f"against k = 30 is {f6(record.get('margin_vs_k30'))} "
                  f"against a tolerance of {record.get('tolerance_ll')}. "
                  "Not selected: "
                  + ", ".join(f"`{c}`" for c in record.get("not_selected")
                              or []) + "."]
        flags = record.get("flags") or []
        lines += ["",
                  ("Interpretation flags: " + "; ".join(flags) + "."
                   if flags else
                   "No interpretation flag fired for this selection.")]
    lines += [
        "",
        normalise(str(record.get("interpretation_guard") or ""))
        + f" The whole sweep is labelled **{record.get('labelling')}**.",
        "",
        normalise(str(record.get("matched_control_constraint") or "")),
        "",
        normalise(str(record.get("provisional_note") or "")),
        "",
    ]
    return lines


def section_mechanism(stats: Mapping[str, Any],
                      blocked: Mapping[str, str] | None = None) -> list[str]:
    """Astra gate 2 round 1 MUST-FIX 1: blocking reaches this section too.

    A dependency block previously reached the results and gate sections only,
    so a mechanism contrast whose endpoint was `NOT_EVALUABLE` still read as an
    ordinary interval here.
    """
    blocked = dict(blocked or {})
    lines = [
        "## 7. Mechanism contrasts (D10.5)",
        "",
        "| contrast | what it measures (registered label) | inferential in a "
        "registered family | seed-7 point | seed-13 point | mean point | "
        "mean 95% interval | mean raw p | disposition |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    blocked_contrasts: list[str] = []
    for entry in stats.get("mechanism_contrasts") or []:
        record = entry.get("record") or {}
        points = record.get("per_seed_points") or {}
        joint = record.get("estimand_ii") or {}
        endpoints = [str(entry["candidate"]), str(entry["reference"])]
        hit = [name for name in endpoints if name in blocked]
        name = f"`{entry['candidate']} − {entry['reference']}`"
        if hit:
            blocked_contrasts.append(name)
            disposition = (f"`{DEPENDENCY_BLOCKED_STATUS}` — dependency "
                           "certificate blocks "
                           + ", ".join(f"`{n}`" for n in hit))
        elif not record.get("available"):
            disposition = f"`{DEPENDENCY_BLOCKED_STATUS}` — member unavailable"
        else:
            disposition = "reported as computed"
        lines.append(
            f"| {name} | "
            f"{normalise(str(entry.get('registered_label') or ''))} | "
            f"{flag(entry.get('inferential_in_a_registered_family'))} | "
            f"{f5(points.get('7'))} | {f5(points.get('13'))} | "
            f"{f5(joint.get('point'))} | {ci(joint.get('ci95'))} | "
            f"{text_or_dash(joint.get('p_display'))} | {disposition} |")
    unavailable = [f"`{e['candidate']} − {e['reference']}`"
                   for e in stats.get("mechanism_contrasts") or []
                   if not (e.get("record") or {}).get("available")]
    lines += [""]
    if blocked_contrasts:
        lines += [
            "**Blocked mechanism contrasts.** "
            + ", ".join(blocked_contrasts)
            + f" are `{DEPENDENCY_BLOCKED_STATUS}`: an endpoint's masked-arm "
            "dependency certificate is failed, missing, unauthenticated, "
            "seed-incomplete, or lacks its registered positive control, so "
            "the arm has not been shown to depend only on the rows its mask "
            "allows and the interval in that row supports no mechanism "
            "reading whatever its value (D10.12; Astra gate 2 round 1 "
            "MUST-FIX 1).",
            "",
        ]
    if unavailable:
        lines += ["`NOT_EVALUABLE` mechanism contrasts (a member is missing "
                  "or the slice is descriptive): " + ", ".join(unavailable)
                  + ".", ""]
    guard = next((e.get("guard") for e in stats.get("mechanism_contrasts")
                  or [] if e.get("guard")), "")
    lines += [normalise(str(guard)), "",
              "No comparative claim in this report is derived from one arm "
              "being significant against `mlp` while another is not. That "
              "inference is not computed and is not reportable.", ""]
    return lines


def _equivalent_slices(stats: Mapping[str, Any]) -> list[tuple[str, str, int]]:
    """Registered slice pairs that select the same row set on this frame.

    Astra gate 1 round 3 ruling: `innings_2` and `chase` are acceptable as two
    registered slices only if their equivalence is disclosed and they are never
    presented as independent corroboration. This finds the equivalence from the
    computed slice statistics rather than asserting it from memory, so a frame
    on which they diverge is reported as diverging.

    Astra gate 2 round 1 SHOULD 7: identity is read from the persisted mask
    digest (`mask_sha256`, the sha256 of the packed boolean row mask), not
    inferred from equal row, match and block counts — two different row sets
    can agree on all three. A statistics JSON written before the digest existed
    still yields the disclosure, because withholding it would be the less
    conservative error, but the basis is labelled `counts_only` in the report
    so no reader takes it for row-membership identity.
    """
    table = ((stats.get("slices") or {}).get("stats") or {})
    out = []
    for left, right in (("innings_2", "chase"),):
        a, b = table.get(left) or {}, table.get(right) or {}
        if not (a.get("available") and b.get("available")):
            continue
        if a.get("mask_sha256") and b.get("mask_sha256"):
            if a["mask_sha256"] == b["mask_sha256"]:
                out.append((left, right, int(a.get("n_rows") or 0),
                            "mask_sha256"))
            continue
        if (a.get("n_rows") == b.get("n_rows")
                and a.get("n_matches") == b.get("n_matches")
                and a.get("n_blocks") == b.get("n_blocks")):
            out.append((left, right, int(a.get("n_rows") or 0),
                        "counts_only"))
    return out


def section_gates(stats: Mapping[str, Any],
                  blocked: Mapping[str, str] | None = None) -> list[str]:
    blocked = dict(blocked or {})
    lines = [
        "## 8. Non-inferiority gates, exploratory slices and the residual "
        "readouts",
        "",
        "### Gates (D10.6)",
        "",
        f"Paired harm `d = LL(candidate) − LL(mlp)` on each gate slice, "
        f"margin +{MARGIN_LL}. An unadjusted numerical pass requires "
        f"**strictly U95 < {MARGIN_LL}**; a family-adjusted gate "
        "additionally requires favourable Holm rejection against the "
        f"+{MARGIN_LL} boundary, at least 10 blocks and complete paired "
        "seeds. Neither a point below the margin nor a failure to detect "
        "harm establishes non-inferiority.",
        "",
        "| candidate | slice | readout | point | 95% interval | U95 | "
        f"U95 < {MARGIN_LL} | raw p | Holm p | status |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for gate in stats.get("gates") or []:
        for readout in readouts_of(stats):
            values = dict((gate.get("readouts") or {}).get(readout) or {})
            if gate["candidate"] in blocked:
                values["status"] = DEPENDENCY_BLOCKED_STATUS
            lines.append(
                f"| `{gate['candidate']}` | `{gate['slice']}` | "
                f"{readout_label(readout)} | "
                f"{f5(values.get('point'))} | {ci(values.get('ci95'))} | "
                f"{f5(values.get('u95'))} | "
                f"{flag(values.get('u95_strictly_below_margin'))} | "
                f"{text_or_dash(values.get('p_display'))} | "
                f"{f4(values.get('p_holm'))} | `{values.get('status')}` |")
    if not stats.get("gates"):
        lines.append("| *no gate is registered* | — | — | — | — | — | — | — "
                     "| — | — |")
    if blocked:
        lines += ["",
                  "Gate rows for " + ", ".join(f"`{name}`" for name
                                               in sorted(blocked))
                  + f" read `{DEPENDENCY_BLOCKED_STATUS}` regardless of their "
                  "intervals: a failed or missing dependency certificate "
                  "blocks eligibility (D10.12)."]

    lines += [
        "",
        "### Exploratory slices (D10.7)",
        "",
        "Every reading below is exploratory: it changes no k, no family and "
        "no advancement, and this validation-only run opened no test rows.",
        "",]
    for left, right, n_rows, basis in _equivalent_slices(stats):
        basis_text = (
            "identity is established by an identical persisted row-mask "
            "sha256 (Astra gate 2 round 1 SHOULD 7)"
            if basis == "mask_sha256" else
            "**identity here is inferred from equal row, match and block "
            "counts only**, because this statistics JSON predates the "
            "persisted row-mask digest; two different row sets could in "
            "principle agree on all three, so treat this as a conservative "
            "disclosure rather than proof of identical membership")
        lines += [
            f"**Disclosure — `{left}` and `{right}` are the same rows on this "
            f"frame.** Both predicates select {n_rows} rows — "
            + basis_text
            + " — so the two readouts are one "
            "readout written twice. They are **never** independent "
            f"corroboration of each other, and only `{right}` is a "
            "three-member family member; the other is exploratory (Astra "
            "gate 1 round 3 ruling).",
            "",]
    lines += [
        "| contrast | slice | mean point | mean 95% interval | blocks | "
        "descriptive |",
        "|---|---|---|---|---|---|",
    ]
    exploratory = [record for record in (stats.get("contrasts") or {}).values()
                   if record.get("role") == "exploratory"
                   and record.get("slice") != "all"]
    for record in exploratory:
        joint = record.get("estimand_ii") or {}
        block = record.get("slice_stats") or {}
        lines.append(
            f"| `{record['candidate']} − {record['reference']}` | "
            f"`{record['slice']}` | {f5(joint.get('point'))} | "
            f"{ci(joint.get('ci95'))} | {text_or_dash(block.get('n_blocks'))} "
            f"| {flag(block.get('descriptive'))} |")
    if not exploratory:
        lines.append("| *no exploratory slice readout is available* | — | — | "
                     "— | — | — |")

    residual = stats.get("residual") or {}
    base = residual.get("base_only") or {}
    lines += [
        "",
        "### Residual readouts (D10.11)",
        "",
        f"The qualifying primary is **{residual.get('qualifying_primary')}**. "
        + normalise(str(residual.get(
            "residual_t1_minus_mlp_cannot_qualify") or ""))
        + " `residual_mlp` is "
        + normalise(str(residual.get("residual_mlp_role") or "")) + ".",
        "",
    ]
    increment = (stats.get("contrasts") or {}).get(
        "residual_t1-residual_mlp@all") or {}
    joint = increment.get("estimand_ii") or {}
    if increment.get("available") and joint.get("ci95"):
        if joint.get("ci_clean_favourable"):
            reading = ("the interval excludes zero favourably, so an "
                       "incremental benefit of residual T1 over residual MLP "
                       "is established at this resolution")
        elif joint.get("l95") is not None and float(joint["l95"]) > 0:
            reading = ("the interval excludes zero adversely, so residual T1 "
                       "is worse than residual MLP at this resolution")
        else:
            reading = ("the interval straddles zero, so **no incremental "
                       "benefit of residual T1 over residual MLP was "
                       "established at this resolution**. That is an "
                       "unresolved interval, **not** a demonstration that "
                       "sequence adds nothing over the production prior "
                       "(Astra gate 2 round 1 MUST-FIX 4)")
        lines += [
            f"`residual_t1 − residual_mlp` on `all` reads "
            f"{f5(joint.get('point'))} {ci(joint.get('ci95'))}: " + reading
            + ".",
            "",
        ]
    if base.get("available"):
        provenance = base.get("provenance") or {}
        lines += [
            f"Base-only validation log loss (the production prior alone, "
            f"exploratory reference): **{f6(base.get('row_ll_mean'))}** over "
            f"{base.get('n_rows')} rows, from "
            f"`{base.get('npz')}` — booster md5 "
            f"`{str(provenance.get('booster_md5'))[:12]}…`, model dir "
            f"`{provenance.get('model_dir')}` (role "
            f"`{provenance.get('model_role')}`), parquet md5 "
            f"`{str(provenance.get('parquet_md5'))[:12]}…`, floor "
            f"{provenance.get('floor')}, renormalised "
            f"{provenance.get('renormalised')}. Row alignment: "
            + normalise(str(provenance.get("row_alignment") or "")) + ".",
            "",
            "| residual arm − base only | seed-7 point | seed-13 point | "
            "mean point | mean 95% interval |",
            "|---|---|---|---|---|",
        ]
        for record in residual.get("against_base_only") or []:
            points = record.get("per_seed_points") or {}
            joint = record.get("estimand_ii") or {}
            lines.append(
                f"| `{record['candidate']} − base_only` | "
                f"{f5(points.get('7'))} | {f5(points.get('13'))} | "
                f"{f5(joint.get('point'))} | {ci(joint.get('ci95'))} |")
        lines.append("")
    else:
        lines += [f"Base-only validation log loss is unavailable: "
                  + normalise(str(base.get("reason") or "")) + ".", ""]
    lines += ["These base-only comparisons are exploratory.", ""]
    return lines


def section_limitations(config: Mapping[str, Any]) -> list[str]:
    lines = [
        "## 9. Registered deviations, asymmetries, limitations "
        "(restated, none dropped)",
        "",
        "Every entry below is restated from "
        "`experiments/configs/seq_stage2_v1.yaml` or enumerated by D10.13, "
        "with its consequence. A coverage check in the generator fails the "
        "render if any one of them is missing.",
        "",
        "### Deviations (config `deviations`)",
        "",
    ]
    applied: list[tuple[str, dict[str, str]]] = []

    def corrected(source: str, text: str) -> str:
        fixed, hits = correct_config_text(normalise(text))
        applied.extend((source, hit) for hit in hits)
        return fixed

    for entry in config.get("deviations") or []:
        key = str(entry.get("id"))
        lines.append(f"- `{key}` — "
                     + corrected(f"deviations[{key}].statement",
                                 str(entry.get("statement") or ""))
                     + (" **Reason:** "
                        + corrected(f"deviations[{key}].reason",
                                    str(entry.get("reason") or ""))
                        if entry.get("reason") else ""))
        for line in entry.get("simplifications") or []:
            lines.append("    - " + corrected(
                f"deviations[{key}].simplifications", str(line)))
    lines += ["", "### Known asymmetries (config `known_asymmetries`)", ""]
    for entry in config.get("known_asymmetries") or []:
        key = str(entry.get("id"))
        lines.append(f"- `{key}` — "
                     + corrected(f"known_asymmetries[{key}].text",
                                 str(entry.get("text") or "")))
    lines += ["", "### Known limitations (config `known_limitations`)", ""]
    for index, entry in enumerate(config.get("known_limitations") or []):
        lines.append(f"- (`known_limitations[{index}]`) "
                     + corrected(f"known_limitations[{index}]", str(entry)))
    lines += ["",
              "### Additional dispositions D10.13 requires by name",
              ""]
    for key, text in D10_13_REQUIRED:
        lines.append(f"- `{key}` — {normalise(text)}")
    lines += [
        "",
        "A shared feature set does not prove a shared effect across "
        "architectures. Where a historical measurement is invalid it is "
        "labelled unavailable, not reconstructed.",
        "",
    ]
    if applied:
        lines += [
            "### Corrections applied to config-sourced wording above",
            "",
            "`experiments/configs/seq_stage2_v1.yaml` is launch-era training "
            "provenance and is not edited. Where it states a conclusion the "
            "evidence does not support, the clause is replaced here and the "
            "replacement is disclosed, original wording included.",
            "",
        ]
        seen: set[tuple[str, str]] = set()
        for source, correction in applied:
            token = (source, correction["old"])
            if token in seen:
                continue
            seen.add(token)
            lines.append(
                f"- `{source}` — rendered as \"{correction['new']}\" in place "
                f"of the config's \"{correction['old']}\". Reason: "
                f"{correction['reason']}.")
        lines.append("")
    return lines


def _best_observed(stats: Mapping[str, Any]) -> dict:
    """The lowest observed two-seed mean validation LL among admitted arms."""
    best = {"config_id": None, "mean_ll": None, "source": None,
            "per_seed": {}}
    for config_id, block in (stats.get("runs") or {}).items():
        if not block.get("complete_paired_seeds"):
            continue
        summary = [row.get("summary_yaml_validation_ll")
                   for row in block.get("seeds") or []]
        source = "summary.yaml"
        if any(value is None for value in summary):
            summary = [row.get("reconstructed_validation_ll")
                       for row in block.get("seeds") or []]
            source = "reconstructed from saved probabilities"
        if any(value is None for value in summary):
            continue
        mean = sum(float(v) for v in summary) / len(summary)
        if best["mean_ll"] is None or mean < best["mean_ll"]:
            best = {"config_id": config_id, "mean_ll": mean, "source": source,
                    "per_seed": {str(row.get("seed")): (
                        row.get("summary_yaml_validation_ll")
                        if source == "summary.yaml"
                        else row.get("reconstructed_validation_ll"))
                        for row in block.get("seeds") or []}}
    return best


def _sequence_gain(stats: Mapping[str, Any],
                   blocked: Mapping[str, str] | None = None) -> dict:
    """Which sequence arms beat mlp CI-clean on the all slice, and on what."""
    contrasts = stats.get("contrasts") or {}
    blocked = dict(blocked or {})
    evaluable, clean = [], []
    for config_id in (stats.get("runs") or {}):
        if config_id in ("mlp", "residual_mlp"):
            continue
        record = contrasts.get(f"{config_id}-mlp@all")
        if not (record or {}).get("available"):
            continue
        if config_id in blocked:
            # Blocked by a failed or missing dependency certificate: it is
            # NOT_EVALUABLE, so it is neither evaluable nor CI-clean here.
            continue
        evaluable.append(config_id)
        # Every registered per-seed readout, derived from the statistics'
        # own readout list, plus the joint mean: at five seeds this is five
        # per-seed readouts, not seeds 7 and 13 alone.
        readouts = [record.get("estimand_ii") if name == JOINT_READOUT
                    else (record.get("estimand_i") or {}).get(name)
                    for name in readouts_of(stats)]
        if all(r and r.get("ci_clean_favourable") for r in readouts):
            clean.append(config_id)
    return {"evaluable": sorted(evaluable), "ci_clean_favourable":
            sorted(clean),
            "blocked_by_dependency_certificate": sorted(blocked),
            "criterion": (
                "CI-clean favourable on "
                + ("both" if len(readouts_of(stats)) == 3
                   else f"all {len(readouts_of(stats)) - 1}")
                + " per-seed estimand (i) readouts and on the estimand (ii) "
                  "seed mean of `arm − mlp` on the `all` slice")}


def section_plain(stats: Mapping[str, Any],
                  k_record: Mapping[str, Any] | None,
                  blocked: Mapping[str, str] | None = None) -> list[str]:
    blocked = dict(blocked or {})
    best = _best_observed(stats)
    gain = _sequence_gain(stats, blocked)
    incomplete = sorted(config_id for config_id, block
                        in (stats.get("runs") or {}).items()
                        if not block.get("complete_paired_seeds"))
    lines = [
        "## 10. In plain language: what was tested, what came out, and why "
        "nothing advances",
        "",
        "**What was tested.** Sixteen small neural models were each trained "
        "on the same 1.88 million historical deliveries and then asked, one "
        "ball at a time, to put a probability on each of the six outcomes of "
        "the *next* delivery of 124,292 held-out balls. The ball that "
        "actually happened is known to the scorer but never to the model. "
        "The change under test is what each is allowed to remember about the "
        "innings so far: nothing at all (the token MLP control), everything "
        "(the full transformer), everything with a fixed decay, everything "
        "with a learned forgetting gate, only the same batter's and bowler's "
        "own past balls, only the last k balls, or a recurrent memory. Two "
        "further arms start from the production ball model's own probability "
        "and learn a correction on top of it. They do **not** differ only in "
        "that: capacity and parameter count, architecture, the positional "
        "scheme, how keys are built from an earlier ball, and — for the two "
        "residual arms — access to the production model's own probabilities "
        "differ as well, and every one of those is confounded with the "
        "memory change in at least one contrast (Astra gate 2 round 1 "
        "MUST-FIX 4; § 9 lists each).",
        "",
        "**This is teacher-forced ball prediction.** It is not rollout — "
        "nobody simulated a match — and it is not market performance. A "
        "better log loss here does not imply a better simulated score "
        "distribution, a better win probability, or any betting edge. No "
        "market price appears anywhere in this stage.",
        "",
    ]
    if best["config_id"] is None:
        lines += ["**Best observed validation configuration.** Not yet "
                  "determinable: no configuration has both registered seeds "
                  "admitted, so there is no observed two-seed mean to "
                  "report.", ""]
    else:
        lines += [
            f"**Best observed validation configuration.** `"
            f"{best['config_id']}`, with an observed two-seed mean validation "
            f"log loss of {f6(best['mean_ll'])} "
            f"(per seed: {', '.join(f'seed {s} {f6(v)}' for s, v in best['per_seed'].items())}; "
            f"source: {best['source']}). That is an *observed* ranking on the "
            "same split its own early stopping used, over two seeds. Its "
            "uncertainty is the paired interval of its registered primary "
            "contrast in § 4, not this number, and the two-seed spread in "
            "§ 5 is an empirical range rather than a confidence interval.",
            "",
        ]
    if gain["evaluable"] and not gain["ci_clean_favourable"]:
        lines += [
            f"**{NO_SEQUENCE_GAIN_SENTENCE}.** " + NO_SEQUENCE_GAIN_QUALIFIER
            + f" The criterion was {gain['criterion']}, over "
            f"{len(gain['evaluable'])} evaluable arms.",
            "",
        ]
    elif gain["ci_clean_favourable"]:
        lines += [
            "Some arms are CI-clean favourable against the token MLP on the "
            "`all` slice under "
            + gain["criterion"] + ": "
            + ", ".join(f"`{a}`" for a in gain["ci_clean_favourable"])
            + ". That is a screening readout on two seeds, and whether any "
            "of them clears its registered family is decided in § 4, not "
            "here.",
            "",
        ]
    else:
        lines += [
            "Whether any sequence arm beats the token MLP is "
            "`NOT_EVALUABLE` tonight: no `arm − mlp` contrast on the `all` "
            "slice has complete paired seeds, so the registered sentence "
            "about no further sequence gain is not yet available and is not "
            "asserted.",
            "",
        ]
    if blocked:
        lines += [
            "**Blocked by dependency certificates.** "
            + ", ".join(f"`{name}`" for name in sorted(blocked))
            + f" are reported `{DEPENDENCY_BLOCKED_STATUS}` and are not "
            "eligible, because a masked-arm dependency certificate is failed "
            "or missing for each. That block is not a result about cricket: "
            "it means the arm has not been shown to depend only on the rows "
            "its registered mask allows, so nothing it scores can be "
            "interpreted as that mechanism (D10.12).",
            "",
        ]
    if incomplete:
        lines += ["Configurations without both registered seeds, which make "
                  "every contrast that needs them `NOT_EVALUABLE`: "
                  + ", ".join(f"`{c}`" for c in incomplete) + ".", ""]
    lines += [
        "**Why no arm advances tonight.** Three reasons, all registered "
        "before the results were read. (1) Two seeds are a directional "
        "screen: the seed-mean estimand is descriptive, so no interval here "
        "is a confirmation. (2) Checkpoint selection used the same "
        "validation split the contrasts are computed on, so every number "
        "carries selection optimism. (3) The one untouched cohort was ruled "
        "**deferred** by the reviewer, so nothing has been confirmed out of "
        "sample. `advances: []`.",
        "",
        "**Cohort confirmation is pending.** `cohort_status: "
        "DEFERRED_UNOPENED`, `cohort_scored: false`: no cohort feature, "
        "prediction or base-logit read happened. **There is no "
        "market claim. There is no LANDED verdict** — two-seed evidence is "
        "provisional and can never be LANDED (invariant 9). **The user's "
        "verdict is outstanding.**",
        "",
        "**The exact next step, and the complete set of conditions that "
        "unlock the cohort (D10.16, restated in full — nothing here is "
        "abbreviated, and Astra gate 2 round 1 MUST-FIX 2 records that "
        "five seeds alone cannot unlock this cohort):**",
        "",
        "0. **The historical-consumption question must be settled first.** "
        "D10.16(0) keeps Astra gate 1 round 2's MUST-FIX 6 open: whether the "
        "2026-04-17 → 2026-08-05 window is untouched as an *evaluation* set "
        "is not established. `docs/sequence_track/"
        "stage2_cohort_consumer_audit.md` establishes clean training frame "
        "and cache **ancestry** only, and clean training ancestry does not "
        "prove untouched evaluation status. Until that is resolved in full, "
        "no number of seeds unlocks anything.",
        "1. Seeds **29, 42 and 101** are added to every retained **whole** "
        "hypothesis family — the candidate, `mlp`, every matched control, and "
        "**all five** `same_entity` k configurations whenever k selection is "
        "involved. Never a single arm.",
        "2. The seed-extension procedure **and** the final k-selection "
        "procedure are **frozen before any new-seed result is inspected**. A "
        "procedure chosen after seeing the new seeds is selection on the "
        "outcome and voids the extension.",
        "3. The registered selection and the registered gates are rerun on "
        "all five seeds, reporting the seed spread and the **4/5 favourable "
        "direction count**; only candidates eligible under those reruns are "
        "retained.",
        "4. The final family, checkpoint and **analysis** freeze is written "
        "and verified — an explicitly versioned final freeze, separate from "
        "the immutable training provenance (`pin_stage2_analysis.py "
        "--verify`).",
        "5. The cohort's **provenance is re-verified unchanged** against its "
        "frozen hashes before it is opened.",
        "6. The read is **one** frozen scoring batch covering every frozen "
        "candidate and control at once, with no interim result-driven change, "
        "no added arm, no changed k and no tuning; a start record and all "
        "outputs are persisted.",
        "7. **Partial-exposure recovery rule:** any recovery or rerun reuses "
        "the identical frozen specification, **discloses the partial "
        "exposure**, and **never claims a fresh untouched read**.",
        "8. If no candidate qualifies, **the cohort is left unopened**.",
        "",
        "So the immediate next step is (0) plus (1)–(3): resolve historical "
        "consumption, freeze the extension and selection procedure, then "
        "train seeds 29, 42 and 101 across whole families and rerun these "
        "identical gates"
        + ("" if k_record is None else
           f"; tonight's k selection is `{k_record.get('selection')}` and is "
           "explicitly provisional")
        + ".",
        "",
    ]
    return lines


def section_falsification(stats: Mapping[str, Any],
                          blocked: Mapping[str, str] | None = None
                          ) -> list[str]:
    """Astra gate 2 round 1 MUST-FIX 1: blocking reaches the falsification
    section too, so a blocked arm's contrast cannot read as an interval here.
    """
    contrasts = stats.get("contrasts") or {}
    blocked = dict(blocked or {})

    def status(key: str) -> str:
        record = contrasts.get(key)
        if record:
            endpoints = (str(record.get("candidate")),
                         str(record.get("reference")))
            if any(name in blocked for name in endpoints):
                return "NOT_EVALUABLE (dependency certificate blocks an "\
                       "endpoint)"
        if not (record or {}).get("available"):
            return "NOT_EVALUABLE"
        joint = record.get("estimand_ii") or {}
        if joint.get("ci_clean_favourable"):
            return "favourable"
        if joint.get("l95") is not None and float(joint["l95"]) > 0:
            return "adverse"
        return "unresolved"

    fox = status("fox-fixed_decay@all")
    ownership = status("same_entity_k30-recency_k30@all")
    mask = status("same_entity_unr-aligned_hist_rf@all")
    death = status("full-mlp@death")
    lines = [
        "## 11. Falsification wording (D10.15) and what a result here can "
        "and cannot mean",
        "",
        "No result tonight licenses \"X is the cause\".",
        "",
        f"* `fox − fixed_decay` reads **{fox}** on the seed-mean estimand. A "
        "favourable reading would support *learned forgetting as an "
        "explanation*, nothing stronger."
        + (" As it reads, FoX establishes **no detected benefit over fixed "
           "decay** at this resolution — which is not the same as learned "
           "forgetting adding nothing (Astra gate 2 round 1 MUST-FIX 4)."
           if fox.startswith("unresolved") else ""),
        f"* `same_entity_k30 − recency_k30` reads **{ownership}**. A "
        "favourable reading would support *ownership plus alignment beyond "
        "recency*, and cannot isolate ownership from alignment.",
        f"* `same_entity_unr − aligned_hist_rf` reads **{mask}**. A "
        "favourable reading would support *excluding other participants' "
        "history*, with alignment, wiring and key construction held fixed.",
        "",
        "An interval that crosses zero means **unresolved evidence**, not "
        "proof that the mechanism does not matter. Where a reading above is "
        f"`{DEPENDENCY_BLOCKED_STATUS}` because a dependency certificate "
        "blocks an endpoint, even that is unavailable: the arm has not been "
        "shown to depend only on the rows its mask allows, so its interval "
        "supports no mechanism reading at all.",
        "",
    ]
    unsupported = ("unresolved", "adverse", "NOT_EVALUABLE")
    if fox.startswith(unsupported) and ownership.startswith(unsupported):
        lines += ["Neither proposed mechanism is supported at this "
                  "resolution."
                  + (" Here that is because neither contrast is evaluable, "
                     "which is weaker still than an unresolved interval: "
                     "nothing was measured."
                     if fox.startswith("NOT_EVALUABLE")
                     or ownership.startswith("NOT_EVALUABLE") else ""),
                  ""]
    lines += [
        "**On the death-over harm.** Explaining it would additionally "
        "require reproducing that harm on the matched `full − mlp` death "
        "slice *and* direct evidence of improvement on death rows; those "
        "mechanism-on-death comparisons are exploratory under the current "
        f"family map. `full − mlp` on `death` reads **{death}** on the "
        "seed-mean estimand.",
        "",
    ]
    if death.startswith("NOT_EVALUABLE"):
        lines += ["That comparison is **not evaluable** tonight, so this "
                  "report neither reproduces nor fails to reproduce the "
                  "original death-over harm, and does not explain it.", ""]
    elif death.startswith(("unresolved", "favourable")):
        lines += ["The original death-over harm is **not reproduced** on "
                  "this frame under this readout, so this report does not "
                  "explain an absent effect.", ""]
    lines += [
        "Nothing in this report is a betting claim, an advancement, or a "
        "verdict. `research/log_verdict.py` was not called.",
        "",
    ]
    return lines


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _checkpoint_seed(checkpoint: str) -> int | None:
    """The TRAINING seed of the certified checkpoint, from its directory name.

    A certificate's own `seed` field is the perturbation seed (29), not the
    checkpoint's training seed, so coverage counted over `seed` would count
    configurations rather than seeds — exactly Astra's MUST-FIX 1 defect.
    """
    match = re.search(r"seed_(\d+)$", Path(str(checkpoint or "")).name)
    return int(match.group(1)) if match else None


def authenticate_checkpoint(checkpoint: Any, recorded_md5: Any) -> dict:
    """Recompute the certified checkpoint's md5 and compare it (MUST-FIX 1).

    A recorded hash nobody recomputes authenticates nothing. Paths inside the
    closed smoke tree are NOT opened — this stage may not read them — so those
    certificates are reported as unauthenticated-by-construction and can never
    count as trained-checkpoint coverage.
    """
    posix = Path(str(checkpoint or "")).as_posix() if checkpoint else ""
    out = {"checkpoint": posix, "recorded_md5": recorded_md5,
           "observed_md5": None, "status": "NO_CHECKPOINT_PATH"}
    if not posix:
        return out
    if any(fragment in posix for fragment in FORBIDDEN_FRAGMENTS):
        out["status"] = "NOT_AUTHENTICATED_CLOSED_TREE"
        return out
    if not recorded_md5:
        out["status"] = "NO_RECORDED_MD5"
        return out
    directory = Path(posix)
    if not directory.is_absolute():
        directory = REPO / directory
    model = directory / "model.pt"
    if not model.is_file():
        out["status"] = "CHECKPOINT_FILE_ABSENT"
        return out
    out["observed_md5"] = md5_file(model)
    out["status"] = ("AUTHENTICATED" if out["observed_md5"] == recorded_md5
                     else "MD5_MISMATCH")
    return out


def dependency_certificates(dependency: Sequence[Mapping[str, Any]]) -> dict:
    """Which masked arms hold a complete, authenticated certificate.

    Astra gate 1 round 3: a failed **or missing** certificate for a masked arm
    blocks that arm's interpretation and eligibility. Astra gate 2 round 1
    MUST-FIX 1 closed four ways in which that enforcement still admitted
    incomplete evidence:

    1. positive controls are matched through `dependency_set_arm` /
       `dependency_set_k` — the masked arm's S(i) they were scored against —
       not through their own `arm`/`k`, which never equal the masked arm's;
    2. trained-checkpoint coverage requires BOTH registered training seeds of
       each masked arm, read from the checkpoint directory, not one
       certificate per configuration;
    3. every certificate's recorded checkpoint md5 is recomputed from the
       checkpoint on disk, and the registered perturbation settings
       (2,000 targets, seed 29, validation) are asserted;
    4. a MISSING registered positive control blocks exactly as a failed one
       does.

    A certificate whose checkpoint lies under `smoke/` counts as PRESENT — the
    masking it certifies is structural, a property of the architecture — but it
    is never trained-checkpoint coverage, and it can never carry an arm's
    eligibility on its own. Astra gate 2 round 2 closed the remaining hole:
    the trained-seed shortfall was only checked for an arm that already held at
    least one trained certificate, so a tree in which the trained
    recertifications had disappeared and only the one-epoch smoke record
    remained read `PASS`. A shortfall against
    `DEPENDENCY_REQUIRED_CHECKPOINT_SEEDS` now yields
    `TRAINED_SEEDS_INCOMPLETE` whether the arm holds one registered seed or
    none.
    """
    by_config: dict[str, dict] = {
        config_id: {"config_id": config_id, "arm": arm, "k": k,
                    "masked_records": 0, "failed": [],
                    "settings_problems": [], "authentication": [],
                    "authentication_problems": [],
                    "trained_seeds": [], "smoke_seeds": [],
                    "controls": [], "control_registered": False,
                    "registered_control_arm": None,
                    "smoke_only": None, "coverage": "none",
                    "status": "MISSING", "reason": None}
        for (arm, k), config_id in DEPENDENCY_REQUIRED.items()}
    for (arm, k), spec in DEPENDENCY_CONTROLS.items():
        block = by_config[spec["config_id"]]
        block["control_registered"] = True
        block["registered_control_arm"] = spec["control_arm"]
    unmatched_controls: list[dict] = []

    for record in dependency:
        checkpoint = str(record.get("checkpoint") or "")
        set_key = (str(record.get("dependency_set_arm")),
                   str(record.get("dependency_set_k")))
        if record.get("positive_control_expected"):
            spec = DEPENDENCY_CONTROLS.get(set_key)
            entry = {"arm": str(record.get("arm")),
                     "checkpoint": checkpoint,
                     "dependency_set": f"{set_key[0]} k={set_key[1]}",
                     "fired": bool(record.get("positive_control_observed")),
                     "max_abs_delta": record.get("max_abs_delta"),
                     "expected_arm": (spec or {}).get("control_arm"),
                     "arm_matches_registration":
                         bool(spec) and str(record.get("arm"))
                         == spec["control_arm"]}
            if spec is None:
                unmatched_controls.append(entry)
                continue
            by_config[spec["config_id"]]["controls"].append(entry)
            continue

        key = (str(record.get("arm")), str(record.get("k")))
        config_id = DEPENDENCY_REQUIRED.get(key)
        if config_id is None:
            continue
        block = by_config[config_id]
        block["masked_records"] += 1
        name = Path(checkpoint).as_posix() or "?"
        if set_key != key:
            block["settings_problems"].append(
                f"certificate `{name}` was scored against the dependency set "
                f"of `{set_key[0]}` k={set_key[1]}, not its own S(i)")
        for field, expected in DEPENDENCY_REGISTERED_SETTINGS.items():
            observed = record.get(field)
            if observed != expected:
                block["settings_problems"].append(
                    f"certificate `{name}` records {field}={observed!r}, not "
                    f"the registered {expected!r}")
        smoke = "/smoke/" in checkpoint
        seed = _checkpoint_seed(checkpoint)
        auth = authenticate_checkpoint(checkpoint, record.get("checkpoint_md5"))
        auth["config_id"] = config_id
        auth["checkpoint_seed"] = seed
        block["authentication"].append(auth)
        if not record.get("pass"):
            block["failed"].append(name)
            continue
        if smoke:
            if seed is not None:
                block["smoke_seeds"].append(seed)
            continue
        if auth["status"] != "AUTHENTICATED":
            block["authentication_problems"].append(
                f"certificate `{name}` is not authenticated "
                f"({auth['status']})")
            continue
        if seed is None:
            block["authentication_problems"].append(
                f"certificate `{name}` does not name a training seed in its "
                "checkpoint path, so it cannot be counted toward either seed")
            continue
        block["trained_seeds"].append(seed)

    blocked: dict[str, str] = {}
    required_seeds = set(DEPENDENCY_REQUIRED_CHECKPOINT_SEEDS)
    for config_id, block in by_config.items():
        block["trained_seeds"] = sorted(set(block["trained_seeds"]))
        block["smoke_seeds"] = sorted(set(block["smoke_seeds"]))
        block["smoke_only"] = bool(block["smoke_seeds"]) and not block[
            "trained_seeds"]
        controls = block["controls"]
        fired = [c for c in controls
                 if c["fired"] and c["arm_matches_registration"]]
        block["controls_fired"] = len(fired)
        missing_seeds = sorted(required_seeds - set(block["trained_seeds"]))
        if block["masked_records"] == 0:
            block["status"] = "MISSING"
            block["reason"] = ("no masked-arm dependency certificate is "
                               "present for this configuration")
        elif block["failed"]:
            block["status"] = "FAILED"
            block["reason"] = ("its dependency certificate failed for "
                               + ", ".join(f"`{name}`"
                                           for name in block["failed"]))
        elif block["settings_problems"]:
            block["status"] = "SETTINGS_MISMATCH"
            block["reason"] = ("its certificate was not computed under the "
                               "registered test: "
                               + "; ".join(block["settings_problems"]))
        elif block["authentication_problems"]:
            block["status"] = "CHECKPOINT_NOT_AUTHENTICATED"
            block["reason"] = ("its certificate's checkpoint hash does not "
                               "authenticate against the checkpoint on disk: "
                               + "; ".join(block["authentication_problems"]))
        elif block["control_registered"] and not controls:
            block["status"] = "CONTROL_MISSING"
            block["reason"] = (
                "its registered matched positive control "
                f"(`{block['registered_control_arm']}` scored against this "
                "arm's S(i)) is absent from the certification evidence, so "
                "the passing certificate establishes no sensitivity to "
                "excluded-past information")
        elif block["control_registered"] and not fired:
            block["status"] = "CONTROL_DID_NOT_FIRE"
            block["reason"] = ("its matched positive control did not fire, so "
                               "the passing certificate establishes no "
                               "sensitivity to excluded-past information")
        elif missing_seeds:
            # A smoke record certifies STRUCTURAL masking and can never carry
            # an arm's eligibility. Before this branch covered the zero-trained
            # case, a tree holding only the one-epoch smoke certificate fell
            # through to `PASS`: "their handling becomes incorrect only when
            # trained certificates disappear and smoke records alone preserve
            # eligibility" (Astra). Any shortfall against the registered
            # trained seeds now blocks, whether the arm holds one of them or
            # none.
            block["status"] = "TRAINED_SEEDS_INCOMPLETE"
            held = (", ".join(str(s) for s in block["trained_seeds"])
                    if block["trained_seeds"] else "no seed at all")
            smoke_note = (
                " Its only certificate is the one-epoch smoke record at "
                "seed(s) " + ", ".join(str(s) for s in block["smoke_seeds"])
                + ", which certifies structural masking and is never "
                  "trained-checkpoint coverage."
                if block["smoke_only"] else "")
            block["reason"] = (
                "it holds an authenticated trained-checkpoint certificate at "
                + held
                + ", not at seed(s) "
                + ", ".join(str(s) for s in missing_seeds)
                + ", and D10.12 requires both registered seeds."
                + smoke_note)
        else:
            block["status"] = "PASS"
        if block["trained_seeds"] and not missing_seeds:
            block["coverage"] = ("trained checkpoints at seeds "
                                 + ", ".join(str(s) for s
                                             in block["trained_seeds"])
                                 + " (md5-authenticated)")
        elif block["trained_seeds"]:
            block["coverage"] = ("trained checkpoints at seed(s) "
                                 + ", ".join(str(s) for s
                                             in block["trained_seeds"])
                                 + " only")
        elif block["smoke_only"]:
            block["coverage"] = "smoke checkpoint only (structural)"
        else:
            block["coverage"] = "none"
        if block["status"] != "PASS":
            blocked[config_id] = block["reason"] or block["status"]

    trained = [config_id for config_id, block in by_config.items()
               if block["status"] == "PASS" and set(block["trained_seeds"])
               >= required_seeds]
    controls_complete = all(
        any(c["fired"] and c["arm_matches_registration"]
            for c in by_config[spec["config_id"]]["controls"])
        for spec in DEPENDENCY_CONTROLS.values())
    return {"by_config": by_config,
            "blocked": blocked,
            "unmatched_controls": unmatched_controls,
            "registered_controls_complete": controls_complete,
            "required_checkpoint_seeds":
                list(DEPENDENCY_REQUIRED_CHECKPOINT_SEEDS),
            "registered_settings": dict(DEPENDENCY_REGISTERED_SETTINGS),
            "trained_checkpoint_coverage": sorted(trained),
            "trained_checkpoint_coverage_complete":
                len(trained) == len(DEPENDENCY_REQUIRED) and controls_complete,
            "configurations_without_a_registered_control": sorted(
                config_id for config_id, block in by_config.items()
                if not block["control_registered"]),
            "required": sorted(DEPENDENCY_REQUIRED.values())}


def blocking_note(config_id: str, blocked: Mapping[str, str]) -> str:
    """One sentence naming why a family is forced to `NOT_EVALUABLE`."""
    return (f"**Dependency certificate blocks `{config_id}`**: "
            f"{blocked[config_id]}. Its whole family is therefore reported as "
            f"`{DEPENDENCY_BLOCKED_STATUS}` and the arm is not eligible, "
            "whatever the intervals below say (D10.12; Astra gate 1 round 3).")


def load_dependency(directory: Path) -> list[dict]:
    """Every certificate under `directory`, recursively (MUST-FIX 6).

    The certification evidence lives in two places — the smoke-era records and
    the registered positive controls directly under `dependency/`, the trained
    recertifications under `dependency/recert/`. Loading only immediate
    `*.json` meant the directory passed on the command line silently chose
    which evidence existed, so `--dependency-dir .../recert` saw no controls at
    all while the default saw no trained certificates. Loading recursively
    makes `--dependency-dir models/embeddings/seq_stage2/dependency` the one
    invocation that sees the whole evidence set; the invocation actually used
    is recorded and verified by `pin_stage2_analysis.py`.
    """
    directory = guard_path(directory)
    if not directory.is_dir():
        return []
    records = []
    for path in sorted(directory.rglob("*.json")):
        try:
            records.append(read_json(path))
        except (OSError, json.JSONDecodeError):
            continue
    return [r for r in records if isinstance(r, Mapping)]


def render(stats_path: Path, config_path: Path, k_path: Path | None,
           dependency_dir: Path, out: Path) -> str:
    stats = read_json(stats_path)
    config = read_yaml(config_path) or {}
    k_record = None
    if k_path is not None and guard_path(k_path).exists():
        k_record = read_json(k_path)
    dependency = load_dependency(dependency_dir)
    # A failed, missing, unauthenticated, seed-incomplete or control-less
    # masked-arm certificate blocks that arm's family and its eligibility,
    # everywhere the report states a status — including the mechanism (D10.5)
    # and falsification (D10.15) sections (Astra gate 2 round 1 MUST-FIX 1).
    certificates = dependency_certificates(dependency)
    blocked = certificates["blocked"]

    lines: list[str] = []
    lines += header(stats, config_path, out, stats_path, k_path)
    lines += section_question(config)
    lines += section_arms(config, stats, dependency, certificates)
    lines += section_rule(config, stats)
    lines += section_results(stats, blocked)
    lines += section_estimands(stats)
    lines += section_ksweep(k_record)
    lines += section_mechanism(stats, blocked)
    lines += section_gates(stats, blocked)
    lines += section_limitations(config)
    lines += section_plain(stats, k_record, blocked)
    lines += section_falsification(stats, blocked)

    markdown = "\n".join(lines).rstrip() + "\n"
    assert_coverage(markdown, coverage_manifest(config))
    return markdown


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stats-json", type=Path, default=DEFAULT_STATS_OUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--k-selection", type=Path, default=DEFAULT_KSWEEP_OUT)
    parser.add_argument("--dependency-dir", type=Path,
                        default=DEFAULT_DEPENDENCY_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    try:
        # A five-seed render may not overwrite the two-seed report of record.
        require_distinct_out(args.config, args.out, DEFAULT_OUT)
        markdown = render(args.stats_json, args.config, args.k_selection,
                          args.dependency_dir, args.out)
    except RefusalError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2
    out = guard_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown)
    print(f"wrote {rel(out)} ({len(markdown.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
