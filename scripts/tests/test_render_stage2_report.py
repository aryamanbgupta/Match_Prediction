"""Tests for `scripts/sequence_track/render_stage2_report.py` (D10.8, D10.13-D10.15).

Synthetic fixtures only: the statistics JSON is produced by
`stage2_stats.compute_statistics` over a tiny fake runs tree, exactly as the
real render will consume it.  Nothing here reads the cohort, the smoke tree or
either sealed holdout.
"""
from __future__ import annotations

import builtins
import io
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from sequence_track import render_stage2_report as rr  # noqa: E402
from sequence_track import stage2_stats as st  # noqa: E402
from tests.test_stage2_stats import (  # noqa: E402
    CONFIG_IDS,
    EXPECT,
    K_IDS,
    SEEDS,
    _config_payload,
    _k_runs,
    _write_run,
    _write_summary,
)

pytest_plugins = ["tests.test_stage2_stats"]


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Realistic dependency evidence (Astra gate 2 round 1 MUST-FIX 1)
#
# The previous fixture held one artificial masked-arm record with no checkpoint
# md5, no `dependency_set_arm`/`k`, no perturbation settings and no training
# seed, plus a "control" that was matched by its own arm. That record shape
# concealed every defect Astra found. These fixtures mirror the real
# certificates field for field, write real checkpoint files so the recorded md5
# can actually be authenticated, and lay the evidence out the way the tree
# does: trained recertifications under `dependency/recert/`, the registered
# positive controls beside them under `dependency/`.
# ---------------------------------------------------------------------------

MASKED_CERTIFICATES = (("same_entity", 0, "same_entity_k0"),
                       ("same_entity", 30, "same_entity_k30"),
                       ("same_entity", "unr", "same_entity_unr"),
                       ("recency", 30, "recency_k30"))
CERTIFICATE_SEEDS = (7, 13)


def _fake_checkpoint(root: Path, config_id: str, seed: int) -> tuple[str, str]:
    """A checkpoint directory with a real `model.pt`, and its md5."""
    import hashlib

    directory = root / config_id / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    model = directory / "model.pt"
    model.write_bytes(f"{config_id}/{seed}".encode())
    return (directory.as_posix(),
            hashlib.md5(model.read_bytes()).hexdigest())


def _masked_record(arm, k, checkpoint: str, md5: str, **overrides) -> dict:
    record = {"arm": arm, "k": k,
              "checkpoint": checkpoint, "checkpoint_md5": md5,
              "control_floor": 0.001,
              "dependency_set_arm": arm, "dependency_set_k": k,
              "device": "cpu", "key_construction": "own_outcome",
              "max_abs_delta": 0.0, "n_innings": 1088,
              "n_nonzero_gt_1e-6": 0, "n_rows": 124292,
              "n_targets": 2000, "n_targets_with_nothing_outside_s": 0,
              "noise_sd": 3.0, "p99_abs_delta": 0.0, "pass": True,
              "positive_control_expected": False,
              "positive_control_observed": False,
              "seed": 29, "split": "validation", "tolerance": 1e-06,
              "wall_seconds": 2.5, "wiring": "relay_free"}
    record.update(overrides)
    return record


def _control_record(arm, set_arm, set_k, checkpoint: str, md5: str,
                    **overrides) -> dict:
    record = {"arm": arm, "k": None,
              "checkpoint": checkpoint, "checkpoint_md5": md5,
              "control_floor": 0.001,
              "dependency_set_arm": set_arm, "dependency_set_k": set_k,
              "device": "cpu", "key_construction": "shifted_history",
              "max_abs_delta": 0.9216543436050415, "n_innings": 1088,
              "n_nonzero_gt_1e-6": 1867, "n_rows": 124292,
              "n_targets": 2000, "n_targets_with_nothing_outside_s": 0,
              "noise_sd": 3.0, "p99_abs_delta": 0.7410516822338105,
              "pass": False,
              "positive_control_expected": True,
              "positive_control_observed": True,
              "seed": 29, "split": "validation", "tolerance": 1e-06,
              "wall_seconds": 2.1, "wiring": "standard"}
    record.update(overrides)
    return record


def _write_dependency_evidence(tmp_path: Path, seeds=CERTIFICATE_SEEDS,
                               controls: bool = True) -> Path:
    """The whole evidence set: 4 masked arms x both seeds, plus both controls."""
    checkpoints = tmp_path / "checkpoints"
    dependency = tmp_path / "dependency"
    recert = dependency / "recert"
    recert.mkdir(parents=True, exist_ok=True)
    for arm, k, config_id in MASKED_CERTIFICATES:
        for seed in seeds:
            checkpoint, md5 = _fake_checkpoint(checkpoints, config_id, seed)
            (recert / f"{config_id}_seed{seed}.json").write_text(json.dumps(
                _masked_record(arm, k, checkpoint, md5)))
    if controls:
        for control_arm, set_arm, set_k in (("full", "recency", 30),
                                           ("aligned_hist", "same_entity",
                                            30)):
            checkpoint, md5 = _fake_checkpoint(checkpoints, control_arm, 7)
            (dependency / f"{control_arm}.json").write_text(json.dumps(
                _control_record(control_arm, set_arm, set_k, checkpoint,
                                md5)))
    return dependency


@pytest.fixture
def built(tmp_path, frame_dir, block_source):
    """A statistics JSON, a config and a k-selection record on disk."""
    import pandas as pd

    config = _config_payload(frame_dir=frame_dir, same_entity=True)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    frame = pd.read_parquet(frame_dir / "cricket_data_i7_validation.parquet")
    runs = tmp_path / "runs"
    biases = {"mlp": 0.0, "full": 0.9, "fixed_decay": 0.5, "fox": 0.7}
    for config_id, bias in biases.items():
        for seed in SEEDS:
            _write_run(runs, config_id, seed, frame, bias=bias)
        _write_summary(runs, config_id, {7: 1.5 - bias / 10,
                                        13: 1.51 - bias / 10})
    for k, config_id in K_IDS.items():
        for seed in SEEDS:
            _write_run(runs, config_id, seed, frame, bias=0.4,
                       arm="same_entity", k=(k if k == "unr" else int(k)))
        _write_summary(runs, config_id, {7: 1.5, 13: 1.5})

    stats = st.compute_statistics(
        config_path, runs, frame_dir, block_source,
        tmp_path / "no_base_logits.npz", reps=50, seed=29, seeds=SEEDS,
        expect=dict(EXPECT), expected_families=8)
    stats_path = tmp_path / "stage2_statistics.json"
    stats_path.write_text(json.dumps(stats, indent=2, default=str))

    k_record = st.k_sweep(runs, config, seeds=SEEDS)
    k_path = tmp_path / "k_selection.json"
    k_path.write_text(json.dumps(k_record, indent=2, default=str))

    dependency = _write_dependency_evidence(tmp_path)
    return {"config": config, "config_path": config_path,
            "stats": stats, "stats_path": stats_path,
            "k_record": k_record, "k_path": k_path,
            "dependency": dependency, "out": tmp_path / "REPORT.md"}


def _render(built, **overrides) -> str:
    params = {"stats_path": built["stats_path"],
              "config_path": built["config_path"],
              "k_path": built["k_path"],
              "dependency_dir": built["dependency"],
              "out": built["out"]}
    params.update(overrides)
    return rr.render(**params)


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------

def test_report_renders_every_required_section(built):
    markdown = _render(built)
    for heading in ("## 1. Question",
                    "## 2. Arms and settings",
                    "## 3. Decision rule",
                    "## 4. Results",
                    "## 5. The two estimands",
                    "## 6. The k sweep (D9)",
                    "## 7. Mechanism contrasts (D10.5)",
                    "## 8. Non-inferiority gates",
                    "## 9. Registered deviations, asymmetries, limitations",
                    "## 10. In plain language",
                    "## 11. Falsification wording"):
        assert heading in markdown, heading


def test_report_states_the_registered_status_block(built):
    markdown = _render(built)
    assert "cohort_status: DEFERRED_UNOPENED" in markdown
    assert "cohort_scored: false" in markdown
    assert "advances: []" in markdown
    assert "No arm advances." in markdown
    assert "No market claim." in markdown
    assert "No LANDED verdict" in markdown
    assert "The user's verdict is outstanding" in markdown
    assert "invariant 9" in markdown


def test_report_carries_the_two_estimands_and_the_rank_local_label(built):
    markdown = _render(built)
    assert "seed mean (estimand ii)" in markdown
    assert "seed 7 (estimand i)" in markdown
    assert "not the log loss of averaged probabilities" in markdown.replace(
        "**not** the log loss", "not the log loss")
    assert st.RANK_LOCAL_NOTE.split(";")[0] in markdown
    assert "descriptive two-seed robustness screen" in markdown


def test_report_lists_admitted_and_incomplete_configurations(built, tmp_path,
                                                             frame_dir,
                                                             block_source):
    markdown = _render(built)
    for config_id in list(CONFIG_IDS) + list(K_IDS.values()):
        assert f"`{config_id}`" in markdown
    assert "Every configuration has both registered seeds." in markdown

    # Drop one seed and the report must name the incomplete configuration.
    stats = json.loads(built["stats_path"].read_text())
    stats["runs"]["fox"]["complete_paired_seeds"] = False
    stats["runs"]["fox"]["admitted_seeds"] = [7]
    path = tmp_path / "partial.json"
    path.write_text(json.dumps(stats))
    markdown = _render(built, stats_path=path)
    assert "**Incomplete configurations**" in markdown
    assert "`fox`" in markdown


def test_report_shows_machine_provenance_and_the_confound(built):
    markdown = _render(built)
    assert "### Machine provenance per run (D8.8)" in markdown
    assert "machine is confounded with seed" in markdown.lower()
    assert "no machine term is fitted" in markdown
    assert "laptop" in markdown and "mini" in markdown


def test_report_shows_the_admission_verification(built):
    markdown = _render(built)
    assert "### Admission and provenance verification" in markdown
    assert "Pin available: **yes**" in markdown
    assert "read-only admission verifier" in markdown
    assert "Duplicate seed rows are rejected" in markdown
    assert "seed-independent training signature" in markdown


def test_report_shows_the_dependency_certificates_and_marks_smoke(built,
                                                                  tmp_path):
    markdown = _render(built)
    assert "Ownership dependency certificates" in markdown
    assert "SEES EXCLUDED-PAST INFORMATION" in markdown
    assert "not specifically multi-layer relay" in markdown
    # A smoke-tree certificate is marked as such, is never authenticated (this
    # stage may not open the smoke tree) and never counts as trained coverage.
    (built["dependency"] / "smoke_extra.json").write_text(json.dumps(
        _masked_record("recency", 30,
                       "models/embeddings/seq_stage2/smoke/recency_k30/seed_7",
                       "0" * 32)))
    markdown = _render(built)
    assert "(smoke checkpoint)" in markdown
    assert "NOT_AUTHENTICATED_CLOSED_TREE" in markdown
    certificates = rr.dependency_certificates(
        rr.load_dependency(built["dependency"]))
    assert certificates["by_config"]["recency_k30"]["smoke_seeds"] == [7]
    assert certificates["by_config"]["recency_k30"]["status"] == "PASS"


def test_report_names_the_registered_k_ids_not_a_template(built):
    markdown = _render(built)
    assert "k unr → `same_entity_unr`" in markdown
    assert "same_entity_kunr" not in markdown


def test_report_reports_thin_pair_as_unavailable(built):
    markdown = _render(built)
    assert "`thin_pair` is reported **unavailable**" in markdown
    assert "exposure" in markdown


def test_report_has_no_hand_entered_numeric_cell(built):
    """Every table number must trace to a file.

    For each numeric cell in every markdown table, some number in the stats
    JSON, the selection record or the config must round to it at the cell's
    own displayed precision. A value the generator invented would match none
    of them. Registered check ids (`D7.4`, `D10.16`) are prose, not results,
    and are removed before scanning.
    """
    import re

    markdown = _render(built)
    sources = (built["stats_path"].read_text() + built["k_path"].read_text()
               + built["config_path"].read_text()
               + "".join(path.read_text()
                         for path in sorted(built["dependency"]
                                            .rglob("*.json"))))
    source_numbers = [float(token) for token
                      in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?",
                                    sources)]
    assert len(source_numbers) > 100

    rows = [line for line in markdown.splitlines()
            if line.startswith("|") and "---" not in line]
    assert len(rows) > 20
    checked = 0
    for line in rows:
        for cell in line.split("|"):
            cell = re.sub(r"\bD\d+\.\d+\b", "", cell)
            for token in re.findall(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?",
                                    cell):
                mantissa = token.split("e")[0].split("E")[0]
                decimals = len(mantissa.split(".")[1])
                spec = (f".{decimals}e" if ("e" in token or "E" in token)
                        else f".{decimals}f")
                wanted = token.lstrip("+")
                assert any(format(candidate, spec) == wanted
                           for candidate in source_numbers), (line, token)
                checked += 1
    assert checked > 50


def _table_row(markdown: str, first_cell: str, section: str | None = None,
               until: str | None = None) -> list[str]:
    """The cells of the one table row whose leading cell is `first_cell`."""
    text = markdown
    if section is not None:
        text = text.split(section)[1]
        if until is not None:
            text = text.split(until)[0]
    matches = [line for line in text.splitlines()
               if line.startswith("|")
               and line.split("|")[1].strip() == first_cell]
    assert len(matches) == 1, (first_cell, len(matches))
    return [cell.strip() for cell in matches[0].split("|")[1:-1]]


def test_named_cells_come_from_their_named_source_fields(built):
    """Astra gate 2 round 1 MUST-FIX 5.

    The scan above only proves that a decimal occurs somewhere among the source
    numbers. These assertions tie a NAMED cell to its NAMED source field, so a
    right-looking number in the wrong column fails.
    """
    markdown = _render(built)
    stats = built["stats"]

    # § 7 mechanism row: the seed-mean point and interval come from
    # stats["mechanism_contrasts"][*]["record"]["estimand_ii"].
    entry = next(e for e in stats["mechanism_contrasts"]
                 if e["candidate"] == "fox" and e["reference"] == "fixed_decay")
    joint = entry["record"]["estimand_ii"]
    cells = _table_row(markdown, "`fox − fixed_decay`",
                       section="## 7. Mechanism contrasts", until="## 8.")
    assert cells[5] == rr.f5(joint["point"]), cells
    assert cells[6] == rr.ci(joint["ci95"]), cells
    per_seed = entry["record"]["per_seed_points"]
    assert cells[3] == rr.f5(per_seed["7"])
    assert cells[4] == rr.f5(per_seed["13"])

    # § 2 dependency row: max|Δ| and p99 come from that certificate's own
    # `max_abs_delta` / `p99_abs_delta`, and the seed from its checkpoint path.
    control = json.loads(
        (built["dependency"] / "aligned_hist.json").read_text())
    cells = _table_row(markdown, "`aligned_hist`",
                       section=rr.DEPENDENCY_HEADING_COMPLETE,
                       until="The positive controls establish")
    assert cells[2] == ("`" + str(control["dependency_set_arm"]) + "` k="
                        + str(control["dependency_set_k"]))
    assert cells[6] == rr.sci(control["max_abs_delta"]), cells
    assert cells[7] == rr.sci(control["p99_abs_delta"]), cells
    assert cells[8] == str(control["n_nonzero_gt_1e-6"])

    # § 2 machine-provenance row: the summary log loss cell is the
    # summary.yaml value, never the reconstruction.
    seed_row = next(row for row in stats["runs"]["mlp"]["seeds"]
                    if row["seed"] == 7)
    line = next(l for l in markdown.splitlines()
                if l.startswith("| `mlp` | 7 |"))
    cells = [c.strip() for c in line.split("|")[1:-1]]
    assert cells[-1] == rr.f6(seed_row["summary_yaml_validation_ll"])
    assert cells[-2] == rr.f6(seed_row["reconstructed_validation_ll"])


# ---------------------------------------------------------------------------
# § 9 coverage (D10.13)
# ---------------------------------------------------------------------------

def test_coverage_manifest_covers_the_config_and_d10_13(built):
    manifest = rr.coverage_manifest(built["config"])
    keys = {entry["key"] for entry in manifest}
    assert "deviation:dev_one" in keys
    assert "asymmetry:asym_one" in keys
    assert "limitation:0" in keys
    for key, _ in rr.D10_13_REQUIRED:
        assert f"d10.13:{key}" in keys
    assert len(rr.D10_13_REQUIRED) == 21
    assert "d10.13:unresolved_historical_consumption" in keys


def test_coverage_check_passes_on_the_rendered_report(built):
    markdown = _render(built)
    rr.assert_coverage(markdown, rr.coverage_manifest(built["config"]))


def test_coverage_check_fails_when_a_limitation_is_dropped(built):
    markdown = _render(built)
    manifest = rr.coverage_manifest(built["config"])
    limitation = built["config"]["known_limitations"][0]
    without = markdown.replace(limitation, "")
    with pytest.raises(st.RefusalError, match="coverage check failed"):
        rr.assert_coverage(without, manifest)


def test_coverage_check_fails_when_a_deviation_or_asymmetry_is_dropped(built):
    markdown = _render(built)
    manifest = rr.coverage_manifest(built["config"])
    for fragment in ("`dev_one`", "`asym_one`"):
        without = markdown.replace(fragment, "")
        with pytest.raises(st.RefusalError, match="coverage check failed"):
            rr.assert_coverage(without, manifest)


def test_coverage_check_fails_when_a_d10_13_item_is_dropped(built):
    markdown = _render(built)
    manifest = rr.coverage_manifest(built["config"])
    for key in ("machine_confounded_with_seed_no_machine_term",
                "two_seed_weakness", "global_prior_not_as_of",
                "wiring_plus_key_construction"):
        text = dict(rr.D10_13_REQUIRED)[key]
        without = markdown.replace(f"`{key}`", "").replace(
            rr.normalise(text)[:70], "")
        with pytest.raises(st.RefusalError, match=key):
            rr.assert_coverage(without, manifest)


def test_coverage_check_fails_when_a_consequence_clause_is_truncated(built):
    """Astra gate 2 round 1 MUST-FIX 5, reproduced exactly.

    Astra removed the two-seed limitation's confirmation / LANDED qualification
    while preserving its prefix, and coverage still passed. The complete
    normalised entry, consequence clause included, is now what is checked.
    """
    markdown = _render(built)
    manifest = rr.coverage_manifest(built["config"])
    text = dict(rr.D10_13_REQUIRED)["two_seed_weakness"]
    assert "can never be LANDED" in text
    truncated = text.split("Estimand")[0].strip()
    mutated = markdown.replace(rr.normalise(text), truncated)
    assert mutated != markdown
    assert truncated in mutated  # the prefix survives, as in Astra's attack
    with pytest.raises(st.RefusalError, match="two_seed_weakness"):
        rr.assert_coverage(mutated, manifest)


def test_coverage_check_fails_when_the_historical_consumption_item_is_dropped(
        built):
    """The unresolved historical-consumption disposition is a required § 9
    entry (MUST-FIX 2): D10.16(0) is what keeps the cohort closed."""
    markdown = _render(built)
    manifest = rr.coverage_manifest(built["config"])
    key = "unresolved_historical_consumption"
    text = dict(rr.D10_13_REQUIRED)[key]
    assert key in {entry_key for entry_key, _ in rr.D10_13_REQUIRED}
    without = markdown.replace(f"`{key}`", "").replace(rr.normalise(text), "")
    with pytest.raises(st.RefusalError, match=key):
        rr.assert_coverage(without, manifest)


def test_coverage_is_checked_inside_section_9_not_anywhere_in_the_document(
        built):
    """An entry rendered in some other section no longer counts as coverage."""
    markdown = _render(built)
    manifest = rr.coverage_manifest(built["config"])
    key = "global_prior_not_as_of"
    text = rr.normalise(dict(rr.D10_13_REQUIRED)[key])
    section = rr.section_9_body(markdown)
    assert section is not None and text in rr.normalise(section)
    # Move the entry out of § 9 and into § 1: it is still in the document.
    moved = markdown.replace(f"- `{key}` — {text}", "")
    moved = moved.replace("## 1. Question",
                          f"## 1. Question\n\n- `{key}` — {text}\n")
    assert text in rr.normalise(moved)
    with pytest.raises(st.RefusalError, match=key):
        rr.assert_coverage(moved, manifest)


def test_coverage_refuses_when_section_9_is_absent(built):
    markdown = _render(built)
    manifest = rr.coverage_manifest(built["config"])
    with pytest.raises(st.RefusalError, match="section heading"):
        rr.assert_coverage(markdown.replace(rr.SECTION_9_HEADING, "## 9. x"),
                           manifest)


def test_the_unsupported_config_conclusion_is_corrected_and_disclosed(
        built, tmp_path):
    """MUST-FIX 4: the venue-feature level-effect claim is removed from the
    prose, the config itself is untouched, and the swap is disclosed."""
    original = ("venue history in the frame is not recency-weighted (TODO.md "
                "backlog item); every arm inherits the same feature, so it is "
                "a level effect on all of them rather than a per-arm "
                "advantage")
    config = json.loads(json.dumps(built["config"], default=str))
    config["known_limitations"] = list(config["known_limitations"]) + [original]
    path = tmp_path / "config_with_the_claim.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))

    clause = ("every arm inherits the same feature, so it is a level effect "
              "on all of them rather than a per-arm advantage")
    reloaded = yaml.safe_load(path.read_text())
    assert any(clause in str(item) for item in reloaded["known_limitations"])
    markdown = _render(built, config_path=path)
    section = rr.normalise(rr.section_9_body(markdown))
    assert "Corrections applied to config-sourced wording above" in section
    assert ("whether its effect is the SAME across architectures is "
            "unmeasured") in section
    # The unsupported clause appears only inside the disclosure of what was
    # replaced, never as a statement of the report's own.
    assert section.count(clause) == 1
    assert clause in section.split("Corrections applied")[1]
    # The coverage manifest asks for the CORRECTED text, so a report that
    # restated the config verbatim would now fail the check.
    manifest = rr.coverage_manifest(config)
    entry = next(e for e in manifest
                 if e["key"] == f"limitation:{len(config['known_limitations']) - 1}")
    assert "no level effect is claimed" in entry["must_contain"][1]
    with pytest.raises(st.RefusalError, match="coverage check failed"):
        rr.assert_coverage(markdown.replace(
            "no level effect is claimed", "it is a level effect"), manifest)


def test_no_unsupported_conclusion_survives_in_the_prose(built):
    markdown = _render(built)
    for banned in ("ownership is not isolated from alignment anywhere "
                   "tonight",
                   "They differ only in what they are allowed to remember",
                   "Sequence adds nothing over the production prior"):
        assert banned not in markdown, banned


def test_a_null_is_never_reported_as_an_absence(built, tmp_path):
    """MUST-FIX 4 wording: an unresolved interval is "no benefit was
    established at this resolution", never "adds nothing"."""
    stats = json.loads(built["stats_path"].read_text())
    for key, contrast in (("fox-fixed_decay@all", ("fox", "fixed_decay")),
                          ("residual_t1-residual_mlp@all",
                           ("residual_t1", "residual_mlp"))):
        stats.setdefault("contrasts", {})[key] = {
            "candidate": contrast[0], "reference": contrast[1],
            "slice": "all", "role": "family", "available": True,
            "per_seed_points": {"7": -0.00001, "13": 0.00002},
            "estimand_i": {}, "estimand_ii": {
                "point": -0.00003, "ci95": [-0.0006, 0.00044],
                "u95": 0.00044, "l95": -0.0006,
                "ci_clean_favourable": False, "descriptive_only": False,
                "p_display": "0.8680"}}
    path = tmp_path / "nulls.json"
    path.write_text(json.dumps(stats))
    markdown = _render(built, stats_path=path)
    assert "no detected benefit over fixed decay" in markdown
    assert ("no incremental benefit of residual T1 over residual MLP was "
            "established at this resolution") in markdown


def test_the_cohort_unlock_conditions_are_rendered_in_full(built):
    """MUST-FIX 2: nothing in D10.16 may be abbreviated."""
    section = _render(built).split("## 10. In plain language")[1]
    for required in (
            "historical-consumption question must be settled first",
            "D10.16(0)",
            "clean training ancestry does not prove untouched evaluation "
            "status",
            "29, 42 and 101",
            "all five",
            "frozen before any new-seed result is inspected",
            "4/5 favourable",
            "analysis** freeze",
            "provenance is re-verified unchanged",
            "one** frozen scoring batch",
            "Partial-exposure recovery rule",
            "discloses the partial exposure",
            "never claims a fresh untouched read",
            "the cohort is left unopened",
            "five seeds alone cannot unlock this cohort"):
        assert required in section, required


def test_render_refuses_when_a_config_entry_is_not_rendered(built, tmp_path,
                                                            monkeypatch):
    """A new config limitation the § 9 renderer does not emit fails the render."""
    config = dict(built["config"])
    config["known_limitations"] = list(config["known_limitations"]) + [
        "an entirely new limitation nobody rendered yet"]
    path = tmp_path / "config_extra.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))

    real = rr.section_limitations

    def without_the_new_one(cfg, *args, **kwargs):
        trimmed = dict(cfg)
        trimmed["known_limitations"] = cfg["known_limitations"][:-1]
        return real(trimmed, *args, **kwargs)

    monkeypatch.setattr(rr, "section_limitations", without_the_new_one)
    with pytest.raises(st.RefusalError, match="coverage check failed"):
        _render(built, config_path=path)


# ---------------------------------------------------------------------------
# § 10 plain language (D10.14)
# ---------------------------------------------------------------------------

def test_plain_language_carries_the_required_statements(built):
    markdown = _render(built)
    section = markdown.split("## 10. In plain language")[1]
    assert "teacher-forced" in section
    assert "not rollout" in section
    assert "not market performance" in section
    assert "Best observed validation configuration" in section
    assert "Why no arm advances tonight" in section
    assert "Cohort confirmation is pending" in section
    assert "There is no market claim" in section
    assert "There is no LANDED verdict" in section
    assert "The user's verdict is outstanding" in section
    assert "The exact next step" in section
    assert "29, 42 and 101" in section


def test_no_further_sequence_gain_sentence_only_when_nothing_is_ci_clean(
        built, tmp_path):
    stats = json.loads(built["stats_path"].read_text())
    # Everything is descriptive on this tiny fixture, so no arm is CI-clean
    # and no arm-minus-mlp contrast is evaluable either: the sentence must NOT
    # be asserted, and the report says the readout is NOT_EVALUABLE.
    markdown = _render(built)
    gain = rr._sequence_gain(stats)
    assert gain["ci_clean_favourable"] == []
    if not gain["evaluable"]:
        assert rr.NO_SEQUENCE_GAIN_SENTENCE not in markdown
        assert "`NOT_EVALUABLE` tonight" in markdown

    # Force one evaluable, non-CI-clean arm and the exact sentence appears,
    # immediately qualified.
    record = stats["contrasts"]["full-mlp@all"]
    record["available"] = True
    for readout in ("seed_7", "seed_13"):
        record["estimand_i"][readout] = {"ci_clean_favourable": False,
                                         "descriptive_only": False,
                                         "point": -0.01,
                                         "ci95": [-0.05, 0.03],
                                         "u95": 0.03, "l95": -0.05,
                                         "p_display": "0.4000"}
    record["estimand_ii"] = dict(record["estimand_i"]["seed_7"])
    path = tmp_path / "forced.json"
    path.write_text(json.dumps(stats))
    markdown = _render(built, stats_path=path)
    assert rr.NO_SEQUENCE_GAIN_SENTENCE in markdown
    assert rr.NO_SEQUENCE_GAIN_QUALIFIER in markdown
    index = markdown.index(rr.NO_SEQUENCE_GAIN_SENTENCE)
    assert markdown.index(rr.NO_SEQUENCE_GAIN_QUALIFIER) - index < 400


def test_plain_language_reports_a_ci_clean_arm_without_advancing_it(built,
                                                                   tmp_path):
    stats = json.loads(built["stats_path"].read_text())
    record = stats["contrasts"]["full-mlp@all"]
    record["available"] = True
    clean = {"ci_clean_favourable": True, "descriptive_only": False,
             "point": -0.05, "ci95": [-0.08, -0.02], "u95": -0.02,
             "l95": -0.08, "p_display": "0.0010"}
    record["estimand_i"] = {"seed_7": dict(clean), "seed_13": dict(clean)}
    record["estimand_ii"] = dict(clean)
    path = tmp_path / "clean.json"
    path.write_text(json.dumps(stats))
    markdown = _render(built, stats_path=path)
    assert rr.NO_SEQUENCE_GAIN_SENTENCE not in markdown
    assert "CI-clean favourable against the token MLP" in markdown
    assert "No arm advances." in markdown


# ---------------------------------------------------------------------------
# § 11 falsification (D10.15)
# ---------------------------------------------------------------------------

def test_falsification_wording_is_registered_and_conditional(built):
    markdown = _render(built)
    section = markdown.split("## 11. Falsification wording")[1]
    assert 'No result tonight licenses "X is the cause".' in section
    assert "learned forgetting as an explanation" in section
    assert "cannot isolate ownership from alignment" in section
    assert "unresolved evidence" in section
    assert "neither proposed mechanism is supported at this resolution" in (
        section.lower())
    # The death slice is evaluable but descriptive on this fixture, so its
    # interval is unresolved and the report says the harm was not reproduced.
    assert "`full − mlp` on `death` reads" in section
    assert "not reproduced" in section
    assert "neither reproduces nor fails to reproduce" not in section


def test_falsification_wording_claims_nothing_when_death_is_not_evaluable(
        built, tmp_path):
    stats = json.loads(built["stats_path"].read_text())
    stats["contrasts"]["full-mlp@death"]["available"] = False
    stats["contrasts"]["full-mlp@death"]["estimand_ii"] = None
    path = tmp_path / "no_death.json"
    path.write_text(json.dumps(stats))
    section = _render(built, stats_path=path).split(
        "## 11. Falsification wording")[1]
    assert "neither reproduces nor fails to reproduce" in section
    assert "not reproduced** on" not in section
    assert "log_verdict.py` was not called" in section


# ---------------------------------------------------------------------------
# determinism, CLI, access
# ---------------------------------------------------------------------------

def test_render_is_deterministic_apart_from_the_declared_timestamp(built):
    first = _render(built)
    second = _render(built)
    assert first == second
    stamp = built["stats"]["generated_at_utc"]
    assert stamp in first
    assert first.count(stamp) == 1


def test_cli_writes_the_report(built, capsys):
    code = rr.main(["--stats-json", str(built["stats_path"]),
                    "--config", str(built["config_path"]),
                    "--k-selection", str(built["k_path"]),
                    "--dependency-dir", str(built["dependency"]),
                    "--out", str(built["out"])])
    assert code == 0
    assert built["out"].exists()
    assert "## 10. In plain language" in built["out"].read_text()
    assert "wrote" in capsys.readouterr().out


def test_cli_returns_two_on_a_refusal(built, tmp_path, capsys, monkeypatch):
    real = rr.coverage_manifest

    def with_an_unrenderable_entry(config, *args, **kwargs):
        return list(real(config, *args, **kwargs)) + [
            {"key": "d10.13:never_rendered_key",
             "must_contain": ["`never_rendered_key`"]}]

    monkeypatch.setattr(rr, "coverage_manifest", with_an_unrenderable_entry)
    code = rr.main(["--stats-json", str(built["stats_path"]),
                    "--config", str(built["config_path"]),
                    "--k-selection", str(built["k_path"]),
                    "--dependency-dir", str(built["dependency"]),
                    "--out", str(tmp_path / "never.md")])
    assert code == 2
    assert "REFUSED" in capsys.readouterr().err
    assert not (tmp_path / "never.md").exists()


def test_report_handles_a_missing_k_record_and_an_empty_dependency_dir(
        built, tmp_path):
    empty = tmp_path / "empty_dependency"
    empty.mkdir()
    markdown = _render(built, k_path=tmp_path / "absent.json",
                       dependency_dir=empty)
    assert "The D9 selection record has not been written" in markdown
    assert "no dependency JSON found" in markdown
    assert "## 10. In plain language" in markdown


def test_report_handles_a_blocked_k_selection(built, tmp_path, frame_dir):
    root = _k_runs(tmp_path, {
        "0": {7: 1.6, 13: 1.6}, "6": {7: 1.6, 13: 1.6},
        "12": {7: 1.6, 13: 1.6}, "30": {7: 1.6, 13: 1.6},
        "unr": {7: 1.6, 13: None}}, frame_dir)
    record = st.k_sweep(root, built["config"], seeds=SEEDS)
    path = tmp_path / "blocked.json"
    path.write_text(json.dumps(record, default=str))
    markdown = _render(built, k_path=path)
    assert "**BLOCKED_INCOMPLETE**" in markdown
    assert "`same_entity_unr`" in markdown


def test_rendering_opens_no_forbidden_path(built, monkeypatch):
    # pathlib routes through `io.open`, not `builtins.open`, so both are
    # instrumented: every real file this render touches is recorded.
    opened: list[str] = []
    real_builtin, real_io = builtins.open, io.open

    def spy(real):
        def wrapper(file, *args, **kwargs):
            opened.append(str(file))
            return real(file, *args, **kwargs)
        return wrapper

    monkeypatch.setattr(builtins, "open", spy(real_builtin))
    monkeypatch.setattr(io, "open", spy(real_io))
    _render(built)
    monkeypatch.setattr(builtins, "open", real_builtin)
    monkeypatch.setattr(io, "open", real_io)
    assert opened
    for path in opened:
        lowered = path.lower()
        for token in ("seq_stage2/cohort", "seq_stage2/smoke", "data/golden",
                      "forward_holdout"):
            assert token not in lowered, path


def test_render_refuses_a_forbidden_output_or_dependency_path(built):
    with pytest.raises(st.RefusalError, match="refusing to open"):
        rr.load_dependency(REPO / "models" / "embeddings" / "seq_stage2"
                           / "cohort")
    with pytest.raises(st.RefusalError, match="refusing to open"):
        _render(built, stats_path=REPO / "models" / "embeddings"
                / "seq_stage2" / "smoke" / "x.json")


# ---------------------------------------------------------------------------
# Astra gate 1 round 3 report rulings: the dependency section is PENDING, a
# failed or missing certificate BLOCKS, and the innings_2 / chase equivalence
# is disclosed
# ---------------------------------------------------------------------------

def test_the_dependency_heading_is_derived_from_verified_coverage(built,
                                                                 tmp_path):
    """SHOULD 8: the heading and the disposition come from the verified
    coverage, so the report can never say "pending" and "complete" at once."""
    markdown = _render(built)
    assert "recertification **complete on trained checkpoints at both " \
           "registered seeds" in markdown
    assert "recertification **pending**" not in markdown
    assert "Coverage complete: **yes**" in markdown

    # Remove one seed's certificate and the heading flips back to pending.
    (built["dependency"] / "recert" / "same_entity_k30_seed13.json").unlink()
    markdown = _render(built)
    assert "recertification **pending**" in markdown
    assert "Coverage complete: **no**" in markdown


def test_coverage_is_not_complete_from_four_seed_7_certificates_and_no_controls(
        tmp_path):
    """Astra gate 2 round 1 MUST-FIX 1, reproduced exactly.

    Astra obtained `trained_checkpoint_coverage_complete=True` from the four
    seed-7 certificates alone, with no positive controls at all. That must now
    fail, and every masked arm must be BLOCKED: two for the missing registered
    control, two for the missing second seed.
    """
    dependency = _write_dependency_evidence(tmp_path, seeds=(7,),
                                            controls=False)
    certificates = rr.dependency_certificates(rr.load_dependency(dependency))
    assert certificates["trained_checkpoint_coverage_complete"] is False
    assert certificates["trained_checkpoint_coverage"] == []
    assert certificates["registered_controls_complete"] is False
    assert set(certificates["blocked"]) == {
        "same_entity_k0", "same_entity_k30", "same_entity_unr",
        "recency_k30"}
    statuses = {name: block["status"]
                for name, block in certificates["by_config"].items()}
    assert statuses["recency_k30"] == "CONTROL_MISSING"
    assert statuses["same_entity_k30"] == "CONTROL_MISSING"
    assert statuses["same_entity_k0"] == "TRAINED_SEEDS_INCOMPLETE"
    assert statuses["same_entity_unr"] == "TRAINED_SEEDS_INCOMPLETE"


def test_both_seeds_are_required_for_trained_coverage(tmp_path):
    full = rr.dependency_certificates(
        rr.load_dependency(_write_dependency_evidence(tmp_path / "a")))
    assert full["trained_checkpoint_coverage_complete"] is True
    assert full["blocked"] == {}
    assert full["by_config"]["same_entity_k30"]["trained_seeds"] == [7, 13]

    one = rr.dependency_certificates(
        rr.load_dependency(_write_dependency_evidence(tmp_path / "b",
                                                      seeds=(7,))))
    assert one["trained_checkpoint_coverage_complete"] is False
    assert "seed(s) 13" in one["by_config"]["same_entity_k0"]["reason"]


def test_the_real_controls_are_matched_through_the_dependency_set(built):
    """The registered controls carry `arm` full / aligned_hist and no k, so
    matching on arm/k skipped them entirely (MUST-FIX 1)."""
    certificates = rr.dependency_certificates(
        rr.load_dependency(built["dependency"]))
    recency = certificates["by_config"]["recency_k30"]
    same = certificates["by_config"]["same_entity_k30"]
    assert [c["arm"] for c in recency["controls"]] == ["full"]
    assert [c["arm"] for c in same["controls"]] == ["aligned_hist"]
    assert recency["controls_fired"] == 1 and same["controls_fired"] == 1
    markdown = _render(built)
    assert "`full` vs S(i) of recency k=30 → fired" in markdown
    assert "`aligned_hist` vs S(i) of same_entity k=30 → fired" in markdown


def test_a_control_of_the_wrong_arm_does_not_satisfy_the_registration(built):
    path = built["dependency"] / "full.json"
    record = json.loads(path.read_text())
    record["arm"] = "fixed_decay"
    path.write_text(json.dumps(record))
    certificates = rr.dependency_certificates(
        rr.load_dependency(built["dependency"]))
    assert certificates["by_config"]["recency_k30"]["status"] == (
        "CONTROL_DID_NOT_FIRE")
    assert "recency_k30" in certificates["blocked"]
    assert certificates["trained_checkpoint_coverage_complete"] is False


def test_a_certificate_whose_checkpoint_md5_does_not_authenticate_blocks(built):
    path = built["dependency"] / "recert" / "same_entity_unr_seed7.json"
    record = json.loads(path.read_text())
    record["checkpoint_md5"] = "0" * 32
    path.write_text(json.dumps(record))
    certificates = rr.dependency_certificates(
        rr.load_dependency(built["dependency"]))
    assert certificates["by_config"]["same_entity_unr"]["status"] == (
        "CHECKPOINT_NOT_AUTHENTICATED")
    assert "same_entity_unr" in certificates["blocked"]
    markdown = _render(built)
    assert "MD5_MISMATCH" in markdown


def test_a_certificate_under_unregistered_settings_blocks(built):
    path = built["dependency"] / "recert" / "same_entity_k0_seed7.json"
    record = json.loads(path.read_text())
    record["n_targets"] = 200
    path.write_text(json.dumps(record))
    certificates = rr.dependency_certificates(
        rr.load_dependency(built["dependency"]))
    block = certificates["by_config"]["same_entity_k0"]
    assert block["status"] == "SETTINGS_MISMATCH"
    assert "n_targets=200" in block["reason"]


def test_a_certificate_scored_against_another_arms_dependency_set_blocks(built):
    path = built["dependency"] / "recert" / "same_entity_k30_seed7.json"
    record = json.loads(path.read_text())
    record["dependency_set_k"] = "unr"
    path.write_text(json.dumps(record))
    certificates = rr.dependency_certificates(
        rr.load_dependency(built["dependency"]))
    block = certificates["by_config"]["same_entity_k30"]
    assert block["status"] == "SETTINGS_MISMATCH"
    assert "not its own S(i)" in block["reason"]


def test_a_failed_masked_arm_certificate_forces_not_evaluable(built):
    """Astra: a failed certificate must BLOCK the affected interpretation and
    eligibility, not merely appear in a table."""
    path = built["dependency"] / "recert" / "same_entity_k30_seed7.json"
    record = json.loads(path.read_text())
    record["pass"] = False
    record["max_abs_delta"] = 0.42
    path.write_text(json.dumps(record))
    certificates = rr.dependency_certificates(
        rr.load_dependency(built["dependency"]))
    assert certificates["by_config"]["same_entity_k30"]["status"] == "FAILED"
    markdown = _render(built)
    assert rr.blocking_note("same_entity_k30",
                            certificates["blocked"]) in markdown
    # Its family screen status reads NOT_EVALUABLE on every readout.
    block = markdown.split("**same_entity_k30** —")[1:]
    assert block, "the family is not rendered at all"
    for chunk in block[:3]:
        assert "screen status `NOT_EVALUABLE`" in chunk.splitlines()[0]
    assert "family forced to `NOT_EVALUABLE`; arm not eligible" in markdown
    assert "not eligible, because a masked-arm dependency certificate is "\
           "failed or missing" in markdown


def test_a_missing_masked_arm_certificate_also_blocks(built):
    for name in ("same_entity_k0_seed7.json", "same_entity_k0_seed13.json"):
        (built["dependency"] / "recert" / name).unlink()
    certificates = rr.dependency_certificates(
        rr.load_dependency(built["dependency"]))
    assert certificates["by_config"]["same_entity_k0"]["status"] == "MISSING"
    assert "same_entity_k0" in certificates["blocked"]
    assert certificates["trained_checkpoint_coverage_complete"] is False
    markdown = _render(built)
    assert "**Dependency certificate blocks `same_entity_k0`**" in markdown


def _smoke_only_dependency(tmp_path: Path) -> Path:
    """A tree with NO trained certificate: only the one-epoch smoke records."""
    dependency = _write_dependency_evidence(tmp_path, seeds=())
    for arm, k, config_id in MASKED_CERTIFICATES:
        (dependency / f"{config_id}.json").write_text(json.dumps(
            _masked_record(
                arm, k,
                f"models/embeddings/seq_stage2/smoke/{config_id}/seed_7",
                "0" * 32)))
    return dependency


def test_a_smoke_only_certificate_is_present_but_blocks_eligibility(tmp_path):
    """Astra gate 2 round 2: a smoke record may not carry eligibility.

    Structural masking from the smoke weights is PRESENT and is never
    trained-checkpoint coverage — and when the trained recertifications are
    absent altogether, the arm must BLOCK rather than inherit `PASS` from the
    smoke record. Previously the trained-seed shortfall was only checked for an
    arm that already held at least one trained certificate, so a tree in which
    trained certificates had disappeared and smoke records alone remained read
    `PASS`.
    """
    dependency = _smoke_only_dependency(tmp_path)
    certificates = rr.dependency_certificates(rr.load_dependency(dependency))
    for config_id in certificates["required"]:
        block = certificates["by_config"][config_id]
        assert block["smoke_only"] is True
        assert block["status"] == "TRAINED_SEEDS_INCOMPLETE", config_id
        assert block["coverage"] == "smoke checkpoint only (structural)"
        assert "no seed at all" in block["reason"]
        assert "one-epoch smoke record" in block["reason"]
        assert config_id in certificates["blocked"]
    assert set(certificates["blocked"]) == set(certificates["required"])
    assert certificates["trained_checkpoint_coverage"] == []
    assert certificates["trained_checkpoint_coverage_complete"] is False


def test_one_trained_seed_still_blocks_and_says_which_seed_is_held(built):
    """The one-trained-seed case keeps its own, more specific wording."""
    (built["dependency"] / "recert" / "recency_k30_seed13.json").unlink()
    certificates = rr.dependency_certificates(
        rr.load_dependency(built["dependency"]))
    block = certificates["by_config"]["recency_k30"]
    assert block["status"] == "TRAINED_SEEDS_INCOMPLETE"
    assert block["trained_seeds"] == [7]
    assert "at 7, not at seed(s) 13" in block["reason"]
    assert "one-epoch smoke record" not in block["reason"]


def test_smoke_only_blocking_propagates_through_the_whole_report(built,
                                                                tmp_path):
    """The block must reach the results tables, the gates, § 7 and § 11.

    Astra gate 2 round 2 required the smoke-only block to be verified all the
    way through, not merely in `dependency_certificates`.
    """
    dependency = _smoke_only_dependency(tmp_path / "smoke_only")
    certificates = rr.dependency_certificates(rr.load_dependency(dependency))
    blocked = certificates["blocked"]
    assert set(blocked) == {"same_entity_k0", "same_entity_k30",
                            "same_entity_unr", "recency_k30"}
    markdown = _render(built, dependency_dir=dependency)

    # § 4 results: every blocked family carries the blocking note and reads
    # NOT_EVALUABLE on every readout, and no eligibility survives.
    families = {f["candidate"] for f in built["stats"]["families"]}
    for config_id in sorted(set(blocked) & families):
        assert rr.blocking_note(config_id, blocked) in markdown
        chunks = markdown.split(f"**{config_id}** —")[1:]
        assert chunks, f"{config_id} family is not rendered at all"
        for chunk in chunks[:len(rr.READOUTS)]:
            assert "screen status `NOT_EVALUABLE`" in chunk.splitlines()[0]
    assert "family forced to `NOT_EVALUABLE`; arm not eligible" in markdown

    # § 8 gates.
    gates = markdown.split("## 8. Non-inferiority gates")[1].split("## 9.")[0]
    assert f"read `{rr.DEPENDENCY_BLOCKED_STATUS}` regardless of their " \
           "intervals" in gates
    for config_id in blocked:
        assert f"`{config_id}`" in gates

    # § 7 mechanism contrasts.
    mechanism = markdown.split("## 7. Mechanism contrasts")[1].split("## 8.")[0]
    assert "**Blocked mechanism contrasts.**" in mechanism
    assert "`same_entity_k30 − recency_k30`" in mechanism
    assert "`same_entity_unr − aligned_hist_rf`" in mechanism

    # § 10 plain language and § 11 falsification.
    assert "**Blocked by dependency certificates.**" in markdown
    falsification = markdown.split("## 11. Falsification wording")[1]
    assert "NOT_EVALUABLE (dependency certificate blocks an endpoint)" in (
        falsification)

    # And no smoke-only arm is CI-clean anywhere in the plain-language summary.
    gain = rr._sequence_gain(built["stats"], blocked)
    for config_id in blocked:
        assert config_id not in gain["evaluable"]
        assert config_id not in gain["ci_clean_favourable"]


def test_a_positive_control_that_did_not_fire_blocks(built):
    path = built["dependency"] / "aligned_hist.json"
    record = json.loads(path.read_text())
    record["positive_control_observed"] = False
    record["max_abs_delta"] = 0.0
    path.write_text(json.dumps(record))
    certificates = rr.dependency_certificates(
        rr.load_dependency(built["dependency"]))
    assert certificates["by_config"]["same_entity_k30"]["status"] == (
        "CONTROL_DID_NOT_FIRE")
    assert "same_entity_k30" in certificates["blocked"]
    assert certificates["trained_checkpoint_coverage_complete"] is False


def test_load_dependency_reads_the_recert_subdirectory(built):
    """MUST-FIX 6: loading only immediate *.json let the command line choose
    which evidence existed."""
    records = rr.load_dependency(built["dependency"])
    paths = {Path(str(r.get("checkpoint"))).parent.name for r in records}
    assert len(records) == 10  # 4 arms x 2 seeds + 2 controls
    assert paths
    only_top = sorted((built["dependency"]).glob("*.json"))
    assert len(only_top) == 2, "the controls alone live at the top level"


def test_blocking_reaches_the_mechanism_and_falsification_sections(built):
    """MUST-FIX 1: blocking previously stopped at the results and gate
    sections, so a blocked endpoint still read as an interval in § 7 and § 11."""
    for name in ("recency_k30.json",):
        pass
    path = built["dependency"] / "recert" / "same_entity_unr_seed13.json"
    record = json.loads(path.read_text())
    record["pass"] = False
    path.write_text(json.dumps(record))
    markdown = _render(built)
    mechanism = markdown.split("## 7. Mechanism contrasts")[1].split("## 8.")[0]
    assert "**Blocked mechanism contrasts.**" in mechanism
    assert "`same_entity_unr − aligned_hist_rf`" in mechanism
    assert "dependency certificate blocks" in mechanism
    falsification = markdown.split("## 11. Falsification wording")[1]
    assert "NOT_EVALUABLE (dependency certificate blocks an endpoint)" in (
        falsification)


def test_a_blocked_arm_cannot_be_reported_as_ci_clean(built):
    stats = built["stats"]
    gain_open = rr._sequence_gain(stats, {})
    gain_blocked = rr._sequence_gain(stats, {name: "blocked" for name
                                            in gain_open["evaluable"]})
    assert gain_open["evaluable"]
    assert gain_blocked["evaluable"] == []
    assert gain_blocked["ci_clean_favourable"] == []
    assert gain_blocked["blocked_by_dependency_certificate"] == sorted(
        gain_open["evaluable"])


def test_the_report_discloses_the_innings_2_and_chase_equivalence(built):
    """Astra: keep both registered predicates, disclose the equivalence, and
    never present them as independent corroboration."""
    equivalent = rr._equivalent_slices(built["stats"])
    assert equivalent, "the fixture frame should make the two coincide"
    assert equivalent[0][0] == "innings_2" and equivalent[0][1] == "chase"
    markdown = _render(built)
    assert ("**Disclosure — `innings_2` and `chase` are the same rows on this "
            "frame.**") in markdown
    assert "never** independent corroboration" in markdown
    assert "only `chase` is a three-member family member" in markdown


def test_diverging_slices_are_not_reported_as_equivalent(built):
    """SHOULD 7: identity comes from the mask digest, not the counts.

    Equal row, match and block counts are NOT enough, and a differing digest
    is enough on its own.
    """
    stats = json.loads(json.dumps(built["stats"], default=str))
    assert rr._equivalent_slices(stats)[0][3] == "mask_sha256"
    stats["slices"]["stats"]["chase"]["mask_sha256"] = "f" * 64
    assert rr._equivalent_slices(stats) == []
    # Counts still agree exactly; the digest is what decides.
    for field in ("n_rows", "n_matches", "n_blocks"):
        assert (stats["slices"]["stats"]["chase"][field]
                == stats["slices"]["stats"]["innings_2"][field])


def test_equivalence_falls_back_to_counts_and_says_so(built):
    """A statistics JSON predating the digest still gets the disclosure, but
    the report must label the basis as counts-only."""
    stats = json.loads(json.dumps(built["stats"], default=str))
    for name in ("innings_2", "chase"):
        stats["slices"]["stats"][name].pop("mask_sha256", None)
    assert rr._equivalent_slices(stats)[0][3] == "counts_only"


# ---------------------------------------------------------------------------
# Five seeds: the PROSE, not only the numbers (Astra gate 2 round 2 follow-up)
#
# `readouts_for`/`contract.readouts` made every Holm table, gate and screen
# seed-derived, but the report still described itself as "two seeds" in a dozen
# authored sentences and § 5 / § 6 / § 7 / § 8 still carried literal `seed-7`
# and `seed-13` columns reading `points['7']` and `points['13']`. These tests
# render a real five-seed statistics JSON and a real five-seed D9 record.
#
# Scope note: the synthetic config here is the fixture config with its seeds
# changed. The renderer still QUOTES config-sourced provenance verbatim (§ 3's
# k-selection rule, § 9's registered entries), and the registered five-seed
# config's own launch-era text does contain the historical phrase "two-seed";
# quoting registered provenance is not the renderer describing itself, and is
# deliberately not rewritten. Everything the renderer authors, and everything
# it derives from the statistics and the D9 record, is checked here.
# ---------------------------------------------------------------------------

FIVE_SEEDS = (7, 13, 29, 42, 101)


@pytest.fixture
def built_five(tmp_path, frame_dir, block_source):
    """A five-seed statistics JSON, config and k-selection record on disk."""
    import pandas as pd

    config = _config_payload(frame_dir=frame_dir, same_entity=True)
    config["training"] = dict(config.get("training") or {})
    config["training"]["seeds"] = list(FIVE_SEEDS)
    config_path = tmp_path / "config_five.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    frame = pd.read_parquet(frame_dir / "cricket_data_i7_validation.parquet")
    runs = tmp_path / "runs"
    biases = {"mlp": 0.0, "full": 0.9, "fixed_decay": 0.5, "fox": 0.7}
    for config_id, bias in biases.items():
        for seed in FIVE_SEEDS:
            _write_run(runs, config_id, seed, frame, bias=bias)
        _write_summary(runs, config_id,
                       {seed: 1.5 - bias / 10 + seed / 100000
                        for seed in FIVE_SEEDS})
    for k, config_id in K_IDS.items():
        for seed in FIVE_SEEDS:
            _write_run(runs, config_id, seed, frame, bias=0.4,
                       arm="same_entity", k=(k if k == "unr" else int(k)))
        _write_summary(runs, config_id,
                       {seed: 1.5 + seed / 100000 for seed in FIVE_SEEDS})

    stats = st.compute_statistics(
        config_path, runs, frame_dir, block_source,
        tmp_path / "no_base_logits.npz", reps=50, seed=29, seeds=FIVE_SEEDS,
        expect=dict(EXPECT), expected_families=8)
    stats_path = tmp_path / "stage2_statistics_five.json"
    stats_path.write_text(json.dumps(stats, indent=2, default=str))

    k_record = st.k_sweep(runs, config, seeds=FIVE_SEEDS)
    k_path = tmp_path / "k_selection_five.json"
    k_path.write_text(json.dumps(k_record, indent=2, default=str))

    dependency = _write_dependency_evidence(tmp_path)
    return {"config": config, "config_path": config_path,
            "stats": stats, "stats_path": stats_path,
            "k_record": k_record, "k_path": k_path,
            "dependency": dependency, "out": tmp_path / "REPORT_FIVE.md"}


def _section(markdown: str, heading: str, next_heading: str) -> str:
    return markdown.split(heading)[1].split(next_heading)[0]


def test_a_five_seed_render_describes_itself_as_five_seed(built_five):
    markdown = _render(built_five)
    for banned in ("two seeds", "two-seed", "2 seed"):
        assert banned not in markdown, banned
    assert "five" in markdown
    assert markdown.startswith(
        "# Sequence track — stage 2 report (sixteen configurations, five "
        "seeds, validation only)")
    assert "validation-only five-seed directional screen" in markdown
    assert "Every configuration has all five registered seeds." in markdown


def test_the_five_seed_mechanism_table_has_one_column_per_seed(built_five):
    markdown = _render(built_five)
    section = _section(markdown, "## 7. Mechanism contrasts (D10.5)",
                       "## 8. Non-inferiority gates")
    header = next(line for line in section.splitlines()
                  if line.startswith("| contrast |"))
    for seed in FIVE_SEEDS:
        assert f"seed-{seed} point" in header, seed
    assert header.count("point") == len(FIVE_SEEDS) + 1  # + the mean point
    # The separator row and every body row carry the same column count.
    rows = [line for line in section.splitlines() if line.startswith("|")]
    widths = {line.count("|") for line in rows}
    assert len(widths) == 1, rows


def test_every_five_seed_table_has_one_column_per_seed(built_five):
    markdown = _render(built_five)
    estimands = _section(markdown, "## 5. The two estimands",
                         "## 6. The k sweep")
    for seed in FIVE_SEEDS:
        assert f"seed-{seed} point" in estimands, seed
    ksweep = _section(markdown, "## 6. The k sweep (D9)",
                      "## 7. Mechanism contrasts")
    for seed in FIVE_SEEDS:
        assert f"seed {seed} LL" in ksweep, seed
    for table in (estimands, ksweep):
        rows = [line for line in table.splitlines() if line.startswith("|")]
        assert len({line.count("|") for line in rows}) == 1, rows


def test_every_seed_appears_in_the_five_seed_holm_readouts(built_five):
    markdown = _render(built_five)
    section = _section(markdown, "## 4. Results", "## 5. The two estimands")
    for seed in FIVE_SEEDS:
        assert f"— seed {seed} (estimand i)," in section, seed
    assert "— seed mean (estimand ii)," in section
    # And the 4/5 favourable-direction requirement is stated as enforced.
    assert "favourable-direction requirement (4/5)" in section
    assert "per-seed primary directions" in section


def test_the_five_seed_reasons_drop_the_seed_count_and_keep_the_cohort(
        built_five):
    markdown = _render(built_five)
    section = _section(markdown, "**Why no arm advances tonight.**",
                       "**The exact next step")
    assert "are a directional screen" not in section
    assert "Two reasons" in section
    assert "selection optimism" in section
    assert "**deferred** by the reviewer" in section
    assert "`advances: []`" in section
    assert "the directional-screen reason no longer applies" in section
    # No advancement, at any seed count: the cohort alone settles it.
    assert "cohort_status: DEFERRED_UNOPENED" in markdown
    assert "There is no LANDED verdict" in markdown
    assert "No arm advances." in markdown


def test_the_five_seed_machine_confound_names_every_seed(built_five):
    markdown = _render(built_five)
    section = _section(markdown, "### Machine provenance per run (D8.8)",
                       "### Admission and provenance verification")
    assert "seed 7 trained on the laptop" in section.lower()
    for seed in (13, 29, 42, 101):
        assert str(seed) in section.split("\n\n")[1]
    # § 9 restates the same derived fact, and coverage demands that text.
    rr.assert_coverage(markdown,
                       rr.coverage_manifest(built_five["config"],
                                            built_five["stats"]))


def test_the_five_seed_next_step_is_no_longer_training_the_new_seeds(
        built_five):
    markdown = _render(built_five)
    section = markdown.split("**The exact next step")[1]
    # The registered unlock conditions are restated in full, unabbreviated.
    assert "Seeds **29, 42 and 101** are added" in section
    # But the derived next step is what actually remains.
    assert "the seed count is no longer what is outstanding" in section
    assert "(0) plus (4)–(6)" in section


# `test_the_two_seed_report_of_record_is_byte_unchanged` is superseded by
# `test_the_two_seed_report_of_record_moves_only_where_disclosed` at the end of
# this file: the committed report is no longer byte-reproducible, because the
# twelve D12 recertifications landed in the dependency tree it enumerates and
# because MUST-FIX 4 relabels a shared column. The successor checks, line by
# line, that NOTHING ELSE moved and that no § 5 number changed.


# ---------------------------------------------------------------------------
# Astra gate 2 round 3
#
# MUST-FIX 1 dependency coverage derived from the analysis registration and
# propagated into the 4/5 qualification table; MUST-FIX 2 the two-seed versus
# five-seed comparison and the withdrawal; MUST-FIX 3 the disclosed k-rule
# amendment; MUST-FIX 4 both direction counts, each labelled.
# ---------------------------------------------------------------------------

def test_the_required_checkpoint_seeds_come_from_the_registration(built,
                                                                 built_five):
    """MUST-FIX 1: the requirement is derived, never the hard-coded pair."""
    assert rr.dependency_required_checkpoint_seeds(built["stats"]) == (7, 13)
    assert rr.dependency_required_checkpoint_seeds(
        built_five["stats"]) == FIVE_SEEDS
    # No argument keeps the historical two-seed default.
    assert rr.dependency_required_checkpoint_seeds() == (7, 13)


def test_a_five_seed_render_blocks_on_two_seed_ownership_certificates(
        built_five, tmp_path):
    """MUST-FIX 1 reproduced: five-seed analysis, two-seed certificates.

    Astra: "The four certified masked configurations hold authenticated
    trained-checkpoint certificates only for seeds 7 and 13. Requiring all five
    seeds in memory correctly blocks all four: 12 recertifications are
    missing." That is what must happen, in the report, until they exist.
    """
    two_seed_evidence = _write_dependency_evidence(tmp_path / "two_seed_only")
    certificates = rr.dependency_certificates(
        rr.load_dependency(two_seed_evidence),
        rr.dependency_required_checkpoint_seeds(built_five["stats"]))
    assert certificates["required_checkpoint_seeds"] == list(FIVE_SEEDS)
    assert certificates["trained_checkpoint_coverage_complete"] is False
    assert set(certificates["blocked"]) == {
        "same_entity_k0", "same_entity_k30", "same_entity_unr",
        "recency_k30"}
    for block in certificates["by_config"].values():
        assert block["status"] == "TRAINED_SEEDS_INCOMPLETE"
        assert "29, 42, 101" in block["reason"]
        assert "all five" in block["reason"]

    markdown = _render(built_five, dependency_dir=two_seed_evidence)
    assert "recertification **pending**" in markdown
    assert "Coverage complete: **no**" in markdown
    # And the same evidence passes once every registered seed is certified.
    five = _write_dependency_evidence(tmp_path / "five_seed", seeds=FIVE_SEEDS)
    complete = rr.dependency_certificates(
        rr.load_dependency(five),
        rr.dependency_required_checkpoint_seeds(built_five["stats"]))
    assert complete["trained_checkpoint_coverage_complete"] is True
    assert complete["blocked"] == {}
    markdown = _render(built_five, dependency_dir=five)
    assert ("complete on trained checkpoints at all five registered seeds"
            in markdown)


def test_the_block_reaches_the_four_five_qualification_table(built_five,
                                                            tmp_path):
    """MUST-FIX 1: "propagate blocking through the qualification table"."""
    two_seed_evidence = _write_dependency_evidence(tmp_path / "two_seed_only")
    markdown = _render(built_five, dependency_dir=two_seed_evidence)
    table = _section(markdown,
                     "### The registered favourable-direction requirement",
                     "**full** —")
    blocked = set(rr.dependency_certificates(
        rr.load_dependency(two_seed_evidence),
        rr.dependency_required_checkpoint_seeds(built_five["stats"]))["blocked"])
    families = {family["candidate"]
                for family in built_five["stats"]["families"]}
    targets = sorted(blocked & families)
    assert targets, (blocked, families)
    for candidate in targets:
        row = next(line for line in table.splitlines()
                   if line.startswith(f"| `{candidate}` |"))
        assert row.count(rr.DEPENDENCY_BLOCKED_STATUS) == 3, row
        assert "blocks this arm" in row
    # An unblocked family still shows its counts.
    assert "| `fixed_decay` | 5/5 |" in table
    assert "Qualification is `NOT_EVALUABLE` for" in table


def test_both_direction_counts_are_reported_and_labelled(built_five):
    """MUST-FIX 4: below-threshold is not the same as favourable."""
    markdown = _render(built_five)
    section = _section(markdown, "## 5. The two estimands", "## 6. The k sweep")
    header = next(line for line in section.splitlines()
                  if line.startswith("| contrast |"))
    assert "registered threshold t" in header
    assert "seeds below t" in header
    assert "seeds with a favourable (negative) delta" in header
    assert "| favourable seeds |" not in section
    assert "The two direction counts are different things" in section
    # Every gate row exposes the margin as its threshold and every primary
    # row zero, so no column can be read as the other by accident.
    for line in section.splitlines():
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.split("|")]
        threshold = cells[-6]
        assert threshold == ("+0.00000" if "`all`" in line else "+0.00200"), line


def test_astras_xlstm_death_row_no_longer_reads_five_favourable_seeds():
    """MUST-FIX 4 on the evidence of record.

    Astra: "xLSTM's death row shows 5/5 although only two seeds have negative
    deltas." Both counts must now appear, distinctly labelled.
    """
    stats_path = REPO / "eval_out" / "seq_stage2_5seed" / "stats.json"
    if not stats_path.is_file():
        pytest.skip("the five-seed statistics of record are not present here")
    stats = json.loads(stats_path.read_text())
    record = stats["contrasts"]["xlstm-mlp@death"]
    points = list(record["per_seed_points"].values())
    assert record["favourable_direction_count"] == 5
    assert sum(1 for value in points if value < 0.002) == 5
    assert rr._below_zero_count(record) == 2
    assert rr._direction_threshold(record) == 0.002
    lines = rr.section_estimands(stats)
    row = next(line for line in lines
               if line.startswith("| `xlstm − mlp` | `death`"))
    cells = [cell.strip() for cell in row.split("|")]
    assert (cells[-6], cells[-5], cells[-4]) == ("+0.00200", "5/5", "2/5")


def test_the_primary_zero_threshold_count_is_preserved(built_five):
    """MUST-FIX 4 preserves the primary count and the gate arithmetic."""
    markdown = _render(built_five)
    for record in built_five["stats"]["contrasts"].values():
        if record.get("role") != "family_primary":
            continue
        assert record["direction_count_threshold"] == 0.0
        assert (record["seeds_below_registered_threshold_count"]
                == record["favourable_direction_count"]
                == record["seeds_below_zero_count"])
    section = _section(markdown, "## 4. Results", "## 5. The two estimands")
    assert "seeds with a favourable (negative) primary delta" in section


def _five_seed_config_with_the_registered_k_rule(built_five, tmp_path):
    """The five-seed fixture, with the real config's historical k-rule text."""
    config = json.loads(json.dumps(built_five["config"]))
    config["statistics"]["k_selection"]["rule"] = (
        "ANY k in the registered sweep {0, 6, 12, 30, unr} may win, unr "
        "included: choose " + rr.K_RULE_OLD + "; if no k beats k = 30 by more "
        "than 0.002, keep 30.")
    path = tmp_path / "config_five_real_k_rule.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config, path


def test_the_k_rule_is_corrected_with_a_disclosure_at_five_seeds(built_five,
                                                                tmp_path):
    """MUST-FIX 3: the config body is untouched; § 3's operative rule is not.

    The historical quotation is preserved — in the § 9 disclosure, exactly as
    `known_limitations[1]` is handled — and the operative statement in § 3 names
    the arithmetic mean over all five registered seeds. Tolerance, default,
    sweep and tie rules are not rewritten.
    """
    config, config_path = _five_seed_config_with_the_registered_k_rule(
        built_five, tmp_path)
    markdown = _render(built_five, config_path=config_path)
    rule = _section(markdown, "* **k selection.**", "* **Intervals.**")
    assert rr.K_RULE_OLD not in rule
    assert ("the best (lowest) arithmetic mean validation log loss over all "
            "five registered seeds (7, 13, 29, 42 and 101)") in rule
    assert "keep 30" in rule and "0.002" in rule
    disclosure = markdown.split(
        "### Corrections applied to config-sourced wording above")[1]
    assert "`statistics.k_selection.rule`" in disclosure
    assert f'in place of the config\'s "{rr.K_RULE_OLD}"' in disclosure
    assert "the tolerance (0.002), the keep-30 default" in disclosure
    # The registered config body itself is never rewritten by the renderer.
    assert (yaml.safe_load(config_path.read_text())["statistics"]
            ["k_selection"]["rule"]
            == config["statistics"]["k_selection"]["rule"])


def test_the_k_rule_is_not_corrected_at_the_registered_two_seeds(built,
                                                                tmp_path):
    """At two seeds the quoted rule and the computed rule agree, so nothing
    fires and the two-seed report of record keeps its § 3 verbatim."""
    config = json.loads(json.dumps(built["config"]))
    config["statistics"]["k_selection"]["rule"] = (
        "choose " + rr.K_RULE_OLD + "; otherwise keep 30.")
    path = tmp_path / "config_two_real_k_rule.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    markdown = _render(built, config_path=path)
    assert rr.K_RULE_OLD in markdown
    assert rr.k_rule_correction(built["stats"]) is None
    assert "`statistics.k_selection.rule`" not in markdown


def test_no_prior_stats_means_no_comparison_section(built_five):
    """MUST-FIX 2 is opt-in, so the committed two-seed report cannot move."""
    assert "## 12." not in _render(built_five)
    assert rr.section_prior_comparison(built_five["stats"], None) == []


def test_the_prior_comparison_compares_every_primary_and_withdraws(
        built_five, built):
    """MUST-FIX 2: the comparison and the withdrawal are in the report itself."""
    markdown = _render(built_five, prior_stats_path=built["stats_path"])
    section = markdown.split(rr.PRIOR_HEADING)[1]
    assert "**The extension is not an independent replication:**" in section
    assert "7 and 13 are in both" in section
    # Every registered primary present in both analyses gets a row.
    shared = [record for key, record
              in built_five["stats"]["contrasts"].items()
              if record.get("role") == "family_primary"
              and key in built["stats"]["contrasts"]]
    assert shared
    for record in shared:
        assert f"`{record['candidate']} − {record['reference']}`" in section
    # Nothing unresolved is ever reported as an absence.
    assert rr.UNRESOLVED_PHRASE in section
    assert "not evidence of no benefit" in section
    # And a contrast that was CI-clean and no longer is must be withdrawn by
    # name, whichever contrast that is on this fixture.
    withdrawn = [f"`{r['candidate']} − {r['reference']}`"
                 for key, r in built_five["stats"]["contrasts"].items()
                 if r.get("role") == "family_primary"
                 and key in built["stats"]["contrasts"]
                 and ((built["stats"]["contrasts"][key].get("estimand_ii")
                       or {}).get("ci_clean_favourable"))
                 and not ((r.get("estimand_ii") or {})
                          .get("ci_clean_favourable"))]
    for label in withdrawn:
        assert f"**Withdrawn: {label}.**" in section
        assert "earlier CI-clean claim for this contrast is withdrawn" in section


def test_the_prior_comparison_refuses_an_identical_seed_list(built_five):
    with pytest.raises(rr.RefusalError, match="same seeds"):
        rr.section_prior_comparison(built_five["stats"], built_five["stats"])


def test_the_death_harm_is_reproduction_not_independent_replication(built_five,
                                                                   built):
    """MUST-FIX 2: reproduced IN five-seed validation, and not independent."""
    markdown = _render(built_five, prior_stats_path=built["stats_path"])
    section = markdown.split("**The death-over harm.**")[1]
    assert "**not** an independent replication" in section
    assert "a different frame does not make two analyses independent" in section
    death = built_five["stats"]["contrasts"].get("full-mlp@death") or {}
    if rr._reading(death) == "adverse" and rr._reading(
            built["stats"]["contrasts"].get("full-mlp@death") or {}
    ) == "adverse":
        assert "reproduced in this five-seed validation extension" in section


def test_an_adverse_reading_is_not_called_a_multiplicity_adjusted_finding(
        built_five, built):
    """Astra's standing ruling on `same_entity_k0 − mlp`: CI-clean adverse,
    Holm-adjusted p 0.088, and it does not isolate history."""
    markdown = _render(built_five, prior_stats_path=built["stats_path"])
    if "**Adverse readings and what they are not.**" not in markdown:
        pytest.skip("no primary reads CI-clean adverse on this fixture")
    section = markdown.split("**Adverse readings and what they are not.**")[1]
    assert "Holm-adjusted p" in section
    assert "multiplicity-adjusted finding only where" in section
    assert "No such contrast isolates history" in section


def test_the_real_five_seed_k0_adverse_reading_keeps_its_holm_p():
    """The real numbers: k0 − mlp is CI-clean adverse at Holm p 0.088."""
    stats_path = REPO / "eval_out" / "seq_stage2_5seed" / "stats.json"
    if not stats_path.is_file():
        pytest.skip("the five-seed statistics of record are not present here")
    stats = json.loads(stats_path.read_text())
    record = stats["contrasts"]["same_entity_k0-mlp@all"]
    assert rr._reading(record) == "adverse"
    holm = rr._holm_p(stats, "same_entity_k0-mlp@all", st.JOINT_READOUT)
    assert holm is not None and 0.05 < float(holm) < 0.1


def test_the_two_seed_report_of_record_is_byte_reproducible():
    """The committed two-seed report must re-render byte-for-byte.

    It was re-rendered and re-committed on 2026-09-12 as a disclosed
    correction, for two reasons recorded in the acceptance file:

    1. **The dependency evidence tree grew.** The twelve D12 recertifications
       (seeds 29, 42, 101 for the four masked configurations) landed on disk
       after the first commit, and § 2 enumerates every certificate the
       recorded `--dependency-dir` yields. More evidence, not a changed result.
    2. **Astra gate 3 MUST-FIX 4.** § 5 previously labelled "seeds below the
       +0.002 non-inferiority margin" as "favourable seeds", which overstates
       seed agreement: fifteen two-seed gate rows disagreed between the two
       counts. Both counts are now reported and labelled. The renderer is
       shared, so the correction reaches the two-seed rendering.

    No result moved: every results section was verified numerically identical
    before the correction was committed, including the 1,217 numbers in § 4's
    Holm tables and the 850 in § 8's gates. Only § 2 (twelve more certificates)
    and § 5 (the added labelled column) changed. From that commit onward the
    report of record is exactly reproducible again, which is what this asserts.
    """
    stats = REPO / "eval_out" / "seq_stage2" / "stats.json"
    k_path = REPO / "eval_out" / "seq_stage2" / "k_selection.json"
    committed = (REPO / "research" / "reports" / "embeddings"
                 / "SEQ_STAGE2_REPORT.md")
    if not (stats.is_file() and committed.is_file()):
        pytest.skip("the two-seed evidence of record is not present here")
    markdown = rr.render(stats, rr.DEFAULT_CONFIG,
                         k_path if k_path.is_file() else None,
                         rr.DEFAULT_DEPENDENCY_DIR, committed)
    assert markdown == committed.read_text(), (
        "the two-seed report of record no longer re-renders byte-for-byte; "
        "if that is intended, re-render it, verify section by section that no "
        "results number moved, and record the reason in the acceptance file")


def test_the_only_two_seed_phrases_in_a_five_seed_render_are_quoted_or_compared(
        built_five, built, tmp_path):
    """`test_a_five_seed_render_describes_itself_as_five_seed` bans "two seeds"
    outright, and that stays right for a bare five-seed render. Once the report
    quotes the registered k rule (MUST-FIX 3) and carries the § 12 comparison
    (MUST-FIX 2), the phrase appears legitimately — and only there.
    """
    _, config_path = _five_seed_config_with_the_registered_k_rule(built_five,
                                                                 tmp_path)
    markdown = _render(built_five, config_path=config_path,
                       prior_stats_path=built["stats_path"])
    corrections = markdown.split(
        "### Corrections applied to config-sourced wording above")[1]
    disclosure, comparison = corrections.split(rr.PRIOR_HEADING)
    for line in markdown.splitlines():
        if "two seed" not in line and "two-seed" not in line:
            continue
        assert line in disclosure or line in comparison, line
