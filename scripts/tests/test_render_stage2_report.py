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

    dependency = tmp_path / "dependency"
    dependency.mkdir()
    (dependency / "same_entity_k30.json").write_text(json.dumps({
        "arm": "same_entity", "k": 30,
        "checkpoint": "models/embeddings/seq_stage2/runs/same_entity_k30/"
                      "seed_7",
        "n_targets": 2000, "max_abs_delta": 0.0, "p99_abs_delta": 0.0,
        "n_nonzero_gt_1e-6": 0, "pass": True,
        "positive_control_expected": False}))
    (dependency / "full.json").write_text(json.dumps({
        "arm": "full", "k": None,
        "checkpoint": "models/embeddings/seq_stage2/smoke/full/seed_7",
        "n_targets": 2000, "max_abs_delta": 0.75, "p99_abs_delta": 0.58,
        "n_nonzero_gt_1e-6": 1449, "pass": False,
        "positive_control_expected": True,
        "positive_control_observed": True}))
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


def test_report_shows_the_dependency_certificates_and_marks_smoke(built):
    markdown = _render(built)
    assert "Ownership dependency certificates" in markdown
    assert "(smoke checkpoint)" in markdown
    assert "SEES EXCLUDED-PAST INFORMATION" in markdown
    assert "not specifically multi-layer relay" in markdown


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
    of them.
    """
    import re

    markdown = _render(built)
    sources = (built["stats_path"].read_text() + built["k_path"].read_text()
               + built["config_path"].read_text()
               + "".join(path.read_text()
                         for path in sorted(built["dependency"].glob("*.json"))))
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
    assert len(rr.D10_13_REQUIRED) == 20


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


def test_render_refuses_when_a_config_entry_is_not_rendered(built, tmp_path,
                                                            monkeypatch):
    """A new config limitation the § 9 renderer does not emit fails the render."""
    config = dict(built["config"])
    config["known_limitations"] = list(config["known_limitations"]) + [
        "an entirely new limitation nobody rendered yet"]
    path = tmp_path / "config_extra.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))

    real = rr.section_limitations

    def without_the_new_one(cfg):
        trimmed = dict(cfg)
        trimmed["known_limitations"] = cfg["known_limitations"][:-1]
        return real(trimmed)

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

    def with_an_unrenderable_entry(config):
        return list(real(config)) + [
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

def test_the_dependency_heading_says_pending_not_recertified(built):
    markdown = _render(built)
    assert "recertification **pending**" in markdown
    assert "recertified in D10.12" not in markdown
    assert "This section is **pending**, not recertified" in markdown


def test_a_failed_masked_arm_certificate_forces_not_evaluable(built, tmp_path):
    """Astra: a failed certificate must BLOCK the affected interpretation and
    eligibility, not merely appear in a table."""
    path = built["dependency"] / "same_entity_k30.json"
    record = json.loads(path.read_text())
    record["pass"] = False
    record["max_abs_delta"] = 0.42
    path.write_text(json.dumps(record))
    markdown = _render(built)
    assert rr.blocking_note("same_entity_k30", {
        "same_entity_k30": "its dependency certificate failed for "
                           "`models/embeddings/seq_stage2/runs/"
                           "same_entity_k30/seed_7`"}) in markdown
    # Its family screen status reads NOT_EVALUABLE on every readout.
    block = markdown.split("**same_entity_k30** —")[1:]
    assert block, "the family is not rendered at all"
    for chunk in block[:3]:
        assert "screen status `NOT_EVALUABLE`" in chunk.splitlines()[0]
    assert "family forced to `NOT_EVALUABLE`; arm not eligible" in markdown
    assert "not eligible, because a masked-arm dependency certificate is "\
           "failed or missing" in markdown


def test_a_missing_masked_arm_certificate_also_blocks(built):
    """The fixture holds a certificate for k30 only, so k0 and unr are MISSING."""
    markdown = _render(built)
    certificates = rr.dependency_certificates(
        rr.load_dependency(built["dependency"]))
    assert certificates["blocked"].keys() >= {"same_entity_k0",
                                             "same_entity_unr",
                                             "recency_k30"}
    assert "same_entity_k30" not in certificates["blocked"]
    assert certificates["trained_checkpoint_coverage_complete"] is False
    for config_id in ("same_entity_k0", "same_entity_unr"):
        assert f"**Dependency certificate blocks `{config_id}`**" in markdown


def test_a_smoke_only_certificate_is_present_but_not_trained_coverage(built):
    path = built["dependency"] / "same_entity_k30.json"
    record = json.loads(path.read_text())
    record["checkpoint"] = ("models/embeddings/seq_stage2/smoke/"
                            "same_entity_k30/seed_7")
    path.write_text(json.dumps(record))
    certificates = rr.dependency_certificates(
        rr.load_dependency(built["dependency"]))
    assert "same_entity_k30" not in certificates["blocked"]
    assert certificates["by_config"]["same_entity_k30"]["smoke_only"] is True
    assert "same_entity_k30" not in certificates["trained_checkpoint_coverage"]
    markdown = _render(built)
    assert "smoke checkpoint only (structural)" in markdown


def test_a_positive_control_that_did_not_fire_blocks(built):
    (built["dependency"] / "control.json").write_text(json.dumps({
        "arm": "same_entity", "k": 30,
        "checkpoint": "models/embeddings/seq_stage2/runs/same_entity_k30/"
                      "seed_13",
        "n_targets": 2000, "max_abs_delta": 0.0, "p99_abs_delta": 0.0,
        "n_nonzero_gt_1e-6": 0,
        "positive_control_expected": True,
        "positive_control_observed": False}))
    certificates = rr.dependency_certificates(
        rr.load_dependency(built["dependency"]))
    assert certificates["by_config"]["same_entity_k30"]["status"] == (
        "CONTROL_DID_NOT_FIRE")
    assert "same_entity_k30" in certificates["blocked"]


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
    stats = json.loads(json.dumps(built["stats"], default=str))
    stats["slices"]["stats"]["chase"]["n_rows"] = 1
    assert rr._equivalent_slices(stats) == []
