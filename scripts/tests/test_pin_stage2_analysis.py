"""Tests for `scripts/sequence_track/pin_stage2_analysis.py` (D10.9).

Astra gate 2 round 1 MUST-FIX 6: the analysis freeze must be a separate,
verifiable manifest that records and verifies the ACTUAL invocation — so that a
regeneration from a different `--dependency-dir` is detected — rather than
hashing whatever files happen to be supplied.

Synthetic fixtures only: a statistics JSON built by `stage2_stats` over a tiny
fake runs tree, a rendered report, and a dependency tree of realistic
certificates. Nothing here reads the cohort, the smoke tree or either sealed
holdout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from sequence_track import pin_stage2_analysis as pin  # noqa: E402
from sequence_track import render_stage2_report as rr  # noqa: E402
from sequence_track import stage2_stats as st  # noqa: E402
from tests.test_render_stage2_report import (  # noqa: E402
    _write_dependency_evidence,
)
from tests.test_stage2_stats import (  # noqa: E402
    EXPECT,
    K_IDS,
    SEEDS,
    _config_payload,
    _write_run,
    _write_summary,
)

pytest_plugins = ["tests.test_stage2_stats"]


@pytest.fixture
def analysis(tmp_path, frame_dir, block_source):
    """A complete analysis on disk: stats, k record, report, certificates."""
    import pandas as pd

    config = _config_payload(frame_dir=frame_dir, same_entity=True)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    frame = pd.read_parquet(frame_dir / "cricket_data_i7_validation.parquet")
    runs = tmp_path / "runs"
    for config_id, bias in (("mlp", 0.0), ("full", 0.9),
                            ("fixed_decay", 0.5), ("fox", 0.7)):
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
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, default=str))

    k_path = tmp_path / "k_selection.json"
    k_path.write_text(json.dumps(st.k_sweep(runs, config, seeds=SEEDS),
                                 indent=2, default=str))

    dependency = _write_dependency_evidence(tmp_path)
    report = tmp_path / "REPORT.md"
    report.write_text(rr.render(stats_path, config_path, k_path, dependency,
                                report))
    return {"config_path": config_path, "stats_path": stats_path,
            "k_path": k_path, "dependency": dependency, "report": report,
            "runs": runs, "frame_dir": frame_dir,
            "block_source": block_source,
            "base_logits": tmp_path / "no_base_logits.npz",
            "pin": tmp_path / "analysis_pin.json"}


def _argv(analysis, mode: str, **overrides) -> list[str]:
    args = {"--pin": analysis["pin"],
            "--stats-json": analysis["stats_path"],
            "--k-selection": analysis["k_path"],
            "--config": analysis["config_path"],
            "--dependency-dir": analysis["dependency"],
            "--report": analysis["report"],
            "--runs-root": analysis["runs"],
            "--frame-dir": analysis["frame_dir"],
            "--block-source-dir": analysis["block_source"],
            "--base-logits": analysis["base_logits"],
            "--reps": 50}
    args.update(overrides)
    out = [mode]
    for flag, value in args.items():
        out += [flag, str(value)]
    return out


def _write(analysis, **overrides) -> int:
    return pin.main(_argv(analysis, "--write", **overrides))


# ---------------------------------------------------------------------------
# write / verify
# ---------------------------------------------------------------------------

def test_write_then_verify_passes(analysis, capsys):
    assert _write(analysis) == 0
    capsys.readouterr()
    assert pin.main(["--verify", "--pin", str(analysis["pin"])]) == 0
    assert "OK" in capsys.readouterr().out


def test_the_manifest_records_the_analysis_code_not_the_training_provenance(
        analysis):
    _write(analysis)
    payload = json.loads(analysis["pin"].read_text())
    code = payload["analysis_code"]
    assert set(code) == set(pin.ANALYSIS_SOURCES)
    for relative, digest in code.items():
        assert digest == st.sha256_file(REPO / relative), relative
    # It is the analysis pin, not the training pin.
    assert "scripts/sequence_track/pin_stage2.py" not in code
    assert "training provenance" in payload["what_this_is"]


def test_the_manifest_records_both_invocations_with_every_path(analysis):
    _write(analysis)
    payload = json.loads(analysis["pin"].read_text())
    renderer = payload["invocations"]["renderer"]
    statistics = payload["invocations"]["statistics"]
    for flag in ("--stats-json", "--k-selection", "--config",
                 "--dependency-dir", "--out"):
        assert flag in renderer, flag
    for flag in ("--config", "--runs-root", "--frame-dir",
                 "--block-source-dir", "--base-logits", "--out", "--reps",
                 "--rng-seed", "--seeds"):
        assert flag in statistics, flag
    assert renderer[renderer.index("--dependency-dir") + 1] == st.rel(
        analysis["dependency"])


def test_the_manifest_records_the_evidence_and_the_decisions(analysis):
    _write(analysis)
    payload = json.loads(analysis["pin"].read_text())
    evidence = payload["evidence"]
    assert evidence["statistics"]["sha256"] == st.sha256_file(
        analysis["stats_path"])
    assert evidence["k_selection"]["sha256"] == st.sha256_file(
        analysis["k_path"])
    assert evidence["report"]["sha256"] == st.sha256_file(analysis["report"])
    # Every certificate actually consumed, including the recert subdirectory.
    consumed = {row["path"] for row in evidence["dependency_certificates"]}
    assert len(consumed) == 10
    assert any("recert" in path for path in consumed)
    # Every summary.yaml as the statistics consumed it.
    summaries = evidence["summary_yaml_at_statistics"]
    assert set(summaries) == {"mlp", "full", "fixed_decay", "fox",
                              *K_IDS.values()}
    stats = json.loads(analysis["stats_path"].read_text())
    for config_id, row in summaries.items():
        assert row["sha256"] == stats["runs"][config_id]["summary"]["sha256"]
    # The decisions.
    assert payload["decisions"]["k_selection"]["selected_config_id"]
    assert payload["decisions"]["family_map"]
    members = payload["decisions"]["family_map"][0]["members"]
    assert [m["member"] for m in members] == ["primary", "death_gate",
                                              "chase_gate"]


def test_verify_refuses_when_the_analysis_code_changes(analysis, monkeypatch,
                                                      capsys):
    _write(analysis)
    payload = json.loads(analysis["pin"].read_text())
    payload["analysis_code"]["scripts/sequence_track/"
                             "render_stage2_report.py"] = "0" * 64
    analysis["pin"].write_text(json.dumps(payload))
    assert pin.main(["--verify", "--pin", str(analysis["pin"])]) == 1
    assert "analysis_code" in capsys.readouterr().err


def test_verify_refuses_when_the_statistics_json_changes(analysis, capsys):
    _write(analysis)
    analysis["stats_path"].write_text(
        analysis["stats_path"].read_text() + "\n")
    assert pin.main(["--verify", "--pin", str(analysis["pin"])]) == 1
    assert "evidence.statistics" in capsys.readouterr().err


def test_verify_refuses_when_the_report_is_regenerated_differently(analysis,
                                                                  capsys):
    """MUST-FIX 6, directly: a report re-rendered from a different dependency
    directory is drift, not an equivalent regeneration."""
    _write(analysis)
    recert = analysis["dependency"] / "recert"
    analysis["report"].write_text(
        rr.render(analysis["stats_path"], analysis["config_path"],
                  analysis["k_path"], recert, analysis["report"]))
    assert pin.main(["--verify", "--pin", str(analysis["pin"])]) == 1
    assert "evidence.report" in capsys.readouterr().err


def test_verify_refuses_when_the_dependency_directory_changes(analysis,
                                                             capsys):
    """The recorded invocation names the directory, so the pin written from
    `recert` and the pin written from its parent are different manifests."""
    _write(analysis)
    recorded = json.loads(analysis["pin"].read_text())
    assert recorded["evidence"]["dependency_dir"] == st.rel(
        analysis["dependency"])

    assert _write(analysis,
                  **{"--dependency-dir": analysis["dependency"] / "recert"}
                  ) == 0
    rewritten = json.loads(analysis["pin"].read_text())
    assert rewritten["evidence"]["dependency_dir"] != recorded["evidence"][
        "dependency_dir"]
    assert len(rewritten["evidence"]["dependency_certificates"]) == 8
    assert rewritten["dependency_coverage"][
        "trained_checkpoint_coverage_complete"] is False
    assert set(rewritten["dependency_coverage"]["blocked"]) == {
        "recency_k30", "same_entity_k30"}

    # And restoring the original manifest, then verifying against the
    # certificate set the RECORDED directory yields, names the field.
    analysis["pin"].write_text(json.dumps(recorded))
    (analysis["dependency"] / "full.json").unlink()
    assert pin.main(["--verify", "--pin", str(analysis["pin"])]) == 1
    err = capsys.readouterr().err
    assert "evidence.dependency_certificates" in err


def test_verify_refuses_when_a_certificate_is_edited(analysis, capsys):
    _write(analysis)
    path = analysis["dependency"] / "recert" / "same_entity_k30_seed7.json"
    record = json.loads(path.read_text())
    record["wall_seconds"] = 99.0
    path.write_text(json.dumps(record))
    assert pin.main(["--verify", "--pin", str(analysis["pin"])]) == 1
    assert "evidence.dependency_certificates" in capsys.readouterr().err


def test_verify_refuses_when_the_config_contents_change(analysis, capsys):
    """Astra gate 2 round 2: changing only the config produced no drift.

    The config hash was recorded in `evidence.config` and never compared, so a
    report pinned against one registration verified against another.
    """
    _write(analysis)
    recorded = json.loads(analysis["pin"].read_text())
    assert recorded["evidence"]["config"]["sha256"] == st.sha256_file(
        analysis["config_path"])
    assert "evidence.config" in pin.COMPARED_FIELDS

    config = yaml.safe_load(analysis["config_path"].read_text())
    config["statistics"]["k_selection"]["tolerance_ll"] = 0.5
    analysis["config_path"].write_text(yaml.safe_dump(config, sort_keys=False))
    assert pin.main(["--verify", "--pin", str(analysis["pin"])]) == 1
    assert "evidence.config" in capsys.readouterr().err


def test_verify_refuses_when_a_checkpoint_is_no_longer_authenticated(analysis,
                                                                    capsys):
    """A certificate file can hash the same while its checkpoint changed.

    Astra gate 2 round 2: with authentication failing for all four masked arms
    in memory, `verify()` returned `True, []` while its own rebuilt coverage
    reported all four blocked, because `dependency_coverage` was not compared.
    """
    _write(analysis)
    before = json.loads(analysis["pin"].read_text())
    assert before["dependency_coverage"][
        "trained_checkpoint_coverage_complete"] is True
    assert before["dependency_coverage"]["blocked"] == []
    assert "dependency_coverage" in pin.COMPARED_FIELDS

    certificate = json.loads(
        (analysis["dependency"] / "recert"
         / "same_entity_k30_seed7.json").read_text())
    model = REPO / certificate["checkpoint"] / "model.pt"
    if not model.is_file():  # the fixture writes absolute checkpoint paths
        model = Path(certificate["checkpoint"]) / "model.pt"
    assert model.is_file()
    model.write_bytes(b"a different checkpoint")

    ok, drift, rebuilt = pin.verify(analysis["pin"])
    assert ok is False
    assert "dependency_coverage" in drift
    assert "same_entity_k30" in rebuilt["dependency_coverage"]["blocked"]
    assert pin.main(["--verify", "--pin", str(analysis["pin"])]) == 1
    assert "dependency_coverage" in capsys.readouterr().err


def test_verify_refuses_when_a_checkpoint_goes_missing(analysis, capsys):
    _write(analysis)
    certificate = json.loads(
        (analysis["dependency"] / "recert"
         / "recency_k30_seed13.json").read_text())
    model = Path(certificate["checkpoint"]) / "model.pt"
    if not model.is_file():
        model = REPO / certificate["checkpoint"] / "model.pt"
    model.unlink()
    ok, drift, rebuilt = pin.verify(analysis["pin"])
    assert ok is False
    assert "dependency_coverage" in drift
    assert "recency_k30" in rebuilt["dependency_coverage"]["blocked"]
    assert pin.main(["--verify", "--pin", str(analysis["pin"])]) == 1
    assert "dependency_coverage" in capsys.readouterr().err


def test_verify_refuses_when_the_k_selection_changes(analysis, capsys):
    _write(analysis)
    payload = json.loads(analysis["pin"].read_text())
    payload["decisions"]["k_selection"]["selected_config_id"] = "same_entity_k0"
    analysis["pin"].write_text(json.dumps(payload))
    assert pin.main(["--verify", "--pin", str(analysis["pin"])]) == 1
    assert "decisions.k_selection" in capsys.readouterr().err


def test_verify_refuses_when_the_family_map_changes(analysis, capsys):
    _write(analysis)
    payload = json.loads(analysis["pin"].read_text())
    payload["decisions"]["family_map"] = payload["decisions"][
        "family_map"][:-1]
    analysis["pin"].write_text(json.dumps(payload))
    assert pin.main(["--verify", "--pin", str(analysis["pin"])]) == 1
    assert "decisions.family_map" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# the live summary.yaml carve-out
# ---------------------------------------------------------------------------

def test_a_rewritten_summary_is_reported_but_does_not_fail_verification(
        analysis, capsys):
    """A later training night rewrites `runs/<config>/summary.yaml`.

    The evidence of record is the hash the statistics consumed — anchored by
    the statistics JSON's own verified hash — so a rewritten summary is
    reported in `live_summary_state` and does not fail the verification. What
    would fail is a changed statistics JSON.
    """
    _write(analysis)
    before = json.loads(analysis["pin"].read_text())
    assert all(row["matches_statistics"] for row
               in before["live_summary_state"].values())

    summary = analysis["runs"] / "mlp" / "summary.yaml"
    summary.write_text(summary.read_text() + "\n# a later night appended here\n")
    assert pin.main(["--verify", "--pin", str(analysis["pin"])]) == 0
    assert "OK" in capsys.readouterr().out

    _write(analysis)
    after = json.loads(analysis["pin"].read_text())
    assert after["live_summary_state"]["mlp"]["matches_statistics"] is False
    assert after["evidence"]["summary_yaml_at_statistics"]["mlp"]["sha256"] == (
        before["evidence"]["summary_yaml_at_statistics"]["mlp"]["sha256"])


# ---------------------------------------------------------------------------
# refusals and access
# ---------------------------------------------------------------------------

def test_write_refuses_without_a_statistics_json(analysis, tmp_path, capsys):
    code = pin.main(_argv(analysis, "--write",
                          **{"--stats-json": tmp_path / "absent.json"}))
    assert code == 2
    assert "REFUSED" in capsys.readouterr().err
    assert not analysis["pin"].exists()


def test_write_and_verify_are_mutually_exclusive(analysis):
    with pytest.raises(SystemExit):
        pin.main(["--write", "--verify"])


def test_the_pin_refuses_a_forbidden_path(analysis, capsys):
    for flag, value in (("--pin", REPO / "models" / "embeddings"
                         / "seq_stage2" / "cohort" / "pin.json"),
                        ("--dependency-dir", REPO / "models" / "embeddings"
                         / "seq_stage2" / "cohort")):
        assert pin.main(_argv(analysis, "--write", **{flag: value})) == 2
        assert "refusing to open" in capsys.readouterr().err


def test_the_compared_field_list_is_explicit(analysis):
    _write(analysis)
    payload = json.loads(analysis["pin"].read_text())
    assert payload["compared_fields"] == list(pin.COMPARED_FIELDS)
    for field in ("written_at_utc", "git_head_short", "live_summary_state"):
        assert field not in pin.COMPARED_FIELDS
    # Every compared field resolves to something in the written manifest.
    for field in pin.COMPARED_FIELDS:
        assert pin._get(payload, field) is not None, field


# ---------------------------------------------------------------------------
# Astra gate 2 round 3 — MUST-FIX 1 ("the pin currently verifies the same
# incomplete requirement") and MUST-FIX 2 (the prior-statistics evidence).
# ---------------------------------------------------------------------------

def test_the_pin_requires_the_registered_seeds_for_dependency_coverage(
        analysis, tmp_path):
    """The required trained-checkpoint seeds are the registration's seeds.

    With the same two-seed certificate tree, a two-seed analysis pin records
    complete coverage and a five-seed one records all four masked arms blocked.
    Before this fix both recorded complete coverage at seeds 7 and 13.
    """
    assert _write(analysis) == 0
    two_seed = json.loads(analysis["pin"].read_text())["dependency_coverage"]
    assert two_seed["required_checkpoint_seeds"] == [7, 13]
    assert two_seed["trained_checkpoint_coverage_complete"] is True
    assert two_seed["blocked"] == []

    assert _write(analysis, **{"--seeds": "7,13,29,42,101"}) == 0
    five_seed = json.loads(analysis["pin"].read_text())["dependency_coverage"]
    assert five_seed["required_checkpoint_seeds"] == [7, 13, 29, 42, 101]
    assert five_seed["trained_checkpoint_coverage_complete"] is False
    assert five_seed["blocked"] == ["recency_k30", "same_entity_k0",
                                    "same_entity_k30", "same_entity_unr"]


def test_the_pin_records_and_verifies_the_prior_statistics_argument(
        analysis, tmp_path, capsys):
    """MUST-FIX 2: which prior analysis § 12 compared against is part of the
    recorded invocation, and its hash is compared evidence."""
    prior = tmp_path / "prior_stats.json"
    prior.write_text(analysis["stats_path"].read_text())
    assert _write(analysis, **{"--prior-stats-json": prior}) == 0
    capsys.readouterr()
    payload = json.loads(analysis["pin"].read_text())
    renderer = payload["invocations"]["renderer"]
    assert "--prior-stats-json" in renderer
    assert renderer[renderer.index("--prior-stats-json") + 1] == pin.rel(prior)
    recorded = payload["evidence"]["prior_statistics"]
    assert recorded["compared"] is True and recorded["present"] is True
    assert pin.main(["--verify", "--pin", str(analysis["pin"])]) == 0
    capsys.readouterr()
    # Editing the prior statistics is drift, because its hash is compared.
    prior.write_text(prior.read_text() + "\n")
    assert pin.main(["--verify", "--pin", str(analysis["pin"])]) == 1
    assert "evidence.prior_statistics" in capsys.readouterr().err


def test_a_pin_without_a_prior_statistics_argument_still_resolves_the_field(
        analysis):
    assert _write(analysis) == 0
    recorded = json.loads(
        analysis["pin"].read_text())["evidence"]["prior_statistics"]
    assert recorded == {"path": None, "present": False, "sha256": None,
                        "compared": False}
