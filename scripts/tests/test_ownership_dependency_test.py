"""Tests for the D7 ownership dependency test (stage 2, D7.1–7.4).

Everything runs on the CPU against the tiny synthetic frame the contract
tests already build (4 validation innings of 12 balls) and against checkpoints
written HERE from freshly initialised models — no repository frame, no eval
kit, no stats cache, nothing under `models/`, and no training.

The point of the file is the two-sided claim D7 makes: the relay-free masked
arms must come back at exactly 0, and the standard-wiring arms must come back
above 1e-3 under the same perturbation, or the test is not measuring anything.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1]
for candidate in (SCRIPTS, SCRIPTS / "sequence_track"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import transformer_t1 as t1  # noqa: E402
from ownership_dependency_test import (CONTROL_FLOOR,  # noqa: E402
                                       PASS_TOLERANCE, DependencyTestError,
                                       exit_code, main, run, summary_line)
from test_transformer_t1_contract import write_frame  # noqa: E402

DMODEL, LAYERS, HEADS, SEED = 24, 2, 2, 11
MASKED = [("same_entity", 0), ("same_entity", 30), ("same_entity", "unr"),
          ("recency", 30)]
# The positive controls, with the masked S(i) each is scored against. The
# real frame's innings are ~120 rows so the registered k = 30 default bites;
# this synthetic innings is 12 rows, where W_30 is the whole prefix and
# nothing would be perturbed, so the window is narrowed explicitly.
CONTROLS = [("full", None, "recency", 3),
            ("aligned_hist", None, "same_entity", 3)]


def write_checkpoint(directory: Path, frame: Path, arm: str, k) -> Path:
    """A checkpoint dir: a freshly initialised model plus its metrics block."""
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    model = t1.T1Model(t1.N_FEATS, DMODEL, LAYERS, HEADS, arm=arm, k=k)
    # The default init is random, not zero, so the map is non-degenerate and a
    # positive control cannot read 0 merely because the weights are flat. No
    # training is needed: D7 certifies a wiring, not a fit.
    torch.save(model.state_dict(), directory / "model.pt")
    n_params = sum(p.numel() for p in model.parameters())
    (directory / "metrics.json").write_text(json.dumps({
        "config": {"dmodel": DMODEL, "layers": LAYERS, "heads": HEADS,
                   "data_dir": str(frame), "frame_version": "i7",
                   "arm": arm, "k": k},
        "n_params": n_params,
        "arm_params": t1.arm_params_block(arm, k, None, None, n_params),
    }, indent=2))
    return directory


@pytest.fixture(scope="module")
def frame(tmp_path_factory) -> Path:
    return write_frame(tmp_path_factory.mktemp("dep_frame") / "frame", "i7")


@pytest.mark.parametrize("arm,k", MASKED)
def test_masked_arms_pass_with_exactly_zero(tmp_path, frame, arm, k):
    ckpt = write_checkpoint(tmp_path / f"{arm}_{k}", frame, arm, k)
    result = run(ckpt, n_targets=48, seed=29, device="cpu")
    assert result["arm"] == arm and result["k"] == k
    assert result["max_abs_delta"] == 0.0, result
    assert result["p99_abs_delta"] == 0.0
    assert result["n_nonzero_gt_1e-6"] == 0
    assert result["pass"] is True
    assert result["positive_control_expected"] is False
    assert result["key_construction"] == "own_outcome"
    assert "PASS" in summary_line(result)


@pytest.mark.parametrize("arm,k,set_arm,set_k", CONTROLS)
def test_standard_arms_show_relay_above_the_control_floor(tmp_path, frame,
                                                         arm, k, set_arm,
                                                         set_k):
    ckpt = write_checkpoint(tmp_path / f"ctrl_{arm}", frame, arm, k)
    result = run(ckpt, n_targets=48, seed=29, device="cpu", set_arm=set_arm,
                 set_k=set_k)
    assert result["dependency_set_arm"] == set_arm
    assert result["dependency_set_k"] == set_k
    assert result["max_abs_delta"] > 1e-3, result
    assert result["pass"] is False
    assert result["positive_control_expected"] is True
    assert result["positive_control_observed"] is True
    assert result["key_construction"] == "shifted_history"
    assert "SEES RELAY" in summary_line(result)


def test_every_target_is_sampled_from_the_validation_split(tmp_path, frame):
    ckpt = write_checkpoint(tmp_path / "rows", frame, "same_entity", 30)
    result = run(ckpt, n_targets=10_000, seed=29, device="cpu")
    # 4 innings of 12 balls; the request is capped at the split's row count and
    # the test split (also 48 rows) is never touched.
    assert result["n_rows"] == 48 and result["n_innings"] == 4
    assert result["n_targets"] == 48
    assert result["split"] == "validation"


def test_the_cli_writes_the_json_and_the_summary(tmp_path, frame, capsys):
    ckpt = write_checkpoint(tmp_path / "cli", frame, "recency", 30)
    out = tmp_path / "dependency" / "recency_k30.json"
    assert main(["--checkpoint", str(ckpt), "--n-targets", "24", "--out",
                 str(out), "--device", "cpu"]) == 0
    payload = json.loads(out.read_text())
    assert set(payload) >= {"arm", "k", "checkpoint_md5", "n_targets",
                            "max_abs_delta", "p99_abs_delta",
                            "n_nonzero_gt_1e-6", "pass",
                            "positive_control_expected"}
    assert payload["checkpoint_md5"] == t1.md5_file(ckpt / "model.pt")
    assert payload["n_targets"] == 24 and payload["seed"] == 29
    assert "recency k=30" in capsys.readouterr().out


def test_the_exit_code_is_the_verdict(tmp_path, frame):
    """SHOULD 1: a completed invocation must not exit 0 on a failure.

    Every path is exercised on the result dict, so the rule is asserted
    without a leaking model to construct: a masked arm above the 1e-6
    tolerance and a positive control that never reaches the 1e-3 floor both
    have to exit 1, and the two passing cases exit 0.
    """
    masked_pass = {"positive_control_expected": False, "pass": True,
                   "positive_control_observed": False}
    masked_fail = {"positive_control_expected": False, "pass": False,
                   "positive_control_observed": True}
    control_ok = {"positive_control_expected": True, "pass": False,
                  "positive_control_observed": True}
    control_blind = {"positive_control_expected": True, "pass": False,
                     "positive_control_observed": False}
    assert exit_code(masked_pass) == 0
    assert exit_code(masked_fail) == 1
    assert exit_code(control_ok) == 0
    assert exit_code(control_blind) == 1
    assert (PASS_TOLERANCE, CONTROL_FLOOR) == (1e-6, 1e-3)

    # and the CLI returns it: a masked arm at exactly 0 exits 0 ...
    masked = write_checkpoint(tmp_path / "masked", frame, "same_entity", 0)
    out = tmp_path / "rc" / "same_entity_k0.json"
    assert main(["--checkpoint", str(masked), "--n-targets", "12", "--out",
                 str(out), "--device", "cpu"]) == 0
    assert json.loads(out.read_text())["pass"] is True

    # ... and a positive control that sees relay also exits 0, so the non-zero
    # path is the verdict and not an artefact of the arm's role.
    control = write_checkpoint(tmp_path / "control", frame, "full", None)
    out_control = tmp_path / "rc" / "full.json"
    assert main(["--checkpoint", str(control), "--n-targets", "12", "--out",
                 str(out_control), "--device", "cpu", "--set-arm", "recency",
                 "--set-k", "3"]) == 0
    payload = json.loads(out_control.read_text())
    assert payload["positive_control_expected"] is True
    assert payload["positive_control_observed"] is True


def test_the_seed_makes_the_run_reproducible(tmp_path, frame):
    ckpt = write_checkpoint(tmp_path / "seeded", frame, "aligned_hist", None)
    first = run(ckpt, n_targets=16, seed=29, device="cpu", set_arm="recency",
                set_k=3)
    again = run(ckpt, n_targets=16, seed=29, device="cpu", set_arm="recency",
                set_k=3)
    assert first["max_abs_delta"] == again["max_abs_delta"]
    assert first["p99_abs_delta"] == again["p99_abs_delta"]


def test_the_perturbation_always_changes_every_outcome_outside_s():
    from ownership_dependency_test import perturb_outside
    rng = np.random.default_rng(29)
    y = np.array([0, 1, 2, 3, 4, 5, 0, 1], dtype=np.int64)
    feats = np.zeros((len(y), t1.N_FEATS), dtype=np.float32)
    keep = {2, 5}
    new_feats, new_y, n_outside = perturb_outside(rng, feats, y, keep)
    assert n_outside == 6
    outside = sorted(set(range(len(y))) - keep)
    assert (new_y[outside] != y[outside]).all(), "a redraw kept a label"
    assert (new_y[sorted(keep)] == y[sorted(keep)]).all()
    assert (new_feats[sorted(keep)] == 0).all()
    assert np.abs(new_feats[outside]).max() > 0
    # Nothing perturbed when S(i) is the whole innings.
    _, same_y, none = perturb_outside(rng, feats, y, set(range(len(y))))
    assert none == 0 and (same_y == y).all()


def test_a_relay_free_arm_refuses_a_loosened_dependency_set(tmp_path, frame):
    """The certificate is the arm's OWN set; nothing may widen it."""
    ckpt = write_checkpoint(tmp_path / "loosen", frame, "same_entity", 0)
    with pytest.raises(DependencyTestError, match="refused for the relay-free"):
        run(ckpt, n_targets=4, seed=29, device="cpu", set_arm="recency",
            set_k=3)


def test_a_control_defaults_to_its_registered_masked_counterpart(tmp_path,
                                                                frame):
    for arm, expected in (("full", ("recency", 30)),
                          ("aligned_hist", ("same_entity", 30))):
        ckpt = write_checkpoint(tmp_path / f"default_{arm}", frame, arm, None)
        result = run(ckpt, n_targets=4, seed=29, device="cpu")
        assert (result["dependency_set_arm"],
                result["dependency_set_k"]) == expected


def test_a_checkpoint_without_arm_params_is_refused(tmp_path, frame):
    directory = tmp_path / "bare"
    directory.mkdir()
    torch.save({}, directory / "model.pt")
    (directory / "metrics.json").write_text(json.dumps({"config": {}}))
    with pytest.raises(DependencyTestError, match="arm_params"):
        run(directory, n_targets=4, seed=29, device="cpu")


def test_a_missing_checkpoint_is_named(tmp_path):
    with pytest.raises(DependencyTestError, match="model.pt not found"):
        run(tmp_path / "absent", n_targets=4, seed=29, device="cpu")


@pytest.mark.parametrize("arm,extra", [("residual_t1", {"residual_l2": 0.001}),
                                       ("lstm", {})])
def test_arms_d7_does_not_cover_are_refused(tmp_path, frame, arm, extra):
    # Written by hand: building these models is not what is under test.
    directory = tmp_path / f"uncovered_{arm}"
    directory.mkdir()
    torch.save({}, directory / "model.pt")
    (directory / "metrics.json").write_text(json.dumps({
        "config": {"dmodel": DMODEL, "layers": LAYERS, "heads": HEADS,
                   "data_dir": str(frame), "frame_version": "i7"},
        "arm_params": t1.arm_params_block(arm, None, extra.get("residual_l2"),
                                         None, 1),
    }))
    with pytest.raises(DependencyTestError):
        run(directory, n_targets=4, seed=29, device="cpu")
