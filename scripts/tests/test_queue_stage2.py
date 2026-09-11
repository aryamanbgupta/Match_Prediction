"""Tests for the stage-2 night-1 queues (D8 checks 8.2, 8.3).

The night is SPLIT BY SEED across two machines (Astra MUST-FIX 9, the user's
decision 2026-09-12): `research/sequence_track/queue_laptop.yaml` runs seed 7
on the laptop and `research/sequence_track/queue_mini.yaml` runs seed 13 on
the Mac mini. Each file holds sixteen jobs, one per registered configuration,
in the registered order.

The queues are what actually run, so they are checked against the arm register
rather than eyeballed: the job count and order, the machine tag, the config
each job names (so a later change to the seed list changes every signature and
the driver's reuse check re-runs only what is missing), the thread caps, the
SEED-SPECIFIC output dir (the runner decides skip-vs-run from the COMPLETE
marker in `output_dir` plus the config sha256 alone, so a shared
per-configuration marker dir would let one seed's COMPLETE suppress the
other's), the per-machine memory defaults and stop files, and the
`expected_hours` budget formula.

Artifact-free: the queue files and the config are text reads, and
`queue_lib.load_queue` is the runner's own parser.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from sequence_track import queue_lib

REPO_ROOT = Path(__file__).resolve().parents[2]
QUEUE_DIR = REPO_ROOT / "research" / "sequence_track"
LAPTOP_QUEUE = QUEUE_DIR / "queue_laptop.yaml"
MINI_QUEUE = QUEUE_DIR / "queue_mini.yaml"
CONFIG_PATH = REPO_ROOT / "experiments" / "configs" / "seq_stage2_v1.yaml"
STAGE2_CONFIG = "experiments/configs/seq_stage2_v1.yaml"
REGISTERED_ORDER = [
    "mlp", "full", "fixed_decay", "fox", "aligned_hist", "recency_k30",
    "same_entity_k30", "aligned_hist_rf", "same_entity_k0", "same_entity_k6",
    "same_entity_k12", "same_entity_unr", "lstm", "xlstm", "residual_mlp",
    "residual_t1",
]

# D6.3 laptop smoke seconds (load + one epoch), verbatim from the acceptance
# file, and the per-machine slowdown factor Astra registered for the budget.
SMOKE_SECONDS = {
    "mlp": 6.5, "lstm": 6.9, "residual_mlp": 7.4, "full": 8.2,
    "aligned_hist": 8.7, "fixed_decay": 10.0, "same_entity_k0": 11.0,
    "same_entity_k30": 11.0, "fox": 11.1, "same_entity_unr": 11.1,
    "recency_k30": 11.2, "residual_t1": 11.4, "aligned_hist_rf": 13.9,
    "same_entity_k6": 14.2, "same_entity_k12": 14.4,
    # xlstm re-smoked after the cell normalisation fix and the restored
    # output gate (2026-09-12): 73.2 s, was 62.7 s.
    "xlstm": 73.2,
}
MACHINES = {
    # machine: (queue path, seed, slowdown factor, floor GB, cap GB, stop file)
    "laptop": (LAPTOP_QUEUE, 7, 1.5, 12, 24,
               "research/sequence_track/STOP_laptop"),
    "mini": (MINI_QUEUE, 13, 3, 3, 12, "research/sequence_track/STOP_mini"),
}
LAPTOP_THREAD_CAPS = ("OMP_NUM_THREADS=2", "OPENBLAS_NUM_THREADS=2",
                      "MKL_NUM_THREADS=2", "VECLIB_MAXIMUM_THREADS=2")


def expected_hours(seconds: float, factor: float) -> float:
    """Astra's budget rule: ceil_to_0.05((30 * smoke * F + 120) / 3600)."""
    raw = (30 * seconds * factor + 120) / 3600.0
    return math.ceil(raw / 0.05) * 0.05


@pytest.fixture(scope="module")
def queues():
    loaded = {}
    for machine, (path, *_rest) in MACHINES.items():
        defaults, jobs, errors, warnings = queue_lib.load_queue(path)
        assert errors == [], (machine, errors)
        assert warnings == [], (machine, warnings)
        loaded[machine] = (defaults, jobs)
    return loaded


@pytest.mark.parametrize("machine", sorted(MACHINES))
def test_sixteen_jobs_in_the_registered_order(queues, machine):
    _defaults, jobs = queues[machine]
    seed = MACHINES[machine][1]
    assert [job["id"] for job in jobs] == [
        f"{config_id}-s{seed}" for config_id in REGISTERED_ORDER]


def test_job_ids_are_the_config_ids():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    by_order = sorted(config["configurations"],
                      key=lambda row: int(row["queue_order"]))
    assert [str(entry["id"]) for entry in by_order] == REGISTERED_ORDER


@pytest.mark.parametrize("machine", sorted(MACHINES))
def test_every_job_is_tagged_for_its_own_machine(queues, machine):
    _defaults, jobs = queues[machine]
    assert {job["machine"] for job in jobs} == {machine}


@pytest.mark.parametrize("machine", sorted(MACHINES))
def test_the_other_machine_selects_nothing(queues, machine):
    """Each half is inert on the other machine, so both can be synced."""
    _defaults, jobs = queues[machine]
    other = "mini" if machine == "laptop" else "laptop"
    assert [job for job in jobs if job["machine"] == other] == []


@pytest.mark.parametrize("machine", sorted(MACHINES))
def test_no_placeholder_survives(queues, machine):
    _defaults, jobs = queues[machine]
    path = MACHINES[machine][0]
    assert "seq_stage1" not in path.read_text()
    for job in jobs:
        assert "placeholder" not in job["id"]
        assert "echo" not in job["command"]
        assert queue_lib.resolve_path(job["config"]).is_file(), job["id"]


@pytest.mark.parametrize("machine", sorted(MACHINES))
def test_every_job_runs_the_stage_2_driver_on_one_config_and_one_seed(
        queues, machine):
    _defaults, jobs = queues[machine]
    seed = MACHINES[machine][1]
    for job, config_id in zip(jobs, REGISTERED_ORDER):
        assert job["config"] == STAGE2_CONFIG, job["id"]
        # D8.8: each command also carries an explicit --machine label so the
        # recorded provenance is registered rather than hostname-derived.
        assert job["command"].endswith(
            "scripts/sequence_track/retrain_stage2.py "
            f"--config-ids {config_id} --seeds {seed} "
            f"--machine {machine}"), job["id"]
        assert "uv run --no-sync python" in job["command"]
        assert "--epochs" not in job["command"]
        assert "--out-root" not in job["command"]
        assert job["expected_hours"] > 0


@pytest.mark.parametrize("machine", sorted(MACHINES))
def test_output_dir_is_seed_specific(queues, machine):
    """8.3: the runner keys skip-vs-run on output_dir + config sha ALONE.

    It reads neither the command nor the seed subset, so a per-configuration
    marker directory shared by the two halves would let seed 7's COMPLETE
    suppress seed 13 once the trees are rsynced into one output root.
    """
    _defaults, jobs = queues[machine]
    seed = MACHINES[machine][1]
    for job, config_id in zip(jobs, REGISTERED_ORDER):
        assert job["output_dir"] == (
            f"models/embeddings/seq_stage2/runs/{config_id}/seed_{seed}")
    # and no two jobs across the two halves share an output dir
    other = "mini" if machine == "laptop" else "laptop"
    _other_defaults, other_jobs = queues[other]
    assert not ({job["output_dir"] for job in jobs}
                & {job["output_dir"] for job in other_jobs})


def test_thread_caps_are_in_each_command(queues):
    """Per-job/extra `defaults` fields are ignored by the helper, so the
    thread caps must be environment prefixes on the command itself."""
    assert "OMP_NUM_THREADS" not in queue_lib.KNOWN_DEFAULT_FIELDS
    assert "OMP_NUM_THREADS" not in queue_lib.KNOWN_JOB_FIELDS
    assert "memory_cap_gb" not in queue_lib.KNOWN_JOB_FIELDS

    _mini_defaults, mini_jobs = queues["mini"]
    for job in mini_jobs:
        assert job["command"].startswith("OMP_NUM_THREADS=4 "), job["id"]

    _laptop_defaults, laptop_jobs = queues["laptop"]
    for job in laptop_jobs:
        for cap in LAPTOP_THREAD_CAPS:
            assert cap in job["command"], (job["id"], cap)


@pytest.mark.parametrize("machine", sorted(MACHINES))
def test_machine_sized_memory_defaults_and_separate_stop_files(
        queues, machine):
    defaults, _jobs = queues[machine]
    _path, _seed, _factor, floor, cap, stop = MACHINES[machine]
    assert defaults["memory_floor_gb"] == floor
    assert defaults["memory_cap_gb"] == cap
    assert defaults["poll_seconds"] == 30
    assert defaults["stop_file"] == stop


def test_the_two_stop_files_are_different(queues):
    laptop_defaults, _laptop_jobs = queues["laptop"]
    mini_defaults, _mini_jobs = queues["mini"]
    assert laptop_defaults["stop_file"] != mini_defaults["stop_file"]


@pytest.mark.parametrize("machine", sorted(MACHINES))
def test_expected_hours_follows_the_registered_budget_formula(
        queues, machine):
    _defaults, jobs = queues[machine]
    factor = MACHINES[machine][2]
    for job, config_id in zip(jobs, REGISTERED_ORDER):
        assert job["expected_hours"] == pytest.approx(
            expected_hours(SMOKE_SECONDS[config_id], factor)), job["id"]
        # no blanket one-hour ceiling: xlstm's budget is its own
        assert job["expected_hours"] >= 0.15


@pytest.mark.parametrize("machine", sorted(MACHINES))
def test_the_xlstm_recalibration_note_is_recorded(machine):
    """The xLSTM budget was recalibrated from the corrected 73.2 s smoke."""
    text = MACHINES[machine][0].read_text()
    assert "xLSTM RECALIBRATED" in text
    assert "73.2" in text
    assert "62.7" in text  # the superseded figure is named, not erased


@pytest.mark.parametrize("machine", sorted(MACHINES))
def test_the_retry_budget_is_recorded(machine):
    """One retry means a hung job can consume 4 x expected_hours."""
    text = MACHINES[machine][0].read_text()
    assert "4 * expected_hours" in text


@pytest.mark.parametrize("machine", sorted(MACHINES))
def test_the_signature_is_the_stage_2_config_so_a_seed_change_re_runs(
        tmp_path, queues, machine):
    """`config` names the config whose seed list defines the night (8.3)."""
    _defaults, jobs = queues[machine]
    live, warning = queue_lib.config_signature(
        queue_lib.resolve_path(jobs[0]["config"]), jobs[0]["config"])
    assert warning is None
    assert len(live) == 64
    edited = tmp_path / "seq_stage2_v1.yaml"
    config = yaml.safe_load(CONFIG_PATH.read_text())
    config["training"]["seeds"] = [7, 13, 29]
    edited.write_text(yaml.safe_dump(config, sort_keys=False))
    changed, _ = queue_lib.config_signature(edited, str(edited))
    assert changed != live
    assert {job["config"] for job in jobs} == {STAGE2_CONFIG}


def test_the_old_single_machine_queue_is_gone():
    assert not (QUEUE_DIR / "queue.yaml").exists()
