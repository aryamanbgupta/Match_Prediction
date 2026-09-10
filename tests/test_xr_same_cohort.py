import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

from run_xr_same_cohort import (  # noqa: E402
    ARMS,
    arm_mask,
    calibration_line,
    cluster_bootstrap,
    match_ids,
    seed_cluster_bootstrap,
)


def _cohort():
    return {
        "x": np.zeros((2, 11), dtype=np.float32),
        "n_base_features": 3,
        "offsets": {
            "line": (0, 2),
            "length": (2, 4),
            "shot": (4, 7),
            "control": (7, 8),
        },
    }


def test_fixed_width_arm_masks_reveal_only_registered_stages():
    expected_counts = {"context": 3, "delivery": 7,
                       "shot": 10, "contact": 11}
    for arm in ARMS:
        mask = arm_mask(_cohort(), arm)
        assert mask.shape == (11,)
        assert int(mask.sum()) == expected_counts[arm]
    assert np.all(arm_mask(_cohort(), "context")[:3] == 1)
    assert np.all(arm_mask(_cohort(), "context")[3:] == 0)


def test_match_id_is_removed_from_innings_prefix_only():
    values = np.asarray(["1_12345", "2_12345", "1_alpha_beta"])
    assert match_ids(values).tolist() == ["12345", "12345", "alpha_beta"]


def test_bootstraps_are_deterministic_and_paired():
    clusters = np.asarray(["a", "a", "b", "b"])
    values = np.asarray([1.0, 1.0, 3.0, 3.0])
    assert cluster_bootstrap(values, clusters, 200, 17) == [1.0, 3.0]
    stacked = np.stack([values, values])
    assert seed_cluster_bootstrap(stacked, clusters, 200, 17) == [1.0, 3.0]


def test_calibration_line_recovers_identity():
    values = np.asarray([0.0, 1.0, 2.0, 3.0])
    result = calibration_line(values, values)
    assert abs(result["intercept"]) < 1e-12
    assert abs(result["slope"] - 1.0) < 1e-12
