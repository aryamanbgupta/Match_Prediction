import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
IMPORTERS = [
    "a16_gate_analysis", "b1_gate_analysis", "b6_gate_analysis",
    "b7_gate_analysis", "b8_gate_analysis", "b10_gate_analysis",
    "b12_gate_analysis", "b13_gate_analysis", "d1_gate_analysis",
    "d2_gate_analysis", "d3_gate_analysis", "d14_gate_analysis",
    "d15_gate_analysis", "d16_gate_analysis", "d17_gate_analysis",
    "d18_gate_analysis",
]


def test_a8_reexports_shared_helper_objects():
    sys.path[:0] = [str(REPO / "scripts" / "auto"),
                    str(REPO / "scripts" / "sim_eval")]
    import a8_gate_analysis
    import eval_statistics
    for name in ("load", "paired_rows", "metric_rows", "cluster_boot",
                 "brier_pair", "mae_pair"):
        assert getattr(a8_gate_analysis, name) is getattr(eval_statistics, name)


@pytest.mark.parametrize("module", IMPORTERS)
def test_a8_importer_imports_without_running_main(module):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([
        str(REPO / "scripts" / "auto"),
        str(REPO / "scripts" / "sim_eval"),
        str(REPO / "scripts"),
        env.get("PYTHONPATH", ""),
    ])
    completed = subprocess.run(
        [sys.executable, "-c", f"import importlib; importlib.import_module('{module}')"],
        cwd=REPO, env=env, text=True, capture_output=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
