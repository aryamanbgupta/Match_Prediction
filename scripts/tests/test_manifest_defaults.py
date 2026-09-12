"""Guard production/evaluation paths against escaping the artifact manifest."""
from __future__ import annotations

import ast
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
PREFIXES = (
    "models/",
    "data/live_state",
    "data/xgb_",
    "data/golden/polymarket_test",
    "data/polymarket_test",
    "eval_out/",
)
MARKER_RE = re.compile(r"manifest-exempt:\s*(\S.*)$")
SHELL_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:models/|data/live_state|data/xgb_|"
    r"data/golden/polymarket_test|data/polymarket_test|eval_out/)"
)


AUTO_EXEMPT_FILES = (
    "scripts/auto/a11_boundary_sweep.py",
    "scripts/auto/a12_fit_dew_calibrator.py",
    "scripts/auto/a12_gate_analysis.py",
    "scripts/auto/a13_dispersion_eval.py",
    "scripts/auto/a13_gate_analysis.py",
    "scripts/auto/a14_fit_over_calibrator.py",
    "scripts/auto/a15_fit_over0_calibrator.py",
    "scripts/auto/a16_gate_analysis.py",
    "scripts/auto/a7_conditional_threshold.py",
    "scripts/auto/a8_gate_analysis.py",
    "scripts/auto/a9_run.py",
    "scripts/auto/b10_build_usage_sidecar.py",
    "scripts/auto/b10_gate_analysis.py",
    "scripts/auto/b10_unit_check.py",
    "scripts/auto/b12_gate_analysis.py",
    "scripts/auto/b13_build_damping_sidecar.py",
    "scripts/auto/b13_gate_analysis.py",
    "scripts/auto/b13_unit_check.py",
    "scripts/auto/b14_fit_quote_calibrator.py",
    "scripts/auto/b14_gate_analysis.py",
    "scripts/auto/b15_gate_analysis.py",
    "scripts/auto/b16_gate_analysis.py",
    "scripts/auto/b17_decompose_quotes.py",
    "scripts/auto/b17_runmass_audit.py",
    "scripts/auto/b18_fit_extras_graft.py",
    "scripts/auto/b18_gate_analysis.py",
    "scripts/auto/b18_unit_check.py",
    "scripts/auto/b1_build_venue_encoder.py",
    "scripts/auto/b1_gate_analysis.py",
    "scripts/auto/b3_shrinkage_blend.py",
    "scripts/auto/b4_pricing_margin.py",
    "scripts/auto/b5_gate_analysis.py",
    "scripts/auto/b5_inplay_quotes.py",
    "scripts/auto/b5_unit_check.py",
    "scripts/auto/b6_gate_analysis.py",
    "scripts/auto/b7_fit_calibrators.py",
    "scripts/auto/b7_gate_analysis.py",
    "scripts/auto/b8_compose_hybrid.py",
    "scripts/auto/b8_gate_analysis.py",
    "scripts/auto/b9_usage_baseline.py",
    "scripts/auto/d11_symmetrize.py",
    "scripts/auto/d12_build_parquet.py",
    "scripts/auto/d12_run.py",
    "scripts/auto/d13_run.py",
    "scripts/auto/d14_gate_analysis.py",
    "scripts/auto/d14_unit_check.py",
    "scripts/auto/d15_build_runout_rates.py",
    "scripts/auto/d15_gate_analysis.py",
    "scripts/auto/d15_unit_check.py",
    "scripts/auto/d16_fit_vector_calibrator.py",
    "scripts/auto/d16_gate_analysis.py",
    "scripts/auto/d16_marginal_audit.py",
    "scripts/auto/d17_gate_analysis.py",
    "scripts/auto/d18_gate_analysis.py",
    "scripts/auto/d18_train_arms.py",
    "scripts/auto/d1_gate_analysis.py",
    "scripts/auto/d2_gate_analysis.py",
    "scripts/auto/d2_unit_check.py",
    "scripts/auto/d3_build_extras_rates.py",
    "scripts/auto/d3_gate_analysis.py",
    "scripts/auto/d3_unit_check.py",
    "scripts/auto/d7_run.py",
    "scripts/auto/d8_run.py",
    "scripts/auto/d9_run.py",
    "scripts/auto/i14b_build_frame.py",
    "scripts/auto/i14b_gate_analysis.py",
    "scripts/auto/i14b_train.py",
    "scripts/auto/i18_frame_parity.py",
    "scripts/auto/i18_stamp_envelope.py",
    "scripts/auto/i19_contract_check.py",
    "scripts/auto/i19_repro_check.py",
    "scripts/auto/p2_recal_paired.py",
)
AUTO_REASON = "closed-idea script, archived by item 8"

EXPECTED_EXEMPTIONS = {
    **{path: AUTO_REASON for path in AUTO_EXEMPT_FILES},
    "scripts/analyze_features.py": "diagnostic one-off named in item 5",
    "scripts/profile_eval.py": "diagnostic one-off named in item 5",
    "scripts/e1_temperature_sharpen.py": "diagnostic one-off named in item 5",
    "scripts/audit_t1_sim_parity.py": "diagnostic feature-contract audit with a frozen replay frame",
    "scripts/build_i7_match_frame.py": "versioned artifact builder owns its output namespace",
    "scripts/build_i9_match_frame.py": "versioned experimental artifact builder owns its output namespace",
    "scripts/build_stats_cache.py": "generic artifact builder owns configurable cache outputs",
    "scripts/convert_weights.py": "legacy model conversion utility",
    "scripts/deepcrease_join.py": "embeddings-ladder experimental artifact namespace",
    "scripts/discriminability_test.py": "embeddings-ladder diagnostic artifact namespace",
    "scripts/e3_seed_ensemble.py": "diagnostic one-off for an archived experiment",
    "scripts/embeddings_e1.py": "embeddings-ladder experimental artifact namespace",
    "scripts/embeddings_e1_analysis.py": "embeddings-ladder diagnostic artifact namespace",
    "scripts/embeddings_e3_probe.py": "embeddings-ladder experimental artifact namespace",
    "scripts/embeddings_eval_kit.py": "embeddings-ladder experimental artifact namespace",
    "scripts/eval_womens_v1.py": "isolated non-production women's research track",
    "scripts/fit_i5_ball_calibrator.py": "isolated unpromoted I5 experiment",
    "scripts/lstm_v1.py": "legacy model-family trainer owns its artifact namespace",
    "scripts/materialize_features.py": "generic artifact builder owns configurable frame outputs",
    "scripts/materialize_match_features.py": "generic artifact builder owns configurable frame outputs",
    "scripts/mlp_v1.py": "legacy model-family trainer owns its artifact namespace",
    "scripts/mlp_v2.py": "legacy model-family trainer owns its artifact namespace",
    "scripts/rebuild_b18_extras_sidecar.py": "diagnostic rebuild for an unpromoted sidecar",
    "scripts/refresh_golden_i7.sh": "reproduction comments name gitignored I18 reference paths",
    "scripts/run_br2_gates.sh": "generated gate output root",
    "scripts/run_experiment.py": "generic experiment orchestrator constructs versioned outputs",
    "scripts/run_i9_direct_seeds.py": "isolated I9 experiment",
    "scripts/run_sliced_eval.sh": "generated evaluation output CLI example",
    "scripts/sequence_track/render_batch2_report.py": "embeddings-ladder experimental artifact namespace",
    "scripts/sequence_track/render_night3_report.py": "embeddings-ladder experimental artifact namespace",
    "scripts/sequence_track/stage3a_freeze_tiers.py": "embeddings-ladder experimental artifact namespace",
    "scripts/sequence_track/stage3c_leakage_audit.py": "embeddings-ladder experimental artifact namespace",
    "scripts/sequence_track/stage4_exposure_sidecar.py": "embeddings-ladder experimental artifact namespace",
    "scripts/sequence_track/stage4_guard.py": "embeddings-ladder experimental artifact namespace",
    "scripts/sequence_track/stage4_pair_graph_audit.py": "embeddings-ladder experimental artifact namespace",
    "scripts/sequence_track/stage4_references.py": "embeddings-ladder experimental artifact namespace",
    "scripts/sim_eval/run_sim_eval.py": "legacy alternate model-family paths remain explicit replay options",
    "scripts/sim_t1.py": "embeddings-ladder experimental artifact namespace",
    "scripts/t1_exposure_bias_check.py": "embeddings-ladder diagnostic artifact namespace",
    "scripts/transformer_t1.py": "embeddings-ladder experimental artifact namespace",
    "scripts/transformer_v1.py": "legacy model-family trainer owns its artifact namespace",
    "scripts/transformer_xr.py": "embeddings-ladder experimental artifact namespace",
    "scripts/validate_numpy_predict.py": "legacy numerical parity diagnostic",
    "scripts/validate_training_cache_match.py": "legacy training-cache diagnostic",
    "scripts/xgboost_match_v1.py": "generic match-model trainer owns configurable outputs",
    "scripts/xgboost_v2.py": "generic ball-model trainer owns configurable outputs",
}


def _excluded(path: Path) -> bool:
    relative = path.relative_to(REPO)
    return "tests" in relative.parts


def _markers(lines: list[str]) -> list[tuple[int, str]]:
    found = []
    for number, line in enumerate(lines, 1):
        match = MARKER_RE.search(line)
        if match:
            found.append((number, match.group(1).strip()))
    return found


def test_manifest_defaults_and_explicit_exemption_inventory():
    violations: list[str] = []
    inventory: dict[str, str] = {}
    paths = sorted(SCRIPTS.rglob("*.py")) + sorted(SCRIPTS.rglob("*.sh"))
    for path in paths:
        if _excluded(path):
            continue
        relative = path.relative_to(REPO).as_posix()
        source = path.read_text()
        lines = source.splitlines()
        markers = _markers(lines)
        for _, reason in markers:
            previous = inventory.setdefault(relative, reason)
            assert previous == reason, f"{relative}: inconsistent exemption reasons"
        header_reasons = {reason for number, reason in markers if number <= 5}

        if path.suffix == ".py":
            tree = ast.parse(source, filename=str(path))
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and node.value.startswith(PREFIXES)
                ):
                    continue
                line = lines[node.lineno - 1]
                if not header_reasons and not MARKER_RE.search(line):
                    violations.append(
                        f"{relative}:{node.lineno}: {node.value!r}"
                    )
        else:
            for number, line in enumerate(lines, 1):
                if SHELL_PATH_RE.search(line):
                    if not header_reasons and not MARKER_RE.search(line):
                        violations.append(f"{relative}:{number}: {line.strip()}")

    assert inventory == EXPECTED_EXEMPTIONS, (
        "manifest exemption inventory changed; review and update the explicit "
        f"list:\nactual={inventory!r}"
    )
    assert not violations, "hard-coded artifact defaults:\n" + "\n".join(violations)
