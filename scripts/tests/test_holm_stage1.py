"""Tests for the stage-1 Holm adjustment (D12.4/D12.5).

Everything here is synthetic. The gate itself is exercised on that synthetic
evidence: `claim_gate.decide` accepts a `registry_path`, so a throwaway odds
registry plus a throwaway Cricsheet-style cluster source is enough to produce
a REAL gate JSON, which `holm_stage1.rebuild_contrast` then has to reproduce.
No repository artifact, and nothing under `data/golden/` or
`data/forward_holdout/`, is read.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from cricsheet_fixtures import match_json  # noqa: E402
from registered_experiment import (  # noqa: E402
    match_cluster_ci,
    seed_mean_match_cluster_ci,
)
from sequence_track import holm_stage1  # noqa: E402
from sequence_track.holm_stage1 import (  # noqa: E402
    Contrast,
    RefusalError,
    analyse,
    bootstrap_draws,
    classify,
    holm_adjust,
    holm_level,
    holm_order,
    holm_rejections,
    percentile_interval,
    rebuild_contrast,
    two_sided_p,
)
from sim_eval import claim_gate  # noqa: E402
from sim_eval.claim_gate import decide  # noqa: E402
from sim_eval.market_math import CostModel  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic evidence
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _repo_root_is_case_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(claim_gate, "REPO_ROOT", tmp_path)


def _write_json(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload))
    return path


N_MATCHES = 12


def _world(tmp_path: Path):
    """Build the odds file, cluster corpus and registry shared by all arms."""
    source = tmp_path / "clusters"
    source.mkdir(exist_ok=True)
    odds_rows = []
    for index in range(N_MATCHES):
        match_id = f"m{index:02d}"
        prices = {"A": 2.0, "B": 2.0, "timestamp": "x"}
        odds_rows.append({"match_id": match_id, "team1": "A", "team2": "B",
                          "odds": {"winner": prices}})
        corpus = match_json(date=f"2026-01-{index + 1:02d}", teams=["A", "B"],
                            venue=f"V{index}", event=f"E{index}",
                            innings_data=[])
        corpus["info"]["event"] = {"name": f"E{index}"}
        _write_json(source / f"{match_id}.json", corpus)
    odds = _write_json(tmp_path / "odds.json", {"matches": odds_rows})
    registry = _write_json(tmp_path / "registry.json", {
        "registered_odds": [{
            "role": "test",
            "path": str(odds),
            "sha256": hashlib.sha256(odds.read_bytes()).hexdigest(),
            "cluster_source_dir": str(source),
        }],
    })
    return source, registry


def _arm(tmp_path: Path, name: str, probs) -> Path:
    """Write one sliced eval JSON whose P(A) per match is ``probs``."""
    matches = []
    for index in range(N_MATCHES):
        prob = float(probs[index])
        prices = {"A": 2.0, "B": 2.0, "timestamp": "x"}
        matches.append({
            "match_id": f"m{index:02d}",
            "teams": ["A", "B"],
            # Alternating winners so the paired deltas carry both signs.
            "actual_winner": "A" if index % 2 == 0 else "B",
            "market_odds": prices,
            "market_prob": {"A": 0.5, "B": 0.5},
            "realized_pnl": 0.0,
            "bet_placed": False,
            "simulated_prob": {"A": prob, "B": 1.0 - prob},
        })
    return _write_json(tmp_path / f"{name}.json",
                       {"summary": {"bootstrap_reliable": True},
                        "matches": matches})


def _gate(tmp_path: Path, source: Path, registry: Path, candidate: Path,
          baseline: Path, out_name: str) -> Path:
    verdict = decide([candidate], [baseline], "match_model", CostModel.none(),
                     odds_role="test", cluster_source_dir=source,
                     registry_path=registry)
    return _write_json(tmp_path / out_name, verdict.as_dict())


def _probs(offset: float, spread: float = 0.02):
    return [0.5 + offset + spread * math.sin(index) for index in range(N_MATCHES)]


def _three_contrasts(tmp_path: Path):
    """A, B, C arms plus the C-B / B-A / C-A gates over them."""
    source, registry = _world(tmp_path)
    arms = {
        "A": _arm(tmp_path, "arm_a", _probs(0.00)),
        "B": _arm(tmp_path, "arm_b", _probs(0.05)),
        "C": _arm(tmp_path, "arm_c", _probs(0.09)),
    }
    gates = {
        "C-B": _gate(tmp_path, source, registry, arms["C"], arms["B"], "gate_cb.json"),
        "B-A": _gate(tmp_path, source, registry, arms["B"], arms["A"], "gate_ba.json"),
        "C-A": _gate(tmp_path, source, registry, arms["C"], arms["A"], "gate_ca.json"),
    }
    return source, registry, arms, gates


# ---------------------------------------------------------------------------
# Estimator identity
# ---------------------------------------------------------------------------


def test_bootstrap_draws_reproduce_both_gate_estimators():
    rng = np.random.default_rng(0)
    clusters = np.asarray([f"blk{i // 4}" for i in range(24)], dtype=str)
    values = rng.normal(size=(4, 24))

    single = bootstrap_draws(values[:1], clusters, 500, 42)
    assert [float(np.percentile(single, 2.5)),
            float(np.percentile(single, 97.5))] == match_cluster_ci(
                values[0], clusters, 500, 42)

    multi = bootstrap_draws(values, clusters, 500, 42)
    assert [float(np.percentile(multi, 2.5)),
            float(np.percentile(multi, 97.5))] == seed_mean_match_cluster_ci(
                values, clusters, 500, 42)


def test_rebuild_reproduces_the_gate_interval_exactly(tmp_path):
    source, registry, _arms, gates = _three_contrasts(tmp_path)
    for name, gate_path in gates.items():
        contrast = rebuild_contrast(name, gate_path, registry)
        stamped = json.loads(gate_path.read_text())
        assert contrast.recomputed_ci95 == pytest.approx(
            tuple(stamped["delta_log_loss"]["ci95"]), abs=1e-12)
        assert contrast.point == pytest.approx(
            stamped["delta_log_loss"]["point"], abs=1e-12)
        assert contrast.block_count == stamped["block_count"] == N_MATCHES
        assert contrast.n_records == stamped["n_records"] == N_MATCHES
        assert contrast.estimator == "match_cluster_ci"
        assert contrast.draws.size == 10_000
        assert len(contrast.evidence) == 2


def test_delta_sign_is_candidate_minus_baseline(tmp_path):
    """A candidate that is uniformly better must produce a negative point."""
    source, registry = _world(tmp_path)
    # Every match is won by A on even indices and B on odd ones; the candidate
    # is pulled toward the truth in both directions, the baseline is flat.
    truth = [0.9 if index % 2 == 0 else 0.1 for index in range(N_MATCHES)]
    good = _arm(tmp_path, "good", truth)
    flat = _arm(tmp_path, "flat", [0.5] * N_MATCHES)
    better = rebuild_contrast("good-flat",
                              _gate(tmp_path, source, registry, good, flat,
                                    "gate_good.json"), registry)
    worse = rebuild_contrast("flat-good",
                             _gate(tmp_path, source, registry, flat, good,
                                   "gate_worse.json"), registry)
    assert better.point < 0.0 < worse.point
    assert better.point == pytest.approx(-worse.point)


def test_dropped_rows_and_block_count_match_the_gate(tmp_path):
    source, registry = _world(tmp_path)
    candidate = json.loads(_arm(tmp_path, "cand", _probs(0.05)).read_text())
    baseline = json.loads(_arm(tmp_path, "base", _probs(0.00)).read_text())
    # One pair loses its outcome on both sides: the gate drops it symmetrically.
    candidate["matches"][0]["actual_winner"] = None
    baseline["matches"][0]["actual_winner"] = None
    # One pair is not in the registered odds index at all.
    candidate["matches"][1]["match_id"] = "unregistered"
    baseline["matches"][1]["match_id"] = "unregistered"
    cand_path = _write_json(tmp_path / "cand2.json", candidate)
    base_path = _write_json(tmp_path / "base2.json", baseline)
    gate_path = _gate(tmp_path, source, registry, cand_path, base_path,
                      "gate_drop.json")
    contrast = rebuild_contrast("drop", gate_path, registry)
    stamped = json.loads(gate_path.read_text())
    assert contrast.n_dropped_symmetrically == stamped["n_dropped_symmetrically"] == 2
    assert contrast.n_records == stamped["n_records"] == N_MATCHES - 2
    assert contrast.block_count == stamped["block_count"] == N_MATCHES - 2


def test_forged_cluster_stamp_is_ignored_like_the_gate(tmp_path):
    """The gate strips stamped cluster ids; so must the reconstruction."""
    source, registry = _world(tmp_path)
    candidate = json.loads(_arm(tmp_path, "cand", _probs(0.05)).read_text())
    baseline = json.loads(_arm(tmp_path, "base", _probs(0.00)).read_text())
    for payload in (candidate, baseline):
        for row in payload["matches"]:
            row["competition_cluster_id"] = "forged-single-block"
    cand_path = _write_json(tmp_path / "cand3.json", candidate)
    base_path = _write_json(tmp_path / "base3.json", baseline)
    gate_path = _gate(tmp_path, source, registry, cand_path, base_path,
                      "gate_forged.json")
    contrast = rebuild_contrast("forged", gate_path, registry)
    assert contrast.block_count == N_MATCHES


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_mutated_sliced_eval_json_is_refused(tmp_path):
    source, registry, arms, gates = _three_contrasts(tmp_path)
    payload = json.loads(arms["C"].read_text())
    payload["matches"][0]["simulated_prob"]["A"] += 1e-6
    payload["matches"][0]["simulated_prob"]["B"] -= 1e-6
    _write_json(arms["C"], payload)
    with pytest.raises(RefusalError, match="sha256 mismatch"):
        rebuild_contrast("C-B", gates["C-B"], registry)


def test_missing_sliced_eval_json_is_refused(tmp_path):
    source, registry, arms, gates = _three_contrasts(tmp_path)
    arms["B"].unlink()
    with pytest.raises(RefusalError, match="does not exist"):
        rebuild_contrast("B-A", gates["B-A"], registry)


def test_tampered_gate_ci95_is_refused(tmp_path):
    source, registry, _arms, gates = _three_contrasts(tmp_path)
    payload = json.loads(gates["C-A"].read_text())
    payload["delta_log_loss"]["ci95"][0] -= 1e-6
    _write_json(gates["C-A"], payload)
    with pytest.raises(RefusalError, match="recomputed ci95"):
        rebuild_contrast("C-A", gates["C-A"], registry)


def test_ci95_within_tolerance_is_accepted(tmp_path):
    """The assertion is an identity check to 1e-9, not an equality of floats."""
    source, registry, _arms, gates = _three_contrasts(tmp_path)
    payload = json.loads(gates["C-A"].read_text())
    payload["delta_log_loss"]["ci95"][1] += 1e-12
    _write_json(gates["C-A"], payload)
    assert rebuild_contrast("C-A", gates["C-A"], registry).name == "C-A"


def test_tampered_gate_point_is_refused(tmp_path):
    source, registry, _arms, gates = _three_contrasts(tmp_path)
    payload = json.loads(gates["B-A"].read_text())
    payload["delta_log_loss"]["point"] += 1e-6
    _write_json(gates["B-A"], payload)
    with pytest.raises(RefusalError, match="recomputed point"):
        rebuild_contrast("B-A", gates["B-A"], registry)


def test_tampered_block_count_is_refused(tmp_path):
    source, registry, _arms, gates = _three_contrasts(tmp_path)
    payload = json.loads(gates["C-B"].read_text())
    payload["block_count"] = N_MATCHES - 1
    _write_json(gates["C-B"], payload)
    with pytest.raises(RefusalError, match="recomputed block_count"):
        rebuild_contrast("C-B", gates["C-B"], registry)


def test_non_match_model_gate_is_refused(tmp_path):
    source, registry, _arms, gates = _three_contrasts(tmp_path)
    payload = json.loads(gates["C-B"].read_text())
    payload["kind"] = "betting_layer"
    _write_json(gates["C-B"], payload)
    with pytest.raises(RefusalError, match="not kind=match_model"):
        rebuild_contrast("C-B", gates["C-B"], registry)


def test_off_contract_bootstrap_is_refused(tmp_path):
    source, registry, _arms, gates = _three_contrasts(tmp_path)
    payload = json.loads(gates["C-B"].read_text())
    payload["bootstrap"]["seed"] = 7
    _write_json(gates["C-B"], payload)
    with pytest.raises(RefusalError, match="resamples at 95%"):
        rebuild_contrast("C-B", gates["C-B"], registry)


def test_sealed_pools_are_refused(tmp_path):
    with pytest.raises(RefusalError, match="refusing sealed path"):
        rebuild_contrast("x", tmp_path / "data" / "golden" / "gate.json",
                         tmp_path / "registry.json")
    with pytest.raises(RefusalError, match="refusing sealed path"):
        rebuild_contrast("x", tmp_path / "forward_holdout" / "gate.json",
                         tmp_path / "registry.json")


# ---------------------------------------------------------------------------
# Bootstrap p-value convention
# ---------------------------------------------------------------------------


def test_two_sided_p_uses_the_registered_convention():
    draws = np.concatenate([-np.ones(30), np.ones(70)])
    p, on_floor = two_sided_p(draws)
    assert p == pytest.approx(0.6)
    assert on_floor is False

    p, on_floor = two_sided_p(np.ones(1000))
    assert p == 0.0 and on_floor is True

    p, on_floor = two_sided_p(-np.ones(1000))
    assert p == 0.0 and on_floor is True


def test_two_sided_p_counts_zero_draws_on_both_sides_and_caps_at_one():
    p, on_floor = two_sided_p(np.asarray([0.0, 0.0, 1.0, 1.0]))
    # P(<=0) = 0.5, P(>=0) = 1.0, so 2 * 0.5 = 1.0 exactly, not above it.
    assert p == 1.0 and on_floor is False
    assert two_sided_p(np.asarray([-1.0, -1.0, 1.0, 1.0]))[0] == 1.0


# ---------------------------------------------------------------------------
# Holm arithmetic
# ---------------------------------------------------------------------------


def test_holm_m3_multipliers_and_monotonicity():
    # sorted: 0.01 (x3), 0.03 (x2 -> 0.06), 0.04 (x1 -> 0.04, raised to 0.06)
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_holm_ties_share_the_largest_step():
    assert holm_adjust([0.02, 0.02, 0.02]) == pytest.approx([0.06, 0.06, 0.06])
    assert holm_order([0.02, 0.02, 0.02]) == [0, 1, 2]


def test_holm_is_capped_at_one_and_never_decreases_in_rank_order():
    adjusted = holm_adjust([0.5, 0.6, 0.7])
    assert adjusted == [1.0, 1.0, 1.0]
    raw = [0.001, 0.049, 0.02]
    adjusted = holm_adjust(raw)
    ordered = [adjusted[index] for index in holm_order(raw)]
    assert ordered == sorted(ordered)
    assert all(a >= r for a, r in zip(adjusted, raw))


def test_holm_single_contrast_is_unchanged():
    assert holm_adjust([0.031]) == pytest.approx([0.031])


def test_holm_rejects_out_of_range_p_values():
    with pytest.raises(ValueError):
        holm_adjust([0.1, 1.5])
    with pytest.raises(ValueError):
        holm_adjust([float("nan")])


def test_holm_levels_step_down():
    assert holm_level(1, 3) == pytest.approx(1 - 0.05 / 3)
    assert holm_level(2, 3) == pytest.approx(1 - 0.05 / 2)
    # The last-ranked contrast keeps the gate's own 95% level.
    assert holm_level(3, 3) == pytest.approx(0.95)
    with pytest.raises(ValueError):
        holm_level(4, 3)


def test_rank_local_interval_widens_with_the_level():
    draws = np.random.default_rng(3).normal(size=20_000)
    narrow = percentile_interval(draws, holm_level(3, 3))
    wide = percentile_interval(draws, holm_level(1, 3))
    assert wide[0] < narrow[0] and narrow[1] < wide[1]


# ---------------------------------------------------------------------------
# Holm step-down rejection (Astra MUST-FIX 1)
# ---------------------------------------------------------------------------


def test_step_down_rejects_only_a_prefix_of_the_ranks():
    # 3 * 0.001 = 0.003 rejects; 2 * 0.030 = 0.060 does not, which STOPS the
    # procedure, so rank 3 cannot reject even though 1 * 0.040 = 0.040 <= 0.05.
    assert holm_rejections([0.001, 0.030, 0.040]) == [True, False, False]
    # All three clear their own multiplier.
    assert holm_rejections([0.001, 0.010, 0.040]) == [True, True, True]
    assert holm_rejections([0.001]) == [True]
    assert holm_rejections([0.060]) == [False]


def test_step_down_rejection_is_the_holm_adjusted_p_at_alpha():
    """The two formulations must agree; the running max is what ties them."""
    rng = np.random.default_rng(17)
    for _ in range(200):
        raw = list(rng.uniform(0.0, 0.1, size=4))
        assert holm_rejections(raw, 0.05) == [p <= 0.05 for p in holm_adjust(raw)]


def test_step_down_rejects_nothing_at_the_reviewer_counterexample():
    """p = 0.020, 0.021, 0.022 -> every adjusted p is 0.060, so none reject."""
    raw = [0.020, 0.021, 0.022]
    assert holm_adjust(raw) == pytest.approx([0.060, 0.060, 0.060])
    assert holm_rejections(raw) == [False, False, False]


def test_holm_rejections_rejects_bad_inputs():
    with pytest.raises(ValueError):
        holm_rejections([0.1, 1.5])
    with pytest.raises(ValueError):
        holm_rejections([float("nan")])
    with pytest.raises(ValueError):
        holm_rejections([0.1], alpha=0.0)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("point, gate_ci95, rejected, expected", [
    # Parity: the whole GATE interval sits inside the band, boundaries
    # included. Parity does not depend on the rejection.
    (0.0, (-0.007, 0.007), False, "parity"),
    (0.0, (-0.007, 0.007), True, "parity"),
    (0.003, (0.001, 0.005), False, "parity"),
    (-0.003, (-0.005, -0.001), False, "parity"),
    (0.0, (0.0, 0.0), False, "parity"),
    # Just outside the band on one side, nothing rejected.
    (-0.004, (-0.0071, 0.006), False, "inconclusive"),
    (0.004, (-0.006, 0.0071), False, "inconclusive"),
    # Favourable / adverse need BOTH the step-down rejection and a point
    # beyond the band.
    (-0.014, (-0.020, -0.008), True, "favourable"),
    (0.014, (0.008, 0.020), True, "adverse"),
    (-0.007, (-0.020, -0.008), True, "inconclusive"),
    (0.007, (0.008, 0.020), True, "inconclusive"),
    (-0.0071, (-0.020, -0.008), True, "favourable"),
    (0.0071, (0.008, 0.020), True, "adverse"),
    # THE FIX: an interval that excludes zero decides nothing on its own.
    (-0.014, (-0.020, -0.008), False, "inconclusive"),
    (0.014, (0.008, 0.020), False, "inconclusive"),
    # A rejection with an interval that touches zero is still a rejection:
    # the rejection, not the interval, is the rule.
    (0.010, (0.0, 0.020), True, "adverse"),
    (-0.010, (-0.020, 0.0), True, "favourable"),
    (0.010, (0.0, 0.020), False, "inconclusive"),
    # Wide interval spanning zero, not rejected.
    (-0.030, (-0.200, 0.150), False, "inconclusive"),
])
def test_classification_rules_and_boundaries(point, gate_ci95, rejected,
                                             expected):
    assert classify(point, gate_ci95, rejected=rejected) == expected


def test_classification_rejects_inverted_intervals():
    with pytest.raises(ValueError):
        classify(0.0, (0.1, -0.1), rejected=False)


# ---------------------------------------------------------------------------
# Family assembly
# ---------------------------------------------------------------------------


def _fake(name: str, draws: np.ndarray) -> Contrast:
    return Contrast(
        name=name, gate_path=f"{name}.json", verdict="FAILED",
        point=float(draws.mean()),
        gate_ci95=percentile_interval(draws, 0.95),
        recomputed_ci95=percentile_interval(draws, 0.95),
        block_count=12, n_records=12, n_dropped_symmetrically=0,
        estimator="match_cluster_ci", seeds=["single"], evidence={},
        odds_role="test", cluster_source_dir="clusters", draws=draws,
    )


def test_exploratory_contrast_is_not_adjusted_and_does_not_enter_m():
    rng = np.random.default_rng(11)
    contrasts = [
        _fake("C-B", rng.normal(0.0, 0.01, 10_000)),
        _fake("B-A", rng.normal(-0.05, 0.01, 10_000)),
        _fake("C-A", rng.normal(-0.05, 0.01, 10_000)),
        _fake("A50-A", rng.normal(-0.05, 0.01, 10_000)),
    ]
    rows = {row["contrast"]: row
            for row in analyse(contrasts, ["C-B", "B-A", "C-A"])}

    for name in ("C-B", "B-A", "C-A"):
        assert rows[name]["family"] == "confirmatory"
        assert rows[name]["holm_family_size"] == 3
        assert rows[name]["p_holm"] is not None
        assert rows[name]["holm_rank"] in {1, 2, 3}

    exploratory = rows["A50-A"]
    assert exploratory["family"] == "exploratory"
    assert exploratory["p_holm"] is None
    assert exploratory["holm_rank"] is None
    assert exploratory["holm_family_size"] is None
    assert exploratory["holm_rejected"] is None
    assert exploratory["rank_local_level"] == 0.95
    assert exploratory["rank_local_interval"] == list(
        exploratory["recomputed_ci95"])
    # Outside the family, so its rejection is the unadjusted raw p.
    assert exploratory["rejected_at_alpha"] is (exploratory["p_raw"] <= 0.05)
    assert "unadjusted" in exploratory["rejection_rule"]
    # Its raw p is still reported.
    assert 0.0 <= exploratory["p_raw"] <= 1.0


def test_family_ranks_levels_and_labels_are_consistent():
    rng = np.random.default_rng(5)
    contrasts = [
        _fake("C-B", rng.normal(0.000, 0.002, 10_000)),
        _fake("B-A", rng.normal(-0.050, 0.005, 10_000)),
        _fake("C-A", rng.normal(-0.030, 0.010, 10_000)),
    ]
    rows = {row["contrast"]: row for row in analyse(contrasts, ["C-B", "B-A", "C-A"])}
    assert sorted(row["holm_rank"] for row in rows.values()) == [1, 2, 3]
    for row in rows.values():
        assert row["rank_local_level"] == pytest.approx(
            holm_level(row["holm_rank"], 3))
        assert row["p_holm"] >= row["p_raw"]
        assert row["holm_rejected"] is (row["p_holm"] <= 0.05)
        assert row["label"] == classify(row["point"],
                                        tuple(row["gate_ci95"]),
                                        rejected=row["holm_rejected"])
    assert rows["C-B"]["label"] == "parity"
    assert rows["B-A"]["label"] == "favourable"
    assert rows["C-A"]["label"] == "favourable"


def test_last_ranked_rank_local_interval_is_the_gate_interval():
    rng = np.random.default_rng(7)
    contrasts = [
        _fake("C-B", rng.normal(-0.05, 0.005, 10_000)),
        _fake("B-A", rng.normal(-0.02, 0.005, 10_000)),
        _fake("C-A", rng.normal(0.000, 0.005, 10_000)),
    ]
    rows = {row["contrast"]: row for row in analyse(contrasts, ["C-B", "B-A", "C-A"])}
    last = next(row for row in rows.values() if row["holm_rank"] == 3)
    assert last["rank_local_interval"] == pytest.approx(last["recomputed_ci95"])
    # The deprecated alias still carries the same numbers for old readers.
    assert last["adjusted_interval"] == last["rank_local_interval"]
    assert last["adjusted_level"] == last["rank_local_level"]


# ---------------------------------------------------------------------------
# The MUST-FIX 1 regression: a rank-local interval is not a rejection
# ---------------------------------------------------------------------------


def _draws_with_p(p: float, *, sign: float = 1.0, size: int = 10_000
                  ) -> np.ndarray:
    """Draws whose two-sided bootstrap p is exactly ``p``.

    ``sign`` +1 puts the mass (and the mean, +0.0296) above the parity band,
    -1 mirrors it below. The minority tail holds ``p * size / 2`` draws, so
    `two_sided_p` returns ``p`` exactly.
    """
    tail = int(round(p * size / 2.0))
    return np.concatenate([np.full(tail, -sign * 0.01),
                           np.full(size - tail, sign * 0.03)])


def test_rank_local_intervals_that_exclude_zero_do_not_label_without_rejection():
    """The reviewer's counterexample, end to end.

    Raw p 0.020 / 0.021 / 0.022 all adjust to 0.060, so the step-down rejects
    nothing. The rank-2 and rank-3 rank-local intervals nevertheless exclude
    zero and every point sits well beyond +0.007 — exactly the state in which
    the old rank-local classifier returned `adverse`. All three must now be
    `inconclusive`.
    """
    contrasts = [_fake(name, _draws_with_p(p)) for name, p in
                 (("C-B", 0.020), ("B-A", 0.021), ("C-A", 0.022))]
    rows = {row["contrast"]: row
            for row in analyse(contrasts, ["C-B", "B-A", "C-A"])}

    assert [rows[n]["p_raw"] for n in ("C-B", "B-A", "C-A")] == pytest.approx(
        [0.020, 0.021, 0.022])
    assert [rows[n]["p_holm"] for n in ("C-B", "B-A", "C-A")] == pytest.approx(
        [0.060, 0.060, 0.060])
    assert [rows[n]["holm_rank"] for n in ("C-B", "B-A", "C-A")] == [1, 2, 3]

    for name in ("C-B", "B-A", "C-A"):
        row = rows[name]
        assert row["holm_rejected"] is False
        assert row["point"] > 0.007
        assert row["label"] == "inconclusive"

    # The premise of the finding: ranks 2 and 3 DO exclude zero rank-locally,
    # so the old rule would have labelled them adverse.
    for name in ("B-A", "C-A"):
        low, high = rows[name]["rank_local_interval"]
        assert low > 0.0 and high > 0.0
        assert classify(rows[name]["point"], (low, high), rejected=True) == \
            "adverse"
    # And the leading contrast's rank-local interval straddles zero, which is
    # why reading labels off these intervals is not even internally ordered.
    assert rows["C-B"]["rank_local_interval"][0] < 0.0


def test_rank_three_cannot_reject_once_rank_two_fails():
    """Rank 1 rejects, rank 2 does not, so rank 3 is dead however small its p."""
    contrasts = [
        _fake("C-B", _draws_with_p(0.001, sign=-1.0)),   # rank 1, favourable
        _fake("B-A", _draws_with_p(0.030)),              # rank 2, 2*p = 0.060
        _fake("C-A", _draws_with_p(0.040)),              # rank 3, 1*p = 0.040
    ]
    rows = {row["contrast"]: row
            for row in analyse(contrasts, ["C-B", "B-A", "C-A"])}

    assert [rows[n]["holm_rank"] for n in ("C-B", "B-A", "C-A")] == [1, 2, 3]
    assert rows["C-B"]["holm_rejected"] is True
    assert rows["B-A"]["holm_rejected"] is False
    assert rows["C-A"]["holm_rejected"] is False

    assert rows["C-B"]["label"] == "favourable"
    assert rows["B-A"]["label"] == "inconclusive"
    assert rows["C-A"]["label"] == "inconclusive"

    # Rank 3's own multiplier (1 x 0.040) clears alpha and its rank-local
    # interval excludes zero, so only the step-down stopping rule keeps it
    # from being labelled adverse.
    assert rows["C-A"]["p_raw"] * 1 <= 0.05
    assert rows["C-A"]["rank_local_interval"][0] > 0.0
    assert rows["C-A"]["p_holm"] > 0.05


def test_parity_is_decided_on_the_gate_interval_not_the_rank_local_one():
    """A rank-1 contrast whose gate ci95 is inside the band is parity.

    Its rank-local interval is wider and can spill outside ±0.007; that must
    not turn a parity result into an inconclusive one.
    """
    # 150 / 9,700 / 150: the 2.5 and 97.5 percentiles land in the tight
    # middle block, the 0.833 and 99.167 ones land in the outer blocks.
    tight = np.concatenate([np.full(150, -0.0090), np.full(9_700, 0.0010),
                            np.full(150, 0.0090)])
    contrasts = [_fake("C-B", tight),
                 _fake("B-A", _draws_with_p(0.30)),
                 _fake("C-A", _draws_with_p(0.40))]
    rows = {row["contrast"]: row
            for row in analyse(contrasts, ["C-B", "B-A", "C-A"])}
    row = rows["C-B"]
    assert row["holm_rank"] == 1
    gate_low, gate_high = row["gate_ci95"]
    assert -0.007 <= gate_low and gate_high <= 0.007
    local_low, local_high = row["rank_local_interval"]
    assert local_low < -0.007 and local_high > 0.007
    assert row["label"] == "parity"
    assert "gate" in row["parity_basis"]


def test_analyse_refuses_unknown_or_empty_confirmatory_names():
    contrast = _fake("C-B", np.random.default_rng(1).normal(size=1_000))
    with pytest.raises(RefusalError, match="not supplied"):
        analyse([contrast], ["B-A"])
    with pytest.raises(RefusalError, match="at least one confirmatory"):
        analyse([contrast], [])


def test_analyse_refuses_duplicate_contrast_names():
    draws = np.random.default_rng(2).normal(size=1_000)
    with pytest.raises(RefusalError, match="unique"):
        analyse([_fake("C-B", draws), _fake("C-B", draws)], ["C-B"])


def test_descriptive_flag_follows_the_ten_block_rule():
    contrast = _fake("C-B", np.random.default_rng(4).normal(size=1_000))
    contrast.block_count = 9
    assert analyse([contrast], ["C-B"])[0]["descriptive_block_count"] is True
    contrast.block_count = 10
    assert analyse([contrast], ["C-B"])[0]["descriptive_block_count"] is False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_writes_json_and_markdown(tmp_path):
    source, registry, arms, gates = _three_contrasts(tmp_path)
    exploratory_arm = _arm(tmp_path, "arm_a50", _probs(0.02))
    gates["A50-A"] = _gate(tmp_path, source, registry, exploratory_arm,
                           arms["A"], "gate_a50.json")
    out_json = tmp_path / "out" / "holm.json"
    out_md = tmp_path / "out" / "holm.md"
    code = holm_stage1.main([
        "--gate", f"C-B={gates['C-B']}",
        "--gate", f"B-A={gates['B-A']}",
        "--gate", f"C-A={gates['C-A']}",
        "--gate", f"A50-A={gates['A50-A']}",
        "--confirmatory", "C-B", "B-A", "C-A",
        "--slice", "50000",
        "--registry-path", str(registry),
        "--out-json", str(out_json),
        "--out-md", str(out_md),
    ])
    assert code == 0

    payload = json.loads(out_json.read_text())
    assert payload["contract"]["alpha"] == 0.05
    assert payload["contract"]["parity_band"] == 0.007
    assert payload["contract"]["registered_expectation"] == \
        "parity everywhere, no arm advancing"
    assert payload["contract"]["delta_sign"].startswith("candidate minus baseline")
    assert payload["contract"]["rank_local_interval"] == \
        holm_stage1.RANK_LOCAL_NOTE
    assert "not a simultaneous Holm confidence interval" in \
        payload["contract"]["rank_local_interval"]
    assert "step-down" in payload["contract"]["rejection_rule"]
    names = [row["contrast"] for row in payload["contrasts"]]
    assert names == ["C-B", "B-A", "C-A", "A50-A"]
    families = {row["contrast"]: row["family"] for row in payload["contrasts"]}
    assert families["A50-A"] == "exploratory"
    for row in payload["contrasts"]:
        assert row["gate_ci95"] == pytest.approx(row["recomputed_ci95"], abs=1e-12)
        assert row["label"] in {"parity", "favourable", "adverse", "inconclusive"}
        assert row["block_count"] == N_MATCHES and row["n_records"] == N_MATCHES
        assert row["rank_local_interval"] == row["adjusted_interval"]
        assert row["rank_local_level"] == row["adjusted_level"]
        assert isinstance(row["rejected_at_alpha"], bool)
        if row["family"] == "confirmatory":
            assert row["rank_local_interval_note"] == \
                holm_stage1.RANK_LOCAL_NOTE
            assert row["holm_rejected"] is (row["p_holm"] <= 0.05)
        else:
            assert row["holm_rejected"] is None
        # No label may be favourable/adverse without a rejection.
        if row["label"] in {"favourable", "adverse"}:
            assert row["rejected_at_alpha"] is True

    text = out_md.read_text()
    assert "parity everywhere, no arm advancing" in text
    assert "tournament_time_block_v1" in text
    assert "candidate minus baseline" in text
    assert "not a simultaneous Holm confidence interval" in text
    assert "rank-local percentile interval at level 1-alpha/(m-k+1)" in text
    assert "| rank-local level | rank-local interval |" in text
    for name in names:
        assert f"| {name} |" in text
    assert "sha256 verified" in text


def test_cli_verify_gate_replays_and_refuses_a_tampered_gate(tmp_path):
    source, registry, _arms, gates = _three_contrasts(tmp_path)
    argv = [
        "--gate", f"C-B={gates['C-B']}", "--gate", f"B-A={gates['B-A']}",
        "--gate", f"C-A={gates['C-A']}",
        "--confirmatory", "C-B", "B-A", "C-A",
        "--registry-path", str(registry), "--verify-gate",
        "--out-json", str(tmp_path / "v.json"),
        "--out-md", str(tmp_path / "v.md"),
    ]
    assert holm_stage1.main(argv) == 0

    # `verdict` is replayed by the gate but never touched by the ci95 identity
    # check, so only --verify-gate can catch this.
    payload = json.loads(gates["C-A"].read_text())
    payload["verdict"] = "LANDED"
    _write_json(gates["C-A"], payload)
    with pytest.raises(RefusalError, match="does not replay"):
        holm_stage1.main(argv)


def test_multi_seed_gate_uses_the_seed_mean_estimator(tmp_path):
    source, registry = _world(tmp_path)
    candidates, baselines = [], []
    for seed in (7, 13, 29):
        for name, offset, store in (("cand", 0.05, candidates),
                                    ("base", 0.00, baselines)):
            payload = json.loads(
                _arm(tmp_path, f"{name}_tmp", _probs(offset)).read_text())
            payload["summary"]["seed"] = seed
            for index, row in enumerate(payload["matches"]):
                jitter = seed * (1e-5 if name == "cand" else 2e-5) * (index + 1)
                row["simulated_prob"]["A"] += jitter
                row["simulated_prob"]["B"] -= jitter
            store.append(_write_json(tmp_path / f"{name}_seed{seed}.json", payload))
    verdict = decide(candidates, baselines, "match_model", CostModel.none(),
                     odds_role="test", cluster_source_dir=source,
                     registry_path=registry)
    gate_path = _write_json(tmp_path / "gate_multi.json", verdict.as_dict())
    contrast = rebuild_contrast("multi", gate_path, registry)
    assert contrast.estimator == "seed_mean_match_cluster_ci"
    assert contrast.seeds == ["13", "29", "7"]
    assert contrast.recomputed_ci95 == pytest.approx(
        tuple(verdict.delta_log_loss.ci95), abs=1e-12)
    assert contrast.point == pytest.approx(verdict.delta_log_loss.point, abs=1e-12)


def test_cli_requires_name_equals_path(tmp_path):
    with pytest.raises(SystemExit):
        holm_stage1.main(["--gate", "justapath.json",
                          "--confirmatory", "C-B",
                          "--out-json", str(tmp_path / "a.json"),
                          "--out-md", str(tmp_path / "a.md")])
