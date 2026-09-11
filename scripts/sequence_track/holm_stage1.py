#!/usr/bin/env python3
"""Holm multiplicity adjustment for the stage-1 confirmatory contrasts (D12.4).

`scripts/sim_eval/claim_gate.py --kind match_model` decides one contrast at a
time and stamps `delta_log_loss: {point, ci95}`.  It exposes no p-value and no
bootstrap draws, so the registered Holm step-down (D12.4) and the
Holm-adjusted classification (D12.5) cannot be read off the gate JSONs.

This script closes that gap WITHOUT re-deciding anything.  For each gate JSON
it re-derives the paired per-record log-loss deltas and their competition
blocks from the very same evidence files the gate hashed, re-runs the very
same block bootstrap with the same seed, and then REFUSES unless the
recomputed 95% percentile interval reproduces the gate's `ci95` to within
1e-9.  That assertion is the proof of estimator identity: only once it passes
are the retained resample draws used for p-values and adjusted intervals.

What the gate does, and what is mirrored here
---------------------------------------------
Deltas (`claim_gate.decide`, the `metrics_json is None` branch):

    seed_ll.append(_log_loss(cand) - _log_loss(base))

so the delta is CANDIDATE MINUS BASELINE log loss.  **Negative is favourable
to the candidate** (`_classification` reads `ll.ci95[1] < 0.0` as the
favourable direction and `-ll.point >= SEED_FLOOR` as the 0.007 floor).

A record pair is DROPPED (symmetrically, counted in
`n_dropped_symmetrically`) when either arm lacks `actual_winner` or
`market_odds`, or when the record is not in the registered odds index.  It
RAISES on an outcome mismatch between arms or on any `market_odds` that
disagrees with the registered price.

Blocks: `eval_statistics.load_competition_clusters(<registered
cluster_source_dir>)` builds the event/120-day-gap lookup, and each kept
record is assigned by `eval_statistics.cluster_id_with_resolution(clean,
lookup)` where `clean` is the candidate record with any stamped
`competition_cluster_id` / `cluster_id` stripped, so a forged stamp cannot
choose its own block.

Bootstrap: `claim_gate._interval` dispatches on the seed count to
`registered_experiment.match_cluster_ci(values[0], clusters, 10000, 42)` for a
single seed and `registered_experiment.seed_mean_match_cluster_ci(values,
clusters, 10000, 42)` for several, i.e. `DEFAULT_BOOTSTRAP_RESAMPLES` = 10,000
and `DEFAULT_BOOTSTRAP_SEED` = 42, both record-weighted percentile intervals
at 2.5 / 97.5 (`np.percentile`).  `bootstrap_draws` below reproduces those two
loops exactly and keeps the 10,000 replicate estimates.

Statistics registered for stage 1
---------------------------------
* Two-sided bootstrap p-value, percentile convention consistent with the way
  the gate forms `ci95`:  ``p = 2 * min(P(d* <= 0), P(d* >= 0))``, capped at
  1, with P the empirical proportion over the retained draws.  Its resolution
  floor is 1/10,000 per tail; a contrast with no draw on one side reports
  `p_raw` 0.0 and `p_at_resolution_floor` true, which reads as "< 2e-4".
* Holm step-down over the confirmatory family only (`--confirmatory`): with
  raw p-values sorted ascending, the k-th of m gets
  ``min(1, max over j <= k of (m - j + 1) * p_(j))`` — the running maximum is
  the monotonicity enforcement.  Extra contrasts (A50-A) are reported
  unadjusted and labelled exploratory, and do not enter m.
* Holm STEP-DOWN REJECTION: walking the ranks ascending, rank k rejects only
  if ``(m - k + 1) * p_(k) <= alpha`` AND every earlier rank rejected.  Once
  a rank fails, every later-ranked contrast fails with it.  This is the
  family-wise decision, and it is the only thing that can make a
  confirmatory contrast favourable or adverse.
* Rank-local percentile interval: the k-th ranked confirmatory contrast also
  gets the percentile interval at level ``1 - 0.05 / (m - k + 1)`` from the
  same draws (the last-ranked one is therefore the gate's own 95%
  interval).  These are REPORTED ONLY.  A rank-local percentile interval is
  **not** a simultaneous Holm confidence interval and is not a rejection
  rule: raw p-values 0.020, 0.021, 0.022 all adjust to 0.060, so nothing in
  the family rejects, yet the rank-1 interval (level 1 - 0.05/3) can still
  exclude zero.  Reading a label off that interval would contradict the
  step-down, so the classifier below does not look at it.
* Classification (D12.5):
  - **parity** if the GATE's own 95% interval (`delta_log_loss.ci95`, which
    this script re-derived and asserted equal to 1e-9) lies inside
    [-0.007, +0.007].  Parity is a statement about the registered 95%
    estimate, so it is decided on that interval, not on a rank-local one;
  - **favourable** if the contrast REJECTED under the Holm step-down
    (adjusted p <= alpha, with the step-down stopping rule) and the point is
    below -0.007;
  - **adverse** if it rejected and the point is above +0.007;
  - **inconclusive** otherwise.
  An exploratory contrast is outside the family, so its "rejection" is the
  unadjusted ``p_raw <= alpha`` and its reported interval is the gate's 95%.

Usage
-----
    uv run --no-sync python scripts/sequence_track/holm_stage1.py \
        --gate C-B=<gate.json> --gate B-A=<gate.json> --gate C-A=<gate.json> \
        --gate A50-A=<gate.json> \
        --confirmatory C-B B-A C-A \
        --slice 50000 \
        --out-json <out.json> --out-md <out.md>

Nothing is written outside `--out-json` / `--out-md`; no gate JSON is
modified, and the sealed golden/forward pools are refused outright.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Imported, never modified: the reconstruction must use the gate's own
# record-keeping, price check, log loss and path resolution so it cannot
# drift from the artifact it is proving.
from sim_eval import claim_gate  # noqa: E402
from sim_eval.eval_statistics import (  # noqa: E402
    BOOTSTRAP_CONTRACT_VERSION,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    cluster_id_with_resolution,
    load_competition_clusters,
)

FAMILY_ALPHA = 0.05
# The registered parity half-width is the gate's own seed-noise floor.
PARITY_BAND = claim_gate.SEED_FLOOR
CI_TOLERANCE = 1e-9
REGISTERED_EXPECTATION = "parity everywhere, no arm advancing"
MIN_RECOMMENDED_BLOCKS = 10

LABEL_PARITY = "parity"
LABEL_FAVOURABLE = "favourable"
LABEL_ADVERSE = "adverse"
LABEL_INCONCLUSIVE = "inconclusive"

# The exact words that must travel with a rank-local interval wherever it is
# reported, so nobody reads it as a simultaneous Holm confidence interval.
RANK_LOCAL_NOTE = (
    "rank-local percentile interval at level 1-alpha/(m-k+1); not a "
    "simultaneous Holm confidence interval"
)
EXPLORATORY_INTERVAL_NOTE = (
    "unadjusted 95% percentile interval; this contrast is outside the Holm "
    "family and carries no multiplicity control"
)
PARITY_BASIS = (
    "the gate's own 95% interval (delta_log_loss.ci95), re-derived and "
    "asserted identical to 1e-9"
)


class RefusalError(RuntimeError):
    """A registered precondition failed; nothing is reported."""


# ---------------------------------------------------------------------------
# Estimator replay
# ---------------------------------------------------------------------------


def bootstrap_draws(values: np.ndarray, clusters: np.ndarray,
                    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
                    seed: int = DEFAULT_BOOTSTRAP_SEED) -> np.ndarray:
    """Replicate estimates of the gate's block bootstrap, draws retained.

    ``values`` has shape ``(n_seeds, n_records)``.  With one seed this is
    `registered_experiment.match_cluster_ci`; with several it is
    `registered_experiment.seed_mean_match_cluster_ci`.  Both are reproduced
    statement for statement, including the draw order of the shared
    ``default_rng(seed)``, so ``np.percentile(draws, [2.5, 97.5])`` equals the
    interval those functions return.
    """
    values = np.atleast_2d(np.asarray(values, dtype=float))
    clusters = np.asarray(clusters)
    if values.shape[1] != len(clusters):
        raise ValueError("delta rows and clusters are not aligned")
    if values.shape[1] == 0:
        raise ValueError("cannot bootstrap an empty record set")
    if resamples <= 0:
        raise ValueError("resamples must be positive")

    _, inverse = np.unique(clusters, return_inverse=True)
    n_clusters = int(inverse.max()) + 1
    counts = np.bincount(inverse, minlength=n_clusters)
    n_seeds = values.shape[0]
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)

    if n_seeds == 1:
        sums = np.bincount(inverse, weights=values[0], minlength=n_clusters)
        for index in range(resamples):
            sampled = rng.integers(0, n_clusters, size=n_clusters)
            estimates[index] = sums[sampled].sum() / counts[sampled].sum()
        return estimates

    sums = np.stack([
        np.bincount(inverse, weights=row, minlength=n_clusters)
        for row in values
    ])
    for index in range(resamples):
        sampled_seeds = rng.integers(0, n_seeds, size=n_seeds)
        sampled_clusters = rng.integers(0, n_clusters, size=n_clusters)
        numerator = sums[sampled_seeds][:, sampled_clusters].sum()
        denominator = n_seeds * counts[sampled_clusters].sum()
        estimates[index] = numerator / denominator
    return estimates


def percentile_interval(draws: np.ndarray, level: float) -> tuple[float, float]:
    """Two-sided percentile interval at ``level`` (the gate's convention)."""
    if not 0.0 < level < 1.0:
        raise ValueError("confidence level must lie in (0, 1)")
    alpha = 1.0 - level
    return (float(np.percentile(draws, 100.0 * alpha / 2.0)),
            float(np.percentile(draws, 100.0 * (1.0 - alpha / 2.0))))


def two_sided_p(draws: np.ndarray) -> tuple[float, bool]:
    """``p = 2 * min(P(d* <= 0), P(d* >= 0))``, capped at 1.

    Returns the p-value and whether it sits on the resample resolution floor
    (one tail empty, so the true value is only known to be below 2/resamples).
    """
    draws = np.asarray(draws, dtype=float)
    if draws.size == 0:
        raise ValueError("cannot form a p-value from no draws")
    p_le = float(np.count_nonzero(draws <= 0.0)) / draws.size
    p_ge = float(np.count_nonzero(draws >= 0.0)) / draws.size
    smaller = min(p_le, p_ge)
    return min(1.0, 2.0 * smaller), smaller == 0.0


# ---------------------------------------------------------------------------
# Holm step-down
# ---------------------------------------------------------------------------


def holm_order(p_values: Sequence[float]) -> list[int]:
    """Ascending rank order of ``p_values``; ties break on input position."""
    return sorted(range(len(p_values)), key=lambda i: (p_values[i], i))


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order.

    The k-th smallest of m raw p-values is multiplied by ``m - k + 1``; the
    running maximum over the sorted sequence enforces monotonicity (so a tie,
    or a small p following a large one, can never be reported as more
    significant than its predecessor), and the result is capped at 1.
    """
    values = [float(p) for p in p_values]
    if any(not np.isfinite(p) or p < 0.0 or p > 1.0 for p in values):
        raise ValueError("raw p-values must be finite and within [0, 1]")
    m = len(values)
    adjusted = [0.0] * m
    running = 0.0
    for k, index in enumerate(holm_order(values), start=1):
        running = max(running, (m - k + 1) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def holm_rejections(p_values: Sequence[float],
                    alpha: float = FAMILY_ALPHA) -> list[bool]:
    """Holm step-down rejection flags, in the input order.

    Walking the ranks ascending, rank k rejects only if
    ``(m - k + 1) * p_(k) <= alpha`` AND every earlier rank rejected: the
    first rank that fails STOPS the procedure, so every later-ranked
    contrast fails with it regardless of its own multiplier.

    This is the only rejection rule the classifier may consult.  It is not
    implied by the rank-local intervals: raw p-values 0.020, 0.021, 0.022
    adjust to 0.060, 0.060, 0.060 — nothing rejects — while the rank-local
    interval of the leading contrast, taken at level 1 - alpha/m, can still
    exclude zero.
    """
    values = [float(p) for p in p_values]
    if any(not np.isfinite(p) or p < 0.0 or p > 1.0 for p in values):
        raise ValueError("raw p-values must be finite and within [0, 1]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    m = len(values)
    rejected = [False] * m
    stepping = True
    for k, index in enumerate(holm_order(values), start=1):
        if not stepping:
            break
        if (m - k + 1) * values[index] <= alpha:
            rejected[index] = True
        else:
            stepping = False
    return rejected


def holm_level(rank: int, m: int, alpha: float = FAMILY_ALPHA) -> float:
    """Interval level for the ``rank``-th (1-based) of ``m``: 1 - a/(m-k+1).

    The interval this level produces is RANK-LOCAL and reported only; see
    `RANK_LOCAL_NOTE`.  Rejection is `holm_rejections`, never this interval.
    """
    if not 1 <= rank <= m:
        raise ValueError("rank must lie in [1, m]")
    return 1.0 - alpha / (m - rank + 1)


# ---------------------------------------------------------------------------
# Registered classification (D12.5)
# ---------------------------------------------------------------------------


def classify(point: float, gate_ci95: tuple[float, float], *,
             rejected: bool, band: float = PARITY_BAND) -> str:
    """Label a contrast from its point, its GATE 95% interval and rejection.

    ``gate_ci95`` is the registered 95% interval (`delta_log_loss.ci95`,
    proved identical to 1e-9 before anything here runs) and decides PARITY
    only.  ``rejected`` is the Holm step-down decision for a confirmatory
    contrast (`holm_rejections`) or the unadjusted ``p_raw <= alpha`` for an
    exploratory one; favourable / adverse require it, and no interval —
    least of all a rank-local one — may stand in for it.
    """
    low, high = float(gate_ci95[0]), float(gate_ci95[1])
    if low > high:
        raise ValueError("interval bounds are inverted")
    if -band <= low and high <= band:
        return LABEL_PARITY
    if rejected and point < -band:
        return LABEL_FAVOURABLE
    if rejected and point > band:
        return LABEL_ADVERSE
    return LABEL_INCONCLUSIVE


# ---------------------------------------------------------------------------
# Gate reconstruction
# ---------------------------------------------------------------------------


@dataclass
class Contrast:
    """One gate JSON, re-derived and proved identical to its stamped result."""

    name: str
    gate_path: str
    verdict: str
    point: float
    gate_ci95: tuple[float, float]
    recomputed_ci95: tuple[float, float]
    block_count: int
    n_records: int
    n_dropped_symmetrically: int
    estimator: str
    seeds: list[str]
    evidence: dict[str, dict[str, str]]
    odds_role: str
    cluster_source_dir: str
    draws: np.ndarray = field(repr=False)


def _refuse(message: str) -> None:
    raise RefusalError(message)


def _sealed_guard(text: str) -> None:
    lowered = str(text).lower()
    if "golden" in lowered or "forward_holdout" in lowered:
        _refuse(f"refusing sealed path: {text}")


def _resolve(value: str) -> Path:
    """Resolve a repo-relative-or-absolute evidence path like the gate does."""
    path = Path(value)
    if path.is_absolute():
        return path
    return claim_gate.REPO_ROOT / path


def _load_gate(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        _refuse(f"unreadable gate JSON {path}: {exc}")
    if not isinstance(payload, dict):
        _refuse(f"gate JSON {path} is not an object")
    if payload.get("kind") != "match_model":
        _refuse(f"gate JSON {path} is not kind=match_model")
    bootstrap = payload.get("bootstrap") or {}
    if (bootstrap.get("seed") != DEFAULT_BOOTSTRAP_SEED
            or bootstrap.get("resamples") != DEFAULT_BOOTSTRAP_RESAMPLES
            or bootstrap.get("confidence") != 0.95):
        _refuse(
            f"gate JSON {path} was not produced under "
            f"{DEFAULT_BOOTSTRAP_RESAMPLES} seed-{DEFAULT_BOOTSTRAP_SEED} "
            "resamples at 95%"
        )
    contract = (payload.get("cluster_contract") or {}).get("contract")
    if contract != BOOTSTRAP_CONTRACT_VERSION:
        _refuse(f"gate JSON {path} does not declare {BOOTSTRAP_CONTRACT_VERSION}")
    return payload


def _verify_evidence(payload: dict[str, Any], gate_path: Path
                     ) -> dict[str, dict[str, str]]:
    """Re-hash every sliced eval JSON against the gate's own record."""
    seeds = [str(seed) for seed in payload.get("seeds") or []]
    arm_files = payload.get("arm_files")
    if not seeds or not isinstance(arm_files, dict):
        _refuse(f"gate JSON {gate_path} has no seeds/arm_files")
    if set(arm_files) != {"candidate", "baseline"}:
        _refuse(f"gate JSON {gate_path} arm_files is not a match_model pair")
    evidence: dict[str, dict[str, str]] = {}
    for arm in ("candidate", "baseline"):
        files = arm_files[arm]
        if set(map(str, files)) != set(seeds):
            _refuse(f"gate JSON {gate_path} arm_files[{arm}] contradicts seeds")
        for seed in seeds:
            entry = files[seed]
            if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
                _refuse(f"gate JSON {gate_path} arm_files[{arm}][{seed}] is "
                        "not a path/sha256 pair")
            recorded = str(entry["path"])
            _sealed_guard(recorded)
            path = _resolve(recorded)
            if not path.is_file():
                _refuse(f"gate evidence file does not exist: {recorded}")
            actual = claim_gate._canonical_json_sha256(json.loads(path.read_text()))
            if actual != str(entry["sha256"]):
                _refuse(
                    f"sliced eval sha256 mismatch for {recorded}: gate recorded "
                    f"{entry['sha256']}, file hashes {actual}"
                )
            evidence[f"{arm}:{seed}"] = {"path": recorded, "sha256": actual}
    return evidence


def _odds_and_clusters(payload: dict[str, Any], registry_path: Path
                       ) -> tuple[dict[str, dict[str, Any]], Path]:
    """Resolve the registered odds index and cluster dir the gate used."""
    odds_role = str(payload.get("odds_role") or "")
    _sealed_guard(odds_role)
    try:
        entry, odds_path = claim_gate._load_registry(registry_path, odds_role)
    except (OSError, ValueError) as exc:
        _refuse(f"registered odds role {odds_role!r}: {exc}")
    if str(entry.get("path")) != str(payload.get("odds_path")):
        _refuse("gate odds_path differs from the registered odds role")
    if str(entry.get("sha256")) != str(payload.get("odds_sha256")):
        _refuse("gate odds_sha256 differs from the registered odds role")
    cluster_source = entry.get("cluster_source_dir")
    if not cluster_source:
        _refuse(f"odds role {odds_role!r} has no registered cluster_source_dir")
    _sealed_guard(str(cluster_source))
    cluster_dir = _resolve(str(cluster_source)).resolve()
    declared = (payload.get("decision_inputs") or {}).get("cluster_source_dir")
    if declared is not None and _resolve(str(declared)).resolve() != cluster_dir:
        _refuse("gate decision_inputs cluster_source_dir differs from the registry")
    stamped = (payload.get("cluster_contract") or {}).get("source_dir")
    if stamped is not None and _resolve(str(stamped)).resolve() != cluster_dir:
        _refuse("gate cluster_contract source_dir differs from the registry")
    return claim_gate._odds_index(odds_path), cluster_dir


def _paired_deltas(payload: dict[str, Any], odds_index: dict[str, dict[str, Any]],
                   cluster_lookup: dict[str, str],
                   ) -> tuple[np.ndarray, list[str], list[str], int]:
    """Rebuild `(deltas, clusters, kept_ids, dropped)` exactly as the gate does."""
    seeds = [str(seed) for seed in payload["seeds"]]
    if seeds != sorted(seeds):
        _refuse("gate seed labels are not in the gate's sorted order")
    arm_files = payload["arm_files"]
    all_ll: list[list[float]] = []
    canonical_ids: list[str] | None = None
    clusters: list[str] | None = None
    dropped_total = 0
    for label in seeds:
        cand_payload = json.loads(_resolve(arm_files["candidate"][label]["path"]).read_text())
        base_payload = json.loads(_resolve(arm_files["baseline"][label]["path"]).read_text())
        cand_records = cand_payload.get("matches", [])
        base_records = base_payload.get("matches", [])
        cand_ids = [str(row.get("match_id")) for row in cand_records]
        base_ids = [str(row.get("match_id")) for row in base_records]
        if cand_ids != base_ids:
            _refuse(f"match id mismatch between arms for seed {label}")

        seed_ll: list[float] = []
        kept_ids: list[str] = []
        seed_clusters: list[str] = []
        for cand, base in zip(cand_records, base_records):
            if (not cand.get("actual_winner") or not base.get("actual_winner")
                    or not cand.get("market_odds") or not base.get("market_odds")):
                dropped_total += 1
                continue
            if cand["actual_winner"] != base["actual_winner"]:
                _refuse(f"outcome mismatch for match {cand.get('match_id')!r}")
            registered = claim_gate._registered_row(cand, odds_index)
            if registered is None:
                dropped_total += 1
                continue
            try:
                cand_odds = claim_gate._check_raw_prices(cand, registered)
                base_odds = claim_gate._check_raw_prices(base, registered)
            except ValueError as exc:
                _refuse(str(exc))
            if cand_odds != base_odds:
                _refuse(f"market_odds mismatch for match {cand.get('match_id')!r}")
            clean = {key: value for key, value in cand.items()
                     if key not in {"competition_cluster_id", "cluster_id"}}
            cluster, _resolution = cluster_id_with_resolution(clean, cluster_lookup)
            seed_clusters.append(cluster)
            kept_ids.append(str(cand["match_id"]))
            seed_ll.append(claim_gate._log_loss(cand) - claim_gate._log_loss(base))
        if not kept_ids:
            _refuse(f"no usable paired records for seed {label}")
        if canonical_ids is None:
            canonical_ids, clusters = kept_ids, seed_clusters
        elif kept_ids != canonical_ids or seed_clusters != clusters:
            _refuse("usable match ids/clusters differ across aligned seeds")
        all_ll.append(seed_ll)
    return (np.asarray(all_ll, dtype=float), list(clusters or []),
            list(canonical_ids or []), dropped_total)


def rebuild_contrast(name: str, gate_path: Path,
                     registry_path: Path = Path("docs/registered_odds.json"),
                     ) -> Contrast:
    """Re-derive one gate's paired deltas and prove the estimator is identical."""
    _sealed_guard(str(gate_path))
    payload = _load_gate(gate_path)
    evidence = _verify_evidence(payload, gate_path)
    registry = registry_path if Path(registry_path).is_absolute() else _resolve(str(registry_path))
    odds_index, cluster_dir = _odds_and_clusters(payload, registry)
    cluster_lookup = load_competition_clusters(cluster_dir)
    deltas, clusters, kept_ids, dropped = _paired_deltas(
        payload, odds_index, cluster_lookup)

    gate_delta = payload.get("delta_log_loss") or {}
    gate_ci = gate_delta.get("ci95")
    if not isinstance(gate_ci, (list, tuple)) or len(gate_ci) != 2:
        _refuse(f"gate JSON {gate_path} has no delta_log_loss.ci95")
    gate_ci95 = (float(gate_ci[0]), float(gate_ci[1]))
    gate_point = float(gate_delta["point"])

    point = float(deltas.mean())
    draws = bootstrap_draws(deltas, np.asarray(clusters, dtype=str))
    recomputed = percentile_interval(draws, 0.95)

    if abs(point - gate_point) > CI_TOLERANCE:
        _refuse(
            f"{name}: recomputed point {point!r} differs from the gate's "
            f"{gate_point!r} by more than {CI_TOLERANCE:g}"
        )
    if any(abs(a - b) > CI_TOLERANCE for a, b in zip(recomputed, gate_ci95)):
        _refuse(
            f"{name}: recomputed ci95 {list(recomputed)!r} differs from the "
            f"gate's {list(gate_ci95)!r} by more than {CI_TOLERANCE:g}; the "
            "estimator could not be proved identical, so nothing is reported"
        )
    expected_estimator = ("match_cluster_ci" if deltas.shape[0] == 1
                          else "seed_mean_match_cluster_ci")
    if payload.get("estimator") != expected_estimator:
        _refuse(f"{name}: gate estimator contradicts its seed count")
    for label, actual, recorded in (
        ("block_count", len(set(clusters)), payload.get("block_count")),
        ("n_records", len(kept_ids), payload.get("n_records")),
        ("n_dropped_symmetrically", dropped,
         payload.get("n_dropped_symmetrically")),
    ):
        if recorded is not None and int(recorded) != int(actual):
            _refuse(f"{name}: recomputed {label} {actual} != gate {recorded}")

    return Contrast(
        name=name,
        gate_path=str(gate_path),
        verdict=str(payload.get("verdict", "")),
        point=point,
        gate_ci95=gate_ci95,
        recomputed_ci95=recomputed,
        block_count=len(set(clusters)),
        n_records=len(kept_ids),
        n_dropped_symmetrically=dropped,
        estimator=expected_estimator,
        seeds=[str(seed) for seed in payload["seeds"]],
        evidence=evidence,
        odds_role=str(payload.get("odds_role", "")),
        cluster_source_dir=str(
            (payload.get("cluster_contract") or {}).get("source_dir", "")),
        draws=draws,
    )


# ---------------------------------------------------------------------------
# Family analysis
# ---------------------------------------------------------------------------


def analyse(contrasts: Sequence[Contrast], confirmatory: Sequence[str],
            alpha: float = FAMILY_ALPHA) -> list[dict[str, Any]]:
    """Attach raw p, Holm p, the step-down rejection, the rank-local
    interval and the label to each contrast.

    The rank-local interval is reported, never consulted: the label comes
    from `holm_rejections` (confirmatory) or the unadjusted `p_raw <= alpha`
    (exploratory), plus the point's position relative to the parity band,
    with parity itself decided on the gate's own 95% interval.
    """
    names = [contrast.name for contrast in contrasts]
    if len(set(names)) != len(names):
        _refuse("contrast names must be unique")
    missing = [name for name in confirmatory if name not in names]
    if missing:
        _refuse(f"confirmatory contrast(s) not supplied: {', '.join(missing)}")
    if not confirmatory:
        _refuse("at least one confirmatory contrast is required")

    raw: dict[str, tuple[float, bool]] = {}
    for contrast in contrasts:
        raw[contrast.name] = two_sided_p(contrast.draws)

    family = [name for name in confirmatory]
    m = len(family)
    family_p = [raw[name][0] for name in family]
    adjusted_p = holm_adjust(family_p)
    step_down = holm_rejections(family_p, alpha)
    ranks = {family[index]: rank
             for rank, index in enumerate(holm_order(family_p), start=1)}

    rows: list[dict[str, Any]] = []
    for contrast in contrasts:
        p_raw, on_floor = raw[contrast.name]
        if contrast.name in family:
            position = family.index(contrast.name)
            rank = ranks[contrast.name]
            level = holm_level(rank, m, alpha)
            interval = percentile_interval(contrast.draws, level)
            rejected = bool(step_down[position])
            row = {
                "contrast": contrast.name,
                "family": "confirmatory",
                "holm_rank": rank,
                "holm_family_size": m,
                "p_holm": adjusted_p[position],
                "holm_rejected": rejected,
                "rejected_at_alpha": rejected,
                "rejection_rule": (
                    "Holm step-down: adjusted p <= alpha AND every "
                    "earlier-ranked contrast rejected"),
                "rank_local_level": level,
                "rank_local_interval": list(interval),
                "rank_local_interval_note": RANK_LOCAL_NOTE,
            }
        else:
            interval = contrast.recomputed_ci95
            rejected = bool(p_raw <= alpha)
            row = {
                "contrast": contrast.name,
                "family": "exploratory",
                "holm_rank": None,
                "holm_family_size": None,
                "p_holm": None,
                "holm_rejected": None,
                "rejected_at_alpha": rejected,
                "rejection_rule": (
                    "unadjusted: p_raw <= alpha (outside the Holm family)"),
                "rank_local_level": 0.95,
                "rank_local_interval": list(interval),
                "rank_local_interval_note": EXPLORATORY_INTERVAL_NOTE,
            }
        # Deprecated aliases. `adjusted_*` is the pre-MUST-FIX-1 name for the
        # same numbers and is kept only so existing readers of these JSONs
        # (scripts/sequence_track/render_stage1_report.py) keep working; the
        # name is wrong — these intervals are rank-local, not simultaneous
        # Holm intervals, and nothing classifies off them.
        row["adjusted_level"] = row["rank_local_level"]
        row["adjusted_interval"] = list(row["rank_local_interval"])
        row["adjusted_interval_deprecated_alias_of"] = "rank_local_interval"
        row.update({
            "point": contrast.point,
            "gate_ci95": list(contrast.gate_ci95),
            "recomputed_ci95": list(contrast.recomputed_ci95),
            "p_raw": p_raw,
            "p_at_resolution_floor": on_floor,
            "label": classify(contrast.point, contrast.gate_ci95,
                              rejected=rejected, band=PARITY_BAND),
            "parity_basis": PARITY_BASIS,
            "block_count": contrast.block_count,
            "n_records": contrast.n_records,
            "descriptive_block_count": contrast.block_count < MIN_RECOMMENDED_BLOCKS,
            "gate_verdict": contrast.verdict,
            "gate_path": contrast.gate_path,
            "estimator": contrast.estimator,
            "seeds": contrast.seeds,
            "odds_role": contrast.odds_role,
            "cluster_source_dir": contrast.cluster_source_dir,
            "n_dropped_symmetrically": contrast.n_dropped_symmetrically,
            "evidence": contrast.evidence,
        })
        rows.append(row)
    return rows


def _format(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _interval_text(interval: Sequence[float], digits: int = 4) -> str:
    return f"[{interval[0]:+.{digits}f}, {interval[1]:+.{digits}f}]"


def _p_text(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if value is None:
        return "— (exploratory)"
    if value == 0.0 and row.get("p_at_resolution_floor"):
        return f"<{2.0 / DEFAULT_BOOTSTRAP_RESAMPLES:g}"
    return f"{value:.4f}"


def render_markdown(rows: Sequence[dict[str, Any]], *,
                    slice_label: str | None = None,
                    alpha: float = FAMILY_ALPHA) -> str:
    """Render the D12.4/D12.5 table. Only the registered quantities appear."""
    confirmatory = [row for row in rows if row["family"] == "confirmatory"]
    exploratory = [row for row in rows if row["family"] == "exploratory"]
    m = confirmatory[0]["holm_family_size"] if confirmatory else 0
    alpha_text = f"{alpha:g}"
    lines = [
        "# Stage 1 — Holm adjustment across the confirmatory contrasts (D12.4/D12.5)",
        "",
        f"Registered expectation: **{REGISTERED_EXPECTATION}**.",
        "",
        f"Paired winner log loss, candidate minus baseline (negative favours the "
        f"candidate), {'slice ' + slice_label + ', ' if slice_label else ''}"
        f"uncertainty from `{BOOTSTRAP_CONTRACT_VERSION}` — "
        f"{DEFAULT_BOOTSTRAP_RESAMPLES:,} seed-{DEFAULT_BOOTSTRAP_SEED} "
        "whole-event resamples. Every interval below is a percentile interval "
        "over those same resamples, and each contrast's 95% interval was "
        f"asserted equal to its gate JSON's `ci95` to within {CI_TOLERANCE:g} "
        "before any p-value was formed.",
        "",
        f"Holm step-down over the {m} confirmatory contrast(s); "
        "exploratory contrasts are reported unadjusted and do not enter the "
        "family. Two-sided bootstrap p-value "
        "`p = 2 * min(P(d* <= 0), P(d* >= 0))`.",
        "",
        "**Rejection.** A confirmatory contrast rejects only under the Holm "
        "step-down: its adjusted p must be at or below "
        f"{alpha_text}, AND every earlier-ranked contrast must have rejected "
        "— the first rank that fails stops the procedure and every "
        "later-ranked contrast fails with it. An exploratory contrast is "
        f"outside the family, so its rejection is the unadjusted `p_raw <= "
        f"{alpha_text}`.",
        "",
        f"**The `rank-local interval` column is a {RANK_LOCAL_NOTE}.** It is "
        "reported, never classified off: raw p-values 0.020, 0.021, 0.022 "
        "all adjust to 0.060, so nothing in the family rejects, yet a "
        "rank-local interval can still exclude zero. Exploratory rows carry "
        "the unadjusted 95% interval in that column instead.",
        "",
        f"**Labels.** Parity if the **gate's own 95% interval** lies inside "
        f"±{PARITY_BAND:g} (parity is a statement about the registered 95% "
        "estimate, so it is decided on `gate ci95`, not on a rank-local "
        "interval); favourable / adverse if the contrast REJECTED and the "
        "point is beyond that band; else inconclusive.",
        "",
        "| contrast | family | point | gate ci95 | recomputed ci95 | raw p | "
        "Holm p | rejects | rank-local level | rank-local interval | label | "
        "blocks | n |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in list(confirmatory) + list(exploratory):
        lines.append(
            f"| {row['contrast']} | {row['family']} | {row['point']:+.4f} | "
            f"{_interval_text(row['gate_ci95'])} | "
            f"{_interval_text(row['recomputed_ci95'])} | "
            f"{_p_text(row, 'p_raw')} | {_p_text(row, 'p_holm')} | "
            f"{'yes' if row['rejected_at_alpha'] else 'no'} | "
            f"{_format(row['rank_local_level'], 4)} | "
            f"{_interval_text(row['rank_local_interval'])} | "
            f"**{row['label']}** | {row['block_count']} | {row['n_records']} |"
        )
    if any(row["descriptive_block_count"] for row in rows):
        lines += [
            "",
            f"A contrast with fewer than {MIN_RECOMMENDED_BLOCKS} blocks is "
            "descriptive (D12.3) and carries no claim.",
        ]
    lines += [
        "",
        "## Evidence",
        "",
        "Every sliced eval JSON below was re-hashed against the hash its gate "
        "JSON recorded, and the deltas, blocks and bootstrap were rebuilt from "
        "those files with `claim_gate`'s own helpers.",
        "",
    ]
    for row in list(confirmatory) + list(exploratory):
        lines.append(f"- **{row['contrast']}** — gate `{row['gate_path']}`, "
                     f"odds role `{row['odds_role']}`, clusters "
                     f"`{row['cluster_source_dir']}`, estimator "
                     f"`{row['estimator']}`, gate verdict "
                     f"`{row['gate_verdict']}`")
        for key in sorted(row["evidence"]):
            entry = row["evidence"][key]
            lines.append(f"  - {key}: `{entry['path']}` sha256 verified")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_gate_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--gate takes NAME=PATH (e.g. C-B=.../gate_c_minus_b.json)")
    name, _, path = value.partition("=")
    name, path = name.strip(), path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError("--gate takes a non-empty NAME=PATH")
    return name, Path(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="holm_stage1.py",
        description="Holm adjustment across the stage-1 confirmatory contrasts.",
    )
    parser.add_argument("--gate", action="append", required=True,
                        type=_parse_gate_argument, metavar="NAME=PATH",
                        help="one claim_gate --kind match_model JSON, named")
    parser.add_argument("--confirmatory", nargs="+", required=True,
                        metavar="NAME",
                        help="the confirmatory family (e.g. C-B B-A C-A); any "
                             "other --gate is reported unadjusted as exploratory")
    parser.add_argument("--registry-path", default="docs/registered_odds.json")
    parser.add_argument("--slice", default=None,
                        help="slice label for the report header, e.g. 50000")
    parser.add_argument("--alpha", type=float, default=FAMILY_ALPHA)
    parser.add_argument("--verify-gate", action="store_true",
                        help="additionally replay each gate in full through "
                             "claim_gate.verify_gate_payload (slow)")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else sys.argv[1:])

    registry = Path(args.registry_path)
    contrasts = []
    for name, gate_path in args.gate:
        contrast = rebuild_contrast(name, gate_path, registry)
        if args.verify_gate:
            payload = json.loads(gate_path.read_text())
            try:
                replayed = claim_gate.verify_gate_payload(payload, registry)
            except (OSError, ValueError) as exc:
                _refuse(f"{name}: gate JSON does not replay: {exc}")
            if replayed != contrast.verdict:
                _refuse(f"{name}: gate JSON does not replay to its own verdict")
        contrasts.append(contrast)

    rows = analyse(contrasts, args.confirmatory, args.alpha)
    payload = {
        "contract": {
            "decision_rule": "docs/sequence_track/stage1_acceptance.md D12.4/D12.5",
            "bootstrap_contract": BOOTSTRAP_CONTRACT_VERSION,
            "resamples": DEFAULT_BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": DEFAULT_BOOTSTRAP_SEED,
            "delta_sign": "candidate minus baseline log loss; negative favours "
                          "the candidate",
            "p_value": "two-sided percentile: 2 * min(P(d* <= 0), P(d* >= 0))",
            "multiplicity": "Holm step-down over the confirmatory family only",
            "rejection_rule": (
                "a confirmatory contrast rejects iff its Holm-adjusted p is "
                "<= alpha AND every earlier-ranked contrast rejected (the "
                "step-down stopping rule); an exploratory contrast is "
                "outside the family and uses the unadjusted p_raw <= alpha"),
            "rank_local_level": "1 - alpha / (m - k + 1) for the k-th of m by raw p",
            "rank_local_interval": RANK_LOCAL_NOTE,
            "rank_local_interval_is_not_a_decision": (
                "rank-local intervals are reported only; nothing is "
                "classified off them. Raw p 0.020, 0.021, 0.022 adjust to "
                "0.060 each, so no contrast rejects, yet a rank-local "
                "interval may still exclude zero"),
            "adjusted_level_deprecated_alias_of": "rank_local_level",
            "adjusted_interval_deprecated_alias_of": "rank_local_interval",
            "classification": (
                "parity if the gate 95% interval lies inside the parity "
                "band; favourable/adverse if the contrast rejected and the "
                "point is beyond that band; else inconclusive"),
            "parity_basis": PARITY_BASIS,
            "alpha": args.alpha,
            "parity_band": PARITY_BAND,
            "ci_identity_tolerance": CI_TOLERANCE,
            "slice": args.slice,
            "registry_path": str(registry),
            "registered_expectation": REGISTERED_EXPECTATION,
        },
        "contrasts": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(
        render_markdown(rows, slice_label=args.slice, alpha=args.alpha))
    for row in rows:
        rejects = "rejects" if row["rejected_at_alpha"] else "does not reject"
        print(f"{row['contrast']}: {row['label']} ({row['family']}; "
              f"{rejects} at alpha={args.alpha:g})")
    print(f"registered expectation: {REGISTERED_EXPECTATION}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RefusalError as error:  # pragma: no cover - CLI surface
        print(f"REFUSED: {error}", file=sys.stderr)
        raise SystemExit(2)
