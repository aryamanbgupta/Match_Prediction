#!/usr/bin/env python
"""Decision gate for paired sliced-evaluation evidence.

Prices, placement, settlement, and competition blocks are reconstructed from
registered sources.  Derived values stamped into an eval JSON are never gate
evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

from registered_experiment import (  # noqa: E402
    match_cluster_ci,
    seed_mean_match_cluster_ci,
)
from sim_eval.eval_statistics import (  # noqa: E402
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    bootstrap_ratio_of_sums_difference_ci,
    cluster_id_with_resolution,
    flat_bet_team,
    load_competition_clusters,
    settle_flat_policy,
    settle_kelly_policy,
)
from sim_eval.market_math import CostModel, DEFAULT_SCENARIOS, implied_probs  # noqa: E402

SEED_FLOOR = 0.007
_SEED_RE = re.compile(r"seed(\d+)", re.IGNORECASE)
GATE_PAIRS = {
    "betting_layer": {
        "metric_a": {"flat_pnl": "higher", "kelly_pnl": "higher"},
    },
}
MANUAL_VERDICTS = {"LANDED", "TABLED", "FAILED", "DESCRIPTIVE"}


@dataclass(frozen=True)
class MetricDelta:
    point: float
    ci95: tuple[float, float]


@dataclass(frozen=True)
class Verdict:
    verdict: str
    kind: str
    seed_count: int
    seeds: list[str]
    arm_files: dict[str, dict[str, dict[str, str]]]
    decision_inputs: dict[str, Any]
    estimator: str
    provisional: bool
    cost_model: dict[str, Any]
    odds_role: str
    odds_path: str
    odds_sha256: str
    cluster_contract: dict[str, Any]
    block_count: int
    n_records: int
    n_profit_rows: int
    profit_block_count: int
    n_dropped_symmetrically: int
    delta_log_loss: MetricDelta
    delta_profit: MetricDelta
    delta_roi: MetricDelta | None
    cost_scenario_diagnostics: list[dict[str, Any]]
    bootstrap: dict[str, Any]
    metric_a: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _repo_path(path: Path) -> str:
    if not path.is_absolute():
        path = REPO_ROOT / path
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"evidence path must be repo-relative: {path}") from exc


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"evidence path must be repo-relative: {value}")
    resolved = (REPO_ROOT / path).resolve()
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:  # pragma: no cover - defense in depth
        raise ValueError(f"evidence path escapes repository: {value}") from exc
    return resolved


def _paths(value: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(value, (str, Path)):
        values = [value]
    else:
        values = list(value)
    paths = [Path(path) if Path(path).is_absolute() else REPO_ROOT / Path(path)
             for path in values]
    if not paths:
        raise ValueError("at least one sliced eval JSON is required per arm")
    return paths


def _seed_labels(paths: list[Path], seeds: Sequence[str | int] | None) -> list[str]:
    if seeds is not None:
        labels = [str(seed) for seed in seeds]
        if len(labels) != len(paths):
            raise ValueError("seed label count must match the number of files per arm")
    else:
        labels = []
        for path in paths:
            match = _SEED_RE.search(path.stem)
            if not match:
                if len(paths) == 1:
                    labels.append("single")
                    continue
                raise ValueError(
                    f"cannot infer seed from {path}; pass seeds explicitly"
                )
            labels.append(match.group(1))
    if len(set(labels)) != len(labels):
        raise ValueError("seed labels must be unique")
    return labels


def _arm_files(paths: list[Path], labels: list[str], arm: str) -> dict[str, dict[str, str]]:
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"duplicate paths within {arm} arm")
    hashes = [_canonical_json_sha256(json.loads(path.read_text())) for path in paths]
    if len(set(hashes)) != len(hashes):
        raise ValueError(f"duplicate file hashes within {arm} arm")
    return {
        label: {"path": _repo_path(path), "sha256": digest}
        for label, path, digest in zip(labels, paths, hashes)
    }


def _stamped_seeds(payload: dict[str, Any]) -> list[Any]:
    summary = payload.get("summary") or {}
    if not isinstance(summary, dict):
        return []
    stamps = []
    if "seed" in summary:
        stamps.append(summary["seed"])
    if "model_seed" in summary:
        stamps.append(summary["model_seed"])
    provenance = summary.get("provenance") or {}
    if isinstance(provenance, dict) and "seed" in provenance:
        stamps.append(provenance["seed"])
    return stamps


def _check_stamped_seed(payload: dict[str, Any], label: str, path: Path,
                        seed_count: int) -> None:
    stamped = _stamped_seeds(payload)
    if seed_count > 1 and not stamped:
        raise ValueError(f"unstamped seed file: {path}")
    if any(str(value) != label for value in stamped):
        raise ValueError(
            f"stamped seed {stamped!r} in {path} does not match seed label {label!r}"
        )


def _prediction_vector(rows: Sequence[dict[str, Any]]) -> str:
    """Canonical numeric form of the per-match predictions (Astra round 4:
    string spellings such as "0.75" / "0.750" must compare equal)."""
    vector = []
    for row in rows:
        probs = row.get("simulated_prob") or {}
        # `+ 0.0` folds -0.0 into 0.0 (Astra round 5).
        vector.append(sorted((str(team), float(value) + 0.0) for team, value in probs.items()))
    return json.dumps(vector, separators=(",", ":"))


def _metric_name(value: Any, slot: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        raise ValueError(f"{slot} must name a metric")
    forbidden = {"direction", "favourable", "favorable"} & set(value)
    if forbidden:
        raise ValueError("metric direction fields are forbidden")
    if set(value) != {"name"}:
        raise ValueError(f"{slot} may contain only name")
    return str(value["name"])


def _reject_direction_fields(value: Any) -> None:
    if isinstance(value, dict):
        if {"direction", "favourable", "favorable"} & set(value):
            raise ValueError("metric direction fields are forbidden")
        for child in value.values():
            _reject_direction_fields(child)
    elif isinstance(value, list):
        for child in value:
            _reject_direction_fields(child)


def _metric_specs(kind: str, payload: dict[str, Any]) -> tuple[dict[str, str], dict[str, str] | None]:
    table = GATE_PAIRS[kind]
    a_name = _metric_name(payload.get("metric_a"), "metric_a")
    b_name = _metric_name(payload.get("metric_b"), "metric_b")
    if a_name not in table["metric_a"]:
        raise ValueError(f"unknown metric_a {a_name!r} for {kind}")
    if b_name is not None:
        raise ValueError(f"{kind} does not accept metric_b")
    a = {"name": a_name, "favourable": table["metric_a"][a_name]}
    return a, None


def _load_registry(path: Path, role: str) -> tuple[dict[str, Any], Path]:
    payload = json.loads(path.read_text())
    entries = payload.get("registered_odds", payload.get("roles", []))
    if isinstance(entries, dict):
        entries = [dict(value, role=key) for key, value in entries.items()]
    matches = [entry for entry in entries if entry.get("role") == role]
    if len(matches) != 1:
        raise ValueError(f"odds role {role!r} is not uniquely registered in {path}")
    entry = matches[0]
    odds_path = Path(entry["path"])
    if not odds_path.is_absolute():
        odds_path = REPO_ROOT / odds_path
    actual_hash = _sha256(odds_path)
    if actual_hash != entry.get("sha256"):
        raise ValueError(f"registered odds sha256 mismatch for {odds_path}")
    return entry, odds_path


def _odds_index(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text())
    rows = payload.get("matches", payload if isinstance(payload, list) else [])
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        for key in ("match_id", "cricsheet_id", "display_match_id"):
            identity = row.get(key)
            if identity:
                identity = str(identity)
                previous = index.get(identity)
                if previous is not None and previous is not row:
                    raise ValueError(f"ambiguous registered odds identity {identity!r}")
                index[identity] = row
    return index


def _registered_row(record: dict[str, Any], index: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    for key in ("match_id", "cricsheet_id", "display_match_id"):
        value = record.get(key)
        if value is not None and str(value) in index:
            return index[str(value)]
    return None


def _prices(row: dict[str, Any]) -> dict[str, float]:
    odds = row.get("odds", {}).get("winner", row.get("market_odds", {}))
    return {str(team): float(value) for team, value in odds.items()
            if team not in {"timestamp", "scheduled_start_timestamp"}}


def _check_raw_prices(record: dict[str, Any], registered: dict[str, Any]) -> dict[str, float]:
    expected = _prices(registered)
    raw = record.get("market_odds") or {}
    actual = {str(team): float(value) for team, value in raw.items()
              if team not in {"timestamp", "scheduled_start_timestamp"}}
    if set(actual) != set(expected) or any(
        not math.isclose(actual[team], expected[team], rel_tol=0.0, abs_tol=1e-12)
        for team in expected if team in actual
    ):
        raise ValueError(
            f"market_odds mismatch for match {record.get('match_id')!r}"
        )
    return expected


def _log_loss(record: dict[str, Any]) -> float:
    winner = str(record["actual_winner"])
    try:
        probability = float(record["simulated_prob"][winner])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid simulated_prob for match {record.get('match_id')!r}") from exc
    if not math.isfinite(probability):
        raise ValueError(f"non-finite simulated probability for match {record.get('match_id')!r}")
    return -math.log(min(max(probability, 1e-15), 1.0))


def _profit(record: dict[str, Any], odds: dict[str, float], cost: CostModel) -> float | None:
    probabilities = implied_probs(odds, remove_margin=True)
    model = {str(team): float(value) for team, value in record["simulated_prob"].items()}
    derived = {
        "actual_winner": record["actual_winner"],
        "market_odds": odds,
        "edge": {team: model[team] - probabilities[team] for team in odds},
        # flat_bet_team uses this only as a legacy evidence-presence sentinel.
        "realized_pnl": 0.0,
    }
    bet_team = flat_bet_team(derived, edge_threshold=0.0)
    if bet_team is None:
        return None
    settled = settle_flat_policy(bet_team, odds[bet_team], record["actual_winner"], cost)
    if settled is None:  # outcome was checked before this point
        raise ValueError(f"could not settle match {record.get('match_id')!r}")
    return float(settled)


def _interval(values: np.ndarray, clusters: np.ndarray, seed_count: int) -> tuple[float, float]:
    if seed_count == 1:
        result = match_cluster_ci(values[0], clusters,
                                  DEFAULT_BOOTSTRAP_RESAMPLES,
                                  DEFAULT_BOOTSTRAP_SEED)
    else:
        result = seed_mean_match_cluster_ci(values, clusters,
                                            DEFAULT_BOOTSTRAP_RESAMPLES,
                                            DEFAULT_BOOTSTRAP_SEED)
    interval = (float(result[0]), float(result[1]))
    if not all(math.isfinite(value) for value in interval):
        raise ValueError("bootstrap produced a non-finite interval")
    return interval


def _favourable(delta: MetricDelta, direction: str) -> bool:
    return delta.ci95[1] < 0.0 if direction == "lower" else delta.ci95[0] > 0.0


def _unfavourable(delta: MetricDelta, direction: str) -> bool:
    return delta.ci95[0] > 0.0 if direction == "lower" else delta.ci95[1] < 0.0


def _classification(kind: str, ll: MetricDelta, profit: MetricDelta,
                    provisional: bool, descriptive: bool, *,
                    metric_a_favourable: str | None = None) -> str:
    if descriptive:
        return "DESCRIPTIVE"
    ll_clear = ll.ci95[1] < 0.0
    ll_floor = -ll.point >= SEED_FLOOR
    profit_harmful = profit.ci95[1] < 0.0
    profit_clear = profit.ci95[0] > 0.0
    if kind == "match_model":
        if ll_clear and (ll_floor or not provisional) and not profit_harmful:
            return "PROMISING" if provisional else "LANDED"
        if ll_clear and (ll_floor or not provisional) and profit_harmful:
            return "TABLED"
        return "FAILED"
    if kind == "betting_layer":
        primary = _favourable(profit, metric_a_favourable or "higher")
        if primary:
            return "PROMISING" if provisional else "LANDED"
        return "FAILED"
    raise ValueError("kind must be match_model or betting_layer")


def _metric_delta(payload: dict[str, Any], key: str) -> MetricDelta:
    value = payload.get(key)
    if not isinstance(value, dict) or not isinstance(value.get("ci95"), (list, tuple)):
        raise ValueError(f"gate payload has invalid {key}")
    ci = value["ci95"]
    if len(ci) != 2:
        raise ValueError(f"gate payload has invalid {key}.ci95")
    point = float(value["point"])
    interval = (float(ci[0]), float(ci[1]))
    if not all(math.isfinite(number) for number in (point, *interval)):
        raise ValueError(f"gate payload has non-finite {key}")
    return MetricDelta(point, interval)


def verify_gate_payload(payload: dict[str, Any],
                        registry_path: str | Path = "docs/registered_odds.json") -> str:
    """Verify evidence hashes and replay an automated gate from source files."""
    kind = str(payload.get("kind", ""))
    if payload.get("gate_mode") == "manual_sim_prop":
        expected = {"gate_mode", "kind", "idea", "gate_script", "detail_json",
                    "verdict", "note"}
        if set(payload) != expected or kind != "sim_prop":
            raise ValueError("invalid manual_sim_prop gate structure")
        verdict = str(payload.get("verdict", "")).upper()
        if verdict not in MANUAL_VERDICTS:
            raise ValueError("manual_sim_prop gate has invalid verdict")
        if not str(payload.get("idea", "")).strip() or not str(payload.get("note", "")).strip():
            raise ValueError("manual_sim_prop gate requires idea and gate-pair note")
        evidence = [payload.get("gate_script"), *(payload.get("detail_json") or [])]
        if (not isinstance(payload.get("detail_json"), list)
                or not payload["detail_json"]):
            raise ValueError("manual_sim_prop gate requires evidence files")
        for item in evidence:
            if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
                raise ValueError("manual_sim_prop evidence requires path and sha256")
            path = _resolve_repo_path(str(item["path"]))
            if not path.is_file() or _sha256(path) != str(item["sha256"]):
                raise ValueError(f"manual_sim_prop evidence sha256 mismatch: {item['path']}")
        return verdict

    seed_count = int(payload.get("seed_count", 0))
    if seed_count < 1:
        raise ValueError("gate payload seed_count must be positive")
    seeds = payload.get("seeds")
    arm_files = payload.get("arm_files")
    if not isinstance(seeds, list) or len(seeds) != seed_count:
        raise ValueError("gate payload seed_count contradicts seeds")
    if len({str(seed) for seed in seeds}) != seed_count:
        raise ValueError("gate payload seeds must be unique")
    if not isinstance(arm_files, dict) or not arm_files:
        raise ValueError("gate payload requires arm_files")
    expected_arms = {"match_model": {"candidate", "baseline"},
                     "betting_layer": {"candidate", "baseline", "metrics"}}.get(kind)
    if expected_arms is None or set(arm_files) != expected_arms:
        raise ValueError("gate payload arm_files contradicts kind")
    for arm, files in arm_files.items():
        if not isinstance(files, dict) or len(files) != seed_count:
            raise ValueError(f"gate payload seed_count contradicts arm_files[{arm!r}]")
        if set(map(str, files)) != set(map(str, seeds)):
            raise ValueError(f"gate payload seeds contradict arm_files[{arm!r}]")
        for entry in files.values():
            if (not isinstance(entry, dict)
                    or set(entry) != {"path", "sha256"}
                    or not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", "")))):
                raise ValueError("every gate payload arm_files entry requires path and sha256")
            path = _resolve_repo_path(str(entry["path"]))
            if not path.is_file():
                raise ValueError(f"gate evidence file does not exist: {entry['path']}")
            actual = _canonical_json_sha256(json.loads(path.read_text()))
            if actual != entry["sha256"]:
                raise ValueError(f"gate evidence sha256 mismatch: {entry['path']}")
    provisional = payload.get("provisional")
    if not isinstance(provisional, bool) or provisional != (seed_count < 5):
        raise ValueError("gate payload provisional flag contradicts seed_count")
    expected_estimator = (
        "match_cluster_ci" if seed_count == 1 else "seed_mean_match_cluster_ci"
    )
    if payload.get("estimator") != expected_estimator:
        raise ValueError("gate payload estimator contradicts seed_count")
    inputs = payload.get("decision_inputs")
    required_inputs = {"kind", "odds_role", "cost_model", "seeds",
                       "cluster_source_dir", "cost_scenarios", "metrics_json"}
    if not isinstance(inputs, dict) or set(inputs) != required_inputs:
        raise ValueError("gate payload has invalid decision_inputs")
    if inputs["kind"] != kind or list(map(str, inputs["seeds"])) != list(map(str, seeds)):
        raise ValueError("gate payload decision_inputs contradict top-level fields")
    cost_data = inputs["cost_model"]
    scenarios = [CostModel(**item) for item in inputs["cost_scenarios"]]
    paths = lambda arm: [arm_files[arm][str(seed)]["path"] for seed in seeds]
    metrics = paths("metrics") if kind == "betting_layer" else None
    if inputs["metrics_json"] != metrics:
        raise ValueError("gate payload metrics_json contradicts evidence paths")
    recomputed = decide(
        paths("candidate"), paths("baseline"), kind, CostModel(**cost_data),
        odds_role=str(inputs["odds_role"]),
        cluster_source_dir=inputs["cluster_source_dir"], seeds=seeds,
        registry_path=registry_path, cost_scenarios=scenarios,
        metrics_json=metrics,
    ).as_dict()
    # Compare in serialized form: the dataclass emits tuples where the JSON
    # file carries lists (Astra round 4).
    if json.loads(json.dumps(recomputed, sort_keys=True)) != json.loads(json.dumps(payload, sort_keys=True)):
        raise ValueError("gate payload differs from replayed decision")
    return str(recomputed["verdict"])


def decide(candidate: str | Path | Sequence[str | Path],
           baseline: str | Path | Sequence[str | Path], kind: str,
           cost: CostModel | None = None, *, odds_role: str = "odds_iteration_v2",
           cluster_source_dir: str | Path | None = None,
           seeds: Sequence[str | int] | None = None,
           registry_path: str | Path = "docs/registered_odds.json",
           cost_scenarios: Sequence[CostModel] | None = None,
           metrics_json: str | Path | Sequence[str | Path] | None = None) -> Verdict:
    """Decide a paired claim from one or more aligned sliced JSONs per arm."""
    cost = cost or CostModel.none()
    if kind not in {"match_model", "betting_layer"}:
        raise ValueError("automated gate kind must be match_model or betting_layer")
    if kind == "match_model" and (candidate is None or baseline is None
                                  or metrics_json is not None):
        raise ValueError("match_model requires candidate and baseline sliced JSONs only")
    if kind == "betting_layer" and (candidate is None or baseline is None
                                    or metrics_json is None):
        raise ValueError("betting_layer requires candidate, baseline, and metrics_json")
    arm_files: dict[str, dict[str, dict[str, str]]] = {}
    candidate_by_seed: dict[str, Path] = {}
    baseline_by_seed: dict[str, Path] = {}
    metrics_by_seed: dict[str, Path] = {}
    labels: list[str]
    if kind in {"match_model", "betting_layer"}:
        candidate_paths, baseline_paths = _paths(candidate), _paths(baseline)
        if len(candidate_paths) != len(baseline_paths):
            raise ValueError("candidate and baseline must contain identical seed sets")
        candidate_seeds = _seed_labels(candidate_paths, seeds)
        baseline_seeds = _seed_labels(baseline_paths, seeds)
        if set(candidate_seeds) != set(baseline_seeds):
            raise ValueError("candidate and baseline must contain identical seed sets")
        candidate_by_seed = dict(zip(candidate_seeds, candidate_paths))
        baseline_by_seed = dict(zip(baseline_seeds, baseline_paths))
        labels = sorted(candidate_by_seed)
        arm_files = {
            "candidate": _arm_files(candidate_paths, candidate_seeds, "candidate"),
            "baseline": _arm_files(baseline_paths, baseline_seeds, "baseline"),
        }
    if metrics_json is not None:
        metric_paths = _paths(metrics_json)
        metric_seeds = _seed_labels(metric_paths, seeds)
        metrics_by_seed = dict(zip(metric_seeds, metric_paths))
        metric_labels = sorted(metrics_by_seed)
        if kind == "betting_layer" and metric_labels != labels:
            raise ValueError("metrics and sliced arms must contain identical seed sets")
        labels = metric_labels
        arm_files["metrics"] = _arm_files(metric_paths, metric_seeds, "metrics")

    scenario_models = list(cost_scenarios or DEFAULT_SCENARIOS)
    decision_inputs = {
        "kind": kind,
        "odds_role": odds_role,
        "cost_model": cost.as_dict(),
        "seeds": labels,
        "cluster_source_dir": _repo_path(Path(cluster_source_dir))
        if cluster_source_dir is not None else None,
        "cost_scenarios": [scenario.as_dict() for scenario in scenario_models],
        "metrics_json": ([arm_files["metrics"][label]["path"] for label in labels]
                         if metrics_json is not None else None),
    }

    registry = Path(registry_path)
    if not registry.is_absolute():
        registry = REPO_ROOT / registry
    entry, odds_path = _load_registry(registry, odds_role)
    odds_index = _odds_index(odds_path)
    cluster_source = entry.get("cluster_source_dir")
    if not cluster_source:
        raise ValueError(f"odds role {odds_role!r} has no registered cluster_source_dir")
    cluster_dir = Path(cluster_source)
    if not cluster_dir.is_absolute():
        cluster_dir = REPO_ROOT / cluster_dir
    cluster_dir = cluster_dir.resolve()
    if cluster_source_dir is not None:
        supplied = Path(cluster_source_dir)
        if not supplied.is_absolute():
            supplied = REPO_ROOT / supplied
        if supplied.resolve() != cluster_dir:
            raise ValueError("cluster_source_dir differs from the registered directory")
    cluster_lookup = load_competition_clusters(cluster_dir)

    sliced_payloads: dict[str, dict[str, dict[str, Any]]] = {}
    for arm, paths_by_seed in (("candidate", candidate_by_seed),
                               ("baseline", baseline_by_seed)):
        if not paths_by_seed:
            continue
        payloads: dict[str, dict[str, Any]] = {}
        for label in labels:
            path = paths_by_seed[label]
            payload = json.loads(path.read_text())
            _check_stamped_seed(payload, label, path, len(labels))
            payloads[label] = payload
        sliced_payloads[arm] = payloads

    metric_payloads: dict[str, dict[str, Any]] = {}
    for label, path in metrics_by_seed.items():
        payload = json.loads(path.read_text())
        _check_stamped_seed(payload, label, path, len(labels))
        metric_payloads[label] = payload

    all_ll: list[list[float]] = []
    all_profit: list[list[float]] = []
    scenario_values: list[list[list[float]]] = [list() for _ in scenario_models]
    betting_pnl = {"candidate": [], "baseline": []}
    betting_stake = {"candidate": [], "baseline": []}
    scenario_betting = [
        {"candidate_pnl": [], "baseline_pnl": [],
         "candidate_stake": [], "baseline_stake": []}
        for _ in scenario_models
    ]
    canonical_ids: list[str] | None = None
    clusters: list[str] | None = None
    canonical_profit_ids: list[str] | None = None
    profit_clusters: list[str] | None = None
    resolutions: Counter[str] = Counter()
    dropped_total = 0
    reliable = True
    metric_a_spec: dict[str, str] | None = None
    kept_prediction_vectors = {"candidate": set(), "baseline": set()}
    kept_placement_vectors = {"candidate": set(), "baseline": set()}

    if metrics_json is not None:
        for label in labels:
            path = metrics_by_seed[label]
            payload = metric_payloads[label]
            if payload.get("kind") != kind:
                raise ValueError(f"metrics kind in {path} does not match {kind}")
            _reject_direction_fields(payload)
            a, b = _metric_specs(kind, payload)
            if metric_a_spec is None:
                metric_a_spec = a
            elif metric_a_spec != a:
                raise ValueError("metric definitions differ across seeds")
            reliable = reliable and payload.get("summary", {}).get(
                "bootstrap_reliable", True
            ) is not False
            metric_rows = payload.get("rows", [])
            seed_a, seed_b, kept_ids, seed_clusters = [], [], [], []
            seed_profit_ids, seed_profit_clusters = [], []
            if kind == "betting_layer":
                cand_payload = sliced_payloads["candidate"][label]
                base_payload = sliced_payloads["baseline"][label]
                cand_records = cand_payload.get("matches", [])
                base_records = base_payload.get("matches", [])
                cand_ids = [str(row.get("match_id")) for row in cand_records]
                base_ids = [str(row.get("match_id")) for row in base_records]
                metric_ids = [str(row.get("match_id")) for row in metric_rows]
                if cand_ids != base_ids or cand_ids != metric_ids:
                    raise ValueError(f"match id mismatch between betting inputs for seed {label}")
                reliable = reliable and cand_payload.get("summary", {}).get(
                    "bootstrap_reliable", True) is not False
                reliable = reliable and base_payload.get("summary", {}).get(
                    "bootstrap_reliable", True) is not False
                row_iter = zip(metric_rows, cand_records, base_records)
            cand_pnls, base_pnls, cand_stakes, base_stakes = [], [], [], []
            seed_scenario_parts = [([], [], [], []) for _ in scenario_models]
            placement_vectors = {"candidate": [], "baseline": []}

            for row, cand_record, base_record in row_iter:
                match_id = str(row.get("match_id"))
                registered = odds_index.get(match_id)
                if registered is None:
                    raise ValueError(f"metrics match_id {match_id!r} is not registered")
                clean = {"match_id": match_id}
                cluster, resolution = cluster_id_with_resolution(clean, cluster_lookup)
                resolutions[resolution] += 1
                if kind == "betting_layer":
                    assert cand_record is not None and base_record is not None
                    if (not cand_record.get("actual_winner")
                            or cand_record.get("actual_winner") != base_record.get("actual_winner")):
                        raise ValueError(f"missing or mismatched outcome for match {match_id!r}")
                    cand_odds = _check_raw_prices(cand_record, registered)
                    base_odds = _check_raw_prices(base_record, registered)
                    if cand_odds != base_odds:
                        raise ValueError(f"market_odds mismatch for match {match_id!r}")
                    cand_probs = {str(k): float(v) for k, v in cand_record["simulated_prob"].items()}
                    base_probs = {str(k): float(v) for k, v in base_record["simulated_prob"].items()}
                    if not all(math.isfinite(v) for v in (*cand_probs.values(), *base_probs.values())):
                        raise ValueError(f"non-finite simulated probability for match {match_id!r}")
                    if set(cand_probs) != set(base_probs) or any(
                        abs(cand_probs[team] - base_probs[team]) > 1e-12 for team in cand_probs
                    ):
                        raise ValueError("betting_layer claim changes probabilities; use kind=match_model")
                    required = {"cand_bet_team", "cand_stake",
                                "base_bet_team", "base_stake"}
                    if not required.issubset(row):
                        raise ValueError(f"incomplete placements for match {match_id!r}")
                    if {"cand_a", "base_a", "cand_b", "base_b"} & set(row):
                        raise ValueError("betting_layer rows supply placements, not values")
                    stakes = [float(row["cand_stake"]), float(row["base_stake"])]
                    if not all(math.isfinite(stake) and stake >= 0 for stake in stakes):
                        raise ValueError(f"invalid stake for match {match_id!r}")
                    if a["name"] == "flat_pnl" and any(stake not in {0.0, 1.0} for stake in stakes):
                        raise ValueError("flat_pnl placements require stakes of zero or one")
                    profits = []
                    for prefix, record, odds, stake in (
                        ("cand", cand_record, cand_odds, stakes[0]),
                        ("base", base_record, base_odds, stakes[1]),
                    ):
                        team = row.get(f"{prefix}_bet_team")
                        if stake == 0:
                            profits.append(None)
                            continue
                        if team not in odds:
                            raise ValueError(f"invalid {prefix}_bet_team for match {match_id!r}")
                        settled = (settle_flat_policy(team, odds[team], record["actual_winner"], cost)
                                   if a["name"] == "flat_pnl" else
                                   settle_kelly_policy(stake, odds[team], team,
                                                       record["actual_winner"], cost))
                        if settled is None:
                            raise ValueError(f"could not settle placement for match {match_id!r}")
                        profits.append(float(settled))
                    if profits[0] is not None or profits[1] is not None:
                        seed_a.append((profits[0] or 0.0) - (profits[1] or 0.0))
                        cand_pnls.append(profits[0] or 0.0)
                        base_pnls.append(profits[1] or 0.0)
                        cand_stakes.append(stakes[0])
                        base_stakes.append(stakes[1])
                        seed_profit_ids.append(match_id)
                        seed_profit_clusters.append(cluster)
                        for scenario_index, scenario in enumerate(scenario_models):
                            scenario_profits = []
                            for prefix, record, odds, stake in (
                                ("cand", cand_record, cand_odds, stakes[0]),
                                ("base", base_record, base_odds, stakes[1]),
                            ):
                                team = row.get(f"{prefix}_bet_team")
                                if stake == 0:
                                    scenario_profits.append(0.0)
                                else:
                                    value = (settle_flat_policy(team, odds[team], record["actual_winner"], scenario)
                                             if a["name"] == "flat_pnl" else
                                             settle_kelly_policy(stake, odds[team], team,
                                                                 record["actual_winner"], scenario))
                                    scenario_profits.append(float(value))
                            parts = seed_scenario_parts[scenario_index]
                            parts[0].append(scenario_profits[0]); parts[1].append(scenario_profits[1])
                            parts[2].append(stakes[0]); parts[3].append(stakes[1])
                    placement_vectors["candidate"].append((row.get("cand_bet_team"), stakes[0]))
                    placement_vectors["baseline"].append((row.get("base_bet_team"), stakes[1]))
                kept_ids.append(match_id)
                seed_clusters.append(cluster)
            if not kept_ids:
                raise ValueError(f"no metric rows for seed {label}")
            if canonical_ids is None:
                canonical_ids, clusters = kept_ids, seed_clusters
            elif kept_ids != canonical_ids or seed_clusters != clusters:
                raise ValueError("metric match ids/clusters differ across aligned seeds")
            if canonical_profit_ids is None:
                canonical_profit_ids = seed_profit_ids
                profit_clusters = seed_profit_clusters
            elif (seed_profit_ids != canonical_profit_ids
                  or seed_profit_clusters != profit_clusters):
                raise ValueError("profit rows differ across aligned seeds")
            all_ll.append([0.0] * len(kept_ids))
            all_profit.append(seed_a)
            betting_pnl["candidate"].append(cand_pnls)
            betting_pnl["baseline"].append(base_pnls)
            betting_stake["candidate"].append(cand_stakes)
            betting_stake["baseline"].append(base_stakes)
            for index, parts in enumerate(seed_scenario_parts):
                for key, values in zip(scenario_betting[index], parts):
                    scenario_betting[index][key].append(values)
            for arm in ("candidate", "baseline"):
                vector = json.dumps(placement_vectors[arm], separators=(",", ":"))
                if vector in kept_placement_vectors[arm]:
                    raise ValueError(f"duplicate placements across seeds in {arm} arm")
                kept_placement_vectors[arm].add(vector)
                records = (cand_records if arm == "candidate" else base_records)
                prediction_vector = _prediction_vector(records)
                if prediction_vector in kept_prediction_vectors[arm]:
                    raise ValueError(f"duplicate predictions across seeds in {arm} arm")
                kept_prediction_vectors[arm].add(prediction_vector)

    else:
        for label in labels:
            cand_payload = sliced_payloads["candidate"][label]
            base_payload = sliced_payloads["baseline"][label]
            cand_records = cand_payload.get("matches", [])
            base_records = base_payload.get("matches", [])
            cand_ids = [str(row.get("match_id")) for row in cand_records]
            base_ids = [str(row.get("match_id")) for row in base_records]
            if cand_ids != base_ids:
                raise ValueError(f"match id mismatch between arms for seed {label}")
            reliable = reliable and cand_payload.get("summary", {}).get(
                "bootstrap_reliable", True
            ) is not False
            reliable = reliable and base_payload.get("summary", {}).get(
                "bootstrap_reliable", True
            ) is not False

            seed_ll, seed_profit = [], []
            seed_profit_ids, seed_profit_clusters = [], []
            seed_scenarios = [[] for _ in scenario_values]
            kept_ids, seed_clusters = [], []
            for cand, base in zip(cand_records, base_records):
                if (not cand.get("actual_winner") or not base.get("actual_winner")
                        or not cand.get("market_odds") or not base.get("market_odds")):
                    dropped_total += 1
                    continue
                if cand["actual_winner"] != base["actual_winner"]:
                    raise ValueError(
                        f"outcome mismatch for match {cand.get('match_id')!r}"
                    )
                registered = _registered_row(cand, odds_index)
                if registered is None:
                    dropped_total += 1
                    continue
                cand_odds = _check_raw_prices(cand, registered)
                base_odds = _check_raw_prices(base, registered)
                if cand_odds != base_odds:
                    raise ValueError(
                        f"market_odds mismatch for match {cand.get('match_id')!r}"
                    )
                clean = {key: value for key, value in cand.items()
                         if key not in {"competition_cluster_id", "cluster_id"}}
                cluster, resolution = cluster_id_with_resolution(clean, cluster_lookup)
                resolutions[resolution] += 1
                seed_clusters.append(cluster)
                kept_ids.append(str(cand["match_id"]))
                seed_ll.append(_log_loss(cand) - _log_loss(base))
                cand_profit = _profit(cand, cand_odds, cost)
                base_profit = _profit(base, base_odds, cost)
                if cand_profit is not None or base_profit is not None:
                    seed_profit.append((cand_profit or 0.0) - (base_profit or 0.0))
                    seed_profit_ids.append(str(cand["match_id"]))
                    seed_profit_clusters.append(cluster)
                for index, scenario in enumerate(cost_scenarios or DEFAULT_SCENARIOS):
                    cand_scenario = _profit(cand, cand_odds, scenario)
                    base_scenario = _profit(base, base_odds, scenario)
                    if cand_profit is not None or base_profit is not None:
                        seed_scenarios[index].append(
                            (cand_scenario or 0.0) - (base_scenario or 0.0)
                        )
            if not kept_ids:
                raise ValueError(f"no usable paired records for seed {label}")
            kept_positions = [cand_ids.index(match_id) for match_id in kept_ids]
            for arm, records in (("candidate", cand_records), ("baseline", base_records)):
                vector = _prediction_vector([records[index] for index in kept_positions])
                if vector in kept_prediction_vectors[arm]:
                    raise ValueError(f"duplicate predictions across seeds in {arm} arm")
                kept_prediction_vectors[arm].add(vector)
            if canonical_ids is None:
                canonical_ids, clusters = kept_ids, seed_clusters
            elif kept_ids != canonical_ids or seed_clusters != clusters:
                raise ValueError("usable match ids/clusters differ across aligned seeds")
            if canonical_profit_ids is None:
                canonical_profit_ids = seed_profit_ids
                profit_clusters = seed_profit_clusters
            elif (seed_profit_ids != canonical_profit_ids
                  or seed_profit_clusters != profit_clusters):
                raise ValueError("union-of-bets rows differ across aligned seeds")
            all_ll.append(seed_ll)
            all_profit.append(seed_profit)
            for index, values in enumerate(seed_scenarios):
                scenario_values[index].append(values)

    ll_values = np.asarray(all_ll, dtype=float)
    profit_values = np.asarray(all_profit, dtype=float)
    cluster_array = np.asarray(clusters, dtype=str)
    ll = MetricDelta(float(ll_values.mean()), _interval(ll_values, cluster_array, len(labels)))
    profit_cluster_array = np.asarray(profit_clusters, dtype=str)
    if profit_values.shape[1] == 0:
        profit = MetricDelta(0.0, (0.0, 0.0))
    else:
        profit = MetricDelta(float(profit_values.mean()),
                             _interval(profit_values, profit_cluster_array, len(labels)))
    delta_roi = None
    if kind == "betting_layer":
        candidate_pnl = np.asarray(betting_pnl["candidate"], dtype=float)
        baseline_pnl = np.asarray(betting_pnl["baseline"], dtype=float)
        candidate_stake = np.asarray(betting_stake["candidate"], dtype=float)
        baseline_stake = np.asarray(betting_stake["baseline"], dtype=float)
        cand_total_stake = float(candidate_stake.sum())
        base_total_stake = float(baseline_stake.sum())
        point = ((float(candidate_pnl.sum()) / cand_total_stake if cand_total_stake else 0.0)
                 - (float(baseline_pnl.sum()) / base_total_stake if base_total_stake else 0.0))
        interval = ((0.0, 0.0) if candidate_pnl.shape[1] == 0 else
                    bootstrap_ratio_of_sums_difference_ci(
                        candidate_pnl, candidate_stake, baseline_pnl, baseline_stake,
                        profit_cluster_array, DEFAULT_BOOTSTRAP_RESAMPLES, .95,
                        DEFAULT_BOOTSTRAP_SEED,
                    ))
        delta_roi = MetricDelta(point, interval)
    block_count = len(set(clusters))
    profit_block_count = len(set(profit_clusters or []))
    fallback_count = resolutions.get("fallback", 0)
    descriptive = ((not reliable) or block_count < 10 or fallback_count > 0
                   or profit_block_count < 10)
    provisional = len(labels) < 5
    verdict = _classification(
        kind, ll, delta_roi or profit, provisional, descriptive,
        metric_a_favourable=(metric_a_spec or {}).get("favourable"),
    )
    diagnostics = []
    diagnostic_source = (zip(scenario_models, scenario_betting)
                         if kind == "betting_layer"
                         else zip(scenario_models, scenario_values))
    for scenario, values in diagnostic_source:
        if kind == "betting_layer":
            cand_pnl = np.asarray(values["candidate_pnl"], dtype=float)
            base_pnl = np.asarray(values["baseline_pnl"], dtype=float)
            cand_stake = np.asarray(values["candidate_stake"], dtype=float)
            base_stake = np.asarray(values["baseline_stake"], dtype=float)
            differences = cand_pnl - base_pnl
            scenario_delta = (MetricDelta(0.0, (0.0, 0.0))
                              if differences.shape[1] == 0 else
                              MetricDelta(float(differences.mean()),
                                          _interval(differences, profit_cluster_array,
                                                    len(labels))))
            cand_sum, base_sum = float(cand_stake.sum()), float(base_stake.sum())
            roi_point = ((float(cand_pnl.sum()) / cand_sum if cand_sum else 0.0)
                         - (float(base_pnl.sum()) / base_sum if base_sum else 0.0))
            scenario_roi = MetricDelta(
                roi_point,
                ((0.0, 0.0) if cand_pnl.shape[1] == 0 else
                 bootstrap_ratio_of_sums_difference_ci(
                     cand_pnl, cand_stake, base_pnl, base_stake,
                     profit_cluster_array, DEFAULT_BOOTSTRAP_RESAMPLES, .95,
                     DEFAULT_BOOTSTRAP_SEED)),
            )
        else:
            array = np.asarray(values, dtype=float)
            scenario_delta = (
                MetricDelta(0.0, (0.0, 0.0))
                if array.shape[1] == 0 else
                MetricDelta(float(array.mean()),
                            _interval(array, profit_cluster_array, len(labels)))
            )
            scenario_roi = None
        diagnostics.append({
            "cost_model": scenario.as_dict(),
            "delta_profit": asdict(scenario_delta),
            **({"delta_roi": asdict(scenario_roi)} if scenario_roi else {}),
        })
    return Verdict(
        verdict=verdict, kind=kind, seed_count=len(labels), seeds=labels,
        arm_files=arm_files, decision_inputs=decision_inputs,
        estimator=("match_cluster_ci" if len(labels) == 1
                   else "seed_mean_match_cluster_ci"), provisional=provisional,
        cost_model=cost.as_dict(), odds_role=odds_role,
        odds_path=entry["path"], odds_sha256=entry["sha256"],
        cluster_contract={
            "source_dir": str(cluster_source),
            "contract": "tournament_time_block_v1",
            "resolution_counts": dict(sorted(resolutions.items())),
            "fallback_count": fallback_count,
        }, block_count=block_count, n_records=len(canonical_ids),
        n_profit_rows=len(canonical_profit_ids or []),
        profit_block_count=profit_block_count,
        n_dropped_symmetrically=dropped_total,
        delta_log_loss=ll, delta_profit=profit, delta_roi=delta_roi,
        cost_scenario_diagnostics=diagnostics,
        bootstrap={"confidence": 0.95, "resamples": DEFAULT_BOOTSTRAP_RESAMPLES,
                   "seed": DEFAULT_BOOTSTRAP_SEED, "reliable": reliable},
        metric_a=metric_a_spec,
    )


def _record_manual(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="claim_gate.py record-manual")
    parser.add_argument("--kind", required=True, choices=("sim_prop",))
    parser.add_argument("--idea", required=True)
    parser.add_argument("--gate-script", type=Path, required=True)
    parser.add_argument("--detail-json", action="extend", nargs="+", required=True)
    parser.add_argument("--verdict", required=True, choices=sorted(MANUAL_VERDICTS))
    parser.add_argument("--note", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.note.strip():
        parser.error("--note must contain the pre-committed gate pair text")
    script = args.gate_script
    details = [Path(path) for path in args.detail_json]
    for path in [script, *details]:
        if not path.is_file():
            parser.error(f"evidence file does not exist: {path}")
    payload = {
        "gate_mode": "manual_sim_prop",
        "kind": "sim_prop",
        "idea": args.idea,
        "gate_script": {"path": _repo_path(script), "sha256": _sha256(script)},
        "detail_json": [
            {"path": _repo_path(path), "sha256": _sha256(path)} for path in details
        ],
        "verdict": args.verdict,
        "note": args.note.strip(),
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    args.out.write_bytes(encoded)
    print(f"{args.verdict} {hashlib.sha256(encoded).hexdigest()}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(argv) if argv is not None else sys.argv[1:]
    if arguments[:1] == ["record-manual"]:
        return _record_manual(arguments[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="extend", nargs="+")
    parser.add_argument("--baseline", action="extend", nargs="+")
    parser.add_argument("--metrics-json", action="extend", nargs="+")
    parser.add_argument("--kind", required=True,
                        choices=("match_model", "betting_layer"))
    parser.add_argument("--seeds", nargs="+")
    parser.add_argument("--cost-spread-bps", type=float, default=0.0)
    parser.add_argument("--cost-fee-bps", type=float, default=0.0)
    parser.add_argument("--cost-fee-basis", choices=("winnings", "stake"),
                        default="winnings")
    parser.add_argument("--odds-role", required=True)
    parser.add_argument("--cluster-source-dir")
    parser.add_argument("--registry-path", default="docs/registered_odds.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(arguments)
    if args.kind == "match_model" and (
            not args.candidate or not args.baseline or args.metrics_json):
        parser.error("match_model requires --candidate and --baseline only")
    if args.kind == "betting_layer" and (
            not args.candidate or not args.baseline or not args.metrics_json):
        parser.error("betting_layer requires --candidate, --baseline, and --metrics-json")
    verdict = decide(
        args.candidate, args.baseline, args.kind,
        CostModel(args.cost_spread_bps, args.cost_fee_bps, args.cost_fee_basis),
        odds_role=args.odds_role, cluster_source_dir=args.cluster_source_dir,
        seeds=args.seeds, registry_path=args.registry_path,
        metrics_json=args.metrics_json,
    )
    encoded = (json.dumps(verdict.as_dict(), indent=2, sort_keys=True) + "\n").encode()
    args.out.write_bytes(encoded)
    print(f"{verdict.verdict} {hashlib.sha256(encoded).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
