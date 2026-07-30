#!/usr/bin/env python3
"""Join locked forward predictions to outcomes and produce the frozen report.

This is deliberately separate from both model scorers.  It runs only after
the protocol is FROZEN, verifies each prediction artifact and checksum, then
joins Polymarket odds and realized results for the first time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

ROOT = Path(__file__).resolve().parent.parent

from forward_eval_contract import (  # noqa: E402
    load_protocol,
    preflight,
    repo_path,
)
from score_forward_match_m7 import (  # noqa: E402
    _assert_outcome_free,
    ordered_holdout_rows,
)
from sim_eval.eval_statistics import (  # noqa: E402
    BOOTSTRAP_CONTRACT_VERSION,
    bootstrap_mean_ci,
    cluster_id_for_record,
    load_competition_clusters,
)


SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
PROBABILITY_EPSILON = 1e-15
RELIABILITY_BINS = 10


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(128 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_locked_prediction_artifact(
    path: Path,
    *,
    expected_model_id: str,
    protocol: Mapping[str, Any],
    preflight_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the write-once checksum and protocol binding before joining."""
    path = path.resolve()
    digest_path = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not digest_path.is_file():
        raise FileNotFoundError(
            f"prediction artifact or checksum is missing: {path}"
        )
    actual_digest = _sha256_file(path)
    parts = digest_path.read_text().strip().split()
    if len(parts) != 2:
        raise RuntimeError(f"malformed prediction checksum: {digest_path}")
    if parts[0] != actual_digest or parts[1] != path.name:
        raise RuntimeError(f"prediction checksum mismatch: {path}")

    artifact = json.loads(path.read_text())
    _assert_outcome_free(artifact)
    expected_header = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "locked_outcome_free_predictions",
        "model_id": expected_model_id,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": preflight_report["protocol_sha256"],
        "holdout_fingerprint_sha256": preflight_report[
            "holdout_fingerprint_sha256"
        ],
        "state_fingerprint_sha256": preflight_report[
            "state_fingerprint_sha256"
        ],
        "outcomes_joined": False,
    }
    mismatches = {
        key: (artifact.get(key), expected)
        for key, expected in expected_header.items()
        if artifact.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            f"{expected_model_id} prediction header mismatch: {mismatches}"
        )
    predictions = artifact.get("predictions")
    if not isinstance(predictions, list):
        raise RuntimeError(f"{expected_model_id} predictions must be a list")
    if artifact.get("prediction_count") != len(predictions):
        raise RuntimeError(
            f"{expected_model_id} prediction count/header mismatch"
        )
    return artifact


def _prediction_lookup(
    artifact: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    rows = artifact["predictions"]
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        match_id = str(row.get("match_id", ""))
        if not match_id or match_id in result:
            raise RuntimeError(
                f"missing/duplicate prediction match_id: {match_id!r}"
            )
        p1 = float(row["p_team1"])
        p2 = float(row["p_team2"])
        if (
            not math.isfinite(p1)
            or not math.isfinite(p2)
            or p1 < 0.0
            or p2 < 0.0
            or not math.isclose(p1 + p2, 1.0, abs_tol=1e-9)
        ):
            raise RuntimeError(f"invalid prediction probabilities: {match_id}")
        result[match_id] = row
    return result


def _market_probabilities(
    odds_row: Mapping[str, Any],
    team1: str,
    team2: str,
) -> tuple[dict[str, float], dict[str, float]]:
    winner = (odds_row.get("odds") or {}).get("winner") or {}
    try:
        decimal = {
            team1: float(winner[team1]),
            team2: float(winner[team2]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"missing/invalid exact-team odds for {team1} vs {team2}"
        ) from exc
    if any(
        not math.isfinite(value) or value < 1.0
        for value in decimal.values()
    ):
        raise RuntimeError(
            f"invalid decimal odds for {team1} vs {team2}: {decimal}"
        )
    inverse = {team: 1.0 / value for team, value in decimal.items()}
    overround = sum(inverse.values())
    if overround <= 0.0:
        raise RuntimeError(f"zero implied-probability sum: {team1} vs {team2}")
    fair = {team: value / overround for team, value in inverse.items()}
    return decimal, fair


def _validate_prediction_identity(
    row: Mapping[str, Any],
    manifest_row: Mapping[str, Any],
    model_id: str,
) -> None:
    expected = (
        str(manifest_row["match_id"]),
        str(manifest_row["cricsheet_id"]),
        str(manifest_row["date"]),
        tuple(map(str, manifest_row["teams"])),
    )
    actual = (
        str(row.get("match_id")),
        str(row.get("cricsheet_id")),
        str(row.get("date")),
        (str(row.get("team1")), str(row.get("team2"))),
    )
    if actual != expected:
        raise RuntimeError(
            f"{model_id} fixture identity/team order mismatch: "
            f"{actual!r} != {expected!r}"
        )
    if manifest_row.get("display_match_id") and (
        str(row.get("display_match_id"))
        != str(manifest_row.get("display_match_id"))
        or str(row.get("match_identity_version"))
        != str(manifest_row.get("match_identity_version"))
    ):
        raise RuntimeError(
            f"{model_id} match-identity contract mismatch"
        )


def join_evaluation_rows(
    protocol: Mapping[str, Any],
    m7_artifact: Mapping[str, Any],
    ball_artifact: Mapping[str, Any],
    odds_document: Mapping[str, Any],
    cluster_lookup: Optional[Mapping[str, str]] = None,
    manifest_rows: Optional[Iterable[Mapping[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """Join predictions, odds, and outcomes with exact identities only."""
    source_manifest = (
        list(manifest_rows)
        if manifest_rows is not None
        else ordered_holdout_rows(dict(protocol))
    )
    ordered_manifest = sorted(
        source_manifest,
        key=lambda row: (str(row["date"]), str(row["cricsheet_id"])),
    )
    expected_count = int(protocol["holdout"]["selected_matches"])
    if len(ordered_manifest) != expected_count:
        raise RuntimeError("manifest count differs from protocol")

    m7 = _prediction_lookup(m7_artifact)
    ball = _prediction_lookup(ball_artifact)
    odds_rows = list(odds_document.get("matches") or [])
    odds: dict[str, Mapping[str, Any]] = {}
    for row in odds_rows:
        match_id = str(row.get("match_id", ""))
        if not match_id or match_id in odds:
            raise RuntimeError(f"missing/duplicate odds match_id: {match_id!r}")
        odds[match_id] = row

    expected_ids = {str(row["match_id"]) for row in ordered_manifest}
    for label, lookup in (("match_m7", m7), ("ball_v7", ball), ("odds", odds)):
        if set(lookup) != expected_ids:
            missing = sorted(expected_ids - set(lookup))
            extra = sorted(set(lookup) - expected_ids)
            raise RuntimeError(
                f"{label} fixture set mismatch: "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )

    joined: list[dict[str, Any]] = []
    for manifest_row in ordered_manifest:
        match_id = str(manifest_row["match_id"])
        teams = list(map(str, manifest_row["teams"]))
        m7_row = m7[match_id]
        ball_row = ball[match_id]
        odds_row = odds[match_id]
        _validate_prediction_identity(m7_row, manifest_row, "match_m7")
        _validate_prediction_identity(ball_row, manifest_row, "ball_v7")

        odds_identity = (
            str(odds_row.get("match_id")),
            str(odds_row.get("date")),
            (str(odds_row.get("team1")), str(odds_row.get("team2"))),
        )
        expected_odds_identity = (
            match_id,
            str(manifest_row["date"]),
            tuple(teams),
        )
        if odds_identity != expected_odds_identity:
            raise RuntimeError(
                f"odds identity/team order mismatch: "
                f"{odds_identity!r} != {expected_odds_identity!r}"
            )
        if manifest_row.get("display_match_id") and (
            str(odds_row.get("cricsheet_id"))
            != str(manifest_row.get("cricsheet_id"))
            or str(odds_row.get("display_match_id"))
            != str(manifest_row.get("display_match_id"))
            or str(odds_row.get("match_identity_version"))
            != str(manifest_row.get("match_identity_version"))
        ):
            raise RuntimeError("odds match-identity contract mismatch")

        decimal_odds, market_probability = _market_probabilities(
            odds_row,
            teams[0],
            teams[1],
        )
        actual_winner = odds_row.get("actual_winner")
        result_status = (
            "resolved" if actual_winner in teams else "unresolved"
        )
        volume = float(odds_row.get("polymarket_volume_usd") or 0.0)
        if not math.isfinite(volume) or volume < 0.0:
            raise RuntimeError(f"invalid Polymarket volume for {match_id}")
        manifest_volume = float(manifest_row.get("volume_usd") or 0.0)
        if not math.isclose(volume, manifest_volume, abs_tol=1e-6):
            raise RuntimeError(f"manifest/odds volume mismatch for {match_id}")

        elo_diff = float(m7_row["top6_batting_elo_diff"])
        if not math.isfinite(elo_diff):
            raise RuntimeError(f"non-finite M7 ELO difference for {match_id}")
        record = {
            "match_id": match_id,
            "cricsheet_id": str(manifest_row["cricsheet_id"]),
            **({
                "display_match_id": str(manifest_row["display_match_id"]),
                "match_identity_version": str(
                    manifest_row["match_identity_version"]
                ),
            } if manifest_row.get("display_match_id") else {}),
            "date": str(manifest_row["date"]),
            "teams": teams,
            "team1": teams[0],
            "team2": teams[1],
            "venue": str(manifest_row["venue"]),
            "competition": str(manifest_row.get("competition") or ""),
            "polymarket_volume_usd": volume,
            "decimal_odds": decimal_odds,
            "market_probability": market_probability,
            "actual_winner": (
                str(actual_winner) if result_status == "resolved" else None
            ),
            "result_status": result_status,
            "top6_batting_elo_diff": elo_diff,
            "predictions": {
                "match_m7": {
                    teams[0]: float(m7_row["p_team1"]),
                    teams[1]: float(m7_row["p_team2"]),
                },
                "ball_v7": {
                    teams[0]: float(ball_row["p_team1"]),
                    teams[1]: float(ball_row["p_team2"]),
                },
            },
        }
        record["competition_cluster_id"] = cluster_id_for_record(
            record,
            cluster_lookup,
        )
        joined.append(record)
    return joined


def _probability_for_field(
    row: Mapping[str, Any],
    field: str,
) -> Mapping[str, float]:
    current: Any = row
    for part in field.split("."):
        current = current[part]
    return current


def _metric_values(
    rows: Iterable[Mapping[str, Any]],
    probability_field: str,
) -> tuple[list[float], list[float], list[str]]:
    log_losses, briers, clusters = [], [], []
    for row in rows:
        if row["result_status"] != "resolved":
            continue
        team1, team2 = row["teams"]
        probability = _probability_for_field(row, probability_field)
        p_team1 = float(probability[team1])
        p_actual = float(probability[row["actual_winner"]])
        p_actual = min(
            1.0 - PROBABILITY_EPSILON,
            max(PROBABILITY_EPSILON, p_actual),
        )
        actual_team1 = 1.0 if row["actual_winner"] == team1 else 0.0
        log_losses.append(-math.log(p_actual))
        briers.append((p_team1 - actual_team1) ** 2)
        clusters.append(str(row["competition_cluster_id"]))
    return log_losses, briers, clusters


def _ci(
    values: list[float],
    clusters: list[str],
    *,
    resamples: int,
    seed: int,
) -> list[Optional[float]]:
    low, high = bootstrap_mean_ci(
        values,
        n_resamples=resamples,
        seed=seed,
        clusters=clusters,
    )
    return [
        None if math.isnan(low) else low,
        None if math.isnan(high) else high,
    ]


def reliability_table(
    rows: Iterable[Mapping[str, Any]],
    probability_field: str,
) -> list[dict[str, Any]]:
    buckets: list[list[tuple[float, float]]] = [
        [] for _ in range(RELIABILITY_BINS)
    ]
    for row in rows:
        if row["result_status"] != "resolved":
            continue
        team1 = row["team1"]
        probability = _probability_for_field(row, probability_field)
        p_team1 = float(probability[team1])
        index = min(int(p_team1 * RELIABILITY_BINS), RELIABILITY_BINS - 1)
        actual = 1.0 if row["actual_winner"] == team1 else 0.0
        buckets[index].append((p_team1, actual))
    result = []
    for index, values in enumerate(buckets):
        count = len(values)
        result.append({
            "lower": index / RELIABILITY_BINS,
            "upper": (index + 1) / RELIABILITY_BINS,
            "count": count,
            "mean_predicted": (
                sum(value[0] for value in values) / count if count else None
            ),
            "observed_team1_win_rate": (
                sum(value[1] for value in values) / count if count else None
            ),
        })
    return result


def probability_summary(
    rows: list[Mapping[str, Any]],
    probability_field: str,
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    log_losses, briers, clusters = _metric_values(rows, probability_field)
    return {
        "n_evaluable": len(log_losses),
        "binary_log_loss": (
            sum(log_losses) / len(log_losses) if log_losses else None
        ),
        "binary_log_loss_ci95": _ci(
            log_losses,
            clusters,
            resamples=resamples,
            seed=seed,
        ),
        "brier_score": sum(briers) / len(briers) if briers else None,
        "brier_score_ci95": _ci(
            briers,
            clusters,
            resamples=resamples,
            seed=seed,
        ),
        "n_bootstrap_clusters": len(set(clusters)),
        "reliability": reliability_table(rows, probability_field),
    }


def _bet_decision(
    row: Mapping[str, Any],
    model_id: str,
    policy: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    if row["result_status"] != "resolved":
        return None
    teams = row["teams"]
    model = row["predictions"][model_id]
    market = row["market_probability"]
    edges = {team: float(model[team]) - float(market[team]) for team in teams}
    bet_team = max(teams, key=lambda team: edges[team])
    policy_name = str(policy["name"])
    if policy_name == "flat":
        threshold = float(policy["minimum_edge"])
    elif policy_name == "a7":
        mismatch = (
            abs(float(row["top6_batting_elo_diff"]))
            > float(policy["elo_boundary"])
        )
        threshold = float(
            policy[
                "mismatch_minimum_edge"
                if mismatch else "close_minimum_edge"
            ]
        )
    else:
        raise RuntimeError(f"unsupported betting policy: {policy_name}")
    if not edges[bet_team] > threshold:
        return None

    odds = float(row["decimal_odds"][bet_team])
    won = bet_team == row["actual_winner"]
    stake = float(policy["stake_units"])
    pnl = stake * (odds - 1.0) if won else -stake
    return {
        "match_id": row["match_id"],
        "bet_team": bet_team,
        "edge": edges[bet_team],
        "threshold": threshold,
        "odds": odds,
        "won": won,
        "pnl": pnl,
        "competition_cluster_id": row["competition_cluster_id"],
    }


def betting_summary(
    rows: list[Mapping[str, Any]],
    model_id: str,
    policy: Mapping[str, Any],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    bets = [
        decision
        for row in rows
        if (decision := _bet_decision(row, model_id, policy)) is not None
    ]
    pnls = [float(bet["pnl"]) for bet in bets]
    clusters = [str(bet["competition_cluster_id"]) for bet in bets]
    total_pnl = sum(pnls)
    n_bets = len(bets)
    roi = total_pnl / n_bets if n_bets else 0.0
    if pnls:
        cumulative = []
        running = 0.0
        for pnl in pnls:
            running += pnl
            cumulative.append(running)
        peak = []
        running_peak = cumulative[0]
        for value in cumulative:
            running_peak = max(running_peak, value)
            peak.append(running_peak)
        max_drawdown = max(
            high - value for high, value in zip(peak, cumulative)
        )
    else:
        max_drawdown = 0.0
    return {
        "n_bets": n_bets,
        "total_pnl": total_pnl,
        "roi": roi,
        "roi_pct": roi * 100.0,
        "roi_ci95": [
            value * 100.0 if value is not None else None
            for value in _ci(
                pnls,
                clusters,
                resamples=resamples,
                seed=seed,
            )
        ],
        "win_rate": (
            sum(bool(bet["won"]) for bet in bets) / n_bets
            if n_bets else 0.0
        ),
        "max_drawdown_units": max_drawdown,
        "n_bootstrap_clusters": len(set(clusters)),
        "bets": bets,
    }


def _policy_documents(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    policies = protocol["betting_policies"]
    flat = {"name": "flat", **policies["flat"]}
    a7 = {"name": "a7", **policies["a7"]}
    if (
        flat.get("edge_comparison") != "strictly_greater"
        or a7.get("edge_comparison") != "strictly_greater"
    ):
        raise RuntimeError("only strictly_greater edge comparison is supported")
    return [flat, a7]


def build_evaluation_report(
    protocol: Mapping[str, Any],
    preflight_report: Mapping[str, Any],
    rows: list[dict[str, Any]],
    *,
    prediction_artifact_sha256: Mapping[str, str],
) -> dict[str, Any]:
    stats = protocol["statistics"]
    resamples = int(stats["bootstrap_resamples"])
    seed = int(stats["bootstrap_seed"])
    thresholds = {
        "all": 0.0,
        "min_volume_50000": 50_000.0,
        "min_volume_100000": 100_000.0,
    }
    if list(protocol["reporting"]["slices"]) != list(thresholds):
        raise RuntimeError("reporting slices differ from implemented contract")

    slices: dict[str, Any] = {}
    policies = _policy_documents(protocol)
    for name, minimum_volume in thresholds.items():
        subset = [
            row
            for row in rows
            if row["polymarket_volume_usd"] >= minimum_volume
        ]
        expected = int(protocol["holdout"]["liquidity_slices"][name])
        if len(subset) != expected:
            raise RuntimeError(
                f"{name} live row count differs from protocol: "
                f"{len(subset)} != {expected}"
            )
        model_summaries = {}
        for model_id in ("match_m7", "ball_v7"):
            model_summaries[model_id] = {
                "probability": probability_summary(
                    subset,
                    "predictions." + model_id,
                    resamples=resamples,
                    seed=seed,
                ),
                "betting": {
                    policy["name"]: betting_summary(
                        subset,
                        model_id,
                        policy,
                        resamples=resamples,
                        seed=seed,
                    )
                    for policy in policies
                },
            }
        # probability_summary accepts a field name; expose nested predictions
        # through temporary top-level aliases without duplicating report rows.
        slices[name] = {
            "minimum_volume_usd": minimum_volume,
            "n_matches": len(subset),
            "n_evaluable": sum(
                row["result_status"] == "resolved" for row in subset
            ),
            "n_unresolved": sum(
                row["result_status"] != "resolved" for row in subset
            ),
            "market": probability_summary(
                subset,
                "market_probability",
                resamples=resamples,
                seed=seed,
            ),
            "models": model_summaries,
        }

    decision = _decision_assessment(protocol, slices)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_type": "forward_evaluation_report",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": preflight_report["protocol_sha256"],
        "holdout_fingerprint_sha256": preflight_report[
            "holdout_fingerprint_sha256"
        ],
        "prediction_artifact_sha256": dict(prediction_artifact_sha256),
        "bootstrap_contract": BOOTSTRAP_CONTRACT_VERSION,
        "bootstrap_seed": seed,
        "bootstrap_resamples": resamples,
        "outcomes_joined": True,
        "slices": slices,
        "decision_assessment": decision,
        "matches": rows,
    }


def _decision_assessment(
    protocol: Mapping[str, Any],
    slices: Mapping[str, Any],
) -> dict[str, Any]:
    decision = protocol["decision"]
    primary = slices[decision["primary_slice"]]
    m7_ll = primary["models"]["match_m7"]["probability"]["binary_log_loss"]
    ball_ll = primary["models"]["ball_v7"]["probability"]["binary_log_loss"]
    market_ll = primary["market"]["binary_log_loss"]
    probability_confirmed = (
        m7_ll is not None
        and ball_ll is not None
        and market_ll is not None
        and m7_ll
        <= market_ll + float(decision["probability_market_log_loss_tolerance"])
        and (
            not decision["probability_requires_better_than_ball_v7"]
            or m7_ll < ball_ll
        )
    )

    policy_name = str(decision["economic_policy"])
    economic = primary["models"]["match_m7"]["betting"][policy_name]
    minimum_clusters = int(decision["economic_requires_minimum_clusters"])
    ci_low = economic["roi_ci95"][0]
    economic_confirmed = (
        economic["n_bootstrap_clusters"] >= minimum_clusters
        and (
            not decision["economic_requires_positive_ci_lower_bound"]
            or (ci_low is not None and ci_low > 0.0)
        )
    )
    return {
        "primary_slice": decision["primary_slice"],
        "probability_confirmed": probability_confirmed,
        "economic_confirmed": economic_confirmed,
        "economic_policy": policy_name,
        "economic_cluster_requirement_met": (
            economic["n_bootstrap_clusters"] >= minimum_clusters
        ),
        "minimum_confirmatory_clusters": minimum_clusters,
    }


def write_locked_report(path: Path, report: Mapping[str, Any]) -> str:
    path = path.resolve()
    digest_path = path.with_suffix(path.suffix + ".sha256")
    if path.exists() or digest_path.exists():
        raise FileExistsError(f"evaluation report already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    digest = hashlib.sha256(payload).hexdigest()
    with path.open("xb") as handle:
        handle.write(payload)
    try:
        with digest_path.open("x") as handle:
            handle.write(f"{digest}  {path.name}\n")
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return digest


def evaluate(
    protocol_path: Path,
    match_predictions_path: Path,
    ball_predictions_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Run the first outcome join only after frozen authorization."""
    gate = preflight(protocol_path, require_frozen=True)
    protocol = load_protocol(protocol_path)
    m7 = load_locked_prediction_artifact(
        match_predictions_path,
        expected_model_id="match_m7",
        protocol=protocol,
        preflight_report=gate,
    )
    ball = load_locked_prediction_artifact(
        ball_predictions_path,
        expected_model_id="ball_v7",
        protocol=protocol,
        preflight_report=gate,
    )
    holdout_dir = repo_path(protocol["holdout"]["directory"])
    odds_document = json.loads((holdout_dir / "betting_odds.json").read_text())
    cluster_lookup = load_competition_clusters(
        holdout_dir / "context_t20s_json"
    )
    rows = join_evaluation_rows(
        protocol,
        m7,
        ball,
        odds_document,
        cluster_lookup,
    )
    report = build_evaluation_report(
        protocol,
        gate,
        rows,
        prediction_artifact_sha256={
            "match_m7": _sha256_file(match_predictions_path),
            "ball_v7": _sha256_file(ball_predictions_path),
        },
    )
    digest = write_locked_report(output_path, report)
    return {
        "status": "LOCKED",
        "output": str(output_path.resolve()),
        "sha256": digest,
        "n_matches": len(rows),
        "outcomes_joined": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--match-predictions", type=Path, required=True)
    parser.add_argument("--ball-predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(
        evaluate(
            args.protocol,
            args.match_predictions,
            args.ball_predictions,
            args.out,
        ),
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
