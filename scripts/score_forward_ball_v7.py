#!/usr/bin/env python3
"""Create locked ball-v7 winner predictions with same-day replay.

The frozen-protocol preflight runs before importing joblib, the simulation
engine, or the mutable replay provider.  On the current DRAFT protocol this
command must fail without loading or scoring a model.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

ROOT = Path(__file__).resolve().parent.parent

from forward_eval_contract import (  # noqa: E402
    load_protocol,
    preflight,
    repo_path,
)
from identity_maps import canonicalize_venue  # noqa: E402
from score_forward_match_m7 import (  # noqa: E402
    _assert_outcome_free,
    _candidate_artifacts,
    ordered_holdout_rows,
    write_locked_artifact,
)


SCHEMA_VERSION = 1
PROBABILITY_FLOOR = 0.05
WINNER_POSTPROCESS = "exclude_ties_clip_0.05_0.95_renormalize_v1"
LINEUP_CONTRACT = "info_players_roster_order_only_v1"


def validate_candidate_contract(candidate: Mapping[str, Any]) -> None:
    expected = {
        "bowler_selector": "empirical",
        "ball_calibration": "vector",
        "parallel_simulation": False,
        "lineup_contract": LINEUP_CONTRACT,
        "winner_probability_postprocess": WINNER_POSTPROCESS,
    }
    mismatches = {
        key: (candidate.get(key), value)
        for key, value in expected.items()
        if candidate.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"ball-v7 protocol recipe is unsupported: {mismatches}"
        )
    if (
        not isinstance(candidate.get("n_simulations"), int)
        or candidate["n_simulations"] <= 0
    ):
        raise RuntimeError("ball-v7 n_simulations must be a positive integer")
    if not isinstance(candidate.get("random_seed"), int):
        raise RuntimeError("ball-v7 random_seed must be an integer")


def pre_match_spec(match_data: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return simulation inputs from ``info`` only."""
    info = match_data["info"]
    dates = info.get("dates") or []
    teams = list(info.get("teams") or [])
    if not dates:
        raise ValueError("match is missing info.dates[0]")
    if len(teams) != 2 or len(set(map(str, teams))) != 2:
        raise ValueError(f"match must contain exactly two distinct teams: {teams}")

    date = str(dates[0])
    datetime.strptime(date, "%Y-%m-%d")
    registry = info.get("registry", {}).get("people", {})
    rosters = info.get("players") or {}
    if set(rosters) != set(teams):
        raise ValueError(
            f"info.players teams differ from info.teams: "
            f"{sorted(rosters)} != {sorted(teams)}"
        )

    players: dict[str, list[dict[str, str]]] = {}
    for team in teams:
        names = list(rosters[team])
        if len(names) < 11:
            raise ValueError(
                f"pre-match roster for {team} has {len(names)} players; "
                "at least 11 are required"
            )
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate player name in {team} roster")
        missing = [name for name in names if name not in registry]
        if missing:
            raise ValueError(
                f"{team} roster names missing from registry: {missing}"
            )
        resolved = [
            {"player_id": str(registry[name]), "name": str(name)}
            for name in names
        ]
        ids = [row["player_id"] for row in resolved]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate player ID in {team} roster")
        players[str(team)] = resolved

    venue = canonicalize_venue(info.get("venue"))
    toss = info.get("toss") or {}
    toss_winner = str(toss.get("winner", ""))
    toss_decision = str(toss.get("decision", ""))
    if toss_winner not in teams:
        raise ValueError(f"toss winner is not one of the teams: {toss_winner}")
    if toss_decision not in {"bat", "field"}:
        raise ValueError(f"unsupported toss decision: {toss_decision}")
    batting_first = (
        toss_winner
        if toss_decision == "bat"
        else next(team for team in teams if team != toss_winner)
    )

    event = info.get("event") or {}
    event_name = event.get("name", "") if isinstance(event, dict) else ""
    return {
        "date": date,
        "teams": list(map(str, teams)),
        "venue": venue,
        "players": players,
        "toss_winner": toss_winner,
        "toss_decision": toss_decision,
        "batting_first": str(batting_first),
        "event_name": str(event_name),
        "team_type": str(info.get("team_type", "unknown")),
    }


def build_match_state_from_info(
    match_data: Mapping[str, Any],
    *,
    player_class,
    lineup_class,
    state_class,
    classify_context: Callable[[str, str, list[str]], dict[str, int]],
):
    """Build a simulation state without consulting innings or outcomes."""
    spec = pre_match_spec(match_data)
    lineups = []
    for team in spec["teams"]:
        players = [
            player_class(row["player_id"], row["name"], team)
            for row in spec["players"][team]
        ]
        lineups.append(lineup_class(team, players))

    context = classify_context(
        spec["event_name"],
        spec["team_type"],
        spec["teams"],
    )
    return state_class(
        team1_lineup=lineups[0],
        team2_lineup=lineups[1],
        batting_first=spec["batting_first"],
        venue=spec["venue"],
        match_date=datetime.strptime(spec["date"], "%Y-%m-%d"),
        toss_winner=spec["toss_winner"],
        chose_to_bat=1 if spec["toss_decision"] == "bat" else 0,
        match_importance=context["match_importance"],
        is_international=context["is_international"],
        competition_tier=context["competition_tier"],
    )


def selected_rows_by_cricsheet(
    protocol: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    rows = ordered_holdout_rows(protocol)
    result = {str(row["cricsheet_id"]): row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError("duplicate selected Cricsheet ID")
    return result


def load_context_batches(
    protocol: dict[str, Any],
) -> list[tuple[str, list[tuple[str, dict[str, Any]]]]]:
    """Load the sealed context in versioned date/Cricsheet-ID order."""
    holdout_dir = repo_path(protocol["holdout"]["directory"])
    context_dir = holdout_dir / "context_t20s_json"
    entries: list[tuple[str, str, dict[str, Any]]] = []
    seen: set[str] = set()
    for path in context_dir.glob("*.json"):
        match_id = path.stem
        if match_id in seen:
            raise RuntimeError(f"duplicate context match ID: {match_id}")
        seen.add(match_id)
        data = json.loads(path.read_text())
        spec = pre_match_spec(data)
        entries.append((spec["date"], match_id, data))

    entries.sort(key=lambda row: (row[0], row[1]))
    if not entries:
        raise RuntimeError("forward context is empty")

    selected = selected_rows_by_cricsheet(protocol)
    missing = sorted(set(selected) - seen)
    if missing:
        raise RuntimeError(
            f"selected fixtures missing from forward context: {missing[:20]}"
        )
    batches: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for date, match_id, data in entries:
        batches[date].append((match_id, data))
    return [(date, batches[date]) for date in sorted(batches)]


def validate_selected_identity(
    manifest_row: Mapping[str, Any],
    cricsheet_id: str,
    match_data: Mapping[str, Any],
) -> dict[str, Any]:
    spec = pre_match_spec(match_data)
    expected = (
        str(manifest_row["cricsheet_id"]),
        str(manifest_row["date"]),
        tuple(map(str, manifest_row["teams"])),
        str(manifest_row["venue"]),
    )
    actual = (
        str(cricsheet_id),
        spec["date"],
        tuple(spec["teams"]),
        spec["venue"],
    )
    if actual != expected:
        raise RuntimeError(
            f"selected fixture identity/team order mismatch: "
            f"{actual!r} != {expected!r}"
        )
    return spec


def postprocess_winner_probabilities(
    p_team1_raw: float,
    p_team2_raw: float,
    p_tie_raw: float,
) -> dict[str, float]:
    """Mirror the landed evaluator's tie exclusion and 5–95% clipping."""
    values = [float(p_team1_raw), float(p_team2_raw), float(p_tie_raw)]
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise RuntimeError(f"invalid raw simulation probabilities: {values}")
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(
            f"raw simulation probabilities do not sum to one: {values}"
        )

    resolved = values[0] + values[1]
    if resolved > 0.0:
        p_team1 = values[0] / resolved
        p_team2 = values[1] / resolved
    else:
        p_team1 = p_team2 = 0.5
    p_team1 = min(1.0 - PROBABILITY_FLOOR, max(PROBABILITY_FLOOR, p_team1))
    p_team2 = min(1.0 - PROBABILITY_FLOOR, max(PROBABILITY_FLOOR, p_team2))
    total = p_team1 + p_team2
    return {
        "p_team1": p_team1 / total,
        "p_team2": p_team2 / total,
        "p_team1_raw": values[0],
        "p_team2_raw": values[1],
        "p_tie_raw": values[2],
    }


def simulate_winner(
    engine,
    match_state,
    *,
    simulation_config_class,
    result_aggregator,
    n_simulations: int,
    random_seed: int,
) -> dict[str, Any]:
    config = simulation_config_class(
        n_simulations=n_simulations,
        parallel=False,
        random_seed=random_seed,
        verbose=False,
    )
    results = engine.simulate_multiple(match_state, config)
    aggregate = result_aggregator.aggregate(results)
    team1, team2 = match_state.team1, match_state.team2
    raw = aggregate["win_probability"]
    probabilities = postprocess_winner_probabilities(
        raw[team1],
        raw[team2],
        raw["tie"],
    )
    return {
        **probabilities,
        "n_simulations": int(aggregate["n_simulations"]),
    }


def walk_context_and_score(
    batches: Iterable[tuple[str, list[tuple[str, dict[str, Any]]]]],
    selected: Mapping[str, Mapping[str, Any]],
    stats_provider,
    *,
    build_state: Callable[[Mapping[str, Any]], Any],
    simulate: Callable[[Any], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Score selected fixtures, then advance every context fixture."""
    predictions: list[dict[str, Any]] = []
    context_count = 0
    selected_seen: set[str] = set()

    for date, batch in batches:
        stats_provider.begin_date(date, [data for _, data in batch])
        for cricsheet_id, match_data in batch:
            context_count += 1
            manifest_row = selected.get(str(cricsheet_id))
            prediction_required = manifest_row is not None
            stats_provider.begin_match(
                str(cricsheet_id),
                match_data,
                prediction_required=prediction_required,
            )
            if prediction_required:
                spec = validate_selected_identity(
                    manifest_row,
                    str(cricsheet_id),
                    match_data,
                )
                state = build_state(match_data)
                simulation = dict(simulate(state))
                required = {
                    "p_team1",
                    "p_team2",
                    "p_team1_raw",
                    "p_team2_raw",
                    "p_tie_raw",
                    "n_simulations",
                }
                if set(simulation) != required:
                    raise RuntimeError(
                        f"simulation output keys differ from contract: "
                        f"{sorted(simulation)}"
                    )
                row = {
                    "match_id": str(manifest_row["match_id"]),
                    "cricsheet_id": str(cricsheet_id),
                    "date": spec["date"],
                    "team1": spec["teams"][0],
                    "team2": spec["teams"][1],
                    **simulation,
                }
                _assert_outcome_free(row)
                predictions.append(row)
                selected_seen.add(str(cricsheet_id))
                # This guard must happen after simulation and before replay.
                stats_provider.lock_prediction(str(cricsheet_id))

            stats_provider.advance_match(str(cricsheet_id), match_data)

    missing = sorted(set(map(str, selected)) - selected_seen)
    if missing:
        raise RuntimeError(f"selected fixtures were not scored: {missing[:20]}")
    if len(predictions) != len(selected):
        raise RuntimeError("ball-v7 prediction count differs from selection")
    return predictions, {
        "context_matches_replayed": context_count,
        "selected_matches_scored": len(predictions),
    }


def build_prediction_artifact(
    protocol: dict[str, Any],
    preflight_report: dict[str, Any],
    predictions: list[dict[str, Any]],
    replay_report: dict[str, int],
) -> dict[str, Any]:
    candidate = protocol["candidates"]["ball_v7"]
    validate_candidate_contract(candidate)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "locked_outcome_free_predictions",
        "model_id": "ball_v7",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": preflight_report["protocol_sha256"],
        "holdout_fingerprint_sha256": preflight_report[
            "holdout_fingerprint_sha256"
        ],
        "state_fingerprint_sha256": preflight_report[
            "state_fingerprint_sha256"
        ],
        "prediction_count": len(predictions),
        "probability_semantics": "p_team1_wins",
        "winner_probability_postprocess": WINNER_POSTPROCESS,
        "lineup_contract": LINEUP_CONTRACT,
        "same_day_order_version": protocol["state"][
            "same_day_order_version"
        ],
        "n_simulations": int(candidate["n_simulations"]),
        "random_seed": int(candidate["random_seed"]),
        "parallel_simulation": False,
        "ball_calibration": str(candidate["ball_calibration"]),
        "bowler_selector": str(candidate["bowler_selector"]),
        "outcome_columns_loaded": False,
        "outcomes_joined": False,
        "completed_match_replay_after_prediction": True,
        "replay_report": replay_report,
        "model_artifacts": candidate["artifacts"],
        "predictions": predictions,
    }
    _assert_outcome_free(artifact)
    return artifact


def score(protocol_path: Path, output_path: Path) -> dict[str, Any]:
    """Score ball v7 only after frozen authorization; never join outcomes."""
    gate = preflight(protocol_path, require_frozen=True)
    protocol = load_protocol(protocol_path)
    candidate = protocol["candidates"]["ball_v7"]
    validate_candidate_contract(candidate)

    # The fail-closed gate intentionally precedes all model/simulation imports.
    import joblib

    from parsing_v2 import classify_match_context
    from player_metadata import PlayerMetadataProvider
    from sim_eval.same_day_stats import SameDayReplayStatsProvider
    from sim_v1_2 import (
        EmpiricalBowlerSelector,
        MatchState,
        Player,
        ResultAggregator,
        SimulationConfig,
        SimulationEngine,
        T20Rules,
        TeamLineup,
        XGBoostModelV2,
    )
    from stats_provider import StatsProvider

    artifacts = _candidate_artifacts(protocol, "ball_v7")
    state_dir = repo_path(protocol["state"]["directory"])
    base_stats = StatsProvider(
        str(state_dir),
        version="v3",
        require_order_contract=True,
    )
    player_metadata = PlayerMetadataProvider(
        str(ROOT / "data" / "all_players_enriched.csv")
    )
    replay_stats = SameDayReplayStatsProvider(
        base_stats,
        player_metadata,
    )
    calibrator = joblib.load(
        repo_path(artifacts["vector_scaling_calibrator_v1.pkl"]["path"])
    )
    model = XGBoostModelV2(
        model_path=str(repo_path(artifacts["xgboost_model_v3.pkl"]["path"])),
        batter_encoder_path=str(
            repo_path(artifacts["batter_encoder_v3.pkl"]["path"])
        ),
        bowler_encoder_path=str(
            repo_path(artifacts["bowler_encoder_v3.pkl"]["path"])
        ),
        feature_columns_path=str(
            repo_path(artifacts["feature_columns_v3.txt"]["path"])
        ),
        stats_provider=replay_stats,
        player_metadata=player_metadata,
        matchup_encoder_path=str(
            repo_path(artifacts["matchup_encoder_v3.pkl"]["path"])
        ),
        ball_calibrator=calibrator,
        venue_encoder_path=str(
            repo_path(artifacts["venue_encoder_v3.pkl"]["path"])
        ),
    )
    selector = EmpiricalBowlerSelector(
        usage_path=str(
            repo_path(artifacts["bowler_phase_usage.json"]["path"])
        )
    )
    engine = SimulationEngine(model, T20Rules(selector))

    def _build_state(match_data):
        return build_match_state_from_info(
            match_data,
            player_class=Player,
            lineup_class=TeamLineup,
            state_class=MatchState,
            classify_context=classify_match_context,
        )

    def _simulate(match_state):
        return simulate_winner(
            engine,
            match_state,
            simulation_config_class=SimulationConfig,
            result_aggregator=ResultAggregator,
            n_simulations=int(candidate["n_simulations"]),
            random_seed=int(candidate["random_seed"]),
        )

    selected = selected_rows_by_cricsheet(protocol)
    predictions, replay_report = walk_context_and_score(
        load_context_batches(protocol),
        selected,
        replay_stats,
        build_state=_build_state,
        simulate=_simulate,
    )
    artifact = build_prediction_artifact(
        protocol,
        gate,
        predictions,
        replay_report,
    )
    digest = write_locked_artifact(output_path, artifact)
    return {
        "status": "LOCKED",
        "model_id": "ball_v7",
        "prediction_count": len(predictions),
        "context_matches_replayed": replay_report[
            "context_matches_replayed"
        ],
        "output": str(output_path.resolve()),
        "sha256": digest,
        "outcomes_joined": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(score(args.protocol, args.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
