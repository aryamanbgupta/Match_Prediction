"""Shared machinery for the registered T1 simulator PPC runners.

`run_t1_sim_ppc.py` (root cohort + orientation audit), `run_t1_sim_extras_ppc.py`
(B18 extras graft), `run_t1_sim_roster_ppc.py` (latent bowling roster) and
`audit_t1_sim_parity.py` all replay the same same-day chronology through a
`SameDayReplayStatsProvider` and summarize simulated innings the same way.
This module is that shared layer; each runner keeps only its cohort, arms,
gates, and summary contract. Behavior is verbatim from the original runners —
the registered artifacts they produced remain reproducible.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from loaders_common import iter_matches_chronological
from player_metadata import PlayerMetadataProvider
from registered_experiment import REPO_ROOT, reject_sealed, sha256
from sim_eval.loaders import TestMatchLoader
from sim_eval.same_day_stats import SameDayReplayStatsProvider
from sim_v1_2 import MatchState, SimulationConfig, SimulationEngine, T20Rules
from stats_provider import StatsProvider

__all__ = [
    "REPO_ROOT", "reject_sealed", "sha256",
    "phase_of_over", "actual_innings", "sim_result", "result_signature",
    "swap_storage", "simulate", "win_probability", "bootstrap_mean",
    "reliability", "ordinary_test_order_reference", "match_metrics",
    "build_context_by_date", "create_state", "build_t1_stack",
    "replay_selected", "provenance_block",
]

PHASES = ("powerplay", "middle", "death")
INNINGS_FIELDS = ("total_runs", "wickets", "legal_balls", "deliveries",
                  "extras_events", "unique_bowlers")


def phase_of_over(over: int) -> str:
    if over < 6:
        return "powerplay"
    if over < 16:
        return "middle"
    return "death"


def actual_innings(document: dict) -> list[dict]:
    """Observed per-innings totals for the two regulation innings."""
    output = []
    for innings in document.get("innings", [])[:2]:
        total = wickets = legal = deliveries = extras_events = 0
        phase_runs = {p: 0 for p in PHASES}
        bowlers = set()
        for over in innings.get("overs", []):
            over_no = int(over.get("over", 0))
            for delivery in over.get("deliveries", []):
                runs = int(delivery.get("runs", {}).get("total", 0))
                extras = delivery.get("extras", {}) or {}
                is_legal = not extras.get("wides") and not extras.get("noballs")
                total += runs
                wickets += len(delivery.get("wickets", []))
                legal += int(is_legal)
                deliveries += 1
                extras_events += int(not is_legal)
                phase_runs[phase_of_over(over_no)] += runs
                if delivery.get("bowler"):
                    bowlers.add(delivery["bowler"])
        output.append({
            "batting_team": innings.get("team"),
            "total_runs": total,
            "wickets": wickets,
            "legal_balls": legal,
            "deliveries": deliveries,
            "extras_events": extras_events,
            "phase_runs": phase_runs,
            "unique_bowlers": len(bowlers),
        })
    return output


def sim_result(result) -> dict:
    """Durable per-simulation record matching `actual_innings`'s schema."""
    innings = []
    for inn in result.innings:
        phase_runs = {p: 0 for p in PHASES}
        extras_events = 0
        for ball in inn.balls:
            phase_runs[phase_of_over(int(ball.over))] += int(ball.runs)
            extras_events += int(not ball.is_legal)
        innings.append({
            "batting_team": inn.batting_team,
            "total_runs": int(inn.total_runs),
            "wickets": int(inn.total_wickets),
            "legal_balls": int(inn.total_balls),
            "deliveries": len(inn.balls),
            "extras_events": extras_events,
            "phase_runs": phase_runs,
            "unique_bowlers": len(inn.bowling_card),
        })
    return {
        "winner": result.winner,
        "scores": {result.team1: int(result.team1_score),
                   result.team2: int(result.team2_score)},
        "wickets": {result.team1: int(result.team1_wickets),
                    result.team2: int(result.team2_wickets)},
        "innings": innings,
    }


def result_signature(result) -> tuple:
    """Storage-label-invariant signature for exact symmetry comparison."""
    by_team = tuple(sorted(((result.team1, int(result.team1_score),
                            int(result.team1_wickets)),
                           (result.team2, int(result.team2_score),
                            int(result.team2_wickets)))))
    innings = []
    for inn in result.innings:
        balls = tuple((int(ball.over), int(ball.ball), ball.outcome.name,
                       int(ball.runs), int(ball.batter_runs),
                       int(ball.extras_runs), bool(ball.is_legal),
                       int(ball.striker_idx), int(ball.bowler_idx))
                      for ball in inn.balls)
        innings.append((inn.batting_team, inn.bowling_team,
                        int(inn.total_runs), int(inn.total_wickets),
                        int(inn.total_balls), balls))
    return result.winner, by_team, tuple(innings)


def swap_storage(state: MatchState) -> MatchState:
    """Swap team1/team2 storage labels while preserving who bats first."""
    return MatchState(
        team1_lineup=state.team2_lineup,
        team2_lineup=state.team1_lineup,
        batting_first=state.batting_first,
        venue=state.venue,
        match_date=state.match_date,
        toss_winner=state.toss_winner,
        chose_to_bat=state.chose_to_bat,
        match_importance=state.match_importance,
        is_international=state.is_international,
        competition_tier=state.competition_tier,
    )


def simulate(engine, state, n: int, seed: int):
    return engine.simulate_multiple(
        state,
        SimulationConfig(n_simulations=n, parallel=False,
                         random_seed=seed, verbose=False),
    )


def win_probability(rows: list[dict], team: str) -> float:
    wins = sum(row["winner"] == team for row in rows)
    losses = sum(row["winner"] not in {team, "Tie"} for row in rows)
    return wins / (wins + losses) if wins + losses else 0.5


def bootstrap_mean(values, reps: int, seed: int):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    stats = np.empty(reps, dtype=float)
    for i in range(reps):
        stats[i] = rng.choice(values, size=len(values), replace=True).mean()
    return [float(np.percentile(stats, 2.5)),
            float(np.percentile(stats, 97.5))]


def reliability(probs, outcomes, n_bins=4):
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    order = np.argsort(probs)
    bins = []
    for idx in np.array_split(order, n_bins):
        if len(idx):
            bins.append({"n": int(len(idx)),
                         "mean_predicted": float(probs[idx].mean()),
                         "observed": float(outcomes[idx].mean())})
    return bins


def ordinary_test_order_reference(match_dir: Path, reps: int,
                                  seed: int) -> dict:
    """Descriptive order reference from every ordinary evaluation fixture.

    This is not a causal estimate because batting order is not randomized and
    team strength can differ. It is nevertheless the relevant sanity check for
    whether a paired simulator order effect is obviously alien to the cohort.
    """
    order_outcomes = []
    for path in sorted(Path(match_dir).glob("*.json")):
        document = json.load(path.open())
        innings = document.get("innings", [])
        outcome = document.get("info", {}).get("outcome", {})
        winner = outcome.get("winner")
        if len(innings) < 2 or not winner:
            continue
        first_team = innings[0].get("team")
        second_team = innings[1].get("team")
        if winner == first_team:
            order_outcomes.append(-1.0)
        elif winner == second_team:
            order_outcomes.append(1.0)
    if not order_outcomes:
        raise ValueError(f"no decided ordinary fixtures in {match_dir}")
    values = np.asarray(order_outcomes, dtype=float)
    return {
        "n_decided_matches": int(len(values)),
        "first_batting_win_rate": float(np.mean(values < 0)),
        "chasing_win_rate": float(np.mean(values > 0)),
        "chasing_minus_first_batting_win_rate": float(values.mean()),
        "chasing_minus_first_batting_ci95": bootstrap_mean(
            values, reps, seed),
        "interpretation": (
            "descriptive marginal reference, not a controlled causal estimate"
        ),
    }


def match_metrics(matches: list[dict]) -> dict:
    """Winner probabilities plus per-innings posterior means for a raw run.

    Each match row needs `first_batting_team`, `actual_winner`,
    `actual_innings` and a `simulations` list of `sim_result` records.
    """
    probs, outcomes = [], []
    first_bias, first_abs, first_cover = [], [], []
    extras_actual, extras_sim = [], []
    innings_metrics = defaultdict(list)
    phase_metrics = defaultdict(list)
    for match in matches:
        first_team = match["first_batting_team"]
        p = win_probability(match["simulations"], first_team)
        y = float(match["actual_winner"] == first_team)
        probs.append(p)
        outcomes.append(y)
        for innings_idx in (0, 1):
            actual = match["actual_innings"][innings_idx]
            sims = [row["innings"][innings_idx]
                    for row in match["simulations"]
                    if len(row["innings"]) > innings_idx]
            prefix = f"innings_{innings_idx + 1}"
            for field in INNINGS_FIELDS:
                values = np.asarray([row[field] for row in sims], dtype=float)
                innings_metrics[f"{prefix}_{field}_actual"].append(
                    actual[field])
                innings_metrics[f"{prefix}_{field}_posterior_mean"].append(
                    float(values.mean()))
            extras_actual.append(actual["extras_events"])
            extras_sim.append(np.mean([row["extras_events"] for row in sims]))
            for phase in PHASES:
                values = [row["phase_runs"][phase] for row in sims]
                phase_metrics[f"{prefix}_{phase}_actual"].append(
                    actual["phase_runs"][phase])
                phase_metrics[f"{prefix}_{phase}_posterior_mean"].append(
                    float(np.mean(values)))
        actual_total = match["actual_innings"][0]["total_runs"]
        posterior = np.asarray(
            [row["innings"][0]["total_runs"]
             for row in match["simulations"]], dtype=float)
        bias = float(posterior.mean() - actual_total)
        first_bias.append(bias)
        first_abs.append(abs(bias))
        lo, hi = np.percentile(posterior, [10, 90])
        first_cover.append(int(lo <= actual_total <= hi))
    return {
        "probabilities": np.asarray(probs),
        "outcomes": np.asarray(outcomes),
        "first_bias_by_match": np.asarray(first_bias),
        "first_abs_by_match": np.asarray(first_abs),
        "first_cover_by_match": np.asarray(first_cover),
        "extras_actual": np.asarray(extras_actual),
        "extras_sim": np.asarray(extras_sim),
        "innings_metrics": innings_metrics,
        "phase_metrics": phase_metrics,
    }


def build_context_by_date(context_dir, selected_dates) -> dict[str, list]:
    """Full same-day context batches for every date carrying a selected match."""
    selected_dates = set(selected_dates)
    context_by_date = defaultdict(list)
    for match_id, json_text, match_date in iter_matches_chronological(
        context_dir, gender="male"
    ):
        date_str = match_date.date().isoformat()
        if date_str in selected_dates:
            context_by_date[date_str].append(
                (str(match_id), json.loads(json_text)))
    return context_by_date


def create_state(loader, document: dict, match_id: str) -> MatchState:
    """Build a pre-match state, failing loudly on any construction problem."""
    _, state = loader.create_match_state(document, cricsheet_id=match_id)
    if state is None:
        raise RuntimeError(f"could not create state for {match_id}")
    observed_first = document["innings"][0]["team"]
    if state.batting_first != observed_first:
        raise RuntimeError(
            f"batting-order mismatch for {match_id}: "
            f"state={state.batting_first}, observed={observed_first}")
    return state


def build_t1_stack(config: dict, selector, extras_graft=None) -> SimpleNamespace:
    """Provider + T1 model + engine + loader for a PPC config.

    `extras_graft=None` clears any inherited graft env (the production flat
    default); a path opts into the B18 sidecar. Import of the model wrapper is
    deferred so this module stays importable without torch.
    """
    from sim_t1 import TransformerT1SimModel

    metadata = PlayerMetadataProvider(config["data"]["player_metadata"])
    provider = SameDayReplayStatsProvider(
        StatsProvider("models", version=config["data"]["stats_version"]),
        metadata)
    os.environ["T1_SIM_MODEL_DIR"] = config["model"]["checkpoint"]
    os.environ["T1_SIM_DEVICE"] = config["model"]["device"]
    if extras_graft:
        os.environ["T1_EXTRAS_GRAFT_PATH"] = str(extras_graft)
    else:
        os.environ.pop("T1_EXTRAS_GRAFT_PATH", None)
    model = TransformerT1SimModel(
        stats_provider=provider, player_metadata=metadata)
    return SimpleNamespace(
        metadata=metadata,
        provider=provider,
        model=model,
        selector=selector,
        engine=SimulationEngine(model, T20Rules(selector)),
        loader=TestMatchLoader(),
    )


def replay_selected(context_by_date: dict, provider, selected_ids,
                    on_match) -> tuple[int, int]:
    """Chronological same-day replay: predict on selected, advance the rest.

    Calls `on_match(match_id, document, date_str)` for each selected match
    between `begin_match` and `lock_prediction`/`advance_match`. Returns
    (selected done, context matches replayed) and verifies completeness.
    """
    selected_ids = set(selected_ids)
    done = 0
    context = 0
    seen = set()
    for date_str in sorted(context_by_date):
        batch = context_by_date[date_str]
        provider.begin_date(date_str, [doc for _, doc in batch])
        for match_id, document in batch:
            active = match_id in selected_ids
            provider.begin_match(
                match_id, document, prediction_required=active)
            if not active:
                provider.advance_match(match_id, document)
                context += 1
                continue
            on_match(match_id, document, date_str)
            provider.lock_prediction(match_id)
            provider.advance_match(match_id, document)
            seen.add(match_id)
            done += 1
    if done != len(selected_ids):
        missing = sorted(selected_ids - seen)
        raise RuntimeError(f"selected matches absent from chronology: {missing}")
    return done, context


def provenance_block(config_path: Path, config: dict) -> dict:
    """Standard PPC provenance: config, checkpoint, sidecars, engine source.

    Sidecar entries appear whenever the config declares them, so every
    runner's summary carries the same provenance surface.
    """
    block = {
        "config_sha256": sha256(config_path),
        "model_checkpoint": config["model"]["checkpoint"],
        "model_sha256": sha256(
            Path(config["model"]["checkpoint"]) / "model.pt"),
    }
    extras_graft = config.get("model", {}).get("extras_graft")
    if extras_graft:
        block["extras_graft_sha256"] = sha256(extras_graft)
    bowler_usage = config.get("simulation", {}).get("bowler_usage")
    if bowler_usage:
        block["bowler_usage_sha256"] = sha256(bowler_usage)
    roster_policy = config.get("simulation", {}).get("bowler_roster_policy")
    if roster_policy:
        block["bowler_roster_policy_sha256"] = sha256(roster_policy)
    block["simulation_engine_sha256"] = sha256(
        REPO_ROOT / "scripts" / "sim_v1_2.py")
    return block
