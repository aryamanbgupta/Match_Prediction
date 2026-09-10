#!/usr/bin/env python3
"""Select protocol-scored lines and write an immutable daily score artifact."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from sim_eval.eval_statistics import (  # noqa: E402
    BOOTSTRAP_CONTRACT_VERSION, DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED, MIN_RECOMMENDED_CLUSTERS, bootstrap_mean_ci,
    cluster_id_for_record, flat_bet_team, load_competition_clusters,
    settle_flat_policy,
)
from sim_eval.market_math import CostModel, implied_probs  # noqa: E402


def _ts(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def select_scored_line(lines: list[dict], cohort_id: str, fixture_id: str):
    """Pure protocol selector; returns (selected-or-None, exclusion counts)."""
    counts = Counter()
    eligible = []
    for line in lines:
        if line.get("cohort_id") != cohort_id or line.get("fixture_id") != fixture_id:
            continue
        if line.get("run_kind") != "t60":
            counts["not_t60"] += 1
            continue
        try:
            start, attempt = _ts(line["scheduled_start"]), _ts(line["attempt_ts"])
        except (KeyError, TypeError, ValueError):
            counts["invalid_timestamp"] += 1
            continue
        lower, upper = start - timedelta(minutes=75), start - timedelta(minutes=45)
        if not lower <= attempt <= upper:
            counts["attempt_outside_window"] += 1
            continue
        quote_value, quote_stamp = line.get("quote"), line.get("quote_ts")
        if quote_value is None or quote_stamp is None:
            counts["missing_quote"] += 1
            continue
        try:
            quote_ts = _ts(quote_stamp)
        except (TypeError, ValueError):
            counts["invalid_quote_ts"] += 1
            continue
        if quote_ts > attempt:
            counts["quote_after_attempt"] += 1
            continue
        if not lower <= quote_ts <= upper:
            counts["quote_outside_window"] += 1
            continue
        if attempt >= start:
            counts["attempt_not_prematch"] += 1
            continue
        eligible.append(line)
    if not eligible:
        return None, dict(counts)
    latest = max(_ts(line["attempt_ts"]) for line in eligible)
    winners = [line for line in eligible if _ts(line["attempt_ts"]) == latest]
    if len(winners) != 1:
        raise ValueError("duplicate maximum attempt_ts; failing closed")
    counts["eligible"] = len(eligible)
    counts["selected"] = 1
    return winners[0], dict(counts)


def highest_settlements(lines: list[dict]) -> dict[str, dict]:
    highest_revision: dict[str, int] = {}
    for line in lines:
        key, revision = line["fixture_id"], int(line["revision"])
        highest_revision[key] = max(highest_revision.get(key, revision), revision)
    grouped: dict[str, dict] = {}
    for line in lines:
        key = line["fixture_id"]
        if int(line["revision"]) != highest_revision[key]:
            continue
        if key in grouped and line != grouped[key]:
            raise ValueError(f"conflicting settlement revision for {key}")
        grouped[key] = line
    return grouped


def evaluator_record(prediction: dict, settlement: dict, cost: CostModel,
                     cluster_lookup=None) -> dict:
    teams = [prediction["team1"], prediction["team2"]]
    p1 = float(prediction["p_team1"])
    quote = prediction.get("quote")
    odds = ({team: 1.0 / float(quote[team]) for team in teams} if quote else {})
    market_p = implied_probs(odds, remove_margin=True) if odds else {}
    edge = ({teams[0]: p1 - market_p[teams[0]],
             teams[1]: 1 - p1 - market_p[teams[1]]} if market_p else {})
    model_probs = {teams[0]: p1, teams[1]: 1 - p1}
    winner = settlement.get("winner")
    won_team1 = winner == teams[0]
    actual_p = p1 if won_team1 else 1 - p1
    base = {
        "match_id": settlement.get("cricsheet_id") or prediction["fixture_id"],
        "cricsheet_id": settlement.get("cricsheet_id"),
        "fixture_id": prediction["fixture_id"], "teams": teams,
        "team1": teams[0], "team2": teams[1], "predicted_prob": p1,
        "simulated_win_prob": model_probs,
        "actual_winner": winner, "market_odds": odds,
        "market_win_prob": market_p, "market_implied_prob": market_p,
        "edge": edge,
        "log_loss": -math.log(min(max(actual_p, 1e-15), 1 - 1e-15)),
        "brier_score": (p1 - float(won_team1)) ** 2,
    }
    base["competition_cluster_id"] = cluster_id_for_record(base, cluster_lookup)
    base["bet_team"] = flat_bet_team({**base, "realized_pnl": 0.0})
    base["bet_placed"] = base["bet_team"] is not None
    selected_odds = odds.get(base["bet_team"], 2.0)
    base["realized_pnl"] = settle_flat_policy(
        base["bet_team"], selected_odds, base["actual_winner"], cost
    )
    return base


def score(predictions: list[dict], settlements: list[dict], cost: CostModel,
          cluster_lookup=None) -> dict:
    settled = highest_settlements(settlements)
    keys = sorted({(row.get("cohort_id"), row.get("fixture_id")) for row in predictions
                   if row.get("cohort_id") == "daily_v1"})
    records, selection_counts, selected_total, quote_coverage = [], Counter(), 0, 0
    for cohort, fixture_id in keys:
        selected, counts = select_scored_line(predictions, cohort, fixture_id)
        selection_counts.update(counts)
        if selected is None:
            continue
        selected_total += 1
        outcome = settled.get(fixture_id)
        if outcome is None or outcome.get("void"):
            continue
        if selected.get("quote") is not None:
            quote_coverage += 1
        records.append(evaluator_record(selected, outcome, cost, cluster_lookup))
    losses, market_losses = [], []
    pnl, clusters = [], []
    for row in records:
        won = row["actual_winner"] == row["team1"]
        probability = row["predicted_prob"] if won else 1 - row["predicted_prob"]
        losses.append(-math.log(min(max(probability, 1e-15), 1 - 1e-15)))
        if row["market_win_prob"]:
            market_probability = row["market_win_prob"][row["actual_winner"]]
            market_losses.append(-math.log(min(max(market_probability, 1e-15), 1 - 1e-15)))
        if row["bet_placed"]:
            pnl.append(float(row["realized_pnl"]))
            clusters.append(row["competition_cluster_id"])
    blocks = len(set(row["competition_cluster_id"] for row in records))
    return {
        "protocol_version": 1, "cohort_id": "daily_v1",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "descriptive" if blocks < MIN_RECOMMENDED_CLUSTERS else "inferential",
        "primary_metric": "log_loss", "n_scored": len(records),
        "log_loss": sum(losses) / len(losses) if losses else None,
        "log_loss_ci95": bootstrap_mean_ci(
            losses, n_resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
            seed=DEFAULT_BOOTSTRAP_SEED,
            clusters=[row["competition_cluster_id"] for row in records],
        ) if losses else [None, None],
        "market_log_loss": (sum(market_losses) / len(market_losses)
                            if market_losses else None),
        "flat_return_mean": sum(pnl) / len(pnl) if pnl else None,
        "flat_return_ci95": bootstrap_mean_ci(
            pnl, n_resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
            seed=DEFAULT_BOOTSTRAP_SEED, clusters=clusters,
        ) if pnl else [None, None],
        "bootstrap": {"contract": BOOTSTRAP_CONTRACT_VERSION,
                      "resamples": DEFAULT_BOOTSTRAP_RESAMPLES,
                      "seed": DEFAULT_BOOTSTRAP_SEED, "block_count": blocks},
        "cost_model": cost.as_dict(), "price_basis": "polymarket_probability",
        "coverage": {"fixtures": len(keys), "selected": selected_total,
                     "market_quoted": quote_coverage,
                     "missing_quote": selection_counts.get("missing_quote", 0)},
        "void_count": sum(bool(row.get("void")) for row in settled.values()),
        "selection_counts": dict(selection_counts), "records": records,
    }


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--settlements", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--source-dir", type=Path, action="append", default=[])
    p.add_argument("--spread-bps", type=float, default=0)
    p.add_argument("--fee-bps", type=float, default=0)
    p.add_argument("--fee-basis", choices=("winnings", "stake"), default="winnings")
    a = p.parse_args(argv)
    cluster_lookup = {}
    for source in a.source_dir:
        cluster_lookup.update(load_competition_clusters(source))
    artifact = score(_jsonl(a.predictions), _jsonl(a.settlements),
                     CostModel(a.spread_bps, a.fee_bps, a.fee_basis),
                     cluster_lookup or None)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    name = "score_v1_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json"
    path = a.out_dir / name
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
