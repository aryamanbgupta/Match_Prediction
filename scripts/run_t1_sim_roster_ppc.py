#!/usr/bin/env python3
"""Isolated causal latent bowling-roster PPC for the corrected T1 sim.

Consumes the B18 extras cohort; shared replay/summary machinery lives in
`t1_ppc_common.py`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from t1_ppc_common import (actual_innings, bootstrap_mean,  # noqa: E402
                           build_context_by_date, build_t1_stack,
                           create_state, match_metrics,
                           ordinary_test_order_reference, provenance_block,
                           reject_sealed, replay_selected, result_signature,
                           sim_result, simulate, swap_storage,
                           win_probability)
from sim_v1_2 import RosterEmpiricalBowlerSelector  # noqa: E402

DEFAULT_CONFIG = Path(
    "experiments/configs/t1_sim_roster_bowling_ppc_v1.yaml")
ENGINE_CONTRACT = "first_over_selector_v1"


def _baseline_matches(raw: dict) -> list[dict]:
    return [{
        "match_id": match["match_id"],
        "first_batting_team": match["first_batting_team"],
        "actual_winner": match["actual_winner"],
        "actual_innings": match["actual_innings"],
        "simulations": match["simulations"],
    } for match in raw["matches"]]


def _unique_by_match(matches: list[dict], innings_index: int) -> tuple:
    actual, posterior = [], []
    for match in matches:
        actual.append(match["actual_innings"][innings_index]["unique_bowlers"])
        values = [row["innings"][innings_index]["unique_bowlers"]
                  for row in match["simulations"]
                  if len(row["innings"]) > innings_index]
        posterior.append(float(np.mean(values)))
    return np.asarray(actual, dtype=float), np.asarray(posterior, dtype=float)


def summarize(candidate_raw: dict, baseline_raw: dict, config: dict) -> dict:
    candidate_matches = candidate_raw["matches"]
    baseline_matches = _baseline_matches(baseline_raw)
    candidate = match_metrics(candidate_matches)
    baseline = match_metrics(baseline_matches)
    reps = int(config["statistics"]["bootstrap_repetitions"])
    seed = int(config["statistics"]["bootstrap_seed"])
    unique_summary, unique_passes = {}, []
    for innings_index in (0, 1):
        actual, candidate_unique = _unique_by_match(
            candidate_matches, innings_index)
        _, baseline_unique = _unique_by_match(baseline_matches, innings_index)
        error = candidate_unique - actual
        base_error = baseline_unique - actual
        passed = abs(float(error.mean())) <= 0.5
        unique_passes.append(passed)
        unique_summary[f"innings_{innings_index + 1}"] = {
            "actual": float(actual.mean()),
            "b18_b10_baseline": float(baseline_unique.mean()),
            "roster_candidate": float(candidate_unique.mean()),
            "candidate_minus_actual": float(error.mean()),
            "candidate_minus_baseline_delta": {
                "mean": float((error - base_error).mean()),
                "ci95": bootstrap_mean(error - base_error, reps, seed),
            },
            "gate_within_0.5": "PASS" if passed else "FAIL",
        }

    y = candidate["outcomes"]
    p, bp = candidate["probabilities"], baseline["probabilities"]
    ll = -(y * np.log(np.clip(p, 1e-15, 1))
           + (1-y) * np.log(np.clip(1-p, 1e-15, 1)))
    base_ll = -(y * np.log(np.clip(bp, 1e-15, 1))
                + (1-y) * np.log(np.clip(1-bp, 1e-15, 1)))
    ll_delta_ci = bootstrap_mean(ll - base_ll, reps, seed)
    comparison_valid = bool(
        candidate_raw.get("engine_contract")
        and candidate_raw.get("engine_contract")
        == baseline_raw.get("engine_contract"))
    score_mae_delta = (candidate["first_abs_by_match"].mean()
                       - baseline["first_abs_by_match"].mean())
    symmetry_pass = candidate_raw["symmetry_mismatches"] == 0
    mae_guard = score_mae_delta <= 2.0
    ll_guard = ll_delta_ci[0] <= 0.0
    flips_complete = all("flipped_order" in match
                         for match in candidate_matches)
    order_result = None
    order_pass = False
    if flips_complete:
        paired = np.asarray([
            win_probability(match["simulations"], match["first_batting_team"])
            - win_probability(match["flipped_order"],
                              match["first_batting_team"])
            for match in candidate_matches
        ])
        reference = ordinary_test_order_reference(
            Path(config["data"]["match_dir"]), reps, seed)
        ci = reference["chasing_minus_first_batting_ci95"]
        order_pass = ci[0] <= -float(paired.mean()) <= ci[1]
        order_result = {
            "paired_batting_first_minus_chasing": {
                "mean": float(paired.mean()),
                "ci95": bootstrap_mean(paired, reps, seed),
                "per_match": paired.tolist(),
            },
            "ordinary_test_order_reference": reference,
        }
    gates_except_order = (comparison_valid and all(unique_passes)
                          and symmetry_pass and mae_guard and ll_guard)
    if gates_except_order and order_pass:
        status = "BOWLING_PPC_PASS"
    elif gates_except_order and not flips_complete:
        # Every completed gate passes; only the flipped-order arm is
        # outstanding (--append-flipped). Not a pass, but not a failure.
        status = "PENDING_FLIPPED_ARM"
    else:
        status = "PPC_FAIL"
    return {
        "contract": "t1_sim_roster_bowling_ppc_v1",
        "status": status,
        "gates": {
            "unique_bowlers_each_innings_within_0.5": (
                "PASS" if all(unique_passes) else "FAIL"),
            "storage_label_symmetry": "PASS" if symmetry_pass else "FAIL",
            "paired_order_effect_reference_compatible": (
                "PASS" if order_pass else
                "PENDING" if not flips_complete else "FAIL"),
            "first_innings_mae_guard_plus_2_runs": (
                "PASS" if mae_guard else "FAIL"),
            "winner_ll_no_ci_clean_regression": (
                "PASS" if ll_guard else "FAIL"),
        },
        "n_matches": len(candidate_matches),
        "n_sims": candidate_raw["n_sims"],
        "calibration_applied": False,
        "baseline_comparison_valid": comparison_valid,
        "unique_bowlers": unique_summary,
        "symmetry": {
            "comparisons": candidate_raw["symmetry_comparisons"],
            "mismatches": candidate_raw["symmetry_mismatches"],
        },
        "first_innings_total": {
            "baseline_bias": float(baseline["first_bias_by_match"].mean()),
            "candidate_bias": float(candidate["first_bias_by_match"].mean()),
            "baseline_mae": float(baseline["first_abs_by_match"].mean()),
            "candidate_mae": float(candidate["first_abs_by_match"].mean()),
            "candidate_minus_baseline_mae": float(score_mae_delta),
            "baseline_p10_p90_coverage": float(
                baseline["first_cover_by_match"].mean()),
            "candidate_p10_p90_coverage": float(
                candidate["first_cover_by_match"].mean()),
        },
        "winner": {
            "baseline_log_loss": float(base_ll.mean()),
            "candidate_log_loss": float(ll.mean()),
            "candidate_minus_baseline_log_loss": {
                "mean": float((ll - base_ll).mean()),
                "ci95": ll_delta_ci,
            },
            "baseline_brier": float(np.mean((bp-y)**2)),
            "candidate_brier": float(np.mean((p-y)**2)),
            **(order_result or {}),
        },
        "innings_metrics": {
            key: float(np.mean(values))
            for key, values in candidate["innings_metrics"].items()
        },
        "phase_metrics": {
            key: float(np.mean(values))
            for key, values in candidate["phase_metrics"].items()
        },
    }


def _write_summary(raw: dict, baseline: dict, config: dict,
                   config_path: Path) -> dict:
    summary = summarize(raw, baseline, config)
    summary["elapsed_seconds"] = raw["elapsed_seconds"]
    summary["provenance"] = provenance_block(config_path, config)
    path = Path(config["outputs"]["summary"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def _require_engine_contract(raw: dict, raw_path, message: str) -> None:
    # Verify, never backfill: restamping would validate a raw produced on a
    # different engine (see run_t1_sim_extras_ppc for the full rationale).
    stored = raw.get("engine_contract")
    if stored != ENGINE_CONTRACT:
        raise SystemExit(
            f"{raw_path} carries engine_contract={stored!r}; {message} "
            f"expects {ENGINE_CONTRACT!r}. Re-run the simulations under "
            "the current engine instead of restamping."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--aggregate-only", action="store_true")
    mode.add_argument(
        "--append-flipped", action="store_true",
        help="append the 200-draw flipped-order arm to an existing raw run")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    for value in (config["data"]["baseline_raw"],
                  config["data"]["match_dir"],
                  config["data"]["context_dir"],
                  config["model"]["extras_graft"],
                  config["simulation"]["bowler_roster_policy"]):
        reject_sealed(value)
    baseline = json.load(open(config["data"]["baseline_raw"]))
    if args.aggregate_only:
        raw = json.load(open(config["outputs"]["raw"]))
        _require_engine_contract(raw, config["outputs"]["raw"], "aggregation")
        _write_summary(raw, baseline, config, args.config)
        return

    selected = set(baseline.get("selected_match_ids", ()))
    if not selected:
        selected = {match["match_id"] for match in baseline["matches"]}
    documents = {
        path.stem: json.load(path.open())
        for path in Path(config["data"]["match_dir"]).glob("*.json")
        if path.stem in selected
    }
    selected_dates = {str(document["info"]["dates"][0])
                      for document in documents.values()}
    context_by_date = build_context_by_date(
        config["data"]["context_dir"], selected_dates)

    selector = RosterEmpiricalBowlerSelector(
        config["simulation"]["bowler_usage"],
        config["simulation"]["bowler_roster_policy"])
    stack = build_t1_stack(
        config, selector, extras_graft=config["model"]["extras_graft"])
    n_sims = int(config["simulation"]["n_sims"])
    symmetry_sims = int(config["simulation"]["symmetry_sims"])
    seed = int(config["simulation"]["random_seed"])
    baseline_by_id = {match["match_id"]: match
                      for match in baseline["matches"]}
    if args.append_flipped:
        raw = json.load(open(config["outputs"]["raw"]))
        # Appending flipped sims mixes new simulations into an existing raw:
        # the stored raw must already carry the current engine's stamp.
        _require_engine_contract(
            raw, config["outputs"]["raw"], "--append-flipped")
        raw_by_id = {match["match_id"]: match for match in raw["matches"]}
        flipped_started = time.time()
    else:
        raw = {
            "contract": "t1_sim_roster_bowling_ppc_v1",
            "config": str(args.config), "sealed_data_used": False,
            "n_sims": n_sims, "selected_match_ids": sorted(selected),
            "matches": [], "symmetry_comparisons": 0,
            "symmetry_mismatches": 0, "started_at_unix": time.time(),
            "engine_contract": ENGINE_CONTRACT,
        }

    progress = {"done": 0}

    def on_match(match_id: str, document: dict, date_text: str) -> None:
        state = create_state(stack.loader, document, match_id)
        progress["done"] += 1
        if args.append_flipped:
            first_team = document["innings"][0]["team"]
            flipped = state.copy()
            flipped.batting_first = (
                state.team2 if first_team == state.team1 else state.team1)
            results = simulate(stack.engine, flipped, n_sims, seed)
            raw_by_id[match_id]["flipped_order"] = [
                sim_result(result) for result in results]
            print(f"[{progress['done']}/{len(selected)}] {match_id} flipped "
                  f"p_same_team={win_probability(raw_by_id[match_id]['flipped_order'], first_team):.3f}",
                  flush=True)
            return
        results = simulate(stack.engine, state, n_sims, seed)
        swapped = simulate(
            stack.engine, swap_storage(state), symmetry_sims, seed)
        for original, swapped_result in zip(
            results[:symmetry_sims], swapped
        ):
            raw["symmetry_comparisons"] += 1
            raw["symmetry_mismatches"] += int(
                result_signature(original)
                != result_signature(swapped_result))
        base = baseline_by_id[match_id]
        row = {
            "match_id": match_id, "date": date_text,
            "first_batting_team": document["innings"][0]["team"],
            "actual_winner": base["actual_winner"],
            "actual_innings": actual_innings(document),
            "simulations": [sim_result(result) for result in results],
        }
        raw["matches"].append(row)
        unique = [np.mean([sim["innings"][index]["unique_bowlers"]
                           for sim in row["simulations"]])
                  for index in (0, 1)]
        print(f"[{progress['done']}/{len(selected)}] {match_id} "
              f"p_first={win_probability(row['simulations'], row['first_batting_team']):.3f} "
              f"unique_bowlers={unique[0]:.3f}/{unique[1]:.3f} "
              f"symmetry_mismatch={raw['symmetry_mismatches']}",
              flush=True)

    replay_selected(context_by_date, stack.provider, selected, on_match)
    if args.append_flipped:
        raw["flipped_elapsed_seconds"] = time.time() - flipped_started
    else:
        raw["elapsed_seconds"] = time.time() - raw["started_at_unix"]
    raw_path = Path(config["outputs"]["raw"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(raw, indent=2) + "\n")
    summary = _write_summary(raw, baseline, config, args.config)
    if summary["status"] != "BOWLING_PPC_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
