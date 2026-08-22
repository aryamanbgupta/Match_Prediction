#!/usr/bin/env python3
"""Isolated B18 empirical-extras posterior-predictive check for T1.

Consumes the cohort frozen by `run_t1_sim_ppc.py`; shared replay/summary
machinery lives in `t1_ppc_common.py`. The registered comparison is the v2
config (its flat baseline is produced by `--flat-only` under the corrected
first-over engine).
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
                           create_state, match_metrics, provenance_block,
                           reject_sealed, reliability, replay_selected,
                           sim_result, simulate, win_probability)
from sim_v1_2 import EmpiricalBowlerSelector  # noqa: E402

ENGINE_CONTRACT = "first_over_selector_v1"

# Compatibility aliases (run_t1_sim_roster_ppc + focused suites import these).
_match_metrics = match_metrics
_reject_sealed = reject_sealed


def _summary(candidate_raw: dict, baseline_raw: dict, config: dict) -> dict:
    candidate = match_metrics(candidate_raw["matches"])
    baseline_matches = []
    for match in baseline_raw["matches"]:
        simulations = match.get("simulations", match.get("actual_order"))
        if simulations is None:
            raise ValueError(
                "baseline match has neither simulations nor actual_order: "
                f"{match.get('match_id', '<unknown>')}")
        baseline_matches.append({
            "match_id": match["match_id"],
            "first_batting_team": match["first_batting_team"],
            "actual_winner": match["actual_winner"],
            "actual_innings": match["actual_innings"],
            "simulations": simulations,
        })
    baseline = match_metrics(baseline_matches)
    reps = int(config["statistics"]["bootstrap_repetitions"])
    seed = int(config["statistics"]["bootstrap_seed"])

    cand_extras = candidate["extras_sim"] - candidate["extras_actual"]
    base_extras = baseline["extras_sim"] - baseline["extras_actual"]
    p = candidate["probabilities"]
    y = candidate["outcomes"]
    cand_ll = -(y * np.log(np.clip(p, 1e-15, 1))
                + (1-y) * np.log(np.clip(1-p, 1e-15, 1)))
    bp = baseline["probabilities"]
    base_ll = -(y * np.log(np.clip(bp, 1e-15, 1))
                + (1-y) * np.log(np.clip(1-bp, 1e-15, 1)))

    extras_gate = abs(candidate["extras_sim"].mean()
                      - candidate["extras_actual"].mean()) <= 1.0
    bias_improves = abs(candidate["first_bias_by_match"].mean()) \
        < abs(baseline["first_bias_by_match"].mean())
    mae_improves = candidate["first_abs_by_match"].mean() \
        < baseline["first_abs_by_match"].mean()
    comparison_valid = bool(
        candidate_raw.get("engine_contract")
        and candidate_raw.get("engine_contract")
        == baseline_raw.get("engine_contract"))
    if comparison_valid:
        status = ("EXTRAS_PPC_PASS"
                  if extras_gate and bias_improves and mae_improves
                  else "PPC_FAIL")
    else:
        status = ("ABSOLUTE_EXTRAS_PPC_PASS_COMPARISON_PENDING"
                  if extras_gate else "PPC_FAIL")
    return {
        "contract": "t1_sim_extras_graft_ppc_v1",
        "status": status,
        "gates": {
            "extras_events_within_1_per_innings": (
                "PASS" if extras_gate else "FAIL"),
            "absolute_first_innings_bias_improves": (
                "PASS" if comparison_valid and bias_improves else
                "FAIL" if comparison_valid else "NOT_COMPARABLE"),
            "first_innings_mae_improves_context": (
                "PASS" if comparison_valid and mae_improves else
                "FAIL" if comparison_valid else "NOT_COMPARABLE"),
        },
        "baseline_comparison_valid": comparison_valid,
        "n_matches": len(candidate_raw["matches"]),
        "n_sims": candidate_raw["n_sims"],
        "calibration_applied": False,
        "extras_events_per_innings": {
            "actual": float(candidate["extras_actual"].mean()),
            "flat_baseline": float(baseline["extras_sim"].mean()),
            "empirical_graft": float(candidate["extras_sim"].mean()),
            "candidate_minus_actual": float(cand_extras.mean()),
            "candidate_minus_flat_delta": {
                "mean": float((cand_extras - base_extras).mean()),
                "ci95": bootstrap_mean(cand_extras - base_extras,
                                       reps, seed),
            },
        },
        "first_innings_total": {
            "flat_bias": float(baseline["first_bias_by_match"].mean()),
            "candidate_bias": float(candidate["first_bias_by_match"].mean()),
            "candidate_minus_flat_bias_delta": {
                "mean": float((candidate["first_bias_by_match"]
                               - baseline["first_bias_by_match"]).mean()),
                "ci95": bootstrap_mean(
                    candidate["first_bias_by_match"]
                    - baseline["first_bias_by_match"], reps, seed),
            },
            "flat_mae": float(baseline["first_abs_by_match"].mean()),
            "candidate_mae": float(candidate["first_abs_by_match"].mean()),
            "flat_p10_p90_coverage": float(
                baseline["first_cover_by_match"].mean()),
            "candidate_p10_p90_coverage": float(
                candidate["first_cover_by_match"].mean()),
        },
        "winner": {
            "flat_log_loss": float(base_ll.mean()),
            "candidate_log_loss": float(cand_ll.mean()),
            "candidate_minus_flat_log_loss": {
                "mean": float((cand_ll - base_ll).mean()),
                "ci95": bootstrap_mean(cand_ll - base_ll, reps, seed),
            },
            "flat_brier": float(np.mean((bp-y)**2)),
            "candidate_brier": float(np.mean((p-y)**2)),
            "candidate_reliability": reliability(p, y),
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


def _require_engine_contract(raw: dict, raw_path, action: str) -> None:
    # Provenance is written ONCE at generation and only VERIFIED here:
    # backfilling the stamp would let a raw produced on a different engine
    # acquire baseline_comparison_valid=true after the fact — exactly the
    # mismatched-engine comparison this contract prevents.
    stored = raw.get("engine_contract")
    if stored != ENGINE_CONTRACT:
        raise SystemExit(
            f"{raw_path} carries engine_contract={stored!r}; {action} "
            f"expects {ENGINE_CONTRACT!r}. Re-run the simulations under "
            "the current engine instead of restamping."
        )


def _finalize(raw: dict, baseline_raw: dict, config: dict,
              config_path: Path) -> None:
    summary = _summary(raw, baseline_raw, config)
    summary["elapsed_seconds"] = raw["elapsed_seconds"]
    summary["provenance"] = provenance_block(config_path, config)
    summary_path = Path(config["outputs"]["summary"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", type=Path,
        # v2 is the registered comparison (v1's baseline predates the
        # corrected engine and degrades to NOT_COMPARABLE).
        default=Path("experiments/configs/"
                     "t1_sim_extras_graft_ppc_v2.yaml"))
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--aggregate-only", action="store_true",
        help="rebuild the derived summary from the durable candidate raw")
    mode.add_argument(
        "--flat-only", action="store_true",
        help="run a corrected-engine flat-extras comparator on the same IDs, "
             "writing to the config's data.baseline_raw path")
    args = ap.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.flat_only and "selection_raw" not in config["data"]:
        raise SystemExit(
            "--flat-only needs a config with separate data.selection_raw "
            "(cohort source) and data.baseline_raw (flat output) roles; "
            f"{args.config} declares no selection_raw")
    baseline_path = Path(
        config["data"]["selection_raw"] if args.flat_only
        else config["data"]["baseline_raw"])
    for value in (baseline_path,
                  config["data"]["match_dir"],
                  config["data"]["context_dir"],
                  config["model"]["extras_graft"]):
        reject_sealed(value)
    baseline_raw = json.load(baseline_path.open())
    if args.aggregate_only:
        raw_path = Path(config["outputs"]["raw"])
        raw = json.load(raw_path.open())
        _require_engine_contract(raw, raw_path, "aggregation")
        _finalize(raw, baseline_raw, config, args.config)
        return
    selected = set(baseline_raw["selected_match_ids"])
    documents = {
        path.stem: json.load(path.open())
        for path in Path(config["data"]["match_dir"]).glob("*.json")
        if path.stem in selected
    }
    selected_dates = {str(doc["info"]["dates"][0])
                      for doc in documents.values()}
    context_by_date = build_context_by_date(
        config["data"]["context_dir"], selected_dates)

    selector = EmpiricalBowlerSelector(config["simulation"]["bowler_usage"])
    stack = build_t1_stack(
        config, selector,
        extras_graft=None if args.flat_only
        else config["model"]["extras_graft"])
    n_sims = int(config["simulation"]["n_sims"])
    seed = int(config["simulation"]["random_seed"])
    baseline_by_id = {m["match_id"]: m for m in baseline_raw["matches"]}
    raw = {
        "contract": ("t1_sim_flat_corrected_ppc_v1" if args.flat_only
                     else "t1_sim_extras_graft_ppc_v1"),
        "config": str(args.config),
        "sealed_data_used": False,
        "n_sims": n_sims,
        "matches": [],
        "started_at_unix": time.time(),
        "engine_contract": ENGINE_CONTRACT,
    }

    def on_match(match_id: str, document: dict, date_str: str) -> None:
        state = create_state(stack.loader, document, match_id)
        results = simulate(stack.engine, state, n_sims, seed)
        base = baseline_by_id[match_id]
        row = {
            "match_id": match_id,
            "date": date_str,
            "first_batting_team": document["innings"][0]["team"],
            "actual_winner": base["actual_winner"],
            "actual_innings": actual_innings(document),
            "simulations": [sim_result(result) for result in results],
        }
        raw["matches"].append(row)
        first = row["first_batting_team"]
        extras = np.mean([
            inn["extras_events"] for sim in row["simulations"]
            for inn in sim["innings"]])
        print(f"[{len(raw['matches'])}/{len(selected)}] {match_id} "
              f"p_first={win_probability(row['simulations'], first):.3f} "
              f"extras/innings={extras:.3f}", flush=True)

    replay_selected(context_by_date, stack.provider, selected, on_match)
    raw["elapsed_seconds"] = time.time() - raw["started_at_unix"]
    if args.flat_only:
        # The flat comparator IS the baseline the registered aggregation
        # reads, so it writes to the config's baseline_raw slot.
        raw_path = Path(config["data"]["baseline_raw"])
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(raw, indent=2) + "\n")
        print(json.dumps({
            "contract": raw["contract"],
            "n_matches": len(raw["matches"]),
            "n_sims": raw["n_sims"],
            "engine_contract": raw["engine_contract"],
            "elapsed_seconds": raw["elapsed_seconds"],
            "raw": str(raw_path),
        }, indent=2))
        return
    raw_path = Path(config["outputs"]["raw"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(raw, indent=2) + "\n")
    _finalize(raw, baseline_raw, config, args.config)


if __name__ == "__main__":
    main()
