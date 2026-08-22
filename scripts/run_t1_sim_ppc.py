#!/usr/bin/env python3
"""Registered posterior-predictive and orientation audit for T1 rollout.

Root of the T1 PPC family: defines the stratified 24-match cohort and the raw
schema `run_t1_sim_extras_ppc.py` / `run_t1_sim_roster_ppc.py` consume. The
shared replay/summary machinery lives in `t1_ppc_common.py`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from t1_ppc_common import (REPO_ROOT, actual_innings,  # noqa: E402
                           bootstrap_mean, build_context_by_date,
                           build_t1_stack, create_state,
                           ordinary_test_order_reference, provenance_block,
                           reject_sealed, reliability, replay_selected,
                           result_signature, sim_result, simulate,
                           swap_storage, win_probability)
from sim_v1_2 import EmpiricalBowlerSelector, T20Rules  # noqa: E402

# Compatibility aliases: the focused suites and the derived PPC runners were
# written against these names before the shared module existed.
_actual_innings = actual_innings
_bootstrap_mean = bootstrap_mean
_ordinary_test_order_reference = ordinary_test_order_reference
_reject_sealed = reject_sealed
_reliability = reliability
_result_signature = result_signature
_sim_result = sim_result
_simulate = simulate
_swap_storage = swap_storage
_win_probability = win_probability


def _stable_rank(seed: int, match_id: str) -> str:
    return hashlib.sha256(f"{seed}:{match_id}".encode()).hexdigest()


def _implied(odds: dict) -> dict:
    raw = {team: 1.0 / float(value) for team, value in odds.items()
           if team != "timestamp" and float(value) > 0}
    total = sum(raw.values())
    return {team: value / total for team, value in raw.items()}


def _summarize(raw: dict, config: dict) -> dict:
    matches = raw["matches"]
    reps = int(config["statistics"]["bootstrap_repetitions"])
    boot_seed = int(config["statistics"]["bootstrap_seed"])
    p_actual, p_flip, delta, y = [], [], [], []
    ll, brier = [], []
    fi_bias, fi_abs, fi_cover = [], [], []
    innings_metrics = defaultdict(list)
    phase_metrics = defaultdict(list)

    for match in matches:
        first_team = match["first_batting_team"]
        pa = win_probability(match["actual_order"], first_team)
        pf = win_probability(match["flipped_order"], first_team)
        p_actual.append(pa)
        p_flip.append(pf)
        delta.append(pa - pf)
        outcome = float(match["actual_winner"] == first_team)
        y.append(outcome)
        ll.append(float(-(outcome * np.log(max(pa, 1e-15))
                          + (1 - outcome) * np.log(max(1 - pa, 1e-15)))))
        brier.append((pa - outcome) ** 2)

        for innings_idx in (0, 1):
            actual = match["actual_innings"][innings_idx]
            sims = [row["innings"][innings_idx]
                    for row in match["actual_order"]
                    if len(row["innings"]) > innings_idx]
            prefix = f"innings_{innings_idx + 1}"
            for field in ("total_runs", "wickets", "legal_balls",
                          "deliveries", "extras_events", "unique_bowlers"):
                values = np.asarray([row[field] for row in sims], dtype=float)
                innings_metrics[f"{prefix}_{field}_actual"].append(
                    actual[field])
                innings_metrics[f"{prefix}_{field}_posterior_mean"].append(
                    float(values.mean()))
            for phase in ("powerplay", "middle", "death"):
                values = np.asarray(
                    [row["phase_runs"][phase] for row in sims], dtype=float)
                phase_metrics[f"{prefix}_{phase}_actual"].append(
                    actual["phase_runs"][phase])
                phase_metrics[f"{prefix}_{phase}_posterior_mean"].append(
                    float(values.mean()))

        actual_total = match["actual_innings"][0]["total_runs"]
        posterior = np.asarray(
            [row["innings"][0]["total_runs"]
             for row in match["actual_order"]], dtype=float)
        fi_bias.append(float(posterior.mean() - actual_total))
        fi_abs.append(abs(float(posterior.mean() - actual_total)))
        lo, hi = np.percentile(posterior, [10, 90])
        fi_cover.append(int(lo <= actual_total <= hi))

    p_actual = np.asarray(p_actual)
    p_flip = np.asarray(p_flip)
    delta = np.asarray(delta)
    y = np.asarray(y)
    symmetry_pass = raw["symmetry_mismatches"] == 0
    chase_delta = float(delta.mean())
    order_reference = ordinary_test_order_reference(
        Path(config["data"]["match_dir"]), reps, boot_seed)
    paired_chasing_advantage = -chase_delta
    reference_ci = order_reference[
        "chasing_minus_first_batting_ci95"]
    # Batting order is a real intervention, not an invariance. The invariant
    # is storage-label symmetry. Treat the flip as a behavioral estimate and
    # reject it only when it is outside the ordinary cohort's descriptive
    # range; a controlled strength/venue estimate remains future evidence.
    order_pass = reference_ci[0] <= paired_chasing_advantage <= reference_ci[1]
    extras_actual = np.mean(
        [m["actual_innings"][i]["extras_events"]
         for m in matches for i in (0, 1)])
    extras_sim = np.mean(
        [row["innings"][i]["extras_events"]
         for m in matches for row in m["actual_order"] for i in (0, 1)
         if len(row["innings"]) > i])
    extras_pass = abs(extras_sim - extras_actual) <= 1.0
    structural_pass = symmetry_pass and order_pass and extras_pass
    summary = {
        "contract": "t1_sim_ppc_v1",
        "status": (
            "STRUCTURAL_AND_PPC_PASS"
            if structural_pass else "PPC_FAIL"
        ),
        "gates": {
            "team_label_symmetry": "PASS" if symmetry_pass else "FAIL",
            "paired_order_effect_compatible_with_descriptive_reference": (
                "PASS" if order_pass else "FAIL"),
            "extras_events_within_1_per_innings": (
                "PASS" if extras_pass else "FAIL"),
        },
        "n_matches": len(matches),
        "n_sims_per_arm": raw["n_sims_per_arm"],
        "calibration_applied": False,
        "empirical_b10_selector_active": raw["empirical_b10_selector_active"],
        "symmetry": {
            "comparisons": raw["symmetry_comparisons"],
            "mismatches": raw["symmetry_mismatches"],
        },
        "winner": {
            "mean_p_observed_first_batting_team": float(p_actual.mean()),
            "actual_first_batting_win_rate": float(y.mean()),
            "mean_p_same_team_when_chasing": float(p_flip.mean()),
            "paired_batting_first_minus_chasing": {
                "mean": float(delta.mean()),
                "ci95": bootstrap_mean(delta, reps, boot_seed),
                "per_match": delta.tolist(),
            },
            "ordinary_test_order_reference": order_reference,
            "raw_log_loss": float(np.mean(ll)),
            "raw_brier": float(np.mean(brier)),
            "reliability": reliability(p_actual, y),
        },
        "first_innings_total": {
            "mean_bias_posterior_minus_actual": float(np.mean(fi_bias)),
            "bias_ci95": bootstrap_mean(fi_bias, reps, boot_seed),
            "mae_of_posterior_mean": float(np.mean(fi_abs)),
            "p10_p90_coverage": float(np.mean(fi_cover)),
        },
        "extras_events_per_innings": {
            "actual": float(extras_actual),
            "posterior_mean": float(extras_sim),
            "delta": float(extras_sim - extras_actual),
        },
        "innings_metrics": {},
        "phase_metrics": {},
        "strata": {},
    }
    for key, values in innings_metrics.items():
        summary["innings_metrics"][key] = float(np.mean(values))
    for key, values in phase_metrics.items():
        summary["phase_metrics"][key] = float(np.mean(values))
    for stratum in sorted({m["stratum"] for m in matches}):
        ix = [i for i, m in enumerate(matches) if m["stratum"] == stratum]
        summary["strata"][stratum] = {
            "n": len(ix),
            "mean_market_p_first_batting": float(np.mean(
                [matches[i]["market_p_first_batting"] for i in ix])),
            "mean_sim_p_first_batting": float(p_actual[ix].mean()),
            "actual_first_batting_win_rate": float(y[ix].mean()),
            "mean_paired_batting_first_minus_chasing": float(
                delta[ix].mean()),
        }

    reference_path = (REPO_ROOT / "research/reports/embeddings/artifacts/"
                      "t4_rollout_results_20260808.json")
    if reference_path.exists():
        ref = json.load(reference_path.open())
        ref_by_id = {str(row["match_id"]): row for row in ref["matches"]}
        ref_probs, ref_y = [], []
        for match in matches:
            row = ref_by_id.get(match["match_id"])
            if row is None:
                continue
            team = match["first_batting_team"]
            ref_probs.append(float(row["simulated_prob"][team]))
            ref_y.append(float(row["actual_winner"] == team))
        if ref_probs:
            rp = np.asarray(ref_probs)
            ry = np.asarray(ref_y)
            summary["old_t4_same_cohort_reference"] = {
                "n": len(rp),
                "mean_p_first_batting_team": float(rp.mean()),
                "actual_first_batting_win_rate": float(ry.mean()),
                "log_loss": float(np.mean(
                    -(ry * np.log(np.clip(rp, 1e-15, 1))
                      + (1 - ry) * np.log(np.clip(1 - rp, 1e-15, 1))))),
                "brier": float(np.mean((rp - ry) ** 2)),
            }
    else:
        print(f"note: old-T4 reference absent ({reference_path}); "
              "old_t4_same_cohort_reference omitted from the summary",
              file=sys.stderr, flush=True)
    return summary


def _finalize(raw: dict, config: dict, config_path: Path,
              write_raw: bool = False) -> None:
    """Summarize, stamp provenance, persist, and exit non-zero on FAIL."""
    summary = _summarize(raw, config)
    summary["selected_match_ids"] = raw["selected_match_ids"]
    summary["elapsed_seconds"] = raw["elapsed_seconds"]
    summary["provenance"] = provenance_block(config_path, config)
    if write_raw:
        raw_path = Path(config["outputs"]["raw"])
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(raw, indent=2) + "\n")
    summary_path = Path(config["outputs"]["summary"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if summary["status"] != "STRUCTURAL_AND_PPC_PASS":
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", type=Path,
        default=Path("experiments/configs/t1_sim_ppc_v1.yaml"))
    ap.add_argument(
        "--aggregate-only", action="store_true",
        help="rebuild the derived summary from the durable raw simulations")
    args = ap.parse_args()
    config = yaml.safe_load(args.config.read_text())
    for value in (config["data"]["match_dir"],
                  config["data"]["context_dir"],
                  config["data"]["odds"]):
        reject_sealed(value)

    if args.aggregate_only:
        raw = json.load(Path(config["outputs"]["raw"]).open())
        _finalize(raw, config, args.config)
        return

    match_dir = Path(config["data"]["match_dir"])
    documents = {path.stem: json.load(path.open())
                 for path in sorted(match_dir.glob("*.json"))}
    odds_doc = json.load(Path(config["data"]["odds"]).open())
    odds_rows = {str(row["cricsheet_id"]): row
                 for row in odds_doc["matches"]}

    candidates = []
    for match_id, document in documents.items():
        if match_id not in odds_rows or len(document.get("innings", [])) < 2:
            continue
        first_team = document["innings"][0]["team"]
        market = _implied(odds_rows[match_id]["odds"]["winner"])
        candidates.append((match_id, first_team, market[first_team]))

    selection_seed = int(config["cohort"]["selection_seed"])
    selected = {}
    for name, spec in config["cohort"]["strata"].items():
        lo, hi = map(float, spec["range"])
        eligible = [row for row in candidates if lo <= row[2] < hi]
        eligible.sort(key=lambda row: _stable_rank(selection_seed, row[0]))
        need = int(spec["n"])
        if len(eligible) < need:
            raise RuntimeError(f"stratum {name} has {len(eligible)} < {need}")
        for match_id, first_team, probability in eligible[:need]:
            selected[match_id] = {
                "stratum": name,
                "first_team": first_team,
                "market_probability": probability,
            }

    selected_dates = {
        str(documents[mid]["info"]["dates"][0]) for mid in selected}
    context_by_date = build_context_by_date(
        config["data"]["context_dir"], selected_dates)

    selector = EmpiricalBowlerSelector(
        usage_path=config["simulation"]["bowler_usage"])
    stack = build_t1_stack(config, selector)

    n_sims = int(config["simulation"]["n_sims_per_arm"])
    symmetry_sims = int(config["simulation"]["symmetry_sims"])
    seed = int(config["simulation"]["random_seed"])
    raw = {
        "contract": "t1_sim_ppc_v1",
        "config": str(args.config),
        "sealed_data_used": False,
        "n_sims_per_arm": n_sims,
        "selected_match_ids": sorted(selected),
        "matches": [],
        "symmetry_comparisons": 0,
        "symmetry_mismatches": 0,
        "empirical_b10_selector_active": False,
        "started_at_unix": time.time(),
    }

    def on_match(match_id: str, document: dict, date_str: str) -> None:
        state = create_state(stack.loader, document, match_id)
        observed_first = document["innings"][0]["team"]

        actual_results = simulate(stack.engine, state, n_sims, seed)
        flipped = state.copy()
        flipped.batting_first = (
            state.team2 if state.batting_first == state.team1
            else state.team1)
        flipped_results = simulate(stack.engine, flipped, n_sims, seed)
        swapped_results = simulate(
            stack.engine, swap_storage(state), symmetry_sims, seed)
        for original, swapped in zip(
            actual_results[:symmetry_sims], swapped_results
        ):
            raw["symmetry_comparisons"] += 1
            raw["symmetry_mismatches"] += int(
                result_signature(original) != result_signature(swapped))

        odds = odds_rows[match_id]
        raw["matches"].append({
            "match_id": match_id,
            "date": date_str,
            "stratum": selected[match_id]["stratum"],
            "teams": list(document["info"]["teams"]),
            "first_batting_team": observed_first,
            "market_p_first_batting": selected[match_id][
                "market_probability"],
            "actual_winner": odds.get("actual_winner"),
            "actual_innings": actual_innings(document),
            "actual_order": [sim_result(r) for r in actual_results],
            "flipped_order": [sim_result(r) for r in flipped_results],
        })
        print(
            f"[{len(raw['matches'])}/{len(selected)}] {match_id} "
            f"stratum={selected[match_id]['stratum']} "
            f"p_first={win_probability(raw['matches'][-1]['actual_order'], observed_first):.3f} "
            f"p_same_chasing={win_probability(raw['matches'][-1]['flipped_order'], observed_first):.3f} "
            f"symmetry_mismatch={raw['symmetry_mismatches']}",
            flush=True,
        )

    replay_selected(context_by_date, stack.provider, selected, on_match)
    raw["empirical_b10_selector_active"] = selector.empirical_usage_active
    raw["elapsed_seconds"] = time.time() - raw["started_at_unix"]
    _finalize(raw, config, args.config, write_raw=True)


if __name__ == "__main__":
    main()
