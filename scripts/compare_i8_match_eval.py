#!/usr/bin/env python3
"""Paired I8-vs-I7 match evaluation with competition-block uncertainty."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SLICE_SUFFIXES = ("all", "min_volume_50000", "min_volume_100000")


def _sibling_slice(path: Path, suffix: str) -> Path:
    stem = path.stem
    if not stem.endswith("_all"):
        raise ValueError(f"expected an *_all.json path, found {path}")
    return path.with_name(f"{stem[:-4]}_{suffix}.json")


def _load(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text())
    return {row["match_id"]: row for row in data["matches"]}


def _finite(value) -> bool:
    return value is not None and math.isfinite(float(value))


def _flat_pnl(row: dict) -> float:
    """Recompute one-unit P&L from the immutable market/outcome fields."""
    if not row.get("bet_placed", True):
        return 0.0
    bet_team = row.get("bet_team")
    if not bet_team:
        raise RuntimeError(f"placed bet has no team for {row['match_id']}")
    if bet_team != row.get("actual_winner"):
        return -1.0
    return float(row["market_odds"][bet_team]) - 1.0


def _cluster_bootstrap(
    paired: list[tuple[dict, dict]],
    *,
    n_resamples: int,
    seed: int,
) -> dict:
    clusters: dict[str, list[tuple[dict, dict]]] = {}
    for baseline, candidate in paired:
        cluster = candidate.get("competition_cluster_id")
        if cluster != baseline.get("competition_cluster_id"):
            raise RuntimeError(
                f"cluster mismatch for {candidate['match_id']}"
            )
        clusters.setdefault(str(cluster), []).append((baseline, candidate))
    cluster_ids = sorted(clusters)
    rng = np.random.default_rng(seed)

    ll_delta = {}
    brier_delta = {}
    roi_values = {}
    for cluster_id, rows in clusters.items():
        ll_pairs = [
            (float(base["log_loss"]), float(cand["log_loss"]))
            for base, cand in rows
            if _finite(base.get("log_loss"))
            and _finite(cand.get("log_loss"))
        ]
        brier_pairs = [
            (float(base["brier_score"]), float(cand["brier_score"]))
            for base, cand in rows
            if _finite(base.get("brier_score"))
            and _finite(cand.get("brier_score"))
        ]
        ll_delta[cluster_id] = (
            sum(cand - base for base, cand in ll_pairs),
            len(ll_pairs),
        )
        brier_delta[cluster_id] = (
            sum(cand - base for base, cand in brier_pairs),
            len(brier_pairs),
        )
        roi_values[cluster_id] = (
            sum(_flat_pnl(base) for base, _ in rows
                if base.get("bet_placed", True)),
            sum(bool(base.get("bet_placed", True)) for base, _ in rows),
            sum(_flat_pnl(cand) for _, cand in rows
                if cand.get("bet_placed", True)),
            sum(bool(cand.get("bet_placed", True)) for _, cand in rows),
        )

    boot_ll = np.empty(n_resamples)
    boot_brier = np.empty(n_resamples)
    boot_roi = np.empty(n_resamples)
    for index in range(n_resamples):
        sampled = rng.integers(0, len(cluster_ids), size=len(cluster_ids))
        selected = [cluster_ids[position] for position in sampled]

        ll_sum = sum(ll_delta[c][0] for c in selected)
        ll_n = sum(ll_delta[c][1] for c in selected)
        brier_sum = sum(brier_delta[c][0] for c in selected)
        brier_n = sum(brier_delta[c][1] for c in selected)
        base_pnl = sum(roi_values[c][0] for c in selected)
        base_bets = sum(roi_values[c][1] for c in selected)
        cand_pnl = sum(roi_values[c][2] for c in selected)
        cand_bets = sum(roi_values[c][3] for c in selected)

        boot_ll[index] = ll_sum / ll_n
        boot_brier[index] = brier_sum / brier_n
        boot_roi[index] = (
            100.0 * cand_pnl / cand_bets
            - 100.0 * base_pnl / base_bets
        )

    def interval(values: np.ndarray) -> list[float]:
        return [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ]

    return {
        "n_clusters": len(cluster_ids),
        "n_resamples": n_resamples,
        "seed": seed,
        "log_loss_delta_ci": interval(boot_ll),
        "brier_delta_ci": interval(boot_brier),
        "flat_roi_pct_point_delta_ci": interval(boot_roi),
    }


def _point_metrics(paired: list[tuple[dict, dict]]) -> dict:
    def arm(index: int) -> dict:
        rows = [pair[index] for pair in paired]
        ll = [float(row["log_loss"]) for row in rows
              if _finite(row.get("log_loss"))]
        brier = [float(row["brier_score"]) for row in rows
                 if _finite(row.get("brier_score"))]
        bets = [row for row in rows if row.get("bet_placed", True)]
        pnl = sum(_flat_pnl(row) for row in bets)
        return {
            "n_scored": len(ll),
            "n_bets": len(bets),
            "log_loss": float(np.mean(ll)),
            "brier": float(np.mean(brier)),
            "flat_pnl": pnl,
            "flat_roi_pct": 100.0 * pnl / len(bets),
        }

    baseline = arm(0)
    candidate = arm(1)
    return {
        "baseline": baseline,
        "candidate": candidate,
        "candidate_minus_baseline": {
            "log_loss": candidate["log_loss"] - baseline["log_loss"],
            "brier": candidate["brier"] - baseline["brier"],
            "flat_roi_pct_points": (
                candidate["flat_roi_pct"] - baseline["flat_roi_pct"]
            ),
        },
    }


def _pair(
    baseline: dict[str, dict],
    candidate: dict[str, dict],
    *,
    remove: set[str] | None = None,
) -> list[tuple[dict, dict]]:
    remove = remove or set()
    match_ids = sorted(
        (set(baseline) & set(candidate)) - remove
    )
    if len(match_ids) != len(set(baseline) - remove):
        raise RuntimeError("candidate is missing baseline match IDs")
    paired = []
    for match_id in match_ids:
        base = baseline[match_id]
        cand = candidate[match_id]
        if (
            base.get("actual_winner") != cand.get("actual_winner")
            or base.get("market_prob") != cand.get("market_prob")
        ):
            raise RuntimeError(f"outcome/market mismatch for {match_id}")
        paired.append((base, cand))
    return paired


def _longshot_wins(candidate: dict[str, dict]) -> list[str]:
    result = []
    for match_id, row in candidate.items():
        bet_team = row.get("bet_team")
        if (
            row.get("bet_placed", True)
            and _flat_pnl(row) > 0.0
            and bet_team
            and float(row["market_prob"][bet_team]) <= 0.10
        ):
            result.append(match_id)
    return sorted(result)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-all",
        type=Path,
        default=ROOT / "eval_out_i7" / "sliced"
        / "xgboost_all_20260725_183709_all.json",
    )
    parser.add_argument(
        "--candidate-all",
        type=Path,
        default=ROOT / "eval_out_i8" / "sliced"
        / "match_evaluation_results_xgboost_all_20260730_104246_all.json",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "i8_match_evaluation.json",
    )
    args = parser.parse_args()

    report = {
        "contract": {
            "baseline": "I7 canonical venue model",
            "candidate": "I8 phase and H2H distributions",
            "n_simulations_per_match": 100,
            "simulation_seed": 42,
            "bootstrap_unit": "competition_time_block",
            "bootstrap_resamples": args.bootstrap_resamples,
            "diagnostic_only": True,
        },
        "slices": {},
    }
    all_candidate = None
    all_baseline = None
    for suffix in SLICE_SUFFIXES:
        baseline_path = _sibling_slice(args.baseline_all, suffix)
        candidate_path = _sibling_slice(args.candidate_all, suffix)
        baseline = _load(baseline_path)
        candidate = _load(candidate_path)
        paired = _pair(baseline, candidate)
        points = _point_metrics(paired)
        points["paired_bootstrap"] = _cluster_bootstrap(
            paired,
            n_resamples=args.bootstrap_resamples,
            seed=args.seed,
        )
        points["source"] = {
            "baseline": str(baseline_path),
            "candidate": str(candidate_path),
        }
        report["slices"][suffix] = points
        if suffix == "all":
            all_baseline = baseline
            all_candidate = candidate

    assert all_baseline is not None and all_candidate is not None
    longshot_wins = _longshot_wins(all_candidate)
    largest_win = max(
        (
            match_id
            for match_id, row in all_candidate.items()
            if row.get("bet_placed", True)
        ),
        key=lambda match_id: _flat_pnl(all_candidate[match_id]),
    )
    report["sensitivity"] = {
        "largest_candidate_win": {
            "match_id": largest_win,
            "candidate_flat_pnl": _flat_pnl(
                all_candidate[largest_win]
            ),
            "after_removal": _point_metrics(_pair(
                all_baseline,
                all_candidate,
                remove={largest_win},
            )),
        },
        "all_winning_bets_at_market_probability_lte_10pct": {
            "match_ids": longshot_wins,
            "after_removal": _point_metrics(_pair(
                all_baseline,
                all_candidate,
                remove=set(longshot_wins),
            )),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
