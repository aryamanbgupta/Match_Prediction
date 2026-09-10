#!/usr/bin/env python3
"""Stage-1 exploratory realism scorer (D7 check 7.4).

Reads one arm directory's `raw_sims.jsonl` (written by
`sequence_track/run_arm.py`, one line per fixture carrying every
simulation's winner/tie and both innings' totals, wickets, balls, extras
events, phase runs, unique bowlers, per-batter runs and per-bowler
wickets) and compares those draws with the cricsheet actuals for the same
fixtures.

Nothing is re-simulated and no model is loaded: the arm runner's own
per-simulation output is the input, and `t1_ppc_common` supplies the
innings summarizer (`actual_innings`), the posterior-mean/coverage
reducer (`match_metrics`), the winner-probability rule
(`win_probability`), the descriptive batting-order reference
(`ordinary_test_order_reference`) and the bootstrap (`bootstrap_mean`).

Reported per fixture and in aggregate:

* first-innings score bias (sim mean minus actual) and |bias|;
* P10 / P50 / P90 of the simulated innings total and the P10–P90
  coverage indicator for the actual total, per innings;
* extras events, wickets and unique bowlers per innings (actual, sim
  posterior mean, bias), plus every other `t1_ppc_common.INNINGS_FIELDS`
  quantity for free;
* second-innings totals over completed chases, split by whether the
  chase succeeded (a successful chase truncates the actual total at the
  winning run, so a positive bias there is expected by construction);
* the batting-first minus chasing win-rate summary implied by the sims,
  next to the descriptive actual-side reference over the same fixtures.

Aggregate intervals are simple by-fixture bootstraps: `numpy`
`default_rng(29)`, 2,000 resamples, 2.5/97.5 percentiles.

This is an exploratory scorer. It emits numbers and no verdicts.

Usage:
    uv run --no-sync python scripts/sequence_track/score_realism.py \
        --arm-dir <arm output dir> [--fixture-dir <cricsheet JSON dir>] \
        [--out <arm output dir>/realism.json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from registered_experiment import reject_sealed  # noqa: E402
from t1_ppc_common import (  # noqa: E402
    INNINGS_FIELDS,
    actual_innings,
    bootstrap_mean,
    match_metrics,
    ordinary_test_order_reference,
    win_probability,
)

CONTRACT = "sequence_track_realism_v1"
RAW_SIMS_FILENAME = "raw_sims.jsonl"
PROVENANCE_FILENAME = "arm_provenance.json"
DEFAULT_OUT_NAME = "realism.json"

BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 29
QUANTILES = (10, 50, 90)

# Required per-simulation keys. A raw file that predates one of them is
# reported by name instead of scored on a fabricated zero.
REQUIRED_SIM_INNINGS_FIELDS = tuple(INNINGS_FIELDS)


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

def load_jsonl(path) -> list[dict]:
    reject_sealed(path)
    rows = []
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: not JSON ({exc})")
    return rows


def provenance_fixture_dir(arm_dir: Path):
    """The fixture dir the arm run recorded, or None when absent.

    The recorded path is rejected here, not at the point of use: a
    provenance file is data, so a sealed fixture dir must fail closed
    however this scorer was invoked.
    """
    reject_sealed(arm_dir)
    path = Path(arm_dir) / PROVENANCE_FILENAME
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    recorded = (payload.get("run") or {}).get("fixture_dir")
    if not recorded:
        return None
    reject_sealed(recorded)
    return Path(recorded)


# Cricsheet ids are digits; a stage-1 smoke id may carry a suffix. Nothing
# else is a match id — a raw-sims row is data, and an absolute or
# traversal-bearing id would otherwise read outside the guarded fixture
# dir. The same guard is spelled identically in `score_props.py`.
MATCH_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def fixture_document_path(match_id, fixture_dir) -> Path:
    """`<fixture_dir>/<match_id>.json`, refusing anything that escapes it.

    The id must be a bare token; the path is then resolved (following
    symlinks) and must be sealed-free and inside the resolved fixture
    directory before any caller opens it.
    """
    if match_id is None or not MATCH_ID_RE.match(str(match_id)):
        raise SystemExit(
            f"refusing match id {match_id!r}: a fixture id must match "
            f"{MATCH_ID_RE.pattern}")
    root = Path(fixture_dir).resolve()
    path = (root / f"{match_id}.json").resolve()
    reject_sealed(path)
    try:
        path.relative_to(root)
    except ValueError:
        raise SystemExit(
            f"refusing fixture path for {match_id}: {path} resolves outside "
            f"the fixture directory {root}")
    return path


def assert_fixture_dir_clean(fixture_dir) -> Path:
    """Resolve the fixture dir and refuse any entry that escapes it.

    `ordinary_test_order_reference` globs the directory itself, so the
    per-id guard above is not enough: a symlinked entry pointing at a
    sealed holdout has to be refused before that glob opens it.
    """
    reject_sealed(fixture_dir)
    root = Path(fixture_dir).resolve()
    reject_sealed(root)
    for entry in sorted(root.glob("*.json")):
        resolved = entry.resolve()
        reject_sealed(resolved)
        try:
            resolved.relative_to(root)
        except ValueError:
            raise SystemExit(
                f"refusing fixture directory {root}: {entry.name} resolves "
                f"outside it ({resolved})")
    return root


def void_reason(info: dict):
    """D/L-shortened or no-result fixtures: truncated actuals.

    Same curation rule `prop_backtest.py` applies before settling props —
    a shortened innings cannot be compared with a sim that always plays
    the full allotment.
    """
    outcome = (info or {}).get("outcome", {}) or {}
    if outcome.get("method"):
        return f"outcome.method={outcome['method']}"
    if str(outcome.get("result", "")).lower() == "no result":
        return "outcome.result=no result"
    return None


def eligible(row: dict, document: dict):
    """(reason to exclude or None) for one raw-sims fixture row."""
    info = document.get("info", {}) or {}
    reason = void_reason(info)
    if reason:
        return reason
    if not row.get("actual_winner"):
        return "no decided winner in the cricsheet outcome"
    observed = row.get("actual_innings") or []
    if len(observed) != 2:
        return f"cricsheet has {len(observed)} regulation innings, not 2"
    simulations = row.get("simulations") or []
    if not simulations:
        return "no simulations recorded"
    for simulation in simulations:
        if len(simulation.get("innings") or []) != 2:
            return "a simulation does not carry two innings"
        for innings in simulation["innings"]:
            missing = [field for field in REQUIRED_SIM_INNINGS_FIELDS
                       if field not in innings]
            if missing:
                return f"simulation innings missing {', '.join(missing)}"
    return None


# ---------------------------------------------------------------------------
# Per-fixture and aggregate reduction
# ---------------------------------------------------------------------------

def innings_row(actual: dict, sim_innings: list[dict]) -> dict:
    totals = np.asarray([innings["total_runs"] for innings in sim_innings],
                        dtype=float)
    p10, p50, p90 = (float(value)
                     for value in np.percentile(totals, QUANTILES))
    row = {
        "sim_p10": p10,
        "sim_p50": p50,
        "sim_p90": p90,
        "covered_p10_p90": int(p10 <= actual["total_runs"] <= p90),
        "fields": {},
    }
    for field in INNINGS_FIELDS:
        values = np.asarray([innings[field] for innings in sim_innings],
                            dtype=float)
        row["fields"][field] = {
            "actual": float(actual[field]),
            "sim_mean": float(values.mean()),
            "bias": float(values.mean() - actual[field]),
        }
    return row


def has_batting_order(scored: list[dict]) -> bool:
    """Whether every raw row carries the batting order AND a sim winner."""
    return all(
        match.get("first_batting_team")
        and all("winner" in simulation for simulation in match["simulations"])
        for match in scored
    )


def metrics_input(scored: list[dict], batting_order: bool) -> list[dict]:
    """Rows `match_metrics` can reduce even without per-simulation winners.

    `match_metrics` computes winner probabilities alongside the innings
    summaries. When the raw file carries no per-simulation winner those
    probabilities are neither reportable nor reported, so a `None`
    placeholder keeps the innings pass running instead of raising.
    """
    if batting_order:
        return scored
    patched = []
    for match in scored:
        row = dict(match)
        row["first_batting_team"] = match.get("first_batting_team") or ""
        row["simulations"] = [
            {**simulation, "winner": simulation.get("winner")}
            for simulation in match["simulations"]
        ]
        patched.append(row)
    return patched


def fixture_rows(scored: list[dict], batting_order: bool) -> list[dict]:
    rows = []
    for match in scored:
        simulations = match["simulations"]
        first_team = match["first_batting_team"]
        second_team = match["actual_innings"][1]["batting_team"]
        chase_successful = match["actual_winner"] == second_team
        row = {
            "match_id": match["match_id"],
            "date": match.get("date"),
            "n_sims": len(simulations),
            "first_batting_team": first_team,
            "actual_winner": match["actual_winner"],
            "chase_successful": bool(chase_successful),
            "sim_first_batting_win_prob": (
                float(win_probability(simulations, first_team))
                if batting_order else None),
            "innings_1": innings_row(
                match["actual_innings"][0],
                [simulation["innings"][0] for simulation in simulations]),
            "innings_2": innings_row(
                match["actual_innings"][1],
                [simulation["innings"][1] for simulation in simulations]),
        }
        rows.append(row)
    return rows


def interval(values) -> dict:
    values = np.asarray(values, dtype=float)
    low, high = bootstrap_mean(values, BOOTSTRAP_REPS, BOOTSTRAP_SEED)
    return {"mean": float(values.mean()), "ci95": [low, high],
            "n": int(values.size)}


def innings_aggregate(metrics: dict, innings_index: int) -> dict:
    """Actual / posterior-mean / bias summaries for one innings."""
    prefix = f"innings_{innings_index + 1}"
    out = {}
    for field in INNINGS_FIELDS:
        actual = np.asarray(
            metrics["innings_metrics"][f"{prefix}_{field}_actual"],
            dtype=float)
        posterior = np.asarray(
            metrics["innings_metrics"][f"{prefix}_{field}_posterior_mean"],
            dtype=float)
        out[field] = {
            "actual_mean": float(actual.mean()),
            "sim_posterior_mean": float(posterior.mean()),
            "bias": interval(posterior - actual),
        }
    return out


def coverage_aggregate(rows: list[dict], key: str) -> dict:
    return interval([row[key]["covered_p10_p90"] for row in rows])


def second_innings_aggregate(rows: list[dict]) -> dict:
    """Second-innings totals over completed chases, split by outcome.

    "Completed" = the fixture reached a decided result over two
    regulation innings (the eligibility filter above). A successful chase
    stops at the winning run, so its actual total is truncated and a
    positive sim-minus-actual bias is expected by construction; the split
    keeps that out of the unsuccessful-chase reading.
    """
    def _block(subset):
        if not subset:
            return None
        bias = [row["innings_2"]["fields"]["total_runs"]["bias"]
                for row in subset]
        actual = [row["innings_2"]["fields"]["total_runs"]["actual"]
                  for row in subset]
        sim = [row["innings_2"]["fields"]["total_runs"]["sim_mean"]
               for row in subset]
        return {
            "n": len(subset),
            "actual_mean": float(np.mean(actual)),
            "sim_posterior_mean": float(np.mean(sim)),
            "bias": interval(bias),
            "abs_bias": interval(np.abs(bias)),
            "coverage_p10_p90": interval(
                [row["innings_2"]["covered_p10_p90"] for row in subset]),
        }

    successful = [row for row in rows if row["chase_successful"]]
    unsuccessful = [row for row in rows if not row["chase_successful"]]
    return {
        "definition": (
            "fixtures with two regulation innings and a decided result; "
            "a successful chase truncates the actual second-innings total "
            "at the winning run, so its bias is positive by construction"
        ),
        "all_completed": _block(rows),
        "chase_successful": _block(successful),
        "chase_unsuccessful": _block(unsuccessful),
    }


def batting_order_summary(rows: list[dict], scored: list[dict],
                          fixture_dir) -> dict:
    """Sim-implied batting-first minus chasing win rate, plus the actual."""
    wins_first = 0
    wins_chasing = 0
    ties = 0
    for match in scored:
        first_team = match["first_batting_team"]
        for simulation in match["simulations"]:
            winner = simulation.get("winner")
            if winner == first_team:
                wins_first += 1
            elif winner in (None, "Tie"):
                ties += 1
            else:
                wins_chasing += 1
    decided = wins_first + wins_chasing
    per_fixture = np.asarray(
        [row["sim_first_batting_win_prob"] for row in rows], dtype=float)
    summary = {
        "basis": (
            "per-simulation winner vs the fixture's batting-first team; "
            "ties excluded from the rate, counted separately"
        ),
        "simulations": {
            "n_simulations": int(decided + ties),
            "n_ties": int(ties),
            "pooled_first_batting_win_rate": (
                float(wins_first / decided) if decided else None),
            "pooled_chasing_win_rate": (
                float(wins_chasing / decided) if decided else None),
            "pooled_first_batting_minus_chasing": (
                float((wins_first - wins_chasing) / decided)
                if decided else None),
            "per_fixture_first_batting_win_rate": interval(per_fixture),
            "per_fixture_first_batting_minus_chasing": interval(
                2.0 * per_fixture - 1.0),
        },
    }
    summary["actual_reference_scope"] = (
        "every decided fixture in the fixture dir, not only the scored ones")
    try:
        summary["actual_reference"] = ordinary_test_order_reference(
            Path(fixture_dir), BOOTSTRAP_REPS, BOOTSTRAP_SEED)
    except ValueError as exc:
        summary["actual_reference"] = None
        summary["actual_reference_note"] = str(exc)
    return summary


def score(raw_rows: list[dict], fixture_dir) -> dict:
    """Realism payload for one arm's raw simulations."""
    fixture_root = assert_fixture_dir_clean(fixture_dir)
    fixture_dir = Path(fixture_dir)
    scored: list[dict] = []
    excluded: list[dict] = []
    for row in raw_rows:
        match_id = str(row.get("match_id"))
        document_path = fixture_document_path(row.get("match_id"),
                                              fixture_root)
        if not document_path.exists():
            raise SystemExit(
                f"fixture JSON not found for {match_id}: {document_path}")
        document = json.loads(document_path.read_text())
        recomputed = actual_innings(document)
        if recomputed != row.get("actual_innings"):
            raise SystemExit(
                f"{match_id}: the cricsheet actual innings recomputed from "
                f"{document_path} differ from the ones recorded in the raw "
                "simulations — the fixture directory does not match the run")
        reason = eligible(row, document)
        if reason:
            excluded.append({"match_id": match_id, "reason": reason})
            continue
        scored.append(row)

    if not scored:
        raise SystemExit(
            "no fixture is eligible for realism scoring: "
            + json.dumps(excluded))

    batting_order = has_batting_order(scored)
    rows = fixture_rows(scored, batting_order)
    metrics = match_metrics(metrics_input(scored, batting_order))

    # `match_metrics` recomputes first-innings coverage independently; the
    # two paths must agree or one of them is reading a different posterior.
    recomputed_cover = [row["innings_1"]["covered_p10_p90"] for row in rows]
    if list(map(int, metrics["first_cover_by_match"])) != recomputed_cover:
        raise RuntimeError(
            "first-innings coverage disagrees between match_metrics and the "
            "per-fixture reduction")

    aggregate = {
        "n_fixtures": len(scored),
        "first_innings": {
            "bias": interval(metrics["first_bias_by_match"]),
            "abs_bias": interval(metrics["first_abs_by_match"]),
            "coverage_p10_p90": interval(metrics["first_cover_by_match"]),
            "quantile_means": {
                "p10": float(np.mean([row["innings_1"]["sim_p10"]
                                      for row in rows])),
                "p50": float(np.mean([row["innings_1"]["sim_p50"]
                                      for row in rows])),
                "p90": float(np.mean([row["innings_1"]["sim_p90"]
                                      for row in rows])),
            },
            "fields": innings_aggregate(metrics, 0),
        },
        "second_innings": {
            "coverage_p10_p90": coverage_aggregate(rows, "innings_2"),
            "quantile_means": {
                "p10": float(np.mean([row["innings_2"]["sim_p10"]
                                      for row in rows])),
                "p50": float(np.mean([row["innings_2"]["sim_p50"]
                                      for row in rows])),
                "p90": float(np.mean([row["innings_2"]["sim_p90"]
                                      for row in rows])),
            },
            "fields": innings_aggregate(metrics, 1),
        },
        "second_innings_completed_chases": second_innings_aggregate(rows),
        "extras_events_pooled_over_innings": {
            "actual_mean": float(np.mean(metrics["extras_actual"])),
            "sim_posterior_mean": float(np.mean(metrics["extras_sim"])),
            "bias": interval(
                np.asarray(metrics["extras_sim"], dtype=float)
                - np.asarray(metrics["extras_actual"], dtype=float)),
        },
    }
    if batting_order:
        aggregate["batting_order"] = batting_order_summary(
            rows, scored, fixture_dir)
    else:
        aggregate["batting_order"] = None
        aggregate["batting_order_note"] = (
            "omitted: the raw simulations do not carry both the fixture's "
            "batting-first team and a per-simulation winner"
        )

    return {
        "contract": CONTRACT,
        "fixture_dir": str(fixture_dir),
        "bootstrap": {
            "reps": BOOTSTRAP_REPS,
            "seed": BOOTSTRAP_SEED,
            "generator": "numpy.random.default_rng",
            "resample": "by fixture",
        },
        "n_fixtures_in_raw_sims": len(raw_rows),
        "n_fixtures_scored": len(scored),
        "excluded_fixtures": excluded,
        "aggregate": aggregate,
        "fixtures": rows,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score one stage-1 arm's raw simulations for realism")
    parser.add_argument("--arm-dir", required=True)
    parser.add_argument(
        "--fixture-dir", default=None,
        help="directory of cricsheet fixture JSONs (default: the fixture "
             "dir recorded in arm_provenance.json)")
    parser.add_argument(
        "--out", default=None,
        help=f"output JSON (default: <arm-dir>/{DEFAULT_OUT_NAME})")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    # Sealed-data guard first: every path this scorer may read or write is
    # refused before anything is opened, exactly as `run_arm.py` does.
    for candidate in (args.arm_dir, args.fixture_dir, args.out):
        if candidate is not None:
            reject_sealed(candidate)
    arm_dir = Path(args.arm_dir)
    raw_path = arm_dir / RAW_SIMS_FILENAME
    reject_sealed(raw_path)
    if not raw_path.exists():
        raise SystemExit(f"no {RAW_SIMS_FILENAME} in {arm_dir}")

    fixture_dir = (Path(args.fixture_dir) if args.fixture_dir
                   else provenance_fixture_dir(arm_dir))
    if fixture_dir is None:
        raise SystemExit(
            f"no --fixture-dir given and {arm_dir / PROVENANCE_FILENAME} "
            "records none")
    if not Path(fixture_dir).is_dir():
        raise SystemExit(f"fixture dir not found: {fixture_dir}")

    payload = score(load_jsonl(raw_path), fixture_dir)
    provenance = arm_dir / PROVENANCE_FILENAME
    if provenance.exists():
        run = (json.loads(provenance.read_text()).get("run") or {})
        payload["arm"] = run.get("arm")
        payload["arm_run"] = {
            key: run.get(key) for key in
            ("arm", "model_dir", "model_dir_hash", "stats_version",
             "base_seed", "n_sims", "engine_md5", "runner_md5")
        }
    payload["arm_dir"] = str(arm_dir)
    payload["raw_sims"] = str(raw_path)

    out_path = Path(args.out) if args.out else arm_dir / DEFAULT_OUT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    print(f"[score_realism] fixtures scored: {payload['n_fixtures_scored']} "
          f"of {payload['n_fixtures_in_raw_sims']}")
    for row in payload["excluded_fixtures"]:
        print(f"[score_realism] excluded {row['match_id']}: {row['reason']}")
    print(f"[score_realism] wrote {out_path}")


if __name__ == "__main__":
    main()
