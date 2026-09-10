#!/usr/bin/env python3
"""Stage-1 exploratory prop scorer (D7 check 7.5).

Scores one arm's `raw_sims.jsonl` (written by `sequence_track/run_arm.py`)
against the cricsheet actuals and the versioned fair baselines, for every
prop family `sim_eval/prop_backtest.py` defines whose inputs the raw file
actually carries.

Nothing is re-simulated and no family is re-implemented:

* the per-simulation aggregation is `prop_backtest.aggregate_per_player`,
  driven with per-simulation records rebuilt from `raw_sims.jsonl`;
* the per-match actuals are `prop_backtest.compute_actuals`;
* the (predicted, observed) pairs per family are
  `prop_backtest.build_observations`;
* the fair baselines and the by-match cluster bootstrap are
  `sim_eval.prop_fair_baselines.baseline_rows` /
  `cluster_bootstrap_delta`, over the corpus resolved through the
  `prop_fair_baseline_corpus_v2` manifest role;
* the lineup index order that turns a raw file's player ids back into
  card indices is `sim_eval.loaders.TestMatchLoader`'s own match state.

`raw_sims.jsonl` keeps, per simulation and innings, the innings totals,
wickets, balls, extras events, phase runs, unique bowlers, per-batter runs
and per-bowler wickets. It does NOT keep the ball log, per-batter
fours/sixes/balls or per-bowler runs/balls, so the families that need
those are reported by name with the exact missing field instead of being
scored on a fabricated zero.

Output (`props.json`): per family, the row count, the sim metric (Brier
for binary families, MAE for `*_mae`), the fair-baseline metric, the
paired difference and its 95% by-match bootstrap interval
(`default_rng(29)`, 2,000 resamples) — numbers only, no verdicts.

Usage:
    uv run --no-sync python scripts/sequence_track/score_props.py \
        --arm-dir <arm output dir> [--out <arm output dir>/props.json] \
        [--families all]
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from artifacts import artifact_path, md5_file  # noqa: E402
from registered_experiment import reject_sealed  # noqa: E402
from sim_eval.loaders import TestMatchLoader  # noqa: E402
from sim_eval.prop_backtest import (  # noqa: E402
    aggregate_per_player,
    build_observations,
    compute_actuals,
)
from sim_eval.prop_fair_baselines import (  # noqa: E402
    BASELINE_VERSION,
    AsOf,
    _venue_from_match_id,
    baseline_rows,
    brier,
    cluster_bootstrap_delta,
)

CONTRACT = "sequence_track_prop_score_v1"
RAW_SIMS_FILENAME = "raw_sims.jsonl"
PROVENANCE_FILENAME = "arm_provenance.json"
DEFAULT_OUT_NAME = "props.json"
CORPUS_ROLE = "prop_fair_baseline_corpus_v2"

BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 29

# Raw per-simulation innings fields, and the families that need each.
RAW_BATTER_RUNS = "innings[].batter_runs"
RAW_BOWLER_WICKETS = "innings[].bowler_wickets"
RAW_TOTAL_RUNS = "innings[].total_runs"
RAW_POWERPLAY = "innings[].phase_runs.powerplay"

FAMILY_REQUIRES = {
    "top_batter": RAW_BATTER_RUNS,
    "batter_50plus": RAW_BATTER_RUNS,
    "batter_runs_mae": RAW_BATTER_RUNS,
    "team_highest_individual_ou_29_5": RAW_BATTER_RUNS,
    "team_highest_individual_ou_34_5": RAW_BATTER_RUNS,
    "team_highest_individual_ou_39_5": RAW_BATTER_RUNS,
    "highest_individual_mae": RAW_BATTER_RUNS,
    "bowler_wkts_1plus": RAW_BOWLER_WICKETS,
    "bowler_wkts_2plus": RAW_BOWLER_WICKETS,
    "bowler_wkts_3plus": RAW_BOWLER_WICKETS,
    "innings_runs_ou_160_5": RAW_TOTAL_RUNS,
    "innings_runs_ou_170_5": RAW_TOTAL_RUNS,
    "innings_runs_ou_180_5": RAW_TOTAL_RUNS,
    "pp_total_ou_45_5": RAW_POWERPLAY,
    "pp_total_ou_50_5": RAW_POWERPLAY,
    "pp_total_ou_55_5": RAW_POWERPLAY,
}

# Families whose inputs `raw_sims.jsonl` does not carry at all, with the
# exact per-simulation quantity that is missing. `top_bowler` is listed
# here because the family's most-wickets-then-fewest-runs tiebreak needs
# runs conceded; `build_observations` would otherwise emit one fabricated
# p=0 row per rostered player, so those rows are dropped explicitly.
MISSING_FIELD = {
    "top_bowler": (
        "per-simulation per-bowler runs conceded (bowling_card index 1), "
        "needed for the most-wickets-then-fewest-runs tiebreak"),
    "batter_6plus_six": "per-simulation per-batter sixes (batting_card index 3)",
    "batter_fours_1plus": "per-simulation per-batter fours (batting_card index 2)",
    "batter_fours_2plus": "per-simulation per-batter fours (batting_card index 2)",
    "batter_fours_3plus": "per-simulation per-batter fours (batting_card index 2)",
    "batter_fours_mae": "per-simulation per-batter fours (batting_card index 2)",
    "team_total_fours_mae": "per-simulation per-team fours (batting_card index 2)",
    "team_total_sixes_mae": "per-simulation per-team sixes (batting_card index 3)",
    "match_total_sixes_ou_15_5": "per-simulation per-team sixes (batting_card index 3)",
    "match_total_sixes_ou_20_5": "per-simulation per-team sixes (batting_card index 3)",
    "team_first_over_mae": "per-simulation first-over runs (the innings ball log)",
    "first_wicket_runs_ou_30_5": (
        "per-simulation runs before the first wicket (the innings ball log)"),
    "highest_over_runs_ou_18_5": (
        "per-simulation maximum single-over runs (the innings ball log)"),
    "highest_over_runs_ou_24_5": (
        "per-simulation maximum single-over runs (the innings ball log)"),
    "bowler_economy_ou_8_5": (
        "per-simulation per-bowler runs conceded and balls bowled "
        "(bowling_card indices 1 and 0); prop_fair_baselines also defines "
        "no fair baseline for this family"),
    "bowler_economy_ou_10_5": (
        "per-simulation per-bowler runs conceded and balls bowled "
        "(bowling_card indices 1 and 0); prop_fair_baselines also defines "
        "no fair baseline for this family"),
}

# Inputs present, but `prop_fair_baselines` defines no fair baseline.
NO_BASELINE = {
    "p_tie": ("prop_fair_baselines defines no fair baseline for this "
              "family (degenerate: ties are ~0.4% of matches)"),
}

# `aggregate_per_player` derives these from the ball log / full cards,
# which the rebuilt per-simulation records do not carry. They are emptied
# so no family can read a fabricated zero out of them.
UNAVAILABLE_AGG_KEYS = (
    "batter_balls", "batter_fours", "batter_sixes",
    "bowler_runs", "bowler_balls",
    "team_fours", "team_sixes", "team_first_over_runs",
    "team_first_wicket_runs", "match_total_sixes", "highest_over_runs",
    "top_bowler_prob",
)


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
# dir. The same guard is spelled identically in `score_realism.py`.
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


def void_reason(info: dict):
    """D/L-shortened or no-result fixtures, per `prop_backtest.main`."""
    outcome = (info or {}).get("outcome", {}) or {}
    if outcome.get("method"):
        return f"outcome.method={outcome['method']}"
    if str(outcome.get("result", "")).lower() == "no result":
        return "outcome.result=no result"
    return None


def raw_fields_present(raw_rows: list[dict]) -> set:
    """Which per-simulation innings quantities every simulation carries."""
    candidates = {
        RAW_BATTER_RUNS: lambda inn: "batter_runs" in inn,
        RAW_BOWLER_WICKETS: lambda inn: "bowler_wickets" in inn,
        RAW_TOTAL_RUNS: lambda inn: "total_runs" in inn,
        RAW_POWERPLAY: lambda inn: "powerplay" in (inn.get("phase_runs") or {}),
    }
    present = set(candidates)
    seen_any = False
    for row in raw_rows:
        for simulation in row.get("simulations") or []:
            for innings in simulation.get("innings") or []:
                seen_any = True
                for field, probe in list(candidates.items()):
                    if field in present and not probe(innings):
                        present.discard(field)
    return present if seen_any else set()


# ---------------------------------------------------------------------------
# Rebuilding per-simulation records the prop aggregator understands
# ---------------------------------------------------------------------------

def player_index_by_team(match_state) -> dict:
    """team -> {player_id: batting-order index}, the run_arm card keys."""
    index = {}
    for lineup in (match_state.team1_lineup, match_state.team2_lineup):
        index[lineup.team_name] = {
            str(player.player_id): position
            for position, player in enumerate(lineup.players)
        }
    return index


def rebuild_sim_results(row: dict, match_state, present: set) -> list:
    """Per-simulation stand-ins carrying only what the raw file records.

    `aggregate_per_player` reads `result.innings[].{batting_team,
    bowling_team, batting_card, bowling_card, total_runs, balls}`. The
    card slots the raw file does not carry (balls/fours/sixes for
    batters, balls/runs for bowlers) are filled with zeros and the ball
    log is empty; every aggregate derived from them is emptied by
    `sim_aggregate` immediately afterwards, so nothing downstream can
    read a fabricated value.
    """
    teams = (match_state.team1, match_state.team2)
    index = player_index_by_team(match_state)
    results = []
    for simulation in row["simulations"]:
        innings_list = []
        for innings in simulation["innings"]:
            batting = innings["batting_team"]
            bowling = next(team for team in teams if team != batting)
            batting_card = {}
            if RAW_BATTER_RUNS in present:
                for player_id, runs in (innings.get("batter_runs") or {}).items():
                    position = index.get(batting, {}).get(str(player_id))
                    if position is None:
                        raise SystemExit(
                            f"{row['match_id']}: batter id {player_id} is not "
                            f"in the {batting} lineup — the fixture directory "
                            "does not match the run")
                    batting_card[position] = (int(runs), 0, 0, 0)
            bowling_card = {}
            if RAW_BOWLER_WICKETS in present:
                for player_id, wickets in (
                        innings.get("bowler_wickets") or {}).items():
                    position = index.get(bowling, {}).get(str(player_id))
                    if position is None:
                        raise SystemExit(
                            f"{row['match_id']}: bowler id {player_id} is not "
                            f"in the {bowling} lineup — the fixture directory "
                            "does not match the run")
                    bowling_card[position] = (0, 0, int(wickets))
            innings_list.append(SimpleNamespace(
                batting_team=batting,
                bowling_team=bowling,
                batting_card=batting_card,
                bowling_card=bowling_card,
                total_runs=int(innings["total_runs"]),
                balls=[],
            ))
        results.append(SimpleNamespace(innings=innings_list))
    return results


def sim_aggregate(match_state, row: dict, present: set) -> dict:
    """`aggregate_per_player` over the rebuilt records, then made honest."""
    aggregate = aggregate_per_player(
        match_state, rebuild_sim_results(row, match_state, present))
    for key in UNAVAILABLE_AGG_KEYS:
        aggregate[key] = type(aggregate[key])()
    # Powerplay totals come from the ball log upstream; the raw file
    # records them directly as `phase_runs.powerplay` (same over<6 split),
    # so they are restored rather than left at the emptied ball log's zero.
    powerplay = defaultdict(list)
    if RAW_POWERPLAY in present:
        for simulation in row["simulations"]:
            for innings in simulation["innings"]:
                powerplay[innings["batting_team"]].append(
                    int(innings["phase_runs"]["powerplay"]))
    aggregate["team_pp_runs"] = powerplay
    return aggregate


def strip_unsupported(obs: dict, unsupported) -> dict:
    """Empty every unsupported family; return what was dropped."""
    dropped = {}
    for family in unsupported:
        rows = obs.get(family)
        if rows:
            dropped[family] = len(rows)
        if family in obs:
            obs[family] = []
    return dropped


def build_detail(raw_rows: list[dict], fixture_dir, unsupported) -> dict:
    """Per-fixture prop observations in `prop_backtest` detail format."""
    reject_sealed(fixture_dir)
    fixture_root = Path(fixture_dir).resolve()
    reject_sealed(fixture_root)
    present = raw_fields_present(raw_rows)
    loader = TestMatchLoader()
    detail, voided, dropped_rows = [], [], defaultdict(int)
    for row in raw_rows:
        match_id = str(row.get("match_id"))
        document_path = fixture_document_path(row.get("match_id"),
                                              fixture_root)
        if not document_path.exists():
            raise SystemExit(
                f"fixture JSON not found for {match_id}: {document_path}")
        document = json.loads(document_path.read_text())
        reason = void_reason(document.get("info", {}))
        if reason:
            voided.append({"match_id": match_id, "reason": reason})
            continue
        if not (row.get("simulations") or []):
            voided.append({"match_id": match_id,
                           "reason": "no simulations recorded"})
            continue
        _, match_state = loader.create_match_state(
            document, cricsheet_id=match_id)
        if match_state is None:
            raise SystemExit(
                f"{match_id}: could not rebuild the match state from "
                f"{document_path}")
        observations = build_observations(
            match_id,
            sim_aggregate(match_state, row, present),
            compute_actuals(document),
        )
        for family, count in strip_unsupported(
                observations["obs"], unsupported).items():
            dropped_rows[family] += count
        observations["cricsheet_id"] = match_id
        observations["display_match_id"] = match_state.display_match_id
        observations["match_identity_version"] = (
            match_state.match_identity_version)
        observations["n_sims"] = len(row["simulations"])
        detail.append(observations)
    return {
        "detail": detail,
        "voided": voided,
        "dropped_fabricated_rows": dict(dropped_rows),
        "raw_fields_present": sorted(present),
    }


# ---------------------------------------------------------------------------
# Corpus and scoring
# ---------------------------------------------------------------------------

def load_corpus(path) -> dict:
    reject_sealed(path)
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def venue_match_arity(logs: dict) -> int:
    for rows in logs.get("venue_match", {}).values():
        if rows:
            return len(rows[0])
    return 0


def family_metrics(family: str, rows: list) -> dict:
    """Sim vs fair-baseline metric, paired difference, by-match interval."""
    outcomes = [row["y"] for row in rows]
    if family.endswith("_mae"):
        metric = "mae"
        sim = float(np.mean([abs(row["p_sim"] - row["y"]) for row in rows]))
        base = float(np.mean([abs(row["p_base"] - row["y"]) for row in rows]))

        def paired(row):
            return abs(row["p_sim"] - row["y"]) - abs(row["p_base"] - row["y"])
    else:
        metric = "brier"
        sim = float(np.mean(brier([row["p_sim"] for row in rows], outcomes)))
        base = float(np.mean(brier([row["p_base"] for row in rows], outcomes)))

        def paired(row):
            return ((row["p_sim"] - row["y"]) ** 2
                    - (row["p_base"] - row["y"]) ** 2)

    low, high = cluster_bootstrap_delta(
        rows, paired, n_boot=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED)
    return {
        "metric": metric,
        "n_rows": len(rows),
        "n_matches": len({row["mid"] for row in rows}),
        "sim": sim,
        "baseline": base,
        "delta_sim_minus_baseline": sim - base,
        "delta_ci95": [low, high],
    }


def score(built: dict, logs: dict, requested=None) -> dict:
    """Per-family metrics plus the scored / skipped family inventory."""
    detail = built["detail"]
    if not detail:
        raise SystemExit("no fixture survived curation; nothing to score")

    # The family universe is exactly what `build_observations` defines.
    universe = sorted(detail[0]["obs"])
    present = set(built["raw_fields_present"])
    arity = venue_match_arity(logs)

    skipped = []
    for family in universe:
        if family in MISSING_FIELD:
            skipped.append({"family": family,
                            "reason": "input absent from raw_sims.jsonl",
                            "missing_field": MISSING_FIELD[family]})
        elif family in NO_BASELINE:
            skipped.append({"family": family, "reason": NO_BASELINE[family]})
        elif FAMILY_REQUIRES.get(family) not in present:
            skipped.append({
                "family": family,
                "reason": "input absent from raw_sims.jsonl",
                "missing_field": FAMILY_REQUIRES.get(
                    family, "unregistered input")})
        elif family == "highest_individual_mae" and arity < 4:
            skipped.append({
                "family": family,
                "reason": (
                    "the fair-baseline corpus predates the match-level "
                    "top-score column (venue_match rows carry "
                    f"{arity} fields, not 4)"),
            })
    skipped_families = {row["family"] for row in skipped}

    scoreable = [family for family in universe
                 if family not in skipped_families]
    if requested:
        for family in scoreable:
            if family not in requested:
                skipped.append({"family": family,
                                "reason": "not requested via --families"})
        skipped_families = {row["family"] for row in skipped}
        scoreable = [family for family in scoreable if family in requested]

    # `baseline_rows` reads every family off each match's `obs`; the ones
    # skipped above are emptied first so no unsupported family can reach
    # a baseline (and so a corpus that predates a column cannot raise).
    trimmed = []
    for match in detail:
        copy = dict(match)
        copy["obs"] = {family: rows for family, rows in match["obs"].items()
                       if family in scoreable}
        trimmed.append(copy)
    paired = baseline_rows(trimmed, AsOf(logs))
    # `baseline_rows` emits per-team rows in set-iteration order, which
    # PYTHONHASHSEED randomizes; that reorders the floating-point sums
    # behind each mean and the bootstrap. Sorting here makes this
    # scorer's output byte-reproducible without touching the frozen
    # baseline module (measured drift without it: ~1e-16 relative).
    for rows in paired.values():
        rows.sort(key=lambda row: (str(row["mid"]), float(row["p_base"]),
                                   float(row["p_sim"]), float(row["y"]),
                                   str(row.get("team") or ""),
                                   str(row.get("name") or "")))

    families = {}
    for family in scoreable:
        rows = paired.get(family) or []
        if not rows:
            skipped.append({
                "family": family,
                "reason": "no paired rows in this fixture cohort",
            })
            continue
        families[family] = family_metrics(family, rows)

    scored_families = sorted(families)
    return {
        "families_scored": scored_families,
        "families_skipped": sorted(
            skipped, key=lambda row: (row["family"], row["reason"])),
        "families": families,
        "family_universe": universe,
        "raw_fields_present": sorted(present),
        "dropped_fabricated_rows": built["dropped_fabricated_rows"],
        "voided_fixtures": built["voided"],
        "n_fixtures_scored": len(detail),
        "baseline_version": BASELINE_VERSION,
        "corpus_venue_match_arity": arity,
        "row_order": (
            "paired rows sorted by (match, baseline, sim, outcome, team, "
            "name) so every mean and bootstrap sums in a fixed order"),
        "bootstrap": {
            "reps": BOOTSTRAP_REPS,
            "seed": BOOTSTRAP_SEED,
            "generator": "numpy.random.default_rng",
            "resample": "by match (cluster)",
        },
    }


def fixture_inventory(built: dict, logs: dict) -> list:
    asof = AsOf(logs)
    rows = []
    for match in built["detail"]:
        display = match.get("display_match_id") or match["match_id"]
        rows.append({
            "match_id": match["match_id"],
            "display_match_id": display,
            "match_identity_version": match.get("match_identity_version"),
            "n_sims": match.get("n_sims"),
            "venue_resolved": _venue_from_match_id(display, asof) is not None,
            "rows_by_family": {family: len(observations)
                               for family, observations
                               in sorted(match["obs"].items())
                               if observations},
        })
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score one stage-1 arm's raw simulations on prop "
                    "families against the fair baselines")
    parser.add_argument("--arm-dir", required=True)
    parser.add_argument("--fixture-dir", default=None,
                        help="directory of cricsheet fixture JSONs (default: "
                             "the fixture dir recorded in "
                             "arm_provenance.json)")
    parser.add_argument("--out", default=None,
                        help=f"output JSON (default: <arm-dir>/"
                             f"{DEFAULT_OUT_NAME})")
    parser.add_argument("--families", default="all",
                        help="'all' (default) or a comma-separated family "
                             "list; families outside the list are skipped")
    parser.add_argument("--corpus", default=None,
                        help=f"fair-baseline corpus pickle (default: the "
                             f"{CORPUS_ROLE} manifest role)")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    # Sealed-data guard first: every path this scorer may read or write is
    # refused before anything is opened, exactly as `run_arm.py` does.
    for candidate in (args.arm_dir, args.fixture_dir, args.out, args.corpus):
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

    corpus_path = Path(artifact_path(CORPUS_ROLE, args.corpus))
    if not corpus_path.is_absolute():
        corpus_path = REPO_ROOT / corpus_path
    reject_sealed(corpus_path)
    if not corpus_path.exists():
        raise SystemExit(f"fair-baseline corpus not found: {corpus_path}")

    requested = None
    if args.families.strip().lower() != "all":
        requested = {name.strip() for name in args.families.split(",")
                     if name.strip()}

    built = build_detail(load_jsonl(raw_path), fixture_dir,
                         set(MISSING_FIELD))
    logs = load_corpus(corpus_path)
    payload = score(built, logs, requested)
    payload["fixtures"] = fixture_inventory(built, logs)
    payload["contract"] = CONTRACT
    payload["arm_dir"] = str(arm_dir)
    payload["raw_sims"] = str(raw_path)
    payload["fixture_dir"] = str(fixture_dir)
    payload["corpus"] = {
        "role": CORPUS_ROLE if args.corpus is None else None,
        "path": str(corpus_path),
        "md5": md5_file(corpus_path),
    }
    provenance = arm_dir / PROVENANCE_FILENAME
    if provenance.exists():
        run = (json.loads(provenance.read_text()).get("run") or {})
        payload["arm"] = run.get("arm")
        payload["arm_run"] = {
            key: run.get(key) for key in
            ("arm", "model_dir", "model_dir_hash", "stats_version",
             "base_seed", "n_sims", "engine_md5", "runner_md5")
        }

    out_path = Path(args.out) if args.out else arm_dir / DEFAULT_OUT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    print(f"[score_props] fixtures scored: {payload['n_fixtures_scored']}")
    print(f"[score_props] families scored ({len(payload['families_scored'])}): "
          + ", ".join(payload["families_scored"]))
    for row in payload["families_skipped"]:
        detail_text = row.get("missing_field")
        print(f"[score_props] skipped {row['family']}: {row['reason']}"
              + (f" — {detail_text}" if detail_text else ""))
    print(f"[score_props] wrote {out_path}")


if __name__ == "__main__":
    main()
