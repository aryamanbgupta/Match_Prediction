"""Score completed 100-ball (The Hundred) matches with the match-level model.

The production winner model is trained on men's T20 only. The Hundred is a
different format (100 balls, 5-ball overs) whose franchises never appear in
the training corpus, so nothing here is a claim of validated edge — it is the
measurement that tells us whether the T20 model transfers at all.

Every match is scored through the same `compute_features` /
`apply_encoders_and_predict` path that `predict_fixture.py` uses live, so a
backtest number and a live number mean the same thing. State is queried
as-of each match date, so a 2022 fixture only ever sees pre-2022 state.

Usage:
    uv run python scripts/backtest_hundred.py \
        --source-dir data/hundred/context_hnd_json \
        --source-dir data/hundred/season_2026_men \
        --state-dir data/forward_state/2026-06-01_2026-07-13 \
        --tracker-snapshot tmp/live_state/tracker_snapshot_2026-07-13.pkl \
        --out-json eval_out/hundred/backtest.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from loaders_common import iter_matches_chronological_multi  # noqa: E402
from player_metadata import PlayerMetadataProvider  # noqa: E402
from stats_provider import StatsProvider  # noqa: E402
from predict_fixture import (  # noqa: E402
    apply_encoders_and_predict,
    compute_features,
    load_trackers,
    build_tracker_snapshot,
    MODEL_DIR,
    VENUE_IDENTITY_I7,
    VENUE_IDENTITY_MODES,
)

DEFAULT_SOURCES = (
    REPO / "data" / "hundred" / "context_hnd_json",
    REPO / "data" / "hundred" / "season_2026_men",
)
DEFAULT_STATE_DIR = REPO / "data" / "forward_state" / "2026-06-01_2026-07-13"
DEFAULT_TRACKER = REPO / "tmp" / "live_state" / "tracker_snapshot_2026-07-13.pkl"
DEFAULT_EVENT_LABEL = "Vitality Blast"


def match_to_fixture(data: dict, event_label: str) -> dict | None:
    """Turn a Cricsheet-shaped record into a `predict_fixture` fixture dict.

    Returns None for matches with no result (abandoned / tied without an
    eliminator), which carry no label to score against.
    """
    info = data.get("info") or {}
    teams = info.get("teams") or []
    if len(teams) != 2:
        return None
    outcome = info.get("outcome") or {}
    winner = outcome.get("winner") or (
        outcome.get("eliminator") if outcome.get("result") == "tie" else None
    )
    if not winner or winner not in teams:
        return None

    registry = ((info.get("registry") or {}).get("people")) or {}
    lineups = []
    for team in teams:
        names = (info.get("players") or {}).get(team) or []
        # Prefer Cricsheet's own name -> id registry; fall back to the raw
        # name so `predict_fixture`'s CSV name resolution can try.
        lineups.append([registry.get(name, name) for name in names])
    if min(len(lu) for lu in lineups) < 7:
        return None

    toss = info.get("toss") or {}
    return {
        "date": info["dates"][0],
        "team1": teams[0],
        "team2": teams[1],
        "venue": info.get("venue"),
        "competition_tier": event_label,
        "team_type": info.get("team_type", "club"),
        "team1_lineup": lineups[0],
        "team2_lineup": lineups[1],
        "toss_winner": toss.get("winner"),
        "toss_decision": toss.get("decision"),
        "_winner": winner,
        "_event": (info.get("event") or {}).get("name"),
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    p = np.array([r["p_team1"] for r in rows])
    y = np.array([1.0 if r["winner"] == r["team1"] else 0.0 for r in rows])
    eps = 1e-15
    ll = float(-np.mean(y * np.log(np.clip(p, eps, 1)) +
                        (1 - y) * np.log(np.clip(1 - p, eps, 1))))
    brier = float(np.mean((p - y) ** 2))
    picked = (p >= 0.5).astype(float)
    acc = float(np.mean(picked == y))
    # Baselines that need no model at all.
    home_acc = float(np.mean(y))  # team1 is the home side by convention
    elo = np.array([r["top6_batting_elo_diff"] for r in rows])
    elo_pick = (elo >= 0).astype(float)
    elo_acc = float(np.mean(elo_pick == y))
    return {
        "n": len(rows),
        "log_loss": ll,
        "brier": brier,
        "accuracy": acc,
        "coinflip_log_loss": float(-math.log(0.5)),
        "home_team_win_rate": home_acc,
        "top6_elo_favourite_accuracy": elo_acc,
        "mean_abs_edge_from_half": float(np.mean(np.abs(p - 0.5))),
        "max_prob": float(p.max()),
        "min_prob": float(p.min()),
    }


def calibration_table(rows: list[dict], n_bins: int = 5) -> list[dict]:
    """Predicted-vs-observed by probability bin, oriented on the model's pick."""
    out = []
    conf = np.array([max(r["p_team1"], 1 - r["p_team1"]) for r in rows])
    hit = np.array([
        1.0 if ((r["p_team1"] >= 0.5) == (r["winner"] == r["team1"])) else 0.0
        for r in rows
    ])
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if not mask.any():
            continue
        out.append({
            "bin": f"{lo:.2f}-{hi:.2f}",
            "n": int(mask.sum()),
            "mean_predicted": float(conf[mask].mean()),
            "observed": float(hit[mask].mean()),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", type=Path, action="append",
                    dest="source_dirs",
                    help="Directory of Cricsheet-shaped matches; repeatable")
    ap.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    ap.add_argument("--state-version", default="v3",
                    help="Stats-cache version suffix inside --state-dir "
                         "(e.g. 'i7' for player_stats_cache_i7.sqlite)")
    ap.add_argument("--venue-identity-mode", choices=VENUE_IDENTITY_MODES,
                    default=VENUE_IDENTITY_I7,
                    help="Identity contract shared by model, state, tracker, "
                         "and fixture (default: i7)")
    ap.add_argument("--tracker-snapshot", type=Path, default=DEFAULT_TRACKER)
    ap.add_argument("--tracker-source-dir", type=Path, action="append",
                    dest="tracker_source_dirs")
    ap.add_argument("--tracker-aux-dir", type=Path, action="append",
                    dest="tracker_aux_dirs",
                    help="Competition pool that feeds form/H2H/home trackers "
                         "without counting toward the SQLite state-source "
                         "contract; repeatable")
    ap.add_argument("--rebuild-snapshot", action="store_true")
    ap.add_argument("--team-aliases", type=Path, default=None,
                    help="JSON map of historical team name -> current name")
    ap.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    ap.add_argument("--event-label", default=DEFAULT_EVENT_LABEL,
                    help="Event name handed to classify_match_context, which "
                         "sets competition_tier. The Hundred is absent from "
                         "both league lists, so the closest in-corpus analogue "
                         "is used by default (Vitality Blast -> tier 2).")
    ap.add_argument("--hide-toss", action="store_true",
                    help="Score as if pre-toss (average both bat-first "
                         "branches), matching live fixture conditions")
    ap.add_argument("--min-date", default=None)
    ap.add_argument("--max-date", default=None)
    ap.add_argument("--out-json", type=Path,
                    default=REPO / "eval_out" / "hundred" / "backtest.json")
    ap.add_argument("--label", default=None,
                    help="Tag recorded in the output for A/B runs")
    args = ap.parse_args()

    sources = tuple(args.source_dirs or DEFAULT_SOURCES)
    tracker_sources = tuple(args.tracker_source_dirs or sources)
    tracker_aux_sources = tuple(args.tracker_aux_dirs or ())
    aliases = {}
    if args.team_aliases:
        aliases = json.loads(args.team_aliases.read_text())["aliases"]
    if args.rebuild_snapshot:
        build_tracker_snapshot(
            tracker_sources,
            args.tracker_snapshot,
            tracker_aux_sources,
            identity_mode=args.venue_identity_mode,
        )

    print(f"State:   {args.state_dir}")
    print(f"Tracker: {args.tracker_snapshot}")
    provider = StatsProvider(str(args.state_dir), version=args.state_version)
    metadata = PlayerMetadataProvider(
        str(REPO / "data" / "all_players_enriched.csv"))
    form, h2h, home = load_trackers(
        args.tracker_snapshot,
        tracker_sources,
        aux_source_dirs=tracker_aux_sources,
        team_aliases=aliases,
        identity_mode=args.venue_identity_mode,
    )
    if aliases:
        print(f"Applied {len(aliases)} franchise renames to tracker state")

    rows, skipped = [], 0
    for match_id, json_text, match_date in iter_matches_chronological_multi(
            sources, gender="male"):
        date_str = match_date.strftime("%Y-%m-%d")
        if args.min_date and date_str < args.min_date:
            continue
        if args.max_date and date_str > args.max_date:
            continue
        fixture = match_to_fixture(json.loads(json_text), args.event_label)
        if fixture is None:
            skipped += 1
            continue
        if args.hide_toss:
            # Match live pre-toss conditions: the toss is unknown, so
            # average both bat-first branches exactly as predict_fixture does.
            fixture["toss_winner"] = None
            fixture["toss_decision"] = None
        try:
            record = compute_features(fixture, provider, metadata,
                                      form, h2h, home,
                                      identity_mode=args.venue_identity_mode)
        except ValueError as exc:
            # An unresolvable XI is a data problem with one match, not a
            # reason to abandon the run.
            print(f"  SKIP {match_id} ({date_str}): {exc}")
            skipped += 1
            continue
        toss_known = bool(record.pop("_toss_known", True))
        if toss_known:
            p_team1, debug = apply_encoders_and_predict(
                record, args.model_dir,
                identity_mode=args.venue_identity_mode)
        else:
            branch = dict(record)
            branch["team1_batting_first"] = 1
            p_bat, debug = apply_encoders_and_predict(
                branch, args.model_dir,
                identity_mode=args.venue_identity_mode)
            branch["team1_batting_first"] = 0
            p_chase, _ = apply_encoders_and_predict(
                branch, args.model_dir,
                identity_mode=args.venue_identity_mode)
            p_team1 = 0.5 * (p_bat + p_chase)
        rows.append({
            "match_id": match_id,
            "date": date_str,
            "season": date_str[:4],
            "venue": fixture["venue"],
            "team1": fixture["team1"],
            "team2": fixture["team2"],
            "winner": fixture["_winner"],
            "p_team1": p_team1,
            "top6_batting_elo_diff": record["top6_batting_elo_diff"],
            "team1_win_rate_last_10": record["team1_win_rate_last_10"],
            "team2_win_rate_last_10": record["team2_win_rate_last_10"],
            "h2h_n_meetings": record["h2h_n_meetings"],
            "is_team1_home": record["is_team1_home"],
            "is_team2_home": record["is_team2_home"],
            "encoder_warnings": debug["encoder_warnings"],
        })
        if len(rows) % 25 == 0:
            print(f"  scored {len(rows)} matches...")

    print(f"\nScored {len(rows)} matches ({skipped} skipped: no result / short lineup)")

    by_season = defaultdict(list)
    for row in rows:
        by_season[row["season"]].append(row)

    out = {
        "config": {
            "sources": [str(p) for p in sources],
            "state_dir": str(args.state_dir),
            "venue_identity_mode": args.venue_identity_mode,
            "tracker_snapshot": str(args.tracker_snapshot),
            "tracker_sources": [str(p) for p in tracker_sources],
            "tracker_aux_sources": [str(p) for p in tracker_aux_sources],
            "model_dir": str(args.model_dir),
            "event_label": args.event_label,
            "team_aliases": aliases,
            "label": args.label,
        },
        "overall": summarize(rows),
        "by_season": {s: summarize(r) for s, r in sorted(by_season.items())},
        "calibration": calibration_table(rows),
        "matches": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2))

    o = out["overall"]
    print(f"\n{'':<8} {'n':>4} {'LL':>8} {'Brier':>8} {'Acc':>7} {'|p-.5|':>8}")
    print(f"{'ALL':<8} {o['n']:>4} {o['log_loss']:>8.4f} {o['brier']:>8.4f} "
          f"{o['accuracy']*100:>6.1f}% {o['mean_abs_edge_from_half']:>8.4f}")
    for season, stats in out["by_season"].items():
        print(f"{season:<8} {stats['n']:>4} {stats['log_loss']:>8.4f} "
              f"{stats['brier']:>8.4f} {stats['accuracy']*100:>6.1f}% "
              f"{stats['mean_abs_edge_from_half']:>8.4f}")
    print(f"\nCoinflip LL {o['coinflip_log_loss']:.4f} | "
          f"home-team win rate {o['home_team_win_rate']*100:.1f}% | "
          f"top6-ELO-favourite accuracy {o['top6_elo_favourite_accuracy']*100:.1f}%")
    print(f"\nWrote {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
