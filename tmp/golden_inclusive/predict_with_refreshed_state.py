"""Predict a fixture using a refreshed SQLite + tracker snapshot that
INCLUDES golden data (2026-04-17+).

This is a temporary experiment to see how much the staleness caveat
matters for May 10 IPL fixtures. We do NOT retrain the model — only
the feature inputs (per-player ELO/stats from SQLite + form/H2H/home
trackers) are refreshed. The XGBoost booster bytes are unchanged.

Outputs go to tmp/golden_inclusive/predictions/.

Usage:
    uv run python tmp/golden_inclusive/predict_with_refreshed_state.py \\
        --fixture fixtures/2026-05-10_csk_lsg.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

# Re-use the production helpers
from materialize_match_features import (  # noqa: E402
    TeamFormTracker, H2HTracker, HomeVenueTracker,
)
from predict_fixture import (  # noqa: E402
    compute_features, apply_encoders_and_predict, compute_bet,
    _build_name_lookup, _resolve_player_ids,
)
import predict_fixture  # noqa: E402

from loaders_common import iter_matches_chronological  # noqa: E402
from player_metadata import PlayerMetadataProvider  # noqa: E402
from stats_provider import StatsProvider  # noqa: E402


TMP_ROOT = REPO / "tmp" / "golden_inclusive"
TMP_SQLITE_DIR = TMP_ROOT  # StatsProvider expects player_stats_cache_v3.sqlite here
TMP_SNAPSHOT = TMP_ROOT / "tracker_snapshot_combined.pkl"
MODEL_DIR = REPO / "models" / "xgb_match_v3_m7_production"  # current production model


def build_combined_snapshot() -> dict:
    """Walk both data/t20s_json AND data/golden/t20s_json chronologically
    and build a single Phase A2 tracker snapshot.
    """
    print(f"Building combined Phase A2 tracker snapshot "
          f"(t20s_json + golden/t20s_json)...")
    t0 = time.time()
    form = TeamFormTracker()
    h2h = H2HTracker()
    home = HomeVenueTracker(lookback_days=730)

    pools = [
        REPO / "data" / "t20s_json",
        REPO / "data" / "golden" / "t20s_json",
        REPO / "tmp" / "golden_inclusive" / "v2_extras_post_may7",
    ]
    # Merge in date order across both pools (mirrors
    # iter_matches_chronological_multi from materialize_match_features.py).
    streams = [iter_matches_chronological(str(p), gender="male") for p in pools]
    tagged = [((d, mid), (mid, txt, d)) for s in streams for (mid, txt, d) in s]
    tagged.sort(key=lambda t: t[0])

    n = 0
    latest = None
    for _, (mid, json_text, match_date) in tagged:
        n += 1
        latest = match_date if latest is None or match_date > latest else latest
        data = json.loads(json_text)
        info = data.get("info") or {}
        teams = info.get("teams") or []
        if len(teams) != 2:
            continue
        outcome = info.get("outcome") or {}
        winner = outcome.get("winner")
        if not winner and outcome.get("result") == "tie":
            winner = outcome.get("eliminator")
        if not winner or winner not in teams:
            continue
        venue = info.get("venue", "unknown")
        t1, t2 = teams
        t1_won = winner == t1
        form.update(t1, match_date, t1_won)
        form.update(t2, match_date, not t1_won)
        h2h.update(t1, t2, match_date, winner)
        home.update(t1, venue, match_date)
        home.update(t2, venue, match_date)

    snapshot = {
        "as_of": latest.strftime("%Y-%m-%d") if latest else None,
        "form_records": dict(form.records),
        "h2h_records": {tuple(sorted(k)): v for k, v in h2h.records.items()},
        "home_records": dict(home.records),
        "n_matches_walked": n,
        "built_at": datetime.utcnow().isoformat() + "Z",
    }
    TMP_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    with open(TMP_SNAPSHOT, "wb") as f:
        pickle.dump(snapshot, f)
    print(f"  walked {n} matches in {time.time()-t0:.1f}s -> {TMP_SNAPSHOT}")
    print(f"  snapshot as_of = {snapshot['as_of']}")
    return snapshot


def load_combined_trackers():
    if not TMP_SNAPSHOT.exists():
        build_combined_snapshot()
    with open(TMP_SNAPSHOT, "rb") as f:
        snap = pickle.load(f)

    form = TeamFormTracker()
    for team, recs in snap["form_records"].items():
        form.records[team] = list(recs)
    h2h = H2HTracker()
    for k_tuple, recs in snap["h2h_records"].items():
        h2h.records[frozenset(k_tuple)] = list(recs)
    home = HomeVenueTracker(lookback_days=730)
    for k, recs in snap["home_records"].items():
        home.records[k] = list(recs)
    return form, h2h, home, snap.get("as_of")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--rebuild-snapshot", action="store_true")
    args = ap.parse_args()

    # Pin the predict_fixture module's MODEL_DIR so apply_encoders_and_predict
    # uses the same trained model. (We're not retraining.)
    predict_fixture.MODEL_DIR = MODEL_DIR

    if args.rebuild_snapshot and TMP_SNAPSHOT.exists():
        TMP_SNAPSHOT.unlink()

    fixture = json.loads(args.fixture.read_text())
    print(f"Predicting (REFRESHED state): {fixture['date']}  "
          f"{fixture['team1']} vs {fixture['team2']}  @ {fixture['venue']}")

    print("Loading providers...")
    provider = StatsProvider(str(TMP_SQLITE_DIR), version="v3")
    metadata = PlayerMetadataProvider(str(REPO / "data" / "all_players_enriched.csv"))
    form, h2h, home, snap_as_of = load_combined_trackers()

    print(f"Tracker snapshot as_of: {snap_as_of}")

    print("Computing features...")
    record = compute_features(fixture, provider, metadata, form, h2h, home)

    print("Applying model...")
    p_team1, debug = apply_encoders_and_predict(record)
    p_team2 = 1.0 - p_team1

    bet_info = compute_bet(fixture["team1"], fixture["team2"], p_team1,
                           fixture.get("polymarket_odds"))

    output = {
        "fixture": {k: v for k, v in fixture.items()
                    if k not in ("team1_lineup", "team2_lineup")},
        "fixture_lineups": {
            "team1": fixture["team1_lineup"],
            "team2": fixture["team2_lineup"],
        },
        "prediction": {
            fixture["team1"]: p_team1,
            fixture["team2"]: p_team2,
        },
        "bet": bet_info,
        "diagnostics": {
            "model": str(MODEL_DIR),
            "rehydrate_as_of": fixture["date"],
            "tracker_snapshot_as_of": snap_as_of,
            "sqlite_cache": str(TMP_SQLITE_DIR / "player_stats_cache_v3.sqlite"),
            "encoder_warnings": debug["encoder_warnings"],
            "h2h_n_meetings": record["h2h_n_meetings"],
            "is_team1_home": record["is_team1_home"],
            "is_team2_home": record["is_team2_home"],
            "team1_win_rate_last_10": record["team1_win_rate_last_10"],
            "team2_win_rate_last_10": record["team2_win_rate_last_10"],
            "top6_batting_elo_diff": record["top6_batting_elo_diff"],
            "bottom5_bowling_elo_diff": record["bottom5_bowling_elo_diff"],
            "elo_diff_batting": record["elo_diff_batting"],
            "elo_diff_bowling": record["elo_diff_bowling"],
        },
        "feature_row": {
            c: (float(debug["feature_row"][c])
                if c not in ("venue", "competition_tier")
                else debug["feature_row"][c])
            for c in debug["feature_columns"]
        },
    }

    out_path = args.out or (TMP_ROOT / "predictions" /
                            f"{fixture['date']}_"
                            f"{fixture['team1'].replace(' ', '_')}_vs_"
                            f"{fixture['team2'].replace(' ', '_')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))

    print()
    print(f"  P({fixture['team1']:<35s} wins) = {p_team1*100:>5.1f}%")
    print(f"  P({fixture['team2']:<35s} wins) = {p_team2*100:>5.1f}%")
    print(f"  Output -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
