"""Predict a single upcoming fixture without going through the full
materialization pipeline.

Default model: xgb_match_v3_m7_production. Override with --model-dir.
Per-fixture inference rehydrates SQLite + tracker queries as of the fixture
date. The CLI inspects both state sources before loading the model and blocks
silently stale or internally mismatched state. Refreshed state must live
outside the frozen production model directory and can be selected explicitly.

Usage:
    uv run python scripts/predict_fixture.py --fixture fixtures/<id>.json
    uv run python scripts/predict_fixture.py --fixture <path> --out predictions/<id>.json
    uv run python scripts/predict_fixture.py --fixture <path> --model-dir models/<other>
    uv run python scripts/predict_fixture.py --fixture <path> \
        --state-dir data/forward_state/2026-06-01_2026-07-13 \
        --tracker-snapshot tmp/live_state/tracker_snapshot.pkl \
        --tracker-source-dir data/t20s_json \
        --tracker-source-dir \
          data/forward_holdout/2026-06-01_2026-07-13/context_t20s_json \
        --rebuild-snapshot
"""
from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
import sys
import time
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from materialize_match_features import (  # noqa: E402
    TeamFormTracker, H2HTracker, HomeVenueTracker,
    _lineup_mix_counts, _split_elo, FEATURE_COLUMNS,
)
from parsing_v2 import classify_match_context  # noqa: E402
from loaders_common import (  # noqa: E402
    SAME_DAY_ORDER_VERSION,
    iter_matches_chronological_multi,
)
from player_metadata import PlayerMetadataProvider  # noqa: E402
from stats_provider import StatsProvider  # noqa: E402
from tracker_rehydration import (  # noqa: E402
    rehydrate_elo_tracker,
    rehydrate_venue_tracker,
)

MODEL_DIR = REPO / "models" / "xgb_match_v3_m7_production"
TRACKER_SNAPSHOT = REPO / "data" / "tracker_snapshot_test_end.pkl"
DEFAULT_STATE_DIR = REPO / "models"
DEFAULT_TRACKER_SOURCE_DIRS = (REPO / "data" / "t20s_json",)
DEFAULT_MAX_STATE_AGE_DAYS = 14
EDGE_THRESHOLD = 0.0


def _normalize_source_dirs(
    source_dirs: Sequence[Path | str] | Path | str,
) -> tuple[Path, ...]:
    if isinstance(source_dirs, (str, Path)):
        source_dirs = (source_dirs,)
    normalized = tuple(Path(p).resolve() for p in source_dirs)
    if not normalized:
        raise ValueError("at least one tracker source directory is required")
    missing = [str(p) for p in normalized if not p.is_dir()]
    if missing:
        raise FileNotFoundError(
            "tracker source directories do not exist: " + ", ".join(missing)
        )
    return normalized


def build_tracker_snapshot(
    source_dirs: Sequence[Path | str] | Path | str = DEFAULT_TRACKER_SOURCE_DIRS,
    snapshot_path: Path = TRACKER_SNAPSHOT,
) -> dict:
    """Walk every match in the corpus and snapshot the three Phase A2
    trackers.

    Multiple source pools are merged in the versioned I6
    ``(date, Cricsheet ID)`` order and duplicate match IDs fail closed.
    Records are filtered by query date at lookup time, so one snapshot can
    safely serve earlier fixture-date queries too.
    """
    normalized_dirs = _normalize_source_dirs(source_dirs)
    print(f"Building Phase A2 tracker snapshot from "
          f"{', '.join(str(p) for p in normalized_dirs)} "
          f"(one-time, ~30s)...")
    t0 = time.time()
    form = TeamFormTracker()
    h2h = H2HTracker()
    home = HomeVenueTracker(lookback_days=730)

    n = 0
    latest = None
    for mid, json_text, match_date in iter_matches_chronological_multi(
            normalized_dirs, gender="male"):
        n += 1
        latest = match_date if latest is None or match_date > latest else latest
        data = json.loads(json_text)
        info = data.get("info") or {}
        teams = info.get("teams") or []
        if len(teams) != 2:
            continue
        outcome = info.get("outcome") or {}
        winner = outcome.get("winner")
        # Carry tied super-overs by promoting the eliminator (the
        # original materializer doesn't, but for a tracker snapshot
        # losing the result is strictly worse than logging it; the
        # match is in the books either way).
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
        "source_dirs": [str(p) for p in normalized_dirs],
        "same_day_order_version": SAME_DAY_ORDER_VERSION,
    }
    snapshot_path = Path(snapshot_path)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    with open(snapshot_path, "wb") as f:
        pickle.dump(snapshot, f)
    dt = time.time() - t0
    print(f"  walked {n} matches in {dt:.1f}s -> {snapshot_path}")
    return snapshot


def _read_tracker_snapshot(snapshot_path: Path = TRACKER_SNAPSHOT) -> dict:
    if not Path(snapshot_path).exists():
        raise FileNotFoundError(f"tracker snapshot not found: {snapshot_path}")
    with open(snapshot_path, "rb") as f:
        snapshot = pickle.load(f)
    if not isinstance(snapshot, dict):
        raise RuntimeError(f"tracker snapshot is not a dict: {snapshot_path}")
    return snapshot


def _peek_snapshot_as_of(snapshot_path: Path = TRACKER_SNAPSHOT) -> str | None:
    """Return the tracker snapshot's `as_of` field for diagnostics."""
    if not Path(snapshot_path).exists():
        return None
    return _read_tracker_snapshot(snapshot_path).get("as_of")


def load_trackers(
    snapshot_path: Path = TRACKER_SNAPSHOT,
    source_dirs: Sequence[Path | str] | Path | str = DEFAULT_TRACKER_SOURCE_DIRS,
) -> tuple[TeamFormTracker, H2HTracker, HomeVenueTracker]:
    if not Path(snapshot_path).exists():
        build_tracker_snapshot(source_dirs, snapshot_path)
    snap = _read_tracker_snapshot(snapshot_path)

    form = TeamFormTracker()
    for team, recs in snap["form_records"].items():
        form.records[team] = list(recs)
    h2h = H2HTracker()
    for k_tuple, recs in snap["h2h_records"].items():
        h2h.records[frozenset(k_tuple)] = list(recs)
    home = HomeVenueTracker(lookback_days=730)
    for k, recs in snap["home_records"].items():
        home.records[k] = list(recs)
    return form, h2h, home


def read_sqlite_state_metadata(
    state_dir: Path,
    version: str = "v3",
) -> dict:
    """Read live-state coverage without mutating or fully loading the cache."""
    sqlite_path = Path(state_dir) / f"player_stats_cache_{version}.sqlite"
    if not sqlite_path.exists():
        raise FileNotFoundError(
            f"{sqlite_path} not found; build a separate refreshed cache first"
        )
    uri = f"{sqlite_path.resolve().as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        row = conn.execute("SELECT MAX(date) FROM dates").fetchone()
        meta = dict(conn.execute("SELECT key, value FROM _meta"))
    as_of = row[0] if row else None
    if not as_of:
        raise RuntimeError(f"SQLite cache has no dated state: {sqlite_path}")
    source_count = meta.get("source_match_count")
    return {
        "path": str(sqlite_path.resolve()),
        "as_of": as_of,
        "source_match_count": int(source_count) if source_count else None,
        "build_timestamp": meta.get("build_timestamp"),
        "same_day_order_version": meta.get("same_day_order_version"),
    }


def read_tracker_state_metadata(snapshot_path: Path) -> dict:
    snapshot = _read_tracker_snapshot(snapshot_path)
    as_of = snapshot.get("as_of")
    if not as_of:
        raise RuntimeError(
            f"tracker snapshot has no as_of date: {snapshot_path}"
        )
    count = snapshot.get("n_matches_walked")
    return {
        "path": str(Path(snapshot_path).resolve()),
        "as_of": as_of,
        "source_match_count": int(count) if count is not None else None,
        "built_at": snapshot.get("built_at"),
        "source_dirs": snapshot.get("source_dirs"),
        "same_day_order_version": snapshot.get("same_day_order_version"),
    }


def assess_state_freshness(
    fixture_date: str,
    sqlite_state: dict,
    tracker_state: dict,
    max_state_age_days: int = DEFAULT_MAX_STATE_AGE_DAYS,
) -> dict:
    """Return a fail-closed freshness/consistency assessment.

    State newer than the fixture is safe because every lookup is filtered by
    fixture date. State older than the fixture is allowed only within the
    explicit age budget.
    """
    if max_state_age_days < 0:
        raise ValueError("max_state_age_days must be non-negative")
    try:
        fixture_day = date.fromisoformat(fixture_date)
        sqlite_day = date.fromisoformat(str(sqlite_state["as_of"]))
        tracker_day = date.fromisoformat(str(tracker_state["as_of"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid state or fixture date: {exc}") from exc

    sqlite_count = sqlite_state.get("source_match_count")
    tracker_count = tracker_state.get("source_match_count")
    if (sqlite_count is not None and tracker_count is not None
            and sqlite_count != tracker_count):
        raise RuntimeError(
            "SQLite/tracker source-count mismatch: "
            f"SQLite={sqlite_count}, tracker={tracker_count}. Rebuild both "
            "from the same ordered source directories."
        )

    sqlite_age = max(0, (fixture_day - sqlite_day).days)
    tracker_age = max(0, (fixture_day - tracker_day).days)
    effective_day = min(sqlite_day, tracker_day)
    effective_age = max(sqlite_age, tracker_age)
    stale = effective_age > max_state_age_days
    return {
        "status": "stale" if stale else "fresh",
        "fixture_date": fixture_date,
        "effective_as_of": effective_day.isoformat(),
        "age_days": effective_age,
        "max_state_age_days": max_state_age_days,
        "sqlite": {**sqlite_state, "age_days": sqlite_age},
        "tracker": {**tracker_state, "age_days": tracker_age},
    }


_NAME_TO_ID_CACHE: dict[str, str] | None = None


def _build_name_lookup(metadata: PlayerMetadataProvider) -> dict[str, str]:
    """Build name -> cricsheet_id lookup from the metadata provider's
    in-memory `players` dict. Memoized for the process lifetime.
    """
    global _NAME_TO_ID_CACHE
    if _NAME_TO_ID_CACHE is not None:
        return _NAME_TO_ID_CACHE
    out: dict[str, str] = {}
    for pid, meta in metadata.players.items():
        for key in (meta.get("name"), meta.get("full_name")):
            if not key or not isinstance(key, str):
                continue
            k = key.strip()
            if k and k not in out:
                out[k] = pid
            kl = k.lower()
            if kl and kl not in out:
                out[kl] = pid
    _NAME_TO_ID_CACHE = out
    return out


def _resolve_player_ids(lineup: list[str],
                        metadata: PlayerMetadataProvider) -> list[str]:
    """Lineup entries may be cricsheet IDs (8-char alphanumeric) OR
    display names. Names are resolved via the enriched-player CSV.
    Unresolved entries are kept as-is and the caller is warned.
    """
    name_lookup = _build_name_lookup(metadata)
    resolved = []
    for entry in lineup:
        s = str(entry).strip()
        if s in metadata.players:
            resolved.append(s)
            continue
        pid = name_lookup.get(s) or name_lookup.get(s.lower())
        if pid:
            resolved.append(pid)
        else:
            print(f"  WARN: could not resolve player '{s}' to a cricsheet ID; "
                  f"keeping as-is (lineup features may be inaccurate)")
            resolved.append(s)
    return resolved


def compute_features(fixture: dict,
                     provider: StatsProvider,
                     metadata: PlayerMetadataProvider,
                     form: TeamFormTracker,
                     h2h: H2HTracker,
                     home: HomeVenueTracker) -> dict:
    """Mirror materialize_match_features._build_match_record but compute
    team-level features directly from StatsProvider getters (no
    parse_match_data_v2 / no ball data required)."""
    date_str = fixture["date"]
    match_date = datetime.strptime(date_str, "%Y-%m-%d")
    team1, team2 = fixture["team1"], fixture["team2"]
    venue = fixture["venue"]
    event_name = fixture.get("competition_tier", "Indian Premier League")
    team_type = fixture.get("team_type", "club")
    # Compute the encoded tier (1..4) and is_international the same way
    # classify_match_context does at parse time, so encoder lookup matches.
    ctx = classify_match_context(event_name, team_type, [team1, team2])
    competition_tier_code = str(ctx["competition_tier"])
    is_international = ctx["is_international"]

    team1_lineup = _resolve_player_ids(fixture["team1_lineup"], metadata)
    team2_lineup = _resolve_player_ids(fixture["team2_lineup"], metadata)

    # Lineup-length guard: the top6/bottom5 ELO splits need ≥7 players to be
    # meaningful (2026-07-16 review I2).
    for side, lu in (("team1", team1_lineup), ("team2", team2_lineup)):
        if len(lu) < 7:
            raise ValueError(
                f"{side}_lineup has only {len(lu)} players; need ≥7 for the "
                f"top6/bottom5 ELO split features")
        if len(lu) < 11:
            print(f"  WARN: {side}_lineup has {len(lu)} players (expected 11)")

    # Toss handling. When the toss is unknown (pre-toss fixture), the caller
    # predicts BOTH bat-first branches and averages (see main) — a fixed
    # default here would be a train/serve skew: training rows always carry
    # the real toss (materializer defaults the rare missing case to True,
    # the old inference default was False). 2026-07-16 review I1.
    toss_winner = fixture.get("toss_winner")
    toss_decision = fixture.get("toss_decision") or "field"
    toss_known = toss_winner in (team1, team2)
    if toss_winner == team1:
        team1_batting_first = (toss_decision == "bat")
    elif toss_winner == team2:
        team1_batting_first = (toss_decision == "field")
    else:
        team1_batting_first = False  # placeholder; overridden by branch-averaging

    # Rehydrate per-player ELO + venue trackers as-of the fixture date.
    # SQLite returns state strictly before this date (first-write-wins),
    # so for any fixture_date past test_end (2026-04-16), this falls back
    # to the latest non-golden state. Within-corpus dates get full
    # pre-fixture history.
    rehydrate_as_of = match_date
    union_pids = set(team1_lineup) | set(team2_lineup)
    elo_tracker = rehydrate_elo_tracker(provider, rehydrate_as_of, union_pids)
    venue_tracker = rehydrate_venue_tracker(provider, rehydrate_as_of, {venue})

    # Team-level batting/bowling aggregates (per StatsProvider).
    t1_bat_strength = provider.get_team_batting_strength(team1_lineup, rehydrate_as_of)
    t1_bow_strength = provider.get_team_bowling_strength(team1_lineup, rehydrate_as_of)
    t2_bat_strength = provider.get_team_batting_strength(team2_lineup, rehydrate_as_of)
    t2_bow_strength = provider.get_team_bowling_strength(team2_lineup, rehydrate_as_of)
    t1_bat_elo = provider.get_team_batting_elo(team1_lineup, rehydrate_as_of)
    t1_bow_elo = provider.get_team_bowling_elo(team1_lineup, rehydrate_as_of)
    t2_bat_elo = provider.get_team_batting_elo(team2_lineup, rehydrate_as_of)
    t2_bow_elo = provider.get_team_bowling_elo(team2_lineup, rehydrate_as_of)

    # Venue features — use the rehydrated tracker so semantics match the
    # parser exactly (venue_avg_score = total_runs/innings_count, not
    # first-innings-mean).
    venue_avg_score = venue_tracker.get_venue_avg_score(venue)
    vp = venue_tracker.get_venue_profile(venue)
    venue_chase_win_pct = vp["venue_chase_win_pct"]
    venue_dot_pct = vp["venue_dot_pct"]
    venue_boundary_pct = vp["venue_boundary_pct"]
    # k_venue=200 matches materialize_match_features._build_match_record.
    venue_dist = provider.get_venue_outcome_dist(venue, rehydrate_as_of, k=200.0)

    # Lineup mix.
    t1_lhb, t1_pace, t1_spin = _lineup_mix_counts(team1_lineup, metadata)
    t2_lhb, t2_pace, t2_spin = _lineup_mix_counts(team2_lineup, metadata)

    # Top-6 batting / bottom-5 bowling ELO splits.
    t1_top6_bat, t1_bot5_bow = _split_elo(team1_lineup, elo_tracker)
    t2_top6_bat, t2_bot5_bow = _split_elo(team2_lineup, elo_tracker)

    # Phase A2 trackers (form, H2H, home). Query with the actual fixture
    # date — the trackers' internal records are frozen at 2025-06-30,
    # but the date controls the lookup window for is_home (730d window).
    t1_form, _ = form.get_last_n_win_rate(team1, match_date)
    t2_form, _ = form.get_last_n_win_rate(team2, match_date)
    h2h_rate, h2h_n = h2h.get_h2h(team1, team2, match_date, k=2.0)
    is_t1_home = home.is_home(team1, venue, match_date)
    is_t2_home = home.is_home(team2, venue, match_date)

    t1_bat_avg = t1_bat_strength.get("team_batting_avg", 0.0)
    t1_bat_sr = t1_bat_strength.get("team_batting_sr", 0.0)
    t1_bow_avg = t1_bow_strength.get("team_bowling_avg", 0.0)
    t1_bow_econ = t1_bow_strength.get("team_bowling_econ", 0.0)
    t2_bat_avg = t2_bat_strength.get("team_batting_avg", 0.0)
    t2_bat_sr = t2_bat_strength.get("team_batting_sr", 0.0)
    t2_bow_avg = t2_bow_strength.get("team_bowling_avg", 0.0)
    t2_bow_econ = t2_bow_strength.get("team_bowling_econ", 0.0)

    record = {
        "team1_batting_elo": t1_bat_elo,
        "team1_bowling_elo": t1_bow_elo,
        "team2_batting_elo": t2_bat_elo,
        "team2_bowling_elo": t2_bow_elo,
        "team1_batting_avg": t1_bat_avg,
        "team1_batting_sr": t1_bat_sr,
        "team1_bowling_avg": t1_bow_avg,
        "team1_bowling_econ": t1_bow_econ,
        "team2_batting_avg": t2_bat_avg,
        "team2_batting_sr": t2_bat_sr,
        "team2_bowling_avg": t2_bow_avg,
        "team2_bowling_econ": t2_bow_econ,
        "elo_diff_batting": t1_bat_elo - t2_bat_elo,
        "elo_diff_bowling": t1_bow_elo - t2_bow_elo,
        "batting_avg_diff": t1_bat_avg - t2_bat_avg,
        "bowling_econ_diff": t1_bow_econ - t2_bow_econ,
        "venue_avg_score": float(venue_avg_score),
        "venue_chase_win_pct": float(venue_chase_win_pct),
        "venue_dot_pct": float(venue_dot_pct),
        "venue_boundary_pct": float(venue_boundary_pct),
        "venue_p4": float(venue_dist["venue_p4"]),
        "venue_p6": float(venue_dist["venue_p6"]),
        "venue_pw": float(venue_dist["venue_pw"]),
        "is_international": int(is_international),
        "team1_batting_first": int(team1_batting_first),
        "toss_winner_is_team1": int(toss_winner == team1) if toss_winner else 0,
        "toss_decision_bat": 1 if toss_decision == "bat" else 0,
        "team1_win_rate_last_10": float(t1_form),
        "team2_win_rate_last_10": float(t2_form),
        "win_rate_diff": float(t1_form - t2_form),
        "h2h_team1_win_rate_shrunk": float(h2h_rate),
        "h2h_n_meetings": int(h2h_n),
        "team1_lhb_count": int(t1_lhb),
        "team1_pace_count": int(t1_pace),
        "team1_spinner_count": int(t1_spin),
        "team2_lhb_count": int(t2_lhb),
        "team2_pace_count": int(t2_pace),
        "team2_spinner_count": int(t2_spin),
        "is_team1_home": int(is_t1_home),
        "is_team2_home": int(is_t2_home),
        "team1_top6_batting_elo_avg": float(t1_top6_bat),
        "team2_top6_batting_elo_avg": float(t2_top6_bat),
        "top6_batting_elo_diff": float(t1_top6_bat - t2_top6_bat),
        "team1_bottom5_bowling_elo_avg": float(t1_bot5_bow),
        "team2_bottom5_bowling_elo_avg": float(t2_bot5_bow),
        "bottom5_bowling_elo_diff": float(t1_bot5_bow - t2_bot5_bow),
        # Categorical raw values — get encoded below.
        "venue": venue,
        "competition_tier": competition_tier_code,
        # Metadata (popped by the caller before prediction, not a feature).
        "_toss_known": toss_known,
    }
    return record


def apply_encoders_and_predict(record: dict,
                                model_dir: Path = MODEL_DIR) -> tuple[float, dict]:
    model = joblib.load(model_dir / "model.pkl")
    encoders = joblib.load(model_dir / "encoders.pkl")
    with open(model_dir / "feature_columns.txt") as f:
        feat_cols = [line.strip() for line in f if line.strip()]

    df = pd.DataFrame([record])
    encoder_warnings = []
    for col, le in encoders.items():
        encoded_col = f"{col}_id_encoded" if col == "venue" else f"{col}_encoded"
        known = set(le.classes_)
        if df[col].iloc[0] not in known:
            fallback = le.classes_[0]
            encoder_warnings.append(
                f"unseen {col}={df[col].iloc[0]!r}; falling back to {fallback!r}")
            df[col] = fallback
        df[encoded_col] = le.transform(df[col].astype(str))

    proba = float(model.predict_proba(df[feat_cols])[0, 1])
    return proba, {"encoder_warnings": encoder_warnings,
                   "feature_columns": feat_cols,
                   "feature_row": {c: (float(df[c].iloc[0]) if c not in ('venue','competition_tier') else df[c].iloc[0]) for c in feat_cols}}


def compute_bet(team1: str, team2: str, p_team1: float,
                polymarket_odds: dict | None) -> dict:
    if not polymarket_odds:
        return {"odds_provided": False}
    try:
        d1 = float(polymarket_odds[team1])
        d2 = float(polymarket_odds[team2])
    except (KeyError, TypeError):
        return {"odds_provided": False, "error": "odds dict must include both teams"}
    market_t1 = 1.0 / d1
    market_t2 = 1.0 / d2
    p_team2 = 1.0 - p_team1
    edge_t1 = p_team1 - market_t1
    edge_t2 = p_team2 - market_t2
    edges = {team1: edge_t1, team2: edge_t2}
    best_team = max(edges, key=edges.get)
    best_edge = edges[best_team]
    placed = best_edge > EDGE_THRESHOLD
    return {
        "odds_provided": True,
        "decimal_odds": {team1: d1, team2: d2},
        "market_implied_prob": {team1: market_t1, team2: market_t2},
        "edge_pp": {team1: edge_t1 * 100, team2: edge_t2 * 100},
        "bet_team": best_team if placed else None,
        "bet_decimal": d1 if (placed and best_team == team1) else (d2 if placed else None),
        "bet_edge_pp": best_edge * 100 if placed else None,
        "expected_pnl_per_unit_if_won": (
            (d1 - 1) if (placed and best_team == team1)
            else ((d2 - 1) if placed else 0.0)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", type=Path, required=True,
                    help="Path to fixture JSON (see fixtures/_template.json)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output JSON path; default predictions/<match_id>.json")
    ap.add_argument("--rebuild-snapshot", action="store_true",
                    help="Force rebuild of the Phase A2 tracker snapshot.")
    ap.add_argument("--model-dir", type=Path, default=MODEL_DIR,
                    help=f"Model artifact dir (default: {MODEL_DIR.name})")
    ap.add_argument(
        "--state-dir",
        type=Path,
        default=DEFAULT_STATE_DIR,
        help=(
            "Directory containing player_stats_cache_v3.sqlite "
            f"(default: {DEFAULT_STATE_DIR})"
        ),
    )
    ap.add_argument(
        "--tracker-snapshot",
        type=Path,
        default=TRACKER_SNAPSHOT,
        help=f"Tracker snapshot path (default: {TRACKER_SNAPSHOT})",
    )
    ap.add_argument(
        "--tracker-source-dir",
        type=Path,
        action="append",
        dest="tracker_source_dirs",
        help=(
            "Ordered-state source pool used to build the tracker snapshot; "
            "repeat for multiple directories"
        ),
    )
    ap.add_argument(
        "--max-state-age-days",
        type=int,
        default=DEFAULT_MAX_STATE_AGE_DAYS,
        help=(
            "Block when either state source is older than the fixture by "
            f"more than this many days (default: {DEFAULT_MAX_STATE_AGE_DAYS})"
        ),
    )
    ap.add_argument(
        "--allow-stale-state",
        action="store_true",
        help=(
            "Allow a diagnostic prediction with stale state. The output is "
            "marked stale and any betting recommendation is suppressed."
        ),
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    tracker_sources = tuple(
        args.tracker_source_dirs or DEFAULT_TRACKER_SOURCE_DIRS
    )
    if args.rebuild_snapshot:
        build_tracker_snapshot(tracker_sources, args.tracker_snapshot)
    elif not args.tracker_snapshot.exists():
        if (args.tracker_snapshot != TRACKER_SNAPSHOT
                and not args.tracker_source_dirs):
            raise RuntimeError(
                "a custom tracker snapshot does not exist; pass each matching "
                "--tracker-source-dir and --rebuild-snapshot"
            )
        build_tracker_snapshot(tracker_sources, args.tracker_snapshot)

    fixture = json.loads(args.fixture.read_text())
    print(f"Predicting: {fixture['date']}  "
          f"{fixture['team1']} vs {fixture['team2']}  @ {fixture['venue']}")

    sqlite_state = read_sqlite_state_metadata(args.state_dir)
    tracker_state = read_tracker_state_metadata(args.tracker_snapshot)
    state_freshness = assess_state_freshness(
        fixture["date"],
        sqlite_state,
        tracker_state,
        max_state_age_days=args.max_state_age_days,
    )
    if state_freshness["status"] == "stale":
        message = (
            f"live state is {state_freshness['age_days']} days behind fixture "
            f"{fixture['date']} (maximum {args.max_state_age_days}); effective "
            f"as-of is {state_freshness['effective_as_of']}. Rebuild a "
            "separate SQLite cache and tracker snapshot from the same sources, "
            "then pass --state-dir/--tracker-snapshot. "
            "--allow-stale-state is diagnostic-only and suppresses betting."
        )
        if not args.allow_stale_state:
            raise RuntimeError(message)
        state_freshness["status"] = "stale_override"
        state_freshness["override_used"] = True
        print(f"  WARN: {message}")
    else:
        state_freshness["override_used"] = False

    print("Loading providers + trackers...")
    provider = StatsProvider(str(args.state_dir), version="v3")
    metadata = PlayerMetadataProvider(str(REPO / "data" / "all_players_enriched.csv"))
    form, h2h, home = load_trackers(args.tracker_snapshot, tracker_sources)

    print("Computing features...")
    record = compute_features(fixture, provider, metadata, form, h2h, home)

    print("Applying model...")
    toss_known = bool(record.pop("_toss_known", True))
    toss_branch_probs = None
    if toss_known:
        p_team1, debug = apply_encoders_and_predict(record, args.model_dir)
    else:
        # Unknown toss: predict both bat-first branches and average
        # (2026-07-16 review I1 — removes the fixed team1-chasing default,
        # which was a systematic train/serve skew on every pre-toss fixture).
        branch = dict(record)
        branch["team1_batting_first"] = 1
        p_bat, debug = apply_encoders_and_predict(branch, args.model_dir)
        branch["team1_batting_first"] = 0
        p_chase, _ = apply_encoders_and_predict(branch, args.model_dir)
        p_team1 = 0.5 * (p_bat + p_chase)
        toss_branch_probs = {"team1_bats_first": p_bat, "team1_chases": p_chase}
        print(f"  (toss unknown — averaged bat-first branches: "
              f"{p_bat*100:.1f}% / {p_chase*100:.1f}%)")
    p_team2 = 1.0 - p_team1

    bet_info = compute_bet(fixture["team1"], fixture["team2"], p_team1,
                           fixture.get("polymarket_odds"))

    output = {
        "fixture": {k: v for k, v in fixture.items() if k != "team1_lineup" and k != "team2_lineup"},
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
            "model": str(args.model_dir),
            "rehydrate_as_of": fixture["date"],
            "state_freshness": state_freshness,
            "sqlite_cache": sqlite_state["path"],
            "tracker_snapshot": tracker_state["path"],
            "tracker_snapshot_as_of": _peek_snapshot_as_of(
                args.tracker_snapshot
            ),
            "toss_known": toss_known,
            "toss_branch_probs": toss_branch_probs,
            "encoder_warnings": debug["encoder_warnings"],
            "h2h_n_meetings": record["h2h_n_meetings"],
            "is_team1_home": record["is_team1_home"],
            "is_team2_home": record["is_team2_home"],
            "top6_batting_elo_diff": record["top6_batting_elo_diff"],
            "bottom5_bowling_elo_diff": record["bottom5_bowling_elo_diff"],
        },
    }
    if args.verbose:
        output["feature_row"] = debug["feature_row"]

    out_path = args.out or (REPO / "predictions" /
                            f"{fixture['date']}_{fixture['team1'].replace(' ','_')}"
                            f"_vs_{fixture['team2'].replace(' ','_')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))

    print()
    print(f"  P({fixture['team1']:<35s} wins) = {p_team1*100:>5.1f}%")
    print(f"  P({fixture['team2']:<35s} wins) = {p_team2*100:>5.1f}%")
    print()
    if bet_info.get("odds_provided"):
        if bet_info.get("bet_team"):
            print(f"  Bet: {bet_info['bet_team']} @ {bet_info['bet_decimal']:.2f}  "
                  f"(edge +{bet_info['bet_edge_pp']:.1f}pp)")
            print(f"  PnL if win: +{bet_info['expected_pnl_per_unit_if_won']:.3f}; "
                  f"PnL if loss: -1.000")
        else:
            print(f"  No bet — best edge {max(bet_info['edge_pp'].values()):+.1f}pp "
                  f"≤ threshold {EDGE_THRESHOLD*100:.0f}pp")
    else:
        print("  (no polymarket odds provided in fixture; pass to compute bet)")

    if debug["encoder_warnings"]:
        print()
        print("  WARNINGS:")
        for w in debug["encoder_warnings"]:
            print(f"    - {w}")

    print()
    print(f"  Full output -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
