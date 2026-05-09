"""Materialize one-row-per-match features for the direct match-level model.

Phase A1 of the match-level direct + sim ensemble plan
(`~/.claude/plans/okay-let-s-go-ahead-reflective-sunrise.md`).

Walks cricsheet JSONs in chronological order, rehydrates trackers per-date
(same pattern as `materialize_features.py`), calls `parse_match_data_v2` for
each match, then collapses the per-ball rows into a single match-level
record by taking the first ball of each innings (where the inning's
team-batting / team-bowling features sit) plus match-level constants from
`info.*`. The match's `team1_wins` target comes from `info.outcome.winner`.

Output: `data/xgb_match_data_v1/{train,validation,test,golden_test}.parquet`,
one row per non-no-result match, with the cheap-subset feature list
documented in the plan.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from loaders_common import (
    DEFAULT_SPLITS,
    extract_match_metadata,
    iter_matches_chronological,
)
from materialize_features import classify_split, group_by_date


def iter_matches_chronological_multi(folders, gender="male"):
    """Walk multiple cricsheet pools and yield merged in date order.

    Re-uses iter_matches_chronological per folder, materializes each into a
    list, then merge-sorts by (date, match_id) for a deterministic global
    chronology. No dedupe — caller ensures pools don't overlap (cricsheet
    filenames are unique stems, so this is safe with our golden / live split).
    """
    streams = [iter_matches_chronological(f, gender=gender) for f in folders]
    tagged = [((d, mid), (mid, txt, d)) for s in streams for (mid, txt, d) in s]
    tagged.sort(key=lambda t: t[0])
    for _, entry in tagged:
        yield entry
from parsing_v2 import parse_match_data_v2
from player_metadata import PlayerMetadataProvider
from stats_provider import StatsProvider
from tracker_rehydration import (
    extract_match_player_ids,
    rehydrate_elo_tracker,
    rehydrate_stats_tracker,
    rehydrate_venue_tracker,
)


FEATURE_COLUMNS = [
    # Team strength (per-team absolutes)
    "team1_batting_elo", "team1_bowling_elo",
    "team2_batting_elo", "team2_bowling_elo",
    "team1_batting_avg", "team1_batting_sr",
    "team1_bowling_avg", "team1_bowling_econ",
    "team2_batting_avg", "team2_batting_sr",
    "team2_bowling_avg", "team2_bowling_econ",
    # Differentials (team1 minus team2)
    "elo_diff_batting", "elo_diff_bowling",
    "batting_avg_diff", "bowling_econ_diff",
    # Venue
    "venue_id_encoded",  # filled at training time across train+val+test
    "venue_avg_score", "venue_chase_win_pct",
    "venue_dot_pct", "venue_boundary_pct",
    # Match context
    "competition_tier_encoded",  # filled at training time
    "is_international",
    "team1_batting_first",
    "toss_winner_is_team1",
    "toss_decision_bat",  # 1 if toss winner chose to bat, else 0
    # === Phase A2 ===
    # Recent form (last-10 win rate per team, neutral 0.5 if <10 prior matches)
    "team1_win_rate_last_10", "team2_win_rate_last_10", "win_rate_diff",
    # Head-to-head (Beta(α=1,β=1) shrinkage prior — eq. to k=2 toward 0.5)
    "h2h_team1_win_rate_shrunk", "h2h_n_meetings",
    # Lineup mix (counts among the 11 squad-list players)
    "team1_lhb_count", "team1_pace_count", "team1_spinner_count",
    "team2_lhb_count", "team2_pace_count", "team2_spinner_count",
    # Home advantage (3+ matches at venue in prior 730 days)
    "is_team1_home", "is_team2_home",
    # Top-of-order ELO splits — squad-list-order proxy for batting order
    "team1_top6_batting_elo_avg", "team2_top6_batting_elo_avg",
    "top6_batting_elo_diff",
    "team1_bottom5_bowling_elo_avg", "team2_bottom5_bowling_elo_avg",
    "bottom5_bowling_elo_diff",
]

METADATA_COLUMNS = [
    "match_id", "cricsheet_id", "match_date", "team1", "team2", "venue",
    "competition_tier",  # raw string, encoded at training time
    "team1_wins",
]


class TeamFormTracker:
    """Records per-team match results in chronological order. Queries
    return the team's win rate over its last `n` matches with date strictly
    before `as_of_date`. Same-day prior matches are excluded so that
    same-day siblings don't see each other's outcomes.
    """

    def __init__(self) -> None:
        self.records: Dict[str, List[Tuple[datetime, bool]]] = defaultdict(list)

    def get_last_n_win_rate(self, team: str, as_of_date: datetime,
                            n: int = 10) -> Tuple[float, int]:
        recs = self.records.get(team, [])
        prior = [r for r in recs if r[0] < as_of_date]
        last_n = prior[-n:]
        if not last_n:
            return 0.5, 0  # neutral prior
        wins = sum(1 for _, won in last_n if won)
        return wins / len(last_n), len(last_n)

    def update(self, team: str, match_date: datetime, won: bool) -> None:
        self.records[team].append((match_date, won))


class H2HTracker:
    """Pairwise head-to-head record. Stores winner per past meeting
    between an unordered pair of teams. Query returns shrunken team1 win
    rate against `team2` and the count of prior meetings.
    """

    def __init__(self) -> None:
        self.records: Dict[frozenset, List[Tuple[datetime, str]]] = defaultdict(list)

    def get_h2h(self, team1: str, team2: str, as_of_date: datetime,
                k: float = 2.0) -> Tuple[float, int]:
        """Returns (shrunk_win_rate_team1, n_meetings). Beta(α, β) prior
        with α=β=k/2 is equivalent to adding k/2 wins and k/2 losses to
        the team1 column. With k=2 (default) and 0 meetings → 0.5; with
        many meetings, shrinkage washes out.
        """
        key = frozenset({team1, team2})
        recs = self.records.get(key, [])
        prior = [r for r in recs if r[0] < as_of_date]
        n = len(prior)
        team1_wins = sum(1 for _, w in prior if w == team1)
        # Posterior mean with Beta(k/2, k/2) prior centered at 0.5.
        return (team1_wins + k / 2) / (n + k), n

    def update(self, team1: str, team2: str, match_date: datetime,
               winner: str) -> None:
        if winner not in (team1, team2):
            return
        key = frozenset({team1, team2})
        self.records[key].append((match_date, winner))


class HomeVenueTracker:
    """Records (team, venue) appearances in chronological order. A team
    is considered 'home' at a venue if it has played `threshold` or more
    matches at that venue in the prior `lookback_days`.
    """

    def __init__(self, lookback_days: int = 730) -> None:
        self.records: Dict[Tuple[str, str], List[datetime]] = defaultdict(list)
        self.lookback = timedelta(days=lookback_days)

    def is_home(self, team: str, venue: str, as_of_date: datetime,
                threshold: int = 3) -> int:
        recs = self.records.get((team, venue), [])
        cutoff = as_of_date - self.lookback
        recent = [d for d in recs if cutoff <= d < as_of_date]
        return 1 if len(recent) >= threshold else 0

    def update(self, team: str, venue: str, match_date: datetime) -> None:
        self.records[(team, venue)].append(match_date)


def _lineup_mix_counts(lineup_ids: List[str], metadata) -> Tuple[int, int, int]:
    """Return (lhb_count, pace_count, spinner_count) over the lineup.
    Unknowns excluded from each count.
    """
    lhb = pace = spin = 0
    for pid in lineup_ids:
        meta = metadata.get_player_metadata(pid)
        if meta.get("batter_hand") == "left":
            lhb += 1
        is_pace = meta.get("is_pace")
        if is_pace is True:
            pace += 1
        elif is_pace is False:
            spin += 1
    return lhb, pace, spin


def _split_elo(lineup_ids: List[str], elo_tracker,
               top_n: int = 6) -> Tuple[float, float]:
    """Return (top-N batting ELO mean, bottom-(len−N) bowling ELO mean)
    using squad-list order as a proxy for batting order. Defaults
    approximate top-6 batters / bottom-5 (typically the bowling unit).
    """
    if not lineup_ids:
        return 0.0, 0.0
    top = lineup_ids[:top_n]
    bottom = lineup_ids[top_n:]
    top_bat_elos = [elo_tracker.get_batting_elo(p) for p in top]
    bot_bow_elos = [elo_tracker.get_bowling_elo(p) for p in bottom] if bottom else top_bat_elos
    return (
        sum(top_bat_elos) / len(top_bat_elos) if top_bat_elos else 0.0,
        sum(bot_bow_elos) / len(bot_bow_elos) if bot_bow_elos else 0.0,
    )


def _build_match_record(
    match_id: str,
    match_date: datetime,
    data: dict,
    rows: List[dict],
    metadata,
    elo_tracker,
    form_tracker: TeamFormTracker,
    h2h_tracker: H2HTracker,
    home_tracker: HomeVenueTracker,
) -> Optional[dict]:
    """Collapse per-ball rows into a single match-level record. Returns
    None if the match has no valid winner (no-result / abandoned).
    """
    info = data.get("info", {})
    teams = info.get("teams", [])
    if len(teams) != 2:
        return None

    outcome = info.get("outcome", {})
    winner = outcome.get("winner")
    if not winner or winner not in teams:
        # No-result / abandoned / tie without super-over winner. Drop.
        return None

    team1, team2 = teams[0], teams[1]

    # Find first row of innings 1 and innings 2 (both share match-level
    # context but each carries the *batting* team's features in that
    # innings).
    innings_first: Dict[int, dict] = {}
    for row in rows:
        idx = int(row.get("inning_idx", 0))
        if idx in (1, 2) and idx not in innings_first:
            innings_first[idx] = row
        if 1 in innings_first and 2 in innings_first:
            break

    if 1 not in innings_first:
        return None  # parse failed or invalid match
    inn1 = innings_first[1]

    # Determine which team batted first by aligning innings team to teams[].
    # parsing_v2 uses inning 'team' field; if absent, falls back to
    # teams[inning_idx-1] — but that assumption breaks when the team that
    # won the toss elects to bowl. We disambiguate via the toss block.
    toss = info.get("toss", {})
    toss_winner = toss.get("winner")
    toss_decision = toss.get("decision", "bat")  # 'bat' or 'field'
    if toss_winner == team1:
        team1_batting_first = (toss_decision == "bat")
    elif toss_winner == team2:
        team1_batting_first = (toss_decision == "field")
    else:
        # Toss data missing — fall back to per-ball striker_id team if
        # we can derive it. For now, default to team1 first (rare path).
        team1_batting_first = True

    if 2 not in innings_first:
        # Match abandoned mid-innings-1; outcome is suspect. Drop.
        return None

    # parsing_v2.py:1145-1148: per-ball row's `team_batting_*` = batting
    # team's stats; `team_bowling_*` = bowling team's stats. So inn1
    # carries (team_batting_first, team_bowling_first_opponent), inn2
    # carries (team_batting_second, team_bowling_second_opponent).
    inn2 = innings_first[2]
    if team1_batting_first:
        t1_bat_elo = float(inn1.get("batting_team_elo", 0.0))
        t1_bat_avg = float(inn1.get("team_batting_avg", 0.0))
        t1_bat_sr  = float(inn1.get("team_batting_sr", 0.0))
        t1_bow_elo = float(inn2.get("bowling_team_elo", 0.0))
        t1_bow_avg = float(inn2.get("team_bowling_avg", 0.0))
        t1_bow_econ = float(inn2.get("team_bowling_econ", 0.0))
        t2_bat_elo = float(inn2.get("batting_team_elo", 0.0))
        t2_bat_avg = float(inn2.get("team_batting_avg", 0.0))
        t2_bat_sr  = float(inn2.get("team_batting_sr", 0.0))
        t2_bow_elo = float(inn1.get("bowling_team_elo", 0.0))
        t2_bow_avg = float(inn1.get("team_bowling_avg", 0.0))
        t2_bow_econ = float(inn1.get("team_bowling_econ", 0.0))
    else:
        t1_bat_elo = float(inn2.get("batting_team_elo", 0.0))
        t1_bat_avg = float(inn2.get("team_batting_avg", 0.0))
        t1_bat_sr  = float(inn2.get("team_batting_sr", 0.0))
        t1_bow_elo = float(inn1.get("bowling_team_elo", 0.0))
        t1_bow_avg = float(inn1.get("team_bowling_avg", 0.0))
        t1_bow_econ = float(inn1.get("team_bowling_econ", 0.0))
        t2_bat_elo = float(inn1.get("batting_team_elo", 0.0))
        t2_bat_avg = float(inn1.get("team_batting_avg", 0.0))
        t2_bat_sr  = float(inn1.get("team_batting_sr", 0.0))
        t2_bow_elo = float(inn2.get("bowling_team_elo", 0.0))
        t2_bow_avg = float(inn2.get("team_bowling_avg", 0.0))
        t2_bow_econ = float(inn2.get("team_bowling_econ", 0.0))

    venue = info.get("venue", "unknown")
    competition_tier = inn1.get("competition_tier", "unknown")

    # Synthesize match_id in the same format as
    # `sim_eval/loaders.py:73` and `betting_odds_polymarket.json` so this
    # parquet's test rows can be joined to the eval JSONs by match_id.
    date_str = match_date.strftime("%Y-%m-%d")
    synth_match_id = f"{date_str}_{team1}_{team2}_{venue}".replace(" ", "_")

    # === Phase A2 ===
    # Resolve lineup player IDs (squad-list order). Same pattern as
    # parsing_v2.py:1066-1067.
    player_registry = info.get("registry", {}).get("people", {})
    team1_lineup_names = info.get("players", {}).get(team1, [])
    team2_lineup_names = info.get("players", {}).get(team2, [])
    team1_lineup_ids = [player_registry.get(n, n) for n in team1_lineup_names]
    team2_lineup_ids = [player_registry.get(n, n) for n in team2_lineup_names]

    # Lineup mix counts
    t1_lhb, t1_pace, t1_spin = _lineup_mix_counts(team1_lineup_ids, metadata)
    t2_lhb, t2_pace, t2_spin = _lineup_mix_counts(team2_lineup_ids, metadata)

    # Top-6 batting ELO + bottom-5 bowling ELO
    t1_top6_bat, t1_bot5_bow = _split_elo(team1_lineup_ids, elo_tracker)
    t2_top6_bat, t2_bot5_bow = _split_elo(team2_lineup_ids, elo_tracker)

    # Recent form (querying BEFORE updating tracker for this match)
    t1_form, t1_form_n = form_tracker.get_last_n_win_rate(team1, match_date)
    t2_form, t2_form_n = form_tracker.get_last_n_win_rate(team2, match_date)

    # Head-to-head (k=2 → Beta(1,1) prior, neutral 0.5 with no meetings)
    h2h_rate, h2h_n = h2h_tracker.get_h2h(team1, team2, match_date, k=2.0)

    # Home advantage
    is_t1_home = home_tracker.is_home(team1, venue, match_date)
    is_t2_home = home_tracker.is_home(team2, venue, match_date)

    record = {
        "match_id": synth_match_id,
        "cricsheet_id": match_id,  # keep the JSON filename stem for debug
        "match_date": date_str,
        "team1": team1,
        "team2": team2,
        "venue": venue,
        "competition_tier": competition_tier,
        "team1_wins": 1 if winner == team1 else 0,

        # Team strength
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

        # Diffs
        "elo_diff_batting": t1_bat_elo - t2_bat_elo,
        "elo_diff_bowling": t1_bow_elo - t2_bow_elo,
        "batting_avg_diff": t1_bat_avg - t2_bat_avg,
        "bowling_econ_diff": t1_bow_econ - t2_bow_econ,

        # Venue (raw values; venue_id_encoded filled at training time)
        "venue_avg_score": float(inn1.get("venue_avg_score", 0.0)),
        "venue_chase_win_pct": float(inn1.get("venue_chase_win_pct", 0.5)),
        "venue_dot_pct": float(inn1.get("venue_dot_pct", 0.0)),
        "venue_boundary_pct": float(inn1.get("venue_boundary_pct", 0.0)),

        # Match context
        "is_international": int(inn1.get("is_international", 0)),
        "team1_batting_first": int(team1_batting_first),
        "toss_winner_is_team1": int(toss_winner == team1) if toss_winner else 0,
        "toss_decision_bat": 1 if toss_decision == "bat" else 0,

        # === Phase A2 ===
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
    }
    return record


def materialize(
    source_dir,  # Path or list[Path]
    sqlite_dir: Path,
    out_dir: Path,
    version: str,
    splits: dict,
    gender: str,
    metadata_csv: Path,
    k_player: float = 30.0,
    k_venue: float = 200.0,
    freeze_trackers_after: Optional[str] = None,
) -> Tuple[int, dict]:
    provider = StatsProvider(str(sqlite_dir), version=version)
    if provider.backend_name != "sqlite":
        raise RuntimeError(
            f"materialize_match_features requires SQLite backend; got "
            f"{provider.backend_name!r}"
        )
    metadata = PlayerMetadataProvider(str(metadata_csv))

    provider._backend._ensure_conn()
    prior = provider._backend._prior
    phase_priors = provider._backend._phase_priors

    split_records: dict[str, List[dict]] = {
        "train": [], "validation": [], "test": [], "golden_test": [],
    }
    n_matches = 0
    n_dropped = 0
    t_start = time.time()

    # === Phase A2 trackers — evolve chronologically across the corpus.
    form_tracker = TeamFormTracker()
    h2h_tracker = H2HTracker()
    home_tracker = HomeVenueTracker(lookback_days=730)

    # No-leakage diagnostic: freeze trackers + SQLite rehydration as of
    # `freeze_trackers_after + 1 day`. Test matches all see the same
    # snapshot; same-day cross-match contamination prevented by per-match
    # fresh rehydration.
    freeze_dt: Optional[datetime] = (
        datetime.strptime(freeze_trackers_after, "%Y-%m-%d")
        if freeze_trackers_after else None
    )
    if freeze_dt is not None:
        freeze_as_of = freeze_dt + timedelta(days=1)
        print(f"  FREEZE MODE: trackers + SQLite rehydration locked at "
              f"{freeze_as_of.strftime('%Y-%m-%d')} for matches with date > "
              f"{freeze_dt.strftime('%Y-%m-%d')}")

    source_dirs = source_dir if isinstance(source_dir, (list, tuple)) else [source_dir]
    iter_fn = (iter_matches_chronological_multi(source_dirs, gender=gender)
               if len(source_dirs) > 1
               else iter_matches_chronological(source_dirs[0], gender=gender))
    for match_date, batch in group_by_date(iter_fn):
        is_frozen_date = freeze_dt is not None and match_date > freeze_dt

        if not is_frozen_date:
            # Normal mode: per-date rehydration + within-date updates.
            # Same-day siblings share temp_* state, mirroring the pipeline.
            union_pids: set = set()
            union_venues: set = set()
            for _, _, data, venue, _ in batch:
                union_pids.update(extract_match_player_ids(data))
                union_venues.add(venue)
            temp_stats = rehydrate_stats_tracker(provider, match_date, union_pids)
            temp_elo = rehydrate_elo_tracker(provider, match_date, union_pids)
            temp_venue = rehydrate_venue_tracker(
                provider, match_date, union_venues)

        for match_id, json_text, data, venue, k_factor in batch:
            if is_frozen_date:
                # Frozen mode: rehydrate fresh per match using the snapshot
                # at freeze_as_of. Prevents within-date cross-match
                # contamination. parse_match_data_v2 still mutates these
                # within a single match, but each match starts clean.
                one_match_pids = extract_match_player_ids(data)
                one_match_venues = {venue}
                temp_stats = rehydrate_stats_tracker(
                    provider, freeze_as_of, one_match_pids)
                temp_elo = rehydrate_elo_tracker(
                    provider, freeze_as_of, one_match_pids)
                temp_venue = rehydrate_venue_tracker(
                    provider, freeze_as_of, one_match_venues)

            # SNAPSHOT temp_elo BEFORE parse_match_data_v2 mutates it. The
            # snapshot is passed to _build_match_record so the top6/bottom5
            # ELO splits reflect truly pre-match state. parse still mutates
            # the live temp_elo so subsequent same-day matches get the
            # post-this-match ELO (matches monolith chronological semantics).
            from parsing_v2 import PlayerEloTracker  # local import; cheap
            pre_match_elo = PlayerEloTracker()
            pre_match_elo.batting_elo = dict(temp_elo.batting_elo)
            pre_match_elo.bowling_elo = dict(temp_elo.bowling_elo)

            rows, _it, vname, innings_details, chase_won = (
                parse_match_data_v2(
                    json_text, temp_stats, temp_venue, metadata,
                    elo_tracker=temp_elo, match_k_factor=k_factor,
                    prior=prior, phase_priors=phase_priors,
                    k_player=k_player, k_venue=k_venue,
                )
            )
            if not is_frozen_date:
                # Within-date temp_venue updates only in normal mode.
                for det in innings_details:
                    temp_venue.update_venue_stats_detailed(vname, det)
                if chase_won is not None:
                    temp_venue.update_venue_match_result(vname, chase_won)

            record = _build_match_record(
                match_id, match_date, data, rows,
                metadata, pre_match_elo, form_tracker, h2h_tracker, home_tracker,
            )
            if record is None:
                n_dropped += 1
                continue

            split = classify_split(match_date, splits)
            split_records[split].append(record)
            n_matches += 1

            # In frozen mode, do NOT update A2 trackers — test matches all
            # read the snapshot frozen at val/test boundary.
            if is_frozen_date:
                continue

            # Normal mode: update Phase A2 trackers AFTER computing this
            # match's features. Same-day later matches will see this
            # match's outcome.
            t1_won = record["team1_wins"] == 1
            form_tracker.update(record["team1"], match_date, t1_won)
            form_tracker.update(record["team2"], match_date, not t1_won)
            winner = record["team1"] if t1_won else record["team2"]
            h2h_tracker.update(record["team1"], record["team2"],
                               match_date, winner)
            home_tracker.update(record["team1"], record["venue"], match_date)
            home_tracker.update(record["team2"], record["venue"], match_date)

        if n_matches % 1000 == 0 and n_matches > 0:
            dt = time.time() - t_start
            print(f"  [{n_matches}] matches in {dt:.0f}s "
                  f"({n_matches / dt:.1f} match/s)", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split_name, records in split_records.items():
        counts[split_name] = len(records)
        if not records:
            continue
        df = pd.DataFrame(records)
        out_path = out_dir / f"{split_name}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  wrote {out_path} ({len(records):,} rows, "
              f"{out_path.stat().st_size / 1e6:.1f} MB)")

    print(f"  dropped {n_dropped} matches with no valid winner / abandoned")
    return n_matches, counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", type=Path, default=Path("data/t20s_json"))
    ap.add_argument("--extra-source-dir", type=Path, action="append",
                    default=[], help="Additional cricsheet pool(s) merged "
                    "chronologically with --source-dir. Use for the golden "
                    "pool: --extra-source-dir data/golden/t20s_json. May be "
                    "repeated.")
    ap.add_argument("--sqlite-dir", type=Path, default=Path("models"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("data/xgb_match_data_v1"))
    ap.add_argument("--metadata-csv", type=Path,
                    default=Path("data/all_players_enriched.csv"))
    ap.add_argument("--version", type=str, default="v3",
                    help="SQLite cache version (matches existing per-ball "
                    "pipeline).")
    ap.add_argument("--freeze-trackers-after", type=str, default=None,
                    help="Freeze A2 trackers + SQLite rehydration after this "
                    "date (YYYY-MM-DD). Diagnostic mode for the no-leakage "
                    "test — matches with date > this value all see the "
                    "snapshot at this date+1, with no cross-match updates "
                    "during the test period. Default: None (no freeze).")
    args = ap.parse_args()

    splits = dict(DEFAULT_SPLITS)
    source_dirs = [args.source_dir] + list(args.extra_source_dir)
    print(f"Source dir(s): {source_dirs}")
    t0 = time.time()
    n_matches, counts = materialize(
        source_dir=source_dirs,
        sqlite_dir=args.sqlite_dir,
        out_dir=args.out_dir,
        version=args.version,
        splits=splits,
        gender="male",
        metadata_csv=args.metadata_csv,
        freeze_trackers_after=args.freeze_trackers_after,
    )
    dt = time.time() - t0
    print(f"\nDONE: {n_matches:,} matches → {args.out_dir} in {dt:.0f}s")
    for name, n in counts.items():
        print(f"  {name}: {n:,} matches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
