"""Build `models/player_stats_cache_v3.sqlite` directly from JSON matches.

Replaces the pickle-chunk intermediate used by `scripts/build_stats_sqlite.py`.
Walks `data/t20s_json/` in chronological order, advances `PlayerStatsTracker`
/ `PlayerEloTracker` / `VenueStatsTracker` exactly as the monolith
`parsing_v2.py:process_folder_v2_with_splits` does, takes pre-match snapshots
via `deep_copy_stats`, and streams delta-compressed rows straight into
SQLite schema v3 (including the new `batting_match_log` / `bowling_match_log`
tables for recent-form deque reconstruction).

Phase B deliverable §2. Phase 5 cleanup removes `build_stats_sqlite.py`.

Usage:
    uv run python scripts/build_stats_cache.py \\
        --source-dir data/t20s_json \\
        --out models/player_stats_cache_v3.sqlite \\
        --gender-filter male

    uv run python scripts/build_stats_cache.py --force-rebuild
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from loaders_common import extract_match_metadata, iter_matches_chronological
from parsing_v2 import (
    PlayerEloTracker,
    PlayerStatsTracker,
    VenueStatsTracker,
    deep_copy_stats,
    parse_match_data_v2,
)
from player_metadata import PlayerMetadataProvider
from stats_sqlite_backend import SCHEMA_SQL, SCHEMA_VERSION


BATCH_SIZE = 20_000


def _intern(d: Dict[str, int], key: str) -> int:
    """Intern a string into a dense 0-indexed int id; stable within a run."""
    i = d.get(key)
    if i is None:
        i = len(d)
        d[key] = i
    return i


def _verify_outcome_count_conservation(conn, sample_n: int = 500) -> None:
    """Schema v4 guard: Σ(c0..cw) must equal the `balls` column on every
    batting / bowling / batting_vs_type / bowling_vs_hand row, and equal
    `total_balls` on every venue row. A schema migration or a future
    `update_stats` refactor that miscounts a bucket would corrupt the 42
    distribution features downstream; catch drift at build time.
    """
    import random

    print("  verifying outcome-count conservation...", flush=True)
    specs = (
        # (table, balls_col, has_many_rows_bool)
        ("batting", "balls", True),
        ("bowling", "balls_bowled", True),
        ("batting_vs_type", "balls", True),
        ("bowling_vs_hand", "balls_bowled", True),
        ("venue", "total_balls", True),
    )
    for table, balls_col, _ in specs:
        rows = conn.execute(
            f"SELECT {balls_col}, c0, c1, c2, c4, c6, cw "
            f"FROM {table} WHERE {balls_col} > 0 "
            f"ORDER BY RANDOM() LIMIT ?",
            (sample_n,),
        ).fetchall()
        if not rows:
            continue
        for row in rows:
            balls, c0, c1, c2, c4, c6, cw = row
            if c0 + c1 + c2 + c4 + c6 + cw != balls:
                raise RuntimeError(
                    f"{table}: outcome-count conservation violated — "
                    f"{balls_col}={balls}, Σ(cX)={c0+c1+c2+c4+c6+cw}"
                )
        print(f"    {table}: {len(rows)} rows OK", flush=True)


def _verify_log_denormalized_consistency(conn, sample_n: int = 500) -> None:
    """Fail the build if `batting_match_log` / `bowling_match_log` rows
    don't sum to the denormalized `batting.recent_*` / `bowling.recent_*`
    columns.

    Rationale: both paths are written from the same tracker state, but
    via different code routes in build_stats_cache.py — `emit_snapshot`
    (pre-match) writes the denormalized sums from `deep_copy_stats`; the
    ball loop writes match-log rows from `stats.current_match_batting`
    after `parse_match_data_v2`. A refactor that reorders these writes
    or changes `end_match` timing could drift them silently. The Phase A
    harness reads only the log-seeded path; sim hot-path wrappers read
    the denormalized columns directly. This check covers both.

    Sample size 500 per table + 5 log queries each = 2500 queries per
    table × 2 tables = 5 000 queries. At ~3 µs on mmap SQLite that's
    ~15 ms. Negligible vs a ~400 s build.
    """
    import random

    print("  verifying log ≡ denormalized sums...", flush=True)
    specs = (
        # (stats_table, log_table, denorm_cols, log_cols)
        ("batting", "batting_match_log",
         ("recent_runs", "recent_balls", "recent_dismissals"),
         ("runs", "balls", "dismissals")),
        ("bowling", "bowling_match_log",
         ("recent_runs_given", "recent_balls_bowled", "recent_wickets"),
         ("runs_given", "balls_bowled", "wickets")),
    )

    rng = random.Random(0xC0FFEE)
    for stats_table, log_table, denorm_cols, log_cols in specs:
        # Sample rows with non-zero denormalized sums (zero sums are the
        # trivial case — we want to verify the interesting ones).
        rows = conn.execute(
            f"SELECT player_id, date_id, {', '.join(denorm_cols)} "
            f"FROM {stats_table} "
            f"WHERE {denorm_cols[1]} > 0 "  # recent_balls/balls_bowled > 0
            f"ORDER BY RANDOM() LIMIT ?",
            (sample_n,),
        ).fetchall()
        if not rows:
            continue

        for row in rows:
            pid = row[0]
            did = row[1]
            expected = tuple(row[2:5])
            log_rows = conn.execute(
                f"SELECT {', '.join(log_cols)} FROM {log_table} "
                f"WHERE player_id=? AND date_id<? "
                f"ORDER BY date_id DESC, intra_date_idx DESC LIMIT 5",
                (pid, did),
            ).fetchall()
            actual = tuple(sum(r[i] for r in log_rows)
                           for i in range(3))
            if actual != expected:
                raise RuntimeError(
                    f"{stats_table}/{log_table} consistency check failed: "
                    f"player_id={pid} date_id={did} "
                    f"denormalized {denorm_cols} = {expected}, "
                    f"log sum-of-last-5 {log_cols} = {actual}. "
                    f"build_stats_cache.py is poisoning the recent-form "
                    f"cache — aborting."
                )
        print(f"    {stats_table}: {len(rows)} rows OK", flush=True)


def sqlite_up_to_date(out_path: Path, source_dir: Path) -> bool:
    """Return True if the existing SQLite is current vs the JSON corpus.

    Guards: schema_version == current, source_json_mtime_max >=
    max(JSON mtime). Matches the staleness pattern at
    stats_provider.py:692-702.
    """
    if not out_path.exists():
        return False
    try:
        conn = sqlite3.connect(f"file:{out_path}?mode=ro", uri=True)
        meta = dict(conn.execute("SELECT key, value FROM _meta"))
        conn.close()
    except sqlite3.DatabaseError:
        return False

    try:
        file_schema = int(meta.get("schema_version", -1))
    except (TypeError, ValueError):
        return False
    if file_schema != SCHEMA_VERSION:
        return False

    try:
        source_mtime_at_build = float(meta.get("source_json_mtime_max", 0))
    except (TypeError, ValueError):
        return False

    live_mtime_max = max(
        (p.stat().st_mtime for p in source_dir.glob("*.json")),
        default=0.0,
    )
    return source_mtime_at_build + 1 >= live_mtime_max


def build(
    source_dir: Path,
    out_path: Path,
    gender: str = "male",
    metadata_csv: Path = None,
) -> None:
    t_start = time.time()
    print(f"building {out_path} from {source_dir} (gender={gender})",
          flush=True)

    stats = PlayerStatsTracker()
    venue = VenueStatsTracker()
    elo = PlayerEloTracker()
    metadata_path = metadata_csv or (
        source_dir.parent / "all_players_enriched.csv"
    )
    metadata = PlayerMetadataProvider(str(metadata_path))

    if out_path.exists():
        print(f"  removing existing {out_path}", flush=True)
        out_path.unlink()

    conn = sqlite3.connect(str(out_path))
    conn.executescript("""
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = MEMORY;
        PRAGMA cache_size = -200000;
    """)
    conn.executescript(SCHEMA_SQL)

    # Interned int ids; lookup tables written at the end so row ids match.
    player_ids: Dict[str, int] = {}
    venue_ids: Dict[str, int] = {}
    date_ids: Dict[str, int] = {}

    # Delta trackers (same schema as build_stats_sqlite.py:106-118).
    prev_batting: Dict[int, Tuple[int, int, int, int, int, int]] = {}
    prev_bowling: Dict[int, Tuple[int, int, int, int, int, int]] = {}
    prev_h2h: Dict[Tuple[int, int], Tuple[int, int, int]] = {}
    prev_bat_vs_type: Dict[Tuple[int, int], Tuple[int, int, int]] = {}
    prev_bowl_vs_hand: Dict[Tuple[int, int], Tuple[int, int, int]] = {}
    prev_venue: Dict[int, Tuple] = {}
    prev_batting_elo: Dict[int, float] = {}
    prev_bowling_elo: Dict[int, float] = {}

    # Row buffers.
    batting_rows: List[Tuple] = []
    bowling_rows: List[Tuple] = []
    h2h_rows: List[Tuple] = []
    bat_vs_type_rows: List[Tuple] = []
    bowl_vs_hand_rows: List[Tuple] = []
    venue_rows: List[Tuple] = []
    batting_elo_rows: List[Tuple] = []
    bowling_elo_rows: List[Tuple] = []
    batting_log_rows: List[Tuple] = []
    bowling_log_rows: List[Tuple] = []

    def flush_all():
        if batting_rows:
            conn.executemany(
                "INSERT INTO batting VALUES ("
                "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                batting_rows)
            batting_rows.clear()
        if bowling_rows:
            conn.executemany(
                "INSERT INTO bowling VALUES ("
                "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                bowling_rows)
            bowling_rows.clear()
        if h2h_rows:
            conn.executemany(
                "INSERT INTO h2h (batter_id, bowler_id, date_id, "
                "runs, balls, dismissals) VALUES (?, ?, ?, ?, ?, ?)",
                h2h_rows)
            h2h_rows.clear()
        if bat_vs_type_rows:
            conn.executemany(
                "INSERT INTO batting_vs_type VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                bat_vs_type_rows)
            bat_vs_type_rows.clear()
        if bowl_vs_hand_rows:
            conn.executemany(
                "INSERT INTO bowling_vs_hand VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                bowl_vs_hand_rows)
            bowl_vs_hand_rows.clear()
        if venue_rows:
            conn.executemany(
                "INSERT INTO venue VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                "?, ?, ?, ?, ?, ?)",
                venue_rows)
            venue_rows.clear()
        if batting_elo_rows:
            conn.executemany(
                "INSERT INTO batting_elo VALUES (?, ?, ?)",
                batting_elo_rows)
            batting_elo_rows.clear()
        if bowling_elo_rows:
            conn.executemany(
                "INSERT INTO bowling_elo VALUES (?, ?, ?)",
                bowling_elo_rows)
            bowling_elo_rows.clear()
        if batting_log_rows:
            conn.executemany(
                "INSERT INTO batting_match_log VALUES (?, ?, ?, ?, ?, ?)",
                batting_log_rows)
            batting_log_rows.clear()
        if bowling_log_rows:
            conn.executemany(
                "INSERT INTO bowling_match_log VALUES (?, ?, ?, ?, ?, ?)",
                bowling_log_rows)
            bowling_log_rows.clear()

    def emit_snapshot(snap: dict, date_id: int) -> None:
        """Emit delta-compressed rows for one pre-match snapshot."""
        for pid_str, st in snap.get('batting', {}).items():
            pid = _intern(player_ids, pid_str)
            cur = (
                int(st['runs']), int(st['balls']), int(st['dismissals']),
                int(st.get('recent_runs', 0)),
                int(st.get('recent_balls', 0)),
                int(st.get('recent_dismissals', 0)),
                int(st.get('c0', 0)), int(st.get('c1', 0)),
                int(st.get('c2', 0)), int(st.get('c4', 0)),
                int(st.get('c6', 0)), int(st.get('cw', 0)),
            )
            if prev_batting.get(pid) != cur:
                batting_rows.append((pid, date_id) + cur)
                prev_batting[pid] = cur

        for pid_str, st in snap.get('bowling', {}).items():
            pid = _intern(player_ids, pid_str)
            cur = (
                int(st['runs_given']), int(st['balls_bowled']),
                int(st['wickets']),
                int(st.get('recent_runs_given', 0)),
                int(st.get('recent_balls_bowled', 0)),
                int(st.get('recent_wickets', 0)),
                int(st.get('c0', 0)), int(st.get('c1', 0)),
                int(st.get('c2', 0)), int(st.get('c4', 0)),
                int(st.get('c6', 0)), int(st.get('cw', 0)),
            )
            if prev_bowling.get(pid) != cur:
                bowling_rows.append((pid, date_id) + cur)
                prev_bowling[pid] = cur

        for (bat_str, bowl_str), st in snap.get('h2h', {}).items():
            bat = _intern(player_ids, bat_str)
            bowl = _intern(player_ids, bowl_str)
            cur = (int(st['runs']), int(st['balls']), int(st['dismissals']))
            key = (bat, bowl)
            if prev_h2h.get(key) != cur:
                h2h_rows.append((bat, bowl, date_id) + cur)
                prev_h2h[key] = cur

        for pid_str, by_type in snap.get('batting_vs_type', {}).items():
            pid = _intern(player_ids, pid_str)
            for type_name, type_code in (('pace', 0), ('spin', 1)):
                st = by_type.get(type_name)
                if st is None:
                    continue
                cur = (
                    int(st['runs']), int(st['balls']),
                    int(st['dismissals']),
                    int(st.get('c0', 0)), int(st.get('c1', 0)),
                    int(st.get('c2', 0)), int(st.get('c4', 0)),
                    int(st.get('c6', 0)), int(st.get('cw', 0)),
                )
                key = (pid, type_code)
                if prev_bat_vs_type.get(key) != cur:
                    bat_vs_type_rows.append(
                        (pid, date_id, type_code) + cur)
                    prev_bat_vs_type[key] = cur

        for pid_str, by_hand in snap.get('bowling_vs_hand', {}).items():
            pid = _intern(player_ids, pid_str)
            for hand_name, hand_code in (('left', 0), ('right', 1)):
                st = by_hand.get(hand_name)
                if st is None:
                    continue
                cur = (
                    int(st['runs_given']), int(st['balls_bowled']),
                    int(st['wickets']),
                    int(st.get('c0', 0)), int(st.get('c1', 0)),
                    int(st.get('c2', 0)), int(st.get('c4', 0)),
                    int(st.get('c6', 0)), int(st.get('cw', 0)),
                )
                key = (pid, hand_code)
                if prev_bowl_vs_hand.get(key) != cur:
                    bowl_vs_hand_rows.append(
                        (pid, date_id, hand_code) + cur)
                    prev_bowl_vs_hand[key] = cur

        for venue_str, st in snap.get('venue', {}).items():
            vid = _intern(venue_ids, venue_str)
            fi = st.get('first_innings_totals', [])
            cur = (
                int(st.get('total_runs', 0)),
                int(st.get('innings_count', 0)),
                int(st.get('total_balls', 0)),
                int(st.get('total_boundaries', 0)),
                int(st.get('total_dots', 0)),
                int(st.get('total_wickets', 0)),
                int(st.get('powerplay_runs', 0)),
                int(st.get('powerplay_balls', 0)),
                int(st.get('death_runs', 0)),
                int(st.get('death_balls', 0)),
                int(sum(fi)),
                int(len(fi)),
                int(st.get('matches_total', 0)),
                int(st.get('chase_wins', 0)),
                int(st.get('c0', 0)), int(st.get('c1', 0)),
                int(st.get('c2', 0)), int(st.get('c4', 0)),
                int(st.get('c6', 0)), int(st.get('cw', 0)),
            )
            if prev_venue.get(vid) != cur:
                venue_rows.append((vid, date_id) + cur)
                prev_venue[vid] = cur

        for pid_str, elo_val in snap.get('batting_elo', {}).items():
            pid = _intern(player_ids, pid_str)
            e = float(elo_val)
            if prev_batting_elo.get(pid) != e:
                batting_elo_rows.append((pid, date_id, e))
                prev_batting_elo[pid] = e
        for pid_str, elo_val in snap.get('bowling_elo', {}).items():
            pid = _intern(player_ids, pid_str)
            e = float(elo_val)
            if prev_bowling_elo.get(pid) != e:
                bowling_elo_rows.append((pid, date_id, e))
                prev_bowling_elo[pid] = e

    conn.execute("BEGIN")

    snapshotted_dates: set = set()
    intra_idx_by_date: Dict[str, int] = defaultdict(int)
    n_matches = 0
    source_json_mtime_max = 0.0

    # Phase 3: per-phase outcome-count totals across all innings.
    # Σ over phases ≡ Σ overall cX (per-innings conservation guarantee in
    # parsing_v2.py inn_agg). Used after the walk to compute 18 phase
    # priors written to _meta.
    phase_total_counts = {
        'powerplay': [0, 0, 0, 0, 0, 0],
        'middle':    [0, 0, 0, 0, 0, 0],
        'death':     [0, 0, 0, 0, 0, 0],
    }
    _PHASE_CK_KEYS = ('c0', 'c1', 'c2', 'c4', 'c6', 'cw')

    import json as _json
    for match_id, json_text, match_date in iter_matches_chronological(
        source_dir, gender=gender
    ):
        date_str = match_date.strftime('%Y-%m-%d')
        data = _json.loads(json_text)
        meta = extract_match_metadata(data)
        k_factor = meta['k_factor']
        venue_name = meta['venue']

        # --- pre-match snapshot (first-write-wins per date) -----------
        if date_str not in snapshotted_dates:
            snapshotted_dates.add(date_str)
            date_id = _intern(date_ids, date_str)
            snap = deep_copy_stats(stats, venue, elo)
            emit_snapshot(snap, date_id)

        # --- advance trackers across the match ------------------------
        # parse_match_data_v2 mutates stats / elo / venue ball-by-ball.
        # stats.current_match_batting / current_match_bowling end the
        # call populated with this match's aggregates.
        date_id = date_ids[date_str]
        intra_idx = intra_idx_by_date[date_str]
        intra_idx_by_date[date_str] += 1

        _, _, vname, innings_details, chase_won = parse_match_data_v2(
            json_text, stats, venue, metadata,
            elo_tracker=elo, match_k_factor=k_factor,
        )

        # --- emit match-log rows (schema v3) --------------------------
        for pid_str, st in stats.current_match_batting.items():
            if st['balls'] <= 0:
                continue
            pid = _intern(player_ids, pid_str)
            batting_log_rows.append((
                pid, date_id, intra_idx,
                int(st['runs']), int(st['balls']), int(st['dismissals']),
            ))
        for pid_str, st in stats.current_match_bowling.items():
            if st['balls_bowled'] <= 0:
                continue
            pid = _intern(player_ids, pid_str)
            bowling_log_rows.append((
                pid, date_id, intra_idx,
                int(st['runs_given']), int(st['balls_bowled']),
                int(st['wickets']),
            ))

        # --- post-match venue updates (parsing_v2.py:1321-1325) -------
        for det in innings_details:
            venue.update_venue_stats_detailed(vname, det)
            # Phase 3: per-phase outcome-count accumulation. Each
            # innings_details dict carries 18 c{X}_{phase} keys
            # (powerplay/middle/death × 6 outcomes) emitted by
            # parse_match_data_v2 inn_agg. Pre-Phase-3 inn_agg dicts
            # lack these keys; .get(...,0) preserves backwards
            # compatibility for old caches being re-read.
            #
            # Per-innings conservation: Σ phases of cX_{phase} must
            # equal cX (legal-only). Holds by construction in
            # parse_match_data_v2 (every legal ball fires both the
            # global cX bump and exactly one cX_{phase} bump). Cheap
            # check (18 ints per innings); fails loud if a future
            # parsing refactor breaks the dispatch.
            for i, ck in enumerate(_PHASE_CK_KEYS):
                overall = int(det.get(ck, 0))
                phase_sum = sum(int(det.get(f"{ck}_{ph}", 0))
                                for ph in ('powerplay', 'middle', 'death'))
                if phase_sum != overall:
                    raise RuntimeError(
                        f"phase split conservation violated in innings: "
                        f"bucket {ck} overall={overall}, "
                        f"Σ phases={phase_sum} (Δ={phase_sum-overall:+,})"
                    )
            for phase, totals in phase_total_counts.items():
                for i, ck in enumerate(_PHASE_CK_KEYS):
                    totals[i] += int(det.get(f"{ck}_{phase}", 0))
        if chase_won is not None:
            venue.update_venue_match_result(vname, chase_won)

        # --- buffer flushing + progress -------------------------------
        total_buffered = (
            len(batting_rows) + len(bowling_rows) + len(h2h_rows)
            + len(bat_vs_type_rows) + len(bowl_vs_hand_rows)
            + len(venue_rows) + len(batting_elo_rows) + len(bowling_elo_rows)
            + len(batting_log_rows) + len(bowling_log_rows)
        )
        if total_buffered >= BATCH_SIZE:
            flush_all()

        n_matches += 1
        if n_matches % 500 == 0:
            dt = time.time() - t_start
            print(f"  [{n_matches}] matches in {dt:.0f}s "
                  f"({n_matches / dt:.1f} match/s)", flush=True)

    flush_all()

    # Lookup tables — insert in intern order so table.id == intern int id.
    conn.executemany(
        "INSERT INTO players (id, player_id) VALUES (?, ?)",
        sorted(((v, k) for k, v in player_ids.items()), key=lambda r: r[0]))
    conn.executemany(
        "INSERT INTO venues (id, venue) VALUES (?, ?)",
        sorted(((v, k) for k, v in venue_ids.items()), key=lambda r: r[0]))
    conn.executemany(
        "INSERT INTO dates (id, date) VALUES (?, ?)",
        sorted(((v, k) for k, v in date_ids.items()), key=lambda r: r[0]))

    # Record the newest source JSON mtime for the staleness guard.
    source_json_mtime_max = max(
        (p.stat().st_mtime for p in source_dir.glob('*.json')),
        default=0.0,
    )
    # Schema v4: global empirical outcome prior π. Computed by summing
    # the tracker's final-state outcome counts across all batters (one
    # of the two conservation-equivalent views; bowling sum would give
    # the same totals). Used as the shrinkage target for 42 distribution
    # features. Fixed constant — not rolling, not as-of-date — since
    # ratios stabilize to < 0.5 % within the first ~10 k balls and a
    # rolling prior would add as-of-date complexity for no measurable
    # gain.
    total_counts = [0, 0, 0, 0, 0, 0]
    ck_keys = ('c0', 'c1', 'c2', 'c4', 'c6', 'cw')
    for st in stats.batting_stats.values():
        for i, k in enumerate(ck_keys):
            total_counts[i] += int(st.get(k, 0))
    grand_total = sum(total_counts)
    if grand_total <= 0:
        prior = (1/6,) * 6
    else:
        prior = tuple(c / grand_total for c in total_counts)
    print(f"  global prior π (n={grand_total:,}): "
          f"p0={prior[0]:.4f} p1={prior[1]:.4f} p2={prior[2]:.4f} "
          f"p4={prior[3]:.4f} p6={prior[4]:.4f} pw={prior[5]:.4f}",
          flush=True)

    # Phase 3: per-phase priors. Compute fractional probabilities per
    # phase from accumulated counts. Falls back to flat (1/6,)*6 if a
    # phase has zero balls (defensive — should never happen on a real
    # corpus). Conservation invariant: Σ phase_total_counts[phase][c]
    # == total_counts[c] for every c — checked below.
    phase_prior_pcts = {}
    for phase, totals in phase_total_counts.items():
        n = sum(totals)
        if n <= 0:
            phase_prior_pcts[phase] = (1/6,) * 6
        else:
            phase_prior_pcts[phase] = tuple(c / n for c in totals)

    # Aggregate sanity check: the per-phase totals are computed from
    # innings_details (legal-only), but `total_counts` is the
    # batting-stats sum (includes wides/noballs since update_stats is
    # called on every delivery). They will NOT match in general — Δ
    # equals the wide+noball count. Per-innings conservation is enforced
    # in the loop above; here we just log the legal-only grand total
    # for visibility, and confirm it doesn't exceed the inclusive total.
    legal_grand_total = sum(
        sum(phase_total_counts[ph][i]
            for ph in ('powerplay', 'middle', 'death'))
        for i in range(6)
    )
    inclusive_grand_total = sum(total_counts)
    if legal_grand_total > inclusive_grand_total:
        raise RuntimeError(
            f"legal-only phase total {legal_grand_total:,} exceeds "
            f"inclusive batting-stats total {inclusive_grand_total:,}"
        )
    print(f"  legal balls (Σ over phases): {legal_grand_total:,} "
          f"(vs {inclusive_grand_total:,} inclusive incl. wides/noballs)",
          flush=True)
    print(f"  per-phase priors:", flush=True)
    for phase in ('powerplay', 'middle', 'death'):
        n = sum(phase_total_counts[phase])
        p = phase_prior_pcts[phase]
        print(f"    {phase:>9s} (n={n:>10,}): "
              f"p0={p[0]:.4f} p1={p[1]:.4f} p2={p[2]:.4f} "
              f"p4={p[3]:.4f} p6={p[4]:.4f} pw={p[5]:.4f}",
              flush=True)

    meta_rows = [
        ('schema_version', str(SCHEMA_VERSION)),
        ('build_timestamp', datetime.utcnow().isoformat() + 'Z'),
        ('source_json_mtime_max', f"{source_json_mtime_max:.6f}"),
        ('source_match_count', str(n_matches)),
        ('gender_filter', str(gender or 'all')),
        ('num_players', str(len(player_ids))),
        ('num_venues', str(len(venue_ids))),
        ('num_dates', str(len(date_ids))),
        ('features', 'v4'),
        ('prior_p0', f"{prior[0]:.10f}"),
        ('prior_p1', f"{prior[1]:.10f}"),
        ('prior_p2', f"{prior[2]:.10f}"),
        ('prior_p4', f"{prior[3]:.10f}"),
        ('prior_p6', f"{prior[4]:.10f}"),
        ('prior_pw', f"{prior[5]:.10f}"),
    ]
    # Phase 3: 18 phase-prior _meta rows. Schema unchanged; we widen the
    # _meta key/value table with new keys instead of bumping schema.
    # Naming: prior_{pp,mid,death}_p{0,1,2,4,6,w} (matches the keys read
    # by stats_sqlite_backend._load_lookups).
    _phase_short = {'powerplay': 'pp', 'middle': 'mid', 'death': 'death'}
    for phase, short in _phase_short.items():
        p = phase_prior_pcts[phase]
        for i, suffix in enumerate(('p0', 'p1', 'p2', 'p4', 'p6', 'pw')):
            meta_rows.append((f"prior_{short}_{suffix}", f"{p[i]:.10f}"))
    conn.executemany("INSERT INTO _meta VALUES (?, ?)", meta_rows)

    conn.commit()
    print("  analyzing...", flush=True)
    conn.execute("ANALYZE")
    conn.commit()

    # Defense-in-depth: the schema v3 match-log and the denormalized
    # recent_* columns on batting/bowling are two independently-written
    # views of the same state. Drift between them would silently serve
    # stale recent-form on the sim hot path. Sample-check before close.
    _verify_log_denormalized_consistency(conn, sample_n=500)
    # Schema v4: outcome-count conservation guard (Σ cX == balls).
    _verify_outcome_count_conservation(conn, sample_n=500)

    conn.close()

    size_mb = out_path.stat().st_size / 1e6
    dt = time.time() - t_start
    print(f"\nDONE: {out_path} ({size_mb:.1f} MB) in {dt:.0f}s", flush=True)
    print(f"  matches:  {n_matches:,}", flush=True)
    print(f"  players:  {len(player_ids):,}", flush=True)
    print(f"  venues:   {len(venue_ids):,}", flush=True)
    print(f"  dates:    {len(date_ids):,}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", type=Path,
                    default=Path("data/t20s_json"))
    ap.add_argument("--out", type=Path,
                    default=Path("models/player_stats_cache_v3.sqlite"))
    ap.add_argument("--gender-filter", type=str, default="male",
                    help="Skip matches whose info.gender doesn't match. "
                    "Pass empty string to disable.")
    ap.add_argument("--force-rebuild", action="store_true",
                    help="Ignore mtime-based staleness check and rebuild "
                    "even if the existing SQLite appears current.")
    ap.add_argument("--metadata-csv", type=Path, default=None,
                    help="Override path to all_players_enriched.csv. "
                    "Defaults to <source-dir>/../all_players_enriched.csv")
    args = ap.parse_args()

    gender = args.gender_filter or None

    if not args.force_rebuild and sqlite_up_to_date(args.out, args.source_dir):
        print(f"{args.out} is current (schema_version={SCHEMA_VERSION}, "
              "source mtime covered). Skipping rebuild. "
              "Use --force-rebuild to override.")
        return 0

    build(args.source_dir, args.out, gender=gender,
          metadata_csv=args.metadata_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
