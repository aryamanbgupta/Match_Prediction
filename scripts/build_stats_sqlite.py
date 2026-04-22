"""One-shot converter: pickle cache chunks → SQLite DB.

Reads `models/cache_chunks_v3/*.pkl` (plus the v3 metadata file), walks
snapshots in global chronological order, and emits delta-compressed
rows into a single SQLite file.

Delta compression: for each (entity, stat_table), the previous tuple of
values is tracked in memory; a row is appended only when the current
snapshot's values differ. Readers recover "value as of date D" via
`WHERE key = ? AND date_id <= ? ORDER BY date_id DESC LIMIT 1`, which
returns the most recent emitted row ≤ D.

Boundary handling: 45 dates appear in two consecutive chunks (same
calendar day, more than one match). StatsProvider resolves this
last-write-wins (later chunk is authoritative). This converter does
the same — when processing a date, it skips if the date reappears in
a later chunk, and emits only at its final chunk. Because chunks are
globally sorted, the date_ids end up in chronological order.

Build-time pragmas prioritize throughput over durability (synchronous
= OFF, journal_mode = OFF). The resulting file is sealed with
PRAGMA integrity_check at the end.

Usage:
    uv run python scripts/build_stats_sqlite.py \\
        --chunks-dir models/cache_chunks_v3 \\
        --metadata models/player_stats_cache_v3_metadata.pkl \\
        --out models/player_stats_cache_v3.sqlite \\
        --max-chunks 5
"""
from __future__ import annotations

import argparse
import pickle
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats_sqlite_backend import SCHEMA_SQL, SCHEMA_VERSION


BATCH_SIZE = 20_000


def _intern(d: Dict[str, int], key: str) -> int:
    """Intern a string into a dense 0-indexed int id; stable within a run."""
    i = d.get(key)
    if i is None:
        i = len(d)
        d[key] = i
    return i


def build(chunks_dir: Path, metadata_path: Path, out_path: Path,
          max_chunks: int | None = None) -> None:
    t_start = time.time()

    print(f"loading metadata: {metadata_path}", flush=True)
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    n_chunks = metadata['num_chunks']
    if max_chunks is not None:
        n_chunks = min(n_chunks, max_chunks)
    print(f"  processing {n_chunks} chunks (of {metadata['num_chunks']} total)",
          flush=True)

    # First pass: for each date, record the FIRST chunk it appears in.
    # The first snapshot for a date is the one taken BEFORE the first match
    # of that date (correct pre-day state). Later snapshots (when a date
    # straddles a chunk boundary) reflect mid-day state after an earlier
    # same-day match, which leaks forward into inference.
    final_chunk_of_date: Dict[str, int] = {}
    for chunk_idx in range(n_chunks):
        for d in metadata['chunks'][chunk_idx]['dates']:
            if d not in final_chunk_of_date:
                final_chunk_of_date[d] = chunk_idx  # first-write-wins
    print(f"  unique dates across selected chunks: {len(final_chunk_of_date):,}",
          flush=True)

    if out_path.exists():
        print(f"  removing existing {out_path}", flush=True)
        out_path.unlink()

    conn = sqlite3.connect(str(out_path))
    conn.executescript("""
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = MEMORY;
        PRAGMA cache_size = -200000;  -- 200 MB page cache during build
    """)
    conn.executescript(SCHEMA_SQL)

    # Interned int ids, built incrementally. Bulk-inserted into lookup
    # tables at the end so the table rowids exactly match the intern ids.
    player_ids: Dict[str, int] = {}
    venue_ids: Dict[str, int] = {}

    # date_id assigned in the order we commit dates, which is chronological.
    date_rows: List[Tuple[int, str]] = []

    # Delta trackers. Keyed by interned int ids so the dicts stay compact.
    # Row schema (v2): (runs, balls, dismissals, recent_runs, recent_balls,
    #   recent_dismissals) for batting; (runs_given, balls_bowled, wickets,
    #   recent_runs_given, recent_balls_bowled, recent_wickets) for bowling.
    # Delta compression fires when *any* of the 6 counters changes.
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

    def flush_all():
        if batting_rows:
            conn.executemany(
                "INSERT INTO batting VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                batting_rows)
            batting_rows.clear()
        if bowling_rows:
            conn.executemany(
                "INSERT INTO bowling VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                bowling_rows)
            bowling_rows.clear()
        if h2h_rows:
            conn.executemany(
                "INSERT INTO h2h (batter_id, bowler_id, date_id, runs, balls, dismissals) "
                "VALUES (?, ?, ?, ?, ?, ?)", h2h_rows)
            h2h_rows.clear()
        if bat_vs_type_rows:
            conn.executemany(
                "INSERT INTO batting_vs_type VALUES (?, ?, ?, ?, ?, ?)",
                bat_vs_type_rows)
            bat_vs_type_rows.clear()
        if bowl_vs_hand_rows:
            conn.executemany(
                "INSERT INTO bowling_vs_hand VALUES (?, ?, ?, ?, ?, ?)",
                bowl_vs_hand_rows)
            bowl_vs_hand_rows.clear()
        if venue_rows:
            conn.executemany(
                "INSERT INTO venue VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                venue_rows)
            venue_rows.clear()
        if batting_elo_rows:
            conn.executemany(
                "INSERT INTO batting_elo VALUES (?, ?, ?)", batting_elo_rows)
            batting_elo_rows.clear()
        if bowling_elo_rows:
            conn.executemany(
                "INSERT INTO bowling_elo VALUES (?, ?, ?)", bowling_elo_rows)
            bowling_elo_rows.clear()

    conn.execute("BEGIN")

    next_date_id = 0
    for chunk_idx in range(n_chunks):
        chunk_info = metadata['chunks'][chunk_idx]
        chunk_path = chunks_dir.parent / chunk_info['file']
        t0 = time.time()
        with open(chunk_path, 'rb') as f:
            chunk_data = pickle.load(f)
        load_dt = time.time() - t0
        print(f"  [chunk {chunk_idx}/{n_chunks-1}] loaded in {load_dt:.1f}s, "
              f"{len(chunk_info['dates'])} dates", flush=True)

        for date_str in sorted(chunk_info['dates']):
            if final_chunk_of_date[date_str] != chunk_idx:
                # A later chunk supersedes this date — skip.
                continue
            date_id = next_date_id
            next_date_id += 1
            date_rows.append((date_id, date_str))

            snap = chunk_data[date_str]

            # --- batting -------------------------------------------------
            for pid_str, st in snap.get('batting', {}).items():
                pid = _intern(player_ids, pid_str)
                cur = (
                    int(st['runs']), int(st['balls']), int(st['dismissals']),
                    int(st.get('recent_runs', 0)),
                    int(st.get('recent_balls', 0)),
                    int(st.get('recent_dismissals', 0)),
                )
                if prev_batting.get(pid) != cur:
                    batting_rows.append((pid, date_id) + cur)
                    prev_batting[pid] = cur

            # --- bowling -------------------------------------------------
            for pid_str, st in snap.get('bowling', {}).items():
                pid = _intern(player_ids, pid_str)
                cur = (
                    int(st['runs_given']), int(st['balls_bowled']),
                    int(st['wickets']),
                    int(st.get('recent_runs_given', 0)),
                    int(st.get('recent_balls_bowled', 0)),
                    int(st.get('recent_wickets', 0)),
                )
                if prev_bowling.get(pid) != cur:
                    bowling_rows.append((pid, date_id) + cur)
                    prev_bowling[pid] = cur

            # --- h2h -----------------------------------------------------
            for (bat_str, bowl_str), st in snap.get('h2h', {}).items():
                bat = _intern(player_ids, bat_str)
                bowl = _intern(player_ids, bowl_str)
                cur = (int(st['runs']), int(st['balls']), int(st['dismissals']))
                key = (bat, bowl)
                if prev_h2h.get(key) != cur:
                    h2h_rows.append((bat, bowl, date_id) + cur)
                    prev_h2h[key] = cur

            # --- batting_vs_type (pace=0, spin=1) ------------------------
            for pid_str, by_type in snap.get('batting_vs_type', {}).items():
                pid = _intern(player_ids, pid_str)
                for type_name, type_code in (('pace', 0), ('spin', 1)):
                    st = by_type.get(type_name)
                    if st is None:
                        continue
                    cur = (int(st['runs']), int(st['balls']),
                           int(st['dismissals']))
                    key = (pid, type_code)
                    if prev_bat_vs_type.get(key) != cur:
                        bat_vs_type_rows.append((pid, date_id, type_code) + cur)
                        prev_bat_vs_type[key] = cur

            # --- bowling_vs_hand (left=0, right=1) -----------------------
            for pid_str, by_hand in snap.get('bowling_vs_hand', {}).items():
                pid = _intern(player_ids, pid_str)
                for hand_name, hand_code in (('left', 0), ('right', 1)):
                    st = by_hand.get(hand_name)
                    if st is None:
                        continue
                    cur = (int(st['runs_given']), int(st['balls_bowled']),
                           int(st['wickets']))
                    key = (pid, hand_code)
                    if prev_bowl_vs_hand.get(key) != cur:
                        bowl_vs_hand_rows.append(
                            (pid, date_id, hand_code) + cur)
                        prev_bowl_vs_hand[key] = cur

            # --- venue (flatten first_innings_totals list to sum+count) --
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
                )
                if prev_venue.get(vid) != cur:
                    venue_rows.append((vid, date_id) + cur)
                    prev_venue[vid] = cur

            # --- elos ----------------------------------------------------
            for pid_str, elo in snap.get('batting_elo', {}).items():
                pid = _intern(player_ids, pid_str)
                e = float(elo)
                if prev_batting_elo.get(pid) != e:
                    batting_elo_rows.append((pid, date_id, e))
                    prev_batting_elo[pid] = e
            for pid_str, elo in snap.get('bowling_elo', {}).items():
                pid = _intern(player_ids, pid_str)
                e = float(elo)
                if prev_bowling_elo.get(pid) != e:
                    bowling_elo_rows.append((pid, date_id, e))
                    prev_bowling_elo[pid] = e

            total_buffered = (
                len(batting_rows) + len(bowling_rows) + len(h2h_rows) +
                len(bat_vs_type_rows) + len(bowl_vs_hand_rows) +
                len(venue_rows) + len(batting_elo_rows) + len(bowling_elo_rows)
            )
            if total_buffered >= BATCH_SIZE:
                flush_all()

        # Drop chunk before loading next — keeps working-set RSS bounded.
        del chunk_data

    flush_all()

    # Lookup tables. Insert in intern order so table.id == intern int id.
    conn.executemany(
        "INSERT INTO players (id, player_id) VALUES (?, ?)",
        sorted(((v, k) for k, v in player_ids.items()), key=lambda r: r[0]))
    conn.executemany(
        "INSERT INTO venues (id, venue) VALUES (?, ?)",
        sorted(((v, k) for k, v in venue_ids.items()), key=lambda r: r[0]))
    conn.executemany(
        "INSERT INTO dates (id, date) VALUES (?, ?)", date_rows)

    # Meta. source_chunks_mtime_max is recorded so runtime can detect
    # when the pickle source has moved since this build.
    mtime_max = max(
        (chunks_dir.parent / c['file']).stat().st_mtime
        for c in metadata['chunks'][:n_chunks]
    )
    meta_rows = [
        ('schema_version', str(SCHEMA_VERSION)),
        ('build_timestamp', datetime.utcnow().isoformat() + 'Z'),
        ('source_chunks_mtime_max', f"{mtime_max:.6f}"),
        ('source_num_chunks', str(n_chunks)),
        ('num_players', str(len(player_ids))),
        ('num_venues', str(len(venue_ids))),
        ('num_dates', str(len(date_rows))),
        ('features', 'v3'),
    ]
    conn.executemany("INSERT INTO _meta VALUES (?, ?)", meta_rows)

    conn.commit()
    print("  analyzing...", flush=True)
    conn.execute("ANALYZE")
    conn.commit()
    conn.close()

    # Report.
    size_mb = out_path.stat().st_size / 1e6
    dt = time.time() - t_start
    print(f"\nDONE: {out_path}  ({size_mb:.1f} MB)  in {dt:.1f}s", flush=True)
    print(f"  players: {len(player_ids):,}  venues: {len(venue_ids):,}  "
          f"dates: {len(date_rows):,}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chunks-dir', type=Path,
                    default=Path('models/cache_chunks_v3'))
    ap.add_argument('--metadata', type=Path,
                    default=Path('models/player_stats_cache_v3_metadata.pkl'))
    ap.add_argument('--out', type=Path,
                    default=Path('models/player_stats_cache_v3.sqlite'))
    ap.add_argument('--max-chunks', type=int, default=None,
                    help="Process only the first N chunks (POC builds).")
    args = ap.parse_args()

    build(args.chunks_dir, args.metadata, args.out, args.max_chunks)


if __name__ == '__main__':
    main()
