"""Phase A parity harness: per-date SQLite-seeded materializer vs monolith.

The corpus has 3664 unique dates and 9519 male T20 matches — 61.5% of
matches share their date with at least one other match. The monolith
accumulates tracker state continuously across same-day matches, but
SQLite stores one snapshot per date (taken before the first match of
that date). A per-match rehydration cannot reproduce within-day drift,
so the harness groups matches by date:

  Per date D:
    1. Collect all same-day matches in monolith iteration order.
    2. Rehydrate temp_* trackers ONCE from SQLite at D, using the union
       of all same-day players + venues.
    3. For each match M on D:
       a. Reference path: parse_match_data_v2(json, live_*); advances
          live_* (ball-by-ball) + post-match venue updates.
       b. Candidate path: parse_match_data_v2(json, temp_*); temp_*
          accumulates across same-day matches; post-match temp_venue
          updates mirror the monolith.
       c. assert_frame_equal(candidate, reference, check_exact=True,
          check_dtype=True).
    4. Discard temp_*; on the next date, rehydrate freshly.

This mirrors the Phase B materializer architecture: cross-date stateless,
within-date stateful. Single-match dates (40% of the corpus) fall out as
the trivial case — one match per batch.

Known limitation (schema v2): recent-form is stored as a single summed
triple per (player, date). The monolith's underlying 5-entry deque
evicts its oldest slot on the first append inside a same-day batch,
which a single-sum seed cannot reproduce. 4 of 63 columns
(batsman_recent_avg/sr, bowler_recent_avg/econ) are therefore excluded
from the comparison on same-day secondary matches (~60% of the corpus).
Deferred fix: schema v3 bump to store deque entries individually,
planned for Phase B. See IMPROVEMENTS.md §"Parsing Pipeline Split".

The --skip-same-day-secondary flag is retained for diagnostic use: it
disables within-date accumulation and checks only first-of-date matches.
Useful when isolating a regression from the same-day architecture from
a bug elsewhere.

Fails fast after MAX_MISMATCHES divergent matches; prints the first
mismatched column + offending row for each.

This is the primary gate for Phase B — the split scripts must pass the
same harness before legacy code gets deleted.

Run:
    uv run python scripts/tests/test_phase_a_parity.py
    uv run python scripts/tests/test_phase_a_parity.py --limit 500
    uv run python scripts/tests/test_phase_a_parity.py --inject-fault elo
    uv run python scripts/tests/test_phase_a_parity.py --skip-same-day-secondary

Exit code: 0 = PASS, 1 = any mismatch.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from loaders_common import iter_matches_chronological  # noqa: E402
from parsing_v2 import (  # noqa: E402
    PlayerEloTracker,
    PlayerStatsTracker,
    VenueStatsTracker,
    classify_match_k_factor,
    parse_match_data_v2,
)
from player_metadata import PlayerMetadataProvider  # noqa: E402
from stats_provider import StatsProvider  # noqa: E402
from tracker_rehydration import (  # noqa: E402
    extract_match_player_ids,
    rehydrate_elo_tracker,
    rehydrate_stats_tracker,
    rehydrate_venue_tracker,
)


MAX_MISMATCHES = 5
PROGRESS_EVERY = 500

# SQLite schema v2 stores the 5-match recent-form window as a single summed
# triple (recent_runs/balls/dismissals). The monolith keeps a deque whose
# eviction order is lost in that compression, so the FIRST append inside a
# same-day batch diverges for players already at deque.maxlen: monolith
# evicts-and-appends; candidate just appends. Affects ~0.25% of matches on
# 4 of 63 columns. Deferred to Phase B via a schema-v3 bump that stores the
# deque entries individually. See IMPROVEMENTS.md §"Parsing Pipeline Split".
RECENT_FORM_COLUMNS = frozenset({
    "batsman_recent_avg",
    "batsman_recent_sr",
    "bowler_recent_avg",
    "bowler_recent_econ",
})


def _first_diff_report(candidate: pd.DataFrame, reference: pd.DataFrame) -> str:
    """Return a short human-readable diff for the first mismatching cell."""
    if candidate.shape != reference.shape:
        return (
            f"shape mismatch: candidate={candidate.shape}, "
            f"reference={reference.shape}"
        )
    if list(candidate.columns) != list(reference.columns):
        cand_only = set(candidate.columns) - set(reference.columns)
        ref_only = set(reference.columns) - set(candidate.columns)
        return (
            f"column set mismatch: candidate_only={cand_only}, "
            f"reference_only={ref_only}"
        )
    for col in candidate.columns:
        c = candidate[col]
        r = reference[col]
        if c.dtype != r.dtype:
            return f"dtype differs on {col!r}: cand={c.dtype}, ref={r.dtype}"
        # `equals` handles NaN == NaN as True, unlike ==. We want bit-exact.
        if c.equals(r):
            continue
        mask = c.ne(r) & ~(c.isna() & r.isna())
        if not mask.any():
            continue
        idx = mask.idxmax()
        return (
            f"col={col!r} row={idx} cand={c.iloc[idx]!r} ref={r.iloc[idx]!r}"
        )
    return "<no diff found — DataFrames compare equal>"


def _inject_fault(stats, elo, venue, kind: str) -> None:
    """Smoke-test the diff reporter by corrupting one tracker value."""
    if kind == "elo":
        if elo.batting_elo:
            pid = next(iter(elo.batting_elo))
            elo.batting_elo[pid] = elo.batting_elo[pid] + 50.0
    elif kind == "batting":
        if stats.batting_stats:
            pid = next(iter(stats.batting_stats))
            stats.batting_stats[pid]["runs"] += 100
    elif kind == "venue":
        if venue.venue_stats:
            v = next(iter(venue.venue_stats))
            venue.venue_stats[v]["total_runs"] += 500
    else:
        raise ValueError(f"unknown --inject-fault kind: {kind}")


def _group_by_date(source_dir: Path) -> Iterator[tuple[datetime, list]]:
    """Yield (date, [ (match_id, json_text, data_dict, venue, k_factor) ... ])
    batches. Consecutive same-date matches are collected into one list."""
    buf_date: Optional[datetime] = None
    buf: list = []

    for match_id, json_text, match_date in iter_matches_chronological(
        source_dir, gender="male"
    ):
        data = json.loads(json_text)
        event_info = data["info"].get("event", {})
        event_name = (
            event_info.get("name", "") if isinstance(event_info, dict) else ""
        )
        team_type = data["info"].get("team_type", "unknown")
        teams = data["info"].get("teams", [])
        k_factor = classify_match_k_factor(event_name, team_type, teams)
        venue = data["info"].get("venue", "unknown")
        entry = (match_id, json_text, data, venue, k_factor)

        if buf_date is None:
            buf_date, buf = match_date, [entry]
        elif match_date == buf_date:
            buf.append(entry)
        else:
            yield buf_date, buf
            buf_date, buf = match_date, [entry]

    if buf_date is not None:
        yield buf_date, buf


def run_harness(
    source_dir: Path,
    limit: Optional[int],
    inject_fault: Optional[str],
    skip_same_day_secondary: bool = False,
) -> int:
    provider = StatsProvider(str(ROOT / "models"), version="v3")
    if provider.backend_name != "sqlite":
        print(
            "ERROR: Phase A harness requires SQLite backend; got "
            f"{provider.backend_name!r}. Run "
            "`uv run python scripts/build_stats_sqlite.py` first.",
            file=sys.stderr,
        )
        return 1

    player_metadata = PlayerMetadataProvider(
        str(ROOT / "data" / "all_players_enriched.csv")
    )

    live_stats = PlayerStatsTracker()
    live_elo = PlayerEloTracker()
    live_venue = VenueStatsTracker()

    mismatches: list[tuple[str, str, str]] = []
    n_checked = 0
    n_checked_full = 0       # all 63 cols compared
    n_checked_partial = 0    # recent-form excluded (same-day secondary)
    n_skipped_before_cache = 0
    n_skipped_same_day = 0
    t_start = time.time()
    cache_min_date = provider.dates[0]
    injected_once = False

    for match_date, batch in _group_by_date(source_dir):
        if limit is not None and n_checked >= limit:
            break
        date_str = match_date.strftime("%Y-%m-%d")

        # --- Rehydrate temp trackers once per date ---------------------
        # Union of players + venues across all same-day matches.
        union_pids: set[str] = set()
        union_venues: set[str] = set()
        for _, _, data, venue, _ in batch:
            union_pids.update(extract_match_player_ids(data))
            union_venues.add(venue)

        in_cache = date_str >= cache_min_date
        if in_cache:
            temp_stats = rehydrate_stats_tracker(
                provider, match_date, union_pids)
            temp_elo = rehydrate_elo_tracker(
                provider, match_date, union_pids)
            temp_venue = rehydrate_venue_tracker(
                provider, match_date, union_venues)
        else:
            temp_stats = temp_elo = temp_venue = None  # won't be used

        # --- Iterate same-day matches ----------------------------------
        for i, (match_id, json_text, data, venue, k_factor) in enumerate(batch):
            if limit is not None and n_checked >= limit:
                break

            # Reference path: live_* (matches monolith's chronological walk).
            try:
                (ref_rows, _itotals, ref_venue, innings_details, chase_won
                 ) = parse_match_data_v2(
                    json_text,
                    live_stats,
                    live_venue,
                    player_metadata,
                    elo_tracker=live_elo,
                    match_k_factor=k_factor,
                )
            except Exception as e:
                print(
                    f"[{match_id}] reference path raised: {e}",
                    file=sys.stderr,
                )
                return 1

            for inn_detail in innings_details:
                live_venue.update_venue_stats_detailed(
                    ref_venue, inn_detail)
            if chase_won is not None:
                live_venue.update_venue_match_result(ref_venue, chase_won)

            if not in_cache:
                n_skipped_before_cache += 1
                continue

            if skip_same_day_secondary and i > 0:
                # Re-rehydrate temp_* to first-of-date state so advancing
                # doesn't matter (we're skipping these anyway). Simplest:
                # just skip the candidate call.
                n_skipped_same_day += 1
                continue

            if inject_fault and not injected_once:
                _inject_fault(temp_stats, temp_elo, temp_venue, inject_fault)
                injected_once = True

            try:
                cand_rows, _citotals, cand_venue, cand_innings_details, cand_cw = (
                    parse_match_data_v2(
                        json_text,
                        temp_stats,
                        temp_venue,
                        player_metadata,
                        elo_tracker=temp_elo,
                        match_k_factor=k_factor,
                    )
                )
            except Exception as e:
                mismatches.append(
                    (match_id, date_str, f"candidate path raised: {e}")
                )
                if len(mismatches) >= MAX_MISMATCHES:
                    break
                continue

            # Advance temp_venue for the NEXT same-day match, mirroring
            # what the monolith does on live_venue (parsing_v2.py:1321-1325).
            for inn_detail in cand_innings_details:
                temp_venue.update_venue_stats_detailed(
                    cand_venue, inn_detail)
            if cand_cw is not None:
                temp_venue.update_venue_match_result(cand_venue, cand_cw)

            cand_df = pd.DataFrame(cand_rows)
            ref_df = pd.DataFrame(ref_rows)

            is_secondary = i > 0
            if is_secondary:
                # Schema-v2 known limitation: recent-form deque eviction
                # is not reproducible from the summed seed. Exclude those
                # 4 columns; everything else must still match bit-exactly.
                cand_cmp = cand_df.drop(columns=list(RECENT_FORM_COLUMNS))
                ref_cmp = ref_df.drop(columns=list(RECENT_FORM_COLUMNS))
            else:
                cand_cmp, ref_cmp = cand_df, ref_df

            try:
                pd.testing.assert_frame_equal(
                    cand_cmp, ref_cmp,
                    check_exact=True, check_dtype=True,
                )
            except AssertionError:
                diff = _first_diff_report(cand_cmp, ref_cmp)
                mismatches.append((match_id, date_str, diff))
                if len(mismatches) >= MAX_MISMATCHES:
                    break

            if is_secondary:
                n_checked_partial += 1
            else:
                n_checked_full += 1
            n_checked += 1
            if n_checked % PROGRESS_EVERY == 0:
                dt = time.time() - t_start
                rate = n_checked / dt if dt > 0 else 0
                print(
                    f"  [{n_checked}] checked in {dt:.0f}s "
                    f"({rate:.1f} match/s); mismatches so far: "
                    f"{len(mismatches)}",
                    flush=True,
                )

        if len(mismatches) >= MAX_MISMATCHES:
            break

    dt = time.time() - t_start

    skip_summary = f"{n_skipped_before_cache} pre-cache"
    if skip_same_day_secondary:
        skip_summary += f", {n_skipped_same_day} same-day-secondary"

    coverage = (
        f"{n_checked_full} first-of-date (63/63 cols), "
        f"{n_checked_partial} same-day-secondary (59/63 cols; "
        f"recent-form excluded per schema-v2 limitation)"
    )

    if mismatches:
        print(
            f"\nFAIL: {len(mismatches)} mismatch(es) in {n_checked} "
            f"checked matches ({dt:.0f}s total; skipped: {skip_summary})"
        )
        for mid, date_str, detail in mismatches:
            print(f"  {mid} @ {date_str}: {detail}")
        return 1

    print(
        f"\nPASS: {n_checked} matches parity-clean in {dt:.0f}s "
        f"(skipped: {skip_summary}; cache first date = {cache_min_date})"
    )
    print(f"  coverage: {coverage}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source-dir", type=Path,
        default=ROOT / "data" / "t20s_json",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Only check the first N matches (useful for dev iteration).",
    )
    p.add_argument(
        "--inject-fault", type=str, default=None,
        choices=["elo", "batting", "venue"],
        help="Corrupt one field on the first check to smoke-test the "
             "diff reporter.",
    )
    p.add_argument(
        "--skip-same-day-secondary", action="store_true",
        help="Skip parity check for matches that share their date with "
             "an earlier match in the same batch (diagnostic aid — "
             "isolates a full-corpus regression from within-day batching).",
    )
    args = p.parse_args()
    return run_harness(
        args.source_dir, args.limit, args.inject_fault,
        skip_same_day_secondary=args.skip_same_day_secondary,
    )


if __name__ == "__main__":
    sys.exit(main())
