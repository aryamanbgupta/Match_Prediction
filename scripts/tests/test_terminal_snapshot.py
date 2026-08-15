"""Regression test for review finding PIPE2 (2026-08-14).

Snapshots are emitted before each date's first match, so the newest
snapshot used to EXCLUDE the final corpus date's matches: an as-of past
corpus end floored to it and served career state missing the freshest day,
while the match-log recent-form getters (strict-before bisect) included
that day — one feature row, two information sets, exactly on the live
serving path. The build now emits a terminal snapshot dated the day after
the last corpus date.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build_stats_cache import build  # noqa: E402
from stats_sqlite_backend import _SQLiteBackend  # noqa: E402


def _delivery(batter, bowler, runs):
    return {
        "batter": batter,
        "bowler": bowler,
        "non_striker": "NS X",
        "runs": {"batter": runs, "extras": 0, "total": runs},
    }


def _match_json(date, star_runs):
    """Minimal male T20 with one star batter scoring `star_runs`."""
    deliveries = [_delivery("Star Batter", "Some Bowler", 1)] * star_runs
    return {
        "info": {
            "match_type": "T20",
            "gender": "male",
            "dates": [date],
            "teams": ["Alpha", "Beta"],
            "players": {
                "Alpha": ["Star Batter", "NS X"],
                "Beta": ["Some Bowler"],
            },
            "registry": {"people": {
                "Star Batter": "p_star", "NS X": "p_ns",
                "Some Bowler": "p_bowl",
            }},
            "venue": "Test Oval",
            "outcome": {"winner": "Alpha"},
            "toss": {"winner": "Alpha", "decision": "bat"},
        },
        "innings": [
            {"team": "Alpha",
             "overs": [{"over": 0, "deliveries": deliveries}]},
            {"team": "Beta",
             "overs": [{"over": 0, "deliveries": [
                 _delivery("Some Bowler", "Star Batter", 1)]}]},
        ],
    }


def _build_mini_cache(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "1001.json").write_text(
        json.dumps(_match_json("2026-05-01", 10)))
    (corpus / "1002.json").write_text(
        json.dumps(_match_json("2026-05-02", 25)))
    out = tmp_path / "cache.sqlite"
    build([corpus], out, gender="male",
          metadata_csv=ROOT.parent / "data" / "all_players_enriched.csv")
    return out


def test_terminal_snapshot_serves_the_final_corpus_day(tmp_path):
    out = _build_mini_cache(tmp_path)
    backend = _SQLiteBackend(out)

    # A beyond-corpus as-of must see BOTH days' runs (pre-fix: the floor
    # snapshot predated 2026-05-02's matches, so only day 1 appeared).
    # Star Batter is never dismissed, so avg == career runs.
    beyond = backend.get_batting_stats("p_star", "2026-06-01")
    assert beyond["avg"] == 35, \
        f"beyond-corpus state lost the final day: {beyond}"

    # Strictly-before semantics stay intact for in-corpus dates.
    on_final_day = backend.get_batting_stats("p_star", "2026-05-02")
    assert on_final_day["avg"] == 10

    # And the sentinel is stamped in _meta for the reader-side guard.
    meta = backend.get_meta()
    assert meta.get("terminal_snapshot_date") == "2026-05-03"


def test_terminal_snapshot_matches_match_log_information_set(tmp_path):
    """Career state and recent-form logs must describe the same set of
    matches for a beyond-corpus as-of."""
    out = _build_mini_cache(tmp_path)
    backend = _SQLiteBackend(out)
    log = backend.get_batting_match_log_recent(
        "p_star", "2026-06-01", limit=10)
    log_runs = sum(row["runs"] for row in log)
    career = backend.get_batting_stats("p_star", "2026-06-01")
    assert log_runs == career["avg"] == 35
