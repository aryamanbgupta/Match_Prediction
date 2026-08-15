"""Train/serve parity harness (2026-08-14 review, Phase 4 item 1).

`predict_fixture.compute_features` is a second, comment-synchronized
implementation of `materialize_match_features._build_match_record`. Nothing
enforced their agreement — the SRV1 pre-toss bug and two default-divergence
minors were exactly this drift class. This harness builds a synthetic
corpus, materializes the training frame, re-serves the final match as a
fixture through the live serving path against the same state, and asserts
per-feature equality on every shared key.

Slow-ish (builds a mini SQLite cache + materializes a corpus) but the
single most protective test in the repo: any feature whose two
implementations diverge fails here by name.
"""

import csv
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build_stats_cache import build  # noqa: E402
from materialize_match_features import materialize  # noqa: E402
from player_metadata import PlayerMetadataProvider  # noqa: E402
from predict_fixture import (  # noqa: E402
    build_tracker_snapshot,
    compute_features,
    load_trackers,
)
from stats_provider import StatsProvider  # noqa: E402

TEAMS = {
    "Alpha CC": [(f"a{i:02d}", f"Alpha P{i}") for i in range(1, 12)],
    "Beta CC": [(f"b{i:02d}", f"Beta P{i}") for i in range(1, 12)],
}
VENUE = "Parity Park"
EVENT = "Indian Premier League"


def _delivery(batter, bowler, runs, wicket=False):
    d = {
        "batter": batter,
        "bowler": bowler,
        "non_striker": batter,
        "runs": {"batter": runs, "extras": 0, "total": runs},
    }
    if wicket:
        d["wickets"] = [{"kind": "bowled", "player_out": batter}]
    return d


def _innings(team, bowling_team, seed):
    """Five overs with seed-varied runs so stats/ELO differ per match."""
    batter_names = [name for _, name in TEAMS[team]]
    bowler_names = [name for _, name in TEAMS[bowling_team]][7:11]
    overs = []
    runs_cycle = [0, 1, (seed % 3) + 1, 4, 6, 1]
    for over_no in range(5):
        batter = batter_names[over_no % 4]
        bowler = bowler_names[over_no % 4]
        deliveries = [
            _delivery(batter, bowler, runs_cycle[(i + seed) % 6],
                      wicket=(over_no == 3 and i == 5))
            for i in range(6)
        ]
        overs.append({"over": over_no, "deliveries": deliveries})
    return {"team": team, "overs": overs}


def _match_json(date, home, away, seed):
    registry = {name: pid for team in (home, away)
                for pid, name in TEAMS[team]}
    first, second = (home, away) if seed % 2 == 0 else (away, home)
    return {
        "info": {
            "match_type": "T20",
            "gender": "male",
            "dates": [date],
            "teams": [home, away],
            "players": {t: [name for _, name in TEAMS[t]]
                        for t in (home, away)},
            "registry": {"people": registry},
            "venue": VENUE,
            "event": {"name": EVENT},
            "team_type": "club",
            "outcome": {"winner": home},
            "toss": {"winner": first, "decision": "bat"},
        },
        "innings": [
            _innings(first, second, seed),
            _innings(second, first, seed + 1),
        ],
    }


FIXTURE_DATE = "2025-08-01"


def _write_corpus(corpus: Path):
    corpus.mkdir()
    dates = ["2024-05-01", "2024-06-01", "2024-07-01", "2024-08-01",
             "2024-09-01", "2024-10-01", FIXTURE_DATE]
    for i, date in enumerate(dates):
        (corpus / f"{9000 + i}.json").write_text(
            json.dumps(_match_json(date, "Alpha CC", "Beta CC", seed=i)))
    return dates


def _write_metadata_csv(path: Path):
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["cricsheet_id", "name", "cricinfo_id",
                         "unique_name", "full_name", "country", "dob",
                         "batting_style", "bowling_style"])
        for team, players in TEAMS.items():
            for pid, name in players:
                writer.writerow([pid, name, "", name, name, "Testland",
                                 "1995-01-01", "Right-hand bat",
                                 "Right-arm medium"])


# Non-feature columns of the training frame / serving record.
_META_KEYS = {
    "match_id", "cricsheet_id", "display_match_id", "match_date",
    "team1", "team2", "venue", "team1_wins", "split",
    "match_identity_version", "elo_update_version", "event_name",
}


@pytest.fixture(scope="module")
def parity_setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("parity")
    corpus = tmp / "corpus"
    _write_corpus(corpus)
    metadata_csv = tmp / "players.csv"
    _write_metadata_csv(metadata_csv)

    cache = tmp / "player_stats_cache_v3.sqlite"
    build([corpus], cache, gender="male", metadata_csv=metadata_csv)

    out_dir = tmp / "frame"
    out_dir.mkdir()
    materialize(
        corpus,
        sqlite_dir=tmp,
        out_dir=out_dir,
        version="v3",
        splits={},            # DEFAULT_SPLITS: 2025-08-01 lands in "test"
        gender="male",
        metadata_csv=metadata_csv,
    )
    test_frame = pd.read_parquet(out_dir / "test.parquet")
    assert len(test_frame) == 1, \
        f"expected exactly the fixture-date match in test: {len(test_frame)}"
    train_row = test_frame.iloc[0].to_dict()

    provider = StatsProvider(str(tmp), version="v3")
    metadata = PlayerMetadataProvider(str(metadata_csv))
    snapshot_path = tmp / "tracker_snapshot.pkl"
    build_tracker_snapshot(corpus, snapshot_path)
    form, h2h, home = load_trackers(snapshot_path, corpus)

    final = json.loads((corpus / "9006.json").read_text())
    info = final["info"]
    fixture = {
        "team1": info["teams"][0],
        "team2": info["teams"][1],
        "team1_lineup": [pid for pid, _ in TEAMS[info["teams"][0]]],
        "team2_lineup": [pid for pid, _ in TEAMS[info["teams"][1]]],
        "venue": info["venue"],
        "date": FIXTURE_DATE,
        "competition_tier": EVENT,
        "team_type": "club",
        "toss_winner": info["toss"]["winner"],
        "toss_decision": info["toss"]["decision"],
    }
    record = compute_features(fixture, provider, metadata, form, h2h, home)
    return train_row, record


def test_every_shared_feature_matches(parity_setup):
    train_row, record = parity_setup
    serving = {k: v for k, v in record.items() if not k.startswith("_")}
    shared = sorted(
        (set(train_row) & set(serving)) - _META_KEYS
    )
    assert len(shared) >= 40, \
        f"parity surface collapsed — only {len(shared)} shared keys: {shared}"

    mismatches = []
    for key in shared:
        a, b = train_row[key], serving[key]
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if not (a == b or abs(float(a) - float(b)) <= 1e-9):
                mismatches.append(f"{key}: train={a!r} serve={b!r}")
        elif str(a) != str(b):
            mismatches.append(f"{key}: train={a!r} serve={b!r}")
    assert not mismatches, (
        "train/serve feature drift detected:\n  " + "\n  ".join(mismatches)
    )


def test_parity_covers_the_toss_features(parity_setup):
    """The SRV1 bug class lived exactly here — pin that the toss trio is
    part of the compared surface."""
    train_row, record = parity_setup
    for key in ("team1_batting_first", "toss_winner_is_team1",
                "toss_decision_bat"):
        assert key in train_row and key in record, \
            f"{key} missing from the parity surface"
