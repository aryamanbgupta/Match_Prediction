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

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

from build_stats_cache import build  # noqa: E402
from materialize_match_features import materialize  # noqa: E402
from player_metadata import PlayerMetadataProvider  # noqa: E402
from predict_fixture import (  # noqa: E402
    build_tracker_snapshot,
    compute_features,
    load_trackers,
)
from stats_provider import StatsProvider  # noqa: E402
from cricsheet_fixtures import (  # noqa: E402
    EVENT,
    FIXTURE_DATE,
    TEAMS,
    VENUE,
    write_corpus,
    write_metadata_csv,
)

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
    write_corpus(corpus)
    metadata_csv = tmp / "players.csv"
    write_metadata_csv(metadata_csv)

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
