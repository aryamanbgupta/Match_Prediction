"""Contracts for the A50 50-feature ball model (sequence track, stage 0 D4).

Two levels:

* the column list itself — 50 names, no duplicates, exactly the concatenation
  the T1 transformer consumes (no artifacts needed);
* the serving path — `sim_v1_2.XGBoostModelV2` loads the trained A50
  directory as-is and returns a six-class probability vector on a synthetic
  match state, with every one of the 50 columns actually produced by the
  wrapper's feature builder (marked `needs_artifacts`).
"""
from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
A50_DIR = REPO / "models" / "embeddings" / "seq_stage1" / "a50"
MODEL_PATH = A50_DIR / "xgboost_model_a50.pkl"
COLUMNS_PATH = A50_DIR / "feature_columns_a50.txt"
METADATA_CSV = REPO / "data" / "all_players_enriched.csv"
STATS_CACHE = REPO / "models" / "player_stats_cache_i7.sqlite"

from sequence_track.train_a50 import (a50_feature_columns,  # noqa: E402
                                      feature_columns_sha256, md5_file, rel,
                                      verify_frame)

OUTCOME_CLASSES = ("dot", "one", "two", "four", "six", "wicket")

_artifacts_present = (
    MODEL_PATH.exists()
    and COLUMNS_PATH.exists()
    and STATS_CACHE.exists()
    and METADATA_CSV.exists()
)
requires_a50 = pytest.mark.skipif(
    not _artifacts_present,
    reason="A50 artifacts / i7 stats cache not present on this checkout",
)


def test_a50_column_list_is_fifty_and_unique():
    from embeddings_e1 import (CTX_COLS, EB_BAT_COLS, EB_BOWL_COLS,
                               VENUE_COLS)
    from transformer_t1 import STATE_COLS

    columns = a50_feature_columns()
    assert len(columns) == 50
    assert len(set(columns)) == 50
    assert columns == (
        list(EB_BAT_COLS)
        + list(EB_BOWL_COLS)
        + list(VENUE_COLS)
        + list(CTX_COLS)
        + list(STATE_COLS)
    )
    assert len(EB_BAT_COLS) == 18
    assert len(EB_BOWL_COLS) == 18
    assert len(VENUE_COLS) == 6
    assert len(CTX_COLS) == 4
    assert len(STATE_COLS) == 4


@pytest.mark.needs_artifacts
@requires_a50
def test_a50_feature_columns_file_matches_the_import():
    written = COLUMNS_PATH.read_text().splitlines()
    assert written == a50_feature_columns()


SYNTHETIC_ROWS = {"train": 4, "validation": 3, "test": 2}


def _synthetic_frame(tmp_path: Path) -> tuple[Path, Path, dict]:
    """A tiny stand-in for the A50 frame: same shape, four-row splits.

    Returns (out_dir, source_dir, expected_rows) with a `feature_hash.json`
    whose recorded hashes are true of the files on disk, so a test can then
    break exactly one thing.
    """
    import json

    import pandas as pd

    columns = a50_feature_columns()
    source_dir = tmp_path / "src"
    out_dir = tmp_path / "a50"
    source_dir.mkdir()
    (out_dir / "data").mkdir(parents=True)

    written = columns + ["ball_outcome", "innings_id", "ball_idx"]
    record = {
        "arm": "a50",
        "data_version": "i7",
        "source_dir": rel(source_dir),
        "feature_columns": columns,
        "feature_columns_sha256": feature_columns_sha256(columns),
        "written_columns": written,
        "bookkeeping_columns": ["innings_id", "ball_idx"],
        "splits": {},
        "row_counts": dict(SYNTHETIC_ROWS),
    }
    for split, rows in SYNTHETIC_ROWS.items():
        frame = pd.DataFrame(
            {name: np.arange(rows, dtype=float) for name in columns}
        )
        frame["ball_outcome"] = ([0, 1, 4, -1] * rows)[:rows]
        frame["innings_id"] = [f"{split}-{i}" for i in range(rows)]
        frame["ball_idx"] = np.arange(rows, dtype=np.int64)
        frame = frame[written]
        source_path = source_dir / f"cricket_data_i7_{split}.parquet"
        a50_path = out_dir / "data" / f"a50_{split}.parquet"
        frame.to_parquet(source_path, index=False)
        frame.to_parquet(a50_path, index=False)
        record["splits"][split] = {
            "source_parquet": rel(source_path),
            "source_parquet_md5": md5_file(source_path),
            "a50_parquet": rel(a50_path),
            "a50_parquet_md5": md5_file(a50_path),
            "rows": rows,
            "expected_rows": rows,
        }
    (out_dir / "data" / "feature_hash.json").write_text(
        json.dumps(record, indent=2)
    )
    return out_dir, source_dir, dict(SYNTHETIC_ROWS)


def test_verify_frame_accepts_an_untouched_reused_frame(tmp_path):
    out_dir, source_dir, expected = _synthetic_frame(tmp_path)
    record, verification, paths = verify_frame(
        out_dir, a50_feature_columns(), source_dir, expected_rows=expected
    )
    assert record["feature_columns"] == a50_feature_columns()
    for split, rows in expected.items():
        # The verifier hands back the exact files training must read.
        assert paths[split] == out_dir / "data" / f"a50_{split}.parquet"
        entry = verification["splits"][split]
        assert entry["rows"] == rows
        assert entry["a50_parquet_md5"] == (
            record["splits"][split]["a50_parquet_md5"]
        )
        assert entry["source_parquet_md5"] == (
            record["splits"][split]["source_parquet_md5"]
        )


def test_verify_frame_refuses_a_tampered_a50_parquet(tmp_path):
    import pandas as pd

    out_dir, source_dir, expected = _synthetic_frame(tmp_path)
    tampered = out_dir / "data" / "a50_validation.parquet"
    frame = pd.read_parquet(tampered)
    frame.loc[0, "score"] = frame.loc[0, "score"] + 1.0
    frame.to_parquet(tampered, index=False)

    with pytest.raises(RuntimeError) as excinfo:
        verify_frame(
            out_dir, a50_feature_columns(), source_dir, expected_rows=expected
        )
    message = str(excinfo.value)
    assert "refusing to train" in message
    assert "a50_validation.parquet" in message
    assert "md5" in message


def test_verify_frame_refuses_a_changed_source_hash(tmp_path):
    import json

    out_dir, source_dir, expected = _synthetic_frame(tmp_path)
    record_path = out_dir / "data" / "feature_hash.json"
    record = json.loads(record_path.read_text())
    record["splits"]["train"]["source_parquet_md5"] = "0" * 32
    record_path.write_text(json.dumps(record, indent=2))

    with pytest.raises(RuntimeError) as excinfo:
        verify_frame(
            out_dir, a50_feature_columns(), source_dir, expected_rows=expected
        )
    message = str(excinfo.value)
    assert "cricket_data_i7_train.parquet" in message
    assert "0" * 32 in message


def test_verify_frame_refuses_a_missing_source_parquet(tmp_path):
    out_dir, source_dir, expected = _synthetic_frame(tmp_path)
    (source_dir / "cricket_data_i7_test.parquet").unlink()

    with pytest.raises(RuntimeError, match="cricket_data_i7_test.parquet"):
        verify_frame(
            out_dir, a50_feature_columns(), source_dir, expected_rows=expected
        )


def test_verify_frame_refuses_a_different_column_list(tmp_path):
    import json

    out_dir, source_dir, expected = _synthetic_frame(tmp_path)
    record_path = out_dir / "data" / "feature_hash.json"
    record = json.loads(record_path.read_text())
    record["feature_columns"] = record["feature_columns"][:-1]
    record_path.write_text(json.dumps(record, indent=2))

    with pytest.raises(RuntimeError, match="feature_columns"):
        verify_frame(
            out_dir, a50_feature_columns(), source_dir, expected_rows=expected
        )


def _rewrite_record_paths(frame_dir: Path, source_dir: Path) -> None:
    """Point a copied manifest at the files under its new location."""
    import json

    record_path = frame_dir / "data" / "feature_hash.json"
    record = json.loads(record_path.read_text())
    record["source_dir"] = rel(source_dir)
    for split in record["splits"]:
        record["splits"][split]["a50_parquet"] = rel(
            frame_dir / "data" / f"a50_{split}.parquet"
        )
        record["splits"][split]["source_parquet"] = rel(
            source_dir / f"cricket_data_i7_{split}.parquet"
        )
    record_path.write_text(json.dumps(record, indent=2))


def test_verify_frame_refuses_a_copied_frame_carrying_its_old_manifest(
    tmp_path,
):
    import shutil

    out_dir, source_dir, expected = _synthetic_frame(tmp_path)
    copied = tmp_path / "elsewhere" / "a50"
    copied.parent.mkdir()
    shutil.copytree(out_dir, copied)

    # The copy's files are byte-identical, so md5s alone would pass; the
    # manifest still names the ORIGINAL location, which is the defect.
    with pytest.raises(RuntimeError) as excinfo:
        verify_frame(
            copied, a50_feature_columns(), source_dir, expected_rows=expected
        )
    message = str(excinfo.value)
    assert "does not resolve to the file this run reads" in message
    assert (copied / "data" / "a50_train.parquet").resolve().as_posix() \
        in message
    assert (out_dir / "data" / "a50_train.parquet").resolve().as_posix() \
        in message


def test_verify_frame_accepts_a_copy_whose_manifest_was_repointed(tmp_path):
    import shutil

    out_dir, source_dir, expected = _synthetic_frame(tmp_path)
    copied = tmp_path / "elsewhere" / "a50"
    copied.parent.mkdir()
    shutil.copytree(out_dir, copied)
    _rewrite_record_paths(copied, source_dir)

    _record, _verification, paths = verify_frame(
        copied, a50_feature_columns(), source_dir, expected_rows=expected
    )
    assert paths["train"] == copied / "data" / "a50_train.parquet"


def test_verify_frame_refuses_a_tampered_repointed_copy(tmp_path):
    import shutil

    import pandas as pd

    out_dir, source_dir, expected = _synthetic_frame(tmp_path)
    copied = tmp_path / "elsewhere" / "a50"
    copied.parent.mkdir()
    shutil.copytree(out_dir, copied)
    _rewrite_record_paths(copied, source_dir)

    tampered = copied / "data" / "a50_train.parquet"
    frame = pd.read_parquet(tampered)
    frame.loc[0, "batter_p0"] = frame.loc[0, "batter_p0"] + 1.0
    frame.to_parquet(tampered, index=False)

    with pytest.raises(RuntimeError) as excinfo:
        verify_frame(
            copied, a50_feature_columns(), source_dir, expected_rows=expected
        )
    message = str(excinfo.value)
    assert tampered.resolve().as_posix() in message
    assert "md5" in message
    # The untouched original must not be what was hashed.
    assert md5_file(out_dir / "data" / "a50_train.parquet") != md5_file(
        tampered
    )


def test_verify_frame_refuses_a_source_dir_that_is_not_the_recorded_one(
    tmp_path,
):
    import shutil

    out_dir, source_dir, expected = _synthetic_frame(tmp_path)
    other_source = tmp_path / "other_src"
    shutil.copytree(source_dir, other_source)

    with pytest.raises(RuntimeError, match="does not resolve to the"):
        verify_frame(
            out_dir, a50_feature_columns(), other_source,
            expected_rows=expected,
        )


def _lineup(name: str, player_ids: list[str]):
    from sim_v1_2 import Player, TeamLineup

    return TeamLineup(
        name,
        [
            Player(str(pid), f"{name} player {i}", name)
            for i, pid in enumerate(player_ids)
        ],
    )


@pytest.mark.needs_artifacts
@requires_a50
def test_a50_serves_a_six_class_vector_through_xgboost_model_v2():
    from player_metadata import PlayerMetadataProvider
    from sim_v1_2 import MatchState, XGBoostModelV2
    from stats_provider import StatsProvider

    stats_provider = StatsProvider("models", version="i7")
    player_metadata = PlayerMetadataProvider(str(METADATA_CSV))

    model = XGBoostModelV2(
        model_path=str(MODEL_PATH),
        batter_encoder_path=str(A50_DIR / "batter_encoder_a50.pkl"),
        bowler_encoder_path=str(A50_DIR / "bowler_encoder_a50.pkl"),
        feature_columns_path=str(COLUMNS_PATH),
        stats_provider=stats_provider,
        player_metadata=player_metadata,
    )

    columns = a50_feature_columns()
    assert model.feature_columns == columns
    assert model.model.n_features_in_ == 50

    batters = [str(c) for c in model.batter_encoder.classes_[:11]]
    bowlers = [str(c) for c in model.bowler_encoder.classes_[:11]]
    match_date = datetime.strptime(
        stats_provider.dates[len(stats_provider.dates) // 2], "%Y-%m-%d"
    )
    state = MatchState(
        team1_lineup=_lineup("A", batters),
        team2_lineup=_lineup("B", bowlers),
        batting_first="A",
        venue="Wankhede Stadium",
        match_date=match_date,
    )
    state.innings = 1
    state.balls = 42
    state.runs[state.current_team_idx] = 55.0
    state.wickets[state.current_team_idx] = 2

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        features = model.extract_features(state)
    zero_filled = [
        str(w.message) for w in caught if "never produced" in str(w.message)
    ]
    assert not zero_filled, zero_filled

    assert features.shape == (50,)
    assert np.all(np.isfinite(features))

    probs = model.model.predict_proba(features.reshape(1, -1))[0]
    assert probs.shape == (6,)
    assert np.all(probs >= 0.0)
    assert probs.sum() == pytest.approx(1.0, abs=1e-6)

    outcome_probs = model.predict_next_ball(features)
    assert set(OUTCOME_CLASSES).issubset(outcome_probs)
    assert all(outcome_probs[name] > 0.0 for name in OUTCOME_CLASSES)
    assert sum(outcome_probs.values()) == pytest.approx(1.0, abs=1e-6)
    # The six model classes carry all but the grafted extras mass.
    assert sum(outcome_probs[name] for name in OUTCOME_CLASSES) > 0.9
