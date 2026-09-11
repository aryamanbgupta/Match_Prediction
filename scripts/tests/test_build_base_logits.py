"""Stage 2 residual base logits — acceptance checks D4.1-D4.4.

Synthetic frames only: no production booster is loaded, no repository parquet
is read, nothing under `models/` is touched. What is pinned:

  * the booster -> T1 class-order permutation is derived from both maps and is
    a permutation of all six classes (D4.1);
  * flooring at 1e-4 and renormalisation, including a row that the floor
    actually binds on (D4.1);
  * the 5 contiguous `match_date` blocks partition the train split, respect
    date granularity and come out roughly equal (D4.2);
  * **no leak**: with the refit mocked so it records the rows it saw, no row is
    scored by a booster that was fit on it, and every row is scored exactly
    once (D4.2);
  * per-refit wall seconds and per-refit md5 land in the sidecar (D4.4);
  * the `npz` / sidecar schema.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from sequence_track import build_base_logits as bbl  # noqa: E402


# --------------------------------------------------------------------------
# D4.1 class order
# --------------------------------------------------------------------------

def test_class_permutation_is_a_permutation_and_matches_t1():
    perm = bbl.class_permutation()
    assert sorted(perm) == [0, 1, 2, 3, 4, 5]
    # The booster's remap and T1's CLASS_MAPPING agree outcome by outcome.
    from embeddings_e1 import CLASS_MAPPING

    t1 = dict(CLASS_MAPPING)
    t1[7] = t1.pop(-1)
    for outcome, booster_index in bbl.PRODUCTION_CLASS_MAPPING.items():
        assert perm[booster_index] == t1[outcome]
    assert bbl.CLASS_NAMES == ("dot", "one", "two", "four", "six", "wicket")


def test_to_t1_order_moves_columns_by_the_permutation():
    perm = bbl.class_permutation()
    proba = np.arange(12, dtype=np.float64).reshape(2, 6)
    out = bbl.to_t1_order(proba, perm)
    for booster_index, t1_index in enumerate(perm):
        assert np.array_equal(out[:, t1_index], proba[:, booster_index])

    # A non-identity permutation is honoured too.
    shuffled = [5, 4, 3, 2, 1, 0]
    out = bbl.to_t1_order(proba, shuffled)
    assert np.array_equal(out[:, 0], proba[:, 5])
    assert np.array_equal(out[:, 5], proba[:, 0])


# --------------------------------------------------------------------------
# D4.1 floor / renormalise
# --------------------------------------------------------------------------

def test_floor_log_floors_renormalises_and_logs():
    proba = np.array([
        [0.5, 0.2, 0.1, 0.1, 0.1, 0.0],      # the 0.0 is floored
        [1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1.0],  # five floors bind
        [0.2, 0.2, 0.2, 0.2, 0.1, 0.1],      # nothing binds
    ])
    logp = bbl.floor_log(proba)
    p = np.exp(logp.astype(np.float64))
    assert logp.dtype == np.float32
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)
    assert p.min() > 0.0
    # Floored entries land at 1e-4 / row_sum, i.e. just under the raw floor.
    assert p[0, 5] == pytest.approx(1e-4 / (1.0 + 1e-4), rel=1e-5)
    assert p[1, 0] == pytest.approx(1e-4 / (1.0 + 5e-4), rel=1e-5)
    # An untouched row is unchanged.
    assert np.allclose(p[2], proba[2], atol=1e-7)


def test_log_loss_from_logp():
    logp = np.log(np.array([[0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                            [0.25, 0.25, 0.25, 0.25, 0.0, 0.0]]) + 1e-12)
    ll = bbl.log_loss_from_logp(logp.astype(np.float32),
                                np.array([0, 2], dtype=np.int64))
    assert ll == pytest.approx((-np.log(0.5) - np.log(0.25)) / 2, rel=1e-5)


# --------------------------------------------------------------------------
# D4.2 date blocks
# --------------------------------------------------------------------------

def _toy_dates(n_dates: int = 20, per_date: int = 7) -> pd.Series:
    dates = [f"2024-01-{d + 1:02d}" for d in range(n_dates)]
    return pd.Series([d for d in dates for _ in range(per_date)])


def test_date_blocks_partition_and_are_roughly_equal():
    dates = _toy_dates()
    blocks = bbl.date_blocks(dates, 5)
    assert [b["block"] for b in blocks] == [0, 1, 2, 3, 4]
    assert sum(b["n_rows"] for b in blocks) == len(dates)
    # contiguous and non-overlapping in date order
    for left, right in zip(blocks, blocks[1:]):
        assert left["date_max"] < right["date_min"]
    assert blocks[0]["date_min"] == dates.min()
    assert blocks[-1]["date_max"] == dates.max()
    target = len(dates) / 5
    assert all(abs(b["n_rows"] - target) <= per for b, per in
               zip(blocks, [7] * 5))


def test_date_blocks_never_split_a_date():
    # One enormous date plus many tiny ones: the big date stays in one block.
    dates = pd.Series(["2024-01-01"] * 500
                      + [f"2024-02-{d + 1:02d}" for d in range(10)])
    blocks = bbl.date_blocks(dates, 5)
    holders = [b for b in blocks
               if b["date_min"] <= "2024-01-01" <= b["date_max"]]
    assert len(holders) == 1
    assert sum(b["n_rows"] for b in blocks) == len(dates)
    assert all(b["n_dates"] >= 1 for b in blocks)


def test_date_blocks_refuse_when_too_few_dates():
    with pytest.raises(RuntimeError, match="cannot make"):
        bbl.date_blocks(pd.Series(["2024-01-01"] * 10 + ["2024-01-02"] * 3), 5)


def test_block_assignment_covers_every_row_exactly_once():
    dates = _toy_dates()
    blocks = bbl.date_blocks(dates, 5)
    assign = bbl.block_assignment(dates, blocks)
    assert len(assign) == len(dates)
    assert set(np.unique(assign)) == {0, 1, 2, 3, 4}
    for block in blocks:
        assert int((assign == block["block"]).sum()) == block["n_rows"]


# --------------------------------------------------------------------------
# D4.2 no leak (mocked refit) + D4.4 wall seconds
# --------------------------------------------------------------------------

class _RecordingModel:
    """Stands in for XGBClassifier: records the rows each fold was fit on."""

    def __init__(self, seen: list, tag: int):
        self.seen = seen
        self.tag = tag
        self.fit_rows: set[int] = set()

    def fit(self, x, y):  # noqa: ANN001
        del y
        self.fit_rows = set(int(v) for v in x["row_id"].to_numpy())
        self.seen.append(self.fit_rows)
        return self

    def predict_proba(self, x):  # noqa: ANN001
        rows = x["row_id"].to_numpy()
        leaked = sorted(set(int(v) for v in rows) & self.fit_rows)
        assert not leaked, f"fold {self.tag} scored rows it was fit on: {leaked}"
        proba = np.full((len(rows), 6), 0.1)
        proba[:, 0] = 0.5
        return proba / proba.sum(axis=1, keepdims=True)


def _toy_train_frame(n_dates: int = 25, per_date: int = 4):
    dates = [f"2024-03-{d + 1:02d}" for d in range(n_dates)]
    rows = []
    for date in dates:
        for _ in range(per_date):
            rows.append(date)
    df = pd.DataFrame({
        bbl.DATE_COL: rows,
        "row_id": np.arange(len(rows), dtype=np.int64),
        "feat_a": np.linspace(0, 1, len(rows)),
    })
    labels = (df["row_id"].to_numpy() % 6).astype(np.int64)
    return df, labels, ["row_id", "feat_a"]


def test_oof_never_scores_a_row_its_booster_saw(monkeypatch):
    df, labels, feature_columns = _toy_train_frame()
    blocks = bbl.date_blocks(df[bbl.DATE_COL], 5)

    fits: list[set[int]] = []
    counter = {"n": 0}

    def fake_refit(hyperparameters):  # noqa: ANN001
        assert hyperparameters["n_estimators"] == 25
        counter["n"] += 1
        return _RecordingModel(fits, counter["n"])

    monkeypatch.setattr(bbl, "_make_refit", fake_refit)
    monkeypatch.setattr(bbl, "booster_md5", lambda m: f"md5-{m.tag:02d}")

    hook_calls = []
    proba, records = bbl.oof_train_logits(
        df, labels, feature_columns, blocks,
        {"n_estimators": 25},
        fit_hook=lambda b, f, s: hook_calls.append((b, set(f.tolist()),
                                                    set(s.tolist()))),
        verbose=False)

    assert len(records) == 5 and len(fits) == 5
    assert np.allclose(proba.sum(axis=1), 1.0)

    # Every row scored exactly once, by a booster that did not see it.
    all_scored: set[int] = set()
    for (block, fit_rows, score_rows), record in zip(hook_calls, records):
        assert not (fit_rows & score_rows)
        assert fit_rows == set(range(len(df))) - score_rows
        assert len(score_rows) == record["n_scored_rows"] == record["n_rows"]
        assert len(fit_rows) == record["n_fit_rows"]
        assert not (all_scored & score_rows)
        all_scored |= score_rows
        assert record["block"] == block
    assert all_scored == set(range(len(df)))

    # D4.4 wall seconds and per-refit md5 recorded.
    for record in records:
        assert record["fit_wall_seconds"] >= 0.0
        assert record["booster_md5"].startswith("md5-")


# --------------------------------------------------------------------------
# output schema
# --------------------------------------------------------------------------

def test_write_split_schema(tmp_path):
    logp = bbl.floor_log(np.full((11, 6), 1.0 / 6))
    sidecar = {"split": "validation", "n_rows": 11, "parquet_md5": "a" * 32,
               "booster_md5": "b" * 32, "class_order": list(bbl.CLASS_NAMES),
               "floor": bbl.FLOOR}
    npz_path, json_path = bbl.write_split(tmp_path, "validation", logp,
                                          sidecar)
    with np.load(npz_path, allow_pickle=False) as data:
        assert set(data.files) == {"logp", "n_rows", "parquet_md5"}
        assert data["logp"].shape == (11, 6)
        assert data["logp"].dtype == np.float32
        assert int(data["n_rows"]) == 11
        assert str(data["parquet_md5"]) == "a" * 32
    payload = json.loads(json_path.read_text())
    for key in ("split", "n_rows", "parquet_md5", "booster_md5",
                "class_order", "floor", "npz_md5"):
        assert key in payload
    assert payload["npz_md5"] == bbl.md5_file(npz_path)


def test_columns_to_read_pulls_encoder_sources_not_encoded_columns():
    feature_columns = ["batter_encoded", "bowler_encoded", "venue_encoded",
                       "matchup_type_encoded", "score", "balls_remaining"]
    cols = bbl.columns_to_read(feature_columns, with_date=True)
    assert "batter_encoded" not in cols
    for source in ("batter_id", "bowler_id", "venue", "matchup_type"):
        assert source in cols
    assert "ball_outcome" in cols and bbl.DATE_COL in cols
    assert len(cols) == len(set(cols))
    assert bbl.DATE_COL not in bbl.columns_to_read(feature_columns,
                                                   with_date=False)
