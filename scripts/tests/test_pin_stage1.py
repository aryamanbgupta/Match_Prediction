"""Tests for the stage-1 config pin script (D5 check 5.1, 5.3, 5.4; D6).

Five things have to hold for the registered config to mean anything:

* `--verify` passes on the committed file, so the pinned hashes are the
  hashes of the artifacts actually on disk;
* a tampered copy fails and names what moved, so the file cannot drift
  from the artifacts (or be hand-edited) unnoticed;
* the checkpoint rule really is "lowest validation LL, ties to the lowest
  seed", read from the i7 retrain summary, over exactly the five registered
  training seeds, at full precision — exercised on synthetic summaries where
  the answer is known (D6 check 6.2);
* a retrained checkpoint is pinned with its training contract, and the
  contract is checked against the frame and the LIVE stats cache (6.3);
* the seed formula the config documents is the one `run_arm.py` runs.

Tests that touch repository artifacts carry the `needs_artifacts` marker so
`-m "not needs_artifacts"` stays green. Tests that additionally need the
COMMITTED config to be current carry `needs_repin`: between the D6 code
change and the post-retrain `--write`, the committed YAML is deliberately
stale (acceptance D6 check 6.8), and there is nothing for them to assert
until the retrain exists and the config is re-pinned.
"""
from __future__ import annotations

import atexit
import json
import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from sequence_track import pin_stage1
from sequence_track.pin_stage1 import (
    CONFIG_PATH,
    PinError,
    build_config,
    choose_seed,
    decimal_places,
    diff_config,
    expected_feature_names,
    feature_names_sha256,
    fixture_seed,
    flatten,
    frame_split_facts,
    main,
    read_training_contract,
    retrain_summary_block,
    selection_table,
    sub_seed,
    training_frame_block,
    verify_config,
)
from sequence_track import run_arm
from sequence_track.run_arm import fixture_seed as run_arm_fixture_seed
from sequence_track.run_arm import sub_seed as run_arm_sub_seed


# The committed config is re-pinned by `--write` only after the D3 retrain
# exists (acceptance D6 checks 6.8 and 6.10). Until then `build_config()`
# fails closed on the missing summary and the committed YAML is knowingly
# stale, so every assertion that compares the two is skipped rather than
# asserted against a file that cannot be current yet.
RETRAIN_SUMMARY_ON_DISK = pin_stage1.REPO_ROOT / pin_stage1.RETRAIN_SUMMARY
needs_repin = pytest.mark.skipif(
    not RETRAIN_SUMMARY_ON_DISK.is_file(),
    reason=(f"{pin_stage1.RETRAIN_SUMMARY.as_posix()} does not exist yet: "
            "arms B and C are pinned from the D3 i7 retrain, and the "
            "committed config is re-pinned after it"))


# ---------------------------------------------------------------------------
# 5.1 — verify passes on the committed file, fails on a tampered copy
# ---------------------------------------------------------------------------

@pytest.mark.needs_artifacts
@needs_repin
def test_verify_passes_on_the_committed_config():
    assert CONFIG_PATH.exists(), (
        f"{CONFIG_PATH} is missing; run pin_stage1.py --write")
    assert verify_config(CONFIG_PATH) == []
    assert main(["--verify"]) == 0


@pytest.mark.needs_artifacts
@needs_repin
def test_tampered_hash_fails_and_names_the_key(tmp_path):
    tampered = tmp_path / "seq_stage1_sim_v1.yaml"
    shutil.copyfile(CONFIG_PATH, tampered)
    payload = yaml.safe_load(tampered.read_text())
    payload["arms"]["A"]["checkpoint_md5"] = "0" * 32
    tampered.write_text(yaml.safe_dump(payload, sort_keys=False))

    problems = verify_config(tampered)
    assert problems, "a changed checkpoint md5 must not verify"
    assert any("arms.A.checkpoint_md5" in problem for problem in problems)
    assert main(["--verify", "--config", str(tampered)]) == 1


@pytest.mark.needs_artifacts
@needs_repin
def test_tampered_prose_fails(tmp_path):
    """Prose lives in the script, so a hand-edited YAML line is a mismatch."""
    tampered = tmp_path / "seq_stage1_sim_v1.yaml"
    shutil.copyfile(CONFIG_PATH, tampered)
    payload = yaml.safe_load(tampered.read_text())
    payload["decision_rule"]["equivalence_margin_ll"] = 0.02
    tampered.write_text(yaml.safe_dump(payload, sort_keys=False))

    problems = verify_config(tampered)
    assert any("decision_rule.equivalence_margin_ll" in problem
               for problem in problems)


@pytest.mark.needs_artifacts
@needs_repin
def test_dropped_key_fails(tmp_path):
    tampered = tmp_path / "seq_stage1_sim_v1.yaml"
    shutil.copyfile(CONFIG_PATH, tampered)
    payload = yaml.safe_load(tampered.read_text())
    del payload["arms"]["C114"]
    tampered.write_text(yaml.safe_dump(payload, sort_keys=False))

    problems = verify_config(tampered)
    assert any(problem.startswith("MISSING") and "C114" in problem
               for problem in problems)


def test_verify_reports_a_missing_file(tmp_path):
    assert verify_config(tmp_path / "absent.yaml") == [
        f"MISSING  file: {tmp_path / 'absent.yaml'}"]


def test_not_verified_keys_are_ignored_by_the_diff():
    """A moved timestamp or a new HEAD must not fail --verify."""
    expected = {
        "provenance": {"pins_generated_at": "2026-09-10T00:00:00Z",
                       "git_head_short": "aaaaaaa",
                       "source_md5": {"x": "1"}},
    }
    actual = {
        "provenance": {"pins_generated_at": "2027-01-01T00:00:00Z",
                       "git_head_short": "bbbbbbb",
                       "source_md5": {"x": "1"}},
    }
    assert diff_config(expected, actual) == []

    actual["provenance"]["source_md5"]["x"] = "2"
    assert any("provenance.source_md5.x" in problem
               for problem in diff_config(expected, actual))


# ---------------------------------------------------------------------------
# 5.3 / D6 6.2 — checkpoint selection from the i7 retrain summary
#
# The rows the real retrain will write are full-precision float64 means that
# differ in the fifth and sixth decimal, so the synthetic tables below use
# values of the same shape. A four-decimal table would make the tie-break,
# not the validation log loss, decide the selection, and is refused.
# ---------------------------------------------------------------------------

MLP_LLS = {7: 1.4381341, 13: 1.4380398, 29: 1.4380532, 42: 1.4386004,
           101: 1.4377551}
FULL_LLS = {7: 1.4369723, 13: 1.4383531, 29: 1.4374518, 42: 1.4372450,
            101: 1.4366347}


# ---------------------------------------------------------------------------
# A synthetic i7 frame (Astra round 1, MUST-FIX 1)
#
# The pin no longer copies the training parquets' md5, row count and
# match_date range out of the checkpoint's own contract — it hashes and reads
# the files. So the synthetic checkpoints below need real parquets to be
# checked against. Two tiny ones, named in the frame's own scheme, stand in
# for the 830 MB pair; the test and golden splits are never written, because
# the pin never names them.
# ---------------------------------------------------------------------------

FRAME_HASH = {
    "hash": "c520a3ba08ae",
    "version": "i7",
    "n_features": 114,
    "delivery_semantics": "inclusive_total_runs_v1",
    "venue_alias_version": "venue_aliases_v1",
    "venue_alias_sha256": "8" * 64,
}
CACHE_ROLE = "stats_cache_i7"
CACHE_PATH = "models/player_stats_cache_i7.sqlite"
CACHE_MD5 = "f" * 32
SAME_DAY_ORDER = "date_then_match_id_lexicographic_v1"

TRAIN_DATES = ["2005-02-17", "2015-06-01", "2024-12-30"]
VALIDATION_DATES = ["2024-12-31", "2025-03-01", "2025-06-29"]


def _write_split_parquet(path, dates):
    import pandas as pd

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "match_date": list(dates),
        "innings_id": [f"1_{index}" for index in range(len(dates))],
    }).to_parquet(path, index=False)
    return Path(path)


def _build_frame(frame_dir, train_dates=None, validation_dates=None):
    """A frame dir with a `.feature_hash` and the two training splits."""
    frame_dir = Path(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    (frame_dir / pin_stage1.FEATURE_HASH_FILENAME).write_text(
        json.dumps(FRAME_HASH))
    _write_split_parquet(frame_dir / "cricket_data_i7_train.parquet",
                         train_dates or TRAIN_DATES)
    _write_split_parquet(frame_dir / "cricket_data_i7_validation.parquet",
                         validation_dates or VALIDATION_DATES)
    pin_stage1._FRAME_SPLIT_FACTS.pop((frame_dir.as_posix(), "i7"), None)
    return frame_split_facts(frame_dir, "i7")


_SHARED_FRAME: dict = {}


def _shared_frame():
    """One synthetic frame for the whole module, built on first use."""
    if not _SHARED_FRAME:
        root = Path(tempfile.mkdtemp(prefix="pin_stage1_frame_"))
        atexit.register(shutil.rmtree, root, ignore_errors=True)
        frame_dir = root / "xgb_data_i7"
        _SHARED_FRAME["facts"] = _build_frame(frame_dir)
        _SHARED_FRAME["dir"] = frame_dir.as_posix()
    return _SHARED_FRAME["dir"], _SHARED_FRAME["facts"]


def _frame_block_from(frame_dir, facts, feature_hash, cache_md5):
    return {
        "dir": frame_dir,
        "version": feature_hash["version"],
        "feature_hash": dict(feature_hash),
        "split_md5s": {split: row["md5"] for split, row in facts.items()},
        "split_rows": {split: row["n_rows"] for split, row in facts.items()},
        "stats_cache": {
            "role": CACHE_ROLE,
            "path": CACHE_PATH,
            "md5": cache_md5,
            "venue_alias_version": feature_hash["venue_alias_version"],
            "same_day_order_version": SAME_DAY_ORDER,
        },
    }


def _rows(lls, retrain_arm):
    return [
        {"seed": seed, "ll": ll,
         "checkpoint_dir": (
             f"models/embeddings/seq_stage1/retrain_i7/{retrain_arm}/"
             f"seed_{seed}")}
        for seed, ll in sorted(lls.items())
    ]


def _synthetic_summary(path, mlp_rows=None, full_rows=None,
                       with_checkpoints=True, frame_dir=None, facts=None,
                       feature_hash=None, cache_md5=CACHE_MD5, **overrides):
    """A retrain summary in the D3 layout, defaulting to a valid one.

    The per-seed checkpoints are written beside it by default, because the
    pin now checks every summary row against the `validation_ll` in that
    seed's own metrics.json.
    """
    mlp = dict(MLP_LLS if mlp_rows is None else dict(mlp_rows))
    full = dict(FULL_LLS if full_rows is None else dict(full_rows))
    shared_dir, shared_facts = _shared_frame()
    frame_dir = shared_dir if frame_dir is None else frame_dir
    facts = shared_facts if facts is None else facts
    feature_hash = dict(FRAME_HASH if feature_hash is None else feature_hash)
    if with_checkpoints:
        for retrain_arm, lls in (("mlp", mlp), ("full", full)):
            for seed, ll in lls.items():
                _write_checkpoint(Path(path).parent / retrain_arm
                                  / f"seed_{seed}", None, validation_ll=ll)
    payload = {
        "experiment": {
            "config": pin_stage1.RETRAIN_CONFIG,
            "config_sha256": pin_stage1.retrain_config_sha256()},
        "frame": _frame_block_from(frame_dir, facts, feature_hash,
                                   cache_md5),
        "splits": {
            "validation": {
                "n_rows": facts["validation"]["n_rows"],
                "n_matches": pin_stage1.validation_match_count(
                    frame_dir, feature_hash["version"]),
                "arms": {
                    "mlp": {"per_seed": _rows(mlp, "mlp")},
                    "full": {"per_seed": _rows(full, "full")},
                },
            },
            # Deliberately better on a DIFFERENT seed: the rule must not
            # look at the test split.
            "test": {
                "arms": {
                    "mlp": {"per_seed": [
                        {"seed": seed, "ll": -ll}
                        for seed, ll in sorted(mlp.items())]},
                    "full": {"per_seed": [
                        {"seed": seed, "ll": -ll}
                        for seed, ll in sorted(full.items())]},
                },
            },
        },
    }
    payload.update(overrides)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def test_choose_seed_picks_the_lowest_validation_ll():
    rows = [{"seed": 7, "ll": 1.5}, {"seed": 13, "ll": 1.2},
            {"seed": 29, "ll": 1.4}]
    assert choose_seed(rows) == (13, 1.2)


def test_choose_seed_breaks_ties_to_the_lowest_seed():
    rows = [{"seed": 101, "ll": 1.2}, {"seed": 13, "ll": 1.2},
            {"seed": 42, "ll": 1.9}]
    assert choose_seed(rows) == (13, 1.2)


def test_choose_seed_rejects_an_empty_table():
    with pytest.raises(PinError):
        choose_seed([])


def test_selection_table_reads_the_retrain_summary_validation_split(tmp_path):
    summary = _synthetic_summary(tmp_path / "summary.yaml")
    table = selection_table(summary)

    # The minimum of each arm's table; the test split's argmin is the
    # MAXIMUM of the validation table and is never consulted.
    assert table["mlp"]["chosen_seed"] == 101
    assert table["mlp"]["chosen_validation_ll"] == MLP_LLS[101]
    assert table["full"]["chosen_seed"] == 101
    assert table["full"]["chosen_validation_ll"] == FULL_LLS[101]
    # The recorded table is seed-ordered and carries every registered seed
    # at full precision.
    assert [row["seed"] for row in table["mlp"]["per_seed"]] == list(
        pin_stage1.REGISTERED_TRAINING_SEEDS)
    assert [row["validation_ll"] for row in table["full"]["per_seed"]] == [
        FULL_LLS[seed] for seed in pin_stage1.REGISTERED_TRAINING_SEEDS]


def test_selection_table_breaks_a_tie_to_the_lowest_seed(tmp_path):
    tied = dict(MLP_LLS)
    tied[101] = tied[13] = 1.4370001
    summary = _synthetic_summary(tmp_path / "summary.yaml", mlp_rows=tied)
    table = selection_table(summary)
    assert table["mlp"]["chosen_seed"] == 13
    assert table["mlp"]["chosen_validation_ll"] == 1.4370001


def test_selection_table_refuses_a_missing_seed(tmp_path):
    short = {seed: ll for seed, ll in MLP_LLS.items() if seed != 29}
    summary = _synthetic_summary(tmp_path / "summary.yaml", mlp_rows=short)
    with pytest.raises(PinError) as excinfo:
        selection_table(summary)
    assert "missing [29]" in str(excinfo.value)


def test_selection_table_refuses_a_duplicate_seed(tmp_path):
    summary = _synthetic_summary(tmp_path / "summary.yaml")
    payload = yaml.safe_load(summary.read_text())
    rows = payload["splits"]["validation"]["arms"]["full"]["per_seed"]
    rows.append({"seed": 42, "ll": 1.4372451})
    summary.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(PinError) as excinfo:
        selection_table(summary)
    assert "appear more than once" in str(excinfo.value)
    assert "[42]" in str(excinfo.value)


def test_selection_table_refuses_an_unregistered_seed(tmp_path):
    extra = dict(MLP_LLS)
    del extra[7]
    extra[1234] = 1.4370002
    summary = _synthetic_summary(tmp_path / "summary.yaml", mlp_rows=extra)
    with pytest.raises(PinError) as excinfo:
        selection_table(summary)
    assert "unexpected [1234]" in str(excinfo.value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"),
                                   float("-inf")])
def test_selection_table_refuses_a_non_finite_ll(tmp_path, value):
    broken = dict(FULL_LLS)
    broken[29] = value
    summary = _synthetic_summary(tmp_path / "summary.yaml", full_rows=broken)
    with pytest.raises(PinError) as excinfo:
        selection_table(summary)
    assert "not finite" in str(excinfo.value)


def test_selection_table_refuses_a_four_decimal_rounded_table(tmp_path):
    """The trainer used to round every stored log loss to four decimals."""
    rounded = {seed: round(ll, 4) for seed, ll in MLP_LLS.items()}
    summary = _synthetic_summary(tmp_path / "summary.yaml", mlp_rows=rounded)
    with pytest.raises(PinError) as excinfo:
        selection_table(summary)
    message = str(excinfo.value)
    assert "rounded to four decimals" in message
    assert "seed 7" in message


def test_selection_table_refuses_one_rounded_row(tmp_path):
    """One rounded row is enough: it competes on a different scale."""
    mixed = dict(FULL_LLS)
    mixed[42] = round(mixed[42], 4)
    summary = _synthetic_summary(tmp_path / "summary.yaml", full_rows=mixed)
    with pytest.raises(PinError):
        selection_table(summary)


def test_selection_table_refuses_a_non_numeric_ll(tmp_path):
    summary = _synthetic_summary(tmp_path / "summary.yaml")
    payload = yaml.safe_load(summary.read_text())
    payload["splits"]["validation"]["arms"]["mlp"]["per_seed"][0]["ll"] = "1.4"
    summary.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(PinError) as excinfo:
        selection_table(summary)
    assert "expected a float" in str(excinfo.value)


def test_selection_table_fails_closed_on_a_missing_arm(tmp_path):
    path = tmp_path / "summary.yaml"
    path.write_text(yaml.safe_dump(
        {"splits": {"validation": {"arms": {"mlp": {"per_seed": _rows(
            MLP_LLS, "mlp")}}}}}))
    with pytest.raises(PinError) as excinfo:
        selection_table(path)
    assert "no validation arm full" in str(excinfo.value)


def test_selection_table_fails_closed_on_a_missing_summary(tmp_path):
    with pytest.raises(PinError) as excinfo:
        selection_table(tmp_path / "absent.yaml")
    assert "missing retrain summary" in str(excinfo.value)


def _summary_block(summary, **overrides):
    """`retrain_summary_block` against the shared synthetic frame."""
    frame_dir, facts = _shared_frame()
    kwargs = {
        "frame_dir": frame_dir,
        "frame_hash": dict(FRAME_HASH),
        "frame_facts": facts,
        "validation_matches": pin_stage1.validation_match_count(
            frame_dir, "i7"),
        "stats_cache_role": CACHE_ROLE,
        "stats_cache_path": CACHE_PATH,
        "stats_cache_md5": CACHE_MD5,
    }
    kwargs.update(overrides)
    return retrain_summary_block(summary, **kwargs)


def _retouch(summary, mutate):
    """Re-write a summary with one field changed."""
    payload = yaml.safe_load(Path(summary).read_text())
    mutate(payload)
    Path(summary).write_text(yaml.safe_dump(payload, sort_keys=False))
    return summary


def test_retrain_summary_block_reads_the_validation_split(tmp_path):
    frame_dir, facts = _shared_frame()
    summary = _synthetic_summary(tmp_path / "summary.yaml")
    block = _summary_block(summary)
    assert block["validation_rows"] == facts["validation"]["n_rows"]
    assert block["validation_matches"] == pin_stage1.validation_match_count(
        frame_dir, "i7") == len(set(VALIDATION_DATES))
    assert block["frame_dir"] == frame_dir
    assert block["frame_dir_role"] == pin_stage1.ROLE_BALL_FRAME_I7
    # The md5s the block records are the RECOMPUTED ones, not the summary's.
    assert block["frame_split_md5s"] == {
        split: row["md5"] for split, row in facts.items()}
    assert block["frame_split_rows"] == {
        split: row["n_rows"] for split, row in facts.items()}
    assert block["retrain_config"] == pin_stage1.RETRAIN_CONFIG
    assert block["retrain_config_sha256"] == (
        pin_stage1.retrain_config_sha256())


def test_retrain_summary_block_refuses_another_frame(tmp_path):
    summary = _retouch(
        _synthetic_summary(tmp_path / "summary.yaml"),
        lambda payload: payload["frame"].update(dir="data/xgb_data_v3"))
    with pytest.raises(PinError) as excinfo:
        _summary_block(summary)
    assert "frame.dir" in str(excinfo.value)


def test_retrain_summary_block_refuses_a_missing_row_count(tmp_path):
    summary = _retouch(
        _synthetic_summary(tmp_path / "summary.yaml"),
        lambda payload: payload["splits"]["validation"].pop("n_matches"))
    with pytest.raises(PinError) as excinfo:
        _summary_block(summary)
    assert "splits.validation.n_matches" in str(excinfo.value)


# --- drift: the summary is not evidence for itself (MUST-FIX 1) ------------

def test_retrain_summary_refuses_a_row_count_the_parquet_denies(tmp_path):
    summary = _retouch(
        _synthetic_summary(tmp_path / "summary.yaml"),
        lambda payload: payload["splits"]["validation"].update(n_rows=999))
    with pytest.raises(PinError) as excinfo:
        _summary_block(summary)
    assert "splits.validation.n_rows" in str(excinfo.value)


def test_retrain_summary_refuses_a_match_count_the_parquet_denies(tmp_path):
    summary = _retouch(
        _synthetic_summary(tmp_path / "summary.yaml"),
        lambda payload: payload["splits"]["validation"].update(n_matches=99))
    with pytest.raises(PinError) as excinfo:
        _summary_block(summary)
    assert "splits.validation.n_matches" in str(excinfo.value)


def test_retrain_summary_refuses_a_split_md5_the_parquet_denies(tmp_path):
    summary = _retouch(
        _synthetic_summary(tmp_path / "summary.yaml"),
        lambda payload: payload["frame"]["split_md5s"].update(
            train="0" * 32))
    with pytest.raises(PinError) as excinfo:
        _summary_block(summary)
    assert "frame.split_md5s" in str(excinfo.value)
    assert "train" in str(excinfo.value)


def test_retrain_summary_refuses_a_changed_feature_hash_key(tmp_path):
    summary = _retouch(
        _synthetic_summary(tmp_path / "summary.yaml"),
        lambda payload: payload["frame"]["feature_hash"].update(
            venue_alias_version="venue_aliases_v2"))
    with pytest.raises(PinError) as excinfo:
        _summary_block(summary)
    assert "frame.feature_hash" in str(excinfo.value)
    assert "venue_alias_version" in str(excinfo.value)


def test_retrain_summary_refuses_a_changed_retrain_config(tmp_path,
                                                          monkeypatch):
    """The config sha256 is re-hashed from the file, not copied."""
    config = tmp_path / "seq_stage1_retrain_i7_v1.yaml"
    shutil.copyfile(
        pin_stage1.REPO_ROOT / pin_stage1.RETRAIN_CONFIG, config)
    monkeypatch.setattr(pin_stage1, "RETRAIN_CONFIG", config.as_posix())
    summary = _synthetic_summary(tmp_path / "summary.yaml")
    assert _summary_block(summary)["retrain_config_sha256"]

    # One line appended to the config on disk, nothing else touched.
    config.write_text(config.read_text() + "\n# drifted\n")
    with pytest.raises(PinError) as excinfo:
        _summary_block(summary)
    assert "experiment.config_sha256" in str(excinfo.value)


def test_retrain_summary_refuses_a_cache_that_moved(tmp_path):
    summary = _synthetic_summary(tmp_path / "summary.yaml")
    with pytest.raises(PinError) as excinfo:
        _summary_block(summary, stats_cache_md5="0" * 32)
    assert "frame.stats_cache.md5" in str(excinfo.value)


def test_summary_row_must_match_the_seed_metrics_json(tmp_path):
    """All ten rows are checked against the run that produced them."""
    summary = _synthetic_summary(tmp_path / "summary.yaml")
    assert selection_table(summary)["full"]["chosen_seed"] == 101

    metrics = tmp_path / "full" / "seed_29" / "metrics.json"
    metrics.write_text(json.dumps({"validation_ll": FULL_LLS[29] + 1e-6}))
    with pytest.raises(PinError) as excinfo:
        selection_table(summary)
    message = str(excinfo.value)
    assert "seed 29" in message
    assert "metrics.json" in message


def test_summary_row_without_a_checkpoint_is_refused(tmp_path):
    summary = _synthetic_summary(tmp_path / "summary.yaml")
    (tmp_path / "mlp" / "seed_42" / "metrics.json").unlink()
    with pytest.raises(PinError) as excinfo:
        selection_table(summary)
    assert "missing metrics.json" in str(excinfo.value)


def test_summary_row_pointing_at_another_seed_is_refused(tmp_path):
    summary = _retouch(
        _synthetic_summary(tmp_path / "summary.yaml"),
        lambda payload: payload["splits"]["validation"]["arms"]["mlp"][
            "per_seed"][0].update(
                checkpoint_dir="models/embeddings/seq_stage1/retrain_i7/"
                               "mlp/seed_13"))
    with pytest.raises(PinError) as excinfo:
        selection_table(summary)
    assert "does not end in mlp/seed_7" in str(excinfo.value)


# The full-precision rule, stated once in pin_stage1.FULL_PRECISION_RULE and
# exercised here on the boundary values it is written for.
@pytest.mark.parametrize("value,places", [
    (1.4377, 4), (1.43775, 5), (1.437755, 6), (1.4377551, 7),
    (1.0, 1), (1e-09, pin_stage1.FULL_PRECISION_MIN_DECIMALS),
])
def test_decimal_places_counts_the_round_trip_repr(value, places):
    assert decimal_places(value) == places


@pytest.mark.parametrize("value", [1.4377, 1.0, 0.5, 1.4375, 1.44])
def test_four_decimal_values_are_refused(value):
    assert pin_stage1._is_four_decimal_rounded(value) is True


@pytest.mark.parametrize("value", [1.437755, 1.4377551, 1.43775, 1e-09,
                                   1.4377512345])
def test_full_precision_values_are_accepted(value):
    assert pin_stage1._is_four_decimal_rounded(value) is False


# ---------------------------------------------------------------------------
# 5.4 — the documented seed formula is run_arm's function
# ---------------------------------------------------------------------------

CRICSHEET_IDS = ["1477609", "1478908", "1501896", "1505127", "1", "999999999"]


@pytest.mark.parametrize("cricsheet_id", CRICSHEET_IDS)
@pytest.mark.parametrize("base_seed", [pin_stage1.BASE_SEED,
                                       pin_stage1.SECOND_BASE_SEED,
                                       pin_stage1.THIRD_BATCH_BASE_SEED, 0])
def test_fixture_seed_matches_run_arm(cricsheet_id, base_seed):
    mine = fixture_seed(cricsheet_id, base_seed)
    theirs = run_arm_fixture_seed(cricsheet_id, base_seed)
    assert mine == theirs
    assert 0 <= mine <= (1 << 31) - 1


@pytest.mark.parametrize("cricsheet_id", CRICSHEET_IDS)
def test_sub_seeds_match_run_arm(cricsheet_id):
    seed = fixture_seed(cricsheet_id, pin_stage1.BASE_SEED)
    for stream in pin_stage1.SUB_SEED_STREAMS:
        assert sub_seed(seed, stream) == run_arm_sub_seed(seed, stream)
    assert tuple(pin_stage1.SUB_SEED_STREAMS) == tuple(run_arm.SUB_SEED_STREAMS)


def test_sub_seeds_differ_from_each_other_and_from_the_fixture_seed():
    seed = fixture_seed("1477609", pin_stage1.BASE_SEED)
    values = {stream: sub_seed(seed, stream)
              for stream in pin_stage1.SUB_SEED_STREAMS}
    assert len(set(values.values())) == len(values)
    assert seed not in set(values.values())


# ---------------------------------------------------------------------------
# 5.11 — no sealed path in the committed file
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 5.8 — every command line carries every flag the protocol depends on
# ---------------------------------------------------------------------------

REQUIRED_FLAGS = [
    # Every registered command re-verifies the pins at launch: run_arm
    # refuses to start if a pinned hash moved or if the invocation differs
    # from the arm block (Astra round 1, item 4).
    "--config experiments/configs/seq_stage1_sim_v1.yaml",
    "--bowler-roster-policy models/bowler_roster_policy.json",
    "--bowler-usage-path models/bowler_phase_usage.json",
    "--extras-graft models/auto/b18/extras_graft_v1.json",
    "--clip-low 0.01 --clip-high 0.99",
    "--context-dir data/t20s_json",
    # 1c / D10: the I3 block ids come from the registered fixture set, never
    # from the (possibly sharded) fixture dir a launch scores.
    "--cluster-source-dir data/polymarket_test_v2",
    "--odds betting_odds_polymarket_v2.json",
    "--player-metadata data/all_players_enriched.csv",
    "--base-seed 20260910",
    f"--threads {pin_stage1.THREADS}",
]

ACTIVE_ARMS = ("A", "A50", "B", "C")


@pytest.fixture(scope="module")
def committed():
    return yaml.safe_load(CONFIG_PATH.read_text())


@pytest.mark.parametrize("arm", ACTIVE_ARMS)
@pytest.mark.parametrize("run", ["full_run", "smoke_1a"])
def test_every_command_carries_every_required_flag(committed, arm, run):
    command = committed["arms"][arm][run]["command"]
    for flag in REQUIRED_FLAGS:
        assert flag in command, f"{arm}/{run} is missing {flag}"
    assert f"--arm {arm} " in command
    assert f"--model-dir {committed['arms'][arm]['model_dir']} " in command


@pytest.mark.parametrize("arm", ACTIVE_ARMS)
def test_smoke_differs_from_full_only_in_three_flags(committed, arm):
    block = committed["arms"][arm]
    full = block["full_run"]["command"].split()
    smoke = block["smoke_1a"]["command"].split()
    assert len(full) == len(smoke)
    differing = {full[index - 1] for index in range(1, len(full))
                 if full[index] != smoke[index]}
    assert differing <= {"--fixture-dir", "--n-sims", "--output-dir"}
    assert differing == {"--fixture-dir", "--n-sims", "--output-dir"}
    assert "--n-sims 10 " in block["smoke_1a"]["command"] + " "
    assert block["smoke_1a"]["fixture_dir"] == (
        "models/embeddings/seq_stage1/smoke/fixtures")
    assert block["smoke_1a"]["output_dir"] == (
        f"models/embeddings/seq_stage1/smoke/{arm}")
    assert block["full_run"]["output_dir"] == (
        f"models/embeddings/seq_stage1/full/{arm}")


@pytest.mark.parametrize("arm", ACTIVE_ARMS)
def test_every_run_block_pins_the_registered_cluster_source(committed, arm):
    """1c / D10: every launch stamps I3 blocks from the registered set."""
    block = committed["arms"][arm]
    registered = block["cluster_source_dir"]
    assert registered == pin_stage1._role_path(pin_stage1.ROLE_FIXTURE_SET)
    assert block["cluster_source_dir_role"] == pin_stage1.ROLE_FIXTURE_SET
    assert block["cluster_source_dir_md5"] == block["full_run"][
        "fixture_dir_md5"]

    flag = f"--cluster-source-dir {registered} "
    for name in ("full_run", "smoke_1a"):
        run_block = block[name]
        assert run_block["cluster_source_dir"] == registered
        assert run_block["cluster_source_dir_md5"] == block[
            "cluster_source_dir_md5"]
        assert flag in run_block["command"] + " "
    timing = block["timing_1b"]
    assert timing["cluster_source_dir"] == registered
    assert flag in timing["command_template"] + " "
    for shard in block["full_run"]["shards"]:
        # The shard scores its own fixtures and stamps from the whole set.
        assert shard["cluster_source_dir"] == registered
        assert shard["fixture_dir"] != registered
        assert flag in shard["command"] + " "


def test_chosen_n_sims_drives_the_full_run_commands(committed):
    chosen = committed["convergence_protocol"]["chosen_n_sims"]
    token = (pin_stage1.N_SIMS_TOKEN if chosen == pin_stage1.CONVERGENCE_PLACEHOLDER
             else str(chosen))
    for arm in ACTIVE_ARMS:
        block = committed["arms"][arm]["full_run"]
        assert block["n_sims"] == chosen
        assert f"--n-sims {token} " in block["command"] + " "


def test_selector_pin_is_arm_independent(committed):
    selector = committed["selector"]
    assert selector["class"] == "RosterEmpiricalBowlerSelector"
    for arm in ACTIVE_ARMS:
        block = committed["arms"][arm]
        assert block["bowler_selector"] == selector["class"]
        assert block["roster_policy_md5"] == selector["roster_policy_md5"]
        assert block["bowler_usage_md5"] == selector["bowler_usage_md5"]
        assert block["b10_corpus_md5"] == selector["b10_corpus_md5"]


def test_every_recorded_role_resolves_to_the_pinned_path(committed):
    """Each `<thing>_role` key names the manifest role that owns `<thing>`."""
    from artifacts import artifact_path

    flat = flatten(committed)
    checked = 0
    for key, role in sorted(flat.items()):
        if not isinstance(role, str):
            continue
        # `odds.role` / `fixture_set.role` name the sibling `path`; every
        # other `<thing>_role` names the sibling `<thing>`.
        if key.endswith(".role"):
            target = key[: -len(".role")] + ".path"
        elif key.endswith("_role"):
            target = key[: -len("_role")]
        else:
            continue
        pinned = flat.get(target)
        if pinned is None:  # e.g. scoring.winner_log_loss.odds_role
            continue
        assert pinned == artifact_path(role).as_posix(), (
            f"{key} = {role!r} does not resolve to {target} = {pinned!r}")
        checked += 1
    assert checked >= 40, f"only {checked} role/path pairs were checked"


def test_role_keys_cover_every_manifest_owned_artifact(committed):
    """No manifest-owned path is pinned without naming its role."""
    from artifacts import load_manifest

    by_path = {row["path"]: role for role, row in load_manifest().items()}
    flat = flatten(committed)
    for key, value in flat.items():
        if not isinstance(value, str) or value not in by_path:
            continue
        if key.endswith(("_role", "_note", "_source", "_md5", "_sha256")):
            continue
        sibling = f"{key}_role" if not key.endswith(".path") else (
            key.rsplit(".", 1)[0] + ".role")
        if key.endswith(".fixture_dir"):
            sibling = key + "_role"
        assert flat.get(sibling) == by_path[value], (
            f"{key} pins manifest-owned {value!r} but {sibling} is "
            f"{flat.get(sibling)!r}, expected {by_path[value]!r}")


def test_c114_is_deferred_with_no_artifacts(committed):
    c114 = committed["arms"]["C114"]
    assert c114["status"] == "deferred"
    assert c114["artifacts"] is None
    assert c114["reason_deferred"]
    assert "command" not in c114 and "checkpoint" not in c114


# ---------------------------------------------------------------------------
# Provenance: the source closure and the runtime are pinned facts
# ---------------------------------------------------------------------------

@needs_repin
def test_source_closure_is_hashed_in_full(committed):
    """Every closure file on this checkout is hashed, with a live md5."""
    from artifacts import md5_file

    provenance = committed["provenance"]
    pinned = provenance["source_md5"]
    absent = provenance["source_md5_absent"]

    assert set(pinned) | set(absent) == set(
        pin_stage1.PROVENANCE_SOURCE_CLOSURE)
    assert provenance["source_count"] == len(pinned)
    assert absent == [], f"closure files missing from the checkout: {absent}"
    for source, recorded in pinned.items():
        path = pin_stage1.REPO_ROOT / source
        assert path.is_file(), source
        assert md5_file(path) == recorded, f"{source} md5 is stale"


def test_source_closure_reaches_past_the_runner_and_the_engine(committed):
    """The Astra finding: five files were not the execution path."""
    pinned = set(committed["provenance"]["source_md5"])
    for required in (
        "scripts/sequence_track/run_arm.py",       # the runner
        "scripts/sequence_track/pin_stage1.py",    # this pin script
        "scripts/sequence_track/audit_cross_arm.py",
        "scripts/sequence_track/score_realism.py",
        "scripts/sequence_track/score_props.py",
        "scripts/sim_eval/run_sim_eval.py",        # the delegated runner
        "scripts/sim_eval/same_day_stats.py",      # replay
        "scripts/tracker_rehydration.py",          # rehydration
        "scripts/loaders_common.py",               # same-day chronology
        "scripts/stats_sqlite_backend.py",         # the cache contract
        "scripts/feature_registry.py",             # the feature contract
        "scripts/sim_eval/eval_statistics.py",     # the block bootstrap
        "scripts/sim_eval/market_math.py",         # cost scenarios
        "scripts/sim_eval/prop_fair_baselines.py",  # the prop reducer
    ):
        assert required in pinned, f"{required} is not pinned"
    assert len(pinned) >= 25


def test_runtime_matches_this_environment(committed):
    import platform
    from importlib import metadata

    runtime = committed["provenance"]["runtime"]
    assert runtime["verified"] is True
    assert runtime["python"] == platform.python_version()
    for name, recorded in runtime["packages"].items():
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = None
        assert recorded == installed, f"{name}: {recorded!r} vs {installed!r}"


@pytest.mark.needs_artifacts
@needs_repin
def test_tampered_source_md5_fails(tmp_path):
    tampered = tmp_path / "seq_stage1_sim_v1.yaml"
    shutil.copyfile(CONFIG_PATH, tampered)
    payload = yaml.safe_load(tampered.read_text())
    payload["provenance"]["source_md5"]["scripts/sim_v1_2.py"] = "0" * 32
    tampered.write_text(yaml.safe_dump(payload, sort_keys=False))
    assert any("provenance.source_md5.scripts/sim_v1_2.py" in problem
               for problem in verify_config(tampered))


@pytest.mark.needs_artifacts
@needs_repin
def test_tampered_runtime_fails(tmp_path):
    tampered = tmp_path / "seq_stage1_sim_v1.yaml"
    shutil.copyfile(CONFIG_PATH, tampered)
    payload = yaml.safe_load(tampered.read_text())
    payload["provenance"]["runtime"]["packages"]["torch"] = "0.0.0"
    tampered.write_text(yaml.safe_dump(payload, sort_keys=False))
    assert any("provenance.runtime.packages.torch" in problem
               for problem in verify_config(tampered))


def test_flatten_records_empty_containers():
    """An empty list or mapping must still be a comparable leaf."""
    assert flatten({"a": []}) == {"a": "<empty list>"}
    assert flatten({"a": {}}) == {"a": "<empty mapping>"}
    # A placeholder that quietly gains a row, or quietly changes shape, is a
    # difference either way round.
    assert diff_config({"a": []}, {"a": {}})
    assert diff_config({"a": []}, {"a": [1]})
    assert diff_config({"a": [1]}, {"a": []})


def test_committed_config_names_no_sealed_path():
    payload = yaml.safe_load(CONFIG_PATH.read_text())
    declared = payload["experiment"]["forbidden_data"]
    assert declared == ["data/golden", "data/forward_holdout"]
    payload["experiment"].pop("forbidden_data")
    text = yaml.safe_dump(payload)
    for forbidden in declared:
        assert forbidden not in text


# ---------------------------------------------------------------------------
# 5.9 — the seed-interval overlap screen (Astra round 1, item 5)
# ---------------------------------------------------------------------------

def test_overlap_check_is_recorded_for_every_registered_base_seed(committed):
    block = committed["seeds"]["overlap_check"]
    assert block["candidates"] == pin_stage1.CONVERGENCE_CANDIDATES
    for base_seed in pin_stage1.BATCH_BASE_SEEDS:
        rows = block["by_base_seed"][str(base_seed)]
        assert [row["n_sims"] for row in rows] == block["candidates"]
        for row in rows:
            assert row["disjoint"] == (row["min_seed_gap"] >= row["n_sims"])
            assert (row["overlapping_pairs"] == 0) is row["disjoint"]


def test_the_joint_screen_is_the_one_that_governs_a_candidate(committed):
    """Astra round 2, item 5: the three batches share one seed line."""
    block = committed["seeds"]["overlap_check"]
    joint = block["joint"]
    assert joint["base_seeds"] == list(pin_stage1.BATCH_BASE_SEEDS)

    # The joint gap is STRICTLY tighter than every single-seed gap: this is
    # exactly why screening each base seed alone was not enough.
    singles = [block["by_base_seed"][str(seed)][0]["min_seed_gap"]
               for seed in pin_stage1.BATCH_BASE_SEEDS]
    assert joint["joint_min_seed_gap"] < min(singles)

    permitted = joint["largest_permitted_candidate"]
    assert permitted is not None and permitted <= joint["joint_min_seed_gap"]
    for row in joint["rows"]:
        expected = row["n_sims"] <= joint["joint_min_seed_gap"]
        assert row["disjoint"] is expected, row
        assert row["permitted"] == (
            "yes" if expected else "not_permitted_without_new_seeds")
    # The ladder must run past the wall, so the config SHOWS where it is.
    assert any(row["permitted"] == "not_permitted_without_new_seeds"
               for row in joint["rows"])


def test_overlap_screen_cli_refuses_a_jointly_forbidden_candidate():
    forbidden = [n for n in pin_stage1.CONVERGENCE_CANDIDATES
                 if n > 3290]
    assert forbidden, "the candidate ladder no longer crosses the wall"
    assert main(["--check-seed-overlap", "--n-sims", str(forbidden[0])]) == 1


def test_timing_and_variability_blocks_are_registered(committed):
    """Astra round 2, item 4c: the 1b batches must be launchable."""
    for arm in ACTIVE_ARMS:
        block = committed["arms"][arm]
        timing = block["timing_1b"]
        assert timing["fixture_dir"] == (
            "models/embeddings/seq_stage1/timing/fixtures")
        assert timing["base_seeds"] == list(pin_stage1.BATCH_BASE_SEEDS)
        # The permitted list is the jointly screened one; the full ladder
        # is recorded separately as what was screened (Astra round 3, #2).
        assert timing["candidates_screened"] == (
            pin_stage1.CONVERGENCE_CANDIDATES)
        assert timing["n_sims"] == [
            row["n_sims"] for row
            in committed["seeds"]["overlap_check"]["joint"]["rows"]
            if row["disjoint"]]
        assert timing["n_sims"] != timing["candidates_screened"], (
            "the ladder must cross the joint wall, or the ceiling is "
            "untested")
        assert timing["fixture_count_expected"] == 10
        assert timing["output_dir_template"] == (
            f"models/embeddings/seq_stage1/timing/{arm}"
            "/seed<base_seed>_n<n_sims>")
        assert "<batch_base_seed>" in timing["command_template"]
        rerun = block["variability_rerun"]
        assert rerun["base_seeds"] == [pin_stage1.SECOND_BASE_SEED,
                                       pin_stage1.THIRD_BATCH_BASE_SEED]
        assert rerun["runs_through"] == "the timing_1b block above"

        # The smoke and full blocks pin exactly one seed each.
        assert block["smoke_1a"]["base_seeds"] == [pin_stage1.BASE_SEED]
        assert block["full_run"]["base_seeds"] == [pin_stage1.BASE_SEED]

        # Astra round 3, item 1: every launchable block binds its own
        # fixture directory, inventory hash and count. The 1b shard was
        # unpinned (and therefore unclaimable) until it was built on
        # 2026-09-11 (stage 1 D9.1); since then it is pinned like the others.
        for name in ("smoke_1a", "full_run", "timing_1b"):
            assert block[name]["fixture_dir_md5"], name
            assert block[name]["fixture_count"] > 0, name
        assert timing["fixture_count"] == timing["fixture_count_expected"] == 10
        assert timing["fixture_dir_status"] == "present"
        # 50 is permitted for 1b (user decision 2026-09-11) and is the
        # smallest screened candidate; 6400 stays unpermitted.
        assert timing["n_sims"][0] == 50
        assert 6400 not in timing["n_sims"]


def test_the_extra_source_files_are_pinned(committed):
    """Astra round 2, item 6."""
    sources = committed["provenance"]["source_md5"]
    for name in ("scripts/identity_maps.py", "scripts/match_identity.py",
                 "scripts/elo_update.py",
                 "scripts/sim_eval/settlement_common.py"):
        assert name in sources, name
        assert len(sources[name]) == 32
    assert committed["provenance"]["source_md5_absent"] == []


def test_overlap_screen_cli_passes_on_the_registered_set():
    assert main(["--check-seed-overlap"]) == 0
    assert main(["--check-seed-overlap", "--n-sims", "100"]) == 0


def test_overlap_prose_states_the_mechanism(committed):
    text = " ".join([
        committed["seeds"]["interval_mechanism"],
        committed["seeds"]["interval_guard"],
    ]).lower()
    assert "random_seed + i" in text
    assert "grows with n_sims" in text
    assert "does not cancel" in text
    assert "refuses" in text
    for deviation in committed["deviations"]:
        if deviation["id"] == "single_global_rng_stream":
            second = deviation["second_consequence"].lower()
            assert "vanishingly" not in second
            assert "grows with n_sims" in second
            assert "does not cancel" in second
            break
    else:
        raise AssertionError("the RNG-stream deviation is missing")


# ---------------------------------------------------------------------------
# D6 6.1 / 6.3 — B and C are pinned from the i7 retrain, with their contract
# ---------------------------------------------------------------------------

# The synthetic frame declaration (`FRAME_HASH`) and the parquets it names
# are built once near the top of this module; the contract below agrees with
# BOTH, and each test breaks one field at a time.

def _contract(retrain_arm="mlp", seed=101, cache_md5=CACHE_MD5,
              frame_dir=None, facts=None, feature_hash=None, **overrides):
    """A training contract that agrees with the synthetic frame on disk."""
    shared_dir, shared_facts = _shared_frame()
    frame_dir = shared_dir if frame_dir is None else frame_dir
    facts = shared_facts if facts is None else facts
    feature_hash = dict(FRAME_HASH if feature_hash is None else feature_hash)
    names = list(expected_feature_names())
    contract = {
        "contract_version": "t1_training_contract_v1",
        "frame_dir": frame_dir,
        "frame_version": feature_hash["version"],
        "feature_hash": feature_hash,
        "delivery_semantics": feature_hash["delivery_semantics"],
        "venue_alias_version": feature_hash["venue_alias_version"],
        "venue_alias_sha256": feature_hash["venue_alias_sha256"],
        "split_files": {split: dict(row) for split, row in facts.items()},
        "feature_names": names,
        "feature_names_sha256": feature_names_sha256(names),
        "architecture": {"dmodel": 128, "layers": 2, "heads": 4,
                         "arm": retrain_arm},
        "seed": seed,
        "best_epoch": 4,
        "stats_cache": {
            "role": CACHE_ROLE,
            "path": CACHE_PATH,
            "md5": cache_md5,
            "venue_alias_version": feature_hash["venue_alias_version"],
            "same_day_order_version": SAME_DAY_ORDER,
        },
    }
    contract.update(overrides)
    return contract


def _write_checkpoint(model_dir, contract, validation_ll=1.4377551):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.pt").write_bytes(str(model_dir).encode())
    payload = {"validation_ll": validation_ll}
    if contract is not None:
        payload["training_contract"] = contract
    (model_dir / "metrics.json").write_text(json.dumps(payload))
    return model_dir


def _frame_block(model_dir, *, frame_hash=None, cache_md5=CACHE_MD5,
                 expected_seed=101, frame_dir=None, facts=None):
    shared_dir, shared_facts = _shared_frame()
    return training_frame_block(
        read_training_contract(model_dir),
        model_dir=model_dir,
        frame_dir=shared_dir if frame_dir is None else frame_dir,
        frame_hash=dict(frame_hash or FRAME_HASH),
        frame_facts=shared_facts if facts is None else facts,
        stats_cache_role=CACHE_ROLE,
        stats_cache_path=CACHE_PATH,
        stats_cache_md5=cache_md5,
        expected_seed=expected_seed)


def test_stage_1_resolves_b_and_c_from_the_retrain_namespace():
    """6.1: composed from segments, and the ablation dir is not on the path."""
    assert pin_stage1.RETRAIN_DIR == (
        pin_stage1.SEQ_STAGE1_ROOT / "retrain_i7")
    assert pin_stage1.RETRAIN_SUMMARY == (
        pin_stage1.RETRAIN_DIR / "summary.yaml")
    assert not hasattr(pin_stage1, "ABLATION_DIR")
    assert not hasattr(pin_stage1, "ABLATION_SUMMARY")
    for arm in ("B", "C"):
        spec = pin_stage1.ARM_SPEC[arm]
        assert spec["stats_version"] == "i7"
        assert spec["retrain_arm"] in ("mlp", "full")
        assert "ablation_arm" not in spec


def test_arm_spec_stats_versions_match_the_runner_spec():
    """6.3: the runner's spec and this config move in lockstep."""
    for arm in ("A", "A50", "B", "C"):
        assert (pin_stage1.ARM_SPEC[arm]["stats_version"]
                == run_arm.ARMS[arm]["stats_version"]), arm


def test_write_fails_closed_when_the_retrain_is_absent(tmp_path, monkeypatch,
                                                       capsys):
    """No traceback, a PinError naming the missing summary, nothing written."""
    monkeypatch.setattr(pin_stage1, "RETRAIN_SUMMARY",
                        tmp_path / "absent" / "summary.yaml")
    monkeypatch.setattr(pin_stage1, "FULL_SHARD_ROOT", tmp_path / "shards")
    target = tmp_path / "written.yaml"
    assert main(["--write", "--config", str(target)]) == 1
    assert not target.exists()
    # Nothing outside the YAML either: the shard copies are the one write a
    # pin makes, and a --write that cannot produce a config must not leave
    # 19 MB of fixtures behind.
    assert not (tmp_path / "shards").exists()
    message = capsys.readouterr().err
    assert "pin_stage1: ERROR: missing retrain summary" in message
    assert "summary.yaml" in message


def test_training_contract_pins_the_frame_and_the_cache(tmp_path):
    frame_dir, facts = _shared_frame()
    block = _frame_block(_write_checkpoint(tmp_path / "seed_101",
                                           _contract()))
    assert block["dir"] == frame_dir
    assert block["dir_role"] == pin_stage1.ROLE_BALL_FRAME_I7
    # The md5s, rows and date ranges are the RECOMPUTED ones.
    assert block["split_md5s"] == {
        split: row["md5"] for split, row in facts.items()}
    assert block["split_rows"] == {
        split: row["n_rows"] for split, row in facts.items()}
    assert block["split_date_range"]["validation"] == [
        VALIDATION_DATES[0], VALIDATION_DATES[-1]]
    assert block["delivery_semantics"] == "inclusive_total_runs_v1"
    assert block["venue_alias_version"] == "venue_aliases_v1"
    assert block["venue_alias_sha256"] == FRAME_HASH["venue_alias_sha256"]
    assert block["feature_names_sha256"] == feature_names_sha256(
        expected_feature_names())
    assert block["feature_count"] == 50
    assert block["stats_cache_at_training"] == CACHE_PATH
    assert block["stats_cache_at_training_role"] == CACHE_ROLE
    assert block["stats_cache_at_training_md5"] == CACHE_MD5
    assert block["training_seed"] == 101
    assert block["stats_cache_same_day_order_version"] == (
        "date_then_match_id_lexicographic_v1")


def test_a_checkpoint_without_a_contract_is_refused(tmp_path):
    model_dir = _write_checkpoint(tmp_path / "seed_101", None)
    with pytest.raises(PinError) as excinfo:
        read_training_contract(model_dir)
    assert "no training_contract block" in str(excinfo.value)


def test_a_checkpoint_without_metrics_is_refused(tmp_path):
    (tmp_path / "seed_101").mkdir()
    with pytest.raises(PinError) as excinfo:
        read_training_contract(tmp_path / "seed_101")
    assert "missing metrics.json" in str(excinfo.value)


@pytest.mark.parametrize("key", list(pin_stage1.REQUIRED_CONTRACT_KEYS))
def test_every_required_contract_key_is_enforced(tmp_path, key):
    contract = _contract()
    contract[key] = None
    model_dir = _write_checkpoint(tmp_path / "seed_101", contract)
    with pytest.raises(PinError) as excinfo:
        read_training_contract(model_dir)
    assert key in str(excinfo.value)


@pytest.mark.parametrize("patch,needle", [
    ({"frame_dir": "data/xgb_data_v3"}, "frame_dir"),
    ({"frame_version": "v3"}, "frame_version"),
    ({"delivery_semantics": "legal_off_bat_v1"}, "delivery_semantics"),
    ({"venue_alias_version": "venue_aliases_v2"}, "venue_alias_version"),
    ({"venue_alias_sha256": "9" * 64}, "venue_alias_sha256"),
    ({"seed": 42}, "seed"),
    ({"feature_names": [f"feature_{index}" for index in range(49)]},
     "feature count"),
    ({"feature_names_sha256": "short"}, "feature_names_sha256"),
])
def test_a_drifted_contract_field_is_refused(tmp_path, patch, needle):
    model_dir = _write_checkpoint(tmp_path / "seed_101", _contract(**patch))
    with pytest.raises(PinError) as excinfo:
        _frame_block(model_dir)
    assert needle in str(excinfo.value)


def test_a_contract_whose_cache_md5_moved_is_refused(tmp_path):
    """The one check against the world as it is now (6.3)."""
    model_dir = _write_checkpoint(tmp_path / "seed_101",
                                  _contract(cache_md5="0" * 32))
    with pytest.raises(PinError) as excinfo:
        _frame_block(model_dir, cache_md5=CACHE_MD5)
    assert "stats_cache.md5" in str(excinfo.value)
    assert CACHE_PATH in str(excinfo.value)


@pytest.mark.parametrize("field", list(pin_stage1.REQUIRED_CACHE_FIELDS))
def test_an_incomplete_cache_block_is_refused(tmp_path, field):
    contract = _contract()
    contract["stats_cache"][field] = None
    model_dir = _write_checkpoint(tmp_path / "seed_101", contract)
    with pytest.raises(PinError) as excinfo:
        _frame_block(model_dir)
    assert field in str(excinfo.value)


@pytest.mark.parametrize("field", list(pin_stage1.REQUIRED_SPLIT_FIELDS))
def test_an_incomplete_split_block_is_refused(tmp_path, field):
    contract = _contract()
    contract["split_files"]["validation"][field] = None
    model_dir = _write_checkpoint(tmp_path / "seed_101", contract)
    with pytest.raises(PinError) as excinfo:
        _frame_block(model_dir)
    assert field in str(excinfo.value)


def test_a_missing_split_is_refused(tmp_path):
    contract = _contract()
    del contract["split_files"]["train"]
    model_dir = _write_checkpoint(tmp_path / "seed_101", contract)
    with pytest.raises(PinError) as excinfo:
        _frame_block(model_dir)
    assert "no 'train' block" in str(excinfo.value)


def test_a_cache_from_another_role_is_refused(tmp_path):
    contract = _contract()
    contract["stats_cache"]["role"] = "stats_cache_v3_legacy"
    model_dir = _write_checkpoint(tmp_path / "seed_101", contract)
    with pytest.raises(PinError) as excinfo:
        _frame_block(model_dir)
    assert "stats_cache.role" in str(excinfo.value)


def test_a_cache_built_on_another_alias_version_is_refused(tmp_path):
    contract = _contract()
    contract["stats_cache"]["venue_alias_version"] = "venue_aliases_v0"
    model_dir = _write_checkpoint(tmp_path / "seed_101", contract)
    with pytest.raises(PinError) as excinfo:
        _frame_block(model_dir)
    assert "stats_cache.venue_alias_version" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Drift: the pin recomputes the training facts (Astra round 1, MUST-FIX 1)
#
# Each test below changes ONE underlying fact while leaving the contract, the
# metrics, the summary and the YAML self-consistent. Before this change every
# one of them verified.
# ---------------------------------------------------------------------------

def test_a_modified_training_parquet_fails_the_pin(tmp_path):
    """A changed parquet, with the contract left untouched."""
    frame_dir = tmp_path / "xgb_data_i7"
    facts = _build_frame(frame_dir)
    contract = _contract(frame_dir=frame_dir.as_posix(), facts=facts)
    model_dir = _write_checkpoint(tmp_path / "seed_101", contract)
    # As built, the contract and the files agree.
    assert _frame_block(model_dir, frame_dir=frame_dir.as_posix(),
                        facts=facts)["split_md5s"]["validation"] == (
        facts["validation"]["md5"])

    # One more ball in the validation split: same schema, different bytes.
    _write_split_parquet(frame_dir / "cricket_data_i7_validation.parquet",
                         VALIDATION_DATES + ["2025-06-29"])
    pin_stage1._FRAME_SPLIT_FACTS.pop((frame_dir.as_posix(), "i7"), None)
    moved = frame_split_facts(frame_dir, "i7")
    assert moved["validation"]["md5"] != facts["validation"]["md5"]

    with pytest.raises(PinError) as excinfo:
        _frame_block(model_dir, frame_dir=frame_dir.as_posix(), facts=moved)
    assert "split_files.validation.md5" in str(excinfo.value)


def test_a_training_parquet_whose_rows_moved_fails_the_pin(tmp_path):
    frame_dir = tmp_path / "xgb_data_i7"
    facts = _build_frame(frame_dir)
    contract = _contract(frame_dir=frame_dir.as_posix(), facts=facts)
    # The md5 still matches; only the recorded row count is wrong.
    contract["split_files"]["train"]["n_rows"] = facts["train"]["n_rows"] + 1
    model_dir = _write_checkpoint(tmp_path / "seed_101", contract)
    with pytest.raises(PinError) as excinfo:
        _frame_block(model_dir, frame_dir=frame_dir.as_posix(), facts=facts)
    assert "split_files.train.n_rows" in str(excinfo.value)


def test_a_training_parquet_whose_dates_moved_fails_the_pin(tmp_path):
    frame_dir = tmp_path / "xgb_data_i7"
    facts = _build_frame(frame_dir)
    contract = _contract(frame_dir=frame_dir.as_posix(), facts=facts)
    contract["split_files"]["validation"]["match_date_max"] = "2026-06-17"
    model_dir = _write_checkpoint(tmp_path / "seed_101", contract)
    with pytest.raises(PinError) as excinfo:
        _frame_block(model_dir, frame_dir=frame_dir.as_posix(), facts=facts)
    assert "split_files.validation.match_date_max" in str(excinfo.value)


def test_a_missing_training_parquet_fails_the_pin(tmp_path):
    frame_dir = tmp_path / "xgb_data_i7"
    _build_frame(frame_dir)
    (frame_dir / "cricket_data_i7_train.parquet").unlink()
    pin_stage1._FRAME_SPLIT_FACTS.pop((frame_dir.as_posix(), "i7"), None)
    with pytest.raises(PinError) as excinfo:
        frame_split_facts(frame_dir, "i7")
    assert "missing training split parquet" in str(excinfo.value)
    assert "cricket_data_i7_train.parquet" in str(excinfo.value)


def test_an_unreadable_training_parquet_fails_the_pin(tmp_path):
    frame_dir = tmp_path / "xgb_data_i7"
    _build_frame(frame_dir)
    (frame_dir / "cricket_data_i7_validation.parquet").write_bytes(b"PAR1junk")
    pin_stage1._FRAME_SPLIT_FACTS.pop((frame_dir.as_posix(), "i7"), None)
    with pytest.raises(PinError) as excinfo:
        frame_split_facts(frame_dir, "i7")
    assert "cricket_data_i7_validation.parquet" in str(excinfo.value)


def test_a_permuted_feature_name_is_refused(tmp_path):
    """Right length, right membership, wrong order: a different input."""
    names = list(expected_feature_names())
    names[0], names[1] = names[1], names[0]
    contract = _contract(feature_names=names,
                         feature_names_sha256=feature_names_sha256(names))
    model_dir = _write_checkpoint(tmp_path / "seed_101", contract)
    with pytest.raises(PinError) as excinfo:
        _frame_block(model_dir)
    message = str(excinfo.value)
    assert "feature_names[0]" in message
    assert "EB_BAT + EB_BOWL + VENUE + CTX + STATE" in message


def test_a_renamed_feature_is_refused(tmp_path):
    names = list(expected_feature_names())
    names[-1] = "run_rate_required_v2"
    contract = _contract(feature_names=names,
                         feature_names_sha256=feature_names_sha256(names))
    model_dir = _write_checkpoint(tmp_path / "seed_101", contract)
    with pytest.raises(PinError) as excinfo:
        _frame_block(model_dir)
    assert f"feature_names[{len(names) - 1}]" in str(excinfo.value)


def test_a_feature_names_sha_that_hashes_nothing_is_refused(tmp_path):
    contract = _contract(feature_names_sha256="d" * 64)
    model_dir = _write_checkpoint(tmp_path / "seed_101", contract)
    with pytest.raises(PinError) as excinfo:
        _frame_block(model_dir)
    assert "feature_names_sha256" in str(excinfo.value)
    assert "recomputed" in str(excinfo.value)


def test_expected_feature_names_are_the_serving_order():
    """The pin composes the same 50 names the T1 wrapper serves."""
    from sim_t1 import EXPECTED_FEATURE_NAMES

    assert tuple(expected_feature_names()) == tuple(EXPECTED_FEATURE_NAMES)
    assert len(expected_feature_names()) == pin_stage1.T1_FEATURE_COUNT


@pytest.mark.parametrize("key,value", [
    ("venue_alias_version", "venue_aliases_v2"),
    ("venue_alias_sha256", "9" * 64),
    ("delivery_semantics", "legal_off_bat_v1"),
    ("hash", "deadbeefcafe"),
    ("n_features", 50),
])
def test_a_changed_feature_hash_key_in_the_contract_is_refused(
        tmp_path, key, value):
    """The whole frame declaration is compared, not three fields of it."""
    feature_hash = dict(FRAME_HASH)
    feature_hash[key] = value
    contract = _contract()
    contract["feature_hash"] = feature_hash
    model_dir = _write_checkpoint(tmp_path / "seed_101", contract)
    with pytest.raises(PinError) as excinfo:
        _frame_block(model_dir)
    message = str(excinfo.value)
    assert "feature_hash" in message
    assert key in message


def test_a_contract_feature_hash_missing_a_key_is_refused(tmp_path):
    feature_hash = dict(FRAME_HASH)
    del feature_hash["n_features"]
    contract = _contract()
    contract["feature_hash"] = feature_hash
    model_dir = _write_checkpoint(tmp_path / "seed_101", contract)
    with pytest.raises(PinError) as excinfo:
        _frame_block(model_dir)
    assert "'n_features' is missing" in str(excinfo.value)


# ---------------------------------------------------------------------------
# D6 6.1/6.3/6.4/6.5/6.6/6.7 — the config the pin script builds
#
# `build_config()` is exercised against a SYNTHETIC retrain tree, so the
# wiring is tested before the real retrain exists. Everything else in the
# config (the caches, the fixture set, the source closure) is the real
# repository, which is why this test needs artifacts.
# ---------------------------------------------------------------------------

def _synthetic_retrain_tree(root, cache_md5, frame_dir, frame_hash, facts):
    """A complete retrain output tree in the D3 layout.

    The checkpoints declare the REAL i7 frame — its `.feature_hash`, its two
    split parquets and their recomputed md5s, rows and date ranges — because
    `build_config()` recomputes all of that from the repository.
    """
    for retrain_arm, lls in (("mlp", MLP_LLS), ("full", FULL_LLS)):
        for seed, ll in lls.items():
            contract = _contract(
                retrain_arm=retrain_arm, seed=seed, cache_md5=cache_md5,
                frame_dir=frame_dir, facts=facts, feature_hash=frame_hash)
            _write_checkpoint(root / retrain_arm / f"seed_{seed}", contract,
                              validation_ll=ll)
    _synthetic_summary(root / "summary.yaml", with_checkpoints=False,
                       frame_dir=frame_dir, facts=facts,
                       feature_hash=frame_hash, cache_md5=cache_md5)
    return root


@pytest.fixture
def synthetic_config(tmp_path, monkeypatch):
    frame_dir = pin_stage1._role_path(pin_stage1.ROLE_BALL_FRAME_I7)
    frame_hash = pin_stage1.frame_feature_hash(frame_dir)
    facts = frame_split_facts(frame_dir, frame_hash["version"])
    cache_md5 = pin_stage1._md5_file(
        pin_stage1._role_path(pin_stage1.ROLE_STATS_CACHE["i7"]))
    root = _synthetic_retrain_tree(tmp_path / "retrain_i7", cache_md5,
                                   frame_dir, frame_hash, facts)
    monkeypatch.setattr(pin_stage1, "RETRAIN_DIR", root)
    monkeypatch.setattr(pin_stage1, "RETRAIN_SUMMARY", root / "summary.yaml")
    return build_config()


@pytest.mark.needs_artifacts
def test_built_config_pins_b_and_c_on_the_retrain_and_the_i7_cache(
        synthetic_config):
    config = synthetic_config
    for arm, retrain_arm in (("B", "mlp"), ("C", "full")):
        block = config["arms"][arm]
        assert block["model_dir"].endswith(f"retrain_i7/{retrain_arm}/"
                                           "seed_101")
        assert block["checkpoint"].endswith("model.pt")
        assert block["retrain_arm"] == retrain_arm
        assert block["checkpoint_seed"] == 101
        assert "ablation_arm" not in block
        # 6.3: one cache for every arm, by role and by hash.
        assert block["stats_version"] == "i7"
        assert block["stats_cache_role"] == "stats_cache_i7"
        assert block["stats_cache_md5"] == config["arms"]["A"][
            "stats_cache_md5"]
        assert block["stats_cache_md5"] == config["arms"]["A50"][
            "stats_cache_md5"]
        frame = block["training_frame"]
        assert frame["dir"] == pin_stage1._role_path(
            pin_stage1.ROLE_BALL_FRAME_I7)
        assert frame["dir_role"] == "ball_frame_i7"
        assert frame["split_md5s"]["validation"]
        assert frame["delivery_semantics"] == "inclusive_total_runs_v1"
        assert frame["stats_cache_at_training_md5"] == block["stats_cache_md5"]
    # 6.3 again, from the command line the config registers.
    for arm in ("A", "A50", "B", "C"):
        assert "--stats-version i7 " in (
            config["arms"][arm]["smoke_1a"]["command"] + " ")


@pytest.mark.needs_artifacts
def test_built_config_records_the_superseded_ablation_checkpoints(
        synthetic_config):
    """6.1: the old dirs survive only as a record of what was dropped."""
    record = synthetic_config["superseded_checkpoints"]
    assert record["reason"] == pin_stage1.SUPERSEDED_REASON
    assert "2026-09-11" in record["reason"]
    seen = {}
    for row in record["checkpoints"]:
        seen[row["arm"]] = row
        assert row["superseded_model_dir"].startswith(
            "models/embeddings/t1_ablation_v1_mps/")
        assert row["superseded_model_dir"].endswith("seed_101")
        assert len(row["superseded_model_dir_md5"]) == 32
        assert row["reason"] == pin_stage1.SUPERSEDED_REASON
        assert "retrain_i7" in row["now_pinned_at"]
    assert set(seen) == {"B", "C"}
    assert (seen["B"]["superseded_model_dir_md5"]
            != seen["C"]["superseded_model_dir_md5"])

    # The ablation path appears nowhere else in the config.
    flat = flatten(synthetic_config)
    elsewhere = [key for key, value in flat.items()
                 if isinstance(value, str)
                 and "t1_ablation_v1_mps" in value
                 and not key.startswith("superseded_checkpoints")]
    assert elsewhere == []


@pytest.mark.needs_artifacts
def test_built_config_moves_the_two_cache_asymmetries_to_removed(
        synthetic_config):
    """6.4 and 6.5."""
    config = synthetic_config
    ids = [row["id"] for row in config["known_asymmetries"]]
    assert "stats_cache_i7_vs_v3" not in ids
    assert "training_frame_i7_vs_v3" not in ids
    assert ids == ["feature_set_114_vs_50",
                   "arm_a_path_is_the_replay_lifecycle",
                   "br2_g1_is_not_a_stage_1_number"]

    removed = {row["id"]: row for row in config["removed_asymmetries"]}
    assert set(removed) == {"stats_cache_i7_vs_v3", "training_frame_i7_vs_v3"}
    for row in removed.values():
        assert row["removed_on"] == "2026-09-11"
        assert "2026-09-11 retrain on the i7 frame" in row["reason"]

    feature = config["known_asymmetries"][0]
    assert "114" in feature["what"] and "50" in feature["what"]
    assert "system contrasts" in feature["consequence"]
    # Astra round 1, SHOULD 3: A50-A is an isolation only at equal
    # information within one family, and C-B is not an isolation at all.
    assert ("A50-A isolates the feature set at equal information, within "
            "one model family") in feature["consequence"]
    assert "isolates sequence memory" not in feature["consequence"]
    assert ("full-sequence T1 versus token-MLP contrast"
            in feature["consequence"])
    assert ("outcome-history embedding, attention and parameter count"
            in feature["consequence"])


@pytest.mark.needs_artifacts
def test_built_config_records_the_corpus_prior_limitation(synthetic_config):
    """6.6: the D2.4 numbers, verbatim, and identical across arms."""
    limitations = {row["id"]: row
                   for row in synthetic_config["known_limitations"]}
    row = limitations["global_prior_not_as_of"]
    # Astra round 1, MUST-FIX 2: the EXPOSURE is shared; the downstream
    # effect on each arm is unknown, and the feature shift is never compared
    # with the log-loss equivalence margin.
    assert row["identical_across_arms"] is True
    assert "EXPOSURE is identical" in row["identical_across_arms_scope"]
    assert row["differential_effect_on_arms"] == "unknown_and_unmeasured"
    assert "shared exposure, unknown differential effect" in row["consequence"]
    assert "cannot favour an arm" not in row["consequence"]
    assert "equivalence margin" not in row["consequence"]
    assert "not a log loss" not in row["consequence"]
    assert "FEATURE space" in row["feature_shift_is_not_a_log_loss_bound"]
    assert row["source"] == (
        "scripts/sequence_track/measure_global_prior_asof.py")
    # The 0.000174 row is 2,000 venue balls (k=200) / 300 player balls (k=30).
    assert "2,000 balls in a VENUE cell" in row["feature_shift_units"]
    assert "300 balls in a PLAYER cell" in row["feature_shift_units"]
    assert row["max_abs_difference"] == 0.001915
    assert row["difference_convention"] == "whole corpus minus train only"
    pi = {entry["outcome"]: entry
          for entry in row["pi_train_only_vs_whole_corpus"]}
    assert set(pi) == {"wicket", "dot", "single", "two", "four", "six"}
    assert pi["wicket"]["train_only"] == 0.054037
    assert pi["wicket"]["whole_corpus"] == 0.054361
    assert pi["single"]["difference"] == -0.001915
    assert pi["six"]["difference"] == 0.001349
    assert pi["four"]["whole_corpus"] == 0.107803
    shifts = {(entry["cell"], entry["n"]): entry["shift"]
              for entry in row["feature_shift"]}
    assert shifts[("venue", 200)] == 0.000957
    assert shifts[("venue", 2000)] == 0.000174
    assert shifts[("player", 30)] == 0.000957
    assert shifts[("player", 300)] == 0.000174


@pytest.mark.needs_artifacts
def test_built_config_labels_the_teacher_forced_gap_a_diagnostic(
        synthetic_config):
    """6.7."""
    block = synthetic_config["checkpoint_selection"]
    diagnostic = block["teacher_forced_diagnostic"]
    assert diagnostic["label"] == (
        "selection_conditioned_diagnostic_not_evidence")
    assert diagnostic["selected_validation_ll"] == {
        "B": MLP_LLS[101], "C": FULL_LLS[101]}
    assert diagnostic["C_minus_B_validation_ll"] == pytest.approx(
        FULL_LLS[101] - MLP_LLS[101])
    assert "neither is a stage-1 contrast" in diagnostic["what"]
    assert block["source"].endswith("summary.yaml")
    assert block["split_rows"] == 124292
    assert block["registered_seeds"] == list(
        pin_stage1.REGISTERED_TRAINING_SEEDS)


@pytest.mark.needs_artifacts
def test_built_config_round_trips_through_yaml(synthetic_config):
    """What --write would emit parses back to what --verify recomputes."""
    reparsed = yaml.safe_load(pin_stage1.dump_config(synthetic_config))
    assert diff_config(synthetic_config, reparsed) == []


# ---------------------------------------------------------------------------
# The committed config, once it has been re-pinned (6.8, 6.10)
# ---------------------------------------------------------------------------

@needs_repin
def test_the_committed_config_pins_the_retrained_checkpoints(committed):
    for arm, retrain_arm in (("B", "mlp"), ("C", "full")):
        block = committed["arms"][arm]
        assert block["model_dir"].startswith(
            f"models/embeddings/seq_stage1/retrain_i7/{retrain_arm}/seed_")
        assert block["stats_version"] == "i7"
        assert block["stats_cache_md5"] == committed["arms"]["A"][
            "stats_cache_md5"]
        assert block["training_frame"]["dir_role"] == "ball_frame_i7"
        assert "--stats-version i7 " in block["smoke_1a"]["command"] + " "
    assert "superseded_checkpoints" in committed
    assert [row["id"] for row in committed["removed_asymmetries"]] == [
        "stats_cache_i7_vs_v3", "training_frame_i7_vs_v3"]
    assert [row["id"] for row in committed["known_limitations"]] == [
        "global_prior_not_as_of"]


@pytest.mark.needs_artifacts
def test_built_config_keeps_every_role_and_path_paired(synthetic_config):
    """The new training_frame keys must name their manifest roles too.

    The two committed-config role tests only run on the file; this runs the
    same two rules on the config the script builds now, so a re-pin cannot
    introduce an unrolled manifest path.
    """
    from artifacts import artifact_path, load_manifest

    flat = flatten(synthetic_config)
    checked = 0
    for key, role in sorted(flat.items()):
        if not isinstance(role, str):
            continue
        if key.endswith(".role"):
            target = key[: -len(".role")] + ".path"
        elif key.endswith("_role"):
            target = key[: -len("_role")]
        else:
            continue
        pinned = flat.get(target)
        if pinned is None:
            continue
        assert pinned == artifact_path(role).as_posix(), (
            f"{key} = {role!r} does not resolve to {target} = {pinned!r}")
        checked += 1
    assert checked >= 40, f"only {checked} role/path pairs were checked"

    by_path = {row["path"]: role for role, row in load_manifest().items()}
    for key, value in flat.items():
        if not isinstance(value, str) or value not in by_path:
            continue
        if key.endswith(("_role", "_note", "_source", "_md5", "_sha256")):
            continue
        sibling = f"{key}_role" if not key.endswith(".path") else (
            key.rsplit(".", 1)[0] + ".role")
        if key.endswith(".fixture_dir"):
            sibling = key + "_role"
        assert flat.get(sibling) == by_path[value], (
            f"{key} pins manifest-owned {value!r} but {sibling} is "
            f"{flat.get(sibling)!r}, expected {by_path[value]!r}")


# ---------------------------------------------------------------------------
# D11 check 11.1 — the 1d shard partition
#
# The partition is a pure function of the fixture set, so every rule below is
# exercised on a synthetic fixture directory whose answer can be written out
# by hand; the last group checks the committed config against the real one.
# ---------------------------------------------------------------------------

SHARD_DATES = ["2025-07-01", "2025-08-15", "2025-11-02", "2026-01-20",
               "2026-04-16"]


def _fixture_json(date: str) -> str:
    return json.dumps({
        "info": {"dates": [date], "gender": "male", "match_type": "T20",
                 "teams": ["A", "B"], "outcome": {"winner": "A"}},
        "innings": [],
    })


def _synthetic_fixture_set(directory: Path, count: int = 255) -> list:
    """`count` fixtures spread over five dates, ids NOT in date order.

    The ids run backwards against the dates on purpose: a partition that
    sorted by filename, or that used `Path.glob` order, would produce a
    different answer from one that used the (date, id) chronology, and the
    tests below can tell them apart.
    """
    directory.mkdir(parents=True, exist_ok=True)
    ids = []
    for index in range(count):
        fixture_id = str(1600000 - index)
        date = SHARD_DATES[index % len(SHARD_DATES)]
        (directory / f"{fixture_id}.json").write_text(_fixture_json(date))
        ids.append(fixture_id)
    return ids


@pytest.fixture
def shard_tree(tmp_path, monkeypatch):
    """A synthetic fixture set plus a shard root pointing inside tmp_path."""
    source = tmp_path / "fixtures"
    ids = _synthetic_fixture_set(source)
    monkeypatch.setattr(pin_stage1, "FULL_SHARD_ROOT", tmp_path / "shards")
    monkeypatch.setattr(pin_stage1, "FULL_RUN_ROOT", tmp_path / "full")
    return source, ids


def test_chronological_order_is_date_then_id(tmp_path):
    source = tmp_path / "fixtures"
    _synthetic_fixture_set(source, count=12)
    ordered = pin_stage1.chronological_fixture_ids(source)

    def _date(fixture_id):
        return json.loads(
            (source / f"{fixture_id}.json").read_text())["info"]["dates"][0]

    assert ordered == sorted(ordered, key=lambda value: (_date(value), value))
    # Not filename order, and not reverse filename order: the dates lead.
    assert ordered != sorted(ordered)
    assert set(ordered) == {path.stem for path in source.glob("*.json")}


def test_chronological_order_refuses_a_document_it_cannot_place(tmp_path):
    """A fixture the chronology skips would belong to no shard."""
    source = tmp_path / "fixtures"
    _synthetic_fixture_set(source, count=6)
    (source / "9999999.json").write_text(json.dumps({"info": {}}))
    with pytest.raises(PinError) as excinfo:
        pin_stage1.chronological_fixture_ids(source)
    assert "9999999" in str(excinfo.value)
    assert "no shard" in str(excinfo.value)


def test_round_robin_partition_covers_every_fixture_exactly_once():
    ids = [f"id{index}" for index in range(255)]
    shards = pin_stage1.round_robin_partition(ids, 10)
    assert len(shards) == 10
    assert sum(len(shard) for shard in shards) == 255
    flat = [value for shard in shards for value in shard]
    assert sorted(flat) == sorted(ids)
    assert len(set(flat)) == len(flat)
    # i mod 10, so the sizes differ by at most one and shard k starts at k.
    assert {len(shard) for shard in shards} == {25, 26}
    assert [shard[0] for shard in shards] == ids[:10]
    assert shards[3][:2] == ["id3", "id13"]


def test_round_robin_partition_is_deterministic():
    ids = [f"id{index}" for index in range(255)]
    assert (pin_stage1.round_robin_partition(ids, 10)
            == pin_stage1.round_robin_partition(list(ids), 10))


def test_round_robin_partition_refuses_an_empty_shard():
    with pytest.raises(PinError) as excinfo:
        pin_stage1.round_robin_partition(["a", "b"], 10)
    assert "empty" in str(excinfo.value)


def test_materialize_copies_the_fixtures_and_is_idempotent(shard_tree):
    source, ids = shard_tree
    first = pin_stage1.materialize_full_run_shards(source)
    assert [row["action"] for row in first] == ["copied"] * 10
    assert sum(row["fixture_count"] for row in first) == len(ids)

    copied = []
    for index in range(10):
        directory = pin_stage1._abs(pin_stage1.shard_fixture_dir(index))
        stems = sorted(path.stem for path in directory.glob("*.json"))
        copied.extend(stems)
        for stem in stems:
            assert ((directory / f"{stem}.json").read_text()
                    == (source / f"{stem}.json").read_text())
    assert sorted(copied) == sorted(ids)

    # The source is untouched, and a second call changes nothing.
    assert sorted(path.stem for path in source.glob("*.json")) == sorted(ids)
    second = pin_stage1.materialize_full_run_shards(source)
    assert [row["action"] for row in second] == ["unchanged"] * 10


def test_materialize_refuses_a_shard_dir_holding_another_fixture(shard_tree):
    source, _ids = shard_tree
    pin_stage1.materialize_full_run_shards(source)
    intruder = pin_stage1._abs(pin_stage1.shard_fixture_dir(4))
    (intruder / "1234567.json").write_text(_fixture_json("2025-07-01"))
    with pytest.raises(PinError) as excinfo:
        pin_stage1.materialize_full_run_shards(source)
    assert "different fixture set" in str(excinfo.value)


def test_materialize_refuses_an_edited_copy(shard_tree):
    source, _ids = shard_tree
    pin_stage1.materialize_full_run_shards(source)
    directory = pin_stage1._abs(pin_stage1.shard_fixture_dir(2))
    victim = sorted(directory.glob("*.json"))[0]
    victim.write_text(_fixture_json("2026-04-16"))
    with pytest.raises(PinError) as excinfo:
        pin_stage1.materialize_full_run_shards(source)
    assert victim.name in str(excinfo.value)
    assert "COPY" in str(excinfo.value)


def test_materialize_refuses_a_stray_non_fixture_file(shard_tree):
    source, _ids = shard_tree
    pin_stage1.materialize_full_run_shards(source)
    directory = pin_stage1._abs(pin_stage1.shard_fixture_dir(0))
    (directory / "notes.txt").write_text("hand written")
    with pytest.raises(PinError) as excinfo:
        pin_stage1.materialize_full_run_shards(source)
    assert "notes.txt" in str(excinfo.value)


def test_shard_facts_pin_ids_count_and_hash(shard_tree):
    from artifacts import md5_directory

    source, ids = shard_tree
    pin_stage1.materialize_full_run_shards(source)
    facts = pin_stage1.full_run_shard_facts(source)

    assert [row["index"] for row in facts] == list(range(10))
    assert sum(row["fixture_count"] for row in facts) == len(ids)
    union = [value for row in facts for value in row["fixture_ids"]]
    assert sorted(union) == sorted(ids)
    assert len(set(union)) == len(union)
    for row in facts:
        directory = pin_stage1._abs(row["fixture_dir"])
        assert row["fixture_count"] == len(row["fixture_ids"])
        assert row["fixture_dir_md5"] == md5_directory(directory)
        assert row["fixture_dir_hash_contract"] == pin_stage1.DIR_HASH_CONTRACT
        assert row["fixture_dir"].endswith(
            f"/{row['index']}/{pin_stage1.SHARD_FIXTURES_DIRNAME}")


def test_shard_facts_refuse_a_missing_shard_dir(shard_tree):
    source, _ids = shard_tree
    pin_stage1.materialize_full_run_shards(source)
    shutil.rmtree(pin_stage1._abs(pin_stage1.shard_fixture_dir(7)))
    with pytest.raises(PinError) as excinfo:
        pin_stage1.full_run_shard_facts(source)
    assert "missing shard fixture dir" in str(excinfo.value)
    assert "/7/" in str(excinfo.value)


def test_shard_facts_refuse_an_inventory_that_drifted(shard_tree):
    source, _ids = shard_tree
    pin_stage1.materialize_full_run_shards(source)
    directory = pin_stage1._abs(pin_stage1.shard_fixture_dir(1))
    sorted(directory.glob("*.json"))[0].unlink()
    with pytest.raises(PinError) as excinfo:
        pin_stage1.full_run_shard_facts(source)
    assert "the directory and the rule disagree" in str(excinfo.value)


def test_shard_facts_refuse_a_fixture_claimed_by_two_shards(shard_tree,
                                                            monkeypatch):
    """The union/duplicate assertion, forced by a partition that overlaps."""
    source, _ids = shard_tree
    pin_stage1.materialize_full_run_shards(source)
    honest = pin_stage1.round_robin_partition(
        pin_stage1.chronological_fixture_ids(source), 10)

    def overlapping(ordered_ids, n_shards=10):
        shards = [list(shard) for shard in honest]
        shards[1] = list(shards[0])
        return shards

    monkeypatch.setattr(pin_stage1, "round_robin_partition", overlapping)
    with pytest.raises(PinError) as excinfo:
        pin_stage1.full_run_shard_facts(source)
    # Shard 1's directory no longer matches its claimed share, which is the
    # first thing the reader hits; either refusal is the partition failing.
    assert "shard" in str(excinfo.value).lower()


def test_shard_facts_refuse_a_union_that_is_not_the_registered_set(
        shard_tree, monkeypatch):
    source, _ids = shard_tree
    pin_stage1.materialize_full_run_shards(source)
    honest = pin_stage1.round_robin_partition(
        pin_stage1.chronological_fixture_ids(source), 10)
    dropped = honest[3].pop()
    (pin_stage1._abs(pin_stage1.shard_fixture_dir(3))
     / f"{dropped}.json").unlink()

    monkeypatch.setattr(pin_stage1, "round_robin_partition",
                        lambda ordered_ids, n_shards=10: honest)
    with pytest.raises(PinError) as excinfo:
        pin_stage1.full_run_shard_facts(source)
    assert "union" in str(excinfo.value)
    assert dropped in str(excinfo.value)


# --- the committed config's shard block -------------------------------------

def test_full_run_registers_the_whole_partition(committed):
    for arm in ACTIVE_ARMS:
        block = committed["arms"][arm]["full_run"]
        shards = block["shards"]
        assert block["shard_count"] == pin_stage1.FULL_RUN_SHARDS
        assert len(shards) == pin_stage1.FULL_RUN_SHARDS
        assert [shard["index"] for shard in shards] == list(
            range(pin_stage1.FULL_RUN_SHARDS))
        assert block["shard_order_version"] == (
            "date_then_match_id_lexicographic_v1")
        assert block["shard_command_differs_only_in"] == [
            "--fixture-dir", "--output-dir"]

        union = [value for shard in shards for value in shard["fixture_ids"]]
        assert len(set(union)) == len(union), f"{arm}: a fixture in 2 shards"
        assert len(union) == block["fixture_count"] == 255
        for shard in shards:
            assert shard["fixture_count"] == len(shard["fixture_ids"])
            assert len(shard["fixture_dir_md5"]) == 32
            assert shard["fixture_dir"] == (
                "models/embeddings/seq_stage1/full/shards/"
                f"{shard['index']}/fixtures")
            assert shard["output_dir"] == (
                f"models/embeddings/seq_stage1/full/{arm}/"
                f"shard{shard['index']}")


def test_every_arm_shards_the_same_way(committed):
    reference = [
        (shard["index"], shard["fixture_dir"], shard["fixture_dir_md5"],
         tuple(shard["fixture_ids"]))
        for shard in committed["arms"]["A"]["full_run"]["shards"]]
    for arm in ACTIVE_ARMS:
        assert [
            (shard["index"], shard["fixture_dir"], shard["fixture_dir_md5"],
             tuple(shard["fixture_ids"]))
            for shard in committed["arms"][arm]["full_run"]["shards"]
        ] == reference, arm


@pytest.mark.parametrize("arm", ACTIVE_ARMS)
def test_a_shard_command_differs_from_the_full_command_in_two_flags(
        committed, arm):
    block = committed["arms"][arm]["full_run"]
    full = block["command"].split()
    for shard in block["shards"]:
        shard_command = shard["command"].split()
        assert len(shard_command) == len(full)
        differing = {full[index - 1] for index in range(1, len(full))
                     if full[index] != shard_command[index]}
        assert differing == {"--fixture-dir", "--output-dir"}, shard["index"]
        assert f"--fixture-dir {shard['fixture_dir']} " in (
            shard["command"] + " ")
        assert shard["command"].endswith(f"--output-dir {shard['output_dir']}")


def test_the_shard_rule_states_how_the_partition_is_formed(committed):
    rule = committed["arms"]["A"]["full_run"]["shard_rule"].lower()
    assert "round-robin" in rule
    assert "match_date" in rule
    assert "union" in rule
    assert "context" in rule


@pytest.mark.needs_artifacts
@needs_repin
def test_the_committed_shards_are_the_recomputed_partition(committed):
    """The union assertion against the fixture set actually on disk."""
    facts = pin_stage1.full_run_shard_facts()
    pinned = committed["arms"]["A"]["full_run"]["shards"]
    assert [row["index"] for row in facts] == [
        shard["index"] for shard in pinned]
    for row, shard in zip(facts, pinned):
        assert row["fixture_ids"] == shard["fixture_ids"]
        assert row["fixture_dir_md5"] == shard["fixture_dir_md5"]
        assert row["fixture_count"] == shard["fixture_count"]
    registered = set(pin_stage1.fixture_ids_for(
        committed["arms"]["A"]["full_run"]["fixture_dir"]))
    union = [value for row in facts for value in row["fixture_ids"]]
    assert sorted(union) == sorted(registered)
    assert len(set(union)) == len(union)


@pytest.mark.needs_artifacts
@needs_repin
def test_a_tampered_shard_hash_fails_verify(tmp_path):
    tampered = tmp_path / "seq_stage1_sim_v1.yaml"
    shutil.copyfile(CONFIG_PATH, tampered)
    payload = yaml.safe_load(tampered.read_text())
    payload["arms"]["C"]["full_run"]["shards"][2]["fixture_dir_md5"] = "0" * 32
    tampered.write_text(yaml.safe_dump(payload, sort_keys=False))
    assert any("arms.C.full_run.shards[2].fixture_dir_md5" in problem
               for problem in verify_config(tampered))
