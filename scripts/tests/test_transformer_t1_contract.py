"""Frame, kit and training-contract tests for `scripts/transformer_t1.py`.

Sequence track stage 1, deliverable D3 (checks 3.1-3.6, 3.9). Every test
runs on a tiny synthetic frame written to `tmp_path`: no repository frame,
no eval kit, no stats cache and no `models/` directory is read or written.

What is pinned here:
  * 3.1 the parquet stem comes from the frame's `.feature_hash` version,
    `--frame-version` overrides it, and an undeclared frame refuses;
  * 3.2 `--no-kit` (and `--kit-dir none`) complete a run without touching
    the eval kit, and leave the unseen-pair/probe fields absent;
  * 3.3 under `--no-kit` the test split is read only with `--score-test`;
  * 3.4 `validation_ll` survives a round trip at full float64 precision,
    through the trainer's own `metrics.json` writer;
  * 3.5 the contract records the frame, splits, features, architecture and
    the resolved stats cache, and an alias-version mismatch between cache
    and frame refuses to start;
  * 3.6 the contract records the device and that MPS is not bit-reproducible.

Astra round 1 additions:
  * SHOULD 2 — a frame that declares no `delivery_semantics` and is trained
    without `--stats-cache-role` gets NO `training_contract` at all plus a
    warning, instead of a contract full of nulls that the D4 serving guard
    would reject; the two mixed cases (contracted frame without a cache
    role, uncontracted frame with one) refuse to start;
  * MUST-FIX 3 — the retrain runner's reuse check compares architecture,
    optimiser, epochs and the recorded CLI overrides, so a completed
    one-epoch run cannot be adopted by a default invocation.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import transformer_t1 as t1  # noqa: E402
from embeddings_e1 import (CTX_COLS, EB_BAT_COLS, EB_BOWL_COLS,  # noqa: E402
                           VENUE_COLS)

I7_FEATURE_HASH = {
    "hash": "c520a3ba08ae",
    "version": "i7",
    "n_features": 114,
    "splits": {"train_end": "2024-12-31", "val_end": "2025-06-30",
               "test_end": "2026-04-16", "golden_start": "2026-04-17"},
    "gender_filter": "male",
    "delivery_semantics": "inclusive_total_runs_v1",
    "venue_alias_version": "venue_aliases_v1",
    "venue_alias_sha256": "853b32b0" + "0" * 56,
    "venue_alias_active_count": 94,
    "k_player": 30.0,
    "k_venue": 200.0,
}
V3_FEATURE_HASH = {
    "hash": "c520a3ba08ae",
    "version": "v3",
    "n_features": 114,
    "splits": I7_FEATURE_HASH["splits"],
    "gender_filter": "male",
    "k_player": 30.0,
    "k_venue": 200.0,
}


# --------------------------------------------------------------- fixtures

def _split_frame(n_innings: int, balls: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_innings * balls
    data = {
        "innings_id": [f"{1 + i % 2}_m{seed}{i // 2}"
                       for i in range(n_innings) for _ in range(balls)],
        "batter_id": [f"b{i % 4}" for i in range(n)],
        "bowler_id": [f"w{i % 3}" for i in range(n)],
        "ball_outcome": rng.choice([0, 1, 2, 4, 6, -1], size=n,
                                   p=[0.35, 0.35, 0.08, 0.11, 0.06, 0.05]),
        "match_date": [f"2024-0{1 + (i // balls) % 9}-15"
                       for i in range(n)],
    }
    for col in EB_BAT_COLS + EB_BOWL_COLS + VENUE_COLS:
        data[col] = rng.uniform(0.01, 0.5, size=n).astype(np.float32)
    data["is_middle_overs"] = ((np.arange(n) % balls) >= 4).astype(np.float32)
    data["is_death_overs"] = ((np.arange(n) % balls) >= 8).astype(np.float32)
    data["wickets_in_hand"] = np.full(n, 8, dtype=np.int64)
    data["chase_target"] = np.where(np.arange(n) % 2 == 0, 0, 165)
    data["balls_remaining"] = (balls - (np.arange(n) % balls)).astype(np.int64)
    data["score"] = (np.arange(n) % balls * 7).astype(np.int64)
    data["run_rate"] = rng.uniform(4.0, 11.0, size=n).astype(np.float32)
    data["run_rate_required"] = rng.uniform(0.0, 40.0, size=n).astype(np.float32)
    assert set(CTX_COLS) <= set(data), "synthetic frame is missing CTX columns"
    return pd.DataFrame(data)


def write_frame(root: Path, version: str,
                feature_hash: dict | None = I7_FEATURE_HASH) -> Path:
    """A three-split synthetic frame under `root`, stem `cricket_data_<v>_`."""
    root.mkdir(parents=True, exist_ok=True)
    for index, (split, n_innings) in enumerate(
            [("train", 8), ("validation", 4), ("test", 4)]):
        _split_frame(n_innings, 12, seed=index + 1).to_parquet(
            root / f"cricket_data_{version}_{split}.parquet", index=False)
    if feature_hash is not None:
        (root / ".feature_hash").write_text(json.dumps(feature_hash))
    return root


def write_cache(path: Path, alias_version: str | None,
                same_day: str | None = "date_then_match_id_lexicographic_v1"
                ) -> Path:
    """A stats-cache stand-in exposing only the `_meta` rows the guard reads."""
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE _meta (key TEXT PRIMARY KEY, value TEXT)")
    rows = [("schema_version", "4")]
    if alias_version is not None:
        rows.append(("venue_alias_version", alias_version))
    if same_day is not None:
        rows.append(("same_day_order_version", same_day))
    conn.executemany("INSERT INTO _meta VALUES (?, ?)", rows)
    conn.commit()
    conn.close()
    return path


def run_trainer(monkeypatch, data_dir: Path, out: Path, *extra: str) -> dict:
    """One real (tiny, cpu, single-epoch) training run; returns metrics.json."""
    argv = ["transformer_t1.py", "--data-dir", str(data_dir),
            "--out", str(out), "--device", "cpu", "--epochs", "1",
            "--dmodel", "32", "--layers", "1", "--heads", "2",
            "--batch", "4", "--seed", "7", *extra]
    monkeypatch.setattr(sys, "argv", argv)
    t1.main()
    return json.loads((out / "metrics.json").read_text())


CACHE_ARGS = ("--stats-cache-role", "stats_cache_i7", "--stats-cache-path")


@pytest.fixture(scope="module")
def no_kit_run(tmp_path_factory):
    """A completed `--no-kit` run against an i7-style synthetic frame.

    The i7-style frame declares `delivery_semantics`, so since the SHOULD 2
    fix a cache role is mandatory: the contract must name the cache or the
    checkpoint would be unservable.
    """
    root = tmp_path_factory.mktemp("no_kit")
    frame = write_frame(root / "frame", "i7")
    cache = write_cache(root / "cache.sqlite", "venue_aliases_v1")
    out = root / "out"
    monkeypatch = pytest.MonkeyPatch()
    try:
        metrics = run_trainer(monkeypatch, frame, out,
                              "--no-kit", "--kit-dir", str(root / "absent"),
                              *CACHE_ARGS, str(cache))
    finally:
        monkeypatch.undo()
    return {"frame": frame, "out": out, "cache": cache, "metrics": metrics}


# ------------------------------------------------- 3.1 stem resolution

def test_stem_follows_the_frame_version(tmp_path):
    for version, payload in (("v3", V3_FEATURE_HASH), ("i7", I7_FEATURE_HASH)):
        frame = write_frame(tmp_path / version, version, payload)
        assert t1.resolve_frame_version(frame) == version
        assert t1.split_path(frame, version, "train").name == (
            f"cricket_data_{version}_train.parquet")
        loaded = t1.load_split("train", frame)
        assert len(loaded) == 96
        assert "y" in loaded.columns


def test_frame_version_override_wins_over_the_declaration(tmp_path):
    # The directory declares i7 while the files carry the v3 stem: only the
    # override can resolve it, which is what makes the flag load-bearing.
    frame = write_frame(tmp_path / "mixed", "v3", I7_FEATURE_HASH)
    assert t1.resolve_frame_version(frame) == "i7"
    assert t1.resolve_frame_version(frame, "v3") == "v3"
    assert len(t1.load_split("validation", frame, "v3")) == 48


def test_undeclared_frame_refuses_instead_of_guessing(tmp_path):
    frame = write_frame(tmp_path / "bare", "v3", feature_hash=None)
    with pytest.raises(RuntimeError, match="does not declare a frame version"):
        t1.resolve_frame_version(frame)
    assert t1.resolve_frame_version(frame, "v3") == "v3"


def test_match_date_is_read_but_never_a_feature(tmp_path):
    frame = write_frame(tmp_path / "dates", "i7")
    loaded = t1.load_split("train", frame)
    assert "match_date" in loaded.columns
    assert t1.build_features(loaded).shape == (96, t1.N_FEATS)


# ------------------------------- 3.2 / 3.3 kit and test-split optionality

def test_no_kit_run_reads_no_kit_and_no_test_split(no_kit_run):
    metrics = no_kit_run["metrics"]
    contract = metrics["training_contract"]
    # The kit dir passed to this run does not exist, so any read would have
    # raised; the absent fields prove nothing was fabricated in its place.
    assert not (no_kit_run["out"].parent / "absent").exists()
    assert contract["kit_used"] is False
    assert contract["test_split_scored"] is False
    for absent in ("validation_ll_unseen_pairs", "test_ll_unseen_pairs",
                   "validation_ll_by_min_train_balls", "test_ll"):
        assert absent not in metrics, absent
    assert set(contract["split_files"]) == {"train", "validation"}
    assert (no_kit_run["out"] / "model.pt").exists()


def test_score_test_loads_the_test_split(tmp_path, monkeypatch):
    frame = write_frame(tmp_path / "frame", "i7")
    cache = write_cache(tmp_path / "cache.sqlite", "venue_aliases_v1")
    metrics = run_trainer(monkeypatch, frame, tmp_path / "out",
                          "--kit-dir", "none", "--score-test",
                          *CACHE_ARGS, str(cache))
    contract = metrics["training_contract"]
    assert contract["kit_used"] is False          # --kit-dir none == --no-kit
    assert contract["test_split_scored"] is True
    assert set(contract["split_files"]) == {"train", "validation", "test"}
    assert isinstance(metrics["test_ll"], float)
    assert "test_ll_unseen_pairs" not in metrics


# ----------------------------------------------- 3.4 full-precision storage

def test_validation_ll_is_stored_unrounded(no_kit_run):
    metrics = no_kit_run["metrics"]
    stored = metrics["validation_ll"]
    assert isinstance(stored, float)
    assert stored != metrics["validation_ll_rounded4"], (
        "validation_ll must keep more than four decimals")
    assert metrics["validation_ll_rounded4"] == round(stored, 4)


def test_the_trainer_writes_the_full_float_to_the_file(no_kit_run):
    """The bytes on disk, from the trainer's own writer — not a hand-built
    JSON document that only proves `json` can round-trip a float."""
    text = (no_kit_run["out"] / "metrics.json").read_text()
    stored = no_kit_run["metrics"]["validation_ll"]
    # `json.dumps` renders a float through `repr`, so the exact float64 is
    # in the file; a writer that rounded would not produce this line.
    assert f'"validation_ll": {stored!r}' in text
    assert float(json.loads(text)["validation_ll"]) == stored
    assert len(repr(stored).partition(".")[2]) > 4


def test_sixth_decimal_differences_survive_the_metrics_file(tmp_path):
    # Means separated only in the sixth decimal — the decimal that separates
    # two seeds — pushed through the trainer's own `metrics.json` writer.
    # The historical four-decimal field collapses them; the stored value
    # must not.
    stored = []
    for mean_ll in (1.4381341, 1.4381349):
        out = tmp_path / f"run_{mean_ll}"
        out.mkdir()
        t1.write_metrics(out, {"validation_ll": float(mean_ll),
                               "validation_ll_rounded4": round(mean_ll, 4)})
        stored.append(json.loads((out / "metrics.json").read_text()))
    assert stored[0]["validation_ll"] != stored[1]["validation_ll"]
    assert (stored[0]["validation_ll_rounded4"]
            == stored[1]["validation_ll_rounded4"])


def test_two_real_runs_store_distinct_full_precision_means(tmp_path,
                                                           monkeypatch):
    """Two real (tiny) runs through `main`, differing only in seed: the
    stored means are distinct at full precision and each is unrounded."""
    frame = write_frame(tmp_path / "frame", "i7")
    cache = write_cache(tmp_path / "cache.sqlite", "venue_aliases_v1")
    means = []
    for seed in ("7", "13"):
        out = tmp_path / f"out_{seed}"
        argv = ["transformer_t1.py", "--data-dir", str(frame), "--out",
                str(out), "--device", "cpu", "--epochs", "1", "--dmodel",
                "32", "--layers", "1", "--heads", "2", "--batch", "4",
                "--seed", seed, "--no-kit", *CACHE_ARGS, str(cache)]
        monkeypatch.setattr(sys, "argv", argv)
        t1.main()
        metrics = json.loads((out / "metrics.json").read_text())
        means.append(metrics["validation_ll"])
        assert metrics["validation_ll"] != metrics["validation_ll_rounded4"]
    assert means[0] != means[1]


# --------------------------------------------------- 3.5 training contract

def test_contract_is_complete(no_kit_run):
    contract = no_kit_run["metrics"]["training_contract"]
    assert contract["contract_version"] == t1.CONTRACT_VERSION
    assert contract["frame_dir"] == Path(no_kit_run["frame"]).as_posix()
    assert contract["frame_version"] == "i7"
    assert contract["feature_hash"] == I7_FEATURE_HASH
    assert contract["delivery_semantics"] == "inclusive_total_runs_v1"
    assert contract["venue_alias_version"] == "venue_aliases_v1"
    assert contract["venue_alias_sha256"] == (
        I7_FEATURE_HASH["venue_alias_sha256"])

    names = (list(EB_BAT_COLS) + list(EB_BOWL_COLS) + list(VENUE_COLS)
             + list(CTX_COLS) + list(t1.STATE_COLS))
    assert contract["feature_names"] == names
    assert contract["feature_names_sha256"] == hashlib.sha256(
        "\n".join(names).encode("utf-8")).hexdigest()
    assert contract["state_normalisers"] == {
        "balls_remaining": 120.0, "score": 200.0, "run_rate": 12.0,
        "run_rate_required_clip": [0, 36], "run_rate_required": 12.0}
    assert contract["class_mapping"] == {"0": 0, "1": 1, "2": 2, "4": 3,
                                         "6": 4, "-1": 5}
    assert contract["architecture"] == {
        "dmodel": 32, "layers": 1, "heads": 2, "arm": "full",
        "n_params": no_kit_run["metrics"]["n_params"]}
    assert contract["optimiser"] == {
        "lr": 3e-4, "batch": 4, "epochs": 1, "patience": 3,
        "aux": False, "aux_weight": 0.2}
    assert contract["seed"] == 7
    assert contract["best_epoch"] == 0 and contract["epochs_run"] == 1
    assert contract["device"] == "cpu"
    assert contract["mps_bit_reproducible"] is False
    assert set(contract["versions"]) == {"python", "torch", "numpy", "pandas"}
    assert contract["git_head_short"]
    assert contract["stats_cache"]["md5"] == t1.md5_file(no_kit_run["cache"]), (
        "a contracted frame must record the cache it was trained against")


def test_split_records_identify_the_files_actually_read(no_kit_run):
    contract = no_kit_run["metrics"]["training_contract"]
    for split, n_rows in (("train", 96), ("validation", 48)):
        record = contract["split_files"][split]
        path = Path(record["path"])
        assert path == t1.split_path(no_kit_run["frame"], "i7", split)
        assert record["md5"] == t1.md5_file(path)
        assert record["n_rows"] == n_rows
        assert record["match_date_min"] <= record["match_date_max"]


def test_matching_stats_cache_is_recorded(tmp_path, monkeypatch):
    frame = write_frame(tmp_path / "frame", "i7")
    cache = write_cache(tmp_path / "cache.sqlite", "venue_aliases_v1")
    metrics = run_trainer(monkeypatch, frame, tmp_path / "out", "--no-kit",
                          "--kit-dir", str(tmp_path / "absent"),
                          "--stats-cache-role", "stats_cache_i7",
                          "--stats-cache-path", str(cache))
    block = metrics["training_contract"]["stats_cache"]
    assert block["role"] == "stats_cache_i7"
    assert Path(block["path"]).name == "cache.sqlite"
    assert block["md5"] == t1.md5_file(cache)
    assert block["venue_alias_version"] == "venue_aliases_v1"
    assert block["same_day_order_version"] == (
        "date_then_match_id_lexicographic_v1")


def test_cache_with_a_different_alias_version_refuses_to_start(tmp_path,
                                                               monkeypatch):
    frame = write_frame(tmp_path / "frame", "i7")
    cache = write_cache(tmp_path / "cache.sqlite", "venue_aliases_v0")
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="venue_aliases_v0"):
        run_trainer(monkeypatch, frame, out, "--no-kit",
                    "--stats-cache-role", "stats_cache_i7",
                    "--stats-cache-path", str(cache))
    assert not (out / "metrics.json").exists(), "refusal must precede training"


def test_uncontracted_frame_with_a_cache_refuses(tmp_path, monkeypatch):
    # A v3-style frame declares no alias version at all; pairing it with the
    # i7 cache is exactly the identity confusion the guard exists to stop.
    frame = write_frame(tmp_path / "frame", "v3", V3_FEATURE_HASH)
    cache = write_cache(tmp_path / "cache.sqlite", "venue_aliases_v1")
    with pytest.raises(RuntimeError, match="does not match the training frame"):
        run_trainer(monkeypatch, frame, tmp_path / "out", "--no-kit",
                    "--stats-cache-role", "stats_cache_i7",
                    "--stats-cache-path", str(cache))


def test_missing_cache_meta_refuses(tmp_path, monkeypatch):
    frame = write_frame(tmp_path / "frame", "i7")
    empty = tmp_path / "empty.sqlite"
    sqlite3.connect(empty).close()
    with pytest.raises(RuntimeError, match="_meta"):
        run_trainer(monkeypatch, frame, tmp_path / "out", "--no-kit",
                    "--stats-cache-role", "stats_cache_i7",
                    "--stats-cache-path", str(empty))


# ------------------------- Astra SHOULD 2: who gets a training_contract
#
# Before the fix, a v3-style frame (no declared delivery semantics) produced
# a contract whose semantics, alias and cache were all null — a checkpoint
# born rejected by the D4 serving guard. The block is now written only for a
# frame that declares its semantics, and the two mixed cases refuse.

def test_undeclared_semantics_writes_no_contract_and_warns(tmp_path,
                                                           monkeypatch,
                                                           capsys):
    frame = write_frame(tmp_path / "frame", "v3", V3_FEATURE_HASH)
    metrics = run_trainer(monkeypatch, frame, tmp_path / "out", "--no-kit",
                          "--kit-dir", str(tmp_path / "absent"))
    assert "training_contract" not in metrics, (
        "an unidentified frame must omit the block, not fill it with nulls")
    assert metrics["validation_ll"] and metrics["config"]["no_kit"] is True
    out = capsys.readouterr().out
    assert "WARNING" in out and "NO training_contract" in out
    assert "legacy route" in out and "xgb_data_v3" in out


def test_declared_semantics_without_a_cache_role_refuses(tmp_path,
                                                         monkeypatch):
    frame = write_frame(tmp_path / "frame", "i7")
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="must record the stats cache|"
                                           "a contract must name the stats"):
        run_trainer(monkeypatch, frame, out, "--no-kit",
                    "--kit-dir", str(tmp_path / "absent"))
    assert not (out / "metrics.json").exists(), "refusal must precede training"


def test_undeclared_semantics_with_a_cache_role_refuses(tmp_path, monkeypatch):
    # A v3-style frame paired with a v3-style cache (no venue identity on
    # either side) passes the alias check, so only the contract rule stops
    # it: there is nothing for the resolved cache to be recorded in.
    frame = write_frame(tmp_path / "frame", "v3", V3_FEATURE_HASH)
    cache = write_cache(tmp_path / "cache.sqlite", None)
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="no training_contract can be "
                                           "written"):
        run_trainer(monkeypatch, frame, out, "--no-kit",
                    *CACHE_ARGS, str(cache))
    assert not (out / "metrics.json").exists()


# ------------------------------------------------------ 3.7 retrain runner
#
# The runner's own guards: nothing here trains, launches a subprocess or
# writes under models/.

from sequence_track import retrain_i7  # noqa: E402


def _registered_config() -> dict:
    return retrain_i7.load_config(retrain_i7.DEFAULT_CONFIG)


def test_registered_config_matches_the_acceptance_table():
    config = _registered_config()
    assert config["arms"] == ["mlp", "full"]
    assert config["data"]["directory"] == "data/xgb_data_i7"
    assert config["data"]["stats_cache_role"] == "stats_cache_i7"
    assert config["training"] == {
        "device": "mps", "seeds": [7, 13, 29, 42, 101], "dmodel": 128,
        "layers": 2, "heads": 4, "batch": 128, "epochs": 30,
        "learning_rate": 0.0003, "patience": 3, "aux": False,
        "no_kit": True, "score_test": False}
    assert config["outputs"]["directory"] == (
        "models/embeddings/seq_stage1/retrain_i7")


def test_every_command_declines_the_kit_and_the_test_split(tmp_path):
    config = _registered_config()
    effective = retrain_i7.effective_settings(config)
    seen = set()
    for arm in config["arms"]:
        for seed in config["training"]["seeds"]:
            out = retrain_i7.out_dir_for(tmp_path, arm, seed)
            command = retrain_i7.command_for(config, arm, seed, out, effective)
            assert "--no-kit" in command
            assert "--score-test" not in command
            assert "--aux" not in command
            assert command[command.index("--seed") + 1] == str(seed)
            assert command[command.index("--arm") + 1] == arm
            assert out.name == f"seed_{seed}" and out.parent.name == arm
            seen.add((arm, seed))
    assert len(seen) == 10, "ten runs, each (arm, seed) exactly once"


def _write_config(tmp_path, **changes) -> Path:
    config = yaml.safe_load(retrain_i7.DEFAULT_CONFIG.read_text())
    for dotted, value in changes.items():
        section, dot, field = dotted.partition(".")
        if dot:
            config[section][field] = value
        else:
            config[section] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.mark.parametrize("changes,expected", [
    ({"training.seeds": [7, 7, 13, 29, 42]}, "duplicate seed"),
    ({"training.seeds": [7, 13, 29, 42]}, "not the registered five"),
    ({"training.seeds": [7, 13, 29, 42, 102]}, "not the registered five"),
    ({"training.score_test": True}, "must be false"),
    ({"training.no_kit": False}, "must be true"),
    ({"training.aux": True}, "must be false"),
    ({"arms": ["mlp", "mlp"]}, "duplicate arm"),
    ({"arms": ["mlp", "gru"]}, "unknown arm"),
])
def test_config_guards(tmp_path, changes, expected):
    path = _write_config(tmp_path, **changes)
    with pytest.raises(retrain_i7.RetrainError, match=expected):
        retrain_i7.load_config(path)


def test_output_root_is_confined_to_the_stage1_namespace(tmp_path):
    with pytest.raises(retrain_i7.RetrainError, match="refusing to write"):
        retrain_i7.assert_writable(tmp_path / "elsewhere")
    inside = retrain_i7.SEQ_STAGE1_ROOT / "retrain_i7"
    assert retrain_i7.assert_writable(inside) == inside.resolve()


# The runner's reuse check compares the frame's declaration and its training
# parquets themselves, so these tests need a frame on disk rather than a name:
# a tiny synthetic i7 frame, written once and never mutated. Tests that need
# the frame to CHANGE under a checkpoint write their own into `tmp_path`.

@pytest.fixture(scope="module")
def live_frame(tmp_path_factory):
    return write_frame(tmp_path_factory.mktemp("runner") / "frame", "i7")


def _resolved_for(frame: Path, cache_md5: str = "abc") -> dict:
    """What `preflight` returns for `frame`, without resolving a real cache."""
    return {"frame_dir": Path(frame), "frame_version": "i7",
            "feature_hash": t1.read_feature_hash(frame),
            "split_files": retrain_i7.frame_split_facts(frame, "i7"),
            "stats_cache": {"md5": cache_md5}}


@pytest.fixture(scope="module")
def resolved(live_frame):
    return _resolved_for(live_frame)


def _checkpoint_metrics(config: dict, resolved: dict,
                        epochs: int | None = None) -> dict:
    """`metrics.json` of a checkpoint that matches the registered config and
    the frame `resolved` measured — the contract records the frame's own
    declaration and the identity of the two parquets it read."""
    training = config["training"]
    return {"validation_ll": 1.4,
            "training_contract": {
                "contract_version": t1.CONTRACT_VERSION, "seed": 7,
                "architecture": {"arm": "mlp", "dmodel": training["dmodel"],
                                 "layers": training["layers"],
                                 "heads": training["heads"]},
                "optimiser": {
                    "lr": training["learning_rate"],
                    "batch": training["batch"],
                    "epochs": training["epochs"] if epochs is None else epochs,
                    "patience": training["patience"], "aux": False,
                    "aux_weight": t1.AUX_WEIGHT_DEFAULT},
                "frame_version": "i7",
                "frame_dir": Path(resolved["frame_dir"]).as_posix(),
                "feature_hash": copy.deepcopy(resolved["feature_hash"]),
                "kit_used": False, "test_split_scored": False,
                "best_epoch": 4,
                "stats_cache": {"md5": resolved["stats_cache"]["md5"]},
                "split_files": copy.deepcopy(resolved["split_files"])}}


def _checkpoint_dir(tmp_path: Path, overrides: dict | None = None) -> Path:
    out = tmp_path / "mlp" / "seed_7"
    out.mkdir(parents=True, exist_ok=True)
    if overrides is not None:
        retrain_i7.write_run_record(out, {"arm": "mlp", "seed": 7,
                                          "wall_seconds": 12.0,
                                          "overrides": overrides})
    return out


def test_completed_checkpoint_is_only_reused_after_verification(tmp_path,
                                                                resolved):
    config = _registered_config()
    effective = retrain_i7.effective_settings(config)
    out = _checkpoint_dir(tmp_path, overrides={})
    metrics = _checkpoint_metrics(config, resolved)
    assert retrain_i7.verify_checkpoint(out, metrics, config, resolved,
                                        "mlp", 7, effective)["best_epoch"] == 4
    for field, value, expected in (
            ("seed", 13, "seed"),
            ("kit_used", True, "kit_used"),
            ("test_split_scored", True, "test_split_scored"),
            ("frame_version", "v3", "frame_version")):
        broken = json.loads(json.dumps(metrics))
        broken["training_contract"][field] = value
        with pytest.raises(retrain_i7.RetrainError, match=expected):
            retrain_i7.verify_checkpoint(out, broken, config, resolved,
                                         "mlp", 7, effective)
    stale = json.loads(json.dumps(metrics))
    stale["training_contract"]["stats_cache"] = {"md5": "other"}
    with pytest.raises(retrain_i7.RetrainError, match="stats cache md5"):
        retrain_i7.verify_checkpoint(out, stale, config, resolved, "mlp", 7,
                                     effective)
    uncontracted = {"validation_ll": 1.4}
    with pytest.raises(retrain_i7.RetrainError, match="no training_contract"):
        retrain_i7.verify_checkpoint(out, uncontracted, config, resolved,
                                     "mlp", 7, effective)


# --------------- Astra MUST-FIX 3: reuse must match the EFFECTIVE config

def test_one_epoch_run_is_not_reused_by_a_default_invocation(tmp_path,
                                                             resolved):
    """The finding itself: a completed `--epochs 1` run sitting in the
    registered output root must not be adopted by a default invocation and
    reported under the default config hash."""
    config = _registered_config()
    out = _checkpoint_dir(tmp_path, overrides={"epochs": 1})
    smoke = _checkpoint_metrics(config, resolved, epochs=1)

    default = retrain_i7.effective_settings(config)
    with pytest.raises(retrain_i7.RetrainError) as error:
        retrain_i7.verify_checkpoint(out, smoke, config, resolved, "mlp", 7,
                                     default)
    message = str(error.value)
    assert "optimiser.epochs 1 != 30" in message
    assert "overrides {'epochs': 1}" in message
    assert "--force" in message, "the operator must be told how to proceed"

    # The same checkpoint under the invocation that actually produced it.
    one_epoch = retrain_i7.effective_settings(config, epochs=1,
                                              overrides={"epochs": 1})
    assert retrain_i7.verify_checkpoint(out, smoke, config, resolved, "mlp", 7,
                                        one_epoch)["best_epoch"] == 4


@pytest.mark.parametrize("block,field,value,expected", [
    ("architecture", "dmodel", 64, "architecture.dmodel 64 != 128"),
    ("architecture", "layers", 4, "architecture.layers 4 != 2"),
    ("architecture", "heads", 8, "architecture.heads 8 != 4"),
    ("optimiser", "lr", 0.001, "optimiser.lr 0.001 != 0.0003"),
    ("optimiser", "batch", 64, "optimiser.batch 64 != 128"),
    ("optimiser", "patience", 5, "optimiser.patience 5 != 3"),
    ("optimiser", "aux", True, "optimiser.aux True != False"),
    ("optimiser", "aux_weight", 0.5, "optimiser.aux_weight 0.5 != 0.2"),
])
def test_architecture_and_optimiser_drift_refuses(tmp_path, resolved, block,
                                                  field, value, expected):
    config = _registered_config()
    effective = retrain_i7.effective_settings(config)
    out = _checkpoint_dir(tmp_path, overrides={})
    metrics = _checkpoint_metrics(config, resolved)
    metrics["training_contract"][block][field] = value
    with pytest.raises(retrain_i7.RetrainError, match=re.escape(expected)):
        retrain_i7.verify_checkpoint(out, metrics, config, resolved, "mlp", 7,
                                     effective)


def test_missing_block_or_run_record_refuses(tmp_path, resolved):
    config = _registered_config()
    effective = retrain_i7.effective_settings(config)
    metrics = _checkpoint_metrics(config, resolved)

    # No run_record.json: the overrides are unverifiable, so reuse is not
    # allowed even though the contract itself is fine.
    bare = _checkpoint_dir(tmp_path / "bare")
    with pytest.raises(retrain_i7.RetrainError, match="run_record.json"):
        retrain_i7.verify_checkpoint(bare, metrics, config, resolved, "mlp", 7,
                                     effective)

    # A contract predating the architecture/optimiser blocks names every
    # missing field rather than passing on absence.
    out = _checkpoint_dir(tmp_path / "old", overrides={})
    old = json.loads(json.dumps(metrics))
    old["training_contract"]["architecture"] = {"arm": "mlp"}
    old["training_contract"].pop("optimiser")
    with pytest.raises(retrain_i7.RetrainError) as error:
        retrain_i7.verify_checkpoint(out, old, config, resolved, "mlp", 7,
                                     effective)
    message = str(error.value)
    for field in ("architecture.dmodel", "optimiser.lr", "optimiser.epochs",
                  "optimiser.aux_weight"):
        assert field in message, field


def test_an_out_root_override_is_recorded_per_seed(tmp_path, resolved):
    """Override provenance survives into the per-seed summary rows, so a row
    can never claim the registered config produced it."""
    config = _registered_config()
    out = _checkpoint_dir(tmp_path, overrides={"out_root": "elsewhere"})
    effective = retrain_i7.effective_settings(
        config, overrides={"out_root": "elsewhere"})
    retrain_i7.verify_checkpoint(out, _checkpoint_metrics(config, resolved),
                                 config, resolved, "mlp", 7, effective)
    assert retrain_i7.read_run_record(out)["overrides"] == {
        "out_root": "elsewhere"}
    with pytest.raises(retrain_i7.RetrainError, match="overrides"):
        retrain_i7.verify_checkpoint(
            out, _checkpoint_metrics(config, resolved), config, resolved,
            "mlp", 7, retrain_i7.effective_settings(config))


# ------ Astra round 2 MUST-FIX: reuse must match the frame, not its name
#
# A directory path and a frame version both survive a same-path parquet
# replacement and an edited declaration untouched, so a checkpoint that
# passes every name-shaped check can still have been trained on data that no
# longer exists. Preflight measures the frame once; every reused checkpoint
# is compared against that measurement, and the summary reports it.

def test_preflight_measures_the_declaration_and_both_training_splits(
        tmp_path, monkeypatch):
    frame = write_frame(tmp_path / "frame", "i7")
    config = _registered_config()
    config["data"]["directory"] = str(frame)
    # The only part of preflight these tests refuse to do for real: resolving
    # the manifest role would read the production stats cache.
    monkeypatch.setattr(retrain_i7.t1, "resolve_stats_cache",
                        lambda *_args, **_kwargs: {"role": "stats_cache_i7",
                                                   "md5": "abc"})
    measured = retrain_i7.preflight(config)

    assert measured["feature_hash"] == I7_FEATURE_HASH
    assert set(measured["split_files"]) == {"train", "validation"}, (
        "the test split is not measured, because measuring it is reading it")
    for split, n_rows in (("train", 96), ("validation", 48)):
        facts = measured["split_files"][split]
        path = t1.split_path(frame, "i7", split)
        assert facts["md5"] == t1.md5_file(path)
        assert facts["n_rows"] == n_rows
        assert facts["match_date_min"] <= facts["match_date_max"]


def test_the_measurement_agrees_with_the_pin_s_own(live_frame):
    """`pin_stage1` recomputes the same facts from the same parquets and
    compares them against the same contract fields. Two implementations that
    measured differently would disagree about every checkpoint, so the
    identity fields are pinned equal here."""
    from sequence_track import pin_stage1

    ours = retrain_i7.frame_split_facts(live_frame, "i7")
    theirs = pin_stage1.frame_split_facts(live_frame, "i7")
    assert set(ours) == set(theirs) == {"train", "validation"}
    for split in ours:
        for field in retrain_i7.SPLIT_IDENTITY_FIELDS:
            assert ours[split][field] == theirs[split][field], (split, field)


def test_an_unchanged_frame_is_reused(tmp_path, resolved):
    config = _registered_config()
    contract = retrain_i7.verify_checkpoint(
        _checkpoint_dir(tmp_path, overrides={}),
        _checkpoint_metrics(config, resolved), config, resolved, "mlp", 7,
        retrain_i7.effective_settings(config))
    assert contract["best_epoch"] == 4


def test_a_same_path_parquet_replacement_is_refused(tmp_path):
    """The finding: same directory, same frame version, same file name, new
    bytes — and here even the same row count and the same date range, so the
    md5 is the only witness left."""
    config = _registered_config()
    effective = retrain_i7.effective_settings(config)
    frame = write_frame(tmp_path / "frame", "i7")
    trained_on = _resolved_for(frame)
    out = _checkpoint_dir(tmp_path, overrides={})
    metrics = _checkpoint_metrics(config, trained_on)
    assert retrain_i7.verify_checkpoint(out, metrics, config, trained_on,
                                        "mlp", 7, effective)["best_epoch"] == 4

    _split_frame(4, 12, seed=99).to_parquet(
        t1.split_path(frame, "i7", "validation"), index=False)
    now = _resolved_for(frame)
    before = trained_on["split_files"]["validation"]
    after = now["split_files"]["validation"]
    assert after["n_rows"] == before["n_rows"]
    assert after["match_date_min"] == before["match_date_min"]
    assert after["md5"] != before["md5"], "the replacement must be real"

    with pytest.raises(retrain_i7.RetrainError) as error:
        retrain_i7.verify_checkpoint(out, metrics, config, now, "mlp", 7,
                                     effective)
    message = str(error.value)
    assert "split_files.validation.md5" in message
    assert before["md5"] in message and after["md5"] in message
    assert "--force" in message, "the operator must be told how to proceed"


def test_a_changed_split_row_count_is_refused(tmp_path):
    config = _registered_config()
    effective = retrain_i7.effective_settings(config)
    frame = write_frame(tmp_path / "frame", "i7")
    metrics = _checkpoint_metrics(config, _resolved_for(frame))
    out = _checkpoint_dir(tmp_path, overrides={})

    _split_frame(6, 12, seed=1).to_parquet(
        t1.split_path(frame, "i7", "train"), index=False)
    with pytest.raises(retrain_i7.RetrainError) as error:
        retrain_i7.verify_checkpoint(out, metrics, config,
                                     _resolved_for(frame), "mlp", 7, effective)
    assert "split_files.train.n_rows 96 != 72" in str(error.value)


REMOVE = object()


@pytest.mark.parametrize("key,value,expected", [
    ("k_player", 45.0, "feature_hash.k_player 30.0 != 45.0"),
    ("venue_alias_version", "venue_aliases_v2",
     "feature_hash.venue_alias_version 'venue_aliases_v1' != "
     "'venue_aliases_v2'"),
    ("delivery_semantics", REMOVE,
     "feature_hash.delivery_semantics 'inclusive_total_runs_v1' != <absent>"),
    ("recency_halflife_days", 180,
     "feature_hash.recency_halflife_days <absent> != 180"),
])
def test_a_changed_declaration_key_is_refused(tmp_path, key, value, expected):
    """The frame keeps its directory, its version and every parquet byte; one
    field of what it declares itself to be has moved. A checkpoint trained
    under the old declaration is a different experiment."""
    config = _registered_config()
    effective = retrain_i7.effective_settings(config)
    frame = write_frame(tmp_path / "frame", "i7")
    metrics = _checkpoint_metrics(config, _resolved_for(frame))
    out = _checkpoint_dir(tmp_path, overrides={})

    declared = dict(I7_FEATURE_HASH)
    if value is REMOVE:
        declared.pop(key)
    else:
        declared[key] = value
    (frame / ".feature_hash").write_text(json.dumps(declared))

    with pytest.raises(retrain_i7.RetrainError, match=re.escape(expected)):
        retrain_i7.verify_checkpoint(out, metrics, config,
                                     _resolved_for(frame), "mlp", 7, effective)


def _summary_row(resolved: dict, seed: int, **drift) -> dict:
    files = copy.deepcopy(resolved["split_files"])
    files["validation"].update(drift)
    return {"seed": seed, "ll": 1.4, "best_epoch": 4, "wall_seconds": 12.0,
            "checkpoint_md5": "0" * 32, "checkpoint_dir": f"mlp/seed_{seed}",
            "overrides": {}, "_split_files": files}


def test_the_summary_frame_block_is_the_measured_frame(tmp_path, resolved):
    """`summary.yaml` must never be able to pair this invocation's
    declaration with an older run's facts, so the frame block is the
    measurement and every reported checkpoint has to match it."""
    config = _registered_config()
    live = resolved["split_files"]
    summary = retrain_i7.build_summary(
        config, retrain_i7.DEFAULT_CONFIG, resolved,
        {"mlp": [_summary_row(resolved, 7)]}, {}, tmp_path)

    assert summary["frame"]["feature_hash"] == I7_FEATURE_HASH
    assert summary["frame"]["split_md5s"] == {
        split: live[split]["md5"] for split in ("train", "validation")}
    assert summary["frame"]["split_rows"] == {"train": 96, "validation": 48}
    assert summary["splits"]["validation"]["n_rows"] == 48

    with pytest.raises(retrain_i7.RetrainError) as error:
        retrain_i7.build_summary(
            config, retrain_i7.DEFAULT_CONFIG, resolved,
            {"mlp": [_summary_row(resolved, 7, md5="0" * 32, n_rows=47)]},
            {}, tmp_path)
    message = str(error.value)
    assert "split_files.validation.md5" in message
    assert live["validation"]["md5"] in message


SMOKE = retrain_i7.SEQ_STAGE1_ROOT / "retrain_i7_smoke" / "mlp" / "seed_7"


@pytest.mark.needs_artifacts
@pytest.mark.skipif(not (SMOKE / "metrics.json").exists(),
                    reason="the one-epoch runner smoke tree is gitignored")
def test_the_real_one_epoch_smoke_run_is_refused_by_a_default_resume():
    """Same rejection, against the real artifact the finding describes: the
    `--epochs 1 --out-root …_smoke` run on disk. Read-only."""
    config = _registered_config()
    metrics = json.loads((SMOKE / "metrics.json").read_text())
    record = json.loads((SMOKE / "run_record.json").read_text())
    assert record["overrides"] == {
        "out_root": "models/embeddings/seq_stage1/retrain_i7_smoke",
        "epochs": 1}
    # The frame block is taken from the smoke run's own contract: this test
    # is about the epoch budget and the overrides, so the frame is held equal
    # on purpose rather than re-measured from a directory that may since have
    # been re-materialised for an unrelated reason.
    contract = metrics["training_contract"]
    resolved = {"frame_dir": Path("data/xgb_data_i7"), "frame_version": "i7",
                "feature_hash": contract["feature_hash"],
                "split_files": contract["split_files"],
                "stats_cache": contract["stats_cache"]}
    with pytest.raises(retrain_i7.RetrainError) as error:
        retrain_i7.verify_checkpoint(SMOKE, metrics, config, resolved, "mlp",
                                     7, retrain_i7.effective_settings(config))
    message = str(error.value)
    assert "optimiser.epochs 1 != 30" in message
    assert "overrides" in message
