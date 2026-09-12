"""Stage 3 Block B (3a negative-transfer screen) trainer behaviour.

Night 3 draft § "Block B". Everything here runs on the CPU against the
synthetic frame the stage 1/2 contract tests already build, extended with the
two match-level columns Block B needs (`competition_tier`,
`is_international`). Nothing reads a repository frame, a stats cache or
anything under `models/`.

The load-bearing claim of this file is the OFF case: `--tier-embed 0`, no
`--train-tier`, no `--train-match-list` and no `--max-steps` must leave the
model, the training data and the schedule exactly as stage 2 left them, so
every stage 2 checkpoint stays replayable. Each ON case is then tested for
what it is supposed to change and, just as importantly, for what it must not
(validation is never filtered, and the 50-feature contract never grows).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import transformer_t1 as t1  # noqa: E402
from test_transformer_t1_contract import (CACHE_ARGS,  # noqa: E402
                                          run_trainer, write_cache,
                                          write_frame)

DMODEL, LAYERS, HEADS = 16, 2, 4


# ---------------------------------------------------------------- fixtures

def _tiered_frame(root: Path, version: str = "i7") -> Path:
    """The shared synthetic frame plus the two Block B match-level columns.

    The tier is assigned BY MATCH (the id after the underscore in
    `innings_id`), because that is what it is in the real frame and what
    `innings_tier` refuses to see vary within an innings.
    """
    write_frame(root, version)
    for split in ("train", "validation", "test"):
        path = root / f"cricket_data_{version}_{split}.parquet"
        frame = pd.read_parquet(path)
        matches = sorted({t1.match_id_of(v) for v in frame["innings_id"]})
        # Cycled so that even the two-match validation split carries the
        # target tier (3) AND a tier the target-only arm never trains on.
        cycle = [3, 1, 4, 2]
        tier_of = {m: cycle[index % len(cycle)]
                   for index, m in enumerate(matches)}
        keys = [t1.match_id_of(v) for v in frame["innings_id"]]
        frame[t1.TIER_COL] = [tier_of[k] for k in keys]
        frame["is_international"] = [
            1 if tier_of[k] == 4 else 0 for k in keys]
        # The stage 4 sidecar alignment key: a within-innings ball index.
        frame[t1.BALL_IDX_COL] = (
            frame.groupby("innings_id", sort=False).cumcount().to_numpy())
        frame.to_parquet(path, index=False)
    return root


@pytest.fixture()
def frame(tmp_path) -> Path:
    return _tiered_frame(tmp_path / "frame")


@pytest.fixture()
def cache(tmp_path) -> Path:
    return write_cache(tmp_path / "cache.sqlite", "venue_aliases_v1")


def _run(monkeypatch, frame, cache, out, *extra):
    return run_trainer(monkeypatch, frame, out, "--no-kit",
                       "--save-predictions", *CACHE_ARGS, str(cache), *extra)


# ------------------------------------------------- the 50-feature contract

def test_tier_column_is_loaded_but_never_enters_the_feature_vector(frame):
    df = t1.load_split("train", frame, "i7")
    assert t1.TIER_COL in df.columns
    assert t1.build_features(df).shape[1] == t1.N_FEATS == 50


def test_load_split_still_works_on_a_frame_without_the_tier_column(tmp_path):
    plain = write_frame(tmp_path / "plain", "i7")
    df = t1.load_split("train", plain, "i7")
    assert t1.TIER_COL not in df.columns
    assert t1.build_features(df).shape[1] == t1.N_FEATS
    # And the codes helper still answers, with everything UNK.
    assert set(t1.tier_codes(df).tolist()) == {0}


def test_tier_codes_map_out_of_range_to_unk(frame):
    df = t1.load_split("train", frame, "i7")
    df[t1.TIER_COL] = df[t1.TIER_COL].astype(float)
    df.loc[df.index[:3], t1.TIER_COL] = [0.0, 99.0, np.nan]
    codes = t1.tier_codes(df)
    assert codes[:3].tolist() == [0, 0, 0]
    assert set(codes.tolist()) <= set(range(t1.N_TIER_SLOTS))


def test_match_id_of_reads_the_suffix():
    assert t1.match_id_of("1_211048") == "211048"
    assert t1.match_id_of("2_211048") == "211048"
    assert t1.match_id_of("211048") == "211048"


# ------------------------------------------------------ tier conditioning

def test_tier_embed_off_leaves_the_model_untouched():
    """The replay claim: dim 0 builds the stage 2 `mlp` exactly."""
    torch.manual_seed(3)
    plain = t1.T1Model(t1.N_FEATS, DMODEL, LAYERS, HEADS, arm="mlp")
    torch.manual_seed(3)
    off = t1.T1Model(t1.N_FEATS, DMODEL, LAYERS, HEADS, arm="mlp",
                     tier_embed=0)
    assert plain.state_dict().keys() == off.state_dict().keys()
    for name, value in plain.state_dict().items():
        assert torch.equal(value, off.state_dict()[name]), name
    assert not hasattr(off, "tier_emb") and not hasattr(off, "tier_bias")
    assert off.feat_proj.in_features == t1.N_FEATS
    # ...and it still runs without any tier input at all.
    feats = torch.zeros(1, 4, t1.N_FEATS)
    zeros = torch.zeros(1, 4, dtype=torch.long)
    logits, _ = off(feats, zeros, torch.zeros(1, 4, dtype=torch.bool))
    assert logits.shape == (1, 4, 6)


def test_tier_embed_widens_the_token_input_and_biases_the_logits():
    model = t1.T1Model(t1.N_FEATS, DMODEL, LAYERS, HEADS, arm="mlp",
                       tier_embed=8).eval()
    assert model.feat_proj.in_features == t1.N_FEATS + 8
    assert model.tier_emb.num_embeddings == t1.N_TIER_SLOTS == 5
    assert model.tier_bias.weight.shape == (5, 6)
    # Zero-initialised: the conditioning starts neutral rather than throwing
    # the logits several nats off the unconditioned model.
    assert torch.equal(model.tier_bias.weight,
                       torch.zeros_like(model.tier_bias.weight))

    feats = torch.zeros(1, 4, t1.N_FEATS)
    zeros = torch.zeros(1, 4, dtype=torch.long)
    pad = torch.zeros(1, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="tier codes"):
        model(feats, zeros, pad)
    # Two different tiers give different logits once the bias is non-zero.
    with torch.no_grad():
        model.tier_bias.weight[3] = 1.0
    a, _ = model(feats, zeros, pad, tier=torch.full((1, 4), 3))
    b, _ = model(feats, zeros, pad, tier=torch.full((1, 4), 2))
    assert not torch.allclose(a, b)


def test_tier_embed_is_refused_for_a_sequence_arm():
    with pytest.raises(ValueError, match="token_mlp"):
        t1.T1Model(t1.N_FEATS, DMODEL, LAYERS, HEADS, arm="full",
                   tier_embed=8)


def test_collate_pads_tier_codes_with_unk():
    feats = np.zeros((6, t1.N_FEATS), dtype=np.float32)
    y = np.zeros(6, dtype=np.int64)
    tier = np.array([3, 3, 3, 1, 1, 1], dtype=np.int64)
    batch = t1.collate([np.arange(3), np.arange(3, 5)], feats, y, "cpu",
                       tier=tier)
    assert batch.tier.tolist() == [[3, 3, 3], [1, 1, 0]]


# ------------------------------------------------------ training-row filters

def test_train_tier_filters_training_only(monkeypatch, frame, cache,
                                          tmp_path):
    out = tmp_path / "tier3"
    metrics = _run(monkeypatch, frame, cache, out,
                   "--arm", "mlp", "--tier-embed", "4", "--train-tier", "3")
    schedule = metrics["training_schedule"]
    assert schedule["target_tier"] == 3
    assert 0 < schedule["train_innings_selected"] < schedule[
        "train_innings_available"]
    # Every training token seen is a target-tier token, by construction.
    assert schedule["target_tier_tokens_seen"] == schedule["tokens_seen"]
    # Validation is untouched: the saved probabilities still cover the whole
    # split, row-aligned with the validation parquet.
    validation = t1.load_split("validation", frame, "i7")
    with np.load(out / "predictions_validation.npz") as archive:
        assert archive["probs"].shape == (len(validation), 6)


def test_train_match_list_filters_training_only(monkeypatch, frame, cache,
                                                tmp_path):
    train = t1.load_split("train", frame, "i7")
    matches = sorted({t1.match_id_of(v) for v in train["innings_id"]})
    keep = matches[:2]
    listing = tmp_path / "matches.json"
    listing.write_text(json.dumps({"match_ids": keep}))

    out = tmp_path / "rowmatch"
    metrics = _run(monkeypatch, frame, cache, out, "--arm", "mlp",
                   "--train-match-list", str(listing))
    schedule = metrics["training_schedule"]
    assert schedule["train_match_list_n"] == len(keep)
    assert len(schedule["train_match_list_sha256"]) == 64
    kept_rows = int(train["innings_id"].map(t1.match_id_of).isin(keep).sum())
    assert schedule["train_rows_selected"] == kept_rows
    assert schedule["train_innings_selected"] < schedule[
        "train_innings_available"]
    # A list is generic: the same flag takes a target list or a control list.
    assert np.isnan(schedule["target_tier_tokens_seen"])


def test_report_match_lists_counts_exposure_without_filtering(
        monkeypatch, frame, cache, tmp_path):
    """Reviewer MUST-FIX 2: target exposure is MEASURED, and measuring it
    changes nothing about what the run trains on."""
    train = t1.load_split("train", frame, "i7")
    matches = sorted({t1.match_id_of(v) for v in train["innings_id"]})
    first = tmp_path / "list_a.json"
    first.write_text(json.dumps({"match_ids": matches[:1]}))
    second = tmp_path / "list_b.json"
    second.write_text(json.dumps({"match_ids": matches}))

    plain = _run(monkeypatch, frame, cache, tmp_path / "plain", "--arm", "mlp")
    out = tmp_path / "reported"
    metrics = _run(monkeypatch, frame, cache, out, "--arm", "mlp",
                   "--report-match-lists", f"{first},{second}")
    schedule = metrics["training_schedule"]
    seen = schedule["tokens_seen_by_list"]
    assert set(seen) == {"list_a.json", "list_b.json"}
    # The list of ALL matches sees every token the run saw; a one-match list
    # sees a strictly smaller, non-zero share of them.
    assert seen["list_b.json"] == schedule["tokens_seen"]
    assert 0 < seen["list_a.json"] < seen["list_b.json"]
    assert len(schedule["report_match_lists"]["list_a.json"]["sha256"]) == 64
    # It filters nothing: the training subset is exactly the unreported run's.
    for field in ("train_innings_selected", "train_rows_selected",
                  "tokens_seen", "total_steps"):
        assert schedule[field] == plain["training_schedule"][field], field
    assert plain["training_schedule"]["tokens_seen_by_list"] == {}


def test_match_list_accepts_a_bare_list(tmp_path):
    path = tmp_path / "bare.json"
    path.write_text('["1", "2", "3"]')
    ids, digest = t1.load_match_list(path)
    assert ids == {"1", "2", "3"} and len(digest) == 64
    path.write_text("[]")
    with pytest.raises(RuntimeError, match="non-empty"):
        t1.load_match_list(path)


def test_a_filter_that_selects_nothing_refuses(frame):
    train = t1.load_split("train", frame, "i7")
    innings = t1.build_innings(train)
    with pytest.raises(RuntimeError, match="selects no training innings"):
        t1.select_train_innings(train, innings, None, {"no-such-match"})


def test_innings_tier_refuses_a_tier_that_varies(frame):
    train = t1.load_split("train", frame, "i7")
    innings = t1.build_innings(train)
    train.loc[train.index[innings[0][0]], t1.TIER_COL] = 1
    train.loc[train.index[innings[0][1]], t1.TIER_COL] = 2
    with pytest.raises(RuntimeError, match="not constant within innings"):
        t1.innings_tier(train, innings)


# ------------------------------------------------------- the step schedule

def test_step_budget_runs_exactly_max_steps_with_no_early_stop(
        monkeypatch, frame, cache, tmp_path):
    out = tmp_path / "steps"
    metrics = _run(monkeypatch, frame, cache, out, "--arm", "mlp",
                   "--max-steps", "7", "--eval-every", "2",
                   "--patience", "1")
    schedule = metrics["training_schedule"]
    assert schedule["mode"] == "max_steps"
    # Patience 1 would have stopped an epoch schedule almost immediately; the
    # step schedule has no early termination at all.
    assert schedule["total_steps"] == 7
    assert schedule["early_stopping"] == "none (fixed step budget)"
    assert 1 <= schedule["steps_to_best"] <= 7
    assert "earliest step" in schedule["selection_rule"]
    assert schedule["tokens_seen"] > 0
    assert schedule["eval_every"] == 2 and schedule["max_steps"] == 7


def test_epoch_schedule_is_the_default_and_is_unchanged(
        monkeypatch, frame, cache, tmp_path):
    out = tmp_path / "epochs"
    metrics = _run(monkeypatch, frame, cache, out, "--arm", "mlp")
    schedule = metrics["training_schedule"]
    assert schedule["mode"] == "epochs"
    assert schedule["max_steps"] is None and schedule["eval_every"] is None
    assert schedule["early_stopping"] == "patience in epochs"
    assert schedule["tier_embed"] == 0
    assert schedule["train_match_list"] is None
    assert np.isnan(schedule["target_tier_tokens_seen"])
    # One epoch of the fixture's 8 training innings at --batch 4 is two steps.
    assert schedule["total_steps"] == 2
    assert schedule["epochs_run"] == 1


def test_half_a_step_budget_is_refused(monkeypatch, frame, cache, tmp_path):
    with pytest.raises(SystemExit):
        _run(monkeypatch, frame, cache, tmp_path / "half", "--arm", "mlp",
             "--max-steps", "4")
    with pytest.raises(SystemExit):
        _run(monkeypatch, frame, cache, tmp_path / "half2", "--arm", "mlp",
             "--eval-every", "4")


def test_tier_flags_are_refused_on_a_frame_without_the_column(
        monkeypatch, tmp_path, cache):
    plain = write_frame(tmp_path / "plain", "i7")
    with pytest.raises(RuntimeError, match="carries no"):
        _run(monkeypatch, plain, cache, tmp_path / "o", "--arm", "mlp",
             "--tier-embed", "4")


def test_arm_params_and_contract_blocks_do_not_grow(
        monkeypatch, frame, cache, tmp_path):
    """The stage 3 accounting lives in `training_schedule`, so the two
    identity blocks the driver's signature pins stay exactly as they were."""
    metrics = _run(monkeypatch, frame, cache, tmp_path / "grow", "--arm",
                   "mlp", "--tier-embed", "4", "--train-tier", "3",
                   "--max-steps", "4", "--eval-every", "2")
    assert set(metrics["arm_params"]) == {
        "arm", "k", "wiring", "history_input", "key_construction",
        "positional_embedding", "bias", "residual_l2", "base_logits_md5",
        "n_parameters"}
    assert set(metrics["training_contract"]["optimiser"]) == {
        "lr", "batch", "epochs", "patience", "aux", "aux_weight"}


# ------------------------------------------------------- the freeze script

def test_freeze_script_emits_every_registered_list(tmp_path, monkeypatch):
    import sequence_track.stage3a_freeze_tiers as freeze

    frame = _tiered_frame(tmp_path / "frame")
    cricsheet = tmp_path / "cricsheet"
    cricsheet.mkdir()
    train = t1.load_split("train", frame, "i7")
    validation = t1.load_split("validation", frame, "i7")
    ids = sorted({t1.match_id_of(v) for v in
                  list(train["innings_id"]) + list(validation["innings_id"])})
    # Alternate a both-full-members pairing with one involving an associate,
    # so the both-full-members rule has something to exclude.
    for index, match_id in enumerate(ids):
        teams = (["India", "Australia"] if index % 2 == 0
                 else ["Nepal", "England"])
        (cricsheet / f"{match_id}.json").write_text(
            json.dumps({"info": {"teams": teams}}))

    out = tmp_path / "stage3a"
    assert freeze.main(["--data-dir", str(frame), "--cricsheet-dir",
                        str(cricsheet), "--out-dir", str(out),
                        "--batch", "2", "--epochs", "3"]) == 0

    manifest = json.loads((out / "manifest.json").read_text())
    assert set(manifest["lists"]) == {"target_P", "target_E", "rowmatch_P",
                                      "rowmatch_E", "target_E_validation",
                                      "big3_validation"}
    # Code gate finding 1: the E slice the family is READ on must have
    # validation rows of its own; the training list cannot serve as the slice.
    assert manifest["lists"]["target_E_validation"]["n_rows"] > 0
    val_e = json.loads(
        (out / "target_E_validation_matches.json").read_text())
    train_e = json.loads((out / "target_E_matches.json").read_text())
    assert not set(val_e["match_ids"]) & set(train_e["match_ids"])
    # Code gate finding 2: big3 is E-intersected, so it is a subset of the
    # validation-side E slice.
    big3 = json.loads((out / "big3_validation_matches.json").read_text())
    assert set(big3["match_ids"]) <= set(val_e["match_ids"])
    # Every training list is disjoint from the validation split.
    for name, row in manifest["train_validation_disjoint"].items():
        assert row["n_in_validation"] == 0, name
    # E is a superset of P, and the row-matched controls reach their target.
    target_p = json.loads((out / "target_P_matches.json").read_text())
    target_e = json.loads((out / "target_E_matches.json").read_text())
    assert set(target_p["match_ids"]) <= set(target_e["match_ids"])
    for target, control in (("target_P", "rowmatch_P"),
                            ("target_E", "rowmatch_E")):
        assert (manifest["lists"][control]["n_rows"]
                >= manifest["lists"][target]["n_rows"])
        assert manifest["lists"][control]["overshoot_rows"] >= 0
    # The step budget is derived, not typed in.
    steps = json.loads((out / "steps.json").read_text())
    assert steps["eval_every"] == -(-steps["n_train_innings"] // 2)
    assert steps["max_steps"] == 3 * steps["eval_every"]
    # Every emitted list is loadable by the trainer's own reader.
    for name in ("target_P", "target_E", "rowmatch_P", "rowmatch_E"):
        path = out / manifest["lists"][name]["file"]
        loaded, digest = t1.load_match_list(path)
        assert loaded and digest == manifest["lists"][name]["sha256"]


def test_freeze_script_is_deterministic(tmp_path):
    """Same pinned parquet in, same sha256 out — the lists are a frozen
    input, so a rerun must not quietly reshuffle the row-matched control."""
    import sequence_track.stage3a_freeze_tiers as freeze

    frame = _tiered_frame(tmp_path / "frame")
    cricsheet = tmp_path / "cricsheet"
    cricsheet.mkdir()
    digests = []
    for run in ("a", "b"):
        out = tmp_path / run
        freeze.main(["--data-dir", str(frame), "--cricsheet-dir",
                     str(cricsheet), "--out-dir", str(out),
                     "--batch", "2", "--epochs", "3"])
        manifest = json.loads((out / "manifest.json").read_text())
        digests.append({name: row["sha256"]
                        for name, row in manifest["lists"].items()})
    assert digests[0] == digests[1]


# ------------------------------------------------ code gate follow-up fixes

def test_tier_conditioning_is_not_weight_decayed(monkeypatch, frame, cache,
                                                 tmp_path):
    """Finding 3: AdamW decays every parameter, gradient or not, so a tier the
    arm never trains on would have its rows pulled off initialisation."""
    import torch as _torch

    seen = {}
    real = _torch.optim.AdamW

    def spy(params, *args, **kwargs):
        opt = real(params, *args, **kwargs)
        seen["decays"] = [group["weight_decay"] for group in opt.param_groups]
        seen["sizes"] = [len(group["params"]) for group in opt.param_groups]
        return opt

    monkeypatch.setattr(_torch.optim, "AdamW", spy)
    _run(monkeypatch, frame, cache, tmp_path / "decay", "--arm", "mlp",
         "--tier-embed", "4")
    assert sorted(seen["decays"]) == [0.0, 0.01]
    # Exactly the two conditioning tensors are undecayed.
    assert seen["sizes"][seen["decays"].index(0.0)] == 2


def test_tier_conditioning_off_keeps_one_param_group(monkeypatch, frame,
                                                     cache, tmp_path):
    import torch as _torch

    seen = {}
    real = _torch.optim.AdamW

    def spy(params, *args, **kwargs):
        opt = real(params, *args, **kwargs)
        seen["decays"] = [group["weight_decay"] for group in opt.param_groups]
        return opt

    monkeypatch.setattr(_torch.optim, "AdamW", spy)
    _run(monkeypatch, frame, cache, tmp_path / "nodecay", "--arm", "mlp")
    assert seen["decays"] == [0.01]


def test_untrained_tiers_are_recorded(monkeypatch, frame, cache, tmp_path):
    metrics = _run(monkeypatch, frame, cache, tmp_path / "untrained",
                   "--arm", "mlp", "--tier-embed", "4", "--train-tier", "3")
    schedule = metrics["training_schedule"]
    assert schedule["tiers_seen_in_training"] == [3]
    # The fixture's validation split covers more than tier 3, so the rest are
    # reported as untrained rather than left to be inferred.
    assert schedule["untrained_tiers"]
    assert 3 not in schedule["untrained_tiers"]


def test_target_tier_is_reporting_only(monkeypatch, frame, cache, tmp_path):
    """Finding 4: a pooled arm records its target-tier exposure without any
    change to training."""
    metrics = _run(monkeypatch, frame, cache, tmp_path / "pooled",
                   "--arm", "mlp", "--tier-embed", "4", "--target-tier", "3")
    schedule = metrics["training_schedule"]
    assert schedule["target_tier"] == 3
    assert schedule["target_tier_source"] == "--target-tier (reporting only)"
    # Pooled: it trained on everything, and only some tokens are target tier.
    assert schedule["train_innings_selected"] == schedule[
        "train_innings_available"]
    assert 0 < schedule["target_tier_tokens_seen"] < schedule["tokens_seen"]
    assert len(schedule["tiers_seen_in_training"]) > 1


def test_target_tier_contradicting_train_tier_is_refused(
        monkeypatch, frame, cache, tmp_path):
    with pytest.raises(SystemExit):
        _run(monkeypatch, frame, cache, tmp_path / "clash", "--arm", "mlp",
             "--train-tier", "3", "--target-tier", "2")


# --------------------------------------------------- rung 4d extra features

def test_extra_features_widen_the_token_input_and_verify_alignment(
        monkeypatch, frame, cache, tmp_path):
    side_dir = tmp_path / "sidecar"
    side_dir.mkdir()
    cols = ["a_N_asof", "a_var_p0"]
    for split in ("train", "validation"):
        df = t1.load_split(split, frame, "i7")
        pd.DataFrame({
            "innings_id": df["innings_id"].to_numpy(),
            "ball_idx": df[t1.BALL_IDX_COL].to_numpy(),
            "a_N_asof": np.arange(len(df), dtype=np.float32),
            # Stored as a VARIANCE; the loader must hand the model its sqrt.
            "a_var_p0": np.full(len(df), 4.0, dtype=np.float32),
        }).to_parquet(side_dir / f"{split}.parquet", index=False)

    train = t1.load_split("train", frame, "i7")
    matrix, digest = t1.load_extra_features(side_dir, "train", train, cols)
    assert matrix.shape == (len(train), 2) and len(digest) == 64
    assert np.allclose(matrix[:, 1], 2.0)  # sqrt(4)

    metrics = _run(monkeypatch, frame, cache, tmp_path / "4d", "--arm", "mlp",
                   "--extra-features", str(side_dir),
                   "--extra-cols", ",".join(cols))
    block = metrics["arm_params"]["extra_features"]
    assert block["cols"] == cols and block["n_cols"] == 2
    assert block["sqrt_applied"] == ["a_var_p0"]
    assert set(block["sha256_by_split"]) == {"train", "validation"}
    assert block["standardiser"]["kind"] == "train-only mean/std"
    # The 50-feature contract is unchanged; the token input is 52 wide.
    assert metrics["config"]["extra_cols"] == ",".join(cols)


def test_extra_features_refuse_a_misaligned_sidecar(frame, tmp_path):
    side_dir = tmp_path / "bad"
    side_dir.mkdir()
    df = t1.load_split("train", frame, "i7")
    shifted = df[t1.BALL_IDX_COL].to_numpy().copy()
    shifted[10] = shifted[10] + 7
    pd.DataFrame({
        "innings_id": df["innings_id"].to_numpy(),
        "ball_idx": shifted,
        "a_N_asof": np.zeros(len(df), dtype=np.float32),
    }).to_parquet(side_dir / "train.parquet", index=False)
    with pytest.raises(RuntimeError, match="not row-aligned"):
        t1.load_extra_features(side_dir, "train", df, ["a_N_asof"])


def test_extra_features_refuse_a_wrong_row_count(frame, tmp_path):
    side_dir = tmp_path / "short"
    side_dir.mkdir()
    df = t1.load_split("train", frame, "i7")
    pd.DataFrame({
        "innings_id": df["innings_id"].to_numpy()[:-1],
        "ball_idx": df[t1.BALL_IDX_COL].to_numpy()[:-1],
        "a_N_asof": np.zeros(len(df) - 1, dtype=np.float32),
    }).to_parquet(side_dir / "train.parquet", index=False)
    with pytest.raises(RuntimeError, match="row-aligned to the frame"):
        t1.load_extra_features(side_dir, "train", df, ["a_N_asof"])


def test_extra_features_refuse_a_sequence_arm(monkeypatch, frame, cache,
                                              tmp_path):
    with pytest.raises(SystemExit):
        _run(monkeypatch, frame, cache, tmp_path / "seq", "--arm", "full",
             "--extra-features", str(tmp_path), "--extra-cols", "a")


def test_standardise_train_only_keeps_a_constant_column_finite():
    matrix = np.hstack([np.arange(10, dtype=np.float32).reshape(-1, 1),
                        np.full((10, 1), 3.0, dtype=np.float32)])
    mean, std = t1.standardise_train_only(matrix)
    assert std[1] == 1.0
    assert np.isfinite((matrix - mean) / std).all()


# ------------------------------------------------ rung 4b identity residual

def _base_probs(path: Path, y: np.ndarray, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(np.ones(6), size=len(y)).astype(np.float32)
    np.savez_compressed(path, probs=probs, y=y.astype(np.int8))
    return path


def test_identity_residual_starts_exactly_at_the_frozen_reference():
    model = t1.T1Model(t1.N_FEATS, DMODEL, LAYERS, HEADS,
                       arm="identity_residual", n_batters=5,
                       n_bowlers=4).eval()
    # No feature pathway at all: the rung is identity-only.
    assert not hasattr(model, "feat_proj")
    assert not hasattr(model, "token_mlp")
    assert model.bat_id_emb.num_embeddings == 5
    assert model.bowl_id_emb.embedding_dim == t1.IDENTITY_EMBED_DIM == 16
    base = torch.log(torch.full((2, 3, 6), 1.0 / 6))
    ids = torch.ones(2, 3, dtype=torch.long)
    logits, aux = model(torch.zeros(2, 3, t1.N_FEATS),
                        torch.zeros(2, 3, dtype=torch.long),
                        torch.zeros(2, 3, dtype=torch.bool),
                        batter=ids, bowler=ids, base_logp=base)
    # Zero-initialised readout: p == p_base EXACTLY at initialisation.
    assert torch.allclose(logits, base)
    assert torch.equal(aux["residual"], torch.zeros_like(base))


def test_identity_residual_needs_its_inputs():
    model = t1.T1Model(t1.N_FEATS, DMODEL, LAYERS, HEADS,
                       arm="identity_residual", n_batters=5, n_bowlers=4)
    feats = torch.zeros(1, 2, t1.N_FEATS)
    zeros = torch.zeros(1, 2, dtype=torch.long)
    pad = torch.zeros(1, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="id codes"):
        model(feats, zeros, pad, base_logp=torch.zeros(1, 2, 6))
    with pytest.raises(ValueError, match="base_logp"):
        model(feats, zeros, pad, batter=zeros, bowler=zeros)
    with pytest.raises(ValueError, match="vocabulary sizes"):
        t1.T1Model(t1.N_FEATS, DMODEL, LAYERS, HEADS,
                   arm="identity_residual")
    with pytest.raises(ValueError, match="token_mlp"):
        t1.T1Model(t1.N_FEATS, DMODEL, LAYERS, HEADS,
                   arm="identity_residual", n_batters=5, n_bowlers=4,
                   tier_embed=4)


def test_identity_vocab_is_train_only_with_unk_zero(frame):
    train = t1.load_split("train", frame, "i7")
    val = t1.load_split("validation", frame, "i7")
    vocabs = t1.identity_vocab(train)
    assert min(vocabs[0].values()) == 1  # 0 is reserved for UNK
    val = val.copy()
    val.loc[val.index[0], "batter_id"] = "never-seen-player"
    bat, bowl = t1.encode_identities(val, vocabs)
    assert bat[0] == 0
    assert len(bat) == len(val) and len(bowl) == len(val)


def test_identity_residual_refuses_a_misaligned_reference(frame, tmp_path):
    train = t1.load_split("train", frame, "i7")
    y = train["y"].to_numpy()
    path = _base_probs(tmp_path / "ref_train_probs.npz", y)
    logp, digest = t1.load_base_probs(tmp_path, "ref", "train", len(y), y)
    assert logp.shape == (len(y), 6) and len(digest) == 64
    # A reference whose stored labels are not this split's is refused.
    wrong = y.copy()
    wrong[5] = (wrong[5] + 1) % 6
    with pytest.raises(RuntimeError, match="not row-aligned"):
        t1.load_base_probs(tmp_path, "ref", "train", len(y), wrong)
    with pytest.raises(RuntimeError, match="rows of probabilities"):
        t1.load_base_probs(tmp_path, "ref", "train", len(y) - 1, y[:-1])
    assert path.exists()


def test_identity_residual_trains_end_to_end(monkeypatch, frame, cache,
                                             tmp_path):
    refs = tmp_path / "refs"
    refs.mkdir()
    for split in ("train", "validation"):
        df = t1.load_split(split, frame, "i7")
        _base_probs(refs / f"eb_ctx_{split}_probs.npz", df["y"].to_numpy())
    metrics = _run(monkeypatch, frame, cache, tmp_path / "4b",
                   "--arm", "identity_residual",
                   "--base-probs-dir", str(refs),
                   "--residual-lambda", "0.001",
                   "--max-steps", "4", "--eval-every", "2")
    block = metrics["arm_params"]["base_probs"]
    assert block["identity_embed_dim"] == 16
    assert block["id_dropout"] == 0.05
    assert block["residual_lambda"] == 0.001
    assert block["vocab"]["unk_index"] == 0
    assert block["vocab"]["fitted_on"] == "train rows only"
    assert set(block["sha256_by_split"]) == {"train", "validation"}
    assert metrics["training_schedule"]["total_steps"] == 4
    assert metrics["arm_params"]["wiring"] == "identity_residual"


def test_id_dropout_only_fires_in_training():
    model = t1.T1Model(t1.N_FEATS, DMODEL, LAYERS, HEADS,
                       arm="identity_residual", n_batters=6, n_bowlers=6)
    with torch.no_grad():
        model.residual_readout.weight.normal_()
        model.bat_id_emb.weight.normal_()
        model.bowl_id_emb.weight.normal_()
    feats = torch.zeros(1, 64, t1.N_FEATS)
    zeros = torch.zeros(1, 64, dtype=torch.long)
    pad = torch.zeros(1, 64, dtype=torch.bool)
    ids = torch.full((1, 64), 3, dtype=torch.long)
    base = torch.zeros(1, 64, 6)
    model.eval()
    torch.manual_seed(0)
    a, _ = model(feats, zeros, pad, batter=ids, bowler=ids, base_logp=base)
    torch.manual_seed(1)
    b, _ = model(feats, zeros, pad, batter=ids, bowler=ids, base_logp=base)
    assert torch.equal(a, b), "eval mode must be deterministic"
    model.train()
    torch.manual_seed(0)
    c, _ = model(feats, zeros, pad, batter=ids, bowler=ids, base_logp=base)
    # With p=0.05 over 128 draws, some row is dropped to UNK almost surely.
    assert not torch.equal(a, c)
