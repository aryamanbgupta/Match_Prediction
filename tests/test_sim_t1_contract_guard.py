"""Pins the T1 serving contract guard (sequence track stage 1, D4).

The wrapper rebuilds its 50 features from live tracker state, so the frame a
checkpoint was trained on and the stats cache it is served from must declare
the same venue identity and delivery semantics. A retrained checkpoint says
so in `metrics.json`'s `training_contract`; a legacy one says nothing, and
then the old `config.data_dir` substring rule stands AND the serving cache
must carry no venue identity at all. Both paths refuse a provider that
cannot expose the cache `_meta`: unverifiable identity is not assumed benign.

Two tests below pin the EDGES of the guard rather than its centre, because
the guard's docstring used to overclaim (Astra round 1, SHOULD 1):
  * the recorded training-cache md5 is required to be present but is never
    compared against the cache actually being served — a checkpoint whose
    training cache has since been rebuilt still LOADS here. The md5 is
    compared against the live cache file by `pin_stage1.py --verify`, which
    `run_arm.py --config` re-runs before a registered arm launches;
  * `_meta.same_day_order_version` is required to be PRESENT, not to equal
    the contract's; a cache that declares no ordering version is refused.
"""
from __future__ import annotations

import json

import pytest
import torch

from embeddings_e1 import CTX_COLS, EB_BAT_COLS, EB_BOWL_COLS, VENUE_COLS
from sim_t1 import TransformerT1SimModel
from transformer_t1 import STATE_COLS, T1Model

# Derived here from the training-side column definitions rather than imported
# from sim_t1, so a reordering inside the wrapper fails this file instead of
# agreeing with it.
SERVING_ORDER = (
    EB_BAT_COLS + EB_BOWL_COLS + VENUE_COLS + CTX_COLS + STATE_COLS)

# Tiny but real: the checkpoint is loaded through the production
# T1Model/torch.load path, not a monkeypatch.
ARCH = {"dmodel": 16, "layers": 1, "heads": 2, "arm": "full"}

I7_META = {
    "schema_version": "4",
    "venue_alias_version": "venue_aliases_v1",
    "same_day_order_version": "date_then_match_id_lexicographic_v1",
}
# The legacy v3 cache declares neither key (checked against the shipped
# caches on 2026-09-11).
V3_META = {"schema_version": "4"}


class _FakeProvider:
    """Duck-types the audited replay surface plus the `_meta` passthrough."""

    def __init__(self, meta):
        self._meta = dict(meta)

    def get_cache_meta(self):
        return dict(self._meta)

    def get_t1_outcome_counts(self, *a, **k):
        return [0] * 6

    def get_t1_outcome_prior(self, *a, **k):
        return (1 / 6,) * 6


class _NoMetaProvider:
    """A live-count provider that cannot reach the cache `_meta`."""

    def get_t1_outcome_counts(self, *a, **k):
        return [0] * 6

    def get_t1_outcome_prior(self, *a, **k):
        return (1 / 6,) * 6


def _contract(**overrides):
    contract = {
        "contract_version": "seq_stage1_training_contract_v1",
        "delivery_semantics": "inclusive_total_runs_v1",
        "venue_alias_version": "venue_aliases_v1",
        "venue_alias_sha256": "853b32b0" + "0" * 56,
        "feature_names": list(SERVING_ORDER),
        "feature_names_sha256": "f" * 64,
        "stats_cache": {
            "role": "stats_cache_i7",
            "path": "models/player_stats_cache_i7.sqlite",
            "md5": "0" * 32,
            "venue_alias_version": "venue_aliases_v1",
            "same_day_order_version": "date_then_match_id_lexicographic_v1",
        },
    }
    contract.update(overrides)
    return contract


def _checkpoint(tmp_path, *, data_dir, contract=None):
    mdir = tmp_path / "ckpt"
    mdir.mkdir(parents=True)
    torch.manual_seed(11)
    model = T1Model(
        len(SERVING_ORDER), dmodel=ARCH["dmodel"], layers=ARCH["layers"],
        heads=ARCH["heads"], arm=ARCH["arm"])
    torch.save(model.state_dict(), mdir / "model.pt")
    metrics = {"config": {"data_dir": data_dir, **ARCH}}
    if contract is not None:
        metrics["training_contract"] = contract
    (mdir / "metrics.json").write_text(json.dumps(metrics))
    return mdir


def _load(monkeypatch, mdir, provider):
    monkeypatch.setenv("T1_SIM_MODEL_DIR", str(mdir))
    monkeypatch.setenv("T1_SIM_DEVICE", "cpu")
    monkeypatch.delenv("T1_EXTRAS_GRAFT_PATH", raising=False)
    monkeypatch.delenv("T1_SIM_PREFIX_CACHE", raising=False)
    return TransformerT1SimModel(
        stats_provider=provider, player_metadata=object())


# (i)
def test_i7_contract_on_i7_cache_loads(tmp_path, monkeypatch):
    mdir = _checkpoint(
        tmp_path, data_dir="data/xgb_data_i7", contract=_contract())
    model = _load(monkeypatch, mdir, _FakeProvider(I7_META))
    assert model.arm == "full"
    assert model.delivery_semantics == "inclusive_total_runs_v1"


# (ii)
def test_uncontracted_v3_on_v3_cache_loads(tmp_path, monkeypatch):
    """The stage 0 configuration: legacy checkpoint, legacy cache."""
    mdir = _checkpoint(tmp_path, data_dir="data/xgb_data_v3")
    model = _load(monkeypatch, mdir, _FakeProvider(V3_META))
    assert model.arm == "full"


# (iii)
def test_uncontracted_v3_on_i7_cache_is_refused(tmp_path, monkeypatch):
    mdir = _checkpoint(tmp_path, data_dir="data/xgb_data_v3")
    with pytest.raises(RuntimeError, match="venue_alias_version"):
        _load(monkeypatch, mdir, _FakeProvider(I7_META))


# (iv)
def test_i7_contract_on_cache_without_alias_is_refused(tmp_path, monkeypatch):
    mdir = _checkpoint(
        tmp_path, data_dir="data/xgb_data_i7", contract=_contract())
    with pytest.raises(RuntimeError, match="venue_alias_version mismatch"):
        _load(monkeypatch, mdir, _FakeProvider(V3_META))


# (v)
def test_foreign_delivery_semantics_is_refused(tmp_path, monkeypatch):
    mdir = _checkpoint(
        tmp_path, data_dir="data/xgb_data_i5",
        contract=_contract(delivery_semantics="legal_off_bat_v1"))
    with pytest.raises(RuntimeError, match="delivery_semantics"):
        _load(monkeypatch, mdir, _FakeProvider(I7_META))


# (vi)
def test_contract_without_venue_alias_version_is_refused(
        tmp_path, monkeypatch):
    mdir = _checkpoint(
        tmp_path, data_dir="data/xgb_data_i7",
        contract=_contract(venue_alias_version=None))
    with pytest.raises(RuntimeError, match="venue_alias_version"):
        _load(monkeypatch, mdir, _FakeProvider(I7_META))


# (vii)
def test_permuted_feature_list_is_refused(tmp_path, monkeypatch):
    permuted = list(SERVING_ORDER)
    permuted[0], permuted[1] = permuted[1], permuted[0]
    mdir = _checkpoint(
        tmp_path, data_dir="data/xgb_data_i7",
        contract=_contract(feature_names=permuted))
    with pytest.raises(RuntimeError, match="feature_names"):
        _load(monkeypatch, mdir, _FakeProvider(I7_META))


# (viii)
def test_provider_without_meta_is_refused(tmp_path, monkeypatch):
    contracted = _checkpoint(
        tmp_path / "a", data_dir="data/xgb_data_i7", contract=_contract())
    with pytest.raises(RuntimeError, match="get_cache_meta"):
        _load(monkeypatch, contracted, _NoMetaProvider())
    # The legacy path is not an escape hatch from the same requirement.
    legacy = _checkpoint(tmp_path / "b", data_dir="data/xgb_data_v3")
    with pytest.raises(RuntimeError, match="get_cache_meta"):
        _load(monkeypatch, legacy, _NoMetaProvider())


# --------------------------------------- Astra SHOULD 1: the exact boundary

def test_a_stale_training_cache_md5_still_loads(tmp_path, monkeypatch):
    """Documents what the guard does NOT do.

    The contract names a cache md5 that matches nothing on this machine — as
    it would after the i7 cache was rebuilt. The wrapper is handed a
    provider, not a file, so it cannot hash the served cache and does not
    try: the alias versions agree, so the checkpoint loads. Catching that
    drift is `pin_stage1.py --verify`'s job (it compares this recorded md5
    against the live cache file), re-run by `run_arm.py --config`.
    """
    stale = dict(_contract()["stats_cache"], md5="d" * 32)
    assert stale["md5"] != _contract()["stats_cache"]["md5"]
    mdir = _checkpoint(tmp_path, data_dir="data/xgb_data_i7",
                       contract=_contract(stats_cache=stale))
    model = _load(monkeypatch, mdir, _FakeProvider(I7_META))
    assert model.arm == "full"
    # And the drift is not silently repaired into the loaded object either.
    assert model.delivery_semantics == "inclusive_total_runs_v1"


def test_serving_meta_without_same_day_order_version_is_refused(tmp_path,
                                                                monkeypatch):
    """Presence, not equality — but presence is mandatory. Same-day ordering
    is a versioned data contract (CLAUDE.md invariant 5); a cache that
    declares no ordering version cannot be shown to honour it."""
    meta = {k: v for k, v in I7_META.items()
            if k != "same_day_order_version"}
    mdir = _checkpoint(tmp_path, data_dir="data/xgb_data_i7",
                       contract=_contract())
    with pytest.raises(RuntimeError, match="same_day_order_version"):
        _load(monkeypatch, mdir, _FakeProvider(meta))


def test_a_differing_same_day_order_version_is_not_compared(tmp_path,
                                                            monkeypatch):
    """The other half of the same boundary: the serving cache's ordering
    version is never checked against the contract's, only required to be
    there. Pinned so a future tightening is a deliberate change."""
    meta = dict(I7_META, same_day_order_version="some_other_ordering_v9")
    mdir = _checkpoint(tmp_path, data_dir="data/xgb_data_i7",
                       contract=_contract())
    assert _load(monkeypatch, mdir, _FakeProvider(meta)).arm == "full"


# Beyond the eight cases of acceptance check 4.5: the non-null stats-cache
# md5 half of check 4.1 (a checkpoint trained without --stats-cache-role).
@pytest.mark.parametrize("stats_cache", [None, {"role": "stats_cache_i7",
                                                "md5": None}])
def test_contract_without_stats_cache_md5_is_refused(
        tmp_path, monkeypatch, stats_cache):
    mdir = _checkpoint(
        tmp_path, data_dir="data/xgb_data_i7",
        contract=_contract(stats_cache=stats_cache))
    with pytest.raises(RuntimeError, match="stats_cache.md5"):
        _load(monkeypatch, mdir, _FakeProvider(I7_META))
