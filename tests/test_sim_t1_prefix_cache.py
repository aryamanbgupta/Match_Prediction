"""Pins the T1 prefix cache to the full causal re-forward, step by step.

`T1_SIM_PREFIX_CACHE=1` swaps predict-time inference from an O(L) full
re-forward per ball to an O(1)-per-ball incremental step over cached
per-layer keys/values. Under the causal mask the two are the same math, so
every step's logits must match the full forward's last position to float
epsilon — on the `full` arm (real outcome history) and the `no_history` arm
(BOS-masked history) alike. The cache stays opt-in because the certified PPC
raws were produced with the full re-forward.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]

from sim_t1 import BOS, TransformerT1SimModel  # noqa: E402
from transformer_t1 import T1Model  # noqa: E402

N_FEATS = 50
SEQ_LEN = 12


def _wrapper(arm: str) -> TransformerT1SimModel:
    torch.manual_seed(23)
    wrapper = TransformerT1SimModel.__new__(TransformerT1SimModel)
    wrapper.arm = arm
    wrapper.device = torch.device("cpu")
    wrapper.model = T1Model(N_FEATS, dmodel=32, layers=2, heads=4,
                            arm=arm).eval()
    wrapper.max_seq_len = 200
    wrapper._cache = wrapper._empty_cache()
    return wrapper


def _full_forward_last_logits(model, feats, prev):
    with torch.no_grad():
        ft = torch.tensor(np.stack(feats)[None, :, :])
        pt = torch.tensor(np.array(prev, dtype=np.int64)[None, :])
        pad = torch.zeros_like(pt, dtype=torch.bool)
        out = model(ft, pt, pad)
        logits = out[0] if isinstance(out, tuple) else out
        return logits[0, -1]


@pytest.mark.parametrize("arm", ["full", "no_history"])
def test_incremental_matches_full_forward_every_step(arm):
    wrapper = _wrapper(arm)
    rng = np.random.default_rng(7)
    feats, prev = [], []
    for step in range(SEQ_LEN):
        feats.append(rng.normal(size=N_FEATS).astype(np.float32))
        prev.append(BOS if step == 0 else int(rng.integers(0, 6)))
        with torch.no_grad():
            incremental = wrapper.model.head(
                wrapper._incremental_last_hidden(feats, prev))
        full = _full_forward_last_logits(wrapper.model, feats, prev)
        torch.testing.assert_close(incremental, full, atol=1e-5, rtol=1e-5)


def test_prefix_cache_desync_fails_closed():
    wrapper = _wrapper("full")
    rng = np.random.default_rng(11)
    feats = [rng.normal(size=N_FEATS).astype(np.float32)]
    prev = [BOS]
    wrapper._incremental_last_hidden(feats, prev)
    # Skipping a token (two appended, cache advanced once more only) must
    # refuse rather than fabricate the missing key/value entries.
    feats.extend([rng.normal(size=N_FEATS).astype(np.float32)] * 2)
    prev.extend([1, 2])
    with pytest.raises(RuntimeError, match="desynchronization"):
        wrapper._incremental_last_hidden(feats, prev)


def test_new_innings_cache_resets_positions():
    wrapper = _wrapper("full")
    rng = np.random.default_rng(13)
    feats = [rng.normal(size=N_FEATS).astype(np.float32) for _ in range(3)]
    prev = [BOS, 2, 4]
    for step in range(3):
        wrapper._incremental_last_hidden(feats[:step + 1], prev[:step + 1])
    # A fresh cache (new innings/state path replaces the dict wholesale)
    # must restart from position zero and agree with the full forward.
    wrapper._cache = wrapper._empty_cache()
    fresh = [rng.normal(size=N_FEATS).astype(np.float32)]
    with torch.no_grad():
        incremental = wrapper.model.head(
            wrapper._incremental_last_hidden(fresh, [BOS]))
    full = _full_forward_last_logits(wrapper.model, fresh, [BOS])
    torch.testing.assert_close(incremental, full, atol=1e-5, rtol=1e-5)
