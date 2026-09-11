"""Stage 2 recurrent arms (`lstm`, `xlstm`) — acceptance check D3.10.

Everything here runs on tiny synthetic tensors on the CPU: no frame, no
parquet, no `models/` read or write. What is pinned:

  * the fixed interface (`build_recurrent_model`, forward signature and
    output shapes, `aux == {}`, `n_parameters()`, `arm_params`);
  * the token is T1's token with no positional embedding
    (`feat_proj` + `out_emb(7, d)` only);
  * causality — perturbing rows strictly after a target leaves the target's
    logits bit-unchanged for both arms;
  * state reset / batch independence — one sequence in a batch cannot
    influence another, including when lengths differ;
  * the xLSTM module equals an independent **unstabilised** step-by-step
    reference implementation (plain python time loop, batch 1, exponential
    gates applied directly with no max-trick, float64) — Astra MUST-FIX 1:
    the stabilised recurrence must be algebraically equal to the unstabilised
    one, which the old `clamp(min=1)` denominator was not;
  * the stabiliser earns its keep: with large gate pre-activations the
    unstabilised reference overflows to inf/nan while the module stays finite;
  * parameter counts are positive and printed.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from sequence_track.recurrent_arms import (  # noqa: E402
    BOS,
    build_recurrent_model,
    simplifications,
)

ARMS = ("lstm", "xlstm")
N_FEATS = 7
DMODEL = 128


def _batch(b: int, length: int, lengths: list[int] | None = None,
           seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(b, length, N_FEATS, generator=g)
    prev_y = torch.randint(0, 6, (b, length), generator=g)
    prev_y[:, 0] = BOS
    pad = torch.zeros(b, length, dtype=torch.bool)
    if lengths is not None:
        for i, n in enumerate(lengths):
            feats[i, n:] = 0.0
            prev_y[i, n:] = BOS
            pad[i, n:] = True
    return feats, prev_y, pad


def _model(arm: str, dmodel: int = DMODEL, dropout: float = 0.1,
           seed: int = 29):
    torch.manual_seed(seed)
    model = build_recurrent_model(arm, N_FEATS, dmodel, dropout)
    model.eval()
    return model


# --------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------

@pytest.mark.parametrize("arm", ARMS)
def test_build_and_forward_shapes(arm):
    model = _model(arm)
    feats, prev_y, pad = _batch(3, 11, lengths=[11, 6, 2])
    logits, aux = model(feats, prev_y, pad)
    assert logits.shape == (3, 11, 6)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()
    assert aux == {}


def test_unknown_arm_refuses():
    with pytest.raises(ValueError, match="unknown recurrent arm"):
        build_recurrent_model("gru", N_FEATS, DMODEL, 0.1)


@pytest.mark.parametrize("arm", ARMS)
def test_token_is_t1s_token_without_positional_embedding(arm):
    model = _model(arm)
    assert model.token.feat_proj.in_features == N_FEATS
    assert model.token.feat_proj.out_features == DMODEL
    assert model.token.out_emb.num_embeddings == 7  # 6 classes + BOS
    assert model.token.out_emb.embedding_dim == DMODEL
    assert model.head.out_features == 6
    names = [n for n, _ in model.named_modules()]
    assert not any("pos" in n for n in names)
    required = {
        "wiring": "recurrent",
        "history_input": "innings_previous",
        "positional_embedding": False,
        "bias": "none",
    }
    for key, value in required.items():
        assert model.arm_params[key] == value
        assert type(model.arm_params[key]) is type(value)
    assert set(model.arm_params) - set(required) <= {"cell", "simplifications"}

    # The token is exactly feat_proj(feats) + out_emb(prev_y).
    feats, prev_y, _ = _batch(2, 4)
    expected = model.token.feat_proj(feats) + model.token.out_emb(prev_y)
    assert torch.allclose(model.token(feats, prev_y), expected, atol=0)


@pytest.mark.parametrize("arm", ARMS)
def test_parameter_counts(arm):
    model = _model(arm)
    n = model.n_parameters()
    print(f"[D3.10] {arm}: n_parameters = {n}")
    assert n > 0
    assert n == sum(p.numel() for p in model.parameters())


def test_simplifications_are_documented():
    assert len(simplifications()) >= 5
    assert any("conv" in s for s in simplifications())


# --------------------------------------------------------------------------
# Causality and state reset
# --------------------------------------------------------------------------

@pytest.mark.parametrize("arm", ARMS)
def test_causal_future_rows_do_not_change_target(arm):
    model = _model(arm)
    length, target = 9, 4
    feats, prev_y, pad = _batch(1, length, seed=3)
    with torch.no_grad():
        base, _ = model(feats, prev_y, pad)
        f2 = feats.clone()
        p2 = prev_y.clone()
        f2[:, target + 1:] = torch.randn_like(f2[:, target + 1:]) * 5.0
        p2[:, target + 1:] = (p2[:, target + 1:] + 3) % 6
        pert, _ = model(f2, p2, pad)
    assert torch.equal(base[:, :target + 1], pert[:, :target + 1])


@pytest.mark.parametrize("arm", ARMS)
def test_state_reset_sequences_are_independent(arm):
    model = _model(arm)
    feats, prev_y, pad = _batch(2, 8, lengths=[8, 3], seed=5)
    with torch.no_grad():
        base, _ = model(feats, prev_y, pad)
        f2, p2 = feats.clone(), prev_y.clone()
        f2[1] = torch.randn_like(f2[1]) * 4.0
        p2[1, 1:] = (p2[1, 1:] + 2) % 6
        pert, _ = model(f2, p2, pad)
    # Row 0's whole sequence is untouched by anything done to row 1.
    assert torch.equal(base[0], pert[0])

    # And a sequence alone in a batch gives the same logits as in company,
    # i.e. the initial state is zero and never carried across sequences.
    with torch.no_grad():
        solo, _ = model(feats[:1], prev_y[:1], pad[:1])
    assert torch.allclose(solo[0], base[0], atol=1e-6)


# --------------------------------------------------------------------------
# xLSTM reference implementation — Astra MUST-FIX 1
#
# This reference is deliberately UNSTABILISED: no `m_t`, no log-space shift,
# exponential gates applied directly as `exp(f_pre)` / `exp(i_pre)`, and the
# mLSTM denominator is the paper's plain `max(|n_t^T q_t|, 1)`. It is therefore
# an independent check of the production module rather than the same formula
# written twice: if the stabilised recurrence is a pure change of variables
# (C' = exp(-m) C, n' = exp(-m) n, denominator floor exp(-m)) the two agree
# exactly, and they disagree for any other floor — `clamp(min=1)` included.
# Gate magnitudes are kept small in the equality tests so the unstabilised
# form cannot overflow; `test_stabiliser_is_needed_for_large_gates` shows what
# happens when they are not.
# --------------------------------------------------------------------------

def _ref_swiglu(block, x):
    gate, value = block.ffn.up(x).chunk(2, dim=-1)
    return block.ffn.down(torch.nn.functional.silu(gate) * value)


def _ref_head_norm(block, h):
    return block.head_norm(h.reshape(1, -1)).reshape(1, -1)


def _ref_slstm_block(block, xs):
    """One pre-norm sLSTM block, batch 1, UNSTABILISED explicit time loop.

    ``c_t = exp(f̃_t) c_{t-1} + exp(ĩ_t) tanh(z_t)``, ``n_t`` likewise, and
    ``h_t = sigmoid(o_t) c_t / n_t`` — no stabiliser state, no clamp. The
    production cell's ratio ``c'/n'`` equals this ``c/n`` exactly because both
    states carry the same ``exp(-m_t)``.
    """
    d = block.cell.dmodel
    h = torch.zeros(d, dtype=xs[0].dtype)
    c = torch.zeros(d, dtype=xs[0].dtype)
    n = torch.zeros(d, dtype=xs[0].dtype)
    outs = []
    for x in xs:
        u = block.norm1(x)
        g = block.cell.w(u) + block.cell.r(h)
        z_pre, i_pre, f_pre, o_pre = torch.split(g, d)
        i_g = torch.exp(i_pre)
        f_g = torch.exp(f_pre)
        c = f_g * c + i_g * torch.tanh(z_pre)
        n = f_g * n + i_g
        h = torch.sigmoid(o_pre) * (c / n)
        y = x + block.out_proj(_ref_head_norm(block, h).reshape(d))
        outs.append(y + _ref_swiglu(block, block.norm2(y)))
    return outs


def _ref_mlstm_block(block, xs):
    """One pre-norm mLSTM block, batch 1, UNSTABILISED explicit time loop.

    ``C_t = exp(f̃_t) C_{t-1} + exp(ĩ_t) v_t k_t^T``, ``n_t`` likewise, and
    ``h̃_t = C_t q_t / max(|n_t^T q_t|, 1)`` straight out of Beck et al. eq.
    21-22, then the block's head-norm, the sigmoid output gate and the output
    projection. No ``m_t`` anywhere.
    """
    cell = block.cell
    nh, dh, d = cell.n_heads, cell.d_head, cell.dmodel
    dtype = xs[0].dtype
    big_c = torch.zeros(nh, dh, dh, dtype=dtype)
    n = torch.zeros(nh, dh, dtype=dtype)
    outs = []
    for x in xs:
        u = block.norm1(x)
        q = cell.q_proj(u).reshape(nh, dh)
        k = cell.k_proj(u).reshape(nh, dh) / math.sqrt(dh)
        v = cell.v_proj(u).reshape(nh, dh)
        i_g = torch.exp(cell.i_gate(u))
        f_g = torch.exp(cell.f_gate(u))
        hs = []
        for head in range(nh):
            big_c[head] = (f_g[head] * big_c[head]
                           + i_g[head] * torch.outer(v[head], k[head]))
            n[head] = f_g[head] * n[head] + i_g[head] * k[head]
            num = big_c[head] @ q[head]
            den = torch.maximum(torch.dot(n[head], q[head]).abs(),
                                torch.ones((), dtype=dtype))
            hs.append(num / den)
        h = _ref_head_norm(block, torch.cat(hs)).reshape(d)
        h = torch.sigmoid(block.o_gate(u)) * h  # restored output gate
        y = x + block.out_proj(h)
        outs.append(y + _ref_swiglu(block, block.norm2(y)))
    return outs


def _ref_logits(model, feats, prev_y, length):
    """The whole xLSTM arm through the unstabilised reference loops."""
    tokens = [model.token(feats, prev_y)[0, t] for t in range(length)]
    hs = _ref_mlstm_block(model.mlstm, _ref_slstm_block(model.slstm, tokens))
    return torch.stack([model.head(model.norm_out(h)) for h in hs])


def test_xlstm_matches_unstabilised_float64_reference():
    """D3.10 / Astra MUST-FIX 1: stabilised == unstabilised in float64.

    Small gate pre-activations (default init, 4 tokens) keep the unstabilised
    recurrence far from overflow, so the only thing this can catch is an
    algebraically wrong stabilisation — e.g. ``clamp(min=1)`` in place of
    ``max(|n^T q|, exp(-m))``.
    """
    model = _model("xlstm", dmodel=32, dropout=0.0, seed=11).double()
    feats, prev_y, pad = _batch(1, 4, seed=17)
    feats, prev_y = feats.double(), prev_y
    with torch.no_grad():
        got, _ = model(feats, prev_y, pad)
        want = _ref_logits(model, feats, prev_y, 4)
    assert got.dtype == torch.float64
    assert got.shape == (1, 4, 6)
    diff = float((got[0] - want).abs().max())
    assert diff < 1e-9, f"max abs diff {diff:.3e}"


def test_xlstm_matches_unstabilised_reference_at_production_width_float32():
    model = _model("xlstm", dmodel=DMODEL, dropout=0.0, seed=13)
    feats, prev_y, pad = _batch(1, 3, seed=19)
    with torch.no_grad():
        got, _ = model(feats, prev_y, pad)
        want = _ref_logits(model, feats, prev_y, 3)
    diff = float((got[0] - want).abs().max())
    assert diff < 1e-5, f"max abs diff {diff:.3e}"


def test_stabiliser_is_needed_for_large_gates():
    """The stabiliser earns its keep: unstabilised overflows, module does not.

    Astra MUST-FIX 1 — with large mLSTM gate pre-activations the raw
    ``exp(f̃)`` products blow past the float64 range, so the reference above
    returns inf/nan while the production stabilised cell stays finite.
    """
    model = _model("xlstm", dmodel=32, dropout=0.0, seed=11).double()
    with torch.no_grad():
        model.mlstm.cell.f_gate.bias.fill_(300.0)
        model.mlstm.cell.i_gate.bias.fill_(300.0)
    feats, prev_y, pad = _batch(1, 4, seed=17)
    feats = feats.double()
    with torch.no_grad():
        got, _ = model(feats, prev_y, pad)
        blown = _ref_logits(model, feats, prev_y, 4)
    assert torch.isfinite(got).all(), "stabilised cell overflowed"
    assert not torch.isfinite(blown).all(), (
        "unstabilised reference stayed finite; the overflow test is vacuous")


def test_gradients_flow_over_the_whole_sequence():
    """Full-sequence backprop: the first token's token weights get gradient."""
    for arm in ARMS:
        torch.manual_seed(7)
        model = build_recurrent_model(arm, N_FEATS, 32, 0.0)
        feats, prev_y, pad = _batch(2, 6, lengths=[6, 4], seed=23)
        logits, _ = model(feats, prev_y, pad)
        target = torch.zeros(2, 6, dtype=torch.long)
        loss = torch.nn.functional.cross_entropy(
            logits[~pad], target[~pad])
        loss.backward()
        grad = model.token.feat_proj.weight.grad
        assert grad is not None and torch.isfinite(grad).all()
        assert float(grad.abs().sum()) > 0.0
