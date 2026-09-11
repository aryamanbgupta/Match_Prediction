#!/usr/bin/env python3
"""Recurrent arms (13 `lstm`, 14 `xlstm`) of the stage 2 arm register.

Both arms consume exactly T1's token — ``feat_proj(feats) + out_emb(prev_y)``
with **no positional embedding** — and emit per-row logits over the six ball
outcome classes ``{0:dot, 1:one, 2:two, 3:four, 4:six, 5:wicket}``. They are
causal by construction (a recurrence only ever sees the past) and each innings
is one sequence with a zeroed initial state, so state is reset per innings.

Interface (fixed; `scripts/transformer_t1.py` codes against it)::

    model = build_recurrent_model(arm, n_feats, dmodel, dropout)
    logits, aux = model(feats, prev_y, pad_mask)

with ``feats (B, L, F) float``, ``prev_y (B, L) long`` (BOS = 6 at row 0 of
each innings), ``pad_mask (B, L) bool`` (True = pad), ``logits (B, L, 6)`` and
``aux = {}``.

Padding convention
------------------
`collate` in `transformer_t1.py` writes every real row of an innings at the
front of the row block and pads at the **end**. A recurrent forward pass at a
real position therefore never reads a padded position, so the modules run a
plain padded forward and the caller discards padded outputs (the trainer
already masks the loss with ``pad_mask``). `pack_padded_sequence` would give
bit-identical results at real positions; the plain forward is used instead
because it is one code path for both arms, keeps the xLSTM time loop simple,
and avoids packed-sequence support gaps on the MPS backend. Padded rows carry
zero features and BOS history, so they cannot corrupt any real row's state.

xLSTM
-----
`XLSTMModel` is a plain-torch transcription of Beck et al. 2024 ("xLSTM:
Extended Long Short-Term Memory"): one pre-norm residual **sLSTM** block
followed by one pre-norm residual **mLSTM** block at ``dmodel`` width.

* sLSTM cell: scalar memory ``c_t`` per hidden unit, exponential input and
  forget gates, the log-space stabiliser state ``m_t = max(f̃_t + m_{t-1},
  ĩ_t)``, the paper's normaliser state ``n_t``, and ``h_t = o_t ⊙ c_t / n_t``.
* mLSTM cell: matrix memory ``C_t`` of shape ``(d_head, d_head)`` per head
  (4 heads of 32 at ``dmodel = 128``), query/key/value projections with the
  paper's ``1/sqrt(d_head)`` key scaling, per-head scalar exponential input and
  forget gates with the same stabiliser, normaliser state ``n_t``, and the
  stabilised denominator ``max(|n_t^T q_t|, exp(-m_t))`` (Astra MUST-FIX 1).
  ``C_t`` and ``n_t`` are carried in the log-space-shifted form
  ``C'_t = exp(-m_t) C_t``, ``n'_t = exp(-m_t) n_t``; since
  ``h_t = C_t q_t / max(|n_t^T q_t|, 1)`` and both numerator and the first
  denominator branch carry the same ``exp(m_t)``, the shifted recurrence is
  algebraically identical to the unstabilised one only with ``exp(-m_t)`` (not
  ``1``) as the floor. The paper's sigmoid **output gate** on the (head-normed)
  cell output is present (Astra SHOULD 4).

Deliberate simplifications (all recorded in `arm_params` /
`simplifications()`):

1. No causal conv1d pre-block on either cell (paper: conv4 before the sLSTM
   cell and before the mLSTM q/k projections).
2. No block-diagonal recurrent or input projections in the sLSTM cell: it runs
   as a single head with dense ``(d, d)`` matrices.
3. No up/down projection around the mLSTM cell. The paper's pre-up-projection
   block widens by 2x; here the cell runs at ``dmodel`` so that "4 heads of 32"
   holds exactly at ``dmodel = 128``. Each block instead carries a SwiGLU
   feed-forward sublayer (hidden ``4*dmodel//3``, the paper's sLSTM-block
   projection factor) to keep per-block capacity. The mLSTM output gate is
   therefore driven by the block's pre-normed input rather than by an
   up-projected branch, but it is applied where the paper applies it: on the
   head-normed cell output, before the block's output projection.
4. No learnable skip from a conv output inside the mLSTM block; the block's
   own pre-norm residual is the only skip.
5. Both cells use the exponential form of the forget gate (the paper permits
   sigmoid or exponential). Cell states start at zero and ``m_0 = 0``.
6. The sLSTM normaliser denominator is clamped to ``1e-6`` from below as a
   numerical guard. It is provably >= 1 once the input-gate branch of the
   stabiliser max has been taken, so the clamp only ever touches a leading
   prefix with a vanishing input gate.

Backpropagation is over the full sequence (no truncation): the time loop is a
plain autograd graph.
"""

from __future__ import annotations

import math

import torch
from torch import nn

BOS = 6  # outcome-history vocab: 6 classes + BOS, matching transformer_t1.BOS
N_CLASSES = 6
RECURRENT_ARMS = ("lstm", "xlstm")

_ARM_PARAMS = {
    "wiring": "recurrent",
    "history_input": "innings_previous",
    "positional_embedding": False,
    "bias": "none",
}

_SIMPLIFICATIONS = (
    "no causal conv1d pre-block on either cell",
    "sLSTM cell is single-head with dense (d, d) input and recurrent matrices "
    "(no block-diagonal projections)",
    "no up/down projection around the mLSTM cell; it runs at dmodel so 4 heads "
    "of 32 holds at dmodel=128, and each block carries a SwiGLU feed-forward "
    "sublayer (hidden 4*dmodel//3) instead; the mLSTM output gate is driven by "
    "the block's pre-normed input rather than by an up-projected branch",
    "no learnable conv skip inside the mLSTM block",
    "exponential forget gate in both cells (paper allows sigmoid or exp); "
    "zero initial cell states and m_0 = 0",
    "sLSTM normaliser denominator clamped below at 1e-6 as a numerical guard",
)


def simplifications() -> tuple[str, ...]:
    """The documented departures from Beck et al. 2024 in `XLSTMModel`."""
    return _SIMPLIFICATIONS


class TokenEmbedding(nn.Module):
    """T1's token, minus the positional embedding.

    ``feat_proj(feats) + out_emb(prev_y)``, identical in shape and semantics to
    the first two terms of `T1Model.forward`.
    """

    def __init__(self, n_feats: int, dmodel: int):
        super().__init__()
        self.feat_proj = nn.Linear(n_feats, dmodel)
        self.out_emb = nn.Embedding(BOS + 1, dmodel)  # 6 classes + BOS

    def forward(self, feats: torch.Tensor,
                prev_y: torch.Tensor) -> torch.Tensor:
        return self.feat_proj(feats) + self.out_emb(prev_y)


class SLSTMCell(nn.Module):
    """sLSTM cell (Beck et al. 2024 eq. 1-7), scalar memory, single head.

    Exponential input and forget gates, log-space stabiliser ``m_t``, the
    paper's normaliser ``n_t``, ``h_t = o_t * (c_t / n_t)``.
    """

    N_DEN_FLOOR = 1e-6

    def __init__(self, dmodel: int):
        super().__init__()
        self.dmodel = dmodel
        # Input projections (z cell input, i input gate, f forget gate,
        # o output gate) and the recurrent projections on h_{t-1}.
        self.w = nn.Linear(dmodel, 4 * dmodel, bias=True)
        self.r = nn.Linear(dmodel, 4 * dmodel, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, length, _ = x.shape
        d = self.dmodel
        wx = self.w(x)  # (B, L, 4d), time-independent part precomputed
        zeros = x.new_zeros(b, d)
        h = zeros
        c = zeros
        n = zeros
        m = zeros
        out = []
        for t in range(length):
            g = wx[:, t] + self.r(h)
            z_pre, i_pre, f_pre, o_pre = torch.split(g, d, dim=-1)
            m_new = torch.maximum(f_pre + m, i_pre)
            i_g = torch.exp(i_pre - m_new)
            f_g = torch.exp(f_pre + m - m_new)
            z = torch.tanh(z_pre)
            c = f_g * c + i_g * z
            n = f_g * n + i_g
            h = torch.sigmoid(o_pre) * (c / n.clamp(min=self.N_DEN_FLOOR))
            m = m_new
            out.append(h)
        return torch.stack(out, dim=1)


class MLSTMCell(nn.Module):
    """mLSTM cell (Beck et al. 2024 eq. 19-22), matrix memory, multi-head.

    Per head: ``C_t (d_head, d_head)``, ``n_t (d_head)``, exponential gating
    with the shared stabiliser ``m_t``, and the stabilised denominator
    ``max(|n_t^T q_t|, exp(-m_t))`` (Astra MUST-FIX 1).

    ``big_c`` and ``n`` hold the shifted states ``exp(-m_t) C_t`` and
    ``exp(-m_t) n_t``, so the unstabilised ``C_t q_t / max(|n_t^T q_t|, 1)``
    becomes ``C'_t q_t / max(|n'^T_t q_t|, exp(-m_t))``: the ``exp(m_t)`` common
    to the numerator and the first denominator branch cancels, and the constant
    branch ``1`` must be shifted by the same factor. ``clamp(min=1.0)`` would
    instead change the recurrence.
    """

    def __init__(self, dmodel: int, n_heads: int = 4):
        super().__init__()
        if dmodel % n_heads:
            raise ValueError(
                f"dmodel {dmodel} is not divisible by n_heads {n_heads}")
        self.dmodel = dmodel
        self.n_heads = n_heads
        self.d_head = dmodel // n_heads
        self.q_proj = nn.Linear(dmodel, dmodel, bias=True)
        self.k_proj = nn.Linear(dmodel, dmodel, bias=True)
        self.v_proj = nn.Linear(dmodel, dmodel, bias=True)
        self.i_gate = nn.Linear(dmodel, n_heads, bias=True)
        self.f_gate = nn.Linear(dmodel, n_heads, bias=True)

    def _heads(self, t: torch.Tensor) -> torch.Tensor:
        b, length, _ = t.shape
        return t.view(b, length, self.n_heads, self.d_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, length, _ = x.shape
        nh, dh = self.n_heads, self.d_head
        q = self._heads(self.q_proj(x))
        k = self._heads(self.k_proj(x)) / math.sqrt(dh)
        v = self._heads(self.v_proj(x))
        i_pre = self.i_gate(x)  # (B, L, H)
        f_pre = self.f_gate(x)  # (B, L, H)
        big_c = x.new_zeros(b, nh, dh, dh)
        n = x.new_zeros(b, nh, dh)
        m = x.new_zeros(b, nh)
        out = []
        for t in range(length):
            m_new = torch.maximum(f_pre[:, t] + m, i_pre[:, t])
            i_g = torch.exp(i_pre[:, t] - m_new)  # (B, H)
            f_g = torch.exp(f_pre[:, t] + m - m_new)  # (B, H)
            kt = k[:, t]  # (B, H, dh)
            vt = v[:, t]  # (B, H, dh)
            qt = q[:, t]  # (B, H, dh)
            outer = vt.unsqueeze(-1) * kt.unsqueeze(-2)  # (B, H, dh, dh)
            big_c = f_g[..., None, None] * big_c + i_g[..., None, None] * outer
            n = f_g[..., None] * n + i_g[..., None] * kt
            num = torch.einsum("bhij,bhj->bhi", big_c, qt)
            # Astra MUST-FIX 1: max(|n^T q|, exp(-m_new)), not clamp(min=1).
            den = torch.maximum(
                torch.einsum("bhi,bhi->bh", n, qt).abs(), torch.exp(-m_new))
            out.append((num / den.unsqueeze(-1)).reshape(b, self.dmodel))
            m = m_new
        return torch.stack(out, dim=1)


class SwiGLU(nn.Module):
    """Gated feed-forward sublayer, hidden ``4*dmodel//3`` as in the paper."""

    def __init__(self, dmodel: int):
        super().__init__()
        hidden = 4 * dmodel // 3
        self.up = nn.Linear(dmodel, 2 * hidden, bias=True)
        self.down = nn.Linear(hidden, dmodel, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.up(x).chunk(2, dim=-1)
        return self.down(torch.nn.functional.silu(gate) * value)


class XLSTMBlock(nn.Module):
    """Pre-norm residual block: norm -> cell -> head-norm -> proj, then SwiGLU.

    The head-norm is the paper's multi-head GroupNorm after the cell (a
    degenerate single-group LayerNorm for the sLSTM cell).

    Astra SHOULD 4 — the mLSTM block carries the paper's sigmoid output gate
    (`xlstm/blocks/mlstm/layer.py`): ``sigmoid(W_o u) * head_norm(cell(u))``
    with ``u`` the pre-normed block input. It lives here, not in the cell,
    because the paper applies it *after* the multi-head norm; applying it
    before would let the GroupNorm divide most of a per-head gate back out.
    The sLSTM cell has its own output gate inside the cell (eq. 4).
    """

    def __init__(self, dmodel: int, dropout: float, kind: str,
                 n_heads: int = 4):
        super().__init__()
        if kind not in {"slstm", "mlstm"}:
            raise ValueError(f"unknown xLSTM block kind: {kind}")
        self.kind = kind
        self.norm1 = nn.LayerNorm(dmodel)
        if kind == "slstm":
            self.cell: nn.Module = SLSTMCell(dmodel)
            self.head_norm = nn.GroupNorm(1, dmodel)
            self.o_gate: nn.Module | None = None
        else:
            self.cell = MLSTMCell(dmodel, n_heads=n_heads)
            self.head_norm = nn.GroupNorm(n_heads, dmodel)
            # Astra SHOULD 4 — restored mLSTM output gate.
            self.o_gate = nn.Linear(dmodel, dmodel, bias=True)
        self.out_proj = nn.Linear(dmodel, dmodel, bias=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dmodel)
        self.ffn = SwiGLU(dmodel)
        self.drop2 = nn.Dropout(dropout)

    def _head_norm(self, h: torch.Tensor) -> torch.Tensor:
        b, length, d = h.shape
        return self.head_norm(h.reshape(b * length, d)).reshape(b, length, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.norm1(x)
        h = self._head_norm(self.cell(u))
        if self.o_gate is not None:  # Astra SHOULD 4 — mLSTM output gate
            h = torch.sigmoid(self.o_gate(u)) * h
        x = x + self.drop1(self.out_proj(h))
        return x + self.drop2(self.ffn(self.norm2(x)))


class _RecurrentArm(nn.Module):
    """Shared token embedding, head, and reporting for both recurrent arms."""

    def __init__(self, arm: str, n_feats: int, dmodel: int):
        super().__init__()
        self.arm = arm
        self.dmodel = dmodel
        self.token = TokenEmbedding(n_feats, dmodel)
        self.head = nn.Linear(dmodel, N_CLASSES)
        self.arm_params = dict(_ARM_PARAMS)

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))


class LSTMModel(_RecurrentArm):
    """Arm 13: 2-layer `nn.LSTM` at hidden ``dmodel`` over T1's token."""

    def __init__(self, n_feats: int, dmodel: int, dropout: float,
                 layers: int = 2):
        super().__init__("lstm", n_feats, dmodel)
        self.layers = layers
        self.rnn = nn.LSTM(dmodel, dmodel, num_layers=layers,
                           dropout=dropout, batch_first=True)
        self.arm_params["cell"] = f"nn.LSTM(hidden={dmodel}, layers={layers})"

    def forward(self, feats: torch.Tensor, prev_y: torch.Tensor,
                pad_mask: torch.Tensor) -> tuple[torch.Tensor, dict]:
        del pad_mask  # padding is trailing; see module docstring
        x = self.token(feats, prev_y)
        h, _ = self.rnn(x)  # zero initial state == state reset per innings
        return self.head(h), {}


class XLSTMModel(_RecurrentArm):
    """Arm 14: sLSTM block then mLSTM block, plain torch, width ``dmodel``."""

    def __init__(self, n_feats: int, dmodel: int, dropout: float,
                 n_heads: int = 4):
        super().__init__("xlstm", n_feats, dmodel)
        self.slstm = XLSTMBlock(dmodel, dropout, "slstm")
        self.mlstm = XLSTMBlock(dmodel, dropout, "mlstm", n_heads=n_heads)
        self.norm_out = nn.LayerNorm(dmodel)
        self.arm_params["cell"] = (
            f"sLSTM block -> mLSTM block (dmodel={dmodel}, "
            f"mlstm_heads={n_heads}, d_head={dmodel // n_heads}, "
            f"mlstm_output_gate=True, "
            f"mlstm_denominator=max(|n^T q|, exp(-m)))")
        self.arm_params["simplifications"] = list(_SIMPLIFICATIONS)

    def forward(self, feats: torch.Tensor, prev_y: torch.Tensor,
                pad_mask: torch.Tensor) -> tuple[torch.Tensor, dict]:
        del pad_mask  # padding is trailing; see module docstring
        x = self.token(feats, prev_y)
        x = self.mlstm(self.slstm(x))
        return self.head(self.norm_out(x)), {}


def build_recurrent_model(arm: str, n_feats: int, dmodel: int,
                          dropout: float) -> nn.Module:
    """Build one recurrent stage 2 arm.

    Args:
        arm: ``"lstm"`` or ``"xlstm"``.
        n_feats: width of the per-row feature vector.
        dmodel: model width (128 in the registered stage 2 training block).
        dropout: dropout probability (0.1 in the registered block).
    """
    if arm == "lstm":
        return LSTMModel(n_feats, dmodel, dropout)
    if arm == "xlstm":
        return XLSTMModel(n_feats, dmodel, dropout)
    raise ValueError(
        f"unknown recurrent arm: {arm!r} (expected one of {RECURRENT_ARMS})")
