"""Per-arm access, parity and CLI-contract tests (stage 2, D3.3/D3.7/D3.9).

Every test here runs on the CPU against a hand-built innings or a tiny
synthetic frame in `tmp_path`. Nothing reads a repository frame, an eval kit,
a stats cache or anything under `models/`, and nothing trains for longer than
one epoch on ~100 rows.

The access tests are the point of the file: each arm is BUILT, run in eval
mode (dropout off), then run again with some part of the input perturbed, and
the target row's logits are compared. An arm that can see something it must
not see fails here rather than in a log loss nobody can attribute.

What "outside the dependency set" means (handoff § 3.1). A relay-free arm
reads an earlier row j < i as its OWN-OUTCOME token e_j = feat_proj(feat_j) +
own_out_emb(y_j) + pos_j, and reads itself as e_i^self = feat_proj(feat_i) +
hist(i) + pos_i. So an earlier row contributes its own outcome, never its
history input, and the dependency set is exactly the acceptance file's

    S(i) = attention_set(i) union {history-source rows of i}

(row i-1 for innings-previous history; the last same-batter and last
same-bowler rows for participant-aligned history). `t1.dependency_set`
computes it and both these tests and
`scripts/sequence_track/ownership_dependency_test.py` perturb outside it.
There is no recursion: an outcome is a raw label, not a function of any other
row. The target's own outcome y_i is NOT in the readable part of S(i) — the
own-outcome key at position i is masked off the diagonal — and one test per
relay-free arm asserts that directly.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import transformer_t1 as t1  # noqa: E402
from test_transformer_t1_contract import (CACHE_ARGS,  # noqa: E402
                                          run_trainer, write_cache,
                                          write_frame)

DMODEL, LAYERS, HEADS, SEED = 16, 2, 4, 19

# A 14-ball innings: batters in pairs with strike rotation, bowlers in
# three-ball spells, and the first bowler returning at row 9.
BATTER = np.array(["A", "A", "B", "B", "C", "C", "A", "A", "B", "B", "C",
                   "A", "A", "B"])
BOWLER = np.array(["P", "P", "P", "Q", "Q", "Q", "R", "R", "R", "P", "P",
                   "P", "Q", "Q"])
Y = np.array([0, 1, 0, 2, 5, 1, 3, 0, 4, 1, 0, 2, 0, 1], dtype=np.int64)
L = len(Y)
INNINGS = [np.arange(L)]
FEATS = np.random.default_rng(5).standard_normal((L, t1.N_FEATS)).astype(
    np.float32)
UNIFORM_LOGP = np.full((L, 6), np.log(1.0 / 6.0), dtype=np.float32)


# --------------------------------------------------------------- helpers

def _codes(values: np.ndarray) -> np.ndarray:
    return np.unique(values, return_inverse=True)[1].astype(np.int64)


def build(arm: str, k=None, seed: int = SEED) -> t1.T1Model:
    torch.manual_seed(seed)
    return t1.T1Model(t1.N_FEATS, DMODEL, LAYERS, HEADS, arm=arm,
                      k=k).eval()


def logits(arm: str, k=None, feats=FEATS, y=Y, batter=BATTER, bowler=BOWLER,
           base_logp=None, model: t1.T1Model | None = None) -> torch.Tensor:
    """The (L, 6) logits of one innings, with every derived input rebuilt."""
    model = model if model is not None else build(arm, k)
    prev_bat, prev_bowl = t1.aligned_history(y, batter, bowler, INNINGS)
    if arm in t1.ARMS_NEEDING_BASE_LOGITS and base_logp is None:
        base_logp = UNIFORM_LOGP
    batch = t1.collate(INNINGS, feats, y, "cpu", batter=_codes(batter),
                       bowler=_codes(bowler), prev_bat=prev_bat,
                       prev_bowl=prev_bowl, base_logp=base_logp)
    with torch.no_grad():
        out, _ = model(batch.feats, batch.prev_y, batch.pad,
                       prev_bat=batch.prev_bat, prev_bowl=batch.prev_bowl,
                       batter=batch.batter, bowler=batch.bowler,
                       base_logp=batch.base_logp, own_y=batch.y)
    return out[0]


def attention_set(arm: str, k, target: int) -> set[int]:
    return t1.attention_set(arm, k, target, _codes(BATTER), _codes(BOWLER), L)


def aligned_sources(target: int) -> set[int]:
    """The rows whose outcomes feed `target`'s participant-aligned inputs."""
    return t1.history_source_rows("aligned_hist", target, _codes(BATTER),
                                  _codes(BOWLER))


def outcome_dependency(arm: str, k, target: int) -> set[int]:
    """S(i) — the arm register's dependency set, from the shared helper.

    Kept as a thin alias so the tests read like the acceptance file. The
    implementation lives in `transformer_t1` so the D7 script uses the same
    function (D7.1).
    """
    return t1.dependency_set(arm, k, target, _codes(BATTER), _codes(BOWLER), L)


def perturb(rows, feats=True, outcomes=True):
    """Copies of FEATS / Y with `rows` replaced by different values."""
    new_feats, new_y = FEATS.copy(), Y.copy()
    rows = sorted(rows)
    if feats:
        noise = np.random.default_rng(77).standard_normal(
            (len(rows), t1.N_FEATS)).astype(np.float32)
        new_feats[rows] = noise * 3.0
    if outcomes:
        new_y[rows] = (new_y[rows] + 3) % 6
    return new_feats, new_y


# ------------------------------------------------- D3.7 access / invariance

def test_mlp_sees_no_outcome_at_all():
    other = (Y + 2) % 6
    torch.testing.assert_close(logits("mlp"), logits("mlp", y=other))


@pytest.mark.parametrize("arm", ["full", "fixed_decay", "fox"])
def test_causal_arms_never_read_a_future_row(arm):
    target = 7
    feats, y = FEATS.copy(), Y.copy()
    feats[target + 1:] *= -5.0
    y[target + 1:] = (y[target + 1:] + 3) % 6
    before, after = logits(arm), logits(arm, feats=feats, y=y)
    torch.testing.assert_close(before[:target + 1], after[:target + 1])
    assert not torch.allclose(before[target + 1:], after[target + 1:])


@pytest.mark.parametrize("arm", ["full", "fixed_decay", "fox"])
def test_causal_arms_are_blind_to_identity(arm):
    # These arms never receive batter/bowler ids; relabelling must be a no-op.
    swapped_bat = np.array(["Z" if v == "A" else v for v in BATTER])
    swapped_bowl = np.array(["Y" if v == "P" else v for v in BOWLER])
    torch.testing.assert_close(
        logits(arm),
        logits(arm, batter=swapped_bat, bowler=swapped_bowl))


@pytest.mark.parametrize("arm,k", [("aligned_hist", None),
                                   ("aligned_hist_rf", None)])
def test_aligned_arms_read_the_same_batters_earlier_outcome(arm, k):
    target = 12  # batter A, whose last earlier row is 11
    assert 11 in aligned_sources(target)
    y = Y.copy()
    y[11] = (y[11] + 3) % 6
    before, after = logits(arm, k), logits(arm, k, y=y)
    assert not torch.allclose(before[target], after[target]), (
        f"{arm} ignored the same batter's earlier outcome")


@pytest.mark.parametrize("arm,k", [("aligned_hist", None),
                                   ("aligned_hist_rf", None)])
def test_aligned_arms_ignore_alignment_preserving_relabelling(arm, k):
    # A bijection on the ids leaves every aligned input identical, so the
    # arms must be numerically unchanged: they read alignment, not identity.
    bat_map = {"A": "K", "B": "L", "C": "M"}
    bowl_map = {"P": "X", "Q": "Y", "R": "Z"}
    relabelled_bat = np.array([bat_map[v] for v in BATTER])
    relabelled_bowl = np.array([bowl_map[v] for v in BOWLER])
    base = t1.aligned_history(Y, BATTER, BOWLER, INNINGS)
    relabelled = t1.aligned_history(Y, relabelled_bat, relabelled_bowl,
                                    INNINGS)
    np.testing.assert_array_equal(base[0], relabelled[0])
    np.testing.assert_array_equal(base[1], relabelled[1])
    torch.testing.assert_close(
        logits(arm, k),
        logits(arm, k, batter=relabelled_bat, bowler=relabelled_bowl))


def test_recency_ignores_rows_older_than_its_window():
    arm, k, target = "recency", 3, 13
    window = attention_set(arm, k, target)
    assert window == {10, 11, 12, 13}
    sources = outcome_dependency(arm, k, target)
    assert sources == {10, 11, 12, 13}, (
        "under the own-outcome key construction an earlier row contributes "
        "its OWN outcome, so S(i) is the window plus row i-1 (already in it) "
        "and reaches no further back than W_k(i)")

    # Features of every row outside the window cannot reach the target.
    feats, _ = perturb(set(range(L)) - window, outcomes=False)
    torch.testing.assert_close(logits(arm, k)[target],
                               logits(arm, k, feats=feats)[target])
    # Nor can the outcomes of rows outside that set, with prev_y rebuilt from
    # the perturbed outcomes.
    _, y = perturb(set(range(L)) - sources, feats=False)
    torch.testing.assert_close(logits(arm, k)[target],
                               logits(arm, k, y=y)[target])
    # Positive control: inside the window it must matter.
    feats_in, y_in = perturb({11})
    assert not torch.allclose(logits(arm, k)[target],
                              logits(arm, k, feats=feats_in, y=y_in)[target])


@pytest.mark.parametrize("k", [0, 6, "unr"])
def test_same_entity_ignores_rows_outside_its_dependency_set(k):
    arm, target = "same_entity", 13
    allowed = attention_set(arm, k, target)
    sources = outcome_dependency(arm, k, target)
    outside_feats = set(range(L)) - allowed
    outside_outcomes = set(range(L)) - sources
    assert outside_feats, f"k={k}: nothing outside the attention set to test"
    assert outside_outcomes, f"k={k}: no outcome outside the dependency set"

    feats, _ = perturb(outside_feats, outcomes=False)
    torch.testing.assert_close(logits(arm, k)[target],
                               logits(arm, k, feats=feats)[target])
    _, y = perturb(outside_outcomes, feats=False)
    torch.testing.assert_close(logits(arm, k)[target],
                               logits(arm, k, y=y)[target])
    # Positive control: the target's own features always matter.
    feats_in, _ = perturb({target}, outcomes=False)
    assert not torch.allclose(logits(arm, k)[target],
                              logits(arm, k, feats=feats_in)[target])


def test_same_entity_unrestricted_set_and_sources_are_the_expected_rows():
    # Written out by hand so a silent change in either is visible: row 13 is
    # batter B on bowler Q; B batted 2, 3, 8, 9 and Q bowled 3, 4, 5, 12.
    assert attention_set("same_entity", "unr", 13) == {2, 3, 4, 5, 8, 9, 12,
                                                      13}
    # S(i) is the attention set plus the two rows feeding row 13's aligned
    # input: batter B's last earlier row is 9 and bowler Q's is 12, both
    # already in the unrestricted set, so S(i) == the attention set here.
    assert outcome_dependency("same_entity", "unr", 13) == {
        2, 3, 4, 5, 8, 9, 12, 13}
    # At k = 0 the attention set is the target alone and the two history
    # sources are all S(i) adds.
    assert outcome_dependency("same_entity", 0, 13) == {9, 12, 13}
    assert t1.history_source_rows("same_entity", 13, _codes(BATTER),
                                  _codes(BOWLER)) == {9, 12}
    assert t1.history_source_rows("recency", 13) == {12}
    assert t1.history_source_rows("recency", 0) == set()
    assert t1.history_source_rows("mlp", 5) == set()


@pytest.mark.parametrize("arm,k", [("recency", 6), ("same_entity", "unr"),
                                   ("same_entity", 0),
                                   ("aligned_hist_rf", None)])
def test_relay_free_arms_never_read_the_targets_own_outcome(arm, k):
    """§ 3.1 — y_i reaches row i through nothing.

    Row i's self key carries hist(i), and the own-outcome key at position i
    is masked off the diagonal, so the target's own realised outcome is
    unreadable at the target even though it IS readable by every later row.
    """
    target = 7
    y = Y.copy()
    y[target] = (y[target] + 3) % 6
    before, after = logits(arm, k), logits(arm, k, y=y)
    torch.testing.assert_close(before[target], after[target], rtol=0, atol=0)
    # Positive control: a LATER row does read it, so the perturbation is real.
    assert not torch.allclose(before[target + 1], after[target + 1])


@pytest.mark.parametrize("arm", [a for a in t1.ALL_ARMS
                                 if t1.ARM_WIRING[a] == "relay_free"])
def test_relay_free_arms_refuse_to_run_without_own_outcomes(arm):
    k = 6 if arm in t1.ARMS_NEEDING_K else None
    model = build(arm, k)
    prev = torch.zeros(1, 3, dtype=torch.long)
    with pytest.raises(ValueError, match="own_y"):
        model(torch.zeros(1, 3, t1.N_FEATS), prev,
              torch.zeros(1, 3, dtype=torch.bool), prev_bat=prev,
              prev_bowl=prev, batter=prev, bowler=prev)


def test_the_key_construction_table_matches_the_wiring():
    for arm in t1.ALL_ARMS:
        expected = {"relay_free": "own_outcome", "standard": "shifted_history",
                    "token_mlp": None, "recurrent": None}[t1.ARM_WIRING[arm]]
        assert t1.ARM_KEY_CONSTRUCTION[arm] == expected, arm
    # Only the relay-free arms own the extra embedding table, and it has no
    # BOS row: a row's own outcome always exists.
    for arm, k in (("recency", 6), ("same_entity", "unr"),
                   ("aligned_hist_rf", None), ("full", None), ("mlp", None)):
        model = build(arm, k)
        has = hasattr(model, "own_out_emb")
        assert has == (t1.ARM_KEY_CONSTRUCTION[arm] == "own_outcome"), arm
        if has:
            assert model.own_out_emb.num_embeddings == t1.N_OUTCOME_CLASSES
            assert t1.N_OUTCOME_CLASSES == 6 and t1.BOS == 6


@pytest.mark.parametrize("arm", ["residual_mlp", "residual_t1"])
def test_residual_arms_reproduce_the_base_at_initialisation(arm):
    # D3.7 — the head is zero-initialised, so r_theta == 0 and the output is
    # exactly log p_base. Exact, not approximate: a zero matmul is zero.
    base = np.log(np.random.default_rng(3).dirichlet(
        np.ones(6), size=L)).astype(np.float32)
    out = logits(arm, base_logp=base)
    torch.testing.assert_close(out, torch.tensor(base), rtol=0, atol=0)


@pytest.mark.parametrize("arm", ["residual_mlp", "residual_t1"])
def test_residual_arms_return_the_raw_residual_for_the_penalty(arm):
    model = build(arm)
    # A trained residual is non-zero; perturb the head so the test is not
    # reading the zero-init special case.
    with torch.no_grad():
        model.head.weight.add_(0.5)
        model.head.bias.add_(0.25)
    prev_bat, prev_bowl = t1.aligned_history(Y, BATTER, BOWLER, INNINGS)
    batch = t1.collate(INNINGS, FEATS, Y, "cpu", batter=_codes(BATTER),
                       bowler=_codes(BOWLER), prev_bat=prev_bat,
                       prev_bowl=prev_bowl, base_logp=UNIFORM_LOGP)
    with torch.no_grad():
        out, aux = model(batch.feats, batch.prev_y, batch.pad,
                         prev_bat=batch.prev_bat, prev_bowl=batch.prev_bowl,
                         batter=batch.batter, bowler=batch.bowler,
                         base_logp=batch.base_logp, own_y=batch.y)
    assert "residual" in aux
    torch.testing.assert_close(out, batch.base_logp + aux["residual"])


def test_residual_arms_refuse_to_run_without_a_base():
    model = build("residual_t1")
    with pytest.raises(ValueError, match="base_logp"):
        model(torch.zeros(1, 3, t1.N_FEATS), torch.zeros(1, 3,
                                                         dtype=torch.long),
              torch.zeros(1, 3, dtype=torch.bool))


def test_aligned_arms_refuse_to_run_without_aligned_inputs():
    model = build("aligned_hist")
    with pytest.raises(ValueError, match="prev_bat"):
        model(torch.zeros(1, 3, t1.N_FEATS), torch.zeros(1, 3,
                                                         dtype=torch.long),
              torch.zeros(1, 3, dtype=torch.bool))


@pytest.mark.parametrize("arm", [a for a in t1.ALL_ARMS
                                 if t1.ARM_WIRING[a] != "recurrent"])
def test_every_arm_is_finite_with_padding(arm):
    k = 6 if arm in t1.ARMS_NEEDING_K else None
    model = build(arm, k)
    pad = torch.zeros(2, L, dtype=torch.bool)
    pad[1, 5:] = True
    feats = torch.tensor(np.stack([FEATS, FEATS]))
    ids = torch.tensor(np.stack([_codes(BATTER), _codes(BATTER)]))
    prev = torch.tensor(np.stack([Y, Y]))
    base = torch.tensor(np.stack([UNIFORM_LOGP, UNIFORM_LOGP]))
    with torch.no_grad():
        out, _ = model(feats, prev, pad, prev_bat=prev, prev_bowl=prev,
                       batter=ids, bowler=ids, base_logp=base, own_y=prev)
    assert torch.isfinite(out[~pad]).all(), arm


# ------------------------------------------------- D3.9 fox / fixed_decay

def test_fixed_decay_and_fox_differ_only_in_the_gate():
    decay, fox = build("fixed_decay"), build("fox")
    decay_params = {n: tuple(p.shape) for n, p in decay.named_parameters()}
    fox_params = {n: tuple(p.shape) for n, p in fox.named_parameters()}
    gate = {n: s for n, s in fox_params.items() if n.startswith("fox_gates.")}
    assert gate, "the fox arm must own gate parameters"
    assert {n: s for n, s in fox_params.items()
            if not n.startswith("fox_gates.")} == decay_params
    # The gates are created last, so every shared parameter drew from the same
    # RNG stream and the two arms are the SAME model plus a bias.
    for name, tensor in decay.named_parameters():
        torch.testing.assert_close(tensor, dict(fox.named_parameters())[name])
    # Neither carries a positional embedding (arm register).
    for model in (decay, fox):
        assert not hasattr(model, "pos_emb")
        assert t1.ARM_POS_EMB[model.arm] is False


def test_fixed_decay_bias_is_the_alibi_formula():
    model = build("fixed_decay")
    tokens = torch.zeros(1, L, DMODEL)
    bias = model._bias(0, tokens, L)
    slopes = torch.pow(2.0, -8.0 * torch.arange(1, HEADS + 1,
                                                dtype=torch.float32) / HEADS)
    torch.testing.assert_close(t1.alibi_slopes(HEADS), slopes)
    expected = torch.zeros(1, HEADS, L, L)
    for h in range(HEADS):
        for i in range(L):
            for j in range(i + 1):
                expected[0, h, i, j] = -slopes[h] * (i - j)
    torch.testing.assert_close(bias * torch.tril(torch.ones(L, L)),
                               expected)
    # Bias is a pure function of position: no learned parameters.
    assert not any(n.startswith("alibi") for n, _ in model.named_parameters())


def test_fox_with_an_open_gate_is_vanilla_no_pos_emb_attention(monkeypatch):
    # sigma(g) -> 1 makes every log-gate 0, so the cumulative bias is exactly
    # zero and the arm collapses to unbiased causal attention. The matched
    # reference is fixed_decay with its slopes forced to zero, which is the
    # same layer stack with an identically-zero bias.
    fox = build("fox")
    for gate in fox.fox_gates:
        with torch.no_grad():
            gate.weight.zero_()
            gate.bias.fill_(1e4)  # exp(-1e4) underflows, so log sigma == 0
    tokens = torch.randn(1, L, DMODEL)
    assert torch.count_nonzero(fox._bias(0, tokens, L)) == 0

    monkeypatch.setattr(t1, "alibi_slopes",
                        lambda heads: torch.zeros(heads))
    decay = build("fixed_decay")
    assert torch.count_nonzero(decay._bias(0, tokens, L)) == 0
    torch.testing.assert_close(logits("fox", model=fox),
                               logits("fixed_decay", model=decay))


# --------------------------------------------------------- D3.11 arm_params

@pytest.mark.parametrize("arm", t1.ALL_ARMS)
def test_arm_params_block_is_complete_for_every_arm(arm):
    k = 6 if arm in t1.ARMS_NEEDING_K else None
    block = t1.arm_params_block(arm, k, 0.001, None, 1234)
    assert set(block) == {"arm", "k", "wiring", "history_input",
                          "key_construction", "positional_embedding", "bias",
                          "residual_l2", "base_logits_md5", "n_parameters"}
    assert block["key_construction"] == t1.ARM_KEY_CONSTRUCTION[arm]
    assert block["key_construction"] in {"own_outcome", "shifted_history",
                                         None}
    assert block["arm"] == arm and block["k"] == k
    assert block["wiring"] in {"token_mlp", "standard", "relay_free",
                              "recurrent"}
    assert block["history_input"] in {"none", "innings_previous",
                                      "participant_aligned"}
    assert block["bias"] in {"none", "alibi", "fox"}
    assert isinstance(block["positional_embedding"], bool)
    assert block["n_parameters"] == 1234
    assert (block["residual_l2"] is None) == (
        arm not in t1.ARMS_NEEDING_BASE_LOGITS)


def test_the_registered_arm_tables_agree_with_the_arm_register():
    assert t1.STAGE1_ARMS == ("full", "mlp", "no_attention", "no_history")
    assert set(t1.ALL_ARMS) == set(t1.ARM_WIRING) == set(t1.ARM_HISTORY) \
        == set(t1.ARM_BIAS) == set(t1.ARM_POS_EMB) \
        == set(t1.ARM_KEY_CONSTRUCTION)
    # fixed_decay and fox are the only arms without a positional embedding
    # among the attention arms; the recurrent and token-MLP arms have none.
    assert not t1.ARM_POS_EMB["fixed_decay"] and not t1.ARM_POS_EMB["fox"]
    for arm in ("full", "aligned_hist", "aligned_hist_rf", "recency",
                "same_entity", "residual_t1"):
        assert t1.ARM_POS_EMB[arm], arm
    # D3.6 — the masked pair shares one wiring and differs in mask + history.
    assert t1.ARM_WIRING["recency"] == t1.ARM_WIRING["same_entity"] == \
        "relay_free" == t1.ARM_WIRING["aligned_hist_rf"]
    assert t1.ARM_HISTORY["recency"] == "innings_previous"
    assert t1.ARM_HISTORY["same_entity"] == "participant_aligned"


# ------------------------------------------------------------- D3.3 the CLI

def _argv(*extra: str) -> list[str]:
    return ["transformer_t1.py", "--data-dir", "data/does_not_exist",
            "--no-kit", "--device", "cpu", *extra]


@pytest.mark.parametrize("args,message", [
    (["--arm", "full", "--k", "6"], "--k is not accepted"),
    (["--arm", "mlp", "--k", "unr"], "--k is not accepted"),
    (["--arm", "recency"], "requires --k"),
    (["--arm", "same_entity"], "requires --k"),
    (["--arm", "recency", "--k", "-1"], ">= 0"),
    (["--arm", "recency", "--k", "wide"], "'unr'"),
    (["--arm", "full", "--base-logits-dir", "x"],
     "--base-logits-dir is not accepted"),
    (["--arm", "recency", "--k", "6", "--base-logits-dir", "x"],
     "--base-logits-dir is not accepted"),
    (["--arm", "residual_t1"], "requires --base-logits-dir"),
    (["--arm", "residual_mlp"], "requires --base-logits-dir"),
])
def test_the_arm_flag_contract_is_enforced(monkeypatch, capsys, args, message):
    monkeypatch.setattr(sys, "argv", _argv(*args))
    with pytest.raises(SystemExit) as excinfo:
        t1.main()
    assert excinfo.value.code == 2
    assert message in capsys.readouterr().err


def test_every_stage2_arm_is_an_accepted_choice(monkeypatch, capsys):
    # An unknown arm is refused by argparse, and every registered arm is not.
    monkeypatch.setattr(sys, "argv", _argv("--arm", "gru"))
    with pytest.raises(SystemExit):
        t1.main()
    assert "invalid choice" in capsys.readouterr().err
    for arm in t1.STAGE2_ARMS:
        assert arm in t1.ALL_ARMS


# ------------------------------------------- D3.10 recurrent arm dispatch

@pytest.mark.parametrize("arm", ["lstm", "xlstm"])
def test_recurrent_arms_dispatch_to_the_recurrent_module(monkeypatch, arm):
    """The dispatch contract, exercised against a stub.

    `scripts/sequence_track/recurrent_arms.py` is a separate deliverable; this
    test pins the interface `transformer_t1` calls it through, so the two
    halves cannot disagree. A stub is installed even when the real module
    exists, so the test never depends on the other arm's internals.
    """
    seen = {}

    class Stub(torch.nn.Module):
        def __init__(self, n_feats, dmodel):
            super().__init__()
            self.proj = torch.nn.Linear(n_feats, 6)
            self.emb = torch.nn.Embedding(7, 6)

        def forward(self, feats, prev_y, pad_mask):
            return self.proj(feats) + self.emb(prev_y), {}

    def build_recurrent_model(arm_name, n_feats, dmodel, dropout):
        seen.update(arm=arm_name, n_feats=n_feats, dmodel=dmodel,
                    dropout=dropout)
        return Stub(n_feats, dmodel)

    module = types.ModuleType("sequence_track.recurrent_arms")
    module.build_recurrent_model = build_recurrent_model
    monkeypatch.setitem(sys.modules, "sequence_track.recurrent_arms", module)

    model = t1.T1Model(t1.N_FEATS, DMODEL, LAYERS, HEADS, arm=arm).eval()
    assert seen == {"arm": arm, "n_feats": t1.N_FEATS, "dmodel": DMODEL,
                    "dropout": 0.1}
    out, aux = model(torch.zeros(1, L, t1.N_FEATS),
                     torch.tensor(Y).view(1, -1),
                     torch.zeros(1, L, dtype=torch.bool))
    assert out.shape == (1, L, 6) and aux == {}
    block = t1.arm_params_block(arm, None, None, None, 1)
    assert block["wiring"] == "recurrent"
    assert block["history_input"] == "innings_previous"
    assert block["positional_embedding"] is False


def test_recurrent_arms_carry_no_aux_heads(monkeypatch):
    module = types.ModuleType("sequence_track.recurrent_arms")
    module.build_recurrent_model = lambda *a, **kw: torch.nn.Linear(1, 1)
    monkeypatch.setitem(sys.modules, "sequence_track.recurrent_arms", module)
    with pytest.raises(ValueError, match="aux heads"):
        t1.T1Model(t1.N_FEATS, DMODEL, LAYERS, HEADS, {"shot": 3}, arm="lstm")


# ------------------------------------------- D3.7 / D4 base-logits loading

def write_base_logits(directory: Path, split: str, n_rows: int,
                      parquet_md5: str, logp=None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    if logp is None:
        probs = np.random.default_rng(11).dirichlet(np.ones(6), size=n_rows)
        logp = np.log(np.clip(probs, 1e-4, None)).astype(np.float32)
    path = directory / f"{split}.npz"
    np.savez_compressed(path, logp=np.asarray(logp, dtype=np.float32),
                        n_rows=np.int64(n_rows),
                        parquet_md5=np.array(parquet_md5))
    return path


def test_base_logits_round_trip(tmp_path):
    path = write_base_logits(tmp_path, "validation", 5, "abc123")
    logp, digest = t1.load_base_logits(tmp_path, "validation", 5, "abc123")
    assert logp.shape == (5, 6) and logp.dtype == np.float32
    assert digest == t1.md5_file(path)


@pytest.mark.parametrize("rows,md5,message", [
    (4, "abc123", "has 4 rows"),
    (5, "other", "refusing to train"),
])
def test_base_logits_refuse_a_mismatched_split(tmp_path, rows, md5, message):
    write_base_logits(tmp_path, "validation", 5, "abc123")
    with pytest.raises(RuntimeError, match=message):
        t1.load_base_logits(tmp_path, "validation", rows, md5)


def test_missing_base_logits_are_named(tmp_path):
    with pytest.raises(RuntimeError, match="not found"):
        t1.load_base_logits(tmp_path, "train", 5, "abc123")


# ------------------------------- D3.11 one real (tiny) run per new wiring

@pytest.mark.parametrize("arm,extra", [
    ("fixed_decay", []),
    ("fox", []),
    ("aligned_hist", []),
    ("aligned_hist_rf", []),
    ("recency", ["--k", "6"]),
    ("same_entity", ["--k", "unr"]),
])
def test_a_tiny_real_run_writes_arm_params_and_predictions(tmp_path,
                                                          monkeypatch, arm,
                                                          extra):
    frame = write_frame(tmp_path / "frame", "i7")
    cache = write_cache(tmp_path / "cache.sqlite", "venue_aliases_v1")
    out = tmp_path / f"out_{arm}"
    metrics = run_trainer(monkeypatch, frame, out, "--no-kit", "--kit-dir",
                          str(tmp_path / "absent"), "--arm", arm,
                          "--save-predictions", *extra, *CACHE_ARGS,
                          str(cache))
    block = metrics["arm_params"]
    assert block["arm"] == arm
    assert block["wiring"] == t1.ARM_WIRING[arm]
    assert block["history_input"] == t1.ARM_HISTORY[arm]
    assert block["key_construction"] == t1.ARM_KEY_CONSTRUCTION[arm]
    assert block["bias"] == t1.ARM_BIAS[arm]
    assert block["residual_l2"] is None
    assert block["base_logits_md5"] is None
    assert block["n_parameters"] == metrics["n_params"]
    assert block["k"] == (6 if arm == "recency"
                          else "unr" if arm == "same_entity" else None)
    # The training contract's pinned blocks are untouched by stage 2.
    assert metrics["training_contract"]["architecture"]["arm"] == arm
    assert set(metrics["training_contract"]["optimiser"]) == {
        "lr", "batch", "epochs", "patience", "aux", "aux_weight"}
    # Row-aligned validation probabilities, exactly as stage 1 wrote them.
    with np.load(out / "predictions_validation.npz") as archive:
        probs = archive["probs"]
        assert probs.shape == (48, 6)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
        assert len(archive["innings_id"]) == 48


def test_a_tiny_residual_run_records_the_base_logits_md5(tmp_path,
                                                         monkeypatch):
    frame = write_frame(tmp_path / "frame", "i7")
    cache = write_cache(tmp_path / "cache.sqlite", "venue_aliases_v1")
    base_dir = tmp_path / "base_logits"
    digests = {}
    for split, rows in (("train", 96), ("validation", 48)):
        parquet = t1.split_path(frame, "i7", split)
        path = write_base_logits(base_dir, split, rows, t1.md5_file(parquet))
        digests[split] = t1.md5_file(path)
    out = tmp_path / "out_residual"
    metrics = run_trainer(monkeypatch, frame, out, "--no-kit", "--kit-dir",
                          str(tmp_path / "absent"), "--arm", "residual_t1",
                          "--base-logits-dir", str(base_dir),
                          "--residual-l2", "0.001", "--save-predictions",
                          *CACHE_ARGS, str(cache))
    block = metrics["arm_params"]
    assert block["residual_l2"] == 0.001
    assert block["base_logits_md5_by_split"] == digests
    assert block["base_logits_md5"] and len(block["base_logits_md5"]) == 32
    on_disk = json.loads((out / "metrics.json").read_text())
    assert on_disk["arm_params"] == block


def test_a_residual_run_refuses_base_logits_from_another_frame(tmp_path,
                                                              monkeypatch):
    frame = write_frame(tmp_path / "frame", "i7")
    cache = write_cache(tmp_path / "cache.sqlite", "venue_aliases_v1")
    base_dir = tmp_path / "base_logits"
    for split, rows in (("train", 96), ("validation", 48)):
        write_base_logits(base_dir, split, rows, "0" * 32)
    with pytest.raises(RuntimeError, match="refusing to train"):
        run_trainer(monkeypatch, frame, tmp_path / "out", "--no-kit",
                    "--kit-dir", str(tmp_path / "absent"), "--arm",
                    "residual_mlp", "--base-logits-dir", str(base_dir),
                    *CACHE_ARGS, str(cache))
