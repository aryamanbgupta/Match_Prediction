"""Participant-aligned history and attention-set masks (stage 2, D3.4/D3.5).

Everything here is a hand-built innings on the CPU: no frame, no checkpoint,
no `models/` directory is read or written. The point of hand-building is that
the expected vectors are written out by hand from the cricket, so a change in
`aligned_history` or `attention_mask` cannot be rationalised after the fact.

The innings below is nine deliveries with, in order: a repeat by the same
batter on the same bowler, an extras row, a batter change, a wicket, a new
batter, a bowler change, and finally the first bowler RETURNING after a gap.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import transformer_t1 as t1  # noqa: E402

BOS = t1.BOS

# row:      0    1    2    3    4    5    6    7    8
# batter:   A    A    B    B    C    C    A    A    B
# bowler:   P    P    P    P    P    Q    Q    Q    P
# outcome:  0    1    0    5    1    2    3    4    0
#                (wide)    (wkt)          (bowler P returns at row 8)
BATTER = np.array(list("AABBCCAAB"))
BOWLER = np.array(list("PPPPPQQQP"))
OUTCOME = np.array([0, 1, 0, 5, 1, 2, 3, 4, 0], dtype=np.int64)
INNINGS = [np.arange(9)]


# ------------------------------------------------- D3.4 aligned history

def test_aligned_history_exact_vectors():
    prev_bat, prev_bowl = t1.aligned_history(OUTCOME, BATTER, BOWLER, INNINGS)

    # A's first ball (row 0) has no earlier A row -> BOS; row 1 reads A's
    # row-0 outcome; row 2 is B's first ball -> BOS; row 3 reads B's row-2
    # outcome; row 4 is C's first -> BOS; row 6 is A again, and the last
    # earlier A row is row 1 (the extras row), so it reads outcome 1; row 8
    # is B again after the wicket, and B's last earlier row is row 3 -> 5.
    np.testing.assert_array_equal(
        prev_bat, np.array([BOS, 0, BOS, 0, BOS, 1, 1, 3, 5]))

    # P bowls rows 0-4 then returns at row 8; Q bowls rows 5-7. Row 1 is an
    # extras row and still counts as P's previous outcome for row 2, and P's
    # return at row 8 reads P's last earlier row (row 4, outcome 1) across
    # the three-ball gap.
    np.testing.assert_array_equal(
        prev_bowl, np.array([BOS, 0, 1, 0, 5, BOS, 2, 3, 1]))


def test_aligned_history_resets_at_the_innings_boundary():
    # The same batter and bowler in a second innings must start from BOS: the
    # aligned input is within-innings by definition (arm register).
    y = np.concatenate([OUTCOME, OUTCOME])
    batter = np.concatenate([BATTER, BATTER])
    bowler = np.concatenate([BOWLER, BOWLER])
    innings = [np.arange(9), np.arange(9, 18)]
    prev_bat, prev_bowl = t1.aligned_history(y, batter, bowler, innings)
    assert prev_bat[9] == BOS and prev_bowl[9] == BOS
    np.testing.assert_array_equal(prev_bat[:9], prev_bat[9:])
    np.testing.assert_array_equal(prev_bowl[:9], prev_bowl[9:])


def test_aligned_history_never_reads_the_current_row():
    # Invariant 2: the outcome of ball r is never an input to ball r. Changing
    # only the LAST row's outcome must leave every aligned input untouched.
    changed = OUTCOME.copy()
    changed[-1] = (changed[-1] + 1) % 6
    before = t1.aligned_history(OUTCOME, BATTER, BOWLER, INNINGS)
    after = t1.aligned_history(changed, BATTER, BOWLER, INNINGS)
    np.testing.assert_array_equal(before[0], after[0])
    np.testing.assert_array_equal(before[1], after[1])


def test_aligned_history_refuses_misaligned_inputs():
    with pytest.raises(ValueError, match="row-aligned"):
        t1.aligned_history(OUTCOME, BATTER[:-1], BOWLER, INNINGS)


# ----------------------------------------------------- D3.5 attention sets

def _codes(values: np.ndarray) -> torch.Tensor:
    _, codes = np.unique(values, return_inverse=True)
    return torch.tensor(codes.reshape(1, -1), dtype=torch.long)


BAT_T = _codes(BATTER)
BOWL_T = _codes(BOWLER)
PAD = torch.zeros(1, 9, dtype=torch.bool)


def _allowed(arm: str, k) -> list[set[int]]:
    mask = t1.attention_mask(arm, k, BAT_T, BOWL_T, PAD)
    assert mask.shape == (1, 9, 9) and mask.dtype == torch.bool
    return [set(torch.nonzero(row).flatten().tolist()) for row in mask[0]]


@pytest.mark.parametrize("arm", ["recency", "same_entity"])
@pytest.mark.parametrize("k", [0, 6, "unr"])
def test_the_target_is_always_in_its_own_set(arm, k):
    # No query row may be empty, or the softmax over an all -inf row is NaN.
    for i, allowed in enumerate(_allowed(arm, k)):
        assert i in allowed, f"{arm} k={k}: row {i} does not see itself"
        assert allowed, f"{arm} k={k}: row {i} has an empty set"


@pytest.mark.parametrize("arm", ["recency", "same_entity"])
def test_k_zero_is_the_target_alone(arm):
    assert _allowed(arm, 0) == [{i} for i in range(9)]


def test_recency_is_a_causal_window_counted_in_rows():
    # k counts delivery ROWS inclusive of extras, so row 1 (the wide) uses up
    # one slot of the window: at k = 6 row 8 reaches back only to row 2.
    assert _allowed("recency", 6)[8] == {2, 3, 4, 5, 6, 7, 8}
    assert _allowed("recency", 6)[3] == {0, 1, 2, 3}
    # Unrestricted is the whole causal prefix.
    assert _allowed("recency", "unr")[8] == set(range(9))
    assert _allowed("recency", None)[8] == set(range(9))


def test_same_entity_keeps_only_shared_batter_or_bowler():
    unrestricted = _allowed("same_entity", "unr")
    # Row 8 is batter B on bowler P. B batted rows 2 and 3; P bowled rows
    # 0-4. So everything up to row 4 qualifies, and Q's rows 5-7 do not.
    assert unrestricted[8] == {0, 1, 2, 3, 4, 8}
    # Row 5 is batter C on bowler Q, Q's first ball: C's earlier row 4 is the
    # only qualifying history, plus itself.
    assert unrestricted[5] == {4, 5}
    # Row 6 is batter A on bowler Q: A batted 0, 1; Q bowled 5.
    assert unrestricted[6] == {0, 1, 5, 6}


def test_a_bowler_returning_after_a_gap_is_still_visible():
    # P bowls rows 0-4, Q bowls 5-7, P returns at row 8. Unrestricted, row 8
    # sees P's earlier work; the three-row gap does not hide it.
    assert {0, 1, 2, 3, 4} <= _allowed("same_entity", "unr")[8]
    # It is the WINDOW, not the entity rule, that hides it at k = 6: rows 0
    # and 1 fall out and rows 2-4 survive.
    at_k6 = _allowed("same_entity", 6)[8]
    assert at_k6 == {2, 3, 4, 8}
    assert 0 not in at_k6 and 1 not in at_k6


def test_every_set_is_causal():
    for arm in ("recency", "same_entity", "aligned_hist_rf"):
        for k in (0, 6, "unr"):
            for i, allowed in enumerate(_allowed(arm, k) if arm in
                                        t1.ARMS_NEEDING_K
                                        else _allowed(arm, None)):
                assert max(allowed) <= i, f"{arm} k={k} row {i} reads ahead"


def test_aligned_hist_rf_is_the_unrestricted_relay_free_endpoint():
    # D3.8: the mask of aligned_hist_rf equals same_entity's at k = unr with
    # the entity rule dropped, i.e. the whole causal prefix.
    mask = t1.attention_mask("aligned_hist_rf", None, BAT_T, BOWL_T, PAD)
    causal = torch.tril(torch.ones(9, 9, dtype=torch.bool)).unsqueeze(0)
    assert torch.equal(mask, causal)


def test_padded_keys_are_never_read_and_no_row_is_empty():
    pad = torch.zeros(1, 9, dtype=torch.bool)
    pad[0, 6:] = True
    mask = t1.attention_mask("same_entity", "unr", BAT_T, BOWL_T, pad)
    # Real queries never read a padded key...
    assert not mask[0, :6, 6:].any()
    # ...and every row, padded or not, keeps its own diagonal so that the
    # softmax has at least one finite score.
    assert mask[0].diagonal().all()


def test_same_entity_refuses_to_build_a_mask_without_ids():
    with pytest.raises(ValueError, match="batter/bowler"):
        t1.attention_mask("same_entity", "unr", None, None, PAD)


# ---------------------------------- stage 3: the registered mask property
#
# `attention_mask` used to spell the ownership restriction as the literal
# `arm == "same_entity"`. It is now read from `t1.ARM_MASK_OWNERSHIP`, so a
# later same-entity variant cannot silently lose its mask by being given a
# different name. These tests pin BOTH halves of that change: the table says
# exactly what the literal said, and every arm that existed before the
# refactor produces byte-identical masks.

# The fixture innings as integer codes (`attention_set` takes id arrays).
BAT_CODE = np.array([0, 0, 1, 1, 2, 2, 0, 0, 1])
BOWL_CODE = np.array([0, 0, 0, 0, 0, 1, 1, 1, 0])

# Verified against the pre-refactor `transformer_t1.py` (git HEAD before the
# stage 3 edit) for EVERY arm, every registered k and every target row: zero
# differences. The two ownership-sensitive arms are pinned literally here.
SAME_ENTITY_K3 = [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3], [1, 2, 3, 4],
                  [4, 5], [5, 6], [5, 6, 7], [8]]
SAME_ENTITY_UNR = [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4],
                   [4, 5], [0, 1, 5, 6], [0, 1, 5, 6, 7],
                   [0, 1, 2, 3, 4, 8]]
RECENCY_K3 = [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3], [1, 2, 3, 4],
              [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]]


def test_arm_mask_ownership_is_exactly_the_old_literal():
    """The table must name every arm, and restrict only `same_entity`."""
    assert set(t1.ARM_MASK_OWNERSHIP) == set(t1.ALL_ARMS)
    owning = {arm for arm, flag in t1.ARM_MASK_OWNERSHIP.items() if flag}
    assert owning == {"same_entity"}


def test_unregistered_arm_raises_rather_than_reading_the_full_prefix():
    """The point of the table: a missing arm is loud, not silently unmasked."""
    with pytest.raises(KeyError):
        t1.attention_mask("not_an_arm", None, BAT_T, BOWL_T, PAD)


@pytest.mark.parametrize("k, expected", [(3, SAME_ENTITY_K3),
                                         ("unr", SAME_ENTITY_UNR)])
def test_same_entity_sets_unchanged_by_the_refactor(k, expected):
    got = [sorted(t1.attention_set("same_entity", k, target,
                                   BAT_CODE, BOWL_CODE, 9))
           for target in range(9)]
    assert got == expected


def test_recency_sets_unchanged_by_the_refactor():
    got = [sorted(t1.attention_set("recency", 3, target,
                                   BAT_CODE, BOWL_CODE, 9))
           for target in range(9)]
    assert got == RECENCY_K3


@pytest.mark.parametrize("arm", [
    arm for arm in t1.STAGE1_ARMS + t1.STAGE2_ARMS
    if arm not in t1.ARMS_NEEDING_K
    and t1.ARM_WIRING[arm] in ("standard", "relay_free")
    and arm != "no_attention"])
def test_unwindowed_arms_still_read_the_full_causal_prefix(arm):
    """Every pre-stage-3 arm without a window is unchanged: full prefix."""
    causal = torch.tril(torch.ones(9, 9, dtype=torch.bool)).unsqueeze(0)
    assert torch.equal(t1.attention_mask(arm, None, BAT_T, BOWL_T, PAD),
                       causal)


def test_aligned_hist_decay_is_not_an_ownership_arm():
    """Block A's arm is `aligned_hist` plus decay: ownership is in its history
    INPUT, never in its mask, so it reads the full causal prefix."""
    assert t1.ARM_MASK_OWNERSHIP["aligned_hist_decay"] is False
    causal = torch.tril(torch.ones(9, 9, dtype=torch.bool)).unsqueeze(0)
    assert torch.equal(
        t1.attention_mask("aligned_hist_decay", None, BAT_T, BOWL_T, PAD),
        causal)
    # Its history sources are the participant-aligned ones, exactly as
    # `aligned_hist`'s are.
    for target in range(9):
        assert (t1.history_source_rows("aligned_hist_decay", target,
                                       BAT_CODE, BOWL_CODE)
                == t1.history_source_rows("aligned_hist", target,
                                          BAT_CODE, BOWL_CODE))
