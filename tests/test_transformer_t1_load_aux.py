"""Regression test for review finding BR4 (2026-08-14).

`load_aux` used to fillna(-1) BEFORE vocab construction, so "-1" (control)
and the placeholder "-" (shot) became genuine vocab classes whose rows
passed the `tgt >= 0` training mask — the aux heads trained and scored on a
fabricated "unlabeled" class. Missing labels must map to the -1 sentinel.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from transformer_t1 import load_aux  # noqa: E402


def _write_labels(tmp_path, split="train"):
    frame = pd.DataFrame({
        "row_idx": [0, 1, 2, 3],
        "shot": ["drive", "-", "pull", "drive"],
        "line": ["stump", "wide", None, "stump"],
        "length": ["full", "short", "full", None],
        "control": [1.0, 0.0, np.nan, 1.0],
    })
    frame.to_parquet(tmp_path / f"{split}.parquet")
    return tmp_path


def test_missing_labels_never_become_classes(tmp_path):
    aux_dir = _write_labels(tmp_path)
    targets, vocabs = load_aux("train", n_rows=6, vocabs=None,
                               aux_dir=aux_dir)

    assert set(vocabs["control"]) == {"0", "1"}, \
        f"'-1' must not be a control class: {vocabs['control']}"
    assert set(vocabs["shot"]) == {"drive", "pull"}, \
        f"the '-' placeholder must not be a shot class: {vocabs['shot']}"

    # Unlabeled rows carry the -1 sentinel (masked by tgt >= 0), labeled
    # rows carry their vocab index; rows outside the join stay -1.
    assert targets["control"][2] == -1
    assert targets["shot"][1] == -1
    assert targets["line"][2] == -1
    assert targets["length"][3] == -1
    assert targets["control"][0] == vocabs["control"]["1"]
    assert targets["shot"][2] == vocabs["shot"]["pull"]
    assert targets["control"][4] == -1 and targets["control"][5] == -1


def test_val_split_reuses_train_vocab_and_sentinels_unknowns(tmp_path):
    aux_dir = _write_labels(tmp_path)
    _, vocabs = load_aux("train", n_rows=6, vocabs=None, aux_dir=aux_dir)
    val = pd.DataFrame({
        "row_idx": [0, 1],
        "shot": ["drive", "unseen_shot"],
        "line": ["stump", "stump"],
        "length": ["full", "full"],
        "control": [np.nan, 1.0],
    })
    val.to_parquet(aux_dir / "val.parquet")
    targets, _ = load_aux("val", n_rows=2, vocabs=vocabs, aux_dir=aux_dir)
    assert targets["control"][0] == -1
    assert targets["shot"][1] == -1, \
        "a label value unseen in train must sentinel to -1, not crash"
