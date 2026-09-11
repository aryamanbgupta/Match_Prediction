# manifest-exempt: embeddings-ladder experimental artifact namespace
"""T1 — minimal innings-scope transformer over ball sequences (Phase 2).

Hypothesis (design doc + Phase-1 exit): within-innings history carries
information no EB marginal or state snapshot captures. Phase 1 answered the
identity question (EB-saturated), so tokens carry the EB features directly
(hybrid) — the learned-ID arm is intentionally absent at T1.

Token t = Linear(pre-ball features_t) + Embedding(outcome_{t-1})
  pre-ball: EB batter 18 + EB bowler 18 + venue 6 + state (phase flags,
  wickets, chasing, balls_remaining, score, RR, RRR) — all known before
  the ball. Outcomes enter only as history (BOS at t=0): the same causal
  discipline as the tracker pipeline, enforced by the shift + causal mask.

Scoreboard: per-ball LL on the frozen eval kit (the registered, converged
exact-feature logistic is 1.4433 validation / 1.4340 test), plus unseen-pair,
frequency-bucket and calibration diagnostics. Match simulation remains a
separate downstream gate.

Frame, kit and contract (sequence track stage 1, D3):
  * the parquet stem is derived from the frame directory's ``.feature_hash``
    ``version`` field, so ``--data-dir data/xgb_data_i7`` reads
    ``cricket_data_i7_{split}.parquet``; ``--frame-version`` overrides it;
  * ``--no-kit`` trains, scores validation and saves without reading the
    frozen eval kit, and then the test split is loaded only under
    ``--score-test``;
  * ``metrics.json`` carries a ``training_contract`` block that records the
    frame, its declared identity and delivery semantics, the md5 of every
    parquet actually read, the ordered 50 feature names, the state
    normalisers, the class mapping, architecture, optimiser, seed, device
    and the resolved stats cache, so a checkpoint can be refused at serve
    time on any identity mismatch.

Who gets a contract, and who does not (Astra round 1, SHOULD 2). The block is
written if and only if the frame declares ``delivery_semantics``:
  * frame declares semantics and ``--stats-cache-role`` is given — the
    contract is written and ``scripts/sim_t1.py`` can serve the checkpoint on
    a cache whose venue identity matches;
  * frame declares semantics and no ``--stats-cache-role`` — refused before
    training starts. A contracted frame must record the cache it was
    materialized from, or the checkpoint is born unservable (a contract with
    a null ``stats_cache.md5`` is rejected by the D4 guard);
  * frame declares no semantics (the v3 frame) and no ``--stats-cache-role``
    — NO ``training_contract`` key is written at all, and a warning says so.
    Such a checkpoint serves only through ``sim_t1.py``'s legacy route: its
    ``config.data_dir`` must contain ``xgb_data_v3`` and the serving cache
    must carry no venue identity;
  * frame declares no semantics but ``--stats-cache-role`` is given —
    refused: there is no contract for the resolved cache to be recorded in,
    and silently dropping an explicit flag would hide that.

Numerics note: ``eval_split`` accumulates the per-row loss mean in float64
(it was float32 before stage 1). That mean is the early-stopping signal, so a
v3-frame rerun is NOT byte-identical to a pre-change run: an epoch may stop
one step earlier or later and the saved weights may differ. The feature
arithmetic is untouched and identical; only the loss reduction changed.

Usage:
    uv run python scripts/transformer_t1.py [--dmodel 128] [--layers 2]
        [--out models/embeddings/t1]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embeddings_e1 import (CLASS_MAPPING, CTX_COLS, EB_BAT_COLS,  # noqa: E402
                           EB_BOWL_COLS, VENUE_COLS, make_ctx)

REPO_ROOT = Path(__file__).resolve().parents[1]
KIT = Path("models/embeddings/eval_kit")
DATA = Path("data/xgb_data_v3")

STATE_COLS = ["balls_remaining", "score", "run_rate", "run_rate_required"]
BOS = 6  # outcome-history vocab: 6 classes + BOS
AUX_DIR = Path("models/embeddings/deepcrease_labels")
AUX_TASKS = ["shot", "line", "length", "control"]

CONTRACT_VERSION = "seq_stage1_t1_training_contract_v1"
# The divisors `build_features` applies to the four state columns. Declared
# once so the contract cannot drift from the tensor the model was trained on.
STATE_NORMALISERS = {
    "balls_remaining": 120.0,
    "score": 200.0,
    "run_rate": 12.0,
    "run_rate_required_clip": [0, 36],
    "run_rate_required": 12.0,
}
N_FEATS = 50
# Default aux loss weight. Named so the retrain runner can compare a
# checkpoint's recorded optimiser block against the value a run that passes
# no --aux-weight would actually have used.
AUX_WEIGHT_DEFAULT = 0.2
# Bookkeeping column read only to record each split's date range; it is
# never part of the feature tensor.
DATE_COL = "match_date"

# --------------------------------------------------------------------------
# Sequence track stage 2 (docs/sequence_track/stage2_acceptance.md, D3).
#
# The four stage 1 arms keep their exact behaviour: `STAGE1_ARMS` is the
# registered stage 1 set and every table below reproduces what those arms
# already did, so adding stage 2 arms cannot move a stage 1 number (D3.3).
STAGE1_ARMS = ("full", "mlp", "no_attention", "no_history")
STAGE2_ARMS = ("fixed_decay", "fox", "aligned_hist", "aligned_hist_rf",
               "recency", "same_entity", "residual_mlp", "residual_t1",
               "lstm", "xlstm")
ALL_ARMS = STAGE1_ARMS + STAGE2_ARMS

# Arms whose attention set is parameterised by a window k (D3.3: --k is
# required for these and refused for every other arm).
ARMS_NEEDING_K = ("recency", "same_entity")
# Arms that consume a base log-probability input (D3.3 / D4).
ARMS_NEEDING_BASE_LOGITS = ("residual_mlp", "residual_t1")

# How every arm is wired. "token_mlp" has no sequence access at all;
# "standard" is `nn.TransformerEncoder`-shaped (queries, keys and values all
# from the running residual stream); "relay_free" takes keys and values from
# the LAYER-ZERO token embedding at every layer, so h_i depends only on
# {e_j : j in the attention set of i} (arm register, "Relay-free wiring";
# certified in D7); "recurrent" dispatches to `sequence_track.recurrent_arms`.
ARM_WIRING = {
    "mlp": "token_mlp", "residual_mlp": "token_mlp",
    "full": "standard", "no_attention": "standard",
    "no_history": "standard", "aligned_hist": "standard",
    "residual_t1": "standard", "fixed_decay": "standard", "fox": "standard",
    "aligned_hist_rf": "relay_free", "recency": "relay_free",
    "same_entity": "relay_free",
    "lstm": "recurrent", "xlstm": "recurrent",
}
# The history input each arm reads. "innings_previous" is the stage 1
# `out_emb(prev_y)`; "participant_aligned" replaces it with
# `bat_emb(prev_bat) + bowl_emb(prev_bowl)` (D3.4). `no_history` reads the
# innings-previous pathway with every value forced to BOS, so its declared
# history input is "none" while it still owns an `out_emb`.
ARM_HISTORY = {
    "mlp": "none", "residual_mlp": "none", "no_history": "none",
    "full": "innings_previous", "no_attention": "innings_previous",
    "residual_t1": "innings_previous", "fixed_decay": "innings_previous",
    "fox": "innings_previous", "recency": "innings_previous",
    "lstm": "innings_previous", "xlstm": "innings_previous",
    "aligned_hist": "participant_aligned",
    "aligned_hist_rf": "participant_aligned",
    "same_entity": "participant_aligned",
}
# How the KEY/VALUE token of an earlier row is built (handoff § 3.1, the
# `relay_free_keys_carry_own_outcome` asymmetry).
#
#   * "own_outcome" (every relay-free arm) — the key/value token for an
#     earlier row j < i is e_j = feat_proj(feat_j) + own_out_emb(y_j) + pos_j,
#     with y_j row j's OWN realised outcome (known before ball i). The query
#     stream's initial state and the self key/value (j == i) are instead
#     e_i^self = feat_proj(feat_i) + hist(i) + pos_i, so row i never reads
#     y_i. The dependency set is then exactly S(i) = attention_set(i) union
#     {history-source rows of i}, which is what D7 certifies;
#   * "shifted_history" (standard wiring) — every token, key or query, is
#     feat_proj(feat_j) + hist(j) + pos_j, i.e. the SHIFTED previous outcome.
#     Unchanged from stage 1;
#   * None — the arm has no keys (token_mlp) or no attention (recurrent).
#
# Consequence, recorded as a known asymmetry: relay-free arms read earlier
# rows as (state, own outcome) pairs and standard arms as (state, shifted
# previous outcome) pairs, so `aligned_hist_rf - aligned_hist` measures the
# wiring AND the key construction, not the wiring alone.
ARM_KEY_CONSTRUCTION = {
    arm: ("own_outcome" if ARM_WIRING[arm] == "relay_free"
          else "shifted_history" if ARM_WIRING[arm] == "standard" else None)
    for arm in ALL_ARMS}
# Own-outcome vocabulary: 6 real classes, no BOS. A row's own outcome always
# exists, so there is nothing for a BOS symbol to stand for.
N_OUTCOME_CLASSES = 6

# The additive attention bias. `fixed_decay` and `fox` differ ONLY here and
# in the gate parameters (D3.9), and neither carries a positional embedding.
ARM_BIAS = {arm: "none" for arm in ALL_ARMS}
ARM_BIAS["fixed_decay"] = "alibi"
ARM_BIAS["fox"] = "fox"
# Positional embedding: every attention arm except the two bias arms, which
# are registered "pos emb: no" because their bias carries position.
ARM_POS_EMB = {
    arm: ARM_WIRING[arm] in ("standard", "relay_free")
    and ARM_BIAS[arm] == "none" for arm in ALL_ARMS}
# The FoX gate bias initialiser: sigma(4) ~ 0.982, so the gate is ~1 (no
# forgetting) at initialisation and the arm starts from vanilla attention.
FOX_GATE_BIAS_INIT = 4.0
RESIDUAL_L2_DEFAULT = 0.001


def load_aux(split: str, n_rows: int, vocabs: dict | None,
             aux_dir: Path = AUX_DIR):
    """Per-row aux targets from the DeepCrease join; -1 = unlabeled.
    Vocabs are built from the train file and reused for val/test."""
    lab = pd.read_parquet(aux_dir / f"{split}.parquet")
    # Missing labels must stay missing until AFTER vocab construction:
    # the old fillna(-1)-then-vocab order made "-1" (and the shot
    # placeholder "-") real trainable classes that passed the tgt >= 0
    # mask, so the control/shot heads trained on a fabricated "unlabeled"
    # class. Mirrors run_xr_same_cohort._normalise_labels semantics.
    lab["control"] = lab["control"].map(
        lambda v: str(int(v)) if pd.notna(v) else np.nan)
    lab["shot"] = lab["shot"].replace("-", np.nan)
    if vocabs is None:
        vocabs = {t: {v: i for i, v in enumerate(sorted(
            lab[t].dropna().unique()))} for t in AUX_TASKS}
    out = {}
    for t in AUX_TASKS:
        arr = np.full(n_rows, -1, dtype=np.int64)
        vals = lab[t].map(vocabs[t]).fillna(-1).astype(np.int64)
        arr[lab["row_idx"].to_numpy()] = vals
        out[t] = arr
    return out, vocabs


def read_feature_hash(data_dir: Path) -> dict | None:
    """The frame directory's ``.feature_hash`` JSON, or None when absent."""
    path = Path(data_dir) / ".feature_hash"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def resolve_frame_version(data_dir: Path, override: str | None = None) -> str:
    """The parquet stem version: explicit override > frame declaration.

    Never falls back to a hardcoded stem: a frame directory that declares no
    version is an unidentified frame, and training on one silently is exactly
    the failure the stage-1 contract exists to prevent.
    """
    if override:
        return str(override)
    declared = (read_feature_hash(data_dir) or {}).get("version")
    if not declared:
        raise RuntimeError(
            f"{Path(data_dir) / '.feature_hash'} does not declare a frame "
            "version; pass --frame-version to name the parquet stem")
    return str(declared)


def split_path(data_dir: Path, version: str, name: str) -> Path:
    return Path(data_dir) / f"cricket_data_{version}_{name}.parquet"


def feature_names() -> list[str]:
    """The 50 pre-ball feature names, in `build_features` concatenation order.

    The four CTX names label the transformed columns `make_ctx` emits
    (phase flags, wickets in hand / 10, chase flag from `chase_target`), and
    the four STATE names label the normalised state columns.
    """
    names = (list(EB_BAT_COLS) + list(EB_BOWL_COLS) + list(VENUE_COLS)
             + list(CTX_COLS) + list(STATE_COLS))
    if len(names) != N_FEATS or len(set(names)) != N_FEATS:
        raise AssertionError(f"expected {N_FEATS} distinct feature names, "
                             f"got {len(names)} ({len(set(names))} distinct)")
    return names


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def md5_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()  # noqa: S324 - artifact identity, not security
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def git_head_short() -> str:
    try:
        done = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=REPO_ROOT, capture_output=True, text=True,
                              check=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return done.stdout.strip() or "unknown"


def write_metrics(out: Path, metrics: dict) -> Path:
    """Serialize `metrics.json`. The ONE writer, so a test can exercise it.

    `json.dumps` renders a float through `repr`, which round-trips a float64
    exactly: a validation mean separated from another only in the sixth
    decimal survives the file. Nothing here rounds.
    """
    path = Path(out) / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2))
    return path


def read_cache_meta(path: Path) -> dict:
    """The stats cache's ``_meta`` rows, read-only."""
    try:
        conn = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise RuntimeError(f"cannot open stats cache {path}: {exc}") from exc
    try:
        rows = conn.execute("SELECT key, value FROM _meta").fetchall()
    except sqlite3.Error as exc:
        raise RuntimeError(
            f"stats cache {path} exposes no _meta table: {exc}") from exc
    finally:
        conn.close()
    return {str(key): value for key, value in rows}


def resolve_stats_cache(role: str, explicit: Path | None,
                        frame_alias_version: str | None) -> dict:
    """Resolve the cache by manifest role and refuse an identity mismatch.

    The frame and the cache must declare the same venue alias version: a
    checkpoint trained on i7 rows against a cache with different venue
    identity would carry venue features the serving cache cannot reproduce.
    """
    from artifacts import artifact_path  # local: keeps import cost off tests

    path = Path(artifact_path(role, explicit))
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise RuntimeError(
            f"stats cache role {role!r} resolves to {path}, which does not "
            "exist")
    meta = read_cache_meta(path)
    cache_alias = meta.get("venue_alias_version")
    if cache_alias != frame_alias_version:
        raise RuntimeError(
            "stats cache identity does not match the training frame: cache "
            f"{path} declares venue_alias_version={cache_alias!r}, the frame "
            f"declares venue_alias_version={frame_alias_version!r}; refusing "
            "to train")
    try:
        recorded = path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        recorded = path.resolve().as_posix()
    return {
        "role": role,
        "path": recorded,
        "md5": md5_file(path),
        "venue_alias_version": cache_alias,
        "same_day_order_version": meta.get("same_day_order_version"),
    }


def load_split(name: str, data_dir: Path = DATA,
               version: str | None = None) -> pd.DataFrame:
    cols = (["innings_id", "batter_id", "bowler_id", "ball_outcome"]
            + EB_BAT_COLS + EB_BOWL_COLS + VENUE_COLS + CTX_COLS + STATE_COLS)
    path = split_path(data_dir, resolve_frame_version(data_dir, version), name)
    if DATE_COL in set(pq.ParquetFile(path).schema_arrow.names):
        cols = cols + [DATE_COL]
    df = pd.read_parquet(path, columns=cols)
    df["y"] = df["ball_outcome"].map(CLASS_MAPPING).astype(np.int64)
    return df


def split_record(path: Path, df: pd.DataFrame) -> dict:
    """Identity of one parquet this run actually read."""
    dates = df[DATE_COL] if DATE_COL in df.columns else None
    return {
        "path": Path(path).as_posix(),
        "md5": md5_file(path),
        "n_rows": int(len(df)),
        "match_date_min": None if dates is None else str(dates.min()),
        "match_date_max": None if dates is None else str(dates.max()),
    }


def build_features(df: pd.DataFrame) -> np.ndarray:
    lo, hi = STATE_NORMALISERS["run_rate_required_clip"]
    state = np.stack([
        (df["balls_remaining"].to_numpy(np.float32)
         / STATE_NORMALISERS["balls_remaining"]),
        df["score"].to_numpy(np.float32) / STATE_NORMALISERS["score"],
        df["run_rate"].to_numpy(np.float32) / STATE_NORMALISERS["run_rate"],
        (np.clip(df["run_rate_required"].to_numpy(np.float32), lo, hi)
         / STATE_NORMALISERS["run_rate_required"]),
    ], axis=1)
    return np.hstack([
        df[EB_BAT_COLS + EB_BOWL_COLS + VENUE_COLS].to_numpy(np.float32),
        make_ctx(df, venue=False), state,
    ])


def build_innings(df: pd.DataFrame):
    """Per-innings row-index lists, preserving parquet (ball) order."""
    return list(df.groupby("innings_id", sort=False).indices.values())


def aligned_history(y, batter_ids, bowler_ids, innings_index_lists):
    """Participant-aligned history inputs, one pair per row (D3.4).

    ``prev_bat[r]`` is the outcome class of the last EARLIER row of the same
    innings with the same ``batter_id``; ``prev_bowl[r]`` the same for
    ``bowler_id``. ``BOS`` where there is no such row — the batter's or
    bowler's first delivery of the innings.

    Rows are walked in parquet (ball) order within each innings, so extras
    (wides, no-balls) are ordinary rows and count: a wide by the same bowler
    is that bowler's previous outcome for the next legal ball.

    Pure, numpy-only and index-aligned to the split's rows, so the unit test
    can assert exact vectors on a hand-built innings and the trainer can
    compute the arrays once per split.
    """
    y = np.asarray(y)
    batter_ids = np.asarray(batter_ids)
    bowler_ids = np.asarray(bowler_ids)
    n = len(y)
    if len(batter_ids) != n or len(bowler_ids) != n:
        raise ValueError("y, batter_ids and bowler_ids must be row-aligned")
    prev_bat = np.full(n, BOS, dtype=np.int64)
    prev_bowl = np.full(n, BOS, dtype=np.int64)
    for index_list in innings_index_lists:
        last_bat: dict = {}
        last_bowl: dict = {}
        for row in index_list:
            bat, bowl = batter_ids[row], bowler_ids[row]
            if bat in last_bat:
                prev_bat[row] = last_bat[bat]
            if bowl in last_bowl:
                prev_bowl[row] = last_bowl[bowl]
            # Updated AFTER the row is read: the outcome of ball r is never
            # an input to ball r (invariant 2, temporal integrity).
            last_bat[bat] = y[row]
            last_bowl[bowl] = y[row]
    return prev_bat, prev_bowl


def attention_mask(arm: str, k, batter, bowler, pad_mask) -> torch.Tensor:
    """Boolean ``(B, L, L)`` attention set: True where query i may read key j.

    D3.5. Every set is causal (j <= i), excludes padded keys, and always
    contains the target itself, so no query row is fully masked (which would
    make softmax produce NaN).

      * ``recency`` — W_k(i) = {j : i-k <= j <= i}; ``k`` may be ``"unr"``
        (or ``None``) for the whole innings prefix, and ``k = 0`` leaves the
        target alone;
      * ``same_entity`` — {j in W_k(i) : batter[j] == batter[i] or
        bowler[j] == bowler[i]} union {i};
      * anything else (``aligned_hist_rf``, and the standard-wiring biased
        arms) — the full causal prefix.

    k counts delivery ROWS inclusive of extras, because the frame's row order
    is the delivery order.
    """
    B, L = pad_mask.shape
    device = pad_mask.device
    pos = torch.arange(L, device=device)
    lag = pos.view(-1, 1) - pos.view(1, -1)  # (L, L): i - j
    allowed = lag >= 0  # causal prefix
    if arm in ARMS_NEEDING_K and k is not None and k != "unr":
        allowed = allowed & (lag <= int(k))
    allowed = allowed.unsqueeze(0).expand(B, L, L)
    if arm == "same_entity":
        if batter is None or bowler is None:
            raise ValueError("same_entity needs batter/bowler id tensors")
        same = ((batter.unsqueeze(2) == batter.unsqueeze(1))
                | (bowler.unsqueeze(2) == bowler.unsqueeze(1)))
        allowed = allowed & same
    allowed = allowed & ~pad_mask.unsqueeze(1)  # padded keys are never read
    diag = torch.eye(L, dtype=torch.bool, device=device).unsqueeze(0)
    return allowed | diag


def attention_set(arm: str, k, target: int, batter=None, bowler=None,
                  length: int | None = None) -> set[int]:
    """The rows query ``target`` may attend to, as a set of row positions.

    Positions are WITHIN-INNINGS (the padded sequence index), matching
    `attention_mask`. `batter`/`bowler` are 1-D per-row integer codes of the
    innings; `length` defaults to their length.

    The token-MLP arms have no attention at all and `no_attention` is masked
    to its own diagonal, so both return ``{target}``.
    """
    if length is None:
        if batter is None:
            raise ValueError("attention_set needs `length` or id arrays")
        length = len(batter)
    if ARM_WIRING[arm] in ("token_mlp", "recurrent") or arm == "no_attention":
        return {int(target)}
    pad = torch.zeros(1, length, dtype=torch.bool)

    def ids(values):
        if values is None:
            return None
        return torch.as_tensor(np.asarray(values, dtype=np.int64)).view(1, -1)

    mask = attention_mask(arm, k, ids(batter), ids(bowler), pad)
    return set(torch.nonzero(mask[0, target]).flatten().tolist())


def history_source_rows(arm: str, target: int, batter=None,
                        bowler=None) -> set[int]:
    """Rows whose OWN outcome feeds ``target``'s history input.

    Innings-previous history reads row ``target - 1``; participant-aligned
    history reads the last earlier same-batter row and the last earlier
    same-bowler row (empty where the target is that participant's first
    delivery of the innings). An arm with no history input reads none.
    """
    history = ARM_HISTORY[arm]
    if history == "innings_previous":
        return {int(target) - 1} if target > 0 else set()
    if history != "participant_aligned":
        return set()
    if batter is None or bowler is None:
        raise ValueError(f"arm {arm!r} needs batter/bowler ids for S(i)")
    out: set[int] = set()
    for values in (np.asarray(batter), np.asarray(bowler)):
        earlier = [j for j in range(int(target)) if values[j] == values[target]]
        if earlier:
            out.add(int(earlier[-1]))
    return out


def dependency_set(arm: str, k, target: int, batter=None, bowler=None,
                   length: int | None = None) -> set[int]:
    """S(i) = attention_set(i) union {history-source rows of i}.

    The acceptance file's dependency set (arm register, "Dependency set
    S(i)"), shared by the D3.7 invariance tests and
    `scripts/sequence_track/ownership_dependency_test.py` so the script and
    the tests cannot disagree about what a relay-free arm is allowed to read.

    Under the own-outcome key construction (`ARM_KEY_CONSTRUCTION`) this set
    is a superset of both dependencies of a relay-free target:

      * features of row j reach i only through e_j, so only for
        j in attention_set(i);
      * the outcome of row j reaches i either as own_out_emb(y_j) in a key
        (j in attention_set(i), j != i) or through hist(i) (j a history
        source). The target's own y_i reaches nothing: e_i^self carries
        hist(i), not y_i, and the own-outcome key at position i is masked
        off the diagonal.
    """
    return (attention_set(arm, k, target, batter, bowler, length)
            | history_source_rows(arm, target, batter, bowler))


def alibi_slopes(heads: int) -> torch.Tensor:
    """m_h = 2^(-8h/H), h = 1..H — the registered fixed-decay slopes."""
    h = torch.arange(1, heads + 1, dtype=torch.float32)
    return torch.pow(2.0, -8.0 * h / heads)


def alibi_bias(length: int, heads: int, device=None) -> torch.Tensor:
    """``(1, H, L, L)`` additive bias -m_h * (i - j), zero above the diagonal.

    Entries with j > i are masked out by the attention set, so their value is
    immaterial; they are clamped to 0 rather than left positive.
    """
    pos = torch.arange(length, device=device)
    lag = (pos.view(-1, 1) - pos.view(1, -1)).clamp(min=0).to(torch.float32)
    slopes = alibi_slopes(heads).to(device=device)
    return -(slopes.view(1, heads, 1, 1) * lag.view(1, 1, length, length))


class TokenMLPBlock(nn.Module):
    """Residual token-local feed-forward block with no sequence inputs."""

    def __init__(self, dmodel: int):
        super().__init__()
        self.norm = nn.LayerNorm(dmodel)
        self.linear1 = nn.Linear(dmodel, 2 * dmodel)
        self.linear2 = nn.Linear(2 * dmodel, dmodel)
        self.dropout = nn.Dropout(0.1)

    def forward(self, value):
        hidden = nn.functional.gelu(self.linear1(self.norm(value)))
        return value + self.dropout(self.linear2(self.dropout(hidden)))


def _masked_attention(query, key, value, allowed, bias, heads: int,
                      dropout: nn.Dropout):
    """Multi-head attention with a boolean set mask and an additive bias.

    ``allowed`` is ``(B, Lq, Lk)`` (broadcast over heads), ``bias`` is
    ``(B, H, Lq, Lk)`` / ``(1, H, Lq, Lk)`` or None. Written with plain matmul,
    softmax and ``masked_fill`` so it runs on both MPS and CPU.

    The key/value length may differ from the query length: the relay-free
    wiring attends over ``2L`` keys (the own-outcome tokens of the earlier
    rows, then the self tokens on the diagonal).
    """
    B, Lq, D = query.shape
    Lk = key.shape[1]
    dh = D // heads

    def split(t, length):
        return t.view(B, length, heads, dh).transpose(1, 2)  # (B, H, len, dh)

    q = split(query, Lq)
    k, v = split(key, Lk), split(value, Lk)
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(dh)
    if bias is not None:
        scores = scores + bias
    scores = scores.masked_fill(~allowed.unsqueeze(1), float("-inf"))
    weights = dropout(torch.softmax(scores, dim=-1))
    out = torch.matmul(weights, v)  # (B, H, Lq, dh)
    return out.transpose(1, 2).contiguous().view(B, Lq, D)


class _FeedForward(nn.Module):
    """Pre-norm ReLU feed-forward block, 2*d hidden, dropout 0.1.

    The same shape AND activation as `nn.TransformerEncoderLayer(
    norm_first=True, dim_feedforward=2*d, dropout=0.1)`'s FF half (whose
    default activation is ReLU), so the cross-wiring contrasts
    `aligned_hist_rf - aligned_hist` (D3.8) and `fixed_decay - full` carry no
    activation difference. (`TokenMLPBlock`, the stage 1 `mlp` arm, keeps its
    GELU untouched.) Orchestrator edit 2026-09-11 after review.
    """

    def __init__(self, dmodel: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dmodel)
        self.linear1 = nn.Linear(dmodel, 2 * dmodel)
        self.linear2 = nn.Linear(2 * dmodel, dmodel)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, value):
        hidden = self.dropout(nn.functional.relu(self.linear1(
            self.norm(value))))
        return value + self.dropout2(self.linear2(hidden))


class BiasedEncoderLayer(nn.Module):
    """Self-attending pre-norm layer with an additive attention bias.

    Used by `fixed_decay` and `fox` (D3.9). Queries, keys and values all come
    from the running residual stream, just as `nn.TransformerEncoderLayer`
    does;
    the only addition is the per-head ``(B, H, L, L)`` bias added to the scaled
    scores before the softmax.
    """

    def __init__(self, dmodel: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dmodel)
        self.qkv = nn.Linear(dmodel, 3 * dmodel)
        self.out_proj = nn.Linear(dmodel, dmodel)
        self.attn_dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.ff = _FeedForward(dmodel, dropout)

    def forward(self, h, allowed, bias):
        normed = self.norm1(h)
        q, k, v = self.qkv(normed).chunk(3, dim=-1)
        attn = _masked_attention(q, k, v, allowed, bias, self.heads,
                                 self.attn_dropout)
        return self.ff(h + self.dropout1(self.out_proj(attn)))


class RelayFreeLayer(nn.Module):
    """Relay-free pre-norm layer: keys and values from the LAYER-ZERO tokens.

    Arm register, "Relay-free wiring" (D3.6/D3.8, certified in D7): at every
    layer the keys and values are linear projections of e_j, the layer-zero
    token embedding, while the query comes from the running residual stream
    h_i. Because i is always in its own attention set, h_i is a function of
    {e_j : j in attention_set(i)} alone — no information can relay in through
    a neighbour's hidden state.

    Since the own-outcome key construction (handoff § 3.1) an earlier row and
    the target itself contribute DIFFERENT tokens, so `tokens` is the
    concatenation ``[own-outcome tokens ; self tokens]`` of length ``2L`` and
    `allowed` is the matching ``(B, L, 2L)`` set: the first half is masked to
    j < i and the second half to j == i. `LayerNorm`, the projections and the
    attention are all per-token, so nothing about this layer needs to know
    which half a key came from.
    """

    def __init__(self, dmodel: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.norm_q = nn.LayerNorm(dmodel)   # normalises the residual stream
        self.norm_kv = nn.LayerNorm(dmodel)  # normalises the layer-zero tokens
        self.q_proj = nn.Linear(dmodel, dmodel)
        self.k_proj = nn.Linear(dmodel, dmodel)
        self.v_proj = nn.Linear(dmodel, dmodel)
        self.out_proj = nn.Linear(dmodel, dmodel)
        self.attn_dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.ff = _FeedForward(dmodel, dropout)

    def forward(self, h, tokens, allowed):
        kv = self.norm_kv(tokens)
        attn = _masked_attention(self.q_proj(self.norm_q(h)), self.k_proj(kv),
                                 self.v_proj(kv), allowed, None, self.heads,
                                 self.attn_dropout)
        return self.ff(h + self.dropout1(self.out_proj(attn)))


class T1Model(nn.Module):
    def __init__(self, n_feats: int, dmodel: int, layers: int, heads: int,
                 aux_sizes: dict | None = None, arm: str = "full",
                 k=None, dropout: float = 0.1):
        super().__init__()
        if arm not in ALL_ARMS:
            raise ValueError(f"unknown T1 ablation arm: {arm}")
        self.arm = arm
        self.wiring = ARM_WIRING[arm]
        self.history_input = ARM_HISTORY[arm]
        self.key_construction = ARM_KEY_CONSTRUCTION[arm]
        self.bias_kind = ARM_BIAS[arm]
        self.has_pos_emb = ARM_POS_EMB[arm]
        self.heads = heads
        self.n_layers = layers
        self.residual = arm in ARMS_NEEDING_BASE_LOGITS
        if (arm in ARMS_NEEDING_K) != (k is not None):
            verb = "requires" if arm in ARMS_NEEDING_K else "does not accept"
            raise ValueError(f"arm {arm!r} {verb} a window k (got {k!r})")
        self.k = k

        # D3.10 — the recurrent arms live in their own module and own their
        # whole token pathway, so nothing else here is built for them.
        if self.wiring == "recurrent":
            if aux_sizes:
                raise ValueError("recurrent arms carry no aux heads")
            try:
                from sequence_track.recurrent_arms import (  # noqa: PLC0415
                    build_recurrent_model)
            except ImportError as exc:  # pragma: no cover - D3.10 module
                raise RuntimeError(
                    f"arm {arm!r} needs scripts/sequence_track/"
                    f"recurrent_arms.py: {exc}") from exc
            self.recurrent = build_recurrent_model(arm, n_feats, dmodel,
                                                   dropout)
            self.aux_heads = nn.ModuleDict()
            return

        self.feat_proj = nn.Linear(n_feats, dmodel)
        if self.wiring == "token_mlp":
            # Two token-local FF blocks per transformer layer approximately
            # match the full arm's parameter budget without sequence access.
            self.token_mlp = nn.Sequential(*[
                TokenMLPBlock(dmodel) for _ in range(2 * layers)])
        else:
            if self.history_input == "participant_aligned":
                # D3.4 — two separate embeddings REPLACING out_emb(prev_y).
                self.bat_emb = nn.Embedding(7, dmodel)   # 6 classes + BOS
                self.bowl_emb = nn.Embedding(7, dmodel)
            else:
                self.out_emb = nn.Embedding(7, dmodel)  # 6 classes + BOS
            if self.has_pos_emb:
                self.pos_emb = nn.Embedding(200, dmodel)
            if self.wiring == "standard" and self.bias_kind == "none":
                layer = nn.TransformerEncoderLayer(
                    d_model=dmodel, nhead=heads, dim_feedforward=2 * dmodel,
                    dropout=0.1, batch_first=True, norm_first=True)
                self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
            elif self.wiring == "standard":
                self.layers = nn.ModuleList([
                    BiasedEncoderLayer(dmodel, heads, dropout)
                    for _ in range(layers)])
            else:
                self.layers = nn.ModuleList([
                    RelayFreeLayer(dmodel, heads, dropout)
                    for _ in range(layers)])
        self.head = nn.Linear(dmodel, 6)
        self.aux_heads = nn.ModuleDict(
            {t: nn.Linear(dmodel, n) for t, n in (aux_sizes or {}).items()})
        if self.residual:
            # D3.7 — zero-initialised head, so at initialisation the residual
            # is identically zero and p == p_base exactly.
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        # The FoX gates are created LAST and initialised deterministically, so
        # every other parameter of `fox` draws from the same RNG stream as
        # `fixed_decay`'s and the two arms differ only in the gate (D3.9).
        if self.bias_kind == "fox":
            self.fox_gates = nn.ModuleList([nn.Linear(dmodel, heads)
                                            for _ in range(layers)])
            for gate in self.fox_gates:
                nn.init.zeros_(gate.weight)
                nn.init.constant_(gate.bias, FOX_GATE_BIAS_INIT)
        # Created LAST, after every parameter that existed before the § 3.1
        # own-outcome redesign, so the RNG stream of all of those is
        # untouched by the addition (the stage 1 arms are relay-free-free, so
        # they never reach this branch at all).
        if self.key_construction == "own_outcome":
            self.own_out_emb = nn.Embedding(N_OUTCOME_CLASSES, dmodel)

    # --------------------------------------------------------------- biases

    def _bias(self, layer_index: int, tokens, length: int):
        """The additive ``(*, H, L, L)`` attention bias for one layer."""
        if self.bias_kind == "alibi":
            # No learned parameters: -m_h * (i - j) with the ALiBi slopes.
            return alibi_bias(length, self.heads, tokens.device)
        # FoX (Lin et al. 2025): bias[i, j] = sum_{t=j+1..i}
        # log sigma(g_h(e_t)),
        # computed as C[i] - C[j] with C the cumulative sum of the per-head
        # log-gate over the sequence. g_h is a per-layer per-head scalar linear
        # function of the KEY token's layer-zero embedding.
        gate = nn.functional.logsigmoid(self.fox_gates[layer_index](tokens))
        cumulative = torch.cumsum(gate, dim=1).permute(0, 2, 1)  # (B, H, L)
        return cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2)

    # -------------------------------------------------------------- forward

    def forward(self, feats, prev_y, pad_mask, prev_bat=None, prev_bowl=None,
                batter=None, bowler=None, base_logp=None, own_y=None):
        """One forward pass.

        `own_y` is the ``(B, L)`` realised outcome class of each row — the
        batch's TARGETS. Only the relay-free arms read it, and only as the
        own-outcome key/value of an EARLIER row (§ 3.1): row i's own key is
        built from hist(i), so y_i is never readable at row i. Padded rows'
        own outcomes are immaterial because their keys are masked off.
        """
        if self.wiring == "recurrent":
            return self.recurrent(feats, prev_y, pad_mask)

        L = feats.shape[1]
        if self.wiring == "token_mlp":
            h = self.token_mlp(self.feat_proj(feats))
        else:
            if self.arm == "no_history":
                prev_y = torch.full_like(prev_y, BOS)
            x = self.feat_proj(feats)
            if self.history_input == "participant_aligned":
                if prev_bat is None or prev_bowl is None:
                    raise ValueError(
                        f"arm {self.arm!r} needs prev_bat/prev_bowl")
                x = x + self.bat_emb(prev_bat) + self.bowl_emb(prev_bowl)
            else:
                x = x + self.out_emb(prev_y)
            pos = (self.pos_emb(torch.arange(L, device=feats.device))
                   if self.has_pos_emb else None)
            if pos is not None:
                x = x + pos
            # § 3.1 — the relay-free arms additionally build the own-outcome
            # key/value token of every row: feat_proj(feat_j) +
            # own_out_emb(y_j) + pos_j, with NO history input. `x` stays the
            # self token (and the query stream's initial state).
            x_kv = None
            if self.key_construction == "own_outcome":
                if own_y is None:
                    raise ValueError(
                        f"arm {self.arm!r} needs own_y (the realised outcome "
                        "of each row) to build its key/value tokens")
                own = own_y.clamp(0, N_OUTCOME_CLASSES - 1)
                x_kv = self.feat_proj(feats) + self.own_out_emb(own)
                if pos is not None:
                    x_kv = x_kv + pos
            h = self._mix(x, pad_mask, batter, bowler, L, x_kv)
        logits = self.head(h)
        aux = {t: hd(h) for t, hd in self.aux_heads.items()}
        if self.residual:
            if base_logp is None:
                raise ValueError(f"arm {self.arm!r} needs base_logp")
            # p = softmax(log p_base + r_theta); the raw residual is returned
            # so the trainer can add lambda * mean(r^2) (arm register).
            aux["residual"] = logits
            logits = base_logp + logits
        return logits, aux

    def _mix(self, x, pad_mask, batter, bowler, L: int, x_kv=None):
        """Sequence mixing for the attention wirings.

        `x` is the self token of every row (and the initial residual stream);
        `x_kv` the own-outcome key/value token of every row, supplied by the
        relay-free arms only.
        """
        if self.wiring == "standard" and self.bias_kind == "none":
            if self.arm == "no_attention":
                # Preserve the transformer's parameters and token pathway while
                # preventing all cross-token mixing. Each token can still use
                # its immediately preceding outcome via the shifted input.
                attn_mask = ~torch.eye(L, dtype=torch.bool, device=x.device)
            else:
                attn_mask = torch.triu(
                    torch.ones((L, L), dtype=torch.bool, device=x.device),
                    diagonal=1)
            # In the diagonal arm, padding cannot mix into real tokens. Passing
            # a key-padding mask would leave each padded query with no legal
            # key and produce NaNs, so padded outputs are simply computed and
            # discarded.
            key_padding = None if self.arm == "no_attention" else pad_mask
            return self.encoder(x, mask=attn_mask,
                                src_key_padding_mask=key_padding)
        allowed = attention_mask(self.arm, self.k, batter, bowler, pad_mask)
        tokens = x
        if self.wiring == "relay_free":
            # § 3.1 — j != i reads the own-outcome token, j == i the self
            # token. Built once as a 2L-key set: [j < i allowed | diagonal].
            eye = torch.eye(L, dtype=torch.bool,
                            device=x.device).unsqueeze(0).expand_as(allowed)
            allowed = torch.cat([allowed & ~eye, eye], dim=2)
            tokens = torch.cat([x_kv, x], dim=1)
        h = x
        for index, layer in enumerate(self.layers):
            if self.wiring == "standard":
                h = layer(h, allowed, self._bias(index, x, L))
            else:
                # Relay-free: keys/values always from the layer-zero tokens.
                h = layer(h, tokens, allowed)
        return h


class Batch(NamedTuple):
    """One padded batch of innings.

    The first five fields are exactly what stage 1's ``collate`` returned, in
    the same order, so ``collate(...)[:5]`` is the old contract verbatim. The
    stage 2 fields are None unless the caller supplied the source arrays.
    """

    feats: torch.Tensor            # (B, L, F)
    prev_y: torch.Tensor           # (B, L) innings-previous outcome, BOS at 0
    y: torch.Tensor                # (B, L) targets
    pad: torch.Tensor             # (B, L) True where padded
    aux: dict                      # task -> (B, L) aux targets
    batter: torch.Tensor | None = None    # (B, L) int codes, -1 where padded
    bowler: torch.Tensor | None = None    # (B, L) int codes, -1 where padded
    prev_bat: torch.Tensor | None = None  # (B, L) participant-aligned history
    prev_bowl: torch.Tensor | None = None
    base_logp: torch.Tensor | None = None  # (B, L, 6) base log-probabilities


def collate(idx_lists, feats, y, device, aux=None, batter=None, bowler=None,
            prev_bat=None, prev_bowl=None, base_logp=None) -> Batch:
    """Pad one chunk of innings into a `Batch`.

    Stage 2 additions (D3.2): integer batter/bowler codes so the attention
    set masks can be built on device, the participant-aligned history inputs,
    and the per-row base log-probabilities of the residual arms. Every
    addition is row-gathered from a full-split array, so nothing is recomputed
    per batch and nothing about the stage 1 fields changes.
    """
    B = len(idx_lists)
    L = max(len(ix) for ix in idx_lists)
    f = np.zeros((B, L, feats.shape[1]), dtype=np.float32)
    py = np.full((B, L), BOS, dtype=np.int64)
    ty = np.zeros((B, L), dtype=np.int64)
    pad = np.ones((B, L), dtype=bool)
    ax = {t: np.full((B, L), -1, dtype=np.int64) for t in (aux or {})}
    # -1 never matches a real code, so a padded key can never look like the
    # same batter or bowler as a real query.
    bat = None if batter is None else np.full((B, L), -1, dtype=np.int64)
    bwl = None if bowler is None else np.full((B, L), -1, dtype=np.int64)
    pbat = None if prev_bat is None else np.full((B, L), BOS, dtype=np.int64)
    pbowl = None if prev_bowl is None else np.full((B, L), BOS, dtype=np.int64)
    blp = (None if base_logp is None
           else np.zeros((B, L, base_logp.shape[1]), dtype=np.float32))
    for b, ix in enumerate(idx_lists):
        n = len(ix)
        f[b, :n] = feats[ix]
        ty[b, :n] = y[ix]
        py[b, 1:n] = y[ix][:-1]  # outcome enters as NEXT ball's history
        pad[b, :n] = False
        for t in ax:
            ax[t][b, :n] = aux[t][ix]
        if bat is not None:
            bat[b, :n] = batter[ix]
        if bwl is not None:
            bwl[b, :n] = bowler[ix]
        if pbat is not None:
            pbat[b, :n] = prev_bat[ix]
        if pbowl is not None:
            pbowl[b, :n] = prev_bowl[ix]
        if blp is not None:
            blp[b, :n] = base_logp[ix]

    def to_device(array):
        return None if array is None else torch.tensor(array).to(device)

    return Batch(
        feats=torch.tensor(f).to(device), prev_y=torch.tensor(py).to(device),
        y=torch.tensor(ty).to(device), pad=torch.tensor(pad).to(device),
        aux={t: torch.tensor(v).to(device) for t, v in ax.items()},
        batter=to_device(bat), bowler=to_device(bwl), prev_bat=to_device(pbat),
        prev_bowl=to_device(pbowl), base_logp=to_device(blp))


def parse_k(raw):
    """``--k`` as an int >= 0 or the literal string ``"unr"``."""
    if raw is None:
        return None
    if str(raw).strip().lower() == "unr":
        return "unr"
    try:
        value = int(str(raw))
    except ValueError as exc:
        raise ValueError(f"--k must be an int >= 0 or 'unr', got {raw!r}"
                         ) from exc
    if value < 0:
        raise ValueError(f"--k must be >= 0 or 'unr', got {raw!r}")
    return value


def load_base_logits(base_dir: Path, split: str, n_rows: int,
                     parquet_md5: str):
    """The production base log-probabilities for one split (D3.7 / D4.1).

    Refuses anything that is not the split this run actually read: the npz
    carries its own row count and the md5 of the parquet it was built from,
    and both must match the split record `transformer_t1` already computes.
    Returns ``(logp (n_rows, 6) float32, md5 of the npz file)``.
    """
    path = Path(base_dir) / f"{split}.npz"
    if not path.exists():
        raise RuntimeError(
            f"base logits for split {split!r} not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        for key in ("logp", "n_rows", "parquet_md5"):
            if key not in archive:
                raise RuntimeError(f"{path} carries no {key!r} key")
        logp = np.asarray(archive["logp"], dtype=np.float32)
        declared_rows = int(np.asarray(archive["n_rows"]).reshape(-1)[0])
        declared_md5 = str(np.asarray(archive["parquet_md5"]
                                     ).reshape(-1)[0])
    if logp.ndim != 2 or logp.shape[1] != 6:
        raise RuntimeError(
            f"{path} logp must be (n_rows, 6), got {logp.shape}")
    if declared_rows != n_rows or logp.shape[0] != n_rows:
        raise RuntimeError(
            f"{path} declares n_rows={declared_rows} with "
            f"{logp.shape[0]} rows "
            f"of logits, but split {split!r} has {n_rows} rows")
    if declared_md5 != parquet_md5:
        raise RuntimeError(
            f"{path} was built from parquet md5 {declared_md5}, but split "
            f"{split!r} read md5 {parquet_md5}; refusing to train")
    return logp, md5_file(path)


def arm_params_block(arm: str, k, residual_l2, base_logits_md5,
                     n_parameters: int, extra: dict | None = None) -> dict:
    """The `arm_params` block of `metrics.json` (D3.11).

    Every field is derived from the registered arm tables, not from the CLI,
    so the recorded wiring cannot disagree with the model that trained.

    Astra MUST-FIX 8 — `extra` carries a sub-model's own `arm_params` (the
    recurrent arms' `cell` and `simplifications`, which no table knows). Only
    keys the table block does not already define are merged, so the registered
    tables stay authoritative for every shared field.
    """
    block = {
        "arm": arm,
        "k": k,
        "wiring": ARM_WIRING[arm],
        "history_input": ARM_HISTORY[arm],
        "key_construction": ARM_KEY_CONSTRUCTION[arm],
        "positional_embedding": bool(ARM_POS_EMB[arm]),
        "bias": ARM_BIAS[arm],
        "residual_l2": (float(residual_l2)
                        if arm in ARMS_NEEDING_BASE_LOGITS else None),
        "base_logits_md5": base_logits_md5,
        "n_parameters": int(n_parameters),
    }
    for key, value in (extra or {}).items():
        if key not in block:
            block[key] = value
    return block


def calibration_metrics(probs: np.ndarray, y: np.ndarray,
                        n_bins: int = 15) -> dict:
    """Multiclass Brier and confidence ECE with fixed-width bins."""
    onehot = np.eye(probs.shape[1], dtype=np.float32)[y]
    brier = np.square(probs - onehot).sum(axis=1).mean()
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        selected = (confidence >= lo) & (
            confidence < hi if hi < 1.0 else confidence <= hi)
        if selected.any():
            ece += selected.mean() * abs(
                confidence[selected].mean() - correct[selected].mean())
    return {"brier": round(float(brier), 6),
            "confidence_ece_15": round(float(ece), 6)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dmodel", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"],
                    default="auto")
    ap.add_argument("--arm", choices=list(ALL_ARMS), default="full")
    # Stage 2 (D3.3). `--k` is required for the windowed arms and refused for
    # every other one, so a k that could not have been used cannot be
    # recorded; likewise `--base-logits-dir` for the residual arms.
    ap.add_argument("--k", default=None,
                    help="attention window in delivery rows: an int >= 0 or "
                         "'unr'; required for --arm recency/same_entity and "
                         "refused for every other arm")
    ap.add_argument("--base-logits-dir", type=Path, default=None,
                    help="directory of <split>.npz base log-probabilities; "
                         "required for --arm residual_mlp/residual_t1 and "
                         "refused for every other arm")
    ap.add_argument("--residual-l2", type=float, default=RESIDUAL_L2_DEFAULT,
                    help="lambda of the lambda*mean(r^2) shrinkage the "
                         "residual arms add to the loss")
    ap.add_argument("--aux", action="store_true",
                    help="T1.5: multi-task heads on DeepCrease "
                         "shot/line/length/control labels")
    ap.add_argument("--aux-weight", type=float, default=AUX_WEIGHT_DEFAULT)
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--frame-version", default=None,
                    help="parquet stem version; default: the frame's "
                         ".feature_hash 'version' field")
    ap.add_argument("--kit-dir", type=Path, default=KIT)
    ap.add_argument("--no-kit", action="store_true",
                    help="train and score without the frozen eval kit "
                         "(no unseen-pair or probe diagnostics); implied by "
                         "--kit-dir none")
    ap.add_argument("--score-test", action="store_true",
                    help="with --no-kit, also load and score the test split; "
                         "off by default so selection never sees test rows")
    ap.add_argument("--stats-cache-role", default=None,
                    help="manifest role of the stats cache the frame was "
                         "built from; recorded in the training contract and "
                         "checked against the frame's venue alias version")
    ap.add_argument("--stats-cache-path", type=Path, default=None,
                    help="explicit path overriding --stats-cache-role's "
                         "manifest resolution (the role is still recorded)")
    ap.add_argument("--aux-dir", type=Path, default=AUX_DIR)
    ap.add_argument("--save-predictions", action="store_true",
                    help="write row-aligned probabilities for paired audits")
    ap.add_argument("--out", type=Path, default=Path("models/embeddings/t1"))
    args = ap.parse_args()
    use_kit = not (args.no_kit or str(args.kit_dir).lower() == "none")
    score_test = use_kit or args.score_test
    if args.stats_cache_path is not None and not args.stats_cache_role:
        ap.error("--stats-cache-path requires --stats-cache-role")
    # D3.3 — the arm/flag contract, checked before anything is loaded.
    needs_k = args.arm in ARMS_NEEDING_K
    if needs_k and args.k is None:
        ap.error(f"--arm {args.arm} requires --k (an int >= 0 or 'unr')")
    if not needs_k and args.k is not None:
        ap.error(f"--k is not accepted by --arm {args.arm}; it is only "
                 f"meaningful for {', '.join(ARMS_NEEDING_K)}")
    try:
        k_value = parse_k(args.k)
    except ValueError as exc:
        ap.error(str(exc))
    needs_base = args.arm in ARMS_NEEDING_BASE_LOGITS
    if needs_base and args.base_logits_dir is None:
        ap.error(f"--arm {args.arm} requires --base-logits-dir")
    if not needs_base and args.base_logits_dir is not None:
        ap.error(f"--base-logits-dir is not accepted by --arm {args.arm}; it "
                 f"is only meaningful for "
                 f"{', '.join(ARMS_NEEDING_BASE_LOGITS)}")
    # Recorded in `config` as the normalised value the model was built with.
    args.k = k_value

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "auto":
        device = ("mps" if torch.backends.mps.is_available()
                  else "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("--device mps requested but MPS is unavailable")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
    print(f"device={device}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)

    # Frame identity and the stats cache are resolved BEFORE any training:
    # an identity mismatch must refuse to start, not surface after an hour.
    version = resolve_frame_version(args.data_dir, args.frame_version)
    fhash = read_feature_hash(args.data_dir)
    frame_alias = (fhash or {}).get("venue_alias_version")
    declared_semantics = (fhash or {}).get("delivery_semantics")
    cache_block = None
    if args.stats_cache_role:
        # Resolved first so an alias mismatch between cache and frame is
        # reported as such, before the coarser contract rules below.
        cache_block = resolve_stats_cache(args.stats_cache_role,
                                          args.stats_cache_path, frame_alias)
        print(f"stats cache: {cache_block['role']} {cache_block['path']} "
              f"md5 {cache_block['md5']}", flush=True)
    # A contract is written iff the frame declares its delivery semantics.
    # See the module docstring: the two mixed cases are refused rather than
    # allowed to produce a checkpoint the serving guard must reject.
    write_contract = bool(declared_semantics)
    if declared_semantics and not args.stats_cache_role:
        raise RuntimeError(
            f"frame {Path(args.data_dir).as_posix()} declares "
            f"delivery_semantics={declared_semantics!r}, so its checkpoints "
            "carry a training_contract — and a contract must name the stats "
            "cache the frame was materialized from. Pass --stats-cache-role "
            "(e.g. --stats-cache-role stats_cache_i7); refusing to train a "
            "checkpoint whose contract would record a null stats_cache and "
            "be refused at serve time.")
    if not declared_semantics and args.stats_cache_role:
        raise RuntimeError(
            f"frame {Path(args.data_dir).as_posix()} declares no "
            "delivery_semantics, so no training_contract can be written and "
            f"--stats-cache-role {args.stats_cache_role!r} has nothing to be "
            "recorded in. Drop the flag to train a legacy-route checkpoint, "
            "or train on a frame that declares its delivery semantics.")
    if not write_contract:
        print(
            "WARNING: frame "
            f"{Path(args.data_dir).as_posix()} declares no "
            "delivery_semantics; this checkpoint's metrics.json will carry "
            "NO training_contract block. It can only be served through "
            "sim_t1.py's legacy route: config.data_dir must contain "
            "'xgb_data_v3' and the serving stats cache must declare no "
            "venue_alias_version. To produce a contracted checkpoint, train "
            "on a frame that declares delivery_semantics (e.g. "
            "data/xgb_data_i7) with --stats-cache-role.", flush=True)

    print("loading splits...", flush=True)
    split_names = (["train", "validation", "test"] if score_test
                   else ["train", "validation"])
    frames = {name: load_split(name, args.data_dir, version)
              for name in split_names}
    split_files = {name: split_record(split_path(args.data_dir, version, name),
                                      frame)
                   for name, frame in frames.items()}
    train, val = frames["train"], frames["validation"]
    test = frames.get("test")
    F_tr, F_va = build_features(train), build_features(val)
    y_tr, y_va = train["y"].to_numpy(), val["y"].to_numpy()
    inn_tr, inn_va = build_innings(train), build_innings(val)
    F_te = y_te = inn_te = None
    if test is not None:
        F_te, y_te, inn_te = (build_features(test), test["y"].to_numpy(),
                              build_innings(test))
    # --- stage 2 per-split inputs (D3.2) ---------------------------------
    # Built once per split and row-gathered by `collate`. Only what the arm
    # actually reads is built, so a stage 1 arm's run does no extra work.
    wants_ids = ARM_WIRING[args.arm] == "relay_free"
    wants_aligned = ARM_HISTORY[args.arm] == "participant_aligned"
    extras: dict[str, dict] = {}
    for name, frame in frames.items():
        block: dict = {}
        if wants_ids:
            # Factorised per split; only equality between rows of one innings
            # is ever asked, so a per-split coding is sufficient.
            block["batter"] = pd.factorize(frame["batter_id"])[0].astype(
                np.int64)
            block["bowler"] = pd.factorize(frame["bowler_id"])[0].astype(
                np.int64)
        if wants_aligned:
            prev_bat, prev_bowl = aligned_history(
                frame["y"].to_numpy(), frame["batter_id"].to_numpy(),
                frame["bowler_id"].to_numpy(), build_innings(frame))
            block["prev_bat"], block["prev_bowl"] = prev_bat, prev_bowl
        extras[name] = block

    base_logits_md5 = None
    base_logits_by_split = None
    if args.arm in ARMS_NEEDING_BASE_LOGITS:
        per_split = {}
        base_logits_by_split = per_split
        for name, frame in frames.items():
            logp, digest = load_base_logits(
                args.base_logits_dir, name, len(frame),
                split_files[name]["md5"])
            extras[name]["base_logp"] = logp
            per_split[name] = digest
        # One string, so a driver's reuse check can compare a single field;
        # the per-split digests are recorded alongside.
        base_logits_md5 = hashlib.md5(  # noqa: S324 - artifact identity
            "\n".join(f"{s}={d}" for s, d in sorted(per_split.items()))
            .encode("utf-8")).hexdigest()
        print(f"base logits: {per_split}", flush=True)

    print(f"innings: train {len(inn_tr)}, val {len(inn_va)}, "
          f"test {'-' if inn_te is None else len(inn_te)}; "
          f"feats {F_tr.shape[1]}", flush=True)
    if F_tr.shape[1] != N_FEATS:
        raise RuntimeError(f"expected {N_FEATS} features, got {F_tr.shape[1]}")

    aux_tr = aux_va = aux_te = None
    aux_sizes = None
    vocabs = None
    if args.aux:
        aux_tr, vocabs = load_aux("train", len(train), None, args.aux_dir)
        aux_va, _ = load_aux("validation", len(val), vocabs, args.aux_dir)
        if test is not None:
            aux_te, _ = load_aux("test", len(test), vocabs, args.aux_dir)
        aux_sizes = {t: len(v) for t, v in vocabs.items()}
        print(f"aux tasks: {aux_sizes}", flush=True)

    model = T1Model(F_tr.shape[1], args.dmodel, args.layers, args.heads,
                    aux_sizes, arm=args.arm, k=args.k).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    def run_model(batch: Batch):
        """One forward pass, feeding whatever the arm's wiring reads."""
        return model(batch.feats, batch.prev_y, batch.pad,
                     prev_bat=batch.prev_bat, prev_bowl=batch.prev_bowl,
                     batter=batch.batter, bowler=batch.bowler,
                     base_logp=batch.base_logp, own_y=batch.y)

    def eval_split(feats, y, innings, df, aux=None, extra=None):
        model.eval()
        probs = np.zeros((len(y), 6), dtype=np.float32)
        aux_hits = {t: [0, 0] for t in (aux or {})}
        with torch.no_grad():
            for s in range(0, len(innings), args.batch):
                chunk = innings[s:s + args.batch]
                batch = collate(chunk, feats, y, device, aux, **(extra or {}))
                ax = batch.aux
                logits, aux_out = run_model(batch)
                p = torch.softmax(logits, dim=-1).cpu().numpy()
                for b, ix in enumerate(chunk):
                    probs[ix] = p[b, :len(ix)]
                for t in aux_hits:
                    tgt = ax[t]
                    sel = tgt >= 0
                    if sel.any():
                        pred = aux_out[t].argmax(-1)
                        aux_hits[t][0] += int((pred[sel] == tgt[sel]).sum())
                        aux_hits[t][1] += int(sel.sum())
        ll = -np.log(np.clip(probs[np.arange(len(y)), y], 1e-15, 1.0))
        aux_acc = {t: round(h / max(n, 1), 4) for t, (h, n) in aux_hits.items()}
        # float64 accumulation: `ll` is float32, and a float32 mean over
        # ~10^5 rows carries ~1e-6 of summation noise — the same decimal
        # that separates seeds. Rounded reporting hid this; full-precision
        # seed selection must not depend on it.
        return probs, float(ll.mean(dtype=np.float64)), ll, aux_acc

    best_val, best_state, bad = np.inf, None, 0
    best_epoch, epochs_run = -1, 0
    order = np.arange(len(inn_tr))
    for epoch in range(args.epochs):
        epochs_run = epoch + 1
        model.train()
        np.random.shuffle(order)
        t0, tot, cnt = time.time(), 0.0, 0
        for s in range(0, len(order), args.batch):
            chunk = [inn_tr[i] for i in order[s:s + args.batch]]
            batch = collate(chunk, F_tr, y_tr, device, aux_tr,
                            **extras["train"])
            ty, pad, ax = batch.y, batch.pad, batch.aux
            opt.zero_grad()
            logits, aux_out = run_model(batch)
            raw = loss_fn(logits.reshape(-1, 6), ty.reshape(-1))
            keep = (~pad).reshape(-1).float()
            loss = (raw * keep).sum() / keep.sum()
            if "residual" in aux_out:
                # lambda * mean(r^2) over REAL tokens only (arm register).
                residual = aux_out.pop("residual")
                penalty = (residual.pow(2).mean(dim=-1).reshape(-1) * keep
                           ).sum() / keep.sum()
                loss = loss + args.residual_l2 * penalty
            for t, tgt in ax.items():
                sel = (tgt >= 0).reshape(-1)
                if sel.any():
                    a = nn.functional.cross_entropy(
                        aux_out[t].reshape(-1, aux_out[t].shape[-1])[sel],
                        tgt.reshape(-1)[sel])
                    loss = loss + args.aux_weight * a
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * keep.sum().item()
            cnt += keep.sum().item()
        _, vll, _, _ = eval_split(F_va, y_va, inn_va, val,
                                  extra=extras["validation"])
        print(f"epoch {epoch}: train_ll={tot/cnt:.4f} val_ll={vll:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if vll < best_val - 1e-5:
            best_val, bad, best_epoch = vll, 0, epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop", flush=True)
                break
    if best_state is None:
        raise RuntimeError("no epoch completed; nothing to save")
    model.load_state_dict(best_state)

    # --- Scoreboard ---------------------------------------------------------
    masks = probes_df = None
    if use_kit:
        masks = np.load(args.kit_dir / "unseen_pair_masks.npz")
        probes_df = pd.read_parquet(args.kit_dir / "probe_labels.parquet")
    n_params = sum(p.numel() for p in model.parameters())
    metrics = {"config": {k: (str(v) if isinstance(v, Path) else v)
                          for k, v in vars(args).items()},
               "n_params": n_params,
               "device": device,
               # D3.11 — the arm's own identity block, separate from the
               # training_contract (whose architecture/optimiser blocks are
               # pinned by the stage 1 contract tests and must not grow).
               # Astra MUST-FIX 8 — a recurrent arm's own `arm_params` (cell,
               # simplifications) is serialized here; it used to be dropped.
               "arm_params": arm_params_block(
                   args.arm, args.k, args.residual_l2, base_logits_md5,
                   n_params,
                   extra=getattr(getattr(model, "recurrent", None),
                                 "arm_params", None))}
    if base_logits_by_split is not None:
        metrics["arm_params"]["base_logits_md5_by_split"] = dict(
            base_logits_by_split)
    scored = [("validation", F_va, y_va, inn_va, val, aux_va)]
    if test is not None:
        scored.append(("test", F_te, y_te, inn_te, test, aux_te))
    for name, feats, y, innings, df, aux in scored:
        probs, mean_ll, ll_vec, aux_acc = eval_split(
            feats, y, innings, df, aux, extra=extras[name])
        if aux_acc:
            metrics[f"{name}_aux_acc"] = aux_acc
        # Full float64 precision: seeds separated in the sixth decimal are
        # separated here too, so seed selection never resolves a real gap as
        # a tie. The historical four-decimal value is kept alongside.
        metrics[f"{name}_ll"] = float(mean_ll)
        metrics[f"{name}_ll_rounded4"] = round(float(mean_ll), 4)
        if use_kit:
            m = masks[name]
            metrics[f"{name}_ll_unseen_pairs"] = round(
                float(ll_vec[m].mean()), 4)
        metrics[f"{name}_calibration"] = calibration_metrics(probs, y)
        if args.save_predictions:
            np.savez_compressed(
                args.out / f"predictions_{name}.npz",
                probs=probs, y=y,
                innings_id=np.asarray(
                    df["innings_id"].astype(str).tolist(), dtype=str))
        if use_kit and name == "validation":
            bc = df["batter_id"].map(probes_df["train_balls_batting"]).fillna(0)
            wc = df["bowler_id"].map(probes_df["train_balls_bowling"]).fillna(0)
            mc = np.minimum(bc.to_numpy(), wc.to_numpy())
            buckets = {}
            for lo, hi, label in [(0, 1, "unknown"), (1, 200, "1-199"),
                                  (200, 1000, "200-999"),
                                  (1000, 5000, "1000-4999"),
                                  (5000, 10**9, "5000+")]:
                sel = (mc >= lo) & (mc < hi)
                if sel.sum():
                    buckets[label] = {"n": int(sel.sum()),
                                      "ll": round(float(ll_vec[sel].mean()), 4)}
            metrics["validation_ll_by_min_train_balls"] = buckets

    # Always built (its length/uniqueness assertion is an invariant of every
    # run); recorded only when this frame gets a contract.
    names_50 = feature_names()
    if write_contract:
        metrics["training_contract"] = {
            "contract_version": CONTRACT_VERSION,
            "frame_dir": Path(args.data_dir).as_posix(),
            "frame_version": version,
            "feature_hash": fhash,
            "delivery_semantics": (fhash or {}).get("delivery_semantics"),
            "venue_alias_version": frame_alias,
            "venue_alias_sha256": (fhash or {}).get("venue_alias_sha256"),
            "split_files": split_files,
            "feature_names": names_50,
            "feature_names_sha256": sha256_text("\n".join(names_50)),
            "state_normalisers": STATE_NORMALISERS,
            "class_mapping": {str(k): v for k, v in CLASS_MAPPING.items()},
            "architecture": {"dmodel": args.dmodel, "layers": args.layers,
                             "heads": args.heads, "arm": args.arm,
                             "n_params": metrics["n_params"]},
            "optimiser": {"lr": args.lr, "batch": args.batch,
                          "epochs": args.epochs,
                          "patience": args.patience,
                          "aux": bool(args.aux),
                          "aux_weight": args.aux_weight},
            "seed": args.seed,
            "best_epoch": best_epoch,
            "epochs_run": epochs_run,
            "device": device,
            # MPS kernels are not guaranteed bit-reproducible across runs or
            # machines, so a rerun may differ in the low decimals.
            "mps_bit_reproducible": False,
            "versions": {"python": platform.python_version(),
                         "torch": torch.__version__, "numpy": np.__version__,
                         "pandas": pd.__version__},
            "git_head_short": git_head_short(),
            "stats_cache": cache_block,
            "kit_used": use_kit,
            "test_split_scored": test is not None,
        }
    print(json.dumps({k: v for k, v in metrics.items()
                      if "ll" in str(k) and k != "training_contract"},
                     indent=2), flush=True)

    torch.save(model.state_dict(), args.out / "model.pt")
    write_metrics(args.out, metrics)
    print(f"saved to {args.out}/", flush=True)


if __name__ == "__main__":
    main()
