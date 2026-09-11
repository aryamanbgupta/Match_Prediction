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


class T1Model(nn.Module):
    def __init__(self, n_feats: int, dmodel: int, layers: int, heads: int,
                 aux_sizes: dict | None = None, arm: str = "full"):
        super().__init__()
        if arm not in {"full", "mlp", "no_attention", "no_history"}:
            raise ValueError(f"unknown T1 ablation arm: {arm}")
        self.arm = arm
        self.feat_proj = nn.Linear(n_feats, dmodel)
        if arm == "mlp":
            # Two token-local FF blocks per transformer layer approximately
            # match the full arm's parameter budget without sequence access.
            self.token_mlp = nn.Sequential(*[
                TokenMLPBlock(dmodel) for _ in range(2 * layers)])
        else:
            self.out_emb = nn.Embedding(7, dmodel)  # 6 classes + BOS
            self.pos_emb = nn.Embedding(200, dmodel)
            layer = nn.TransformerEncoderLayer(
                d_model=dmodel, nhead=heads, dim_feedforward=2 * dmodel,
                dropout=0.1, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(dmodel, 6)
        self.aux_heads = nn.ModuleDict(
            {t: nn.Linear(dmodel, n) for t, n in (aux_sizes or {}).items()})

    def forward(self, feats, prev_y, pad_mask):
        L = feats.shape[1]
        if self.arm == "mlp":
            h = self.token_mlp(self.feat_proj(feats))
            aux = {t: hd(h) for t, hd in self.aux_heads.items()}
            return self.head(h), aux

        pos = torch.arange(L, device=feats.device)
        if self.arm == "no_history":
            prev_y = torch.full_like(prev_y, BOS)
        x = self.feat_proj(feats) + self.out_emb(prev_y) + self.pos_emb(pos)
        if self.arm == "no_attention":
            # Preserve the transformer's parameters and token pathway while
            # preventing all cross-token mixing. Each token can still use its
            # immediately preceding outcome via the causally shifted input.
            attn_mask = ~torch.eye(L, dtype=torch.bool, device=feats.device)
        else:
            attn_mask = torch.triu(
                torch.ones((L, L), dtype=torch.bool, device=feats.device),
                diagonal=1)
        # In the diagonal arm, padding cannot mix into real tokens. Passing a
        # key-padding mask would leave each padded query with no legal key and
        # produce NaNs, so padded outputs are simply computed and discarded.
        key_padding = None if self.arm == "no_attention" else pad_mask
        h = self.encoder(x, mask=attn_mask,
                         src_key_padding_mask=key_padding)
        aux = {t: hd(h) for t, hd in self.aux_heads.items()}
        return self.head(h), aux


def collate(idx_lists, feats, y, device, aux=None):
    B = len(idx_lists)
    L = max(len(ix) for ix in idx_lists)
    f = np.zeros((B, L, feats.shape[1]), dtype=np.float32)
    py = np.full((B, L), BOS, dtype=np.int64)
    ty = np.zeros((B, L), dtype=np.int64)
    pad = np.ones((B, L), dtype=bool)
    ax = {t: np.full((B, L), -1, dtype=np.int64) for t in (aux or {})}
    for b, ix in enumerate(idx_lists):
        n = len(ix)
        f[b, :n] = feats[ix]
        ty[b, :n] = y[ix]
        py[b, 1:n] = y[ix][:-1]  # outcome enters as NEXT ball's history
        pad[b, :n] = False
        for t in ax:
            ax[t][b, :n] = aux[t][ix]
    return (torch.tensor(f).to(device), torch.tensor(py).to(device),
            torch.tensor(ty).to(device), torch.tensor(pad).to(device),
            {t: torch.tensor(v).to(device) for t, v in ax.items()})


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
    ap.add_argument("--arm", choices=["full", "mlp", "no_attention",
                                      "no_history"], default="full")
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
                    aux_sizes, arm=args.arm).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    def eval_split(feats, y, innings, df, aux=None):
        model.eval()
        probs = np.zeros((len(y), 6), dtype=np.float32)
        aux_hits = {t: [0, 0] for t in (aux or {})}
        with torch.no_grad():
            for s in range(0, len(innings), args.batch):
                chunk = innings[s:s + args.batch]
                f, py, ty, pad, ax = collate(chunk, feats, y, device, aux)
                logits, aux_out = model(f, py, pad)
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
            f, py, ty, pad, ax = collate(chunk, F_tr, y_tr, device, aux_tr)
            opt.zero_grad()
            logits, aux_out = model(f, py, pad)
            raw = loss_fn(logits.reshape(-1, 6), ty.reshape(-1))
            keep = (~pad).reshape(-1).float()
            loss = (raw * keep).sum() / keep.sum()
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
        _, vll, _, _ = eval_split(F_va, y_va, inn_va, val)
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
    metrics = {"config": {k: (str(v) if isinstance(v, Path) else v)
                          for k, v in vars(args).items()},
               "n_params": sum(p.numel() for p in model.parameters()),
               "device": device}
    scored = [("validation", F_va, y_va, inn_va, val, aux_va)]
    if test is not None:
        scored.append(("test", F_te, y_te, inn_te, test, aux_te))
    for name, feats, y, innings, df, aux in scored:
        probs, mean_ll, ll_vec, aux_acc = eval_split(feats, y, innings, df, aux)
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
