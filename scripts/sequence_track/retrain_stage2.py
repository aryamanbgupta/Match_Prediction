#!/usr/bin/env python3
"""Train the sixteen registered stage-2 configurations on the i7 frame.

Sequence track stage 2, deliverable D3 (check 3.12). COPIED from
`scripts/sequence_track/retrain_i7.py` rather than rewritten: the structure,
the refusal logic and the preflight measurement are that file's, so a stage-2
table means exactly what a stage-1 table meant. What is new is only what
stage 2 added to the trainer.

The runner is a launcher and a bookkeeper, never a second implementation of
training: every number it reports is read back out of the `metrics.json` that
`scripts/transformer_t1.py` wrote, and every identity field comes from that
file's `training_contract` and `arm_params` blocks.

What it guarantees, and why each one is here:

  * exactly the seeds the configuration registers, each once. A duplicate or
    an unregistered seed in the config is refused, and `--seeds` may NARROW
    the list (one seed tonight, the rest later) but never widen it: a night
    that quietly adds a seed to one arm and not to its control would turn a
    matched contrast into an unmatched one;
  * the test split is never loaded. `--no-kit` is passed and `--score-test`
    is not, so selection happens on validation rows only;
  * `--save-predictions` is always passed, because D10 computes per-row log
    loss from the saved row-aligned probabilities, not from a scalar;
  * the stats cache resolves through the manifest role and its venue alias
    version must equal the frame's, checked once before the first run and
    again inside every run;
  * a completed checkpoint is never overwritten. It is reused only after its
    contract is re-verified against the EFFECTIVE settings of THIS
    invocation — the config id, the arm and its `arm_params` (k, wiring,
    history input, key construction, bias, residual L2, base-logits md5),
    the seed, the frame dir and version, the frame's declaration and parquet
    identity, the stats-cache md5, the kit and test-split flags, the
    architecture, the optimiser and the CLI overrides recorded in its
    `run_record.json`. Any mismatch refuses and names the field; `--force`
    retrains it instead. Without the `arm_params` check a `same_entity_k6`
    checkpoint sitting in the `same_entity_k30` directory (a mistyped
    `--config-ids`, a rerun after an edited config) would be reused and
    reported as k=30;
  * a reused checkpoint must additionally carry this invocation's
    seed-INDEPENDENT TRAINING SIGNATURE (Astra MUST-FIX 2): one sha256 over
    everything two seeds of one configuration must share to belong in the
    same table — the effective training block, the arm and the whole
    `arm_params` identity (`key_construction` included), the frame identity,
    the stats-cache identity, the base-logits digest and the sha256 of the
    implementation files the arm actually runs (`transformer_t1.py`,
    `embeddings_e1.py` and `artifacts.py`, plus `recurrent_arms.py` for the
    recurrent arms). The field-by-field contract
    checks cannot see an edited model file, so without this an edited
    implementation would silently reuse seeds trained by the old code. The
    signature is written into every `run_record.json`, every `COMPLETE.json`
    and every `summary.yaml`, and a reuse whose stored signature differs is
    refused naming the first differing component. This is what makes the
    two-machine split safe: seed 7 on the laptop and seed 13 on the mini are
    rsynced into one tree, and the consolidating `--consolidate --seeds 7,13`
    invocation verifies that both seeds were produced by the same
    configuration, frame and code before it writes one summary over the two.
    Every component of the signature is a LOGICAL identity — the frame
    directory exactly as the config spells it plus the content hashes of its
    declaration and parquets, the cache's manifest ROLE plus its md5, content
    digests for the base logits and the implementation files — and never a
    resolved absolute path (Astra round-2 MUST-FIX 1). Seed 7 trains from a
    git worktree in which `data/xgb_data_i7` and the cache are SYMLINKS into
    the main checkout, so a signature over resolved physical paths would give
    the two machines different signatures and neither checkout could
    consolidate both. The physical paths are still recorded, beside the
    signature and not inside it;
  * `--consolidate` is verification-and-summary only (Astra round-2
    MUST-FIX 3). It never deletes an artefact and never launches training: an
    incomplete or unverifiable (configuration, seed) refuses and is named, so
    a half-transferred seed 13 is re-queued on the mini that owns it instead
    of being silently retrained on the laptop under the other machine's seed
    assignment;
  * the machine that ran each seed is RECORDED WHILE IT RUNS (Astra round-2
    MUST-FIX 4, D8.8): hostname and machine label, chip, OS and release,
    python and torch versions and the MPS backend, the device, all four
    thread caps as they stood in the environment, the repository root and
    whether the frame, cache and model paths were symlinks. None of it is
    reconstructible after an rsync or an accidental cross-machine recovery,
    so none of it is reconstructed;
  * the frame is MEASURED once, in preflight: the whole `.feature_hash`
    declaration, and the md5, row count and `match_date` range of the train
    and validation parquets (the test split is not among them and is never
    opened). The residual arms' base-logit npz files are measured the same
    way, and each one's recorded parquet md5 must equal the md5 of the split
    the run will actually read;
  * the measurement is then COMPARED against the config's registered
    `provenance:` block (Astra MUST-FIX 2), so a frame or cache that moved
    since the pin refuses before the first launch instead of producing a
    table under a stale identity. A config with no provenance block refuses
    and names the pin. The pin also records a cohort hash; the cohort is
    NOT required to exist here, because the mini has no cohort artifacts, so
    that component is skipped with a recorded note rather than failing. The
    pin's `sources` block IS compared, narrowly: the live sha256 of the files
    this invocation's arms actually train through must equal the hash the pin
    registered for each of them, so the driver establishes agreement with the
    REGISTERED hashes instead of merely hashing whatever is on disk (Astra
    round-1 MUST-FIX 2). A training source the pin never recorded is named in
    the recorded result rather than passing silently, and a source outside
    that closure (an analysis script) is left to `pin_stage2.py --verify`;
  * a run is COMPLETE only when all four of its artefacts exist and are
    readable — `metrics.json`, `model.pt`, `run_record.json` and
    `predictions_validation.npz` (Astra MUST-FIX 3). `model.pt` is READ, not
    merely sized: it must load as a state dict and carry the parameter names
    the arm's own wiring produces (Astra round-1 MUST-FIX 3), and on reuse
    every artefact's size and md5 must still equal what `COMPLETE.json`
    recorded, which is what catches a truncated or half-overwritten rsync. A
    run interrupted
    between any two of them is retried, not refused and not silently
    accepted as complete. After the four are validated, one atomic
    `COMPLETE.json` is written (temp file in the same directory, then
    `os.replace`), carrying the training signature, the seed and the
    artefact list with sizes and hashes. A run that IS complete but whose
    contract or signature disagrees is still a refusal; only `--force`
    retrains it;
  * wall seconds and the child's peak RSS are recorded next to each
    checkpoint, so a resumed invocation still reports what the run cost. The
    RSS is SAMPLED from the child's own process tree while it runs (Astra
    MUST-FIX 4). `resource.getrusage(RUSAGE_CHILDREN).ru_maxrss` is a
    process-wide high-water mark over every child this process has already
    reaped, so a delta across sequential children is not a peak and can be
    zero or negative; records carrying that old measurement are reported
    with `peak_rss_bytes_valid: false` and are never rewritten into
    something that looks correct.

Usage:
    uv run --no-sync python scripts/sequence_track/retrain_stage2.py --dry-run
    uv run --no-sync python scripts/sequence_track/retrain_stage2.py \
        --config-ids mlp,full
    uv run --no-sync python scripts/sequence_track/retrain_stage2.py \
        --epochs 1 --out-root models/embeddings/seq_stage2/smoke
    # after both seed trees are rsynced into one tree (D8.7): verify and
    # rewrite the summaries, training nothing and deleting nothing
    uv run --no-sync python scripts/sequence_track/retrain_stage2.py \
        --consolidate --seeds 7,13
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import transformer_t1 as t1  # noqa: E402
from registered_experiment import match_ids  # noqa: E402

# Astra MUST-FIX 4: the per-child peak RSS is sampled from the child's own
# process tree. `psutil` is a pinned dependency of this repository
# (pyproject.toml, psutil==6.1.0); the `ps` fallback exists so a driver on a
# checkout without it still records a real measurement instead of a wrong one.
try:  # pragma: no cover - import availability is environmental
    import psutil  # noqa: E402
except Exception:  # noqa: BLE001 - any import failure falls back to `ps`
    psutil = None

DEFAULT_CONFIG = REPO / "experiments/configs/seq_stage2_v1.yaml"
TRAINER = REPO / "scripts" / "transformer_t1.py"
SEQ_STAGE2_ROOT = REPO / "models" / "embeddings" / "seq_stage2"
# The seed pool the stage may draw from (D2.1). Night 1 registers (7, 13);
# 29, 42 and 101 may be added later, to whole hypothesis families only.
REGISTERED_SEED_POOL = (7, 13, 29, 42, 101)
SELECTION_SPLIT = "validation"
N_MATCHES_SOURCE = (
    "distinct match ids, the leading '<innings>_' prefix stripped from "
    "innings_id (registered_experiment.match_ids)")
# The only two splits a stage-2 run may read, and the fields that identify
# each one. `path` is measured and reported but deliberately NOT compared:
# it is how the file was spelled on a command line, not what is in it.
CONTRACT_SPLITS = ("train", SELECTION_SPLIT)
SPLIT_IDENTITY_FIELDS = ("md5", "n_rows", "match_date_min", "match_date_max")
# The `arm_params` fields a reused checkpoint must still agree with. The
# block the trainer writes carries more (positional_embedding, parameter
# count, per-split base digests, the xLSTM simplification list); those are
# derived from these or from the arm name, so these are the identity.
# `key_construction` is in the identity (Astra MUST-FIX 2): it is what makes
# a relay-free arm relay-free, and an arm whose key construction changed is a
# different model even when every other field still matches.
ARM_PARAM_FIELDS = ("arm", "k", "wiring", "history_input", "key_construction",
                    "bias", "residual_l2", "base_logits_md5")
CONFIG_REQUIRED_FIELDS = ("id", "arm", "access", "history_input", "wiring",
                          "reference", "tests", "queue_order", "command")
# The four artefacts a finished run leaves behind (Astra MUST-FIX 3). All four
# must exist and be readable before a run counts as complete; anything less is
# retried, never reused and never reported.
RUN_ARTIFACTS = ("metrics.json", "model.pt", "run_record.json",
                 f"predictions_{SELECTION_SPLIT}.npz")
# The atomic completion record, written after the four are validated.
COMPLETION_RECORD = "COMPLETE.json"
# The components of the seed-independent training signature, in the order a
# mismatch is reported (Astra MUST-FIX 2).
SIGNATURE_COMPONENTS = ("config_id", "arm", "arm_params", "training_block",
                        "frame", "stats_cache", "base_logits",
                        "implementation")
# The implementation files each arm actually depends on. `transformer_t1.py`
# is the trainer for every arm; the recurrent arms additionally run
# `recurrent_arms.py`. The driver itself is deliberately NOT in the set: it
# launches training, it does not define it, and hashing it would invalidate
# every seed whenever the bookkeeping changed.
#
# Astra round-1 MUST-FIX 2: the trainer is not the whole implementation. The
# set below is the AST import closure of `transformer_t1.py` and
# `recurrent_arms.py` over `scripts/` — every module in it is imported by the
# training path and changes what training reads or computes:
#   * `embeddings_e1.py` defines the 50-feature contract, the class mapping
#     and the column lists `transformer_t1` imports at module scope, so an
#     edit there changes the training INPUTS with no trainer edit at all;
#   * `artifacts.py` resolves the stats cache by manifest role (the trainer's
#     `resolve_stats_cache` imports it), so it decides which cache file the
#     features are read from.
# A test asserts this set still equals that closure, so a new import cannot
# quietly leave the signature behind.
TRAINER_SOURCE = "scripts/transformer_t1.py"
RECURRENT_SOURCE = "scripts/sequence_track/recurrent_arms.py"
FEATURE_CONTRACT_SOURCE = "scripts/embeddings_e1.py"
ARTIFACT_RESOLVER_SOURCE = "scripts/artifacts.py"
# The closure roots, for the test that re-derives the set by AST.
IMPLEMENTATION_CLOSURE_ROOTS = (TRAINER_SOURCE, RECURRENT_SOURCE)
# The config key `pin_stage2.py --write` authors, and the provenance
# components this driver compares / records-but-skips (Astra MUST-FIX 2).
PROVENANCE_KEY = "provenance"
PROVENANCE_SKIPPED = {
    "cohort": ("the cohort is not on every machine that trains (the mini has "
               "no cohort artifacts), and no training run reads it; it is "
               "verified by pin_stage2.py --verify on the laptop"),
    # Astra round-1 MUST-FIX 2: the training closure of `sources` IS compared
    # against the pin (see `source_problems`). What stays skipped is the rest
    # of the pinned closure — the builders, the pin itself, the analysis
    # scripts — none of which a training run executes.
    "sources_outside_the_training_closure": (
        "the pin hashes 22 stage-2 sources; this driver compares the ones the "
        "selected arms actually train through and leaves the builders, the "
        "pin and the analysis scripts to pin_stage2.py --verify"),
    "runtime": ("recorded by the pin on the machine that pinned; the two "
                "training machines record their OWN runtime and host identity "
                "in every run_record.json (D8.8)"),
    "features": ("the feature list is inside the frame's .feature_hash, "
                 "which IS compared"),
}
# Rendered commands in the config name the interpreter portably; the argv the
# driver actually runs starts with this process's own interpreter.
COMMAND_TEXT_PREFIX = "uv run --no-sync python"


class RetrainError(RuntimeError):
    """The configuration or an existing checkpoint failed a contract."""


class _Absent:
    """A field the contract does not carry at all, printable in a message."""

    def __repr__(self) -> str:
        return "<absent>"


ABSENT = _Absent()


# ------------------------------------------------------------- utilities

def rel(path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def assert_writable(out_root: Path) -> Path:
    """Stage-2 output is confined to the stage-2 embeddings namespace."""
    allowed = SEQ_STAGE2_ROOT.resolve()
    resolved = Path(out_root).resolve()
    if resolved != allowed and allowed not in resolved.parents:
        raise RetrainError(
            f"refusing to write outside {rel(allowed)}: {rel(resolved)}")
    return resolved


def parse_id_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not values:
        raise RetrainError("--config-ids was given no id")
    if len(set(values)) != len(values):
        raise RetrainError(f"--config-ids repeats an id: {values}")
    return values


def parse_seed_list(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    values = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(int(part))
        except ValueError as exc:
            raise RetrainError(f"--seeds: {part!r} is not an integer") from exc
    if not values:
        raise RetrainError("--seeds was given no seed")
    if len(set(values)) != len(values):
        raise RetrainError(f"--seeds repeats a seed: {values}")
    return values


def narrow_seeds(registered: list[int], requested: list[int] | None
                 ) -> list[int]:
    """`--seeds` may narrow a configuration's seed list, never widen it.

    Adding a seed on the command line would produce a run nothing registered,
    reported under the config's hash, and (D2.1) a family whose candidate has
    more seeds than its control.
    """
    if requested is None:
        return list(registered)
    extra = [seed for seed in requested if seed not in registered]
    if extra:
        raise RetrainError(
            f"--seeds {requested} is not a subset of the registered seeds "
            f"{registered}: {extra} would widen the seed list. Seeds are "
            "added by editing the config for a whole hypothesis family "
            "(stage 2 acceptance D2.1), never on the command line")
    return [seed for seed in registered if seed in requested]


# -------------------------------------------------------- config loading

def _check_params(config_path: Path, entry: dict) -> dict:
    """The `params` block of one configuration, checked against the trainer.

    The trainer refuses `--k` for an arm that has no window and refuses a
    residual arm without base logits (D3.3). Those refusals arrive after a
    launch; these arrive before the first one.
    """
    arm = entry["arm"]
    params = entry.get("params") or {}
    if not isinstance(params, dict):
        raise RetrainError(
            f"{rel(config_path)}: {entry['id']}: 'params' must be a mapping")
    unknown = sorted(set(params) - {"k", "base_logits_dir", "residual_l2"})
    if unknown:
        raise RetrainError(
            f"{rel(config_path)}: {entry['id']}: unknown param(s) {unknown}")

    needs_k = arm in t1.ARMS_NEEDING_K
    if needs_k and params.get("k") is None:
        raise RetrainError(
            f"{rel(config_path)}: {entry['id']}: arm {arm!r} requires "
            "params.k (an int >= 0 or 'unr')")
    if not needs_k and params.get("k") is not None:
        raise RetrainError(
            f"{rel(config_path)}: {entry['id']}: params.k is not accepted by "
            f"arm {arm!r}; the trainer refuses --k for it")
    try:
        k_value = t1.parse_k(params.get("k"))
    except ValueError as exc:
        raise RetrainError(
            f"{rel(config_path)}: {entry['id']}: {exc}") from exc

    needs_base = arm in t1.ARMS_NEEDING_BASE_LOGITS
    base_dir = params.get("base_logits_dir")
    if needs_base and not base_dir:
        raise RetrainError(
            f"{rel(config_path)}: {entry['id']}: arm {arm!r} requires "
            "params.base_logits_dir")
    if not needs_base and base_dir:
        raise RetrainError(
            f"{rel(config_path)}: {entry['id']}: params.base_logits_dir is "
            f"not accepted by arm {arm!r}; the trainer refuses it")
    residual_l2 = params.get("residual_l2")
    if not needs_base and residual_l2 is not None:
        raise RetrainError(
            f"{rel(config_path)}: {entry['id']}: params.residual_l2 is "
            f"meaningless for arm {arm!r}; only the residual arms add it to "
            "the loss")
    if needs_base:
        residual_l2 = float(t1.RESIDUAL_L2_DEFAULT if residual_l2 is None
                            else residual_l2)
    return {"k": k_value, "base_logits_dir": base_dir,
            "residual_l2": residual_l2}


def load_config(path: Path) -> dict:
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict):
        raise RetrainError(f"{rel(path)}: config is not a mapping")

    entries = config.get("configurations")
    if not isinstance(entries, list) or not entries:
        raise RetrainError(
            f"{rel(path)}: 'configurations' must be a non-empty list")

    seen: set[str] = set()
    orders: list[int] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise RetrainError(
                f"{rel(path)}: configurations[{index}] is not a mapping")
        missing = [field for field in CONFIG_REQUIRED_FIELDS
                   if entry.get(field) is None]
        if missing:
            raise RetrainError(
                f"{rel(path)}: configurations[{index}] is missing "
                f"{missing}")
        config_id = str(entry["id"])
        if config_id in seen:
            raise RetrainError(f"{rel(path)}: duplicate config id "
                               f"{config_id!r}")
        seen.add(config_id)
        arm = str(entry["arm"])
        if arm not in t1.ALL_ARMS:
            raise RetrainError(
                f"{rel(path)}: {config_id}: unknown arm {arm!r}; the trainer "
                f"accepts {list(t1.ALL_ARMS)}")
        # The wiring and the history input are the trainer's registered
        # tables, not free text: a config that claims relay-free wiring for
        # an arm the trainer wires as standard would mislabel every contrast.
        if str(entry["wiring"]) != t1.ARM_WIRING[arm]:
            raise RetrainError(
                f"{rel(path)}: {config_id}: wiring {entry['wiring']!r} is "
                f"not the trainer's {t1.ARM_WIRING[arm]!r} for arm {arm!r}")
        if str(entry["history_input"]) != t1.ARM_HISTORY[arm]:
            raise RetrainError(
                f"{rel(path)}: {config_id}: history_input "
                f"{entry['history_input']!r} is not the trainer's "
                f"{t1.ARM_HISTORY[arm]!r} for arm {arm!r}")
        entry["_params"] = _check_params(path, entry)
        orders.append(int(entry["queue_order"]))

    if sorted(orders) != list(range(1, len(entries) + 1)):
        raise RetrainError(
            f"{rel(path)}: 'queue_order' must be 1..{len(entries)} exactly "
            f"once each, got {sorted(orders)}")

    training = config.get("training") or {}
    seeds = training.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise RetrainError(
            f"{rel(path)}: 'training.seeds' must be a non-empty list")
    if len(set(seeds)) != len(seeds):
        raise RetrainError(
            f"{rel(path)}: duplicate seed in {seeds}; each registered seed "
            "runs exactly once")
    unregistered = [seed for seed in seeds if seed not in REGISTERED_SEED_POOL]
    if unregistered:
        raise RetrainError(
            f"{rel(path)}: seed(s) {unregistered} are outside the registered "
            f"pool {list(REGISTERED_SEED_POOL)}")
    if training.get("no_kit") is not True:
        raise RetrainError(f"{rel(path)}: 'training.no_kit' must be true")
    if training.get("score_test") is not False:
        raise RetrainError(
            f"{rel(path)}: 'training.score_test' must be false; stage 2 "
            "never loads the test split")
    if training.get("aux") is not False:
        raise RetrainError(f"{rel(path)}: 'training.aux' must be false")
    if training.get("save_predictions") is not True:
        raise RetrainError(
            f"{rel(path)}: 'training.save_predictions' must be true; D10 "
            "reads per-row probabilities, not a scalar")
    for field in ("device", "dmodel", "layers", "heads", "batch", "epochs",
                  "learning_rate", "patience"):
        if training.get(field) is None:
            raise RetrainError(f"{rel(path)}: 'training.{field}' is required")

    data = config.get("data") or {}
    for field in ("directory", "frame_version", "stats_cache_role"):
        if not data.get(field):
            raise RetrainError(f"{rel(path)}: 'data.{field}' is required")
    if not ((config.get("outputs") or {}).get("directory")):
        raise RetrainError(f"{rel(path)}: 'outputs.directory' is required")
    return config


def configurations(config: dict) -> dict:
    """`{config_id: entry}` in registered queue order."""
    return {str(entry["id"]): entry
            for entry in sorted(config["configurations"],
                                key=lambda row: int(row["queue_order"]))}


def entry_for(config: dict, config_id: str) -> dict:
    entries = configurations(config)
    if config_id not in entries:
        raise RetrainError(
            f"config id {config_id!r} is not registered; the config holds "
            f"{list(entries)}")
    return entries[config_id]


# ------------------------------------------------- effective run identity

def expected_arm_params(entry: dict, base_logits_digest: str | None) -> dict:
    """The `arm_params` identity this invocation would produce.

    Derived from the trainer's own registered tables so that the driver's
    expectation and the trainer's record cannot drift apart.
    """
    arm = str(entry["arm"])
    params = entry["_params"]
    return {
        "arm": arm,
        "k": params["k"],
        "wiring": t1.ARM_WIRING[arm],
        "history_input": t1.ARM_HISTORY[arm],
        # Astra MUST-FIX 2: part of the identity, not a derived label.
        "key_construction": t1.ARM_KEY_CONSTRUCTION[arm],
        "bias": t1.ARM_BIAS[arm],
        "residual_l2": params["residual_l2"],
        "base_logits_md5": (base_logits_digest
                            if arm in t1.ARMS_NEEDING_BASE_LOGITS else None),
    }


def effective_settings(config: dict, entry: dict, epochs: int | None = None,
                       overrides: dict | None = None,
                       base_logits_digest: str | None = None) -> dict:
    """What this invocation would actually train, config plus CLI overrides.

    Reuse and reporting both compare a checkpoint's `training_contract` and
    `arm_params` against THIS, never against the config alone: `--epochs`
    changes what a run is, and a run that was launched with an override is a
    different run even though the config file is byte-identical.
    """
    training = config["training"]
    return {
        "config_id": str(entry["id"]),
        "arch": {"dmodel": int(training["dmodel"]),
                 "layers": int(training["layers"]),
                 "heads": int(training["heads"])},
        "optimiser": {
            "lr": float(training["learning_rate"]),
            "batch": int(training["batch"]),
            "epochs": int(training["epochs"] if epochs is None else epochs),
            "patience": int(training["patience"]),
            "aux": bool(training["aux"]),
            "aux_weight": float(
                training.get("aux_weight", t1.AUX_WEIGHT_DEFAULT)),
        },
        "arm_params": expected_arm_params(entry, base_logits_digest),
        "overrides": dict(overrides or {}),
    }


# -------------------------------------- seed-independent training signature
# Astra MUST-FIX 2 / 11. Everything in this section answers one question: do
# two checkpoints belong in the same table? The field-by-field contract checks
# answer it for the recorded settings, but they cannot see an edited model
# file, and they never looked at `key_construction` at all. The signature is
# deliberately INDEPENDENT of the seed, of the output root and of which
# machine ran it, because those are exactly the things that are allowed to
# differ between the two halves of a split night.

def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, default=str,
                      separators=(",", ":"))


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def implementation_sources(arm: str) -> list[str]:
    """The implementation files this arm's training actually depends on.

    Astra round-1 MUST-FIX 2: the trainer's imported feature contract
    (`embeddings_e1.py`) and its manifest resolver (`artifacts.py`) are in the
    set, because both change what training reads without any change to
    `transformer_t1.py` itself. The set is the AST import closure of the
    trainer (and, for a recurrent arm, of `recurrent_arms.py`) over `scripts/`.
    """
    sources = [TRAINER_SOURCE, FEATURE_CONTRACT_SOURCE,
               ARTIFACT_RESOLVER_SOURCE]
    if t1.ARM_WIRING[arm] == "recurrent":
        sources.append(RECURRENT_SOURCE)
    return sources


def implementation_identity(arm: str) -> dict:
    """`{repo-relative path: sha256}` for the arm's implementation files."""
    identity = {}
    for name in implementation_sources(arm):
        path = REPO / name
        if not path.is_file():
            raise RetrainError(
                f"implementation file {name} is missing, so the training "
                "signature cannot be computed")
        identity[name] = t1.sha256_text(path.read_text())
    return identity


def logical_frame_dir(resolved: dict) -> str:
    """The frame directory as the CONFIG spells it — never a resolved path.

    Astra round-2 MUST-FIX 1. `rel()` resolves symlinks, and seed 7 trains
    from a git worktree whose `data/xgb_data_i7` is a symlink into the main
    checkout: resolving it makes the laptop's frame component
    `/Users/.../Match_Prediction/data/xgb_data_i7` and the mini's
    `data/xgb_data_i7`, so neither checkout could consolidate both seeds. The
    configured spelling is identical on every machine, and the frame's real
    identity is carried by the content hashes next to it (`.feature_hash` and
    the two parquets' md5 / row count / date range).
    """
    configured = resolved.get("frame_dir_configured")
    if not configured:
        raise RetrainError(
            "preflight recorded no frame_dir_configured, so the training "
            "signature would have to hash a machine-specific resolved path "
            "(Astra round-2 MUST-FIX 1)")
    return Path(str(configured)).as_posix()


def logical_stats_cache(resolved: dict) -> dict:
    """The cache's manifest ROLE and md5 — never its resolved path.

    Same reason as the frame: the laptop's worktree reaches the cache through
    a symlink. The role is what the config asks for and the md5 is what the
    file is; the path is how one machine spells it.
    """
    cache = resolved["stats_cache"]
    return {"role": str(cache.get("role")), "md5": cache["md5"]}


def signature_components(effective: dict, resolved: dict) -> dict:
    """The named parts of the training signature, each already digested.

    Stored alongside the signature so a mismatch can name WHICH part moved
    rather than only that a hash changed. Every component is machine- and
    checkout-independent (Astra round-2 MUST-FIX 1): logical names plus
    content hashes, no resolved absolute path anywhere.
    """
    arm = str(effective["arm_params"]["arm"])
    frame = {
        "dir_configured": logical_frame_dir(resolved),
        "version": resolved["frame_version"],
        "feature_hash": resolved["feature_hash"],
        "split_files": {
            split: {field: as_n_rows(facts[field])
                    if field == "n_rows" else facts[field]
                    for field in SPLIT_IDENTITY_FIELDS}
            for split, facts in sorted(resolved["split_files"].items())},
    }
    # The base-logits digest is part of identity only for the arms that read
    # it; for every other arm it is None, so a rebuilt base-logits directory
    # does not invalidate the fourteen arms that never opened it.
    base = (effective["arm_params"]["base_logits_md5"]
            if arm in t1.ARMS_NEEDING_BASE_LOGITS else None)
    raw = {
        "config_id": effective["config_id"],
        "arm": arm,
        "arm_params": effective["arm_params"],
        # The effective training block: the architecture and the optimiser as
        # this invocation would run them, `--epochs` override included.
        "training_block": {"arch": effective["arch"],
                           "optimiser": effective["optimiser"]},
        "frame": frame,
        "stats_cache": logical_stats_cache(resolved),
        "base_logits": base,
        "implementation": implementation_identity(arm),
    }
    if sorted(raw) != sorted(SIGNATURE_COMPONENTS):
        raise RetrainError(
            "training signature components drifted from "
            f"{list(SIGNATURE_COMPONENTS)}: {sorted(raw)}")
    return {name: _digest(raw[name]) for name in SIGNATURE_COMPONENTS}


def training_signature(components: dict) -> str:
    """One sha256 over the named components, in the registered order."""
    return hashlib.sha256("\n".join(
        f"{name}={components[name]}" for name in SIGNATURE_COMPONENTS
    ).encode("utf-8")).hexdigest()


def signature_for(effective: dict, resolved: dict) -> dict:
    components = signature_components(effective, resolved)
    return {"training_signature": training_signature(components),
            "training_signature_components": components,
            "training_signature_rule": (
                "sha256 of '<component>=<sha256>' lines in the order "
                f"{list(SIGNATURE_COMPONENTS)}; seed-independent by "
                "construction, so the two halves of a split night can be "
                "verified against each other"),
            "training_signature_sources": implementation_sources(
                str(effective["arm_params"]["arm"])),
            }


def signature_problems(record: dict | None, signature: dict) -> list[str]:
    """Disagreements between a stored run's signature and this invocation's.

    A stored run with NO signature is unverifiable, not compatible: its code
    and frame identity were never recorded, so reuse is refused the same way
    a missing `run_record.json` was always refused.
    """
    want = signature["training_signature"]
    want_parts = signature["training_signature_components"]
    got = (record or {}).get("training_signature")
    if not got:
        return ["run_record.json carries no training_signature, so the code "
                "and frame this checkpoint was trained under cannot be "
                "verified (retrain it with --force)"]
    if got == want:
        return []
    got_parts = (record or {}).get("training_signature_components")
    if isinstance(got_parts, dict):
        for name in SIGNATURE_COMPONENTS:
            if got_parts.get(name, ABSENT) != want_parts[name]:
                return [f"training_signature component {name!r} differs "
                        f"({got_parts.get(name, ABSENT)!r} != "
                        f"{want_parts[name]!r}); this checkpoint was not "
                        "trained by this configuration, frame or code"]
    return [f"training_signature {got!r} != {want!r}"]


# ----------------------------------------------- machine provenance (D8.8)
# Astra round-2 MUST-FIX 4. Captured WHILE the run executes and written into
# `run_record.json`, `COMPLETE.json` and `summary.yaml`. None of it survives
# an rsync into another machine's tree and none of it can be reconstructed
# afterwards: a seed 13 directory sitting on the laptop looks exactly like a
# seed 13 directory the laptop trained unless the mini said so at the time.
# Deliberately NOT part of the training signature — machine is exactly what
# the two halves of a split night are allowed to differ in.

THREAD_CAP_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                   "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def cpu_brand() -> tuple[str | None, str]:
    """(chip, how it was read). `sysctl` on darwin, `platform` elsewhere."""
    if sys.platform == "darwin":
        try:
            brand = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, check=True).stdout.strip()
        except Exception:  # noqa: BLE001 - fall back, never invent
            brand = ""
        if brand:
            return brand, "sysctl machdep.cpu.brand_string"
    brand = platform.processor() or platform.machine()
    return (brand or None), "platform.processor()"


def thread_caps() -> dict:
    """All four caps as they stand in THIS process's environment.

    The laptop queue sets four (D8.5) and the mini sets one; an unset cap is
    recorded as None rather than as the default it silently takes.
    """
    return {name: os.environ.get(name) for name in THREAD_CAP_VARS}


def path_link_facts(paths: dict) -> dict:
    """`{label: {path, is_symlink, resolved}}` for the paths a run reads.

    Astra round-2 MUST-FIX 1/4: the resolved physical path is recorded HERE,
    outside the signature, together with whether the spelling was a symlink.
    Seed 7's worktree reaches the frame, the cache and `models/` through
    symlinks; that fact belongs in the record even though it must not reach
    the signature.
    """
    facts = {}
    for label, path in paths.items():
        if path is None:
            continue
        path = Path(path)
        facts[label] = {
            "path": path.as_posix(),
            "is_symlink": bool(path.is_symlink()),
            "resolved": path.resolve().as_posix(),
        }
    return facts


def machine_provenance(machine: str | None = None,
                       device_requested=None, device_used=None,
                       paths: dict | None = None) -> dict:
    """The host identity of the machine that is running this run (D8.8)."""
    import torch  # noqa: PLC0415 - already imported by the trainer module

    chip, chip_source = cpu_brand()
    hostname = platform.node()
    label = str(machine) if machine else hostname
    return {
        "hostname": hostname,
        "machine": label,
        "machine_label_source": ("--machine" if machine else "platform.node()"),
        "chip": chip,
        "chip_source": chip_source,
        "os": platform.system(),
        "os_release": platform.release(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        # `str()`, because torch.__version__ is a str SUBCLASS that
        # yaml.safe_dump refuses to represent.
        "torch_version": str(torch.__version__),
        "torch_mps_available": bool(torch.backends.mps.is_available()),
        "torch_mps_built": bool(torch.backends.mps.is_built()),
        "device_requested": (None if device_requested is None
                             else str(device_requested)),
        # What the trainer says it actually ran on, read back out of
        # metrics.json; None until the run has written one.
        "device_used": None if device_used is None else str(device_used),
        "thread_caps": thread_caps(),
        "repo_root": str(REPO),
        # A git worktree has a `.git` FILE, not a directory; that is how the
        # laptop's pinned training checkout is distinguishable from the main
        # one after the fact.
        "repo_is_worktree": (REPO / ".git").is_file(),
        "paths": path_link_facts(paths or {}),
        "captured": ("recorded while the run executed; never reconstructed "
                     "after the fact (D8.8, Astra round-2 MUST-FIX 4)"),
    }


# --------------------------------------------------- preflight measurement

def frame_split_facts(frame_dir: Path, version: str) -> dict:
    """Identity of the two parquets a stage-2 run is allowed to read.

    `{split: {path, md5, n_rows, match_date_min, match_date_max}}` for train
    and validation only. The test split has no entry here and is not opened:
    selection must not be able to see a test row, and a measurement is a
    read.
    """
    facts = {}
    for split in CONTRACT_SPLITS:
        path = t1.split_path(Path(frame_dir), version, split)
        if not path.is_file():
            raise RetrainError(
                f"missing {split} parquet {rel(path)}; the frame a checkpoint "
                "is verified against must be on disk")
        try:
            table = pq.read_table(path, columns=[t1.DATE_COL])
        except Exception as exc:  # noqa: BLE001 - any read failure refuses
            raise RetrainError(
                f"{rel(path)}: cannot read column {t1.DATE_COL!r} ({exc})"
            ) from exc
        if table.num_rows == 0:
            raise RetrainError(f"{rel(path)}: no rows")
        bounds = pc.min_max(table.column(t1.DATE_COL)).as_py()
        if bounds.get("min") is None or bounds.get("max") is None:
            raise RetrainError(
                f"{rel(path)}: {t1.DATE_COL} is entirely null, so the split's "
                "date range cannot be established")
        facts[split] = {
            "path": rel(path),
            "md5": t1.md5_file(path),
            "n_rows": int(table.num_rows),
            "match_date_min": str(bounds["min"]),
            "match_date_max": str(bounds["max"]),
        }
    return facts


def base_logits_facts(base_dir: Path, split_files: dict) -> dict:
    """Measure the residual arms' base-logit npz files before any launch.

    The trainer refuses an npz whose recorded parquet md5 is not the md5 of
    the split it read; doing it here too means a stale base-logits build
    fails in seconds rather than after a queue of launches, and gives the
    reuse check one digest to compare. The digest is composed exactly as
    `transformer_t1` composes it, over the splits a stage-2 run reads.
    """
    base_dir = Path(base_dir)
    if not base_dir.is_absolute():
        base_dir = REPO / base_dir
    if not base_dir.is_dir():
        raise RetrainError(
            f"base logits directory {rel(base_dir)} does not exist; the "
            "residual arms cannot be trained without it (D4)")
    per_split = {}
    for split in CONTRACT_SPLITS:
        path = base_dir / f"{split}.npz"
        if not path.is_file():
            raise RetrainError(f"missing base logits {rel(path)}")
        with np.load(path, allow_pickle=False) as archive:
            for key in ("logp", "n_rows", "parquet_md5"):
                if key not in archive:
                    raise RetrainError(f"{rel(path)} carries no {key!r} key")
            declared_rows = int(np.asarray(archive["n_rows"]).reshape(-1)[0])
            declared_md5 = str(
                np.asarray(archive["parquet_md5"]).reshape(-1)[0])
        live = split_files[split]
        if declared_md5 != live["md5"]:
            raise RetrainError(
                f"{rel(path)} was built from parquet md5 {declared_md5}, but "
                f"the {split} split on disk is md5 {live['md5']}; rebuild the "
                "base logits (D4) before training a residual arm")
        if declared_rows != live["n_rows"]:
            raise RetrainError(
                f"{rel(path)} declares {declared_rows} rows, the {split} "
                f"split has {live['n_rows']}")
        per_split[split] = {"path": rel(path), "npz_md5": t1.md5_file(path),
                            "n_rows": declared_rows,
                            "parquet_md5": declared_md5}
    digest = hashlib.md5(  # noqa: S324 - artifact identity, not a secret
        "\n".join(f"{split}={facts['npz_md5']}"
                  for split, facts in sorted(per_split.items()))
        .encode("utf-8")).hexdigest()
    return {"dir": rel(base_dir), "splits": per_split, "digest": digest}


def source_comparison(pinned: dict, training_sources: list[str]) -> dict:
    """Compare the live training sources against the pin's `sources` block.

    Astra round-1 MUST-FIX 2: hashing whatever is on disk establishes that two
    checkpoints agree with EACH OTHER, not that either agrees with the
    REGISTERED source hashes. This is that comparison, and it is deliberately
    narrow — only the files the selected arms train through, so an edited
    analysis script cannot refuse a night.

    Returns `{"problems": [...], "compared": {...}, "not_pinned": [...]}`. A
    source the pin never recorded is NOT a silent pass: it is returned and
    recorded in the summary's `provenance_check`.
    """
    block = pinned.get("sources")
    pinned_hashes = (block or {}).get("source_sha256")
    if not isinstance(pinned_hashes, dict):
        return {"problems": [
            f"provenance.sources.source_sha256 {pinned_hashes!r} is not a "
            "mapping, so the training sources cannot be compared against the "
            "registered hashes (re-pin with pin_stage2.py --write)"],
            "compared": {}, "not_pinned": []}
    problems, compared, not_pinned = [], {}, []
    for name in training_sources:
        path = REPO / name
        if not path.is_file():
            problems.append(
                f"training source {name} is missing from this checkout")
            continue
        live = t1.sha256_text(path.read_text())
        want = pinned_hashes.get(name)
        if want is None:
            not_pinned.append(name)
            continue
        compared[name] = live
        if str(want) != live:
            problems.append(
                f"training source {name} sha256 {live!r} != the hash this "
                f"config pins {str(want)!r}; the code that would train "
                "tonight is not the code the pin registered")
    return {"problems": problems, "compared": compared,
            "not_pinned": sorted(not_pinned)}


def provenance_check(config: dict, config_path: Path, frame_version: str,
                     feature_hash: dict, split_files: dict, cache: dict,
                     base_logits: dict | None,
                     training_sources: list[str] | None = None) -> dict:
    """COMPARE the live measurement against the config's registered pin.

    Astra MUST-FIX 2: preflight used to measure the frame and never look at
    what the configuration says the frame IS, so a moved parquet or a
    rebuilt cache produced a table under a stale identity instead of a
    refusal. `pin_stage2.py --write` authors the `provenance` block; this
    compares the parts a training run depends on and refuses on the first
    disagreement. The cohort is NOT among them (see PROVENANCE_SKIPPED): the
    mini trains without any cohort artifact, so requiring one here would
    make half a split night impossible to start.
    """
    pinned = config.get(PROVENANCE_KEY)
    if not isinstance(pinned, dict) or not pinned:
        raise RetrainError(
            f"{rel(config_path)} carries no {PROVENANCE_KEY!r} block, so the "
            "frame, cache and base logits this night trains on cannot be "
            "compared against anything. Run: uv run --no-sync python "
            "scripts/sequence_track/pin_stage2.py --write")

    problems = []
    frame = pinned.get("frame")
    if not isinstance(frame, dict):
        problems.append(f"provenance.frame {frame!r} is not a mapping")
    else:
        if str(frame.get("version")) != str(frame_version):
            problems.append(
                f"provenance.frame.version {frame.get('version')!r} != the "
                f"frame on disk {frame_version!r}")
        pinned_hash = frame.get("feature_hash")
        if not isinstance(pinned_hash, dict):
            problems.append(
                f"provenance.frame.feature_hash {pinned_hash!r} is not a "
                "mapping")
        else:
            for key in sorted(set(pinned_hash) | set(feature_hash)):
                got = feature_hash.get(key, ABSENT)
                want = pinned_hash.get(key, ABSENT)
                if got != want:
                    problems.append(
                        f"frame .feature_hash {key} {got!r} != pinned "
                        f"{want!r}")
        pinned_splits = frame.get("splits")
        if not isinstance(pinned_splits, dict):
            problems.append(
                f"provenance.frame.splits {pinned_splits!r} is not a mapping")
            pinned_splits = {}
        for split in CONTRACT_SPLITS:
            want = pinned_splits.get(split)
            if not isinstance(want, dict):
                problems.append(
                    f"provenance.frame.splits.{split} {want!r} is not the "
                    "record of a pinned parquet")
                continue
            live = split_files[split]
            for field in SPLIT_IDENTITY_FIELDS:
                pinned_value = want.get(field, ABSENT)
                if field == "n_rows":
                    pinned_value = as_n_rows(pinned_value)
                if live[field] != pinned_value:
                    problems.append(
                        f"{split} parquet {field} {live[field]!r} != pinned "
                        f"{pinned_value!r}")

    pinned_cache = pinned.get("stats_cache")
    if not isinstance(pinned_cache, dict):
        problems.append(
            f"provenance.stats_cache {pinned_cache!r} is not a mapping")
    else:
        if pinned_cache.get("md5") != cache.get("md5"):
            problems.append(
                f"stats cache md5 {cache.get('md5')!r} != pinned "
                f"{pinned_cache.get('md5')!r}")
        if (pinned_cache.get("role")
                and str(pinned_cache["role"]) != str(cache.get("role"))):
            problems.append(
                f"stats cache role {cache.get('role')!r} != pinned "
                f"{pinned_cache.get('role')!r}")

    compared = ["frame.version", "frame.feature_hash", "frame.splits",
                "stats_cache.md5", "stats_cache.role"]
    skipped = dict(PROVENANCE_SKIPPED)
    if base_logits is None:
        # No residual arm is selected, so no base-logit file is opened and
        # there is nothing to compare. Recorded, not silently dropped.
        skipped["base_logits"] = (
            "no residual arm is selected in this invocation, so no "
            "base-logits file is read")
    else:
        pinned_base = pinned.get("base_logits")
        if not isinstance(pinned_base, dict):
            problems.append(
                f"provenance.base_logits {pinned_base!r} is not a mapping")
        else:
            want_digest = pinned_base.get("train_validation_digest")
            if want_digest != base_logits["digest"]:
                problems.append(
                    f"base-logits digest {base_logits['digest']!r} != pinned "
                    f"{want_digest!r}")
            pinned_base_splits = pinned_base.get("splits") or {}
            for split, facts in sorted(base_logits["splits"].items()):
                want = pinned_base_splits.get(split)
                if not isinstance(want, dict):
                    problems.append(
                        f"provenance.base_logits.splits.{split} {want!r} is "
                        "not the record of a pinned npz")
                    continue
                if want.get("npz_md5") != facts["npz_md5"]:
                    problems.append(
                        f"base logits {split} npz md5 {facts['npz_md5']!r} != "
                        f"pinned {want.get('npz_md5')!r}")
            compared += ["base_logits.train_validation_digest",
                         "base_logits.splits.npz_md5"]

    # Astra round-1 MUST-FIX 2: agreement with the REGISTERED source hashes,
    # for the training closure only.
    sources = source_comparison(pinned, list(training_sources or ()))
    problems.extend(sources["problems"])
    if sources["compared"]:
        compared.append("sources.source_sha256 (training closure)")
    if sources["not_pinned"]:
        skipped["sources_not_pinned"] = (
            "the pin records no sha256 for "
            f"{sources['not_pinned']}, so agreement with a registered hash "
            "could not be established for them; re-pin with pin_stage2.py "
            "--write to bring them under the comparison")

    if problems:
        raise RetrainError(
            f"{rel(config_path)}: the files on disk are not the files this "
            "config pins; refusing to train (re-pin with pin_stage2.py "
            "--write only when the move is intended):\n  "
            + "\n  ".join(problems))
    return {
        "pinned_by": pinned.get("pinned_by"),
        "pins_generated_at": pinned.get("pins_generated_at"),
        "config_body_sha256": pinned.get("config_body_sha256"),
        "compared": compared,
        "skipped": skipped,
        "training_sources_compared": sources["compared"],
        "training_sources_not_pinned": sources["not_pinned"],
    }


def preflight(config: dict, entries: list[dict],
              config_path: Path | None = None) -> dict:
    """Resolve and MEASURE the frame, the stats cache and the base logits.

    The measurement is taken exactly once and is what every checkpoint is
    then verified against and what the summaries report, so sixteen configs,
    thirty-two checkpoints and sixteen summaries all describe one frame, read
    at one moment.
    """
    data = config["data"]
    frame_dir = REPO / data["directory"]
    declared = t1.resolve_frame_version(frame_dir)
    if declared != data["frame_version"]:
        raise RetrainError(
            f"{rel(frame_dir)} declares frame version {declared!r}, the "
            f"config registers {data['frame_version']!r}")
    feature_hash = t1.read_feature_hash(frame_dir) or {}
    # Identical to the check every run repeats; done here so a mismatch
    # costs seconds instead of thirty-two launches.
    cache = t1.resolve_stats_cache(data["stats_cache_role"], None,
                                   feature_hash.get("venue_alias_version"))
    split_files = frame_split_facts(frame_dir, declared)

    base_logits = None
    base_dirs = {entry["_params"]["base_logits_dir"] for entry in entries
                 if entry["_params"]["base_logits_dir"]}
    if len(base_dirs) > 1:
        raise RetrainError(
            f"the selected configurations name {len(base_dirs)} different "
            f"base-logits directories {sorted(base_dirs)}; one stage-2 night "
            "uses one base-logits build")
    for base_dir in base_dirs:
        base_logits = base_logits_facts(Path(base_dir), split_files)

    # Astra round-1 MUST-FIX 2: the union of the training sources the selected
    # arms actually run, compared against the pin's registered hashes below.
    training_sources = sorted({name for entry in entries
                               for name in implementation_sources(
                                   str(entry["arm"]))})

    # Astra MUST-FIX 2: measuring is not verifying. Compare what is on disk
    # against what the config pins, before the first launch.
    provenance = provenance_check(
        config,
        Path(config_path if config_path is not None else DEFAULT_CONFIG),
        declared, feature_hash, split_files, cache, base_logits,
        training_sources)

    return {"frame_dir": frame_dir,
            # Astra round-2 MUST-FIX 1: the configured spelling is what the
            # signature hashes; `frame_dir` above is this machine's path and
            # is reported, never hashed.
            "frame_dir_configured": str(data["directory"]),
            "frame_version": declared,
            "feature_hash": feature_hash,
            "split_files": split_files,
            "stats_cache": cache,
            "base_logits": base_logits,
            "provenance": provenance,
            # Resolved physical paths, recorded for humans and excluded from
            # the signature (Astra round-2 MUST-FIX 1).
            "physical_paths": path_link_facts({
                "frame_dir": frame_dir,
                "stats_cache": cache.get("path"),
                "models_embeddings": REPO / "models" / "embeddings",
            }),
            "training_sources": training_sources}


# -------------------------------------------------------------- commands

def out_dir_for(out_root: Path, config_id: str, seed: int) -> Path:
    return Path(out_root) / config_id / f"seed_{seed}"


def _trainer_args(config: dict, entry: dict, seed, out_dir: Path,
                  effective: dict) -> list[str]:
    """The trainer flags for one (configuration, seed), without argv[0..1].

    Every flag is taken from `effective` and from the configuration's checked
    params, so the argv, the reuse check and the config's rendered command all
    read the same numbers and cannot drift apart. `--no-kit` is always
    present and `--score-test` never is: stage-2 selection must not be able
    to see a test row. `--save-predictions` is always present (D3.11).
    `--aux` is never emitted — `load_config` refuses `training.aux` true.
    """
    training = config["training"]
    data = config["data"]
    arch, opt = effective["arch"], effective["optimiser"]
    params = entry["_params"]
    args = ["--arm", str(entry["arm"]), "--seed", str(seed),
            "--data-dir", rel(REPO / data["directory"])]
    if params["k"] is not None:
        args += ["--k", str(params["k"])]
    if params["base_logits_dir"]:
        args += ["--base-logits-dir", str(params["base_logits_dir"]),
                 "--residual-l2", repr(float(params["residual_l2"]))]
    args += [
        "--dmodel", str(arch["dmodel"]),
        "--layers", str(arch["layers"]),
        "--heads", str(arch["heads"]),
        "--batch", str(opt["batch"]),
        "--epochs", str(opt["epochs"]),
        "--lr", repr(opt["lr"]),
        "--patience", str(opt["patience"]),
        "--device", str(training["device"]),
        "--no-kit",
        "--save-predictions",
        "--stats-cache-role", str(data["stats_cache_role"]),
        "--out", rel(out_dir),
    ]
    return args


def command_for(config: dict, entry: dict, seed: int, out_dir: Path,
                effective: dict) -> list[str]:
    """The exact `transformer_t1.py` argv this (configuration, seed) runs."""
    return [sys.executable, str(TRAINER)] + _trainer_args(
        config, entry, seed, out_dir, effective)


def command_text(config: dict, entry: dict, seed: int, out_root: Path,
                 effective: dict) -> str:
    """The portable rendering of that argv, as the config records it.

    `sys.executable` is this machine's interpreter path, which differs
    between the laptop and the mini, so the registered string names the
    repository's own launcher instead. Everything after it is the argv, one
    string per registered seed.
    """
    out_dir = Path(out_root) / str(entry["id"]) / f"seed_{int(seed)}"
    args = _trainer_args(config, entry, int(seed), out_dir, effective)
    return f"{COMMAND_TEXT_PREFIX} " + shlex.join(
        [TRAINER.relative_to(REPO).as_posix()] + args)


def registered_commands(config: dict, entry: dict, out_root: Path,
                        epochs: int | None = None) -> list[str]:
    """The command list the config registers for one configuration."""
    effective = effective_settings(config, entry, epochs)
    return [command_text(config, entry, seed, out_root, effective)
            for seed in config["training"]["seeds"]]


# ------------------------------------------------- checkpoint bookkeeping

def _readable_json(path: Path) -> str | None:
    """None when the file parses as a JSON object, else why it does not."""
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return f"{path.name} is unreadable ({exc})"
    if not isinstance(payload, dict):
        return f"{path.name} is not a JSON object"
    return None


def _readable_npz(path: Path) -> str | None:
    """None when the npz opens and carries the probabilities D10 reads."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "probs" not in archive:
                return f"{path.name} carries no 'probs' array"
            np.asarray(archive["probs"]).shape
    except Exception as exc:  # noqa: BLE001 - any read failure is incomplete
        return f"{path.name} is unreadable ({exc})"
    return None


def _readable_checkpoint(path: Path, arm: str | None = None) -> str | None:
    """None when `model.pt` really is this arm's state dict.

    Astra round-1 MUST-FIX 3: any nonempty file used to pass, so a truncated
    or mismatched checkpoint could be admitted after a transfer. The file is
    LOADED (`weights_only=True`, on the CPU — no model is instantiated, which
    keeps this structural and fast) and its top-level parameter names are
    checked against what the arm's own wiring produces: a recurrent arm's
    parameters all live under `recurrent.`, and every other arm has
    `feat_proj` and `head` and no `recurrent.` key at all.
    """
    import torch  # noqa: PLC0415 - already imported by the trainer module

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - truncation, garbage, wrong format
        return f"{path.name} does not load as a checkpoint ({exc})"
    if not hasattr(payload, "keys"):
        return (f"{path.name} loaded as {type(payload).__name__}, not the "
                "state dict the trainer saves")
    keys = [str(key) for key in payload.keys()]
    if not keys:
        return f"{path.name} is an empty state dict"
    if arm is None:
        return None
    recurrent = [key for key in keys if key.startswith("recurrent.")]
    if t1.ARM_WIRING[arm] == "recurrent":
        if len(recurrent) != len(keys):
            return (f"{path.name} carries "
                    f"{len(keys) - len(recurrent)} parameter(s) outside "
                    f"'recurrent.', so it is not arm {arm!r}'s checkpoint")
        return None
    if recurrent:
        return (f"{path.name} carries 'recurrent.' parameters, so it is not "
                f"arm {arm!r}'s checkpoint")
    missing = [prefix for prefix in ("feat_proj", "head")
               if not any(key.startswith(f"{prefix}.") for key in keys)]
    if missing:
        return (f"{path.name} carries no {missing} parameter(s), so it is not "
                f"arm {arm!r}'s checkpoint")
    return None


def artifact_problems(out_dir: Path, arm: str | None = None) -> list[str]:
    """Why this directory is not a finished run — empty when it is one.

    Astra MUST-FIX 3. The old check accepted `metrics.json` + `model.pt`,
    which silently accepted a run whose row-aligned validation
    probabilities were never written (D10 reads those, not a scalar) and
    refused, as a contract violation, a run interrupted before its
    `run_record.json`. Both are the same thing: an unfinished run. All four
    artefacts must be there and readable; anything less is retried.
    """
    out_dir = Path(out_dir)
    problems = []
    for name in RUN_ARTIFACTS:
        path = out_dir / name
        if not path.is_file():
            problems.append(f"{name} is missing")
            continue
        if path.stat().st_size == 0:
            problems.append(f"{name} is empty")
            continue
        if name.endswith(".json"):
            problem = _readable_json(path)
        elif name.endswith(".npz"):
            problem = _readable_npz(path)
        elif name.endswith(".pt"):
            # Astra round-1 MUST-FIX 3: the checkpoint is read, not sized.
            problem = _readable_checkpoint(path, arm)
        else:
            problem = None
        if problem:
            problems.append(problem)
    return problems


def is_complete(out_dir: Path, arm: str | None = None) -> bool:
    """A run is complete only with all four artefacts, readable."""
    return not artifact_problems(out_dir, arm)


def completion_record_path(out_dir: Path) -> Path:
    return Path(out_dir) / COMPLETION_RECORD


def read_completion_record(out_dir: Path) -> dict | None:
    path = completion_record_path(out_dir)
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return record if isinstance(record, dict) else None


# The three states a completion record can be in, distinguished explicitly
# (Astra round-1 MUST-FIX 3). "absent" is a legacy run and is allowed: those
# predate the record and are verified by contract instead. "malformed" is NOT
# the same thing — a record that is there but unreadable means the run's own
# claim about itself cannot be read, which is unverifiable, not legacy.
COMPLETION_ABSENT = "absent"
COMPLETION_MALFORMED = "malformed"
COMPLETION_PRESENT = "present"
COMPLETION_REQUIRED_FIELDS = ("config_id", "seed", "artifacts",
                              "training_signature")


def completion_status(out_dir: Path) -> dict:
    """`{"state": absent|malformed|present, "record":…, "reason":…}`."""
    path = completion_record_path(out_dir)
    if not path.exists():
        return {"state": COMPLETION_ABSENT, "record": None,
                "reason": (f"{COMPLETION_RECORD} is absent; this run predates "
                           "the completion record and is verified by contract "
                           "alone")}
    try:
        record = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"state": COMPLETION_MALFORMED, "record": None,
                "reason": f"{COMPLETION_RECORD} does not parse ({exc})"}
    if not isinstance(record, dict):
        return {"state": COMPLETION_MALFORMED, "record": None,
                "reason": f"{COMPLETION_RECORD} is not a JSON object"}
    missing = [field for field in COMPLETION_REQUIRED_FIELDS
               if record.get(field) is None]
    if missing:
        return {"state": COMPLETION_MALFORMED, "record": record,
                "reason": f"{COMPLETION_RECORD} is missing {missing}"}
    manifest = record.get("artifacts")
    if not isinstance(manifest, dict) or not manifest:
        return {"state": COMPLETION_MALFORMED, "record": record,
                "reason": (f"{COMPLETION_RECORD} artifacts "
                           f"{manifest!r} is not an artefact manifest")}
    for name, facts in manifest.items():
        if not isinstance(facts, dict) or facts.get("md5") is None or (
                facts.get("bytes") is None):
            return {"state": COMPLETION_MALFORMED, "record": record,
                    "reason": (f"{COMPLETION_RECORD} artifacts.{name} "
                               f"{facts!r} carries no size and md5")}
    return {"state": COMPLETION_PRESENT, "record": record, "reason": None}


def manifest_problems(out_dir: Path, record: dict) -> list[str]:
    """Every artefact that no longer matches what `COMPLETE.json` recorded.

    Astra round-1 MUST-FIX 3: reuse never compared the stored sizes and
    hashes, so a truncated or half-overwritten rsync of a finished run was
    admitted. This is the check that catches it.
    """
    problems = []
    manifest = record.get("artifacts") or {}
    for name in RUN_ARTIFACTS:
        facts = manifest.get(name)
        if not isinstance(facts, dict):
            problems.append(
                f"{COMPLETION_RECORD} records no manifest entry for {name}")
            continue
        path = Path(out_dir) / name
        if not path.is_file():
            problems.append(f"{name} is recorded in {COMPLETION_RECORD} but "
                            "is not on disk")
            continue
        size = int(path.stat().st_size)
        if int(facts["bytes"]) != size:
            problems.append(
                f"{name} is {size} bytes, {COMPLETION_RECORD} recorded "
                f"{int(facts['bytes'])} (a truncated or replaced transfer)")
        live = t1.md5_file(path)
        if str(facts["md5"]) != live:
            problems.append(
                f"{name} md5 {live!r} != the {COMPLETION_RECORD} manifest's "
                f"{str(facts['md5'])!r}")
    return problems


def artifact_manifest(out_dir: Path) -> dict:
    """Size and md5 of each artefact, as validated."""
    return {name: {"bytes": int((Path(out_dir) / name).stat().st_size),
                   "md5": t1.md5_file(Path(out_dir) / name)}
            for name in RUN_ARTIFACTS}


def write_completion_record(out_dir: Path, seed: int, config_id: str,
                            signature: dict, machine: dict | None = None,
                            arm: str | None = None) -> dict:
    """Validate the four artefacts, then write ONE atomic record.

    Astra MUST-FIX 3: the record is written to a temp file in the same
    directory and `os.replace`d, so a crash at any point leaves either the
    previous state or the whole record — never a half-written marker that a
    later invocation would read as a finished run.
    """
    out_dir = Path(out_dir)
    problems = artifact_problems(out_dir, arm)
    if problems:
        raise RetrainError(
            f"{rel(out_dir)}: the run exited 0 but did not leave a complete "
            "set of artefacts:\n  " + "\n  ".join(problems))
    payload = {
        "config_id": config_id,
        "seed": int(seed),
        "artifacts": artifact_manifest(out_dir),
        # Astra round-2 MUST-FIX 4 / D8.8: which machine produced this run.
        "machine_provenance": machine,
        "training_signature": signature["training_signature"],
        "training_signature_components":
            signature["training_signature_components"],
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "rule": ("written only after all of "
                 f"{list(RUN_ARTIFACTS)} validated; atomic via os.replace"),
    }
    temp = out_dir / f".{COMPLETION_RECORD}.tmp"
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temp, completion_record_path(out_dir))
    return payload


def clear_run_artifacts(out_dir: Path) -> list[str]:
    """Remove a partial (or force-retrained) run's artefacts before relaunch.

    Astra MUST-FIX 3: a retry must not be able to inherit a stale artefact
    from the interrupted attempt — a leftover `metrics.json` beside a new
    `model.pt` would be read as one run.
    """
    removed = []
    for name in tuple(RUN_ARTIFACTS) + (COMPLETION_RECORD,
                                        f".{COMPLETION_RECORD}.tmp"):
        path = Path(out_dir) / name
        if path.exists():
            path.unlink()
            removed.append(name)
    return removed


def read_metrics(out_dir: Path) -> dict:
    return json.loads((out_dir / "metrics.json").read_text())


def as_n_rows(value):
    """A row count as an int when it is a number, and unchanged otherwise."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    return int(value)


def frame_problems(contract: dict, resolved: dict) -> list[str]:
    """Every disagreement between a checkpoint and the frame on disk NOW."""
    live_hash = resolved.get("feature_hash")
    live_files = resolved.get("split_files")
    if not isinstance(live_hash, dict) or not isinstance(live_files, dict):
        raise RetrainError(
            "preflight measured no frame identity (feature_hash / "
            "split_files), so no checkpoint can be verified against it")

    problems = []
    recorded_hash = contract.get("feature_hash")
    if not isinstance(recorded_hash, dict):
        problems.append(
            f"feature_hash {recorded_hash!r} is not the frame declaration "
            "this checkpoint was trained under")
    else:
        for key in sorted(set(live_hash) | set(recorded_hash)):
            got, want = recorded_hash.get(key, ABSENT), live_hash.get(key,
                                                                      ABSENT)
            if got != want:
                problems.append(f"feature_hash.{key} {got!r} != {want!r}")

    recorded_files = contract.get("split_files")
    if not isinstance(recorded_files, dict):
        problems.append(f"split_files {recorded_files!r} is not a mapping")
        recorded_files = {}
    for split in CONTRACT_SPLITS:
        live = live_files.get(split)
        if not isinstance(live, dict):
            raise RetrainError(
                f"preflight measured no {split!r} split, so a checkpoint "
                "cannot be verified against it")
        recorded = recorded_files.get(split)
        if not isinstance(recorded, dict):
            problems.append(
                f"split_files.{split} {recorded!r} is not the record of a "
                "parquet this checkpoint read")
            continue
        for field in SPLIT_IDENTITY_FIELDS:
            got, want = recorded.get(field, ABSENT), live[field]
            if field == "n_rows":
                got = as_n_rows(got)
            if got != want:
                problems.append(
                    f"split_files.{split}.{field} {got!r} != {want!r}")
    return problems


def arm_params_problems(metrics: dict, effective: dict) -> list[str]:
    """Every disagreement between a checkpoint's arm and this config's arm.

    The stage-2 addition to the stage-1 reuse check (D3.12). Sixteen
    configurations share one architecture and one optimiser, so without this
    a `same_entity_k6` checkpoint in the `same_entity_k30` directory, or a
    residual arm trained against superseded base logits, would pass every
    other field and be reported under the wrong identity.
    """
    recorded = metrics.get("arm_params")
    if not isinstance(recorded, dict):
        raise RetrainError(
            "metrics.json carries no arm_params block; this checkpoint "
            "predates the stage-2 trainer surface and cannot be reused "
            "(retrain it with --force)")
    problems = []
    want_all = effective["arm_params"]
    for field in ARM_PARAM_FIELDS:
        got, want = recorded.get(field, ABSENT), want_all[field]
        if isinstance(want, float) and isinstance(got, (int, float)) and not (
                isinstance(got, bool)):
            ok = float(got) == want
        else:
            ok = got == want
        if not ok:
            problems.append(f"arm_params.{field} {got!r} != {want!r}")
    return problems


def verify_checkpoint(out_dir: Path, metrics: dict, config: dict,
                      resolved: dict, entry: dict, seed: int,
                      effective: dict, signature: dict | None = None) -> dict:
    """Re-verify a checkpoint before reusing or reporting it."""
    if signature is None:
        signature = signature_for(effective, resolved)
    contract = metrics.get("training_contract")
    if not contract:
        raise RetrainError(
            f"{rel(out_dir)}: metrics.json carries no training_contract; "
            "retrain it with --force")
    arm = str(entry["arm"])
    problems = []
    if contract.get("contract_version") != t1.CONTRACT_VERSION:
        problems.append(
            f"contract_version {contract.get('contract_version')!r} != "
            f"{t1.CONTRACT_VERSION!r}")
    if int(contract.get("seed", -1)) != int(seed):
        problems.append(f"seed {contract.get('seed')!r} != {seed}")
    if (contract.get("architecture") or {}).get("arm") != arm:
        problems.append(
            f"arm {(contract.get('architecture') or {}).get('arm')!r} != "
            f"{arm!r}")
    problems.extend(arm_params_problems(metrics, effective))
    if contract.get("frame_version") != resolved["frame_version"]:
        problems.append(
            f"frame_version {contract.get('frame_version')!r} != "
            f"{resolved['frame_version']!r}")
    # Astra round-2 MUST-FIX 1: the two machines spell the frame differently
    # (the laptop's worktree reaches it through a symlink), so the comparison
    # is of the PHYSICAL directory the two spellings name, not of their text.
    # The frame's content identity is compared field by field below.
    recorded_frame = Path(contract.get("frame_dir", ""))
    if not recorded_frame.is_absolute():
        recorded_frame = REPO / recorded_frame
    if recorded_frame.resolve() != Path(resolved["frame_dir"]).resolve():
        problems.append(
            f"frame_dir {contract.get('frame_dir')!r} resolves to "
            f"{recorded_frame.resolve().as_posix()!r}, not to the measured "
            f"frame {Path(resolved['frame_dir']).resolve().as_posix()!r}")
    problems.extend(frame_problems(contract, resolved))
    if contract.get("kit_used") is not False:
        problems.append("kit_used is not false")
    if contract.get("test_split_scored") is not False:
        problems.append("test_split_scored is not false")
    cache = contract.get("stats_cache") or {}
    if cache.get("md5") != resolved["stats_cache"]["md5"]:
        problems.append(
            f"stats cache md5 {cache.get('md5')!r} != "
            f"{resolved['stats_cache']['md5']!r}")

    recorded_arch = contract.get("architecture") or {}
    for field, want in effective["arch"].items():
        got = recorded_arch.get(field)
        if not isinstance(got, int) or isinstance(got, bool) or got != want:
            problems.append(f"architecture.{field} {got!r} != {want!r}")
    recorded_opt = contract.get("optimiser") or {}
    for field, want in effective["optimiser"].items():
        got = recorded_opt.get(field)
        if isinstance(want, bool):
            ok = got is want
        else:
            ok = (isinstance(got, (int, float)) and not isinstance(got, bool)
                  and float(got) == float(want))
        if not ok:
            problems.append(f"optimiser.{field} {got!r} != {want!r}")

    # The config id and the overrides the checkpoint was launched under. The
    # contract cannot carry either (the trainer never sees them), so the
    # runner's own `run_record.json` is the only witness; without it the
    # provenance of a reused checkpoint is unverifiable and reuse is refused.
    record = read_run_record(out_dir)
    if record is None:
        problems.append(
            f"{rel(run_record_path(out_dir))} is absent or unreadable, so "
            "the config id and overrides this checkpoint was trained under "
            "cannot be verified")
    else:
        if str(record.get("config_id")) != effective["config_id"]:
            problems.append(
                f"run_record.json config_id {record.get('config_id')!r} != "
                f"{effective['config_id']!r}")
        if record.get("overrides") != effective["overrides"]:
            problems.append(
                f"run_record.json overrides {record.get('overrides')!r} != "
                f"this invocation's {effective['overrides']!r}")
        # Astra MUST-FIX 2/11: the seed-independent signature. This is the
        # only check that can see an edited implementation file, and the one
        # that makes two separately-trained seeds safe to put in one table.
        problems.extend(signature_problems(record, signature))

    # When the atomic completion record is there it must describe THIS run
    # (Astra MUST-FIX 3), and its artefact manifest must still hold (Astra
    # round-1 MUST-FIX 3). Three states, distinguished explicitly:
    #   absent    — a legacy run that predates the record; allowed, and
    #               verified by contract instead;
    #   malformed — present but unreadable or incomplete: the run's own claim
    #               about itself cannot be read, so it is UNVERIFIABLE and
    #               refused, which is not the same as legacy;
    #   present   — every recorded field must agree, named field by field.
    status = completion_status(out_dir)
    if status["state"] == COMPLETION_MALFORMED:
        problems.append(
            f"{status['reason']}; a completion record that is present but "
            "unreadable makes this run unverifiable (it is not a legacy run "
            "with no record at all)")
    completion = status["record"] if status["state"] == COMPLETION_PRESENT \
        else None
    if completion is not None:
        problems.extend(manifest_problems(out_dir, completion))
        if (completion.get("training_signature")
                != signature["training_signature"]):
            problems.append(
                f"{COMPLETION_RECORD} training_signature "
                f"{completion.get('training_signature')!r} != "
                f"{signature['training_signature']!r}")
        if completion.get("seed") is not None and int(
                completion["seed"]) != int(seed):
            problems.append(
                f"{COMPLETION_RECORD} seed {completion.get('seed')!r} != "
                f"{seed}")

    if metrics.get(f"{SELECTION_SPLIT}_ll") is None:
        problems.append(f"no {SELECTION_SPLIT}_ll")
    if problems:
        raise RetrainError(
            f"{rel(out_dir)} does not match this config; refusing to reuse "
            "it (rerun with --force to retrain):\n  " + "\n  ".join(problems))
    return contract


def run_record_path(out_dir: Path) -> Path:
    return Path(out_dir) / "run_record.json"


def write_run_record(out_dir: Path, payload: dict) -> None:
    run_record_path(out_dir).write_text(json.dumps(payload, indent=2) + "\n")


def read_run_record(out_dir: Path) -> dict | None:
    """The run record next to a checkpoint, or None when it is unusable."""
    path = run_record_path(out_dir)
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return record if isinstance(record, dict) else None


def read_wall_seconds(record: dict | None) -> float | None:
    value = (record or {}).get("wall_seconds")
    return None if value is None else float(value)


INVALID_RSS_METHOD = "rusage_children_delta"
INVALID_RSS_REASON = (
    "recorded as a delta of resource.getrusage(RUSAGE_CHILDREN).ru_maxrss "
    "across sequential children; that field is a process-wide high-water "
    "mark, so the delta is not this child's peak and can be zero. The "
    "observation is preserved, not reinterpreted (Astra MUST-FIX 4)")


def read_peak_rss(record: dict | None) -> dict:
    """What a run record says about peak RSS, and whether it is a peak.

    Astra MUST-FIX 4. Records written by the sampler carry
    `peak_rss_bytes_sampled` and are valid. Records written by the earlier
    `RUSAGE_CHILDREN` delta (the D6 smoke runs) carry `peak_rss_bytes` and
    are NOT: their number is reported as an invalid observation with the
    reason, and the historical file is never rewritten to look correct.
    """
    record = record or {}
    sampled = record.get("peak_rss_bytes_sampled")
    if sampled is not None:
        return {"peak_rss_bytes": int(sampled),
                "peak_rss_bytes_valid": True,
                "rss_method": str(record.get("rss_method") or "sampled"),
                "peak_rss_unit": "bytes"}
    legacy = record.get("peak_rss_bytes")
    if legacy is None:
        return {"peak_rss_bytes": None, "peak_rss_bytes_valid": False,
                "rss_method": None, "peak_rss_unit": "bytes"}
    return {"peak_rss_bytes": None, "peak_rss_bytes_valid": False,
            "rss_method": INVALID_RSS_METHOD,
            "peak_rss_unit": "bytes",
            "peak_rss_bytes_observed_invalid": int(legacy),
            "peak_rss_invalid_reason": INVALID_RSS_REASON}


def _tree_rss_bytes_psutil(pid: int) -> int | None:
    """Summed RSS of `pid` and every descendant, or None if none is alive."""
    try:
        parent = psutil.Process(pid)
        procs = [parent] + parent.children(recursive=True)
    except Exception:  # noqa: BLE001 - the child exited between samples
        return None
    total, seen = 0, False
    for proc in procs:
        try:
            total += int(proc.memory_info().rss)
            seen = True
        except Exception:  # noqa: BLE001 - one descendant exited mid-sample
            continue
    return total if seen else None


def _tree_rss_bytes_ps(pid: int) -> int | None:
    """The same measurement through `ps`, for a checkout without psutil.

    `ps -A -o pid=,ppid=,rss=` gives every process once; RSS is kilobytes.
    """
    try:
        out = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,rss="],
                             capture_output=True, text=True, check=True).stdout
    except Exception:  # noqa: BLE001 - ps unavailable: no sample, not a guess
        return None
    rss, children = {}, {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            this, parent, kb = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        rss[this] = kb * 1024
        children.setdefault(parent, []).append(this)
    if pid not in rss:
        return None
    total, stack = 0, [pid]
    while stack:
        current = stack.pop()
        total += rss.get(current, 0)
        stack.extend(children.get(current, ()))
    return total


def tree_rss_bytes(pid: int) -> int | None:
    return (_tree_rss_bytes_psutil(pid) if psutil is not None
            else _tree_rss_bytes_ps(pid))


class ChildRssSampler:
    """Sample one child's own process-tree RSS while it runs.

    Astra MUST-FIX 4: this replaces the `RUSAGE_CHILDREN` delta. The peak is
    the largest total observed over the child and its descendants, so it is
    a property of THAT child and cannot be contaminated by an earlier one.
    A run short enough that no sample lands records no peak and says so,
    rather than reporting a number nothing measured.
    """

    def __init__(self, pid: int, interval: float = 0.5) -> None:
        self.pid = int(pid)
        self.interval = float(interval)
        self.peak: int | None = None
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _sample_once(self) -> None:
        value = tree_rss_bytes(self.pid)
        if value is None:
            return
        self.samples += 1
        if self.peak is None or value > self.peak:
            self.peak = value

    def _loop(self) -> None:
        # Sample immediately, so even a very short child is observed once.
        self._sample_once()
        while not self._stop.wait(self.interval):
            self._sample_once()

    def start(self) -> "ChildRssSampler":
        self._thread.start()
        return self

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join(timeout=5.0)
        method = ("psutil_process_tree_sampled" if psutil is not None
                  else "ps_process_tree_sampled")
        payload = {
            "peak_rss_bytes_sampled": self.peak,
            "peak_rss_bytes_valid": self.peak is not None,
            "peak_rss_unit": "bytes",
            "rss_method": method,
            "rss_sample_interval_seconds": self.interval,
            "rss_samples": self.samples,
            "rss_method_note": (
                "max over samples of the summed RSS of the training child and "
                "its descendants, taken while it ran; NOT a getrusage delta"),
        }
        if self.peak is None:
            payload["peak_rss_invalid_reason"] = (
                "no sample landed while the child was alive")
        return payload


# ----------------------------------------------------------------- summary

def validation_match_count(frame_dir: Path, version: str) -> tuple[int, int]:
    """(rows, distinct matches) of the split selection reads. Test is not
    touched: this reads the validation parquet only."""
    path = t1.split_path(frame_dir, version, SELECTION_SPLIT)
    innings = pd.read_parquet(path, columns=["innings_id"])["innings_id"]
    return int(len(innings)), int(
        len(set(match_ids(innings.astype(str).to_numpy()).tolist())))


PER_SEED_FIELDS = ("seed", "ll", "best_epoch", "wall_seconds",
                   "peak_rss_bytes", "peak_rss_bytes_valid", "rss_method",
                   "peak_rss_unit", "checkpoint_md5", "checkpoint_dir",
                   "overrides", "omp_num_threads", "training_signature",
                   # Astra round-2 MUST-FIX 4 / D8.8: the host identity the
                   # run recorded WHILE it ran, carried into the report. None
                   # on a run written before this field existed — never
                   # reconstructed from the machine reading the tree.
                   "machine_provenance")
# Carried through only when a row has them: the preserved (invalid) historical
# RSS observation and why it is not a peak (Astra MUST-FIX 4).
PER_SEED_OPTIONAL_FIELDS = ("peak_rss_bytes_observed_invalid",
                            "peak_rss_invalid_reason")


def per_seed_row(row: dict) -> dict:
    payload = {key: row[key] for key in PER_SEED_FIELDS}
    payload.update({key: row[key] for key in PER_SEED_OPTIONAL_FIELDS
                    if key in row})
    return payload


def build_summary(config: dict, config_path: Path, resolved: dict,
                  entry: dict, rows: list, overrides: dict, out_root: Path,
                  effective: dict, validation: tuple | None = None,
                  signature: dict | None = None,
                  machine: dict | None = None) -> dict:
    """One configuration's `summary.yaml` (D3.12)."""
    seeds = config["training"]["seeds"]
    live_files = resolved["split_files"]

    # Every reused checkpoint must agree with preflight's measurement field
    # by field. `verify_checkpoint` has already required it; it is re-asserted
    # here because THIS is the function that publishes the numbers.
    for row in rows:
        for split, record in (row["_split_files"] or {}).items():
            live = live_files.get(split)
            if live is None:
                raise RetrainError(
                    f"{row['checkpoint_dir']}: trained on a {split!r} split "
                    "the measured frame does not have; the summary would "
                    "describe a frame that is not on disk")
            for field in SPLIT_IDENTITY_FIELDS:
                got = record.get(field, ABSENT)
                if field == "n_rows":
                    got = as_n_rows(got)
                if got != live[field]:
                    raise RetrainError(
                        f"{row['checkpoint_dir']}: split_files.{split}."
                        f"{field} {got!r} != the measured frame's "
                        f"{live[field]!r}")

    rows_count, n_matches = (validation if validation is not None
                             else (None, None))
    if signature is None:
        signature = signature_for(effective, resolved)
    # Astra MUST-FIX 2/11: every seed in this table was verified against ONE
    # training signature, so the table itself records which one. A consolidated
    # two-machine night is exactly this: two seeds, one signature.
    for row in rows:
        got = row.get("training_signature")
        if got != signature["training_signature"]:
            raise RetrainError(
                f"{row['checkpoint_dir']}: training_signature {got!r} != "
                f"{signature['training_signature']!r}; the summary would put "
                "two differently-trained runs in one table")
    return {
        "experiment": {
            "name": config["experiment"]["name"],
            "config": rel(config_path),
            "config_sha256": t1.sha256_text(Path(config_path).read_text()),
            "config_id": str(entry["id"]),
            "runner": rel(Path(__file__)),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "git_head_short": t1.git_head_short(),
            "runs_expected": len(seeds),
            "runs_recorded": len(rows),
            "complete": len(rows) == len(seeds),
            "overrides": overrides,
            "test_split_scored": False,
            "training_signature": signature["training_signature"],
            "training_signature_components":
                signature["training_signature_components"],
            "training_signature_rule": signature["training_signature_rule"],
            "training_signature_sources":
                signature["training_signature_sources"],
            "provenance_check": resolved.get("provenance"),
            # Astra round-2 MUST-FIX 1: physical paths are reported here,
            # outside the signature, so a worktree's symlinked frame is
            # visible to a reader without entering the identity.
            "physical_paths": resolved.get("physical_paths"),
            # Astra round-2 MUST-FIX 4: the machine that wrote this summary.
            # The per-seed rows carry the machine each SEED was trained on.
            "consolidating_machine": machine,
        },
        "arm": {
            "arm": str(entry["arm"]),
            "queue_order": int(entry["queue_order"]),
            "reference": list(entry["reference"]),
            "tests": entry["tests"],
            "arm_params_expected": effective["arm_params"],
        },
        "frame": {
            "dir": rel(resolved["frame_dir"]),
            "version": resolved["frame_version"],
            "feature_hash": resolved["feature_hash"],
            "split_md5s": {split: facts["md5"]
                           for split, facts in live_files.items()},
            "split_rows": {split: int(facts["n_rows"])
                           for split, facts in live_files.items()},
            "stats_cache": resolved["stats_cache"],
            "base_logits": resolved.get("base_logits"),
        },
        "splits": {
            SELECTION_SPLIT: {
                "n_rows": rows_count,
                "n_matches": n_matches,
                "n_matches_source": N_MATCHES_SOURCE,
                "per_seed": [per_seed_row(row) for row in rows],
            },
        },
        "output_root": rel(out_root),
    }


def print_table(config_id: str, summary: dict) -> None:
    block = summary["splits"][SELECTION_SPLIT]
    print(f"\n{config_id}: validation rows {block['n_rows']} / matches "
          f"{block['n_matches']}")
    for row in block["per_seed"]:
        wall = ("        -" if row["wall_seconds"] is None
                else f"{row['wall_seconds']:9.1f}s")
        # Astra MUST-FIX 4: an invalid historical observation prints as
        # "invalid", never as a peak.
        if row["peak_rss_bytes"] is not None:
            rss = f"{row['peak_rss_bytes'] / 1024 ** 3:6.2f}G"
        elif row["rss_method"] == INVALID_RSS_METHOD:
            rss = "invalid"
        else:
            rss = "      -"
        # Astra round-2 MUST-FIX 4 / D8.8: the machine is part of the table.
        host = (row.get("machine_provenance") or {}).get("machine") or "-"
        print(f"  seed {row['seed']:>3}  validation LL {row['ll']!r:<20} "
              f"best epoch {row['best_epoch']:>3}  wall {wall}  rss {rss}  "
              f"machine {host}")


# -------------------------------------------------------------------- main

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-root", type=Path, default=None,
                        help="override outputs.directory (must stay under "
                             "models/embeddings/seq_stage2)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override training.epochs; recorded in "
                             "summary.yaml as an override")
    parser.add_argument("--config-ids", default=None,
                        help="comma-separated configuration ids to run "
                             "(default: all, in queue order)")
    parser.add_argument("--seeds", default=None,
                        help="comma-separated subset of the configuration's "
                             "registered seeds; may narrow, never widen")
    parser.add_argument("--force", action="store_true",
                        help="retrain runs whose checkpoints are complete")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the commands and exit without training")
    # Astra round-2 MUST-FIX 3: the consolidation of a split night is a
    # verification, not a training invocation.
    parser.add_argument("--consolidate", action="store_true",
                        help="verify every selected (configuration, seed) and "
                             "rewrite summary.yaml from the completed runs; "
                             "never trains, never deletes. An incomplete or "
                             "unverifiable run refuses and is named, so "
                             "recovery is queued on the machine that owns "
                             "that seed")
    # Astra round-2 MUST-FIX 4 / D8.8.
    parser.add_argument("--machine", default=None,
                        help="machine label recorded with every run this "
                             "invocation writes (default: the hostname)")
    args = parser.parse_args(argv)

    if args.consolidate:
        if args.force:
            raise RetrainError(
                "--consolidate and --force are contradictory: consolidation "
                "verifies and summarises the runs that exist and never "
                "retrains one. Retrain on the machine the seed is assigned to")
        if args.dry_run:
            raise RetrainError(
                "--consolidate and --dry-run are contradictory: there is no "
                "command to print, because consolidation launches nothing")

    config = load_config(args.config)
    out_root = assert_writable(
        args.out_root if args.out_root is not None
        else REPO / config["outputs"]["directory"])
    epochs = (args.epochs if args.epochs is not None
              else int(config["training"]["epochs"]))
    # An override is a DEVIATION from the registered config, not merely a
    # flag that was typed: `--out-root <the configured directory>` and
    # `--epochs <the configured epochs>` describe the registered run, and
    # recording them would make a resumed invocation disagree with the run
    # records its own earlier invocation wrote.
    overrides = {}
    if args.out_root is not None and rel(out_root) != config["outputs"][
            "directory"]:
        overrides["out_root"] = rel(out_root)
    if args.epochs is not None and epochs != int(config["training"]["epochs"]):
        overrides["epochs"] = epochs

    entries = configurations(config)
    requested_ids = parse_id_list(args.config_ids)
    if requested_ids is not None:
        unknown = [name for name in requested_ids if name not in entries]
        if unknown:
            raise RetrainError(
                f"--config-ids {unknown} are not registered; the config holds "
                f"{list(entries)}")
        ids_to_run = [name for name in entries if name in requested_ids]
    else:
        ids_to_run = list(entries)

    registered_seeds = list(config["training"]["seeds"])
    seeds_to_run = narrow_seeds(registered_seeds, parse_seed_list(args.seeds))

    if args.dry_run:
        for config_id in ids_to_run:
            entry = entries[config_id]
            effective = effective_settings(config, entry, epochs, overrides)
            for seed in seeds_to_run:
                print(shlex.join(command_for(
                    config, entry, seed,
                    out_dir_for(out_root, config_id, seed), effective)))
        return 0

    resolved = preflight(config, [entries[name] for name in ids_to_run],
                         args.config)
    digest = (resolved["base_logits"] or {}).get("digest")
    print(f"frame {rel(resolved['frame_dir'])} ({resolved['frame_version']}), "
          f"stats cache {resolved['stats_cache']['role']} md5 "
          f"{resolved['stats_cache']['md5']}", flush=True)
    for split, facts in resolved["split_files"].items():
        print(f"  {split} md5 {facts['md5']} rows {facts['n_rows']} "
              f"{facts['match_date_min']}..{facts['match_date_max']}",
              flush=True)
    if resolved["base_logits"]:
        print(f"  base logits {resolved['base_logits']['dir']} digest "
              f"{digest}", flush=True)

    omp = os.environ.get("OMP_NUM_THREADS")
    if omp:
        print(f"  OMP_NUM_THREADS={omp} (inherited by every child)",
              flush=True)
    # Astra MUST-FIX 2: say what was compared, and what was recorded-but-
    # skipped, so an operator on the mini can see that the cohort was skipped
    # on purpose rather than wonder whether it was checked.
    check = resolved["provenance"]
    print(f"  provenance compared {check['compared']} against the pin "
          f"({check['pins_generated_at']}); skipped "
          f"{sorted(check['skipped'])}", flush=True)
    # Astra round-1 MUST-FIX 2: say which training sources agreed with the
    # registered hashes, and which the pin never recorded.
    print(f"  training sources verified against the pin: "
          f"{sorted(check.get('training_sources_compared') or {})}", flush=True)
    if check.get("training_sources_not_pinned"):
        print(f"  training sources the pin does NOT record: "
              f"{check['training_sources_not_pinned']}", flush=True)

    # Astra round-2 MUST-FIX 4 / D8.8: the host identity, captured now and
    # written into every artefact this invocation produces.
    machine = machine_provenance(
        args.machine, device_requested=config["training"]["device"],
        paths={"frame_dir": resolved["frame_dir"],
               "stats_cache": resolved["stats_cache"].get("path"),
               "models_embeddings": REPO / "models" / "embeddings",
               "out_root": out_root})
    print(f"  machine {machine['machine']} ({machine['hostname']}, "
          f"{machine['chip']}, {machine['os']} {machine['os_release']}, "
          f"torch {machine['torch_version']}, mps "
          f"{machine['torch_mps_available']}); thread caps "
          f"{machine['thread_caps']}", flush=True)

    # One signature per configuration, computed once and used for every seed:
    # it is seed-independent by construction (Astra MUST-FIX 2).
    signatures = {}
    for config_id in ids_to_run:
        effective = effective_settings(config, entries[config_id], epochs,
                                       overrides, digest)
        signatures[config_id] = signature_for(effective, resolved)
        print(f"  {config_id} training signature "
              f"{signatures[config_id]['training_signature'][:16]}...",
              flush=True)

    # Everything already complete under a selected configuration must
    # describe the same invocation as the one about to start, checked BEFORE
    # the first launch: the summary reports every complete run for that
    # configuration, not only the ones launched here, so discovering the
    # mismatch afterwards would mean hours of training followed by a refusal
    # to write any table at all. Runs this invocation will replace are exempt.
    for config_id in ids_to_run:
        entry = entries[config_id]
        effective = effective_settings(config, entry, epochs, overrides,
                                       digest)
        for seed in registered_seeds:
            out_dir = out_dir_for(out_root, config_id, seed)
            # Astra MUST-FIX 3: an incomplete directory is not verified here.
            # It is either retrained below (when its seed is selected) or left
            # alone; either way it contributes nothing to the table.
            if not is_complete(out_dir, str(entry["arm"])):
                continue
            if args.force and seed in seeds_to_run:
                continue
            verify_checkpoint(out_dir, read_metrics(out_dir), config, resolved,
                              entry, seed, effective, signatures[config_id])

    # ------------------------------------------------------- consolidation
    # Astra round-2 MUST-FIX 3. `--seeds 7,13` used to clear and retrain any
    # incomplete seed on the INVOKING machine, so a half-transferred seed 13
    # would have been trained on the laptop — silently violating the immutable
    # seed-to-machine assignment (D8.3). This path refuses instead: it deletes
    # nothing, launches nothing, names every run it could not admit, and only
    # then rewrites the summaries.
    if args.consolidate:
        refusals = []
        for config_id in ids_to_run:
            entry = entries[config_id]
            effective = effective_settings(config, entry, epochs, overrides,
                                           digest)
            for seed in seeds_to_run:
                out_dir = out_dir_for(out_root, config_id, seed)
                problems = artifact_problems(out_dir, str(entry["arm"]))
                if problems:
                    refusals.append(
                        f"{config_id} seed {seed} ({rel(out_dir)}) is "
                        "incomplete: " + "; ".join(problems))
                    continue
                try:
                    verify_checkpoint(out_dir, read_metrics(out_dir), config,
                                      resolved, entry, seed, effective,
                                      signatures[config_id])
                except RetrainError as error:
                    refusals.append(
                        f"{config_id} seed {seed} ({rel(out_dir)}) is "
                        f"unverifiable: {error}")
                    continue
                print(f"{config_id} seed {seed}: verified ({rel(out_dir)})",
                      flush=True)
        if refusals:
            raise RetrainError(
                "--consolidate refuses: no artefact was deleted and no "
                "training was launched. Re-queue each run below on the "
                "machine its seed is assigned to (D8.3), rsync again, then "
                "consolidate:\n  " + "\n  ".join(refusals))

    validation = validation_match_count(resolved["frame_dir"],
                                        resolved["frame_version"])
    if resolved["split_files"][SELECTION_SPLIT]["n_rows"] != validation[0]:
        raise RetrainError(
            f"{SELECTION_SPLIT} parquet holds {validation[0]} rows by "
            "innings_id but "
            f"{resolved['split_files'][SELECTION_SPLIT]['n_rows']} by the "
            "preflight measurement; the file changed underneath this run")

    # The training loop. Empty under `--consolidate`, which is the whole point
    # of that flag (Astra round-2 MUST-FIX 3): it has already verified every
    # selected run above and falls straight through to the summaries.
    for config_id in ([] if args.consolidate else ids_to_run):
        entry = entries[config_id]
        effective = effective_settings(config, entry, epochs, overrides,
                                       digest)
        signature = signatures[config_id]
        for seed in seeds_to_run:
            out_dir = out_dir_for(out_root, config_id, seed)
            if is_complete(out_dir, str(entry["arm"])) and not args.force:
                verify_checkpoint(out_dir, read_metrics(out_dir), config,
                                  resolved, entry, seed, effective, signature)
                print(f"{config_id} seed {seed}: complete; not overwriting "
                      f"({rel(out_dir)}); --force retrains", flush=True)
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            # Astra MUST-FIX 3: an interrupted attempt is RETRIED, and it is
            # retried cleanly — no artefact of the previous attempt survives
            # into the new run's directory.
            partial = artifact_problems(out_dir, str(entry["arm"]))
            if partial and any((out_dir / name).exists()
                              for name in RUN_ARTIFACTS):
                print(f"{config_id} seed {seed}: incomplete "
                      f"({'; '.join(partial)}); retraining", flush=True)
            removed = clear_run_artifacts(out_dir)
            if removed:
                print(f"{config_id} seed {seed}: cleared {removed}",
                      flush=True)
            command = command_for(config, entry, seed, out_dir, effective)
            print(f"\n{config_id} seed {seed}: {shlex.join(command)}",
                  flush=True)
            started = datetime.now(timezone.utc)
            clock = time.time()
            # `env` is this process's environment: OMP_NUM_THREADS and any
            # other thread cap the queue set is passed straight through.
            # Popen rather than run, because the RSS sampler needs the pid
            # while the child is alive (Astra MUST-FIX 4).
            child = subprocess.Popen(command, cwd=REPO, env=os.environ.copy())
            sampler = ChildRssSampler(child.pid).start()
            try:
                returncode = child.wait()
            finally:
                rss = sampler.stop()
            wall = time.time() - clock
            if returncode != 0:
                raise SystemExit(
                    f"{config_id} seed {seed}: transformer_t1.py exited "
                    f"{returncode} after {wall:.1f}s")
            # Astra round-2 MUST-FIX 4 / D8.8: the device the trainer says it
            # actually ran on, read back out of the metrics this run wrote —
            # the requested device is in `machine` already.
            device_used = None
            try:
                device_used = read_metrics(out_dir).get("device")
            except (OSError, json.JSONDecodeError, KeyError):
                device_used = None
            run_machine = dict(machine, device_used=(
                None if device_used is None else str(device_used)))
            write_run_record(out_dir, {
                "config_id": config_id,
                "arm": str(entry["arm"]),
                "seed": seed,
                "command": command,
                "wall_seconds": wall,
                **rss,
                "omp_num_threads": omp,
                # Every thread cap, the host identity and the symlink facts,
                # recorded while the run executed (D8.8).
                "machine_provenance": run_machine,
                "started_at": started.isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "config": rel(args.config),
                "config_sha256": t1.sha256_text(Path(args.config).read_text()),
                "overrides": overrides,
                "arm_params_expected": effective["arm_params"],
                "training_signature": signature["training_signature"],
                "training_signature_components":
                    signature["training_signature_components"],
                "training_signature_rule":
                    signature["training_signature_rule"],
                "training_signature_sources":
                    signature["training_signature_sources"],
            })
            # All four artefacts are validated here, and only then is the
            # atomic completion record written (Astra MUST-FIX 3).
            write_completion_record(out_dir, seed, config_id, signature,
                                    run_machine, str(entry["arm"]))
            print(f"{config_id} seed {seed}: {wall:.1f}s", flush=True)

    # Report every complete run of each selected configuration, not only the
    # ones this invocation launched, so a resumed job still writes one whole
    # table per configuration.
    written = []
    for config_id in ids_to_run:
        entry = entries[config_id]
        effective = effective_settings(config, entry, epochs, overrides,
                                       digest)
        signature = signatures[config_id]
        rows = []
        for seed in registered_seeds:
            out_dir = out_dir_for(out_root, config_id, seed)
            if not is_complete(out_dir, str(entry["arm"])):
                continue
            metrics = read_metrics(out_dir)
            contract = verify_checkpoint(out_dir, metrics, config, resolved,
                                         entry, seed, effective, signature)
            record = read_run_record(out_dir)
            rows.append({
                "seed": int(seed),
                "ll": float(metrics[f"{SELECTION_SPLIT}_ll"]),
                "best_epoch": contract["best_epoch"],
                "overrides": dict((record or {}).get("overrides") or {}),
                "wall_seconds": read_wall_seconds(record),
                # Astra MUST-FIX 4: validity travels with the number.
                **read_peak_rss(record),
                "training_signature": (record or {}).get(
                    "training_signature"),
                "omp_num_threads": (record or {}).get("omp_num_threads"),
                # Astra round-2 MUST-FIX 4: read out of the seed's OWN record,
                # so a consolidated table says which machine trained which
                # seed; None for a record written before the field existed.
                "machine_provenance": (record or {}).get(
                    "machine_provenance"),
                "checkpoint_md5": t1.md5_file(out_dir / "model.pt"),
                "checkpoint_dir": rel(out_dir),
                "_split_files": contract["split_files"],
            })
        summary = build_summary(config, args.config, resolved, entry, rows,
                                overrides, out_root, effective, validation,
                                signature, machine)
        summary_path = (out_root / config_id
                        / (config["outputs"].get("summary") or "summary.yaml"))
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(yaml.safe_dump(summary, sort_keys=False))
        print_table(config_id, summary)
        written.append((summary_path, summary))

    print()
    for summary_path, summary in written:
        print(f"wrote {rel(summary_path)} "
              f"({summary['experiment']['runs_recorded']}/"
              f"{summary['experiment']['runs_expected']} runs recorded)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RetrainError as error:
        print(f"retrain_stage2: {error}", file=sys.stderr)
        raise SystemExit(2) from None
