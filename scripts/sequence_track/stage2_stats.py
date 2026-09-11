#!/usr/bin/env python
"""Stage 2 statistics, gates, mechanism contrasts and the k sweep (D9, D10.1-D10.7, D10.11).

Two subcommands:

``stats``
    Reads every admitted ``models/embeddings/seq_stage2/runs/<config_id>/seed_<s>/``
    run, checks row alignment against the pinned validation parquet, maps
    every row to its I3 tournament block, and writes one machine-readable
    JSON holding the registered contrasts, the two estimands, the 15
    three-member Holm families, the non-inferiority gates, the mechanism
    contrasts, the exploratory slices and the residual readouts.

``ksweep``
    Reads the five ``same_entity`` ``summary.yaml`` files and applies the
    registered D9 k-selection rule, writing a selection record.

What this tool will NOT do
--------------------------
* It never opens ``models/embeddings/seq_stage2/cohort/`` (Astra ruled the
  cohort ``DEFERRED_UNOPENED`` for this stage) or
  ``models/embeddings/seq_stage2/smoke/`` (no smoke log loss is read
  anywhere, D6.2), or anything under ``data/golden/`` or
  ``data/forward_holdout/``.  Every file open goes through ``guard_path``.
* It emits no status containing the word "advance": the only statuses are
  ``SCREEN_PASS``, ``SCREEN_NOT_PASS`` and ``NOT_EVALUABLE``.
* It never lets an unavailable family member shrink a family or let it pass
  (D10.4): the member gets a non-rejecting placeholder and the family is
  ``NOT_EVALUABLE``.
* It never replaces the ``summary.yaml`` log loss with the log loss
  reconstructed from the saved probabilities; both are reported, labelled
  apart (D10.1).
* It imports no stage 1 margin and no stage 1 parity classification: this
  stage's only margin is +0.002 on log loss (D10.4).

Missing runs degrade cleanly: an unavailable arm makes every contrast that
needs it ``NOT_EVALUABLE`` and nothing crashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from registered_experiment import match_ids, row_log_loss  # noqa: E402
from sim_eval.eval_statistics import (  # noqa: E402
    AMBIGUOUS_CLUSTER_ALIAS,
    BOOTSTRAP_CONTRACT_VERSION,
    MAX_EVENT_GAP_DAYS,
    count_unique_clusters,
    load_competition_clusters,
)

# ---------------------------------------------------------------------------
# Registered constants (D10)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = REPO / "experiments" / "configs" / "seq_stage2_v1.yaml"
DEFAULT_RUNS_ROOT = REPO / "models" / "embeddings" / "seq_stage2" / "runs"
DEFAULT_FRAME_DIR = REPO / "data" / "xgb_data_i7"
DEFAULT_BLOCK_SOURCE = REPO / "data" / "t20s_json"
DEFAULT_BASE_LOGITS = (REPO / "models" / "embeddings" / "seq_stage2"
                       / "base_logits" / "validation.npz")
DEFAULT_STATS_OUT = (REPO / "models" / "embeddings" / "seq_stage2" / "stats"
                     / "stage2_statistics.json")
DEFAULT_KSWEEP_OUT = (REPO / "models" / "embeddings" / "seq_stage2" / "stats"
                      / "k_selection.json")

SPLIT = "validation"
FRAME_VERSION = "i7"
N_CLASSES = 6
REPS = 2000
RNG_SEED = 29
ALPHA = 0.05
MARGIN_LL = 0.002
K_TOLERANCE_LL = 0.002
MIN_BLOCKS = 10
REGISTERED_SEEDS = (7, 13)
REGISTERED_K_ORDER = ("0", "6", "12", "30", "unr")
SAME_ENTITY_ARM = "same_entity"
N_FAMILIES = 15
FAMILY_SIZE = 3
FAMILY_MEMBER_ORDER = ("primary", "death_gate", "chase_gate")

# The frame's own class mapping. Held here so the tool imports no torch;
# `test_stage2_stats.py` asserts it equals `embeddings_e1.CLASS_MAPPING`.
CLASS_MAPPING = {0: 0, 1: 1, 2: 2, 4: 3, 6: 4, -1: 5}

EXPECTED_ROWS = 124292
EXPECTED_MATCHES = 545
EXPECTED_BLOCKS = 47
EXPECTED_UNMAPPED = 0

RUN_ARTEFACTS = ("metrics.json", "model.pt", "run_record.json",
                 "predictions_validation.npz")
COMPLETION_RECORD = "COMPLETE.json"

STATUS_PASS = "SCREEN_PASS"
STATUS_NOT_PASS = "SCREEN_NOT_PASS"
STATUS_NOT_EVALUABLE = "NOT_EVALUABLE"
ALLOWED_STATUSES = (STATUS_PASS, STATUS_NOT_PASS, STATUS_NOT_EVALUABLE)

RANK_LOCAL_NOTE = ("rank-local percentile interval; not simultaneous; "
                   "not the rejection rule")
ESTIMAND_II_LABEL = ("descriptive two-seed robustness screen (joint seed and "
                     "tournament-block resampling); not five-seed evidence "
                     "and not uncertainty for a newly trained single "
                     "checkpoint")
P_CONVENTION = ("p_raw = min(1, 2 * min[P(d* - t <= 0), P(d* - t >= 0)]) over "
                "the bootstrap delta draws, with t = 0 for superiority and "
                "t = +0.002 for non-inferiority")
HOLM_FORMULA = ("p_holm(r) = min(1, max over j <= r of (4 - j) * p(j)) with "
                "the step-down stopping rule, within each three-member "
                "family, per estimand")
SIGN_CONVENTION = ("candidate minus reference, in validation log loss; "
                   "negative favourable")
RECONSTRUCTED_LL_LABEL = ("log loss reconstructed from the run's saved "
                          "row-aligned probabilities; reported separately "
                          "from the summary.yaml log loss used in D9 and "
                          "never a replacement for it")

# Paths no code path in stage 2 statistics may open (D5.8, D6.2, D10.10).
FORBIDDEN_FRAGMENTS = ("seq_stage2/cohort", "seq_stage2/smoke",
                       "data/golden", "forward_holdout")


class RefusalError(RuntimeError):
    """A registered precondition failed; nothing is reported."""


def guard_path(path: Path | str) -> Path:
    """Refuse the cohort, the smoke tree and the two sealed holdouts."""
    resolved = Path(path)
    text = resolved.as_posix()
    for fragment in FORBIDDEN_FRAGMENTS:
        if fragment in text:
            raise RefusalError(
                f"refusing to open {text}: this stage may not read "
                f"{fragment!r} (cohort DEFERRED_UNOPENED; no smoke log loss; "
                "no sealed holdout)")
    return resolved


def read_text(path: Path | str) -> str:
    return guard_path(path).read_text()


def read_json(path: Path | str) -> Any:
    return json.loads(read_text(path))


def read_yaml(path: Path | str) -> Any:
    return yaml.safe_load(read_text(path))


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with guard_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def md5_file(path: Path | str) -> str:
    digest = hashlib.md5()
    with guard_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


# ---------------------------------------------------------------------------
# The analysis pin: what every hash is checked against
# ---------------------------------------------------------------------------

@dataclass
class Pin:
    """The config's generated `provenance` block, as far as this tool needs it.

    Nothing in this module trusts a sidecar's or a summary's claim about what
    it was built from: every hash is compared against this pin, and a
    disagreement refuses instead of being copied into the report.
    """

    available: bool
    frame_dir: str | None = None
    frame_version: str | None = None
    validation_md5: str | None = None
    validation_rows: int | None = None
    feature_hash: dict = field(default_factory=dict)
    stats_cache_md5: str | None = None
    base_logits: dict = field(default_factory=dict)
    config_body_sha256: str | None = None
    reason: str | None = None


def load_pin(config: Mapping[str, Any]) -> Pin:
    provenance = config.get("provenance")
    if not isinstance(provenance, Mapping):
        return Pin(available=False,
                   reason="the config carries no generated `provenance` "
                          "block, so there is no pin to verify against; run "
                          "`pin_stage2.py --write`")
    frame = provenance.get("frame") or {}
    validation = ((frame.get("splits") or {}).get(SPLIT) or {})
    return Pin(
        available=True,
        frame_dir=frame.get("dir"),
        frame_version=frame.get("version"),
        validation_md5=validation.get("md5"),
        validation_rows=(None if validation.get("n_rows") is None
                         else int(validation["n_rows"])),
        feature_hash=dict(frame.get("feature_hash") or {}),
        stats_cache_md5=(provenance.get("stats_cache") or {}).get("md5"),
        base_logits=dict((provenance.get("base_logits") or {}).get("splits")
                         or {}),
        config_body_sha256=provenance.get("config_body_sha256"))


# ---------------------------------------------------------------------------
# D9 — the k-to-configuration mapping, derived from registration
# ---------------------------------------------------------------------------

def k_configurations(config: Mapping[str, Any]) -> list[tuple[str, str]]:
    """``[(k, config_id)]`` in registered sweep order, from the config itself.

    The registered ids are NOT a string template: the unrestricted arm is
    `same_entity_unr`, with no `k` before `unr`, so building an id by
    concatenation silently invents a directory that does not exist and turns
    a complete sweep into `BLOCKED_INCOMPLETE`. The mapping is therefore read
    out of the `configurations` list by `arm` and `params.k`.
    """
    by_k: dict[str, list[str]] = {}
    for entry in config.get("configurations") or []:
        if str(entry.get("arm")) != SAME_ENTITY_ARM:
            continue
        params = entry.get("params") or {}
        if "k" not in params:
            raise RefusalError(
                f"configuration {entry.get('id')!r} has arm "
                f"{SAME_ENTITY_ARM!r} but registers no `params.k`")
        by_k.setdefault(str(params["k"]), []).append(str(entry["id"]))
    duplicates = {k: ids for k, ids in by_k.items() if len(ids) > 1}
    if duplicates:
        raise RefusalError(
            f"more than one configuration registers the same k: {duplicates}")
    missing = [k for k in REGISTERED_K_ORDER if k not in by_k]
    if missing:
        raise RefusalError(
            f"the config registers no `{SAME_ENTITY_ARM}` configuration for "
            f"k {missing}; the registered sweep is "
            f"{list(REGISTERED_K_ORDER)}")
    extra = sorted(set(by_k) - set(REGISTERED_K_ORDER))
    if extra:
        raise RefusalError(
            f"the config registers unregistered k values {extra}; the "
            f"registered sweep is {list(REGISTERED_K_ORDER)}")
    return [(k, by_k[k][0]) for k in REGISTERED_K_ORDER]


# ---------------------------------------------------------------------------
# D10.7 — frozen slice predicates
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SlicePredicate:
    name: str
    expression: str
    columns: tuple[str, ...]
    role: str
    mask: Any
    note: str = ""


def _flag(column: str):
    return lambda df: df[column].to_numpy(np.int64) == 1


SLICE_PREDICATES: tuple[SlicePredicate, ...] = (
    SlicePredicate("all", "every validation row", (), "primary",
                   lambda df: np.ones(len(df), dtype=bool)),
    SlicePredicate("death", "is_death_overs == 1", ("is_death_overs",),
                   "gate", _flag("is_death_overs"),
                   "the frame's own death-over flag (overs 16-20)"),
    SlicePredicate("chase", "chase_target > 0", ("chase_target",), "gate",
                   lambda df: df["chase_target"].to_numpy(np.float64) > 0,
                   "the frame's own chase flag, identical to the CTX "
                   "second-innings feature the models see "
                   "(embeddings_e1.make_ctx)"),
    SlicePredicate("powerplay", "is_powerplay == 1", ("is_powerplay",),
                   "exploratory", _flag("is_powerplay")),
    SlicePredicate("middle", "is_middle_overs == 1", ("is_middle_overs",),
                   "exploratory", _flag("is_middle_overs")),
    SlicePredicate("innings_1", "inning_idx == 1", ("inning_idx",),
                   "exploratory",
                   lambda df: df["inning_idx"].to_numpy(np.int64) == 1),
    SlicePredicate("innings_2", "inning_idx == 2", ("inning_idx",),
                   "exploratory",
                   lambda df: df["inning_idx"].to_numpy(np.int64) == 2),
)

THIN_PAIR_NAME = "thin_pair"
THIN_PAIR_UNAVAILABLE_REASON = (
    "thin_pair needs per-row batter and bowler EXPOSURE columns (career or "
    "as-of ball counts) and a threshold. The i7 ball frame carries no "
    "exposure column: `batter_balls_faced` and `bowler_balls_in_innings` are "
    "within-innings counters, not exposure, and the eval kit that carries "
    "`train_balls_batting` / `train_balls_bowling` is off for this stage "
    "(`eval_kit: none`, `--no-kit`). D10.7 requires the slice to name its "
    "exposure columns and threshold or be reported unavailable, so it is "
    "reported unavailable rather than invented. To enable it, register "
    "`statistics.slice_predicates.thin_pair.{exposure_columns, threshold}` "
    "in the config and re-run.")

GATE_SLICES = ("death", "chase")
PRIMARY_SLICE = "all"


def thin_pair_spec(config: Mapping[str, Any]) -> dict | None:
    """The registered thin_pair exposure columns and threshold, or None.

    D10.7: never invented.  Only an explicit config registration turns the
    slice on.
    """
    spec = (((config.get("statistics") or {}).get("slice_predicates") or {})
            .get(THIN_PAIR_NAME))
    if not isinstance(spec, Mapping):
        return None
    columns = spec.get("exposure_columns")
    threshold = spec.get("threshold")
    if not columns or threshold is None:
        return None
    return {"exposure_columns": [str(c) for c in columns],
            "threshold": float(threshold),
            "comparison": str(spec.get("comparison") or "lt")}


# ---------------------------------------------------------------------------
# D10.1 — the pinned validation frame
# ---------------------------------------------------------------------------

@dataclass
class Frame:
    """The pinned validation parquet: row order, labels, slice masks."""

    path: Path
    sha256: str
    md5: str
    n_rows: int
    innings_id: np.ndarray
    y: np.ndarray
    match_id: np.ndarray
    masks: dict[str, np.ndarray]
    predicates: list[dict]


def frame_path(frame_dir: Path, version: str = FRAME_VERSION,
               split: str = SPLIT) -> Path:
    return Path(frame_dir) / f"cricket_data_{version}_{split}.parquet"


def load_frame(frame_dir: Path, config: Mapping[str, Any] | None = None,
               version: str = FRAME_VERSION, split: str = SPLIT) -> Frame:
    """Read the pinned validation parquet and freeze the slice predicates.

    The row ORDER of this parquet is the alignment reference for every run
    (D10.1): nothing here joins or sorts by innings id.
    """
    path = guard_path(frame_path(frame_dir, version, split))
    if not path.exists():
        raise RefusalError(f"validation parquet not found: {rel(path)}")
    config = dict(config or {})
    pin = load_pin(config)
    if not pin.available:
        raise RefusalError(
            f"refusing to read {rel(path)}: {pin.reason}. The frame hash must "
            "be compared against the pin, not merely recorded")
    frame_md5 = md5_file(path)
    if pin.validation_md5 and frame_md5 != pin.validation_md5:
        raise RefusalError(
            f"{rel(path)} has md5 {frame_md5}, the pin records "
            f"{pin.validation_md5}: the frame drifted and no number may be "
            "computed from it")
    if pin.frame_dir and Path(pin.frame_dir).name != Path(frame_dir).name:
        raise RefusalError(
            f"frame dir {rel(frame_dir)} is not the pinned "
            f"{pin.frame_dir!r}")
    spec = thin_pair_spec(config)
    present = set(pq.ParquetFile(path).schema_arrow.names)
    wanted = ["innings_id", "ball_outcome"]
    for predicate in SLICE_PREDICATES:
        wanted.extend(predicate.columns)
    # A registered exposure column that the frame does not carry makes
    # thin_pair unavailable; it must not break the read of every other slice.
    thin_pair_missing = ([] if spec is None else
                         [c for c in spec["exposure_columns"]
                          if c not in present])
    if spec is not None and not thin_pair_missing:
        wanted.extend(spec["exposure_columns"])
    columns = list(dict.fromkeys(wanted))
    try:
        df = pd.read_parquet(path, columns=columns)
    except Exception as error:  # noqa: BLE001 - a missing column is a refusal
        raise RefusalError(
            f"{rel(path)} does not carry every pinned predicate column "
            f"{columns}: {error}") from error

    innings = df["innings_id"].astype(str).to_numpy()
    outcome = df["ball_outcome"]
    mapped = outcome.map(CLASS_MAPPING)
    if mapped.isna().any():
        bad = sorted(set(outcome[mapped.isna()].tolist()))[:8]
        raise RefusalError(
            f"{rel(path)} carries ball_outcome values outside the registered "
            f"6-class mapping {CLASS_MAPPING}: {bad}")
    y = mapped.to_numpy(np.int64)

    masks: dict[str, np.ndarray] = {}
    predicates: list[dict] = []
    for predicate in SLICE_PREDICATES:
        for column in predicate.columns:
            if df[column].isna().any():
                raise RefusalError(
                    f"slice {predicate.name!r} reads {column!r}, which has "
                    "null values in the pinned frame; missing-value handling "
                    "must be registered before the slice is computed")
        mask = np.asarray(predicate.mask(df), dtype=bool)
        masks[predicate.name] = mask
        predicates.append({"slice": predicate.name,
                           "predicate": predicate.expression,
                           "columns": list(predicate.columns),
                           "role": predicate.role,
                           "note": predicate.note,
                           "available": True,
                           "n_rows": int(mask.sum())})

    if spec is None:
        predicates.append({"slice": THIN_PAIR_NAME,
                           "predicate": None,
                           "columns": [],
                           "role": "exploratory",
                           "available": False,
                           "unavailable_reason": THIN_PAIR_UNAVAILABLE_REASON,
                           "n_rows": 0})
    else:
        missing = thin_pair_missing
        if missing:
            predicates.append({"slice": THIN_PAIR_NAME, "predicate": None,
                               "columns": spec["exposure_columns"],
                               "role": "exploratory", "available": False,
                               "unavailable_reason":
                                   f"registered exposure columns absent from "
                                   f"the frame: {missing}",
                               "n_rows": 0})
        else:
            mask = np.ones(len(df), dtype=bool)
            for column in spec["exposure_columns"]:
                values = df[column].to_numpy(np.float64)
                mask &= values < spec["threshold"]
            masks[THIN_PAIR_NAME] = mask
            predicates.append({
                "slice": THIN_PAIR_NAME,
                "predicate": " and ".join(
                    f"{c} < {spec['threshold']}"
                    for c in spec["exposure_columns"]),
                "columns": spec["exposure_columns"],
                "threshold": spec["threshold"],
                "role": "exploratory", "available": True,
                "n_rows": int(mask.sum())})

    if (pin.validation_rows is not None
            and int(len(df)) != pin.validation_rows):
        raise RefusalError(
            f"{rel(path)} holds {len(df)} rows, the pin records "
            f"{pin.validation_rows}")
    return Frame(path=path, sha256=sha256_file(path), md5=frame_md5,
                 n_rows=int(len(df)), innings_id=innings, y=y,
                 match_id=match_ids(innings), masks=masks,
                 predicates=predicates)


def assert_alignment(name: str, probs: np.ndarray, y: np.ndarray,
                     innings_id: np.ndarray, frame: Frame) -> None:
    """Refuse anything that is not row-for-row the pinned frame (D10.1).

    Asserted BEFORE any differencing: shapes, finiteness, six valid classes,
    rows summing to one, the label vector and the innings-id vector in the
    parquet's own order.  No join, no sort.
    """
    if probs.ndim != 2 or probs.shape[1] != N_CLASSES:
        raise RefusalError(
            f"{name}: probabilities have shape {probs.shape}, expected "
            f"(n, {N_CLASSES})")
    if probs.shape[0] != frame.n_rows:
        raise RefusalError(
            f"{name}: {probs.shape[0]} prediction rows against the pinned "
            f"frame's {frame.n_rows}")
    if len(y) != frame.n_rows or len(innings_id) != frame.n_rows:
        raise RefusalError(
            f"{name}: label/innings vectors are {len(y)}/{len(innings_id)} "
            f"long against the pinned frame's {frame.n_rows}")
    if not np.isfinite(probs).all():
        raise RefusalError(f"{name}: probabilities are not all finite")
    if (probs < 0).any() or (probs > 1 + 1e-4).any():
        raise RefusalError(f"{name}: probabilities outside [0, 1]")
    sums = probs.astype(np.float64).sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-4):
        worst = float(np.max(np.abs(sums - 1.0)))
        raise RefusalError(
            f"{name}: probability rows do not sum to 1 (max deviation "
            f"{worst:.3g})")
    y = np.asarray(y).astype(np.int64)
    if y.min() < 0 or y.max() >= N_CLASSES:
        raise RefusalError(f"{name}: labels outside 0..{N_CLASSES - 1}")
    if not np.array_equal(y, frame.y):
        first = int(np.flatnonzero(y != frame.y)[0])
        raise RefusalError(
            f"{name}: label vector disagrees with the pinned frame at row "
            f"{first} ({y[first]} vs {frame.y[first]}); the run is not "
            "row-aligned to the parquet and nothing may be differenced")
    saved = np.asarray(innings_id).astype(str)
    if not np.array_equal(saved, frame.innings_id):
        first = int(np.flatnonzero(saved != frame.innings_id)[0])
        raise RefusalError(
            f"{name}: innings_id ordering disagrees with the pinned frame at "
            f"row {first} ({saved[first]!r} vs "
            f"{frame.innings_id[first]!r}); correspondence to the parquet's "
            "ROW ORDER is required, not a join or a sort by innings id")


# ---------------------------------------------------------------------------
# D10.2 — tournament-block bootstrap
# ---------------------------------------------------------------------------

@dataclass
class Blocks:
    block_id: np.ndarray
    lookup_source: str
    lookup_sha256: str
    n_matches: int
    n_blocks: int
    unmapped: int
    contract: dict


def resolve_blocks(frame: Frame, source_dir: Path,
                   expect_rows: int = EXPECTED_ROWS,
                   expect_matches: int = EXPECTED_MATCHES,
                   expect_blocks: int = EXPECTED_BLOCKS,
                   expect_unmapped: int = EXPECTED_UNMAPPED) -> Blocks:
    """Map every row to its I3 tournament block, or refuse (D10.2).

    Missing or ambiguous mappings are refused; no per-match block is ever
    invented as a fallback.
    """
    if BOOTSTRAP_CONTRACT_VERSION != "tournament_time_block_v1":
        raise RefusalError(
            "the bootstrap contract is "
            f"{BOOTSTRAP_CONTRACT_VERSION!r}, not tournament_time_block_v1")
    if MAX_EVENT_GAP_DAYS != 120:
        raise RefusalError(
            f"MAX_EVENT_GAP_DAYS is {MAX_EVENT_GAP_DAYS}, not 120")
    guard_path(source_dir)
    lookup = load_competition_clusters(source_dir)
    lookup_sha = sha256_text(json.dumps(sorted(lookup.items()),
                                        sort_keys=True))
    unique = sorted(set(frame.match_id.tolist()))
    missing = [m for m in unique if m not in lookup]
    ambiguous = [m for m in unique
                 if lookup.get(m) == AMBIGUOUS_CLUSTER_ALIAS]
    if missing or ambiguous:
        raise RefusalError(
            f"{len(missing)} validation matches have no tournament block and "
            f"{len(ambiguous)} resolve to the ambiguous doubleheader alias "
            f"(first missing {missing[:5]}, first ambiguous {ambiguous[:5]}); "
            "refusing rather than inventing per-match blocks")
    block_id = np.asarray([lookup[m] for m in frame.match_id.tolist()],
                          dtype=object)
    n_blocks = count_unique_clusters(block_id)
    facts = {"rows": frame.n_rows, "matches": len(unique),
             "blocks": int(n_blocks), "unmapped": len(missing)}
    expected = {"rows": expect_rows, "matches": expect_matches,
                "blocks": expect_blocks, "unmapped": expect_unmapped}
    if facts != expected:
        raise RefusalError(
            f"validation totals {facts} != the registered {expected}")
    return Blocks(block_id=block_id,
                  lookup_source=rel(source_dir),
                  lookup_sha256=lookup_sha,
                  n_matches=len(unique), n_blocks=int(n_blocks),
                  unmapped=len(missing),
                  contract={
                      "bootstrap_contract_version": BOOTSTRAP_CONTRACT_VERSION,
                      "max_event_gap_days": int(MAX_EVENT_GAP_DAYS),
                      "block_key": "cricsheet match id",
                      "resample": "complete tournament blocks, not matches",
                      "weighting": "ball_weighted",
                      "min_blocks": MIN_BLOCKS,
                      "below_min_blocks": "descriptive only; no CI-clean "
                                          "claim and no gate pass",
                      "resampled_unit_passed_to_estimator": "block ids",
                  })


def bootstrap_draws(values: np.ndarray, blocks: np.ndarray,
                    reps: int = REPS, seed: int = RNG_SEED) -> np.ndarray:
    """Retained replicate estimates of the registered block bootstrap.

    ``values`` has shape ``(n_seeds, n_rows)`` and ``blocks`` carries one
    BLOCK id per row (never a match id, D10.2).  With one seed row this
    reproduces ``registered_experiment.match_cluster_ci`` statement for
    statement; with several it reproduces
    ``registered_experiment.seed_mean_match_cluster_ci`` — each replicate
    draws S seed indices with replacement, then B block indices with
    replacement, applies the SAME sampled seeds and blocks to both arms
    (the values are already paired deltas), and forms
    ``summed sampled losses / (S * summed sampled block row counts)``.
    It is not the log loss of averaged probabilities.  Draws are retained so
    percentiles, tail probabilities and rank-local levels all come from one
    set of replicates.
    """
    values = np.atleast_2d(np.asarray(values, dtype=np.float64))
    blocks = np.asarray(blocks)
    if values.shape[1] != len(blocks):
        raise RefusalError("delta rows and block ids are not aligned")
    if values.shape[1] == 0:
        raise RefusalError("cannot bootstrap an empty row set")
    if reps <= 0:
        raise RefusalError("reps must be positive")
    _, inverse = np.unique(blocks, return_inverse=True)
    n_blocks = int(inverse.max()) + 1
    counts = np.bincount(inverse, minlength=n_blocks)
    n_seeds = values.shape[0]
    rng = np.random.default_rng(seed)
    estimates = np.empty(reps, dtype=np.float64)
    if n_seeds == 1:
        sums = np.bincount(inverse, weights=values[0], minlength=n_blocks)
        for index in range(reps):
            sampled = rng.integers(0, n_blocks, size=n_blocks)
            estimates[index] = sums[sampled].sum() / counts[sampled].sum()
        return estimates
    sums = np.stack([np.bincount(inverse, weights=row, minlength=n_blocks)
                     for row in values])
    for index in range(reps):
        sampled_seeds = rng.integers(0, n_seeds, size=n_seeds)
        sampled_blocks = rng.integers(0, n_blocks, size=n_blocks)
        numerator = sums[sampled_seeds][:, sampled_blocks].sum()
        denominator = n_seeds * counts[sampled_blocks].sum()
        estimates[index] = numerator / denominator
    return estimates


def joint_seed_block_draws(per_seed_values: Sequence[np.ndarray],
                           blocks: np.ndarray, reps: int = REPS,
                           seed: int = RNG_SEED) -> np.ndarray:
    """Estimand (ii): joint seed-and-block resampling of the across-seed mean."""
    stacked = np.stack([np.asarray(v, dtype=np.float64)
                        for v in per_seed_values])
    if stacked.shape[0] < 2:
        raise RefusalError("estimand (ii) needs at least two seeds")
    return bootstrap_draws(stacked, blocks, reps=reps, seed=seed)


def percentile_interval(draws: np.ndarray, level: float) -> list[float]:
    if not 0.0 < level < 1.0:
        raise RefusalError("confidence level must lie in (0, 1)")
    alpha = 1.0 - level
    return [float(np.percentile(draws, 100.0 * alpha / 2.0)),
            float(np.percentile(draws, 100.0 * (1.0 - alpha / 2.0)))]


def threshold_centred_p(draws: np.ndarray, threshold: float,
                        reps: int | None = None) -> dict:
    """``p_raw = min(1, 2*min[P(d*-t<=0), P(d*-t>=0)])`` (D10.4 convention).

    An empty tail is not exact zero evidence: with R draws the value is only
    known to be below ``2/R``, so that bound is what is carried into Holm and
    the display reads ``< 0.001`` at R = 2000, with a resolution flag.
    """
    draws = np.asarray(draws, dtype=np.float64)
    if draws.size == 0:
        raise RefusalError("cannot form a p-value from no draws")
    reps = int(reps or draws.size)
    centred = draws - float(threshold)
    p_le = float(np.count_nonzero(centred <= 0.0)) / draws.size
    p_ge = float(np.count_nonzero(centred >= 0.0)) / draws.size
    smaller = min(p_le, p_ge)
    floor = smaller == 0.0
    bound = 2.0 / reps
    p_raw = bound if floor else min(1.0, 2.0 * smaller)
    return {"p_raw": float(p_raw),
            "p_resolution_floor": bool(floor),
            "p_display": (f"< {bound:.3f}" if floor else f"{p_raw:.4f}"),
            "threshold": float(threshold),
            "tail_p_le": p_le, "tail_p_ge": p_ge,
            "n_draws": int(draws.size)}


# ---------------------------------------------------------------------------
# D10.4 — Holm step-down within a three-member family
# ---------------------------------------------------------------------------

def holm_order(p_values: Sequence[float]) -> list[int]:
    """Ascending rank order; ties stable in the registered member order."""
    return sorted(range(len(p_values)), key=lambda i: (p_values[i], i))


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """``min(1, max_{j<=r} (m-j+1) p_(j))`` in input order; monotone in rank."""
    values = [float(p) for p in p_values]
    if any(not np.isfinite(p) or p < 0.0 or p > 1.0 for p in values):
        raise RefusalError("raw p-values must be finite and within [0, 1]")
    m = len(values)
    adjusted = [1.0] * m
    running = 0.0
    for rank, index in enumerate(holm_order(values), start=1):
        running = max(running, (m - rank + 1) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def holm_step_down(p_values: Sequence[float],
                   alpha: float = ALPHA) -> list[bool]:
    """Step-down rejection flags; the first rank that fails stops the walk."""
    values = [float(p) for p in p_values]
    if any(not np.isfinite(p) or p < 0.0 or p > 1.0 for p in values):
        raise RefusalError("raw p-values must be finite and within [0, 1]")
    m = len(values)
    rejected = [False] * m
    for rank, index in enumerate(holm_order(values), start=1):
        if (m - rank + 1) * values[index] <= alpha:
            rejected[index] = True
        else:
            break
    return rejected


def holm_rank(p_values: Sequence[float]) -> list[int]:
    ranks = [0] * len(p_values)
    for rank, index in enumerate(holm_order(p_values), start=1):
        ranks[index] = rank
    return ranks


def holm_level(rank: int, m: int = FAMILY_SIZE, alpha: float = ALPHA) -> float:
    """Rank-local interval level ``1 - alpha/(m - r + 1)``; reported only."""
    if not 1 <= rank <= m:
        raise RefusalError("rank must lie in [1, m]")
    return 1.0 - alpha / (m - rank + 1)


# ---------------------------------------------------------------------------
# Run loading
# ---------------------------------------------------------------------------

@dataclass
class Run:
    config_id: str
    seed: int
    directory: Path
    admitted: bool
    reason: str | None = None
    artefacts: dict = field(default_factory=dict)
    row_ll: np.ndarray | None = None
    reconstructed_ll: float | None = None
    summary_ll: float | None = None
    checkpoint_md5: str | None = None
    arm_params: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    admission: Admission | None = None


# ---------------------------------------------------------------------------
# Read-only admission verifier
# ---------------------------------------------------------------------------
#
# Existence plus row alignment is not enough to admit a run: a directory can
# hold a complete, aligned, perfectly readable set of artefacts that belong to
# a different configuration, a different seed, or a different build of the
# training code. Every entry point in this module — the contrast machinery,
# the k sweep and the base-only readout — goes through this verifier BEFORE a
# number is read, and it rejects rather than warns.
#
# On the training signature: the driver's signature is seed-INDEPENDENT but
# deliberately arm-DEPENDENT — `config_id`, `arm`, `arm_params` and
# `implementation` are components of it (retrain_stage2.SIGNATURE_COMPONENTS),
# so two different arms cannot share one signature and requiring that would
# reject every real night. What must agree across arms is the part that
# describes the DATA and the training block, and that is checked component by
# component below.

SIGNATURE_SHARED_COMPONENTS = ("training_block", "frame", "stats_cache")
SIGNATURE_ARM_SPECIFIC_COMPONENTS = ("config_id", "arm", "arm_params",
                                     "implementation")


@dataclass
class Admission:
    """Whether one run directory may contribute a number, and why not."""

    ok: bool
    reason: str | None = None
    training_signature: str | None = None
    components: dict = field(default_factory=dict)
    checkpoint_md5: str | None = None


def _int_or_none(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def verify_run_admission(directory: Path, config_id: str, seed: int,
                         expected_arm: str | None = None,
                         expected_k: Any = "__unset__") -> Admission:
    """Reject a run whose own records disagree with what the tree claims.

    Checks, in order: the four artefacts plus `COMPLETE.json` exist and parse;
    `metrics.json`, `run_record.json` and `COMPLETE.json` all name this
    configuration id and this seed; the arm and `k` match registration; a
    `training_signature` is present and identical in `run_record.json` and
    `COMPLETE.json`; and the completion record's per-artefact size and md5
    manifest still matches the files on disk.
    """
    directory = guard_path(directory)
    if not directory.is_dir():
        return Admission(False, f"no run directory {rel(directory)}")
    missing = [name for name in RUN_ARTEFACTS + (COMPLETION_RECORD,)
               if not (directory / name).exists()]
    if missing:
        return Admission(False, f"incomplete run; missing {missing}")
    try:
        metrics = read_json(directory / "metrics.json")
        record = read_json(directory / "run_record.json")
        completion = read_json(directory / COMPLETION_RECORD)
    except RefusalError:
        raise
    except Exception as error:  # noqa: BLE001 - unreadable record
        return Admission(False, f"unreadable run records: {error}")
    for name, payload in (("run_record.json", record),
                          (COMPLETION_RECORD, completion)):
        if not isinstance(payload, Mapping):
            return Admission(False, f"{name} is not an object")
        if str(payload.get("config_id")) != str(config_id):
            return Admission(
                False, f"{name} names config_id "
                       f"{payload.get('config_id')!r}, the tree says "
                       f"{config_id!r}")
        if _int_or_none(payload.get("seed")) != int(seed):
            return Admission(
                False, f"{name} names seed {payload.get('seed')!r}, the tree "
                       f"says {seed}")
    metrics_config = (metrics or {}).get("config") or {}
    if _int_or_none(metrics_config.get("seed")) != int(seed):
        return Admission(
            False, f"metrics.json trained seed {metrics_config.get('seed')!r}, "
                   f"the tree says {seed}")
    arm_params = (metrics or {}).get("arm_params") or {}
    if not arm_params:
        return Admission(False, "metrics.json carries no arm_params block")
    if expected_arm is not None and str(arm_params.get("arm")) != str(
            expected_arm):
        return Admission(
            False, f"metrics.json arm_params.arm {arm_params.get('arm')!r} != "
                   f"the registered {expected_arm!r}")
    if expected_k != "__unset__" and str(arm_params.get("k")) != str(
            expected_k):
        return Admission(
            False, f"metrics.json arm_params.k {arm_params.get('k')!r} != the "
                   f"registered {expected_k!r}")

    signature = record.get("training_signature")
    if not signature:
        return Admission(
            False, "run_record.json carries no training_signature, so this "
                   "checkpoint cannot be shown to belong in any table")
    if completion.get("training_signature") != signature:
        return Admission(
            False, f"{COMPLETION_RECORD} training_signature "
                   f"{completion.get('training_signature')!r} != "
                   f"run_record.json's {signature!r}")
    components = dict(record.get("training_signature_components") or {})
    completion_components = dict(
        completion.get("training_signature_components") or {})
    if completion_components and completion_components != components:
        return Admission(
            False, "run_record.json and "
                   f"{COMPLETION_RECORD} disagree about the training "
                   "signature components")

    manifest = completion.get("artifacts") or {}
    if manifest:
        for name, facts in manifest.items():
            path = directory / name
            if not path.exists():
                return Admission(False, f"{COMPLETION_RECORD} manifests "
                                        f"{name}, which is absent")
            declared_bytes = _int_or_none((facts or {}).get("bytes"))
            if (declared_bytes is not None
                    and path.stat().st_size != declared_bytes):
                return Admission(
                    False, f"{name} is {path.stat().st_size} bytes, the "
                           f"completion manifest recorded {declared_bytes}")
            declared_md5 = (facts or {}).get("md5")
            if declared_md5 and md5_file(path) != declared_md5:
                return Admission(
                    False, f"{name} md5 has changed since the run completed "
                           f"(manifest {declared_md5})")
    checkpoint = manifest.get("model.pt") or {}
    return Admission(True, None, str(signature), components,
                     checkpoint.get("md5") or record.get("checkpoint_md5"))


def assert_comparable(admissions: Mapping[str, Mapping[int, Admission]]
                      ) -> dict:
    """Refuse to difference runs whose data or training block moved.

    Within one configuration every admitted seed must carry ONE training
    signature (the signature is seed-independent). Across configurations the
    shared components — the training block, the frame identity and the stats
    cache — must agree; an arm-specific component legitimately differs.
    """
    per_config: dict[str, str] = {}
    shared: dict[str, tuple[str, str]] = {}
    for config_id, seeds in admissions.items():
        signatures = {seed: value.training_signature
                      for seed, value in seeds.items() if value.ok}
        if not signatures:
            continue
        distinct = sorted(set(signatures.values()))
        if len(distinct) != 1:
            raise RefusalError(
                f"{config_id}: its admitted seeds carry {len(distinct)} "
                f"different training signatures {distinct}; the signature is "
                "seed-independent, so this means the code or the data moved "
                "mid-experiment and these runs cannot be differenced")
        per_config[config_id] = distinct[0]
        for seed, value in seeds.items():
            if not value.ok or not value.components:
                continue
            for name in SIGNATURE_SHARED_COMPONENTS:
                digest = value.components.get(name)
                if digest is None:
                    continue
                previous = shared.get(name)
                if previous is None:
                    shared[name] = (digest, f"{config_id} seed {seed}")
                elif previous[0] != digest:
                    raise RefusalError(
                        f"training signature component {name!r} differs "
                        f"between {previous[1]} and {config_id} seed {seed}; "
                        "the two runs did not see the same data or the same "
                        "training block, so no contrast between them is "
                        "admissible")
    return {
        "rule": ("one seed-independent training signature per configuration; "
                 f"the shared components {list(SIGNATURE_SHARED_COMPONENTS)} "
                 "identical across every admitted configuration"),
        "arm_specific_components": list(SIGNATURE_ARM_SPECIFIC_COMPONENTS),
        "arm_specific_note": (
            "the driver's signature includes config_id, arm, arm_params and "
            "implementation, so it is arm-dependent by construction and is "
            "NOT required to match across arms; what must match is the data "
            "and training-block part, checked component by component"),
        "training_signature_by_config": per_config,
        "shared_components": {name: digest
                              for name, (digest, _) in shared.items()},
    }


def machine_provenance(run_record: Mapping[str, Any] | None,
                       metrics: Mapping[str, Any] | None) -> dict:
    """D8.8 provenance, read from the driver's own `machine_provenance` block.

    Every field is reported as recorded or as ``None``; nothing is guessed and
    nothing is reconstructed after the fact. A record written before the
    driver grew the block degrades to ``None`` cells rather than failing.
    """
    record = dict(run_record or {})
    host = dict(record.get("machine_provenance") or {})
    contract = dict((metrics or {}).get("training_contract") or {})

    def pick(*names):
        for name in names:
            for source in (host, record, contract):
                if source.get(name) not in (None, ""):
                    return source[name]
        return None

    mps_built = host.get("torch_mps_built")
    mps_available = host.get("torch_mps_available")
    device = pick("device_used", "device_requested", "device")
    if mps_available is None and mps_built is None:
        mps = device
    else:
        mps = (f"{device} (mps built {mps_built}, available "
               f"{mps_available})")
    caps = host.get("thread_caps")
    return {
        "machine": pick("machine", "machine_label"),
        "machine_label_source": host.get("machine_label_source"),
        "hostname": pick("hostname", "host", "node"),
        "chip": pick("chip", "cpu", "processor"),
        "chip_source": host.get("chip_source"),
        "os": (pick("os", "platform")
               if host.get("os_release") is None
               else f"{host.get('os')} {host.get('os_release')}"),
        "platform": host.get("platform"),
        "torch_version": pick("torch_version", "torch"),
        "mps_backend": mps,
        "mps_bit_reproducible": contract.get("mps_bit_reproducible"),
        "thread_caps": caps,
        "omp_num_threads": ((caps or {}).get("OMP_NUM_THREADS")
                            if isinstance(caps, Mapping)
                            else pick("omp_num_threads")),
        "repo_is_worktree": host.get("repo_is_worktree"),
        "seed": pick("seed"),
        "wall_seconds": record.get("wall_seconds"),
        "training_signature": record.get("training_signature"),
        "machine_confounded_with_seed": True,
        "machine_term_fitted": False,
        "recorded": bool(host),
    }


def summary_lls(runs_root: Path, config_id: str) -> tuple[dict, dict]:
    """``{seed: full-precision validation LL}`` from ``summary.yaml`` (D9.2)."""
    path = guard_path(Path(runs_root) / config_id / "summary.yaml")
    if not path.exists():
        return {}, {"summary_path": rel(path), "available": False,
                    "reason": "summary.yaml absent"}
    payload = read_yaml(path) or {}
    block = ((payload.get("splits") or {}).get(SPLIT) or {})
    rows = block.get("per_seed") or []
    values: dict[int, float] = {}
    for row in rows:
        if row.get("seed") is None or row.get("ll") is None:
            continue
        if int(row["seed"]) in values:
            raise RefusalError(
                f"{rel(path)} records seed {row['seed']} more than once; a "
                "duplicate seed row is rejected, never deduplicated")
        values[int(row["seed"])] = float(row["ll"])
    meta = {"summary_path": rel(path), "available": True,
            "sha256": sha256_file(path),
            "seeds_recorded": sorted(values),
            "n_rows": block.get("n_rows"),
            "n_matches": block.get("n_matches"),
            "runs_recorded": (payload.get("experiment") or {}).get(
                "runs_recorded"),
            "complete": (payload.get("experiment") or {}).get("complete"),
            "training_signature": (payload.get("experiment") or {}).get(
                "training_signature"),
            "per_seed": [{"seed": row.get("seed"), "ll": row.get("ll"),
                          "best_epoch": row.get("best_epoch"),
                          "wall_seconds": row.get("wall_seconds"),
                          "checkpoint_md5": row.get("checkpoint_md5"),
                          "checkpoint_dir": row.get("checkpoint_dir")}
                         for row in rows]}
    return values, meta


def load_run(runs_root: Path, config_id: str, seed: int, frame: Frame,
             summary: Mapping[int, float], entry: Mapping[str, Any] | None = None
             ) -> Run:
    """Load one admitted run, or return an unadmitted Run with the reason.

    Admission is decided by `verify_run_admission` BEFORE any log loss is
    computed, so a run whose own records disagree with the tree contributes no
    number at all.
    """
    directory = guard_path(Path(runs_root) / config_id / f"seed_{seed}")
    run = Run(config_id=config_id, seed=seed, directory=directory,
              admitted=False)
    run.artefacts = {name: (directory / name).exists()
                     for name in RUN_ARTEFACTS + (COMPLETION_RECORD,)}
    entry = dict(entry or {})
    admission = verify_run_admission(
        directory, config_id, seed,
        expected_arm=entry.get("arm"),
        expected_k=(entry.get("params") or {}).get("k", "__unset__")
        if entry else "__unset__")
    run.admission = admission
    if not admission.ok:
        run.reason = admission.reason
        return run
    try:
        metrics = read_json(directory / "metrics.json")
        payload = np.load(guard_path(directory / "predictions_validation.npz"),
                          allow_pickle=False)
        probs = np.asarray(payload["probs"])
        y = np.asarray(payload["y"])
        innings = np.asarray(payload["innings_id"]).astype(str)
    except RefusalError:
        raise
    except Exception as error:  # noqa: BLE001 - unreadable artefact
        run.reason = f"unreadable artefacts: {error}"
        return run
    assert_alignment(f"{config_id} seed {seed}", probs, y, innings, frame)
    row_ll = row_log_loss(probs.astype(np.float64), frame.y)
    run.admitted = True
    run.row_ll = row_ll
    run.reconstructed_ll = float(row_ll.mean())
    run.summary_ll = (float(summary[seed]) if seed in summary else None)
    run.arm_params = dict(metrics.get("arm_params") or {})
    record = read_json(directory / "run_record.json")
    run.checkpoint_md5 = admission.checkpoint_md5
    run.provenance = machine_provenance(record, metrics)
    return run


def load_runs(runs_root: Path, config_ids: Sequence[str], frame: Frame,
              seeds: Sequence[int] = REGISTERED_SEEDS,
              entries: Mapping[str, Mapping[str, Any]] | None = None
              ) -> tuple[dict[str, dict[int, Run]], dict, dict]:
    runs: dict[str, dict[int, Run]] = {}
    report: dict[str, Any] = {}
    admissions: dict[str, dict[int, Admission]] = {}
    for config_id in config_ids:
        summary, meta = summary_lls(runs_root, config_id)
        runs[config_id] = {}
        admissions[config_id] = {}
        rows = []
        for seed in seeds:
            run = load_run(runs_root, config_id, seed, frame, summary,
                           (entries or {}).get(config_id))
            runs[config_id][seed] = run
            admissions[config_id][seed] = run.admission or Admission(
                False, run.reason)
            entry = {"seed": seed, "admitted": run.admitted,
                     "directory": rel(run.directory),
                     "artefacts": run.artefacts,
                     "reason": run.reason,
                     "reconstructed_validation_ll": run.reconstructed_ll,
                     "reconstructed_validation_ll_label":
                         RECONSTRUCTED_LL_LABEL,
                     "summary_yaml_validation_ll": run.summary_ll,
                     "summary_minus_reconstructed": (
                         None if (run.summary_ll is None
                                  or run.reconstructed_ll is None)
                         else run.summary_ll - run.reconstructed_ll),
                     "arm_params": run.arm_params,
                     "checkpoint_md5": run.checkpoint_md5,
                     "training_signature": (run.admission.training_signature
                                            if run.admission else None),
                     "provenance": run.provenance}
            rows.append(entry)
        report[config_id] = {
            "seeds": rows,
            "admitted_seeds": sorted(s for s in seeds
                                     if runs[config_id][s].admitted),
            "complete_paired_seeds": all(runs[config_id][s].admitted
                                         for s in seeds),
            "summary": meta}
    return runs, report, assert_comparable(admissions)


# ---------------------------------------------------------------------------
# Contrast machinery (D10.1, D10.3, D10.5, D10.6)
# ---------------------------------------------------------------------------

def slice_stats(frame: Frame, blocks: Blocks, name: str) -> dict:
    mask = frame.masks.get(name)
    if mask is None:
        return {"slice": name, "available": False, "n_rows": 0,
                "n_matches": 0, "n_blocks": 0, "descriptive": True}
    n_blocks = count_unique_clusters(blocks.block_id[mask])
    return {"slice": name, "available": True,
            "n_rows": int(mask.sum()),
            "n_matches": int(len(set(frame.match_id[mask].tolist()))),
            "n_blocks": int(n_blocks),
            "descriptive": bool(n_blocks < MIN_BLOCKS),
            "min_blocks": MIN_BLOCKS}


def _readout(draws: np.ndarray, point: float, threshold: float,
             descriptive: bool, reps: int) -> dict:
    p = threshold_centred_p(draws, threshold, reps=reps)
    ci95 = percentile_interval(draws, 0.95)
    readout = {"point": float(point),
               "ci95": ci95,
               "ci95_label": "ordinary two-sided 95% percentile interval",
               "u95": float(ci95[1]),
               "l95": float(ci95[0]),
               "margin_ll": MARGIN_LL,
               "descriptive_only": bool(descriptive)}
    readout.update(p)
    readout["ci_clean_favourable"] = bool(ci95[1] < 0.0 and not descriptive)
    readout["u95_below_zero"] = bool(ci95[1] < 0.0)
    readout["u95_below_margin"] = bool(ci95[1] < MARGIN_LL)
    return readout


def evaluate_contrast(candidate: str, reference: str, slice_name: str,
                      runs: Mapping[str, Mapping[int, Run]], frame: Frame,
                      blocks: Blocks, *, role: str, label: str,
                      threshold: float = 0.0, reps: int = REPS,
                      seed: int = RNG_SEED,
                      seeds: Sequence[int] = REGISTERED_SEEDS,
                      draws_out: dict | None = None) -> dict:
    """One candidate-minus-reference contrast on one slice, both estimands.

    Every delta is candidate minus reference (D10.1).  Estimand (i) is each
    registered seed checkpoint with paired block-only uncertainty; estimand
    (ii) is the across-seed arithmetic mean with joint seed-and-block
    resampling (D10.3).  No best-seed selection and no averaging of CI
    endpoints anywhere.
    """
    key = f"{candidate}-{reference}@{slice_name}"
    record: dict[str, Any] = {
        "key": key, "candidate": candidate, "reference": reference,
        "slice": slice_name, "role": role, "label": label,
        "sign_convention": SIGN_CONVENTION,
        "threshold": float(threshold),
        "slice_stats": slice_stats(frame, blocks, slice_name),
        "estimand_i": {}, "estimand_ii": None,
        "available": False, "reason": None,
        "rank_local_note": RANK_LOCAL_NOTE,
    }
    if not record["slice_stats"]["available"]:
        record["reason"] = f"slice {slice_name!r} is unavailable"
        return record
    mask = frame.masks[slice_name]
    if int(mask.sum()) == 0:
        record["reason"] = f"slice {slice_name!r} has no rows"
        return record

    missing = []
    for arm in (candidate, reference):
        for one in seeds:
            run = (runs.get(arm) or {}).get(one)
            if run is None or not run.admitted:
                missing.append(f"{arm} seed {one}")
    if missing:
        record["reason"] = ("incomplete paired seeds: "
                            + ", ".join(sorted(missing)))
        record["missing_members"] = sorted(missing)
        return record

    descriptive = record["slice_stats"]["descriptive"]
    block_ids = blocks.block_id[mask]
    per_seed_values = []
    per_seed_points = {}
    for one in seeds:
        delta = (runs[candidate][one].row_ll[mask]
                 - runs[reference][one].row_ll[mask])
        per_seed_values.append(delta)
        point = float(delta.mean())
        per_seed_points[str(one)] = point
        draws = bootstrap_draws(delta, block_ids, reps=reps, seed=seed)
        if draws_out is not None:
            draws_out[(key, f"seed_{one}")] = draws
        record["estimand_i"][f"seed_{one}"] = _readout(
            draws, point, threshold, descriptive, reps)
        record["estimand_i"][f"seed_{one}"]["estimand"] = (
            "(i) per seed checkpoint, paired block-only uncertainty")

    joint = joint_seed_block_draws(per_seed_values, block_ids, reps=reps,
                                  seed=seed)
    if draws_out is not None:
        draws_out[(key, "seed_mean_joint")] = joint
    mean_point = float(np.mean([per_seed_points[str(s)] for s in seeds]))
    record["estimand_ii"] = _readout(joint, mean_point, threshold,
                                     descriptive, reps)
    record["estimand_ii"]["estimand"] = (
        "(ii) arithmetic across-seed mean, joint seed-and-block resampling")
    record["estimand_ii"]["label"] = ESTIMAND_II_LABEL
    record["estimand_ii"]["is_log_loss_of_averaged_probabilities"] = False

    points = [per_seed_points[str(s)] for s in seeds]
    record["per_seed_points"] = per_seed_points
    record["seed_spread"] = {"min": float(min(points)),
                             "max": float(max(points)),
                             "range": float(max(points) - min(points))}
    record["favourable_direction_count"] = int(
        sum(1 for p in points if p < threshold))
    record["n_seeds"] = len(points)
    record["available"] = True
    return record


READOUTS = ("seed_7", "seed_13", "seed_mean_joint")


def readout_of(record: Mapping[str, Any], readout: str) -> dict | None:
    if not record.get("available"):
        return None
    if readout == "seed_mean_joint":
        return record.get("estimand_ii")
    return (record.get("estimand_i") or {}).get(readout)


# ---------------------------------------------------------------------------
# D10.4 / D10.6 — families, Holm, gates
# ---------------------------------------------------------------------------

def registered_families(config: Mapping[str, Any],
                        expected: int | None = N_FAMILIES) -> list[dict]:
    """Assert and return the config's explicit 15 three-member families.

    ``expected`` is the registered family count; only a synthetic fixture ever
    passes anything else, and the real config is asserted at 15.
    """
    statistics = config.get("statistics") or {}
    families = (statistics.get("families") or {})
    raw = families.get("map")
    if not isinstance(raw, list):
        raise RefusalError(
            "the config carries no statistics.families.map; D10.4 requires an "
            "explicit family map, never a template")
    if expected is not None and len(raw) != expected:
        raise RefusalError(
            f"the config registers {len(raw)} families, not {expected}")
    ids = {str(entry["id"]) for entry in (config.get("configurations") or [])}
    expected_slices = {"primary": str(families.get("primary_slice") or "all"),
                       "death_gate": str(families.get("death_gate_slice")
                                         or "death"),
                       "chase_gate": str(families.get("chase_gate_slice")
                                         or "chase")}
    out: list[dict] = []
    seen: set[str] = set()
    for entry in raw:
        candidate = str(entry.get("candidate"))
        if candidate not in ids:
            raise RefusalError(
                f"family candidate {candidate!r} is not a registered "
                "configuration id")
        if candidate in seen:
            raise RefusalError(f"family candidate {candidate!r} is repeated")
        seen.add(candidate)
        members = []
        for name in FAMILY_MEMBER_ORDER:
            member = entry.get(name)
            if not isinstance(member, Mapping):
                raise RefusalError(
                    f"family {candidate!r} has no {name!r} member; every "
                    f"family needs exactly {FAMILY_SIZE}")
            reference = str(member.get("reference"))
            if reference not in ids:
                raise RefusalError(
                    f"family {candidate!r} member {name!r} names reference "
                    f"{reference!r}, which is not a registered configuration")
            slice_name = str(member.get("slice"))
            if slice_name != expected_slices[name]:
                raise RefusalError(
                    f"family {candidate!r} member {name!r} is registered on "
                    f"slice {slice_name!r}, expected "
                    f"{expected_slices[name]!r}")
            members.append({"member": name, "candidate": candidate,
                            "reference": reference, "slice": slice_name,
                            "contrast": str(member.get("contrast")
                                            or f"{candidate} - {reference}"),
                            "threshold": (0.0 if name == "primary"
                                          else MARGIN_LL),
                            "kind": ("superiority" if name == "primary"
                                     else "non_inferiority")})
        extra = [k for k in entry
                 if k not in {"candidate", "holm_group", "note",
                              *FAMILY_MEMBER_ORDER}]
        out.append({"candidate": candidate,
                    "holm_group": str(entry.get("holm_group")
                                      or f"family_{candidate}"),
                    "note": entry.get("note"),
                    "unknown_keys": extra,
                    "members": members})
    extra_candidates = ids - seen - {"mlp"}
    if extra_candidates:
        raise RefusalError(
            "every configuration except mlp must be a family candidate "
            f"exactly once; missing {sorted(extra_candidates)}")
    if "mlp" in seen:
        raise RefusalError("mlp is the shared control and is a candidate in "
                           "no family")
    return out


def _check_status(status: str) -> str:
    if status not in ALLOWED_STATUSES:
        raise RefusalError(f"status {status!r} is not one of "
                           f"{ALLOWED_STATUSES}")
    if "advance" in status.lower():
        raise RefusalError("this tool emits no advancement status")
    return status


def member_status(member: Mapping[str, Any], readout: Mapping[str, Any] | None,
                  rejected: bool) -> str:
    """Per-member status.  Numerical pass needs a strict upper bound (D10.6)."""
    if readout is None:
        return _check_status(STATUS_NOT_EVALUABLE)
    if readout["descriptive_only"]:
        return _check_status(STATUS_NOT_EVALUABLE)
    if member["kind"] == "non_inferiority":
        numerical = readout["u95"] < MARGIN_LL
    else:
        numerical = readout["u95"] < 0.0
    return _check_status(STATUS_PASS if (numerical and rejected)
                         else STATUS_NOT_PASS)


def holm_family(family: Mapping[str, Any],
                contrasts: Mapping[str, Mapping[str, Any]],
                readout: str) -> dict:
    """Holm step-down over one family's three members, for one readout.

    An unavailable member takes a non-rejecting placeholder (raw p 1.0,
    adjusted 1.0, no rejection) and the family cannot pass (D10.4).  The
    placeholder still occupies its slot, so an absent run can never shrink
    the family or lower the multiplier for the others.
    """
    rows = []
    for member in family["members"]:
        key = (f"{member['candidate']}-{member['reference']}"
               f"@{member['slice']}")
        record = contrasts.get(key)
        values = readout_of(record or {}, readout)
        rows.append({"member": member, "key": key, "readout": values,
                     "record_available": bool((record or {}).get(
                         "available")),
                     "record_reason": (record or {}).get("reason")})

    p_values = []
    for row in rows:
        values = row["readout"]
        if values is None or values["descriptive_only"]:
            p_values.append(1.0)
            row["placeholder"] = True
        else:
            p_values.append(float(values["p_raw"]))
            row["placeholder"] = False
    adjusted = holm_adjust(p_values)
    step = holm_step_down(p_values)
    ranks = holm_rank(p_values)

    members_out = []
    family_evaluable = True
    for row, p_raw, p_adj, reject, rank in zip(rows, p_values, adjusted,
                                               step, ranks):
        values = row["readout"]
        member = row["member"]
        favourable = (None if values is None
                      else bool(values["point"] < member["threshold"]))
        strict = (None if values is None else
                  bool(values["u95"] < (MARGIN_LL
                                        if member["kind"] == "non_inferiority"
                                        else 0.0)))
        rejected = bool(reject and not row["placeholder"]
                        and favourable and strict)
        status = member_status(member, values, rejected)
        if status == STATUS_NOT_EVALUABLE:
            family_evaluable = False
        level = holm_level(rank, FAMILY_SIZE)
        members_out.append({
            "member": member["member"],
            "contrast_key": row["key"],
            "contrast": member["contrast"],
            "slice": member["slice"],
            "kind": member["kind"],
            "threshold": member["threshold"],
            "available": row["record_available"] and values is not None,
            "reason": row["record_reason"],
            "placeholder_non_rejecting": row["placeholder"],
            "point": (None if values is None else values["point"]),
            "ci95": (None if values is None else values["ci95"]),
            "u95": (None if values is None else values["u95"]),
            "p_raw": p_raw,
            "p_display": (None if values is None else values["p_display"]),
            "p_resolution_floor": (None if values is None
                                   else values["p_resolution_floor"]),
            "p_holm": p_adj,
            "rank": rank,
            "holm_step_down_rejects": bool(reject),
            "direction_favourable": favourable,
            "strict_upper_bound_ok": strict,
            "rejected": rejected,
            "rank_local_level": level,
            "rank_local_interval": None,
            "rank_local_note": RANK_LOCAL_NOTE,
            "status": status,
        })
    return {"readout": readout, "alpha": ALPHA, "m": FAMILY_SIZE,
            "holm_formula": HOLM_FORMULA,
            "scope": "within this family only; never pooled across the 15 "
                     "families and never applied across the k search",
            "members": members_out,
            "all_members_evaluable": family_evaluable}


def rank_local_intervals(family_table: Mapping[str, Any],
                         contrasts: Mapping[str, Mapping[str, Any]],
                         draws_cache: Mapping[tuple, np.ndarray]) -> None:
    """Fill each member's rank-local interval from its own retained draws."""
    for member in family_table["members"]:
        key = (member["contrast_key"], family_table["readout"])
        draws = draws_cache.get(key)
        if draws is None or member["point"] is None:
            continue
        member["rank_local_interval"] = percentile_interval(
            draws, member["rank_local_level"])


def family_screen(family: Mapping[str, Any], tables: Mapping[str, Any],
                  contrasts: Mapping[str, Mapping[str, Any]],
                  readout: str) -> dict:
    """The registered numerical validation screen for one candidate (D10.6).

    A candidate clears it only when its registered primary is favourable
    under Holm with an ordinary upper 95% endpoint below 0, BOTH gates pass,
    and its all-row ``candidate - mlp`` interval is CI-clean favourable.
    That last reading is exploratory wherever it falls outside the family; it
    is a necessary condition here and never a replacement for the primary.
    """
    table = tables[readout]
    by_member = {row["member"]: row for row in table["members"]}
    primary = by_member["primary"]
    gates = [by_member["death_gate"], by_member["chase_gate"]]
    candidate = family["candidate"]
    all_row_key = f"{candidate}-mlp@{PRIMARY_SLICE}"
    all_row = readout_of(contrasts.get(all_row_key) or {}, readout)
    all_row_clean = (None if all_row is None
                     else bool(all_row["ci_clean_favourable"]))

    statuses = [primary["status"], *[g["status"] for g in gates]]
    if (STATUS_NOT_EVALUABLE in statuses or all_row is None
            or all_row["descriptive_only"]):
        status = STATUS_NOT_EVALUABLE
    elif (primary["status"] == STATUS_PASS
          and all(g["status"] == STATUS_PASS for g in gates)
          and all_row_clean):
        status = STATUS_PASS
    else:
        status = STATUS_NOT_PASS
    return {"readout": readout,
            "candidate": candidate,
            "status": _check_status(status),
            "primary_status": primary["status"],
            "primary_rejected": primary["rejected"],
            "primary_u95_below_zero": (
                None if primary["u95"] is None else primary["u95"] < 0.0),
            "death_gate_status": by_member["death_gate"]["status"],
            "chase_gate_status": by_member["chase_gate"]["status"],
            "all_row_candidate_minus_mlp_key": all_row_key,
            "all_row_candidate_minus_mlp_ci_clean_favourable": all_row_clean,
            "all_row_reading_is_exploratory_when_outside_the_family": (
                family["members"][0]["reference"] != "mlp"),
            "evidence_status": ("screening: two seeds, validation only, "
                               "checkpoint selected on the same split"),
            "note": ("neither a point below the margin nor a failure to "
                     "detect harm establishes non-inferiority")}


# ---------------------------------------------------------------------------
# D10.5 — mechanism contrasts
# ---------------------------------------------------------------------------

MECHANISM_CONTRASTS = (
    ("fox", "fixed_decay", "learned forgetting beyond fixed decay"),
    ("same_entity_k30", "recency_k30",
     "ownership plus alignment beyond recency"),
    ("same_entity_unr", "aligned_hist_rf",
     "mask alone, given aligned inputs and matched relay-free keys"),
    ("aligned_hist_rf", "aligned_hist",
     "the relay-free wiring PLUS the key construction, not the wiring alone"),
    ("aligned_hist", "full", "the aligned history input"),
)

MECHANISM_GUARD = (
    "a mechanism contrast is inferential only in its registered family and "
    "slice; every other slice readout is exploratory. No difference is ever "
    "inferred from one arm being significant against mlp and another not — "
    "that comparison is not computed by this tool and is not reportable")


# ---------------------------------------------------------------------------
# D10.11 — residual controls
# ---------------------------------------------------------------------------

def base_only_readout(base_npz: Path, base_sidecar: Path, frame: Frame,
                      blocks: Blocks, pin: Pin | None = None) -> dict:
    """Base-only validation log loss and its provenance (D10.11).

    Every provenance field is CHECKED, not copied: the npz md5, the sidecar
    sha256, the booster md5 and the sidecar's recorded `parquet_md5` are all
    compared against the analysis pin and against the live validation
    parquet's own md5. Any disagreement makes the readout unavailable with the
    reason, so a stale or mislabelled base-logit file contributes no number.

    The base logits are the production ball model's floored, renormalised
    probabilities on this split; ``residual_mlp`` is a PRODUCTION-PRIOR
    control, not a sequence gain.
    """
    record: dict[str, Any] = {
        "available": False, "reason": None,
        "npz": rel(base_npz), "sidecar": rel(base_sidecar),
        "role": "exploratory reference; the production prior alone",
        "not_a_stage2_number": (
            "this is the production ball model's own validation log loss "
            "(D4.3), reported as an exploratory reference; it is not a "
            "stage 2 result and is used for no selection"),
        "residual_mlp_is": ("a production-prior control, not a sequence "
                            "gain"),
    }
    base_npz = guard_path(base_npz)
    base_sidecar = guard_path(base_sidecar)
    if not base_npz.exists():
        record["reason"] = f"{rel(base_npz)} absent"
        return record
    payload = np.load(base_npz, allow_pickle=False)
    if "logp" not in payload.files:
        record["reason"] = f"{rel(base_npz)} carries no `logp`"
        return record
    logp = np.asarray(payload["logp"], dtype=np.float64)
    if logp.shape != (frame.n_rows, N_CLASSES):
        record["reason"] = (f"base logits are {logp.shape}, expected "
                            f"({frame.n_rows}, {N_CLASSES})")
        return record
    if not base_sidecar.exists():
        record["reason"] = (f"{rel(base_sidecar)} is absent, so the base "
                            "logits carry no verifiable provenance")
        return record
    sidecar = read_json(base_sidecar)
    declared_rows = sidecar.get("n_rows")
    if declared_rows is not None and int(declared_rows) != frame.n_rows:
        record["reason"] = (f"base-logit sidecar declares {declared_rows} "
                            f"rows against the frame's {frame.n_rows}")
        return record
    # The sidecar's claim about the parquet it was built from is checked
    # against the live parquet's own md5, not taken on trust.
    declared_parquet = sidecar.get("parquet_md5")
    if declared_parquet and declared_parquet != frame.md5:
        record["reason"] = (
            f"the sidecar was built from parquet md5 {declared_parquet}, the "
            f"live validation parquet is {frame.md5}")
        return record
    pinned = dict((pin.base_logits if pin else {}).get(SPLIT) or {})
    checks = {"npz_md5": (md5_file(base_npz), pinned.get("npz_md5")),
              "sidecar_sha256": (sha256_file(base_sidecar),
                                 pinned.get("sidecar_sha256")),
              "parquet_md5": (frame.md5, pinned.get("parquet_md5")),
              "booster_md5": (sidecar.get("booster_md5"),
                              pinned.get("booster_md5")),
              "n_rows": (frame.n_rows,
                         None if pinned.get("n_rows") is None
                         else int(pinned["n_rows"]))}
    if not pinned:
        record["reason"] = (
            "the analysis pin records no base-logit provenance for the "
            f"{SPLIT} split, so the npz/sidecar/booster hashes cannot be "
            "verified")
        record["hash_checks"] = {name: {"measured": got, "pinned": want}
                                 for name, (got, want) in checks.items()}
        return record
    mismatched = {name: {"measured": got, "pinned": want}
                  for name, (got, want) in checks.items()
                  if want is not None and got != want}
    record["hash_checks"] = {name: {"measured": got, "pinned": want,
                                    "agrees": want is None or got == want}
                             for name, (got, want) in checks.items()}
    if mismatched:
        record["reason"] = (
            "base-logit provenance disagrees with the analysis pin: "
            + json.dumps(mismatched, sort_keys=True, default=str))
        return record
    probs = np.exp(logp)
    probs = probs / probs.sum(axis=1, keepdims=True)
    row_ll = row_log_loss(probs, frame.y)
    record.update({
        "available": True,
        "row_ll_mean": float(row_ll.mean()),
        "n_rows": int(frame.n_rows),
        "provenance": {
            "booster_md5": sidecar.get("booster_md5"),
            "model_dir": sidecar.get("model_dir"),
            "model_role": sidecar.get("model_role"),
            "parquet_md5": sidecar.get("parquet_md5"),
            "npz_md5": sidecar.get("npz_md5"),
            "floor": sidecar.get("floor"),
            "renormalised": sidecar.get("renormalised"),
            "script_sha256": sidecar.get("script_sha256"),
            "row_alignment": ("verified: the sidecar's recorded parquet md5 "
                              "equals the live validation parquet's own md5 "
                              "and its row count equals the frame's, and the "
                              "logits are stored in parquet row order"),
            "verified_against_pin": True,
        },
        "per_slice": {name: float(row_ll[frame.masks[name]].mean())
                      for name in frame.masks
                      if int(frame.masks[name].sum()) > 0},
    })
    record["_row_ll"] = row_ll
    return record


def residual_vs_base(arm: str, runs: Mapping[str, Mapping[int, Run]],
                     base: Mapping[str, Any], frame: Frame, blocks: Blocks,
                     *, reps: int = REPS, seed: int = RNG_SEED,
                     seeds: Sequence[int] = REGISTERED_SEEDS) -> dict:
    """``arm - base_only`` on the all slice, exploratory (D10.11)."""
    record = {"key": f"{arm}-base_only@{PRIMARY_SLICE}",
              "candidate": arm, "reference": "base_only",
              "slice": PRIMARY_SLICE, "role": "exploratory",
              "label": "residual arm against the production prior alone",
              "available": False, "reason": None,
              "sign_convention": SIGN_CONVENTION}
    if not base.get("available"):
        record["reason"] = f"base-only logits unavailable: {base.get('reason')}"
        return record
    missing = [f"{arm} seed {s}" for s in seeds
               if not (runs.get(arm) or {}).get(s, Run(arm, s, Path("."),
                                                       False)).admitted]
    if missing:
        record["reason"] = "incomplete paired seeds: " + ", ".join(missing)
        return record
    base_ll = base["_row_ll"]
    stats = slice_stats(frame, blocks, PRIMARY_SLICE)
    mask = frame.masks[PRIMARY_SLICE]
    block_ids = blocks.block_id[mask]
    per_seed = []
    points = {}
    record["estimand_i"] = {}
    for one in seeds:
        delta = runs[arm][one].row_ll[mask] - base_ll[mask]
        per_seed.append(delta)
        point = float(delta.mean())
        points[str(one)] = point
        draws = bootstrap_draws(delta, block_ids, reps=reps, seed=seed)
        record["estimand_i"][f"seed_{one}"] = _readout(
            draws, point, 0.0, stats["descriptive"], reps)
    joint = joint_seed_block_draws(per_seed, block_ids, reps=reps, seed=seed)
    mean_point = float(np.mean(list(points.values())))
    record["estimand_ii"] = _readout(joint, mean_point, 0.0,
                                     stats["descriptive"], reps)
    record["estimand_ii"]["label"] = ESTIMAND_II_LABEL
    record["per_seed_points"] = points
    record["slice_stats"] = stats
    record["available"] = True
    return record


# ---------------------------------------------------------------------------
# D9 — the k sweep
# ---------------------------------------------------------------------------

def verify_summary_provenance(runs_root: Path, config_id: str,
                              entry: Mapping[str, Any] | None,
                              meta: Mapping[str, Any],
                              seeds: Sequence[int]) -> list[str]:
    """Every reason this summary's log losses may not be read (D9.2).

    Called BEFORE the log losses are used. Each recorded seed row must name a
    checkpoint directory that itself passes `verify_run_admission`, must carry
    the same training signature as the summary's own header, and the
    checkpoint md5 it claims must still be the md5 of the file on disk.
    """
    problems: list[str] = []
    if not meta.get("available"):
        return [f"{config_id}: {meta.get('reason', 'summary.yaml absent')}"]
    header_signature = meta.get("training_signature")
    if not header_signature:
        problems.append(f"{config_id}/summary.yaml carries no "
                        "training_signature")
    seen: set[int] = set()
    for row in meta.get("per_seed") or []:
        seed = _int_or_none(row.get("seed"))
        if seed is None:
            problems.append(f"{config_id}/summary.yaml has a row with no seed")
            continue
        if seed in seen:
            problems.append(f"{config_id}/summary.yaml repeats seed {seed}")
            continue
        seen.add(seed)
        if seed not in set(seeds):
            problems.append(f"{config_id}/summary.yaml records unregistered "
                            f"seed {seed}")
            continue
        value = row.get("ll")
        if value is None or not np.isfinite(float(value)):
            problems.append(f"{config_id} seed {seed}: validation LL is "
                            f"{value!r}")
        row_signature = row.get("training_signature")
        if (row_signature is not None and header_signature is not None
                and row_signature != header_signature):
            problems.append(
                f"{config_id} seed {seed}: the summary row's "
                "training_signature differs from the summary header's")
        directory = Path(runs_root) / config_id / f"seed_{seed}"
        admission = verify_run_admission(
            directory, config_id, seed,
            expected_arm=(entry or {}).get("arm"),
            expected_k=((entry or {}).get("params") or {}).get(
                "k", "__unset__") if entry else "__unset__")
        if not admission.ok:
            problems.append(f"{config_id} seed {seed}: {admission.reason}")
            continue
        if (header_signature is not None
                and admission.training_signature != header_signature):
            problems.append(
                f"{config_id} seed {seed}: the checkpoint's training "
                "signature does not match the summary it is tabulated in")
        claimed = row.get("checkpoint_md5")
        if claimed and admission.checkpoint_md5 and claimed != (
                admission.checkpoint_md5):
            problems.append(
                f"{config_id} seed {seed}: summary.yaml claims checkpoint md5 "
                f"{claimed}, the completion manifest records "
                f"{admission.checkpoint_md5}")
        declared_dir = row.get("checkpoint_dir")
        if declared_dir and Path(str(declared_dir)).parts[-2:] != (
                config_id, f"seed_{seed}"):
            problems.append(
                f"{config_id} seed {seed}: summary.yaml points at "
                f"{declared_dir!r}, not this configuration's seed directory")
    return problems


def k_sweep(runs_root: Path, config: Mapping[str, Any],
            seeds: Sequence[int] = REGISTERED_SEEDS,
            tolerance: float = K_TOLERANCE_LL) -> dict:
    """Apply the registered D9 k-selection rule to the five same_entity arms.

    Reads full-precision per-seed validation log loss from each
    configuration's ``summary.yaml`` only (D9.2, D9.4): no test rows, no
    cohort rows, no smoke value, no LL reconstructed from probabilities. The
    k-to-configuration mapping comes from the config's own `configurations`
    list, never from a string template — the unrestricted arm is registered as
    `same_entity_unr`.
    """
    if not isinstance(config, Mapping) or not config:
        raise RefusalError(
            "the k sweep needs the registered config: the k-to-configuration "
            "mapping is read from it, not built by string concatenation")
    registered = list(REGISTERED_K_ORDER)
    declared = ((config.get("statistics") or {}).get("k_selection")
                or {}).get("registered_sweep")
    if declared is not None:
        got = [str(k) for k in declared]
        if got != registered:
            raise RefusalError(
                f"the config registers sweep order {got}, not {registered}")
    declared_tolerance = ((config.get("statistics") or {})
                          .get("k_selection") or {}).get("tolerance_ll")
    if declared_tolerance is not None:
        tolerance = float(declared_tolerance)
    mapping = k_configurations(config)
    entries = {str(entry["id"]): entry
               for entry in (config.get("configurations") or [])}

    rows = []
    sources = []
    incomplete = []
    rejected: list[str] = []
    for k, config_id in mapping:
        values, meta = summary_lls(runs_root, config_id)
        problems = verify_summary_provenance(runs_root, config_id,
                                             entries.get(config_id), meta,
                                             seeds)
        sources.append({"k": k, "config_id": config_id,
                        "admission_problems": problems, **meta})
        if problems:
            # Provenance is verified BEFORE the numbers are read: a summary
            # that cannot be verified contributes no log loss at all.
            rejected.extend(problems)
            incomplete.append(config_id)
            rows.append({"k": k, "config_id": config_id,
                         "per_seed": {str(s): None for s in seeds},
                         "mean_ll": None, "min_ll": None, "max_ll": None,
                         "range_ll": None, "complete": False,
                         "rejected_reasons": problems})
            continue
        per_seed = {}
        for one in seeds:
            value = values.get(one)
            per_seed[str(one)] = (None if value is None
                                  or not np.isfinite(value) else float(value))
        recorded = [v for v in per_seed.values() if v is not None]
        if len(recorded) != len(seeds):
            incomplete.append(config_id)
        extra = sorted(set(values) - set(seeds))
        if extra:
            raise RefusalError(
                f"{config_id}/summary.yaml records unregistered seeds "
                f"{extra}; D9.2 asserts exactly {list(seeds)}")
        rows.append({"k": k, "config_id": config_id, "per_seed": per_seed,
                     "mean_ll": (float(np.mean(recorded))
                                 if len(recorded) == len(seeds) else None),
                     "min_ll": (float(min(recorded)) if recorded else None),
                     "max_ll": (float(max(recorded)) if recorded else None),
                     "range_ll": (float(max(recorded) - min(recorded))
                                  if len(recorded) == len(seeds) else None),
                     "complete": len(recorded) == len(seeds)})

    record: dict[str, Any] = {
        "tool": rel(__file__),
        "check": "D9 k sweep",
        "generated_at_utc": _now(),
        "runs_root": rel(runs_root),
        "seeds": list(seeds),
        "registered_sweep": registered,
        "tolerance_ll": float(tolerance),
        "tie_rule": ("exact ties between qualifying minima use the registered "
                     "sweep order " + str(registered) + ", recorded before "
                     "selection"),
        "rule": ("choose the k with the lowest two-seed arithmetic mean "
                 "validation log loss, but only when "
                 "mean_LL(30) - mean_LL(best) > tolerance; otherwise keep 30. "
                 "Equality at the tolerance keeps 30. Any k may win, unr "
                 "included."),
        "numbers_of_record": ("full-precision per-seed validation LL from "
                             "runs/<config_id>/summary.yaml, after D8.7 "
                             "consolidation; never rounded report values, "
                             "smoke LL, checkpoint re-scores or LL "
                             "reconstructed from saved probabilities"),
        "validation_only": ("no test predictions, test rows, cohort rows or "
                            "cohort base logits are loaded; test and cohort "
                            "results cannot change k"),
        "labelling": "two-seed directional screen",
        "interpretation_guard": (
            "the 0.002 tolerance is a selection tolerance, not a significance "
            "threshold and not demonstrated two-seed resolution; the chosen k "
            "is never called reliably optimal"),
        "provisional": True,
        "provisional_note": ("the two-seed selection remains explicitly "
                             "provisional pending any registered "
                             "whole-family seed extension"),
        "k_to_config_id": {k: config_id for k, config_id in mapping},
        "k_to_config_id_source": ("derived from the config's own "
                                  "`configurations` entries by arm and "
                                  "`params.k`, never by string template"),
        "admission_rejections": rejected,
        "rows": rows,
        "sources": sources,
        "other_k_arms": ("all five k arms are retained in the validation "
                         "report; the four not selected are named 'not "
                         "selected' and none is suppressed"),
        "matched_control_constraint": (
            "only k=30 has the registered recency_k30 comparison; "
            "same_entity_unr - aligned_hist_rf is the registered "
            "unrestricted-mask comparison; k=0/6/12 keep their registered "
            "mlp primary references and no k borrows another window's "
            "recency control"),
    }

    if incomplete:
        record.update({
            "selection": "BLOCKED_INCOMPLETE",
            "selected_config_id": None,
            "selected_k": None,
            "blocked_on": sorted(incomplete),
            "paired_vs_k30": {},
            "flags": [],
            "note": ("selection is BLOCKED_INCOMPLETE with no "
                     "default-to-30 decision: a missing k cannot justify "
                     "'keep 30 because none beat it' (D9.1)"),
        })
        return record

    ids = dict(mapping)
    by_k = {row["k"]: row for row in rows}
    means = {row["k"]: row["mean_ll"] for row in rows}
    best_k = min(registered, key=lambda k: (means[k], registered.index(k)))
    margin = means["30"] - means[best_k]
    selected_k = best_k if margin > tolerance else "30"

    paired = {}
    for k in registered:
        diffs = {str(s): by_k[k]["per_seed"][str(s)] - by_k["30"]["per_seed"][
            str(s)] for s in seeds}
        values = list(diffs.values())
        paired[k] = {
            "per_seed_delta_vs_k30": diffs,
            "mean_delta": float(np.mean(values)),
            "min_delta": float(min(values)),
            "max_delta": float(max(values)),
            "range_delta": float(max(values) - min(values)),
            "favourable_direction_count": int(sum(1 for v in values if v < 0)),
            "n_seeds": len(values),
            "spread_note": ("empirical spread over two seeds, not a "
                            "confidence interval"),
        }

    flags = []
    if selected_k != "30":
        block = paired[selected_k]
        values = list(block["per_seed_delta_vs_k30"].values())
        if min(values) < 0 < max(values):
            flags.append("paired differences change sign across the two seeds")
        if block["range_delta"] >= tolerance:
            flags.append(f"paired range {block['range_delta']:.6f} >= the "
                         f"{tolerance} tolerance")
        if block["range_delta"] >= abs(block["mean_delta"]):
            flags.append("paired range is at least the absolute mean "
                         "difference")

    record.update({
        "selection": "SELECTED",
        "means": means,
        "best_mean_k": best_k,
        "margin_vs_k30": float(margin),
        "margin_rule": "select best only when mean_LL(30) - mean_LL(best) > "
                       f"{tolerance}; equality keeps 30",
        "selected_k": selected_k,
        "selected_config_id": ids[selected_k],
        "not_selected": [ids[k] for k in registered if k != selected_k],
        "paired_vs_k30": paired,
        "flags": flags,
        "flag_note": ("flags are reported beside the selection and never "
                      "change the registered rule"),
        "source_lls": {ids[k]: by_k[k]["per_seed"] for k in registered},
    })
    record["selection_record_sha256_rule"] = (
        "sha256 of this document with `selection_record_sha256` removed")
    body = {k: v for k, v in record.items()
            if k != "selection_record_sha256"}
    record["selection_record_sha256"] = sha256_text(
        json.dumps(body, sort_keys=True, default=str))
    return record


# ---------------------------------------------------------------------------
# stats: the whole D10.1-D10.7 / D10.11 run
# ---------------------------------------------------------------------------

def compute_statistics(config_path: Path, runs_root: Path, frame_dir: Path,
                       block_source: Path, base_npz: Path, *,
                       reps: int = REPS, seed: int = RNG_SEED,
                       seeds: Sequence[int] = REGISTERED_SEEDS,
                       expect: Mapping[str, int] | None = None,
                       expected_families: int | None = N_FAMILIES) -> dict:
    config = read_yaml(config_path) or {}
    entries = {str(entry["id"]): entry
               for entry in (config.get("configurations") or [])}
    config_ids = list(entries)
    if not config_ids:
        raise RefusalError(f"{rel(config_path)} registers no configurations")
    families = registered_families(config, expected_families)
    pin = load_pin(config)

    frame = load_frame(frame_dir, config)
    expect = dict(expect or {})
    blocks = resolve_blocks(
        frame, block_source,
        expect_rows=int(expect.get("rows", EXPECTED_ROWS)),
        expect_matches=int(expect.get("matches", EXPECTED_MATCHES)),
        expect_blocks=int(expect.get("blocks", EXPECTED_BLOCKS)),
        expect_unmapped=int(expect.get("unmapped", EXPECTED_UNMAPPED)))
    runs, runs_report, comparability = load_runs(runs_root, config_ids, frame,
                                                 seeds, entries)

    slice_names = [p.name for p in SLICE_PREDICATES] + [THIN_PAIR_NAME]
    declared = (config.get("statistics") or {}).get("slices")
    if declared is not None and [str(s) for s in declared] != slice_names:
        raise RefusalError(
            f"the config registers slices {list(declared)}, this tool "
            f"computes {slice_names}")

    contrasts: dict[str, dict] = {}
    draws_cache: dict[tuple, np.ndarray] = {}

    def add(candidate: str, reference: str, slice_name: str, role: str,
            label: str, threshold: float = 0.0) -> str:
        """Register one contrast once; the first (registered) role wins."""
        key = f"{candidate}-{reference}@{slice_name}"
        if key in contrasts:
            return key
        contrasts[key] = evaluate_contrast(
            candidate, reference, slice_name, runs, frame, blocks,
            role=role, label=label, threshold=threshold, reps=reps,
            seed=seed, seeds=seeds, draws_out=draws_cache)
        return key

    # Family members first, so their registered role wins over exploratory.
    for family in families:
        for member in family["members"]:
            add(member["candidate"], member["reference"], member["slice"],
                f"family_{member['member']}", member["contrast"],
                member["threshold"])
    for candidate, reference, label in MECHANISM_CONTRASTS:
        add(candidate, reference, PRIMARY_SLICE, "mechanism", label)
    # Exploratory: every registered contrast on every available slice.
    registered_pairs = {(m["candidate"], m["reference"])
                        for f in families for m in f["members"]}
    registered_pairs |= {(c, r) for c, r, _ in MECHANISM_CONTRASTS}
    registered_pairs |= {(cid, "mlp") for cid in config_ids if cid != "mlp"}
    for candidate, reference in sorted(registered_pairs):
        for name in slice_names:
            if name not in frame.masks:
                continue
            add(candidate, reference, name, "exploratory",
                f"{candidate} - {reference} on {name} (exploratory)",
                MARGIN_LL if name in GATE_SLICES and reference == "mlp"
                else 0.0)

    family_out = []
    for family in families:
        tables = {}
        for readout in READOUTS:
            table = holm_family(family, contrasts, readout)
            rank_local_intervals(table, contrasts, draws_cache)
            tables[readout] = table
        screens = {readout: family_screen(family, tables, contrasts, readout)
                   for readout in READOUTS}
        family_out.append({
            "candidate": family["candidate"],
            "holm_group": family["holm_group"],
            "note": family["note"],
            "members": family["members"],
            "holm": tables,
            "screen": screens,
            "registered_for_confirmation_results_remain_screening": True,
        })

    gates = []
    for family in families:
        for member in family["members"]:
            if member["member"] == "primary":
                continue
            key = f"{member['candidate']}-{member['reference']}@{member['slice']}"
            record = contrasts.get(key) or {}
            row = {"candidate": member["candidate"],
                   "reference": member["reference"],
                   "slice": member["slice"],
                   "contrast_key": key,
                   "margin_ll": MARGIN_LL,
                   "numerical_pass_rule": "strictly U95 < 0.002",
                   "family_adjusted_gate_additionally_requires": (
                       "favourable Holm rejection against the +0.002 "
                       "boundary, at least 10 blocks, and complete paired "
                       "seeds"),
                   "slice_stats": record.get("slice_stats"),
                   "readouts": {}}
            for readout in READOUTS:
                values = readout_of(record, readout)
                holm_row = next(
                    (r for r in
                     next(f for f in family_out
                          if f["candidate"] == family["candidate"]
                          )["holm"][readout]["members"]
                     if r["member"] == member["member"]), None)
                row["readouts"][readout] = {
                    "point": (None if values is None else values["point"]),
                    "ci95": (None if values is None else values["ci95"]),
                    "u95": (None if values is None else values["u95"]),
                    "p_raw": (None if holm_row is None
                              else holm_row["p_raw"]),
                    "p_display": (None if values is None
                                  else values["p_display"]),
                    "p_holm": (None if holm_row is None
                               else holm_row["p_holm"]),
                    "holm_rejected": (None if holm_row is None
                                      else holm_row["rejected"]),
                    "u95_strictly_below_margin": (
                        None if values is None
                        else bool(values["u95"] < MARGIN_LL)),
                    "status": (STATUS_NOT_EVALUABLE if holm_row is None
                               else holm_row["status"]),
                }
            gates.append(row)

    mechanism = []
    for candidate, reference, label in MECHANISM_CONTRASTS:
        key = f"{candidate}-{reference}@{PRIMARY_SLICE}"
        in_family = any(m["candidate"] == candidate
                        and m["reference"] == reference
                        and m["slice"] == PRIMARY_SLICE
                        and m["member"] == "primary"
                        for f in families for m in f["members"])
        mechanism.append({
            "contrast_key": key, "candidate": candidate,
            "reference": reference, "registered_label": label,
            "inferential_in_a_registered_family": in_family,
            "guard": MECHANISM_GUARD,
            "record": contrasts.get(key),
        })

    base_sidecar = Path(base_npz).with_suffix(".json")
    base = base_only_readout(Path(base_npz), base_sidecar, frame, blocks, pin)
    residual = {
        "qualifying_primary": "residual_t1 - residual_mlp on the all slice",
        "residual_t1_minus_mlp_cannot_qualify": (
            "residual_t1 - mlp mixes the sequence and the production prior; "
            "it is reported and never a qualifying primary"),
        "residual_mlp_role": ("a production-prior control, not a sequence "
                             "gain"),
        "base_only": {k: v for k, v in base.items() if k != "_row_ll"},
        "against_base_only": [
            residual_vs_base(arm, runs, base, frame, blocks, reps=reps,
                             seed=seed, seeds=seeds)
            for arm in ("residual_mlp", "residual_t1")],
        "registered_primary_key": "residual_t1-residual_mlp@all",
        "reported_keys": ["residual_t1-mlp@all", "residual_mlp-mlp@all"],
    }

    not_evaluable = sorted(
        key for key, record in contrasts.items()
        if not record.get("available"))

    return {
        "tool": rel(__file__),
        "check": "D10.1-D10.7, D10.11",
        "generated_at_utc": _now(),
        "config": {"path": rel(config_path),
                   "sha256": sha256_file(config_path),
                   "name": (config.get("experiment") or {}).get("name")},
        "evidence_status": (config.get("experiment") or {}).get(
            "evidence_status"),
        "contract": {
            "sign_convention": SIGN_CONVENTION,
            "reps": reps, "rng_seed": seed, "alpha": ALPHA,
            "margin_ll": MARGIN_LL,
            "p_convention": P_CONVENTION,
            "holm_formula": HOLM_FORMULA,
            "holm_scope": ("within each three-member family, per estimand; "
                           "never pooled across the 15 families and never "
                           "across the k search"),
            "no_correction_across_families": True,
            "rank_local_note": RANK_LOCAL_NOTE,
            "estimand_ii_label": ESTIMAND_II_LABEL,
            "reconstructed_ll_label": RECONSTRUCTED_LL_LABEL,
            "allowed_statuses": list(ALLOWED_STATUSES),
            "stage1_margin_imported": False,
            "stage1_parity_classification_imported": False,
            "row_log_loss": ("registered_experiment.row_log_loss: "
                             "-log(clip(probs[row, y], 1e-15, 1))"),
            **blocks.contract,
        },
        "pin": {"available": pin.available, "reason": pin.reason,
                "config_body_sha256": pin.config_body_sha256,
                "validation_parquet_md5": pin.validation_md5,
                "validation_parquet_rows": pin.validation_rows,
                "stats_cache_md5": pin.stats_cache_md5,
                "feature_hash": pin.feature_hash,
                "verified": ("the frame md5 and row count, and every "
                             "base-logit hash, are compared against this pin "
                             "before any number is computed")},
        "admission": comparability,
        "frame": {"dir": rel(frame_dir), "version": FRAME_VERSION,
                  "split": SPLIT, "parquet": rel(frame.path),
                  "parquet_sha256": frame.sha256,
                  "parquet_md5": frame.md5,
                  "n_rows": frame.n_rows,
                  "class_mapping": {str(k): v
                                    for k, v in CLASS_MAPPING.items()}},
        "alignment": {
            "reference": rel(frame.path),
            "rule": ("identical y and innings_id ordering across every "
                     "compared arm and seed, asserted before differencing, "
                     "and correspondence to the pinned parquet's ROW ORDER "
                     "rather than a join or a sort by innings id"),
            "checked": [f"{cid} seed {s}" for cid in config_ids
                        for s in seeds if runs[cid][s].admitted],
        },
        "blocks": {"lookup_source": blocks.lookup_source,
                   "lookup_sha256": blocks.lookup_sha256,
                   "n_matches": blocks.n_matches,
                   "n_blocks": blocks.n_blocks,
                   "unmapped": blocks.unmapped},
        "slices": {"predicates": frame.predicates,
                   "stats": {name: slice_stats(frame, blocks, name)
                             for name in slice_names},
                   "frozen_before_computation": True,
                   "gate_slices": list(GATE_SLICES),
                   "exploratory_note": ("no exploratory result changes k, "
                                        "families or advancement; this "
                                        "validation-only run opens no test "
                                        "rows")},
        "runs": runs_report,
        "families": family_out,
        "gates": gates,
        "mechanism_contrasts": mechanism,
        "residual": residual,
        "contrasts": contrasts,
        "not_evaluable": not_evaluable,
        "cohort": {"cohort_status": "DEFERRED_UNOPENED",
                   "cohort_scored": False,
                   "advances": [],
                   "provisional": True,
                   "reads_performed": "none"},
    }


def _summary_lines(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        f"stage2_stats  config {payload['config']['path']}",
        f"  frame        {payload['frame']['parquet']} "
        f"({payload['frame']['n_rows']} rows)",
        f"  blocks       {payload['blocks']['n_blocks']} tournament blocks / "
        f"{payload['blocks']['n_matches']} matches "
        f"({payload['contract']['bootstrap_contract_version']}, "
        f"gap {payload['contract']['max_event_gap_days']}d)",
        f"  bootstrap    {payload['contract']['reps']} reps, rng seed "
        f"{payload['contract']['rng_seed']}, ball-weighted, whole blocks",
    ]
    admitted = sum(1 for block in payload["runs"].values()
                   for row in block["seeds"] if row["admitted"])
    expected = sum(len(block["seeds"]) for block in payload["runs"].values())
    lines.append(f"  runs         {admitted}/{expected} seed runs admitted")
    missing = [cid for cid, block in payload["runs"].items()
               if not block["complete_paired_seeds"]]
    if missing:
        lines.append("  incomplete   " + ", ".join(sorted(missing)))
    lines.append(f"  families     {len(payload['families'])} "
                 f"(3 members each)")
    for family in payload["families"]:
        statuses = {readout: family["screen"][readout]["status"]
                    for readout in READOUTS}
        lines.append(
            f"    {family['candidate']:<18} "
            + "  ".join(f"{readout}={statuses[readout]}"
                        for readout in READOUTS))
    lines.append(f"  not evaluable {len(payload['not_evaluable'])} contrasts")
    lines.append("  cohort       DEFERRED_UNOPENED (not scored, not read)")
    lines.append("  statuses     " + ", ".join(ALLOWED_STATUSES)
                 + " — no advancement status exists")
    return lines


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path = guard_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False,
                               default=str) + "\n")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    stats = sub.add_parser("stats", help="D10.1-D10.7, D10.11")
    stats.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    stats.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    stats.add_argument("--frame-dir", type=Path, default=DEFAULT_FRAME_DIR)
    stats.add_argument("--block-source-dir", type=Path,
                       default=DEFAULT_BLOCK_SOURCE)
    stats.add_argument("--base-logits", type=Path, default=DEFAULT_BASE_LOGITS)
    stats.add_argument("--out", type=Path, default=DEFAULT_STATS_OUT)
    stats.add_argument("--reps", type=int, default=REPS)
    stats.add_argument("--rng-seed", type=int, default=RNG_SEED)
    stats.add_argument("--seeds", default=",".join(str(s) for s
                                                   in REGISTERED_SEEDS))
    stats.add_argument("--expect-rows", type=int, default=EXPECTED_ROWS)
    stats.add_argument("--expect-matches", type=int, default=EXPECTED_MATCHES)
    stats.add_argument("--expect-blocks", type=int, default=EXPECTED_BLOCKS)
    stats.add_argument("--expect-unmapped", type=int,
                       default=EXPECTED_UNMAPPED)
    stats.add_argument("--expected-families", type=int, default=N_FAMILIES,
                       help="the registered family count (15); only a "
                            "synthetic fixture ever changes it")

    sweep = sub.add_parser("ksweep", help="D9 k selection")
    sweep.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sweep.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    sweep.add_argument("--out", type=Path, default=DEFAULT_KSWEEP_OUT)
    sweep.add_argument("--seeds", default=",".join(str(s) for s
                                                   in REGISTERED_SEEDS))

    args = parser.parse_args(argv)
    seeds = tuple(int(s) for s in str(args.seeds).split(",") if s.strip())

    try:
        if args.command == "stats":
            payload = compute_statistics(
                args.config, args.runs_root, args.frame_dir,
                args.block_source_dir, args.base_logits,
                reps=args.reps, seed=args.rng_seed, seeds=seeds,
                expect={"rows": args.expect_rows,
                        "matches": args.expect_matches,
                        "blocks": args.expect_blocks,
                        "unmapped": args.expect_unmapped},
                expected_families=args.expected_families)
            path = write_json(args.out, payload)
            print("\n".join(_summary_lines(payload)))
            print(f"wrote {rel(path)}")
            return 0
        record = k_sweep(args.runs_root, read_yaml(args.config) or {},
                         seeds=seeds)
        path = write_json(args.out, record)
        print(f"k sweep: {record['selection']}"
              + (f" -> {record['selected_config_id']}"
                 if record.get("selected_config_id") else ""))
        for row in record["rows"]:
            print(f"  k={row['k']:<4} seeds {row['per_seed']} mean "
                  f"{row['mean_ll']!r}")
        if record.get("flags"):
            for flag in record["flags"]:
                print(f"  flag: {flag}")
        print(f"wrote {rel(path)}")
        return 0
    except RefusalError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
