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
import re
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

# D10.16 — the registered five-seed extension qualification. At five seeds a
# family qualifies only with a 4/5 favourable-direction count on its primary,
# IN ADDITION to everything the two-seed screen already requires (primary
# CI-clean under Holm, both non-inferiority gates strictly U95 < 0.002, at
# least 10 blocks, complete paired seeds). Below five seeds the count cannot
# reach 4/5, so the requirement does not apply and a two-seed run reports
# exactly what it reported before (Astra gate 2 round 2).
FIVE_SEED_MINIMUM = 5
FIVE_SEED_FAVOURABLE_DIRECTIONS = 4
FIVE_SEED_ELIGIBILITY_RULE = (
    f"at {FIVE_SEED_MINIMUM} or more seeds a family qualifies for the "
    f"extension only with at least {FIVE_SEED_FAVOURABLE_DIRECTIONS} of "
    f"{FIVE_SEED_MINIMUM} favourable per-seed directions on its registered "
    "primary, in addition to the primary being CI-clean under Holm, both "
    f"non-inferiority gates passing with strictly U95 < {MARGIN_LL}, at least "
    f"{MIN_BLOCKS} blocks and complete paired seeds; below "
    f"{FIVE_SEED_MINIMUM} seeds the count cannot reach "
    f"{FIVE_SEED_FAVOURABLE_DIRECTIONS}/{FIVE_SEED_MINIMUM}, so the "
    "requirement does not apply and no two-seed result is retroactively "
    "failed by it")
FAMILY_SIZE = 3
FAMILY_MEMBER_ORDER = ("primary", "death_gate", "chase_gate")
# A general family registers 2-6 members. The legacy fixed
# {primary, death_gate, chase_gate} form is translated into the general form,
# so FAMILY_SIZE remains the legacy size and is no longer the only size.
FAMILY_MIN_MEMBERS = 2
FAMILY_MAX_MEMBERS = 6
MEMBER_KINDS = ("superiority", "non_inferiority")
# The shared control is a config key (`statistics.families.shared_control`).
# It defaults to the Stage 2 literal so a config carrying none of the night-3
# keys behaves exactly as before.
DEFAULT_SHARED_CONTROL = "mlp"
CONTROL_ROLE = "control"
CANDIDATE_ROLE = "candidate"
# `all_row_condition` values. "inherited" is the Stage 2 behaviour: the extra
# `candidate - shared_control @ all` clean-interval condition, read outside the
# family. "member" discharges it through a registered family member instead;
# "none" registers that the family carries no such extra condition.
ALL_ROW_CONDITIONS = ("inherited", "member", "none")

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
SEED_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
              6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def seed_word(n_seeds: int) -> str:
    return SEED_WORDS.get(int(n_seeds), str(int(n_seeds)))


ESTIMAND_II_LABEL = ("descriptive two-seed robustness screen (joint seed and "
                     "tournament-block resampling); not five-seed evidence "
                     "and not uncertainty for a newly trained single "
                     "checkpoint")


def estimand_ii_label(n_seeds: int = len(REGISTERED_SEEDS)) -> str:
    """The estimand (ii) label, with the seed count it was computed at.

    At two seeds this is the registered constant verbatim, so the two-seed
    numbers of record keep their exact label; at any other count it names that
    count, so no five-seed table can carry a two-seed label.
    """
    if int(n_seeds) == 2:
        return ESTIMAND_II_LABEL
    return (f"descriptive {seed_word(n_seeds)}-seed robustness screen (joint "
            "seed and tournament-block resampling); not uncertainty for a "
            "newly trained single checkpoint")
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
    """Refuse the cohort, the smoke tree and the two sealed holdouts.

    The fragments are tested against BOTH the path as written and its resolved
    form, so a symlink, a `..` hop or a relative spelling cannot carry a read
    into the cohort, the smoke tree or a sealed holdout. The path is returned as
    written: resolving it here would turn every reported path absolute.
    """
    given = Path(path)
    candidates = [given.as_posix()]
    try:
        candidates.append(given.resolve().as_posix())
    except OSError:  # pragma: no cover - an unresolvable path is still checked
        pass
    for fragment in FORBIDDEN_FRAGMENTS:
        for text in candidates:
            if fragment in text:
                raise RefusalError(
                    f"refusing to open {candidates[0]}: this stage may not "
                    f"read {fragment!r} (cohort DEFERRED_UNOPENED; no smoke "
                    "log loss; no sealed holdout)"
                    + ("" if text == candidates[0] else
                       f"; it resolves to {text}"))
    return given


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


# ---------------------------------------------------------------------------
# The training driver, imported for its own signature construction
# ---------------------------------------------------------------------------
#
# Astra gate 1 round 3 MUST-FIX 2: a recorded `training_signature` string is
# not evidence on its own, because nothing recomputed it. The signature is
# built by `scripts/sequence_track/retrain_stage2.py`, and this module imports
# that construction rather than re-implementing it, so the two can never drift
# apart. The import is lazy: `retrain_stage2` imports `transformer_t1`, which
# imports torch, and neither the k sweep's caller nor a plain `--help` should
# pay for that. A failed import is a REFUSAL, never a fall back to a local
# copy of the rule.

_DRIVER: Any = None

# The components of the driver's seed-independent signature, in its registered
# order. Held here so a drift in the driver is named rather than absorbed.
EXPECTED_SIGNATURE_COMPONENTS = ("config_id", "arm", "arm_params",
                                 "training_block", "frame", "stats_cache",
                                 "base_logits", "implementation")


def training_driver() -> Any:
    """`scripts/sequence_track/retrain_stage2` — imported, never mirrored."""
    global _DRIVER
    if _DRIVER is None:
        try:
            from sequence_track import retrain_stage2 as module
        except Exception as error:  # noqa: BLE001 - import failure is a refusal
            raise RefusalError(
                "scripts/sequence_track/retrain_stage2.py could not be "
                "imported, so a recorded training signature cannot be "
                "recomputed from its components and no run may be admitted: "
                f"{error}") from error
        order = tuple(getattr(module, "SIGNATURE_COMPONENTS", ()))
        if order != EXPECTED_SIGNATURE_COMPONENTS:
            raise RefusalError(
                f"the driver registers signature components {list(order)}, "
                f"this tool expects {list(EXPECTED_SIGNATURE_COMPONENTS)}; "
                "the signature construction moved and the analysis must be "
                "updated deliberately, not silently")
        _DRIVER = module
    return _DRIVER


def recompute_training_signature(components: Mapping[str, str]) -> str:
    """The driver's signature, recomputed from the recorded components."""
    return training_driver().training_signature(dict(components))


def component_digest(value: Any) -> str:
    """The driver's own component digest, for anchoring to the pin."""
    return training_driver()._digest(value)  # noqa: SLF001 - the registered rule


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
    stats_cache_role: str | None = None
    base_logits: dict = field(default_factory=dict)
    config_body_sha256: str | None = None
    reason: str | None = None
    # Raw pinned blocks, kept so a run's signature components can be
    # recomputed FROM the pin (Astra round 3, cross-arm requirement 4).
    frame_block: dict = field(default_factory=dict)
    stats_cache_block: dict = field(default_factory=dict)
    base_logits_block: dict = field(default_factory=dict)
    base_logits_digest: str | None = None
    source_sha256: dict = field(default_factory=dict)


# What a `provenance` block must actually carry before it counts as a pin.
# Astra round 3 MUST-FIX 2, last clause: a non-empty block WITHOUT a validation
# hash used to qualify, so a pin could be "available" while anchoring nothing.
PIN_REQUIRED = ("frame.dir", "frame.version", f"frame.splits.{SPLIT}.md5",
                f"frame.splits.{SPLIT}.n_rows", "frame.feature_hash",
                "stats_cache.md5", "config_body_sha256")


def load_pin(config: Mapping[str, Any]) -> Pin:
    provenance = config.get("provenance")
    if not isinstance(provenance, Mapping) or not provenance:
        return Pin(available=False,
                   reason="the config carries no generated `provenance` "
                          "block, so there is no pin to verify against; run "
                          "`pin_stage2.py --write`")
    frame = provenance.get("frame") or {}
    validation = ((frame.get("splits") or {}).get(SPLIT) or {})
    cache = provenance.get("stats_cache") or {}
    base = provenance.get("base_logits") or {}
    present = {
        "frame.dir": frame.get("dir"),
        "frame.version": frame.get("version"),
        f"frame.splits.{SPLIT}.md5": validation.get("md5"),
        f"frame.splits.{SPLIT}.n_rows": validation.get("n_rows"),
        "frame.feature_hash": frame.get("feature_hash") or None,
        "stats_cache.md5": cache.get("md5"),
        "config_body_sha256": provenance.get("config_body_sha256"),
    }
    missing = [name for name in PIN_REQUIRED if not present.get(name)]
    if missing:
        return Pin(available=False,
                   reason=("the config's `provenance` block is not a usable "
                           f"pin: it carries no {missing}. A non-empty "
                           "provenance block without a validation hash "
                           "anchors nothing (Astra gate 1 round 3 MUST-FIX "
                           "2); run `pin_stage2.py --write`"))
    return Pin(
        available=True,
        frame_dir=frame.get("dir"),
        frame_version=frame.get("version"),
        validation_md5=validation.get("md5"),
        validation_rows=int(validation["n_rows"]),
        feature_hash=dict(frame.get("feature_hash") or {}),
        stats_cache_md5=cache.get("md5"),
        stats_cache_role=cache.get("role"),
        base_logits=dict(base.get("splits") or {}),
        config_body_sha256=provenance.get("config_body_sha256"),
        frame_block=dict(frame),
        stats_cache_block=dict(cache),
        base_logits_block=dict(base),
        base_logits_digest=base.get("train_validation_digest"),
        source_sha256=dict((provenance.get("sources") or {})
                           .get("source_sha256") or {}))


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

# D-night3 — optional registered slices (tier, match-list, and composed).
#
# Every optional slice is OPT-IN through `statistics.slice_predicates`, never
# added by default: the frozen `SLICE_PREDICATES` tuple above is what
# `statistics.slices` is checked against, and a Stage 2 config that registers
# eight slice names must keep computing exactly those eight.
#
# `statistics.slice_predicates` is a mapping of slice name -> spec, read in
# config order. `thin_pair` keeps its own inherited spec shape; every other
# entry is one of
#
#   <name>: {tier: 3}                     competition_tier == 3
#                                         (`target_P` and `tier3` are the same
#                                          slice under two registered names)
#   <name>: {match_list: <path>,          the frame rows whose innings_id
#            sha256: <pinned digest>}     SUFFIX (the match id) is in the
#                                         registered JSON list. The sha256 is
#                                         REQUIRED: an unpinned match list is
#                                         refused at config load.
#   <name>: {base: death,                 an existing slice AND an optional one
#            restrict: target_P}
#   death@target_P: {}                    the same composition by naming
#                                         convention
#
# A composed name written as `<base>@<restrict>` needs no body. If the frame
# cannot serve a slice — no `competition_tier` column, or a match-list file
# that is absent or whose sha256 does not match the pin — the slice is reported
# UNAVAILABLE with its reason, exactly as `thin_pair` is, rather than invented
# or silently dropped.
TIER_COLUMN = "competition_tier"
REGISTERED_TIERS = (1, 2, 3, 4)
TIER_COLUMN_MISSING_REASON = (
    f"the pinned frame carries no {TIER_COLUMN!r} column, so this tier slice "
    "is reported unavailable rather than invented. The i7 ball frame does "
    "carry it; a frame that does not cannot serve a tier slice.")


def optional_slice_specs(config: Mapping[str, Any]) -> list[dict]:
    """The registered optional slices, in config order.

    Returns ``[{"name", "kind", ...}]`` with ``kind`` one of ``tier``,
    ``match_list`` or ``composed``. ``thin_pair`` is excluded: it keeps its own
    inherited reader. Absent key -> no optional slice, and the computed slice
    list is exactly the frozen one.
    """
    raw = ((config.get("statistics") or {}).get("slice_predicates") or {})
    if not isinstance(raw, Mapping):
        raise RefusalError(
            "statistics.slice_predicates must be a mapping of slice name to "
            "its registered spec")
    base_names = {p.name for p in SLICE_PREDICATES}
    out: list[dict] = []
    for name, spec in raw.items():
        name = str(name)
        if name == THIN_PAIR_NAME:
            continue
        spec = spec if isinstance(spec, Mapping) else {}
        base, _, restrict = name.rpartition("@")
        if spec.get("base") or spec.get("restrict"):
            base = str(spec.get("base") or "")
            restrict = str(spec.get("restrict") or "")
        if base:
            if base not in base_names:
                raise RefusalError(
                    f"slice {name!r} restricts {base!r}, which is not a "
                    f"registered slice predicate {sorted(base_names)}")
            if not restrict:
                raise RefusalError(
                    f"slice {name!r} names no slice to restrict to")
            out.append({"name": name, "kind": "composed", "base": base,
                        "restrict": restrict})
            continue
        if "tier" in spec:
            tier = int(spec["tier"])
            if tier not in REGISTERED_TIERS:
                raise RefusalError(
                    f"slice {name!r} names tier {tier}, which is not one of "
                    f"{list(REGISTERED_TIERS)}")
            out.append({"name": name, "kind": "tier", "tier": tier})
            continue
        if spec.get("match_list"):
            if not spec.get("sha256"):
                raise RefusalError(
                    f"slice {name!r} registers the match list "
                    f"{spec['match_list']!r} with no `sha256`; an unpinned "
                    "match list is not a registered slice, because the rows it "
                    "selects could change under the analysis")
            out.append({"name": name, "kind": "match_list",
                        "match_list": str(spec["match_list"]),
                        "sha256": str(spec["sha256"])})
            continue
        raise RefusalError(
            f"slice {name!r} registers none of `tier`, `match_list` or "
            "`base`/`restrict`; an optional slice is never invented")
    names = [row["name"] for row in out]
    duplicated = sorted({n for n in names if names.count(n) > 1})
    if duplicated:
        raise RefusalError(f"slices {duplicated} are registered twice")
    by_name = {row["name"]: row for row in out}
    for row in out:
        if row["kind"] != "composed":
            continue
        target = by_name.get(row["restrict"])
        if target is None:
            raise RefusalError(
                f"slice {row['name']!r} restricts to {row['restrict']!r}, "
                "which is not a registered optional slice")
        if target["kind"] == "composed":
            raise RefusalError(
                f"slice {row['name']!r} restricts to {row['restrict']!r}, "
                "which is itself composed; compose once, not twice")
        if names.index(row["restrict"]) > names.index(row["name"]):
            raise RefusalError(
                f"slice {row['name']!r} is registered before the "
                f"{row['restrict']!r} it restricts to")
    return out


def _match_list_mask(spec: Mapping[str, Any], match_id: np.ndarray,
                     ) -> tuple[Any, str | None]:
    """The mask for a match-list slice, or (None, reason) when unavailable."""
    path = guard_path(Path(spec["match_list"]))
    if not path.exists():
        return None, (f"the registered match list {rel(path)} does not exist, "
                      "so this slice is reported unavailable rather than "
                      "invented")
    digest = sha256_file(path)
    if digest != spec["sha256"]:
        return None, (f"the registered match list {rel(path)} has sha256 "
                      f"{digest}, the config pins {spec['sha256']}: the list "
                      "drifted and no number may be computed from it")
    payload = json.loads(path.read_text())
    if isinstance(payload, Mapping):
        payload = (payload.get("match_ids") or payload.get("matches") or [])
    wanted = {str(value) for value in payload}
    if not wanted:
        return None, (f"the registered match list {rel(path)} names no "
                      "matches")
    return np.isin(match_id, sorted(wanted)), None


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
    optional = optional_slice_specs(config)
    present = set(pq.ParquetFile(path).schema_arrow.names)
    wanted = ["innings_id", "ball_outcome"]
    for predicate in SLICE_PREDICATES:
        wanted.extend(predicate.columns)
    tier_column_present = TIER_COLUMN in present
    if tier_column_present and any(r["kind"] == "tier" for r in optional):
        wanted.append(TIER_COLUMN)
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

    # Optional registered slices, appended after thin_pair in config order so
    # an existing slice list keeps its order and its contents.
    row_match_id = match_ids(innings)
    for request in optional:
        name, kind = request["name"], request["kind"]
        role = "exploratory"
        columns: list[str] = []
        expression: str | None = None
        mask = None
        reason: str | None = None
        if kind == "tier":
            columns = [TIER_COLUMN]
            expression = f"{TIER_COLUMN} == {request['tier']}"
            if not tier_column_present:
                reason = TIER_COLUMN_MISSING_REASON
            elif df[TIER_COLUMN].isna().any():
                raise RefusalError(
                    f"slice {name!r} reads {TIER_COLUMN!r}, which has null "
                    "values in the pinned frame; missing-value handling must "
                    "be registered before the slice is computed")
            else:
                mask = df[TIER_COLUMN].to_numpy(np.int64) == request["tier"]
        elif kind == "match_list":
            columns = ["innings_id"]
            expression = (f"the innings_id match-id suffix is in "
                          f"{request['match_list']}")
            mask, reason = _match_list_mask(request, row_match_id)
        else:
            base_predicate = next(p for p in SLICE_PREDICATES
                                  if p.name == request["base"])
            role = "gate" if request["base"] in GATE_SLICES else "exploratory"
            columns = list(base_predicate.columns)
            target = next(r for r in optional
                          if r["name"] == request["restrict"])
            restrict_row = next(row for row in predicates
                                if row["slice"] == request["restrict"])
            expression = (f"{base_predicate.expression} and "
                          f"{restrict_row['predicate']}")
            if request["restrict"] not in masks:
                reason = (f"the {request['restrict']!r} slice this one "
                          f"restricts to is unavailable: "
                          f"{restrict_row.get('unavailable_reason')}")
                expression = None
            else:
                mask = masks[request["base"]] & masks[request["restrict"]]
                columns = list(dict.fromkeys(
                    columns + list(restrict_row["columns"])))
            del target
        if mask is None:
            predicates.append({
                "slice": name, "predicate": None, "columns": columns,
                "role": role, "available": False,
                "unavailable_reason": reason, "n_rows": 0})
            continue
        mask = np.asarray(mask, dtype=bool)
        masks[name] = mask
        row = {"slice": name, "predicate": expression, "columns": columns,
               "role": role, "slice_kind": kind, "available": True,
               "n_rows": int(mask.sum())}
        if kind == "tier":
            row["tier"] = request["tier"]
        elif kind == "match_list":
            row["match_list"] = request["match_list"]
            row["match_list_sha256"] = sha256_file(
                guard_path(Path(request["match_list"])))
            row["n_matches"] = int(len(set(row_match_id[mask])))
        else:
            row["restricts_slice"] = request["base"]
            row["restricted_to"] = request["restrict"]
        predicates.append(row)

    if (pin.validation_rows is not None
            and int(len(df)) != pin.validation_rows):
        raise RefusalError(
            f"{rel(path)} holds {len(df)} rows, the pin records "
            f"{pin.validation_rows}")
    return Frame(path=path, sha256=sha256_file(path), md5=frame_md5,
                 n_rows=int(len(df)), innings_id=innings, y=y,
                 match_id=row_match_id, masks=masks,
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
# `implementation` and `base_logits` are neither freely arm-specific nor
# blindly shared, so each has its own rule (Astra round 3, cross-arm
# requirements 1 and 2): the implementation sources common to every arm must
# agree, with only the recurrent-only entry allowed to differ, and the two
# residual arms must carry ONE base-logits identity.
SIGNATURE_GROUPED_COMPONENTS = ("implementation", "base_logits")
# The arm_params fields a configuration's registered entry fixes directly.
REGISTERED_ARM_PARAM_FIELDS = ("arm", "k", "wiring", "history_input",
                               "residual_l2")
# A summary log loss and its metrics.json authenticator are the same float
# written twice; yaml round-trips a float exactly, so anything above this is an
# edit, not a rounding artefact (Astra round 3 MUST-FIX 3).
SUMMARY_LL_TOLERANCE = 1e-12

_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")
_HEX32 = re.compile(r"\A[0-9a-f]{32}\Z")


@dataclass
class Admission:
    """Whether one run directory may contribute a number, and why not."""

    ok: bool
    reason: str | None = None
    training_signature: str | None = None
    components: dict = field(default_factory=dict)
    checkpoint_md5: str | None = None
    # Read once, here, so the comparability and summary-authentication checks
    # never re-open a run's records with a different set of assumptions.
    metrics_validation_ll: float | None = None
    metrics_arm_params: dict = field(default_factory=dict)
    arm_params_expected: dict = field(default_factory=dict)
    # The implementation files the run itself hashed (`training_signature_sources`
    # in run_record.json). A run is anchored to the pin's hashes of ITS OWN
    # source list, so a later stage that widens the driver's source set does
    # not retroactively refuse a sealed earlier stage's runs.
    signature_sources: list = field(default_factory=list)
    artefact_manifest: dict = field(default_factory=dict)


def _int_or_none(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def verify_run_admission(directory: Path, config_id: str, seed: int,
                         expected_arm: str | None = None,
                         expected_k: Any = "__unset__") -> Admission:
    """Reject a run whose own records disagree with what the tree claims.

    Checks, in order: the four artefacts plus `COMPLETE.json` exist and parse;
    `metrics.json`, `run_record.json` and `COMPLETE.json` all name this
    configuration id and this seed; the arm and `k` match registration; a
    `training_signature` is present, identical in `run_record.json` and
    `COMPLETE.json`, and equal to the signature RECOMPUTED from its own
    recorded components by the driver's rule; every component is present and
    of the right type; and the completion record manifests every expected
    artefact with a size and an md5 that still match the file on disk.

    Astra gate 1 round 3 MUST-FIX 2 — this verifier fails CLOSED. An absent
    manifest, an empty manifest, a manifest missing one expected artefact, a
    missing size or md5 declaration, an absent component block and a signature
    string that its own components do not reproduce are each a refusal, not a
    pass. The driver writes the full manifest and the full component block for
    every real run (`retrain_stage2.write_completion_record`), so there is no
    legacy shape to accommodate.
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

    validation_ll = _float_or_none((metrics or {}).get(f"{SPLIT}_ll"))
    if validation_ll is None:
        return Admission(
            False, f"metrics.json carries no finite {SPLIT}_ll, so no summary "
                   "log loss attributed to this run can be authenticated "
                   "(D3.11)")

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
    raw_components = record.get("training_signature_components")
    raw_completion = completion.get("training_signature_components")
    for name, payload in (("run_record.json", raw_components),
                          (COMPLETION_RECORD, raw_completion)):
        if not isinstance(payload, Mapping) or not payload:
            return Admission(
                False, f"{name} carries no training_signature_components "
                       "block, so the recorded signature cannot be "
                       "recomputed and this run is not admissible")
    components = dict(raw_components)
    completion_components = dict(raw_completion)
    if completion_components != components:
        return Admission(
            False, "run_record.json and "
                   f"{COMPLETION_RECORD} disagree about the training "
                   "signature components")
    missing_components = [name for name in EXPECTED_SIGNATURE_COMPONENTS
                          if name not in components]
    if missing_components:
        return Admission(
            False, "training_signature_components is missing "
                   f"{missing_components}; every registered component must be "
                   "present, never skipped")
    unknown_components = sorted(set(components)
                                - set(EXPECTED_SIGNATURE_COMPONENTS))
    if unknown_components:
        return Admission(
            False, "training_signature_components carries unregistered "
                   f"entries {unknown_components}")
    mistyped = sorted(name for name, value in components.items()
                      if not (isinstance(value, str) and _HEX64.match(value)))
    if mistyped:
        return Admission(
            False, f"training_signature_components entries {mistyped} are not "
                   "sha256 hex digests")
    try:
        recomputed = recompute_training_signature(components)
    except RefusalError:
        raise
    if recomputed != str(signature):
        return Admission(
            False, f"the recorded training_signature {str(signature)[:16]}… is "
                   "not the signature its own components produce "
                   f"({recomputed[:16]}…); the string was edited or the "
                   "components were, and neither run may be differenced")

    manifest = completion.get("artifacts")
    if not isinstance(manifest, Mapping) or not manifest:
        return Admission(
            False, f"{COMPLETION_RECORD} carries no artefact manifest "
                   f"({manifest!r}); a run with no declared sizes and hashes "
                   "is not admissible")
    for name in RUN_ARTEFACTS:
        facts = manifest.get(name)
        if not isinstance(facts, Mapping):
            return Admission(
                False, f"{COMPLETION_RECORD} manifests no entry for {name}, "
                       "which every completed run must declare")
        declared_bytes = _int_or_none(facts.get("bytes"))
        declared_md5 = facts.get("md5")
        if declared_bytes is None:
            return Admission(
                False, f"{COMPLETION_RECORD} artifacts.{name} declares no "
                       f"integer size ({facts.get('bytes')!r})")
        if not (isinstance(declared_md5, str) and _HEX32.match(declared_md5)):
            return Admission(
                False, f"{COMPLETION_RECORD} artifacts.{name} declares no md5 "
                       f"({declared_md5!r})")
    for name, facts in sorted(manifest.items()):
        if not isinstance(facts, Mapping):
            return Admission(
                False, f"{COMPLETION_RECORD} artifacts.{name} {facts!r} is "
                       "not a size-and-md5 declaration")
        path = directory / name
        if not path.is_file():
            return Admission(False, f"{COMPLETION_RECORD} manifests "
                                    f"{name}, which is absent")
        declared_bytes = _int_or_none(facts.get("bytes"))
        if declared_bytes is None or path.stat().st_size != declared_bytes:
            return Admission(
                False, f"{name} is {path.stat().st_size} bytes, the "
                       f"completion manifest recorded {facts.get('bytes')!r}")
        declared_md5 = facts.get("md5")
        if not isinstance(declared_md5, str) or md5_file(path) != declared_md5:
            return Admission(
                False, f"{name} md5 has changed since the run completed "
                       f"(manifest {declared_md5!r})")
    checkpoint = manifest.get("model.pt") or {}
    return Admission(
        True, None, str(signature), components,
        checkpoint.get("md5") or record.get("checkpoint_md5"),
        metrics_validation_ll=validation_ll,
        metrics_arm_params=dict(arm_params),
        arm_params_expected=dict(record.get("arm_params_expected") or {}),
        signature_sources=[str(x) for x in
                           (record.get("training_signature_sources") or [])],
        artefact_manifest={name: dict(facts)
                           for name, facts in manifest.items()})


def _is_recurrent(entry: Mapping[str, Any] | None) -> bool:
    return str((entry or {}).get("wiring")) == "recurrent"


def _reads_base_logits(entry: Mapping[str, Any] | None) -> bool:
    """Whether a registered configuration is one of the residual arms."""
    access = (entry or {}).get("access") or {}
    return bool(access.get("prod_logits")) or bool(
        ((entry or {}).get("params") or {}).get("base_logits_dir"))


def driver_effective_settings(config: Mapping[str, Any],
                              entry: Mapping[str, Any], pin: Pin) -> dict:
    """The driver's own `effective_settings` for one registered entry.

    The entry is checked through the driver's `_check_params` when it has not
    already been (a config read straight from yaml has no `_params`), so the
    stage 3 params reach `effective_settings` exactly as a launch would
    present them — including the sha256 of a frozen train-match list.
    """
    driver = training_driver()
    entry = dict(entry)
    if not entry.get("_params"):
        entry["_params"] = driver._check_params(  # noqa: SLF001 - registered
            Path(str(config.get("_config_path") or "config.yaml")), entry)
    digest = (pin.base_logits_digest
              if str(entry.get("arm")) in driver.t1.ARMS_NEEDING_BASE_LOGITS
              else None)
    return driver.effective_settings(dict(config), entry,
                                     base_logits_digest=digest)


def pinned_component_digests(config: Mapping[str, Any], pin: Pin) -> dict:
    """The signature components every admitted run must carry, FROM the pin.

    Astra round 3, cross-arm requirement 4: the shared frame, cache and
    training identity must be anchored to the pin, not merely to agreement
    among the runs — two runs can agree with each other and with nothing
    registered. Requirement 1 is the `implementation` entry: the common
    sources must agree across arms, and only the recurrent-only source may
    differ, so one expected digest is built per arm group from the pinned
    source hashes.

    Every raw block below is assembled in the driver's own shape and digested
    with the driver's own `_digest`, and a dedicated test locks this
    translation against `retrain_stage2.signature_components` directly.
    """
    driver = training_driver()
    if not pin.available:
        raise RefusalError(
            f"no run may be admitted: {pin.reason}. Every shared signature "
            "component is anchored to the pin, so without a pin there is "
            "nothing to anchor to")
    training = config.get("training") or {}
    missing = [name for name in ("dmodel", "layers", "heads", "batch",
                                 "epochs", "learning_rate", "patience", "aux")
               if training.get(name) is None]
    if missing:
        raise RefusalError(
            f"the config's `training` block carries no {missing}, so the "
            "registered training identity cannot be recomputed")
    # Night 3 Block B: the expected `training_block` and `arm_params` are
    # DERIVED from the driver's own `effective_settings`, never rebuilt in a
    # fixed shape here. A stage 3 configuration's step budget lives in a
    # `schedule` sub-block the driver adds when the config sets it, and a
    # reconstruction that did not know about it made every night-3 run
    # incomparable against an expectation nothing could produce. Reading the
    # driver means every future default-off param is picked up for free, and a
    # stage 2 configuration — which sets none — digests exactly as before.
    training_block_by_config: dict[str, str] = {}
    arm_params_by_config: dict[str, str] = {}
    for config_id, entry in driver.configurations(config).items():
        effective = driver_effective_settings(config, entry, pin)
        block = {"arch": effective["arch"], "optimiser": effective["optimiser"]}
        if effective.get("schedule"):
            block["schedule"] = effective["schedule"]
        training_block_by_config[config_id] = component_digest(block)
        arm_params_by_config[config_id] = component_digest(
            effective["arm_params"])
    distinct_blocks = sorted(set(training_block_by_config.values()))
    if len(distinct_blocks) == 1:
        training_block_digest = distinct_blocks[0]
    elif not distinct_blocks:
        raise RefusalError(
            "the config registers no configuration, so the registered "
            "training identity cannot be recomputed")
    else:
        # Configurations that disagree about the training block are caught by
        # the per-configuration comparison below; there is no single shared
        # digest to publish in that case.
        training_block_digest = None
    pinned_splits = (pin.frame_block.get("splits") or {})
    split_files = {}
    for split in driver.CONTRACT_SPLITS:
        facts = pinned_splits.get(split)
        if not isinstance(facts, Mapping):
            raise RefusalError(
                f"the pin records no `{split}` split, so the registered frame "
                "identity cannot be recomputed")
        row = {}
        for field_name in driver.SPLIT_IDENTITY_FIELDS:
            if facts.get(field_name) is None:
                raise RefusalError(
                    f"the pin's {split} split carries no {field_name!r}")
            row[field_name] = (driver.as_n_rows(facts[field_name])
                               if field_name == "n_rows"
                               else facts[field_name])
        split_files[split] = row
    frame_raw = {
        "dir_configured": Path(str(pin.frame_dir)).as_posix(),
        "version": pin.frame_version,
        "feature_hash": pin.feature_hash,
        "split_files": {split: split_files[split]
                        for split in sorted(split_files)},
    }
    if not pin.stats_cache_role:
        raise RefusalError(
            "the pin's `stats_cache` block carries no manifest role, so the "
            "registered cache identity cannot be recomputed")
    cache_raw = {"role": str(pin.stats_cache_role), "md5": pin.stats_cache_md5}
    sources = dict(pin.source_sha256)
    common = (driver.TRAINER_SOURCE, driver.FEATURE_CONTRACT_SOURCE,
              driver.ARTIFACT_RESOLVER_SOURCE)
    absent = [name for name in common + (driver.RECURRENT_SOURCE,)
              if not sources.get(name)]
    if absent:
        raise RefusalError(
            f"the pin records no source hash for {absent}, so the "
            "implementation identity cannot be anchored to it")
    implementation = {
        False: {name: sources[name] for name in common},
        True: {name: sources[name]
               for name in common + (driver.RECURRENT_SOURCE,)},
    }
    return {
        "training_block": training_block_digest,
        "training_block_by_config": dict(training_block_by_config),
        "arm_params_by_config": dict(arm_params_by_config),
        "frame": component_digest(frame_raw),
        "stats_cache": component_digest(cache_raw),
        "implementation_by_recurrent": {
            recurrent: component_digest(identity)
            for recurrent, identity in implementation.items()},
        "implementation_sources_by_recurrent": {
            str(recurrent): sorted(identity)
            for recurrent, identity in implementation.items()},
        "base_logits_by_reads_base": {
            False: component_digest(None),
            True: component_digest(pin.base_logits_digest),
        },
        "base_logits_pinned_digest": pin.base_logits_digest,
    }


def expected_arm_identity(config_id: str, entry: Mapping[str, Any],
                          pin: Pin) -> dict:
    """One configuration's registered arm-specific identity (requirement 3)."""
    params = dict(entry.get("params") or {})
    reads_base = _reads_base_logits(entry)
    arm = str(entry.get("arm"))
    # `key_construction` is what makes a relay-free arm relay-free, and the
    # trainer's own registered table is its authority — the config carries the
    # wiring and history input, which the driver already validates against the
    # same tables. It is legitimately `None` for an arm with no attention.
    table = training_driver().t1.ARM_KEY_CONSTRUCTION
    return {
        "config_id": component_digest(str(config_id)),
        "arm": component_digest(arm),
        "key_construction": table.get(arm, "__unregistered__"),
        "arm_params_fields": {
            "arm": arm,
            "k": params.get("k"),
            "wiring": str(entry.get("wiring")),
            "history_input": str(entry.get("history_input")),
            "residual_l2": params.get("residual_l2"),
            "base_logits_md5": (pin.base_logits_digest if reads_base
                                else None),
        },
        "reads_base_logits": reads_base,
        "recurrent": _is_recurrent(entry),
    }


def _arm_param_mismatches(recorded: Mapping[str, Any],
                          expected: Mapping[str, Any],
                          where: str) -> list[str]:
    problems = []
    for field_name in REGISTERED_ARM_PARAM_FIELDS + ("base_logits_md5",):
        want = expected.get(field_name)
        got = recorded.get(field_name, "__absent__")
        if isinstance(want, float) and isinstance(got, (int, float)) and not (
                isinstance(got, bool)):
            same = float(got) == want
        elif want is None and field_name == "residual_l2":
            # A non-residual arm records `residual_l2: None`; the registered
            # entry simply omits it. Both spellings mean "not in the loss".
            same = got in (None, "__absent__")
        else:
            same = str(got) == str(want)
        if not same:
            problems.append(f"{where}.{field_name} {got!r} != the registered "
                            f"{want!r}")
    return problems


def assert_comparable(admissions: Mapping[str, Mapping[int, Admission]],
                      entries: Mapping[str, Mapping[str, Any]] | None = None,
                      config: Mapping[str, Any] | None = None,
                      pin: Pin | None = None) -> dict:
    """Refuse to difference runs whose data, code or registered identity moved.

    Within one configuration every admitted seed must carry ONE training
    signature (the signature is seed-independent). Every expected component
    must be PRESENT and of the right type — an absent component is never
    skipped (Astra round 3 MUST-FIX 2). The shared components are anchored to
    the pin rather than to agreement among the runs; the implementation
    sources common to every arm must agree with only the recurrent-only entry
    differing; the two residual arms must carry one base-logits identity; and
    each configuration's arm-specific components are verified against its
    registered expected identity (Astra round 3, the four cross-arm
    requirements it added when it ruled on the signature deviation).

    Every violation is collected and reported together (Astra round 3, minor
    item): several inadmissible or incomparable runs are all named, never just
    the first one found.
    """
    problems: list[str] = []
    inadmissible: list[dict] = []
    per_config: dict[str, str] = {}
    shared: dict[str, tuple[str, str]] = {}
    run_source_sets: dict = {}  # per-run hashed source lists (not emitted)
    anchored: dict[str, Any] | None = None
    if entries is not None and config is not None and pin is not None:
        anchored = pinned_component_digests(config, pin)
    base_logits_seen: dict[str, list[str]] = {}

    for config_id, seeds in sorted(admissions.items()):
        for seed, value in sorted(seeds.items()):
            if not value.ok:
                inadmissible.append({"config_id": config_id, "seed": seed,
                                     "reason": value.reason})
        signatures = {seed: value.training_signature
                      for seed, value in seeds.items() if value.ok}
        if not signatures:
            continue
        distinct = sorted(set(signatures.values()))
        if len(distinct) != 1:
            problems.append(
                f"{config_id}: its admitted seeds carry {len(distinct)} "
                f"different training signatures {distinct}; the signature is "
                "seed-independent, so this means the code or the data moved "
                "mid-experiment and these runs cannot be differenced")
        else:
            per_config[config_id] = distinct[0]
        entry = (entries or {}).get(config_id)
        identity = (expected_arm_identity(config_id, entry, pin)
                    if (entry is not None and anchored is not None
                        and pin is not None) else None)
        for seed, value in sorted(seeds.items()):
            if not value.ok:
                continue
            where = f"{config_id} seed {seed}"
            components = value.components or {}
            absent = [name for name in EXPECTED_SIGNATURE_COMPONENTS
                      if not isinstance(components.get(name), str)]
            if absent:
                problems.append(
                    f"{where}: training signature components {absent} are "
                    "absent or not strings; an expected component is never "
                    "skipped")
                continue
            for name in SIGNATURE_SHARED_COMPONENTS:
                digest = components[name]
                previous = shared.get(name)
                if previous is None:
                    shared[name] = (digest, where)
                elif previous[0] != digest:
                    problems.append(
                        f"training signature component {name!r} differs "
                        f"between {previous[1]} and {where}; the two runs did "
                        "not see the same data or the same training block, so "
                        "no contrast between them is admissible")
            if anchored is not None:
                for name in SIGNATURE_SHARED_COMPONENTS:
                    want = anchored[name]
                    if name == "training_block":
                        # Derived per configuration from the driver, because a
                        # stage 3 configuration's step schedule joins it.
                        want = anchored["training_block_by_config"].get(
                            config_id, want)
                    if want is None:
                        problems.append(
                            f"{where}: no registered {name!r} digest could be "
                            "recomputed for this configuration")
                    elif components[name] != want:
                        problems.append(
                            f"{where}: training signature component {name!r} "
                            f"{components[name][:12]}… is not the pinned "
                            f"{anchored[name][:12]}…; the shared identity is "
                            "anchored to the pin, not to agreement among runs")
                if identity is not None:
                    want_impl = anchored["implementation_by_recurrent"][
                        identity["recurrent"]]
                    run_sources = list(value.signature_sources or [])
                    if run_sources:
                        # Anchor to the pin's hashes of the run's OWN source
                        # list (night 3: the driver hashes feature_registry.py
                        # too; the sealed stage 2 runs hashed four files).
                        missing = [n for n in run_sources
                                   if not (pin.source_sha256 or {}).get(n)]
                        if missing:
                            problems.append(
                                f"{where}: the run hashed {missing} but the pin "
                                "records no hash for them, so its implementation "
                                "cannot be anchored")
                            continue
                        want_impl = component_digest(
                            {n: pin.source_sha256[n] for n in run_sources})
                        non_rec = tuple(sorted(
                            n for n in run_sources
                            if n != training_driver().RECURRENT_SOURCE))
                        prev = run_source_sets.get("common")
                        if prev is None:
                            run_source_sets["common"] = (non_rec, where)
                        elif prev[0] != non_rec:
                            problems.append(
                                f"{where}: hashed implementation sources "
                                f"{list(non_rec)} differ from {prev[1]}'s "
                                f"{list(prev[0])}; the common implementation "
                                "sources must agree across arms")
                    if components["implementation"] != want_impl:
                        problems.append(
                            f"{where}: the `implementation` component "
                            f"{components['implementation'][:12]}… is not the "
                            f"identity of the pinned sources "
                            f"{anchored['implementation_sources_by_recurrent'][str(identity['recurrent'])]}"
                            " — the common implementation sources must agree "
                            "across arms, with only the recurrent-only entry "
                            "allowed to differ")
                    want_base = anchored["base_logits_by_reads_base"][
                        identity["reads_base_logits"]]
                    if components["base_logits"] != want_base:
                        problems.append(
                            f"{where}: the `base_logits` component is not the "
                            "pinned base-logits identity for a "
                            f"{'residual' if identity['reads_base_logits'] else 'non-residual'}"
                            " arm")
                    if identity["reads_base_logits"]:
                        base_logits_seen.setdefault(
                            components["base_logits"], []).append(where)
                    for name in ("config_id", "arm"):
                        if components[name] != identity[name]:
                            problems.append(
                                f"{where}: the {name!r} component is not the "
                                "digest of its registered value")
                    expected_params = identity["arm_params_fields"]
                    recorded_expected = value.arm_params_expected
                    if not recorded_expected:
                        problems.append(
                            f"{where}: run_record.json carries no "
                            "`arm_params_expected` block, so the arm-specific "
                            "components cannot be verified against the "
                            "registered identity")
                    else:
                        problems.extend(_arm_param_mismatches(
                            recorded_expected, expected_params,
                            f"{where} arm_params_expected"))
                        if components["arm_params"] != component_digest(
                                dict(recorded_expected)):
                            problems.append(
                                f"{where}: the `arm_params` component is not "
                                "the digest of the recorded "
                                "arm_params_expected block")
                    problems.extend(_arm_param_mismatches(
                        value.metrics_arm_params, expected_params,
                        f"{where} metrics.json arm_params"))
                    want_key = identity["key_construction"]
                    for name, recorded in (
                            ("metrics.json", value.metrics_arm_params),
                            ("run_record.json arm_params_expected",
                             recorded_expected)):
                        if not recorded:
                            continue
                        if "key_construction" not in recorded:
                            problems.append(
                                f"{where}: {name} declares no "
                                "`key_construction`, which is what makes a "
                                "relay-free arm relay-free and is part of the "
                                "registered identity")
                        elif recorded["key_construction"] != want_key:
                            problems.append(
                                f"{where}: {name} key_construction "
                                f"{recorded['key_construction']!r} != the "
                                f"trainer's registered {want_key!r} for this "
                                "arm")

    if len(base_logits_seen) > 1:
        problems.append(
            "the residual arms carry different `base_logits` identities "
            + "; ".join(f"{digest[:12]}… for {sorted(where)}"
                        for digest, where in sorted(base_logits_seen.items()))
            + "; `residual_t1 - residual_mlp` is only a residual contrast if "
            "both arms sat on ONE set of base logits")

    if problems:
        raise RefusalError(
            f"{len(problems)} comparability violation(s); no contrast is "
            "admissible until every one is resolved:\n  - "
            + "\n  - ".join(problems))
    return {
        "rule": ("one seed-independent training signature per configuration; "
                 f"the shared components {list(SIGNATURE_SHARED_COMPONENTS)} "
                 "identical across every admitted configuration AND equal to "
                 "the digests recomputed from the pin; the common "
                 "implementation sources identical across arms with only the "
                 "recurrent-only entry differing; one base-logits identity "
                 "across the two residual arms; every arm-specific component "
                 "equal to its registered expected identity"),
        "fails_closed": ("every expected component must be present and of the "
                         "right type; an absent component is a refusal, not a "
                         "skip"),
        "anchored_to_the_pin": anchored is not None,
        "pinned_components": (None if anchored is None else {
            name: anchored[name] for name in SIGNATURE_SHARED_COMPONENTS}),
        "pinned_implementation_sources": (
            None if anchored is None
            else anchored["implementation_sources_by_recurrent"]),
        "arm_specific_components": list(SIGNATURE_ARM_SPECIFIC_COMPONENTS),
        "grouped_components": list(SIGNATURE_GROUPED_COMPONENTS),
        "arm_specific_note": (
            "the driver's signature includes config_id, arm, arm_params and "
            "implementation, so it is arm-dependent by construction and is "
            "NOT required to match across arms; what must match is the data "
            "and training-block part, checked component by component, plus "
            "each arm's own components against its registered identity"),
        "training_signature_by_config": per_config,
        "signature_recomputed_from_components": True,
        "shared_components": {name: digest
                              for name, (digest, _) in shared.items()},
        "residual_base_logits_identities": {
            digest: sorted(where)
            for digest, where in sorted(base_logits_seen.items())},
        "inadmissible_runs": inadmissible,
        "n_inadmissible_runs": len(inadmissible),
        "inadmissible_note": ("every inadmissible run is listed, not only the "
                              "first one found"),
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


def summary_ll_problem(config_id: str, seed: int, summary_ll: float | None,
                       metrics_ll: float | None) -> str | None:
    """Why this summary log loss is not the run's own validation log loss.

    Astra round 3 MUST-FIX 3, verbatim in substance: compare each summary LL
    with the manifest-verified ``metrics.json`` validation LL. That preserves
    the designated summary number rather than substituting the LL
    reconstructed from the saved probabilities, which D10.1 forbids as a
    replacement. ``metrics.json`` is inside the completion manifest, so it
    cannot be edited without the admission check failing first.
    """
    if summary_ll is None:
        return None
    if metrics_ll is None:
        return (f"{config_id} seed {seed}: summary.yaml records a validation "
                f"log loss of {summary_ll!r} but the run's metrics.json "
                f"carries no finite {SPLIT}_ll to authenticate it against")
    if abs(float(summary_ll) - float(metrics_ll)) > SUMMARY_LL_TOLERANCE:
        return (f"{config_id} seed {seed}: summary.yaml records validation log "
                f"loss {summary_ll!r}, the run's own manifest-verified "
                f"metrics.json records {metrics_ll!r}; the summary was edited "
                "and no number from it may be read")
    return None


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
    # Astra round 3 MUST-FIX 3: the summary log loss stays the number of
    # record (D9.2), and the manifest-verified metrics.json validation LL
    # authenticates it. Editing only an `ll` in summary.yaml left every
    # checkpoint and signature check satisfied and could change a winner.
    run.summary_ll = (float(summary[seed]) if seed in summary else None)
    if run.summary_ll is not None:
        problem = summary_ll_problem(config_id, seed, run.summary_ll,
                                     admission.metrics_validation_ll)
        if problem:
            raise RefusalError(problem)
    run.arm_params = dict(metrics.get("arm_params") or {})
    record = read_json(directory / "run_record.json")
    run.checkpoint_md5 = admission.checkpoint_md5
    run.provenance = machine_provenance(record, metrics)
    return run


# ---------------------------------------------------------------------------
# Fixed references (rung 4b) — a deterministic npz as a family member's
# reference
# ---------------------------------------------------------------------------
#
# A family member's `contrast.reference` may name a registered CONFIGURATION,
# whose per-seed run directories carry `predictions_validation.npz`, or an
# entry of `statistics.fixed_references`:
#
#   statistics:
#     fixed_references:
#       ref_eb_ctx:
#         npz: models/embeddings/stage4/refs/eb_ctx_validation_probs.npz
#         sha256: <hex>
#         log_loss: 1.4500
#         source: models/embeddings/stage4/refs/references.json
#
# The artifact is DETERMINISTIC: one fit, no seeds. Its per-row probabilities
# are therefore the same at every seed, and it is loaded once and presented to
# `evaluate_contrast` as the identical `Run` at each registered seed. That is
# exactly the registered pairing: estimand (i) resamples blocks alone (the
# reference term is constant within a seed) and estimand (ii) resamples
# seed x block on the CANDIDATE side only, because the reference contributes
# the same row vector at every seed.
#
# ROW IDENTITY. The npz carries `probs` and `y` and no `innings_id`, because
# `scripts/sequence_track/stage4_references.py` writes it row-aligned to the
# validation parquet's own row order (`pd.read_parquet(split_path(...))`, no
# sort, no join) — the same order `load_frame` reads. The row keys are
# therefore DERIVED from the pinned frame, and the alignment is then ASSERTED,
# not assumed: `assert_alignment` compares the npz's own `y` vector against the
# frame's label vector row by row and refuses on the first disagreement, on top
# of the row count, the six-class shape and the row sums. A file written in any
# other order fails that comparison. The sha256 is verified at load, so the
# bytes are the registered artifact's.

FIXED_REFERENCE_ROW_KEY_RULE = (
    "the fixed reference npz carries no innings_id: it is written row-aligned "
    "to the validation parquet's row order by "
    "scripts/sequence_track/stage4_references.py, so the row keys are taken "
    "from the pinned frame in that order and the alignment is asserted "
    "row-by-row against the npz's own y vector (assert_alignment), never "
    "assumed")


def fixed_reference_specs(config: Mapping[str, Any]) -> dict[str, dict]:
    """The config's `statistics.fixed_references`, validated.

    Absent -> `{}`, so every config that registers none behaves exactly as
    before and no code path below it changes.
    """
    raw = ((config.get("statistics") or {}).get("fixed_references") or {})
    if not isinstance(raw, Mapping):
        raise RefusalError(
            "statistics.fixed_references is not a mapping of "
            "{name: {npz, sha256, log_loss, source}}")
    config_ids = {str(entry["id"])
                  for entry in (config.get("configurations") or [])}
    out: dict[str, dict] = {}
    for name, spec in raw.items():
        name = str(name)
        if not isinstance(spec, Mapping):
            raise RefusalError(
                f"fixed reference {name!r} is not a mapping")
        if name in config_ids:
            raise RefusalError(
                f"fixed reference {name!r} is also a registered configuration "
                "id; a reference is one or the other, never both")
        missing = [key for key in ("npz", "sha256") if not spec.get(key)]
        if missing:
            raise RefusalError(
                f"fixed reference {name!r} registers no {missing}; a fixed "
                "reference is a path plus the sha256 its bytes must have")
        digest = str(spec["sha256"]).lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef"
                                    for c in digest):
            raise RefusalError(
                f"fixed reference {name!r} registers sha256 "
                f"{spec['sha256']!r}, which is not a 64-character hex digest")
        out[name] = {
            "name": name,
            "npz": str(spec["npz"]),
            "sha256": digest,
            "log_loss": (None if spec.get("log_loss") is None
                         else float(spec["log_loss"])),
            "source": (None if spec.get("source") is None
                       else str(spec["source"])),
            "label": str(spec.get("label") or name),
        }
    return out


def load_fixed_reference(spec: Mapping[str, Any], frame: Frame,
                         seeds: Sequence[int] = REGISTERED_SEEDS
                         ) -> tuple[dict[int, Run], dict]:
    """Load one deterministic reference as the same `Run` at every seed."""
    name = str(spec["name"])
    path = guard_path(Path(spec["npz"]))
    report: dict[str, Any] = {
        "name": name,
        "npz": rel(path),
        "registered_sha256": spec["sha256"],
        "registered_log_loss": spec.get("log_loss"),
        "source": spec.get("source"),
        "deterministic": True,
        "seed_independent": True,
        "row_key_rule": FIXED_REFERENCE_ROW_KEY_RULE,
        "paired_on_the_candidate_side_only": True,
        "available": False,
        "reason": None,
    }
    if not path.exists():
        report["reason"] = f"{rel(path)} is absent"
        raise RefusalError(
            f"fixed reference {name!r}: {rel(path)} is absent, so no member "
            "taking it as a reference can be computed")
    measured = sha256_file(path)
    report["measured_sha256"] = measured
    if measured != spec["sha256"]:
        report["reason"] = (f"sha256 {measured} != the registered "
                            f"{spec['sha256']}")
        raise RefusalError(
            f"fixed reference {name!r}: {rel(path)} hashes to {measured}, the "
            f"config registers {spec['sha256']}; the bytes are not the "
            "registered artifact's and nothing may be differenced against "
            "them")
    payload = np.load(path, allow_pickle=False)
    for field in ("probs", "y"):
        if field not in payload.files:
            raise RefusalError(
                f"fixed reference {name!r}: {rel(path)} carries no {field!r}")
    probs = np.asarray(payload["probs"], dtype=np.float64)
    y = np.asarray(payload["y"]).astype(np.int64)
    # The row keys the tool pairs on come from the pinned frame, in the
    # parquet's own row order, and the alignment is then ASSERTED: the npz's
    # own label vector must equal the frame's, row for row.
    assert_alignment(f"fixed reference {name}", probs, y, frame.innings_id,
                     frame)
    row_ll = row_log_loss(probs, frame.y)
    report["n_rows"] = int(frame.n_rows)
    report["reconstructed_log_loss"] = float(row_ll.mean())
    report["reconstructed_log_loss_label"] = RECONSTRUCTED_LL_LABEL
    if spec.get("log_loss") is not None:
        report["registered_minus_reconstructed"] = (
            float(spec["log_loss"]) - float(row_ll.mean()))
        if abs(report["registered_minus_reconstructed"]) > 1e-4 + 5e-5:
            raise RefusalError(
                f"fixed reference {name!r}: the config registers log loss "
                f"{spec['log_loss']}, the file reconstructs to "
                f"{row_ll.mean():.6f}; the registration and the artifact "
                "disagree")
    report["available"] = True
    runs = {}
    for seed in seeds:
        run = Run(config_id=name, seed=int(seed), directory=path,
                  admitted=True)
        run.row_ll = row_ll
        run.reconstructed_ll = float(row_ll.mean())
        run.summary_ll = spec.get("log_loss")
        run.arm_params = {"fixed_reference": True}
        run.provenance = {"kind": "fixed reference",
                          "deterministic": True,
                          "sha256": measured}
        runs[int(seed)] = run
    return runs, report


def load_fixed_references(config: Mapping[str, Any], frame: Frame,
                          seeds: Sequence[int] = REGISTERED_SEEDS
                          ) -> tuple[dict[str, dict[int, Run]], dict]:
    """Every registered fixed reference, loaded and hash-verified."""
    runs: dict[str, dict[int, Run]] = {}
    report: dict[str, Any] = {}
    for name, spec in sorted(fixed_reference_specs(config).items()):
        runs[name], report[name] = load_fixed_reference(spec, frame, seeds)
    return runs, report


def load_runs(runs_root: Path, config_ids: Sequence[str], frame: Frame,
              seeds: Sequence[int] = REGISTERED_SEEDS,
              entries: Mapping[str, Mapping[str, Any]] | None = None,
              config: Mapping[str, Any] | None = None,
              pin: Pin | None = None
              ) -> tuple[dict[str, dict[int, Run]], dict, dict]:
    runs: dict[str, dict[int, Run]] = {}
    report: dict[str, Any] = {}
    admissions: dict[str, dict[int, Admission]] = {}
    tampered: list[str] = []
    for config_id in config_ids:
        summary, meta = summary_lls(runs_root, config_id)
        # Astra round 3 MUST-FIX 3: `load_runs` used to read summaries without
        # ever calling the summary verifier, so only the k sweep saw its
        # findings. A summary that does not verify contributes no number: its
        # values are dropped here and the reasons are reported.
        authentication: list[str] = []
        summary_problems: list[str] = []
        if meta.get("available"):
            summary_problems = verify_summary_provenance(
                runs_root, config_id, (entries or {}).get(config_id), meta,
                seeds, out_authentication=authentication)
            if summary_problems:
                summary = {}
        tampered.extend(authentication)
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
            "summary": meta,
            "summary_verification_problems": summary_problems,
            "summary_log_losses_dropped": bool(summary_problems)}
    if tampered:
        # Every tampered summary is named, not only the first one found.
        raise RefusalError(
            f"{len(tampered)} summary log loss(es) do not match the "
            "manifest-verified metrics.json validation log loss of the run "
            "they are attributed to; no number may be read:\n  - "
            + "\n  - ".join(tampered))
    return runs, report, assert_comparable(admissions, entries, config, pin)


# ---------------------------------------------------------------------------
# Contrast machinery (D10.1, D10.3, D10.5, D10.6)
# ---------------------------------------------------------------------------

def mask_digest(mask: np.ndarray) -> str:
    """A row-membership digest for one slice mask (Astra gate 2 round 1 SHOULD 7).

    Slice identity must be compared by row membership, not by equal row, match
    and block counts: two different row sets can agree on all three. The digest
    is the sha256 of the mask's length and its packed bits, so it is stable
    across runs and cheap to persist.
    """
    packed = np.packbits(np.asarray(mask, dtype=bool))
    digest = hashlib.sha256()
    digest.update(str(int(np.asarray(mask).size)).encode())
    digest.update(b":")
    digest.update(packed.tobytes())
    return digest.hexdigest()


def slice_stats(frame: Frame, blocks: Blocks, name: str) -> dict:
    mask = frame.masks.get(name)
    if mask is None:
        return {"slice": name, "available": False, "n_rows": 0,
                "n_matches": 0, "n_blocks": 0, "descriptive": True,
                "mask_sha256": None}
    n_blocks = count_unique_clusters(blocks.block_id[mask])
    return {"slice": name, "available": True,
            "n_rows": int(mask.sum()),
            "n_matches": int(len(set(frame.match_id[mask].tolist()))),
            "n_blocks": int(n_blocks),
            "descriptive": bool(n_blocks < MIN_BLOCKS),
            "min_blocks": MIN_BLOCKS,
            "mask_sha256": mask_digest(mask),
            "mask_digest_note": ("sha256 of the mask length and its packed "
                                 "bits; slice identity is compared by this "
                                 "row-membership digest, never by equal row, "
                                 "match and block counts")}


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
        draws_out[(key, JOINT_READOUT)] = joint
    mean_point = float(np.mean([per_seed_points[str(s)] for s in seeds]))
    record["estimand_ii"] = _readout(joint, mean_point, threshold,
                                     descriptive, reps)
    record["estimand_ii"]["estimand"] = (
        "(ii) arithmetic across-seed mean, joint seed-and-block resampling")
    record["estimand_ii"]["label"] = estimand_ii_label(len(seeds))
    record["estimand_ii"]["is_log_loss_of_averaged_probabilities"] = False

    points = [per_seed_points[str(s)] for s in seeds]
    record["per_seed_points"] = per_seed_points
    record["seed_spread"] = {"min": float(min(points)),
                             "max": float(max(points)),
                             "range": float(max(points) - min(points))}
    # Astra gate 2 round 3 MUST-FIX 4: this count is "seeds whose point is
    # below the registered threshold", and on a non-inferiority gate that
    # threshold is +0.002, not zero. Reporting it as "favourable seeds" made a
    # row of five adverse-but-small deltas read as five favourable seeds. The
    # count itself and its key are unchanged — the gate arithmetic and every
    # primary's zero-threshold count depend on them — and the strictly
    # favourable (negative-delta) count is reported beside it, with the
    # threshold exposed so neither number can be read as the other.
    below_threshold = int(sum(1 for p in points if p < threshold))
    record["favourable_direction_count"] = below_threshold
    record["direction_count_threshold"] = float(threshold)
    record["seeds_below_registered_threshold_count"] = below_threshold
    record["seeds_below_zero_count"] = int(sum(1 for p in points if p < 0.0))
    record["direction_count_note"] = (
        "`seeds_below_registered_threshold_count` (= "
        "`favourable_direction_count`, the registered key) counts per-seed "
        f"points below the registered threshold {float(threshold)!r}; on a "
        "non-inferiority gate that threshold is the margin, so a seed counted "
        "there may still have an ADVERSE (positive) delta. "
        "`seeds_below_zero_count` is the count of strictly favourable "
        "per-seed deltas. The two coincide only where the threshold is zero")
    record["n_seeds"] = len(points)
    record["available"] = True
    return record


JOINT_READOUT = "seed_mean_joint"


def readouts_for(seeds: Sequence[int] = REGISTERED_SEEDS) -> tuple[str, ...]:
    """One estimand (i) readout per registered seed, then the joint mean.

    Astra gate 2 round 2: this was the hard-coded tuple
    `("seed_7", "seed_13", "seed_mean_joint")`, so a five-seed run computed
    five checkpoint contrasts and then dropped seeds 29, 42 and 101 out of the
    Holm tables, the gates and the family screens. Every consumer derives its
    readout list from the seeds it was given, and the statistics JSON records
    the list it used in `contract.readouts`.
    """
    return tuple(f"seed_{int(s)}" for s in seeds) + (JOINT_READOUT,)


# The two-seed default, retained so callers that never pass seeds keep working.
READOUTS = readouts_for(REGISTERED_SEEDS)


def readout_of(record: Mapping[str, Any], readout: str) -> dict | None:
    if not record.get("available"):
        return None
    if readout == JOINT_READOUT:
        return record.get("estimand_ii")
    return (record.get("estimand_i") or {}).get(readout)


# ---------------------------------------------------------------------------
# D10.4 / D10.6 — families, Holm, gates
# ---------------------------------------------------------------------------

def shared_control(config: Mapping[str, Any]) -> str:
    """The config's shared control id.

    `statistics.families.shared_control` replaces the `"mlp"` literal that was
    written into the family count, the candidate check, the screen's extra
    all-row condition and the exploratory pair generator. Absent -> `"mlp"`,
    so a config carrying none of the night-3 keys behaves exactly as before.
    """
    families = ((config.get("statistics") or {}).get("families") or {})
    return str(families.get("shared_control") or DEFAULT_SHARED_CONTROL)


def config_roles(config: Mapping[str, Any]) -> dict[str, str]:
    """Per-configuration role: `candidate` (default) or `control`.

    The shared control is a control whether or not it says so. A control-role
    configuration need not be a candidate in any family, so the "every
    non-control configuration is a candidate exactly once" check applies to
    candidates only.
    """
    control = shared_control(config)
    roles: dict[str, str] = {}
    for entry in (config.get("configurations") or []):
        config_id = str(entry["id"])
        role = str(entry.get("role") or CANDIDATE_ROLE)
        if role not in (CANDIDATE_ROLE, CONTROL_ROLE):
            raise RefusalError(
                f"configuration {config_id!r} registers role {role!r}, which "
                f"is not one of {[CANDIDATE_ROLE, CONTROL_ROLE]}")
        roles[config_id] = CONTROL_ROLE if config_id == control else role
    return roles


def candidate_ids(config: Mapping[str, Any]) -> set[str]:
    """The configuration ids that must each be a family candidate once."""
    return {cid for cid, role in config_roles(config).items()
            if role == CANDIDATE_ROLE}


def expected_family_count(config: Mapping[str, Any]) -> int:
    """How many families this config must register, derived from the config.

    Every candidate-role configuration is a family candidate exactly once
    (asserted at the end of `registered_families`), so the count is the number
    of registered configurations minus the shared control and minus every
    explicit `role: control` configuration. Astra gate 2 round 2: this was the
    constant 15, so a config registering seven families — or any other
    legitimate subset — refused unless the caller happened to pass a flag.
    Deriving it lets a seven-family and a fifteen-family config both run with
    no flag, and still refuses a config whose family map does not cover its own
    candidates.
    """
    return len(candidate_ids(config))


def _legacy_members(entry: Mapping[str, Any], candidate: str,
                    ids: set[str], expected_slices: Mapping[str, str],
                    ) -> list[dict]:
    """Translate the fixed {primary, death_gate, chase_gate} form.

    This is the ONLY place the legacy shape is read. Everything downstream sees
    the general member form, so the Stage 2 families and a night-3 family go
    through identical Holm, gate, screen and reporting code.
    """
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
                                 else "non_inferiority"),
                        "primary": name == "primary"})
    return members


def _general_members(entry: Mapping[str, Any], candidate: str,
                     ids: set[str],
                     reference_ids: set[str] | None = None) -> list[dict]:
    """Read the general `members:` list — 2 to 6 members, each self-describing.

    Each member carries its own name, contrast (candidate and reference), slice,
    threshold and kind. Exactly one member is flagged `primary: true`; that is
    the member the 4/5 favourable-direction rule reads. Registered member order
    is the Holm tie order, unchanged.

    A member's CANDIDATE is always a registered configuration. Its REFERENCE
    may be a registered configuration or, since rung 4b, a registered
    `statistics.fixed_references` entry; `reference_ids` is the allowed
    reference set and defaults to `ids`, so a config registering no fixed
    reference is validated exactly as before.
    """
    reference_ids = set(ids if reference_ids is None else reference_ids)
    raw = entry.get("members")
    if not isinstance(raw, list):
        raise RefusalError(
            f"family {candidate!r} registers `members` that is not a list")
    if not FAMILY_MIN_MEMBERS <= len(raw) <= FAMILY_MAX_MEMBERS:
        raise RefusalError(
            f"family {candidate!r} registers {len(raw)} members; a family has "
            f"{FAMILY_MIN_MEMBERS}-{FAMILY_MAX_MEMBERS}")
    members: list[dict] = []
    for position, member in enumerate(raw):
        if not isinstance(member, Mapping):
            raise RefusalError(
                f"family {candidate!r} member {position} is not a mapping")
        name = str(member.get("name") or "")
        if not name:
            raise RefusalError(
                f"family {candidate!r} member {position} has no name")
        if any(m["member"] == name for m in members):
            raise RefusalError(
                f"family {candidate!r} registers member {name!r} twice")
        contrast = member.get("contrast")
        if not isinstance(contrast, Mapping):
            raise RefusalError(
                f"family {candidate!r} member {name!r} has no "
                "`contrast: {candidate, reference}`")
        member_candidate = str(contrast.get("candidate") or candidate)
        reference = str(contrast.get("reference") or "")
        if member_candidate not in ids:
            raise RefusalError(
                f"family {candidate!r} member {name!r} names candidate "
                f"{member_candidate!r}, which is not a registered "
                "configuration")
        if reference not in reference_ids:
            extra = ("" if reference_ids == ids else
                     " and no registered fixed reference")
            raise RefusalError(
                f"family {candidate!r} member {name!r} names reference "
                f"{reference!r}, which is not a registered configuration"
                + extra)
        kind = str(member.get("kind") or "superiority")
        if kind not in MEMBER_KINDS:
            raise RefusalError(
                f"family {candidate!r} member {name!r} registers kind "
                f"{kind!r}, not one of {list(MEMBER_KINDS)}")
        if "threshold" in member:
            threshold = float(member["threshold"])
        else:
            threshold = 0.0 if kind == "superiority" else MARGIN_LL
        if kind == "superiority" and threshold != 0.0:
            raise RefusalError(
                f"family {candidate!r} member {name!r} is a superiority test, "
                f"whose threshold is 0, not {threshold}")
        slice_name = str(member.get("slice") or "")
        if not slice_name:
            raise RefusalError(
                f"family {candidate!r} member {name!r} names no slice")
        members.append({
            "member": name, "candidate": member_candidate,
            "reference": reference, "slice": slice_name,
            "contrast": str(member.get("label")
                            or f"{member_candidate} - {reference}"),
            "threshold": threshold, "kind": kind,
            "primary": bool(member.get("primary")),
        })
    flagged = [m["member"] for m in members if m["primary"]]
    if len(flagged) != 1:
        raise RefusalError(
            f"family {candidate!r} flags {len(flagged)} members "
            f"`primary: true` ({flagged}); exactly one is required, and it is "
            "the member the favourable-direction rule reads")
    return members


def _family_screen_spec(entry: Mapping[str, Any], candidate: str,
                        members: Sequence[Mapping[str, Any]],
                        legacy: bool) -> dict:
    """The per-family screen condition and its all-row condition.

    Default, when the family registers no `screen`, is the inherited rule: the
    primary rejects and every non-inferiority member holds. `screen.require`
    names the members that must ALL reject favourably for SCREEN_PASS.
    `all_row_condition` is `inherited` unless registered otherwise.
    `direction_required` names the members the >=4/5 favourable-direction rule
    is enforced on; the default is the primary member alone, so Stage 2 and
    every legacy family are unchanged.
    """
    names = [m["member"] for m in members]
    raw = entry.get("screen")
    if raw is None:
        require = list(names) if legacy else [
            m["member"] for m in members
            if m["primary"] or m["kind"] == "non_inferiority"]
        registered = False
    else:
        if not isinstance(raw, Mapping):
            raise RefusalError(
                f"family {candidate!r} registers a `screen` that is not a "
                "mapping")
        require = [str(n) for n in (raw.get("require") or [])]
        if not require:
            raise RefusalError(
                f"family {candidate!r} registers a screen with no "
                "`require: [member names]`")
        unknown = [n for n in require if n not in names]
        if unknown:
            raise RefusalError(
                f"family {candidate!r} screen requires members {unknown} that "
                f"the family does not register ({names})")
        registered = True
    condition = str(entry.get("all_row_condition") or "inherited")
    if condition not in ALL_ROW_CONDITIONS:
        raise RefusalError(
            f"family {candidate!r} registers all_row_condition "
            f"{condition!r}, not one of {list(ALL_ROW_CONDITIONS)}")
    primary_names = [m["member"] for m in members if m.get("primary")]
    raw_directions = entry.get("direction_required")
    if raw_directions is None:
        directions = list(primary_names)
    else:
        if not isinstance(raw_directions, (list, tuple)):
            raise RefusalError(
                f"family {candidate!r} registers `direction_required` that is "
                "not a list of member names")
        directions = [str(n) for n in raw_directions]
        if not directions:
            raise RefusalError(
                f"family {candidate!r} registers an empty "
                "`direction_required`; the rule is enforced on at least the "
                "primary member")
        unknown = [n for n in directions if n not in names]
        if unknown:
            raise RefusalError(
                f"family {candidate!r} requires favourable directions on "
                f"members {unknown} that the family does not register "
                f"({names})")
        missing_primary = [n for n in primary_names if n not in directions]
        if missing_primary:
            raise RefusalError(
                f"family {candidate!r} registers `direction_required` "
                f"{directions}, which omits its primary member "
                f"{missing_primary}; the rule always covers the primary")
    return {"require": require,
            "screen_is_registered_per_family": registered,
            "all_row_condition": condition,
            "all_row_condition_is_registered": (
                "all_row_condition" in entry),
            "direction_required": directions,
            "direction_required_is_registered": (
                "direction_required" in entry)}


def registered_families(config: Mapping[str, Any],
                        expected: int | None = None) -> list[dict]:
    """Assert and return the config's explicit families.

    A family entry uses either the legacy fixed
    {primary, death_gate, chase_gate} form — translated internally into the
    general form by `_legacy_members` — or the general
    ``members: [{name, contrast: {candidate, reference}, slice, threshold,
    kind}]`` form with 2 to 6 members. Holm step-down, tie order (registered
    member order), the missing-member placeholder, strict bounds, both
    estimands, the 4/5 favourable-direction rule on the member flagged
    `primary: true` and the ten-block rule are the same for both.

    ``expected`` is the registered family count. When it is None — the default
    — it is DERIVED from the config by `expected_family_count`, one family per
    candidate-role configuration. A caller may still pass a number to assert a
    specific count.
    """
    if expected is None:
        expected = expected_family_count(config)
    statistics = config.get("statistics") or {}
    families = (statistics.get("families") or {})
    control = shared_control(config)
    roles = config_roles(config)
    raw = families.get("map")
    if not isinstance(raw, list):
        raise RefusalError(
            "the config carries no statistics.families.map; D10.4 requires an "
            "explicit family map, never a template")
    if expected is not None and len(raw) != expected:
        raise RefusalError(
            f"the config registers {len(raw)} families, not {expected}")
    ids = {str(entry["id"]) for entry in (config.get("configurations") or [])}
    # Rung 4b: a member's reference may be a deterministic npz registered in
    # `statistics.fixed_references`. Empty for every config that registers
    # none, which is every config written before this.
    fixed_names = set(fixed_reference_specs(config))
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
        legacy = "members" not in entry
        if legacy:
            members = _legacy_members(entry, candidate, ids, expected_slices)
        else:
            members = _general_members(entry, candidate, ids,
                                       ids | fixed_names)
        known = {"candidate", "holm_group", "note"}
        known |= (set(FAMILY_MEMBER_ORDER) if legacy
                  else {"members", "screen", "all_row_condition"})
        extra = [k for k in entry if k not in known]
        family = {"candidate": candidate,
                  "holm_group": str(entry.get("holm_group")
                                    or f"family_{candidate}"),
                  "note": entry.get("note"),
                  "unknown_keys": extra,
                  "members": members,
                  # Carried on every family so the screen reads the registered
                  # control rather than a literal. Emitted only for a general
                  # family, so a legacy family's payload is unchanged.
                  "shared_control": control}
        if not legacy:
            # Only a general family carries the generalised keys, so a legacy
            # config's family payload is byte-for-byte what it was.
            family["legacy_member_form_translated"] = False
            family["shared_control"] = control
            family["screen_spec"] = _family_screen_spec(
                entry, candidate, members, legacy)
        out.append(family)
    required = {cid for cid in ids if roles.get(cid) == CANDIDATE_ROLE}
    extra_candidates = required - seen
    if extra_candidates:
        raise RefusalError(
            f"every candidate-role configuration must be a family candidate "
            f"exactly once; missing {sorted(extra_candidates)}")
    controls_as_candidates = sorted(
        cid for cid in seen if roles.get(cid) == CONTROL_ROLE)
    if controls_as_candidates:
        which = controls_as_candidates[0]
        if which == control:
            raise RefusalError(
                f"{which} is the shared control and is a candidate in "
                "no family")
        raise RefusalError(
            f"{which} registers `role: control` and is a candidate in "
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
        numerical = readout["u95"] < float(member.get("threshold", MARGIN_LL))
    else:
        numerical = readout["u95"] < 0.0
    return _check_status(STATUS_PASS if (numerical and rejected)
                         else STATUS_NOT_PASS)


def holm_family(family: Mapping[str, Any],
                contrasts: Mapping[str, Mapping[str, Any]],
                readout: str) -> dict:
    """Holm step-down over one family's members (2-6), for one readout.

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
                  bool(values["u95"] < (float(member["threshold"])
                                        if member["kind"] == "non_inferiority"
                                        else 0.0)))
        rejected = bool(reject and not row["placeholder"]
                        and favourable and strict)
        status = member_status(member, values, rejected)
        if status == STATUS_NOT_EVALUABLE:
            family_evaluable = False
        level = holm_level(rank, len(family["members"]))
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
    return {"readout": readout, "alpha": ALPHA, "m": len(family["members"]),
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

    A candidate clears it only when every member the family's screen REQUIRES
    rejects favourably under Holm with a strict bound (U95 < 0 for a
    superiority member, U95 < its margin for a non-inferiority member) and, when
    the family's ``all_row_condition`` is the inherited default, its all-row
    ``candidate - shared_control`` interval is CI-clean favourable. Default
    ``require`` is the old rule: the registered primary plus both gates. A
    family may register ``screen: {require: [...]}`` and
    ``all_row_condition: member|none``; the screen records which it used.
    That last reading is exploratory wherever it falls outside the family; it
    is a necessary condition here and never a replacement for the primary.

    At five or more seeds the registered extension qualification adds one
    requirement: at least 4 of 5 favourable per-seed directions on the
    registered primary (``FIVE_SEED_ELIGIBILITY_RULE``). Astra gate 2 round 2:
    the count was computed and never enforced, so a reported PASS was not the
    extension qualification. Below five seeds the count cannot reach 4/5, so
    the requirement is reported as not applicable and a two-seed run's status
    is exactly what it was before.
    """
    table = tables[readout]
    by_member = {row["member"]: row for row in table["members"]}
    spec = family.get("screen_spec") or {
        "require": list(FAMILY_MEMBER_ORDER),
        "screen_is_registered_per_family": False,
        "all_row_condition": "inherited",
        "all_row_condition_is_registered": False,
        "direction_required": None,
        "direction_required_is_registered": False}
    registered = [m for m in family["members"] if m.get("primary")]
    primary_name = registered[0]["member"] if registered else "primary"
    primary = by_member[primary_name]
    required = [by_member[name] for name in spec["require"]]
    gates = [row for row in required if row["member"] != primary_name]
    control = family.get("shared_control") or DEFAULT_SHARED_CONTROL
    candidate = family["candidate"]
    all_row_key = f"{candidate}-{control}@{PRIMARY_SLICE}"
    if spec["all_row_condition"] == "inherited":
        all_row = readout_of(contrasts.get(all_row_key) or {}, readout)
        all_row_clean = (None if all_row is None
                         else bool(all_row["ci_clean_favourable"]))
    else:
        # `member` discharges the condition through a registered member, whose
        # own status already enters the screen; `none` registers that this
        # family carries no extra all-row condition. Either way the extra
        # exploratory read is not a necessary condition here, and the screen
        # cannot be made NOT_EVALUABLE by its absence.
        all_row = None
        all_row_clean = None

    # The registered five-seed extension qualification, read off each covered
    # member's own per-seed points rather than any one readout. The default
    # coverage is the primary member alone — the Stage 2 rule — and a family may
    # register `direction_required` to enforce it on several members, as the 3a
    # families do on BOTH superiority members.
    primary_record = contrasts.get(primary["contrast_key"]) or {}
    n_seeds = int(primary_record.get("n_seeds") or 0)
    direction_count = primary_record.get("favourable_direction_count")
    applies = n_seeds >= FIVE_SEED_MINIMUM

    def _direction_ok(record: Mapping[str, Any]) -> bool | None:
        """None when the rule cannot apply; else whether the member meets it."""
        seeds_seen = int(record.get("n_seeds") or 0)
        if seeds_seen < FIVE_SEED_MINIMUM:
            return None
        count = record.get("favourable_direction_count")
        if count is None:
            return False
        return int(count) >= FIVE_SEED_FAVOURABLE_DIRECTIONS

    covered = [name for name in spec.get("direction_required")
               or [primary_name]]
    by_direction_member: dict[str, dict] = {}
    for name in covered:
        row = by_member[name]
        record = contrasts.get(row["contrast_key"]) or {}
        by_direction_member[name] = {
            "contrast_key": row["contrast_key"],
            "n_seeds": int(record.get("n_seeds") or 0),
            "favourable_direction_count": record.get(
                "favourable_direction_count"),
            "requirement_applies": (
                int(record.get("n_seeds") or 0) >= FIVE_SEED_MINIMUM),
            "favourable_direction_requirement_met": _direction_ok(record),
        }
    per_member_ok = [row["favourable_direction_requirement_met"]
                     for row in by_direction_member.values()]
    if any(value is False for value in per_member_ok):
        direction_ok: bool | None = False
    elif per_member_ok and all(value is True for value in per_member_ok):
        direction_ok = True
    else:
        # No covered member reaches the registered seed minimum, so the rule
        # neither applies nor can fail the status.
        direction_ok = None

    # GENERAL-form families (the night-3 schema) are five-seed screens by
    # construction: a reduced-seed run makes the direction requirement "not
    # applicable", and without this a three-seed or two-seed screen could
    # still report SCREEN_PASS. Every REQUIRED member must carry five complete
    # paired seeds, whatever `--seeds` was passed. Legacy-form families keep
    # the Stage 2 behaviour, so the two-seed screen reproduces exactly.
    general_form = "screen_spec" in family
    required_seed_counts = {
        row["member"]: int(
            (contrasts.get(row["contrast_key"]) or {}).get("n_seeds") or 0)
        for row in required}
    members_below_five = sorted(
        name for name, count in required_seed_counts.items()
        if count < FIVE_SEED_MINIMUM)
    seeds_complete = not members_below_five
    status_reason = None

    statuses = [row["status"] for row in required]
    inherited = spec["all_row_condition"] == "inherited"
    if (STATUS_NOT_EVALUABLE in statuses
            or (inherited and (all_row is None
                               or all_row["descriptive_only"]))):
        status = STATUS_NOT_EVALUABLE
    elif general_form and not seeds_complete:
        status = STATUS_NOT_PASS
        status_reason = "fewer_than_five_seeds"
    elif (all(row["status"] == STATUS_PASS for row in required)
          and (all_row_clean if inherited else True)
          and (direction_ok is not False)):
        status = STATUS_PASS
    else:
        status = STATUS_NOT_PASS
    # The five-complete-seeds keys are emitted for GENERAL-form families only:
    # a legacy stage 2 family's payload must stay byte-identical.
    five_seed_keys = ({
        "status_reason": status_reason,
        "five_complete_seeds_required": general_form,
        "required_member_n_seeds": dict(required_seed_counts),
        "required_members_below_five_seeds": list(members_below_five),
    } if general_form else {})
    return {"readout": readout,
            "candidate": candidate,
            "status": _check_status(status),
            **five_seed_keys,
            "n_seeds": n_seeds,
            "five_seed_eligibility_rule": FIVE_SEED_ELIGIBILITY_RULE,
            "five_seed_direction_requirement_applies": applies,
            "required_favourable_directions": (
                FIVE_SEED_FAVOURABLE_DIRECTIONS if applies else None),
            "favourable_direction_count": direction_count,
            # MUST-FIX 4: the primary's threshold is zero, so this count is
            # both "below the registered threshold" and "strictly favourable".
            # Both keys are published so the qualification table can label the
            # column it prints, and the threshold is exposed beside them.
            "direction_count_threshold":
                primary_record.get("direction_count_threshold"),
            "seeds_below_registered_threshold_count":
                primary_record.get("seeds_below_registered_threshold_count"),
            "seeds_below_zero_count":
                primary_record.get("seeds_below_zero_count"),
            "favourable_direction_requirement_met": direction_ok,
            "five_seed_extension_qualified": (
                bool(status == STATUS_PASS and applies and direction_ok)),
            "five_seed_qualification_note": (
                (f"{direction_count} of {n_seeds} per-seed primary directions "
                 f"are favourable; "
                 f"{FIVE_SEED_FAVOURABLE_DIRECTIONS} of "
                 f"{FIVE_SEED_MINIMUM} are required, so this requirement is "
                 + ("met" if direction_ok else "NOT met"))
                if applies else
                (f"not applicable at {n_seeds} seed(s): the "
                 f"{FIVE_SEED_FAVOURABLE_DIRECTIONS}/"
                 f"{FIVE_SEED_MINIMUM} direction count cannot be reached "
                 "below five seeds, so it is neither required nor able to "
                 "fail this status. A PASS here is the two-seed directional "
                 "screen and is NOT the five-seed extension qualification")),
            "primary_status": primary["status"],
            "primary_rejected": primary["rejected"],
            "primary_u95_below_zero": (
                None if primary["u95"] is None else primary["u95"] < 0.0),
            **({} if "screen_spec" not in family else {
                "direction_required_members": list(covered),
                "direction_required_is_registered":
                    spec["direction_required_is_registered"],
                "direction_requirement_by_member": by_direction_member,
                "all_direction_requirements_met": direction_ok}),
            **({"death_gate_status": by_member["death_gate"]["status"],
                "chase_gate_status": by_member["chase_gate"]["status"]}
               if "screen_spec" not in family else
               {"primary_member": primary_name,
                "shared_control": control,
                "screen_required_members": list(spec["require"]),
                "screen_is_registered_per_family":
                    spec["screen_is_registered_per_family"],
                "screen_required_member_statuses": {
                    row["member"]: row["status"] for row in required},
                "all_row_condition": spec["all_row_condition"],
                "all_row_condition_is_registered":
                    spec["all_row_condition_is_registered"],
                "gate_statuses": {row["member"]: row["status"]
                                  for row in gates}}),
            "all_row_candidate_minus_mlp_key": all_row_key,
            "all_row_candidate_minus_mlp_ci_clean_favourable": all_row_clean,
            "all_row_reading_is_exploratory_when_outside_the_family": (
                family["members"][0]["reference"] != control),
            # Astra gate 2 round 3 SHOULD 5: this said "two seeds" whatever the
            # run's seed count, so a five-seed family JSON described itself as
            # a two-seed screen. The count is derived from the primary
            # contrast's own seeds; what does NOT change with the count is that
            # this is validation-only, same-split checkpoint selection.
            "evidence_status": (f"screening: {seed_word(n_seeds)} "
                                f"seed{'' if n_seeds == 1 else 's'}, "
                                "validation only, "
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

MECHANISM_SOURCE_CONFIG = "statistics.contrasts entries with role: mechanism"
MECHANISM_SOURCE_BUILTIN = "the frozen Stage 2 MECHANISM_CONTRASTS tuple"


def mechanism_contrasts(config: Mapping[str, Any]
                        ) -> tuple[tuple[tuple[str, str, str], ...], str]:
    """The registered mechanism contrasts, and where they came from.

    Read from `statistics.contrasts` entries with `role: mechanism` when EVERY
    such entry carries an explicit non-empty `registered_pairs`. A config in
    which some mechanism entries name their pairs and others only describe them
    in prose is not a complete registration, so the frozen tuple is used and
    Stage 2 reproduces exactly. Each pair is `[candidate, reference]`,
    `[candidate, reference, label]`, or
    `{candidate, reference, label}`; the label defaults to the entry's `tests`
    text, then its `name`.
    """
    entries = [e for e in ((config.get("statistics") or {}).get("contrasts")
                           or []) if isinstance(e, Mapping)
               and str(e.get("role") or "") == "mechanism"]
    if not entries or not all(e.get("registered_pairs") for e in entries):
        return MECHANISM_CONTRASTS, MECHANISM_SOURCE_BUILTIN
    out: list[tuple[str, str, str]] = []
    for entry in entries:
        default_label = str(entry.get("tests") or entry.get("name") or "")
        for pair in entry["registered_pairs"]:
            if isinstance(pair, Mapping):
                candidate = str(pair.get("candidate") or "")
                reference = str(pair.get("reference") or "")
                label = str(pair.get("label") or default_label)
            else:
                items = list(pair)
                if len(items) not in (2, 3):
                    raise RefusalError(
                        f"mechanism contrast {entry.get('name')!r} registers "
                        f"pair {pair!r}, which is not [candidate, reference] "
                        "or [candidate, reference, label]")
                candidate, reference = str(items[0]), str(items[1])
                label = str(items[2]) if len(items) == 3 else default_label
            if not candidate or not reference:
                raise RefusalError(
                    f"mechanism contrast {entry.get('name')!r} registers a "
                    "pair with no candidate or no reference")
            if not label:
                raise RefusalError(
                    f"mechanism contrast {candidate} - {reference} carries no "
                    "label; a mechanism contrast states what it tests")
            out.append((candidate, reference, label))
    return tuple(out), MECHANISM_SOURCE_CONFIG


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
    record["estimand_ii"]["label"] = estimand_ii_label(len(seeds))
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
                              seeds: Sequence[int],
                              out_authentication: list[str] | None = None
                              ) -> list[str]:
    """Every reason this summary's log losses may not be read (D9.2).

    Called BEFORE the log losses are used. Each recorded seed row must name a
    checkpoint directory that itself passes `verify_run_admission`, must carry
    the same training signature as the summary's own header, the checkpoint
    md5 it claims must still be the md5 of the file on disk, and — Astra round
    3 MUST-FIX 3 — its log loss must equal the manifest-verified
    ``metrics.json`` validation log loss of that run. `out_authentication`
    collects that last class of problem separately, because a summary whose
    numbers were edited is tampering rather than an incomplete night.
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
        # The designated summary number, authenticated against the run's own
        # manifest-verified metrics.json (never replaced by it).
        authentication = summary_ll_problem(
            config_id, seed, _float_or_none(value),
            admission.metrics_validation_ll)
        if authentication:
            problems.append(authentication)
            if out_authentication is not None:
                out_authentication.append(authentication)
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
    # Astra gate 2 round 2 follow-up: the record's own prose named "two-seed"
    # regardless of the seeds it was given, so a five-seed selection record
    # labelled itself a two-seed screen and the report quoted that label. The
    # count is written from the seed list; at two seeds the wording is the
    # registered wording verbatim.
    word = seed_word(len(seeds))

    # Astra round 3 MUST-FIX 3: `k_sweep` never applied comparability, so five
    # individually consistent summaries from incompatible training runs could
    # enter one selection. The five k configurations are checked against each
    # other and against the pin BEFORE any mean is computed.
    pin = load_pin(config)
    admissions: dict[str, dict[int, Admission]] = {}
    for _, config_id in mapping:
        admissions[config_id] = {
            one: verify_run_admission(
                Path(runs_root) / config_id / f"seed_{one}", config_id, one,
                expected_arm=(entries.get(config_id) or {}).get("arm"),
                expected_k=((entries.get(config_id) or {}).get("params")
                            or {}).get("k", "__unset__"))
            for one in seeds}
    comparability = assert_comparable(admissions, entries, config, pin)

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
        "rule": (f"choose the k with the lowest {word}-seed arithmetic mean "
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
        "labelling": f"{word}-seed directional screen",
        "interpretation_guard": (
            "the 0.002 tolerance is a selection tolerance, not a significance "
            f"threshold and not demonstrated {word}-seed resolution; the "
            "chosen k "
            "is never called reliably optimal"),
        "provisional": True,
        # Astra gate 2 round 3 SHOULD 5: below the registered five-seed minimum
        # the outstanding condition really is a whole-family seed extension, and
        # that wording is kept verbatim. At or above it the extension has
        # happened, so the note must not imply that another one is already
        # required; what remains outstanding is that this is validation-only
        # and awaits final disposition.
        "provisional_note": (
            f"the {word}-seed selection remains explicitly provisional "
            "pending any registered whole-family seed extension"
            if len(seeds) < FIVE_SEED_MINIMUM else
            f"the {word}-seed selection is validation-only and remains "
            "explicitly provisional, awaiting final disposition; no further "
            "seed extension is implied by this note"),
        "k_to_config_id": {k: config_id for k, config_id in mapping},
        "k_to_config_id_source": ("derived from the config's own "
                                  "`configurations` entries by arm and "
                                  "`params.k`, never by string template"),
        "admission_rejections": rejected,
        "comparability": comparability,
        "comparability_note": ("the five k configurations are checked against "
                               "each other and against the pin before any "
                               "mean is computed, so individually consistent "
                               "summaries from incompatible training runs "
                               "cannot enter one selection"),
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
            "spread_note": (f"empirical spread over {word} seeds, not a "
                            "confidence interval"),
        }

    flags = []
    if selected_k != "30":
        block = paired[selected_k]
        values = list(block["per_seed_delta_vs_k30"].values())
        if min(values) < 0 < max(values):
            flags.append("paired differences change sign across the "
                         f"{word} seeds")
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
                       expected_families: int | None = None) -> dict:
    config = read_yaml(config_path) or {}
    entries = {str(entry["id"]): entry
               for entry in (config.get("configurations") or [])}
    config_ids = list(entries)
    if not config_ids:
        raise RefusalError(f"{rel(config_path)} registers no configurations")
    families = registered_families(config, expected_families)
    readouts = readouts_for(seeds)
    pin = load_pin(config)
    control = shared_control(config)
    mechanisms, mechanism_source = mechanism_contrasts(config)

    frame = load_frame(frame_dir, config)
    expect = dict(expect or {})
    blocks = resolve_blocks(
        frame, block_source,
        expect_rows=int(expect.get("rows", EXPECTED_ROWS)),
        expect_matches=int(expect.get("matches", EXPECTED_MATCHES)),
        expect_blocks=int(expect.get("blocks", EXPECTED_BLOCKS)),
        expect_unmapped=int(expect.get("unmapped", EXPECTED_UNMAPPED)))
    runs, runs_report, comparability = load_runs(runs_root, config_ids, frame,
                                                 seeds, entries, config, pin)
    # Rung 4b: the deterministic references a family member may take as its
    # reference. They are NOT runs — they never enter `runs_report` or the
    # comparability anchoring, which are about training provenance — and they
    # present the same row vector at every seed, so estimand (ii) resamples
    # seed x block on the candidate side only.
    fixed_runs, fixed_report = load_fixed_references(config, frame, seeds)
    runs.update(fixed_runs)

    slice_names = ([p.name for p in SLICE_PREDICATES] + [THIN_PAIR_NAME]
                   + [r["name"] for r in optional_slice_specs(config)])
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
    for candidate, reference, label in mechanisms:
        add(candidate, reference, PRIMARY_SLICE, "mechanism", label)
    # Exploratory: every registered contrast on every available slice.
    registered_pairs = {(m["candidate"], m["reference"])
                        for f in families for m in f["members"]}
    registered_pairs |= {(c, r) for c, r, _ in mechanisms}
    registered_pairs |= {(cid, control) for cid in config_ids
                         if cid != control}
    gate_slice_names = set(GATE_SLICES) | {
        row["slice"] for row in frame.predicates if row["role"] == "gate"}
    for candidate, reference in sorted(registered_pairs):
        for name in slice_names:
            if name not in frame.masks:
                continue
            add(candidate, reference, name, "exploratory",
                f"{candidate} - {reference} on {name} (exploratory)",
                MARGIN_LL if name in gate_slice_names and reference == control
                else 0.0)

    family_out = []
    for family in families:
        tables = {}
        for readout in readouts:
            table = holm_family(family, contrasts, readout)
            rank_local_intervals(table, contrasts, draws_cache)
            tables[readout] = table
        screens = {readout: family_screen(family, tables, contrasts, readout)
                   for readout in readouts}
        general = "screen_spec" in family
        family_out.append({
            "candidate": family["candidate"],
            "holm_group": family["holm_group"],
            "note": family["note"],
            # The `primary` flag is an artefact of the general form; a
            # translated legacy family emits exactly the member fields it
            # always emitted.
            "members": (family["members"] if general else
                        [{k: v for k, v in member.items() if k != "primary"}
                         for member in family["members"]]),
            "holm": tables,
            "screen": screens,
            "registered_for_confirmation_results_remain_screening": True,
            **({"shared_control": family["shared_control"],
                "screen_spec": family["screen_spec"]} if general else {}),
        })

    gates = []
    for family in families:
        for member in family["members"]:
            # Every non-inferiority member of every family is a gate row. In
            # the legacy form that is exactly "not the primary".
            if member["kind"] != "non_inferiority":
                continue
            key = f"{member['candidate']}-{member['reference']}@{member['slice']}"
            record = contrasts.get(key) or {}
            margin = float(member["threshold"])
            row = {"candidate": member["candidate"],
                   "reference": member["reference"],
                   "slice": member["slice"],
                   "contrast_key": key,
                   "margin_ll": margin,
                   "numerical_pass_rule": f"strictly U95 < {margin}",
                   "family_adjusted_gate_additionally_requires": (
                       f"favourable Holm rejection against the +{margin} "
                       f"boundary, at least {MIN_BLOCKS} blocks, and complete "
                       "paired seeds"),
                   "slice_stats": record.get("slice_stats"),
                   "readouts": {}}
            for readout in readouts:
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
                        else bool(values["u95"] < margin)),
                    "status": (STATUS_NOT_EVALUABLE if holm_row is None
                               else holm_row["status"]),
                }
            gates.append(row)

    mechanism = []
    for candidate, reference, label in mechanisms:
        key = f"{candidate}-{reference}@{PRIMARY_SLICE}"
        in_family = any(m["candidate"] == candidate
                        and m["reference"] == reference
                        and m["slice"] == PRIMARY_SLICE
                        and m.get("primary")
                        for f in families for m in f["members"])
        row = {
            "contrast_key": key, "candidate": candidate,
            "reference": reference, "registered_label": label,
            "inferential_in_a_registered_family": in_family,
            "guard": MECHANISM_GUARD,
            "record": contrasts.get(key),
        }
        if mechanism_source != MECHANISM_SOURCE_BUILTIN:
            row["registered_in"] = mechanism_source
        mechanism.append(row)

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
            "seeds": [int(s) for s in seeds],
            "n_seeds": len(seeds),
            "readouts": list(readouts),
            "readouts_are_seed_derived": (
                "one estimand (i) readout per registered seed plus the joint "
                "seed-and-block mean; the Holm tables, the gates and the "
                "family screens all iterate this list, so no registered seed "
                "is omitted from any of them"),
            "five_seed_eligibility_rule": FIVE_SEED_ELIGIBILITY_RULE,
            "five_seed_direction_requirement_applies": (
                len(seeds) >= FIVE_SEED_MINIMUM),
            "estimand_ii_label": estimand_ii_label(len(seeds)),
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
        **({"fixed_references": fixed_report} if fixed_report else {}),
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
    readouts = list((payload.get("contract") or {}).get("readouts")
                    or READOUTS)
    n_seeds = int((payload.get("contract") or {}).get("n_seeds")
                  or len(REGISTERED_SEEDS))
    lines.append(f"  readouts     {', '.join(readouts)} "
                 f"({n_seeds} registered seeds)")
    sizes = sorted({len(family["members"])
                    for family in payload["families"]})
    shape = (f"{sizes[0]} members each" if len(sizes) == 1
             else f"{sizes[0]}-{sizes[-1]} members")
    lines.append(f"  families     {len(payload['families'])} "
                 f"({shape})")
    for family in payload["families"]:
        statuses = {readout: family["screen"][readout]["status"]
                    for readout in readouts}
        lines.append(
            f"    {family['candidate']:<18} "
            + "  ".join(f"{readout}={statuses[readout]}"
                        for readout in readouts))
    # The registered five-seed extension qualification, reported explicitly
    # whether or not it applies (Astra gate 2 round 2).
    if n_seeds >= FIVE_SEED_MINIMUM:
        lines.append(f"  extension    {FIVE_SEED_FAVOURABLE_DIRECTIONS}/"
                     f"{FIVE_SEED_MINIMUM} favourable-direction count "
                     "REQUIRED and enforced on every readout")
        for family in payload["families"]:
            screen = family["screen"][readouts[-1]]
            lines.append(
                f"    {family['candidate']:<18} directions "
                f"{screen.get('favourable_direction_count')}/"
                f"{screen.get('n_seeds')} "
                + ("met" if screen.get(
                    "favourable_direction_requirement_met") else "NOT met")
                + ", extension qualified "
                + ("yes" if screen.get("five_seed_extension_qualified")
                   else "no"))
    else:
        lines.append(f"  extension    the {FIVE_SEED_FAVOURABLE_DIRECTIONS}/"
                     f"{FIVE_SEED_MINIMUM} favourable-direction count does "
                     f"NOT apply at {n_seeds} seeds and fails nothing here; "
                     "no screen status below is the five-seed extension "
                     "qualification")
    lines.append(f"  not evaluable {len(payload['not_evaluable'])} contrasts")
    lines.append("  cohort       DEFERRED_UNOPENED (not scored, not read)")
    lines.append("  statuses     " + ", ".join(ALLOWED_STATUSES)
                 + " — no advancement status exists")
    return lines


def require_distinct_out(config: Path, out: Path, default: Path,
                         flag: str = "--out") -> Path:
    """A non-default config may not write to the default output path.

    Astra's operational instruction: the five-seed statistics, k selection,
    report and analysis pin must go to DISTINCT paths from the two-seed ones,
    so night 1's evidence can never be overwritten by a five-seed rerun. The
    choice made here is to REQUIRE the path explicitly rather than to guess a
    new default: any config other than the registered two-seed
    `seq_stage2_v1.yaml` refuses unless the caller names its own output path.
    """
    config, out, default = Path(config), Path(out), Path(default)
    try:
        same_config = config.resolve() == DEFAULT_CONFIG.resolve()
        same_out = out.resolve() == default.resolve()
    except OSError:  # pragma: no cover - resolve() does not touch the disk
        same_config, same_out = config == DEFAULT_CONFIG, out == default
    if same_config or not same_out:
        return out
    suggestion = (default.parent / f"{config.stem}_{default.name}")
    raise RefusalError(
        f"{rel(config)} is not the registered two-seed config "
        f"{rel(DEFAULT_CONFIG)}, and {flag} still points at the two-seed "
        f"default {rel(default)}. Name a distinct path explicitly — for "
        f"example {flag} {rel(suggestion)} — so a five-seed run cannot "
        "overwrite night 1's evidence")


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
    stats.add_argument("--expected-families", type=int, default=None,
                       help="assert a specific registered family count; by "
                            "default it is DERIVED from the config (one "
                            "family per configuration other than the shared "
                            "mlp control), so a seven-family and a "
                            "fifteen-family registration both run with no "
                            "flag")

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
            require_distinct_out(args.config, args.out, DEFAULT_STATS_OUT)
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
        require_distinct_out(args.config, args.out, DEFAULT_KSWEEP_OUT)
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
