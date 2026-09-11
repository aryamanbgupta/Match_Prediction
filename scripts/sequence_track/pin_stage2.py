#!/usr/bin/env python3
"""Write and verify the stage-2 config's `provenance` block (D3 check 3.2).

`experiments/configs/seq_stage2_v1.yaml` is a HAND-WRITTEN registration: the
arms, the seeds, the statistics and the prose are decisions, not
measurements, and nothing here invents them. What this script owns is the
config's `provenance:` block, in the `pin_stage1.py` mould:

* `--write` recomputes every pinned fact from the files on disk and writes
  the block into the config, leaving every other key byte-identical.
* `--verify` recomputes the same block, reads the committed one, and diffs
  the two key by key. Any difference — a re-materialised frame, a rebuilt
  stats cache, an edited stage-2 script, a changed base-logits build, a
  frozen cohort that moved — exits 1 and names the field.

What is recomputed, and why each one is here:

  * the frame: `.feature_hash` key by key, and the md5, row count and
    `match_date` range of the train and validation parquets. A directory name
    and a frame version are not an identity: a same-path parquet replacement
    leaves both unchanged;
  * the stats cache: resolved through the manifest role (`artifacts
    .artifact_path`), hashed, and its `venue_alias_version` compared against
    the frame's, because a cache with different venue identity produces
    different venue features for the same rows;
  * the 50 feature names, in `transformer_t1`'s construction order, and their
    sha256. A permutation has the right length and the right membership and
    is still a different model input, so the order is pinned position by
    position;
  * the base logits: each split's npz md5, and the parquet md5 each npz
    records. That md5 MUST equal the md5 of the split on disk — otherwise the
    residual arms would train against a prior built from data that no longer
    exists. The `train`/`validation` digest is composed exactly as
    `transformer_t1` composes it, so the driver's reuse check and this pin
    speak of one number;
  * the source closure: sha256 of every stage-2 script AND of the imported
    modules they depend on. Pinning the trainer alone would leave a change to
    the recurrent arms, the base-logits builder, the cohort builder, the
    driver or this pin able to move every stage-2 result while `--verify`
    still passed — and pinning only the stage-2 scripts would leave the same
    hole one level down, in the feature contract, the materialiser, the stats
    cache, the identity maps, the ELO update, the chronology and the tracker
    rehydration (Astra SHOULD 3);
  * the cohort: the sha256 of `cohort/FROZEN.json` once it exists. Before the
    freeze the field records `not_yet_frozen`; after it, a changed freeze
    fails `--verify`, which is the point — the untouched cohort is opened
    once (D5.7, D5.9);
  * the config's own body sha256: the config with its `provenance` block
    removed, canonically dumped. A config cannot contain its own hash, so the
    hash is of everything the pin does not write.

Two fields are recorded but never compared, because they change without the
config's meaning changing: `pins_generated_at` and `git_head_short`.

The test split's parquet is HASHED here (the base-logits sidecar for it
records a parquet md5, and a pin that did not check it would let that file
drift) but never opened as data: no column, no row and no label is read.

Usage:
    uv run --no-sync python scripts/sequence_track/pin_stage2.py --write
    uv run --no-sync python scripts/sequence_track/pin_stage2.py --verify
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

CONFIG_PATH = REPO_ROOT / "experiments" / "configs" / "seq_stage2_v1.yaml"
STAGE2_ROOT = REPO_ROOT / "models" / "embeddings" / "seq_stage2"
BASE_LOGITS_DIR = STAGE2_ROOT / "base_logits"
COHORT_FROZEN = STAGE2_ROOT / "cohort" / "FROZEN.json"

# The frame, its version and the stats cache are NOT literals here: they are
# read from the config being pinned (`data.directory`, `data.frame_version`,
# `data.frame_role`, `data.stats_cache_role`) and the directory is
# cross-checked against what the artifact manifest records for the declared
# role, so the pin cannot name a frame the manifest does not.
N_FEATURES = 50
# Splits a stage-2 run reads, in the order the base-logits digest composes.
TRAINING_SPLITS = ("train", "validation")
# The test split is hashed only because a base-logits sidecar names it.
HASHED_SPLITS = ("train", "validation", "test")

# The SOURCE CLOSURE on the stage-2 execution path. A file listed here but
# absent on the checkout is dropped and named in
# `provenance.source_sha256_absent` rather than silently skipped: the later
# deliverables (D7's dependency test, D10's statistics) are listed before
# they are written, so the pin does not have to be edited to admit them.
PROVENANCE_SOURCE_CLOSURE = [
    # stage-2 scripts: the trainer, the recurrent arms, the two builders, the
    # driver, this pin, the dependency certification and the statistics.
    "scripts/transformer_t1.py",
    "scripts/sequence_track/recurrent_arms.py",
    "scripts/sequence_track/build_base_logits.py",
    "scripts/sequence_track/build_cohort_stage2.py",
    "scripts/sequence_track/retrain_stage2.py",
    "scripts/sequence_track/pin_stage2.py",
    "scripts/sequence_track/ownership_dependency_test.py",
    "scripts/sequence_track/stage2_stats.py",
    # D10's report generator and the block/cluster contract the statistics
    # import from. Astra gate 1 round 3 MUST-FIX 4: the closure previously
    # claimed to cover the stage-2 execution path while omitting these, so a
    # change to the bootstrap contract or the renderer moved reported numbers
    # without moving any recorded hash.
    "scripts/sequence_track/render_stage2_report.py",
    "scripts/sim_eval/eval_statistics.py",
    # IMPORTED modules that move stage-2 behaviour without any stage-2 script
    # changing (Astra SHOULD 3). Each one is genuinely on a stage-2 path:
    #   transformer_t1 imports embeddings_e1 (the 50-feature contract: class
    #   mapping, context and EB column lists);
    "scripts/embeddings_e1.py",
    #   build_base_logits imports artifacts (manifest role resolution, the
    #   hashes this pin records) and calibration (the production encoders
    #   applied to the frame before the booster scores it);
    "scripts/artifacts.py",
    "scripts/calibration.py",
    #   retrain_stage2 imports registered_experiment (match ids, the
    #   environment/provenance block in every run record);
    "scripts/registered_experiment.py",
    #   build_cohort_stage2 imports the feature-construction and cache stack,
    #   which decides what the cohort parquet's 114 feature columns ARE:
    "scripts/materialize_features.py",
    "scripts/build_stats_cache.py",
    "scripts/feature_registry.py",
    #   and their own imports, each of which changes those columns or the
    #   chronology they are computed in:
    "scripts/parsing_v2.py",
    "scripts/tracker_rehydration.py",
    "scripts/stats_provider.py",
    "scripts/stats_sqlite_backend.py",
    "scripts/loaders_common.py",
    "scripts/identity_maps.py",
    "scripts/elo_update.py",
    "scripts/player_metadata.py",
]

RUNTIME_DISTRIBUTIONS = ("torch", "numpy", "pandas", "xgboost", "pyarrow")

NOT_VERIFIED = ("provenance.pins_generated_at", "provenance.git_head_short")

PROVENANCE_KEY = "provenance"
COHORT_NOT_FROZEN = "not_yet_frozen"


class PinError(RuntimeError):
    """A pinned fact could not be recomputed from the files on disk."""


# ---------------------------------------------------------------- helpers

def _abs(relative) -> Path:
    path = Path(relative)
    return path if path.is_absolute() else REPO_ROOT / path


def _rel(path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _md5_file(relative) -> str:
    from artifacts import md5_file  # local: keeps import cost off --help

    path = _abs(relative)
    if not path.is_file():
        raise PinError(f"missing file artifact: {_rel(path)}")
    return md5_file(path)


def _sha256_file(relative) -> str:
    path = _abs(relative)
    if not path.is_file():
        raise PinError(f"missing file artifact: {_rel(path)}")
    digester = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digester.update(chunk)
    return digester.hexdigest()


def _git_head_short() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                cwd=REPO_ROOT, capture_output=True, text=True,
                                check=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


# ------------------------------------------------------- recomputed facts

def _data_block(config: dict) -> dict:
    data = config.get("data") or {}
    for field in ("directory", "frame_version", "frame_role",
                  "stats_cache_role"):
        if not data.get(field):
            raise PinError(f"the config declares no data.{field}")
    return data


def frame_block(config: dict) -> dict:
    """The frame declaration and the identity of the parquets on disk."""
    import pyarrow.compute as pc  # noqa: PLC0415 - a second of import cost
    import pyarrow.parquet as pq  # noqa: PLC0415

    from artifacts import artifact_path  # noqa: PLC0415

    import transformer_t1 as t1  # noqa: PLC0415 - imports torch

    data = _data_block(config)
    frame_dir = _abs(data["directory"])
    role_dir = _abs(artifact_path(data["frame_role"]))
    if role_dir.resolve() != frame_dir.resolve():
        raise PinError(
            f"the config's data.directory {data['directory']!r} is not what "
            f"the manifest records for role {data['frame_role']!r} "
            f"({_rel(role_dir)})")
    declared = t1.resolve_frame_version(frame_dir)
    if declared != data["frame_version"]:
        raise PinError(
            f"{_rel(frame_dir)} declares frame version {declared!r}, the "
            f"config registers {data['frame_version']!r}")
    feature_hash = t1.read_feature_hash(frame_dir)
    if not isinstance(feature_hash, dict) or not feature_hash:
        raise PinError(
            f"{_rel(frame_dir)}/.feature_hash is missing or empty")

    splits = {}
    for split in HASHED_SPLITS:
        path = t1.split_path(frame_dir, declared, split)
        if not path.is_file():
            raise PinError(f"missing {split} parquet {_rel(path)}")
        record = {"path": _rel(path), "md5": t1.md5_file(path)}
        if split in TRAINING_SPLITS:
            # Row count and date range are measured for the two splits a
            # stage-2 run reads. The test split is hashed and nothing more.
            table = pq.read_table(path, columns=[t1.DATE_COL])
            if table.num_rows == 0:
                raise PinError(f"{_rel(path)}: no rows")
            bounds = pc.min_max(table.column(t1.DATE_COL)).as_py()
            if bounds.get("min") is None or bounds.get("max") is None:
                raise PinError(
                    f"{_rel(path)}: {t1.DATE_COL} is entirely null, so the "
                    "split's date range cannot be established")
            record.update({"n_rows": int(table.num_rows),
                           "match_date_min": str(bounds["min"]),
                           "match_date_max": str(bounds["max"])})
        else:
            record["read"] = (
                "hashed only: a base-logits sidecar records this parquet's "
                "md5, and no stage-2 step opens its rows")
        splits[split] = record
    return {"dir": _rel(frame_dir), "version": declared,
            "feature_hash": feature_hash, "splits": splits}


def stats_cache_block(role: str, frame_alias_version) -> dict:
    """The stats cache resolved by manifest role, hashed, alias checked."""
    from artifacts import artifact_path  # noqa: PLC0415

    import transformer_t1 as t1  # noqa: PLC0415

    path = _abs(artifact_path(role))
    if not path.is_file():
        raise PinError(
            f"stats cache role {role!r} resolves to "
            f"{_rel(path)}, which does not exist")
    meta = t1.read_cache_meta(path)
    cache_alias = meta.get("venue_alias_version")
    if cache_alias != frame_alias_version:
        raise PinError(
            f"stats cache {_rel(path)} declares venue_alias_version "
            f"{cache_alias!r}, the frame declares {frame_alias_version!r}; "
            "the cache and the frame must share one venue identity")
    return {"role": role, "path": _rel(path),
            "md5": _md5_file(path), "venue_alias_version": cache_alias,
            "same_day_order_version": meta.get("same_day_order_version")}


def feature_block() -> dict:
    """The 50 feature names in construction order, and their sha256."""
    import transformer_t1 as t1  # noqa: PLC0415

    names = list(t1.feature_names())
    if len(names) != N_FEATURES or len(set(names)) != len(names):
        raise PinError(
            f"the T1 feature stack builds {len(names)} names "
            f"({len(set(names))} distinct); stage 2 pins {N_FEATURES}")
    return {"count": len(names), "names": names,
            "sha256": hashlib.sha256("\n".join(names).encode("utf-8")
                                     ).hexdigest(),
            "order": ("EB_BAT + EB_BOWL + VENUE + CTX + STATE "
                      "(embeddings_e1, transformer_t1); a permutation is a "
                      "different model input, not a relabelling")}


def base_logits_block(splits: dict, config: dict | None = None) -> dict:
    """Each split's base-logit npz, checked against the live parquets.

    The residual configurations must name THIS directory: a config pointing
    somewhere else would be pinned against a build it does not use.
    """
    import numpy as np  # noqa: PLC0415

    for entry in (config or {}).get("configurations") or []:
        declared = ((entry.get("params") or {}).get("base_logits_dir"))
        if declared and _abs(declared).resolve() != BASE_LOGITS_DIR.resolve():
            raise PinError(
                f"configuration {entry.get('id')!r} names base logits "
                f"{declared!r}, the pin measures {_rel(BASE_LOGITS_DIR)}")
    if not BASE_LOGITS_DIR.is_dir():
        raise PinError(
            f"{_rel(BASE_LOGITS_DIR)} does not exist; the residual arms have "
            "no base logits to pin (D4)")
    per_split = {}
    for split in HASHED_SPLITS:
        npz = BASE_LOGITS_DIR / f"{split}.npz"
        sidecar = BASE_LOGITS_DIR / f"{split}.json"
        if not npz.is_file():
            raise PinError(f"missing base logits {_rel(npz)}")
        with np.load(npz, allow_pickle=False) as archive:
            for key in ("logp", "n_rows", "parquet_md5"):
                if key not in archive:
                    raise PinError(f"{_rel(npz)} carries no {key!r} key")
            declared_rows = int(np.asarray(archive["n_rows"]).reshape(-1)[0])
            declared_md5 = str(
                np.asarray(archive["parquet_md5"]).reshape(-1)[0])
        live_md5 = splits[split]["md5"]
        if declared_md5 != live_md5:
            raise PinError(
                f"{_rel(npz)} was built from parquet md5 {declared_md5}, but "
                f"the {split} split on disk is md5 {live_md5}; the base "
                "logits are stale and the residual arms must not train "
                "against them")
        live_rows = splits[split].get("n_rows")
        if live_rows is not None and declared_rows != live_rows:
            raise PinError(
                f"{_rel(npz)} declares {declared_rows} rows, the {split} "
                f"split has {live_rows}")
        npz_md5 = _md5_file(npz)
        record = {"npz": _rel(npz), "npz_md5": npz_md5,
                  "n_rows": declared_rows, "parquet_md5": declared_md5}
        if not sidecar.is_file():
            raise PinError(f"missing base logits sidecar {_rel(sidecar)}")
        payload = json.loads(sidecar.read_text())
        if payload.get("npz_md5") != npz_md5:
            raise PinError(
                f"{_rel(sidecar)} records npz_md5 {payload.get('npz_md5')!r}, "
                f"the file on disk is {npz_md5}")
        if payload.get("parquet_md5") != declared_md5:
            raise PinError(
                f"{_rel(sidecar)} records parquet_md5 "
                f"{payload.get('parquet_md5')!r}, the npz records "
                f"{declared_md5}")
        record["sidecar"] = _rel(sidecar)
        record["sidecar_sha256"] = _sha256_file(sidecar)
        record["booster_md5"] = payload.get("booster_md5")
        per_split[split] = record

    # Composed exactly as `transformer_t1` composes it over the splits a
    # stage-2 run reads, so `retrain_stage2`'s reuse check and this pin are
    # comparing one number.
    digest = hashlib.md5(  # noqa: S324 - artifact identity, not a secret
        "\n".join(f"{split}={per_split[split]['npz_md5']}"
                  for split in sorted(TRAINING_SPLITS)).encode("utf-8")
    ).hexdigest()
    return {"dir": _rel(BASE_LOGITS_DIR), "splits": per_split,
            "train_validation_digest": digest,
            "digest_rule": ("md5 of '<split>=<npz md5>' lines, sorted by "
                            "split, over the splits a stage-2 run reads; "
                            "transformer_t1 records the same value in "
                            "arm_params.base_logits_md5")}


def source_block() -> dict:
    """sha256 of every stage-2 script present on this checkout."""
    present = {}
    absent = []
    for source in PROVENANCE_SOURCE_CLOSURE:
        if _abs(source).is_file():
            present[source] = _sha256_file(source)
        else:
            absent.append(source)
    return {"count": len(present), "source_sha256": present,
            "source_sha256_absent": absent,
            "source_sha256_absent_note": (
                "closure files not present on this checkout (a later "
                "deliverable's script); they are dropped from the hash set "
                "and named here rather than skipped silently"),
            "closure_rule": (
                "every source file on the stage-2 execution path is hashed, "
                "not just the trainer: the recurrent arms, the base-logits "
                "builder, the cohort builder, the multi-seed driver and this "
                "pin all move stage-2 results, so a change to any of them "
                "must fail --verify. It also covers the IMPORTED modules "
                "those scripts actually depend on — the 50-feature contract "
                "(embeddings_e1), the manifest and encoder helpers "
                "(artifacts, calibration), the run-provenance helper "
                "(registered_experiment), and the feature-construction, "
                "materialisation, stats-cache, identity, ELO, chronology and "
                "rehydration stack the cohort's 114 columns are built by — "
                "because each of them can change stage-2 behaviour without "
                "any stage-2 script changing (Astra SHOULD 3)")}


def cohort_block(frozen_path: Path = COHORT_FROZEN) -> dict:
    """The frozen cohort's identity, or `not_yet_frozen` before the freeze.

    The cohort is opened exactly once (D5.9), so its freeze must be immutable
    from the moment it exists: pinned here, a later edit to `FROZEN.json`
    fails `--verify` and names this field.
    """
    frozen_path = Path(frozen_path)
    if not frozen_path.is_file():
        return {"frozen": _rel(frozen_path), "status": COHORT_NOT_FROZEN,
                "note": ("the cohort is not frozen yet; re-pin after "
                         "build_cohort_stage2.py writes FROZEN.json, and "
                         "--verify will then refuse any later change to it")}
    payload = json.loads(frozen_path.read_text())
    ids = payload.get("eligible_match_ids")
    return {"frozen": _rel(frozen_path), "status": "frozen",
            "frozen_json_sha256": _sha256_file(frozen_path),
            "eligible_match_count": payload.get("eligible_match_count"),
            "eligible_ids_recorded": (len(ids) if isinstance(ids, list)
                                      else None),
            "frozen_at": payload.get("frozen_at_utc") or payload.get(
                "frozen_at"),
            "scored": "once, in D10, after the family freeze"}


def config_body_sha256(config: dict) -> str:
    """sha256 of the config with its `provenance` block removed.

    A config cannot contain its own hash, so what is pinned is everything the
    pin does not write: the arms, the seeds, the statistics and the prose.
    Canonically dumped (sorted keys) so the hash depends on the content
    rather than on the file's formatting.
    """
    body = {key: value for key, value in config.items()
            if key != PROVENANCE_KEY}
    return hashlib.sha256(
        yaml.safe_dump(body, sort_keys=True, allow_unicode=True)
        .encode("utf-8")).hexdigest()


def runtime_block() -> dict:
    packages = {}
    for name in RUNTIME_DISTRIBUTIONS:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "packages": packages,
            "read_via": "importlib.metadata.version",
            "note": ("torch and numpy change training output, so a stage-2 "
                     "result produced under different versions is a "
                     "different result")}


def build_provenance(config: dict, *, frozen_path: Path = COHORT_FROZEN
                     ) -> dict:
    """The whole `provenance` block, recomputed from the files on disk."""
    frame = frame_block(config)
    return {
        "pins_generated_at": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "git_head_short": _git_head_short(),
        "not_verified": list(NOT_VERIFIED),
        "not_verified_reason": (
            "both change without the config's meaning changing (a re-pin "
            "moves the timestamp; the commit that lands this file moves "
            "HEAD), so --verify records them and does not compare them"),
        "pinned_by": "scripts/sequence_track/pin_stage2.py",
        "config_body_sha256": config_body_sha256(config),
        "config_body_sha256_rule": (
            "sha256 of yaml.safe_dump(config without 'provenance', "
            "sort_keys=True); a config cannot carry its own hash, so what is "
            "pinned is every key the pin does not write"),
        "frame": frame,
        "stats_cache": stats_cache_block(
            _data_block(config)["stats_cache_role"],
            frame["feature_hash"].get("venue_alias_version")),
        "features": feature_block(),
        "base_logits": base_logits_block(frame["splits"], config),
        "sources": source_block(),
        "cohort": cohort_block(frozen_path),
        "runtime": runtime_block(),
    }


# --------------------------------------------------------- write / verify

def load_config(path: Path = CONFIG_PATH) -> dict:
    if not Path(path).is_file():
        raise PinError(f"{_rel(path)} does not exist")
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict):
        raise PinError(f"{_rel(path)}: top level is not a mapping")
    return config


def flatten(value, prefix: str = "") -> dict:
    """`{'a.b[0].c': leaf}` for a nested mapping/list payload.

    An EMPTY container becomes its own leaf, so a structural change (an empty
    list becoming an empty mapping, or the key disappearing) cannot slip past
    `--verify`.
    """
    out = {}
    if isinstance(value, dict):
        if not value:
            out[prefix] = "<empty mapping>"
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten(item, child))
    elif isinstance(value, list):
        if not value:
            out[prefix] = "<empty list>"
        for index, item in enumerate(value):
            out.update(flatten(item, f"{prefix}[{index}]"))
    else:
        out[prefix] = value
    return out


def diff_provenance(expected: dict, actual) -> list[str]:
    """Human-readable differences, excluding the not-verified keys."""
    flat_expected = flatten(expected, PROVENANCE_KEY)
    flat_actual = flatten(actual, PROVENANCE_KEY)
    for key in NOT_VERIFIED:
        flat_expected.pop(key, None)
        flat_actual.pop(key, None)
    problems = []
    for key in sorted(set(flat_expected) - set(flat_actual)):
        problems.append(f"MISSING  {key}: expected {flat_expected[key]!r}")
    for key in sorted(set(flat_actual) - set(flat_expected)):
        problems.append(f"EXTRA    {key}: found {flat_actual[key]!r}")
    for key in sorted(set(flat_expected) & set(flat_actual)):
        if flat_expected[key] != flat_actual[key]:
            problems.append(
                f"CHANGED  {key}: expected {flat_expected[key]!r}, "
                f"found {flat_actual[key]!r}")
    return problems


def write_provenance(path: Path = CONFIG_PATH) -> dict:
    """Recompute the block and write it into the config, nothing else.

    The config's other keys are dumped back from the parsed document, so the
    pin cannot half-edit prose; only `provenance` is authored here.
    """
    config = load_config(path)
    config.pop(PROVENANCE_KEY, None)
    provenance = build_provenance(config)
    config[PROVENANCE_KEY] = provenance
    text = Path(path).read_text()
    body, _, _ = text.partition(f"\n{PROVENANCE_KEY}:\n")
    # The generated block's own comment header is part of the block, so strip
    # any copy already in the body before writing a fresh one; otherwise every
    # --write leaves another header behind (five had accumulated by 2026-09-12).
    while body.rstrip("\n").endswith(HEADER_PROVENANCE.rstrip("\n")):
        body = body.rstrip("\n")[:-len(HEADER_PROVENANCE.rstrip("\n"))]
    block = yaml.safe_dump({PROVENANCE_KEY: provenance}, sort_keys=False,
                           default_flow_style=False, width=78,
                           allow_unicode=True)
    Path(path).write_text(body.rstrip("\n") + "\n\n" + HEADER_PROVENANCE
                          + block)
    return provenance


HEADER_PROVENANCE = """\
# --------------------------------------------------------------------------
# GENERATED BLOCK. Written by `scripts/sequence_track/pin_stage2.py --write`
# and checked by `--verify`, which recomputes every hash from the files on
# disk and exits non-zero on any difference. Do not edit by hand.
# --------------------------------------------------------------------------
"""


def verify(path: Path = CONFIG_PATH) -> list[str]:
    config = load_config(path)
    actual = config.get(PROVENANCE_KEY)
    if actual is None:
        return [f"MISSING  {PROVENANCE_KEY}: {_rel(path)} carries no "
                "provenance block; run pin_stage2.py --write"]
    body = copy.deepcopy(config)
    body.pop(PROVENANCE_KEY, None)
    return diff_provenance(build_provenance(body), actual)


def print_summary(provenance: dict) -> None:
    frame = provenance["frame"]
    print(f"frame {frame['dir']} ({frame['version']})")
    for split, record in frame["splits"].items():
        rows = record.get("n_rows")
        print(f"  {split:<11} md5 {record['md5']}"
              + (f" rows {rows} {record['match_date_min']}.."
                 f"{record['match_date_max']}" if rows is not None
                 else "  (hashed only)"))
    cache = provenance["stats_cache"]
    print(f"stats cache {cache['role']} md5 {cache['md5']} "
          f"({cache['venue_alias_version']})")
    print(f"features {provenance['features']['count']} sha "
          f"{provenance['features']['sha256'][:16]}...")
    base = provenance["base_logits"]
    print(f"base logits {base['dir']} digest "
          f"{base['train_validation_digest']}")
    print(f"sources {provenance['sources']['count']} hashed"
          + (f", absent {provenance['sources']['source_sha256_absent']}"
             if provenance["sources"]["source_sha256_absent"] else ""))
    cohort = provenance["cohort"]
    print(f"cohort {cohort['status']}"
          + (f" sha {cohort['frozen_json_sha256'][:16]}... "
             f"({cohort['eligible_match_count']} matches)"
             if cohort["status"] == "frozen" else ""))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true",
                       help="recompute the provenance block and write it")
    group.add_argument("--verify", action="store_true",
                       help="recompute every pinned fact and diff the file")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args(argv)

    try:
        if args.write:
            provenance = write_provenance(args.config)
            print_summary(provenance)
            print(f"wrote {_rel(args.config)} provenance block")
            return 0
        problems = verify(args.config)
    except PinError as exc:
        print(f"pin_stage2: ERROR: {exc}", file=sys.stderr)
        return 1
    if problems:
        print(f"pin_stage2: {len(problems)} mismatch(es) in "
              f"{_rel(args.config)}:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print_summary(load_config(args.config)[PROVENANCE_KEY])
    print("not compared (recorded only): " + ", ".join(NOT_VERIFIED))
    print(f"pin_stage2: OK — {_rel(args.config)} matches every recomputed "
          "fact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
