#!/usr/bin/env python3
"""Retrain the stage-1 sequence arms B (mlp) and C (full) on the i7 frame.

Sequence track stage 1, deliverable D3 (checks 3.7, 3.10). The runner is a
launcher and a bookkeeper, never a second implementation of training: every
number it reports is read back out of the checkpoint `metrics.json` that
`scripts/transformer_t1.py` wrote, and every identity field it reports comes
from that file's `training_contract` block.

What it guarantees, and why each one is here:

  * exactly the five registered seeds, each once. A duplicate or an unknown
    seed in the config is refused, so a "five seed" table can never be four
    seeds and a repeat;
  * the test split is never loaded. `--no-kit` is passed and `--score-test`
    is not, so selection happens on validation rows only;
  * the stats cache resolves through the manifest role and its venue alias
    version must equal the frame's, checked once before the first run and
    again inside every run;
  * a completed checkpoint is never overwritten. It is reused only after its
    contract is re-verified against the EFFECTIVE settings of THIS
    invocation — arm, seed, frame dir and version, stats-cache role and md5,
    kit and test-split flags, the architecture (dmodel, layers, heads), the
    optimiser (lr, batch, epochs, patience, aux, aux_weight) and the CLI
    overrides recorded in its `run_record.json`. Any mismatch refuses and
    names the field; `--force` retrains it instead. Without the architecture,
    optimiser and override checks a completed `--epochs 1` smoke run sitting
    in the registered output root would be reused by a default invocation and
    reported in `summary.yaml` under the default config hash;
  * the frame is MEASURED once, in preflight: the whole `.feature_hash`
    declaration, and the md5, row count and `match_date` range of the train
    and validation parquets (the test split is not among them and is never
    opened). Every reused checkpoint's contract is compared against that
    measurement key by key, and `summary.yaml`'s `frame` block is that
    measurement rather than a copy taken from a checkpoint. A directory name
    and a frame version are not an identity: a same-path parquet replacement,
    or an edited declaration field such as `k_player`, leaves both unchanged,
    and without this check the runner would silently reuse checkpoints
    trained on data that no longer exists and publish a table combining this
    invocation's declaration with an older run's facts;
  * wall seconds per run are recorded next to each checkpoint, so a resumed
    invocation still reports the time the run actually took.

Usage:
    uv run --no-sync python scripts/sequence_track/retrain_i7.py --dry-run
    uv run --no-sync python scripts/sequence_track/retrain_i7.py
    uv run --no-sync python scripts/sequence_track/retrain_i7.py \
        --only-arm mlp --only-seed 7
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import transformer_t1 as t1  # noqa: E402
from registered_experiment import match_ids  # noqa: E402

DEFAULT_CONFIG = REPO / "experiments/configs/seq_stage1_retrain_i7_v1.yaml"
TRAINER = REPO / "scripts" / "transformer_t1.py"
SEQ_STAGE1_ROOT = REPO / "models" / "embeddings" / "seq_stage1"
REGISTERED_SEEDS = (7, 13, 29, 42, 101)
KNOWN_ARMS = ("full", "mlp", "no_attention", "no_history")
SELECTION_SPLIT = "validation"
N_MATCHES_SOURCE = (
    "distinct match ids, the leading '<innings>_' prefix stripped from "
    "innings_id (registered_experiment.match_ids)")
# The only two splits a stage-1 run may read, and the fields that identify
# each one. `path` is measured and reported but deliberately NOT compared:
# it is how the file was spelled on a command line, not what is in it.
CONTRACT_SPLITS = ("train", SELECTION_SPLIT)
SPLIT_IDENTITY_FIELDS = ("md5", "n_rows", "match_date_min", "match_date_max")


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
    """Stage-1 output is confined to the stage-1 embeddings namespace."""
    allowed = SEQ_STAGE1_ROOT.resolve()
    resolved = Path(out_root).resolve()
    if resolved != allowed and allowed not in resolved.parents:
        raise RetrainError(
            f"refusing to write outside {rel(allowed)}: {rel(resolved)}")
    return resolved


def load_config(path: Path) -> dict:
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict):
        raise RetrainError(f"{rel(path)}: config is not a mapping")

    arms = config.get("arms")
    if not isinstance(arms, list) or not arms:
        raise RetrainError(f"{rel(path)}: 'arms' must be a non-empty list")
    if len(set(arms)) != len(arms):
        raise RetrainError(f"{rel(path)}: duplicate arm in {arms}")
    unknown = [arm for arm in arms if arm not in KNOWN_ARMS]
    if unknown:
        raise RetrainError(f"{rel(path)}: unknown arm(s) {unknown}")

    training = config.get("training") or {}
    seeds = training.get("seeds")
    if not isinstance(seeds, list):
        raise RetrainError(f"{rel(path)}: 'training.seeds' must be a list")
    if len(set(seeds)) != len(seeds):
        raise RetrainError(
            f"{rel(path)}: duplicate seed in {seeds}; each registered seed "
            "runs exactly once")
    if tuple(seeds) != REGISTERED_SEEDS:
        raise RetrainError(
            f"{rel(path)}: seeds {seeds} are not the registered five "
            f"{list(REGISTERED_SEEDS)}; changing them changes the protocol")
    if training.get("no_kit") is not True:
        raise RetrainError(f"{rel(path)}: 'training.no_kit' must be true")
    if training.get("score_test") is not False:
        raise RetrainError(
            f"{rel(path)}: 'training.score_test' must be false; the retrain "
            "never loads the test split")
    if training.get("aux") is not False:
        raise RetrainError(f"{rel(path)}: 'training.aux' must be false")
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


def effective_settings(config: dict, epochs: int | None = None,
                       overrides: dict | None = None) -> dict:
    """What this invocation would actually train, config plus CLI overrides.

    Reuse and reporting both compare a checkpoint's `training_contract`
    against THIS, never against the config alone: `--epochs` changes what a
    run is, and a run that was launched with an override is a different run
    even though the config file is byte-identical.

    `aux_weight` is never passed on the command line, so the effective value
    is the trainer's own default unless the config names one; it is compared
    because the contract records it.
    """
    training = config["training"]
    return {
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
        "overrides": dict(overrides or {}),
    }


def frame_split_facts(frame_dir: Path, version: str) -> dict:
    """Identity of the two parquets a stage-1 run is allowed to read.

    `{split: {path, md5, n_rows, match_date_min, match_date_max}}` for train
    and validation only. The test split has no entry here and is not opened:
    the retrain's selection must not be able to see a test row, and a
    measurement is a read.

    The field set and the way the dates are rendered match
    `pin_stage1.frame_split_facts` on purpose — the pin later compares its
    own measurement against the same contract fields, so the two must mean
    the same thing. That helper is not imported because it memoises per
    (directory, version) for the life of the process, and these facts have
    to describe the frame as it is at THIS preflight, not as it was the
    first time something in the process asked.
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


def preflight(config: dict) -> dict:
    """Resolve and MEASURE the frame and the stats cache, before any run.

    The measurement — the declaration plus the train/validation parquet
    identity — is taken exactly once and is what every checkpoint is then
    verified against and what the summary reports, so ten reused checkpoints
    and one summary all describe one frame, read at one moment.
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
    # costs seconds instead of ten launches.
    cache = t1.resolve_stats_cache(data["stats_cache_role"], None,
                                   feature_hash.get("venue_alias_version"))
    return {"frame_dir": frame_dir, "frame_version": declared,
            "feature_hash": feature_hash,
            "split_files": frame_split_facts(frame_dir, declared),
            "stats_cache": cache}


# -------------------------------------------------------------- commands

def out_dir_for(out_root: Path, arm: str, seed: int) -> Path:
    return Path(out_root) / arm / f"seed_{seed}"


def command_for(config: dict, arm: str, seed: int, out_dir: Path,
                effective: dict) -> list[str]:
    """The exact `transformer_t1.py` argv this (arm, seed) will run.

    Every trainer flag is taken from `effective`, so the argv and the reuse
    check read the same numbers and cannot drift apart. `--no-kit` is always
    present and `--score-test` never is: the retrain's selection must not be
    able to see a test row. `--aux` is never emitted — `load_config` refuses
    a config with `training.aux` true.
    """
    training = config["training"]
    data = config["data"]
    arch, opt = effective["arch"], effective["optimiser"]
    return [
        sys.executable, str(TRAINER),
        "--arm", str(arm),
        "--seed", str(int(seed)),
        "--data-dir", rel(REPO / data["directory"]),
        "--dmodel", str(arch["dmodel"]),
        "--layers", str(arch["layers"]),
        "--heads", str(arch["heads"]),
        "--batch", str(opt["batch"]),
        "--epochs", str(opt["epochs"]),
        "--lr", repr(opt["lr"]),
        "--patience", str(opt["patience"]),
        "--device", str(training["device"]),
        "--no-kit",
        "--stats-cache-role", str(data["stats_cache_role"]),
        "--out", rel(out_dir),
    ]


# ------------------------------------------------- checkpoint bookkeeping

def is_complete(out_dir: Path) -> bool:
    return ((out_dir / "metrics.json").exists()
            and (out_dir / "model.pt").exists())


def read_metrics(out_dir: Path) -> dict:
    return json.loads((out_dir / "metrics.json").read_text())


def as_n_rows(value):
    """A row count as an int when it is a number, and unchanged otherwise.

    Kept out of the comparison's way so that `48.0` and `48` are the same
    count while `"48"` and a missing field still show up as what they are.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    return int(value)


def frame_problems(contract: dict, resolved: dict) -> list[str]:
    """Every disagreement between a checkpoint and the frame on disk NOW.

    The declaration is compared key by key rather than as a whole, and the
    train/validation parquets by md5, row count and date range, so the
    refusal names the one field that moved. `path` is not compared: the same
    file addressed as `data/xgb_data_i7/...` and as an absolute path is the
    same file, and the md5 is what settles that question.
    """
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


def verify_checkpoint(out_dir: Path, metrics: dict, config: dict,
                      resolved: dict, arm: str, seed: int,
                      effective: dict) -> dict:
    """Re-verify a checkpoint before reusing or reporting it.

    Reuse on file existence alone would let this summary describe a
    checkpoint trained on another frame, another cache, another arm — or, the
    case this check exists for, the same arm and seed trained under different
    architecture, optimiser or override settings (a `--epochs 1` smoke run
    reused by a default invocation and reported under the default config).
    Every field below is compared against `effective`, the config plus this
    invocation's CLI overrides, and against `resolved`, the frame as
    preflight measured it.

    The frame comparison is by CONTENT, not by name: a directory path and a
    frame version survive a same-path parquet replacement and an edited
    declaration field untouched, so those two alone would let a checkpoint
    trained on data that no longer exists be reused and reported.
    """
    contract = metrics.get("training_contract")
    if not contract:
        raise RetrainError(
            f"{rel(out_dir)}: metrics.json carries no training_contract; "
            "retrain it with --force")
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
    if contract.get("frame_version") != resolved["frame_version"]:
        problems.append(
            f"frame_version {contract.get('frame_version')!r} != "
            f"{resolved['frame_version']!r}")
    recorded_frame = Path(contract.get("frame_dir", ""))
    if not recorded_frame.is_absolute():
        recorded_frame = REPO / recorded_frame
    if rel(recorded_frame) != rel(resolved["frame_dir"]):
        problems.append(
            f"frame_dir {contract.get('frame_dir')!r} is not "
            f"{rel(resolved['frame_dir'])!r}")
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

    # Architecture and optimiser: a checkpoint of the right arm and seed can
    # still be the wrong model (a different width) or the wrong run (a
    # different epoch budget). The contract records both; compare both.
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

    # The overrides the checkpoint was launched under. The contract cannot
    # carry these (the trainer never sees them), so the runner's own
    # `run_record.json` is the only witness; without it the provenance of a
    # reused checkpoint is unverifiable and reuse is refused.
    record = read_run_record(out_dir)
    if record is None:
        problems.append(
            f"{rel(run_record_path(out_dir))} is absent or unreadable, so "
            "the overrides this checkpoint was trained under cannot be "
            "verified")
    elif record.get("overrides") != effective["overrides"]:
        problems.append(
            f"run_record.json overrides {record.get('overrides')!r} != this "
            f"invocation's {effective['overrides']!r}")

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


# ----------------------------------------------------------------- summary

def validation_match_count(frame_dir: Path, version: str) -> tuple[int, int]:
    """(rows, distinct matches) of the split selection reads. Test is not
    touched: this reads the validation parquet only."""
    path = t1.split_path(frame_dir, version, SELECTION_SPLIT)
    innings = pd.read_parquet(path, columns=["innings_id"])["innings_id"]
    return int(len(innings)), int(
        len(set(match_ids(innings.astype(str).to_numpy()).tolist())))


def build_summary(config: dict, config_path: Path, resolved: dict,
                  rows: dict, overrides: dict, out_root: Path) -> dict:
    arms = config["arms"]
    seeds = config["training"]["seeds"]
    recorded = sum(len(rows.get(arm, [])) for arm in arms)
    expected = len(arms) * len(seeds)

    # The frame block is preflight's measurement of the parquets on disk,
    # never a copy taken from a checkpoint — otherwise a summary could pair
    # this invocation's declaration with facts recorded by an older run on a
    # file that has since been replaced. Every reused checkpoint must agree
    # with that measurement field by field, which `verify_checkpoint` has
    # already required and which is re-asserted here because THIS is the
    # function that publishes the numbers.
    live_files = resolved["split_files"]
    split_md5s = {split: facts["md5"] for split, facts in live_files.items()}
    n_rows = {split: int(facts["n_rows"])
              for split, facts in live_files.items()}
    for arm in arms:
        for row in rows.get(arm, []):
            for split, record in (row["_split_files"] or {}).items():
                live = live_files.get(split)
                if live is None:
                    raise RetrainError(
                        f"{row['checkpoint_dir']}: trained on a {split!r} "
                        "split the measured frame does not have; the summary "
                        "would describe a frame that is not on disk")
                for field in SPLIT_IDENTITY_FIELDS:
                    got = record.get(field, ABSENT)
                    if field == "n_rows":
                        got = as_n_rows(got)
                    if got != live[field]:
                        raise RetrainError(
                            f"{row['checkpoint_dir']}: split_files.{split}."
                            f"{field} {got!r} != the measured frame's "
                            f"{live[field]!r}")

    if recorded:
        rows_count, n_matches = validation_match_count(
            resolved["frame_dir"], resolved["frame_version"])
        if n_rows.get(SELECTION_SPLIT) not in (None, rows_count):
            raise RetrainError(
                f"{SELECTION_SPLIT} parquet holds {rows_count} rows by "
                f"innings_id but {n_rows[SELECTION_SPLIT]} by the preflight "
                "measurement; the file changed underneath this run")
    else:
        rows_count, n_matches = None, None

    arm_block = {}
    for arm in arms:
        # `overrides` is per-seed provenance, not decoration: a row whose
        # checkpoint was launched under an override is not describable by the
        # config hash alone, and the reuse check refuses to mix the two.
        arm_block[arm] = {"per_seed": [
            {key: row[key] for key in ("seed", "ll", "best_epoch",
                                       "wall_seconds", "checkpoint_md5",
                                       "checkpoint_dir", "overrides")}
            for row in rows.get(arm, [])]}

    return {
        "experiment": {
            "name": config["experiment"]["name"],
            "config": rel(config_path),
            "config_sha256": t1.sha256_text(Path(config_path).read_text()),
            "runner": rel(Path(__file__)),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "git_head_short": t1.git_head_short(),
            "runs_expected": expected,
            "runs_recorded": recorded,
            "complete": recorded == expected,
            "overrides": overrides,
            "test_split_scored": False,
        },
        "frame": {
            "dir": rel(resolved["frame_dir"]),
            "version": resolved["frame_version"],
            "feature_hash": resolved["feature_hash"],
            "split_md5s": split_md5s,
            "split_rows": n_rows,
            "stats_cache": resolved["stats_cache"],
        },
        "splits": {
            SELECTION_SPLIT: {
                "n_rows": rows_count,
                "n_matches": n_matches,
                "n_matches_source": N_MATCHES_SOURCE,
                "arms": arm_block,
            },
        },
        "selection": config.get("selection", {}),
        "output_root": rel(out_root),
    }


def print_table(summary: dict) -> None:
    block = summary["splits"][SELECTION_SPLIT]
    print(f"\nvalidation rows {block['n_rows']} / matches "
          f"{block['n_matches']}")
    for arm, payload in block["arms"].items():
        print(f"arm {arm}")
        for row in payload["per_seed"]:
            wall = ("        -" if row["wall_seconds"] is None
                    else f"{row['wall_seconds']:9.1f}s")
            print(f"  seed {row['seed']:>3}  validation LL "
                  f"{row['ll']!r:<20} best epoch {row['best_epoch']:>3}  "
                  f"wall {wall}")


# -------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-root", type=Path, default=None,
                        help="override outputs.directory (must stay under "
                             "models/embeddings/seq_stage1)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override training.epochs; recorded in "
                             "summary.yaml as an override")
    parser.add_argument("--only-arm", default=None)
    parser.add_argument("--only-seed", type=int, default=None)
    parser.add_argument("--force", action="store_true",
                        help="retrain runs whose checkpoints are complete")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the commands and exit without training")
    args = parser.parse_args()

    config = load_config(args.config)
    out_root = assert_writable(
        args.out_root if args.out_root is not None
        else REPO / config["outputs"]["directory"])
    epochs = (args.epochs if args.epochs is not None
              else int(config["training"]["epochs"]))
    # An override is a DEVIATION from the registered config, not merely a
    # flag that was typed: `--out-root <the configured directory>` and
    # `--epochs <the configured epochs>` describe the registered run, and
    # recording them would make a resumed invocation disagree with the
    # run records its own earlier invocation wrote.
    overrides = {}
    if args.out_root is not None and rel(out_root) != config["outputs"][
            "directory"]:
        overrides["out_root"] = rel(out_root)
    if args.epochs is not None and epochs != int(config["training"]["epochs"]):
        overrides["epochs"] = epochs
    effective = effective_settings(config, epochs, overrides)

    arms = list(config["arms"])
    seeds = list(config["training"]["seeds"])
    if args.only_arm is not None:
        if args.only_arm not in arms:
            raise SystemExit(
                f"--only-arm {args.only_arm!r} is not in {arms}")
        arms_to_run = [args.only_arm]
    else:
        arms_to_run = list(arms)
    if args.only_seed is not None:
        if args.only_seed not in seeds:
            raise SystemExit(
                f"--only-seed {args.only_seed} is not in {seeds}")
        seeds_to_run = [args.only_seed]
    else:
        seeds_to_run = list(seeds)

    if args.dry_run:
        for arm in arms_to_run:
            for seed in seeds_to_run:
                print(shlex.join(command_for(
                    config, arm, seed, out_dir_for(out_root, arm, seed),
                    effective)))
        return 0

    resolved = preflight(config)
    print(f"frame {rel(resolved['frame_dir'])} ({resolved['frame_version']}), "
          f"stats cache {resolved['stats_cache']['role']} md5 "
          f"{resolved['stats_cache']['md5']}", flush=True)
    # What every checkpoint is about to be verified against, printed so the
    # operator sees the frame this invocation measured, not just its name.
    for split, facts in resolved["split_files"].items():
        print(f"  {split} md5 {facts['md5']} rows {facts['n_rows']} "
              f"{facts['match_date_min']}..{facts['match_date_max']}",
              flush=True)

    # Everything already complete under this root must describe the same
    # invocation as the one about to start, checked BEFORE the first launch.
    # The summary reports every complete run under the root, not only the
    # ones launched here, so discovering the mismatch afterwards would mean
    # hours of training followed by a refusal to write any table at all.
    # Runs this invocation is about to replace are exempt.
    for arm in arms:
        for seed in seeds:
            out_dir = out_dir_for(out_root, arm, seed)
            if not is_complete(out_dir):
                continue
            if args.force and arm in arms_to_run and seed in seeds_to_run:
                continue
            verify_checkpoint(out_dir, read_metrics(out_dir), config,
                              resolved, arm, seed, effective)

    for arm in arms_to_run:
        for seed in seeds_to_run:
            out_dir = out_dir_for(out_root, arm, seed)
            if is_complete(out_dir) and not args.force:
                verify_checkpoint(out_dir, read_metrics(out_dir), config,
                                  resolved, arm, seed, effective)
                print(f"{arm} seed {seed}: complete; not overwriting "
                      f"({rel(out_dir)}); --force retrains", flush=True)
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            command = command_for(config, arm, seed, out_dir, effective)
            print(f"\n{arm} seed {seed}: {shlex.join(command)}", flush=True)
            started = datetime.now(timezone.utc)
            clock = time.time()
            done = subprocess.run(command, cwd=REPO)
            wall = time.time() - clock
            if done.returncode != 0:
                raise SystemExit(
                    f"{arm} seed {seed}: transformer_t1.py exited "
                    f"{done.returncode} after {wall:.1f}s")
            write_run_record(out_dir, {
                "arm": arm, "seed": seed,
                "command": command,
                "wall_seconds": wall,
                "started_at": started.isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "config": rel(args.config),
                "config_sha256": t1.sha256_text(Path(args.config).read_text()),
                "overrides": overrides,
            })
            print(f"{arm} seed {seed}: {wall:.1f}s", flush=True)

    # Report every complete run under this root, not only the ones this
    # invocation launched, so a resumed job still writes one whole table.
    rows: dict[str, list] = {}
    for arm in arms:
        for seed in seeds:
            out_dir = out_dir_for(out_root, arm, seed)
            if not is_complete(out_dir):
                continue
            metrics = read_metrics(out_dir)
            contract = verify_checkpoint(out_dir, metrics, config, resolved,
                                         arm, seed, effective)
            record = read_run_record(out_dir)
            rows.setdefault(arm, []).append({
                "seed": int(seed),
                "ll": float(metrics[f"{SELECTION_SPLIT}_ll"]),
                "best_epoch": contract["best_epoch"],
                "overrides": dict((record or {}).get("overrides") or {}),
                "wall_seconds": read_wall_seconds(record),
                "checkpoint_md5": t1.md5_file(out_dir / "model.pt"),
                "checkpoint_dir": rel(out_dir),
                "_split_files": contract["split_files"],
            })

    summary = build_summary(config, args.config, resolved, rows, overrides,
                            out_root)
    summary_path = out_root / (config["outputs"].get("summary")
                               or "summary.yaml")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False))
    print_table(summary)
    print(f"\nwrote {rel(summary_path)} "
          f"({summary['experiment']['runs_recorded']}/"
          f"{summary['experiment']['runs_expected']} runs recorded)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RetrainError as error:
        print(f"retrain_i7: {error}", file=sys.stderr)
        raise SystemExit(2) from None
