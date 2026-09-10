#!/usr/bin/env python3
"""Paired, resumable multi-seed match-model experiment harness."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from artifacts import load_manifest, md5_directory, verify_role  # noqa: E402
from sim_eval import blend_eval_json, claim_gate, reslice_eval_json  # noqa: E402
from sim_eval.market_math import CostModel  # noqa: E402
from xgboost_match_v1 import _SWAP_NEGATE, _swap_augment_train  # noqa: E402

DEFAULT_SEEDS = [29, 7, 13, 42, 101]
DEFAULT_SLICES = ["all", 50_000, 100_000]
MANIFEST_PATH = ROOT / "models" / "MANIFEST.yaml"
VERIFY_NAMES = {"diff_negation_identity", "h2h_prior_half", "swap_doubling"}


class ConfigError(ValueError):
    """A named harness configuration field is invalid."""


@dataclass(frozen=True)
class HarnessConfig:
    raw: dict[str, Any]
    name: str
    frame: Path
    baseline_args: list[str]
    candidate_kind: str
    candidate_args: list[str]
    seeds: list[int]
    slices: list[str | int]
    cost: CostModel
    expected_minutes_per_seed: float
    ceiling_minutes: float
    odds_role: str
    folds: list[dict[str, str]]
    verify: list[str]
    out_dir: Path
    config_hash: str


def _error(field: str, message: str) -> ConfigError:
    return ConfigError(f"{field}: {message}")


def _trainer_args(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list) and all(isinstance(item, (str, int, float)) for item in value):
        return [str(item) for item in value]
    if isinstance(value, dict):
        result: list[str] = []
        for key, item in value.items():
            flag = str(key)
            if not flag.startswith("--"):
                flag = "--" + flag.replace("_", "-")
            if item is True:
                result.append(flag)
            elif item not in (False, None):
                result.extend([flag, str(item)])
        return result
    raise _error(field, "must be a list of CLI tokens or a mapping")


def _manifest_root() -> Path:
    return (MANIFEST_PATH.parent.parent if MANIFEST_PATH.parent.name == "models"
            else MANIFEST_PATH.parent)


def _resolve_frame(value: Any, field: str) -> tuple[Path, str]:
    if not isinstance(value, str) or not value:
        raise _error(field, "must be a manifest role or directory")
    entries = load_manifest(MANIFEST_PATH)
    if value in entries:
        status, actual = verify_role(value, manifest_path=MANIFEST_PATH)
        if status != "OK" or actual is None:
            raise _error(field, f"manifest role {value!r} verification {status}")
        candidate = Path(entries[value]["path"])
        if not candidate.is_absolute():
            candidate = _manifest_root() / candidate
        return candidate, actual
    candidate = Path(value)
    if candidate.is_absolute():
        if not candidate.is_dir():
            raise _error(field, f"directory does not exist: {candidate}")
        return candidate, md5_directory(candidate)
    if (ROOT / candidate).is_dir():
        resolved = ROOT / candidate
        return resolved, md5_directory(resolved)
    raise _error(field, f"unknown role or missing directory {value!r}")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_argv(argv: Iterable[str]) -> str:
    encoded = json.dumps(list(argv), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _canonical_hash(payload: dict[str, Any], frame: Path, frame_hash: str) -> str:
    canonical = dict(payload)
    canonical.pop("out_dir", None)
    digest = hashlib.sha256()
    digest.update(yaml.safe_dump(canonical, sort_keys=True).encode("utf-8"))
    digest.update(b"\0frame_path\0")
    digest.update(str(frame.resolve()).encode("utf-8"))
    digest.update(b"\0frame_hash\0")
    digest.update(frame_hash.encode("ascii"))
    digest.update(b"\0trainer_sha256\0")
    digest.update(_sha256_file(SCRIPTS / "xgboost_match_v1.py").encode("ascii"))
    return digest.hexdigest()


def load_config(path: Path) -> HarnessConfig:
    try:
        raw = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"config: {exc}") from exc
    if not isinstance(raw, dict):
        raise _error("config", "top level must be a mapping")
    for field in ("name", "frame", "baseline", "candidate",
                  "expected_minutes_per_seed", "out_dir"):
        if field not in raw:
            raise _error(field, "is required")
    name = raw["name"]
    if not isinstance(name, str) or not name.strip():
        raise _error("name", "must be a non-empty string")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise _error("name", "must contain only letters, numbers, '.', '_' or '-'")
    frame, frame_hash = _resolve_frame(raw["frame"], "frame")
    baseline = raw["baseline"]
    candidate = raw["candidate"]
    if not isinstance(baseline, dict):
        raise _error("baseline", "must be a mapping")
    if not isinstance(candidate, dict):
        raise _error("candidate", "must be a mapping")
    if "kind" not in candidate:
        raise _error("candidate.kind", "is required")
    if "trainer_args" not in candidate:
        raise _error("candidate.trainer_args", "is required")
    kind = candidate["kind"]
    if kind not in {"trainer", "ensemble"}:
        raise _error("candidate.kind", "must be trainer or ensemble")
    if kind == "trainer" and "trainer_args" not in baseline:
        raise _error("baseline.trainer_args", "is required")
    baseline_args = _trainer_args(baseline.get("trainer_args"), "baseline.trainer_args")
    candidate_args = _trainer_args(candidate.get("trainer_args"), "candidate.trainer_args")
    forbidden = {
        "--cmd", "--seed", "--model-dir", "--data-dir",
        "--fit-encoders-on", "--config-json", "--early-stop-before",
    }
    for field, values in (("baseline.trainer_args", baseline_args),
                          ("candidate.trainer_args", candidate_args)):
        overlap = {
            value for value in values
            if value.startswith("--")
            and any(flag.startswith(value.split("=", 1)[0]) for flag in forbidden)
        }
        if overlap:
            raise _error(field, f"harness-owned argument(s): {sorted(overlap)}")
    seed_value = raw.get("seeds", DEFAULT_SEEDS)
    if isinstance(seed_value, int) and not isinstance(seed_value, bool):
        if not 1 <= seed_value <= len(DEFAULT_SEEDS):
            raise _error("seeds", f"integer count must be between 1 and {len(DEFAULT_SEEDS)}")
        seeds = DEFAULT_SEEDS[:seed_value]
    elif (isinstance(seed_value, list) and seed_value
          and all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seed_value)):
        seeds = list(seed_value)
    else:
        raise _error("seeds", "must be an integer or a non-empty list of integers")
    if len(set(seeds)) != len(seeds):
        raise _error("seeds", "must be unique")
    if kind == "ensemble" and len(seeds) != 5:
        raise _error("seeds", "ensemble requires exactly five candidate seeds")
    if kind == "ensemble" and 29 not in seeds:
        raise _error("seeds", "ensemble requires deployed baseline seed 29")
    slices = raw.get("slices", DEFAULT_SLICES)
    if not isinstance(slices, list) or not slices:
        raise _error("slices", "must be a non-empty list")
    normalized_slices: list[str | int] = []
    for index, value in enumerate(slices):
        if value == "all":
            normalized_slices.append("all")
        elif isinstance(value, int) and value >= 0:
            normalized_slices.append(value)
        else:
            raise _error(f"slices[{index}]", "must be 'all' or a non-negative integer")
    if 50_000 not in normalized_slices:
        raise _error("slices", "must include 50000 for the decision gate")
    if "cost" not in raw:
        raise _error("cost", "is required")
    cost_raw = raw["cost"]
    if not isinstance(cost_raw, dict):
        raise _error("cost", "must be a mapping")
    missing_cost = {"spread_bps", "fee_bps", "fee_basis"} - set(cost_raw)
    if missing_cost:
        raise _error("cost", f"missing {sorted(missing_cost)}")
    try:
        cost = CostModel(
            float(cost_raw.get("spread_bps", 0)),
            float(cost_raw.get("fee_bps", 0)),
            str(cost_raw.get("fee_basis", "winnings")),
        )
    except (TypeError, ValueError) as exc:
        raise _error("cost", str(exc)) from exc
    try:
        expected = float(raw["expected_minutes_per_seed"])
        ceiling = float(raw.get("ceiling_minutes", 240))
    except (TypeError, ValueError) as exc:
        raise _error("expected_minutes_per_seed/ceiling_minutes", "must be numeric") from exc
    if expected < 0 or ceiling <= 0:
        raise _error("expected_minutes_per_seed/ceiling_minutes", "must be non-negative/positive")
    folds = raw.get("folds", [])
    if not isinstance(folds, list):
        raise _error("folds", "must be a list")
    normalized_folds = []
    for index, fold in enumerate(folds):
        required = {"train_until", "select_from", "select_until"}
        if not isinstance(fold, dict) or not required.issubset(fold):
            raise _error(f"folds[{index}]", f"requires {sorted(required)}")
        parsed = {key: str(fold[key]) for key in required}
        try:
            ordered = (pd.Timestamp(parsed["train_until"]),
                       pd.Timestamp(parsed["select_from"]),
                       pd.Timestamp(parsed["select_until"]))
        except (TypeError, ValueError) as exc:
            raise _error(f"folds[{index}]", "contains an invalid date") from exc
        if not ordered[0] < ordered[1] <= ordered[2]:
            raise _error(f"folds[{index}]", "dates must satisfy train_until < select_from <= select_until")
        normalized_folds.append(parsed)
    if kind == "ensemble" and normalized_folds:
        raise _error("folds", "are not supported for an immutable production baseline")
    verify = raw.get("verify", [])
    if not isinstance(verify, list) or not all(isinstance(value, str) for value in verify):
        raise _error("verify", "must be a list of hook names")
    unknown_verify = set(verify) - VERIFY_NAMES
    if unknown_verify:
        raise _error("verify", f"unknown hook(s): {sorted(unknown_verify)}")
    if not isinstance(raw["out_dir"], str) or not raw["out_dir"]:
        raise _error("out_dir", "must be a non-empty path string")
    out = Path(raw["out_dir"])
    out = out if out.is_absolute() else ROOT / out
    return HarnessConfig(
        raw=raw, name=name, frame=frame, baseline_args=baseline_args,
        candidate_kind=kind, candidate_args=candidate_args, seeds=seeds,
        slices=normalized_slices, cost=cost,
        expected_minutes_per_seed=expected, ceiling_minutes=ceiling,
        odds_role=str(raw.get("odds_role", "odds_iteration_v2")),
        folds=normalized_folds, verify=verify, out_dir=out,
        config_hash=_canonical_hash(raw, frame, frame_hash),
    )


_DIFF_IDENTITIES = {
    "elo_diff_batting": ("team1_batting_elo", "team2_batting_elo"),
    "elo_diff_bowling": ("team1_bowling_elo", "team2_bowling_elo"),
    "batting_avg_diff": ("team1_batting_avg", "team2_batting_avg"),
    "bowling_econ_diff": ("team1_bowling_econ", "team2_bowling_econ"),
    "win_rate_diff": ("team1_win_rate_last_10", "team2_win_rate_last_10"),
    "top6_batting_elo_diff": ("team1_top6_batting_elo_avg", "team2_top6_batting_elo_avg"),
    "bottom5_bowling_elo_diff": ("team1_bottom5_bowling_elo_avg", "team2_bottom5_bowling_elo_avg"),
}


def diff_negation_identity(frame_dir: Path) -> None:
    train = pd.read_parquet(frame_dir / "train.parquet")
    checked = 0
    for diff, (left, right) in _DIFF_IDENTITIES.items():
        if diff not in train.columns:
            continue
        if left not in train or right not in train:
            raise AssertionError(f"diff_negation_identity: missing inputs for {diff}")
        if not np.array_equal(train[diff].to_numpy(),
                              (train[left] - train[right]).to_numpy()):
            raise AssertionError(f"diff_negation_identity: {diff} != {left} - {right}")
        checked += 1
    if not checked:
        raise AssertionError("diff_negation_identity: no known diff columns")


def h2h_prior_half(frame_dir: Path) -> None:
    train = pd.read_parquet(frame_dir / "train.parquet")
    rows = train.loc[train["h2h_n_meetings"] == 0,
                     "h2h_team1_win_rate_shrunk"]
    if rows.empty or not (rows == 0.5).all():
        raise AssertionError("h2h_prior_half: n=0 prior is not uniformly 0.5")


def swap_doubling(frame_dir: Path) -> None:
    train = pd.read_parquet(frame_dir / "train.parquet")
    augmented = _swap_augment_train(train)
    if len(augmented) != 2 * len(train) or abs(augmented["team1_wins"].mean() - .5) >= 1e-12:
        raise AssertionError("swap_doubling: augmentation contract failed")


VERIFY_HOOKS = {
    "diff_negation_identity": diff_negation_identity,
    "h2h_prior_half": h2h_prior_half,
    "swap_doubling": swap_doubling,
}


def run_verify_hooks(frame_dir: Path, hooks: Iterable[str]) -> None:
    for hook in hooks:
        VERIFY_HOOKS[hook](frame_dir)
        print(f"verify {hook}: OK")


def _prediction_parts(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = json.loads(path.read_text())
    if "predictions" in payload and isinstance(payload.get("summary"), dict):
        return dict(payload["summary"]), dict(payload["predictions"])
    return {}, payload


def _write_predictions(path: Path, summary: dict[str, Any], rows: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"summary": summary, "predictions": rows}, indent=2))


def _stub_train(frame: Path, model_dir: Path, seed: int, arm: str,
                config_hash: str) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.pkl").unlink(missing_ok=True)
    test = pd.read_parquet(frame / "test.parquet")
    rows: dict[str, dict[str, Any]] = {}
    arm_strength = .14 if arm == "candidate" else .08
    jitter = ((seed * 37) % 19 - 9) / 1000
    for index, row in test.iterrows():
        truth = int(row["team1_wins"])
        p = .5 + (arm_strength if truth else -arm_strength) + jitter
        match_id = str(row.get("cricsheet_id", row["match_id"]))
        rows[match_id] = {
            "match_id": match_id,
            "cricsheet_id": match_id,
            "display_match_id": str(row.get("display_match_id", row["match_id"])),
            "team1": str(row["team1"]), "team2": str(row["team2"]),
            "p_team1": float(p), "p_team2": float(1 - p),
            "team1_wins": truth, "match_date": str(row["match_date"]),
        }
    out = model_dir / "test_predictions.json"
    _write_predictions(out, {
        "arm": arm, "model_seed": seed, "config_hash": config_hash,
        "stub_trainer": True, "dry_run": True,
        "degraded_unseen_categories": 0,
    }, rows)
    (model_dir / "stub_model.json").write_text(json.dumps({"seed": seed, "arm": arm}))
    return out


def _stamp_prediction(path: Path, arm: str, seed: int | str, config_hash: str) -> None:
    summary, rows = _prediction_parts(path)
    summary.update({"arm": arm, "model_seed": seed, "config_hash": config_hash})
    _write_predictions(path, summary, rows)


def _trainer_argv(config: HarnessConfig, arm: str, seed: int, frame: Path,
                  model_dir: Path, dry_run: bool) -> list[str]:
    args = config.baseline_args if arm == "baseline" else config.candidate_args
    if dry_run:
        return ["dry-run-stub", "--data-dir", str(frame), "--model-dir", str(model_dir),
                "--seed", str(seed), "--arm", arm, *args]
    return ["uv", "run", "--no-sync", "python", "scripts/xgboost_match_v1.py",
            "--cmd", "both", "--data-dir", str(frame), "--model-dir", str(model_dir),
            "--seed", str(seed), "--fit-encoders-on", "train", *args]


def _write_completion(config: HarnessConfig, arm: str, seed: int, frame: Path,
                      model_dir: Path, *, dry_run: bool) -> None:
    argv = _trainer_argv(config, arm, seed, frame, model_dir, dry_run)
    model = model_dir / "model.pkl"
    stub_model = model_dir / "stub_model.json"
    record = {
        "arm": arm,
        "seed": seed,
        "config_hash": config.config_hash,
        "dry_run": dry_run,
        "trainer_argv": argv,
        "trainer_argv_sha256": _sha256_argv(argv),
        "model_sha256": _sha256_file(model) if model.is_file() else None,
        "stub_model_sha256": _sha256_file(stub_model) if stub_model.is_file() else None,
        "test_predictions_sha256": _sha256_file(model_dir / "test_predictions.json"),
        "sliced_sha256": {
            str(value): _sha256_file(model_dir / f"sliced_{value}.json")
            for value in config.slices
        },
    }
    (model_dir / "harness_run.json").write_text(json.dumps(record, indent=2))


def _summary_matches(path: Path, arm: str, seed: int, config_hash: str) -> bool:
    try:
        summary = json.loads(path.read_text())["summary"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return False
    return (summary.get("arm") == arm and summary.get("model_seed") == seed
            and summary.get("config_hash") == config_hash)


def _completed(config: HarnessConfig, arm: str, seed: int, frame: Path,
               model_dir: Path, *, dry_run: bool) -> bool:
    completion_path = model_dir / "harness_run.json"
    prediction_path = model_dir / "test_predictions.json"
    blended_path = model_dir / "blended.json"
    try:
        record = json.loads(completion_path.read_text())
    except (OSError, json.JSONDecodeError, TypeError):
        return False
    argv = _trainer_argv(config, arm, seed, frame, model_dir, dry_run)
    expected_identity = {
        "arm": arm,
        "seed": seed,
        "config_hash": config.config_hash,
        "dry_run": dry_run,
        "trainer_argv": argv,
        "trainer_argv_sha256": _sha256_argv(argv),
    }
    if any(record.get(key) != value for key, value in expected_identity.items()):
        return False
    if not _summary_matches(prediction_path, arm, seed, config.config_hash):
        return False
    if not _summary_matches(blended_path, arm, seed, config.config_hash):
        return False
    if dry_run:
        try:
            summary, _ = _prediction_parts(prediction_path)
        except (OSError, json.JSONDecodeError, TypeError):
            return False
        stub_model = model_dir / "stub_model.json"
        if (summary.get("stub_trainer") is not True or record.get("model_sha256") is not None
                or not stub_model.is_file()
                or record.get("stub_model_sha256") != _sha256_file(stub_model)):
            return False
    else:
        model = model_dir / "model.pkl"
        try:
            summary, _ = _prediction_parts(prediction_path)
        except (OSError, json.JSONDecodeError, TypeError):
            return False
        if (summary.get("stub_trainer") is not None or not model.is_file()
                or record.get("model_sha256") != _sha256_file(model)
                or record.get("stub_model_sha256") is not None):
            return False
    try:
        if record.get("test_predictions_sha256") != _sha256_file(prediction_path):
            return False
    except OSError:
        return False
    recorded_slices = record.get("sliced_sha256")
    if not isinstance(recorded_slices, dict):
        return False
    for value in config.slices:
        sliced = model_dir / f"sliced_{value}.json"
        if not _summary_matches(sliced, arm, seed, config.config_hash):
            return False
        try:
            if recorded_slices.get(str(value)) != _sha256_file(sliced):
                return False
        except OSError:
            return False
    if set(recorded_slices) != {str(value) for value in config.slices}:
        return False
    return True


def _run_trainer(config: HarnessConfig, arm: str, seed: int, frame: Path,
                 model_dir: Path, dry_run: bool) -> Path:
    (model_dir / "harness_run.json").unlink(missing_ok=True)
    if dry_run:
        return _stub_train(frame, model_dir, seed, arm, config.config_hash)
    (model_dir / "stub_model.json").unlink(missing_ok=True)
    command = _trainer_argv(config, arm, seed, frame, model_dir, dry_run)
    print("$ " + " ".join(command), flush=True)
    run_started_ns = time.time_ns()
    subprocess.run(command, cwd=ROOT, check=True)
    path = model_dir / "test_predictions.json"
    if not path.is_file() or path.stat().st_mtime_ns < run_started_ns:
        raise RuntimeError("trainer did not write fresh test_predictions.json in this run")
    if not (model_dir / "model.pkl").is_file():
        raise RuntimeError("trainer did not write model.pkl alongside its predictions")
    _stamp_prediction(path, arm, seed, config.config_hash)
    return path


def _registry_entry(role: str, registry_path: Path, root: Path) -> tuple[Path, Path]:
    payload = json.loads(registry_path.read_text())
    matches = [row for row in payload["registered_odds"] if row["role"] == role]
    if len(matches) != 1:
        raise ConfigError(f"odds_role: {role!r} is not uniquely registered")
    row = matches[0]
    odds = Path(row["path"])
    clusters = Path(row["cluster_source_dir"])
    return ((root / odds) if not odds.is_absolute() else odds,
            (root / clusters) if not clusters.is_absolute() else clusters)


def _direct_envelope(prediction_rows: dict[str, dict[str, Any]], odds_path: Path) -> dict[str, Any]:
    odds_payload = json.loads(odds_path.read_text())
    odds_rows = odds_payload.get("matches", odds_payload if isinstance(odds_payload, list) else [])
    index = {str(row.get("cricsheet_id", row.get("match_id"))): row for row in odds_rows}
    matches = []
    for match_id, pred in prediction_rows.items():
        odds = index.get(str(match_id))
        if odds is None:
            continue
        teams = [pred["team1"], pred["team2"]]
        raw_odds = dict((odds.get("odds") or {}).get("winner") or odds.get("market_odds") or {})
        raw_odds.pop("timestamp", None)
        inverse = {team: 1 / float(raw_odds[team]) for team in teams}
        total = sum(inverse.values())
        market_prob = {team: inverse[team] / total for team in teams}
        matches.append({
            "match_id": str(match_id), "cricsheet_id": str(match_id),
            "display_match_id": pred.get("display_match_id"), "teams": teams,
            "actual_winner": odds.get("actual_winner", teams[0] if pred["team1_wins"] else teams[1]),
            "simulated_prob": {teams[0]: .5, teams[1]: .5},
            "market_prob": market_prob, "market_odds": raw_odds,
            "edge": {team: .5 - market_prob[team] for team in teams},
        })
    if len(matches) != len(prediction_rows):
        raise ValueError("registered odds do not cover every test prediction")
    return {"summary": {"envelope_for": "direct-only"}, "matches": matches}


def _evaluate(config: HarnessConfig, model_dir: Path, arm: str, seed: int | str,
              odds_path: Path, cluster_dir: Path,
              *, config_hash: str | None = None) -> dict[str | int, Path]:
    _, rows = _prediction_parts(model_dir / "test_predictions.json")
    envelope = _direct_envelope(rows, odds_path)
    blended = blend_eval_json.blend(envelope, rows, 0.0)
    prediction_ids = set(map(str, rows))
    blended_ids = {str(row["match_id"]) for row in blended["matches"]}
    if blended_ids != prediction_ids:
        raise AssertionError("blended match-id set differs from prediction set")
    stamp_hash = config_hash or config.config_hash
    blended.setdefault("summary", {}).update({
        "arm": arm, "model_seed": seed, "config_hash": stamp_hash,
    })
    blended_path = model_dir / "blended.json"
    blended_path.write_text(json.dumps(blended, indent=2))
    result: dict[str | int, Path] = {}
    for slice_value in config.slices:
        minimum = None if slice_value == "all" else int(slice_value)
        payload = reslice_eval_json.reslice(
            blended_path, odds_path, minimum, cluster_source_dir=cluster_dir,
            cost_model=config.cost,
        )
        payload["summary"].update({
            "arm": arm, "model_seed": seed, "config_hash": stamp_hash,
        })
        out = model_dir / f"sliced_{slice_value}.json"
        out.write_text(json.dumps(payload, indent=2))
        result[slice_value] = out
    return result


def _production_baseline(config: HarnessConfig, odds_path: Path,
                         cluster_dir: Path) -> tuple[Path, dict[str | int, Path]]:
    """Copy and evaluate the manifest-registered production predictions."""
    entries = load_manifest(MANIFEST_PATH)
    try:
        entry = entries["match_model_prod"]
    except KeyError as exc:
        raise ConfigError("baseline.role: match_model_prod is not registered") from exc
    status, actual_hash = verify_role("match_model_prod", manifest_path=MANIFEST_PATH)
    if status != "OK" or actual_hash is None:
        raise ConfigError(
            f"baseline.role: match_model_prod verification {status}"
        )
    manifest_root = _manifest_root()
    source_dir = Path(entry["path"])
    if not source_dir.is_absolute():
        source_dir = manifest_root / source_dir
    source = source_dir / "test_predictions.json"
    if not source.is_file():
        raise ConfigError(f"baseline.role: production predictions missing: {source}")
    source_summary, rows = _prediction_parts(source)
    source_sha256 = _sha256_file(source)
    target_dir = config.out_dir / "baseline_prod_seed29"
    target = target_dir / "test_predictions.json"
    manifest_hash = actual_hash
    _write_predictions(target, {
        "arm": "baseline", "model_seed": "prod_seed29", "config_hash": manifest_hash,
        "manifest_hash": manifest_hash, "artifact_role": "match_model_prod",
        "source_sha256": source_sha256, "source_summary": source_summary,
    }, rows)
    sliced = _evaluate(
        config, target_dir, "baseline", "prod_seed29", odds_path, cluster_dir,
        config_hash=manifest_hash,
    )
    return target, sliced


def _prepare_dry_registry(config: HarnessConfig) -> Path:
    """Create outcome-complete, registered prices beside dry-run outputs."""
    registry = config.out_dir / "dry_registry.json"
    if registry.exists():
        return registry
    test = pd.read_parquet(config.frame / "test.parquet")
    odds_rows = []
    cluster_dir = config.out_dir / "dry_clusters"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    for index, row in test.iterrows():
        match_id = str(row.get("cricsheet_id", row["match_id"]))
        team1, team2 = str(row["team1"]), str(row["team2"])
        winner = team1 if int(row["team1_wins"]) else team2
        odds_rows.append({
            "match_id": match_id, "cricsheet_id": match_id,
            "display_match_id": str(row.get("display_match_id", row["match_id"])),
            "actual_winner": winner, "team1": team1, "team2": team2,
            "odds": {"winner": {team1: 2.0, team2: 2.0}},
            "polymarket_volume_usd": 200_000.0,
        })
        fixture = {"info": {
            "dates": [str(row["match_date"])[:10]], "teams": [team1, team2],
            "venue": str(row.get("venue", "Dry Run Ground")),
            "event": {"name": f"Dry Run Block {index}"},
        }}
        (cluster_dir / f"{match_id}.json").write_text(json.dumps(fixture))
    odds_path = config.out_dir / "dry_odds.json"
    odds_path.write_text(json.dumps({"matches": odds_rows}, indent=2))
    digest = hashlib.sha256(odds_path.read_bytes()).hexdigest()
    registry.write_text(json.dumps({"registered_odds": [{
        "role": config.odds_role, "path": odds_path.name,
        "cluster_source_dir": cluster_dir.name, "sha256": digest,
        "row_count": len(odds_rows),
    }]}, indent=2))
    return registry


def _assert_pairing(candidate: Path, baseline: Path, label: str) -> None:
    _, candidate_rows = _prediction_parts(candidate)
    _, baseline_rows = _prediction_parts(baseline)
    ids = lambda rows: {
        str(row.get("match_id", key)) for key, row in rows.items()
    }
    if ids(candidate_rows) != ids(baseline_rows):
        raise ValueError(f"pairing check failed for {label}: match-id sets differ")


def _logit_mean(paths: list[Path], out: Path, config_hash: str) -> Path:
    parsed = [_prediction_parts(path)[1] for path in paths]
    ordered_ids = list(parsed[0])
    ids = set(ordered_ids)
    if any(set(rows) != ids for rows in parsed[1:]):
        raise ValueError("ensemble candidate seed match-id sets differ")
    result = {}
    for match_id in ordered_ids:
        template = dict(parsed[0][match_id])
        probabilities = np.array([rows[match_id]["p_team1"] for rows in parsed], dtype=float)
        logits = np.log(np.clip(probabilities, 1e-9, 1 - 1e-9) /
                        np.clip(1 - probabilities, 1e-9, 1 - 1e-9))
        probability = 1 / (1 + math.exp(-float(logits.mean())))
        template.update({"p_team1": probability, "p_team2": 1 - probability})
        result[match_id] = template
    _write_predictions(out, {"model_seed": "ensemble", "config_hash": config_hash,
                             "ensemble_method": "logit_mean", "member_count": len(paths)}, result)
    return out


def _metrics(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text())["summary"]
    return {"ll": summary["avg_log_loss"], "roi": summary["flat_betting_roi_pct"],
            "n_matches": summary["n_matches_evaluated"],
            "n_bets": summary["flat_betting_bets_placed"]}


def _make_fold_frame(config: HarnessConfig, index: int, fold: dict[str, str]) -> Path:
    frames = [pd.read_parquet(config.frame / f"{name}.parquet")
              for name in ("train", "validation", "test")]
    combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["match_id"])
    dates = pd.to_datetime(combined["match_date"], errors="raise")
    training = combined.loc[dates <= pd.Timestamp(fold["train_until"])].copy()
    selection = combined.loc[(dates >= pd.Timestamp(fold["select_from"]))
                             & (dates <= pd.Timestamp(fold["select_until"]))].copy()
    if training.empty or selection.empty:
        raise ValueError(f"folds[{index}]: empty train or selection rows")
    target = config.out_dir / "fold_frames" / f"fold{index}"
    target.mkdir(parents=True, exist_ok=True)
    training.to_parquet(target / "train.parquet", index=False)
    selection.to_parquet(target / "validation.parquet", index=False)
    selection.to_parquet(target / "test.parquet", index=False)
    for sidecar in config.frame.glob("*.json"):
        shutil.copy2(sidecar, target / sidecar.name)
    return target


def _run_folds(config: HarnessConfig, dry_run: bool) -> list[dict[str, Any]]:
    results = []
    for index, fold in enumerate(config.folds):
        frame = _make_fold_frame(config, index, fold)
        arm_means = {}
        for arm in ("baseline", "candidate"):
            losses = []
            for seed in config.seeds:
                model_dir = config.out_dir / f"{arm}_seed{seed}" / "folds" / f"fold{index}"
                path = _run_trainer(config, arm, seed, frame, model_dir, dry_run)
                _, rows = _prediction_parts(path)
                losses.extend(-math.log(max(min(row["p_team1"] if row["team1_wins"] else row["p_team2"],
                                                     1 - 1e-9), 1e-9)) for row in rows.values())
            arm_means[arm] = float(np.mean(losses))
        results.append({"fold": index, **fold, "log_loss": arm_means,
                        "delta_log_loss": arm_means["candidate"] - arm_means["baseline"]})
    return results


def run(config: HarnessConfig, *, dry_run: bool = False,
        allow_long: bool = False) -> dict[str, Any]:
    estimate = len(config.seeds) * config.expected_minutes_per_seed
    if estimate > config.ceiling_minutes and not allow_long:
        raise RuntimeError(f"budget guard: {estimate:g} expected minutes exceeds ceiling_minutes={config.ceiling_minutes:g}; pass --allow-long")
    run_verify_hooks(config.frame, config.verify)
    config.out_dir.mkdir(parents=True, exist_ok=True)

    registry_path = ROOT / "docs/registered_odds.json"
    registry_root = ROOT
    odds_role = config.odds_role
    if dry_run:
        registry_path = _prepare_dry_registry(config)
        registry_root = config.out_dir
    odds_path, cluster_dir = _registry_entry(odds_role, registry_path, registry_root)

    predictions: dict[str, dict[int, Path]] = {"baseline": {}, "candidate": {}}
    sliced: dict[str, dict[int, dict[str | int, Path]]] = {"baseline": {}, "candidate": {}}
    for seed in config.seeds:
        arms = ("candidate",) if config.candidate_kind == "ensemble" else ("baseline", "candidate")
        for arm in arms:
            model_dir = config.out_dir / f"{arm}_seed{seed}"
            if _completed(config, arm, seed, config.frame, model_dir, dry_run=dry_run):
                print(f"resume: {arm} seed {seed} (matching config_hash)")
                prediction = model_dir / "test_predictions.json"
            else:
                prediction = _run_trainer(config, arm, seed, config.frame, model_dir, dry_run)
                _evaluate(config, model_dir, arm, seed, odds_path, cluster_dir)
                _write_completion(
                    config, arm, seed, config.frame, model_dir, dry_run=dry_run
                )
            predictions[arm][seed] = prediction
            sliced[arm][seed] = {value: model_dir / f"sliced_{value}.json" for value in config.slices}
        if config.candidate_kind != "ensemble":
            _assert_pairing(predictions["candidate"][seed], predictions["baseline"][seed], str(seed))

    diagnostics: list[dict[str, Any]] = []
    if config.candidate_kind == "ensemble":
        production_prediction, production_slices = _production_baseline(
            config, odds_path, cluster_dir
        )
        ensemble_dir = config.out_dir / "candidate_ensemble"
        ensemble_path = _logit_mean([predictions["candidate"][seed] for seed in config.seeds],
                                    ensemble_dir / "test_predictions.json", config.config_hash)
        _assert_pairing(ensemble_path, production_prediction, "ensemble_vs_prod_seed29")
        ensemble_slices = _evaluate(
            config, ensemble_dir, "candidate", "ensemble", odds_path, cluster_dir
        )
        deciding_candidate = ensemble_slices[50_000]
        deciding_baseline = production_slices[50_000]
        payload = json.loads(deciding_candidate.read_text())
        payload["summary"]["model_seed"] = "prod_seed29"
        deciding_candidate.write_text(json.dumps(payload, indent=2))
        for seed in config.seeds:
            if seed == 29:
                continue
            candidate_metrics = _metrics(sliced["candidate"][seed][50_000])
            baseline_metrics = _metrics(production_slices[50_000])
            diagnostics.append({"candidate_seed": seed,
                                "candidate_member": candidate_metrics,
                                "production_baseline": baseline_metrics,
                                "delta_ll": candidate_metrics["ll"] - baseline_metrics["ll"],
                                "delta_roi": candidate_metrics["roi"] - baseline_metrics["roi"]})
        candidate_gate, baseline_gate = [deciding_candidate], [deciding_baseline]
        gate_seeds = ["prod_seed29"]
    else:
        candidate_gate = [sliced["candidate"][seed][50_000] for seed in config.seeds]
        baseline_gate = [sliced["baseline"][seed][50_000] for seed in config.seeds]
        gate_seeds = [str(seed) for seed in config.seeds]

    # H6 is deliberately complete before entering the decision function.
    prior_root = claim_gate.REPO_ROOT
    if dry_run and registry_root != ROOT:
        claim_gate.REPO_ROOT = registry_root
    try:
        verdict = claim_gate.decide(
            candidate_gate, baseline_gate, "match_model", config.cost,
            odds_role=odds_role, cluster_source_dir=cluster_dir,
            seeds=gate_seeds, registry_path=registry_path,
        )
    finally:
        claim_gate.REPO_ROOT = prior_root
    gate_path = config.out_dir / "gate.json"
    gate_path.write_text(json.dumps(verdict.as_dict(), indent=2))
    rows = []
    arms = ("candidate",) if config.candidate_kind == "ensemble" else ("baseline", "candidate")
    for arm in arms:
        for seed in config.seeds:
            for slice_value in config.slices:
                rows.append({"arm": arm, "seed": seed, "slice": slice_value,
                             **_metrics(sliced[arm][seed][slice_value])})
    if config.candidate_kind == "ensemble":
        for slice_value in config.slices:
            rows.append({"arm": "baseline", "seed": "prod_seed29", "slice": slice_value,
                         **_metrics(production_slices[slice_value])})
        for slice_value in config.slices:
            rows.append({"arm": "candidate", "seed": "ensemble", "slice": slice_value,
                         **_metrics(ensemble_slices[slice_value])})
    fold_rows = _run_folds(config, dry_run) if config.folds else []
    report = {
        "name": config.name, "config_hash": config.config_hash,
        "config": config.raw, "rows": rows, "folds": fold_rows,
        "fold_mean_log_loss": ({arm: float(np.mean([row["log_loss"][arm] for row in fold_rows]))
                                for arm in ("baseline", "candidate")} if fold_rows else None),
        "ensemble_diagnostics_only": diagnostics,
        "verdict": verdict.as_dict(), "gate_json": str(gate_path),
    }
    harness_path = config.out_dir / "harness.json"
    harness_path.write_text(json.dumps(report, indent=2))
    print(f"log_verdict.py verdict {config.name} {verdict.verdict} --gate-json {gate_path}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("config", type=Path)
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--allow-long", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        run(config, dry_run=args.dry_run, allow_long=args.allow_long)
    except (ConfigError, RuntimeError, ValueError, AssertionError,
            FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
