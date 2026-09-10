#!/usr/bin/env python3
"""Write and verify the registered stage-1 simulator config (D5).

`experiments/configs/seq_stage1_sim_v1.yaml` is a GENERATED file. Every
prose field is a literal in this script; every hash, checkpoint choice,
seed example and command line is computed from the files on disk. That
makes the config's two jobs separable:

* `--write` regenerates the YAML from the template below, so a re-pin
  after an artifact changes is one command and cannot half-update.
* `--verify` rebuilds the same payload in memory, parses the committed
  YAML, and diffs the two key by key. Any difference — a re-hashed
  artifact, an edited prose line, a hand-patched command — exits 1 and
  names the differing keys.

Two provenance fields are recorded but never compared, because they
change without the config's meaning changing: `pins_generated_at` and
`git_head_short`. They are listed in the YAML itself under
`provenance.not_verified`, which IS compared.

Checkpoint selection (D5 check 5.3) is computed here, never hardcoded:
for the `mlp` (arm B) and `full` (arm C) ablation arms, the seed with the
lowest VALIDATION log loss in `models/embeddings/t1_ablation_v1_mps/
summary.yaml` wins, ties break to the lowest seed number, and the test
split is not read.

Usage:
    uv run --no-sync python scripts/sequence_track/pin_stage1.py --write
    uv run --no-sync python scripts/sequence_track/pin_stage1.py --verify
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
from functools import lru_cache
from importlib import metadata
from pathlib import Path

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from artifacts import artifact_path, md5_directory, md5_file  # noqa: E402
from registered_experiment import sha256  # noqa: E402
# The runner's own implementation, imported rather than restated: the
# overlap table below must describe what `run_arm.py` will actually refuse.
# `run_arm`'s module level is stdlib-only, and its own import of this module
# is lazy (inside `main`), so there is no import cycle.
from sequence_track.run_arm import (  # noqa: E402
    largest_permitted_n_sims, min_seed_gap, seed_intervals_disjoint)

CONFIG_PATH = REPO_ROOT / "experiments" / "configs" / "seq_stage1_sim_v1.yaml"

# ---------------------------------------------------------------------------
# Registered constants (the decisions, not the measurements)
# ---------------------------------------------------------------------------

REGISTERED_DATE = "2026-09-10"
BASE_SEED = 20260910
SECOND_BASE_SEED = 20260911
THIRD_BATCH_BASE_SEED = 20260912  # third convergence batch only
# The seed line the 1b batches share; every joint screen uses this set.
BATCH_BASE_SEEDS = (BASE_SEED, SECOND_BASE_SEED, THIRD_BATCH_BASE_SEED)
CLIP_BOUNDS = [0.01, 0.99]
THREADS = 4
DEVICE = "cpu"
EQUIVALENCE_MARGIN_LL = 0.007
CONVERGENCE_THRESHOLD = 0.002
CONVERGENCE_PLACEHOLDER = "to_be_filled_before_1d"
CONVERGENCE_FIRST_CANDIDATE = 100
# The n_sims candidates screened for seed-interval overlap. The engine seeds
# simulation i with random_seed + i, so a fixture occupies
# [fixture_seed, fixture_seed + n_sims): the bigger the candidate, the more
# of the 2^31 seed space each fixture claims.
# The list runs two doublings past the joint seed-interval wall on purpose:
# the config must SHOW where the registered base seeds stop being usable
# rather than stopping short of it.
CONVERGENCE_CANDIDATES = [100, 200, 400, 800, 1600, 3200, 6400]
# Set to an int once the 1b replicated convergence check is read; every
# `--n-sims` in the full-run command lines is rendered from it, so the
# commands and the protocol can never disagree. `None` = not yet chosen,
# and the commands carry the `<chosen_n_sims>` token instead of a number.
CHOSEN_N_SIMS = None
N_SIMS_TOKEN = "<chosen_n_sims>"
SMOKE_N_SIMS = 10
FIXTURE_SET_EXPECTED_COUNT = 255

# The B18 extras sidecar's identity, quoted in the PPC report. A stage-1 run
# whose extras law is not this file is not comparable to the PPC cohort.
EXTRAS_GRAFT_SHA256 = (
    "ad6e863b1a2dfa3b47259d3d952157f392a0579af960cf597bcb90d2eb68bfa1")
EXTRAS_GRAFT_REPORT = (
    "research/reports/embeddings/T1_SIM_PARITY_PPC_V1.md line 59")

ODDS_REGISTRY = "docs/registered_odds.json"

# Every artifact of record is resolved by MANIFEST ROLE, never spelled as a
# path literal, so a promotion that moves an artifact moves this config too
# (and `scripts/tests/test_manifest_defaults.py` stays green). The role name
# is recorded beside each resolved path in the YAML.
ROLE_BALL_MODEL_PROD = "ball_model_prod"
ROLE_BOWLER_USAGE = "bowler_phase_usage"
ROLE_ROSTER_POLICY = "bowler_roster_policy"
ROLE_B10_CORPUS = "bowler_usage_corpus_b10"
ROLE_PROP_CORPUS = "prop_fair_baseline_corpus_v2"
ROLE_FIXTURE_SET = "iteration_set_v2"
ROLE_ODDS = "odds_iteration_v2"
ROLE_STATS_CACHE = {"i7": "stats_cache_i7", "v3": "stats_cache_v3_legacy"}

# The two artifact families with no manifest role — the stage-1 namespace and
# the unpromoted B18 sidecar — are COMPOSED from segments, the way
# `run_arm.py` composes them, rather than written as one literal.
MODELS_ROOT = Path("models")
SEQ_STAGE1_ROOT = MODELS_ROOT / "embeddings" / "seq_stage1"
ABLATION_DIR = MODELS_ROOT / "embeddings" / "t1_ablation_v1_mps"
ABLATION_SUMMARY = ABLATION_DIR / "summary.yaml"
SMOKE_FIXTURE_DIR = SEQ_STAGE1_ROOT / "smoke" / "fixtures"
# Populated at 1b; registered now so the convergence and variability
# batches can pass run_arm's --config preflight (Astra round 2, item 4c).
TIMING_FIXTURE_DIR = SEQ_STAGE1_ROOT / "timing" / "fixtures"
TIMING_SHARD_SIZE = 10
A50_DIR = SEQ_STAGE1_ROOT / "a50"
EXTRAS_GRAFT = MODELS_ROOT / "auto" / "b18" / "extras_graft_v1.json"

# Neither of these matches an artifact-manifest role, and neither is under a
# manifest-owned prefix; `run_arm.py` carries the same two literals.
CONTEXT_DIR = "data/t20s_json"
PLAYER_METADATA = "data/all_players_enriched.csv"

RUNNER = "scripts/sequence_track/run_arm.py"
ENGINE = "scripts/sim_v1_2.py"
DIR_HASH_CONTRACT = "md5_file_or_sorted_relative_path_md5_lines_v1"

# The SOURCE CLOSURE on the stage-1 execution path. Pinning only the runner
# and the engine would leave a change to replay, rehydration, the delegated
# runner, the statistics or the scorers able to move every result while
# `--verify` still passed, so everything a stage-1 number actually depends on
# is hashed. A file listed here but absent on the checkout is dropped and
# named in `provenance.source_md5_absent` rather than silently skipped.
PROVENANCE_SOURCE_CLOSURE = [
    # stage-1 tools: the arm runner, the cross-arm audit, the two scorers,
    # and this pin script itself.
    "scripts/sequence_track/run_arm.py",
    "scripts/sequence_track/audit_cross_arm.py",
    "scripts/sequence_track/score_realism.py",
    "scripts/sequence_track/score_props.py",
    "scripts/sequence_track/pin_stage1.py",
    # the delegated runner, the evaluator, the replay lifecycle, the loaders,
    # the reslicer, and the gate with its statistics and market arithmetic.
    "scripts/sim_eval/run_sim_eval.py",
    "scripts/sim_eval/run_sim_eval_t1.py",
    "scripts/sim_eval/match_evaluator.py",
    "scripts/sim_eval/same_day_stats.py",
    "scripts/sim_eval/loaders.py",
    "scripts/sim_eval/reslice_eval_json.py",
    "scripts/sim_eval/claim_gate.py",
    "scripts/sim_eval/eval_statistics.py",
    "scripts/sim_eval/market_math.py",
    "scripts/sim_eval/settlement_common.py",
    # the prop reducers the exploratory scorer compares against.
    "scripts/sim_eval/prop_backtest.py",
    "scripts/sim_eval/prop_fair_baselines.py",
    # the engine and the two model wrappers, plus the T1 feature stack.
    "scripts/sim_v1_2.py",
    "scripts/sim_t1.py",
    "scripts/transformer_t1.py",
    "scripts/embeddings_e1.py",
    "scripts/t1_ppc_common.py",
    # tracker rehydration, parsing, the stats cache, chronology, metadata,
    # the feature contract, and the two provenance helpers this file uses.
    "scripts/tracker_rehydration.py",
    "scripts/parsing_v2.py",
    "scripts/stats_provider.py",
    "scripts/stats_sqlite_backend.py",
    "scripts/loaders_common.py",
    "scripts/player_metadata.py",
    "scripts/feature_registry.py",
    # match/venue identity and the ELO contract: the odds join, the venue
    # canonicalisation the replay lifecycle checks, and the model/state ELO
    # version guard all live here (Astra round 2, item 6).
    "scripts/identity_maps.py",
    "scripts/match_identity.py",
    "scripts/elo_update.py",
    "scripts/registered_experiment.py",
    "scripts/artifacts.py",
]

# Verified like any other pinned fact: a different interpreter or a different
# xgboost/torch build changes simulation output, so it changes the config.
RUNTIME_DISTRIBUTIONS = ("xgboost", "torch", "numpy", "pandas",
                         "scikit-learn")

NOT_VERIFIED = ("provenance.pins_generated_at", "provenance.git_head_short")

SUB_SEED_STREAMS = ("outcome", "extras", "selector")
_LOW31 = (1 << 31) - 1

# Every binary and MAE family `prop_fair_baselines.baseline_rows` pairs.
PROP_FAMILIES = [
    "top_batter", "top_bowler",
    "batter_50plus", "batter_6plus_six",
    "batter_fours_1plus", "batter_fours_2plus", "batter_fours_3plus",
    "bowler_wkts_1plus", "bowler_wkts_2plus", "bowler_wkts_3plus",
    "innings_runs_ou_160_5", "innings_runs_ou_170_5", "innings_runs_ou_180_5",
    "pp_total_ou_45_5", "pp_total_ou_50_5", "pp_total_ou_55_5",
    "team_highest_individual_ou_29_5", "team_highest_individual_ou_34_5",
    "team_highest_individual_ou_39_5", "first_wicket_runs_ou_30_5",
    "match_total_sixes_ou_15_5", "match_total_sixes_ou_20_5",
    "highest_over_runs_ou_18_5", "highest_over_runs_ou_24_5",
    "batter_runs_mae", "team_total_fours_mae", "team_total_sixes_mae",
    "team_first_over_mae", "highest_individual_mae",
]

# `sim_eval.market_math.DEFAULT_SCENARIOS`, spelled out so the config records
# the cost grid a market claim has to survive.
COST_SCENARIOS = [
    "spread 0 bps, fee 0 bps on winnings",
    "spread 100 bps, fee 0 bps on winnings",
    "spread 200 bps, fee 0 bps on winnings",
    "spread 100 bps, fee 100 bps on winnings",
    "spread 100 bps, fee 100 bps on stake",
]

ARM_SPEC = {
    "A": {
        "role": "production system (reference for B-A, C-A and A50-A)",
        "description": (
            "promoted i7 no-class-weights XGBoost ball model, 114 features, "
            "manifest role ball_model_prod, served RAW (no ball calibrator, "
            "per D16/D17)"),
        "model_dir_role": ROLE_BALL_MODEL_PROD,
        "checkpoint_name": "xgboost_model_i7.pkl",
        "stats_version": "i7",
        "model_type": "xgboost",
    },
    "A50": {
        "role": (
            "production family at equal information (exploratory system "
            "control)"),
        "description": (
            "XGBoost on the production hyperparameters trained on T1's 50 "
            "features only, from the i7 frame by column selection (stage-1 "
            "D4 artifact)"),
        "model_dir_composed": A50_DIR,
        "checkpoint_name": "xgboost_model_a50.pkl",
        "stats_version": "i7",
        "model_type": "xgboost",
    },
    "B": {
        "role": "nonlinear control (reference for C-B)",
        "description": (
            "token MLP over the same 50 pre-ball features, no sequence "
            "memory; registered checkpoint from the t1_ablation_v1_mps "
            "`mlp` arm"),
        "ablation_arm": "mlp",
        "stats_version": "v3",
        "model_type": "transformer",
    },
    "C": {
        "role": "sequence candidate",
        "description": (
            "full T1 transformer over the same 50 pre-ball features; "
            "registered checkpoint from the t1_ablation_v1_mps `full` arm"),
        "ablation_arm": "full",
        "stats_version": "v3",
        "model_type": "transformer",
    },
}


class PinError(RuntimeError):
    """A pinned fact could not be established."""


# ---------------------------------------------------------------------------
# Seed derivation (D5 check 5.4)
#
# Written out here from the documented formula rather than imported from
# `run_arm`, so `scripts/tests/test_pin_stage1.py` can assert the two agree.
# ---------------------------------------------------------------------------

def fixture_seed(cricsheet_id: str, base_seed: int) -> int:
    """Low 31 bits of sha256("<cricsheet_id>:<base_seed>")."""
    digest = hashlib.sha256(
        f"{cricsheet_id}:{int(base_seed)}".encode()).digest()
    return int.from_bytes(digest, "big") & _LOW31


def sub_seed(seed: int, stream: str) -> int:
    """Low 31 bits of sha256("<fixture_seed>:<stream>")."""
    digest = hashlib.sha256(f"{int(seed)}:{stream}".encode()).digest()
    return int.from_bytes(digest, "big") & _LOW31


# ---------------------------------------------------------------------------
# Seed-interval overlap screen (D5 check 5.9)
# ---------------------------------------------------------------------------

def fixture_ids_for(fixture_dir) -> list:
    """Cricsheet ids (JSON stems) of a fixture directory, sorted."""
    directory = _abs(fixture_dir)
    ids = sorted(path.stem for path in directory.glob("*.json"))
    if not ids:
        raise PinError(f"no fixture JSON files in {fixture_dir}")
    return ids


def overlap_rows(fixture_ids, base_seed, candidates=None) -> list:
    """One row per n_sims candidate: overlapping pairs and the seed margin.

    `run_arm.seed_intervals_disjoint` is the implementation, so this table
    describes exactly what the runner will refuse at launch. `base_seed` may
    be one seed or the whole batch cohort (joint screen).
    """
    gap = min_seed_gap(fixture_ids, base_seed)
    rows = []
    for n_sims in (candidates or CONVERGENCE_CANDIDATES):
        overlaps = seed_intervals_disjoint(fixture_ids, base_seed, n_sims)
        rows.append({
            "n_sims": int(n_sims),
            "overlapping_pairs": len(overlaps),
            "disjoint": not overlaps,
            "permitted": (
                "yes" if not overlaps else "not_permitted_without_new_seeds"),
            "example_overlap": (
                None if not overlaps else
                f"{overlaps[0][0]} (seed {overlaps[0][1]}) and "
                f"{overlaps[0][2]} (seed {overlaps[0][3]}): gap "
                f"{overlaps[0][4]}"),
            "min_seed_gap": gap,
        })
    return rows


def joint_overlap_block(fixture_ids, base_seeds, candidates=None) -> dict:
    """The screen that matters: every batch base seed on one seed line.

    The 1b protocol runs the same fixtures under three batch base seeds, so
    the intervals a batch actually consumes are the UNION over those seeds.
    Screening each seed alone passes candidates that collide across batches
    (Astra round 2, item 5).
    """
    candidates = list(candidates or CONVERGENCE_CANDIDATES)
    seeds = [int(value) for value in base_seeds]
    gap = min_seed_gap(fixture_ids, seeds)
    permitted = largest_permitted_n_sims(fixture_ids, seeds, candidates)
    return {
        "base_seeds": seeds,
        "joint_min_seed_gap": gap,
        "largest_permitted_candidate": permitted,
        "largest_permitted_n_sims_any": gap,
        "rule": (
            "a candidate is permitted only while n_sims <= "
            "joint_min_seed_gap; larger candidates need new base seeds and "
            "a re-screen, and run_arm refuses them at launch"),
        "rows": overlap_rows(fixture_ids, seeds, candidates),
    }


# ---------------------------------------------------------------------------
# Checkpoint selection (D5 check 5.3)
# ---------------------------------------------------------------------------

def choose_seed(per_seed_rows) -> tuple[int, float]:
    """Lowest validation LL; ties break to the lowest seed number.

    `per_seed_rows` is the `summary.yaml` list of `{seed, ll, ...}` mappings
    from the VALIDATION split. The test split is never consulted.
    """
    rows = list(per_seed_rows or [])
    if not rows:
        raise PinError("no per-seed rows to choose from")
    best = min(rows, key=lambda row: (float(row["ll"]), int(row["seed"])))
    return int(best["seed"]), float(best["ll"])


def selection_table(summary_path: Path) -> dict:
    """Per-seed validation LL tables plus the chosen seed for mlp and full."""
    payload = yaml.safe_load(Path(summary_path).read_text())
    try:
        arms = payload["splits"]["validation"]["arms"]
    except (KeyError, TypeError) as exc:
        raise PinError(
            f"{summary_path}: no splits.validation.arms block") from exc
    out = {}
    for ablation_arm in ("mlp", "full"):
        if ablation_arm not in arms:
            raise PinError(f"{summary_path}: no validation arm {ablation_arm}")
        rows = arms[ablation_arm].get("per_seed")
        seed, ll = choose_seed(rows)
        out[ablation_arm] = {
            "per_seed": [
                {"seed": int(row["seed"]), "validation_ll": float(row["ll"])}
                for row in sorted(rows, key=lambda row: int(row["seed"]))
            ],
            "chosen_seed": seed,
            "chosen_validation_ll": ll,
        }
    return out


# ---------------------------------------------------------------------------
# Hash helpers, all anchored at the repo root
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _role_path(role: str) -> str:
    """The repository-relative path the manifest records for ``role``."""
    return artifact_path(role).as_posix()


def _p(path) -> str:
    """A composed or resolved path as the repository-relative string."""
    return Path(path).as_posix()


def _abs(relative) -> Path:
    return REPO_ROOT / relative


def _md5_file(relative: str) -> str:
    path = _abs(relative)
    if not path.is_file():
        raise PinError(f"missing file artifact: {relative}")
    return md5_file(path)


def _md5_dir(relative: str) -> str:
    path = _abs(relative)
    if not path.is_dir():
        raise PinError(f"missing directory artifact: {relative}")
    return md5_directory(path)


def _sha256(relative: str) -> str:
    path = _abs(relative)
    if not path.is_file():
        raise PinError(f"missing file artifact: {relative}")
    return sha256(relative)


def _json_count(relative: str) -> int:
    return len(sorted(_abs(relative).glob("*.json")))


def _require(actual, expected, label: str) -> None:
    if actual != expected:
        raise PinError(f"{label}: expected {expected!r}, found {actual!r}")


def _runout_rule() -> dict:
    """The run-out law: a code constant, read by import, with its line.

    `models/auto/d15/runout_rates.json` is cited by the engine but absent on
    this checkout, so the constant IS the rule (stage-0 design fact).
    """
    import sim_v1_2  # noqa: PLC0415 - deliberately late, and cheap

    line = None
    for index, text in enumerate(
            _abs(ENGINE).read_text().splitlines(), start=1):
        if text.startswith("RUNOUT_P"):
            line = index
            break
    if line is None:
        raise PinError(f"{ENGINE}: no top-level RUNOUT_P assignment")
    return {
        "source": ENGINE,
        "constant": "RUNOUT_P",
        "value": float(sim_v1_2.RUNOUT_P),
        "line": line,
        "note": (
            "a code constant read by import; the "
            "models/auto/d15/runout_rates.json sidecar the engine banner "
            "cites is absent on this checkout, so the constant is the rule"),
    }


def _odds_block() -> dict:
    """The registered odds role, re-hashed and checked against the registry."""
    registry = json.loads(_abs(ODDS_REGISTRY).read_text())
    rows = [row for row in registry["registered_odds"]
            if row["role"] == ROLE_ODDS]
    if len(rows) != 1:
        raise PinError(f"{ODDS_REGISTRY}: {len(rows)} rows for {ROLE_ODDS}")
    row = rows[0]
    # The odds registry and the artifact manifest both own this role; a
    # disagreement between them is a defect, not a choice to make here.
    _require(row["path"], _role_path(ROLE_ODDS),
             f"{ROLE_ODDS} path ({ODDS_REGISTRY} vs models/MANIFEST.yaml)")
    _require(row["cluster_source_dir"], _role_path(ROLE_FIXTURE_SET),
             f"{ROLE_ODDS} cluster_source_dir (vs role {ROLE_FIXTURE_SET})")
    recomputed = _sha256(row["path"])
    _require(recomputed, row["sha256"],
             f"{ROLE_ODDS} sha256 (recomputed vs {ODDS_REGISTRY})")
    return {
        "role": ROLE_ODDS,
        "path": row["path"],
        "sha256": recomputed,
        "sha256_source": (
            f"{ODDS_REGISTRY} (role {ROLE_ODDS}), recomputed and equal"),
        "cluster_source_dir": row["cluster_source_dir"],
        "cluster_source_dir_role": ROLE_FIXTURE_SET,
        "row_count": int(row["row_count"]),
    }


def _source_closure() -> tuple[dict, list]:
    """`{source: md5}` for every closure file present, plus the absent ones."""
    present = {}
    absent = []
    for source in PROVENANCE_SOURCE_CLOSURE:
        if _abs(source).is_file():
            present[source] = _md5_file(source)
        else:
            absent.append(source)
    return present, absent


def _runtime_block() -> dict:
    """Interpreter and numeric-stack versions, read from the environment."""
    packages = {}
    for name in RUNTIME_DISTRIBUTIONS:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "packages": packages,
        "read_via": "importlib.metadata.version",
        "verified": True,
        "note": (
            "these are pinned facts, not documentation: xgboost, torch and "
            "numpy all change simulation output, so a stage-1 result "
            "produced under different versions is a different result. "
            "pin_stage1.py --verify fails on the environment, and the fix is "
            "to restore the environment or re-pin deliberately"),
    }


def _git_head_short() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


# ---------------------------------------------------------------------------
# Command lines (D5 check 5.8)
# ---------------------------------------------------------------------------

def _env_for(arm: str, model_dir: str) -> dict:
    env = {
        "OMP_NUM_THREADS": str(THREADS),
        "MKL_NUM_THREADS": str(THREADS),
        "OPENBLAS_NUM_THREADS": str(THREADS),
        "VECLIB_MAXIMUM_THREADS": str(THREADS),
        "NUMEXPR_NUM_THREADS": str(THREADS),
    }
    if ARM_SPEC[arm]["model_type"] == "transformer":
        env["T1_SIM_MODEL_DIR"] = model_dir
        env["T1_SIM_DEVICE"] = DEVICE
        env["T1_EXTRAS_GRAFT_PATH"] = _p(EXTRAS_GRAFT)
        env["T1_SIM_PREFIX_CACHE"] = "<unset: prefix cache OFF>"
    return env


def _command(arm: str, *, model_dir: str, fixture_dir: str, n_sims,
             output_dir: str, base_seed=None) -> str:
    """One complete, repo-root-runnable invocation, every flag explicit."""
    prefix = ["env"]
    if ARM_SPEC[arm]["model_type"] == "transformer":
        prefix += ["-u", "T1_SIM_PREFIX_CACHE"]
    for key, value in _env_for(arm, model_dir).items():
        if value.startswith("<unset"):
            continue
        prefix.append(f"{key}={value}")
    return " ".join(prefix + [
        "uv", "run", "--no-sync", "python", RUNNER,
        "--arm", arm,
        # The runner re-verifies every pin in this file and refuses to launch
        # unless the invocation matches the arm block below.
        "--config", _p(CONFIG_PATH.relative_to(REPO_ROOT)),
        "--model-dir", model_dir,
        "--stats-version", ARM_SPEC[arm]["stats_version"],
        "--fixture-dir", _p(fixture_dir),
        "--context-dir", CONTEXT_DIR,
        "--odds", _role_path(ROLE_ODDS),
        "--player-metadata", PLAYER_METADATA,
        "--extras-graft", _p(EXTRAS_GRAFT),
        "--bowler-usage-path", _role_path(ROLE_BOWLER_USAGE),
        "--bowler-roster-policy", _role_path(ROLE_ROSTER_POLICY),
        "--clip-low", str(CLIP_BOUNDS[0]),
        "--clip-high", str(CLIP_BOUNDS[1]),
        "--n-sims", str(n_sims),
        "--base-seed", str(base_seed if base_seed is not None else BASE_SEED),
        "--threads", str(THREADS),
        "--output-dir", output_dir,
    ])


# ---------------------------------------------------------------------------
# The config
# ---------------------------------------------------------------------------

def _arm_block(arm: str, shared: dict, selection: dict) -> dict:
    spec = ARM_SPEC[arm]
    model_dir_role = spec.get("model_dir_role")
    if spec["model_type"] == "transformer":
        ablation_arm = spec["ablation_arm"]
        seed = selection[ablation_arm]["chosen_seed"]
        model_dir = _p(ABLATION_DIR / ablation_arm / f"seed_{seed}")
        checkpoint = _p(Path(model_dir) / "model.pt")
    elif model_dir_role:
        model_dir = _role_path(model_dir_role)
        checkpoint = _p(Path(model_dir) / spec["checkpoint_name"])
    else:
        model_dir = _p(spec["model_dir_composed"])
        checkpoint = _p(Path(model_dir) / spec["checkpoint_name"])

    stats_version = spec["stats_version"]
    stats_cache_role = ROLE_STATS_CACHE[stats_version]
    stats_cache = _role_path(stats_cache_role)
    full_n_sims = shared["full_run_n_sims"]
    output_full = _p(SEQ_STAGE1_ROOT / "full" / arm)
    output_smoke = _p(SEQ_STAGE1_ROOT / "smoke" / arm)

    block = {
        "status": "active",
        "role": spec["role"],
        "description": spec["description"],
        "model_type": spec["model_type"],
        "model_dir": model_dir,
        "model_dir_role": model_dir_role,
        "model_dir_md5": _md5_dir(model_dir),
        "model_dir_hash_contract": DIR_HASH_CONTRACT,
        "checkpoint": checkpoint,
        "checkpoint_md5": _md5_file(checkpoint),
    }
    if spec["model_type"] == "transformer":
        chosen = selection[spec["ablation_arm"]]
        block["ablation_arm"] = spec["ablation_arm"]
        block["checkpoint_seed"] = chosen["chosen_seed"]
        block["checkpoint_validation_ll"] = chosen["chosen_validation_ll"]
        block["checkpoint_selection"] = (
            "see checkpoint_selection: lowest validation LL in "
            f"{_p(ABLATION_SUMMARY)}, ties to the lowest seed, test LL not "
            "consulted")
    block.update({
        "stats_version": stats_version,
        "stats_cache": stats_cache,
        "stats_cache_role": stats_cache_role,
        "stats_cache_md5": shared["stats_cache_md5"][stats_version],
        "context_dir": CONTEXT_DIR,
        "context_dir_md5": shared["context_dir_md5"],
        "context_dir_hash_contract": DIR_HASH_CONTRACT,
        "context_dir_json_count": shared["context_dir_json_count"],
        "player_metadata": PLAYER_METADATA,
        "player_metadata_sha256": shared["player_metadata_sha256"],
        "bowler_selector": "RosterEmpiricalBowlerSelector",
        "bowler_selector_note": (
            "the phase-aware empirical selector with the corrected "
            "roster-aware policy; identical class and hashes on every arm"),
        "bowler_usage": _role_path(ROLE_BOWLER_USAGE),
        "bowler_usage_role": ROLE_BOWLER_USAGE,
        "bowler_usage_md5": shared["bowler_usage_md5"],
        "b10_corpus": _role_path(ROLE_B10_CORPUS),
        "b10_corpus_role": ROLE_B10_CORPUS,
        "b10_corpus_md5": shared["b10_corpus_md5"],
        "b10_corpus_note": (
            "the usage corpus behind bowler_phase_usage.json; pinned for "
            "provenance, not passed as a runtime flag"),
        "roster_policy": _role_path(ROLE_ROSTER_POLICY),
        "roster_policy_role": ROLE_ROSTER_POLICY,
        "roster_policy_md5": shared["roster_policy_md5"],
        "extras_graft": _p(EXTRAS_GRAFT),
        "extras_graft_role": None,
        "extras_graft_role_note": (
            "the B18 sidecar is unpromoted and has no manifest role, so its "
            "path is composed from segments and its identity is the sha256 "
            "below"),
        "extras_graft_sha256": shared["extras_graft_sha256"],
        "extras_graft_matches_report": EXTRAS_GRAFT_REPORT,
        "extras_graft_applied_via": (
            "T1_EXTRAS_GRAFT_PATH (sim_t1.py)"
            if spec["model_type"] == "transformer" else
            "the XGBoostModelV2.extras_graft attribute, set by "
            "run_arm._apply_extras_graft; sim_v1_2.py exposes no "
            "explicit-path seam (see deviations)"),
        # Deep-copied so each arm block is written out in full: a registered
        # config that leans on YAML anchors is harder to read and harder to
        # diff than one that repeats itself.
        "run_out": copy.deepcopy(shared["run_out"]),
        "odds": copy.deepcopy(shared["odds"]),
        "prop_fair_baseline_corpus": _role_path(ROLE_PROP_CORPUS),
        "prop_fair_baseline_corpus_role": ROLE_PROP_CORPUS,
        "prop_fair_baseline_corpus_md5": shared["prop_corpus_md5"],
        "fixture_set": copy.deepcopy(shared["fixture_set"]),
        "prefix_cache": "off",
        "device": DEVICE,
        "threads": THREADS,
        "clip": list(CLIP_BOUNDS),
        "base_seed": BASE_SEED,
        "env": _env_for(arm, model_dir),
        "full_run": {
            "fixture_dir": _role_path(ROLE_FIXTURE_SET),
            "fixture_dir_role": ROLE_FIXTURE_SET,
            "fixture_dir_md5": shared["fixture_set_md5"],
            "fixture_count": shared["fixture_set_count"],
            "n_sims": full_n_sims,
            "base_seeds": [BASE_SEED],
            "n_sims_source": shared["full_run_n_sims_source"],
            "output_dir": output_full,
            "command": _command(arm, model_dir=model_dir,
                                fixture_dir=_role_path(ROLE_FIXTURE_SET),
                                n_sims=shared["full_run_n_sims_token"],
                                output_dir=output_full),
            "command_completeness": shared["full_run_command_completeness"],
        },
        "smoke_1a": {
            "fixture_dir": _p(SMOKE_FIXTURE_DIR),
            "fixture_dir_md5": shared["smoke_fixture_dir_md5"],
            "fixture_count": shared["smoke_fixture_count"],
            "n_sims": SMOKE_N_SIMS,
            "base_seeds": [BASE_SEED],
            "output_dir": output_smoke,
            "differs_from_full_run_only_in": [
                "--fixture-dir", "--n-sims", "--output-dir"],
            "command": _command(arm, model_dir=model_dir,
                                fixture_dir=SMOKE_FIXTURE_DIR,
                                n_sims=SMOKE_N_SIMS,
                                output_dir=output_smoke),
        },
        "timing_1b": {
            "purpose": (
                "the 1b convergence protocol and the variability rerun: the "
                "same timing shard simulated under each registered batch "
                "base seed at each candidate n_sims"),
            "fixture_dir": _p(TIMING_FIXTURE_DIR),
            "fixture_dir_status": shared["timing_fixture_dir_status"],
            "fixture_dir_md5": shared["timing_fixture_dir_md5"],
            "fixture_count": shared["timing_fixture_count"],
            "fixture_count_expected": TIMING_SHARD_SIZE,
            "base_seeds": list(BATCH_BASE_SEEDS),
            # The PERMITTED list is the jointly screened one, not the whole
            # ladder: a candidate whose intervals collide across batches is
            # not launchable, and both run_arm and audit_cross_arm enforce
            # this list (Astra round 3, item 2).
            "n_sims": shared["joint_permitted_candidates"],
            "candidates_screened": list(CONVERGENCE_CANDIDATES),
            "n_sims_source": (
                "seeds.overlap_check.joint: every screened candidate whose "
                "(fixture, base seed) intervals stay disjoint across all "
                f"{len(BATCH_BASE_SEEDS)} batch base seeds; the rest are "
                "recorded as not_permitted_without_new_seeds"),
            "output_dir_template": _p(
                SEQ_STAGE1_ROOT / "timing" / arm
                / "seed<base_seed>_n<n_sims>"),
            "seed_cohort": (
                "the three batch base seeds share one seed line, so run_arm "
                "screens their JOINT intervals for any launch on this block "
                "and refuses a candidate above "
                f"{shared['joint_min_seed_gap']}"),
            "command_template": _command(
                arm, model_dir=model_dir, fixture_dir=TIMING_FIXTURE_DIR,
                n_sims="<candidate>", output_dir=_p(
                    SEQ_STAGE1_ROOT / "timing" / arm
                    / "seed<base_seed>_n<n_sims>"),
                base_seed="<batch_base_seed>"),
        },
        "variability_rerun": {
            "what": (
                "the 1b Monte Carlo variability rerun: the timing shard at "
                "the chosen n_sims, re-run under the second registered base "
                f"seed {SECOND_BASE_SEED} (and the third, "
                f"{THIRD_BATCH_BASE_SEED}, for the third batch)"),
            "runs_through": "the timing_1b block above",
            "base_seeds": [SECOND_BASE_SEED, THIRD_BATCH_BASE_SEED],
            "reads": (
                "the paired spread across batches, which is what the "
                "convergence stop rule thresholds"),
        },
    })
    return block


def build_config() -> dict:
    selection = selection_table(_abs(ABLATION_SUMMARY))
    odds = _odds_block()
    extras_sha = _sha256(EXTRAS_GRAFT)
    _require(extras_sha, EXTRAS_GRAFT_SHA256,
             f"{EXTRAS_GRAFT} sha256 (PPC report identity)")

    fixture_set = _role_path(ROLE_FIXTURE_SET)
    fixture_count = _json_count(fixture_set)
    _require(fixture_count, FIXTURE_SET_EXPECTED_COUNT,
             f"{fixture_set} fixture count")
    fixture_ids = fixture_ids_for(fixture_set)
    joint_overlap = joint_overlap_block(fixture_ids, BATCH_BASE_SEEDS)

    shared = {
        "stats_cache_md5": {
            version: _md5_file(_role_path(role))
            for version, role in ROLE_STATS_CACHE.items()},
        "context_dir_md5": _md5_dir(CONTEXT_DIR),
        "context_dir_json_count": _json_count(CONTEXT_DIR),
        "player_metadata_sha256": _sha256(PLAYER_METADATA),
        "bowler_usage_md5": _md5_file(_role_path(ROLE_BOWLER_USAGE)),
        "b10_corpus_md5": _md5_file(_role_path(ROLE_B10_CORPUS)),
        "roster_policy_md5": _md5_file(_role_path(ROLE_ROSTER_POLICY)),
        "extras_graft_sha256": extras_sha,
        "prop_corpus_md5": _md5_file(_role_path(ROLE_PROP_CORPUS)),
        "run_out": _runout_rule(),
        "odds": odds,
        "fixture_set": {
            "path": fixture_set,
            "role": ROLE_FIXTURE_SET,
            "dir_md5": _md5_dir(fixture_set),
            "dir_hash_contract": DIR_HASH_CONTRACT,
            "count": fixture_count,
        },
        "fixture_set_md5": _md5_dir(fixture_set),
        "fixture_set_count": fixture_count,
        "smoke_fixture_dir_md5": _md5_dir(SMOKE_FIXTURE_DIR),
        "smoke_fixture_count": _json_count(SMOKE_FIXTURE_DIR),
        # The 1b timing shard is chosen and copied at 1b, so it may be
        # absent now. It is registered either way: an absent directory is
        # recorded as absent, never silently skipped.
        "timing_fixture_dir_md5": (
            _md5_dir(TIMING_FIXTURE_DIR)
            if _abs(TIMING_FIXTURE_DIR).is_dir() else None),
        "timing_fixture_count": (
            _json_count(TIMING_FIXTURE_DIR)
            if _abs(TIMING_FIXTURE_DIR).is_dir() else None),
        "timing_fixture_dir_status": (
            "present" if _abs(TIMING_FIXTURE_DIR).is_dir()
            else "absent: populated at 1b, re-pin after it is built"),
        "joint_min_seed_gap": joint_overlap["joint_min_seed_gap"],
        "joint_permitted_candidates": [
            row["n_sims"] for row in joint_overlap["rows"] if row["disjoint"]],
        "full_run_n_sims": (
            CHOSEN_N_SIMS if CHOSEN_N_SIMS is not None
            else CONVERGENCE_PLACEHOLDER),
        "full_run_n_sims_token": (
            str(CHOSEN_N_SIMS) if CHOSEN_N_SIMS is not None else N_SIMS_TOKEN),
        "full_run_n_sims_source": (
            "convergence_protocol.chosen_n_sims, set from the 1b replicated "
            "convergence check; pin_stage1.py renders --n-sims from the same "
            "constant, so the command and the protocol cannot disagree"),
        "full_run_command_completeness": (
            f"every flag and env var is final except the single token "
            f"'{N_SIMS_TOKEN}', which is unresolved until 1d by design "
            "(check 5.6 forbids choosing n_sims before the convergence "
            "check). Set CHOSEN_N_SIMS in pin_stage1.py and re-run --write "
            "to render a runnable command."
            if CHOSEN_N_SIMS is None else
            "complete and runnable from the repo root as written"),
    }

    worked = []
    for path in sorted(_abs(SMOKE_FIXTURE_DIR).glob("*.json")):
        seed = fixture_seed(path.stem, BASE_SEED)
        worked.append({
            "cricsheet_id": path.stem,
            "base_seed": BASE_SEED,
            "fixture_seed": seed,
            "sub_seeds": {name: sub_seed(seed, name)
                          for name in SUB_SEED_STREAMS},
        })

    config = {}

    config["experiment"] = {
        "name": "seq_stage1_sim_v1",
        "registered": REGISTERED_DATE,
        "purpose": (
            "Run four ball models (A production i7, A50 the same family at "
            "equal information, B token MLP, C full T1) through ONE "
            "simulator, one chronology, one set of same-day replay cutoffs, "
            "one bowler selector, one extras law and one per-fixture seed "
            "schedule, so the winner-market comparison isolates the model "
            "and nothing else. Two questions: system comparison (B-A, C-A) "
            "and incremental sequence value in rollout (C-B)."),
        "plan": 'docs/SEQUENCE_TRACK_PLAN.md, sections "Stage 0" and "Stage 1"',
        "acceptance": 'docs/sequence_track/stage0_acceptance.md, section "D5"',
        "statistical_rules": (
            'docs/SEQUENCE_TRACK_PLAN.md, section "Statistical rules for the '
            'whole track"'),
        "design": "single-checkpoint screen; one training seed per neural arm",
        "expected_outcome_written_before_the_run": (
            "A-C parity, B-C parity, realism close on all arms; no arm "
            "advances"),
        "forbidden_data": ["data/golden", "data/forward_holdout"],
        "generated_by": "scripts/sequence_track/pin_stage1.py",
        "generated_note": (
            "this file is generated; edit pin_stage1.py and re-run --write, "
            "never the YAML by hand (pin_stage1.py --verify fails on a hand "
            "edit)"),
    }

    config["arms"] = {
        arm: _arm_block(arm, shared, selection) for arm in ("A", "A50", "B", "C")
    }
    config["arms"]["C114"] = {
        "status": "deferred",
        "role": "sequence beyond hand-built history (optional, after A50)",
        "description": (
            "full T1 given the production 114 features, to test whether "
            "learned sequence adds anything beyond the 9 hand-built history "
            "columns production already carries"),
        "reason_deferred": (
            "no checkpoint exists: a 114-feature T1 has never been trained, "
            "and training one is a stage-2 sized job, not a stage-1 screen. "
            "It is also the wrong order — A50 answers the equal-information "
            "question first, and C114 only becomes informative if a "
            "50-feature sequence arm shows something. No artifact, hash, "
            "seed or command is registered for it, so it cannot be run "
            "under this config."),
        "artifacts": None,
    }

    # Arm-independent by construction: the cross-arm audit asserts these are
    # byte-identical across A, A50, B and C, and a difference is a failure,
    # not a finding.
    config["selector"] = {
        "class": "RosterEmpiricalBowlerSelector",
        "identical_across_arms": True,
        "asserted_by": (
            "scripts/sequence_track/audit_cross_arm.py, which fails when any "
            "arm's selector class, usage hash or roster hash differs"),
        "bowler_usage": _role_path(ROLE_BOWLER_USAGE),
        "bowler_usage_role": ROLE_BOWLER_USAGE,
        "bowler_usage_md5": shared["bowler_usage_md5"],
        "roster_policy": _role_path(ROLE_ROSTER_POLICY),
        "roster_policy_role": ROLE_ROSTER_POLICY,
        "roster_policy_md5": shared["roster_policy_md5"],
        "b10_corpus": _role_path(ROLE_B10_CORPUS),
        "b10_corpus_role": ROLE_B10_CORPUS,
        "b10_corpus_md5": shared["b10_corpus_md5"],
        "k": 30,
        "enabled_by": (
            f"run_arm.py --bowler-usage-path {_role_path(ROLE_BOWLER_USAGE)} "
            f"--bowler-roster-policy {_role_path(ROLE_ROSTER_POLICY)}; "
            "WITHOUT the roster flag run_arm builds a plain "
            "EmpiricalBowlerSelector and no roster policy is applied, so the "
            "flag is not optional"),
        "why": (
            "the plan requires the corrected roster-aware policy: the B10 "
            "selector used 7.5-7.9 distinct bowlers against 6.0-6.1 "
            "observed"),
    }

    config["checkpoint_selection"] = {
        "rule": (
            "lowest validation LL in "
            f"{_p(ABLATION_SUMMARY)} per arm; ties break to the lowest seed "
            "number; test LL is not consulted"),
        "computed_by": "scripts/sequence_track/pin_stage1.py (not hardcoded)",
        "source": _p(ABLATION_SUMMARY),
        "source_role": None,
        "source_role_note": (
            "the T1 ablation lineage is an unpromoted research artifact with "
            "no manifest role; its path is composed from segments and its "
            "identity is the sha256 below"),
        "source_sha256": _sha256(ABLATION_SUMMARY),
        "split": "validation",
        "split_rows": 124292,
        "per_seed_validation_ll": {
            ablation_arm: selection[ablation_arm]["per_seed"]
            for ablation_arm in ("mlp", "full")
        },
        "chosen": {
            "B": {
                "ablation_arm": "mlp",
                "seed": selection["mlp"]["chosen_seed"],
                "validation_ll": selection["mlp"]["chosen_validation_ll"],
                "model_dir": config["arms"]["B"]["model_dir"],
                "checkpoint": config["arms"]["B"]["checkpoint"],
            },
            "C": {
                "ablation_arm": "full",
                "seed": selection["full"]["chosen_seed"],
                "validation_ll": selection["full"]["chosen_validation_ll"],
                "model_dir": config["arms"]["C"]["model_dir"],
                "checkpoint": config["arms"]["C"]["checkpoint"],
            },
        },
    }

    config["seeds"] = {
        "base_seed": BASE_SEED,
        "second_base_seed": SECOND_BASE_SEED,
        "second_base_seed_use": (
            "the 1b variability rerun: the timing shard is re-run with this "
            "base seed to measure paired Monte Carlo variability"),
        "per_fixture_rule": (
            "low 31 bits of sha256(f'{cricsheet_id}:{base_seed}')"),
        "per_fixture_implementation": (
            "int.from_bytes(hashlib.sha256("
            "f'{cricsheet_id}:{int(base_seed)}'.encode()).digest(), 'big') "
            "& ((1 << 31) - 1)"),
        "substream_rule": (
            "low 31 bits of sha256(f'{fixture_seed}:{name}')"),
        "substream_implementation": (
            "int.from_bytes(hashlib.sha256("
            "f'{int(fixture_seed)}:{name}'.encode()).digest(), 'big') "
            "& ((1 << 31) - 1)"),
        "substreams": list(SUB_SEED_STREAMS),
        "implemented_by": (
            "scripts/sequence_track/run_arm.py functions fixture_seed and "
            "sub_seed; scripts/tests/test_pin_stage1.py asserts this "
            "config's documented formula reproduces them"),
        "evaluator_seam": (
            "MatchLevelEvaluator._simulation_seed in "
            "scripts/sim_eval/match_evaluator.py, overridden by "
            "run_arm._ArmEvaluator; the legacy fixed LEGACY_SIMULATION_SEED "
            "= 42 is not used by any stage-1 arm"),
        "consumed_as": "SimulationConfig.random_seed, one value per fixture",
        "substream_status": (
            "the three named sub-seeds are derived and written to "
            "arm_provenance.json but the engine cannot consume them "
            "separately — see deviations.single_global_rng_stream"),
        "worked_example": worked,
        "interval_mechanism": (
            "the engine seeds simulation i of a fixture with "
            "SimulationConfig.random_seed + i, so ONE fixture occupies the "
            "half-open seed interval [fixture_seed, fixture_seed + n_sims). "
            "If two fixtures' intervals intersect, the simulations at the "
            "shared seeds replay the same draw schedule: those two "
            "fixtures' Monte Carlo errors are then correlated inside every "
            "arm. The chance of some collision GROWS with n_sims (about "
            "N^2 * n_sims / 2^31 for N fixtures), so it cannot be dismissed "
            "for the larger convergence candidates, and it does NOT cancel "
            "in a paired contrast — the draws are shared, but each arm's "
            "model maps them to a different match, so the induced error is "
            "model-dependent"),
        "interval_guard": (
            "run_arm.seed_intervals_disjoint is called in preflight over the "
            "run's own fixture set and REFUSES to launch on any overlap, "
            "naming the pair; pin_stage1.py --check-seed-overlap --n-sims N "
            "screens a candidate over the full registered fixture set. The "
            "table below is that screen for both registered base seeds"),
        "overlap_check": {
            "fixture_dir": fixture_set,
            "fixture_dir_role": ROLE_FIXTURE_SET,
            "fixture_count": fixture_count,
            "candidates": list(CONVERGENCE_CANDIDATES),
            "implementation": (
                "scripts/sequence_track/run_arm.py seed_intervals_disjoint, "
                "imported by this script so the screen and the runtime "
                "refusal are one implementation"),
            "by_base_seed": {
                str(seed): overlap_rows(fixture_ids, seed)
                for seed in BATCH_BASE_SEEDS
            },
            "joint": joint_overlap,
            "joint_note": (
                "the per-base-seed rows above are NOT sufficient: the three "
                "1b batches run the same fixtures on one seed line, so the "
                "screen that governs a convergence candidate is the joint "
                "one. run_arm screens the joint cohort for any block that "
                "registers several base seeds"),
        },
    }

    config["clipping"] = {
        "bounds": list(CLIP_BOUNDS),
        "seam": (
            "MatchLevelEvaluator.prob_clip, applied at the single seam "
            "MatchLevelEvaluator._clip_and_normalize"),
        "seam_source": "scripts/sim_eval/match_evaluator.py",
        "applied_where": (
            "after the tie-free renormalisation and before log loss, Brier, "
            "edge and bet placement, on both evaluation paths"),
        "default": (
            "prob_clip = None, which reproduces the historical "
            "DEFAULT_PROB_CLIP = [0.05, 0.95] exactly, so every existing "
            "runner and every stored BR2 result is unchanged"),
        "replaces_not_composes": (
            "the seam REPLACES the evaluator's historical unconditional "
            "[0.05, 0.95] clip; it does not compose with it. The registered "
            "stage-1 clip is therefore WIDER than every previous run's, "
            "including BR2 G1, so a stage-1 winner log loss is not "
            "comparable to any number produced under the [0.05, 0.95] clip"),
        "enabled_by": "run_arm.py --clip-low 0.01 --clip-high 0.99",
        "why": (
            "at 100 sims the plug-in winner probability has SE 0.05 at "
            "p = 0.5; a 0/100 or 100/100 draw must not cost an unbounded "
            "log loss, and a 0.05 floor would hide real confidence at the "
            "resolution stage 1 is trying to measure"),
    }

    config["convergence_protocol"] = {
        "candidates": (
            "n_sims doubles from 100 with no ceiling: 100, 200, 400, 800, ..."),
        "first_candidate": CONVERGENCE_FIRST_CANDIDATE,
        "batches_per_arm_per_candidate": 3,
        "batch_base_seeds": [
            BASE_SEED, SECOND_BASE_SEED, THIRD_BATCH_BASE_SEED],
        "batch_independence": (
            "three independent simulation batches per arm per candidate, one "
            "per registered batch base seed; the fixture set, chronology, "
            "selector and extras law are identical across batches"),
        "shard": (
            "the 1b timing shard: 10 fixtures chosen to include long innings"),
        "statistic": (
            "the paired C-B and B-A winner log loss on the shard, computed "
            "per batch"),
        "stop_rule": (
            "stop at the smallest n_sims at which the 95% range of the "
            "paired C-B and B-A winner LL across the three batches is below "
            f"{CONVERGENCE_THRESHOLD}"),
        "threshold": CONVERGENCE_THRESHOLD,
        "chosen_n_sims": shared["full_run_n_sims"],
        "chosen_n_sims_drives": (
            "the --n-sims token in every arm's full_run.command; both are "
            "rendered from pin_stage1.CHOSEN_N_SIMS"),
        "spread_table_columns": [
            "n_sims", "contrast", "batch_1_delta_ll", "batch_2_delta_ll",
            "batch_3_delta_ll", "range_95", "below_threshold"],
        "spread_table": [],
        "fill_before": "1d (the full 255-fixture run)",
    }

    config["decision_rule"] = {
        "primary_slice": ">=$50k (--min-volume 50000)",
        "reported_slices": ["all", "50000", "100000"],
        "sign_convention": (
            "every contrast is candidate minus reference, in winner log "
            "loss, so a negative delta is favourable to the candidate"),
        "equivalence_margin_ll": EQUIVALENCE_MARGIN_LL,
        "margin_basis": (
            "the branch-wide seed-noise floor (claim_gate SEED_FLOOR = "
            "0.007); a difference below it is not resolvable by this "
            "protocol"),
        "confirmatory_family": [
            {"contrast": "C-B", "candidate": "C", "reference": "B",
             "slice": "50000",
             "question": "incremental sequence value in rollout",
             "decision_it_feeds": (
                 "whether learned sequence adds anything over the same "
                 "features without sequence memory; it is evidence, not by "
                 "itself an advancement")},
            {"contrast": "B-A", "candidate": "B", "reference": "A",
             "slice": "50000",
             "question": "system comparison, token MLP vs production",
             "decision_it_feeds": "B advances only if this is favourable"},
            {"contrast": "C-A", "candidate": "C", "reference": "A",
             "slice": "50000",
             "question": "system comparison, full T1 vs production",
             "decision_it_feeds": "C advances only if this is favourable"},
        ],
        "multiplicity": (
            "Holm adjustment across the three confirmatory contrasts "
            "(C-B, B-A, C-A); the family is fixed here, before any run"),
        "exploratory_contrasts": [
            {"contrast": "A50-A", "candidate": "A50", "reference": "A",
             "slice": "50000",
             "label": "exploratory system contrast",
             "note": (
                 "A50 is the equal-information control for the A-vs-B/C "
                 "comparison; it is outside the confirmatory family, is not "
                 "Holm-adjusted, and cannot advance anything")},
        ],
        "outcomes": {
            "parity": (
                "the 95% block interval lies entirely inside "
                "[-0.007, +0.007]"),
            "favourable": (
                "the 95% block interval excludes zero and the point estimate "
                "is below -0.007"),
            "adverse": (
                "the 95% block interval excludes zero and the point estimate "
                "is above +0.007"),
            "inconclusive": "anything else",
        },
        "advancement": (
            "symmetric: B advances only if B-A is favourable, C advances "
            "only if C-A is favourable; a favourable C-B on its own is "
            "sequence evidence for stage 2, not an advancement"),
        "market_claim": (
            "no winner-LL result is a market claim. A market claim "
            "additionally requires the claim gate's market comparison with "
            "the registered cost scenarios below, and a paired Δprofit or "
            "ΔROI interval, per program.md's LANDED rule"),
        "cost_scenarios": list(COST_SCENARIOS),
        "estimands_reported_separately": [
            "the registered checkpoint's own performance, with uncertainty "
            "from paired tournament-block resampling of matches and "
            "independent simulation batches",
            "across-training-seed robustness, which this single-checkpoint "
            "screen does NOT measure and must not be read from it",
        ],
    }

    config["scoring"] = {
        "winner_log_loss": {
            "status": "confirmatory",
            "gate": "scripts/sim_eval/claim_gate.py",
            "kind": "match_model",
            "odds_role": ROLE_ODDS,
            "cluster_source_dir": odds["cluster_source_dir"],
            "cluster_source_dir_role": ROLE_FIXTURE_SET,
            "slices": ["all", "50000", "100000"],
            "uncertainty": (
                "tournament_time_block_v1: 10,000 seed-42 whole-event "
                "resamples with explicit bet placement; fewer than 10 blocks "
                "is descriptive, never a claim"),
            "prices": (
                "the gate recomputes prices, placement and profit from the "
                "registered odds role; stored prices in an eval JSON are "
                "never evidence"),
            "cost_scenarios": list(COST_SCENARIOS),
            "command_template": (
                "uv run --no-sync python scripts/sim_eval/claim_gate.py "
                "--kind match_model --candidate <candidate arm dir>/"
                "sliced_50000.json --baseline <reference arm dir>/"
                f"sliced_50000.json --odds-role {ROLE_ODDS} "
                f"--cluster-source-dir {odds['cluster_source_dir']} "
                "--out <gate output>.json"),
        },
        "realism": {
            "status": "exploratory",
            "scorer": "scripts/sequence_track/score_realism.py",
            "input": "<arm output dir>/raw_sims.jsonl",
            "output": "<arm output dir>/realism.json",
            "metrics": [
                "innings score P10/P50/P90 coverage of the actual innings",
                "first-innings score bias",
                "extras per innings",
                "wickets per innings",
                "unique bowlers used per innings",
                "batting-first minus chasing flip on the PPC cohort",
            ],
            "note": (
                "read after the confirmatory contrasts; no realism number "
                "advances an arm or supports a claim"),
        },
        "props": {
            "status": "exploratory",
            "scorer": "scripts/sequence_track/score_props.py",
            "baseline": "scripts/sim_eval/prop_fair_baselines.py",
            "corpus": _role_path(ROLE_PROP_CORPUS),
            "corpus_role": ROLE_PROP_CORPUS,
            "corpus_md5": shared["prop_corpus_md5"],
            "input": "<arm output dir>/raw_sims.jsonl",
            "output": "<arm output dir>/props.json",
            "metric": (
                "paired per-family Brier (mean absolute error for the four "
                "*_mae families), sim minus as-of fair baseline, per arm"),
            "bar": (
                "E2 v2: no binary prop family beats an as-of fair baseline, "
                "so a prop claim must clear the versioned fair baseline, "
                "never a base rate"),
            "families": list(PROP_FAMILIES),
            "note": (
                "prop_backtest.py builds its own XGBoostModelV2 and has no "
                "T1 seam, so the stage-1 prop scorer consumes the arm "
                "runner's own per-simulation output instead of "
                "re-simulating"),
        },
    }

    config["deviations"] = [
        {
            "id": "single_global_rng_stream",
            "what": (
                "the plan registers three named RNG sub-streams (outcome, "
                "extras, selector); the engine seeds ONE global "
                "random/np.random stream per simulation "
                "(SimulationConfig.random_seed + i) and outcome sampling, "
                "extras, selector draws and run-outs all draw from it"),
            "why": (
                "separating the streams needs an engine-wide RNG refactor of "
                "sim_v1_2.SimulationEngine, which is the BR2-gated engine; "
                "stage 1 must not change the engine it is measuring"),
            "consequence": (
                "the sub-seeds are derived and recorded in "
                "arm_provenance.json but are not separately consumed. Every "
                "arm still shares one identical draw schedule per fixture, "
                "which is what the paired comparison needs; what is lost is "
                "the ability to hold, say, the extras draws fixed while the "
                "outcome draws move"),
            "second_consequence": (
                "the engine's additive scheme (random_seed + i for the i-th "
                "simulation) makes ONE fixture occupy the seed interval "
                "[fixture_seed, fixture_seed + n_sims). Two fixtures whose "
                "intervals intersect replay the same draw schedule at the "
                "shared seeds, so their Monte Carlo errors are correlated "
                "inside every arm and the run has fewer independent draws "
                "than n_sims x fixtures suggests. The chance of some "
                "collision GROWS with n_sims (about N^2 * n_sims / 2^31 for "
                "N fixtures), and it does NOT cancel in a paired contrast: "
                "the draws are shared, but each arm's model maps them to a "
                "different match, so the induced error is model-dependent. "
                "This is guarded, not assumed away — run_arm refuses to "
                "launch on any overlap in its own fixture set, and "
                "seeds.overlap_check screens every convergence candidate "
                "over the full registered set for every registered base "
                "seed"),
            "status": (
                "stream separation deferred; the interval overlap it causes "
                "is screened and refused at launch"),
        },
        {
            "id": "extras_graft_attribute_seam",
            "what": (
                "the B18 extras sidecar reaches the XGBoost arms (A, A50) "
                "through the public XGBoostModelV2.extras_graft attribute "
                "set by run_arm._apply_extras_graft, not through a "
                "constructor argument"),
            "why": (
                "XGBoostModelV2 exposes no explicit-path seam for the graft; "
                "it only auto-detects an extras_graft_v1.json sitting beside "
                "the model artifact, and neither production model directory "
                "carries one. The attribute is what the engine reads and "
                "what the T1 wrapper sets from T1_EXTRAS_GRAFT_PATH, so both "
                "families end up on the same law without editing sim_v1_2.py"),
            "consequence": (
                "run_arm refuses to run if a model directory carries its own "
                "sidecar that differs from the requested one, so an "
                "ambiguous extras law fails closed rather than passing "
                "silently"),
            "status": "accepted",
        },
        {
            "id": "run_out_sidecar_absent",
            "what": (
                "the plan lists a run-out sidecar among the pinned "
                "artifacts; models/auto/d15/runout_rates.json is absent on "
                "this checkout"),
            "why": (
                "the run-out rate is a code constant, sim_v1_2.RUNOUT_P; the "
                "sidecar the engine banner cites was never committed"),
            "consequence": (
                "the config pins the constant, its value read by import, and "
                "its source line, and the engine md5 in provenance covers "
                "any change to it"),
            "status": "accepted",
        },
        {
            "id": "scorers_consume_raw_sims",
            "what": (
                "the realism and prop scorers read the arm runner's "
                "raw_sims.jsonl instead of re-simulating through "
                "prop_backtest.py"),
            "why": (
                "prop_backtest.py constructs XGBoostModelV2 directly and has "
                "no T1 seam, so it cannot score B or C at all; "
                "re-simulating would also break the identical-draw-schedule "
                "guarantee"),
            "consequence": (
                "realism and prop numbers describe exactly the simulations "
                "the winner-LL numbers came from; they are not comparable "
                "line for line with historical prop_backtest.py reports"),
            "status": "accepted",
        },
        {
            "id": "no_parallel_within_a_run",
            "what": (
                "run_arm.py refuses --parallel; sharding across processes is "
                "the only parallelism"),
            "why": (
                "the same-day replay lifecycle is strictly sequential — it "
                "orders every stats read — and stage 1 needs one "
                "deterministic prediction-time cutoff per fixture"),
            "consequence": (
                "the 1c shard-consistency check exists precisely because "
                "sharding is the only way to use more than one core; each "
                "shard keeps the full replay context"),
            "status": "accepted",
        },
        {
            "id": "clip_seam_replaces_rather_than_composes",
            "what": (
                "MatchLevelEvaluator.prob_clip REPLACES the evaluator's "
                "historical unconditional [0.05, 0.95] clip; it does not "
                "compose with it, so stage 1 runs at a strictly wider clip "
                "than every earlier run"),
            "why": (
                "composing would leave the 0.05 floor in force and make the "
                "registered [0.01, 0.99] bound decorative; replacing is the "
                "only way to measure at the resolution the plan asks for"),
            "consequence": (
                "no stage-1 winner log loss is comparable to any number "
                "produced under the [0.05, 0.95] clip, BR2 G1 included. "
                "prob_clip = None still reproduces the legacy behaviour "
                "exactly, so no existing runner or stored result moves"),
            "status": "accepted",
        },
        {
            "id": "resolved_flag_reads_cricsheet_not_the_odds_row",
            "what": (
                "the `resolved` eligibility flag is read from the cricsheet "
                "document's info.outcome.winner, not from the odds row's "
                "actual_winner field"),
            "why": (
                "eligibility must be identical across arms by construction, "
                "and the cricsheet document is the one input every arm sees "
                "identically; an odds-row field could differ if the odds "
                "loader ever changed"),
            "consequence": (
                "a fixture the odds file calls settled but cricsheet records "
                "with no winner (a no-result or an abandoned match) counts "
                "as unresolved for every arm alike. The cross-arm audit "
                "tests a real invariant rather than a copied field"),
            "status": "accepted",
        },
        {
            "id": "as_of_stamp_for_odds_less_fixtures",
            "what": (
                "for a fixture with no odds row there is no prediction lock, "
                "so its as-of stamp is taken at begin_match instead of at "
                "lock_prediction"),
            "why": (
                "the runner only locks a prediction when one is required; "
                "nothing advances between begin_match and lock_prediction, "
                "so the begin_match snapshot IS the state at the point a "
                "lock would have happened"),
            "consequence": (
                "the two stamps are equal wherever both exist, and the audit "
                "compares them across arms uniformly; an odds-less fixture "
                "is still stamped and still audited"),
            "status": "accepted",
        },
        {
            "id": "roster_selector_installed_by_factory_swap",
            "what": (
                "run_sim_eval.py has no roster-policy flag, so run_arm swaps "
                "the module-level EmpiricalBowlerSelector for a factory that "
                "builds RosterEmpiricalBowlerSelector when "
                "--bowler-roster-policy is given"),
            "why": (
                "adding a flag to the frozen runner would change the runner "
                "stage 1 is meant to leave alone; the factory swap is "
                "confined to the arm runner"),
            "consequence": (
                "WITHOUT --bowler-roster-policy no roster policy is applied "
                "and the arm silently runs the plain empirical selector, "
                "which is NOT the registered configuration. Every command "
                "line in this config carries the flag, and the cross-arm "
                "audit asserts the selector class and roster hash match "
                "across arms"),
            "status": "accepted",
        },
    ]

    config["known_asymmetries"] = [
        {
            "id": "stats_cache_i7_vs_v3",
            "what": (
                "A and A50 serve from the i7 stats cache "
                f"({_role_path(ROLE_STATS_CACHE['i7'])}); B and C serve from "
                f"the legacy v3 cache "
                f"({_role_path(ROLE_STATS_CACHE['v3'])})"),
            "why": (
                "the ablation checkpoints were trained on data/xgb_data_v3 "
                "and sim_t1.py refuses any other frame"),
            "consequence": (
                "B-A and C-A are system contrasts across two caches, not "
                "model-only contrasts. C-B is within one cache and is the "
                "only clean incremental comparison"),
            "removable_in_stage_0": False,
        },
        {
            "id": "training_frame_i7_vs_v3",
            "what": (
                "A and A50 are trained on the i7 identity frame "
                "(data/xgb_data_i7); B and C are trained on the pre-I7 v3 "
                "frame (data/xgb_data_v3, 467 raw venue strings)"),
            "why": (
                "the same constraint: the ablation lineage predates the I7 "
                "identity contract, and the v3 frame fail-closes under it, "
                "so those checkpoints cannot be retrained on i7 without a "
                "new training run outside stage 1's scope"),
            "consequence": (
                "any A-vs-B or A-vs-C difference confounds venue identity "
                "resolution with the model; A50 narrows the feature-set part "
                "of that confound but not the frame part"),
            "removable_in_stage_0": False,
        },
        {
            "id": "arm_a_path_is_the_replay_lifecycle",
            "what": (
                "arm A runs through the T1 replay lifecycle "
                "(SameDayReplayStatsProvider + _T1ReplayEvaluator via "
                "run_arm.py), not through the BR2 snapshot runner it was "
                "measured on"),
            "why": (
                "identical prediction-time cutoffs across arms are only "
                "possible if one runner drives every arm; the replay "
                "provider's tracker view implements every provider method "
                "XGBoostModelV2's feature builder calls"),
            "consequence": (
                "BR2's G1 winner-LL line is NOT a stage-1 reference number "
                "for arm A. The only valid reference for every stage-1 "
                "contrast is arm A's own stage-1 run under this config"),
            "removable_in_stage_0": False,
        },
        {
            "id": "br2_g1_is_not_a_stage_1_number",
            "what": (
                "BR2 G1 ran arm A's model with the plain "
                "EmpiricalBowlerSelector, the snapshot stats provider, and "
                "the evaluator's unconditional [0.05, 0.95] clip. Stage 1 "
                "runs the same model with RosterEmpiricalBowlerSelector, the "
                "SameDayReplayStatsProvider, and the registered "
                "[0.01, 0.99] clip"),
            "why": (
                "three separate stage-1 requirements force the change: the "
                "plan's roster-aware selector, one replay chronology shared "
                "by every arm, and a clip wide enough not to mask the "
                "resolution being measured"),
            "consequence": (
                "NO stage-1 number is comparable to BR2, in either "
                "direction. A stage-1 result that differs from BR2's G1 line "
                "is not evidence of anything; the only admissible baseline "
                "for B-A, C-A and A50-A is arm A's own run under this "
                "config"),
            "removable_in_stage_0": False,
        },
    ]

    source_md5, source_md5_absent = _source_closure()
    config["provenance"] = {
        "pins_generated_at": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "git_head_short": _git_head_short(),
        "not_verified": list(NOT_VERIFIED),
        "not_verified_reason": (
            "both change without the config's meaning changing (a re-pin "
            "moves the timestamp; the commit that lands this file moves "
            "HEAD), so --verify records them and does not compare them"),
        "config_sha256_self_reference": (
            "deliberately absent: a config cannot contain its own sha256. "
            "The integrity check is pin_stage1.py --verify, which recomputes "
            "every pinned fact from the files"),
        "source_closure_rule": (
            "every source file on the stage-1 execution path is hashed, not "
            "just the runner and the engine: replay and rehydration, the "
            "delegated runner, the loaders and chronology, the stats cache, "
            "the feature contract, the gate's statistics and market "
            "arithmetic, and both scorers with their reducers. A change to "
            "any of them moves stage-1 results, so a change to any of them "
            "must fail --verify"),
        "source_count": len(source_md5),
        "source_md5": source_md5,
        "source_md5_absent": source_md5_absent,
        "source_md5_absent_note": (
            "closure files not present on this checkout; they are dropped "
            "from the hash set and named here rather than skipped silently"),
        "runtime": _runtime_block(),
        "pinned_by": "scripts/sequence_track/pin_stage1.py",
    }

    _assert_no_sealed_paths(config)
    return config


def _assert_no_sealed_paths(config: dict) -> None:
    """D5 check 5.11: no sealed path outside experiment.forbidden_data."""
    declared = config["experiment"]["forbidden_data"]
    scrubbed = dict(config)
    scrubbed["experiment"] = {
        key: value for key, value in config["experiment"].items()
        if key != "forbidden_data"}
    text = json.dumps(scrubbed)
    for forbidden in declared:
        if forbidden in text:
            raise PinError(
                f"sealed path {forbidden!r} appears in the config outside "
                "experiment.forbidden_data")


# ---------------------------------------------------------------------------
# Write / verify
# ---------------------------------------------------------------------------

HEADER = """\
# Stage 1 registered configuration — sequence and embeddings track.
#
# GENERATED FILE. Written by `scripts/sequence_track/pin_stage1.py --write`
# and checked by `--verify`, which recomputes every hash, the checkpoint
# choice and the command lines from the files on disk and exits non-zero on
# any difference. Do not edit by hand: change pin_stage1.py and re-run.
#
# Acceptance: docs/sequence_track/stage0_acceptance.md, section "D5".
"""


def dump_config(config: dict) -> str:
    body = yaml.safe_dump(config, sort_keys=False, default_flow_style=False,
                          width=78, allow_unicode=True)
    return HEADER + body


def write_config(path: Path = CONFIG_PATH) -> dict:
    config = build_config()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_config(config))
    return config


def flatten(value, prefix: str = "") -> dict:
    """`{'a.b[0].c': leaf}` for a nested mapping/list payload.

    An EMPTY container becomes its own leaf. Without that, `spread_table: []`
    and `source_md5_absent: []` would contribute no key at all and a
    structural change (an empty list becoming an empty mapping, or the key
    disappearing entirely) could slip past `--verify`.
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


def diff_config(expected: dict, actual: dict) -> list[str]:
    """Human-readable differences, excluding the not-verified keys."""
    flat_expected = flatten(expected)
    flat_actual = flatten(actual)
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


def verify_config(path: Path = CONFIG_PATH) -> list[str]:
    if not Path(path).exists():
        return [f"MISSING  file: {path}"]
    try:
        actual = yaml.safe_load(Path(path).read_text())
    except yaml.YAMLError as exc:
        return [f"UNPARSEABLE {path}: {exc}"]
    if not isinstance(actual, dict):
        return [f"UNPARSEABLE {path}: top level is not a mapping"]
    return diff_config(build_config(), actual)


def print_selection(config: dict) -> None:
    block = config["checkpoint_selection"]
    print("checkpoint selection (validation LL only; ties -> lowest seed):")
    for arm, ablation_arm in (("B", "mlp"), ("C", "full")):
        chosen = block["chosen"][arm]
        print(f"  arm {arm} <- ablation arm {ablation_arm!r}")
        for row in block["per_seed_validation_ll"][ablation_arm]:
            mark = " <-- chosen" if row["seed"] == chosen["seed"] else ""
            print(f"    seed {row['seed']:>3}  "
                  f"validation LL {row['validation_ll']:.6f}{mark}")
        print(f"    chosen: seed {chosen['seed']}, "
              f"validation LL {chosen['validation_ll']:.6f}")
        print(f"    checkpoint: {chosen['checkpoint']}")


def print_overlap_check(n_sims=None) -> int:
    """Screen the registered fixture set for seed-interval overlap.

    Exits non-zero when any screened candidate overlaps under any registered
    base seed, because that base seed cannot be used at that n_sims.
    """
    fixture_set = _role_path(ROLE_FIXTURE_SET)
    fixture_ids = fixture_ids_for(fixture_set)
    joint_overlap = joint_overlap_block(fixture_ids, BATCH_BASE_SEEDS)
    candidates = [int(n_sims)] if n_sims else list(CONVERGENCE_CANDIDATES)
    print(f"seed-interval overlap screen over {len(fixture_ids)} fixtures "
          f"in {fixture_set}")
    print("the engine seeds simulation i with random_seed + i, so one "
          "fixture occupies [seed, seed + n_sims)")
    bad = 0
    for base_seed in BATCH_BASE_SEEDS:
        rows = overlap_rows(fixture_ids, base_seed, candidates)
        gap = rows[0]["min_seed_gap"]
        print(f"  base seed {base_seed} alone (closest two seeds {gap} "
              "apart):")
        for row in rows:
            status = "disjoint" if row["disjoint"] else (
                f"OVERLAP x{row['overlapping_pairs']} "
                f"({row['example_overlap']})")
            print(f"    n_sims {row['n_sims']:>6}  {status}")

    joint = joint_overlap_block(fixture_ids, BATCH_BASE_SEEDS, candidates)
    print(f"  JOINT over base seeds {joint['base_seeds']} — the 1b batches "
          "share one seed line")
    print(f"    joint minimum seed gap: {joint['joint_min_seed_gap']}")
    for row in joint["rows"]:
        status = "disjoint" if row["disjoint"] else (
            f"OVERLAP x{row['overlapping_pairs']} ({row['example_overlap']})")
        print(f"    n_sims {row['n_sims']:>6}  {status}")
        bad += 0 if row["disjoint"] else 1
    print(f"    largest permitted candidate: "
          f"{joint['largest_permitted_candidate']} "
          f"(any n_sims <= {joint['joint_min_seed_gap']})")
    if bad:
        print(f"{bad} candidate(s) are NOT permitted jointly and need new "
              "base seeds and a re-screen; run_arm refuses them at launch",
              file=sys.stderr)
    # A specific request is a yes/no question; the full sweep is a survey
    # that must be allowed to report the wall it finds.
    if n_sims is not None:
        return 1 if bad else 0
    if not joint["rows"][0]["disjoint"]:
        print("even the smallest candidate collides: these base seeds are "
              "unusable", file=sys.stderr)
        return 1
    print("permitted candidates are disjoint per base seed AND jointly")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true",
                       help="regenerate the config from the files on disk")
    group.add_argument("--verify", action="store_true",
                       help="recompute every pinned fact and diff the file")
    group.add_argument("--check-seed-overlap", action="store_true",
                       help="screen the registered fixture set for "
                            "seed-interval overlap at every convergence "
                            "candidate (or just --n-sims N)")
    parser.add_argument("--n-sims", type=int, default=None,
                        help="single n_sims candidate for "
                             "--check-seed-overlap")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args(argv)

    try:
        if args.check_seed_overlap:
            return print_overlap_check(args.n_sims)
        if args.write:
            config = write_config(args.config)
            print_selection(config)
            print(f"wrote {args.config}")
            return 0
        problems = verify_config(args.config)
    except PinError as exc:
        print(f"pin_stage1: ERROR: {exc}", file=sys.stderr)
        return 1
    if problems:
        print(f"pin_stage1: {len(problems)} mismatch(es) in {args.config}:",
              file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    config = yaml.safe_load(Path(args.config).read_text())
    print_selection(config)
    print("not compared (recorded only): " + ", ".join(NOT_VERIFIED))
    print(f"pin_stage1: OK — {args.config} matches every recomputed fact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
