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

`--write` is also the only thing that MATERIALISES the 1d shard partition
(D11 check 11.1): it copies the registered fixture set into ten
`models/embeddings/seq_stage1/full/shards/<k>/fixtures` directories by the
round-robin rule in `SHARD_RULE`, refusing a directory that already holds
something else, and then pins each shard's ids, count and directory hash
plus a per-shard command that differs from the whole-set `full_run` command
only in `--fixture-dir` and `--output-dir`. `--verify` recomputes the
partition from the fixture set, re-reads and re-hashes the ten directories,
and asserts their union is the registered set with no fixture in two shards.

Two provenance fields are recorded but never compared, because they
change without the config's meaning changing: `pins_generated_at` and
`git_head_short`. They are listed in the YAML itself under
`provenance.not_verified`, which IS compared.

Checkpoint selection (D5 check 5.3, rewired by D6 check 6.2) is computed
here, never hardcoded: for the `mlp` (arm B) and `full` (arm C) retrain
arms, the seed with the lowest VALIDATION log loss in the i7 retrain's
`summary.yaml` (`RETRAIN_SUMMARY`) wins, ties break to the lowest seed
number, and the test split is not read. The per-seed table must carry
exactly the five registered training seeds, each once, each finite and none
rejected by the rounded-value heuristic: a log loss indistinguishable from
one rounded to four decimals before it was written is refused (the heuristic
cannot prove full precision; that is established by the trainer's unrounded
writer and by comparing every summary value to its own metrics.json), because the seeds in this family are separated in
the fifth and sixth decimal and a rounded table would decide the selection
by a tie-break rather than by validation log loss.

Training facts are RECOMPUTED, never copied (Astra round 1, MUST-FIX 1).
Checking that a checkpoint's training contract merely has the right shaped
fields would let a training parquet be replaced while the contract, the
metrics, the summary and the YAML all still verified. So both `--write` and
`--verify`:

* hash `data/xgb_data_i7/cricket_data_i7_{train,validation}.parquet` and read
  their row counts and `match_date` ranges with pyarrow, and compare every
  one against the contract and against the retrain summary;
* compare the contract's and the summary's copies of the frame's
  `.feature_hash` against the file on disk, key by key;
* rebuild the 50 feature names from `embeddings_e1` / `transformer_t1` in
  the wrapper's construction order, compare them position by position, and
  recompute their sha256;
* re-hash the retrain config on disk; and
* compare every one of the ten per-seed log losses in `summary.yaml` against
  the `validation_ll` in that seed's own `metrics.json`.

Any disagreement is a `PinError` naming the key.

Usage:
    uv run --no-sync python scripts/sequence_track/pin_stage1.py --write
    uv run --no-sync python scripts/sequence_track/pin_stage1.py --verify
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import shutil
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
THREADS = 1  # 2026-09-11: was 4; the 1b probe showed threads buy nothing per process and block sharding
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
# 50 added 2026-09-11 (user decision for 1b): a noise-curve point only —
# at 50 simulations the plug-in log-loss bias is about 0.01, above the
# 0.007 floor, so 50 can never be the full-run count.
CONVERGENCE_CANDIDATES = [50, 100, 200, 400, 800, 1600, 3200, 6400]
TIMING_1B_CANDIDATES = [50, 100, 200, 400, 800]  # fixed by the user 2026-09-11
# Set to an int once the 1b replicated convergence check is read; every
# `--n-sims` in the full-run command lines is rendered from it, so the
# commands and the protocol can never disagree. `None` = not yet chosen,
# and the commands carry the `<chosen_n_sims>` token instead of a number.
CHOSEN_N_SIMS = 1600  # user decision 2026-09-11 after the 1b table (see convergence_protocol)
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
# The frame every stage-1 arm is now trained on; B and C reached it through
# the 2026-09-11 retrain (D3), so the frame is a pinned fact of their arm
# blocks rather than an asymmetry to declare.
ROLE_BALL_FRAME_I7 = "ball_frame_i7"

# The two artifact families with no manifest role — the stage-1 namespace and
# the unpromoted B18 sidecar — are COMPOSED from segments, the way
# `run_arm.py` composes them, rather than written as one literal.
MODELS_ROOT = Path("models")
SEQ_STAGE1_ROOT = MODELS_ROOT / "embeddings" / "seq_stage1"
# D6 check 6.1: B and C resolve from the i7 retrain, never from the ablation
# lineage. The ablation directory below survives ONLY inside the
# `superseded_checkpoints` record, which names what stage 1 stopped using.
RETRAIN_DIR = SEQ_STAGE1_ROOT / "retrain_i7"
RETRAIN_SUMMARY = RETRAIN_DIR / "summary.yaml"
SUPERSEDED_ABLATION_DIR = MODELS_ROOT / "embeddings" / "t1_ablation_v1_mps"
SUPERSEDED_REASON = (
    "superseded 2026-09-11 by the i7-frame retrain; not a stage 1 arm")
# The five training seeds registered by the D2 cache decision. The retrain
# summary must carry exactly these, once each.
REGISTERED_TRAINING_SEEDS = (7, 13, 29, 42, 101)
# The seed each superseded ablation arm had been pinned at before the
# retrain, recorded so the record names the exact directories that moved.
SUPERSEDED_SEEDS = {"mlp": 101, "full": 101}
# A validation log loss written at four decimals cannot separate these
# seeds; see `_is_four_decimal_rounded`.
FULL_PRECISION_MIN_DECIMALS = 6
# The frame contract every retrained checkpoint must declare (D3 check 3.5,
# D4 check 4.1). The 50-column T1 feature list is a subset of the frame's
# 114 columns, so the frame's own n_features is NOT the contract's.
DELIVERY_SEMANTICS = "inclusive_total_runs_v1"
T1_FEATURE_COUNT = 50
FEATURE_HASH_FILENAME = ".feature_hash"
METRICS_FILENAME = "metrics.json"
# The training parquets the pin RECOMPUTES from (Astra round 1, MUST-FIX 1).
# The stem is the one `transformer_t1.split_path` builds, so the pin reads the
# same files the trainer read; the test and golden splits are never named,
# never opened and never hashed.
FRAME_SPLIT_STEM = "cricket_data_{version}_{split}.parquet"
MATCH_DATE_COLUMN = "match_date"
INNINGS_ID_COLUMN = "innings_id"
# The D3 retrain config, hashed from disk rather than copied out of the
# summary that the same runner wrote.
RETRAIN_CONFIG = "experiments/configs/seq_stage1_retrain_i7_v1.yaml"
# Every key the D3 training contract must carry before a checkpoint can be
# pinned; a null or empty value is as bad as a missing key.
REQUIRED_CONTRACT_KEYS = (
    "contract_version", "frame_dir", "frame_version", "feature_hash",
    "delivery_semantics",
    "venue_alias_version", "venue_alias_sha256", "feature_names",
    "feature_names_sha256", "split_files", "architecture", "seed",
    "best_epoch", "stats_cache",
)
REQUIRED_CONTRACT_SPLITS = ("train", "validation")
REQUIRED_SPLIT_FIELDS = ("path", "md5", "n_rows", "match_date_min",
                         "match_date_max")
REQUIRED_CACHE_FIELDS = ("role", "path", "md5", "venue_alias_version",
                         "same_day_order_version")
ASYMMETRIES_REMOVED_ON = "2026-09-11"
SMOKE_FIXTURE_DIR = SEQ_STAGE1_ROOT / "smoke" / "fixtures"
# Populated at 1b; registered now so the convergence and variability
# batches can pass run_arm's --config preflight (Astra round 2, item 4c).
TIMING_FIXTURE_DIR = SEQ_STAGE1_ROOT / "timing" / "fixtures"
TIMING_SHARD_SIZE = 10
A50_DIR = SEQ_STAGE1_ROOT / "a50"

# --- the 1d shard partition (D11 check 11.1) -------------------------------
#
# `run_arm.py` refuses `--parallel` (deviation `no_parallel_within_a_run`:
# the same-day replay lifecycle is strictly sequential), so the only
# parallelism available for the full run is disjoint fixture shards in
# separate processes. That makes the partition a registered artifact rather
# than an operator's `ls | split`: it is computed here from the fixture set,
# materialised as ten directories, pinned with its ids, count and hash, and
# recomputed by `--verify`.
FULL_RUN_SHARDS = 10
FULL_RUN_ROOT = SEQ_STAGE1_ROOT / "full"
FULL_SHARD_ROOT = FULL_RUN_ROOT / "shards"
SHARD_FIXTURES_DIRNAME = "fixtures"
SHARD_OUTPUT_PREFIX = "shard"
SHARD_RULE = (
    "round-robin over the registered fixture set in the repo's own "
    "(match_date, cricsheet id) chronology "
    "(loaders_common.iter_matches_chronological, the ordering contract every "
    f"tracker walk uses): the i-th fixture of that order goes to shard "
    f"i mod {FULL_RUN_SHARDS}. Round-robin rather than contiguous blocks, so "
    "long and short innings spread evenly and every shard spans the whole "
    "iteration window instead of being one tournament; that keeps the "
    "per-shard wall clocks comparable and keeps any single shard from being "
    "a slice with its own venue or era. The partition is a function of the "
    "fixture set alone, so --verify recomputes it and re-reads the ten "
    "directories: each shard's inventory must be exactly its share of the "
    "order, the union of the ten must be the registered fixture set, and no "
    "fixture may appear in two shards. Every shard still runs with the FULL "
    "same-day replay context (--context-dir data/t20s_json), so a sharded "
    "run sees the same chronology a serial one does — which is what 1c "
    "checks")
SHARD_COMMAND_DIFFERS_ONLY_IN = ["--fixture-dir", "--output-dir"]
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
    # the D3 retrain runner: it produced B's and C's checkpoints, so a
    # change to it changes what those arms are.
    "scripts/sequence_track/retrain_i7.py",
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
            "memory; registered checkpoint from the 2026-09-11 retrain of "
            "the `mlp` arm on the i7 identity frame"),
        "retrain_arm": "mlp",
        "stats_version": "i7",
        "model_type": "transformer",
    },
    "C": {
        "role": "sequence candidate",
        "description": (
            "full T1 transformer over the same 50 pre-ball features; "
            "registered checkpoint from the 2026-09-11 retrain of the "
            "`full` arm on the i7 identity frame"),
        "retrain_arm": "full",
        "stats_version": "i7",
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


# ---------------------------------------------------------------------------
# The 1d shard partition (D11 check 11.1)
# ---------------------------------------------------------------------------

def same_day_order_version() -> str:
    """The repo's versioned within-date ordering contract."""
    from loaders_common import SAME_DAY_ORDER_VERSION  # noqa: PLC0415

    return SAME_DAY_ORDER_VERSION


def chronological_fixture_ids(fixture_dir) -> list:
    """The directory's cricsheet ids in the repo's (match_date, id) order.

    The order comes from `loaders_common.iter_matches_chronological`, which
    IS the versioned same-day contract (`SAME_DAY_ORDER_VERSION`) that every
    tracker walk and the stage-1 replay loader use — the partition therefore
    orders fixtures exactly the way the run will visit them, rather than
    inventing a second chronology here.

    The helper is asked for EVERY document (`gender=None`) and its output is
    compared against the directory's JSON stems: a file it skipped (bad JSON,
    no `info.dates`) would otherwise drop out of the partition silently, and
    a dropped fixture is a fixture no shard would ever run.
    """
    from loaders_common import iter_matches_chronological  # noqa: PLC0415

    directory = _abs(fixture_dir)
    ordered = [match_id for match_id, _text, _date
               in iter_matches_chronological(directory, gender=None)]
    duplicates = sorted({value for value in ordered
                         if ordered.count(value) > 1})
    if duplicates:
        raise PinError(
            f"{_p(fixture_dir)}: the chronology yields fixture(s) "
            f"{duplicates} more than once")
    on_disk = set(fixture_ids_for(fixture_dir))
    missing = sorted(on_disk - set(ordered))
    if missing:
        raise PinError(
            f"{_p(fixture_dir)}: {len(missing)} fixture file(s) are not in "
            f"the chronology and would belong to no shard: {missing[:5]}"
            f"{' ...' if len(missing) > 5 else ''}. "
            "loaders_common.iter_matches_chronological skips a document it "
            "cannot parse or that carries no info.dates")
    unexpected = sorted(set(ordered) - on_disk)
    if unexpected:
        raise PinError(
            f"{_p(fixture_dir)}: the chronology yields {unexpected} which is "
            "not a JSON file of that directory")
    return ordered


def round_robin_partition(ordered_ids, n_shards: int = FULL_RUN_SHARDS
                          ) -> list:
    """`[[ids of shard 0], ...]` by `i mod n_shards` over a fixed order."""
    count = int(n_shards)
    if count < 1:
        raise PinError(f"n_shards must be >= 1, got {n_shards!r}")
    shards = [[] for _ in range(count)]
    for index, fixture_id in enumerate(ordered_ids):
        shards[index % count].append(str(fixture_id))
    empty = [index for index, ids in enumerate(shards) if not ids]
    if empty:
        raise PinError(
            f"shard(s) {empty} would be empty: {len(list(ordered_ids))} "
            f"fixtures do not fill {count} shards")
    return shards


def shard_fixture_dir(index: int) -> Path:
    """`models/embeddings/seq_stage1/full/shards/<k>/fixtures`, composed."""
    return FULL_SHARD_ROOT / str(int(index)) / SHARD_FIXTURES_DIRNAME


def shard_output_dir(arm: str, index: int) -> Path:
    """`models/embeddings/seq_stage1/full/<arm>/shard<k>`, composed."""
    return FULL_RUN_ROOT / str(arm) / f"{SHARD_OUTPUT_PREFIX}{int(index)}"


def _materialize_one_shard(source_dir: str, index: int,
                           fixture_ids) -> dict:
    """Copy one shard's fixtures, or confirm the directory already holds them.

    Refuses rather than repairs: a shard directory that holds a DIFFERENT set
    of fixtures, or the right names with different bytes, is evidence that a
    run under way covered something other than the registered partition, and
    quietly overwriting it would hide that. Deleting it is a deliberate act.
    """
    label = _p(shard_fixture_dir(index))
    target = _abs(shard_fixture_dir(index))
    wanted = {}
    for fixture_id in fixture_ids:
        relative = Path(source_dir) / f"{fixture_id}.json"
        path = _abs(relative)
        if not path.is_file():
            raise PinError(
                f"shard {index}: {_p(relative)} is not on disk, so the "
                "partition cannot be materialised")
        wanted[str(fixture_id)] = path

    if target.exists():
        if not target.is_dir():
            raise PinError(f"{label} exists and is not a directory")
        children = sorted(target.iterdir())
        strays = [child.name for child in children
                  if not (child.is_file() and child.suffix == ".json")]
        if strays:
            raise PinError(
                f"{label} holds non-fixture entries {strays}; a shard "
                "directory holds exactly its cricsheet JSONs")
        if children:
            present = {child.stem for child in children}
            if present != set(wanted):
                raise PinError(
                    f"{label} already holds a different fixture set "
                    f"({len(present)} file(s), "
                    f"{len(present - set(wanted))} unexpected, "
                    f"{len(set(wanted) - present)} missing). The 1d "
                    "partition is deterministic, so this directory was "
                    "built from something else; delete it deliberately and "
                    "re-run --write")
            for fixture_id, path in sorted(wanted.items()):
                if md5_file(target / f"{fixture_id}.json") != md5_file(path):
                    raise PinError(
                        f"{label}/{fixture_id}.json differs from "
                        f"{source_dir}/{fixture_id}.json; a shard is a COPY "
                        "of the registered fixture set, never an edit of it")
            return {"shard": int(index), "action": "unchanged",
                    "fixture_count": len(wanted), "fixture_dir": label}

    target.mkdir(parents=True, exist_ok=True)
    for fixture_id, path in sorted(wanted.items()):
        shutil.copyfile(path, target / f"{fixture_id}.json")
    return {"shard": int(index), "action": "copied",
            "fixture_count": len(wanted), "fixture_dir": label}


def materialize_full_run_shards(fixture_dir=None,
                                n_shards: int = FULL_RUN_SHARDS) -> list:
    """Build the ten shard fixture directories by COPYING the JSONs.

    Idempotent: a shard directory that already holds exactly its fixtures,
    byte for byte, is left alone and reported as `unchanged`. Called by
    `--write` only; `--verify` reads the directories and never writes.
    """
    source = (_role_path(ROLE_FIXTURE_SET) if fixture_dir is None
              else _p(fixture_dir))
    partition = round_robin_partition(
        chronological_fixture_ids(source), n_shards)
    return [_materialize_one_shard(source, index, ids)
            for index, ids in enumerate(partition)]


def full_run_shard_facts(fixture_dir=None,
                         n_shards: int = FULL_RUN_SHARDS) -> list:
    """Per-shard pinned facts, RECOMPUTED from the directories on disk.

    Both `--write` and `--verify` reach this, so the config's shard block is
    a record of the files rather than of what `--write` intended: the
    partition is re-derived from the registered fixture set, each shard
    directory's inventory is re-read and re-hashed and must equal its share
    of the order, the union of the ten inventories must be the registered
    fixture set, and no fixture may appear twice.
    """
    source = (_role_path(ROLE_FIXTURE_SET) if fixture_dir is None
              else _p(fixture_dir))
    registered = sorted(fixture_ids_for(source))
    partition = round_robin_partition(
        chronological_fixture_ids(source), n_shards)

    facts = []
    owner = {}
    for index, ids in enumerate(partition):
        relative = _p(shard_fixture_dir(index))
        directory = _abs(relative)
        if not directory.is_dir():
            raise PinError(
                f"missing shard fixture dir {relative}: the 1d partition is "
                "materialised by pin_stage1.py --write, which copies each "
                f"shard's fixtures out of {source}")
        on_disk = sorted(path.stem for path in directory.glob("*.json"))
        if on_disk != sorted(ids):
            raise PinError(
                f"{relative} holds {len(on_disk)} fixture(s), the partition "
                f"gives it {len(ids)}; the directory and the rule disagree "
                f"(first difference: "
                f"{sorted(set(on_disk) ^ set(ids))[:3]}). Delete the shard "
                "directory and re-run --write")
        for fixture_id in ids:
            if fixture_id in owner:
                raise PinError(
                    f"fixture {fixture_id} is in shard {owner[fixture_id]} "
                    f"and in shard {index}; the shards must be disjoint")
            owner[fixture_id] = index
        facts.append({
            "index": int(index),
            "fixture_dir": relative,
            "fixture_dir_md5": _md5_dir(relative),
            "fixture_dir_hash_contract": DIR_HASH_CONTRACT,
            "fixture_count": len(ids),
            "fixture_ids": list(ids),
        })

    union = sorted(owner)
    if union != registered:
        missing = sorted(set(registered) - set(union))
        extra = sorted(set(union) - set(registered))
        raise PinError(
            f"the union of the {n_shards} shard inventories is not the "
            f"registered fixture set {source} ({len(union)} vs "
            f"{len(registered)}; missing {missing[:5]}, unexpected "
            f"{extra[:5]})")
    return facts


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

SELECTION_RULE = (
    "lowest VALIDATION log loss per retrain arm; ties break to the lowest "
    "seed number; the test split is never consulted")

FULL_PRECISION_RULE = (
    "a per-seed log loss is refused when it equals its own round(ll, 4) AND "
    f"its float repr carries fewer than {FULL_PRECISION_MIN_DECIMALS} "
    "decimal places — i.e. when the value on disk is indistinguishable from "
    "one that was rounded to four decimals before it was written. The "
    "retrained seeds are separated in the fifth and sixth decimal, so a "
    "four-decimal table would decide the selection by the tie-break instead "
    "of by validation log loss. A value that is genuinely a four-decimal "
    "number is refused too: it cannot be told apart from a rounded one, and "
    "an unrounded float64 mean lands on one with vanishing probability")


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


def decimal_places(value: float) -> int:
    """Decimal places in the shortest round-trip repr of ``value``.

    A scientific-notation repr (1e-05) carries its precision in the mantissa
    and is never a four-decimal rounding of anything, so it reports the
    full-precision floor rather than a misleading zero.
    """
    text = repr(float(value))
    if "e" in text or "E" in text:
        return FULL_PRECISION_MIN_DECIMALS
    _, _, fraction = text.partition(".")
    return len(fraction)


def _is_four_decimal_rounded(value: float) -> bool:
    """True when ``value`` cannot be told apart from round(value, 4).

    Both halves of the rule are checked, in the order `FULL_PRECISION_RULE`
    states them: the value equals its own four-decimal rounding, and its
    repr shows fewer decimals than a full-precision mean would.
    """
    number = float(value)
    return (number == round(number, 4)
            and decimal_places(number) < FULL_PRECISION_MIN_DECIMALS)


def _spread_table_from_convergence_1b() -> list:
    """Spread rows from the 1b convergence JSON (spreads only, no levels).

    Returns [] before 1b has run so the config can still be written; once
    the file exists every row is copied verbatim and --verify recomputes
    it from the same file.
    """
    path = SEQ_STAGE1_ROOT / "timing" / "convergence_1b.json"
    if not _abs(path).is_file():
        return []
    payload = json.loads(_abs(path).read_text())
    spread = payload.get("spread", payload.get("spread_table", []))
    variability = {
        (int(r["n_sims"]), r["contrast"]): r
        for r in payload.get("variability", payload.get("variability_rerun", []))}
    rows = []
    for r in spread:
        key = (int(r["n_sims"]), r["contrast"])
        v = variability.get(key, {})
        row = {
            "n_sims": int(r["n_sims"]), "contrast": r["contrast"],
            "range_95": float(r["range_95"]), "sd": float(r["sd"]),
            "below_threshold": bool(r["below_threshold"]),
            "scaled_range_95": float(r["scaled_range_95"]),
            "scaled_below_threshold": bool(r["scaled_below_threshold"]),
            "variability_scaled_shard_mean_sd": (
                float(v["scaled_paired_diff_sd"]) if "scaled_paired_diff_sd" in v else None),
        }
        for name in row:
            for banned in ("log_loss", "delta", "ll_mean"):
                assert banned not in name, name
        rows.append(row)
    return rows


def _validated_ll(value, label: str) -> float:
    """One per-seed log loss: a finite float that passes the rounded-value
    rejection heuristic, or PinError."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PinError(
            f"{label}: validation ll is {value!r}, expected a float")
    number = float(value)
    if not math.isfinite(number):
        raise PinError(f"{label}: validation ll {number!r} is not finite")
    if _is_four_decimal_rounded(number):
        raise PinError(
            f"{label}: validation ll {number!r} is rounded to four decimals "
            f"(repr shows {decimal_places(number)} decimal place(s)); "
            "stage 1 selects on the unrounded float64 mean. "
            + FULL_PRECISION_RULE)
    return number


def _validated_seeds(rows, label: str) -> None:
    """Exactly the five registered training seeds, once each."""
    seeds = []
    for row in rows:
        if not isinstance(row, dict) or "seed" not in row or "ll" not in row:
            raise PinError(
                f"{label}: per-seed row {row!r} needs both 'seed' and 'll'")
        seeds.append(int(row["seed"]))
    duplicates = sorted({seed for seed in seeds if seeds.count(seed) > 1})
    if duplicates:
        raise PinError(
            f"{label}: seed(s) {duplicates} appear more than once; each "
            "registered training seed must appear exactly once")
    missing = sorted(set(REGISTERED_TRAINING_SEEDS) - set(seeds))
    unexpected = sorted(set(seeds) - set(REGISTERED_TRAINING_SEEDS))
    if missing or unexpected:
        raise PinError(
            f"{label}: per-seed table is {sorted(seeds)}, expected exactly "
            f"the registered seeds {list(REGISTERED_TRAINING_SEEDS)} "
            f"(missing {missing}, unexpected {unexpected})")


def load_retrain_summary(summary_path) -> dict:
    """The i7 retrain summary, or a PinError naming what is missing."""
    path = Path(summary_path)
    if not path.is_absolute():
        path = _abs(path)
    if not path.is_file():
        raise PinError(
            f"missing retrain summary: {_p(summary_path)}. Arms B and C are "
            "pinned from the D3 i7 retrain "
            "(scripts/sequence_track/retrain_i7.py); run it before pinning "
            "stage 1")
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise PinError(f"{_p(summary_path)}: top level is not a mapping")
    return payload


def checkpoint_dir_for(summary_path, retrain_arm: str, seed: int) -> Path:
    """The seed's checkpoint directory, composed beside its own summary."""
    return Path(summary_path).parent / retrain_arm / f"seed_{int(seed)}"


def _checkpoint_validation_ll(model_dir, label: str) -> float:
    """The `validation_ll` the seed's own metrics.json records."""
    payload = read_metrics(model_dir)
    if "validation_ll" not in payload:
        raise PinError(
            f"{label}: {_p(Path(model_dir) / METRICS_FILENAME)} records no "
            "validation_ll, so the summary's row cannot be checked against "
            "the run that produced it")
    return _validated_ll(payload["validation_ll"], label)


def _require_summary_matches_metrics(summary_path, retrain_arm: str,
                                     rows, label: str) -> list:
    """Every per-seed log loss, re-read from that seed's own metrics.json.

    Astra round 1, MUST-FIX 1: the summary is written by the retrain runner,
    so believing it is believing one writer twice. Each row is compared
    against the `validation_ll` in the checkpoint directory the row names,
    and the directory itself is composed rather than taken from the row.
    """
    checked = []
    for row in rows:
        seed = int(row["seed"])
        seed_label = f"{label} seed {seed}"
        model_dir = checkpoint_dir_for(summary_path, retrain_arm, seed)
        recorded = row.get("checkpoint_dir")
        if recorded is not None:
            tail = Path(str(recorded)).parts[-2:]
            if tail != (retrain_arm, f"seed_{seed}"):
                raise PinError(
                    f"{seed_label}: checkpoint_dir {recorded!r} does not end "
                    f"in {retrain_arm}/seed_{seed}")
        measured = _checkpoint_validation_ll(model_dir, seed_label)
        if measured != row["validation_ll"]:
            raise PinError(
                f"{seed_label}: summary validation ll "
                f"{row['validation_ll']!r} differs from the "
                f"{METRICS_FILENAME} the run wrote "
                f"({measured!r}, {_p(model_dir)}). The summary is not "
                "evidence for itself")
        checked.append({"seed": seed,
                        "checkpoint_dir": _p(model_dir),
                        "validation_ll": measured})
    return checked


def selection_table(summary_path) -> dict:
    """Per-seed validation LL tables plus the chosen seed for mlp and full.

    D6 check 6.2: the rows must be exactly the five registered training
    seeds, each once, each finite and none rejected by the rounded-value
    heuristic. Anything else
    is a PinError, so a half-finished or rounded retrain table can never
    silently decide which checkpoint stage 1 runs. Each surviving row is then
    checked against the `validation_ll` in that seed's own metrics.json.
    """
    payload = load_retrain_summary(summary_path)
    label_root = _p(summary_path)
    try:
        arms = payload["splits"]["validation"]["arms"]
    except (KeyError, TypeError) as exc:
        raise PinError(
            f"{label_root}: no splits.validation.arms block") from exc
    if not isinstance(arms, dict):
        raise PinError(f"{label_root}: splits.validation.arms is not a "
                       "mapping")
    # The table's SHAPE first, for both arms, before any checkpoint is
    # opened: a summary that is missing an arm or a seed fails on that,
    # not on the first metrics.json the missing arm would have needed.
    per_arm = {}
    for retrain_arm in ("mlp", "full"):
        if retrain_arm not in arms:
            raise PinError(f"{label_root}: no validation arm {retrain_arm}")
        label = f"{label_root} [{retrain_arm}]"
        rows = (arms[retrain_arm] or {}).get("per_seed")
        if not isinstance(rows, list) or not rows:
            raise PinError(f"{label}: no per_seed rows")
        _validated_seeds(rows, label)
        per_arm[retrain_arm] = sorted(rows,
                                      key=lambda row: int(row["seed"]))

    out = {}
    for retrain_arm, ordered in per_arm.items():
        label = f"{label_root} [{retrain_arm}]"
        clean = [
            {"seed": int(row["seed"]),
             "validation_ll": _validated_ll(
                 row["ll"], f"{label} seed {int(row['seed'])}")}
            for row in ordered
        ]
        _require_summary_matches_metrics(
            summary_path, retrain_arm,
            [dict(row, checkpoint_dir=source.get("checkpoint_dir"))
             for row, source in zip(clean, ordered)],
            label)
        seed, ll = choose_seed(
            [{"seed": row["seed"], "ll": row["validation_ll"]}
             for row in clean])
        out[retrain_arm] = {
            "per_seed": clean,
            "chosen_seed": seed,
            "chosen_validation_ll": ll,
        }
    return out


def retrain_summary_block(summary_path, *, frame_dir: str, frame_hash: dict,
                          frame_facts: dict, validation_matches: int,
                          stats_cache_role: str,
                          stats_cache_path: str,
                          stats_cache_md5: str) -> dict:
    """The retrain's identity, every fact checked against something else.

    The summary is the retrain runner's own account of the retrain, so
    nothing in it is taken on trust (Astra round 1, MUST-FIX 1): the frame
    declaration is compared key by key against the frame's `.feature_hash` on
    disk, the split md5s and row counts against `frame_facts` recomputed from
    the parquets, the training cache against the LIVE cache, and the config
    sha256 against a fresh hash of the config file.
    """
    payload = load_retrain_summary(summary_path)
    label = _p(summary_path)
    validation = ((payload.get("splits") or {}).get("validation") or {})
    for field in ("n_rows", "n_matches"):
        if not isinstance(validation.get(field), int) or isinstance(
                validation.get(field), bool):
            raise PinError(
                f"{label}: splits.validation.{field} is "
                f"{validation.get(field)!r}, expected an int")
    _require(int(validation["n_rows"]),
             int(frame_facts["validation"]["n_rows"]),
             f"{label}: splits.validation.n_rows (vs the rows in "
             f"{frame_facts['validation']['path']})")
    _require(int(validation["n_matches"]), int(validation_matches),
             f"{label}: splits.validation.n_matches (vs the distinct match "
             f"ids in {frame_facts['validation']['path']})")

    experiment = payload.get("experiment") or {}
    _require(experiment.get("config"), RETRAIN_CONFIG,
             f"{label}: experiment.config")
    _require(experiment.get("config_sha256"), retrain_config_sha256(),
             f"{label}: experiment.config_sha256 (recorded vs recomputed "
             f"from {RETRAIN_CONFIG} on disk)")

    frame = payload.get("frame") or {}
    _require(frame.get("dir"), frame_dir,
             f"{label}: frame.dir (vs role {ROLE_BALL_FRAME_I7})")
    _require(frame.get("version"), frame_hash.get("version"),
             f"{label}: frame.version (vs the frame's "
             f"{FEATURE_HASH_FILENAME})")
    _require_mapping(frame.get("feature_hash"), frame_hash,
                     f"{label}: frame.feature_hash (vs "
                     f"{_p(Path(frame_dir) / FEATURE_HASH_FILENAME)})")
    split_md5s = frame.get("split_md5s")
    if not isinstance(split_md5s, dict) or not split_md5s:
        raise PinError(f"{label}: frame.split_md5s is {split_md5s!r}, "
                       "expected a mapping of split -> md5")
    _require_mapping(
        split_md5s,
        {split: facts["md5"] for split, facts in frame_facts.items()},
        f"{label}: frame.split_md5s (recorded vs recomputed from the "
        "parquets)")
    split_rows = frame.get("split_rows")
    if split_rows is not None:
        _require_mapping(
            split_rows,
            {split: facts["n_rows"] for split, facts in frame_facts.items()},
            f"{label}: frame.split_rows (recorded vs the parquets)")
    cache = frame.get("stats_cache")
    if not isinstance(cache, dict):
        raise PinError(f"{label}: frame.stats_cache is {cache!r}, expected a "
                       "mapping")
    _require(cache.get("role"), stats_cache_role,
             f"{label}: frame.stats_cache.role")
    _require(cache.get("path"), stats_cache_path,
             f"{label}: frame.stats_cache.path")
    _require(cache.get("md5"), stats_cache_md5,
             f"{label}: frame.stats_cache.md5 (recorded at training vs the "
             f"live {stats_cache_path})")
    _require(cache.get("venue_alias_version"),
             frame_hash.get("venue_alias_version"),
             f"{label}: frame.stats_cache.venue_alias_version (vs the "
             "frame's)")

    return {
        "validation_rows": int(frame_facts["validation"]["n_rows"]),
        "validation_matches": int(validation_matches),
        "validation_split_source": (
            "row count and distinct-match count recomputed from "
            f"{frame_facts['validation']['path']} (match ids are innings ids "
            "with the leading '<innings>_' prefix stripped) and compared "
            "against the summary's copy"),
        "retrain_config": RETRAIN_CONFIG,
        "retrain_config_sha256": retrain_config_sha256(),
        "retrain_config_sha256_source": (
            f"recomputed from {RETRAIN_CONFIG} on disk and compared against "
            "the summary's copy"),
        "frame_dir": frame_dir,
        "frame_dir_role": ROLE_BALL_FRAME_I7,
        "frame_split_md5s": {split: facts["md5"]
                             for split, facts in sorted(frame_facts.items())},
        "frame_split_rows": {split: facts["n_rows"]
                             for split, facts in sorted(frame_facts.items())},
        "frame_split_md5s_source": (
            "recomputed from the parquets by pin_stage1.frame_split_facts, "
            "then compared against the summary's copy"),
    }


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


def _require_mapping(actual, expected: dict, label: str) -> None:
    """Key-by-key equality, so an added or dropped key is a failure too."""
    if not isinstance(actual, dict):
        raise PinError(f"{label}: expected a mapping, found {actual!r}")
    for key in sorted(set(expected) | set(actual), key=str):
        if key not in actual:
            raise PinError(
                f"{label}: key {key!r} is missing (expected "
                f"{expected[key]!r})")
        if key not in expected:
            raise PinError(
                f"{label}: unexpected key {key!r} = {actual[key]!r}")
        if actual[key] != expected[key]:
            raise PinError(
                f"{label}: key {key!r} is {actual[key]!r}, expected "
                f"{expected[key]!r}")


# ---------------------------------------------------------------------------
# Recomputed training facts (Astra round 1, MUST-FIX 1)
#
# The checkpoint's training contract, the retrain summary and the config all
# record the same facts about the training frame. Comparing them against each
# other proves only that one writer was self-consistent. Everything below
# reads the FILES instead: the parquets are hashed and their row counts and
# date ranges read with pyarrow, the feature list is rebuilt from the modules
# that define it, and the retrain config is re-hashed from disk.
# ---------------------------------------------------------------------------

# Hashing ~830 MB of parquet is the expensive half of a pin, and one
# `--write` or `--verify` asks for the same frame three times (arm B, arm C,
# and the retrain summary). It is read once per process and handed out as a
# copy, so no caller can mutate another's facts.
_FRAME_SPLIT_FACTS: dict = {}


def _compute_frame_split_facts(frame_dir: str, frame_version: str) -> dict:
    # Imported late and only here: pyarrow costs a second to import and the
    # seed-overlap screen has no use for it.
    import pyarrow.compute as pc  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    facts = {}
    for split in REQUIRED_CONTRACT_SPLITS:
        stem = FRAME_SPLIT_STEM.format(version=frame_version, split=split)
        relative = f"{frame_dir}/{stem}"
        path = Path(relative)
        if not path.is_absolute():
            path = _abs(relative)
        if not path.is_file():
            raise PinError(
                f"missing training split parquet: {relative}. Stage 1 "
                "recomputes every training fact from the frame, so a split "
                "that is not on disk cannot be pinned")
        try:
            table = pq.read_table(path, columns=[MATCH_DATE_COLUMN])
        except Exception as exc:  # noqa: BLE001 - any read failure fails the pin
            raise PinError(
                f"{relative}: cannot read column "
                f"{MATCH_DATE_COLUMN!r} ({exc})") from exc
        if table.num_rows == 0:
            raise PinError(f"{relative}: no rows")
        bounds = pc.min_max(table.column(MATCH_DATE_COLUMN)).as_py()
        if bounds.get("min") is None or bounds.get("max") is None:
            raise PinError(
                f"{relative}: {MATCH_DATE_COLUMN} is entirely null, so the "
                "split's date range cannot be established")
        facts[split] = {
            "path": relative,
            "md5": md5_file(path),
            "n_rows": int(table.num_rows),
            "match_date_min": str(bounds["min"]),
            "match_date_max": str(bounds["max"]),
        }
    return facts


def frame_split_facts(frame_dir, frame_version: str) -> dict:
    """`{split: {path, md5, n_rows, match_date_min, match_date_max}}`.

    Read from the parquets themselves, for the two splits a stage-1
    checkpoint is allowed to have been trained on. `data/golden` and
    `data/forward_holdout` have no stem here and are never opened.
    """
    key = (_p(frame_dir), str(frame_version))
    if key not in _FRAME_SPLIT_FACTS:
        _FRAME_SPLIT_FACTS[key] = _compute_frame_split_facts(*key)
    return copy.deepcopy(_FRAME_SPLIT_FACTS[key])


_VALIDATION_MATCH_COUNT: dict = {}


def validation_match_count(frame_dir, frame_version: str) -> int:
    """Distinct matches in the validation split, read from the parquet.

    The same rule the retrain runner records using
    (`registered_experiment.match_ids`): strip the leading ``<innings>_``
    prefix from ``innings_id`` and count what is left.
    """
    key = (_p(frame_dir), str(frame_version))
    if key not in _VALIDATION_MATCH_COUNT:
        import pyarrow.parquet as pq  # noqa: PLC0415

        from registered_experiment import match_ids  # noqa: PLC0415

        stem = FRAME_SPLIT_STEM.format(version=key[1], split="validation")
        relative = f"{key[0]}/{stem}"
        path = Path(relative)
        if not path.is_absolute():
            path = _abs(relative)
        try:
            column = pq.read_table(
                path, columns=[INNINGS_ID_COLUMN]).column(INNINGS_ID_COLUMN)
        except Exception as exc:  # noqa: BLE001 - any read failure fails the pin
            raise PinError(
                f"{relative}: cannot read column "
                f"{INNINGS_ID_COLUMN!r} ({exc})") from exc
        _VALIDATION_MATCH_COUNT[key] = int(
            len(set(match_ids(column.to_pylist()))))
    return _VALIDATION_MATCH_COUNT[key]


_EXPECTED_FEATURE_NAMES: tuple = ()


def expected_feature_names() -> tuple:
    """The 50 T1 feature names, in the wrapper's construction order.

    IMPORTED from the modules that define them — `embeddings_e1` for the four
    column groups, `transformer_t1` for the state columns — never restated
    here. A contract whose list is a permutation of these names has the right
    length and the right membership and is still a different model input, so
    the order is compared position by position.
    """
    global _EXPECTED_FEATURE_NAMES  # noqa: PLW0603 - one-shot import cache
    if not _EXPECTED_FEATURE_NAMES:
        # Late: both modules import torch.
        from embeddings_e1 import (CTX_COLS, EB_BAT_COLS,  # noqa: PLC0415
                                   EB_BOWL_COLS, VENUE_COLS)
        from transformer_t1 import STATE_COLS  # noqa: PLC0415

        names = tuple(list(EB_BAT_COLS) + list(EB_BOWL_COLS)
                      + list(VENUE_COLS) + list(CTX_COLS) + list(STATE_COLS))
        if len(names) != T1_FEATURE_COUNT or len(set(names)) != len(names):
            raise PinError(
                f"the T1 feature stack builds {len(names)} names "
                f"({len(set(names))} distinct); stage 1 pins "
                f"{T1_FEATURE_COUNT}")
        _EXPECTED_FEATURE_NAMES = names
    return _EXPECTED_FEATURE_NAMES


def feature_names_sha256(names) -> str:
    """sha256 of the newline-joined names, as `transformer_t1` records it."""
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def _require_feature_names(actual, expected, label: str) -> None:
    if not isinstance(actual, list) or not all(
            isinstance(name, str) for name in actual):
        raise PinError(f"{label}: feature_names is not a list of strings")
    if len(actual) != len(expected):
        raise PinError(
            f"{label}: feature count: expected {len(expected)}, found "
            f"{len(actual)}")
    for index, (found, wanted) in enumerate(zip(actual, expected)):
        if found != wanted:
            raise PinError(
                f"{label}: feature_names[{index}] is {found!r}, expected "
                f"{wanted!r}. The registered order is EB_BAT + EB_BOWL + "
                "VENUE + CTX + STATE (embeddings_e1, transformer_t1); a "
                "permutation is a different model input, not a relabelling")


def retrain_config_sha256() -> str:
    """sha256 of the registered retrain config, read from disk."""
    return _sha256(RETRAIN_CONFIG)


# ---------------------------------------------------------------------------
# The D3 training contract (D6 check 6.3)
#
# `metrics.json` beside every retrained checkpoint carries the frame, the
# delivery semantics, the venue-alias identity, the split md5s and the
# stats cache the checkpoint was TRAINED against. Stage 1 pins those facts
# and `--verify` re-reads them, so a checkpoint that was retrained, or a
# frame or cache that moved underneath it, fails the pin instead of quietly
# serving.
# ---------------------------------------------------------------------------

def read_metrics(model_dir) -> dict:
    """A checkpoint's `metrics.json`, parsed, or a PinError naming it."""
    metrics = _abs(Path(model_dir) / METRICS_FILENAME)
    if not metrics.is_file():
        raise PinError(
            f"missing {METRICS_FILENAME} for checkpoint dir "
            f"{_p(model_dir)}: a stage-1 transformer arm is pinned from its "
            "training contract")
    try:
        return json.loads(metrics.read_text())
    except json.JSONDecodeError as exc:
        raise PinError(f"{_p(metrics)}: unparseable JSON ({exc})") from exc


def read_training_contract(model_dir) -> dict:
    """The `training_contract` block of a checkpoint's metrics.json."""
    payload = read_metrics(model_dir)
    contract = payload.get("training_contract")
    if not isinstance(contract, dict):
        raise PinError(
            f"{_p(Path(model_dir) / METRICS_FILENAME)}: no training_contract "
            "block. A checkpoint trained without --stats-cache-role records "
            "no contract and is refused (D3 check 3.5, D4 check 4.1)")
    missing = [key for key in REQUIRED_CONTRACT_KEYS
               if contract.get(key) in (None, "", [], {})]
    if missing:
        raise PinError(
            f"{_p(Path(model_dir) / METRICS_FILENAME)}: training_contract is "
            f"missing or null for {missing}")
    return contract


def frame_feature_hash(frame_dir) -> dict:
    """The frame's own `.feature_hash` declaration."""
    path = _abs(Path(frame_dir) / FEATURE_HASH_FILENAME)
    if not path.is_file():
        raise PinError(
            f"missing {_p(Path(frame_dir) / FEATURE_HASH_FILENAME)}")
    return json.loads(path.read_text())


def training_frame_block(contract: dict, *, model_dir, frame_dir: str,
                         frame_hash: dict, frame_facts: dict,
                         stats_cache_role: str,
                         stats_cache_path: str, stats_cache_md5: str,
                         expected_seed: int,
                         expected_names=None) -> dict:
    """The pinned training identity of one retrained checkpoint.

    Every field is read from the checkpoint's own contract and checked
    against something that lives outside it: the frame's `.feature_hash`,
    the parquets themselves (`frame_facts`), the feature list the serving
    stack builds, the manifest role paths, the LIVE stats-cache md5, and the
    seed the selection rule chose. A disagreement is a PinError naming the
    field.
    """
    label = _p(Path(model_dir) / METRICS_FILENAME)
    expected_names = (expected_feature_names() if expected_names is None
                      else tuple(expected_names))

    _require(contract["frame_dir"], frame_dir, f"{label}: frame_dir")
    _require(contract["frame_version"], frame_hash.get("version"),
             f"{label}: frame_version (vs the frame's .feature_hash)")
    # The whole frame declaration, key by key: a contract that copied only
    # the three fields below could disagree with the frame about the split
    # boundaries, the gender filter or the shrinkage constants and still pass.
    _require_mapping(contract["feature_hash"], frame_hash,
                     f"{label}: feature_hash (vs "
                     f"{_p(Path(frame_dir) / FEATURE_HASH_FILENAME)})")
    _require(contract["delivery_semantics"], DELIVERY_SEMANTICS,
             f"{label}: delivery_semantics")
    _require(contract["delivery_semantics"],
             frame_hash.get("delivery_semantics"),
             f"{label}: delivery_semantics (vs the frame's .feature_hash)")
    _require(contract["venue_alias_version"],
             frame_hash.get("venue_alias_version"),
             f"{label}: venue_alias_version (vs the frame's .feature_hash)")
    _require(contract["venue_alias_sha256"],
             frame_hash.get("venue_alias_sha256"),
             f"{label}: venue_alias_sha256 (vs the frame's .feature_hash)")
    _require(int(contract["seed"]), int(expected_seed),
             f"{label}: seed (vs the seed the selection rule chose)")

    names = contract["feature_names"]
    _require_feature_names(names, expected_names, label)
    names_sha = contract["feature_names_sha256"]
    if not isinstance(names_sha, str) or len(names_sha) != 64:
        raise PinError(
            f"{label}: feature_names_sha256 is {names_sha!r}, expected a "
            "64-character sha256")
    _require(names_sha, feature_names_sha256(names),
             f"{label}: feature_names_sha256 (recorded vs recomputed from "
             "the names it claims to hash)")

    splits = contract["split_files"]
    if not isinstance(splits, dict):
        raise PinError(f"{label}: split_files is not a mapping")
    for split in REQUIRED_CONTRACT_SPLITS:
        row = splits.get(split)
        if not isinstance(row, dict):
            raise PinError(f"{label}: split_files has no {split!r} block")
        absent = [field for field in REQUIRED_SPLIT_FIELDS
                  if row.get(field) in (None, "")]
        if absent:
            raise PinError(
                f"{label}: split_files.{split} is missing {absent}")
        # The comparison that makes the contract mean something: the parquet
        # as it is NOW, hashed and read, against what training recorded.
        measured = frame_facts.get(split)
        if not isinstance(measured, dict):
            raise PinError(
                f"{label}: no recomputed facts for split {split!r}; "
                "frame_split_facts must cover every contracted split")
        for field in REQUIRED_SPLIT_FIELDS:
            found = row[field]
            wanted = measured[field]
            if field == "n_rows":
                found, wanted = int(found), int(wanted)
            _require(found, wanted,
                     f"{label}: split_files.{split}.{field} (recorded at "
                     f"training vs {measured['path']} on disk)")

    cache = contract["stats_cache"]
    if not isinstance(cache, dict):
        raise PinError(f"{label}: stats_cache is not a mapping")
    absent = [field for field in REQUIRED_CACHE_FIELDS
              if cache.get(field) in (None, "")]
    if absent:
        raise PinError(f"{label}: stats_cache is missing {absent}")
    _require(cache["role"], stats_cache_role, f"{label}: stats_cache.role")
    _require(cache["path"], stats_cache_path, f"{label}: stats_cache.path")
    # The one comparison against the world as it is NOW: the cache this
    # checkpoint was trained against must still be the cache stage 1 serves.
    _require(cache["md5"], stats_cache_md5,
             f"{label}: stats_cache.md5 (recorded at training vs the live "
             f"{stats_cache_path})")
    _require(cache["venue_alias_version"], contract["venue_alias_version"],
             f"{label}: stats_cache.venue_alias_version (vs the frame's)")

    return {
        "dir": frame_dir,
        "dir_role": ROLE_BALL_FRAME_I7,
        "frame_version": contract["frame_version"],
        # Written from the RECOMPUTED facts, not from the contract: the two
        # are equal by the checks above, and pinning the measured value makes
        # the YAML a record of the files rather than of the trainer's claim.
        "split_md5s": {split: frame_facts[split]["md5"]
                       for split in sorted(frame_facts)},
        "split_rows": {split: frame_facts[split]["n_rows"]
                       for split in sorted(frame_facts)},
        "split_date_range": {
            split: [frame_facts[split]["match_date_min"],
                    frame_facts[split]["match_date_max"]]
            for split in sorted(frame_facts)},
        "split_facts_source": (
            "recomputed from the parquets (md5, row count and match_date "
            "range) and compared against the checkpoint's training contract"),
        "delivery_semantics": contract["delivery_semantics"],
        "venue_alias_version": contract["venue_alias_version"],
        "venue_alias_sha256": contract["venue_alias_sha256"],
        "feature_count": len(names),
        "feature_names_sha256": feature_names_sha256(expected_names),
        "feature_names_source": (
            "EB_BAT + EB_BOWL + VENUE (embeddings_e1) + CTX (embeddings_e1) "
            "+ STATE (transformer_t1), imported in that order; the contract's "
            "list is compared position by position and its sha256 "
            "recomputed"),
        "architecture": copy.deepcopy(contract["architecture"]),
        "training_seed": int(contract["seed"]),
        "best_epoch": contract["best_epoch"],
        "contract_version": contract["contract_version"],
        "contract_source": _p(Path(model_dir) / METRICS_FILENAME),
        "stats_cache_at_training": stats_cache_path,
        "stats_cache_at_training_role": stats_cache_role,
        "stats_cache_at_training_md5": cache["md5"],
        "stats_cache_same_day_order_version": cache["same_day_order_version"],
        "verified_by": (
            "pin_stage1.py --verify RECOMPUTES rather than copies: it "
            f"re-reads the checkpoint's {METRICS_FILENAME}, hashes both "
            "training parquets and re-reads their row counts and match_date "
            "ranges, compares the contract's whole frame declaration against "
            f"{_p(Path(frame_dir) / FEATURE_HASH_FILENAME)} key by key, "
            "rebuilds the 50 feature names from embeddings_e1 and "
            "transformer_t1 and re-derives their sha256, and compares "
            "stats_cache.md5 against the live cache; any difference fails "
            "the pin"),
    }


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
        # Every launch stamps its I3 block ids from the whole registered
        # fixture set, never from its own (possibly sharded) fixture dir
        # (stage-1 1c, D10).
        "--cluster-source-dir", _role_path(ROLE_FIXTURE_SET),
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
        retrain_arm = spec["retrain_arm"]
        seed = selection[retrain_arm]["chosen_seed"]
        model_dir = _p(RETRAIN_DIR / retrain_arm / f"seed_{seed}")
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
        chosen = selection[spec["retrain_arm"]]
        block["retrain_arm"] = spec["retrain_arm"]
        block["checkpoint_seed"] = chosen["chosen_seed"]
        block["checkpoint_validation_ll"] = chosen["chosen_validation_ll"]
        block["checkpoint_selection"] = (
            f"see checkpoint_selection: {SELECTION_RULE}, read from "
            f"{_p(RETRAIN_SUMMARY)}")
        block["training_frame"] = training_frame_block(
            read_training_contract(model_dir),
            model_dir=model_dir,
            frame_dir=_role_path(ROLE_BALL_FRAME_I7),
            frame_hash=shared["frame_feature_hash"],
            frame_facts=shared["frame_split_facts"],
            stats_cache_role=ROLE_STATS_CACHE[spec["stats_version"]],
            stats_cache_path=_role_path(
                ROLE_STATS_CACHE[spec["stats_version"]]),
            stats_cache_md5=shared["stats_cache_md5"][spec["stats_version"]],
            expected_seed=chosen["chosen_seed"])
    block.update({
        "stats_version": stats_version,
        "stats_cache": stats_cache,
        "stats_cache_role": stats_cache_role,
        "stats_cache_md5": shared["stats_cache_md5"][stats_version],
        "context_dir": CONTEXT_DIR,
        "context_dir_md5": shared["context_dir_md5"],
        "context_dir_hash_contract": DIR_HASH_CONTRACT,
        "context_dir_json_count": shared["context_dir_json_count"],
        # The corpus the I3 competition-cluster lookup is built from, for
        # EVERY run block including each 1d shard. A block id carries the
        # first date of that event's members in the directory the lookup was
        # built from, so building it from a shard's own fixtures moves the
        # id (stage-1 1c, D10); the registered fixture set is the one corpus
        # that makes a sharded run and a serial run stamp the same blocks.
        "cluster_source_dir": _role_path(ROLE_FIXTURE_SET),
        "cluster_source_dir_role": ROLE_FIXTURE_SET,
        "cluster_source_dir_md5": shared["fixture_set_md5"],
        "cluster_source_dir_hash_contract": DIR_HASH_CONTRACT,
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
            "cluster_source_dir": _role_path(ROLE_FIXTURE_SET),
            "cluster_source_dir_role": ROLE_FIXTURE_SET,
            "cluster_source_dir_md5": shared["fixture_set_md5"],
            "n_sims": full_n_sims,
            "base_seeds": [BASE_SEED],
            "n_sims_source": shared["full_run_n_sims_source"],
            "output_dir": output_full,
            "command": _command(arm, model_dir=model_dir,
                                fixture_dir=_role_path(ROLE_FIXTURE_SET),
                                n_sims=shared["full_run_n_sims_token"],
                                output_dir=output_full),
            "command_completeness": shared["full_run_command_completeness"],
            # D11 check 11.1. The whole-set command above stays the
            # REFERENCE — it is what the ten shard commands are a partition
            # of, and it is what a serial rerun would use.
            "shard_rule": SHARD_RULE,
            "shard_count": FULL_RUN_SHARDS,
            "shard_order_version": shared["same_day_order_version"],
            "shard_command_differs_only_in": list(
                SHARD_COMMAND_DIFFERS_ONLY_IN),
            "shards_source": (
                "pin_stage1.full_run_shard_facts: the partition is "
                "recomputed from the registered fixture set and every shard "
                "directory is re-read and re-hashed on both --write and "
                "--verify; --write additionally materialises the "
                "directories by copying the fixture JSONs"),
            "shards": [
                {
                    "index": shard["index"],
                    "fixture_dir": shard["fixture_dir"],
                    "fixture_dir_md5": shard["fixture_dir_md5"],
                    "fixture_dir_hash_contract": shard[
                        "fixture_dir_hash_contract"],
                    "fixture_count": shard["fixture_count"],
                    "fixture_ids": list(shard["fixture_ids"]),
                    # The shard scores its own fixtures but stamps its I3
                    # block ids from the WHOLE set (1c, D10).
                    "cluster_source_dir": _role_path(ROLE_FIXTURE_SET),
                    "cluster_source_dir_role": ROLE_FIXTURE_SET,
                    "cluster_source_dir_md5": shared["fixture_set_md5"],
                    "output_dir": _p(shard_output_dir(arm, shard["index"])),
                    "command": _command(
                        arm, model_dir=model_dir,
                        fixture_dir=shard["fixture_dir"],
                        n_sims=shared["full_run_n_sims_token"],
                        output_dir=_p(shard_output_dir(arm, shard["index"]))),
                }
                for shard in shared["full_run_shards"]
            ],
        },
        "smoke_1a": {
            "fixture_dir": _p(SMOKE_FIXTURE_DIR),
            "fixture_dir_md5": shared["smoke_fixture_dir_md5"],
            "fixture_count": shared["smoke_fixture_count"],
            "cluster_source_dir": _role_path(ROLE_FIXTURE_SET),
            "cluster_source_dir_role": ROLE_FIXTURE_SET,
            "cluster_source_dir_md5": shared["fixture_set_md5"],
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
            "cluster_source_dir": _role_path(ROLE_FIXTURE_SET),
            "cluster_source_dir_role": ROLE_FIXTURE_SET,
            "cluster_source_dir_md5": shared["fixture_set_md5"],
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
    selection = selection_table(RETRAIN_SUMMARY)
    frame_dir = _role_path(ROLE_BALL_FRAME_I7)
    frame_hash = frame_feature_hash(frame_dir)
    frame_facts = frame_split_facts(frame_dir, frame_hash.get("version"))
    i7_cache_role = ROLE_STATS_CACHE["i7"]
    i7_cache_path = _role_path(i7_cache_role)
    i7_cache_md5 = _md5_file(i7_cache_path)
    retrain_summary = retrain_summary_block(
        RETRAIN_SUMMARY,
        frame_dir=frame_dir,
        frame_hash=frame_hash,
        frame_facts=frame_facts,
        validation_matches=validation_match_count(
            frame_dir, frame_hash.get("version")),
        stats_cache_role=i7_cache_role,
        stats_cache_path=i7_cache_path,
        stats_cache_md5=i7_cache_md5)
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
        # Only the versions the arms actually use: since the 2026-09-11
        # retrain every stage-1 arm reads the i7 cache, and the legacy v3
        # cache is no longer a stage-1 artifact to hash.
        "stats_cache_md5": {
            version: (i7_cache_md5 if version == "i7"
                      else _md5_file(_role_path(ROLE_STATS_CACHE[version])))
            for version in sorted(
                {spec["stats_version"] for spec in ARM_SPEC.values()})},
        # Read once and handed to every transformer arm: the frame's own
        # declaration is what each checkpoint's training contract is checked
        # against.
        "frame_feature_hash": frame_hash,
        # Recomputed once from the two training parquets (md5, row count,
        # match_date range) and handed to both transformer arms; the test and
        # golden splits are never named or opened.
        "frame_split_facts": frame_facts,
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
        # The 1d partition, recomputed from the fixture set and re-read from
        # the ten shard directories once per pin and shared by every arm
        # (only the output dir and the command differ per arm).
        "full_run_shards": full_run_shard_facts(fixture_set),
        "same_day_order_version": same_day_order_version(),
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
            "and the registered full-sequence T1 versus token-MLP contrast "
            "in rollout (C-B), which compares two specific architectures "
            "rather than isolating sequence memory."),
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

    selected_ll = {
        arm: selection[retrain_arm]["chosen_validation_ll"]
        for arm, retrain_arm in (("B", "mlp"), ("C", "full"))
    }
    config["checkpoint_selection"] = {
        "rule": SELECTION_RULE,
        "computed_by": "scripts/sequence_track/pin_stage1.py (not hardcoded)",
        "source": _p(RETRAIN_SUMMARY),
        "source_role": None,
        "source_role_note": (
            "the stage-1 retrain namespace is an unpromoted research "
            "artifact with no manifest role; its path is composed from "
            "segments and its identity is the sha256 below"),
        "source_sha256": _sha256(RETRAIN_SUMMARY),
        "registered_seeds": list(REGISTERED_TRAINING_SEEDS),
        "registered_seeds_rule": (
            "the per-seed table must carry exactly these seeds, once each, "
            "every log loss finite and none rejected by the rounded-value "
            "heuristic (a value indistinguishable from a four-decimal "
            "rounding); anything else fails the pin rather than selecting "
            "from a partial table"),
        "full_precision_rule": FULL_PRECISION_RULE,
        "split": "validation",
        "split_rows": retrain_summary["validation_rows"],
        "split_matches": retrain_summary["validation_matches"],
        "split_source": retrain_summary["validation_split_source"],
        "retrain_config": retrain_summary["retrain_config"],
        "retrain_config_sha256": retrain_summary["retrain_config_sha256"],
        "retrain_config_sha256_source": retrain_summary[
            "retrain_config_sha256_source"],
        "frame_dir": retrain_summary["frame_dir"],
        "frame_dir_role": retrain_summary["frame_dir_role"],
        "frame_split_md5s": retrain_summary["frame_split_md5s"],
        "frame_split_rows": retrain_summary["frame_split_rows"],
        "frame_split_md5s_source": retrain_summary["frame_split_md5s_source"],
        "per_seed_validation_ll": {
            retrain_arm: selection[retrain_arm]["per_seed"]
            for retrain_arm in ("mlp", "full")
        },
        "per_seed_validation_ll_source": (
            "read from the retrain summary and then checked, seed by seed, "
            "against the validation_ll in that seed's own metrics.json; a "
            "summary row that disagrees with the run that produced it fails "
            "the pin, so the summary is never evidence for itself"),
        "chosen": {
            "B": {
                "retrain_arm": "mlp",
                "seed": selection["mlp"]["chosen_seed"],
                "validation_ll": selection["mlp"]["chosen_validation_ll"],
                "model_dir": config["arms"]["B"]["model_dir"],
                "checkpoint": config["arms"]["B"]["checkpoint"],
            },
            "C": {
                "retrain_arm": "full",
                "seed": selection["full"]["chosen_seed"],
                "validation_ll": selection["full"]["chosen_validation_ll"],
                "model_dir": config["arms"]["C"]["model_dir"],
                "checkpoint": config["arms"]["C"]["checkpoint"],
            },
        },
        # D6 check 6.7. Teacher-forced validation log loss on the split the
        # seed was CHOSEN on: it is the selection statistic itself, not an
        # independent read of either arm, and it is not a rollout number.
        "teacher_forced_diagnostic": {
            "label": "selection_conditioned_diagnostic_not_evidence",
            "selected_validation_ll": dict(selected_ll),
            "C_minus_B_validation_ll": selected_ll["C"] - selected_ll["B"],
            "what": (
                "the selected checkpoints' own teacher-forced validation log "
                "loss, and C minus B on it. Both numbers are conditioned on "
                "the selection that produced them (the minimum over five "
                "seeds), they are measured on next-ball prediction rather "
                "than on simulated match outcomes, and neither is a stage-1 "
                "contrast. The stage-1 C-B result is the paired winner log "
                "loss from the registered run; this line is recorded so the "
                "selection is auditable, and it advances nothing"),
        },
    }

    # D6 check 6.1: the ablation checkpoints stage 1 used to pin, named with
    # their hashes so the record says exactly what was dropped and what it
    # was replaced by. They are NOT a stage-1 path: no arm resolves here.
    config["superseded_checkpoints"] = {
        "reason": SUPERSEDED_REASON,
        "replaced_by": _p(RETRAIN_DIR),
        "dir_hash_contract": DIR_HASH_CONTRACT,
        "checkpoints": [
            {
                "arm": arm,
                "retrain_arm": retrain_arm,
                "superseded_model_dir": _p(
                    SUPERSEDED_ABLATION_DIR / retrain_arm
                    / f"seed_{SUPERSEDED_SEEDS[retrain_arm]}"),
                "superseded_model_dir_md5": _md5_dir(
                    SUPERSEDED_ABLATION_DIR / retrain_arm
                    / f"seed_{SUPERSEDED_SEEDS[retrain_arm]}"),
                "seed": SUPERSEDED_SEEDS[retrain_arm],
                "reason": SUPERSEDED_REASON,
                "now_pinned_at": config["arms"][arm]["model_dir"],
            }
            for arm, retrain_arm in (("B", "mlp"), ("C", "full"))
        ],
        "note": (
            "these directories are read-only for stage 1 (D3 check 3.8 "
            "records their hashes before and after the retrain) and are "
            "listed here only so the change of source is auditable. They "
            "were trained on the legacy v3 frame, so no number produced "
            "from them is comparable to a retrained arm's"),
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
            "n_sims doubles from 100 with no ceiling: 100, 200, 400, 800, ...; "
            "the 1b run measures 50, 100, 200, 400, 800 (user decision "
            "2026-09-11), where 50 is a noise-curve point only and never a "
            "full-run candidate (plug-in bias about 0.01 at 50 simulations)"),
        "timing_1b_candidates": list(TIMING_1B_CANDIDATES),
        "stop_rule_reading": (
            "USER DECISION 2026-09-11, recorded before 1d. The literal rule "
            "(95% range of the paired C-B and B-A shard-mean contrast across "
            "three batches below 0.002) was NOT met at any permitted count: "
            "on the 10-fixture shard the raw range at 800 was 0.034961 (C-B) "
            "and 0.043664 (B-A), the full-set equivalent (x sqrt(10/255)) "
            "0.006923 and 0.008647, and the a/sqrt(n) fit crosses 0.002 on "
            "the scaled reading at about n = 5,264 (C-B) and 4,497 (B-A), "
            "above the 3,200 joint seed cap. The three-batch range is a "
            "poor estimator with n = 3 (non-monotone 200 -> 400). The "
            "operational reading adopted: the full-set-equivalent Monte "
            "Carlo SD of the paired primary contrasts, measured as the "
            "per-fixture paired difference between batch seeds 20260910 and "
            "20260911 divided by sqrt(10) and scaled by sqrt(10/255), must "
            "be at or below about 0.002 at the chosen count. Measured at "
            "800: C-B 0.002513, B-A 0.003221; extrapolated (/sqrt 2) at "
            "1,600: about 0.0018 and 0.0023. Chosen n_sims = 1,600, the "
            "0.007 equivalence margin being 3-4 x that SD. This is a change "
            "of statistic from the registered range rule, made by the user "
            "with the numbers in hand, and is recorded as such"),
        "stop_rule_reading_source": (
            f"{_p(SEQ_STAGE1_ROOT / 'timing' / 'convergence_1b')}.{{md,json}}; "
            "docs/sequence_track/stage1_acceptance.md D9"),
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
            "n_sims", "contrast", "range_95", "sd", "below_threshold",
            "scaled_range_95", "scaled_below_threshold",
            "variability_scaled_shard_mean_sd"],
        "spread_table_columns_note": (
            "the three per-batch paired-difference columns of the original "
            "registration are deliberately NOT recorded: they are level-"
            "bearing (a paired delta on the shard) and the 1b rule forbids "
            "reading any shard log loss; only spreads are kept"),
        "spread_table": _spread_table_from_convergence_1b(),
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
             "question": (
                 "the registered full-sequence T1 versus token-MLP contrast "
                 "in rollout, on the same 50 features; the two architectures "
                 "differ in outcome-history embedding, attention and "
                 "parameter count, so this is a comparison of that specific "
                 "pair and not an isolation of sequence memory"),
             "decision_it_feeds": (
                 "whether this T1 beats this MLP on the same features; it is "
                 "evidence about that pair, not by itself an advancement, "
                 "and not a measurement of what sequence memory is worth")},
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
            "evidence about that architecture pair for stage 2, not an "
            "advancement and not a measurement of sequence memory"),
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
            # D6 check 6.5. What the cache unification did NOT remove: the
            # two arms still see different information.
            "id": "feature_set_114_vs_50",
            "what": (
                "A carries 114 features; A50, B and C carry 50"),
            "why": (
                "the T1 stack was built on a 50-column pre-ball feature "
                "list; the production ball model uses the full 114-column "
                "i7 frame"),
            "consequence": (
                "B-A and C-A are system contrasts, not model-only "
                "contrasts: they compare a 114-feature system with a "
                "50-feature one. A50-A isolates the feature set at equal "
                "information, within one model family (same family, same "
                "frame, same cache, same hyperparameters, 50 columns instead "
                "of 114). C-B is the registered full-sequence T1 versus "
                "token-MLP contrast (architectures differ in outcome-history "
                "embedding, attention and parameter count), so it is not an "
                "isolation of sequence memory: it is a comparison of two "
                "specific architectures that happen to differ in several "
                "ways at once"),
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

    # D6 check 6.4: the two asymmetries the retrain removed, kept as a
    # record so a reader of an earlier config sees where they went.
    config["removed_asymmetries"] = [
        {
            "id": "stats_cache_i7_vs_v3",
            "removed_on": ASYMMETRIES_REMOVED_ON,
            "what_it_said": (
                "A and A50 served from the i7 stats cache while B and C "
                "served from the legacy v3 cache, so B-A and C-A were "
                "contrasts across two caches"),
            "reason": (
                "removed by the 2026-09-11 retrain on the i7 frame: all "
                "four arms now serve from "
                f"{_role_path(ROLE_STATS_CACHE['i7'])}, and every arm "
                "block pins the same cache md5"),
        },
        {
            "id": "training_frame_i7_vs_v3",
            "removed_on": ASYMMETRIES_REMOVED_ON,
            "what_it_said": (
                "A and A50 were trained on the i7 identity frame while B "
                "and C were trained on the pre-I7 v3 frame, so any A-vs-B "
                "or A-vs-C difference confounded venue identity resolution "
                "with the model"),
            "reason": (
                "removed by the 2026-09-11 retrain on the i7 frame: B and C "
                "are retrained on "
                f"{_role_path(ROLE_BALL_FRAME_I7)}, and each arm block pins "
                "that frame's split md5s and alias identity from the "
                "checkpoint's own training contract"),
        },
    ]

    # D6 check 6.6: what the shared cache does NOT fix. Recorded as a
    # limitation of the screen, not as an asymmetry between arms.
    config["known_limitations"] = [
        {
            "id": "global_prior_not_as_of",
            "what": (
                "the stats cache's global outcome prior π is summed over "
                "the tracker's FINAL-state counts (scripts/"
                "build_stats_cache.py, the 'global prior π' block: 'Fixed "
                "constant — not rolling, not as-of-date'), so it reflects "
                "the whole corpus rather than the state before each match. "
                "Per-player, per-venue and per-match counts remain "
                "as-of-date (invariant 2)"),
            "measured": (
                "2026-09-11 on "
                f"{_role_path(ROLE_BALL_FRAME_I7)} (train 2005-02-17 -> "
                "2024-12-30; corpus -> 2026-04-16)"),
            "pi_train_only_vs_whole_corpus": [
                {"outcome": "wicket", "train_only": 0.054037,
                 "whole_corpus": 0.054361, "difference": 0.000324},
                {"outcome": "dot", "train_only": 0.303601,
                 "whole_corpus": 0.304017, "difference": 0.000415},
                {"outcome": "single", "train_only": 0.413227,
                 "whole_corpus": 0.411313, "difference": -0.001915},
                {"outcome": "two", "train_only": 0.076086,
                 "whole_corpus": 0.075796, "difference": -0.000291},
                {"outcome": "four", "train_only": 0.107686,
                 "whole_corpus": 0.107803, "difference": 0.000117},
                {"outcome": "six", "train_only": 0.045362,
                 "whole_corpus": 0.046711, "difference": 0.001349},
            ],
            "difference_convention": "whole corpus minus train only",
            "max_abs_difference": 0.001915,
            "feature_shift_rule": "difference x k/(n+k)",
            "feature_shift_units": (
                "n is the number of balls the cell itself has seen and k is "
                "its shrinkage constant, so k/(n+k) is the prior's share of "
                "the shrunk cell. The 0.000174 row is therefore reached at "
                "2,000 balls in a VENUE cell (k = 200) and at 300 balls in a "
                "PLAYER cell (k = 30) — not at the same ball count for both"),
            "feature_shift": [
                {"cell": "venue", "k": 200, "n": 0, "shift": 0.001915},
                {"cell": "venue", "k": 200, "n": 200, "shift": 0.000957},
                {"cell": "venue", "k": 200, "n": 2000, "shift": 0.000174},
                {"cell": "player", "k": 30, "n": 0, "shift": 0.001915},
                {"cell": "player", "k": 30, "n": 30, "shift": 0.000957},
                {"cell": "player", "k": 30, "n": 300, "shift": 0.000174},
            ],
            "feature_shift_is_not_a_log_loss_bound": (
                "the shifts above are measured in FEATURE space: they are "
                "how far one shrunk probability column moves. Nothing here "
                "measures how far a model's winner log loss moves as a "
                "result, and the two are not the same quantity. They are "
                "deliberately NOT compared with the "
                f"{EQUIVALENCE_MARGIN_LL} equivalence margin, which is a log "
                "loss; an earlier version of this block made that comparison "
                "and it was withdrawn (Astra round 1)"),
            "identical_across_arms": True,
            "identical_across_arms_scope": (
                "the EXPOSURE is identical, and that is all this key claims: "
                "all four stage-1 arms read one cache, so every arm's "
                "features carry the same non-as-of prior. It does NOT claim "
                "the exposure affects the arms equally"),
            "differential_effect_on_arms": "unknown_and_unmeasured",
            "consequence": (
                "shared exposure, unknown differential effect. The shift "
                "enters every arm's inputs identically, but the four arms "
                "are different functions of those inputs — a 114-feature "
                "gradient-boosted model, the same family at 50 columns, a "
                "token MLP and a transformer — and how much each one's "
                "output moves in response has not been measured. This "
                "limitation therefore cannot be used to argue that a "
                "stage-1 contrast is unaffected, nor that any arm is "
                "favoured or disfavoured: the direction and size of the "
                "effect on B-A, C-A, C-B and A50-A are simply unknown. It "
                "is an inherited limitation of the iteration screen, not a "
                "stage-1 defect, and it is on the backlog (TODO.md, 'Global "
                "outcome prior is not as-of-date')"),
            "source": "scripts/sequence_track/measure_global_prior_asof.py",
            "source_note": (
                "the script that measures the two priors and the k/(n+k) "
                "table above; it measures the FEATURE shift only, and "
                "measures no downstream model effect"),
            "recorded_in": (
                'docs/sequence_track/stage1_acceptance.md, D2 check 2.4'),
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


def write_config(path: Path = CONFIG_PATH, *, materialize: bool = True
                 ) -> dict:
    # The only write outside the YAML: `--write` materialises the ten 1d
    # shard fixture directories before it pins them, so the hashes it records
    # are hashes of directories that exist. `--verify` never writes.
    if materialize:
        selection_table(RETRAIN_SUMMARY)
        materialize_full_run_shards()
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
    for arm, retrain_arm in (("B", "mlp"), ("C", "full")):
        chosen = block["chosen"][arm]
        print(f"  arm {arm} <- retrain arm {retrain_arm!r}")
        for row in block["per_seed_validation_ll"][retrain_arm]:
            mark = " <-- chosen" if row["seed"] == chosen["seed"] else ""
            print(f"    seed {row['seed']:>3}  "
                  f"validation LL {row['validation_ll']:.6f}{mark}")
        print(f"    chosen: seed {chosen['seed']}, "
              f"validation LL {chosen['validation_ll']:.6f}")
        print(f"    checkpoint: {chosen['checkpoint']}")
    diagnostic = block.get("teacher_forced_diagnostic") or {}
    if diagnostic:
        print(f"  teacher-forced C-B validation LL "
              f"{diagnostic['C_minus_B_validation_ll']:+.6f} "
              f"({diagnostic['label']})")


def print_shards(config: dict, actions=None) -> None:
    """The 1d partition, one line per shard (D11 check 11.1)."""
    block = ((config.get("arms") or {}).get("A") or {}).get("full_run") or {}
    shards = block.get("shards") or []
    if not shards:
        return
    by_index = {int(row["shard"]): row for row in (actions or [])}
    total = sum(int(shard["fixture_count"]) for shard in shards)
    print(f"1d shard partition ({len(shards)} shards, {total} fixtures, "
          f"{SHARD_COMMAND_DIFFERS_ONLY_IN[0]} / "
          f"{SHARD_COMMAND_DIFFERS_ONLY_IN[1]} are the only per-shard "
          "flags):")
    for shard in shards:
        action = by_index.get(int(shard["index"]), {}).get("action")
        suffix = "" if action is None else f"  [{action}]"
        print(f"  shard {shard['index']:>2}  "
              f"{shard['fixture_count']:>3} fixtures  "
              f"md5 {shard['fixture_dir_md5']}  "
              f"{shard['fixture_dir']}{suffix}")


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
            # Fail closed BEFORE copying anything. Materialising the shards
            # is the one write a pin makes outside the YAML, and a --write
            # that cannot produce a config must not leave 19 MB of fixture
            # copies behind: the retrain summary is the precondition that a
            # stale or absent retrain breaks (D6 check 6.8).
            selection_table(RETRAIN_SUMMARY)
            actions = materialize_full_run_shards()
            config = write_config(args.config, materialize=False)
            print_selection(config)
            print_shards(config, actions)
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
    print_shards(config)
    print("not compared (recorded only): " + ", ".join(NOT_VERIFIED))
    print(f"pin_stage1: OK — {args.config} matches every recomputed fact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
