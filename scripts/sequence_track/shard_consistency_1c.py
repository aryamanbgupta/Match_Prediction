#!/usr/bin/env python3
"""Stage 1 step 1c: sharding a run must not change any fixture's output.

`docs/sequence_track/stage1_acceptance.md`, D10 (checks 10.1-10.6). 1c is a
CONSISTENCY check, not a result: it reads no log loss, Brier, ROI or edge.
The comparison is byte equality of parsed JSON, and a failure is reported by
naming the first differing FIELD, never its value.

What it does
------------

1. Builds a case fixture set (`CASE_SET` below) covering every registered
   boundary case, and shards it by a stated deterministic rule
   (`SHARD_RULE`) chosen so the registered same-day pair lands in two
   different shards.
2. Runs every arm (A, A50, B, C) twice: once serially over the whole case
   set, and once per shard. Each run keeps the FULL replay context
   (`--context-dir data/t20s_json`), the same base seed, the same small
   n_sims and `--threads 1`. The commands are the registered `smoke_1a`
   command strings with `--config` dropped (the case set is not a registered
   block, so the runs are deliberately unregistered) and only the fixture
   dir, the simulation count and the output dir substituted.
3. Compares, per arm and per case fixture: the `eval.json` `matches` record,
   the `raw_sims.jsonl` row and the `arm_provenance.json` per-fixture entry
   (as-of stamp, eligibility, seeds), between the serial run and the shard
   that holds the fixture.

One field is EXCLUDED from the provenance comparison, and it is the whole
reason this step exists as a check rather than an assumption:

    as_of.matches_advanced

is a RUN-cumulative counter (`_TrackerStatsView.matches_advanced` increments
on every `advance_match` for the life of the process), so a shard that
starts on a later date has advanced fewer matches by the time it reaches a
fixture. It is not a per-fixture as-of fact. The shard-invariant as-of facts
are `as_of.date` and the ordered `as_of.same_day_advanced_before` list, both
compared in full; on top of that the script asserts the decomposition
`matches_advanced == (matches advanced on strictly earlier dates in this
run) + len(same_day_advanced_before)` and that the earlier-dates term is
constant across every fixture sharing a date within one run. That is the
part of the counter that carries same-day information, and it is checked.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

CONTRACT = "seq_stage1_shard_consistency_1c_v1"

DEFAULT_CONFIG = "experiments/configs/seq_stage1_sim_v1.yaml"
# Paths are composed from segments, or resolved through a manifest role, so
# no artifact of record is spelled as a literal (scripts/tests/
# test_manifest_defaults.py).
MODELS_ROOT = Path("models")
SEQ_STAGE1_ROOT = MODELS_ROOT / "embeddings" / "seq_stage1"
DEFAULT_OUT_ROOT = str(SEQ_STAGE1_ROOT / "consistency_1c")
DEFAULT_N_SIMS = 20
DEFAULT_BASE_SEED = 20260910
DEFAULT_SHARDS = 3
ARMS = ("A", "A50", "B", "C")

# The registered iteration set is an artifact of record: a case fixture names
# its manifest ROLE and the role is resolved at build time. The replay corpus
# is not a manifest artifact and is named directly.
ITERATION_SET_ROLE = "iteration_set_v2"
CONTEXT_DIR = "data/t20s_json"


def resolve_source_dir(source: str, repo_root: Path = REPO_ROOT) -> Path:
    """A case fixture's source: a manifest role, or a plain directory."""
    if source == ITERATION_SET_ROLE:
        from artifacts import artifact_path  # noqa: E402
        resolved = Path(artifact_path(ITERATION_SET_ROLE))
    else:
        resolved = Path(source)
    return resolved if resolved.is_absolute() else repo_root / resolved


EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REFUSED = 2

EVAL_FILENAME = "eval.json"
RAW_FILENAME = "raw_sims.jsonl"
PROVENANCE_FILENAME = "arm_provenance.json"


# ---------------------------------------------------------------------------
# The case set (check 10.1)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CaseFixture:
    cricsheet_id: str
    match_date: str
    source_dir: str
    cases: Tuple[str, ...]
    rule: str


# Every id was chosen by the rule recorded beside it, from
# `data/polymarket_test_v2` (the registered iteration set, 255 fixtures,
# 2025-09-10 .. 2026-04-16 on disk) and `data/t20s_json` (the 11,264-fixture
# replay corpus). The case labels are the five registered boundary cases:
#   a  same-day pair, split across two shards
#   b  the first and the last date of the iteration set
#   c  a male T20 inside the window with NO odds row
#   d  an iteration fixture whose date carries a male T20 that stays only in
#      the context corpus
#   e  the first fixture of a shard (process initialisation)
#   ctl a control: an iteration fixture that is the only male T20 on its date
CASE_SET: Tuple[CaseFixture, ...] = (
    CaseFixture(
        cricsheet_id="1448357", match_date="2025-09-10",
        source_dir=CONTEXT_DIR, cases=("c", "e"),
        rule="lowest cricsheet id among male T20s in data/t20s_json inside "
             "the iteration window whose id is not in the odds file"),
    CaseFixture(
        cricsheet_id="1496921", match_date="2025-09-10",
        source_dir=ITERATION_SET_ROLE, cases=("b", "d", "e"),
        rule="the only iteration fixture on the FIRST date of the iteration "
             "set; its date also carries 1477997, a male T20 that stays only "
             "in the context corpus"),
    CaseFixture(
        cricsheet_id="1477609", match_date="2026-01-27",
        source_dir=ITERATION_SET_ROLE, cases=("ctl", "e"),
        rule="lowest iteration id that is the ONLY male T20 on its date "
             "(no same-day sibling at all)"),
    CaseFixture(
        cricsheet_id="1477610", match_date="2026-01-29",
        source_dir=ITERATION_SET_ROLE, cases=("d",),
        rule="lowest iteration id whose date carries male T20 siblings "
             "(1519139, 1519636) absent from the iteration set, on a date "
             "not already in the case set"),
    CaseFixture(
        cricsheet_id="1512766", match_date="2026-02-26",
        source_dir=ITERATION_SET_ROLE, cases=("a",),
        rule="registered same-day pair, lower id (2026-02-26)"),
    CaseFixture(
        cricsheet_id="1525160", match_date="2026-02-26",
        source_dir=ITERATION_SET_ROLE, cases=("a",),
        rule="registered same-day pair, higher id (2026-02-26)"),
    CaseFixture(
        cricsheet_id="1527575", match_date="2026-04-16",
        source_dir=ITERATION_SET_ROLE, cases=("b",),
        rule="lowest iteration id on the LAST date of the iteration set"),
    CaseFixture(
        cricsheet_id="1527576", match_date="2026-04-16",
        source_dir=ITERATION_SET_ROLE, cases=("a", "b"),
        rule="second-lowest iteration id on the LAST date, so the boundary "
             "date is a second same-day pair split across shards (1529267 "
             "stays context-only in both runs)"),
)

SHARD_RULE = (
    "round_robin_over_date_then_id_v1: the case fixtures are sorted by "
    "(match_date, cricsheet_id) and the fixture at index i goes to shard "
    "i % <n_shards>. Round-robin over that order puts consecutive fixtures "
    "in different shards, which is what splits the same-day pairs.")

# Provenance fields not compared, and why.
PROVENANCE_EXCLUSIONS: Dict[str, str] = {
    "as_of.matches_advanced":
        "run-cumulative counter (_TrackerStatsView.matches_advanced "
        "increments on every advance_match for the life of the process), so "
        "a shard that starts on a later date has advanced fewer matches by "
        "the time it reaches the fixture. It is a run-level field, not a "
        "per-fixture as-of fact. The as-of facts it is made of are compared: "
        "as_of.date, the ordered as_of.same_day_advanced_before list, and "
        "the decomposition asserted by check_matches_advanced_decomposition",
}

# Differences 1c has DIAGNOSED. A diagnosis does not suppress the failure —
# the field still differs and the verdict is still FAIL — it records what the
# cause is and what 1d has to do about it.
KNOWN_SHARD_DEPENDENT_FIELDS: Dict[str, str] = {
    "competition_cluster_id":
        "run_sim_eval.py builds its cluster lookup with "
        "load_competition_clusters(args.test_dir), i.e. from the FIXTURE DIR "
        "of that run. The I3 block id is `event:<name>|block_start:<date>` "
        "and block_start is the first date of the event's members IN THAT "
        "DIRECTORY, so a shard that holds only the later fixture of an event "
        "stamps a later block start. This is the registered block key "
        "(critical invariant 7, tournament_time_block_v1), so a sharded 1d "
        "run must re-derive it over the whole registered fixture set — "
        "merge_shards.py --cluster-source-dir does exactly that, and "
        "cluster_id_with_resolution prefers a STAMPED id, so reslicing alone "
        "would not repair it",
}

# Run-level provenance fields that MUST agree between the serial run and a
# shard run: they identify the model, the caches and the draw schedule.
RUN_IDENTITY_KEYS = (
    "arm", "model_dir", "model_dir_hash", "checkpoint_path", "checkpoint_md5",
    "stats_version", "stats_cache_path", "stats_cache_md5", "selector_class",
    "bowler_usage_path", "bowler_usage_md5", "bowler_roster_policy_path",
    "bowler_roster_policy_md5", "extras_graft_path", "extras_graft_sha256",
    "extras_graft_applied_via", "runout_p", "clip", "base_seed", "n_sims",
    "engine_md5", "runner_md5", "threads", "device", "context_dir",
    "context_dir_hash", "odds", "odds_sha256", "player_metadata_sha256",
    # The corpus the I3 block ids were stamped from: a shard and the serial
    # run must build the SAME lookup, which is the D10 fix this step checks.
    "cluster_source_dir", "cluster_source_dir_hash",
)

ARTEFACTS = ("eval_record", "raw_sims_row", "provenance")

# Numbers this script may write. Anything else fails the guard, which is how
# a metric can never reach the 1c outputs.
ALLOWED_NUMERIC_KEYS = frozenset({
    "n_sims", "base_seed", "n_shards", "shard", "left_shard", "right_shard",
    "index", "exit_code", "wall_seconds", "fixture_count", "n_fixtures",
    "n_cases", "n_pass", "n_fail", "n_checks", "n_serial_runs",
    "n_shard_runs", "same_day_advanced_count", "matches_advanced",
    "earlier_date_matches",
})


class ConsistencyError(RuntimeError):
    """Refusal: the 1c setup itself is wrong."""


# ---------------------------------------------------------------------------
# Case set / sharding
# ---------------------------------------------------------------------------

def ordered_cases(cases: Sequence[CaseFixture] = CASE_SET
                  ) -> List[CaseFixture]:
    """Case fixtures in registered `(match_date, cricsheet_id)` order."""
    return sorted(cases, key=lambda case: (case.match_date,
                                           case.cricsheet_id))


def shard_assignment(cases: Sequence[CaseFixture] = CASE_SET,
                     n_shards: int = DEFAULT_SHARDS) -> Dict[str, int]:
    """`cricsheet id -> shard index` under `SHARD_RULE`."""
    if n_shards < 2:
        raise ConsistencyError("1c needs at least 2 shards")
    return {case.cricsheet_id: index % n_shards
            for index, case in enumerate(ordered_cases(cases))}


def shard_members(cases: Sequence[CaseFixture] = CASE_SET,
                  n_shards: int = DEFAULT_SHARDS) -> Dict[int, List[str]]:
    """`shard index -> its ids`, each shard in (date, id) order."""
    assignment = shard_assignment(cases, n_shards)
    members: Dict[int, List[str]] = {index: [] for index in range(n_shards)}
    for case in ordered_cases(cases):
        members[assignment[case.cricsheet_id]].append(case.cricsheet_id)
    return members


def same_day_pairs(cases: Sequence[CaseFixture] = CASE_SET
                   ) -> List[Tuple[str, str]]:
    """Every pair of case fixtures sharing a date, in (date, id) order."""
    by_date: Dict[str, List[str]] = {}
    for case in ordered_cases(cases):
        by_date.setdefault(case.match_date, []).append(case.cricsheet_id)
    pairs = []
    for ids in by_date.values():
        for left_index in range(len(ids)):
            for right_index in range(left_index + 1, len(ids)):
                pairs.append((ids[left_index], ids[right_index]))
    return pairs


def check_case_set(cases: Sequence[CaseFixture] = CASE_SET,
                   n_shards: int = DEFAULT_SHARDS) -> List[str]:
    """The case set must actually cover the registered boundary cases."""
    problems = []
    labels = {label for case in cases for label in case.cases}
    for required in ("a", "b", "c", "d"):
        if required not in labels:
            problems.append(f"no case fixture carries boundary case {required!r}")
    assignment = shard_assignment(cases, n_shards)
    split = [pair for pair in same_day_pairs(cases)
             if assignment[pair[0]] != assignment[pair[1]]]
    if not split:
        problems.append(
            "no same-day pair is split across shards; case (a) is not covered "
            "by this shard rule")
    members = shard_members(cases, n_shards)
    empty = [index for index, ids in members.items() if not ids]
    if empty:
        problems.append(f"shard(s) {empty} would be empty")
    ids = [case.cricsheet_id for case in cases]
    if len(set(ids)) != len(ids):
        problems.append("the case set repeats a cricsheet id")
    return problems


def build_fixture_dirs(out_root: Path, cases: Sequence[CaseFixture] = CASE_SET,
                       n_shards: int = DEFAULT_SHARDS,
                       repo_root: Path = REPO_ROOT) -> Dict[str, Path]:
    """Materialise the serial and per-shard fixture dirs by copying JSONs."""
    problems = check_case_set(cases, n_shards)
    if problems:
        raise ConsistencyError("; ".join(problems))
    assignment = shard_assignment(cases, n_shards)
    dirs = {"serial": out_root / "serial" / "fixtures"}
    for index in range(n_shards):
        dirs[f"shard{index}"] = out_root / "shards" / str(index) / "fixtures"
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
        for stale in path.glob("*.json"):
            stale.unlink()
    for case in ordered_cases(cases):
        source = (resolve_source_dir(case.source_dir, repo_root)
                  / f"{case.cricsheet_id}.json")
        if not source.is_file():
            raise ConsistencyError(f"case fixture not found: {source}")
        payload = source.read_bytes()
        document = json.loads(payload)
        actual_date = str(document["info"]["dates"][0])
        if actual_date != case.match_date:
            raise ConsistencyError(
                f"{case.cricsheet_id}: recorded date {case.match_date} but "
                f"the document says {actual_date}")
        for target in (dirs["serial"],
                       dirs[f"shard{assignment[case.cricsheet_id]}"]):
            (target / f"{case.cricsheet_id}.json").write_bytes(payload)
    return dirs


# ---------------------------------------------------------------------------
# Commands (check 10.2)
# ---------------------------------------------------------------------------

def smoke_command(config: dict, arm: str) -> str:
    block = ((config.get("arms") or {}).get(arm) or {}).get("smoke_1a") or {}
    command = block.get("command")
    if not command:
        raise ConsistencyError(f"config has no arms.{arm}.smoke_1a.command")
    return str(command)


def render_command(command: str, *, fixture_dir, n_sims: int, output_dir,
                   base_seed: Optional[int] = None,
                   cluster_source_dir=None) -> List[str]:
    """The registered smoke command with `--config` dropped.

    Only three tokens are substituted (fixture dir, n_sims, output dir). The
    env prefix, every model/cache/sidecar path, the clip bounds, the base
    seed and `--threads 1` are carried through verbatim, so a 1c run differs
    from the registered smoke run only in what it must.

    `cluster_source_dir` is a fourth substitution, and it is what 1c is
    checking: the run must stamp its I3 block ids from ONE corpus whichever
    shard it is, and for 1c that corpus is the whole case set (the registered
    iteration set does not contain every case fixture). The registered
    command carries `--cluster-source-dir` since the D10 fix; if it does not,
    the flag is appended rather than silently dropped.
    """
    tokens = shlex.split(command)
    out: List[str] = []
    replacements = {
        "--fixture-dir": str(fixture_dir),
        "--n-sims": str(int(n_sims)),
        "--output-dir": str(output_dir),
    }
    if cluster_source_dir is not None:
        replacements["--cluster-source-dir"] = str(cluster_source_dir)
    seen = {flag: 0 for flag in replacements}
    dropped_config = 0
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--config":
            dropped_config += 1
            index += 2
            continue
        if token in replacements:
            seen[token] += 1
            out.append(token)
            out.append(replacements[token])
            index += 2
            continue
        out.append(token)
        index += 1
    if dropped_config != 1:
        raise ConsistencyError(
            f"expected exactly one --config in the registered command, "
            f"found {dropped_config}")
    for flag, count in seen.items():
        if count == 0 and flag == "--cluster-source-dir":
            out += [flag, replacements[flag]]
            continue
        if count != 1:
            raise ConsistencyError(
                f"expected exactly one {flag} in the registered command, "
                f"found {count}")
    if "--threads" not in out or out[out.index("--threads") + 1] != "1":
        raise ConsistencyError("the rendered command must keep --threads 1")
    if not any(token == "OMP_NUM_THREADS=1" for token in out):
        raise ConsistencyError(
            "the rendered command must keep the OMP_NUM_THREADS=1 env")
    if base_seed is not None:
        if "--base-seed" not in out:
            raise ConsistencyError("the registered command has no --base-seed")
        found = out[out.index("--base-seed") + 1]
        if found != str(int(base_seed)):
            raise ConsistencyError(
                f"the registered command's --base-seed is {found}, not the "
                f"{base_seed} this run records")
    return out


def repo_relative(path) -> str:
    """A repo-relative spelling when possible: the runs have cwd=REPO_ROOT."""
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(candidate)


@dataclass
class RunSpec:
    label: str
    arm: str
    shard: Optional[int]
    fixture_dir: Path
    output_dir: Path
    argv: List[str]

    @property
    def command(self) -> str:
        return shlex.join(self.argv)

    @property
    def log_path(self) -> Path:
        return self.output_dir / "run.log"


@dataclass
class RunResult:
    label: str
    arm: str
    shard: Optional[int]
    output_dir: str
    exit_code: int
    wall_seconds: float
    skipped: bool = False


def plan_runs(config: dict, out_root: Path, fixture_dirs: Dict[str, Path], *,
              arms: Sequence[str] = ARMS, n_sims: int = DEFAULT_N_SIMS,
              n_shards: int = DEFAULT_SHARDS,
              base_seed: Optional[int] = DEFAULT_BASE_SEED) -> List[RunSpec]:
    specs: List[RunSpec] = []
    # The case set IS the registered set for 1c (it deliberately holds
    # fixtures that are not in the iteration set), so every run — serial and
    # sharded alike — builds its competition-cluster lookup from the SERIAL
    # case fixture directory. That is the whole point of check 10.x: the
    # block id a fixture is stamped with must not depend on which shard ran
    # it.
    cluster_source = repo_relative(fixture_dirs["serial"])
    for arm in arms:
        command = smoke_command(config, arm)
        serial_out = out_root / "serial" / arm
        specs.append(RunSpec(
            label=f"serial/{arm}", arm=arm, shard=None,
            fixture_dir=fixture_dirs["serial"], output_dir=serial_out,
            argv=render_command(
                command, fixture_dir=repo_relative(fixture_dirs["serial"]),
                n_sims=n_sims, output_dir=repo_relative(serial_out),
                base_seed=base_seed, cluster_source_dir=cluster_source)))
    for index in range(n_shards):
        for arm in arms:
            command = smoke_command(config, arm)
            shard_out = out_root / "shards" / str(index) / arm
            specs.append(RunSpec(
                label=f"shard{index}/{arm}", arm=arm, shard=index,
                fixture_dir=fixture_dirs[f"shard{index}"],
                output_dir=shard_out,
                argv=render_command(
                    command,
                    fixture_dir=repo_relative(fixture_dirs[f"shard{index}"]),
                    n_sims=n_sims, output_dir=repo_relative(shard_out),
                    base_seed=base_seed,
                    cluster_source_dir=cluster_source)))
    return specs


def is_complete(output_dir) -> bool:
    path = Path(output_dir)
    return all((path / name).is_file()
               for name in (EVAL_FILENAME, RAW_FILENAME,
                            PROVENANCE_FILENAME))


RUNS_RECORD = "runs_record.json"


def load_runs_record(path) -> Dict[str, dict]:
    """`label -> run record` from a previous invocation, or empty."""
    record = Path(path)
    if not record.is_file():
        return {}
    try:
        return {str(key): value
                for key, value in json.loads(record.read_text()).items()}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_runs_record(path, results: Sequence[RunResult]) -> None:
    payload = {result.label: {
        "label": result.label, "arm": result.arm, "shard": result.shard,
        "output_dir": result.output_dir, "exit_code": result.exit_code,
        "wall_seconds": result.wall_seconds, "skipped": result.skipped}
        for result in results}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def launch(specs: Sequence[RunSpec], *, max_concurrent: int = 4,
           cwd: Path = REPO_ROOT, force: bool = False,
           record_path=None) -> List[RunResult]:
    """Run every spec, at most `max_concurrent` processes at a time.

    A completed output dir is skipped, and its wall time and exit code are
    taken from `record_path` so a later comparison-only invocation does not
    report every run as having taken no time.
    """
    previous = load_runs_record(record_path) if record_path else {}
    results: List[RunResult] = []
    pending = [spec for spec in specs]
    for spec in list(pending):
        if not force and is_complete(spec.output_dir):
            print(f"[1c] skip (complete): {spec.label}")
            prior = previous.get(spec.label, {})
            results.append(RunResult(
                label=spec.label, arm=spec.arm, shard=spec.shard,
                output_dir=str(spec.output_dir),
                exit_code=int(prior.get("exit_code", 0)),
                wall_seconds=float(prior.get("wall_seconds", 0.0)),
                skipped=True))
            pending.remove(spec)

    running: Dict[str, Tuple[RunSpec, subprocess.Popen, object, float]] = {}
    queue = list(pending)

    def _start(spec: RunSpec):
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        handle = spec.log_path.open("a", encoding="utf-8", errors="replace")
        handle.write(f"=== shard_consistency_1c {spec.label} "
                     f"{datetime.now(timezone.utc).isoformat()} ===\n")
        handle.write(f"$ {spec.command}\n")
        handle.flush()
        print(f"[1c] launching {spec.label}")
        process = subprocess.Popen(spec.argv, cwd=str(cwd), stdout=handle,
                                   stderr=subprocess.STDOUT,
                                   env=os.environ.copy())
        running[spec.label] = (spec, process, handle, time.monotonic())

    try:
        while queue or running:
            while queue and len(running) < max(1, int(max_concurrent)):
                _start(queue.pop(0))
            time.sleep(0.5)
            for label in list(running):
                spec, process, handle, started = running[label]
                if process.poll() is None:
                    continue
                handle.close()
                del running[label]
                results.append(RunResult(
                    label=spec.label, arm=spec.arm, shard=spec.shard,
                    output_dir=str(spec.output_dir),
                    exit_code=int(process.returncode),
                    wall_seconds=round(time.monotonic() - started, 1)))
                print(f"[1c] finished {spec.label}: exit "
                      f"{process.returncode} in "
                      f"{results[-1].wall_seconds:.1f}s")
    except BaseException:
        for _label, (_spec, process, handle, _started) in running.items():
            if process.poll() is None:
                process.terminate()
            handle.close()
        raise
    results = sorted(results, key=lambda result: result.label)
    if record_path:
        _write_runs_record(record_path, results)
    return results


# ---------------------------------------------------------------------------
# Comparison (check 10.3)
# ---------------------------------------------------------------------------

def first_difference(left, right, *, path: str = "",
                     exclude: Iterable[str] = ()) -> Optional[str]:
    """The dotted path of the first difference, or None.

    Only the PATH is returned. No value is ever produced, so a metric field
    can be named as differing without its level being read.
    """
    exclude = set(exclude)
    if path in exclude:
        return None
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else str(key)
            if child in exclude:
                continue
            if (key in left) != (key in right):
                return child
            found = first_difference(left[key], right[key], path=child,
                                     exclude=exclude)
            if found is not None:
                return found
        return None
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return f"{path}[len]" if path else "[len]"
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            child = f"{path}[{index}]"
            found = first_difference(left_item, right_item, path=child,
                                     exclude=exclude)
            if found is not None:
                return found
        return None
    if type(left) is not type(right) and not (
            isinstance(left, (int, float)) and isinstance(right, (int, float))
            and not isinstance(left, bool) and not isinstance(right, bool)):
        return path or "<root>"
    if left != right:
        return path or "<root>"
    return None


def all_differences(left, right, *, path: str = "",
                    exclude: Iterable[str] = (), limit: int = 40
                    ) -> List[str]:
    """Every differing path, names only, up to `limit`."""
    exclude = set(exclude)
    found: List[str] = []

    def _walk(left_value, right_value, current: str) -> None:
        if len(found) >= limit or current in exclude:
            return
        if isinstance(left_value, dict) and isinstance(right_value, dict):
            for key in sorted(set(left_value) | set(right_value)):
                child = f"{current}.{key}" if current else str(key)
                if child in exclude:
                    continue
                if (key in left_value) != (key in right_value):
                    found.append(child)
                    continue
                _walk(left_value[key], right_value[key], child)
            return
        if isinstance(left_value, list) and isinstance(right_value, list):
            if len(left_value) != len(right_value):
                found.append(f"{current}[len]" if current else "[len]")
                return
            for index, (one, other) in enumerate(zip(left_value,
                                                     right_value)):
                _walk(one, other, f"{current}[{index}]")
            return
        if left_value != right_value:
            found.append(current or "<root>")

    _walk(left, right, path)
    return found


def diagnose(paths: Sequence[str]) -> Optional[str]:
    """The recorded diagnosis for a differing path, if there is one."""
    for candidate in paths:
        leaf = candidate.split(".")[-1].split("[")[0]
        if leaf in KNOWN_SHARD_DEPENDENT_FIELDS:
            return KNOWN_SHARD_DEPENDENT_FIELDS[leaf]
    return None


@dataclass
class RunArtefacts:
    label: str
    path: Path
    arm: str
    run: dict
    match_identity: dict
    eval_records: Dict[str, dict] = field(default_factory=dict)
    raw_rows: Dict[str, dict] = field(default_factory=dict)
    fixtures: Dict[str, dict] = field(default_factory=dict)


def _record_key(record: dict) -> str:
    for key in ("cricsheet_id", "match_id", "display_match_id"):
        value = record.get(key)
        if value:
            return str(value)
    raise ConsistencyError("evaluator record carries no match identity")


def load_run(directory, label: Optional[str] = None) -> RunArtefacts:
    path = Path(directory)
    provenance_path = path / PROVENANCE_FILENAME
    eval_path = path / EVAL_FILENAME
    for required in (provenance_path, eval_path):
        if not required.is_file():
            raise ConsistencyError(f"{path}: missing {required.name}")
    provenance = json.loads(provenance_path.read_text())
    evaluation = json.loads(eval_path.read_text())
    artefacts = RunArtefacts(
        label=str(label if label is not None else path.name),
        path=path,
        arm=str(provenance.get("arm")),
        run=provenance.get("run") or {},
        match_identity=evaluation.get("match_identity") or {},
    )
    for row in provenance.get("fixtures", []):
        artefacts.fixtures[str(row["cricsheet_id"])] = row
    for record in evaluation.get("matches", []):
        artefacts.eval_records[_record_key(record)] = record
    raw_path = path / RAW_FILENAME
    if raw_path.is_file():
        with raw_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                artefacts.raw_rows[str(row["match_id"])] = row
    return artefacts


def compare_run_identity(serial: RunArtefacts, shard: RunArtefacts
                         ) -> Tuple[bool, Optional[str]]:
    """The two runs must be the same arm, model, cache and seed."""
    if serial.arm != shard.arm:
        return False, "arm"
    for key in RUN_IDENTITY_KEYS:
        found = first_difference(serial.run.get(key), shard.run.get(key),
                                 path=f"run.{key}")
        if found is not None:
            return False, found
    found = first_difference(serial.match_identity, shard.match_identity,
                             path="match_identity")
    if found is not None:
        return False, found
    return True, None


@dataclass
class ArtefactResult:
    arm: str
    fixture: str
    artefact: str
    status: str                      # PASS | FAIL
    detail: str                      # field name, or the presence note
    shard: Optional[int] = None
    differences: Tuple[str, ...] = ()
    diagnosis: Optional[str] = None


def compare_fixture(serial: RunArtefacts, shard: RunArtefacts, fixture: str,
                    *, shard_index: Optional[int] = None
                    ) -> List[ArtefactResult]:
    """The three per-fixture artefacts, compared field by field."""
    results: List[ArtefactResult] = []

    def _presence(artefact, left, right):
        if (left is None) != (right is None):
            missing = "serial" if left is None else f"shard{shard_index}"
            results.append(ArtefactResult(
                serial.arm, fixture, artefact, "FAIL",
                f"present in one run only (absent in {missing})",
                shard_index))
            return False
        return True

    def _compare(artefact, left, right, *, exclude=(), identical_note):
        found = first_difference(left, right, exclude=exclude)
        paths = () if found is None else tuple(
            all_differences(left, right, exclude=exclude))
        results.append(ArtefactResult(
            serial.arm, fixture, artefact,
            "PASS" if found is None else "FAIL",
            identical_note if found is None else f"first difference: {found}",
            shard_index, paths, diagnose(paths)))

    serial_record = serial.eval_records.get(fixture)
    shard_record = shard.eval_records.get(fixture)
    if _presence("eval_record", serial_record, shard_record):
        if serial_record is None:
            results.append(ArtefactResult(
                serial.arm, fixture, "eval_record", "PASS",
                "absent in both (fixture is not scored)", shard_index))
        else:
            _compare("eval_record", serial_record, shard_record,
                     identical_note="identical")

    serial_raw = serial.raw_rows.get(fixture)
    shard_raw = shard.raw_rows.get(fixture)
    if _presence("raw_sims_row", serial_raw, shard_raw):
        if serial_raw is None:
            results.append(ArtefactResult(
                serial.arm, fixture, "raw_sims_row", "PASS",
                "absent in both (fixture is not simulated)", shard_index))
        else:
            _compare("raw_sims_row", serial_raw, shard_raw,
                     identical_note="identical")

    serial_prov = serial.fixtures.get(fixture)
    shard_prov = shard.fixtures.get(fixture)
    if _presence("provenance", serial_prov, shard_prov):
        if serial_prov is None:
            results.append(ArtefactResult(
                serial.arm, fixture, "provenance", "FAIL",
                "the fixture was never stamped in either run", shard_index))
        else:
            _compare("provenance", serial_prov, shard_prov,
                     exclude=PROVENANCE_EXCLUSIONS,
                     identical_note="identical outside the excluded field")
    return results


def check_matches_advanced_decomposition(run: RunArtefacts) -> List[str]:
    """`matches_advanced - len(same_day_advanced_before)` per date.

    Within one run that term is the number of matches advanced on strictly
    earlier dates, so it must be constant across every fixture on a date and
    non-decreasing in date. This is what makes the excluded counter's
    shard-invariant part checkable.
    """
    problems: List[str] = []
    by_date: Dict[str, Dict[str, int]] = {}
    for fixture, row in sorted(run.fixtures.items()):
        as_of = row["as_of"]
        earlier = (int(as_of["matches_advanced"])
                   - len(as_of["same_day_advanced_before"]))
        if earlier < 0:
            problems.append(
                f"{run.label}: fixture {fixture} advanced more same-day "
                "matches than the run has advanced in total")
        by_date.setdefault(str(as_of["date"]), {})[fixture] = earlier
    for date in sorted(by_date):
        values = set(by_date[date].values())
        if len(values) > 1:
            problems.append(
                f"{run.label}: fixtures on {date} disagree on how many "
                "matches were advanced on earlier dates "
                f"({sorted(by_date[date])})")
    previous = None
    for date in sorted(by_date):
        value = min(by_date[date].values())
        if previous is not None and value < previous:
            problems.append(
                f"{run.label}: the earlier-date match count falls at {date}")
        previous = value
    return problems


# ---------------------------------------------------------------------------
# Reporting (check 10.6)
# ---------------------------------------------------------------------------

def assert_no_metric_values(payload, *, path: str = "",
                            allowed: Iterable[str] = ALLOWED_NUMERIC_KEYS
                            ) -> None:
    """Refuse an output carrying a number outside the allowed count keys."""
    allowed = set(allowed)
    if isinstance(payload, dict):
        for key in sorted(payload):
            child = f"{path}.{key}" if path else str(key)
            value = payload[key]
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and key not in allowed:
                raise ConsistencyError(
                    f"1c output would carry a number at {child!r}; 1c writes "
                    "counts and field names only")
            assert_no_metric_values(value, path=child, allowed=allowed)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            assert_no_metric_values(value, path=f"{path}[{index}]",
                                    allowed=allowed)


def build_report(*, cases: Sequence[CaseFixture], n_shards: int,
                 n_sims: int, base_seed: int, specs: Sequence[RunSpec],
                 runs: Sequence[RunResult], results: Sequence[ArtefactResult],
                 identity_rows: Sequence[dict],
                 decomposition_problems: Sequence[str],
                 merge_rows: Sequence[dict] = ()) -> dict:
    assignment = shard_assignment(cases, n_shards)
    members = shard_members(cases, n_shards)
    n_fail = sum(1 for row in results if row.status == "FAIL")
    n_fail += sum(1 for row in identity_rows if row["status"] == "FAIL")
    n_fail += len(decomposition_problems)
    n_fail += sum(1 for row in merge_rows if row["status"] == "FAIL")
    payload = {
        "contract": CONTRACT,
        "generated_at": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "step": "stage 1 step 1c (shard consistency)",
        "acceptance": "docs/sequence_track/stage1_acceptance.md, D10",
        "reads_no_metric": (
            "no log loss, Brier, ROI or edge is read: the comparison is byte "
            "equality of parsed JSON and a difference is reported by field "
            "name only"),
        "n_sims": int(n_sims),
        "base_seed": int(base_seed),
        "n_shards": int(n_shards),
        "shard_rule": SHARD_RULE,
        "context_dir": CONTEXT_DIR,
        "cases": [
            {
                "cricsheet_id": case.cricsheet_id,
                "match_date": case.match_date,
                "source_dir": case.source_dir,
                "boundary_cases": list(case.cases),
                "selection_rule": case.rule,
                "shard": assignment[case.cricsheet_id],
                "first_of_shard": bool(
                    members[assignment[case.cricsheet_id]][0]
                    == case.cricsheet_id),
            }
            for case in ordered_cases(cases)
        ],
        "shard_members": {str(index): ids for index, ids in members.items()},
        "same_day_pairs_split": [
            {"left": left, "right": right,
             "left_shard": assignment[left], "right_shard": assignment[right]}
            for left, right in same_day_pairs(cases)
            if assignment[left] != assignment[right]
        ],
        "commands": [
            {"label": spec.label, "arm": spec.arm,
             "shard": spec.shard, "command": spec.command}
            for spec in specs
        ],
        "runs": [
            {"label": run.label, "arm": run.arm, "shard": run.shard,
             "output_dir": run.output_dir, "exit_code": run.exit_code,
             "wall_seconds": run.wall_seconds, "skipped": run.skipped}
            for run in runs
        ],
        "excluded_fields": [
            {"artefact": "provenance", "field": field_name, "reason": reason}
            for field_name, reason in sorted(PROVENANCE_EXCLUSIONS.items())
        ],
        "run_identity": list(identity_rows),
        "matches_advanced_decomposition": {
            "status": "PASS" if not decomposition_problems else "FAIL",
            "problems": list(decomposition_problems),
        },
        "merge_assertions": list(merge_rows),
        "known_shard_dependent_fields": [
            {"field": name, "explanation": text}
            for name, text in sorted(KNOWN_SHARD_DEPENDENT_FIELDS.items())
        ],
        "results": [
            {"arm": row.arm, "fixture": row.fixture, "shard": row.shard,
             "artefact": row.artefact, "status": row.status,
             "detail": row.detail,
             "differences": list(row.differences),
             "diagnosis": row.diagnosis}
            for row in results
        ],
        "summary": {
            "n_cases": len(cases),
            "n_checks": (len(results) + len(identity_rows)
                         + len(merge_rows) + 1),
            "n_fail": n_fail,
            "verdict": "PASS" if n_fail == 0 else "FAIL",
        },
    }
    assert_no_metric_values(payload)
    return payload


def render_markdown(payload: dict) -> str:
    lines: List[str] = []
    add = lines.append
    add("# Stage 1 step 1c — shard consistency")
    add("")
    add(f"Contract `{payload['contract']}`, written "
        f"{payload['generated_at']}.")
    add(f"Acceptance: {payload['acceptance']}.")
    add("")
    add("**No metric is read.** " + payload["reads_no_metric"] + ".")
    add("")
    add(f"n_sims {payload['n_sims']}, base seed {payload['base_seed']}, "
        f"{payload['n_shards']} shards, replay context "
        f"`{payload['context_dir']}`, threads 1, runs unregistered "
        "(`--config` dropped: the case set is not a registered block).")
    add("")
    add("## Shard rule")
    add("")
    add(payload["shard_rule"])
    add("")
    for index, ids in sorted(payload["shard_members"].items()):
        add(f"* shard {index}: {', '.join(ids)}")
    add("")
    add("Same-day pairs split across shards: "
        + "; ".join(f"{row['left']} (shard {row['left_shard']}) / "
                    f"{row['right']} (shard {row['right_shard']})"
                    for row in payload["same_day_pairs_split"]))
    add("")
    add("## Case set")
    add("")
    add("| cricsheet id | date | source | cases | shard | first of shard | "
        "selection rule |")
    add("|---|---|---|---|---|---|---|")
    for case in payload["cases"]:
        add("| {id} | {date} | `{src}` | {cases} | {shard} | {first} | {rule} "
            "|".format(id=case["cricsheet_id"], date=case["match_date"],
                       src=case["source_dir"],
                       cases=", ".join(case["boundary_cases"]),
                       shard=case["shard"],
                       first="yes" if case["first_of_shard"] else "",
                       rule=case["selection_rule"]))
    add("")
    add("Boundary cases: (a) same-day pair split across shards; (b) first and "
        "last date of the iteration set; (c) a male T20 with no odds row; "
        "(d) an iteration fixture whose date carries a male T20 that stays "
        "only in the context corpus; (e) the first fixture of a shard; "
        "(ctl) a control with no same-day sibling at all.")
    add("")
    add("## Commands")
    add("")
    add("```")
    for row in payload["commands"]:
        add(f"# {row['label']}")
        add(row["command"])
    add("```")
    add("")
    add("## Runs")
    add("")
    add("| run | exit | wall s | skipped |")
    add("|---|---|---|---|")
    for row in payload["runs"]:
        add(f"| {row['label']} | {row['exit_code']} | "
            f"{row['wall_seconds']} | {'yes' if row['skipped'] else ''} |")
    add("")
    add("## Excluded fields")
    add("")
    for row in payload["excluded_fields"]:
        add(f"* `{row['artefact']}` → `{row['field']}`: {row['reason']}")
    add("")
    add("Decomposition check on the excluded counter: "
        f"**{payload['matches_advanced_decomposition']['status']}**")
    for problem in payload["matches_advanced_decomposition"]["problems"]:
        add(f"  * {problem}")
    add("")
    add("## Run identity (serial vs shard)")
    add("")
    add("| arm | shard | status | first difference |")
    add("|---|---|---|---|")
    for row in payload["run_identity"]:
        add(f"| {row['arm']} | {row['shard']} | {row['status']} | "
            f"{row.get('detail') or ''} |")
    add("")
    if payload["merge_assertions"]:
        add("## Merge assertions (check 10.5)")
        add("")
        add("| arm | assertion | status | detail |")
        add("|---|---|---|---|")
        for row in payload["merge_assertions"]:
            add(f"| {row['arm']} | {row['assertion']} | {row['status']} | "
                f"{row.get('detail') or ''} |")
        add("")
    add("## Per (arm, fixture, artefact)")
    add("")
    add("| arm | fixture | shard | artefact | status | detail | all "
        "differing fields |")
    add("|---|---|---|---|---|---|---|")
    for row in payload["results"]:
        add(f"| {row['arm']} | {row['fixture']} | {row['shard']} | "
            f"{row['artefact']} | {row['status']} | {row['detail']} | "
            f"{', '.join(row.get('differences') or []) or '—'} |")
    add("")
    diagnosed = [row for row in payload["results"] if row.get("diagnosis")]
    if diagnosed:
        add("## Diagnosed differences")
        add("")
        add("A diagnosis does NOT suppress the failure: the field differs, "
            "the verdict is FAIL, and the note records the cause.")
        add("")
        for row in payload["known_shard_dependent_fields"]:
            hits = sorted({(hit["arm"], hit["fixture"])
                           for hit in diagnosed
                           if row["field"] in " ".join(hit["differences"])})
            add(f"* `{row['field']}` — {row['explanation']}.")
            add(f"  Hit on: {', '.join(f'{arm}/{fixture}' for arm, fixture in hits)}")
        add("")
    summary = payload["summary"]
    add(f"**Verdict: {summary['verdict']}** — {summary['n_checks']} checks, "
        f"{summary['n_fail']} failing, over {summary['n_cases']} case "
        "fixtures.")
    add("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 1 step 1c: a sharded run must reproduce the "
                    "serial run's per-fixture outputs byte for byte")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--n-sims", type=int, default=DEFAULT_N_SIMS)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--shards", type=int, default=DEFAULT_SHARDS)
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--odds", default="betting_odds_polymarket_v2.json")
    parser.add_argument("--stage", default="all",
                        choices=("build", "run", "compare", "all"))
    parser.add_argument("--force", action="store_true",
                        help="re-run even a complete output dir")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the rendered commands and stop")
    return parser


def _run_merge(out_root: Path, arm: str, n_shards: int, odds: str,
               fixture_dirs: Dict[str, Path]) -> List[dict]:
    """Run the merge assertions for one arm, then merged-vs-serial."""
    from sequence_track import merge_shards

    shard_dirs = [str(out_root / "shards" / str(index) / arm)
                  for index in range(n_shards)]
    merged_dir = out_root / "merged" / arm
    argv = []
    for directory in shard_dirs:
        argv += ["--shard-dir", directory]
    argv += ["--expected-fixtures", str(fixture_dirs["serial"]),
             "--odds", odds,
             # the case set IS the registered set for 1c, so the merged
             # records get their block ids re-derived over it
             "--cluster-source-dir", str(fixture_dirs["serial"]),
             "--out-dir", str(merged_dir)]
    code = merge_shards.main(argv)
    rows = [{"arm": arm, "assertion": "merge_shards (10.5)",
             "status": "PASS" if code == 0 else "FAIL",
             "detail": f"merge_shards exit {code}"}]
    if code != 0:
        return rows

    # End to end: after the merge repairs the one shard-local field, the
    # merged artefacts must be the serial run's, record for record.
    serial = load_run(out_root / "serial" / arm, f"serial/{arm}")
    merged = load_run(merged_dir, f"merged/{arm}")
    problems: List[str] = []
    if set(serial.eval_records) != set(merged.eval_records):
        problems.append("evaluator record ids differ")
    if set(serial.raw_rows) != set(merged.raw_rows):
        problems.append("raw simulation row ids differ")
    for fixture in sorted(set(serial.eval_records) & set(merged.eval_records)):
        found = first_difference(serial.eval_records[fixture],
                                 merged.eval_records[fixture])
        if found is not None:
            problems.append(f"eval_record {fixture}: {found}")
    for fixture in sorted(set(serial.raw_rows) & set(merged.raw_rows)):
        found = first_difference(serial.raw_rows[fixture],
                                 merged.raw_rows[fixture])
        if found is not None:
            problems.append(f"raw_sims_row {fixture}: {found}")
    rows.append({
        "arm": arm,
        "assertion": "merged equals serial (after the cluster re-stamp)",
        "status": "PASS" if not problems else "FAIL",
        "detail": ("every merged record is the serial run's"
                   if not problems else "; ".join(problems[:5]))})
    return rows


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    import yaml

    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    try:
        config = yaml.safe_load(config_path.read_text())
    except OSError as error:
        print(f"shard_consistency_1c: REFUSED: {error}")
        return EXIT_REFUSED

    try:
        problems = check_case_set(CASE_SET, args.shards)
        if problems:
            raise ConsistencyError("; ".join(problems))
        fixture_dirs = {
            "serial": out_root / "serial" / "fixtures",
        }
        for index in range(args.shards):
            fixture_dirs[f"shard{index}"] = (
                out_root / "shards" / str(index) / "fixtures")
        if args.stage in ("build", "all"):
            fixture_dirs = build_fixture_dirs(out_root, CASE_SET, args.shards)
            for name, path in sorted(fixture_dirs.items()):
                print(f"[1c] fixtures {name}: {path} "
                      f"({len(list(path.glob('*.json')))} files)")
        specs = plan_runs(config, out_root, fixture_dirs, arms=args.arms,
                          n_sims=args.n_sims, n_shards=args.shards,
                          base_seed=args.base_seed)
    except ConsistencyError as error:
        print(f"shard_consistency_1c: REFUSED: {error}")
        return EXIT_REFUSED

    if args.dry_run:
        for spec in specs:
            print(f"# {spec.label}")
            print(spec.command)
        return EXIT_OK

    runs: List[RunResult] = []
    record_path = out_root / RUNS_RECORD
    if args.stage in ("run", "all"):
        runs = launch(specs, max_concurrent=args.max_concurrent,
                      force=args.force, record_path=record_path)
        failed = [run for run in runs if run.exit_code != 0]
        if failed:
            print("shard_consistency_1c: run(s) failed: "
                  + ", ".join(f"{run.label} (exit {run.exit_code})"
                              for run in failed))
            return EXIT_FAILED
    if args.stage in ("build", "run"):
        return EXIT_OK
    if not runs:
        previous = load_runs_record(record_path)
        runs = [RunResult(
            label=spec.label, arm=spec.arm, shard=spec.shard,
            output_dir=str(spec.output_dir),
            exit_code=int(previous.get(spec.label, {}).get("exit_code", 0)),
            wall_seconds=float(
                previous.get(spec.label, {}).get("wall_seconds", 0.0)),
            skipped=True)
            for spec in specs]

    assignment = shard_assignment(CASE_SET, args.shards)
    results: List[ArtefactResult] = []
    identity_rows: List[dict] = []
    decomposition: List[str] = []
    merge_rows: List[dict] = []
    try:
        for arm in args.arms:
            serial = load_run(out_root / "serial" / arm, f"serial/{arm}")
            decomposition += check_matches_advanced_decomposition(serial)
            for index in range(args.shards):
                shard = load_run(out_root / "shards" / str(index) / arm,
                                 f"shard{index}/{arm}")
                decomposition += check_matches_advanced_decomposition(shard)
                ok, detail = compare_run_identity(serial, shard)
                identity_rows.append({
                    "arm": arm, "shard": index,
                    "status": "PASS" if ok else "FAIL",
                    "detail": detail or "identical"})
                for case in ordered_cases(CASE_SET):
                    if assignment[case.cricsheet_id] != index:
                        continue
                    results += compare_fixture(
                        serial, shard, case.cricsheet_id, shard_index=index)
            merge_rows += _run_merge(out_root, arm, args.shards,
                                     args.odds, fixture_dirs)
    except ConsistencyError as error:
        print(f"shard_consistency_1c: REFUSED: {error}")
        return EXIT_REFUSED

    payload = build_report(cases=CASE_SET, n_shards=args.shards,
                           n_sims=args.n_sims, base_seed=args.base_seed,
                           specs=specs, runs=runs, results=results,
                           identity_rows=identity_rows,
                           decomposition_problems=decomposition,
                           merge_rows=merge_rows)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "consistency_1c.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    (out_root / "consistency_1c.md").write_text(render_markdown(payload))
    print(f"[1c] report: {out_root / 'consistency_1c.md'}")
    summary = payload["summary"]
    print(f"[1c] {summary['verdict']}: {summary['n_checks']} checks, "
          f"{summary['n_fail']} failing")
    for row in payload["results"]:
        if row["status"] == "FAIL":
            print(f"     FAIL {row['arm']} {row['fixture']} "
                  f"{row['artefact']}: {row['detail']}")
    return EXIT_OK if summary["n_fail"] == 0 else EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
