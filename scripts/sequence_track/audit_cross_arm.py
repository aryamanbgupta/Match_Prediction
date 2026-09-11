#!/usr/bin/env python3
"""Cross-arm audit for sequence-track stage 1 (D6 checks 6.2-6.5).

Usage:
    uv run --no-sync python scripts/sequence_track/audit_cross_arm.py \
        <arm_dir> <arm_dir> [...] [--expected-fixtures <dir>]
        [--allow-unregistered]

Each directory must hold the `arm_provenance.json` written by `run_arm.py`.
The audit asserts that the arms saw the SAME evaluation problem — same
fixtures, same prediction-time cutoffs, same eligibility, same odds rows,
same selector, sidecars, seeds, engine, runner and registered config — so
that a difference in their results can only come from the ball model.

It fails closed rather than passing on absent evidence:

* every provenance file must carry every field of the mandatory schema
  below, with the right type. A field missing from EVERY arm is a failure,
  not an agreement.
* the fixture set must be non-empty, and must equal the JSON stems of the
  fixture directory: `--expected-fixtures`, or (by default) the
  `run.fixture_dir` recorded in the provenance, which must itself agree
  across arms. For a registered run that directory, its md5 and its count
  are bound to the block the run claims, so a run cannot relabel itself as
  another block, and a block whose fixture set is not pinned yet (the 1b
  timing shard, before it is built) cannot be claimed at all.
* each arm is checked against ITSELF as well as against the others: every
  fixture's seed and sub-seeds are recomputed from the recorded base seed
  and cricsheet id, and every per-fixture n_sims must equal the run's.
* every arm must have run under a registered config, and that claim is
  RE-DERIVED here, not trusted: the recorded `config_path` must exist, hash
  to the recorded `config_sha256`, and pin exactly the artifacts, seeds,
  n_sims, threads, device and source hashes the run recorded. A set of arms
  that all carry the same wrong hashes therefore fails, where cross-arm
  equality alone would have passed it. `--allow-unregistered` is for an
  ad-hoc set in which NO arm claims a registered config.

Deliberately NOT asserted (expected to differ; listed in the table for the
record): model directory hash, checkpoint md5, device and threads. Stats
version and cache md5 are listed too, but since the 2026-09-11 retrain on the
i7 frame all four arms serve the SAME i7 cache, so a difference there is now
a defect rather than the registered asymmetry it used to be (the old
`stats_cache_i7_vs_v3` and `training_frame_i7_vs_v3` entries were moved to
`removed_asymmetries` in the stage-1 config).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# The seed rule is recomputed here with the RUNNER's own implementation, so
# the audit can never drift from what the runner produced.
from sequence_track.run_arm import fixture_seed, sub_seeds  # noqa: E402

PROVENANCE_FILENAME = "arm_provenance.json"
PROVENANCE_CONTRACT = "sequence_track_arm_provenance_v2"

_TEXT = (str,)
_MAYBE_TEXT = (str, type(None))
_NUMBER = (int, float)

# Mandatory per-run schema: field -> accepted types (None = the field may be
# null but must be present).
RUN_SCHEMA = {
    "arm": _TEXT,
    "model_dir": _TEXT,
    "model_dir_hash": _TEXT,
    "checkpoint_md5": _TEXT,
    "stats_version": _TEXT,
    "stats_cache_md5": _TEXT,
    "selector_class": _TEXT,
    "bowler_usage_md5": _TEXT,
    "bowler_roster_policy_md5": _MAYBE_TEXT,
    "extras_graft_sha256": _MAYBE_TEXT,
    "runout_p": _NUMBER,
    "clip": (dict,),
    "base_seed": (int,),
    "n_sims": (int,),
    "engine_md5": _TEXT,
    "runner_md5": _TEXT,
    "threads": (int,),
    "device": _TEXT,
    "config_path": _MAYBE_TEXT,
    "config_sha256": _MAYBE_TEXT,
    "config_verified": (bool,),
    "fixture_dir": _TEXT,
    "fixture_dir_hash": _TEXT,
    "fixture_count": (int,),
    "player_metadata_sha256": _TEXT,
    "context_dir": _TEXT,
    "context_dir_hash": _TEXT,
    "odds": _TEXT,
    "odds_sha256": _TEXT,
}

FIXTURE_SCHEMA = {
    "cricsheet_id": _TEXT,
    "match_date": _TEXT,
    "as_of": (dict,),
    "eligibility": (dict,),
    "fixture_seed": (int,),
    "sub_seeds": (dict,),
    "n_sims": (int,),
}

AS_OF_SCHEMA = {
    "date": _TEXT,
    "same_day_advanced_before": (list,),
    "matches_advanced": (int,),
}

# Every eligibility fact is mandatory and every one is asserted identical
# across arms: the odds-row half is what the evaluator scores against, the
# cricsheet half is the fixture's own outcome, and `scored` is whether the
# fixture produced a result at all.
ELIGIBILITY_SCHEMA = {
    "odds_row_found": (bool,),
    "odds_row_id": (dict, type(None)),
    "odds_row_sha256": _MAYBE_TEXT,
    "odds_actual_winner": _MAYBE_TEXT,
    "cricsheet_resolved": (bool,),
    "cricsheet_winner": _MAYBE_TEXT,
    "male_t20": (bool,),
    "scored": (bool,),
    "skip_reason": _MAYBE_TEXT,
}

CLIP_SCHEMA = {
    "low": _NUMBER,
    "high": _NUMBER,
    "seam_enabled": (bool,),
}

# 6.5 — run-level fields that must be identical across every arm.
RUN_IDENTITY_FIELDS = (
    "selector_class",
    "bowler_usage_md5",
    "bowler_roster_policy_md5",
    "extras_graft_sha256",
    "runout_p",
    "clip",
    "n_sims",
    # Beyond the acceptance text, all required by Astra round 1: paired
    # contrasts are only paired if the arms share the base seed, the odds
    # file, the same-day context corpus, the engine, the runner and the
    # registered config that pins all of them.
    "base_seed",
    "config_sha256",
    "odds_sha256",
    "context_dir_hash",
    "engine_md5",
    "runner_md5",
)

# Listed in the table, never asserted.
RUN_LISTED_FIELDS = (
    "model_dir",
    "model_dir_hash",
    "checkpoint_md5",
    "stats_version",
    "stats_cache_md5",
    "config_verified",
    "device",
    "threads",
)

ELIGIBILITY_FLAGS = tuple(ELIGIBILITY_SCHEMA)


def _type_name(types) -> str:
    return "/".join(
        "null" if candidate is type(None) else candidate.__name__
        for candidate in types)


def _check_schema(payload, schema, label, failures) -> None:
    """Every schema field must be present with an accepted type."""
    if not isinstance(payload, dict):
        failures.append(f"[schema] {label}: expected a mapping, "
                        f"found {type(payload).__name__}")
        return
    for field, types in schema.items():
        if field not in payload:
            failures.append(f"[schema] {label}: missing field {field!r}")
            continue
        value = payload[field]
        # bool is an int subclass; an int field must not silently accept True.
        if types == (int,) and isinstance(value, bool):
            failures.append(
                f"[schema] {label}: {field} is a bool, expected int")
            continue
        if not isinstance(value, types):
            failures.append(
                f"[schema] {label}: {field} is "
                f"{type(value).__name__}, expected {_type_name(types)}")


class ArmRecord:
    """One arm's provenance, indexed by fixture."""

    def __init__(self, path: Path, payload: dict):
        self.path = path
        self.payload = payload
        self.arm = str(payload.get("arm", path.parent.name))
        # Overwritten by `_label_arms` when two given dirs carry the same arm
        # id (comparing two runs of one arm), so every message stays
        # unambiguous about which directory it is talking about.
        self.label = self.arm
        self.run = payload.get("run", {}) or {}
        rows = payload.get("fixtures", []) or []
        self.fixtures = {}
        for row in rows:
            if not isinstance(row, dict) or "cricsheet_id" not in row:
                raise SystemExit(
                    f"{path}: a fixture row has no cricsheet_id")
            fixture_id = str(row["cricsheet_id"])
            if fixture_id in self.fixtures:
                raise SystemExit(
                    f"{path}: duplicate fixture {fixture_id} in arm "
                    f"{self.arm}")
            self.fixtures[fixture_id] = row

    @property
    def ids(self) -> set:
        return set(self.fixtures)


def load_arm(arm_dir) -> ArmRecord:
    directory = Path(arm_dir)
    path = directory / PROVENANCE_FILENAME
    if directory.is_file() and directory.name == PROVENANCE_FILENAME:
        path = directory
    if not path.exists():
        raise SystemExit(f"missing {PROVENANCE_FILENAME}: {path}")
    payload = json.loads(path.read_text())
    contract = payload.get("contract")
    if contract != PROVENANCE_CONTRACT:
        raise SystemExit(
            f"{path}: contract {contract!r} != {PROVENANCE_CONTRACT!r}")
    return ArmRecord(path, payload)


def _label_arms(arms) -> None:
    """Disambiguate arms that share an id by appending their directory."""
    seen = {}
    for arm in arms:
        seen.setdefault(arm.arm, []).append(arm)
    for arm_id, group in seen.items():
        for arm in group:
            arm.label = (
                arm_id if len(group) == 1
                else f"{arm_id}@{arm.path.parent.name}")


def _fixture_dir_stems(fixture_dir) -> set:
    directory = Path(fixture_dir)
    if not directory.is_dir():
        raise SystemExit(f"fixture directory not found: {directory}")
    stems = {path.stem for path in directory.glob("*.json")}
    if not stems:
        raise SystemExit(f"no fixture JSON files in {directory}")
    return stems


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_path(left, right) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return (Path(left).expanduser().resolve()
            == Path(right).expanduser().resolve())


_CONFIG_CACHE = {}


def _load_config(path):
    key = str(Path(path).resolve())
    if key not in _CONFIG_CACHE:
        _CONFIG_CACHE[key] = yaml.safe_load(Path(path).read_text())
    return _CONFIG_CACHE[key]


def registration_failures(arm) -> list:
    """Check one arm's provenance against the config it claims to have used.

    `config_verified: true` is the RUNNER's word for it; this re-derives it
    from the file. The config must exist, hash to the recorded sha256, and
    pin exactly what the run recorded — otherwise a set of arms that all
    carry the same wrong hashes would agree with each other and pass
    (Astra round 2, item 4b).
    """
    run = arm.run
    label = f"arm {arm.label}"
    config_path = run.get("config_path")
    config_sha = run.get("config_sha256")
    if not config_path or not config_sha:
        return [f"[config] {label}: config_path/config_sha256 is null; the "
                "run recorded no registered config to check against"]
    path = Path(config_path)
    if not path.exists():
        return [f"[config] {label}: recorded config {config_path} does not "
                "exist, so its pins cannot be checked"]
    recomputed = _sha256(path)
    if recomputed != config_sha:
        return [f"[config] {label}: {config_path} hashes to {recomputed}, "
                f"the run recorded {config_sha}; the registered config "
                "changed after the run"]

    config = _load_config(path)
    arm_id = str(run.get("arm"))
    block = (config.get("arms") or {}).get(arm_id)
    if not isinstance(block, dict):
        return [f"[config] {label}: {config_path} has no arm block for "
                f"{arm_id!r}"]

    failures = []

    def _check(field, pinned, found, *, path_compare=False):
        same = (_same_path(pinned, found) if path_compare
                else pinned == found)
        if not same:
            failures.append(
                f"[config] {label}: {field} = {found!r}, "
                f"{config_path} pins {pinned!r}")

    _check("model_dir", block.get("model_dir"), run.get("model_dir"),
           path_compare=True)
    _check("checkpoint_md5", block.get("checkpoint_md5"),
           run.get("checkpoint_md5"))
    _check("stats_version", block.get("stats_version"),
           run.get("stats_version"))
    _check("stats_cache_md5", block.get("stats_cache_md5"),
           run.get("stats_cache_md5"))
    _check("extras_graft_sha256", block.get("extras_graft_sha256"),
           run.get("extras_graft_sha256"))
    _check("bowler_usage_md5", block.get("bowler_usage_md5"),
           run.get("bowler_usage_md5"))
    _check("bowler_roster_policy_md5", block.get("roster_policy_md5"),
           run.get("bowler_roster_policy_md5"))
    _check("odds_sha256", (block.get("odds") or {}).get("sha256"),
           run.get("odds_sha256"))
    _check("context_dir_hash", block.get("context_dir_md5"),
           run.get("context_dir_hash"))
    _check("threads", block.get("threads"), run.get("threads"))
    _check("device", block.get("device"), run.get("device"))
    _check("selector_class", block.get("bowler_selector"),
           run.get("selector_class"))
    # Astra round 3, item 3: both of these were recorded but never checked,
    # so changing them uniformly across arms passed.
    _check("player_metadata_sha256", block.get("player_metadata_sha256"),
           run.get("player_metadata_sha256"))
    _check("runout_p", (block.get("run_out") or {}).get("value"),
           run.get("runout_p"))

    clip = run.get("clip") or {}
    pinned_clip = [float(value) for value in (block.get("clip") or [])]
    found_clip = [clip.get("low"), clip.get("high")]
    if pinned_clip != [
            None if value is None else float(value) for value in found_clip]:
        failures.append(
            f"[config] {label}: clip = {found_clip!r}, {config_path} pins "
            f"{pinned_clip!r}")
    elif not clip.get("seam_enabled"):
        failures.append(
            f"[config] {label}: the clip seam is disabled, but "
            f"{config_path} pins bounds {pinned_clip!r}")

    sources = ((config.get("provenance") or {}).get("source_md5") or {})
    _check("engine_md5", sources.get("scripts/sim_v1_2.py"),
           run.get("engine_md5"))
    _check("runner_md5", sources.get("scripts/sequence_track/run_arm.py"),
           run.get("runner_md5"))

    block_name = run.get("config_block")
    run_block = block.get(block_name) if block_name else None
    if not isinstance(run_block, dict):
        failures.append(
            f"[config] {label}: config_block = {block_name!r} is not a "
            f"registered run block of arm {arm_id}")
        return failures

    seeds = [int(value) for value in (run_block.get("base_seeds") or [])]
    if not seeds:
        failures.append(
            f"[config] {label}: the {block_name} block registers no base "
            "seed")
    elif int(run.get("base_seed", -1)) not in seeds:
        failures.append(
            f"[config] {label}: base_seed = {run.get('base_seed')!r}, the "
            f"{block_name} block permits {seeds!r}")

    # The block's OWN fixture directory, inventory hash and count — without
    # this binding a run could claim any block it liked (Astra round 3,
    # item 1: the smoke records relabelled as timing_1b passed).
    pinned_dir = run_block.get("fixture_dir")
    if not _same_path(pinned_dir, run.get("fixture_dir")):
        failures.append(
            f"[config] {label}: fixture_dir = {run.get('fixture_dir')!r}, "
            f"the {block_name} block registers {pinned_dir!r}")
    pinned_hash = run_block.get("fixture_dir_md5")
    if pinned_hash is None:
        failures.append(
            f"[config] {label}: the {block_name} block pins no fixture "
            f"directory hash (status: "
            f"{run_block.get('fixture_dir_status', 'unpinned')!r}); a run "
            "cannot claim a block whose fixture set is not registered yet")
    elif run.get("fixture_dir_hash") != pinned_hash:
        failures.append(
            f"[config] {label}: fixture_dir_hash = "
            f"{run.get('fixture_dir_hash')!r}, the {block_name} block pins "
            f"{pinned_hash!r}")
    pinned_count = run_block.get("fixture_count")
    if pinned_count is not None:
        if run.get("fixture_count") != pinned_count:
            failures.append(
                f"[config] {label}: fixture_count = "
                f"{run.get('fixture_count')!r}, the {block_name} block "
                f"registers {pinned_count!r}")
        if len(arm.fixtures) != pinned_count:
            failures.append(
                f"[config] {label}: {len(arm.fixtures)} fixture rows, the "
                f"{block_name} block registers {pinned_count!r}")

    permitted = run_block.get("n_sims")
    permitted = (permitted if isinstance(permitted, list) else [permitted])
    if not all(isinstance(value, int) for value in permitted):
        failures.append(
            f"[config] {label}: the {block_name} block's n_sims is "
            f"{run_block.get('n_sims')!r}, not a launchable value")
    else:
        if int(run.get("n_sims", -1)) not in permitted:
            failures.append(
                f"[config] {label}: n_sims = {run.get('n_sims')!r}, the "
                f"{block_name} block permits {permitted!r}")
        # Every per-fixture count too: the run-level field alone left the
        # per-fixture ones unchecked.
        for fixture_id, row in sorted(arm.fixtures.items()):
            if row.get("n_sims") not in permitted:
                failures.append(
                    f"[config] {label}, fixture {fixture_id}: n_sims = "
                    f"{row.get('n_sims')!r}, the {block_name} block permits "
                    f"{permitted!r}")
    return failures


def internal_failures(arm) -> list:
    """Checks inside ONE arm's provenance, independent of the other arms.

    Cross-arm equality cannot see a record that is wrong the same way
    everywhere: every fixture count set to 999, or every seed set to 42,
    used to pass (Astra round 3, item 3). The seeds are a pure function of
    (cricsheet id, base seed), so they are recomputed here with the runner's
    own functions.
    """
    failures = []
    run = arm.run
    label = f"arm {arm.label}"
    base_seed = run.get("base_seed")
    run_n_sims = run.get("n_sims")

    if run.get("fixture_count") != len(arm.fixtures):
        failures.append(
            f"[internal] {label}: run.fixture_count = "
            f"{run.get('fixture_count')!r} but {len(arm.fixtures)} fixture "
            "rows are recorded")

    for fixture_id, row in sorted(arm.fixtures.items()):
        if row.get("n_sims") != run_n_sims:
            failures.append(
                f"[internal] {label}, fixture {fixture_id}: n_sims = "
                f"{row.get('n_sims')!r}, the run recorded {run_n_sims!r}")
        if not isinstance(base_seed, int):
            continue
        expected_seed = fixture_seed(fixture_id, base_seed)
        if row.get("fixture_seed") != expected_seed:
            failures.append(
                f"[internal] {label}, fixture {fixture_id}: fixture_seed = "
                f"{row.get('fixture_seed')!r}, but sha256 of "
                f"'{fixture_id}:{base_seed}' gives {expected_seed}")
            continue
        expected_subs = sub_seeds(expected_seed)
        if row.get("sub_seeds") != expected_subs:
            failures.append(
                f"[internal] {label}, fixture {fixture_id}: sub_seeds = "
                f"{row.get('sub_seeds')!r}, the rule gives {expected_subs!r}")
    return failures


def audit(arms, expected_fixtures=None, allow_unregistered: bool = False
          ) -> list:
    """Return the list of failure messages (empty = pass)."""
    if len(arms) < 2:
        raise SystemExit("give at least two arm directories to compare")
    _label_arms(arms)
    failures = []
    reference = arms[0]

    # Schema first: a field missing from every arm must fail, not "agree".
    for arm in arms:
        _check_schema(arm.run, RUN_SCHEMA, f"arm {arm.label} run", failures)
        _check_schema(arm.run.get("clip"), CLIP_SCHEMA,
                      f"arm {arm.label} run.clip", failures)
        if not arm.fixtures:
            failures.append(
                f"[6.2 fixture set] arm {arm.label}: no fixtures recorded; "
                "an empty provenance proves nothing")
        for fixture_id, row in sorted(arm.fixtures.items()):
            label = f"arm {arm.label}, fixture {fixture_id}"
            _check_schema(row, FIXTURE_SCHEMA, label, failures)
            _check_schema(row.get("as_of"), AS_OF_SCHEMA,
                          f"{label} as_of", failures)
            _check_schema(row.get("eligibility"), ELIGIBILITY_SCHEMA,
                          f"{label} eligibility", failures)

    # Registration. `config_verified` is the runner's claim; unless the
    # caller has asked for an ad-hoc set, every arm's provenance is
    # re-derived from the registered config itself.
    for arm in arms:
        failures.extend(internal_failures(arm))

    registered = [arm for arm in arms if arm.run.get("config_verified")]
    if allow_unregistered:
        if registered:
            failures.append(
                "[config] --allow-unregistered is for a set where NO arm "
                "ran under a registered config; these did: "
                + ", ".join(arm.label for arm in registered))
        else:
            print("note: no arm ran under a registered config "
                  "(--allow-unregistered); this set is an ad-hoc check, not "
                  "stage-1 evidence")
    else:
        for arm in arms:
            if not arm.run.get("config_verified"):
                failures.append(
                    f"[config] arm {arm.label}: config_verified is "
                    f"{arm.run.get('config_verified')!r}; the run did not "
                    "re-verify experiments/configs/seq_stage1_sim_v1.yaml. "
                    "Re-run with --config, or pass --allow-unregistered for "
                    "an ad-hoc check.")
                continue
            failures.extend(registration_failures(arm))

    # 6.2 fixture sets identical, and equal to the fixture directory.
    for arm in arms[1:]:
        missing = sorted(reference.ids - arm.ids)
        extra = sorted(arm.ids - reference.ids)
        for fixture_id in missing:
            failures.append(
                f"[6.2 fixture set] arm {arm.label}: fixture {fixture_id} "
                f"present in arm {reference.label}, absent here")
        for fixture_id in extra:
            failures.append(
                f"[6.2 fixture set] arm {arm.label}: fixture {fixture_id} "
                f"absent from arm {reference.label}, present here")

    fixture_dirs = {arm.label: arm.run.get("fixture_dir") for arm in arms}
    resolved_dir = expected_fixtures
    if resolved_dir is None:
        distinct = {value for value in fixture_dirs.values()}
        if len(distinct) != 1 or None in distinct:
            failures.append(
                "[6.2 fixture set] arms record different fixture dirs "
                f"{fixture_dirs!r}; pass --expected-fixtures to say which "
                "set the run was supposed to cover")
        else:
            resolved_dir = distinct.pop()
    if resolved_dir is not None:
        expected_ids = _fixture_dir_stems(resolved_dir)
        for arm in arms:
            for fixture_id in sorted(expected_ids - arm.ids):
                failures.append(
                    f"[6.2 fixture set] arm {arm.label}: fixture "
                    f"{fixture_id} is in {resolved_dir} but not in this "
                    "arm's provenance")
            for fixture_id in sorted(arm.ids - expected_ids):
                failures.append(
                    f"[6.2 fixture set] arm {arm.label}: fixture "
                    f"{fixture_id} is in this arm's provenance but not in "
                    f"{resolved_dir}")

    shared = sorted(set.intersection(*[arm.ids for arm in arms]))

    for fixture_id in shared:
        base = reference.fixtures[fixture_id]
        for arm in arms[1:]:
            row = arm.fixtures[fixture_id]

            # 6.3 as-of stamps identical.
            for key in ("date", "same_day_advanced_before",
                        "matches_advanced"):
                expected = (base.get("as_of") or {}).get(key)
                found = (row.get("as_of") or {}).get(key)
                if expected != found:
                    failures.append(
                        f"[6.3 as-of] arm {arm.label}, fixture {fixture_id}: "
                        f"as_of.{key} = {found!r}, arm {reference.label} has "
                        f"{expected!r}")

            # 6.4 eligibility identical — including which odds row was
            # resolved and whether the fixture was scored.
            for flag in ELIGIBILITY_FLAGS:
                expected = (base.get("eligibility") or {}).get(flag)
                found = (row.get("eligibility") or {}).get(flag)
                if expected != found:
                    failures.append(
                        f"[6.4 eligibility] arm {arm.label}, fixture "
                        f"{fixture_id}: {flag} = {found!r}, arm "
                        f"{reference.label} has {expected!r}")

            # Per-fixture halves of 6.5 (seeds, n_sims).
            if base.get("fixture_seed") != row.get("fixture_seed"):
                failures.append(
                    f"[6.5 seeds] arm {arm.label}, fixture {fixture_id}: "
                    f"fixture_seed = {row.get('fixture_seed')!r}, arm "
                    f"{reference.label} has {base.get('fixture_seed')!r}")
            if base.get("sub_seeds") != row.get("sub_seeds"):
                failures.append(
                    f"[6.5 seeds] arm {arm.label}, fixture {fixture_id}: "
                    f"sub_seeds differ from arm {reference.label}")
            if base.get("n_sims") != row.get("n_sims"):
                failures.append(
                    f"[6.5 n_sims] arm {arm.label}, fixture {fixture_id}: "
                    f"n_sims = {row.get('n_sims')!r}, arm {reference.label} "
                    f"has {base.get('n_sims')!r}")

    # 6.5 run-level selector / sidecars / clip / seeds / pins.
    for field in RUN_IDENTITY_FIELDS:
        expected = reference.run.get(field)
        for arm in arms[1:]:
            found = arm.run.get(field)
            if expected != found:
                failures.append(
                    f"[6.5 run] arm {arm.label}: {field} = {found!r}, arm "
                    f"{reference.label} has {expected!r}")

    return failures


def _table(arms) -> str:
    _label_arms(arms)
    rows = [("field", *[arm.label for arm in arms])]
    rows.append(("fixtures", *[str(len(arm.ids)) for arm in arms]))
    rows.append(("scored", *[
        str(sum(1 for row in arm.fixtures.values()
                if (row.get("eligibility") or {}).get("scored")))
        for arm in arms]))
    for field in RUN_IDENTITY_FIELDS + RUN_LISTED_FIELDS:
        values = []
        for arm in arms:
            value = arm.run.get(field)
            if isinstance(value, dict):
                value = json.dumps(value, sort_keys=True)
            text = "-" if value is None else str(value)
            values.append(text if len(text) <= 34 else text[:31] + "...")
        marker = "=" if field in RUN_IDENTITY_FIELDS else " "
        rows.append((f"{marker} {field}", *values))
    widths = [max(len(str(row[i])) for row in rows)
              for i in range(len(rows[0]))]
    lines = []
    for index, row in enumerate(rows):
        lines.append("  ".join(
            str(cell).ljust(widths[i]) for i, cell in enumerate(row)).rstrip())
        if index == 0:
            lines.append("  ".join("-" * width for width in widths))
    lines.append("")
    lines.append("'=' marks fields asserted identical across arms (6.5); the "
                 "rest are listed only.")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit that stage-1 arms saw the same evaluation problem")
    parser.add_argument("arm_dirs", nargs="+")
    parser.add_argument(
        "--expected-fixtures", default=None,
        help="fixture directory the arms were supposed to cover; default is "
             "the run.fixture_dir recorded in the provenance, which must "
             "agree across arms")
    parser.add_argument(
        "--allow-unregistered", action="store_true",
        help="accept a set in which NO arm ran under a registered config "
             "(ad-hoc checks only; a mixed set is refused)")
    args = parser.parse_args(argv)

    arms = [load_arm(arm_dir) for arm_dir in args.arm_dirs]
    print(_table(arms))
    failures = audit(arms, expected_fixtures=args.expected_fixtures,
                     allow_unregistered=args.allow_unregistered)
    print()
    if failures:
        print(f"CROSS-ARM AUDIT FAILED ({len(failures)} mismatch"
              f"{'es' if len(failures) != 1 else ''}):")
        for failure in failures:
            print(f"  {failure}")
        return 1
    fixtures = len(arms[0].ids)
    print(f"CROSS-ARM AUDIT PASSED: {len(arms)} arms "
          f"({', '.join(arm.label for arm in arms)}), {fixtures} fixtures, "
          "schema complete, checks 6.2-6.5 identical.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
