#!/usr/bin/env python3
"""Merge one arm's shard outputs and assert the union is the registered set.

Stage 1 step 1c (`docs/sequence_track/stage1_acceptance.md`, D10 check 10.5)
and step 1d (D11 checks 11.4-11.5). A sharded run is only usable if the
shards partition the registered fixture set exactly: every expected fixture
present, none twice, every odds record claimed by exactly one fixture, and
the coverage counts (scored / unscored / skipped) summing to the totals.

This script asserts all four and then writes the merged artefacts:

* `eval.json` — the shards' per-fixture `matches` records concatenated in
  registered chronological `(match_date, cricsheet_id)` order, with a
  summary that is **counts only**. Every metric field the evaluator's own
  summary carries (log loss, Brier, ROI, edge, P&L, Kelly, Sharpe, win
  rate, ECE, timings) is OMITTED and named in `omitted_summary_fields`,
  because a pooled metric must be recomputed by the scorer over the merged
  `matches`, never averaged across shards here. `reslice_eval_json.py`
  reads `matches` only, so the merged file is a complete input for it.
* `raw_sims.jsonl` — the shards' per-fixture raw simulation rows in the
  same order.
* `arm_provenance.json` — every shard's per-fixture provenance row with the
  shard of origin added, plus the run-level block each shard agreed on and
  the fields that necessarily differ per shard listed separately.

NOTHING here reads or prints a log loss, Brier, ROI or edge: the merge
prints counts, and the metric-value guard below refuses to write an output
that carries a number under any metric-named key.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

CONTRACT = "seq_stage1_merged_shards_v1"

EVAL_FILENAME = "eval.json"
RAW_FILENAME = "raw_sims.jsonl"
PROVENANCE_FILENAME = "arm_provenance.json"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REFUSED = 2

# ---------------------------------------------------------------------------
# What survives into the merged summary.
# ---------------------------------------------------------------------------

# Non-metric identity/configuration fields. Every shard must agree on them,
# or the shards did not run one experiment and the merge fails closed.
SUMMARY_IDENTITY_KEYS = (
    "model_type",
    "slice",
    "min_volume",
    "cost_model",
    "price_basis",
    "volume_basis",
    "bootstrap_contract",
    "bootstrap_seed",
    "bootstrap_resamples",
    "calibration_method",
    "ball_calibration_enabled",
    "ball_calibrator_path",
)

# Run-level provenance fields that MUST be identical across shards: they
# identify the model, the data, the BLOCK STRUCTURE and the draw schedule. A
# difference means the shards are not one run.
#
# `cluster_source_dir` / `cluster_source_dir_hash` are here because the
# cluster source is what turns records into I3 competition blocks (critical
# invariant 7). Two shards run against different cluster sources — or against
# the same path whose contents changed mid-run — would be merged into one
# `matches` list whose block ids came from two different registered sets, and
# every downstream interval would be computed over blocks that never existed
# together. The hash is checked as well as the path for exactly that reason:
# the path agreeing is not the same fact as the fixture set agreeing.
RUN_IDENTITY_KEYS = (
    "arm",
    "model_dir",
    "model_dir_hash",
    "checkpoint_path",
    "checkpoint_md5",
    "stats_version",
    "stats_cache_path",
    "stats_cache_md5",
    "selector_class",
    "bowler_usage_path",
    "bowler_usage_md5",
    "bowler_roster_policy_path",
    "bowler_roster_policy_md5",
    "extras_graft_path",
    "extras_graft_sha256",
    "extras_graft_applied_via",
    "runout_p",
    "clip",
    "base_seed",
    "n_sims",
    "engine_md5",
    "runner_md5",
    "threads",
    "device",
    "context_dir",
    "context_dir_hash",
    "cluster_source_dir",
    "cluster_source_dir_hash",
    "odds",
    "odds_sha256",
    "player_metadata_sha256",
    "rng_streams",
    "config_path",
    "config_sha256",
    "config_verified",
)

# Run-level provenance fields that differ per shard BY DESIGN.
RUN_PER_SHARD_KEYS = (
    "fixture_dir",
    "fixture_dir_hash",
    "fixture_dir_hash_contract",
    "fixture_count",
    "min_seed_gap",
    "config_block",
    "started_at",
    "finished_at",
    "elapsed_seconds",
)

# Any key matching this pattern may not carry a number in a merged output.
METRIC_KEY_PATTERN = re.compile(
    r"log_loss|logloss|brier|roi|_edge|^edge|pnl|kelly|sharpe|win_rate|"
    r"expected_value|ece|profit|payout|_prob|odds$",
    re.IGNORECASE,
)

# Numbers this script is allowed to emit, by key name. Everything else must
# be a count under one of these names or the guard refuses the write.
ALLOWED_NUMERIC_KEYS = frozenset({
    # counts written by the merge
    "n_shards", "n_fixtures", "n_expected", "n_scored", "n_unscored",
    "n_skipped", "n_raw_rows", "n_matches", "n_matches_evaluated",
    "n_odds_rows_in_file", "n_odds_rows_expected", "n_odds_rows_claimed",
    "shard", "index", "fixture_count", "matches_advanced", "n_sims",
    "base_seed", "same_day_advanced_count",
    # configuration numbers carried through from the evaluator's summary:
    # a cost model and a bootstrap setting are settings, not results
    "spread_bps", "fee_bps", "bootstrap_seed", "bootstrap_resamples",
    "min_volume", "n_changed",
})


class MergeError(RuntimeError):
    """A merge assertion failed."""


# ---------------------------------------------------------------------------
# The metric-value guard
# ---------------------------------------------------------------------------

def assert_no_metric_values(payload, *, path: str = "",
                            allowed: Iterable[str] = ALLOWED_NUMERIC_KEYS
                            ) -> None:
    """Refuse a payload that carries a number under a metric-named key.

    The merge prints and writes counts. A pooled metric is the scorer's job
    and must be recomputed over the merged `matches`; a number that reached
    a merged summary would be an average of shard averages, which is both
    wrong and a reading of a result this step must not make.
    """
    allowed = set(allowed)
    if isinstance(payload, dict):
        for key in sorted(payload):
            child = f"{path}.{key}" if path else str(key)
            value = payload[key]
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and key not in allowed:
                raise MergeError(
                    f"merged output would carry a number at {child!r}; only "
                    "counts may be written by the merge")
            if METRIC_KEY_PATTERN.search(str(key)) and isinstance(
                    value, (int, float)) and not isinstance(value, bool):
                raise MergeError(
                    f"merged output would carry a metric value at {child!r}")
            assert_no_metric_values(value, path=child, allowed=allowed)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            assert_no_metric_values(value, path=f"{path}[{index}]",
                                    allowed=allowed)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@dataclass
class Shard:
    """One arm's output directory for one shard."""

    name: str
    path: Path
    arm: str
    run: dict
    fixtures: Dict[str, dict]          # cricsheet id -> provenance row
    eval_matches: Dict[str, dict]      # cricsheet id -> eval record
    match_identity: dict
    summary: dict
    raw_rows: Dict[str, dict]          # cricsheet id -> raw_sims row
    order: List[str] = field(default_factory=list)

    @property
    def scored_ids(self) -> List[str]:
        return [fid for fid, row in self.fixtures.items()
                if row["eligibility"]["scored"]]

    @property
    def unscored_ids(self) -> List[str]:
        """Stamped, not scored, because no odds row resolved."""
        return [fid for fid, row in self.fixtures.items()
                if not row["eligibility"]["scored"]
                and row["eligibility"].get("skip_reason") == "no_odds_row"]

    @property
    def skipped_ids(self) -> List[str]:
        """Stamped, not scored, for any other reason (evaluation error)."""
        return [fid for fid, row in self.fixtures.items()
                if not row["eligibility"]["scored"]
                and row["eligibility"].get("skip_reason") != "no_odds_row"]


def _eval_match_key(record: dict) -> str:
    """The cricsheet id an evaluator record belongs to."""
    for key in ("cricsheet_id", "match_id", "display_match_id"):
        value = record.get(key)
        if value:
            return str(value)
    raise MergeError("evaluator record carries no match identity")


def load_shard(directory, name: Optional[str] = None) -> Shard:
    path = Path(directory)
    provenance_path = path / PROVENANCE_FILENAME
    eval_path = path / EVAL_FILENAME
    raw_path = path / RAW_FILENAME
    for required in (provenance_path, eval_path):
        if not required.is_file():
            raise MergeError(f"{path}: missing {required.name}")

    provenance = json.loads(provenance_path.read_text())
    evaluation = json.loads(eval_path.read_text())

    fixtures: Dict[str, dict] = {}
    order: List[str] = []
    for row in provenance.get("fixtures", []):
        fid = str(row["cricsheet_id"])
        if fid in fixtures:
            raise MergeError(f"{path}: fixture {fid} appears twice in "
                             f"{PROVENANCE_FILENAME}")
        fixtures[fid] = row
        order.append(fid)

    eval_matches: Dict[str, dict] = {}
    for record in evaluation.get("matches", []):
        key = _eval_match_key(record)
        if key in eval_matches:
            raise MergeError(f"{path}: evaluator record {key} appears twice")
        eval_matches[key] = record

    raw_rows: Dict[str, dict] = {}
    if raw_path.is_file():
        with raw_path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = str(row["match_id"])
                if key in raw_rows:
                    raise MergeError(
                        f"{path}: {RAW_FILENAME} line {line_no}: fixture "
                        f"{key} appears twice")
                raw_rows[key] = row

    return Shard(
        name=str(name if name is not None else path.name),
        path=path,
        arm=str(provenance.get("arm")),
        run=provenance.get("run") or {},
        fixtures=fixtures,
        eval_matches=eval_matches,
        match_identity=evaluation.get("match_identity") or {},
        summary=evaluation.get("summary") or {},
        raw_rows=raw_rows,
        order=order,
    )


def shard_names(directories: Sequence) -> List[str]:
    """Short, UNIQUE labels for a set of shard dirs.

    Shard output dirs are `<root>/shards/<k>/<arm>`, so their basenames are
    all the arm name. The shortest path suffix that distinguishes them is
    used, and the full path is the last resort: two shards sharing a label
    would silently collapse the merge.
    """
    paths = [Path(directory) for directory in directories]
    for depth in (1, 2, 3):
        labels = ["/".join(path.parts[-depth:]) for path in paths]
        if len(set(labels)) == len(labels):
            return labels
    labels = [str(path) for path in paths]
    if len(set(labels)) != len(labels):
        raise MergeError("two shard directories are the same path")
    return labels


def read_expected_fixtures(source) -> List[str]:
    """Expected cricsheet ids from a fixture dir, a JSON list or a id list."""
    path = Path(source)
    if path.is_dir():
        ids = sorted(p.stem for p in path.glob("*.json"))
        if not ids:
            raise MergeError(f"{path}: no fixture JSON files")
        return ids
    if not path.is_file():
        raise MergeError(f"expected-fixture source not found: {path}")
    text = path.read_text().strip()
    if text.startswith("["):
        return [str(value) for value in json.loads(text)]
    return [line.strip() for line in text.splitlines() if line.strip()]


def read_odds_ids(odds_path) -> List[str]:
    """Primary ids of every row in an odds file, in file order."""
    payload = json.loads(Path(odds_path).read_text())
    ids = []
    for row in payload.get("matches", []):
        value = row.get("cricsheet_id") or row.get("match_id")
        if value is None:
            raise MergeError("odds row carries no cricsheet_id or match_id")
        ids.append(str(value))
    return ids


# ---------------------------------------------------------------------------
# Assertions (each has a negative test)
# ---------------------------------------------------------------------------

def assert_run_identity(shards: Sequence[Shard]) -> List[str]:
    """Same arm, model, cache, cluster source, seed and engine on every shard."""
    problems: List[str] = []
    if not shards:
        return ["no shard directories given"]
    first = shards[0]
    for shard in shards[1:]:
        if shard.arm != first.arm:
            problems.append(
                f"arm: shard {shard.name!r} is {shard.arm!r}, shard "
                f"{first.name!r} is {first.arm!r}")
        for key in RUN_IDENTITY_KEYS:
            if shard.run.get(key) != first.run.get(key):
                problems.append(
                    f"run.{key}: shard {shard.name!r} differs from shard "
                    f"{first.name!r}")
        if shard.match_identity != first.match_identity:
            problems.append(
                f"match_identity: shard {shard.name!r} differs from shard "
                f"{first.name!r}")
        for key in SUMMARY_IDENTITY_KEYS:
            if shard.summary.get(key) != first.summary.get(key):
                problems.append(
                    f"summary.{key}: shard {shard.name!r} differs from shard "
                    f"{first.name!r}")
    return problems


def assert_union(shards: Sequence[Shard], expected: Sequence[str]
                 ) -> List[str]:
    """The union of shard inventories must equal the expected set."""
    expected_set = {str(value) for value in expected}
    union = set()
    for shard in shards:
        union |= set(shard.fixtures)
    problems = []
    missing = sorted(expected_set - union)
    extra = sorted(union - expected_set)
    if missing:
        problems.append(
            f"{len(missing)} expected fixture(s) absent from every shard: "
            f"{missing[:10]}")
    if extra:
        problems.append(
            f"{len(extra)} fixture(s) in a shard but not expected: "
            f"{extra[:10]}")
    return problems


def assert_no_duplicates(shards: Sequence[Shard]) -> List[str]:
    """No fixture may appear in two shards."""
    seen: Dict[str, str] = {}
    problems = []
    for shard in shards:
        for fid in sorted(shard.fixtures):
            prior = seen.get(fid)
            if prior is not None:
                problems.append(
                    f"fixture {fid} appears in shard {prior!r} and shard "
                    f"{shard.name!r}")
            else:
                seen[fid] = shard.name
    return problems


def assert_odds_claims(shards: Sequence[Shard], expected: Sequence[str],
                       odds_ids: Optional[Sequence[str]]) -> List[str]:
    """Every odds record is claimed by exactly one scored fixture.

    `odds_ids` is every primary id in the odds file. The claim set is
    checked against the odds rows whose id is IN the expected fixture set —
    the rows this run was supposed to score. Rows outside it (the rest of
    the odds file, when the expected set is a case set rather than the
    registered 255) are reported as a count, not a failure.
    """
    problems: List[str] = []
    claimed: Dict[str, str] = {}
    for shard in shards:
        for fid, row in sorted(shard.fixtures.items()):
            eligibility = row["eligibility"]
            if not eligibility["scored"]:
                if eligibility.get("odds_row_found"):
                    problems.append(
                        f"fixture {fid} (shard {shard.name!r}) resolved an "
                        "odds row but was not scored")
                continue
            if not eligibility.get("odds_row_found"):
                problems.append(
                    f"fixture {fid} (shard {shard.name!r}) is scored with no "
                    "odds row")
                continue
            identity = eligibility.get("odds_row_id") or {}
            key = str(identity.get("cricsheet_id")
                      or identity.get("display_match_id") or fid)
            prior = claimed.get(key)
            if prior is not None:
                problems.append(
                    f"odds row {key} claimed by fixture {prior} and by "
                    f"fixture {fid} (shard {shard.name!r})")
            else:
                claimed[key] = fid

    if odds_ids is not None:
        expected_set = {str(value) for value in expected}
        expected_rows = {value for value in odds_ids
                         if str(value) in expected_set}
        unclaimed = sorted(expected_rows - set(claimed))
        foreign = sorted(set(claimed) - set(str(v) for v in odds_ids))
        if unclaimed:
            problems.append(
                f"{len(unclaimed)} odds row(s) for expected fixtures claimed "
                f"by no scored fixture: {unclaimed[:10]}")
        if foreign:
            problems.append(
                f"{len(foreign)} claimed odds row(s) are not in the odds "
                f"file: {foreign[:10]}")
    return problems


def assert_coverage_counts(shards: Sequence[Shard]) -> List[str]:
    """scored + unscored + skipped == fixtures, and the artefacts agree."""
    problems = []
    for shard in shards:
        total = len(shard.fixtures)
        scored = set(shard.scored_ids)
        unscored = set(shard.unscored_ids)
        skipped = set(shard.skipped_ids)
        if len(scored) + len(unscored) + len(skipped) != total:
            problems.append(
                f"shard {shard.name!r}: coverage counts "
                f"{len(scored)}+{len(unscored)}+{len(skipped)} do not sum to "
                f"{total} fixtures")
        if set(shard.eval_matches) != scored:
            only_eval = sorted(set(shard.eval_matches) - scored)
            only_prov = sorted(scored - set(shard.eval_matches))
            problems.append(
                f"shard {shard.name!r}: evaluator records and scored "
                f"provenance disagree (in eval only: {only_eval[:5]}; scored "
                f"only: {only_prov[:5]})")
        if set(shard.raw_rows) != scored:
            only_raw = sorted(set(shard.raw_rows) - scored)
            only_prov = sorted(scored - set(shard.raw_rows))
            problems.append(
                f"shard {shard.name!r}: raw simulation rows and scored "
                f"provenance disagree (raw only: {only_raw[:5]}; scored "
                f"only: {only_prov[:5]})")
        recorded = shard.run.get("fixture_count")
        if recorded is not None and int(recorded) != total:
            problems.append(
                f"shard {shard.name!r}: run.fixture_count {recorded} is not "
                f"the {total} fixtures in the provenance")
    return problems


def check_all(shards: Sequence[Shard], expected: Sequence[str],
              odds_ids: Optional[Sequence[str]]) -> Dict[str, List[str]]:
    """Every assertion, named, so a caller can report which one failed."""
    return {
        "run_identity": assert_run_identity(shards),
        "union_equals_expected": assert_union(shards, expected),
        "no_duplicate_fixture": assert_no_duplicates(shards),
        "odds_claimed_once": assert_odds_claims(shards, expected, odds_ids),
        "coverage_counts": assert_coverage_counts(shards),
    }


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def chronological_order(shards: Sequence[Shard]) -> List[Tuple[str, str]]:
    """`(cricsheet_id, shard name)` in registered `(date, id)` order."""
    rows = []
    for shard in shards:
        for fid, row in shard.fixtures.items():
            rows.append((str(row["match_date"]), str(fid), shard.name))
    rows.sort()
    return [(fid, shard_name) for _date, fid, shard_name in rows]


def merged_summary(shards: Sequence[Shard], merged_matches: Sequence[dict]
                   ) -> dict:
    """Counts only. Every metric field is omitted and named."""
    first = shards[0].summary if shards else {}
    recomputed = ("n_matches", "n_matches_evaluated")
    omitted = sorted(key for key in first
                     if key not in SUMMARY_IDENTITY_KEYS
                     and key not in recomputed)
    summary = {key: first.get(key) for key in SUMMARY_IDENTITY_KEYS
               if key in first}
    summary["n_matches"] = len(merged_matches)
    summary["n_matches_evaluated"] = len(merged_matches)
    summary["omitted_summary_fields"] = omitted
    summary["omitted_reason"] = (
        "the merge writes counts only; every evaluator summary metric (log "
        "loss, Brier, ROI, edge, P&L, Kelly, Sharpe, win rate, ECE, timings, "
        "bootstrap outputs) is omitted because a pooled value must be "
        "recomputed by the scorer over the merged matches, never averaged "
        "across shards")
    return summary


def restamp_clusters(matches: Sequence[dict], cluster_source_dir
                     ) -> Tuple[List[dict], int]:
    """Re-derive `competition_cluster_id` over the WHOLE registered set.

    `run_sim_eval.py` builds its cluster lookup from `--test-dir`, i.e. from
    the fixture directory the run was given. A shard therefore stamps
    `event:<name>|block_start:<first date of that event IN THE SHARD>`, which
    is not the registered I3 block (critical invariant 7). Reslicing does not
    repair it: `cluster_id_with_resolution` prefers a stamped id. So the
    merge — the point at which the union becomes the registered set again —
    re-derives every id from `cluster_source_dir`, and fails closed on a
    fixture the lookup does not cover.

    Returns `(matches, n_changed)`; the input records are not mutated.
    """
    from sim_eval.eval_statistics import (  # noqa: E402
        AMBIGUOUS_CLUSTER_ALIAS, load_competition_clusters)

    lookup = load_competition_clusters(cluster_source_dir)
    out: List[dict] = []
    changed = 0
    for record in matches:
        keys = [str(record[key]) for key in
                ("cricsheet_id", "match_id", "display_match_id")
                if record.get(key)]
        cluster = None
        for key in keys:
            if key in lookup:
                cluster = lookup[key]
                break
        if cluster is None or cluster == AMBIGUOUS_CLUSTER_ALIAS:
            raise MergeError(
                f"cluster source {cluster_source_dir} does not resolve a "
                f"block for fixture {keys[0] if keys else '?'}; the merged "
                "file would carry a shard-local block id")
        updated = dict(record)
        if updated.get("competition_cluster_id") != cluster:
            changed += 1
        updated["competition_cluster_id"] = cluster
        out.append(updated)
    return out, changed


def merge(shards: Sequence[Shard], *, cluster_source_dir=None
          ) -> Tuple[dict, List[dict], dict]:
    """`(eval payload, raw rows, provenance payload)` in merged order."""
    by_name = {shard.name: shard for shard in shards}
    if len(by_name) != len(shards):
        raise MergeError(
            "two shards carry the same label; the merge would drop one "
            "(use shard_names to label them)")
    order = chronological_order(shards)

    matches = []
    raw_rows = []
    fixtures = []
    for fid, shard_name in order:
        shard = by_name[shard_name]
        record = shard.eval_matches.get(fid)
        if record is not None:
            matches.append(record)
        raw = shard.raw_rows.get(fid)
        if raw is not None:
            raw_rows.append(raw)
        row = dict(shard.fixtures[fid])
        row["shard"] = shard_name
        row["shard_output_dir"] = str(shard.path)
        fixtures.append(row)

    first = shards[0]
    restamp = None
    if cluster_source_dir is not None:
        matches, changed = restamp_clusters(matches, cluster_source_dir)
        restamp = {
            "field": "competition_cluster_id",
            "source_dir": str(cluster_source_dir),
            "contract": "tournament_time_block_v1",
            "n_changed": changed,
            "why": (
                "run_sim_eval builds its cluster lookup from the run's own "
                "fixture dir, so a shard stamps a block start computed over "
                "that shard only; the merge re-derives every block id over "
                "the registered set"),
        }
    evaluation = {
        "match_identity": first.match_identity,
        "summary": merged_summary(shards, matches),
        "matches": matches,
    }

    provenance = {
        "contract": CONTRACT,
        "arm": first.arm,
        "merged_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "shard_order_rule": (
            "fixtures concatenated in registered chronological "
            "(match_date, cricsheet_id) order across every shard"),
        "shards": [
            {
                "shard": shard.name,
                "output_dir": str(shard.path),
                "fixture_count": len(shard.fixtures),
                "n_scored": len(shard.scored_ids),
                "n_unscored": len(shard.unscored_ids),
                "n_skipped": len(shard.skipped_ids),
                "fixture_dir": shard.run.get("fixture_dir"),
                "fixture_dir_hash": shard.run.get("fixture_dir_hash"),
                "started_at": shard.run.get("started_at"),
                "finished_at": shard.run.get("finished_at"),
            }
            for shard in shards
        ],
        "cluster_restamp": restamp,
        "run": {key: first.run.get(key) for key in RUN_IDENTITY_KEYS
                if key in first.run},
        "run_fields_that_differ_per_shard": list(RUN_PER_SHARD_KEYS),
        "run_field_note": (
            "as_of.matches_advanced is a RUN-cumulative counter, so a shard's "
            "value counts only the matches that shard advanced; the "
            "shard-invariant as-of facts are as_of.date and "
            "as_of.same_day_advanced_before"),
        "fixtures": fixtures,
    }
    return evaluation, raw_rows, provenance


def write_merged(out_dir, evaluation: dict, raw_rows: Sequence[dict],
                 provenance: dict) -> Dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    assert_no_metric_values(provenance["shards"])
    assert_no_metric_values(provenance.get("cluster_restamp"))
    assert_no_metric_values(evaluation["summary"])

    eval_path = out / EVAL_FILENAME
    eval_path.write_text(json.dumps(evaluation, indent=2) + "\n")
    raw_path = out / RAW_FILENAME
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in raw_rows:
            handle.write(json.dumps(row) + "\n")
    provenance_path = out / PROVENANCE_FILENAME
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    return {"eval": eval_path, "raw_sims": raw_path,
            "provenance": provenance_path}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge one arm's shard outputs after asserting the union "
                    "is the registered fixture set (counts only; no metric "
                    "is read or written)")
    parser.add_argument("--shard-dir", action="append", required=True,
                        help="one shard's arm output dir; repeat per shard")
    parser.add_argument("--expected-fixtures", default=None,
                        help="fixture dir, JSON id list or newline id list")
    parser.add_argument("--config", default=None,
                        help="registered stage-1 config; with --arm and "
                             "--block, its fixture dir is the expected set")
    parser.add_argument("--arm", default=None)
    parser.add_argument("--block", default="full_run",
                        help="config block whose fixture_dir is expected "
                             "(default full_run)")
    parser.add_argument("--cluster-source-dir", default=None,
                        help="re-derive competition_cluster_id over this "
                             "whole fixture dir (the registered set). "
                             "Without it the merged records keep the "
                             "shard-local block ids run_sim_eval stamped "
                             "from each shard's own fixture dir, which is "
                             "NOT the registered I3 block")
    parser.add_argument("--odds", default=None,
                        help="odds file whose records must each be claimed "
                             "by exactly one scored fixture")
    parser.add_argument("--out-dir", default=None,
                        help="where the merged artefacts are written; "
                             "omit with --check-only")
    parser.add_argument("--check-only", action="store_true",
                        help="run the assertions and print counts, write "
                             "nothing")
    return parser


def _expected_from_config(config_path, arm, block) -> List[str]:
    import yaml
    if not arm:
        raise MergeError("--config needs --arm")
    payload = yaml.safe_load(Path(config_path).read_text())
    arms = payload.get("arms") or {}
    arm_block = arms.get(arm) or {}
    run_block = arm_block.get(block) or {}
    fixture_dir = run_block.get("fixture_dir") or arm_block.get(
        "fixture_set", {}).get("path")
    if not fixture_dir:
        raise MergeError(
            f"config has no fixture dir for arm {arm!r} block {block!r}")
    return read_expected_fixtures(REPO_ROOT / fixture_dir
                                  if not Path(fixture_dir).is_absolute()
                                  else fixture_dir)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        names = shard_names(args.shard_dir)
        shards = [load_shard(directory, name)
                  for directory, name in zip(args.shard_dir, names)]
        if args.expected_fixtures:
            expected = read_expected_fixtures(args.expected_fixtures)
        elif args.config:
            expected = _expected_from_config(args.config, args.arm, args.block)
        else:
            raise MergeError(
                "give --expected-fixtures or --config with --arm")
        odds_ids = read_odds_ids(args.odds) if args.odds else None
    except MergeError as error:
        print(f"merge_shards: REFUSED: {error}")
        return EXIT_REFUSED
    except (OSError, json.JSONDecodeError) as error:
        print(f"merge_shards: REFUSED: {error}")
        return EXIT_REFUSED

    problems = check_all(shards, expected, odds_ids)
    n_scored = sum(len(shard.scored_ids) for shard in shards)
    n_unscored = sum(len(shard.unscored_ids) for shard in shards)
    n_skipped = sum(len(shard.skipped_ids) for shard in shards)
    n_fixtures = sum(len(shard.fixtures) for shard in shards)

    print(f"merge_shards: arm {shards[0].arm} over {len(shards)} shard(s)")
    for shard in shards:
        print(f"  shard {shard.name}: fixtures {len(shard.fixtures)} "
              f"scored {len(shard.scored_ids)} "
              f"unscored {len(shard.unscored_ids)} "
              f"skipped {len(shard.skipped_ids)}")
    print(f"  union {n_fixtures} fixtures; expected {len(expected)}")
    print(f"  scored {n_scored} + unscored {n_unscored} + skipped "
          f"{n_skipped} = {n_scored + n_unscored + n_skipped}")
    if odds_ids is not None:
        expected_set = {str(value) for value in expected}
        print(f"  odds file rows {len(odds_ids)}; rows for expected fixtures "
              f"{len([v for v in odds_ids if str(v) in expected_set])}")

    failed = {name: rows for name, rows in problems.items() if rows}
    for name, rows in problems.items():
        print(f"  [{'FAIL' if rows else 'pass'}] {name}")
        for row in rows:
            print(f"      {row}")
    if failed:
        print(f"merge_shards: FAILED ({len(failed)} assertion(s))")
        return EXIT_FAILED

    if args.check_only:
        print("merge_shards: assertions passed (check-only, nothing written)")
        return EXIT_OK
    if not args.out_dir:
        print("merge_shards: REFUSED: give --out-dir or --check-only")
        return EXIT_REFUSED

    try:
        evaluation, raw_rows, provenance = merge(
            shards, cluster_source_dir=args.cluster_source_dir)
        written = write_merged(args.out_dir, evaluation, raw_rows, provenance)
    except MergeError as error:
        print(f"merge_shards: REFUSED: {error}")
        return EXIT_REFUSED
    print(f"merge_shards: merged {len(evaluation['matches'])} evaluator "
          f"record(s), {len(raw_rows)} raw simulation row(s), "
          f"{len(provenance['fixtures'])} provenance row(s)")
    if provenance.get("cluster_restamp"):
        print(f"  competition_cluster_id re-derived over "
              f"{provenance['cluster_restamp']['source_dir']}: "
              f"{provenance['cluster_restamp']['n_changed']} record(s) "
              "changed")
    print(f"  omitted summary fields: "
          f"{len(evaluation['summary']['omitted_summary_fields'])}")
    for label, path in written.items():
        print(f"  {label}: {path}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
