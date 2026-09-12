#!/usr/bin/env python3
"""Write and verify the stage 2 ANALYSIS and DECISION manifest (D10.9).

Astra gate 2 round 1 MUST-FIX 6: "no separate verifiable analysis manifest
exists; the config still carries launch-era analysis-source hashes that differ
from current code; and the report uses `dependency/recert/`, while the renderer
defaults to its parent and loads only immediate `*.json`, so a default
regeneration therefore changes the certification evidence. Record and verify
the actual invocation rather than merely hashing whatever files happen to be
supplied."

This script is that manifest. It is deliberately SEPARATE from
`scripts/sequence_track/pin_stage2.py`, which owns the config's `provenance:`
block — the immutable TRAINING provenance, hashed at launch and never rewritten
afterwards. Nothing here touches `experiments/configs/seq_stage2_v1.yaml`.

What the manifest records:

* **analysis code** — the current sha256 of `stage2_stats.py`,
  `render_stage2_report.py`, `sim_eval/eval_statistics.py`,
  `registered_experiment.py` and this pin. These are the files that turn the
  runs into the numbers and the prose; the config's launch-era hashes for them
  are training provenance and are (correctly) stale.
* **the exact invocations** — the full argv of the statistics run and of the
  renderer run, every path argument included, as an ordered list. A
  regeneration that changes `--dependency-dir`, `--stats-json`, `--out`, the
  reps or the rng seed is therefore drift that `--verify` names, rather than a
  silent change of which evidence the report was built from.
* **the evidence** — sha256 of the statistics JSON, the k-selection record, the
  rendered report and the registered config; the certificate set the recorded
  `--dependency-dir`
  actually yields, each with its sha256, so a certificate added, removed or
  edited under that directory is drift; and, per configuration, the
  `summary.yaml` sha256 **as the statistics consumed it**, copied out of the
  (hash-verified) statistics JSON.
* **the dependency coverage** — which masked arms hold a complete,
  md5-authenticated trained certificate, and which are blocked. This is
  recomputed from the checkpoints on disk and compared, so a checkpoint whose
  bytes changed or went missing after the report was written refuses, even
  though every certificate FILE still hashes the same (Astra gate 2 round 2).
* **the decisions** — the selected k with its rule and margin, and the family
  map (candidate, reference and slice of every registered family member).

On `summary.yaml`: those files are rewritten whenever another training night
adds a seed, so their live hash is NOT evidence of record and is not compared.
What is compared is the hash each summary had when the statistics read it,
which is anchored by the statistics JSON's own verified hash. The live state is
recorded separately as `live_summary_state`, informational only.

Usage:
    uv run --no-sync python scripts/sequence_track/pin_stage2_analysis.py --write
    uv run --no-sync python scripts/sequence_track/pin_stage2_analysis.py --verify

`--verify` prints `OK` and exits 0 only when every compared field matches; on
any drift it names the field and exits 1.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from sequence_track.render_stage2_report import (  # noqa: E402
    DEFAULT_DEPENDENCY_DIR,
    DEFAULT_OUT as DEFAULT_REPORT,
    dependency_certificates,
    load_dependency,
)
from sequence_track.stage2_stats import (  # noqa: E402
    DEFAULT_BASE_LOGITS,
    DEFAULT_BLOCK_SOURCE,
    DEFAULT_CONFIG,
    DEFAULT_FRAME_DIR,
    DEFAULT_KSWEEP_OUT,
    DEFAULT_RUNS_ROOT,
    DEFAULT_STATS_OUT,
    REGISTERED_SEEDS,
    REPS,
    RNG_SEED,
    RefusalError,
    guard_path,
    read_json,
    rel,
    require_distinct_out,
    sha256_file,
)

PIN_VERSION = "seq_stage2_analysis_pin_v1"
DEFAULT_PIN = REPO / "eval_out" / "seq_stage2" / "analysis_pin.json"

# The analysis and decision code. `pin_stage2.py` is NOT here: it pins training
# provenance, which this manifest does not own.
ANALYSIS_SOURCES: tuple[str, ...] = (
    "scripts/sequence_track/stage2_stats.py",
    "scripts/sequence_track/render_stage2_report.py",
    "scripts/sim_eval/eval_statistics.py",
    "scripts/registered_experiment.py",
    "scripts/sequence_track/pin_stage2_analysis.py",
)

# Every field `--verify` compares. Anything outside this list is recorded for
# the reader and never used to pass or fail a verification.
# `evidence.config` and `dependency_coverage` are compared because of Astra
# gate 2 round 2: without them the pin did not compare what it records. In an
# in-memory simulation where checkpoint authentication failed for all four
# masked arms, `verify()` still returned `True, []` while its own rebuilt
# coverage reported all four blocked; and changing only the config's contents
# produced no drift at all, because the config hash was recorded and never
# read back.
COMPARED_FIELDS: tuple[str, ...] = (
    "pin_version",
    "analysis_code",
    "invocations",
    "evidence.statistics",
    "evidence.k_selection",
    "evidence.report",
    "evidence.config",
    "evidence.dependency_dir",
    "evidence.dependency_certificates",
    "evidence.summary_yaml_at_statistics",
    "dependency_coverage",
    "decisions.k_selection",
    "decisions.family_map",
)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_head() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short",
                              "HEAD"], capture_output=True, text=True,
                             timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _hash_or_missing(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        return {"path": rel(path), "present": False, "sha256": None}
    return {"path": rel(path), "present": True, "sha256": sha256_file(path)}


def analysis_code() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for relative in ANALYSIS_SOURCES:
        path = REPO / relative
        out[relative] = sha256_file(path) if path.is_file() else None
    return out


def statistics_invocation(config: Path, runs_root: Path, frame_dir: Path,
                          block_source: Path, base_logits: Path,
                          stats_json: Path, reps: int, rng_seed: int,
                          seeds: Sequence[int]) -> list[str]:
    return ["uv", "run", "--no-sync", "python",
            "scripts/sequence_track/stage2_stats.py", "stats",
            "--config", rel(config),
            "--runs-root", rel(runs_root),
            "--frame-dir", rel(frame_dir),
            "--block-source-dir", rel(block_source),
            "--base-logits", rel(base_logits),
            "--out", rel(stats_json),
            "--reps", str(reps),
            "--rng-seed", str(rng_seed),
            "--seeds", ",".join(str(s) for s in seeds)]


def ksweep_invocation(config: Path, runs_root: Path, k_selection: Path,
                      seeds: Sequence[int]) -> list[str]:
    return ["uv", "run", "--no-sync", "python",
            "scripts/sequence_track/stage2_stats.py", "ksweep",
            "--config", rel(config),
            "--runs-root", rel(runs_root),
            "--out", rel(k_selection),
            "--seeds", ",".join(str(s) for s in seeds)]


def renderer_invocation(stats_json: Path, k_selection: Path, config: Path,
                        dependency_dir: Path, report: Path) -> list[str]:
    return ["uv", "run", "--no-sync", "python",
            "scripts/sequence_track/render_stage2_report.py",
            "--stats-json", rel(stats_json),
            "--k-selection", rel(k_selection),
            "--config", rel(config),
            "--dependency-dir", rel(dependency_dir),
            "--out", rel(report)]


def consumed_certificates(dependency_dir: Path) -> list[dict[str, Any]]:
    """Every certificate the recorded `--dependency-dir` actually yields.

    This is the field that detects MUST-FIX 6's defect directly: rendering from
    `dependency/recert` and rendering from `dependency` consume different
    certificate sets, and the two produce different lists here.
    """
    directory = guard_path(dependency_dir)
    rows: list[dict[str, Any]] = []
    if not directory.is_dir():
        return rows
    for path in sorted(directory.rglob("*.json")):
        rows.append({"path": rel(path), "sha256": sha256_file(path)})
    return rows


def summary_hashes_at_statistics(stats: Mapping[str, Any]
                                 ) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for config_id, block in sorted((stats.get("runs") or {}).items()):
        summary = block.get("summary") or {}
        out[config_id] = {"summary_path": summary.get("summary_path"),
                          "sha256": summary.get("sha256"),
                          "seeds_recorded": summary.get("seeds_recorded")}
    return out


def live_summary_state(stats: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The summaries as they are on disk NOW — informational, never compared."""
    out: dict[str, dict[str, Any]] = {}
    for config_id, recorded in summary_hashes_at_statistics(stats).items():
        path = recorded.get("summary_path")
        row: dict[str, Any] = {"summary_path": path, "sha256_now": None,
                               "matches_statistics": None}
        if path:
            absolute = REPO / path
            if absolute.is_file():
                row["sha256_now"] = sha256_file(absolute)
                row["matches_statistics"] = (row["sha256_now"]
                                             == recorded.get("sha256"))
        out[config_id] = row
    return out


def family_map(stats: Mapping[str, Any]) -> list[dict[str, Any]]:
    out = []
    for family in stats.get("families") or []:
        out.append({
            "candidate": family.get("candidate"),
            "holm_group": family.get("holm_group"),
            "members": [{"member": member.get("member"),
                         "candidate": member.get("candidate"),
                         "reference": member.get("reference"),
                         "slice": member.get("slice"),
                         "threshold": member.get("threshold"),
                         "kind": member.get("kind")}
                        for member in family.get("members") or []]})
    return out


def k_decision(k_record: Mapping[str, Any] | None) -> dict[str, Any]:
    if not k_record:
        return {"available": False}
    return {"available": True,
            "selection": k_record.get("selection"),
            "selected_k": k_record.get("selected_k"),
            "selected_config_id": k_record.get("selected_config_id"),
            "rule": k_record.get("rule"),
            "tolerance_ll": k_record.get("tolerance_ll"),
            "registered_sweep": k_record.get("registered_sweep"),
            "means": k_record.get("means"),
            "best_mean_k": k_record.get("best_mean_k"),
            "margin_vs_k30": k_record.get("margin_vs_k30"),
            "provisional": k_record.get("provisional"),
            "selection_record_sha256":
                k_record.get("selection_record_sha256"),
            "rows": [{"k": row.get("k"), "config_id": row.get("config_id"),
                      "mean_ll": row.get("mean_ll"),
                      "per_seed": row.get("per_seed")}
                     for row in k_record.get("rows") or []]}


def build(stats_json: Path, k_selection: Path, config: Path,
          dependency_dir: Path, report: Path, runs_root: Path,
          frame_dir: Path, block_source: Path, base_logits: Path,
          reps: int, rng_seed: int, seeds: Sequence[int]) -> dict[str, Any]:
    stats_json = guard_path(stats_json)
    if not stats_json.is_file():
        raise RefusalError(
            f"no statistics JSON at {rel(stats_json)}: the analysis pin "
            "records the evidence of an analysis that has run, never a "
            "planned one")
    stats = read_json(stats_json)
    k_record = (read_json(k_selection)
                if guard_path(k_selection).is_file() else None)
    certificates = dependency_certificates(load_dependency(dependency_dir))
    return {
        "pin_version": PIN_VERSION,
        "what_this_is": (
            "the stage 2 ANALYSIS and DECISION manifest (D10.9). It is "
            "separate from experiments/configs/seq_stage2_v1.yaml's "
            "provenance block, which is immutable training provenance and is "
            "not written or verified here."),
        "written_at_utc": _now(),
        "git_head_short": _git_head(),
        "compared_fields": list(COMPARED_FIELDS),
        "not_compared": [
            "written_at_utc", "git_head_short", "live_summary_state",
            ("live_summary_state is informational: a later training night "
             "rewrites runs/<config>/summary.yaml, so the evidence of record "
             "is the hash the statistics consumed, not the file's live hash")],
        "analysis_code": analysis_code(),
        "invocations": {
            "statistics": statistics_invocation(
                config, runs_root, frame_dir, block_source, base_logits,
                stats_json, reps, rng_seed, seeds),
            "ksweep": ksweep_invocation(config, runs_root, k_selection, seeds),
            "renderer": renderer_invocation(stats_json, k_selection, config,
                                            dependency_dir, report),
        },
        "evidence": {
            "statistics": _hash_or_missing(stats_json),
            "k_selection": _hash_or_missing(k_selection),
            "report": _hash_or_missing(report),
            "config": _hash_or_missing(config),
            "dependency_dir": rel(dependency_dir),
            "dependency_certificates": consumed_certificates(dependency_dir),
            "summary_yaml_at_statistics": summary_hashes_at_statistics(stats),
        },
        "live_summary_state": live_summary_state(stats),
        "dependency_coverage": {
            "trained_checkpoint_coverage":
                certificates["trained_checkpoint_coverage"],
            "trained_checkpoint_coverage_complete":
                certificates["trained_checkpoint_coverage_complete"],
            "registered_controls_complete":
                certificates["registered_controls_complete"],
            "blocked": sorted(certificates["blocked"]),
        },
        "decisions": {
            "k_selection": k_decision(k_record),
            "family_map": family_map(stats),
            "seeds": list(seeds),
            "reps": reps,
            "rng_seed": rng_seed,
        },
    }


def _get(payload: Mapping[str, Any], dotted: str) -> Any:
    node: Any = payload
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def compare(recorded: Mapping[str, Any], rebuilt: Mapping[str, Any]
            ) -> list[str]:
    drift = []
    for field in COMPARED_FIELDS:
        left, right = _get(recorded, field), _get(rebuilt, field)
        if json.dumps(left, sort_keys=True, default=str) != json.dumps(
                right, sort_keys=True, default=str):
            drift.append(field)
    return drift


def verify(pin_path: Path) -> tuple[bool, list[str], dict[str, Any]]:
    """Rebuild from the RECORDED invocation and compare, field by field.

    The recorded renderer invocation names the `--dependency-dir` the report
    was built from, so the rebuild reads that directory rather than a default:
    a report regenerated from a different directory shows up as drift in
    `evidence.dependency_dir`, `evidence.dependency_certificates` and
    `evidence.report`.
    """
    recorded = read_json(guard_path(pin_path))
    invocations = recorded.get("invocations") or {}
    renderer = list(invocations.get("renderer") or [])
    statistics = list(invocations.get("statistics") or [])
    ksweep = list(invocations.get("ksweep") or [])

    def argument(argv: Sequence[str], flag: str, default: Any) -> Any:
        argv = list(argv)
        return argv[argv.index(flag) + 1] if flag in argv else default

    stats_json = REPO / str(argument(renderer, "--stats-json",
                                     rel(DEFAULT_STATS_OUT)))
    k_selection = REPO / str(argument(renderer, "--k-selection",
                                      rel(DEFAULT_KSWEEP_OUT)))
    config = REPO / str(argument(renderer, "--config", rel(DEFAULT_CONFIG)))
    dependency_dir = REPO / str(argument(renderer, "--dependency-dir",
                                         rel(DEFAULT_DEPENDENCY_DIR)))
    report = REPO / str(argument(renderer, "--out", rel(DEFAULT_REPORT)))
    runs_root = REPO / str(argument(statistics, "--runs-root",
                                    rel(DEFAULT_RUNS_ROOT)))
    frame_dir = REPO / str(argument(statistics, "--frame-dir",
                                    rel(DEFAULT_FRAME_DIR)))
    block_source = REPO / str(argument(statistics, "--block-source-dir",
                                       rel(DEFAULT_BLOCK_SOURCE)))
    base_logits = REPO / str(argument(statistics, "--base-logits",
                                      rel(DEFAULT_BASE_LOGITS)))
    reps = int(argument(statistics, "--reps", REPS))
    rng_seed = int(argument(statistics, "--rng-seed", RNG_SEED))
    seed_text = str(argument(statistics, "--seeds",
                             ",".join(str(s) for s in REGISTERED_SEEDS)))
    seeds = tuple(int(s) for s in seed_text.split(",") if s.strip())
    # A ksweep invocation that names a different record than the renderer reads
    # is itself drift, and is reported as such rather than silently ignored.
    k_from_sweep = argument(ksweep, "--out", None)

    rebuilt = build(stats_json, k_selection, config, dependency_dir, report,
                    runs_root, frame_dir, block_source, base_logits,
                    reps, rng_seed, seeds)
    drift = compare(recorded, rebuilt)
    if k_from_sweep is not None and str(k_from_sweep) != rel(k_selection):
        drift.append("invocations.ksweep --out disagrees with "
                     "invocations.renderer --k-selection")
    return (not drift), drift, rebuilt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true",
                      help="recompute and write the analysis manifest")
    mode.add_argument("--verify", action="store_true",
                      help="recompute and diff against the written manifest")
    parser.add_argument("--pin", type=Path, default=DEFAULT_PIN)
    parser.add_argument("--stats-json", type=Path, default=DEFAULT_STATS_OUT)
    parser.add_argument("--k-selection", type=Path, default=DEFAULT_KSWEEP_OUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dependency-dir", type=Path,
                        default=DEFAULT_DEPENDENCY_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--frame-dir", type=Path, default=DEFAULT_FRAME_DIR)
    parser.add_argument("--block-source-dir", type=Path,
                        default=DEFAULT_BLOCK_SOURCE)
    parser.add_argument("--base-logits", type=Path, default=DEFAULT_BASE_LOGITS)
    parser.add_argument("--reps", type=int, default=REPS)
    parser.add_argument("--rng-seed", type=int, default=RNG_SEED)
    parser.add_argument("--seeds",
                        default=",".join(str(s) for s in REGISTERED_SEEDS))
    args = parser.parse_args(argv)
    seeds = tuple(int(s) for s in str(args.seeds).split(",") if s.strip())

    try:
        # A five-seed analysis pin may not overwrite the two-seed one.
        require_distinct_out(args.config, args.pin, DEFAULT_PIN, "--pin")
        if args.write:
            payload = build(args.stats_json, args.k_selection, args.config,
                            args.dependency_dir, args.report, args.runs_root,
                            args.frame_dir, args.block_source_dir,
                            args.base_logits, args.reps, args.rng_seed, seeds)
            pin = guard_path(args.pin)
            pin.parent.mkdir(parents=True, exist_ok=True)
            pin.write_text(json.dumps(payload, indent=2, sort_keys=False,
                                      default=str) + "\n")
            coverage = payload["dependency_coverage"]
            print(f"wrote {rel(pin)}")
            print("  analysis code: "
                  + ", ".join(f"{Path(name).name} "
                              f"{str(digest)[:12]}…"
                              for name, digest
                              in payload["analysis_code"].items()))
            print("  renderer: "
                  + " ".join(payload["invocations"]["renderer"]))
            print(f"  dependency dir: {payload['evidence']['dependency_dir']} "
                  f"({len(payload['evidence']['dependency_certificates'])} "
                  "certificates consumed), trained coverage complete: "
                  f"{coverage['trained_checkpoint_coverage_complete']}")
            return 0
        ok, drift, _ = verify(args.pin)
        if ok:
            print(f"OK: {rel(guard_path(args.pin))} verifies — analysis code, "
                  "invocations, evidence hashes (config included), dependency "
                  "coverage, k selection and family map all match")
            return 0
        print("DRIFT: the stage 2 analysis pin does not verify. Fields that "
              "differ:", file=sys.stderr)
        for field in drift:
            print(f"  - {field}", file=sys.stderr)
        return 1
    except RefusalError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
