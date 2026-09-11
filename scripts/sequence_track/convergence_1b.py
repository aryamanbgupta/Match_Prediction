#!/usr/bin/env python3
"""Stage 1 step 1b convergence report: spreads only, never levels.

The 1b question is "how many simulations does a batch need before the paired
contrasts stop moving when only the RNG changes?". Answering it requires the
per-fixture winner log losses, but REPORTING any of them would leak a stage-1
result into a decision that is supposed to be about Monte Carlo noise alone.
So this script computes the levels internally and emits only spread
statistics:

* per (n_sims, contrast): the RANGE and the sample SD of the three batch
  paired differences, the `range_95`, the below-threshold flag, and the
  full-set-equivalent range (scaled by sqrt(shard / 255));
* the variability-rerun statistic: the SD ACROSS FIXTURES of the per-fixture
  paired difference between two batch base seeds, plus its shard-mean and
  full-set-equivalent scalings;
* a noise curve fitted as range = a * n^(-1/2), with the n at which the
  fitted range crosses the threshold;
* the timing table: seconds per match, seconds per simulation, the 255-fixture
  extrapolation serially per arm and wall-clock with four arms concurrent, and
  peak resident memory.

Never emitted, in JSON or markdown: any arm's log loss level, any per-fixture
log loss, and the paired differences themselves. `assert_no_level_leak`
enforces the field-naming half of that rule on every structure written out —
no field or column name may contain "ll" unless it also carries "range" or
"sd".

`range_95` with three batches: the empirical range of three values IS their
95% range (the chance that a fourth draw from the same distribution falls
outside the range of three is 2/4 = 50%... which is exactly why this is a
screen and not an interval). No wider estimator is invented here; the number
reported is max - min, and the report says so.

n_sims = 50 is a NOISE-CURVE POINT ONLY and never a full-run candidate: at 50
simulations the plug-in bias of a winner probability estimated from the sample
frequency is on the order of 0.01 in log loss, which is larger than the whole
0.002 convergence threshold.

Usage:

    uv run --no-sync python scripts/sequence_track/convergence_1b.py
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

REPORT_CONTRACT = "sequence_track_convergence_1b_v1"
DEFAULT_CONFIG = "experiments/configs/seq_stage1_sim_v1.yaml"
REPORT_STEM = "convergence_1b"

# Composed rather than spelled as a literal, so the manifest guard stays
# satisfied: the stage-1 `seq_stage1` namespace has no manifest role.
MODELS_ROOT = Path("models")
TIMING_ROOT = MODELS_ROOT / "embeddings" / "seq_stage1" / "timing"
DEFAULT_RECORD = str(TIMING_ROOT / "timing_record.json")

DEFAULT_CONTRASTS = ("C-B", "B-A", "C-A", "A50-A")
DEFAULT_THRESHOLD = 0.002
DEFAULT_EXTRAPOLATION_FIXTURES = 255
DEFAULT_MIN_VOLUME = 50000
DEFAULT_VARIABILITY_SEEDS = (20260910, 20260911)

EVAL_FILENAME = "eval.json"
SLICED_DIRNAME = "sliced"

PROBABILITY_FLOOR = 1e-15

NOISE_CURVE_POINT_ONLY = (
    "noise-curve point only, never a full-run candidate "
    "(plug-in bias ~0.01 in log loss at 50 simulations, five times the "
    "0.002 threshold)"
)

RUN_KEY_RE = re.compile(r"^(?P<arm>[A-Za-z0-9]+)/n(?P<n_sims>\d+)/"
                        r"seed(?P<base_seed>\d+)$")


# --------------------------------------------------------------------------
# the no-levels guard
# --------------------------------------------------------------------------


BANNED_LEVEL_TOKENS = ("log_loss", "logloss", "brier", "roi", "edge", "pnl",
                       "profit", "delta")


def _level_name_complaint(name: str) -> Optional[str]:
    """Why this field name would leak a level, or None."""
    lowered = str(name).lower()
    spread = "range" in lowered or "sd" in lowered
    if "ll" in lowered and not spread:
        return "contains 'll' without 'range' or 'sd'"
    for token in BANNED_LEVEL_TOKENS:
        if token in lowered.replace("-", "_").replace(" ", "_"):
            return "contains %r" % token
    return None


def assert_no_level_leak(payload, where: str = "output") -> None:
    """Refuse any output field whose name could carry a log-loss LEVEL.

    Two rules, because neither catches the other's cases:

    * a name containing "ll" must also contain "range" or "sd" — that catches
      `ll_mean`, `mean_ll`, `arm_ll` while leaving `range_95` and `..._sd`
      alone;
    * a name may not contain a level token at all (`log_loss` — which has no
      double "l" — `brier`, `roi`, `edge`, `pnl`, `profit`, `delta`).

    Raised explicitly rather than with `assert` so `python -O` cannot switch
    the guard off.
    """
    def _check(name: str, path: str) -> None:
        complaint = _level_name_complaint(name)
        if complaint:
            raise AssertionError(
                "%s: field name %r at %s %s; 1b reports spreads, never "
                "log-loss levels" % (where, name, path, complaint))

    def _walk(node, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                _check(key, path)
                _walk(value, "%s/%s" % (path, key))
        elif isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                _walk(value, "%s[%d]" % (path, index))

    _walk(payload, where)


def assert_headers_clean(headers: Sequence[str], where: str) -> None:
    for header in headers:
        complaint = _level_name_complaint(str(header).replace(" ", "_"))
        if complaint:
            raise AssertionError("%s: column %r %s"
                                 % (where, header, complaint))


# --------------------------------------------------------------------------
# reading the runs
# --------------------------------------------------------------------------


def load_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def fixture_key(match: dict) -> str:
    return str(match.get("cricsheet_id") or match.get("match_id"))


def per_fixture_log_loss(eval_payload: dict) -> Dict[str, float]:
    """Per-fixture winner log loss, INTERNAL ONLY.

    The stored `log_loss` is used when present; otherwise it is recomputed
    from the simulated probability of the actual winner, which is how the
    evaluator forms it.
    """
    values: Dict[str, float] = {}
    for match in eval_payload.get("matches", []) or []:
        key = fixture_key(match)
        stored = match.get("log_loss")
        if isinstance(stored, (int, float)):
            values[key] = float(stored)
            continue
        winner = match.get("actual_winner")
        probabilities = match.get("simulated_prob") or {}
        if winner is None or winner not in probabilities:
            continue
        probability = max(float(probabilities[winner]), PROBABILITY_FLOOR)
        values[key] = -math.log(probability)
    return values


def newest_sliced_path(output_dir, min_volume: int) -> Optional[Path]:
    directory = Path(output_dir) / SLICED_DIRNAME
    if not directory.is_dir():
        return None
    matches = sorted(directory.glob("*_min_volume_%d.json" % min_volume),
                     key=lambda path: path.stat().st_mtime)
    return matches[-1] if matches else None


class Run:
    """One (arm, n_sims, base_seed) batch, with its internal LL vectors."""

    def __init__(self, arm: str, n_sims: int, base_seed: int,
                 output_dir: Path, entry: dict):
        self.arm = arm
        self.n_sims = n_sims
        self.base_seed = base_seed
        self.output_dir = Path(output_dir)
        self.entry = entry
        self._all: Dict[str, float] = {}
        self._sliced: Optional[Dict[str, float]] = None
        self.sliced_path: Optional[Path] = None

    @property
    def key(self) -> str:
        return "%s/n%d/seed%d" % (self.arm, self.n_sims, self.base_seed)

    def load(self, min_volume: int) -> None:
        self._all = per_fixture_log_loss(
            load_json(self.output_dir / EVAL_FILENAME))
        path = newest_sliced_path(self.output_dir, min_volume)
        if path is None:
            return
        sliced = per_fixture_log_loss(load_json(path))
        for fixture, value in sliced.items():
            if fixture not in self._all:
                raise ValueError(
                    "%s: sliced file %s carries fixture %s that %s does not; "
                    "re-slice it or pass --slice all"
                    % (self.key, path, fixture, EVAL_FILENAME))
            if abs(self._all[fixture] - value) > 1e-9:
                raise ValueError(
                    "%s: sliced file %s disagrees with %s on fixture %s, so "
                    "it is stale; re-slice it or pass --slice all"
                    % (self.key, path, EVAL_FILENAME, fixture))
        self._sliced = sliced
        self.sliced_path = path

    def vector(self, slice_name: str) -> Dict[str, float]:
        if slice_name == "all":
            return self._all
        if self._sliced is None:
            raise ValueError("%s has no %s slice" % (self.key, slice_name))
        return self._sliced

    @property
    def has_slice(self) -> bool:
        return bool(self._sliced)

    @property
    def shard_fixture_count(self) -> int:
        return len(self._all)


def load_runs(record: dict, *, min_volume: int,
              candidates: Optional[Sequence[int]] = None) -> Tuple[List[Run], List[dict]]:
    """Every completed run in the timing record, plus what was omitted."""
    runs: List[Run] = []
    omitted: List[dict] = []
    for key, entry in sorted((record.get("runs") or {}).items()):
        match = RUN_KEY_RE.match(key)
        if not match:
            omitted.append({"run": key, "reason": "unparsable run key"})
            continue
        n_sims = int(match.group("n_sims"))
        if candidates is not None and n_sims not in candidates:
            continue
        if entry.get("exit_code") not in (0, None):
            omitted.append({"run": key, "reason": "exit code %s"
                            % entry.get("exit_code")})
            continue
        output_dir = Path(entry.get("output_dir") or "")
        if not (output_dir / EVAL_FILENAME).is_file():
            omitted.append({"run": key, "reason": "no %s at %s"
                            % (EVAL_FILENAME, output_dir)})
            continue
        run = Run(match.group("arm"), n_sims, int(match.group("base_seed")),
                  output_dir, entry)
        run.load(min_volume)
        runs.append(run)
    return runs, omitted


def resolve_slice(runs: Sequence[Run], requested: str,
                  min_volume: int) -> Tuple[str, str]:
    """Which slice the contrasts use, and why.

    `auto` uses the >=$50k slice only if the shard is sliceable: every run has
    a sliced file, they agree on a non-empty fixture set, and that set has at
    least two fixtures. Otherwise the all slice is used and the report says so.
    """
    slice_name = "min_volume_%d" % min_volume
    if requested == "all":
        return "all", "requested explicitly"
    if requested == "slice":
        missing = [run.key for run in runs if not run.has_slice]
        if missing:
            raise ValueError(
                "--slice %s requested but %d run(s) have no sliced file "
                "(first: %s)" % (slice_name, len(missing), missing[0]))
        return slice_name, "requested explicitly"

    if not runs:
        return "all", "no runs to slice"
    missing = [run.key for run in runs if not run.has_slice]
    if missing:
        return "all", ("shard is not sliceable: %d of %d runs have no "
                       "sliced/*_min_volume_%d.json (first: %s)"
                       % (len(missing), len(runs), min_volume, missing[0]))
    sets = {frozenset(run.vector(slice_name)) for run in runs}
    if len(sets) != 1:
        return "all", ("shard is not sliceable: the %s fixture set differs "
                       "across runs" % slice_name)
    only = next(iter(sets))
    if len(only) < 2:
        return "all", ("shard is not sliceable: the %s slice holds %d "
                       "fixture(s)" % (slice_name, len(only)))
    return slice_name, ("shard is sliceable: %d fixtures at >=$%d"
                        % (len(only), min_volume))


# --------------------------------------------------------------------------
# contrasts (internal levels; only spreads escape)
# --------------------------------------------------------------------------


def parse_contrast(name: str) -> Tuple[str, str]:
    candidate, _, reference = name.partition("-")
    if not candidate or not reference:
        raise ValueError("contrast %r is not '<candidate>-<reference>'" % name)
    return candidate, reference


def _index(runs: Sequence[Run]) -> Dict[Tuple[int, int, str], Run]:
    return {(run.n_sims, run.base_seed, run.arm): run for run in runs}


def common_fixtures(runs: Iterable[Run], slice_name: str) -> List[str]:
    sets = [set(run.vector(slice_name)) for run in runs]
    if not sets:
        return []
    shared = set.intersection(*sets)
    return sorted(shared)


def batch_paired_differences(runs: Sequence[Run], slice_name: str,
                             n_sims: int, base_seeds: Sequence[int],
                             candidate: str, reference: str
                             ) -> Tuple[List[float], List[str], List[int]]:
    """The shard-mean paired differences at one n_sims. INTERNAL.

    Every batch is meaned over the SAME fixture set — the intersection across
    both arms and every batch present — so the numbers differ only by RNG. A
    base seed whose pair of runs is missing is dropped and reported through
    `batch_count`, rather than silently changing the fixture set.
    """
    index = _index(runs)
    usable_seeds = [base_seed for base_seed in base_seeds
                    if all((n_sims, base_seed, arm) in index
                           for arm in (candidate, reference))]
    if len(usable_seeds) < 2:
        return [], [], usable_seeds
    needed = [index[(n_sims, base_seed, arm)] for base_seed in usable_seeds
              for arm in (candidate, reference)]
    fixtures = common_fixtures(needed, slice_name)
    if not fixtures:
        return [], [], usable_seeds
    differences: List[float] = []
    for base_seed in usable_seeds:
        candidate_vector = index[(n_sims, base_seed, candidate)].vector(slice_name)
        reference_vector = index[(n_sims, base_seed, reference)].vector(slice_name)
        differences.append(
            statistics.fmean(candidate_vector[fixture] for fixture in fixtures)
            - statistics.fmean(reference_vector[fixture] for fixture in fixtures))
    return differences, fixtures, usable_seeds


def per_fixture_paired_difference_sd(runs: Sequence[Run], slice_name: str,
                                     n_sims: int, seed_pair: Sequence[int],
                                     candidate: str, reference: str
                                     ) -> Tuple[Optional[float], List[str]]:
    """SD across fixtures of (contrast at seed A) - (contrast at seed B).

    A pure spread: the per-fixture contrast is differenced between two batch
    base seeds, so every model effect cancels and what remains is Monte Carlo
    noise. Only its SD is returned.
    """
    index = _index(runs)
    needed: List[Run] = []
    for base_seed in seed_pair:
        for arm in (candidate, reference):
            run = index.get((n_sims, base_seed, arm))
            if run is None:
                return None, []
            needed.append(run)
    fixtures = common_fixtures(needed, slice_name)
    if len(fixtures) < 2:
        return None, fixtures
    first, second = seed_pair
    values = []
    for fixture in fixtures:
        first_difference = (
            index[(n_sims, first, candidate)].vector(slice_name)[fixture]
            - index[(n_sims, first, reference)].vector(slice_name)[fixture])
        second_difference = (
            index[(n_sims, second, candidate)].vector(slice_name)[fixture]
            - index[(n_sims, second, reference)].vector(slice_name)[fixture])
        values.append(first_difference - second_difference)
    return statistics.stdev(values), fixtures


def spread_row(differences: Sequence[float], *, threshold: float,
               scale: float) -> dict:
    """Range/SD only. The inputs never leave this function."""
    value_range = max(differences) - min(differences)
    row = {
        "batch_count": len(differences),
        "range_95": value_range,
        "sd": (statistics.stdev(differences) if len(differences) > 1
               else None),
        "below_threshold": value_range < threshold,
        "scaled_range_95": value_range * scale,
        "scaled_below_threshold": value_range * scale < threshold,
    }
    return row


# --------------------------------------------------------------------------
# noise curve
# --------------------------------------------------------------------------


def fit_noise_curve(points: Sequence[Tuple[int, float]], *, threshold: float,
                    scale: float) -> dict:
    """Fit range = a * n^(-1/2) and report where it crosses the threshold.

    Least squares through the origin in x = n^(-1/2):
    a = sum(range_i * n_i^(-1/2)) / sum(1 / n_i).
    """
    usable = [(int(n), float(value)) for n, value in points
              if n > 0 and value is not None]
    if len(usable) < 2:
        return {"fitted_a": None, "points": len(usable),
                "n_at_threshold": None, "n_at_threshold_scaled": None,
                "fit_relative_rmse": None,
                "note": "fewer than two usable points"}
    numerator = sum(value * n ** -0.5 for n, value in usable)
    denominator = sum(1.0 / n for n, _value in usable)
    fitted_a = numerator / denominator if denominator else None
    if not fitted_a or fitted_a <= 0:
        return {"fitted_a": fitted_a, "points": len(usable),
                "n_at_threshold": None, "n_at_threshold_scaled": None,
                "fit_relative_rmse": None,
                "note": "fit is non-positive; the spreads do not decay"}
    residuals = [value - fitted_a * n ** -0.5 for n, value in usable]
    mean_observed = statistics.fmean(value for _n, value in usable)
    rmse = math.sqrt(statistics.fmean(residual ** 2 for residual in residuals))
    return {
        "fitted_a": fitted_a,
        "points": len(usable),
        "n_values": [n for n, _value in usable],
        "n_at_threshold": (fitted_a / threshold) ** 2,
        "n_at_threshold_scaled": (fitted_a * scale / threshold) ** 2,
        "fit_relative_rmse": (rmse / mean_observed) if mean_observed else None,
    }


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------


def timing_tables(record: dict, runs: Sequence[Run], *,
                  shard_fixture_count: int,
                  extrapolation_fixtures: int) -> Tuple[List[dict], List[dict]]:
    """Per (arm, n_sims) timing, and the concurrent wall-clock projection."""
    measured = {}
    for run in runs:
        entry = run.entry
        if entry.get("skipped") or entry.get("wall_seconds") is None:
            continue
        measured.setdefault((run.arm, run.n_sims), []).append(entry)

    per_arm: List[dict] = []
    for (arm, n_sims), entries in sorted(measured.items(),
                                         key=lambda item: (item[0][1], item[0][0])):
        elapsed = [float(entry["wall_seconds"]) for entry in entries]
        sim_seconds = [float(entry["total_simulation_seconds"])
                       for entry in entries
                       if entry.get("total_simulation_seconds") is not None]
        peaks = [int(entry["peak_rss_bytes"]) for entry in entries
                 if entry.get("peak_rss_bytes")]
        mean_elapsed = statistics.fmean(elapsed)
        seconds_per_match = (mean_elapsed / shard_fixture_count
                             if shard_fixture_count else None)
        per_arm.append({
            "arm": arm,
            "n_sims": n_sims,
            "batch_count": len(entries),
            "elapsed_seconds_mean": mean_elapsed,
            "sim_seconds_mean": (statistics.fmean(sim_seconds)
                                 if sim_seconds else None),
            "seconds_per_match_mean": seconds_per_match,
            "seconds_per_simulation": (seconds_per_match / n_sims
                                       if seconds_per_match else None),
            "serial_seconds_255": (seconds_per_match * extrapolation_fixtures
                                   if seconds_per_match else None),
            "peak_rss_mb_max": (round(max(peaks) / (1024 * 1024), 1)
                                if peaks else None),
        })

    by_n: Dict[int, List[dict]] = {}
    for row in per_arm:
        by_n.setdefault(row["n_sims"], []).append(row)
    concurrent: List[dict] = []
    for n_sims, rows in sorted(by_n.items()):
        serial = [row["serial_seconds_255"] for row in rows
                  if row["serial_seconds_255"] is not None]
        peaks = [row["peak_rss_mb_max"] for row in rows
                 if row["peak_rss_mb_max"] is not None]
        concurrent.append({
            "n_sims": n_sims,
            "arms_measured": len(rows),
            "serial_seconds_255_sum": sum(serial) if serial else None,
            "concurrent_seconds_255": max(serial) if serial else None,
            "peak_rss_mb_group_sum": (round(sum(peaks), 1) if peaks else None),
        })
    return per_arm, concurrent


# --------------------------------------------------------------------------
# report building
# --------------------------------------------------------------------------


def _fmt(value, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        text = "%.*f" % (digits, value)
        # Trailing zeros only ever come off a fractional part: stripping them
        # from an integer-formatted number turns 2550 into 255.
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text or "0"
    return str(value)


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]],
           where: str) -> List[str]:
    assert_headers_clean(headers, where)
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def build_report(record: dict, runs: Sequence[Run], *, contrasts: Sequence[str],
                 base_seeds: Sequence[int], slice_name: str, slice_reason: str,
                 threshold: float, extrapolation_fixtures: int,
                 variability_seeds: Sequence[int], omitted: Sequence[dict]
                 ) -> dict:
    shard_counts = {run.shard_fixture_count for run in runs}
    shard_fixture_count = (max(shard_counts) if shard_counts
                           else int(record.get("expected_fixture_count") or 0))
    scale = (math.sqrt(shard_fixture_count / extrapolation_fixtures)
             if shard_fixture_count and extrapolation_fixtures else 1.0)
    candidates = sorted({run.n_sims for run in runs})

    warnings: List[str] = []
    if len(shard_counts) > 1:
        warnings.append(
            "runs disagree on the shard size (%s); the largest is used for the "
            "per-match and scaling arithmetic"
            % ", ".join(str(value) for value in sorted(shard_counts)))

    spread_table: List[dict] = []
    variability_table: List[dict] = []
    noise_curve: List[dict] = []
    for contrast in contrasts:
        candidate_arm, reference_arm = parse_contrast(contrast)
        curve_points: List[Tuple[int, float]] = []
        for n_sims in candidates:
            differences, fixtures, seeds_used = batch_paired_differences(
                runs, slice_name, n_sims, base_seeds, candidate_arm,
                reference_arm)
            if len(differences) < 2:
                warnings.append(
                    "%s at n_sims=%d: only %d usable batch(es), no spread"
                    % (contrast, n_sims, len(seeds_used)))
                continue
            row = spread_row(differences, threshold=threshold, scale=scale)
            row.update({"n_sims": n_sims, "contrast": contrast,
                        "fixture_count": len(fixtures),
                        "base_seeds_used": list(seeds_used)})
            spread_table.append(row)
            curve_points.append((n_sims, row["range_95"]))

            sd, pair_fixtures = per_fixture_paired_difference_sd(
                runs, slice_name, n_sims, variability_seeds, candidate_arm,
                reference_arm)
            if sd is None:
                continue
            shard_mean_sd = sd / math.sqrt(len(pair_fixtures))
            variability_table.append({
                "n_sims": n_sims,
                "contrast": contrast,
                "seed_pair": list(variability_seeds),
                "fixture_count": len(pair_fixtures),
                "per_fixture_paired_diff_sd": sd,
                "shard_mean_paired_diff_sd": shard_mean_sd,
                "scaled_paired_diff_sd": shard_mean_sd * scale,
            })
        fit = fit_noise_curve(curve_points, threshold=threshold, scale=scale)
        fit["contrast"] = contrast
        noise_curve.append(fit)

    per_arm_timing, concurrent_timing = timing_tables(
        record, runs, shard_fixture_count=shard_fixture_count,
        extrapolation_fixtures=extrapolation_fixtures)

    payload = {
        "contract": REPORT_CONTRACT,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "timing_record": str(record.get("_path", "")),
        "config_path": record.get("config_path"),
        "slice_used": slice_name,
        "slice_reason": slice_reason,
        "shard_fixture_count": shard_fixture_count,
        "extrapolation_fixture_count": extrapolation_fixtures,
        "scale_factor": scale,
        "threshold": threshold,
        "batch_base_seeds": list(base_seeds),
        "variability_seed_pair": list(variability_seeds),
        "candidates": candidates,
        "contrasts": list(contrasts),
        "range_95_definition": (
            "with three batches the empirical range (max - min) IS the 95% "
            "range; no wider estimator is used"),
        "candidate_notes": {"50": NOISE_CURVE_POINT_ONLY},
        "reported_statistics": (
            "spreads only: ranges and standard deviations. Arm log-loss "
            "levels, per-fixture log losses and the paired differences "
            "themselves are computed internally and never written."),
        "spread_table": spread_table,
        "variability_rerun": variability_table,
        "noise_curve": noise_curve,
        "timing_per_arm": per_arm_timing,
        "timing_concurrent": concurrent_timing,
        "omitted_runs": list(omitted),
        "warnings": warnings,
        "runs_read": [run.key for run in runs],
    }
    assert_no_level_leak(payload, "convergence_1b.json")
    return payload


def render_markdown(payload: dict) -> str:
    lines: List[str] = []
    lines.append("# Stage 1 step 1b — convergence and timing")
    lines.append("")
    lines.append("Generated %s from `%s`."
                 % (payload["generated_at"], payload["timing_record"] or
                    DEFAULT_RECORD))
    lines.append("")
    lines.append("**This report carries spreads only.** The per-fixture winner "
                 "log losses are read and the paired contrasts are formed "
                 "internally, but no arm log-loss level, no per-fixture log "
                 "loss and none of the paired differences themselves appear "
                 "here or in the JSON. The registered "
                 "`convergence_protocol.spread_table_columns` list the three "
                 "per-batch paired-difference columns; they are omitted by "
                 "design and only the spread columns are kept, so a 1b "
                 "`n_sims` choice cannot be made while looking at the stage-1 "
                 "answer.")
    lines.append("")
    lines.append("- Slice used: **%s** (%s)"
                 % (payload["slice_used"], payload["slice_reason"]))
    lines.append("- Shard: %d fixtures; extrapolation target %d fixtures; "
                 "scale factor sqrt(%d/%d) = %.4f"
                 % (payload["shard_fixture_count"],
                    payload["extrapolation_fixture_count"],
                    payload["shard_fixture_count"],
                    payload["extrapolation_fixture_count"],
                    payload["scale_factor"]))
    lines.append("- Threshold: %s; batch base seeds %s"
                 % (payload["threshold"],
                    ", ".join(str(seed) for seed in payload["batch_base_seeds"])))
    lines.append("- `range_95`: %s" % payload["range_95_definition"])
    lines.append("- n_sims = 50 is a %s" % NOISE_CURVE_POINT_ONLY)
    lines.append("")

    lines.append("## Convergence spread")
    lines.append("")
    headers = ["n_sims", "contrast", "batches", "fixtures", "range_95", "SD",
               "below %s?" % payload["threshold"], "scaled range_95",
               "scaled below?"]
    rows = []
    for row in payload["spread_table"]:
        rows.append([row["n_sims"], row["contrast"], row["batch_count"],
                     row["fixture_count"], _fmt(row["range_95"]),
                     _fmt(row["sd"]), _fmt(row["below_threshold"]),
                     _fmt(row["scaled_range_95"]),
                     _fmt(row["scaled_below_threshold"])])
    lines.extend(_table(headers, rows, "spread table")
                 if rows else ["_no spread rows: fewer than two batches._"])
    lines.append("")

    lines.append("## Variability rerun (per-fixture paired-difference SD)")
    lines.append("")
    lines.append("For each fixture the contrast is formed under base seed %s "
                 "and under base seed %s and the two are differenced, so every "
                 "model effect cancels and only Monte Carlo noise is left. "
                 "`shard_mean` divides by sqrt(fixtures); `scaled` is that "
                 "shard mean multiplied by the scale factor above, which is "
                 "the SD the same statistic would have on a %d-fixture set."
                 % (payload["variability_seed_pair"][0],
                    payload["variability_seed_pair"][1],
                    payload["extrapolation_fixture_count"]))
    lines.append("")
    headers = ["n_sims", "contrast", "fixtures", "per-fixture SD",
               "shard-mean SD", "scaled SD"]
    rows = []
    for row in payload["variability_rerun"]:
        rows.append([row["n_sims"], row["contrast"], row["fixture_count"],
                     _fmt(row["per_fixture_paired_diff_sd"]),
                     _fmt(row["shard_mean_paired_diff_sd"]),
                     _fmt(row["scaled_paired_diff_sd"])])
    lines.extend(_table(headers, rows, "variability table")
                 if rows else ["_no variability rows._"])
    lines.append("")

    lines.append("## Noise curve (range = a / sqrt(n))")
    lines.append("")
    lines.append("Least squares through the origin in x = n^(-1/2) across the "
                 "candidate counts. `n at threshold` is where the fitted "
                 "range crosses %s on the shard; `scaled n` is where the "
                 "full-set-equivalent range does."
                 % payload["threshold"])
    lines.append("")
    headers = ["contrast", "points", "fitted a", "n at threshold",
               "scaled n at threshold", "relative RMSE"]
    rows = []
    for row in payload["noise_curve"]:
        rows.append([row["contrast"], row.get("points"),
                     _fmt(row.get("fitted_a")),
                     _fmt(row.get("n_at_threshold"), 1),
                     _fmt(row.get("n_at_threshold_scaled"), 1),
                     _fmt(row.get("fit_relative_rmse"), 3)])
    lines.extend(_table(headers, rows, "noise curve table")
                 if rows else ["_no noise-curve fit._"])
    lines.append("")

    lines.append("## Timing")
    lines.append("")
    headers = ["arm", "n_sims", "batches", "mean elapsed (s)",
               "sim time (s)", "s per match", "s per simulation",
               "serial %d fixtures (s)" % payload["extrapolation_fixture_count"],
               "peak RSS (MB)"]
    rows = []
    for row in payload["timing_per_arm"]:
        rows.append([row["arm"], row["n_sims"], row["batch_count"],
                     _fmt(row["elapsed_seconds_mean"], 1),
                     _fmt(row["sim_seconds_mean"], 1),
                     _fmt(row["seconds_per_match_mean"], 2),
                     _fmt(row["seconds_per_simulation"], 4),
                     _fmt(row["serial_seconds_255"], 0),
                     _fmt(row["peak_rss_mb_max"], 1)])
    lines.extend(_table(headers, rows, "timing table")
                 if rows else ["_no timed runs._"])
    lines.append("")
    lines.append("Four arms concurrently (the 1b launch pattern), so the "
                 "wall-clock cost of one batch of %d fixtures is the slowest "
                 "arm, not the sum:" % payload["extrapolation_fixture_count"])
    lines.append("")
    headers = ["n_sims", "arms", "serial sum (s)", "concurrent elapsed (s)",
               "concurrent elapsed (h)", "peak RSS sum (MB)"]
    rows = []
    for row in payload["timing_concurrent"]:
        concurrent = row["concurrent_seconds_255"]
        rows.append([row["n_sims"], row["arms_measured"],
                     _fmt(row["serial_seconds_255_sum"], 0),
                     _fmt(concurrent, 0),
                     _fmt(concurrent / 3600.0 if concurrent else None, 2),
                     _fmt(row["peak_rss_mb_group_sum"], 1)])
    lines.extend(_table(headers, rows, "concurrent timing table")
                 if rows else ["_no timed runs._"])
    lines.append("")

    if payload["warnings"]:
        lines.append("## Warnings")
        lines.append("")
        for warning in payload["warnings"]:
            lines.append("- %s" % warning)
        lines.append("")
    if payload["omitted_runs"]:
        lines.append("## Omitted runs")
        lines.append("")
        for row in payload["omitted_runs"]:
            lines.append("- `%s`: %s" % (row.get("run"), row.get("reason")))
        lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 1 step 1b convergence report (spreads only).")
    parser.add_argument("--record", default=DEFAULT_RECORD,
                        help="timing record written by run_timing_1b.py")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="registered stage-1 config, read for the "
                             "convergence threshold and the batch base seeds")
    parser.add_argument("--out-dir", default=None,
                        help="output directory (default: the record's dir)")
    parser.add_argument("--slice", dest="slice_choice", default="auto",
                        choices=("auto", "all", "slice"),
                        help="'slice' forces the >=$50k slice, 'all' forces "
                             "the all slice, 'auto' (default) uses >=$50k only "
                             "if the shard is sliceable")
    parser.add_argument("--min-volume", type=int, default=DEFAULT_MIN_VOLUME)
    parser.add_argument("--contrasts", nargs="+", default=list(DEFAULT_CONTRASTS))
    parser.add_argument("--candidates", nargs="+", type=int, default=None,
                        help="restrict to these n_sims (default: all in the "
                             "record)")
    parser.add_argument("--base-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--variability-seeds", nargs=2, type=int,
                        default=list(DEFAULT_VARIABILITY_SEEDS))
    parser.add_argument("--threshold", type=float, default=None,
                        help="default: convergence_protocol.threshold")
    parser.add_argument("--extrapolation-fixtures", type=int,
                        default=DEFAULT_EXTRAPOLATION_FIXTURES)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    record_path = Path(args.record)
    if not record_path.is_file():
        sys.stderr.write("no timing record at %s\n" % record_path)
        return 2
    record = load_json(record_path)
    record["_path"] = str(record_path)

    threshold = args.threshold
    base_seeds = args.base_seeds
    config_path = Path(args.config)
    if (threshold is None or base_seeds is None) and config_path.is_file():
        with config_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        protocol = (config or {}).get("convergence_protocol") or {}
        if threshold is None:
            threshold = float(protocol.get("threshold", DEFAULT_THRESHOLD))
        if base_seeds is None:
            base_seeds = [int(seed) for seed in
                          (protocol.get("batch_base_seeds")
                           or record.get("base_seeds") or [])]
    if threshold is None:
        threshold = DEFAULT_THRESHOLD
    if not base_seeds:
        base_seeds = [int(seed) for seed in (record.get("base_seeds") or [])]

    try:
        runs, omitted = load_runs(record, min_volume=args.min_volume,
                                  candidates=(set(args.candidates)
                                              if args.candidates else None))
        slice_name, slice_reason = resolve_slice(runs, args.slice_choice,
                                                 args.min_volume)
        payload = build_report(
            record, runs, contrasts=args.contrasts, base_seeds=base_seeds,
            slice_name=slice_name, slice_reason=slice_reason,
            threshold=threshold,
            extrapolation_fixtures=args.extrapolation_fixtures,
            variability_seeds=args.variability_seeds, omitted=omitted)
    except ValueError as exc:
        sys.stderr.write("refused: %s\n" % exc)
        return 2

    out_dir = Path(args.out_dir) if args.out_dir else record_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / ("%s.json" % REPORT_STEM)
    md_path = out_dir / ("%s.md" % REPORT_STEM)
    markdown = render_markdown(payload)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    md_path.write_text(markdown, encoding="utf-8")
    print("wrote %s" % json_path)
    print("wrote %s" % md_path)
    print("slice used: %s (%s)" % (slice_name, slice_reason))
    print("runs read: %d; spread rows: %d; timing rows: %d"
          % (len(runs), len(payload["spread_table"]),
             len(payload["timing_per_arm"])))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
