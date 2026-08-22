"""Shared utilities for registered experiment runners.

One implementation of the hygiene primitives every registered runner needs:
sealed-path refusal, checksums, match clustering, log loss, the bootstrap
estimators, and provenance blocks. Runners import (and may re-export) these
so focused tests pin a single implementation instead of one drifting copy
per script.

The two seed-aware bootstrap estimators answer DIFFERENT questions and must
never share a report label:

* ``seed_mean_match_cluster_ci`` — uncertainty of the ACROSS-SEED MEAN
  (resamples the seed set with replacement and averages). Report as
  "seed-mean+match bootstrap 95% CI".
* ``seed_draw_match_cluster_ci`` — uncertainty of a SINGLE fitted seed
  (draws one seed per replicate; systematically wider). Report as
  "seed-draw+match bootstrap 95% CI".
"""
from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

# Anchor all provenance hashing at the repo root so a wrong-CWD run cannot
# silently stamp the wrong file (or omit one that "does not exist").
REPO_ROOT = Path(__file__).resolve().parents[1]


def reject_sealed(path) -> None:
    """Refuse any path that touches the sealed golden or forward holdouts."""
    lowered = str(path).lower()
    if "golden" in lowered or "forward_holdout" in lowered:
        raise SystemExit(f"refusing sealed path: {path}")


def resolve_input(path) -> Path:
    """Resolve a (possibly repo-relative) input path against the repo root."""
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def sha256(path) -> str:
    digest = hashlib.sha256()
    with resolve_input(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def match_ids(innings_ids: np.ndarray) -> np.ndarray:
    """Strip the leading ``<innings>_`` prefix from innings identifiers."""
    return np.asarray([str(value).split("_", 1)[-1]
                       for value in innings_ids], dtype=str)


def row_log_loss(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(probs[np.arange(len(y)), y], 1e-15, 1.0))


def match_cluster_ci(values: np.ndarray, clusters: np.ndarray, reps: int,
                     seed: int) -> list[float]:
    """Ball-weighted mean, resampling complete matches. [lo95, hi95]."""
    _, inverse = np.unique(clusters, return_inverse=True)
    n_clusters = int(inverse.max()) + 1
    sums = np.bincount(inverse, weights=values, minlength=n_clusters)
    counts = np.bincount(inverse, minlength=n_clusters)
    rng = np.random.default_rng(seed)
    estimates = np.empty(reps, dtype=np.float64)
    for index in range(reps):
        sampled = rng.integers(0, n_clusters, size=n_clusters)
        estimates[index] = sums[sampled].sum() / counts[sampled].sum()
    return [float(np.percentile(estimates, 2.5)),
            float(np.percentile(estimates, 97.5))]


def seed_mean_match_cluster_ci(deltas: np.ndarray, clusters: np.ndarray,
                               reps: int, seed: int) -> list[float]:
    """CI of the ACROSS-SEED MEAN: two-level bootstrap over training seeds
    (resampled with replacement, then averaged) and complete matches.

    ``deltas`` has shape ``(n_seeds, n_rows)``. Seed indices are paired
    across compared arms before entering this function, and all rows for a
    sampled match remain together. The estimator is ball-weighted.
    """
    deltas = np.atleast_2d(deltas)
    if deltas.shape[1] != len(clusters):
        raise ValueError("delta rows and clusters are not aligned")
    _, inverse = np.unique(clusters, return_inverse=True)
    n_clusters = int(inverse.max()) + 1
    sums = np.stack([
        np.bincount(inverse, weights=values, minlength=n_clusters)
        for values in deltas
    ])
    counts = np.bincount(inverse, minlength=n_clusters)
    rng = np.random.default_rng(seed)
    estimates = np.empty(reps, dtype=np.float64)
    n_seeds = deltas.shape[0]
    for index in range(reps):
        sampled_seeds = rng.integers(0, n_seeds, size=n_seeds)
        sampled_clusters = rng.integers(0, n_clusters, size=n_clusters)
        numerator = sums[sampled_seeds][:, sampled_clusters].sum()
        denominator = n_seeds * counts[sampled_clusters].sum()
        estimates[index] = numerator / denominator
    return [float(np.percentile(estimates, 2.5)),
            float(np.percentile(estimates, 97.5))]


def seed_draw_match_cluster_ci(values: np.ndarray, clusters: np.ndarray,
                               reps: int, seed: int) -> list[float]:
    """CI including SINGLE-SEED variance: each replicate draws one fitted
    seed and resamples matches. Systematically wider than the seed-mean
    estimator; do not present the two under one label."""
    unique, inverse = np.unique(clusters, return_inverse=True)
    sums = np.stack([
        np.bincount(inverse, weights=row) for row in values
    ])
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    stats = np.empty(reps)
    for index in range(reps):
        seed_index = int(rng.integers(0, values.shape[0]))
        sampled = rng.integers(0, len(unique), size=len(unique))
        stats[index] = sums[seed_index, sampled].sum() / counts[sampled].sum()
    return [float(np.percentile(stats, 2.5)),
            float(np.percentile(stats, 97.5))]


def environment_block(*modules) -> dict:
    """Python/platform plus ``__name__: __version__`` for each module."""
    block = {"python": sys.version.split()[0],
             "platform": platform.platform()}
    for module in modules:
        name = module.__name__.split(".")[0]
        block[name] = str(module.__version__)
    return block


def git_provenance() -> dict:
    status = subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True,
        text=True, cwd=REPO_ROOT).stdout
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True,
        text=True, cwd=REPO_ROOT).stdout.strip()
    return {"git_commit": commit,
            "git_dirty_at_aggregation": bool(status.strip())}
