#!/usr/bin/env python3
# manifest-exempt: diagnostic rebuild for an unpromoted sidecar
"""Rebuild the ignored B18 extras sidecar from committed fit evidence.

The original fitted parquet and promoted i7 booster are not present in every
clone, but the landed fit log records the exact rounded rates and integer event
counts. Those are sufficient to reconstruct the runtime contract without
inventing or refitting any value.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

EVIDENCE = Path("research/handoff/B18/raw/fit_ruling.txt")
DEFAULT_OUT = Path("models/auto/b18/extras_graft_v1.json")
POPULATION_HASH = "4ac7accc089e4cc3d8df761a117cbb4c"
WIDE_COUNTS = {1: 4270, 2: 199, 3: 52, 4: 5, 5: 160}
NO_BALL_COUNTS = {1: 509, 2: 39}


def run_law(counts: dict[int, int]) -> dict:
    counts = Counter(counts)
    support = sorted(counts)
    total = sum(counts.values())
    probabilities = [counts[value] / total for value in support]
    probabilities[-1] = 1.0 - sum(probabilities[:-1])
    return {
        "support": support,
        "probs": probabilities,
        "mean": sum(value * probability
                    for value, probability in zip(support, probabilities)),
    }


def payload() -> dict:
    return {
        "version": "extras_graft_v1",
        "p_wide": 0.037702,
        "p_no_ball": 0.004409,
        "wide_runs": run_law(WIDE_COUNTS),
        "no_ball_runs": run_law(NO_BALL_COUNTS),
        "reconstructed_from": {
            "evidence": str(EVIDENCE),
            "population_hash": POPULATION_HASH,
            "wide_counts": {str(key): value
                            for key, value in WIDE_COUNTS.items()},
            "no_ball_counts": {str(key): value
                               for key, value in NO_BALL_COUNTS.items()},
            "note": (
                "Exact landed B18 rates/counts reconstructed because the "
                "ignored sidecar was absent locally."
            ),
        },
    }


def validate_evidence() -> None:
    text = EVIDENCE.read_text()
    required = [
        f"population hash (sorted match ids + row count): {POPULATION_HASH}",
        "p_wide    = 0.037701542",
        "p_no_ball = 0.004408972",
        "counts 1x4270  2x199  3x52  4x5  5x160",
        "counts 1x509  2x39",
    ]
    missing = [value for value in required if value not in text]
    if missing:
        raise SystemExit(f"committed B18 evidence contract changed: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--check", action="store_true",
        help="compare an existing sidecar semantically without rewriting it")
    args = parser.parse_args()
    validate_evidence()
    expected = payload()
    if args.check:
        actual = json.loads(args.out.read_text())
        if actual != expected:
            raise SystemExit(f"sidecar differs from committed evidence: {args.out}")
        print(f"PASS: {args.out} matches {EVIDENCE}")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(expected, indent=2) + "\n")
    print(f"wrote {args.out} from {EVIDENCE}")


if __name__ == "__main__":
    main()
