"""Stamp cricsheet_id onto a legacy (pre-I15) sim-eval envelope.

Legacy envelopes key matches by the synthetic display id built from the RAW
Cricsheet venue string (see eval_statistics.py). Models trained on
identity-declaring frames key their predictions by cricsheet_id and by the
CANONICAL display id, so joins against a legacy envelope silently drop any
fixture whose venue was renamed by canonicalization. This tool derives each
envelope entry's Cricsheet file stem from the eval-set JSONs and writes a
patched copy of the envelope with ``cricsheet_id`` on every match entry, so
downstream joins are exact.

Fails closed if any envelope entry cannot be matched to exactly one stem
(same-day doubleheaders sharing a display id are ambiguous by construction
and must be resolved by hand).

Usage:
    uv run python scripts/patch_envelope_cricsheet_ids.py \
        --envelope eval_out/phase5_hier/hier_all_20260425_165622.json \
        --test-dir data/polymarket_test \
        --out eval_out/i17/hier_all_cricsheet.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def raw_display_id(info: dict) -> str:
    """The pre-I15 synthetic id: raw venue string, no canonicalization."""
    teams = info.get("teams") or []
    dates = info.get("dates") or []
    if len(teams) != 2 or not dates:
        raise ValueError("match info must contain two teams and a date")
    venue = str(info.get("venue") or "").strip()
    return f"{dates[0]}_{teams[0]}_{teams[1]}_{venue}".replace(" ", "_")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--envelope", type=Path, required=True)
    ap.add_argument("--test-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    stems_by_display: dict[str, list[str]] = defaultdict(list)
    for path in sorted(args.test_dir.glob("*.json")):
        info = json.loads(path.read_text()).get("info", {})
        stems_by_display[raw_display_id(info)].append(path.stem)

    envelope = json.loads(args.envelope.read_text())
    matches = envelope["matches"]
    unmatched, ambiguous = [], []
    for entry in matches:
        stems = stems_by_display.get(entry["match_id"], [])
        if len(stems) == 1:
            entry["cricsheet_id"] = stems[0]
        elif not stems:
            unmatched.append(entry["match_id"])
        else:
            ambiguous.append((entry["match_id"], stems))

    if unmatched or ambiguous:
        for mid in unmatched:
            print(f"UNMATCHED: {mid}")
        for mid, stems in ambiguous:
            print(f"AMBIGUOUS: {mid} -> {stems}")
        raise SystemExit(
            f"{len(unmatched)} unmatched / {len(ambiguous)} ambiguous "
            "envelope entries; refusing to write a partial patch"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(envelope, indent=1))
    print(f"stamped cricsheet_id on {len(matches)} entries -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
