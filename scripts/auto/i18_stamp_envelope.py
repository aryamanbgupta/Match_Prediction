#!/usr/bin/env python3
"""I18 Gate A join: stamp cricsheet_id onto the HALF-MIGRATED golden envelope.

``scripts/patch_envelope_cricsheet_ids.py`` assumes a fully legacy (pre-I15)
envelope whose every ``match_id`` is a raw display id. That precondition does
not hold for the 124-row golden envelope after the 2026-07-30 extension:

  * the 55 original golden rows are legacy — ``match_id`` is a raw display
    id and ``cricsheet_id`` is null;
  * the 69 extension rows are already migrated — ``match_id`` IS the bare
    cricsheet id and ``cricsheet_id`` is already populated.

Applied to that mix, the shared tool reports the 69 already-migrated rows as
UNMATCHED (their ``match_id`` is not a raw display id) and fails closed. That
is correct behaviour for the tool; it is simply the wrong tool for this input.

This helper handles the mix WITHOUT modifying the shared tool, and applies a
strictly stronger check than it does:

  * already-stamped rows: VERIFY the existing cricsheet_id is a real stem in
    the eval-set directory, then keep it;
  * legacy rows: derive the id exactly like the shared tool, by importing its
    ``raw_display_id`` and reusing its ``stems_by_display`` construction, so
    the derivation cannot drift from the shared implementation;
  * fail closed on any unmatched, ambiguous, or duplicate id;
  * FINAL GATE: the stamped ids must be unique and EXACTLY equal, as a set,
    to the file stems in the eval-set directory.

No output file is written unless every check passes.

Usage:
    uv run python scripts/auto/i18_stamp_envelope.py \
        --envelope models/auto/i18/golden_envelope.json \
        --test-dir data/golden/polymarket_test \
        --out models/auto/i18/golden_envelope_cricsheet.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

# Reuse the shared tool's derivation so the two cannot drift apart.
from patch_envelope_cricsheet_ids import raw_display_id  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--envelope", type=Path, required=True)
    ap.add_argument("--test-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    # Identical construction to patch_envelope_cricsheet_ids.main().
    stems_by_display: dict[str, list[str]] = defaultdict(list)
    all_stems: set[str] = set()
    for path in sorted(args.test_dir.glob("*.json")):
        info = json.loads(path.read_text()).get("info", {})
        stems_by_display[raw_display_id(info)].append(path.stem)
        all_stems.add(path.stem)
    print(f"eval-set stems in {args.test_dir}: {len(all_stems)}")

    envelope = json.loads(args.envelope.read_text())
    matches = envelope["matches"]
    print(f"envelope entries: {len(matches)}")

    kept, stamped = 0, 0
    unmatched: list[str] = []
    ambiguous: list[tuple[str, list[str]]] = []
    bad_existing: list[tuple[str, str]] = []

    for entry in matches:
        existing = entry.get("cricsheet_id")
        if existing:
            existing = str(existing)
            if existing not in all_stems:
                bad_existing.append((str(entry["match_id"]), existing))
                continue
            entry["cricsheet_id"] = existing
            kept += 1
            continue
        stems = stems_by_display.get(entry["match_id"], [])
        if len(stems) == 1:
            entry["cricsheet_id"] = stems[0]
            stamped += 1
        elif not stems:
            unmatched.append(str(entry["match_id"]))
        else:
            ambiguous.append((str(entry["match_id"]), stems))

    print(f"  already-stamped and verified: {kept}")
    print(f"  legacy rows newly stamped   : {stamped}")

    failed = False
    for mid, cid in bad_existing:
        print(f"BAD_EXISTING_ID: {mid} -> {cid} is not a stem in {args.test_dir}")
        failed = True
    for mid in unmatched:
        print(f"UNMATCHED: {mid}")
        failed = True
    for mid, stems in ambiguous:
        print(f"AMBIGUOUS: {mid} -> {stems}")
        failed = True

    ids = [str(e.get("cricsheet_id")) for e in matches if e.get("cricsheet_id")]
    dupes = {i: n for i, n in Counter(ids).items() if n > 1}
    if dupes:
        print(f"DUPLICATE_IDS: {dupes}")
        failed = True

    # FINAL GATE: exact set equality against the eval-set stems.
    id_set = set(ids)
    print(f"stamped ids: {len(ids)} total, {len(id_set)} unique")
    if len(ids) != len(matches):
        print(f"COVERAGE_FAIL: {len(matches) - len(ids)} entries left unstamped")
        failed = True
    missing = all_stems - id_set
    extra = id_set - all_stems
    if missing or extra:
        print(f"SET_MISMATCH: missing_from_envelope={sorted(missing)}")
        print(f"SET_MISMATCH: not_an_eval_stem={sorted(extra)}")
        failed = True
    else:
        print(
            f"FINAL GATE OK: stamped id set == eval-set stem set "
            f"({len(id_set)} ids, exact)"
        )

    if failed:
        print("REFUSING to write output: envelope stamping failed closed")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(envelope, indent=1))
    print(f"stamped cricsheet_id on {len(matches)} entries -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
