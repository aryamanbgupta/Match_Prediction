#!/usr/bin/env python
"""I12: extract women's T20Is from the stat-generator cricsheet mirror.

Writes every `info.gender == "female"` match JSON from t20s_json.zip into
data/w_t20s_json/ (raw bytes, cricsheet stem filenames). T20I-only by scope
(league zips are men's competitions; women's leagues need new downloads and
are deliberately out of v1 — docs/I12_WOMENS_TRACK_SCOPING.md). Idempotent:
existing files are overwritten byte-identically.

Usage:
    uv run python scripts/extract_womens_t20s.py \
        [--zip <stat-generator>/data/cricsheet/t20s_json.zip] \
        [--out data/w_t20s_json]
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

DEFAULT_ZIP = Path("/Users/aryamangupta/Projects/stat-generator/data/"
                   "cricsheet/t20s_json.zip")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    ap.add_argument("--out", type=Path, default=Path("data/w_t20s_json"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    written, skipped = 0, 0
    with zipfile.ZipFile(args.zip) as zp:
        for name in zp.namelist():
            if not name.endswith(".json"):
                continue
            raw = zp.read(name)
            try:
                gender = json.loads(raw).get("info", {}).get("gender")
            except json.JSONDecodeError:
                skipped += 1
                continue
            if gender != "female":
                skipped += 1
                continue
            (args.out / Path(name).name).write_bytes(raw)
            written += 1
    print(f"wrote {written} women's T20Is to {args.out} "
          f"(skipped {skipped} non-female/invalid)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
