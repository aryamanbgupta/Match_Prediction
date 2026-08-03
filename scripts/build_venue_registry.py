#!/usr/bin/env python
"""Fill physical columns of the I14 venue registry (pass 1: coords + altitude).

Reads config/identity/venue_registry_v0.csv, resolves latitude/longitude for
each canonical venue via the Wikipedia search API (top hit carrying page
coordinates), then altitude for all resolved rows in one batched Open-Meteo
elevation call. Rows that already have a latitude are left untouched, so the
script is resumable and manual corrections are never overwritten.

Provenance: `coord_source` records the Wikipedia page title the coordinates
came from; altitude is always `open-meteo-elevation` at those coordinates.
Boundary geometry and climate normals are later passes (see
docs/I14_VENUE_REGISTRY_PLAN.md).

Usage:
    uv run python scripts/build_venue_registry.py \
        [--registry config/identity/venue_registry_v0.csv] [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

WIKI_API = "https://en.wikipedia.org/w/api.php"
ELEVATION_API = "https://api.open-meteo.com/v1/elevation"
USER_AGENT = "CricML-venue-registry/0.1 (research; contact: repo owner)"
REQUEST_GAP_S = 1.0
BACKOFF_S = (5, 15, 45, 90)


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt, backoff in enumerate((*BACKOFF_S, None)):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            if backoff is None or err.code not in (429, 500, 502, 503):
                raise
            retry_after = err.headers.get("Retry-After")
            wait = max(backoff, int(retry_after)) if (
                retry_after and retry_after.isdigit()) else backoff
            print(f"      HTTP {err.code}; retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def wiki_coords(query: str) -> tuple[float, float, str] | None:
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": 1,
        "prop": "coordinates",
        "colimit": 1,
        "format": "json",
        "redirects": 1,
    }
    data = _get(WIKI_API + "?" + urllib.parse.urlencode(params))
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        coords = page.get("coordinates")
        if coords:
            return coords[0]["lat"], coords[0]["lon"], page.get("title", "")
    return None


def batched_elevation(coords: list[tuple[float, float]]) -> list[float]:
    lats = ",".join(f"{lat:.5f}" for lat, _ in coords)
    lons = ",".join(f"{lon:.5f}" for _, lon in coords)
    data = _get(f"{ELEVATION_API}?latitude={lats}&longitude={lons}")
    return data["elevation"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", type=Path,
                    default=Path("config/identity/venue_registry_v0.csv"))
    ap.add_argument("--limit", type=int, default=None,
                    help="only resolve the first N unresolved rows")
    args = ap.parse_args()

    rows = list(csv.DictReader(args.registry.open()))
    fieldnames = list(rows[0].keys())
    if "coord_source" not in fieldnames:
        fieldnames.append("coord_source")

    unresolved = [r for r in rows if not r.get("latitude")]
    if args.limit is not None:
        unresolved = unresolved[: args.limit]

    def flush() -> None:
        with args.registry.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in fieldnames})

    hits, misses = [], []
    for row in unresolved:
        name = row["canonical_venue"]
        found = wiki_coords(name)
        if found is None:
            # Bare-name retry helps suffixed canonicals whose page lacks the
            # city ("Eden Gardens, Kolkata" -> "Eden Gardens").
            bare = name.rsplit(",", 1)[0]
            found = wiki_coords(bare) if bare != name else None
        if found is None:
            misses.append(name)
            print(f"MISS  {name}")
        else:
            lat, lon, title = found
            row["latitude"], row["longitude"] = f"{lat:.5f}", f"{lon:.5f}"
            row["coord_source"] = f"wikipedia:{title}"
            hits.append(row)
            print(f"OK    {name}  ->  {title}  ({lat:.4f}, {lon:.4f})")
            flush()
        time.sleep(REQUEST_GAP_S)

    need_alt = [r for r in rows if r.get("latitude") and not r.get("altitude_m")]
    if need_alt:
        elevations = batched_elevation(
            [(float(r["latitude"]), float(r["longitude"])) for r in need_alt])
        for row, elev in zip(need_alt, elevations):
            row["altitude_m"] = f"{elev:.0f}"
    flush()

    print(f"\nresolved {len(hits)} / {len(unresolved)} attempted; "
          f"{len(misses)} misses")
    if misses:
        print("misses (manual pass needed):")
        for name in misses:
            print(f"  - {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
