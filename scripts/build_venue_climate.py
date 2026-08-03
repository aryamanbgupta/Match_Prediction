#!/usr/bin/env python
"""I14 pass 2: monthly climate normals per canonical venue.

For every venue with coordinates in the registry, pulls daily history
2015-01-01..2024-12-31 from the Open-Meteo archive API and aggregates to
monthly normals (mean temperature, mean daily precipitation, mean daily
max windspeed in km/h — Open-Meteo's default unit — and mean relative
humidity). Writes long-format
config/identity/venue_climate_normals_v0.csv, flushing after every venue
so the run is resumable — venues already present in the output are skipped.

Normals are climatology (safe as static features); match-day weather is a
separate, time-indexed concern (docs/I14_VENUE_REGISTRY_PLAN.md).

Usage:
    uv run python scripts/build_venue_climate.py [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import time
import urllib.parse
from collections import defaultdict
from pathlib import Path

from build_venue_registry import _get

ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"
PERIOD = ("2015-01-01", "2024-12-31")
DAILY_VARS = ("temperature_2m_mean", "precipitation_sum",
              "windspeed_10m_max", "relative_humidity_2m_mean")
FIELDS = ("canonical_venue", "month", "temp_c_mean", "precip_mm_daily_mean",
          "windmax_kmh_mean", "rh_pct_mean", "source", "period")
REQUEST_GAP_S = 1.0


def monthly_normals(lat: float, lon: float) -> dict[int, dict[str, float]]:
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": PERIOD[0], "end_date": PERIOD[1],
        "daily": ",".join(DAILY_VARS), "timezone": "UTC",
    }
    data = _get(ARCHIVE_API + "?" + urllib.parse.urlencode(params))["daily"]
    acc: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    for i, day in enumerate(data["time"]):
        month = int(day[5:7])
        for var in DAILY_VARS:
            val = data.get(var, [None])[i] if data.get(var) else None
            if val is not None:
                acc[month][var].append(val)
    return {
        m: {var: sum(vals) / len(vals)
            for var, vals in bymonth.items() if vals}
        for m, bymonth in acc.items()
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", type=Path,
                    default=Path("config/identity/venue_registry_v0.csv"))
    ap.add_argument("--out", type=Path,
                    default=Path("config/identity/venue_climate_normals_v0.csv"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    venues = [r for r in csv.DictReader(args.registry.open())
              if r.get("latitude")]
    done: set[str] = set()
    out_rows: list[dict] = []
    if args.out.exists():
        out_rows = list(csv.DictReader(args.out.open()))
        done = {r["canonical_venue"] for r in out_rows}

    todo = [v for v in venues if v["canonical_venue"] not in done]
    if args.limit is not None:
        todo = todo[: args.limit]

    for i, venue in enumerate(todo):
        name = venue["canonical_venue"]
        try:
            normals = monthly_normals(float(venue["latitude"]),
                                      float(venue["longitude"]))
        except Exception as err:  # keep the sweep alive; rerun picks it up
            print(f"FAIL  {name}: {err}")
            time.sleep(REQUEST_GAP_S)
            continue
        for month in sorted(normals):
            n = normals[month]
            out_rows.append({
                "canonical_venue": name, "month": month,
                "temp_c_mean": f"{n.get('temperature_2m_mean', float('nan')):.2f}",
                "precip_mm_daily_mean": f"{n.get('precipitation_sum', float('nan')):.2f}",
                "windmax_kmh_mean": f"{n.get('windspeed_10m_max', float('nan')):.2f}",
                "rh_pct_mean": f"{n.get('relative_humidity_2m_mean', float('nan')):.1f}",
                "source": "open-meteo-archive",
                "period": f"{PERIOD[0]}..{PERIOD[1]}",
            })
        with args.out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"OK    [{i + 1}/{len(todo)}] {name}")
        time.sleep(REQUEST_GAP_S)

    print(f"\n{len({r['canonical_venue'] for r in out_rows})} venues in "
          f"{args.out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
