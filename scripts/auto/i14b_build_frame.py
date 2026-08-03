#!/usr/bin/env python
"""I14b: augment the i7 ball frame with physical venue features.

Joins the I14 registry (`config/identity/venue_registry_v0.csv`) and monthly
climate normals (`config/identity/venue_climate_normals_v0.csv`) onto the i7
ball parquets by canonical venue string (and month-of-match for climate).
Writes an isolated frame `data/xgb_data_i14b/` (`data.version = i14b`);
`data/xgb_data_i7` is never modified. The `.feature_hash` sidecar is copied
with the I7 venue-alias contract fields intact (same alias map, same rows —
this frame only ADDS per-venue-constant columns).

Feature columns (all `vphys_` to avoid collision with the learned
`venue_p*` outcome features; NaN where unknown + explicit known-flags —
XGBoost handles NaN natively, the flags let trees separate "unknown" from
any imputed magnitude):

  vphys_straight_mid_m   midpoint of published straight-boundary range
  vphys_square_mid_m     midpoint of published square-boundary range
  vphys_boundary_known   1 if either boundary figure exists for the venue
  vphys_altitude_m       registry altitude (Open-Meteo elevation)
  vphys_altitude_known   1 if altitude exists
  vphys_temp_c           monthly normal, joined on month(match_date)
  vphys_precip_mm        monthly normal (daily mean, mm)
  vphys_windmax_kmh      monthly normal (daily max wind, km/h)
  vphys_rh_pct           monthly normal (relative humidity)
  vphys_climate_known    1 if the venue has climate normals

Run: uv run python scripts/auto/i14b_build_frame.py
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "data/xgb_data_i7"
DST = REPO / "data/xgb_data_i14b"
REGISTRY = REPO / "config/identity/venue_registry_v0.csv"
CLIMATE = REPO / "config/identity/venue_climate_normals_v0.csv"

VPHYS_COLS = [
    "vphys_straight_mid_m", "vphys_square_mid_m", "vphys_boundary_known",
    "vphys_altitude_m", "vphys_altitude_known",
    "vphys_temp_c", "vphys_precip_mm", "vphys_windmax_kmh", "vphys_rh_pct",
    "vphys_climate_known",
]


def _mid(*vals: str) -> float:
    nums = [float(v) for v in vals if v not in ("", None)]
    return float(np.mean(nums)) if nums else float("nan")


def load_registry() -> dict[str, dict]:
    table: dict[str, dict] = {}
    for r in csv.DictReader(REGISTRY.open()):
        straight = _mid(r["boundary_straight_m_min"], r["boundary_straight_m_max"])
        square = _mid(r["boundary_square_m_min"], r["boundary_square_m_max"])
        alt = float(r["altitude_m"]) if r["altitude_m"] else float("nan")
        table[r["canonical_venue"]] = {
            "vphys_straight_mid_m": straight,
            "vphys_square_mid_m": square,
            "vphys_boundary_known": float(
                not (np.isnan(straight) and np.isnan(square))),
            "vphys_altitude_m": alt,
            "vphys_altitude_known": float(not np.isnan(alt)),
        }
    return table


def load_climate() -> dict[tuple[str, int], dict]:
    table: dict[tuple[str, int], dict] = {}
    for r in csv.DictReader(CLIMATE.open()):
        table[(r["canonical_venue"], int(r["month"]))] = {
            "vphys_temp_c": float(r["temp_c_mean"]),
            "vphys_precip_mm": float(r["precip_mm_daily_mean"]),
            "vphys_windmax_kmh": float(r["windmax_kmh_mean"]),
            "vphys_rh_pct": float(r["rh_pct_mean"]),
        }
    return table


def main() -> int:
    reg = load_registry()
    cli = load_climate()
    DST.mkdir(parents=True, exist_ok=True)

    for split in ("train", "validation", "test"):
        src = SRC / f"cricket_data_i7_{split}.parquet"
        df = pd.read_parquet(src)
        months = pd.to_datetime(df["match_date"]).dt.month

        reg_rows = df["venue"].map(reg)
        for col in ("vphys_straight_mid_m", "vphys_square_mid_m",
                    "vphys_boundary_known", "vphys_altitude_m",
                    "vphys_altitude_known"):
            df[col] = reg_rows.map(
                lambda d, c=col: d[c] if isinstance(d, dict) else float("nan"))
        df["vphys_boundary_known"] = df["vphys_boundary_known"].fillna(0.0)
        df["vphys_altitude_known"] = df["vphys_altitude_known"].fillna(0.0)

        keys = list(zip(df["venue"], months))
        cli_rows = [cli.get(k) for k in keys]
        for col in ("vphys_temp_c", "vphys_precip_mm", "vphys_windmax_kmh",
                    "vphys_rh_pct"):
            df[col] = [d[col] if d else float("nan") for d in cli_rows]
        df["vphys_climate_known"] = [1.0 if d else 0.0 for d in cli_rows]

        out = DST / f"cricket_data_i14b_{split}.parquet"
        df.to_parquet(out, index=False)

        n = len(df)
        print(f"{split}: {n:,} balls | boundary known "
              f"{df['vphys_boundary_known'].mean():.1%} | altitude known "
              f"{df['vphys_altitude_known'].mean():.1%} | climate known "
              f"{df['vphys_climate_known'].mean():.1%} | venues "
              f"{df['venue'].nunique()}")

    sidecar = json.loads((SRC / ".feature_hash").read_text())
    sidecar["version"] = "i14b"
    sidecar["derived_from"] = "i7 (additive vphys_* venue-physical columns; I14 registry v0)"
    sidecar["n_features"] = sidecar.get("n_features")
    (DST / ".feature_hash").write_text(json.dumps(sidecar))
    for extra in SRC.glob(".*"):
        if extra.name != ".feature_hash":
            shutil.copyfile(extra, DST / extra.name)
    print(f"\nsidecar copied with venue-alias contract intact -> {DST}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
