#!/usr/bin/env python3
"""A6 — historical weather (dew proxy) feature builder.

GET-only pulls from open-meteo (geocoding + historical archive). Joins evening
humidity / dew-point onto the EXISTING materialized match parquet (no career
re-materialization needed — weather is a per-(venue,date) scalar). All raw
pulls cached under data/external/weather/ so re-runs are cheap and offline.

Features added (numeric → auto-picked up by xgboost_match_v1.py):
  wx_evening_humidity  mean RH%      18:00-22:00 local, match date
  wx_evening_dewpoint  mean dewpt°C  same window
  wx_dew_adv_team1     (RH - train_mean_RH) * (1 - 2*team1_batting_first)
                       oriented dew advantage toward team1 (chaser gains on dew)

Temporal note: weather actuals on the match date are an exogenous environmental
condition (forecastable pre-match, not derived from match play) — same class as
the M6 month/day/dew condition features. No match-outcome info is used.
"""
import json, os, time, urllib.request, urllib.parse, socket, re, hashlib
from collections import defaultdict
import pandas as pd
import numpy as np

socket.setdefaulttimeout(30)
DATA_DIR = "data/xgb_match_data_v2_clean"
OUT_DIR = "data/auto/a6"
WX_DIR = "data/external/weather"
GEO_CACHE = os.path.join(WX_DIR, "geocode.json")
ARCH_DIR = os.path.join(WX_DIR, "archive")
EVENING_HOURS = [18, 19, 20, 21, 22]

os.makedirs(ARCH_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


def _get(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url) as r:
                return json.load(r)
        except Exception as e:
            code = getattr(e, "code", None)
            if i == tries - 1:
                return {"__error__": f"{type(e).__name__}:{code}"}
            time.sleep(1.5 * (i + 1))
    return {"__error__": "unreachable"}


def _sanitize(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:80]


# ---------- geocoding ----------
def _geo_candidates(venue):
    """Ordered query strings to try for one venue name."""
    v = venue.strip()
    cands = []
    if "," in v:
        cands.append(v.rsplit(",", 1)[1].strip())      # city after last comma
    # strip common ground words for a cleaner name query
    cleaned = re.sub(r"\b(Cricket|Stadium|Ground|Oval|Park|International|Field|"
                     r"Complex|Academy|Turf|Regional|Sports|Club|No\.?\s*\d+|"
                     r"\(.*?\))\b", " ", v)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,")
    if cleaned and cleaned not in cands:
        cands.append(cleaned)
    if v not in cands:
        cands.append(v)
    if "," in v:                                         # first token fallback
        first = v.split(",")[0].strip()
        if first not in cands:
            cands.append(first)
    return [c for c in cands if len(c) >= 3]


def geocode_all(venues):
    cache = json.load(open(GEO_CACHE)) if os.path.exists(GEO_CACHE) else {}
    changed = False
    for v in venues:
        if v in cache:
            continue
        hit = None
        for q in _geo_candidates(v):
            url = ("https://geocoding-api.open-meteo.com/v1/search?name="
                   + urllib.parse.quote(q) + "&count=1&language=en&format=json")
            d = _get(url)
            res = d.get("results") if isinstance(d, dict) else None
            if res:
                r0 = res[0]
                hit = {"lat": r0["latitude"], "lon": r0["longitude"],
                       "query": q, "name": r0.get("name"),
                       "country": r0.get("country")}
                break
            time.sleep(0.05)
        cache[v] = hit  # may be None (uncodable)
        changed = True
        if changed and len(cache) % 25 == 0:
            json.dump(cache, open(GEO_CACHE, "w"))
    json.dump(cache, open(GEO_CACHE, "w"))
    return cache


# ---------- archive ----------
def fetch_archive(lat, lon, start, end):
    """Hourly RH/temp/dewpoint over [start,end] (YYYY-MM-DD), local tz. Cached."""
    key = _sanitize(f"{lat:.3f}_{lon:.3f}_{start}_{end}")
    fp = os.path.join(ARCH_DIR, key + ".json")
    if os.path.exists(fp):
        return json.load(open(fp))
    url = ("https://archive-api.open-meteo.com/v1/archive?"
           f"latitude={lat:.4f}&longitude={lon:.4f}"
           f"&start_date={start}&end_date={end}"
           "&hourly=relative_humidity_2m,temperature_2m,dew_point_2m"
           "&timezone=auto")
    d = _get(url)
    if "__error__" not in d:
        json.dump(d, open(fp, "w"))
    return d


def evening_by_date(arch):
    """{date -> (mean_rh, mean_dewpt)} from hourly archive payload."""
    h = arch.get("hourly", {}) if isinstance(arch, dict) else {}
    times = h.get("time", [])
    rh = h.get("relative_humidity_2m", [])
    dp = h.get("dew_point_2m", [])
    acc = defaultdict(lambda: {"rh": [], "dp": []})
    for i, t in enumerate(times):
        # t like '2024-04-01T18:00'
        date, hh = t[:10], int(t[11:13])
        if hh in EVENING_HOURS:
            if i < len(rh) and rh[i] is not None:
                acc[date]["rh"].append(rh[i])
            if i < len(dp) and dp[i] is not None:
                acc[date]["dp"].append(dp[i])
    out = {}
    for date, a in acc.items():
        if a["rh"]:
            out[date] = (float(np.mean(a["rh"])),
                         float(np.mean(a["dp"])) if a["dp"] else np.nan)
    return out


def main():
    splits = {s: pd.read_parquet(f"{DATA_DIR}/{s}.parquet")
              for s in ["train", "validation", "test"]}
    all_df = pd.concat([splits[s][["venue", "match_date"]] for s in splits])
    venues = sorted(all_df["venue"].dropna().unique())
    print(f"[geo] geocoding {len(venues)} venues ...")
    geo = geocode_all(venues)
    n_geo = sum(1 for v in venues if geo.get(v))
    print(f"[geo] coverage {n_geo}/{len(venues)} = {n_geo/len(venues):.1%}")

    # per-venue date ranges (union of years -> one archive call per venue span)
    vdates = all_df.dropna().groupby("venue")["match_date"].agg(["min", "max"])
    lookup = {}          # (venue, date) -> (rh, dewpt)
    covered_venues = 0
    for vi, v in enumerate(venues):
        g = geo.get(v)
        if not g:
            continue
        start = str(vdates.loc[v, "min"])[:10]
        end = str(vdates.loc[v, "max"])[:10]
        arch = fetch_archive(g["lat"], g["lon"], start, end)
        if "__error__" in arch:
            print(f"[arch] FAIL {v[:40]} {arch['__error__']}")
            continue
        ebd = evening_by_date(arch)
        if ebd:
            covered_venues += 1
        for date, (rh, dp) in ebd.items():
            lookup[(v, date)] = (rh, dp)
        time.sleep(0.05)
        if (vi + 1) % 50 == 0:
            print(f"[arch] {vi+1}/{len(venues)} venues fetched, "
                  f"{len(lookup)} (venue,date) rows")
    print(f"[arch] evening weather for {covered_venues} venues, "
          f"{len(lookup)} (venue,date) rows")

    # train-only centering constant for the oriented feature
    def rh_of(row):
        return lookup.get((row["venue"], str(row["match_date"])[:10]),
                          (np.nan, np.nan))[0]
    tr = splits["train"]
    tr_rh = tr.apply(rh_of, axis=1)
    train_mean_rh = float(np.nanmean(tr_rh))
    print(f"[feat] train-mean evening RH = {train_mean_rh:.2f}%")

    meta = {"train_mean_rh": train_mean_rh,
            "geo_coverage": f"{n_geo}/{len(venues)}",
            "wx_rows": len(lookup)}
    for s, df in splits.items():
        rh = df.apply(lambda r: lookup.get(
            (r["venue"], str(r["match_date"])[:10]), (np.nan, np.nan))[0], axis=1)
        dp = df.apply(lambda r: lookup.get(
            (r["venue"], str(r["match_date"])[:10]), (np.nan, np.nan))[1], axis=1)
        df = df.copy()
        df["wx_evening_humidity"] = rh.astype(float)
        df["wx_evening_dewpoint"] = dp.astype(float)
        # oriented: dew helps the chaser; team1 chases when team1_batting_first==0
        sign = 1.0 - 2.0 * df["team1_batting_first"].astype(float)
        df["wx_dew_adv_team1"] = (rh.astype(float) - train_mean_rh) * sign
        df.to_parquet(f"{OUT_DIR}/{s}.parquet")
        nn = int(rh.notna().sum())
        print(f"[out] {s}: {len(df)} rows, weather present {nn} "
              f"({nn/len(df):.1%})")
    json.dump(meta, open(f"{OUT_DIR}/_wx_meta.json", "w"), indent=2)
    # also copy golden_test parquet through UNCHANGED is NOT needed (never used)
    print("[done]")


if __name__ == "__main__":
    main()
