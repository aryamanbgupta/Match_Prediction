# I14 — Physical venue registry: build plan

**Status: PASSES 1–3 DONE (2026-08-01, interactive track).**
`config/identity/venue_registry_v0.csv` has coordinates + altitude for all
92 canonical venues (Wikipedia GeoData with per-row page provenance; 5
grounds flagged `manual:approx_known`), and
`config/identity/venue_climate_normals_v0.csv` has 92 × 12 monthly normals
(Open-Meteo archive 2015–2024, sanity-verified; wind column is **km/h** —
mislabeled m/s until the 2026-08-01 fix). Pass 3 (commit `22020d1` + the
2026-08-01 repair of the shifted Taunton/Gabba rows) adds boundary geometry
for the top 30 registry venues by volume: 23 directional, 7 honest gaps.
Builders: `scripts/build_venue_registry.py`, `scripts/build_venue_climate.py`
— both resumable.

**Pass 3.5 DONE (2026-08-01).** 27 rows added — every missing venue with
≥40 corpus matches (the registry universe was the alias map only; the true
top-volume grounds had single stable spellings): Mirpur 297, Dubai 262,
Sharjah 116, Al Amerat 108, MCG 95, Adelaide 95, Sylhet 92, SCG 89, and 19
more. Registry now 119 rows, 118 with coords/altitude/climate (Udayana,
Bali: no wiki page or GeoData — honest gap, manual/satellite pass needed).
Boundary geometry for the top-10 additions: 7 directional (incl. MCG
83.4/86.2 and Mirpur 77/61 — the I5 three-run linkage now has its
geometry), Chattogram weak shared-range, Sylhet + Udayana honest gaps.
Same-ground cross-refs: Zayed=Sheikh Zayed, Feroz Shah Kotla=Arun Jaitley,
the Warner Park spelling pair, and the two Wanderers rows — dedupe on any
physical join.

**Identity findings for a future I7 pass (excluded from the registry):**
bare `County Ground` is 196 corpus matches spanning SIX physical grounds
(Northampton 43 / Chelmsford 40 / Bristol 39 / Hove 36 / Derby 31 /
Taunton 7) under one venue string, and bare `National Stadium` is Karachi
38 + Hamilton 2. The ball frame's venue identity smears these; a
venue+city split is the candidate fix.

**Remaining: the modeling integration below** (physical features + missing
indicators, grouped-by-venue holdout gate), plus the 20–39-match tail
(38 venues) if coverage wants widening first.

## Registry schema (v0)

One row per canonical venue (I7 identity is the join key — prerequisite
already landed):

| column | notes |
|---|---|
| `latitude`, `longitude`, `altitude_m` | point facts, one-time sourcing |
| `boundary_straight_m_min/max`, `boundary_square_m_min/max` | **ranges, not constants** — ropes move per match |
| `dimensions_source`, `dimensions_obs_date` | provenance is mandatory (IDEAS requirement); a dimension without a date is not usable |
| `climate_normals_source` | monthly normals live in a separate long-format file, `venue_climate_normals_v0.csv` (venue × month × {temp, dewpoint, wind, precip}) |

Match-day weather stays out of the registry by design — it's a time-indexed
feature with as-of discipline, not climatology.

## Sourcing plan (three passes, by match volume)

1. **Coordinates + altitude (all 92, cheap, high-trust).** Lat/lon from
   public ground pages; altitude from lat/lon via open elevation data.
   Fully automatable with a web pass; near-zero licensing risk for facts.
2. **Climate normals (all 92, free).** Open-Meteo climate API (or NOAA
   normals) at the venue coordinates — monthly temp/dew/wind/precip. No
   paid API needed.
3. **Boundary geometry (top ~30 venues first, the hard 20%).** ESPNcricinfo
   ground profiles, MCC/venue sites, broadcast measurements — patchy,
   inconsistent conventions (some quote rope-to-pitch, some fence). Every
   entry carries source + observation date; unknown stays empty with a
   missing-value indicator rather than a guessed number. The long tail may
   never fill — the model must tolerate that (indicator columns).

## Modeling integration (unchanged from IDEAS, condensed)

Concatenate normalized physical features + missing indicators alongside the
learned venue embedding (never encode continuous values as IDs). Gate on
**grouped-by-venue holdout** so high-volume grounds can't mask sparse-venue
regressions; targets: unseen/low-history venue LL, six/four rates, innings
totals, chase calibration. The temporal-venue hypothesis (era-indexed venue
state, change points tied to dated registry evidence) is a follow-up
experiment on top of this registry, not part of v0.

Also unlocks: the deferred three-run modeling question (I5) explicitly waits
on ground dimensions (MCG 1.06% threes vs Mirpur 0.25%).

## Decisions required from you

- **V1 — Sourcing effort tier**: coordinates+altitude+climate only
  (automatable, ~a session, already useful for altitude/dew effects) vs
  including boundary geometry for the top-30 (manual-ish, the part with
  real six-rate signal but real drudgery). Recommend both, top-30 only for
  geometry.
- **V2 — Source policy**: is ESPNcricinfo/Wikipedia-grade sourcing with
  per-row provenance acceptable for research use? (No paid API is needed;
  if you want a paid weather/geo source anyway, say which.)
- **V3 — Who fills it**: I can do passes 1–2 interactively with web access
  now; pass 3 is also doable but slower — or it becomes a supervised
  checklist you spot-check.
