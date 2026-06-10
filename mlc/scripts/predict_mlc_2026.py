"""Predict the MLC 2026 slate and write a win-probability report.

Loads the production match-level model + providers once (via predict_fixture's
internals) and scores every fixture in fixtures/mlc_2026/, then renders a
markdown summary table. Per-fixture JSONs are written to predictions/ exactly
as scripts/predict_fixture.py would.

Run build_mlc_fixtures.py first to (re)generate the fixtures.

Usage:
    uv run python mlc/scripts/predict_mlc_2026.py            # Dallas leg (7)
    uv run python mlc/scripts/predict_mlc_2026.py --all      # full league (30)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from predict_fixture import (  # noqa: E402
    MODEL_DIR, load_trackers, compute_features, apply_encoders_and_predict,
    _peek_snapshot_as_of,
)
from player_metadata import PlayerMetadataProvider  # noqa: E402
from stats_provider import StatsProvider  # noqa: E402

FIX_DIR = REPO / "fixtures" / "mlc_2026"
PRED_DIR = REPO / "predictions"
ABBR = {
    "MI New York": "MINY", "Texas Super Kings": "TSK", "Washington Freedom": "WSH",
    "San Francisco Unicorns": "SFU", "Seattle Orcas": "ORCA",
    "Los Angeles Knight Riders": "LAKR",
}
VENUE_SHORT = {
    "Grand Prairie Stadium, Dallas": "Grand Prairie, Dallas",
    "Oakland Coliseum,Oakland": "Oakland Coliseum",
    "Central Broward Regional Park Stadium Turf Ground, Lauderhill": "Broward, Lauderhill",
}
DALLAS_LEG_DATES = {"2026-06-18", "2026-06-19", "2026-06-20", "2026-06-21"}


def predict_one(fixture: dict, provider, metadata, form, h2h, home) -> dict:
    record = compute_features(fixture, provider, metadata, form, h2h, home)
    p1, debug = apply_encoders_and_predict(record, MODEL_DIR)
    p2 = 1.0 - p1
    out = {
        "fixture": {k: v for k, v in fixture.items()
                    if k not in ("team1_lineup", "team2_lineup")},
        "fixture_lineups": {"team1": fixture["team1_lineup"],
                            "team2": fixture["team2_lineup"]},
        "prediction": {fixture["team1"]: p1, fixture["team2"]: p2},
        "diagnostics": {
            "model": str(MODEL_DIR),
            "rehydrate_as_of": fixture["date"],
            "tracker_snapshot_as_of": _peek_snapshot_as_of(),
            "encoder_warnings": debug["encoder_warnings"],
            "is_team1_home": record["is_team1_home"],
            "is_team2_home": record["is_team2_home"],
            "top6_batting_elo_diff": record["top6_batting_elo_diff"],
            "bottom5_bowling_elo_diff": record["bottom5_bowling_elo_diff"],
        },
    }
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    fn = (f"{fixture['date']}_{fixture['team1'].replace(' ', '_')}"
          f"_vs_{fixture['team2'].replace(' ', '_')}.json")
    (PRED_DIR / fn).write_text(json.dumps(out, indent=2))
    return out


def conf(p: float) -> str:
    return "lean" if p < 0.55 else ("solid" if p < 0.65 else "strong")


def render(rows: list[dict], all_league: bool) -> str:
    title = ("full league" if all_league else "opening Grand Prairie (Dallas) leg")
    L = [f"# MLC 2026 — {title}: model win probabilities\n",
         "*Model `xgb_match_v3_m7_production` (raw probs, no Platt). "
         "PRE-XI / PRE-TOSS estimates — projected XIs from the announced 2026 "
         "squads (`mlc_2026_rosters.csv`), max 6 overseas + 5 USA-developed per XI; "
         "provisional. Win % only (no betting odds).*\n",
         "| # | Date | Match | Venue | Win %  (team1 / team2) | Model pick | Conf. | top6 ELO Δ |",
         "|---|------|-------|-------|------------------------|-----------|-------|-----------|"]
    for i, r in enumerate(rows, 1):
        p1, p2 = r["p1"], r["p2"]
        fav = r["t1"] if p1 >= p2 else r["t2"]
        favp = max(p1, p2)
        L.append(
            f"| {i} | {r['date'][5:]} | **{ABBR[r['t1']]}** v {ABBR[r['t2']]} "
            f"| {r['venue']} | {p1*100:4.1f}% / {p2*100:4.1f}% "
            f"| **{ABBR[fav]}** ({favp*100:.1f}%) | {conf(favp)} | {r['elo']:+.0f} |")
    L += ["",
          "team1 = listed-first / nominal home side. \"Conf.\": lean <55%, solid 55–65%, "
          "strong >65%. top6 ELO Δ = team1 − team2 top-6 batting ELO (the model's "
          "dominant feature; + favors team1).\n",
          "## Notes",
          "- Same production pipeline used for IPL, applied unchanged; the model carries "
          "real MLC team ELOs (75 cricsheet matches, 2023–2025) plus per-player "
          "career/ELO features. New overseas signings (Smith, Narine, Russell, Hales, "
          "Rachin, Ferguson, Ngidi, Shanaka, …) bring rich international-T20 ELOs.",
          "- XIs respect the **MLC 6-overseas cap**, which binds hard on the stacked "
          "squads (LAKR has 8 internationals, WSH 9) — only 6 of each can play.",
          "- Re-run any fixture once real XIs/toss are known: edit the lineup arrays in "
          "`fixtures/mlc_2026/<file>.json` and rerun this script."]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="score the full league (default: Dallas leg)")
    args = ap.parse_args()

    fixtures = sorted(FIX_DIR.glob("2026-*.json"))
    if not args.all:
        fixtures = [f for f in fixtures
                    if json.loads(f.read_text())["date"] in DALLAS_LEG_DATES]
    if not fixtures:
        print(f"No fixtures in {FIX_DIR}; run build_mlc_fixtures.py first.")
        return 1

    print(f"Loading providers + trackers (once) for {len(fixtures)} fixtures...")
    provider = StatsProvider(str(REPO / "models"), version="v3")
    metadata = PlayerMetadataProvider(str(REPO / "data" / "all_players_enriched.csv"))
    form, h2h, home = load_trackers()

    rows = []
    for fp in fixtures:
        fx = json.loads(fp.read_text())
        out = predict_one(fx, provider, metadata, form, h2h, home)
        p1 = out["prediction"][fx["team1"]]
        p2 = out["prediction"][fx["team2"]]
        rows.append(dict(date=fx["date"], t1=fx["team1"], t2=fx["team2"],
                         p1=p1, p2=p2,
                         venue=VENUE_SHORT.get(fx["venue"], fx["venue"]),
                         elo=out["diagnostics"]["top6_batting_elo_diff"]))
        warns = out["diagnostics"]["encoder_warnings"]
        flag = f"   [{'; '.join(warns)}]" if warns else ""
        print(f"  {fx['date']}  {ABBR[fx['team1']]:>4} {p1*100:5.1f}% / "
              f"{p2*100:5.1f}% {ABBR[fx['team2']]:<4}{flag}")

    rows.sort(key=lambda r: (r["date"], r["t1"]))
    report = render(rows, args.all)
    out_md = REPO / "reports" / ("mlc_2026_league.md" if args.all
                                 else "mlc_2026_dallas_leg.md")
    out_md.write_text(report)
    print(f"\nWrote report -> {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
