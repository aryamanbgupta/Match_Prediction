"""Score validated MLC 2026 best XIs head-to-head, plus lineup variations.

Holds the user-validated best XIs (5 USA-developed + 6 overseas) for SFU / TSK /
LAKR and runs them through the production match-level model on the two Dallas-leg
head-to-heads where both sides are validated (LAKR v SFU, TSK v SFU). Then runs a
handful of "different players playing" variations and reports the win-% delta.

Does NOT mutate fixtures/mlc_2026/ (those hold the projected-XI slate); it only
borrows each fixture's date/venue and swaps in these XIs.

Usage:
    uv run python mlc/scripts/sim_best_xi.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from predict_fixture import (  # noqa: E402
    MODEL_DIR, load_trackers, compute_features, apply_encoders_and_predict,
)
from player_metadata import PlayerMetadataProvider  # noqa: E402
from stats_provider import StatsProvider  # noqa: E402

ROSTER = Path(
    "/Users/aryamangupta/Projects/cric-analysis/mlc-2026/mlc_2026_rosters.csv"
)
FIX_DIR = REPO / "fixtures" / "mlc_2026"
ABBR = {"San Francisco Unicorns": "SFU", "Texas Super Kings": "TSK",
        "Los Angeles Knight Riders": "LAKR"}

# --- validated best XIs (ordered top-6 bat then bowlers) ---------------------
BEST_XI = {
    "San Francisco Unicorns": [
        "Finn Allen", "Matt Short", "Lhuan-dre Pretorius", "Sanjay Krishnamurthi",
        "Hammad Azam", "Hassan Khan", "Ravichandran Ashwin", "Xavier Bartlett",
        "Haris Rauf", "Zia-ul-Haq", "Juanoy Drysdale",
    ],
    "Texas Super Kings": [
        "Faf du Plessis", "Saiteja Mukkamalla", "Rilee Rossouw", "Donovan Ferreira",
        "Shubham Ranjane", "Calvin Savage", "Milind Kumar", "Akeal Hosein",
        "Adam Milne", "Hardus Viljoen", "Mohammad Mohsin",
    ],
    "Los Angeles Knight Riders": [
        "Alex Hales", "Sunil Narine", "Unmukt Chand", "Rovman Powell",
        "Andre Russell", "Saif Badar", "Jason Holder", "Jahmar Hamilton",
        "Shadley van Schalkwyk", "Karthik Gattepalli", "Ali Khan",
    ],
}

# --- variations: each = a full alternate XI for one team --------------------
VARIATIONS = {
    "San Francisco Unicorns": [
        ("Connolly IN for Rauf (+bat depth, −express pace)", [
            "Finn Allen", "Matt Short", "Lhuan-dre Pretorius", "Sanjay Krishnamurthi",
            "Cooper Connolly", "Hammad Azam", "Hassan Khan", "Ravichandran Ashwin",
            "Xavier Bartlett", "Zia-ul-Haq", "Juanoy Drysdale",
        ]),
    ],
    "Los Angeles Knight Riders": [
        ("Munro IN for Shadley (+bat, −pace)", [
            "Alex Hales", "Sunil Narine", "Unmukt Chand", "Rovman Powell",
            "Andre Russell", "Colin Munro", "Saif Badar", "Jason Holder",
            "Jahmar Hamilton", "Karthik Gattepalli", "Ali Khan",
        ]),
        ("Fletcher keeps for Hamilton (−Shadley, +Tromp)", [
            "Alex Hales", "Sunil Narine", "Unmukt Chand", "Rovman Powell",
            "Andre Russell", "Saif Badar", "Jason Holder", "Andre Fletcher",
            "Matthew Tromp", "Karthik Gattepalli", "Ali Khan",
        ]),
    ],
}

# (fixture file stem, team1, team2)
MATCHES = [
    ("2026-06-19_lakr_sfu", "Los Angeles Knight Riders", "San Francisco Unicorns"),
    ("2026-06-20_tsk_sfu", "Texas Super Kings", "San Francisco Unicorns"),
]


def load_name2id() -> dict[str, str]:
    return {r["player"]: r["cricsheet_id"].strip() for r in csv.DictReader(open(ROSTER))}


def main() -> int:
    name2id = load_name2id()

    def ids(names):
        out = []
        for n in names:
            if n not in name2id:
                raise KeyError(f"{n!r} not in roster")
            out.append(name2id[n] if name2id[n] else n)
        return out

    print("Loading providers + trackers (once)...")
    provider = StatsProvider(str(REPO / "models"), version="v3")
    metadata = PlayerMetadataProvider(str(REPO / "data" / "all_players_enriched.csv"))
    form, h2h, home = load_trackers()

    def predict(fx_meta, t1_xi, t2_xi):
        fx = dict(fx_meta)
        fx["team1_lineup"] = ids(t1_xi)
        fx["team2_lineup"] = ids(t2_xi)
        rec = compute_features(fx, provider, metadata, form, h2h, home)
        p1, _ = apply_encoders_and_predict(rec, MODEL_DIR)
        return p1, rec

    for stem, t1, t2 in MATCHES:
        fx_meta = json.loads((FIX_DIR / f"{stem}.json").read_text())
        a1, a2 = ABBR[t1], ABBR[t2]
        print(f"\n================  {a1} v {a2}  @ {fx_meta['venue']}  ({fx_meta['date']})  ================")

        # base: both teams best XI
        p1, rec = predict(fx_meta, BEST_XI[t1], BEST_XI[t2])
        print(f"  {'BASE  (both best XI)':<46}{a1} {p1*100:5.1f}%  |  {a2} {(1-p1)*100:5.1f}%")

        # variations on team1
        for label, xi in VARIATIONS.get(t1, []):
            pv, rv = predict(fx_meta, xi, BEST_XI[t2])
            d = (pv - p1) * 100
            print(f"  {a1+': '+label:<46}{a1} {pv*100:5.1f}%  ({d:+.1f})   "
                  f"[top6bat {rv['team1_top6_batting_elo_avg']:.0f} "
                  f"bot5bowl {rv['team1_bottom5_bowling_elo_avg']:.0f}]")
        # variations on team2
        for label, xi in VARIATIONS.get(t2, []):
            pv, rv = predict(fx_meta, BEST_XI[t1], xi)
            d = ((1 - pv) - (1 - p1)) * 100
            print(f"  {a2+': '+label:<46}{a2} {(1-pv)*100:5.1f}%  ({d:+.1f})   "
                  f"[top6bat {rv['team2_top6_batting_elo_avg']:.0f} "
                  f"bot5bowl {rv['team2_bottom5_bowling_elo_avg']:.0f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
