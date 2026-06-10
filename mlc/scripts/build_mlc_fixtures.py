"""Generate MLC 2026 fixture JSONs (predict_fixture.py inputs) from the
announced squads in cric-analysis/mlc-2026/mlc_2026_rosters.csv.

The season hasn't started, so lineups are PROJECTED XIs from the announced
2026 squads, ordered batting-order-then-bowlers (the match model splits top-6
batters / bottom-5 bowlers by list position). Players are emitted as
cricsheet_ids (robust against display-name mismatches); any player without a
cricsheet id falls back to their display name (default ELO = correct "no prior
data" signal).

XI selection honors the **MLC playing-XI rule: max 6 overseas + min 5
USA-developed players** (the ICC 4-overseas cap currently exempts MLC). That
constraint binds hard on the stacked squads — LAKR (Narine/Russell/Hales/
Munro/Powell/Fletcher/Holder/Pathirana = 8 overseas) and WSH (Smith/Maxwell/
Rachin/Owen/Chapman/Edwards/Jansen/Ferguson/Dwarshuis = 9) can field only 6 of
their internationals. Domestic/overseas classification is by player nationality
(not a roster column) and is approximate for a handful of naturalized cases.

Venues in the published schedule use loose names ("Grand Prairie, Texas").
We map to the canonical cricsheet strings the model holds venue ELO/dist for.

Usage:
    uv run python mlc/scripts/build_mlc_fixtures.py            # Dallas leg (1-7)
    uv run python mlc/scripts/build_mlc_fixtures.py --all      # full league (30)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROSTER_CSV = Path(
    "/Users/aryamangupta/Projects/cric-analysis/mlc-2026/mlc_2026_rosters.csv"
)
OUT_DIR = REPO / "fixtures" / "mlc_2026"

VENUE_CANONICAL = {
    "Grand Prairie": "Grand Prairie Stadium, Dallas",
    "Oakland": "Oakland Coliseum,Oakland",
    "Broward": "Central Broward Regional Park Stadium Turf Ground, Lauderhill",
}

# Team abbreviations used in fixture filenames / schedule.
ABBR = {
    "MI New York": "miny",
    "Texas Super Kings": "tsk",
    "Washington Freedom": "wsh",
    "San Francisco Unicorns": "sfu",
    "Seattle Orcas": "orca",
    "Los Angeles Knight Riders": "lakr",
}

# Projected XIs — ordered top-6 batters then bowlers; max 6 overseas (o) + 5
# USA-developed (d). Names must match the `player` column in
# mlc_2026_rosters.csv. Built from the June-2026 franchise-site squads;
# provisional and subject to the actual matchday XI / toss.
PROJECTED_XI = {
    # 6 o: de Kock, Pooran, Pollard, Shepherd, Boult, Ghazanfar
    "MI New York": [
        "Quinton de Kock", "Monank Patel", "Nicholas Pooran", "Kieron Pollard",
        "Corey Anderson", "Romario Shepherd", "Tajinder Singh", "Trent Boult",
        "Allah Ghazanfar", "Nosthush Kenjige", "Rushil Ugarkar",
    ],
    # 6 o: Faf, Rossouw, Ferreira, Mulder, Hosein, Milne
    "Texas Super Kings": [
        "Faf du Plessis", "Saiteja Mukkamalla", "Rilee Rossouw",
        "Donovan Ferreira", "Wiaan Mulder", "Smit Patel", "Calvin Savage",
        "Shubham Ranjane", "Akeal Hosein", "Adam Milne", "Mohammad Mohsin",
    ],
    # 6 o: Rachin, Smith, Maxwell, Owen, Jansen, Ferguson
    "Washington Freedom": [
        "Andries Gous", "Rachin Ravindra", "Steven Smith", "Glenn Maxwell",
        "Mitchell Owen", "Mukhtar Ahmed", "Ian Holland", "Marco Jansen",
        "Lockie Ferguson", "Saurabh Netravalkar", "Yasir Mohammad",
    ],
    # 6 o: Allen, Short, Pretorius, Ashwin, Rauf, Bartlett
    "San Francisco Unicorns": [
        "Finn Allen", "Matt Short", "Lhuan-dre Pretorius",
        "Sanjay Krishnamurthi", "Hammad Azam", "Ravichandran Ashwin",
        "Hassan Khan", "Haris Rauf", "Xavier Bartlett", "Juanoy Drysdale",
        "Zia-ul-Haq",
    ],
    # 6 o: Seifert, Hetmyer, Breetzke, Stoinis, Shanaka, Ngidi
    "Seattle Orcas": [
        "Tim Seifert", "Shimron Hetmyer", "Matthew Breetzke", "Marcus Stoinis",
        "Dasun Shanaka", "Shayan Jahangir", "Harmeet Singh", "Lungi Ngidi",
        "Jasdeep Singh", "Ayan Desai", "Ali Sheikh",
    ],
    # 6 o: Narine, Hales, Powell, Russell, Holder, Pathirana
    "Los Angeles Knight Riders": [
        "Sunil Narine", "Alex Hales", "Unmukt Chand", "Rovman Powell",
        "Andre Russell", "Jahmar Hamilton", "Saif Badar", "Jason Holder",
        "Matheesha Pathirana", "Ali Khan", "Nitish Kumar",
    ],
}

# (date, home/team1, away/team2, venue_key). team1 = listed-first (home) side.
SCHEDULE = [
    ("2026-06-18", "Texas Super Kings", "Seattle Orcas", "Grand Prairie"),
    ("2026-06-19", "Los Angeles Knight Riders", "San Francisco Unicorns", "Grand Prairie"),
    ("2026-06-19", "Seattle Orcas", "Washington Freedom", "Grand Prairie"),
    ("2026-06-20", "Texas Super Kings", "San Francisco Unicorns", "Grand Prairie"),
    ("2026-06-20", "Washington Freedom", "MI New York", "Grand Prairie"),
    ("2026-06-21", "Seattle Orcas", "Los Angeles Knight Riders", "Grand Prairie"),
    ("2026-06-21", "Texas Super Kings", "MI New York", "Grand Prairie"),
    # --- rest of league (Oakland + Broward + Dallas legs) for --all ---
    ("2026-06-24", "San Francisco Unicorns", "Texas Super Kings", "Oakland"),
    ("2026-06-25", "Washington Freedom", "Seattle Orcas", "Oakland"),
    ("2026-06-26", "MI New York", "Texas Super Kings", "Oakland"),
    ("2026-06-26", "San Francisco Unicorns", "Seattle Orcas", "Oakland"),
    ("2026-06-27", "MI New York", "Los Angeles Knight Riders", "Oakland"),
    ("2026-06-27", "Washington Freedom", "Texas Super Kings", "Oakland"),
    ("2026-06-28", "Los Angeles Knight Riders", "Seattle Orcas", "Oakland"),
    ("2026-06-28", "San Francisco Unicorns", "Washington Freedom", "Oakland"),
    ("2026-07-01", "Los Angeles Knight Riders", "Washington Freedom", "Broward"),
    ("2026-07-02", "Seattle Orcas", "MI New York", "Broward"),
    ("2026-07-03", "Los Angeles Knight Riders", "Texas Super Kings", "Broward"),
    ("2026-07-04", "Washington Freedom", "San Francisco Unicorns", "Broward"),
    ("2026-07-04", "Los Angeles Knight Riders", "MI New York", "Broward"),
    ("2026-07-05", "Seattle Orcas", "Texas Super Kings", "Broward"),
    ("2026-07-05", "San Francisco Unicorns", "MI New York", "Broward"),
    ("2026-07-08", "MI New York", "San Francisco Unicorns", "Grand Prairie"),
    ("2026-07-09", "Washington Freedom", "Los Angeles Knight Riders", "Grand Prairie"),
    ("2026-07-10", "MI New York", "Seattle Orcas", "Grand Prairie"),
    ("2026-07-10", "San Francisco Unicorns", "Los Angeles Knight Riders", "Grand Prairie"),
    ("2026-07-11", "Texas Super Kings", "Washington Freedom", "Grand Prairie"),
    ("2026-07-11", "Seattle Orcas", "San Francisco Unicorns", "Grand Prairie"),
    ("2026-07-12", "MI New York", "Washington Freedom", "Grand Prairie"),
    ("2026-07-12", "Texas Super Kings", "Los Angeles Knight Riders", "Grand Prairie"),
]

DALLAS_LEG = 7  # first N rows = opening Grand Prairie leg


def load_name_to_id() -> dict[str, str]:
    """name -> cricsheet_id (blank for the uncapped domestic signings)."""
    out: dict[str, str] = {}
    with open(ROSTER_CSV, newline="") as f:
        for row in csv.DictReader(f):
            out[row["player"]] = row["cricsheet_id"].strip()
    return out


def lineup_ids(team: str, name2id: dict[str, str]) -> tuple[list[str], list[str]]:
    ids, names = [], []
    for nm in PROJECTED_XI[team]:
        if nm not in name2id:
            raise KeyError(f"{nm!r} not in roster CSV for {team}")
        pid = name2id[nm]
        ids.append(pid if pid else nm)  # fall back to name if no cricsheet id
        names.append(nm)
    return ids, names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="emit the full 30-match league (default: Dallas leg, 7)")
    args = ap.parse_args()

    name2id = load_name_to_id()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = SCHEDULE if args.all else SCHEDULE[:DALLAS_LEG]

    written = []
    for date, t1, t2, vkey in rows:
        venue = VENUE_CANONICAL[vkey]
        t1_ids, t1_names = lineup_ids(t1, name2id)
        t2_ids, t2_names = lineup_ids(t2, name2id)
        fixture = {
            "_comment": (
                f"MLC 2026 league: {t1} vs {t2} @ {venue} ({date}). "
                "PRE-XI / PRE-TOSS projected estimate. Lineups are projected "
                "XIs from announced 2026 squads (mlc_2026_rosters.csv), ordered "
                "batting-order-then-bowlers, max 6 overseas + 5 USA-developed. "
                "team1 = listed-first (home) side. Provisional."
            ),
            "date": date,
            "team1": t1,
            "team2": t2,
            "venue": venue,
            "competition_tier": "Major League Cricket",
            "team_type": "club",
            "team1_lineup": t1_ids,
            "_team1_lineup_names": t1_names,
            "team2_lineup": t2_ids,
            "_team2_lineup_names": t2_names,
            "toss_winner": None,
            "toss_decision": None,
            "polymarket_odds": None,
        }
        fn = f"{date}_{ABBR[t1]}_{ABBR[t2]}.json"
        path = OUT_DIR / fn
        with open(path, "w") as f:
            json.dump(fixture, f, indent=2)
        written.append(path.name)

    print(f"Wrote {len(written)} fixtures to {OUT_DIR}:")
    for w in written:
        print(f"  {w}")


if __name__ == "__main__":
    main()
