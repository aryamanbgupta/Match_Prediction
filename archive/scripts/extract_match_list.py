"""Extract match metadata from all JSON files for odds lookup.

Outputs a JSON file with date, teams, tournament, and result for each match.
"""

import json
import os
from pathlib import Path


def extract_matches(data_dir: str = "data/t20s_json") -> list:
    matches = []
    json_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".json"))

    for fname in json_files:
        with open(os.path.join(data_dir, fname)) as fh:
            data = json.load(fh)

        info = data["info"]
        match_id = fname.replace(".json", "")
        date = info["dates"][0]  # first day
        teams = info.get("teams", list(info.get("players", {}).keys()))
        gender = info.get("gender", "unknown")
        event = info.get("event", {})
        event_name = event.get("name", None) if isinstance(event, dict) else None
        match_number = event.get("match_number") if isinstance(event, dict) else None
        group = event.get("group") if isinstance(event, dict) else None
        season = info.get("season", None)

        outcome = info.get("outcome", {})
        winner = outcome.get("winner", None)
        result = outcome.get("result", None)  # "no result", "tie", etc.

        matches.append({
            "match_id": match_id,
            "date": date,
            "team1": teams[0] if len(teams) > 0 else None,
            "team2": teams[1] if len(teams) > 1 else None,
            "gender": gender,
            "tournament": event_name,
            "season": str(season) if season else None,
            "group": group,
            "match_number": match_number,
            "winner": winner,
            "result": result,
        })

    # Sort by date, then match_id
    matches.sort(key=lambda m: (m["date"], m["match_id"]))
    return matches


if __name__ == "__main__":
    matches = extract_matches()
    output_path = "data/all_matches.json"
    with open(output_path, "w") as f:
        json.dump(matches, f, indent=2)
    print(f"Extracted {len(matches)} matches -> {output_path}")

    # Summary stats
    with_winner = sum(1 for m in matches if m["winner"])
    male = sum(1 for m in matches if m["gender"] == "male")
    female = sum(1 for m in matches if m["gender"] == "female")
    with_tournament = sum(1 for m in matches if m["tournament"])
    tournaments = sorted(set(m["tournament"] for m in matches if m["tournament"]))

    print(f"  With winner: {with_winner}")
    print(f"  Male: {male}, Female: {female}")
    print(f"  With tournament name: {with_tournament}")
    print(f"  Unique tournaments: {len(tournaments)}")
    print(f"  Date range: {matches[0]['date']} to {matches[-1]['date']}")
