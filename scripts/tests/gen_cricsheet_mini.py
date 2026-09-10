"""Generate the checked-in deterministic mini Cricsheet corpus.

Seed: 20260910. Output uses sorted keys, a fixed indent, and no timestamps.
"""

from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path

from cricsheet_fixtures import innings, match_json


SEED = 20260910
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "cricsheet_mini"


def main() -> None:
    rng = random.Random(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dates = [date(2025, 1, 1) + timedelta(days=index) for index in range(20)]
    dates[9] = dates[8]  # Required deterministic same-day sibling pair.

    for index, match_date in enumerate(dates):
        cricsheet_id = str(710000 + index)
        competition = "Indian Premier League" if index < 10 else "Big Bash League"
        document = match_json(
            match_date.isoformat(),
            "Alpha CC",
            "Beta CC",
            seed=rng.randrange(10_000),
            match_id=cricsheet_id,
            event=competition,
        )
        document["meta"] = {"data_version": "1.1.0", "revision": 1}
        document["info"]["cricsheet_id"] = cricsheet_id

        if index == 5:
            document["info"]["outcome"] = {
                "winner": "Alpha CC",
                "method": "D/L",
                "by": {"runs": 7},
            }
        if index == 12:
            document["info"]["outcome"] = {
                "winner": "Beta CC",
                "eliminator": "Beta CC",
            }
            document["innings"].extend([
                innings("Alpha CC", batter="Alpha P1", bowler="Beta P8",
                        runs_per_over=[[1, 1, 4]], super_over=True),
                innings("Beta CC", batter="Beta P1", bowler="Alpha P8",
                        runs_per_over=[[2, 2, 2]], super_over=True),
            ])

        output = OUTPUT_DIR / f"{cricsheet_id}.json"
        output.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
