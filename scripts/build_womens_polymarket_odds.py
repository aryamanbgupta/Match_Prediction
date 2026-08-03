#!/usr/bin/env python
"""Join the strict women's Polymarket capture to the I12 w1 fixture pool.

The w1 women's track (`docs/I12_WOMENS_TRACK_SCOPING.md`) was built with no
market data at all — the scoping memo recorded "there is no odds-based eval
gate available today" on the strength of a 2026-07-23 men's-scoped capture
that found only four women's markets.  That reading was an artifact of the
scope filter, not of the market: Gamma lists women's internationals under
plain ``crint-`` slugs and carries the gender only in the event *title*, so a
slug-driven look misses them entirely.  A female-scoped strict pull returns
hundreds of women's T20 markets from 2026-01-20 onward.

This script is the join half of that correction.  Extraction stays in the
sibling `polymarket-cricket` repo (``extract_match_prematch_odds_strict.py
--gender female --format t20``); everything here is fixture matching and
provenance.

Isolation contract — nothing existing is read for writing or overwritten:

* every output lands under ``data/womens_polymarket/``;
* the men's odds sets (``betting_odds_polymarket.json``, ``data/golden/``,
  ``data/forward_holdout/``) are never opened;
* the script refuses to replace its own outputs without ``--overwrite``.

Join contract:

1. **Exact date only.** Women's series routinely repeat the same pairing on
   consecutive days, and Polymarket's event date can sit up to two days off
   the Cricsheet date, so a fuzzy window cannot tell which leg of a series a
   market belongs to.  Near misses are reported, never joined.
2. **Outcome-blind.** The realized winner is never used to select, dedupe, or
   orient a market.  It is compared afterwards as an integrity check and any
   disagreement is reported.
3. **Explicit aliases.** Team names are matched under a small, listed alias
   table; anything outside it stays unmatched and visible in the report.

Usage:
    uv run python scripts/build_womens_polymarket_odds.py
    uv run python scripts/build_womens_polymarket_odds.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CAPTURE = Path(
    "/Users/aryamangupta/Projects/polymarket-cricket/data/"
    "polymarket_match_odds_strict_female_2025-07-01_2026-07-12.json"
)
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "xgb_match_data_w1"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "womens_polymarket"

# Split parquet -> label used in the report and in each emitted row.
SPLITS = {"test": "test", "golden_test": "golden"}

# Polymarket spelling -> Cricsheet spelling.  A trailing " Women" is stripped
# before lookup (the WC-Qualifier events use "Nepal Women", the bilateral
# series use bare "Nepal" for the same side, and every WPL side carries the
# suffix).  Kept as two tables so each corpus's provenance stays readable.
INTERNATIONAL_TEAM_ALIASES = {
    "hongkongchina": "hongkong",
    "usa": "unitedstatesofamerica",
    "uae": "unitedarabemirates",
    "png": "papuanewguinea",
}

# Franchise sides where Polymarket carries the marketing name and Cricsheet
# the county name.
LEAGUE_TEAM_ALIASES = {
    "lancashirethunder": "lancashire",
}

TEAM_ALIASES = {**INTERNATIONAL_TEAM_ALIASES, **LEAGUE_TEAM_ALIASES}

NEAR_MISS_DAYS = 3


def normalise_team(name: str) -> str:
    """Alias-resolved comparison key for a team name."""
    stripped = re.sub(r"\s+women$", "", str(name).strip(), flags=re.IGNORECASE)
    key = "".join(ch for ch in stripped.lower() if ch.isalnum())
    return TEAM_ALIASES.get(key, key)


def pair_key(team1: str, team2: str) -> frozenset[str]:
    return frozenset({normalise_team(team1), normalise_team(team2)})


def load_fixtures(data_dir: Path) -> list[dict]:
    """Fixture rows the w1 model actually scored, from the split parquets."""
    import pandas as pd

    fixtures: list[dict] = []
    for parquet_name, split in SPLITS.items():
        path = data_dir / f"{parquet_name}.parquet"
        if not path.exists():
            raise SystemExit(f"missing split parquet: {path}")
        frame = pd.read_parquet(path)
        for row in frame.itertuples(index=False):
            fixtures.append(
                {
                    "match_id": str(row.match_id),
                    "split": split,
                    "match_date": str(row.match_date)[:10],
                    "team1": str(row.team1),
                    "team2": str(row.team2),
                    "venue": str(row.venue),
                    "team1_wins": int(row.team1_wins),
                }
            )
    return fixtures


def load_predictions(arm_dir: Path) -> dict[str, float]:
    """p_team1 by match id, unioned across the arm's scored splits."""
    preds: dict[str, float] = {}
    for name in ("test_predictions.json", "golden_predictions.json"):
        path = arm_dir / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        rows = payload.get("predictions", payload)
        for key, record in rows.items():
            preds[str(key)] = (
                float(record["p_team1"])
                if isinstance(record, dict)
                else float(record)
            )
    return preds


def select_market(candidates: list[dict]) -> dict:
    """Outcome-blind tiebreak, mirroring the strict extractor's documented rule.

    Highest volume, then the latest valid pre-start quote, then stable market
    id.  The realized winner is deliberately absent from the key.
    """
    ordered = sorted(candidates, key=lambda m: str(m.get("market_id") or ""))
    ordered.sort(key=lambda m: m.get("price_timestamp") or "", reverse=True)
    ordered.sort(key=lambda m: float(m.get("volume_usd") or 0.0), reverse=True)
    return ordered[0]


def decimal_odds(probability: float) -> float | None:
    if not probability or probability <= 0.0:
        return None
    return round(1.0 / probability, 4)


def build(args: argparse.Namespace) -> dict:
    capture = json.loads(args.capture.read_text())
    if capture.get("gender_scope") != "female":
        raise SystemExit(
            f"{args.capture} has gender_scope="
            f"{capture.get('gender_scope')!r}; this builder requires a "
            "female-scoped strict capture"
        )
    markets = capture["matches"]
    fixtures = load_fixtures(args.data_dir)
    predictions = {
        arm.name: load_predictions(arm) for arm in args.arm
    }

    by_date: dict[str, list[dict]] = defaultdict(list)
    by_pair: dict[frozenset[str], list[dict]] = defaultdict(list)
    for market in markets:
        by_date[market["date"]].append(market)
        by_pair[pair_key(market["team1"], market["team2"])].append(market)

    rows: list[dict] = []
    consumed: set[int] = set()
    unmatched_fixtures: list[dict] = []
    near_misses: list[dict] = []
    winner_conflicts: list[dict] = []

    for fixture in sorted(fixtures, key=lambda f: (f["match_date"], f["match_id"])):
        key = pair_key(fixture["team1"], fixture["team2"])
        same_day = [
            m
            for m in by_date.get(fixture["match_date"], [])
            if id(m) not in consumed
            and pair_key(m["team1"], m["team2"]) == key
        ]
        if not same_day:
            unmatched_fixtures.append(fixture)
            continue

        market = select_market(same_day)
        consumed.add(id(market))

        # Orient market probabilities onto the fixture's team order by name.
        if normalise_team(market["team1"]) == normalise_team(fixture["team1"]):
            p1, p2 = market["prematch_price_team1"], market["prematch_price_team2"]
        else:
            p1, p2 = market["prematch_price_team2"], market["prematch_price_team1"]

        # Integrity check only — never a selection input.  A market winner
        # that normalizes to NEITHER team (void market, odd spelling) is a
        # conflict in its own right — without this branch it would silently
        # pass whenever team1 happened to lose.
        market_winner = normalise_team(market["winner"])
        fixture_team_keys = {
            normalise_team(fixture["team1"]),
            normalise_team(fixture["team2"]),
        }
        if market_winner not in fixture_team_keys:
            winner_conflicts.append(
                {
                    "match_id": fixture["match_id"],
                    "date": fixture["match_date"],
                    "fixture": f"{fixture['team1']} vs {fixture['team2']}",
                    "cricsheet_team1_wins": fixture["team1_wins"],
                    "market_winner": market["winner"],
                    "event_slug": market["event_slug"],
                    "reason": "market winner matches neither fixture team",
                }
            )
        elif int(market_winner == normalise_team(fixture["team1"])) != fixture[
            "team1_wins"
        ]:
            winner_conflicts.append(
                {
                    "match_id": fixture["match_id"],
                    "date": fixture["match_date"],
                    "fixture": f"{fixture['team1']} vs {fixture['team2']}",
                    "cricsheet_team1_wins": fixture["team1_wins"],
                    "market_winner": market["winner"],
                    "event_slug": market["event_slug"],
                    "reason": "market winner disagrees with Cricsheet",
                }
            )

        row = {
            "match_id": fixture["match_id"],
            "split": fixture["split"],
            "date": fixture["match_date"],
            "team1": fixture["team1"],
            "team2": fixture["team2"],
            "venue": fixture["venue"],
            "actual_winner": (
                fixture["team1"] if fixture["team1_wins"] else fixture["team2"]
            ),
            "odds": {
                "winner": {
                    fixture["team1"]: decimal_odds(p1),
                    fixture["team2"]: decimal_odds(p2),
                    "timestamp": market["price_timestamp"],
                }
            },
            "prematch_prob_team1": p1,
            "prematch_prob_team2": p2,
            "source": "polymarket",
            "polymarket_event_slug": market["event_slug"],
            "polymarket_event_title": market["event_title"],
            "polymarket_market_id": market["market_id"],
            "polymarket_volume_usd": market["volume_usd"],
            "scheduled_start_timestamp": market["scheduled_start_timestamp"],
            "price_lag_seconds": market["price_lag_seconds"],
            "tournament": market["tournament"],
            "inferred_format": market["inferred_format"],
        }
        for arm_name, arm_preds in predictions.items():
            if fixture["match_id"] in arm_preds:
                row[f"p_team1_{arm_name}"] = arm_preds[fixture["match_id"]]
        rows.append(row)

    leftover = [m for m in markets if id(m) not in consumed]

    # Near misses are reported against the FINAL consumed set — computing
    # them mid-walk would list markets that a later exact-date fixture
    # legitimately joins.
    for fixture in unmatched_fixtures:
        key = pair_key(fixture["team1"], fixture["team2"])
        for market in by_pair.get(key, []):
            if id(market) in consumed:
                continue
            offset = (
                datetime.fromisoformat(market["date"])
                - datetime.fromisoformat(fixture["match_date"])
            ).days
            if 0 < abs(offset) <= NEAR_MISS_DAYS:
                near_misses.append(
                    {
                        "match_id": fixture["match_id"],
                        "fixture_date": fixture["match_date"],
                        "fixture": f"{fixture['team1']} vs {fixture['team2']}",
                        "market_date": market["date"],
                        "market": f"{market['team1']} vs {market['team2']}",
                        "day_offset": offset,
                        "event_slug": market["event_slug"],
                        "volume_usd": market["volume_usd"],
                        "note": "NOT joined; exact-date contract",
                    }
                )

    matched_by_split = Counter(r["split"] for r in rows)
    fixture_totals = Counter(f["split"] for f in fixtures)
    volume_thresholds = [1000, 5000, 10000, 50000, 100000]

    output = {
        "schema_version": 1,
        "source": "polymarket",
        "purpose": (
            f"women's market odds joined to the {args.label} fixture pool"
        ),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gender_scope": "female",
        "capture": {
            "path": str(args.capture),
            "extracted_at": capture.get("extracted_at"),
            "cutoff_date": capture.get("cutoff_date"),
            "through_date": capture.get("through_date"),
            "format_scope": capture.get("format_scope"),
            "rows": len(markets),
            "selection_contract": capture.get("selection_contract"),
        },
        "join_contract": {
            "date_rule": "exact Cricsheet date == Polymarket event date",
            "near_miss_window_days": NEAR_MISS_DAYS,
            "near_misses_joined": False,
            "team_alias_table": TEAM_ALIASES,
            "team_alias_sources": {
                "international": INTERNATIONAL_TEAM_ALIASES,
                "league": LEAGUE_TEAM_ALIASES,
            },
            "womens_suffix_stripped": True,
            "duplicate_rule": "highest volume, latest pre-start quote, market id",
            "winner_used_for_selection": False,
        },
        "fixture_pool": {
            "data_dir": str(args.data_dir),
            "splits": dict(fixture_totals),
            "total": len(fixtures),
        },
        "coverage": {
            "matched": len(rows),
            "matched_by_split": dict(matched_by_split),
            "coverage_by_split": {
                split: round(matched_by_split[split] / fixture_totals[split], 4)
                for split in fixture_totals
            },
            "matched_by_volume_threshold": {
                f">= ${threshold}": sum(
                    1
                    for r in rows
                    if (r["polymarket_volume_usd"] or 0.0) >= threshold
                )
                for threshold in volume_thresholds
            },
            "matched_by_tournament": dict(Counter(r["tournament"] for r in rows)),
            "unmatched_fixtures": len(unmatched_fixtures),
            "unjoined_markets": len(leftover),
            "unjoined_markets_by_tournament": dict(
                Counter(m["tournament"] for m in leftover)
            ),
            "winner_conflicts": len(winner_conflicts),
        },
        "matches": sorted(rows, key=lambda r: (r["date"], r["match_id"])),
    }

    report = {
        "generated_at": output["generated_at"],
        "coverage": output["coverage"],
        "winner_conflicts": winner_conflicts,
        "near_misses": sorted(near_misses, key=lambda r: r["fixture_date"]),
        "unmatched_fixtures": [
            {
                "match_id": f["match_id"],
                "split": f["split"],
                "date": f["match_date"],
                "fixture": f"{f['team1']} vs {f['team2']}",
            }
            for f in unmatched_fixtures
        ],
        "unjoined_markets": [
            {
                "date": m["date"],
                "market": f"{m['team1']} vs {m['team2']}",
                "tournament": m["tournament"],
                "event_slug": m["event_slug"],
                "volume_usd": m["volume_usd"],
            }
            for m in sorted(leftover, key=lambda m: m["date"])
        ],
    }

    if args.dry_run:
        print(json.dumps(output["coverage"], indent=2))
        print(f"\n(dry run — nothing written to {args.out_dir})")
        return output

    args.out_dir.mkdir(parents=True, exist_ok=True)
    targets = {
        args.out_dir / args.odds_filename: output,
        args.out_dir / "join_report.json": report,
    }
    for path in targets:
        if path.exists() and not args.overwrite:
            raise SystemExit(
                f"refusing to overwrite {path}; pass --overwrite to replace"
            )
    for path, payload in targets.items():
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(path)
        print(f"wrote {path}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--arm",
        type=Path,
        nargs="+",
        default=[
            REPO_ROOT / "models" / "xgb_match_w1_base",
            REPO_ROOT / "models" / "xgb_match_w1_swap",
        ],
        help="Model dirs whose p_team1 is attached to each joined row.",
    )
    parser.add_argument(
        "--label",
        default="I12 w1",
        help="Corpus label recorded in the output's purpose string.",
    )
    parser.add_argument(
        "--odds-filename",
        default="betting_odds_womens_w1.json",
        help="Filename for the joined odds set inside --out-dir.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = build(args)
    coverage = result["coverage"]
    print(
        f"\nmatched {coverage['matched']} / {result['fixture_pool']['total']} "
        f"{args.label} fixtures ({coverage['matched_by_split']})"
    )
    if coverage["winner_conflicts"]:
        print(
            f"WARNING: {coverage['winner_conflicts']} market/Cricsheet winner "
            "disagreements — see join_report.json"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
