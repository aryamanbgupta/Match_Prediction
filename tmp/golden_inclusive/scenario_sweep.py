"""Sweep predictions across {toss × XI} combinations for a fixture.

Takes one or more base fixture JSONs and writes one prediction per
(toss_winner, toss_decision) combo, plus the null-toss default. If
multiple fixtures are passed they are treated as XI variants and all
get the same toss sweep.

Uses the refreshed SQLite + tracker snapshot in tmp/golden_inclusive/
and the production M7 model.

Usage:
    uv run python tmp/golden_inclusive/scenario_sweep.py \\
        --fixture fixtures/2026-05-17_pbks_rcb.json \\
        --label pbks_rcb_main
    uv run python tmp/golden_inclusive/scenario_sweep.py \\
        --fixture fixtures/2026-05-17_dc_rr.json \\
        --fixture fixtures/2026-05-17_dc_rr_alt_hetmyer.json \\
        --label dc_rr_sweep
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tmp" / "golden_inclusive"))

from predict_fixture import (  # noqa: E402
    compute_features, apply_encoders_and_predict,
)
import predict_fixture  # noqa: E402
from player_metadata import PlayerMetadataProvider  # noqa: E402
from stats_provider import StatsProvider  # noqa: E402

import predict_with_refreshed_state as refresh_driver  # noqa: E402


TOSS_COMBOS = [
    ("(null/default)", None, None),
    ("T1 wins → BAT",  "team1", "bat"),
    ("T1 wins → FIELD","team1", "field"),
    ("T2 wins → BAT",  "team2", "bat"),
    ("T2 wins → FIELD","team2", "field"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", type=Path, action="append", required=True,
                    help="Base fixture JSON. Repeat to add XI variants.")
    ap.add_argument("--label", type=str, required=True)
    args = ap.parse_args()

    # Pin to production M7
    predict_fixture.MODEL_DIR = refresh_driver.MODEL_DIR

    # Load shared resources once
    provider = StatsProvider(str(refresh_driver.TMP_SQLITE_DIR), version="v3")
    metadata = PlayerMetadataProvider(
        str(REPO / "data" / "all_players_enriched.csv"))
    form, h2h, home, snap_as_of = refresh_driver.load_combined_trackers()
    print(f"Snapshot as_of: {snap_as_of}")
    print()

    out_root = REPO / "tmp" / "golden_inclusive" / "sweep_predictions"
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for fix_path in args.fixture:
        base = json.loads(fix_path.read_text())
        variant_name = fix_path.stem
        t1, t2 = base["team1"], base["team2"]
        for combo_label, tw_alias, td in TOSS_COMBOS:
            f = deepcopy(base)
            if tw_alias == "team1":
                f["toss_winner"] = t1
            elif tw_alias == "team2":
                f["toss_winner"] = t2
            else:
                f["toss_winner"] = None
            f["toss_decision"] = td

            rec = compute_features(f, provider, metadata, form, h2h, home)
            p1, debug = apply_encoders_and_predict(rec)
            rows.append({
                "variant": variant_name,
                "scenario": combo_label,
                "toss_winner": f["toss_winner"],
                "toss_decision": td,
                "P_team1": p1,
                "P_team2": 1 - p1,
                "team1_batting_first": int(rec["team1_batting_first"]),
                "top6_batting_elo_diff": rec["top6_batting_elo_diff"],
                "bottom5_bowling_elo_diff": rec["bottom5_bowling_elo_diff"],
                "elo_diff_batting": rec["elo_diff_batting"],
                "elo_diff_bowling": rec["elo_diff_bowling"],
                "team1_win_rate_last_10": rec["team1_win_rate_last_10"],
                "team2_win_rate_last_10": rec["team2_win_rate_last_10"],
                "h2h_n_meetings": rec["h2h_n_meetings"],
                "is_team1_home": rec["is_team1_home"],
                "is_team2_home": rec["is_team2_home"],
                "venue_chase_win_pct": rec["venue_chase_win_pct"],
            })

    # Pretty-print
    print(f"=== {args.label}  ({t1} vs {t2}) ===")
    print(f"{'variant':38s}  {'scenario':18s}  T1bat  P(T1)   P(T2)")
    for r in rows:
        print(f"  {r['variant']:36s}  {r['scenario']:18s}  {r['team1_batting_first']:>4d}  "
              f"{r['P_team1']*100:>5.1f}%  {r['P_team2']*100:>5.1f}%")

    # Save
    out_path = out_root / f"{args.label}.json"
    out_path.write_text(json.dumps({
        "label": args.label,
        "team1": t1,
        "team2": t2,
        "snapshot_as_of": snap_as_of,
        "scenarios": rows,
    }, indent=2))
    print(f"\nSaved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
