"""Seal the still-unresolved Hundred 2026 forward predictions.

Writes one outcome-blind JSON holding, for every fixture whose result is not
yet known, both model arms under both state bases, the exact market quote the
predeclared rule would settle at, and the provenance of each XI. The file is
written with sorted keys and no volatile fields, then hashed, so the hash in
the report pins these numbers before the results exist.

Deterministic; no network access.

Usage:
    uv run python scripts/seal_hundred_forward.py \
        --roi-eval eval_out/hundred_roi_2026-08-03/roi_eval.json \
        --out predictions/hundred/forward_2026-08-03_sealed.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SEED_LIQUIDITY_USD = 1_000.0  # below this a quote is a seed, not a price


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi-eval", type=Path, required=True)
    ap.add_argument("--odds", type=Path,
                    default=REPO / "data/hundred/polymarket_odds_2026_v2.json")
    ap.add_argument("--pred-dir", type=Path,
                    default=REPO / "predictions/hundred")
    ap.add_argument("--fixture-dir", type=Path, default=REPO / "fixtures/hundred")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    roi = json.loads(args.roi_eval.read_text())
    odds_payload = json.loads(args.odds.read_text())
    odds_by_key = {(m["date"], m["team1"], m["team2"]): m
                   for m in odds_payload["matches"]}

    # Unresolved = every forward-slice ledger row with no winner, taken from
    # the primary arm's ledger so the set is defined once.
    unresolved = [r for r in roi["arms"]["i7"]["ledger"]["forward"]
                  if r["winner"] is None]

    entries = []
    for row in unresolved:
        date, t1, t2 = row["date"], row["team1"], row["team2"]
        odds = odds_by_key[(date, t1, t2)]
        slug = date + "_" + "_".join(
            t.lower().replace(" ", "_") for t in (t1, t2))
        # Fixture-file basename is the date + a short slug; find it by date+teams.
        fixture_path = None
        for cand in sorted(args.fixture_dir.glob(f"{date}_*.json")):
            fx = json.loads(cand.read_text())
            if fx["team1"] == t1 and fx["team2"] == t2:
                fixture_path = cand
                break
        if fixture_path is None:
            raise SystemExit(f"no fixture JSON for {date} {t1} v {t2}")
        fixture = json.loads(fixture_path.read_text())

        probs: dict = {}
        for arm in ("i7", "swap"):
            for base in ("cut0801", "full0802"):
                path = args.pred_dir / f"{fixture_path.stem}__{arm}__{base}.json"
                payload = json.loads(path.read_text())
                probs.setdefault(arm, {})[base] = {
                    "p_team1": payload["prediction"][t1],
                    "p_team2": payload["prediction"][t2],
                    "toss_branch_probs":
                        payload["diagnostics"]["toss_branch_probs"],
                    "tracker_snapshot":
                        payload["diagnostics"]["tracker_snapshot"],
                    "tracker_aux_match_count":
                        payload["diagnostics"]["tracker_aux_match_count"],
                    "state_freshness":
                        payload["diagnostics"]["state_freshness"]["status"],
                    "prediction_file": str(path.relative_to(REPO)),
                }

        pretoss = odds.get("pretoss_prob_team1")
        volume = odds.get("market_volume_usd")
        # No reported traded volume at all is thinner than a tiny one, not
        # richer: treat a missing figure as seed liquidity, never as unknown.
        seed = volume is None or volume < SEED_LIQUIDITY_USD

        # The rule settles at the pre-toss quote; record what it would do with
        # the primary arm under the frozen (cut0801) state base.
        p_primary = probs["i7"]["cut0801"]["p_team1"]
        if pretoss is None:
            declared = {"bet": None,
                        "reason": "no pre-toss quote captured for this fixture"}
        elif p_primary == pretoss:
            declared = {"bet": None, "reason": "model probability equals quote"}
        elif p_primary > pretoss:
            declared = {"side": t1, "price": pretoss,
                        "profit_if_win": 1.0 / pretoss - 1.0}
        else:
            declared = {"side": t2, "price": 1.0 - pretoss,
                        "profit_if_win": 1.0 / (1.0 - pretoss) - 1.0}

        entries.append({
            "slug": slug,
            "date": date,
            "team1": t1,
            "team2": t2,
            "venue": row["venue"],
            "scheduled_start_utc": odds.get("scheduled_start_utc"),
            "polymarket_slug": odds.get("slug"),
            "model_probabilities": probs,
            "quote": {
                "pretoss_prob_team1": pretoss,
                "pretoss_price_timestamp": odds.get("pretoss_price_timestamp"),
                "pretoss_lead_seconds": odds.get("pretoss_lead_seconds"),
                "posttoss_prematch_prob_team1": odds.get("prematch_prob_team1"),
                "posttoss_prematch_price_timestamp":
                    odds.get("prematch_price_timestamp"),
                "market_volume_usd": volume,
                "event_volume_usd": odds.get("event_volume_usd"),
                "liquidity_usd": odds.get("liquidity_usd"),
                "seed_liquidity_non_actionable": seed,
                "odds_status_at_pull": odds.get("status"),
            },
            "declared_bet_primary_arm_pretoss": declared,
            "xi_provenance": {
                "note": fixture.get("_lineup_provenance"),
                "comment": fixture.get("_comment"),
                "fixture_file": str(fixture_path.relative_to(REPO)),
                "team1_lineup": fixture["team1_lineup"],
                "team2_lineup": fixture["team2_lineup"],
                "toss_withheld": fixture.get("toss_winner") is None,
            },
        })

    sealed = {
        "sealed_for": "The Hundred men's 2026 — unresolved fixtures as of the seal",
        "predeclared_rule": roi["predeclared_rule"],
        "uncertainty_contract": roi["uncertainty_contract"],
        "primary_arm": "i7 (models/xgb_match_i7), toss hidden",
        "secondary_arm": "swap (models/xgb_match_i7_swap_production) — DIAGNOSTIC ONLY",
        "state": {
            "sqlite": "data/hundred/state/player_stats_cache_i7.sqlite (T20 only, "
                      "through 2026-07-13; STALE for these fixtures, override used)",
            "cut0801": "tracker/aux state frozen at 2026-08-01 "
                       "(data/hundred/tracker_snapshot_2026-08-01_aux_hundred_cut_v2.pkl)",
            "full0802": "tracker/aux state through 2026-08-02 "
                        "(data/hundred/tracker_snapshot_2026-08-02_aux_hundred_v2.pkl)",
        },
        "odds_source": str(args.odds.relative_to(REPO)),
        "odds_fetched_at": odds_payload.get("fetched_at"),
        "n_unresolved": len(entries),
        "fixtures": sorted(entries, key=lambda e: (e["date"], e["slug"])),
    }

    payload = json.dumps(sealed, indent=2, sort_keys=True) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(payload)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    print(f"Sealed {len(entries)} unresolved fixtures -> {args.out}")
    for e in entries:
        d = e["declared_bet_primary_arm_pretoss"]
        bet = (f"{d['side']} @ {d['price']:.3f}" if d.get("side")
               else f"NO BET ({d['reason']})")
        print(f"  {e['date']}  {e['team1']} v {e['team2']}: "
              f"p_i7(cut0801)={e['model_probabilities']['i7']['cut0801']['p_team1']:.4f}  "
              f"-> {bet}"
              + ("   [SEED LIQUIDITY]" if e["quote"]["seed_liquidity_non_actionable"]
                 else ""))
    print(f"\nsha256({args.out.name}) = {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
