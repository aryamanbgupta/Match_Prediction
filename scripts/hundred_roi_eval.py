"""Predeclared flat-stake betting evaluation of the match model on The Hundred 2026.

Joins model predictions (from `backtest_hundred.py` for played matches, and from
`predict_fixture.py` for forward fixtures) against the Polymarket odds pull, then
applies ONE betting rule, fixed before any number was looked at:

    p_m = model P(team1); q = market P(team1) at the PRE-TOSS quote.
    p_m > q  -> 1 unit on team1 at price q         (win +1/q - 1, lose -1)
    p_m < q  -> 1 unit on team2 at price (1 - q)   (win +1/(1-q) - 1, lose -1)
    p_m == q, or a missing quote -> skip.
    Flat stakes. No edge threshold. No sizing rule.

Everything else in this file is a DIAGNOSTIC and is labelled as such: the
post-toss price basis, the min-edge threshold ladder, the favourite/underdog
split, and the volume note. They exist to explain the headline, never to
replace it.

Uncertainty: one tournament is one I3 tournament-time block, so no confidence
interval is computed or reported. The only inferential statistic is an exact
Poisson-binomial tail probability on the bet win count under the market's own
per-bet probabilities, which is i.i.d.-optimistic (it ignores block structure)
and is labelled that way everywhere it appears.

Deterministic; no network access.

Usage:
    uv run python scripts/hundred_roi_eval.py \
        --odds data/hundred/polymarket_odds_2026_v2.json \
        --arm i7=eval_out/hundred_roi_2026-08-03/preds_i7.json \
        --arm swap=eval_out/hundred_roi_2026-08-03/preds_swap.json \
        --forward-arm i7=eval_out/hundred_roi_2026-08-03/preds_i7_cutaux0801.json \
        --out-json eval_out/hundred_roi_2026-08-03/roi_eval.json

Settling fixtures that were still unresolved when the odds pull was taken adds
`--forward-results data/hundred/forward_results_<date>.json`. That file carries
WINNERS ONLY, so it can fill a result but can never touch a sealed model
probability or a recorded price. Omitting the flag reproduces the pre-settlement
run exactly.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

COINFLIP_LL = -math.log(0.5)
BACKTEST_END = "2026-08-01"   # slice (a): played matches up to and including
FORWARD_START = "2026-08-02"  # slice (b): state frozen at 2026-08-01
EDGE_LADDER = (0.02, 0.05, 0.10)
EPS = 1e-15


# ---------------------------------------------------------------- joining


def load_aliases(path: Path | None) -> dict[str, str]:
    if not path:
        return {}
    return json.loads(Path(path).read_text())["aliases"]


def canon(team: str, aliases: dict[str, str]) -> str:
    return aliases.get(team, team)


def team_key(date: str, t1: str, t2: str, aliases: dict[str, str]) -> tuple:
    """Order-insensitive fixture key so a team1/team2 flip still joins."""
    return (date, frozenset({canon(t1, aliases), canon(t2, aliases)}))


def load_odds(path: Path, aliases: dict[str, str]) -> tuple[dict, dict]:
    """Return (by_cricsheet_id, by_date_teams) maps of odds rows."""
    payload = json.loads(Path(path).read_text())
    by_id, by_key = {}, {}
    for row in payload["matches"]:
        if row.get("cricsheet_match_id"):
            by_id[str(row["cricsheet_match_id"])] = row
        by_key[team_key(row["date"], row["team1"], row["team2"], aliases)] = row
    return by_id, by_key


def load_forward_results(path: Path | None,
                         aliases: dict[str, str]) -> tuple[dict, dict]:
    """Late-arriving results for fixtures that were unresolved at seal time.

    The file carries WINNERS ONLY -- never a model probability and never a
    price -- so ingesting it cannot alter a sealed prediction or a quote. Rows
    are keyed by cricsheet id when known and always by (date, {teams}).
    """
    if not path:
        return {}, {}
    payload = json.loads(Path(path).read_text())
    by_id, by_key = {}, {}
    for row in payload["results"]:
        if row.get("cricsheet_match_id"):
            by_id[str(row["cricsheet_match_id"])] = row
        by_key[team_key(row["date"], row["team1"], row["team2"], aliases)] = row
    return by_id, by_key


def load_backtest_preds(path: Path) -> list[dict]:
    payload = json.loads(Path(path).read_text())
    return payload["matches"]


def load_fixture_pred(path: Path) -> dict:
    """One `predict_fixture.py` output -> a row shaped like a backtest row."""
    payload = json.loads(Path(path).read_text())
    fixture = payload["fixture"]
    team1 = fixture["team1"]
    return {
        "match_id": None,
        "date": fixture["date"],
        "team1": team1,
        "team2": fixture["team2"],
        "venue": fixture["venue"],
        "winner": None,          # filled from the odds row if the result is in
        "p_team1": payload["prediction"][team1],
        "source": "predict_fixture",
        "source_path": str(path),
        "lineup_provenance": fixture.get("_lineup_provenance"),
        "state_freshness": payload["diagnostics"]["state_freshness"]["status"],
        "tracker_snapshot": payload["diagnostics"]["tracker_snapshot"],
        "tracker_aux_match_count":
            payload["diagnostics"]["tracker_aux_match_count"],
    }


# ---------------------------------------------------------------- the rule


def settle(p_model: float, q_market: float, team1: str, team2: str,
           winner: str | None) -> dict | None:
    """Apply the predeclared rule to one fixture. None = no bet / not settled."""
    if q_market is None or p_model is None:
        return None
    if p_model == q_market:
        return None
    if p_model > q_market:
        side, price = team1, q_market
    else:
        side, price = team2, 1.0 - q_market
    row = {
        "side": side,
        "price": price,
        "decimal_odds": 1.0 / price if price > 0 else None,
        "edge": abs(p_model - q_market),
        "backed_market_underdog": price < 0.5,
        "won": None,
        "pnl": None,
    }
    if winner is not None:
        row["won"] = winner == side
        row["pnl"] = (1.0 / price - 1.0) if row["won"] else -1.0
    return row


# ---------------------------------------------------------------- metrics


def _ll(p: float, y: int) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def poisson_binomial_tail(probs: list[float], k: int) -> float:
    """Exact P(X >= k) for independent Bernoulli trials with these probs.

    i.i.d.-OPTIMISTIC: it treats every bet as independent, which a single
    tournament's fixtures are not (shared squads, shared conditions, shared
    schedule). Reported for orientation only.
    """
    dist = [1.0]
    for p in probs:
        nxt = [0.0] * (len(dist) + 1)
        for i, mass in enumerate(dist):
            nxt[i] += mass * (1 - p)
            nxt[i + 1] += mass * p
        dist = nxt
    return sum(dist[k:])


def summarize(ledger: list[dict], basis: str) -> dict:
    """Betting + probability metrics over the settled rows of one slice."""
    settled = [r for r in ledger if r[basis] and r[basis]["won"] is not None]
    scored = [r for r in ledger if r["winner"] is not None
              and r["market_prob_team1"][basis] is not None]

    out: dict = {"n_fixtures_in_slice": len(ledger),
                 "n_scored": len(scored),
                 "bets": len(settled)}

    if settled:
        wins = sum(1 for r in settled if r[basis]["won"])
        pnl = sum(r[basis]["pnl"] for r in settled)
        out.update({
            "wins": wins,
            "win_rate": wins / len(settled),
            "total_pnl_units": pnl,
            "flat_roi_pct": 100.0 * pnl / len(settled),
            "market_implied_expected_wins":
                sum(r[basis]["price"] for r in settled),
            "sign_test_p_ge_wins_iid_optimistic": poisson_binomial_tail(
                [r[basis]["price"] for r in settled], wins),
        })
    else:
        out.update({"wins": 0, "win_rate": None, "total_pnl_units": 0.0,
                    "flat_roi_pct": None})

    if scored:
        ys = [1 if r["winner"] == r["team1"] else 0 for r in scored]
        pm = [r["p_team1"] for r in scored]
        qm = [r["market_prob_team1"][basis] for r in scored]
        out.update({
            "model_log_loss": sum(_ll(p, y) for p, y in zip(pm, ys)) / len(ys),
            "market_log_loss": sum(_ll(q, y) for q, y in zip(qm, ys)) / len(ys),
            "coinflip_log_loss": COINFLIP_LL,
            "model_brier": sum((p - y) ** 2 for p, y in zip(pm, ys)) / len(ys),
            "market_brier": sum((q - y) ** 2 for q, y in zip(qm, ys)) / len(ys),
            "model_accuracy": sum(
                1 for p, y in zip(pm, ys) if (p >= 0.5) == (y == 1)) / len(ys),
            "market_accuracy": sum(
                1 for q, y in zip(qm, ys) if (q >= 0.5) == (y == 1)) / len(ys),
            "mean_abs_model_edge_from_half":
                sum(abs(p - 0.5) for p in pm) / len(pm),
            "mean_abs_model_minus_market":
                sum(abs(p - q) for p, q in zip(pm, qm)) / len(pm),
            "model_prob_min": min(pm),
            "model_prob_max": max(pm),
        })

    # --- DIAGNOSTIC: model-only scoring, market quote NOT required ---
    # `scored` above needs a market price so model and market are compared on
    # the same rows. Fixtures with no actionable quote are NO BET, but their
    # directional accuracy is still informative, so score them model-only.
    model_only = [r for r in ledger if r["winner"] is not None]
    if model_only:
        ys = [1 if r["winner"] == r["team1"] else 0 for r in model_only]
        pm = [r["p_team1"] for r in model_only]
        out["model_only_DIAGNOSTIC"] = {
            "n_scored": len(model_only),
            "model_log_loss": sum(_ll(p, y) for p, y in zip(pm, ys)) / len(ys),
            "model_brier": sum((p - y) ** 2 for p, y in zip(pm, ys)) / len(ys),
            "model_accuracy": sum(
                1 for p, y in zip(pm, ys) if (p >= 0.5) == (y == 1)) / len(ys),
            "coinflip_log_loss": COINFLIP_LL,
            "per_fixture": [
                {"date": r["date"], "match_id": r["match_id"],
                 "team1": r["team1"], "team2": r["team2"],
                 "p_team1": r["p_team1"], "winner": r["winner"],
                 "model_pick": r["team1"] if r["p_team1"] >= 0.5 else r["team2"],
                 "correct": (r["p_team1"] >= 0.5) == (r["winner"] == r["team1"]),
                 "log_loss": _ll(r["p_team1"],
                                 1 if r["winner"] == r["team1"] else 0),
                 "bet_placed": r[basis] is not None,
                 "winner_source": r.get("winner_source")}
                for r in model_only
            ],
        }

    # --- DIAGNOSTIC: min-edge threshold ladder (never the headline) ---
    ladder = {}
    for thr in EDGE_LADDER:
        sub = [r for r in settled if r[basis]["edge"] >= thr]
        if sub:
            pnl = sum(r[basis]["pnl"] for r in sub)
            ladder[f"{thr*100:.0f}pp"] = {
                "bets": len(sub),
                "wins": sum(1 for r in sub if r[basis]["won"]),
                "total_pnl_units": pnl,
                "flat_roi_pct": 100.0 * pnl / len(sub),
            }
        else:
            ladder[f"{thr*100:.0f}pp"] = {"bets": 0, "wins": 0,
                                          "total_pnl_units": 0.0,
                                          "flat_roi_pct": None}
    out["diagnostic_min_edge_ladder"] = ladder

    # --- DIAGNOSTIC: favourite / underdog split of the bets placed ---
    if settled:
        dogs = [r for r in settled if r[basis]["backed_market_underdog"]]
        favs = [r for r in settled if not r[basis]["backed_market_underdog"]]
        def blk(rows):
            if not rows:
                return {"bets": 0, "wins": 0, "total_pnl_units": 0.0,
                        "flat_roi_pct": None}
            pnl = sum(r[basis]["pnl"] for r in rows)
            return {"bets": len(rows),
                    "wins": sum(1 for r in rows if r[basis]["won"]),
                    "total_pnl_units": pnl,
                    "flat_roi_pct": 100.0 * pnl / len(rows)}
        out["diagnostic_side_split"] = {
            "backed_market_underdog": blk(dogs),
            "backed_market_favourite": blk(favs),
            "underdog_share": len(dogs) / len(settled),
        }

    # --- DIAGNOSTIC: volume note ---
    vols = [r["market_volume_usd"] for r in settled
            if r.get("market_volume_usd") is not None]
    if vols and settled:
        pnl_w = sum(r[basis]["pnl"] * (r.get("market_volume_usd") or 0.0)
                    for r in settled)
        vsum = sum(r.get("market_volume_usd") or 0.0 for r in settled)
        out["diagnostic_volume"] = {
            "n_with_volume": len(vols),
            "total_market_volume_usd": sum(vols),
            "median_market_volume_usd": sorted(vols)[len(vols) // 2],
            "min_market_volume_usd": min(vols),
            "max_market_volume_usd": max(vols),
            "volume_weighted_roi_pct": (100.0 * pnl_w / vsum) if vsum else None,
            "n_at_or_above_50k": sum(1 for v in vols if v >= 50_000),
        }
    return out


# ---------------------------------------------------------------- driver


def build_ledger(rows: list[dict], by_id: dict, by_key: dict,
                 aliases: dict[str, str],
                 res_by_id: dict | None = None,
                 res_by_key: dict | None = None,
                 ) -> tuple[list[dict], list[dict]]:
    ledger, unjoined = [], []
    for row in rows:
        odds = None
        if row.get("match_id"):
            odds = by_id.get(str(row["match_id"]))
        if odds is None:
            odds = by_key.get(team_key(row["date"], row["team1"],
                                       row["team2"], aliases))
        if odds is None:
            unjoined.append({"date": row["date"], "team1": row["team1"],
                             "team2": row["team2"],
                             "reason": "no odds row matched"})
            continue

        # Orient the market probabilities onto the PREDICTION's team1.
        flipped = canon(odds["team1"], aliases) != canon(row["team1"], aliases)

        def orient(value):
            if value is None:
                return None
            return (1.0 - value) if flipped else value

        pretoss = orient(odds.get("pretoss_prob_team1"))
        posttoss = orient(odds.get("prematch_prob_team1"))
        winner = row.get("winner") or odds.get("winner")
        winner_source = ("prediction_row" if row.get("winner")
                         else ("odds_row" if odds.get("winner") else None))

        # Late-arriving result for a fixture that was unresolved at seal time.
        result_row = None
        if winner is None and (res_by_id or res_by_key):
            if row.get("match_id"):
                result_row = (res_by_id or {}).get(str(row["match_id"]))
            if result_row is None:
                result_row = (res_by_key or {}).get(
                    team_key(row["date"], row["team1"], row["team2"], aliases))
            if result_row is not None:
                winner = result_row.get("winner")
                winner_source = "forward_results"

        entry = {
            "match_id": row.get("match_id") or (
                result_row or {}).get("cricsheet_match_id"),
            "date": row["date"],
            "team1": row["team1"],
            "team2": row["team2"],
            "venue": row.get("venue"),
            "winner": winner,
            "p_team1": row["p_team1"],
            "market_prob_team1": {"pretoss": pretoss, "posttoss": posttoss},
            "pretoss_price_timestamp": odds.get("pretoss_price_timestamp"),
            "posttoss_price_timestamp": odds.get("prematch_price_timestamp"),
            "scheduled_start_utc": odds.get("scheduled_start_utc"),
            "market_volume_usd": odds.get("market_volume_usd"),
            "event_volume_usd": odds.get("event_volume_usd"),
            "odds_teams_flipped_vs_prediction": flipped,
            "odds_status": odds.get("status"),
            "pred_source": row.get("source", "backtest_hundred"),
            "winner_source": winner_source,
        }
        if result_row is not None:
            entry["result_provenance"] = {
                "margin": result_row.get("margin"),
                "scores": result_row.get("scores"),
                "cricsheet_published": result_row.get("cricsheet_published"),
                "sources": result_row.get("sources"),
            }
        for key, extra in (("lineup_provenance", "lineup_provenance"),
                           ("state_freshness", "state_freshness"),
                           ("tracker_snapshot", "tracker_snapshot"),
                           ("tracker_aux_match_count",
                            "tracker_aux_match_count")):
            if key in row:
                entry[extra] = row[key]
        for basis in ("pretoss", "posttoss"):
            entry[basis] = settle(entry["p_team1"],
                                  entry["market_prob_team1"][basis],
                                  entry["team1"], entry["team2"], winner)
        ledger.append(entry)
    return ledger, unjoined


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--odds", type=Path,
                    default=REPO / "data/hundred/polymarket_odds_2026_v2.json")
    ap.add_argument("--aliases", type=Path,
                    default=REPO / "data/hundred/team_aliases_2026.json")
    ap.add_argument("--arm", action="append", dest="arms", required=True,
                    help="NAME=path to a backtest_hundred.py predictions JSON")
    ap.add_argument("--forward-arm", action="append", dest="forward_arms",
                    default=[],
                    help="NAME=path to the state-frozen-at-2026-08-01 "
                         "predictions JSON used for the FORWARD slice")
    ap.add_argument("--fixture-pred", action="append", dest="fixture_preds",
                    default=[],
                    help="NAME=path to a predict_fixture.py output JSON")
    ap.add_argument("--forward-results", type=Path, default=None,
                    help="Optional JSON of late-arriving WINNERS for fixtures "
                         "that were unresolved when the odds pull was taken "
                         "(keyed by cricsheet id and/or date+teams). Winners "
                         "only -- it can never change a sealed probability or "
                         "a recorded price.")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    aliases = load_aliases(args.aliases)
    by_id, by_key = load_odds(args.odds, aliases)
    res_by_id, res_by_key = load_forward_results(args.forward_results, aliases)

    def split(spec: str) -> tuple[str, Path]:
        name, _, path = spec.partition("=")
        return name, Path(path)

    arms = dict(split(s) for s in args.arms)
    forward_arms = dict(split(s) for s in args.forward_arms)
    fixture_by_arm: dict[str, list[Path]] = {}
    for spec in args.fixture_preds:
        name, path = split(spec)
        fixture_by_arm.setdefault(name, []).append(path)

    result: dict = {
        "predeclared_rule": (
            "p_m > q -> 1 unit on team1 at price q; p_m < q -> 1 unit on team2 "
            "at price (1-q); skip on exact tie or missing quote. Flat stakes, "
            "no edge threshold. q is the PRE-TOSS quote (last trade strictly "
            "before scheduled start minus 60 minutes)."
        ),
        "uncertainty_contract": (
            "One tournament = one I3 tournament_time_block_v1 block, so NO "
            "confidence interval is reported. All results are DESCRIPTIVE. The "
            "only test shown is an exact Poisson-binomial tail on bet wins "
            "under the market's own per-bet probabilities, which is "
            "i.i.d.-optimistic."
        ),
        "slices": {
            "backtest": f"played matches, date <= {BACKTEST_END}",
            "forward": (f"date >= {FORWARD_START}, predicted with tracker/aux "
                        f"state frozen at {BACKTEST_END}"),
        },
        "inputs": {
            "odds": str(args.odds),
            "aliases": str(args.aliases),
            "forward_results": (str(args.forward_results)
                                if args.forward_results else None),
            "arms": {k: str(v) for k, v in arms.items()},
            "forward_arms": {k: str(v) for k, v in forward_arms.items()},
            "fixture_preds": {k: [str(p) for p in v]
                              for k, v in fixture_by_arm.items()},
        },
        "arms": {},
        "unjoined": [],
    }

    for name, path in arms.items():
        rows = load_backtest_preds(path)
        backtest_rows = [r for r in rows if r["date"] <= BACKTEST_END]
        # The FORWARD slice must be scored from state frozen at BACKTEST_END.
        fwd_src = forward_arms.get(name, path)
        fwd_rows = [r for r in load_backtest_preds(fwd_src)
                    if r["date"] >= FORWARD_START]
        for spec_path in fixture_by_arm.get(name, []):
            fwd_rows.append(load_fixture_pred(spec_path))
        fwd_rows.sort(key=lambda r: (r["date"], str(r.get("match_id") or "")))

        bt_ledger, bt_bad = build_ledger(backtest_rows, by_id, by_key, aliases,
                                         res_by_id, res_by_key)
        fw_ledger, fw_bad = build_ledger(fwd_rows, by_id, by_key, aliases,
                                         res_by_id, res_by_key)
        result["unjoined"].extend(
            [{**b, "arm": name, "slice": "backtest"} for b in bt_bad] +
            [{**b, "arm": name, "slice": "forward"} for b in fw_bad])

        combined = bt_ledger + fw_ledger
        result["arms"][name] = {
            "forward_predictions_from": str(fwd_src),
            "ledger": {"backtest": bt_ledger, "forward": fw_ledger},
            "summary": {
                slice_name: {
                    "pretoss_HEADLINE": summarize(led, "pretoss"),
                    "posttoss_DIAGNOSTIC": summarize(led, "posttoss"),
                }
                for slice_name, led in (("backtest", bt_ledger),
                                        ("forward", fw_ledger),
                                        ("combined", combined))
            },
        }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2))

    # ------------------------------------------------------------- console
    for name, arm in result["arms"].items():
        print(f"\n=== ARM {name} ===")
        for slice_name in ("backtest", "forward", "combined"):
            s = arm["summary"][slice_name]["pretoss_HEADLINE"]
            roi = ("n/a" if s["flat_roi_pct"] is None
                   else f"{s['flat_roi_pct']:+7.2f}%")
            print(f"  {slice_name:<9} fixtures={s['n_fixtures_in_slice']:>2} "
                  f"scored={s['n_scored']:>2} bets={s['bets']:>2} "
                  f"wins={s['wins']:>2} "
                  f"pnl={s['total_pnl_units']:+7.3f}u roi={roi}")
            if s.get("model_log_loss") is not None:
                print(f"            LL model={s['model_log_loss']:.4f} "
                      f"market={s['market_log_loss']:.4f} "
                      f"coinflip={COINFLIP_LL:.4f}")
            mo = s.get("model_only_DIAGNOSTIC")
            if mo and mo["n_scored"] != s["n_scored"]:
                print(f"            model-only (no quote needed) "
                      f"n={mo['n_scored']} LL={mo['model_log_loss']:.4f} "
                      f"acc={mo['model_accuracy']:.3f}")
    if result["unjoined"]:
        print("\nUNJOINED:")
        for row in result["unjoined"]:
            print("  ", row)
    print(f"\nWrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
