"""Per-match HTML ROI dashboard for the T20 Blast 2026 out-of-sample eval.

The Blast analogue of build_ipl_dashboard.py. Joins three sources:
  - cricsheet JSONs in data/golden_blast/t20s_json/ (metadata + scores + result)
  - data/golden_blast/betting_odds_blast_v2.json (polymarket pre-match odds,
    rebuilt under the fixed head-to-head selection rule — see
    reports/blast_v2_restatement_20260807.md)
  - models/xgb_match_v3_m7_production/blast_golden_predictions.json (model probs)

Bet rule mirrors the production sizing rule (M8) + build_ipl_dashboard: 1 unit
flat on the team with the largest positive edge (model_prob − market_prob),
no bet if best edge ≤ 0. PnL = (decimal − 1) on a win, −1 on a loss.

This is a genuine out-of-sample read: the model never saw the 2026 Blast
season, and the odds are real pre-match polymarket prices.

Output: reports/blast_2026_dashboard.html

Usage:
    uv run python scripts/build_blast_dashboard.py
"""
from __future__ import annotations

import glob
import html
import json
import math
import os
from datetime import datetime
from pathlib import Path

# Reuse the pure helpers from the IPL dashboard so the two stay in lockstep.
import sys
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from build_ipl_dashboard import (  # noqa: E402
    innings_score, format_outcome, synth_match_id, compute_bet, build_rows,
)
from identity_maps import canonicalize_venue  # noqa: E402

POOL = REPO / "data" / "golden_blast" / "t20s_json"
# Odds file of record since the 2026-08-05 market-selection fix. The pre-_v2
# data/golden_blast/betting_odds_blast.json is frozen evidence of what the
# defective builder shipped (reports/market_benchmark_toss_defect_20260805.md,
# restated in reports/blast_v2_restatement_20260807.md).
ODDS = REPO / "data" / "golden_blast" / "betting_odds_blast_v2.json"
PRED = REPO / "models" / "xgb_match_v3_m7_production" / "blast_golden_predictions.json"
OUT_HTML = REPO / "reports" / "blast_2026_dashboard.html"


def collect_blast_matches() -> list[dict]:
    rows = []
    for fp in glob.glob(f"{POOL}/*.json"):
        try:
            with open(fp) as f:
                d = json.load(f)
        except Exception:
            continue
        info = d.get("info") or {}
        ev = info.get("event", {})
        ev_name = ev.get("name", "") if isinstance(ev, dict) else ""
        if "Blast" not in ev_name:
            continue
        date = (info.get("dates") or [""])[0]
        cricsheet_id = os.path.basename(fp).replace(".json", "")
        teams = info.get("teams", [])
        venue = canonicalize_venue(info.get("venue"))
        mid = synth_match_id(date, teams[0], teams[1], venue) if len(teams) == 2 else None
        scores = [innings_score(inn) for inn in (d.get("innings") or [])[:2]]
        rows.append({
            "cricsheet_id": cricsheet_id,
            "date": date,
            "team1": teams[0] if teams else "?",
            "team2": teams[1] if len(teams) > 1 else "?",
            "venue": venue,
            "venue_short": venue.split(",")[0] if venue else "unknown",
            "match_id": mid,
            "pool": "golden",
            "scores": scores,
            "actual_winner": (info.get("outcome") or {}).get("winner"),
            "outcome_str": format_outcome(info),
            "toss_winner": (info.get("toss") or {}).get("winner"),
            "toss_decision": (info.get("toss") or {}).get("decision"),
        })
    rows.sort(key=lambda r: (r["date"], r["cricsheet_id"]))
    return rows


def load_odds_lookup() -> dict:
    """Index each odds entry under every identity it carries.

    The v2 rebuild emits Cricsheet-primary `match_id`s (I15) while the frozen
    pre-v2 file keys on the synthetic display id, and `build_rows` joins on
    `cricsheet_id` first — so index both rather than assuming either.
    """
    out = {}
    if ODDS.exists():
        for m in json.load(open(ODDS)).get("matches", []):
            for key in (m.get("match_id"), m.get("cricsheet_id"),
                        m.get("display_match_id")):
                if key:
                    out[str(key)] = m
    return out


def load_predictions() -> dict:
    out = {}
    if PRED.exists():
        for mid, pred in json.load(open(PRED)).items():
            entry = dict(pred)
            entry["_pred_source"] = "golden"
            out[mid] = entry
    return out


def render_html(rows: list[dict]) -> str:
    n_matches = len(rows)
    n_odds = sum(1 for r in rows if r["odds_entry"])
    n_pred = sum(1 for r in rows if r["pred"])
    resolved = [r for r in rows if r["bet"]["placed"] and r["bet"]["pnl"] is not None]
    n_bets = len(resolved)
    n_wins = sum(1 for r in resolved if r["bet"]["pnl"] > 0)
    total_pnl = sum(r["bet"]["pnl"] for r in resolved)
    roi = (total_pnl / n_bets * 100) if n_bets else 0.0
    win_rate = (n_wins / n_bets * 100) if n_bets else 0.0

    # Model vs market log loss on the joined set.
    model_ll = market_ll = 0.0
    n_ll = 0
    for r in rows:
        pred, oe = r["pred"], r["odds_entry"]
        if not (pred and oe and oe.get("actual_winner")):
            continue
        t1 = pred["team1"]
        y = 1 if oe["actual_winner"] == t1 else 0
        p = min(max(pred["p_team1"], 1e-9), 1 - 1e-9)
        o = oe["odds"]["winner"]
        mp = 1.0 / o[t1] if o.get(t1) else 0.5
        # de-vig market (2-way book) to a clean probability
        mp2 = 1.0 / o[r["team2"]] if o.get(r["team2"]) else 0.5
        s = mp + mp2
        mp = mp / s if s else 0.5
        mp = min(max(mp, 1e-9), 1 - 1e-9)
        model_ll += -(y * math.log(p) + (1 - y) * math.log(1 - p))
        market_ll += -(y * math.log(mp) + (1 - y) * math.log(1 - mp))
        n_ll += 1
    model_ll = model_ll / n_ll if n_ll else 0.0
    market_ll = market_ll / n_ll if n_ll else 0.0

    chart_labels = json.dumps([f"{r['date']} {r['team1'][:3]}-{r['team2'][:3]}" for r in rows])
    chart_pnl = json.dumps([round(r["cum_pnl"], 3) for r in rows])
    chart_roi = json.dumps([round(r["cum_roi"], 2) for r in rows])

    table_rows = []
    for i, r in enumerate(rows, 1):
        bet, pred, oe = r["bet"], r["pred"], r["odds_entry"]
        if pred:
            pred_html = (f"<div><b>{html.escape(pred['team1'])}</b>: {pred['p_team1']*100:.1f}%</div>"
                         f"<div><b>{html.escape(pred['team2'])}</b>: {pred['p_team2']*100:.1f}%</div>")
        else:
            pred_html = "<span class='dim'>—</span>"
        if oe:
            o = oe["odds"]["winner"]; t1 = oe["team1"]; t2 = oe["team2"]
            mp1 = (1.0 / o[t1]) * 100 if o.get(t1) else 0
            mp2 = (1.0 / o[t2]) * 100 if o.get(t2) else 0
            vol = oe.get("polymarket_volume_usd", 0) or 0
            market_html = (f"<div><b>{html.escape(t1)}</b>: {o[t1]:.2f} ({mp1:.0f}%)</div>"
                           f"<div><b>{html.escape(t2)}</b>: {o[t2]:.2f} ({mp2:.0f}%)</div>"
                           f"<div class='vol'>vol ${vol:,.0f}</div>")
        else:
            market_html = "<span class='dim'>—</span>"
        if bet["placed"]:
            decimal = oe["odds"]["winner"].get(bet["bet_team"]) if oe else None
            dstr = f" @ {decimal:.2f}" if decimal else ""
            bet_html = (f"<div><b>{html.escape(bet['bet_team'] or '')}</b></div>"
                        f"<div class='edge'>edge +{bet['best_edge']*100:.1f}pp{dstr}</div>")
        elif pred and oe:
            be = bet.get("best_edge")
            bet_html = (f"<span class='dim'>no bet</span>"
                        f"<div class='edge'>(best {be*100:+.1f}pp)</div>" if be is not None
                        else "<span class='dim'>no bet</span>")
        else:
            bet_html = "<span class='dim'>—</span>"
        result_html = f"<div class='result-line'>{html.escape(r['outcome_str'])}</div>"
        for sl in [f"{s['team']} {s['runs']}/{s['wickets']} ({s['overs']})" for s in r["scores"]]:
            result_html += f"<div class='score'>{html.escape(sl)}</div>"
        if bet["placed"] and bet["pnl"] is not None:
            cls = "pnl-win" if bet["pnl"] > 0 else ("pnl-loss" if bet["pnl"] < 0 else "")
            pnl_html = f"<span class='{cls}'>{bet['pnl']:+.3f}</span>"
        else:
            pnl_html = "<span class='dim'>0</span>"
        cum_html = (f"<div>{r['cum_pnl']:+.2f}</div>"
                    f"<div class='dim'>ROI {r['cum_roi']:+.1f}%</div>"
                    f"<div class='dim'>{r['cum_wins']}/{r['cum_bets']}</div>")
        row_cls = ""
        if bet["placed"]:
            row_cls = "row-win" if (bet["pnl"] or 0) > 0 else ("row-loss" if (bet["pnl"] or 0) < 0 else "")
        toss = (f"{html.escape(r['toss_winner'])} elected to {r.get('toss_decision','?')}"
                if r.get("toss_winner") else "")
        table_rows.append(f"""
        <tr class='{row_cls}'>
          <td>{i}</td>
          <td>{r['date']}<br><span class='dim'>{html.escape(r['venue_short'])}</span></td>
          <td><div class='matchup'><b>{html.escape(r['team1'])}</b><br>vs<br><b>{html.escape(r['team2'])}</b></div>
              <div class='dim toss'>{toss}</div></td>
          <td>{result_html}</td>
          <td>{market_html}</td>
          <td>{pred_html}</td>
          <td>{bet_html}</td>
          <td class='pnl-cell'>{pnl_html}</td>
          <td class='cum-cell'>{cum_html}</td>
        </tr>""")

    ll_verdict = ("beats" if model_ll < market_ll else "trails")
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>T20 Blast 2026 — model ROI dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  body {{ font-family: -apple-system, system-ui, Segoe UI, sans-serif; margin:0; padding:24px; background:#fafbfc; color:#222; }}
  h1 {{ margin:0 0 4px 0; }} .sub {{ color:#666; margin-bottom:24px; }}
  .panels {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; margin-bottom:24px; }}
  .panel {{ background:#fff; border:1px solid #e1e4e8; border-radius:8px; padding:16px; }}
  .panel h2 {{ margin:0 0 12px 0; font-size:14px; color:#586069; text-transform:uppercase; letter-spacing:.06em; }}
  .big {{ font-size:32px; font-weight:600; }} .big-pos {{ color:#28a745; }} .big-neg {{ color:#d73a49; }}
  .stat-row {{ display:flex; justify-content:space-between; margin:4px 0; font-size:14px; }}
  .stat-row .lbl {{ color:#666; }}
  .chart-wrap {{ background:#fff; border:1px solid #e1e4e8; border-radius:8px; padding:16px; margin-bottom:24px; height:320px; }}
  table {{ width:100%; border-collapse:collapse; background:#fff; border:1px solid #e1e4e8; border-radius:8px; overflow:hidden; }}
  th,td {{ padding:10px 12px; text-align:left; vertical-align:top; border-bottom:1px solid #eaecef; font-size:13px; }}
  th {{ background:#f6f8fa; font-weight:600; color:#24292e; }}
  tr.row-win {{ background:#f0fff4; }} tr.row-loss {{ background:#fff5f5; }}
  .matchup {{ font-size:13px; line-height:1.4; }} .toss {{ font-size:11px; margin-top:4px; }}
  .dim {{ color:#959da5; font-size:11px; }}
  .vol,.edge {{ color:#6a737d; font-size:11px; margin-top:3px; font-style:italic; }}
  .result-line {{ font-weight:600; margin-bottom:4px; }}
  .score {{ color:#586069; font-size:11px; font-family:SF Mono, monospace; }}
  .pnl-win {{ color:#28a745; font-weight:600; }} .pnl-loss {{ color:#d73a49; font-weight:600; }}
  .pnl-cell,.cum-cell {{ font-family:SF Mono, monospace; font-size:13px; }}
  .footnote {{ color:#586069; font-size:12px; margin-top:24px; border-top:1px solid #eaecef; padding-top:16px; }}
</style></head><body>
<h1>T20 Blast 2026 — model ROI dashboard</h1>
<div class="sub">
  Out-of-sample audit of <code>xgb_match_v3_m7_production</code> on the 2026
  Vitality Blast. The model never saw the 2026 season; odds are real pre-match
  polymarket prices. Bet rule: 1 unit flat on the largest positive edge
  (model − market), no bet if best edge ≤ 0. Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%MZ')}.
</div>
<div class="panels">
  <div class="panel">
    <h2>Total return</h2>
    <div class="big {'big-pos' if total_pnl >= 0 else 'big-neg'}">{total_pnl:+.2f} units</div>
    <div class="stat-row"><span class="lbl">Bets placed</span><span>{n_bets}</span></div>
    <div class="stat-row"><span class="lbl">Wins</span><span>{n_wins}</span></div>
    <div class="stat-row"><span class="lbl">Win rate</span><span>{win_rate:.1f}%</span></div>
    <div class="stat-row"><span class="lbl">Flat ROI</span><span>{roi:+.1f}%</span></div>
  </div>
  <div class="panel">
    <h2>Coverage</h2>
    <div class="big">{n_matches}</div>
    <div class="stat-row"><span class="lbl">Blast 2026 matches</span><span>{n_matches}</span></div>
    <div class="stat-row"><span class="lbl">With polymarket odds</span><span>{n_odds}</span></div>
    <div class="stat-row"><span class="lbl">With model prediction</span><span>{n_pred}</span></div>
  </div>
  <div class="panel">
    <h2>Model vs market (log loss)</h2>
    <div class="big {'big-pos' if model_ll < market_ll else 'big-neg'}">{model_ll:.4f}</div>
    <div class="stat-row"><span class="lbl">Model LL</span><span>{model_ll:.4f}</span></div>
    <div class="stat-row"><span class="lbl">Market LL (de-vigged)</span><span>{market_ll:.4f}</span></div>
    <div class="stat-row"><span class="lbl">Coinflip</span><span>0.6931</span></div>
    <div class="stat-row"><span class="lbl">Verdict</span><span>model {ll_verdict} market</span></div>
  </div>
</div>
<div class="chart-wrap"><canvas id="cumChart"></canvas></div>
<table>
<thead><tr><th>#</th><th>Date / venue</th><th>Match</th><th>Result</th>
<th>Polymarket odds</th><th>Our prediction</th><th>Bet placed</th><th>PnL</th><th>Cumulative</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody></table>
<div class="footnote">
  <b>How to read this</b>: rows are chronological. Green = bet won, red = bet lost.
  In a 2-way market exactly one side carries positive edge at threshold 0, so the
  model bets every match. "Cumulative" tracks PnL/ROI across resolved bets.
  <br><br>
  <b>Caveat</b>: n={n_bets} is a tiny sample (3 match-days, 2026-05-22→25) — treat
  ROI as directional, not a verdict. The wide swings on the chart are single-bet
  variance. Pure out-of-sample: the 2026 Blast was never in training/selection.
</div>
<script>
new Chart(document.getElementById('cumChart'), {{
  type:'line',
  data:{{ labels:{chart_labels}, datasets:[
    {{ label:'Cumulative PnL (units)', data:{chart_pnl}, borderColor:'#0366d6',
       backgroundColor:'rgba(3,102,214,0.10)', tension:0.1, yAxisID:'y' }},
    {{ label:'Cumulative ROI (%)', data:{chart_roi}, borderColor:'#28a745',
       backgroundColor:'rgba(40,167,69,0.10)', tension:0.1, yAxisID:'y1', borderDash:[5,5] }} ]}},
  options:{{ responsive:true, maintainAspectRatio:false,
    interaction:{{ mode:'index', intersect:false }},
    scales:{{ y:{{ position:'left', title:{{ display:true, text:'PnL (units)' }} }},
      y1:{{ position:'right', title:{{ display:true, text:'ROI (%)' }}, grid:{{ drawOnChartArea:false }} }} }} }}
}});
</script>
</body></html>
"""


def main() -> int:
    print("Collecting Blast 2026 matches...")
    matches = collect_blast_matches()
    print(f"  found {len(matches)} matches")
    odds = load_odds_lookup()
    preds = load_predictions()
    print(f"  {len(odds)} odds entries, {len(preds)} predictions")
    rows = build_rows(matches, odds, preds)
    resolved = [r for r in rows if r["bet"]["placed"] and r["bet"]["pnl"] is not None]
    total_pnl = sum(r["bet"]["pnl"] for r in resolved)
    print(f"  {len(resolved)} bets, total PnL {total_pnl:+.2f} units, "
          f"ROI {(total_pnl/len(resolved)*100) if resolved else 0:+.1f}%")
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(render_html(rows))
    print(f"  -> {OUT_HTML} ({OUT_HTML.stat().st_size/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
