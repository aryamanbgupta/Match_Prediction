"""Build a per-match HTML dashboard for IPL 2026: actual result, score,
pre-match polymarket odds, model prediction, our bet, PnL, cumulative ROI.

Joins three sources:
  - cricsheet JSONs (live pool + golden pool) for match metadata + scores
  - betting_odds_polymarket.json + betting_odds_golden.json for market odds
  - test_predictions.json + golden_predictions.json for model probabilities

Bet rule mirrors match_evaluator / blend_eval_json: bet 1 unit flat on the
team with the largest positive edge (model_prob - market_prob), threshold > 0.
PnL = (odds - 1) on a win, -1 on a loss, 0 if no bet.

Output: reports/ipl_2026_dashboard.html (single file, no server, embeds
data + Chart.js via CDN).

Usage:
    uv run python scripts/build_ipl_dashboard.py
"""
from __future__ import annotations

import json
import glob
import html
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from identity_maps import canonicalize_match_id, canonicalize_venue

REPO = Path(__file__).resolve().parent.parent
LIVE_POOL = REPO / 'data' / 't20s_json'
GOLDEN_POOL = REPO / 'data' / 'golden' / 't20s_json'
PRED_TEST = REPO / 'models' / 'xgb_match_v2_clean' / 'test_predictions.json'
PRED_GOLDEN = REPO / 'models' / 'xgb_match_v2_clean' / 'golden_predictions.json'
ODDS_MAIN = REPO / 'betting_odds_polymarket.json'
ODDS_GOLDEN = REPO / 'data' / 'golden' / 'betting_odds_golden.json'
OUT_HTML = REPO / 'reports' / 'ipl_2026_dashboard_clean.html'

EDGE_THRESHOLD = 0.0  # match blend_eval_json.py


def synth_match_id(date: str, team1: str, team2: str, venue: str) -> str:
    """Mirror build_polymarket_odds.build_match_id and
    materialize_match_features._build_match_record."""
    venue = canonicalize_venue(venue)
    return f'{date}_{team1}_{team2}_{venue}'.replace(' ', '_')


def innings_score(innings: dict) -> dict:
    """Return {team, runs, wickets, balls, overs_str} for an innings dict."""
    team = innings.get('team', '?')
    runs = wickets = balls = 0
    for over in innings.get('overs', []):
        for d in over.get('deliveries', []):
            balls += 1
            extras = d.get('extras') or {}
            # Wides + no-balls don't count as a legal ball
            if 'wides' in extras or 'noballs' in extras:
                balls -= 1
            runs += int((d.get('runs') or {}).get('total', 0))
            if d.get('wickets'):
                wickets += len(d['wickets'])
    overs_str = f'{balls // 6}.{balls % 6}'
    return {'team': team, 'runs': runs, 'wickets': wickets,
            'balls': balls, 'overs': overs_str}


def format_outcome(info: dict) -> str:
    o = info.get('outcome') or {}
    if o.get('winner'):
        by = o.get('by') or {}
        if by.get('wickets'):
            return f"{o['winner']} won by {by['wickets']} wickets"
        if by.get('runs'):
            return f"{o['winner']} won by {by['runs']} runs"
        return f"{o['winner']} won"
    if o.get('result') == 'tie':
        elim = o.get('eliminator')
        if elim:
            return f'Tied — {elim} won super over'
        return 'Tied (no eliminator)'
    if o.get('result') == 'no result':
        return 'No result'
    return 'Result unclear'


def collect_ipl_matches() -> list[dict]:
    """Walk both cricsheet pools, filter to IPL 2026, return list sorted
    chronologically with metadata + scores + outcome."""
    seen_ids: set = set()
    rows = []
    for pool_path, pool_label in [(LIVE_POOL, 'live'), (GOLDEN_POOL, 'golden')]:
        for fp in glob.glob(f'{pool_path}/*.json'):
            try:
                with open(fp) as f:
                    d = json.load(f)
            except Exception:
                continue
            info = d.get('info') or {}
            event = info.get('event', {})
            ev_name = event.get('name', '') if isinstance(event, dict) else ''
            if 'Indian Premier League' not in ev_name:
                continue
            date = (info.get('dates') or [''])[0]
            if not date.startswith('2026'):
                continue
            cricsheet_id = os.path.basename(fp).replace('.json', '')
            if cricsheet_id in seen_ids:
                continue
            seen_ids.add(cricsheet_id)

            teams = info.get('teams', [])
            venue = canonicalize_venue(info.get('venue'))
            mid = synth_match_id(date, teams[0], teams[1], venue) if len(teams) == 2 else None

            innings_data = d.get('innings') or []
            scores = [innings_score(inn) for inn in innings_data[:2]]

            rows.append({
                'cricsheet_id': cricsheet_id,
                'date': date,
                'team1': teams[0] if teams else '?',
                'team2': teams[1] if len(teams) > 1 else '?',
                'venue': venue,
                'venue_short': venue.split(',')[0] if venue else 'unknown',
                'event': info.get('event', {}).get('match_number')
                         if isinstance(info.get('event'), dict) else None,
                'match_id': mid,
                'pool': pool_label,
                'scores': scores,
                'actual_winner': (info.get('outcome') or {}).get('winner'),
                'outcome_str': format_outcome(info),
                'toss_winner': (info.get('toss') or {}).get('winner'),
                'toss_decision': (info.get('toss') or {}).get('decision'),
            })
    rows.sort(key=lambda r: (r['date'], r['cricsheet_id']))
    return rows


def load_odds_lookup() -> dict:
    """Build match_id -> odds entry from both polymarket files.

    I15: index by Cricsheet ID too when present; a duplicate key *within* one
    file means a same-day doubleheader is sharing a synthetic id and the
    lookup would silently drop a market — fail closed. Cross-file overlap
    keeps the historical golden-takes-precedence semantics.
    """
    lookup = {}
    for path, label in [(ODDS_MAIN, 'iteration'), (ODDS_GOLDEN, 'golden')]:
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        seen_in_file = set()
        for m in data.get('matches', []):
            entry = dict(m)
            entry['_odds_source'] = label
            keys = {canonicalize_match_id(m['match_id'])}
            if m.get('cricsheet_id'):
                keys.add(canonicalize_match_id(m['cricsheet_id']))
            for key in keys:
                if key in seen_in_file:
                    raise RuntimeError(
                        f"duplicate odds key {key!r} in {path} — rebuild the "
                        "odds file with Cricsheet primary IDs")
                seen_in_file.add(key)
                lookup[key] = entry
    return lookup


def load_predictions() -> dict:
    """Build match_id -> prediction. golden takes precedence (newer model
    state — though for v2_frozen they should be identical for any overlap,
    which there shouldn't be since the cutoffs are disjoint).

    I15: prediction JSONs may be keyed by Cricsheet ID (new) or synthetic id
    (frozen); index every alias each entry carries, but drop any alias that
    two different fixtures share — a doubleheader display id must not
    silently join one fixture's prediction to the other match.
    """
    out = {}
    owners: dict[str, str] = {}
    dropped: set[str] = set()
    for path, label in [(PRED_TEST, 'iteration'), (PRED_GOLDEN, 'golden')]:
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        for mid, pred in data.items():
            entry = dict(pred)
            entry['_pred_source'] = label
            primary = canonicalize_match_id(pred.get('cricsheet_id') or mid)
            keys = {canonicalize_match_id(mid), primary}
            if pred.get('display_match_id'):
                keys.add(canonicalize_match_id(pred['display_match_id']))
            for key in keys:
                prev_owner = owners.get(key)
                if prev_owner is not None and prev_owner != primary:
                    dropped.add(key)
                    out.pop(key, None)
                    continue
                owners[key] = primary
                if key not in dropped:
                    out[key] = entry
    if dropped:
        print(f"  WARN: dropped {len(dropped)} ambiguous doubleheader "
              f"prediction alias(es): {sorted(dropped)[:3]}")
    return out


def compute_bet(pred: dict, odds_entry: dict) -> dict:
    """Apply the bet rule. Returns dict with selected team, edge, decision,
    pnl (if actual known)."""
    if not pred or not odds_entry:
        return {'bet_team': None, 'best_edge': None, 'placed': False,
                'pnl': None, 'note': 'missing prediction or odds'}
    team1, team2 = pred['team1'], pred['team2']
    p1 = pred['p_team1']
    p2 = pred['p_team2']
    odds = odds_entry['odds']['winner']
    market_t1 = 1.0 / odds[team1] if odds.get(team1) else None
    market_t2 = 1.0 / odds[team2] if odds.get(team2) else None
    if market_t1 is None or market_t2 is None:
        return {'bet_team': None, 'best_edge': None, 'placed': False,
                'pnl': None, 'note': 'missing market price'}
    edge_t1 = p1 - market_t1
    edge_t2 = p2 - market_t2
    edges = {team1: edge_t1, team2: edge_t2}
    best_team = max(edges, key=edges.get)
    best_edge = edges[best_team]
    if best_edge <= EDGE_THRESHOLD:
        return {'bet_team': None, 'best_edge': best_edge, 'edges': edges,
                'placed': False, 'pnl': 0.0, 'note': 'edge below threshold'}
    actual = odds_entry.get('actual_winner')
    if not actual:
        # Tied / abandoned — no PnL realisation
        return {'bet_team': best_team, 'best_edge': best_edge, 'edges': edges,
                'placed': True, 'pnl': None, 'note': 'no resolved winner',
                'market_prob': {team1: market_t1, team2: market_t2}}
    if actual == best_team:
        pnl = float(odds[best_team]) - 1.0
    else:
        pnl = -1.0
    return {'bet_team': best_team, 'best_edge': best_edge, 'edges': edges,
            'placed': True, 'pnl': pnl, 'note': '',
            'market_prob': {team1: market_t1, team2: market_t2}}


def build_rows(matches, odds_lookup, preds) -> list[dict]:
    out = []
    cumulative_pnl = 0.0
    cumulative_stake = 0.0
    cumulative_wins = 0
    cumulative_bets = 0
    claimed: dict[int, str] = {}
    for m in matches:
        mid = m['match_id']
        # Join Cricsheet-primary first (I15); the synthetic id remains a
        # legacy fallback for frozen artifacts.
        join_keys = [k for k in (m.get('cricsheet_id'), mid) if k]
        odds_entry = next(
            (odds_lookup[k] for k in join_keys if k in odds_lookup), None)
        pred = next((preds[k] for k in join_keys if k in preds), None)
        # Two rows resolving the same entry = doubleheader sharing one
        # legacy alias; keep the dashboard rendering but never attribute
        # one market/prediction to both matches.
        row_id = m.get('cricsheet_id') or mid
        for label in ('odds', 'pred'):
            obj = odds_entry if label == 'odds' else pred
            if obj is None:
                continue
            prior = claimed.get(id(obj))
            if prior is not None and prior != row_id:
                print(f"  WARN: {label} entry shared by {prior} and {row_id} "
                      "(doubleheader alias) — dropping join for the latter")
                if label == 'odds':
                    odds_entry = None
                else:
                    pred = None
            else:
                claimed[id(obj)] = row_id
        bet = compute_bet(pred, odds_entry) if (odds_entry and pred) else \
              {'bet_team': None, 'best_edge': None, 'placed': False,
               'pnl': None, 'note': 'no prediction or odds'}

        if bet['placed'] and bet['pnl'] is not None:
            cumulative_bets += 1
            cumulative_pnl += bet['pnl']
            cumulative_stake += 1.0
            if bet['pnl'] > 0:
                cumulative_wins += 1

        cum_roi = (cumulative_pnl / cumulative_stake * 100) if cumulative_stake > 0 else 0.0

        # Compose display fields
        score_lines = [f"{s['team']} {s['runs']}/{s['wickets']} ({s['overs']})"
                       for s in m['scores']]

        out.append({
            **m,
            'odds_entry': odds_entry,
            'pred': pred,
            'bet': bet,
            'score_lines': score_lines,
            'cum_pnl': cumulative_pnl,
            'cum_stake': cumulative_stake,
            'cum_roi': cum_roi,
            'cum_bets': cumulative_bets,
            'cum_wins': cumulative_wins,
        })
    return out


def render_html(rows: list[dict]) -> str:
    """Render a single self-contained HTML file."""
    # Aggregate stats
    n_matches = len(rows)
    n_with_odds = sum(1 for r in rows if r['odds_entry'])
    n_with_pred = sum(1 for r in rows if r['pred'])
    n_bets = sum(1 for r in rows if r['bet']['placed'] and r['bet']['pnl'] is not None)
    n_wins = sum(1 for r in rows if r['bet']['placed'] and (r['bet']['pnl'] or 0) > 0)
    total_pnl = sum((r['bet']['pnl'] or 0) for r in rows
                    if r['bet']['placed'] and r['bet']['pnl'] is not None)
    total_stake = float(n_bets)
    overall_roi = (total_pnl / total_stake * 100) if total_stake > 0 else 0.0
    win_rate = (n_wins / n_bets * 100) if n_bets > 0 else 0.0

    # Per-set sub-aggregates
    set_stats = defaultdict(lambda: {'n_bets': 0, 'pnl': 0.0, 'wins': 0})
    for r in rows:
        if r['bet']['placed'] and r['bet']['pnl'] is not None:
            label = r['pred']['_pred_source'] if r.get('pred') else 'unknown'
            set_stats[label]['n_bets'] += 1
            set_stats[label]['pnl'] += r['bet']['pnl']
            if r['bet']['pnl'] > 0:
                set_stats[label]['wins'] += 1

    # Build chart data
    chart_labels = json.dumps([r['date'] for r in rows])
    chart_cum_pnl = json.dumps([round(r['cum_pnl'], 3) for r in rows])
    chart_cum_roi = json.dumps([round(r['cum_roi'], 2) for r in rows])

    # Build table rows
    table_rows = []
    for i, r in enumerate(rows, 1):
        bet = r['bet']
        pred = r['pred']
        odds_entry = r['odds_entry']

        # Prediction column
        if pred:
            p1 = pred['p_team1']; p2 = pred['p_team2']
            pred_html = (f"<div><b>{html.escape(pred['team1'])}</b>: {p1*100:.1f}%</div>"
                         f"<div><b>{html.escape(pred['team2'])}</b>: {p2*100:.1f}%</div>"
                         f"<div class='pred-source'>{pred['_pred_source']}</div>")
        else:
            pred_html = "<span class='dim'>—</span>"

        # Market column
        if odds_entry:
            o = odds_entry['odds']['winner']
            t1 = odds_entry['team1']; t2 = odds_entry['team2']
            mp1 = (1.0 / o[t1]) * 100 if o.get(t1) else 0
            mp2 = (1.0 / o[t2]) * 100 if o.get(t2) else 0
            vol = odds_entry.get('polymarket_volume_usd', 0) or 0
            market_html = (f"<div><b>{html.escape(t1)}</b>: {o[t1]:.2f} ({mp1:.1f}%)</div>"
                           f"<div><b>{html.escape(t2)}</b>: {o[t2]:.2f} ({mp2:.1f}%)</div>"
                           f"<div class='vol'>vol ${vol:,.0f}</div>")
        else:
            market_html = "<span class='dim'>—</span>"

        # Bet column
        if bet['placed']:
            bet_team = bet['bet_team']
            edge_pct = bet['best_edge'] * 100
            decimal = (odds_entry['odds']['winner'].get(bet_team) if odds_entry else None)
            bet_html = (f"<div><b>{html.escape(bet_team or '')}</b></div>"
                        f"<div class='edge'>edge +{edge_pct:.1f}pp @ {decimal:.2f}</div>"
                        if decimal else f"<div><b>{html.escape(bet_team or '')}</b></div>"
                        f"<div class='edge'>edge +{edge_pct:.1f}pp</div>")
        elif pred and odds_entry:
            best_edge = bet.get('best_edge')
            be_str = f"{best_edge*100:+.1f}pp" if best_edge is not None else "?"
            bet_html = f"<span class='dim'>no bet</span><div class='edge'>(best edge {be_str})</div>"
        else:
            bet_html = f"<span class='dim'>—</span><div class='note'>{html.escape(bet.get('note',''))}</div>"

        # Result column
        result_html = f"<div class='result-line'>{html.escape(r['outcome_str'])}</div>"
        for sl in r['score_lines']:
            result_html += f"<div class='score'>{html.escape(sl)}</div>"

        # PnL column
        if bet['placed'] and bet['pnl'] is not None:
            pnl = bet['pnl']
            cls = 'pnl-win' if pnl > 0 else ('pnl-loss' if pnl < 0 else '')
            pnl_html = f"<span class='{cls}'>{pnl:+.3f}</span>"
        elif bet['placed']:
            pnl_html = "<span class='dim'>(unresolved)</span>"
        else:
            pnl_html = "<span class='dim'>0</span>"

        # Cumulative
        cum_html = (f"<div>{r['cum_pnl']:+.2f}</div>"
                    f"<div class='dim'>ROI {r['cum_roi']:+.1f}%</div>"
                    f"<div class='dim'>{r['cum_wins']}/{r['cum_bets']}</div>")

        # Row colour by win/loss
        row_cls = ''
        if bet['placed']:
            if (bet['pnl'] or 0) > 0:
                row_cls = 'row-win'
            elif (bet['pnl'] or 0) < 0:
                row_cls = 'row-loss'

        # Pool tag
        pool_tag = (f"<span class='tag tag-{r['pool']}'>{r['pool']}</span>"
                    if r.get('pool') else '')

        # Toss
        toss = ''
        if r.get('toss_winner'):
            toss = f"{html.escape(r['toss_winner'])} elected to {r.get('toss_decision','?')}"

        table_rows.append(f"""
        <tr class='{row_cls}'>
          <td>{i}</td>
          <td>{r['date']}<br><span class='dim'>{html.escape(r['venue_short'])}</span><br>{pool_tag}</td>
          <td>
            <div class='matchup'><b>{html.escape(r['team1'])}</b><br>vs<br><b>{html.escape(r['team2'])}</b></div>
            <div class='dim toss'>{toss}</div>
          </td>
          <td>{result_html}</td>
          <td>{market_html}</td>
          <td>{pred_html}</td>
          <td>{bet_html}</td>
          <td class='pnl-cell'>{pnl_html}</td>
          <td class='cum-cell'>{cum_html}</td>
        </tr>""")

    # Compose summary panel
    set_stat_rows = []
    for label, s in sorted(set_stats.items()):
        if s['n_bets'] == 0:
            continue
        roi = s['pnl'] / s['n_bets'] * 100
        wr = s['wins'] / s['n_bets'] * 100
        set_stat_rows.append(
            f"<tr><td>{html.escape(label)}</td>"
            f"<td>{s['n_bets']}</td>"
            f"<td>{s['wins']}</td>"
            f"<td>{wr:.1f}%</td>"
            f"<td>{s['pnl']:+.2f}</td>"
            f"<td>{roi:+.1f}%</td></tr>")
    set_stat_table = '\n'.join(set_stat_rows) or "<tr><td colspan='6'>(no resolved bets yet)</td></tr>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>IPL 2026 — model dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  body {{ font-family: -apple-system, system-ui, Segoe UI, sans-serif;
         margin: 0; padding: 24px; background: #fafbfc; color: #222; }}
  h1 {{ margin: 0 0 4px 0; }}
  .sub {{ color: #666; margin-bottom: 24px; }}
  .panels {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px;
             margin-bottom: 24px; }}
  .panel {{ background: white; border: 1px solid #e1e4e8; border-radius: 8px;
            padding: 16px; }}
  .panel h2 {{ margin: 0 0 12px 0; font-size: 14px; color: #586069;
               text-transform: uppercase; letter-spacing: 0.06em; }}
  .big {{ font-size: 32px; font-weight: 600; }}
  .big-pos {{ color: #28a745; }}
  .big-neg {{ color: #d73a49; }}
  .stat-row {{ display: flex; justify-content: space-between; margin: 4px 0;
               font-size: 14px; }}
  .stat-row .lbl {{ color: #666; }}
  .chart-wrap {{ background: white; border: 1px solid #e1e4e8; border-radius: 8px;
                 padding: 16px; margin-bottom: 24px; height: 320px; }}
  table {{ width: 100%; border-collapse: collapse; background: white;
           border: 1px solid #e1e4e8; border-radius: 8px; overflow: hidden; }}
  th, td {{ padding: 10px 12px; text-align: left; vertical-align: top;
            border-bottom: 1px solid #eaecef; font-size: 13px; }}
  th {{ background: #f6f8fa; font-weight: 600; color: #24292e;
        position: sticky; top: 0; z-index: 1; cursor: pointer;
        user-select: none; }}
  th:hover {{ background: #e1e4e8; }}
  tr.row-win {{ background: #f0fff4; }}
  tr.row-loss {{ background: #fff5f5; }}
  .matchup {{ font-size: 13px; line-height: 1.4; }}
  .toss {{ font-size: 11px; margin-top: 4px; }}
  .dim {{ color: #959da5; font-size: 11px; }}
  .pred-source, .vol, .edge, .note {{ color: #6a737d; font-size: 11px;
                                       margin-top: 3px; font-style: italic; }}
  .result-line {{ font-weight: 600; margin-bottom: 4px; }}
  .score {{ color: #586069; font-size: 11px; font-family: SF Mono, monospace; }}
  .pnl-win {{ color: #28a745; font-weight: 600; }}
  .pnl-loss {{ color: #d73a49; font-weight: 600; }}
  .pnl-cell, .cum-cell {{ font-family: SF Mono, monospace; font-size: 13px; }}
  .tag {{ display: inline-block; padding: 1px 8px; border-radius: 10px;
          font-size: 10px; font-weight: 600; text-transform: uppercase;
          margin-top: 3px; }}
  .tag-live {{ background: #fff5b1; color: #735c0f; }}
  .tag-golden {{ background: #c9e8c8; color: #24523f; }}
  .filter-bar {{ margin-bottom: 12px; }}
  .filter-bar label {{ margin-right: 16px; font-size: 13px; cursor: pointer; }}
  .footnote {{ color: #586069; font-size: 12px; margin-top: 24px;
               border-top: 1px solid #eaecef; padding-top: 16px; }}
</style>
</head>
<body>
<h1>IPL 2026 — model dashboard</h1>
<div class="sub">
  Per-match audit of <code>xgb_match_v2_frozen</code> on every IPL 2026 fixture
  we have data for. Bet rule: 1 unit flat on the team with the largest positive edge
  (edge = model_prob − market_prob), no bet if best edge ≤ 0.
  Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%MZ')}.
</div>

<div class="panels">
  <div class="panel">
    <h2>Total return</h2>
    <div class="big {'big-pos' if total_pnl >= 0 else 'big-neg'}">{total_pnl:+.2f} units</div>
    <div class="stat-row"><span class="lbl">Bets placed</span><span>{n_bets}</span></div>
    <div class="stat-row"><span class="lbl">Wins</span><span>{n_wins}</span></div>
    <div class="stat-row"><span class="lbl">Win rate</span><span>{win_rate:.1f}%</span></div>
    <div class="stat-row"><span class="lbl">Flat ROI</span><span>{overall_roi:+.1f}%</span></div>
  </div>
  <div class="panel">
    <h2>Coverage</h2>
    <div class="big">{n_matches}</div>
    <div class="stat-row"><span class="lbl">IPL 2026 matches found</span><span>{n_matches}</span></div>
    <div class="stat-row"><span class="lbl">With polymarket odds</span><span>{n_with_odds}</span></div>
    <div class="stat-row"><span class="lbl">With model prediction</span><span>{n_with_pred}</span></div>
    <div class="stat-row"><span class="lbl">No-bet (edge ≤ 0)</span><span>{n_with_pred - n_bets}</span></div>
  </div>
  <div class="panel">
    <h2>By prediction set</h2>
    <table style="width:100%; font-size:12px;">
      <thead><tr><th>set</th><th>n</th><th>w</th><th>wr</th><th>pnl</th><th>roi</th></tr></thead>
      <tbody>{set_stat_table}</tbody>
    </table>
    <div style="font-size:11px; color:#586069; margin-top:8px;">
      <span class='tag tag-live'>iteration</span> = test split, model evaluated during selection<br>
      <span class='tag tag-golden'>golden</span> = post-2026-04-17, never seen by tuning
    </div>
  </div>
</div>

<div class="chart-wrap">
  <canvas id="cumChart"></canvas>
</div>

<div class="filter-bar">
  <label><input type="checkbox" id="f-bets" checked> Show only matches we bet on</label>
  <label><input type="checkbox" id="f-golden"> Show only golden set</label>
  <label><input type="checkbox" id="f-loss"> Show only losing bets</label>
</div>

<table id="matchTable">
<thead>
  <tr>
    <th>#</th>
    <th>Date / venue</th>
    <th>Match</th>
    <th>Result</th>
    <th>Polymarket odds</th>
    <th>Our prediction</th>
    <th>Bet placed</th>
    <th>PnL</th>
    <th>Cumulative</th>
  </tr>
</thead>
<tbody>
  {''.join(table_rows)}
</tbody>
</table>

<div class="footnote">
  <b>How to read this</b>: rows are chronological. Green-tinted rows are matches
  we bet on and won; red-tinted rows are bets we lost. "Cumulative" tracks total
  PnL and ROI across only resolved bets — unbet matches don't move the line.
  Predictions and market probs always sum to ~100%; the difference is our edge.
  Bet decimal odds are what we'd be paid per unit staked (PnL = decimal − 1 on win).
  <br><br>
  <b>Caveat</b>: iteration-set IPL matches were part of the test split that
  drove model selection — those numbers are not strictly out-of-sample. Golden-set
  matches never touched any tuning. Compare the two rows in the "By prediction set"
  panel for the cleanest read.
</div>

<script>
// Cumulative chart
const ctx = document.getElementById('cumChart');
new Chart(ctx, {{
  type: 'line',
  data: {{
    labels: {chart_labels},
    datasets: [
      {{ label: 'Cumulative PnL (units)', data: {chart_cum_pnl},
         borderColor: '#0366d6', backgroundColor: 'rgba(3,102,214,0.10)',
         tension: 0.1, yAxisID: 'y' }},
      {{ label: 'Cumulative ROI (%)', data: {chart_cum_roi},
         borderColor: '#28a745', backgroundColor: 'rgba(40,167,69,0.10)',
         tension: 0.1, yAxisID: 'y1', borderDash: [5,5] }},
    ],
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    interaction: {{ mode: 'index', intersect: false }},
    scales: {{
      y:  {{ position: 'left',  title: {{ display: true, text: 'PnL (units)' }} }},
      y1: {{ position: 'right', title: {{ display: true, text: 'ROI (%)' }},
             grid: {{ drawOnChartArea: false }} }},
    }},
  }},
}});

// Sortable table
document.querySelectorAll('#matchTable th').forEach((th, idx) => {{
  let asc = true;
  th.addEventListener('click', () => {{
    const tbody = document.querySelector('#matchTable tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    rows.sort((a, b) => {{
      const av = a.cells[idx].innerText.trim();
      const bv = b.cells[idx].innerText.trim();
      const an = parseFloat(av.replace(/[^\\d.\\-+]/g, ''));
      const bn = parseFloat(bv.replace(/[^\\d.\\-+]/g, ''));
      if (!isNaN(an) && !isNaN(bn)) return asc ? an - bn : bn - an;
      return asc ? av.localeCompare(bv) : bv.localeCompare(av);
    }});
    tbody.innerHTML = '';
    rows.forEach(r => tbody.appendChild(r));
    asc = !asc;
  }});
}});

// Filters
function applyFilters() {{
  const onlyBets = document.getElementById('f-bets').checked;
  const onlyGolden = document.getElementById('f-golden').checked;
  const onlyLoss = document.getElementById('f-loss').checked;
  document.querySelectorAll('#matchTable tbody tr').forEach(tr => {{
    let show = true;
    const cls = tr.className;
    const isBet = cls.includes('row-win') || cls.includes('row-loss');
    const isGolden = tr.innerHTML.includes('tag-golden');
    const isLoss = cls.includes('row-loss');
    if (onlyBets && !isBet) show = false;
    if (onlyGolden && !isGolden) show = false;
    if (onlyLoss && !isLoss) show = false;
    tr.style.display = show ? '' : 'none';
  }});
}}
['f-bets','f-golden','f-loss'].forEach(id =>
  document.getElementById(id).addEventListener('change', applyFilters));
applyFilters();
</script>
</body>
</html>
"""


def main() -> int:
    print('Collecting IPL 2026 matches...')
    matches = collect_ipl_matches()
    print(f'  found {len(matches)} matches')

    print('Loading odds...')
    odds = load_odds_lookup()
    print(f'  {len(odds)} odds entries indexed')

    print('Loading predictions...')
    preds = load_predictions()
    print(f'  {len(preds)} predictions indexed')

    print('Joining + computing bets...')
    rows = build_rows(matches, odds, preds)

    n_bets = sum(1 for r in rows if r['bet']['placed'] and r['bet']['pnl'] is not None)
    total_pnl = sum((r['bet']['pnl'] or 0) for r in rows
                    if r['bet']['placed'] and r['bet']['pnl'] is not None)
    print(f'  {n_bets} bets placed, total PnL {total_pnl:+.2f} units')

    print(f'Rendering HTML...')
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(render_html(rows))
    print(f'  -> {OUT_HTML} ({OUT_HTML.stat().st_size / 1024:.1f} KB)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
