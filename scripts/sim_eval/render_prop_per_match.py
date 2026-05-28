#!/usr/bin/env python3
"""Render per-match prop comparison views from a prop_backtest detail JSON.

Writes one markdown file per match to <out-dir>/<match_id>.md showing
sim predictions side-by-side with actual cricsheet outcomes, plus an
index.md with a hit/miss summary table.

Usage:
    uv run python scripts/sim_eval/render_prop_per_match.py \
        --detail reports/prop_calibration_detail.json \
        --out-dir reports/prop_per_match/
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


MATCH_ID_RX = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<rest>.+)$")


def _parse_match_id(mid: str) -> Tuple[str, str]:
    """Best-effort split into (date, rest) — rest contains teams+venue."""
    m = MATCH_ID_RX.match(mid)
    if not m:
        return ("?", mid)
    return (m.group("date"), m.group("rest").replace("_", " "))


def _top_n(rows: List[dict], by: str, n: int) -> List[dict]:
    return sorted(rows, key=lambda r: -r.get(by, 0))[:n]


def _hit(rows: List[dict]) -> bool:
    """For binary families: did the sim's top-prob row match the y=1 row?"""
    if not rows:
        return False
    sim_top = max(rows, key=lambda r: r.get("p", 0))
    return bool(sim_top.get("y", 0))


def _ou_hit(rows: List[dict]) -> bool:
    """For O/U families: was the sim's higher-confidence side correct?

    Sim picks OVER if p>0.5, UNDER otherwise. Correct if y matches direction."""
    if not rows:
        return False
    # Aggregate over all rows in this family for this match.
    correct = sum(
        1 for r in rows
        if (r["p"] > 0.5) == bool(r["y"])
    )
    return correct == len(rows)


def render_match(match_obs: dict) -> str:
    mid = match_obs["match_id"]
    obs = match_obs["obs"]
    date, header = _parse_match_id(mid)

    L: List[str] = []
    L.append(f"# {mid}")
    L.append("")
    L.append(f"**Date**: {date}    **Match**: {header}")
    L.append("")

    # ---------------- Top scorer ----------------
    L.append("## Top scorer (per team)")
    L.append("")
    by_team_bat: Dict[str, List[dict]] = defaultdict(list)
    for r in obs.get("top_batter", []):
        by_team_bat[r["team"]].append(r)
    for team, rows in by_team_bat.items():
        actual = next((r["name"] for r in rows if r.get("y")), "—")
        top3 = _top_n(rows, "p", 3)
        L.append(f"### {team}")
        L.append("")
        L.append(f"Actual top scorer: **{actual}**")
        L.append("")
        L.append("| rank | sim's pick | P(top) | hit? |")
        L.append("|---:|---|---:|:---:|")
        for i, r in enumerate(top3, 1):
            mark = "✓" if r.get("y") else " "
            L.append(f"| {i} | {r['name']} | {r['p']:.3f} | {mark} |")
        L.append("")

    # ---------------- Top wicket-taker ----------------
    L.append("## Top wicket-taker (per team)")
    L.append("")
    by_team_bw: Dict[str, List[dict]] = defaultdict(list)
    for r in obs.get("top_bowler", []):
        by_team_bw[r["team"]].append(r)
    for team, rows in by_team_bw.items():
        actual = next((r["name"] for r in rows if r.get("y")), "—")
        top3 = _top_n(rows, "p", 3)
        L.append(f"### {team}")
        L.append("")
        L.append(f"Actual top wicket-taker: **{actual}**")
        L.append("")
        L.append("| rank | sim's pick | P(top) | hit? |")
        L.append("|---:|---|---:|:---:|")
        for i, r in enumerate(top3, 1):
            mark = "✓" if r.get("y") else " "
            L.append(f"| {i} | {r['name']} | {r['p']:.3f} | {mark} |")
        L.append("")

    # ---------------- Innings runs ----------------
    L.append("## Innings totals — sim P10/mean/P90 vs actual")
    L.append("")
    # Reconstruct per-team continuous innings stats from team_runs equivalents.
    # The current detail JSON doesn't include a `team_runs_ou` mean directly;
    # use innings_runs_ou_170_5 rows (each has sim_mean + actual).
    runs_rows = obs.get("innings_runs_ou_170_5", [])
    if runs_rows:
        L.append("| team | sim mean | actual | verdict |")
        L.append("|---|---:|---:|:---:|")
        for r in runs_rows:
            verdict = "✓" if abs(r["sim_mean"] - r["actual"]) < 25 else " "
            L.append(
                f"| {r.get('team','')} | {r['sim_mean']:.1f} | {r['actual']:.0f} | {verdict} |"
            )
        L.append("")
        L.append("(Verdict ✓ = within ±25 runs of actual.)")
        L.append("")

    # ---------------- Boundary counts ----------------
    L.append("## Boundary counts — sim mean vs actual")
    L.append("")
    fours_rows = {r.get("team"): r for r in obs.get("team_total_fours_mae", [])}
    sixes_rows = {r.get("team"): r for r in obs.get("team_total_sixes_mae", [])}
    teams = sorted(set(fours_rows) | set(sixes_rows))
    if teams:
        L.append("| team | sim 4s | actual 4s | sim 6s | actual 6s |")
        L.append("|---|---:|---:|---:|---:|")
        for team in teams:
            f = fours_rows.get(team, {})
            s = sixes_rows.get(team, {})
            L.append(
                f"| {team} | "
                f"{f.get('sim_mean', 0):.1f} | {f.get('actual', 0):.0f} | "
                f"{s.get('sim_mean', 0):.1f} | {s.get('actual', 0):.0f} |"
            )
        L.append("")

    # ---------------- Highest individual ----------------
    hi_rows = obs.get("highest_individual_mae", [])
    if hi_rows:
        r = hi_rows[0]
        L.append("## Highest individual score (match-level)")
        L.append("")
        L.append(
            f"- Sim P10 / mean / P90: **{r['sim_p10']:.0f} / "
            f"{r['sim_mean']:.0f} / {r['sim_p90']:.0f}**"
        )
        L.append(f"- Actual: **{r['actual']:.0f}**")
        within = r["sim_p10"] <= r["actual"] <= r["sim_p90"]
        L.append(f"- In sim P10–P90 band: {'✓' if within else '✗'}")
        L.append("")

    # ---------------- Verdict summary ----------------
    L.append("## Verdict summary")
    L.append("")
    L.append("| family | result |")
    L.append("|---|:---:|")
    for fam, label in [
        ("top_batter", "Top scorer (≥1 team correct)"),
        ("top_bowler", "Top wicket-taker (≥1 team correct)"),
    ]:
        L.append(f"| {label} | {'✓' if _hit(obs.get(fam, [])) else '✗'} |")
    # O/U families: percent correct over all (team × line) rows in this match.
    for fam, label in [
        ("innings_runs_ou_170_5", "Innings runs O/U 170.5"),
        ("pp_total_ou_50_5", "Powerplay O/U 50.5"),
        ("team_highest_individual_ou_34_5", "Team top scorer O/U 34.5"),
        ("first_wicket_runs_ou_30_5", "1st-wicket runs O/U 30.5"),
        ("match_total_sixes_ou_15_5", "Match total sixes O/U 15.5"),
        ("highest_over_runs_ou_18_5", "Biggest over O/U 18.5"),
    ]:
        rows = obs.get(fam, [])
        if not rows:
            continue
        correct = sum(1 for r in rows if (r["p"] > 0.5) == bool(r["y"]))
        L.append(f"| {label} | {correct}/{len(rows)} |")
    L.append("")

    return "\n".join(L)


def render_index(all_matches: List[dict]) -> str:
    """Cross-match summary: hit/miss counts per family."""
    L: List[str] = []
    L.append("# Prop predictions — per-match index")
    L.append("")
    L.append(f"{len(all_matches)} matches.")
    L.append("")
    L.append("| match | top batter | top bowler | innings runs O/U 170.5 |")
    L.append("|---|:---:|:---:|:---:|")
    for d in all_matches:
        mid = d["match_id"]
        date, _ = _parse_match_id(mid)
        # Per-team correctness for binary picks.
        tb = defaultdict(list)
        for r in d["obs"].get("top_batter", []):
            tb[r["team"]].append(r)
        tb_hits = sum(
            1 for rows in tb.values()
            if rows and max(rows, key=lambda x: x["p"]).get("y")
        )
        bw = defaultdict(list)
        for r in d["obs"].get("top_bowler", []):
            bw[r["team"]].append(r)
        bw_hits = sum(
            1 for rows in bw.values()
            if rows and max(rows, key=lambda x: x["p"]).get("y")
        )
        runs = d["obs"].get("innings_runs_ou_170_5", [])
        runs_correct = sum(1 for r in runs if (r["p"] > 0.5) == bool(r["y"]))
        L.append(
            f"| [{mid}]({mid}.md) | {tb_hits}/{len(tb)} | "
            f"{bw_hits}/{len(bw)} | {runs_correct}/{len(runs)} |"
        )
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", required=True)
    ap.add_argument("--out-dir", default="reports/prop_per_match")
    args = ap.parse_args()

    with open(args.detail) as f:
        data = json.load(f)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for match_obs in data:
        mid = match_obs["match_id"]
        # Sanitize filename: most match_ids are already safe, but commas/colons happen.
        fname = re.sub(r"[^A-Za-z0-9._\-]", "_", mid) + ".md"
        (out_dir / fname).write_text(render_match(match_obs))

    (out_dir / "index.md").write_text(render_index(data))
    print(f"Rendered {len(data)} matches to {out_dir}")
    print(f"Index: {out_dir}/index.md")


if __name__ == "__main__":
    main()
