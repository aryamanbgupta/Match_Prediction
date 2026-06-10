"""Weighted factor (TreeSHAP) report for the two validated MLC head-to-heads.

Holds best XIs (5 USA-developed + 6 overseas) for ALL SIX teams, then for the
two Dallas-leg matches with both sides validated (LAKR v SFU, TSK v SFU) it:
  * predicts the win % with both teams at best XI, and
  * decomposes that prediction with exact XGBoost TreeSHAP contributions,
    grouped into interpretable factors, so each factor's push (in log-odds)
    sums to the model's margin.

For each factor we show the team1-vs-team2 difference and how hard (and which
way) it pushes the result, plus its share of the total movement ("weight").

Usage:
    uv run python mlc/scripts/factor_report.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from predict_fixture import (  # noqa: E402
    MODEL_DIR, load_trackers, compute_features,
)
from player_metadata import PlayerMetadataProvider  # noqa: E402
from stats_provider import StatsProvider  # noqa: E402

ROSTER = Path("/Users/aryamangupta/Projects/cric-analysis/mlc-2026/mlc_2026_rosters.csv")
FIX_DIR = REPO / "fixtures" / "mlc_2026"
ABBR = {"MI New York": "MINY", "Texas Super Kings": "TSK", "Washington Freedom": "WSH",
        "San Francisco Unicorns": "SFU", "Seattle Orcas": "ORCA",
        "Los Angeles Knight Riders": "LAKR"}

BEST_XI = {
    "San Francisco Unicorns": [
        "Finn Allen", "Matt Short", "Lhuan-dre Pretorius", "Sanjay Krishnamurthi",
        "Hammad Azam", "Hassan Khan", "Ravichandran Ashwin", "Xavier Bartlett",
        "Haris Rauf", "Zia-ul-Haq", "Juanoy Drysdale"],
    "Texas Super Kings": [
        "Faf du Plessis", "Saiteja Mukkamalla", "Rilee Rossouw", "Donovan Ferreira",
        "Shubham Ranjane", "Calvin Savage", "Milind Kumar", "Akeal Hosein",
        "Adam Milne", "Hardus Viljoen", "Mohammad Mohsin"],
    "Los Angeles Knight Riders": [
        "Alex Hales", "Sunil Narine", "Unmukt Chand", "Rovman Powell",
        "Andre Russell", "Saif Badar", "Jason Holder", "Jahmar Hamilton",
        "Shadley van Schalkwyk", "Karthik Gattepalli", "Ali Khan"],
    "MI New York": [
        "Quinton de Kock", "Ryan Rickelton", "Nicholas Pooran", "Monank Patel",
        "Kieron Pollard", "Romario Shepherd", "Corey Anderson", "Tajinder Singh",
        "Trent Boult", "Nosthush Kenjige", "Rushil Ugarkar"],
    "Washington Freedom": [
        "Andries Gous", "Steven Smith", "Mitchell Owen", "Glenn Maxwell",
        "Rachin Ravindra", "Obus Pienaar", "Nikhil Chaudhary", "Marco Jansen",
        "Lockie Ferguson", "Ian Holland", "Saurabh Netravalkar"],
    "Seattle Orcas": [
        "Tim Seifert", "Shimron Hetmyer", "Tim Robinson", "Dasun Shanaka",
        "Shayan Jahangir", "Marcus Stoinis", "Harmeet Singh", "Lungi Ngidi",
        "Ali Sheikh", "Ayan Desai", "Jasdeep Singh"],
}

MATCHES = [
    ("2026-06-19_lakr_sfu", "Los Angeles Knight Riders", "San Francisco Unicorns"),
    ("2026-06-20_tsk_sfu", "Texas Super Kings", "San Francisco Unicorns"),
]

# factor group -> member features
GROUPS = {
    "Top-6 batting (ELO)": ["team1_top6_batting_elo_avg", "team2_top6_batting_elo_avg",
                            "top6_batting_elo_diff"],
    "Bottom-5 bowling (ELO)": ["team1_bottom5_bowling_elo_avg", "team2_bottom5_bowling_elo_avg",
                               "bottom5_bowling_elo_diff"],
    "Whole-XI batting (ELO)": ["team1_batting_elo", "team2_batting_elo", "elo_diff_batting"],
    "Whole-XI bowling (ELO)": ["team1_bowling_elo", "team2_bowling_elo", "elo_diff_bowling"],
    "Career batting (avg/SR)": ["team1_batting_avg", "team1_batting_sr",
                                "team2_batting_avg", "team2_batting_sr", "batting_avg_diff"],
    "Career bowling (avg/econ)": ["team1_bowling_avg", "team1_bowling_econ",
                                  "team2_bowling_avg", "team2_bowling_econ", "bowling_econ_diff"],
    "Recent form (last 10)": ["team1_win_rate_last_10", "team2_win_rate_last_10", "win_rate_diff"],
    "Head-to-head": ["h2h_team1_win_rate_shrunk", "h2h_n_meetings"],
    "Home advantage": ["is_team1_home", "is_team2_home"],
    "Venue profile": ["venue_avg_score", "venue_chase_win_pct", "venue_dot_pct",
                      "venue_boundary_pct", "venue_p4", "venue_p6", "venue_pw", "venue_id_encoded"],
    "Lineup matchup (hand/pace/spin)": ["team1_lhb_count", "team1_pace_count", "team1_spinner_count",
                                        "team2_lhb_count", "team2_pace_count", "team2_spinner_count"],
    "Toss / bat-first": ["team1_batting_first", "toss_winner_is_team1", "toss_decision_bat"],
    "Competition / international": ["is_international", "competition_tier_encoded"],
}


def load_name2id():
    return {r["player"]: r["cricsheet_id"].strip() for r in csv.DictReader(open(ROSTER))}


def main() -> int:
    name2id = load_name2id()

    def ids(names):
        return [name2id[n] if name2id.get(n) else n for n in names]

    model = joblib.load(MODEL_DIR / "model.pkl")
    encoders = joblib.load(MODEL_DIR / "encoders.pkl")
    feat = [l.strip() for l in open(MODEL_DIR / "feature_columns.txt") if l.strip()]
    booster = model.get_booster()

    provider = StatsProvider(str(REPO / "models"), version="v3")
    metadata = PlayerMetadataProvider(str(REPO / "data" / "all_players_enriched.csv"))
    form, h2h, home = load_trackers()

    def encode(record):
        df = pd.DataFrame([record])
        for col, le in encoders.items():
            ec = f"{col}_id_encoded" if col == "venue" else f"{col}_encoded"
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(lambda v: v if v in known else le.classes_[0])
            df[ec] = le.transform(df[col].astype(str))
        return df

    out_lines = ["# MLC 2026 — weighted factor report (best XIs)\n",
                 "*Exact XGBoost TreeSHAP attribution of `xgb_match_v3_m7_production`. "
                 "Each factor's push is in log-odds and the pushes + base rate sum to the "
                 "model's margin. \"Favors\" = which side the factor pushes toward; "
                 "\"weight\" = share of total absolute movement.*\n"]

    for stem, t1, t2 in MATCHES:
        meta = json.loads((FIX_DIR / f"{stem}.json").read_text())
        fx = dict(meta)
        fx["team1_lineup"] = ids(BEST_XI[t1])
        fx["team2_lineup"] = ids(BEST_XI[t2])
        rec = compute_features(fx, provider, metadata, form, h2h, home)
        df = encode(rec)
        p1 = float(model.predict_proba(df[feat])[0, 1])
        contribs = booster.predict(xgb.DMatrix(df[feat], feature_names=feat),
                                   pred_contribs=True)[0]
        bias = float(contribs[-1])
        cmap = {f: float(contribs[i]) for i, f in enumerate(feat)}

        # group sums
        grp = {g: sum(cmap[f] for f in members) for g, members in GROUPS.items()}
        total_abs = sum(abs(v) for v in grp.values()) or 1.0
        margin = sum(cmap.values()) + bias

        a1, a2 = ABBR[t1], ABBR[t2]
        hdr = (f"\n## {a1} v {a2} — {a1} {p1*100:.1f}% / {a2} {(1-p1)*100:.1f}%  "
               f"({meta['venue']}, {meta['date']})")
        print(hdr)
        out_lines.append(hdr + "\n")
        print(f"  base rate (bias) = {1/(1+math.exp(-bias))*100:.1f}% {a1}; "
              f"margin {margin:+.3f} -> {1/(1+math.exp(-margin))*100:.1f}% {a1}")

        def diff(g):
            r = rec
            if g == "Top-6 batting (ELO)":
                return f"{r['team1_top6_batting_elo_avg']:.0f} vs {r['team2_top6_batting_elo_avg']:.0f} ({r['top6_batting_elo_diff']:+.0f})"
            if g == "Bottom-5 bowling (ELO)":
                return f"{r['team1_bottom5_bowling_elo_avg']:.0f} vs {r['team2_bottom5_bowling_elo_avg']:.0f} ({r['bottom5_bowling_elo_diff']:+.0f})"
            if g == "Whole-XI batting (ELO)":
                return f"{r['team1_batting_elo']:.0f} vs {r['team2_batting_elo']:.0f} ({r['elo_diff_batting']:+.0f})"
            if g == "Whole-XI bowling (ELO)":
                return f"{r['team1_bowling_elo']:.0f} vs {r['team2_bowling_elo']:.0f} ({r['elo_diff_bowling']:+.0f})"
            if g == "Career batting (avg/SR)":
                return f"avg {r['team1_batting_avg']:.1f} vs {r['team2_batting_avg']:.1f}; SR {r['team1_batting_sr']:.0f} vs {r['team2_batting_sr']:.0f}"
            if g == "Career bowling (avg/econ)":
                return f"econ {r['team1_bowling_econ']:.2f} vs {r['team2_bowling_econ']:.2f}"
            if g == "Recent form (last 10)":
                return f"{r['team1_win_rate_last_10']:.2f} vs {r['team2_win_rate_last_10']:.2f}"
            if g == "Head-to-head":
                return f"rate {r['h2h_team1_win_rate_shrunk']:.2f}, n={r['h2h_n_meetings']}"
            if g == "Home advantage":
                return f"home: T1 {r['is_team1_home']}, T2 {r['is_team2_home']}"
            if g == "Lineup matchup (hand/pace/spin)":
                return f"LHB {r['team1_lhb_count']}/{r['team2_lhb_count']}, pace {r['team1_pace_count']}/{r['team2_pace_count']}, spin {r['team1_spinner_count']}/{r['team2_spinner_count']}"
            if g == "Venue profile":
                return f"avg {r['venue_avg_score']:.0f}, chase {r['venue_chase_win_pct']:.2f} (shared)"
            if g == "Toss / bat-first":
                return "pre-toss default (shared)"
            if g == "Competition / international":
                return "MLC / club (shared)"
            return ""

        rows = sorted(grp.items(), key=lambda kv: -abs(kv[1]))
        tbl = ["", f"| Factor | {a1} vs {a2} difference | push (logit) | favors | weight |",
               "|---|---|---|---|---|"]
        for g, v in rows:
            favors = a1 if v > 0 else (a2 if v < 0 else "—")
            w = abs(v) / total_abs * 100
            tbl.append(f"| {g} | {diff(g)} | {v:+.3f} | {favors} | {w:4.1f}% |")
        block = "\n".join(tbl)
        print(block)
        out_lines.append(block + "\n")

    out_md = REPO / "reports" / "mlc_2026_factor_report.md"
    out_md.write_text("\n".join(out_lines) + "\n")
    print(f"\nWrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
