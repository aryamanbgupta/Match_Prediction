"""E2 — Fair baselines for the prop-bet backtest.

The MLC 2025 backtest correction (2026-06-08) proved that base-rate
Brier-skill is too weak a bar: the sim's top-scorer hit rate equalled a
positional baseline and its SR ranking equalled a career lookup. This
script recomputes every binary prop family's skill against the *fair*
baseline a competent bettor would actually use, built strictly as-of
each match date from the cricsheet corpus (no future data: only matches
with date < eval match date contribute; same first-write-wins semantics
as the SQLite cache).

Baselines per family:
  top_batter            positional prior P(top scorer | lineup slot), as-of
  top_bowler            usage share: expected balls × wicket rate within XI
  batter_50plus         EB-shrunk career P(50+ | batted), k=20
  batter_6plus_six      EB-shrunk career P(>=1 six | batted), k=20
  batter_fours_{1,2,3}plus  EB-shrunk career P(>=k fours | batted), k=20
  bowler_wkts_{1,2,3}plus   EB-shrunk career P(>=k wkts | bowled), k=20
  innings_runs_ou_*     venue-shrunk historical P(innings total > line), k=20
  pp_total_ou_*         venue-shrunk historical P(PP total > line), k=20
  team_highest_individual_ou_*  venue-shrunk P(team top score > line), k=20
  first_wicket_runs_ou_30_5     venue-shrunk P(1st-wkt runs > line), k=20
  match_total_sixes_ou_*        venue-shrunk P(match sixes > line), k=20
  highest_over_runs_ou_*        venue-shrunk P(max over runs > line), k=20

MAE families: baseline point forecast = shrunk career / venue as-of mean
  batter_runs_mae, team_total_fours_mae, team_total_sixes_mae,
  team_first_over_mae, highest_individual_mae

Skipped (stated, not silently): bowler_economy_ou_* (spell-length
conditioning makes a fair career baseline ill-defined without modelling
overs bowled), p_tie (degenerate).

Comparison: paired rows (sim p, baseline p, y); ΔBrier = sim − baseline
with a cluster bootstrap BY MATCH (rows within a match are correlated;
row-level bootstrap overstates significance). Negative Δ ⇒ sim adds
skill over the fair baseline.

Usage:
    uv run python scripts/sim_eval/prop_fair_baselines.py \
        --detail reports/prop_calibration_detail_emp_n261.json \
        --out reports/e2_prop_fair_baselines.md
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from identity_maps import canonicalize_venue

REPO = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO / "data" / "t20s_json"
CACHE = REPO / "models" / "prop_fair_baseline_corpus_v2.pkl"

BOWLER_KINDS = {"bowled", "caught", "lbw", "stumped", "caught and bowled",
                "hit wicket"}
BASELINE_VERSION = "e2-v2-usage-top-bowler"
K_PLAYER = 20.0
K_VENUE = 20.0
K_USAGE = 5.0
K_WICKET_RATE = 120.0


# ------------------------------------------------------------- corpus pass
def build_corpus_logs(source_dir: Path) -> dict:
    """One pass over cricsheet JSONs -> date-stamped per-entity logs.

    Only male T20s, innings 1-2. Every log is a list of (date, *values)
    later sorted by date for strict as-of queries.
    """
    batter_log = defaultdict(list)   # name -> [(date, runs, fours, sixes)]
    bowler_log = defaultdict(list)   # name -> [(date, wkts)]
    # name -> [(date, balls, wkts)], including zero-ball XI appearances
    bowling_usage = defaultdict(list)
    venue_inn = defaultdict(list)    # venue -> [(date, total, pp, top_score,
                                     #            first_wkt_runs, first_over,
                                     #            team_fours)]
    venue_match = defaultdict(list)  # venue -> [(date, match_sixes, max_over)]
    pos_top = []                     # [(date, appearance_pos_of_top_scorer)]

    files = sorted(source_dir.glob("*.json"))
    n_used = 0
    for f in files:
        try:
            j = json.load(open(f))
        except Exception:
            continue
        info = j.get("info", {})
        if info.get("gender") != "male":
            continue
        dates = info.get("dates") or []
        if not dates:
            continue
        date = str(dates[0])
        teams = info.get("teams") or []
        players = info.get("players") or {}
        venue = canonicalize_venue(info.get("venue"), fallback="?")
        innings = j.get("innings", [])[:2]
        match_sixes = 0
        match_max_over = 0
        for inn in innings:
            batters = {}
            appear = []
            total = 0
            pp = 0
            first_over = 0
            first_wkt_runs = None
            wkts_by_bowler = defaultdict(int)
            balls_by_bowler = defaultdict(int)
            for ov in inn.get("overs", []):
                over_no = ov.get("over", 0)
                over_runs = 0
                for d in ov.get("deliveries", []):
                    b = d["batter"]
                    balls_by_bowler[d["bowler"]] += 1
                    if b not in batters:
                        batters[b] = {"runs": 0, "fours": 0, "sixes": 0}
                        appear.append(b)
                    r = d["runs"]["batter"]
                    tr = d["runs"]["total"]
                    total += tr
                    over_runs += tr
                    batters[b]["runs"] += r
                    if r == 4:
                        batters[b]["fours"] += 1
                    elif r == 6:
                        batters[b]["sixes"] += 1
                        match_sixes += 1
                    if r == 4:
                        pass
                    for w in d.get("wickets", []):
                        if first_wkt_runs is None:
                            first_wkt_runs = total
                        if w.get("kind") in BOWLER_KINDS:
                            wkts_by_bowler[d["bowler"]] += 1
                    if over_no < 6:
                        pp += tr
                    if over_no == 0:
                        first_over += tr
                match_max_over = max(match_max_over, over_runs)
            if not batters:
                continue
            if first_wkt_runs is None:
                first_wkt_runs = total
            team_fours = sum(v["fours"] for v in batters.values())
            top_score = max(v["runs"] for v in batters.values())
            top_name = max(batters, key=lambda b: batters[b]["runs"])
            pos_top.append((date, appear.index(top_name) + 1))
            for b, v in batters.items():
                batter_log[b].append((date, v["runs"], v["fours"], v["sixes"]))
            for bw, w in wkts_by_bowler.items():
                bowler_log[bw].append((date, w))
            # bowlers with 0 wickets still bowled — capture them too
            seen_bowlers = set(balls_by_bowler)
            for bw in seen_bowlers - set(wkts_by_bowler):
                bowler_log[bw].append((date, 0))
            bat_team = inn.get("team")
            bowl_teams = [team for team in teams if team != bat_team]
            if len(bowl_teams) == 1:
                bowl_team = bowl_teams[0]
                appearances = set(players.get(bowl_team, [])) | seen_bowlers
                for bw in appearances:
                    bowling_usage[bw].append(
                        (date, balls_by_bowler[bw], wkts_by_bowler[bw]))
            venue_inn[venue].append((date, total, pp, top_score,
                                     first_wkt_runs, first_over, team_fours))
        venue_match[venue].append((date, match_sixes, match_max_over))
        n_used += 1

    def _sortlog(d):
        return {k: sorted(v) for k, v in d.items()}

    print(f"corpus pass: {n_used} male T20s used of {len(files)} files")
    return {
        "batter": _sortlog(batter_log),
        "bowler": _sortlog(bowler_log),
        "bowling_usage": _sortlog(bowling_usage),
        "venue_inn": _sortlog(venue_inn),
        "venue_match": _sortlog(venue_match),
        "pos_top": sorted(pos_top),
    }


# ------------------------------------------------------------ as-of queries
class AsOf:
    """Strict as-of (< date) rate/mean queries over sorted date-stamped logs."""

    def __init__(self, logs: dict):
        self.logs = logs
        # flatten global logs once for the global priors
        self.all_batter = sorted(
            row for rows in logs["batter"].values() for row in rows)
        self.all_bowler = sorted(
            row for rows in logs["bowler"].values() for row in rows)
        self.all_bowling_usage = sorted(
            row for rows in logs["bowling_usage"].values() for row in rows)
        self.all_venue_inn = sorted(
            row for rows in logs["venue_inn"].values() for row in rows)
        self.all_venue_match = sorted(
            row for rows in logs["venue_match"].values() for row in rows)
        self._usage_dates = [row[0] for row in self.all_bowling_usage]
        self._usage_cum_balls = np.cumsum(
            [row[1] for row in self.all_bowling_usage])
        self._usage_cum_wkts = np.cumsum(
            [row[2] for row in self.all_bowling_usage])
        self._usage_global_cache = {}
        self._usage_player_cache = {}

    @staticmethod
    def _before(rows, date):
        i = bisect_left(rows, (date,))
        return rows[:i]

    def rate(self, rows, date, fn) -> tuple:
        """(rate of fn(row)==True among rows before date, n)."""
        sel = self._before(rows, date)
        if not sel:
            return 0.0, 0
        hits = sum(1 for r in sel if fn(r))
        return hits / len(sel), len(sel)

    def mean(self, rows, date, fn) -> tuple:
        sel = self._before(rows, date)
        if not sel:
            return 0.0, 0
        vals = [fn(r) for r in sel]
        return float(np.mean(vals)), len(sel)

    def shrunk_rate(self, ent_rows, glob_rows, date, fn, k) -> float:
        g, gn = self.rate(glob_rows, date, fn)
        if not ent_rows:
            return g
        e, n = self.rate(ent_rows, date, fn)
        return (k * g + n * e) / (k + n)

    def shrunk_mean(self, ent_rows, glob_rows, date, fn, k) -> float:
        g, gn = self.mean(glob_rows, date, fn)
        if not ent_rows:
            return g
        e, n = self.mean(ent_rows, date, fn)
        return (k * g + n * e) / (k + n)

    def career_wickets(self, name, date) -> float:
        rows = self._before(self.logs["bowler"].get(name, []), date)
        return float(sum(r[1] for r in rows))

    def global_bowling_usage(self, date) -> tuple[float, float]:
        """Mean balls/XI appearance and bowler wickets/ball before date."""
        if date in self._usage_global_cache:
            return self._usage_global_cache[date]
        i = bisect_left(self._usage_dates, date)
        if i == 0:
            result = (120.0 / 11.0, 0.05)
        else:
            balls = float(self._usage_cum_balls[i - 1])
            wkts = float(self._usage_cum_wkts[i - 1])
            result = (balls / i, wkts / balls if balls else 0.05)
        self._usage_global_cache[date] = result
        return result

    def player_bowling_usage(self, name, date) -> tuple[int, int, int]:
        """XI appearances, deliveries and bowler wickets strictly before date."""
        key = (name, date)
        if key not in self._usage_player_cache:
            rows = self._before(self.logs["bowling_usage"].get(name, []), date)
            self._usage_player_cache[key] = (
                len(rows),
                sum(row[1] for row in rows),
                sum(row[2] for row in rows),
            )
        return self._usage_player_cache[key]

    def bowling_expectation(self, name, date) -> tuple[float, float]:
        """EB-shrunk expected deliveries and wickets/delivery."""
        prior_balls, global_rate = self.global_bowling_usage(date)
        appearances, balls, wkts = self.player_bowling_usage(name, date)
        expected_balls = (
            (K_USAGE * prior_balls + balls) / (K_USAGE + appearances)
            if appearances else prior_balls
        )
        wicket_rate = (
            (K_WICKET_RATE * global_rate + wkts) /
            (K_WICKET_RATE + balls)
        )
        return expected_balls, wicket_rate

    def expected_wickets(self, name, date) -> float:
        expected_balls, wicket_rate = self.bowling_expectation(name, date)
        return expected_balls * wicket_rate

    def pos_top_prior(self, date, pos) -> float:
        sel = self._before(self.logs["pos_top"], date)
        if not sel:
            return 1.0 / 11
        hits = sum(1 for r in sel if r[1] == pos)
        # +1/+11 smoothing so deep positions never get exactly 0
        return (hits + 1) / (len(sel) + 11)


def poisson_at_least(expected_count: float, threshold: int) -> float:
    """P(X >= threshold) for X ~ Poisson(expected_count)."""
    if threshold <= 0:
        return 1.0
    expected_count = max(0.0, float(expected_count))
    term = 1.0
    cdf = term
    for value in range(1, threshold):
        term *= expected_count / value
        cdf += term
    return float(np.clip(1.0 - np.exp(-expected_count) * cdf, 0.0, 1.0))


# ---------------------------------------------------------------- baselines
def baseline_rows(detail: list, asof: AsOf) -> dict:
    """family -> list of {p_sim, p_base, y, match_id} paired rows."""
    out = defaultdict(list)
    skipped = defaultdict(int)

    for m in detail:
        mid = m["match_id"]
        date = mid[:10]
        obs = m["obs"]
        venue_key = _venue_from_match_id(mid, asof)

        v_inn = asof.logs["venue_inn"].get(venue_key, []) if venue_key else []
        v_match = asof.logs["venue_match"].get(venue_key, []) if venue_key else []

        # --- top_batter: positional prior over each team's 11 slots
        rows = obs.get("top_batter", [])
        for team in {r["team"] for r in rows}:
            trows = [r for r in rows if r["team"] == team]
            priors = np.array([asof.pos_top_prior(date, i + 1)
                               for i in range(len(trows))])
            priors = priors / priors.sum()
            for r, pb in zip(trows, priors):
                out["top_batter"].append(
                    {"p_sim": r["p"], "p_base": float(pb), "y": r["y"],
                     "mid": mid})

        # --- top_bowler: expected usage × wicket rate, normalized within XI
        rows = obs.get("top_bowler", [])
        for team in {r["team"] for r in rows}:
            trows = [r for r in rows if r["team"] == team]
            w = np.array([asof.expected_wickets(r["name"], date)
                          for r in trows])
            if w.sum() <= 0:
                w = np.ones(len(trows), dtype=float)
            w = w / w.sum()
            for r, pb in zip(trows, w):
                out["top_bowler"].append(
                    {"p_sim": r["p"], "p_base": float(pb), "y": r["y"],
                     "mid": mid})

        # --- per-batter career-rate families
        bat_fams = {
            "batter_50plus": lambda row: row[1] >= 50,
            "batter_6plus_six": lambda row: row[3] >= 1,
            "batter_fours_1plus": lambda row: row[2] >= 1,
            "batter_fours_2plus": lambda row: row[2] >= 2,
            "batter_fours_3plus": lambda row: row[2] >= 3,
        }
        for fam, fn in bat_fams.items():
            for r in obs.get(fam, []):
                pb = asof.shrunk_rate(asof.logs["batter"].get(r["name"], []),
                                      asof.all_batter, date, fn, K_PLAYER)
                out[fam].append({"p_sim": r["p"], "p_base": pb, "y": r["y"],
                                 "mid": mid})

        # --- per-bowler career-rate families. The I13 usage × rate Poisson
        # candidate is retained on each row for an explicit stronger-bar
        # comparison in the report, but is not promoted unless it wins.
        for thr, fam in ((1, "bowler_wkts_1plus"), (2, "bowler_wkts_2plus"),
                         (3, "bowler_wkts_3plus")):
            fn = (lambda t: lambda row: row[1] >= t)(thr)
            for r in obs.get(fam, []):
                pb = asof.shrunk_rate(
                    asof.logs["bowler"].get(r["name"], []),
                    asof.all_bowler, date, fn, K_PLAYER)
                p_usage_count = poisson_at_least(
                    asof.expected_wickets(r["name"], date), thr)
                out[fam].append({"p_sim": r["p"], "p_base": pb, "y": r["y"],
                                 "p_usage_count": p_usage_count, "mid": mid})

        # --- venue-shrunk innings-level O/U families
        inn_fams = {
            "innings_runs_ou_160_5": lambda row: row[1] > 160.5,
            "innings_runs_ou_170_5": lambda row: row[1] > 170.5,
            "innings_runs_ou_180_5": lambda row: row[1] > 180.5,
            "pp_total_ou_45_5": lambda row: row[2] > 45.5,
            "pp_total_ou_50_5": lambda row: row[2] > 50.5,
            "pp_total_ou_55_5": lambda row: row[2] > 55.5,
            "team_highest_individual_ou_29_5": lambda row: row[3] > 29.5,
            "team_highest_individual_ou_34_5": lambda row: row[3] > 34.5,
            "team_highest_individual_ou_39_5": lambda row: row[3] > 39.5,
            "first_wicket_runs_ou_30_5": lambda row: row[4] > 30.5,
        }
        for fam, fn in inn_fams.items():
            for r in obs.get(fam, []):
                pb = asof.shrunk_rate(v_inn, asof.all_venue_inn, date, fn,
                                      K_VENUE)
                out[fam].append({"p_sim": r["p"], "p_base": pb, "y": r["y"],
                                 "mid": mid})

        # --- venue-shrunk match-level O/U families
        match_fams = {
            "match_total_sixes_ou_15_5": lambda row: row[1] > 15.5,
            "match_total_sixes_ou_20_5": lambda row: row[1] > 20.5,
            "highest_over_runs_ou_18_5": lambda row: row[2] > 18.5,
            "highest_over_runs_ou_24_5": lambda row: row[2] > 24.5,
        }
        for fam, fn in match_fams.items():
            for r in obs.get(fam, []):
                pb = asof.shrunk_rate(v_match, asof.all_venue_match, date, fn,
                                      K_VENUE)
                out[fam].append({"p_sim": r["p"], "p_base": pb, "y": r["y"],
                                 "mid": mid})

        # --- MAE families: baseline point forecast = shrunk as-of mean
        for r in obs.get("batter_runs_mae", []):
            pb = asof.shrunk_mean(asof.logs["batter"].get(r["name"], []),
                                  asof.all_batter, date, lambda row: row[1],
                                  K_PLAYER)
            out["batter_runs_mae"].append(
                {"p_sim": r["sim_mean"], "p_base": pb, "y": r["actual"],
                 "mid": mid})
        mae_v = {
            "team_total_fours_mae": ("venue_inn", lambda row: row[6]),
            "team_total_sixes_mae": ("venue_match", lambda row: row[1] / 2.0),
            "team_first_over_mae": ("venue_inn", lambda row: row[5]),
            "highest_individual_mae": ("venue_inn", lambda row: row[3]),
        }
        for fam, spec in mae_v.items():
            rows = obs.get(fam, [])
            if not rows:
                continue
            if spec is None:
                skipped[fam] += len(rows)
                continue
            log_name, fn = spec
            ent = v_inn if log_name == "venue_inn" else v_match
            glob = (asof.all_venue_inn if log_name == "venue_inn"
                    else asof.all_venue_match)
            for r in rows:
                pb = asof.shrunk_mean(ent, glob, date, fn, K_VENUE)
                out[fam].append({"p_sim": r["sim_mean"], "p_base": pb,
                                 "y": r["actual"], "mid": mid})

    if skipped:
        print(f"skipped rows (no fair baseline defined): {dict(skipped)}")
    return out


_VENUE_CACHE: dict = {}


def _venue_from_match_id(mid: str, asof: AsOf):
    """match_id = '<date>_<Team1>_<Team2>_<venue with underscores>'.

    Match the trailing portion against known corpus venue names
    (underscores vs spaces/commas make exact split unreliable).
    """
    if mid in _VENUE_CACHE:
        return _VENUE_CACHE[mid]
    flat = mid.replace("_", " ")
    best = None
    for v in asof.logs["venue_inn"]:
        if v.replace(",", " ").replace("  ", " ") in flat or \
           flat.endswith(v.replace(",", " ")):
            if best is None or len(v) > len(best):
                best = v
    if best is None:
        # fuzzy: longest venue whose alnum form is a suffix of mid's
        norm = "".join(c for c in flat.lower() if c.isalnum())
        for v in asof.logs["venue_inn"]:
            vn = "".join(c for c in v.lower() if c.isalnum())
            if norm.endswith(vn):
                if best is None or len(v) > len(best):
                    best = v
    _VENUE_CACHE[mid] = best
    return best


# ------------------------------------------------------------------ scoring
def brier(p, y):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    return (p - y) ** 2


def cluster_bootstrap_delta(rows, metric, n_boot=2000, seed=29):
    """Bootstrap mean(metric_sim - metric_base) clustering by match."""
    rng = np.random.default_rng(seed)
    by_match = defaultdict(list)
    for r in rows:
        by_match[r["mid"]].append(r)
    mids = list(by_match)
    deltas = []
    for _ in range(n_boot):
        sample = rng.choice(len(mids), size=len(mids), replace=True)
        ds = []
        for i in sample:
            for r in by_match[mids[i]]:
                ds.append(metric(r))
        deltas.append(np.mean(ds))
    deltas = np.array(deltas)
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", type=Path,
                    default=REPO / "reports/prop_calibration_detail_emp_n261.json")
    ap.add_argument("--out", type=Path,
                    default=REPO / "reports/e2_prop_fair_baselines.md")
    ap.add_argument("--rebuild-corpus", action="store_true")
    args = ap.parse_args()

    if CACHE.exists() and not args.rebuild_corpus:
        logs = pickle.load(open(CACHE, "rb"))
        print(f"corpus logs loaded from cache {CACHE.name}")
    else:
        logs = build_corpus_logs(SOURCE_DIR)
        pickle.dump(logs, open(CACHE, "wb"))
        print(f"corpus logs cached -> {CACHE}")

    asof = AsOf(logs)
    detail = json.load(open(args.detail))
    print(f"eval matches: {len(detail)}")
    paired = baseline_rows(detail, asof)

    binary_fams = [f for f in paired if not f.endswith("_mae")]
    mae_fams = [f for f in paired if f.endswith("_mae")]

    lines = [
        "# E2 v2 — Prop families vs FAIR baselines (not base rates)",
        "",
        f"Detail: `{args.detail.name}` (n={len(detail)} matches). "
        "Baselines built strictly as-of each match date from "
        "`data/t20s_json` (male T20s, innings 1–2). Δ = sim − baseline; "
        "**negative Δ ⇒ sim beats the fair baseline**. 95% CIs from "
        "cluster bootstrap by match (2,000 resamples).",
        "",
        f"**Baseline version:** `{BASELINE_VERSION}`. `top_bowler` uses "
        f"EB-shrunk expected deliveries (K={K_USAGE:g} XI appearances) × "
        f"wickets/delivery (K={K_WICKET_RATE:g} deliveries), normalized "
        "within the team. XI histories include zero-ball appearances. "
        "`bowler_wkts_{1,2,3}plus` retains the stronger EB-shrunk as-of "
        "threshold-rate baseline (K=20 bowling appearances).",
        "",
        "## Binary families (Brier)",
        "",
        "| family | n | Brier sim | Brier fair-base | ΔBrier | Δ 95% CI | verdict |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    summary = {}
    for fam in sorted(binary_fams):
        rows = paired[fam]
        bs = float(np.mean(brier([r["p_sim"] for r in rows],
                                 [r["y"] for r in rows])))
        bb = float(np.mean(brier([r["p_base"] for r in rows],
                                 [r["y"] for r in rows])))
        lo, hi = cluster_bootstrap_delta(
            rows, lambda r: (r["p_sim"] - r["y"]) ** 2 - (r["p_base"] - r["y"]) ** 2)
        if hi < 0:
            verdict = "✅ sim adds skill"
        elif lo > 0:
            verdict = "❌ baseline wins"
        else:
            verdict = "≈ parity"
        lines.append(f"| `{fam}` | {len(rows)} | {bs:.4f} | {bb:.4f} | "
                     f"{bs - bb:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {verdict} |")
        summary[fam] = {"n": len(rows), "brier_sim": bs, "brier_base": bb,
                        "delta_ci": [lo, hi], "verdict": verdict,
                        "baseline_version": BASELINE_VERSION}

    lines += [
        "",
        "## I13 count-baseline candidate decision",
        "",
        "The analogous expected-balls × wicket-rate Poisson tail was "
        "evaluated but not promoted. Positive Δ below means that candidate "
        "has worse Brier score than the retained as-of threshold-rate "
        "baseline.",
        "",
        "| family | retained Brier | usage-count Brier | Δ candidate − "
        "retained | Δ 95% CI | decision |",
        "|---|---:|---:|---:|---|---|",
    ]
    for fam in ("bowler_wkts_1plus", "bowler_wkts_2plus",
                "bowler_wkts_3plus"):
        rows = paired[fam]
        retained = float(np.mean([
            (r["p_base"] - r["y"]) ** 2 for r in rows]))
        candidate = float(np.mean([
            (r["p_usage_count"] - r["y"]) ** 2 for r in rows]))
        lo, hi = cluster_bootstrap_delta(
            rows,
            lambda r: ((r["p_usage_count"] - r["y"]) ** 2 -
                       (r["p_base"] - r["y"]) ** 2),
        )
        decision = ("retain threshold-rate"
                    if candidate >= retained else "promote usage-count")
        lines.append(
            f"| `{fam}` | {retained:.4f} | {candidate:.4f} | "
            f"{candidate - retained:+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
            f"{decision} |")
        summary[fam]["usage_count_candidate"] = {
            "brier": candidate,
            "delta_vs_retained": candidate - retained,
            "delta_ci": [lo, hi],
            "decision": decision,
        }

    lines += ["", "## Continuous families (MAE)", "",
              "| family | n | MAE sim | MAE fair-base | ΔMAE | Δ 95% CI | verdict |",
              "|---|---:|---:|---:|---:|---|---|"]
    for fam in sorted(mae_fams):
        rows = paired[fam]
        ms = float(np.mean([abs(r["p_sim"] - r["y"]) for r in rows]))
        mb = float(np.mean([abs(r["p_base"] - r["y"]) for r in rows]))
        lo, hi = cluster_bootstrap_delta(
            rows, lambda r: abs(r["p_sim"] - r["y"]) - abs(r["p_base"] - r["y"]))
        if hi < 0:
            verdict = "✅ sim adds skill"
        elif lo > 0:
            verdict = "❌ baseline wins"
        else:
            verdict = "≈ parity"
        lines.append(f"| `{fam}` | {len(rows)} | {ms:.2f} | {mb:.2f} | "
                     f"{ms - mb:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {verdict} |")
        summary[fam] = {"n": len(rows), "mae_sim": ms, "mae_base": mb,
                        "delta_ci": [lo, hi], "verdict": verdict,
                        "baseline_version": BASELINE_VERSION}

    lines += ["", "## Skipped families",
              "",
              "- `bowler_economy_ou_*`: fair career baseline ill-defined "
              "without modelling overs bowled per spell.",
              "- `p_tie`: degenerate (ties ~0.4% of matches).",
              ""]
    args.out.write_text("\n".join(lines))
    json.dump(summary, open(args.out.with_suffix(".json"), "w"), indent=2)
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
