"""B9 — top_bowler margin vs a usage-share fair baseline (B4 follow-up).

B4 showed the E2 career-wickets-share baseline is weak exactly where the
sim's top_bowler profit concentrates (p_base<2% longshots): it prices
debutants at 0.26% when they top the wickets 0.90% of the time, because
career-wickets share cannot see *who actually bowls*. This script builds
the stronger "competent bettor" bar and re-runs the B4 margin + pricing
analysis against it. Pure analysis on the canonical D15 detail JSON — no
sim run, no model change, `prop_fair_baselines.py` untouched (read-only
import for the career baseline + corpus-cache reuse).

Usage baseline (stated, fixed pre-run):
  For player i in the XI at date D (strictly as-of, < D):
    exp_balls_i = (K_USAGE * prior_balls + n_i * mean_balls_i)
                  / (K_USAGE + n_i)
        n_i = XI appearances with a bowling innings; mean_balls_i = mean
        deliveries bowled per appearance (0 for listed non-bowlers);
        prior_balls = as-of global mean deliveries per XI appearance
        (lineup-uniform prior — a true debutant sits at the prior).
    rate_i = (K_RATE * g + career_wkts_i) / (K_RATE + career_balls_i)
        g = as-of global bowler-credited wickets per delivery
        (BOWLER_KINDS, matching E2's y semantics).
    p_base_i ∝ exp_balls_i * rate_i, normalized within the market rows.
  Deliveries counted = ALL deliveries (incl. wides/no-balls); innings
  1–2, male T20s only; appearances = XI list ∪ observed bowlers of the
  bowling side (a team that never bowled contributes no appearance).

PRE-COMMITTED parameters and conclusion mapping (before any result):
  - Headline shrinkage: K_USAGE = 5 appearances, K_RATE = 120 balls.
    Robustness grid (context, reported verbatim): (2, 60), (10, 240).
  - Stronger-bar check: paired dBrier (usage − career), cluster boot by
    match. Usage is a stronger bar only if CI-clean negative; if not,
    the E2 bar stands and the margin question is moot (report as such).
  - Margin verdict: sim's skill claim SURVIVES if paired dBrier
    (sim − usage) is CI-clean negative at the headline params; FLIPS if
    CI includes 0 or is positive. Grid disagreement is reported.
  - Bootstrap: 2000 resamples, seed 29 (identical to B4/E2 statistic).

Outputs: research/reports/auto/B9_usage_pricing.md +
models/auto/b9/usage_numbers.json (+ usage_corpus.pkl cache).

Usage:
    uv run python scripts/auto/b9_usage_baseline.py \
        --detail models/auto/d15/detail_d15_s43_n261.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO / "data" / "t20s_json"
CORPUS_CACHE = REPO / "models" / "auto" / "b9" / "usage_corpus.pkl"

BOWLER_KINDS = {"bowled", "caught", "lbw", "stumped", "caught and bowled",
                "hit wicket"}

K_USAGE = 5.0        # appearances — EB shrink for expected deliveries
K_RATE = 120.0       # deliveries — EB shrink for per-ball wicket rate
ROBUST_GRID = [(2.0, 60.0), (10.0, 240.0)]   # context only; headline above

VIGS = [0.00, 0.02, 0.05, 0.10]
THRESHOLDS = [0.00, 0.05, 0.10, 0.20]
N_BOOT = 2000
BOOT_SEED = 29


def _load_pfb():
    """Import scripts/sim_eval/prop_fair_baselines.py (read-only reuse)."""
    p = REPO / "scripts" / "sim_eval" / "prop_fair_baselines.py"
    spec = importlib.util.spec_from_file_location("pfb", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pfb"] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------- corpus pass
def build_usage_corpus(source_dir: Path) -> dict:
    """One pass over cricsheet JSONs -> per-player XI-appearance bowling log.

    player_log: name -> sorted [(date, balls, wkts)] — one row per
    team-match in which the player's team bowled an innings (balls = 0
    for XI members who never bowled). Male T20s, innings 1-2 only.
    """
    player_log = defaultdict(list)
    n_used = 0
    n_no_players = 0
    files = sorted(source_dir.glob("*.json"))
    for f in files:
        try:
            j = json.load(open(f))
        except Exception:
            continue
        info = j.get("info", {})
        if info.get("gender") != "male":
            continue
        dates = info.get("dates") or []
        teams = info.get("teams") or []
        if not dates or len(teams) != 2:
            continue
        date = str(dates[0])
        players = info.get("players", {})
        if not players:
            n_no_players += 1
        for inn in j.get("innings", [])[:2]:
            bat_team = inn.get("team")
            others = [t for t in teams if t != bat_team]
            if len(others) != 1:
                continue
            bowl_team = others[0]
            balls = defaultdict(int)
            wkts = defaultdict(int)
            for ov in inn.get("overs", []):
                for d in ov.get("deliveries", []):
                    bw = d["bowler"]
                    balls[bw] += 1
                    for w in d.get("wickets", []):
                        if w.get("kind") in BOWLER_KINDS:
                            wkts[bw] += 1
            appearances = set(players.get(bowl_team, [])) | set(balls)
            if not appearances:
                continue
            for name in appearances:
                player_log[name].append((date, balls[name], wkts[name]))
        n_used += 1
    print(f"usage corpus pass: {n_used} male T20s used of {len(files)} "
          f"files; {n_no_players} without info.players")
    return {"player": {k: sorted(v) for k, v in player_log.items()}}


class AsOfUsage:
    """Strict as-of (< date) usage/rate queries with global prefix sums."""

    def __init__(self, corpus: dict):
        self.player = corpus["player"]
        rows = sorted(r for v in self.player.values() for r in v)
        self._dates = [r[0] for r in rows]
        self._cum_balls = np.cumsum([r[1] for r in rows])
        self._cum_wkts = np.cumsum([r[2] for r in rows])
        self._glob_cache: dict = {}

    def global_stats(self, date: str):
        """(mean balls per appearance, wkts per ball) strictly before date."""
        if date in self._glob_cache:
            return self._glob_cache[date]
        i = bisect_left(self._dates, date)
        if i == 0:
            out = (120.0 / 11.0, 0.05)   # cold-start fallback, never hit here
        else:
            tb = float(self._cum_balls[i - 1])
            tw = float(self._cum_wkts[i - 1])
            out = (tb / i, tw / tb if tb else 0.05)
        self._glob_cache[date] = out
        return out

    def player_sums(self, name: str, date: str):
        """(n_appearances, sum_balls, sum_wkts) strictly before date."""
        rows = self.player.get(name, [])
        i = bisect_left(rows, (date,))
        sel = rows[:i]
        return len(sel), sum(r[1] for r in sel), sum(r[2] for r in sel)

    def price(self, name: str, date: str, k_u: float, k_w: float):
        """Unnormalized p_raw = exp_balls * shrunk wicket rate."""
        prior_balls, g = self.global_stats(date)
        n, b, w = self.player_sums(name, date)
        exp_balls = (k_u * prior_balls + b) / (k_u + n) if n else prior_balls
        rate = (k_w * g + w) / (k_w + b)
        return exp_balls * rate


# ------------------------------------------------------------ paired rows
def build_markets(detail: list, asof_pfb, asof_use: AsOfUsage,
                  k_u: float, k_w: float) -> list[dict]:
    """B4's market grouping with BOTH baselines priced per row."""
    markets = []
    for m in detail:
        mid = m["match_id"]
        date = mid[:10]
        rows = m["obs"].get("top_bowler", [])
        for team in sorted({r["team"] for r in rows}):
            trows = [r for r in rows if r["team"] == team]
            w = np.array([asof_pfb.career_wickets(r["name"], date) + 1.0
                          for r in trows])
            w = w / w.sum()
            u = np.array([asof_use.price(r["name"], date, k_u, k_w)
                          for r in trows])
            assert u.sum() > 0
            u = u / u.sum()
            markets.append({
                "mid": mid, "team": team,
                "rows": [{"name": r["name"], "p_sim": float(r["p"]),
                          "p_career": float(pc), "p_usage": float(pu),
                          "y": int(r["y"])}
                         for r, pc, pu in zip(trows, w, u)],
            })
    return markets


def flat_rows(markets):
    out = []
    for mk in markets:
        for r in mk["rows"]:
            out.append({"mid": mk["mid"], **r})
    return out


def cluster_boot_mean(values_by_match: dict, n_boot=N_BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    mids = list(values_by_match)
    arrs = [np.asarray(values_by_match[m], dtype=float) for m in mids]
    stats = []
    for _ in range(n_boot):
        idx = rng.choice(len(mids), size=len(mids), replace=True)
        pooled = np.concatenate([arrs[i] for i in idx if len(arrs[i])])
        stats.append(pooled.mean() if len(pooled) else np.nan)
    stats = np.array(stats)
    stats = stats[~np.isnan(stats)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def paired_dbrier(rows, a_key: str, b_key: str):
    """mean((a-y)^2 - (b-y)^2) with cluster-boot CI by match."""
    by_match = defaultdict(list)
    for r in rows:
        d = (r[a_key] - r["y"]) ** 2 - (r[b_key] - r["y"]) ** 2
        by_match[r["mid"]].append(d)
    pooled = np.concatenate([np.asarray(v) for v in by_match.values()])
    lo, hi = cluster_boot_mean(by_match)
    return float(pooled.mean()), lo, hi, len(pooled)


def brier(rows, key):
    return float(np.mean([(r[key] - r["y"]) ** 2 for r in rows]))


# ------------------------------------------------------------------ pricing
def price_and_bet(rows, vig, thr, base_key="p_usage"):
    bets = []
    for r in rows:
        q = r[base_key] * (1.0 + vig)
        if q >= 1.0:
            continue
        o = 1.0 / q
        ev = r["p_sim"] * o - 1.0
        if ev > thr:
            pnl = (o - 1.0) if r["y"] == 1 else -1.0
            kelly = (r["p_sim"] * o - 1.0) / (o - 1.0)
            bets.append({"mid": r["mid"], "name": r["name"], "odds": o,
                         "ev": ev, "pnl": pnl, "y": r["y"], "kelly": kelly})
    return bets


def roi_with_ci(bets):
    if not bets:
        return None
    by_match = defaultdict(list)
    for b in bets:
        by_match[b["mid"]].append(b["pnl"])
    pnls = np.array([b["pnl"] for b in bets])
    lo, hi = cluster_boot_mean(by_match)
    return {
        "n_bets": len(bets), "n_matches": len(by_match),
        "roi_pct": float(pnls.mean() * 100),
        "roi_ci_pct": [lo * 100, hi * 100],
        "win_rate_pct": float((pnls > 0).mean() * 100),
        "total_pnl_u": float(pnls.sum()),
        "avg_odds": float(np.mean([b["odds"] for b in bets])),
        "avg_ev_pct": float(np.mean([b["ev"] for b in bets]) * 100),
        "mean_kelly": float(np.mean([b["kelly"] for b in bets])),
    }


def breakeven_vig(rows, thr=0.0, base_key="p_usage"):
    prev_v, prev_roi = None, None
    for v in np.arange(0.0, 0.5001, 0.005):
        bets = price_and_bet(rows, float(v), thr, base_key)
        if not bets:
            return prev_v, prev_roi, float(v), None
        roi = float(np.mean([b["pnl"] for b in bets]) * 100)
        if roi <= 0:
            return prev_v, prev_roi, float(v), roi
        prev_v, prev_roi = float(v), roi
    return prev_v, prev_roi, None, None


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", type=Path,
                    default=REPO / "models/auto/d15/detail_d15_s43_n261.json")
    ap.add_argument("--out", type=Path,
                    default=REPO / "research/reports/auto/B9_usage_pricing.md")
    ap.add_argument("--json-out", type=Path,
                    default=REPO / "models/auto/b9/usage_numbers.json")
    ap.add_argument("--rebuild-corpus", action="store_true")
    args = ap.parse_args()

    pfb = _load_pfb()
    assert pfb.CACHE.exists(), "E2 corpus cache missing"
    logs = pickle.load(open(pfb.CACHE, "rb"))
    asof_pfb = pfb.AsOf(logs)
    print(f"E2 corpus cache loaded: {pfb.CACHE.name}")

    CORPUS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if CORPUS_CACHE.exists() and not args.rebuild_corpus:
        corpus = pickle.load(open(CORPUS_CACHE, "rb"))
        print(f"usage corpus loaded from cache {CORPUS_CACHE.name}")
    else:
        corpus = build_usage_corpus(SOURCE_DIR)
        pickle.dump(corpus, open(CORPUS_CACHE, "wb"))
        print(f"usage corpus cached -> {CORPUS_CACHE}")
    asof_use = AsOfUsage(corpus)

    detail = json.load(open(args.detail))
    res = {"detail": str(args.detail), "n_matches": len(detail),
           "headline_k": [K_USAGE, K_RATE], "grid": {}}

    # ---------- headline pairing
    markets = build_markets(detail, asof_pfb, asof_use, K_USAGE, K_RATE)
    rows = flat_rows(markets)
    for mk in markets:
        assert abs(sum(r["p_career"] for r in mk["rows"]) - 1.0) < 1e-9
        assert abs(sum(r["p_usage"] for r in mk["rows"]) - 1.0) < 1e-9
    res["n_markets"] = len(markets)
    res["n_rows"] = len(rows)

    # cross-check: sim-vs-career margin must reproduce B4 exactly
    m_sc = paired_dbrier(rows, "p_sim", "p_career")
    res["sim_vs_career"] = {"mean": m_sc[0], "ci": [m_sc[1], m_sc[2]]}
    b4_json = REPO / "models/auto/b4/pricing_numbers.json"
    if b4_json.exists():
        b4 = json.load(open(b4_json))[0]
        drift = abs(b4["dbrier_mean"] - m_sc[0])
        print(f"B4 reproduction check: |d| = {drift:.2e}")
        assert drift < 1e-9, "career-baseline pairing drifted from B4"

    res["brier"] = {k: brier(rows, k)
                    for k in ("p_sim", "p_career", "p_usage")}

    # ---------- stronger-bar check: usage vs career
    m_uc = paired_dbrier(rows, "p_usage", "p_career")
    res["usage_vs_career"] = {"mean": m_uc[0], "ci": [m_uc[1], m_uc[2]]}
    stronger = m_uc[2] < 0
    res["usage_is_stronger_bar"] = bool(stronger)

    # ---------- the margin: sim vs usage (headline + grid)
    m_su = paired_dbrier(rows, "p_sim", "p_usage")
    res["sim_vs_usage"] = {"mean": m_su[0], "ci": [m_su[1], m_su[2]]}
    for k_u, k_w in ROBUST_GRID:
        g_rows = flat_rows(build_markets(detail, asof_pfb, asof_use,
                                         k_u, k_w))
        g_su = paired_dbrier(g_rows, "p_sim", "p_usage")
        g_uc = paired_dbrier(g_rows, "p_usage", "p_career")
        res["grid"][f"ku{k_u:g}_kw{k_w:g}"] = {
            "sim_vs_usage": {"mean": g_su[0], "ci": [g_su[1], g_su[2]]},
            "usage_vs_career": {"mean": g_uc[0], "ci": [g_uc[1], g_uc[2]]},
            "brier_usage": brier(g_rows, "p_usage"),
        }

    # ---------- pricing vs the usage baseline (headline params)
    res["table"] = {}
    for vig in VIGS:
        for thr in THRESHOLDS:
            res["table"][f"vig{vig:.2f}_thr{thr:.2f}"] = roi_with_ci(
                price_and_bet(rows, vig, thr))
    bv = breakeven_vig(rows)
    res["breakeven"] = {"last_pos_vig": bv[0], "last_pos_roi_pct": bv[1],
                        "first_nonpos_vig": bv[2],
                        "first_nonpos_roi_pct": bv[3]}

    bands = [(0.20, 1.01, ">=20%"), (0.10, 0.20, "10-20%"),
             (0.05, 0.10, "5-10%"), (0.02, 0.05, "2-5%"),
             (0.00, 0.02, "<2%")]
    bets = price_and_bet(rows, 0.05, 0.0)
    key_of = {(r["mid"], r["name"]): r["p_usage"] for r in rows}
    res["bands_vig05_thr0"] = {}
    for lo_b, hi_b, lbl in bands:
        sub = [b for b in bets if lo_b <= key_of[(b["mid"], b["name"])] < hi_b]
        res["bands_vig05_thr0"][lbl] = roi_with_ci(sub)

    # ---------- zero-career-wicket diagnostic, split by appearance history
    zc = [r for r in rows
          if asof_pfb.career_wickets(r["name"], r["mid"][:10]) == 0]
    true_deb, seen_nw = [], []
    for r in zc:
        n, b, w = asof_use.player_sums(r["name"], r["mid"][:10])
        (true_deb if n == 0 else seen_nw).append(r)

    def _grp(rr):
        if not rr:
            return None
        return {"n_rows": len(rr),
                "n_actual_top": int(sum(r["y"] for r in rr)),
                "actual_rate": float(np.mean([r["y"] for r in rr])),
                "mean_p_sim": float(np.mean([r["p_sim"] for r in rr])),
                "mean_p_career": float(np.mean([r["p_career"] for r in rr])),
                "mean_p_usage": float(np.mean([r["p_usage"] for r in rr]))}

    res["zero_career_wkts"] = {
        "all": _grp(zc), "true_debutant": _grp(true_deb),
        "seen_never_took_wkt": _grp(seen_nw),
        "bets_vig05": roi_with_ci(price_and_bet(zc, 0.05, 0.0)),
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.json_out, "w"), indent=1)
    print(f"numbers -> {args.json_out}")

    # ------------------------------------------------------------- report
    def _ci(d, scale=1.0, fmt="+.4f"):
        return (f"{d['mean']*scale:{fmt}} CI [{d['ci'][0]*scale:{fmt}}, "
                f"{d['ci'][1]*scale:{fmt}}]")

    lines = [
        "# B9 — top_bowler margin vs a usage-share fair baseline",
        "",
        f"Detail: `{Path(str(args.detail)).name}` (n={res['n_matches']} "
        f"matches, {res['n_markets']} team-markets, {res['n_rows']} rows). "
        f"Headline shrinkage K_USAGE={K_USAGE:g} appearances / "
        f"K_RATE={K_RATE:g} balls (pre-committed); cluster bootstrap by "
        f"match, {N_BOOT} resamples, seed {BOOT_SEED}.",
        "",
        "## Baseline strength (standalone Brier; lower = better)",
        "",
        f"- sim (calibrated): {res['brier']['p_sim']:.4f}",
        f"- E2 career-wickets share: {res['brier']['p_career']:.4f}",
        f"- usage-share baseline: {res['brier']['p_usage']:.4f}",
        "",
        "## Stronger-bar check (paired dBrier usage − career; "
        "negative = usage stronger)",
        "",
        f"- headline: {_ci(res['usage_vs_career'])} → "
        + ("**usage IS the stronger bar** (CI-clean)" if stronger
           else "**usage NOT CI-clean stronger** — E2 bar stands"),
        "",
        "## The margin (paired dBrier sim − baseline; negative = sim adds "
        "skill)",
        "",
        f"- sim − career (B4 reproduction): {_ci(res['sim_vs_career'])}",
        f"- **sim − usage (headline)**: {_ci(res['sim_vs_usage'])}",
    ]
    for key, g in res["grid"].items():
        lines.append(f"- sim − usage ({key}): {_ci(g['sim_vs_usage'])}; "
                     f"usage − career: {_ci(g['usage_vs_career'])}")
    lines += [
        "",
        "## Flat 1u YES ROI vs the usage-priced market (vig × threshold)",
        "",
        "| vig | thr | bets | ROI % | ROI 95% CI | win % | avg odds | "
        "avg EV % |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for vig in VIGS:
        for thr in THRESHOLDS:
            t = res["table"][f"vig{vig:.2f}_thr{thr:.2f}"]
            if t is None:
                lines.append(f"| {vig:.0%} | {thr:.0%} | 0 | — | — | — | — "
                             "| — |")
                continue
            lines.append(
                f"| {vig:.0%} | {thr:.0%} | {t['n_bets']} | "
                f"{t['roi_pct']:+.2f} | [{t['roi_ci_pct'][0]:+.2f}, "
                f"{t['roi_ci_pct'][1]:+.2f}] | {t['win_rate_pct']:.1f} | "
                f"{t['avg_odds']:.2f} | {t['avg_ev_pct']:+.1f} |")
    bv = res["breakeven"]
    lines += [
        "",
        "- break-even vig (thr 0, 0.5% grid, in-sample): last positive "
        + ("—" if bv["last_pos_vig"] is None
           else f"{bv['last_pos_vig']:.1%} (ROI {bv['last_pos_roi_pct']:+.2f}%)")
        + "; first non-positive "
        + ("none ≤ 50%" if bv["first_nonpos_vig"] is None
           else f"{bv['first_nonpos_vig']:.1%} "
                f"(ROI {bv['first_nonpos_roi_pct']:+.2f}%)"),
        "",
        "## Baseline-price bands (vig 5%, thr 0)",
        "",
        "| p_usage band | bets | ROI % | ROI 95% CI | win % | total PnL u |",
        "|---|---|---|---|---|---|",
    ]
    for lbl, t in res["bands_vig05_thr0"].items():
        if t is None:
            lines.append(f"| {lbl} | 0 | — | — | — | — |")
        else:
            lines.append(
                f"| {lbl} | {t['n_bets']} | {t['roi_pct']:+.2f} | "
                f"[{t['roi_ci_pct'][0]:+.2f}, {t['roi_ci_pct'][1]:+.2f}] | "
                f"{t['win_rate_pct']:.1f} | {t['total_pnl_u']:+.1f} |")
    zc = res["zero_career_wkts"]
    lines += [
        "",
        "## Zero-career-wicket players — how each baseline prices them",
        "",
        "| group | rows | actual top rate | mean p_sim | mean p_career | "
        "mean p_usage |",
        "|---|---|---|---|---|---|",
    ]
    for lbl, key in (("all zero-career-wkt", "all"),
                     ("true debutant (0 appearances)", "true_debutant"),
                     ("seen, never took a wkt", "seen_never_took_wkt")):
        g = zc[key]
        if g is None:
            lines.append(f"| {lbl} | 0 | — | — | — | — |")
        else:
            lines.append(
                f"| {lbl} | {g['n_rows']} | {g['actual_rate']*100:.2f}% "
                f"({g['n_actual_top']}) | {g['mean_p_sim']*100:.2f}% | "
                f"{g['mean_p_career']*100:.2f}% | "
                f"{g['mean_p_usage']*100:.2f}% |")
    zb = zc["bets_vig05"]
    lines.append("")
    lines.append(
        "- YES bets the sim would place on them vs usage prices (vig 5%, "
        "thr 0): "
        + (f"{zb['n_bets']} bets, ROI {zb['roi_pct']:+.2f}% CI "
           f"[{zb['roi_ci_pct'][0]:+.2f}, {zb['roi_ci_pct'][1]:+.2f}], "
           f"total PnL {zb['total_pnl_u']:+.1f}u" if zb else "0 bets"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"report -> {args.out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
