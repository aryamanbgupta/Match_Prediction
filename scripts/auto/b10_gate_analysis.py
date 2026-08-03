"""B10 gate analysis — who-bowls usage alignment (blind twin vs b10 arm).

WRITTEN AND COMMITTED BEFORE EITHER EVAL RESULT EXISTED.

Two prop_backtest detail JSONs, both fresh in-session at seed 43, n=261 x 100
sims, venue-ON default path, `--ball-calibrator vector`, identical model /
encoders / calibrator. ONLY delta = the bowler-selection weighting for players
ABSENT from `models/bowler_phase_usage.json`:

  blind  models/auto/b10/detail_blind_s43_n261.json
         current default path (flat alpha = k * league_share for unknowns)
  b10    models/auto/b10/detail_b10_s43_n261.json
         unknowns re-weighted to their B9 as-of expected-balls share
         (`--bowler-usage-path models/auto/b10/bowler_phase_usage_b10.json`)

The D15 detail is NOT the paired baseline (sim_v1_2.py was refactored after it
by 7f159a5 / f846484 / f766476); it is used only for a descriptive drift check.

PRE-COMMITTED GATE (per research/handoff/B10/plan.md):

  GATE 1 (primary, both required):
    top_bowler Brier improves CI-clean paired (b10 - blind, cluster boot by
    match, CI upper bound < 0) AND G5 bowler coverage in the b10 arm >= 0.90.
  GATE 2 (guards): no CI-clean regression (CI lower bound > 0) on
    bowler_wkts_1plus, bowler_wkts_2plus, batter_runs_mae,
    team_first_over_mae.

  Both -> LANDED; exactly one -> TABLED; none -> FAILED.
  This script PRINTS the mapping; the ORCHESTRATOR issues the verdict.

PAIRING NOTE (pre-committed): the statistic is identical to D15/A8 (paired
per-row delta, cluster bootstrap by match, 2000 resamples, seed 29). Rows are
matched by IDENTITY (team, name) rather than positionally, because
`prop_backtest.aggregate_per_player` emits bowler_wkts_* rows in
first-appearance order across sims — an order the selector change itself can
permute. The positional (D15-identical) numbers are printed alongside as a
cross-check.

Run:
  uv run python scripts/auto/b10_gate_analysis.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "auto"))

from a8_gate_analysis import cluster_boot  # noqa: E402

N_BOOT = 2000
BOOT_SEED = 29

GATE1_FAMILY = "top_bowler"
GUARD_BINARY = ["bowler_wkts_1plus", "bowler_wkts_2plus"]
GUARD_MAE = ["batter_runs_mae", "team_first_over_mae"]
DRIFT_FAMILIES = [GATE1_FAMILY] + GUARD_BINARY + GUARD_MAE

G5_MIN_COVERAGE = 0.90


# --------------------------------------------------------------- IO helpers
def load(path):
    return json.load(open(path))


def rekey(detail, field):
    """Return a copy of `detail` with match_id replaced by `field`."""
    out = []
    for r in detail:
        if field not in r:
            continue
        out.append({"match_id": r[field], "obs": r["obs"]})
    return out


def is_mae(fam):
    return fam.endswith("_mae")


def row_key(r, i):
    t, n = r.get("team"), r.get("name")
    if t is None and n is None:
        return ("__pos__", i)
    return (t, n)


def keyed_rows(det_a, det_b, fam):
    """[(mid, actual, val_a, val_b)] matched on (team, name) identity."""
    vk, ak = ("sim_mean", "actual") if is_mae(fam) else ("p", "y")
    idx_b = {r["match_id"]: r["obs"] for r in det_b}
    rows, dropped = [], 0
    for ra in det_a:
        mid = ra["match_id"]
        oa = ra["obs"].get(fam) or []
        ob = idx_b.get(mid, {}).get(fam) or []
        mb = {row_key(x, i): x for i, x in enumerate(ob)}
        for i, xa in enumerate(oa):
            xb = mb.get(row_key(xa, i))
            if xb is None or xa.get(ak) != xb.get(ak):
                dropped += 1
                continue
            rows.append((mid, float(xa[ak]), float(xa[vk]), float(xb[vk])))
    return rows, dropped


def positional_rows(det_a, det_b, fam):
    """D15-identical positional pairing (cross-check only)."""
    vk, ak = ("sim_mean", "actual") if is_mae(fam) else ("p", "y")
    idx_b = {r["match_id"]: r["obs"] for r in det_b}
    rows = []
    for ra in det_a:
        mid = ra["match_id"]
        oa = ra["obs"].get(fam)
        ob = idx_b.get(mid, {}).get(fam)
        if not oa or not ob or len(oa) != len(ob):
            continue
        for xa, xb in zip(oa, ob):
            if xa.get(ak) != xb.get(ak):
                continue
            rows.append((mid, float(xa[ak]), float(xa[vk]), float(xb[vk])))
    return rows


def paired_stat(rows, fam):
    """(score_a, score_b, delta(b-a), lo, hi, n). Brier or MAE by family."""
    if is_mae(fam):
        sa = float(np.mean([abs(a - y) for _, y, a, _ in rows]))
        sb = float(np.mean([abs(b - y) for _, y, _, b in rows]))
        fn = lambda r: abs(r[3] - r[1]) - abs(r[2] - r[1])  # noqa: E731
    else:
        sa = float(np.mean([(a - y) ** 2 for _, y, a, _ in rows]))
        sb = float(np.mean([(b - y) ** 2 for _, y, _, b in rows]))
        fn = lambda r: (r[3] - r[1]) ** 2 - (r[2] - r[1]) ** 2  # noqa: E731
    lo, hi = cluster_boot(rows, fn, n_boot=N_BOOT, seed=BOOT_SEED)
    return sa, sb, sb - sa, lo, hi, len(rows)


def flag_of(lo, hi):
    return "DOWN(better)" if hi < 0 else "UP(worse)" if lo > 0 else "~noise"


def report(det_a, det_b, fams, la, lb, show_positional=False):
    unit = {}
    print(f"{'family':<34}{'n':>6}{'drop':>6}{la:>12}{lb:>12}"
          f"{'delta':>10}   95% CI ({lb}-{la})   flag")
    for fam in fams:
        rows, dropped = keyed_rows(det_a, det_b, fam)
        if not rows:
            continue
        sa, sb, d, lo, hi, n = paired_stat(rows, fam)
        unit[fam] = (d, lo, hi, flag_of(lo, hi), n, dropped)
        print(f"{fam:<34}{n:>6}{dropped:>6}{sa:>12.4f}{sb:>12.4f}{d:>+10.4f}"
              f"   [{lo:+.4f},{hi:+.4f}]  {flag_of(lo, hi)}")
        if show_positional:
            prows = positional_rows(det_a, det_b, fam)
            if prows:
                _, _, pd_, plo, phi, pn = paired_stat(prows, fam)
                print(f"{'  (positional cross-check)':<34}{pn:>6}{'':>6}"
                      f"{'':>12}{'':>12}{pd_:>+10.4f}"
                      f"   [{plo:+.4f},{phi:+.4f}]  {flag_of(plo, phi)}")
    return unit


# ------------------------------------------------------------ G5 coverage
def real_bowlers(test_dir: Path):
    """cricsheet_id -> {bowler name: deliveries} from the actual match JSONs."""
    out = {}
    for fp in sorted(test_dir.glob("*.json")):
        data = json.load(open(fp))
        cnt = defaultdict(int)
        for inn in data.get("innings", []):
            for ov in inn.get("overs", []):
                for d in ov.get("deliveries", []):
                    cnt[d["bowler"]] += 1
        out[fp.stem] = dict(cnt)
    return out


def g5_coverage(detail, truth, label):
    """Fraction of real match bowlers with p_sim > 0 on bowler_wkts_1plus."""
    tot = hit = 0
    missing_matches = 0
    for r in detail:
        cid = r.get("cricsheet_id")
        real = truth.get(str(cid))
        if real is None:
            missing_matches += 1
            continue
        priced = {x["name"]: float(x["p"])
                  for x in r["obs"].get("bowler_wkts_1plus", [])}
        for name in real:
            tot += 1
            if priced.get(name, 0.0) > 0.0:
                hit += 1
    cov = hit / tot if tot else float("nan")
    print(f"  G5 coverage [{label}]: {hit}/{tot} = {cov:.4f}"
          + (f"  (unmatched matches: {missing_matches})"
             if missing_matches else ""))
    return cov


# ------------------------------------------------ B9 sim-vs-usage margin
def load_b9():
    p = REPO / "scripts" / "auto" / "b9_usage_baseline.py"
    spec = importlib.util.spec_from_file_location("b9_usage_baseline", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["b9_usage_baseline"] = mod
    spec.loader.exec_module(mod)
    return mod


def b9_margin(b9, detail, label):
    """Recompute B9's paired dBrier (sim - usage) on a detail JSON.

    B9 derives the as-of date from match_id[:10], so the detail must be
    re-keyed to the display match id (post-I15 runs key on cricsheet id).
    """
    pfb = b9._load_pfb()
    asof_pfb = pfb.AsOf(pickle.load(open(pfb.CACHE, "rb")))
    corpus = pickle.load(open(REPO / "models/auto/b10/usage_corpus.pkl", "rb"))
    asof_use = b9.AsOfUsage(corpus)
    det = rekey(detail, "display_match_id") if "display_match_id" in detail[0] \
        else detail
    markets = b9.build_markets(det, asof_pfb, asof_use, b9.K_USAGE, b9.K_RATE)
    rows = b9.flat_rows(markets)
    m, lo, hi, n = b9.paired_dbrier(rows, "p_sim", "p_usage")
    bs = b9.brier(rows, "p_sim")
    bu = b9.brier(rows, "p_usage")
    print(f"  [{label}] rows={n}  Brier_sim={bs:.4f}  Brier_usage={bu:.4f}  "
          f"sim-usage={m:+.4f} CI [{lo:+.4f},{hi:+.4f}]  {flag_of(lo, hi)}")
    # head-only (both >= 2%)
    head = [r for r in rows if r["p_sim"] >= 0.02 and r["p_usage"] >= 0.02]
    hm, hlo, hhi, hn = b9.paired_dbrier(head, "p_sim", "p_usage")
    print(f"  [{label}] head-only (both p>=2%) rows={hn}  "
          f"sim-usage={hm:+.4f} CI [{hlo:+.4f},{hhi:+.4f}]  "
          f"{flag_of(hlo, hhi)}")
    return m, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blind",
                    default=str(REPO / "models/auto/b10/detail_blind_s43_n261.json"))
    ap.add_argument("--b10",
                    default=str(REPO / "models/auto/b10/detail_b10_s43_n261.json"))
    ap.add_argument("--d15",
                    default=str(REPO / "models/auto/d15/detail_d15_s43_n261.json"))
    ap.add_argument("--b10-log",
                    default=str(REPO / "research/handoff/B10/raw/run_b10.log"))
    ap.add_argument("--test-dir", default=str(REPO / "data/polymarket_test"))
    ap.add_argument("--skip-b9", action="store_true")
    args = ap.parse_args()

    da, db = load(args.blind), load(args.b10)
    print(f"blind: {len(da)} matches | b10: {len(db)} matches\n")

    print("=" * 96)
    print("GATE 1 — top_bowler paired Brier (b10 - blind) must be CI-clean "
          "negative, AND b10 G5 coverage >= 0.90")
    print("=" * 96)
    g1 = report(da, db, [GATE1_FAMILY], "blind", "b10", show_positional=True)
    print()
    cov_blind = g5_coverage(da, real_bowlers(Path(args.test_dir)), "blind")
    cov_b10 = g5_coverage(db, real_bowlers(Path(args.test_dir)), "b10")
    tb = g1.get(GATE1_FAMILY)
    g1_brier = bool(tb and tb[2] < 0)
    g1_cov = bool(cov_b10 >= G5_MIN_COVERAGE)
    gate1 = g1_brier and g1_cov
    print(f"\n  top_bowler CI-clean improvement: "
          f"{'MET' if g1_brier else 'NOT MET'}")
    print(f"  b10 G5 coverage >= {G5_MIN_COVERAGE}: "
          f"{'MET' if g1_cov else 'NOT MET'}")
    print(f"  GATE 1: {'MET' if gate1 else 'NOT MET'}")

    print("\n" + "=" * 96)
    print("GATE 2 — guards: no CI-clean regression")
    print("=" * 96)
    g2 = report(da, db, GUARD_BINARY + GUARD_MAE, "blind", "b10",
                show_positional=True)
    gate2 = (len(g2) == len(GUARD_BINARY) + len(GUARD_MAE)
             and all(not (v[1] > 0) for v in g2.values()))
    for fam, v in g2.items():
        print(f"  {fam:<28} {'REGRESSED CI-clean' if v[1] > 0 else 'ok'}")
    print(f"  GATE 2: {'MET' if gate2 else 'NOT MET'}")

    print("\n" + "=" * 96)
    print("CONTEXT — full family scan (cannot flip the verdict)")
    print("=" * 96)
    all_fams = sorted(set(da[0]["obs"]) - {"cricsheet_id", "display_match_id",
                                           "match_identity_version"})
    scan = report(da, db, all_fams, "blind", "b10")
    clean = [(f, v) for f, v in scan.items() if v[1] > 0 or v[2] < 0]
    print("\n  CI-excludes-0 families (either direction):")
    if not clean:
        print("    (none)")
    for f, v in sorted(clean, key=lambda x: x[1][0]):
        print(f"    {f:<34}{v[0]:>+10.4f}  [{v[1]:+.4f},{v[2]:+.4f}]  {v[3]}")

    print("\n" + "=" * 96)
    print("CONTEXT — B9 sim-vs-usage top_bowler margin (how much of the "
          "+0.0038 gap closed)")
    print("=" * 96)
    if not args.skip_b9:
        try:
            b9 = load_b9()
            b9_margin(b9, da, "blind")
            b9_margin(b9, db, "b10")
            print("  (B9 headline on the D15 detail was +0.0038 "
                  "[+0.0026,+0.0051]; head-only +0.0049 [+0.0032,+0.0067])")
        except Exception as e:  # pragma: no cover
            print(f"  B9 recomputation failed: {type(e).__name__}: {e}")

    print("\n" + "=" * 96)
    print("CONTEXT — B10 relaxation triggers (from the b10 run log)")
    print("=" * 96)
    lp = Path(args.b10_log)
    if lp.exists():
        txt = lp.read_text(errors="replace")
        n_relax = txt.count("B10 relaxation triggered")
        n_active = txt.count("B10 usage-aligned bowler selector ACTIVE")
        print(f"  'B10 relaxation triggered' lines: {n_relax}")
        print(f"  'B10 ... ACTIVE' startup lines:   {n_active}")
    else:
        print(f"  log not found: {lp}")

    print("\n" + "=" * 96)
    print("CONTEXT — drift check: blind arm vs the pre-refactor D15 detail "
          "(same seed 43; DESCRIPTIVE ONLY)")
    print("=" * 96)
    d15p = Path(args.d15)
    if d15p.exists():
        d15 = load(d15p)
        blind_disp = rekey(da, "display_match_id")
        if not blind_disp:
            print("  blind detail has no display_match_id — cannot re-key")
        else:
            report(d15, blind_disp, DRIFT_FAMILIES, "d15", "blind")
    else:
        print(f"  D15 detail not found: {d15p}")

    print("\n" + "=" * 96)
    print(f"GATE 1: {'MET' if gate1 else 'NOT MET'} | "
          f"GATE 2: {'MET' if gate2 else 'NOT MET'}")
    mapping = ("LANDED" if gate1 and gate2
               else "TABLED" if gate1 or gate2 else "FAILED")
    print(f"Pre-committed verdict MAPPING (orchestrator decides): {mapping}")


if __name__ == "__main__":
    main()
