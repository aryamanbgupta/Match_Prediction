"""B13 gate analysis — never-bowler damping in the usage-absent branch.

WRITTEN AND COMMITTED BEFORE ANY B13 EVAL RESULT EXISTED.

B10/B12 (LANDED, shipped) aligned the usage-ABSENT branch of
`EmpiricalBowlerSelector` to B9's as-of expected-balls share. That fixed
defect (a) but mechanically WORSENED defect (b): a veteran never-bowler
(n>=20 XI appearances, 0 corpus balls) still draws
`k_u*prior/(k_u+n)` ~ 1-2 balls, so the cohort's share ROSE 0.270% -> 0.496%
(actual ~0). B13 damps exactly that cohort with
`P(bowls|n, 0 balls) = k_damp/(k_damp+n)` times `mu_active`, leaving n=0
debutants and usage-present players byte-untouched.

Two prop_backtest detail JSONs, seed 46, n=261 x 100 sims, promoted i7
no-weights RAW stack (NO calibrator), venue encoder ACTIVE (373 venues),
run-out channel ACTIVE, B10 selector ACTIVE. ONLY delta = the B13 damping:

  blind  models/auto/d16/detail_noweights_raw_s46_n261.json
         the canonical production-stack baseline (program.md recipe B); its
         run log shows the same engine + selector state, and
         `git diff ea4acdb..HEAD -- scripts/sim_v1_2.py` was EMPTY at claim
         time, so the pairing is damping-only.
  b13    models/auto/b13/detail_b13_s46_n261.json
         `--bowler-usage-path models/auto/b13/bowler_phase_usage_b13.json`

PRE-COMMITTED GATE (per research/handoff/B13/plan.md, verbatim from the
IDEAS B13 entry):

  GATE 1 (PRIMARY, both conjuncts required):
    1a. `top_bowler` Brier improves CI-clean paired (b13 - blind, CI < 0).
    1b. The recomputed sim-usage margin SHRINKS vs the blind arm
        (margin = sim top_bowler Brier - B9 usage-baseline top_bowler Brier,
        both computed on the promoted stack; point comparison, the paired CI
        is reported as context).
  GATE 2 (guards): `bowler_wkts_1plus`, `bowler_wkts_2plus`,
    `batter_runs_mae`, `team_first_over_mae` — no CI-clean regression.

  Both -> LANDED; exactly one -> TABLED; none -> FAILED.
  This script PRINTS the mapping; the ORCHESTRATOR issues the verdict.

PAIRING (identical machinery to B10/B12/D15/A8): paired per-row delta,
cluster bootstrap by match, 2000 resamples, seed 29. Rows matched by IDENTITY
(team, name) because `prop_backtest.aggregate_per_player` emits bowler rows in
first-appearance order across sims — an order the selector change itself can
permute. Positional (D15-identical) numbers printed alongside as a cross-check.

Run:
  uv run python scripts/auto/b13_gate_analysis.py
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

G5_REPORT_FLOOR = 0.90  # context only in B13


# --------------------------------------------------------------- IO helpers
def load(path):
    return json.load(open(path))


def rekey(detail, field):
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
    Returns (margin, lo, hi, brier_sim, brier_usage).
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
    print(f"  [{label}] rows={n}  Brier_sim={bs:.6f}  Brier_usage={bu:.6f}  "
          f"sim-usage={m:+.6f} CI [{lo:+.4f},{hi:+.4f}]  {flag_of(lo, hi)}")
    head = [r for r in rows if r["p_sim"] >= 0.02 and r["p_usage"] >= 0.02]
    hm, hlo, hhi, hn = b9.paired_dbrier(head, "p_sim", "p_usage")
    print(f"  [{label}] head-only (both p>=2%) rows={hn}  "
          f"sim-usage={hm:+.6f} CI [{hlo:+.4f},{hhi:+.4f}]  "
          f"{flag_of(hlo, hhi)}")
    return m, lo, hi, bs, bu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blind",
                    default=str(REPO / "models/auto/d16/detail_noweights_raw_s46_n261.json"))
    ap.add_argument("--b13",
                    default=str(REPO / "models/auto/b13/detail_b13_s46_n261.json"))
    ap.add_argument("--b13-log",
                    default=str(REPO / "research/handoff/B13/raw/run_b13_s46.log"))
    ap.add_argument("--test-dir", default=str(REPO / "data/polymarket_test"))
    ap.add_argument("--skip-b9", action="store_true")
    args = ap.parse_args()

    da, db = load(args.blind), load(args.b13)
    print(f"blind: {len(da)} matches | b13: {len(db)} matches  (seed 46)\n")

    print("=" * 96)
    print(f"GATE 1a (PRIMARY) — {GATE1_FAMILY} paired Brier (b13 - blind) "
          "must be CI-clean negative")
    print("=" * 96)
    g1 = report(da, db, [GATE1_FAMILY], "blind", "b13", show_positional=True)
    tb = g1.get(GATE1_FAMILY)
    gate1a = bool(tb and tb[2] < 0)
    print(f"\n  GATE 1a: {'MET' if gate1a else 'NOT MET'}")

    print("\n" + "=" * 96)
    print("GATE 1b — recomputed B9 sim-usage top_bowler margin must SHRINK "
          "(margin_b13 < margin_blind)")
    print("=" * 96)
    gate1b = None
    if args.skip_b9:
        print("  (skipped by flag)")
    else:
        try:
            b9 = load_b9()
            m_blind, lo_a, hi_a, bs_a, bu_a = b9_margin(b9, da, "blind")
            m_b13, lo_b, hi_b, bs_b, bu_b = b9_margin(b9, db, "b13")
            print(f"\n  margin_blind = {m_blind:+.6f}  "
                  f"(Brier_sim {bs_a:.6f} - Brier_usage {bu_a:.6f})")
            print(f"  margin_b13   = {m_b13:+.6f}  "
                  f"(Brier_sim {bs_b:.6f} - Brier_usage {bu_b:.6f})")
            print(f"  change       = {m_b13 - m_blind:+.6f}")
            gate1b = bool(m_b13 < m_blind)
            print(f"\n  GATE 1b: {'MET' if gate1b else 'NOT MET'} "
                  "(point comparison, pre-committed)")
            print("  (B9 headline on the D15 legacy detail was +0.0038 "
                  "[+0.0026,+0.0051]; B12 blind->b10 on the legacy stack "
                  "moved +0.0028 -> +0.0026)")
        except Exception as e:  # pragma: no cover
            print(f"  B9 recomputation failed: {type(e).__name__}: {e}")

    gate1 = bool(gate1a and gate1b)
    print(f"\n  GATE 1 (1a AND 1b): {'MET' if gate1 else 'NOT MET'}")

    print("\n" + "=" * 96)
    print("GATE 2 — guards: no CI-clean regression on "
          + ", ".join(GUARD_BINARY + GUARD_MAE))
    print("=" * 96)
    g2 = report(da, db, GUARD_BINARY + GUARD_MAE, "blind", "b13",
                show_positional=True)
    gate2 = (len(g2) == len(GUARD_BINARY) + len(GUARD_MAE)
             and all(not (v[1] > 0) for v in g2.values()))
    for fam, v in g2.items():
        print(f"  {fam:<28} {'REGRESSED CI-clean' if v[1] > 0 else 'ok'}")
    print(f"  GATE 2: {'MET' if gate2 else 'NOT MET'}")

    print("\n" + "=" * 96)
    print(f"CONTEXT — G5 bowler coverage, both arms (NOT a B13 gate; B10 used "
          f"a >= {G5_REPORT_FLOOR} floor)")
    print("=" * 96)
    truth = real_bowlers(Path(args.test_dir))
    g5_coverage(da, truth, "blind")
    g5_coverage(db, truth, "b13")

    print("\n" + "=" * 96)
    print("CONTEXT — full family scan (cannot flip the verdict)")
    print("=" * 96)
    all_fams = sorted(set(da[0]["obs"]) - {"cricsheet_id", "display_match_id",
                                           "match_identity_version"})
    scan = report(da, db, all_fams, "blind", "b13")
    clean = [(f, v) for f, v in scan.items() if v[1] > 0 or v[2] < 0]
    n_better = sum(1 for _, v in clean if v[2] < 0)
    n_worse = sum(1 for _, v in clean if v[1] > 0)
    print(f"\n  families scanned: {len(scan)}")
    print(f"  CI-excludes-0 families: {len(clean)} "
          f"({n_better} better, {n_worse} worse)")
    if not clean:
        print("    (none)")
    for f, v in sorted(clean, key=lambda x: x[1][0]):
        print(f"    {f:<34}{v[0]:>+10.4f}  [{v[1]:+.4f},{v[2]:+.4f}]  {v[3]}")

    print("\n" + "=" * 96)
    print("CONTEXT — B10 relaxation triggers / B13 banner (from the b13 log)")
    print("=" * 96)
    lp = Path(args.b13_log)
    if lp.exists():
        txt = lp.read_text(errors="replace")
        print(f"  'B10 relaxation triggered' lines:  "
              f"{txt.count('B10 relaxation triggered')}")
        print(f"  'B10 ... ACTIVE' startup lines:     "
              f"{txt.count('B10 usage-aligned bowler selector ACTIVE')}")
        print(f"  'B13 never-bowler damping ACTIVE':  "
              f"{txt.count('B13 never-bowler damping ACTIVE')}")
        print(f"  'Ball calibrator' lines (must be 0): "
              f"{txt.count('Ball calibrator')}")
    else:
        print(f"  log not found: {lp}")

    print("\n" + "=" * 96)
    print(f"GATE 1a: {'MET' if gate1a else 'NOT MET'} | "
          f"GATE 1b: {'MET' if gate1b else 'NOT MET'} | "
          f"GATE 1: {'MET' if gate1 else 'NOT MET'} | "
          f"GATE 2: {'MET' if gate2 else 'NOT MET'}")
    mapping = ("LANDED" if gate1 and gate2
               else "TABLED" if gate1 or gate2 else "FAILED")
    print(f"Pre-committed verdict MAPPING (orchestrator decides): {mapping}")


if __name__ == "__main__":
    main()
