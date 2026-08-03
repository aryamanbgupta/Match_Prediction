"""B10 unit check — who-bowls usage alignment, verified before any sim run.

Five parts, ALL must pass:

  1. `scripts/auto/d15_unit_check.py` still passes on HEAD (subprocess). A
     failure here is a BLOCKING engine-state finding (the I5/I9 refactors
     broke the legacy path), not a B10 failure.
  2. Legacy parity — with the default `models/bowler_phase_usage.json` the
     selector has `_b10 is None`, its weight vector equals
     `phase_balls + k*league_share` recomputed independently from the raw
     payload (float-exact), and its live `select_bowler` reproduces the
     pre-B10 (HEAD) selection sequence draw-for-draw under a shared seed on
     >=20 real `data/polymarket_test` lineups.
  3. exp_balls parity — `sim_v1_2._B10AsOfExpBalls` matches
     `scripts/auto/b9_usage_baseline.AsOfUsage` on >=1,000 sampled
     (name, date) pairs, exact to 1e-12 for both `prior_balls` and
     `exp_balls`.
  4. Weight-mechanism table (deterministic, no sim) over the 261 test
     lineups: mean full-XI selection share, old vs new, for true debutants /
     veteran never-bowlers / known bowlers.
  5. md5 of `models/bowler_phase_usage.json` unchanged.

Run: uv run python scripts/auto/b10_unit_check.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402

from sim_v1_2 import (EmpiricalBowlerSelector, _B10AsOfExpBalls,  # noqa: E402
                      _B10_INNINGS_BALLS)
from sim_eval.loaders import TestMatchLoader  # noqa: E402

PROD_USAGE = REPO / "models" / "bowler_phase_usage.json"
B10_USAGE = REPO / "models" / "auto" / "b10" / "bowler_phase_usage_b10.json"
TEST_DIR = REPO / "data" / "polymarket_test"

FAILURES: list = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_b9():
    p = REPO / "scripts" / "auto" / "b9_usage_baseline.py"
    spec = importlib.util.spec_from_file_location("b9_usage_baseline", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["b9_usage_baseline"] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------- references
def ref_cumulative(payload, year):
    """Independent re-derivation of EmpiricalBowlerSelector._as_of."""
    cum = {}
    for cid, years in payload["by_player"].items():
        agg = {"pp": 0, "mid": 0, "death": 0, "total": 0}
        for y_str, counts in years.items():
            if int(y_str) < year:
                for kk in agg:
                    agg[kk] += counts.get(kk, 0)
        if agg["total"] > 0:
            cum[cid] = agg
    return cum


def ref_league(payload, year):
    by_year = payload.get("by_year_league", {})
    for y in (year, year - 1):
        entry = by_year.get(str(y))
        if entry and entry.get("total_balls", 0) > 0:
            return {"pp": entry["pp_share"], "mid": entry["mid_share"],
                    "death": entry["death_share"]}
    g = payload["global_league"]
    return {"pp": g["pp_share"], "mid": g["mid_share"],
            "death": g["death_share"]}


def ref_legacy_weights(payload, players, available, phase, year, k=30):
    """Verbatim copy of the pre-B10 (HEAD) weight loop + formula."""
    as_of = ref_cumulative(payload, year)
    alpha = k * ref_league(payload, year)[phase]
    out = []
    for idx in available:
        usage = as_of.get(players[idx].player_id)
        phase_balls = usage[phase] if usage else 0
        out.append(float(phase_balls) + alpha)
    return out


def head_select_bowler(weights, available):
    """Verbatim copy of the pre-B10 (HEAD) sampling tail of select_bowler."""
    total = sum(weights)
    if total <= 0:
        return random.choice(available)
    r = random.random() * total
    upto = 0.0
    for idx, w in zip(available, weights):
        upto += w
        if r <= upto:
            return idx
    return available[-1]


def phase_of(balls):
    return "pp" if balls < 36 else ("mid" if balls < 96 else "death")


# ------------------------------------------------------------------- loaders
def load_lineups(limit=None):
    """(match_id, date_iso, year, team_name, [Player,...]) per bowling side."""
    loader = TestMatchLoader()
    out = []
    files = sorted(TEST_DIR.glob("*.json"))
    if limit:
        files = files[:limit]
    for fp in files:
        with open(fp) as f:
            data = json.load(f)
        mid, state = loader._create_match_state(data, cricsheet_id=fp.stem)
        if state is None:
            continue
        d = state.match_date.date().isoformat()
        for lu in (state.team1_lineup, state.team2_lineup):
            out.append((mid, d, state.match_date.year, lu.team_name,
                        lu.players, state))
    return out


class _FakeState:
    """Minimal stand-in exposing what select_bowler reads."""

    def __init__(self, real_state, lineup, balls):
        self.match_date = real_state.match_date
        self.balls = balls
        self._lineup = lineup

    @property
    def bowling_lineup(self):
        return self._lineup


def main():
    print("== Part 1: d15_unit_check.py on HEAD ==")
    r = subprocess.run([sys.executable, str(REPO / "scripts/auto/d15_unit_check.py")],
                       capture_output=True, text=True, cwd=str(REPO))
    n_pass = r.stdout.count("[PASS]")
    n_fail = r.stdout.count("[FAIL]")
    check(f"d15 unit check {n_pass} PASS / {n_fail} FAIL, exit={r.returncode}",
          r.returncode == 0 and n_fail == 0 and n_pass == 30,
          "BLOCKING engine-state finding if FAIL" if r.returncode else "")
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print("STOP: d15 unit check failed on HEAD — blocking engine-state "
              "finding, not a B10 failure.")
        sys.exit(2)

    md5_before = md5(PROD_USAGE)
    print(f"\nmodels/bowler_phase_usage.json md5 BEFORE: {md5_before}")

    print("\n== Loading real lineups from data/polymarket_test ==")
    lineups = load_lineups()
    print(f"  {len(lineups)} team lineups from "
          f"{len(set(x[0] for x in lineups))} matches")

    # ------------------------------------------------------- Part 2: legacy
    print("\n== Part 2: legacy parity (default usage path) ==")
    payload = json.load(open(PROD_USAGE))
    sel = EmpiricalBowlerSelector(usage_path=str(PROD_USAGE))
    sel._ensure_b10()
    check("default payload leaves _b10 None", sel._b10 is None)
    check("default payload has no b10_asof_usage key",
          "b10_asof_usage" not in payload)

    rng = random.Random(7)
    n_lineups, n_cfg, max_abs = 0, 0, 0.0
    seq_mismatch = 0
    for (mid, date, year, team, players, state) in lineups[:40]:
        n_lineups += 1
        for balls in (0, 12, 30, 42, 66, 90, 102, 114):
            phase = phase_of(balls)
            n_all = len(players)
            drop = rng.randrange(n_all)
            available = [i for i in range(n_all) if i != drop]
            fs = _FakeState(state, type("L", (), {
                "players": players, "team_name": team})(), balls)
            # (a) selector weight vector vs independent recomputation
            as_of = sel._as_of(year)
            alpha = sel.k * sel._league_share(year)[phase]
            got = []
            for idx in available:
                usage = as_of.get(players[idx].player_id)
                got.append(float(usage[phase] if usage else 0) + alpha)
            want = ref_legacy_weights(payload, players, available, phase,
                                      year, k=sel.k)
            max_abs = max(max_abs, max(abs(a - b) for a, b in zip(got, want)))
            # (b) live select_bowler vs the HEAD sampling tail, same seed
            seed = rng.randrange(1 << 30)
            random.seed(seed)
            live = sel.select_bowler(fs, available)
            random.seed(seed)
            head = head_select_bowler(want, available)
            if live != head:
                seq_mismatch += 1
            n_cfg += 1
    check(f"weight vectors float-exact vs independent recomputation "
          f"({n_lineups} lineups x {n_cfg // max(1, n_lineups)} configs)",
          max_abs == 0.0, f"max |delta| = {max_abs:.3e}")
    check(f"live select_bowler == HEAD sampling on {n_cfg} same-seed draws",
          seq_mismatch == 0, f"mismatches = {seq_mismatch}")

    # --------------------------------------------------- Part 3: exp_balls
    print("\n== Part 3: exp_balls parity vs b9_usage_baseline.AsOfUsage ==")
    b9 = load_b9()
    import pickle
    corpus = pickle.load(open(REPO / "models/auto/b10/usage_corpus.pkl", "rb"))
    b9_asof = b9.AsOfUsage(corpus)
    cfg = json.load(open(B10_USAGE))["b10_asof_usage"]
    mine = _B10AsOfExpBalls(corpus, cfg["k_usage"])
    check("sidecar k_usage == b9_usage_baseline.K_USAGE",
          float(cfg["k_usage"]) == float(b9.K_USAGE),
          f"{cfg['k_usage']} vs {b9.K_USAGE}")

    names = sorted(corpus["player"])
    dates = sorted({d for (_, d, _, _, _, _) in lineups})
    rng2 = random.Random(11)
    pairs = [(rng2.choice(names), rng2.choice(dates)) for _ in range(1200)]
    # plus every real (name, date) pair from the first 20 lineups
    for (mid, date, year, team, players, state) in lineups[:20]:
        for p in players:
            pairs.append((p.name, date))
    d_prior, d_exp = 0.0, 0.0
    for name, date in pairs:
        pb_ref, _ = b9_asof.global_stats(date)
        n, b, _w = b9_asof.player_sums(name, date)
        eb_ref = (b9.K_USAGE * pb_ref + b) / (b9.K_USAGE + n) if n else pb_ref
        d_prior = max(d_prior, abs(mine.prior_balls(date) - pb_ref))
        d_exp = max(d_exp, abs(mine.exp_balls(name, date) - eb_ref))
    check(f"prior_balls parity on {len(pairs)} (name,date) pairs",
          d_prior < 1e-12, f"max |delta| = {d_prior:.3e}")
    check(f"exp_balls parity on {len(pairs)} (name,date) pairs",
          d_exp < 1e-12, f"max |delta| = {d_exp:.3e}")
    # n == 0 must return the prior EXACTLY (the `if n else` branch)
    zero_n = [(nm, dt) for nm, dt in pairs
              if b9_asof.player_sums(nm, dt)[0] == 0]
    check(f"n=0 rows return prior_balls exactly ({len(zero_n)} such pairs)",
          all(mine.exp_balls(nm, dt) == mine.prior_balls(dt)
              for nm, dt in zero_n))

    # ------------------------------------------- Part 4: mechanism table
    print("\n== Part 4: full-XI share mechanism table (261 lineups, no sim) ==")
    selb = EmpiricalBowlerSelector(usage_path=str(B10_USAGE))
    selb._ensure_b10()
    check("sidecar payload activates _b10", selb._b10 is not None)

    PHASE_W = {"pp": 36.0 / 120.0, "mid": 60.0 / 120.0, "death": 24.0 / 120.0}
    groups = ("true_debutant", "veteran_never_bowler", "known_bowler",
              "other_unknown")
    acc = {ph: {g: {"old": [], "new": []} for g in groups}
           for ph in list(PHASE_W) + ["blend"]}
    counts = defaultdict(int)
    n_relax_before = selb.b10_relaxation_triggers

    for (mid, date, year, team, players, state) in lineups:
        as_of = selb._as_of(year)
        idxs = list(range(len(players)))
        per_phase = {}
        for phase in PHASE_W:
            alpha = selb.k * selb._league_share(year)[phase]
            old = ref_legacy_weights(payload, players, idxs, phase, year,
                                     k=selb.k)
            new = selb._b10_share_weights(players, idxs, as_of, phase, alpha,
                                          date)
            so, sn = sum(old), sum(new)
            per_phase[phase] = ([x / so for x in old], [x / sn for x in new])
        for i, p in enumerate(players):
            known = as_of.get(p.player_id) is not None
            n_app, sum_balls = selb._b10.player_sums(p.name, date)
            if known:
                g = "known_bowler"
            elif n_app == 0:
                g = "true_debutant"
            elif n_app >= 20 and sum_balls == 0:
                g = "veteran_never_bowler"
            else:
                g = "other_unknown"
            counts[g] += 1
            bo, bn = 0.0, 0.0
            for phase, wgt in PHASE_W.items():
                o, n_ = per_phase[phase]
                acc[phase][g]["old"].append(o[i])
                acc[phase][g]["new"].append(n_[i])
                bo += wgt * o[i]
                bn += wgt * n_[i]
            acc["blend"][g]["old"].append(bo)
            acc["blend"][g]["new"].append(bn)

    print(f"\n  group sizes (player-lineup rows): "
          + ", ".join(f"{g}={counts[g]}" for g in groups))
    print(f"\n  {'phase':<8}{'group':<24}{'rows':>7}"
          f"{'mean share OLD':>16}{'mean share NEW':>16}{'ratio':>9}")
    for phase in list(PHASE_W) + ["blend"]:
        for g in groups:
            o = acc[phase][g]["old"]
            n_ = acc[phase][g]["new"]
            if not o:
                continue
            mo, mn = float(np.mean(o)), float(np.mean(n_))
            print(f"  {phase:<8}{g:<24}{len(o):>7}{mo*100:>15.3f}%"
                  f"{mn*100:>15.3f}%{(mn/mo if mo else float('nan')):>9.2f}")
    print(f"\n  relaxation triggers during the table pass: "
          f"{selb.b10_relaxation_triggers - n_relax_before}")

    deb_o = float(np.mean(acc["blend"]["true_debutant"]["old"]))
    deb_n = float(np.mean(acc["blend"]["true_debutant"]["new"]))
    vet_o = float(np.mean(acc["blend"]["veteran_never_bowler"]["old"]))
    vet_n = float(np.mean(acc["blend"]["veteran_never_bowler"]["new"]))
    kn_o = float(np.mean(acc["blend"]["known_bowler"]["old"]))
    kn_n = float(np.mean(acc["blend"]["known_bowler"]["new"]))
    check("true debutants get a LARGER share under B10 (plan: tiny -> ~9%)",
          deb_n > deb_o, f"{deb_o*100:.3f}% -> {deb_n*100:.3f}%")
    check("known bowlers keep ~their share (only renormalization)",
          abs(kn_n - kn_o) / kn_o < 0.25,
          f"{kn_o*100:.3f}% -> {kn_n*100:.3f}%")
    # Share-identity: by construction an unknown player's normalized full-XI
    # share must equal its as-of expected-balls share s_i exactly.
    mid_alpha = selb.k * selb._league_share(lineups[0][2])["mid"]
    _mid, _date, _yr = lineups[0][0], lineups[0][1], lineups[0][2]
    _players = lineups[0][4]
    _as_of = selb._as_of(_yr)
    _w = selb._b10_share_weights(_players, list(range(len(_players))),
                                 _as_of, "mid", mid_alpha, _date)
    _tot = sum(_w)
    _dmax = 0.0
    for i, p in enumerate(_players):
        if _as_of.get(p.player_id) is None:
            s_i = selb._b10.exp_balls(p.name, _date) / _B10_INNINGS_BALLS
            _dmax = max(_dmax, abs(_w[i] / _tot - s_i))
    check("unknown players' normalized share == as-of expected-balls share",
          _dmax < 1e-12, f"max |delta| = {_dmax:.3e}")

    print("\n  [INFO / MECHANISM FINDING — not a pass/fail condition]")
    print(f"  veteran never-bowlers (>=20 appearances, 0 career balls): "
          f"{vet_o*100:.3f}% -> {vet_n*100:.3f}% "
          f"(ratio {vet_n/vet_o:.2f})")
    print("  The plan expected this group to FALL toward ~0. It RISES: B9's")
    print("  exp_balls formula shrinks a 0-ball veteran to "
          "k_u*prior/(k_u+n) balls,")
    print("  which at k_u=5 and n~20-60 is still ~1-2 balls (~0.5-1.8% of an")
    print("  innings) — well above the legacy alpha share (~0.27%). The B9")
    print("  usage baseline itself prices that group at p_usage 1.30% vs")
    print("  p_sim 1.29% (B9 report table), i.e. it never fixed defect (b)")
    print("  either; the CI-clean flip came from defect (a) + the head. B10")
    print("  as specified therefore attacks (a) and MOVES (b) THE WRONG WAY.")

    # ------------------------------------------------------ Part 5: md5
    print("\n== Part 5: production prior untouched ==")
    md5_after = md5(PROD_USAGE)
    check(f"models/bowler_phase_usage.json md5 unchanged",
          md5_after == md5_before, f"{md5_before} == {md5_after}")

    print()
    if FAILURES:
        print(f"B10 UNIT CHECK FAILED: {len(FAILURES)} failure(s): {FAILURES}")
        sys.exit(1)
    print("B10 UNIT CHECK PASSED: all assertions hold.")


if __name__ == "__main__":
    main()
