"""B13 unit check — never-bowler damping, verified before any sim run.

Mirrors `scripts/auto/b10_unit_check.py`'s lineup battery / weight-table
construction over the 261 `data/polymarket_test` matches (that file is NOT
edited — its md5 pin is pre-ship-stale by design). Three arms per lineup and
phase:

  legacy   flat alpha floor for usage-absent players (pre-B10; == the
           `force_legacy=True` relaxation path)
  b10      shipped default `models/bowler_phase_usage.json` (B10/B12 active,
           no B13 key)
  b13      `models/auto/b13/bowler_phase_usage_b13.json` (B13 key present)

PASS requires ALL of:

  1. Cohort (b) — >=20 prior appearances, 0 prior balls: b13 mean full-XI
     share < 0.10% (legacy alpha gave 0.270%, b10 gave 0.496%).
  2. Cohort (a) — true debutants (0 prior appearances): b13 within +-0.5pp
     of b10 (the n=0 path is untouched).
  3. Usage-PRESENT players' weights float-equal b10 vs b13 in every battery
     lineup (their `legacy[i]` weights never pass through the scale).
  4. Inertness without the key: on the production json the damped branch is
     never taken (`_b13_k_damp is None`, `b13_damped_rows == 0`) AND the live
     b10 weight vectors are float-exact against an INDEPENDENT re-derivation
     of the pre-B13 B10 formula, and `select_bowler` reproduces the same
     same-seed draws.
  5. Context (not pass/fail): relaxation-trigger counts, b10 vs b13, over the
     battery; plus the relaxation-aware ("effective") cohort table.

Part 0 re-runs `scripts/auto/d15_unit_check.py` on HEAD: the B13 edit touches
`sim_v1_2.py`, so a failure there is a BLOCKING engine-state finding.

Run: uv run python scripts/auto/b13_unit_check.py
"""
from __future__ import annotations

import hashlib
import json
import pickle
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
PROD_USAGE_MD5 = "2e650423f0c949631fca1f15dd1c8a56"
B13_USAGE = REPO / "models" / "auto" / "b13" / "bowler_phase_usage_b13.json"
CORPUS = REPO / "models" / "b10_usage_corpus.pkl"
TEST_DIR = REPO / "data" / "polymarket_test"

PHASE_W = {"pp": 36.0 / 120.0, "mid": 60.0 / 120.0, "death": 24.0 / 120.0}
GROUPS = ("true_debutant", "veteran_never_bowler", "known_bowler",
          "other_unknown")

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


# --------------------------------------------------------------- references
def ref_cumulative(payload, year):
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


def ref_b10_weights(payload, asof, players, available, phase, year, date,
                    k=30):
    """INDEPENDENT re-derivation of the pre-B13 B10 share weights."""
    as_of = ref_cumulative(payload, year)
    alpha = k * ref_league(payload, year)[phase]
    legacy, is_unknown, shares = [], [], []
    for idx in available:
        p = players[idx]
        usage = as_of.get(p.player_id)
        legacy.append(float(usage[phase] if usage else 0) + alpha)
        if usage is None:
            is_unknown.append(True)
            shares.append(asof.exp_balls(p.name, date) / _B10_INNINGS_BALLS)
        else:
            is_unknown.append(False)
            shares.append(0.0)
    if not any(is_unknown):
        return legacy
    w_known = sum(w for w, u in zip(legacy, is_unknown) if not u)
    s_unknown = sum(s for s, u in zip(shares, is_unknown) if u)
    scale = (w_known / max(1.0 - s_unknown, 0.05)) if w_known > 0.0 else 1.0
    out = [(shares[i] * scale) if is_unknown[i] else legacy[i]
           for i in range(len(available))]
    return [w if w > 1e-9 else 1e-9 for w in out]


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
    def __init__(self, real_state, lineup, balls):
        self.match_date = real_state.match_date
        self.balls = balls
        self._lineup = lineup

    @property
    def bowling_lineup(self):
        return self._lineup


def relaxed_for(sel, w, cfg):
    total = sum(w)
    n_elig = sum(1 for x in w if total > 0 and (x / total) > cfg["min_share"])
    return n_elig < cfg["min_eligible"], n_elig


def main():
    print("== Part 0: d15_unit_check.py on HEAD (B13 edits sim_v1_2.py) ==")
    r = subprocess.run([sys.executable, str(REPO / "scripts/auto/d15_unit_check.py")],
                       capture_output=True, text=True, cwd=str(REPO))
    n_pass = r.stdout.count("[PASS]")
    n_fail = r.stdout.count("[FAIL]")
    check(f"d15 unit check {n_pass} PASS / {n_fail} FAIL, exit={r.returncode}",
          r.returncode == 0 and n_fail == 0 and n_pass == 30,
          "BLOCKING engine-state finding if FAIL" if r.returncode else "")
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print("STOP: d15 unit check failed on HEAD — blocking engine-state "
              "finding, not a B13 failure.")
        sys.exit(2)

    md5_before = md5(PROD_USAGE)
    print(f"\nmodels/bowler_phase_usage.json md5 BEFORE: {md5_before}")
    check("production prior md5 is the B12-shipped one",
          md5_before == PROD_USAGE_MD5, md5_before)

    prod_payload = json.load(open(PROD_USAGE))
    b13_payload = json.load(open(B13_USAGE))
    check("production prior carries NO b13 key",
          "b13_never_bowler_damping" not in prod_payload["b10_asof_usage"])
    b13cfg = b13_payload["b10_asof_usage"].get("b13_never_bowler_damping")
    check("sidecar carries the b13 key nested in b10_asof_usage",
          b13cfg is not None, json.dumps(b13cfg))
    for k in ("corpus_path", "k_usage", "min_eligible", "min_share"):
        check(f"sidecar b10 cfg field '{k}' unchanged",
              b13_payload["b10_asof_usage"][k]
              == prod_payload["b10_asof_usage"][k],
              str(b13_payload["b10_asof_usage"][k]))

    print("\n== Loading real lineups from data/polymarket_test ==")
    lineups = load_lineups()
    print(f"  {len(lineups)} team lineups from "
          f"{len(set(x[0] for x in lineups))} matches")

    sel10 = EmpiricalBowlerSelector(usage_path=str(PROD_USAGE))
    sel10._ensure_b10()
    sel13 = EmpiricalBowlerSelector(usage_path=str(B13_USAGE))
    sel13._ensure_b10()
    check("production json leaves _b13_k_damp None",
          sel10._b13_k_damp is None and sel10._b13_mu_active is None)
    check("sidecar json activates the B13 damping",
          sel13._b13_k_damp is not None and sel13._b13_mu_active is not None,
          f"k_damp={sel13._b13_k_damp}, mu_active={sel13._b13_mu_active}")
    cfg = sel13._b10_cfg

    corpus = pickle.load(open(CORPUS, "rb"))
    ref_asof = _B10AsOfExpBalls(corpus, prod_payload["b10_asof_usage"]["k_usage"])

    # ------------------------------------------- Parts 1-3: weight tables
    print("\n== Parts 1-3: full-XI share tables, 3 arms (no sim) ==")
    acc = {ph: {g: {"legacy": [], "b10": [], "b13": [], "b13eff": []}
                for g in GROUPS}
           for ph in list(PHASE_W) + ["blend"]}
    counts = defaultdict(int)
    max_known_delta = 0.0
    max_ref_delta = 0.0
    relax_b10 = relax_b13 = 0
    n_lineup_phase = 0

    for (mid, date, year, team, players, state) in lineups:
        as_of10 = sel10._as_of(year)
        as_of13 = sel13._as_of(year)
        idxs = list(range(len(players)))
        per_phase = {}
        for phase in PHASE_W:
            alpha10 = sel10.k * sel10._league_share(year)[phase]
            alpha13 = sel13.k * sel13._league_share(year)[phase]
            leg = ref_legacy_weights(prod_payload, players, idxs, phase, year,
                                     k=sel10.k)
            w10 = sel10._b10_share_weights(players, idxs, as_of10, phase,
                                           alpha10, date)
            w13 = sel13._b10_share_weights(players, idxs, as_of13, phase,
                                           alpha13, date)
            # independent re-derivation of the pre-B13 B10 formula
            ref10 = ref_b10_weights(prod_payload, ref_asof, players, idxs,
                                    phase, year, date, k=sel10.k)
            max_ref_delta = max(max_ref_delta,
                                max(abs(a - b) for a, b in zip(w10, ref10)))
            # PASS 3: usage-present players' weights float-equal b10 vs b13
            for i, p in enumerate(players):
                if as_of10.get(p.player_id) is not None:
                    max_known_delta = max(max_known_delta, abs(w10[i] - w13[i]))
            rl10, _ = relaxed_for(sel10, w10, cfg)
            rl13, _ = relaxed_for(sel13, w13, cfg)
            relax_b10 += int(rl10)
            relax_b13 += int(rl13)
            n_lineup_phase += 1
            w13eff = leg if rl13 else w13
            sl, s10, s13, s13e = sum(leg), sum(w10), sum(w13), sum(w13eff)
            per_phase[phase] = ([x / sl for x in leg],
                                [x / s10 for x in w10],
                                [x / s13 for x in w13],
                                [x / s13e for x in w13eff])
        for i, p in enumerate(players):
            known = as_of10.get(p.player_id) is not None
            n_app, sum_balls = sel13._b10.player_sums(p.name, date)
            if known:
                g = "known_bowler"
            elif n_app == 0:
                g = "true_debutant"
            elif n_app >= 20 and sum_balls == 0:
                g = "veteran_never_bowler"
            else:
                g = "other_unknown"
            counts[g] += 1
            b = {"legacy": 0.0, "b10": 0.0, "b13": 0.0, "b13eff": 0.0}
            for phase, wgt in PHASE_W.items():
                cols = per_phase[phase]
                for ci, arm in enumerate(("legacy", "b10", "b13", "b13eff")):
                    acc[phase][g][arm].append(cols[ci][i])
                    b[arm] += wgt * cols[ci][i]
            for arm in b:
                acc["blend"][g][arm].append(b[arm])

    print(f"\n  group sizes (player-lineup rows): "
          + ", ".join(f"{g}={counts[g]}" for g in GROUPS))
    print(f"\n  {'phase':<8}{'group':<24}{'rows':>7}{'legacy':>12}"
          f"{'b10':>12}{'b13':>12}{'b13 eff':>12}{'b13/b10':>10}")
    for phase in list(PHASE_W) + ["blend"]:
        for g in GROUPS:
            rows = acc[phase][g]["legacy"]
            if not rows:
                continue
            m = {a: float(np.mean(acc[phase][g][a]))
                 for a in ("legacy", "b10", "b13", "b13eff")}
            print(f"  {phase:<8}{g:<24}{len(rows):>7}{m['legacy']*100:>11.3f}%"
                  f"{m['b10']*100:>11.3f}%{m['b13']*100:>11.3f}%"
                  f"{m['b13eff']*100:>11.3f}%"
                  f"{(m['b13']/m['b10'] if m['b10'] else float('nan')):>10.3f}")

    bl = lambda g, a: float(np.mean(acc["blend"][g][a]))  # noqa: E731
    vet13, vet10, vetl = bl("veteran_never_bowler", "b13"), \
        bl("veteran_never_bowler", "b10"), bl("veteran_never_bowler", "legacy")
    deb13, deb10 = bl("true_debutant", "b13"), bl("true_debutant", "b10")

    print("\n== PASS conditions ==")
    check("1. cohort (b) veteran never-bowlers: b13 blend share < 0.10%",
          vet13 * 100 < 0.10,
          f"legacy {vetl*100:.3f}% / b10 {vet10*100:.3f}% -> "
          f"b13 {vet13*100:.3f}%")
    check("2. cohort (a) true debutants: b13 within +-0.5pp of b10",
          abs(deb13 - deb10) * 100 <= 0.5,
          f"b10 {deb10*100:.3f}% vs b13 {deb13*100:.3f}% "
          f"(delta {abs(deb13-deb10)*100:.4f}pp)")
    check("3. usage-PRESENT players' weights float-equal b10 vs b13",
          max_known_delta == 0.0, f"max |delta| = {max_known_delta:.3e}")
    check("4a. production json: damped branch never taken "
          f"(b13_damped_rows on the b10 arm)",
          sel10.b13_damped_rows == 0, f"{sel10.b13_damped_rows} rows")
    check("4b. production-json weights float-exact vs an INDEPENDENT "
          "pre-B13 B10 re-derivation",
          max_ref_delta == 0.0, f"max |delta| = {max_ref_delta:.3e}")
    print(f"  [INFO] b13 arm damped rows during the table pass: "
          f"{sel13.b13_damped_rows}")

    # --------------------------------------------- 4c: same-seed draw parity
    print("\n== Part 4c: live select_bowler parity on the production json ==")
    rng = random.Random(7)
    n_cfg, seq_mismatch = 0, 0
    for (mid, date, year, team, players, state) in lineups[:40]:
        for balls in (0, 12, 30, 42, 66, 90, 102, 114):
            phase = phase_of(balls)
            n_all = len(players)
            drop = rng.randrange(n_all)
            available = [i for i in range(n_all) if i != drop]
            fs = _FakeState(state, type("L", (), {
                "players": players, "team_name": team})(), balls)
            as_of10 = sel10._as_of(year)
            alpha10 = sel10.k * sel10._league_share(year)[phase]
            w = sel10._b10_share_weights(players, available, as_of10, phase,
                                         alpha10, date)
            wf = sel10._b10_share_weights(players, list(range(n_all)), as_of10,
                                          phase, alpha10, date)
            rl, _ = relaxed_for(sel10, wf, cfg)
            if rl:
                w = ref_legacy_weights(prod_payload, players, available, phase,
                                       year, k=sel10.k)
            seed = rng.randrange(1 << 30)
            random.seed(seed)
            live = sel10.select_bowler(fs, available)
            random.seed(seed)
            head = head_select_bowler(w, available)
            if live != head:
                seq_mismatch += 1
            n_cfg += 1
    check(f"4c. live select_bowler == recomputed B10 sampling on {n_cfg} "
          "same-seed draws (production json)",
          seq_mismatch == 0, f"mismatches = {seq_mismatch}")
    check("4d. production arm still shows 0 damped rows after live draws",
          sel10.b13_damped_rows == 0, f"{sel10.b13_damped_rows} rows")

    # --------------------------------------------- 5: relaxation (context)
    print("\n== Part 5 (CONTEXT): relaxation triggers over the battery ==")
    print(f"  (lineup, phase) cells evaluated: {n_lineup_phase}")
    print(f"  relaxation-triggering cells  b10: {relax_b10}  "
          f"b13: {relax_b13}   (delta {relax_b13 - relax_b10:+d})")
    if relax_b13 > 5 * max(relax_b10, 1) and relax_b13 > 40:
        print("  *** LOUD WARNING: b13 relaxation triggers EXPLODED vs b10 — "
              "damping pushes never-bowlers under min_share and shrinks the "
              "eligible count. Report this prominently. ***")

    # -------------------------------------------------------------- md5
    print("\n== Part 6: production prior untouched ==")
    md5_after = md5(PROD_USAGE)
    check("models/bowler_phase_usage.json md5 unchanged",
          md5_after == md5_before, f"{md5_before} == {md5_after}")

    print()
    if FAILURES:
        print(f"B13 UNIT CHECK FAILED: {len(FAILURES)} failure(s): {FAILURES}")
        sys.exit(1)
    print("B13 UNIT CHECK PASSED: all assertions hold.")


if __name__ == "__main__":
    main()
