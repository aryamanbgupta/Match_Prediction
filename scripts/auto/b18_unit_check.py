"""B18 unit check — opt-in extras graft: default-path inertness + contract.

Part 1 (INERTNESS, the one that matters for a committed engine edit):
  the pre-B18 `scripts/sim_v1_2.py` is materialised from git (ea4acdb, the
  D16 engine that produced the canonical baseline) into a temp module and
  run SIDE BY SIDE with the current engine.
    1a. `XGBoostModelV2.predict_next_ball` is float-EXACT (bit equality on
        every key) on a grid of probability vectors, sidecar absent.
    1b. N same-seed `T20Rules.simulate_ball` draws produce the IDENTICAL
        (outcome, runs) sequence AND leave the `random` module in the
        IDENTICAL state -> zero extra RNG draws consumed (B13 precedent).

Part 2: `scripts/auto/d15_unit_check.py` (30 assertions: D2 extras
  semantics + D14 attribution + run-out channel) must still pass.

Part 3 (CONTRACT, sidecar present): by default a SYNTHETIC fixture sidecar
  — NOT the B18 fit artifact — is written to a temp dir next to a stub model
  so the auto-detect, composition and event-run draw are exercised end to
  end. Pass `--sidecar models/auto/b18/extras_graft_v1.json` to run the same
  contract against the REAL fitted artifact (plan unit checks 2 and 3):
    3a. extras mass exact (p_wide / p_no_ball set, not renormalised) and
        the 6-class RELATIVE marginals preserved exactly (D3's contract).
    3b. ~300k live `T20Rules.simulate_ball` draws: wide / no-ball
        frequencies within 3 sigma of the fixture rates, and the sampled
        per-event run means within 3 sigma of the fixture means.
    3c. the event-run draw is reproducible under a fixed seed.

Run: uv run python scripts/auto/b18_unit_check.py
"""
from __future__ import annotations

import importlib.util
import json
import math
import random
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from sim_v1_2 import (  # noqa: E402
    EXTRAS_GRAFT_SIDECAR,
    ExtrasGraftConfig,
    MatchState,
    Outcome,
    Player,
    RandomBowlerSelector,
    T20Rules,
    TeamLineup,
    XGBoostModelV2,
    graft_extras,
)

PRE_B18_REV = "ea4acdb"          # D16 engine == pre-B18 HEAD (verified empty diff)
SIX = ('dot', 'one', 'two', 'four', 'six', 'wicket')
FAILURES: list = []

# Synthetic FIXTURE rates/laws — deliberately NOT the B18 fit values, so a
# passing unit check can never be mistaken for a fitted artifact.
FIXTURE = {
    "version": "extras_graft_v1",
    "p_wide": 0.0400,
    "p_no_ball": 0.0060,
    "wide_runs": {"support": [1, 2, 5], "probs": [0.90, 0.06, 0.04],
                  "mean": 1.22},
    "no_ball_runs": {"support": [1, 2], "probs": [0.85, 0.15], "mean": 1.15},
}


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def load_pre_b18_module(tmpdir: Path):
    """Import the pre-B18 sim engine from git as a separate module."""
    src = subprocess.run(
        ["git", "show", f"{PRE_B18_REV}:scripts/sim_v1_2.py"],
        cwd=REPO, capture_output=True, text=True, check=True).stdout
    path = tmpdir / "sim_v1_2_pre_b18.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("sim_v1_2_pre_b18", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sim_v1_2_pre_b18"] = mod
    spec.loader.exec_module(mod)
    return mod


class _StubBooster:
    def __init__(self, probs6):
        self._p = np.asarray(probs6, dtype=float)

    def predict_proba(self, X):
        return np.tile(self._p, (len(X), 1))


def make_stub(base_cls, probs6, extras_graft=None):
    """Real predict_next_ball code path, stub booster, no calibrator."""

    class _Stub(base_cls):
        def __init__(self):
            self.model = _StubBooster(probs6)
            self.ball_calibrator = None
            self.delivery_semantics = None
            self.extras_graft = extras_graft
            self.class_to_outcome = {0: 'dot', 1: 'one', 2: 'two',
                                     3: 'four', 4: 'six', 5: 'wicket'}

        def extract_features(self, state):
            return np.zeros(4)

    return _Stub()


def fresh_state(mod):
    def lineup(team):
        return mod.TeamLineup(team, [mod.Player(f"{team}_p{i}",
                                                f"{team}_p{i}", team)
                                     for i in range(11)])
    return mod.MatchState(team1_lineup=lineup("A"), team2_lineup=lineup("B"),
                          batting_first="A", venue="Test Ground",
                          match_date=datetime(2026, 1, 1))


def ratios_preserved(pre, post, tol=1e-12):
    for i, a in enumerate(SIX):
        for b in SIX[i + 1:]:
            if pre[a] > 0 and pre[b] > 0:
                if abs(post[a] / post[b] - pre[a] / pre[b]) > tol:
                    return False
    return True


GRID = [
    [0.35, 0.38, 0.07, 0.11, 0.045, 0.045],
    [0.50, 0.25, 0.05, 0.12, 0.05, 0.03],
    [0.28, 0.42, 0.09, 0.10, 0.06, 0.05],
    [0.60, 0.20, 0.03, 0.09, 0.04, 0.04],
    [0.10, 0.10, 0.10, 0.20, 0.30, 0.20],
    [0.99, 0.002, 0.002, 0.002, 0.002, 0.002],
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
]


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar", default=None,
                    help="run Part 3 against a REAL fitted sidecar instead of "
                         "the synthetic fixture (plan unit checks 2 and 3)")
    args = ap.parse_args(argv)

    tmpdir = Path(tempfile.mkdtemp(prefix="b18_unit_"))
    try:
        import sim_v1_2 as cur
        pre = load_pre_b18_module(tmpdir)
        print(f"== Part 1: default-path inertness vs pre-B18 engine "
              f"({PRE_B18_REV}) ==")
        print(f"  pre-B18 module: {tmpdir / 'sim_v1_2_pre_b18.py'}")

        # ---- 1a: predict_next_ball float-exact on a grid, sidecar absent
        exact = True
        worst = 0.0
        for p6 in GRID:
            a = make_stub(pre.XGBoostModelV2, p6).predict_next_ball(
                np.zeros(4))
            b = make_stub(cur.XGBoostModelV2, p6).predict_next_ball(
                np.zeros(4))
            if set(a) != set(b):
                exact = False
                break
            for k in a:
                if a[k] != b[k]:          # bit equality, not np.isclose
                    exact = False
                    worst = max(worst, abs(a[k] - b[k]))
        check(f"1a. predict_next_ball float-EXACT on {len(GRID)} prob vectors "
              "(no sidecar)", exact,
              "bit-identical on every key" if exact
              else f"max |delta| {worst:.3e}")

        # ---- 1b: same-seed draw sequence + RNG state identical
        n_draws = 400
        seqs = {}
        states = {}
        for tag, mod in (("pre", pre), ("cur", cur)):
            random.seed(1234)
            rules = mod.T20Rules(bowler_selector=mod.RandomBowlerSelector())
            stub = make_stub(mod.XGBoostModelV2,
                             [0.35, 0.38, 0.07, 0.11, 0.045, 0.045])
            st = fresh_state(mod)
            seq = []
            for _ in range(n_draws):
                if st.is_innings_over() or st.balls >= 119:
                    st = fresh_state(mod)
                outcome, runs = rules.simulate_ball(st, stub)
                seq.append((outcome.name, runs))
            seqs[tag] = seq
            states[tag] = random.getstate()
        same_seq = seqs["pre"] == seqs["cur"]
        n_extras = sum(1 for o, _ in seqs["cur"]
                       if o in ("WIDE", "NO_BALL"))
        check(f"1b. {n_draws} same-seed simulate_ball draws identical "
              "(outcome, runs)", same_seq,
              f"{n_extras} wide/no-ball events in the sequence")
        check("1b. RNG state after the draws identical "
              "(ZERO extra draws consumed)", states["pre"] == states["cur"])

        # ---- Part 2: d15 unit check unchanged
        print("\n== Part 2: d15_unit_check.py unchanged "
              "(D2+D14+run-out, 30 assertions) ==")
        r = subprocess.run(
            [sys.executable, str(REPO / "scripts/auto/d15_unit_check.py")],
            capture_output=True, text=True)
        tail = (r.stdout.strip().splitlines() or ["(none)"])[-1]
        check("2. d15_unit_check.py passes", r.returncode == 0, tail)

        # ---- Part 3: sidecar contract
        if args.sidecar:
            print("\n== Part 3: sidecar contract (REAL fitted B18 artifact) ==")
            sidecar_path = Path(args.sidecar)
            spec = json.loads(sidecar_path.read_text())
            print(f"  sidecar: {sidecar_path}")
        else:
            print("\n== Part 3: sidecar contract (SYNTHETIC fixture, NOT the "
                  "B18 fit) ==")
            fixture_dir = tmpdir / "fixture_model"
            fixture_dir.mkdir()
            sidecar_path = fixture_dir / EXTRAS_GRAFT_SIDECAR
            sidecar_path.write_text(json.dumps(FIXTURE))
            spec = FIXTURE
        cfg = ExtrasGraftConfig.from_path(sidecar_path)
        print(f"  {cfg.banner()}")
        p_extras = cfg.p_wide + cfg.p_no_ball

        raw = {'dot': 0.35, 'one': 0.38, 'two': 0.07, 'four': 0.11,
               'six': 0.045, 'wicket': 0.045, 'wide': 0.0, 'no_ball': 0.0}
        out = graft_extras(dict(raw), cfg)
        check("3a. wide mass == p_wide exactly", out['wide'] == cfg.p_wide)
        check("3a. no_ball mass == p_no_ball exactly",
              out['no_ball'] == cfg.p_no_ball)
        check("3a. sums to 1", abs(sum(out.values()) - 1.0) < 1e-12,
              f"total={sum(out.values()):.15f}")
        check("3a. 6-class relative marginals preserved",
              ratios_preserved(raw, out))
        check("3a. 6-class block mass == 1 - p_extras",
              abs(sum(out[k] for k in SIX) - (1.0 - p_extras)) < 1e-12)

        half = {k: v / 2 for k, v in raw.items()}
        out_h = graft_extras(dict(half), cfg)
        check("3a. non-unit block: extras mass still exact + marginals kept",
              out_h['wide'] == cfg.p_wide
              and out_h['no_ball'] == cfg.p_no_ball
              and abs(sum(out_h.values()) - 1.0) < 1e-12
              and ratios_preserved(half, out_h))
        degen = graft_extras({k: 0.0 for k in raw}, cfg)
        check("3a. degenerate block: uniform spread, sums to 1",
              abs(sum(degen.values()) - 1.0) < 1e-12
              and abs(degen['dot'] - (1.0 - p_extras) / 6) < 1e-12)

        stub_g = make_stub(cur.XGBoostModelV2,
                           [0.35, 0.38, 0.07, 0.11, 0.045, 0.045],
                           extras_graft=cfg)
        live = stub_g.predict_next_ball(np.zeros(4))
        check("3a. LIVE XGBoostModelV2 path: extras mass exact",
              live['wide'] == cfg.p_wide and live['no_ball'] == cfg.p_no_ball)
        check("3a. LIVE path: relative marginals preserved",
              ratios_preserved(raw, live))
        check("3a. LIVE path: sums to 1",
              abs(sum(live.values()) - 1.0) < 1e-12)

        # 3b: live draw rates + sampled event-run means
        random.seed(18)
        rules = T20Rules(bowler_selector=RandomBowlerSelector())
        n = 300_000
        counts = Counter()
        wide_runs, nb_runs = [], []
        st = fresh_state(cur)
        for _ in range(n):
            if st.is_innings_over() or st.balls >= 119:
                st = fresh_state(cur)
            outcome, runs = rules.simulate_ball(st, stub_g)
            counts[outcome] += 1
            if outcome == Outcome.WIDE:
                wide_runs.append(runs)
            elif outcome == Outcome.NO_BALL:
                nb_runs.append(runs)
        f_w = counts[Outcome.WIDE] / n
        f_nb = counts[Outcome.NO_BALL] / n
        tol_w = 3 * math.sqrt(cfg.p_wide * (1 - cfg.p_wide) / n)
        tol_nb = 3 * math.sqrt(cfg.p_no_ball * (1 - cfg.p_no_ball) / n)
        check("3b. simulated wide rate == sidecar p_wide (3 sigma)",
              abs(f_w - cfg.p_wide) <= tol_w,
              f"{f_w:.6f} vs {cfg.p_wide:.6f} (tol {tol_w:.6f})")
        check("3b. simulated no-ball rate == sidecar p_no_ball (3 sigma)",
              abs(f_nb - cfg.p_no_ball) <= tol_nb,
              f"{f_nb:.6f} vs {cfg.p_no_ball:.6f} (tol {tol_nb:.6f})")

        for label, obs, law_mean, support, probs in (
                ("wide", wide_runs, cfg.wide_mean, cfg.wide_support,
                 spec["wide_runs"]["probs"]),
                ("no-ball", nb_runs, cfg.no_ball_mean, cfg.no_ball_support,
                 spec["no_ball_runs"]["probs"])):
            m = float(np.mean(obs))
            var = sum(p * (s - law_mean) ** 2
                      for s, p in zip(support, probs))
            tol = 3 * math.sqrt(var / len(obs))
            check(f"3b. sampled {label} event-run mean == sidecar mean "
                  "(3 sigma)", abs(m - law_mean) <= tol,
                  f"{m:.4f} vs {law_mean:.4f} (tol {tol:.4f}, "
                  f"n={len(obs)})")
            check(f"3b. sampled {label} runs stay inside the sidecar support",
                  set(int(v) for v in obs) <= set(support),
                  f"observed {sorted(set(int(v) for v in obs))}")

        # 3c: reproducibility under a fixed seed
        def draw_seq(seed):
            random.seed(seed)
            r2 = T20Rules(bowler_selector=RandomBowlerSelector())
            s2 = fresh_state(cur)
            out = []
            for _ in range(2000):
                if s2.is_innings_over() or s2.balls >= 119:
                    s2 = fresh_state(cur)
                o, rr = r2.simulate_ball(s2, stub_g)
                out.append((o.name, rr))
            return out
        check("3c. graft draws reproducible under a fixed seed",
              draw_seq(7) == draw_seq(7))

        print(f"\n{'ALL PASS' if not FAILURES else 'FAILURES: ' + str(FAILURES)}")
        return 0 if not FAILURES else 1
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
