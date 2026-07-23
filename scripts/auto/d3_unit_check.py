"""D3 unit check — empirical extras graft, verified on the live path.

Part 1 exercises `graft_extras` (the shared helper all six model-wrapper
predict_next_ball sites now call) and the REAL
`XGBoostModelV2.predict_next_ball` (live v7 path, stub booster, no
calibrator) for the exact D3 contract:

  1. wide == EXTRAS_P_WIDE and no_ball == EXTRAS_P_NO_BALL exactly.
  2. The 6-class RELATIVE marginals are preserved exactly (every pairwise
     ratio unchanged) — the point of the (1 - p_extras) composition.
  3. The dict sums to exactly 1 (fp tolerance), including when the model
     block does not (robustness the old renormalize lacked).
  4. Degenerate all-zero block spreads (1 - p_extras) uniformly.

Part 2 re-runs scripts/auto/d15_unit_check.py (30 assertions: D2 extras
semantics + D14 attribution + run-out channel). predict_next_ball is not
on those code paths, so it must still pass byte-for-byte.

Part 3 pushes ~300k deliveries through the REAL `T20Rules.simulate_ball`
(stub-boosted XGBoostModelV2, seeded RNG, fresh state on innings end /
ball 119 so the last-ball legality clamp never censors the draw): the
empirical WIDE / NO_BALL frequencies must match the module constants
within 3-sigma binomial tolerance — proves the graft is wired into the
live sampling path, not just the helper.

Run: uv run python scripts/auto/d3_unit_check.py
"""
import math
import random
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from sim_v1_2 import (EXTRAS_P_NO_BALL, EXTRAS_P_WIDE, MatchState,  # noqa
                      Outcome, Player, RandomBowlerSelector, T20Rules,
                      TeamLineup, XGBoostModelV2, graft_extras)

FAILURES = []
SIX = ('dot', 'one', 'two', 'four', 'six', 'wicket')


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def fresh_state():
    def lineup(team):
        return TeamLineup(team, [Player(f"{team}_p{i}", f"{team}_p{i}", team)
                                 for i in range(11)])
    return MatchState(team1_lineup=lineup("A"), team2_lineup=lineup("B"),
                      batting_first="A", venue="Test Ground",
                      match_date=datetime(2026, 1, 1))


class _StubBooster:
    def __init__(self, probs6):
        self._p = np.asarray(probs6, dtype=float)

    def predict_proba(self, X):
        return np.tile(self._p, (len(X), 1))


class StubV2(XGBoostModelV2):
    """Real predict_next_ball code path, stub booster, no calibrator."""

    def __init__(self, probs6):
        self.model = _StubBooster(probs6)
        self.ball_calibrator = None
        self.class_to_outcome = {0: 'dot', 1: 'one', 2: 'two',
                                 3: 'four', 4: 'six', 5: 'wicket'}

    def extract_features(self, state):
        return np.zeros(4)


def ratios_preserved(pre, post, tol=1e-12):
    """Every pairwise 6-class ratio unchanged (skip zero-prob classes)."""
    for i, a in enumerate(SIX):
        for b in SIX[i + 1:]:
            if pre[a] > 0 and pre[b] > 0:
                if abs(post[a] / post[b] - pre[a] / pre[b]) > tol:
                    return False
    return True


def main():
    p_extras = EXTRAS_P_WIDE + EXTRAS_P_NO_BALL

    print("== Part 1: graft contract (helper + live XGBoostModelV2 path) ==")
    raw = {'dot': 0.35, 'one': 0.38, 'two': 0.07, 'four': 0.11,
           'six': 0.045, 'wicket': 0.045, 'wide': 0.0, 'no_ball': 0.0}
    pre = dict(raw)
    out = graft_extras(dict(raw))
    check("wide == EXTRAS_P_WIDE exactly", out['wide'] == EXTRAS_P_WIDE)
    check("no_ball == EXTRAS_P_NO_BALL exactly",
          out['no_ball'] == EXTRAS_P_NO_BALL)
    check("sums to 1", abs(sum(out.values()) - 1.0) < 1e-12,
          f"total={sum(out.values()):.15f}")
    check("6-class relative marginals preserved", ratios_preserved(pre, out))
    check("6-class block mass == 1 - p_extras",
          abs(sum(out[k] for k in SIX) - (1.0 - p_extras)) < 1e-12)

    # Block that does NOT sum to 1 (the old renormalize would tilt the
    # extras share here; D3 keeps the extras mass exact).
    half = {k: v / 2 for k, v in raw.items()}
    pre_h = dict(half)
    out_h = graft_extras(dict(half))
    check("non-unit block: extras mass still exact",
          out_h['wide'] == EXTRAS_P_WIDE
          and out_h['no_ball'] == EXTRAS_P_NO_BALL
          and abs(sum(out_h.values()) - 1.0) < 1e-12)
    check("non-unit block: relative marginals preserved",
          ratios_preserved(pre_h, out_h))

    degen = {k: 0.0 for k in raw}
    out_d = graft_extras(dict(degen))
    check("degenerate block: uniform spread, sums to 1",
          abs(sum(out_d.values()) - 1.0) < 1e-12
          and abs(out_d['dot'] - (1.0 - p_extras) / 6) < 1e-12)

    # Live wrapper path (the gated one): stub booster through the REAL
    # XGBoostModelV2.predict_next_ball.
    stub = StubV2([0.35, 0.38, 0.07, 0.11, 0.045, 0.045])
    live = stub.predict_next_ball(np.zeros(4))
    check("live path: extras exact",
          live['wide'] == EXTRAS_P_WIDE and live['no_ball'] == EXTRAS_P_NO_BALL)
    check("live path: relative marginals preserved",
          ratios_preserved(raw, live))
    check("live path: sums to 1", abs(sum(live.values()) - 1.0) < 1e-12)
    print(f"    before (old flat graft): p_wide = p_no_ball = 0.01/1.02 = "
          f"{0.01 / 1.02:.6f}")
    print(f"    after  (D3 empirical):   p_wide = {live['wide']:.6f}, "
          f"p_no_ball = {live['no_ball']:.6f}")

    print("\n== Part 2: D15 unit check unchanged (D2+D14+run-out contracts) ==")
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts/auto/d15_unit_check.py")],
        capture_output=True, text=True)
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "(none)"
    check("d15_unit_check.py passes", r.returncode == 0, tail)

    print("\n== Part 3: live-path draw frequencies (real T20Rules."
          "simulate_ball) ==")
    random.seed(3)
    rules = T20Rules(bowler_selector=RandomBowlerSelector())
    counts = Counter()
    n_draws = 300_000
    state = fresh_state()
    for _ in range(n_draws):
        if state.is_innings_over() or state.balls >= 119:
            state = fresh_state()
        outcome, _ = rules.simulate_ball(state, stub)
        counts[outcome] += 1
    f_wide = counts[Outcome.WIDE] / n_draws
    f_nb = counts[Outcome.NO_BALL] / n_draws
    tol_w = 3 * math.sqrt(EXTRAS_P_WIDE * (1 - EXTRAS_P_WIDE) / n_draws)
    tol_n = 3 * math.sqrt(EXTRAS_P_NO_BALL * (1 - EXTRAS_P_NO_BALL) / n_draws)
    check("simulated wide rate == empirical val rate (3-sigma)",
          abs(f_wide - EXTRAS_P_WIDE) <= tol_w,
          f"{f_wide:.6f} vs {EXTRAS_P_WIDE:.6f} (tol {tol_w:.6f})")
    check("simulated no-ball rate == empirical val rate (3-sigma)",
          abs(f_nb - EXTRAS_P_NO_BALL) <= tol_n,
          f"{f_nb:.6f} vs {EXTRAS_P_NO_BALL:.6f} (tol {tol_n:.6f})")

    print(f"\n{'ALL PASS' if not FAILURES else 'FAILURES: ' + str(FAILURES)}")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    sys.exit(main())
