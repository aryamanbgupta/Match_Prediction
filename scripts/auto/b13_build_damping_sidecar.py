"""B13 — build the opt-in never-bowler damping sidecar (B10 defect-(b) fix).

B10/B12 (LANDED, shipped) aligned the usage-ABSENT branch of
`EmpiricalBowlerSelector` to B9's as-of expected-balls share. That fixed
defect (a) (true debutants now bowl ~9%) but mechanically WORSENED defect
(b): a veteran never-bowler (n>=20 XI appearances, 0 corpus balls) still gets
`exp_balls = k_u * prior / (k_u + n)` ~ 1-2 balls at k_u=5, ABOVE the legacy
alpha floor, so that cohort's share ROSE 0.270% -> 0.496% (actual ~0).

This script fits a zero-usage-aware damping from `models/b10_usage_corpus.pkl`
and writes the opt-in sidecar:

  P(bowls at all | n prior appearances, 0 prior balls) = k_damp / (k_damp + n)
  damped expected balls                                = P(bowls) * mu_active

with `mu_active` = pooled mean deliveries on the appearances where such a
player DID finally bowl. As n grows the damped expectation -> 0, which is the
cohort's true rate; n = 0 (true debutants) is NOT an event and its sim-side
path is untouched.

Event set (fit data), mirroring `_B10AsOfExpBalls`'s row ordering:
  for each player, for each row i (0-indexed) in its date-sorted appearance
  log with  n = i >= 1  AND  prior sum of balls == 0  AND  date < FIT_CUTOFF:
      event = (n, bowled = balls_i > 0, balls_i)

`k_damp` is the MLE of P(bowls | n) = k/(k+n) on a deterministic 1-D grid.

Artifacts (gitignored, under `models/auto/b13/`):
  bowler_phase_usage_b13.json   deep copy of `models/bowler_phase_usage.json`
                                with ONE addition nested INSIDE the existing
                                `b10_asof_usage` object:
                                  "b13_never_bowler_damping": {k_damp, mu_active}
  damping_fit.json              the fit + all printed diagnostics

The production prior `models/bowler_phase_usage.json` is READ ONLY — its md5
is asserted unchanged before and after. `corpus_path` is kept as-is
(`models/b10_usage_corpus.pkl`) — no copy needed.

Run: uv run python scripts/auto/b13_build_damping_sidecar.py
"""
from __future__ import annotations

import copy
import hashlib
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

PROD_USAGE = REPO / "models" / "bowler_phase_usage.json"
PROD_USAGE_MD5 = "2e650423f0c949631fca1f15dd1c8a56"
CORPUS = REPO / "models" / "b10_usage_corpus.pkl"
OUT_DIR = REPO / "models" / "auto" / "b13"
OUT_USAGE = OUT_DIR / "bowler_phase_usage_b13.json"
OUT_FIT = OUT_DIR / "damping_fit.json"

FIT_CUTOFF = "2025-07-01"     # training-window discipline (D15 run-out rates)
K_GRID = (0.01, 1000.0, 2001)  # geomspace(lo, hi, n) — deterministic MLE grid

# Diagnostic bins over n (prior appearances), inclusive ranges.
N_BINS = [(1, 1, "1"), (2, 2, "2"), (3, 5, "3-5"), (6, 10, "6-10"),
          (11, 20, "11-20"), (21, 50, "21-50"), (51, 10 ** 9, "51+")]
EXAMPLE_NS = [1, 5, 20, 100, 285]


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_events(corpus: dict, cutoff: str):
    """[(n, bowled, balls)] over never-yet-bowled appearances before cutoff."""
    ns, bowled, balls = [], [], []
    for name, rows in corpus["player"].items():
        rows = sorted(rows)          # same ordering as _B10AsOfExpBalls
        prior_balls = 0
        for i, r in enumerate(rows):
            date, b = r[0], r[1]
            if i >= 1 and prior_balls == 0 and date < cutoff:
                ns.append(i)
                bowled.append(b > 0)
                balls.append(b)
            prior_balls += b
    return (np.asarray(ns, dtype=np.int64),
            np.asarray(bowled, dtype=bool),
            np.asarray(balls, dtype=np.float64))


def fit_k_damp(ns: np.ndarray, bowled: np.ndarray):
    """MLE of k in P(bowls|n) = k/(k+n) on a fixed geomspace grid."""
    # Aggregate to (n, bowled) counts — exact and fast.
    cnt = Counter(zip(ns.tolist(), bowled.tolist()))
    n_arr = np.asarray([k[0] for k in cnt], dtype=np.float64)
    is_b = np.asarray([k[1] for k in cnt], dtype=bool)
    w = np.asarray([cnt[k] for k in cnt], dtype=np.float64)

    grid = np.geomspace(*K_GRID)
    lls = np.empty_like(grid)
    for j, k in enumerate(grid):
        p = k / (k + n_arr)
        ll = np.where(is_b, np.log(p), np.log1p(-p))
        lls[j] = float(np.dot(w, ll))
    j = int(np.argmax(lls))
    return float(grid[j]), float(lls[j]), grid, lls


def main() -> None:
    assert PROD_USAGE.exists(), f"missing {PROD_USAGE}"
    assert CORPUS.exists(), f"missing {CORPUS}"

    md5_before = md5(PROD_USAGE)
    print(f"models/bowler_phase_usage.json md5 BEFORE: {md5_before}")
    assert md5_before == PROD_USAGE_MD5, (
        f"production prior md5 drifted: {md5_before} != {PROD_USAGE_MD5}")

    with open(CORPUS, "rb") as f:
        corpus = pickle.load(f)
    n_players = len(corpus["player"])
    n_rows = sum(len(v) for v in corpus["player"].values())
    print(f"corpus: {CORPUS} (md5 {md5(CORPUS)}) — {n_players} players, "
          f"{n_rows} appearance rows")

    ns, bowled, balls = build_events(corpus, FIT_CUTOFF)
    n_events = int(ns.size)
    n_bowled = int(bowled.sum())
    print(f"\nevent set (n>=1, prior balls==0, date < {FIT_CUTOFF}): "
          f"{n_events} events, {n_bowled} bowled "
          f"({n_bowled / n_events * 100:.3f}%)")

    k_damp, ll_max, grid, lls = fit_k_damp(ns, bowled)
    mu_active = float(balls[bowled].mean())
    print(f"k_damp (MLE on geomspace{K_GRID}): {k_damp:.6f}  "
          f"(logL {ll_max:.3f})")
    print(f"mu_active = mean(balls | bowled) = {mu_active:.6f}  "
          f"(n={n_bowled})")
    # Grid-edge sanity
    print(f"  grid neighbours: k={grid[max(0, int(np.argmax(lls)) - 1)]:.6f} "
          f"logL={lls[max(0, int(np.argmax(lls)) - 1)]:.3f} | "
          f"k={grid[min(len(grid) - 1, int(np.argmax(lls)) + 1)]:.6f} "
          f"logL={lls[min(len(grid) - 1, int(np.argmax(lls)) + 1)]:.3f}")

    # ------------------------------------------------------ binned diagnostic
    print(f"\n  {'n bin':<10}{'events':>10}{'bowled':>9}"
          f"{'empirical P':>14}{'fitted P':>12}")
    bin_rows = []
    for lo, hi, lbl in N_BINS:
        m = (ns >= lo) & (ns <= hi)
        ne = int(m.sum())
        if ne == 0:
            continue
        nb = int(bowled[m].sum())
        emp = nb / ne
        fit = float(np.mean(k_damp / (k_damp + ns[m])))
        bin_rows.append({"bin": lbl, "events": ne, "bowled": nb,
                         "empirical_p": emp, "fitted_p": fit})
        print(f"  {lbl:<10}{ne:>10}{nb:>9}{emp * 100:>13.3f}%"
              f"{fit * 100:>11.3f}%")

    # ------------------------------------- damped vs undamped expected balls
    from sim_v1_2 import _B10AsOfExpBalls  # noqa: E402  (after path insert)
    k_u = 5.0
    asof = _B10AsOfExpBalls(corpus, k_u)
    prior_fit = asof.prior_balls(FIT_CUTOFF)
    prior_end = asof.prior_balls("2026-04-17")   # end of the iteration window
    print(f"\n  B9 global prior_balls @ {FIT_CUTOFF}: {prior_fit:.6f}; "
          f"@ 2026-04-17: {prior_end:.6f}  (k_usage={k_u})")
    print(f"\n  {'n':>6}{'damped exp_balls':>20}{'undamped B9':>16}"
          f"{'ratio':>10}{'damped share':>15}{'undamped share':>17}")
    ex_rows = []
    for n in EXAMPLE_NS:
        damped = (k_damp / (k_damp + n)) * mu_active
        undamped = (k_u * prior_end) / (k_u + n)
        ex_rows.append({"n": n, "damped_exp_balls": damped,
                        "undamped_exp_balls": undamped,
                        "ratio": damped / undamped if undamped else float("nan")})
        print(f"  {n:>6}{damped:>20.6f}{undamped:>16.6f}"
              f"{damped / undamped:>10.4f}{damped / 120.0 * 100:>14.4f}%"
              f"{undamped / 120.0 * 100:>16.4f}%")

    # -------------------------------------------------------------- sidecar
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROD_USAGE) as f:
        payload = json.load(f)
    assert "b10_asof_usage" in payload, "production prior is not B10-active"
    out = copy.deepcopy(payload)
    b10 = out["b10_asof_usage"]
    assert "b13_never_bowler_damping" not in b10, "already tagged"
    b10["b13_never_bowler_damping"] = {"k_damp": k_damp,
                                       "mu_active": mu_active}
    with open(OUT_USAGE, "w") as f:
        json.dump(out, f)
    print(f"\nsidecar prior -> {OUT_USAGE} (md5 {md5(OUT_USAGE)})")
    print(f"  b10_asof_usage = {json.dumps(b10)}")

    fit = {
        "fit_cutoff": FIT_CUTOFF,
        "k_grid": list(K_GRID),
        "corpus": str(CORPUS),
        "corpus_md5": md5(CORPUS),
        "n_players": n_players,
        "n_appearance_rows": n_rows,
        "n_events": n_events,
        "n_bowled": n_bowled,
        "k_damp": k_damp,
        "logL": ll_max,
        "mu_active": mu_active,
        "k_usage": k_u,
        "prior_balls_at_cutoff": prior_fit,
        "prior_balls_at_2026_04_17": prior_end,
        "bins": bin_rows,
        "examples": ex_rows,
    }
    with open(OUT_FIT, "w") as f:
        json.dump(fit, f, indent=1)
    print(f"fit numbers -> {OUT_FIT}")

    md5_after = md5(PROD_USAGE)
    print(f"\nmodels/bowler_phase_usage.json md5 AFTER:  {md5_after}")
    assert md5_after == md5_before, "production usage prior was modified!"
    print("OK: production prior unchanged.")


if __name__ == "__main__":
    main()
