"""D3 — empirical per-delivery wide / no-ball rates from cricsheet.

Why: every sim model wrapper grafts a flat 1% wide + 1% no-ball onto the
6-class model output and renormalizes (`sim_v1_2.py` predict_next_ball
sites). Real T20 wides run ~3.5-4% per delivery and no-balls ~0.5%, so the
flat graft under-produces wides ~4x and over-produces no-balls ~2x. D3
replaces the hardcoded rates with empirical ones and composes the extras
mass so the calibrated 6-class RELATIVE marginals are preserved exactly
(scale the model block by (1 - p_extras) instead of renormalizing after
the graft). Labels still fold extras into the 6 classes (a wide-1 is
labeled `one`) — that label-side rework is I5; this is the sim-side half.

Rate definition (matches the sim's sampling semantics — predict_next_ball
is drawn once per DELIVERY, and wides/no-balls are re-bowled):

  p_wide    = deliveries with extras.wides   / all deliveries
  p_no_ball = deliveries with extras.noballs / all deliveries

PRE-COMMITTED: the sim constants are the VAL-split values
(2024-12-31 <= date < 2025-06-30 — the ball model's validation window per
`loaders_common.DEFAULT_SPLITS`, same convention as the E5 calibrator's
val fit; nothing from the eval period leaks in). The as-of (< 2025-07-01),
recent-year and full-corpus splits are printed for context only.

Artifact: models/auto/d3/extras_rates.json
Run: uv run python scripts/auto/d3_build_extras_rates.py
"""
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "data" / "t20s_json"
OUT = REPO / "models" / "auto" / "d3"

TRAIN_END = "2024-12-31"       # val window start (classify_split: < train_end -> train)
VAL_END = "2025-06-30"         # val window end   (< val_end -> val)
ASOF_CUTOFF = "2025-07-01"     # iteration test window start (D15 convention)
RECENT_START = "2024-07-01"    # context split only


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    buckets = {"val": Counter(), "asof": Counter(),
               "recent": Counter(), "full": Counter()}
    n_matches = Counter()
    n_skipped_gender = 0
    n_unreadable = 0

    for f in sorted(SRC.glob("*.json")):
        try:
            with open(f) as fh:
                m = json.load(fh)
        except Exception:
            n_unreadable += 1
            continue
        info = m.get("info", {})
        if info.get("gender") != "male":
            n_skipped_gender += 1
            continue
        dates = info.get("dates") or []
        if not dates:
            continue
        date = str(dates[0])

        active = ["full"]
        if date < ASOF_CUTOFF:
            active.append("asof")
            if date >= RECENT_START:
                active.append("recent")
        if TRAIN_END <= date < VAL_END:
            active.append("val")
        for b in active:
            n_matches[b] += 1

        for inn in m.get("innings", []):
            for over in inn.get("overs", []):
                for d in over.get("deliveries", []):
                    extras = d.get("extras") or {}
                    for b in active:
                        buckets[b]["deliveries"] += 1
                        if "wides" in extras:
                            buckets[b]["wides"] += 1
                        if "noballs" in extras:
                            buckets[b]["noballs"] += 1

    out = {"skipped_gender": n_skipped_gender, "unreadable": n_unreadable,
           "buckets": {}}
    for b in ("val", "asof", "recent", "full"):
        c = buckets[b]
        n = c["deliveries"]
        out["buckets"][b] = {
            "n_matches": n_matches[b],
            "n_deliveries": n,
            "n_wides": c["wides"],
            "n_noballs": c["noballs"],
            "p_wide": c["wides"] / n if n else None,
            "p_no_ball": c["noballs"] / n if n else None,
        }
        r = out["buckets"][b]
        print(f"{b:>7}: {r['n_matches']:>6} matches  {n:>10,} deliveries  "
              f"p_wide={r['p_wide']:.6f}  p_no_ball={r['p_no_ball']:.6f}")

    v = out["buckets"]["val"]
    out["sim_constants"] = {"P_WIDE": round(v["p_wide"], 6),
                           "P_NO_BALL": round(v["p_no_ball"], 6),
                           "source": "val split (pre-committed)"}
    print(f"\nPRE-COMMITTED sim constants (val split): "
          f"P_WIDE={out['sim_constants']['P_WIDE']}  "
          f"P_NO_BALL={out['sim_constants']['P_NO_BALL']}")
    print(f"old flat graft (before): p_wide = p_no_ball = 0.01/1.02 = "
          f"{0.01/1.02:.6f} (exact — flat mass then renormalize)")

    with open(OUT / "extras_rates.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {OUT / 'extras_rates.json'}")


if __name__ == "__main__":
    main()
