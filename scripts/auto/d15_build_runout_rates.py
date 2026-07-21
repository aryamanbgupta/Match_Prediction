"""D15 — empirical run-out dismissal rates from cricsheet (as-of, pre-test).

Why: the sim credits EVERY sampled WICKET to the bowler (bowling card in
`sim_v1_2.SimulationEngine._simulate_innings`), but the eval's actuals side
excludes run-outs from bowler wickets (`scripts/sim_eval/prop_backtest.py:326`
counts a bowler wicket only when kind != "run out" — and ONLY that kind is
excluded). Training likewise labels every delivery with a wickets[] entry as
WICKET (`parsing_v2.py:909`), run-outs included, so the sim's TOTAL wicket
rate is right and only the attribution is skewed — exactly what IDEAS.md D4
names as the residual bowler_wkts overshoot mechanism.

Computes over male T20 matches in data/t20s_json with date strictly before
2025-07-01 (start of the iteration test window; same boundary as the
materializer's --freeze-trackers-after — nothing from the eval period leaks
into the constants):

  p_runout          = run-out dismissals / all delivery-recorded dismissals
  nonstriker_share  = fraction of run outs where player_out != the delivery's
                      batter (i.e. the non-striker was dismissed)

PRE-COMMITTED: the sim channel constants are the pre-2025-07-01 values.
The recent-year and full-corpus splits are printed for context only.

Artifact: models/auto/d15/runout_rates.json
"""
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "data" / "t20s_json"
OUT = REPO / "models" / "auto" / "d15"
CUTOFF = "2025-07-01"          # test window start; as-of boundary
RECENT_START = "2024-07-01"    # context split only


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    buckets = {"asof": Counter(), "recent": Counter(), "full": Counter()}
    # per bucket: kind counts + special keys __runout_nonstriker/__runout_striker
    n_matches = {"asof": 0, "recent": 0, "full": 0}
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
        if date < CUTOFF:
            active.append("asof")
            if date >= RECENT_START:
                active.append("recent")
        for b in active:
            n_matches[b] += 1

        for inn in m.get("innings", []):
            for over in inn.get("overs", []):
                for d in over.get("deliveries", []):
                    for w in d.get("wickets", []) or []:
                        kind = (w.get("kind") or "").lower()
                        for b in active:
                            buckets[b][kind] += 1
                        if kind == "run out":
                            who = ("nonstriker"
                                   if w.get("player_out") != d.get("batter")
                                   else "striker")
                            for b in active:
                                buckets[b][f"__runout_{who}"] += 1

    def summarize(b):
        c = buckets[b]
        kinds = {k: v for k, v in c.items() if not k.startswith("__")}
        total = sum(kinds.values())
        runouts = kinds.get("run out", 0)
        ns = c.get("__runout_nonstriker", 0)
        st = c.get("__runout_striker", 0)
        assert ns + st == runouts, (b, ns, st, runouts)
        return {
            "n_matches": n_matches[b],
            "total_dismissals": total,
            "runout_count": runouts,
            "p_runout": runouts / total if total else None,
            "nonstriker_count": ns,
            "striker_count": st,
            "nonstriker_share": ns / runouts if runouts else None,
            "kind_distribution": dict(
                sorted(kinds.items(), key=lambda kv: -kv[1])),
        }

    result = {
        "source": str(SRC),
        "asof_cutoff": CUTOFF,
        "recent_start": RECENT_START,
        "n_skipped_non_male": n_skipped_gender,
        "n_unreadable": n_unreadable,
        "asof": summarize("asof"),
        "recent_context_only": summarize("recent"),
        "full_context_only": summarize("full"),
    }

    out_path = OUT / "runout_rates.json"
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)

    a = result["asof"]
    print(f"as-of (< {CUTOFF}): {a['n_matches']} male matches, "
          f"{a['total_dismissals']} dismissals")
    print(f"  p_runout          = {a['p_runout']:.6f} "
          f"({a['runout_count']} run outs)")
    print(f"  nonstriker_share  = {a['nonstriker_share']:.6f} "
          f"({a['nonstriker_count']} non-striker / {a['striker_count']} striker)")
    print("  kind distribution (top 8):")
    for k, v in list(a["kind_distribution"].items())[:8]:
        print(f"    {k:<24}{v:>8}  {v / a['total_dismissals']:.4%}")
    r, fu = result["recent_context_only"], result["full_context_only"]
    print(f"context recent year : p_runout={r['p_runout']:.6f} "
          f"nonstriker_share={r['nonstriker_share']:.6f} (n={r['n_matches']})")
    print(f"context full corpus : p_runout={fu['p_runout']:.6f} "
          f"nonstriker_share={fu['nonstriker_share']:.6f} (n={fu['n_matches']})")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
