#!/usr/bin/env python3
"""Aggregate per-bowler phase-usage counts from cricsheet for the
EmpiricalBowlerSelector.

For each delivery in `data/t20s_json/`, bucket `(cricsheet_id, year, phase)`
where phase = PP (over<6) / mid (6≤over<16) / death (over≥16). Also
accumulate per-year league marginal shares so the selector can shrink
unknown bowlers toward a sensible prior.

Output: a single JSON keyed by cricsheet_id with per-year counts. The
selector consumes this as-of-fixture-date by summing all year buckets
strictly before the match year.

I20 (2026-08-01): the payload also carries the `b10_asof_usage` key that
activates the B12-shipped usage-aligned selector branch in
`sim_v1_2.EmpiricalBowlerSelector`. This builder predated B10, so a rerun
used to regenerate the payload WITHOUT the key and silently revert the
shipped selector (the only tell was a missing
`B10 usage-aligned bowler selector ACTIVE` banner in run logs). The key is
now stamped by default and the builder FAILS CLOSED if the corpus sidecar
(`models/b10_usage_corpus.pkl`) is missing; pass `--no-b10-key` to build a
legacy keyless payload deliberately. Corpus rebuild path:
`uv run python scripts/auto/b9_usage_baseline.py --rebuild-corpus`, then
copy `models/auto/b9/usage_corpus.pkl` -> `models/b10_usage_corpus.pkl`.
`k_usage` is imported from `scripts/auto/b9_usage_baseline.K_USAGE`
(B10 convention: never hardcoded).

Usage:
    uv run python scripts/build_bowler_phase_usage.py \
        --source-dir data/t20s_json \
        --out models/bowler_phase_usage.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from loaders_common import iter_matches_chronological  # noqa: E402


PHASES = ("pp", "mid", "death")

# B10 selector-branch config (values shipped by B12; see
# scripts/auto/b10_build_usage_sidecar.py for the original derivation).
B10_MIN_ELIGIBLE = 5
B10_MIN_SHARE = 0.01


def _load_b9_k_usage() -> float:
    """Import K_USAGE from the B9 script so the stamped value can't drift."""
    import importlib.util

    path = Path(__file__).resolve().parent / "auto" / "b9_usage_baseline.py"
    spec = importlib.util.spec_from_file_location("b9_usage_baseline", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["b9_usage_baseline"] = mod
    spec.loader.exec_module(mod)
    return float(mod.K_USAGE)


def _phase_from_over(over_idx: int) -> str:
    if over_idx < 6:
        return "pp"
    if over_idx < 16:
        return "mid"
    return "death"


def aggregate(source_dir: str, gender: str | None) -> dict:
    """Walk every match and accumulate per-(player, year, phase) ball counts."""
    by_player: dict[str, dict[int, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {p: 0 for p in PHASES} | {"total": 0})
    )
    by_year_league: dict[int, dict[str, int]] = defaultdict(
        lambda: {p: 0 for p in PHASES} | {"total": 0}
    )

    n_matches = 0
    n_deliveries = 0
    n_unresolved = 0
    min_date_per_year: dict[int, str] = {}
    max_date_per_year: dict[int, str] = {}

    t0 = time.time()
    for match_id, json_text, match_date in iter_matches_chronological(
        source_dir, gender=gender
    ):
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            continue

        registry = (data.get("info", {}).get("registry", {}) or {}).get(
            "people", {}
        ) or {}
        year = match_date.year
        date_str = match_date.strftime("%Y-%m-%d")
        min_date_per_year.setdefault(year, date_str)
        max_date_per_year[year] = date_str

        for inn in data.get("innings", []):
            for over in inn.get("overs", []):
                over_idx = over.get("over", 0)
                phase = _phase_from_over(over_idx)
                for d in over.get("deliveries", []):
                    bowler_name = d.get("bowler")
                    if not bowler_name:
                        continue
                    cid = registry.get(bowler_name)
                    if not cid:
                        n_unresolved += 1
                        # Fall back to the name string so we still capture
                        # usage; the selector resolves names that miss the
                        # registry the same way.
                        cid = bowler_name

                    by_player[cid][year][phase] += 1
                    by_player[cid][year]["total"] += 1
                    by_year_league[year][phase] += 1
                    by_year_league[year]["total"] += 1
                    n_deliveries += 1

        n_matches += 1
        if n_matches % 1000 == 0:
            print(
                f"  ... {n_matches} matches, {n_deliveries:,} deliveries, "
                f"{time.time()-t0:.1f}s",
                flush=True,
            )

    elapsed = time.time() - t0
    print(
        f"Done in {elapsed:.1f}s: {n_matches} matches, "
        f"{n_deliveries:,} deliveries, {len(by_player):,} unique players, "
        f"{n_unresolved} unresolved-name fallbacks"
    )

    # Year-level league shares.
    by_year_league_shares: dict[int, dict[str, float]] = {}
    global_counts = {p: 0 for p in PHASES} | {"total": 0}
    for year, counts in by_year_league.items():
        total = counts["total"]
        shares = {
            f"{p}_share": (counts[p] / total) if total else 0.0 for p in PHASES
        }
        shares["total_balls"] = total
        by_year_league_shares[year] = shares
        for p in PHASES:
            global_counts[p] += counts[p]
        global_counts["total"] += total

    global_shares = {
        f"{p}_share": (global_counts[p] / global_counts["total"])
        if global_counts["total"]
        else 0.0
        for p in PHASES
    }
    global_shares["total_balls"] = global_counts["total"]

    # Print per-year sanity for the user.
    print("\nPer-year league phase shares (PP / mid / death, total balls):")
    for year in sorted(by_year_league_shares):
        s = by_year_league_shares[year]
        print(
            f"  {year}: {s['pp_share']:.3f} / {s['mid_share']:.3f} / "
            f"{s['death_share']:.3f}   "
            f"(n={s['total_balls']:>7,}, dates "
            f"{min_date_per_year[year]} → {max_date_per_year[year]})"
        )
    print(
        f"\nGlobal: {global_shares['pp_share']:.3f} / "
        f"{global_shares['mid_share']:.3f} / "
        f"{global_shares['death_share']:.3f}   "
        f"(n={global_shares['total_balls']:,})"
    )

    return {
        "schema_version": 1,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_dir": str(source_dir),
        "gender": gender,
        "n_matches": n_matches,
        "n_deliveries": n_deliveries,
        "n_unresolved_names": n_unresolved,
        "by_player": {
            cid: {str(y): dict(counts) for y, counts in years.items()}
            for cid, years in by_player.items()
        },
        "by_year_league": {str(y): s for y, s in by_year_league_shares.items()},
        "global_league": global_shares,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", default="data/t20s_json")
    ap.add_argument("--extra-source-dir", default=None,
                    help="Optional second source (e.g. data/golden/t20s_json). "
                         "Walked separately and merged.")
    ap.add_argument("--gender", default="male",
                    help="Gender filter (default 'male'; pass 'none' to disable).")
    ap.add_argument("--out", default="models/bowler_phase_usage.json")
    ap.add_argument("--b10-corpus", default="models/b10_usage_corpus.pkl",
                    help="As-of usage corpus the b10_asof_usage key points at "
                         "(shipped by B12).")
    ap.add_argument("--no-b10-key", action="store_true",
                    help="Deliberately build a legacy keyless payload "
                         "(pre-B12 behavior; the shipped usage-aligned "
                         "selector branch will NOT activate).")
    args = ap.parse_args()

    gender = None if args.gender == "none" else args.gender

    print(f"Walking {args.source_dir} (gender={gender}) ...")
    payload = aggregate(args.source_dir, gender)

    if args.extra_source_dir:
        print(f"\nWalking extra source {args.extra_source_dir} ...")
        extra = aggregate(args.extra_source_dir, gender)
        # Merge extra into payload.
        for cid, years in extra["by_player"].items():
            for y_str, counts in years.items():
                target = payload["by_player"].setdefault(cid, {}).setdefault(
                    y_str, {p: 0 for p in PHASES} | {"total": 0}
                )
                for k, v in counts.items():
                    target[k] = target.get(k, 0) + v
        # Recompute league shares using merged counts.
        # (Cheap: re-derive from by_player.)
        league_counts: dict[str, dict[str, int]] = {}
        for years in payload["by_player"].values():
            for y_str, counts in years.items():
                lc = league_counts.setdefault(
                    y_str, {p: 0 for p in PHASES} | {"total": 0}
                )
                for p in PHASES:
                    lc[p] += counts.get(p, 0)
                lc["total"] += counts.get("total", 0)
        by_year_league_shares = {}
        global_counts = {p: 0 for p in PHASES} | {"total": 0}
        for y_str, counts in league_counts.items():
            total = counts["total"]
            shares = {
                f"{p}_share": (counts[p] / total) if total else 0.0 for p in PHASES
            }
            shares["total_balls"] = total
            by_year_league_shares[y_str] = shares
            for p in PHASES:
                global_counts[p] += counts[p]
            global_counts["total"] += total
        payload["by_year_league"] = by_year_league_shares
        payload["global_league"] = {
            f"{p}_share": (global_counts[p] / global_counts["total"])
            if global_counts["total"]
            else 0.0
            for p in PHASES
        } | {"total_balls": global_counts["total"]}
        payload["n_matches"] += extra["n_matches"]
        payload["n_deliveries"] += extra["n_deliveries"]
        payload["n_unresolved_names"] += extra["n_unresolved_names"]

    if args.no_b10_key:
        print("\nWARNING: --no-b10-key set — payload will NOT activate the "
              "B12-shipped usage-aligned selector branch.")
    else:
        corpus = Path(args.b10_corpus)
        if not corpus.exists():
            raise SystemExit(
                f"missing {corpus}: refusing to write a payload that would "
                "silently revert the B12-shipped selector (I20). Rebuild the "
                "corpus (scripts/auto/b9_usage_baseline.py --rebuild-corpus, "
                "then copy models/auto/b9/usage_corpus.pkl -> "
                f"{corpus}) or pass --no-b10-key to build a legacy keyless "
                "payload deliberately."
            )
        payload["b10_asof_usage"] = {
            "corpus_path": str(corpus),
            "k_usage": _load_b9_k_usage(),
            "min_eligible": B10_MIN_ELIGIBLE,
            "min_share": B10_MIN_SHARE,
        }
        print(f"\nb10_asof_usage stamped: "
              f"{json.dumps(payload['b10_asof_usage'])}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nWrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
