"""Phase 4 equivalence check: assert simulated_prob is bit-identical across
backends for every match.

Usage:
    uv run python scripts/tests/compare_phase4_evals.py \\
        phase4_sqlite_eval/match_evaluation_results_xgboost_*.json \\
        phase4_chunks_eval/match_evaluation_results_xgboost_*.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: compare_phase4_evals.py <sqlite_json> <chunks_json>")
        return 2

    sqlite_path, chunks_path = sys.argv[1], sys.argv[2]
    sqlite = load(sqlite_path)
    chunks = load(chunks_path)

    sqlite_matches = {m['match_id']: m for m in sqlite['matches']}
    chunks_matches = {m['match_id']: m for m in chunks['matches']}

    sqlite_ids = set(sqlite_matches.keys())
    chunks_ids = set(chunks_matches.keys())
    if sqlite_ids != chunks_ids:
        missing_in_chunks = sqlite_ids - chunks_ids
        missing_in_sqlite = chunks_ids - sqlite_ids
        print(f"FAIL: match ID sets differ")
        print(f"  in sqlite only: {sorted(missing_in_chunks)[:10]}")
        print(f"  in chunks only: {sorted(missing_in_sqlite)[:10]}")
        return 1

    n_match = len(sqlite_ids)
    n_ok = 0
    diffs = []
    for mid in sorted(sqlite_ids):
        s = sqlite_matches[mid]
        c = chunks_matches[mid]
        sp_s = s['simulated_prob']
        sp_c = c['simulated_prob']
        if set(sp_s.keys()) != set(sp_c.keys()):
            diffs.append((mid, 'key mismatch', sp_s, sp_c))
            continue
        bit_exact = all(sp_s[k] == sp_c[k] for k in sp_s)
        if bit_exact:
            n_ok += 1
        else:
            deltas = {k: sp_s[k] - sp_c[k] for k in sp_s}
            max_abs = max(abs(v) for v in deltas.values())
            diffs.append((mid, f'max |delta|={max_abs:.3e}', sp_s, sp_c))

    print(f"Total matches: {n_match}")
    print(f"  Bit-identical: {n_ok}")
    print(f"  Divergent:     {len(diffs)}")

    if diffs:
        print(f"\nFirst 5 diverging matches:")
        for mid, msg, s, c in diffs[:5]:
            print(f"  {mid}: {msg}")
            print(f"    sqlite: {s}")
            print(f"    chunks: {c}")
        return 1

    # Also confirm aggregate summary numbers match bit-exactly — cheap sanity
    # check that nothing else diverged silently.
    for key in ('avg_log_loss', 'avg_brier_score', 'avg_edge'):
        s_val = sqlite['summary'].get(key)
        c_val = chunks['summary'].get(key)
        print(f"  summary.{key}: sqlite={s_val}  chunks={c_val}  "
              f"{'MATCH' if s_val == c_val else 'DIFF'}")

    print(f"\nPASS: all {n_match} matches bit-identical on simulated_prob")
    return 0


if __name__ == '__main__':
    sys.exit(main())
