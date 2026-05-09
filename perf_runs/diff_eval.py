"""Compare two run_sim_eval.py output JSONs for bit-identical results."""
import json
import sys
import glob

if len(sys.argv) == 3:
    base_path, new_path = sys.argv[1], sys.argv[2]
else:
    base_path = sorted(glob.glob("perf_runs/baseline/*.json"))[-1]
    new_path = sorted(glob.glob("perf_runs/phase1/*.json"))[-1]

print(f"baseline: {base_path}")
print(f"new:      {new_path}")

with open(base_path) as f:
    base = json.load(f)
with open(new_path) as f:
    new = json.load(f)

keys_to_check = [
    "simulated_win_prob",
    "log_loss",
    "brier_score",
    "edge",
    "realized_pnl",
    "simulated_scores",
]

base_matches = {m["match_id"]: m for m in base["matches"]}
new_matches = {m["match_id"]: m for m in new["matches"]}

assert set(base_matches) == set(new_matches), "match-id sets differ!"

mismatches = 0
for mid in base_matches:
    b, n = base_matches[mid], new_matches[mid]
    for k in keys_to_check:
        if k not in b:
            continue
        if b[k] != n[k]:
            mismatches += 1
            if mismatches <= 3:
                print(f"  DIFF {mid} [{k}]:")
                print(f"    baseline: {b[k]}")
                print(f"    new:      {n[k]}")

for k in (
    "avg_log_loss",
    "avg_brier_score",
    "flat_betting_total_pnl",
    "flat_betting_roi_pct",
):
    if base["summary"].get(k) != new["summary"].get(k):
        print(
            f"  SUMMARY DIFF [{k}]: {base['summary'].get(k)} vs {new['summary'].get(k)}"
        )
        mismatches += 1

if mismatches == 0:
    print(f"\n✓ BIT-IDENTICAL across {len(base_matches)} matches × {len(keys_to_check)} fields")
else:
    print(f"\n✗ {mismatches} mismatches found")
sys.exit(0 if mismatches == 0 else 1)
