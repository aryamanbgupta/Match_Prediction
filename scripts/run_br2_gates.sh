#!/usr/bin/env bash
# BR2 gate rerun + engine-change A/B — run on the FULL checkout (needs the
# production ball artifacts and data/polymarket_test_v2; this repo's light
# checkouts lack both).
#
# Scope: quantifies the combined effect of
#   (a) the branch's engine changes (first-over selector, ball-119 rule
#       removal, _league_share as-of fix), and
#   (b) the committed 2026-08-14 review fixes (SIM1 mid-over guard, SIM2
#       toss features, PROP3 super-over exclusion, dismissal-kind and
#       tie-break conventions, D/L voids)
# against the recorded pre-fix prop numbers. Expect movement — the verdict
# is whether the GATES still hold (G1 winner-LL parity, G3 top-batter
# no-regression, G5 bowler coverage), not numeric identity.
#
# Usage (from repo root):
#   bash scripts/run_br2_gates.sh <RECORDED_DETAIL_JSON>
# where RECORDED_DETAIL_JSON is the pre-fix detail of record, e.g.
#   reports/prop_calibration_detail_emp_n261.json
set -euo pipefail

RECORDED="${1:?pass the recorded pre-fix prop detail JSON as arg 1}"
TEST_DIR="data/polymarket_test_v2"
ODDS="betting_odds_polymarket_v2.json"
OUT_DIR="eval_out/br2_gates"
SEED=42

for f in "$RECORDED" "$TEST_DIR" "$ODDS" models/bowler_phase_usage.json; do
  [[ -e "$f" ]] || { echo "run_br2_gates: missing $f"; exit 1; }
done
mkdir -p "$OUT_DIR"

echo "=== 1/4 prop backtest under the current engine (production stack defaults) ==="
uv run python scripts/sim_eval/prop_backtest.py \
  --test-dir "$TEST_DIR" --n-sims 100 --seed "$SEED" \
  --detail-out "$OUT_DIR/prop_detail_new_engine.json" \
  --report-out "$OUT_DIR/prop_report_new_engine.md"

echo "=== 2/4 paired diff vs the recorded pre-fix detail (G3 = top_batter row) ==="
uv run python scripts/sim_eval/compare_selector_eval.py \
  --left "$OUT_DIR/prop_detail_new_engine.json" \
  --right "$RECORDED" \
  --left-label new_engine --right-label recorded \
  --out "$OUT_DIR/engine_ab_comparison.md"

echo "=== 3/4 G1 winner-market LL parity (empirical selector, v2 odds) ==="
uv run python scripts/sim_eval/run_sim_eval.py \
  --test-dir "$TEST_DIR" --odds "$ODDS" \
  --n-sims 100 --bowler-selector empirical \
  --output-dir "$OUT_DIR"

echo "=== 4/4 G5 bowler coverage ==="
uv run python scripts/sim_eval/check_bowler_coverage.py \
  --test-dir "$TEST_DIR" \
  --usage models/bowler_phase_usage.json --threshold 100 \
  | tee "$OUT_DIR/g5_coverage.txt"

echo
echo "Outputs -> $OUT_DIR/"
echo "Read: engine_ab_comparison.md (per-family paired deltas; top_batter = G3),"
echo "      the run_sim_eval summary LL vs the recorded parity number (G1),"
echo "      g5_coverage.txt (G5 >= 90%)."
