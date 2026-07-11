#!/usr/bin/env bash
# Run a 3-slice liquidity eval (all / >=$50k / >=$100k).
#
# Phase 1 of the outcome-dist follow-up plan. Wraps run_sim_eval.py three
# times with the same model/odds/test-dir, varying only --min-volume.
# Output filenames already encode the slice (run_sim_eval.py:slice_tag).
#
# Usage:
#   bash scripts/run_sliced_eval.sh [--model-version v3] [--n-sims 100]
#                                   [--output-dir eval_out/sliced]
#                                   [extra args passed through]
#
# Defaults are tuned for v6 outcome-dist eval against polymarket_test.

set -euo pipefail

TEST_DIR="${TEST_DIR:-data/polymarket_test}"
ODDS_FILE="${ODDS_FILE:-betting_odds_polymarket.json}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_out/sliced/$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$OUTPUT_DIR"

EXTRA_ARGS=("$@")

run_one() {
    local label="$1"
    shift
    echo
    echo "=================================================="
    echo "  Slice: $label"
    echo "=================================================="
    uv run python scripts/sim_eval/run_sim_eval.py \
        --test-dir "$TEST_DIR" \
        --odds "$ODDS_FILE" \
        --output-dir "$OUTPUT_DIR" \
        "$@" \
        "${EXTRA_ARGS[@]}"
}

run_one "all"            # no --min-volume
run_one "min_volume_50000"  --min-volume 50000
run_one "min_volume_100000" --min-volume 100000

echo
echo "All three slices written to: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"
