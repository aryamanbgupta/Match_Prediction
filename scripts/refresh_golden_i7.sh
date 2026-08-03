#!/usr/bin/env bash
# Golden-refresh for the i7 production line (post 2026-07-31 promotion).
#
# Scores the production match model on the golden set under the i7 identity
# contract: predict_golden on the i7_v2 golden frame -> blend (w=0.0, pure
# direct) into the cricsheet-stamped golden envelope -> reslice vs the golden
# odds at all three volume slices with I3 tournament blocks.
#
# The stamped envelope is the I18 reference artifact. If it is missing
# (models/auto is gitignored), rebuild it first:
#   uv run python scripts/synthesize_golden_envelope.py \
#       --out models/auto/i18/golden_envelope.json     # see I18.md
#   uv run python scripts/auto/i18_stamp_envelope.py \
#       --envelope models/auto/i18/golden_envelope.json \
#       --test-dir data/golden/polymarket_test \
#       --out models/auto/i18/golden_envelope_cricsheet.json
#
# Usage (from repo root):
#   bash scripts/refresh_golden_i7.sh [MODEL_DIR] [GOLDEN_PARQUET]
set -euo pipefail

MODEL_DIR="${1:-models/xgb_match_i7_swap_production}"
FRAME="${2:-data/xgb_match_data_i7_v2/golden_test.parquet}"
ENVELOPE="models/auto/i18/golden_envelope_cricsheet.json"
ODDS="data/golden/betting_odds_golden.json"
OUT="eval_out/golden_i7_refresh"

for f in "$ENVELOPE" "$ODDS" "$FRAME"; do
  [[ -e "$f" ]] || { echo "refresh_golden_i7: missing $f (see header)"; exit 1; }
done
mkdir -p "$OUT"

echo "=== 1/3 predict golden (${MODEL_DIR}) ==="
uv run python scripts/predict_golden.py \
  --model-dir "$MODEL_DIR" \
  --parquet "$FRAME" \
  --out-json "$MODEL_DIR/golden_predictions.json"

echo "=== 2/3 blend w=0.0 (pure direct) ==="
uv run python scripts/sim_eval/blend_eval_json.py \
  --sim-json "$ENVELOPE" \
  --direct-json "$MODEL_DIR/golden_predictions.json" \
  --w 0.0 --out-dir "$OUT"

echo "=== 3/3 reslice (all / >=50k / >=100k, I3 blocks) ==="
uv run python scripts/sim_eval/reslice_eval_json.py \
  --in "$OUT/golden_envelope_cricsheet_w0p00.json" \
  --odds "$ODDS" \
  --cluster-source-dir data/golden/t20s_json \
  --out-dir "$OUT/sliced"

echo "=== summary ==="
grep -h -E '"min_volume"|"n_matches"|"avg_log_loss"|"market_log_loss"' \
  "$OUT"/sliced/*.json 2>/dev/null | head -30 || true
echo "sliced outputs -> $OUT/sliced/"
