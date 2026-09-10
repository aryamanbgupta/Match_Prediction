#!/bin/sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
run_date=${RUN_DATE:-$(date -u +%F)}
daily_dir=${DAILY_DIR:-"$repo_dir/daily"}
fixture_file="$daily_dir/fixtures/$run_date.jsonl"
lineup_file="$daily_dir/lineups/$run_date.jsonl"
prediction_file="$daily_dir/predictions.jsonl"
settlement_file="$daily_dir/settled.jsonl"
score_dir="$daily_dir/scores"
base_source=${BASE_SOURCE_DIR:-"$repo_dir/data/t20s_json"}
context_root=${CONTEXT_ROOT:-"$daily_dir/context"}
state_parent=${STATE_PARENT:-"$(dirname -- "$base_source")"}
stable_link=${STATE_LINK:-"$state_parent/live_state_i7"}

model_dir=${MODEL_DIR:-$(uv run --no-sync python "$repo_dir/scripts/artifacts.py" path match_model_prod)}
state_dir=${STATE_DIR:-$stable_link}
case "$model_dir" in /*) ;; *) model_dir="$repo_dir/$model_dir" ;; esac
case "$state_dir" in /*) ;; *) state_dir="$repo_dir/$state_dir" ;; esac

if [ -n "${CRICSHEET_FETCH_CMD:-}" ]; then
  uv run --no-sync python "$repo_dir/scripts/daily/refresh_state.py" \
    --base-source "$base_source" --context-root "$context_root" \
    --state-parent "$state_parent" --stable-link "$stable_link" \
    --context-date "$run_date" --fetch-cmd "$CRICSHEET_FETCH_CMD"
else
  uv run --no-sync python "$repo_dir/scripts/daily/refresh_state.py" \
    --base-source "$base_source" --context-root "$context_root" \
    --state-parent "$state_parent" --stable-link "$stable_link" \
    --context-date "$run_date"
fi
if [ -n "${GAMMA_RESPONSE:-}" ]; then
  uv run --no-sync python "$repo_dir/scripts/daily/fetch_fixtures.py" \
    --date "$run_date" --out "$fixture_file" --response "$GAMMA_RESPONSE"
else
  uv run --no-sync python "$repo_dir/scripts/daily/fetch_fixtures.py" \
    --date "$run_date" --out "$fixture_file"
fi
set -- --source-dir "$base_source"
for context_dir in "$context_root"/????-??-??; do
  [ -d "$context_dir" ] && set -- "$@" --source-dir "$context_dir"
done
uv run --no-sync python "$repo_dir/scripts/daily/resolve_lineups.py" \
  --fixtures "$fixture_file" --out "$lineup_file" "$@"
set -- --tracker-source-dir "$base_source"
for context_dir in "$context_root"/????-??-??; do
  [ -d "$context_dir" ] && set -- "$@" --tracker-source-dir "$context_dir"
done
uv run --no-sync python "$repo_dir/scripts/daily/predict_daily.py" \
  --fixtures "$fixture_file" --lineups "$lineup_file" --out "$prediction_file" \
  --run-kind t60 --model-dir "$model_dir" \
  --state-dir "$state_dir" --tracker-snapshot "$state_dir/tracker_snapshot.pkl" \
  "$@"
set -- --source-dir "$base_source"
for context_dir in "$context_root"/????-??-??; do
  [ -d "$context_dir" ] && set -- "$@" --source-dir "$context_dir"
done
uv run --no-sync python "$repo_dir/scripts/daily/settle_daily.py" \
  --fixtures-dir "$daily_dir/fixtures" --out "$settlement_file" "$@"
uv run --no-sync python "$repo_dir/scripts/daily/score_daily.py" \
  --predictions "$prediction_file" --settlements "$settlement_file" \
  --out-dir "$score_dir" "$@"
