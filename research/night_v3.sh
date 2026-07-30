#!/usr/bin/env bash
# Overnight autonomous research loop — split-model orchestration.
#
# Fable runs the loop (ideation, planning, verdict, logging) and delegates all
# implementation and evaluation to Opus subagents. See RUNNER_PROMPT_V3.md for
# the division of labour.
#
# Usage:
#   caffeinate -i ./research/night_v3.sh          # run inside tmux
#   ./research/night_v3.sh --dry-run              # validate, no side effects
#   ./research/night_v3.sh --smoke                # verify model routing, ~1 min
#   ./research/night_v3.sh --once                 # exactly one iteration
#
# Stop: touch research/STOP (or kill the tmux session)
# Env:  NIGHT_HOURS (default 9), ITER_TIMEOUT seconds (default 7200),
#       RETRY_SLEEP seconds (default 1800), NIGHT_BRANCH (default auto-YYYYMMDD),
#       ORCHESTRATOR_MODEL (default fable), EXECUTOR_MODEL (default opus),
#       NIGHT_USE_ENV_AUTH (set to preserve environment auth)

set -u

DRY_RUN=0
RUN_ONCE=0
SMOKE=0

usage() {
  cat <<'EOF'
Usage: ./research/night_v3.sh [--dry-run] [--smoke] [--once]

  --dry-run  Validate configuration and print the planned invocation without
             changing branches, creating files, writing logs, or calling a model
  --smoke    Launch one throwaway orchestrator turn that spawns one subagent and
             reports which model each is running on, then exit. Touches no repo
             state. Use this to confirm the Fable/Opus split is wired correctly.
  --once     Run exactly one research iteration, then exit
  -h, --help Show this help

Environment:
  NIGHT_HOURS        Total runtime in hours (default: 9)
  ITER_TIMEOUT       Per-iteration timeout in seconds (default: 7200)
  RETRY_SLEEP        Delay after a failed iteration in seconds (default: 1800)
  NIGHT_BRANCH       Branch to use (default: auto-YYYYMMDD)
  ORCHESTRATOR_MODEL Model running the loop itself (default: fable)
  EXECUTOR_MODEL     Model running every subagent (default: opus)
  NIGHT_USE_ENV_AUTH Preserve Claude/Anthropic environment auth when set
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --smoke)   SMOKE=1 ;;
    --once)    RUN_ONCE=1 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "night_v3.sh: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROMPT_FILE="$REPO_ROOT/research/RUNNER_PROMPT_V3.md"
LOG_FILE="$REPO_ROOT/research/night.log"
BRANCH="${NIGHT_BRANCH:-auto-$(date +%Y%m%d)}"
NIGHT_HOURS_VALUE="${NIGHT_HOURS:-9}"
ITER_TIMEOUT_VALUE="${ITER_TIMEOUT:-7200}"
RETRY_SLEEP_VALUE="${RETRY_SLEEP:-1800}"
# Captured before the auth strip below, which would otherwise unset them.
ORCHESTRATOR_MODEL_VALUE="${ORCHESTRATOR_MODEL:-fable}"
EXECUTOR_MODEL_VALUE="${EXECUTOR_MODEL:-opus}"

if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "night_v3.sh: prompt file not found: $PROMPT_FILE" >&2
  exit 1
fi

if ! command -v claude >/dev/null 2>&1; then
  echo "night_v3.sh: required command not found: claude" >&2
  exit 1
fi

case "$NIGHT_HOURS_VALUE:$ITER_TIMEOUT_VALUE:$RETRY_SLEEP_VALUE" in
  *[!0-9:]*|:*|*::*|*:)
    echo "night_v3.sh: NIGHT_HOURS, ITER_TIMEOUT, and RETRY_SLEEP must be non-negative integers" >&2
    exit 1
    ;;
esac

PROMPT="$(cat "$PROMPT_FILE")"

# `--disallowed-tools` and friends are variadic, so the prompt must come after
# a `--` separator or its words are parsed as flag values.
AGENT_COMMAND=(
  claude -p
  --model "$ORCHESTRATOR_MODEL_VALUE"
  --fallback-model "$EXECUTOR_MODEL_VALUE"
  --permission-mode auto
  -- "$PROMPT"
)

if (( DRY_RUN )); then
  echo "night_v3.sh dry run OK"
  echo "  repo:              $REPO_ROOT"
  echo "  branch:            $BRANCH"
  echo "  prompt:            $PROMPT_FILE (${#PROMPT} characters)"
  echo "  log:               $LOG_FILE"
  echo "  orchestrator:      $ORCHESTRATOR_MODEL_VALUE"
  echo "  executor subagents: $EXECUTOR_MODEL_VALUE (CLAUDE_CODE_SUBAGENT_MODEL)"
  echo "  night hours:       $NIGHT_HOURS_VALUE"
  echo "  iter timeout:      $ITER_TIMEOUT_VALUE seconds"
  echo "  retry sleep:       $RETRY_SLEEP_VALUE seconds"
  if (( RUN_ONCE )); then
    echo "  iterations:        one"
  else
    echo "  iterations:        until time limit or STOP"
  fi
  echo "  command:           claude -p --model $ORCHESTRATOR_MODEL_VALUE --fallback-model $EXECUTOR_MODEL_VALUE --permission-mode auto -- <prompt>"
  echo "  side effects:      none"
  exit 0
fi

cd "$REPO_ROOT"

# Authenticate the child `claude -p` with the normal subscription login: shed
# env-var auth (a stale ANTHROPIC_API_KEY / _BASE_URL in a shell profile
# silently overrides login and 401s) and any parent Claude session env if
# launched from inside one. Set NIGHT_USE_ENV_AUTH=1 to keep env auth.
if [[ -z "${NIGHT_USE_ENV_AUTH:-}" ]]; then
  while IFS= read -r variable_name; do
    unset "$variable_name"
  done < <(
    env | cut -d= -f1 | grep -E '^(CLAUDE|ANTHROPIC|AI_AGENT|BAGGAGE)'
  )
fi

# MUST come after the strip above — CLAUDE_CODE_SUBAGENT_MODEL matches its
# pattern and would be unset. This env var is what routes every subagent to
# the executor model; it outranks agent frontmatter and per-call overrides.
export CLAUDE_CODE_SUBAGENT_MODEL="$EXECUTOR_MODEL_VALUE"

if (( SMOKE )); then
  echo "night_v3.sh smoke test: orchestrator=$ORCHESTRATOR_MODEL_VALUE subagent=$EXECUTOR_MODEL_VALUE"
  perl -e 'alarm shift; exec @ARGV' 300 \
    claude -p --model "$ORCHESTRATOR_MODEL_VALUE" --permission-mode auto -- \
    "State the exact model ID you are running on. Then use the Agent tool (subagent_type general-purpose, run_in_background false) to ask one subagent to state the exact model ID it is running on. Print exactly two lines: 'orchestrator: <id>' and 'executor: <id>'. Do not read, write, or modify any repository files." \
    < /dev/null
  exit $?
fi

# A dirty tree would carry uncommitted interactive work onto the auto
# branch, mix it into Auto[<id>] commits, and expose it to a FAILED-verdict
# revert. Refuse to start until the tree is clean (untracked files are fine).
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "night_v3.sh: working tree has uncommitted changes — commit or stash" \
       "interactive work before launching the loop." >&2
  git status --short | head -20 >&2
  exit 1
fi

git checkout "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH" || exit 1
mkdir -p research/reports/auto research/handoff

END=$(( $(date +%s) + NIGHT_HOURS_VALUE * 3600 ))
i=0
LAST_RC=0
while (( $(date +%s) < END )) && [[ ! -f research/STOP ]]; do
  i=$((i + 1))
  echo "=== v3 orch=$ORCHESTRATOR_MODEL_VALUE exec=$EXECUTOR_MODEL_VALUE iter=$i start $(date '+%F %T') ===" >> "$LOG_FILE"

  # perl alarm is a portable timeout for macOS, which lacks GNU timeout.
  perl -e 'alarm shift; exec @ARGV' "$ITER_TIMEOUT_VALUE" \
    "${AGENT_COMMAND[@]}" < /dev/null >> "$LOG_FILE" 2>&1
  rc=$?
  LAST_RC=$rc

  echo "=== v3 iter=$i exit=$rc $(date '+%F %T') ===" >> "$LOG_FILE"
  (( RUN_ONCE )) && break
  # A non-zero status may indicate a usage limit, classifier abort, or timeout.
  # Retry until NIGHT_HOURS expires, sleeping through part of the usage window.
  (( rc != 0 )) && sleep "$RETRY_SLEEP_VALUE"
done

echo "=== night_v3.sh done $(date '+%F %T') ===" >> "$LOG_FILE"

if (( RUN_ONCE )); then
  exit "$LAST_RC"
fi
