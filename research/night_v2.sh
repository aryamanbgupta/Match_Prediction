#!/usr/bin/env bash
# Overnight autonomous research loop with selectable agent support.
#
# Usage:
#   caffeinate -i ./research/night_v2.sh              # Claude (default)
#   caffeinate -i ./research/night_v2.sh --codex      # Codex
#   ./research/night_v2.sh --codex --dry-run          # Validate without side effects
#   ./research/night_v2.sh --codex --once             # Run exactly one iteration
#
# Stop: touch research/STOP (or kill the tmux session)
# Env:  NIGHT_HOURS (default 9), ITER_TIMEOUT seconds (default 7200),
#       RETRY_SLEEP seconds (default 1800), NIGHT_BRANCH (default auto-YYYYMMDD)

set -u

AGENT="claude"
DRY_RUN=0
RUN_ONCE=0

usage() {
  cat <<'EOF'
Usage: ./research/night_v2.sh [--claude | --codex] [--dry-run] [--once]

Agent selection:
  --claude   Run Claude (default)
  --codex    Run Codex non-interactively

Other options:
  --dry-run  Validate configuration and print the planned invocation without
             changing branches, creating files, writing logs, or calling an agent
  --once     Run exactly one agent iteration, then exit
  -h, --help Show this help

Environment:
  NIGHT_HOURS        Total runtime in hours (default: 9)
  ITER_TIMEOUT       Per-iteration timeout in seconds (default: 7200)
  RETRY_SLEEP        Delay after a failed iteration in seconds (default: 1800)
  NIGHT_BRANCH       Branch to use (default: auto-YYYYMMDD)
  NIGHT_USE_ENV_AUTH Preserve Claude/Anthropic environment auth when set
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --claude)
      AGENT="claude"
      ;;
    --codex)
      AGENT="codex"
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
    --once)
      RUN_ONCE=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "night_v2.sh: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROMPT_FILE="$REPO_ROOT/research/RUNNER_PROMPT.md"
LOG_FILE="$REPO_ROOT/research/night.log"
BRANCH="${NIGHT_BRANCH:-auto-$(date +%Y%m%d)}"
NIGHT_HOURS_VALUE="${NIGHT_HOURS:-9}"
ITER_TIMEOUT_VALUE="${ITER_TIMEOUT:-7200}"
RETRY_SLEEP_VALUE="${RETRY_SLEEP:-1800}"

if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "night_v2.sh: prompt file not found: $PROMPT_FILE" >&2
  exit 1
fi

if ! command -v "$AGENT" >/dev/null 2>&1; then
  echo "night_v2.sh: required command not found: $AGENT" >&2
  exit 1
fi

case "$NIGHT_HOURS_VALUE:$ITER_TIMEOUT_VALUE:$RETRY_SLEEP_VALUE" in
  *[!0-9:]*|:*|*::*|*:)
    echo "night_v2.sh: NIGHT_HOURS, ITER_TIMEOUT, and RETRY_SLEEP must be non-negative integers" >&2
    exit 1
    ;;
esac

PROMPT="$(cat "$PROMPT_FILE")"
if [[ "$AGENT" == "codex" ]]; then
  AGENT_COMMAND=(
    codex exec
    --sandbox workspace-write
    --cd "$REPO_ROOT"
    "$PROMPT"
  )
else
  AGENT_COMMAND=(
    claude -p "$PROMPT"
    --permission-mode auto
  )
fi

if (( DRY_RUN )); then
  echo "night_v2.sh dry run OK"
  echo "  agent:          $AGENT"
  echo "  repo:           $REPO_ROOT"
  echo "  branch:         $BRANCH"
  echo "  prompt:         $PROMPT_FILE (${#PROMPT} characters)"
  echo "  log:            $LOG_FILE"
  echo "  night hours:    $NIGHT_HOURS_VALUE"
  echo "  iter timeout:   $ITER_TIMEOUT_VALUE seconds"
  echo "  retry sleep:    $RETRY_SLEEP_VALUE seconds"
  if (( RUN_ONCE )); then
    echo "  iterations:     one"
  else
    echo "  iterations:     until time limit or STOP"
  fi
  if [[ "$AGENT" == "codex" ]]; then
    echo "  command:        codex exec --sandbox workspace-write --cd <repo> <prompt>"
  else
    echo "  command:        claude -p <prompt> --permission-mode auto"
  fi
  echo "  side effects:   none"
  exit 0
fi

cd "$REPO_ROOT"

# Claude should normally authenticate with its subscription login. Remove stale
# environment auth only for Claude runs. Set NIGHT_USE_ENV_AUTH=1 to preserve it.
if [[ "$AGENT" == "claude" && -z "${NIGHT_USE_ENV_AUTH:-}" ]]; then
  while IFS= read -r variable_name; do
    unset "$variable_name"
  done < <(
    env | cut -d= -f1 | grep -E '^(CLAUDE|ANTHROPIC|AI_AGENT|BAGGAGE)'
  )
fi

git checkout "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH" || exit 1
mkdir -p research/reports/auto

END=$(( $(date +%s) + NIGHT_HOURS_VALUE * 3600 ))
i=0
LAST_RC=0
while (( $(date +%s) < END )) && [[ ! -f research/STOP ]]; do
  i=$((i + 1))
  echo "=== agent=$AGENT iter=$i start $(date '+%F %T') ===" >> "$LOG_FILE"

  # perl alarm is a portable timeout for macOS, which lacks GNU timeout.
  perl -e 'alarm shift; exec @ARGV' "$ITER_TIMEOUT_VALUE" \
    "${AGENT_COMMAND[@]}" < /dev/null >> "$LOG_FILE" 2>&1
  rc=$?
  LAST_RC=$rc

  echo "=== agent=$AGENT iter=$i exit=$rc $(date '+%F %T') ===" >> "$LOG_FILE"
  (( RUN_ONCE )) && break
  # A non-zero status may indicate a usage limit, classifier abort, or timeout.
  # Retry until NIGHT_HOURS expires, sleeping through part of the usage window.
  (( rc != 0 )) && sleep "$RETRY_SLEEP_VALUE"
done

echo "=== night_v2.sh agent=$AGENT done $(date '+%F %T') ===" >> "$LOG_FILE"

if (( RUN_ONCE )); then
  exit "$LAST_RC"
fi
