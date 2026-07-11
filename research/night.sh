#!/usr/bin/env bash
# Overnight autonomous research loop.
# Usage:  caffeinate -i ./research/night.sh        (run inside tmux)
# Stop:   touch research/STOP   (or kill the tmux session)
# Env:    NIGHT_HOURS (default 9), ITER_TIMEOUT seconds (default 7200)

cd "$(dirname "$0")/.."

# If launched from inside a Claude Code session, drop its session env so the
# child `claude -p` authenticates with the normal user login instead of the
# parent session's proxy (which 401s for children).
if [[ -n "${CLAUDECODE:-}" ]]; then
  while IFS= read -r v; do unset "$v"; done < <(
    env | cut -d= -f1 | grep -E '^(CLAUDE|ANTHROPIC|AI_AGENT|BAGGAGE)')
fi

BRANCH="auto-$(date +%Y%m%d)"
git checkout "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH" || exit 1
mkdir -p research/reports/auto

END=$(( $(date +%s) + ${NIGHT_HOURS:-9} * 3600 ))
i=0
while (( $(date +%s) < END )) && [[ ! -f research/STOP ]]; do
  i=$((i + 1))
  echo "=== iter $i start $(date '+%F %T') ===" >> research/night.log
  # perl alarm = portable timeout (macOS has no GNU `timeout`)
  perl -e 'alarm shift; exec @ARGV' "${ITER_TIMEOUT:-7200}" \
      claude -p "$(cat research/RUNNER_PROMPT.md)" \
      --permission-mode auto < /dev/null >> research/night.log 2>&1
  rc=$?
  echo "=== iter $i exit=$rc $(date '+%F %T') ===" >> research/night.log
  # Non-zero: usage limit, classifier abort, or timeout. Sleep out part of the
  # 5h usage window and retry; the wall clock (NIGHT_HOURS) is the final stop.
  (( rc != 0 )) && sleep 1800
done
echo "=== night.sh done $(date '+%F %T') ===" >> research/night.log
