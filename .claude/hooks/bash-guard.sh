#!/usr/bin/env bash
set -euo pipefail

# Escape hatch: set CLAUDE_UNRESTRICTED=1 to disable all checks
[[ "${CLAUDE_UNRESTRICTED:-}" == "1" ]] && exit 0

# Read hook input from stdin
INPUT=$(cat)

# Extract permission mode
MODE=$(echo "$INPUT" | jq -r '.permission_mode // ""')

# Only enforce in bypass mode (--dangerously-skip-permissions)
# In interactive mode, the normal permission prompts are the safety net
[[ "$MODE" != "bypassPermissions" ]] && exit 0

# Extract the bash command
CMD=$(echo "$INPUT" | jq -r '.tool_input.command // ""')
[[ -z "$CMD" ]] && exit 0

# Normalize: collapse newlines and extra whitespace for matching
CMD_NORM=$(echo "$CMD" | tr '\n' ' ' | sed 's/  */ /g')

# --- DENYLIST ---
# Combined regex per category, checked sequentially for clear error messages

check_pattern() {
    local category="$1"
    local pattern="$2"
    if echo "$CMD_NORM" | grep -qE "$pattern"; then
        local matched
        matched=$(echo "$CMD_NORM" | grep -oE "$pattern" | head -1)
        jq -n \
            --arg reason "Blocked by bash-guard ($category): matched '$matched'" \
            '{hookSpecificOutput: {hookEventName: "PreToolUse", permissionDecision: "deny", permissionDecisionReason: $reason}}'
        exit 0
    fi
}

# Category 1: File deletion
check_pattern "deletion" '\brm\b|\brmdir\b|\bunlink\b|\bgit\s+rm\b|find.*-delete'

# Category 2: Git destructive
check_pattern "git-destructive" '\bgit\s+reset\b|\bgit\s+clean\b|\bgit\s+push\s+.*(-f\b|--force)|\bgit\s+branch\s+-D\b|\bgit\s+checkout\s+--\s*\.|\bgit\s+restore\s+\.|\bgit\s+commit\s+.*--amend\b|\bgit\s+stash\s+drop\b'

# Category 3: System commands
check_pattern "system" '\bsudo\b|\bchmod\b|\bchown\b|\bkill\b|\bkillall\b|\bpkill\b'

# Category 4: Data exfiltration
check_pattern "exfiltration" '\bcurl\b.*(-X\s*(POST|PUT|PATCH|DELETE)|-d\b|--data\b|-F\b|--form\b|--upload-file\b)|\bwget\b.*--post'

# Category 5: Shell injection / evasion
check_pattern "shell-injection" '\|\s*(ba)?sh\b|\|\s*zsh\b|\beval\b'

# Category 6: Disk destruction
check_pattern "disk-destruction" '\bdd\b.*\bof=|\bmkfs\b|\bfdisk\b'

# Category 7: Python-based deletion (inline commands / prompt injection vector)
check_pattern "python-deletion" 'python.*\b(rmtree|os\.remove|os\.unlink|shutil\.rmtree)\b|python.*Path.*\.unlink'

# Category 8: Dependency modification (program.md forbids new dependencies)
check_pattern "dependency-mod" '\bpip\s+install\b|\bpip\s+uninstall\b|\buv\s+add\b|\buv\s+remove\b|\buv\s+pip\s+install\b'

# No match — allow the command
exit 0
