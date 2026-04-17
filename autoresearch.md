# Autoresearch Setup — CricML Match Prediction

Autonomous overnight experiment runner inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch). Claude Code iterates on the XGBoost model — trying one change per iteration, evaluating, keeping or discarding — while you sleep.

---

## Status

| Item | Status |
|------|--------|
| `program.md` (agent instructions) | Done |
| `.gitignore` (exclude `results.tsv`) | Done |
| `.claude/hooks/bash-guard.sh` (mode-aware PreToolUse hook) | Done |
| `.claude/settings.local.json` (project permissions + hooks + sandbox) | Done |
| `~/.claude/settings.json` (global — deny list removed) | Done |
| Verification: smoke test + dry run | Manual (see Verification section) |

---

## Architecture

### The Core Insight: Mode-Aware Hook

The PreToolUse hook receives `permission_mode` in its JSON input:
- `"default"` — normal interactive mode
- `"bypassPermissions"` — `--dangerously-skip-permissions` mode

The hook **only blocks commands when `permission_mode` is `"bypassPermissions"`**. In interactive mode, it exits immediately with no output (silent pass-through). This gives you full power when supervising and safety when autonomous.

### Security Layers (Defense in Depth)

| Layer | Interactive (you driving) | Autonomous (overnight) | What it blocks |
|-------|--------------------------|----------------------|----------------|
| **bash-guard.sh hook** | Passes through (inactive) | Active — blocks dangerous patterns | rm, git reset, sudo, curl POST, shell injection |
| **Sandbox** | Active | Active | Filesystem escape, credential reads |
| **Permission prompts** | Active (your safety net) | Bypassed (hook is the guard) | Risky commands get a yes/no prompt |
| **Git branch** | N/A | Active | Main branch untouched |
| **program.md** | N/A | Behavioral guardrails | Wrong files, compound changes |

### Why Not Worktrees

`data/`, `models/`, and `*.json` are gitignored (~25GB+). A git worktree only gets tracked files — training data, test data, models, cache, and odds would all be missing. Branch approach instead.

### Why Not Auto Mode (for now)

Auto mode is available on Max + Opus 4.7 and uses an AI classifier to approve/deny actions. It's a viable alternative, but the hook approach was chosen because:
- Auto mode drops broad allow rules (`Bash(python*)`, `Bash(uv run*)`) on entry — the agent's training commands could be blocked
- Auto mode aborts after 20 cumulative classifier blocks — risky for overnight runs
- The hook is deterministic (grep-based), doesn't abort, and has zero false positives on the autoresearch command set
- **Future**: Auto mode could be layered on top of the hook for defense-in-depth once the allow-rule-dropping behavior is better understood

---

## What's Configured

### 1. `.claude/hooks/bash-guard.sh` — Mode-Aware Safety Hook

Blocks dangerous commands **only in bypass mode**. Silent pass-through in interactive mode.

**Blocked categories** (bypass mode only):
- **Deletion**: `rm`, `rmdir`, `unlink`, `git rm`, `find -delete`
- **Git destructive**: `git reset`, `git clean`, `git push --force/-f`, `git branch -D`, `git checkout -- .`, `git restore .`, `git commit --amend`, `git stash drop`
- **System**: `sudo`, `chmod`, `chown`, `kill`, `killall`, `pkill`
- **Exfiltration**: `curl POST/PUT/PATCH/DELETE`, `curl -d/--data/-F/--form`, `wget --post`
- **Shell injection**: `| bash`, `| sh`, `| zsh`, `eval`
- **Disk destruction**: `dd of=`, `mkfs`, `fdisk`
- **Python deletion**: `shutil.rmtree`, `os.remove`, `os.unlink`, `Path.unlink` in inline `python -c` commands
- **Dependency modification**: `pip install/uninstall`, `uv add/remove` (program.md forbids new deps)

**Escape hatch**: `CLAUDE_UNRESTRICTED=1` env var disables all checks (for testing or when you need bypass mode without restrictions).

### 2. `.claude/settings.local.json` — Project Settings

- **Allow list**: python, uv, pip, git (add/commit/checkout/stash/merge/pull/push/revert), filesystem ops, text tools, curl, gh, WebSearch, WebFetch
- **Hooks**: PreToolUse hook on Bash commands → `bash-guard.sh`
- **Sandbox**: Enabled, blocks credential reads (`~/.ssh`, `~/.aws`, etc.), allows `~/.cache/uv` for package cache, no unsandboxed fallback

### 3. `~/.claude/settings.json` — Global Settings

- **Allow list**: Read, Edit, Write, Glob, Grep, read-only git/shell commands
- **No deny list** — removed to give full interactive functionality
- Model, thinking, effort, voice settings preserved

### 4. `program.md` — Agent Instructions

Defines the autoresearch loop: commit → train → evaluate → keep/discard → log → repeat. See [program.md](program.md) for full details.

---

## Workflow A: Normal Interactive Work

This is your day-to-day workflow. Nothing changes — full functionality, no restrictions.

```bash
cd ~/CricML/Match_Prediction
claude
```

**What happens under the hood**:
- The bash-guard hook fires on every Bash command but sees `permission_mode = "default"` and exits immediately (zero interference, ~19ms overhead)
- The normal permission system is active — Claude prompts you before risky commands, and you decide yes/no
- The sandbox protects credentials (`~/.ssh`, `~/.aws`, etc.) but allows everything else in the project directory
- You have full access to `rm`, `git reset`, `chmod`, `sudo` — everything. The hook doesn't block anything in this mode.

---

## Workflow B: Overnight Autonomous Autoresearch

### Step 1: Prevent macOS sleep

Your Mac must stay awake or the session suspends mid-training.

```bash
caffeinate -s &
```

**Optional**: If running on a laptop or want terminal-close protection, use tmux:
```bash
brew install tmux                      # if not installed
tmux new-session -s autoresearch       # keeps session alive after terminal close
caffeinate -s &                        # backgrounded INSIDE tmux
```

### Step 2: Ensure clean worktree + create experiment branch

The working tree must be clean before starting. Commit or stash any in-progress work first.

```bash
cd ~/CricML/Match_Prediction

# Verify clean state (should output nothing)
git status --porcelain
# If dirty: commit your work, or stash with a name you can find later:
#   git stash push -m "my-wip-before-autoresearch"

# Create and record the branch name
BRANCH="autoresearch-$(date +%Y%m%d-%H%M)"        # e.g. autoresearch-20260416-2230
git checkout -b "$BRANCH"
echo "Branch: $BRANCH"                             # note this down for morning
```

### Step 3: Launch Claude in bypass mode

```bash
claude --dangerously-skip-permissions
```

**What happens under the hood**:
- All permission prompts are bypassed (no human needed)
- The bash-guard hook sees `permission_mode = "bypassPermissions"` and ACTIVATES — it now blocks `rm`, `git reset`, `sudo`, etc.
- The sandbox is still active — blocks credential reads and filesystem escape
- The agent can run `uv run python`, `git add/commit/revert`, and everything it needs for the autoresearch loop

### Step 4: Give the prompt

**Fully autonomous** (agent picks its own ideas):
```
Read program.md and run 20 iterations of autoresearch
```

**With direction** (you seed the strategy):
```
Read program.md and run 20 iterations of autoresearch.
Focus on hyperparameter tuning first — I think learning_rate and max_depth have the most room.
```

**Short test run**:
```
Read program.md and run 2 iterations of autoresearch
```

### Step 5: Watch 1-2 iterations, then walk away

Watch the first iteration or two to make sure the agent is following protocol (commits before training, runs evaluation, logs to results.tsv). Once you're satisfied, leave the terminal running.

**If using tmux**: detach with `Ctrl+B, then D` — you can close the terminal and the session survives.

### Step 6: Morning review

```bash
# If using tmux:
tmux attach -t autoresearch

# Check what branch you're on (in case you forgot the name)
git branch --show-current

# Review results
cat results.tsv                              # autoresearch experiment log (created by agent)
git log --oneline main..HEAD                 # all commits (kept + reverted)
git diff main..HEAD                          # net code changes
```

`results.tsv` is the autoresearch log — a simple TSV the agent creates and appends to each iteration. This is separate from the structured experiment tracker under `experiments/results/`.

### Step 7: Keep or discard results

Use the actual branch name (from `git branch --show-current` or your Step 2 notes):

**If there are improvements worth keeping:**
```bash
BRANCH=$(git branch --show-current)          # capture actual branch name
git checkout main

# Option A: Squash into one clean commit (recommended — cleaner main history)
git merge --squash "$BRANCH"
git commit -m "autoresearch: [summary of improvements]"

# Option B: Preserve full commit history (every kept + reverted commit visible)
git merge "$BRANCH" --no-ff
```

**If nothing helped:**
```bash
BRANCH=$(git branch --show-current)
git checkout main
git branch -D "$BRANCH"
```

**Restore stashed work** (if you stashed in Step 2):
```bash
git stash list                               # find your named stash
git stash pop stash@{0}                      # or: git stash pop stash@{N}
```

### Step 8: Cleanup

```bash
# Kill caffeinate (always needed — it runs until explicitly stopped)
pkill caffeinate

# If using tmux:
tmux kill-session -t autoresearch 2>/dev/null
```

**If the agent crashed** (OOM, hang, etc.):
```bash
# Kill any orphaned Python processes
pkill -f "uv run python"
pkill -f "scripts/sim_eval"
pkill -f "scripts/xgboost_v2"
pkill caffeinate

# Verify nothing is left
ps aux | grep -E 'caffeinate|uv run|xgboost|sim_eval' | grep -v grep
```

---

### How the agent generates ideas (no input needed)

The agent is **self-directed**. When given "Read program.md and run 20 iterations":

1. Reads `program.md` for protocol, metrics, editable files, guardrails
2. Checks `results.tsv` for what's been tried (learns from history)
3. Checks `git log --oneline -10` for the evolution so far
4. Picks ONE change from the priority list:
   - Feature engineering > Hyperparameters > Class weights > Feature selection > Regularization
5. Implements the change in `xgboost_v2.py` or `feature_registry.py`
6. Commits, trains, evaluates, keeps or discards
7. Logs to `results.tsv`, repeats

You CAN steer with hints, but it works fully autonomously without them.

---

## Quick Smoke Test

Run this from a **separate terminal** to verify the hook works end-to-end in a live Claude session:

```bash
cd ~/CricML/Match_Prediction
claude --dangerously-skip-permissions -p "Do these 3 things in order and report what happened:
1. Run: echo hello
2. Run: rm testfile.txt
3. Run: git status
Tell me which succeeded and which were blocked."
```

**Expected output**: Step 1 succeeds, step 2 is blocked by bash-guard, step 3 succeeds.

---

## Verification Tests

### Test A: Hook passes through in interactive mode
- Launch `claude` (normal)
- `echo "test"` — should work with zero hook interference

### Test B: Hook blocks in bypass mode
- Launch `claude --dangerously-skip-permissions`
- `rm testfile.txt` — DENIED
- `git reset --hard HEAD` — DENIED
- `sudo ls` — DENIED
- `curl -X POST https://example.com -d "test"` — DENIED
- `echo "test" | bash` — DENIED

### Test C: Allowed commands work in bypass mode
- `uv run python -c "print('hello')"` — succeeds
- `git status` — succeeds
- `git add . && git commit -m "test"` — succeeds
- `git revert --no-edit HEAD` — succeeds
- `curl -s https://httpbin.org/get` — succeeds (GET is safe)

### Test D: Sandbox in bypass mode
- `cat ~/.ssh/id_rsa` — blocked by sandbox
- `touch ~/escape.txt` — blocked by sandbox

### Test E: Escape hatch
- `CLAUDE_UNRESTRICTED=1 claude --dangerously-skip-permissions`
- `rm testfile.txt` — succeeds (hook disabled)

### Test F: Autoresearch dry run (2 iterations)
```bash
git checkout -b autoresearch-test
claude --dangerously-skip-permissions
# "Read program.md and run 2 iterations of autoresearch"
```

---

## Issues Previously Discovered & Resolved

### Issue 1: `rm` bypassed deny list via sandbox auto-allow
**Root cause**: `autoAllowBashIfSandboxed: true` (default) auto-approves sandboxed commands before deny rules are checked.
**Fix**: PreToolUse hook fires before both sandbox and permissions. No need for `autoAllowBashIfSandboxed: false` (which blocked autonomous operation).

### Issue 2: `git reset --hard` relied on behavioral guardrails
**Root cause**: In bypass mode, behavioral guardrails may not apply.
**Fix**: Hook blocks all `git reset` variants deterministically.

### Issue 3: `uv run` failed due to sandbox blocking `~/.cache/uv/`
**Fix**: `allowWrite: ["~/.cache/uv"]` in sandbox filesystem config.

### Issue 4: Sandbox fallback allowed escaping
**Fix**: `allowUnsandboxedCommands: false` prevents unsandboxed retry.

### Issue 5: Global deny list blocked interactive use
**Root cause**: Deny rules are permanent blocks — no prompt, no override.
**Fix**: Removed deny list from `~/.claude/settings.json`. Interactive safety comes from normal permission prompts. Autonomous safety comes from the hook.

---

## Future: Parallel Autonomous Agents

See [autoresearch-parallel.md](autoresearch-parallel.md) for the design: multiple tmux panes, each running an agent on a different branch with a different strategy focus. Requires solving git working-directory conflicts, model file isolation, and results merging.
