# Parallel Autonomous Agents — Future Improvement

Run multiple Claude Code agents simultaneously, each exploring a different research direction. Instead of one sequential agent doing 20 iterations overnight, run 3-4 agents in parallel doing 5-8 iterations each across different strategies.

---

## Approach: Multiple tmux Panes + Branches

```bash
# Launch 3 parallel agents, each on its own branch
tmux new-session -s autoresearch

# Pane 1: Hyperparameter exploration
tmux send-keys "cd ~/CricML/Match_Prediction && \
  git checkout -b auto-hparams-$(date +%Y%m%d) && \
  claude --dangerously-skip-permissions -p \
  'Read program.md and run 8 iterations. Focus ONLY on hyperparameter tuning: learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, reg_alpha, reg_lambda.'" Enter

# Pane 2: Feature engineering
tmux split-window -h
tmux send-keys "cd ~/CricML/Match_Prediction && \
  git checkout -b auto-features-$(date +%Y%m%d) && \
  claude --dangerously-skip-permissions -p \
  'Read program.md and run 8 iterations. Focus ONLY on feature engineering: ablating weak features, adding feature interactions, modifying feature groups.'" Enter

# Pane 3: Class weights / regularization
tmux split-window -v
tmux send-keys "cd ~/CricML/Match_Prediction && \
  git checkout -b auto-weights-$(date +%Y%m%d) && \
  claude --dangerously-skip-permissions -p \
  'Read program.md and run 8 iterations. Focus ONLY on class weight rebalancing and L1/L2 regularization.'" Enter
```

---

## Challenges to Solve

1. **Git conflicts**: Multiple agents on different branches but the same working directory will conflict. Each agent needs its own working copy.
   - **Solution A**: Use `git worktree` with symlinks for data/models (requires a `WorktreeCreate` hook to set up symlinks for gitignored data)
   - **Solution B**: Clone the repo multiple times into separate directories, symlink `data/` and `models/` into each clone
   - **Solution C**: Use containers (Docker) with mounted volumes for data/models

2. **Model file conflicts**: All agents write to `models/xgb_v3/xgboost_model_v3.pkl`. With shared data directories (symlinks), one agent's trained model could overwrite another's mid-evaluation.
   - **Solution**: Modify `xgboost_v2.py` to accept a `--model-dir` flag so each agent writes to a unique output directory (e.g., `models/xgb_v3_hparams/`, `models/xgb_v3_features/`)

3. **Resource contention**: XGBoost training uses multiple CPU cores. 3 concurrent training runs may slow each other down.
   - **Solution**: Set `n_jobs=2` per agent (instead of default all cores) or stagger launches with a 10-minute offset

4. **Results merging**: Each agent produces its own `results.tsv`. Need to merge results and identify the best changes across all branches.
   - **Solution**: Use `compare_experiments.py` or a simple script to merge and rank results by log loss across branches

---

## Implementation Steps

1. **Add `--model-dir` flag** to `xgboost_v2.py` so agents can write to separate output directories
2. **Create a `setup-parallel.sh` script** that:
   - Creates N clones of the repo
   - Symlinks `data/` and `models/cache_chunks_v3/` into each clone
   - Creates a unique branch per clone
3. **Create a `merge-results.sh` script** that:
   - Collects `results.tsv` from each clone
   - Ranks all experiments by log loss
   - Identifies the best changes for manual review
4. **Update `program.md`** to support a `--model-dir` argument for isolated model output

---

## Worktree Approach (Alternative)

If we add a `WorktreeCreate` hook to symlink data directories into worktrees, the native `--worktree` flag becomes viable for parallel agents:

```bash
# .claude/settings.local.json — add WorktreeCreate hook
{
  "hooks": {
    "WorktreeCreate": [
      {
        "hooks": [{
          "type": "command",
          "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/worktree-setup.sh"
        }]
      }
    ]
  }
}
```

```bash
# .claude/hooks/worktree-setup.sh
#!/bin/bash
INPUT=$(cat)
WORKTREE_PATH=$(echo "$INPUT" | jq -r '.worktree_path // ""')
MAIN_REPO="$CLAUDE_PROJECT_DIR"

# Symlink large gitignored directories instead of copying
ln -sf "$MAIN_REPO/data" "$WORKTREE_PATH/data"
ln -sf "$MAIN_REPO/models" "$WORKTREE_PATH/models"
ln -sf "$MAIN_REPO/betting_odds_v3.json" "$WORKTREE_PATH/betting_odds_v3.json"
```

This would enable:
```bash
claude --worktree agent-hparams --dangerously-skip-permissions -p "..."
claude --worktree agent-features --dangerously-skip-permissions -p "..."
claude --worktree agent-weights --dangerously-skip-permissions -p "..."
```

**Note**: Shared model output directories via symlinks cause write conflicts. The `--model-dir` fix is needed regardless of approach.

---

## Estimated Throughput

| Setup | Iterations/Night | Strategies Explored |
|-------|-----------------|-------------------|
| Single agent (current) | ~20 | 1 direction |
| 3 parallel agents | ~24 (8 each) | 3 directions |
| 4 parallel agents | ~28 (7 each) | 4 directions |

Diminishing returns beyond 4 agents due to CPU contention on training.
