# Archive

Local-only archive for non-documentation artifacts. **Not authoritative for
current code or pipelines.** Gitignored — these files exist only on this
machine. For documentation history, see [`docs/archive/`](../docs/archive/).

## Layout

```
archive/
├── logs/           — training logs, parsing logs, phase 4 migration logs
├── eval_results/   — historical eval JSONs + one-time eval-output dirs
│   └── old_dec_2024/   — Dec 2024 mlp_v1 / mlp_v2_v1 / xgboost_v1 result files
├── scripts/        — superseded Python scripts (kept for reference)
└── notebooks/      — pre-pipeline Jupyter notebooks (EDA + early experiments)
```

## Why these were moved

| Subdir | Contents | Why kept |
|---|---|---|
| `logs/` | `training_debug*.log`, `phase4_*.log`, `cron_log.log`, `parsing_output.log` | Reference for past training runs / SQLite migration |
| `eval_results/` | 26× `match_evaluation_results_xgboost_*.json` from root, `eval_out_baseline/`, `eval_out_postfix/`, `phase4_*_eval/`, `eval_profile.prof` | Historical evaluations from before sliced eval was wired (Phase 1, 2026-04-24); the `eval_out_baseline`/`eval_out_postfix` dirs document the recent-form fix A/B (2026-04-21) |
| `scripts/` | 30 legacy scripts (sim v1/v1.1, xgboost v1, gbm v1, cricinfo HTML scrapers, pre-Cricsheet parsers, sim_eval debug scripts, top-level test files, plus `bet_scraper.py` + `parse_gbm.py` from the old `src/` dir) | Historical code, replaced by current pipeline. Use `git log -- archive/scripts/<name>.py` to recover history. |
| `notebooks/` | 4 ipynb files (`baseline_model`, `feature_eng`, `model_development`, `simulation`) + `trial.py` | Pre-pipeline EDA and early model experiments (Dec 2024 – Aug 2025). Superseded by the `experiments/` infrastructure. |

## What's NOT here

- **Current eval runs** — `eval_out_phase{1,2,3,5,6}_*/` at the repo root.
  These are active Phase 1–6 sliced/ablation outputs, not archived.
- **Current scripts** — `scripts/mlp_v2.py` and `scripts/transformer_mlx.py`
  are still imported by live code; both stay in `scripts/`.
- **`scripts/enrich_players_cricketdata.py`** — the R `cricketdata` script,
  current player-metadata path. Stays in `scripts/`.
- **Documentation history** — see `docs/archive/` for old design docs and
  one-time migration memos.

## Recovery

These files are gitignored; the legacy scripts in `archive/scripts/` *are*
tracked and have full git history under the new path. To resurrect any of
them:

```bash
# Move it back
git mv archive/scripts/<name>.py scripts/<name>.py

# Or just inspect the history
git log -p archive/scripts/<name>.py
```

For deleted (not-archived) files, use `git log -- <path>` from before the
cleanup commit.

## Deletion policy

Anything in `archive/` can be deleted at any time without affecting the
running system. The reason it's local-only (not git-tracked) is exactly that
— it should never become load-bearing.
