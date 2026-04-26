# Archive

Historical documentation. **Not authoritative for current code.** Useful for
"how did this get built?" questions, but everything load-bearing for present
work has been folded into the live docs:

- [../ARCHITECTURE.md](../ARCHITECTURE.md) — current system architecture, data
  formats, design decisions
- [../OPERATIONS.md](../OPERATIONS.md) — current commands and procedures
- [../ADDING_NEW_MODELS.md](../ADDING_NEW_MODELS.md) — current model-plugin guide
- [../feature_roadmap.md](../feature_roadmap.md) — current feature catalog
- [../../IMPROVEMENTS.md](../../IMPROVEMENTS.md) — current research log
- [../../TODO.md](../../TODO.md) — current workstreams

## What's in here

| File | Origin | Replaced by |
|---|---|---|
| `CLAUDE_REFERENCE.md` | Comprehensive technical reference, last updated Oct 2024 | `ARCHITECTURE.md` |
| `DATA_FORMATS.md` | Standalone data-format spec (chunked-pickle era) | `ARCHITECTURE.md` § 5 |
| `DESIGN_DECISIONS.md` | Standalone design-rationale doc (chunked-pickle era) | `ARCHITECTURE.md` § 6 |
| `DATA_REFRESH_HANDOFF.md` | One-time handoff for the April 2026 Cricsheet refresh + Polymarket build | Work complete; see `IMPROVEMENTS.md` |
| `SQLITE_MIGRATION_PROFILE.md` | One-time migration memo (chunks → SQLite, April 2026) | Work complete; rationale captured in `ARCHITECTURE.md` § 6.3 |
| `POLYMARKET_INTEGRATION.md` | Test-set expansion notes (44 → 261 matches) | Build complete; eval gates in `TODO.md` |
| `EVAL_PROFILING.md` | One-time profiling memo for the eval hot-path (April 2026) | Work complete; XGBoost Fix A+B shipped |
| `claude_to_do_improvements.md` | Pre-fix improvement roadmap (Jan 2026) | Items either shipped or rolled into `IMPROVEMENTS.md` / `TODO.md` |
| `cricket_prediction_workflow.md` | Pre-implementation aspirational design (Sept 2025) | Superseded by actual implementation |
| `cricket_simulation_docs.md` | Sim-engine documentation (Sept 2025) | `ARCHITECTURE.md` § 4 (current state) |
