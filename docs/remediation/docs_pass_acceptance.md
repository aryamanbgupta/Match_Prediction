# Rolling documentation pass — acceptance checks

Written 2026-09-10 by Claude at the user's request: documentation is kept
current with each landed item rather than deferred entirely to item 8
(item 8 still owns the final consolidation and feature-count sweep).

| # | Check | Pass condition |
|---|---|---|
| G1 | CLAUDE.md, model 1 paragraph | feature count reads 48 (46 numeric + 2 encoded, per `build_i7_match_frame.py`), with a one-line note that 49 was a doc error |
| G2 | CLAUDE.md quick start | pytest instructions (`uv run --no-sync pytest -q`, markers, `-m "not needs_artifacts"`), CI on every push; `--rebuild-snapshot`/state paths refer to the `data/live_state_i7` symlink and versioned build dirs; `artifacts.py verify` mentioned as the first step before any eval; cost flags mentioned on the reslice/blend commands |
| G3 | CLAUDE.md "Where to look" table | rows for `docs/REMEDIATION_PLAN_2026-09-09.md` (what is being fixed and in what order), `docs/DAILY_PREDICTION_PROTOCOL.md`, `models/MANIFEST.yaml` + `docs/registered_odds.json` (artifacts and evidence of record), `docs/remediation/` (acceptance checks and results) |
| G4 | CLAUDE.md critical invariants | new invariants: 8 — artifacts of record are named by manifest role, hashes verified, live state immutable per build dir; 9 — verdicts are written only through the gate (log loss decides, returns are a safety check, provisional evidence can never LAND); 10 — reslice does not reprice stored records, so market comparisons must be built on registered odds (the gate recomputes) |
| G5 | docs/OPERATIONS.md | new sections: "Running the tests" (both suites, markers, artifact-free parity procedure); "Cost model and volume basis" (flags on the five scripts, scenario defaults, stamps); "Artifacts of record" (`artifacts.py verify/path/pull/rebuild`, precedence, live-state versioning and `BUILT`, how to add a role); "Claim gate" (three invocations, gate JSON, `log_verdict verdict --gate-json`, `reassess`, `queue-confirm`, `review-done`, what refusals mean); Operation 6 updated for the symlink and manifest roles |
| G6 | docs/ARCHITECTURE.md | §2.5 lists `market_math.py`, `claim_gate.py`; §2 gains `artifacts.py`; §5.5 evaluation JSON documents the new stamps (`cost_model`, `price_basis`, `volume_basis`, `pnl_unrecomputable`, `price_rejected`, `cluster_resolution`); §6 gains "6.15 One home per job" (betting math, legacy price boundary in `eval_statistics`, gate) and "6.16 Evidence of record" (registered odds, manifest, why stored prices are never evidence); feature count 48 where stated |
| G7 | TODO.md | header note pointing to the plan and `docs/remediation/`; items 1, 2, 5 (1–4) marked done with commit hashes; items 3 (step 1), 4 (in review) marked in progress; the corpus-swap note |
| G8 | README.md | test/CI line and the artifacts verify line updated; nothing else unless wrong |
| G9 | IMPROVEMENTS.md | one dated "Remediation plan progress" entry summarising items 1, 2, 4, 5 with the before/after facts already recorded in the acceptance docs (no new numbers) |
| G10 | Consistency | `grep -rn "49 match-level\|49 features"` empty outside archives; `grep -rn "live_state_i7/"` only where the symlink is meant; no doc instructs `pytest scripts/tests` without `uv run --no-sync`; every command shown is copy-paste runnable on this checkout |
| G11 | No evidence or reports edited; `docs/cricml_atlas_updated.html` untouched; program.md and research/ untouched (item 4 owns them) |
| G12 | Claude reads the full diff |
