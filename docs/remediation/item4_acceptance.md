# Item 4 acceptance checks (verdict rule and gate function)

Written 2026-09-10 by Claude before implementation. Steps 1–4 in scope;
step 5 (TABLED re-run) waits for item 6; step 6 is the standing review.

| # | Check | Pass condition |
|---|---|---|
| D1 | `decide(candidate, baseline, kind, cost) -> Verdict` exists in `scripts/sim_eval/claim_gate.py` | inputs are sliced eval JSON paths; the Verdict carries verdict, seed_count, estimator, provisional flag, cost model, odds sha256, cluster contract (source dir + resolution counts), block count, n dropped symmetrically, ΔLL point + CI, Δprofit point + CI, cost-scenario diagnostics |
| D2 | Match-id contract | any id mismatch between arms raises; records missing outcome or market are dropped from both arms and counted |
| D3 | Seed rule | ≥5 aligned seeds → `seed_mean_match_cluster_ci`, not provisional; 2–4 → same estimator, provisional; 1 → `match_cluster_ci`, provisional; a provisional verdict can never be LANDED: favourable CI + point ΔLL ≥ 0.007 → PROMISING, else TABLED/FAILED per the rules; a test per branch |
| D4 | Cluster contract | any `competition_cluster_id`/`cluster_id` stamped on records is stripped before resolution; blocks come only from the registered cluster source dir; any fallback resolution → DESCRIPTIVE; the function has no parameter for labels |
| D5 | Evidence check | loads the odds file whose sha256 is in `docs/registered_odds.json`; each record's `market_odds` must match it (mismatch → error naming the match); `market_prob`, placement and profit are recomputed from the registered odds under the requested cost model; stored `market_prob`/`realized_pnl` are ignored (test: corrupting both changes nothing; test: an eval JSON on other prices fails even with a matching hash declared) |
| D6 | Reliability | `bootstrap_reliable: false` or <10 blocks → DESCRIPTIVE (test proves it cannot be LANDED); non-finite interval → error |
| D7 | Verdict rules | match-model: LANDED iff ΔLL CI excludes 0 favourably AND Δprofit CI does not exclude 0 unfavourably (zero-cost default); TABLED if ΔLL clears but profit clearly harmful; FAILED otherwise; betting-layer and sim/prop kinds keep their gate pairs via `kind`; Δprofit on the union of bets with zero stake where no bet |
| D8 | Intervals | 95%, 10,000 resamples, seed 42, via `eval_statistics.bootstrap_mean_ci` / `registered_experiment` estimators |
| D9 | `log_verdict.py verdict` | requires `--gate-json`; refuses if the stated verdict ≠ gate verdict, if the gate's odds sha256 is not registered, or if cluster fallbacks > 0; existing claim/verdict tests unchanged |
| D10 | `log_verdict.py reassess <ID>` | appends a dated `**Reassessment:**` block under the idea and a `results.tsv` row `<ID>-r2` (then `-r3`…); test proves every pre-existing line of IDEAS.md and results.tsv is byte-identical after |
| D11 | Review reminder | program.md carries `review_due`; `log_verdict.py` prints a reminder when overdue (date or ≥10 verdicts since last review); test |
| D12 | program.md | VERDICT RULE replaced by the gate-based rule; old text kept verbatim in a dated "superseded" block; `night_v3.sh` / `RUNNER_PROMPT_V3.md` instruct calling the gate and passing `--gate-json` |
| D13 | a8 shim | helpers `claim_gate` needs move into `eval_statistics.py`; `scripts/auto/a8_gate_analysis.py` re-exports them; the 16 importers still work (import smoke test) |
| D14 | `docs/registered_odds.json` | sha256 of `betting_odds_polymarket_v2.json` and `data/golden/betting_odds_golden_v2.json` (and the sealed forward odds file), each with path, row count, and registration date; a test recomputes and matches |
| D15 | Suite | 0 failures; artifact-free collection unchanged; frozen evidence untouched |
| D16 | Review | Astra light (plan §11) plus Claude |
