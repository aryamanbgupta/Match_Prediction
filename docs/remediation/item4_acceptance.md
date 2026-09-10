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

## Fix pass — 2026-09-10

The Astra/Claude findings are locked by the following checks:

| Finding | Implementation | Locking test |
|---|---|---|
| F1 seed identity bypass | unique resolved paths and SHA-256s per arm; stamped summary seeds must match labels; `Verdict.arm_files` records hashes | `test_duplicate_seed_files_and_wrong_stamped_seed_raise`; `test_five_seeds_can_land_and_three_are_promising` |
| F2 cluster directory bypass | only the registry's `cluster_source_dir` is used; an optional caller value must resolve identically | `test_registered_cluster_dir_is_authoritative` |
| F3 edited verdict string | `verify_gate_payload` recomputes classification and validates provisional/seed/estimator/block/reliability/fallback contracts; logging independently rejects provisional or sub-five-seed LANDED claims | `test_landed_claim_is_recomputed_and_tampering_is_refused` (six tamper cases) |
| F4 kind-specific gates | `betting_layer` and `sim_prop` consume registered, paired per-match metric vectors from `--metrics-json` | `test_kind_specific_betting_and_sim_prop_gate_pairs` |
| F5 profit population | Δprofit uses the union of placed bets, zero for the non-betting arm, and excludes both-no-bet rows; `n_profit_rows` is stamped | `test_profit_delta_uses_union_with_zero_for_no_bet` |
| F6 PROMISING retention | only TABLED/FAILED/DESCRIPTIVE/CRASH are reverted; `queue-confirm` appends `<ID>-confirm` as PENDING | `test_queue_confirm_is_append_only_and_pending` |
| F7 review counter | every verdict/reassessment increments the stored counter; reminders use date OR counter; `review-done` resets and advances 90 days | `test_verdict_appends_gate_hash_to_notes`; `test_reassess_only_appends_history_and_numbers_revisions`; `test_review_reminder_when_date_or_count_due`; `test_review_done_resets_counter_and_sets_due_plus_90_days` |
| F8 reassessment append proof | reassessments append to both files and do not insert or rewrite prior IDEAS lines | `test_reassess_only_appends_history_and_numbers_revisions` starts with a completed Result and existing results row, then checks byte prefixes |
| F9 fewer than ten blocks | reliability downgrades the result to DESCRIPTIVE | `test_fewer_than_ten_blocks_is_descriptive` |

## Fix pass 2 — 2026-09-10

| Finding | Implementation | Locking test |
|---|---|---|
| G1 seed identity | multi-seed sliced/metrics JSONs require a matching summary seed stamp; arm identity hashes canonical JSON; repeated per-match prediction vectors are rejected | `test_seed_identity_is_content_stamped_and_prediction_distinct` |
| G2 payload reconciliation | verification reconciles seed/arm cardinality and hashes, estimator/provisional state, registered role hash and cluster directory, reliability/fallback/block requirements, and both profit-population counts | `test_coordinated_gate_metadata_edits_are_refused`; `test_landed_claim_is_recomputed_and_tampering_is_refused` |
| G3 fixed metrics and provenance | `GATE_PAIRS` owns accepted names/directions; betting files supply placements settled from registered prices and sliced outcomes/probabilities; sim/prop files cite a hash-verified detail JSON stamped as `metric_source` | `test_kind_specific_betting_and_sim_prop_gate_pairs`; `test_metric_names_directions_settlement_and_source_hash_are_locked` |
| G4 profit-population blocks | the Verdict stamps unique `profit_block_count`; fewer than ten profit blocks makes every profit-deciding claim DESCRIPTIVE | `test_profit_population_with_only_nine_blocks_is_descriptive` |
| G5 PROMISING ordering | program and runner require logging PROMISING before `queue-confirm`, which continues to require PROMISING | `test_queue_confirm_is_append_only_and_pending`; `test_queue_confirm_requires_promising` |
| G6 kind-specific CLI | match-model uses sliced arms only; betting-layer uses sliced arms plus placements; sim/prop uses metrics only; CLI validation and both runner documents match | `test_kind_specific_betting_and_sim_prop_gate_pairs`; CLI coverage in `test_cli_writes_json_and_prints_its_sha256` |

`reslice_eval_json.py` was intentionally not changed: it reconstructs a new
summary rather than copying the input summary, and the item 6 harness owns the
`model_seed` stamp.

## Fix pass 3 — 2026-09-10

This section supersedes the fix-pass-2 descriptions of automated sim/prop and
betting metric B.

| Finding | Implementation | Locking test |
|---|---|---|
| H1 fabricated gate JSON | Automated gates stamp repo-relative path + canonical JSON SHA-256 for every arm/seed and all normalized `decide()` inputs. Verification performs cheap schema/hash checks, re-runs `decide()` from those files, and requires the complete dictionaries to match. | `test_gate_replay_refuses_fabricated_numbers_and_edited_evidence`; genuine replay assertion in `test_five_seeds_can_land_and_three_are_promising` |
| H2/H6 sim/prop authority | Automated `decide()` accepts only match-model and betting-layer work. `record-manual --kind sim_prop` records the idea, allowed verdict, pre-committed pair text, and raw SHA-256 of its gate script and every detail JSON under `gate_mode: manual_sim_prop`; logging rechecks them and stamps the mode in notes. Standing review owns judgment. | `test_kind_specific_betting_gate_pair_and_manual_sim_prop`; `test_verdict_appends_gate_hash_to_notes`; manual tamper parameterizations in `test_log_verdict_gate.py` |
| H3 duplicate seeds after filtering | Match-model prediction vectors are compared only after symmetric filtering. Betting-layer placement vectors are also unique per arm after row validation. | `test_duplicate_checks_use_kept_predictions_and_placements` |
| H4 betting ROI | Betting headline is ΔROI, where each arm is `sum(pnl) / sum(stake)` over the union-of-placements population. Its CI resamples tournament blocks and paired seeds, recomputes both ratios, and differences them (10,000 draws, seed 42). Δprofit remains stamped. | `test_betting_layer_decides_ratio_of_sums_roi_and_preserves_probabilities` (2× stake, equal ROI cannot LAND) |
| H5 probability invariance | Betting candidate/baseline `simulated_prob` maps must have identical keys and values within absolute tolerance 1e-12. Betting has no metric B. | `test_betting_layer_decides_ratio_of_sums_roi_and_preserves_probabilities`; `test_metric_names_directions_settlement_and_source_hash_are_locked` |
| H7 cost diagnostics | Betting placements are re-settled under every configured scenario; diagnostics stamp both Δprofit and ratio-of-sums ΔROI. | `test_betting_layer_decides_ratio_of_sums_roi_and_preserves_probabilities` |

## Review closure (Claude, 2026-09-10)

Five Astra rounds. Rounds 1–3 found real bypasses (seed duplication, cluster
directory substitution, trusted verdict strings, generic sim/prop pairs,
profit population, PROMISING reverts, ROI vs profit, optional non-degradation),
all fixed by Sol with locking tests. Round 4 found three defects Claude fixed
directly (tuple/list replay mismatch, textual prediction comparison, NaN past
the invariance check). Round 5 found only signed zero in the canonical
prediction vector, fixed by Claude. Review closed at that point: findings had
moved from rule bypasses to numeric representation edge cases, each with a
regression test. Suite: 458 passed, 0 failed. Steps 1–4 complete; step 5
(TABLED re-run) waits for item 6, step 6 is the standing review
(`review_due: 2026-12-09`).
