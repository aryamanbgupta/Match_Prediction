# Stage 2 acceptance checks (sequence and embeddings track)

Written 2026-09-11 by Claude Fable before any stage 2 work started, per
`docs/sequence_track/stage2_kickoff_brief.md` § 1 and
`docs/SEQUENCE_TRACK_PLAN.md` § "Stage 2". Each table is written before its
deliverable begins; the "Result" block under each table is filled in only
when the deliverable is done, with numbers copied verbatim from files. A run
that did not finish has no number. Tables for D9–D11 are completed
immediately before those steps start, after the user's go for that step.

Standing rules (unchanged from stage 1 unless stated): no read of
`data/golden/` or `data/forward_holdout/` **except the two id-only reads
recorded in D2.3**; nothing written under `models/` outside
`models/embeddings/seq_stage2/` (code, configs, tests and this file live in
their usual places: `scripts/`, `scripts/sequence_track/`, `tests/`,
`scripts/tests/`, `experiments/configs/`, `research/sequence_track/`,
`docs/sequence_track/`); no edit to frozen evidence, committed reports,
`IMPROVEMENTS.md`, `program.md`, `research/results.tsv`,
`models/MANIFEST.yaml`, `scripts/daily/`, `daily/`, the live-state build
dirs, anything under `models/embeddings/seq_stage1/` or
`models/embeddings/t1_ablation_v1_mps/`, the committed stage 1 report and
addendum, or `experiments/configs/seq_stage1_*.yaml` (`TODO.md` backlog
entries are editable on the user's instruction, as in D2.4). Python runs
through `uv run --no-sync`. Multi-seed training runs through
`research/sequence_track/run_queue.sh`; **amended 2026-09-12 (D8 preamble):
the two registered seeds are split across two machines — seed 7 on the
laptop, seed 13 on the Mac mini, every configuration on both** — everything
else runs on the laptop. No log loss from the D6 smoke is read or
reported anywhere. Opus subagents implement; Sonnet does mechanical work
with a specified case list; Fable reviews every diff; Codex Astra reviews
to sign-off before any commit that lands results; Fable is the only
committer and the only caller of `research/log_verdict.py`.

Design facts established while reading (2026-09-11), which the checks
rely on:

- `data/xgb_data_i7` holds train (1,876,971 rows), validation (124,292 rows,
  545 matches, 2024-12-31 → 2025-06-29) and test (186,667) parquets only.
  There is **no ball-level golden parquet on the i7 frame**; the frame's
  `.feature_hash` declares `golden_start: 2026-04-17` as a date boundary.
  The repo corpus `data/t20s_json` ends 2026-04-16 (11,264 files, 0 later).
- Raw cricsheet for 2026-04-17 onward exists in two places: the closed
  `data/golden/t20s_json`, and the stat-generator zips
  (`/Users/aryamangupta/Projects/stat-generator/data/cricsheet/*.zip`,
  refreshed 2026-09-11 12:24). Reading only each file's `info` block in
  the zips: **471** unique male six-ball T20 matches dated
  2026-04-17 → 2026-08-05 (the plan's 471); 208 before 2026-06-02, 205 in
  the forward window 2026-06-02 → 2026-07-13, 58 in 2026-07-14 → 2026-08-05.
  The sealed forward set is 137 of those 205.
- The match-level `data/xgb_match_data_i7_v2/golden_test.parquet` (not a
  closed path) has 227 matches 2026-04-17 → 2026-07-12; 124 of the
  window's matches were scored at match level in the golden evaluation.
- The validation split maps to **47 tournament blocks** under
  `scripts/sim_eval/eval_statistics.load_competition_clusters` over
  `data/t20s_json` (event name, 120-day inactivity gap), 0 unmapped
  matches; the largest are IPL 2025 (74 matches), Vitality Blast 2025
  (71), BPL 2024-25 (44).
- `scripts/transformer_t1.py`: `T1Model.__init__` accepts arms
  `{full, mlp, no_attention, no_history}`; tokens are
  `feat_proj(feats) + out_emb(prev_y) + pos_emb(pos)`; `collate` builds
  `prev_y` as the innings-previous outcome (BOS at row 0); innings are
  `innings_id` groups in parquet order; `--save-predictions` writes
  row-aligned validation probabilities (`npz`). The frame carries
  `batter_id`, `bowler_id`, `non_striker_id`, `inning_idx`,
  `is_death_overs`, `chase_target` per row.
- `scripts/sequence_track/retrain_i7.py` is the multi-seed driver; its
  reuse check compares architecture, optimiser, seed, frame hash, parquet
  md5/rows/dates and cache md5. It does not yet know arm parameters such
  as `k` or wiring.
- `research/sequence_track/queue.yaml` holds the stage 1 placeholder job;
  `defaults.memory_floor_gb: 8` and `memory_cap_gb: 36` are laptop-sized.
- Mac mini (checked 2026-09-11 evening): branch `embeddings-ladder` at
  `dfa45ad` (6 behind), 21 GB free, `models/embeddings/` absent; the i7
  frame and cache are byte-identical to the laptop (brief § 4).
- No `xlstm` package is installed; the xLSTM arm is written in plain torch.
- Laptop training wall from stage 1 (MPS, 30 epochs, patience 3): token
  MLP 53–93 s per seed, full T1 122–226 s per seed.

---

## D1. This file

| # | Check | Pass condition |
|---|---|---|
| 1.1 | Written first | this file exists before any code change or training run for stage 2; each table precedes its Result block |
| 1.2 | Coverage | one table each for D2–D8 now; D9–D11 tables completed before those steps start, after the user's go |

---

## D2. Decision record

| # | Check | Pass condition |
|---|---|---|
| 2.1 | Seeds tonight | the user's decision (2026-09-11) is recorded: every configuration trains at **two seeds (7, 13)** on the first night as a **directional screen**; seeds 29, 42, 101 may be added later, and only to whole hypothesis families (candidate plus every matched control plus every `k` if `k` is selected), never to single arms; every validation and test number of the stage is screening evidence; the untouched cohort is opened exactly once, after candidates, `k`, controls and families are frozen (Astra round 1 MUST-FIX 1, 2) |
| 2.2 | No reuse | the user's decision: stage 1 `mlp`/`full` checkpoints are **not** reused; every arm trains fresh under the stage 2 config into `models/embeddings/seq_stage2/`; the training block equals stage 1's (d 128, 2 layers, 4 heads, batch 128, 30 epochs, lr 3e-4, patience 3, early stop on the full validation split, no aux, `--no-kit`, no `--score-test`) so protocols are comparable, and the report states these are separate runs, not replications |
| 2.3 | Cohort | the user's decision: cohort window 2026-04-17 → **2026-08-05**; source is the stat-generator zips, never `data/golden/`; **the user permitted id-only reads** ("you can read the files") of `data/forward_holdout/2026-06-01_2026-07-13/manifest.json` and of the golden odds/fixture ids for exclusion; the 137 forward fixtures and the 124 match-level golden fixtures are excluded from *scoring* (Fable's recommendation, plan rule "never used for inspected evaluation by any pipeline on the branch", Astra MUST-FIX 8); excluded matches remain chronological *context* for tracker state (Astra MUST-FIX 9); the cohort is frozen with hashes before the first mini training job launches |
| 2.4 | Backlog | `TODO.md` carries the user's item: the match-level golden set definition must move to matches dated **2026-08-06 onward** (fresh cricsheet and Polymarket captures under the daily protocol), because the 2026-04-17 → 2026-08-05 window is now the ball-level untouched cohort |
| 2.5 | Dependency test | the user's decision: the D7 design stands (laptop, CPU, relay-free wiring for every masked arm, one-layer fallback); it is a certification step (2c) between smoke and the mini, not a configuration |
| 2.6 | Block bootstrap | the user's decision: uncertainty resamples **tournament blocks** (I3 contract, `load_competition_clusters` over `data/t20s_json`), not matches; estimand (ii) resamples seeds and blocks jointly; a slice with fewer than 10 blocks is descriptive (invariant 7). The plan's "tournament blocks jointly with seeds" wording is honoured as written (Astra MUST-FIX 4) |
| 2.7 | Astra round 1 | prompt `<scratchpad>/astra/stage2_plan_prompt.md`, output `<scratchpad>/astra/stage2_plan_round1.md`, verdict `AGREE WITH CHANGES`; each MUST-FIX mapped to a check id (table below) |
| 2.8 | A15 deviation | early stopping on a validation date-prefix disjoint from the selection rows is **not adopted** for any arm (one protocol for all arms; the k-sweep tolerance is the registered guard); recorded as a deviation with Astra's SHOULD 12 caveat that this does not correct checkpoint-selection optimism, which is one reason all validation numbers are screening |

Astra round 1 MUST-FIX → check ids: 1, 2 → 2.1, 10.9; 3 → 10.4; 4 → 2.6, 10.2;
5 → 3.6, 3.7; 6 → 3.8; 7 → 7.2; 8 → 5.3; 9 → 5.5; 10 → 5.2, 5.7, 5.8;
11 → 3.12, 8.3. SHOULD 12 → 2.8; 13 → 8.6; 14 → 3.9, 3.10, 4.x.

### Result (D2)

Recorded 2026-09-11 (evening, in conversation with Fable).

| # | Result |
|---|---|
| 2.1 | Decision by the user: "I'm fine with going through all 15 configurations on two seeds today, and then we can see about the other seeds later." Recorded with the family rule above. |
| 2.2 | Decision by the user: "no reuse is good." |
| 2.3 | Decision by the user: "August 5th cutoff date is fine. You can read the files." Golden-fixture exclusion is Fable's recommendation under the plan's literal rule; the user was told it leaves roughly 210 matches and did not object. |
| 2.4 | Decision by the user: "add a backlog item or a to-do item that we need to update our definition of golden set to be from 5th August onwards." |
| 2.5 | Decision by the user: "decision 4 is fine." |
| 2.6 | Decision by the user: "they wanted to be split across tournaments so that we get a robust view instead of just sampling matches." |
| 2.7 | Verdict line: `VERDICT: AGREE WITH CHANGES`; 11 MUST-FIX, 3 SHOULD, 3 NOTE, all mapped above. |
| 2.8 | Recorded as a deviation. |
| 2.9 | **Orchestration handover.** Fable orchestrated through the D3/D4/D5 implementation reviews (2026-09-11 ≈23:15 IST). From the next step onward an Opus session orchestrates and executes per `docs/sequence_track/stage2_handoff_opus.md`, with Codex Astra (`gpt-6-astra`, medium reasoning per the user's instruction, read-only) as reviewer and design-question answerer at every registered gate; the acceptance file remains the contract; only the orchestrator commits and calls `research/log_verdict.py`. |

---

## Arm register (the sixteen configurations)

Every configuration is one entry in `experiments/configs/seq_stage2_v1.yaml`
with an `arm` name, its parameters, its access row, its matched control and
its queue position. Sign convention: candidate − reference, in log loss,
negative favourable.

| # | config id | `--arm` | k | history input | attention set for target i | wiring | pos emb | access (feat/hist/identity/prod) | reference | tests |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `mlp` | `mlp` | – | none | – | token MLP | – | y/n/n/n | – | control |
| 2 | `full` | `full` | – | innings-previous outcome | all j ≤ i | standard | yes | y/y/n/n | mlp | reference |
| 3 | `fixed_decay` | `fixed_decay` | – | innings-previous | all j ≤ i, additive bias −m_h·(i−j), ALiBi slopes m_h = 2^(−8h/H) | standard | **no** | y/y/n/n | mlp; control for fox | fixed decay |
| 4 | `fox` | `fox` | – | innings-previous | all j ≤ i, additive bias Σ_{t=j+1..i} log σ(g_h(e_t)), per-head scalar gate from the token input | standard | **no** | y/y/n/n | mlp; fixed_decay | learned forgetting |
| 5 | `aligned_hist` | `aligned_hist` | – | **participant-aligned**: last outcome of the same batter, and of the same bowler, in this innings (BOS if none), two separate embeddings, replacing the innings-previous outcome | all j ≤ i | standard | yes | y/y/y/n | mlp; full | aligned input alone |
| 6 | `aligned_hist_rf` | `aligned_hist_rf` | – | participant-aligned | all j ≤ i | **relay-free** | yes | y/y/y/n | aligned_hist | wiring effect; matched unrestricted endpoint (Astra MUST-FIX 6) |
| 7 | `recency_k30` | `recency` | 30 | innings-previous | W_k(i) = {j : i−k ≤ j ≤ i} | relay-free | yes | y/y/n/n | mlp; control for same_entity_k30 | recency component |
| 8–12 | `same_entity_k{0,6,12,30,unr}` | `same_entity` | 0/6/12/30/unr | participant-aligned | {j ∈ W_k(i) : batter_id[j]=batter_id[i] or bowler_id[j]=bowler_id[i]} ∪ {i} | relay-free | yes | y/y/y/n | mlp; recency at the same k; aligned_hist_rf at k=unr | ownership; k sweep |
| 13 | `lstm` | `lstm` | – | innings-previous | recurrent, state reset at innings start | 2-layer LSTM, hidden 128 | – | y/y/n/n | mlp | classic forget gate |
| 14 | `xlstm` | `xlstm` | – | innings-previous | recurrent | sLSTM block then mLSTM block, 128-d, plain torch | – | y/y/n/n | mlp | exponential gating |
| 15 | `residual_mlp` | `residual_mlp` | – | none | – | token MLP over log p_base | – | y/n/n/y | mlp | control for residual_t1 |
| 16 | `residual_t1` | `residual_t1` | – | innings-previous | all j ≤ i | standard | yes | y/y/n/y | residual_mlp (confirmatory); mlp (not qualifying) | sequence over production |

Definitions fixed here:

- **W_k(i)** is in delivery rows inclusive of extras; `unr` is the whole
  innings; `k = 0` is the target row alone. Every attention set includes
  the target itself, so no query has an empty set.
- **Relay-free wiring**: at every layer, keys and values are computed from
  the layer-zero token embedding e_j (feature projection + outcome
  embedding(s) + positional embedding, before any mixing); queries from
  the running residual stream. Hence h_i depends only on {e_j : j ∈
  attention set(i)}. Certified in D7.
- **Dependency set S(i)** for the D7 test: the attention set of i, plus
  the rows whose outcomes feed i's history input (row i−1 for
  innings-previous; the last same-batter row and the last same-bowler
  row for participant-aligned). Perturbing every row outside S(i) must
  leave i's logits unchanged.
- **fixed_decay and fox** are identical in every setting other than the
  bias (no positional embedding in both, QK-norm off in both, same width,
  depth, dropout, optimiser). The FoX gate is computed from the key token's
  layer-zero embedding, one scalar per head per token, cumulative log-gate
  added before softmax (Lin et al. 2025 form).
- **Recurrent arms**: hidden 128 matched to T1's d; state reset per
  innings (each innings is one sequence in the batch); full-innings
  backpropagation; the "equal tuning budget" is one setting (lr 3e-4,
  dropout 0.1), because T1's registered budget is one setting; parameter
  counts reported.
- **Residual arms**: p = softmax(log p_base + r_θ); p_base from the
  production ball model `models/xgb_i7_noweights_production` floored at
  1e-4 and renormalised; the last linear layer of r_θ zero-initialised;
  L2 shrinkage λ·mean(r_θ²) added to the loss with λ registered in the
  config (0.001); train-split base logits are leave-one-date-block-out
  refits (5 contiguous date blocks of the train split, production
  hyperparameters, same 114 features), validation/test/cohort base logits
  from the production booster itself (D4). Confirmatory contrast:
  residual_t1 − residual_mlp; also reported: each residual arm − mlp and
  the base-only log loss, as exploratory.
- **Queue order** (night 1): 1, 2, 3, 4, 5, 7, 11 (k=30), then 6, 8, 9,
  10, 12, then 13, 14, then 15, 16. Two seeds each: 32 runs.

---

## D3. Config, arm code, access tests (2a)

| # | Check | Pass condition |
|---|---|---|
| 3.1 | Config | `experiments/configs/seq_stage2_v1.yaml` registers every configuration in the arm register with its parameters, the frame (`data/xgb_data_i7`, role `ball_frame_i7`), cache role `stats_cache_i7`, seeds `[7, 13]` for night 1 with the family rule from D2.1 written in, the training block from D2.2, the statistics block (D10: contrasts, families, slices, tournament-block bootstrap, 2000 reps, seed 29, margins +0.002, tolerance 0.002 for k), the queue order, the command line per configuration, `forbidden_data`, `deviations` (A15 not adopted; two seeds on night 1), `known_asymmetries`, `known_limitations` (global prior not as-of; venue history not recency-weighted; test split read by the 2026-08 program) |
| 3.2 | Pin | `scripts/sequence_track/pin_stage2.py --write` / `--verify` in the `pin_stage1.py` mould: recomputes frame md5/rows/dates, `.feature_hash` key-by-key, cache md5, the 50 feature names and sha, the source closure sha256 of every stage 2 script, and refuses on drift; `--verify` prints `OK` |
| 3.3 | Arms exist | `transformer_t1.py --arm` accepts `fixed_decay`, `fox`, `aligned_hist`, `aligned_hist_rf`, `recency`, `same_entity`, `residual_mlp`, `residual_t1`; recurrent arms `lstm`, `xlstm` live in a separate module (`scripts/sequence_track/recurrent_arms.py`) selected through the same `--arm` flag; `--k` (int or `unr`) is required for `recency`/`same_entity` and refused elsewhere; `--base-logits-dir` is required for residual arms and refused elsewhere; the four stage 1 arms are byte-for-byte unchanged in behaviour (existing ablation tests pass) |
| 3.4 | Aligned inputs | for each row, the last same-batter and last same-bowler outcome within the innings are computed from `batter_id`/`bowler_id`/`ball_outcome` in parquet order, BOS where none; a unit test on a hand-built innings (batter change, bowler change, wicket, extras rows) asserts the exact expected vectors |
| 3.5 | Masks | unit tests assert the attention set for hand-built innings at k ∈ {0, 6, unr} for both `recency` and `same_entity` (including that the target is always in its own set, that extras rows count toward k, and that a bowler who returns after a gap is still visible) |
| 3.6 | Matched masked pair | `recency` and `same_entity` share the relay-free layer implementation, width, depth, dropout, positional embedding and optimiser; they differ only in the attention set and in the history input (innings-previous vs aligned, as registered); the report states the contrast is "ownership plus alignment beyond recency" and `same_entity_k − aligned_hist_rf` isolates the mask given aligned inputs (Astra MUST-FIX 5) |
| 3.7 | Access tests | per-arm invariance tests on a tiny synthetic frame: `mlp` invariant to all outcomes; `full`/`fixed_decay`/`fox` invariant to future rows (causality) and to `batter_id`/`bowler_id` permutation; `aligned_hist`/`aligned_hist_rf` change when the same batter's earlier outcome changes and are invariant to identity-free relabelling that preserves alignment; `recency_k` invariant to rows older than k under relay-free wiring; `same_entity_k` invariant to rows outside S(i); residual arms invariant to nothing new but reproduce `p_base` exactly at initialisation (zero-init head) |
| 3.8 | Unrestricted endpoint | `aligned_hist_rf` exists so that `same_entity_unr − aligned_hist_rf` compares two relay-free models differing only in the mask, and `aligned_hist_rf − aligned_hist` measures the wiring alone |
| 3.9 | FoX/fixed-decay parity | a test constructs both arms from the same seed and asserts identical parameter shapes except the gate parameters, no positional embedding in either, and that fixed_decay's bias equals the ALiBi formula; fox at gate logits → +∞ reproduces the vanilla no-pos-emb attention |
| 3.10 | Recurrent arms | `lstm`: `nn.LSTM(128, 128, num_layers=2, dropout=0.1)` over the token embedding; `xlstm`: sLSTM (exponential gating with stabiliser state) then mLSTM (matrix memory, exponential gating), both 128-d, written in plain torch with a unit test against a reference step-by-step implementation on a 3-token sequence; both causal by construction (a test perturbs future rows); parameter counts written to `metrics.json` |
| 3.11 | Trainer outputs | every run writes `metrics.json` with the full `training_contract` plus `arm_params` (k, wiring, λ, base-logits md5, parameter count), `model.pt`, `run_record.json`, and row-aligned validation probabilities (`--save-predictions` on for every configuration); `validation_ll` unrounded |
| 3.12 | Driver | `scripts/sequence_track/retrain_stage2.py` (copied from `retrain_i7.py`, not a rewrite): output root `models/embeddings/seq_stage2/runs/<config_id>/seed_<s>/`; reuse check extended to `arm_params` (k, wiring, λ, base-logits md5) and the config id; `--config-ids` selects a subset for the queue; `--seeds` may narrow but never widen the config's seed list; `summary.yaml` per config with full-precision per-seed validation LL, wall seconds and peak RSS (from `resource.getrusage` of the child) (Astra MUST-FIX 11) |
| 3.13 | Tests | new `scripts/tests/test_stage2_arms.py`, `test_stage2_masks.py`, `test_stage2_recurrent.py`, `test_retrain_stage2.py`, `test_pin_stage2.py`; the full artifact-free suite passes; count recorded |

### Result (D3) — interim, recorded at the Fable→Opus handover 2026-09-11 ≈23:30 IST

Implemented by three Opus subagents (transformer arms; recurrent arms;
config/driver/pin/queue), each reviewed by Fable. Files: `scripts/transformer_t1.py`
(+707/−51), `scripts/transformer_xr.py` (1 line), `scripts/sequence_track/recurrent_arms.py`
(343), `retrain_stage2.py` (1,160), `pin_stage2.py` (617),
`experiments/configs/seq_stage2_v1.yaml` (715 incl. provenance),
`research/sequence_track/queue.yaml` (177), `scripts/tests/test_run_queue.py`
(+26/−3), new tests `test_stage2_arms.py` (630), `test_stage2_masks.py` (185),
`test_stage2_recurrent.py` (276), `test_retrain_stage2.py` (461),
`test_pin_stage2.py` (300), `test_queue_stage2.py` (117).

| # | Result |
|---|---|
| 3.1 | PASS — sixteen configurations registered with params, access, history input, wiring, reference, tests, queue order and per-seed `command` lists; training block equals stage 1's; seeds [7, 13] with the D2.1 family rule quoted; statistics, cohort pointer, forbidden_data, deviations, asymmetries, limitations present |
| 3.2 | PASS — `pin_stage2.py --write` then `--verify` on the real tree (Fable re-ran): `pin_stage2: OK — experiments/configs/seq_stage2_v1.yaml matches every recomputed fact`; frame md5s train `fac0b7be…` / validation `32643631…` / test `64b58fec…` (hashed only), cache `671ac820…`, 50-feature sha `14f40e4a…`, base-logits digest `20286a8ffe8dc4c4915b20f1b908e141`, cohort frozen sha `32265943…` (210), sources 6 hashed, `ownership_dependency_test.py` and `stage2_stats.py` recorded absent |
| 3.3 | PASS — twelve stage 2 arms accepted (plus the stage 1 `no_attention`/`no_history`, a harmless superset); `--k` and `--base-logits-dir` required/refused as registered; the four stage 1 arms verified bitwise identical (logits, state-dict keys, parameters) against `git show HEAD:scripts/transformer_t1.py` |
| 3.4 | PASS — `aligned_history()` pure function; exact-vector test on a hand-built innings |
| 3.5 | PASS — `attention_mask()` tests at k ∈ {0, 6, unr} for both masked arms |
| 3.6 | PASS with a **pending redesign** (handoff § 3.1): keys/values of earlier rows must carry the row's own outcome so S(i) is exactly the registered set; until then the dependency set is wider than registered and D7 would fail |
| 3.7 | PASS on the current construction (78 tests); the 3.7 tests are to be updated with § 3.1 |
| 3.8 | PASS — `aligned_hist_rf` registered and implemented |
| 3.9 | PASS — fox/fixed_decay parameter-shape parity, no positional embedding in either, ALiBi formula test, fox gate → +∞ reproduces the unbiased attention; Fable switched the custom feed-forward to ReLU to match the stock layer |
| 3.10 | PASS, **corrected 2026-09-12** after Astra gate 1 MUST-FIX 1 found the mLSTM normalisation mathematically wrong: the stabilised cell carries shifted states `C' = exp(−m)·C`, `n' = exp(−m)·n`, so the denominator must be `max(|n'ᵀq|, exp(−m))`; the previous `clamp(|nᵀq|, min=1)` left an un-cancelled `exp(m)` whenever the constant branch was active, and the old reference test repeated the same error so it passed. Fixed, and the reference is now a genuinely **unstabilised float64** per-token loop (no `m_t`, gates applied directly, the paper's plain `max(|nᵀq|, 1)`): equality to <1e-9 at d=32 double and <1e-5 at d=128 float32, plus a large-gate test asserting the unstabilised reference goes non-finite while the module stays finite. Mutation-checked: reverting to `clamp(min=1)` fails both equality tests (max |Δ| 3.06e-04 at d=128). SHOULD 4's omitted mLSTM output gate was **restored** (in the block, after the multi-head norm, per the paper). 16 tests. Parameter counts at n_feats=7 lstm 266,886 / xlstm **367,670**; at the real 50-feature frame lstm 272,390 / xlstm **373,174** |
| 3.11 | PASS, **corrected 2026-09-12** after Astra gate 1 MUST-FIX 8 showed the recorded Result was false: the recurrent module's `cell` / `simplifications` block never reached `metrics.json`. Now `arm_params_block(..., extra=…)` merges the recurrent module's own keys (registered table keys stay authoritative) and both are verified present in the re-smoked `metrics.json` for `lstm` (`cell: "nn.LSTM(hidden=128, layers=2)"`) and `xlstm` (`cell` naming `mlstm_output_gate=True` and `mlstm_denominator=max(|n^T q|, exp(-m))`, plus six recorded simplifications). The nine table-derived fields are present on every arm, with `base_logits_md5_by_split` additionally on residual arms; `--save-predictions` writes `predictions_<split>.npz` (`probs`, `y`, `innings_id`) |
| 3.12 | PASS, **hardened 2026-09-12** after Astra gate 1 MUST-FIX 2, 3, 4 and plan item 11 (driver 1,160 → 1,883 lines; tests 40 → 87). (a) **Seed-independent training signature**: sha256 over eight named components in fixed order — `config_id`, `arm`, `arm_params` (now including `key_construction`), `training_block` (effective arch and optimiser, so an `--epochs` override enters it), `frame` (dir, version, whole `.feature_hash`, train and validation md5/rows/date-range), `stats_cache` md5, `base_logits` digest (None for the fourteen arms that never read it, so a rebuild cannot invalidate them) and `implementation` (`scripts/transformer_t1.py`, plus `recurrent_arms.py` for recurrent arms). Seed, out-root and machine are deliberately not inputs, so the two machines' seeds share one signature; the driver itself is excluded because it launches training rather than defining it. Written into every `run_record.json`, every `COMPLETE.json` and `summary.yaml`; `verify_checkpoint` refuses on mismatch naming the **first differing component**, a record with no signature is refused as unverifiable, and `build_summary` refuses to put two differently-signed rows in one table. (b) **Provenance comparison**: `provenance_check` in preflight compares live frame version, whole `.feature_hash`, train and validation md5/rows/date-range, cache md5 and role, and (only when a residual arm is selected) the base-logits digest and per-split npz md5s against the config's `provenance:` block; an absent block refuses naming `pin_stage2.py --write`. All seven components compare clean on the committed config. `cohort` is in the provenance skip set with the recorded reason that the mini has no cohort artifacts and no training run reads one, so **no cohort file is ever opened** and the mini path verifies. (c) **Completion**: all four artefacts must exist, be non-empty, parse (JSON to an object, npz exposing `probs`); a run missing predictions is no longer accepted and a run interrupted before `run_record.json` is incomplete and **retried**, with artefacts cleared before relaunch; `COMPLETE.json` (signature, seed, config id, per-artefact size and md5, UTC stamp) is written to a same-directory temp file and `os.replace`d. A `COMPLETE.json` whose signature or seed disagrees refuses; its absence does not, since older runs predate it. Contract disagreement on a complete run still refuses, `--force` only. (d) **RSS**: `resource` dropped; `Popen` plus a `ChildRssSampler` polling the child's own process tree every 0.5 s via the already-pinned `psutil==6.1.0` (a `ps` tree walk is the fallback, and `rss_method` says which ran), recorded as `peak_rss_bytes_sampled` with interval, sample count and `peak_rss_bytes_valid`; if no sample lands no number is invented. Legacy `RUSAGE_CHILDREN`-delta values are **read as invalid, never rewritten** — reported as `peak_rss_bytes: None`, `peak_rss_bytes_valid: false`, `rss_method: rusage_children_delta`, the old observation preserved under `peak_rss_bytes_observed_invalid`, and a test asserts the legacy file's bytes are unchanged after reading. The consolidated flow is covered end to end: a two-seed tree written separately, each with its own one-seed summary, is verified and reused by one no-`--force` `--seeds 7,13` invocation which rewrites the summary listing both seeds at full precision; a seed whose signature claims different implementation code is refused naming `implementation`; a half-rsynced seed is retried while its sibling is reused |
| 3.13 | **1 failed, 1433 passed, 54 deselected** (`uv run --no-sync pytest -q --strict-markers -m "not needs_artifacts"`, 79.78 s, Fable re-ran). The failure is `test_pin_stage1.py::test_source_closure_is_hashed_in_full` (stage 1 closure hashes the live trainer); disposition in handoff § 3.2 |

**Addendum (D3, 2026-09-11 ≈00:20 IST, still under Fable):** the handoff § 3.1
redesign landed — relay-free arms read earlier rows as (state, own outcome)
pairs via a concatenated key/value set `[x_kv rows j<i, x_self at i]`
(`own_out_emb = nn.Embedding(6, d)` created last so existing parameter RNG
order is preserved); `arm_params.key_construction` ∈ {own_outcome,
shifted_history, null}; `attention_set`, `history_source_rows`,
`dependency_set` are shared functions in `transformer_t1.py`; S(i) tests
updated; y_i-invariance test added for every relay-free arm; the
`relay_free_keys_carry_own_outcome` asymmetry recorded in the config (all
`known_asymmetries` entries now `{id, text}`). § 3.2 landed: erratum
`docs/sequence_track/stage1_erratum_trainer_closure.md` (pinned md5
`2fa53441…` equals the trainer at `2c7ea6f`; live `9d7fc79e…`), and
`test_source_closure_is_hashed_in_full` accepts the stage 1 commit's hash.
`pin_stage2 --verify`: OK. Suites: artifact-free **1458 passed, 54
deselected**; full **1 failed, 1506 passed, 5 skipped** — the failure is
`test_pin_stage1.py::test_verify_passes_on_the_committed_config`
(`pin_stage1.verify()` recomputes the live trainer md5); **open item for
Astra at gate 1**: give `pin_stage1.verify()` the same stage-1-commit
fallback, or record an accepted exception; the stage 1 config stays frozen.

D8.2 (queue) is already satisfied: 16 jobs, mini order, `OMP_NUM_THREADS=4`,
floor 3 GB / cap 12 GB, `expected_hours` placeholder 0.5 to be overwritten
from D6.3; `--machine mini --dry-run` lists 16 with `decision=run`,
`--machine laptop` selects none.

---

## D4. Residual base logits (2a, laptop)

| # | Check | Pass condition |
|---|---|---|
| 4.1 | Production logits | `scripts/sequence_track/build_base_logits.py` scores validation and test (and later the cohort) with `models/xgb_i7_noweights_production` through the repository's own load path; class order mapped to the T1 class order `{0,1,2,4,6,wicket}`; floored 1e-4, renormalised; written as `npz` under `models/embeddings/seq_stage2/base_logits/<split>.npz` with row count, parquet md5 and booster md5 in a sidecar json |
| 4.2 | Train OOF | the train split is cut into 5 contiguous date blocks (boundaries recorded); for each block a booster is refit on the other four with the production hyperparameters (read from the production training contract/config, recorded), on the same 114 feature columns; the block's rows are scored by that refit only; a test on a toy frame asserts no row is scored by a booster that saw it |
| 4.3 | Sanity | validation log loss of `p_base` from the production booster is recorded (it is the production model's own validation LL, not a stage 2 number) and the OOF train log loss is recorded; neither is used for any selection |
| 4.4 | Wall time | per-refit wall seconds recorded |

### Result (D4)

Recorded 2026-09-11 22:35 IST. `scripts/sequence_track/build_base_logits.py`
(553 lines), `scripts/tests/test_build_base_logits.py` (266); Opus subagent,
reviewed by Fable. `uv run --no-sync pytest -q scripts/tests/test_build_base_logits.py` → `11 passed in 0.66s`.

| # | Result |
|---|---|
| 4.1 | PASS — load path is the repo's own (`feature_columns_i7.txt` → `calibration._apply_encoders_to_df` → `XGBClassifier.predict_proba`), roles `ball_model_prod` / `ball_frame_i7`; class permutation derived from `xgboost_v2` remap and `embeddings_e1.CLASS_MAPPING`, comes out identity `[0,1,2,3,4,5]`, applied generically; `validation.npz` (124,292 rows, parquet md5 `326436317310adadabe0175825e57d1b`), `test.npz` (186,667, `64b58fec51509567580c9122f2b5183f`); booster md5 `7ee1e1809917f45be7e726b3ea4a8a6c`; floor 1e-4 never binds (min real probability 0.0042–0.0047) |
| 4.2 | PASS — 5 contiguous whole-date blocks: 2005-02-17→2016-01-02 (378,069 rows), 2016-01-03→2019-03-01 (372,878), 2019-03-02→2021-10-08 (375,775), 2021-10-09→2023-06-01 (375,537), 2023-06-02→2024-12-30 (374,712); each refit on the other four with `max_depth 10`, `learning_rate 0.24036372383981375`, `n_estimators 25` (= production `best_iteration` 24 + 1; `_get_iteration_range` confirms production predicts with 25 rounds), no `sample_weight`, production encoders reused not refit; hyperparameters from the pickled booster cross-checked against `experiments/configs/xgb_i7_venue_identity.yaml`; toy-frame no-leak test passes. Deviation recorded: refits run a fixed 25 rounds with no early stopping and no eval set |
| 4.3 | Recorded (not stage 2 numbers): `production_model_validation_ll_not_a_stage2_number` = 1.4334372099035435 (booster's own `best_score` 1.4334372096); test LL 1.4252763480455588 (D16 reported 1.4253); `oof_train_log_loss` = 1.404018522500772 |
| 4.4 | Refit wall seconds 12.0 / 12.1 / 12.9 / 12.7 / 12.6 (8 threads); validation and test scoring ≈1.6 s each |

---

## D5. Untouched cohort (2g, laptop; frozen before the first mini job)

| # | Check | Pass condition |
|---|---|---|
| 5.1 | Extraction | `scripts/sequence_track/build_cohort_stage2.py` reads the stat-generator zips, keeps `info.gender == male`, `match_type == T20`, `balls_per_over == 6`, first date in 2026-04-17 → 2026-08-05, de-duplicates by cricsheet file stem across zips (byte-identical duplicates recorded; differing duplicates refuse), writes the JSONs to `models/embeddings/seq_stage2/cohort/raw_json/` with a sha256 manifest; count equals **471** |
| 5.2 | Exclusions | an eligibility audit CSV lists every one of the 471 with `eligible` or the reason: `forward_fixture` (ids read from `data/forward_holdout/2026-06-01_2026-07-13/manifest.json`, ids only, 137 expected), `golden_fixture` (ids read from `data/golden/betting_odds_golden_v2.json`, ids only, 124 expected), `in_repo_corpus` (id present in `data/t20s_json`, expected 0); the script reads no other field from either closed file and asserts that by reading through a helper that returns only the id list; counts per reason and the surviving count recorded |
| 5.3 | Eligibility rule | the surviving set contains no match used for training, tuning or inspected evaluation by any pipeline on the branch: `data/t20s_json`, the golden and forward fixture ids, and the Hundred (`balls_per_over` 5) are the only known consumers of post-2026-04-16 cricsheet on the branch; `grep` of `data/`, `scripts/daily/`, `daily/` for other consumers recorded |
| 5.4 | State | a sidecar stats cache is built under `models/embeddings/seq_stage2/cohort/state/` from `data/t20s_json` **plus all 471 raw matches** as chronological context, with the same-day ordering contract, `venue_aliases_v1`, and the global/phase priors **frozen from the production i7 cache `_meta`**, never recomputed over post-2026-04-16 data; the builder does not import `build_forward_state.py` (which reads the sealed holdout) |
| 5.5 | Context vs scoring | excluded matches (forward, golden) are present in the state as context and absent from the scored rows; a check asserts that the scored parquet's match ids equal the eligible set exactly and that tracker state for an eligible match dated after an excluded match reflects that excluded match (one hand-verified case) |
| 5.6 | Parity | the cohort rows are materialised through `materialize_features.py` with the i7 contract into `cohort/cricket_data_i7_cohort.parquet`; the `.feature_hash` written equals the frame's (`c520a3ba08ae`, `n_features` 114, same semantics, alias version and sha) and, as a pipeline parity check, 20 matches from the i7 **test** split rebuilt through the same sidecar path reproduce the frame's rows bit-for-bit on every feature column |
| 5.7 | Freeze | `cohort/FROZEN.json` records: eligible ids (sorted), row count, parquet sha256, sidecar cache md5, raw manifest sha256, builder source sha256, the two id-only reads with the user's permission quoted from D2.3, and a UTC timestamp; the timestamp precedes the first mini job's start stamp in D8 |
| 5.8 | Access | no stage 2 code reads any other file under `data/golden/` or `data/forward_holdout/`; `grep -rn "data/golden\|forward_holdout" scripts/sequence_track/*stage2*` shows only the two id helpers |
| 5.9 | Scored once | the cohort parquet is not loaded by any trainer, smoke or selection step; it is read only in D10 after the family freeze |

### Result (D5)

Recorded 2026-09-11 (frozen 17:23:20 UTC = 22:53 IST). `scripts/sequence_track/build_cohort_stage2.py`
(1,132 lines), `scripts/tests/test_build_cohort_stage2.py` (341); Opus
subagent, reviewed by Fable. `uv run --no-sync pytest -q scripts/tests/test_build_cohort_stage2.py` → `25 passed in 0.27s` (Fable re-ran).

| # | Result |
|---|---|
| 5.1 | PASS — 471 matches from 12 zips (11,340 members scanned, 0 differing duplicates); `raw_manifest.json` sha256 `7f3ba7307a66c6bfe886917cae5a1f5f7bb5d68b73f8047d42331cd9d3d0920d` |
| 5.2 | PASS — `eligibility_audit.csv` (sha256 `52c99df5200e…`): eligible **210**, forward_fixture 137 (expected 137), golden_fixture 124 (expected 124), in_repo_corpus 0, hundred 0; forward and golden id sets disjoint; helpers return ids only (fields `matches[].match_id`, `matches[].cricsheet_id`), sentinel test proves nothing else leaks |
| 5.3 | PASS, **strengthened 2026-09-12** after Astra gate 1 MUST-FIX 6 held that a grep for current references cannot establish historical non-consumption (the parked export *was* `data/t20s_json` from 2026-08-05 to 2026-09-10). `docs/sequence_track/stage2_cohort_consumer_audit.md` closes it on provenance, opening no closed directory: the i7 stats cache was built **2026-07-26 03:13** and the frame **03:21**, ten days before the 2026-08-05 export existed; the cache's `_meta` records `source_json_file_count` **11264** / `source_match_count` **9519** (the frozen corpus, not the export's 10,189 files) and `source_json_mtime_max` **2026-04-19T04:14:12**. So the frame, the cache, the cohort's frozen priors and the cohort rows cannot carry the export. The one recorded interaction between a pipeline and it was the sealed-set preflight **failing closed**, which is a refusal, not a consumption. `notes_consumers.md`'s grep stands as far as it goes |
| 5.4 | PASS — sidecar cache built in 446 s over a merged symlink corpus (`data/t20s_json` + all 471), 9,990 matches; priors frozen via `build_stats_cache.freeze_priors_from_sqlite` from `models/player_stats_cache_i7.sqlite`: all 24 `prior_*` `_meta` keys byte-equal to production (`prior_p0 = 0.3040165819`), `prior_contract = frozen_external_sqlite_v1`; ordering `date_then_match_id_lexicographic_v1`, `venue_aliases_v1` (94), schema 4; `build_forward_state.py` never imported |
| 5.5 | PASS — scored ids == 210 eligible ids exactly; hand case (`notes_context_case.md`): player `6b16f8ba`, excluded forward fixture `1512839` (2026-06-28) → eligible `1512850` (2026-07-03): as-of batting 720/564 → 770/607, delta +50 runs / +43 balls equals his batting in the excluded match |
| 5.6 | PASS — materialise 173 s; split rows train 1,876,971 / validation 124,292 / test 186,667 identical to the frame; golden_test 108,356 (471) → 48,406 rows for the 210 eligible; `.feature_hash` byte-identical to the frame's (`c520a3ba08ae`); parity 20 test matches = 4,820 rows × 156 columns 0 mismatches, and the whole test split 822 matches / 186,667 rows × 156 columns 0 mismatches; two extra non-feature I9 columns (`batter_elo_exposure`, `bowler_elo_exposure`) present in the rebuild, asserted outside the registry |
| 5.7 | PASS — `FROZEN.json`: `frozen_at_utc 2026-09-11T17:23:20Z`, 210 ids, `row_count 48406`, parquet sha256 `75daedde37dfab475c97ac495f638da404229858bb6c289feab13f53f3487a5b`, sidecar cache md5 `8308b324f640c0babc70129a07dfe88a`, builder sha256 `c935821f…`, both permitted reads recorded with the user's quote |
| 5.8 | PASS — grep shows only the access-rule docstring and the forward-manifest path constant; AST test restricts the openers to the two helpers |
| 5.9 | Intact — nothing has loaded the cohort parquet beyond the id/row checks |

Deviations recorded: merged symlink corpus (the materialiser fail-closes on
a multi-directory corpus); `feature_hash_info` passed explicitly so the
marker matches the frame; `materialized_full/` train and validation parquets
deleted after the parity check (750 MB), test and golden_test retained.

---

## D6. Smoke (2b, laptop)

| # | Check | Pass condition |
|---|---|---|
| 6.1 | Every configuration | all 16 configurations train seed 7 for `--epochs 1` on the laptop through `retrain_stage2.py --epochs 1 --out-root models/embeddings/seq_stage2/smoke/`; exit 0; `metrics.json` with contract and `arm_params`, `model.pt`, row-aligned validation probabilities present |
| 6.2 | No numbers | no smoke log loss is read into this file or any report |
| 6.3 | Wall time | laptop wall seconds per configuration recorded, as the basis for the mini budget |
| 6.4 | Tests | the D3.13 suite passes on the same tree |

### Result (D6)

Recorded 2026-09-12 ~00:45 IST. Six configurations were smoked under Fable
(2026-09-11 ~23:20 IST) and the remaining ten by the Opus orchestrator in one
driver invocation (`--config-ids mlp,fixed_decay,fox,aligned_hist_rf,same_entity_k6,same_entity_k12,lstm,xlstm,residual_mlp,residual_t1 --seeds 7 --epochs 1 --out-root models/embeddings/seq_stage2/smoke`, exit 0).

| # | Result |
|---|---|
| 6.1 | PASS — all **sixteen** configurations trained seed 7 for one epoch, exit 0, and each `smoke/<id>/seed_7/` holds all four artefacts (`metrics.json`, `model.pt`, `run_record.json`, `predictions_validation.npz`) — 4/4 on every configuration |
| 6.2 | PASS — only wall seconds, exit codes and artefact counts were read; no smoke log loss appears in this file, any report, or any commit message |
| 6.3 | Wall seconds per configuration (load + one epoch, laptop, MPS, seed 7), verbatim from each `smoke/<id>/summary.yaml`: `mlp` 6.5 · `lstm` 6.9 → **6.0 on re-smoke** · `residual_mlp` 7.4 · `full` 8.2 · `aligned_hist` 8.7 · `fixed_decay` 10.0 · `same_entity_k0` 11.0 · `same_entity_k30` 11.0 · `fox` 11.1 · `same_entity_unr` 11.1 · `recency_k30` 11.2 · `residual_t1` 11.4 · `aligned_hist_rf` 13.9 · `same_entity_k6` 14.2 · `same_entity_k12` 14.4 · `xlstm` 62.7 → **73.2 on re-smoke**. `lstm` and `xlstm` were re-smoked 2026-09-12 after the MUST-FIX 1 / SHOULD 4 changes to the recurrent cell; the xLSTM increase is the restored output gate's extra per-step linear and sigmoid. One earlier xLSTM re-smoke on a contended machine read 289.8 s and is discarded as a contention artifact, not a timing of record (a standalone trainer invocation on the idle machine read 61 s). D8.6's budget recalibration uses **73.2** |
| 6.4 | PASS — `uv run --no-sync pytest -q --strict-markers -m "not needs_artifacts"` on the same tree: **1458 passed, 54 deselected, 51 warnings in 81.50s**, zero failures |

**Budget note for D8 (not a check):** `xlstm` is ~9× `mlp` per epoch, so it is
the night's long pole; at 30 epochs, ×2 seeds and a 2–3× mini factor it is
plausibly 2–3 hours by itself. `expected_hours` per job is set from these
numbers in D8.2 and the total is put to Astra at gate 1 (question 7).

---

## D7. Ownership dependency test (2c, laptop, CPU)

| # | Check | Pass condition |
|---|---|---|
| 7.1 | Script | `scripts/sequence_track/ownership_dependency_test.py` loads a checkpoint (the D6 smoke checkpoints), samples 2,000 target rows from validation innings with seed 29, and for each target replaces the 50 features of every row outside S(i) with N(0, 9) noise **and** replaces the raw outcomes of every row outside S(i) with random classes, rebuilding the innings-previous and aligned history inputs from the perturbed outcomes; reports max and 99th-percentile |Δ logits| over targets |
| 7.2 | Raw outcomes | the perturbation acts on raw outcomes, not on the derived inputs, so a leak through the aligned input path is detectable (Astra MUST-FIX 7) |
| 7.3 | Pass rule | `same_entity_k0`, `same_entity_k30`, `same_entity_unr`, `recency_k30`: max |Δ| ≤ 1e-6 (expected exactly 0 under −∞ masking); observed max recorded per arm |
| 7.4 | Positive controls | `full` and `aligned_hist` (standard wiring) under the same perturbation show max |Δ| > 1e-3, proving the test can see relay |
| 7.5 | Failure rule | if any masked arm fails 7.3, that arm is rebuilt as one layer, re-smoked and re-tested; no result from it is read before it passes; `k = 0` is called "ownership" only after 7.3 passes |
| 7.6 | Record | per-arm table (n targets, max, p99, pass) pasted here verbatim from the script's json |

### Result (D7)

Recorded 2026-09-11. `scripts/sequence_track/ownership_dependency_test.py` (324 lines), `scripts/tests/test_ownership_dependency_test.py` (198); run on the D6 smoke checkpoints, CPU, 124,292 validation rows / 1,088 innings, 2,000 targets, seed 29, ≈3 s per arm; JSONs in `models/embeddings/seq_stage2/dependency/`. Verbatim:

```
same_entity k=0 (masked arm): n=2000 max|d|=0.000e+00 p99=0.000e+00 n>1e-6=0 PASS
same_entity k=30 (masked arm): n=2000 max|d|=0.000e+00 p99=0.000e+00 n>1e-6=0 PASS
same_entity k=unr (masked arm): n=2000 max|d|=0.000e+00 p99=0.000e+00 n>1e-6=0 PASS
recency k=30 (masked arm): n=2000 max|d|=0.000e+00 p99=0.000e+00 n>1e-6=0 PASS
full k=None (positive control vs S(i) of recency k=30): n=2000 max|d|=7.517e-01 p99=5.774e-01 n>1e-6=1449 SEES RELAY
aligned_hist k=None (positive control vs S(i) of same_entity k=30): n=2000 max|d|=9.217e-01 p99=7.411e-01 n>1e-6=1867 SEES RELAY
```

| # | Result |
|---|---|
| 7.1 | PASS — features N(0, 9) and raw outcomes randomised outside S(i), derived inputs rebuilt, max and p99 reported |
| 7.2 | PASS — perturbation acts on raw outcomes; S(i) from the shared `dependency_set` |
| 7.3 | PASS — all four masked arms exactly 0 |
| 7.4 | PASS. Astra gate 1 ruled the interpretation faithful but "proves relay" too strong, and supplied the wording adopted verbatim: **"PASS — using identical sampled targets and perturbations outside the matched masked arm's S(i), full versus recency_k30 gives max |Δ|=0.7517492175102234 and aligned_hist versus same_entity_k30 gives 0.9216543436050415, both >1e-3. These controls establish sensitivity to excluded-past information, not specifically multi-layer relay. Relay-free certificates always use their own S(i); overrides are refused."** |
| 7.5 | Not triggered |
| 7.6 | Pasted above |

`k = 0` may be called "ownership" (subject to Astra's reading of 7.4). These are one-epoch smoke checkpoints; the certification is structural (masking), so it holds for any trained weights of the same architecture; the D8 checkpoints are re-certified in D10 as a cheap re-run.

---

## D8. Overnight training, split by seed across two machines (2d)

**Amendment recorded 2026-09-12.** The table as first written required
mini-only training. Codex Astra was asked the machine question in a dedicated
round (prompt `docs/sequence_track/astra/machine_split_prompt.md`, output
`<scratchpad>/astra/machine_split_round1.md`, verdict line
`RECOMMENDATION: B — Pair every arm within seed; isolate markers and rebuild
summaries.`) and the user adopted it. Mini-only projected **7.3–11.0 h**
against **3.7–5.5 h** on the mini under the split. The plan permits a
per-stage division (`docs/SEQUENCE_TRACK_PLAN.md` § Bookkeeping, "The exact
division is decided per stage when work starts").

**The split:** every configuration's **seed 7 on the laptop**, every
configuration's **seed 13 on the Mac mini**. Every registered contrast is then
within-machine at each seed; machine is confounded with **seed**, not with
arm, so an additive machine effect cancels inside every contrast. Astra:
"B leaves no registered contrast machine-confounded, including the k
comparisons, provided every compared configuration has both assigned seeds."
Splitting by *arm* was rejected as confounding machine with arm. Arm-by-machine
interaction remains inseparable from seed variation in this design and **no
machine term is to be fitted**; the report must say so.

| # | Check | Pass condition |
|---|---|---|
| 8.1 | Sync | the stage 2 commit is pushed to the mini and checked out there; the mini's HEAD equals the laptop's committed HEAD; `models/embeddings/seq_stage2/base_logits/` and nothing else under `models/` is rsynced (`--stats`); frame md5s, cache md5 and `.feature_hash` re-verified on the mini against the pin |
| 8.2 | Queue files | two registered queues, `research/sequence_track/queue_laptop.yaml` (16 jobs, `machine: laptop`, ids `<config_id>-s7`, `--seeds 7`) and `queue_mini.yaml` (16 jobs, `machine: mini`, ids `<config_id>-s13`, `--seeds 13`), both naming the unchanged `experiments/configs/seq_stage2_v1.yaml`; `output_dir` **seed-specific** (`runs/<config_id>/seed_<seed>`) because the runner keys COMPLETE on the config sha alone and a shared marker dir would let seed 7 suppress seed 13; separate STOP files; `expected_hours` from Astra's rule `ceil_to_0.05((30 × smoke_seconds × F + 120)/3600)` with F = 3 (mini) and F = 1.5 (laptop) — **not** the original "smoke × 3 × seeds", which omitted the epoch count and would have made every timeout ~20× too small |
| 8.3 | Identity | a seed-job COMPLETE is never evidence that a configuration is complete; the driver's reuse check plus its seed-independent training signature (D3.12, Astra MUST-FIX 2) is what admits a run into the table; seed assignment per job is immutable |
| 8.4 | Launch | both dry runs list 16 jobs with `decision=run` on their own machine and none on the other; the laptop's training runs from a **separate git worktree pinned to the commit** so editing the main checkout cannot change training sources mid-queue (Astra MUST-DO 6); `run_queue.sh` calls `caffeinate` itself; start stamps recorded and confirmed after the cohort's `frozen_at_utc` `2026-09-11T17:23:20Z` |
| 8.5a | Smoke records are not reusable | the sixteen D6 smoke `run_record.json` files predate the training signature, so a rerun over the smoke out-root now **refuses** rather than reusing; intended, since their code identity was never recorded. The night starts fresh under `runs/`, and any smoke wall time or RSS quoted anywhere must read `peak_rss_bytes_valid` before quoting a number |
| 8.5 | Resources | mini `memory_floor_gb: 3`, `memory_cap_gb: 12`, `OMP_NUM_THREADS=4`; laptop `memory_floor_gb: 12`, `memory_cap_gb: 24`, thread caps 2 for `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`; `poll_seconds: 30` on both; one job at a time per machine; no simulator or heavy pytest concurrent with training. Effective caps recorded. Note: one retry means a hung job can consume 4 × `expected_hours`, and the memory floor is checked only before launch |
| 8.6 | Budget | the first completed job's wall time on each machine is compared with its budget and the remainder recalibrated **from timing only**; the xLSTM budget is recomputed from its corrected smoke (Astra MUST-FIX 1 changed the cell); if a night cannot finish, the report names the configurations that did not run |
| 8.7 | Consolidation | seed directories are rsynced into one tree on the laptop (`--stats`, never while a run is being written, and only after both machines' writers are confirmed dead), hashes verified; the two one-seed `summary.yaml` files **collide and rsync does not merge them**, so after all 32 seed artefacts exist the driver is invoked as **`--consolidate --seeds 7,13`** (no `--force`), which verifies every run and rewrites each summary from the completed ones. **Amended 2026-09-12 after Astra round 2 new MUST-FIX 3:** a plain `--seeds 7,13` would clear and retrain an incomplete seed **on the invoking machine**, so a half-transferred seed 13 would train on the laptop and silently break the machine assignment the design rests on. `--consolidate` refuses `--force` and `--dry-run` as contradictory, never unlinks an artefact, has an empty training loop by construction, and on any incomplete or unverifiable run names each one with its path and reason and tells the operator to re-queue it on the machine its seed is assigned to. Completeness requires all four artefacts, `model.pt` loading as a state dict whose parameter names match the arm's wiring, and the `COMPLETE.json` size-and-md5 manifest still matching on reuse |
| 8.8 | Provenance | **implemented 2026-09-12** (Astra round 2 new MUST-FIX 4): `machine_provenance()` captures, at run time, hostname, machine label and its source, chip and how it was read, OS and release, platform, python and torch versions, MPS availability and build, the device requested and the device actually used (read back from the run's own `metrics.json`), **all four** thread caps with unset recorded as null, the repo root, whether that root is a git worktree, and the symlink/resolved status of the frame directory, stats cache, `models/embeddings` and the out-root. Written into `run_record.json`, `COMPLETE.json` and `summary.yaml`, and shown per seed by `print_table`. It sits **outside** the training signature, asserted by a test. Both queue command lines pass `--machine laptop` / `--machine mini` explicitly so the label is registered rather than hostname-derived |
| 8.9 | Partial-night readability | if the night is cut short, a reported contrast requires **both** seeds for every member: `mlp` anchors every arm-minus-MLP result; `fixed_decay` + `fox` for learned forgetting; `recency_k30` + `same_entity_k30` for ownership-plus-alignment; `full` + `aligned_hist` + `aligned_hist_rf` + `same_entity_unr` for input, wiring and unrestricted-mask; **all five `same_entity` k configurations** before any k selection (a missing k cannot justify "keep 30 because none beat it"); `residual_mlp` + `residual_t1` together. Only k = 30 has a registered recency control, so a selected k other than 30 does not acquire one by borrowing `recency_k30` |


### Result (D8) — launched 2026-09-12

| # | Result |
|---|---|
| 8.1 | PASS — stage 2 committed at **`972cdf3`**; pushed to the mini (which needed `receive.denyCurrentBranch=updateInstead`, since `embeddings-ladder` is checked out there and its tree was clean apart from an untracked `daily/`); both the mini and the laptop training worktree verified at `972cdf30a0ac3fbffe4d1a49cf328381358c7c49`. Base logits rsynced (6 files, 52,525,270 B). Mini identity re-verified: train `fac0b7bededf0aae1989d654e3eba9e8`, validation `326436317310adadabe0175825e57d1b`, cache `671ac8200b275fa3d11d848e609f5132`, 21 GiB free |
| 8.2–8.3 | PASS — both dry runs list **16 jobs, all `decision=run`**, and neither selects the other machine's. Laptop: floor 12 GiB, cap 24 GiB, poll 30 s, stop `STOP_laptop`, available 20.15 GiB at launch. Mini: floor 3 GiB, cap 12 GiB, poll 30 s, stop `STOP_mini`, available 8.51 GiB at launch |
| 8.4 | PASS — launched **2026-09-12T01:31+0530** (laptop, from the worktree `/Users/aryamangupta/CricML/MP_train_s7`, log `/tmp/stage2_laptop_20260912_0131.log`) and **2026-09-11T16:01−0400, the same minute** (mini, log `/tmp/stage2_mini_20260911_1601.log`, pid 15558). Both after the cohort's `frozen_at_utc` `2026-09-11T17:23:20Z`. First job on both is `mlp`; the laptop reached epoch 14 at about 2 s/epoch while the mini was at epoch 0 at about 7 s/epoch, so the registered F = 3 mini factor is holding. The mini printed 272,006 parameters, the 50-feature token MLP as expected |
| 8.5 | PASS as registered; **operational note**: `uv` is not on the PATH for a non-interactive `ssh` to the mini (`run_queue.sh: line 105: uv: command not found`), which failed queue validation, launched nothing, and still exited 0 — exactly the silent no-op class Astra warned about. The mini is launched with `export PATH="$HOME/.local/bin:$PATH"` through a copied script; recorded in the runbook |
| 8.6, 8.7, 8.8, 8.9 | pending the morning |

## D9. k sweep (2e)

Table written 2026-09-12 before the step. Content supplied by Codex Astra
(prompt `docs/sequence_track/astra/d9d10_prompt.md`, output
`<scratchpad>/astra/d9d10_round1.md`) under the user's autonomy grant of
2026-09-12, which directs that where this contract said "after the user's go"
the orchestrator applies a deterministic registered rule itself and otherwise
asks Astra. The k rule is deterministic, so it is applied without asking.

| # | Check | Pass condition |
|---|---|---|
| 9.1 | Complete sweep | all five configurations `same_entity_k0`, `same_entity_k6`, `same_entity_k12`, `same_entity_k30`, `same_entity_unr` have admitted runs at both seeds `[7, 13]`, satisfying D8.3 and D8.7; otherwise selection is `BLOCKED_INCOMPLETE`, with no default-to-30 decision |
| 9.2 | Numbers of record | reads full-precision per-seed validation LL from `models/embeddings/seq_stage2/runs/<config_id>/summary.yaml`, after D8.7 consolidation; asserts exactly the two registered seeds, finite values and matching run provenance; records source paths and hashes; never substitutes rounded report values, smoke LL, checkpoint re-scores or LL reconstructed from saved probabilities |
| 9.3 | Selection rule | computes each k's arithmetic mean over the two recorded seed LLs; let `best` minimise that mean over **all five** k values, including `unr`; chooses `best` only when `mean_LL(30) − mean_LL(best) > 0.002`, otherwise chooses 30; equality at 0.002 keeps 30; exact ties between qualifying minima use registered sweep order `[0, 6, 12, 30, unr]`, recorded before selection |
| 9.4 | Validation only | selection reads only admitted validation summaries and provenance; no test predictions, test rows, cohort rows or cohort base logits are loaded; test and cohort results cannot change k |
| 9.5 | Seed spread | table reports both seed LLs, mean, empirical min–max and range for every k; additionally reports paired `LL(k,s) − LL(30,s)` for each seed, their mean, range and favourable-direction count out of 2; these ranges are empirical spreads, not confidence intervals |
| 9.6 | Interpretation guard | labels the entire sweep **two-seed directional screen**; 0.002 is a selection tolerance, not a significance threshold or demonstrated two-seed resolution; flags any non-default selection whose paired differences change sign, whose paired range is ≥0.002, or whose paired range is ≥the absolute mean difference; reports the flag beside the selection without changing the registered rule; never calls the chosen k reliably optimal |
| 9.7 | Selection record and pin | writes a machine-readable selection record containing all ten source LLs, computed means and paired spreads, tolerance, tie rule, selected config id/k, seed list, source hashes and UTC timestamp; records the chosen k and selection-record hash in the config and analysis pin; verification refuses drift; the two-seed selection remains explicitly provisional pending any registered whole-family seed extension |
| 9.8 | Other k arms and controls | retains all five k arms in the validation report, naming the other four "not selected"; does not suppress adverse or inconclusive results or substitute another k after gates are read; only k=30 has the registered `recency_k30` comparison, while `same_entity_unr − aligned_hist_rf` is the registered unrestricted-mask comparison; k=0/6/12 retain their registered MLP primary references; no k borrows a different window's recency control; any new matched control requires an explicit registration amendment and complete paired seeds before its analysis |

---

## D10. Statistics, gates, mechanism contrasts, report (2f)

Table written 2026-09-12 before the step, content supplied by Astra in the
same round as D9. **Astra ruled the untouched cohort DEFERRED** (verdict line
`COHORT: DEFER — Five-seed family confirmation must precede the only untouched
read.`), so tonight is a validation-only two-seed screen and 10.10 records a
deferral rather than a scoring.

| # | Check | Pass condition |
|---|---|---|
| 10.1 | Saved predictions and alignment | `scripts/sequence_track/stage2_stats.py` reads admitted `runs/<config_id>/seed_<s>/predictions_validation.npz`; asserts finite, valid six-class probabilities and labels, expected shapes, and identical `y` and `innings_id` ordering across every compared arm and seed **before differencing**; verifies correspondence to the pinned validation parquet and its row order, rather than joining or sorting only by innings id; computes `−log(clip(probs[row,y], 1e-15, 1))` using `registered_experiment.row_log_loss`; every delta is candidate minus reference; reconstructed LL is labelled separately from the summary LL used in D9 and never replaces it |
| 10.2 | Tournament-block bootstrap | obtains match ids with `registered_experiment.match_ids`, stripping only the leading innings prefix; maps them through `load_competition_clusters("data/t20s_json")`; records lookup/source hashes, `BOOTSTRAP_CONTRACT_VERSION == tournament_time_block_v1` and `MAX_EVENT_GAP_DAYS == 120`; asserts validation totals **124,292 rows / 545 matches / 47 blocks / 0 unmapped**; refuses missing or ambiguous mappings rather than inventing per-match blocks. Uses 2,000 replicates, `np.random.default_rng(29)`, ball weighting and complete tournament blocks; passes **block ids**, not match ids, to the existing cluster estimators. Reports rows, matches and `count_unique_clusters` separately for every slice; fewer than 10 blocks means descriptive only, no CI-clean claim and no gate pass |
| 10.3 | Two estimands | reports (i) each registered seed checkpoint separately, with paired block-only uncertainty, and (ii) the arithmetic across-seed mean with joint seed-and-block resampling; for (ii) each replicate samples S seed indices with replacement, then B block indices with replacement, applying the same sampled seeds and blocks to both arms; the estimate is summed sampled losses divided by `S × summed sampled block row counts`; it is **not** the LL of averaged probabilities. Retained draws reproduce the existing estimators' 2.5/97.5 percentiles within 1e-9 on identical inputs. Reports per-seed paired deltas, seed spread and direction count; labels tonight's (ii) **descriptive two-seed robustness screen**, not five-seed evidence and not uncertainty for a newly trained single checkpoint; no best-seed selection and no averaging of CI endpoints |
| 10.4 | Registered families and Holm | asserts the config's explicit **15 families**, each with exactly three members: primary on `all`, plus `candidate − mlp` non-inferiority on `death` and on `chase`. Primary references: `full→mlp`, `fixed_decay→mlp`, `fox→fixed_decay`, `aligned_hist→full`, `aligned_hist_rf→aligned_hist`, `recency_k30→mlp`, `same_entity_k30→recency_k30`, `same_entity_unr→aligned_hist_rf`, `same_entity_k0/k6/k12→mlp`, `lstm→mlp`, `xlstm→mlp`, `residual_mlp→mlp`, `residual_t1→residual_mlp`. Holm operates within each three-member family, separately for each estimand/checkpoint readout; never pools or selects between estimands. Uses the threshold-centred percentile convention below, α = 0.05, stable ties in registered member order, monotone adjusted p-values and the full step-down stopping rule; an unavailable member cannot shrink the family or permit a family pass. Reports raw p, resolution flag, adjusted p, rank and rejection. Family members are **registered for confirmation**, but their validation results remain screening; every other contrast and slice is exploratory. No claim of correction across all 15 families or across the k search |
| 10.5 | Direct mechanism contrasts | reports paired intervals for `fox − fixed_decay` ("learned forgetting beyond fixed decay") and `same_entity_k30 − recency_k30` ("ownership plus alignment beyond recency"), regardless of the selected k; additionally `same_entity_unr − aligned_hist_rf` ("mask alone, given aligned inputs and matched relay-free keys"), `aligned_hist_rf − aligned_hist` ("wiring plus key construction") and `aligned_hist − full` ("aligned history input"). Each is inferential only in its registered family and slice; other slice readouts are exploratory. **Never** infers a difference because one arm is significant against MLP and another is not; applies the falsification wording in D10.15 |
| 10.6 | Non-inferiority and eligibility | on each gate slice computes paired row harms `d = LL(candidate) − LL(mlp)`; reports point, ordinary two-sided 95% percentile interval, its upper endpoint `U95 = percentile(draws, 97.5)`, the margin +0.002, the threshold-centred p, the Holm result and the gate status. An unadjusted numerical pass requires **strictly `U95 < 0.002`**; a family-adjusted gate additionally requires favourable Holm rejection against the **+0.002 boundary**, ≥10 blocks and complete paired seeds. Neither a point below the margin nor a failure to detect harm establishes non-inferiority. A candidate clears the numerical validation screen only if its registered primary is favourable under Holm with an upper ordinary 95% endpoint below 0, both gates pass, and its all-row `candidate − mlp` interval is CI-clean favourable as the plan requires; that last reading remains exploratory where it falls outside the family and cannot replace the registered primary. Tonight every status is `SCREEN_PASS`, `SCREEN_NOT_PASS` or `NOT_EVALUABLE`, never confirmatory advancement |
| 10.7 | Exploratory slices | reports `powerplay`, `middle`, `innings_1`, `innings_2`, `thin_pair`, plus registered contrasts outside their family membership, as exploratory; freezes exact slice predicates and missing-value handling before computing, using the frame's phase and innings definitions; `thin_pair` must name its exposure columns and threshold explicitly or be reported unavailable rather than silently invented. The `death` and `chase` gate predicates are likewise pinned and checked against the frame. Blocks are counted within each slice after masking. No exploratory result changes k, families or advancement; this validation-only run opens no test rows |
| 10.8 | Generated report | a generator writes `research/reports/embeddings/SEQ_STAGE2_REPORT.md` from hashed summaries, the statistics and gate JSON, the selection record, the dependency results, run provenance and the config; every numerical value comes from a file, with no hand-entered result cells; full precision in machine-readable outputs, rounding only for display. The report includes all admitted configurations, every incomplete configuration, every registered contrast, per-seed and seed-mean readouts, access differences, parameter counts, compute and machine provenance; regeneration reproduces its content apart from a declared timestamp. Reads no smoke LL |
| 10.9 | Family freeze and pin | before any eventual cohort scoring, freezes the complete candidate list, the selected k, every required control and checkpoint, the training seeds, the three-member family map, slice predicates, estimands, p and interval conventions, eligibility rules, analysis code and source and evidence hashes; includes MLP for every gate and every matched control needed for a retained mechanism claim. The decision and analysis pin is stored separately from immutable training provenance; both verify without rewriting historical run identity. **Tonight writes a `screen freeze / cohort deferred` record, not an authorisation to score**; any later whole-family seed extension produces an explicitly versioned final freeze before cohort access |
| 10.10 | Cohort disposition | tonight records `cohort_status: DEFERRED_UNOPENED`, `cohort_scored: false`, `advances: []`, `provisional: true` and Astra's ruling; performs **no** cohort feature, prediction or base-logit read, and does not run `research/log_verdict.py`. Later execution requires the five-seed preconditions in D10.16 and the final freeze; a single registered scoring batch then covers all frozen candidates and controls, with cohort base logits built only at that point, over the frozen 210-match / 48,406-row set with the cohort block lookup taken from `cohort/raw_json/` under the same per-slice ≥10-block rule; it preserves all results and cannot add arms, change k or tune anything |
| 10.11 | Residual controls | `residual_t1 − residual_mlp` is the residual arm's qualifying primary; `residual_t1 − mlp` alone cannot qualify it. Reports base-only validation LL and paired comparisons of each residual arm against base-only as exploratory, with row alignment and production-booster / base-logit provenance checked; reports residual-versus-MLP results in their registered family or exploratory roles. Calls `residual_mlp` a production-prior control, not a sequence gain |
| 10.12 | Dependency recertification | reruns D7 on the admitted trained checkpoints of `same_entity_k0`, `same_entity_k30`, `same_entity_unr`, `recency_k30` at **both** seeds, with own-S(i) certificates and the registered matched positive controls; records target counts, max and p99, changes and status. Retains the corrected interpretation that positive controls establish sensitivity to excluded-past information, not specifically multi-layer relay. A failure blocks the affected interpretation and eligibility and cannot be repaired by silently substituting another architecture after results are read |
| 10.13 | Deviations, asymmetries, limitations | report § 9 restates **every** current config entry and every accepted D2–D8 and gate-review disposition with its consequence; a manifest-to-report coverage check finds no missing entry. Explicitly includes: blocked cross-fitting rather than past-only forecasting, later-block training for earlier OOF predictions, caches and encoders not fold-refitted, fixed-round residual refits, wiring **plus key construction**, own-outcome versus shifted-history keys, ownership **plus alignment**, machine confounded with seed with arm-by-machine interaction inseparable from seed variation (**no machine term fitted**), all registered xLSTM simplifications, A15 not adopted and checkpoint-selection optimism, two-seed weakness, unmatched parameter counts, access and position schemes, MPS non-bit-reproducibility, fresh stage 2 runs rather than stage 1 replications, the global prior not being as-of, venue history not being recency-weighted, and prior test inspection plus the D4 and D5 test scoring and parity reads. Shared features do not prove shared effects across architectures; invalid historical RSS is labelled unavailable, not reconstructed |
| 10.14 | Plain-language section and status | report § 10 explains what was tested, what came out, the best **observed validation** configuration and its uncertainty, why no arm advances tonight, and the exact next step; distinguishes teacher-forced ball prediction from rollout and from market performance. If no sequence arm beats MLP CI-clean it states "This model family, at this resolution, shows no further sequence gain" and immediately qualifies that this is screening evidence and **not proof that the scoreboard summary suffices**. States that cohort confirmation is pending, that there is no market claim, no LANDED verdict, and that the user's verdict is outstanding |
| 10.15 | Falsification wording | the report uses Astra's registered wording: no result tonight licenses "X is the cause". A favourable `fox − fixed_decay` supports "learned forgetting as an explanation"; a favourable `same_entity_k30 − recency_k30` supports "ownership plus alignment beyond recency" and cannot isolate ownership from alignment; a favourable `same_entity_unr − aligned_hist_rf` supports excluding other participants' history with alignment, wiring and key construction held fixed. Explaining the **death-over harm** additionally requires reproducing that harm on the matched `full − mlp` death slice and direct evidence of improvement on death rows, and those mechanism-on-death comparisons are exploratory under the current map. If neither mechanism contrast establishes benefit, the wording is "neither proposed mechanism is supported at this resolution"; intervals crossing zero mean unresolved evidence, not proof that neither matters. If the `full − mlp` death harm is not reproduced, the report says the original harm was not reproduced on this frame and does not explain an absent effect |
| 10.16 | Cohort unlock preconditions | recorded for the future: **(0)** Astra gate 1 round 2's MUST-FIX 6 must be closed in full first — the historical-consumption question for the 2026-04-17 → 2026-08-05 window must be settled beyond the frame-and-cache ancestry that `docs/sequence_track/stage2_cohort_consumer_audit.md` establishes, because clean training ancestry does not by itself prove untouched *evaluation* status; **(1)** the cohort unlocks only after seeds **29, 42 and 101** are added to every retained whole hypothesis family — candidate, MLP, matched controls, and **all five k configurations whenever k selection is involved**; the seed-extension and final k-selection procedure is frozen before those new results are inspected; the registered selection and gates are rerun on all five seeds reporting spread and the 4/5 favourable-direction count; only eligible candidates are retained; the final family, checkpoint and analysis freeze and the unchanged cohort provenance are verified. If no candidate qualifies, the cohort is left unopened. The later read is one frozen scoring batch with no interim result-driven change; a start record and all outputs are persisted, and any recovery reuses the identical frozen specification, discloses partial exposure and never claims a fresh untouched read |

**Calculation convention for 10.4–10.6** (Astra, verbatim in substance): retain
stage 1's two-sided percentile-tail convention, extended to the non-inferiority
boundary. For bootstrap delta draws `d*`, with threshold `t = 0` for primary
superiority and `t = 0.002` for non-inferiority,

```
p_raw = min(1, 2 · min[ P(d* − t ≤ 0), P(d* − t ≥ 0) ])
```

a conservative boundary test for directional non-inferiority. Do **not**
silently replace the registered upper endpoint of a two-sided 95% interval
with a one-sided 95% bound. With three sorted family p-values,

```
p_holm(r) = min(1, max over j ≤ r of (4 − j) · p(j))
```

Rejection requires the step-down rule, the favourable direction and the
relevant strict upper-bound condition. An unavailable member receives a
non-rejecting placeholder for adjustment and remains `NOT_EVALUABLE`; the
family never passes. Rank-local percentile intervals are reported at level
`1 − 0.05/(4 − r)` and labelled **"rank-local percentile interval; not
simultaneous; not the rejection rule"**; classification is never made from
those intervals alone. With 2,000 draws an empty tail is reported as a
resolution flag and `p < 0.001`, never as exact zero evidence. Stage 1's
±0.007 margin and parity classification are **not** imported.

---

## D11. Astra review rounds, commit, verdict

| # | Check | Pass condition |
|---|---|---|
| 11.1 | Gate 1 (code) | Codex Astra (`gpt-6-astra`, medium reasoning, read-only) reviews D2–D8 and every changed file before the stage 2 code commit; every round's prompt path, output path and verdict line recorded; iterate to `SIGN-OFF` with no MUST-FIX outstanding, or record an explicit disposition for any item deliberately deferred, naming why and when |
| 11.2 | Gate 1 commit | one commit of the stage 2 code, config, tests, acceptance file, handoff, runbook, errata, audit note and `TODO.md`; plain summary plus body per `CLAUDE.md`, **no AI-attribution trailers**; nothing under `models/` committed; hash recorded here |
| 11.3 | Gate 2 (results) | a second Astra review over the statistics, the report and D9–D10 before the results commit; rounds recorded the same way |
| 11.4 | Verdict | `research/log_verdict.py` called only by the orchestrator, only with the unchanged gate JSON, and only on the user's decision; two-seed evidence is provisional and can never be LANDED (invariant 9), and the report says so |

### Result (D11) — in progress

Astra rounds so far:

| round | scope | prompt | output | verdict |
|---|---|---|---|---|
| machine split | the D8 machine question only | `docs/sequence_track/astra/machine_split_prompt.md` | `<scratchpad>/astra/machine_split_round1.md` | `RECOMMENDATION: B — Pair every arm within seed; isolate markers and rebuild summaries.` — adopted by the user, D8 amended above |
| D9/D10 design + cohort | the content of the D9 and D10 tables, the falsification wording, and whether to open the untouched cohort tonight | `docs/sequence_track/astra/d9d10_prompt.md` | `<scratchpad>/astra/d9d10_round1.md` | `COHORT: DEFER — Five-seed family confirmation must precede the only untouched read.` — adopted; D9, D10 and D10.15/10.16 written from it verbatim in substance |
| gate 1 round 1 | D2–D8 and every changed file | `docs/sequence_track/astra/gate1_prompt.md` | `<scratchpad>/astra/gate1_round1.md` | **`VERDICT: AGREE WITH CHANGES`** — 9 MUST-FIX, 4 SHOULD, 5 NOTE; plan MUST-FIX closure 7 CLOSED, 4 STILL OPEN (3, 6, 8, 11) |
| gate 1 round 2 | verification of every round 1 closure, plus regression and unattended-run risk | `docs/sequence_track/astra/gate1_round2_prompt.md` | `<scratchpad>/astra/gate1_round2.md` | **`VERDICT: AGREE WITH CHANGES`** — "Not ready for unattended launch." Round 1: MUST-FIX 1, 4, 5, 7, 8, 9 and SHOULD 1–4 and NOTE 1–5 **CLOSED**; MUST-FIX 2, 3 and 6 **STILL OPEN**; **7 new MUST-FIX** (below). Astra could not execute Python (`uv` cache refused in the read-only sandbox) so it did not re-run pytest; `bash -n` passed |

Gate 1 round 1 dispositions:

| Astra item | disposition |
|---|---|
| MUST-FIX 1 — mLSTM denominator must be `max(\|nᵀq\|, exp(−m))`, not `clamp(min=1)`; the reference test repeats the error | **CLOSED 2026-09-12** — fixed, unstabilised float64 reference written, mutation-checked, output gate restored (SHOULD 4 closed with it), `lstm` and `xlstm` re-smoked; see D3.10 |
| MUST-FIX 2 — driver must verify against registered provenance and include `key_construction` plus implementation identity in reuse; needs a mini-compatible path with no cohort artifacts | **CLOSED 2026-09-12** — eight-component seed-independent signature; `provenance_check` in preflight; `cohort` in `PROVENANCE_SKIPPED` with a recorded reason; see D3.12 |
| MUST-FIX 3 — completion must require all four artefacts and be atomic; incomplete runs retried | **CLOSED 2026-09-12** — four-artefact validation then an `os.replace`d `COMPLETE.json`; incomplete runs cleared and retried |
| MUST-FIX 4 — per-seed peak RSS is invalid (cumulative high-water deltas); old values marked invalid, not reconstructed | **CLOSED 2026-09-12** — per-child process-tree sampling via the already-pinned psutil, method recorded; legacy values read as invalid and left byte-unchanged |
| MUST-FIX 5 — config: drop "wiring alone", let any k win including `unr`, replace the family template with an explicit candidate/reference/slice map | **CLOSED 2026-09-12** — (a) `aligned_hist_rf − aligned_hist` now reads "relay-free wiring PLUS the key construction", matching the `relay_free_keys_carry_own_outcome` asymmetry verbatim, while `same_entity_unr − aligned_hist_rf` is stated to isolate the mask because both endpoints are relay-free with matched key construction; (b) `registered_sweep: [0, 6, 12, 30, unr]` with "ANY k may win, `unr` included" and a `matched_control_constraint` recording that only k = 30 has a registered recency control, that a selected k ≠ 30 may not borrow `recency_k30`, that k = unr's matched relay-free endpoint is `aligned_hist_rf`, and that a control at a newly selected k is a new run queued before 2f; (c) 15 explicit family rows checked against Astra's authoritative map, plus `gate_reference_rule` (both gates are always candidate − mlp whatever the primary reference is), `map_is_complete`, and an `exploratory` list naming the non-qualifying contrasts. Config 751 → 965 lines |
| MUST-FIX 6 — D5.3 overstates the eligibility audit | **PARTIALLY CLOSED, deliberately left open.** Astra round 2: the provenance audit "supports the frozen frame/cache ancestry" but "does not establish D5.3's broader absence of historical training, tuning or inspected evaluation elsewhere… Clean current training ancestry is insufficient to prove untouched evaluation status. Keep this unresolved and make its closure a cohort-unlock prerequisite; it need not prevent validation-only training." Adopted verbatim: the audit stands for what it proves, D5.3's Result says so, and closing it in full is now a registered cohort-unlock prerequisite in D10.16. Tonight's validation-only training proceeds |
| MUST-FIX 7 — stage 1 verification: historical-source mode against `2c7ea6f`, strict live mode retained (Astra's option (c)) | **CLOSED 2026-09-12** — `pin_stage1.py` gained `STAGE1_COMMIT`, `_git_blob`, a `historical_sources()` context manager and `--historical-sources [COMMIT]`; `--write` refuses the flag. Default strict live verification is unchanged and **still fails on this tree** (`pin_stage1: 2 mismatch(es)` on `transformer_t1.py` and `pin_stage1.py`), asserted by a new test; historical mode reports `pin_stage1: OK (historical sources @ 2c7ea6f)` and says explicitly that it is historical-evidence verification, not live-closure certification. `test_source_closure_is_hashed_in_full` now asserts equality with the `2c7ea6f` blob rather than a silent live-or-historical OR, and the previously failing `test_verify_passes_on_the_committed_config` passes through the historical mode citing the erratum. `seq_stage1_sim_v1.yaml` not re-pinned; the erratum gained a paragraph recording NOTE 5 |
| MUST-FIX 8 — recurrent `cell`/`simplifications` never reach `metrics.json`; D3.11's Result is false about it | **CLOSED 2026-09-12** — `arm_params_block(..., extra=…)`; verified in both re-smoked `metrics.json`; D3.11 Result corrected |
| MUST-FIX 9 — D8.2 unfinished and its budget formula omits epochs | **CLOSED 2026-09-12** — two queues of 16 jobs each, seed-specific `output_dir` (confirmed from source that `run_queue.sh` compares only `COMPLETE`'s `config_sha256` and never the command or seed subset), separate STOP files, and thread caps as command env prefixes because `queue_lib.KNOWN_JOB_FIELDS` ignores per-job memory fields (also confirmed from source). Budgets from `ceil_to_0.05((30·smoke·F + 120)/3600)`: mini total **6.75 h**, laptop **3.85 h**, xLSTM 1.90 / 0.95 from its corrected 73.2 s smoke. Three cells sit above Astra's cross-check table on rounding boundaries (lstm 0.25 vs 0.20 at 0.2058, aligned_hist 0.30 vs 0.25 at 0.2508, xlstm recomputed); all three are ≥ the formula, never below. Both files record the 4 × `expected_hours` retry ceiling and name the superseded 62.7 s rather than erasing it |
| SHOULD 1 — D7 exits zero even on failure | **CLOSED** — `exit_code(result)`: masked arm 0 iff max \|Δ\| ≤ 1e-6, positive control 0 iff max \|Δ\| > 1e-3; all four verdict combinations tested plus two real CLI runs |
| SHOULD 2 — refuse cohort rebuild after `FROZEN.json`; stop re-opening closed files in later steps | **CLOSED** — `_guard_frozen(step)` on all seven mutating steps, `--allow-refreeze` required and warned; closed-file invocations cut from 5 to **2, both in `step_audit`**, asserted by an AST test, with `step_context` reading forward ids from the written `eligibility_audit.csv` and `step_freeze` quoting counts from a new `eligibility_summary.json` sidecar (sentinel readers fail the test if called). The builder was not re-run: the cohort dir is untouched and `75daedde…` / 210 / 48,406 stand |
| SHOULD 3 — source closure omits imported feature/materialisation/cache code | **CLOSED** — expanded 8 → 22 entries, verified by AST that every `scripts/` module a stage 2 path imports is covered (adds `embeddings_e1`, `artifacts`, `calibration`, `registered_experiment` and the cohort builder's feature stack), with a test enforcing it going forward |
| NOTE 1 — retain the relay-free own-outcome keys: `y_i` unreachable, every consumed `y_j` precedes `i`; widening S(i) instead would admit other participants' outcomes and weaken the ownership question | **adopted**; the handoff § 3.1 decision stands |
| NOTE 2 — D7.4 wording | adopted verbatim in D7.4 above |
| NOTE 3 — the residual base logits are **blocked cross-fitting, not past-only forecasting**: earlier blocks are scored by boosters fitted on later blocks, and caches/encoders are not fold-refitted. Must not be described as fully as-of or whole-pipeline leakage-free | to be stated in D4's Result and in the report's limitations |
| NOTE 4 — access confirmed; the two closed **files** are opened by more than two invocations and `json.load` parses the whole document while only ids are retained | SHOULD 2 reduces the invocations |
| NOTE 5 — an immutable git blob is sufficient historical source evidence; neither a permanently failing test nor a silent fallback should be the normal contract | MUST-FIX 7's implementation |

Astra could not execute Python (the read-only sandbox refused `uv` cache
initialisation), so its check-by-check disposition rests on reading code and
artifacts, and it did not independently re-run the suites (D3.13, D6.4).

Gate 1 round 2 new findings and dispositions:

| Astra round 2 item | disposition |
|---|---|
| **new 1 — worktree frame identity breaks consolidation** | **CLOSED 2026-09-12.** `frame` now hashes `dir_configured` (the frame directory exactly as `data.directory` is written, via `logical_frame_dir()`, refusing if preflight did not record it), the version, the whole `.feature_hash` and per-split md5 / rows / date range; `stats_cache` hashes the configured manifest role plus the file md5. No resolved path enters any component. `verify_checkpoint`'s frame check compares resolved physical directories instead of relative text, so a worktree symlink and the real directory are one frame. Resolved paths are recorded outside the signature in `physical_paths` and in each run's `machine_provenance.paths`. Two new tests pass: identical signatures **and** identical per-component digests across a real-frame root and a symlinked-frame root, and a consolidation of seed 7 written from the worktree root with seed 13 written from the main root |
| **new 2 — the runbook launches zero mini jobs and uses the wrong STOP file.** `run_queue.sh` defaults to `queue_laptop.yaml`, so `--machine mini` without `--queue` selects nothing and exits zero | **CLOSED by the orchestrator** — runbook § 3 rewritten with explicit `--queue` on both machines, the two per-queue STOP files, the worktree launch, stopped-writer verification and a § 3b consolidation block |
| **new 3 — consolidation can silently violate the machine assignment** | **CLOSED 2026-09-12** — `--consolidate` added, documented in `--help`, the module docstring and a usage example; refuses `--force` and `--dry-run`; its training loop is empty by construction; it never unlinks; on failure it names every incomplete run with its path and reason and directs re-queueing on the owning machine. The old test asserting the dangerous behaviour was replaced by one asserting refusal with zero trainer launches and a byte-identical seed directory, with the ordinary path's retry semantics kept in a separate test. D8.7 and runbook § 3b amended |
| **new 4 — D8.8 machine provenance is not recorded** | **CLOSED 2026-09-12** — see the amended D8.8 |
| **new 5 — the k sweep cannot find its unrestricted run**: `stage2_stats.py` builds `same_entity_kunr` where registration uses `same_entity_unr`, so a complete sweep reads `BLOCKED_INCOMPLETE` | → stats implementer: derive the k-to-configuration mapping from registration, never by string concatenation; test an `unr` winner on the real registered ids. Not a launch blocker (analysis only) |
| **new 6 — the statistics script admits unverified evidence**: existence and alignment are checked but not the seed/config contract, the training signature or the completion manifest; `k_sweep` reads summary LLs without verifying provenance; `base_only_readout` copies provenance without checking hashes; `load_frame` never compares to the pin | → stats implementer: one read-only admission verifier every entry point passes through, rejecting rather than warning. Not a launch blocker (analysis only) |
| **new 7 — the pin is already stale**: the config records `stage2_stats.py` absent and it now exists | **CLOSED at commit time** — the pin is re-written and re-verified as the last action before the commit, after every agent lands; `sim_eval/eval_statistics.py` is added to the closure or the claim is delimited |
| MUST-FIX 2 (round 1) | **CLOSED 2026-09-12** — `scripts/embeddings_e1.py` (the 50-feature contract imported at trainer module scope) and `scripts/artifacts.py` (resolves the cache by manifest role) added to every arm's implementation sources, with `recurrent_arms.py` still recurrent-only; the closure was derived by an AST import walk over `scripts/` rooted at the two trainer modules and a test re-derives it so a new import cannot slip past. `source_comparison()` compares the live sha256 of the **selected arms' training closure only** against the pin's `provenance.sources.source_sha256`, refusing on mismatch with "the code that would train tonight is not the code the pin registered"; a non-training source can never refuse a night. A training source with no pin entry is **not** silently passed — it is recorded in `training_sources_not_pinned` and in the skip record with a re-pin instruction, and a `sources` block without a usable mapping refuses outright |
| MUST-FIX 3 (round 1) | **CLOSED 2026-09-12** — `model.pt` is loaded with `weights_only=True`, must be a non-empty state dict, and its parameter names must match the arm's wiring (recurrent arms all under `recurrent.`; others carrying `feat_proj.`/`head.` and no `recurrent.` key). `manifest_problems()` re-checks every artefact's size and md5 against the `COMPLETE.json` manifest on reuse. The three record states are explicit in `completion_status()` and in the refusal text: **absent** (legacy, allowed, verified by contract), **malformed** (refused as unverifiable, explicitly distinguished from legacy), **disagreeing** (refused naming the field). Test fixtures now write real state dicts, with new tests for a truncated `model.pt`, a wrong-arm checkpoint and a replaced predictions file |

### D9 and D10 implementation (recorded 2026-09-12, before the runs exist)

`scripts/sequence_track/stage2_stats.py` (2,427 lines),
`render_stage2_report.py` (1,293), `scripts/tests/test_stage2_stats.py`
(1,591), `test_render_stage2_report.py` (524); 108 tests pass. Enforcement,
check by check: D10.1 `assert_alignment()` (shape, non-finite, row sums,
class range, and elementwise agreement of `y` and `innings_id` with the
parquet row order — no join, no sort); D10.2 `resolve_blocks()` (contract
version and 120-day gap asserted, lookup hashed, ambiguous mappings refused,
124,292 / 545 / 47 / 0 asserted, **block ids** returned) with `slice_stats()`
setting `descriptive` under 10 blocks which forces `NOT_EVALUABLE`; D10.3
`bootstrap_draws()` and `joint_seed_block_draws()` with
`is_log_loss_of_averaged_probabilities: false` recorded explicitly; D10.4
`registered_families()`, `threshold_centred_p()`, `holm_family()` giving an
unavailable or descriptive member a non-rejecting placeholder that keeps
m = 3 so a family can never shrink or pass, and a test that greps for
`0.007`, `holm_stage1` and `PARITY_BAND` to prove stage 1's margin was not
imported; D10.5 `MECHANISM_CONTRASTS` plus `MECHANISM_GUARD`, with the
significant-versus-not inference never computed; D10.6 `member_status()`
requiring strictly `U95 < 0.002` **and** Holm rejection, and `_check_status()`
raising on any status containing "advance"; D10.7 `SLICE_PREDICATES` with
frozen expressions and columns; D10.11 `base_only_readout()` and
`residual_vs_base()`; D9 `k_configurations()` and `k_sweep()`.

The joint-resampling estimator was validated against the repository's own:
retained draws reproduce `seed_mean_match_cluster_ci` and `match_cluster_ci`
percentiles to better than 1e-9.

**Deviation from Astra's relayed wording, flagged for its round 3.** Astra
asked that the training signature be "identical across every run admitted into
one comparison". That cannot hold: the driver's signature includes
`config_id`, `arm`, `arm_params` and `implementation`, so it is
seed-independent but arm-**dependent** by construction and the rule as stated
would reject every real night. Implemented instead: **one signature per
configuration across its seeds**, plus **component-by-component equality of
`training_block`, `frame` and `stats_cache` across all arms**, with a test
asserting arm-specific components may differ.

**Three items not implementable as specified**, each disclosed rather than
worked around: `thin_pair` is reported **unavailable with its reason** because
the i7 frame carries no exposure column (`batter_balls_faced` and
`bowler_balls_in_innings` are within-innings counters and the kit carrying
training-ball exposure is off for this stage) — D10.7 permits exactly this and
a config block turns it on; **D10.12 recertification is read, not run** — the
report tabulates whatever is in `models/embeddings/seq_stage2/dependency/` and
marks rows whose checkpoint lies under `smoke/` as the structural one-epoch
certificate pending recertification, which is a results-gate action; and
`innings_2` and `chase` select **identical row sets** on this frame (57,994
each), reported as two registered slices rather than collapsed.

A dry execution against the real tree with no runs present reported 0 of 32
seed runs admitted, all 16 configurations incomplete, 15 families × 3 members,
147 contrasts `NOT_EVALUABLE`, every family screen `NOT_EVALUABLE` on all
three readouts, `cohort DEFERRED_UNOPENED`, and `ksweep BLOCKED_INCOMPLETE`;
the render produced a report passing its own § 9 coverage check. The only
numbers read were structural: the frame and block assertion at 124,292 rows /
545 matches / 47 blocks / 0 unmapped, and the base-logit provenance verifying
against the pin on all five hashes.

Gate 1 round 3 (final) — `VERDICT: AGREE WITH CHANGES`, prompt
`docs/sequence_track/astra/gate1_round3_prompt.md`, output
`<scratchpad>/astra/gate1_round3.md`. It independently re-hashed all 23
recorded sources with `shasum` and found every one current; it could not run
Python (`uv` cache refused) so it did not reproduce the pytest counts.
Dispositions: new 1, 3, 4, 5 **CLOSED**; round-1 MUST-FIX 2 and 3 **CLOSED**
for the registered training tree; round-1 MUST-FIX 6 **still open by design**
as the cohort-unlock prerequisite; new 2 and 7 were still open and are closed
below; new 6 remains open into the results gate.

| round 3 item | disposition |
|---|---|
| **MUST-FIX 1 — the documented launch selects the wrong worktree commit.** `git -C "$WT" checkout --detach HEAD` resolves HEAD *in the worktree*, which is still `2c7ea6f`, so the documented command would have **trained stale code**. Also: the stopped-writer check omitted `retrain_stage2.py`, which writes run records and summaries after the trainer exits, so a half-written run could be copied; and the morning transfer copied the mini's whole runs tree rather than only its seed 13 directories | **CLOSED by the orchestrator 2026-09-12** — the runbook now resolves `TRAIN_COMMIT=$(git -C "$MAIN" rev-parse HEAD)`, checks the worktree out at that commit, asserts both the worktree and the mini are on it before launch, includes `retrain_stage2.py` in the writer check, and restricts the rsync to `seed_13/***` so a laptop-owned seed 7 directory can never be overwritten |
| **MUST-FIX 4 — complete or explicitly delimit the analysis pin**: the closure omitted `sim_eval/eval_statistics.py` and the report generator, so a change to the bootstrap contract or the renderer could move reported numbers without moving any recorded hash | **CLOSED by the orchestrator 2026-09-12** — both added to `PROVENANCE_SOURCE_CLOSURE`; re-pinned, **25 sources hashed**, `pin_stage2: OK` |
| **MUST-FIX 2 — admission must fail closed**: `verify_run_admission()` accepts an absent or empty artefact manifest and missing size/hash declarations; `assert_comparable()` skips absent components; a non-empty `provenance` block without a validation hash qualifies as an available pin | **OPEN into the results gate.** Analysis-only: it cannot affect tonight's training, and no number is read before the morning. Assigned before any result is quoted |
| **MUST-FIX 3 — authenticate summary values and apply comparability to k selection**: editing only an `ll` in `summary.yaml` leaves the checkpoint and signature checks satisfied and can change the winner; `load_runs()` reads summaries without the summary verifier; `k_sweep()` never calls `assert_comparable()` | **OPEN into the results gate**, same reason. Astra's fix: compare each summary LL against the manifest-verified `metrics.json` validation LL, which preserves the designated summary number rather than substituting a reconstructed probability loss |
| minor — `--consolidate`'s earlier verification loop can raise before the aggregation block, so it does not always name *every* invalid run | **OPEN into the results gate** (disclosure completeness, not safety) |

**Astra's ruling on the signature deviation:** "yes, one signature per
configuration across seeds is the correct interpretation of my intent.
Requiring identical full signatures across arms was overbroad." It added four
requirements the statistics must also enforce, recorded here for the results
gate: common implementation-source hashes must agree across arms (the
recurrent-only entry excepted); `base_logits` identity must be identical
between the two residual arms; each configuration's arm-specific components
must be verified against its registered expected identity; and shared
frame/cache/training identity must be anchored **to the pin**, not merely to
agreement among runs.

**Astra's three rulings**, all adopted: `thin_pair` unavailable with its
exposure-column reason is acceptable; the deferred dependency recertification
is acceptable but the report's heading must say **pending** rather than
"recertified" until trained-checkpoint coverage is verified, and a failed
certificate must **block** the affected interpretation and eligibility rather
than merely appear in a table; and the identical `innings_2` / `chase` row sets
are acceptable with their equivalence disclosed and never described as
independent corroboration, with only `chase` belonging to the three-member
family.

**Astra's launch answer, verbatim in substance:** with the correct committed
checkout, explicit queue paths, fresh markers and exclusive writers it found
"no further deterministic trainer/consolidation failure in the inspected
paths" and confirmed "consolidation will not retrain seed 13 on the laptop".
The three conditions it named were the stale-checkout command, the incomplete
writer check and the analysis admission gaps; the first two are closed above
and the third cannot touch tonight's training.

Gate 1 round 4 (scoped) — prompt
`docs/sequence_track/astra/gate1_round4_prompt.md`, output
`<scratchpad>/astra/gate1_round4.md`, verdict **`VERDICT: SIGN-OFF FOR
TRAINING AND LAUNCH; ANALYSIS DEFERRED TO GATE 2`**. It confirmed MUST-FIX 1
closed (`TRAIN_COMMIT` resolves from `$MAIN`, both checkouts asserted against
it, the writer check covers queue, driver and trainer on both machines, and the
scoped rsync moves seed 13 without overwriting seed 7), confirmed MUST-FIX 4
closed and **independently verified all 25 pinned hashes match disk**, and
found that "the two statistics defects have no call path into tonight's
training, artifact writes or queue resumption" and that consolidation's early
exception "cannot enter its empty training loop, delete runs or impair
tonight's queues". Its one added condition, now in the runbook: the preflight
snippets do not themselves abort what follows, so **every check is a mandatory
operator gate** and a failed assertion must stop the launch. It also noted it
could not run full pin verification (that would read excluded cohort metadata)
or pytest (`uv` cache refused in the sandbox), and that the sign-off holds
"provided the documented preflights pass, writers remain exclusive, and
training checkouts, inputs and shared dependencies remain unchanged overnight".

Astra's standing observations for the night: sharing `models/embeddings` across
the worktree and the main checkout is workable **with one writer per seed
directory, no concurrent consolidation and no rsync into an active run**; a
shared `.venv` is workable only while dependencies do not change, since
`--no-sync` stops the queue commands from syncing it but not another session
from altering it, so **no dependency change during the night**; and the
runner's exit status is not evidence of 32 completed runs, because memory
refusals and exhausted retries still exit zero.
