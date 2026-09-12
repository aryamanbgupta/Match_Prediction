# Sequence track — stage 2 report (sixteen configurations, two seeds, validation only)

Generated (declared timestamp, the only value that moves on a re-render): **2026-09-11T21:48:09.995179+00:00** by `scripts/sequence_track/render_stage2_report.py` from the files of record. Every number in every table is read from a file; the narrative passages are authored text in the generator.

* statistics: `eval_out/seq_stage2/stats.json` (sha256 `5d0539c841d4…`)
* k selection: `eval_out/seq_stage2/k_selection.json` (sha256 `423b121f0a9e…`)
* config: `experiments/configs/seq_stage2_v1.yaml` (sha256 `ff3341a4b057…`)
* acceptance checks and every result block: `docs/sequence_track/stage2_acceptance.md`

**Status: validation-only two-seed directional screen. `cohort_status: DEFERRED_UNOPENED`, `cohort_scored: false`, `advances: []`, `provisional: true`. No arm advances. No market claim. No LANDED verdict — two-seed evidence is provisional and can never be LANDED (invariant 9). The user's verdict is outstanding.**

Statuses this stage can emit: `SCREEN_PASS`, `SCREEN_NOT_PASS`, `NOT_EVALUABLE`. There is no advancement status.

## 1. Question

Does any registered change to the within-innings sequence model improve teacher-forced ball-outcome prediction over the memory-less token MLP on the i7 identity frame — and if so, which mechanism explains it? Sixteen configurations were trained fresh under one protocol so that every contrast in the arm register is a matched pair: fixed decay, learned forgetting (FoX), participant-aligned history, relay-free wiring, recency and ownership masks across five window sizes, two recurrent arms, and two residual arms over the production ball model.

Registered purpose, verbatim from the config: Register the sixteen stage-2 sequence configurations — the mechanism arms (fixed decay, learned forgetting, participant-aligned history, relay-free wiring, recency and ownership masks, two recurrent arms) and the two residual arms over the production ball model — each trained fresh on the i7 identity frame under one protocol, so every contrast in the arm register is a matched pair. Stage 1 checkpoints are NOT reused (D2.2): these are separate runs, not replications.

Sign convention: candidate minus reference, in validation log loss, negative favourable.

## 2. Arms and settings

One training block for every arm, identical to stage 1's so the two stages' protocols are comparable: device `mps`, seeds [7, 13], d_model 128, 2 layers, 4 heads, batch 128, 30 epochs, lr 0.0003, patience 3, aux False, no-kit True, score_test False. Frame `data/xgb_data_i7` (50 features, role `ball_frame_i7`), cache role `stats_cache_i7`. Stage 1 checkpoints are **not** reused (D2.2).

| config id | arm | k | history input | wiring | access feat/hist/identity/prod | reference | seeds admitted | parameters |
|---|---|---|---|---|---|---|---|---|
| `mlp` | `mlp` | – | none | token_mlp | y/n/n/n | – | [7, 13] | 272006 |
| `full` | `full` | – | innings_previous | standard | y/y/n/n | `mlp` | [7, 13] | 298758 |
| `fixed_decay` | `fixed_decay` | – | innings_previous | standard | y/y/n/n | `mlp` | [7, 13] | 273158 |
| `fox` | `fox` | – | innings_previous | standard | y/y/n/n | `mlp`, `fixed_decay` | [7, 13] | 274190 |
| `aligned_hist` | `aligned_hist` | – | participant_aligned | standard | y/y/y/n | `mlp`, `full` | [7, 13] | 299654 |
| `recency_k30` | `recency` | 30 | innings_previous | relay_free | y/y/n/n | `mlp` | [7, 13] | 300038 |
| `same_entity_k30` | `same_entity` | 30 | participant_aligned | relay_free | y/y/y/n | `mlp`, `recency_k30` | [7, 13] | 300934 |
| `aligned_hist_rf` | `aligned_hist_rf` | – | participant_aligned | relay_free | y/y/y/n | `aligned_hist` | [7, 13] | 300934 |
| `same_entity_k0` | `same_entity` | 0 | participant_aligned | relay_free | y/y/y/n | `mlp` | [7, 13] | 300934 |
| `same_entity_k6` | `same_entity` | 6 | participant_aligned | relay_free | y/y/y/n | `mlp` | [7, 13] | 300934 |
| `same_entity_k12` | `same_entity` | 12 | participant_aligned | relay_free | y/y/y/n | `mlp` | [7, 13] | 300934 |
| `same_entity_unr` | `same_entity` | unr | participant_aligned | relay_free | y/y/y/n | `mlp`, `aligned_hist_rf` | [7, 13] | 300934 |
| `lstm` | `lstm` | – | innings_previous | recurrent | y/y/n/n | `mlp` | [7, 13] | 272390 |
| `xlstm` | `xlstm` | – | innings_previous | recurrent | y/y/n/n | `mlp` | [7, 13] | 373174 |
| `residual_mlp` | `residual_mlp` | – | none | token_mlp | y/n/n/y | `mlp` | [7, 13] | 272006 |
| `residual_t1` | `residual_t1` | – | innings_previous | standard | y/y/n/y | `residual_mlp` | [7, 13] | 298758 |

Every configuration has both registered seeds.

`arm_params` is read verbatim from each run's `metrics.json`; the recurrent arms carry extra `cell` and `simplifications` keys beyond the standard block and those are displayed as recorded.

### Machine provenance per run (D8.8)

Seed 7 trained on the laptop and seed 13 on the Mac mini, so **machine is confounded with seed**. Every registered contrast is within-machine at each seed, so an additive machine effect cancels inside it; arm-by-machine interaction remains inseparable from seed variation and **no machine term is fitted**.

| config id | seed | machine | host | chip | os | torch | device / MPS | thread caps | wall s | reconstructed val LL | summary.yaml val LL |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `mlp` | 7 | laptop | Aryamans-MacBook-Pro.local | Apple M5 Pro | Darwin 25.4.0 | 2.5.1 | mps (mps built True, available True) | MKL=2, OMP=2, OPENBLAS=2, VECLIB=2 | 43.579952001571655 | 1.438131 | 1.438131 |
| `mlp` | 13 | mini | Aryamans-Mac-mini.local | Apple M4 | Darwin 24.0.0 | 2.5.1 | mps (mps built True, available True) | MKL=None, OMP=4, OPENBLAS=None, VECLIB=None | 102.35219097137451 | 1.438036 | 1.438036 |
| `full` | 7 | laptop | Aryamans-MacBook-Pro.local | Apple M5 Pro | Darwin 25.4.0 | 2.5.1 | mps (mps built True, available True) | MKL=2, OMP=2, OPENBLAS=2, VECLIB=2 | 134.71318817138672 | 1.436984 | 1.436984 |
| `full` | 13 | mini | Aryamans-Mac-mini.local | Apple M4 | Darwin 24.0.0 | 2.5.1 | mps (mps built True, available True) | MKL=None, OMP=4, OPENBLAS=None, VECLIB=None | 206.16198015213013 | 1.438350 | 1.438350 |
| `fixed_decay` | 7 | laptop | Aryamans-MacBook-Pro.local | Apple M5 Pro | Darwin 25.4.0 | 2.5.1 | mps (mps built True, available True) | MKL=2, OMP=2, OPENBLAS=2, VECLIB=2 | 123.58378911018372 | 1.433149 | 1.433149 |
| `fixed_decay` | 13 | mini | Aryamans-Mac-mini.local | Apple M4 | Darwin 24.0.0 | 2.5.1 | mps (mps built True, available True) | MKL=None, OMP=4, OPENBLAS=None, VECLIB=None | 119.96700024604797 | 1.436028 | 1.436028 |
| `fox` | 7 | laptop | Aryamans-MacBook-Pro.local | Apple M5 Pro | Darwin 25.4.0 | 2.5.1 | mps (mps built True, available True) | MKL=2, OMP=2, OPENBLAS=2, VECLIB=2 | 113.72289896011353 | 1.433167 | 1.433167 |
| `fox` | 13 | mini | Aryamans-Mac-mini.local | Apple M4 | Darwin 24.0.0 | 2.5.1 | mps (mps built True, available True) | MKL=None, OMP=4, OPENBLAS=None, VECLIB=None | 128.43939805030823 | 1.435915 | 1.435915 |
| `aligned_hist` | 7 | laptop | Aryamans-MacBook-Pro.local | Apple M5 Pro | Darwin 25.4.0 | 2.5.1 | mps (mps built True, available True) | MKL=2, OMP=2, OPENBLAS=2, VECLIB=2 | 144.86188077926636 | 1.436137 | 1.436137 |
| `aligned_hist` | 13 | mini | Aryamans-Mac-mini.local | Apple M4 | Darwin 24.0.0 | 2.5.1 | mps (mps built True, available True) | MKL=None, OMP=4, OPENBLAS=None, VECLIB=None | 281.57271456718445 | 1.436127 | 1.436127 |
| `recency_k30` | 7 | laptop | Aryamans-MacBook-Pro.local | Apple M5 Pro | Darwin 25.4.0 | 2.5.1 | mps (mps built True, available True) | MKL=2, OMP=2, OPENBLAS=2, VECLIB=2 | 227.76395225524902 | 1.437450 | 1.437450 |
| `recency_k30` | 13 | mini | Aryamans-Mac-mini.local | Apple M4 | Darwin 24.0.0 | 2.5.1 | mps (mps built True, available True) | MKL=None, OMP=4, OPENBLAS=None, VECLIB=None | 315.9129550457001 | 1.438170 | 1.438170 |
| `same_entity_k30` | 7 | laptop | Aryamans-MacBook-Pro.local | Apple M5 Pro | Darwin 25.4.0 | 2.5.1 | mps (mps built True, available True) | MKL=2, OMP=2, OPENBLAS=2, VECLIB=2 | 184.5499930381775 | 1.436865 | 1.436865 |
| `same_entity_k30` | 13 | mini | Aryamans-Mac-mini.local | Apple M4 | Darwin 24.0.0 | 2.5.1 | mps (mps built True, available True) | MKL=None, OMP=4, OPENBLAS=None, VECLIB=None | 430.5857539176941 | 1.436406 | 1.436406 |
| `aligned_hist_rf` | 7 | laptop | Aryamans-MacBook-Pro.local | Apple M5 Pro | Darwin 25.4.0 | 2.5.1 | mps (mps built True, available True) | MKL=2, OMP=2, OPENBLAS=2, VECLIB=2 | 241.1092529296875 | 1.435854 | 1.435854 |
| `aligned_hist_rf` | 13 | mini | Aryamans-Mac-mini.local | Apple M4 | Darwin 24.0.0 | 2.5.1 | mps (mps built True, available True) | MKL=None, OMP=4, OPENBLAS=None, VECLIB=None | 319.35297298431396 | 1.437583 | 1.437583 |
| `same_entity_k0` | 7 | laptop | Aryamans-MacBook-Pro.local | Apple M5 Pro | Darwin 25.4.0 | 2.5.1 | mps (mps built True, available True) | MKL=2, OMP=2, OPENBLAS=2, VECLIB=2 | 211.77813291549683 | 1.439333 | 1.439333 |
| `same_entity_k0` | 13 | mini | Aryamans-Mac-mini.local | Apple M4 | Darwin 24.0.0 | 2.5.1 | mps (mps built True, available True) | MKL=None, OMP=4, OPENBLAS=None, VECLIB=None | 430.60792303085327 | 1.438877 | 1.438877 |
| `same_entity_k6` | 7 | laptop | Aryamans-MacBook-Pro.local | Apple M5 Pro | Darwin 25.4.0 | 2.5.1 | mps (mps built True, available True) | MKL=2, OMP=2, OPENBLAS=2, VECLIB=2 | 211.03894305229187 | 1.437994 | 1.437994 |
| `same_entity_k6` | 13 | mini | Aryamans-Mac-mini.local | Apple M4 | Darwin 24.0.0 | 2.5.1 | mps (mps built True, available True) | MKL=None, OMP=4, OPENBLAS=None, VECLIB=None | 351.5826871395111 | 1.438248 | 1.438248 |
| `same_entity_k12` | 7 | laptop | Aryamans-MacBook-Pro.local | Apple M5 Pro | Darwin 25.4.0 | 2.5.1 | mps (mps built True, available True) | MKL=2, OMP=2, OPENBLAS=2, VECLIB=2 | 183.380619764328 | 1.437303 | 1.437303 |
| `same_entity_k12` | 13 | mini | Aryamans-Mac-mini.local | Apple M4 | Darwin 24.0.0 | 2.5.1 | mps (mps built True, available True) | MKL=None, OMP=4, OPENBLAS=None, VECLIB=None | 430.1521780490875 | 1.436995 | 1.436995 |
| `same_entity_unr` | 7 | laptop | Aryamans-MacBook-Pro.local | Apple M5 Pro | Darwin 25.4.0 | 2.5.1 | mps (mps built True, available True) | MKL=2, OMP=2, OPENBLAS=2, VECLIB=2 | 243.10666704177856 | 1.436195 | 1.436195 |
| `same_entity_unr` | 13 | mini | Aryamans-Mac-mini.local | Apple M4 | Darwin 24.0.0 | 2.5.1 | mps (mps built True, available True) | MKL=None, OMP=4, OPENBLAS=None, VECLIB=None | 477.95039916038513 | 1.436222 | 1.436222 |
| `lstm` | 7 | laptop | Aryamans-MacBook-Pro.local | Apple M5 Pro | Darwin 25.4.0 | 2.5.1 | mps (mps built True, available True) | MKL=2, OMP=2, OPENBLAS=2, VECLIB=2 | 70.04596400260925 | 1.440031 | 1.440031 |
| `lstm` | 13 | mini | Aryamans-Mac-mini.local | Apple M4 | Darwin 24.0.0 | 2.5.1 | mps (mps built True, available True) | MKL=None, OMP=4, OPENBLAS=None, VECLIB=None | 95.30028629302979 | 1.441506 | 1.441506 |
| `xlstm` | 7 | laptop | Aryamans-MacBook-Pro.local | Apple M5 Pro | Darwin 25.4.0 | 2.5.1 | mps (mps built True, available True) | MKL=2, OMP=2, OPENBLAS=2, VECLIB=2 | 1707.6815888881683 | 1.432509 | 1.432509 |
| `xlstm` | 13 | mini | Aryamans-Mac-mini.local | Apple M4 | Darwin 24.0.0 | 2.5.1 | mps (mps built True, available True) | MKL=None, OMP=4, OPENBLAS=None, VECLIB=None | 2170.550081014633 | 1.434290 | 1.434290 |
| `residual_mlp` | 7 | laptop | Aryamans-MacBook-Pro.local | Apple M5 Pro | Darwin 25.4.0 | 2.5.1 | mps (mps built True, available True) | MKL=2, OMP=2, OPENBLAS=2, VECLIB=2 | 35.02935981750488 | 1.432464 | 1.432464 |
| `residual_mlp` | 13 | mini | Aryamans-Mac-mini.local | Apple M4 | Darwin 24.0.0 | 2.5.1 | mps (mps built True, available True) | MKL=None, OMP=4, OPENBLAS=None, VECLIB=None | 39.191575050354004 | 1.432654 | 1.432654 |
| `residual_t1` | 7 | laptop | Aryamans-MacBook-Pro.local | Apple M5 Pro | Darwin 25.4.0 | 2.5.1 | mps (mps built True, available True) | MKL=2, OMP=2, OPENBLAS=2, VECLIB=2 | 1037.5431499481201 | 1.432556 | 1.432556 |
| `residual_t1` | 13 | mini | Aryamans-Mac-mini.local | Apple M4 | Darwin 24.0.0 | 2.5.1 | mps (mps built True, available True) | MKL=None, OMP=4, OPENBLAS=None, VECLIB=None | 75.812016248703 | 1.432499 | 1.432499 |

The reconstructed column is log loss reconstructed from the run's saved row-aligned probabilities; reported separately from the summary.yaml log loss used in D9 and never a replacement for it — the `summary.yaml` value is the number of record for D9 and is never replaced by the reconstruction.

### Admission and provenance verification

No number in this report comes from a run that was admitted on file existence alone. Every run passed a read-only admission verifier first: its `metrics.json`, `run_record.json` and `COMPLETE.json` all name this configuration id and this seed, the arm and `k` match registration, a `training_signature` is present and identical in the run record and the completion record, and the completion record's per-artefact size and md5 manifest still matches the files on disk. Duplicate seed rows are rejected, never deduplicated.

* Pin available: **yes** (config body sha256 `463d022b6d59…`)
* Validation parquet md5 `326436317310adadabe0175825e57d1b`, pinned `326436317310adadabe0175825e57d1b`; 124292 rows, pinned 124292. the frame md5 and row count, and every base-logit hash, are compared against this pin before any number is computed
* one seed-independent training signature per configuration; the shared components ['training_block', 'frame', 'stats_cache'] identical across every admitted configuration AND equal to the digests recomputed from the pin; the common implementation sources identical across arms with only the recurrent-only entry differing; one base-logits identity across the two residual arms; every arm-specific component equal to its registered expected identity
* the driver's signature includes config_id, arm, arm_params and implementation, so it is arm-dependent by construction and is NOT required to match across arms; what must match is the data and training-block part, checked component by component, plus each arm's own components against its registered identity

| configuration | seed-independent training signature |
|---|---|
| `aligned_hist` | `0b28732a576b7c20…` |
| `aligned_hist_rf` | `6b7f72cb7159b803…` |
| `fixed_decay` | `baa25a1e28748262…` |
| `fox` | `fa26eaf6f11250d6…` |
| `full` | `f30f355fd05ef001…` |
| `lstm` | `19492ee2b325a40d…` |
| `mlp` | `123a2841069b7bc2…` |
| `recency_k30` | `d3041ef36a6bac4d…` |
| `residual_mlp` | `89326bbcee04396c…` |
| `residual_t1` | `cf1955ca621ac891…` |
| `same_entity_k0` | `14c29ce1ec37d762…` |
| `same_entity_k12` | `f628c20ad64d0a13…` |
| `same_entity_k30` | `7c005a58edef5c89…` |
| `same_entity_k6` | `cfded4801b54c384…` |
| `same_entity_unr` | `d26ae21c72a81b93…` |
| `xlstm` | `86c75b12094e2ef4…` |

Shared signature components, identical across every admitted configuration: `training_block` = `3c17a16c7ba6…`, `frame` = `cc3270fde76d…`, `stats_cache` = `b46b06e42951…`.

### Ownership dependency certificates (D7; D10.12 recertification **complete on trained checkpoints at both registered seeds, with the registered positive controls matched**)

| arm | k | scored against S(i) of | checkpoint | seed | n targets | max |Δ| | p99 |Δ| | n > 1e-6 | checkpoint md5 | role | result |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `aligned_hist` | – | `same_entity` k=30 | `seed_7` (smoke checkpoint) | 7 | 2000 | 9.217e-01 | 7.411e-01 | 1867 | `NOT_A_REQUIRED_MASKED_ARM` | positive control | SEES EXCLUDED-PAST INFORMATION |
| `full` | – | `recency` k=30 | `seed_7` (smoke checkpoint) | 7 | 2000 | 7.517e-01 | 5.774e-01 | 1449 | `NOT_A_REQUIRED_MASKED_ARM` | positive control | SEES EXCLUDED-PAST INFORMATION |
| `recency` | 30 | `recency` k=30 | `seed_7` (smoke checkpoint) | 7 | 2000 | 0.000e+00 | 0.000e+00 | 0 | `NOT_AUTHENTICATED_CLOSED_TREE` | masked arm | PASS |
| `recency` | 30 | `recency` k=30 | `seed_13` | 13 | 2000 | 0.000e+00 | 0.000e+00 | 0 | `AUTHENTICATED` | masked arm | PASS |
| `recency` | 30 | `recency` k=30 | `seed_7` | 7 | 2000 | 0.000e+00 | 0.000e+00 | 0 | `AUTHENTICATED` | masked arm | PASS |
| `same_entity` | 0 | `same_entity` k=0 | `seed_13` | 13 | 2000 | 0.000e+00 | 0.000e+00 | 0 | `AUTHENTICATED` | masked arm | PASS |
| `same_entity` | 0 | `same_entity` k=0 | `seed_7` | 7 | 2000 | 0.000e+00 | 0.000e+00 | 0 | `AUTHENTICATED` | masked arm | PASS |
| `same_entity` | 30 | `same_entity` k=30 | `seed_13` | 13 | 2000 | 0.000e+00 | 0.000e+00 | 0 | `AUTHENTICATED` | masked arm | PASS |
| `same_entity` | 30 | `same_entity` k=30 | `seed_7` | 7 | 2000 | 0.000e+00 | 0.000e+00 | 0 | `AUTHENTICATED` | masked arm | PASS |
| `same_entity` | unr | `same_entity` k=unr | `seed_13` | 13 | 2000 | 0.000e+00 | 0.000e+00 | 0 | `AUTHENTICATED` | masked arm | PASS |
| `same_entity` | unr | `same_entity` k=unr | `seed_7` | 7 | 2000 | 0.000e+00 | 0.000e+00 | 0 | `AUTHENTICATED` | masked arm | PASS |
| `same_entity` | 0 | `same_entity` k=0 | `seed_7` (smoke checkpoint) | 7 | 2000 | 0.000e+00 | 0.000e+00 | 0 | `NOT_AUTHENTICATED_CLOSED_TREE` | masked arm | PASS |
| `same_entity` | 30 | `same_entity` k=30 | `seed_7` (smoke checkpoint) | 7 | 2000 | 0.000e+00 | 0.000e+00 | 0 | `NOT_AUTHENTICATED_CLOSED_TREE` | masked arm | PASS |
| `same_entity` | unr | `same_entity` k=unr | `seed_7` (smoke checkpoint) | 7 | 2000 | 0.000e+00 | 0.000e+00 | 0 | `NOT_AUTHENTICATED_CLOSED_TREE` | masked arm | PASS |

The positive controls establish sensitivity to excluded-past information, not specifically multi-layer relay (Astra gate 1, D7.4). A relay-free certificate always uses its own S(i); a control is a standard-wiring arm scored against the **masked arm's** S(i), which is why it is matched through the `scored against S(i) of` column and never through its own arm and k (Astra gate 2 round 1 MUST-FIX 1). Where a row is marked *smoke checkpoint* the certificate is the structural one from the one-epoch smoke weights; masking is a property of the architecture, and that row is never counted as trained-checkpoint coverage. Checkpoints inside the closed smoke tree are not opened by this report, so their recorded md5 reads `NOT_AUTHENTICATED_CLOSED_TREE`. A smoke record can never carry an arm's eligibility either: an arm whose trained recertifications are absent reads `TRAINED_SEEDS_INCOMPLETE` and is blocked, even when the smoke record is the only certificate present (Astra gate 2 round 2).

Trained-checkpoint coverage is verified, not asserted: it requires, for each of the four masked configurations, a passing certificate at **both** registered training seeds 7, 13, each certificate's recorded checkpoint md5 recomputed from `model.pt` on disk and matching, the registered perturbation settings (n_targets 2000, seed 29, split 'validation'), and the registered matched positive control present and fired. Coverage complete: **yes** — `recency_k30`, `same_entity_k0`, `same_entity_k30`, `same_entity_unr`.

| configuration | certificate | trained-checkpoint coverage | matched positive control | consequence |
|---|---|---|---|---|
| `recency_k30` | `PASS` | trained checkpoints at seeds 7, 13 (md5-authenticated) | `full` vs S(i) of recency k=30 → fired | no block from this check |
| `same_entity_k0` | `PASS` | trained checkpoints at seeds 7, 13 (md5-authenticated) | none registered (D7.4 registers controls only against `recency` k=30 and `same_entity` k=30) | no block from this check |
| `same_entity_k30` | `PASS` | trained checkpoints at seeds 7, 13 (md5-authenticated) | `aligned_hist` vs S(i) of same_entity k=30 → fired | no block from this check |
| `same_entity_unr` | `PASS` | trained checkpoints at seeds 7, 13 (md5-authenticated) | none registered (D7.4 registers controls only against `recency` k=30 and `same_entity` k=30) | no block from this check |

**Disclosed gap in the certification design.** `same_entity_k0`, `same_entity_unr` have no separately registered positive control: D7.4 registers `full` against `recency_k30`'s S(i) and `aligned_hist` against `same_entity_k30`'s S(i) only. Their sensitivity evidence is inherited from the same-entity construction at k = 30 and is not independent of it. That is recorded here rather than treated as satisfied.

## 3. Decision rule (as registered, before any result was read)

* **Families.** one registered family per candidate configuration, with exactly three members: {primary contrast = candidate - its named matched control on the `all` slice, death-over non-inferiority gate = candidate - mlp on the `death` slice, chase non-inferiority gate = candidate - mlp on the `chase` slice}. The map below is the registration: it is not a template. Every reference is a config id from the `configurations` list, and every family member names the slice it is evaluated on.
* **Gate reference.** both non-inferiority gates are taken against mlp for every candidate, whatever the candidate's primary control is (D10.6), so the gate asks "does this arm lose ground against the shared control on death overs / in chases", a question that is comparable across every family.
* **Multiplicity.** Holm operates WITHIN each three-member family, separately for each estimand / checkpoint readout. It is never pooled across the 15 families and never applied across the k search. with three sorted raw p-values p_(1) <= p_(2) <= p_(3), the adjusted value at rank r is min(1, max over j <= r of (4 - j) * p_(j)) — the step-down stopping rule, which makes the adjusted values monotone in rank. Ties: stable in the registered member order (primary, death, chase). Missing member: an unavailable member (a run that did not finish, a slice with no rows) gets a non-rejecting placeholder (adjusted p = 1) and the family CANNOT pass.
* **Raw p.** p_raw = min(1, 2 * min[P(d* - t <= 0), P(d* - t >= 0)]) over the bootstrap delta draws, with t = 0 for superiority and t = +0.002 for non-inferiority
* **Non-inferiority.** margin 0.002 log loss on ['death', 'chase'] against `mlp`; a candidate is non-inferior on a gate slice when U95, the 97.5th percentile of the paired harm draws of (candidate - mlp), is STRICTLY below +0.002 log loss. U95 = 0.002 exactly does not pass.
* **Bootstrap.** 2000 replicates, rng seed 29, ball_weighted, tournament blocks, not matches, blocks from `scripts/sim_eval/eval_statistics.load_competition_clusters` over `data/t20s_json`, contract `tournament_time_block_v1` with `MAX_EVENT_GAP_DAYS = 120`. Fewer than 10 blocks on a slice is descriptive. block ids are what reach the cluster estimators, never match ids.
* **Estimands.** (i) each registered seed checkpoint separately with paired block-only uncertainty; (ii) the arithmetic across-seed mean under joint seed-and-block resampling. Reported separately, never pooled, never selected between. No best-seed selection and no averaging of CI endpoints.
* **k selection.** tolerance 0.002 over the registered sweep [0, 6, 12, 30, 'unr']: ANY k in the registered sweep {0, 6, 12, 30, unr} may win, unr included: choose the k with the best (lowest) two-seed mean validation log loss; if no k beats k = 30 by more than 0.002, keep 30. The rule is not restricted to k smaller than 30 — a larger window or the unrestricted innings wins on the same terms as a smaller one.
* **Intervals.** every per-arm or per-rank interval is labelled "rank-local percentile interval; not simultaneous; not the rejection rule", as the stage 1 report did. The rejection rule is the Holm-adjusted family above, never an interval read off a table.
* **Not imported.** stage 1's +-0.007 parity margin and its parity classification are NOT imported; this stage's only margin is +0.002 on log loss.

Registered validation totals, asserted before anything was differenced: 124292 rows / 545 matches / 47 tournament blocks / 0 unmapped, block lookup sha256 `819a860265cf…` over `data/t20s_json`. Row alignment rule: identical y and innings_id ordering across every compared arm and seed, asserted before differencing, and correspondence to the pinned parquet's ROW ORDER rather than a join or a sort by innings id.

### Slice predicates (frozen before computation, D10.7)

| slice | role | predicate | rows | matches | blocks | descriptive (<10 blocks) |
|---|---|---|---|---|---|---|
| `all` | primary | `every validation row` | 124292 | 545 | 47 | no |
| `death` | gate | `is_death_overs == 1` | 20301 | 514 | 47 | no |
| `chase` | gate | `chase_target > 0` | 57994 | 533 | 47 | no |
| `powerplay` | exploratory | `is_powerplay == 1` | 40594 | 545 | 47 | no |
| `middle` | exploratory | `is_middle_overs == 1` | 63397 | 541 | 47 | no |
| `innings_1` | exploratory | `inning_idx == 1` | 66252 | 545 | 47 | no |
| `innings_2` | exploratory | `inning_idx == 2` | 57994 | 533 | 47 | no |
| `thin_pair` | exploratory | **unavailable** | — | — | — | — |

`thin_pair` is reported **unavailable**: thin_pair needs per-row batter and bowler EXPOSURE columns (career or as-of ball counts) and a threshold. The i7 ball frame carries no exposure column: `batter_balls_faced` and `bowler_balls_in_innings` are within-innings counters, not exposure, and the eval kit that carries `train_balls_batting` / `train_balls_bowling` is off for this stage (`eval_kit: none`, `--no-kit`). D10.7 requires the slice to name its exposure columns and threshold or be reported unavailable, so it is reported unavailable rather than invented. To enable it, register `statistics.slice_predicates.thin_pair.{exposure_columns, threshold}` in the config and re-run.

## 4. Results — the 15 registered families, Holm step-down

Each family has exactly three members: the registered primary contrast on `all`, and `candidate − mlp` non-inferiority gates on `death` and on `chase`. Holm runs **within** a family and **separately per estimand**; it is never pooled across the 15 families and never applied across the k search, and no correction across families is claimed. Rank-local intervals are rank-local percentile interval; not simultaneous; not the rejection rule: classification is never read off one.

Rejection requires all three of the step-down rule, the favourable direction, and the relevant strict upper-bound condition (U95 < 0 for a primary, U95 < 0.002 for a gate). An empty bootstrap tail is reported as `< 0.001` with a resolution flag, never as exact zero.

**full** — seed 7 (estimand i), Holm group `family_full`, screen status `SCREEN_NOT_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `full - mlp` | `all` | 0.0 | -0.00115 | [-0.00225, -0.00013] | -0.00013 | 0.0260 | 0.0520 | 2 | [-0.00252, +0.00002] | no | `SCREEN_NOT_PASS` |
| death_gate | `full - mlp` | `death` | 0.002 | +0.00337 | [+0.00072, +0.00588] | +0.00588 | 0.2870 | 0.2870 | 3 | [+0.00072, +0.00588] | no | `SCREEN_NOT_PASS` |
| chase_gate | `full - mlp` | `chase` | 0.002 | -0.00012 | [-0.00176, +0.00144] | +0.00144 | 0.0120 | 0.0360 | 1 | [-0.00226, +0.00185] | yes | `SCREEN_PASS` |

**full** — seed 13 (estimand i), Holm group `family_full`, screen status `SCREEN_NOT_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `full - mlp` | `all` | 0.0 | +0.00031 | [-0.00083, +0.00134] | +0.00134 | 0.5700 | 1.0000 | 2 | [-0.00104, +0.00147] | no | `SCREEN_NOT_PASS` |
| death_gate | `full - mlp` | `death` | 0.002 | +0.00760 | [+0.00499, +0.00998] | +0.00998 | < 0.001 | 0.0030 | 1 | [+0.00433, +0.01048] | no | `SCREEN_NOT_PASS` |
| chase_gate | `full - mlp` | `chase` | 0.002 | +0.00181 | [+0.00007, +0.00341] | +0.00341 | 0.8570 | 1.0000 | 3 | [+0.00007, +0.00341] | no | `SCREEN_NOT_PASS` |

**full** — seed mean (estimand ii), Holm group `family_full`, screen status `SCREEN_NOT_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `full - mlp` | `all` | 0.0 | -0.00042 | [-0.00195, +0.00100] | +0.00100 | 0.5410 | 0.6000 | 3 | [-0.00195, +0.00100] | no | `SCREEN_NOT_PASS` |
| death_gate | `full - mlp` | `death` | 0.002 | +0.00548 | [+0.00168, +0.00900] | +0.00900 | 0.0730 | 0.2190 | 1 | [+0.00090, +0.00981] | no | `SCREEN_NOT_PASS` |
| chase_gate | `full - mlp` | `chase` | 0.002 | +0.00085 | [-0.00135, +0.00293] | +0.00293 | 0.3000 | 0.6000 | 2 | [-0.00162, +0.00314] | no | `SCREEN_NOT_PASS` |

**fixed_decay** — seed 7 (estimand i), Holm group `family_fixed_decay`, screen status `SCREEN_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `fixed_decay - mlp` | `all` | 0.0 | -0.00498 | [-0.00617, -0.00415] | -0.00415 | < 0.001 | 0.0030 | 1 | [-0.00646, -0.00395] | yes | `SCREEN_PASS` |
| death_gate | `fixed_decay - mlp` | `death` | 0.002 | -0.00079 | [-0.00379, +0.00156] | +0.00156 | 0.0180 | 0.0180 | 3 | [-0.00379, +0.00156] | yes | `SCREEN_PASS` |
| chase_gate | `fixed_decay - mlp` | `chase` | 0.002 | -0.00432 | [-0.00566, -0.00310] | -0.00310 | < 0.001 | 0.0030 | 2 | [-0.00586, -0.00296] | yes | `SCREEN_PASS` |

**fixed_decay** — seed 13 (estimand i), Holm group `family_fixed_decay`, screen status `SCREEN_NOT_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `fixed_decay - mlp` | `all` | 0.0 | -0.00201 | [-0.00319, -0.00102] | -0.00102 | < 0.001 | 0.0030 | 1 | [-0.00350, -0.00077] | yes | `SCREEN_PASS` |
| death_gate | `fixed_decay - mlp` | `death` | 0.002 | +0.00423 | [+0.00149, +0.00642] | +0.00642 | 0.1070 | 0.1070 | 3 | [+0.00149, +0.00642] | no | `SCREEN_NOT_PASS` |
| chase_gate | `fixed_decay - mlp` | `chase` | 0.002 | -0.00095 | [-0.00266, +0.00062] | +0.00062 | < 0.001 | 0.0030 | 2 | [-0.00291, +0.00074] | yes | `SCREEN_PASS` |

**fixed_decay** — seed mean (estimand ii), Holm group `family_fixed_decay`, screen status `SCREEN_NOT_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `fixed_decay - mlp` | `all` | 0.0 | -0.00349 | [-0.00576, -0.00136] | -0.00136 | < 0.001 | 0.0030 | 1 | [-0.00613, -0.00116] | yes | `SCREEN_PASS` |
| death_gate | `fixed_decay - mlp` | `death` | 0.002 | +0.00172 | [-0.00252, +0.00563] | +0.00563 | 0.8770 | 0.8770 | 3 | [-0.00252, +0.00563] | no | `SCREEN_NOT_PASS` |
| chase_gate | `fixed_decay - mlp` | `chase` | 0.002 | -0.00263 | [-0.00526, +0.00012] | +0.00012 | < 0.001 | 0.0030 | 2 | [-0.00547, +0.00042] | yes | `SCREEN_PASS` |

**fox** — seed 7 (estimand i), Holm group `family_fox`, screen status `SCREEN_NOT_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `fox - fixed_decay` | `all` | 0.0 | +0.00002 | [-0.00011, +0.00015] | +0.00015 | 0.7540 | 0.7540 | 3 | [-0.00011, +0.00015] | no | `SCREEN_NOT_PASS` |
| death_gate | `fox - mlp` | `death` | 0.002 | -0.00073 | [-0.00358, +0.00157] | +0.00157 | 0.0120 | 0.0240 | 2 | [-0.00410, +0.00178] | yes | `SCREEN_PASS` |
| chase_gate | `fox - mlp` | `chase` | 0.002 | -0.00440 | [-0.00578, -0.00318] | -0.00318 | < 0.001 | 0.0030 | 1 | [-0.00613, -0.00294] | yes | `SCREEN_PASS` |

**fox** — seed 13 (estimand i), Holm group `family_fox`, screen status `SCREEN_NOT_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `fox - fixed_decay` | `all` | 0.0 | -0.00011 | [-0.00027, +0.00004] | +0.00004 | 0.1460 | 0.2920 | 2 | [-0.00029, +0.00007] | no | `SCREEN_NOT_PASS` |
| death_gate | `fox - mlp` | `death` | 0.002 | +0.00390 | [+0.00119, +0.00605] | +0.00605 | 0.1550 | 0.2920 | 3 | [+0.00119, +0.00605] | no | `SCREEN_NOT_PASS` |
| chase_gate | `fox - mlp` | `chase` | 0.002 | -0.00123 | [-0.00299, +0.00042] | +0.00042 | < 0.001 | 0.0030 | 1 | [-0.00351, +0.00066] | yes | `SCREEN_PASS` |

**fox** — seed mean (estimand ii), Holm group `family_fox`, screen status `SCREEN_NOT_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `fox - fixed_decay` | `all` | 0.0 | -0.00005 | [-0.00022, +0.00011] | +0.00011 | 0.5560 | 1.0000 | 2 | [-0.00026, +0.00013] | no | `SCREEN_NOT_PASS` |
| death_gate | `fox - mlp` | `death` | 0.002 | +0.00159 | [-0.00238, +0.00527] | +0.00527 | 0.8120 | 1.0000 | 3 | [-0.00238, +0.00527] | no | `SCREEN_NOT_PASS` |
| chase_gate | `fox - mlp` | `chase` | 0.002 | -0.00282 | [-0.00538, -0.00011] | -0.00011 | < 0.001 | 0.0030 | 1 | [-0.00564, +0.00031] | yes | `SCREEN_PASS` |

**aligned_hist** — seed 7 (estimand i), Holm group `family_aligned_hist`, screen status `SCREEN_NOT_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `aligned_hist - full` | `all` | 0.0 | -0.00085 | [-0.00143, -0.00029] | -0.00029 | 0.0030 | 0.0060 | 2 | [-0.00151, -0.00021] | yes | `SCREEN_PASS` |
| death_gate | `aligned_hist - mlp` | `death` | 0.002 | +0.00368 | [+0.00097, +0.00701] | +0.00701 | 0.2500 | 0.2500 | 3 | [+0.00097, +0.00701] | no | `SCREEN_NOT_PASS` |
| chase_gate | `aligned_hist - mlp` | `chase` | 0.002 | -0.00080 | [-0.00243, +0.00075] | +0.00075 | < 0.001 | 0.0030 | 1 | [-0.00282, +0.00104] | yes | `SCREEN_PASS` |

**aligned_hist** — seed 13 (estimand i), Holm group `family_aligned_hist`, screen status `SCREEN_NOT_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `aligned_hist - full` | `all` | 0.0 | -0.00222 | [-0.00282, -0.00154] | -0.00154 | < 0.001 | 0.0030 | 1 | [-0.00294, -0.00141] | yes | `SCREEN_PASS` |
| death_gate | `aligned_hist - mlp` | `death` | 0.002 | +0.00429 | [+0.00136, +0.00723] | +0.00723 | 0.1200 | 0.1200 | 3 | [+0.00136, +0.00723] | no | `SCREEN_NOT_PASS` |
| chase_gate | `aligned_hist - mlp` | `chase` | 0.002 | -0.00054 | [-0.00230, +0.00110] | +0.00110 | < 0.001 | 0.0030 | 2 | [-0.00259, +0.00127] | yes | `SCREEN_PASS` |

**aligned_hist** — seed mean (estimand ii), Holm group `family_aligned_hist`, screen status `SCREEN_NOT_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `aligned_hist - full` | `all` | 0.0 | -0.00154 | [-0.00263, -0.00051] | -0.00051 | 0.0010 | 0.0030 | 1 | [-0.00279, -0.00037] | yes | `SCREEN_PASS` |
| death_gate | `aligned_hist - mlp` | `death` | 0.002 | +0.00399 | [+0.00127, +0.00707] | +0.00707 | 0.1610 | 0.1610 | 3 | [+0.00127, +0.00707] | no | `SCREEN_NOT_PASS` |
| chase_gate | `aligned_hist - mlp` | `chase` | 0.002 | -0.00067 | [-0.00237, +0.00098] | +0.00098 | 0.0010 | 0.0030 | 2 | [-0.00265, +0.00126] | yes | `SCREEN_PASS` |

**aligned_hist_rf** — seed 7 (estimand i), Holm group `family_aligned_hist_rf`, screen status `SCREEN_NOT_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `aligned_hist_rf - aligned_hist` | `all` | 0.0 | -0.00028 | [-0.00088, +0.00023] | +0.00023 | 0.3290 | 0.6580 | 2 | [-0.00095, +0.00030] | no | `SCREEN_NOT_PASS` |
| death_gate | `aligned_hist_rf - mlp` | `death` | 0.002 | +0.00138 | [-0.00099, +0.00350] | +0.00350 | 0.5480 | 0.6580 | 3 | [-0.00099, +0.00350] | no | `SCREEN_NOT_PASS` |
| chase_gate | `aligned_hist_rf - mlp` | `chase` | 0.002 | -0.00130 | [-0.00283, +0.00015] | +0.00015 | < 0.001 | 0.0030 | 1 | [-0.00319, +0.00044] | yes | `SCREEN_PASS` |

**aligned_hist_rf** — seed 13 (estimand i), Holm group `family_aligned_hist_rf`, screen status `SCREEN_NOT_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `aligned_hist_rf - aligned_hist` | `all` | 0.0 | +0.00146 | [+0.00087, +0.00201] | +0.00201 | < 0.001 | 0.0030 | 1 | [+0.00074, +0.00212] | no | `SCREEN_NOT_PASS` |
| death_gate | `aligned_hist_rf - mlp` | `death` | 0.002 | +0.00584 | [+0.00366, +0.00848] | +0.00848 | 0.0010 | 0.0030 | 2 | [+0.00329, +0.00907] | no | `SCREEN_NOT_PASS` |
| chase_gate | `aligned_hist_rf - mlp` | `chase` | 0.002 | +0.00064 | [-0.00106, +0.00216] | +0.00216 | 0.0800 | 0.0800 | 3 | [-0.00106, +0.00216] | no | `SCREEN_NOT_PASS` |

**aligned_hist_rf** — seed mean (estimand ii), Holm group `family_aligned_hist_rf`, screen status `SCREEN_NOT_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `aligned_hist_rf - aligned_hist` | `all` | 0.0 | +0.00059 | [-0.00064, +0.00184] | +0.00184 | 0.4480 | 0.8020 | 3 | [-0.00064, +0.00184] | no | `SCREEN_NOT_PASS` |
| death_gate | `aligned_hist_rf - mlp` | `death` | 0.002 | +0.00361 | [+0.00003, +0.00732] | +0.00732 | 0.4010 | 0.8020 | 2 | [-0.00058, +0.00778] | no | `SCREEN_NOT_PASS` |
| chase_gate | `aligned_hist_rf - mlp` | `chase` | 0.002 | -0.00033 | [-0.00235, +0.00165] | +0.00165 | 0.0210 | 0.0630 | 1 | [-0.00278, +0.00208] | no | `SCREEN_NOT_PASS` |

Registered note on `aligned_hist_rf`: the primary contrast measures the relay-free wiring PLUS the key construction, not the wiring alone (known_asymmetries relay_free_keys_carry_own_outcome).

**recency_k30** — seed 7 (estimand i), Holm group `family_recency_k30`, screen status `SCREEN_NOT_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `recency_k30 - mlp` | `all` | 0.0 | -0.00068 | [-0.00181, +0.00019] | +0.00019 | 0.1420 | 0.2840 | 2 | [-0.00198, +0.00037] | no | `SCREEN_NOT_PASS` |
| death_gate | `recency_k30 - mlp` | `death` | 0.002 | +0.00395 | [+0.00087, +0.00621] | +0.00621 | 0.1820 | 0.2840 | 3 | [+0.00087, +0.00621] | no | `SCREEN_NOT_PASS` |
| chase_gate | `recency_k30 - mlp` | `chase` | 0.002 | -0.00021 | [-0.00181, +0.00129] | +0.00129 | < 0.001 | 0.0030 | 1 | [-0.00220, +0.00148] | yes | `SCREEN_PASS` |

**recency_k30** — seed 13 (estimand i), Holm group `family_recency_k30`, screen status `SCREEN_NOT_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `recency_k30 - mlp` | `all` | 0.0 | +0.00013 | [-0.00092, +0.00099] | +0.00099 | 0.7830 | 0.7830 | 3 | [-0.00092, +0.00099] | no | `SCREEN_NOT_PASS` |
| death_gate | `recency_k30 - mlp` | `death` | 0.002 | +0.00729 | [+0.00481, +0.00969] | +0.00969 | < 0.001 | 0.0030 | 1 | [+0.00421, +0.01026] | no | `SCREEN_NOT_PASS` |
| chase_gate | `recency_k30 - mlp` | `chase` | 0.002 | +0.00118 | [-0.00028, +0.00245] | +0.00245 | 0.2130 | 0.4260 | 2 | [-0.00070, +0.00258] | no | `SCREEN_NOT_PASS` |

**recency_k30** — seed mean (estimand ii), Holm group `family_recency_k30`, screen status `SCREEN_NOT_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `recency_k30 - mlp` | `all` | 0.0 | -0.00027 | [-0.00157, +0.00077] | +0.00077 | 0.5910 | 0.5910 | 3 | [-0.00157, +0.00077] | no | `SCREEN_NOT_PASS` |
| death_gate | `recency_k30 - mlp` | `death` | 0.002 | +0.00562 | [+0.00192, +0.00876] | +0.00876 | 0.0550 | 0.1650 | 1 | [+0.00087, +0.00956] | no | `SCREEN_NOT_PASS` |
| chase_gate | `recency_k30 - mlp` | `chase` | 0.002 | +0.00049 | [-0.00144, +0.00215] | +0.00215 | 0.0800 | 0.1650 | 2 | [-0.00169, +0.00235] | no | `SCREEN_NOT_PASS` |

**same_entity_k30** — seed 7 (estimand i), Holm group `family_same_entity_k30`, screen status `SCREEN_NOT_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `same_entity_k30 - recency_k30` | `all` | 0.0 | -0.00058 | [-0.00166, +0.00054] | +0.00054 | 0.3490 | 0.3840 | 3 | [-0.00166, +0.00054] | no | `SCREEN_NOT_PASS` |
| death_gate | `same_entity_k30 - mlp` | `death` | 0.002 | +0.00396 | [+0.00119, +0.00719] | +0.00719 | 0.1920 | 0.3840 | 2 | [+0.00068, +0.00769] | no | `SCREEN_NOT_PASS` |
| chase_gate | `same_entity_k30 - mlp` | `chase` | 0.002 | -0.00080 | [-0.00250, +0.00121] | +0.00121 | 0.0030 | 0.0090 | 1 | [-0.00289, +0.00149] | yes | `SCREEN_PASS` |

**same_entity_k30** — seed 13 (estimand i), Holm group `family_same_entity_k30`, screen status `SCREEN_NOT_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `same_entity_k30 - recency_k30` | `all` | 0.0 | -0.00176 | [-0.00249, -0.00082] | -0.00082 | 0.0010 | 0.0030 | 1 | [-0.00264, -0.00064] | yes | `SCREEN_PASS` |
| death_gate | `same_entity_k30 - mlp` | `death` | 0.002 | +0.00485 | [+0.00243, +0.00773] | +0.00773 | 0.0240 | 0.0240 | 3 | [+0.00243, +0.00773] | no | `SCREEN_NOT_PASS` |
| chase_gate | `same_entity_k30 - mlp` | `chase` | 0.002 | -0.00089 | [-0.00235, +0.00064] | +0.00064 | 0.0010 | 0.0030 | 2 | [-0.00258, +0.00087] | yes | `SCREEN_PASS` |

**same_entity_k30** — seed mean (estimand ii), Holm group `family_same_entity_k30`, screen status `SCREEN_NOT_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `same_entity_k30 - recency_k30` | `all` | 0.0 | -0.00117 | [-0.00226, +0.00021] | +0.00021 | 0.1040 | 0.1840 | 3 | [-0.00226, +0.00021] | no | `SCREEN_NOT_PASS` |
| death_gate | `same_entity_k30 - mlp` | `death` | 0.002 | +0.00440 | [+0.00155, +0.00736] | +0.00736 | 0.0920 | 0.1840 | 2 | [+0.00108, +0.00791] | no | `SCREEN_NOT_PASS` |
| chase_gate | `same_entity_k30 - mlp` | `chase` | 0.002 | -0.00084 | [-0.00248, +0.00096] | +0.00096 | 0.0020 | 0.0060 | 1 | [-0.00280, +0.00127] | yes | `SCREEN_PASS` |

Registered note on `same_entity_k30`: the primary contrast is "ownership plus alignment beyond recency" (known_asymmetries same_entity_vs_recency_history_input).

**same_entity_unr** — seed 7 (estimand i), Holm group `family_same_entity_unr`, screen status `SCREEN_NOT_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `same_entity_unr - aligned_hist_rf` | `all` | 0.0 | +0.00034 | [-0.00024, +0.00104] | +0.00104 | 0.2820 | 0.5640 | 2 | [-0.00033, +0.00118] | no | `SCREEN_NOT_PASS` |
| death_gate | `same_entity_unr - mlp` | `death` | 0.002 | +0.00227 | [-0.00053, +0.00482] | +0.00482 | 0.8350 | 0.8350 | 3 | [-0.00053, +0.00482] | no | `SCREEN_NOT_PASS` |
| chase_gate | `same_entity_unr - mlp` | `chase` | 0.002 | -0.00141 | [-0.00294, +0.00018] | +0.00018 | < 0.001 | 0.0030 | 1 | [-0.00339, +0.00063] | yes | `SCREEN_PASS` |

**same_entity_unr** — seed 13 (estimand i), Holm group `family_same_entity_unr`, screen status `SCREEN_NOT_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `same_entity_unr - aligned_hist_rf` | `all` | 0.0 | -0.00136 | [-0.00202, -0.00049] | -0.00049 | 0.0050 | 0.0100 | 2 | [-0.00212, -0.00032] | yes | `SCREEN_PASS` |
| death_gate | `same_entity_unr - mlp` | `death` | 0.002 | +0.00414 | [+0.00188, +0.00678] | +0.00678 | 0.0590 | 0.0590 | 3 | [+0.00188, +0.00678] | no | `SCREEN_NOT_PASS` |
| chase_gate | `same_entity_unr - mlp` | `chase` | 0.002 | -0.00158 | [-0.00315, +0.00020] | +0.00020 | < 0.001 | 0.0030 | 1 | [-0.00365, +0.00064] | yes | `SCREEN_PASS` |

**same_entity_unr** — seed mean (estimand ii), Holm group `family_same_entity_unr`, screen status `SCREEN_NOT_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `same_entity_unr - aligned_hist_rf` | `all` | 0.0 | -0.00051 | [-0.00181, +0.00080] | +0.00080 | 0.5170 | 0.7400 | 3 | [-0.00181, +0.00080] | no | `SCREEN_NOT_PASS` |
| death_gate | `same_entity_unr - mlp` | `death` | 0.002 | +0.00320 | [+0.00026, +0.00592] | +0.00592 | 0.3700 | 0.7400 | 2 | [-0.00025, +0.00644] | no | `SCREEN_NOT_PASS` |
| chase_gate | `same_entity_unr - mlp` | `chase` | 0.002 | -0.00150 | [-0.00303, +0.00020] | +0.00020 | < 0.001 | 0.0030 | 1 | [-0.00349, +0.00065] | yes | `SCREEN_PASS` |

Registered note on `same_entity_unr`: both endpoints are relay-free with matched key construction, so this is the mask alone given aligned inputs.

**same_entity_k0** — seed 7 (estimand i), Holm group `family_same_entity_k0`, screen status `SCREEN_NOT_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `same_entity_k0 - mlp` | `all` | 0.0 | +0.00120 | [+0.00008, +0.00241] | +0.00241 | 0.0350 | 0.0700 | 2 | [-0.00004, +0.00265] | no | `SCREEN_NOT_PASS` |
| death_gate | `same_entity_k0 - mlp` | `death` | 0.002 | +0.00582 | [+0.00323, +0.00889] | +0.00889 | 0.0060 | 0.0180 | 1 | [+0.00274, +0.01000] | no | `SCREEN_NOT_PASS` |
| chase_gate | `same_entity_k0 - mlp` | `chase` | 0.002 | +0.00160 | [+0.00022, +0.00345] | +0.00345 | 0.6080 | 0.6080 | 3 | [+0.00022, +0.00345] | no | `SCREEN_NOT_PASS` |

**same_entity_k0** — seed 13 (estimand i), Holm group `family_same_entity_k0`, screen status `SCREEN_NOT_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `same_entity_k0 - mlp` | `all` | 0.0 | +0.00084 | [-0.00024, +0.00207] | +0.00207 | 0.1500 | 0.3000 | 2 | [-0.00040, +0.00226] | no | `SCREEN_NOT_PASS` |
| death_gate | `same_entity_k0 - mlp` | `death` | 0.002 | +0.00637 | [+0.00409, +0.00946] | +0.00946 | 0.0010 | 0.0030 | 1 | [+0.00353, +0.01032] | no | `SCREEN_NOT_PASS` |
| chase_gate | `same_entity_k0 - mlp` | `chase` | 0.002 | +0.00166 | [+0.00029, +0.00337] | +0.00337 | 0.6360 | 0.6360 | 3 | [+0.00029, +0.00337] | no | `SCREEN_NOT_PASS` |

**same_entity_k0** — seed mean (estimand ii), Holm group `family_same_entity_k0`, screen status `SCREEN_NOT_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `same_entity_k0 - mlp` | `all` | 0.0 | +0.00102 | [-0.00011, +0.00225] | +0.00225 | 0.0840 | 0.1680 | 2 | [-0.00027, +0.00239] | no | `SCREEN_NOT_PASS` |
| death_gate | `same_entity_k0 - mlp` | `death` | 0.002 | +0.00610 | [+0.00354, +0.00910] | +0.00910 | 0.0030 | 0.0090 | 1 | [+0.00308, +0.00994] | no | `SCREEN_NOT_PASS` |
| chase_gate | `same_entity_k0 - mlp` | `chase` | 0.002 | +0.00163 | [+0.00023, +0.00337] | +0.00337 | 0.6650 | 0.6650 | 3 | [+0.00023, +0.00337] | no | `SCREEN_NOT_PASS` |

Registered note on `same_entity_k0`: no recency control is registered at this k, so the primary contrast is against the shared control and the k sweep carries the rest.

**same_entity_k6** — seed 7 (estimand i), Holm group `family_same_entity_k6`, screen status `SCREEN_NOT_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `same_entity_k6 - mlp` | `all` | 0.0 | -0.00014 | [-0.00128, +0.00106] | +0.00106 | 0.8120 | 0.8120 | 3 | [-0.00128, +0.00106] | no | `SCREEN_NOT_PASS` |
| death_gate | `same_entity_k6 - mlp` | `death` | 0.002 | +0.00448 | [+0.00165, +0.00792] | +0.00792 | 0.0990 | 0.2820 | 2 | [+0.00130, +0.00855] | no | `SCREEN_NOT_PASS` |
| chase_gate | `same_entity_k6 - mlp` | `chase` | 0.002 | +0.00046 | [-0.00106, +0.00222] | +0.00222 | 0.0940 | 0.2820 | 1 | [-0.00143, +0.00252] | no | `SCREEN_NOT_PASS` |

**same_entity_k6** — seed 13 (estimand i), Holm group `family_same_entity_k6`, screen status `SCREEN_NOT_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `same_entity_k6 - mlp` | `all` | 0.0 | +0.00021 | [-0.00121, +0.00164] | +0.00164 | 0.7990 | 0.8620 | 3 | [-0.00121, +0.00164] | no | `SCREEN_NOT_PASS` |
| death_gate | `same_entity_k6 - mlp` | `death` | 0.002 | +0.00713 | [+0.00452, +0.01008] | +0.01008 | < 0.001 | 0.0030 | 1 | [+0.00390, +0.01104] | no | `SCREEN_NOT_PASS` |
| chase_gate | `same_entity_k6 - mlp` | `chase` | 0.002 | +0.00120 | [-0.00066, +0.00316] | +0.00316 | 0.4310 | 0.8620 | 2 | [-0.00097, +0.00348] | no | `SCREEN_NOT_PASS` |

**same_entity_k6** — seed mean (estimand ii), Holm group `family_same_entity_k6`, screen status `SCREEN_NOT_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `same_entity_k6 - mlp` | `all` | 0.0 | +0.00004 | [-0.00123, +0.00138] | +0.00138 | 0.9610 | 0.9610 | 3 | [-0.00123, +0.00138] | no | `SCREEN_NOT_PASS` |
| death_gate | `same_entity_k6 - mlp` | `death` | 0.002 | +0.00580 | [+0.00232, +0.00931] | +0.00931 | 0.0300 | 0.0900 | 1 | [+0.00152, +0.00998] | no | `SCREEN_NOT_PASS` |
| chase_gate | `same_entity_k6 - mlp` | `chase` | 0.002 | +0.00083 | [-0.00092, +0.00283] | +0.00283 | 0.2400 | 0.4800 | 2 | [-0.00120, +0.00308] | no | `SCREEN_NOT_PASS` |

Registered note on `same_entity_k6`: no recency control is registered at this k.

**same_entity_k12** — seed 7 (estimand i), Holm group `family_same_entity_k12`, screen status `SCREEN_NOT_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `same_entity_k12 - mlp` | `all` | 0.0 | -0.00083 | [-0.00223, +0.00067] | +0.00067 | 0.2600 | 0.5200 | 2 | [-0.00250, +0.00087] | no | `SCREEN_NOT_PASS` |
| death_gate | `same_entity_k12 - mlp` | `death` | 0.002 | +0.00374 | [+0.00093, +0.00720] | +0.00720 | 0.2690 | 0.5200 | 3 | [+0.00093, +0.00720] | no | `SCREEN_NOT_PASS` |
| chase_gate | `same_entity_k12 - mlp` | `chase` | 0.002 | -0.00012 | [-0.00161, +0.00170] | +0.00170 | 0.0260 | 0.0780 | 1 | [-0.00204, +0.00207] | no | `SCREEN_NOT_PASS` |

**same_entity_k12** — seed 13 (estimand i), Holm group `family_same_entity_k12`, screen status `SCREEN_NOT_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `same_entity_k12 - mlp` | `all` | 0.0 | -0.00104 | [-0.00241, +0.00040] | +0.00040 | 0.1510 | 0.1510 | 3 | [-0.00241, +0.00040] | no | `SCREEN_NOT_PASS` |
| death_gate | `same_entity_k12 - mlp` | `death` | 0.002 | +0.00482 | [+0.00217, +0.00819] | +0.00819 | 0.0400 | 0.0800 | 2 | [+0.00188, +0.00868] | no | `SCREEN_NOT_PASS` |
| chase_gate | `same_entity_k12 - mlp` | `chase` | 0.002 | -0.00007 | [-0.00169, +0.00162] | +0.00162 | 0.0240 | 0.0720 | 1 | [-0.00214, +0.00209] | no | `SCREEN_NOT_PASS` |

**same_entity_k12** — seed mean (estimand ii), Holm group `family_same_entity_k12`, screen status `SCREEN_NOT_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `same_entity_k12 - mlp` | `all` | 0.0 | -0.00093 | [-0.00236, +0.00052] | +0.00052 | 0.2100 | 0.2440 | 3 | [-0.00236, +0.00052] | no | `SCREEN_NOT_PASS` |
| death_gate | `same_entity_k12 - mlp` | `death` | 0.002 | +0.00428 | [+0.00145, +0.00771] | +0.00771 | 0.1220 | 0.2440 | 2 | [+0.00115, +0.00831] | no | `SCREEN_NOT_PASS` |
| chase_gate | `same_entity_k12 - mlp` | `chase` | 0.002 | -0.00010 | [-0.00174, +0.00167] | +0.00167 | 0.0280 | 0.0840 | 1 | [-0.00206, +0.00215] | no | `SCREEN_NOT_PASS` |

Registered note on `same_entity_k12`: no recency control is registered at this k.

**lstm** — seed 7 (estimand i), Holm group `family_lstm`, screen status `SCREEN_NOT_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `lstm - mlp` | `all` | 0.0 | +0.00190 | [-0.00201, +0.00809] | +0.00809 | 0.4840 | 1.0000 | 2 | [-0.00238, +0.00945] | no | `SCREEN_NOT_PASS` |
| death_gate | `lstm - mlp` | `death` | 0.002 | +0.00420 | [-0.00019, +0.01034] | +0.01034 | 0.3920 | 1.0000 | 1 | [-0.00088, +0.01212] | no | `SCREEN_NOT_PASS` |
| chase_gate | `lstm - mlp` | `chase` | 0.002 | +0.00209 | [-0.00179, +0.00796] | +0.00796 | 0.9370 | 1.0000 | 3 | [-0.00179, +0.00796] | no | `SCREEN_NOT_PASS` |

**lstm** — seed 13 (estimand i), Holm group `family_lstm`, screen status `SCREEN_NOT_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `lstm - mlp` | `all` | 0.0 | +0.00347 | [-0.00094, +0.01031] | +0.01031 | 0.1580 | 0.3160 | 2 | [-0.00131, +0.01187] | no | `SCREEN_NOT_PASS` |
| death_gate | `lstm - mlp` | `death` | 0.002 | +0.00607 | [+0.00194, +0.01111] | +0.01111 | 0.0550 | 0.1650 | 1 | [+0.00150, +0.01240] | no | `SCREEN_NOT_PASS` |
| chase_gate | `lstm - mlp` | `chase` | 0.002 | +0.00404 | [-0.00027, +0.01059] | +0.01059 | 0.4860 | 0.4860 | 3 | [-0.00027, +0.01059] | no | `SCREEN_NOT_PASS` |

**lstm** — seed mean (estimand ii), Holm group `family_lstm`, screen status `SCREEN_NOT_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `lstm - mlp` | `all` | 0.0 | +0.00269 | [-0.00153, +0.00934] | +0.00934 | 0.3090 | 0.6180 | 2 | [-0.00198, +0.01040] | no | `SCREEN_NOT_PASS` |
| death_gate | `lstm - mlp` | `death` | 0.002 | +0.00514 | [+0.00069, +0.01053] | +0.01053 | 0.1870 | 0.5610 | 1 | [-0.00036, +0.01210] | no | `SCREEN_NOT_PASS` |
| chase_gate | `lstm - mlp` | `chase` | 0.002 | +0.00306 | [-0.00114, +0.00952] | +0.00952 | 0.7640 | 0.7640 | 3 | [-0.00114, +0.00952] | no | `SCREEN_NOT_PASS` |

**xlstm** — seed 7 (estimand i), Holm group `family_xlstm`, screen status `SCREEN_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `xlstm - mlp` | `all` | 0.0 | -0.00562 | [-0.00684, -0.00474] | -0.00474 | < 0.001 | 0.0030 | 1 | [-0.00716, -0.00451] | yes | `SCREEN_PASS` |
| death_gate | `xlstm - mlp` | `death` | 0.002 | -0.00276 | [-0.00601, +0.00003] | +0.00003 | 0.0010 | 0.0030 | 2 | [-0.00648, +0.00048] | yes | `SCREEN_PASS` |
| chase_gate | `xlstm - mlp` | `chase` | 0.002 | -0.00453 | [-0.00582, -0.00339] | -0.00339 | < 0.001 | 0.0030 | 3 | [-0.00582, -0.00339] | yes | `SCREEN_PASS` |

**xlstm** — seed 13 (estimand i), Holm group `family_xlstm`, screen status `SCREEN_NOT_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `xlstm - mlp` | `all` | 0.0 | -0.00375 | [-0.00488, -0.00288] | -0.00288 | < 0.001 | 0.0030 | 1 | [-0.00521, -0.00265] | yes | `SCREEN_PASS` |
| death_gate | `xlstm - mlp` | `death` | 0.002 | +0.00096 | [-0.00165, +0.00309] | +0.00309 | 0.3630 | 0.3630 | 3 | [-0.00165, +0.00309] | no | `SCREEN_NOT_PASS` |
| chase_gate | `xlstm - mlp` | `chase` | 0.002 | -0.00304 | [-0.00428, -0.00192] | -0.00192 | < 0.001 | 0.0030 | 2 | [-0.00444, -0.00179] | yes | `SCREEN_PASS` |

**xlstm** — seed mean (estimand ii), Holm group `family_xlstm`, screen status `SCREEN_NOT_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `xlstm - mlp` | `all` | 0.0 | -0.00468 | [-0.00636, -0.00316] | -0.00316 | < 0.001 | 0.0030 | 1 | [-0.00678, -0.00294] | yes | `SCREEN_PASS` |
| death_gate | `xlstm - mlp` | `death` | 0.002 | -0.00090 | [-0.00467, +0.00246] | +0.00246 | 0.1000 | 0.1000 | 3 | [-0.00467, +0.00246] | no | `SCREEN_NOT_PASS` |
| chase_gate | `xlstm - mlp` | `chase` | 0.002 | -0.00378 | [-0.00544, -0.00229] | -0.00229 | < 0.001 | 0.0030 | 2 | [-0.00561, -0.00205] | yes | `SCREEN_PASS` |

**residual_mlp** — seed 7 (estimand i), Holm group `family_residual_mlp`, screen status `SCREEN_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `residual_mlp - mlp` | `all` | 0.0 | -0.00567 | [-0.00679, -0.00453] | -0.00453 | < 0.001 | 0.0030 | 1 | [-0.00707, -0.00430] | yes | `SCREEN_PASS` |
| death_gate | `residual_mlp - mlp` | `death` | 0.002 | -0.00436 | [-0.00730, -0.00166] | -0.00166 | < 0.001 | 0.0030 | 2 | [-0.00772, -0.00110] | yes | `SCREEN_PASS` |
| chase_gate | `residual_mlp - mlp` | `chase` | 0.002 | -0.00435 | [-0.00649, -0.00229] | -0.00229 | < 0.001 | 0.0030 | 3 | [-0.00649, -0.00229] | yes | `SCREEN_PASS` |

**residual_mlp** — seed 13 (estimand i), Holm group `family_residual_mlp`, screen status `SCREEN_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `residual_mlp - mlp` | `all` | 0.0 | -0.00538 | [-0.00654, -0.00420] | -0.00420 | < 0.001 | 0.0030 | 1 | [-0.00673, -0.00384] | yes | `SCREEN_PASS` |
| death_gate | `residual_mlp - mlp` | `death` | 0.002 | -0.00277 | [-0.00561, -0.00012] | -0.00012 | 0.0020 | 0.0030 | 3 | [-0.00561, -0.00012] | yes | `SCREEN_PASS` |
| chase_gate | `residual_mlp - mlp` | `chase` | 0.002 | -0.00396 | [-0.00625, -0.00174] | -0.00174 | < 0.001 | 0.0030 | 2 | [-0.00658, -0.00150] | yes | `SCREEN_PASS` |

**residual_mlp** — seed mean (estimand ii), Holm group `family_residual_mlp`, screen status `SCREEN_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `residual_mlp - mlp` | `all` | 0.0 | -0.00552 | [-0.00671, -0.00435] | -0.00435 | < 0.001 | 0.0030 | 1 | [-0.00697, -0.00412] | yes | `SCREEN_PASS` |
| death_gate | `residual_mlp - mlp` | `death` | 0.002 | -0.00357 | [-0.00659, -0.00072] | -0.00072 | < 0.001 | 0.0030 | 2 | [-0.00690, -0.00025] | yes | `SCREEN_PASS` |
| chase_gate | `residual_mlp - mlp` | `chase` | 0.002 | -0.00415 | [-0.00633, -0.00198] | -0.00198 | < 0.001 | 0.0030 | 3 | [-0.00633, -0.00198] | yes | `SCREEN_PASS` |

Registered note on `residual_mlp`: residual_mlp is registered as residual_t1's matched control; its own family measures what the production prior alone buys over mlp and it is read as such, not as a sequence claim.

**residual_t1** — seed 7 (estimand i), Holm group `family_residual_t1`, screen status `SCREEN_NOT_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `residual_t1 - residual_mlp` | `all` | 0.0 | +0.00009 | [-0.00052, +0.00057] | +0.00057 | 0.7660 | 0.7660 | 3 | [-0.00052, +0.00057] | no | `SCREEN_NOT_PASS` |
| death_gate | `residual_t1 - mlp` | `death` | 0.002 | -0.00378 | [-0.00660, -0.00075] | -0.00075 | < 0.001 | 0.0030 | 1 | [-0.00747, +0.00008] | yes | `SCREEN_PASS` |
| chase_gate | `residual_t1 - mlp` | `chase` | 0.002 | -0.00406 | [-0.00672, -0.00159] | -0.00159 | < 0.001 | 0.0030 | 2 | [-0.00709, -0.00118] | yes | `SCREEN_PASS` |

**residual_t1** — seed 13 (estimand i), Holm group `family_residual_t1`, screen status `SCREEN_NOT_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `residual_t1 - residual_mlp` | `all` | 0.0 | -0.00015 | [-0.00065, +0.00022] | +0.00022 | 0.4550 | 0.4550 | 3 | [-0.00065, +0.00022] | no | `SCREEN_NOT_PASS` |
| death_gate | `residual_t1 - mlp` | `death` | 0.002 | -0.00301 | [-0.00599, -0.00008] | -0.00008 | 0.0030 | 0.0060 | 2 | [-0.00658, +0.00047] | yes | `SCREEN_PASS` |
| chase_gate | `residual_t1 - mlp` | `chase` | 0.002 | -0.00413 | [-0.00669, -0.00165] | -0.00165 | < 0.001 | 0.0030 | 1 | [-0.00739, -0.00117] | yes | `SCREEN_PASS` |

**residual_t1** — seed mean (estimand ii), Holm group `family_residual_t1`, screen status `SCREEN_NOT_PASS`

| member | contrast | slice | t | point | 95% interval | U95 | raw p | Holm p | rank | rank-local interval | rejected | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | `residual_t1 - residual_mlp` | `all` | 0.0 | -0.00003 | [-0.00060, +0.00044] | +0.00044 | 0.8680 | 0.8680 | 3 | [-0.00060, +0.00044] | no | `SCREEN_NOT_PASS` |
| death_gate | `residual_t1 - mlp` | `death` | 0.002 | -0.00339 | [-0.00632, -0.00042] | -0.00042 | < 0.001 | 0.0030 | 1 | [-0.00707, +0.00039] | yes | `SCREEN_PASS` |
| chase_gate | `residual_t1 - mlp` | `chase` | 0.002 | -0.00409 | [-0.00679, -0.00163] | -0.00163 | < 0.001 | 0.0030 | 2 | [-0.00712, -0.00125] | yes | `SCREEN_PASS` |

Registered note on `residual_t1`: residual_t1 - mlp is reported but does not qualify as a primary contrast (it mixes the sequence and the production prior).

Family members are registered **for confirmation**, but their validation results remain screening evidence; every other contrast and every other slice is exploratory.

## 5. The two estimands, seed spread, and what (ii) is not

Estimand (i) is one seed checkpoint with paired block-only uncertainty. Estimand (ii) is the arithmetic across-seed mean under joint resampling: each replicate draws S seed indices with replacement, then B block indices with replacement, applies the same sampled seeds and blocks to both arms, and forms summed sampled losses over (S × summed sampled block row counts). It is **not** the log loss of averaged probabilities. Tonight's (ii) is descriptive two-seed robustness screen (joint seed and tournament-block resampling); not five-seed evidence and not uncertainty for a newly trained single checkpoint.

| contrast | slice | seed-7 point | seed-13 point | seed range | favourable seeds | mean point | mean 95% interval |
|---|---|---|---|---|---|---|---|
| `full − mlp` | `all` | -0.00115 | +0.00031 | +0.00146 | 1/2 | -0.00042 | [-0.00195, +0.00100] |
| `full − mlp` | `death` | +0.00337 | +0.00760 | +0.00423 | 0/2 | +0.00548 | [+0.00168, +0.00900] |
| `full − mlp` | `chase` | -0.00012 | +0.00181 | +0.00193 | 2/2 | +0.00085 | [-0.00135, +0.00293] |
| `fixed_decay − mlp` | `all` | -0.00498 | -0.00201 | +0.00297 | 2/2 | -0.00349 | [-0.00576, -0.00136] |
| `fixed_decay − mlp` | `death` | -0.00079 | +0.00423 | +0.00501 | 1/2 | +0.00172 | [-0.00252, +0.00563] |
| `fixed_decay − mlp` | `chase` | -0.00432 | -0.00095 | +0.00337 | 2/2 | -0.00263 | [-0.00526, +0.00012] |
| `fox − fixed_decay` | `all` | +0.00002 | -0.00011 | +0.00013 | 1/2 | -0.00005 | [-0.00022, +0.00011] |
| `fox − mlp` | `death` | -0.00073 | +0.00390 | +0.00463 | 1/2 | +0.00159 | [-0.00238, +0.00527] |
| `fox − mlp` | `chase` | -0.00440 | -0.00123 | +0.00317 | 2/2 | -0.00282 | [-0.00538, -0.00011] |
| `aligned_hist − full` | `all` | -0.00085 | -0.00222 | +0.00138 | 2/2 | -0.00154 | [-0.00263, -0.00051] |
| `aligned_hist − mlp` | `death` | +0.00368 | +0.00429 | +0.00061 | 0/2 | +0.00399 | [+0.00127, +0.00707] |
| `aligned_hist − mlp` | `chase` | -0.00080 | -0.00054 | +0.00026 | 2/2 | -0.00067 | [-0.00237, +0.00098] |
| `aligned_hist_rf − aligned_hist` | `all` | -0.00028 | +0.00146 | +0.00174 | 1/2 | +0.00059 | [-0.00064, +0.00184] |
| `aligned_hist_rf − mlp` | `death` | +0.00138 | +0.00584 | +0.00445 | 1/2 | +0.00361 | [+0.00003, +0.00732] |
| `aligned_hist_rf − mlp` | `chase` | -0.00130 | +0.00064 | +0.00195 | 2/2 | -0.00033 | [-0.00235, +0.00165] |
| `recency_k30 − mlp` | `all` | -0.00068 | +0.00013 | +0.00082 | 1/2 | -0.00027 | [-0.00157, +0.00077] |
| `recency_k30 − mlp` | `death` | +0.00395 | +0.00729 | +0.00334 | 0/2 | +0.00562 | [+0.00192, +0.00876] |
| `recency_k30 − mlp` | `chase` | -0.00021 | +0.00118 | +0.00139 | 2/2 | +0.00049 | [-0.00144, +0.00215] |
| `same_entity_k30 − recency_k30` | `all` | -0.00058 | -0.00176 | +0.00118 | 2/2 | -0.00117 | [-0.00226, +0.00021] |
| `same_entity_k30 − mlp` | `death` | +0.00396 | +0.00485 | +0.00088 | 0/2 | +0.00440 | [+0.00155, +0.00736] |
| `same_entity_k30 − mlp` | `chase` | -0.00080 | -0.00089 | +0.00009 | 2/2 | -0.00084 | [-0.00248, +0.00096] |
| `same_entity_unr − aligned_hist_rf` | `all` | +0.00034 | -0.00136 | +0.00170 | 1/2 | -0.00051 | [-0.00181, +0.00080] |
| `same_entity_unr − mlp` | `death` | +0.00227 | +0.00414 | +0.00187 | 0/2 | +0.00320 | [+0.00026, +0.00592] |
| `same_entity_unr − mlp` | `chase` | -0.00141 | -0.00158 | +0.00017 | 2/2 | -0.00150 | [-0.00303, +0.00020] |
| `same_entity_k0 − mlp` | `all` | +0.00120 | +0.00084 | +0.00036 | 0/2 | +0.00102 | [-0.00011, +0.00225] |
| `same_entity_k0 − mlp` | `death` | +0.00582 | +0.00637 | +0.00055 | 0/2 | +0.00610 | [+0.00354, +0.00910] |
| `same_entity_k0 − mlp` | `chase` | +0.00160 | +0.00166 | +0.00006 | 2/2 | +0.00163 | [+0.00023, +0.00337] |
| `same_entity_k6 − mlp` | `all` | -0.00014 | +0.00021 | +0.00035 | 1/2 | +0.00004 | [-0.00123, +0.00138] |
| `same_entity_k6 − mlp` | `death` | +0.00448 | +0.00713 | +0.00265 | 0/2 | +0.00580 | [+0.00232, +0.00931] |
| `same_entity_k6 − mlp` | `chase` | +0.00046 | +0.00120 | +0.00074 | 2/2 | +0.00083 | [-0.00092, +0.00283] |
| `same_entity_k12 − mlp` | `all` | -0.00083 | -0.00104 | +0.00021 | 2/2 | -0.00093 | [-0.00236, +0.00052] |
| `same_entity_k12 − mlp` | `death` | +0.00374 | +0.00482 | +0.00108 | 0/2 | +0.00428 | [+0.00145, +0.00771] |
| `same_entity_k12 − mlp` | `chase` | -0.00012 | -0.00007 | +0.00006 | 2/2 | -0.00010 | [-0.00174, +0.00167] |
| `lstm − mlp` | `all` | +0.00190 | +0.00347 | +0.00157 | 0/2 | +0.00269 | [-0.00153, +0.00934] |
| `lstm − mlp` | `death` | +0.00420 | +0.00607 | +0.00186 | 0/2 | +0.00514 | [+0.00069, +0.01053] |
| `lstm − mlp` | `chase` | +0.00209 | +0.00404 | +0.00195 | 0/2 | +0.00306 | [-0.00114, +0.00952] |
| `xlstm − mlp` | `all` | -0.00562 | -0.00375 | +0.00188 | 2/2 | -0.00468 | [-0.00636, -0.00316] |
| `xlstm − mlp` | `death` | -0.00276 | +0.00096 | +0.00372 | 2/2 | -0.00090 | [-0.00467, +0.00246] |
| `xlstm − mlp` | `chase` | -0.00453 | -0.00304 | +0.00149 | 2/2 | -0.00378 | [-0.00544, -0.00229] |
| `residual_mlp − mlp` | `all` | -0.00567 | -0.00538 | +0.00028 | 2/2 | -0.00552 | [-0.00671, -0.00435] |
| `residual_mlp − mlp` | `death` | -0.00436 | -0.00277 | +0.00159 | 2/2 | -0.00357 | [-0.00659, -0.00072] |
| `residual_mlp − mlp` | `chase` | -0.00435 | -0.00396 | +0.00040 | 2/2 | -0.00415 | [-0.00633, -0.00198] |
| `residual_t1 − residual_mlp` | `all` | +0.00009 | -0.00015 | +0.00025 | 1/2 | -0.00003 | [-0.00060, +0.00044] |
| `residual_t1 − mlp` | `death` | -0.00378 | -0.00301 | +0.00077 | 2/2 | -0.00339 | [-0.00632, -0.00042] |
| `residual_t1 − mlp` | `chase` | -0.00406 | -0.00413 | +0.00007 | 2/2 | -0.00409 | [-0.00679, -0.00163] |

Seed ranges are empirical spreads over two seeds, not confidence intervals. No seed is selected and no CI endpoint is averaged.

## 6. The k sweep (D9)

Numbers of record: full-precision per-seed validation LL from runs/<config_id>/summary.yaml, after D8.7 consolidation; never rounded report values, smoke LL, checkpoint re-scores or LL reconstructed from saved probabilities. The k-to-configuration mapping is derived from the config's own `configurations` entries by arm and `params.k`, never by string template: k 0 → `same_entity_k0`, k 6 → `same_entity_k6`, k 12 → `same_entity_k12`, k 30 → `same_entity_k30`, k unr → `same_entity_unr`.

Rule: choose the k with the lowest two-seed arithmetic mean validation log loss, but only when mean_LL(30) - mean_LL(best) > tolerance; otherwise keep 30. Equality at the tolerance keeps 30. Any k may win, unr included. Tie rule: exact ties between qualifying minima use the registered sweep order ['0', '6', '12', '30', 'unr'], recorded before selection

**Selection: `SELECTED` → `same_entity_k30` (k = 30)**

| k | config id | seed 7 LL | seed 13 LL | mean | min | max | range | paired mean vs k30 | paired range | favourable seeds |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | `same_entity_k0` | 1.439333 | 1.438877 | 1.439105 | 1.438877 | 1.439333 | 0.000455 | +0.00247 | +0.00000 | 0/2 |
| 6 | `same_entity_k6` | 1.437994 | 1.438248 | 1.438121 | 1.437994 | 1.438248 | 0.000254 | +0.00149 | +0.00071 | 0/2 |
| 12 | `same_entity_k12` | 1.437303 | 1.436995 | 1.437149 | 1.436995 | 1.437303 | 0.000307 | +0.00051 | +0.00015 | 0/2 |
| 30 | `same_entity_k30` | 1.436865 | 1.436406 | 1.436635 | 1.436406 | 1.436865 | 0.000459 | +0.00000 | +0.00000 | 0/2 |
| unr | `same_entity_unr` | 1.436195 | 1.436222 | 1.436208 | 1.436195 | 1.436222 | 0.000027 | -0.00043 | +0.00049 | 2/2 |

Best mean is k = unr; margin against k = 30 is 0.000427 against a tolerance of 0.002. Not selected: `same_entity_k0`, `same_entity_k6`, `same_entity_k12`, `same_entity_unr`.

No interpretation flag fired for this selection.

the 0.002 tolerance is a selection tolerance, not a significance threshold and not demonstrated two-seed resolution; the chosen k is never called reliably optimal The whole sweep is labelled **two-seed directional screen**.

only k=30 has the registered recency_k30 comparison; same_entity_unr - aligned_hist_rf is the registered unrestricted-mask comparison; k=0/6/12 keep their registered mlp primary references and no k borrows another window's recency control

the two-seed selection remains explicitly provisional pending any registered whole-family seed extension

## 7. Mechanism contrasts (D10.5)

| contrast | what it measures (registered label) | inferential in a registered family | seed-7 point | seed-13 point | mean point | mean 95% interval | mean raw p | disposition |
|---|---|---|---|---|---|---|---|---|
| `fox − fixed_decay` | learned forgetting beyond fixed decay | yes | +0.00002 | -0.00011 | -0.00005 | [-0.00022, +0.00011] | 0.5560 | reported as computed |
| `same_entity_k30 − recency_k30` | ownership plus alignment beyond recency | yes | -0.00058 | -0.00176 | -0.00117 | [-0.00226, +0.00021] | 0.1040 | reported as computed |
| `same_entity_unr − aligned_hist_rf` | mask alone, given aligned inputs and matched relay-free keys | yes | +0.00034 | -0.00136 | -0.00051 | [-0.00181, +0.00080] | 0.5170 | reported as computed |
| `aligned_hist_rf − aligned_hist` | the relay-free wiring PLUS the key construction, not the wiring alone | yes | -0.00028 | +0.00146 | +0.00059 | [-0.00064, +0.00184] | 0.4480 | reported as computed |
| `aligned_hist − full` | the aligned history input | yes | -0.00085 | -0.00222 | -0.00154 | [-0.00263, -0.00051] | 0.0010 | reported as computed |

a mechanism contrast is inferential only in its registered family and slice; every other slice readout is exploratory. No difference is ever inferred from one arm being significant against mlp and another not — that comparison is not computed by this tool and is not reportable

No comparative claim in this report is derived from one arm being significant against `mlp` while another is not. That inference is not computed and is not reportable.

## 8. Non-inferiority gates, exploratory slices and the residual readouts

### Gates (D10.6)

Paired harm `d = LL(candidate) − LL(mlp)` on each gate slice, margin +0.002. An unadjusted numerical pass requires **strictly U95 < 0.002**; a family-adjusted gate additionally requires favourable Holm rejection against the +0.002 boundary, at least 10 blocks and complete paired seeds. Neither a point below the margin nor a failure to detect harm establishes non-inferiority.

| candidate | slice | readout | point | 95% interval | U95 | U95 < 0.002 | raw p | Holm p | status |
|---|---|---|---|---|---|---|---|---|---|
| `full` | `death` | seed 7 (estimand i) | +0.00337 | [+0.00072, +0.00588] | +0.00588 | no | 0.2870 | 0.2870 | `SCREEN_NOT_PASS` |
| `full` | `death` | seed 13 (estimand i) | +0.00760 | [+0.00499, +0.00998] | +0.00998 | no | < 0.001 | 0.0030 | `SCREEN_NOT_PASS` |
| `full` | `death` | seed mean (estimand ii) | +0.00548 | [+0.00168, +0.00900] | +0.00900 | no | 0.0730 | 0.2190 | `SCREEN_NOT_PASS` |
| `full` | `chase` | seed 7 (estimand i) | -0.00012 | [-0.00176, +0.00144] | +0.00144 | yes | 0.0120 | 0.0360 | `SCREEN_PASS` |
| `full` | `chase` | seed 13 (estimand i) | +0.00181 | [+0.00007, +0.00341] | +0.00341 | no | 0.8570 | 1.0000 | `SCREEN_NOT_PASS` |
| `full` | `chase` | seed mean (estimand ii) | +0.00085 | [-0.00135, +0.00293] | +0.00293 | no | 0.3000 | 0.6000 | `SCREEN_NOT_PASS` |
| `fixed_decay` | `death` | seed 7 (estimand i) | -0.00079 | [-0.00379, +0.00156] | +0.00156 | yes | 0.0180 | 0.0180 | `SCREEN_PASS` |
| `fixed_decay` | `death` | seed 13 (estimand i) | +0.00423 | [+0.00149, +0.00642] | +0.00642 | no | 0.1070 | 0.1070 | `SCREEN_NOT_PASS` |
| `fixed_decay` | `death` | seed mean (estimand ii) | +0.00172 | [-0.00252, +0.00563] | +0.00563 | no | 0.8770 | 0.8770 | `SCREEN_NOT_PASS` |
| `fixed_decay` | `chase` | seed 7 (estimand i) | -0.00432 | [-0.00566, -0.00310] | -0.00310 | yes | < 0.001 | 0.0030 | `SCREEN_PASS` |
| `fixed_decay` | `chase` | seed 13 (estimand i) | -0.00095 | [-0.00266, +0.00062] | +0.00062 | yes | < 0.001 | 0.0030 | `SCREEN_PASS` |
| `fixed_decay` | `chase` | seed mean (estimand ii) | -0.00263 | [-0.00526, +0.00012] | +0.00012 | yes | < 0.001 | 0.0030 | `SCREEN_PASS` |
| `fox` | `death` | seed 7 (estimand i) | -0.00073 | [-0.00358, +0.00157] | +0.00157 | yes | 0.0120 | 0.0240 | `SCREEN_PASS` |
| `fox` | `death` | seed 13 (estimand i) | +0.00390 | [+0.00119, +0.00605] | +0.00605 | no | 0.1550 | 0.2920 | `SCREEN_NOT_PASS` |
| `fox` | `death` | seed mean (estimand ii) | +0.00159 | [-0.00238, +0.00527] | +0.00527 | no | 0.8120 | 1.0000 | `SCREEN_NOT_PASS` |
| `fox` | `chase` | seed 7 (estimand i) | -0.00440 | [-0.00578, -0.00318] | -0.00318 | yes | < 0.001 | 0.0030 | `SCREEN_PASS` |
| `fox` | `chase` | seed 13 (estimand i) | -0.00123 | [-0.00299, +0.00042] | +0.00042 | yes | < 0.001 | 0.0030 | `SCREEN_PASS` |
| `fox` | `chase` | seed mean (estimand ii) | -0.00282 | [-0.00538, -0.00011] | -0.00011 | yes | < 0.001 | 0.0030 | `SCREEN_PASS` |
| `aligned_hist` | `death` | seed 7 (estimand i) | +0.00368 | [+0.00097, +0.00701] | +0.00701 | no | 0.2500 | 0.2500 | `SCREEN_NOT_PASS` |
| `aligned_hist` | `death` | seed 13 (estimand i) | +0.00429 | [+0.00136, +0.00723] | +0.00723 | no | 0.1200 | 0.1200 | `SCREEN_NOT_PASS` |
| `aligned_hist` | `death` | seed mean (estimand ii) | +0.00399 | [+0.00127, +0.00707] | +0.00707 | no | 0.1610 | 0.1610 | `SCREEN_NOT_PASS` |
| `aligned_hist` | `chase` | seed 7 (estimand i) | -0.00080 | [-0.00243, +0.00075] | +0.00075 | yes | < 0.001 | 0.0030 | `SCREEN_PASS` |
| `aligned_hist` | `chase` | seed 13 (estimand i) | -0.00054 | [-0.00230, +0.00110] | +0.00110 | yes | < 0.001 | 0.0030 | `SCREEN_PASS` |
| `aligned_hist` | `chase` | seed mean (estimand ii) | -0.00067 | [-0.00237, +0.00098] | +0.00098 | yes | 0.0010 | 0.0030 | `SCREEN_PASS` |
| `aligned_hist_rf` | `death` | seed 7 (estimand i) | +0.00138 | [-0.00099, +0.00350] | +0.00350 | no | 0.5480 | 0.6580 | `SCREEN_NOT_PASS` |
| `aligned_hist_rf` | `death` | seed 13 (estimand i) | +0.00584 | [+0.00366, +0.00848] | +0.00848 | no | 0.0010 | 0.0030 | `SCREEN_NOT_PASS` |
| `aligned_hist_rf` | `death` | seed mean (estimand ii) | +0.00361 | [+0.00003, +0.00732] | +0.00732 | no | 0.4010 | 0.8020 | `SCREEN_NOT_PASS` |
| `aligned_hist_rf` | `chase` | seed 7 (estimand i) | -0.00130 | [-0.00283, +0.00015] | +0.00015 | yes | < 0.001 | 0.0030 | `SCREEN_PASS` |
| `aligned_hist_rf` | `chase` | seed 13 (estimand i) | +0.00064 | [-0.00106, +0.00216] | +0.00216 | no | 0.0800 | 0.0800 | `SCREEN_NOT_PASS` |
| `aligned_hist_rf` | `chase` | seed mean (estimand ii) | -0.00033 | [-0.00235, +0.00165] | +0.00165 | yes | 0.0210 | 0.0630 | `SCREEN_NOT_PASS` |
| `recency_k30` | `death` | seed 7 (estimand i) | +0.00395 | [+0.00087, +0.00621] | +0.00621 | no | 0.1820 | 0.2840 | `SCREEN_NOT_PASS` |
| `recency_k30` | `death` | seed 13 (estimand i) | +0.00729 | [+0.00481, +0.00969] | +0.00969 | no | < 0.001 | 0.0030 | `SCREEN_NOT_PASS` |
| `recency_k30` | `death` | seed mean (estimand ii) | +0.00562 | [+0.00192, +0.00876] | +0.00876 | no | 0.0550 | 0.1650 | `SCREEN_NOT_PASS` |
| `recency_k30` | `chase` | seed 7 (estimand i) | -0.00021 | [-0.00181, +0.00129] | +0.00129 | yes | < 0.001 | 0.0030 | `SCREEN_PASS` |
| `recency_k30` | `chase` | seed 13 (estimand i) | +0.00118 | [-0.00028, +0.00245] | +0.00245 | no | 0.2130 | 0.4260 | `SCREEN_NOT_PASS` |
| `recency_k30` | `chase` | seed mean (estimand ii) | +0.00049 | [-0.00144, +0.00215] | +0.00215 | no | 0.0800 | 0.1650 | `SCREEN_NOT_PASS` |
| `same_entity_k30` | `death` | seed 7 (estimand i) | +0.00396 | [+0.00119, +0.00719] | +0.00719 | no | 0.1920 | 0.3840 | `SCREEN_NOT_PASS` |
| `same_entity_k30` | `death` | seed 13 (estimand i) | +0.00485 | [+0.00243, +0.00773] | +0.00773 | no | 0.0240 | 0.0240 | `SCREEN_NOT_PASS` |
| `same_entity_k30` | `death` | seed mean (estimand ii) | +0.00440 | [+0.00155, +0.00736] | +0.00736 | no | 0.0920 | 0.1840 | `SCREEN_NOT_PASS` |
| `same_entity_k30` | `chase` | seed 7 (estimand i) | -0.00080 | [-0.00250, +0.00121] | +0.00121 | yes | 0.0030 | 0.0090 | `SCREEN_PASS` |
| `same_entity_k30` | `chase` | seed 13 (estimand i) | -0.00089 | [-0.00235, +0.00064] | +0.00064 | yes | 0.0010 | 0.0030 | `SCREEN_PASS` |
| `same_entity_k30` | `chase` | seed mean (estimand ii) | -0.00084 | [-0.00248, +0.00096] | +0.00096 | yes | 0.0020 | 0.0060 | `SCREEN_PASS` |
| `same_entity_unr` | `death` | seed 7 (estimand i) | +0.00227 | [-0.00053, +0.00482] | +0.00482 | no | 0.8350 | 0.8350 | `SCREEN_NOT_PASS` |
| `same_entity_unr` | `death` | seed 13 (estimand i) | +0.00414 | [+0.00188, +0.00678] | +0.00678 | no | 0.0590 | 0.0590 | `SCREEN_NOT_PASS` |
| `same_entity_unr` | `death` | seed mean (estimand ii) | +0.00320 | [+0.00026, +0.00592] | +0.00592 | no | 0.3700 | 0.7400 | `SCREEN_NOT_PASS` |
| `same_entity_unr` | `chase` | seed 7 (estimand i) | -0.00141 | [-0.00294, +0.00018] | +0.00018 | yes | < 0.001 | 0.0030 | `SCREEN_PASS` |
| `same_entity_unr` | `chase` | seed 13 (estimand i) | -0.00158 | [-0.00315, +0.00020] | +0.00020 | yes | < 0.001 | 0.0030 | `SCREEN_PASS` |
| `same_entity_unr` | `chase` | seed mean (estimand ii) | -0.00150 | [-0.00303, +0.00020] | +0.00020 | yes | < 0.001 | 0.0030 | `SCREEN_PASS` |
| `same_entity_k0` | `death` | seed 7 (estimand i) | +0.00582 | [+0.00323, +0.00889] | +0.00889 | no | 0.0060 | 0.0180 | `SCREEN_NOT_PASS` |
| `same_entity_k0` | `death` | seed 13 (estimand i) | +0.00637 | [+0.00409, +0.00946] | +0.00946 | no | 0.0010 | 0.0030 | `SCREEN_NOT_PASS` |
| `same_entity_k0` | `death` | seed mean (estimand ii) | +0.00610 | [+0.00354, +0.00910] | +0.00910 | no | 0.0030 | 0.0090 | `SCREEN_NOT_PASS` |
| `same_entity_k0` | `chase` | seed 7 (estimand i) | +0.00160 | [+0.00022, +0.00345] | +0.00345 | no | 0.6080 | 0.6080 | `SCREEN_NOT_PASS` |
| `same_entity_k0` | `chase` | seed 13 (estimand i) | +0.00166 | [+0.00029, +0.00337] | +0.00337 | no | 0.6360 | 0.6360 | `SCREEN_NOT_PASS` |
| `same_entity_k0` | `chase` | seed mean (estimand ii) | +0.00163 | [+0.00023, +0.00337] | +0.00337 | no | 0.6650 | 0.6650 | `SCREEN_NOT_PASS` |
| `same_entity_k6` | `death` | seed 7 (estimand i) | +0.00448 | [+0.00165, +0.00792] | +0.00792 | no | 0.0990 | 0.2820 | `SCREEN_NOT_PASS` |
| `same_entity_k6` | `death` | seed 13 (estimand i) | +0.00713 | [+0.00452, +0.01008] | +0.01008 | no | < 0.001 | 0.0030 | `SCREEN_NOT_PASS` |
| `same_entity_k6` | `death` | seed mean (estimand ii) | +0.00580 | [+0.00232, +0.00931] | +0.00931 | no | 0.0300 | 0.0900 | `SCREEN_NOT_PASS` |
| `same_entity_k6` | `chase` | seed 7 (estimand i) | +0.00046 | [-0.00106, +0.00222] | +0.00222 | no | 0.0940 | 0.2820 | `SCREEN_NOT_PASS` |
| `same_entity_k6` | `chase` | seed 13 (estimand i) | +0.00120 | [-0.00066, +0.00316] | +0.00316 | no | 0.4310 | 0.8620 | `SCREEN_NOT_PASS` |
| `same_entity_k6` | `chase` | seed mean (estimand ii) | +0.00083 | [-0.00092, +0.00283] | +0.00283 | no | 0.2400 | 0.4800 | `SCREEN_NOT_PASS` |
| `same_entity_k12` | `death` | seed 7 (estimand i) | +0.00374 | [+0.00093, +0.00720] | +0.00720 | no | 0.2690 | 0.5200 | `SCREEN_NOT_PASS` |
| `same_entity_k12` | `death` | seed 13 (estimand i) | +0.00482 | [+0.00217, +0.00819] | +0.00819 | no | 0.0400 | 0.0800 | `SCREEN_NOT_PASS` |
| `same_entity_k12` | `death` | seed mean (estimand ii) | +0.00428 | [+0.00145, +0.00771] | +0.00771 | no | 0.1220 | 0.2440 | `SCREEN_NOT_PASS` |
| `same_entity_k12` | `chase` | seed 7 (estimand i) | -0.00012 | [-0.00161, +0.00170] | +0.00170 | yes | 0.0260 | 0.0780 | `SCREEN_NOT_PASS` |
| `same_entity_k12` | `chase` | seed 13 (estimand i) | -0.00007 | [-0.00169, +0.00162] | +0.00162 | yes | 0.0240 | 0.0720 | `SCREEN_NOT_PASS` |
| `same_entity_k12` | `chase` | seed mean (estimand ii) | -0.00010 | [-0.00174, +0.00167] | +0.00167 | yes | 0.0280 | 0.0840 | `SCREEN_NOT_PASS` |
| `lstm` | `death` | seed 7 (estimand i) | +0.00420 | [-0.00019, +0.01034] | +0.01034 | no | 0.3920 | 1.0000 | `SCREEN_NOT_PASS` |
| `lstm` | `death` | seed 13 (estimand i) | +0.00607 | [+0.00194, +0.01111] | +0.01111 | no | 0.0550 | 0.1650 | `SCREEN_NOT_PASS` |
| `lstm` | `death` | seed mean (estimand ii) | +0.00514 | [+0.00069, +0.01053] | +0.01053 | no | 0.1870 | 0.5610 | `SCREEN_NOT_PASS` |
| `lstm` | `chase` | seed 7 (estimand i) | +0.00209 | [-0.00179, +0.00796] | +0.00796 | no | 0.9370 | 1.0000 | `SCREEN_NOT_PASS` |
| `lstm` | `chase` | seed 13 (estimand i) | +0.00404 | [-0.00027, +0.01059] | +0.01059 | no | 0.4860 | 0.4860 | `SCREEN_NOT_PASS` |
| `lstm` | `chase` | seed mean (estimand ii) | +0.00306 | [-0.00114, +0.00952] | +0.00952 | no | 0.7640 | 0.7640 | `SCREEN_NOT_PASS` |
| `xlstm` | `death` | seed 7 (estimand i) | -0.00276 | [-0.00601, +0.00003] | +0.00003 | yes | 0.0010 | 0.0030 | `SCREEN_PASS` |
| `xlstm` | `death` | seed 13 (estimand i) | +0.00096 | [-0.00165, +0.00309] | +0.00309 | no | 0.3630 | 0.3630 | `SCREEN_NOT_PASS` |
| `xlstm` | `death` | seed mean (estimand ii) | -0.00090 | [-0.00467, +0.00246] | +0.00246 | no | 0.1000 | 0.1000 | `SCREEN_NOT_PASS` |
| `xlstm` | `chase` | seed 7 (estimand i) | -0.00453 | [-0.00582, -0.00339] | -0.00339 | yes | < 0.001 | 0.0030 | `SCREEN_PASS` |
| `xlstm` | `chase` | seed 13 (estimand i) | -0.00304 | [-0.00428, -0.00192] | -0.00192 | yes | < 0.001 | 0.0030 | `SCREEN_PASS` |
| `xlstm` | `chase` | seed mean (estimand ii) | -0.00378 | [-0.00544, -0.00229] | -0.00229 | yes | < 0.001 | 0.0030 | `SCREEN_PASS` |
| `residual_mlp` | `death` | seed 7 (estimand i) | -0.00436 | [-0.00730, -0.00166] | -0.00166 | yes | < 0.001 | 0.0030 | `SCREEN_PASS` |
| `residual_mlp` | `death` | seed 13 (estimand i) | -0.00277 | [-0.00561, -0.00012] | -0.00012 | yes | 0.0020 | 0.0030 | `SCREEN_PASS` |
| `residual_mlp` | `death` | seed mean (estimand ii) | -0.00357 | [-0.00659, -0.00072] | -0.00072 | yes | < 0.001 | 0.0030 | `SCREEN_PASS` |
| `residual_mlp` | `chase` | seed 7 (estimand i) | -0.00435 | [-0.00649, -0.00229] | -0.00229 | yes | < 0.001 | 0.0030 | `SCREEN_PASS` |
| `residual_mlp` | `chase` | seed 13 (estimand i) | -0.00396 | [-0.00625, -0.00174] | -0.00174 | yes | < 0.001 | 0.0030 | `SCREEN_PASS` |
| `residual_mlp` | `chase` | seed mean (estimand ii) | -0.00415 | [-0.00633, -0.00198] | -0.00198 | yes | < 0.001 | 0.0030 | `SCREEN_PASS` |
| `residual_t1` | `death` | seed 7 (estimand i) | -0.00378 | [-0.00660, -0.00075] | -0.00075 | yes | < 0.001 | 0.0030 | `SCREEN_PASS` |
| `residual_t1` | `death` | seed 13 (estimand i) | -0.00301 | [-0.00599, -0.00008] | -0.00008 | yes | 0.0030 | 0.0060 | `SCREEN_PASS` |
| `residual_t1` | `death` | seed mean (estimand ii) | -0.00339 | [-0.00632, -0.00042] | -0.00042 | yes | < 0.001 | 0.0030 | `SCREEN_PASS` |
| `residual_t1` | `chase` | seed 7 (estimand i) | -0.00406 | [-0.00672, -0.00159] | -0.00159 | yes | < 0.001 | 0.0030 | `SCREEN_PASS` |
| `residual_t1` | `chase` | seed 13 (estimand i) | -0.00413 | [-0.00669, -0.00165] | -0.00165 | yes | < 0.001 | 0.0030 | `SCREEN_PASS` |
| `residual_t1` | `chase` | seed mean (estimand ii) | -0.00409 | [-0.00679, -0.00163] | -0.00163 | yes | < 0.001 | 0.0030 | `SCREEN_PASS` |

### Exploratory slices (D10.7)

Every reading below is exploratory: it changes no k, no family and no advancement, and this validation-only run opened no test rows.

**Disclosure — `innings_2` and `chase` are the same rows on this frame.** Both predicates select 57994 rows — **identity here is inferred from equal row, match and block counts only**, because this statistics JSON predates the persisted row-mask digest; two different row sets could in principle agree on all three, so treat this as a conservative disclosure rather than proof of identical membership — so the two readouts are one readout written twice. They are **never** independent corroboration of each other, and only `chase` is a three-member family member; the other is exploratory (Astra gate 1 round 3 ruling).

| contrast | slice | mean point | mean 95% interval | blocks | descriptive |
|---|---|---|---|---|---|
| `aligned_hist − full` | `death` | -0.00149 | [-0.00462, +0.00168] | 47 | no |
| `aligned_hist − full` | `chase` | -0.00152 | [-0.00295, -0.00011] | 47 | no |
| `aligned_hist − full` | `powerplay` | -0.00165 | [-0.00269, -0.00075] | 47 | no |
| `aligned_hist − full` | `middle` | -0.00147 | [-0.00265, -0.00029] | 47 | no |
| `aligned_hist − full` | `innings_1` | -0.00157 | [-0.00256, -0.00049] | 47 | no |
| `aligned_hist − full` | `innings_2` | -0.00152 | [-0.00295, -0.00011] | 47 | no |
| `aligned_hist − mlp` | `powerplay` | -0.00358 | [-0.00510, -0.00216] | 47 | no |
| `aligned_hist − mlp` | `middle` | -0.00281 | [-0.00486, -0.00103] | 47 | no |
| `aligned_hist − mlp` | `innings_1` | -0.00307 | [-0.00480, -0.00173] | 47 | no |
| `aligned_hist − mlp` | `innings_2` | -0.00067 | [-0.00237, +0.00098] | 47 | no |
| `aligned_hist_rf − aligned_hist` | `death` | -0.00038 | [-0.00429, +0.00306] | 47 | no |
| `aligned_hist_rf − aligned_hist` | `chase` | +0.00034 | [-0.00094, +0.00166] | 47 | no |
| `aligned_hist_rf − aligned_hist` | `powerplay` | +0.00076 | [+0.00010, +0.00158] | 47 | no |
| `aligned_hist_rf − aligned_hist` | `middle` | +0.00078 | [-0.00066, +0.00223] | 47 | no |
| `aligned_hist_rf − aligned_hist` | `innings_1` | +0.00081 | [-0.00061, +0.00222] | 47 | no |
| `aligned_hist_rf − aligned_hist` | `innings_2` | +0.00034 | [-0.00094, +0.00166] | 47 | no |
| `aligned_hist_rf − mlp` | `powerplay` | -0.00282 | [-0.00418, -0.00134] | 47 | no |
| `aligned_hist_rf − mlp` | `middle` | -0.00203 | [-0.00459, +0.00039] | 47 | no |
| `aligned_hist_rf − mlp` | `innings_1` | -0.00226 | [-0.00420, -0.00055] | 47 | no |
| `aligned_hist_rf − mlp` | `innings_2` | -0.00033 | [-0.00235, +0.00165] | 47 | no |
| `fixed_decay − mlp` | `powerplay` | -0.00397 | [-0.00575, -0.00215] | 47 | no |
| `fixed_decay − mlp` | `middle` | -0.00486 | [-0.00748, -0.00217] | 47 | no |
| `fixed_decay − mlp` | `innings_1` | -0.00422 | [-0.00659, -0.00210] | 47 | no |
| `fixed_decay − mlp` | `innings_2` | -0.00263 | [-0.00526, +0.00012] | 47 | no |
| `fox − fixed_decay` | `death` | -0.00014 | [-0.00057, +0.00036] | 47 | no |
| `fox − fixed_decay` | `chase` | -0.00019 | [-0.00044, +0.00007] | 47 | no |
| `fox − fixed_decay` | `powerplay` | +0.00004 | [-0.00029, +0.00041] | 47 | no |
| `fox − fixed_decay` | `middle` | -0.00008 | [-0.00035, +0.00021] | 47 | no |
| `fox − fixed_decay` | `innings_1` | +0.00007 | [-0.00009, +0.00024] | 47 | no |
| `fox − fixed_decay` | `innings_2` | -0.00019 | [-0.00044, +0.00007] | 47 | no |
| `fox − mlp` | `powerplay` | -0.00393 | [-0.00589, -0.00194] | 47 | no |
| `fox − mlp` | `middle` | -0.00494 | [-0.00744, -0.00243] | 47 | no |
| `fox − mlp` | `innings_1` | -0.00415 | [-0.00645, -0.00210] | 47 | no |
| `fox − mlp` | `innings_2` | -0.00282 | [-0.00538, -0.00011] | 47 | no |
| `full − mlp` | `powerplay` | -0.00193 | [-0.00357, -0.00033] | 47 | no |
| `full − mlp` | `middle` | -0.00134 | [-0.00309, +0.00034] | 47 | no |
| `full − mlp` | `innings_1` | -0.00150 | [-0.00329, +0.00000] | 47 | no |
| `full − mlp` | `innings_2` | +0.00085 | [-0.00135, +0.00293] | 47 | no |
| `lstm − mlp` | `powerplay` | +0.00286 | [-0.00133, +0.00949] | 47 | no |
| `lstm − mlp` | `middle` | +0.00179 | [-0.00291, +0.00910] | 47 | no |
| `lstm − mlp` | `innings_1` | +0.00236 | [-0.00214, +0.00947] | 47 | no |
| `lstm − mlp` | `innings_2` | +0.00306 | [-0.00114, +0.00952] | 47 | no |
| `recency_k30 − mlp` | `powerplay` | -0.00228 | [-0.00373, -0.00093] | 47 | no |
| `recency_k30 − mlp` | `middle` | -0.00088 | [-0.00258, +0.00049] | 47 | no |
| `recency_k30 − mlp` | `innings_1` | -0.00090 | [-0.00254, +0.00034] | 47 | no |
| `recency_k30 − mlp` | `innings_2` | +0.00049 | [-0.00144, +0.00215] | 47 | no |
| `residual_mlp − mlp` | `powerplay` | -0.00427 | [-0.00631, -0.00241] | 47 | no |
| `residual_mlp − mlp` | `middle` | -0.00696 | [-0.00859, -0.00533] | 47 | no |
| `residual_mlp − mlp` | `innings_1` | -0.00649 | [-0.00746, -0.00514] | 47 | no |
| `residual_mlp − mlp` | `innings_2` | -0.00415 | [-0.00633, -0.00198] | 47 | no |
| `residual_t1 − mlp` | `powerplay` | -0.00458 | [-0.00665, -0.00268] | 47 | no |
| `residual_t1 − mlp` | `middle` | -0.00687 | [-0.00885, -0.00510] | 47 | no |
| `residual_t1 − mlp` | `innings_1` | -0.00663 | [-0.00780, -0.00538] | 47 | no |
| `residual_t1 − mlp` | `innings_2` | -0.00409 | [-0.00679, -0.00163] | 47 | no |
| `residual_t1 − residual_mlp` | `death` | +0.00018 | [-0.00083, +0.00130] | 47 | no |
| `residual_t1 − residual_mlp` | `chase` | +0.00006 | [-0.00073, +0.00082] | 47 | no |
| `residual_t1 − residual_mlp` | `powerplay` | -0.00032 | [-0.00122, +0.00037] | 47 | no |
| `residual_t1 − residual_mlp` | `middle` | +0.00009 | [-0.00065, +0.00061] | 47 | no |
| `residual_t1 − residual_mlp` | `innings_1` | -0.00014 | [-0.00087, +0.00035] | 47 | no |
| `residual_t1 − residual_mlp` | `innings_2` | +0.00006 | [-0.00073, +0.00082] | 47 | no |
| `same_entity_k0 − mlp` | `powerplay` | -0.00020 | [-0.00224, +0.00180] | 47 | no |
| `same_entity_k0 − mlp` | `middle` | +0.00018 | [-0.00121, +0.00157] | 47 | no |
| `same_entity_k0 − mlp` | `innings_1` | +0.00049 | [-0.00098, +0.00182] | 47 | no |
| `same_entity_k0 − mlp` | `innings_2` | +0.00163 | [+0.00023, +0.00337] | 47 | no |
| `same_entity_k12 − mlp` | `powerplay` | -0.00255 | [-0.00427, -0.00071] | 47 | no |
| `same_entity_k12 − mlp` | `middle` | -0.00157 | [-0.00345, +0.00012] | 47 | no |
| `same_entity_k12 − mlp` | `innings_1` | -0.00166 | [-0.00351, +0.00011] | 47 | no |
| `same_entity_k12 − mlp` | `innings_2` | -0.00010 | [-0.00174, +0.00167] | 47 | no |
| `same_entity_k30 − mlp` | `powerplay` | -0.00297 | [-0.00481, -0.00112] | 47 | no |
| `same_entity_k30 − mlp` | `middle` | -0.00234 | [-0.00439, -0.00052] | 47 | no |
| `same_entity_k30 − mlp` | `innings_1` | -0.00197 | [-0.00367, -0.00034] | 47 | no |
| `same_entity_k30 − mlp` | `innings_2` | -0.00084 | [-0.00248, +0.00096] | 47 | no |
| `same_entity_k30 − recency_k30` | `death` | -0.00122 | [-0.00386, +0.00197] | 47 | no |
| `same_entity_k30 − recency_k30` | `chase` | -0.00133 | [-0.00263, +0.00020] | 47 | no |
| `same_entity_k30 − recency_k30` | `powerplay` | -0.00070 | [-0.00182, +0.00068] | 47 | no |
| `same_entity_k30 − recency_k30` | `middle` | -0.00147 | [-0.00278, +0.00004] | 47 | no |
| `same_entity_k30 − recency_k30` | `innings_1` | -0.00106 | [-0.00230, +0.00046] | 47 | no |
| `same_entity_k30 − recency_k30` | `innings_2` | -0.00133 | [-0.00263, +0.00020] | 47 | no |
| `same_entity_k6 − mlp` | `powerplay` | -0.00128 | [-0.00281, +0.00019] | 47 | no |
| `same_entity_k6 − mlp` | `middle` | -0.00096 | [-0.00265, +0.00076] | 47 | no |
| `same_entity_k6 − mlp` | `innings_1` | -0.00065 | [-0.00207, +0.00069] | 47 | no |
| `same_entity_k6 − mlp` | `innings_2` | +0.00083 | [-0.00092, +0.00283] | 47 | no |
| `same_entity_unr − aligned_hist_rf` | `death` | -0.00041 | [-0.00299, +0.00190] | 47 | no |
| `same_entity_unr − aligned_hist_rf` | `chase` | -0.00117 | [-0.00282, +0.00039] | 47 | no |
| `same_entity_unr − aligned_hist_rf` | `powerplay` | -0.00013 | [-0.00118, +0.00092] | 47 | no |
| `same_entity_unr − aligned_hist_rf` | `middle` | -0.00079 | [-0.00249, +0.00084] | 47 | no |
| `same_entity_unr − aligned_hist_rf` | `innings_1` | +0.00006 | [-0.00116, +0.00128] | 47 | no |
| `same_entity_unr − aligned_hist_rf` | `innings_2` | -0.00117 | [-0.00282, +0.00039] | 47 | no |
| `same_entity_unr − mlp` | `powerplay` | -0.00294 | [-0.00469, -0.00104] | 47 | no |
| `same_entity_unr − mlp` | `middle` | -0.00282 | [-0.00476, -0.00093] | 47 | no |
| `same_entity_unr − mlp` | `innings_1` | -0.00220 | [-0.00368, -0.00086] | 47 | no |
| `same_entity_unr − mlp` | `innings_2` | -0.00150 | [-0.00303, +0.00020] | 47 | no |
| `xlstm − mlp` | `powerplay` | -0.00461 | [-0.00584, -0.00345] | 47 | no |
| `xlstm − mlp` | `middle` | -0.00595 | [-0.00819, -0.00393] | 47 | no |
| `xlstm − mlp` | `innings_1` | -0.00545 | [-0.00757, -0.00359] | 47 | no |
| `xlstm − mlp` | `innings_2` | -0.00378 | [-0.00544, -0.00229] | 47 | no |

### Residual readouts (D10.11)

The qualifying primary is **residual_t1 - residual_mlp on the all slice**. residual_t1 - mlp mixes the sequence and the production prior; it is reported and never a qualifying primary `residual_mlp` is a production-prior control, not a sequence gain.

`residual_t1 − residual_mlp` on `all` reads -0.00003 [-0.00060, +0.00044]: the interval straddles zero, so **no incremental benefit of residual T1 over residual MLP was established at this resolution**. That is an unresolved interval, **not** a demonstration that sequence adds nothing over the production prior (Astra gate 2 round 1 MUST-FIX 4).

Base-only validation log loss (the production prior alone, exploratory reference): **1.433437** over 124292 rows, from `models/embeddings/seq_stage2/base_logits/validation.npz` — booster md5 `7ee1e1809917…`, model dir `models/xgb_i7_noweights_production` (role `ball_model_prod`), parquet md5 `326436317310…`, floor 0.0001, renormalised True. Row alignment: verified: the sidecar's recorded parquet md5 equals the live validation parquet's own md5 and its row count equals the frame's, and the logits are stored in parquet row order.

| residual arm − base only | seed-7 point | seed-13 point | mean point | mean 95% interval |
|---|---|---|---|---|
| `residual_mlp − base_only` | -0.00097 | -0.00078 | -0.00088 | [-0.00122, -0.00053] |
| `residual_t1 − base_only` | -0.00088 | -0.00094 | -0.00091 | [-0.00141, -0.00052] |

These base-only comparisons are exploratory.

## 9. Registered deviations, asymmetries, limitations (restated, none dropped)

Every entry below is restated from `experiments/configs/seq_stage2_v1.yaml` or enumerated by D10.13, with its consequence. A coverage check in the generator fails the render if any one of them is missing.

### Deviations (config `deviations`)

- `a15_not_adopted` — early stopping on a validation date-prefix disjoint from the selection rows is NOT adopted for any arm. **Reason:** one protocol for all arms, and the k-sweep tolerance is the registered guard. It does not correct checkpoint-selection optimism, which is one reason every validation number of this stage is screening evidence (D2.8, Astra SHOULD 12).
- `two_seeds_night_1` — every configuration runs at seeds 7 and 13 on night 1. **Reason:** the user's decision (D2.1); more seeds are added later to whole hypothesis families, never to single arms.
- `xlstm_simplifications` — the xLSTM arm is written in plain torch (no xlstm package is installed) and departs from Beck et al. 2024 as recorded by scripts/sequence_track/recurrent_arms.simplifications().
    - no causal conv1d pre-block on either cell
    - sLSTM cell is single-head with dense (d, d) input and recurrent matrices (no block-diagonal projections)
    - no up/down projection around the mLSTM cell; it runs at dmodel so 4 heads of 32 holds at dmodel=128, and each block carries a SwiGLU feed-forward sublayer (hidden 4*dmodel//3) instead
    - no learnable conv skip inside the mLSTM block
    - exponential forget gate in both cells (paper allows sigmoid or exp); zero initial cell states and m_0 = 0
    - sLSTM normaliser denominator clamped below at 1e-6 as a numerical guard
- `residual_oof_refits` — the train-split base logits are leave-one-date-block-out refits run at a fixed 25 rounds (production best_iteration + 1) with no early stopping and no eval set, instead of re-running production's early stopping per fold; the production label encoders are reused verbatim. **Reason:** no fold may re-select its round count against the validation split. Recorded in the base-logits sidecars (D4).
- `encoders_reused` — every arm reuses the frame's own 50-feature contract and the production encoders; no encoder is refit for stage 2.

### Known asymmetries (config `known_asymmetries`)

- `same_entity_vs_recency_history_input` — same_entity and recency differ in the HISTORY INPUT as well as the mask: same_entity reads the participant-aligned input, recency reads the innings-previous one. So same_entity_k - recency_k is "ownership plus alignment beyond recency"; the mask alone is isolated by same_entity_unr - aligned_hist_rf (D3.6, Astra MUST-FIX 5).
- `relay_free_keys_carry_own_outcome` — the relay-free arms (recency, same_entity, aligned_hist_rf) read an earlier row j < i as a (state, OWN outcome) pair: the key/value token is feat_proj(feat_j) + own_out_emb(y_j) + pos_j, with y_j row j's own realised outcome. The standard-wiring arms (full, aligned_hist, fixed_decay, fox, residual_t1) read every row, key or query, as a (state, SHIFTED previous outcome) pair: feat_proj(feat_j) + hist(j) + pos_j. The target's own token in both wirings carries hist(i), never y_i, so no arm can read its own label (certified in D7). Consequence: aligned_hist_rf - aligned_hist measures the wiring PLUS the key construction, not the wiring alone, and the same qualification attaches to same_entity_unr - aligned_hist_rf only through its aligned_hist_rf endpoint (both of its arms are relay-free, so the key construction is matched there). Decided at the 2026-09-11 Fable-to-Opus handover, § 3.1: the alternative was to keep one construction and widen the dependency set S(i), which would have made the ownership certificate weaker than the arm register states.
- `residual_arms_see_the_production_base` — the residual arms carry the production ball model's log probabilities, which no other arm sees; residual_t1 - mlp therefore mixes two changes and only residual_t1 - residual_mlp is confirmatory.
- `positional_embedding_not_uniform` — the recurrent arms have no positional embedding by construction (position is carried by the recurrence), while the masked and standard attention arms do — except fixed_decay and fox, whose bias carries position and which are registered with no positional embedding.
- `token_mlp_parameter_count_not_matched` — a token_mlp arm (mlp, residual_mlp) has no attention layer at all, so its parameter count is not matched to the attention arms parameter for parameter; counts are reported per run in arm_params.n_parameters.

### Known limitations (config `known_limitations`)

- (`known_limitations[0]`) the global outcome prior in the stats cache is not as-of-date, an inherited limitation of the frame; the direction and size of its effect on any stage-2 contrast are unknown.
- (`known_limitations[1]`) venue history in the frame is not recency-weighted (TODO.md backlog item); every arm is given the same feature, so no arm has it while another lacks it; whether its effect is the SAME across architectures is unmeasured, and no level effect is claimed.
- (`known_limitations[2]`) the i7 test split was read by the 2026-08 program, so it is not a clean holdout; stage 2 does not load it (score_test false) and the untouched cohort exists because of this.
- (`known_limitations[3]`) MPS kernels are not bit-reproducible across runs or machines, so a rerun of any checkpoint may differ in the low decimals; every checkpoint records mps_bit_reproducible false.
- (`known_limitations[4]`) checkpoint selection uses the same validation split the contrasts are computed on (A15 not adopted), so every number is screening evidence.

### Additional dispositions D10.13 requires by name

- `blocked_cross_fitting_not_past_only_forecasting` — the residual base logits on the train split are BLOCKED cross-fitting, not past-only forecasting: earlier date blocks are scored by boosters fitted on later blocks, so no residual arm may be described as fully as-of or whole-pipeline leakage-free (Astra gate 1 NOTE 3).
- `later_block_training_for_earlier_oof` — the same construction means a fold's out-of-fold predictions for early dates come from a model that saw later cricket; the direction of that exposure on any residual contrast is unmeasured.
- `caches_and_encoders_not_fold_refitted` — the stats cache and the production label encoders are not refitted per fold, so fold independence is partial by construction.
- `fixed_round_residual_refits` — each leave-one-block-out refit runs a fixed 25 boosting rounds with no early stopping and no eval set, so no fold re-selects its round count against the validation split.
- `wiring_plus_key_construction` — `aligned_hist_rf - aligned_hist` measures the relay-free wiring PLUS the key construction. It is not the wiring alone and is never reported as such.
- `own_outcome_versus_shifted_history_keys` — the relay-free arms read an earlier row as a (state, OWN outcome) pair while the standard-wiring arms read every row as a (state, SHIFTED previous outcome) pair; no arm can read its own label (certified in D7).
- `ownership_plus_alignment` — `same_entity_k30 - recency_k30` is ownership PLUS alignment beyond recency, because the two arms differ in the history input as well as the mask, so that contrast does not isolate ownership from alignment. The registered `same_entity_unr - aligned_hist_rf` contrast DOES hold the aligned history input, the relay-free wiring and the key construction fixed, so it isolates the attention mask given aligned inputs; no claim of isolated ownership is drawn from the k30 contrast (Astra gate 2 round 1 MUST-FIX 4).
- `machine_confounded_with_seed_no_machine_term` — seed 7 trained on the laptop and seed 13 on the Mac mini, so MACHINE IS CONFOUNDED WITH SEED. Every registered contrast is within-machine at each seed, so an additive machine effect cancels inside it, but arm-by-machine interaction is inseparable from seed variation and **no machine term is fitted**.
- `xlstm_simplifications` — every registered xLSTM simplification stands as recorded by `scripts/sequence_track/recurrent_arms.simplifications()`; the arm is plain torch, not the paper's reference implementation.
- `a15_not_adopted_and_checkpoint_selection_optimism` — A15 is not adopted: early stopping reads the same validation split the contrasts are computed on, so checkpoint-selection optimism is uncorrected and every number here is screening evidence.
- `two_seed_weakness` — two seeds are a directional screen. Estimand (ii) is descriptive; no interval here is a confirmation, and a two-seed result can never be LANDED (invariant 9).
- `unmatched_parameter_counts` — parameter counts are not matched across arms — a token-MLP arm has no attention layer at all — so no contrast is a capacity-controlled comparison; counts are reported per run.
- `access_and_position_schemes` — access rows and positional-embedding schemes differ by arm: the recurrent arms carry no positional embedding, fixed_decay and fox carry none by registration, and the residual arms alone see the production prior.
- `mps_non_bit_reproducibility` — MPS kernels are not bit-reproducible across runs or machines, so any rerun may differ in the low decimals; every checkpoint records `mps_bit_reproducible false`.
- `fresh_stage2_runs_not_stage1_replications` — stage 1 checkpoints are not reused: these are fresh stage 2 runs under the stage 2 config, not replications of stage 1 numbers.
- `global_prior_not_as_of` — the global outcome prior in the stats cache is not as-of-date; the direction and size of its effect on any stage 2 contrast are unknown, so it cannot be used to argue a contrast is unaffected.
- `venue_history_not_recency_weighted` — venue history in the frame is not recency-weighted; every arm is given the same feature, so no arm has it while another lacks it. Whether its effect is the SAME across architectures is unmeasured — a shared feature can have different effects in different functions of it — so it is not asserted to be a level effect rather than a per-arm advantage (Astra gate 2 round 1 MUST-FIX 4).
- `unresolved_historical_consumption` — whether the 2026-04-17 -> 2026-08-05 cohort window is untouched as an EVALUATION set is NOT settled. `docs/sequence_track/stage2_cohort_consumer_audit.md` establishes clean training frame and cache ancestry only, and clean training ancestry does not prove untouched evaluation status; D10.16(0) keeps Astra gate 1 round 2 MUST-FIX 6 open, so the cohort may not be opened on seed extension alone, however many seeds are added.
- `prior_test_inspection_and_d4_d5_reads` — the i7 test split was read by the 2026-08 program and is not a clean holdout; D4 scored test base logits and D5 ran a whole-test parity rebuild. Stage 2 does not train or select on test, and the untouched cohort exists because of this exposure.
- `shared_features_do_not_prove_shared_effects` — a shared feature set does not prove a shared effect across architectures: the arms are different functions of the same inputs.
- `invalid_historical_rss_unavailable` — the historical per-seed peak RSS recorded as a `RUSAGE_CHILDREN` delta is invalid and is reported as unavailable, never reconstructed.

A shared feature set does not prove a shared effect across architectures. Where a historical measurement is invalid it is labelled unavailable, not reconstructed.

### Corrections applied to config-sourced wording above

`experiments/configs/seq_stage2_v1.yaml` is launch-era training provenance and is not edited. Where it states a conclusion the evidence does not support, the clause is replaced here and the replacement is disclosed, original wording included.

- `known_limitations[1]` — rendered as "every arm is given the same feature, so no arm has it while another lacks it; whether its effect is the SAME across architectures is unmeasured, and no level effect is claimed" in place of the config's "every arm inherits the same feature, so it is a level effect on all of them rather than a per-arm advantage". Reason: Astra gate 2 round 1 MUST-FIX 4: a shared venue feature can have different effects in different architectures, so "a level effect rather than a per-arm advantage" is not established by the fact that every arm reads the feature.

## 10. In plain language: what was tested, what came out, and why nothing advances

**What was tested.** Sixteen small neural models were each trained on the same 1.88 million historical deliveries and then asked, one ball at a time, to put a probability on each of the six outcomes of the *next* delivery of 124,292 held-out balls. The ball that actually happened is known to the scorer but never to the model. The change under test is what each is allowed to remember about the innings so far: nothing at all (the token MLP control), everything (the full transformer), everything with a fixed decay, everything with a learned forgetting gate, only the same batter's and bowler's own past balls, only the last k balls, or a recurrent memory. Two further arms start from the production ball model's own probability and learn a correction on top of it. They do **not** differ only in that: capacity and parameter count, architecture, the positional scheme, how keys are built from an earlier ball, and — for the two residual arms — access to the production model's own probabilities differ as well, and every one of those is confounded with the memory change in at least one contrast (Astra gate 2 round 1 MUST-FIX 4; § 9 lists each).

**This is teacher-forced ball prediction.** It is not rollout — nobody simulated a match — and it is not market performance. A better log loss here does not imply a better simulated score distribution, a better win probability, or any betting edge. No market price appears anywhere in this stage.

**Best observed validation configuration.** `residual_t1`, with an observed two-seed mean validation log loss of 1.432528 (per seed: seed 7 1.432556, seed 13 1.432499; source: summary.yaml). That is an *observed* ranking on the same split its own early stopping used, over two seeds. Its uncertainty is the paired interval of its registered primary contrast in § 4, not this number, and the two-seed spread in § 5 is an empirical range rather than a confidence interval.

Some arms are CI-clean favourable against the token MLP on the `all` slice under CI-clean favourable on both per-seed estimand (i) readouts and on the estimand (ii) seed mean of `arm − mlp` on the `all` slice: `aligned_hist`, `fixed_decay`, `fox`, `residual_t1`, `same_entity_unr`, `xlstm`. That is a screening readout on two seeds, and whether any of them clears its registered family is decided in § 4, not here.

**Why no arm advances tonight.** Three reasons, all registered before the results were read. (1) Two seeds are a directional screen: the seed-mean estimand is descriptive, so no interval here is a confirmation. (2) Checkpoint selection used the same validation split the contrasts are computed on, so every number carries selection optimism. (3) The one untouched cohort was ruled **deferred** by the reviewer, so nothing has been confirmed out of sample. `advances: []`.

**Cohort confirmation is pending.** `cohort_status: DEFERRED_UNOPENED`, `cohort_scored: false`: no cohort feature, prediction or base-logit read happened. **There is no market claim. There is no LANDED verdict** — two-seed evidence is provisional and can never be LANDED (invariant 9). **The user's verdict is outstanding.**

**The exact next step, and the complete set of conditions that unlock the cohort (D10.16, restated in full — nothing here is abbreviated, and Astra gate 2 round 1 MUST-FIX 2 records that five seeds alone cannot unlock this cohort):**

0. **The historical-consumption question must be settled first.** D10.16(0) keeps Astra gate 1 round 2's MUST-FIX 6 open: whether the 2026-04-17 → 2026-08-05 window is untouched as an *evaluation* set is not established. `docs/sequence_track/stage2_cohort_consumer_audit.md` establishes clean training frame and cache **ancestry** only, and clean training ancestry does not prove untouched evaluation status. Until that is resolved in full, no number of seeds unlocks anything.
1. Seeds **29, 42 and 101** are added to every retained **whole** hypothesis family — the candidate, `mlp`, every matched control, and **all five** `same_entity` k configurations whenever k selection is involved. Never a single arm.
2. The seed-extension procedure **and** the final k-selection procedure are **frozen before any new-seed result is inspected**. A procedure chosen after seeing the new seeds is selection on the outcome and voids the extension.
3. The registered selection and the registered gates are rerun on all five seeds, reporting the seed spread and the **4/5 favourable direction count**; only candidates eligible under those reruns are retained.
4. The final family, checkpoint and **analysis** freeze is written and verified — an explicitly versioned final freeze, separate from the immutable training provenance (`pin_stage2_analysis.py --verify`).
5. The cohort's **provenance is re-verified unchanged** against its frozen hashes before it is opened.
6. The read is **one** frozen scoring batch covering every frozen candidate and control at once, with no interim result-driven change, no added arm, no changed k and no tuning; a start record and all outputs are persisted.
7. **Partial-exposure recovery rule:** any recovery or rerun reuses the identical frozen specification, **discloses the partial exposure**, and **never claims a fresh untouched read**.
8. If no candidate qualifies, **the cohort is left unopened**.

So the immediate next step is (0) plus (1)–(3): resolve historical consumption, freeze the extension and selection procedure, then train seeds 29, 42 and 101 across whole families and rerun these identical gates; tonight's k selection is `SELECTED` and is explicitly provisional.

## 11. Falsification wording (D10.15) and what a result here can and cannot mean

No result tonight licenses "X is the cause".

* `fox − fixed_decay` reads **unresolved** on the seed-mean estimand. A favourable reading would support *learned forgetting as an explanation*, nothing stronger. As it reads, FoX establishes **no detected benefit over fixed decay** at this resolution — which is not the same as learned forgetting adding nothing (Astra gate 2 round 1 MUST-FIX 4).
* `same_entity_k30 − recency_k30` reads **unresolved**. A favourable reading would support *ownership plus alignment beyond recency*, and cannot isolate ownership from alignment.
* `same_entity_unr − aligned_hist_rf` reads **unresolved**. A favourable reading would support *excluding other participants' history*, with alignment, wiring and key construction held fixed.

An interval that crosses zero means **unresolved evidence**, not proof that the mechanism does not matter. Where a reading above is `NOT_EVALUABLE` because a dependency certificate blocks an endpoint, even that is unavailable: the arm has not been shown to depend only on the rows its mask allows, so its interval supports no mechanism reading at all.

Neither proposed mechanism is supported at this resolution.

**On the death-over harm.** Explaining it would additionally require reproducing that harm on the matched `full − mlp` death slice *and* direct evidence of improvement on death rows; those mechanism-on-death comparisons are exploratory under the current family map. `full − mlp` on `death` reads **adverse** on the seed-mean estimand.

Nothing in this report is a betting claim, an advancement, or a verdict. `research/log_verdict.py` was not called.
