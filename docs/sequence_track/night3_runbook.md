# Night 3 runbook — Block B, and resume state for any session

Purpose: if the orchestrating session dies, hits a usage limit, or is replaced,
**read this file and pick up without re-deriving anything.** The contract is
`docs/sequence_track/night3_acceptance.md`; the design is
`docs/sequence_track/night3_design_draft.md` (v7, 2026-09-12); the registration
is `experiments/configs/seq_stage3_night3_v1.yaml`.

**Scope of this file:** Block B (the 3a negative-transfer screen, 30 runs).
The night's other blocks — E(4d), C114, E(4b) — are appended to the same two
queues at their own readiness cutoffs, in the registered order B, E(4d), C114,
E(4b), and each gets its own rows in the capacity table below before it may be
appended.

**Last updated:** 2026-09-12, before launch. Nothing has been launched.

---

## 0. NEXT ACTION (read this first)

> Block B is REGISTERED and both queues validate. The three items standing
> between here and launch are in § 1. Nothing is launched until every row of
> the capacity table in § 2 is filled and the § 4 preflight passes.

---

## 1. Pre-launch state

| item | state |
|---|---|
| `experiments/configs/seq_stage3_night3_v1.yaml` | WRITTEN; accepted by `retrain_stage2.load_config` and by `stage2_stats` (`shared_control`, `config_roles`, `optional_slice_specs`, `mechanism_contrasts`, `registered_families`) |
| all 30 registered `command` lines | verified byte-identical to `retrain_stage2.registered_commands` |
| `research/sequence_track/queue_laptop_night3.yaml` | WRITTEN; dry-run lists 18 jobs, all `decision=run` |
| `research/sequence_track/queue_mini_night3.yaml` | WRITTEN; dry-run lists 12 jobs, all `decision=run` |
| frozen inputs under `experiments/stage3a/` | present and pinned (§ 3) |
| `retrain_stage2.assert_writable` | RESOLVED 2026-09-13 00:26 IST: the driver admits `models/embeddings/seq_stage3` alongside `seq_stage2` (both roots, nothing else); the night-3 dry-run renders every job. |
| Block B smoke (per configuration, per machine) | laptop DONE 2026-09-13 00:34 IST (69 s full run, pre-commit working tree, trainer byte-identical to 90acca8); mini DONE 2026-09-13 00:55 IST (20 s for 384 steps at 90acca8); queues launched 00:56 IST at 65f2f63 |
| the single frozen commit on both machines | MADE: `90acca8` (freeze) + `65f2f63` (docs only), both machines reset to `65f2f63` before the 00:56 IST launch — see acceptance N7 |

---

## 2. Capacity table

Budget rule, per run: `max duration = 2 x expected_hours` is `run_queue.sh`'s
per-attempt alarm, and the runner **retries once**, so the worst case a single
job can consume is `4 x expected_hours`.

All six Block B arms are the same token-MLP architecture under the same
3,840-step budget and differ only in the training-row filter and the tier
embedding, so one budget per machine covers all six. `expected_hours` comes
from the measured full-run estimate of **~80 s per Block B run on the laptop**,
plus frame-load and startup overhead, rounded up to the 0.05 h grid with the
same margin the night-1 queues used.

### Block B — per configuration, per machine

| configuration | machine | smoke s (load + 1 eval window) | measured full run s | expected_hours (budget) | max duration (2x) | worst case (4x, one retry) |
|---|---|---|---|---|---|---|
| `mlp_pool` | laptop | 8 (200-step smoke 2026-09-12) | 69 (full 3,840-step `mlp_pool_tiercond` s7, 2026-09-13 00:34 IST, working tree with the trainer byte-identical to 90acca8 — see acceptance N8 deviation) | 0.15 | 0.30 h (1,080 s) | 0.60 h |
| `mlp_pool_tiercond` | laptop | 8 (200-step smoke 2026-09-12) | 69 (full 3,840-step `mlp_pool_tiercond` s7, 2026-09-13 00:34 IST, working tree with the trainer byte-identical to 90acca8 — see acceptance N8 deviation) | 0.15 | 0.30 h (1,080 s) | 0.60 h |
| `mlp_P_tiercond` | laptop | 8 (200-step smoke 2026-09-12) | 69 (full 3,840-step `mlp_pool_tiercond` s7, 2026-09-13 00:34 IST, working tree with the trainer byte-identical to 90acca8 — see acceptance N8 deviation) | 0.15 | 0.30 h (1,080 s) | 0.60 h |
| `mlp_rowmatch_P_tiercond` | laptop | 8 (200-step smoke 2026-09-12) | 69 (full 3,840-step `mlp_pool_tiercond` s7, 2026-09-13 00:34 IST, working tree with the trainer byte-identical to 90acca8 — see acceptance N8 deviation) | 0.15 | 0.30 h (1,080 s) | 0.60 h |
| `mlp_E_tiercond` | laptop | 8 (200-step smoke 2026-09-12) | 69 (full 3,840-step `mlp_pool_tiercond` s7, 2026-09-13 00:34 IST, working tree with the trainer byte-identical to 90acca8 — see acceptance N8 deviation) | 0.15 | 0.30 h (1,080 s) | 0.60 h |
| `mlp_rowmatch_E_tiercond` | laptop | 8 (200-step smoke 2026-09-12) | 69 (full 3,840-step `mlp_pool_tiercond` s7, 2026-09-13 00:34 IST, working tree with the trainer byte-identical to 90acca8 — see acceptance N8 deviation) | 0.15 | 0.30 h (1,080 s) | 0.60 h |
| `mlp_pool` | mini | 20 (384-step smoke `mlp_pool_tiercond` s13, 2026-09-13 00:55 IST at 90acca8) | ~200 (extrapolated from the 384-step smoke: 10× steps) | 0.25 | 0.50 h (1,800 s) | 1.00 h |
| `mlp_pool_tiercond` | mini | 20 (384-step smoke `mlp_pool_tiercond` s13, 2026-09-13 00:55 IST at 90acca8) | ~200 (extrapolated from the 384-step smoke: 10× steps) | 0.25 | 0.50 h (1,800 s) | 1.00 h |
| `mlp_P_tiercond` | mini | 20 (384-step smoke `mlp_pool_tiercond` s13, 2026-09-13 00:55 IST at 90acca8) | ~200 (extrapolated from the 384-step smoke: 10× steps) | 0.25 | 0.50 h (1,800 s) | 1.00 h |
| `mlp_rowmatch_P_tiercond` | mini | 20 (384-step smoke `mlp_pool_tiercond` s13, 2026-09-13 00:55 IST at 90acca8) | ~200 (extrapolated from the 384-step smoke: 10× steps) | 0.25 | 0.50 h (1,800 s) | 1.00 h |
| `mlp_E_tiercond` | mini | 20 (384-step smoke `mlp_pool_tiercond` s13, 2026-09-13 00:55 IST at 90acca8) | ~200 (extrapolated from the 384-step smoke: 10× steps) | 0.25 | 0.50 h (1,800 s) | 1.00 h |
| `mlp_rowmatch_E_tiercond` | mini | 20 (384-step smoke `mlp_pool_tiercond` s13, 2026-09-13 00:55 IST at 90acca8) | ~200 (extrapolated from the 384-step smoke: 10× steps) | 0.25 | 0.50 h (1,800 s) | 1.00 h |

`SMOKE` = to be filled from the pre-launch smoke, before the freeze commit.

### Queue totals

| machine | jobs | sum expected_hours | worst case (4x each) |
|---|---|---|---|
| laptop (seeds 7, 29, 42) | 18 | 2.70 h | 10.80 h |
| mini (seeds 13, 101) | 12 | 3.00 h | 12.00 h |
| **Block B total** | **30** | **5.70 h** (in parallel across two machines: 3.00 h critical path) | 12.00 h critical path |

### Overhead lines (not queue jobs; serial, on the laptop)

| phase | what it is | budget |
|---|---|---|
| preparation | smoke both machines, fill this table, commit the freeze, push to the mini, rsync nothing (Block B needs no sidecar), verify frame/cache md5 on both, both dry-runs | 1.00 h |
| certification | morning consolidation (verify-only, per seed), COMPLETE counts, comparability assertions | 0.50 h |
| reporting | statistics, pin, rendered report, acceptance results filled from files | 1.50 h |
| Astra gate | prompt filled with structural facts, round(s) to sign-off | 1.00 h (elapsed, not compute) |

### Queue deadline

**07:00 IST.** Any Block B job not started by 07:00 IST is not started at all:
touch the machine's STOP file and report the night at the run count it reached.
A partially complete Block B is reported as `NOT_EVALUABLE` per affected family
(fewer than five complete paired seeds), **never at a reduced seed count**, and
an experiment that is not ready at the freeze is deferred WHOLE rather than
trimmed in seeds.

### Refusal rule

> **Launch is REFUSED if any row of the capacity table above is blank** (any
> cell still reading `SMOKE`), or if this file is absent. This is not advisory:
> the design's § "Capacity" makes the populated table part of the frozen
> commit, so an unfilled table means the commit is not the frozen commit.

---

## 3. Frozen inputs and their pins

Generated by `scripts/sequence_track/stage3a_freeze_tiers.py` before any run.
Frame: train md5 `fac0b7bededf0aae1989d654e3eba9e8`, validation md5
`326436317310adadabe0175825e57d1b`.

| file | sha256 | split | matches / innings / rows |
|---|---|---|---|
| `experiments/stage3a/manifest.json` | `006ce060f1b17093194aa370a44b2a0491ffec8eac3e551e3cab2bf37138c3eb` | — | — |
| `experiments/stage3a/steps.json` | `10c342e4729bfe5289fe94e1bc2ebbcd079bea503a212105882b6bf598061ac8` | — | S = 3,840, E = 128, n_train_innings 16,319 |
| `target_P_matches.json` | `ad05b7fd1b75feaa15b1b3c2a2013dadfbd04094d72e429a3520c245458a9413` | train | 2,516 / 5,063 / 591,342 |
| `rowmatch_P_matches.json` | `82ab077b71d895a70934c6e796e8cc8854bee378fe7d57c3c658dde6a586a605` | train | 2,570 / 5,152 / 591,582 (overshoot 240) |
| `target_E_matches.json` | `5ff0ebd088079139a2fcbca62ac3c30b4506164715e6202bae99aaab2ab7385d` | train | 3,475 / 6,990 / 812,169 |
| `rowmatch_E_matches.json` | `4237f52132db20f5da465199773a4b9f952b29eb5be8060f0a830ea65ffc8f9e` | train | 3,525 / 7,070 / 812,250 (overshoot 81) |
| `target_E_validation_matches.json` | `626889fa8d70f0fbcfa9b5690aec6cb651fdbb742c2ccf300a9b9af971019b6d` | validation | 243 / 480 / 55,981 (222 tier 3) |
| `big3_validation_matches.json` | `49e8f62476d9b21352e324da7a660bd1407423fa12a995c166a9a6a545929b77` | validation | 8 / 16 / 1,895 |

**The two E lists are not interchangeable.** `target_E_matches.json` is what
`mlp_E_tiercond` TRAINS on; `target_E_validation_matches.json` is what the
`target_E` SLICE is read on. The manifest asserts they share no match, so
swapping them silently produces a zero-row slice.

---

## 4. Launching Block B (copy-paste)

```bash
MAIN=/Users/aryamangupta/CricML/Match_Prediction
WT=/Users/aryamangupta/CricML/MP_train_s7          # laptop training worktree
CFG=experiments/configs/seq_stage3_night3_v1.yaml

# --- 0. the commit must exist first; each queue job keys off the config sha ---
cd "$MAIN" && git log --oneline -1

# --- 1. point the worktree at that commit ---
# Resolve the commit from $MAIN explicitly: `git -C "$WT" checkout --detach HEAD`
# resolves HEAD *in the worktree* and would leave it training STALE CODE.
TRAIN_COMMIT=$(git -C "$MAIN" rev-parse HEAD)
git -C "$WT" checkout --detach "$TRAIN_COMMIT"
test "$(git -C "$WT" rev-parse HEAD)" = "$TRAIN_COMMIT" \
    || { echo "ABORT: worktree not at $TRAIN_COMMIT"; exit 1; }
ls -l "$WT"/data/xgb_data_i7 "$WT"/models/embeddings   # must still be symlinks into $MAIN

# --- 2. push to the mini and check it out there ---
# one-time on the mini, because embeddings-ladder is checked out there:
#   ssh mac-mini 'cd ~/CricML/Match_Prediction && git config receive.denyCurrentBranch updateInstead'
cd "$MAIN" && git push mini embeddings-ladder
ssh mac-mini 'cd ~/CricML/Match_Prediction && git checkout embeddings-ladder && git pull --ff-only && git rev-parse HEAD'
test "$(ssh mac-mini 'cd ~/CricML/Match_Prediction && git rev-parse HEAD')" = "$TRAIN_COMMIT" \
    || { echo "ABORT: mini not at $TRAIN_COMMIT"; exit 1; }

# --- 3. Block B needs NO sidecar and NO base logits. The only thing that must
#        match is the frame, the cache and the frozen lists. ---
ssh mac-mini 'cd ~/CricML/Match_Prediction && md5 -q data/xgb_data_i7/cricket_data_i7_train.parquet data/xgb_data_i7/cricket_data_i7_validation.parquet models/player_stats_cache_i7.sqlite && cat data/xgb_data_i7/.feature_hash'
# expect train      fac0b7bededf0aae1989d654e3eba9e8
#        validation 326436317310adadabe0175825e57d1b
#        cache      671ac8200b275fa3d11d848e609f5132
ssh mac-mini 'cd ~/CricML/Match_Prediction && shasum -a 256 experiments/stage3a/*.json'
# every digest must equal the table in § 3

# --- 4. dry runs. NOTE --queue on BOTH. laptop must list 18 jobs, mini 12,
#        all decision=run. ---
cd "$WT" && ./research/sequence_track/run_queue.sh \
    --queue research/sequence_track/queue_laptop_night3.yaml --machine laptop --dry-run
ssh mac-mini 'cd ~/CricML/Match_Prediction && ./research/sequence_track/run_queue.sh \
    --queue research/sequence_track/queue_mini_night3.yaml --machine mini --dry-run'

# --- 5. launch. run_queue.sh invokes caffeinate itself. ---
cd "$WT" && rm -f research/sequence_track/STOP_laptop_night3 && \
  nohup ./research/sequence_track/run_queue.sh \
    --queue research/sequence_track/queue_laptop_night3.yaml --machine laptop \
    > /tmp/night3_laptop_$(date +%Y%m%d_%H%M).log 2>&1 & echo "laptop pid $!"

# The mini needs uv on the PATH: a non-interactive ssh does not get it, and
# run_queue.sh then fails queue validation, launches NOTHING and exits 0.
# Quoting nohup through ssh is also fragile, so send a script instead.
cat > /tmp/launch_mini_night3.sh <<'EOS'
cd ~/CricML/Match_Prediction
export PATH="$HOME/.local/bin:$PATH"
rm -f research/sequence_track/STOP_mini_night3
LOG=/tmp/night3_mini_$(date +%Y%m%d_%H%M).log
nohup ./research/sequence_track/run_queue.sh --queue research/sequence_track/queue_mini_night3.yaml --machine mini > "$LOG" 2>&1 < /dev/null &
echo "launched pid $! log $LOG"
EOS
scp -q /tmp/launch_mini_night3.sh mac-mini:/tmp/launch_mini_night3.sh && ssh mac-mini 'bash /tmp/launch_mini_night3.sh'
```

**Every check above is a mandatory operator gate.** These snippets do not abort
the commands that follow, so a failed check must stop the launch by hand. Do
not proceed past a failed assertion.

**The runner's exit status is not evidence.** Memory refusals and exhausted
retries still let `run_queue.sh` exit zero. Count `COMPLETE.json` markers,
never exit codes.

**Stopping.** Per queue, checked between jobs and never mid-job:
`touch "$WT"/research/sequence_track/STOP_laptop_night3` and
`ssh mac-mini 'touch ~/CricML/Match_Prediction/research/sequence_track/STOP_mini_night3'`.
Remove the file and re-run the same command to resume; finished jobs are
skipped by config sha.

---

## 5. Morning sequence

```bash
MAIN=/Users/aryamangupta/CricML/Match_Prediction
CFG=experiments/configs/seq_stage3_night3_v1.yaml
RUNS=models/embeddings/seq_stage3/night3/runs

# --- 1. confirm nothing is writing on either machine. retrain_stage2.py MUST
#        be in this list: the driver writes run records and summaries AFTER
#        transformer_t1.py exits, so checking only the trainer can copy a
#        half-written run. ---
pgrep -fl "run_queue.sh|retrain_stage2.py|transformer_t1.py" || echo "laptop quiet"
ssh mac-mini 'pgrep -fl "run_queue.sh|retrain_stage2.py|transformer_t1.py" || echo "mini quiet"'

# --- 2. count completions, per seed. 30 expected: 18 laptop + 12 mini. ---
for s in 7 29 42; do echo "laptop seed $s: $(ls -d $MAIN/$RUNS/*/seed_$s/COMPLETE.json 2>/dev/null | wc -l) / 6"; done
ssh mac-mini "for s in 13 101; do echo \"mini seed \$s: \$(ls -d ~/CricML/Match_Prediction/$RUNS/*/seed_\$s/COMPLETE.json 2>/dev/null | wc -l) / 6\"; done"

# --- 3. bring ONLY the mini's seeds home, one seed at a time. Copying the
#        mini's whole runs tree could overwrite laptop-owned seed dirs. ---
for s in 13 101; do
  rsync -a --stats --include='*/' --include="seed_$s/***" --exclude='*' \
      mac-mini:CricML/Match_Prediction/$RUNS/ "$MAIN"/$RUNS/
done

# --- 4. consolidate: VERIFY-ONLY, per seed. It refuses and NAMES any
#        incomplete or unverifiable run rather than clearing and retraining it.
#        Never run a plain `--seeds 7,13,...` recovery: that would clear a
#        half-transferred mini seed and retrain it on the LAPTOP, silently
#        breaking the machine assignment the whole design rests on. Recovery
#        belongs in the owning machine's queue. ---
cd "$MAIN" && uv run --no-sync python scripts/sequence_track/retrain_stage2.py \
    --config $CFG --consolidate

# --- 5. statistics. --config and --runs-root are BOTH explicit: the tools
#        refuse the stage-2 defaults for any other config, and a night-3 table
#        written into a stage-2 path would overwrite night 1's evidence. ---
uv run --no-sync python scripts/sequence_track/stage2_stats.py stats \
    --config $CFG \
    --runs-root $RUNS \
    --out models/embeddings/seq_stage3/night3/stats/night3_block_b_stats.json \
    --seeds 7,13,29,42,101

# --- 6. pin, then report. The pin recomputes every hash from the files on disk
#        and exits non-zero on any difference. ---
uv run --no-sync python scripts/sequence_track/pin_stage2.py --config $CFG --write
uv run --no-sync python scripts/sequence_track/pin_stage2.py --config $CFG --verify
uv run --no-sync python scripts/sequence_track/render_stage2_report.py \
    --config $CFG \
    --stats-json models/embeddings/seq_stage3/night3/stats/night3_block_b_stats.json \
    --out research/reports/embeddings/NIGHT3_BLOCK_B_REPORT.md
# NOTE: render_stage2_report.py also defaults --k-selection to the stage-2
# k-sweep output, which Block B does not produce. If it refuses on a missing
# k-selection file, that is a renderer generalisation the code gate owns; the
# statistics JSON is the evidence either way and the report is written from it.

# --- 7. fill docs/sequence_track/night3_acceptance.md N1..N12 Results from
#        files, verbatim. A run that did not finish has no number. ---
```

There is **no k sweep** in Block B: `stage2_stats.py ksweep` requires five
`same_entity` configurations and this config registers none, so the subcommand
is not run and its absence is not a gap.

---

## 6. Invariants a fresh session must not break

- No read or evaluation against `data/golden/` or `data/forward_holdout/`.
- The untouched cohort stays `DEFERRED_UNOPENED`. Nothing in night 3 opens it.
- The test split is never loaded (`score_test: false`, `--no-kit`).
- Writes under `models/` only inside `models/embeddings/seq_stage3/night3/`.
- Stage 1 and stage 2 artifacts, reports, configs and pins are FROZEN. Night 3
  writes new files; it never edits a stage-2 evidence file.
- Python only through `uv run --no-sync`.
- `--queue` and `--config` are always explicit, on both machines.
- Never hand-run a single job to "catch up": put it back through the owning
  machine's queue so its provenance and markers match the others.
- Never launch before the night-3 commit exists; the job signature is the
  config hash, and an uncommitted tree makes the night unreproducible.
- Block B is validation-only screening evidence. Nothing here can be LANDED,
  nothing advances, and `research/log_verdict.py` is not run on it.
- Claims are FAMILY-LOCAL: `family_3a_P` and `family_3a_E` are correlated
  screens, never independent replications, and there is no "either target
  passes" reading.

---

## Batch 2 (4d, 4b, C114)

Appended 2026-09-13, **before any batch-2 run**. Acceptance contract:
`docs/sequence_track/batch2_acceptance.md` (checks B1..B10). Design:
`docs/sequence_track/night3_design_draft.md` v7 § "Block E" (4d, 4b) and
§ "C114". Block B above is a **separate, closed freeze**; nothing in this
section changes it, and no batch-2 arm may be tabled against a Block B arm —
Block B ran a fixed 3,840-step budget and batch 2 runs the stage 2 epoch loop
with live early stopping (`epochs: 30`, `patience: 3`).

### B2.0 What batch 2 is

Three config files, three job groups, 35 runs.

| rung | config | configurations | runs |
|---|---|---|---|
| 4d | `experiments/configs/seq_stage3_batch2_4d_v1.yaml` | `mlp_counts` (control), `mlp_spread_recency` | 10 |
| C114 | `experiments/configs/seq_stage3_batch2_c114_v1.yaml` | `full_114`, `mlp_114` (control), `full_50` | 15 |
| 4b | `experiments/configs/seq_stage3_batch2_4b_v1.yaml` | `identity_residual_l3` (λ 0.001), `identity_residual_l2` (λ 0.01) | 10 |

Queues: `research/sequence_track/queue_laptop_batch2.yaml` (21 jobs, seeds 7,
29, 42) and `research/sequence_track/queue_mini_batch2.yaml` (14 jobs, seeds
13, 101). Output tree `models/embeddings/seq_stage3/batch2/runs/<config_id>/seed_<seed>`.
Stop files `research/sequence_track/STOP_laptop_batch2` and
`.../STOP_mini_batch2` — distinct from night 3's, so stopping one batch never
stops the other.

**Three files and not one** because `statistics.families.shared_control` is
file-wide and the three rungs have three different controls (`mlp_counts`,
`mlp_114`, and — for 4b — a frozen npz that is not a configuration at all).
Every queue job therefore names its own `--config` explicitly.

**4b's family is DEFERRED.** Its runs happen; its readouts are descriptive
only. The full argument, the rejected alternatives and the one-line change to
`stage2_stats.py` that would lift the deferral are in
`docs/sequence_track/batch2_acceptance.md` § "The 4b decision". Do not compute
or quote any interval of a 4b arm against `ref_eb_ctx`.

### B2.1 Capacity table

Budget rule, per run: `max duration = 2 x expected_hours` is `run_queue.sh`'s
per-attempt alarm, and the runner **retries once**, so the worst case a single
job can consume is `4 x expected_hours`.

Budgets are per ARM CLASS, not per rung: the cost driver is the architecture
and the input width. From the code-gate smokes at 200 steps —
`identity_residual` 6 s, `mlp` with 66 inputs 7 s, `full` with 114 inputs
16 s — a full 30-epoch run is about 45 s (identity_residual), 60 s (any token
arm) and 240 s (any `full` arm) on the laptop, and roughly 3x that on the mini.

| configuration | arm class | machine | smoke s (200-step gate smoke) | estimated full run s | expected_hours (budget) | max duration (2x) | worst case (4x, one retry) |
|---|---|---|---|---|---|---|---|
| `mlp_counts` | token | laptop | 7 | 60 | 0.15 | 0.30 h (1,080 s) | 0.60 h |
| `mlp_spread_recency` | token | laptop | 7 | 60 | 0.15 | 0.30 h (1,080 s) | 0.60 h |
| `mlp_114` | token | laptop | 7 | 60 | 0.15 | 0.30 h (1,080 s) | 0.60 h |
| `full_114` | full | laptop | 16 | 240 | 0.40 | 0.80 h (2,880 s) | 1.60 h |
| `full_50` | full | laptop | 16 | 240 | 0.40 | 0.80 h (2,880 s) | 1.60 h |
| `identity_residual_l3` | identity_residual | laptop | 6 | 45 | 0.15 | 0.30 h (1,080 s) | 0.60 h |
| `identity_residual_l2` | identity_residual | laptop | 6 | 45 | 0.15 | 0.30 h (1,080 s) | 0.60 h |
| `mlp_counts` | token | mini | 7 (laptop, 200 steps, gate-2 smoke at 65f2f63 with 66 token inputs); mini: NOT smoked — deviation, see B6 | ~180 | 0.30 | 0.60 h (2,160 s) | 1.20 h |
| `mlp_spread_recency` | token | mini | 7 (laptop, 200 steps, gate-2 smoke); mini: NOT smoked — deviation, see B6 | ~180 | 0.30 | 0.60 h (2,160 s) | 1.20 h |
| `mlp_114` | token | mini | laptop not smoked separately (token arm under the 114 contract); mini: NOT smoked — deviation, see B6 | ~180 | 0.30 | 0.60 h (2,160 s) | 1.20 h |
| `full_114` | full | mini | 16 (laptop, 200 steps, gate-2 smoke); mini: NOT smoked — deviation, see B6 | ~720 | 0.90 | 1.80 h (6,480 s) | 3.60 h |
| `full_50` | full | mini | laptop 135 s full run (stage 2 measurement of the same arm); mini: NOT smoked — deviation, see B6 | ~720 | 0.90 | 1.80 h (6,480 s) | 3.60 h |
| `identity_residual_l3` | identity_residual | mini | 7 (mini, 1 epoch, 02:14 IST at 86e049e); laptop 6 (200 steps, gate-2 smoke) | ~135 | 0.30 | 0.60 h (2,160 s) | 1.20 h |
| `identity_residual_l2` | identity_residual | mini | same arm as l3: mini 7 (1 epoch); laptop 6 (200 steps) | ~135 | 0.30 | 0.60 h (2,160 s) | 1.20 h |

`full_50` is budgeted at the `full` rate although it reads only 50 columns: it
is the same architecture, and the 240 s smoke was the wider of the two, so the
budget is deliberately generous rather than measured separately.

`SMOKE` = to be filled from the pre-launch smoke on that machine, **before the
batch-2 freeze commit**.

#### Queue totals

| machine | jobs | sum expected_hours | worst case (4x each) |
|---|---|---|---|
| laptop (seeds 7, 29, 42) | 21 | 4.65 h | 18.60 h |
| mini (seeds 13, 101) | 14 | 6.60 h | 26.40 h |
| **batch 2 total** | **35** | **11.25 h** (in parallel across two machines: 6.60 h critical path) | 26.40 h critical path |

#### Refusal rule

> **Launch is REFUSED if any row of the batch-2 capacity table above is blank**
> (any cell still reading `SMOKE`). The populated table is part of the frozen
> commit; an unfilled table means the commit is not the frozen commit.

#### Queue deadline

Same rule as Block B: any batch-2 job not started by the operator's declared
deadline is not started at all — touch the machine's STOP file and report the
batch at the run count it reached. A partially complete rung is reported as
`NOT_EVALUABLE` per affected family (fewer than five complete paired seeds),
**never at a reduced seed count**.

### B2.2 Frozen inputs and their pins

Neither the frozen match lists nor `experiments/stage3a/steps.json` are read by
batch 2: no batch-2 configuration names a `params.train_match_list`, none
carries a step budget, and none sets `experiment.freeze_manifest`, so
`retrain_stage2._check_freeze_manifest` returns `None` for all three and
`experiments/stage3a/manifest.json` is never opened. Batch 2's frozen inputs
are the stage 4 sidecar and the stage 4 reference instead.

| file | sha256 | recorded in | read by |
|---|---|---|---|
| `models/embeddings/stage4/exposure/manifest.json` | `a1072431d5042a7f726789de96666035e89e8f778933451797ed793860e1e60d` | (pinned in the 4d config) | — |
| `models/embeddings/stage4/exposure/train.parquet` | `167730fb55c958c68606aea9cf68dbb29bc2690c01a36a0cdfc8b758f42c0ccc` | exposure manifest | 4d, both arms |
| `models/embeddings/stage4/exposure/validation.parquet` | `aaeab6ba07c225f53d0ba3ac9c6d7efe6770e8bc767d2d14af011da00e6c0316` | exposure manifest | 4d, both arms |
| `models/embeddings/stage4/refs/references.json` | `cb8688335193bd3d679b1bcb18e31578a39fff7c792156c78692eef68e80fa82` | (pinned in the 4b config) | — |
| `models/embeddings/stage4/refs/eb_ctx.joblib` | `c676435fa4407fa44be1762b7c9c4068db9dfb3e42d1d4bff4fa0bb97b104218` | references.json | — (the npz files are what the trainer reads) |
| `models/embeddings/stage4/refs/eb_ctx_train_probs.npz` | `35f28faf6ea3d874f25f5cac31d57d9e003518e1dd145d15853d5f57ac7da0d3` | references.json | 4b, both arms (IN-SAMPLE by construction; never a performance read) |
| `models/embeddings/stage4/refs/eb_ctx_validation_probs.npz` | `86c5a531dcbdcf18067a0aa749cadc356db373d7834aa66970a31d04a2437d62` | references.json | 4b, both arms |

The C114 feature contract is not a file: `transformer_t1.v7_114_columns()`
resolves it from `feature_registry.V6_GROUPS` at load time, and the driver
hashes the ordered list into `arm_params.feature_contract_sha256`. The expected
value is `d968f9f93c539c00491fb6b866ba764e9ae365266e5813b12401f9336fefe988`
over 114 columns. A different digest means `feature_registry` moved and the
batch is not the frozen batch.

Frame and cache are unchanged from Block B: train md5
`fac0b7bededf0aae1989d654e3eba9e8`, validation md5
`326436317310adadabe0175825e57d1b`, cache md5
`671ac8200b275fa3d11d848e609f5132`.

### B2.3 Syncing the stage 4 artifacts to the mini

Block B needed no sidecar. **Batch 2 does**: the mini runs 4d and 4b, so it
needs the exposure sidecar and the `eb_ctx` reference before its queue starts.
Seven files, enumerated — no `*` anywhere, so nothing else under
`models/embeddings/stage4/` can be pulled in by accident. About 96 MB in total,
dominated by `train.parquet` (51 MB) and `eb_ctx_train_probs.npz` (39 MB).

```bash
MAIN=/Users/aryamangupta/CricML/Match_Prediction
MINI=mac-mini:CricML/Match_Prediction

ssh mac-mini 'mkdir -p ~/CricML/Match_Prediction/models/embeddings/stage4/exposure ~/CricML/Match_Prediction/models/embeddings/stage4/refs'

rsync -a --checksum \
  "$MAIN"/models/embeddings/stage4/exposure/manifest.json \
  "$MAIN"/models/embeddings/stage4/exposure/train.parquet \
  "$MAIN"/models/embeddings/stage4/exposure/validation.parquet \
  "$MINI"/models/embeddings/stage4/exposure/

rsync -a --checksum \
  "$MAIN"/models/embeddings/stage4/refs/references.json \
  "$MAIN"/models/embeddings/stage4/refs/eb_ctx.joblib \
  "$MAIN"/models/embeddings/stage4/refs/eb_ctx_train_probs.npz \
  "$MAIN"/models/embeddings/stage4/refs/eb_ctx_validation_probs.npz \
  "$MINI"/models/embeddings/stage4/refs/

# Verify on the MINI against the table in B2.2 -- the transfer is not evidence,
# the recomputed digest is. Expect exactly the seven sha256s above.
ssh mac-mini 'cd ~/CricML/Match_Prediction && shasum -a 256 \
  models/embeddings/stage4/exposure/manifest.json \
  models/embeddings/stage4/exposure/train.parquet \
  models/embeddings/stage4/exposure/validation.parquet \
  models/embeddings/stage4/refs/references.json \
  models/embeddings/stage4/refs/eb_ctx.joblib \
  models/embeddings/stage4/refs/eb_ctx_train_probs.npz \
  models/embeddings/stage4/refs/eb_ctx_validation_probs.npz'
```

The three sibling references (`raw_rate_ctx*`, `ref_lin_50*`) are deliberately
NOT synced: no run reads them, and the descriptive level table that quotes
their validation log losses is computed on the laptop at report time.

### B2.4 Launching batch 2 (copy-paste)

Same shape as § 4, with the batch-2 queues and stop files. The commit must
exist first: each queue job keys off the config sha256.

```bash
MAIN=/Users/aryamangupta/CricML/Match_Prediction
WT=/Users/aryamangupta/CricML/MP_train_s7   # the laptop training worktree

# --- 0. dry runs. NOTE --queue on BOTH. laptop must list 21 jobs, mini 14,
#        all decision=run, no warnings. ---
cd "$WT" && ./research/sequence_track/run_queue.sh \
  --queue research/sequence_track/queue_laptop_batch2.yaml --machine laptop --dry-run
ssh mac-mini 'cd ~/CricML/Match_Prediction && export PATH="$HOME/.local/bin:$PATH" && ./research/sequence_track/run_queue.sh --queue research/sequence_track/queue_mini_batch2.yaml --machine mini --dry-run'

# --- 1. launch. run_queue.sh invokes caffeinate itself. ---
cd "$WT" && rm -f research/sequence_track/STOP_laptop_batch2 && \
  nohup ./research/sequence_track/run_queue.sh \
    --queue research/sequence_track/queue_laptop_batch2.yaml --machine laptop \
    > /tmp/batch2_laptop_$(date +%Y%m%d_%H%M).log 2>&1 & echo "laptop pid $!"

cat > /tmp/launch_mini_batch2.sh <<'EOS'
cd ~/CricML/Match_Prediction
export PATH="$HOME/.local/bin:$PATH"
rm -f research/sequence_track/STOP_mini_batch2
LOG=/tmp/batch2_mini_$(date +%Y%m%d_%H%M).log
nohup ./research/sequence_track/run_queue.sh --queue research/sequence_track/queue_mini_batch2.yaml --machine mini > "$LOG" 2>&1 < /dev/null &
echo "launched pid $! log $LOG"
EOS
scp -q /tmp/launch_mini_batch2.sh mac-mini:/tmp/launch_mini_batch2.sh && ssh mac-mini 'bash /tmp/launch_mini_batch2.sh'
```

**Stopping.** `touch "$WT"/research/sequence_track/STOP_laptop_batch2` and
`ssh mac-mini 'touch ~/CricML/Match_Prediction/research/sequence_track/STOP_mini_batch2'`.
Checked between jobs, never mid-job. Remove the file and re-run the same
command to resume; finished jobs are skipped by config sha.

**The runner's exit status is not evidence.** Count `COMPLETE.json` markers.

### B2.5 Morning sequence

```bash
MAIN=/Users/aryamangupta/CricML/Match_Prediction
RUNS=models/embeddings/seq_stage3/batch2/runs
C4D=experiments/configs/seq_stage3_batch2_4d_v1.yaml
C114=experiments/configs/seq_stage3_batch2_c114_v1.yaml
C4B=experiments/configs/seq_stage3_batch2_4b_v1.yaml

# --- 1. confirm nothing is writing on either machine. ---
pgrep -fl "run_queue.sh|retrain_stage2.py|transformer_t1.py" || echo "laptop quiet"
ssh mac-mini 'pgrep -fl "run_queue.sh|retrain_stage2.py|transformer_t1.py" || echo "mini quiet"'

# --- 2. count completions, per seed. 35 expected: 21 laptop + 14 mini,
#        7 configurations per seed. ---
for s in 7 29 42; do echo "laptop seed $s: $(ls -d $MAIN/$RUNS/*/seed_$s/COMPLETE.json 2>/dev/null | wc -l) / 7"; done
ssh mac-mini "for s in 13 101; do echo \"mini seed \$s: \$(ls -d ~/CricML/Match_Prediction/$RUNS/*/seed_\$s/COMPLETE.json 2>/dev/null | wc -l) / 7\"; done"

# --- 3. bring ONLY the mini's seeds home, one seed at a time. ---
for s in 13 101; do
  rsync -a --stats --include='*/' --include="seed_$s/***" --exclude='*' \
      mac-mini:CricML/Match_Prediction/$RUNS/ "$MAIN"/$RUNS/
done

# --- 4. consolidate: VERIFY-ONLY, once PER CONFIG (three configs share one
#        runs root, and --consolidate verifies the configurations its own
#        config registers). It refuses and NAMES any incomplete or
#        unverifiable run rather than clearing and retraining it. Never run a
#        plain seed recovery on the laptop for a mini-owned seed. ---
cd "$MAIN"
for CFG in $C4D $C114 $C4B; do
  uv run --no-sync python scripts/sequence_track/retrain_stage2.py --config $CFG --consolidate
done

# --- 5. statistics, once per config. --config and --runs-root are BOTH
#        explicit, and each writes its own output file: a batch-2 table written
#        into a stage-2 or night-3 path would overwrite earlier evidence. ---
mkdir -p models/embeddings/seq_stage3/batch2/stats
uv run --no-sync python scripts/sequence_track/stage2_stats.py stats \
    --config $C4D --runs-root $RUNS --seeds 7,13,29,42,101 \
    --out models/embeddings/seq_stage3/batch2/stats/batch2_4d_stats.json
uv run --no-sync python scripts/sequence_track/stage2_stats.py stats \
    --config $C114 --runs-root $RUNS --seeds 7,13,29,42,101 \
    --out models/embeddings/seq_stage3/batch2/stats/batch2_c114_stats.json
# 4b registers NO family. Its stats run produces the descriptive table only;
# if the tool refuses an empty family map, that refusal is the finding and the
# 4b table is read from summary.yaml and metrics.json instead. Either way, no
# PASS/SCREEN status exists for 4b and none may be quoted.
uv run --no-sync python scripts/sequence_track/stage2_stats.py stats \
    --config $C4B --runs-root $RUNS --seeds 7,13,29,42,101 \
    --out models/embeddings/seq_stage3/batch2/stats/batch2_4b_stats.json

# --- 6. pin each config, then report. The pin recomputes every hash from the
#        files on disk and exits non-zero on any difference. ---
for CFG in $C4D $C114 $C4B; do
  uv run --no-sync python scripts/sequence_track/pin_stage2.py --config $CFG --write
  uv run --no-sync python scripts/sequence_track/pin_stage2.py --config $CFG --verify
done

# --- 7. fill docs/sequence_track/batch2_acceptance.md B1..B10 Results from
#        files, verbatim. A run that did not finish has no number. ---
```

There is **no k sweep** in batch 2: `stage2_stats.py ksweep` requires five
`same_entity` configurations and none of the three configs registers any.

### B2.6 Invariants specific to batch 2

All of § 6 applies unchanged, plus:

- Writes under `models/` only inside `models/embeddings/seq_stage3/batch2/`.
  The stage 4 sidecar and references under `models/embeddings/stage4/` are
  READ-ONLY inputs; a batch-2 step that rebuilds either invalidates every run
  whose `arm_params` hashed the old bytes.
- **No 4b number is ever presented as a comparison against `ref_eb_ctx`.** 4b
  has no registered family; its table is levels beside levels.
- Block B's arms, ids and numbers are a different schedule. No batch-2 arm is
  tabled against `mlp_pool`, `mlp_pool_tiercond`, either target arm or either
  row-matched control.
- `full_50` is a fresh run and is NOT stage 1's or stage 2's `full`; never
  substitute an older checkpoint for it.
- The two C114 families share a member and a gate reference and are correlated
  screens. Claims are FAMILY-LOCAL: no aggregate, and no "either question
  passes" reading.
