# Stage 2 night runbook — resume state for any session

Purpose: if the orchestrating session dies, hits a usage limit, or is
replaced, **read this file and you can pick up without re-deriving
anything.** It is updated by the orchestrator at every step boundary.
The contract is still `docs/sequence_track/stage2_acceptance.md`; the
roles are still `docs/sequence_track/stage2_handoff_opus.md`.

**Last updated:** 2026-09-12 ~01:20 IST, by the Opus orchestrator.

## 0. NEXT ACTION (read this first)

> **STAGE 2 IS CLOSED — 2026-09-12, `VERDICT: SIGN-OFF`.** Nothing remains in
> this runbook. 80 runs, 20 ownership certificates all exactly 0, two reports,
> two pins that verify and replay byte-identically, suite 1799 passed. No arm
> advances; the cohort stays `DEFERRED_UNOPENED`; `log_verdict.py` was never
> called and the verdict is the user's.
>
> The stage's own record is `docs/sequence_track/stage2_acceptance.md` — see its
> closing "Stage 2: CLOSED" table. The next stage's work is stage 3 or 4 in
> `docs/SEQUENCE_TRACK_PLAN.md`, neither of which has been started.

### Autonomy grant (user, 2026-09-12 ~01:15 IST)

Verbatim: *"can you work on this autonomously and just orchestrate these
things overnight? Make sure that you set up reminders in case we run out of
usage limits, so you can pick up where you left off and you are not asking me
for decisions. Just go through everything, complete it, and go on to the next
task. Then I will take the report in the morning of where we preached, what is
done, and what is left."*

So, for the rest of this night, **do not stop to ask the user anything.**
Where the acceptance file says "after the user's go", read that as: decide it
by the registered rule if the rule is deterministic, and otherwise **ask Codex
Astra and follow its ruling**, recording the round. Specifically:

- **The `k` sweep (D9)** is a deterministic rule — best validation LL unless
  no k beats k = 30 by more than 0.002, then keep 30. Apply it; do not ask.
- **The D9 and D10 acceptance tables** are WRITTEN (2026-09-12), from Astra's
  dedicated round; D10 includes the falsification wording (10.15) and the
  cohort unlock preconditions (10.16).
- **Opening the untouched cohort (D10.10): ASKED AND RULED — DEFER.** Astra,
  2026-09-12: `COHORT: DEFER — Five-seed family confirmation must precede the
  only untouched read.` So tonight is a validation-only two-seed screen; the
  cohort is **not** read, and D10.10 records `DEFERRED_UNOPENED`. The unlock
  preconditions are registered in D10.16.
- **`research/log_verdict.py` is NOT run tonight.** Two-seed evidence is
  provisional and can never be LANDED (invariant 9), and the verdict is the
  one durable write to the research ledger. Prepare the gate JSON and the
  proposed ledger row, leave them uncommitted, and put them in the morning
  report for the user's decision.
- **No promotion, no manifest change, no merge.** Those are separate
  decisions with their own audits (plan § Stage 6).

---

## 1. Where we are right now

| step | state |
|---|---|
| 2a code, config, driver, pin, queue (D3, D4, D5) | DONE, recorded in the acceptance file |
| 2b smoke, all sixteen configurations (D6) | IN PROGRESS — six done under Fable, ten launched 2026-09-12 ~00:35 IST |
| 2c dependency test (D7) | DONE and PASSED for the four masked arms (exactly 0), two positive controls see relay |
| Astra gate 1 (code review) | prompt written, NOT yet run |
| commit of the stage 2 code | NOT yet made — nothing of stage 2 is committed |
| 2d overnight training on the mini (D8) | NOT started |
| 2e k sweep (D9), 2f statistics (D10) | acceptance tables WRITTEN from Astra's round; the work itself needs the training runs |
| D11 gate 2 | round 1 of gate 1 done and dispositioned; round 2 pending the implementer fixes |
| laptop training worktree | CREATED at `/Users/aryamangupta/CricML/MP_train_s7` with `data/xgb_data_i7`, `models/player_stats_cache_i7.sqlite`, `models/embeddings`, `models/xgb_i7_noweights_production` and `.venv` symlinked to the main checkout; frame/cache/base-logit resolution verified. **Re-point it at the stage 2 commit before launch** (`git -C <worktree> checkout --detach <commit>`) |

`git log --oneline -1` must still read `2c7ea6f` until the first stage 2
commit lands. `git status` shows the stage 2 files uncommitted.

---

## 2. The exact remaining sequence (morning onward)

Everything through the launch is done. What is left, in order:

1. **Both queues finish.** 16 jobs each, seed 7 laptop / seed 13 mini. Check
   completions by counting `runs/*/seed_<s>/COMPLETE.json`, never by the
   runner's exit status, which is zero even after memory refusals and
   exhausted retries.
2. **Consolidate (§ 3b).** Confirm no `run_queue.sh`, `retrain_stage2.py` or
   `transformer_t1.py` is alive on either machine, rsync only `seed_13/***`
   home, then `retrain_stage2.py --consolidate`. It refuses and names any
   incomplete run rather than retraining it; re-queue those on the machine
   that owns the seed. Fill D8.5 through D8.9.
3. **Land the deferred analysis fixes** before any number is quoted: the three
   items in D11's round 3 section (admission failing open, summary-LL
   authentication and k-sweep comparability, `--consolidate` diagnostic
   completeness) plus Astra's four cross-arm signature requirements and the two
   report fixes. An agent was dispatched for these during the night; verify
   its work landed and the suite is green.
4. **Recertify ownership (D10.12)** on the trained checkpoints of
   `same_entity_k0`, `same_entity_k30`, `same_entity_unr` and `recency_k30` at
   both seeds, with the matched positive controls. A failed certificate blocks
   that arm's family; it is not a footnote.
5. **D9, the k sweep.** Run it; the rule is deterministic (best two-seed mean
   unless no k beats k = 30 by more than 0.002, then 30; any k including `unr`
   may win). Record the D9.6 interpretation flags beside the selection. If the
   chosen k is not 30, note that it has **no** registered recency control and
   that borrowing `recency_k30` is forbidden.
6. **D10, the statistics and the report.** Then re-pin.
7. **Astra gate 2.** Fill the structural facts into
   `docs/sequence_track/astra/gate2_prompt_template.md` and send it. Iterate to
   SIGN-OFF, recording each round in D11.
8. **Commit the results.** Then stop: `research/log_verdict.py` is **not** run
   tonight or in the morning. Leave the gate JSON and the proposed ledger row
   prepared, and put them in the report for the user's decision.

## 3. Launching the night (copy-paste)

**Corrected 2026-09-12 after Astra gate 1 round 2 new MUST-FIX 2**: the
earlier version of this section omitted `--queue`, and `run_queue.sh` defaults
to `queue_laptop.yaml`, so `--machine mini` would have selected **zero jobs
and exited zero** — a silent no-op night. It also referenced the deleted
`STOP` file instead of the two per-queue stop files. Always pass `--queue`
explicitly, on both machines.

Seed 7 runs on the **laptop, from the training worktree** so that editing the
main checkout cannot change training sources mid-queue. Seed 13 runs on the
**mini**. One job at a time per machine.

```bash
MAIN=/Users/aryamangupta/CricML/Match_Prediction
WT=/Users/aryamangupta/CricML/MP_train_s7          # laptop training worktree

# --- 0. the commit must exist first; the queue keys jobs by config sha ---
cd "$MAIN" && git log --oneline -1

# --- 1. point the worktree at that commit ---
# Astra gate 1 round 3 MUST-FIX 1: resolve the commit from $MAIN explicitly.
# `git -C "$WT" checkout --detach HEAD` resolves HEAD *in the worktree*, which
# leaves it on its old commit and would train STALE CODE.
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
# assert the mini is on the same commit before launching
test "$(ssh mac-mini 'cd ~/CricML/Match_Prediction && git rev-parse HEAD')" = "$TRAIN_COMMIT" \
    || { echo "ABORT: mini not at $TRAIN_COMMIT"; exit 1; }

# --- 3. base logits are the only thing under models/ the mini needs ---
rsync -a --stats "$MAIN"/models/embeddings/seq_stage2/base_logits/ \
    mac-mini:CricML/Match_Prediction/models/embeddings/seq_stage2/base_logits/

# --- 4. re-verify frame and cache identity on the mini ---
ssh mac-mini 'cd ~/CricML/Match_Prediction && md5 -q data/xgb_data_i7/cricket_data_i7_train.parquet data/xgb_data_i7/cricket_data_i7_validation.parquet models/player_stats_cache_i7.sqlite && cat data/xgb_data_i7/.feature_hash'
# expect train      fac0b7bededf0aae1989d654e3eba9e8
#        validation 326436317310adadabe0175825e57d1b
#        cache      671ac8200b275fa3d11d848e609f5132

# --- 5. dry runs. NOTE --queue on BOTH. Each must list 16 jobs, all decision=run ---
cd "$WT" && ./research/sequence_track/run_queue.sh \
    --queue research/sequence_track/queue_laptop.yaml --machine laptop --dry-run
ssh mac-mini 'cd ~/CricML/Match_Prediction && ./research/sequence_track/run_queue.sh \
    --queue research/sequence_track/queue_mini.yaml --machine mini --dry-run'

# --- 6. launch. run_queue.sh invokes caffeinate itself. ---
cd "$WT" && rm -f research/sequence_track/STOP_laptop && \
  nohup ./research/sequence_track/run_queue.sh \
    --queue research/sequence_track/queue_laptop.yaml --machine laptop \
    > /tmp/stage2_laptop_$(date +%Y%m%d_%H%M).log 2>&1 & echo "laptop pid $!"

# The mini needs uv on the PATH: a non-interactive ssh does not get it, and
# run_queue.sh then fails queue validation, launches NOTHING and exits 0.
# Quoting nohup through ssh is also fragile, so send a script instead.
cat > /tmp/launch_mini.sh <<'EOS'
cd ~/CricML/Match_Prediction
export PATH="$HOME/.local/bin:$PATH"
rm -f research/sequence_track/STOP_mini
LOG=/tmp/stage2_mini_$(date +%Y%m%d_%H%M).log
nohup ./research/sequence_track/run_queue.sh --queue research/sequence_track/queue_mini.yaml --machine mini > "$LOG" 2>&1 < /dev/null &
echo "launched pid $! log $LOG"
EOS
scp -q /tmp/launch_mini.sh mac-mini:/tmp/launch_mini.sh && ssh mac-mini 'bash /tmp/launch_mini.sh'
```

Record both launch stamps in D8.4. They are necessarily after the cohort's
`frozen_at_utc` (`2026-09-11T17:23:20Z`).

**Stopping.** Per queue, checked between jobs and never mid-job:
`touch "$WT"/research/sequence_track/STOP_laptop` and
`ssh mac-mini 'touch ~/CricML/Match_Prediction/research/sequence_track/STOP_mini'`.
Remove the file and re-run the same command to resume; finished jobs are
skipped by config sha.

**Every check above is a mandatory operator gate.** Astra round 4 point 3:
these snippets do not themselves abort the commands that follow, so a failed
check must stop the launch by hand. Do not proceed past a failed assertion.

**The runner's exit status is not evidence.** Astra: memory refusals and
exhausted retries still let `run_queue.sh` exit zero, so a clean exit does not
mean 32 runs completed. Count admitted runs, never exit codes.

## 3b. Morning consolidation

Do this only when **both** queues have stopped writing — check that no
`run_queue.sh` and no `transformer_t1.py` process is alive on either machine
before rsyncing, or a half-written run will be copied.

```bash
# confirm nothing is writing. retrain_stage2.py MUST be in this list: the
# driver writes run records and summaries AFTER transformer_t1.py exits, so
# checking only the trainer can copy a half-written run (Astra r3 MUST-FIX 1).
pgrep -fl "run_queue.sh|retrain_stage2.py|transformer_t1.py" || echo "laptop quiet"
ssh mac-mini 'pgrep -fl "run_queue.sh|retrain_stage2.py|transformer_t1.py" || echo "mini quiet"'

# Bring ONLY seed 13 home; seed 7 is already in $MAIN via the worktree symlink.
# Copying the mini's whole runs tree could overwrite laptop-owned seed 7 dirs.
rsync -a --stats --include='*/' --include='seed_13/***' --exclude='*' \
    mac-mini:CricML/Match_Prediction/models/embeddings/seq_stage2/runs/ \
    "$MAIN"/models/embeddings/seq_stage2/runs/

# verification-and-summary only: never deletes, never trains (D8.7)
cd "$MAIN" && uv run --no-sync python scripts/sequence_track/retrain_stage2.py --consolidate
```

`--consolidate` refuses and names any incomplete or unverifiable run rather
than clearing and retraining it. Astra new MUST-FIX 3: a plain `--seeds 7,13`
invocation would clear a half-transferred seed 13 and **retrain it on the
laptop**, silently breaking the machine assignment the whole design rests on.
Recovery belongs in the owning machine's queue, not in consolidation.

## 4. What is safe to interrupt, and what is not

- **The queue is independent of any Claude session.** It runs under
  `nohup` on the mini with its own `caffeinate`, per-job timeout, memory
  floor, RSS watchdog and STOP file. If this session dies the night
  continues. Nothing needs a model in the loop.
- **Resumability** is by `COMPLETE` marker plus config sha256 per job, so
  re-running the queue after any interruption picks up where it stopped
  and never retrains a finished (configuration, seed).
- **Never** hand-run a single job to "catch up"; put it back through the
  queue so its provenance and markers match the others.
- **Never** launch the queue before the stage 2 code is committed: the
  job signature is the config hash, and an uncommitted tree makes the
  night unreproducible.

## 5. Invariants a fresh session must not break

- No read or evaluation against `data/golden/` or `data/forward_holdout/`.
  The two id-only reads are done and frozen in the cohort's `FROZEN.json`.
- The cohort (`models/embeddings/seq_stage2/cohort/`) is opened exactly
  once, at D10.10, after the family freeze and the user's go.
- Writes under `models/` only inside `models/embeddings/seq_stage2/`.
- Stage 1 artifacts, reports and configs are frozen; errata are new files.
- Python only through `uv run --no-sync`.
- Two seeds are a directional screen. Nothing here can be LANDED.
- Every number is copied from a file; an unfinished run has no number.
- Design questions go to Astra, not to the orchestrator's own judgment;
  advancement, `k`, the family freeze and every verdict go to the user.
