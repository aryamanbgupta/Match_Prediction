# AutoResearch loop — candidate improvements

**Status: proposals only. Nothing here is implemented or decided.**

Measured 2026-08-03 against the two `night_v3.sh` runs to date (2026-07-31 and
2026-08-03). The loop works — both nights produced real verdicts — but the
orchestrator consumes the majority of the token budget doing work that does not
need Fable.

The autonomous loop must not edit this file (same rule as `program.md`,
`research/night*.sh`, `research/RUNNER_PROMPT*.md`).

---

## Standing constraints

Ordered. A proposal that trades against a higher line loses.

1. **Preserve research quality.** Verdict integrity, the DO-NOT-CHEAT rules,
   and the quality of idea selection come first.
2. **Reduce Fable usage.**
3. **Increase throughput** (ideas resolved per night).

**Rejected 2026-08-03:** demoting the orchestrator to Opus, whether as a
failure-mode fallback or as policy. Fable stays the orchestrator. Proposals
below are constrained accordingly.

---

## What was measured

Per-turn token usage read from the session transcripts (see *How to re-measure*).

### Night of 2026-08-03

| lane | turns | output | cache write | cache read |
|---|---:|---:|---:|---:|
| Fable (orchestrator) | 474 | 971,044 | 4,955,723 | 52,774,872 |
| Opus (subagents) | 471 | 242,445 | 3,882,042 | 47,944,716 |

Fable = **80.0%** of output tokens, 56.1% of cache writes, 52.4% of cache
reads. On 2026-07-31 the same split was 65.4% of output (Fable 104 turns /
156,809 out; Opus 201 turns / 82,980 out).

### Findings behind that split

- **The orchestrator takes as many turns as the executor** — 474 vs 471, at a
  mean peak resident context of **161,325 tokens** on live iterations. Cost
  scales as turns × resident context; both terms are large.
- **87% of everything the orchestrator reads is queue state.** Of ~270k tokens
  pulled in by `Read`: `IDEAS.md` ~113k across 23 calls, `results.tsv` ~103k
  across 11 calls, `program.md` ~19k across 8 full reads. `IDEAS.md` is 115 KB
  and `results.tsv` 43 KB; both are re-read each iteration and then stay
  resident for the following ~60 turns.
- **Bookkeeping is done by hand.** Orchestrator Bash census for the night: 131
  calls — 35 `git`, 34 `grep`, 18 `sed`, 12 `ls`. Each one-line edit to
  `results.tsv` or `IDEAS.md` costs a full context read.
- **The no-heavy-work rule is not holding.** 18 orchestrator commands invoked
  the train/eval pipeline, and 2 of 8 live iterations launched no subagent at
  all. `RUNNER_PROMPT_V3.md` forbids this; a prompt does not survive 60 turns.
- **Throughput loss on exhaustion.** 8 of 17 iterations did real work; 9
  produced nothing, each followed by a 30-minute sleep — roughly 4.5 hours idle
  while Opus capacity was untouched. `--fallback-model` did not engage; it
  covers overload, not usage exhaustion.

---

## Candidate improvements

### P1 — Feed the orchestrator a digest, not the queue

Generate a small digest with a plain script (no model): PENDING ideas with
priorities, the last ~10 `results.tsv` rows, the gate rule. Orchestrator reads
that instead of `IDEAS.md` + `results.tsv` + `program.md`.

Replaces ~42k tokens of resident context that is currently re-read on every
turn of the iteration. Largest single multiplier, lowest effort.

*Quality risk: low-moderate.* The orchestrator picks ideas from what it can
see, so a digest that drops context silently changes selection. Full verdict
history for TABLED ideas matters when step 1 designs a combination. Mitigation:
digest carries every PENDING and TABLED entry in full, truncating only resolved
LANDED/FAILED history, and the orchestrator keeps the ability to read the full
file when it decides it needs to.

### P2 — One deterministic command for bookkeeping

A `log_verdict.py`-style entry point taking the id, verdict and numbers, doing
the `results.tsv` append, the `IDEAS.md` status flip and the commit in one
call. Same for the claim step and the revert.

Turns ~20 turns of `grep`/`sed`/`git` into 1. Compounds with P1: turn count is
the multiplier on resident context.

*Quality risk: low, and arguably negative.* Mechanical edits by script are more
reliable than hand-editing a 115 KB file with `sed`. Numbers still originate
from executor output; the orchestrator still decides the verdict.

### P3 — Enforce the no-heavy-work rule in the hook

`.claude/hooks/bash-guard.sh` already exists and already knows how to deny. Have
`night_v3.sh` export an orchestrator marker and let the guard reject `uv run` /
`prop_backtest.py` / `run_sim_eval.py` in that session.

Makes it impossible for the orchestrator to absorb eval output into its context,
rather than asking it not to.

*Quality risk: low.* Needs an escape route for the legitimate cheap case (the
`raw/` spot-check greps in step 5 must still work), and a clear failure message
so the orchestrator delegates instead of retrying. Note the guard currently only
enforces in `bypassPermissions` mode; the loop runs `--permission-mode auto`, so
that condition would need revisiting.

### P4 — Narrow the orchestrator's remit toward planning

Steps 5–8 are largely mechanical: compare two numbers against a noise floor,
write a row, update status. The differentiated judgement is step 1 — which idea
is worth a night, and how to frame it. Narrowing Fable toward "read digest →
write plan.md" is a few turns at ~30k context instead of ~60 turns at 161k.

*Quality risk: this is the one to be careful with, and it is the reason this is
P4 and not P1.* The verdict step is where DO-NOT-CHEAT integrity is enforced and
where the raw-output spot-check happens. Moving judgement to the executor
creates a self-grading loop, which is exactly the failure mode the
orchestrator/executor split exists to prevent. Any version of this must keep
the verdict and the spot-check with Fable and move only genuinely clerical work
— which is largely what P2 already achieves. Consider P4 mostly subsumed by
P1+P2 unless a specific further step is identified.

---

## Open problem — no accepted solution

**Fable exhaustion stalls the night.** Once the orchestrator hits its limit, the
loop retries every 30 minutes and produces nothing until the window resets;
Opus capacity sits idle. On 2026-08-03 this cost roughly 4.5 hours and ~6
iterations.

Demoting the orchestrator is rejected, so the remaining directions are:

- **Reduce the burn so the wall arrives later or not at all.** P1+P2 attack this
  directly and may dissolve the problem without any failure-handling change.
- **Sleep smarter.** Blind 30-minute retries are not aligned to the usage window
  reset. Backing off to the actual reset time would at least stop burning wall
  clock on retries that cannot succeed.
- **Shorten the night to what the budget supports** rather than scheduling 9
  hours and stalling partway.

Not resolved. Listed so the finding is not lost.

---

## How to re-measure

Session transcripts live at
`~/.claude/projects/-Users-aryamangupta-CricML-Match-Prediction/`:

- `<session-id>.jsonl` — one file per orchestrator iteration.
- `<session-id>/subagents/agent-*.jsonl` — **executor transcripts live here.**
  A top-level-only glob finds zero subagent turns and makes delegation look
  completely broken. It is not.

Each `type: "assistant"` record carries `message.model` and `message.usage`
(`output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`).
Select files by mtime over the night's window, group by model and by lane
(top-level vs `subagents/`), and sum. The same records carry `tool_use` blocks,
which give the orchestrator's Bash/Read census and its `Agent` call count.

---

## Loose end

**B19 is orphaned.** Marked `RUNNING 2026-08-03T13:12:18Z` in `IDEAS.md` with
its claim and plan commits landed (`ab32820`, `4e27ee4`) but no verdict — the
loop died mid-idea. Clear or re-claim it before the next launch, or the next
orchestrator will skip it as already running.

*(Resolved 2026-08-07: claim reset to PENDING by supervisor note in IDEAS.md —
no executor artifacts existed, so no verdict was lost; the committed plan is
reusable on re-claim.)*
