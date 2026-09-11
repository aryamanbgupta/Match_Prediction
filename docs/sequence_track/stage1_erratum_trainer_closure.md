# Stage 1 erratum: trainer source-closure hash

Date: 2026-09-11

## What changed

`scripts/transformer_t1.py` was extended for stage 2 (additional arms added
to the trainer). This is a live-file change, not an edit to any frozen stage
1 artifact. `experiments/configs/seq_stage1_sim_v1.yaml` pins the file's md5
at the stage 1 commit and therefore no longer matches the live file:

- Pinned in `seq_stage1_sim_v1.yaml` (stage 1 commit `2c7ea6f`):
  `2fa534412e8061352115014346b88100`
- `git show 2c7ea6f:scripts/transformer_t1.py | md5`:
  `2fa534412e8061352115014346b88100` (matches the pin, confirming the pin is
  correct for the commit it was taken at)
- Live `md5 scripts/transformer_t1.py` (post stage-2 edits, 2026-09-11):
  `9d7fc79e003a97e0e8b2ea5814377eea`

## Why stage 1 evidence stands

- The stage 2 implementer verified the four stage 1 arms (`full`, `mlp`,
  `no_attention`, `no_history`) are bitwise identical in logits, state-dict
  key order, and parameters under the same seed, comparing the extended
  trainer against the pre-edit file.
- The stage 1 checkpoints under `models/embeddings/seq_stage1/` are
  untouched.
- The stage 1 report and `experiments/configs/seq_stage1_sim_v1.yaml` are
  unchanged and remain frozen; this erratum does not re-pin either.

## What the test now checks

`test_pin_stage1.py::test_source_closure_is_hashed_in_full` now accepts a
recorded md5 for `scripts/transformer_t1.py` (and any other closure file)
that matches **either** the live checkout **or** the file's content at
commit `2c7ea6f` — the commit stage 1's evidence was produced against. This
tolerates post-freeze extensions to shared source files without re-pinning
frozen stage 1 config or loosening the check for any other file.

This is an erratum note only. It does not edit any frozen stage 1 evidence,
config, or report.

## How the closure is verified now (added 2026-09-12, Astra MUST-FIX 7)

Verification of the stage 1 source closure is done in **historical-source
mode** against the immutable commit `2c7ea6f`:
`pin_stage1.py --verify --historical-sources` (and
`verify_config(..., historical_sources_commit=...)`) resolves every closure
file from that commit's git blobs instead of the working tree, so what it
certifies is the source the stage 1 numbers were produced by.
`test_pin_stage1.py::test_source_closure_is_hashed_in_full` and
`::test_verify_passes_on_the_committed_config` both invoke that mode
explicitly; the earlier "live md5 **or** stage-1-commit md5, whichever
matches" tolerance is gone, because a silent OR would have accepted arbitrary
drift in any closure file, not just the trainer.

**Strict live verification remains the default and intentionally fails on the
current tree** (`pin_stage1.py --verify` with no flag, asserted by
`test_strict_live_verification_still_fails_on_the_drifted_tree`): the live
`scripts/transformer_t1.py` and `scripts/sequence_track/pin_stage1.py` have
both moved on for stage 2. The two modes answer different questions and the
fallback must not become the default — copying it into default verification
would conceal any later drift, live-closure certification would stop meaning
anything, and `--write` refuses the flag outright so a pin can never record a
historical blob as a live fact.

Why an immutable git blob is sufficient historical evidence (Astra NOTE 5): a
commit object fixes the bytes, so the md5 recomputed from `2c7ea6f` is the md5
of the file stage 1 ran, independently of anything in the working tree. The
behavioural-equivalence check recorded above (the four stage 1 arms bitwise
identical under the extended trainer) *supports* this erratum but cannot
replace source identity, and is not what verification rests on.
