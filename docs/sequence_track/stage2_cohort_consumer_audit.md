# Stage 2 cohort: historical-consumption audit of the parked cricsheet export

Written 2026-09-12 by the stage 2 orchestrator, closing Codex Astra's gate 1
MUST-FIX 6 / plan MUST-FIX 8: *"D5.3 overstates the eligibility audit.
Absence of references to the renamed export does not establish absence of
historical consumption: that export previously occupied `data/t20s_json`.
Audit its earlier use through permitted provenance/reports and record the
conclusion; no closed-directory reads are needed."*

No closed directory was opened for this audit. It reads only
`docs/remediation/item5_acceptance.md`, `git log`, filesystem timestamps and
the i7 stats cache's `_meta` table.

## What the export is

`docs/remediation/item5_acceptance.md` records that until 2026-09-10 this
checkout's `data/t20s_json` **was** a fresh cricsheet 1.2.0 export which
contained post-2026-04-16 matches, including the 137 sealed forward fixtures,
and that the sealed-set preflight in `test_forward_eval_contract.py`
**failed closed** on it — three recorded failures, an environment fact rather
than a code defect. With the user's approval on 2026-09-10 the export was
renamed aside to `data/t20s_json_fresh_export_20260805/` (10,189 files) and
the Mac mini's frozen corpus of record (11,264 files, 1.0.0 export) was synced
into `data/t20s_json`. The full suite then read 443 passed, 5 skipped, 0 failed.

So the question D5.3 must answer is not "does anything reference the parked
directory now" (nothing does) but "did anything consume post-2026-04-16 ball
data during the window in which that export occupied `data/t20s_json`".

## The decisive evidence: the frame and the cache predate the export

| artifact | built | provenance |
|---|---|---|
| `models/player_stats_cache_i7.sqlite` | **2026-07-26 03:13** | `_meta.source_dirs_json` = `["…/data/t20s_json"]`, `source_json_file_count` = **11264**, `source_match_count` = **9519**, `source_json_mtime_max` = **2026-04-19T04:14:12** |
| `data/xgb_data_i7/cricket_data_i7_{train,validation,test}.parquet` | **2026-07-26 03:21** | the frame materialised from that cache |

Three independent facts follow:

1. Both artifacts were built on 2026-07-26, **ten days before** the export
   dated 2026-08-05 existed, so neither can contain it.
2. The cache's recorded corpus is **11,264 files / 9,519 matches**, which is
   the frozen corpus of record, not the 10,189-file fresh export.
3. The cache's newest source file mtime is **2026-04-19**, consistent with a
   corpus that ends 2026-04-16 and inconsistent with any post-April ingest.

## Why that closes the question for the cohort

Everything the untouched cohort depends on is either older than the export or
newer than its removal:

- The cohort's global and per-phase priors are frozen from
  `models/player_stats_cache_i7.sqlite` (D5.4), built 2026-07-26 from the
  frozen corpus — so the frozen priors cannot carry post-2026-04-16 data.
- The cohort's own sidecar cache and ball rows were built 2026-09-11, after
  the 2026-09-10 swap, over `data/t20s_json` (frozen, 11,264) plus the 471
  raw window matches (D5.4, D5.6).
- The stage 1 runs that stage 2 compares against ran 2026-09-11, also after
  the swap; their block ids come from `load_competition_clusters` over the
  frozen `data/t20s_json`.
- No stage 2 training path reads `data/t20s_json` at all: the trainer loads
  the i7 train and validation parquets only.

## What remains, stated honestly

- The one recorded interaction between a pipeline and the export was a
  **refusal**: the sealed-set preflight failed closed rather than consuming it.
  A refusal is not consumption.
- This audit establishes that the **frame, the cache, the cohort state and the
  cohort rows** are free of the export. It does **not** attempt to prove that
  no diagnostic, report or one-off anywhere on the branch ever read a
  post-2026-04-16 ball during 2026-08-05 → 2026-09-10; that is unbounded. It
  is bounded where it matters: the cohort's eligibility rule excludes every
  match with a known consumer (the 137 forward fixtures and the 124
  match-level golden fixtures, D5.2), and the artifacts the stage 2 arms are
  trained and selected on demonstrably predate the export.
- `data/t20s_json_fresh_export_20260805/` is referenced by no script, config
  or report (D5.3's original grep) **and** is not a provenance ancestor of any
  artifact stage 2 uses (this audit).

## Disposition

D5.3's Result is amended to cite this file rather than the grep alone. The
grep result stands as far as it goes; the provenance argument above is what
actually supports the eligibility claim.
