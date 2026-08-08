#!/usr/bin/env python
"""Build `research/digest.md` — the orchestrator's iteration-start reading.

LOOP_IMPROVEMENTS.md P1: the orchestrator burns ~87% of its `Read` budget on
queue state (IDEAS.md + results.tsv + program.md re-read every iteration and
then resident for ~60 turns).  This script produces a much smaller file that
still carries everything idea *selection* depends on:

* the gate rule in summary, with a pointer to `program.md` for the full text;
* EVERY `PENDING` and `TABLED` idea reproduced verbatim (P1's mitigation:
  selection and C-series combination design both need the full entry);
* every `RUNNING` idea verbatim, flagged loudly as already claimed;
* one line each for resolved ideas (LANDED / FAILED / CRASH / SUPERSEDED /
  DONE / DEFERRED / INTERACTIVE) so nothing disappears silently;
* the last N rows of `research/results.tsv` verbatim;
* a closing note reminding the orchestrator it may still read the full files.

Deterministic and plain: no model calls, no network, no repo writes other than
the digest itself.  Pass `--now` to make the output byte-reproducible.

Usage:
    uv run python research/make_digest.py
    uv run python research/make_digest.py --stdout --rows 5
    uv run python research/make_digest.py --ideas-path /tmp/IDEAS.md --out /tmp/d.md
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ideas_lib import (  # noqa: E402
    DIGEST_PATH,
    REPO_ROOT,
    IDEAS_PATH,
    ONE_LINE_STATUSES,
    RESULTS_PATH,
    Idea,
    QueueFormatError,
    load_ideas,
    load_results,
)

GATE_SUMMARY = """\
- **Primary eval**: iteration Polymarket set, **≥$50k slice (n=168)**, scored with
  `--odds betting_odds_polymarket_v2.json --cluster-source-dir data/polymarket_test_v2`.
  The pre-v2 odds files carry the toss-market defect — never score against them.
- **Metric pair**: Avg Log Loss (down is better) + Flat ROI % (up is better).
  Sim/prop ideas use the gate pair stated in their own IDEAS.md entry instead.
- **BOTH** metrics improve → **LANDED** (keep the commits). **Exactly one** →
  **TABLED** (revert the code, keep the entry — raw material for a C-series
  combination). **Neither** → **FAILED** (revert). Crash or >2× the stated
  budget → **CRASH** (kill, revert).
- **Noise floors**: ideas that retrain → 0.007 LL / 2.3pp ROI (A1, 5 seeds);
  eval-only ideas → 0.002 LL / 2pp ROI. A move inside the floor is NOT an
  improvement. Compare against a same-session or logged fresh baseline, not a
  historical headline — seed variance is real.
- **Betting-layer ideas** (sizing, thresholds) cannot move LL by construction:
  ROI up with nothing else degraded → LANDED.
- Market reference ≥$50k is **LL 0.5940**; production `xgb_match_i7_swap_production`
  is LL 0.6249 / ROI +3.38%. No production arm beats the market on LL on any
  iteration slice, so "closes the gap" is the honest framing of a win.
- Report numbers **verbatim** from tool output. If an eval did not finish, the
  number does not exist.

`program.md` is authoritative for the verdict rule, the DO-NOT-CHEAT rules, the
eval recipes, and the PROTOCOL. Read it in full whenever you need a recipe or
are unsure — this summary never overrides it."""

BOOKKEEPING_NOTE = """\
Use the deterministic bookkeeping entry point instead of hand-editing IDEAS.md
or results.tsv with `sed`/`grep`:

```bash
# PROTOCOL step 2 — claim the idea you picked
uv run python research/log_verdict.py claim <ID>

# PROTOCOL step 6 — one results.tsv row + status flip + **Result:** text
uv run python research/log_verdict.py verdict <ID> <LANDED|TABLED|FAILED|CRASH> \\
    --date YYYY-MM-DD --commit <sha> \\
    --metrics <ll_50k> <market_ll> <roi_50k_pct> <roi_ci> <n_bets> \\
    --notes "<one-line notes field>" \\
    --result-text-file research/handoff/<ID>/result_line.md
```

`--metrics` defaults to `(sim-gate)` in all five columns, which is the right
default for sim/prop ideas. Add `--dry-run` to any subcommand to see the exact
edits without touching anything. The script never commits — it prints the
`git add … && git commit …` command for you to run."""

CLOSING_NOTE = """\
This digest is a *summary of the queue*, not a replacement for the repo's
memory. Read the full files directly whenever the digest is not enough — that
is expected, not a failure:

- `research/IDEAS.md` — the full history of every resolved idea, including the
  `**Result:**` paragraphs one-lined above. Read it when you need to know *why*
  a past idea landed or failed, or when a C-series combination needs the detail
  of a LANDED entry.
- `research/results.tsv` — every verdict row ever logged. Read it when you need
  a baseline row older than the rows quoted above, or a paired same-session
  baseline for your idea.
- `program.md` — authoritative for the verdict rule, DO-NOT-CHEAT rules, eval
  recipes, and PROTOCOL. Always read it before designing an eval.
- `git log --oneline -10` — still PROTOCOL step 0.

Nothing in this digest weakens any rule in `program.md` or
`research/RUNNER_PROMPT_V3.md`. If they disagree with this file, they win."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _rel(path: Path) -> str:
    """Repo-relative path when possible, so the digest header stays portable."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _wrap(text: str, width: int = 79) -> str:
    """Wrap generated prose so the digest stays diff-friendly."""
    return "\n".join(
        textwrap.fill(para, width=width) if para.strip() else ""
        for para in text.split("\n")
    )


def _fence_for(blob: str) -> str:
    """Pick a backtick fence longer than any run of backticks in `blob`."""
    longest = 0
    run = 0
    for char in blob:
        run = run + 1 if char == "`" else 0
        longest = max(longest, run)
    return "`" * max(3, longest + 1)


def build_digest(
    ideas_path: Path,
    results_path: Path,
    rows: int,
    now: str,
) -> str:
    _, ideas_lines, ideas, prose = load_ideas(ideas_path)
    results_text, header, data_rows = load_results(results_path)

    by_status: dict[str, list[Idea]] = {}
    for idea in ideas:
        by_status.setdefault(idea.status, []).append(idea)
    for bucket in by_status.values():
        bucket.sort(key=lambda i: i.sort_key)

    running = by_status.get("RUNNING", [])
    pending = by_status.get("PENDING", [])
    tabled = by_status.get("TABLED", [])
    resolved = [
        idea
        for idea in sorted(ideas, key=lambda i: i.sort_key)
        if idea.status in ONE_LINE_STATUSES
    ]
    unknown = [
        idea
        for idea in ideas
        if idea.status not in ONE_LINE_STATUSES
        and idea.status not in ("RUNNING", "PENDING", "TABLED")
    ]

    out: list[str] = []
    add = out.append

    add("# AutoResearch digest — read this first")
    add("")
    add(_wrap(
        f"Generated {now} by `research/make_digest.py` from "
        f"`{_rel(ideas_path)}` (sha {_sha(ideas_path)}, "
        f"{ideas_path.stat().st_size:,} B) and "
        f"`{_rel(results_path)}` (sha {_sha(results_path)}, "
        f"{results_path.stat().st_size:,} B)."
    ))
    add("")
    add(_wrap(
        "**This file replaces the routine iteration-start read of "
        "`research/IDEAS.md`, `research/results.tsv`, and `program.md`.** It "
        "carries every idea you may pick, in full. It one-lines only ideas "
        "that are already resolved. You may still open the full files at any "
        "time — see the closing note."
    ))
    add("")
    add("---")
    add("")
    add("## Gate rule (summary — `program.md` is authoritative)")
    add("")
    add(GATE_SUMMARY)
    add("")
    add("## Bookkeeping (PROTOCOL steps 2 and 6)")
    add("")
    add(BOOKKEEPING_NOTE)
    add("")
    add("---")
    add("")
    add("## Queue snapshot")
    add("")
    add(f"- RUNNING (already claimed — do NOT pick): {len(running)}"
        + (f" — {', '.join(i.ident for i in running)}" if running else " — none"))
    add(f"- PENDING (claimable): {len(pending)}"
        + (f" — {', '.join(i.ident for i in pending)}" if pending else " — none"))
    add(f"- TABLED (C-series raw material): {len(tabled)}"
        + (f" — {', '.join(i.ident for i in tabled)}" if tabled else " — none"))
    add(f"- Resolved / not claimable: {len(resolved)}")
    if unknown:
        add(
            "- **Unrecognised statuses** (treat as not claimable, and tell the "
            f"supervisor): {', '.join(i.ident + ' [' + i.status_raw + ']' for i in unknown)}"
        )
    add("")
    add(_wrap(
        "Statuses in the `Interactive backlog` section (`DONE`, `INTERACTIVE`, "
        "`DEFERRED`) are supervisor-only and are **never** claimable by the "
        "loop, whatever their status reads."
    ))
    add("")
    add("---")
    add("")
    add("## Queue rules and section notes (verbatim from IDEAS.md)")
    add("")
    for block in prose:
        text = block.text.strip()
        if text:
            add(text)
            add("")
    add("---")
    add("")

    # (b) RUNNING first, loudly flagged.
    add("## RUNNING — ALREADY CLAIMED, DO NOT PICK")
    add("")
    if not running:
        add("None. No idea is currently claimed.")
        add("")
    else:
        add(_wrap(
            "The following ideas are **already claimed by an earlier iteration**. "
            "Do not pick them, do not re-run them, and do not reset their status "
            "— a stale `RUNNING` claim is resolved by the supervisor, not by the "
            "loop. Reproduced in full only so you can tell whether your candidate "
            "duplicates one."
        ))
        add("")
        for idea in running:
            add(f"> !!! ALREADY CLAIMED — DO NOT PICK {idea.ident} !!!")
            add(f"> Status: `{idea.status_raw}`")
            add("")
            add(idea.text)
            add("")
    add("---")
    add("")

    add("## PENDING ideas — full text (pick exactly ONE, highest priority first)")
    add("")
    if not pending:
        add(_wrap(
            "None. Follow PROTOCOL step 1: design ONE combination of TABLED ideas "
            "as a new `C<n>` entry, or — if nothing is tabled either — write "
            "`research/reports/auto/NIGHT_SUMMARY.md`, `touch research/STOP`, "
            "commit, and exit."
        ))
        add("")
    else:
        add(_wrap(
            "Listed in claim order (priority, then id). Ideas are reproduced "
            "verbatim from IDEAS.md, including hypothesis, method, gate, and "
            "budget."
        ))
        add("")
        for idea in pending:
            add(idea.text)
            add("")
    add("---")
    add("")

    add("## TABLED ideas — full text (raw material for C-series combinations)")
    add("")
    add(_wrap(
        "TABLED = exactly one gate metric moved. These are not claimable "
        "directly; they are the inputs to a C-series combination when PENDING "
        "runs dry (PROTOCOL step 1). Their `**Result:**` paragraphs record which "
        "metric moved and by how much."
    ))
    add("")
    for idea in tabled:
        add(idea.text)
        add("")
    add("---")
    add("")

    add("## Resolved ideas — one line each")
    add("")
    add(_wrap(
        "Full text and `**Result:**` paragraphs for these live in "
        "`research/IDEAS.md`. Read it there when you need the detail."
    ))
    add("")
    add("```")
    add(f"{'id':<5} {'pri':<3} {'status':<38} title")
    for idea in resolved:
        add(idea.one_line())
    add("```")
    add("")
    add("---")
    add("")

    tail = data_rows[-rows:] if rows > 0 else []
    add(f"## `research/results.tsv` — last {len(tail)} rows (verbatim)")
    add("")
    add(_wrap(
        f"{len(data_rows)} rows total. Columns: "
        + " | ".join(f"`{c}`" for c in header.split("\t"))
        + ". `(sim-gate)` in the metric columns means the idea was gated on its "
        "own sim/prop metric pair, stated in its IDEAS.md entry."
    ))
    add("")
    blob = "\n".join([header] + tail)
    fence = _fence_for(blob)
    add(fence + "text")
    add(blob)
    add(fence)
    add("")
    add("---")
    add("")
    add("## When this digest is not enough")
    add("")
    add(CLOSING_NOTE)
    add("")

    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate research/digest.md from IDEAS.md + results.tsv."
    )
    parser.add_argument("--ideas-path", type=Path, default=IDEAS_PATH)
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    parser.add_argument("--out", type=Path, default=DIGEST_PATH)
    parser.add_argument(
        "--rows",
        type=int,
        default=10,
        help="how many trailing results.tsv rows to reproduce (default 10)",
    )
    parser.add_argument(
        "--now",
        default=None,
        help="override the generation timestamp (for byte-reproducible output)",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="write the digest to stdout instead of --out",
    )
    args = parser.parse_args(argv)

    now = args.now or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        digest = build_digest(args.ideas_path, args.results_path, args.rows, now)
    except (QueueFormatError, OSError) as exc:
        print(f"make_digest.py: {exc}", file=sys.stderr)
        return 1

    if args.stdout:
        sys.stdout.write(digest)
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(digest.encode("utf-8"))
    size = len(digest.encode("utf-8"))
    source = (
        args.ideas_path.stat().st_size
        + args.results_path.stat().st_size
    )
    print(
        f"make_digest.py: wrote {_rel(args.out)} ({size:,} B) from {source:,} B "
        f"of queue state ({source / max(size, 1):.1f}x smaller)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
