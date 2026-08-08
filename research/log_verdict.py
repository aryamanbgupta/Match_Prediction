#!/usr/bin/env python
"""Deterministic bookkeeping for the autonomous research loop (LOOP_IMPROVEMENTS P2).

One entry point for the two mechanical steps of the PROTOCOL, replacing ~20
turns of hand `grep`/`sed` against a 190 KB `IDEAS.md`:

    # PROTOCOL step 2 — claim the idea you picked
    uv run python research/log_verdict.py claim B20

    # PROTOCOL step 6 — results.tsv row + status flip + **Result:** text
    uv run python research/log_verdict.py verdict B20 TABLED \\
        --date 2026-08-07 --commit 1a2b3c4 \\
        --metrics 0.6231 0.5940 +18.40 "[-4.10,+41.02]" 168 \\
        --notes "one-line notes field for results.tsv" \\
        --result-text-file research/handoff/B20/result_line.md

`--metrics` defaults to `(sim-gate)` in all five metric columns, which is the
right default for sim/prop ideas gated on their own metric pair.

Guarantees
----------
* **results.tsv is append-only.** Existing rows are never rewritten; the new
  file is verified byte-for-byte against the old one after the write, and the
  write is rolled back if the prefix moved.
* **IDEAS.md history is append-only.** Only the target idea's status token and
  its `**Result:** —` placeholder change; every other idea and every prose
  block is re-parsed after the write and compared byte-for-byte.
* **A result is written once.** `verdict` refuses unless the current
  `**Result:**` line is still the em-dash placeholder.
* **Invalid transitions are refused**: `claim` requires `PENDING`, `verdict`
  requires `RUNNING` (an idea that was never claimed cannot get a verdict).
  `--force-from <STATUS>` overrides, but only if you name the current status
  exactly — so a supervisor can re-claim a `CRASH`ed or stale-`RUNNING` idea
  and the loop cannot do it by accident.
* **`--dry-run` on every subcommand** prints the exact diff and the exact
  results.tsv row, and touches nothing.
* **No git.** The script prints the `git add … && git commit …` command to run;
  committing stays an explicit act.
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ideas_lib import (  # noqa: E402
    IDEAS_PATH,
    REPO_ROOT,
    METRIC_COLUMNS,
    METRIC_PLACEHOLDER,
    RESULTS_COLUMNS,
    RESULTS_PATH,
    RESULT_LINE_RE,
    RESULT_PLACEHOLDER,
    VERDICTS,
    Idea,
    QueueFormatError,
    find_idea,
    load_ideas,
    load_results,
    parse_ideas,
)

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class BookkeepingError(RuntimeError):
    """A refused operation. Message is printed; nothing is written."""


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _snapshot(text: str) -> dict[str, str]:
    """Map every idea id / prose heading to its verbatim text, for diffing."""
    ideas, prose = parse_ideas(text)
    snap = {f"idea:{i.ident}": i.text for i in ideas}
    for pos, block in enumerate(prose):
        snap[f"prose:{pos}:{block.heading}"] = block.text
    return snap


def _replace_status(heading: str, old_raw: str, new_raw: str) -> str:
    """Swap the status bracket on a heading line, leaving everything else alone."""
    token = f"[{old_raw}]"
    if token not in heading:
        raise BookkeepingError(
            f"could not locate status token {token!r} in heading:\n  {heading}"
        )
    return heading.replace(token, f"[{new_raw}]", 1)


def _atomic_write(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _diff(old: str, new: str, label: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{label}",
            tofile=f"b/{label}",
            n=1,
        )
    )


def _write_ideas(path: Path, old_text: str, new_text: str, ident: str) -> None:
    """Write IDEAS.md, then verify no other entry moved. Rolls back on failure."""
    before = _snapshot(old_text)
    _atomic_write(path, new_text.encode("utf-8"))
    try:
        after = _snapshot(path.read_bytes().decode("utf-8"))
    except QueueFormatError as exc:  # pragma: no cover - defensive
        _atomic_write(path, old_text.encode("utf-8"))
        raise BookkeepingError(f"write left {path} unparseable ({exc}); rolled back")

    target = f"idea:{ident}"
    if set(before) != set(after):
        _atomic_write(path, old_text.encode("utf-8"))
        raise BookkeepingError(
            f"write changed the set of entries in {path}; rolled back"
        )
    collateral = [
        key for key in before if key != target and before[key] != after[key]
    ]
    if collateral:
        _atomic_write(path, old_text.encode("utf-8"))
        raise BookkeepingError(
            "write touched entries other than "
            f"{ident}: {', '.join(sorted(collateral))}; rolled back"
        )


def _append_results_row(path: Path, row: str, dry_run: bool) -> None:
    """Append exactly one row, then prove the existing bytes did not move."""
    old_bytes = path.read_bytes()
    if not old_bytes.endswith(b"\n"):
        raise BookkeepingError(f"{path} does not end with a newline; refusing to append")
    addition = (row + "\n").encode("utf-8")
    if dry_run:
        return
    new_bytes = old_bytes + addition
    _atomic_write(path, new_bytes)

    check = path.read_bytes()
    if check[: len(old_bytes)] != old_bytes or check[len(old_bytes) :] != addition:
        _atomic_write(path, old_bytes)
        raise BookkeepingError(
            f"append to {path} was not byte-append-only; rolled back"
        )


def _clean_notes(notes: str) -> str:
    """results.tsv is one row per line: collapse newlines/tabs into spaces."""
    cleaned = notes.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        raise BookkeepingError("--notes is empty after whitespace cleanup")
    return cleaned


def _rel(path: Path) -> str:
    """Repo-relative path when possible, so the printed git command is copy-pasteable."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _git_hint(paths: list[Path], message: str) -> str:
    return (
        "\nNOT COMMITTED. Run this yourself when you are satisfied:\n\n"
        f'  git add {" ".join(_rel(p) for p in paths)} && git commit -m {message!r}\n'
    )


# --------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------


def _resolve_idea(ideas: list[Idea], ident: str) -> Idea:
    try:
        return find_idea(ideas, ident)
    except QueueFormatError as exc:
        raise BookkeepingError(str(exc)) from exc


def _check_transition(idea: Idea, required: str, force_from: str | None, verb: str) -> None:
    if idea.status == required:
        return
    if force_from is not None:
        if force_from.upper() != idea.status:
            raise BookkeepingError(
                f"refusing to {verb} {idea.ident}: --force-from "
                f"{force_from.upper()!r} does not match its current status "
                f"{idea.status!r} (full token: [{idea.status_raw}])"
            )
        print(
            f"log_verdict.py: WARNING — supervisor override: {verb} {idea.ident} "
            f"from [{idea.status_raw}] (this transition normally requires "
            f"{required})",
            file=sys.stderr,
        )
        return
    raise BookkeepingError(
        f"refusing to {verb} {idea.ident}: its status is [{idea.status_raw}], "
        f"not {required}. "
        + (
            "Only PENDING ideas may be claimed; a RUNNING idea is already "
            "claimed by another iteration and a resolved idea is history. "
            if required == "PENDING"
            else "A verdict requires a claim first (PROTOCOL step 2) — "
            "run `claim` before `verdict`. "
        )
        + f"If this is a deliberate supervisor override, pass --force-from {idea.status}."
    )


def cmd_claim(args: argparse.Namespace) -> int:
    ts = args.ts or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if not TS_RE.match(ts):
        raise BookkeepingError(
            f"--ts {ts!r} is not a UTC ISO timestamp of the form 2026-08-07T13:12:18Z"
        )

    old_text, lines, ideas, _ = load_ideas(args.ideas_path)
    idea = _resolve_idea(ideas, args.ident)
    _check_transition(idea, "PENDING", args.force_from, "claim")

    new_lines = list(lines)
    new_lines[idea.start] = _replace_status(idea.heading, idea.status_raw, f"RUNNING {ts}")
    new_text = "\n".join(new_lines)

    print(f"--- {args.ident}: [{idea.status_raw}] -> [RUNNING {ts}]")
    print(_diff(old_text, new_text, args.ideas_path.name), end="")

    if args.dry_run:
        print("\nDRY RUN — nothing written.")
        return 0

    _write_ideas(args.ideas_path, old_text, new_text, idea.ident)
    print(f"\nlog_verdict.py: claimed {idea.ident} in {args.ideas_path}")
    print(_git_hint([args.ideas_path], f"Auto[{idea.ident}]: claim"), end="")
    return 0


def cmd_verdict(args: argparse.Namespace) -> int:
    verdict = args.verdict.upper()
    if verdict not in VERDICTS:
        raise BookkeepingError(
            f"verdict {verdict!r} is not one of {', '.join(VERDICTS)}"
        )
    if not DATE_RE.match(args.date):
        raise BookkeepingError(f"--date {args.date!r} is not YYYY-MM-DD")
    commit = args.commit.strip()
    if not commit or re.search(r"\s", commit):
        raise BookkeepingError("--commit must be a single non-empty token")

    metrics = list(args.metrics) if args.metrics else [METRIC_PLACEHOLDER] * 5
    if len(metrics) != 5:
        raise BookkeepingError(
            f"--metrics takes exactly 5 values ({', '.join(METRIC_COLUMNS)}); "
            f"got {len(metrics)}"
        )
    for name, value in zip(METRIC_COLUMNS, metrics):
        if "\t" in value or "\n" in value or value.strip() == "":
            raise BookkeepingError(f"--metrics value for {name} is empty or has a tab")

    notes = _clean_notes(args.notes)

    result_text = args.result_text_file.read_bytes().decode("utf-8").strip()
    if not result_text:
        raise BookkeepingError(f"{args.result_text_file} is empty")

    old_text, lines, ideas, _ = load_ideas(args.ideas_path)
    idea = _resolve_idea(ideas, args.ident)
    _check_transition(idea, "RUNNING", args.force_from, "log a verdict for")

    result_idx = idea.result_line_index()
    if result_idx is None:
        raise BookkeepingError(
            f"{idea.ident} has no `**Result:**` line in IDEAS.md; add one by hand "
            "(the entry predates the current template)"
        )
    if not idea.result_is_placeholder(lines):
        current = RESULT_LINE_RE.match(lines[result_idx]).group(1)
        raise BookkeepingError(
            f"refusing to overwrite {idea.ident}'s result: its `**Result:**` line "
            f"already reads {current[:80]!r}…, not the {RESULT_PLACEHOLDER!r} "
            "placeholder. Results are append-only — edit by hand if the "
            "supervisor really wants this changed."
        )

    _, _, existing_rows = load_results(args.results_path)
    prior = [r for r in existing_rows if r.split("\t")[1] == idea.ident]
    if prior:
        print(
            f"log_verdict.py: NOTE — {idea.ident} already has "
            f"{len(prior)} row(s) in {args.results_path.name} "
            f"(verdicts: {', '.join(r.split(chr(9))[8] for r in prior)}). "
            "Appending another (legitimate after a re-claim; B5 did this).",
            file=sys.stderr,
        )

    row = "\t".join(
        [args.date, idea.ident, commit, *metrics, verdict, notes]
    )
    if len(row.split("\t")) != len(RESULTS_COLUMNS):
        raise BookkeepingError("constructed row does not have 10 columns")

    new_lines = list(lines)
    new_lines[idea.start] = _replace_status(idea.heading, idea.status_raw, verdict)
    new_lines[result_idx] = f"**Result:** {result_text}"
    new_text = "\n".join(new_lines)

    print(f"--- {args.ident}: [{idea.status_raw}] -> [{verdict}]")
    print(_diff(old_text, new_text, args.ideas_path.name), end="")
    print(f"\n--- appending 1 row to {args.results_path}:")
    print(row)

    if args.dry_run:
        print("\nDRY RUN — nothing written.")
        return 0

    _append_results_row(args.results_path, row, dry_run=False)
    try:
        _write_ideas(args.ideas_path, old_text, new_text, idea.ident)
    except BookkeepingError:
        # results.tsv already grew; undo it so the two files stay consistent.
        raw = args.results_path.read_bytes()
        args.results_path.write_bytes(raw[: -(len(row.encode("utf-8")) + 1)])
        raise

    print(
        f"\nlog_verdict.py: {idea.ident} -> {verdict}; 1 row appended to "
        f"{args.results_path}, status + result updated in {args.ideas_path}"
    )
    print(
        _git_hint(
            [args.ideas_path, args.results_path],
            f"Auto[{idea.ident}]: {verdict} — <one-line result>",
        ),
        end="",
    )
    print(
        "(also `git add research/reports/auto/"
        f"{idea.ident}.md` if you wrote the report — PROTOCOL step 6.)"
    )
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="log_verdict.py",
        description="Claim an idea / log a verdict, deterministically and append-only.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("ident", metavar="ID", help="idea id, e.g. B20")
        p.add_argument("--ideas-path", type=Path, default=IDEAS_PATH)
        p.add_argument(
            "--dry-run",
            action="store_true",
            help="print the exact changes and write nothing",
        )
        p.add_argument(
            "--force-from",
            metavar="STATUS",
            default=None,
            help="supervisor override: allow the transition from this exact "
            "current status (e.g. --force-from CRASH to re-claim)",
        )

    claim = sub.add_parser("claim", help="PENDING -> RUNNING <ts> (PROTOCOL step 2)")
    common(claim)
    claim.add_argument(
        "--ts",
        default=None,
        help="UTC ISO timestamp for the claim (default: now, e.g. 2026-08-07T13:12:18Z)",
    )
    claim.set_defaults(func=cmd_claim)

    verdict = sub.add_parser(
        "verdict",
        help="RUNNING -> LANDED/TABLED/FAILED/CRASH/SUPERSEDED + results.tsv row",
    )
    common(verdict)
    verdict.add_argument("verdict", metavar="VERDICT", choices=[*VERDICTS, *[v.lower() for v in VERDICTS]])
    verdict.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    verdict.add_argument("--date", required=True, help="YYYY-MM-DD for the results.tsv row")
    verdict.add_argument("--commit", required=True, help="commit sha or tag for the row")
    verdict.add_argument("--notes", required=True, help="the results.tsv notes field")
    verdict.add_argument(
        "--result-text-file",
        required=True,
        type=Path,
        help="file whose contents replace the idea's `**Result:** —` placeholder",
    )
    verdict.add_argument(
        "--metrics",
        nargs=5,
        metavar=tuple(METRIC_COLUMNS),
        default=None,
        help=f"the 5 metric columns (default: {METRIC_PLACEHOLDER} in each)",
    )
    verdict.set_defaults(func=cmd_verdict)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (BookkeepingError, QueueFormatError) as exc:
        print(f"log_verdict.py: ERROR — {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"log_verdict.py: ERROR — {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
