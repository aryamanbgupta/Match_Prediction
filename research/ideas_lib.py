"""Shared parsing for the autonomous research loop's queue files.

Used by `research/make_digest.py` (P1) and `research/log_verdict.py` (P2).
Deterministic, dependency-free, no model calls.

`research/IDEAS.md` format contract (as observed 2026-08-07)
-----------------------------------------------------------
Idea entries are level-2 headings::

    ## <ID> [<PRIORITY>] [<STATUS ...>] <title>

with these real-world variations, all of which must parse:

* ``## A1 [P0] [LANDED] Fresh baseline + seed-variance floor``
* ``## B19 [P2] [RUNNING 2026-08-03T13:12:18Z] Fresh-seed confirmation ...``
* ``## D4 [P2] [SUPERSEDED by D15 — do not claim] Wicket-type modeling ...``
* ``## A13 [P3] [SUPERSEDED-by-D16 2026-08-03] Sim dispersion calibration ...``
* ``## D18 [P3] [FAILED 2026-08-01 — INTERACTIVE] No-weights-adapted ...``
* ``## I1 [DONE 2026-07-16] Toss both-branch averaging`` (no priority bracket)
* ``## I12 [INTERACTIVE] Women's-corpus model (new track)``
* ``## I14 [INTERACTIVE — first integration test FAILED 2026-08-02] ...``
  (followed by an italic ``*(Status: ...)*`` supervisor note before the body)

Level-2 headings whose first token is not an idea id (e.g.
``## Combination ideas (C-series)``) are section prose, not ideas.  Level-1
headings (``# Research Queue``, ``# D-series: ...``, ``# Interactive backlog``)
open prose sections too.  Headings inside fenced code blocks are ignored.

The status keyword is the leading run of capitals in the status bracket, so
``SUPERSEDED-by-D16 2026-08-03`` and ``SUPERSEDED by D15 — do not claim`` both
normalise to ``SUPERSEDED``.

`research/results.tsv` format contract
--------------------------------------
Ten tab-separated columns, one header line, one row per verdict::

    date  idea  commit  ll_50k  market_ll  roi_50k_pct  roi_ci  n_bets  verdict  notes

Sim/prop ideas have no match-level numbers and use the literal placeholder
``(sim-gate)`` in all five metric columns (``(crash)`` is also attested).  An
idea may legitimately appear more than once (B5: CRASH, then TABLED after a
re-claim).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
IDEAS_PATH = REPO_ROOT / "research" / "IDEAS.md"
RESULTS_PATH = REPO_ROOT / "research" / "results.tsv"
DIGEST_PATH = REPO_ROOT / "research" / "digest.md"
PROGRAM_PATH = REPO_ROOT / "program.md"

#: An idea id: one or two letters then digits (A1, B19, D18, I20, C1).
IDEA_ID_RE = re.compile(r"^[A-Z]{1,2}[0-9]{1,3}$")
#: A leading ``[...]`` bracket group on a heading.
LEADING_BRACKET_RE = re.compile(r"^\[([^\]]*)\]\s*")
PRIORITY_RE = re.compile(r"^P[0-9]$")
#: Leading run of capitals inside a status bracket.
STATUS_KEYWORD_RE = re.compile(r"^[A-Z]+")
FENCE_RE = re.compile(r"^\s*(```+|~~~+)")
RESULT_LINE_RE = re.compile(r"^\*\*Result:\*\*\s*(.*)$")

#: The placeholder that marks an idea whose result has not been written yet.
RESULT_PLACEHOLDER = "—"

#: Statuses the digest reproduces in full.
FULL_TEXT_STATUSES = ("RUNNING", "PENDING", "TABLED")
#: Statuses the digest reduces to one line each.
ONE_LINE_STATUSES = (
    "LANDED",
    "FAILED",
    "CRASH",
    "SUPERSEDED",
    "DONE",
    "DEFERRED",
    "INTERACTIVE",
)
#: Verdicts `log_verdict.py verdict` accepts.
VERDICTS = ("LANDED", "TABLED", "FAILED", "CRASH", "SUPERSEDED")

RESULTS_COLUMNS = (
    "date",
    "idea",
    "commit",
    "ll_50k",
    "market_ll",
    "roi_50k_pct",
    "roi_ci",
    "n_bets",
    "verdict",
    "notes",
)
#: Indices of the five metric columns between `commit` and `verdict`.
METRIC_COLUMNS = RESULTS_COLUMNS[3:8]
METRIC_PLACEHOLDER = "(sim-gate)"


class QueueFormatError(RuntimeError):
    """Raised when IDEAS.md / results.tsv does not match the format contract."""


@dataclass
class Idea:
    """One ``## <ID> ...`` entry in IDEAS.md."""

    ident: str
    priority: str | None
    status: str  # normalised keyword, e.g. "RUNNING"
    status_raw: str  # full bracket contents, e.g. "RUNNING 2026-08-03T13:12:18Z"
    title: str
    heading: str  # the heading line, verbatim
    start: int  # 0-based index of the heading line
    end: int  # exclusive 0-based index of the line after the block
    lines: list[str] = field(default_factory=list)  # heading + body, verbatim

    @property
    def body_lines(self) -> list[str]:
        return self.lines[1:]

    def trimmed_lines(self) -> list[str]:
        """Heading + body with trailing blank lines / `---` separators dropped."""
        out = list(self.lines)
        while out and (not out[-1].strip() or out[-1].strip() == "---"):
            out.pop()
        return out

    @property
    def text(self) -> str:
        return "\n".join(self.trimmed_lines())

    @property
    def sort_key(self) -> tuple:
        """Priority first (P0 before P3, unprioritised last), then id."""
        prio = int(self.priority[1:]) if self.priority else 99
        series = re.match(r"^([A-Z]+)([0-9]+)$", self.ident)
        return (prio, series.group(1), int(series.group(2)))

    def result_line_index(self) -> int | None:
        """Absolute index of this idea's ``**Result:**`` line, if it has one."""
        for offset, line in enumerate(self.body_lines):
            if RESULT_LINE_RE.match(line):
                return self.start + 1 + offset
        return None

    def result_is_placeholder(self, lines: Sequence[str]) -> bool:
        idx = self.result_line_index()
        if idx is None:
            return False
        match = RESULT_LINE_RE.match(lines[idx])
        return bool(match) and match.group(1).strip() == RESULT_PLACEHOLDER

    def one_line(self) -> str:
        prio = self.priority or "--"
        return f"{self.ident:<5} {prio:<3} {self.status_raw:<38} {self.title}"


@dataclass
class ProseBlock:
    """A non-idea block: the file preamble, a `# X-series` header, a `## ` section."""

    heading: str | None
    start: int
    end: int
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        out = list(self.lines)
        while out and not out[-1].strip():
            out.pop()
        return "\n".join(out)


def _split_heading(heading: str) -> tuple[str, str | None, str, str, str] | None:
    """Parse a ``## `` heading into (id, priority, status_raw, status, title).

    Returns None when the heading is not an idea entry (section prose).
    """
    rest = heading[3:].strip()
    parts = rest.split(None, 1)
    if not parts or not IDEA_ID_RE.match(parts[0]):
        return None
    ident = parts[0]
    remainder = parts[1] if len(parts) > 1 else ""

    priority: str | None = None
    status_raw: str | None = None
    while True:
        match = LEADING_BRACKET_RE.match(remainder)
        if not match:
            break
        content = match.group(1).strip()
        remainder = remainder[match.end() :]
        if priority is None and PRIORITY_RE.match(content):
            priority = content
            continue
        status_raw = content
        break  # anything after the status bracket belongs to the title

    if status_raw is None:
        raise QueueFormatError(
            f"idea heading has no status bracket: {heading!r}"
        )
    keyword = STATUS_KEYWORD_RE.match(status_raw)
    if not keyword:
        raise QueueFormatError(
            f"status bracket does not start with a keyword: {heading!r}"
        )
    return ident, priority, status_raw, keyword.group(0), remainder.strip()


def parse_ideas(text: str) -> tuple[list[Idea], list[ProseBlock]]:
    """Split IDEAS.md into idea entries and prose blocks, preserving line indices."""
    lines = text.split("\n")
    fence: str | None = None
    boundaries: list[int] = []  # indices of heading lines, outside code fences
    for idx, line in enumerate(lines):
        fence_match = FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)[0]
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            continue
        if fence is None and (line.startswith("# ") or line.startswith("## ")):
            boundaries.append(idx)

    ideas: list[Idea] = []
    prose: list[ProseBlock] = []
    starts = boundaries or []
    if not starts or starts[0] != 0:
        end = starts[0] if starts else len(lines)
        prose.append(ProseBlock(None, 0, end, lines[0:end]))
    for pos, start in enumerate(starts):
        end = starts[pos + 1] if pos + 1 < len(starts) else len(lines)
        heading = lines[start]
        parsed = _split_heading(heading) if heading.startswith("## ") else None
        if parsed is None:
            prose.append(ProseBlock(heading, start, end, lines[start:end]))
            continue
        ident, priority, status_raw, status, title = parsed
        ideas.append(
            Idea(
                ident=ident,
                priority=priority,
                status=status,
                status_raw=status_raw,
                title=title,
                heading=heading,
                start=start,
                end=end,
                lines=lines[start:end],
            )
        )

    seen: dict[str, int] = {}
    for idea in ideas:
        if idea.ident in seen:
            raise QueueFormatError(
                f"duplicate idea id {idea.ident} at lines "
                f"{seen[idea.ident] + 1} and {idea.start + 1}"
            )
        seen[idea.ident] = idea.start
    return ideas, prose


def load_ideas(path: Path) -> tuple[str, list[str], list[Idea], list[ProseBlock]]:
    """Read IDEAS.md and return (raw_text, lines, ideas, prose_blocks)."""
    text = path.read_text(encoding="utf-8")
    ideas, prose = parse_ideas(text)
    return text, text.split("\n"), ideas, prose


def find_idea(ideas: Iterable[Idea], ident: str) -> Idea:
    for idea in ideas:
        if idea.ident == ident:
            return idea
    raise QueueFormatError(f"no idea with id {ident!r} in IDEAS.md")


def load_results(path: Path) -> tuple[str, str, list[str]]:
    """Read results.tsv and return (raw_text, header_line, data_rows)."""
    text = path.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        raise QueueFormatError(f"{path} does not end with a newline")
    rows = text.split("\n")[:-1]
    if not rows:
        raise QueueFormatError(f"{path} is empty")
    header = rows[0]
    if header.split("\t") != list(RESULTS_COLUMNS):
        raise QueueFormatError(
            f"{path} header is {header.split(chr(9))!r}, "
            f"expected {list(RESULTS_COLUMNS)!r}"
        )
    for lineno, row in enumerate(rows[1:], start=2):
        fields = row.split("\t")
        if len(fields) != len(RESULTS_COLUMNS):
            raise QueueFormatError(
                f"{path}:{lineno} has {len(fields)} columns, "
                f"expected {len(RESULTS_COLUMNS)}"
            )
    return text, header, rows[1:]
