"""Fail-closed enumeration of consumed fixture pools (ODDS1, 2026-08-14).

The sealing guards ("a new holdout must not absorb evaluated fixtures",
"golden must not absorb consumed forward fixtures") previously glob()'d
directories that may be absent from a light checkout — a missing dir
contributed zero fixtures with no error, so an integrity report could
assert "overlap: 0" when nothing had actually been checked. Enumerating
pools through this helper raises instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def require_pool_ids(
    pool_dirs: Iterable[Path],
    context: str,
    *,
    use_name: bool = False,
) -> tuple[set[str], dict[str, int]]:
    """Return (union of fixture ids, per-pool counts) for `pool_dirs`.

    Raises RuntimeError if any listed pool is missing or empty: a sealing
    check that cannot see a consumed pool must not report zero overlap.
    ``use_name=True`` collects file names (with .json); the default collects
    stems (Cricsheet ids).
    """
    found: set[str] = set()
    counts: dict[str, int] = {}
    bad: list[Path] = []
    for directory in pool_dirs:
        directory = Path(directory)
        if directory.is_dir():
            ids = {
                path.name if use_name else path.stem
                for path in directory.glob("*.json")
            }
        else:
            ids = set()
        counts[str(directory)] = len(ids)
        if not ids:
            bad.append(directory)
        found |= ids
    if bad:
        raise RuntimeError(
            f"{context}: consumed pool(s) missing or empty: "
            + ", ".join(str(b) for b in bad)
            + ". Regenerate the pool(s) on this checkout (they are "
            "regenerable; see CLAUDE.md eval-set docs) or explicitly amend "
            "the pool list — do not run a sealing check blind."
        )
    return found, counts
