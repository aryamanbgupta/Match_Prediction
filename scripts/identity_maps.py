"""Validated, versioned identity maps shared by training and inference.

Venue aliases are exact-string mappings. Raw source files remain immutable;
canonicalization happens only when a match is parsed or a fixture is supplied.
"""
from __future__ import annotations

import csv
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
VENUE_ALIAS_VERSION = "venue_aliases_v1"
DEFAULT_VENUE_ALIAS_PATH = (
    REPO_ROOT / "config" / "identity" / "venue_aliases_v1.csv"
)
ALLOWED_STATUSES = frozenset({"active", "proposed", "rejected"})
REQUIRED_COLUMNS = frozenset({"version", "alias", "canonical", "status"})


def _clean_label(value: object, *, fallback: str = "") -> str:
    if value is None:
        return fallback
    cleaned = str(value).strip()
    return cleaned or fallback


@lru_cache(maxsize=8)
def _load_venue_aliases_cached(path_string: str) -> dict[str, str]:
    path = Path(path_string)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = frozenset(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {sorted(missing)}"
            )

        active: dict[str, str] = {}
        seen_aliases: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            version = _clean_label(row.get("version"))
            alias = _clean_label(row.get("alias"))
            canonical = _clean_label(row.get("canonical"))
            status = _clean_label(row.get("status")).lower()

            if version != VENUE_ALIAS_VERSION:
                raise ValueError(
                    f"{path}:{line_number} has version {version!r}; "
                    f"expected {VENUE_ALIAS_VERSION!r}"
                )
            if not alias or not canonical:
                raise ValueError(
                    f"{path}:{line_number} has a blank alias or canonical label"
                )
            if alias == canonical:
                raise ValueError(
                    f"{path}:{line_number} defines a self-alias for {alias!r}"
                )
            if status not in ALLOWED_STATUSES:
                raise ValueError(
                    f"{path}:{line_number} has unsupported status {status!r}"
                )
            if alias in seen_aliases:
                raise ValueError(
                    f"{path}:{line_number} repeats alias {alias!r}"
                )
            seen_aliases.add(alias)

            if status == "active":
                active[alias] = canonical

    chained = set(active) & set(active.values())
    if chained:
        raise ValueError(
            "active canonical targets must not also be active aliases: "
            f"{sorted(chained)}"
        )
    return active


def load_venue_aliases(
    path: Path | str = DEFAULT_VENUE_ALIAS_PATH,
) -> dict[str, str]:
    """Load and validate active venue aliases, returning a defensive copy."""
    resolved = str(Path(path).expanduser().resolve())
    return dict(_load_venue_aliases_cached(resolved))


def canonicalize_venue(
    value: object,
    aliases: Mapping[str, str] | None = None,
    *,
    fallback: str = "unknown",
) -> str:
    """Return the active canonical venue for an exact source label."""
    venue = _clean_label(value, fallback=fallback)
    active_aliases = aliases if aliases is not None else load_venue_aliases()
    return active_aliases.get(venue, venue)


def canonicalize_match_id(
    match_id: object,
    aliases: Mapping[str, str] | None = None,
) -> str:
    """Canonicalize the venue suffix in the repository's match-ID format."""
    value = _clean_label(match_id)
    if not value:
        return value
    active_aliases = aliases if aliases is not None else load_venue_aliases()
    suffixes = sorted(
        (
            ("_" + alias.replace(" ", "_"),
             "_" + canonical.replace(" ", "_"))
            for alias, canonical in active_aliases.items()
        ),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )
    for alias_suffix, canonical_suffix in suffixes:
        if value.endswith(alias_suffix):
            return value[:-len(alias_suffix)] + canonical_suffix
    return value


def venue_alias_contract(
    path: Path | str = DEFAULT_VENUE_ALIAS_PATH,
) -> dict[str, str | int]:
    """Return provenance fields suitable for cache/model metadata."""
    resolved = Path(path).expanduser().resolve()
    active = load_venue_aliases(resolved)
    with resolved.open("rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    return {
        "venue_alias_version": VENUE_ALIAS_VERSION,
        "venue_alias_sha256": digest,
        "venue_alias_active_count": len(active),
    }


def clear_identity_map_caches() -> None:
    """Clear process-local loader caches (primarily useful in tests)."""
    _load_venue_aliases_cached.cache_clear()
