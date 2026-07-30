"""Stable match identity shared by materialization, odds, and evaluation.

New artifacts use the immutable Cricsheet file stem as their primary
``match_id``.  The historical ``date_team1_team2_venue`` string remains useful
for display and for reading frozen artifacts, but it is not unique for
same-day doubleheaders and must never be used as a new primary key.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from identity_maps import canonicalize_match_id, canonicalize_venue


MATCH_IDENTITY_VERSION = "cricsheet_primary_v1"
LEGACY_MATCH_IDENTITY_VERSION = "synthetic_fixture_v1"

def _clean(value: object) -> str:
    return str(value or "").strip()


def build_display_match_id(
    date_text: object,
    team1: object,
    team2: object,
    venue: object,
) -> str:
    """Build the canonical human-readable fixture label used historically."""
    date_value = _clean(date_text)
    team1_value = _clean(team1)
    team2_value = _clean(team2)
    if not date_value or not team1_value or not team2_value:
        raise ValueError("display match identity requires date and two teams")
    venue_value = canonicalize_venue(venue)
    return (
        f"{date_value}_{team1_value}_{team2_value}_{venue_value}"
        .replace(" ", "_")
    )


def build_display_match_id_from_info(info: Mapping[str, Any]) -> str:
    teams = info.get("teams") or []
    dates = info.get("dates") or []
    if len(teams) != 2 or not dates:
        raise ValueError("match info must contain two teams and a date")
    return build_display_match_id(
        dates[0],
        teams[0],
        teams[1],
        info.get("venue"),
    )


@dataclass(frozen=True)
class MatchIdentity:
    """Resolved primary/display identity for one artifact row."""

    primary_id: str
    cricsheet_id: str | None
    display_id: str
    version: str

    def as_fields(self) -> dict[str, str]:
        if not self.cricsheet_id:
            raise ValueError("new match identity requires a Cricsheet ID")
        return {
            "match_id": self.primary_id,
            "cricsheet_id": self.cricsheet_id,
            "display_match_id": self.display_id,
            "match_identity_version": self.version,
        }


def new_match_identity(
    cricsheet_id: object,
    *,
    date_text: object,
    team1: object,
    team2: object,
    venue: object,
) -> MatchIdentity:
    primary = _clean(cricsheet_id)
    if not primary:
        raise ValueError("Cricsheet ID must be non-empty")
    return MatchIdentity(
        primary_id=primary,
        cricsheet_id=primary,
        display_id=build_display_match_id(
            date_text,
            team1,
            team2,
            venue,
        ),
        version=MATCH_IDENTITY_VERSION,
    )


def resolve_match_identity(row: Mapping[str, Any]) -> MatchIdentity:
    """Resolve either a new contract row or a frozen legacy row.

    New rows fail closed unless ``match_id == cricsheet_id`` and carry an
    explicit display ID. Legacy rows prefer a present Cricsheet ID as their
    true primary identity while retaining the historical ``match_id`` as the
    display alias.
    """
    version = _clean(row.get("match_identity_version"))
    raw_match_id = _clean(row.get("match_id"))
    cricsheet_id = _clean(row.get("cricsheet_id")) or None
    explicit_display = _clean(row.get("display_match_id"))

    if version == MATCH_IDENTITY_VERSION:
        if not raw_match_id or not cricsheet_id:
            raise ValueError(
                "cricsheet_primary_v1 requires match_id and cricsheet_id"
            )
        if raw_match_id != cricsheet_id:
            raise ValueError(
                "cricsheet_primary_v1 requires match_id == cricsheet_id"
            )
        if not explicit_display:
            raise ValueError(
                "cricsheet_primary_v1 requires display_match_id"
            )
        return MatchIdentity(
            primary_id=cricsheet_id,
            cricsheet_id=cricsheet_id,
            display_id=explicit_display,
            version=version,
        )

    if version and version != LEGACY_MATCH_IDENTITY_VERSION:
        raise ValueError(f"unsupported match identity version {version!r}")
    if not raw_match_id and not cricsheet_id:
        raise ValueError("artifact row has no match identity")
    return MatchIdentity(
        primary_id=cricsheet_id or raw_match_id,
        cricsheet_id=cricsheet_id,
        display_id=explicit_display or raw_match_id or cricsheet_id or "",
        version=LEGACY_MATCH_IDENTITY_VERSION,
    )


def build_primary_lookup(
    rows: Iterable[Mapping[str, Any]],
    *,
    context: str,
) -> dict[str, Mapping[str, Any]]:
    """Index rows by stable primary ID, rejecting duplicates."""
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        identity = resolve_match_identity(row)
        if identity.primary_id in result:
            raise ValueError(
                f"{context}: duplicate primary match ID "
                f"{identity.primary_id!r}"
            )
        result[identity.primary_id] = row
    return result


class CompatibilityAliasLookup(dict):
    """Primary/legacy alias index that raises only when ambiguity is used."""

    def __init__(self, *args, ambiguous: Iterable[str], context: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.ambiguous = frozenset(ambiguous)
        self.context = context

    def _key(self, key: object) -> str:
        canonical = canonicalize_match_id(key)
        if canonical in self.ambiguous:
            raise ValueError(
                f"{self.context}: ambiguous match alias {canonical!r}"
            )
        return canonical

    def __contains__(self, key: object) -> bool:
        return super().__contains__(self._key(key))

    def __getitem__(self, key: object):
        return super().__getitem__(self._key(key))

    def get(self, key: object, default=None):
        return super().get(self._key(key), default)


def build_compatibility_alias_lookup(
    rows: Iterable[Mapping[str, Any]],
    *,
    context: str,
    value_fn: Callable[[Mapping[str, Any]], Any] | None = None,
) -> CompatibilityAliasLookup:
    """Index primaries and unique aliases; defer ambiguity errors to lookup.

    This lets a large artifact contain unrelated doubleheaders while still
    supporting an unambiguous legacy join for the requested fixture.
    """
    result: dict[str, Any] = {}
    owners: dict[str, str] = {}
    ambiguous: set[str] = set()
    seen_primary: set[str] = set()
    for row in rows:
        identity = resolve_match_identity(row)
        primary = canonicalize_match_id(identity.primary_id)
        if primary in seen_primary:
            raise ValueError(
                f"{context}: duplicate primary match ID {primary!r}"
            )
        seen_primary.add(primary)
        value = value_fn(row) if value_fn is not None else row
        aliases = {
            primary,
            canonicalize_match_id(identity.display_id),
            canonicalize_match_id(row.get("match_id")),
            canonicalize_match_id(row.get("cricsheet_id")),
        }
        for alias in sorted(value for value in aliases if value):
            previous_owner = owners.get(alias)
            if previous_owner is not None and previous_owner != primary:
                ambiguous.add(alias)
                result.pop(alias, None)
                continue
            if alias not in ambiguous:
                owners[alias] = primary
                result[alias] = value
    return CompatibilityAliasLookup(
        result,
        ambiguous=ambiguous,
        context=context,
    )


def build_unambiguous_alias_lookup(
    rows: Iterable[Mapping[str, Any]],
    *,
    context: str,
) -> dict[str, Mapping[str, Any]]:
    """Index primary and display IDs, rejecting every one-to-many alias.

    This is intended only for compatibility joins against frozen synthetic-ID
    artifacts. New joins should use :func:`build_primary_lookup`.
    """
    result = build_compatibility_alias_lookup(rows, context=context)
    if result.ambiguous:
        alias = sorted(result.ambiguous)[0]
        raise ValueError(
            f"{context}: ambiguous match alias {alias!r}"
        )
    return dict(result)


def identity_contract() -> dict[str, str]:
    return {
        "match_identity_version": MATCH_IDENTITY_VERSION,
        "primary_key": "cricsheet_id",
        "display_key": "display_match_id",
    }


def legacy_identity_contract() -> dict[str, str]:
    """Describe frozen artifacts that predate Cricsheet-primary IDs."""
    return {
        "match_identity_version": LEGACY_MATCH_IDENTITY_VERSION,
        "primary_key": "match_id",
        "display_key": "match_id",
    }
