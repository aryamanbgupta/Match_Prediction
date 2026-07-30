"""Versioned player-ELO update contracts.

Artifacts built before I9 omitted this field and are interpreted as the
fixed competition-K baseline. New provisional artifacts must declare their
version explicitly and are never compatible with baseline state.
"""
from __future__ import annotations

from typing import Mapping


BASELINE_ELO_UPDATE_VERSION = "fixed_competition_k_v1"
PROVISIONAL_ELO_UPDATE_VERSION = "provisional_linear_120_x4_v1"
ELO_UPDATE_VERSIONS = (
    BASELINE_ELO_UPDATE_VERSION,
    PROVISIONAL_ELO_UPDATE_VERSION,
)


def resolve_elo_update_version(metadata: Mapping[str, object]) -> str:
    """Resolve legacy-missing metadata to the frozen baseline contract."""
    value = str(
        metadata.get("elo_update_version")
        or BASELINE_ELO_UPDATE_VERSION
    ).strip()
    if value not in ELO_UPDATE_VERSIONS:
        raise ValueError(f"unsupported ELO update version {value!r}")
    return value


def assert_elo_update_version(
    metadata: Mapping[str, object],
    *,
    expected: str,
    context: str,
) -> str:
    if expected not in ELO_UPDATE_VERSIONS:
        raise ValueError(f"unsupported ELO update version {expected!r}")
    actual = resolve_elo_update_version(metadata)
    if actual != expected:
        raise RuntimeError(
            f"{context} ELO update mismatch: artifact={actual!r}, "
            f"required={expected!r}"
        )
    return actual


def elo_update_contract(version: str) -> dict[str, str]:
    if version not in ELO_UPDATE_VERSIONS:
        raise ValueError(f"unsupported ELO update version {version!r}")
    return {"elo_update_version": version}


def provider_elo_update_version(provider) -> str | None:
    """Resolve the contract through cache/replay provider wrappers."""
    current = provider
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        declared = getattr(current, "elo_update_version", None)
        if declared is not None:
            return resolve_elo_update_version(
                {"elo_update_version": declared}
            )
        backend = getattr(current, "_backend", None)
        if backend is not None and hasattr(backend, "get_meta"):
            return resolve_elo_update_version(backend.get_meta())
        current = (
            getattr(current, "_provider", None)
            or getattr(current, "_base_provider", None)
        )
    return None
