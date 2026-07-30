"""Fail-closed I8 adapter for the frozen v1.2 simulator.

I8 keeps the existing 114-feature simulation implementation unchanged and
overwrites the 18 schema-v5 feature slots after the base row is assembled.
Keeping this adapter separate avoids changing the scoring code pinned by the
consumed forward protocol.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from identity_maps import assert_venue_alias_contract
from sim_v1_2 import MatchState, XGBoostModelV2


_OUTCOME_SUFFIXES = ("0", "1", "2", "4", "6", "w")
I8_FEATURE_COLUMNS = (
    *(f"batter_phase_p{suffix}" for suffix in _OUTCOME_SUFFIXES),
    *(f"bowler_phase_p{suffix}" for suffix in _OUTCOME_SUFFIXES),
    *(f"h2h_p{suffix}" for suffix in _OUTCOME_SUFFIXES),
)


def _schema_version(provider) -> int | None:
    """Return the underlying SQLite schema through optional memo wrappers."""
    current = provider
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        backend = getattr(current, "_backend", None)
        if backend is not None:
            value = getattr(backend, "schema_version", None)
            return int(value) if value is not None else None
        current = getattr(current, "_provider", None)
    return None


def get_i8_outcome_dists(
    provider,
    striker_pid,
    bowler_pid,
    match_date,
    balls_bowled: int,
    *,
    k_player: float,
    k_phase: float,
    k_h2h: float,
) -> Dict[str, float]:
    """Read the 18 pre-ball I8 distributions from a schema-v5 provider."""
    if provider is None or _schema_version(provider) != 5:
        raise RuntimeError("I8 simulation requires SQLite schema 5")

    values: Dict[str, float] = {}
    values.update(provider.get_batter_phase_outcome_dist(
        striker_pid,
        match_date,
        balls_bowled,
        k_player=k_player,
        k_phase=k_phase,
    ))
    values.update(provider.get_bowler_phase_outcome_dist(
        bowler_pid,
        match_date,
        balls_bowled,
        k_player=k_player,
        k_phase=k_phase,
    ))
    values.update(provider.get_h2h_outcome_dist(
        striker_pid,
        bowler_pid,
        match_date,
        k_player=k_player,
        k_h2h=k_h2h,
    ))

    missing = sorted(set(I8_FEATURE_COLUMNS) - values.keys())
    if missing:
        raise RuntimeError(
            "I8 stats provider omitted required distributions: "
            + ", ".join(missing)
        )
    return values


class XGBoostModelI8(XGBoostModelV2):
    """XGBoost simulator model for the isolated I8 feature contract."""

    def __init__(self, *args, **kwargs):
        provider = kwargs.get("stats_provider")
        if provider is None or _schema_version(provider) != 5:
            raise RuntimeError("I8 simulation requires SQLite schema 5")

        super().__init__(*args, **kwargs)

        missing = sorted(
            set(I8_FEATURE_COLUMNS) - set(self.feature_columns)
        )
        if missing:
            raise RuntimeError(
                "I8 feature contract is incomplete; missing: "
                + ", ".join(missing)
            )

        model_path = Path(kwargs.get("model_path", args[0] if args else ""))
        suffix = model_path.stem.removeprefix("xgboost_model_")
        sidecar = model_path.parent / f"outcome_dist_config_{suffix}.json"
        if not sidecar.exists():
            raise RuntimeError(f"I8 shrinkage sidecar is missing: {sidecar}")

        training_contract_path = (
            model_path.parent / f"training_contract_{suffix}.json"
        )
        if not training_contract_path.exists():
            raise RuntimeError(
                f"I8 training contract is missing: {training_contract_path}"
            )

        try:
            config = json.loads(sidecar.read_text())
            required = ("k_player", "k_venue", "k_phase", "k_h2h")
            absent = [key for key in required if key not in config]
            if absent:
                raise KeyError(", ".join(absent))
            self.k_player = float(config["k_player"])
            self.k_venue = float(config["k_venue"])
            self.k_phase = float(config["k_phase"])
            self.k_h2h = float(config["k_h2h"])
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise RuntimeError(
                f"Invalid I8 shrinkage sidecar {sidecar}: {exc}"
            ) from exc

        try:
            training_contract = json.loads(
                training_contract_path.read_text()
            )
            if training_contract.get("data_version") != "i8":
                raise ValueError(
                    "data_version must be 'i8', found "
                    f"{training_contract.get('data_version')!r}"
                )
            if int(training_contract.get("cache_schema_version", -1)) != 5:
                raise ValueError(
                    "cache_schema_version must be 5, found "
                    f"{training_contract.get('cache_schema_version')!r}"
                )
            assert_venue_alias_contract(
                training_contract.get("venue_identity", {}),
                context="I8 model",
            )
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise RuntimeError(
                f"Invalid I8 training contract "
                f"{training_contract_path}: {exc}"
            ) from exc

        self._i8_feature_indices = {
            name: self.feature_columns.index(name)
            for name in I8_FEATURE_COLUMNS
        }

    def extract_features(self, state: MatchState):
        row = super().extract_features(state)
        values = get_i8_outcome_dists(
            self.stats_provider,
            state.current_striker.player_id,
            state.current_bowler.player_id,
            state.match_date,
            state.balls,
            k_player=self.k_player,
            k_phase=self.k_phase,
            k_h2h=self.k_h2h,
        )
        for name, index in self._i8_feature_indices.items():
            row[index] = values[name]
        return row


class FailClosedXGBoostModelI8(XGBoostModelI8):
    """CLI variant that cannot be caught by the legacy dummy fallback."""

    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
        except Exception as exc:
            raise SystemExit(f"I8 model loading aborted: {exc}") from exc
