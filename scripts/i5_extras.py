"""Empirical extras contract for the I5 legal/off-bat ball model.

The model predicts six legal-delivery off-bat outcomes. This module builds
and serves the orthogonal scoring channels that the label no longer carries:
wides, no-balls, byes, leg-byes, and penalty runs.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Mapping

from loaders_common import DEFAULT_SPLITS

I5_EXTRAS_CONTRACT = "empirical_extras_v1"


def _distribution(counter: Counter) -> dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {"0": 1.0}
    return {
        str(value): count / total
        for value, count in sorted(counter.items())
    }


def _probabilities(counter: Counter, ordered_keys) -> dict[str, float]:
    total = sum(counter[key] for key in ordered_keys)
    if total <= 0:
        raise ValueError("cannot normalize an empty extras counter")
    return {key: counter[key] / total for key in ordered_keys}


def _noball_distribution(counter: Counter) -> dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {"1,0": 1.0}
    return {
        f"{bowler_runs},{non_bowler_runs}": count / total
        for (bowler_runs, non_bowler_runs), count in sorted(counter.items())
    }


def build_i5_extras_model(
    source_dir: Path,
    *,
    splits: Mapping[str, str] | None = None,
    gender: str | None = "male",
) -> dict:
    """Fit the extras process on the configured validation date window."""
    split_cfg = {**DEFAULT_SPLITS, **(dict(splits or {}))}
    val_start = split_cfg["train_end"]
    val_end = split_cfg["val_end"]

    event_counts = Counter()
    wide_team_runs = Counter()
    noball_extras = Counter()
    dot_extra_types = Counter()
    dot_extra_runs = {
        "none": Counter(),
        "byes": Counter(),
        "legbyes": Counter(),
        "penalty": Counter(),
    }
    n_matches = 0

    for path in sorted(Path(source_dir).glob("*.json")):
        with path.open() as handle:
            match = json.load(handle)
        info = match.get("info", {})
        if gender and info.get("gender") != gender:
            continue
        dates = info.get("dates") or []
        if not dates:
            continue
        match_date = str(dates[0])
        if not (val_start <= match_date < val_end):
            continue
        n_matches += 1

        for innings in match.get("innings", []):
            for over in innings.get("overs", []):
                for delivery in over.get("deliveries", []):
                    extras = delivery.get("extras", {})
                    run_data = delivery.get("runs", {})
                    batter_runs = int(run_data.get("batter", 0))
                    team_runs = int(run_data.get("total", 0))

                    if extras.get("wides", 0) > 0:
                        event_counts["wide"] += 1
                        wide_team_runs[team_runs] += 1
                        continue
                    if extras.get("noballs", 0) > 0:
                        event_counts["no_ball"] += 1
                        bowler_extras = int(extras.get("noballs", 0))
                        non_bowler_extras = (
                            int(extras.get("byes", 0))
                            + int(extras.get("legbyes", 0))
                            + int(extras.get("penalty", 0))
                        )
                        noball_extras[
                            (bowler_extras, non_bowler_extras)
                        ] += 1
                        continue

                    event_counts["legal"] += 1
                    if batter_runs != 0 or delivery.get("wickets"):
                        continue
                    if extras.get("byes", 0) > 0:
                        extra_type = "byes"
                    elif extras.get("legbyes", 0) > 0:
                        extra_type = "legbyes"
                    elif extras.get("penalty", 0) > 0:
                        extra_type = "penalty"
                    else:
                        extra_type = "none"
                    dot_extra_types[extra_type] += 1
                    dot_extra_runs[extra_type][team_runs] += 1

    if n_matches <= 0:
        raise ValueError(
            f"no {gender or 'all-gender'} validation matches in "
            f"[{val_start}, {val_end}) under {source_dir}"
        )

    event_order = ("legal", "wide", "no_ball")
    dot_order = ("none", "byes", "legbyes", "penalty")
    return {
        "contract": I5_EXTRAS_CONTRACT,
        "fit_split": "validation",
        "fit_start_inclusive": val_start,
        "fit_end_exclusive": val_end,
        "gender_filter": gender or "all",
        "n_matches": n_matches,
        "n_deliveries": sum(event_counts.values()),
        "delivery_event_probabilities": _probabilities(
            event_counts, event_order),
        "wide_team_runs_distribution": _distribution(wide_team_runs),
        "noball_extras_distribution": _noball_distribution(noball_extras),
        "legal_dot_extra_probabilities": _probabilities(
            dot_extra_types, dot_order),
        "legal_dot_extra_runs_distributions": {
            key: _distribution(dot_extra_runs[key])
            for key in dot_order
        },
    }


def write_i5_extras_model(model: Mapping, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        json.dump(dict(model), handle, indent=2, sort_keys=True)


class EmpiricalExtrasProcess:
    """Seed-compatible discrete draws from an ``empirical_extras_v1`` file."""

    def __init__(self, model: Mapping):
        if model.get("contract") != I5_EXTRAS_CONTRACT:
            raise ValueError(
                f"unsupported extras contract: {model.get('contract')!r}")
        self.model = dict(model)

    @classmethod
    def from_path(cls, path: Path) -> "EmpiricalExtrasProcess":
        with Path(path).open() as handle:
            return cls(json.load(handle))

    @staticmethod
    def _draw(distribution: Mapping[str, float], rng=random):
        values = list(distribution)
        weights = [float(distribution[value]) for value in values]
        return rng.choices(values, weights=weights)[0]

    def draw_delivery_event(self, rng=random) -> str:
        return self._draw(
            self.model["delivery_event_probabilities"], rng=rng)

    def draw_wide_team_runs(self, rng=random) -> int:
        return int(self._draw(
            self.model["wide_team_runs_distribution"], rng=rng))

    def draw_noball_extras(self, rng=random) -> tuple[int, int]:
        value = self._draw(
            self.model["noball_extras_distribution"], rng=rng)
        bowler_runs, non_bowler_runs = value.split(",", maxsplit=1)
        return int(bowler_runs), int(non_bowler_runs)

    def draw_legal_dot_extra(self, rng=random) -> tuple[str, int]:
        extra_type = self._draw(
            self.model["legal_dot_extra_probabilities"], rng=rng)
        runs = int(self._draw(
            self.model["legal_dot_extra_runs_distributions"][extra_type],
            rng=rng,
        ))
        return extra_type, runs
