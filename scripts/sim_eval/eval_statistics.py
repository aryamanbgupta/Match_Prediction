"""Shared evaluation statistics and betting-decision contracts.

I3 replaces per-match i.i.d. headline intervals with whole-competition block
resampling. A block is a contiguous Cricsheet event/tournament run; when event
metadata is unavailable, an unordered team-pair plus season is used.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from identity_maps import canonicalize_venue
from sim_eval.market_math import CostModel, settle_flat, settle_kelly


BOOTSTRAP_CONTRACT_VERSION = "tournament_time_block_v1"
# Stamped instead of BOOTSTRAP_CONTRACT_VERSION whenever any row's block came
# from fallback_competition_cluster (team-pair/singleton) rather than a real
# tournament block — those inflate the block count, so a CI produced under
# this stamp must never be read as contract-conforming.
BOOTSTRAP_CONTRACT_FALLBACK = "tournament_time_block_v1_fallback_pair_blocks"
DEFAULT_BOOTSTRAP_SEED = 42
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
MAX_EVENT_GAP_DAYS = 120
MIN_RECOMMENDED_CLUSTERS = 10
# Sentinel for a display alias shared by a same-day doubleheader: clustering
# through such an alias is ambiguous and must fail loudly at use.
AMBIGUOUS_CLUSTER_ALIAS = "__ambiguous_doubleheader_alias__"
_DATE_PREFIX = re.compile(r"^(\d{4})-(\d{2})-(\d{2})(?:_|$)")


def settle_flat_policy(
    bet_team: str | None,
    odds: float,
    winner: str | None,
    cost: CostModel,
) -> float | None:
    """Settle the flat policy, including its legacy zero-cost boundary."""
    if cost == CostModel.none() and odds == 1.0:
        if winner is None:
            return None
        if bet_team is None:
            return 0.0
        return 0.0 if bet_team == winner else -1.0
    return settle_flat(bet_team, odds, winner, cost)


def settle_kelly_policy(
    fraction: float,
    odds: float,
    bet_team: str,
    winner: str | None,
    cost: CostModel,
) -> float | None:
    """Settle Kelly sizing, including its legacy zero-cost boundary."""
    if fraction <= 0.0 or winner is None:
        return None
    if cost == CostModel.none() and odds == 1.0:
        return 0.0 if bet_team == winner else -fraction
    return settle_kelly(fraction, odds, bet_team, winner, cost)


def _field(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def season_label(date_text: str) -> str:
    """Return a July-to-June cricket-season label."""
    match = _DATE_PREFIX.match(str(date_text))
    if not match:
        return "unknown"
    year, month = int(match.group(1)), int(match.group(2))
    start = year if month >= 7 else year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def match_id_from_info(info: Mapping[str, Any]) -> str:
    teams = info.get("teams") or []
    dates = info.get("dates") or []
    if len(teams) != 2 or not dates:
        raise ValueError("match info must contain two teams and a date")
    venue = canonicalize_venue(info.get("venue"))
    return f"{dates[0]}_{teams[0]}_{teams[1]}_{venue}".replace(" ", "_")


def competition_cluster_from_info(info: Mapping[str, Any]) -> str:
    """Build a single-record event/team fallback cluster.

    Dataset-level loading uses inactivity-gap blocks instead; this helper is
    for isolated records that do not provide neighboring event dates.
    """
    teams = [str(team) for team in (info.get("teams") or [])]
    dates = info.get("dates") or []
    date_text = str(dates[0]) if dates else "unknown"
    event = info.get("event") or {}
    event_name = event.get("name") if isinstance(event, Mapping) else None
    if event_name:
        identity = f"event:{str(event_name).strip()}"
    elif len(teams) == 2:
        identity = "pair:" + "|".join(sorted(teams))
    else:
        identity = "match:unknown"
    return f"{identity}|season:{season_label(date_text)}"


def fallback_competition_cluster(record: Any) -> str:
    """Build a deterministic team-pair block for legacy eval rows."""
    match_id = str(_field(record, "match_id", "unknown"))
    teams = _field(record, "teams")
    if not teams:
        team1 = _field(record, "team1")
        team2 = _field(record, "team2")
        teams = [team1, team2] if team1 and team2 else []
    if len(teams) == 2:
        identity = "pair:" + "|".join(sorted(str(team) for team in teams))
    else:
        # No metadata: use a singleton rather than silently merge unrelated
        # fixtures. This reduces to the legacy i.i.d. bootstrap for such rows.
        identity = f"match:{match_id}"
    return f"{identity}|season:{season_label(match_id)}"


def load_competition_clusters(source_dir: Path | str) -> dict[str, str]:
    """Return generated eval match ID -> competition time block.

    Matches sharing an event name remain in one block until that event has
    been inactive for more than 120 days. This keeps leagues spanning a
    calendar/season boundary together while separating later editions.
    """
    source = Path(source_dir)
    if not source.is_dir():
        raise FileNotFoundError(source)
    records: list[tuple[str, str, str, datetime]] = []
    for path in sorted(source.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
            info = payload["info"]
            primary_id = path.stem
            display_id = match_id_from_info(info)
            teams = [str(team) for team in (info.get("teams") or [])]
            event = info.get("event") or {}
            event_name = (
                event.get("name") if isinstance(event, Mapping) else None
            )
            if event_name:
                identity = f"event:{str(event_name).strip()}"
            elif len(teams) == 2:
                identity = "pair:" + "|".join(sorted(teams))
            else:
                identity = f"match:{primary_id}"
            match_date = datetime.strptime(
                str(info["dates"][0]),
                "%Y-%m-%d",
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        records.append((primary_id, display_id, identity, match_date))

    lookup: dict[str, str] = {}
    by_identity: dict[str, list[tuple[str, datetime]]] = defaultdict(list)
    display_ids: dict[str, list[str]] = defaultdict(list)
    for primary_id, display_id, identity, match_date in records:
        by_identity[identity].append((primary_id, match_date))
        display_ids[display_id].append(primary_id)
    for identity, members in sorted(by_identity.items()):
        block_start: Optional[datetime] = None
        previous_date: Optional[datetime] = None
        for match_id, match_date in sorted(
            members,
            key=lambda item: (item[1], item[0]),
        ):
            if (
                previous_date is None
                or (match_date - previous_date).days > MAX_EVENT_GAP_DAYS
            ):
                block_start = match_date
            cluster_id = (
                f"{identity}|block_start:{block_start.strftime('%Y-%m-%d')}"
            )
            previous = lookup.get(match_id)
            if previous is not None and previous != cluster_id:
                raise RuntimeError(
                    f"conflicting competition clusters for {match_id}: "
                    f"{previous!r} vs {cluster_id!r}"
                )
            lookup[match_id] = cluster_id
            previous_date = match_date
    # Frozen eval JSONs carry display IDs. Preserve compatibility only where
    # that alias resolves to exactly one Cricsheet primary; doubleheader
    # aliases map to a sentinel so a join through them fails loudly in
    # cluster_id_for_record instead of silently landing in the team-pair
    # fallback block.
    for display_id, primary_ids in display_ids.items():
        if len(primary_ids) == 1:
            lookup[display_id] = lookup[primary_ids[0]]
        elif display_id not in lookup:
            lookup[display_id] = AMBIGUOUS_CLUSTER_ALIAS
    return lookup


def cluster_id_with_resolution(
    record: Any,
    cluster_lookup: Optional[Mapping[str, str]] = None,
) -> tuple[str, str]:
    """Return (cluster_id, resolution) where resolution is one of
    "stamped" (record carries its own competition_cluster_id),
    "lookup" (resolved through the corpus lookup), or
    "fallback" (team-pair/singleton block — NOT a tournament block)."""
    explicit = _field(record, "competition_cluster_id")
    if explicit is None:
        explicit = _field(record, "cluster_id")
    if explicit:
        return str(explicit), "stamped"
    identity_keys = [
        str(value) for value in (
            _field(record, "match_id"),
            _field(record, "cricsheet_id"),
            _field(record, "display_match_id"),
        )
        if value
    ]
    if cluster_lookup:
        for match_id in identity_keys:
            if match_id in cluster_lookup:
                cluster = str(cluster_lookup[match_id])
                if cluster == AMBIGUOUS_CLUSTER_ALIAS:
                    raise RuntimeError(
                        f"match alias {match_id!r} is shared by a same-day "
                        "doubleheader; a competition cluster cannot be "
                        "assigned through it — re-key the eval artifact "
                        "with Cricsheet primary IDs"
                    )
                return cluster, "lookup"
    return fallback_competition_cluster(record), "fallback"


def cluster_id_for_record(
    record: Any,
    cluster_lookup: Optional[Mapping[str, str]] = None,
) -> str:
    return cluster_id_with_resolution(record, cluster_lookup)[0]


def flat_bet_team(
    record: Any,
    edge_threshold: float = 0.0,
) -> Optional[str]:
    """Return the explicitly placed flat-bet team, or None.

    New records persist ``bet_placed`` and ``bet_team``. Legacy records are
    reconstructed from the same edge/odds contract that generated P&L.
    P&L itself is never used as the placement sentinel.
    """
    explicit_placed = _field(record, "bet_placed")
    explicit_team = _field(record, "bet_team")
    if explicit_placed is False:
        return None
    if explicit_placed is True and explicit_team:
        return str(explicit_team)

    realized_pnl = _field(record, "realized_pnl")
    actual_winner = _field(record, "actual_winner")
    edge = _field(record, "edge") or {}
    market_odds = _field(record, "market_odds") or {}
    if realized_pnl is None or not actual_winner or not edge or not market_odds:
        return None
    best_team: Optional[str] = None
    best_edge = float("-inf")
    for team, value in edge.items():
        try:
            numeric_edge = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric_edge) and numeric_edge > best_edge:
            best_team = str(team)
            best_edge = numeric_edge
    if best_team is None:
        return None
    try:
        odds = float(market_odds[best_team])
    except (KeyError, TypeError, ValueError):
        return None
    if (
        best_edge <= edge_threshold
        or not np.isfinite(odds)
        or odds < 1.0
    ):
        return None
    return best_team


def flat_bet_won(record: Any, edge_threshold: float = 0.0) -> bool:
    team = flat_bet_team(record, edge_threshold=edge_threshold)
    return team is not None and team == _field(record, "actual_winner")


def bootstrap_mean_ci(
    values: Sequence[float],
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    ci: float = 0.95,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    clusters: Optional[Sequence[Any]] = None,
    strata: Optional[Sequence[Any]] = None,
) -> tuple[float, float]:
    """Percentile CI for a mean, optionally resampling whole blocks.

    With ``clusters``, each unique cluster is sampled with replacement and all
    of its observations are carried into the replicate. The replicate mean is
    the observation-weighted mean (for flat one-unit ROI, total P&L / sampled
    bets). With no clusters, the legacy i.i.d./stratified behavior is retained
    for low-level compatibility tests.
    """
    if not values or n_resamples <= 0:
        return (float("nan"), float("nan"))
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    rng = np.random.default_rng(seed)

    if clusters is not None:
        if len(clusters) != n:
            raise ValueError(
                f"clusters length {len(clusters)} does not match values "
                f"length {n}"
            )
        cluster_to_idx: dict[Any, list[int]] = defaultdict(list)
        for index, cluster in enumerate(clusters):
            cluster_to_idx[cluster].append(index)
        cluster_items = list(cluster_to_idx.items())
        if not cluster_items:
            return (float("nan"), float("nan"))

        if strata is not None:
            if len(strata) != n:
                raise ValueError(
                    f"strata length {len(strata)} does not match values "
                    f"length {n}"
                )
            stratum_clusters: dict[Any, list[np.ndarray]] = defaultdict(list)
            for _cluster, members in cluster_items:
                labels = {strata[index] for index in members}
                if len(labels) != 1:
                    raise ValueError(
                        "a competition cluster spans multiple bootstrap "
                        "strata; choose cluster bootstrap without stratification"
                    )
                stratum_clusters[next(iter(labels))].append(
                    np.asarray(members, dtype=int)
                )
            sampled_means = np.empty(n_resamples, dtype=float)
            for draw in range(n_resamples):
                sampled: list[np.ndarray] = []
                for blocks in stratum_clusters.values():
                    picks = rng.integers(0, len(blocks), size=len(blocks))
                    sampled.extend(blocks[index] for index in picks)
                sampled_idx = np.concatenate(sampled)
                sampled_means[draw] = float(arr[sampled_idx].mean())
            means = sampled_means
        else:
            blocks = [
                np.asarray(members, dtype=int)
                for _cluster, members in cluster_items
            ]
            sampled_means = np.empty(n_resamples, dtype=float)
            for draw in range(n_resamples):
                picks = rng.integers(0, len(blocks), size=len(blocks))
                sampled_idx = np.concatenate([blocks[index] for index in picks])
                sampled_means[draw] = float(arr[sampled_idx].mean())
            means = sampled_means
    elif strata is None:
        idx = rng.integers(0, n, size=(n_resamples, n))
        means = arr[idx].mean(axis=1)
    else:
        if len(strata) != n:
            raise ValueError(
                f"strata length {len(strata)} does not match values length {n}"
            )
        stratum_to_idx: dict[Any, list[int]] = defaultdict(list)
        for index, stratum in enumerate(strata):
            stratum_to_idx[stratum].append(index)
        sums = np.zeros(n_resamples)
        for members in stratum_to_idx.values():
            size = len(members)
            resampled = rng.integers(
                0, size, size=(n_resamples, size)
            )
            sums += arr[np.asarray(members)][resampled].sum(axis=1)
        means = sums / n

    alpha = (1 - ci) / 2
    return (
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1 - alpha)),
    )


def bootstrap_ratio_of_sums_difference_ci(
    candidate_pnl: Sequence[Sequence[float]],
    candidate_stake: Sequence[Sequence[float]],
    baseline_pnl: Sequence[Sequence[float]],
    baseline_stake: Sequence[Sequence[float]],
    clusters: Sequence[Any],
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    ci: float = 0.95,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Tournament-block CI for candidate ROI minus baseline ROI.

    Each replicate recomputes each arm's ROI as a ratio of sums. With multiple
    fitted seeds, paired seed indices are resampled before tournament blocks,
    matching the seed-mean gate estimator. A zero-stake arm has ROI zero.
    """
    arrays = [np.atleast_2d(np.asarray(values, dtype=float)) for values in (
        candidate_pnl, candidate_stake, baseline_pnl, baseline_stake
    )]
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays[1:]):
        raise ValueError("ROI pnl/stake arrays are not aligned")
    if shape[1] != len(clusters):
        raise ValueError("ROI rows and clusters are not aligned")
    if shape[1] == 0 or n_resamples <= 0:
        return (float("nan"), float("nan"))

    _, inverse = np.unique(np.asarray(clusters), return_inverse=True)
    n_clusters = int(inverse.max()) + 1
    block_sums = [np.stack([
        np.bincount(inverse, weights=row, minlength=n_clusters)
        for row in array
    ]) for array in arrays]
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_resamples, dtype=float)
    n_seeds = shape[0]
    for draw in range(n_resamples):
        sampled_seeds = rng.integers(0, n_seeds, size=n_seeds)
        sampled_blocks = rng.integers(0, n_clusters, size=n_clusters)
        totals = [values[sampled_seeds][:, sampled_blocks].sum()
                  for values in block_sums]
        candidate_roi = totals[0] / totals[1] if totals[1] else 0.0
        baseline_roi = totals[2] / totals[3] if totals[3] else 0.0
        estimates[draw] = candidate_roi - baseline_roi
    alpha = (1 - ci) / 2
    return (float(np.quantile(estimates, alpha)),
            float(np.quantile(estimates, 1 - alpha)))


def count_unique_clusters(clusters: Iterable[Any]) -> int:
    return len(set(clusters))


# ---------------------------------------------------------------------------
# Paired prop/sim helpers
# ---------------------------------------------------------------------------


def load(path: Path | str) -> Any:
    """Load a JSON artifact (legacy public name used by auto gate scripts)."""
    with Path(path).open() as handle:
        return json.load(handle)


def paired_rows(det_a: Sequence[Mapping[str, Any]],
                det_b: Sequence[Mapping[str, Any]], fam: str) -> list[tuple]:
    """Return positionally paired ``(match_id, y, p_a, p_b)`` rows."""
    idx_b = {record["match_id"]: record["obs"] for record in det_b}
    out = []
    for record_a in det_a:
        match_id = record_a["match_id"]
        obs_a = record_a["obs"].get(fam)
        obs_b = idx_b.get(match_id, {}).get(fam)
        if not obs_a or not obs_b or len(obs_a) != len(obs_b):
            continue
        for value_a, value_b in zip(obs_a, obs_b):
            if value_a.get("y") != value_b.get("y"):
                continue
            out.append((match_id, float(value_a["y"]),
                        float(value_a["p"]), float(value_b["p"])))
    return out


def metric_rows(detail: Sequence[Mapping[str, Any]], fam: str) -> list[tuple]:
    """Return ``(match_id, y, p)`` rows for a detail metric family."""
    out = []
    for record in detail:
        for value in record["obs"].get(fam, []):
            out.append((record["match_id"], float(value["y"]),
                        float(value["p"])))
    return out


def cluster_boot(rows: Sequence[Sequence[Any]], fn, n_boot: int = 2000,
                 seed: int = 29) -> tuple[float, float]:
    """Legacy A8 whole-match bootstrap, retained with identical defaults."""
    rng = np.random.default_rng(seed)
    by: dict[Any, list] = defaultdict(list)
    for row in rows:
        by[row[0]].append(row)
    match_ids = list(by)
    values = []
    for _ in range(n_boot):
        sampled = rng.choice(len(match_ids), size=len(match_ids), replace=True)
        accumulated = []
        for index in sampled:
            accumulated.extend(fn(row) for row in by[match_ids[index]])
        values.append(np.mean(accumulated))
    return (float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)))


def brier_pair(rows: Sequence[Sequence[Any]]) -> tuple:
    """Return Brier A/B, paired delta B-A, and its match-bootstrap CI."""
    brier_a = float(np.mean([(p_a - y) ** 2 for _, y, p_a, _ in rows]))
    brier_b = float(np.mean([(p_b - y) ** 2 for _, y, _, p_b in rows]))
    interval = cluster_boot(
        rows,
        lambda row: ((row[3] - row[1]) ** 2 - (row[2] - row[1]) ** 2),
    )
    return brier_a, brier_b, brier_b - brier_a, interval


def mae_pair(det_a: Sequence[Mapping[str, Any]],
             det_b: Sequence[Mapping[str, Any]], fam: str) -> tuple | None:
    """Return paired MAE A/B, delta B-A, CI, and row count."""
    rows_a = {record["match_id"]: record["obs"].get(fam) for record in det_a}
    rows_b = {record["match_id"]: record["obs"].get(fam) for record in det_b}
    rows = []
    for match_id in rows_a:
        obs_a, obs_b = rows_a.get(match_id), rows_b.get(match_id)
        if not obs_a or not obs_b or len(obs_a) != len(obs_b):
            continue
        for value_a, value_b in zip(obs_a, obs_b):
            if value_a.get("actual") != value_b.get("actual"):
                continue
            rows.append((match_id, float(value_a["actual"]),
                         float(value_a["sim_mean"]),
                         float(value_b["sim_mean"])))
    if not rows:
        return None
    mae_a = float(np.mean([abs(p_a - y) for _, y, p_a, _ in rows]))
    mae_b = float(np.mean([abs(p_b - y) for _, y, _, p_b in rows]))
    interval = cluster_boot(
        rows, lambda row: abs(row[3] - row[1]) - abs(row[2] - row[1])
    )
    return mae_a, mae_b, mae_b - mae_a, interval, len(rows)
