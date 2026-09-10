#!/usr/bin/env python3
"""Append protocol-v1 daily predictions using the manifest production roles."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from artifacts import artifact_path, load_manifest  # noqa: E402
from predict_fixture import predict_record  # noqa: E402
from daily.fetch_fixtures import HMAC_FIELDS, load_writer_key, verify_fixture_line  # noqa: E402

COHORT_ID = "daily_v1"
PROTOCOL_VERSION = 1


class ExpiredFixtureError(RuntimeError):
    """The writer clock reached scheduled start before append."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("daily timestamps must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def model_md5_from_manifest(role: str, manifest: Path) -> str:
    row = load_manifest(manifest)[role]
    explicit = row.get("model_md5")
    if explicit:
        return str(explicit)
    match = re.search(r"model\.pkl md5 ([0-9a-f]{32})", str(row.get("notes", "")))
    if not match:
        raise RuntimeError(f"manifest role {role!r} does not stamp model.pkl md5")
    return match.group(1)


def make_prediction_line(
    fixture: dict,
    lineup: dict,
    prediction: dict,
    *,
    run_kind: str,
    model_role: str,
    model_md5: str,
    state_as_of: str,
    toss: dict | None = None,
) -> dict:
    """Create a line; the writer timestamp intentionally has no parameter."""
    if run_kind not in {"t60", "toss"}:
        raise ValueError("run_kind must be t60 or toss")
    if run_kind == "toss" and (lineup.get("lineup_mode") != "confirmed" or not toss):
        raise ValueError("toss runs require a supplied toss result and confirmed XI")
    if run_kind == "t60" and toss:
        raise ValueError("t60 is toss-blind")
    verify_fixture_line(fixture)
    start = _parse_ts(fixture["scheduled_start"])
    quote = fixture.get("quote")
    quote_ts = fixture.get("quote_ts") if quote is not None else None
    return {
        "protocol_version": PROTOCOL_VERSION,
        "cohort_id": COHORT_ID,
        "fixture_id": fixture["fixture_id"],
        "market_id": fixture.get("market_id", fixture["fixture_id"]),
        "cricsheet_id": fixture.get("cricsheet_id"),
        "run_kind": run_kind,
        "fixture_writer": fixture["writer"],
        "fixture_line_hmac_sha256": fixture["line_hmac_sha256"],
        "quote_ts": quote_ts,
        "scheduled_start": start.isoformat(),
        "team1": fixture["team1"],
        "team2": fixture["team2"],
        "venue": fixture.get("venue"),
        "model_role": model_role,
        "model_md5": model_md5,
        "state_as_of": state_as_of,
        "lineup_mode": lineup["lineup_mode"],
        "toss_mode": "supplied" if toss else "unknown",
        "toss": toss,
        "p_team1": float(prediction["prediction"][fixture["team1"]]),
        "quote": quote,
        "market_volume_usd": fixture.get("market_volume_usd"),
    }


def append_prediction(output: Path, line_without_timestamp: dict) -> dict:
    """Lock, stamp, validate, append and durably flush one prediction."""
    if "attempt_ts" in line_without_timestamp:
        raise ValueError("attempt_ts is writer-owned and may not be supplied")
    row = dict(line_without_timestamp)
    try:
        fixture = {field: row[field] for field in HMAC_FIELDS}
        fixture["writer"] = row["fixture_writer"]
        fixture["line_hmac_sha256"] = row["fixture_line_hmac_sha256"]
    except KeyError as exc:
        raise ValueError("prediction lacks verified fixture provenance") from exc
    verify_fixture_line(fixture, load_writer_key())
    if row.get("quote") is None and row.get("quote_ts") is not None:
        raise ValueError("quote_ts must be null when quote is null")
    start = _parse_ts(row["scheduled_start"])
    output.parent.mkdir(parents=True, exist_ok=True)
    # Build an initial index before taking the exclusive lock. The complete
    # file is read again under lock below, which is the authoritative check.
    identities = {
        tuple(existing.get(k) for k in (
            "cohort_id", "fixture_id", "run_kind", "attempt_ts"
        ))
        for existing in _jsonl(output)
    }
    # Force all caller-controlled content through serialization before the
    # final writer clock is sampled. Unique JSON string placeholders let the
    # final append avoid another potentially slow object serialization.
    attempt_placeholder = "__CRICML_ATTEMPT_TS_PLACEHOLDER__"
    minutes_placeholder = "__CRICML_MINUTES_TO_START_PLACEHOLDER__"
    template = {**row, "attempt_ts": attempt_placeholder,
                "minutes_to_start": minutes_placeholder}
    serialized = json.dumps(template, sort_keys=True)
    lock_path = output.with_name(output.name + ".lock")
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        with output.open("a+", encoding="utf-8") as handle:
            handle.seek(0)
            identities.update({
                tuple(existing.get(k) for k in (
                    "cohort_id", "fixture_id", "run_kind", "attempt_ts"
                ))
                for line in handle.read().splitlines() if line.strip()
                for existing in (json.loads(line),)
            })
            attempt = utc_now()
            if attempt >= start:
                raise ExpiredFixtureError(
                    "prediction append refused at or after scheduled_start")
            row["attempt_ts"] = attempt.isoformat()
            row["minutes_to_start"] = (start - attempt).total_seconds() / 60.0
            identity = tuple(
                row[k] for k in ("cohort_id", "fixture_id", "run_kind", "attempt_ts")
            )
            if identity in identities:
                raise RuntimeError(f"prediction identity already exists: {identity}")
            encoded = serialized.replace(
                f'"attempt_ts": {json.dumps(attempt_placeholder)}',
                f'"attempt_ts": "{row["attempt_ts"]}"', 1,
            ).replace(
                f'"minutes_to_start": {json.dumps(minutes_placeholder)}',
                f'"minutes_to_start": {row["minutes_to_start"]!r}', 1,
            )
            handle.seek(0, os.SEEK_END)
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return row


def run_predictions(fixtures: list[dict], lineups: list[dict], *, run_kind: str,
                    model_dir: Path, state_dir: Path, tracker_snapshot: Path,
                    tracker_source_dirs: list[Path], model_role: str,
                    model_md5: str, output: Path, tosses: dict | None = None,
                    now: datetime | None = None) -> dict[str, int]:
    lineup_by_id = {row["fixture_id"]: row for row in lineups}
    writer_key = load_writer_key()
    for fixture in fixtures:
        verify_fixture_line(fixture, writer_key)
    latest: dict[str, dict] = {}
    for fixture in fixtures:
        current = latest.get(fixture["fixture_id"])
        key = (str(fixture.get("quote_ts") or ""), fixture["line_hmac_sha256"])
        if current is None or key > (str(current.get("quote_ts") or ""),
                                     current["line_hmac_sha256"]):
            latest[fixture["fixture_id"]] = fixture
    cutoff = (now or utc_now()).astimezone(timezone.utc)
    added = 0
    expired_skipped = 0
    for fixture in [latest[key] for key in sorted(latest)]:
        if _parse_ts(fixture["scheduled_start"]) <= cutoff:
            expired_skipped += 1
            continue
        lineup = lineup_by_id[fixture["fixture_id"]]
        toss = (tosses or {}).get(fixture["fixture_id"])
        serving_fixture = {
            **fixture,
            "date": fixture["scheduled_start"][:10],
            "venue": fixture.get("venue") or "Unknown venue",
            "team1_lineup": lineup["team1_lineup"],
            "team2_lineup": lineup["team2_lineup"],
            "polymarket_odds": ({team: 1.0 / price for team, price in fixture["quote"].items()}
                                 if fixture.get("quote") else None),
            "polymarket_volume_usd": fixture.get("market_volume_usd"),
        }
        if toss:
            serving_fixture["toss_winner"] = toss["winner"]
            serving_fixture["toss_decision"] = toss["decision"]
        else:
            serving_fixture.pop("toss_winner", None)
            serving_fixture.pop("toss_decision", None)
        predicted = predict_record(
            serving_fixture, model_dir=model_dir, state_dir=state_dir,
            tracker_snapshot=tracker_snapshot,
            tracker_source_dirs=tracker_source_dirs,
        )
        state_as_of = predicted["diagnostics"]["state_freshness"]["state_available_through"]
        made = make_prediction_line(
            fixture, lineup, predicted, run_kind=run_kind, model_role=model_role,
            model_md5=model_md5, state_as_of=state_as_of, toss=toss,
        )
        try:
            append_prediction(output, made)
            added += 1
        except ExpiredFixtureError:
            expired_skipped += 1
    return {"added": added, "expired_skipped": expired_skipped}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--lineups", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--run-kind", choices=("t60", "toss"), default="t60")
    p.add_argument("--tosses", type=Path)
    p.add_argument("--manifest", type=Path, default=REPO / "models" / "MANIFEST.yaml")
    p.add_argument("--model-role", default="match_model_prod")
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--state-dir", type=Path)
    p.add_argument("--tracker-snapshot", type=Path)
    p.add_argument("--tracker-source-dir", type=Path, action="append", required=True)
    a = p.parse_args(argv)
    model_dir = a.model_dir or REPO / artifact_path(a.model_role, manifest_path=a.manifest)
    state_dir = a.state_dir or REPO / artifact_path("live_state_i7", manifest_path=a.manifest)
    tracker = a.tracker_snapshot or state_dir / "tracker_snapshot.pkl"
    tosses = json.loads(a.tosses.read_text()) if a.tosses else None
    counts = run_predictions(
        _jsonl(a.fixtures), _jsonl(a.lineups), run_kind=a.run_kind,
        model_dir=model_dir, state_dir=state_dir, tracker_snapshot=tracker,
        tracker_source_dirs=a.tracker_source_dir, model_role=a.model_role,
        model_md5=model_md5_from_manifest(a.model_role, a.manifest),
        output=a.out, tosses=tosses,
    )
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
