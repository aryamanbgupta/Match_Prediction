from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from daily import fetch_fixtures
from daily.fetch_fixtures import append_fixtures, is_h2h_market, records_from_gamma
from daily.fetch_fixtures import verify_fixture_line


FIXTURE = Path(__file__).parent / "fixtures" / "gamma_recorded" / "open_t20.json"


@pytest.fixture(autouse=True)
def writer_key(tmp_path, monkeypatch):
    path = tmp_path / ".writer_key"
    monkeypatch.setenv("DAILY_WRITER_KEY", str(path))
    return path


def test_recorded_gamma_selects_moneyline_and_rejects_toss():
    payload = json.loads(FIXTURE.read_text())
    event = payload["events"][0]
    assert is_h2h_market(event, event["markets"][0]) is True
    assert is_h2h_market(event, event["markets"][1]) is False
    rows = records_from_gamma(payload["events"])
    assert len(rows) == 1
    assert rows[0]["fixture_id"] == "m-win"
    assert rows[0]["quote"] == {"Alpha CC": 0.58, "Beta CC": 0.42}
    assert "quote_ts" not in rows[0]


def test_moneyline_identity_conflict_fails_closed():
    event = {"title": "Winner", "markets": []}
    assert is_h2h_market(event, {"question": "Who wins the toss?",
                                 "sportsMarketType": "moneyline"}) is False


def test_append_owns_quote_timestamp_and_null_quote_has_null_timestamp(
    tmp_path, monkeypatch, writer_key
):
    stamp = datetime(2026, 9, 10, 10, tzinfo=timezone.utc)
    monkeypatch.setattr(fetch_fixtures, "utc_now", lambda: stamp)
    output = tmp_path / "fixtures.jsonl"
    rows = [
        {"fixture_id": "quoted", "market_id": "quoted",
         "scheduled_start": "2026-09-10T11:00:00+00:00",
         "market_volume_usd": 10.0, "quote": {"A": 0.5, "B": 0.5}},
        {"fixture_id": "missing", "market_id": "missing",
         "scheduled_start": "2026-09-10T11:00:00+00:00",
         "market_volume_usd": None, "quote": None},
    ]
    assert append_fixtures(rows, output) == 2
    written = [json.loads(line) for line in output.read_text().splitlines()]
    assert written[0]["quote_ts"] == stamp.isoformat()
    assert written[1]["quote_ts"] is None
    assert written[0]["writer"] == written[1]["writer"] == "fetch_fixtures"
    verify_fixture_line(written[0])
    verify_fixture_line(written[1])
    tampered = {**written[0], "quote_ts": "2026-09-10T09:00:00+00:00"}
    assert written[0]["line_hmac_sha256"]
    assert writer_key.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="line_hmac_sha256"):
        verify_fixture_line(tampered)
    with pytest.raises(ValueError, match="writer-owned"):
        append_fixtures([{**rows[0], "quote_ts": stamp.isoformat()}], output)


def test_writer_key_is_created_once_and_genuine_line_verifies(tmp_path, writer_key):
    output = tmp_path / "fixtures.jsonl"
    row = {
        "fixture_id": "m", "market_id": "market", "quote": None,
        "market_volume_usd": None,
        "scheduled_start": "2026-09-11T11:00:00+00:00",
    }
    append_fixtures([row], output)
    original_key = writer_key.read_bytes()
    genuine = json.loads(output.read_text())
    verify_fixture_line(genuine)
    append_fixtures([{**row, "fixture_id": "m2", "market_id": "market2"}], output)
    assert writer_key.read_bytes() == original_key
