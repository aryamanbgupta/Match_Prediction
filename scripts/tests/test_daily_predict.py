from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest

from daily import predict_daily
from daily.fetch_fixtures import fixture_line_hmac_sha256


@pytest.fixture(autouse=True)
def writer_key(tmp_path, monkeypatch):
    path = tmp_path / ".writer_key"
    path.write_bytes(b"k" * 32)
    path.chmod(0o600)
    monkeypatch.setenv("DAILY_WRITER_KEY", str(path))
    return path


def _signed_fixture(**changes):
    row = {
        "fixture_id": "m", "team1": "A", "team2": "B",
        "scheduled_start": "2026-09-10T11:00:00+00:00", "quote": None,
        "quote_ts": None, "market_volume_usd": None, "market_id": "market-m",
        "writer": "fetch_fixtures", **changes,
    }
    row["line_hmac_sha256"] = fixture_line_hmac_sha256(row)
    return row


def _prediction_base(**changes):
    fixture = _signed_fixture()
    return {
        "cohort_id": "daily_v1", "fixture_id": fixture["fixture_id"],
        "market_id": fixture["market_id"], "run_kind": "t60",
        "scheduled_start": fixture["scheduled_start"], "quote_ts": fixture["quote_ts"],
        "quote": fixture["quote"], "market_volume_usd": fixture["market_volume_usd"],
        "fixture_writer": fixture["writer"],
        "fixture_line_hmac_sha256": fixture["line_hmac_sha256"],
        **changes,
    }


def test_append_stamps_identity_and_refuses_collision(tmp_path, monkeypatch):
    stamp = datetime(2026, 9, 10, 10, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(predict_daily, "utc_now", lambda: stamp)
    output = tmp_path / "predictions.jsonl"
    base = _prediction_base()
    row = predict_daily.append_prediction(output, base)
    assert row["attempt_ts"] == stamp.isoformat()
    assert row["minutes_to_start"] == 60
    with pytest.raises(RuntimeError, match="identity already exists"):
        predict_daily.append_prediction(output, base)
    with pytest.raises(ValueError, match="writer-owned"):
        predict_daily.append_prediction(output, {**base, "attempt_ts": stamp.isoformat()})
    assert len(output.read_text().splitlines()) == 1
    assert output.with_name(output.name + ".lock").is_file()


def test_concurrent_identical_appends_are_serialized(tmp_path, monkeypatch):
    stamp = datetime(2026, 9, 10, 10, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(predict_daily, "utc_now", lambda: stamp)
    output = tmp_path / "predictions.jsonl"
    base = _prediction_base()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(predict_daily.append_prediction, output, base)
                   for _ in range(2)]
    outcomes = []
    for future in futures:
        try:
            future.result()
            outcomes.append("written")
        except RuntimeError:
            outcomes.append("rejected")
    assert sorted(outcomes) == ["rejected", "written"]
    assert len(output.read_text().splitlines()) == 1


@pytest.mark.parametrize("stamp", [
    datetime(2026, 9, 10, 11, 0, tzinfo=timezone.utc),
    datetime(2026, 9, 10, 11, 1, tzinfo=timezone.utc),
])
def test_append_refuses_at_or_after_start(tmp_path, monkeypatch, stamp):
    monkeypatch.setattr(predict_daily, "utc_now", lambda: stamp)
    base = _prediction_base()
    output = tmp_path / "predictions.jsonl"
    with pytest.raises(RuntimeError, match="at or after"):
        predict_daily.append_prediction(output, base)
    assert output.read_text() == ""


def test_toss_requires_confirmed_xi_and_supplied_result():
    fixture = {"fixture_id": "m", "team1": "A", "team2": "B",
               "scheduled_start": "2026-09-10T11:00:00+00:00", "quote": None}
    prediction = {"prediction": {"A": 0.6, "B": 0.4}}
    with pytest.raises(ValueError, match="confirmed XI"):
        predict_daily.make_prediction_line(
            fixture, {"lineup_mode": "projected"}, prediction, run_kind="toss",
            model_role="match_model_prod", model_md5="a" * 32,
            state_as_of="2026-09-09", toss={"winner": "A", "decision": "bat"})


def test_prediction_line_refuses_unsigned_fixture_quote_timestamp():
    fixture = {"fixture_id": "m", "team1": "A", "team2": "B",
               "scheduled_start": "2026-09-10T11:00:00+00:00",
               "quote": {"A": .5, "B": .5},
               "quote_ts": "2026-09-10T10:00:00+00:00"}
    with pytest.raises(ValueError, match="not written"):
        predict_daily.make_prediction_line(
            fixture, {"lineup_mode": "projected"},
            {"prediction": {"A": .5, "B": .5}}, run_kind="t60",
            model_role="match_model_prod", model_md5="a" * 32,
            state_as_of="2026-09-09")


def test_clock_is_sampled_after_identity_scan_and_serialization(tmp_path, monkeypatch):
    output = tmp_path / "predictions.jsonl"
    output.write_text(json.dumps({"cohort_id": "old"}) + "\n")
    calls = {"dumps": 0}
    real_dumps = predict_daily.json.dumps
    def observed_dumps(*args, **kwargs):
        calls["dumps"] += 1
        return real_dumps(*args, **kwargs)
    monkeypatch.setattr(predict_daily.json, "dumps", observed_dumps)
    monkeypatch.setattr(predict_daily, "utc_now", lambda: (
        datetime(2026, 9, 10, 10, tzinfo=timezone.utc)
        if calls["dumps"] else pytest.fail("clock sampled before serialization")
    ))
    row = predict_daily.append_prediction(output, _prediction_base())
    assert row["attempt_ts"] == "2026-09-10T10:00:00+00:00"


def test_fixture_provenance_latest_capture_and_expired_skip(tmp_path, monkeypatch):
    expired = _signed_fixture(scheduled_start="2026-09-10T09:00:00+00:00")
    older = _signed_fixture(quote={"A": 0.4, "B": 0.6},
                            quote_ts="2026-09-10T09:55:00+00:00")
    latest = _signed_fixture(quote={"A": 0.6, "B": 0.4},
                             quote_ts="2026-09-10T10:00:00+00:00")
    # Give the expired fixture a distinct identity.
    expired["fixture_id"] = "expired"
    expired["line_hmac_sha256"] = fixture_line_hmac_sha256(expired)
    lineups = [
        {"fixture_id": "m", "lineup_mode": "projected", "team1_lineup": [],
         "team2_lineup": []},
        {"fixture_id": "expired", "lineup_mode": "projected", "team1_lineup": [],
         "team2_lineup": []},
    ]
    seen = []
    def fake_predict(fixture, **kwargs):
        seen.append(fixture["quote"])
        return {"prediction": {"A": .5, "B": .5}, "diagnostics": {
            "state_freshness": {"state_available_through": "2026-09-09"}}}
    monkeypatch.setattr(predict_daily, "predict_record", fake_predict)
    monkeypatch.setattr(predict_daily, "utc_now", lambda: datetime(
        2026, 9, 10, 10, tzinfo=timezone.utc))
    counts = predict_daily.run_predictions(
        [older, expired, latest], lineups, run_kind="t60", model_dir=tmp_path,
        state_dir=tmp_path, tracker_snapshot=tmp_path / "tracker",
        tracker_source_dirs=[], model_role="match_model_prod", model_md5="a" * 32,
        output=tmp_path / "predictions.jsonl",
        now=datetime(2026, 9, 10, 10, tzinfo=timezone.utc),
    )
    assert counts == {"added": 1, "expired_skipped": 1}
    assert seen == [{"A": 0.6, "B": 0.4}]


def test_unsigned_or_tampered_fixture_is_refused_before_inference(tmp_path, monkeypatch):
    monkeypatch.setattr(predict_daily, "predict_record",
                        lambda *args, **kwargs: pytest.fail("inference called"))
    with pytest.raises(ValueError, match="not written"):
        predict_daily.run_predictions(
            [{"fixture_id": "m"}], [], run_kind="t60", model_dir=tmp_path,
            state_dir=tmp_path, tracker_snapshot=tmp_path, tracker_source_dirs=[],
            model_role="r", model_md5="a" * 32, output=tmp_path / "out.jsonl")


@pytest.mark.parametrize("change", ["absent", "forged", "quote_ts"])
def test_fixture_hmac_is_mandatory_and_authenticates_quote_timestamp(
    tmp_path, monkeypatch, change
):
    fixture = _signed_fixture(
        quote={"A": 0.5, "B": 0.5},
        quote_ts="2026-09-10T10:00:00+00:00",
    )
    if change == "absent":
        fixture.pop("line_hmac_sha256")
    elif change == "forged":
        fixture["line_hmac_sha256"] = "0" * 64
    else:
        fixture["quote_ts"] = "2026-09-10T10:01:00+00:00"
    monkeypatch.setattr(predict_daily, "predict_record",
                        lambda *args, **kwargs: pytest.fail("inference called"))
    with pytest.raises(ValueError, match="line_hmac_sha256"):
        predict_daily.run_predictions(
            [fixture], [], run_kind="t60", model_dir=tmp_path,
            state_dir=tmp_path, tracker_snapshot=tmp_path, tracker_source_dirs=[],
            model_role="r", model_md5="a" * 32, output=tmp_path / "out.jsonl")


def test_fixture_with_forged_writer_is_refused(tmp_path, monkeypatch):
    fixture = _signed_fixture(writer="someone_else")
    monkeypatch.setattr(predict_daily, "predict_record",
                        lambda *args, **kwargs: pytest.fail("inference called"))
    with pytest.raises(ValueError, match="not written by fetch_fixtures"):
        predict_daily.run_predictions(
            [fixture], [], run_kind="t60", model_dir=tmp_path,
            state_dir=tmp_path, tracker_snapshot=tmp_path, tracker_source_dirs=[],
            model_role="r", model_md5="a" * 32, output=tmp_path / "out.jsonl")


def test_append_cannot_bypass_fixture_hmac_with_modified_quote_timestamp(tmp_path):
    with pytest.raises(ValueError, match="line_hmac_sha256"):
        predict_daily.append_prediction(
            tmp_path / "out.jsonl",
            _prediction_base(quote_ts="2026-09-10T10:01:00+00:00"),
        )
