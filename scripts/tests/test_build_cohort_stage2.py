"""Unit tests for the stage 2 cohort builder (acceptance D5.1, D5.2, D5.7).

Synthetic fixtures only: no cricsheet corpus, no stats cache, no parquet.
The closed-file helpers are exercised against synthetic files carrying a
sentinel value in every non-id field; the sentinel must never appear in
anything the helper returns.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from sequence_track import build_cohort_stage2 as bc

SENTINEL = "SENTINEL-MUST-NOT-LEAK-3f9a1c"


# --------------------------------------------------------------------------
# 5.1 zip filter + dedupe
# --------------------------------------------------------------------------

def _info(**over):
    info = {
        "gender": "male",
        "match_type": "T20",
        "balls_per_over": 6,
        "dates": ["2026-05-01"],
    }
    info.update(over)
    return info


@pytest.mark.parametrize(
    "info,expected",
    [
        (_info(), True),
        (_info(gender="female"), False),
        (_info(gender=None), False),
        (_info(match_type="ODI"), False),
        (_info(match_type="T20I"), False),
        (_info(balls_per_over=5), False),
        # absent balls_per_over defaults to 6
        ({"gender": "male", "match_type": "T20", "dates": ["2026-05-01"]},
         True),
        (_info(dates=["2026-04-16"]), False),   # day before the window
        (_info(dates=["2026-04-17"]), True),    # inclusive start
        (_info(dates=["2026-08-05"]), True),    # inclusive end
        (_info(dates=["2026-08-06"]), False),   # day after the window
        (_info(dates=[]), False),
        # a multi-day listing is judged on its FIRST date
        (_info(dates=["2026-04-16", "2026-04-18"]), False),
    ],
)
def test_keep_match_filter(info, expected):
    assert bc.keep_match(info) is expected


def _write_zip(path: Path, members: dict) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


def _match_json(date="2026-05-01", extra_field=None, **info_over):
    data = {"info": _info(dates=[date], **info_over),
            "innings": []}
    if extra_field is not None:
        data["info"]["event"] = {"name": extra_field}
    return json.dumps(data, sort_keys=True)


def test_extract_dedupe_and_refusal(tmp_path, monkeypatch):
    zip_a, zip_b = tmp_path / "a_json.zip", tmp_path / "b_json.zip"
    identical = _match_json(extra_field="Shared Cup")
    _write_zip(zip_a, {
        "111.json": identical,
        "222.json": _match_json(date="2026-08-05"),
        "333.json": _match_json(date="2026-08-06"),          # out of window
        "444.json": _match_json(balls_per_over=5),           # the Hundred
        "odis_skip.txt": "not json",
    })
    _write_zip(zip_b, {"111.json": identical})               # byte-identical

    monkeypatch.setattr(bc, "ZIP_DIR", tmp_path)
    monkeypatch.setattr(bc, "RAW_JSON_DIR", tmp_path / "raw_json")
    monkeypatch.setattr(bc, "RAW_MANIFEST", tmp_path / "raw_manifest.json")
    # the frozen-cohort guard would otherwise refuse the step (SHOULD 2)
    monkeypatch.setattr(bc, "FROZEN_JSON", tmp_path / "FROZEN.json")

    manifest = bc.step_extract(strict=False)
    assert sorted(manifest["matches"]) == ["111", "222"]
    assert len(manifest["byte_identical_duplicates"]) == 1
    assert manifest["byte_identical_duplicates"][0]["stem"] == "111"
    assert sorted(
        p.name for p in (tmp_path / "raw_json").glob("*.json")
    ) == ["111.json", "222.json"]

    # A differing duplicate refuses loudly.
    _write_zip(zip_b, {"111.json": _match_json(extra_field="Other Cup")})
    with pytest.raises(RuntimeError, match="differing duplicate"):
        bc.step_extract(strict=False)


def test_extract_refuses_on_count_mismatch(tmp_path, monkeypatch):
    _write_zip(tmp_path / "a_json.zip", {"111.json": _match_json()})
    monkeypatch.setattr(bc, "ZIP_DIR", tmp_path)
    monkeypatch.setattr(bc, "RAW_JSON_DIR", tmp_path / "raw_json")
    monkeypatch.setattr(bc, "RAW_MANIFEST", tmp_path / "raw_manifest.json")
    # the frozen-cohort guard would otherwise refuse the step (SHOULD 2)
    monkeypatch.setattr(bc, "FROZEN_JSON", tmp_path / "FROZEN.json")
    with pytest.raises(RuntimeError, match="expected 471 matches, got 1"):
        bc.step_extract(strict=True)


def test_skip_zips_are_the_non_t20_archives():
    assert bc.SKIP_ZIPS == {"odis_json.zip", "tests_json.zip"}


# --------------------------------------------------------------------------
# 5.2 id-only helpers
# --------------------------------------------------------------------------

def _closed_file(tmp_path: Path, name: str) -> Path:
    """A synthetic closed file whose every non-id field is the sentinel."""
    payload = {
        "source": SENTINEL,
        "total_matches": 2,
        "filters": {"min_volume_usd": SENTINEL},
        "matches": [
            {
                "match_id": "1500001",
                "cricsheet_id": "1500001",
                "date": SENTINEL,
                "teams": [SENTINEL, SENTINEL],
                "venue": SENTINEL,
                "volume_usd": SENTINEL,
                "actual_winner": SENTINEL,
                "odds": {"prob_team1": SENTINEL},
            },
            {
                "match_id": "1500002",
                "cricsheet_id": "legacy-1500002",
                "date": SENTINEL,
                "polymarket_market_id": SENTINEL,
            },
        ],
    }
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


@pytest.mark.parametrize(
    "reader", [bc.read_forward_fixture_ids, bc.read_golden_fixture_ids]
)
def test_id_only_readers_return_only_ids(tmp_path, reader):
    path = _closed_file(tmp_path, "closed.json")
    ids = reader(path)
    assert ids == ["1500001", "1500002", "legacy-1500002"]
    assert ids == sorted(ids)
    blob = json.dumps(ids)
    assert SENTINEL not in blob
    for value in ids:
        assert SENTINEL not in value


def test_id_only_readers_tolerate_missing_fields(tmp_path):
    path = tmp_path / "sparse.json"
    path.write_text(json.dumps({"matches": [{}, "junk", {"match_id": "9"}]}))
    assert bc.read_forward_fixture_ids(path) == ["9"]


def test_closed_path_constants_are_used_only_by_the_permitted_readers():
    """D5.8 — nothing but the two id helpers touches the closed trees.

    ``step_freeze`` is allowed to name the two paths because it records the
    permitted reads in FROZEN.json; it never opens them (it calls the
    helpers). Any other function referencing them is a violation.
    """
    import ast

    tree = ast.parse(Path(bc.__file__).read_text())
    allowed = {
        "read_forward_fixture_ids",
        "read_golden_fixture_ids",
        "step_freeze",
    }
    offenders = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        names = {
            n.id for n in ast.walk(node) if isinstance(n, ast.Name)
        }
        used = names & {"FORWARD_MANIFEST", "GOLDEN_ODDS"}
        if used and node.name not in allowed:
            offenders[node.name] = sorted(used)
    assert not offenders, offenders
    # and the only function that opens a file by that path is the shared
    # id-extractor, which returns ids only (tested above).
    openers = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(n, ast.Attribute) and n.attr == "load"
            and isinstance(n.value, ast.Name) and n.value.id == "json"
            for n in ast.walk(node)
        )
    ]
    assert "_match_ids_only" in openers


# --------------------------------------------------------------------------
# 5.2 reason precedence
# --------------------------------------------------------------------------

def test_reason_precedence():
    forward, golden, repo = {"f"}, {"g"}, {"r"}
    call = lambda stem, bpo=6: bc.classify(
        stem, bpo, forward, golden, repo
    )
    assert call("x") == "eligible"
    assert call("f") == "forward_fixture"
    assert call("g") == "golden_fixture"
    assert call("r") == "in_repo_corpus"
    # precedence: hundred > in_repo_corpus > forward_fixture > golden_fixture
    assert call("f", bpo=5) == "hundred"
    assert bc.classify("f", 6, forward, golden, {"f"}) == "in_repo_corpus"
    assert bc.classify("f", 6, forward, {"f"}, repo) == "forward_fixture"
    assert bc.REASON_ORDER == (
        "hundred", "in_repo_corpus", "forward_fixture", "golden_fixture"
    )


def test_audit_writes_one_row_per_match(tmp_path, monkeypatch):
    manifest = {
        "matches": {
            "1": {"date": "2026-05-01", "event": "A"},
            "2": {"date": "2026-06-05", "event": "B"},
            "3": {"date": "2026-07-01", "event": "C"},
        }
    }
    (tmp_path / "raw_manifest.json").write_text(json.dumps(manifest))
    repo = tmp_path / "t20s_json"
    repo.mkdir()
    (repo / "3.json").write_text("{}")
    monkeypatch.setattr(bc, "RAW_MANIFEST", tmp_path / "raw_manifest.json")
    monkeypatch.setattr(bc, "AUDIT_CSV", tmp_path / "audit.csv")
    monkeypatch.setattr(bc, "AUDIT_SUMMARY_JSON", tmp_path / "summary.json")
    monkeypatch.setattr(bc, "FROZEN_JSON", tmp_path / "FROZEN.json")
    monkeypatch.setattr(bc, "REPO_CORPUS", repo)
    monkeypatch.setattr(bc, "read_forward_fixture_ids", lambda *a: ["2"])
    monkeypatch.setattr(bc, "read_golden_fixture_ids", lambda *a: [])
    monkeypatch.setattr(bc, "write_consumer_notes", lambda summary: None)

    summary = bc.step_audit()
    assert summary["total"] == 3
    assert summary["counts_by_reason"] == {
        "eligible": 1, "forward_fixture": 1, "in_repo_corpus": 1
    }
    assert bc.eligible_stems() == ["1"]
    assert bc.stems_by_reason("forward_fixture") == ["2"]
    # the two permitted reads are recorded so no later step re-opens them
    recorded = json.loads((tmp_path / "summary.json").read_text())
    assert recorded["forward_ids_read"] == 1
    assert recorded["golden_ids_read"] == 0
    assert bc.recorded_id_read_counts() == {
        "forward": 1, "golden": 0, "source": str(tmp_path / "summary.json")}


# --------------------------------------------------------------------------
# 5.7 FROZEN schema
# --------------------------------------------------------------------------

FROZEN_REQUIRED_KEYS = {
    "frozen_at_utc",
    "eligible_match_ids",
    "eligible_match_count",
    "row_count",
    "parquet",
    "parquet_sha256",
    "feature_hash",
    "sidecar_cache",
    "sidecar_cache_md5",
    "raw_manifest_sha256",
    "builder_script_sha256",
    "permitted_id_only_reads",
}


def test_frozen_schema(tmp_path, monkeypatch):
    pd = pytest.importorskip("pandas")
    parquet = tmp_path / "cohort.parquet"
    pd.DataFrame({"innings_id": ["1_1", "1_1", "2_2"]}).to_parquet(
        parquet, index=False
    )
    (tmp_path / "audit.csv").write_text(
        "stem,date,event,eligible,reason\n"
        "1,2026-05-01,A,True,eligible\n"
        "2,2026-05-02,B,True,eligible\n"
        "3,2026-05-03,C,False,forward_fixture\n"
    )
    (tmp_path / ".feature_hash").write_text(
        json.dumps({"hash": "c520a3ba08ae", "n_features": 114})
    )
    (tmp_path / "raw_manifest.json").write_text("{}")
    (tmp_path / "cache.sqlite").write_bytes(b"sqlite-bytes")
    frozen_path = tmp_path / "FROZEN.json"

    monkeypatch.setattr(bc, "COHORT_PARQUET", parquet)
    monkeypatch.setattr(bc, "AUDIT_CSV", tmp_path / "audit.csv")
    monkeypatch.setattr(bc, "COHORT_FEATURE_HASH", tmp_path / ".feature_hash")
    monkeypatch.setattr(bc, "RAW_MANIFEST", tmp_path / "raw_manifest.json")
    monkeypatch.setattr(bc, "SIDECAR_SQLITE", tmp_path / "cache.sqlite")
    monkeypatch.setattr(bc, "FROZEN_JSON", frozen_path)
    (tmp_path / "summary.json").write_text(
        json.dumps({"forward_ids_read": 1, "golden_ids_read": 1})
    )
    monkeypatch.setattr(bc, "AUDIT_SUMMARY_JSON", tmp_path / "summary.json")
    # SHOULD 2: the freeze must not re-open either closed file. Both readers
    # are replaced with sentinels that fail if anything calls them.
    def _refuse(*_args, **_kwargs):
        raise AssertionError("step_freeze re-opened a closed file")

    monkeypatch.setattr(bc, "read_forward_fixture_ids", _refuse)
    monkeypatch.setattr(bc, "read_golden_fixture_ids", _refuse)

    frozen = bc.step_freeze()
    assert FROZEN_REQUIRED_KEYS <= set(frozen)
    assert frozen["eligible_match_ids"] == ["1", "2"]
    assert frozen["eligible_match_count"] == 2
    assert frozen["row_count"] == 3
    assert frozen["feature_hash"]["hash"] == "c520a3ba08ae"
    permission = frozen["permitted_id_only_reads"]["permission"]
    assert permission["quote"] == "you can read the files"
    assert permission["date"] == "2026-09-11"
    reads = frozen["permitted_id_only_reads"]["reads"]
    assert [r["helper"] for r in reads] == [
        "read_forward_fixture_ids", "read_golden_fixture_ids"
    ]
    assert all(
        r["fields_read"] == ["matches[].match_id", "matches[].cricsheet_id"]
        for r in reads
    )
    assert [r["ids_returned"] for r in reads] == [1, 1]
    assert frozen["permitted_id_only_reads"]["reads_recorded_from"] == str(
        tmp_path / "summary.json")
    assert json.loads(frozen_path.read_text())["row_count"] == 3
    assert frozen["frozen_at_utc"].endswith("Z")


# --------------------------------------------------------------------------
# SHOULD 2 — a frozen cohort is not rebuilt, and exclusions are consumed
# --------------------------------------------------------------------------

@pytest.mark.parametrize("step", bc.MUTATING_STEPS)
def test_every_mutating_step_refuses_once_the_cohort_is_frozen(
        tmp_path, monkeypatch, step):
    frozen_path = tmp_path / "FROZEN.json"
    frozen_path.write_text(json.dumps({"eligible_match_count": 210}))
    monkeypatch.setattr(bc, "FROZEN_JSON", frozen_path)
    monkeypatch.setattr(bc, "ALLOW_REFREEZE", False)
    with pytest.raises(RuntimeError) as excinfo:
        bc._guard_frozen(step)
    message = str(excinfo.value)
    assert step in message and "--allow-refreeze" in message


@pytest.mark.parametrize("step", bc.MUTATING_STEPS)
def test_allow_refreeze_lifts_the_refusal(tmp_path, monkeypatch, step):
    frozen_path = tmp_path / "FROZEN.json"
    frozen_path.write_text("{}")
    monkeypatch.setattr(bc, "FROZEN_JSON", frozen_path)
    monkeypatch.setattr(bc, "ALLOW_REFREEZE", True)
    bc._guard_frozen(step)  # does not raise


@pytest.mark.parametrize("step", bc.MUTATING_STEPS)
def test_an_unfrozen_cohort_is_never_refused(tmp_path, monkeypatch, step):
    monkeypatch.setattr(bc, "FROZEN_JSON", tmp_path / "absent.json")
    monkeypatch.setattr(bc, "ALLOW_REFREEZE", False)
    bc._guard_frozen(step)


def test_the_guard_covers_every_step_the_builder_runs():
    assert set(bc.MUTATING_STEPS) == set(bc.STEPS)


def test_recorded_counts_fall_back_to_the_freeze(tmp_path, monkeypatch):
    """With no audit summary, the counts come from FROZEN.json, never a read."""
    frozen_path = tmp_path / "FROZEN.json"
    frozen_path.write_text(json.dumps({
        "permitted_id_only_reads": {"reads": [
            {"helper": "read_forward_fixture_ids", "ids_returned": 137},
            {"helper": "read_golden_fixture_ids", "ids_returned": 124},
        ]}
    }))
    monkeypatch.setattr(bc, "AUDIT_SUMMARY_JSON", tmp_path / "absent.json")
    monkeypatch.setattr(bc, "FROZEN_JSON", frozen_path)
    counts = bc.recorded_id_read_counts()
    assert (counts["forward"], counts["golden"]) == (137, 124)
    assert counts["source"] == str(frozen_path)


def test_recorded_counts_refuse_to_guess(tmp_path, monkeypatch):
    monkeypatch.setattr(bc, "AUDIT_SUMMARY_JSON", tmp_path / "absent.json")
    monkeypatch.setattr(bc, "FROZEN_JSON", tmp_path / "also_absent.json")
    with pytest.raises(RuntimeError, match="--step audit"):
        bc.recorded_id_read_counts()


def test_only_the_audit_step_opens_a_closed_file():
    """The closed-file helpers are called exactly twice, both in step_audit."""
    import ast

    source = Path(bc.__file__).read_text()
    tree = ast.parse(source)
    callers = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if (isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Name)
                    and inner.func.id in ("read_forward_fixture_ids",
                                          "read_golden_fixture_ids")):
                callers.setdefault(node.name, []).append(inner.func.id)
    assert callers == {"step_audit": ["read_forward_fixture_ids",
                                      "read_golden_fixture_ids"]}, callers


def test_match_id_from_innings_id():
    assert bc.match_id_from_innings_id("1_1482013") == "1482013"
    assert bc.match_id_from_innings_id("2_1482013") == "1482013"


def test_window_and_i7_contract_constants():
    assert (bc.WINDOW_START, bc.WINDOW_END) == ("2026-04-17", "2026-08-05")
    assert bc.EXPECTED_RAW_COUNT == 471
    assert bc.I7_SPLITS["golden_start"] == "2026-04-17"
    assert bc.I7_DELIVERY_SEMANTICS == "inclusive_total_runs_v1"
    assert (bc.I7_K_PLAYER, bc.I7_K_VENUE) == (30.0, 200.0)
