import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "research"))

import log_verdict
from sim_eval import claim_gate


HEADER = "date\tidea\tcommit\tll_50k\tmarket_ll\troi_50k_pct\troi_ci\tn_bets\tverdict\tnotes\n"


@pytest.fixture(autouse=True)
def _repo_root_is_case_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(claim_gate, "REPO_ROOT", tmp_path)


def _files(tmp_path, *, status="RUNNING 2026-09-10T00:00:00Z",
           due="2026-12-09", review_count=0):
    ideas = tmp_path / "IDEAS.md"
    ideas.write_text(
        "# Queue\n\n"
        f"## X1 [P1] [{status}] Test idea\n\n"
        "**Method:** test\n\n**Result:** —\n\n---\n"
    )
    results = tmp_path / "results.tsv"
    results.write_text(HEADER)
    program = tmp_path / "program.md"
    program.write_text(
        f"review_due: {due}\nverdicts_since_review: {review_count}\n"
    )
    odds = tmp_path / "odds.json"
    odds.write_text(json.dumps({"matches": []}))
    odds_hash = hashlib.sha256(odds.read_bytes()).hexdigest()
    clusters = tmp_path / "clusters"
    clusters.mkdir()
    registry = tmp_path / "registered.json"
    registry.write_text(json.dumps({"registered_odds": [{
        "role": "test", "path": str(odds), "sha256": odds_hash,
        "cluster_source_dir": str(clusters),
    }]}))
    gate = tmp_path / "gate.json"
    gate_script = tmp_path / "gate_script.py"
    gate_script.write_text("# pre-committed gate\n")
    detail = tmp_path / "detail.json"
    detail.write_text("{}")
    gate.write_text(json.dumps({
        "gate_mode": "manual_sim_prop", "kind": "sim_prop", "idea": "X1",
        "gate_script": {"path": "gate_script.py",
                        "sha256": hashlib.sha256(gate_script.read_bytes()).hexdigest()},
        "detail_json": [{"path": "detail.json",
                         "sha256": hashlib.sha256(detail.read_bytes()).hexdigest()}],
        "verdict": "LANDED", "note": "metric A lower; metric B neutral",
    }))
    result_line = tmp_path / "result.md"
    result_line.write_text("gate-backed result")
    return ideas, results, program, registry, gate, result_line


def _verdict_args(files, verdict="LANDED"):
    ideas, results, program, registry, gate, result_line = files
    return ["verdict", "X1", verdict, "--ideas-path", str(ideas),
            "--results-path", str(results), "--program-path", str(program),
            "--registry-path", str(registry), "--gate-json", str(gate),
            "--date", "2026-09-10", "--commit", "abc123", "--notes", "ok",
            "--result-text-file", str(result_line)]


def test_verdict_requires_gate_json(tmp_path):
    files = _files(tmp_path)
    args = _verdict_args(files)
    gate_index = args.index("--gate-json")
    del args[gate_index:gate_index + 2]
    with pytest.raises(SystemExit):
        log_verdict.main(args)


@pytest.mark.parametrize("fault", ["verdict", "evidence_hash", "mode"])
def test_verdict_refuses_gate_contract_faults(tmp_path, fault):
    files = _files(tmp_path)
    gate = files[4]
    payload = json.loads(gate.read_text())
    stated = "LANDED"
    if fault == "verdict":
        stated = "FAILED"
    elif fault == "evidence_hash":
        payload["gate_script"]["sha256"] = "b" * 64
    else:
        payload["gate_mode"] = "forged"
    gate.write_text(json.dumps(payload))
    before = (files[0].read_bytes(), files[1].read_bytes())
    assert log_verdict.main(_verdict_args(files, stated)) == 1
    assert (files[0].read_bytes(), files[1].read_bytes()) == before


@pytest.mark.parametrize("fault", ["kind", "note", "idea", "detail", "extra"])
def test_landed_claim_is_recomputed_and_tampering_is_refused(tmp_path, fault):
    files = _files(tmp_path)
    gate = files[4]
    payload = json.loads(gate.read_text())
    if fault == "kind":
        payload["kind"] = "match_model"
    elif fault == "note":
        payload["note"] = ""
    elif fault == "idea":
        payload["idea"] = ""
    elif fault == "detail":
        payload["detail_json"] = []
    else:
        payload["fabricated"] = 1
    gate.write_text(json.dumps(payload))
    before = files[0].read_bytes(), files[1].read_bytes()
    assert log_verdict.main(_verdict_args(files)) == 1
    assert (files[0].read_bytes(), files[1].read_bytes()) == before


@pytest.mark.parametrize("fault", ["script_path", "detail_path", "script_shape"])
def test_coordinated_gate_metadata_edits_are_refused(tmp_path, fault):
    files = _files(tmp_path)
    payload = json.loads(files[4].read_text())
    if fault == "script_path":
        payload["gate_script"]["path"] = "missing.py"
    elif fault == "detail_path":
        payload["detail_json"][0]["path"] = "missing.json"
    else:
        payload["gate_script"]["extra"] = True
    files[4].write_text(json.dumps(payload))
    before = files[0].read_bytes(), files[1].read_bytes()
    assert log_verdict.main(_verdict_args(files)) == 1
    assert (files[0].read_bytes(), files[1].read_bytes()) == before


def test_verdict_appends_gate_hash_to_notes(tmp_path):
    files = _files(tmp_path)
    expected = hashlib.sha256(files[4].read_bytes()).hexdigest()
    assert log_verdict.main(_verdict_args(files)) == 0
    row = files[1].read_text().splitlines()[-1]
    assert row.split("\t")[8] == "LANDED"
    assert row.endswith(f"gate_sha256={expected}")
    assert "gate_mode=manual_sim_prop" in row
    assert "verdicts_since_review: 1" in files[2].read_text()


def test_reassess_only_appends_history_and_numbers_revisions(tmp_path):
    files = _files(tmp_path, status="TABLED")
    ideas, results, program, registry, gate, _ = files
    ideas.write_text(ideas.read_text().replace("**Result:** —", "**Result:** completed"))
    results.write_text(
        results.read_text()
        + "2026-01-01\tX1\told\t0\t0\t0\t[0,0]\t0\tTABLED\texisting\n"
    )
    original_ideas = ideas.read_bytes()
    original_results = results.read_bytes()
    base = ["reassess", "X1", "--ideas-path", str(ideas),
            "--results-path", str(results), "--program-path", str(program),
            "--registry-path", str(registry), "--gate-json", str(gate),
            "--date", "2026-09-10", "--commit", "abc123", "--notes", "again"]
    assert log_verdict.main(base) == 0
    assert results.read_bytes().startswith(original_results)
    assert results.read_text().splitlines()[-1].split("\t")[1] == "X1-r2"
    assert ideas.read_bytes().startswith(original_ideas)
    assert "**Reassessment:**" in ideas.read_text()
    assert log_verdict.main(base) == 0
    assert results.read_text().splitlines()[-1].split("\t")[1] == "X1-r3"
    assert "verdicts_since_review: 2" in program.read_text()


@pytest.mark.parametrize("due,count", [("2026-01-01", 0), ("2099-01-01", 10)])
def test_review_reminder_when_date_or_count_due(tmp_path, capsys, due, count):
    files = _files(tmp_path, due=due, review_count=count)
    assert log_verdict.main(_verdict_args(files)) == 0
    assert "REVIEW REMINDER" in capsys.readouterr().err


def test_review_done_resets_counter_and_sets_due_plus_90_days(tmp_path):
    files = _files(tmp_path, review_count=12)
    assert log_verdict.main([
        "review-done", "2026-09-10", "--program-path", str(files[2])
    ]) == 0
    assert files[2].read_text() == (
        "review_due: 2026-12-09\nverdicts_since_review: 0\n"
    )


def test_queue_confirm_is_append_only_and_pending(tmp_path):
    files = _files(tmp_path, status="PROMISING")
    ideas = files[0]
    original = ideas.read_bytes()
    assert log_verdict.main([
        "queue-confirm", "X1", "--ideas-path", str(ideas)
    ]) == 0
    assert ideas.read_bytes().startswith(original)
    assert "## X1-confirm [P1] [PENDING]" in ideas.read_text()
    assert "five aligned seeds" in ideas.read_text()


def test_queue_confirm_requires_promising(tmp_path):
    files = _files(tmp_path, status="RUNNING 2026-09-10T00:00:00Z")
    before = files[0].read_bytes()
    assert log_verdict.main([
        "queue-confirm", "X1", "--ideas-path", str(files[0])
    ]) == 1
    assert files[0].read_bytes() == before
