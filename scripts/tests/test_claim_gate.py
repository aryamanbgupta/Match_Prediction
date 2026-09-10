import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from sim_eval import claim_gate
from sim_eval.claim_gate import decide
from sim_eval.market_math import CostModel
from cricsheet_fixtures import match_json


@pytest.fixture(autouse=True)
def _repo_root_is_case_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(claim_gate, "REPO_ROOT", tmp_path)


def _write_json(path, payload):
    path.write_text(json.dumps(payload))
    return path


def _case(tmp_path, *, n=10, cand_p=0.75, base_p=0.55, reliable=True,
          clusters=True, stamped=False):
    source = tmp_path / "clusters"
    source.mkdir()
    odds_rows, candidate, baseline = [], [], []
    for index in range(n):
        match_id = f"m{index}"
        teams = ["A", "B"]
        prices = {"A": 2.0, "B": 2.0, "timestamp": "x"}
        odds_rows.append({"match_id": match_id, "team1": "A", "team2": "B",
                          "odds": {"winner": prices}})
        if clusters:
            corpus_row = match_json(
                date=f"2026-01-{index + 1:02d}", teams=teams,
                venue=f"V{index}", event=f"E{index}", innings_data=[],
            )
            corpus_row["info"]["event"] = {"name": f"E{index}"}
            _write_json(source / f"{match_id}.json", corpus_row)
        common = {"match_id": match_id, "teams": teams, "actual_winner": "A",
                  "market_odds": prices, "market_prob": {"A": -99, "B": 100},
                  "realized_pnl": 999, "bet_placed": False,
                  "competition_cluster_id": "forged" if stamped else None}
        candidate.append({**common, "simulated_prob": {"A": cand_p, "B": 1-cand_p}})
        baseline.append({**common, "simulated_prob": {"A": base_p, "B": 1-base_p}})
    odds = _write_json(tmp_path / "odds.json", {"matches": odds_rows})
    import hashlib
    registry = _write_json(tmp_path / "registry.json", {
        "registered_odds": [{"role": "test", "path": str(odds),
                             "sha256": hashlib.sha256(odds.read_bytes()).hexdigest(),
                             "cluster_source_dir": str(source)}]
    })
    summary = {"bootstrap_reliable": reliable}
    return source, registry, {"summary": summary, "matches": candidate}, {
        "summary": summary, "matches": baseline}


def _seed_files(tmp_path, candidate, baseline, count):
    candidates, baselines = [], []
    for seed in range(count):
        cand = json.loads(json.dumps(candidate))
        base = json.loads(json.dumps(baseline))
        cand.setdefault("summary", {})["seed"] = seed
        base.setdefault("summary", {})["seed"] = seed
        for row in cand.get("matches", []):
            first, second = row["teams"]
            row["simulated_prob"][first] += seed * 1e-5
            row["simulated_prob"][second] -= seed * 1e-5
        for row in base.get("matches", []):
            first, second = row["teams"]
            row["simulated_prob"][first] += seed * 2e-5
            row["simulated_prob"][second] -= seed * 2e-5
        candidates.append(_write_json(tmp_path / f"candidate_seed{seed}.json", cand))
        baselines.append(_write_json(tmp_path / f"baseline_seed{seed}.json", base))
    return candidates, baselines


def _identical_seed_files(tmp_path, payload, count=5):
    candidates, baselines = _seed_files(tmp_path, payload, payload, count)
    for candidate, baseline in zip(candidates, baselines):
        baseline.write_text(candidate.read_text())
    return candidates, baselines


def _decide(candidates, baselines, source, registry, **kwargs):
    return decide(candidates, baselines, "match_model", CostModel.none(),
                  odds_role="test", cluster_source_dir=source,
                  registry_path=registry, **kwargs)


def test_five_seeds_can_land_and_three_are_promising(tmp_path):
    source, registry, candidate, baseline = _case(tmp_path)
    candidates, baselines = _seed_files(tmp_path, candidate, baseline, 5)
    result = _decide(candidates, baselines, source, registry)
    assert result.verdict == "LANDED"
    assert result.estimator == "seed_mean_match_cluster_ci"
    assert result.provisional is False
    assert result.cost_model == CostModel.none().as_dict()
    assert result.odds_sha256
    assert set(result.arm_files) == {"candidate", "baseline"}
    assert result.block_count == 10
    assert result.profit_block_count == 10
    assert result.bootstrap == {"confidence": .95, "resamples": 10_000,
                                "seed": 42, "reliable": True}
    assert len(result.cost_scenario_diagnostics) == 5
    assert claim_gate.verify_gate_payload(result.as_dict(), registry) == "LANDED"
    result3 = _decide(candidates[:3], baselines[:3], source, registry)
    assert result3.verdict == "PROMISING"
    assert result3.provisional is True


def test_one_seed_uses_match_cluster_estimator_and_failed_branch(tmp_path):
    source, registry, candidate, baseline = _case(tmp_path, cand_p=.55, base_p=.55)
    candidates, baselines = _seed_files(tmp_path, candidate, baseline, 1)
    result = _decide(candidates, baselines, source, registry)
    assert result.verdict == "FAILED"
    assert result.estimator == "match_cluster_ci"
    assert result.provisional is True


def test_single_seed_may_be_unstamped_and_uses_single_label(tmp_path):
    source, registry, candidate, baseline = _case(tmp_path)
    cand_path = _write_json(tmp_path / "candidate.json", candidate)
    base_path = _write_json(tmp_path / "baseline.json", baseline)
    result = _decide([cand_path], [base_path], source, registry)
    assert result.seeds == ["single"]


def test_provisional_match_model_must_clear_seed_floor(tmp_path):
    source, registry, candidate, baseline = _case(tmp_path, cand_p=.553, base_p=.55)
    candidates, baselines = _seed_files(tmp_path, candidate, baseline, 3)
    result = _decide(candidates, baselines, source, registry)
    assert result.delta_log_loss.ci95[1] < 0
    assert -result.delta_log_loss.point < .007
    assert result.verdict == "FAILED"


def test_stamped_clusters_are_ignored(tmp_path):
    source, registry, candidate, baseline = _case(tmp_path, stamped=True)
    candidates, baselines = _seed_files(tmp_path, candidate, baseline, 5)
    result = _decide(candidates, baselines, source, registry)
    assert result.cluster_contract["resolution_counts"] == {"lookup": 50}
    assert result.block_count == 10


def test_fallback_and_unreliable_bootstrap_are_descriptive(tmp_path):
    source, registry, candidate, baseline = _case(tmp_path)
    (source / "m9.json").unlink()
    candidates, baselines = _seed_files(tmp_path, candidate, baseline, 5)
    assert _decide(candidates, baselines, source, registry).verdict == "DESCRIPTIVE"

    other = tmp_path / "unreliable"
    other.mkdir()
    source2, registry2, candidate2, baseline2 = _case(other, reliable=False)
    candidates2, baselines2 = _seed_files(other, candidate2, baseline2, 5)
    assert _decide(candidates2, baselines2, source2, registry2).verdict == "DESCRIPTIVE"


def test_fewer_than_ten_blocks_is_descriptive(tmp_path):
    source, registry, candidate, baseline = _case(tmp_path, n=9)
    candidates, baselines = _seed_files(tmp_path, candidate, baseline, 5)
    assert _decide(candidates, baselines, source, registry).verdict == "DESCRIPTIVE"


def test_profit_population_with_only_nine_blocks_is_descriptive(tmp_path):
    source, registry, candidate, baseline = _case(tmp_path)
    candidate["matches"][-1]["simulated_prob"] = {"A": .5, "B": .5}
    baseline["matches"][-1]["simulated_prob"] = {"A": .5, "B": .5}
    candidates, baselines = _seed_files(tmp_path, candidate, baseline, 1)
    result = _decide(candidates, baselines, source, registry)
    assert result.block_count == 10
    assert result.n_profit_rows == 9
    assert result.profit_block_count == 9
    assert result.verdict == "DESCRIPTIVE"


def test_seed_identity_is_content_stamped_and_prediction_distinct(tmp_path):
    source, registry, candidate, baseline = _case(tmp_path)
    candidates, baselines = _seed_files(tmp_path, candidate, baseline, 5)
    with pytest.raises(ValueError, match="duplicate paths"):
        _decide([candidates[0]] * 5, baselines, source, registry,
                seeds=[1, 2, 3, 4, 5])
    copies = []
    for seed, whitespace in enumerate((None, 1, 2, 3, 4)):
        copy = tmp_path / f"copied_candidate_{seed}.json"
        payload = json.loads(candidates[0].read_text())
        copy.write_text(json.dumps(payload, indent=whitespace))
        copies.append(copy)
    with pytest.raises(ValueError, match="duplicate file hashes"):
        _decide(copies, baselines, source, registry, seeds=[0, 1, 2, 3, 4])

    unstamped = json.loads(candidates[1].read_text())
    unstamped["summary"].pop("seed")
    candidates[1].write_text(json.dumps(unstamped))
    with pytest.raises(ValueError, match="unstamped seed file"):
        _decide(candidates, baselines, source, registry,
                seeds=[0, 1, 2, 3, 4])
    candidates, baselines = _seed_files(tmp_path, candidate, baseline, 5)

    duplicate_predictions = json.loads(candidates[1].read_text())
    duplicate_predictions["matches"] = json.loads(
        json.dumps(json.loads(candidates[0].read_text())["matches"])
    )
    candidates[1].write_text(json.dumps(duplicate_predictions))
    with pytest.raises(ValueError, match="duplicate predictions across seeds"):
        _decide(candidates, baselines, source, registry)
    candidates, baselines = _seed_files(tmp_path, candidate, baseline, 5)

    wrong = json.loads(candidates[2].read_text())
    wrong["summary"]["seed"] = 99
    candidates[2].write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="stamped seed"):
        _decide(candidates, baselines, source, registry,
                seeds=[0, 1, 2, 3, 4])


def test_registered_cluster_dir_is_authoritative(tmp_path):
    source, registry, candidate, baseline = _case(tmp_path)
    candidates, baselines = _seed_files(tmp_path, candidate, baseline, 1)
    other = tmp_path / "other-clusters"
    shutil.copytree(source, other)
    with pytest.raises(ValueError, match="registered directory"):
        _decide(candidates, baselines, other, registry)


def test_id_and_registered_odds_mismatches_raise(tmp_path):
    source, registry, candidate, baseline = _case(tmp_path)
    baseline["matches"][0]["match_id"] = "other"
    candidates, baselines = _seed_files(tmp_path, candidate, baseline, 1)
    with pytest.raises(ValueError, match="match id mismatch"):
        _decide(candidates, baselines, source, registry)

    other = tmp_path / "odds-mismatch"
    other.mkdir()
    source, registry, candidate, baseline = _case(other)
    candidate["matches"][3]["market_odds"]["A"] = 2.1
    candidates, baselines = _seed_files(other, candidate, baseline, 1)
    with pytest.raises(ValueError, match="m3"):
        _decide(candidates, baselines, source, registry)


def test_stored_derived_fields_do_not_affect_result(tmp_path):
    source, registry, candidate, baseline = _case(tmp_path)
    candidates, baselines = _seed_files(tmp_path, candidate, baseline, 1)
    first = _decide(candidates, baselines, source, registry)
    for payload in (candidate, baseline):
        for row in payload["matches"]:
            row["market_prob"] = "corrupt"
            row["realized_pnl"] = -123456
            row["edge"] = "corrupt"
            row["bet_placed"] = not row["bet_placed"]
    candidates2 = [_write_json(tmp_path / "candidate2_seed0.json", candidate)]
    baselines2 = [_write_json(tmp_path / "baseline2_seed0.json", baseline)]
    second = _decide(candidates2, baselines2, source, registry)
    first_dict, second_dict = first.as_dict(), second.as_dict()
    first_dict.pop("arm_files")
    second_dict.pop("arm_files")
    assert first_dict == second_dict


def test_profit_delta_uses_union_with_zero_for_no_bet(tmp_path):
    source, registry, candidate, baseline = _case(tmp_path, cand_p=.6, base_p=.5)
    candidates, baselines = _seed_files(tmp_path, candidate, baseline, 1)
    result = _decide(candidates, baselines, source, registry)
    assert result.delta_profit.point == pytest.approx(1.0)
    assert result.n_profit_rows == 10

    before = result.delta_profit
    odds_path = tmp_path / "odds.json"
    odds_payload = json.loads(odds_path.read_text())
    for index in range(10, 20):
        match_id = f"m{index}"
        prices = {"A": 2.0, "B": 2.0, "timestamp": "x"}
        odds_payload["matches"].append({
            "match_id": match_id, "team1": "A", "team2": "B",
            "odds": {"winner": prices},
        })
        corpus = match_json(date=f"2026-02-{index - 9:02d}", teams=["A", "B"],
                            venue=f"N{index}", event=f"N{index}", innings_data=[])
        corpus["info"]["event"] = {"name": f"N{index}"}
        _write_json(source / f"{match_id}.json", corpus)
        row = {"match_id": match_id, "teams": ["A", "B"],
               "actual_winner": "A", "market_odds": prices,
               "simulated_prob": {"A": .5, "B": .5}}
        candidate["matches"].append(dict(row))
        baseline["matches"].append(dict(row))
    odds_path.write_text(json.dumps(odds_payload))
    registry.write_text(json.dumps({"registered_odds": [{
        "role": "test", "path": str(odds_path),
        "sha256": hashlib.sha256(odds_path.read_bytes()).hexdigest(),
        "cluster_source_dir": str(source),
    }]}))
    more = tmp_path / "more"
    more.mkdir()
    candidates2, baselines2 = _seed_files(more, candidate, baseline, 1)
    after = _decide(candidates2, baselines2, source, registry)
    assert after.n_profit_rows == result.n_profit_rows
    assert after.delta_profit == before


def _betting_files(tmp_path, count=5, *, stake=1.0, baseline_stake=0.0):
    files = []
    for seed in range(count):
        rows = [{"match_id": f"m{i}", "cand_bet_team": "A",
                 "cand_stake": stake, "base_bet_team": ("A" if baseline_stake else None),
                 "base_stake": baseline_stake} for i in range(10)]
        # Lock that placements, not mere seed labels, differ across seeds.
        rows[seed % 10]["cand_bet_team"] = None
        rows[seed % 10]["cand_stake"] = 0.0
        if baseline_stake == 0.0:
            rows[seed % 10]["base_bet_team"] = "B"
            rows[seed % 10]["base_stake"] = 1.0
        files.append(_write_json(tmp_path / f"bet_seed{seed}.json", {
            "kind": "betting_layer", "summary": {"seed": seed},
            "metric_a": {"name": "flat_pnl" if stake in (0.0, 1.0) else "kelly_pnl"},
            "rows": rows,
        }))
    return files


def test_kind_specific_betting_gate_pair_and_manual_sim_prop(tmp_path, capsys):
    source, registry, candidate, _ = _case(tmp_path, cand_p=.6, base_p=.5)
    baseline = json.loads(json.dumps(candidate))
    common = dict(odds_role="test", cluster_source_dir=source,
                  registry_path=registry)
    candidates, baselines = _identical_seed_files(tmp_path, candidate)
    betting_files = _betting_files(tmp_path)
    betting = decide(candidates, baselines, "betting_layer", CostModel.none(),
                     metrics_json=betting_files, **common)
    assert betting.verdict == "LANDED"
    assert "metric_b" not in betting.as_dict()
    assert betting.delta_roi.ci95[0] > 0

    script = tmp_path / "gate.py"
    script.write_text("# pre-committed gate\n")
    detail = _write_json(tmp_path / "detail.json", {"observations": [1, 2]})
    out = tmp_path / "manual.json"
    assert claim_gate.main(["record-manual", "--kind", "sim_prop", "--idea", "S1",
                            "--gate-script", str(script), "--detail-json", str(detail),
                            "--verdict", "TABLED", "--note", "MAE lower; bias neutral",
                            "--out", str(out)]) == 0
    payload = json.loads(out.read_text())
    assert payload["gate_mode"] == "manual_sim_prop"
    assert claim_gate.verify_gate_payload(payload, registry) == "TABLED"
    detail.write_text("{}")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        claim_gate.verify_gate_payload(payload, registry)


def test_metric_names_directions_settlement_and_source_hash_are_locked(tmp_path):
    source, registry, candidate, _ = _case(tmp_path, cand_p=.6, base_p=.5)
    baseline = json.loads(json.dumps(candidate))
    candidates, baselines = _identical_seed_files(tmp_path, candidate)
    common = dict(odds_role="test", cluster_source_dir=source,
                  registry_path=registry, seeds=range(5))
    placements = _betting_files(tmp_path)
    for seed, path in enumerate(placements):
        payload = json.loads(path.read_text())
        for row in payload["rows"]:
            if row["cand_stake"]:
                row["cand_bet_team"] = "B"
        path.write_text(json.dumps(payload))
    losing = decide(candidates, baselines, "betting_layer", CostModel.none(),
                    metrics_json=placements, **common)
    assert losing.verdict != "LANDED"
    assert losing.delta_profit.point < 0

    bad_direction = json.loads(placements[0].read_text())
    bad_direction["metric_a"]["favourable"] = "lower"
    placements[0].write_text(json.dumps(bad_direction))
    with pytest.raises(ValueError, match="direction fields"):
        decide(candidates, baselines, "betting_layer", CostModel.none(),
               metrics_json=placements, **common)
    bad_direction["metric_a"] = {"name": "roi"}
    placements[0].write_text(json.dumps(bad_direction))
    with pytest.raises(ValueError, match="unknown metric_a"):
        decide(candidates, baselines, "betting_layer", CostModel.none(),
               metrics_json=placements, **common)

    bad_direction["metric_a"] = {"name": "flat_pnl"}
    placements[0].write_text(json.dumps(bad_direction))
    payload = json.loads(placements[1].read_text())
    payload["metric_b"] = {"name": "log_loss"}
    placements[1].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="does not accept metric_b"):
        decide(candidates, baselines, "betting_layer", CostModel.none(),
               metrics_json=placements, **common)


def test_ll_clear_but_profit_clearly_harmful_is_tabled(tmp_path):
    source, registry, candidate, baseline = _case(tmp_path, n=20)
    odds_path = tmp_path / "odds.json"
    odds_payload = json.loads(odds_path.read_text())
    for index, (cand, base, odds_row) in enumerate(zip(
            candidate["matches"], baseline["matches"], odds_payload["matches"])):
        cluster_path = source / f"m{index}.json"
        cluster_payload = json.loads(cluster_path.read_text())
        cluster_payload["info"]["event"]["name"] = f"E{index // 2}"
        cluster_path.write_text(json.dumps(cluster_payload))
        if index % 2 == 0:
            prices = {"A": 1.0101010101, "B": 100.0, "timestamp": "x"}
            cand["simulated_prob"] = {"A": .9, "B": .1}
            base["simulated_prob"] = {"A": .1, "B": .9}
        else:
            prices = {"A": 2.0, "B": 2.0, "timestamp": "x"}
            cand["simulated_prob"] = {"A": .49, "B": .51}
            base["simulated_prob"] = {"A": .5, "B": .5}
        cand["market_odds"] = prices
        base["market_odds"] = prices
        odds_row["odds"]["winner"] = prices
    odds_path.write_text(json.dumps(odds_payload))
    import hashlib
    registry.write_text(json.dumps({"registered_odds": [{
        "role": "test", "path": str(odds_path),
        "sha256": hashlib.sha256(odds_path.read_bytes()).hexdigest(),
        "cluster_source_dir": str(source),
    }]}))
    candidates, baselines = _seed_files(tmp_path, candidate, baseline, 5)
    result = _decide(candidates, baselines, source, registry)
    assert result.delta_log_loss.ci95[1] < 0
    assert result.delta_profit.ci95[1] < 0
    assert result.verdict == "TABLED"


def test_missing_outcome_is_dropped_from_both_arms(tmp_path):
    source, registry, candidate, baseline = _case(tmp_path, n=11)
    candidate["matches"][0]["actual_winner"] = None
    candidates, baselines = _seed_files(tmp_path, candidate, baseline, 1)
    result = _decide(candidates, baselines, source, registry)
    assert result.n_records == 10
    assert result.n_dropped_symmetrically == 1


def test_non_finite_interval_is_an_error(tmp_path, monkeypatch):
    source, registry, candidate, baseline = _case(tmp_path)
    candidates, baselines = _seed_files(tmp_path, candidate, baseline, 1)
    monkeypatch.setattr(claim_gate, "match_cluster_ci",
                        lambda *args, **kwargs: [float("nan"), 0.0])
    with pytest.raises(ValueError, match="non-finite"):
        _decide(candidates, baselines, source, registry)


def test_gate_replay_refuses_fabricated_numbers_and_edited_evidence(tmp_path):
    source, registry, candidate, baseline = _case(tmp_path)
    candidates, baselines = _seed_files(tmp_path, candidate, baseline, 5)
    result = _decide(candidates, baselines, source, registry)
    genuine = result.as_dict()
    assert claim_gate.verify_gate_payload(genuine, registry) == result.verdict

    fabricated = json.loads(json.dumps(genuine))
    fabricated["delta_log_loss"] = {"point": -999.0, "ci95": [-1000.0, -998.0]}
    with pytest.raises(ValueError, match="differs from replayed decision"):
        claim_gate.verify_gate_payload(fabricated, registry)

    edited = json.loads(candidates[0].read_text())
    edited["matches"][0]["simulated_prob"] = {"A": .51, "B": .49}
    candidates[0].write_text(json.dumps(edited))
    with pytest.raises(ValueError, match="evidence sha256 mismatch"):
        claim_gate.verify_gate_payload(genuine, registry)


def test_duplicate_checks_use_kept_predictions_and_placements(tmp_path):
    source, registry, candidate, baseline = _case(tmp_path)
    candidates, baselines = _seed_files(tmp_path, candidate, baseline, 5)
    first = json.loads(candidates[0].read_text())
    second = json.loads(candidates[1].read_text())
    second["matches"] = json.loads(json.dumps(first["matches"]))
    for seed, (cand_path, base_path) in enumerate(zip(candidates, baselines)):
        cand = json.loads(cand_path.read_text())
        base = json.loads(base_path.read_text())
        dropped = {"match_id": "unregistered", "teams": ["A", "B"],
                   "actual_winner": None, "market_odds": {},
                   "simulated_prob": {"A": .1 + seed / 100, "B": .9 - seed / 100}}
        cand["matches"].append(dropped)
        base["matches"].append(dropped)
        cand_path.write_text(json.dumps(cand)); base_path.write_text(json.dumps(base))
    candidates[1].write_text(json.dumps({**second, "matches": second["matches"] + [
        {"match_id": "unregistered", "teams": ["A", "B"], "actual_winner": None,
         "market_odds": {}, "simulated_prob": {"A": .99, "B": .01}}
    ]}))
    with pytest.raises(ValueError, match="duplicate predictions across seeds in candidate"):
        _decide(candidates, baselines, source, registry)

    candidates, baselines = _identical_seed_files(tmp_path, candidate)
    placements = _betting_files(tmp_path)
    duplicate = json.loads(placements[0].read_text())
    duplicate["summary"]["seed"] = 1
    placements[1].write_text(json.dumps(duplicate))
    with pytest.raises(ValueError, match="duplicate placements across seeds"):
        decide(candidates, baselines, "betting_layer", CostModel.none(),
               odds_role="test", cluster_source_dir=source, registry_path=registry,
               metrics_json=placements)


def test_betting_layer_decides_ratio_of_sums_roi_and_preserves_probabilities(tmp_path):
    source, registry, candidate, _ = _case(tmp_path, cand_p=.6)
    candidates, baselines = _identical_seed_files(tmp_path, candidate)
    placements = []
    for seed in range(5):
        base_stake = .1 + seed / 1000
        rows = [{"match_id": f"m{i}", "cand_bet_team": "A",
                 "cand_stake": 2 * base_stake,
                 "base_bet_team": "A", "base_stake": base_stake}
                for i in range(10)]
        placements.append(_write_json(tmp_path / f"kelly_seed{seed}.json", {
            "kind": "betting_layer", "summary": {"seed": seed},
            "metric_a": {"name": "kelly_pnl"}, "rows": rows,
        }))
    result = decide(candidates, baselines, "betting_layer", CostModel.none(),
                    odds_role="test", cluster_source_dir=source, registry_path=registry,
                    metrics_json=placements)
    assert result.delta_profit.point > 0
    assert result.delta_roi == claim_gate.MetricDelta(0.0, (0.0, 0.0))
    assert result.verdict == "FAILED"
    assert len(result.cost_scenario_diagnostics) == 5
    assert all("delta_roi" in item for item in result.cost_scenario_diagnostics)

    changed = json.loads(candidates[0].read_text())
    changed["matches"][0]["simulated_prob"]["A"] += 1e-6
    candidates[0].write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="betting_layer claim changes probabilities"):
        decide(candidates, baselines, "betting_layer", CostModel.none(),
               odds_role="test", cluster_source_dir=source, registry_path=registry,
               metrics_json=placements)


def test_cli_writes_json_and_prints_its_sha256(tmp_path, capsys):
    source, registry, candidate, baseline = _case(tmp_path)
    candidates, baselines = _seed_files(tmp_path, candidate, baseline, 1)
    output = tmp_path / "gate.json"
    assert claim_gate.main([
        "--candidate", str(candidates[0]), "--baseline", str(baselines[0]),
        "--kind", "match_model", "--odds-role", "test",
        "--cluster-source-dir", str(source), "--registry-path", str(registry),
        "--out", str(output),
    ]) == 0
    import hashlib
    printed = capsys.readouterr().out.strip().split()
    assert printed[0] == json.loads(output.read_text())["verdict"]
    assert printed[1] == hashlib.sha256(output.read_bytes()).hexdigest()


@pytest.mark.parametrize("kind,evidence", [
    ("match_model", ["--metrics-json", "metrics.json"]),
    ("betting_layer", ["--candidate", "candidate.json",
                       "--baseline", "baseline.json"]),
    ("sim_prop", ["--candidate", "candidate.json",
                  "--baseline", "baseline.json", "--metrics-json", "metrics.json"]),
])
def test_cli_rejects_wrong_evidence_shape(kind, evidence, tmp_path):
    with pytest.raises(SystemExit):
        claim_gate.main([
            "--kind", kind, *evidence, "--odds-role", "test",
            "--out", str(tmp_path / "gate.json"),
        ])


def test_genuine_gate_json_survives_a_file_round_trip(tmp_path):
    """Astra round 4: the dataclass emits tuples, the JSON file carries lists;
    a genuine gate file written to disk must verify identically."""
    source, registry, candidate, baseline = _case(tmp_path)
    candidates, baselines = _seed_files(tmp_path, candidate, baseline, 5)
    result = _decide(candidates, baselines, source, registry)
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    reloaded = json.loads(gate_path.read_text())
    assert claim_gate.verify_gate_payload(reloaded, registry) == result.verdict


def test_duplicate_seed_detection_is_numeric_not_textual(tmp_path):
    """Astra round 4: "0.75" and "0.750" are the same prediction."""
    source, registry, candidate, baseline = _case(tmp_path)
    candidates, baselines = _seed_files(tmp_path, candidate, baseline, 5)
    first = json.loads(candidates[0].read_text())
    for index, path in enumerate(candidates[1:], start=1):
        copy = json.loads(json.dumps(first))
        copy["summary"] = json.loads(path.read_text())["summary"]
        for row in copy["matches"]:
            row["simulated_prob"] = {
                team: f"{float(value):.{3 + index}f}"
                for team, value in row["simulated_prob"].items()
            }
        path.write_text(json.dumps(copy))
    with pytest.raises(ValueError, match="duplicate predictions across seeds"):
        _decide(candidates, baselines, source, registry)
    # Signed zero is the same number (Astra round 5).
    assert claim_gate._prediction_vector([{"simulated_prob": {"A": -0.0, "B": 1.0}}]) == \
        claim_gate._prediction_vector([{"simulated_prob": {"A": 0.0, "B": 1.0}}])


def test_betting_layer_rejects_non_finite_probabilities(tmp_path):
    """Astra round 4: NaN compares unequal to everything, so the
    probability-invariance check must validate finiteness first."""
    source, registry, candidate, baseline = _case(tmp_path)
    candidates, baselines = _identical_seed_files(tmp_path, candidate)
    placements = _betting_files(tmp_path)
    poisoned = json.loads(candidates[0].read_text())
    poisoned["matches"][0]["simulated_prob"] = {"A": float("nan"), "B": float("nan")}
    candidates[0].write_text(json.dumps(poisoned))
    with pytest.raises(ValueError, match="non-finite simulated probability"):
        decide(candidates, baselines, "betting_layer", CostModel.none(),
               odds_role="test", cluster_source_dir=source, registry_path=registry,
               metrics_json=placements)
