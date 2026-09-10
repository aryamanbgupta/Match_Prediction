"""Pins the sequence-track stage-1 arm runner's seams (D5 5.4/5.5, D6 6.1).

Four things must hold before any arm is run for real:

* the per-fixture seed rule is a pure function of (cricsheet id, base seed);
* the evaluator's clip seam is OFF by default and byte-identical to the
  historical inline clip, and clips to the requested bounds when ON;
* the evaluator's simulation-seed seam defaults to the legacy 42 and is
  overridable by a subclass;
* the provenance writer emits every field the cross-arm audit reads.

None of these need a model artifact or a stats cache.
"""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from sequence_track.run_arm import (SUB_SEED_STREAMS, _LifecycleRecorder,
                                    _resolve_device,
                                    assert_seed_intervals_disjoint,
                                    fixture_provenance, fixture_seed,
                                    largest_permitted_n_sims, min_seed_gap,
                                    registered_mismatches,
                                    seed_intervals_disjoint, sub_seed,
                                    sub_seeds, write_provenance)
from sim_eval.match_evaluator import (DEFAULT_PROB_CLIP,
                                      LEGACY_SIMULATION_SEED,
                                      MatchLevelEvaluator)


LOW31 = (1 << 31) - 1


# --------------------------------------------------------------------------
# D5 5.4 — per-fixture seeds
# --------------------------------------------------------------------------

def _expected_fixture_seed(match_id: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{match_id}:{base_seed}".encode()).hexdigest()
    return int(digest, 16) & LOW31


def test_fixture_seed_matches_the_written_rule():
    assert fixture_seed("1478908", 20260910) == _expected_fixture_seed(
        "1478908", 20260910)


def test_fixture_seed_is_deterministic_and_in_range():
    first = fixture_seed("1478908", 20260910)
    assert first == fixture_seed("1478908", 20260910)
    assert 0 <= first <= LOW31


def test_fixture_seed_differs_across_ids_and_base_seeds():
    base = 20260910
    ids = ["1478908", "1501896", "1505127", "1477609"]
    seeds = {fixture_seed(match_id, base) for match_id in ids}
    assert len(seeds) == len(ids)
    assert fixture_seed("1478908", base) != fixture_seed("1478908", base + 1)
    # The 1b variability rerun changes only the base seed: every fixture must
    # move.
    other = {fixture_seed(match_id, base + 1) for match_id in ids}
    assert seeds.isdisjoint(other)


def test_sub_seeds_are_three_distinct_deterministic_streams():
    seed = fixture_seed("1478908", 20260910)
    derived = sub_seeds(seed)
    assert tuple(derived) == SUB_SEED_STREAMS
    assert len(set(derived.values())) == 3
    for name, value in derived.items():
        assert value == sub_seed(seed, name)
        assert 0 <= value <= LOW31
        expected = int(
            hashlib.sha256(f"{seed}:{name}".encode()).hexdigest(), 16) & LOW31
        assert value == expected
    assert sub_seeds(seed + 1) != derived


# --------------------------------------------------------------------------
# D5 5.5 — the clipping seam
# --------------------------------------------------------------------------

def _evaluator(**kwargs):
    return MatchLevelEvaluator(
        model=None, simulation_engine=None, n_simulations=1, parallel=False,
        **kwargs)


@pytest.mark.parametrize("pair", [
    (1.0, 0.0), (0.0, 1.0), (0.5, 0.5), (0.97, 0.03), (0.06, 0.94),
    (0.5123456789, 0.4876543211),
])
def test_clip_off_is_byte_identical_to_the_historical_block(pair):
    evaluator = _evaluator()
    assert evaluator.prob_clip is None

    # Verbatim copy of the block the seam replaced.
    prob_floor, prob_ceiling = 0.05, 0.95
    team1_prob = max(prob_floor, min(prob_ceiling, pair[0]))
    team2_prob = max(prob_floor, min(prob_ceiling, pair[1]))
    clip_total = team1_prob + team2_prob
    expected = (team1_prob / clip_total, team2_prob / clip_total)

    assert evaluator._clip_and_normalize(*pair) == expected
    assert DEFAULT_PROB_CLIP == (0.05, 0.95)


def test_clip_on_pins_a_certain_result_to_the_requested_bounds():
    evaluator = _evaluator()
    evaluator.prob_clip = (0.01, 0.99)
    high, low = evaluator._clip_and_normalize(1.0, 0.0)
    assert high == pytest.approx(0.99)
    assert low == pytest.approx(0.01)
    assert high + low == pytest.approx(1.0)

    # A pair inside the bounds is untouched (beyond renormalisation).
    inside = evaluator._clip_and_normalize(0.62, 0.38)
    assert inside == pytest.approx((0.62, 0.38))


def test_clip_seam_is_per_instance_not_global():
    clipped = _evaluator()
    clipped.prob_clip = (0.01, 0.99)
    assert _evaluator().prob_clip is None
    assert MatchLevelEvaluator.prob_clip is None


# --------------------------------------------------------------------------
# D5 5.4 — the evaluator's simulation-seed seam
# --------------------------------------------------------------------------

def _match_result(winner, team1_score=160, team2_score=150):
    from sim_v1_2 import MatchResult
    return MatchResult(
        match_id="sim", team1="A", team2="B", winner=winner, margin="",
        innings=[], team1_score=team1_score, team1_wickets=5,
        team2_score=team2_score, team2_wickets=7)


class _StubEngine:
    def __init__(self, results):
        self.results = results
        self.configs = []

    def simulate_multiple(self, state, config):
        self.configs.append(config)
        return self.results


def _single_match(evaluator_cls, **attrs):
    engine = _StubEngine([_match_result("A"), _match_result("B", 150, 160)])
    evaluator = evaluator_cls(
        model=None, simulation_engine=engine, n_simulations=2, parallel=False)
    for key, value in attrs.items():
        setattr(evaluator, key, value)
    state = SimpleNamespace(team1="A", team2="B")
    odds = {"odds": {"winner": {"A": 2.0, "B": 2.0}}, "actual_winner": "A"}
    result = evaluator._evaluate_single_match("1478908", state, odds)
    return engine, result


def test_legacy_simulation_seed_is_42():
    assert MatchLevelEvaluator._simulation_seed(
        MatchLevelEvaluator, "1478908") == LEGACY_SIMULATION_SEED
    engine, _result = _single_match(MatchLevelEvaluator)
    assert [config.random_seed for config in engine.configs] == [42]


def test_subclass_can_derive_a_per_fixture_seed():
    class _ArmEvaluator(MatchLevelEvaluator):
        base_seed = 20260910

        def _simulation_seed(self, match_id):
            return fixture_seed(str(match_id), self.base_seed)

    engine, _result = _single_match(_ArmEvaluator)
    assert engine.configs[0].random_seed == fixture_seed("1478908", 20260910)
    assert engine.configs[0].random_seed != 42


def test_clip_seam_reaches_the_scored_probabilities():
    # Both simulations won by A -> raw 1.0 / 0.0 before clipping.
    engine = _StubEngine([_match_result("A"), _match_result("A")])
    evaluator = MatchLevelEvaluator(
        model=None, simulation_engine=engine, n_simulations=2, parallel=False)
    evaluator.prob_clip = (0.01, 0.99)
    state = SimpleNamespace(team1="A", team2="B")
    odds = {"odds": {"winner": {"A": 2.0, "B": 2.0}}, "actual_winner": "A"}
    result = evaluator._evaluate_single_match("1478908", state, odds)
    assert result.simulated_win_prob["A"] == pytest.approx(0.99)
    assert result.simulated_win_prob["B"] == pytest.approx(0.01)


# --------------------------------------------------------------------------
# D6 6.1 — the provenance record
# --------------------------------------------------------------------------

def _document(date="2025-10-31", winner="India", gender="male",
              match_type="T20"):
    return {
        "info": {
            "dates": [date],
            "gender": gender,
            "match_type": match_type,
            "teams": ["India", "Australia"],
            "outcome": ({"winner": winner} if winner
                        else {"result": "no result"}),
        },
        "innings": [{"team": "India"}, {"team": "Australia"}],
    }


ODDS_ROW = {
    "match_id": "1478908",
    "cricsheet_id": "1478908",
    "display_match_id": "2025-10-31_India_Australia_Bellerive_Oval",
    "actual_winner": "Australia",
    "odds": {"winner": {"India": 2.0, "Australia": 2.0}},
}


def test_fixture_provenance_has_every_audited_field():
    row = fixture_provenance(
        cricsheet_id="1478908",
        document=_document(winner="Australia"),
        as_of_date="2025-10-31",
        same_day_advanced_before=["1477000", "1477001"],
        matches_advanced=17,
        odds_row=ODDS_ROW,
        scored=True,
        skip_reason=None,
        base_seed=20260910,
        n_sims=100,
    )
    assert set(row) == {
        "cricsheet_id", "match_date", "as_of", "eligibility",
        "fixture_seed", "sub_seeds", "n_sims",
    }
    assert row["match_date"] == "2025-10-31"
    assert row["as_of"] == {
        "date": "2025-10-31",
        "same_day_advanced_before": ["1477000", "1477001"],
        "matches_advanced": 17,
    }
    assert set(row["eligibility"]) == {
        "odds_row_found", "odds_row_id", "odds_row_sha256",
        "odds_actual_winner", "cricsheet_resolved", "cricsheet_winner",
        "male_t20", "scored", "skip_reason",
    }
    assert row["fixture_seed"] == fixture_seed("1478908", 20260910)
    assert row["sub_seeds"] == sub_seeds(row["fixture_seed"])
    assert row["n_sims"] == 100


def test_fixture_provenance_records_the_odds_row_the_evaluator_resolved():
    row = fixture_provenance(
        cricsheet_id="1478908", document=_document(winner="Australia"),
        as_of_date="2025-10-31", same_day_advanced_before=[],
        matches_advanced=0, odds_row=ODDS_ROW, scored=True, skip_reason=None,
        base_seed=1, n_sims=1)["eligibility"]
    assert row["odds_row_found"] is True
    assert row["odds_row_id"] == {
        "cricsheet_id": "1478908",
        "display_match_id": "2025-10-31_India_Australia_Bellerive_Oval"}
    assert row["odds_actual_winner"] == "Australia"
    assert row["cricsheet_winner"] == "Australia"
    assert row["cricsheet_resolved"] is True
    assert row["scored"] is True and row["skip_reason"] is None

    # The two winners are DIFFERENT facts and both are recorded: the
    # evaluator scores against the odds row, not the cricsheet document.
    disagreeing = dict(ODDS_ROW, actual_winner="India")
    other = fixture_provenance(
        cricsheet_id="1478908", document=_document(winner="Australia"),
        as_of_date="2025-10-31", same_day_advanced_before=[],
        matches_advanced=0, odds_row=disagreeing, scored=True,
        skip_reason=None, base_seed=1, n_sims=1)["eligibility"]
    assert other["odds_actual_winner"] == "India"
    assert other["cricsheet_winner"] == "Australia"
    assert other["odds_row_sha256"] != row["odds_row_sha256"]


def test_fixture_provenance_row_hash_is_canonical_and_order_free():
    shuffled = {key: ODDS_ROW[key] for key in reversed(list(ODDS_ROW))}
    def _hash(row):
        return fixture_provenance(
            cricsheet_id="1", document=_document(), as_of_date="2025-10-31",
            same_day_advanced_before=[], matches_advanced=0, odds_row=row,
            scored=True, skip_reason=None, base_seed=1,
            n_sims=1)["eligibility"]["odds_row_sha256"]
    assert _hash(ODDS_ROW) == _hash(shuffled)
    assert len(_hash(ODDS_ROW)) == 64


def test_fixture_provenance_records_a_skip():
    row = fixture_provenance(
        cricsheet_id="1", document=_document(winner=None),
        as_of_date="2025-10-31", same_day_advanced_before=[],
        matches_advanced=0, odds_row=None, scored=False,
        skip_reason="no_odds_row", base_seed=1, n_sims=1)["eligibility"]
    assert row == {
        "odds_row_found": False,
        "odds_row_id": None,
        "odds_row_sha256": None,
        "odds_actual_winner": None,
        "cricsheet_resolved": False,
        "cricsheet_winner": None,
        "male_t20": True,
        "scored": False,
        "skip_reason": "no_odds_row",
    }


def test_fixture_provenance_eligibility_reads_the_cricsheet_document():
    womens = fixture_provenance(
        cricsheet_id="2", document=_document(gender="female"),
        as_of_date="2025-10-31", same_day_advanced_before=[],
        matches_advanced=0, odds_row=ODDS_ROW, scored=True, skip_reason=None,
        base_seed=1, n_sims=1)
    assert womens["eligibility"]["male_t20"] is False

    odi = fixture_provenance(
        cricsheet_id="3", document=_document(match_type="ODI"),
        as_of_date="2025-10-31", same_day_advanced_before=[],
        matches_advanced=0, odds_row=ODDS_ROW, scored=True, skip_reason=None,
        base_seed=1, n_sims=1)
    assert odi["eligibility"]["male_t20"] is False


def test_write_provenance_orders_fixtures_and_keeps_the_contract(tmp_path):
    rows = [
        fixture_provenance(
            cricsheet_id="1501896", document=_document("2025-10-31"),
            as_of_date="2025-10-31", same_day_advanced_before=["1478908"],
            matches_advanced=18, odds_row=ODDS_ROW, scored=True,
            skip_reason=None, base_seed=7, n_sims=10),
        fixture_provenance(
            cricsheet_id="1477609", document=_document("2025-07-05"),
            as_of_date="2025-07-05", same_day_advanced_before=[],
            matches_advanced=3, odds_row=ODDS_ROW, scored=True,
            skip_reason=None, base_seed=7, n_sims=10),
    ]
    run = {"arm": "A", "base_seed": 7, "n_sims": 10}
    path = write_provenance(tmp_path, "A", run, rows)
    payload = json.loads(path.read_text())
    assert path.name == "arm_provenance.json"
    assert payload["contract"] == "sequence_track_arm_provenance_v2"
    assert payload["arm"] == "A"
    assert payload["run"] == run
    assert [row["cricsheet_id"] for row in payload["fixtures"]] == [
        "1477609", "1501896"]


# --------------------------------------------------------------------------
# D6 6.1/6.3 — the as-of stamp comes from the replay lifecycle itself
# --------------------------------------------------------------------------

class _FakeProvider:
    """Minimal stand-in for SameDayReplayStatsProvider's lifecycle."""

    def __init__(self):
        self.calls = []
        self.matches_advanced = 0

    def begin_date(self, date_str, documents):
        self.calls.append(("begin_date", date_str))

    def begin_match(self, match_id, document, *, prediction_required):
        self.calls.append(("begin_match", match_id, prediction_required))

    def lock_prediction(self, match_id):
        self.calls.append(("lock", match_id))

    def advance_match(self, match_id, document):
        self.calls.append(("advance", match_id))
        self.matches_advanced += 1
        return {"match_id": match_id}


def test_lifecycle_recorder_stamps_the_prediction_lock():
    inner = _FakeProvider()
    recorder = _LifecycleRecorder(inner)

    recorder.begin_date("2025-10-31", [{}, {}, {}])
    # A context fixture advances first.
    recorder.begin_match("ctx", {}, prediction_required=False)
    recorder.advance_match("ctx", {})
    # Then an evaluated fixture locks before its own replay.
    recorder.begin_match("1478908", {}, prediction_required=True)
    recorder.lock_prediction("1478908")
    recorder.advance_match("1478908", {})
    # A same-day sibling now sees both predecessors.
    recorder.begin_match("1501896", {}, prediction_required=True)
    recorder.lock_prediction("1501896")
    recorder.advance_match("1501896", {})

    assert recorder.stamps["1478908"] == {
        "date": "2025-10-31",
        "same_day_advanced_before": ["ctx"],
        "matches_advanced": 1,
        "odds_row_found": True,
    }
    assert recorder.stamps["1501896"] == {
        "date": "2025-10-31",
        "same_day_advanced_before": ["ctx", "1478908"],
        "matches_advanced": 2,
        "odds_row_found": True,
    }
    assert recorder.stamps["ctx"]["odds_row_found"] is False
    # Every lifecycle call reached the real provider, in order.
    assert inner.calls == [
        ("begin_date", "2025-10-31"),
        ("begin_match", "ctx", False), ("advance", "ctx"),
        ("begin_match", "1478908", True), ("lock", "1478908"),
        ("advance", "1478908"),
        ("begin_match", "1501896", True), ("lock", "1501896"),
        ("advance", "1501896"),
    ]


def test_lifecycle_recorder_resets_predecessors_on_a_new_date():
    inner = _FakeProvider()
    recorder = _LifecycleRecorder(inner)
    recorder.begin_date("2025-10-31", [{}])
    recorder.begin_match("a", {}, prediction_required=True)
    recorder.lock_prediction("a")
    recorder.advance_match("a", {})
    recorder.begin_date("2025-11-01", [{}])
    recorder.begin_match("b", {}, prediction_required=True)
    recorder.lock_prediction("b")

    assert recorder.stamps["b"]["same_day_advanced_before"] == []
    # `matches_advanced` is the run-wide counter, so it keeps climbing.
    assert recorder.stamps["b"]["matches_advanced"] == 1
    assert recorder.matches_advanced == 1


# --------------------------------------------------------------------------
# Astra round 1, item 5 — seed intervals must not overlap
# --------------------------------------------------------------------------

def _colliding_ids():
    """Two synthetic ids whose seeds are closest, and that distance.

    20k ids over a 2^31 space put the expected closest pair a handful of
    seeds apart, so the collision is CONSTRUCTED, not hypothetical: it is
    the same birthday arithmetic that makes the probability grow with
    n_sims.
    """
    base_seed = 20260910
    ids = [f"synthetic_{index}" for index in range(20000)]
    ordered = sorted((fixture_seed(i, base_seed), i) for i in ids)
    gaps = [(b[0] - a[0], a[1], b[1])
            for a, b in zip(ordered, ordered[1:])]
    gap, id_a, id_b = min(gaps)
    return base_seed, ids, gap, id_a, id_b


def test_seed_intervals_disjoint_finds_a_constructed_collision():
    base_seed, ids, gap, id_a, id_b = _colliding_ids()
    assert gap < 100, f"expected a close pair among 20k ids, got {gap}"

    # n_sims exactly the gap: the intervals touch but do not overlap.
    assert not any(row[0] in (id_a, id_b)
                   for row in seed_intervals_disjoint(ids, base_seed, gap))
    # One more simulation and they share a seed.
    overlaps = seed_intervals_disjoint(ids, base_seed, gap + 1)
    assert (id_a, id_b) in {(row[0], row[2]) for row in overlaps}
    assert min_seed_gap(ids, base_seed) == gap


def test_seed_intervals_disjoint_grows_with_n_sims():
    ids = [f"synthetic_{index}" for index in range(2000)]
    counts = [len(seed_intervals_disjoint(ids, 20260910, n))
              for n in (100, 10_000, 1_000_000)]
    assert counts == sorted(counts)
    assert counts[-1] > counts[0]


def test_seed_intervals_are_disjoint_on_a_small_realistic_set():
    ids = ["1477609", "1478908", "1501896", "1505127"]
    assert seed_intervals_disjoint(ids, 20260910, 1600) == []
    assert min_seed_gap(ids, 20260910) > 1600
    assert min_seed_gap(["only_one"], 1) is None


def test_preflight_refuses_an_overlapping_pair_and_names_it():
    base_seed, ids, gap, id_a, id_b = _colliding_ids()
    assert_seed_intervals_disjoint(ids, base_seed, gap)  # no raise
    with pytest.raises(SystemExit) as excinfo:
        assert_seed_intervals_disjoint(ids, base_seed, gap + 1)
    message = str(excinfo.value)
    assert id_a in message and id_b in message
    assert "random_seed + i" in message
    assert "Register a different base seed" in message


# --------------------------------------------------------------------------
# Astra round 1, item 4 — the registered config is enforced at launch
# --------------------------------------------------------------------------

def _config(tmp_path):
    return {
        "arms": {
            "A": {
                "model_dir": "models/xgb_i7_noweights_production",
                "stats_version": "i7",
                "extras_graft": "models/auto/b18/extras_graft_v1.json",
                "bowler_usage": "models/bowler_phase_usage.json",
                "roster_policy": "models/bowler_roster_policy.json",
                "context_dir": "data/t20s_json",
                "player_metadata": "data/all_players_enriched.csv",
                "odds": {"path": "betting_odds_polymarket_v2.json"},
                "clip": [0.01, 0.99],
                "base_seed": 20260910,
                "threads": 4,
                "device": "cpu",
                "smoke_1a": {"fixture_dir": str(tmp_path / "smoke"),
                             "n_sims": 10, "base_seeds": [20260910]},
                "full_run": {"fixture_dir": str(tmp_path / "full"),
                             "n_sims": "to_be_filled_before_1d",
                             "base_seeds": [20260910]},
                "timing_1b": {
                    "fixture_dir": str(tmp_path / "timing"),
                    # The PERMITTED list is the jointly screened one; 6400
                    # is screened and refused (Astra round 3, item 2).
                    "n_sims": [100, 200, 400, 800, 1600, 3200],
                    "candidates_screened": [100, 200, 400, 800, 1600,
                                            3200, 6400],
                    "base_seeds": [20260910, 20260911, 20260912]},
            },
        },
    }


def _effective(tmp_path, **overrides):
    values = {
        "model_dir": "models/xgb_i7_noweights_production",
        "stats_version": "i7",
        "extras_graft": "models/auto/b18/extras_graft_v1.json",
        "bowler_usage_path": "models/bowler_phase_usage.json",
        "roster_policy_path": "models/bowler_roster_policy.json",
        "context_dir": "data/t20s_json",
        "player_metadata": "data/all_players_enriched.csv",
        "odds": "betting_odds_polymarket_v2.json",
        "clip": [0.01, 0.99],
        "base_seed": 20260910,
        "n_sims": 10,
        "threads": 4,
        "device": "cpu",
        "fixture_dir": str(tmp_path / "smoke"),
    }
    values.update(overrides)
    return values


def test_registered_mismatches_accepts_the_registered_invocation(tmp_path):
    effective = _effective(tmp_path)
    assert registered_mismatches(_config(tmp_path), "A", effective) == []
    assert effective["config_block"] == "smoke_1a"

    # The full block carries the n_sims placeholder until the 1b read, so it
    # is registered but NOT launchable, and says so.
    full = _effective(tmp_path, fixture_dir=str(tmp_path / "full"))
    problems = registered_mismatches(_config(tmp_path), "A", full)
    assert full["config_block"] == "full_run"
    assert any("not launchable yet" in problem for problem in problems)


@pytest.mark.parametrize("field,value,needle", [
    ("model_dir", "models/somewhere_else", "model dir"),
    ("stats_version", "v3", "stats version"),
    ("extras_graft", "models/auto/b18/other.json", "extras graft"),
    ("bowler_usage_path", "models/other_usage.json", "bowler usage"),
    ("roster_policy_path", None, "roster policy"),
    ("context_dir", "data/other_json", "context dir"),
    ("odds", "betting_odds_polymarket.json", "odds"),
    ("clip", [0.05, 0.95], "clip bounds"),
    ("base_seed", 1, "base seed"),
    ("threads", 8, "threads"),
    ("player_metadata", "data/other.csv", "player metadata"),
])
def test_registered_mismatches_refuses_every_drifted_value(
        tmp_path, field, value, needle):
    problems = registered_mismatches(
        _config(tmp_path), "A", _effective(tmp_path, **{field: value}))
    assert any(needle in problem for problem in problems), problems


def test_registered_mismatches_refuses_an_unregistered_fixture_dir(tmp_path):
    problems = registered_mismatches(
        _config(tmp_path), "A",
        _effective(tmp_path, fixture_dir=str(tmp_path / "elsewhere")))
    assert any("fixture dir" in problem for problem in problems)


def test_registered_mismatches_refuses_an_unknown_arm(tmp_path):
    problems = registered_mismatches(
        _config(tmp_path), "B", _effective(tmp_path))
    assert problems and "arm" in problems[0]


def test_registered_mismatches_refuses_clipping_left_off(tmp_path):
    problems = registered_mismatches(
        _config(tmp_path), "A", _effective(tmp_path, clip=None))
    assert any("clip bounds" in problem for problem in problems)


# --------------------------------------------------------------------------
# Astra round 2, item 4a — n_sims and the RESOLVED device are checked
# --------------------------------------------------------------------------

def test_a_drifted_n_sims_is_refused(tmp_path):
    """Astra ran the registered smoke invocation at n_sims 999."""
    problems = registered_mismatches(
        _config(tmp_path), "A", _effective(tmp_path, n_sims=999))
    assert any("n_sims" in problem and "permits" in problem
               for problem in problems), problems
    assert registered_mismatches(
        _config(tmp_path), "A", _effective(tmp_path, n_sims=10)) == []


def test_a_non_cpu_device_is_refused(tmp_path):
    """Astra ran the registered smoke invocation on device mps."""
    problems = registered_mismatches(
        _config(tmp_path), "A", _effective(tmp_path, device="mps"))
    assert any("device" in problem and "cpu" in problem
               for problem in problems), problems


def test_resolve_device_prefers_the_flag_then_the_env(monkeypatch):
    monkeypatch.delenv("T1_SIM_DEVICE", raising=False)
    assert _resolve_device("transformer", None) == "cpu"
    assert _resolve_device("transformer", "cpu") == "cpu"

    monkeypatch.setenv("T1_SIM_DEVICE", "mps")
    # An inherited env used to reach the wrapper unchecked; it is now the
    # RESOLVED value, so --config compares it against the pinned cpu.
    assert _resolve_device("transformer", None) == "mps"
    assert _resolve_device("transformer", "cpu") == "cpu"


def test_resolve_device_refuses_a_non_cpu_xgboost_arm(monkeypatch):
    monkeypatch.delenv("T1_SIM_DEVICE", raising=False)
    assert _resolve_device("xgboost", None) == "cpu"
    assert _resolve_device("xgboost", "auto") == "cpu"
    with pytest.raises(SystemExit):
        _resolve_device("xgboost", "mps")
    monkeypatch.setenv("T1_SIM_DEVICE", "cuda")
    with pytest.raises(SystemExit):
        _resolve_device("xgboost", None)


# --------------------------------------------------------------------------
# Astra round 2, item 4c — the registered 1b batches pass the preflight
# --------------------------------------------------------------------------

def test_a_registered_batch_seed_on_the_timing_shard_is_accepted(tmp_path):
    effective = _effective(tmp_path, fixture_dir=str(tmp_path / "timing"),
                           base_seed=20260911, n_sims=200)
    assert registered_mismatches(_config(tmp_path), "A", effective) == []
    assert effective["config_block"] == "timing_1b"
    # The whole batch cohort shares one seed line, so the seed screen must
    # cover all three, not just this launch's seed.
    assert effective["seed_cohort"] == [20260910, 20260911, 20260912]


def test_an_unregistered_seed_on_the_timing_shard_is_refused(tmp_path):
    effective = _effective(tmp_path, fixture_dir=str(tmp_path / "timing"),
                           base_seed=20261231, n_sims=200)
    problems = registered_mismatches(_config(tmp_path), "A", effective)
    assert any("base seed" in problem and "permits" in problem
               for problem in problems), problems


def test_an_unregistered_candidate_on_the_timing_shard_is_refused(tmp_path):
    effective = _effective(tmp_path, fixture_dir=str(tmp_path / "timing"),
                           base_seed=20260911, n_sims=250)
    problems = registered_mismatches(_config(tmp_path), "A", effective)
    assert any("n_sims" in problem and "permits" in problem
               for problem in problems), problems


def test_the_smoke_block_still_pins_one_seed(tmp_path):
    effective = _effective(tmp_path, base_seed=20260911)
    problems = registered_mismatches(_config(tmp_path), "A", effective)
    assert any("base seed" in problem for problem in problems)
    assert effective["seed_cohort"] == [20260911]


# --------------------------------------------------------------------------
# Astra round 2, item 5 — the joint (cross-batch) seed screen
# --------------------------------------------------------------------------

def test_joint_screen_sees_a_collision_each_base_seed_alone_misses():
    """A constructed cross-seed collision: both singles pass, the union
    does not. This is the shape of the defect Astra found on the real
    fixture set (joint gap 3290 while every single-seed gap exceeded
    8000)."""
    ids = [f"synthetic_{index}" for index in range(400)]
    seeds = [20260910, 20260911, 20260912]
    single_gap = min(min_seed_gap(ids, seed) for seed in seeds)
    joint_gap = min_seed_gap(ids, seeds)
    assert joint_gap < single_gap, (joint_gap, single_gap)

    n_sims = joint_gap + 1
    for seed in seeds:
        assert seed_intervals_disjoint(ids, seed, n_sims) == []
    overlaps = seed_intervals_disjoint(ids, seeds, n_sims)
    assert overlaps
    # The joint labels carry the base seed, so the operator can see which
    # two BATCHES collide.
    assert all("@" in row[0] and "@" in row[2] for row in overlaps)
    assert {row[0].split("@")[1] for row in overlaps} <= {
        str(seed) for seed in seeds}


def test_joint_preflight_refuses_a_cross_batch_collision():
    ids = [f"synthetic_{index}" for index in range(400)]
    seeds = [20260910, 20260911, 20260912]
    joint_gap = min_seed_gap(ids, seeds)
    assert_seed_intervals_disjoint(ids, seeds, joint_gap)  # no raise
    with pytest.raises(SystemExit) as excinfo:
        assert_seed_intervals_disjoint(ids, seeds, joint_gap + 1)
    message = str(excinfo.value)
    assert "seed cohort" in message
    assert "@" in message


def test_largest_permitted_n_sims_walks_the_candidate_ladder():
    ids = [f"synthetic_{index}" for index in range(400)]
    seeds = [20260910, 20260911, 20260912]
    gap = min_seed_gap(ids, seeds)
    candidates = [100, 200, 400, 800, 1600, 3200, 6400]
    permitted = largest_permitted_n_sims(ids, seeds, candidates)
    assert permitted is None or permitted <= gap
    assert all(
        seed_intervals_disjoint(ids, seeds, candidate) == []
        for candidate in candidates if candidate <= (permitted or 0))
    # Anything above the gap collides, by construction.
    for candidate in candidates:
        if candidate > gap:
            assert seed_intervals_disjoint(ids, seeds, candidate)


# --------------------------------------------------------------------------
# Astra round 3, item 2 — the registered ceiling, not just the launched shard
# --------------------------------------------------------------------------

def test_a_jointly_forbidden_candidate_is_refused_by_the_preflight(tmp_path):
    """6400 is screened but not permitted: the ten-fixture shard alone has
    no collision at that count, so only the REGISTERED list catches it."""
    config = _config(tmp_path)
    permitted = config["arms"]["A"]["timing_1b"]["n_sims"]
    assert 6400 not in permitted and 3200 in permitted

    effective = _effective(tmp_path, fixture_dir=str(tmp_path / "timing"),
                           base_seed=20260911, n_sims=6400)
    problems = registered_mismatches(config, "A", effective)
    assert any("n_sims" in problem and "permits" in problem
               for problem in problems), problems

    # The shard on its own is innocent — which is the point.
    shard = [f"150000{index}" for index in range(10)]
    assert seed_intervals_disjoint(shard, 20260911, 6400) == []

    ok = _effective(tmp_path, fixture_dir=str(tmp_path / "timing"),
                    base_seed=20260911, n_sims=3200)
    assert registered_mismatches(config, "A", ok) == []
