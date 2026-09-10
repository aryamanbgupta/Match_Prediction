"""Tests for the stage-1 config pin script (D5 check 5.1, 5.3, 5.4).

Four things have to hold for the registered config to mean anything:

* `--verify` passes on the committed file, so the pinned hashes are the
  hashes of the artifacts actually on disk;
* a tampered copy fails and names what moved, so the file cannot drift
  from the artifacts (or be hand-edited) unnoticed;
* the checkpoint rule really is "lowest validation LL, ties to the lowest
  seed", exercised on a synthetic summary where the answer is known;
* the seed formula the config documents is the one `run_arm.py` runs.

Only the first two touch repository artifacts; they carry the
`needs_artifacts` marker so `-m "not needs_artifacts"` stays green.
"""
from __future__ import annotations

import shutil

import pytest
import yaml

from sequence_track import pin_stage1
from sequence_track.pin_stage1 import (
    CONFIG_PATH,
    PinError,
    choose_seed,
    diff_config,
    fixture_seed,
    flatten,
    main,
    selection_table,
    sub_seed,
    verify_config,
)
from sequence_track import run_arm
from sequence_track.run_arm import fixture_seed as run_arm_fixture_seed
from sequence_track.run_arm import sub_seed as run_arm_sub_seed


# ---------------------------------------------------------------------------
# 5.1 — verify passes on the committed file, fails on a tampered copy
# ---------------------------------------------------------------------------

@pytest.mark.needs_artifacts
def test_verify_passes_on_the_committed_config():
    assert CONFIG_PATH.exists(), (
        f"{CONFIG_PATH} is missing; run pin_stage1.py --write")
    assert verify_config(CONFIG_PATH) == []
    assert main(["--verify"]) == 0


@pytest.mark.needs_artifacts
def test_tampered_hash_fails_and_names_the_key(tmp_path):
    tampered = tmp_path / "seq_stage1_sim_v1.yaml"
    shutil.copyfile(CONFIG_PATH, tampered)
    payload = yaml.safe_load(tampered.read_text())
    payload["arms"]["A"]["checkpoint_md5"] = "0" * 32
    tampered.write_text(yaml.safe_dump(payload, sort_keys=False))

    problems = verify_config(tampered)
    assert problems, "a changed checkpoint md5 must not verify"
    assert any("arms.A.checkpoint_md5" in problem for problem in problems)
    assert main(["--verify", "--config", str(tampered)]) == 1


@pytest.mark.needs_artifacts
def test_tampered_prose_fails(tmp_path):
    """Prose lives in the script, so a hand-edited YAML line is a mismatch."""
    tampered = tmp_path / "seq_stage1_sim_v1.yaml"
    shutil.copyfile(CONFIG_PATH, tampered)
    payload = yaml.safe_load(tampered.read_text())
    payload["decision_rule"]["equivalence_margin_ll"] = 0.02
    tampered.write_text(yaml.safe_dump(payload, sort_keys=False))

    problems = verify_config(tampered)
    assert any("decision_rule.equivalence_margin_ll" in problem
               for problem in problems)


@pytest.mark.needs_artifacts
def test_dropped_key_fails(tmp_path):
    tampered = tmp_path / "seq_stage1_sim_v1.yaml"
    shutil.copyfile(CONFIG_PATH, tampered)
    payload = yaml.safe_load(tampered.read_text())
    del payload["arms"]["C114"]
    tampered.write_text(yaml.safe_dump(payload, sort_keys=False))

    problems = verify_config(tampered)
    assert any(problem.startswith("MISSING") and "C114" in problem
               for problem in problems)


def test_verify_reports_a_missing_file(tmp_path):
    assert verify_config(tmp_path / "absent.yaml") == [
        f"MISSING  file: {tmp_path / 'absent.yaml'}"]


def test_not_verified_keys_are_ignored_by_the_diff():
    """A moved timestamp or a new HEAD must not fail --verify."""
    expected = {
        "provenance": {"pins_generated_at": "2026-09-10T00:00:00Z",
                       "git_head_short": "aaaaaaa",
                       "source_md5": {"x": "1"}},
    }
    actual = {
        "provenance": {"pins_generated_at": "2027-01-01T00:00:00Z",
                       "git_head_short": "bbbbbbb",
                       "source_md5": {"x": "1"}},
    }
    assert diff_config(expected, actual) == []

    actual["provenance"]["source_md5"]["x"] = "2"
    assert any("provenance.source_md5.x" in problem
               for problem in diff_config(expected, actual))


# ---------------------------------------------------------------------------
# 5.3 — checkpoint selection rule on a synthetic summary
# ---------------------------------------------------------------------------

def _synthetic_summary(path, mlp_rows, full_rows):
    payload = {
        "splits": {
            "validation": {
                "arms": {
                    "mlp": {"per_seed": [
                        {"seed": seed, "ll": ll} for seed, ll in mlp_rows]},
                    "full": {"per_seed": [
                        {"seed": seed, "ll": ll} for seed, ll in full_rows]},
                },
            },
            # Deliberately better on a DIFFERENT seed: the rule must not
            # look at the test split.
            "test": {
                "arms": {
                    "mlp": {"per_seed": [
                        {"seed": seed, "ll": -ll} for seed, ll in mlp_rows]},
                    "full": {"per_seed": [
                        {"seed": seed, "ll": -ll} for seed, ll in full_rows]},
                },
            },
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def test_choose_seed_picks_the_lowest_validation_ll():
    rows = [{"seed": 7, "ll": 1.5}, {"seed": 13, "ll": 1.2},
            {"seed": 29, "ll": 1.4}]
    assert choose_seed(rows) == (13, 1.2)


def test_choose_seed_breaks_ties_to_the_lowest_seed():
    rows = [{"seed": 101, "ll": 1.2}, {"seed": 13, "ll": 1.2},
            {"seed": 42, "ll": 1.9}]
    assert choose_seed(rows) == (13, 1.2)


def test_choose_seed_rejects_an_empty_table():
    with pytest.raises(PinError):
        choose_seed([])


def test_selection_table_uses_validation_only(tmp_path):
    summary = _synthetic_summary(
        tmp_path / "summary.yaml",
        mlp_rows=[(7, 1.30), (13, 1.10), (29, 1.10), (42, 1.40)],
        full_rows=[(7, 1.05), (13, 1.20), (29, 1.30), (42, 1.40)],
    )
    table = selection_table(summary)

    # mlp: 13 and 29 tie at the minimum -> the lower seed wins.
    assert table["mlp"]["chosen_seed"] == 13
    assert table["mlp"]["chosen_validation_ll"] == pytest.approx(1.10)
    # full: a clear minimum at seed 7. The test split's argmin is seed 42.
    assert table["full"]["chosen_seed"] == 7
    assert table["full"]["chosen_validation_ll"] == pytest.approx(1.05)
    # The printed table is seed-ordered and carries every seed.
    assert [row["seed"] for row in table["mlp"]["per_seed"]] == [7, 13, 29, 42]


def test_selection_table_fails_closed_on_a_missing_arm(tmp_path):
    path = tmp_path / "summary.yaml"
    path.write_text(yaml.safe_dump(
        {"splits": {"validation": {"arms": {"mlp": {"per_seed": [
            {"seed": 7, "ll": 1.0}]}}}}}))
    with pytest.raises(PinError):
        selection_table(path)


# ---------------------------------------------------------------------------
# 5.4 — the documented seed formula is run_arm's function
# ---------------------------------------------------------------------------

CRICSHEET_IDS = ["1477609", "1478908", "1501896", "1505127", "1", "999999999"]


@pytest.mark.parametrize("cricsheet_id", CRICSHEET_IDS)
@pytest.mark.parametrize("base_seed", [pin_stage1.BASE_SEED,
                                       pin_stage1.SECOND_BASE_SEED,
                                       pin_stage1.THIRD_BATCH_BASE_SEED, 0])
def test_fixture_seed_matches_run_arm(cricsheet_id, base_seed):
    mine = fixture_seed(cricsheet_id, base_seed)
    theirs = run_arm_fixture_seed(cricsheet_id, base_seed)
    assert mine == theirs
    assert 0 <= mine <= (1 << 31) - 1


@pytest.mark.parametrize("cricsheet_id", CRICSHEET_IDS)
def test_sub_seeds_match_run_arm(cricsheet_id):
    seed = fixture_seed(cricsheet_id, pin_stage1.BASE_SEED)
    for stream in pin_stage1.SUB_SEED_STREAMS:
        assert sub_seed(seed, stream) == run_arm_sub_seed(seed, stream)
    assert tuple(pin_stage1.SUB_SEED_STREAMS) == tuple(run_arm.SUB_SEED_STREAMS)


def test_sub_seeds_differ_from_each_other_and_from_the_fixture_seed():
    seed = fixture_seed("1477609", pin_stage1.BASE_SEED)
    values = {stream: sub_seed(seed, stream)
              for stream in pin_stage1.SUB_SEED_STREAMS}
    assert len(set(values.values())) == len(values)
    assert seed not in set(values.values())


# ---------------------------------------------------------------------------
# 5.11 — no sealed path in the committed file
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 5.8 — every command line carries every flag the protocol depends on
# ---------------------------------------------------------------------------

REQUIRED_FLAGS = [
    # Every registered command re-verifies the pins at launch: run_arm
    # refuses to start if a pinned hash moved or if the invocation differs
    # from the arm block (Astra round 1, item 4).
    "--config experiments/configs/seq_stage1_sim_v1.yaml",
    "--bowler-roster-policy models/bowler_roster_policy.json",
    "--bowler-usage-path models/bowler_phase_usage.json",
    "--extras-graft models/auto/b18/extras_graft_v1.json",
    "--clip-low 0.01 --clip-high 0.99",
    "--context-dir data/t20s_json",
    "--odds betting_odds_polymarket_v2.json",
    "--player-metadata data/all_players_enriched.csv",
    "--base-seed 20260910",
    "--threads 4",
]

ACTIVE_ARMS = ("A", "A50", "B", "C")


@pytest.fixture(scope="module")
def committed():
    return yaml.safe_load(CONFIG_PATH.read_text())


@pytest.mark.parametrize("arm", ACTIVE_ARMS)
@pytest.mark.parametrize("run", ["full_run", "smoke_1a"])
def test_every_command_carries_every_required_flag(committed, arm, run):
    command = committed["arms"][arm][run]["command"]
    for flag in REQUIRED_FLAGS:
        assert flag in command, f"{arm}/{run} is missing {flag}"
    assert f"--arm {arm} " in command
    assert f"--model-dir {committed['arms'][arm]['model_dir']} " in command


@pytest.mark.parametrize("arm", ACTIVE_ARMS)
def test_smoke_differs_from_full_only_in_three_flags(committed, arm):
    block = committed["arms"][arm]
    full = block["full_run"]["command"].split()
    smoke = block["smoke_1a"]["command"].split()
    assert len(full) == len(smoke)
    differing = {full[index - 1] for index in range(1, len(full))
                 if full[index] != smoke[index]}
    assert differing <= {"--fixture-dir", "--n-sims", "--output-dir"}
    assert differing == {"--fixture-dir", "--n-sims", "--output-dir"}
    assert "--n-sims 10 " in block["smoke_1a"]["command"] + " "
    assert block["smoke_1a"]["fixture_dir"] == (
        "models/embeddings/seq_stage1/smoke/fixtures")
    assert block["smoke_1a"]["output_dir"] == (
        f"models/embeddings/seq_stage1/smoke/{arm}")
    assert block["full_run"]["output_dir"] == (
        f"models/embeddings/seq_stage1/full/{arm}")


def test_chosen_n_sims_drives_the_full_run_commands(committed):
    chosen = committed["convergence_protocol"]["chosen_n_sims"]
    token = (pin_stage1.N_SIMS_TOKEN if chosen == pin_stage1.CONVERGENCE_PLACEHOLDER
             else str(chosen))
    for arm in ACTIVE_ARMS:
        block = committed["arms"][arm]["full_run"]
        assert block["n_sims"] == chosen
        assert f"--n-sims {token} " in block["command"] + " "


def test_selector_pin_is_arm_independent(committed):
    selector = committed["selector"]
    assert selector["class"] == "RosterEmpiricalBowlerSelector"
    for arm in ACTIVE_ARMS:
        block = committed["arms"][arm]
        assert block["bowler_selector"] == selector["class"]
        assert block["roster_policy_md5"] == selector["roster_policy_md5"]
        assert block["bowler_usage_md5"] == selector["bowler_usage_md5"]
        assert block["b10_corpus_md5"] == selector["b10_corpus_md5"]


def test_every_recorded_role_resolves_to_the_pinned_path(committed):
    """Each `<thing>_role` key names the manifest role that owns `<thing>`."""
    from artifacts import artifact_path

    flat = flatten(committed)
    checked = 0
    for key, role in sorted(flat.items()):
        if not isinstance(role, str):
            continue
        # `odds.role` / `fixture_set.role` name the sibling `path`; every
        # other `<thing>_role` names the sibling `<thing>`.
        if key.endswith(".role"):
            target = key[: -len(".role")] + ".path"
        elif key.endswith("_role"):
            target = key[: -len("_role")]
        else:
            continue
        pinned = flat.get(target)
        if pinned is None:  # e.g. scoring.winner_log_loss.odds_role
            continue
        assert pinned == artifact_path(role).as_posix(), (
            f"{key} = {role!r} does not resolve to {target} = {pinned!r}")
        checked += 1
    assert checked >= 40, f"only {checked} role/path pairs were checked"


def test_role_keys_cover_every_manifest_owned_artifact(committed):
    """No manifest-owned path is pinned without naming its role."""
    from artifacts import load_manifest

    by_path = {row["path"]: role for role, row in load_manifest().items()}
    flat = flatten(committed)
    for key, value in flat.items():
        if not isinstance(value, str) or value not in by_path:
            continue
        if key.endswith(("_role", "_note", "_source", "_md5", "_sha256")):
            continue
        sibling = f"{key}_role" if not key.endswith(".path") else (
            key.rsplit(".", 1)[0] + ".role")
        if key.endswith(".fixture_dir"):
            sibling = key + "_role"
        assert flat.get(sibling) == by_path[value], (
            f"{key} pins manifest-owned {value!r} but {sibling} is "
            f"{flat.get(sibling)!r}, expected {by_path[value]!r}")


def test_c114_is_deferred_with_no_artifacts(committed):
    c114 = committed["arms"]["C114"]
    assert c114["status"] == "deferred"
    assert c114["artifacts"] is None
    assert c114["reason_deferred"]
    assert "command" not in c114 and "checkpoint" not in c114


# ---------------------------------------------------------------------------
# Provenance: the source closure and the runtime are pinned facts
# ---------------------------------------------------------------------------

def test_source_closure_is_hashed_in_full(committed):
    """Every closure file on this checkout is hashed, with a live md5."""
    from artifacts import md5_file

    provenance = committed["provenance"]
    pinned = provenance["source_md5"]
    absent = provenance["source_md5_absent"]

    assert set(pinned) | set(absent) == set(
        pin_stage1.PROVENANCE_SOURCE_CLOSURE)
    assert provenance["source_count"] == len(pinned)
    assert absent == [], f"closure files missing from the checkout: {absent}"
    for source, recorded in pinned.items():
        path = pin_stage1.REPO_ROOT / source
        assert path.is_file(), source
        assert md5_file(path) == recorded, f"{source} md5 is stale"


def test_source_closure_reaches_past_the_runner_and_the_engine(committed):
    """The Astra finding: five files were not the execution path."""
    pinned = set(committed["provenance"]["source_md5"])
    for required in (
        "scripts/sequence_track/run_arm.py",       # the runner
        "scripts/sequence_track/pin_stage1.py",    # this pin script
        "scripts/sequence_track/audit_cross_arm.py",
        "scripts/sequence_track/score_realism.py",
        "scripts/sequence_track/score_props.py",
        "scripts/sim_eval/run_sim_eval.py",        # the delegated runner
        "scripts/sim_eval/same_day_stats.py",      # replay
        "scripts/tracker_rehydration.py",          # rehydration
        "scripts/loaders_common.py",               # same-day chronology
        "scripts/stats_sqlite_backend.py",         # the cache contract
        "scripts/feature_registry.py",             # the feature contract
        "scripts/sim_eval/eval_statistics.py",     # the block bootstrap
        "scripts/sim_eval/market_math.py",         # cost scenarios
        "scripts/sim_eval/prop_fair_baselines.py",  # the prop reducer
    ):
        assert required in pinned, f"{required} is not pinned"
    assert len(pinned) >= 25


def test_runtime_matches_this_environment(committed):
    import platform
    from importlib import metadata

    runtime = committed["provenance"]["runtime"]
    assert runtime["verified"] is True
    assert runtime["python"] == platform.python_version()
    for name, recorded in runtime["packages"].items():
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = None
        assert recorded == installed, f"{name}: {recorded!r} vs {installed!r}"


@pytest.mark.needs_artifacts
def test_tampered_source_md5_fails(tmp_path):
    tampered = tmp_path / "seq_stage1_sim_v1.yaml"
    shutil.copyfile(CONFIG_PATH, tampered)
    payload = yaml.safe_load(tampered.read_text())
    payload["provenance"]["source_md5"]["scripts/sim_v1_2.py"] = "0" * 32
    tampered.write_text(yaml.safe_dump(payload, sort_keys=False))
    assert any("provenance.source_md5.scripts/sim_v1_2.py" in problem
               for problem in verify_config(tampered))


@pytest.mark.needs_artifacts
def test_tampered_runtime_fails(tmp_path):
    tampered = tmp_path / "seq_stage1_sim_v1.yaml"
    shutil.copyfile(CONFIG_PATH, tampered)
    payload = yaml.safe_load(tampered.read_text())
    payload["provenance"]["runtime"]["packages"]["torch"] = "0.0.0"
    tampered.write_text(yaml.safe_dump(payload, sort_keys=False))
    assert any("provenance.runtime.packages.torch" in problem
               for problem in verify_config(tampered))


def test_flatten_records_empty_containers():
    """An empty list or mapping must still be a comparable leaf."""
    assert flatten({"a": []}) == {"a": "<empty list>"}
    assert flatten({"a": {}}) == {"a": "<empty mapping>"}
    # A placeholder that quietly gains a row, or quietly changes shape, is a
    # difference either way round.
    assert diff_config({"a": []}, {"a": {}})
    assert diff_config({"a": []}, {"a": [1]})
    assert diff_config({"a": [1]}, {"a": []})


def test_committed_config_names_no_sealed_path():
    payload = yaml.safe_load(CONFIG_PATH.read_text())
    declared = payload["experiment"]["forbidden_data"]
    assert declared == ["data/golden", "data/forward_holdout"]
    payload["experiment"].pop("forbidden_data")
    text = yaml.safe_dump(payload)
    for forbidden in declared:
        assert forbidden not in text


# ---------------------------------------------------------------------------
# 5.9 — the seed-interval overlap screen (Astra round 1, item 5)
# ---------------------------------------------------------------------------

def test_overlap_check_is_recorded_for_every_registered_base_seed(committed):
    block = committed["seeds"]["overlap_check"]
    assert block["candidates"] == pin_stage1.CONVERGENCE_CANDIDATES
    for base_seed in pin_stage1.BATCH_BASE_SEEDS:
        rows = block["by_base_seed"][str(base_seed)]
        assert [row["n_sims"] for row in rows] == block["candidates"]
        for row in rows:
            assert row["disjoint"] == (row["min_seed_gap"] >= row["n_sims"])
            assert (row["overlapping_pairs"] == 0) is row["disjoint"]


def test_the_joint_screen_is_the_one_that_governs_a_candidate(committed):
    """Astra round 2, item 5: the three batches share one seed line."""
    block = committed["seeds"]["overlap_check"]
    joint = block["joint"]
    assert joint["base_seeds"] == list(pin_stage1.BATCH_BASE_SEEDS)

    # The joint gap is STRICTLY tighter than every single-seed gap: this is
    # exactly why screening each base seed alone was not enough.
    singles = [block["by_base_seed"][str(seed)][0]["min_seed_gap"]
               for seed in pin_stage1.BATCH_BASE_SEEDS]
    assert joint["joint_min_seed_gap"] < min(singles)

    permitted = joint["largest_permitted_candidate"]
    assert permitted is not None and permitted <= joint["joint_min_seed_gap"]
    for row in joint["rows"]:
        expected = row["n_sims"] <= joint["joint_min_seed_gap"]
        assert row["disjoint"] is expected, row
        assert row["permitted"] == (
            "yes" if expected else "not_permitted_without_new_seeds")
    # The ladder must run past the wall, so the config SHOWS where it is.
    assert any(row["permitted"] == "not_permitted_without_new_seeds"
               for row in joint["rows"])


def test_overlap_screen_cli_refuses_a_jointly_forbidden_candidate():
    forbidden = [n for n in pin_stage1.CONVERGENCE_CANDIDATES
                 if n > 3290]
    assert forbidden, "the candidate ladder no longer crosses the wall"
    assert main(["--check-seed-overlap", "--n-sims", str(forbidden[0])]) == 1


def test_timing_and_variability_blocks_are_registered(committed):
    """Astra round 2, item 4c: the 1b batches must be launchable."""
    for arm in ACTIVE_ARMS:
        block = committed["arms"][arm]
        timing = block["timing_1b"]
        assert timing["fixture_dir"] == (
            "models/embeddings/seq_stage1/timing/fixtures")
        assert timing["base_seeds"] == list(pin_stage1.BATCH_BASE_SEEDS)
        # The permitted list is the jointly screened one; the full ladder
        # is recorded separately as what was screened (Astra round 3, #2).
        assert timing["candidates_screened"] == (
            pin_stage1.CONVERGENCE_CANDIDATES)
        assert timing["n_sims"] == [
            row["n_sims"] for row
            in committed["seeds"]["overlap_check"]["joint"]["rows"]
            if row["disjoint"]]
        assert timing["n_sims"] != timing["candidates_screened"], (
            "the ladder must cross the joint wall, or the ceiling is "
            "untested")
        assert timing["fixture_count_expected"] == 10
        assert timing["output_dir_template"] == (
            f"models/embeddings/seq_stage1/timing/{arm}"
            "/seed<base_seed>_n<n_sims>")
        assert "<batch_base_seed>" in timing["command_template"]
        rerun = block["variability_rerun"]
        assert rerun["base_seeds"] == [pin_stage1.SECOND_BASE_SEED,
                                       pin_stage1.THIRD_BATCH_BASE_SEED]
        assert rerun["runs_through"] == "the timing_1b block above"

        # The smoke and full blocks pin exactly one seed each.
        assert block["smoke_1a"]["base_seeds"] == [pin_stage1.BASE_SEED]
        assert block["full_run"]["base_seeds"] == [pin_stage1.BASE_SEED]

        # Astra round 3, item 1: every launchable block binds its own
        # fixture directory, inventory hash and count; the 1b shard is
        # unpinned until it is built, which is what makes it unclaimable.
        for name in ("smoke_1a", "full_run"):
            assert block[name]["fixture_dir_md5"], name
            assert block[name]["fixture_count"] > 0, name
        assert timing["fixture_dir_md5"] is None
        assert timing["fixture_dir_status"].startswith("absent")


def test_the_extra_source_files_are_pinned(committed):
    """Astra round 2, item 6."""
    sources = committed["provenance"]["source_md5"]
    for name in ("scripts/identity_maps.py", "scripts/match_identity.py",
                 "scripts/elo_update.py",
                 "scripts/sim_eval/settlement_common.py"):
        assert name in sources, name
        assert len(sources[name]) == 32
    assert committed["provenance"]["source_md5_absent"] == []


def test_overlap_screen_cli_passes_on_the_registered_set():
    assert main(["--check-seed-overlap"]) == 0
    assert main(["--check-seed-overlap", "--n-sims", "100"]) == 0


def test_overlap_prose_states_the_mechanism(committed):
    text = " ".join([
        committed["seeds"]["interval_mechanism"],
        committed["seeds"]["interval_guard"],
    ]).lower()
    assert "random_seed + i" in text
    assert "grows with n_sims" in text
    assert "does not cancel" in text
    assert "refuses" in text
    for deviation in committed["deviations"]:
        if deviation["id"] == "single_global_rng_stream":
            second = deviation["second_consequence"].lower()
            assert "vanishingly" not in second
            assert "grows with n_sims" in second
            assert "does not cancel" in second
            break
    else:
        raise AssertionError("the RNG-stream deviation is missing")
