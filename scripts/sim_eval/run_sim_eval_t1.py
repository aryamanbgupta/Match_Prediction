#!/usr/bin/env python3
"""Run match simulation with the T1 innings transformer (candidate runner).

i8 pattern: never edits the frozen runner — imports `run_sim_eval` and swaps
three seams before delegating to its `main()`:

1. `TransformerModelV1` -> a factory returning `TransformerT1SimModel` bound
   to a **SameDayReplayStatsProvider** (the certified T1 feature path). The
   old delegation handed the wrapper `run_sim_eval`'s plain snapshot
   StatsProvider, whose counts exclude earlier same-day fixtures — the exact
   divergence the 2026-08-14 review flagged; `OnlineT1OutcomeDists` now
   refuses snapshot providers outright.
2. `TestMatchLoader` -> a chronological loader: test matches ordered by
   `(match_date, match_id)` from the same-day context corpus
   (`--t1-context-dir`, default `data/t20s_json`), states built through the
   public `create_match_state`.
3. `MatchLevelEvaluator` -> a subclass whose `evaluate_all` drives the replay
   provider lifecycle (`begin_date` / `begin_match` / `lock_prediction` /
   `advance_match`) around each simulated match, replaying every same-day
   context fixture in between. Single pass, strictly chronological — the
   provider enforces both.

Extra flag: `--t1-context-dir <dir>` (stripped before delegation). Refused
flags: `--parallel` (the lifecycle is strictly sequential) and `--calibrate`
(the T1 program forbids post-hoc match calibration). `--allow-stateful-wrapper`
is injected automatically: the generic guard targets wrappers with silent
cross-match state, while the T1 wrapper's token cache is bound to the exact
MatchState and fails closed on desync (PPC-certified).

Checkpoint dir override: T1_SIM_MODEL_DIR env var. All other run_sim_eval
flags pass through; `--model-version` (default v3) picks the stats cache.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
# The frozen runner is a sibling script, importable by bare name only when
# this file runs as a script; make module imports (tests) resolve it too.
if str(SCRIPTS_DIR / "sim_eval") not in sys.path:
    sys.path.insert(1, str(SCRIPTS_DIR / "sim_eval"))

import run_sim_eval as runner  # noqa: E402
from player_metadata import PlayerMetadataProvider  # noqa: E402
from registered_experiment import reject_sealed  # noqa: E402
from sim_eval.loaders import TestMatchLoader  # noqa: E402
from sim_eval.match_evaluator import MatchLevelEvaluator  # noqa: E402
from sim_eval.same_day_stats import SameDayReplayStatsProvider  # noqa: E402
from sim_t1 import TransformerT1SimModel  # noqa: E402
from stats_provider import StatsProvider  # noqa: E402
from t1_ppc_common import build_context_by_date  # noqa: E402


def _arg_value(flag: str, default: str) -> str:
    try:
        return sys.argv[sys.argv.index(flag) + 1]
    except (ValueError, IndexError):
        return default


def _pop_flag_with_value(flag: str, default: str) -> str:
    """Read and remove `flag <value>` from sys.argv (runner doesn't know it)."""
    if flag not in sys.argv:
        return default
    index = sys.argv.index(flag)
    try:
        value = sys.argv[index + 1]
    except IndexError:
        raise SystemExit(f"{flag} requires a value")
    del sys.argv[index:index + 2]
    return value


class _T1ReplayEvaluator(MatchLevelEvaluator):
    """`evaluate_all` with the same-day replay lifecycle woven in.

    Mirrors the frozen evaluator's loop (odds resolution, skip semantics,
    per-match prints, `_aggregate_results`) while walking the full context
    chronology: non-test and odds-less fixtures advance the trackers as
    context; simulated fixtures lock their prediction before replay.
    """

    # Bound by main() before delegation.
    provider: SameDayReplayStatsProvider = None
    context_by_date: dict = None

    def evaluate_all(self, matches, odds_lookup):
        if self.parallel:
            raise SystemExit(
                "run_sim_eval_t1 is strictly sequential (the replay "
                "lifecycle orders every stats read); drop --parallel")
        received = {match_id: state for match_id, state in matches}
        match_results = []
        total_time = 0
        claimed_odds_rows = {}
        covered = 0

        print(f"\nEvaluating {len(received)} matches with "
              f"{self.n_simulations} simulations each (same-day replay)...")

        for date_str in sorted(self.context_by_date):
            batch = self.context_by_date[date_str]
            if not any(mid in received for mid, _doc in batch):
                # --max-matches trims the chronological prefix; dates with
                # no evaluated fixture are skipped whole (begin_date only
                # requires strictly increasing dates, not contiguity).
                continue
            self.provider.begin_date(date_str, [doc for _mid, doc in batch])
            for match_id, document in batch:
                state = received.get(match_id)
                if state is None:
                    self.provider.begin_match(
                        match_id, document, prediction_required=False)
                    self.provider.advance_match(match_id, document)
                    continue
                covered += 1
                print(f"\n[{covered}/{len(received)}] Evaluating {match_id}")
                odds_data = self._resolve_odds_row(
                    match_id, state, odds_lookup,
                    claimed_rows=claimed_odds_rows)
                required = odds_data is not None
                self.provider.begin_match(
                    match_id, document, prediction_required=required)
                if not required:
                    print(f"  Warning: No odds found for {match_id}, "
                          "skipping...")
                    self.provider.advance_match(match_id, document)
                    continue
                try:
                    result = self._evaluate_single_match(
                        match_id, state, odds_data)
                    match_results.append(result)
                    total_time += result.simulation_time
                    print(f"  Simulated: {state.team1} "
                          f"{result.simulated_win_prob[state.team1]:.1%} vs "
                          f"{state.team2} "
                          f"{result.simulated_win_prob[state.team2]:.1%}")
                    if result.actual_winner:
                        print(f"  Actual Winner: {result.actual_winner}")
                    print(f"  Log Loss: {result.log_loss:.3f}")
                except Exception as e:
                    print(f"  Error evaluating match: {e}")
                finally:
                    # Lock before replay even on error: the fixture's actual
                    # deliveries must never precede its own prediction.
                    self.provider.lock_prediction(match_id)
                    self.provider.advance_match(match_id, document)

        if covered != len(received):
            missing = sorted(set(received)
                             - {m for d in self.context_by_date
                                for m, _ in self.context_by_date[d]})
            raise RuntimeError(
                f"replay chronology covered {covered}/{len(received)} "
                f"requested matches; absent from context: {missing}")
        return self._aggregate_results(match_results, total_time)


def main() -> None:
    if _arg_value("--model-type", "xgboost") != "transformer":
        raise SystemExit("run_sim_eval_t1.py requires --model-type transformer")
    for flag in ("--parallel", "--calibrate"):
        if flag in sys.argv:
            raise SystemExit(
                f"run_sim_eval_t1 refuses {flag}: the replay lifecycle is "
                "strictly sequential and the T1 program forbids post-hoc "
                "match calibration")

    context_dir = _pop_flag_with_value("--t1-context-dir", "data/t20s_json")
    test_dir = _arg_value("--test-dir", "data/test_matches")
    stats_version = _arg_value("--model-version", "v3")
    reject_sealed(context_dir)
    reject_sealed(test_dir)

    stems = {path.stem for path in sorted(Path(test_dir).glob("*.json"))}
    if not stems:
        raise SystemExit(f"no match JSON files found in {test_dir}")
    selected_dates = set()
    for path in sorted(Path(test_dir).glob("*.json")):
        with path.open() as fh:
            selected_dates.add(str(json.load(fh)["info"]["dates"][0]))

    print(f"Building same-day replay context from {context_dir} "
          f"({len(stems)} test matches over {len(selected_dates)} dates)...")
    started = time.time()
    context_by_date = build_context_by_date(context_dir, selected_dates)
    print(f"✓ Context ready: {sum(len(b) for b in context_by_date.values())} "
          f"fixtures on {len(context_by_date)} dates "
          f"({time.time() - started:.0f}s)")

    metadata = PlayerMetadataProvider("data/all_players_enriched.csv")
    provider = SameDayReplayStatsProvider(
        StatsProvider("models", version=stats_version), metadata)

    class _T1ChronologicalLoader(TestMatchLoader):
        def load_matches(self, _test_dir):
            matches = []
            for date_str in sorted(context_by_date):
                for match_id, document in context_by_date[date_str]:
                    if match_id not in stems:
                        continue
                    _, state = self.create_match_state(
                        document, cricsheet_id=match_id)
                    if state is None:
                        raise RuntimeError(
                            f"could not create state for {match_id}")
                    matches.append((match_id, state))
            missing = stems - {match_id for match_id, _state in matches}
            if missing:
                raise RuntimeError(
                    f"test matches absent from {context_dir} chronology: "
                    f"{sorted(missing)}")
            print(f"Successfully loaded {len(matches)} matches "
                  "(chronological)")
            return matches

    def _t1_model_factory(**_ignored_legacy_kwargs):
        return TransformerT1SimModel(
            stats_provider=provider, player_metadata=metadata)

    _T1ReplayEvaluator.provider = provider
    _T1ReplayEvaluator.context_by_date = context_by_date

    if "--allow-stateful-wrapper" not in sys.argv:
        # The generic guard targets wrappers with silent cross-match state.
        # T1's token cache binds to the exact MatchState and fails closed on
        # any desync (PPC-certified), so the runner opts in explicitly.
        print("note: injecting --allow-stateful-wrapper (T1's cache is "
              "state-exact and fail-closed; the generic warning targets "
              "the legacy wrappers)")
        sys.argv.append("--allow-stateful-wrapper")

    runner.TransformerModelV1 = _t1_model_factory
    runner.TestMatchLoader = _T1ChronologicalLoader
    runner.MatchLevelEvaluator = _T1ReplayEvaluator
    runner.main()


if __name__ == "__main__":
    main()
