"""
Phase 3 phase-prior unit tests.

Covers the 5 implementation surfaces:
1. parsing_v2._classify_phase_pre_ball — phase boundaries (PP/mid/death).
2. parsing_v2._phase_dist_from_priors — dispatch + zero fallback.
3. parsing_v2.parse_match_data_v2 — emits 6 phase_p* per ball.
4. parsing_v2.parse_match_data_v2 — inn_agg conservation
   (Σ phase counts ≡ Σ overall cX).
5. stats_sqlite_backend._SQLiteBackend.get_phase_outcome_dist — pre-Phase-3
   fallback to global prior + post-rebuild dispatch.
6. sim_v1_2._OUTCOME_DIST_ZERO has 6 phase keys; _fill_outcome_dists
   threads balls_bowled correctly.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from parsing_v2 import (
    _classify_phase_pre_ball,
    _phase_dist_from_priors,
    _ZERO_PHASE_DIST,
    parse_match_data_v2,
    PlayerStatsTracker,
    VenueStatsTracker,
)


# ─── 1. _classify_phase_pre_ball ──────────────────────────────────────────
def test_phase_boundaries():
    cases = [
        (0, 'powerplay'),
        (1, 'powerplay'),
        (35, 'powerplay'),
        (36, 'middle'),
        (37, 'middle'),
        (95, 'middle'),
        (96, 'death'),
        (119, 'death'),
        (120, 'death'),
        (200, 'death'),
    ]
    for bb, expect in cases:
        got = _classify_phase_pre_ball(bb)
        assert got == expect, f"balls_bowled={bb}: got {got}, expected {expect}"


# ─── 2. _phase_dist_from_priors ───────────────────────────────────────────
_TEST_PHASE_PRIORS = {
    'powerplay': (0.30, 0.45, 0.05, 0.12, 0.05, 0.03),
    'middle':    (0.32, 0.42, 0.08, 0.10, 0.04, 0.04),
    'death':     (0.20, 0.35, 0.10, 0.18, 0.08, 0.09),
}


def test_phase_dispatch_returns_correct_tuple():
    pp_dist = _phase_dist_from_priors(_TEST_PHASE_PRIORS, balls_bowled=10)
    assert abs(pp_dist['phase_p0'] - 0.30) < 1e-12, pp_dist
    assert abs(pp_dist['phase_p4'] - 0.12) < 1e-12, pp_dist

    mid_dist = _phase_dist_from_priors(_TEST_PHASE_PRIORS, balls_bowled=50)
    assert abs(mid_dist['phase_p2'] - 0.08) < 1e-12, mid_dist

    death_dist = _phase_dist_from_priors(_TEST_PHASE_PRIORS, balls_bowled=100)
    assert abs(death_dist['phase_p6'] - 0.08) < 1e-12, death_dist


def test_phase_dispatch_zero_on_none():
    d = _phase_dist_from_priors(None, balls_bowled=10)
    assert d == _ZERO_PHASE_DIST


def test_phase_dispatch_zero_on_missing_phase():
    """Defensive: malformed prior dict (missing phase) → zero, not crash."""
    d = _phase_dist_from_priors({'powerplay': (0.3,)*6}, balls_bowled=50)
    assert d == _ZERO_PHASE_DIST


# ─── 3. parse_match_data_v2 emits phase features per ball ─────────────────
def _minimal_match():
    """Tiny synthetic match: 1 over PP + 1 over mid + 1 over death, single
    side bats. Enough balls to land one in each phase."""
    return {
        "info": {
            "venue": "Test Ground",
            "dates": ["2025-01-01"],
            "teams": ["A", "B"],
            "registry": {"people": {"alice": "p_alice", "bob": "p_bob",
                                     "carol": "p_carol", "dan": "p_dan"}},
            "toss": {"winner": "A", "decision": "bat"},
            "team_type": "international",
            "players": {"A": ["alice", "bob"] * 6, "B": ["carol", "dan"] * 6},
            "outcome": {"winner": "A"},
        },
        "innings": [{
            "team": "A",
            "overs": [
                # over 0 (PP, balls 0-5 pre-ball)
                {"over": 0, "deliveries": [
                    {"batter": "alice", "non_striker": "bob", "bowler": "carol",
                     "runs": {"batter": 1, "extras": 0, "total": 1}}
                ]},
                # 6 powerplay balls — but actually we want one in each phase
                # over 6 (mid, balls 36-41)
                # filler to reach ball 36
            ] + [
                {"over": i, "deliveries": [
                    {"batter": "alice", "non_striker": "bob", "bowler": "carol",
                     "runs": {"batter": 0, "extras": 0, "total": 0}}
                    for _ in range(6)
                ]}
                for i in range(1, 17)  # overs 1-16, 96 balls so we span all 3 phases
            ],
        }],
    }


def test_parse_emits_phase_features_per_ball():
    stats = PlayerStatsTracker()
    venue = VenueStatsTracker()
    rows, _, _, _, _ = parse_match_data_v2(
        json.dumps(_minimal_match()), stats, venue, None,
        prior=(0.3, 0.4, 0.08, 0.1, 0.05, 0.07),
        phase_priors=_TEST_PHASE_PRIORS,
    )
    assert rows, "expected at least one ball"
    for r in rows:
        for k in ('phase_p0', 'phase_p1', 'phase_p2',
                  'phase_p4', 'phase_p6', 'phase_pw'):
            assert k in r, f"missing {k} in ball record"
        # phase_p* sum should equal 1.0 within fp tolerance
        s = sum(r[f'phase_p{c}'] for c in ('0', '1', '2', '4', '6', 'w'))
        assert abs(s - 1.0) < 1e-9, f"phase_p* sum={s} on ball {r.get('ball_idx')}"


def test_parse_phase_dispatch_matches_classification():
    stats = PlayerStatsTracker()
    venue = VenueStatsTracker()
    rows, _, _, _, _ = parse_match_data_v2(
        json.dumps(_minimal_match()), stats, venue, None,
        prior=(0.3, 0.4, 0.08, 0.1, 0.05, 0.07),
        phase_priors=_TEST_PHASE_PRIORS,
    )

    # First ball: pre-ball balls_bowled=0 → PP → p0=0.30
    first = rows[0]
    assert abs(first['phase_p0'] - 0.30) < 1e-9, \
        f"first ball phase_p0={first['phase_p0']}, expected 0.30 (PP)"

    # Find a mid-phase ball (pre-ball 36..<96)
    mid = next((r for r in rows if 36 <= r['balls_bowled'] < 96), None)
    if mid is not None:
        assert abs(mid['phase_p2'] - 0.08) < 1e-9, \
            f"mid ball phase_p2={mid['phase_p2']}, expected 0.08"


# ─── 4. inn_agg conservation: Σ phase counts ≡ Σ overall cX ──────────────
def test_inn_agg_phase_conservation():
    """parse_match_data_v2's inn_agg dict puts every legal ball into
    exactly one phase bucket. Σ over phases of cX_{phase} must equal
    overall cX, otherwise build_stats_cache's phase-prior accumulation
    will be wrong."""
    stats = PlayerStatsTracker()
    venue = VenueStatsTracker()
    _, _, _, innings_details, _ = parse_match_data_v2(
        json.dumps(_minimal_match()), stats, venue, None,
        prior=(0.3, 0.4, 0.08, 0.1, 0.05, 0.07),
        phase_priors=_TEST_PHASE_PRIORS,
    )

    for det in innings_details:
        for ck in ('c0', 'c1', 'c2', 'c4', 'c6', 'cw'):
            phase_sum = (det.get(f'{ck}_powerplay', 0) +
                         det.get(f'{ck}_middle', 0) +
                         det.get(f'{ck}_death', 0))
            overall = det.get(ck, 0)
            assert phase_sum == overall, \
                f"bucket {ck}: phase_sum={phase_sum} != overall={overall}"


# ─── 5. SQLite backend get_phase_outcome_dist ─────────────────────────────
def test_sqlite_phase_getter_pre_phase3_fallback():
    """On a pre-Phase-3 cache (no prior_pp_p* / prior_mid_p* / prior_death_p*
    in _meta), every phase falls back to the global prior π. The getter
    still returns 6 keys with sane (sum-to-1) values."""
    from stats_sqlite_backend import _SQLiteBackend
    sqlite_path = PROJECT_ROOT / "models" / "player_stats_cache_v3.sqlite"
    if not sqlite_path.exists():
        print(f"[SKIP] {sqlite_path} not present")
        return
    backend = _SQLiteBackend(str(sqlite_path))
    backend._ensure_conn()
    for bb in (0, 50, 100):
        d = backend.get_phase_outcome_dist(bb)
        assert sorted(d.keys()) == sorted(_ZERO_PHASE_DIST.keys())
        s = sum(d.values())
        assert abs(s - 1.0) < 1e-6, f"phase dist for bb={bb} sums to {s}"


# ─── 6. sim_v1_2 wiring ──────────────────────────────────────────────────
def test_sim_outcome_dist_zero_includes_phase():
    from sim_v1_2 import _OUTCOME_DIST_ZERO
    phase_keys = [k for k in _OUTCOME_DIST_ZERO if k.startswith('phase_p')]
    assert len(phase_keys) == 6, f"expected 6 phase keys, got {phase_keys}"


def test_fill_outcome_dists_threads_balls_bowled():
    from sim_v1_2 import _fill_outcome_dists
    from stats_sqlite_backend import _SQLiteBackend

    sqlite_path = PROJECT_ROOT / "models" / "player_stats_cache_v3.sqlite"
    if not sqlite_path.exists():
        print(f"[SKIP] {sqlite_path} not present")
        return
    backend = _SQLiteBackend(str(sqlite_path))
    backend._ensure_conn()

    d = {}
    _fill_outcome_dists(d, backend, 'fake_pid', 'fake_pid2', 'MCG',
                        '2024-06-15', balls_bowled=50)
    for k in ('phase_p0', 'phase_p1', 'phase_p2',
              'phase_p4', 'phase_p6', 'phase_pw'):
        assert k in d, f"missing {k}"
    s = sum(d[f'phase_p{c}'] for c in ('0', '1', '2', '4', '6', 'w'))
    assert abs(s - 1.0) < 1e-6


def test_fill_outcome_dists_zero_when_balls_bowled_none():
    from sim_v1_2 import _fill_outcome_dists
    from stats_sqlite_backend import _SQLiteBackend

    sqlite_path = PROJECT_ROOT / "models" / "player_stats_cache_v3.sqlite"
    if not sqlite_path.exists():
        print(f"[SKIP] {sqlite_path} not present")
        return
    backend = _SQLiteBackend(str(sqlite_path))
    backend._ensure_conn()

    d = {}
    _fill_outcome_dists(d, backend, 'fake_pid', 'fake_pid2', 'MCG',
                        '2024-06-15', balls_bowled=None)
    assert d['phase_p0'] == 0.0


# ─── runner ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [v for k, v in globals().items()
             if k.startswith("test_") and callable(v)]
    failures = []
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failures.append((t.__name__, str(e)))
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:
            failures.append((t.__name__, f"{type(e).__name__}: {e}"))
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")

    print()
    if failures:
        print(f"FAILED {len(failures)} / {len(tests)}")
        sys.exit(1)
    print(f"All {len(tests)} Phase-3 tests passed.")
