# v7 ball-level pipeline — leakage audit (2026-05-09)

**TL;DR**: The v7 ball-level sim does NOT have an analogous leakage to the
match-level `_split_elo` bug. Verified structurally and empirically.

The match-level bug pattern was: `_build_match_record` ran AFTER
`parse_match_data_v2` had mutated `temp_elo` ball-by-ball with this match's
own outcomes; reading position-split ELOs at that point produced
post-match values as features for predicting the same match. v7 has no
analogous second-pass tracker read.

## Structural audit

Inspected `scripts/parsing_v2.py:parse_match_data_v2`,
`scripts/materialize_features.py`, `scripts/build_stats_cache.py`,
`scripts/tracker_rehydration.py`, `scripts/stats_sqlite_backend.py`.

### Per-ball features (clean)

Per-ball ordering inside the inner ball loop (`parsing_v2.py:1154-1366`):

1. Read `state` from delivery (line 1157).
2. Read all features from trackers — `batting_features`, `bowling_features`,
   `h2h_features`, `striker_elo`, `bowler_elo_rating`, `batter/bowler/h2h`
   outcome distributions, etc. (lines 1162-1305).
3. Append `ball_record` to `all_balls` (line 1327).
4. Call `player_stats_tracker.update_stats(...)` (line 1332).
5. Call `elo_tracker.update(...)` (line 1343).
6. Call `innings_calc.update_ball_history(...)` (line 1352).

Every per-ball feature read happens BEFORE that ball's outcome is fed to
any tracker. Ball N+1 sees state updated by ball N — that is the correct
"intra-match prior balls" semantics, not leakage.

### Match-start team-level features (clean)

`team_batting_elo`, `team_bowling_elo`, `team_batting_avg`,
`team_bowling_econ`, etc. are computed once before the ball loop at
`parsing_v2.py:1061-1098`, immediately after `start_match()` (which only
clears per-match accumulators, doesn't touch cumulative stats or ELO).
At that point `temp_elo` and `player_stats_tracker` reflect pre-match
state. There is no equivalent of the match-level `_build_match_record`
that re-reads trackers AFTER the parser has mutated them.

### v7 has no top-6 / bottom-5 ELO splits

The leak in match-level was specifically in the position-split ELO
computation inside `_build_match_record`. v7 does not compute these
splits at all — it uses lineup-wide aggregates only — so the specific
bug pattern doesn't apply.

### Cross-match (same-day) semantics

`materialize_features.py:156-187`:

* Rehydrate `temp_stats / temp_elo / temp_venue` once per date from
  SQLite as-of `match_date`.
* Iterate same-day matches in monolith order; each match calls
  `parse_match_data_v2` which mutates the shared trackers ball-by-ball.
* After parser returns, `temp_venue` is updated with that match's innings
  details and chase outcome.

So same-day match 2 sees post-match-1 state. This is the documented
"same-day match order matters" invariant (CLAUDE.md §invariant 5), not
leakage. The downstream signal is real: a match starting later on the
same day genuinely happens after earlier ones.

### SQLite snapshot semantics (clean)

`scripts/build_stats_cache.py:466-471`:

```python
if date_str not in snapshotted_dates:
    snapshotted_dates.add(date_str)
    date_id = _intern(date_ids, date_str)
    snap = deep_copy_stats(stats, venue, elo)
    emit_snapshot(snap, date_id)
```

First-write-wins per date: the row tagged `date_id=D` captures tracker
state as-of the start of the first match on D, i.e. strictly pre-D.
Subsequent matches on D advance the trackers but do not re-emit a row
for D.

`stats_sqlite_backend.py:455-459`:

```python
def _resolve_date_id(self, as_of_date) -> int:
    """Largest date_id whose date ≤ as_of_date, or -1 if none."""
    target = self._norm_date(as_of_date)
    idx = bisect.bisect_right(self._date_strs, target)
    return idx - 1
```

Querying with `as_of_date=D` returns the row tagged D, which is pre-D
state by the build-time invariant. Calls like
`rehydrate_elo_tracker(provider, match_date)` therefore return state
strictly before the match's date.

### `venue_tracker` is not mutated mid-match

Inside `parse_match_data_v2`, `venue_tracker` is read-only (lines 1041,
1042, 1046). Innings details are accumulated locally and returned;
`temp_venue.update_*` calls happen in the caller AFTER the parser
returns (`materialize_features.py:183-186`). So the venue features
emitted in this match's per-ball rows reflect pre-match venue state.

## Empirical audit

Script: `/tmp/claude/audit_v7_leakage_v2.py` (kept for reproducibility,
not committed).

Strategy: pick test-period dates that contain exactly ONE match in the
JSON corpus (no same-day siblings to confuse comparison). For the first
ball of innings 1, compare two values:

- `striker_elo` from the parquet (what the parser emitted).
- `elo_pre.get_batting_elo(batter_id)` from a fresh
  `rehydrate_elo_tracker(provider, match_date)` (which is, by build-time
  semantics, pre-D state).

If structurally clean, these must be bit-exact equal. Same comparison
for `batting_team_elo` vs `elo_pre.get_team_batting_elo(lineup_ids)`.

Result on 10 sampled solo-date test matches:

```
Checked 10 first-of-day matches:
  striker_elo  max |delta|: 0.000000
  team_elo     max |delta|: 0.000000
  nonzero rows: 0/10

  ALL CLEAN. No leakage detected on first-of-day v7 features.
```

A first pass that compared by `(date, batter_id)` without filtering on
same-day siblings showed 8/35 nonzero striker drifts (max ~3 ELO units)
and 9/35 nonzero team drifts (max ~6 ELO units). Investigation: those
drifts are entirely from same-day match-2/3 first balls, where parquet
correctly reflects post-match-1 state but the SQLite pre-D snapshot
predates match 1. Once the comparison is restricted to first-of-day
matches, all drift goes to zero — confirming the cross-match semantics
match the documented invariant rather than leaking.

## Why v7 is structurally different from match-level

The match-level bug had two ingredients:

1. **A second-pass feature builder** (`_build_match_record`) that re-read
   trackers AFTER the parser had already advanced them.
2. **A feature that needed per-player state** (`_split_elo` reading
   individual `temp_elo.batting_elo[pid]` for top-6 batters, bottom-5
   bowlers).

v7 has neither: there is no second-pass builder (per-ball rows are
materialized in-line during the parser's ball loop), and no per-ball
feature uses tracker state from after that ball's own update.

The narrowest formulation: v7 reads tracker[pid] only at points where
ball N's outcome has not yet been written. That invariant is upheld by
the read-then-update ordering inside the ball loop and by the
match-start team aggregates being computed before the ball loop runs.

## Conclusion

v7 ball-level pipeline is clean. The decision to demote v7 sim to
props/scores/in-play (per the match-level direct vs sim comparison) is
NOT confounded by symmetric leakage in v7. The honest LL gap (v7 0.7402
on ≥$50k vs market 0.6267 vs match-level direct clean 0.6747) reflects a
genuine resolution problem in v7, not a fixable measurement artifact.

## Open follow-ups

The audit covered the leakage class symmetric to the match-level bug
(temporal mis-ordering within the parsing pipeline). It does NOT prove
v7 is free from other classes of leakage (e.g. label leakage in
SQLite-built training data, gender/competition contamination, or feature
aliasing inside `feature_registry`). Those were already implicitly
audited during the earlier no-leakage diagnostic for the match-level
model and the v6/v7 phase ablations; no new audit is recommended unless
v7 is revived as a winner-market candidate.
