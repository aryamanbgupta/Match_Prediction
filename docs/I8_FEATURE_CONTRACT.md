# I8 per-player phase and matchup distribution contract

## Problem

The current ball model knows each batter’s and bowler’s overall six-class
outcome distribution, plus batter-vs-pace and bowler-vs-hand distributions.
It does not know whether a specific player changes their scoring/dismissal
profile between the powerplay, middle overs, and death, and its H2H features
are only average and strike rate.

Raw cells are too sparse to use directly. In the I7 SQLite cache there are
287,096 distinct batter–bowler pairs, but only 51,324 have at least 12 balls
and 8,112 have at least 30. Phase splitting also divides each player’s history
across three cells. I8 therefore treats shrinkage as part of the feature
definition, not a later tuning choice.

## Candidate features

I8 adds exactly 18 probabilities in six-class order `0, 1, 2, 4, 6, W`:

- `batter_phase_p*`: batter distribution in the current pre-ball phase;
- `bowler_phase_p*`: bowler distribution in the current pre-ball phase; and
- `h2h_p*`: exact striker–bowler distribution.

It keeps all 114 I7 ball features unchanged. The optional six global
`phase_p*` experiment is not added, so the I8 comparison isolates these 18
features.

## Phase and counting semantics

Phase is determined from legal balls bowled before the delivery:

- powerplay: `balls_bowled < 36`;
- middle: `36 <= balls_bowled < 96`;
- death: `balls_bowled >= 96`.

Counts use the same normalized model target as the rest of the selected I7
recipe: `0, 1, 2, 4, 6, W`, with 3→2, 5→4, and 7+→6. Wicket counts follow the
ball target (any wicket event), while scalar H2H dismissals retain their
existing bowler-wicket semantics. All features are read from pre-delivery
state; current-ball outcomes are applied only afterward.

## Fixed hierarchy

For counts `n` and parent distribution `q`, every child uses:

`p_c = (n_c + k q_c) / (sum(n) + k)`.

- Batter phase parent: the batter’s overall distribution, itself shrunk to
  the global corpus prior with `k_player=30`.
- Bowler phase parent: the bowler’s overall distribution under the same rule.
- H2H parent: the arithmetic mean of those shrunk batter and bowler overall
  distributions.
- `k_phase=30`; `k_h2h=60`.

An unseen child cell therefore equals its player-aware parent. An unseen
player resolves through the global prior. The arithmetic H2H parent remains
normalized and avoids the unsupported sharpening that a multiplicative
combination would introduce.

These strengths are precommitted for the first I8 run. A later sweep would be
a new experiment and may use validation only; the frozen test and forward
evaluation sets cannot select them.

## Artifact isolation

I8 requires SQLite schema version 5 and the active I7 venue identity contract.
It writes only to:

- `models/player_stats_cache_i8.sqlite`;
- `data/xgb_data_i8/`; and
- `models/xgb_i8/`.

Global and global-phase priors are copied from the frozen I7 SQLite artifact,
not recomputed from any post-test context that happens to exist when I8 is
built. Player, phase, and H2H rows remain as-of-date queried; the frozen prior
closes the only corpus-wide constant that could otherwise see later matches.

Schema-v4 readers remain supported for frozen production. Schema-v4 caches
must fail clearly if an I8 getter is requested. The I8 model sidecar records
all four shrinkage strengths, the schema version, and the venue identity
contract so simulation cannot silently serve a different feature definition.

I8 simulation runs through `scripts/sim_eval/run_sim_eval_i8.py`, which layers
the 18 new values onto the unchanged frozen simulator. It requires the I8
model, all 18 feature columns, the four-value shrinkage sidecar, and a
schema-v5 provider. Any mismatch aborts the run; I8 never substitutes zeros or
the base runner's demonstration-only dummy model.

## Evaluation

The first gate compares unchanged I7 and I8 architectures on the existing
chronological validation and frozen test splits using ball log loss,
multiclass Brier score, and class calibration. Simulation on the already-used
261-match Polymarket set is diagnostic and must use paired seeds under the
same current simulator.

No consumed forward fixture may be used to tune or promote I8. Because I8 was
specified on 2026-07-30, final promotion requires a later untouched terminal
window. Until then, even a positive diagnostic result remains an experiment,
not a production replacement.

## Implementation checkpoint

I8 was implemented and evaluated on 2026-07-30. Schema-v5 storage, additive
readers, pre-ball feature materialization, model sidecars, and the isolated
fail-closed simulation runner are complete. The first frozen run preserved
all 9,519 historical matches and produced 132-feature rows exactly paired
with I7.

I8 modestly improved validation/test ball log loss; the test multiclass
Brier delta was the only confidence-clean result. On the consumed
Polymarket diagnostic, full-slice match log loss improved from 0.7042 to
0.6825, while flat ROI fell from +0.46% to -1.49%; all competition-block
delta intervals crossed zero. I8 is therefore retained but not promoted.
No shrinkage or hyperparameter sweep may use these results. See
`reports/i8_phase_matchup_checkpoint_20260730.md` for the complete paired
results, upset sensitivity, runtime cost, and terminal gate.
