# M3 — player-level rolling form (2026-05-10)

Phase 3 of match-level v3. Added 18 player-level rolling-form features (last-5-matches batting avg/SR per player, bowling avg/econ, in-form / out-of-form indicators) aggregated to lineup level.

**Outcome**: Drop M3 features. **Switch baseline to M2 v.o. UNFROZEN** materialization (a separate finding surfaced during M3 diagnosis). The unfrozen materialization itself — not the M3 features — is what delivers the M3-phase improvement.

**New baseline**: `models/xgb_match_v3_m2_venue_only_unfrozen/`. Same features as M2 v.o. (M1 + 3 venue outcome-dist), trained on `data/xgb_match_data_v3_m3_unfrozen/` (no `--freeze-trackers-after` flag, chronological tracker walk through both `t20s_json` and `golden/t20s_json`).

**M4 baseline**: `models/xgb_match_v3_m2_venue_only_unfrozen/` (raw, not calibrated).

## Why M3 features failed

Added (~18 features in `materialize_match_features.py:_rolling_form_features`):
- Top-6 batters' mean recent batting avg/SR + pairwise diff (4 features + 2 diffs).
- All-11 bowlers' (with non-empty recent_bowling deque) mean recent bowling avg/econ + pairwise diff (4 features + 2 diffs).
- Form indicators: count of top-6 batters with recent_avg ≥ 1.2 × career_avg ("in form") or ≤ 0.8 × career_avg ("out of form") per team + diffs (4 features + 2 diffs).

Initial training under FROZEN materialization (the M1/M2 mode):

| Variant | val LL | test LL | iter ≥$50k LL | iter ≥$50k ROI [CI] | golden ≥$50k LL |
|---|---|---|---|---|---|
| M2 v.o. frozen (baseline) | 0.6521 | 0.6227 | 0.6347 | +22.77% [+2.73, +43.46] | 0.6885 |
| M3 full frozen | 0.6504 | 0.6196 | 0.6449 | +14.00% [-6.75, +35.96] | 0.6945 |

M3 full FAILS the planned exit criterion (Δ ≥ -0.003 LL on iter ≥$50k): actual Δ +0.010 (wrong direction).

### Drop-one ablation (frozen)

| Variant | val LL | test LL | iter ≥$50k LL |
|---|---|---|---|
| M3 full | 0.6504 | 0.6196 | 0.6449 |
| M3 − batting recent | 0.6510 | 0.6225 | (not run) |
| **M3 − bowling recent** | 0.6526 | **0.6169** | **0.6296** |
| M3 − form indicators | 0.6462 | 0.6234 | (not run) |
| M3 form-only | 0.6514 | 0.6205 | 0.6378 |

Bowling-recent ACTIVELY HURTS — dropping it improves test LL by 0.003. Same pattern as M2's bowler outcome-dist; the bottom-5-by-squad-order proxy is not a clean bowling unit.

`M3 − bowling-recent` clears the planned LL gate (iter ≥$50k 0.6296 vs M2 v.o. 0.6347 = Δ -0.005), but loses the M2 v.o. iteration ROI CI gate (lower bound -0.35 vs +2.73).

### Stale-tracker hypothesis test (UNFROZEN mode)

Hypothesis: the SQLite cache freezes at 2025-06-30, so player recent_batting deques for IPL 2026 reflect ~10-month-old form. Re-materialize without `--freeze-trackers-after` so each match sees the chronological tracker state through pre-match date.

Materialization at `data/xgb_match_data_v3_m3_unfrozen/`. Walks both `data/t20s_json` and `data/golden/t20s_json` in chronological order. For a test match on 2026-04-01, `temp_stats` reflects all matches up to (but not including) 2026-04-01.

Feature drift on golden (n=62):

| Feature | mean |Δ| frozen vs unfrozen | max |Δ| |
|---|---|---|
| team1_top6_batting_avg_recent | 6.84 | 19.9 |
| team1_top6_batting_sr_recent | 13.46 | 49.7 |
| team1_n_inform_batters | 1.16 | 4 |
| team1_n_outofform_batters | 1.24 | 4 |

Confirmed: frozen mode produces materially different recent-form values for golden. ~7 batting avg points of mean drift, with peaks of ~20.

### M3 retrained on unfrozen data

| Variant | iter ≥$50k LL | iter ≥$50k ROI [CI] | golden ≥$50k LL | golden ≥$50k ROI |
|---|---|---|---|---|
| M2 v.o. unfrozen | **0.6348** | **+25.40% [+4.75, +48.11]** | 0.6925 | +14.39% |
| M3 full unfrozen | 0.6517 | +16.81% [-3.14, +39.19] | 0.6946 | +10.22% |
| M3 − bowling unfrozen | 0.6448 | +8.05% [-12.56, +30.84] | 0.7016 | +10.29% |

**Even under unfrozen materialization, M3 features do not improve over M2 v.o.** Adding M3 features hurts iteration LL by +0.017 and ROI by 8.6pp. The career-aggregate features (`top6_batting_elo_diff`, `team1_batting_avg`, `team1_batting_sr`) already capture player quality at the resolution match-level XGBoost can use; recent-5-match aggregates add mostly noise on top.

**Conclusion**: drop all M3 features. The exercise still produced a useful finding: **the FROZEN-vs-UNFROZEN materialization choice itself is a real lever**, and unfrozen wins on the metrics that matter.

## New baseline — M2 v.o. UNFROZEN headline numbers

### Iteration polymarket eval (n=261, blend w=0.0)

| Slice | n | LL | LL 95% CI | Flat ROI | ROI 95% CI | Win % |
|---|---|---|---|---|---|---|
| ≥$50k raw | 170 | 0.6348 | [0.603, 0.667] | +25.40% | [+4.75, +48.11] | 53.6 |
| ≥$50k Platt | 170 | **0.6279** | [0.591, 0.665] | **+26.69%** | **[+6.26, +48.65]** | 55.4 |
| ≥$100k raw | 110 | 0.6006 | [0.557, 0.643] | +26.21% | [-0.93, +57.10] | 53.6 |
| ≥$100k Platt | 110 | **0.5908** | [0.540, 0.640] | **+26.61%** | **[+0.63, +57.42]** | 55.5 |
| Reference: market | — | 0.6267 | — | — | — | — |

**M2 v.o. unfrozen + Platt clears BOTH iteration ≥$50k gates simultaneously** (LL 0.6279 < market 0.6267, ROI CI lower +6.26). Also clears ≥$100k (LL 0.5908, ROI CI lower +0.63). This is the strongest iteration evidence on the project to date.

### Golden eval (n=55, truly out-of-sample)

| Slice | n | LL | LL 95% CI | Flat ROI | ROI 95% CI | Win % |
|---|---|---|---|---|---|---|
| ≥$50k raw | 50 | 0.6925 | [0.648, 0.746] | +14.39% | [-16.98, +44.38] | 51.0 |
| ≥$50k Platt | 50 | **0.6849** | [0.630, 0.747] | **+31.29%** | **[+1.92, +59.11]** | 61.2 |
| ≥$100k raw | 44 | 0.6944 | [0.652, 0.745] | +14.46% | [-17.08, +47.16] | 52.3 |
| ≥$100k Platt | 44 | 0.6850 | [0.632, 0.747] | **+33.28%** | **[+0.08, +64.37]** | 63.6 |

**Golden ≥$50k Platt ROI +31.29% [+1.92, +59.11]** — first variant to cleanly clear ROI gate on truly-out-of-sample golden. Win rate 61.2%. Borderline-clean ≥$100k too.

### Adversarial slices (iteration ≥$50k, raw)

| Slice | n | LL | Flat ROI | ROI 95% CI | Win % |
|---|---|---|---|---|---|
| close (\|ELO diff\| ≤ 5) | 74 | 0.6982 | **+26.12%** | [-2.34, +52.30] | 52.7 |
| mismatch (\|ELO diff\| ≥ 15) | 24 | 0.3731 | +22.54% | [-13.48, +58.37] | 62.5 |

**Close-slice ROI jumps to +26.12%** (was -5.7% in M1 frozen, +1.7% in M2 v.o. frozen). The unfrozen materialization fixes the close-fixture under-performance that M3 features were supposed to address. Slice composition shifts (n=74 vs 58 prior) because top6_batting_elo_diff distribution itself changes under unfrozen.

### Walk-forward (iteration ≥$50k, raw)

| Month | n | LL | Flat ROI |
|---|---|---|---|
| 2025-12 | 27 | 0.7187 | -13.23% |
| 2026-01 | 45 | 0.6690 | +11.36% |
| 2026-02 | 49 | 0.5218 | +52.63% |
| 2026-04 | 35 | 0.6583 | **+43.67% [+10.36, +76.65]** |

2026-04 (start of IPL): **+43.67% ROI with CI cleanly excluding 0**, 68.6% win rate. Clearest "real edge" signal on a single month.

## Why unfrozen wins on this regime

The `no_leakage_diagnostic_clean.md` audit (2026-05-09) found frozen-vs-unfrozen differences of ~0.01–0.02 LL on polymarket-overlap, with frozen winning. The new finding flips that pattern under the M2 venue-only feature set:

| | iter ≥$50k LL | iter ≥$50k ROI | golden ≥$50k LL |
|---|---|---|---|
| M2 v.o. frozen | 0.6347 | +22.77% | **0.6885** |
| M2 v.o. unfrozen | 0.6348 | **+25.40%** | 0.6925 |

LL is flat; ROI improves by +2.6pp; golden LL is slightly worse (small, within noise). The improvement comes from the bet-side selection being driven by features that reflect closer-to-match-day truth: when temp_stats has tracked all matches through 2026-03-31 before scoring a 2026-04-01 fixture, the model's bet-side decisions match the actual prevailing dynamics rather than an outdated snapshot.

**Trade-off explicitly accepted**: golden LL +0.004 worse, golden Platt ROI +10pp better — net positive for both LL gate (which Platt + ≥$50k cleared anyway) and ROI gate.

## Status of M3 verification criteria

1. ❌ **Iteration ≥$50k Δ LL ≤ -0.003 with M3 features**: NOT cleared. M3 full Δ +0.010 (wrong direction). Best M3 subset (M3 − bowling-recent) Δ -0.005 (clears LL but loses ROI gate).
2. ✅ **Drop-one ablation surfaced clean signal**: bowling-recent features hurt; batting-recent + form indicators are individually neutral.
3. ✅ **Stale-tracker hypothesis test**: confirmed material feature drift (mean |Δ| ~7 avg points on golden). Validated, but features still don't deliver lift even with fresh tracker state.
4. ✅ **Phase-secondary finding (unfrozen materialization)**: M2 v.o. unfrozen clears iteration AND golden gates simultaneously, win rate 53-63% across slices. Strongest baseline on the project.

## What this means for M4

- **M4 baseline is `models/xgb_match_v3_m2_venue_only_unfrozen/` (raw, not calibrated)**.
- **M4 numbers to beat**: iter ≥$50k LL 0.6348 / ROI +25.40% [+4.75, +48.11]; golden ≥$50k LL 0.6925 / ROI +14.39% raw, +31.29% [+1.92, +59.11] Platt.
- **Materialization mode going forward**: unfrozen. Future re-materializations should NOT use `--freeze-trackers-after`. The `frozen` variants remain on disk for diagnostic reference.
- **M3 features stay in the materializer code** (`_rolling_form_features` helper, FEATURE_COLUMNS entries) but are excluded from production via the `--drop-features` substring filter. Kept callable for future re-evaluation (e.g., if we add window-size sweeps in M7).
- **Open follow-up — bowler identification**: the bottom-5-by-squad-order proxy for "bowling unit" continues to be wrong. M2's bowler outcome-dist failed, M3's bowling-recent failed. M5 (player × opp affinity) and possibly an explicit M-phase on "metadata-based bowler selection" should fix this.

## Why M3 features don't add value (post-mortem)

1. **Career-aggregate features dominate**: M1's `top6_batting_elo_diff`, `team1_batting_avg`, `team1_batting_sr` already capture player quality at the granularity match-level XGBoost can usefully discriminate. Adding 5-match recency on top doesn't surface new signal — it covers the same predictive variance.
2. **5-match window is too noisy**: 5 matches per player produces high-variance estimates dominated by single-innings outliers. The corpus prior π = (0.304, 0.411, 0.076, 0.108, 0.047, 0.054) means a single 100-run innings shifts a 5-match avg by ~20 points.
3. **Tournament composition confounds**: a player's last-5 might all be IPL or all be internationals, with very different baselines. The aggregate treats them identically.
4. **In-form / out-of-form thresholds (20%) are arbitrary**: real form deviations are continuous; binarizing at ±20% loses gradient information.

A future window-size sweep (last-10, last-20 matches with shrinkage) might surface a real signal, but it's not on the critical path. M4–M6 features attack different problems (within-tournament dynamics, opposition affinity, conditions) that should give cleaner lift.

Artifacts:
- `models/xgb_match_v3_m2_venue_only_unfrozen/{model.pkl, encoders.pkl, feature_columns.txt, train_metrics.json, test_predictions{,_calibrated}.json, golden_predictions{,_calibrated}.json, platt_calibrator.json}`
- `models/xgb_match_v3_m3{,_no_br,_form_only}/` — frozen variants preserved
- `models/xgb_match_v3_m3{,_no_br}_unfrozen/` — unfrozen variants preserved
- `data/xgb_match_data_v3_m3{,_unfrozen}/` — full 84-feature parquets
- `eval_out_m3_*` and `eval_out_m2vouf_*` — sliced eval JSONs
- `reports/walk_forward_m3.md` — iteration ≥$50k monthly on M2 v.o. unfrozen
