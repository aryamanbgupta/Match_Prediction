# I18 executor result — i7-identity golden frame + swap-i7 golden audit

**Status: COMPLETE. Gate A PASS (relaxed contract, per orchestrator rulings
1 and 2). Gate B readout below.**

The verdict is **DESCRIPTIVE regardless of the numbers** — golden slices carry
11 / 9 / 5 tournament blocks, and every ROI CI straddles zero. This is a gate
readout for a human promotion decision. Nothing here was used to select
anything: both model artifacts were frozen inputs, scored once.

---

## Gate B — golden readout (the numbers)

124 golden fixtures, 2026-04-17 → 2026-06-17. One fixture (cricsheet
`1529281`, Kolkata Knight Riders vs Lucknow Super Giants) has no result, so
LL and betting are scored over 123 / 74 / 65 while the match counts read
124 / 75 / 66. Market LL is a property of fixtures + odds, so it is identical
across arms.

### swap_seed29 — the I17 successor candidate

| slice | matches | bets | Avg Log Loss | market LL | Flat ROI | ROI 95% CI (I3 blocks) | win rate |
|---|---|---|---|---|---|---|---|
| all | 124 / 124 | 123 | **0.5988** | 0.5513 | −1.14% | [−45.24%, +23.29%] (11 blocks) | 48.0% |
| ≥$50k | 75 / 124 | 74 | **0.6538** | 0.6573 | +15.70% | [−57.35%, +32.24%] (9 blocks, descriptive) | 55.4% |
| ≥$100k | 66 / 124 | 65 | **0.6767** | 0.6843 | +21.23% | [−100.00%, +32.55%] (5 blocks, descriptive) | 58.5% |

### base_seed29 — control

| slice | matches | bets | Avg Log Loss | market LL | Flat ROI | ROI 95% CI (I3 blocks) | win rate |
|---|---|---|---|---|---|---|---|
| all | 124 / 124 | 123 | **0.6056** | 0.5513 | −17.68% | [−57.14%, +1.60%] (11 blocks) | 34.1% |
| ≥$50k | 75 / 124 | 74 | **0.6636** | 0.6573 | +1.23% | [−54.20%, +11.18%] (9 blocks, descriptive) | 45.9% |
| ≥$100k | 66 / 124 | 65 | **0.6859** | 0.6843 | +4.36% | [−100.00%, +10.30%] (5 blocks, descriptive) | 47.7% |

Verbatim reslice output is in `raw/15_reslice_all_arms.log`. The LL 95% CIs
printed alongside (not tabulated above) are swap 0.5988 [0.4492, 0.6718] /
0.6538 [0.4069, 0.6889] / 0.6767 [0.4277, 0.6938] and base 0.6056 [0.4526,
0.6840] / 0.6636 [0.4173, 0.7042] / 0.6859 [0.4393, 0.7102].

### Descriptive observations (no decision implied)

- **Swap beats base on LL at all three slices**: ΔLL −0.0068 (all), −0.0098
  (≥$50k), −0.0092 (≥$100k). The two sharp-slice deltas sit right at the
  logged seed noise floor of ~0.007 LL, on a single seed.
- **Swap beats base on ROI at all three slices**: −1.14% vs −17.68%,
  +15.70% vs +1.23%, +21.23% vs +4.36%. Win rate is higher on all three
  (48.0 / 55.4 / 58.5 vs 34.1 / 45.9 / 47.7).
- **Swap beats the slice-matched market LL on both sharp slices** (0.6538 <
  0.6573; 0.6767 < 0.6843) and loses on the all-slice (0.5988 > 0.5513).
  **Base beats the market on none** of the three. This is the same pattern
  I17 reported on the iteration set.
- **Every ROI CI straddles zero.** The ≥$100k lower bound is −100.00% on
  5 blocks. No economic edge is established by this readout.

### Context rows (from the plan; NOT part of any decision)

Legacy-line golden extension, same 124 fixtures, legacy sidecar features
(`reports/golden_extension_eval_20260730.md`) vs this i7-identity readout:

| slice | legacy swap LL | i7 swap LL | legacy swap ROI | i7 swap ROI |
|---|---|---|---|---|
| all | 0.5831 | 0.5988 | −4.35% | −1.14% |
| ≥$50k | 0.6685 | **0.6538** | +9.69% | **+15.70%** |
| ≥$100k | 0.6938 | **0.6767** | +14.38% | **+21.23%** |

The i7 frame is better than the legacy line on both metrics on both sharp
slices, and worse on all-slice LL. The legacy line beat the market on neither
sharp slice (0.6685 > 0.6573, 0.6938 > 0.6843), whereas i7 swap beats it on
both.

**Market LL cross-check:** the market LL computed here from the sliced JSONs
(0.5513 / 0.6573 / 0.6843) reproduces the legacy golden-extension report's
market numbers **exactly** to 4 dp on all three slices. Same fixtures, same
odds — this independently confirms the join and the slicing are correct.
See `raw/16_market_ll.log`.

Standalone LL over ALL 227 golden-frame rows (context only, no liquidity
slice, no market join): swap **0.6116** / Brier 0.2145; base **0.6220** /
Brier 0.2183; coinflip ref 0.6931.

---

## Gate A — integrity

### A1. Preconditions

`verify_forward_holdout` — PASS (`raw/00_verify_forward_holdout.log`):
```
{
  "status": "PASS",
  "model_scoring_performed": false,
  "selected_matches": 137,
  "context_matches": 401,
  "strict_timestamp_checks": 137,
  "cricsheet_hash_checks": 137,
  "overlap_with_existing_pools": 0,
  "dataset_fingerprint_sha256": "82ccde16cf2b7e5f13a9236f2788f3c8be1582f312f5c028ec44a6ab76561028"
}
```

Determinism spot-check — **max |Δp_team1| = 0.0**
(`raw/01_determinism_spotcheck.log`):
```
n_a (cricsheet-keyed): 798   n_b (display-keyed): 788
unique display ids in a: 788  unambiguous: 778  colliding: 10
max_abs_delta_p_team1 (unambiguous join, n=778) = 0.0
max over colliding of best-matching row: 0.0
```
The two artifacts use different key spaces, so the join is on
`display_match_id`. The 10 collisions are same-day same-venue doubleheaders
(e.g. `2025-08-03_Switzerland_Estonia_...` covers cricsheet 1494774 and
1494775); the display-keyed artifact retains one of each pair and the
retained row matches at exactly 0.0. No artifact drift since I17.

### A2. Materialization + frame subset

`raw/materialize.log`, `raw/03_build_i7_match_frame.log`:
```
DONE: 9,525 matches → data/auto/i18/i7_full_golden in 168s
  train: 7,972   validation: 528   test: 798   golden_test: 227
  dropped 224 matches with no valid winner / abandoned
```
```
  train: 131 -> 57 columns, 7,972 rows
  validation: 131 -> 57 columns, 528 rows
  test: 131 -> 57 columns, 798 rows
  golden_test: 131 -> 57 columns, 227 rows
```
**`golden_test` row count: 227.** **Materialization wall-clock: 168 s.**

Operational note: the first attempt was launched with `run_in_background` and
was killed when the turn ended (dead at ~match 1000/20s, empty output dir) —
the known D2 failure mode, flagged by the orchestrator. Everything after that
ran in the foreground. Nothing else crashed or ran long.

### A3. Parity gate — STRICT FAIL / RELAXED PASS (ruling 1)

`raw/04_parity_gate.log` (strict, first run), `raw/05_parity_diagnosis.log`
(diagnosis), `raw/10_parity_gate_relaxed.log` (both passes after the ruling).

Strict `DataFrame.equals` parity **FAILS**: the candidate is 57 columns vs
the frozen 54. Diagnosis showed the cause is schema drift, not data drift —
`materialize_match_features.py` gained the I15/I16 identity columns after the
frozen frame was built on 2026-07-25.

| | frozen `data/xgb_match_data_i7` (2026-07-25) | fresh `data/auto/i18/frame` |
|---|---|---|
| `match_id` holds | legacy display string | cricsheet id |
| `display_match_id` | absent | present |
| `match_identity_version` | absent | `cricsheet_primary_v1` |
| `elo_update_version` | absent | `fixed_competition_k_v1` |
| columns | 54 | 57 |

The I17 models were trained on the frozen frame; their
`match_identity.json` records the fallback contract
`{"match_identity_version": "synthetic_fixture_v1", "primary_key":
"match_id", "display_key": "match_id"}`, confirming it predates the contract.

Per ruling 1a the harness now encodes a relaxed contract as a second,
binding pass. Verbatim (`raw/10_parity_gate_relaxed.log`):
```
STRICT PARITY: FAIL

PASS 2 — RELAXED identity contract (the BINDING gate)
  [train]
    OK: added columns are exactly ['display_match_id', 'elo_update_version', 'match_identity_version']
    reference.match_id == candidate.display_match_id on 7972/7972 rows
    candidate.match_id == candidate.cricsheet_id on 7972/7972 rows
    cricsheet_id set equal: True  order equal: True
    shared columns compared (excl. match_id): 53
    OK: all shared columns (excl. match_id) bit-identical
  [validation]  ... 528/528, 528/528, 53 columns bit-identical
  [test]        ... 798/798, 798/798, 53 columns bit-identical
  [venue_identity.json]
    OK: {"venue_alias_active_count": 94, "venue_alias_sha256": "853b32b0ce3098dd8c0f33ba1437846f5505d50d9a425fbd37bff9c9f76745d8", "venue_alias_version": "venue_aliases_v1"}
  [golden_test.parquet]
    rows=227  columns=57
    match_identity_version values: ['cricsheet_primary_v1']
    cricsheet_id: 227 unique, 0 null
    match_date range: 2026-04-17 -> 2026-07-12

  train: PASS   validation: PASS   test: PASS
  venue_identity: PASS   golden_test: PASS

I18 RELAXED PARITY GATE: PASS
```

All 46 numeric features plus the 7 other shared metadata columns are
bit-identical, in identical row order, on all three splits. The golden merge
perturbed no pre-golden row.

**Deliberately unmet plan deliverable (ruling 1b):** the plan called for the
golden frame to land at `data/xgb_match_data_i7/golden_test.parquet`. **No
copy was made, by ruling.** The frozen siblings predate I15 and key
`match_id` by display string, so a copy would produce a mixed-contract
directory — the exact silent-join hazard I15 exists to prevent. I18's own
spec requires `match_identity_version` on the golden frame, which the frozen
siblings cannot carry, so this deviation follows the spec rather than
violating it. The coherent current-contract frame, all four splits, lives at
**`data/auto/i18/frame/`**; scoring read `golden_test.parquet` from there
(ruling 1c). Re-keying the frozen frame is a human decision.

### A4. Envelope stamping — 124/124 (ruling 2)

`synthesize_golden_envelope.py` wrote a 124-match envelope to
`models/auto/i18/golden_envelope.json` (explicit `--out`; the tool's default
`data/golden/golden_sim_envelope.json` was NOT written — still dated May 10).

The shared `patch_envelope_cricsheet_ids.py` **failed closed** with
`69 unmatched / 0 ambiguous` (`raw/07_...log`). Cause
(`raw/08_envelope_key_diagnosis.log`): the golden odds file is **mixed-key**
after the 2026-07-30 extension —
```
  match_id that are bare numeric (cricsheet-style): 69
  match_id that are legacy display strings        : 55
entries with non-empty cricsheet_id     : 69/124
numeric envelope ids that ARE polymarket_test stems: 69 / 69
```
The 55 original rows are legacy (display `match_id`, null `cricsheet_id`);
the 69 extension rows are already migrated. The shared tool indexes by raw
display id, so the 69 already-migrated rows read as UNMATCHED. The tool
behaved exactly as designed; it is simply the wrong tool for this input. The
plan's premise ("golden odds rows ... have NO `cricsheet_id` field") holds
only for the original 55.

Per ruling 2 the shared tool was **not modified**. New helper
`scripts/auto/i18_stamp_envelope.py` imports `raw_display_id` from the shared
tool and reuses its `stems_by_display` construction so the derivation cannot
drift, verifies already-present ids against the eval-set stems, stamps only
the legacy rows, and fails closed on unmatched / ambiguous / duplicate ids.
Verbatim (`raw/11_stamp_envelope.log`):
```
eval-set stems in data/golden/polymarket_test: 124
envelope entries: 124
  already-stamped and verified: 69
  legacy rows newly stamped   : 55
stamped ids: 124 total, 124 unique
FINAL GATE OK: stamped id set == eval-set stem set (124 ids, exact)
stamped cricsheet_id on 124 entries -> models/auto/i18/golden_envelope_cricsheet.json
```
This satisfies "all 124 golden odds rows join by `cricsheet_id`" with a
strictly stronger check than the shared tool: exact set equality against the
eval-set stems, plus a uniqueness gate.

### A5. Join gate — passthrough 0 on both arms

`raw/14_blend_both_arms.log`:
```
=== swap_seed29 ===
  w=0.00: 124 blended, 0 passthrough → models/auto/i18/swap_seed29/golden_envelope_cricsheet_w0p00.json
=== base_seed29 ===
  w=0.00: 124 blended, 0 passthrough → models/auto/i18/base_seed29/golden_envelope_cricsheet_w0p00.json
```

### A6. Unseen-venue WARN from `predict_golden.py`

Identical on both arms (`raw/12_...`, `raw/13_...`):
```
  WARN: unseen categorical values mapped to fallback class:
    venue: 3 unseen → ['Botswana Cricket Association Oval 2, Gaborone', 'Korogi Sports Park, Nisshin', 'Sano International Cricket Ground 2']
```
3 unseen venues (the legacy extension audit had 5). All three are associate /
development grounds outside the odds-matched 124.

### A7. Write-safety audit — clean

`raw/02_checksums_before.log` vs `raw/17_checksums_final.log`, `diff` of the
md5 blocks: **NO CHANGES to protected artifacts.**
```
22ceb40fd10cc3db46887a94193abe0a data/xgb_match_data_i7/test.parquet
9f225b2f9af63b1a57ba9553802b7a54 data/xgb_match_data_i7/train.parquet
e3df5b812389464aa8f91d831050699a data/xgb_match_data_i7/validation.parquet
72f62135985943c7a7fa412aef78c646 data/xgb_match_data_i7/venue_identity.json
1682b75284b9c47b377d8d780f218cb4 data/xgb_match_data_i7_full/test.parquet
b9d6a70c2f3adc317fba949e66d430ca data/xgb_match_data_i7_full/train.parquet
d458d03f2416c29960a1db1db8524a8f data/xgb_match_data_i7_full/validation.parquet
671ac8200b275fa3d11d848e609f5132 models/player_stats_cache_i7.sqlite
3faf24a6d457e9fca0f98c026b3b0675 models/player_stats_cache_v3.sqlite
```
- `data/xgb_match_data_i7/` still holds exactly its 4 original files, all
  dated Jul 25 18:02 — **no `golden_test.parquet`**.
- `data/golden/` untouched: 230 `t20s_json`, 124 `polymarket_test`,
  `golden_sim_envelope.json` still May 10, `betting_odds_golden.json` still
  Jul 30 17:20. Read-only, per the scoped authorization.
- `data/forward_holdout/` untouched (read once by `verify_forward_holdout`).
- New artifacts confined to `data/auto/i18/` and `models/auto/i18/`, both
  gitignored.

---

## Commits created

```
7bdd052 Auto[I18]: i7 golden frame build + parity harness
6b9fa11 Auto[I18]: relaxed identity parity contract + half-migrated envelope stamper
```
plus the handoff commit carrying this file and `raw/`.

`git diff --stat d3e4d64` (before the handoff commit):
```
 research/handoff/I18/plan.md       | 250 +++++++++++++++++++++++++++++
 scripts/auto/i18_frame_parity.py   | 311 +++++++++++++++++++++++++++++++++++
 scripts/auto/i18_stamp_envelope.py | 148 +++++++++++++++++
```

Not touched: `research/results.tsv`, `research/IDEAS.md`, `program.md`,
`research/night*.sh`, `research/RUNNER_PROMPT*.md`, `scripts/sim_eval/`,
`parsing_v2.py`, `stats_provider.py`, `stats_sqlite_backend.py`,
`predict_golden.py`, `synthesize_golden_envelope.py`,
`patch_envelope_cricsheet_ids.py`, `materialize_match_features.py`,
`build_i7_match_frame.py`. Nothing reverted, nothing pushed, no verdict
issued, no promotion or serving change made.

## Reproduce

```bash
uv run python scripts/materialize_match_features.py --version i7 \
  --extra-source-dir data/golden/t20s_json --out-dir data/auto/i18/i7_full_golden
uv run python scripts/build_i7_match_frame.py \
  --source-dir data/auto/i18/i7_full_golden --out-dir data/auto/i18/frame
uv run python scripts/auto/i18_frame_parity.py \
  --candidate-dir data/auto/i18/frame --reference-dir data/xgb_match_data_i7
uv run python scripts/synthesize_golden_envelope.py \
  --odds data/golden/betting_odds_golden.json \
  --out models/auto/i18/golden_envelope.json
uv run python scripts/auto/i18_stamp_envelope.py \
  --envelope models/auto/i18/golden_envelope.json \
  --test-dir data/golden/polymarket_test \
  --out models/auto/i18/golden_envelope_cricsheet.json
for arm in swap_seed29 base_seed29; do
  uv run python scripts/predict_golden.py --model-dir models/auto/i17/$arm \
    --parquet data/auto/i18/frame/golden_test.parquet \
    --out-json models/auto/i18/${arm}_golden_predictions.json
  uv run python scripts/sim_eval/blend_eval_json.py \
    --sim-json models/auto/i18/golden_envelope_cricsheet.json \
    --direct-json models/auto/i18/${arm}_golden_predictions.json \
    --w 0.0 --out-dir models/auto/i18/$arm
  uv run python scripts/sim_eval/reslice_eval_json.py \
    --in models/auto/i18/$arm/golden_envelope_cricsheet_w0p00.json \
    --odds data/golden/betting_odds_golden.json \
    --out-dir models/auto/i18/$arm/sliced_all \
    --cluster-source-dir data/golden/t20s_json
  # repeat with --min-volume 50000 / 100000 into sliced_50k / sliced_100k
done
```
