# Phase A2 — Caveats and follow-up diagnostics

**TL;DR**: A2 formally clears the go/no-go gate on every w ∈ [0.0, 0.65] and every slice
(LL on ≥$50k = 0.5135 vs market 0.6267, ROI CI lower bound +29.1%). Three findings argue
for treating the headline number as **plausibly real but inflated** and not yet shippable
without one more validation pass.

## 1. Temporal divergence within the test set

Splitting the 255-match polymarket-overlap test slice in half by date:

| Slice | LL | ROI | Bets |
|---|---|---|---|
| Early (2025-09-10 → 2026-01-18, n=130) | (NaN-mixed) | **+21.47%** | 124 |
| Late (2026-01-19 → 2026-04-16, n=131) | 0.4605 | **+63.62%** | 131 |

Late test ROI is ~3× early test ROI. The form / H2H / home-venue trackers
accumulate state as we walk matches chronologically across the full corpus.
For the first test match (Sept 2025), the trackers contain only train data
(through 2024-12-31). For the last test match (April 2026), the trackers
contain ~9 months of test-era results — including the WC qualifying tournaments
that dominate the sharp-market polymarket eval set.

A real bookmaker also has this data at game time, so this is not strictly
*leakage*. But it does mean the headline number reflects a "warm-start" regime
where trackers grow alongside the eval period. The early-test number (+21%)
is the closer proxy for "what we'd realize betting in real time using only
data known beforehand", and even that's far above realistic 1-3% market edges.

## 2. Outlier sensitivity

Two bets — France @ 20.0 odds (won, +19.0 PnL) and Zimbabwe @ 11.76 odds
(beat Australia, +10.76 PnL) — account for ~30 of the +110 total PnL on the
all-261 slice. Stripping them:

| | n bets | ROI |
|---|---|---|
| All bets | 255 | +43.12% |
| Strip top-2 outlier wins | 253 | **+31.70%** |

Still very high. Indicates the headline ROI has meaningful tail dependence
on a small number of long-shot wins, which is the betting strategy that
naturally emerges when a model is confident the favorite is over-priced
and the implied probability of the underdog is fatter than market.

## 3. ROI grows with liquidity — counter to expectation

| Slice | Direct-alone ROI | LL |
|---|---|---|
| all (255) | +43.12% | 0.5226 |
| ≥$50k (168) | **+47.35%** | 0.5135 |
| ≥$100k (110) | **+51.04%** | 0.4554 |

Sharp markets (≥$100k) are typically harder to beat — efficiency increases
with liquidity. Our direct-model ROI moves the *opposite* way, which is
either a rare instance of a real edge that scales with sharpness, or a
signal that the model is exploiting something the sharp market also doesn't
fully price (which is also rare).

## Recommended diagnostic before declaring victory

**Freeze trackers at the train+val boundary (2025-06-30) and re-evaluate.**
The current materializer updates trackers in place across the entire corpus.
A no-leakage variant should:

1. Build trackers chronologically through 2025-06-30 (end of val).
2. Snapshot tracker state at that boundary.
3. For each test match, query the *frozen* tracker (no within-test updates).
4. Re-train direct model and re-evaluate.

If the resulting LL on ≥$50k stays comfortably below 0.62 and ROI CI still
excludes zero, we have a real edge. If LL drifts up substantially (say, to
0.60 or higher), the headline result was driven by tracker accumulation
during the test period, and the real-world edge is more like the early-test
+21% (still good but more believable).

A complementary check: report metrics on the early-test half only as a
"strict-holdout" headline and treat the late-test half as a "hot-tracker"
sensitivity number.

## Why the result is plausibly real anyway

- High-confidence predictions calibrate well: 100% accuracy at p > 0.85
  (n=26), 84.3% at 0.7-0.85 (n=102). The model is not just overconfident.
- Top-importance features (`bottom5_bowling_elo_diff`, `top6_batting_elo_diff`)
  are well-motivated: top-of-order batting strength and back-end bowling
  unit quality. v7 sim only sees the lineup-wide aggregates, not the
  position-split versions.
- Direct-model architecture has a known information advantage over the
  ball-aggregating sim (see plan §6). Some of the LL improvement is
  exactly what the architectural premise predicts.

## Decision

**Don't ship A2 yet.** Run the no-leakage diagnostic above; if it confirms,
ship A2 with the early-test ROI as the conservative headline and the
late-test number as an upper bound on what's realistically achievable
when trackers are well-populated.
