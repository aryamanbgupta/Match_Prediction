# Delphyne → CricML Stage 3 mapping

Source: Ding, Mittal, Gopal, "DELPHYNE: A Pre-Trained Model for General and Financial
Time Series", NeurIPS 2025 (arXiv 2506.06288). Delphyne is a 12-layer, 768-dim encoder
transformer over *continuous* multivariate series (§4.2; 8×H100 for 4 days), pre-trained
on LOTSA plus financial data at 85/15 sampling weights (§4.1, Table 6). Our data is
discrete six-class ball events with a 50-dim context vector, ~1.9M balls, fully
supervised. Nothing drops in; three mechanisms map.

## 1. What the paper actually does

**3a Negative transfer (§2).** The evidence is synthetic and deliberately clean. Three
plain autoregressive transformer decoders — "no additional embeddings or patching" — are
trained on GARCH data (G-Model), Wavelet data (W-Model), and a half-and-half mixture
(M-Model). Measurement is zero-shot NLL on each source domain. Table 1: on Wavelet at
context 128, W-Model −0.1330 vs M-Model −0.0732; at context 32, −0.1020 vs −0.0733. The
mixture is strictly worse on the domain the specialist owns. Table 2 rules out capacity
and estimation error by redoing it with Bayesian MCMC that *knows* both generating
processes: Wavelet-MCMC NLL 0.2692 / MSE 0.1058 vs Mixture-MCMC 0.2937 / 0.1127. So the
harm is from the mixture objective, not from a finite-capacity network. Figure 2 shows it
on real data: adding finance data hurts Delphyne-A's zero-shot ETTh2 performance, and
fine-tuning "undoes" it. Their remedy is *not* domain conditioning or learned mixture
weights — it is cheap few-shot fine-tuning on the target domain (§2).

**3b Patching (§3.1).** Patch size is fixed at 32 timesteps; there is no multi-patch-size
ladder. A trainable linear projection maps each patch to an embedding. Crucially, the
[FORECAST MASK] and [MISSING MASK] are applied to the raw series *before* patching, so a
patch that straddles the boundary carries mask channels rather than future values.
Instance normalization is applied independently per variate (§3.1). The backbone is an
encoder with any-variate attention (§3.3), not a causal decoder — targets are hidden by
masking, not by a triangular mask.

**3c Pretraining objective (§3, Eq. 2; §3.2).** Pretraining is forecasting, not generic
reconstruction: minimize NLL of the masked/horizon region under a Student-T mixture head
(§3.4). Per-variate masking ratio is drawn from a beta-binomial with α=5, β=10 (mean
≈30%); each variate gets its own horizon h_j. Ablations: Table 4 sweeps ratios
{0.25, 0.5, 0.99} and finds mild masking (0.25) beats aggressive (0.99); Table 3 shows
pretraining context 128 beats 64 in the 10–100-sample fine-tuning regime.

## 2. What transfers, and the design choice

**3a — transfers, with one correction.** The mechanism (mixture objective degrades the
target tier even at infinite capacity) is domain-agnostic and applies to pooling IPL /
BBL / associate T20 / Hundred. What does *not* transfer is their remedy: we have no
few-shot fine-tuning regime — every tier has plenty of balls — so the plan's
tier-embedding-plus-per-tier-bias route is right, but note Delphyne never validated
conditioning. **Choice:** keep the plan's four arms and hold the conditioning-matched
pooled control as the only legitimate comparator; also mirror Table 2's spirit by running
the tier-restricted arm at matched row count, so a "negative transfer" verdict can never
be a data-quantity artifact (the paper does not state that its arms were row-matched, and
that is its weakest link).

**3b — transfers in form, not in mechanism.** A completed over is a natural fixed-size
patch (6 legal balls) exactly as 32 timesteps is theirs, and their pre-patch masking shows
how to keep a patch honest. What does not transfer is the encoder: their bidirectional
any-variate attention is illegal for us. **Choice:** completed overs only as patch tokens
(linear projection of 6 concatenated ball embeddings plus aggregated over context), a
strictly causal decoder over patches, and a separate within-over causal decoder for the
0–5 balls already bowled in the current over. A partial over is never patched.

**3c — weakest transfer, correctly deprioritized.** Their objective earns its keep because
unlabeled series vastly outnumber labeled ones. Our 1.9M balls are already labeled with
the exact quantity we predict, so masked reconstruction is at best a regularizer.
**Choice:** if run at all, mask *outcome tokens only* over contiguous spans (a whole over),
never individual context fields, and use their mild ratio (~25–30%, β-binomial α=5/β=10)
rather than an aggressive one.

## 3. Leakage risks, per idea

**3a:** tier labels can proxy for era or venue; a tier embedding must not be allowed to
carry match identity, and tier assignment must be derivable before the match starts. Split
by match, not ball, or the same innings appears in train and eval across arms.

**3b:** the over token must contain only balls strictly before the predicted ball —
the current over is the trap. Also the over-level aggregates (runs in over, wickets in
over) are functions of the balls inside it; if the current over is ever patched, the
label is in the input. Delphyne's "mask before patching" is the right pattern to copy.

**3c:** this is where our scoreboard kills the objective. `total_runs`, `wickets_fallen`,
`balls_remaining`, run rate, and any cumulative field at ball *t+1* algebraically recover
the outcome at ball *t*. A masked ball whose neighbours' cumulative fields are visible is
reconstructible by subtraction, not learned. Any masking scheme must mask the outcome and
every cumulative field downstream of it, or mask forward-only (their forecast-mask shape)
so no future cumulative state is ever visible.

## 4. Equal-compute baseline, per idea

**3a:** equal gradient steps, equal tokens *seen*, and equal target-tier rows across all
four arms; the tier-restricted arm must be allowed more epochs to match steps.

**3b:** the ball-level control gets the same parameter count, the same optimizer budget,
and the same wall-clock/FLOP budget — patching shortens sequences ~6×, so an unmatched
comparison would credit the patch for cheaper training. Match FLOPs, then report both.

**3c:** a from-scratch supervised model trained for pretrain-plus-finetune total steps,
on the same rows — not the current baseline at its current budget.
