You are Codex Astra, reviewing a DESIGN DRAFT (read-only) for the CricML sequence track in this repo (branch embeddings-ladder). You have reviewed every stage of this track so far (stage 1, stage 2 two-seed and five-seed, night 3 Block B, batch 2) and signed them off. You are about to become the orchestrator of the track; this review is the last one before that handover, so review it as the person who will have to run it.

Read, in this order: docs/sequence_track/rollout_protocol.md (the draft under review), scripts/sim_t1.py lines 1-200 and the TransformerT1SimModel class (what the simulator's model adapter supports today), scripts/sequence_track/run_arm.py (how stage 1 launched an arm), experiments/configs/seq_stage1_sim_v1.yaml (the stage 1 pins), research/reports/embeddings/SEQ_STAGE1_REPORT.md § 7-8 (cost and shard consistency), research/reports/embeddings/SEQ_STAGE3_BATCH2_REPORT.md § 5.2 and § 7 (the C114 result that motivates this), scripts/sequence_track/retrain_stage2.py (what a run directory carries: metrics.json training_contract / arm_params, model.pt, predictions_validation.npz), scripts/sequence_track/recurrent_arms.py, and docs/SEQUENCE_TRACK_PLAN.md § "Statistical rules". Do not open data/golden, data/forward_holdout or models/embeddings/seq_stage2/cohort.

The user's complaint that this draft answers, verbatim: "Why do we have to keep rerunning these rollout tests? Can we design a comprehensive one? Why are we having to redo this again and again every time we come up with a new reason for having to rerun these experiments? We should have one seed per arm rollout. Once we have a directional signal, after that, one seed, then we can do the five-seed rollout."

Questions:
1. Does the root-cause analysis in § 1 match the code? Name anything in sim_t1.py / run_arm.py that makes the "one adapter for every arm" design harder than the draft assumes (feature building from simulated state, the serving-contract guard, prefix caches, thread caps, the residual arms' need for production logits, the 114-contract encoders, recurrent state stepping).
2. Is the per-checkpoint teacher-forced parity test (§ 2.1) a sufficient admission gate? What would you add or drop?
3. Is importing the stage 1 A / A50 / B / C runs as ledger rows (§ 2.2) legitimate, and what exactly must be verified for that import?
4. The seed policy (§ 2.4): one checkpoint per configuration by lowest validation LL, five seeds only for a promotion candidate. Any objection, and is the promotion trigger stated tightly enough?
5. The first-batch families (§ 3): are the three registered families and their anchors the right ones, and is the expected-outcome statement honest?
6. Answer the five open questions in § 5 directly.
7. Estimate the engineering cost of the adapter per arm family in hours, and say which families you would defer.

Output: numbered MUST-FIX, then SHOULD, then NOTE, each with file:line where applicable; then direct answers to questions 1-7; then a plain-language paragraph for the user (no jargon) on whether this design ends the rerun cycle. End with exactly one line: "VERDICT: SIGN-OFF" or "VERDICT: AGREE WITH CHANGES" or "VERDICT: NO SIGN-OFF".
