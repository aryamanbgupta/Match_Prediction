**MUST-FIX**

1. **R2-M1 remains partially open: family decisions contradict § 2.7.** [rollout_protocol.md:249](docs/sequence_track/rollout_protocol.md:249) requires a favourable family primary for expansion and makes production comparisons cumulative. But [rollout_protocol.md:308](docs/sequence_track/rollout_protocol.md:308) expands `mlp_114` on an **adverse** primary, while [rollout_protocol.md:324](docs/sequence_track/rollout_protocol.md:324) permits `residual_mlp − A` to trigger expansion even when the primary is inconclusive or parity. Make these rules consistent. If reverse-direction expansion is intended, explicitly define candidate-oriented primary success for expansion and promotion; remove the residual A-only trigger under the adopted cumulative rule.

**NOTE**

1. **R2-M1’s membership and SQ5 requirements are closed.** Family membership is frozen across batches, fixed-decay/FoX are research-only, and SQ5 names the C114 primary. [rollout_protocol.md:303](docs/sequence_track/rollout_protocol.md:303), [rollout_protocol.md:330](docs/sequence_track/rollout_protocol.md:330), [astra_handoff_2026-09-13.md:138](docs/sequence_track/astra_handoff_2026-09-13.md:138).

2. **R2-M2 is closed at specification level.** Actual selected-seed identities survive the analysis alias; the five-seed production comparison is explicitly typed against one fixed A artifact, with LANDED blocked pending implementation and testing. [rollout_protocol.md:233](docs/sequence_track/rollout_protocol.md:233), [rollout_protocol.md:255](docs/sequence_track/rollout_protocol.md:255).

3. **R2-M3 is closed.** Every entry must match the registered 252/167 eligibility inventories and exclusion reasons; comparisons require identical IDs, labels and blocks, without silently intersecting omissions away. [rollout_protocol.md:196](docs/sequence_track/rollout_protocol.md:196).

4. **R2-M4 is closed.** The single Stage 1 reading rule now specifies strict point thresholds, interval boundaries and precedence, and removes the contradictory touching rule. [rollout_protocol.md:241](docs/sequence_track/rollout_protocol.md:241).

5. **R2-S1 is closed.** Acceptance checks come first, the writer precedes imports, standard serving certification precedes B/C admission, and source hashes freeze after launcher implementation. [astra_handoff_2026-09-13.md:97](docs/sequence_track/astra_handoff_2026-09-13.md:97).

6. **R2-S2 is partially closed.** Launcher generalisation and full cluster-source merging are explicit. Exact source/output locations and operational commands remain assigned implementation deliverables rather than supplied entry points. This does not block registration. [astra_handoff_2026-09-13.md:117](docs/sequence_track/astra_handoff_2026-09-13.md:117).

7. **R2-S3 is partially closed; qualification contains a new metric mismatch.** Serving/contract identity and typed pins are corrected. However, the proposed per-fixture winner-probability tolerance refers to Stage 1’s simulation SD, which measures scaled **log-loss contrast** variability, not probability differences. Pin a numerical tolerance for the actual qualification metric before qualifying the mini; laptop registration need not wait. [rollout_protocol.md:183](docs/sequence_track/rollout_protocol.md:183), [rollout_protocol.md:152](docs/sequence_track/rollout_protocol.md:152), [rollout_protocol.md:273](docs/sequence_track/rollout_protocol.md:273), [SEQ_STAGE1_REPORT.md:219](research/reports/embeddings/SEQ_STAGE1_REPORT.md:219).

Most round-2 concerns are resolved. Registration still needs one consistent answer to which primary result permits each candidate to expand and later advance. The remaining operational details can be completed during implementation before their respective steps run. This review changed no files, ran no simulations, and did not open any excluded directory.

VERDICT: AGREE WITH CHANGES
