# I7 live compatibility modes

> **Lifecycle status:** transitional. These two modes protect frozen artifacts
> during migration; they are not intended to become a permanent multi-mode
> serving API. The consolidation target is one canonical main path plus a
> temporary quarantined legacy replay command. See
> `docs/REPOSITORY_CONSOLIDATION.md`.

## Purpose

I7 fixes venue identity by joining reviewed aliases such as `Kennington Oval`
and `Kennington Oval, London` before state is accumulated or a category is
encoded. That is the correct contract for all future models. The frozen
production match model, v3 SQLite cache, and tracker snapshot were built before
that contract and do not carry its provenance.

Relabeling those frozen artifacts would be false provenance, while applying I7
canonicalization to only the fixture would mix identities at serving time.
The live CLI therefore makes the compatibility decision explicit.

## Modes

### `legacy` (live default)

Use only with the frozen pre-I7 artifact family. The fixture venue is trimmed
but otherwise preserved. Missing I7 metadata is accepted. This maintains
historical serving behavior, including fragmented venue history and the risk
that an alias spelling is unseen by the frozen encoder.

```bash
uv run python scripts/predict_fixture.py \
  --fixture fixtures/<match>.json \
  --venue-identity-mode legacy
```

If a tracker snapshot must be rebuilt for a refreshed legacy state family, pass
the same mode. The snapshot will retain raw venue labels and declare itself
`legacy`.

### `i7` (required for new models)

Use only when the match model, SQLite cache, and tracker snapshot were all
built under the active I7 map. The fixture venue is canonicalized and every
artifact must carry the exact map version, SHA-256, and active-row count.

```bash
uv run python scripts/predict_fixture.py \
  --fixture fixtures/<match>.json \
  --venue-identity-mode i7 \
  --model-dir models/<i7-model> \
  --state-dir data/<i7-state> \
  --state-version <i7-version> \
  --tracker-snapshot data/<i7-tracker>.pkl
```

I7 is deliberately opt-in on the current live command because neither
retrained I7 model passed its promotion gate. It remains the only permitted
identity foundation for I8 and later experiments.

## Guardrails

- A declared `legacy` snapshot cannot be served in `i7` mode, or vice versa.
- I7 mode rejects a missing or mismatched identity contract on any artifact.
- Legacy mode never manufactures or copies I7 provenance.
- Prediction output records `venue_identity_mode`, `fixture_venue_raw`, and
  `fixture_venue_effective`.
- Mode selection changes identity compatibility only. It does not waive
  freshness, source-count, player-resolution, liquidity, or betting-execution
  guardrails.

The modes are not candidates in a model-selection sweep. `legacy` is a
temporary serving bridge; I7 is the forward data contract.

## Retirement plan

No third venue-identity mode should be added. The next cleanup refactor should:

1. make canonical I7 identity the only behavior of the primary live command;
2. move frozen pre-I7 serving to a clearly named replay-only entry point;
3. replace generation-specific checks with one generic model-bundle manifest;
   and
4. delete the replay entry point after a canonical bundle has an accepted
   operational smoke test and tagged rollback release.

Until that refactor is approved, keep the present modes fail-closed and do not
extend legacy mode with new state, feature, or model capabilities.
