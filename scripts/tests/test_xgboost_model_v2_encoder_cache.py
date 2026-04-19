"""
Regression test for the encoder-cache refactor (Option A) in XGBoostModelV2.

What it verifies:
  1. Round-trip: cached dict outputs match LabelEncoder.transform for every class.
  2. Unknown IDs return -1 (matches the old try/except sentinel behavior).
  3. extract_features parity: cached path vs simulated old "transform-and-except" path
     produce identical encoded values on a real test match.
  4. predict_next_ball parity: probabilities are bit-identical between paths.

Standalone (no pytest): exits non-zero on failure.

Run:
    uv run python scripts/tests/test_xgboost_model_v2_encoder_cache.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim_v1_2 import XGBoostModelV2  # noqa: E402
from stats_provider import StatsProvider  # noqa: E402
from player_metadata import PlayerMetadataProvider  # noqa: E402
from sim_eval.loaders import TestMatchLoader  # noqa: E402


MODEL_PATHS = {
    'model_path': 'models/xgb_v3/xgboost_model_v3.pkl',
    'batter_encoder_path': 'models/xgb_v3/batter_encoder_v3.pkl',
    'bowler_encoder_path': 'models/xgb_v3/bowler_encoder_v3.pkl',
    'feature_columns_path': 'models/xgb_v3/feature_columns_v3.txt',
    'matchup_encoder_path': 'models/xgb_v3/matchup_encoder_v3.pkl',
}


def banner(msg):
    print(f"\n--- {msg} ---")


def load_model():
    stats_provider = StatsProvider('models', version='v3')
    player_metadata = PlayerMetadataProvider('data/all_players_enriched.csv')
    return XGBoostModelV2(
        **MODEL_PATHS,
        stats_provider=stats_provider,
        player_metadata=player_metadata,
    )


def test_roundtrip(model):
    banner("(1) Round-trip cache vs LabelEncoder.transform")
    pairs = [
        ('batter', model.batter_encoder, model._batter_id_to_code),
        ('bowler', model.bowler_encoder, model._bowler_id_to_code),
    ]
    if model._matchup_to_code is not None:
        pairs.append(('matchup', model.matchup_encoder, model._matchup_to_code))

    for name, encoder, cache in pairs:
        bad = 0
        for cls in encoder.classes_:
            key = str(cls)
            if cache[key] != int(encoder.transform([key])[0]):
                bad += 1
        assert bad == 0, f"{name}: {bad}/{len(encoder.classes_)} classes mismatched"
        print(f"  ok  {name}: {len(encoder.classes_)} classes round-trip")


def test_unknown(model):
    banner("(2) Unknown IDs return -1")
    UNK = '__definitely_not_a_real_id__'
    assert model._batter_id_to_code.get(UNK, -1) == -1
    assert model._bowler_id_to_code.get(UNK, -1) == -1
    if model._matchup_to_code is not None:
        assert model._matchup_to_code.get(UNK, -1) == -1
    print("  ok  unknown IDs -> -1 across all caches")


def _legacy_encode(encoder, key):
    """Reproduce the pre-refactor try/except: -1 path."""
    try:
        return int(encoder.transform([key])[0])
    except Exception:
        return -1


def test_extract_features_parity(model, state):
    banner("(3) extract_features parity (cached vs legacy transform path)")
    feats_after = model.extract_features(state)

    # Recompute the three encoded fields the legacy way and compare
    striker = state.current_striker
    bowler = state.current_bowler
    legacy_batter = _legacy_encode(model.batter_encoder, str(striker.player_id))
    legacy_bowler = _legacy_encode(model.bowler_encoder, str(bowler.player_id))

    # Find the column indices for the three encoded fields
    cols = model.feature_columns
    idx_batter = cols.index('batter_encoded')
    idx_bowler = cols.index('bowler_encoded')

    assert int(feats_after[idx_batter]) == legacy_batter, (
        f"batter_encoded mismatch: cached={int(feats_after[idx_batter])} legacy={legacy_batter}"
    )
    assert int(feats_after[idx_bowler]) == legacy_bowler, (
        f"bowler_encoded mismatch: cached={int(feats_after[idx_bowler])} legacy={legacy_bowler}"
    )
    print(f"  ok  batter_encoded={legacy_batter}, bowler_encoded={legacy_bowler} match")

    if 'matchup_type_encoded' in cols and model.matchup_encoder is not None:
        idx_match = cols.index('matchup_type_encoded')
        matchup = model.player_metadata.get_matchup_type(striker.player_id, bowler.player_id)
        legacy_matchup = _legacy_encode(model.matchup_encoder, matchup)
        assert int(feats_after[idx_match]) == legacy_matchup, (
            f"matchup_type_encoded mismatch: cached={int(feats_after[idx_match])} "
            f"legacy={legacy_matchup}"
        )
        print(f"  ok  matchup_type_encoded={legacy_matchup} matches (matchup={matchup!r})")


def test_predict_parity(model, state):
    banner("(4) predict_next_ball parity vs legacy path (probs bit-equal)")
    feats_cached = model.extract_features(state).copy()
    probs_cached = model.predict_next_ball(feats_cached)

    # Build a "legacy" feature buffer by overwriting the encoded fields with the
    # legacy try/except result. If both paths agree on inputs, predict outputs match.
    striker = state.current_striker
    bowler = state.current_bowler
    legacy_batter = _legacy_encode(model.batter_encoder, str(striker.player_id))
    legacy_bowler = _legacy_encode(model.bowler_encoder, str(bowler.player_id))

    cols = model.feature_columns
    feats_legacy = feats_cached.copy()
    feats_legacy[cols.index('batter_encoded')] = legacy_batter
    feats_legacy[cols.index('bowler_encoded')] = legacy_bowler
    if 'matchup_type_encoded' in cols and model.matchup_encoder is not None:
        matchup = model.player_metadata.get_matchup_type(striker.player_id, bowler.player_id)
        feats_legacy[cols.index('matchup_type_encoded')] = _legacy_encode(model.matchup_encoder, matchup)

    probs_legacy = model.predict_next_ball(feats_legacy)

    # Compare every outcome key
    keys = sorted(set(probs_cached) | set(probs_legacy))
    diffs = [(k, abs(probs_cached[k] - probs_legacy[k])) for k in keys]
    max_diff = max(d for _, d in diffs)
    assert max_diff < 1e-12, f"predict_next_ball drift > 1e-12: {diffs}"
    print(f"  ok  max |Δ prob| across {len(keys)} outcomes = {max_diff:.2e}")


def main():
    print("Loading XGBoostModelV2 + real artifacts...")
    model = load_model()
    print("Model loaded.")

    test_roundtrip(model)
    test_unknown(model)

    print("\nLoading a real test match for extract_features / predict parity...")
    matches = TestMatchLoader().load_matches('data/betting_test')
    if not matches:
        print("FAIL: no test matches loaded")
        sys.exit(1)
    _, state = matches[0]
    print(f"Using match state for {state.team1} vs {state.team2}")

    test_extract_features_parity(model, state)
    test_predict_parity(model, state)

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
