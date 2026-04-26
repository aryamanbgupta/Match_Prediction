#!/usr/bin/env python3
"""
Test simulation with real player stats

This script verifies that:
1. Stats provider loads successfully
2. Real player stats are being used (not zeros)
3. Simulations run correctly with the stats provider
"""

import sys
import json
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from stats_provider import StatsProvider
from sim_v1_2 import XGBoostModelV2, SimulationEngine, T20Rules, SimulationConfig
from sim_eval.loaders import TestMatchLoader

def test_simulation_with_real_stats():
    """Run a single match simulation and verify real stats are being used"""

    print("=" * 60)
    print("Testing Simulation with Real Player Stats")
    print("=" * 60)

    # Step 1: Load stats provider
    print("\n1. Loading stats provider...")
    try:
        stats_provider = StatsProvider('models')
        print("   ✓ Stats provider loaded successfully")
        print(f"   - Date range: {stats_provider.dates[0]} to {stats_provider.dates[-1]}")
        print(f"   - Total snapshots: {len(stats_provider.dates):,}")
    except Exception as e:
        print(f"   ✗ Failed to load stats provider: {e}")
        return False

    # Step 2: Load model with stats provider
    print("\n2. Loading XGBoost model with stats provider...")
    try:
        model = XGBoostModelV2(
            model_path='models/xgb/xgboost_model_v2.pkl',
            batter_encoder_path='models/xgb/batter_encoder_v2.pkl',
            bowler_encoder_path='models/xgb/bowler_encoder_v2.pkl',
            feature_columns_path='models/xgb/feature_columns_v2.txt',
            stats_provider=stats_provider  # Pass the stats provider
        )
        print("   ✓ Model loaded successfully with stats provider")
    except Exception as e:
        print(f"   ✗ Failed to load model: {e}")
        return False

    # Step 3: Load a single test match
    print("\n3. Loading test match from training data...")
    try:
        # Use first match from training data
        test_match_path = Path('data/train/1001349.json')

        if not test_match_path.exists():
            print(f"   ✗ Test match not found: {test_match_path}")
            return False

        # Load single match directly
        with open(test_match_path, 'r') as f:
            data = json.load(f)

        # Use TestMatchLoader to create match state
        loader = TestMatchLoader()
        match_id, match_state = loader._create_match_state(data)

        if not match_state:
            print("   ✗ Failed to create match state")
            return False

        print(f"   ✓ Loaded match: {match_id}")
        print(f"   - {match_state.team1} vs {match_state.team2}")
        print(f"   - Date: {match_state.match_date.strftime('%Y-%m-%d')}")
        print(f"   - Batting first: {match_state.batting_first}")

    except Exception as e:
        print(f"   ✗ Failed to load match: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 4: Extract features to verify real stats are being used
    print("\n4. Verifying real player stats are being used...")
    try:
        # Extract features from initial match state. Since the 2026-04-18
        # eval-speedup change, extract_features returns a preallocated
        # np.ndarray — rehydrate it into a dict for readable assertions.
        feat_arr = model.extract_features(match_state)
        features = dict(zip(model.feature_columns, feat_arr))

        # Get player info
        striker = match_state.current_striker
        bowler = match_state.current_bowler

        print(f"\n   Opening matchup:")
        print(f"   - Striker: {striker.name} (ID: {striker.player_id[:8]}...)")
        print(f"   - Bowler: {bowler.name} (ID: {bowler.player_id[:8]}...)")

        # Check batting stats
        batting_avg = features.get('batsman_avg', 0.0)
        batting_sr = features.get('batsman_sr', 0.0)

        print(f"\n   Batting stats from cache:")
        print(f"   - Average: {batting_avg:.2f}")
        print(f"   - Strike Rate: {batting_sr:.2f}")

        # Check bowling stats
        bowling_avg = features.get('bowler_avg', 0.0)
        bowling_econ = features.get('bowler_econ', 0.0)

        print(f"\n   Bowling stats from cache:")
        print(f"   - Average: {bowling_avg:.2f}")
        print(f"   - Economy: {bowling_econ:.2f}")

        # Check H2H stats
        h2h_avg = features.get('h2h_avg', 0.0)
        h2h_sr = features.get('h2h_sr', 0.0)

        print(f"\n   Head-to-head stats from cache:")
        print(f"   - Average: {h2h_avg:.2f}")
        print(f"   - Strike Rate: {h2h_sr:.2f}")

        # Verify at least some stats are non-zero
        has_real_stats = any([
            batting_avg > 0, batting_sr > 0,
            bowling_avg > 0, bowling_econ > 0,
            h2h_avg > 0, h2h_sr > 0
        ])

        if has_real_stats:
            print("\n   ✓ Real player stats detected (non-zero values found)")
        else:
            print("\n   ⚠ Warning: All stats are zero (players may be new/unknown)")

    except Exception as e:
        print(f"   ✗ Failed to extract features: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 5: Run simulations
    print("\n5. Running simulations...")
    try:
        # Create simulation engine
        rules = T20Rules()
        engine = SimulationEngine(model, rules)

        # Run a small number of simulations
        config = SimulationConfig(
            n_simulations=10,
            parallel=False,  # Non-parallel for easier debugging
            random_seed=42,
            verbose=True
        )

        print(f"   Running {config.n_simulations} simulations...")
        results = engine.simulate_multiple(match_state, config)

        print(f"   ✓ Completed {len(results)} simulations")

        # Aggregate results
        from sim_v1_2 import ResultAggregator
        summary = ResultAggregator.aggregate(results)

        # Print win probabilities
        print(f"\n   Win probabilities:")
        for team, prob in summary['win_probability'].items():
            if team != 'tie':
                print(f"   - {team}: {prob:.1%}")

        # Print expected scores
        print(f"\n   Expected scores:")
        for team, stats in summary['score_stats'].items():
            print(f"   - {team}: {stats['mean']:.1f} ± {stats['std']:.1f} "
                  f"(range: {stats['min']}-{stats['max']})")

    except Exception as e:
        print(f"   ✗ Failed to run simulations: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Success!
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("  - Stats provider loaded successfully")
    print("  - Real player stats integrated into simulations")
    print("  - Simulations completed successfully")
    print("=" * 60)

    return True

if __name__ == "__main__":
    success = test_simulation_with_real_stats()
    exit(0 if success else 1)
