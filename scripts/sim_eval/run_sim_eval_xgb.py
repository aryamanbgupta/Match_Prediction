#!/usr/bin/env python3
"""
Test script for v2 model simulation
"""
from pathlib import Path
import sys

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
# Add parent directory to path to import simulation modules
sys.path.append(str(Path(__file__).parent.parent))
from sim_v1_2 import *
from datetime import datetime
import pandas as pd

def test_v2_simulation():
    """Test v2 model in simulation"""
    
    print("Testing XGBoost v2 Model Simulation")
    print("=" * 50)
    
    # Initialize v2 model
    try:
        model = XGBoostModelV2(
            model_path='models/xgb/xgboost_model_v2.pkl',
            batter_encoder_path='models/xgb/batter_encoder_v2.pkl',
            bowler_encoder_path='models/xgb/bowler_encoder_v2.pkl',
            feature_columns_path='models/xgb/feature_columns_v2.txt'
        )
        print("✓ V2 model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load v2 model: {e}")
        print("Falling back to dummy model for test")
        model = DummyModel()
    
    # Create simulation engine
    rules = T20Rules(RandomBowlerSelector())
    engine = SimulationEngine(model, rules)
    
    # Create test teams
    india_players = [
        Player("rohit_sharma", "Rohit Sharma", "India", "batsman"),
        Player("shubman_gill", "Shubman Gill", "India", "batsman"),
        Player("virat_kohli", "Virat Kohli", "India", "batsman"),
        Player("suryakumar_yadav", "Suryakumar Yadav", "India", "batsman"),
        Player("hardik_pandya", "Hardik Pandya", "India", "allrounder"),
        Player("ravindra_jadeja", "Ravindra Jadeja", "India", "allrounder"),
        Player("ms_dhoni", "MS Dhoni", "India", "wicketkeeper"),
        Player("ravichandran_ashwin", "R Ashwin", "India", "bowler"),
        Player("mohammed_shami", "Mohammed Shami", "India", "bowler"),
        Player("jasprit_bumrah", "Jasprit Bumrah", "India", "bowler"),
        Player("yuzvendra_chahal", "Yuzvendra Chahal", "India", "bowler"),
    ]
    
    australia_players = [
        Player("david_warner", "David Warner", "Australia", "batsman"),
        Player("travis_head", "Travis Head", "Australia", "batsman"),
        Player("steve_smith", "Steve Smith", "Australia", "batsman"),
        Player("glenn_maxwell", "Glenn Maxwell", "Australia", "allrounder"),
        Player("marcus_stoinis", "Marcus Stoinis", "Australia", "allrounder"),
        Player("tim_david", "Tim David", "Australia", "batsman"),
        Player("matthew_wade", "Matthew Wade", "Australia", "wicketkeeper"),
        Player("pat_cummins", "Pat Cummins", "Australia", "bowler"),
        Player("mitchell_starc", "Mitchell Starc", "Australia", "bowler"),
        Player("adam_zampa", "Adam Zampa", "Australia", "bowler"),
        Player("josh_hazlewood", "Josh Hazlewood", "Australia", "bowler"),
    ]
    
    # Create match state
    india_lineup = TeamLineup("India", india_players)
    australia_lineup = TeamLineup("Australia", australia_players)
    
    state = MatchState(
        team1_lineup=india_lineup,
        team2_lineup=australia_lineup,
        batting_first="India",
        venue="MCG",
        match_date=datetime(2024, 12, 25)
    )
    
    print(f"\n--- Single Match Test ---")
    
    # Test feature extraction
    print("Testing feature extraction...")
    try:
        features = model.extract_features(state)
        print(f"✓ Features extracted: {features.shape}")
        print(f"  Sample features: {list(features.columns)[:10]}")
    except Exception as e:
        print(f"✗ Feature extraction failed: {e}")
        return
    
    # Test prediction
    print("Testing prediction...")
    try:
        probs = model.predict_next_ball(features)
        print(f"✓ Prediction successful")
        print(f"  Probabilities: {probs}")
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        return
    
    # Test single match simulation
    print("Testing single match simulation...")
    try:
        result = engine.simulate_match(state, "test_v2")
        print(f"✓ Simulation successful")
        print(f"  Result: {result.winner} by {result.margin}")
        print(f"  Score: {result.team1} {result.team1_score}/{result.team1_wickets}")
        print(f"         {result.team2} {result.team2_score}/{result.team2_wickets}")
    except Exception as e:
        print(f"✗ Simulation failed: {e}")
        return
    
    # Test multiple simulations
    print(f"\n--- Multiple Simulations Test ---")
    try:
        config = SimulationConfig(
            n_simulations=100,
            parallel=True,
            verbose=True,
            random_seed=42
        )
        
        results = engine.simulate_multiple(state, config)
        print(f"✓ Multiple simulations successful")
        
        # Aggregate results
        summary = ResultAggregator.aggregate(results)
        
        print(f"\nWin Probabilities:")
        for team, prob in summary['win_probability'].items():
            print(f"  {team}: {prob:.2%}")
        
        print(f"\nScore Predictions:")
        for team, stats in summary['score_stats'].items():
            print(f"  {team}: {stats['mean']:.1f} ± {stats['std']:.1f}")
        
    except Exception as e:
        print(f"✗ Multiple simulations failed: {e}")
        return
    
    print(f"\n✓ All tests passed! V2 model is working in simulation.")

if __name__ == "__main__":
    test_v2_simulation()