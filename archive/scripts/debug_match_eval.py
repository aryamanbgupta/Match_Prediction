#!/usr/bin/env python3
"""
Debug script specifically for the match evaluation pipeline
Tests the data flow from JSON to simulation
"""

import json
import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from sim_v1_2 import MatchState, SimulationEngine, XGBoostModel, T20Rules
from sim_eval.loaders import TestMatchLoader, BettingOddsLoader
from sim_eval.match_evaluator import MatchLevelEvaluator


def debug_player_extraction():
    """Debug player extraction from real match JSON"""
    print("="*60)
    print("DEBUGGING PLAYER EXTRACTION")
    print("="*60)
    
    # Load a match that has odds
    test_file = Path("data/betting_test/1342425.json")  # Use a file we know has odds
    if not test_file.exists():
        # Try to find any match file
        test_files = list(Path("data/betting_test").glob("*.json"))
        if test_files:
            test_file = test_files[0]
    
    print(f"\nUsing test file: {test_file}")
    
    with open(test_file, 'r') as f:
        data = json.load(f)
    
    # Check player registry
    print("\n1. Checking player registry...")
    registry = data['info']['registry']['people']
    print(f"Registry type: {type(registry)}")
    print(f"Number of players: {len(registry)}")
    
    # Check types in registry
    print("\nSample registry entries:")
    for i, (name, player_id) in enumerate(list(registry.items())[:3]):
        print(f"  {name}: {player_id} (type: {type(player_id)})")
    
    # Test player extraction
    print("\n2. Testing player extraction...")
    loader = TestMatchLoader()
    match_id, match_state = loader._create_match_state(data)
    
    if match_state:
        print(f"✓ Match state created: {match_id}")
        
        # Check player details
        print("\n3. Checking extracted players...")
        print(f"Team 1: {match_state.team1}")
        for i, player in enumerate(match_state.team1_lineup.players[:3]):
            print(f"  [{i}] {player.name} (ID: {player.player_id}, type: {type(player.player_id)})")
        
        print(f"\nTeam 2: {match_state.team2}")
        for i, player in enumerate(match_state.team2_lineup.players[:3]):
            print(f"  [{i}] {player.name} (ID: {player.player_id}, type: {type(player.player_id)})")
    else:
        print("✗ Failed to create match state")
    
    return match_state


def debug_simulation_with_real_data(match_state):
    """Debug simulation with real player data"""
    print("\n" + "="*60)
    print("DEBUGGING SIMULATION WITH REAL DATA")
    print("="*60)
    
    # Load model
    try:
        model = XGBoostModel(
            model_path='models/gradient_boosting_model.pkl',
            batter_encoder_path='models/batter_encoder.pkl',
            bowler_encoder_path='models/bowler_encoder.pkl'
        )
        print("✓ Model loaded")
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return
    
    # Test state properties
    print("\n1. Checking initial state...")
    print(f"Batting first: {match_state.batting_first}")
    print(f"Current team index: {match_state.current_team_idx}")
    print(f"Striker index: {match_state.striker_idx} (type: {type(match_state.striker_idx)})")
    print(f"Non-striker index: {match_state.non_striker_idx} (type: {type(match_state.non_striker_idx)})")
    print(f"Bowler index: {match_state.bowler_idx} (type: {type(match_state.bowler_idx)})")
    print(f"Last bowler index: {match_state.last_bowler_idx} (type: {type(match_state.last_bowler_idx)})")
    
    # Test player access
    print("\n2. Testing player access...")
    try:
        striker = match_state.current_striker
        bowler = match_state.current_bowler
        print(f"✓ Striker: {striker.name if striker else 'None'}")
        print(f"✓ Bowler: {bowler.name if bowler else 'None'}")
    except Exception as e:
        print(f"✗ Error accessing players: {e}")
        import traceback
        traceback.print_exc()
    
    # Test available bowlers
    print("\n3. Testing available bowlers...")
    try:
        available = match_state.get_available_bowlers()
        print(f"✓ Available bowlers: {available}")
        print(f"  Types: {[type(x) for x in available]}")
    except Exception as e:
        print(f"✗ Error getting available bowlers: {e}")
        import traceback
        traceback.print_exc()
    
    # Test feature extraction
    print("\n4. Testing feature extraction...")
    try:
        features = model.extract_features(match_state)
        print(f"✓ Features extracted: {features}")
    except Exception as e:
        print(f"✗ Error extracting features: {e}")
        import traceback
        traceback.print_exc()
    
    # Test single ball simulation
    print("\n5. Testing single ball simulation...")
    try:
        rules = T20Rules()
        outcome, runs = rules.simulate_ball(match_state, model)
        print(f"✓ Ball simulated: {outcome}, {runs} runs")
    except Exception as e:
        print(f"✗ Error simulating ball: {e}")
        import traceback
        traceback.print_exc()


def debug_full_evaluation_pipeline():
    """Debug the full evaluation pipeline with a single match"""
    print("\n" + "="*60)
    print("DEBUGGING FULL EVALUATION PIPELINE")
    print("="*60)
    
    # Load matches and odds
    loader = TestMatchLoader()
    matches = loader.load_matches('data/betting_test')
    odds = BettingOddsLoader.load_odds('betting_odds_v2.json')
    
    # Find a match that has odds
    matched_match = None
    for match_id, match_state in matches:
        if match_id in odds:
            matched_match = (match_id, match_state)
            break
    
    if not matched_match:
        print("✗ No matches found with odds!")
        return
    
    match_id, match_state = matched_match
    print(f"\nEvaluating match: {match_id}")
    
    # Create evaluator with n_simulations=1
    try:
        model = XGBoostModel(
            model_path='models/gradient_boosting_model.pkl',
            batter_encoder_path='models/batter_encoder.pkl',
            bowler_encoder_path='models/bowler_encoder.pkl'
        )
    except:
        from sim_v1_2 import DummyModel
        model = DummyModel()
        print("Using dummy model")
    
    engine = SimulationEngine(model, T20Rules())
    evaluator = MatchLevelEvaluator(model, engine, n_simulations=1)
    
    # Try to evaluate single match
    print("\nEvaluating single match...")
    try:
        result = evaluator._evaluate_single_match(match_id, match_state, odds[match_id])
        print(f"✓ Evaluation successful!")
        print(f"  Winner probabilities: {result.simulated_win_prob}")
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


def check_type_consistency():
    """Check for type consistency issues in the codebase"""
    print("\n" + "="*60)
    print("CHECKING TYPE CONSISTENCY")
    print("="*60)
    
    # Test numpy array comparisons
    print("\n1. Testing numpy array behavior...")
    arr = np.zeros(2)
    print(f"Array: {arr}, dtype: {arr.dtype}")
    print(f"arr[0] > 5: {arr[0] > 5}")  # Should work
    
    # Test mixed type comparisons
    print("\n2. Testing mixed type comparisons...")
    try:
        result = "5" > 3
        print(f"'5' > 3: {result}")
    except TypeError as e:
        print(f"✓ Expected error: {e}")
    
    # Test index types
    print("\n3. Testing index types...")
    idx_int = 5
    idx_str = "5"
    last_idx = -1
    
    print(f"idx_int == last_idx: {idx_int == last_idx}")
    try:
        print(f"idx_str == last_idx: {idx_str == last_idx}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("Cricket Match Evaluation Pipeline Debugger")
    print("=" * 60)
    
    # Run debugging steps
    match_state = debug_player_extraction()
    
    if match_state:
        debug_simulation_with_real_data(match_state)
    
    debug_full_evaluation_pipeline()
    
    check_type_consistency()
    
    print("\n" + "="*60)
    print("DEBUGGING COMPLETE")
    print("="*60)