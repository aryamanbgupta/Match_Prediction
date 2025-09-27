#!/usr/bin/env python3
"""
Main script to run match-level evaluation against betting odds

Usage:
    python evaluate_matches.py --test-dir data/test_matches --odds data/betting_odds.json
"""

import argparse
import joblib
from pathlib import Path
import sys

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
# Add parent directory to path to import simulation modules
sys.path.append(str(Path(__file__).parent.parent))

from sim_v1_2 import SimulationEngine, XGBoostModel, T20Rules, XGBoostModelV2
from sim_eval.loaders import TestMatchLoader, BettingOddsLoader
from sim_eval.match_evaluator import MatchLevelEvaluator, print_evaluation_summary


def create_example_odds_file():
    """Create an example betting odds file for reference"""
    example = {
        "matches": [
            {
                "match_id": "2024-01-15_India_Australia_MCG",
                "date": "2024-01-15",
                "team1": "India", 
                "team2": "Australia",
                "venue": "MCG",
                "odds": {
                    "winner": {
                        "India": 2.10,
                        "Australia": 1.75,
                        "timestamp": "2024-01-14T10:00:00Z"
                    }
                }
            },
            {
                "match_id": "2024-01-20_England_Pakistan_Lords",
                "date": "2024-01-20",
                "team1": "England",
                "team2": "Pakistan", 
                "venue": "Lords",
                "odds": {
                    "winner": {
                        "England": 1.65,
                        "Pakistan": 2.35,
                        "timestamp": "2024-01-19T15:00:00Z"
                    }
                }
            }
        ]
    }
    
    import json
    with open('example_betting_odds.json', 'w') as f:
        json.dump(example, f, indent=2)
    
    print("Created example_betting_odds.json")


def main():
    parser = argparse.ArgumentParser(description='Evaluate cricket match predictions against betting odds')
    parser.add_argument('--test-dir', type=str, default='data/test_matches',
                       help='Directory containing test match JSON files')
    parser.add_argument('--odds', type=str, default='data/betting_odds.json',
                       help='JSON file containing betting odds')
    parser.add_argument('--model', type=str, default='models/gradient_boosting_model.pkl',
                       help='Path to trained model')
    parser.add_argument('--batter-encoder', type=str, default='models/batter_encoder.pkl',
                       help='Path to batter encoder')
    parser.add_argument('--bowler-encoder', type=str, default='models/bowler_encoder.pkl',
                       help='Path to bowler encoder')
    parser.add_argument('--n-sims', type=int, default=1000,
                       help='Number of simulations per match')
    parser.add_argument('--create-example', action='store_true',
                       help='Create example betting odds file')
    
    args = parser.parse_args()
    
    # Create example file if requested
    if args.create_example:
        create_example_odds_file()
        return
    
    # Check if files exist
    if not Path(args.test_dir).exists():
        print(f"Error: Test directory not found: {args.test_dir}")
        return
    
    if not Path(args.odds).exists():
        print(f"Error: Odds file not found: {args.odds}")
        print("Run with --create-example to create an example odds file")
        return
    
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        return
    
    print("Cricket Match-Level Evaluation")
    print("=" * 60)
    
    # Load model and encoders
    print("\nLoading model...")
    try:
        model = XGBoostModelV2(
            model_path='models/xgb/xgboost_model_v2.pkl',
            batter_encoder_path='models/xgb/batter_encoder_v2.pkl',
            bowler_encoder_path='models/xgb/bowler_encoder_v2.pkl',
            feature_columns_path='models/xgb/feature_columns_v2.txt'
        )
        print("✓ V2 model loaded successfully")

        # model = XGBoostModel(
        #     model_path=args.model,
        #     batter_encoder_path=args.batter_encoder,
        #     bowler_encoder_path=args.bowler_encoder
        # )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using dummy model for demonstration")
        from sim_v1_2 import DummyModel
        model = DummyModel()
    # print("Using dummy model for demonstration")
    # from sim_v1_2 import DummyModel
    # model = DummyModel()

    # Create simulation engine
    rules = T20Rules()
    engine = SimulationEngine(model, rules)
    
    # Load test matches
    print("\nLoading test matches...")
    match_loader = TestMatchLoader()
    matches = match_loader.load_matches(args.test_dir)
    
    if not matches:
        print("No matches loaded!")
        return
    
    # Load betting odds
    print("\nLoading betting odds...")
    odds_lookup = BettingOddsLoader.load_odds(args.odds)
    
    if not odds_lookup:
        print("No odds loaded!")
        return
    
    # Create evaluator
    evaluator = MatchLevelEvaluator(
        model=model,
        simulation_engine=engine,
        n_simulations=args.n_sims
    )
    
    # Run evaluation
    results = evaluator.evaluate_all(matches, odds_lookup)
    
    # Print summary
    print_evaluation_summary(results)
    
    # Optional: Save detailed results
    print("\n\nWould you like to save detailed results? (y/n): ", end='')
    if input().lower() == 'y':
        import json
        
        # Convert results to JSON-serializable format
        results_dict = {
            'summary': {
                'n_matches': results.n_matches,
                'avg_log_loss': results.avg_log_loss,
                'avg_brier_score': results.avg_brier_score,
                'avg_edge': results.avg_edge,
                'profitable_bets': results.profitable_bets,
                'total_time': results.total_simulation_time
            },
            'matches': []
        }
        
        for match in results.match_results:
            results_dict['matches'].append({
                'match_id': match.match_id,
                'teams': [match.team1, match.team2],
                'simulated_prob': match.simulated_win_prob,
                'market_prob': match.market_win_prob,
                'edge': match.edge,
                'log_loss': match.log_loss,
                'brier_score': match.brier_score
            })
        
        with open('match_evaluation_results.json', 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print("Results saved to match_evaluation_results.json")


if __name__ == "__main__":
    main()