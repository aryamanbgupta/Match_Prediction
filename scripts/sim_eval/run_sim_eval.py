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

from sim_v1_2 import SimulationEngine, XGBoostModel, T20Rules, XGBoostModelV2, LSTMModelV1, MLPModelV1, MLPModelV2, TransformerModelV1, LLMModelV1
from stats_provider import StatsProvider
from player_metadata import PlayerMetadataProvider
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
    parser.add_argument('--model-type', type=str, default='xgboost', choices=['xgboost', 'lstm', 'mlp', 'mlp_v2', 'transformer', 'llm'],
                       help='Model type to use (xgboost, lstm, mlp, mlp_v2, transformer, or llm)')
    parser.add_argument('--model-version', type=str, default='v3', choices=['v2', 'v3'],
                       help='Model version to use (v2=legacy 29 features, v3=46+ features with player metadata)')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model (overrides --model-version)')
    parser.add_argument('--batter-encoder', type=str, default=None,
                       help='Path to batter encoder (overrides --model-version)')
    parser.add_argument('--bowler-encoder', type=str, default=None,
                       help='Path to bowler encoder (overrides --model-version)')
    parser.add_argument('--n-sims', type=int, default=1000,
                       help='Number of simulations per match')
    parser.add_argument('--max-matches', type=int, default=None,
                       help='Maximum number of matches to evaluate (for testing)')
    parser.add_argument('--parallel', action='store_true', default=False,
                       help='Enable parallel processing for simulations')
    parser.add_argument('--create-example', action='store_true',
                       help='Create example betting odds file')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save detailed results JSON (auto-saves if provided)')
    parser.add_argument('--mlx', action='store_true',
                       help='Use MLX backend for transformer model (Apple Silicon only, faster on Mac)')

    args = parser.parse_args()

    # Set model paths based on version (can be overridden by explicit args)
    if args.model_version == 'v3':
        default_model = 'models/xgb_v3/xgboost_model_v3.pkl'
        default_batter_encoder = 'models/xgb_v3/batter_encoder_v3.pkl'
        default_bowler_encoder = 'models/xgb_v3/bowler_encoder_v3.pkl'
        default_feature_columns = 'models/xgb_v3/feature_columns_v3.txt'
        stats_version = 'v3'
    else:
        default_model = 'models/xgb/xgboost_model_v2.pkl'
        default_batter_encoder = 'models/xgb/batter_encoder_v2.pkl'
        default_bowler_encoder = 'models/xgb/bowler_encoder_v2.pkl'
        default_feature_columns = 'models/xgb/feature_columns_v2.txt'
        stats_version = 'v2'

    model_path = args.model or default_model
    batter_encoder_path = args.batter_encoder or default_batter_encoder
    bowler_encoder_path = args.bowler_encoder or default_bowler_encoder
    feature_columns_path = default_feature_columns
    
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

    # Only check XGBoost model path if using XGBoost model type
    if args.model_type == 'xgboost' and not Path(model_path).exists():
        print(f"Error: Model file not found: {model_path}")
        print(f"  (Using --model-version={args.model_version})")
        return

    print("Cricket Match-Level Evaluation")
    print("=" * 60)
    print(f"Model type: {args.model_type}")
    print(f"Model version: {args.model_version}")

    # Load player stats cache (chunked format)
    print("\nLoading player stats cache...")
    try:
        stats_provider = StatsProvider('models', version=stats_version)
        print(f"✓ Stats cache loaded successfully (version: {stats_version})")
    except Exception as e:
        print(f"Warning: Could not load stats cache: {e}")
        print("Continuing without stats cache (will use zeros)")
        stats_provider = None

    # Load player metadata (needed for LSTM and v3 features)
    player_metadata = None
    if args.model_type == 'lstm' or args.model_version == 'v3':
        print("\nLoading player metadata...")
        try:
            player_metadata = PlayerMetadataProvider('data/all_players_enriched.csv')
            print("✓ Player metadata loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load player metadata: {e}")

    # Load model and encoders
    if args.model_type == 'lstm':
        print(f"\nLoading LSTM model...")
        try:
            model = LSTMModelV1(
                model_path='models/lstm_v1/lstm_model_v1.pt',
                batter_encoder_path='models/lstm_v1/batter_encoder_v1.pkl',
                bowler_encoder_path='models/lstm_v1/bowler_encoder_v1.pkl',
                feature_columns_path='models/lstm_v1/feature_columns_v1.txt',
                scaler_path='models/lstm_v1/feature_scaler_v1.pkl',
                config_path='models/lstm_v1/lstm_config_v1.json',
                stats_provider=stats_provider,
                player_metadata=player_metadata,
                matchup_encoder_path='models/lstm_v1/matchup_encoder_v1.pkl',
                venue_encoder_path='models/lstm_v1/venue_encoder_v1.pkl',
                window_size=10,
                device='cpu'
            )
            print("✓ LSTM model loaded successfully")
        except Exception as e:
            print(f"Error loading LSTM model: {e}")
            print("Using dummy model for demonstration")
            from sim_v1_2 import DummyModel
            model = DummyModel()
    elif args.model_type == 'mlp':
        print(f"\nLoading MLP model...")
        try:
            model = MLPModelV1(
                model_path='models/mlp_v1/mlp_model_v1.pt',
                batter_encoder_path='models/mlp_v1/batter_encoder_v1.pkl',
                bowler_encoder_path='models/mlp_v1/bowler_encoder_v1.pkl',
                feature_columns_path='models/mlp_v1/feature_columns_v1.txt',
                scaler_path='models/mlp_v1/feature_scaler_v1.pkl',
                config_path='models/mlp_v1/mlp_config_v1.json',
                stats_provider=stats_provider,
                player_metadata=player_metadata,
                device='cpu'
            )
            print("✓ MLP model loaded successfully")
        except Exception as e:
            print(f"Error loading MLP model: {e}")
            print("Using dummy model for demonstration")
            from sim_v1_2 import DummyModel
            model = DummyModel()
    elif args.model_type == 'mlp_v2':
        print(f"\nLoading MLP v2 model with embeddings...")
        try:
            model = MLPModelV2(
                model_path='models/mlp_v2/mlp_model_v2.pt',
                batter_encoder_path='models/mlp_v2/batter_encoder_v2.pkl',
                bowler_encoder_path='models/mlp_v2/bowler_encoder_v2.pkl',
                continuous_columns_path='models/mlp_v2/continuous_columns_v2.txt',
                categorical_columns_path='models/mlp_v2/categorical_columns_v2.json',
                scaler_path='models/mlp_v2/feature_scaler_v2.pkl',
                config_path='models/mlp_v2/mlp_config_v2.json',
                stats_provider=stats_provider,
                player_metadata=player_metadata,
                device='cpu'
            )
            print("✓ MLP v2 model loaded successfully")
        except Exception as e:
            print(f"Error loading MLP v2 model: {e}")
            print("Using dummy model for demonstration")
            from sim_v1_2 import DummyModel
            model = DummyModel()
    elif args.model_type == 'transformer':
        backend = "MLX (unified memory)" if args.mlx else "PyTorch"
        print(f"\nLoading Transformer model ({backend})...")
        try:
            model = TransformerModelV1(
                model_path='models/transformer_v1/transformer_model_v1.pt',
                batter_encoder_path='models/transformer_v1/batter_encoder_v1.pkl',
                bowler_encoder_path='models/transformer_v1/bowler_encoder_v1.pkl',
                feature_columns_path='models/transformer_v1/feature_columns_v1.txt',
                scaler_path='models/transformer_v1/feature_scaler_v1.pkl',
                config_path='models/transformer_v1/transformer_config_v1.json',
                stats_provider=stats_provider,
                player_metadata=player_metadata,
                matchup_encoder_path='models/transformer_v1/matchup_encoder_v1.pkl',
                venue_encoder_path='models/transformer_v1/venue_encoder_v1.pkl',
                max_seq_len=120,
                device='cpu',
                use_mlx=args.mlx
            )
            print(f"✓ Transformer model loaded successfully ({backend}, 120-ball context)")
        except Exception as e:
            print(f"Error loading Transformer model: {e}")
            print("Using dummy model for demonstration")
            from sim_v1_2 import DummyModel
            model = DummyModel()
    elif args.model_type == 'llm':
        print(f"\nLoading LLM model (Qwen 1.5-1.8B with LoRA)...")
        print("NOTE: LLM model requires GPU for reasonable inference speed.")

        # Check for GPU
        import torch
        if not torch.cuda.is_available():
            print("\n" + "="*60)
            print("ERROR: LLM model requires CUDA GPU but none is available.")
            print("Please run on a machine with CUDA GPU (e.g., cloud GPU instance).")
            print("="*60 + "\n")
            return

        try:
            model = LLMModelV1(
                checkpoint_path='models/llm_v1',
                device='cuda'
            )
            print("✓ LLM model loaded successfully on GPU")
        except Exception as e:
            print(f"Error loading LLM model: {e}")
            print("Check that:")
            print("  1. models/llm_v1/ directory exists with checkpoint files")
            print("  2. transformers and peft packages are installed")
            print("  3. GPU has sufficient memory (~4GB)")
            return
    else:
        print(f"\nLoading XGBoost model from {model_path}...")
        try:
            model = XGBoostModelV2(
                model_path=model_path,
                batter_encoder_path=batter_encoder_path,
                bowler_encoder_path=bowler_encoder_path,
                feature_columns_path=feature_columns_path,
                stats_provider=stats_provider,
                player_metadata=player_metadata
            )
            print(f"✓ XGBoost model loaded successfully ({args.model_version})")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using dummy model for demonstration")
            from sim_v1_2 import DummyModel
            model = DummyModel()

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

    # Limit matches if requested (for testing)
    if args.max_matches:
        matches = matches[:args.max_matches]
        print(f"✓ Limiting to first {args.max_matches} matches for testing")

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
        n_simulations=args.n_sims,
        parallel=args.parallel
    )
    
    # Run evaluation
    results = evaluator.evaluate_all(matches, odds_lookup)
    
    # Print summary
    print_evaluation_summary(results)
    
    # Save detailed results
    save_results = True
    from datetime import datetime as _dt
    _timestamp = _dt.now().strftime('%Y%m%d_%H%M%S')

    if args.output_dir:
        # Auto-save to specified directory with timestamp
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        output_path = f"{args.output_dir}/{args.model_type}_{_timestamp}.json"
    else:
        # Auto-save to default path with timestamp
        output_path = f"match_evaluation_results_{args.model_type}_{_timestamp}.json"
    
    if save_results:
        import json
        
        # Convert results to JSON-serializable format
        results_dict = {
            'summary': {
                'model_type': args.model_type,
                'n_matches': results.n_matches,
                'avg_log_loss': results.avg_log_loss,
                'avg_brier_score': results.avg_brier_score,
                'avg_edge': results.avg_edge,
                'avg_signed_edge': results.avg_signed_edge,
                # Flat staking
                'flat_betting_total_pnl': results.total_pnl,
                'flat_betting_roi_pct': results.roi,
                'flat_betting_win_rate': results.win_rate,
                'flat_betting_bets_placed': results.bets_placed,
                'flat_betting_sharpe': results.sharpe_ratio_flat,
                # Full Kelly
                'full_kelly_total_pnl': results.full_kelly_total_pnl,
                'full_kelly_roi_pct': results.full_kelly_roi,
                'full_kelly_win_rate': results.full_kelly_win_rate,
                'full_kelly_bets_placed': results.full_kelly_bets_placed,
                'full_kelly_sharpe': results.sharpe_ratio_full_kelly,
                # Fractional Kelly
                'frac_kelly_total_pnl': results.fractional_kelly_total_pnl,
                'frac_kelly_roi_pct': results.fractional_kelly_roi,
                'frac_kelly_win_rate': results.fractional_kelly_win_rate,
                'frac_kelly_bets_placed': results.fractional_kelly_bets_placed,
                'frac_kelly_sharpe': results.sharpe_ratio_fractional_kelly,
                # Expected value
                'total_expected_value': results.total_expected_value,
                # Metadata
                'total_time': results.total_simulation_time
            },
            'matches': []
        }

        for match in results.match_results:
            results_dict['matches'].append({
                'match_id': match.match_id,
                'teams': [match.team1, match.team2],
                'actual_winner': match.actual_winner,
                'simulated_prob': match.simulated_win_prob,
                'market_prob': match.market_win_prob,
                'market_odds': match.market_odds,
                'edge': match.edge,
                'log_loss': match.log_loss,
                'brier_score': match.brier_score,
                'realized_pnl': match.realized_pnl,
                'expected_value': match.expected_value,
                'full_kelly_fraction': match.full_kelly_fraction,
                'full_kelly_pnl': match.full_kelly_pnl,
                'fractional_kelly_pnl': match.fractional_kelly_pnl,
            })
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()