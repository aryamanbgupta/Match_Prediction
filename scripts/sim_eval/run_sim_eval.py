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

from sim_v1_2 import SimulationEngine, XGBoostModel, T20Rules, XGBoostModelV2, LSTMModelV1, MLPModelV1, MLPModelV2, TransformerModelV1, LLMModelV1, EmpiricalBowlerSelector, RandomBowlerSelector
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
    parser.add_argument('--model-version', type=str, default='v3', choices=['v3'],
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
    parser.add_argument('--min-volume', type=float, default=None,
                       help='Drop matches whose polymarket_volume_usd is below this threshold. '
                            'Use for liquidity-sliced eval (e.g., 50000 / 100000). '
                            'Default: None (no filter; preserves non-polymarket odds files).')
    parser.add_argument('--bootstrap-resamples', type=int, default=1000,
                       help='Number of bootstrap resamples for ROI / log-loss CIs (default: 1000)')

    # Calibration arguments
    parser.add_argument('--calibrate', action='store_true',
                       help='Enable match-level LOOCV calibration (Platt or isotonic)')
    parser.add_argument('--calibration-method', type=str, default='platt', choices=['platt', 'isotonic'],
                       help='Match-level calibration method (default: platt)')
    parser.add_argument('--ball-calibrate', action='store_true',
                       help='Enable ball-level calibration (per-class isotonic on validation data)')
    parser.add_argument('--ball-calibrate-data', type=str, default=None,
                       help='Path to validation parquet for fitting ball-level calibrator (default: auto-detect)')
    parser.add_argument('--ball-diagnostics', action='store_true',
                       help='Run ball-level ECE diagnostics (read-only, no correction)')
    parser.add_argument('--save-calibrator', type=str, default=None,
                       help='Save fitted match-level calibrator to PATH for reuse')
    parser.add_argument('--load-calibrator', type=str, default=None,
                       help='Load pre-fitted match-level calibrator from PATH')
    parser.add_argument('--bowler-selector', choices=['empirical', 'random'],
                       default='empirical',
                       help='Bowler selection strategy. Default = empirical (phase-aware).')
    parser.add_argument('--bowler-usage-path',
                       default='models/bowler_phase_usage.json',
                       help='Usage prior JSON for EmpiricalBowlerSelector.')

    args = parser.parse_args()

    # Set model paths based on version (can be overridden by explicit args).
    # Only v3 is supported — v2 artifacts (models/xgb/) were deleted in the
    # 2026-04-26 cleanup. The arg is retained for forward-compat with future
    # versions, but unknown values fall back to v3 paths.
    if args.model_version != 'v3':
        print(f"warning: --model-version={args.model_version!r} is unsupported; "
              f"v2 artifacts were deleted. Falling back to v3 paths.",
              file=sys.stderr)
    default_model = 'models/xgb_v3/xgboost_model_v3.pkl'
    default_batter_encoder = 'models/xgb_v3/batter_encoder_v3.pkl'
    default_bowler_encoder = 'models/xgb_v3/bowler_encoder_v3.pkl'
    default_feature_columns = 'models/xgb_v3/feature_columns_v3.txt'
    stats_version = 'v3'

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
    if args.ball_calibrate:
        print(f"Ball-level calibration: enabled (isotonic per-class)")
    if args.calibrate:
        print(f"Match-level calibration: {args.calibration_method} (LOOCV)")

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

    # Ball-level calibration is loaded after model init (needs raw XGBoost model)
    ball_calibrator = None
    _ball_cal_data_path = None
    if args.ball_calibrate or args.ball_diagnostics:
        if args.ball_calibrate_data:
            _ball_cal_data_path = args.ball_calibrate_data
        else:
            # Only v3 parquet exists post-2026-04-26; v2 was deleted.
            _ball_cal_data_path = 'data/xgb_data_v3/cricket_data_v3_validation.parquet'

        if not Path(_ball_cal_data_path).exists():
            print(f"\nWarning: Validation data not found at {_ball_cal_data_path}")
            print("Ball-level calibration/diagnostics disabled. Use --ball-calibrate-data to specify path.")
            _ball_cal_data_path = None

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
        # Fit ball-level calibrator before loading the wrapper model
        if _ball_cal_data_path and args.model_type == 'xgboost':
            from calibration import BallLevelCalibrationDiagnostics, fit_ball_calibrator_from_data
            feature_columns = [line.strip() for line in open(feature_columns_path) if line.strip()]
            raw_xgb_model = joblib.load(model_path)
            encoder_dir = str(Path(model_path).parent)

            if args.ball_calibrate:
                print("\nFitting ball-level calibrator...")
                try:
                    ball_calibrator = fit_ball_calibrator_from_data(
                        model=raw_xgb_model,
                        data_path=_ball_cal_data_path,
                        feature_columns=feature_columns,
                        encoder_dir=encoder_dir
                    )
                    print(f"✓ Ball-level calibrator fitted")
                except Exception as e:
                    print(f"Warning: Could not fit ball-level calibrator: {e}")

            if args.ball_diagnostics:
                print("\nRunning ball-level calibration diagnostics...")
                try:
                    diagnostics = BallLevelCalibrationDiagnostics(
                        raw_xgb_model, _ball_cal_data_path, feature_columns,
                        encoder_dir=encoder_dir
                    )
                    diagnostics.compute_all()
                    diagnostics.print_summary()
                except Exception as e:
                    print(f"Warning: Ball-level diagnostics failed: {e}")

        print(f"\nLoading XGBoost model from {model_path}...")
        try:
            model = XGBoostModelV2(
                model_path=model_path,
                batter_encoder_path=batter_encoder_path,
                bowler_encoder_path=bowler_encoder_path,
                feature_columns_path=feature_columns_path,
                stats_provider=stats_provider,
                player_metadata=player_metadata,
                ball_calibrator=ball_calibrator
            )
            print(f"✓ XGBoost model loaded successfully ({args.model_version})")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using dummy model for demonstration")
            from sim_v1_2 import DummyModel
            model = DummyModel()

    # Create simulation engine
    if args.bowler_selector == 'empirical':
        selector = EmpiricalBowlerSelector(usage_path=args.bowler_usage_path)
    else:
        selector = RandomBowlerSelector()
    print(f"Bowler selector: {args.bowler_selector}")
    rules = T20Rules(selector)
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
    odds_lookup = BettingOddsLoader.load_odds(args.odds, min_volume=args.min_volume)

    if not odds_lookup:
        print("No odds loaded!")
        return

    # Create evaluator
    evaluator = MatchLevelEvaluator(
        model=model,
        simulation_engine=engine,
        n_simulations=args.n_sims,
        parallel=args.parallel,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    
    # Run evaluation (with or without match-level calibration)
    if args.calibrate or args.load_calibrator:
        print(f"\nMatch-level calibration: {args.calibration_method} (LOOCV)")
        results = evaluator.evaluate_all_with_calibration(
            matches, odds_lookup,
            calibration_method=args.calibration_method
        )
        # Save calibrator if requested
        if args.save_calibrator and hasattr(results, '_calibrator'):
            results._calibrator.save(args.save_calibrator)
            print(f"Calibrator saved to {args.save_calibrator}")
    else:
        results = evaluator.evaluate_all(matches, odds_lookup)

    # Print summary
    print_evaluation_summary(results)
    
    # Save detailed results
    save_results = True
    from datetime import datetime as _dt
    _timestamp = _dt.now().strftime('%Y%m%d_%H%M%S')

    # Slice tag for sliced-eval workflows; encoded into both the saved
    # filename and the JSON payload so downstream comparison tools can
    # group runs by liquidity bucket without parsing filenames.
    if args.min_volume is None:
        slice_tag = "all"
    else:
        slice_tag = f"min_volume_{int(args.min_volume)}"

    if args.output_dir:
        # Auto-save to specified directory with timestamp
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        output_path = f"{args.output_dir}/{args.model_type}_{slice_tag}_{_timestamp}.json"
    else:
        # Auto-save to default path with timestamp
        output_path = f"match_evaluation_results_{args.model_type}_{slice_tag}_{_timestamp}.json"
    
    if save_results:
        import json
        
        # Convert results to JSON-serializable format
        results_dict = {
            'summary': {
                'model_type': args.model_type,
                'slice': slice_tag,
                'min_volume': args.min_volume,
                'n_matches_evaluated': results.n_matches,
                'n_matches': results.n_matches,
                'avg_log_loss': results.avg_log_loss,
                'avg_log_loss_ci_low': results.avg_log_loss_ci_low,
                'avg_log_loss_ci_high': results.avg_log_loss_ci_high,
                'avg_brier_score': results.avg_brier_score,
                'avg_edge': results.avg_edge,
                'avg_signed_edge': results.avg_signed_edge,
                # Flat staking
                'flat_betting_total_pnl': results.total_pnl,
                'flat_betting_roi_pct': results.roi,
                'flat_betting_roi_ci_low': results.flat_roi_ci_low,
                'flat_betting_roi_ci_high': results.flat_roi_ci_high,
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
                'total_time': results.total_simulation_time,
                # Calibration (if applied)
                'calibration_method': results.calibration_method,
                'pre_calibration_ece': results.pre_calibration_ece,
                'post_calibration_ece': results.post_calibration_ece,
                'pre_calibration_log_loss': results.pre_calibration_log_loss,
                'post_calibration_log_loss': results.post_calibration_log_loss,
                'pre_calibration_brier': results.pre_calibration_brier,
                'post_calibration_brier': results.post_calibration_brier,
                'ball_calibration_enabled': args.ball_calibrate,
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