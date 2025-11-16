"""
Estimate expected runs model θ(o,w) using negative binomial regression.

This script:
1. Loads IPL ball-by-ball data
2. Fits negative binomial regression: θ(o,w) = E[runs | over, wickets]
3. Predicts expected runs for each ball
4. Saves model and predictions
5. Generates diagnostic statistics

The expected runs model forms the baseline for all cricWAR calculations.
It captures how many runs we expect in each game state (over, wickets).

Model specification (from paper):
- Dependent variable: batter_runs (0, 1, 2, 4, 6)
- Independent variables: over, wickets_before
- Distribution: Negative binomial (handles overdispersion in count data)
- Link function: log (standard for count models)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import pickle
import json

def load_ipl_data(data_file: Path) -> pd.DataFrame:
    """
    Load IPL ball-by-ball data.

    Args:
        data_file: Path to parquet file

    Returns:
        DataFrame with ball-by-ball data
    """
    print(f"Loading data from {data_file}...")
    df = pd.read_parquet(data_file)
    print(f"✓ Loaded {len(df):,} balls from {df['match_id'].nunique()} matches")
    return df


def prepare_regression_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for negative binomial regression.

    Args:
        df: Raw ball-by-ball data

    Returns:
        DataFrame ready for regression
    """
    print("\nPreparing regression data...")

    # Keep only legal deliveries (exclude wides and no-balls)
    # This is important because wides/no-balls have different run distributions
    legal_balls = df[(df['wides'] == 0) & (df['noballs'] == 0)].copy()
    print(f"✓ Using {len(legal_balls):,} legal deliveries (excluded {len(df) - len(legal_balls):,} wides/no-balls)")

    # Create regression dataframe
    reg_data = pd.DataFrame({
        'runs': legal_balls['batter_runs'],  # Dependent variable
        'over': legal_balls['over'],  # Over number (0-19)
        'wickets': legal_balls['wickets_before'],  # Wickets lost (0-9)
        'match_id': legal_balls['match_id'],
        'innings': legal_balls['innings'],
    })

    # Summary statistics
    print("\nDependent variable (runs) distribution:")
    print(reg_data['runs'].value_counts().sort_index().to_string())
    print(f"\nMean: {reg_data['runs'].mean():.3f}")
    print(f"Variance: {reg_data['runs'].var():.3f}")
    print(f"Variance/Mean ratio: {reg_data['runs'].var() / reg_data['runs'].mean():.3f}")
    print("(Ratio > 1 indicates overdispersion, justifying negative binomial)")

    return reg_data


def fit_negative_binomial_model(reg_data: pd.DataFrame):
    """
    Fit negative binomial regression model.

    Model: θ(o,w) = exp(β₀ + β₁·over + β₂·wickets)

    Args:
        reg_data: Prepared regression data

    Returns:
        Fitted model results
    """
    print("\n" + "="*70)
    print("FITTING NEGATIVE BINOMIAL REGRESSION")
    print("="*70)
    print("\nModel specification:")
    print("  θ(o,w) ~ NegativeBinomial(μ, α)")
    print("  log(μ) = β₀ + β₁·over + β₂·wickets")
    print("\nFitting model (this may take 1-2 minutes)...")

    # Fit negative binomial GLM
    # Formula: runs ~ over + wickets
    # Family: Negative Binomial with log link
    model = smf.glm(
        formula='runs ~ over + wickets',
        data=reg_data,
        family=sm.families.NegativeBinomial()
    ).fit()

    print("\n✓ Model fitted successfully")

    # Print model summary
    print("\n" + "="*70)
    print("MODEL SUMMARY")
    print("="*70)
    print(model.summary())

    # Extract key statistics
    print("\n" + "="*70)
    print("KEY STATISTICS")
    print("="*70)
    print(f"Log-Likelihood: {model.llf:.2f}")
    print(f"AIC: {model.aic:.2f}")
    print(f"BIC: {model.bic:.2f}")
    print(f"Pearson chi2: {model.pearson_chi2:.2f}")

    # Coefficient interpretation
    print("\n" + "="*70)
    print("COEFFICIENT INTERPRETATION")
    print("="*70)
    for param, coef in model.params.items():
        exp_coef = np.exp(coef)
        print(f"{param:15s}: {coef:7.4f}  (exp: {exp_coef:.4f})")

        if param == 'over':
            pct_change = (exp_coef - 1) * 100
            print(f"  → Each additional over changes expected runs by {pct_change:+.2f}%")
        elif param == 'wickets':
            pct_change = (exp_coef - 1) * 100
            print(f"  → Each additional wicket changes expected runs by {pct_change:+.2f}%")

    return model


def predict_expected_runs(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict expected runs θ(o,w) for each ball.

    Args:
        model: Fitted negative binomial model
        df: Original ball-by-ball data

    Returns:
        DataFrame with expected runs added
    """
    print("\n" + "="*70)
    print("PREDICTING EXPECTED RUNS")
    print("="*70)

    # Prepare prediction data (same structure as training)
    pred_data = pd.DataFrame({
        'over': df['over'],
        'wickets': df['wickets_before'],
    })

    # Predict expected runs
    df['expected_runs'] = model.predict(pred_data)

    print(f"✓ Predicted expected runs for {len(df):,} balls")

    # Summary statistics
    print("\nExpected runs statistics:")
    print(f"  Mean: {df['expected_runs'].mean():.4f}")
    print(f"  Std:  {df['expected_runs'].std():.4f}")
    print(f"  Min:  {df['expected_runs'].min():.4f}")
    print(f"  Max:  {df['expected_runs'].max():.4f}")

    # Compare to actual
    print("\nActual vs Expected:")
    print(f"  Actual mean:   {df['batter_runs'].mean():.4f}")
    print(f"  Expected mean: {df['expected_runs'].mean():.4f}")
    print(f"  Difference:    {df['batter_runs'].mean() - df['expected_runs'].mean():.4f}")

    return df


def analyze_by_game_state(df: pd.DataFrame):
    """
    Analyze expected runs by game state (over, wickets).

    Args:
        df: DataFrame with expected_runs
    """
    print("\n" + "="*70)
    print("EXPECTED RUNS BY GAME STATE")
    print("="*70)

    # Group by phase
    print("\nBy phase:")
    phase_stats = df.groupby('phase').agg({
        'expected_runs': ['mean', 'std'],
        'batter_runs': 'mean'
    }).round(4)
    print(phase_stats.to_string())

    # By wickets
    print("\nBy wickets lost:")
    wicket_stats = df.groupby('wickets_before').agg({
        'expected_runs': ['mean', 'std', 'count'],
        'batter_runs': 'mean'
    }).round(4)
    print(wicket_stats.to_string())

    # By over (first 10 overs)
    print("\nBy over (first 10 overs):")
    over_stats = df[df['over'] < 10].groupby('over').agg({
        'expected_runs': ['mean', 'std'],
        'batter_runs': 'mean'
    }).round(4)
    print(over_stats.to_string())


def save_model_and_results(model, df: pd.DataFrame, output_dir: Path):
    """
    Save model, predictions, and diagnostics.

    Args:
        model: Fitted model
        df: DataFrame with predictions
        output_dir: Output directory
    """
    print("\n" + "="*70)
    print("SAVING MODEL AND RESULTS")
    print("="*70)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_file = output_dir / 'expected_runs_model.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Saved model to {model_file}")

    # Save coefficients
    coef_file = output_dir / 'model_coefficients.json'
    coefficients = {
        'params': model.params.to_dict(),
        'pvalues': model.pvalues.to_dict(),
        'conf_int': model.conf_int().to_dict(),
        'alpha': float(model.scale),  # Dispersion parameter
    }
    with open(coef_file, 'w') as f:
        json.dump(coefficients, f, indent=2)
    print(f"✓ Saved coefficients to {coef_file}")

    # Save model summary
    summary_file = output_dir / 'model_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(str(model.summary()))
    print(f"✓ Saved summary to {summary_file}")

    # Save predictions (add expected_runs to original data)
    pred_file = output_dir / 'ipl_with_expected_runs.parquet'
    df.to_parquet(pred_file, index=False)
    print(f"✓ Saved predictions to {pred_file}")

    print(f"\n✓ All results saved to {output_dir}")


def main():
    """Estimate expected runs model."""

    # Paths
    project_root = Path(__file__).parent.parent
    data_file = project_root / 'data' / 'ipl_matches.parquet'
    output_dir = project_root / 'results' / '03_expected_runs'

    print("="*70)
    print("EXPECTED RUNS MODEL θ(o,w)")
    print("="*70)
    print("\nThis model estimates baseline run expectation for each game state.")
    print("It forms the foundation for all cricWAR calculations.\n")

    # Load data
    df = load_ipl_data(data_file)

    # Prepare regression data
    reg_data = prepare_regression_data(df)

    # Fit model
    model = fit_negative_binomial_model(reg_data)

    # Predict expected runs for all balls
    df = predict_expected_runs(model, df)

    # Analyze by game state
    analyze_by_game_state(df)

    # Save results
    save_model_and_results(model, df, output_dir)

    print("\n" + "="*70)
    print("✓ EXPECTED RUNS MODEL COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Review model diagnostics in results/03_expected_runs/")
    print("2. Create validation notebook (02_expected_runs_validation.ipynb)")
    print("3. Calculate run values δ = r - θ (script 04)")


if __name__ == '__main__':
    main()
