"""
Implement context adjustment regressions to isolate player skill.

This script:
1. Loads ball-by-ball data with leverage-weighted run values
2. Runs linear regressions to control for contextual factors:
   - Venue effects (some grounds favor batting/bowling)
   - Innings effects (batting 1st vs 2nd)
   - Platoon advantage (same vs opposite handedness)
   - Bowling type (pace vs spin)
3. Extracts residuals as context-neutral player contributions
4. Calculates RAA (Runs Above Average) for batters and bowlers

The residuals represent player skill AFTER controlling for context.
This ensures fair comparison across different situations.

Runs conservation: RAA_batter + RAA_bowler = 0 for each ball
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pickle
import json


def load_data_with_leverage(data_file: Path) -> pd.DataFrame:
    """
    Load ball-by-ball data with leverage index.

    Args:
        data_file: Path to parquet with leverage_index

    Returns:
        DataFrame with weighted_run_value
    """
    print(f"Loading data from {data_file}...")
    df = pd.read_parquet(data_file)

    required_cols = ['weighted_run_value', 'venue_normalized', 'is_second_innings',
                     'platoon_advantage', 'bowling_type']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print(f"✓ Loaded {len(df):,} balls from {df['match_id'].nunique()} matches")
    return df


def prepare_regression_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for context adjustment regressions.

    Args:
        df: Raw ball-by-ball data

    Returns:
        DataFrame ready for regression
    """
    print("\n" + "="*70)
    print("PREPARING REGRESSION DATA")
    print("="*70)

    # Create dummy variables for categorical factors
    print("\nCreating dummy variables...")

    # Clean function for column names (make valid Python identifiers)
    def clean_column_name(name):
        # Replace special characters with underscore
        import re
        name = re.sub(r'[^\w\s]', '_', name)  # Replace special chars with _
        name = re.sub(r'\s+', '_', name)       # Replace whitespace with _
        name = re.sub(r'_+', '_', name)        # Collapse multiple underscores
        name = name.strip('_')                 # Remove leading/trailing underscores
        return name.lower()

    # Venue dummies (drop first to avoid multicollinearity)
    venue_dummies = pd.get_dummies(df['venue_normalized'], prefix='venue', drop_first=True, dtype=int)
    venue_dummies.columns = [clean_column_name(col) for col in venue_dummies.columns]
    print(f"✓ Created {len(venue_dummies.columns)} venue dummies")

    # Platoon advantage dummies
    platoon_dummies = pd.get_dummies(df['platoon_advantage'], prefix='platoon', drop_first=True, dtype=int)
    platoon_dummies.columns = [clean_column_name(col) for col in platoon_dummies.columns]
    print(f"✓ Created {len(platoon_dummies.columns)} platoon dummies")

    # Bowling type dummies
    bowling_dummies = pd.get_dummies(df['bowling_type'], prefix='bowling', drop_first=True, dtype=int)
    bowling_dummies.columns = [clean_column_name(col) for col in bowling_dummies.columns]
    print(f"✓ Created {len(bowling_dummies.columns)} bowling type dummies")

    # Combine all features
    reg_data = pd.concat([
        df[['weighted_run_value', 'is_second_innings', 'batter_id', 'bowler_id',
            'match_id', 'innings', 'batter_name', 'bowler_name']],
        venue_dummies,
        platoon_dummies,
        bowling_dummies
    ], axis=1)

    print(f"\n✓ Prepared regression data: {len(reg_data):,} rows, {len(reg_data.columns)} columns")

    # Return the cleaned column names
    return reg_data, list(venue_dummies.columns), list(platoon_dummies.columns), list(bowling_dummies.columns)


def fit_batter_context_model(reg_data: pd.DataFrame, venue_cols: list, platoon_cols: list, bowling_cols: list):
    """
    Fit context adjustment model for batters.

    Model: weighted_run_value ~ venue + innings + platoon + bowling_type

    Args:
        reg_data: Prepared regression data
        venue_cols: List of venue dummy columns
        platoon_cols: List of platoon dummy columns
        bowling_cols: List of bowling type dummy columns

    Returns:
        Fitted model
    """
    print("\n" + "="*70)
    print("FITTING BATTER CONTEXT ADJUSTMENT MODEL")
    print("="*70)

    # Build feature matrix using direct column selection
    context_vars = ['is_second_innings'] + venue_cols + platoon_cols + bowling_cols

    print(f"\nModel specification:")
    print(f"  weighted_run_value ~ innings + {len(venue_cols)} venues + {len(platoon_cols)} platoon + {len(bowling_cols)} bowling")
    print(f"  Total features: {len(context_vars)}")

    # Prepare X and y
    X = reg_data[context_vars]
    y = reg_data['weighted_run_value']

    # Add constant
    X = sm.add_constant(X)

    print("\nFitting model...")
    model = sm.OLS(y, X).fit()

    print("✓ Model fitted successfully")

    # Print summary statistics
    print("\n" + "-"*70)
    print("MODEL STATISTICS")
    print("-"*70)
    print(f"R-squared:     {model.rsquared:.4f}")
    print(f"Adj R-squared: {model.rsquared_adj:.4f}")
    print(f"F-statistic:   {model.fvalue:.2f}")
    print(f"Prob (F):      {model.f_pvalue:.4e}")
    print(f"N:             {int(model.nobs):,}")

    # Print key coefficients
    print("\n" + "-"*70)
    print("KEY COEFFICIENTS")
    print("-"*70)

    # Innings effect
    if 'is_second_innings' in model.params:
        coef = model.params['is_second_innings']
        pval = model.pvalues['is_second_innings']
        print(f"2nd innings:   {coef:+.4f} (p={pval:.4f})")

    # Platoon effects
    for col in platoon_cols:
        if col in model.params:
            coef = model.params[col]
            pval = model.pvalues[col]
            print(f"{col:15s}: {coef:+.4f} (p={pval:.4f})")

    # Bowling type effects
    for col in bowling_cols:
        if col in model.params:
            coef = model.params[col]
            pval = model.pvalues[col]
            print(f"{col:15s}: {coef:+.4f} (p={pval:.4f})")

    # Top 5 venue effects
    print("\nTop 5 venue effects (favorable for batting):")
    venue_effects = []
    for col in venue_cols:
        if col in model.params.index:
            coef_val = model.params[col]
            pval_val = model.pvalues[col]

            # Handle case where we get a Series (duplicate column names)
            if isinstance(coef_val, pd.Series):
                if len(coef_val) > 1:
                    print(f"  Warning: Skipping duplicate column '{col}'")
                    continue
                coef_val = float(coef_val.iloc[0])
                pval_val = float(pval_val.iloc[0])
            else:
                coef_val = float(coef_val)
                pval_val = float(pval_val)

            venue_effects.append((col, coef_val, pval_val))

    if venue_effects:
        top_venues = sorted(venue_effects, key=lambda x: x[1], reverse=True)[:5]
        for venue, coef, pval in top_venues:
            print(f"  {venue}: {coef:+.4f} (p={pval:.4f})")

        print("\nBottom 5 venue effects (difficult for batting):")
        bottom_venues = sorted(venue_effects, key=lambda x: x[1])[:5]
        for venue, coef, pval in bottom_venues:
            print(f"  {venue}: {coef:+.4f} (p={pval:.4f})")

    return model


def calculate_batter_raa(df: pd.DataFrame, model, reg_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate RAA (Runs Above Average) for batters.

    RAA = residual from context adjustment regression

    Args:
        df: Original ball-by-ball data
        model: Fitted context model
        reg_data: Regression data (for alignment)

    Returns:
        DataFrame with batter_RAA column
    """
    print("\n" + "="*70)
    print("CALCULATING BATTER RAA")
    print("="*70)

    # Get residuals from model
    residuals = model.resid

    # Ensure alignment with original dataframe
    df = df.copy()
    df['batter_RAA'] = residuals.values

    print(f"\n✓ Calculated RAA for {len(df):,} balls")

    # Summary statistics
    print("\nBatter RAA statistics:")
    print(f"  Mean:   {df['batter_RAA'].mean():7.4f}")
    print(f"  Std:    {df['batter_RAA'].std():7.4f}")
    print(f"  Min:    {df['batter_RAA'].min():7.4f}")
    print(f"  Max:    {df['batter_RAA'].max():7.4f}")

    return df


def calculate_bowler_raa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate RAA for bowlers using runs conservation.

    RAA_bowler = -RAA_batter

    Args:
        df: DataFrame with batter_RAA

    Returns:
        DataFrame with bowler_RAA column
    """
    print("\n" + "="*70)
    print("CALCULATING BOWLER RAA")
    print("="*70)

    # Runs conservation: bowler gets negative of batter RAA
    df['bowler_RAA'] = -df['batter_RAA']

    print(f"\n✓ Calculated bowler RAA for {len(df):,} balls")

    # Verify runs conservation
    print("\nRuns conservation check:")
    total_batter_raa = df['batter_RAA'].sum()
    total_bowler_raa = df['bowler_RAA'].sum()
    total = total_batter_raa + total_bowler_raa

    print(f"  Total batter RAA:  {total_batter_raa:,.4f}")
    print(f"  Total bowler RAA:  {total_bowler_raa:,.4f}")
    print(f"  Sum (should be 0): {total:,.4f}")

    if abs(total) < 0.01:
        print("  ✓ Runs conservation verified")
    else:
        print(f"  ⚠ Warning: Runs not conserved (diff = {total:.4f})")

    # Summary statistics
    print("\nBowler RAA statistics:")
    print(f"  Mean:   {df['bowler_RAA'].mean():7.4f}")
    print(f"  Std:    {df['bowler_RAA'].std():7.4f}")
    print(f"  Min:    {df['bowler_RAA'].min():7.4f}")
    print(f"  Max:    {df['bowler_RAA'].max():7.4f}")

    return df


def aggregate_player_raa(df: pd.DataFrame) -> tuple:
    """
    Aggregate RAA by player.

    Args:
        df: Ball-by-ball data with RAA

    Returns:
        Tuple of (batter_raa_df, bowler_raa_df)
    """
    print("\n" + "="*70)
    print("AGGREGATING PLAYER RAA")
    print("="*70)

    # Aggregate batter RAA
    print("\nAggregating batter RAA...")
    batter_raa = df.groupby(['batter_id', 'batter_name']).agg({
        'batter_RAA': ['sum', 'mean', 'count'],
        'weighted_run_value': 'sum'
    }).round(4)

    batter_raa.columns = ['RAA', 'RAA_per_ball', 'balls_faced', 'total_weighted_runs']
    batter_raa = batter_raa.reset_index()
    batter_raa = batter_raa.sort_values('RAA', ascending=False)

    print(f"✓ Aggregated RAA for {len(batter_raa)} batters")

    # Aggregate bowler RAA
    print("Aggregating bowler RAA...")
    bowler_raa = df.groupby(['bowler_id', 'bowler_name']).agg({
        'bowler_RAA': ['sum', 'mean', 'count'],
        'weighted_run_value': 'sum'
    }).round(4)

    bowler_raa.columns = ['RAA', 'RAA_per_ball', 'balls_bowled', 'total_weighted_runs_against']
    bowler_raa = bowler_raa.reset_index()
    bowler_raa = bowler_raa.sort_values('RAA', ascending=False)

    print(f"✓ Aggregated RAA for {len(bowler_raa)} bowlers")

    # Show top performers
    print("\n" + "-"*70)
    print("TOP 10 BATTERS BY RAA")
    print("-"*70)
    print(batter_raa.head(10)[['batter_name', 'RAA', 'balls_faced', 'RAA_per_ball']].to_string(index=False))

    print("\n" + "-"*70)
    print("TOP 10 BOWLERS BY RAA")
    print("-"*70)
    print(bowler_raa.head(10)[['bowler_name', 'RAA', 'balls_bowled', 'RAA_per_ball']].to_string(index=False))

    return batter_raa, bowler_raa


def save_raa_results(df: pd.DataFrame, batter_raa: pd.DataFrame, bowler_raa: pd.DataFrame,
                     model, output_dir: Path):
    """
    Save RAA results and models.

    Args:
        df: Ball-by-ball data with RAA
        batter_raa: Aggregated batter RAA
        bowler_raa: Aggregated bowler RAA
        model: Context adjustment model
        output_dir: Output directory
    """
    print("\n" + "="*70)
    print("SAVING RAA RESULTS")
    print("="*70)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save ball-by-ball data with RAA
    ball_file = output_dir / 'ipl_with_raa.parquet'
    df.to_parquet(ball_file, index=False)
    print(f"✓ Saved ball-by-ball data to {ball_file}")

    # Save aggregated RAA
    batter_file = output_dir / 'batter_raa.csv'
    batter_raa.to_csv(batter_file, index=False)
    print(f"✓ Saved batter RAA to {batter_file}")

    bowler_file = output_dir / 'bowler_raa.csv'
    bowler_raa.to_csv(bowler_file, index=False)
    print(f"✓ Saved bowler RAA to {bowler_file}")

    # Save context model
    model_file = output_dir / 'context_adjustment_model.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Saved context model to {model_file}")

    # Save model summary
    summary_file = output_dir / 'context_model_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(str(model.summary()))
    print(f"✓ Saved model summary to {summary_file}")

    print(f"\n✓ All results saved to {output_dir}")


def main():
    """Implement context adjustments and calculate RAA."""

    # Paths
    project_root = Path(__file__).parent.parent
    data_file = project_root / 'results' / '05_leverage_index' / 'ipl_with_leverage_index.parquet'
    output_dir = project_root / 'results' / '06_context_adjustments'

    print("="*70)
    print("CONTEXT ADJUSTMENTS & RAA CALCULATION")
    print("="*70)
    print("\nContext adjustments isolate player skill from environmental factors.")
    print("RAA = Runs Above Average after controlling for context.\n")

    # Load data
    df = load_data_with_leverage(data_file)

    # Prepare regression data
    reg_data, venue_cols, platoon_cols, bowling_cols = prepare_regression_data(df)

    # Fit context adjustment model for batters
    model = fit_batter_context_model(reg_data, venue_cols, platoon_cols, bowling_cols)

    # Calculate batter RAA (residuals)
    df = calculate_batter_raa(df, model, reg_data)

    # Calculate bowler RAA (runs conservation)
    df = calculate_bowler_raa(df)

    # Aggregate by player
    batter_raa, bowler_raa = aggregate_player_raa(df)

    # Save results
    save_raa_results(df, batter_raa, bowler_raa, model, output_dir)

    print("\n" + "="*70)
    print("✓ CONTEXT ADJUSTMENTS COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Define replacement level players (script 08)")
    print("2. Calculate VORP (Value Over Replacement Player)")
    print("3. Calculate WAR (Wins Above Replacement)")


if __name__ == '__main__':
    main()
