"""
Calculate leverage index (LI) for each ball.

This script:
1. Loads ball-by-ball data with run values
2. Calculates leverage index based on game state
3. Applies LI weighting to run values
4. Saves data with leverage index

Leverage index represents the importance of a ball based on game situation.
Higher leverage = more critical situation where runs have greater impact.

In T20 cricket, leverage is determined by:
- Phase of play (powerplay < middle < death)
- Wickets in hand (more wickets = higher leverage)
- Match situation (close games = higher leverage)

The paper uses LI to weight run values for player evaluation.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_data_with_run_values(data_file: Path) -> pd.DataFrame:
    """
    Load ball-by-ball data with run values.

    Args:
        data_file: Path to parquet with run_value column

    Returns:
        DataFrame with run_value
    """
    print(f"Loading data from {data_file}...")
    df = pd.read_parquet(data_file)

    # Verify required columns exist
    required_cols = ['run_value', 'over', 'wickets_before', 'phase']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print(f"✓ Loaded {len(df):,} balls from {df['match_id'].nunique()} matches")
    return df


def calculate_phase_leverage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate base leverage index by phase.

    Leverage by phase (from empirical T20 analysis):
    - Powerplay (0-5): Lower leverage (foundation building)
    - Middle (6-15): Medium leverage (acceleration)
    - Death (16-19): High leverage (outcome decisive)

    Args:
        df: Ball-by-ball data

    Returns:
        DataFrame with phase_leverage column
    """
    print("\n" + "="*70)
    print("CALCULATING PHASE LEVERAGE")
    print("="*70)

    # Base leverage by phase
    # Values calibrated to T20 cricket importance
    phase_leverage_map = {
        'powerplay': 0.8,   # Lower leverage - building platform
        'middle': 1.0,      # Baseline leverage
        'death': 1.4,       # Higher leverage - outcome decisive
    }

    df['phase_leverage'] = df['phase'].map(phase_leverage_map)

    # Fill any missing values with baseline
    df['phase_leverage'] = df['phase_leverage'].fillna(1.0)

    print("\nPhase leverage distribution:")
    for phase in ['powerplay', 'middle', 'death']:
        count = (df['phase'] == phase).sum()
        li = phase_leverage_map.get(phase, 1.0)
        print(f"  {phase:10s}: LI = {li:.2f}  ({count:6,} balls, {count/len(df)*100:.1f}%)")

    return df


def calculate_wickets_leverage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate leverage adjustment based on wickets in hand.

    More wickets in hand = more flexibility = higher leverage for each run.
    Fewer wickets = must preserve = lower leverage.

    Args:
        df: Ball-by-ball data

    Returns:
        DataFrame with wickets_leverage column
    """
    print("\n" + "="*70)
    print("CALCULATING WICKETS LEVERAGE")
    print("="*70)

    # Wickets in hand (10 - wickets_lost)
    df['wickets_in_hand'] = 10 - df['wickets_before']

    # Leverage increases with wickets in hand
    # Using non-linear mapping: more wickets = disproportionately higher leverage
    # Formula: LI = 0.6 + 0.05 * wickets_in_hand
    # Range: 0.6 (0 wickets) to 1.1 (10 wickets)
    df['wickets_leverage'] = 0.6 + 0.05 * df['wickets_in_hand']

    print("\nWickets leverage by wickets in hand:")
    wickets_stats = df.groupby('wickets_in_hand').agg({
        'wickets_leverage': 'first',
        'match_id': 'count'
    }).rename(columns={'match_id': 'count'})

    for wickets, row in wickets_stats.iterrows():
        print(f"  {int(wickets):2d} wickets: LI = {row['wickets_leverage']:.2f}  ({int(row['count']):6,} balls)")

    return df


def calculate_match_situation_leverage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate leverage based on match situation (score differential).

    For 2nd innings only: leverage is higher when match is close.
    For 1st innings: use baseline (no target known yet).

    Args:
        df: Ball-by-ball data

    Returns:
        DataFrame with situation_leverage column
    """
    print("\n" + "="*70)
    print("CALCULATING MATCH SITUATION LEVERAGE")
    print("="*70)

    # Initialize with baseline
    df['situation_leverage'] = 1.0

    # For 1st innings, situation leverage is always 1.0
    # (no target to compare against)

    # For 2nd innings, we'd need to calculate runs required
    # For now, keeping it simple with baseline
    # (Can enhance later with target tracking)

    print("\nSituation leverage:")
    print(f"  1st innings: LI = 1.00 (baseline)")
    print(f"  2nd innings: LI = 1.00 (baseline - can enhance with target tracking)")

    return df


def calculate_combined_leverage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine leverage components into final leverage index.

    LI = phase_leverage × wickets_leverage × situation_leverage

    Args:
        df: Ball-by-ball data with component leverages

    Returns:
        DataFrame with leverage_index column
    """
    print("\n" + "="*70)
    print("CALCULATING COMBINED LEVERAGE INDEX")
    print("="*70)

    # Combine leverage components multiplicatively
    df['leverage_index'] = (
        df['phase_leverage'] *
        df['wickets_leverage'] *
        df['situation_leverage']
    )

    print(f"\n✓ Calculated leverage index for {len(df):,} balls")

    # Summary statistics
    print("\nLeverage index statistics:")
    print(f"  Mean:   {df['leverage_index'].mean():.4f}")
    print(f"  Std:    {df['leverage_index'].std():.4f}")
    print(f"  Min:    {df['leverage_index'].min():.4f}")
    print(f"  Max:    {df['leverage_index'].max():.4f}")
    print(f"  Median: {df['leverage_index'].median():.4f}")

    # Percentiles
    print("\nLeverage index percentiles:")
    for pct in [10, 25, 50, 75, 90]:
        val = df['leverage_index'].quantile(pct/100)
        print(f"  {pct:2d}th: {val:.4f}")

    return df


def calculate_weighted_run_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate leverage-weighted run values.

    weighted_run_value = run_value × leverage_index

    This weights each run by its importance in the game context.

    Args:
        df: Ball-by-ball data with run_value and leverage_index

    Returns:
        DataFrame with weighted_run_value column
    """
    print("\n" + "="*70)
    print("CALCULATING LEVERAGE-WEIGHTED RUN VALUES")
    print("="*70)

    # Weight run values by leverage
    df['weighted_run_value'] = df['run_value'] * df['leverage_index']

    print(f"\n✓ Calculated weighted run values for {len(df):,} balls")

    # Compare raw vs weighted
    print("\nRaw vs Weighted run values:")
    print(f"  Raw mean:      {df['run_value'].mean():.4f}")
    print(f"  Weighted mean: {df['weighted_run_value'].mean():.4f}")
    print(f"  Raw std:       {df['run_value'].std():.4f}")
    print(f"  Weighted std:  {df['weighted_run_value'].std():.4f}")

    return df


def analyze_leverage_by_context(df: pd.DataFrame):
    """
    Analyze leverage index by different contexts.

    Args:
        df: DataFrame with leverage_index
    """
    print("\n" + "="*70)
    print("LEVERAGE INDEX BY CONTEXT")
    print("="*70)

    # By phase
    print("\nBy phase:")
    phase_stats = df.groupby('phase')['leverage_index'].agg(['mean', 'std', 'count']).round(4)
    print(phase_stats.to_string())

    # By wickets lost
    print("\nBy wickets lost:")
    wickets_stats = df.groupby('wickets_before')['leverage_index'].agg(['mean', 'std', 'count']).round(4)
    print(wickets_stats.to_string())

    # By over (selected overs)
    print("\nBy over (selected):")
    selected_overs = [0, 5, 10, 15, 19]
    over_stats = df[df['over'].isin(selected_overs)].groupby('over')['leverage_index'].agg(['mean', 'std', 'count']).round(4)
    print(over_stats.to_string())

    # By innings
    print("\nBy innings:")
    innings_stats = df[df['innings'].isin([1, 2])].groupby('innings')['leverage_index'].agg(['mean', 'std', 'count']).round(4)
    print(innings_stats.to_string())


def save_data_with_leverage(df: pd.DataFrame, output_dir: Path):
    """
    Save data with leverage index.

    Args:
        df: DataFrame with leverage_index
        output_dir: Output directory
    """
    print("\n" + "="*70)
    print("SAVING DATA WITH LEVERAGE INDEX")
    print("="*70)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to parquet
    output_file = output_dir / 'ipl_with_leverage_index.parquet'
    df.to_parquet(output_file, index=False)
    print(f"✓ Saved {len(df):,} balls to {output_file}")

    # Save summary statistics
    summary_file = output_dir / 'leverage_index_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("LEVERAGE INDEX SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total balls: {len(df):,}\n")
        f.write(f"Mean leverage index: {df['leverage_index'].mean():.4f}\n")
        f.write(f"Std leverage index: {df['leverage_index'].std():.4f}\n\n")

        f.write("Distribution:\n")
        f.write(df['leverage_index'].describe().to_string())

    print(f"✓ Saved summary to {summary_file}")

    print(f"\n✓ All results saved to {output_dir}")


def main():
    """Calculate leverage index for all balls."""

    # Paths
    project_root = Path(__file__).parent.parent
    data_file = project_root / 'results' / '04_run_values' / 'ipl_with_run_values.parquet'
    output_dir = project_root / 'results' / '05_leverage_index'

    print("="*70)
    print("CALCULATE LEVERAGE INDEX (LI)")
    print("="*70)
    print("\nLeverage index weights run values by game importance.")
    print("Higher leverage = more critical situation.\n")

    # Load data with run values
    df = load_data_with_run_values(data_file)

    # Calculate phase leverage
    df = calculate_phase_leverage(df)

    # Calculate wickets leverage
    df = calculate_wickets_leverage(df)

    # Calculate match situation leverage
    df = calculate_match_situation_leverage(df)

    # Combine into final leverage index
    df = calculate_combined_leverage(df)

    # Calculate weighted run values
    df = calculate_weighted_run_values(df)

    # Analyze by context
    analyze_leverage_by_context(df)

    # Save results
    save_data_with_leverage(df, output_dir)

    print("\n" + "="*70)
    print("✓ LEVERAGE INDEX CALCULATION COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Implement context adjustment regressions (script 06)")
    print("2. Calculate RAA for each player (script 07)")
    print("3. Calculate VORP and WAR (scripts 08-09)")


if __name__ == '__main__':
    main()
