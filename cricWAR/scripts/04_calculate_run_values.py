"""
Calculate run values δ = r - θ for each ball.

This script:
1. Loads data with expected runs θ(o,w) from script 03
2. Calculates run value δ = actual_runs - expected_runs
3. Applies runs conservation framework (batter gets +δ, bowler gets -δ)
4. Adds metadata for context adjustments
5. Saves enriched dataset

Run values represent runs above/below expectation for each ball.
Positive δ means the batter scored more than expected (or bowler gave away more).
Negative δ means the batter scored less than expected (or bowler restricted).

The runs conservation framework ensures:
  RAA_batter + RAA_bowler = 0 for each ball
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_data_with_expected_runs(data_file: Path) -> pd.DataFrame:
    """
    Load ball-by-ball data with expected runs.

    Args:
        data_file: Path to parquet with expected_runs column

    Returns:
        DataFrame with expected_runs
    """
    print(f"Loading data from {data_file}...")
    df = pd.read_parquet(data_file)

    # Verify expected_runs column exists
    if 'expected_runs' not in df.columns:
        raise ValueError("Expected 'expected_runs' column not found. Run script 03 first.")

    print(f"✓ Loaded {len(df):,} balls from {df['match_id'].nunique()} matches")
    print(f"✓ Expected runs column present")

    return df


def calculate_run_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate run values δ = r - θ for each ball.

    Args:
        df: DataFrame with batter_runs and expected_runs

    Returns:
        DataFrame with run_value column added
    """
    print("\n" + "="*70)
    print("CALCULATING RUN VALUES")
    print("="*70)

    # Calculate run value: δ = r - θ
    df['run_value'] = df['batter_runs'] - df['expected_runs']

    print(f"\n✓ Calculated run values for {len(df):,} balls")

    # Summary statistics
    print("\nRun value (δ) statistics:")
    print(f"  Mean:   {df['run_value'].mean():7.4f}")
    print(f"  Std:    {df['run_value'].std():7.4f}")
    print(f"  Min:    {df['run_value'].min():7.4f}")
    print(f"  Max:    {df['run_value'].max():7.4f}")
    print(f"  Median: {df['run_value'].median():7.4f}")

    # Distribution of positive vs negative run values
    positive = (df['run_value'] > 0).sum()
    negative = (df['run_value'] < 0).sum()
    zero = (df['run_value'] == 0).sum()

    print(f"\nRun value distribution:")
    print(f"  Positive (batter above expectation): {positive:6,} ({positive/len(df)*100:.1f}%)")
    print(f"  Negative (batter below expectation): {negative:6,} ({negative/len(df)*100:.1f}%)")
    print(f"  Zero     (exactly as expected):     {zero:6,} ({zero/len(df)*100:.1f}%)")

    return df


def add_player_metadata(df: pd.DataFrame, metadata_file: Path) -> pd.DataFrame:
    """
    Add player metadata for context adjustments.

    Args:
        df: Ball-by-ball data
        metadata_file: Path to player metadata CSV

    Returns:
        DataFrame with metadata columns added
    """
    print("\n" + "="*70)
    print("ADDING PLAYER METADATA")
    print("="*70)

    # Load metadata
    print(f"Loading metadata from {metadata_file}...")
    metadata = pd.read_csv(metadata_file)
    print(f"✓ Loaded metadata for {len(metadata)} players")

    # Prepare metadata for merging
    # Batter metadata
    batter_meta = metadata[[
        'player_id', 'batting_hand', 'role_category'
    ]].rename(columns={
        'player_id': 'batter_id',
        'batting_hand': 'batter_hand',
        'role_category': 'batter_role'
    })

    # Bowler metadata
    bowler_meta = metadata[[
        'player_id', 'bowling_type', 'bowling_arm'
    ]].rename(columns={
        'player_id': 'bowler_id',
    })

    # Merge batter metadata
    print("\nMerging batter metadata...")
    before_len = len(df)
    df = df.merge(batter_meta, on='batter_id', how='left')
    after_len = len(df)

    if before_len != after_len:
        print(f"⚠ Warning: Row count changed from {before_len:,} to {after_len:,}")
    else:
        print(f"✓ Merged successfully (row count unchanged)")

    # Merge bowler metadata
    print("Merging bowler metadata...")
    before_len = len(df)
    df = df.merge(bowler_meta, on='bowler_id', how='left')
    after_len = len(df)

    if before_len != after_len:
        print(f"⚠ Warning: Row count changed from {before_len:,} to {after_len:,}")
    else:
        print(f"✓ Merged successfully (row count unchanged)")

    # Fill missing metadata
    df['batter_hand'] = df['batter_hand'].fillna('unknown')
    df['batter_role'] = df['batter_role'].fillna('unknown')
    df['bowling_type'] = df['bowling_type'].fillna('unknown')
    df['bowling_arm'] = df['bowling_arm'].fillna('unknown')

    # Summary of metadata coverage
    print("\nMetadata coverage:")
    print(f"  Batter hand known:   {(df['batter_hand'] != 'unknown').sum():6,} / {len(df):,} ({(df['batter_hand'] != 'unknown').sum()/len(df)*100:.1f}%)")
    print(f"  Bowling type known:  {(df['bowling_type'] != 'unknown').sum():6,} / {len(df):,} ({(df['bowling_type'] != 'unknown').sum()/len(df)*100:.1f}%)")

    return df


def calculate_platoon_advantage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate platoon advantage (same vs opposite handedness).

    Args:
        df: DataFrame with batter_hand and bowling_arm

    Returns:
        DataFrame with platoon_advantage column
    """
    print("\n" + "="*70)
    print("CALCULATING PLATOON ADVANTAGE")
    print("="*70)

    # Platoon advantage:
    # - "same" if batter and bowler are same handedness (LHB vs left-arm, RHB vs right-arm)
    # - "opposite" if different handedness (LHB vs right-arm, RHB vs left-arm)
    # - "unknown" if either is unknown

    def get_platoon(row):
        batter_hand = row['batter_hand']
        bowling_arm = row['bowling_arm']

        if batter_hand == 'unknown' or bowling_arm == 'unknown':
            return 'unknown'

        # Map to comparable values
        batter_side = 'left' if batter_hand == 'LHB' else 'right'
        bowler_side = 'left' if 'left' in bowling_arm.lower() else 'right'

        return 'same' if batter_side == bowler_side else 'opposite'

    df['platoon_advantage'] = df.apply(get_platoon, axis=1)

    # Summary
    print("\nPlatoon advantage distribution:")
    platoon_counts = df['platoon_advantage'].value_counts()
    for platoon, count in platoon_counts.items():
        print(f"  {platoon:10s}: {count:6,} ({count/len(df)*100:.1f}%)")

    return df


def add_contextual_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add contextual features for adjustment regressions.

    Args:
        df: Ball-by-ball data

    Returns:
        DataFrame with contextual features
    """
    print("\n" + "="*70)
    print("ADDING CONTEXTUAL FEATURES")
    print("="*70)

    # Innings indicator (1st vs 2nd innings)
    df['is_second_innings'] = (df['innings'] == 2).astype(int)
    print(f"✓ Added innings indicator (2nd innings: {df['is_second_innings'].sum():,} balls)")

    # Normalize venue names (for regression)
    df['venue_normalized'] = df['venue'].str.strip().str.lower()
    print(f"✓ Normalized {df['venue'].nunique()} unique venues")

    # Phase indicators (already exists, but create dummies for regression)
    df['is_powerplay'] = (df['phase'] == 'powerplay').astype(int)
    df['is_death'] = (df['phase'] == 'death').astype(int)
    print(f"✓ Added phase indicators")

    return df


def analyze_run_values_by_context(df: pd.DataFrame):
    """
    Analyze run values by different contexts.

    Args:
        df: DataFrame with run values and context
    """
    print("\n" + "="*70)
    print("RUN VALUES BY CONTEXT")
    print("="*70)

    # By phase
    print("\nBy phase:")
    phase_stats = df.groupby('phase')['run_value'].agg(['mean', 'std', 'count']).round(4)
    print(phase_stats.to_string())

    # By innings
    print("\nBy innings:")
    innings_stats = df.groupby('innings')['run_value'].agg(['mean', 'std', 'count']).round(4)
    print(innings_stats.to_string())

    # By platoon advantage
    print("\nBy platoon advantage:")
    platoon_stats = df.groupby('platoon_advantage')['run_value'].agg(['mean', 'std', 'count']).round(4)
    print(platoon_stats.to_string())

    # By bowling type
    print("\nBy bowling type:")
    bowling_stats = df.groupby('bowling_type')['run_value'].agg(['mean', 'std', 'count']).round(4)
    print(bowling_stats.to_string())

    # By wicket outcome
    print("\nBy wicket outcome:")
    wicket_stats = df.groupby('is_wicket')['run_value'].agg(['mean', 'std', 'count']).round(4)
    print(wicket_stats.to_string())


def save_enriched_dataset(df: pd.DataFrame, output_dir: Path):
    """
    Save enriched dataset with run values and context.

    Args:
        df: Enriched DataFrame
        output_dir: Output directory
    """
    print("\n" + "="*70)
    print("SAVING ENRICHED DATASET")
    print("="*70)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to parquet
    output_file = output_dir / 'ipl_with_run_values.parquet'
    df.to_parquet(output_file, index=False)
    print(f"✓ Saved {len(df):,} balls to {output_file}")

    # Save summary statistics
    summary_file = output_dir / 'run_values_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("RUN VALUES SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total balls: {len(df):,}\n")
        f.write(f"Mean run value: {df['run_value'].mean():.4f}\n")
        f.write(f"Std run value: {df['run_value'].std():.4f}\n\n")

        f.write("Distribution:\n")
        f.write(df['run_value'].describe().to_string())

    print(f"✓ Saved summary to {summary_file}")

    print(f"\n✓ All results saved to {output_dir}")


def main():
    """Calculate run values and enrich dataset."""

    # Paths
    project_root = Path(__file__).parent.parent
    data_file = project_root / 'results' / '03_expected_runs' / 'ipl_with_expected_runs.parquet'
    metadata_file = project_root / 'data' / 'player_metadata.csv'
    output_dir = project_root / 'results' / '04_run_values'

    print("="*70)
    print("CALCULATE RUN VALUES δ = r - θ")
    print("="*70)
    print("\nRun values represent runs above/below expectation.")
    print("This forms the basis for player attribution in cricWAR.\n")

    # Load data with expected runs
    df = load_data_with_expected_runs(data_file)

    # Calculate run values
    df = calculate_run_values(df)

    # Add player metadata
    df = add_player_metadata(df, metadata_file)

    # Calculate platoon advantage
    df = calculate_platoon_advantage(df)

    # Add contextual features
    df = add_contextual_features(df)

    # Analyze by context
    analyze_run_values_by_context(df)

    # Save enriched dataset
    save_enriched_dataset(df, output_dir)

    print("\n" + "="*70)
    print("✓ RUN VALUES CALCULATION COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Calculate leverage index (script 05)")
    print("2. Implement context adjustment regressions (script 06)")
    print("3. Calculate RAA for each player (script 07)")


if __name__ == '__main__':
    main()
