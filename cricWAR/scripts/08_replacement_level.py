"""
Define replacement level players and calculate avg.RAA_rep.

This script:
1. Loads player RAA data from script 06
2. Defines replacement-level players (bottom tier of regular players)
3. Calculates average RAA per ball for replacement-level players
4. Saves replacement level thresholds for VORP calculation

Replacement level represents the performance of a readily available player
who could be called up from reserves. This is typically:
- Bottom 20-25% of players by playing time
- Or players with very limited appearances

The avg.RAA_rep is used to calculate VORP:
  VORP_X = RAA_X - (avg.RAA_rep · B_X)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json


def load_raa_data(input_dir: Path) -> tuple:
    """
    Load RAA data for batters and bowlers.

    Args:
        input_dir: Directory with RAA CSV files

    Returns:
        Tuple of (batter_raa, bowler_raa) DataFrames
    """
    print(f"Loading RAA data from {input_dir}...")

    batter_file = input_dir / 'batter_raa.csv'
    bowler_file = input_dir / 'bowler_raa.csv'

    batter_raa = pd.read_csv(batter_file)
    bowler_raa = pd.read_csv(bowler_file)

    print(f"✓ Loaded {len(batter_raa)} batters")
    print(f"✓ Loaded {len(bowler_raa)} bowlers")

    return batter_raa, bowler_raa


def define_replacement_level_batters(batter_raa: pd.DataFrame, min_balls: int = 60,
                                     percentile: float = 0.25) -> pd.DataFrame:
    """
    Define replacement-level batters.

    Replacement level is defined as:
    1. Players who faced at least min_balls (to exclude very limited appearances)
    2. Bottom percentile of players by RAA per ball

    Args:
        batter_raa: DataFrame with batter RAA
        min_balls: Minimum balls faced to be considered
        percentile: Percentile to use for replacement level (default 0.25 = bottom 25%)

    Returns:
        DataFrame with replacement-level batters flagged
    """
    print("\n" + "="*70)
    print("DEFINING REPLACEMENT-LEVEL BATTERS")
    print("="*70)

    # Filter batters with sufficient appearances
    qualified_batters = batter_raa[batter_raa['balls_faced'] >= min_balls].copy()

    print(f"\nQualification criteria:")
    print(f"  Minimum balls faced: {min_balls}")
    print(f"  Qualified batters: {len(qualified_batters)} / {len(batter_raa)}")

    # Calculate replacement level threshold (bottom percentile)
    threshold = qualified_batters['RAA_per_ball'].quantile(percentile)

    print(f"\nReplacement level threshold:")
    print(f"  Percentile: {percentile*100:.0f}th (bottom {percentile*100:.0f}%)")
    print(f"  RAA per ball threshold: {threshold:.4f}")

    # Mark replacement-level batters
    batter_raa['is_replacement'] = (
        (batter_raa['balls_faced'] >= min_balls) &
        (batter_raa['RAA_per_ball'] <= threshold)
    )

    replacement_batters = batter_raa[batter_raa['is_replacement']]

    print(f"\nReplacement-level batters identified: {len(replacement_batters)}")
    print(f"  Percentage of qualified batters: {len(replacement_batters)/len(qualified_batters)*100:.1f}%")

    # Summary statistics
    print("\nReplacement-level batter statistics:")
    print(f"  Mean RAA per ball: {replacement_batters['RAA_per_ball'].mean():.4f}")
    print(f"  Std RAA per ball:  {replacement_batters['RAA_per_ball'].std():.4f}")
    print(f"  Min RAA per ball:  {replacement_batters['RAA_per_ball'].min():.4f}")
    print(f"  Max RAA per ball:  {replacement_batters['RAA_per_ball'].max():.4f}")

    return batter_raa


def define_replacement_level_bowlers(bowler_raa: pd.DataFrame, min_balls: int = 60,
                                     percentile: float = 0.25) -> pd.DataFrame:
    """
    Define replacement-level bowlers.

    Replacement level is defined as:
    1. Players who bowled at least min_balls (to exclude very limited appearances)
    2. Bottom percentile of players by RAA per ball

    Args:
        bowler_raa: DataFrame with bowler RAA
        min_balls: Minimum balls bowled to be considered
        percentile: Percentile to use for replacement level (default 0.25 = bottom 25%)

    Returns:
        DataFrame with replacement-level bowlers flagged
    """
    print("\n" + "="*70)
    print("DEFINING REPLACEMENT-LEVEL BOWLERS")
    print("="*70)

    # Filter bowlers with sufficient appearances
    qualified_bowlers = bowler_raa[bowler_raa['balls_bowled'] >= min_balls].copy()

    print(f"\nQualification criteria:")
    print(f"  Minimum balls bowled: {min_balls}")
    print(f"  Qualified bowlers: {len(qualified_bowlers)} / {len(bowler_raa)}")

    # Calculate replacement level threshold (bottom percentile)
    threshold = qualified_bowlers['RAA_per_ball'].quantile(percentile)

    print(f"\nReplacement level threshold:")
    print(f"  Percentile: {percentile*100:.0f}th (bottom {percentile*100:.0f}%)")
    print(f"  RAA per ball threshold: {threshold:.4f}")

    # Mark replacement-level bowlers
    bowler_raa['is_replacement'] = (
        (bowler_raa['balls_bowled'] >= min_balls) &
        (bowler_raa['RAA_per_ball'] <= threshold)
    )

    replacement_bowlers = bowler_raa[bowler_raa['is_replacement']]

    print(f"\nReplacement-level bowlers identified: {len(replacement_bowlers)}")
    print(f"  Percentage of qualified bowlers: {len(replacement_bowlers)/len(qualified_bowlers)*100:.1f}%")

    # Summary statistics
    print("\nReplacement-level bowler statistics:")
    print(f"  Mean RAA per ball: {replacement_bowlers['RAA_per_ball'].mean():.4f}")
    print(f"  Std RAA per ball:  {replacement_bowlers['RAA_per_ball'].std():.4f}")
    print(f"  Min RAA per ball:  {replacement_bowlers['RAA_per_ball'].min():.4f}")
    print(f"  Max RAA per ball:  {replacement_bowlers['RAA_per_ball'].max():.4f}")

    return bowler_raa


def calculate_avg_raa_rep(batter_raa: pd.DataFrame, bowler_raa: pd.DataFrame) -> dict:
    """
    Calculate average RAA per ball for replacement-level players.

    Args:
        batter_raa: DataFrame with is_replacement flag
        bowler_raa: DataFrame with is_replacement flag

    Returns:
        Dictionary with avg_raa_rep for batters and bowlers
    """
    print("\n" + "="*70)
    print("CALCULATING AVG.RAA_REP")
    print("="*70)

    # Get replacement-level players
    replacement_batters = batter_raa[batter_raa['is_replacement']]
    replacement_bowlers = bowler_raa[bowler_raa['is_replacement']]

    # Calculate weighted average (weighted by balls)
    avg_raa_rep_bat = (
        (replacement_batters['RAA'] * replacement_batters['balls_faced']).sum() /
        replacement_batters['balls_faced'].sum()
    ) / replacement_batters['balls_faced'].mean()  # Normalize to per-ball

    avg_raa_rep_bowl = (
        (replacement_bowlers['RAA'] * replacement_bowlers['balls_bowled']).sum() /
        replacement_bowlers['balls_bowled'].sum()
    ) / replacement_bowlers['balls_bowled'].mean()  # Normalize to per-ball

    # Alternative: simple mean of RAA_per_ball
    avg_raa_rep_bat_simple = replacement_batters['RAA_per_ball'].mean()
    avg_raa_rep_bowl_simple = replacement_bowlers['RAA_per_ball'].mean()

    print(f"\nAverage RAA per ball for replacement level:")
    print(f"  Batters (weighted):  {avg_raa_rep_bat:.4f}")
    print(f"  Batters (simple):    {avg_raa_rep_bat_simple:.4f}")
    print(f"  Bowlers (weighted):  {avg_raa_rep_bowl:.4f}")
    print(f"  Bowlers (simple):    {avg_raa_rep_bowl_simple:.4f}")

    # Use simple mean (more interpretable and standard in WAR calculations)
    results = {
        'avg_raa_rep_batting': float(avg_raa_rep_bat_simple),
        'avg_raa_rep_bowling': float(avg_raa_rep_bowl_simple),
        'n_replacement_batters': int(len(replacement_batters)),
        'n_replacement_bowlers': int(len(replacement_bowlers)),
    }

    print(f"\n✓ Using simple mean for VORP calculation")
    print(f"  avg.RAA_rep (batting):  {results['avg_raa_rep_batting']:.4f}")
    print(f"  avg.RAA_rep (bowling):  {results['avg_raa_rep_bowling']:.4f}")

    return results


def display_replacement_level_players(batter_raa: pd.DataFrame, bowler_raa: pd.DataFrame):
    """
    Display replacement-level players for inspection.

    Args:
        batter_raa: DataFrame with is_replacement flag
        bowler_raa: DataFrame with is_replacement flag
    """
    print("\n" + "="*70)
    print("REPLACEMENT-LEVEL PLAYERS")
    print("="*70)

    # Get replacement-level players
    replacement_batters = batter_raa[batter_raa['is_replacement']].sort_values('RAA_per_ball')
    replacement_bowlers = bowler_raa[bowler_raa['is_replacement']].sort_values('RAA_per_ball')

    print("\nReplacement-level batters (sorted by RAA per ball):")
    print(replacement_batters[['batter_name', 'RAA', 'balls_faced', 'RAA_per_ball']].head(15).to_string(index=False))

    print("\n\nReplacement-level bowlers (sorted by RAA per ball):")
    print(replacement_bowlers[['bowler_name', 'RAA', 'balls_bowled', 'RAA_per_ball']].head(15).to_string(index=False))


def save_replacement_level_data(batter_raa: pd.DataFrame, bowler_raa: pd.DataFrame,
                                avg_raa_rep: dict, output_dir: Path):
    """
    Save replacement level data and thresholds.

    Args:
        batter_raa: DataFrame with is_replacement flag
        bowler_raa: DataFrame with is_replacement flag
        avg_raa_rep: Dictionary with avg.RAA_rep values
        output_dir: Output directory
    """
    print("\n" + "="*70)
    print("SAVING REPLACEMENT LEVEL DATA")
    print("="*70)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save updated RAA data with replacement flag
    batter_file = output_dir / 'batter_raa_with_replacement.csv'
    batter_raa.to_csv(batter_file, index=False)
    print(f"✓ Saved batter data to {batter_file}")

    bowler_file = output_dir / 'bowler_raa_with_replacement.csv'
    bowler_raa.to_csv(bowler_file, index=False)
    print(f"✓ Saved bowler data to {bowler_file}")

    # Save avg.RAA_rep values
    avg_raa_file = output_dir / 'avg_raa_replacement.json'
    with open(avg_raa_file, 'w') as f:
        json.dump(avg_raa_rep, f, indent=2)
    print(f"✓ Saved avg.RAA_rep to {avg_raa_file}")

    # Save replacement-level players only
    replacement_batters = batter_raa[batter_raa['is_replacement']]
    replacement_bowlers = bowler_raa[bowler_raa['is_replacement']]

    rep_bat_file = output_dir / 'replacement_batters.csv'
    replacement_batters.to_csv(rep_bat_file, index=False)
    print(f"✓ Saved {len(replacement_batters)} replacement batters to {rep_bat_file}")

    rep_bowl_file = output_dir / 'replacement_bowlers.csv'
    replacement_bowlers.to_csv(rep_bowl_file, index=False)
    print(f"✓ Saved {len(replacement_bowlers)} replacement bowlers to {rep_bowl_file}")

    print(f"\n✓ All results saved to {output_dir}")


def main():
    """Define replacement level and calculate avg.RAA_rep."""

    # Paths
    project_root = Path(__file__).parent.parent
    input_dir = project_root / 'results' / '06_context_adjustments'
    output_dir = project_root / 'results' / '08_replacement_level'

    print("="*70)
    print("REPLACEMENT LEVEL DEFINITION")
    print("="*70)
    print("\nReplacement level represents the performance of a readily available")
    print("player who could be called up from reserves.")
    print("\nThis is defined as the bottom 25% of qualified players by RAA per ball.\n")

    # Load RAA data
    batter_raa, bowler_raa = load_raa_data(input_dir)

    # Define replacement level (bottom 25% of qualified players)
    batter_raa = define_replacement_level_batters(batter_raa, min_balls=60, percentile=0.25)
    bowler_raa = define_replacement_level_bowlers(bowler_raa, min_balls=60, percentile=0.25)

    # Calculate avg.RAA_rep
    avg_raa_rep = calculate_avg_raa_rep(batter_raa, bowler_raa)

    # Display replacement-level players
    display_replacement_level_players(batter_raa, bowler_raa)

    # Save results
    save_replacement_level_data(batter_raa, bowler_raa, avg_raa_rep, output_dir)

    print("\n" + "="*70)
    print("✓ REPLACEMENT LEVEL DEFINITION COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Calculate VORP using avg.RAA_rep (script 09)")
    print("2. Estimate Runs Per Win (RPW)")
    print("3. Calculate WAR (Wins Above Replacement)")


if __name__ == '__main__':
    main()
