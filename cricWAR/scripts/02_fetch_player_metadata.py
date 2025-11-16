"""
Fetch player metadata using existing player info CSV.

This script:
1. Loads the list of unique players from IPL data
2. Uses existing players_info.csv (from your cricinfo scraper)
3. Maps player IDs to metadata (batting style, bowling style, playing role)
4. Saves to data/player_metadata.csv
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm


def load_existing_player_info():
    """
    Load player metadata from existing players_info.csv.

    Returns:
        pandas DataFrame with player metadata
    """
    # Try to find the players_info.csv file
    project_root = Path(__file__).parent.parent
    player_info_file = project_root.parent / 'players_info.csv'

    if not player_info_file.exists():
        print(f"Warning: players_info.csv not found at {player_info_file}")
        print("Creating empty metadata template...")
        return pd.DataFrame(columns=['player_id', 'full_name', 'batting_style', 'bowling_style', 'playing_role'])

    print(f"Loading player info from {player_info_file}...")
    player_info = pd.read_csv(player_info_file)

    print(f"✓ Loaded {len(player_info)} players from existing data")

    return player_info


def map_player_ids(ipl_df, player_meta_df):
    """
    Map player IDs from IPL data to player metadata.

    Args:
        ipl_df: IPL ball-by-ball DataFrame
        player_meta_df: Player metadata DataFrame

    Returns:
        DataFrame with mapped player information
    """
    # Get unique players from IPL data
    batters = ipl_df[['batter_id', 'batter_name']].drop_duplicates()
    batters.columns = ['player_id', 'player_name']

    bowlers = ipl_df[['bowler_id', 'bowler_name']].drop_duplicates()
    bowlers.columns = ['player_id', 'player_name']

    non_strikers = ipl_df[['non_striker_id', 'non_striker_name']].drop_duplicates()
    non_strikers.columns = ['player_id', 'player_name']

    # Combine and remove duplicates
    all_players = pd.concat([batters, bowlers, non_strikers]).drop_duplicates()
    all_players = all_players[all_players['player_id'].notna()]

    print(f"\n✓ Found {len(all_players)} unique players in IPL data")

    # If no metadata loaded, return just the player list
    if len(player_meta_df) == 0:
        print("⚠ No existing player metadata found - creating basic template")
        return all_players

    # Convert player_id to string for merging
    all_players['player_id'] = all_players['player_id'].astype(str)
    if 'player_id' in player_meta_df.columns:
        player_meta_df['player_id'] = player_meta_df['player_id'].astype(str)

    # Merge with metadata
    merged = all_players.merge(
        player_meta_df,
        on='player_id',
        how='left'
    )

    # Fill missing metadata
    metadata_col = 'full_name' if 'full_name' in merged.columns else ('name' if 'name' in merged.columns else None)
    if metadata_col:
        matched = merged[metadata_col].notna().sum()
        total = len(merged)
        print(f"✓ Matched {matched}/{total} players ({matched/total*100:.1f}%)")

    return merged


def categorize_players(metadata_df):
    """
    Add categorized columns for analysis.

    Args:
        metadata_df: Player metadata DataFrame

    Returns:
        DataFrame with additional categorized columns
    """
    print("\nCategorizing players...")

    # Add missing columns with defaults if they don't exist
    if 'batting_style' not in metadata_df.columns:
        metadata_df['batting_style'] = None
    if 'bowling_style' not in metadata_df.columns:
        metadata_df['bowling_style'] = None
    if 'playing_role' not in metadata_df.columns:
        metadata_df['playing_role'] = None

    # Batting hand: Left (L) or Right (R)
    metadata_df['batting_hand'] = metadata_df['batting_style'].apply(
        lambda x: 'LHB' if pd.notna(x) and 'L' in str(x).upper() else ('RHB' if pd.notna(x) else 'unknown')
    )

    # Bowling type: pace or spin
    def categorize_bowling(style):
        if pd.isna(style):
            return 'unknown'
        style_str = str(style).upper()

        # Spin indicators
        if any(indicator in style_str for indicator in ['SL', 'OB', 'LB', 'LBG', 'SLOW']):
            return 'spin'

        # Pace indicators
        if any(indicator in style_str for indicator in ['FAST', 'MEDIUM', 'F', 'M']):
            return 'pace'

        return 'unknown'

    metadata_df['bowling_type'] = metadata_df['bowling_style'].apply(categorize_bowling)

    # Bowling arm: Left or Right
    metadata_df['bowling_arm'] = metadata_df['bowling_style'].apply(
        lambda x: 'left-arm' if pd.notna(x) and 'L' in str(x).upper() else ('right-arm' if pd.notna(x) else 'unknown')
    )

    # Standardize playing role
    def standardize_role(role):
        if pd.isna(role):
            return 'unknown'
        role_str = str(role).lower()

        if 'all' in role_str:
            return 'allrounder'
        elif 'bat' in role_str:
            return 'batter'
        elif 'bowl' in role_str:
            return 'bowler'
        elif 'keep' in role_str or 'wicket' in role_str:
            return 'wicketkeeper'
        else:
            return 'unknown'

    metadata_df['role_category'] = metadata_df['playing_role'].apply(standardize_role)

    # Statistics
    print("\nPlayer categories:")
    print("\nBatting hand:")
    print(metadata_df['batting_hand'].value_counts())

    print("\nBowling type:")
    print(metadata_df['bowling_type'].value_counts())

    print("\nPlaying role:")
    print(metadata_df['role_category'].value_counts())

    return metadata_df


def main():
    """Fetch and process player metadata."""

    # Paths
    project_root = Path(__file__).parent.parent
    ipl_file = project_root / 'data' / 'ipl_matches.parquet'
    output_file = project_root / 'data' / 'player_metadata.csv'

    print("="*60)
    print("PROCESSING PLAYER METADATA")
    print("="*60)

    # Load IPL data
    print(f"\nLoading IPL data from {ipl_file}...")
    ipl_df = pd.read_parquet(ipl_file)
    print(f"✓ Loaded {len(ipl_df):,} balls")

    # Load existing player info
    player_meta_df = load_existing_player_info()

    # Map player IDs
    metadata = map_player_ids(ipl_df, player_meta_df)

    # Categorize players
    metadata = categorize_players(metadata)

    # Save to CSV
    print(f"\nSaving to {output_file}...")
    metadata.to_csv(output_file, index=False)
    print(f"✓ Saved {len(metadata)} players")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total players: {len(metadata)}")

    # Check if we have the 'name' column or use a different one
    if 'name' in metadata.columns:
        print(f"With metadata: {metadata['name'].notna().sum()}")
        print(f"Missing metadata: {metadata['name'].isna().sum()}")

        # Show sample of players with metadata
        if metadata['name'].notna().any():
            print("\nSample of players with metadata:")
            sample = metadata[metadata['name'].notna()].head(10)
            cols = [c for c in ['player_name', 'country', 'role_category', 'batting_hand', 'bowling_type'] if c in sample.columns]
            print(sample[cols].to_string(index=False))

        # Show sample of players missing metadata
        if metadata['name'].isna().any():
            print("\nSample of players MISSING metadata (need manual lookup):")
            missing = metadata[metadata['name'].isna()].head(10)
            print(missing[['player_id', 'player_name']].to_string(index=False))
    else:
        # Use full_name if available
        if 'full_name' in metadata.columns:
            print(f"With metadata: {metadata['full_name'].notna().sum()}")
            print(f"Missing metadata: {metadata['full_name'].isna().sum()}")

    print("\n" + "="*60)
    print("✓ PLAYER METADATA PROCESSING COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
