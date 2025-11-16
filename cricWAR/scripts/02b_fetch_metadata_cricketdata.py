"""
Fetch player metadata from cricketdata R package for all IPL players.

This script:
1. Loads the list of unique players from IPL data
2. Searches for cricinfo player ID by name using find_player_id()
3. Fetches metadata for each player using fetch_player_meta() with cricinfo ID
4. Maps to batting/bowling styles and playing roles
5. Saves complete metadata to data/player_metadata.csv

Note: Uses two-step approach to map Cricsheet IDs to cricinfo IDs.
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import time


def find_cricinfo_player_id(cricketdata, player_name: str):
    """
    Find cricinfo player ID by searching for player name.

    Args:
        cricketdata: R cricketdata package
        player_name: Player name to search

    Returns:
        Cricinfo player ID as string or None if not found
    """
    try:
        # Search for player by name
        r_result = cricketdata.find_player_id(player_name)

        # Convert to pandas
        with localconverter(robjects.default_converter + pandas2ri.converter):
            result_df = robjects.conversion.rpy2py(r_result)

        if len(result_df) > 0 and 'ID' in result_df.columns:
            # Return first matching ID
            player_id = result_df.iloc[0]['ID']
            # Check if it's a valid ID (not NA)
            if pd.notna(player_id):
                return str(int(player_id))
        return None

    except Exception as e:
        return None


def fetch_player_metadata_by_name(cricketdata, player_name: str):
    """
    Fetch metadata for a single player from cricketdata.

    This uses a two-step approach:
    1. Find cricinfo player ID by name
    2. Fetch metadata using that ID

    Args:
        cricketdata: R cricketdata package
        player_name: Player name

    Returns:
        Dictionary with player metadata or None if fetch fails
    """
    try:
        # Step 1: Find cricinfo player ID
        cricinfo_id = find_cricinfo_player_id(cricketdata, player_name)

        if not cricinfo_id:
            return None

        # Step 2: Fetch metadata with cricinfo ID
        r_meta = cricketdata.fetch_player_meta(playerid=cricinfo_id)

        # Convert to pandas
        with localconverter(robjects.default_converter + pandas2ri.converter):
            meta_df = robjects.conversion.rpy2py(r_meta)

        if len(meta_df) > 0:
            # Return first row as dict, include cricinfo_id
            result = meta_df.iloc[0].to_dict()
            result['cricinfo_id'] = cricinfo_id
            return result
        else:
            return None

    except Exception as e:
        return None


def categorize_batting_hand(batting_style):
    """Extract batting hand (LHB/RHB) from batting style."""
    if pd.isna(batting_style):
        return 'unknown'
    style_str = str(batting_style).upper()
    if 'LEFT' in style_str or 'LH' in style_str:
        return 'LHB'
    elif 'RIGHT' in style_str or 'RH' in style_str:
        return 'RHB'
    return 'unknown'


def categorize_bowling_type(bowling_style):
    """Categorize bowling as pace or spin."""
    if pd.isna(bowling_style):
        return 'unknown'
    style_str = str(bowling_style).upper()

    # Spin indicators
    spin_keywords = ['SPIN', 'LEG', 'OFF', 'BREAK', 'GOOGLY', 'CHINAMAN', 'ORTHODOX']
    if any(keyword in style_str for keyword in spin_keywords):
        return 'spin'

    # Pace indicators
    pace_keywords = ['FAST', 'PACE', 'MEDIUM', 'QUICK', 'SEAM']
    if any(keyword in style_str for keyword in pace_keywords):
        return 'pace'

    return 'unknown'


def categorize_bowling_arm(bowling_style):
    """Extract bowling arm (left/right) from bowling style."""
    if pd.isna(bowling_style):
        return 'unknown'
    style_str = str(bowling_style).upper()
    if 'LEFT' in style_str or 'LH' in style_str:
        return 'left-arm'
    elif 'RIGHT' in style_str or 'RH' in style_str:
        return 'right-arm'
    return 'unknown'


def standardize_playing_role(role):
    """Standardize playing role categories."""
    if pd.isna(role):
        return 'unknown'
    role_str = str(role).lower()

    if 'all' in role_str or 'rounder' in role_str:
        return 'allrounder'
    elif 'bat' in role_str:
        return 'batter'
    elif 'bowl' in role_str:
        return 'bowler'
    elif 'keep' in role_str or 'wicket' in role_str:
        return 'wicketkeeper'
    else:
        return 'unknown'


def convert_r_date(r_date):
    """Convert R date (days since 1970-01-01) to YYYY-MM-DD string."""
    if pd.isna(r_date):
        return None
    try:
        # R's origin is 1970-01-01
        return (pd.to_datetime('1970-01-01') + pd.to_timedelta(r_date, unit='d')).strftime('%Y-%m-%d')
    except (ValueError, TypeError):
        return None


def main():
    """Fetch player metadata from cricketdata for all IPL players."""

    # Paths
    project_root = Path(__file__).parent.parent
    ipl_file = project_root / 'data' / 'ipl_matches.parquet'
    output_file = project_root / 'data' / 'player_metadata.csv'

    print("="*70)
    print("FETCHING PLAYER METADATA FROM CRICKETDATA R PACKAGE")
    print("="*70)

    # Load IPL data to get player list
    print(f"\nLoading IPL data from {ipl_file}...")
    ipl_df = pd.read_parquet(ipl_file)
    print(f"✓ Loaded {len(ipl_df):,} balls")

    # Get unique players
    batters = ipl_df[['batter_id', 'batter_name']].drop_duplicates()
    batters.columns = ['player_id', 'player_name']

    bowlers = ipl_df[['bowler_id', 'bowler_name']].drop_duplicates()
    bowlers.columns = ['player_id', 'player_name']

    non_strikers = ipl_df[['non_striker_id', 'non_striker_name']].drop_duplicates()
    non_strikers.columns = ['player_id', 'player_name']

    all_players = pd.concat([batters, bowlers, non_strikers]).drop_duplicates()
    all_players = all_players[all_players['player_id'].notna()].reset_index(drop=True)

    print(f"✓ Found {len(all_players)} unique players")

    # Load cricketdata R package
    print("\nLoading cricketdata R package...")
    cricketdata = importr('cricketdata')
    print("✓ cricketdata loaded")

    # Fetch metadata for each player
    print(f"\nFetching metadata for {len(all_players)} players...")
    print("(This will take a few minutes...)\n")

    metadata_list = []
    successful = 0
    failed = 0

    for idx, row in tqdm(all_players.iterrows(), total=len(all_players), desc="Fetching metadata"):
        player_id = str(row['player_id'])
        player_name = row['player_name']

        # Fetch metadata using two-step approach (find ID by name, then fetch)
        meta = fetch_player_metadata_by_name(cricketdata, player_name)

        if meta and pd.notna(meta.get('name')) and str(meta.get('name')) != 'NA_character_':
            # Extract relevant fields
            player_data = {
                'player_id': player_id,
                'player_name': player_name,
                'cricinfo_id': meta.get('cricinfo_id', None),
                'full_name': meta.get('name', player_name),
                'dob': meta.get('dob', None),
                'country': meta.get('country', None),
                'batting_style': meta.get('batting_style', None),
                'bowling_style': meta.get('bowling_style', None),
                'playing_role': meta.get('playing_role', None),
            }
            successful += 1
        else:
            # No metadata found
            player_data = {
                'player_id': player_id,
                'player_name': player_name,
                'cricinfo_id': None,
                'full_name': player_name,
                'dob': None,
                'country': None,
                'batting_style': None,
                'bowling_style': None,
                'playing_role': None,
            }
            failed += 1

        metadata_list.append(player_data)

        # Small delay to avoid overwhelming the API
        if idx % 10 == 0 and idx > 0:
            time.sleep(0.3)

    # Create DataFrame
    metadata_df = pd.DataFrame(metadata_list)

    print(f"\n✓ Successfully fetched metadata for {successful} players")
    print(f"✗ Failed to fetch metadata for {failed} players")

    # Convert R date to datetime
    metadata_df['dob'] = metadata_df['dob'].apply(convert_r_date)

    # Add categorized columns
    print("\nCategorizing player attributes...")
    metadata_df['batting_hand'] = metadata_df['batting_style'].apply(categorize_batting_hand)
    metadata_df['bowling_type'] = metadata_df['bowling_style'].apply(categorize_bowling_type)
    metadata_df['bowling_arm'] = metadata_df['bowling_style'].apply(categorize_bowling_arm)
    metadata_df['role_category'] = metadata_df['playing_role'].apply(standardize_playing_role)

    # Statistics
    print("\n" + "="*70)
    print("METADATA STATISTICS")
    print("="*70)

    print(f"\nTotal players: {len(metadata_df)}")
    print(f"With complete metadata: {metadata_df['batting_style'].notna().sum()}")
    print(f"With date of birth: {metadata_df['dob'].notna().sum()}")

    print("\nBatting hand distribution:")
    print(metadata_df['batting_hand'].value_counts().to_string())

    print("\nBowling type distribution:")
    print(metadata_df['bowling_type'].value_counts().to_string())

    print("\nPlaying role distribution:")
    print(metadata_df['role_category'].value_counts().to_string())

    # Sample of successful fetches
    print("\n" + "-"*70)
    print("Sample of players WITH metadata:")
    print("-"*70)
    sample = metadata_df[metadata_df['batting_style'].notna()].head(10)
    display_cols = ['player_name', 'cricinfo_id', 'dob', 'country', 'batting_hand', 'bowling_type', 'role_category']
    print(sample[display_cols].to_string(index=False))

    # Sample of failed fetches
    if failed > 0:
        print("\n" + "-"*70)
        print(f"Sample of players WITHOUT metadata ({failed} total - may need manual lookup):")
        print("-"*70)
        missing = metadata_df[metadata_df['batting_style'].isna()].head(10)
        print(missing[['player_name', 'country']].to_string(index=False))

    # Save to CSV
    print(f"\nSaving metadata to {output_file}...")
    metadata_df.to_csv(output_file, index=False)
    print(f"✓ Saved {len(metadata_df)} players")

    print("\n" + "="*70)
    print("✓ PLAYER METADATA FETCH COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
