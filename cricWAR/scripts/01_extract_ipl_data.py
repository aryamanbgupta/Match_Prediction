"""
Extract IPL ball-by-ball data from Cricsheet JSON files.

This script:
1. Reads Cricsheet JSON files from ../data/t20s_json/
2. Filters for IPL matches (2015-2022, excluding 2020)
3. Converts nested JSON to flat ball-by-ball dataframe
4. Saves to data/ipl_matches.parquet
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from tqdm import tqdm
from utils import (
    load_cricsheet_match,
    is_ipl_match,
    extract_player_id,
    normalize_runs,
    get_wicket_info,
    get_extras_info,
    create_match_id,
    save_dataframe,
)


def extract_ball_data(match_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract ball-by-ball data from a match.

    Args:
        match_data: Match data from Cricsheet JSON

    Returns:
        List of dictionaries, one per ball
    """
    info = match_data.get('info', {})
    innings_list = match_data.get('innings', [])
    registry = info.get('registry', {})

    # Match-level info
    match_id = create_match_id(info)
    venue = info.get('venue', '')
    match_date = info.get('dates', [''])[0]
    teams = info.get('teams', [])
    toss_winner = info.get('toss', {}).get('winner', '')
    toss_decision = info.get('toss', {}).get('decision', '')
    season = info.get('season', '')
    match_type = info.get('match_type', '')

    balls = []

    for innings_idx, innings in enumerate(innings_list):
        innings_number = innings_idx + 1  # 1-based: 1st innings, 2nd innings
        batting_team = innings.get('team', '')

        # Determine bowling team
        bowling_team = teams[1] if teams[0] == batting_team else teams[0] if len(teams) > 1 else ''

        # Initialize innings state
        score = 0
        wickets = 0
        total_balls = 0

        overs = innings.get('overs', [])
        for over_data in overs:
            over_number = over_data.get('over', 0)  # 0-based

            deliveries = over_data.get('deliveries', [])
            for ball_number, delivery in enumerate(deliveries):
                # Player info
                batter_name = delivery.get('batter', '')
                bowler_name = delivery.get('bowler', '')
                non_striker_name = delivery.get('non_striker', '')

                # Get player IDs
                batter_id = extract_player_id(batter_name, registry)
                bowler_id = extract_player_id(bowler_name, registry)
                non_striker_id = extract_player_id(non_striker_name, registry)

                # Runs info
                runs_info = delivery.get('runs', {})
                batter_runs = runs_info.get('batter', 0)
                extras = runs_info.get('extras', 0)
                total_runs = runs_info.get('total', 0)

                # Extras breakdown
                extras_info = get_extras_info(delivery)

                # Wicket info
                wicket = get_wicket_info(delivery)
                is_wicket = wicket is not None
                dismissal_type = wicket.get('kind', '') if wicket else ''
                dismissed_player = wicket.get('player_out', '') if wicket else ''

                # Ball-level data
                ball_data = {
                    # Match info
                    'match_id': match_id,
                    'venue': venue,
                    'match_date': match_date,
                    'season': season,
                    'match_type': match_type,
                    'toss_winner': toss_winner,
                    'toss_decision': toss_decision,

                    # Innings info
                    'innings': innings_number,
                    'batting_team': batting_team,
                    'bowling_team': bowling_team,

                    # Ball position
                    'over': over_number,
                    'ball_in_over': ball_number,
                    'total_balls': total_balls,

                    # State before ball
                    'score_before': score,
                    'wickets_before': wickets,

                    # Players
                    'batter_name': batter_name,
                    'batter_id': batter_id,
                    'bowler_name': bowler_name,
                    'bowler_id': bowler_id,
                    'non_striker_name': non_striker_name,
                    'non_striker_id': non_striker_id,

                    # Runs
                    'batter_runs': batter_runs,
                    'extras': extras,
                    'total_runs': total_runs,
                    'wides': extras_info['wides'],
                    'noballs': extras_info['noballs'],
                    'byes': extras_info['byes'],
                    'legbyes': extras_info['legbyes'],

                    # Wicket
                    'is_wicket': is_wicket,
                    'dismissal_type': dismissal_type,
                    'dismissed_player': dismissed_player,
                }

                balls.append(ball_data)

                # Update state
                score += total_runs
                if is_wicket:
                    wickets += 1

                # Only count as a ball if not a wide or no-ball
                if extras_info['wides'] == 0 and extras_info['noballs'] == 0:
                    total_balls += 1

    return balls


def main():
    """Extract IPL data from Cricsheet JSON files."""

    # Paths
    project_root = Path(__file__).parent.parent
    cricsheet_dir = project_root.parent / 'data' / 't20s_json'
    output_file = project_root / 'data' / 'ipl_matches.parquet'

    print(f"Reading Cricsheet files from: {cricsheet_dir}")
    print(f"Output will be saved to: {output_file}")

    # Check if cricsheet directory exists
    if not cricsheet_dir.exists():
        raise FileNotFoundError(f"Cricsheet directory not found: {cricsheet_dir}")

    # Get all JSON files
    json_files = list(cricsheet_dir.glob('*.json'))
    print(f"Found {len(json_files)} JSON files")

    # Extract IPL matches
    all_balls = []
    ipl_match_count = 0

    print("\nExtracting IPL matches (2025 season only)...")

    for json_file in tqdm(json_files, desc="Processing files"):
        try:
            match_data = load_cricsheet_match(json_file)

            # Check if it's an IPL match in our range
            if is_ipl_match(match_data, year_start=2025, year_end=2025, exclude_years=[]):
                balls = extract_ball_data(match_data)
                all_balls.extend(balls)
                ipl_match_count += 1

        except Exception as e:
            print(f"\nError processing {json_file.name}: {e}")
            continue

    print(f"\n✓ Extracted {ipl_match_count} IPL matches")
    print(f"✓ Total balls: {len(all_balls):,}")

    # Create DataFrame
    df = pd.DataFrame(all_balls)

    # Add derived columns
    print("\nAdding derived columns...")

    # Convert season to string for consistency
    df['season'] = df['season'].astype(str)

    # Normalize runs (3→2, 5→4, 7+→6)
    df['batter_runs_normalized'] = df['batter_runs'].apply(normalize_runs)

    # Calculate balls remaining
    df['balls_remaining'] = 120 - df['total_balls']

    # Add phase indicator
    df['phase'] = df['over'].apply(lambda o: 'powerplay' if o < 6 else ('death' if o >= 16 else 'middle'))
    df['is_powerplay'] = df['over'] < 6

    # Add run rate
    df['run_rate'] = df.apply(
        lambda x: (x['score_before'] / x['total_balls'] * 6) if x['total_balls'] > 0 else 0.0,
        axis=1
    )

    # Summary statistics
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    print(f"Total matches: {df['match_id'].nunique()}")
    print(f"Total balls: {len(df):,}")
    print(f"Seasons: {sorted([str(s) for s in df['season'].unique()])}")
    print(f"Teams: {df['batting_team'].nunique()}")
    print(f"Venues: {df['venue'].nunique()}")
    print(f"Unique batters: {df['batter_id'].nunique()}")
    print(f"Unique bowlers: {df['bowler_id'].nunique()}")

    print("\n" + "Balls by season:")
    season_counts = df.groupby('season').size().sort_index()
    for season, count in season_counts.items():
        print(f"  {season}: {count:,}")

    print("\n" + "Balls by innings:")
    print(df.groupby('innings').size())

    print("\n" + "Run distribution:")
    print(df['batter_runs'].value_counts().sort_index())

    print("\n" + "Wickets:")
    print(f"Total: {df['is_wicket'].sum():,}")
    print(f"Percentage: {df['is_wicket'].mean()*100:.2f}%")

    # Save to parquet
    print(f"\nSaving to {output_file}...")
    save_dataframe(df, output_file, format='parquet')

    # Also save a CSV sample for inspection
    sample_file = project_root / 'data' / 'ipl_matches_sample.csv'
    df.head(1000).to_csv(sample_file, index=False)
    print(f"✓ Saved sample (1000 rows) to {sample_file}")

    print("\n" + "="*60)
    print("✓ IPL DATA EXTRACTION COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
