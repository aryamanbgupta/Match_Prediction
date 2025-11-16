"""
Utility functions for cricWAR implementation.

This module provides helper functions for data processing, validation, and common operations
used across the cricWAR pipeline.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime


def load_cricsheet_match(filepath: Path) -> Dict[str, Any]:
    """
    Load a Cricsheet JSON file.

    Args:
        filepath: Path to the JSON file

    Returns:
        Dictionary containing match data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def is_ipl_match(match_data: Dict[str, Any], year_start: int = 2015, year_end: int = 2022,
                 exclude_years: Optional[List[int]] = None) -> bool:
    """
    Check if a match is an IPL match within the specified year range.

    Args:
        match_data: Match data dictionary from Cricsheet
        year_start: Starting year (inclusive)
        year_end: Ending year (inclusive)
        exclude_years: List of years to exclude (e.g., [2020])

    Returns:
        True if match is an IPL match in the specified range
    """
    if exclude_years is None:
        exclude_years = [2020]  # Default: exclude 2020 (played outside India)

    info = match_data.get('info', {})

    # Check if it's an IPL match
    event = info.get('event', {})
    if isinstance(event, dict):
        event_name = event.get('name', '')
    else:
        event_name = str(event) if event else ''

    is_ipl = 'Indian Premier League' in event_name or 'IPL' in event_name

    if not is_ipl:
        return False

    # Check year
    dates = info.get('dates', [])
    if not dates:
        return False

    match_date = dates[0]  # First date
    try:
        year = datetime.strptime(match_date, '%Y-%m-%d').year
    except:
        return False

    return year_start <= year <= year_end and year not in exclude_years


def extract_player_id(player_name: str, registry: Dict[str, str]) -> Optional[str]:
    """
    Extract player ID from registry.

    Args:
        player_name: Player's name
        registry: Registry mapping names to IDs

    Returns:
        Player ID or None if not found
    """
    people = registry.get('people', {})
    return people.get(player_name)


def normalize_runs(runs: int) -> int:
    """
    Normalize rare run outcomes to standard values.

    Args:
        runs: Actual runs scored (0-7+)

    Returns:
        Normalized runs (0, 1, 2, 4, 6)
    """
    if runs in [0, 1, 2, 4, 6]:
        return runs
    elif runs == 3:
        return 2  # Treat 3 as 2
    elif runs == 5:
        return 4  # Treat 5 as 4
    else:  # 7+
        return 6  # Treat 7+ as 6


def get_bowling_phase(over: int) -> str:
    """
    Determine the phase of the innings based on over number.

    Args:
        over: Over number (0-based)

    Returns:
        Phase name: 'powerplay', 'middle', or 'death'
    """
    if over < 6:
        return 'powerplay'
    elif over < 16:
        return 'middle'
    else:
        return 'death'


def is_powerplay(over: int) -> bool:
    """Check if the over is in powerplay (first 6 overs)."""
    return over < 6


def calculate_run_rate(runs: int, balls: int) -> float:
    """
    Calculate run rate.

    Args:
        runs: Total runs scored
        balls: Total balls bowled

    Returns:
        Run rate (runs per over, 6 balls)
    """
    if balls == 0:
        return 0.0
    return (runs / balls) * 6


def get_wicket_info(delivery: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract wicket information from a delivery.

    Args:
        delivery: Delivery dictionary

    Returns:
        Wicket information or None if no wicket fell
    """
    wickets = delivery.get('wickets', [])
    if not wickets:
        return None

    # Return first wicket (typically only one per ball)
    return wickets[0]


def get_extras_info(delivery: Dict[str, Any]) -> Dict[str, int]:
    """
    Extract extras information from a delivery.

    Args:
        delivery: Delivery dictionary

    Returns:
        Dictionary with extras breakdown
    """
    runs = delivery.get('runs', {})
    extras = runs.get('extras', 0)

    # Get extras breakdown
    extras_dict = {
        'total': extras,
        'wides': delivery.get('extras', {}).get('wides', 0),
        'noballs': delivery.get('extras', {}).get('noballs', 0),
        'byes': delivery.get('extras', {}).get('byes', 0),
        'legbyes': delivery.get('extras', {}).get('legbyes', 0),
        'penalty': delivery.get('extras', {}).get('penalty', 0),
    }

    return extras_dict


def create_match_id(info: Dict[str, Any]) -> str:
    """
    Create a unique match ID from match info.

    Args:
        info: Match info dictionary

    Returns:
        Unique match ID string
    """
    dates = info.get('dates', [])
    teams = info.get('teams', [])
    venue = info.get('venue', '')

    if dates and len(teams) >= 2:
        date_str = dates[0].replace('-', '')
        team1 = teams[0].replace(' ', '').lower()
        team2 = teams[1].replace(' ', '').lower()
        return f"{date_str}_{team1}_vs_{team2}"

    return "unknown_match"


def validate_game_state(over: int, wickets_lost: int, max_overs: int = 20,
                       max_wickets: int = 10) -> bool:
    """
    Validate that game state is legal.

    Args:
        over: Over number (0-based)
        wickets_lost: Number of wickets lost
        max_overs: Maximum overs in format (20 for T20)
        max_wickets: Maximum wickets (10)

    Returns:
        True if state is valid
    """
    return 0 <= over < max_overs and 0 <= wickets_lost <= max_wickets


def get_platoon_advantage(batter_hand: str, bowler_hand: str) -> str:
    """
    Determine platoon advantage.

    Args:
        batter_hand: 'LHB' or 'RHB'
        bowler_hand: Contains 'L' for left-arm or 'R' for right-arm

    Returns:
        'same' or 'opposite'
    """
    if not batter_hand or not bowler_hand:
        return 'unknown'

    batter_left = 'L' in batter_hand.upper()
    bowler_left = 'L' in bowler_hand.upper()

    if batter_left == bowler_left:
        return 'same'
    else:
        return 'opposite'


def get_bowling_type(bowling_style: str) -> str:
    """
    Categorize bowling style as pace or spin.

    Args:
        bowling_style: Bowling style code (e.g., 'RF', 'LFM', 'SLA', 'OB')

    Returns:
        'pace' or 'spin'
    """
    if not bowling_style:
        return 'unknown'

    style_upper = bowling_style.upper()

    # Pace bowlers (Fast, Fast-Medium, Medium)
    pace_indicators = ['F', 'M']

    # Spin bowlers (Leg-break, Off-break, Slow Left-arm)
    spin_indicators = ['SL', 'OB', 'LB', 'LBG']

    # Check for spin first (more specific)
    for indicator in spin_indicators:
        if indicator in style_upper:
            return 'spin'

    # Then check for pace
    for indicator in pace_indicators:
        if indicator in style_upper:
            return 'pace'

    return 'unknown'


def calculate_confidence_interval(values: np.ndarray, confidence: float = 0.95) -> tuple:
    """
    Calculate confidence interval from array of values.

    Args:
        values: Array of values
        confidence: Confidence level (default 0.95)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    return np.percentile(values, lower_percentile), np.percentile(values, upper_percentile)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.

    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero

    Returns:
        Result of division or default
    """
    if denominator == 0:
        return default
    return numerator / denominator


def print_progress(current: int, total: int, prefix: str = '', suffix: str = '',
                  bar_length: int = 50):
    """
    Print progress bar.

    Args:
        current: Current progress
        total: Total items
        prefix: Prefix string
        suffix: Suffix string
        bar_length: Length of progress bar
    """
    filled_length = int(bar_length * current // total)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    percent = f"{100 * (current / float(total)):.1f}" if total > 0 else "0.0"
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if current == total:
        print()


def save_dataframe(df: pd.DataFrame, filepath: Path, format: str = 'parquet'):
    """
    Save dataframe to file.

    Args:
        df: DataFrame to save
        filepath: Output filepath
        format: Format ('parquet' or 'csv')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if format == 'parquet':
        df.to_parquet(filepath, index=False)
    elif format == 'csv':
        df.to_csv(filepath, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

    print(f"Saved {len(df)} rows to {filepath}")


def load_dataframe(filepath: Path, format: str = 'parquet') -> pd.DataFrame:
    """
    Load dataframe from file.

    Args:
        filepath: Input filepath
        format: Format ('parquet' or 'csv')

    Returns:
        Loaded DataFrame
    """
    if format == 'parquet':
        return pd.read_parquet(filepath)
    elif format == 'csv':
        return pd.read_csv(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")
