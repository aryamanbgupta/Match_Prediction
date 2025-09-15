#!/usr/bin/env python3
"""
Betting Odds Parser - Converts Claude output + match JSONs to betting odds format

Usage:
    python parse_betting_odds.py --claude-output odds_output.txt --matches-dir data/matches --output betting_odds.json
"""


import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import re


def american_to_decimal(american_odds: int) -> float:
    """Convert American odds to decimal odds"""
    if american_odds > 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1


def normalize_team_name(team: str) -> str:
    """Normalize team name for matching"""
    # Handle common variations
    mapping = {
        'south africa': 'South Africa',
        'new zealand': 'New Zealand',
        'sri lanka': 'Sri Lanka',
        'west indies': 'West Indies',
        'united states': 'United States of America',  # Note: full name used in match files
        'usa': 'United States of America',
        'png': 'Papua New Guinea',
        'uae': 'United Arab Emirates',
    }
    
    # First try exact match
    normalized = mapping.get(team.lower(), team)
    
    # If not found, try to match partial names
    if normalized == team:
        for key, value in mapping.items():
            if key in team.lower():
                return value
    
    return normalized

def extract_match_result(match_data: Dict) -> Optional[str]:
    """
    Extracts the winning team's name from the match JSON data.

    Args:
        match_data: A dictionary containing the parsed JSON data of a match.

    Returns:
        The name of the winning team as a string, or None if no winner is specified
        (e.g., for a tied or abandoned match).
    """
    try:
        # The winner's name is located at: match_data['info']['outcome']['winner']
        winner = match_data.get('info', {}).get('outcome', {}).get('winner')
        return winner
    except AttributeError:
        # This handles cases where the nested structure might be malformed
        return None
    
# --------------------vvv THIS FUNCTION HAS BEEN UPDATED AGAIN vvv--------------------
def extract_match_id_from_json(match_data: Dict) -> str:
    """
    Extracts a match ID from match JSON data, replicating the logic
    used by the TestMatchLoader for consistent formatting.
    
    Format: date_team1_team2_venue_city
    All spaces are replaced with underscores.
    """
    info = match_data['info']
    
    date = info['dates'][0]
    
    # Format team names by replacing spaces with underscores
    team1 = info['teams'][0].replace(' ', '_')
    team2 = info['teams'][1].replace(' ', '_')
    
    # Start with the base venue string
    venue = info.get('venue', 'Unknown Venue')
    city = info.get('city')
    
    full_location = venue
    
    # ONLY append the city if it exists and is NOT already in the venue string
    if city and city not in venue:
        full_location = f"{venue}, {city}"
    
    # Format the final location string by replacing spaces
    location_formatted = full_location.replace(' ', '_')
    
    # Assemble the final match_id
    match_id = f"{date}_{team1}_{team2}_{location_formatted}"
    
    return match_id
# --------------------^^^ THIS FUNCTION HAS BEEN UPDATED AGAIN ^^^--------------------


def find_match_file(date: str, team1: str, team2: str, matches_dir: Path) -> Optional[Tuple[Dict, str]]:
    """
    Find matching JSON file by date and teams
    
    Returns tuple of (match_data, match_id) if found, None otherwise
    """
    team1_norm = normalize_team_name(team1)
    team2_norm = normalize_team_name(team2)
    
    # Look for JSON files in the matches directory
    for json_file in matches_dir.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                match_data = json.load(f)
            
            # Check if this is the right match
            match_date = match_data['info']['dates'][0]  # First date
            match_teams = set(match_data['info']['teams'])
            odds_teams = {team1_norm, team2_norm}
            
            if match_date == date and match_teams == odds_teams:
                print(f"✓ Found match: {json_file.name} for {team1} vs {team2} on {date}")
                match_id = extract_match_id_from_json(match_data)
                return match_data, match_id
                
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠ Error reading {json_file.name}: {e}")
            continue
    
    print(f"✗ No match found for {team1} vs {team2} on {date}")
    return None


def parse_claude_output(claude_file: Path) -> List[Tuple[str, str, str, int, int]]:
    """
    Parse Claude's pipe-delimited output
    
    Returns list of (date, team1, team2, odds1, odds2) tuples
    """
    matches = []
    
    with open(claude_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or '|' not in line:
                continue
                
            try:
                parts = line.split('|')
                if len(parts) != 5:
                    print(f"⚠ Line {line_num}: Expected 5 parts, got {len(parts)}: {line}")
                    continue
                
                date, team1, team2, odds1_str, odds2_str = parts
                
                # Parse odds (remove + sign, keep - sign)
                odds1 = int(odds1_str.replace('+', ''))
                odds2 = int(odds2_str.replace('+', ''))
                
                matches.append((date, team1, team2, odds1, odds2))
                
            except ValueError as e:
                print(f"⚠ Line {line_num}: Error parsing odds: {line} - {e}")
                continue
    
    print(f"✓ Parsed {len(matches)} matches from Claude output")
    return matches


def create_betting_odds_json(claude_output: Path, matches_dir: Path) -> Dict:
    """
    Main function to create betting odds JSON from Claude output and match files
    """
    # Parse Claude output
    odds_matches = parse_claude_output(claude_output)
    
    # Create final structure
    betting_odds = {"matches": []}
    
    for date, team1, team2, odds1, odds2 in odds_matches:
        # Find corresponding match JSON
        match_result = find_match_file(date, team1, team2, matches_dir)
        
        if not match_result:
            print(f"⚠ Skipping {team1} vs {team2} on {date} - no match data found")
            continue
        
        match_data, match_id = match_result
        
        # Extract match info
        venue = match_data['info']['venue']
        match_teams = match_data['info']['teams']  # Ordered list from JSON
        
        # Ensure team order matches between odds and JSON
        # Use JSON team order as canonical
        json_team1 = match_teams[0]
        json_team2 = match_teams[1]
        
        # Map odds to correct teams based on provided names
        team1_norm = normalize_team_name(team1)
        team2_norm = normalize_team_name(team2)
        
        if team1_norm == json_team1:
            team1_odds, team2_odds = odds1, odds2
        elif team2_norm == json_team1:
            # Swap if order is reversed
            team1_odds, team2_odds = odds2, odds1
        else:
            print(f"⚠ Team name mismatch for {date}: odds=({team1},{team2}) vs json=({match_teams})")
            print(f"  Normalized odds teams: ({team1_norm},{team2_norm})")
            continue
        
        actual_winner = extract_match_result(match_data)

        # Create match entry using extracted match ID
        match_entry = {
            "match_id": match_id,  # Use the extracted match ID
            "date": date,
            "team1": json_team1,
            "team2": json_team2,
            "venue": venue,
            "actual_winner": actual_winner,
            "odds": {
                "winner": {
                    json_team1: american_to_decimal(team1_odds),
                    json_team2: american_to_decimal(team2_odds),
                    "timestamp": f"{date}T10:00:00Z"
                }
            }
        }
        
        betting_odds["matches"].append(match_entry)
        print(f"✓ Added: {json_team1} vs {json_team2} at {venue}")
        print(f"  Match ID: {match_id}")
    
    return betting_odds


def main():
    parser = argparse.ArgumentParser(description="Parse betting odds and merge with match data")
    parser.add_argument("--claude-output", type=Path, required=True,
                       help="Text file with Claude's parsed odds output")
    parser.add_argument("--matches-dir", type=Path, required=True,
                       help="Directory containing match JSON files")
    parser.add_argument("--output", type=Path, default="betting_odds.json",
                       help="Output JSON file")
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed output")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.claude_output.exists():
        print(f"Error: Claude output file not found: {args.claude_output}")
        return
    
    if not args.matches_dir.exists():
        print(f"Error: Matches directory not found: {args.matches_dir}")
        return
    
    print(f"Processing odds from: {args.claude_output}")
    print(f"Using match data from: {args.matches_dir}")
    print("-" * 50)
    
    # Create betting odds JSON
    betting_odds = create_betting_odds_json(args.claude_output, args.matches_dir)
    
    # Save output
    with open(args.output, 'w') as f:
        json.dump(betting_odds, f, indent=2)
    
    print("-" * 50)
    print(f"✓ Created betting odds file: {args.output}")
    print(f"✓ Processed {len(betting_odds['matches'])} matches")
    
    # Show summary
    if args.verbose:
        print("\nMatches processed:")
        for match in betting_odds['matches']:
            odds = match['odds']['winner']
            team1_prob = 1 / odds[match['team1']]
            team2_prob = 1 / odds[match['team2']]
            margin = (team1_prob + team2_prob - 1) * 100
            print(f"  {match['date']}: {match['team1']} ({team1_prob:.1%}) vs {match['team2']} ({team2_prob:.1%})")
            print(f"    Match ID: {match['match_id']}")
            print(f"    Bookmaker margin: {margin:.1f}%")


if __name__ == "__main__":
    main()