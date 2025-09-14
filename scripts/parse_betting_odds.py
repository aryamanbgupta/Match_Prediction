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
        'united states': 'United States',
    }
    return mapping.get(team.lower(), team)


def extract_venue_short(venue: str) -> str:
    """Extract short venue name from full venue string"""
    # Example: "Kensington Oval, Bridgetown, Barbados" -> "kensington_oval"
    return venue.split(',')[0].lower().replace(' ', '_')


def create_match_id(date: str, team1: str, team2: str, venue: str) -> str:
    """Create standardized match ID"""
    venue_short = extract_venue_short(venue)
    return f"{date}_{team1.lower().replace(' ', '_')}_{team2.lower().replace(' ', '_')}_{venue_short}"


def find_match_file(date: str, team1: str, team2: str, matches_dir: Path) -> Optional[Dict]:
    """
    Find matching JSON file by date and teams
    
    Returns the match JSON data if found, None otherwise
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
                return match_data
                
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
        match_data = find_match_file(date, team1, team2, matches_dir)
        
        if not match_data:
            print(f"⚠ Skipping {team1} vs {team2} on {date} - no match data found")
            continue
        
        # Extract match info
        venue = match_data['info']['venue']
        match_teams = match_data['info']['teams']  # Ordered list from JSON
        
        # Ensure team order matches between odds and JSON
        # Use JSON team order as canonical
        if team1 in match_teams and team2 in match_teams:
            json_team1 = match_teams[0]
            json_team2 = match_teams[1]
            
            # Map odds to correct teams based on JSON order
            if team1 == json_team1:
                team1_odds, team2_odds = odds1, odds2
            else:
                team1_odds, team2_odds = odds2, odds1
                
        else:
            print(f"⚠ Team name mismatch for {date}: odds=({team1},{team2}) vs json=({match_teams})")
            continue
        
        # Create match entry
        match_entry = {
            "match_id": create_match_id(date, json_team1, json_team2, venue),
            "date": date,
            "team1": json_team1,
            "team2": json_team2,
            "venue": venue,
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
            print(f"  {match['date']}: {match['team1']} ({team1_prob:.1%}) vs {match['team2']} ({team2_prob:.1%})")


if __name__ == "__main__":
    main()