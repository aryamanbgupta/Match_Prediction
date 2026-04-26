#!/usr/bin/env python3
"""
Check match IDs between test data and betting odds file
"""

import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from sim_eval.loaders import TestMatchLoader, BettingOddsLoader


def check_match_ids():
    """Compare match IDs from test data vs betting odds"""
    
    # Load test matches
    print("Loading test matches...")
    loader = TestMatchLoader()
    matches = loader.load_matches('data/betting_test')
    
    test_match_ids = [match_id for match_id, _ in matches]
    print(f"\nFound {len(test_match_ids)} test matches")
    print("First 5 test match IDs:")
    for match_id in test_match_ids[:5]:
        print(f"  {match_id}")
    
    # Load betting odds
    print("\n" + "-"*60)
    print("Loading betting odds...")
    odds = BettingOddsLoader.load_odds('betting_odds_v2.json')
    
    odds_match_ids = list(odds.keys())
    print(f"\nFound {len(odds_match_ids)} matches in odds file")
    print("First 5 odds match IDs:")
    for match_id in odds_match_ids[:5]:
        print(f"  {match_id}")
    
    # Find matches
    print("\n" + "-"*60)
    print("Checking for matches...")
    
    # Exact matches
    exact_matches = set(test_match_ids) & set(odds_match_ids)
    print(f"\nExact matches: {len(exact_matches)}")
    if exact_matches:
        print("Examples:")
        for match_id in list(exact_matches)[:3]:
            print(f"  ✓ {match_id}")
    
    # Check for partial matches (ignoring venue differences)
    print("\nChecking for partial matches (date + teams)...")
    test_partial = {}
    for match_id in test_match_ids:
        parts = match_id.split('_')
        if len(parts) >= 3:
            partial = '_'.join(parts[:3])  # date_team1_team2
            test_partial[partial] = match_id
    
    odds_partial = {}
    for match_id in odds_match_ids:
        parts = match_id.split('_')
        if len(parts) >= 3:
            partial = '_'.join(parts[:3])  # date_team1_team2
            odds_partial[partial] = match_id
    
    partial_matches = set(test_partial.keys()) & set(odds_partial.keys())
    print(f"Partial matches (same date and teams): {len(partial_matches)}")
    
    if partial_matches:
        print("\nMismatched venue formats:")
        for partial in list(partial_matches)[:5]:
            test_id = test_partial[partial]
            odds_id = odds_partial[partial]
            if test_id != odds_id:
                print(f"  Test: {test_id}")
                print(f"  Odds: {odds_id}")
                print()
    
    # Find unmatched
    unmatched_test = set(test_match_ids) - set(odds_match_ids)
    unmatched_odds = set(odds_match_ids) - set(test_match_ids)
    
    if unmatched_test:
        print(f"\nTest matches without odds: {len(unmatched_test)}")
        print("First 3 examples:")
        for match_id in list(unmatched_test)[:3]:
            print(f"  ✗ {match_id}")
    
    if unmatched_odds:
        print(f"\nOdds without test matches: {len(unmatched_odds)}")
        print("First 3 examples:")
        for match_id in list(unmatched_odds)[:3]:
            print(f"  ✗ {match_id}")


if __name__ == "__main__":
    check_match_ids()