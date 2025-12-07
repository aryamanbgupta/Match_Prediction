"""
Player Metadata Provider for Match Prediction

This module provides access to player metadata (batting style, bowling style, DOB)
from the enriched player CSV file. Used for:
- Batter/bowler hand features
- Bowling type classification (pace vs spin)
- Player age calculation
- Matchup type features

Usage:
    provider = PlayerMetadataProvider('data/all_players_enriched.csv')
    meta = provider.get_player_metadata('abc123')
    # Returns: {'batter_hand': 'right', 'bowler_arm': 'right', 'bowler_type': 'medium-fast', ...}
"""

import pandas as pd
from datetime import datetime
from typing import Dict, Optional, Tuple
from pathlib import Path


class PlayerMetadataProvider:
    """
    Provides player metadata lookups by cricsheet_id.

    DESIGN DECISION: Load once, query many times.
    REASONING: The CSV is small (~1MB) and lookups are O(1) with dict.
    """

    # Bowling style classification mappings
    PACE_STYLES = {
        'Right-arm fast', 'Left-arm fast',
        'Right-arm medium-fast', 'Left-arm medium-fast',
        'Right-arm fast-medium', 'Left-arm fast-medium',
        'Right-arm medium', 'Left-arm medium',
        'Right-arm slow-medium', 'Left-arm slow-medium',
    }

    SPIN_STYLES = {
        'Right-arm offbreak', 'Slow left-arm orthodox',
        'Legbreak', 'Legbreak googly',
        'Left-arm wrist-spin', 'Left-arm chinaman',
        'Right-arm slow', 'Left-arm slow',
    }

    # More granular bowling type mapping
    BOWLING_TYPE_MAP = {
        # Pace - Fast
        'Right-arm fast': 'fast',
        'Left-arm fast': 'fast',
        # Pace - Medium-fast
        'Right-arm medium-fast': 'medium-fast',
        'Left-arm medium-fast': 'medium-fast',
        'Right-arm fast-medium': 'medium-fast',
        'Left-arm fast-medium': 'medium-fast',
        # Pace - Medium
        'Right-arm medium': 'medium',
        'Left-arm medium': 'medium',
        'Right-arm slow-medium': 'medium',
        'Left-arm slow-medium': 'medium',
        # Spin - Offspin
        'Right-arm offbreak': 'offspin',
        # Spin - Orthodox (left-arm finger spin)
        'Slow left-arm orthodox': 'orthodox',
        # Spin - Legspin
        'Legbreak': 'legspin',
        'Legbreak googly': 'legspin',
        # Spin - Wrist spin
        'Left-arm wrist-spin': 'wrist-spin',
        'Left-arm chinaman': 'wrist-spin',
        # Slow/part-time
        'Right-arm slow': 'part-time',
        'Left-arm slow': 'part-time',
        'Right-arm bowler': 'unknown',
    }

    def __init__(self, csv_path: str = 'data/all_players_enriched.csv'):
        """
        Load player metadata from CSV.

        Args:
            csv_path: Path to the enriched player CSV
        """
        self.csv_path = Path(csv_path)
        self._load_data()

    def _load_data(self):
        """Load and index player data by cricsheet_id"""
        print(f"Loading player metadata from {self.csv_path}...")

        df = pd.read_csv(self.csv_path)

        # Create lookup dict indexed by cricsheet_id
        self.players = {}

        for _, row in df.iterrows():
            cricsheet_id = row['cricsheet_id']

            # Parse batting style
            batting_style = row.get('batting_style', '')
            if pd.isna(batting_style):
                batting_style = ''
            batter_hand = self._parse_batting_hand(batting_style)

            # Parse bowling style
            bowling_style = row.get('bowling_style', '')
            if pd.isna(bowling_style):
                bowling_style = ''
            bowler_arm, is_pace, bowling_type = self._parse_bowling_style(bowling_style)

            # Parse DOB
            dob = row.get('dob', '')
            if pd.isna(dob) or dob == '':
                dob_parsed = None
            else:
                try:
                    dob_parsed = datetime.strptime(str(dob), '%Y-%m-%d')
                except:
                    dob_parsed = None

            self.players[cricsheet_id] = {
                'name': row.get('name', ''),
                'full_name': row.get('full_name', ''),
                'country': row.get('country', ''),
                'dob': dob_parsed,
                'batting_style_raw': batting_style,
                'bowling_style_raw': bowling_style,
                'batter_hand': batter_hand,  # 'right', 'left', or 'unknown'
                'bowler_arm': bowler_arm,    # 'right', 'left', or 'unknown'
                'is_pace': is_pace,          # True, False, or None
                'bowling_type': bowling_type, # 'fast', 'medium-fast', 'offspin', etc.
            }

        print(f"  Loaded {len(self.players):,} players")

        # Stats
        hands = [p['batter_hand'] for p in self.players.values()]
        arms = [p['bowler_arm'] for p in self.players.values()]
        print(f"  Batting: {hands.count('right')} right, {hands.count('left')} left, {hands.count('unknown')} unknown")
        print(f"  Bowling: {arms.count('right')} right, {arms.count('left')} left, {arms.count('unknown')} unknown")

    def _parse_batting_hand(self, style: str) -> str:
        """Parse batting hand from batting_style string"""
        if not style:
            return 'unknown'
        style_lower = style.lower()
        if 'right' in style_lower:
            return 'right'
        elif 'left' in style_lower:
            return 'left'
        return 'unknown'

    def _parse_bowling_style(self, style: str) -> Tuple[str, Optional[bool], str]:
        """
        Parse bowling style into arm, pace/spin, and type.

        Returns:
            Tuple of (arm, is_pace, bowling_type)
        """
        if not style:
            return ('unknown', None, 'unknown')

        # Determine arm
        style_lower = style.lower()
        if 'right' in style_lower:
            arm = 'right'
        elif 'left' in style_lower:
            arm = 'left'
        else:
            arm = 'unknown'

        # Determine pace vs spin
        if style in self.PACE_STYLES:
            is_pace = True
        elif style in self.SPIN_STYLES:
            is_pace = False
        else:
            is_pace = None

        # Determine specific type
        bowling_type = self.BOWLING_TYPE_MAP.get(style, 'unknown')

        return (arm, is_pace, bowling_type)

    def get_player_metadata(self, cricsheet_id: str) -> Dict:
        """
        Get metadata for a player by cricsheet_id.

        Args:
            cricsheet_id: The player's cricsheet identifier

        Returns:
            Dict with player metadata, or default values if not found
        """
        if cricsheet_id in self.players:
            return self.players[cricsheet_id]

        # Return defaults for unknown players
        return {
            'name': '',
            'full_name': '',
            'country': '',
            'dob': None,
            'batting_style_raw': '',
            'bowling_style_raw': '',
            'batter_hand': 'unknown',
            'bowler_arm': 'unknown',
            'is_pace': None,
            'bowling_type': 'unknown',
        }

    def get_batter_hand(self, cricsheet_id: str) -> str:
        """Get batter's dominant hand ('right', 'left', 'unknown')"""
        return self.get_player_metadata(cricsheet_id)['batter_hand']

    def get_bowler_arm(self, cricsheet_id: str) -> str:
        """Get bowler's arm ('right', 'left', 'unknown')"""
        return self.get_player_metadata(cricsheet_id)['bowler_arm']

    def get_is_pace(self, cricsheet_id: str) -> Optional[bool]:
        """Get whether bowler is pace (True) or spin (False), None if unknown"""
        return self.get_player_metadata(cricsheet_id)['is_pace']

    def get_bowling_type(self, cricsheet_id: str) -> str:
        """Get granular bowling type ('fast', 'medium-fast', 'offspin', etc.)"""
        return self.get_player_metadata(cricsheet_id)['bowling_type']

    def get_player_age(self, cricsheet_id: str, as_of_date) -> Optional[float]:
        """
        Calculate player's age as of a given date.

        Args:
            cricsheet_id: Player identifier
            as_of_date: Date to calculate age at (datetime or string)

        Returns:
            Age in years (float), or None if DOB unknown
        """
        meta = self.get_player_metadata(cricsheet_id)
        dob = meta['dob']

        if dob is None:
            return None

        # Handle string dates
        if isinstance(as_of_date, str):
            as_of_date = datetime.strptime(as_of_date, '%Y-%m-%d')

        # Calculate age in years
        age_days = (as_of_date - dob).days
        return age_days / 365.25

    def get_matchup_type(self, batter_id: str, bowler_id: str) -> str:
        """
        Get encoded matchup type between batter and bowler.

        Examples: 'RHB_vs_offspin', 'LHB_vs_pace', 'RHB_vs_orthodox'
        """
        batter_hand = self.get_batter_hand(batter_id)
        bowling_type = self.get_bowling_type(bowler_id)
        is_pace = self.get_is_pace(bowler_id)

        # Simplified hand label
        hand_label = 'RHB' if batter_hand == 'right' else ('LHB' if batter_hand == 'left' else 'UNK')

        # Use pace/spin if specific type unknown
        if bowling_type == 'unknown' and is_pace is not None:
            type_label = 'pace' if is_pace else 'spin'
        else:
            type_label = bowling_type

        return f"{hand_label}_vs_{type_label}"

    def get_spin_matchup_advantage(self, batter_id: str, bowler_id: str) -> int:
        """
        Check if batter has known spin matchup advantage.

        Known advantages:
        - LHB vs right-arm offspin (ball turning away)
        - RHB vs legspin (ball turning away)

        Returns:
            1 if batter has advantage, -1 if bowler has advantage, 0 if neutral/unknown
        """
        batter_hand = self.get_batter_hand(batter_id)
        bowling_type = self.get_bowling_type(bowler_id)

        # LHB vs offspin = advantage (ball turning away)
        if batter_hand == 'left' and bowling_type == 'offspin':
            return 1

        # RHB vs legspin = slight advantage (ball turning away)
        if batter_hand == 'right' and bowling_type == 'legspin':
            return 1

        # RHB vs offspin = bowler advantage (ball turning in)
        if batter_hand == 'right' and bowling_type == 'offspin':
            return -1

        # LHB vs orthodox = bowler advantage (ball turning in)
        if batter_hand == 'left' and bowling_type == 'orthodox':
            return -1

        return 0

    def get_same_arm_matchup(self, batter_id: str, bowler_id: str) -> Optional[bool]:
        """
        Check if batter and bowler have same dominant side.

        Returns:
            True if same side, False if opposite, None if unknown
        """
        batter_hand = self.get_batter_hand(batter_id)
        bowler_arm = self.get_bowler_arm(bowler_id)

        if batter_hand == 'unknown' or bowler_arm == 'unknown':
            return None

        return batter_hand == bowler_arm


# Encoding helpers for model features
def encode_batter_hand(hand: str) -> int:
    """Encode batter hand: right=0, left=1, unknown=2"""
    return {'right': 0, 'left': 1, 'unknown': 2}.get(hand, 2)

def encode_bowler_arm(arm: str) -> int:
    """Encode bowler arm: right=0, left=1, unknown=2"""
    return {'right': 0, 'left': 1, 'unknown': 2}.get(arm, 2)

def encode_is_pace(is_pace: Optional[bool]) -> int:
    """Encode pace/spin: spin=0, pace=1, unknown=2"""
    if is_pace is None:
        return 2
    return 1 if is_pace else 0

def encode_bowling_type(bowling_type: str) -> int:
    """Encode bowling type to integer"""
    type_map = {
        'fast': 0,
        'medium-fast': 1,
        'medium': 2,
        'offspin': 3,
        'orthodox': 4,
        'legspin': 5,
        'wrist-spin': 6,
        'part-time': 7,
        'unknown': 8,
    }
    return type_map.get(bowling_type, 8)


# Testing
if __name__ == "__main__":
    provider = PlayerMetadataProvider()

    # Test some known players
    test_ids = [
        ('4a8a2e3b', 'MS Dhoni'),      # RHB, right-arm medium
        ('fe93fd9d', 'Ravindra Jadeja'), # LHB, slow left-arm orthodox
        ('e62dd25d', 'Kagiso Rabada'),   # LHB, right-arm fast
        ('5b7ab5a9', 'Varun Chakravarthy'), # RHB, legbreak googly
    ]

    print("\n=== TEST LOOKUPS ===")
    for cid, name in test_ids:
        meta = provider.get_player_metadata(cid)
        print(f"\n{name} ({cid}):")
        print(f"  Hand: {meta['batter_hand']}, Arm: {meta['bowler_arm']}")
        print(f"  Is Pace: {meta['is_pace']}, Type: {meta['bowling_type']}")

        # Test age calculation
        age = provider.get_player_age(cid, '2024-06-15')
        if age:
            print(f"  Age (Jun 2024): {age:.1f} years")

    # Test matchup
    print("\n=== TEST MATCHUPS ===")
    # Jadeja (LHB) vs Chakravarthy (legspin)
    matchup = provider.get_matchup_type('fe93fd9d', '5b7ab5a9')
    advantage = provider.get_spin_matchup_advantage('fe93fd9d', '5b7ab5a9')
    print(f"Jadeja vs Chakravarthy: {matchup}, advantage: {advantage}")
