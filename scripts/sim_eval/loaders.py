import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import numpy as np

# Import from the simulation file
from sim_v1_2 import MatchState, Player, TeamLineup

class TestMatchLoader:
    """Loads test matches and creates initial MatchState objects for simulation"""
    
    def __init__(self, batter_encoder=None, bowler_encoder=None):
        """
        Args:
            batter_encoder: Optional encoder for player IDs (for future use)
            bowler_encoder: Optional encoder for player IDs (for future use)
        """
        self.batter_encoder = batter_encoder
        self.bowler_encoder = bowler_encoder
    
    def load_matches(self, folder_path: str) -> List[Tuple[str, MatchState]]:
        """Load all test matches from folder
        
        Returns:
            List of (match_id, initial_match_state) tuples
        """
        matches = []
        json_files = sorted(Path(folder_path).glob('*.json'))
        
        print(f"Loading {len(json_files)} test matches...")
        
        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                match_id, match_state = self._create_match_state(data)
                if match_state:
                    matches.append((match_id, match_state))
                    print(f"  Loaded: {match_id}")
                
            except Exception as e:
                print(f"  Error loading {file_path.name}: {e}")
        
        print(f"Successfully loaded {len(matches)} matches")
        return matches
    
    def _create_match_state(self, data: dict) -> Tuple[Optional[str], Optional[MatchState]]:
        """Create initial MatchState from match JSON
        
        Design decisions:
        - Only extract initial conditions (teams, venue, toss)
        - Use player registry to map names to IDs
        - Assume batting order from first appearance
        """
        try:
            # Extract match info
            info = data['info']
            
            # Get teams
            teams = info['teams']
            team1_name = teams[0]
            team2_name = teams[1]
            
            # Get venue and date
            venue = info.get('venue', 'Unknown')
            dates = info.get('dates', ['2024-01-01'])
            match_date = datetime.strptime(dates[0], '%Y-%m-%d')
            
            # Create match ID
            match_id = f"{dates[0]}_{team1_name}_{team2_name}_{venue}".replace(' ', '_')
            
            # Get toss info
            toss = info.get('toss', {})
            toss_winner = toss.get('winner', team1_name)
            toss_decision = toss.get('decision', 'bat')
            
            # Determine batting first
            if toss_decision == 'bat':
                batting_first = toss_winner
            else:
                batting_first = team2_name if toss_winner == team1_name else team1_name
            
            # Get player registry
            player_registry = info['registry']['people']
            
            # Extract players from innings (in batting order)
            team1_players = self._extract_team_players(data, team1_name, player_registry)
            team2_players = self._extract_team_players(data, team2_name, player_registry)
            
            # Ensure we have 11 players per team
            if len(team1_players) < 11 or len(team2_players) < 11:
                print(f"    Warning: Incomplete team lineups for {match_id}")
                # Pad with dummy players if needed
                while len(team1_players) < 11:
                    team1_players.append(Player(f"player_{len(team1_players)}", f"Player {len(team1_players)}", team1_name))
                while len(team2_players) < 11:
                    team2_players.append(Player(f"player_{len(team2_players)}", f"Player {len(team2_players)}", team2_name))
            
            # Create lineups
            team1_lineup = TeamLineup(team1_name, team1_players[:11])
            team2_lineup = TeamLineup(team2_name, team2_players[:11])
            
            # Create match state
            match_state = MatchState(
                team1_lineup=team1_lineup,
                team2_lineup=team2_lineup,
                batting_first=batting_first,
                venue=venue,
                match_date=match_date
            )
            
            return match_id, match_state
            
        except Exception as e:
            print(f"    Error creating match state: {e}")
            return None, None
        
    def _extract_team_players(self, data: dict, team_name: str, player_registry: dict) -> List[Player]:
        """Extract players for a team in batting order
        
        Design decision: Use order of appearance in innings as batting order
        """
        players = []
        seen_players = set()
        
        # Look through innings to find batting order
        for innings in data.get('innings', []):
            if innings.get('team') == team_name:
                for over in innings.get('overs', []):
                    for delivery in over.get('deliveries', []):
                        # Add batsmen in order of appearance
                        for role in ['batter', 'non_striker']:
                            if role in delivery:
                                player_name = delivery[role]
                                if player_name not in seen_players:
                                    player_id = player_registry.get(player_name, player_name.lower().replace(' ', '_'))
                                    player_id = str(player_id)
                                    players.append(Player(player_id, player_name, team_name))
                                    seen_players.add(player_name)
                        
                        # Also add bowlers (they might bat later)
                        if 'bowler' in delivery:
                            player_name = delivery['bowler']
                            if player_name not in seen_players:
                                player_id = player_registry.get(player_name, player_name.lower().replace(' ', '_'))
                                players.append(Player(player_id, player_name, team_name))
                                seen_players.add(player_name)
        
        return players
    
class BettingOddsLoader:
    """Loads and processes betting odds data"""
    
    @staticmethod
    def load_odds(file_path: str) -> Dict[str, Dict]:
        """Load betting odds from JSON file
        
        Returns:
            Dict mapping match_id to odds data
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Create lookup dictionary
            odds_lookup = {}
            for match in data.get('matches', []):
                match_id = match['match_id']
                odds_lookup[match_id] = match
            
            print(f"Loaded odds for {len(odds_lookup)} matches")
            return odds_lookup
            
        except Exception as e:
            print(f"Error loading odds file: {e}")
            return {}
    
    @staticmethod
    def get_implied_probabilities(odds: Dict[str, float], remove_margin: bool = True) -> Dict[str, float]:
        """Convert decimal odds to implied probabilities
        
        Args:
            odds: Dict of team -> decimal odds
            remove_margin: If True, normalize probabilities to sum to 1
            
        Design decision: 
        - Always remove margin by default for fair comparison
        - Handle missing odds gracefully
        """
        if not odds:
            return {}
        
        # Convert to raw implied probabilities
        implied = {}
        for team, decimal_odd in odds.items():
            # Ensure odds are numeric (handle string inputs from JSON)
           try:
               odd_value = float(decimal_odd)
               if odd_value > 0:
                   implied[team] = 1.0 / odd_value
           except (ValueError, TypeError):
           # Skip invalid odds
               continue
        
        if not implied:
            return {}
        
        # Remove bookmaker margin if requested
        if remove_margin:
            total = sum(implied.values())
            if total > 0:
                implied = {team: prob / total for team, prob in implied.items()}
        
        return implied
    
    @staticmethod
    def calculate_margin(odds: Dict[str, float]) -> float:
        """Calculate bookmaker margin (overround)
        
        Returns margin as percentage (e.g., 5.0 for 5%)
        """
        if not odds:
            return 0.0
        
        total_implied = 0.0
        for odd in odds.values():
           try:
               odd_value = float(odd)
               if odd_value > 0:
                   total_implied += 1.0 / odd_value
           except (ValueError, TypeError):
               continue
        return (total_implied - 1.0) * 100
