import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import numpy as np

# Import from the simulation file
from identity_maps import canonicalize_match_id, canonicalize_venue
from match_identity import (
    MATCH_IDENTITY_VERSION,
    build_display_match_id_from_info,
    resolve_match_identity,
)
from sim_v1_2 import MatchState, Player, TeamLineup
from parsing_v2 import classify_match_context

class TestMatchLoader:
    """Loads test matches and creates initial MatchState objects for simulation"""

    __test__ = False
    
    def __init__(self, batter_encoder=None, bowler_encoder=None):
        """
        Args:
            batter_encoder: Optional encoder for player IDs (for future use)
            bowler_encoder: Optional encoder for player IDs (for future use)
        """
        self.batter_encoder = batter_encoder
        self.bowler_encoder = bowler_encoder
        self.identities: Dict[str, Dict[str, str]] = {}
    
    def load_matches(self, folder_path: str) -> List[Tuple[str, MatchState]]:
        """Load all test matches from folder
        
        Returns:
            List of (match_id, initial_match_state) tuples
        """
        self.identities = {}
        matches = []
        json_files = sorted(Path(folder_path).glob('*.json'))
        
        print(f"Loading {len(json_files)} test matches...")
        
        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                match_id, match_state = self._create_match_state(
                    data,
                    cricsheet_id=file_path.stem,
                )
                if match_state:
                    if match_id in self.identities:
                        raise RuntimeError(
                            f"duplicate Cricsheet match ID: {match_id}"
                        )
                    self.identities[match_id] = {
                        "match_id": match_id,
                        "cricsheet_id": file_path.stem,
                        "display_match_id": getattr(
                            match_state,
                            "display_match_id",
                        ),
                        "match_identity_version": MATCH_IDENTITY_VERSION,
                    }
                    matches.append((match_id, match_state))
                    print(f"  Loaded: {match_id}")
                
            except RuntimeError:
                raise
            except Exception as e:
                print(f"  Error loading {file_path.name}: {e}")
        
        print(f"Successfully loaded {len(matches)} matches")
        return matches
    
    def _create_match_state(
        self,
        data: dict,
        cricsheet_id: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[MatchState]]:
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
            venue = canonicalize_venue(info.get('venue'))
            dates = info.get('dates', ['2024-01-01'])
            match_date = datetime.strptime(dates[0], '%Y-%m-%d')
            
            display_match_id = build_display_match_id_from_info(info)
            match_id = str(cricsheet_id or display_match_id)
            
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
            
            # Roster from info.players[team] is the authoritative team sheet.
            info_players = info.get('players', {}) or {}
            team1_roster = list(info_players.get(team1_name, []))
            team2_roster = list(info_players.get(team2_name, []))

            team1_players = self._extract_team_players(data, team1_name, player_registry, team1_roster)
            team2_players = self._extract_team_players(data, team2_name, player_registry, team2_roster)

            # Only pad with dummies if the authoritative roster itself is < 11.
            if len(team1_players) < 11 or len(team2_players) < 11:
                print(f"    Warning: Incomplete team lineups for {match_id}")
                while len(team1_players) < 11:
                    team1_players.append(Player(f"player_{len(team1_players)}", f"Player {len(team1_players)}", team1_name))
                while len(team2_players) < 11:
                    team2_players.append(Player(f"player_{len(team2_players)}", f"Player {len(team2_players)}", team2_name))
            
            # Create lineups. We keep all rostered players so 12-man Impact Sub
            # squads (IPL 2023+, ILT20, SMAT) expose the 12th eligible player
            # to the simulator; the innings-over-at-10-wickets rule ends the
            # innings naturally before all 12 are exhausted.
            team1_lineup = TeamLineup(team1_name, team1_players)
            team2_lineup = TeamLineup(team2_name, team2_players)
            
            # Match context features
            event_info = info.get('event', {})
            event_name = event_info.get('name', '') if isinstance(event_info, dict) else ''
            team_type = info.get('team_type', 'unknown')
            match_ctx = classify_match_context(event_name, team_type, teams)

            # Create match state
            match_state = MatchState(
                team1_lineup=team1_lineup,
                team2_lineup=team2_lineup,
                batting_first=batting_first,
                venue=venue,
                match_date=match_date,
                toss_winner=toss_winner,
                chose_to_bat=1 if toss_decision == 'bat' else 0,
                match_importance=match_ctx['match_importance'],
                is_international=match_ctx['is_international'],
                competition_tier=match_ctx['competition_tier'],
            )
            # MatchState is intentionally a simulation-domain object. These
            # non-feature attributes carry identity through evaluation without
            # making the simulator depend on artifact schemas.
            match_state.cricsheet_id = (
                str(cricsheet_id) if cricsheet_id is not None else None
            )
            match_state.display_match_id = display_match_id
            match_state.match_identity_version = (
                MATCH_IDENTITY_VERSION
                if cricsheet_id is not None
                else "synthetic_fixture_v1"
            )

            return match_id, match_state
            
        except Exception as e:
            print(f"    Error creating match state: {e}")
            return None, None
        
    def _extract_team_players(
        self,
        data: dict,
        team_name: str,
        player_registry: dict,
        roster: Optional[List[str]] = None,
    ) -> List[Player]:
        """Extract players for a team in batting order.

        Order:
          1. Batters in appearance order in the team's own innings.
          2. Bowlers (not already seen) in appearance order in the opposing innings.
          3. Any roster members not yet seen, appended in the roster's own order.
             These are typically tail-enders who neither batted nor bowled.

        `roster` is the authoritative 11-man team sheet from
        `data.info.players[team_name]`. When provided, step 3 recovers players
        missed by a deliveries-only walk (chase-won-quickly tail-enders,
        bat-all-20-overs lower orders, rain-abandoned squads).
        """
        players: List[Player] = []
        seen_names: set = set()

        def _add(name: str) -> None:
            if name in seen_names:
                return
            player_id = player_registry.get(name, name.lower().replace(' ', '_'))
            players.append(Player(str(player_id), name, team_name))
            seen_names.add(name)

        for innings in data.get('innings', []):
            if innings.get('team') == team_name:
                for over in innings.get('overs', []):
                    for delivery in over.get('deliveries', []):
                        for role in ('batter', 'non_striker'):
                            if role in delivery:
                                _add(delivery[role])

        for innings in data.get('innings', []):
            if innings.get('team') != team_name:
                for over in innings.get('overs', []):
                    for delivery in over.get('deliveries', []):
                        if 'bowler' in delivery:
                            _add(delivery['bowler'])

        if roster:
            for name in roster:
                _add(name)

        return players
    
class BettingOddsLoader:
    """Loads and processes betting odds data"""

    @staticmethod
    def load_odds(file_path: str, min_volume: Optional[float] = None) -> Dict[str, Dict]:
        """Load betting odds from JSON file.

        Args:
            file_path: Path to the odds JSON.
            min_volume: If set, drop entries whose `polymarket_volume_usd` is
                below the threshold. Entries without that field are also
                dropped when `min_volume` is set (only liquidity-tagged odds
                files such as betting_odds_polymarket.json carry the field).
                When `min_volume is None`, no filtering — preserves the
                non-polymarket odds files (betting_odds_v3.json etc).

        Returns:
            Dict mapping match_id to odds data.
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            odds_lookup = {}
            kept = 0
            dropped = 0
            for match in data.get('matches', []):
                if min_volume is not None:
                    vol = match.get('polymarket_volume_usd', 0) or 0
                    if vol < min_volume:
                        dropped += 1
                        continue
                identity = resolve_match_identity(match)
                match_id = identity.primary_id
                if identity.cricsheet_id is None:
                    # Frozen rows use the synthetic display ID, whose venue
                    # suffix must still honor the active exact alias map.
                    match_id = canonicalize_match_id(match_id)
                if match_id in odds_lookup:
                    raise ValueError(
                        f"duplicate odds primary match ID: {match_id}"
                    )
                odds_lookup[match_id] = match
                kept += 1

            if min_volume is not None:
                print(f"Loaded odds for {kept} matches "
                      f"(dropped {dropped} below ${min_volume:,.0f} volume)")
            else:
                print(f"Loaded odds for {kept} matches")
            return odds_lookup

        except Exception as e:
            raise RuntimeError(
                f"failed to load odds file {file_path}: {e}"
            ) from e
    
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
