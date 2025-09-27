# parse for xgboost model.py
import json
import numpy as np
from pathlib import Path
import pandas as pd
from collections import defaultdict, deque
from datetime import datetime

class PlayerStatsTracker:
    """
    DESIGN DECISION: Separate class for player stats to maintain state across matches.
    REASONING: Encapsulation - keeps complex state management isolated from parsing logic.
    This makes it easy to add new stats without cluttering the main parsing code.
    """
    def __init__(self):
        # Career stats - accumulate over time
        self.batting_stats = defaultdict(lambda: {'runs': 0, 'balls': 0, 'dismissals': 0})
        self.bowling_stats = defaultdict(lambda: {'runs_given': 0, 'balls_bowled': 0, 'wickets': 0})
        
        # Head-to-head records
        # DESIGN DECISION: Use tuple (batter, bowler) as key for h2h
        # REASONING: Direct lookup, memory efficient, naturally handles bidirectional relationships
        self.h2h_stats = defaultdict(lambda: {'runs': 0, 'balls': 0, 'dismissals': 0})
        
        # Recent form tracking (last 5 matches)
        # DESIGN DECISION: Track last 5 match performances
        # REASONING: Recent form often more predictive than career average
        self.recent_batting = defaultdict(lambda: deque(maxlen=5))
        self.recent_bowling = defaultdict(lambda: deque(maxlen=5))
    
    def get_batting_features(self, batter_id):
        """Get batting stats features at current point in time"""
        stats = self.batting_stats[batter_id]
        # DESIGN DECISION: Return 0 for unknown players rather than None
        # REASONING: Models handle 0s better than missing values, represents "no history"
        if stats['balls'] == 0:
            return {'batsman_avg': 0, 'batsman_sr': 0}
        
        avg = stats['runs'] / max(stats['dismissals'], 1)  # Avoid division by zero
        sr = (stats['runs'] / stats['balls']) * 100 if stats['balls'] > 0 else 0
        return {'batsman_avg': avg, 'batsman_sr': sr}
    
    def get_bowling_features(self, bowler_id):
        """Get bowling stats features at current point in time"""
        stats = self.bowling_stats[bowler_id]
        if stats['balls_bowled'] == 0:
            return {'bowler_avg': 0, 'bowler_econ': 0}
        
        avg = stats['runs_given'] / max(stats['wickets'], 1)
        econ = (stats['runs_given'] / stats['balls_bowled']) * 6 if stats['balls_bowled'] > 0 else 0
        return {'bowler_avg': avg, 'bowler_econ': econ}
    
    def get_h2h_features(self, batter_id, bowler_id):
        """Get head-to-head matchup features"""
        stats = self.h2h_stats[(batter_id, bowler_id)]
        if stats['balls'] == 0:
            return {'h2h_avg': 0, 'h2h_sr': 0}
        
        avg = stats['runs'] / max(stats['dismissals'], 1)
        sr = (stats['runs'] / stats['balls']) * 100 if stats['balls'] > 0 else 0
        return {'h2h_avg': avg, 'h2h_sr': sr}
    
    def update_stats(self, batter_id, bowler_id, runs, is_wicket):
        """Update all statistics after a ball"""
        # Update batting stats
        self.batting_stats[batter_id]['runs'] += runs
        self.batting_stats[batter_id]['balls'] += 1
        if is_wicket:
            self.batting_stats[batter_id]['dismissals'] += 1
        
        # Update bowling stats
        self.bowling_stats[bowler_id]['runs_given'] += runs
        self.bowling_stats[bowler_id]['balls_bowled'] += 1
        if is_wicket:
            self.bowling_stats[bowler_id]['wickets'] += 1
        
        # Update h2h
        self.h2h_stats[(batter_id, bowler_id)]['runs'] += runs
        self.h2h_stats[(batter_id, bowler_id)]['balls'] += 1
        if is_wicket:
            self.h2h_stats[(batter_id, bowler_id)]['dismissals'] += 1


class InningsFeatureCalculator:
    """
    DESIGN DECISION: Separate class for innings-level features that reset each innings.
    REASONING: Clear separation between match-level and innings-level state.
    Makes it obvious which features reset and which persist.
    """
    def __init__(self):
        # Recent ball tracking for momentum features
        self.last_5_balls = deque(maxlen=5)
        self.last_10_balls = deque(maxlen=10)
        self.last_30_balls = deque(maxlen=30)
        self.balls_since_boundary = 0
    
    def update_ball_history(self, runs, is_boundary):
        """Update rolling windows after each ball"""
        self.last_5_balls.append(runs)
        self.last_10_balls.append(runs)
        self.last_30_balls.append(runs)
        
        if is_boundary:
            self.balls_since_boundary = 0
        else:
            self.balls_since_boundary += 1
    
    def get_momentum_features(self):
        """Calculate all momentum-based features"""
        # DESIGN DECISION: Return 0 for insufficient history rather than None
        # REASONING: Allows model to learn from early-ball situations too
        return {
            'last_5_balls_runs': sum(self.last_5_balls),
            'last_10_balls_runs': sum(self.last_10_balls),
            'last_30_balls_runs': sum(self.last_30_balls),
            'balls_since_boundary': self.balls_since_boundary,
            # Dot ball pressure
            'last_10_dots': sum(1 for r in self.last_10_balls if r == 0),
        }


def normalize_ball_outcome(runs, is_wicket):
    """
    DESIGN DECISION: Normalize rare run outcomes to reduce class imbalance.
    REASONING: 3,5,7+ runs are very rare and hurt model performance.
    """
    if is_wicket:
        return -1  # Keep wickets as -1
    
    # Normalize rare run values
    if runs == 3:
        return 2
    elif runs == 5:
        return 4
    elif runs >= 7:
        return 6
    else:
        return runs  # 0,1,2,4,6 stay the same
    

def extract_raw_state(delivery, player_registry, score, wickets, balls):
    """
    DESIGN DECISION: Pure function that extracts raw state from delivery.
    REASONING: Separation of concerns - parsing vs feature engineering.
    This function only extracts what's directly in the data.
    """
    batter = delivery['batter']
    non_striker = delivery['non_striker']
    bowler = delivery['bowler']
    runs = delivery['runs']['total']
    
    # Player IDs from registry
    batter_id = player_registry[batter]
    non_striker_id = player_registry[non_striker]
    bowler_id = player_registry[bowler]
    
    # Check for events
    is_wicket = 'wickets' in delivery
    extra_type = list(delivery.get('extras', {}).keys()) if 'extras' in delivery else []
    is_wide = 'wides' in extra_type
    is_noball = 'noballs' in extra_type
    
    return {
        'batter_id': batter_id,
        'non_striker_id': non_striker_id,
        'bowler_id': bowler_id,
        'runs': runs,
        'is_wicket': is_wicket,
        'is_wide': is_wide,
        'is_noball': is_noball,
        'score': score,
        'wickets': wickets,
        'balls_bowled': balls,
    }


def calculate_basic_features(state):
    """
    DESIGN DECISION: Separate function for stateless features.
    REASONING: These can be calculated independently without any history.
    Easy to test and reason about.
    """
    features = {}
    
    # Run rate and required rate
    overs = state['balls_bowled'] / 6
    features['run_rate'] = state['score'] / max(overs, 0.1)  # Avoid division by zero
    
    # Resource percentages
    features['wickets_ratio'] = state['wickets'] / 10
    features['balls_ratio'] = state['balls_bowled'] / 120  # T20 format
    features['wickets_in_hand'] = 10 - state['wickets']
    
    # Match phase indicators
    # DESIGN DECISION: Use simple binary flags for phases
    # REASONING: Easier for tree-based models than continuous over count
    features['is_powerplay'] = state['balls_bowled'] < 36
    features['is_middle_overs'] = 36 <= state['balls_bowled'] < 96
    features['is_death_overs'] = state['balls_bowled'] >= 96
    
    # Current over progress
    features['balls_in_over'] = state['balls_bowled'] % 6
    
    return features


def calculate_pressure_features(state, innings_calc):
    """
    DESIGN DECISION: Separate pressure indicators as they're conceptually related.
    REASONING: Groups related features, makes it easy to experiment with adding/removing
    the entire pressure feature set.
    """
    features = {}
    
    momentum = innings_calc.get_momentum_features()
    
    # Dot ball percentage in recent balls
    if len(innings_calc.last_30_balls) > 0:
        features['dot_percentage_recent'] = momentum['last_10_dots'] / min(len(innings_calc.last_10_balls), 10)
    else:
        features['dot_percentage_recent'] = 0
    
    # Boundary percentage
    boundaries_recent = sum(1 for r in innings_calc.last_30_balls if r >= 4)
    if len(innings_calc.last_30_balls) > 0:
        features['boundary_percentage_recent'] = boundaries_recent / len(innings_calc.last_30_balls)
    else:
        features['boundary_percentage_recent'] = 0
    
    return features


def parse_match_data_v2(json_data, player_stats_tracker):
    """
    DESIGN DECISION: Pass tracker as parameter rather than global.
    REASONING: Makes dependencies explicit, easier to test, allows multiple trackers
    for different experiments (e.g., one with h2h, one without).
    """
    data = json.loads(json_data)
    player_registry = data['info']['registry']['people']
    
    # DESIGN DECISION: Store match metadata for potential venue/team features
    # REASONING: Might want venue-specific features later
    match_info = {
        'venue': data['info'].get('venue', 'unknown'),
        'date': data['info']['dates'][0] if 'dates' in data['info'] else None,
        'teams': data['info'].get('teams', []),
    }
    
    all_balls = []
    
    for inning_idx, inning in enumerate(data['innings'], 1):
        score = 0
        wickets = 0
        balls = 0
        
        # DESIGN DECISION: Reset innings calculator per innings
        # REASONING: Momentum features should not carry over between innings
        innings_calc = InningsFeatureCalculator()
        
        for over_idx, over in enumerate(inning['overs']):
            for delivery in over['deliveries']:
                # Extract raw state
                state = extract_raw_state(delivery, player_registry, score, wickets, balls)
                
                # Get player statistics BEFORE this ball
                # DESIGN DECISION: Features reflect state BEFORE the ball
                # REASONING: This is what the model would know when predicting
                batting_features = player_stats_tracker.get_batting_features(state['batter_id'])
                bowling_features = player_stats_tracker.get_bowling_features(state['bowler_id'])
                h2h_features = player_stats_tracker.get_h2h_features(state['batter_id'], state['bowler_id'])
                
                # Calculate all features
                basic_features = calculate_basic_features(state)
                momentum_features = innings_calc.get_momentum_features()
                pressure_features = calculate_pressure_features(state, innings_calc)
                
                # Combine all features
                # DESIGN DECISION: Flatten all features into single dict
                # REASONING: Simpler for DataFrame creation and model training
                ball_record = {
                    'innings_id': f"{inning_idx}_{hash(json_data) % 100000}",
                    'inning_idx': inning_idx,
                    'over_idx': over_idx,
                    'ball_idx': balls,
                    # Raw state
                    **state,
                    # Computed features
                    **basic_features,
                    **batting_features,
                    **bowling_features,
                    **h2h_features,
                    **momentum_features,
                    **pressure_features,
                    # Target
                    'ball_outcome': normalize_ball_outcome(state['runs'], state['is_wicket'])
                }
                
                all_balls.append(ball_record)
                
                # Update states AFTER recording the ball
                # DESIGN DECISION: Update after recording
                # REASONING: Features should reflect pre-ball state
                player_stats_tracker.update_stats(
                    state['batter_id'], 
                    state['bowler_id'],
                    state['runs'],
                    state['is_wicket']
                )
                
                innings_calc.update_ball_history(
                    state['runs'],
                    is_boundary=(state['runs'] >= 4)
                )
                
                # Update match state
                if state['is_wicket']:
                    wickets += 1
                score += state['runs']
                if not (state['is_wide'] or state['is_noball']):
                    balls += 1
    
    return all_balls

'''
def process_folder_v2(folder_path):
    """
    DESIGN DECISION: Single pass through chronologically sorted matches.
    REASONING: Ensures player stats are accumulated in correct temporal order,
    preventing data leakage.
    """
    # Initialize tracker that will accumulate across all matches
    player_stats_tracker = PlayerStatsTracker()
    
    all_balls = []
    processed_files = 0
    
    # CRITICAL: Sort by date to ensure correct temporal ordering
    json_files = sorted(
        Path(folder_path).glob('*.json'),
        key=lambda x: json.loads(x.read_text())['info']['dates'][0]
    )
    
    print(f"Processing {len(json_files)} files in chronological order...")
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as file:
                json_data = file.read()
            
            match_balls = parse_match_data_v2(json_data, player_stats_tracker)
            all_balls.extend(match_balls)
            processed_files += 1
            
            if processed_files % 100 == 0:
                print(f"Processed {processed_files} matches, {len(all_balls)} balls")
                
        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")
    
    return all_balls, processed_files


# Main execution
if __name__ == "__main__":
    all_balls, total_files = process_folder_v2('data/t20s_json')
    
    print(f"\nTotal files processed: {total_files}")
    print(f"Total balls collected: {len(all_balls)}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_balls)
    
    # DESIGN DECISION: Save with version suffix
    # REASONING: Can compare different feature sets without losing previous work
    output_file = 'cricket_data_v2_with_features.parquet'
    df.to_parquet(output_file, index=False)
    
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Number of features: {len(df.columns)}")
    print(f"Saved to {output_file}")
    
    # Display feature categories for verification
    print("\nFeature categories:")
    print("- Basic features:", [c for c in df.columns if c.startswith('is_') or '_ratio' in c])
    print("- Player stats:", [c for c in df.columns if 'batsman_' in c or 'bowler_' in c])
    print("- H2H features:", [c for c in df.columns if 'h2h_' in c])
    print("- Momentum features:", [c for c in df.columns if 'last_' in c or 'since_boundary' in c])
'''

def process_folder_v2_with_splits(folder_path):
    """
    Process matches chronologically and save separate parquet files for each temporal split
    """
    # Hardcoded date ranges from split summary
    train_end = datetime(2022, 12, 29)
    val_start = datetime(2022, 12, 29)
    val_end = datetime(2024, 1, 11)
    test_start = datetime(2024, 1, 11)
    test_end = datetime(2024, 9, 30)
    betting_start = datetime(2024, 6, 1)
    betting_end = datetime(2024, 6, 29)
    golden_start = datetime(2024, 10, 1)
    
    # Initialize tracker that will accumulate across all matches
    player_stats_tracker = PlayerStatsTracker()
    
    # Data containers for each split
    split_data = {
        'train': [],
        'validation': [],
        'test': [],
        'betting_test': [],
        'golden_test': []
    }
    
    processed_files = 0
    
    # Sort files chronologically
    json_files = sorted(
        Path(folder_path).glob('*.json'),
        key=lambda x: json.loads(x.read_text())['info']['dates'][0]
    )
    
    print(f"Processing {len(json_files)} files in chronological order...")
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as file:
                json_data = file.read()
            
            # Get match date
            data = json.loads(json_data)
            match_date = datetime.strptime(data['info']['dates'][0], '%Y-%m-%d')
            
            # Determine which split this match belongs to
            if match_date < train_end:
                current_split = 'train'
            elif match_date < test_start:
                current_split = 'validation'
            elif match_date < golden_start:
                current_split = 'test'
            else:
                current_split = 'golden_test'
            
            # Check if it's also a betting test match (T20 WC)
            is_betting_match = (
                betting_start <= match_date <= betting_end
                and 't20' in data['info'].get('event', {}).get('name', '').lower()
                and 'world cup' in data['info'].get('event', {}).get('name', '').lower()
            )
            
            # Process the match
            match_balls = parse_match_data_v2(json_data, player_stats_tracker)
            
            # Add to appropriate split(s)
            split_data[current_split].extend(match_balls)
            if is_betting_match:
                split_data['betting_test'].extend(match_balls)
            
            processed_files += 1
            
            if processed_files % 100 == 0:
                total_balls = sum(len(balls) for balls in split_data.values())
                print(f"Processed {processed_files} matches, {total_balls} total balls")
                
        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")
    
    # Save separate parquet files for each split
    for split_name, balls in split_data.items():
        if balls:  # Only save if there's data
            df = pd.DataFrame(balls)
            output_file = f'data/xgb_data/cricket_data_v2_{split_name}.parquet'
            df.to_parquet(output_file, index=False)
            print(f"Saved {split_name}: {len(balls)} balls to {output_file}")
    
    return split_data, processed_files


# Update the main execution
if __name__ == "__main__":
    split_data, total_files = process_folder_v2_with_splits('data/t20s_json')
    
    print(f"\nTotal files processed: {total_files}")
    
    # Summary statistics
    for split_name, balls in split_data.items():
        if balls:
            df = pd.DataFrame(balls)
            print(f"\n{split_name.upper()}:")
            print(f"  Balls: {len(balls)}")
            print(f"  Features: {len(df.columns)}")