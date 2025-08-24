import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class BallByBallFeatureExtractor:
    def __init__(self, data_dir='data', output_dir='features/v1'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load ground averages
        try:
            with open('data/ground_average_scores.json', 'r') as f:
                self.ground_averages = json.load(f)
        except:
            self.ground_averages = {}
        
        self.all_features = []
        
    def extract_features_from_match(self, json_path, match_date):
        """Extract ball-by-ball features from a single match"""
        
        with open(json_path, 'r') as f:
            match_data = json.load(f)
        
        match_id = json_path.stem
        info = match_data['info']
        
        # Get player registry (more reliable than names)
        player_registry = info.get('registry', {}).get('people', {})
        
        # Match metadata
        teams = info['teams']
        venue = info.get('venue', 'Unknown')
        default_target = self.ground_averages.get(venue, 160)
        toss_winner = info.get('toss', {}).get('winner', '')
        
        first_innings_score = 0
        
        # Process each innings
        for innings_num, innings in enumerate(match_data['innings'], 1):
            batting_team = innings['team']
            bowling_team = teams[1] if teams[0] == batting_team else teams[0]
            
            # Initialize innings state
            runs_scored = 0
            wickets_fallen = 0
            balls_bowled = 0
            
            # Calculate target correctly
            if innings_num == 1:
                target = default_target
            else:
                target = first_innings_score + 1
            
            # Process each over
            for over_data in innings['overs']:
                over_num = over_data['over']
                
                # Process each delivery
                for delivery in over_data['deliveries']:
                    # Extract player info using registry
                    batter = delivery.get('batter', 'Unknown')
                    bowler = delivery.get('bowler', 'Unknown')
                    non_striker = delivery.get('non_striker', 'Unknown')
                    
                    # Get actual player IDs from registry
                    batter_id = player_registry.get(batter, f"unknown_{hash(batter) % 10000}")
                    bowler_id = player_registry.get(bowler, f"unknown_{hash(bowler) % 10000}")
                    non_striker_id = player_registry.get(non_striker, f"unknown_{hash(non_striker) % 10000}")
                    
                    # Extract run info
                    runs = delivery.get('runs', {})
                    batter_runs = runs.get('batter', 0)
                    extras = runs.get('extras', 0)
                    total_runs = runs.get('total', 0)
                    
                    # Check for wicket
                    wickets = delivery.get('wickets', [])
                    is_wicket = len(wickets) > 0
                    
                    # Check for extras
                    extras_dict = delivery.get('extras', {})
                    is_wide = 'wides' in extras_dict
                    is_noball = 'noballs' in extras_dict
                    is_legal_ball = not (is_wide or is_noball)
                    
                    # Calculate rates
                    current_run_rate = (runs_scored / balls_bowled * 6) if balls_bowled > 0 else 0
                    
                    if innings_num == 2:
                        runs_needed = max(0, target - runs_scored)
                        balls_remaining = (20 * 6) - balls_bowled
                        required_run_rate = (runs_needed / balls_remaining * 6) if balls_remaining > 0 else 0
                    else:
                        runs_needed = 0
                        required_run_rate = 0
                    
                    # Create feature dict for this ball
                    features = {
                        # Match context
                        'match_id': match_id,
                        'match_date': match_date,
                        'venue': venue,
                        'batting_team': batting_team,
                        'bowling_team': bowling_team,
                        'innings': innings_num,
                        'toss_winner_batting': batting_team == toss_winner,
                        'batting_first': innings_num == 1,
                        
                        # Ball context  
                        'over': over_num,
                        'ball_in_over': len([d for d in over_data['deliveries'][:over_data['deliveries'].index(delivery)+1] 
                                           if not ('wides' in d.get('extras', {}) or 'noballs' in d.get('extras', {}))]),
                        'balls_bowled': balls_bowled,
                        
                        # Current state (BEFORE this ball)
                        'runs_scored': runs_scored,
                        'wickets_fallen': wickets_fallen,
                        'runs_needed': runs_needed,
                        'current_run_rate': round(current_run_rate, 2),
                        'required_run_rate': round(required_run_rate, 2),
                        'target': target,
                        
                        # Players
                        'batter_id': batter_id,
                        'bowler_id': bowler_id,
                        'non_striker_id': non_striker_id,
                        
                        # Phase of play
                        'is_powerplay': over_num < 6,
                        'is_middle_overs': 6 <= over_num < 16,
                        'is_death_overs': over_num >= 16,
                        
                        # Ball outcome (targets)
                        'runs_scored_ball': batter_runs,
                        'total_runs_ball': total_runs,
                        'is_wicket': is_wicket,
                        'is_wide': is_wide,
                        'is_noball': is_noball,
                        'is_dot': total_runs == 0 and not is_wicket and is_legal_ball,
                        'is_boundary': batter_runs >= 4,
                    }
                    
                    self.all_features.append(features)
                    
                    # Update state AFTER this ball
                    runs_scored += total_runs
                    if is_wicket:
                        wickets_fallen += 1
                    
                    # Increment balls only for legal deliveries
                    if is_legal_ball:
                        balls_bowled += 1
                    
                    # Check for innings end
                    if wickets_fallen >= 10 or balls_bowled >= 120:  # 20 overs = 120 balls
                        break
                
                if wickets_fallen >= 10 or balls_bowled >= 120:
                    break
            
            # Store first innings score for target calculation
            if innings_num == 1:
                first_innings_score = runs_scored
    
    def process_split(self, split_name):
        """Process all matches in a split folder"""
        split_dir = self.data_dir / split_name
        
        if not split_dir.exists():
            print(f"Skipping {split_name} - directory not found")
            return []
        
        # Get all JSON files with dates
        matches = []
        for json_file in split_dir.glob('*.json'):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                match_date = datetime.strptime(data['info']['dates'][0], '%Y-%m-%d')
                matches.append((json_file, match_date))
            except Exception as e:
                print(f"Error reading {json_file}: {e}")
        
        # Sort by date
        matches.sort(key=lambda x: x[1])
        
        print(f"\nProcessing {split_name}: {len(matches)} matches")
        
        # Process each match
        split_features = []
        for i, (json_file, match_date) in enumerate(matches):
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(matches)} matches")
            
            try:
                features_before = len(self.all_features)
                self.extract_features_from_match(json_file, match_date)
                features_added = len(self.all_features) - features_before
                
                # Track which features belong to this split
                for _ in range(features_added):
                    split_features.append(split_name)
            except Exception as e:
                print(f"Error processing {json_file.name}: {e}")
        
        return split_features
    
    def run(self):
        """Main processing pipeline"""
        print("="*60)
        print("BALL-BY-BALL FEATURE EXTRACTION")
        print("="*60)
        
        # Process each split
        splits = ['train', 'validation', 'test', 'golden_test']
        split_labels = []
        
        for split in splits:
            labels = self.process_split(split)
            split_labels.extend(labels)
        
        # Convert to DataFrame
        print(f"\nTotal features extracted: {len(self.all_features)}")
        df = pd.DataFrame(self.all_features)
        
        # Add split column
        df['split'] = split_labels
        
        # Sort by date and reset index
        df = df.sort_values(['match_date', 'match_id', 'innings', 'over', 'balls_bowled'])
        df = df.reset_index(drop=True)
        
        # Save separate parquet for each split
        for split in splits:
            split_df = df[df['split'] == split].copy()
            split_df = split_df.drop(columns=['split'])
            
            if len(split_df) > 0:
                output_path = self.output_dir / f'{split}.parquet'
                split_df.to_parquet(output_path, compression='snappy', index=False)
                print(f"\nSaved {split}:")
                print(f"  Path: {output_path}")
                print(f"  Rows: {len(split_df):,}")
                print(f"  Matches: {split_df['match_id'].nunique()}")
        
        print("\n" + "="*60)
        print("FEATURE EXTRACTION COMPLETE")
        print("="*60)
        
        return df

# Run the extraction
if __name__ == "__main__":
    extractor = BallByBallFeatureExtractor()
    df = extractor.run()