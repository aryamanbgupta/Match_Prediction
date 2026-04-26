# Initial working parsing file, stores data in parquet
import json
import numpy as np
from pathlib import Path
import pandas as pd
from collections import Counter
from datetime import datetime

def parse_match_data(json_data):
    data = json.loads(json_data)
    player_registry = data['info']['registry']['people']

    all_balls = []  # Flatten to list of individual balls
    
    for inning_idx, inning in enumerate(data['innings'], 1):
        score = 0
        wickets = 0
        balls = 0
        ball_in_innings = 0  # Position within this innings
        
        for over in inning['overs']:
            for delivery in over['deliveries']:
                batter = delivery['batter']
                non_striker = delivery['non_striker']
                bowler = delivery['bowler']
                runs = delivery['runs']['total']
                
                batter_id = player_registry[batter]
                non_striker_id = player_registry[non_striker]
                bowler_id = player_registry[bowler]
                
                # Check for extras and wickets
                is_extra = 'extras' in delivery
                extra_type = delivery.get('extras', {}).keys()
                is_wide_or_noball = 'wides' in extra_type or 'noballs' in extra_type
                is_wicket = 'wickets' in delivery

                if is_wicket:
                    ball_outcome = -1
                else:
                    ball_outcome = runs
                
                # Create ball record
                ball_record = {
                    'innings_id': f"{inning_idx}_{hash(json_data) % 100000}",  # Unique innings identifier
                    'ball_position': ball_in_innings,  # Position within innings for sequencing
                    'inning_idx': inning_idx,
                    'score': score,
                    'wickets': wickets,
                    'balls_bowled': balls,
                    'batter_id': batter_id,
                    'non_striker_id': non_striker_id,
                    'bowler_id': bowler_id,
                    'ball_outcome': ball_outcome
                }
                
                all_balls.append(ball_record)
                
                # Update state
                if is_wicket:
                    wickets += 1
                else:
                    score += runs
                
                if not is_wide_or_noball:
                    balls += 1
                
                ball_in_innings += 1

    return all_balls, set(player_registry.values())

def process_folder(folder_path):
    all_balls = []
    all_player_ids = set()
    processed_files = 0
    run_outcomes = []

    # Get all JSON files and sort them by date
    json_files = sorted(
        Path(folder_path).glob('*.json'),
        key=lambda x: json.loads(x.read_text())['info']['dates'][0]
    )

    for file_path in json_files:
        try:
            with open(file_path, 'r') as file:
                json_data = file.read()
            
            match_balls, match_player_ids = parse_match_data(json_data)
            all_balls.extend(match_balls)
            all_player_ids.update(match_player_ids)
            processed_files += 1
            
            # Collect run outcomes
            run_outcomes.extend([ball['ball_outcome'] for ball in match_balls])
            
            print(f"Processed {file_path.name}: {len(match_balls)} balls, {len(match_player_ids)} players")
        except json.JSONDecodeError:
            print(f"Error decoding JSON in file: {file_path.name}")
        except Exception as e:
            print(f"Error processing file {file_path.name}: {str(e)}")

    return all_balls, processed_files, all_player_ids, run_outcomes

# Process data
all_balls, total_files, unique_player_ids, run_outcomes = process_folder('data/t20s_json')

print(f"\nTotal files processed: {total_files}")
print(f"Total balls collected: {len(all_balls)}")
print(f"Total unique players across all matches: {len(unique_player_ids)}")

# Count and calculate percentages for run outcomes
total_balls = len(run_outcomes)
outcome_counts = Counter(run_outcomes)
print(f"\nTotal number of balls: {total_balls}")
print("\nDistribution of run outcomes:")
for outcome, count in sorted(outcome_counts.items()):
    percentage = (count / total_balls) * 100
    print(f"  {outcome}: {count} ({percentage:.2f}%)")

# Convert to DataFrame
df = pd.DataFrame(all_balls)

print(f"\nDataFrame shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Data types:\n{df.dtypes}")

# Save as Parquet - done!
df.to_parquet('cricket_data.parquet', index=False)
print(f"\nSaved to cricket_data.parquet")

# # Quick verification
# print(f"File size: {Path('cricket_data.parquet').stat().st_size / 1024 / 1024:.2f} MB")

# # Example: Loading for model training
# print("\n--- Loading for model training ---")
# df_loaded = pd.read_parquet('cricket_data.parquet')

# # Convert to sequences for LSTM/Transformer
# def df_to_sequences(df, max_length=None):
#     """Convert flat DataFrame back to sequences grouped by innings"""
#     sequences = []
#     for innings_id, group in df.groupby('innings_id'):
#         # Sort by ball position to maintain order
#         seq_data = group.sort_values('ball_position')
        
#         # Extract features (excluding sequence identifiers)
#         feature_cols = ['inning_idx', 'score', 'wickets', 'balls_bowled', 
#                        'batter_id', 'non_striker_id', 'bowler_id']
#         sequence = seq_data[feature_cols].values
#         target = seq_data['ball_outcome'].values
        
#         sequences.append({
#             'features': sequence,
#             'targets': target,
#             'innings_id': innings_id
#         })
    
#     return sequences

# # Example usage
# sequences = df_to_sequences(df_loaded)
# print(f"Created {len(sequences)} sequences")
# print(f"First sequence shape: {sequences[0]['features'].shape}")