import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from collections import defaultdict

def segregate_matches_final(source_dir='data/t20s_json', base_output_dir='data'):
    """
    Final split strategy with 1000 matches for val/test, T20 WC in both test and betting
    """
    # First pass: collect all matches with dates to determine optimal cutoffs
    all_matches = []
    
    print("Analyzing matches to determine optimal date cutoffs...")
    for json_file in Path(source_dir).glob('*.json'):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            match_date = datetime.strptime(data['info']['dates'][0], '%Y-%m-%d')
            event_name = data['info'].get('event', {}).get('name', 'Unknown')
            all_matches.append({
                'file': json_file,
                'date': match_date,
                'event': event_name
            })
        except Exception as e:
            print(f"Error reading {json_file.name}: {e}")
    
    # Sort by date
    all_matches.sort(key=lambda x: x['date'])
    
    # Find matches from Oct 2024 onwards for golden set
    golden_start = datetime(2024, 10, 1)
    golden_matches = [m for m in all_matches if m['date'] >= golden_start]
    remaining_matches = [m for m in all_matches if m['date'] < golden_start]
    
    # We need ~1000 for test (working backwards from Sep 30, 2024)
    # and ~1000 for validation before that
    test_end = datetime(2024, 9, 30)
    
    # Estimate: need ~1000 matches for test, ~1000 for val
    # Work backwards to find appropriate cutoff dates
    test_candidates = [m for m in remaining_matches if m['date'] <= test_end]
    
    # Find test start date to get ~1000 matches
    if len(test_candidates) >= 1000:
        test_matches = test_candidates[-1000:]  # Last 1000 matches before Oct 2024
        test_start = test_matches[0]['date']
    else:
        test_matches = test_candidates
        test_start = test_candidates[0]['date'] if test_candidates else datetime(2024, 1, 1)
    
    # Find validation matches (1000 before test period)
    val_candidates = [m for m in remaining_matches if m['date'] < test_start]
    if len(val_candidates) >= 1000:
        val_matches = val_candidates[-1000:]  # Last 1000 matches before test
        val_start = val_matches[0]['date']
    else:
        val_matches = val_candidates
        val_start = val_candidates[0]['date'] if val_candidates else datetime(2023, 1, 1)
    
    # Everything before validation is training
    train_matches = [m for m in all_matches if m['date'] < val_start]
    
    # Find T20 WC 2024 matches specifically
    t20_wc_matches = [
        m for m in all_matches 
        if datetime(2024, 6, 1) <= m['date'] <= datetime(2024, 6, 29)
        and 't20' in m['event'].lower() and 'world cup' in m['event'].lower()
    ]
    
    print(f"\nDetermined date cutoffs:")
    print(f"Training: Up to {val_start.date()}")
    print(f"Validation: {val_start.date()} to {test_start.date()}")
    print(f"Test: {test_start.date()} to {test_end.date()}")
    print(f"Golden: {golden_start.date()} onwards")
    print(f"T20 WC 2024: {len(t20_wc_matches)} matches found")
    
    # Now segregate based on determined cutoffs
    splits = defaultdict(list)
    
    # Create directories
    for split in ['train', 'validation', 'test', 'betting_test', 'golden_test']:
        Path(f"{base_output_dir}/{split}").mkdir(parents=True, exist_ok=True)
    
    # Process each match
    for match_info in all_matches:
        json_file = match_info['file']
        match_date = match_info['date']
        
        # Determine split
        if match_date < val_start:
            split = 'train'
        elif match_date < test_start:
            split = 'validation'
        elif match_date < golden_start:
            split = 'test'
        else:
            split = 'golden_test'
        
        # Copy to appropriate folder
        dest_path = Path(f"{base_output_dir}/{split}/{json_file.name}")
        shutil.copy2(json_file, dest_path)
        splits[split].append(json_file.name)
        
        # ALSO copy T20 WC matches to betting_test folder
        if datetime(2024, 6, 1) <= match_date <= datetime(2024, 6, 29):
            # Check if it's actually a T20 WC match (not just any June match)
            with open(json_file, 'r') as f:
                data = json.load(f)
            event = data['info'].get('event', {}).get('name', '').lower()
            if 't20' in event and 'world cup' in event:
                betting_dest = Path(f"{base_output_dir}/betting_test/{json_file.name}")
                shutil.copy2(json_file, betting_dest)
                splits['betting_test'].append(json_file.name)
    
    # Print final distribution
    print("\n" + "="*50)
    print("FINAL MATCH DISTRIBUTION:")
    print("="*50)
    
    split_stats = {
        'train': len(train_matches),
        'validation': len(val_matches),
        'test': len(test_matches),
        'betting_test': len(t20_wc_matches),
        'golden_test': len(golden_matches)
    }
    
    total = sum(v for k, v in split_stats.items() if k != 'betting_test')
    
    for split, count in split_stats.items():
        if split == 'betting_test':
            print(f"{split:15} : {count:5} matches (copies from test)")
        else:
            pct = (count / total) * 100
            print(f"{split:15} : {count:5} matches ({pct:.1f}%)")
    
    print(f"\nTotal unique matches: {total}")
    
    # Save summary report
    with open(f"{base_output_dir}/split_summary.txt", 'w') as f:
        f.write(f"Dataset Split Summary\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Date Ranges:\n")
        f.write(f"Training: Up to {val_start.date()}\n")
        f.write(f"Validation: {val_start.date()} to {test_start.date()}\n")
        f.write(f"Test: {test_start.date()} to {test_end.date()}\n")
        f.write(f"Betting Test: 2024-06-01 to 2024-06-29 (T20 WC)\n")
        f.write(f"Golden: {golden_start.date()} onwards\n\n")
        f.write(f"Match Counts:\n")
        for split, count in split_stats.items():
            f.write(f"{split}: {count}\n")
    
    print(f"\nSummary saved to {base_output_dir}/split_summary.txt")
    
    return splits

# Execute the segregation
if __name__ == "__main__":
    results = segregate_matches_final()