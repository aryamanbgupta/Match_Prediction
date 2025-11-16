import json
import csv
from pathlib import Path
from tqdm import tqdm

def extract_players_from_json_files(json_dir='data/t20s_json', output_file='data/player_registry_ids.csv'):
    """
    Extract all unique players from T20 match JSON files.

    Scans all JSON files in json_dir, extracts player names and registry IDs,
    and outputs unique players to a CSV file.

    Args:
        json_dir: Directory containing match JSON files
        output_file: Output CSV file path
    """
    json_path = Path(json_dir)
    json_files = list(json_path.glob('*.json'))

    print(f"Found {len(json_files)} JSON files")

    # Set to store unique (registry_id, name) pairs
    unique_players = set()

    # Process each match file
    for json_file in tqdm(json_files, desc="Extracting players"):
        try:
            with open(json_file, 'r') as f:
                match_data = json.load(f)

            # Extract registry.people section
            registry = match_data.get('info', {}).get('registry', {}).get('people', {})

            # Add each player to the set
            for name, registry_id in registry.items():
                unique_players.add((registry_id, name))

        except Exception as e:
            print(f"\nError processing {json_file}: {e}")
            continue

    print(f"\nFound {len(unique_players)} unique players")

    # Sort by registry_id for consistent output
    sorted_players = sorted(unique_players, key=lambda x: x[0])

    # Write to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['registry_id', 'name'])
        writer.writerows(sorted_players)

    print(f"Saved to {output_file}")

if __name__ == '__main__':
    extract_players_from_json_files()
