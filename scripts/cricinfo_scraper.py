import csv
import requests
from bs4 import BeautifulSoup
from rpy2 import robjects
from rpy2.robjects.packages import importr
from collections import Counter
import os
import json
import time
import shutil

# def clear_cache():
#     if os.path.exists(cache_dir):
#         shutil.rmtree(cache_dir)
#         print(f"Cache directory '{cache_dir}' has been cleared.")
#     else:
#         print(f"Cache directory '{cache_dir}' does not exist.")
    
#     # Recreate the empty cache directory
#     os.makedirs(cache_dir, exist_ok=True)

# Import the R package that contains find_player_id
r_package = importr('cricketdata')

# Create a cache directory if it doesn't exist
# cache_dir = 'player_cache'
# os.makedirs(cache_dir, exist_ok=True)

def get_player_ids(player_name):
    result = r_package.find_player_id(player_name)
    player_ids = result.rx2('ID')
    return [str(int(id)) for id in player_ids]

def get_player_info(player_name):
    # cache_file = os.path.join(cache_dir, f"{player_name.replace(' ', '_')}.json")
    
    # Check if we have cached data for this player
    # if os.path.exists(cache_file):
    #     with open(cache_file, 'r') as f:
    #         return json.load(f)
    
    player_ids = get_player_ids(player_name)
    all_info = []
    
    for player_id in player_ids:
        url = f"https://www.espncricinfo.com/cricketers/{player_name.lower().replace(' ', '-')}-{player_id}"
        
        try:
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                info = {
                    "player_id": player_id,
                    "bowling_style": "Not found",
                    "batting_style": "Not found",
                    "playing_role": "Not found",
                    "full_name": "Not found",
                    "date_of_birth": "Not found",
                    "teams": "Not found"
                }
                
                # Extract bowling style
                bowling_style_div = soup.find('p', string='Bowling Style')
                if bowling_style_div:
                    info["bowling_style"] = bowling_style_div.find_next_sibling('span').text.strip()
                
                # Extract batting style
                batting_style_div = soup.find('p', string='Batting Style')
                if batting_style_div:
                    info["batting_style"] = batting_style_div.find_next_sibling('span').text.strip()
                
                # Extract playing role
                playing_role_div = soup.find('p', string='Playing Role')
                if playing_role_div:
                    info["playing_role"] = playing_role_div.find_next_sibling('span').text.strip()
                
                # Extract full name
                full_name_div = soup.find('p', string='Full Name')
                if full_name_div:
                    full_name_span = full_name_div.find_next_sibling('span', class_='ds-text-title-s')
                    if full_name_span:
                        info["full_name"] = full_name_span.text.strip()
                
                # Extract playing role
                dob_div = soup.find('p', string='Born')
                if dob_div:
                    info["date_of_birth"] = dob_div.find_next_sibling('span').text.strip()

                # Extract teams
                teams_div = soup.find('p', string='TEAMS')
                if teams_div:
                    teams_grid = teams_div.find_next_sibling('div', class_='ds-grid')
                    if teams_grid:
                        team_links = teams_grid.find_all('a')
                        teams = [link['href'].split('/')[-1] for link in team_links]
                        info["teams"] = '; '.join(teams)
                
                all_info.append(info)
            else:
                print(f"Failed to retrieve the page for {player_name} (ID: {player_id}). Status code: {response.status_code}")
        
        except requests.RequestException as e:
            print(f"Request failed for {player_name} (ID: {player_id}): {str(e)}")
    
    # # Cache the results
    # with open(cache_file, 'w') as f:
    #     json.dump(all_info, f)
    
    return all_info

def process_names_csv(input_file, output_file, tally_file):
    bowling_types = Counter()
    
    with open(input_file, 'r') as csvfile, \
         open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(csvfile)
        writer = csv.writer(outfile)
        
        # Write header
        writer.writerow(['identifier', 'name', 'full_name', 'bowling_styles', 'batting_styles', 'playing_roles', 'teams'])
        
        # Skip header in input file
        next(reader)
        
        for row in reader:
            original_identifier, name = row
            print(f"Processing: {name}")
            info_list = get_player_info(name)
            
            # Sort info_list by completeness (number of fields that are not "Not found")
            info_list.sort(key=lambda x: sum(1 for v in x.values() if v != "Not found"), reverse=True)
            
            for i, info in enumerate(info_list):
                if i == 0:
                    identifier = original_identifier
                else:
                    identifier = f"{original_identifier}_{i:03d}"
                
                writer.writerow([
                    identifier,
                    name,
                    info['full_name'],
                    info['bowling_style'],
                    info['batting_style'],
                    info['playing_role'],
                    info['teams']
                ])
                
                if info['bowling_style'] != "Not found":
                    bowling_types[info['bowling_style']] += 1

    # Write bowling type tally to a separate CSV file
    with open(tally_file, 'w', newline='') as tallyfile:
        tally_writer = csv.writer(tallyfile)
        tally_writer.writerow(['Bowling Style', 'Count'])
        for style, count in bowling_types.items():
            tally_writer.writerow([style, count])

    # Print bowling type tally to console
    print("\nBowling Style Tally:")
    for style, count in bowling_types.items():
        print(f"{style}: {count}")

# Main execution
input_file = 'data/names_v2.csv'
output_file = 'players_info.csv'
tally_file = 'bowling_style_tally.csv'

#clear_cache()
start_time = time.time()
process_names_csv(input_file, output_file, tally_file)
end_time = time.time()

print(f"\nPlayer information has been saved to {output_file}")
print(f"Bowling style tally has been saved to {tally_file}")
print(f"Total execution time: {end_time - start_time:.2f} seconds")