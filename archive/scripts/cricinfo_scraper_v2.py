import csv
import requests
from bs4 import BeautifulSoup
from rpy2.robjects.packages import importr
import time
from pathlib import Path
from tqdm import tqdm

# Import R package for cricinfo ID lookup
r_package = importr('cricketdata')

def get_cricinfo_ids(player_name):
    """Get cricinfo ID(s) for a player name using R cricketdata package"""
    try:
        result = r_package.find_player_id(player_name)
        player_ids = result.rx2('ID')
        return [str(int(id)) for id in player_ids]
    except Exception as e:
        return []

def scrape_player_info(player_name, cricinfo_id):
    """
    Scrape player information from cricinfo page.

    Returns dict with player info or None if scraping failed.
    """
    # Construct URL - use player name in slug
    url = f"https://www.espncricinfo.com/cricketers/{player_name.lower().replace(' ', '-')}-{cricinfo_id}"

    # Browser-like headers to avoid bot detection
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': 'https://www.espncricinfo.com/',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)

        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.content, 'html.parser')

        info = {
            'cricinfo_id': cricinfo_id,
            'full_name': '',
            'bowling_style': '',
            'batting_style': '',
            'playing_role': '',
            'date_of_birth': '',
            'teams': ''
        }

        # Extract bowling style
        bowling_style_div = soup.find('p', string='Bowling Style')
        if bowling_style_div:
            span = bowling_style_div.find_next_sibling('span')
            if span:
                info['bowling_style'] = span.text.strip()

        # Extract batting style
        batting_style_div = soup.find('p', string='Batting Style')
        if batting_style_div:
            span = batting_style_div.find_next_sibling('span')
            if span:
                info['batting_style'] = span.text.strip()

        # Extract playing role
        playing_role_div = soup.find('p', string='Playing Role')
        if playing_role_div:
            span = playing_role_div.find_next_sibling('span')
            if span:
                info['playing_role'] = span.text.strip()

        # Extract full name
        full_name_div = soup.find('p', string='Full Name')
        if full_name_div:
            span = full_name_div.find_next_sibling('span', class_='ds-text-title-s')
            if span:
                info['full_name'] = span.text.strip()

        # Extract date of birth
        dob_div = soup.find('p', string='Born')
        if dob_div:
            span = dob_div.find_next_sibling('span')
            if span:
                info['date_of_birth'] = span.text.strip()

        # Extract teams
        teams_div = soup.find('p', string='TEAMS')
        if teams_div:
            teams_grid = teams_div.find_next_sibling('div', class_='ds-grid')
            if teams_grid:
                team_links = teams_grid.find_all('a')
                teams = [link['href'].split('/')[-1] for link in team_links]
                info['teams'] = '; '.join(teams)

        return info

    except Exception as e:
        return None

def load_scraped_cricinfo_ids(output_file):
    """
    Load set of cricinfo IDs already scraped from output file.
    This is our in-memory cache.
    """
    scraped_ids = set()
    output_path = Path(output_file)

    if output_path.exists():
        with open(output_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['cricinfo_id']:
                    scraped_ids.add(row['cricinfo_id'])

    return scraped_ids

def scrape_players(input_file='data/player_registry_ids.csv',
                   output_file='data/player_metadata.csv',
                   rate_limit=5):
    """
    Main scraping function.

    Args:
        input_file: CSV with registry_id, name columns
        output_file: Output CSV to append results
        rate_limit: Seconds to wait between requests (default 5)
    """

    # Load existing scraped IDs into memory (our cache)
    scraped_ids = load_scraped_cricinfo_ids(output_file)
    print(f"Loaded {len(scraped_ids)} already-scraped cricinfo IDs from cache")

    # Read input players
    input_path = Path(input_file)
    with open(input_path, 'r') as f:
        reader = csv.DictReader(f)
        players = list(reader)

    print(f"Found {len(players)} players to process")

    # Open output file for appending
    output_path = Path(output_file)
    file_exists = output_path.exists()

    with open(output_file, 'a', newline='') as outfile:
        writer = csv.writer(outfile)

        # Write header if new file
        if not file_exists:
            writer.writerow(['registry_id', 'name', 'cricinfo_id', 'full_name',
                           'bowling_style', 'batting_style', 'playing_role',
                           'teams', 'date_of_birth'])

        # Process each player
        stats = {'total': 0, 'skipped': 0, 'scraped': 0, 'failed': 0}

        for player in tqdm(players, desc="Processing players"):
            registry_id = player['registry_id']
            name = player['name']
            stats['total'] += 1

            # Get cricinfo ID(s) from R package
            cricinfo_ids = get_cricinfo_ids(name)

            if not cricinfo_ids:
                stats['failed'] += 1
                continue

            # Process each cricinfo ID for this player
            for cricinfo_id in cricinfo_ids:
                # Check cache - skip if already scraped
                if cricinfo_id in scraped_ids:
                    stats['skipped'] += 1
                    continue

                # Scrape the player
                info = scrape_player_info(name, cricinfo_id)

                if info:
                    # Write to output
                    writer.writerow([
                        registry_id,
                        name,
                        info['cricinfo_id'],
                        info['full_name'],
                        info['bowling_style'],
                        info['batting_style'],
                        info['playing_role'],
                        info['teams'],
                        info['date_of_birth']
                    ])

                    # Add to in-memory cache
                    scraped_ids.add(cricinfo_id)
                    stats['scraped'] += 1

                    # Flush to disk immediately
                    outfile.flush()
                else:
                    stats['failed'] += 1

                # Rate limiting - be nice to cricinfo
                time.sleep(rate_limit)

    # Print summary
    print(f"\n{'='*50}")
    print(f"Scraping complete!")
    print(f"Total players: {stats['total']}")
    print(f"Already cached: {stats['skipped']}")
    print(f"Newly scraped: {stats['scraped']}")
    print(f"Failed: {stats['failed']}")
    print(f"{'='*50}")
    print(f"Output saved to: {output_file}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Scrape player info from cricinfo')
    parser.add_argument('--input', default='data/player_registry_ids.csv',
                       help='Input CSV with registry IDs')
    parser.add_argument('--output', default='data/player_metadata.csv',
                       help='Output CSV for player metadata')
    parser.add_argument('--rate-limit', type=int, default=5,
                       help='Seconds between requests (default 5)')

    args = parser.parse_args()

    start_time = time.time()
    scrape_players(args.input, args.output, args.rate_limit)
    end_time = time.time()

    print(f"\nTotal execution time: {(end_time - start_time)/60:.1f} minutes")
