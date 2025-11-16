"""
Validate cricWAR results against the original paper.

The paper (Rafique, 2023) reports results for IPL 2019:
- Top Batters (Table 1): AD Russell, HH Pandya, CH Gayle
- Top Bowlers (Table 2): JJ Bumrah, JC Archer, Rashid Khan
- WAR Leaders (Table 3): AD Russell (2.25), JJ Bumrah (2.19), JC Archer (2.06)
- RPW: ~84.5 runs per win

This script:
1. Filters results to 2019 season only
2. Recalculates season-specific RPW
3. Compares with paper's reported values
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import statsmodels.api as sm


def load_ball_level_data(data_file: Path) -> pd.DataFrame:
    """Load ball-by-ball data with RAA."""
    print(f"Loading data from {data_file}...")
    df = pd.read_parquet(data_file)
    print(f"✓ Loaded {len(df):,} balls")
    print(f"  Seasons: {sorted(df['season'].unique())}")
    return df


def filter_to_season(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Filter data to specific season."""
    print(f"\nFiltering to season {season}...")
    df_season = df[df['season'] == season].copy()
    print(f"✓ Filtered to {len(df_season):,} balls from {df_season['match_id'].nunique()} matches")
    return df_season


def calculate_season_raa(df_season: pd.DataFrame) -> tuple:
    """Calculate RAA for a specific season."""
    print("\nCalculating season-specific RAA...")

    # Aggregate batter RAA
    batter_raa = df_season.groupby(['batter_id', 'batter_name']).agg({
        'batter_RAA': ['sum', 'mean', 'count'],
        'weighted_run_value': 'sum'
    }).round(4)

    batter_raa.columns = ['RAA', 'RAA_per_ball', 'balls_faced', 'total_weighted_runs']
    batter_raa = batter_raa.reset_index()
    batter_raa = batter_raa.sort_values('RAA', ascending=False)

    # Aggregate bowler RAA
    bowler_raa = df_season.groupby(['bowler_id', 'bowler_name']).agg({
        'bowler_RAA': ['sum', 'mean', 'count'],
        'weighted_run_value': 'sum'
    }).round(4)

    bowler_raa.columns = ['RAA', 'RAA_per_ball', 'balls_bowled', 'total_weighted_runs_against']
    bowler_raa = bowler_raa.reset_index()
    bowler_raa = bowler_raa.sort_values('RAA', ascending=False)

    print(f"✓ Calculated RAA for {len(batter_raa)} batters and {len(bowler_raa)} bowlers")

    return batter_raa, bowler_raa


def estimate_season_rpw(df_season: pd.DataFrame) -> float:
    """Estimate Runs Per Win for specific season."""
    print("\nEstimating season-specific RPW...")

    # Calculate cumulative score per match-innings
    df_sorted = df_season.sort_values(['match_id', 'innings', 'over', 'ball_in_over'])
    df_sorted['cumulative_runs'] = df_sorted.groupby(['match_id', 'innings'])['total_runs'].cumsum()

    # Get final score for each innings
    innings_final = df_sorted.groupby(['match_id', 'innings', 'batting_team']).agg({
        'cumulative_runs': 'last'
    }).reset_index()
    innings_final.rename(columns={'cumulative_runs': 'final_score'}, inplace=True)

    # Filter for main innings only
    innings_main = innings_final[innings_final['innings'].isin([1, 2])]

    # Pivot to get both innings scores
    match_scores = innings_main.pivot_table(
        index='match_id',
        columns='innings',
        values='final_score',
        aggfunc='first'
    ).reset_index()

    if 1 in match_scores.columns and 2 in match_scores.columns:
        match_scores.rename(columns={1: 'innings_1_score', 2: 'innings_2_score'}, inplace=True)

        # Get team names
        team_map = innings_main.pivot_table(
            index='match_id',
            columns='innings',
            values='batting_team',
            aggfunc='first'
        )
        if 1 in team_map.columns and 2 in team_map.columns:
            match_scores['team_1'] = team_map[1].values
            match_scores['team_2'] = team_map[2].values

    # Drop matches with missing scores
    match_scores = match_scores.dropna(subset=['innings_1_score', 'innings_2_score'])

    print(f"  Matches: {len(match_scores)}")

    # Calculate run differential and winner
    match_scores['run_diff_team1'] = match_scores['innings_1_score'] - match_scores['innings_2_score']
    match_scores['run_diff_team2'] = match_scores['innings_2_score'] - match_scores['innings_1_score']
    match_scores['team1_won'] = (match_scores['run_diff_team1'] > 0).astype(int)
    match_scores['team2_won'] = (match_scores['run_diff_team2'] > 0).astype(int)

    # Create team-level observations
    team_obs = []
    for _, row in match_scores.iterrows():
        team_obs.append({
            'match_id': row['match_id'],
            'team': row['team_1'],
            'run_diff': row['run_diff_team1'],
            'won': row['team1_won']
        })
        team_obs.append({
            'match_id': row['match_id'],
            'team': row['team_2'],
            'run_diff': row['run_diff_team2'],
            'won': row['team2_won']
        })

    team_df = pd.DataFrame(team_obs)

    # OLS Regression
    X = team_df['run_diff'].values.reshape(-1, 1)
    y = team_df['won'].values
    X = sm.add_constant(X)

    ols_model = sm.OLS(y, X).fit()
    beta_ols = ols_model.params[1]
    rpw_ols = 1 / beta_ols

    print(f"  β (slope):     {beta_ols:.6f}")
    print(f"  RPW (1/β):     {rpw_ols:.2f}")
    print(f"  R-squared:     {ols_model.rsquared:.4f}")

    return rpw_ols


def calculate_season_vorp_war(batter_raa: pd.DataFrame, bowler_raa: pd.DataFrame,
                              avg_raa_rep: dict, rpw: float) -> tuple:
    """Calculate VORP and WAR for season."""
    print("\nCalculating season-specific VORP and WAR...")

    # Batter VORP and WAR
    batter_raa['VORP'] = (
        batter_raa['RAA'] -
        (avg_raa_rep['avg_raa_rep_batting'] * batter_raa['balls_faced'])
    )
    batter_raa['WAR'] = batter_raa['VORP'] / rpw

    # Bowler VORP and WAR
    bowler_raa['VORP'] = (
        bowler_raa['RAA'] -
        (avg_raa_rep['avg_raa_rep_bowling'] * bowler_raa['balls_bowled'])
    )
    bowler_raa['WAR'] = bowler_raa['VORP'] / rpw

    # Sort by WAR
    batter_raa = batter_raa.sort_values('WAR', ascending=False)
    bowler_raa = bowler_raa.sort_values('WAR', ascending=False)

    print(f"✓ Calculated VORP and WAR")

    return batter_raa, bowler_raa


def validate_results(batter_war: pd.DataFrame, bowler_war: pd.DataFrame, rpw: float):
    """Validate results against paper."""
    print("\n" + "="*70)
    print("VALIDATION AGAINST PAPER (Rafique, 2023)")
    print("="*70)

    # Paper's expected results for IPL 2019
    paper_top_batters = ['AD Russell', 'HH Pandya', 'CH Gayle']
    paper_top_bowlers = ['JJ Bumrah', 'JC Archer', 'Rashid Khan']
    paper_war_leaders = {
        'AD Russell': 2.25,
        'JJ Bumrah': 2.19,
        'JC Archer': 2.06
    }
    paper_rpw = 84.5

    print("\n" + "-"*70)
    print("1. TOP BATTERS COMPARISON")
    print("-"*70)
    print(f"\nPaper's Top 3: {', '.join(paper_top_batters)}")
    print(f"\nOur Top 15:")
    top_batters = batter_war.head(15)
    print(top_batters[['batter_name', 'WAR', 'RAA', 'balls_faced']].to_string(index=False))

    # Check if paper's top batters are in our top 15
    print(f"\nValidation:")
    for name in paper_top_batters:
        batter_row = batter_war[batter_war['batter_name'] == name]
        if not batter_row.empty:
            rank = (batter_war['WAR'] > batter_row['WAR'].iloc[0]).sum() + 1
            war_val = batter_row['WAR'].iloc[0]
            print(f"  ✓ {name}: Rank #{rank}, WAR = {war_val:.2f}")
        else:
            print(f"  ✗ {name}: Not found in data")

    print("\n" + "-"*70)
    print("2. TOP BOWLERS COMPARISON")
    print("-"*70)
    print(f"\nPaper's Top 3: {', '.join(paper_top_bowlers)}")
    print(f"\nOur Top 15:")
    top_bowlers = bowler_war.head(15)
    print(top_bowlers[['bowler_name', 'WAR', 'RAA', 'balls_bowled']].to_string(index=False))

    # Check if paper's top bowlers are in our top 15
    print(f"\nValidation:")
    for name in paper_top_bowlers:
        bowler_row = bowler_war[bowler_war['bowler_name'] == name]
        if not bowler_row.empty:
            rank = (bowler_war['WAR'] > bowler_row['WAR'].iloc[0]).sum() + 1
            war_val = bowler_row['WAR'].iloc[0]
            print(f"  ✓ {name}: Rank #{rank}, WAR = {war_val:.2f}")
        else:
            print(f"  ✗ {name}: Not found in data")

    print("\n" + "-"*70)
    print("3. WAR LEADERS COMPARISON")
    print("-"*70)
    print("\nPaper's WAR values:")
    for name, war_val in paper_war_leaders.items():
        print(f"  {name}: {war_val:.2f}")

    # Map players to their primary role based on paper
    player_roles = {
        'AD Russell': 'batter',
        'JJ Bumrah': 'bowler',
        'JC Archer': 'bowler'
    }

    print("\nOur WAR values:")
    for name, paper_war in paper_war_leaders.items():
        role = player_roles.get(name, 'unknown')

        # Check appropriate table based on role
        if role == 'batter':
            batter_row = batter_war[batter_war['batter_name'] == name]
            if not batter_row.empty:
                our_war = batter_row['WAR'].iloc[0]
                diff = our_war - paper_war
                pct_diff = (diff / paper_war) * 100
                print(f"  {name} (bat): {our_war:.2f} (paper: {paper_war:.2f}, diff: {diff:+.2f}, {pct_diff:+.1f}%)")
            else:
                print(f"  {name}: Not found in batters")

        elif role == 'bowler':
            bowler_row = bowler_war[bowler_war['bowler_name'] == name]
            if not bowler_row.empty:
                our_war = bowler_row['WAR'].iloc[0]
                diff = our_war - paper_war
                pct_diff = (diff / paper_war) * 100
                print(f"  {name} (bowl): {our_war:.2f} (paper: {paper_war:.2f}, diff: {diff:+.2f}, {pct_diff:+.1f}%)")
            else:
                print(f"  {name}: Not found in bowlers")

        else:
            print(f"  {name}: Role unknown")

    print("\n" + "-"*70)
    print("4. RUNS PER WIN COMPARISON")
    print("-"*70)
    print(f"\nPaper RPW: ~{paper_rpw:.1f} runs")
    print(f"Our RPW:    {rpw:.1f} runs")
    diff = rpw - paper_rpw
    pct_diff = (diff / paper_rpw) * 100
    print(f"Difference: {diff:+.1f} runs ({pct_diff:+.1f}%)")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    # Check how many of paper's top players we got right
    batter_matches = sum(1 for name in paper_top_batters
                         if not batter_war[batter_war['batter_name'] == name].empty)
    bowler_matches = sum(1 for name in paper_top_bowlers
                         if not bowler_war[bowler_war['bowler_name'] == name].empty)

    print(f"\nTop Batters: {batter_matches}/{len(paper_top_batters)} found in our data")
    print(f"Top Bowlers: {bowler_matches}/{len(paper_top_bowlers)} found in our data")

    # RPW comparison
    if abs(pct_diff) < 10:
        print(f"RPW: ✓ Within 10% of paper's estimate")
    elif abs(pct_diff) < 20:
        print(f"RPW: ~ Within 20% of paper's estimate")
    else:
        print(f"RPW: ✗ More than 20% difference from paper")

    print("\nNote: Differences may be due to:")
    print("  - Different data versions/sources")
    print("  - Implementation details in regression models")
    print("  - Rounding in the paper")
    print("  - Different handling of edge cases")


def main():
    """Validate cricWAR implementation against paper."""

    project_root = Path(__file__).parent.parent
    data_file = project_root / 'results' / '06_context_adjustments' / 'ipl_with_raa.parquet'
    avg_raa_file = project_root / 'results' / '08_replacement_level' / 'avg_raa_replacement.json'
    output_dir = project_root / 'results' / 'validation'

    print("="*70)
    print("CRICWAR VALIDATION - IPL 2019")
    print("="*70)
    print("\nComparing our results with Rafique (2023) paper\n")

    # Load data
    df = load_ball_level_data(data_file)

    # Filter to 2019 season
    df_2019 = filter_to_season(df, '2019')

    # Calculate season-specific RAA
    batter_raa_2019, bowler_raa_2019 = calculate_season_raa(df_2019)

    # Estimate season-specific RPW
    rpw_2019 = estimate_season_rpw(df_2019)

    # Load replacement level (use overall avg.RAA_rep from all seasons)
    with open(avg_raa_file, 'r') as f:
        avg_raa_rep = json.load(f)

    # Calculate VORP and WAR
    batter_war_2019, bowler_war_2019 = calculate_season_vorp_war(
        batter_raa_2019, bowler_raa_2019, avg_raa_rep, rpw_2019
    )

    # Validate against paper
    validate_results(batter_war_2019, bowler_war_2019, rpw_2019)

    # Save 2019 results
    output_dir.mkdir(parents=True, exist_ok=True)

    batter_file = output_dir / 'batter_war_2019.csv'
    batter_war_2019.to_csv(batter_file, index=False)
    print(f"\n✓ Saved 2019 batter WAR to {batter_file}")

    bowler_file = output_dir / 'bowler_war_2019.csv'
    bowler_war_2019.to_csv(bowler_file, index=False)
    print(f"✓ Saved 2019 bowler WAR to {bowler_file}")

    rpw_data = {'rpw_2019': float(rpw_2019), 'paper_rpw': 84.5}
    rpw_file = output_dir / 'rpw_2019.json'
    with open(rpw_file, 'w') as f:
        json.dump(rpw_data, f, indent=2)
    print(f"✓ Saved 2019 RPW to {rpw_file}")


if __name__ == '__main__':
    main()
