
import pandas as pd                                                                                                   
import numpy as np

print('=' * 70)
print('IPL 2025 SEASON - WAR AND VORP CALCULATION')
print('=' * 70)

# Load ball-by-ball data with RAA
df_all = pd.read_parquet('/Users/aryamangupta/CricML/Match_Prediction/cricWAR/results/06_context_adjustments/ipl_with_raa.parquet')

# Filter for 2025 season only
df = df_all[df_all['season'] == '2025'].copy()
print(f'\n2025 Season Data:')
print(f'  Matches: {df["match_id"].nunique()} ')
print(f'  Balls: {len(df):,}')
print(f'  Unique batters: {df["batter_id"].nunique()}')
print(f'  Unique bowlers: {df["bowler_id"].nunique()}')

# Load replacement level values (from all seasons)
import json
with open('/Users/aryamangupta/CricML/Match_Prediction/cricWAR/results/08_replacement_level/avg_raa_replacement.json','r') as f:
    replacement = json.load(f)

avg_raa_rep_batting = replacement['avg_raa_rep_batting']
avg_raa_rep_bowling = replacement['avg_raa_rep_bowling']

print(f'\nUsing replacement level values (from all seasons):')
print(f'  avg.RAA_rep (batting): {avg_raa_rep_batting:.4f}')
print(f'  avg.RAA_rep (bowling): {avg_raa_rep_bowling:.4f}')

# Aggregate batter RAA for 2025
batter_agg = df.groupby(['batter_id', 'batter_name']).agg({
    'batter_RAA': 'sum',
    'batter_id': 'count'
}).rename(columns={'batter_id': 'balls_faced', 'batter_RAA': 'RAA'})
batter_agg['RAA_per_ball'] = batter_agg['RAA'] / batter_agg['balls_faced']
batter_agg = batter_agg.reset_index()

# Aggregate bowler RAA for 2025
bowler_agg = df.groupby(['bowler_id', 'bowler_name']).agg({
    'bowler_RAA': 'sum',
    'bowler_id': 'count'
}).rename(columns={'bowler_id': 'balls_bowled', 'bowler_RAA': 'RAA'})
bowler_agg['RAA_per_ball'] = bowler_agg['RAA'] / bowler_agg['balls_bowled']
bowler_agg = bowler_agg.reset_index()

print(f'\n2025 Season Players:')
print(f'  Batters: {len(batter_agg)}')
print(f'  Bowlers: {len(bowler_agg)}')

# Calculate VORP for batters
batter_agg['VORP'] = batter_agg['RAA'] - (avg_raa_rep_batting * batter_agg['balls_faced'])

# Calculate VORP for bowlers
bowler_agg['VORP'] = bowler_agg['RAA'] - (avg_raa_rep_bowling * bowler_agg['balls_bowled'])

# Use RPW from all seasons (107.37)
RPW = 107.37

# Calculate WAR
batter_agg['WAR'] = batter_agg['VORP'] / RPW
batter_agg['WAR_per_ball'] = batter_agg['WAR'] / batter_agg['balls_faced']

bowler_agg['WAR'] = bowler_agg['VORP'] / RPW
bowler_agg['WAR_per_ball'] = bowler_agg['WAR'] / bowler_agg['balls_bowled']

# Sort by WAR
batter_agg = batter_agg.sort_values('WAR', ascending=False)
bowler_agg = bowler_agg.sort_values('WAR', ascending=False)

print('\n' + '=' * 70)
print('TOP 15 BATTERS - IPL 2025 SEASON')
print('=' * 70)
print(batter_agg.head(15)[['batter_name', 'WAR', 'VORP', 'RAA', 'balls_faced',
'WAR_per_ball']].to_string(index=False))

print('\n' + '=' * 70)
print('TOP 15 BOWLERS - IPL 2025 SEASON')
print('=' * 70)
print(bowler_agg.head(15)[['bowler_name', 'WAR', 'VORP', 'RAA', 'balls_bowled',
'WAR_per_ball']].to_string(index=False))

# Combined leaderboard
batters_combined = batter_agg[['batter_name', 'WAR', 'VORP', 'balls_faced']].copy()
batters_combined.columns = ['player_name', 'WAR', 'VORP', 'balls']
batters_combined['role'] = 'Batter'

bowlers_combined = bowler_agg[['bowler_name', 'WAR', 'VORP', 'balls_bowled']].copy()
bowlers_combined.columns = ['player_name', 'WAR', 'VORP', 'balls']
bowlers_combined['role'] = 'Bowler'

combined = pd.concat([batters_combined, bowlers_combined]).sort_values('WAR', ascending=False)

print('\n' + '=' * 70)
print('TOP 20 PLAYERS (ALL ROLES) - IPL 2025 SEASON')
print('=' * 70)
print(combined.head(20).to_string(index=False))

# Summary statistics
print('\n' + '=' * 70)
print('IPL 2025 SEASON SUMMARY STATISTICS')
print('=' * 70)
print(f'\nBATTERS:')
print(f'  Total players: {len(batter_agg)}')
print(f'  Mean WAR: {batter_agg["WAR"].mean():.3f}')
print(f'  Median WAR: {batter_agg["WAR"].median():.3f}')
print(f'  Max WAR: {batter_agg["WAR"].max():.3f} ({batter_agg.iloc[0]["batter_name"]})')
print(f'  Min WAR: {batter_agg["WAR"].min():.3f} ({batter_agg.iloc[-1]["batter_name"]})')
print(f'  Players with WAR > 1.0: {(batter_agg["WAR"] > 1.0).sum()}')
print(f'  Players with WAR > 0.5: {(batter_agg["WAR"] > 0.5).sum()}')

print(f'\nBOWLERS:')
print(f'  Total players: {len(bowler_agg)}')
print(f'  Mean WAR: {bowler_agg["WAR"].mean():.3f}')
print(f'  Median WAR: {bowler_agg["WAR"].median():.3f}')
print(f'  Max WAR: {bowler_agg["WAR"].max():.3f} ({bowler_agg.iloc[0]["bowler_name"]})')
print(f'  Min WAR: {bowler_agg["WAR"].min():.3f} ({bowler_agg.iloc[-1]["bowler_name"]})')
print(f'  Players with WAR > 1.0: {(bowler_agg["WAR"] > 1.0).sum()}')
print(f'  Players with WAR > 0.5: {(bowler_agg["WAR"] > 0.5).sum()}')

# Save results
output_dir = '/Users/aryamangupta/CricML/Match_Prediction/cricWAR/results/2025_season/'
import os
os.makedirs(output_dir, exist_ok=True)

batter_agg.to_csv(output_dir + 'batter_war_2025.csv', index=False)
bowler_agg.to_csv(output_dir + 'bowler_war_2025.csv', index=False)

print(f'\n✓ Results saved to {output_dir}')
print('  - batter_war_2025.csv')
print('  - bowler_war_2025.csv')

print('\n' + '=' * 70)
print('✓ IPL 2025 SEASON WAR CALCULATION COMPLETE')
print('=' * 70)

'''
  ⎿  ======================================================================                     
     IPL 2025 SEASON - WAR AND VORP CALCULATION
     ======================================================================

     2025 Season Data:
       Matches: 74
       Balls: 17,444
       Unique batters: 166
       Unique bowlers: 128

     Using replacement level values (from all seasons):
       avg.RAA_rep (batting): -0.2573
       avg.RAA_rep (bowling): -0.2362

     2025 Season Players:
       Batters: 166
       Bowlers: 128

     ======================================================================
     TOP 15 BATTERS - IPL 2025 SEASON
     ======================================================================
         batter_name      WAR       VORP        RAA  balls_faced  WAR_per_ball
       Priyansh Arya 1.880232 201.880526 120.305414          317      0.005931
            SA Yadav 1.868321 200.601653  86.602364          443      0.004217
             SS Iyer 1.699279 182.451551  89.811046          360      0.004720
      P Simran Singh 1.576502 169.268966  72.511105          376      0.004193
     Abhishek Sharma 1.454505 156.170227  93.895221          242      0.006010
            N Pooran 1.426220 153.133254  77.219506          295      0.004835
            MR Marsh 1.343911 144.295672  38.788430          410      0.003278
             PD Salt 1.326228 142.397075  80.379404          241      0.005503
         YBK Jaiswal 1.260777 135.369654  42.729149          360      0.003502
     B Sai Sudharsan 1.250655 134.282839   3.299458          509      0.002457
           H Klaasen 1.229648 132.027339  56.370927          294      0.004182
            KL Rahul 1.168481 125.459825  28.701964          376      0.003108
            A Mhatre 1.032257 110.833417  75.578558          137      0.007535
          AK Markram 0.984488 105.704488  24.644046          315      0.003125
        RD Rickelton 0.972594 104.427460  35.976420          266      0.003656

     ======================================================================
     TOP 15 BOWLERS - IPL 2025 SEASON
     ======================================================================
           bowler_name      WAR       VORP        RAA  balls_bowled  WAR_per_ball
             JJ Bumrah 1.593887 171.135632 102.875601           289      0.005515
         Kuldeep Yadav 1.492679 160.268925  83.742109           324      0.004607
     M Prasidh Krishna 1.350993 145.056125  59.081553           364      0.003712
              CV Varun 1.039262 111.585603  38.837889           308      0.003374
            Noor Ahmad 1.011046 108.555959  33.446306           318      0.003179
             SP Narine 1.009509 108.390945  43.437629           275      0.003671
        Arshdeep Singh 0.989061 106.195441  19.276094           368      0.002688
           M Pathirana 0.953810 102.410626  33.442014           292      0.003266
              DS Rathi 0.911136  97.828698  22.010463           321      0.002838
              HV Patel 0.864820  92.855713  25.304264           286      0.003024
             JC Archer 0.803234  86.243246  17.274634           292      0.002751
             KH Pandya 0.780859  83.840831  16.997964           283      0.002759
            PJ Cummins 0.761081  81.717273   8.024783           312      0.002439
            MJ Santner 0.739943  79.447682  21.107794           247      0.002996
        Sandeep Sharma 0.721353  77.451673  20.056561           243      0.002969

     ======================================================================
     TOP 20 PLAYERS (ALL ROLES) - IPL 2025 SEASON
     ======================================================================
           player_name      WAR       VORP  balls   role
         Priyansh Arya 1.880232 201.880526    317 Batter
              SA Yadav 1.868321 200.601653    443 Batter
               SS Iyer 1.699279 182.451551    360 Batter
             JJ Bumrah 1.593887 171.135632    289 Bowler
        P Simran Singh 1.576502 169.268966    376 Batter
         Kuldeep Yadav 1.492679 160.268925    324 Bowler
       Abhishek Sharma 1.454505 156.170227    242 Batter
              N Pooran 1.426220 153.133254    295 Batter
     M Prasidh Krishna 1.350993 145.056125    364 Bowler
              MR Marsh 1.343911 144.295672    410 Batter
               PD Salt 1.326228 142.397075    241 Batter
           YBK Jaiswal 1.260777 135.369654    360 Batter
       B Sai Sudharsan 1.250655 134.282839    509 Batter
             H Klaasen 1.229648 132.027339    294 Batter
              KL Rahul 1.168481 125.459825    376 Batter
              CV Varun 1.039262 111.585603    308 Bowler
              A Mhatre 1.032257 110.833417    137 Batter
            Noor Ahmad 1.011046 108.555959    318 Bowler
             SP Narine 1.009509 108.390945    275 Bowler
        Arshdeep Singh 0.989061 106.195441    368 Bowler

     ======================================================================
     IPL 2025 SEASON SUMMARY STATISTICS
     ======================================================================

     BATTERS:
       Total players: 166
       Mean WAR: 0.304
       Median WAR: 0.104
       Max WAR: 1.880 (Priyansh Arya)
       Min WAR: -0.103 (M Theekshana)
       Players with WAR > 1.0: 13
       Players with WAR > 0.5: 44

     BOWLERS:
       Total players: 128
       Mean WAR: 0.232
       Median WAR: 0.091
       Max WAR: 1.594 (JJ Bumrah)
       Min WAR: -0.347 (Fazalhaq Farooqi)
       Players with WAR > 1.0: 6
       Players with WAR > 0.5: 27

     ✓ Results saved to /Users/aryamangupta/CricML/Match_Prediction/cricWAR/results/2025_season/
       - batter_war_2025.csv
       - bowler_war_2025.csv

     ======================================================================
     ✓ IPL 2025 SEASON WAR CALCULATION COMPLETE
     ======================================================================

'''