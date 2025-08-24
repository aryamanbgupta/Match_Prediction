import pandas as pd

# Read and view
df = pd.read_parquet('cricket_data.parquet')

# Quick inspection
print(df.info())
print(df.head())
print(df.describe())

# Check specific columns
print(df[['match_id', 'runs_scored', 'wickets_fallen', 'is_wicket']].head(10))