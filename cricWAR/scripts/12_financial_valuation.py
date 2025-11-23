"""
Calculate Financial Valuation Metrics (Moneyball Stats) for IPL 2025.

Metrics:
1. ROI_basic: Wins per 1% of Cap Spent.
2. VOPE_simple: Value Over Price Expectation (Polynomial Regression).
3. VOMAM: Value Over Market Adjusted Model (Multivariate Regression).

Inputs:
- WAR results (2025)
- Player Prices (2025)
- Player Metadata (Role, Country)

Constraints:
- Team Purse 2025 = 120 Cr
- Uncapped status ignored (data unavailable)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
TEAM_PURSE_2025 = 120.0  # Crores

def load_and_prep_data(project_root):
    """Load data and prepare features."""
    print("Loading and preparing data...")
    
    # Load mapped price/WAR data from previous step
    # This has price_cr and total_WAR, but we need to re-merge metadata for Role/Country
    # Or we can load the raw files again. Let's load the output from step 11 as a base.
    war_price_file = project_root / 'results' / '11_war_vs_price' / 'war_vs_price_full.csv'
    df = pd.read_csv(war_price_file)
    
    # Filter out unmapped players (where cricwar_name is NaN or match_type is 'none')
    df = df[df['match_type'] != 'none'].copy()
    print(f"✓ Loaded {len(df)} mapped players")
    
    # Load metadata for Role and Country
    meta_file = project_root / 'data' / 'player_metadata.csv'
    metadata = pd.read_csv(meta_file)
    
    # Merge metadata
    # df has 'cricwar_name', metadata has 'player_name'
    df = df.merge(metadata[['player_name', 'country', 'role_category']], 
                  left_on='cricwar_name', right_on='player_name', how='left')
    
    # 1. Calculate price_norm (% of Cap)
    df['price_norm'] = (df['price_cr'] / TEAM_PURSE_2025) * 100
    
    # 2. Derive Is_Overseas
    # Assuming 'India' is domestic. 
    # Note: country names might vary (e.g. 'India', 'IND'). Let's check unique values if needed.
    # For now, assume standard names.
    df['is_overseas'] = df['country'].apply(lambda x: 0 if str(x).lower() in ['india', 'ind'] else 1)
    
    # 3. Process Role
    # Fill missing roles
    df['role_category'] = df['role_category'].fillna('Unknown')
    
    print(f"✓ Data prepared. Columns: {df.columns.tolist()}")
    return df

def calculate_roi_basic(df):
    """Calculate Metric 1: ROI Basic (Wins per 1% Cap)."""
    print("\nCalculating Metric 1: ROI Basic...")
    
    # Avoid division by zero for replacement players (price ~ 0)
    # Set a floor of 0.05% (approx 6 Lakhs)
    min_price_norm = 0.05
    df['price_norm_adj'] = df['price_norm'].clip(lower=min_price_norm)
    
    df['roi_basic'] = df['total_WAR'] / df['price_norm_adj']
    
    # Handle negative WAR (Pay to Lose)
    # The formula naturally handles this (negative numerator -> negative ROI)
    
    print("✓ ROI Basic calculated")
    return df

def calculate_vope_simple(df):
    """Calculate Metric 2: VOPE Simple (Polynomial Regression)."""
    print("\nCalculating Metric 2: VOPE Simple...")
    
    # Target: WAR_realized
    # Feature: price_norm
    
    X = df[['price_norm']].values
    y = df['total_WAR'].values
    
    # Polynomial features (Degree 2)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # Fit model
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Predict
    df['xWAR_price_only'] = model.predict(X_poly)
    
    # Calculate Residual
    df['vope_simple'] = df['total_WAR'] - df['xWAR_price_only']
    
    print(f"✓ Model fit: R² = {model.score(X_poly, y):.4f}")
    return df

def calculate_vomam(df):
    """Calculate Metric 3: VOMAM (Multivariate Regression)."""
    print("\nCalculating Metric 3: VOMAM...")
    
    # Features: price_norm, is_overseas, role dummies
    # Excluded: is_uncapped (unavailable)
    
    # Create dummies for role
    role_dummies = pd.get_dummies(df['role_category'], prefix='role', drop_first=True, dtype=int)
    
    # Prepare X matrix
    X = df[['price_norm', 'is_overseas']]
    X = pd.concat([X, role_dummies], axis=1)
    
    # Add constant
    X = sm.add_constant(X)
    y = df['total_WAR']
    
    # Fit OLS
    model = sm.OLS(y, X).fit()
    
    # Predict
    df['xWAR_market_adj'] = model.predict(X)
    
    # Calculate Residual
    df['vomam_score'] = df['total_WAR'] - df['xWAR_market_adj']
    
    print("✓ Model Summary:")
    print(model.summary())
    
    return df

def generate_visualizations(df, output_dir):
    """Generate plots."""
    print("\nGenerating visualizations...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Price vs WAR with Regression Curve (VOPE)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='price_norm', y='total_WAR', hue='role_category', alpha=0.7)
    
    # Plot regression line
    x_range = np.linspace(df['price_norm'].min(), df['price_norm'].max(), 100).reshape(-1, 1)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    x_poly = poly.fit_transform(x_range)
    
    # Re-fit simple model for plotting line (quick & dirty)
    model = LinearRegression()
    model.fit(poly.fit_transform(df[['price_norm']]), df['total_WAR'])
    y_pred = model.predict(x_poly)
    
    plt.plot(x_range, y_pred, color='red', label='Expected WAR (Price only)')
    
    plt.title('Price (% of Cap) vs Realized WAR (2025)')
    plt.xlabel('Price (% of Salary Cap)')
    plt.ylabel('Realized WAR')
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_dir / 'price_vs_war_curve.png', dpi=300)
    plt.close()
    
    # 2. VOMAM Score Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='vomam_score', kde=True, bins=20)
    plt.title('Distribution of VOMAM Scores (Value Over Market Adjusted Model)')
    plt.xlabel('VOMAM Score (WAR above expectation)')
    plt.axvline(0, color='red', linestyle='--')
    
    plt.savefig(output_dir / 'vomam_distribution.png', dpi=300)
    plt.close()

def save_results(df, output_dir):
    """Save final artifacts."""
    print("\nSaving results...")
    
    # Select columns for final output
    cols = [
        'price_name', 'team', 'role_category', 'is_overseas',
        'price_cr', 'price_norm', 'total_WAR',
        'roi_basic', 
        'xWAR_price_only', 'vope_simple',
        'xWAR_market_adj', 'vomam_score'
    ]
    
    final_df = df[cols].sort_values('vomam_score', ascending=False)
    
    final_df.to_csv(output_dir / 'financial_valuation_full.csv', index=False)
    
    # Top 10 Steals (VOMAM)
    print("\nTOP 10 STEALS (Highest VOMAM):")
    print(final_df.head(10)[['price_name', 'price_cr', 'total_WAR', 'vomam_score']].to_string(index=False))
    
    # Top 10 Overpays (Lowest VOMAM)
    print("\nTOP 10 OVERPAYS (Lowest VOMAM):")
    print(final_df.tail(10)[['price_name', 'price_cr', 'total_WAR', 'vomam_score']].to_string(index=False))
    
    print(f"\n✓ Saved results to {output_dir}")

def main():
    project_root = Path(__file__).parent.parent
    output_dir = project_root / 'results' / '12_financial_valuation'
    
    # 1. Load & Prep
    df = load_and_prep_data(project_root)
    
    # 2. Metric 1: ROI Basic
    df = calculate_roi_basic(df)
    
    # 3. Metric 2: VOPE Simple
    df = calculate_vope_simple(df)
    
    # 4. Metric 3: VOMAM
    df = calculate_vomam(df)
    
    # 5. Visualize
    generate_visualizations(df, output_dir)
    
    # 6. Save
    save_results(df, output_dir)

if __name__ == "__main__":
    main()
