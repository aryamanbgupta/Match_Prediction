# cricWAR Visualization Notebooks

Interactive Jupyter notebooks for exploring and visualizing cricWAR results.

## Available Notebooks

### 01_war_results_visualization.ipynb
**Purpose**: Comprehensive visualization of all-time IPL WAR results (all seasons)

**Contents**:
- WAR distribution histograms for batters and bowlers
- Top 15 performers horizontal bar charts
- WAR vs playing time scatter plots with annotations
- Efficiency metrics (WAR per 100 balls)
- Combined leaderboard (batters + bowlers)
- Summary statistics
- Export functionality for top 50 players

**Use this for**: Overall historical performance analysis, identifying all-time greats, comparing batting vs bowling impact

---

### 02_expected_runs_validation.ipynb
**Purpose**: Validate and visualize the expected runs model θ(o,w)

**Contents**:
- Model coefficients display and interpretation
- Expected runs by game state (over × wickets)
- Expected vs actual runs scatter plots
- Run value distribution analysis
- Run values by phase and wickets
- Model fit statistics (Pseudo R², coefficients)
- Comprehensive validation summary

**Use this for**: Understanding the expected runs model, validating model assumptions, exploring game state dynamics

---

### 03_ipl_2025_war_analysis.ipynb
**Purpose**: In-depth analysis of IPL 2025 season performance

**Contents**:
- Top 15 batters and bowlers for 2025 season
- WAR distribution for 2025 season
- WAR vs playing time scatter plots
- Efficiency metrics (WAR per ball)
- Combined leaderboard (top 20 all roles)
- RAA vs VORP comparison
- Summary statistics for 2025
- Key insights and breakout performers

**Use this for**: Analyzing current season performance, identifying 2025 standouts, season-specific trends

---

### 04_historical_comparison.ipynb
**Purpose**: Compare 2025 season against all-time historical data

**Contents**:
- Top 2025 performers vs all-time rankings
- Distribution comparisons (WAR, efficiency)
- Elite players count (WAR > 1.0) comparison
- Per-season averages vs 2025
- Summary statistics table
- Historical context for 2025 performances

**Use this for**: Contextualizing 2025 season, comparing current vs historical performance, trend analysis

---

## How to Use

### Prerequisites
```bash
# Install required packages
uv sync

# Or with pip
pip install pandas numpy matplotlib seaborn jupyter
```

### Running the Notebooks

```bash
# Navigate to notebooks directory
cd /Users/aryamangupta/CricML/Match_Prediction/cricWAR/notebooks

# Launch Jupyter
jupyter notebook

# Or use Jupyter Lab
jupyter lab
```

Then open any notebook and run all cells (Cell → Run All) or run them sequentially.

---

## Data Sources

All notebooks load data from the `../results/` directory:

- **All-time data**: `results/09_vorp_war/`
  - `batter_war.csv`
  - `bowler_war.csv`

- **2025 season data**: `results/2025_season/`
  - `batter_war_2025.csv`
  - `bowler_war_2025.csv`

- **Model artifacts**: `results/03_expected_runs/`
  - `expected_runs_model.pkl`
  - `ipl_with_expected_runs.parquet`

---

## Visualization Types

### 1. Distribution Plots
- Histograms showing WAR distribution
- Comparison of different seasons/eras
- Mean and median indicators

### 2. Rankings
- Horizontal bar charts for top performers
- Annotated with WAR values and balls played
- Color-coded by role (batter/bowler)

### 3. Scatter Plots
- WAR vs playing time
- RAA vs VORP relationships
- Efficiency analysis
- Top players highlighted

### 4. Comparison Charts
- Side-by-side comparisons
- Historical vs current season
- All-time vs single-season

### 5. Summary Tables
- Statistical summaries
- Player rankings
- Distribution percentiles

---

## Key Metrics Visualized

- **WAR** (Wins Above Replacement): Total wins contributed above replacement level
- **VORP** (Value Over Replacement Player): Runs contributed above replacement level
- **RAA** (Runs Above Average): Context-adjusted runs above average
- **WAR per ball**: Efficiency metric (WAR per 100 balls)
- **Elite threshold**: WAR > 1.0 (season MVP candidates)

---

## Customization

Each notebook can be easily customized:

1. **Change thresholds**: Modify WAR > 1.0 to different values
2. **Filter players**: Add minimum balls faced/bowled filters
3. **Adjust visualizations**: Change colors, sizes, fonts
4. **Export results**: Save figures or data to CSV

Example:
```python
# Show top 20 instead of top 15
top_batters = batter_war.head(20).sort_values('WAR')

# Filter for players with more playing time
qualified = batter_war[batter_war['balls_faced'] >= 200]

# Export to CSV
top_batters.to_csv('my_custom_ranking.csv', index=False)
```

---

## Tips

- **Large datasets**: If visualizations are slow, reduce sample size or use `%%time` magic to profile
- **Export plots**: Use `plt.savefig('filename.png', dpi=300, bbox_inches='tight')`
- **Interactive plots**: Consider using `plotly` for interactive visualizations
- **Custom analysis**: Add new cells to explore specific questions

---

## Troubleshooting

**Problem**: Module not found error
**Solution**: Make sure you're in the cricWAR virtual environment (`uv sync`)

**Problem**: File not found error
**Solution**: Check that you've run scripts 01-09 to generate data files

**Problem**: Plots not showing
**Solution**: Add `%matplotlib inline` at the top of the notebook

**Problem**: Out of memory
**Solution**: Reduce data size or close other applications

---

## Future Enhancements

Potential additions:
- Season-by-season trend analysis
- Player career trajectory visualizations
- Team-level WAR aggregation
- Match-level impact analysis
- Uncertainty visualization (from bootstrap results)
- Venue-specific performance analysis

---

**Framework**: cricWAR (Rafique, 2023)
**Implementation**: Aryaman Gupta
**Last Updated**: November 2024
