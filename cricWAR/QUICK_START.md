# cricWAR Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Explore Pre-Computed Results

Results are already computed! Just open the visualizations:

```bash
cd /Users/aryamangupta/CricML/Match_Prediction/cricWAR/notebooks
uv run jupyter lab
```

**Open**: `03_ipl_2025_war_analysis.ipynb` for latest season results

**Select kernel**: "Python (cricWAR)" (top right)

**Run**: `Kernel → Restart & Run All`

---

### 2. View Results Files

All results are in the `results/` directory:

```bash
# All-time WAR leaders
cat results/09_vorp_war/batter_war.csv | head -20
cat results/09_vorp_war/bowler_war.csv | head -20

# 2025 season results
cat results/2025_season/batter_war_2025.csv | head -10
cat results/2025_season/bowler_war_2025.csv | head -10
```

---

### 3. Run the Full Pipeline (Optional)

If you want to recompute from scratch:

```bash
# Navigate to project
cd /Users/aryamangupta/CricML/Match_Prediction/cricWAR

# Run pipeline (takes ~15-20 minutes)
uv run python scripts/01_extract_ipl_data.py        # ~2 min
uv run python scripts/02_fetch_player_metadata.py   # ~1 min
uv run python scripts/03_expected_runs_model.py     # ~2 min
uv run python scripts/04_calculate_run_values.py    # ~1 min
uv run python scripts/05_calculate_leverage_index.py # ~1 min
uv run python scripts/06_context_adjustments.py     # ~2 min
uv run python scripts/08_replacement_level.py       # <1 min
uv run python scripts/09_vorp_war.py                # ~1 min
```

---

## 📊 What's Available

### All-Time Results (All Seasons: 2007-2025)
- **1,169 matches**
- **278,205 balls**
- **703 batters**, 551 bowlers
- **Top player**: Sunil Narine (17.80 WAR)

### 2025 Season Results
- **74 matches**
- **17,444 balls**
- **166 batters**, 128 bowlers
- **Top batter**: Priyansh Arya (1.88 WAR)
- **Top bowler**: Jasprit Bumrah (1.59 WAR)

### Visualizations
- 4 comprehensive Jupyter notebooks
- 20+ interactive plots
- All-time and 2025 comparisons

---

## 🎯 Common Tasks

### Check Top Players

```bash
cd /Users/aryamangupta/CricML/Match_Prediction/cricWAR

# Top 10 batters all-time
uv run python -c "
import pandas as pd
df = pd.read_csv('results/09_vorp_war/batter_war.csv')
print(df.head(10)[['batter_name', 'WAR', 'balls_faced']])
"

# Top 10 bowlers 2025
uv run python -c "
import pandas as pd
df = pd.read_csv('results/2025_season/bowler_war_2025.csv')
print(df.head(10)[['bowler_name', 'WAR', 'balls_bowled']])
"
```

### Find a Specific Player

```bash
# Example: Find Jasprit Bumrah
uv run python -c "
import pandas as pd
# All-time
all_time = pd.read_csv('results/09_vorp_war/bowler_war.csv')
player = all_time[all_time['bowler_name'] == 'JJ Bumrah']
print('All-time:', player[['bowler_name', 'WAR', 'balls_bowled']].to_string(index=False))

# 2025
season_2025 = pd.read_csv('results/2025_season/bowler_war_2025.csv')
player = season_2025[season_2025['bowler_name'] == 'JJ Bumrah']
print('\n2025:', player[['bowler_name', 'WAR', 'balls_bowled']].to_string(index=False))
"
```

---

## 📂 Key Files

| File | Description |
|------|-------------|
| `results/09_vorp_war/batter_war.csv` | All-time batter WAR rankings |
| `results/09_vorp_war/bowler_war.csv` | All-time bowler WAR rankings |
| `results/2025_season/batter_war_2025.csv` | 2025 season batters |
| `results/2025_season/bowler_war_2025.csv` | 2025 season bowlers |
| `notebooks/03_ipl_2025_war_analysis.ipynb` | 2025 visualizations |
| `notebooks/04_historical_comparison.ipynb` | Historical trends |

---

## 🆘 Quick Help

**Problem**: Can't open notebooks
**Solution**:
```bash
uv add jupyter ipykernel matplotlib seaborn
uv run python -m ipykernel install --user --name=cricwar --display-name="Python (cricWAR)"
```

**Problem**: Wrong Python version
**Solution**: Ensure you're using `uv run` prefix for all commands

**Problem**: Missing data
**Solution**: Run the pipeline scripts in order (01 through 09)

---

## 📖 Full Documentation

- **README.md** - Complete project overview
- **METHODOLOGY.md** - Technical details
- **notebooks/GETTING_STARTED.md** - Jupyter setup
- **notebooks/README.md** - Visualization guide

---

## 🏏 Quick Facts

**What is WAR?**
- Wins Above Replacement: How many wins a player adds vs a replacement-level player
- WAR > 2.0: Elite (MVP candidates)
- WAR 1.0-2.0: Very good (key contributors)
- WAR 0.5-1.0: Good (solid regulars)
- WAR < 0.5: Below average

**Key Metrics**:
- **VORP**: Value Over Replacement Player (in runs)
- **RAA**: Runs Above Average (context-adjusted)
- **RPW**: Runs Per Win (~107 for IPL)

---

**Ready to explore? Start with the notebooks!**

```bash
cd notebooks/
uv run jupyter lab
```

Open `03_ipl_2025_war_analysis.ipynb` and enjoy! 🎉
