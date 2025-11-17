# Getting Started with cricWAR Notebooks

## ✅ Setup Complete!

Your `uv` environment is now configured for Jupyter notebooks. The kernel **"Python (cricWAR)"** has been installed and all notebooks have been updated to use it.

---

## 🚀 How to Launch Jupyter

### Option 1: Jupyter Notebook (Classic Interface)
```bash
cd /Users/aryamangupta/CricML/Match_Prediction/cricWAR/notebooks
uv run jupyter notebook
```

### Option 2: Jupyter Lab (Modern Interface - Recommended)
```bash
cd /Users/aryamangupta/CricML/Match_Prediction/cricWAR/notebooks
uv run jupyter lab
```

This will:
1. Start the Jupyter server
2. Open your browser automatically
3. Show the notebooks directory

---

## 📝 Using the Notebooks

### When you open a notebook:

1. **Verify the kernel** (top right corner):
   - Should show: **"Python (cricWAR)"**
   - If it shows something else, click it and select "Python (cricWAR)"

2. **Run all cells**:
   - Menu: `Kernel → Restart & Run All`
   - Or use keyboard shortcut: `Shift + Enter` to run cells one by one

3. **View outputs**:
   - Plots will appear inline
   - Tables will be formatted nicely
   - Statistics will be printed below cells

---

## 🎯 Quick Start

### Recommended Order:

1. **Start here**: `01_war_results_visualization.ipynb`
   - Overview of all-time results
   - Top performers across all seasons

2. **Understand the model**: `02_expected_runs_validation.ipynb`
   - How the expected runs model works
   - Model validation and diagnostics

3. **2025 season**: `03_ipl_2025_war_analysis.ipynb`
   - Latest IPL 2025 results
   - Current season standouts

4. **Historical context**: `04_historical_comparison.ipynb`
   - Compare 2025 vs all-time
   - Trends and insights

---

## 🔧 Troubleshooting

### Problem: "Kernel not found" or wrong kernel selected

**Solution**:
```bash
# Re-register the kernel
uv run python -m ipykernel install --user --name=cricwar --display-name="Python (cricWAR)"
```

Then restart Jupyter and select "Python (cricWAR)" as the kernel.

---

### Problem: ModuleNotFoundError

**Solution**: Make sure all packages are installed
```bash
# From cricWAR directory
uv sync
```

---

### Problem: Plots not showing

**Solution**: Add this to the first cell of the notebook:
```python
%matplotlib inline
```

---

### Problem: "No module named 'pandas'" or similar

**Solution**: You're using the wrong kernel.
1. Click the kernel name (top right)
2. Select "Python (cricWAR)"
3. Click "Select"
4. Restart kernel: `Kernel → Restart & Run All`

---

## 🎨 Tips & Tricks

### 1. Change Plot Size
```python
plt.rcParams['figure.figsize'] = (16, 8)  # Larger plots
```

### 2. Export Plots
```python
plt.savefig('my_plot.png', dpi=300, bbox_inches='tight')
```

### 3. Export Data
```python
# Save filtered results
top_players = batter_war.head(20)
top_players.to_csv('top_20_batters.csv', index=False)
```

### 4. Interactive Plots (Optional)
```python
# Install plotly first: uv add plotly
import plotly.express as px
fig = px.scatter(batter_war, x='balls_faced', y='WAR', hover_data=['batter_name'])
fig.show()
```

### 5. Keyboard Shortcuts
- `Shift + Enter`: Run current cell and move to next
- `Ctrl + Enter`: Run current cell and stay
- `A`: Insert cell above
- `B`: Insert cell below
- `DD`: Delete cell
- `M`: Convert to Markdown
- `Y`: Convert to Code

---

## 📊 What's in Each Notebook

### 01_war_results_visualization.ipynb
- **8 sections**, 20+ visualizations
- All-time IPL leaders
- Historical performance analysis

### 02_expected_runs_validation.ipynb
- **7 sections**, model validation
- Expected runs by game state
- Run value distributions

### 03_ipl_2025_war_analysis.ipynb
- **9 sections**, 2025 season deep dive
- Top performers this season
- Breakout players

### 04_historical_comparison.ipynb
- **6 sections**, comparative analysis
- 2025 vs all-time rankings
- Distribution comparisons

---

## 🌟 Advanced: VS Code Integration (Optional)

If you use VS Code:

1. Install Jupyter extension in VS Code
2. Open any `.ipynb` file
3. Click "Select Kernel" (top right)
4. Choose "Python (cricWAR)"
5. Run cells directly in VS Code!

---

## 📦 Kernel Information

**Kernel Name**: `cricwar`
**Display Name**: `Python (cricWAR)`
**Location**: `/Users/aryamangupta/Library/Jupyter/kernels/cricwar`

To view all available kernels:
```bash
jupyter kernelspec list
```

To remove the kernel (if needed):
```bash
jupyter kernelspec uninstall cricwar
```

---

## 🆘 Need Help?

1. **Check kernel**: Make sure "Python (cricWAR)" is selected
2. **Restart kernel**: `Kernel → Restart & Clear Output`
3. **Check data files**: Ensure you've run scripts 01-09
4. **Update packages**: Run `uv sync` in the cricWAR directory

---

## ✨ You're Ready!

Everything is set up and ready to use. Just run:

```bash
cd /Users/aryamangupta/CricML/Match_Prediction/cricWAR/notebooks
uv run jupyter lab
```

And start exploring the cricWAR visualizations! 🏏📊

---

**Last Updated**: November 2024
**Environment**: uv-managed Python virtual environment
**Kernel**: Python (cricWAR)
