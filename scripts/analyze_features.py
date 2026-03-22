"""
Feature Importance Analysis for XGBoost model.

Computes:
1. XGBoost built-in importances (gain, weight, cover)
2. Spearman correlation of each feature with target
3. Feature-to-feature Pearson correlation (redundancy detection)
4. Per-group importance aggregation (from feature_registry.py)

Usage:
    uv run python scripts/analyze_features.py
    uv run python scripts/analyze_features.py --model-dir models/xgb_v3
    uv run python scripts/analyze_features.py --output-dir experiments/results/some_exp/
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from feature_registry import FEATURE_GROUPS

parser = argparse.ArgumentParser(description='Analyze XGBoost feature importances')
parser.add_argument('--model-dir', type=str, default='models/xgb_v3',
                    help='Directory containing trained model and feature columns')
parser.add_argument('--data', type=str, default='data/xgb_data_v3/cricket_data_v3_train.parquet',
                    help='Path to training parquet file')
parser.add_argument('--output-dir', type=str, default=None,
                    help='Directory to save output JSON (defaults to model-dir)')
args = parser.parse_args()

model_dir = Path(args.model_dir)
output_dir = Path(args.output_dir) if args.output_dir else model_dir

# ── Load model and data ─────────────────────────────────────────────────

print("Loading model...")
model = joblib.load(model_dir / 'xgboost_model_v3.pkl')

print("Loading feature columns...")
with open(model_dir / 'feature_columns_v3.txt') as f:
    feature_cols = [line.strip() for line in f if line.strip()]

print(f"  {len(feature_cols)} features")

print("Loading training data...")
df = pd.read_parquet(args.data)
print(f"  {len(df)} rows")

# Prepare target (same remapping as xgboost_v2.py)
df['target'] = df['ball_outcome'].copy()
df.loc[df['target'] == -1, 'target'] = 7  # Wicket
df = df[df['target'] <= 7]

# ── 1. XGBoost built-in importances ──────────────────────────────────────

print("\n" + "=" * 70)
print("1. XGBoost Feature Importances")
print("=" * 70)

booster = model.get_booster()

importance_types = {}
for imp_type in ['gain', 'weight', 'cover']:
    raw_scores = booster.get_score(importance_type=imp_type)
    # Keys may be feature names or f0/f1/... depending on how model was trained
    scores = {}
    for key, value in raw_scores.items():
        if key.startswith('f') and key[1:].isdigit():
            idx = int(key[1:])
            if idx < len(feature_cols):
                scores[feature_cols[idx]] = value
        else:
            scores[key] = value

    # Normalize to sum to 1.0
    total = sum(scores.values()) if scores else 1.0
    scores = {k: v / total for k, v in scores.items()}
    importance_types[imp_type] = scores

# Print top 20 by gain
gain_sorted = sorted(importance_types['gain'].items(), key=lambda x: -x[1])
print(f"\n{'Rank':<5} {'Feature':<30} {'Gain':>8} {'Weight':>8} {'Cover':>8}")
print("-" * 63)
for i, (feat, gain) in enumerate(gain_sorted[:20], 1):
    weight = importance_types['weight'].get(feat, 0.0)
    cover = importance_types['cover'].get(feat, 0.0)
    print(f"{i:<5} {feat:<30} {gain:>8.4f} {weight:>8.4f} {cover:>8.4f}")

# Features with zero importance
zero_features = [f for f in feature_cols if importance_types['gain'].get(f, 0.0) == 0.0]
if zero_features:
    print(f"\nFeatures with ZERO gain importance ({len(zero_features)}):")
    for f in zero_features:
        print(f"  - {f}")

# ── 2. Spearman correlation with target ──────────────────────────────────

print("\n" + "=" * 70)
print("2. Feature Correlation with Target (Spearman)")
print("=" * 70)

correlations = {}
available_cols = [c for c in feature_cols if c in df.columns]
target = df['target'].values

for col in available_cols:
    vals = df[col].values
    # Skip if constant
    if np.std(vals) == 0:
        correlations[col] = {'rho': 0.0, 'pvalue': 1.0}
        continue
    rho, pval = spearmanr(vals, target)
    correlations[col] = {'rho': float(rho), 'pvalue': float(pval)}

corr_sorted = sorted(correlations.items(), key=lambda x: -abs(x[1]['rho']))
print(f"\n{'Rank':<5} {'Feature':<30} {'|rho|':>8} {'rho':>8} {'p-value':>10}")
print("-" * 65)
for i, (feat, stats) in enumerate(corr_sorted[:20], 1):
    print(f"{i:<5} {feat:<30} {abs(stats['rho']):>8.4f} {stats['rho']:>8.4f} {stats['pvalue']:>10.2e}")

# ── 3. Feature-to-feature correlation (redundancy) ──────────────────────

print("\n" + "=" * 70)
print("3. Highly Correlated Feature Pairs (|r| > 0.8)")
print("=" * 70)

# Use only numeric features that exist in the dataframe
numeric_cols = [c for c in available_cols if df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
corr_matrix = df[numeric_cols].corr(method='pearson')

redundant_pairs = []
for i in range(len(numeric_cols)):
    for j in range(i + 1, len(numeric_cols)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.8:
            redundant_pairs.append((numeric_cols[i], numeric_cols[j], float(r)))

redundant_pairs.sort(key=lambda x: -abs(x[2]))

if redundant_pairs:
    print(f"\n{'Feature 1':<30} {'Feature 2':<30} {'r':>8}")
    print("-" * 70)
    for f1, f2, r in redundant_pairs:
        print(f"{f1:<30} {f2:<30} {r:>8.4f}")
else:
    print("\nNo highly correlated pairs found (|r| > 0.8)")

print(f"\nTotal redundant pairs: {len(redundant_pairs)}")

# ── 4. Per-group importance aggregation ──────────────────────────────────

print("\n" + "=" * 70)
print("4. Feature Group Importance (by Gain)")
print("=" * 70)

# Build reverse map: feature -> group
feature_to_group = {}
for group_name, features in FEATURE_GROUPS.items():
    for feat in features:
        feature_to_group[feat] = group_name

group_stats = {}
for group_name in FEATURE_GROUPS:
    group_features = FEATURE_GROUPS[group_name]
    gains = [importance_types['gain'].get(f, 0.0) for f in group_features]
    total_gain = sum(gains)
    n_features = len(group_features)
    n_nonzero = sum(1 for g in gains if g > 0)
    mean_gain = total_gain / n_features if n_features > 0 else 0.0

    # Mean absolute correlation with target
    group_corrs = [abs(correlations.get(f, {}).get('rho', 0.0)) for f in group_features]
    mean_corr = np.mean(group_corrs) if group_corrs else 0.0

    group_stats[group_name] = {
        'total_gain': total_gain,
        'n_features': n_features,
        'n_nonzero': n_nonzero,
        'mean_gain': mean_gain,
        'mean_abs_correlation': float(mean_corr),
    }

group_sorted = sorted(group_stats.items(), key=lambda x: -x[1]['total_gain'])

print(f"\n{'Rank':<5} {'Group':<20} {'Total Gain':>11} {'Features':>9} {'Active':>7} {'Mean Gain':>10} {'Mean |r|':>9}")
print("-" * 75)
for i, (group, stats) in enumerate(group_sorted, 1):
    print(f"{i:<5} {group:<20} {stats['total_gain']:>11.4f} {stats['n_features']:>9} "
          f"{stats['n_nonzero']:>7} {stats['mean_gain']:>10.4f} {stats['mean_abs_correlation']:>9.4f}")

# ── Save results ─────────────────────────────────────────────────────────

output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'feature_importance_analysis.json'

results = {
    'n_features': len(feature_cols),
    'n_training_rows': len(df),
    'importances': {
        imp_type: dict(sorted(scores.items(), key=lambda x: -x[1]))
        for imp_type, scores in importance_types.items()
    },
    'correlations': correlations,
    'redundant_pairs': [
        {'feature_1': f1, 'feature_2': f2, 'pearson_r': r}
        for f1, f2, r in redundant_pairs
    ],
    'group_stats': group_stats,
}

with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {output_path}")
