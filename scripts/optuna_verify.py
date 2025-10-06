"""
Quick Optuna verification script - tests that optimization is working
Uses small data subset and few trials for fast verification
"""

import optuna
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

print("=== OPTUNA VERIFICATION TEST ===")
print("Loading data...")

# Load your data (same as original)
train_df = pd.read_parquet('data/xgb_data/cricket_data_v2_train.parquet')
val_df = pd.read_parquet('data/xgb_data/cricket_data_v2_validation.parquet')

# QUICK TEST: Use only a small subset for fast verification
SAMPLE_SIZE = 10000  # Use 10k samples instead of full dataset
print(f"Using {SAMPLE_SIZE} samples for quick test...")

train_sample = train_df.sample(n=min(SAMPLE_SIZE, len(train_df)), random_state=42)
val_sample = val_df.sample(n=min(SAMPLE_SIZE//2, len(val_df)), random_state=42)

print(f"Train sample: {len(train_sample)} rows")
print(f"Val sample: {len(val_sample)} rows")

# Quick preprocessing (same logic as your original)
for df in [train_sample, val_sample]:
    df['target'] = df['ball_outcome'].copy()
    df.loc[df['target'] == -1, 'target'] = 7  # Wicket class

# Basic encoding (simplified)
print("Quick encoding...")
unique_batters = pd.concat([
    train_sample['batter_id'].astype(str),
    val_sample['batter_id'].astype(str)
]).unique()

unique_bowlers = pd.concat([
    train_sample['bowler_id'].astype(str),
    val_sample['bowler_id'].astype(str)
]).unique()

le_batter = LabelEncoder()
le_bowler = LabelEncoder()
le_batter.fit(unique_batters)
le_bowler.fit(unique_bowlers)

train_sample['batter_encoded'] = le_batter.transform(train_sample['batter_id'].astype(str))
train_sample['bowler_encoded'] = le_bowler.transform(train_sample['bowler_id'].astype(str))
val_sample['batter_encoded'] = le_batter.transform(val_sample['batter_id'].astype(str))
val_sample['bowler_encoded'] = le_bowler.transform(val_sample['bowler_id'].astype(str))

# Use basic features only for quick test
basic_features = [
    'inning_idx', 'score', 'wickets', 'balls_bowled', 'run_rate',
    'batter_encoded', 'bowler_encoded'
]

# Filter to only existing features
feature_cols = [col for col in basic_features if col in train_sample.columns]
print(f"Using {len(feature_cols)} features: {feature_cols}")

# Clean and remap targets (same logic as original script)
print("Cleaning target values...")
print("Before cleaning - Train targets:", sorted(train_sample['target'].unique()))
print("Before cleaning - Val targets:", sorted(val_sample['target'].unique()))

# Filter out invalid targets (keep only 0-7)
train_sample = train_sample[train_sample['target'] <= 7].copy()
val_sample = val_sample[val_sample['target'] <= 7].copy()

print("After filtering - Train targets:", sorted(train_sample['target'].unique()))
print("After filtering - Val targets:", sorted(val_sample['target'].unique()))

# Create mapping for remaining classes to be consecutive
class_mapping = {0: 0, 1: 1, 2: 2, 4: 3, 6: 4, 7: 5}
reverse_mapping = {v: k for k, v in class_mapping.items()}

print("Class mapping:", class_mapping)

# Apply mapping
train_sample['target'] = train_sample['target'].map(class_mapping)
val_sample['target'] = val_sample['target'].map(class_mapping)

# Remove any rows with NaN targets (unmapped values)
train_sample = train_sample.dropna(subset=['target']).copy()
val_sample = val_sample.dropna(subset=['target']).copy()

print("After mapping - Train targets:", sorted(train_sample['target'].unique()))
print("After mapping - Val targets:", sorted(val_sample['target'].unique()))

# Prepare data
X_train = train_sample[feature_cols].fillna(0)
y_train = train_sample['target'].astype(int)  # Ensure integer type
X_val = val_sample[feature_cols].fillna(0)
y_val = val_sample['target'].astype(int)

print(f"Final data shapes: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Target classes: {sorted(y_train.unique())}")
print(f"Target value counts:\n{y_train.value_counts().sort_index()}")

# Calculate class weights (ensure we have the right classes)
unique_classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
weight_dict = dict(zip(unique_classes, class_weights))
sample_weights = np.array([weight_dict[y] for y in y_train])

print(f"Class weights: {weight_dict}")

print("\n=== STARTING OPTUNA TEST ===")

def objective(trial):
    """Simplified objective function for testing"""
    print(f"Trial {trial.number}: Testing hyperparameters...")
    
    # Suggest fewer parameters for quick test
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 150),  # Smaller range
        'max_depth': trial.suggest_int('max_depth', 3, 6),           # Smaller range
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
        'random_state': 42,
        'eval_metric': 'mlogloss',
        'early_stopping_rounds': 20,  # Faster stopping
        'verbosity': 0  # Silent
    }
    
    try:
        # Verify data shapes before training
        print(f"  Data check: X_train={X_train.shape}, y_train={y_train.shape}")
        print(f"  Target classes: {sorted(y_train.unique())}")
        
        # Create and train model
        model = XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            verbose=0
        )
        
        # Calculate validation loss
        y_val_proba = model.predict_proba(X_val)
        val_loss = log_loss(y_val, y_val_proba)
        
        print(f"  Trial {trial.number} completed: val_loss = {val_loss:.4f}")
        return val_loss
        
    except Exception as e:
        print(f"  Trial {trial.number} failed: {e}")
        print(f"  Error type: {type(e).__name__}")
        return float('inf')  # Return bad score for failed trials

# Run quick optimization test
print("Creating study...")
study = optuna.create_study(
    direction='minimize',
    study_name='quick_test',
    sampler=optuna.samplers.TPESampler(seed=42)
)

print("Running 5 quick trials...")
study.optimize(objective, n_trials=5, show_progress_bar=True)

# Show results
print("\n=== TEST RESULTS ===")
print(f"✅ Optuna completed successfully!")
print(f"Best validation loss: {study.best_value:.4f}")
print(f"Best parameters found:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

print(f"\nAll trials:")
for trial in study.trials:
    status = "✅ COMPLETE" if trial.state == optuna.trial.TrialState.COMPLETE else "❌ FAILED"
    value = f"{trial.value:.4f}" if trial.value else "N/A"
    print(f"  Trial {trial.number}: {status} - Loss: {value}")

print("\n🎉 Optuna verification successful!")
print("You can now run the full optimization with more trials and complete dataset.")

# Test a quick prediction with best model
print("\n=== TESTING BEST MODEL ===")
best_model = XGBClassifier(**study.best_params, random_state=42, verbosity=0)
best_model.fit(X_train, y_train, sample_weight=sample_weights)

y_pred = best_model.predict(X_val)
y_proba = best_model.predict_proba(X_val)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_val, y_pred)
final_loss = log_loss(y_val, y_proba)

print(f"Best model validation accuracy: {accuracy:.4f}")
print(f"Best model validation loss: {final_loss:.4f}")
print("\nVerification complete! ✅")