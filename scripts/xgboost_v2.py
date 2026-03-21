from pathlib import Path
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import joblib
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train XGBoost v3 model')
parser.add_argument('--tune', action='store_true', help='Run Optuna hyperparameter tuning (slow, ~30-60 min)')
parser.add_argument('--n-trials', type=int, default=50, help='Number of Optuna trials (default: 50)')
parser.add_argument('--config-json', type=str, default=None,
                    help='JSON config from experiment runner (overrides feature list and hyperparameters)')
args = parser.parse_args()

# Best hyperparameters from previous Optuna run (v2, trial 42)
# These will be used if --tune is not specified
DEFAULT_BEST_PARAMS = {
    'n_estimators': 444,
    'max_depth': 10,
    'learning_rate': 0.24036372383981375,
    'subsample': 0.8776663421127178,
    'colsample_bytree': 0.7424085095268674,
    'reg_alpha': 0.8503122682099661,
    'reg_lambda': 0.18186045420525845,
}

# Load split data
# v3: Data with player metadata features (Tier 1/2/3)
print("Loading split datasets (v3 with player metadata features)...")
train_df = pd.read_parquet('data/xgb_data_v3/cricket_data_v3_train.parquet')
val_df = pd.read_parquet('data/xgb_data_v3/cricket_data_v3_validation.parquet')
test_df = pd.read_parquet('data/xgb_data_v3/cricket_data_v3_test.parquet')

print(f"Train: {len(train_df)} balls")
print(f"Validation: {len(val_df)} balls") 
print(f"Test: {len(test_df)} balls")

# Convert wickets (-1) to class 7 for multi-class classification
for df in [train_df, val_df, test_df]:
    df['target'] = df['ball_outcome'].copy()
    df.loc[df['target'] == -1, 'target'] = 7  # Wicket class
    # Clean up invalid outcomes
    df = df[df['target'] <= 7]

print("Data preprocessing complete")

# NOTE: Chase features (run_rate_required, lead_gap, target) are now computed
# in parsing_v2.py during feature engineering, so no placeholder needed here.

# Feature engineering for player encoding (optimized for large datasets)
print("Encoding categorical variables...")

# Convert to string and get unique values more efficiently
print("  Processing batter IDs...")
unique_batters = pd.concat([
    train_df['batter_id'].astype(str),
    val_df['batter_id'].astype(str), 
    test_df['batter_id'].astype(str)
]).unique()

print("  Processing bowler IDs...")
unique_bowlers = pd.concat([
    train_df['bowler_id'].astype(str),
    val_df['bowler_id'].astype(str),
    test_df['bowler_id'].astype(str)
]).unique()

print(f"  Found {len(unique_batters)} unique batters, {len(unique_bowlers)} unique bowlers")

# Fit encoders on unique values (much faster)
le_batter = LabelEncoder()
le_bowler = LabelEncoder()
le_batter.fit(unique_batters)
le_bowler.fit(unique_bowlers)

print("  Processing venues...")
unique_venues = pd.concat([
    train_df['venue'].astype(str),
    val_df['venue'].astype(str),
    test_df['venue'].astype(str)
]).unique()

le_venue = LabelEncoder()
le_venue.fit(unique_venues)

# Transform datasets
print("  Encoding training data...")
train_df['batter_encoded'] = le_batter.transform(train_df['batter_id'].astype(str))
train_df['bowler_encoded'] = le_bowler.transform(train_df['bowler_id'].astype(str))
train_df['venue_encoded'] = le_venue.transform(train_df['venue'].astype(str))

print("  Encoding validation data...")
val_df['batter_encoded'] = le_batter.transform(val_df['batter_id'].astype(str))
val_df['bowler_encoded'] = le_bowler.transform(val_df['bowler_id'].astype(str))
val_df['venue_encoded'] = le_venue.transform(val_df['venue'].astype(str))

print("  Encoding test data...")
test_df['batter_encoded'] = le_batter.transform(test_df['batter_id'].astype(str))
test_df['bowler_encoded'] = le_bowler.transform(test_df['bowler_id'].astype(str))
test_df['venue_encoded'] = le_venue.transform(test_df['venue'].astype(str))

# NEW: Encode matchup_type if it exists
if 'matchup_type' in train_df.columns:
    print("  Processing matchup types...")
    unique_matchups = pd.concat([
        train_df['matchup_type'].astype(str),
        val_df['matchup_type'].astype(str),
        test_df['matchup_type'].astype(str)
    ]).unique()

    le_matchup = LabelEncoder()
    le_matchup.fit(unique_matchups)

    train_df['matchup_type_encoded'] = le_matchup.transform(train_df['matchup_type'].astype(str))
    val_df['matchup_type_encoded'] = le_matchup.transform(val_df['matchup_type'].astype(str))
    test_df['matchup_type_encoded'] = le_matchup.transform(test_df['matchup_type'].astype(str))
    print(f"  Found {len(unique_matchups)} unique matchup types")

print("Encoding complete!")

# # Transform validation and test (handle unknown players)
# def safe_transform(encoder, values):
#     """Transform with fallback for unknown categories"""
#     transformed = []
#     for val in values.astype(str):
#         try:
#             transformed.append(encoder.transform([val])[0])
#         except:
#             transformed.append(-1)  # Unknown player
#     return np.array(transformed)

# val_df['batter_encoded'] = safe_transform(le_batter, val_df['batter_id'])
# val_df['bowler_encoded'] = safe_transform(le_bowler, val_df['bowler_id'])
# test_df['batter_encoded'] = safe_transform(le_batter, test_df['batter_id'])  
# test_df['bowler_encoded'] = safe_transform(le_bowler, test_df['bowler_id'])

# Resolve feature list: from config-json or hardcoded defaults
if args.config_json:
    import json as _json
    _config = _json.loads(args.config_json)
    from feature_registry import resolve_feature_list
    all_potential_features = resolve_feature_list(
        _config['features']['groups'],
        _config['features'].get('exclude'),
        _config['features'].get('include_extra'),
    )
    # Override hyperparameters if provided
    _hp = _config.get('model', {}).get('hyperparameters', {})
    if _hp:
        DEFAULT_BEST_PARAMS.update(_hp)
    print(f"[config-json] Using {len(all_potential_features)} features from experiment config")
else:
    # Original hardcoded feature list (default behavior)
    basic_features = [
        'inning_idx', 'score', 'wickets', 'balls_bowled', 'run_rate',
        'wickets_ratio', 'balls_ratio', 'wickets_in_hand', 'balls_remaining',
        'is_powerplay', 'is_middle_overs', 'is_death_overs', 'balls_in_over',
        'venue_encoded', 'is_toss_winner', 'is_batting_first',
    ]

    player_features = [
        'batter_encoded', 'bowler_encoded', 'batsman_avg', 'batsman_sr',
        'bowler_avg', 'bowler_econ',
        'batsman_recent_avg', 'batsman_recent_sr',
        'bowler_recent_avg', 'bowler_recent_econ',
        'batter_balls_faced', 'batter_runs_scored',
        'bowler_balls_in_innings', 'bowler_overs_in_innings',
    ]

    h2h_features = ['h2h_avg', 'h2h_sr']

    momentum_features = [
        'last_5_balls_runs', 'last_10_balls_runs', 'last_30_balls_runs',
        'balls_since_boundary', 'last_10_dots',
        'partnership_runs',
    ]

    pressure_features = [
        'dot_percentage_recent', 'boundary_percentage_recent',
        'pressure_cooker_index',
    ]

    chase_features = [
        'chase_target', 'run_rate_required', 'lead_gap',
    ]

    medium_features = [
        'venue_avg_score',
        'non_striker_sr',
    ]

    player_metadata_features = [
        'batter_hand', 'bowler_arm', 'is_pace', 'bowling_type',
        'batter_age', 'bowler_age',
    ]

    matchup_features = [
        'spin_matchup_advantage', 'same_arm_matchup', 'matchup_type_encoded',
    ]

    type_based_features = [
        'batter_avg_vs_pace', 'batter_sr_vs_pace',
        'batter_avg_vs_spin', 'batter_sr_vs_spin',
        'bowler_avg_vs_lhb', 'bowler_econ_vs_lhb',
        'bowler_avg_vs_rhb', 'bowler_econ_vs_rhb',
    ]

    all_potential_features = (basic_features + player_features +
                             h2h_features + momentum_features + pressure_features +
                             chase_features + medium_features +
                             player_metadata_features + matchup_features + type_based_features)

# Only use features that actually exist in the dataframes
feature_cols = [col for col in all_potential_features if col in train_df.columns]

print(f"Using {len(feature_cols)} features:")
for i, feat in enumerate(feature_cols):
    print(f"  {i+1:2d}. {feat}")

# Prepare data
X_train = train_df[feature_cols]
y_train = train_df['target']
X_val = val_df[feature_cols]
y_val = val_df['target']
X_test = test_df[feature_cols]
y_test = test_df['target']

print("Cleaning invalid target values...")
print("Before cleaning:", sorted(y_train.unique()))

# Keep only valid cricket outcomes (0-7)
for df_name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
    before_len = len(df)
    df = df[df['target'] <= 7]
    after_len = len(df)
    print(f"  {df_name}: {before_len} -> {after_len} balls ({before_len-after_len} removed)")
    
    if df_name == 'train':
        train_df = df
    elif df_name == 'val':
        val_df = df  
    else:
        test_df = df

# Remap target classes to be consecutive after filtering
print("Remapping target classes to be consecutive...")

# Create mapping for remaining classes
class_mapping = {0: 0, 1: 1, 2: 2, 4: 3, 6: 4, 7: 5}
reverse_mapping = {v: k for k, v in class_mapping.items()}

print("Class mapping:", class_mapping)

# Apply mapping to all datasets
for df_name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
    df['target'] = df['target'].map(class_mapping)
    print(f"  {df_name} target range: {df['target'].min()} to {df['target'].max()}")
    
    if df_name == 'train':
        train_df = df
    elif df_name == 'val':
        val_df = df  
    else:
        test_df = df

# Redefine data after remapping
X_train = train_df[feature_cols]
y_train = train_df['target']
X_val = val_df[feature_cols]
y_val = val_df['target']
X_test = test_df[feature_cols]
y_test = test_df['target']

print(f"\nRemapped target distribution (train):")
print(y_train.value_counts().sort_index())


print(f"\nData shapes:")
print(f"  Train: X={X_train.shape}, y={y_train.shape}")
print(f"  Val:   X={X_val.shape}, y={y_val.shape}")  
print(f"  Test:  X={X_test.shape}, y={y_test.shape}")

print(f"\nTarget distribution (train):")
print(y_train.value_counts().sort_index())

print("Calculating class weights for imbalanced data...")
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
weight_dict = dict(zip(np.unique(y_train), class_weights))
print("Class weights:", {k: f"{v:.2f}" for k, v in weight_dict.items()})

# Optuna hyperparameter tuning (only if --tune flag is set)
if args.tune:
    import optuna

    # Define the objective function for Optuna
    def objective(trial):
        """
        Objective function that Optuna will optimize.
        This function should return the metric you want to minimize.
        """

        # Suggest hyperparameters for this trial
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
            'random_state': 29,
            'eval_metric': 'mlogloss',
            'early_stopping_rounds': 50,  # Reduced for faster trials
            'scale_pos_weight': None
        }

        # Create and train model with suggested parameters
        model = XGBClassifier(**params)

        # Calculate sample weights (same as your original code)
        sample_weights = np.array([weight_dict[y] for y in y_train])

        # Fit model with early stopping on validation set
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            verbose=0  # Silent training for cleaner output
        )

        # Get validation predictions and calculate log loss
        y_val_proba = model.predict_proba(X_val)
        val_logloss = log_loss(y_val, y_val_proba)

        # Return the metric to minimize (log loss)
        return val_logloss


    # Create and run the optimization study
    def run_optuna_optimization(n_trials=100):
        """
        Run Optuna hyperparameter optimization
        """
        print(f"\nStarting Optuna optimization with {n_trials} trials...")

        # Create study object
        study = optuna.create_study(
            direction='minimize',  # We want to minimize log loss
            study_name='xgboost_cricket_optimization',
            sampler=optuna.samplers.TPESampler(seed=29)  # Tree-structured Parzen Estimator
        )

        # Run optimization
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Print results
        print(f"\nOptimization completed!")
        print(f"Best validation log loss: {study.best_value:.4f}")
        print(f"Best parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

        return study

    # Run the optimization
    study = run_optuna_optimization(n_trials=args.n_trials)
    best_params = study.best_params.copy()

    # Save study for analysis
    Path('models/xgb_v3').mkdir(exist_ok=True)
    joblib.dump(study, 'models/xgb_v3/optuna_study_v3.pkl')
    print(f"Saved Optuna study to models/xgb_v3/optuna_study_v3.pkl")
else:
    # Use default best params from previous run
    print("\n--- USING SAVED HYPERPARAMETERS (use --tune to re-optimize) ---")
    print("Best hyperparameters (from v2 Optuna, trial 42):")
    for key, value in DEFAULT_BEST_PARAMS.items():
        print(f"  {key}: {value}")
    best_params = DEFAULT_BEST_PARAMS.copy()

# Train final model with best parameters
print("\n--- TRAINING FINAL MODEL WITH BEST PARAMETERS ---")
best_params.update({
    'random_state': 29,
    'eval_metric': 'mlogloss',
    'early_stopping_rounds': 100,  # Use longer early stopping for final model
    'scale_pos_weight': None
})

final_model = XGBClassifier(**best_params)
sample_weights = np.array([weight_dict[y] for y in y_train])

final_model.fit(
    X_train, y_train,
    sample_weight=sample_weights,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=50
)

# Evaluate final model
print("\n--- FINAL MODEL RESULTS ---")
y_val_pred = final_model.predict(X_val)
y_val_proba = final_model.predict_proba(X_val)
y_test_pred = final_model.predict(X_test)
y_test_proba = final_model.predict_proba(X_test)

val_accuracy = accuracy_score(y_val, y_val_pred)
val_logloss = log_loss(y_val, y_val_proba)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_logloss = log_loss(y_test, y_test_proba)

print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Log Loss: {val_logloss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Log Loss: {test_logloss:.4f}")

# Classification report with correct target names after remapping
# reverse_mapping: {0: 0 (dot), 1: 1, 2: 2, 3: 4, 4: 6, 5: wicket}
print("\nClassification Report (Test):")
unique_classes = sorted(y_test.unique())
target_names = []
for cls in unique_classes:
    original = reverse_mapping.get(cls, cls)
    if original == 7:
        target_names.append('wicket')
    else:
        target_names.append(f'{original}_runs')
print(classification_report(y_test, y_test_pred, labels=unique_classes, target_names=target_names))

# Feature importance
print(f"\nTop 15 Feature Importances:")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

for _, row in feature_importance.head(15).iterrows():
    print(f"  {row['feature']:25s}: {row['importance']:.4f}")

# Save model and encoders
# v3: Model with player metadata features (Tier 1/2/3)
print("\nSaving model (v3)...")
Path('models/xgb_v3').mkdir(exist_ok=True)

joblib.dump(final_model, 'models/xgb_v3/xgboost_model_v3.pkl')
joblib.dump(le_batter, 'models/xgb_v3/batter_encoder_v3.pkl')
joblib.dump(le_bowler, 'models/xgb_v3/bowler_encoder_v3.pkl')

# Save matchup encoder if it was created
if 'le_matchup' in dir():
    joblib.dump(le_matchup, 'models/xgb_v3/matchup_encoder_v3.pkl')
    print("  Saved matchup_encoder_v3.pkl")

# Save feature list for consistency
with open('models/xgb_v3/feature_columns_v3.txt', 'w') as f:
    for feat in feature_cols:
        f.write(f"{feat}\n")

print("Training complete!")
print(f"Model saved as: models/xgb_v3/xgboost_model_v3.pkl")
print(f"Final validation log loss: {val_logloss:.4f}")
print(f"Final test log loss: {test_logloss:.4f}")