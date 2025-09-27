from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import joblib

# Load split data
print("Loading split datasets...")
train_df = pd.read_parquet('data/xgb_data/cricket_data_v2_train.parquet')
val_df = pd.read_parquet('data/xgb_data/cricket_data_v2_validation.parquet')
test_df = pd.read_parquet('data/xgb_data/cricket_data_v2_test.parquet')

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

# Feature engineering for player encoding (fit on train, transform others)
print("Encoding categorical variables...")

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

# Transform datasets
print("  Encoding training data...")
train_df['batter_encoded'] = le_batter.transform(train_df['batter_id'].astype(str))
train_df['bowler_encoded'] = le_bowler.transform(train_df['bowler_id'].astype(str))

print("  Encoding validation data...")
val_df['batter_encoded'] = le_batter.transform(val_df['batter_id'].astype(str))
val_df['bowler_encoded'] = le_bowler.transform(val_df['bowler_id'].astype(str))

print("  Encoding test data...")
test_df['batter_encoded'] = le_batter.transform(test_df['batter_id'].astype(str))
test_df['bowler_encoded'] = le_bowler.transform(test_df['bowler_id'].astype(str))

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

# Select ALL available features (use comprehensive feature set)
basic_features = [
    'inning_idx', 'score', 'wickets', 'balls_bowled', 'run_rate',
    'wickets_ratio', 'balls_ratio', 'wickets_in_hand',
    'is_powerplay', 'is_middle_overs', 'is_death_overs', 'balls_in_over'
]

player_features = [
    'batter_encoded', 'bowler_encoded', 'batsman_avg', 'batsman_sr', 
    'bowler_avg', 'bowler_econ'
]

h2h_features = ['h2h_avg', 'h2h_sr']

momentum_features = [
    'last_5_balls_runs', 'last_10_balls_runs', 'last_30_balls_runs',
    'balls_since_boundary', 'last_10_dots'
]

pressure_features = ['dot_percentage_recent', 'boundary_percentage_recent']

# Combine all features that exist in the data
all_potential_features = (basic_features + player_features + 
                         h2h_features + momentum_features + pressure_features)

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

# Train model with validation monitoring
print("\nTraining XGBoost model...")

model = XGBClassifier(
    n_estimators=200,  # Increased for more complex features
    max_depth=4,       # Deeper trees for more features
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,  # Feature sampling
    reg_alpha = 1.0,
    reg_lambda = 1.0,
    random_state=29,
    eval_metric='mlogloss',
    early_stopping_rounds=100,
    scale_pos_weight = None
)

sample_weights = np.array([weight_dict[y] for y in y_train])

# Fit with validation monitoring
model.fit(
    X_train, y_train,
    sample_weight= sample_weights,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=50  # Print every 50 rounds
)

# Validation evaluation
print("\n--- VALIDATION RESULTS ---")
y_val_pred = model.predict(X_val)
y_val_proba = model.predict_proba(X_val)

val_accuracy = accuracy_score(y_val, y_val_pred)
val_logloss = log_loss(y_val, y_val_proba)

print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Log Loss: {val_logloss:.4f}")

# Test evaluation  
print("\n--- TEST RESULTS ---")
y_test_pred = model.predict(X_test)
y_test_proba = model.predict_proba(X_test)

test_accuracy = accuracy_score(y_test, y_test_pred)
test_logloss = log_loss(y_test, y_test_proba)

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Log Loss: {test_logloss:.4f}")

print("\nClassification Report (Test):")
unique_classes = sorted(y_test.unique())
target_names = [f'{i}_runs' if i < 7 else 'wicket' for i in unique_classes]
print(classification_report(y_test, y_test_pred, labels=unique_classes, target_names=target_names))

# Feature importance
print(f"\nTop 15 Feature Importances:")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for _, row in feature_importance.head(15).iterrows():
    print(f"  {row['feature']:25s}: {row['importance']:.4f}")

# Save model and encoders
print("\nSaving model...")
from pathlib import Path
Path('models/xgb').mkdir(exist_ok=True)

joblib.dump(model, 'models/xgb/xgboost_model_v2.pkl')
joblib.dump(le_batter, 'models/xgb/batter_encoder_v2.pkl')
joblib.dump(le_bowler, 'models/xgb/bowler_encoder_v2.pkl')

# Save feature list for consistency
with open('models/xgb/feature_columns_v2.txt', 'w') as f:
    for feat in feature_cols:
        f.write(f"{feat}\n")

print("Training complete!")
print(f"Model saved as: models/xgb/gradient_boosting_model_v2.pkl")
print(f"Final validation log loss: {val_logloss:.4f}")
print(f"Final test log loss: {test_logloss:.4f}")