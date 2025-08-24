import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data
print("Loading data...")
df = pd.read_parquet('cricket_data.parquet')
print(f"Loaded {len(df)} balls")

# Convert wickets (-1) to class 7 for multi-class classification
df['target'] = df['ball_outcome'].copy()
df.loc[df['target'] == -1, 'target'] = 7  # Wicket class

# Clean up invalid outcomes (>6 runs shouldn't exist in cricket)
df = df[df['target'] <= 7]  # Keep only valid outcomes (0-6 runs, wicket)

print(f"Cleaned data shape: {len(df)} balls")

# Basic feature engineering
print("Engineering features...")

# Encode categorical variables
le_batter = LabelEncoder()
le_bowler = LabelEncoder()

df['batter_encoded'] = le_batter.fit_transform(df['batter_id'])
df['bowler_encoded'] = le_bowler.fit_transform(df['bowler_id'])

# Create additional features
df['run_rate'] = df['score'] / (df['balls_bowled'] + 1)  # +1 to avoid division by zero
df['wickets_ratio'] = df['wickets'] / 10.0
df['balls_ratio'] = df['balls_bowled'] / 120.0  # T20 has 120 balls

# Clean up invalid outcomes (>6 runs shouldn't exist in cricket)
df = df[df['target'] <= 7]  # Keep only valid outcomes (0-6 runs, wicket)

print(f"Cleaned data shape: {len(df)} balls")

# Select features for training
feature_cols = [
    'inning_idx', 'score', 'wickets', 'balls_bowled',
    'batter_encoded', 'bowler_encoded', 'run_rate', 
    'wickets_ratio', 'balls_ratio'
]

X = df[feature_cols]
y = df['target']

print(f"Feature matrix shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts().sort_index()}")

# Temporal split (simulate real-world usage)
# Use first 80% of data for training, last 20% for testing
split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"\nTrain size: {len(X_train)}")
print(f"Test size: {len(X_test)}")

# Train Gradient Boosting model
print("\nTraining Gradient Boosting model...")

model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# Make predictions
print("Making predictions...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.3f}")

print("\nClassification Report:")
# Only use target names for classes that actually exist in the data
unique_classes = sorted(y_test.unique())
target_names = [f'{i}_runs' if i < 7 else 'wicket' for i in unique_classes]
print(classification_report(y_test, y_pred, labels=unique_classes, target_names=target_names))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
print("\nTop 10 Feature Importances:")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for _, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']}: {row['importance']:.3f}")

# Save model and encoders
print("\nSaving model...")
from pathlib import Path
Path('models').mkdir(exist_ok=True)

joblib.dump(model, 'models/gradient_boosting_model.pkl')
joblib.dump(le_batter, 'models/batter_encoder.pkl')
joblib.dump(le_bowler, 'models/bowler_encoder.pkl')

print("Gradient Boosting training complete!")

# Quick prediction example
print("\nExample prediction:")
sample_ball = X_test.iloc[0:1]
pred_proba = model.predict_proba(sample_ball)[0]
print(f"Sample features: {sample_ball.iloc[0].to_dict()}")
print("Probabilities:")
for i, prob in enumerate(pred_proba):
    outcome = f'{i}_runs' if i < 7 else 'wicket'
    print(f"  {outcome}: {prob:.3f}")