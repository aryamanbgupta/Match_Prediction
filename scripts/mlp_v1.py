#!/usr/bin/env python3
"""
MLP Model Training for Cricket Ball Prediction

Simple Multi-Layer Perceptron baseline for ball-by-ball prediction.
Uses same features as XGBoost v3 for fair comparison.

Usage:
    python scripts/mlp_v1.py                    # Full training
    python scripts/mlp_v1.py --quick            # Quick test (5% data, 5 epochs)
    python scripts/mlp_v1.py --epochs 50        # Custom epochs
"""

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import joblib
from tqdm import tqdm


# ============================================================================
# FOCAL LOSS - Better for class imbalance
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification.
    
    Reduces loss for well-classified examples, focusing on hard examples.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, 
            weight=self.alpha, 
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ============================================================================
# DATASET
# ============================================================================

class CricketMLPDataset(Dataset):
    """Dataset for MLP training - simple tabular features."""
    
    def __init__(self, df: pd.DataFrame, feature_cols: list, 
                 scaler: StandardScaler = None, fit_scaler: bool = False):
        self.feature_cols = feature_cols
        
        # Get features
        X = df[feature_cols].values.astype(np.float32)
        
        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        if fit_scaler:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        elif scaler is not None:
            self.scaler = scaler
            X = self.scaler.transform(X)
        else:
            self.scaler = None
        
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(df['target'].values)
        
        print(f"  Dataset: {len(self.X)} samples, {len(feature_cols)} features")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================================
# MODEL
# ============================================================================

class MLPBallPredictor(nn.Module):
    """Simple MLP for ball outcome prediction.
    
    Architecture:
        Input → 256 → 128 → 64 → 6 classes
        With BatchNorm, ReLU, and Dropout
    """
    
    def __init__(self, n_features: int, hidden_sizes: list = [256, 128, 64],
                 dropout: float = 0.3, n_classes: int = 6):
        super().__init__()
        
        self.n_features = n_features
        self.hidden_sizes = hidden_sizes
        self.n_classes = n_classes
        
        layers = []
        in_size = n_features
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            # Less dropout in later layers
            drop_rate = dropout if i < len(hidden_sizes) - 1 else dropout * 0.67
            layers.append(nn.Dropout(drop_rate))
            in_size = hidden_size
        
        self.feature_layers = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_sizes[-1], n_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.feature_layers(x)
        return self.classifier(features)


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device, grad_clip=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X, y in tqdm(loader, desc="Training", leave=False):
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        total_loss += loss.item() * X.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(y).sum().item()
        total += y.size(0)
    
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            
            total_loss += loss.item() * X.size(0)
            _, predicted = logits.max(1)
            correct += predicted.eq(y).sum().item()
            total += y.size(0)
    
    return total_loss / total, correct / total


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train MLP for cricket ball prediction')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--quick', action='store_true', help='Quick test with 5% data')
    parser.add_argument('--device', type=str, default='auto', help='Device (cpu/cuda/mps/auto)')
    args = parser.parse_args()
    
    # Device selection
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load data
    print("\n--- Loading Data ---")
    train_df = pd.read_parquet('data/xgb_data_v3/cricket_data_v3_train.parquet')
    val_df = pd.read_parquet('data/xgb_data_v3/cricket_data_v3_validation.parquet')
    test_df = pd.read_parquet('data/xgb_data_v3/cricket_data_v3_test.parquet')
    
    print(f"Train: {len(train_df)} samples")
    print(f"Val: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    # Quick mode - sample 5% of data
    if args.quick:
        print("\n[QUICK MODE] Using 5% of data")
        train_df = train_df.sample(frac=0.05, random_state=42)
        val_df = val_df.sample(frac=0.05, random_state=42)
        args.epochs = 5
    
    # Remap target classes (same as XGBoost)
    print("\n--- Preparing Targets ---")
    class_mapping = {0: 0, 1: 1, 2: 2, 4: 3, 6: 4, 7: 5}
    
    for df in [train_df, val_df, test_df]:
        # Convert wickets (-1) to class 7
        df.loc[df['ball_outcome'] == -1, 'ball_outcome'] = 7
        # Remap to consecutive classes
        df['target'] = df['ball_outcome'].map(class_mapping)
    
    # Remove any unmapped classes
    train_df = train_df[train_df['target'].notna()]
    val_df = val_df[val_df['target'].notna()]
    test_df = test_df[test_df['target'].notna()]
    train_df['target'] = train_df['target'].astype(int)
    val_df['target'] = val_df['target'].astype(int)
    test_df['target'] = test_df['target'].astype(int)
    
    print(f"Class distribution:\n{train_df['target'].value_counts().sort_index()}")
    
    # Define feature columns (same as XGBoost v3)
    print("\n--- Preparing Features ---")
    basic_features = [
        'inning_idx', 'score', 'wickets', 'balls_bowled', 'run_rate',
        'wickets_ratio', 'balls_ratio', 'wickets_in_hand', 'balls_remaining',
        'is_powerplay', 'is_middle_overs', 'is_death_overs', 'balls_in_over',
    ]
    
    player_features = [
        'batter_encoded', 'bowler_encoded',
        'batsman_avg', 'batsman_sr', 'bowler_avg', 'bowler_econ',
        'batter_runs_scored', 'batter_balls_faced',
        'bowler_balls_in_innings', 'bowler_overs_in_innings',
    ]
    
    h2h_features = ['h2h_avg', 'h2h_sr']
    
    momentum_features = [
        'last_5_balls_runs', 'last_10_balls_runs', 'last_30_balls_runs',
        'balls_since_boundary', 'last_10_dots',
    ]
    
    pressure_features = ['dot_percentage_recent', 'boundary_percentage_recent']
    chase_features = ['chase_target', 'run_rate_required', 'lead_gap']
    medium_features = ['venue_avg_score', 'non_striker_sr', 'partnership_runs']
    
    player_metadata_features = [
        'batter_hand', 'bowler_arm', 'is_pace', 'bowling_type',
        'batter_age', 'bowler_age',
    ]
    
    matchup_features = ['matchup_type_encoded', 'spin_matchup_advantage', 'same_arm_matchup']
    
    type_based_features = [
        'batter_avg_vs_pace', 'batter_sr_vs_pace',
        'batter_avg_vs_spin', 'batter_sr_vs_spin',
        'bowler_avg_vs_lhb', 'bowler_econ_vs_lhb',
        'bowler_avg_vs_rhb', 'bowler_econ_vs_rhb',
    ]
    
    advanced_features = ['pressure_cooker_index']
    
    all_features = (basic_features + player_features + h2h_features + 
                   momentum_features + pressure_features + chase_features +
                   medium_features + player_metadata_features + matchup_features +
                   type_based_features + advanced_features)
    
    # Filter to features that exist in data
    feature_cols = [f for f in all_features if f in train_df.columns]
    print(f"Using {len(feature_cols)} features")
    
    # Fill NaN values
    for df in [train_df, val_df, test_df]:
        df[feature_cols] = df[feature_cols].fillna(0)
    
    # Create datasets
    print("\n--- Creating Datasets ---")
    train_dataset = CricketMLPDataset(train_df, feature_cols, fit_scaler=True)
    val_dataset = CricketMLPDataset(val_df, feature_cols, scaler=train_dataset.scaler)
    test_dataset = CricketMLPDataset(test_df, feature_cols, scaler=train_dataset.scaler)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    
    # Class weights for loss
    class_weights = compute_class_weight('balanced', classes=np.arange(6), 
                                         y=train_df['target'].values)
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"Class weights: {class_weights.cpu().numpy().round(2)}")
    
    # Create model
    print("\n--- Creating Model ---")
    model = MLPBallPredictor(
        n_features=len(feature_cols),
        hidden_sizes=[256, 128, 64],
        dropout=0.3,
        n_classes=6
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    print("\n--- Training ---")
    best_val_acc = 0
    best_epoch = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | LR: {lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save checkpoint
            Path('models/mlp_v1').mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), 'models/mlp_v1/mlp_model_v1.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    print(f"\nBest validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    
    # Load best model and evaluate on test
    print("\n--- Test Evaluation ---")
    model.load_state_dict(torch.load('models/mlp_v1/mlp_model_v1.pt', map_location=device, weights_only=True))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Save artifacts
    print("\n--- Saving Artifacts ---")
    Path('models/mlp_v1').mkdir(parents=True, exist_ok=True)
    
    # Save scaler
    joblib.dump(train_dataset.scaler, 'models/mlp_v1/feature_scaler_v1.pkl')
    
    # Copy encoders from XGBoost (they use the same encoding)
    import shutil
    for encoder_name in ['batter_encoder_v3.pkl', 'bowler_encoder_v3.pkl']:
        src = f'models/xgb_v3/{encoder_name}'
        dst = f'models/mlp_v1/{encoder_name.replace("v3", "v1")}'
        if Path(src).exists():
            shutil.copy(src, dst)
            print(f"  Copied {encoder_name}")
    
    # Save feature columns
    with open('models/mlp_v1/feature_columns_v1.txt', 'w') as f:
        for col in feature_cols:
            f.write(f"{col}\n")
    
    # Save config
    config = {
        'n_features': len(feature_cols),
        'hidden_sizes': [256, 128, 64],
        'dropout': 0.3,
        'n_classes': 6,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
    }
    with open('models/mlp_v1/mlp_config_v1.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\nArtifacts saved to models/mlp_v1/")
    print("Training complete!")


if __name__ == "__main__":
    main()
