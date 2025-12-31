#!/usr/bin/env python3
"""
MLP Model V2 Training for Cricket Ball Prediction

Enhanced MLP with:
- Embedding layers for batter, bowler, venue, and matchup type
- Same continuous features as XGBoost v3
- Larger model capacity

Usage:
    uv run python scripts/mlp_v2.py                    # Full training
    uv run python scripts/mlp_v2.py --quick            # Quick test (5% data, 5 epochs)
    uv run python scripts/mlp_v2.py --epochs 50        # Custom epochs
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
# FOCAL LOSS
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification."""
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
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
# DATASET WITH EMBEDDINGS
# ============================================================================

class CricketMLPDatasetV2(Dataset):
    """Dataset for MLP v2 with separate handling of categorical and continuous features."""
    
    def __init__(self, df: pd.DataFrame, continuous_cols: list, categorical_cols: dict,
                 scaler: StandardScaler = None, fit_scaler: bool = False):
        """
        Args:
            df: DataFrame with ball-by-ball data
            continuous_cols: List of continuous feature column names
            categorical_cols: Dict of {col_name: vocab_size} for embedding columns
            scaler: StandardScaler for continuous features
            fit_scaler: Whether to fit the scaler on this data
        """
        self.continuous_cols = continuous_cols
        self.categorical_cols = categorical_cols
        
        # Process continuous features
        X_cont = df[continuous_cols].values.astype(np.float32)
        X_cont = np.nan_to_num(X_cont, nan=0.0, posinf=0.0, neginf=0.0)
        
        if fit_scaler:
            self.scaler = StandardScaler()
            X_cont = self.scaler.fit_transform(X_cont)
        elif scaler is not None:
            self.scaler = scaler
            X_cont = self.scaler.transform(X_cont)
        else:
            self.scaler = None
        
        self.X_continuous = torch.FloatTensor(X_cont)
        
        # Process categorical features (for embeddings)
        self.X_categorical = {}
        for col in categorical_cols.keys():
            # Clamp to valid range [0, vocab_size-1]
            values = df[col].values.astype(np.int64)
            values = np.clip(values, 0, categorical_cols[col] - 1)
            self.X_categorical[col] = torch.LongTensor(values)
        
        self.y = torch.LongTensor(df['target'].values)
        
        print(f"  Dataset: {len(self.X_continuous)} samples")
        print(f"    Continuous features: {len(continuous_cols)}")
        print(f"    Categorical features: {list(categorical_cols.keys())}")
    
    def __len__(self):
        return len(self.X_continuous)
    
    def __getitem__(self, idx):
        cat_features = {k: v[idx] for k, v in self.X_categorical.items()}
        return self.X_continuous[idx], cat_features, self.y[idx]


def collate_fn(batch):
    """Custom collate function to handle dict of categorical features."""
    continuous = torch.stack([item[0] for item in batch])
    categorical = {k: torch.stack([item[1][k] for item in batch]) for k in batch[0][1].keys()}
    targets = torch.stack([item[2] for item in batch])
    return continuous, categorical, targets


# ============================================================================
# MODEL WITH EMBEDDINGS
# ============================================================================

class MLPBallPredictorV2(nn.Module):
    """MLP with embedding layers for categorical features.
    
    Architecture:
        Categorical inputs → Embeddings → Concat with continuous → MLP layers
    """
    
    def __init__(self, 
                 n_continuous: int,
                 n_batters: int = 8000,
                 n_bowlers: int = 6000,
                 n_venues: int = 500,
                 n_matchups: int = 50,
                 embed_dim_player: int = 16,  # Reduced from 32 to prevent overfitting
                 embed_dim_venue: int = 8,    # Reduced from 16
                 embed_dim_matchup: int = 4,  # Reduced from 8
                 hidden_sizes: list = [256, 128, 64],  # Reduced from [512, 256, 128]
                 dropout: float = 0.4,        # Increased from 0.3
                 embed_dropout: float = 0.3,  # New: dropout on embeddings
                 n_classes: int = 6):
        super().__init__()
        
        self.n_continuous = n_continuous
        self.embed_dropout = nn.Dropout(embed_dropout)
        
        # Embedding layers with smaller dimensions
        self.batter_embed = nn.Embedding(n_batters, embed_dim_player)
        self.bowler_embed = nn.Embedding(n_bowlers, embed_dim_player)
        self.venue_embed = nn.Embedding(n_venues, embed_dim_venue)
        self.matchup_embed = nn.Embedding(n_matchups, embed_dim_matchup)
        
        # Total input size after concatenation
        total_embed_dim = 2 * embed_dim_player + embed_dim_venue + embed_dim_matchup
        input_size = n_continuous + total_embed_dim
        
        # MLP layers (smaller to reduce overfitting)
        layers = []
        in_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.GELU())
            drop_rate = dropout if i < len(hidden_sizes) - 1 else dropout * 0.5
            layers.append(nn.Dropout(drop_rate))
            in_size = hidden_size
        
        self.feature_layers = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_sizes[-1], n_classes)
        
        # Store config for saving
        self.config = {
            'n_continuous': n_continuous,
            'n_batters': n_batters,
            'n_bowlers': n_bowlers,
            'n_venues': n_venues,
            'n_matchups': n_matchups,
            'embed_dim_player': embed_dim_player,
            'embed_dim_venue': embed_dim_venue,
            'embed_dim_matchup': embed_dim_matchup,
            'hidden_sizes': hidden_sizes,
            'dropout': dropout,
            'embed_dropout': embed_dropout,
            'n_classes': n_classes,
        }
        
        self._init_weights()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        embed_params = sum(p.numel() for p in [self.batter_embed.weight, self.bowler_embed.weight,
                                                self.venue_embed.weight, self.matchup_embed.weight])
        print(f"  Model parameters: {total_params:,} (embeddings: {embed_params:,})")
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
    
    def forward(self, continuous, categorical):
        """
        Args:
            continuous: (batch, n_continuous) - continuous features
            categorical: dict of {col_name: (batch,) tensor} - categorical features
        """
        # Get embeddings and apply dropout to prevent overfitting
        batter_emb = self.embed_dropout(self.batter_embed(categorical['batter_encoded']))
        bowler_emb = self.embed_dropout(self.bowler_embed(categorical['bowler_encoded']))
        venue_emb = self.embed_dropout(self.venue_embed(categorical['venue_encoded']))
        matchup_emb = self.embed_dropout(self.matchup_embed(categorical['matchup_type_encoded']))
        
        # Concatenate all features
        x = torch.cat([continuous, batter_emb, bowler_emb, venue_emb, matchup_emb], dim=-1)
        
        # MLP forward
        features = self.feature_layers(x)
        return self.classifier(features)


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device, grad_clip=1.0):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for continuous, categorical, targets in tqdm(loader, desc="Training", leave=False):
        continuous = continuous.to(device)
        categorical = {k: v.to(device) for k, v in categorical.items()}
        targets = targets.to(device)
        
        optimizer.zero_grad()
        logits = model(continuous, categorical)
        loss = criterion(logits, targets)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        total_loss += loss.item() * continuous.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
    
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for continuous, categorical, targets in loader:
            continuous = continuous.to(device)
            categorical = {k: v.to(device) for k, v in categorical.items()}
            targets = targets.to(device)
            
            logits = model(continuous, categorical)
            loss = criterion(logits, targets)
            
            total_loss += loss.item() * continuous.size(0)
            _, predicted = logits.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    
    return total_loss / total, correct / total


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train MLP v2 with embeddings')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--quick', action='store_true', help='Quick test with 5% data')
    parser.add_argument('--device', type=str, default='auto', help='Device (cpu/cuda/mps/auto)')
    parser.add_argument('--embed-dim', type=int, default=32, help='Player embedding dimension')
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
    
    # Quick mode
    if args.quick:
        print("\n[QUICK MODE] Using 5% of data")
        train_df = train_df.sample(frac=0.05, random_state=42)
        val_df = val_df.sample(frac=0.05, random_state=42)
        args.epochs = 5
    
    # Prepare targets
    print("\n--- Preparing Targets ---")
    class_mapping = {0: 0, 1: 1, 2: 2, 4: 3, 6: 4, 7: 5}
    
    for df in [train_df, val_df, test_df]:
        df.loc[df['ball_outcome'] == -1, 'ball_outcome'] = 7
        df['target'] = df['ball_outcome'].map(class_mapping)
    
    train_df = train_df[train_df['target'].notna()]
    val_df = val_df[val_df['target'].notna()]
    test_df = test_df[test_df['target'].notna()]
    train_df['target'] = train_df['target'].astype(int)
    val_df['target'] = val_df['target'].astype(int)
    test_df['target'] = test_df['target'].astype(int)
    
    print(f"Class distribution:\n{train_df['target'].value_counts().sort_index()}")
    
    # Define feature columns (same as XGBoost v3, but separate continuous and categorical)
    print("\n--- Preparing Features ---")
    
    # Continuous features (will be scaled)
    continuous_features = [
        # Basic state
        'inning_idx', 'score', 'wickets', 'balls_bowled', 'run_rate',
        'wickets_ratio', 'balls_ratio', 'wickets_in_hand', 'balls_remaining',
        'is_powerplay', 'is_middle_overs', 'is_death_overs', 'balls_in_over',
        # Player stats from StatsProvider (same as XGBoost)
        'batsman_avg', 'batsman_sr', 'bowler_avg', 'bowler_econ',
        'batter_runs_scored', 'batter_balls_faced',
        'bowler_balls_in_innings', 'bowler_overs_in_innings',
        # H2H stats
        'h2h_avg', 'h2h_sr',
        # Momentum
        'last_5_balls_runs', 'last_10_balls_runs', 'last_30_balls_runs',
        'balls_since_boundary', 'last_10_dots',
        # Pressure
        'dot_percentage_recent', 'boundary_percentage_recent',
        # Chase
        'chase_target', 'run_rate_required', 'lead_gap',
        # Medium
        'venue_avg_score', 'non_striker_sr', 'partnership_runs',
        # Player metadata
        'batter_hand', 'bowler_arm', 'is_pace', 'bowling_type',
        'batter_age', 'bowler_age',
        # Matchup features
        'spin_matchup_advantage', 'same_arm_matchup',
        # Type-based stats
        'batter_avg_vs_pace', 'batter_sr_vs_pace',
        'batter_avg_vs_spin', 'batter_sr_vs_spin',
        'bowler_avg_vs_lhb', 'bowler_econ_vs_lhb',
        'bowler_avg_vs_rhb', 'bowler_econ_vs_rhb',
        # Advanced
        'pressure_cooker_index',
    ]
    
    # Create encoders for categorical features from raw IDs
    print("\n--- Encoding Categorical Features ---")
    
    # Combine all data for fitting encoders
    all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # Batter encoder
    batter_encoder = LabelEncoder()
    all_batters = all_data['batter_id'].astype(str).unique()
    batter_encoder.fit(list(all_batters) + ['unknown'])
    for df in [train_df, val_df, test_df]:
        df['batter_encoded'] = batter_encoder.transform(df['batter_id'].astype(str).fillna('unknown'))
    n_batters = len(batter_encoder.classes_)
    print(f"  Batters: {n_batters} unique")
    
    # Bowler encoder
    bowler_encoder = LabelEncoder()
    all_bowlers = all_data['bowler_id'].astype(str).unique()
    bowler_encoder.fit(list(all_bowlers) + ['unknown'])
    for df in [train_df, val_df, test_df]:
        df['bowler_encoded'] = bowler_encoder.transform(df['bowler_id'].astype(str).fillna('unknown'))
    n_bowlers = len(bowler_encoder.classes_)
    print(f"  Bowlers: {n_bowlers} unique")
    
    # Venue encoder
    venue_encoder = LabelEncoder()
    if 'venue' in all_data.columns:
        all_venues = all_data['venue'].astype(str).unique()
        venue_encoder.fit(list(all_venues) + ['unknown'])
        for df in [train_df, val_df, test_df]:
            df['venue_encoded'] = venue_encoder.transform(df['venue'].astype(str).fillna('unknown'))
        n_venues = len(venue_encoder.classes_)
    else:
        for df in [train_df, val_df, test_df]:
            df['venue_encoded'] = 0
        n_venues = 1
    print(f"  Venues: {n_venues} unique")
    
    # Matchup type encoder
    matchup_encoder = LabelEncoder()
    if 'matchup_type' in all_data.columns:
        all_matchups = all_data['matchup_type'].astype(str).unique()
        matchup_encoder.fit(list(all_matchups) + ['unknown'])
        for df in [train_df, val_df, test_df]:
            df['matchup_type_encoded'] = matchup_encoder.transform(df['matchup_type'].astype(str).fillna('unknown'))
        n_matchups = len(matchup_encoder.classes_)
    else:
        for df in [train_df, val_df, test_df]:
            df['matchup_type_encoded'] = 0
        n_matchups = 1
    print(f"  Matchup types: {n_matchups} unique")
    
    categorical_cols = {
        'batter_encoded': int(n_batters),
        'bowler_encoded': int(n_bowlers),
        'venue_encoded': int(n_venues),
        'matchup_type_encoded': int(n_matchups),
    }
    
    print(f"Vocab sizes: batters={n_batters}, bowlers={n_bowlers}, venues={n_venues}, matchups={n_matchups}")
    
    # Filter to features that exist
    continuous_cols = [f for f in continuous_features if f in train_df.columns]
    print(f"Using {len(continuous_cols)} continuous features")
    
    # Fill NaN values
    for df in [train_df, val_df, test_df]:
        df[continuous_cols] = df[continuous_cols].fillna(0)
        for col in categorical_cols.keys():
            df[col] = df[col].fillna(0).astype(int)
    
    # Store encoders for later saving
    encoders = {
        'batter_encoder': batter_encoder,
        'bowler_encoder': bowler_encoder,
        'venue_encoder': venue_encoder,
        'matchup_encoder': matchup_encoder,
    }
    
    # Create datasets
    print("\n--- Creating Datasets ---")
    train_dataset = CricketMLPDatasetV2(train_df, continuous_cols, categorical_cols, fit_scaler=True)
    val_dataset = CricketMLPDatasetV2(val_df, continuous_cols, categorical_cols, scaler=train_dataset.scaler)
    test_dataset = CricketMLPDatasetV2(test_df, continuous_cols, categorical_cols, scaler=train_dataset.scaler)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=0, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=0, pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True, collate_fn=collate_fn)
    
    # Class weights
    class_weights = compute_class_weight('balanced', classes=np.arange(6), 
                                         y=train_df['target'].values)
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"Class weights: {class_weights.cpu().numpy().round(2)}")
    
    # Create model with regularization to prevent overfitting
    print("\n--- Creating Model ---")
    model = MLPBallPredictorV2(
        n_continuous=len(continuous_cols),
        n_batters=int(n_batters),
        n_bowlers=int(n_bowlers),
        n_venues=int(n_venues),
        n_matchups=int(n_matchups),
        embed_dim_player=16,      # Smaller embeddings
        embed_dim_venue=8,
        embed_dim_matchup=4,
        hidden_sizes=[256, 128, 64],  # Smaller network
        dropout=0.4,
        embed_dropout=0.3,        # Dropout on embeddings
        n_classes=6
    ).to(device)
    
    # Loss and optimizer with stronger weight decay
    criterion = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-3)  # Start with lower LR
    
    # Use ReduceLROnPlateau - reduce LR when val loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6, verbose=True
    )
    
    # Warmup settings
    warmup_epochs = 3
    warmup_factor = 10  # Ramp up from lr*0.1 to lr over warmup_epochs
    
    # Training loop
    print("\n--- Training ---")
    best_val_loss = float('inf')  # Monitor loss instead of accuracy for early stopping
    best_val_acc = 0
    best_epoch = 0
    patience = 8  # Reduced from 10 since we're using ReduceLROnPlateau
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Learning rate warmup
        if epoch < warmup_epochs:
            warmup_lr = args.lr * 0.1 + (args.lr * 0.9) * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Step scheduler after warmup
        if epoch >= warmup_epochs:
            scheduler.step(val_loss)
        
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | LR: {lr:.6f}")
        
        # Track best based on validation LOSS (more stable than accuracy)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            
            Path('models/mlp_v2').mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), 'models/mlp_v2/mlp_model_v2.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    print(f"\nBest validation loss: {best_val_loss:.4f} (acc: {best_val_acc:.4f}) at epoch {best_epoch}")
    
    # Test evaluation
    print("\n--- Test Evaluation ---")
    model.load_state_dict(torch.load('models/mlp_v2/mlp_model_v2.pt', map_location=device, weights_only=True))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Save artifacts
    print("\n--- Saving Artifacts ---")
    Path('models/mlp_v2').mkdir(parents=True, exist_ok=True)
    
    # Save scaler
    joblib.dump(train_dataset.scaler, 'models/mlp_v2/feature_scaler_v2.pkl')
    
    # Save encoders
    joblib.dump(encoders['batter_encoder'], 'models/mlp_v2/batter_encoder_v2.pkl')
    joblib.dump(encoders['bowler_encoder'], 'models/mlp_v2/bowler_encoder_v2.pkl')
    joblib.dump(encoders['venue_encoder'], 'models/mlp_v2/venue_encoder_v2.pkl')
    joblib.dump(encoders['matchup_encoder'], 'models/mlp_v2/matchup_encoder_v2.pkl')
    print("  Saved all encoders")
    
    # Save feature columns
    with open('models/mlp_v2/continuous_columns_v2.txt', 'w') as f:
        for col in continuous_cols:
            f.write(f"{col}\n")
    
    with open('models/mlp_v2/categorical_columns_v2.json', 'w') as f:
        json.dump(categorical_cols, f, indent=2)
    
    # Save config
    config = model.config.copy()
    config['best_val_acc'] = best_val_acc
    config['test_acc'] = test_acc
    config['continuous_cols'] = continuous_cols
    
    with open('models/mlp_v2/mlp_config_v2.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\nArtifacts saved to models/mlp_v2/")
    print("Training complete!")


if __name__ == "__main__":
    main()
