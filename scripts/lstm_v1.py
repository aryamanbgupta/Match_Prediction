#!/usr/bin/env python3
"""
LSTM Model Training for Cricket Ball Prediction

Uses sliding window of last N balls as sequence context.
Same features as XGBoost v3, same 6-class prediction task.

Usage:
    uv run python scripts/lstm_v1.py                         # Train with defaults
    uv run python scripts/lstm_v1.py --epochs 50 --quick     # Quick test run
    uv run python scripts/lstm_v1.py --tune --n-trials 20    # Hyperparameter tuning
"""

import argparse
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, log_loss, classification_report
import joblib
from tqdm import tqdm


# ============================================================================
# FOCAL LOSS - Better for class imbalance
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    Reduces loss for well-classified examples, focusing on hard examples.
    Particularly useful for class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter (0 = CE, higher = more focus on hard examples)
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Apply label smoothing
        n_classes = inputs.size(-1)

        # Compute softmax probabilities
        p = F.softmax(inputs, dim=-1)

        # Get probability of true class
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha,
                                   label_smoothing=self.label_smoothing, reduction='none')

        # Get p_t (probability of true class)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Apply focal weight
        focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ============================================================================
# LEARNING RATE SCHEDULER WITH WARMUP
# ============================================================================

class WarmupCosineScheduler:
    """Cosine annealing with linear warmup."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            scale = (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            scale = 0.5 * (1 + math.cos(math.pi * progress))

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = max(self.min_lr, base_lr * scale)

        return self.optimizer.param_groups[0]['lr']

# ============================================================================
# DATASET
# ============================================================================

class CricketSequenceDataset(Dataset):
    """
    Dataset that provides sliding windows of ball sequences.

    For each ball, returns:
    - Last N balls as sequence (padded if at start of innings)
    - Target label (6 classes after remapping)
    """

    def __init__(self, df: pd.DataFrame, feature_cols: list, categorical_cols: dict,
                 scaler: StandardScaler = None, window_size: int = 10, fit_scaler: bool = False):
        """
        Args:
            df: DataFrame with ball-by-ball data
            feature_cols: List of all feature columns
            categorical_cols: Dict mapping categorical col name to vocab size
            scaler: StandardScaler for continuous features (fit if None and fit_scaler=True)
            window_size: Number of balls in sequence context
            fit_scaler: Whether to fit the scaler on this data
        """
        self.window_size = window_size
        self.feature_cols = feature_cols
        self.categorical_cols = categorical_cols

        # Identify continuous vs categorical columns
        self.continuous_cols = [c for c in feature_cols if c not in categorical_cols]

        # Prepare target (remap classes)
        df = df.copy()
        df['target'] = df['ball_outcome'].copy()
        df.loc[df['target'] == -1, 'target'] = 7  # Wicket

        # Class remapping: {0: 0, 1: 1, 2: 2, 4: 3, 6: 4, 7: 5}
        class_mapping = {0: 0, 1: 1, 2: 2, 4: 3, 6: 4, 7: 5}
        df['target'] = df['target'].map(class_mapping)

        # Remove invalid targets
        df = df[df['target'].notna()].copy()

        # Sort by innings and ball index for proper sequencing
        df = df.sort_values(['innings_id', 'ball_idx']).reset_index(drop=True)

        # Fit or use scaler for continuous features
        self.scaler = scaler
        if fit_scaler and scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(df[self.continuous_cols].fillna(0).values)

        # Store data
        self.df = df
        self.targets = df['target'].values.astype(np.int64)

        # Build innings index for efficient sequence lookup
        self.innings_groups = df.groupby('innings_id').indices
        self.innings_ids = list(self.innings_groups.keys())

        # Create sample index: (innings_id, position_in_innings)
        self.samples = []
        for innings_id in self.innings_ids:
            indices = self.innings_groups[innings_id]
            for pos in range(len(indices)):
                self.samples.append((innings_id, pos, indices[pos]))

        print(f"  Dataset: {len(self.samples)} samples from {len(self.innings_ids)} innings")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        innings_id, pos, global_idx = self.samples[idx]

        # Get all indices for this innings
        innings_indices = self.innings_groups[innings_id]

        # Get window of balls ending at current position
        start_pos = max(0, pos - self.window_size + 1)
        window_indices = innings_indices[start_pos:pos + 1]

        # Extract features for window
        window_df = self.df.iloc[window_indices]

        # Prepare continuous features (use .values to avoid sklearn warning)
        continuous = window_df[self.continuous_cols].fillna(0).values
        if self.scaler is not None:
            # Transform expects 2D array
            continuous = self.scaler.transform(continuous)

        # Prepare categorical features
        categorical = {}
        for col in self.categorical_cols:
            categorical[col] = window_df[col].fillna(0).values.astype(np.int64)

        # Pad if sequence is shorter than window_size
        pad_length = self.window_size - len(window_df)
        if pad_length > 0:
            # Pad continuous
            continuous = np.vstack([np.zeros((pad_length, continuous.shape[1])), continuous])
            # Pad categorical
            for col in categorical:
                categorical[col] = np.concatenate([np.zeros(pad_length, dtype=np.int64), categorical[col]])

        # Get target (last ball in sequence)
        target = self.targets[global_idx]

        return {
            'continuous': torch.FloatTensor(continuous),
            'batter_encoded': torch.LongTensor(categorical.get('batter_encoded', np.zeros(self.window_size))),
            'bowler_encoded': torch.LongTensor(categorical.get('bowler_encoded', np.zeros(self.window_size))),
            'venue_encoded': torch.LongTensor(categorical.get('venue_encoded', np.zeros(self.window_size))),
            'matchup_type_encoded': torch.LongTensor(categorical.get('matchup_type_encoded', np.zeros(self.window_size))),
            'target': torch.LongTensor([target])[0]
        }


# ============================================================================
# MODEL
# ============================================================================

class LSTMBallPredictor(nn.Module):
    """
    LSTM model for ball outcome prediction with embeddings for categorical features.

    Improvements:
    - Layer normalization for better training stability
    - Residual connection in output head
    - Wider output head for better representation
    - Embedding dropout for regularization
    """

    def __init__(self,
                 n_continuous: int,
                 n_batters: int = 8000,
                 n_bowlers: int = 6000,
                 n_venues: int = 300,
                 n_matchups: int = 100,
                 embed_dim_player: int = 64,
                 embed_dim_venue: int = 32,
                 embed_dim_matchup: int = 16,
                 hidden_size: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 n_classes: int = 6):
        super().__init__()

        self.n_continuous = n_continuous
        self.hidden_size = hidden_size

        # Embeddings for categorical features (padding_idx=0)
        self.batter_embed = nn.Embedding(n_batters + 1, embed_dim_player, padding_idx=0)
        self.bowler_embed = nn.Embedding(n_bowlers + 1, embed_dim_player, padding_idx=0)
        self.venue_embed = nn.Embedding(n_venues + 1, embed_dim_venue, padding_idx=0)
        self.matchup_embed = nn.Embedding(n_matchups + 1, embed_dim_matchup, padding_idx=0)

        # Embedding dropout
        self.embed_dropout = nn.Dropout(dropout)

        # Total input dimension per timestep
        self.input_dim = (n_continuous +
                         2 * embed_dim_player +  # batter + bowler
                         embed_dim_venue +
                         embed_dim_matchup)

        # Input projection with layer norm
        self.input_proj = nn.Linear(self.input_dim, hidden_size)
        self.input_ln = nn.LayerNorm(hidden_size)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False  # Causal - only use past context
        )

        # Layer norm after LSTM
        self.lstm_ln = nn.LayerNorm(hidden_size)

        # Improved output head (wider with residual-like structure)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),  # GELU works better than ReLU for transformers/LSTMs
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),  # Less dropout in later layers
            nn.Linear(hidden_size // 2, n_classes)
        )

        # Initialize weights
        self._init_weights()

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
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'n_classes': n_classes
        }

    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, continuous, batter_ids, bowler_ids, venue_ids, matchup_ids):
        """
        Forward pass.

        Args:
            continuous: (batch, seq_len, n_continuous) - continuous features
            batter_ids: (batch, seq_len) - encoded batter IDs
            bowler_ids: (batch, seq_len) - encoded bowler IDs
            venue_ids: (batch, seq_len) - encoded venue IDs
            matchup_ids: (batch, seq_len) - encoded matchup type IDs

        Returns:
            logits: (batch, n_classes) - class logits
        """
        # Get embeddings
        batter_emb = self.batter_embed(batter_ids)      # (batch, seq, embed_dim)
        bowler_emb = self.bowler_embed(bowler_ids)
        venue_emb = self.venue_embed(venue_ids)
        matchup_emb = self.matchup_embed(matchup_ids)

        # Apply embedding dropout
        batter_emb = self.embed_dropout(batter_emb)
        bowler_emb = self.embed_dropout(bowler_emb)
        venue_emb = self.embed_dropout(venue_emb)
        matchup_emb = self.embed_dropout(matchup_emb)

        # Concatenate all features
        x = torch.cat([
            continuous, batter_emb, bowler_emb, venue_emb, matchup_emb
        ], dim=-1)  # (batch, seq, input_dim)

        # Project and normalize
        x = self.input_proj(x)
        x = self.input_ln(x)

        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Apply layer norm after LSTM
        lstm_out = self.lstm_ln(lstm_out)

        # Use last hidden state for classification
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)

        # Classification
        logits = self.classifier(last_hidden)  # (batch, n_classes)

        return logits


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device, grad_clip=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc='Training', leave=False)
    for batch in pbar:
        # Move to device
        continuous = batch['continuous'].to(device)
        batter_ids = batch['batter_encoded'].to(device)
        bowler_ids = batch['bowler_encoded'].to(device)
        venue_ids = batch['venue_encoded'].to(device)
        matchup_ids = batch['matchup_type_encoded'].to(device)
        targets = batch['target'].to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(continuous, batter_ids, bowler_ids, venue_ids, matchup_ids)
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # Track metrics
        total_loss += loss.item() * len(targets)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_targets, all_preds)
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating', leave=False):
            continuous = batch['continuous'].to(device)
            batter_ids = batch['batter_encoded'].to(device)
            bowler_ids = batch['bowler_encoded'].to(device)
            venue_ids = batch['venue_encoded'].to(device)
            matchup_ids = batch['matchup_type_encoded'].to(device)
            targets = batch['target'].to(device)

            logits = model(continuous, batter_ids, bowler_ids, venue_ids, matchup_ids)
            loss = criterion(logits, targets)

            total_loss += loss.item() * len(targets)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_targets, all_preds)
    logloss = log_loss(all_targets, all_probs, labels=list(range(6)))

    return avg_loss, accuracy, logloss, all_preds, all_targets


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train LSTM v1 model')
    # Training config
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size (smaller = more updates)')
    parser.add_argument('--window-size', type=int, default=10, help='Sequence window size')

    # Optimizer config
    parser.add_argument('--lr', type=float, default=3e-4, help='Peak learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-2, help='Weight decay (L2 regularization)')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Warmup epochs')

    # Model architecture
    parser.add_argument('--hidden-size', type=int, default=256, help='LSTM hidden size')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')

    # Loss function
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Focal loss gamma (0=CE)')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing')

    # Training options
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--quick', action='store_true', help='Quick test with subset of data')
    parser.add_argument('--tune', action='store_true', help='Run Optuna hyperparameter tuning')
    parser.add_argument('--n-trials', type=int, default=20, help='Number of Optuna trials')
    parser.add_argument('--use-focal-loss', action='store_true', default=True, help='Use focal loss')
    parser.add_argument('--config-json', type=str, default=None,
                        help='JSON config from experiment runner (overrides feature list)')
    args = parser.parse_args()

    print("=" * 60)
    print("LSTM TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}, Weight decay: {args.weight_decay}")
    print(f"Warmup epochs: {args.warmup_epochs}")
    print(f"Hidden size: {args.hidden_size}, Layers: {args.num_layers}")
    print(f"Dropout: {args.dropout}")
    print(f"Focal loss: gamma={args.focal_gamma}, label_smoothing={args.label_smoothing}")
    print("=" * 60)

    # Device - use CPU by default for stability (MPS can cause crashes)
    # Set PYTORCH_ENABLE_MPS_FALLBACK=1 if you want to try MPS
    import os
    if os.environ.get('USE_MPS', '0') == '1' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load data
    print("\n--- LOADING DATA ---")
    print("Loading v3 parquet files...")
    train_df = pd.read_parquet('data/xgb_data_v3/cricket_data_v3_train.parquet')
    val_df = pd.read_parquet('data/xgb_data_v3/cricket_data_v3_validation.parquet')
    test_df = pd.read_parquet('data/xgb_data_v3/cricket_data_v3_test.parquet')

    print(f"Train: {len(train_df)} balls")
    print(f"Val: {len(val_df)} balls")
    print(f"Test: {len(test_df)} balls")

    # Quick mode - use subset
    if args.quick:
        print("\n[QUICK MODE] Using 5% of data for fast testing...")
        train_df = train_df.sample(frac=0.05, random_state=42)
        val_df = val_df.sample(frac=0.05, random_state=42)
        test_df = test_df.sample(frac=0.05, random_state=42)
        args.epochs = min(args.epochs, 3)

    # Encode categorical variables (same as XGBoost)
    print("\n--- ENCODING CATEGORICAL VARIABLES ---")

    # Batter encoder
    print("  Encoding batters...")
    unique_batters = pd.concat([
        train_df['batter_id'].astype(str),
        val_df['batter_id'].astype(str),
        test_df['batter_id'].astype(str)
    ]).unique()
    le_batter = LabelEncoder()
    le_batter.fit(unique_batters)
    n_batters = len(unique_batters)

    # Bowler encoder
    print("  Encoding bowlers...")
    unique_bowlers = pd.concat([
        train_df['bowler_id'].astype(str),
        val_df['bowler_id'].astype(str),
        test_df['bowler_id'].astype(str)
    ]).unique()
    le_bowler = LabelEncoder()
    le_bowler.fit(unique_bowlers)
    n_bowlers = len(unique_bowlers)

    # Venue encoder
    print("  Encoding venues...")
    unique_venues = pd.concat([
        train_df['venue'].astype(str),
        val_df['venue'].astype(str),
        test_df['venue'].astype(str)
    ]).unique()
    le_venue = LabelEncoder()
    le_venue.fit(unique_venues)
    n_venues = len(unique_venues)

    # Matchup encoder
    print("  Encoding matchup types...")
    unique_matchups = pd.concat([
        train_df['matchup_type'].astype(str),
        val_df['matchup_type'].astype(str),
        test_df['matchup_type'].astype(str)
    ]).unique()
    le_matchup = LabelEncoder()
    le_matchup.fit(unique_matchups)
    n_matchups = len(unique_matchups)

    print(f"  Batters: {n_batters}, Bowlers: {n_bowlers}, Venues: {n_venues}, Matchups: {n_matchups}")

    # Apply encoding (+1 to leave 0 for padding)
    for df in [train_df, val_df, test_df]:
        df['batter_encoded'] = le_batter.transform(df['batter_id'].astype(str)) + 1
        df['bowler_encoded'] = le_bowler.transform(df['bowler_id'].astype(str)) + 1
        df['venue_encoded'] = le_venue.transform(df['venue'].astype(str)) + 1
        df['matchup_type_encoded'] = le_matchup.transform(df['matchup_type'].astype(str)) + 1

    # Define feature columns
    if args.config_json:
        import json as _json
        _config = _json.loads(args.config_json)
        from feature_registry import resolve_feature_list
        feature_cols = resolve_feature_list(
            _config['features']['groups'],
            _config['features'].get('exclude'),
            _config['features'].get('include_extra'),
        )
        print(f"[config-json] Using {len(feature_cols)} features from experiment config")
    else:
        # Original hardcoded feature list (default behavior)
        feature_cols = [
            # Basic state
            'inning_idx', 'score', 'wickets', 'balls_bowled', 'run_rate',
            'wickets_ratio', 'balls_ratio', 'wickets_in_hand', 'balls_remaining',
            'is_powerplay', 'is_middle_overs', 'is_death_overs', 'balls_in_over',
            'is_toss_winner', 'is_batting_first',
            # Player stats
            'batsman_avg', 'batsman_sr', 'bowler_avg', 'bowler_econ',
            'batsman_recent_avg', 'batsman_recent_sr', 'bowler_recent_avg', 'bowler_recent_econ',
            'batter_balls_faced', 'batter_runs_scored', 'bowler_balls_in_innings', 'bowler_overs_in_innings',
            # H2H
            'h2h_avg', 'h2h_sr',
            # Momentum
            'last_5_balls_runs', 'last_10_balls_runs', 'last_30_balls_runs',
            'balls_since_boundary', 'last_10_dots', 'partnership_runs',
            # Pressure
            'dot_percentage_recent', 'boundary_percentage_recent', 'pressure_cooker_index',
            # Chase
            'chase_target', 'run_rate_required', 'lead_gap',
            # Medium
            'venue_avg_score', 'non_striker_sr',
            # Player metadata
            'batter_hand', 'bowler_arm', 'is_pace', 'bowling_type', 'batter_age', 'bowler_age',
            # Matchup
            'spin_matchup_advantage', 'same_arm_matchup',
            # Type-based stats
            'batter_avg_vs_pace', 'batter_sr_vs_pace', 'batter_avg_vs_spin', 'batter_sr_vs_spin',
            'bowler_avg_vs_lhb', 'bowler_econ_vs_lhb', 'bowler_avg_vs_rhb', 'bowler_econ_vs_rhb',
            # Categorical (encoded)
            'batter_encoded', 'bowler_encoded', 'venue_encoded', 'matchup_type_encoded',
        ]

    # Filter to available columns
    feature_cols = [c for c in feature_cols if c in train_df.columns]

    # Define categorical columns with vocab sizes
    categorical_cols = {
        'batter_encoded': n_batters + 1,
        'bowler_encoded': n_bowlers + 1,
        'venue_encoded': n_venues + 1,
        'matchup_type_encoded': n_matchups + 1,
    }

    # Continuous columns (everything except categorical)
    continuous_cols = [c for c in feature_cols if c not in categorical_cols]
    n_continuous = len(continuous_cols)

    print(f"\nFeatures: {len(feature_cols)} total ({n_continuous} continuous, {len(categorical_cols)} categorical)")

    # Create datasets
    print("\n--- CREATING DATASETS ---")
    train_dataset = CricketSequenceDataset(
        train_df, feature_cols, categorical_cols,
        window_size=args.window_size, fit_scaler=True
    )
    scaler = train_dataset.scaler

    val_dataset = CricketSequenceDataset(
        val_df, feature_cols, categorical_cols,
        scaler=scaler, window_size=args.window_size
    )

    test_dataset = CricketSequenceDataset(
        test_df, feature_cols, categorical_cols,
        scaler=scaler, window_size=args.window_size
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Calculate class weights with stronger emphasis on rare classes
    print("\n--- CALCULATING CLASS WEIGHTS ---")
    train_targets = train_dataset.targets
    classes = np.unique(train_targets)
    class_counts = np.bincount(train_targets)
    print(f"Class distribution: {dict(zip(classes, class_counts))}")

    # Use inverse frequency with smoothing for more aggressive weighting
    # This gives higher weight to rare classes (sixes, wickets)
    total = len(train_targets)
    class_weights = []
    for c in classes:
        # Inverse frequency with sqrt smoothing (less aggressive than pure inverse)
        weight = np.sqrt(total / (len(classes) * class_counts[c]))
        class_weights.append(weight)

    class_weights = np.array(class_weights)
    # Normalize so mean weight is 1
    class_weights = class_weights / class_weights.mean()
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"Class weights: {class_weights.cpu().numpy().round(3)}")

    # Create model
    print("\n--- CREATING MODEL ---")
    model = LSTMBallPredictor(
        n_continuous=n_continuous,
        n_batters=n_batters + 1,
        n_bowlers=n_bowlers + 1,
        n_venues=n_venues + 1,
        n_matchups=n_matchups + 1,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        n_classes=6
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Loss function - use Focal Loss for better handling of class imbalance
    print("\n--- LOSS FUNCTION ---")
    if args.use_focal_loss:
        criterion = FocalLoss(
            alpha=class_weights,
            gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing
        )
        print(f"Using Focal Loss (gamma={args.focal_gamma}, label_smoothing={args.label_smoothing})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
        print(f"Using CrossEntropyLoss (label_smoothing={args.label_smoothing})")

    # Optimizer with stronger weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler with warmup
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        min_lr=1e-6
    )
    print(f"Using WarmupCosineScheduler (warmup={args.warmup_epochs} epochs)")

    # Training loop
    print("\n--- TRAINING ---")
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_logloss': []}

    for epoch in range(args.epochs):
        # Update learning rate at start of epoch
        current_lr = scheduler.step(epoch)

        print(f"\nEpoch {epoch + 1}/{args.epochs} (LR: {current_lr:.6f})")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc, val_logloss, _, _ = evaluate(model, val_loader, criterion, device)

        # Track history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_logloss'].append(val_logloss)
        history['lr'] = history.get('lr', [])
        history['lr'].append(current_lr)

        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val LogLoss: {val_logloss:.4f}")

        # Early stopping check based on validation log loss (better metric for probability calibration)
        if val_logloss < best_val_loss:
            best_val_loss = val_logloss
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            Path('models/lstm_v1').mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), 'models/lstm_v1/lstm_model_v1_best.pt')
            print(f"  ★ New best model (val_logloss={val_logloss:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print(f"\n[EARLY STOPPING] at epoch {epoch + 1} (best was epoch {best_epoch + 1})")
                break

    # Load best model for final evaluation
    print("\n--- FINAL EVALUATION ---")
    model.load_state_dict(torch.load('models/lstm_v1/lstm_model_v1_best.pt', weights_only=True))

    test_loss, test_acc, test_logloss, test_preds, test_targets = evaluate(
        model, test_loader, criterion, device
    )

    print(f"\nTest Results:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Log Loss: {test_logloss:.4f}")

    # Classification report
    reverse_mapping = {0: 0, 1: 1, 2: 2, 3: 4, 4: 6, 5: 'wicket'}
    target_names = [f'{reverse_mapping[i]}_runs' if isinstance(reverse_mapping[i], int) else reverse_mapping[i]
                   for i in range(6)]
    print("\nClassification Report:")
    print(classification_report(test_targets, test_preds, target_names=target_names))

    # Save final model and artifacts
    print("\n--- SAVING ARTIFACTS ---")
    Path('models/lstm_v1').mkdir(parents=True, exist_ok=True)

    # Save model state dict (for loading)
    torch.save(model.state_dict(), 'models/lstm_v1/lstm_model_v1.pt')

    # Save model config
    config = model.config.copy()
    config['window_size'] = args.window_size
    config['continuous_cols'] = continuous_cols
    with open('models/lstm_v1/lstm_config_v1.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Save scaler
    joblib.dump(scaler, 'models/lstm_v1/feature_scaler_v1.pkl')

    # Save encoders
    joblib.dump(le_batter, 'models/lstm_v1/batter_encoder_v1.pkl')
    joblib.dump(le_bowler, 'models/lstm_v1/bowler_encoder_v1.pkl')
    joblib.dump(le_venue, 'models/lstm_v1/venue_encoder_v1.pkl')
    joblib.dump(le_matchup, 'models/lstm_v1/matchup_encoder_v1.pkl')

    # Save feature columns
    with open('models/lstm_v1/feature_columns_v1.txt', 'w') as f:
        for col in feature_cols:
            f.write(f"{col}\n")

    # Save continuous columns list
    with open('models/lstm_v1/continuous_columns_v1.txt', 'w') as f:
        for col in continuous_cols:
            f.write(f"{col}\n")

    # Save training history
    with open('models/lstm_v1/training_history_v1.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\nArtifacts saved to models/lstm_v1/:")
    print("  - lstm_model_v1.pt (model weights)")
    print("  - lstm_config_v1.json (architecture config)")
    print("  - feature_scaler_v1.pkl (StandardScaler)")
    print("  - *_encoder_v1.pkl (LabelEncoders)")
    print("  - feature_columns_v1.txt")
    print("  - continuous_columns_v1.txt")
    print("  - training_history_v1.json")

    print(f"\n--- TRAINING COMPLETE ---")
    print(f"Best validation loss: {best_val_loss:.4f} (epoch {best_epoch + 1})")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test log loss: {test_logloss:.4f}")


if __name__ == '__main__':
    main()
