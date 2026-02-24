#!/usr/bin/env python3
"""
Transformer Model Training for Cricket Ball Prediction

Uses FULL INNINGS CONTEXT (up to 120 balls) instead of sliding window.
Key differentiator from LSTM: can attend to entire innings history.

Same features as LSTM v1, same 6-class prediction task.

Usage:
    uv run python scripts/transformer_v1.py                         # Train with defaults
    uv run python scripts/transformer_v1.py --epochs 50 --quick     # Quick test run
    uv run python scripts/transformer_v1.py --batch-size 64         # Adjust batch size
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
from sklearn.metrics import accuracy_score, log_loss, classification_report
import joblib
from tqdm import tqdm

# Import reusable components from lstm_v1
from lstm_v1 import FocalLoss, WarmupCosineScheduler


# ============================================================================
# DATASET - Full Innings Context (Variable Length 1-120 balls)
# ============================================================================

class CricketFullContextDataset(Dataset):
    """
    Dataset that provides FULL INNINGS CONTEXT (1-120 balls).

    Unlike LSTM's 10-ball sliding window, each sample returns ALL balls
    from the start of the innings up to the current position.

    Key differences from CricketSequenceDataset:
    - Variable-length sequences (1 to 120 balls)
    - No fixed window size
    - Requires custom collate_fn for batch padding
    """

    def __init__(self, df: pd.DataFrame, feature_cols: list, categorical_cols: dict,
                 scaler: StandardScaler = None, max_seq_len: int = 120, fit_scaler: bool = False):
        """
        Args:
            df: DataFrame with ball-by-ball data
            feature_cols: List of all feature columns
            categorical_cols: Dict mapping categorical col name to vocab size
            scaler: StandardScaler for continuous features
            max_seq_len: Maximum sequence length (120 for full T20 innings)
            fit_scaler: Whether to fit the scaler on this data
        """
        self.max_seq_len = max_seq_len
        self.feature_cols = feature_cols
        self.categorical_cols = categorical_cols
        self.continuous_cols = [c for c in feature_cols if c not in categorical_cols]

        # Prepare target (remap classes) - same as LSTM
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

        # Create samples: (innings_id, position_in_innings, global_idx, seq_length)
        # Each sample is a full innings context up to that position
        self.samples = []
        for innings_id in self.innings_ids:
            indices = self.innings_groups[innings_id]
            for pos in range(len(indices)):
                # Sequence length = position + 1 (1 to N balls)
                seq_len = min(pos + 1, max_seq_len)
                self.samples.append((innings_id, pos, indices[pos], seq_len))

        print(f"  Dataset: {len(self.samples)} samples from {len(self.innings_ids)} innings (full context)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        innings_id, pos, global_idx, seq_len = self.samples[idx]

        # Get all indices for this innings
        innings_indices = self.innings_groups[innings_id]

        # Get ALL balls from start of innings to current position
        # If pos > max_seq_len, take last max_seq_len balls
        start_pos = max(0, pos - self.max_seq_len + 1)
        window_indices = innings_indices[start_pos:pos + 1]

        # Extract features for full context window
        window_df = self.df.iloc[window_indices]
        actual_seq_len = len(window_df)

        # Prepare continuous features
        continuous = window_df[self.continuous_cols].fillna(0).values
        if self.scaler is not None:
            continuous = self.scaler.transform(continuous)

        # Prepare categorical features
        categorical = {}
        for col in self.categorical_cols:
            categorical[col] = window_df[col].fillna(0).values.astype(np.int64)

        # Get target (last ball in sequence)
        target = self.targets[global_idx]

        return {
            'continuous': torch.FloatTensor(continuous),
            'batter_encoded': torch.LongTensor(categorical.get('batter_encoded', np.zeros(actual_seq_len))),
            'bowler_encoded': torch.LongTensor(categorical.get('bowler_encoded', np.zeros(actual_seq_len))),
            'venue_encoded': torch.LongTensor(categorical.get('venue_encoded', np.zeros(actual_seq_len))),
            'matchup_type_encoded': torch.LongTensor(categorical.get('matchup_type_encoded', np.zeros(actual_seq_len))),
            'target': torch.LongTensor([target])[0],
            'seq_len': actual_seq_len  # Actual sequence length for masking
        }


def collate_fn(batch):
    """
    Custom collate function for variable-length sequences.

    Left-pads shorter sequences with zeros so all sequences in batch
    have the same length (max length in batch).
    """
    max_len = max(item['seq_len'] for item in batch)
    batch_size = len(batch)
    n_continuous = batch[0]['continuous'].shape[1]

    # Initialize padded tensors with zeros
    continuous = torch.zeros(batch_size, max_len, n_continuous)
    batter = torch.zeros(batch_size, max_len, dtype=torch.long)
    bowler = torch.zeros(batch_size, max_len, dtype=torch.long)
    venue = torch.zeros(batch_size, max_len, dtype=torch.long)
    matchup = torch.zeros(batch_size, max_len, dtype=torch.long)
    targets = torch.zeros(batch_size, dtype=torch.long)
    seq_lens = torch.zeros(batch_size, dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = item['seq_len']
        # Left-pad: put actual data at the end
        start = max_len - seq_len
        continuous[i, start:] = item['continuous']
        batter[i, start:] = item['batter_encoded']
        bowler[i, start:] = item['bowler_encoded']
        venue[i, start:] = item['venue_encoded']
        matchup[i, start:] = item['matchup_type_encoded']
        targets[i] = item['target']
        seq_lens[i] = seq_len

    return {
        'continuous': continuous,
        'batter_encoded': batter,
        'bowler_encoded': bowler,
        'venue_encoded': venue,
        'matchup_type_encoded': matchup,
        'target': targets,
        'seq_len': seq_lens
    }


# ============================================================================
# MODEL - Transformer Encoder with Causal Masking
# ============================================================================

class TransformerBallPredictor(nn.Module):
    """
    Transformer model for ball outcome prediction with FULL INNINGS CONTEXT.

    Key features:
    - Causal masking: only attends to past balls
    - Learnable positional encoding for 120 positions
    - Pre-LayerNorm architecture for better training stability
    - Same embeddings as LSTM for categorical features
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
                 num_layers: int = 4,
                 nhead: int = 8,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 max_seq_len: int = 120,
                 n_classes: int = 6):
        super().__init__()

        self.n_continuous = n_continuous
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

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

        # Learnable positional encoding for full innings (120 balls)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_size)

        # Transformer Encoder with Pre-LN for better gradient flow
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN architecture
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final layer norm before classifier
        self.final_ln = nn.LayerNorm(hidden_size)

        # Output head (similar to LSTM)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
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
            'nhead': nhead,
            'dim_feedforward': dim_feedforward,
            'dropout': dropout,
            'max_seq_len': max_seq_len,
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
        Forward pass with full innings context.

        Args:
            continuous: (batch, seq_len, n_continuous) - continuous features
            batter_ids: (batch, seq_len) - encoded batter IDs
            bowler_ids: (batch, seq_len) - encoded bowler IDs
            venue_ids: (batch, seq_len) - encoded venue IDs
            matchup_ids: (batch, seq_len) - encoded matchup type IDs

        Returns:
            logits: (batch, n_classes) - class logits
        """
        batch_size, seq_len = batter_ids.shape

        # Get embeddings
        batter_emb = self.batter_embed(batter_ids)
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

        # Add positional encoding
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pos_embed(positions)

        # No causal mask is required here because each sample is already a
        # prefix sequence (innings start -> current ball). There are no future
        # tokens in the input to leak. This also avoids MPS instability seen
        # with attention mask kernels on older PyTorch builds.
        x = self.transformer(x)

        # Apply final layer norm
        x = self.final_ln(x)

        # Use last position for classification (most recent ball context)
        last_hidden = x[:, -1, :]  # (batch, hidden_size)

        # Classification
        logits = self.classifier(last_hidden)

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
# MLX BACKEND TRAINING
# ============================================================================

def train_with_mlx(args):
    """
    Train Transformer model using MLX backend (Apple Silicon optimized).

    Leverages unified memory architecture for faster training.
    """
    import platform

    # Check if running on Apple Silicon
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        print("=" * 60)
        print("ERROR: --mlx flag requires Apple Silicon Mac (M1/M2/M3/M4)")
        print("Remove --mlx flag to use PyTorch backend")
        print("=" * 60)
        return

    try:
        import mlx.core as mx
        import mlx.optimizers as mlx_optim
        from transformer_mlx import (
            TransformerBallPredictorMLX,
            CricketDatasetMLX,
            WarmupCosineSchedulerMLX,
            train_epoch_mlx,
            evaluate_mlx,
            save_mlx_weights,
            count_parameters
        )
    except ImportError as e:
        print("=" * 60)
        print("ERROR: MLX not installed.")
        print("Install with: pip install mlx safetensors")
        print(f"Details: {e}")
        print("=" * 60)
        return

    print("=" * 60)
    print("TRANSFORMER TRAINING - MLX BACKEND (Apple Silicon)")
    print("=" * 60)
    print(f"MLX version: {mx.__version__}")
    print("Using unified memory (CPU + GPU shared)")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"Max sequence length: {args.max_seq_len} balls (full innings)")
    print(f"Learning rate: {args.lr}, Weight decay: {args.weight_decay}")
    print(f"Hidden size: {args.hidden_size}, Layers: {args.num_layers}")
    print(f"Attention heads: {args.nhead}, FFN dim: {args.dim_feedforward}")
    print("=" * 60)

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

    # Encode categorical variables (same as PyTorch version)
    print("\n--- ENCODING CATEGORICAL VARIABLES ---")

    print("  Encoding batters...")
    unique_batters = pd.concat([
        train_df['batter_id'].astype(str),
        val_df['batter_id'].astype(str),
        test_df['batter_id'].astype(str)
    ]).unique()
    le_batter = LabelEncoder()
    le_batter.fit(unique_batters)
    n_batters = len(unique_batters)

    print("  Encoding bowlers...")
    unique_bowlers = pd.concat([
        train_df['bowler_id'].astype(str),
        val_df['bowler_id'].astype(str),
        test_df['bowler_id'].astype(str)
    ]).unique()
    le_bowler = LabelEncoder()
    le_bowler.fit(unique_bowlers)
    n_bowlers = len(unique_bowlers)

    print("  Encoding venues...")
    unique_venues = pd.concat([
        train_df['venue'].astype(str),
        val_df['venue'].astype(str),
        test_df['venue'].astype(str)
    ]).unique()
    le_venue = LabelEncoder()
    le_venue.fit(unique_venues)
    n_venues = len(unique_venues)

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

    # Define feature columns (same as PyTorch version)
    feature_cols = [
        'inning_idx', 'score', 'wickets', 'balls_bowled', 'run_rate',
        'wickets_ratio', 'balls_ratio', 'wickets_in_hand', 'balls_remaining',
        'is_powerplay', 'is_middle_overs', 'is_death_overs', 'balls_in_over',
        'is_toss_winner', 'is_batting_first',
        'batsman_avg', 'batsman_sr', 'bowler_avg', 'bowler_econ',
        'batsman_recent_avg', 'batsman_recent_sr', 'bowler_recent_avg', 'bowler_recent_econ',
        'batter_balls_faced', 'batter_runs_scored', 'bowler_balls_in_innings', 'bowler_overs_in_innings',
        'h2h_avg', 'h2h_sr',
        'last_5_balls_runs', 'last_10_balls_runs', 'last_30_balls_runs',
        'balls_since_boundary', 'last_10_dots', 'partnership_runs',
        'dot_percentage_recent', 'boundary_percentage_recent', 'pressure_cooker_index',
        'chase_target', 'run_rate_required', 'lead_gap',
        'venue_avg_score', 'non_striker_sr',
        'batter_hand', 'bowler_arm', 'is_pace', 'bowling_type', 'batter_age', 'bowler_age',
        'spin_matchup_advantage', 'same_arm_matchup',
        'batter_avg_vs_pace', 'batter_sr_vs_pace', 'batter_avg_vs_spin', 'batter_sr_vs_spin',
        'bowler_avg_vs_lhb', 'bowler_econ_vs_lhb', 'bowler_avg_vs_rhb', 'bowler_econ_vs_rhb',
        'batter_encoded', 'bowler_encoded', 'venue_encoded', 'matchup_type_encoded',
    ]
    feature_cols = [c for c in feature_cols if c in train_df.columns]

    categorical_cols = {
        'batter_encoded': n_batters + 1,
        'bowler_encoded': n_bowlers + 1,
        'venue_encoded': n_venues + 1,
        'matchup_type_encoded': n_matchups + 1,
    }

    continuous_cols = [c for c in feature_cols if c not in categorical_cols]
    n_continuous = len(continuous_cols)

    print(f"\nFeatures: {len(feature_cols)} total ({n_continuous} continuous, {len(categorical_cols)} categorical)")

    # Create MLX datasets
    print("\n--- CREATING MLX DATASETS (Full Innings Context) ---")
    train_dataset = CricketDatasetMLX(
        train_df, feature_cols, categorical_cols,
        max_seq_len=args.max_seq_len, fit_scaler=True
    )
    scaler = train_dataset.scaler

    val_dataset = CricketDatasetMLX(
        val_df, feature_cols, categorical_cols,
        scaler=scaler, max_seq_len=args.max_seq_len
    )

    test_dataset = CricketDatasetMLX(
        test_df, feature_cols, categorical_cols,
        scaler=scaler, max_seq_len=args.max_seq_len
    )

    # Calculate class weights
    print("\n--- CALCULATING CLASS WEIGHTS ---")
    train_targets = train_dataset.targets
    classes = np.unique(train_targets)
    class_counts = np.bincount(train_targets)
    print(f"Class distribution: {dict(zip(classes, class_counts))}")

    total = len(train_targets)
    class_weights_np = np.array([
        np.sqrt(total / (len(classes) * class_counts[c])) for c in classes
    ])
    class_weights_np = class_weights_np / class_weights_np.mean()
    class_weights = mx.array(class_weights_np.astype(np.float32))
    print(f"Class weights: {class_weights_np.round(3)}")

    # Create MLX model
    print("\n--- CREATING MLX TRANSFORMER MODEL ---")
    model = TransformerBallPredictorMLX(
        n_continuous=n_continuous,
        n_batters=n_batters + 1,
        n_bowlers=n_bowlers + 1,
        n_venues=n_venues + 1,
        n_matchups=n_matchups + 1,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        n_classes=6
    )

    total_params = count_parameters(model)
    print(f"Model parameters: {total_params:,}")

    # MLX optimizer
    print("\n--- MLX OPTIMIZER ---")
    scheduler = WarmupCosineSchedulerMLX(
        base_lr=args.lr,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        min_lr=1e-6
    )

    optimizer = mlx_optim.AdamW(
        learning_rate=scheduler.get_lr(0),
        weight_decay=args.weight_decay
    )
    print(f"Using AdamW with warmup cosine schedule")

    # Training loop
    print("\n--- TRAINING (MLX) ---")
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_logloss': [], 'lr': []}

    Path('models/transformer_v1').mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        # Update learning rate
        current_lr = scheduler.get_lr(epoch)
        optimizer.learning_rate = current_lr

        print(f"\nEpoch {epoch + 1}/{args.epochs} (LR: {current_lr:.6f})")

        # Train
        train_loss, train_acc = train_epoch_mlx(
            model, train_dataset, optimizer, class_weights,
            args.batch_size, args.focal_gamma, args.label_smoothing
        )

        # Validate
        val_loss, val_acc, val_logloss, _, _ = evaluate_mlx(
            model, val_dataset, class_weights,
            args.batch_size, args.focal_gamma, args.label_smoothing
        )

        # Track history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_logloss'].append(val_logloss)
        history['lr'].append(current_lr)

        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val LogLoss: {val_logloss:.4f}")

        # Early stopping check
        if val_logloss < best_val_loss:
            best_val_loss = val_logloss
            best_epoch = epoch
            patience_counter = 0

            # Save best model (MLX format)
            save_mlx_weights(model, 'models/transformer_v1/transformer_model_v1_mlx_best.safetensors')
            print(f"  * New best model (val_logloss={val_logloss:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print(f"\n[EARLY STOPPING] at epoch {epoch + 1} (best was epoch {best_epoch + 1})")
                break

    # Final evaluation
    print("\n--- FINAL EVALUATION ---")
    from transformer_mlx import load_mlx_weights
    load_mlx_weights(model, 'models/transformer_v1/transformer_model_v1_mlx_best.safetensors')

    test_loss, test_acc, test_logloss, test_preds, test_targets = evaluate_mlx(
        model, test_dataset, class_weights,
        args.batch_size, args.focal_gamma, args.label_smoothing
    )

    print(f"\nTest Results:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Log Loss: {test_logloss:.4f}")

    # Save artifacts
    print("\n--- SAVING ARTIFACTS ---")

    # Save MLX model weights
    save_mlx_weights(model, 'models/transformer_v1/transformer_model_v1_mlx.safetensors')

    # Save config
    config = model.config.copy()
    config['continuous_cols'] = continuous_cols
    with open('models/transformer_v1/transformer_config_v1.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Save scaler
    joblib.dump(scaler, 'models/transformer_v1/feature_scaler_v1.pkl')

    # Save encoders
    joblib.dump(le_batter, 'models/transformer_v1/batter_encoder_v1.pkl')
    joblib.dump(le_bowler, 'models/transformer_v1/bowler_encoder_v1.pkl')
    joblib.dump(le_venue, 'models/transformer_v1/venue_encoder_v1.pkl')
    joblib.dump(le_matchup, 'models/transformer_v1/matchup_encoder_v1.pkl')

    # Save feature columns
    with open('models/transformer_v1/feature_columns_v1.txt', 'w') as f:
        for col in feature_cols:
            f.write(f"{col}\n")

    with open('models/transformer_v1/continuous_columns_v1.txt', 'w') as f:
        for col in continuous_cols:
            f.write(f"{col}\n")

    # Save training history
    with open('models/transformer_v1/training_history_v1.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\nArtifacts saved to models/transformer_v1/:")
    print("  - transformer_model_v1_mlx.safetensors (MLX weights)")
    print("  - transformer_config_v1.json")
    print("  - feature_scaler_v1.pkl")
    print("  - *_encoder_v1.pkl")
    print("  - feature_columns_v1.txt")
    print("  - training_history_v1.json")

    print(f"\n--- MLX TRAINING COMPLETE ---")
    print(f"Best validation loss: {best_val_loss:.4f} (epoch {best_epoch + 1})")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test log loss: {test_logloss:.4f}")


# ============================================================================
# MAIN (PyTorch Backend)
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Transformer v1 model (full innings context)')
    # Training config
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size (smaller due to longer sequences)')
    parser.add_argument('--max-seq-len', type=int, default=120, help='Maximum sequence length (full innings)')

    # Optimizer config
    parser.add_argument('--lr', type=float, default=1e-4, help='Peak learning rate (lower for transformer)')
    parser.add_argument('--weight-decay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Warmup epochs')

    # Model architecture
    parser.add_argument('--hidden-size', type=int, default=256, help='Transformer hidden size')
    parser.add_argument('--num-layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--dim-feedforward', type=int, default=512, help='FFN hidden size')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (lower for transformer)')

    # Loss function
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Focal loss gamma')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing')

    # Training options
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--quick', action='store_true', help='Quick test with subset of data')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda, mps, cpu)')
    parser.add_argument('--mlx', action='store_true',
                       help='Use MLX backend (Apple Silicon only, faster on Mac)')
    args = parser.parse_args()

    # MLX backend selection
    if args.mlx:
        train_with_mlx(args)
        return

    print("=" * 60)
    print("TRANSFORMER TRAINING CONFIGURATION (Full Innings Context)")
    print("=" * 60)
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"Max sequence length: {args.max_seq_len} balls (full innings)")
    print(f"Learning rate: {args.lr}, Weight decay: {args.weight_decay}")
    print(f"Warmup epochs: {args.warmup_epochs}")
    print(f"Hidden size: {args.hidden_size}, Layers: {args.num_layers}")
    print(f"Attention heads: {args.nhead}, FFN dim: {args.dim_feedforward}")
    print(f"Dropout: {args.dropout}")
    print(f"Focal loss: gamma={args.focal_gamma}, label_smoothing={args.label_smoothing}")
    print("=" * 60)

    # Device selection
    import os
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif os.environ.get('USE_MPS', '0') == '1' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # MPS stability guidance for older PyTorch builds.
    if device.type == 'mps':
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
        torch_version = tuple(int(p) for p in torch.__version__.split('+')[0].split('.')[:2])
        if torch_version <= (2, 0):
            print("\n[WARNING] PyTorch 2.0.x on MPS is known to be unstable for some Transformer workloads.")
            print("Recommended: use --mlx backend on Apple Silicon for this model.")
            print("Fallback path enabled: PYTORCH_ENABLE_MPS_FALLBACK=1")

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

    # Encode categorical variables (same as LSTM)
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

    # Define feature columns (same as LSTM v1)
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

    # Continuous columns
    continuous_cols = [c for c in feature_cols if c not in categorical_cols]
    n_continuous = len(continuous_cols)

    print(f"\nFeatures: {len(feature_cols)} total ({n_continuous} continuous, {len(categorical_cols)} categorical)")

    # Create datasets with full context
    print("\n--- CREATING DATASETS (Full Innings Context) ---")
    train_dataset = CricketFullContextDataset(
        train_df, feature_cols, categorical_cols,
        max_seq_len=args.max_seq_len, fit_scaler=True
    )
    scaler = train_dataset.scaler

    val_dataset = CricketFullContextDataset(
        val_df, feature_cols, categorical_cols,
        scaler=scaler, max_seq_len=args.max_seq_len
    )

    test_dataset = CricketFullContextDataset(
        test_df, feature_cols, categorical_cols,
        scaler=scaler, max_seq_len=args.max_seq_len
    )

    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn
    )

    # Calculate class weights
    print("\n--- CALCULATING CLASS WEIGHTS ---")
    train_targets = train_dataset.targets
    classes = np.unique(train_targets)
    class_counts = np.bincount(train_targets)
    print(f"Class distribution: {dict(zip(classes, class_counts))}")

    # Use inverse frequency with smoothing
    total = len(train_targets)
    class_weights = []
    for c in classes:
        weight = np.sqrt(total / (len(classes) * class_counts[c]))
        class_weights.append(weight)

    class_weights = np.array(class_weights)
    class_weights = class_weights / class_weights.mean()
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"Class weights: {class_weights.cpu().numpy().round(3)}")

    # Create model
    print("\n--- CREATING TRANSFORMER MODEL ---")
    model = TransformerBallPredictor(
        n_continuous=n_continuous,
        n_batters=n_batters + 1,
        n_bowlers=n_bowlers + 1,
        n_venues=n_venues + 1,
        n_matchups=n_matchups + 1,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        n_classes=6
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Loss function - Focal Loss
    print("\n--- LOSS FUNCTION ---")
    criterion = FocalLoss(
        alpha=class_weights,
        gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing
    )
    print(f"Using Focal Loss (gamma={args.focal_gamma}, label_smoothing={args.label_smoothing})")

    # Optimizer
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
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_logloss': [], 'lr': []}

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
        history['lr'].append(current_lr)

        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val LogLoss: {val_logloss:.4f}")

        # Early stopping check
        if val_logloss < best_val_loss:
            best_val_loss = val_logloss
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            Path('models/transformer_v1').mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), 'models/transformer_v1/transformer_model_v1_best.pt')
            print(f"  ★ New best model (val_logloss={val_logloss:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print(f"\n[EARLY STOPPING] at epoch {epoch + 1} (best was epoch {best_epoch + 1})")
                break

    # Load best model for final evaluation
    print("\n--- FINAL EVALUATION ---")
    model.load_state_dict(torch.load('models/transformer_v1/transformer_model_v1_best.pt', weights_only=True))

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
    Path('models/transformer_v1').mkdir(parents=True, exist_ok=True)

    # Save model state dict
    torch.save(model.state_dict(), 'models/transformer_v1/transformer_model_v1.pt')

    # Save model config
    config = model.config.copy()
    config['continuous_cols'] = continuous_cols
    with open('models/transformer_v1/transformer_config_v1.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Save scaler
    joblib.dump(scaler, 'models/transformer_v1/feature_scaler_v1.pkl')

    # Save encoders
    joblib.dump(le_batter, 'models/transformer_v1/batter_encoder_v1.pkl')
    joblib.dump(le_bowler, 'models/transformer_v1/bowler_encoder_v1.pkl')
    joblib.dump(le_venue, 'models/transformer_v1/venue_encoder_v1.pkl')
    joblib.dump(le_matchup, 'models/transformer_v1/matchup_encoder_v1.pkl')

    # Save feature columns
    with open('models/transformer_v1/feature_columns_v1.txt', 'w') as f:
        for col in feature_cols:
            f.write(f"{col}\n")

    # Save continuous columns list
    with open('models/transformer_v1/continuous_columns_v1.txt', 'w') as f:
        for col in continuous_cols:
            f.write(f"{col}\n")

    # Save training history
    with open('models/transformer_v1/training_history_v1.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\nArtifacts saved to models/transformer_v1/:")
    print("  - transformer_model_v1.pt (model weights)")
    print("  - transformer_config_v1.json (architecture config)")
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
