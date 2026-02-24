#!/usr/bin/env python3
"""
MLX Backend for Transformer Model

Optimized for Apple Silicon unified memory architecture.
Only imported when --mlx flag is used.

Key advantages over PyTorch on Apple Silicon:
- Unified memory: CPU and GPU share the same RAM - no data copying
- Metal GPU acceleration: Direct access to Apple's Metal backend
- Lazy evaluation: Efficient computation graph execution

Usage:
    # Training
    uv run python scripts/transformer_v1.py --mlx --epochs 50

    # Inference
    uv run python scripts/sim_eval/run_sim_eval.py --model-type transformer --mlx
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# MLX imports (only available on Apple Silicon)
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    # Create dummy classes for type hints when MLX not available
    class mx:
        pass
    class nn:
        class Module:
            pass


def check_mlx_available():
    """Check if MLX is available and raise helpful error if not."""
    if not MLX_AVAILABLE:
        raise ImportError(
            "MLX is not installed. Install with: pip install mlx\n"
            "Note: MLX only works on Apple Silicon Macs (M1/M2/M3/M4)"
        )


# ============================================================================
# MLX Model Architecture
# ============================================================================

class TransformerEncoderLayerMLX(nn.Module):
    """
    Single transformer encoder layer with Pre-LN architecture.

    Pre-LN (norm before attention/FFN) provides better gradient flow
    compared to Post-LN, especially for deeper models.
    """

    def __init__(self, hidden_size: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()

        # Multi-head self-attention
        self.self_attn = nn.MultiHeadAttention(hidden_size, nhead)

        # Layer norms (Pre-LN)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        # Feed-forward network
        self.linear1 = nn.Linear(hidden_size, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, hidden_size)

        # Dropout
        self.dropout_p = dropout

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        """Forward pass with Pre-LN architecture."""
        # Pre-LN Self-Attention
        residual = x
        x = self.ln1(x)
        x = self.self_attn(x, x, x, mask=mask)
        if self.dropout_p > 0:
            x = nn.Dropout(self.dropout_p)(x)
        x = residual + x

        # Pre-LN Feed-Forward
        residual = x
        x = self.ln2(x)
        x = self.linear1(x)
        x = nn.gelu(x)
        if self.dropout_p > 0:
            x = nn.Dropout(self.dropout_p)(x)
        x = self.linear2(x)
        if self.dropout_p > 0:
            x = nn.Dropout(self.dropout_p)(x)
        x = residual + x

        return x


class TransformerBallPredictorMLX(nn.Module):
    """
    MLX version of TransformerBallPredictor.

    Leverages Apple Silicon's unified memory - no CPU/GPU data copying.
    Uses Metal for GPU acceleration automatically.

    Architecture matches PyTorch version exactly for weight compatibility.
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
        self.dropout_p = dropout

        # Embeddings (MLX doesn't have padding_idx, we handle it in forward)
        self.batter_embed = nn.Embedding(n_batters + 1, embed_dim_player)
        self.bowler_embed = nn.Embedding(n_bowlers + 1, embed_dim_player)
        self.venue_embed = nn.Embedding(n_venues + 1, embed_dim_venue)
        self.matchup_embed = nn.Embedding(n_matchups + 1, embed_dim_matchup)

        # Input dimension per timestep
        self.input_dim = (n_continuous +
                         2 * embed_dim_player +  # batter + bowler
                         embed_dim_venue +
                         embed_dim_matchup)

        # Input projection with layer norm
        self.input_proj = nn.Linear(self.input_dim, hidden_size)
        self.input_ln = nn.LayerNorm(hidden_size)

        # Learnable positional encoding for full innings (120 balls)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_size)

        # Transformer encoder layers
        self.transformer_layers = [
            TransformerEncoderLayerMLX(hidden_size, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ]

        # Final layer norm
        self.final_ln = nn.LayerNorm(hidden_size)

        # Output classifier head (matches PyTorch structure)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, n_classes)

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

    def __call__(self, continuous: mx.array, batter_ids: mx.array,
                 bowler_ids: mx.array, venue_ids: mx.array,
                 matchup_ids: mx.array) -> mx.array:
        """
        Forward pass - MLX uses __call__ instead of forward().

        Args:
            continuous: (batch, seq_len, n_continuous) - continuous features
            batter_ids: (batch, seq_len) - encoded batter IDs (+1, 0=padding)
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

        # Zero out padding embeddings (index 0)
        # MLX doesn't have padding_idx, so we mask manually
        pad_mask_batter = mx.expand_dims(batter_ids == 0, axis=-1)
        pad_mask_bowler = mx.expand_dims(bowler_ids == 0, axis=-1)
        pad_mask_venue = mx.expand_dims(venue_ids == 0, axis=-1)
        pad_mask_matchup = mx.expand_dims(matchup_ids == 0, axis=-1)

        batter_emb = mx.where(pad_mask_batter, mx.zeros_like(batter_emb), batter_emb)
        bowler_emb = mx.where(pad_mask_bowler, mx.zeros_like(bowler_emb), bowler_emb)
        venue_emb = mx.where(pad_mask_venue, mx.zeros_like(venue_emb), venue_emb)
        matchup_emb = mx.where(pad_mask_matchup, mx.zeros_like(matchup_emb), matchup_emb)

        # Apply embedding dropout
        if self.dropout_p > 0:
            batter_emb = nn.Dropout(self.dropout_p)(batter_emb)
            bowler_emb = nn.Dropout(self.dropout_p)(bowler_emb)
            venue_emb = nn.Dropout(self.dropout_p)(venue_emb)
            matchup_emb = nn.Dropout(self.dropout_p)(matchup_emb)

        # Concatenate all features
        x = mx.concatenate([
            continuous, batter_emb, bowler_emb, venue_emb, matchup_emb
        ], axis=-1)

        # Project and normalize
        x = self.input_proj(x)
        x = self.input_ln(x)

        # Add positional encoding
        positions = mx.arange(seq_len)
        x = x + self.pos_embed(positions)

        # Generate causal mask for autoregressive attention
        # MLX's MultiHeadAttention expects additive mask (0 = attend, -inf = mask)
        causal_mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)

        # Transformer layers
        for layer in self.transformer_layers:
            x = layer(x, causal_mask)

        # Final layer norm
        x = self.final_ln(x)

        # Use last position for classification (most recent ball context)
        last_hidden = x[:, -1, :]

        # Classifier head
        x = self.fc1(last_hidden)
        x = self.ln1(x)
        x = nn.gelu(x)
        if self.dropout_p > 0:
            x = nn.Dropout(self.dropout_p)(x)
        x = self.fc2(x)
        x = nn.gelu(x)
        if self.dropout_p > 0:
            x = nn.Dropout(self.dropout_p * 0.5)(x)
        logits = self.fc3(x)

        return logits


# ============================================================================
# MLX Loss Functions
# ============================================================================

def focal_loss_mlx(logits: mx.array, targets: mx.array,
                   alpha: Optional[mx.array] = None,
                   gamma: float = 2.0,
                   label_smoothing: float = 0.1) -> mx.array:
    """
    Focal Loss for handling class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        logits: (batch, n_classes) - raw model outputs
        targets: (batch,) - class indices
        alpha: (n_classes,) - class weights
        gamma: focusing parameter (higher = more focus on hard examples)
        label_smoothing: smooth labels to prevent overconfidence
    """
    n_classes = logits.shape[-1]
    batch_size = targets.shape[0]

    # Create one-hot encoding using eye matrix indexing
    one_hot = mx.eye(n_classes)[targets]  # (batch, n_classes)

    # Apply label smoothing
    if label_smoothing > 0:
        smooth_targets = one_hot * (1 - label_smoothing) + label_smoothing / n_classes
    else:
        smooth_targets = one_hot

    # Compute softmax probabilities
    # MLX uses nn.softmax, not mx.softmax
    probs = mx.softmax(logits, axis=-1)
    log_probs = mx.log(probs + 1e-10)  # log_softmax = log(softmax)

    # Compute focal weight
    pt = mx.sum(probs * smooth_targets, axis=-1)
    focal_weight = (1 - pt) ** gamma

    # Compute cross entropy
    ce = -mx.sum(smooth_targets * log_probs, axis=-1)

    # Apply alpha weights if provided
    if alpha is not None:
        alpha_t = alpha[targets]
        loss = alpha_t * focal_weight * ce
    else:
        loss = focal_weight * ce

    return mx.mean(loss)


def cross_entropy_mlx(logits: mx.array, targets: mx.array) -> mx.array:
    """Simple cross entropy loss."""
    return mx.mean(nn.losses.cross_entropy(logits, targets))


# ============================================================================
# MLX Dataset
# ============================================================================

class CricketDatasetMLX:
    """
    MLX-compatible dataset for full innings context.

    Returns MLX arrays directly - unified memory means no CPU->GPU transfer.
    Uses same preprocessing as PyTorch version for compatibility.
    """

    def __init__(self, df: pd.DataFrame, feature_cols: List[str],
                 categorical_cols: Dict[str, int], scaler=None,
                 max_seq_len: int = 120, fit_scaler: bool = False):
        """
        Args:
            df: DataFrame with ball-by-ball data
            feature_cols: List of all feature columns
            categorical_cols: Dict mapping categorical col name to vocab size
            scaler: StandardScaler for continuous features
            max_seq_len: Maximum sequence length (120 for full T20 innings)
            fit_scaler: Whether to fit the scaler on this data
        """
        from sklearn.preprocessing import StandardScaler

        self.max_seq_len = max_seq_len
        self.feature_cols = feature_cols
        self.categorical_cols = categorical_cols
        self.continuous_cols = [c for c in feature_cols if c not in categorical_cols]

        # Prepare target (remap classes) - same as PyTorch
        df = df.copy()
        df['target'] = df['ball_outcome'].copy()
        df.loc[df['target'] == -1, 'target'] = 7  # Wicket

        # Class remapping: {0: 0, 1: 1, 2: 2, 4: 3, 6: 4, 7: 5}
        class_mapping = {0: 0, 1: 1, 2: 2, 4: 3, 6: 4, 7: 5}
        df['target'] = df['target'].map(class_mapping)
        df = df[df['target'].notna()].copy()

        # Sort by innings and ball index
        df = df.sort_values(['innings_id', 'ball_idx']).reset_index(drop=True)

        # Fit or use scaler
        self.scaler = scaler
        if fit_scaler and scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(df[self.continuous_cols].fillna(0).values)

        # Store data
        self.df = df
        self.targets = df['target'].values.astype(np.int32)

        # Build innings index
        self.innings_groups = df.groupby('innings_id').indices
        self.innings_ids = list(self.innings_groups.keys())

        # Create samples: (innings_id, position, global_idx, seq_length)
        self.samples = []
        for innings_id in self.innings_ids:
            indices = self.innings_groups[innings_id]
            for pos in range(len(indices)):
                seq_len = min(pos + 1, max_seq_len)
                self.samples.append((innings_id, pos, indices[pos], seq_len))

        print(f"  MLX Dataset: {len(self.samples)} samples from {len(self.innings_ids)} innings")

    def __len__(self) -> int:
        return len(self.samples)

    def _get_sample(self, idx: int) -> Dict[str, np.ndarray]:
        """Get a single sample as numpy arrays."""
        innings_id, pos, global_idx, seq_len = self.samples[idx]
        innings_indices = self.innings_groups[innings_id]

        # Get balls from start of innings to current position
        start_pos = max(0, pos - self.max_seq_len + 1)
        window_indices = innings_indices[start_pos:pos + 1]
        window_df = self.df.iloc[window_indices]
        actual_seq_len = len(window_df)

        # Prepare continuous features
        continuous = window_df[self.continuous_cols].fillna(0).values.astype(np.float32)
        if self.scaler is not None:
            continuous = self.scaler.transform(continuous)

        # Prepare categorical features
        categorical = {}
        for col in self.categorical_cols:
            categorical[col] = window_df[col].fillna(0).values.astype(np.int32)

        return {
            'continuous': continuous,
            'batter_encoded': categorical.get('batter_encoded', np.zeros(actual_seq_len, dtype=np.int32)),
            'bowler_encoded': categorical.get('bowler_encoded', np.zeros(actual_seq_len, dtype=np.int32)),
            'venue_encoded': categorical.get('venue_encoded', np.zeros(actual_seq_len, dtype=np.int32)),
            'matchup_type_encoded': categorical.get('matchup_type_encoded', np.zeros(actual_seq_len, dtype=np.int32)),
            'target': self.targets[global_idx],
            'seq_len': actual_seq_len
        }

    def iterate_batches(self, batch_size: int, shuffle: bool = True):
        """
        Iterate over dataset in batches.

        Yields MLX arrays with left-padding for variable-length sequences.
        """
        indices = np.arange(len(self.samples))
        if shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, len(indices), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]

            # Gather samples
            samples = [self._get_sample(i) for i in batch_indices]

            # Find max sequence length in batch
            max_len = max(s['seq_len'] for s in samples)
            batch_size_actual = len(samples)
            n_continuous = samples[0]['continuous'].shape[1]

            # Initialize padded arrays (left-padding)
            continuous = np.zeros((batch_size_actual, max_len, n_continuous), dtype=np.float32)
            batter = np.zeros((batch_size_actual, max_len), dtype=np.int32)
            bowler = np.zeros((batch_size_actual, max_len), dtype=np.int32)
            venue = np.zeros((batch_size_actual, max_len), dtype=np.int32)
            matchup = np.zeros((batch_size_actual, max_len), dtype=np.int32)
            targets = np.zeros(batch_size_actual, dtype=np.int32)

            for i, sample in enumerate(samples):
                seq_len = sample['seq_len']
                start = max_len - seq_len  # Left-pad
                continuous[i, start:] = sample['continuous']
                batter[i, start:] = sample['batter_encoded']
                bowler[i, start:] = sample['bowler_encoded']
                venue[i, start:] = sample['venue_encoded']
                matchup[i, start:] = sample['matchup_type_encoded']
                targets[i] = sample['target']

            # Convert to MLX arrays (very fast - unified memory)
            yield {
                'continuous': mx.array(continuous),
                'batter_encoded': mx.array(batter),
                'bowler_encoded': mx.array(bowler),
                'venue_encoded': mx.array(venue),
                'matchup_type_encoded': mx.array(matchup),
                'target': mx.array(targets)
            }


# ============================================================================
# MLX Training Utilities
# ============================================================================

class WarmupCosineSchedulerMLX:
    """
    Learning rate scheduler with linear warmup and cosine annealing.
    MLX version - returns learning rate values for optimizer.
    """

    def __init__(self, base_lr: float, warmup_epochs: int,
                 total_epochs: int, min_lr: float = 1e-6):
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr

    def get_lr(self, epoch: int) -> float:
        """Get learning rate for given epoch."""
        if epoch < self.warmup_epochs:
            # Linear warmup
            return self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))


def train_epoch_mlx(model: TransformerBallPredictorMLX,
                    dataset: CricketDatasetMLX,
                    optimizer: optim.Optimizer,
                    class_weights: mx.array,
                    batch_size: int,
                    gamma: float = 2.0,
                    label_smoothing: float = 0.1) -> Tuple[float, float]:
    """
    Train for one epoch using MLX.

    Key MLX differences from PyTorch:
    - No .to(device) calls - unified memory
    - mx.eval() forces computation (lazy evaluation)
    - Gradient computation via nn.value_and_grad()
    """

    def loss_fn(model, batch):
        logits = model(
            batch['continuous'],
            batch['batter_encoded'],
            batch['bowler_encoded'],
            batch['venue_encoded'],
            batch['matchup_type_encoded']
        )
        return focal_loss_mlx(logits, batch['target'], class_weights, gamma, label_smoothing)

    # Create loss and gradient function
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    n_batches = 0

    for batch in tqdm(dataset.iterate_batches(batch_size, shuffle=True),
                      desc='Training', leave=False):
        # Forward + backward in one call
        loss, grads = loss_and_grad_fn(model, batch)

        # Update weights
        optimizer.update(model, grads)

        # Force computation (MLX is lazy)
        mx.eval(model.parameters(), optimizer.state, loss)

        # Track metrics
        total_loss += float(loss) * batch['target'].shape[0]

        # Compute accuracy
        logits = model(
            batch['continuous'],
            batch['batter_encoded'],
            batch['bowler_encoded'],
            batch['venue_encoded'],
            batch['matchup_type_encoded']
        )
        preds = mx.argmax(logits, axis=-1)
        correct = mx.sum(preds == batch['target'])
        mx.eval(correct)
        total_correct += int(correct)
        total_samples += batch['target'].shape[0]
        n_batches += 1

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def evaluate_mlx(model: TransformerBallPredictorMLX,
                 dataset: CricketDatasetMLX,
                 class_weights: mx.array,
                 batch_size: int,
                 gamma: float = 2.0,
                 label_smoothing: float = 0.1) -> Tuple[float, float, float, List, List]:
    """Evaluate model on dataset."""
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_targets = []

    for batch in tqdm(dataset.iterate_batches(batch_size, shuffle=False),
                      desc='Evaluating', leave=False):
        logits = model(
            batch['continuous'],
            batch['batter_encoded'],
            batch['bowler_encoded'],
            batch['venue_encoded'],
            batch['matchup_type_encoded']
        )

        loss = focal_loss_mlx(logits, batch['target'], class_weights, gamma, label_smoothing)
        probs = mx.softmax(logits, axis=-1)
        preds = mx.argmax(logits, axis=-1)

        # Force computation
        mx.eval(loss, probs, preds)

        total_loss += float(loss) * batch['target'].shape[0]
        all_preds.extend(np.array(preds).tolist())
        all_probs.extend(np.array(probs).tolist())
        all_targets.extend(np.array(batch['target']).tolist())

    avg_loss = total_loss / len(all_targets)
    accuracy = sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_targets)

    # Compute log loss
    from sklearn.metrics import log_loss as sklearn_log_loss
    logloss = sklearn_log_loss(all_targets, all_probs, labels=list(range(6)))

    return avg_loss, accuracy, logloss, all_preds, all_targets


# ============================================================================
# Weight Conversion Utilities
# ============================================================================

def convert_pytorch_to_mlx(pt_state_dict: Dict, model: TransformerBallPredictorMLX) -> None:
    """
    Load PyTorch weights into MLX model.

    Handles naming differences between PyTorch and MLX parameter names.
    """
    # Map PyTorch parameter names to MLX parameter names
    # PyTorch: model.transformer.layers.0.self_attn.in_proj_weight
    # MLX: transformer_layers.0.self_attn.query_proj.weight

    mlx_params = dict(model.named_parameters())

    for pt_name, pt_tensor in pt_state_dict.items():
        # Convert to numpy
        np_array = pt_tensor.numpy()

        # Map PyTorch name to MLX name
        mlx_name = _map_pytorch_to_mlx_name(pt_name)

        if mlx_name in mlx_params:
            # Check shapes match
            if mlx_params[mlx_name].shape == np_array.shape:
                mlx_params[mlx_name] = mx.array(np_array)
            else:
                print(f"Warning: Shape mismatch for {mlx_name}: "
                      f"MLX={mlx_params[mlx_name].shape}, PT={np_array.shape}")
        else:
            # Handle transformer layer parameters specially
            if 'transformer.layers' in pt_name:
                # Extract layer number and remap
                mlx_name = _remap_transformer_layer(pt_name)
                if mlx_name and mlx_name in mlx_params:
                    mlx_params[mlx_name] = mx.array(np_array)

    model.update(mlx_params)


def _map_pytorch_to_mlx_name(pt_name: str) -> str:
    """Map PyTorch parameter name to MLX equivalent."""
    # Direct mappings
    mappings = {
        'batter_embed.weight': 'batter_embed.weight',
        'bowler_embed.weight': 'bowler_embed.weight',
        'venue_embed.weight': 'venue_embed.weight',
        'matchup_embed.weight': 'matchup_embed.weight',
        'input_proj.weight': 'input_proj.weight',
        'input_proj.bias': 'input_proj.bias',
        'input_ln.weight': 'input_ln.weight',
        'input_ln.bias': 'input_ln.bias',
        'pos_embed.weight': 'pos_embed.weight',
        'final_ln.weight': 'final_ln.weight',
        'final_ln.bias': 'final_ln.bias',
    }

    # Classifier mappings
    for layer in ['fc1', 'fc2', 'fc3', 'ln1']:
        for param in ['weight', 'bias']:
            mappings[f'classifier.{layer}.{param}'] = f'{layer}.{param}'
            mappings[f'{layer}.{param}'] = f'{layer}.{param}'

    return mappings.get(pt_name, pt_name)


def _remap_transformer_layer(pt_name: str) -> Optional[str]:
    """Remap transformer layer parameter names."""
    import re

    # Pattern: transformer.layers.{idx}.{component}
    match = re.match(r'transformer\.layers\.(\d+)\.(.+)', pt_name)
    if not match:
        return None

    layer_idx = match.group(1)
    component = match.group(2)

    # Map component names
    component_map = {
        'self_attn.in_proj_weight': 'self_attn.query_proj.weight',  # Approximate
        'self_attn.in_proj_bias': 'self_attn.query_proj.bias',
        'self_attn.out_proj.weight': 'self_attn.out_proj.weight',
        'self_attn.out_proj.bias': 'self_attn.out_proj.bias',
        'norm1.weight': 'ln1.weight',
        'norm1.bias': 'ln1.bias',
        'norm2.weight': 'ln2.weight',
        'norm2.bias': 'ln2.bias',
        'linear1.weight': 'linear1.weight',
        'linear1.bias': 'linear1.bias',
        'linear2.weight': 'linear2.weight',
        'linear2.bias': 'linear2.bias',
    }

    mlx_component = component_map.get(component, component)
    return f'transformer_layers.{layer_idx}.{mlx_component}'


def save_mlx_weights(model: TransformerBallPredictorMLX, path: str) -> None:
    """Save MLX model weights using MLX's native save_weights method."""
    import mlx.core as mx
    import mlx.nn as nn

    # Use MLX's native save method which handles the structure properly
    # Convert safetensors path to npz if needed
    if path.endswith('.safetensors'):
        npz_path = path.replace('.safetensors', '.npz')
    else:
        npz_path = path

    # Save using flat_params and savez_compressed
    flat_params = dict(tree_flatten(model.parameters()))
    arrays = {k: np.array(v) for k, v in flat_params.items()}
    np.savez_compressed(npz_path, **arrays)
    print(f"Saved MLX weights to {npz_path}")


def tree_flatten(tree, prefix=""):
    """Flatten a nested dict/list into key-value pairs with dotted keys."""
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from tree_flatten(v, f"{prefix}{k}." if prefix else f"{k}.")
    elif isinstance(tree, list):
        for i, v in enumerate(tree):
            yield from tree_flatten(v, f"{prefix}{i}.")
    else:
        # It's a leaf (mx.array)
        yield prefix.rstrip('.'), tree


def load_mlx_weights(model: TransformerBallPredictorMLX, path: str) -> None:
    """Load MLX model weights from safetensors or npz format."""
    import mlx.core as mx

    def convert_dicts_with_int_keys_to_lists(obj):
        """Recursively convert dicts with consecutive int keys (0,1,2...) to lists."""
        if isinstance(obj, dict):
            # First, recursively process all values
            processed = {k: convert_dicts_with_int_keys_to_lists(v) for k, v in obj.items()}

            # Check if all keys are consecutive integers starting from 0
            try:
                int_keys = sorted([int(k) for k in processed.keys()])
                if int_keys == list(range(len(int_keys))):
                    # All keys are consecutive ints, convert to list
                    return [processed[str(i)] for i in int_keys]
            except (ValueError, TypeError):
                pass

            return processed
        elif isinstance(obj, list):
            return [convert_dicts_with_int_keys_to_lists(item) for item in obj]
        else:
            return obj

    def unflatten_params(flat_weights):
        """Convert flat dotted keys back to nested dict structure."""
        nested = {}
        for key, value in flat_weights.items():
            parts = key.split('.')
            current = nested
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            # Set final value
            current[parts[-1]] = mx.array(value)

        # Convert dicts with integer keys to lists (for transformer_layers)
        return convert_dicts_with_int_keys_to_lists(nested)

    # Load weights from file
    flat_weights = {}
    if path.endswith('.safetensors'):
        try:
            from safetensors import safe_open
            with safe_open(path, framework="numpy") as f:
                for key in f.keys():
                    flat_weights[key] = f.get_tensor(key)
        except ImportError:
            raise ImportError("safetensors package required. Install with: pip install safetensors")
    elif path.endswith('.npz'):
        data = np.load(path)
        flat_weights = {key: data[key] for key in data.files}
    else:
        # Try safetensors first, then npz
        safetensors_path = path if path.endswith('.safetensors') else path + '.safetensors'
        npz_path = path.replace('.safetensors', '.npz') if path.endswith('.safetensors') else path + '.npz'

        if Path(safetensors_path).exists():
            return load_mlx_weights(model, safetensors_path)
        elif Path(npz_path).exists():
            return load_mlx_weights(model, npz_path)
        else:
            raise FileNotFoundError(f"No weights found at {path}")

    # Convert to nested structure and update model
    nested_weights = unflatten_params(flat_weights)
    model.update(nested_weights)
    print(f"Loaded MLX weights from {path}")


# ============================================================================
# Convenience Functions
# ============================================================================

def create_mlx_model_from_config(config_path: str) -> TransformerBallPredictorMLX:
    """Create MLX model from config file."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Remove non-model keys
    model_config = {k: v for k, v in config.items()
                   if k not in ['continuous_cols']}

    return TransformerBallPredictorMLX(**model_config)


def count_parameters(model: nn.Module) -> int:
    """Count total parameters in MLX model."""
    import mlx.core as mx

    def count_array(arr):
        if isinstance(arr, mx.array):
            return arr.size
        return 0

    def count_recursive(params):
        total = 0
        if isinstance(params, dict):
            for v in params.values():
                total += count_recursive(v)
        elif isinstance(params, list):
            for v in params:
                total += count_recursive(v)
        elif isinstance(params, mx.array):
            total += params.size
        return total

    return count_recursive(model.parameters())
