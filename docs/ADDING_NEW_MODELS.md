# Adding a New Prediction Model

How to plug a new model type into CricML's training, simulation, and
evaluation pipelines. Models implement the `PredictionModel` ABC defined in
`scripts/sim_v1_2.py`.

**Lifecycle rule:** an experiment should use isolated artifacts and a
fail-closed candidate runner. Do not permanently register every experiment in
the main runtime. Registration in the single current runner is a promotion
step: the new bundle replaces the previous current bundle. Rejected
experiments retain their config/report/source commit but lose active runtime
hooks. See [REPOSITORY_CONSOLIDATION.md](REPOSITORY_CONSOLIDATION.md).

For the higher-level architecture, see [ARCHITECTURE.md](ARCHITECTURE.md).
For day-to-day commands, see [OPERATIONS.md](OPERATIONS.md).

---

## 1. Required Interface

`scripts/sim_v1_2.py`:

```python
class PredictionModel(ABC):
    @abstractmethod
    def extract_features(self, state: MatchState) -> Any:
        """Return features in whatever format predict_next_ball expects.
        Common choices: pd.DataFrame (XGBoost), np.ndarray (MLP),
        Tuple[np.ndarray, dict] (LSTM with sequence + categorical embeds)."""

    @abstractmethod
    def predict_next_ball(self, features) -> Dict[str, float]:
        """Return a probability distribution over the 8 outcomes the
        simulation engine consumes. Must sum to ~1.0."""
```

### Outcome dictionary contract

The model is trained on **6 classes** (`{0:dot, 1:one, 2:two, 3:four, 4:six,
5:wicket}`). The wrapper's `predict_next_ball` adds a small fixed probability
for `'wide'` and `'no_ball'` and re-normalizes — these aren't predicted, just
inserted for the simulation's match-flow realism.

```python
# After the model gives you P(class) for class in 0..5:
outcome_probs = {
    'dot':    p_class[0],
    'one':    p_class[1],
    'two':    p_class[2],
    'four':   p_class[3],
    'six':    p_class[4],
    'wicket': p_class[5],
    'wide':   0.01,
    'no_ball': 0.01,
}
total = sum(outcome_probs.values())
return {k: v / total for k, v in outcome_probs.items()}
```

**Critical**: never return raw 8-class outputs from a 6-class model — only
the simulation needs the 8-key dict, and the extras are post-hoc.

---

## 2. Training Script

**Location**: `scripts/{model_name}_v1.py`

Follow the pattern from `xgboost_v2.py`, `lstm_v1.py`, or `mlp_v1.py`. Key
contracts:

```python
import argparse, json, joblib, pandas as pd
from pathlib import Path

# 1. Argparse — accept --config-json from run_experiment.py
parser = argparse.ArgumentParser()
parser.add_argument('--config-json', type=str, default=None,
                    help='JSON config from experiment runner')
# ... model-specific flags ...
args = parser.parse_args()

# 2. Resolve features (config-driven if invoked via run_experiment.py;
#    fall back to V6_GROUPS defaults for standalone use)
import sys
sys.path.insert(0, 'scripts')
from feature_registry import resolve_feature_list, V6_GROUPS

if args.config_json:
    config = json.loads(args.config_json)
    feature_cols = resolve_feature_list(
        config['features']['groups'],
        config['features'].get('exclude'),
        config['features'].get('include_extra'),
    )
else:
    feature_cols = resolve_feature_list(V6_GROUPS)

# 3. Load data — v3 paths (105 columns under schema v4)
DATA_DIR = Path('data/xgb_data_v3')
train_df = pd.read_parquet(DATA_DIR / 'cricket_data_v3_train.parquet')
val_df   = pd.read_parquet(DATA_DIR / 'cricket_data_v3_validation.parquet')
test_df  = pd.read_parquet(DATA_DIR / 'cricket_data_v3_test.parquet')

# 4. Filter to features that actually exist in the parquet — feature
#    registry may include features not yet materialized
feature_cols = [c for c in feature_cols if c in train_df.columns]

# 5. 6-class remapping
CLASS_MAP = {0: 0, 1: 1, 2: 2, 4: 3, 6: 4, -1: 5}   # raw → trained
train_df = train_df[train_df['ball_outcome'].isin(CLASS_MAP)].copy()
y_train  = train_df['ball_outcome'].map(CLASS_MAP).values
# (same for val/test)

# 6. Save artifacts to models/{model_name}_v1/
out_dir = Path('models/your_model_v1')
out_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(model, out_dir / 'your_model_v1.pkl')
joblib.dump(batter_encoder, out_dir / 'batter_encoder_v1.pkl')
joblib.dump(bowler_encoder, out_dir / 'bowler_encoder_v1.pkl')
(out_dir / 'feature_columns_v1.txt').write_text('\n'.join(feature_cols))
```

### Required artifacts in `models/{model_name}_v1/`

| File | Purpose |
|---|---|
| `your_model_v1.{pkl,pt,safetensors}` | Trained weights / checkpoint |
| `batter_encoder_v1.pkl` | sklearn `LabelEncoder` for batter IDs |
| `bowler_encoder_v1.pkl` | sklearn `LabelEncoder` for bowler IDs |
| `feature_columns_v1.txt` | Newline-separated feature list (must match training) |
| `feature_scaler_v1.pkl` | sklearn `StandardScaler` (only for NN models) |
| `config.json` | Architecture / hyperparams (optional but recommended) |

---

## 3. Model Wrapper Class

Add to `scripts/sim_v1_2.py` near the existing wrappers (`XGBoostModelV2`,
`LSTMModelV1`, `MLPModelV1`, `TransformerModelV1`):

```python
class YourModelV1(PredictionModel):
    """Plug-in wrapper for the YourModel architecture."""

    def __init__(self,
                 model_path: str,
                 batter_encoder_path: str,
                 bowler_encoder_path: str,
                 feature_columns_path: str,
                 scaler_path: Optional[str] = None,
                 stats_provider=None,
                 player_metadata=None):
        import joblib

        # 1. Load model + encoders
        self.model = self._load_model(model_path)
        self.batter_encoder = joblib.load(batter_encoder_path)
        self.bowler_encoder = joblib.load(bowler_encoder_path)
        self.scaler = joblib.load(scaler_path) if scaler_path else None

        with open(feature_columns_path) as f:
            self.feature_columns = [line.strip() for line in f if line.strip()]

        # 2. Cache encoder lookups as dicts (Fix A from XGBoost speedup —
        #    LabelEncoder.transform is slow per-call)
        self._batter_to_idx = {c: i for i, c in enumerate(self.batter_encoder.classes_)}
        self._bowler_to_idx = {c: i for i, c in enumerate(self.bowler_encoder.classes_)}

        # 3. Memoize team-strength / venue lookups for the duration of
        #    each match (constant across every ball of the match)
        from stats_provider import wrap_with_cache
        self.stats_provider = wrap_with_cache(stats_provider)
        self.player_metadata = player_metadata

        # 4. 6-class output mapping
        self.class_to_outcome = {
            0: 'dot', 1: 'one', 2: 'two',
            3: 'four', 4: 'six', 5: 'wicket',
        }

    def extract_features(self, state: MatchState):
        """Build feature vector. See XGBoostModelV2.extract_features for
        the canonical implementation — copy and adapt for your input shape."""
        # ...

    def predict_next_ball(self, features) -> Dict[str, float]:
        # 1. Run inference → 6 class probabilities
        class_probs = self.model.predict(features)   # shape: (1, 6) or (6,)
        if class_probs.ndim == 2:
            class_probs = class_probs[0]

        # 2. Map to outcome dict
        outcome_probs = {
            self.class_to_outcome[i]: float(class_probs[i])
            for i in range(6)
        }

        # 3. Add fixed extras and normalize
        outcome_probs['wide'] = 0.01
        outcome_probs['no_ball'] = 0.01
        total = sum(outcome_probs.values())
        return {k: v / total for k, v in outcome_probs.items()}
```

**Why `wrap_with_cache` matters**: every model wrapper calls it in `__init__`
(idempotent). Without it, each ball re-runs an 11-player loop inside the
provider for team-strength lookups; with it, each match computes them once.
Roughly 1.2× sim speedup. The wrapper is pickle-safe for
`multiprocessing.Pool.starmap`.

---

## 4. Add an isolated evaluation entry point

Before promotion, follow the I8 pattern: make a small candidate runner that
validates the exact model, feature order, stats schema, identity contract, and
sidecars, then delegates to the shared evaluation framework. It must fail
closed and must not fall back to the demonstration model or frozen state.

Only after promotion should the model replace the current branch in
`run_sim_eval.py`. The following registration pattern describes that
promotion step:

`scripts/sim_eval/run_sim_eval.py`:

```python
# Add to imports
from sim_v1_2 import (..., YourModelV1)

# Add to --model-type choices
parser.add_argument('--model-type',
                    choices=['xgboost', 'lstm', 'mlp', 'transformer', 'yourmodel'],
                    default='xgboost')

# Add to model-loading branch in main()
elif args.model_type == 'yourmodel':
    model = YourModelV1(
        model_path='models/yourmodel_v1/yourmodel_v1.pkl',
        batter_encoder_path='models/yourmodel_v1/batter_encoder_v1.pkl',
        bowler_encoder_path='models/yourmodel_v1/bowler_encoder_v1.pkl',
        feature_columns_path='models/yourmodel_v1/feature_columns_v1.txt',
        scaler_path='models/yourmodel_v1/feature_scaler_v1.pkl',
        stats_provider=stats_provider,
        player_metadata=player_metadata,
    )
```

---

## 5. Register in `run_experiment.py`

`scripts/run_experiment.py:build_training_cmd` has a `script_map`:

```python
script_map = {
    "xgboost": "scripts/xgboost_v2.py",
    "lstm": "scripts/lstm_v1.py",
    "transformer": "scripts/transformer_v1.py",
    "mlp": "scripts/mlp_v1.py",
    "yourmodel": "scripts/yourmodel_v1.py",   # add this
}
```

If your model accepts hyperparameters via CLI flags, also extend `cli_map` in
the same function — that's how YAML `model.hyperparameters` translates to
`--epochs`, `--batch-size`, etc.

---

## 6. Create an Experiment Config

```bash
cp experiments/configs/xgb_v6_outcome_dist.yaml \
   experiments/configs/yourmodel_v1_baseline.yaml
```

Edit:
- `experiment.name` → `yourmodel_v1_baseline`
- `experiment.tags` → `["yourmodel", "v1", "baseline"]`
- `model.type` → `"yourmodel"`
- `model.hyperparameters` → your model's hyperparams

Then:
```bash
uv run python scripts/run_experiment.py \
    experiments/configs/yourmodel_v1_baseline.yaml --skip-parsing
```

---

## 7. Integration Checklist

- [ ] Training script accepts `--config-json` for experiment runner
- [ ] Training script saves all required artifacts to `models/yourmodel_v1/`
- [ ] Wrapper class implements `PredictionModel` (6-class output)
- [ ] `__init__` calls `wrap_with_cache(stats_provider)`
- [ ] `predict_next_ball` adds extras and normalizes
- [ ] `extract_features` matches training-time feature order
- [ ] Registered in `run_sim_eval.py` (imports + CLI choices + load branch)
- [ ] Registered in `run_experiment.py:build_training_cmd:script_map`
- [ ] Experiment config in `experiments/configs/`
- [ ] Smoke test: `--max-matches 5 --n-sims 10` runs end-to-end without crashing

---

## 8. Testing

```bash
# Train
uv run python scripts/yourmodel_v1.py

# Smoke test (5 matches, 10 sims, ~1 min)
uv run python scripts/sim_eval/run_sim_eval.py \
    --model-type yourmodel \
    --test-dir data/polymarket_test \
    --odds betting_odds_polymarket.json \
    --max-matches 5 --n-sims 10

# Full eval via experiment runner (n_sims comes from YAML)
uv run python scripts/run_experiment.py \
    experiments/configs/yourmodel_v1_baseline.yaml --only-eval
```

The smoke test catches the common failure modes: encoder vocabulary mismatch,
feature-column drift, output shape errors, and missing artifact files. If it
passes 5×10, scaling to 261×100 is mechanical.

---

## Reference Implementations

| Model | Training | Wrapper |
|---|---|---|
| XGBoost (production) | `scripts/xgboost_v2.py` | `sim_v1_2.py::XGBoostModelV2` |
| LSTM | `scripts/lstm_v1.py` | `sim_v1_2.py::LSTMModelV1` |
| MLP | `scripts/mlp_v1.py` | `sim_v1_2.py::MLPModelV1` / `MLPModelV2` |
| Transformer (PyTorch + MLX) | `scripts/transformer_v1.py` | `sim_v1_2.py::TransformerModelV1` |

`XGBoostModelV2` is the cleanest reference — it has the dict-cached
`LabelEncoder` lookups and the preallocated row buffer that `validate_numpy_predict.py`
guards against regression on. The PyTorch wrappers are heavier because they
manage device placement, batching, and (for `LSTMModelV1`) sequence padding.
