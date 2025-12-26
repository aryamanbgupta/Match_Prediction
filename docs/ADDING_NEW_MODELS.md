# Adding a New Prediction Model

This document provides a comprehensive guide for adding a new prediction model to the Match_Prediction system.

## Overview

The prediction system uses a **plugin architecture** where models implement the `PredictionModel` abstract base class. Adding a new model requires:

1. Creating a **training script** in `scripts/`
2. Creating a **model class** in `scripts/sim_v1_2.py`
3. Updating the **evaluation entry point** in `scripts/sim_eval/run_sim_eval.py`
4. Optional: modifying **feature engineering** if new features are needed

---

## 1. Required Interface

All models must implement this interface (defined in `scripts/sim_v1_2.py`):

```python
class PredictionModel(ABC):
    @abstractmethod
    def predict_next_ball(self, features) -> Dict[str, float]:
        """Returns probability distribution over outcomes"""
        pass
    
    @abstractmethod
    def extract_features(self, state: MatchState) -> Any:
        """Extract features from match state for prediction"""
        pass
```

### Required Outcomes

Your model must return probabilities for these 8 outcomes:

| Outcome | Description |
|---------|-------------|
| `dot` | No runs scored |
| `one` | 1 run |
| `two` | 2 runs |
| `four` | Boundary (4 runs) |
| `six` | Maximum (6 runs) |
| `wicket` | Batsman dismissed |
| `wide` | Wide delivery |
| `no_ball` | No ball |

---

## 2. Training Script

**Location**: `scripts/{model_name}_v1.py`

### Required functionality:

```python
# 1. Load training data
train_df = pd.read_parquet('data/xgb_data_v3/cricket_data_v3_train.parquet')
val_df = pd.read_parquet('data/xgb_data_v3/cricket_data_v3_validation.parquet')
test_df = pd.read_parquet('data/xgb_data_v3/cricket_data_v3_test.parquet')

# 2. Use 6-class remapping
class_mapping = {0: 0, 1: 1, 2: 2, 4: 3, 6: 4, 7: 5}
# 0=dot, 1=one, 2=two, 3=four, 4=six, 5=wicket

# 3. Train your model
model = YourModel(...)
model.fit(X_train, y_train)

# 4. Save artifacts to models/{model_name}_v1/
Path('models/{model_name}_v1').mkdir(exist_ok=True)
```

### Required artifacts:
- Model weights/checkpoint
- `batter_encoder_v1.pkl` - LabelEncoder for batter IDs
- `bowler_encoder_v1.pkl` - LabelEncoder for bowler IDs
- `feature_columns_v1.txt` - List of feature names
- `feature_scaler_v1.pkl` - StandardScaler (if applicable)
- `config.json` - Architecture config (optional)

---

## 3. Model Class Template

Add to `scripts/sim_v1_2.py` after the `LSTMModelV1` class:

```python
class YourModelV1(PredictionModel):
    """Your model description"""
    
    def __init__(self,
                 model_path: str,
                 batter_encoder_path: str,
                 bowler_encoder_path: str,
                 feature_columns_path: str,
                 scaler_path: str = None,
                 stats_provider=None,
                 player_metadata=None):
        import joblib
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Load encoders
        self.batter_encoder = joblib.load(batter_encoder_path)
        self.bowler_encoder = joblib.load(bowler_encoder_path)
        self.scaler = joblib.load(scaler_path) if scaler_path else None
        
        # Load feature columns
        with open(feature_columns_path, 'r') as f:
            self.feature_columns = [line.strip() for line in f.readlines()]
        
        self.stats_provider = stats_provider
        self.player_metadata = player_metadata
        
        # Class mapping (6 classes)
        self.class_to_outcome = {
            0: 'dot', 1: 'one', 2: 'two', 3: 'four', 4: 'six', 5: 'wicket'
        }
    
    def extract_features(self, state: MatchState) -> Dict:
        """Copy feature extraction from XGBoostModelV2"""
        # See XGBoostModelV2.extract_features() for reference
        pass
    
    def predict_next_ball(self, features) -> Dict[str, float]:
        """Predict probabilities for next ball"""
        outcome_probs = {
            'dot': 0.0, 'one': 0.0, 'two': 0.0, 'four': 0.0,
            'six': 0.0, 'wicket': 0.0, 'wide': 0.01, 'no_ball': 0.01
        }
        
        # Your inference logic here
        
        # Normalize
        total = sum(outcome_probs.values())
        return {k: v/total for k, v in outcome_probs.items()}
```

---

## 4. Evaluation Integration

**File**: `scripts/sim_eval/run_sim_eval.py`

### Add to imports:
```python
from sim_v1_2 import (..., YourModelV1)
```

### Add to CLI arguments:
```python
parser.add_argument('--model-type', choices=['xgboost', 'lstm', 'yourmodel'], ...)
```

### Add loading logic:
```python
elif args.model_type == 'yourmodel':
    model = YourModelV1(
        model_path='models/yourmodel_v1/model.pkl',
        batter_encoder_path='models/yourmodel_v1/batter_encoder_v1.pkl',
        bowler_encoder_path='models/yourmodel_v1/bowler_encoder_v1.pkl',
        feature_columns_path='models/yourmodel_v1/feature_columns_v1.txt',
        stats_provider=stats_provider,
        player_metadata=player_metadata
    )
```

---

## 5. Directory Structure

```
models/{model_name}_v1/
├── {model_name}_model_v1.{ext}     # Model weights
├── batter_encoder_v1.pkl           # Batter ID encoder
├── bowler_encoder_v1.pkl           # Bowler ID encoder
├── feature_scaler_v1.pkl           # Feature scaler (optional)
├── feature_columns_v1.txt          # Feature names
└── config.json                     # Architecture config (optional)
```

---

## 6. Integration Checklist

- [ ] Training script saves all required artifacts
- [ ] Model class implements `PredictionModel` interface
- [ ] `extract_features()` matches training feature extraction
- [ ] `predict_next_ball()` returns all 8 outcomes
- [ ] Probabilities are normalized to sum to 1.0
- [ ] Added to `run_sim_eval.py` imports and CLI

---

## 7. Testing

```bash
# Train model
python scripts/{model_name}_v1.py

# Quick test (5 matches, 100 sims)
python scripts/sim_eval/run_sim_eval.py \
    --model-type yourmodel \
    --test-dir data/betting_test \
    --odds betting_odds_v3.json \
    --max-matches 5 --n-sims 100

# Full evaluation
python scripts/sim_eval/run_sim_eval.py \
    --model-type yourmodel \
    --test-dir data/betting_test \
    --odds betting_odds_v3.json \
    --n-sims 1000
```

---

## Reference Implementations

- **XGBoost**: `scripts/xgboost_v2.py`, `sim_v1_2.py::XGBoostModelV2`
- **LSTM**: `scripts/lstm_v1.py`, `sim_v1_2.py::LSTMModelV1`
