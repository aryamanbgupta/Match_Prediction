import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import joblib
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from collections import Counter


@dataclass
class EvaluationResults:
    """Container for evaluation metrics"""
    negative_log_loss: float
    average_precision_per_class: Dict[str, float]
    confusion_matrix: np.ndarray
    accuracy: float
    total_balls: int
    skipped_balls: int
    outcome_distribution: Dict[int, int]
    # Additional metrics
    lift_at_k_per_class: Dict[str, float]
    recall_at_k_per_class: Dict[str, float]
    multiclass_brier_score: float
    brier_skill_score: float
    binary_metrics: Dict[str, float]
    expected_calibration_error: float
    calibration_per_class: Dict[str, float]
    reliability_plot_data: Dict[str, Any]
    

class ModelEvaluatorV2:
    """Evaluation class for v2 cricket ball prediction models with comprehensive features"""
    
    def __init__(self, model_path: str, batter_encoder_path: str, bowler_encoder_path: str, feature_columns_path: str):
        """Initialize with trained model and encoders"""
        self.model = joblib.load(model_path)
        self.batter_encoder = joblib.load(batter_encoder_path)
        self.bowler_encoder = joblib.load(bowler_encoder_path)
        
        # Load feature columns to ensure consistency
        with open(feature_columns_path, 'r') as f:
            self.feature_columns = [line.strip() for line in f.readlines()]
        
        print(f"Loaded model with {len(self.feature_columns)} features")
        
    def evaluate_from_parquet(self, test_parquet_path: str) -> EvaluationResults:
        """Run evaluation on preprocessed test data (much faster)"""
        print(f"Loading test data from {test_parquet_path}...")
        test_df = pd.read_parquet(test_parquet_path)
        
        print(f"Loaded {len(test_df)} balls from test set")
        
        # Encode players (same approach as training code)
        print("Encoding categorical variables...")
        test_df['batter_encoded'] = self.batter_encoder.transform(test_df['batter_id'].astype(str))
        test_df['bowler_encoded'] = self.bowler_encoder.transform(test_df['bowler_id'].astype(str))
        
        # Prepare features and targets
        X_test = test_df[self.feature_columns]
        
        # Convert ball_outcome to target (same mapping as training)
        y_test = test_df['ball_outcome'].copy()
        y_test = y_test.replace(-1, 7)  # Convert wickets to class 7
        
        # Apply class mapping (same as training)
        class_mapping = {0: 0, 1: 1, 2: 2, 4: 3, 6: 4, 7: 5}
        y_test = y_test.map(class_mapping)
        
        # Remove invalid outcomes
        valid_mask = y_test.notna() & (y_test <= 5)
        X_test = X_test[valid_mask]
        y_test = y_test[valid_mask]
        
        print(f"Using {len(X_test)} valid balls for evaluation")
        
        # Generate predictions
        print("Generating predictions...")
        y_probs = self.model.predict_proba(X_test)
        y_pred = self.model.predict(X_test)
        y_true = y_test.values
        
        skipped = len(test_df) - len(X_test)
        
        # Calculate all metrics
        return self._calculate_all_metrics(y_true, y_pred, y_probs, len(X_test), skipped)
    
    def _calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray, 
                             total_balls: int, skipped_balls: int) -> EvaluationResults:
        """Calculate comprehensive evaluation metrics"""
        
        # Basic metrics
        accuracy = np.mean(y_true == y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Negative log loss
        neg_log_loss = self._calculate_negative_log_loss_from_arrays(y_true, y_probs)
        
        # Average precision per class
        ap_per_class = self._calculate_average_precision_per_class(y_true, y_probs)
        
        # Lift and Recall at k
        lift_at_k, recall_at_k = self._calculate_lift_recall_at_k(y_true, y_probs, k=0.1)
        
        # Multiclass Brier score
        mc_brier, brier_skill = self._calculate_multiclass_brier(y_true, y_probs)
        
        # Binary metrics (wicket vs non-wicket)
        binary_metrics = self._calculate_binary_metrics(y_true, y_probs)
        
        # Calibration metrics
        ece, cal_per_class, reliability_data = self._calculate_calibration_metrics(y_true, y_probs)
        
        # Outcome distribution
        outcome_dist = dict(Counter(y_true))
        
        return EvaluationResults(
            negative_log_loss=neg_log_loss,
            average_precision_per_class=ap_per_class,
            confusion_matrix=conf_matrix,
            accuracy=accuracy,
            total_balls=total_balls,
            skipped_balls=skipped_balls,
            outcome_distribution=outcome_dist,
            lift_at_k_per_class=lift_at_k,
            recall_at_k_per_class=recall_at_k,
            multiclass_brier_score=mc_brier,
            brier_skill_score=brier_skill,
            binary_metrics=binary_metrics,
            expected_calibration_error=ece,
            calibration_per_class=cal_per_class,
            reliability_plot_data=reliability_data
        )
    
    def _calculate_negative_log_loss_from_arrays(self, y_true: np.ndarray, y_probs: np.ndarray) -> float:
        """Calculate negative log loss from arrays"""
        total_loss = 0.0
        for i, actual in enumerate(y_true):
            actual = int(actual)
            if actual < y_probs.shape[1]:
                p = y_probs[i, actual]
            else:
                p = 1e-15
            
            p = max(p, 1e-15)
            total_loss += -np.log(p)
        
        return total_loss / len(y_true)
    
    def _calculate_average_precision_per_class(self, y_true: np.ndarray, y_probs: np.ndarray) -> Dict[str, float]:
        """Calculate average precision for each outcome class"""
        ap_scores = {}
        unique_classes = sorted(np.unique(y_true))
        
        # Define outcome names for v2 mapping
        outcome_names = {
            0: 'dot', 1: '1_run', 2: '2_runs', 3: '4_runs', 4: '6_runs', 5: 'wicket'
        }
        
        for class_idx in unique_classes:
            class_idx = int(class_idx)
            y_true_binary = (y_true == class_idx).astype(int)
            
            if class_idx < y_probs.shape[1]:
                y_score = y_probs[:, class_idx]
                
                try:
                    ap = average_precision_score(y_true_binary, y_score)
                    class_name = outcome_names.get(class_idx, f'class_{class_idx}')
                    ap_scores[class_name] = ap
                except:
                    pass
        
        return ap_scores
    
    def _calculate_lift_recall_at_k(self, y_true: np.ndarray, y_probs: np.ndarray, k: float = 0.1) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calculate lift and recall at k for each outcome class"""
        lift_scores = {}
        recall_scores = {}
        unique_classes = sorted(np.unique(y_true))
        n_samples = len(y_true)
        k_samples = int(n_samples * k)
        
        outcome_names = {
            0: 'dot', 1: '1_run', 2: '2_runs', 3: '4_runs', 4: '6_runs', 5: 'wicket'
        }
        
        for class_idx in unique_classes:
            class_idx = int(class_idx)
            if class_idx >= y_probs.shape[1]:
                continue
                
            class_probs = y_probs[:, class_idx]
            sorted_indices = np.argsort(class_probs)[::-1]
            top_k_indices = sorted_indices[:k_samples]
            
            y_true_binary = (y_true == class_idx).astype(int)
            
            # Lift calculation
            precision_at_k = np.mean(y_true_binary[top_k_indices])
            class_frequency = np.mean(y_true_binary)
            
            if class_frequency > 0:
                lift = precision_at_k / class_frequency
            else:
                lift = 0.0
            
            # Recall calculation
            true_positives_at_k = np.sum(y_true_binary[top_k_indices])
            total_positives = np.sum(y_true_binary)
            
            if total_positives > 0:
                recall = true_positives_at_k / total_positives
            else:
                recall = 0.0
            
            class_name = outcome_names.get(class_idx, f'class_{class_idx}')
            lift_scores[class_name] = lift
            recall_scores[class_name] = recall
        
        return lift_scores, recall_scores
    
    def _calculate_multiclass_brier(self, y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[float, float]:
        """Calculate multiclass Brier score and Brier skill score"""
        n_samples = len(y_true)
        n_classes = y_probs.shape[1]
        
        # Convert y_true to one-hot encoding
        y_true_onehot = np.zeros((n_samples, n_classes))
        for i, class_idx in enumerate(y_true):
            class_idx = int(class_idx)
            if class_idx < n_classes:
                y_true_onehot[i, class_idx] = 1
        
        # Multiclass Brier score
        mc_brier = np.mean(np.sum((y_probs - y_true_onehot) ** 2, axis=1))
        
        # Reference Brier score
        class_frequencies = np.array([np.mean(y_true == i) for i in range(n_classes)])
        ref_probs = np.tile(class_frequencies, (n_samples, 1))
        ref_brier = np.mean(np.sum((ref_probs - y_true_onehot) ** 2, axis=1))
        
        # Brier skill score
        if ref_brier > 0:
            brier_skill = 1 - (mc_brier / ref_brier)
        else:
            brier_skill = 0.0
        
        return mc_brier, brier_skill
    
    def _calculate_binary_metrics(self, y_true: np.ndarray, y_probs: np.ndarray) -> Dict[str, float]:
        """Calculate binary metrics for wicket vs non-wicket"""
        # Create binary labels: 1 for wicket (class 5 in v2 mapping), 0 for others
        y_true_binary = (y_true == 5).astype(int)
        
        # Get wicket probabilities
        if 5 < y_probs.shape[1]:
            wicket_probs = y_probs[:, 5]
        else:
            return {
                'binary_log_loss': np.nan,
                'binary_brier': np.nan,
                'binary_auc': np.nan
            }
        
        # Binary log loss
        eps = 1e-15
        wicket_probs_clipped = np.clip(wicket_probs, eps, 1 - eps)
        binary_log_loss = -np.mean(
            y_true_binary * np.log(wicket_probs_clipped) + 
            (1 - y_true_binary) * np.log(1 - wicket_probs_clipped)
        )
        
        # Binary Brier score
        binary_brier = np.mean((wicket_probs - y_true_binary) ** 2)
        
        # AUC
        try:
            binary_auc = roc_auc_score(y_true_binary, wicket_probs)
        except:
            binary_auc = np.nan
        
        return {
            'binary_log_loss': binary_log_loss,
            'binary_brier': binary_brier,
            'binary_auc': binary_auc
        }
    
    def _calculate_calibration_metrics(self, y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 10) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
        """Calculate expected calibration error and reliability plot data"""
        # Overall ECE
        y_pred = np.argmax(y_probs, axis=1)
        max_probs = np.max(y_probs, axis=1)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(max_probs, bin_boundaries) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        ece = 0.0
        reliability_data = {
            'overall': {
                'bin_centers': [],
                'bin_accuracies': [],
                'bin_confidences': [],
                'bin_counts': []
            }
        }
        
        for bin_idx in range(n_bins):
            mask = bin_indices == bin_idx
            n_in_bin = np.sum(mask)
            
            if n_in_bin > 0:
                bin_accuracy = np.mean((y_pred == y_true)[mask])
                bin_confidence = np.mean(max_probs[mask])
                bin_weight = n_in_bin / len(y_true)
                
                ece += bin_weight * np.abs(bin_accuracy - bin_confidence)
                
                bin_center = (bin_boundaries[bin_idx] + bin_boundaries[bin_idx + 1]) / 2
                reliability_data['overall']['bin_centers'].append(bin_center)
                reliability_data['overall']['bin_accuracies'].append(bin_accuracy)
                reliability_data['overall']['bin_confidences'].append(bin_confidence)
                reliability_data['overall']['bin_counts'].append(n_in_bin)
        
        # Per-class calibration
        calibration_per_class = {}
        unique_classes = sorted(np.unique(y_true))
        
        outcome_names = {
            0: 'dot', 1: '1_run', 2: '2_runs', 3: '4_runs', 4: '6_runs', 5: 'wicket'
        }
        
        for class_idx in unique_classes:
            class_idx = int(class_idx)
            if class_idx >= y_probs.shape[1]:
                continue
                
            class_probs = y_probs[:, class_idx]
            y_true_binary = (y_true == class_idx).astype(int)
            
            class_ece = self._calculate_binary_ece(y_true_binary, class_probs, n_bins)
            class_name = outcome_names.get(class_idx, f'class_{class_idx}')
            calibration_per_class[class_name] = class_ece
        
        return ece, calibration_per_class, reliability_data
    
    def _calculate_binary_ece(self, y_true_binary: np.ndarray, probs: np.ndarray, n_bins: int) -> float:
        """Helper to calculate ECE for binary case"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(probs, bin_boundaries) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        ece = 0.0
        for bin_idx in range(n_bins):
            mask = bin_indices == bin_idx
            n_in_bin = np.sum(mask)
            
            if n_in_bin > 0:
                bin_accuracy = np.mean(y_true_binary[mask])
                bin_confidence = np.mean(probs[mask])
                bin_weight = n_in_bin / len(y_true_binary)
                ece += bin_weight * np.abs(bin_accuracy - bin_confidence)
        
        return ece


def print_results(results: EvaluationResults):
    """Print comprehensive evaluation results"""
    print(f"\n{'='*60}")
    print(f"{'CRICKET BALL PREDICTION - V2 MODEL EVALUATION':^60}")
    print(f"{'='*60}")
    
    print(f"\nDataset Statistics:")
    print(f"  Total balls evaluated: {results.total_balls:,}")
    print(f"  Skipped balls: {results.skipped_balls}")
    
    print(f"\n--- Core Metrics ---")
    print(f"Accuracy: {results.accuracy:.4f} ({results.accuracy*100:.2f}%)")
    print(f"Negative Log Loss: {results.negative_log_loss:.4f}")
    print(f"Multiclass Brier Score: {results.multiclass_brier_score:.4f}")
    print(f"Brier Skill Score: {results.brier_skill_score:.4f}")
    
    print(f"\n--- Binary Metrics (Wicket vs Non-Wicket) ---")
    for metric, value in results.binary_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\n--- Calibration Metrics ---")
    print(f"Expected Calibration Error (ECE): {results.expected_calibration_error:.4f}")
    print(f"Per-class calibration:")
    for outcome, ece in sorted(results.calibration_per_class.items()):
        print(f"  {outcome:8s}: {ece:.4f}")
    
    print(f"\n--- Average Precision per Outcome ---")
    for outcome, ap in sorted(results.average_precision_per_class.items()):
        print(f"  {outcome:8s}: {ap:.4f}")
    
    print(f"\n--- Lift@10% per Outcome ---")
    for outcome, lift in sorted(results.lift_at_k_per_class.items()):
        recall = results.recall_at_k_per_class.get(outcome, 0)
        print(f"  {outcome:8s}: Lift={lift:.2f}, Recall={recall:.2f}")
    
    print(f"\n--- Outcome Distribution ---")
    total_outcomes = sum(results.outcome_distribution.values())
    for outcome, count in sorted(results.outcome_distribution.items()):
        pct = (count / total_outcomes) * 100
        outcome_name = {0: 'dot', 1: '1_run', 2: '2_runs', 3: '4_runs', 4: '6_runs', 5: 'wicket'}.get(outcome, str(outcome))
        print(f"  {outcome_name:8s}: {count:6d} ({pct:5.2f}%)")
    
    print(f"\n--- Confusion Matrix ---")
    print("Rows=Actual, Cols=Predicted")
    print("Classes: 0=dot, 1=1run, 2=2runs, 3=4runs, 4=6runs, 5=wicket")
    print(results.confusion_matrix)


if __name__ == "__main__":
    # Initialize evaluator with v2 model
    evaluator = ModelEvaluatorV2(
        model_path='models/xgb/xgboost_model_v2.pkl',
        batter_encoder_path='models/xgb/batter_encoder_v2.pkl',
        bowler_encoder_path='models/xgb/bowler_encoder_v2.pkl',
        feature_columns_path='models/xgb/feature_columns_v2.txt'
    )
    
    # Run evaluation on test set
    results = evaluator.evaluate_from_parquet('data/xgb_data/cricket_data_v2_test.parquet')
    
    # Print results
    print_results(results)