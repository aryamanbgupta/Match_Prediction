import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import joblib
import json
from sklearn.metrics import average_precision_score, confusion_matrix
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
    

class ModelEvaluator:
    """Main evaluation class for cricket ball prediction models"""
    
    def __init__(self, model_path: str, batter_encoder_path: str, bowler_encoder_path: str):
        """Initialize with trained model and encoders"""
        self.model = joblib.load(model_path)
        self.batter_encoder = joblib.load(batter_encoder_path)
        self.bowler_encoder = joblib.load(bowler_encoder_path)
        
    def extract_features_from_ball(self, ball_data: dict) -> pd.DataFrame:
        """Extract features from a single ball record
        
        This mirrors the feature extraction in XGBoostModel but takes
        parsed ball data instead of MatchState
        """
        # Direct mapping from parsed data
        inning_idx = ball_data['inning_idx']
        score = ball_data['score']
        wickets = ball_data['wickets']
        balls_bowled = ball_data['balls_bowled']
        
        # Encode players - with error handling for unknown players
        try:
            batter_encoded = self.batter_encoder.transform([ball_data['batter_id']])[0]
        except:
            batter_encoded = -1  # Unknown batter
            
        try:
            bowler_encoded = self.bowler_encoder.transform([ball_data['bowler_id']])[0]
        except:
            bowler_encoded = -1  # Unknown bowler
        
        # Derived features (same as training)
        run_rate = score / (balls_bowled + 1)  # +1 to avoid division by zero
        wickets_ratio = wickets / 10.0
        balls_ratio = balls_bowled / 120.0
        
        # Return as DataFrame with proper column names (must match training)
        feature_names = [
            'inning_idx', 'score', 'wickets', 'balls_bowled',
            'batter_encoded', 'bowler_encoded', 'run_rate', 
            'wickets_ratio', 'balls_ratio'
        ]
        
        features = pd.DataFrame([[
            inning_idx, score, wickets, balls_bowled,
            batter_encoded, bowler_encoded, run_rate,
            wickets_ratio, balls_ratio
        ]], columns=feature_names)
        
        return features
    
    def evaluate(self, test_folder: str) -> EvaluationResults:
        """Run evaluation on test matches"""
        # Load test data
        print("Loading test matches...")
        test_balls = self._load_test_data(test_folder)
        print(f"Loaded {len(test_balls)} balls from test matches")
        
        # Collect predictions
        predictions = []  # List of (probs, actual) tuples
        y_true = []      # Actual outcomes
        y_pred = []      # Predicted outcomes
        y_probs = []     # Probability arrays
        skipped = 0
        
        print("Generating predictions...")
        for i, ball in enumerate(test_balls):
            try:
                # Extract features
                features = self.extract_features_from_ball(ball)
                
                # Get prediction probabilities
                probs = self.model.predict_proba(features)[0]
                
                # Get actual outcome (convert -1 to 7 for wicket)
                actual = ball['ball_outcome']
                if actual == -1:
                    actual = 7
                
                # Store for different metrics
                predictions.append((probs, actual))
                y_true.append(actual)
                y_pred.append(np.argmax(probs))
                y_probs.append(probs)
                
                # Progress indicator
                if (i + 1) % 10000 == 0:
                    print(f"  Processed {i + 1} balls...")
                
            except Exception as e:
                # Skip problematic balls but count them
                skipped += 1
                if skipped <= 5:  # Only print first few errors
                    print(f"Skipped ball due to error: {e}")
        
        print(f"Generated predictions for {len(predictions)} balls ({skipped} skipped)")
        
        # Calculate metrics
        neg_log_loss = self._calculate_negative_log_loss(predictions)
        
        # Convert to numpy arrays for sklearn
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_probs = np.array(y_probs)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Accuracy
        accuracy = np.mean(y_true == y_pred)
        
        # Average precision per class (one-vs-rest)
        ap_per_class = self._calculate_average_precision_per_class(y_true, y_probs)
        
        # Lift and Recall at k (default k=10%)
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
            total_balls=len(predictions),
            skipped_balls=skipped,
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
    
    def _load_test_data(self, folder_path: str) -> List[Dict]:
        """Load and parse test match data
        
        Inline the parsing logic to avoid import issues
        """
        all_balls = []
        json_files = sorted(Path(folder_path).glob('*.json'))
        
        for file_path in json_files:
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)
                
                match_balls = self._parse_match_data(data)
                all_balls.extend(match_balls)
                
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")
        
        return all_balls
    
    def _parse_match_data(self, data: dict) -> List[Dict]:
        """Parse a single match JSON into ball records
        
        Simplified version of parse_gbm.py logic
        """
        player_registry = data['info']['registry']['people']
        all_balls = []
        
        for inning_idx, inning in enumerate(data['innings'], 1):
            score = 0
            wickets = 0
            balls = 0
            
            for over in inning['overs']:
                for delivery in over['deliveries']:
                    # Extract ball info
                    batter_id = player_registry[delivery['batter']]
                    bowler_id = player_registry[delivery['bowler']]
                    runs = delivery['runs']['total']
                    
                    # Check for wicket
                    is_wicket = 'wickets' in delivery
                    ball_outcome = -1 if is_wicket else runs
                    
                    # Create ball record
                    ball_record = {
                        'inning_idx': inning_idx,
                        'score': score,
                        'wickets': wickets,
                        'balls_bowled': balls,
                        'batter_id': batter_id,
                        'bowler_id': bowler_id,
                        'ball_outcome': ball_outcome
                    }
                    
                    all_balls.append(ball_record)
                    
                    # Update state
                    if is_wicket:
                        wickets += 1
                    else:
                        score += runs
                    
                    # Check for extras
                    is_wide_or_noball = False
                    if 'extras' in delivery:
                        extra_types = delivery['extras'].keys()
                        is_wide_or_noball = 'wides' in extra_types or 'noballs' in extra_types
                    
                    if not is_wide_or_noball:
                        balls += 1
        
        return all_balls
    
    def _calculate_negative_log_loss(self, predictions: List[Tuple[np.ndarray, int]]) -> float:
        """Calculate negative log loss
        
        For each ball, we look at the probability assigned to the actual outcome
        and calculate -log(p). Average across all balls.
        """
        if not predictions:
            return float('inf')
        
        total_loss = 0.0
        for probs, actual in predictions:
            # Get probability of actual outcome
            if actual < len(probs):
                p = probs[actual]
            else:
                # This shouldn't happen but let's be defensive
                p = 1e-15
            
            # Avoid log(0) by using small epsilon
            p = max(p, 1e-15)
            
            # Add to total loss
            total_loss += -np.log(p)
        
        return total_loss / len(predictions)
    
    def _calculate_average_precision_per_class(self, y_true: np.ndarray, y_probs: np.ndarray) -> Dict[str, float]:
        """Calculate average precision for each outcome class
        
        Uses one-vs-rest approach for multi-class classification
        """
        ap_scores = {}
        unique_classes = sorted(np.unique(y_true))
        
        # Define outcome names for readability
        outcome_names = {
            0: 'dot', 1: '1_run', 2: '2_runs', 3: '3_runs',
            4: '4_runs', 5: '5_runs', 6: '6_runs', 7: 'wicket'
        }
        
        for class_idx in unique_classes:
            # Create binary labels: 1 for this class, 0 for others
            y_true_binary = (y_true == class_idx).astype(int)
            
            # Get probabilities for this class
            if class_idx < y_probs.shape[1]:
                y_score = y_probs[:, class_idx]
                
                # Calculate average precision
                try:
                    ap = average_precision_score(y_true_binary, y_score)
                    class_name = outcome_names.get(class_idx, f'class_{class_idx}')
                    ap_scores[class_name] = ap
                except:
                    # Skip if calculation fails (e.g., only one class present)
                    pass
        
        return ap_scores
    
    def _calculate_lift_recall_at_k(self, y_true: np.ndarray, y_probs: np.ndarray, k: float = 0.1) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calculate lift and recall at k for each outcome class
        
        Args:
            k: Fraction of predictions to consider (default 0.1 = top 10%)
        """
        lift_scores = {}
        recall_scores = {}
        unique_classes = sorted(np.unique(y_true))
        n_samples = len(y_true)
        k_samples = int(n_samples * k)
        
        outcome_names = {
            0: 'dot', 1: '1_run', 2: '2_runs', 3: '3_runs',
            4: '4_runs', 5: '5_runs', 6: '6_runs', 7: 'wicket'
        }
        
        for class_idx in unique_classes:
            if class_idx >= y_probs.shape[1]:
                continue
                
            # Get probabilities for this class
            class_probs = y_probs[:, class_idx]
            
            # Sort by probability (descending)
            sorted_indices = np.argsort(class_probs)[::-1]
            
            # Take top k%
            top_k_indices = sorted_indices[:k_samples]
            
            # Calculate metrics
            y_true_binary = (y_true == class_idx).astype(int)
            
            # Lift: (precision at k) / (class frequency)
            precision_at_k = np.mean(y_true_binary[top_k_indices])
            class_frequency = np.mean(y_true_binary)
            
            if class_frequency > 0:
                lift = precision_at_k / class_frequency
            else:
                lift = 0.0
            
            # Recall: (true positives in top k) / (total positives)
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
        """Calculate multiclass Brier score and Brier skill score
        
        Returns:
            multiclass_brier_score: Lower is better (0 = perfect)
            brier_skill_score: Higher is better (1 = perfect, 0 = no skill)
        """
        n_samples = len(y_true)
        n_classes = y_probs.shape[1]
        
        # Convert y_true to one-hot encoding
        y_true_onehot = np.zeros((n_samples, n_classes))
        for i, class_idx in enumerate(y_true):
            if class_idx < n_classes:
                y_true_onehot[i, class_idx] = 1
        
        # Multiclass Brier score
        mc_brier = np.mean(np.sum((y_probs - y_true_onehot) ** 2, axis=1))
        
        # Reference Brier score (using marginal probabilities)
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
        """Calculate binary metrics for wicket vs non-wicket
        
        Returns dict with:
            binary_log_loss: Log loss for wicket prediction
            binary_brier: Brier score for wicket prediction
            binary_auc: AUC for wicket prediction
        """
        # Create binary labels: 1 for wicket (class 7), 0 for others
        y_true_binary = (y_true == 7).astype(int)
        
        # Get wicket probabilities
        if 7 < y_probs.shape[1]:
            wicket_probs = y_probs[:, 7]
        else:
            # Fallback if wicket class missing
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
        from sklearn.metrics import roc_auc_score
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
        """Calculate expected calibration error and reliability plot data
        
        Returns:
            expected_calibration_error: Overall ECE
            calibration_per_class: ECE for each outcome class
            reliability_plot_data: Data for plotting calibration curves
        """
        # Overall ECE (using max probability)
        y_pred = np.argmax(y_probs, axis=1)
        max_probs = np.max(y_probs, axis=1)
        
        # Bin predictions
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
                
                # Store for plotting
                bin_center = (bin_boundaries[bin_idx] + bin_boundaries[bin_idx + 1]) / 2
                reliability_data['overall']['bin_centers'].append(bin_center)
                reliability_data['overall']['bin_accuracies'].append(bin_accuracy)
                reliability_data['overall']['bin_confidences'].append(bin_confidence)
                reliability_data['overall']['bin_counts'].append(n_in_bin)
        
        # Per-class calibration
        calibration_per_class = {}
        unique_classes = sorted(np.unique(y_true))
        
        outcome_names = {
            0: 'dot', 1: '1_run', 2: '2_runs', 3: '3_runs',
            4: '4_runs', 5: '5_runs', 6: '6_runs', 7: 'wicket'
        }
        
        for class_idx in unique_classes:
            if class_idx >= y_probs.shape[1]:
                continue
                
            class_probs = y_probs[:, class_idx]
            y_true_binary = (y_true == class_idx).astype(int)
            
            # Calculate ECE for this class
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


# Example usage
if __name__ == "__main__":
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path='models/gradient_boosting_model.pkl',
        batter_encoder_path='models/batter_encoder.pkl',
        bowler_encoder_path='models/bowler_encoder.pkl'
    )
    
    # Run evaluation
    results = evaluator.evaluate('data/test/')
    
    # Print results
    print(f"\n{'='*60}")
    print(f"{'CRICKET BALL PREDICTION - EVALUATION RESULTS':^60}")
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
        outcome_name = {0: 'dot', 1: '1_run', 2: '2_runs', 3: '3_runs',
                       4: '4_runs', 5: '5_runs', 6: '6_runs', 7: 'wicket'}.get(outcome, str(outcome))
        print(f"  {outcome_name:8s}: {count:6d} ({pct:5.2f}%)")
    
    print(f"\n--- Confusion Matrix ---")
    print("Rows=Actual, Cols=Predicted")
    print("Classes: 0=dot, 1=1run, 2=2runs, 3=3runs, 4=4runs, 5=5runs, 6=6runs, 7=wicket")
    print(results.confusion_matrix)
    
    # Optional: Save reliability plot data
    print(f"\nReliability plot data available in results.reliability_plot_data")
    print("Use matplotlib to visualize calibration curves if needed.")