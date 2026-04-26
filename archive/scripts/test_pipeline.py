import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
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
        
        # Outcome distribution
        outcome_dist = dict(Counter(y_true))
        
        return EvaluationResults(
            negative_log_loss=neg_log_loss,
            average_precision_per_class=ap_per_class,
            confusion_matrix=conf_matrix,
            accuracy=accuracy,
            total_balls=len(predictions),
            skipped_balls=skipped,
            outcome_distribution=outcome_dist
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
    print(f"\n{'='*50}")
    print(f"{'EVALUATION RESULTS':^50}")
    print(f"{'='*50}")
    
    print(f"\nTotal balls evaluated: {results.total_balls:,}")
    print(f"Skipped balls: {results.skipped_balls}")
    
    print(f"\n--- Overall Metrics ---")
    print(f"Negative Log Loss: {results.negative_log_loss:.4f}")
    print(f"Accuracy: {results.accuracy:.4f} ({results.accuracy*100:.2f}%)")
    
    print(f"\n--- Average Precision per Outcome ---")
    for outcome, ap in sorted(results.average_precision_per_class.items()):
        print(f"  {outcome:8s}: {ap:.4f}")
    
    print(f"\n--- Outcome Distribution ---")
    total_outcomes = sum(results.outcome_distribution.values())
    for outcome, count in sorted(results.outcome_distribution.items()):
        pct = (count / total_outcomes) * 100
        outcome_name = {0: 'dot', 1: '1_run', 2: '2_runs', 3: '3_runs',
                       4: '4_runs', 5: '5_runs', 6: '6_runs', 7: 'wicket'}.get(outcome, str(outcome))
        print(f"  {outcome_name:8s}: {count:6d} ({pct:5.2f}%)")
    
    print(f"\n--- Confusion Matrix ---")
    print("Predicted →")
    print("Actual ↓")
    print(results.confusion_matrix)