import pandas as pd
import os

def verify_parquet():
    files = [
        'data/xgb_data/cricket_data_v2_train.parquet',
        'data/xgb_data/cricket_data_v2_validation.parquet',
        'data/xgb_data/cricket_data_v2_test.parquet'
    ]
    
    expected_features = [
        'venue', 
        'is_toss_winner', 
        'batsman_recent_avg', 
        'batsman_recent_sr',
        'bowler_recent_avg', 
        'bowler_recent_econ'
    ]
    
    for f in files:
        if not os.path.exists(f):
            print(f"❌ File not found: {f}")
            continue
            
        print(f"Checking {f}...")
        try:
            df = pd.read_parquet(f)
            print(f"  Shape: {df.shape}")
            
            missing = [feat for feat in expected_features if feat not in df.columns]
            
            if missing:
                print(f"  ❌ Missing features: {missing}")
            else:
                print(f"  ✅ All expected features present")
                
            # Check for non-null values in new features
            for feat in expected_features:
                if feat in df.columns:
                    nulls = df[feat].isnull().sum()
                    if nulls > 0:
                        print(f"    ⚠️ {feat} has {nulls} null values")
                    else:
                        # Print sample values
                        print(f"    {feat} sample: {df[feat].iloc[0]}")
                        
        except Exception as e:
            print(f"  ❌ Error reading file: {e}")

if __name__ == "__main__":
    verify_parquet()
