import pandas as pd
import numpy as np

def check_targets():
    print("Loading train data...")
    df = pd.read_parquet('data/xgb_data/cricket_data_v2_train.parquet')
    
    print(f"Shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    
    if 'ball_outcome' in df.columns:
        print("\nball_outcome value counts:")
        print(df['ball_outcome'].value_counts().sort_index())
        
        # Simulate the preprocessing in xgboost_v2.py
        df['target'] = df['ball_outcome'].copy()
        df.loc[df['target'] == -1, 'target'] = 7
        df = df[df['target'] <= 7]
        
        print("\nAfter mapping -1 to 7:")
        print(df['target'].value_counts().sort_index())
        
        class_mapping = {0: 0, 1: 1, 2: 2, 4: 3, 6: 4, 7: 5}
        df['target'] = df['target'].map(class_mapping)
        
        print("\nAfter final mapping:")
        print(df['target'].value_counts().sort_index())
        
        print("\nUnique values:", df['target'].unique())
        print("NaN count:", df['target'].isnull().sum())
    else:
        print("ball_outcome column missing!")

if __name__ == "__main__":
    check_targets()
