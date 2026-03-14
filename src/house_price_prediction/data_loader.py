import pandas as pd
import os

def load_data(data_dir="data/raw"):
    """Loads train and test datasets."""
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Data files not found in {data_dir}. Please ensure train.csv and test.csv exist.")
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    return train_df, test_df

def load_sample_submission(data_dir="data/raw"):
    """Loads sample submission file."""
    path = os.path.join(data_dir, "sample_submission.csv")
    return pd.read_csv(path)
