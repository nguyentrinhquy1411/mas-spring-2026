import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

def preprocess_data(train_df, test_df):
    """
    Cleans and preprocesses the datasets.
    """
    # Separate target variable
    y = train_df['SalePrice']
    
    # Drop features with > 50% missing values and IDs
    drop_cols = ['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature']
    train_df = train_df.drop(drop_cols + ['SalePrice'], axis=1)
    test_df = test_df.drop(drop_cols, axis=1)
    
    # Identify column types
    num_cols = [col for col in train_df.columns if train_df[col].dtype in ['float64', 'int64']]
    cat_cols = [col for col in train_df.columns if train_df[col].dtype not in ['float64', 'int64']]
    
    # Preprocessors
    num_imputer = SimpleImputer(strategy='mean')
    scaler = MinMaxScaler()
    cat_imputer = SimpleImputer(strategy='constant', fill_value='none')
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    # Fit and transform
    train_df[num_cols] = num_imputer.fit_transform(train_df[num_cols])
    test_df[num_cols] = num_imputer.transform(test_df[num_cols])
    
    train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
    test_df[num_cols] = scaler.transform(test_df[num_cols])
    
    # Categorical Imputation
    train_df[cat_cols] = cat_imputer.fit_transform(train_df[cat_cols])
    test_df[cat_cols] = cat_imputer.transform(test_df[cat_cols])
    
    # One-hot encoding
    encoder.fit(train_df[cat_cols])
    encoded_cols = list(encoder.get_feature_names_out(cat_cols))
    
    train_encoded = pd.DataFrame(encoder.transform(train_df[cat_cols]), columns=encoded_cols, index=train_df.index)
    test_encoded = pd.DataFrame(encoder.transform(test_df[cat_cols]), columns=encoded_cols, index=test_df.index)
    
    # Combine processed data
    train_final = pd.concat([train_df[num_cols], train_encoded], axis=1)
    test_final = pd.concat([test_df[num_cols], test_encoded], axis=1)
    
    transformers = {
        'num_imputer': num_imputer,
        'scaler': scaler,
        'cat_imputer': cat_imputer,
        'encoder': encoder,
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'feature_names': train_final.columns.tolist()
    }
    
    return train_final, test_final, y, transformers

def transform_inference_data(data_dict, transformers):
    """
    Transforms raw input dictionary for inference using fitted transformers.
    """
    # Create empty dataframe with all original raw columns
    raw_df = pd.DataFrame([data_dict])
    
    # We need to handle the case where the user only provides a subset of features
    # Fill missing columns with reasonable defaults (training means/modes)
    # For simplicity, we'll assume the web app provides the key ones
    # and we use the transformers fitted on full training data
    
    # 1. Prepare numerical part
    num_cols = transformers['num_cols']
    num_data = pd.DataFrame(index=[0], columns=num_cols)
    for col in num_cols:
        val = data_dict.get(col, 0) # Fallback to 0 if not provided
        num_data.at[0, col] = float(val)
        
    # Impute and Scale
    num_data = transformers['num_imputer'].transform(num_data)
    num_data = transformers['scaler'].transform(num_data)
    num_df = pd.DataFrame(num_data, columns=num_cols)
    
    # 2. Prepare categorical part
    cat_cols = transformers['cat_cols']
    cat_data = pd.DataFrame(index=[0], columns=cat_cols)
    for col in cat_cols:
        cat_data.at[0, col] = data_dict.get(col, 'none')
        
    # Impute and Encode
    cat_data = transformers['cat_imputer'].transform(cat_data)
    cat_encoded = transformers['encoder'].transform(cat_data)
    cat_df = pd.DataFrame(cat_encoded, columns=transformers['encoder'].get_feature_names_out(cat_cols))
    
    # Combine
    final_df = pd.concat([num_df, cat_df], axis=1)
    return final_df[transformers['feature_names']]

