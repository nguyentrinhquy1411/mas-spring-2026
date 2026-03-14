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
    
    # Impute numeric values
    num_imputer = SimpleImputer(strategy='mean')
    train_df[num_cols] = num_imputer.fit_transform(train_df[num_cols])
    test_df[num_cols] = num_imputer.transform(test_df[num_cols])
    
    # Scale numeric values
    scaler = MinMaxScaler()
    train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
    test_df[num_cols] = scaler.transform(test_df[num_cols])
    
    # Impute categorical values (specifics from notebook)
    columns_none = [
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'GarageType', 'GarageFinish', 'GarageQual', 'FireplaceQu', 'GarageCond',
        'MasVnrType', 'Electrical', 'MSZoning', 'Utilities', 'Functional',
        'Exterior2nd', 'KitchenQual', 'Exterior1st', 'SaleType'
    ]
    
    # Ensure all columns in columns_none exist in the dataframe before filling
    existing_cols_none = [c for c in columns_none if c in train_df.columns]
    train_df[existing_cols_none] = train_df[existing_cols_none].fillna('none')
    test_df[existing_cols_none] = test_df[existing_cols_none].fillna('none')
    
    # General categorical imputation for any remaining
    cat_imputer = SimpleImputer(strategy='constant', fill_value='none')
    train_df[cat_cols] = cat_imputer.fit_transform(train_df[cat_cols])
    test_df[cat_cols] = cat_imputer.transform(test_df[cat_cols])
    
    # One-hot encoding
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(train_df[cat_cols])
    
    encoded_cols = list(encoder.get_feature_names_out(cat_cols))
    
    train_encoded = pd.DataFrame(encoder.transform(train_df[cat_cols]), columns=encoded_cols, index=train_df.index)
    test_encoded = pd.DataFrame(encoder.transform(test_df[cat_cols]), columns=encoded_cols, index=test_df.index)
    
    # Combine processed data
    train_final = pd.concat([train_df[num_cols], train_encoded], axis=1)
    test_final = pd.concat([test_df[num_cols], test_encoded], axis=1)
    
    return train_final, test_final, y

def create_inference_df(data_dict, feature_names):
    """
    Creates a full feature dataframe from a dictionary of key inputs.
    Uses defaults for missing features.
    """
    # Create an empty df with correct columns
    inf_df = pd.DataFrame(columns=feature_names)
    inf_df.loc[0] = 0.0 # Default to 0 for all
    
    # Fill in provided values
    for key, value in data_dict.items():
        if key in feature_names:
            inf_df.at[0, key] = float(value)
            
    return inf_df
