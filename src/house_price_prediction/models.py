import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge, ElasticNet, SGDRegressor, BayesianRidge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
import optuna

def get_base_models():
    """Returns a dictionary of standard regression models."""
    return {
        'SVR': SVR(),
        'XGBRegressor': XGBRegressor(),
        'Ridge': Ridge(),
        'ElasticNet': ElasticNet(),
        'SGDRegressor': SGDRegressor(),
        'BayesianRidge': BayesianRidge(),
        'LinearRegression': LinearRegression(),
        'RandomForestRegressor': RandomForestRegressor()
    }

def train_and_evaluate(X, y):
    """Trains multiple models and prints their RMSE."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    models = get_base_models()
    results = []
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        results.append({'Model': name, 'RMSE': rmse})
        
    df_results = pd.DataFrame(results).sort_values(by='RMSE')
    return df_results

def tune_xgboost(X, y, n_trials=50):
    """Uses Optuna to find the best hyperparameters for XGBoost."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    def objective(trial):
        params = {
            'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
            'learning_rate': trial.suggest_categorical('learning_rate', [0.008, 0.009, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02]),
            'n_estimators': trial.suggest_categorical('n_estimators', [1000, 2000, 3000, 4000]),
            'max_depth': trial.suggest_categorical('max_depth', [5, 7, 9, 11, 13, 15, 17, 20]),
            'random_state': trial.suggest_categorical('random_state', [24, 48, 2020]),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
        }
        
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        preds = model.predict(X_test)
        return np.sqrt(mean_squared_error(y_test, preds))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def train_best_xgboost(X, y, params=None):
    """Trains XGBoost with optimized parameters."""
    if params is None:
        # Default best params from the notebook
        params = {
            'lambda': 3.559040735218393,
            'alpha': 0.25772549522868987,
            'colsample_bytree': 0.6,
            'subsample': 0.5,
            'learning_rate': 0.014,
            'n_estimators': 4000,
            'max_depth': 11,
            'random_state': 24,
            'min_child_weight': 3
        }
    
    model = XGBRegressor(**params)
    model.fit(X, y)
    return model
