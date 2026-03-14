import os
import joblib
from house_price_prediction.data_loader import load_data, load_sample_submission
from house_price_prediction.preprocessing import preprocess_data
from house_price_prediction.models import train_and_evaluate, train_best_xgboost

def main():
    print("--- Loading Data ---")
    train_df, test_df = load_data()
    
    print("\n--- Preprocessing Data ---")
    X_train, X_test, y, transformers = preprocess_data(train_df, test_df)
    print(f"Features after preprocessing: {X_train.shape[1]}")
    
    print("\n--- Evaluating Base Models ---")
    results = train_and_evaluate(X_train, y)
    print(results)
    
    print("\n--- Training Best XGBoost Model ---")
    # Using parameters found in the notebook
    model = train_best_xgboost(X_train, y)
    
    print("\n--- Saving Model and Artifacts ---")
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgboost_model.joblib")
    joblib.dump(transformers, "models/transformers.joblib")
    print("Model and transformers saved to models/")
    
    print("\n--- Generating Predictions ---")
    test_preds = model.predict(X_test)
    
    print("\n--- Creating Submission File ---")
    submission_df = load_sample_submission()
    submission_df['SalePrice'] = test_preds
    
    output_path = "finalSubmission.csv"
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

if __name__ == "__main__":
    main()
